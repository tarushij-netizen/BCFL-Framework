"""
Step B: do inference attacks really leak client data, and does local DP stop them?

Standalone (no blockchain, no Anvil). Uses the same simpleCNN, FashionMNIST and the same
client-side DP rule as the training pipeline (clip update to C, add N(0, (sigma*C)^2)).

Attack 1  Gradient inversion ("Inverting Gradients", Geiping et al. 2020, cosine loss + TV).
          Worst case for the client: the attacker sees an update computed from ONE batch of
          B images (one SGD step) and knows the labels. Metric: PSNR of the best-matching
          reconstruction per original image (higher = more leak; roughly >18 dB is recognisable).
          Also saves results/inversion_grid.png (row 1 = originals, then one row per sigma).

Attack 3  (C1) Gradient inversion when the attacker sees only a k-dimensional CountSketch
          of the update, as the server does in sketchSecAgg. Saves results/sketch_leakage.csv
          and results/sketch_inversion_grid.png. Compare with attack 1 at sigma 0 (plaintext).

Attack 4  Analytic attack (Phong et al. 2017): from a batch-1 update the input features of the
          last layer are recovered exactly (dW[i] / db[i]); an image is then fitted to them.
          Reported: feature cosine (1.0 = exact recovery) and image PSNR next to the all-black
          baseline. Run with --analytic-only. This replaces attacks 1/3 as the leakage evidence,
          because their reconstructions did not beat an all-black image.

Attack 2  Membership inference (loss threshold, Yeom et al. 2018). A client trains on a small
          shard (members); the attacker gets the released model and guesses "member" when the
          loss on a sample is low. Metric: AUC (0.5 = no leak, 1.0 = full leak) plus the
          released model's test accuracy, so the privacy/utility cost is visible.

Clip C is set to the norm of the clean update (as in run_dp_experiments.sh, clip ~ typical
norm), so sigma is the only knob. epsilon is reported for the same sigma over 30 rounds.

Run from the repo root:
    python inference_attacks.py                 # full run, a few minutes on CPU
    python inference_attacks.py --quick         # smoke test, ~1 minute
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model.simple_cnn import simpleCNN

DATA = "./data"
OUT = "results"


# ------------------------------------------------------------------ helpers
def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def params(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def privatize(update, sigma, clip, gen):
    """Same rule as robustAgg.privatize_state_dict, on a flat torch vector."""
    norm = update.norm()
    if norm > clip > 0:
        update = update * (clip / norm)
    if sigma > 0:
        update = update + torch.randn(update.shape, generator=gen) * (sigma * clip)
    return update


def dp_epsilon(sigma, rounds, delta=1e-5):
    if sigma <= 0:
        return float("inf")
    a = np.concatenate([np.linspace(1.01, 10, 400), np.arange(11, 2000)])
    return float((rounds * a / (2 * sigma ** 2) + np.log(1 / delta) / (a - 1)).min())


def psnr(x, y):
    mse = F.mse_loss(x, y).item()
    return 100.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def auc(scores_pos, scores_neg):
    """Probability that a random member scores higher than a random non-member (ties = 0.5)."""
    s = np.concatenate([scores_pos, scores_neg])
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n1, n0 = len(scores_pos), len(scores_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_data():
    tf = transforms.ToTensor()  # no augmentation: we want the true images
    train = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    return train, test


def stack(ds, idx):
    xs, ys = zip(*[ds[i] for i in idx])
    return torch.stack(xs), torch.tensor(ys)


# ------------------------------------------------------------------ attack 1
def client_update_one_batch(model, x, y, lr):
    loss = F.cross_entropy(model(x), y)
    grads = torch.autograd.grad(loss, params(model))
    return flat([-lr * g for g in grads])  # what the client would upload (w_new - w_old)


def invert(model, upload, y, lr, shape, iters, tv, seed):
    """Reconstruct images whose gradient matches the observed upload (labels known)."""
    target = (-upload / lr).detach()
    g = torch.Generator().manual_seed(seed)
    dummy = torch.rand(shape, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([dummy], lr=0.1)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * f) for f in (0.375, 0.625, 0.875)], 0.1)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.cross_entropy(model(dummy), y)
        dg = flat(torch.autograd.grad(loss, params(model), create_graph=True))
        rec = 1 - F.cosine_similarity(dg, target, dim=0)
        d = dummy
        tvl = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean() + (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
        (rec + tv * tvl).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
    return dummy.detach()


def torch_count_sketch(v, h, sgn, k):
    return torch.zeros(k, dtype=v.dtype).index_add(0, h, sgn * v)


def invert_from_sketch(model, sketch, y, lr, shape, iters, tv, seed, h, sgn, k):
    """Like invert(), but the attacker only sees a k-dim CountSketch of the upload (C1)."""
    target = (-sketch / lr).detach()
    g = torch.Generator().manual_seed(seed)
    dummy = torch.rand(shape, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([dummy], lr=0.1)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(iters * f) for f in (0.375, 0.625, 0.875)], 0.1)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.cross_entropy(model(dummy), y)
        dg = flat(torch.autograd.grad(loss, params(model), create_graph=True))
        rec = 1 - F.cosine_similarity(torch_count_sketch(dg, h, sgn, k), target, dim=0)
        d = dummy
        tvl = (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean() + (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()
        (rec + tv * tvl).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            dummy.clamp_(0, 1)
    return dummy.detach()


def blind_baselines(train, batch, seed):
    """PSNR an attacker gets WITHOUT seeing anything: an all-black image, and the average
    training image. Any reconstruction near these numbers has learned nothing."""
    idx = np.random.default_rng(seed).choice(len(train), batch, replace=False)
    x, _ = stack(train, idx)
    mean_img = stack(train, np.random.default_rng(seed + 9).choice(len(train), 2000, replace=False))[0].mean(0)
    black = float(np.mean([psnr(torch.zeros_like(x[i]), x[i]) for i in range(batch)]))
    avg = float(np.mean([psnr(mean_img, x[i]) for i in range(batch)]))
    print(f"  blind baselines (no information): all-black image {black:5.2f} dB | "
          f"average training image {avg:5.2f} dB", flush=True)
    return black, avg


def attack_sketch_inversion(train, dims, batch, iters, lr, seed):
    """C1 leakage: gradient inversion when the server sees only a k-dim sketch (no DP)."""
    set_seed(seed)
    model = simpleCNN(10)
    model.eval()
    idx = np.random.default_rng(seed).choice(len(train), batch, replace=False)
    x, y = stack(train, idx)
    clean = client_update_one_batch(model, x, y, lr)
    d = clean.numel()
    rows, results = [x], []
    for k in dims:
        g = torch.Generator().manual_seed(seed + 100 + k)
        h = torch.randint(0, k, (d,), generator=g)
        sgn = torch.randint(0, 2, (d,), generator=g).float() * 2 - 1
        sk = torch_count_sketch(clean, h, sgn, k)
        rec = invert_from_sketch(model, sk, y, lr, x.shape, iters, 1e-2, seed, h, sgn, k)
        p = best_match_psnr(x, rec)
        results.append((k, float(np.mean(p))))
        rows.append(rec)
        print(f"  sketch k={k:<5d} ({100.0 * k / d:5.2f}% of {d} numbers) -> mean PSNR {np.mean(p):6.2f} dB",
              flush=True)
    try:
        from torchvision.utils import save_image
        os.makedirs(OUT, exist_ok=True)
        save_image(torch.cat(rows), os.path.join(OUT, "sketch_inversion_grid.png"), nrow=batch, padding=2)
        print(f"  saved {OUT}/sketch_inversion_grid.png (row 1 = originals, then k = {dims})")
    except Exception as e:
        print(f"  (could not save image grid: {e})")
    return results


def best_match_psnr(orig, recon):
    """For each original image, PSNR of the closest reconstruction (order-free)."""
    return [max(psnr(recon[j], orig[i]) for j in range(len(recon))) for i in range(len(orig))]


def attack_inversion(train, sigmas, batch, iters, lr, seed):
    set_seed(seed)
    model = simpleCNN(10)
    model.eval()
    idx = np.random.default_rng(seed).choice(len(train), batch, replace=False)
    x, y = stack(train, idx)
    clean = client_update_one_batch(model, x, y, lr)
    clip = clean.norm().item()
    gen = torch.Generator().manual_seed(seed + 1)
    rows, results = [x], []
    for s in sigmas:
        up = privatize(clean.clone(), s, clip, gen)
        rec = invert(model, up, y, lr, x.shape, iters, tv=1e-2, seed=seed)
        p = best_match_psnr(x, rec)
        results.append((s, float(np.mean(p))))
        rows.append(rec)
        print(f"  inversion  sigma {s:<5g} -> mean PSNR {np.mean(p):6.2f} dB", flush=True)
    try:
        from torchvision.utils import save_image
        os.makedirs(OUT, exist_ok=True)
        save_image(torch.cat(rows), os.path.join(OUT, "inversion_grid.png"), nrow=batch, padding=2)
        print(f"  saved {OUT}/inversion_grid.png (row 1 = originals, then sigma = {sigmas})")
    except Exception as e:  # image saving is optional
        print(f"  (could not save image grid: {e})")
    return results


# ------------------------------------------------------------------ attack 4 (analytic)
def features(model, x):
    """Input to the last linear layer (everything in the stack except that layer)."""
    return model.linear_relu_stack[:-1](x)


def analytic_feature_recovery(g, sizes):
    """Phong et al. 2017: for batch 1, dL/dW = delta h^T and dL/db = delta, so any row i
    with delta_i != 0 gives h = dW[i] / db[i] exactly. g = gradient estimate (flat)."""
    n_w, n_b = sizes[-2], sizes[-1]
    dW = g[-(n_w + n_b):-n_b].view(n_b, -1)
    db = g[-n_b:]
    i = int(db.abs().argmax())
    return dW[i] / db[i]


def invert_features(model, h_target, shape, iters, tv, seed):
    """Find an image whose last-layer features match h_target."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=g).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)
    scale = h_target.norm().clamp_min(1e-8)
    for _ in range(iters):
        opt.zero_grad()
        f = features(model, x).view(-1)
        loss = ((f - h_target) ** 2).sum() / scale ** 2
        tvl = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        (loss + tv * tvl).backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(0, 1)
    return x.detach()


def attack_analytic(train, sigmas, dims, iters, lr, seed, n_images=4):
    """Attack 4: exact feature recovery from a batch-1 update, then image inversion from
    the features. Run on the plaintext update, DP-noised updates and C1 sketches (the
    attacker's best per-coordinate estimate from a CountSketch is s[c] * sketch[h[c]])."""
    set_seed(seed)
    model = simpleCNN(10)
    model.eval()
    sizes = [p.numel() for p in params(model)]
    d = sum(sizes)
    rng = np.random.default_rng(seed + 50)
    idx = rng.choice(len(train), n_images, replace=False)
    settings = [("plaintext", 0.0, None)] + [("DP", s_, None) for s_ in sigmas if s_ > 0] + \
               [("sketch", 0.0, k) for k in dims]
    stats = {st: {"cos": [], "psnr": []} for st in settings}
    rows = {st: [] for st in settings}
    originals, blacks = [], []
    for n, j in enumerate(idx):
        x, y = stack(train, [j])
        originals.append(x)
        blacks.append(psnr(torch.zeros_like(x[0]), x[0]))
        h_true = features(model, x).view(-1).detach()
        clean = client_update_one_batch(model, x, y, lr).detach()
        gen = torch.Generator().manual_seed(seed + 200 + n)
        for st in settings:
            kind, sig, k = st
            if kind == "plaintext":
                up = clean
            elif kind == "DP":
                up = privatize(clean.clone(), sig, clean.norm().item(), gen)
            else:
                gk = torch.Generator().manual_seed(seed + 100 + k)
                hh = torch.randint(0, k, (d,), generator=gk)
                sg = torch.randint(0, 2, (d,), generator=gk).float() * 2 - 1
                up = sg * torch_count_sketch(clean, hh, sg, k)[hh]   # unbiased estimate of each coordinate
            h_hat = analytic_feature_recovery(-up / lr, sizes)
            cos = F.cosine_similarity(h_hat, h_true, dim=0).item()
            rec = invert_features(model, h_hat, x.shape, iters, 1e-3, seed)
            stats[st]["cos"].append(cos)
            stats[st]["psnr"].append(psnr(rec[0], x[0]))
            rows[st].append(rec)
    print(f"  blind baseline (all-black image): {np.mean(blacks):6.2f} dB")
    up_bound = invert_features(model, features(model, originals[0]).view(-1).detach(),
                               originals[0].shape, iters, 1e-3, seed)
    print(f"  (feature-inversion ceiling on image 1, true features: {psnr(up_bound[0], originals[0][0]):.2f} dB)")
    results = []
    for st in settings:
        kind, sig, k = st
        label = kind if kind == "plaintext" else (f"DP sigma {sig:g}" if kind == "DP" else f"sketch k={k}")
        c, p_ = float(np.mean(stats[st]["cos"])), float(np.mean(stats[st]["psnr"]))
        results.append((label, c, p_, float(np.mean(blacks))))
        print(f"  {label:16s} feature cosine {c:6.3f} | image PSNR {p_:6.2f} dB "
              f"({p_ - np.mean(blacks):+.2f} vs black)", flush=True)
    try:
        from torchvision.utils import save_image
        os.makedirs(OUT, exist_ok=True)
        grid = [torch.cat(originals)] + [torch.cat(rows[st]) for st in settings]
        save_image(torch.cat(grid), os.path.join(OUT, "analytic_inversion_grid.png"), nrow=n_images, padding=2)
        print(f"  saved {OUT}/analytic_inversion_grid.png (row 1 = originals, then: "
              + ", ".join(r[0] for r in results) + ")")
    except Exception as e:
        print(f"  (could not save image grid: {e})")
    with open(os.path.join(OUT, "analytic_leakage.csv"), "w") as f:
        f.write("setting,feature_cosine,image_psnr_db,black_baseline_db\n")
        for r in results:
            f.write(f"{r[0]},{r[1]:.4f},{r[2]:.2f},{r[3]:.2f}\n")
    print(f"  saved {OUT}/analytic_leakage.csv")
    return results


# ------------------------------------------------------------------ attack 2
def train_local(model, x, y, epochs, lr, bs=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(x[b]), y[b]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def per_sample_loss(model, x, y):
    return F.cross_entropy(model(x), y, reduction="none").numpy()


@torch.no_grad()
def accuracy(model, x, y):
    return 100.0 * (model(x).argmax(1) == y).float().mean().item()


def set_flat(model, vec):
    i = 0
    for p in params(model):
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


def attack_membership(train, test, sigmas, shard, epochs, n_eval, seed):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    mem_idx = rng.choice(len(train), shard, replace=False)
    xm, ym = stack(train, mem_idx)
    xt, yt = stack(test, rng.choice(len(test), max(n_eval, 2000), replace=False))
    xn, yn = xt[:n_eval], yt[:n_eval]           # non-members
    xa, ya = xt[n_eval:], yt[n_eval:]           # held-out set for accuracy

    init = simpleCNN(10)
    w0 = flat([p.detach().clone() for p in params(init)])
    # client starts from the global model (here: init) and trains on its shard
    model = simpleCNN(10)
    model.load_state_dict(init.state_dict())
    model = train_local(model, xm, ym, epochs, lr=0.05, seed=seed)
    w1 = flat([p.detach().clone() for p in params(model)])
    clean = w1 - w0
    clip = clean.norm().item()
    gen = torch.Generator().manual_seed(seed + 2)
    xm_eval, ym_eval = xm[:n_eval], ym[:n_eval]
    results = []
    for s in sigmas:
        released = simpleCNN(10)
        set_flat(released, w0 + privatize(clean.clone(), s, clip, gen))
        released.eval()
        a = auc(-per_sample_loss(released, xm_eval, ym_eval), -per_sample_loss(released, xn, yn))
        acc = accuracy(released, xa, ya)
        results.append((s, a, acc))
        print(f"  membership sigma {s:<5g} -> AUC {a:.3f} | released model test acc {acc:5.1f}%", flush=True)
    return results


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.01, 0.05, 0.1, 0.5, 1])
    ap.add_argument("--batch", type=int, default=4, help="images in the attacked batch")
    ap.add_argument("--iters", type=int, default=2000, help="optimisation steps per inversion")
    ap.add_argument("--shard", type=int, default=500, help="membership: client's training samples")
    ap.add_argument("--epochs", type=int, default=30, help="membership: local epochs")
    ap.add_argument("--rounds", type=int, default=30, help="rounds used for the epsilon column")
    ap.add_argument("--sketch-dims", type=int, nargs="+", default=[16, 64, 256, 1024],
                    help="C1: sketch sizes for the sketch-only inversion attack")
    ap.add_argument("--only-sketch", action="store_true", help="run only the sketch leakage attack")
    ap.add_argument("--skip-mia", action="store_true", help="skip membership inference (attack 2)")
    ap.add_argument("--analytic-only", action="store_true",
                    help="run only attack 4 (analytic feature recovery + image inversion)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.sigmas, a.iters, a.shard, a.epochs, a.sketch_dims = [0, 0.1], 200, 200, 5, [64]

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    train, test = load_data()
    if a.analytic_only:
        print(f"\nAttack 4: analytic feature recovery (batch 1, 4 images, {a.iters} inversion steps)")
        attack_analytic(train, a.sigmas, a.sketch_dims, a.iters, lr=0.01, seed=a.seed)
        raise SystemExit(0)
    print(f"\nBlind baselines (batch {a.batch})")
    blind_black, blind_avg = blind_baselines(train, a.batch, a.seed)
    print(f"\nAttack 3 (C1): gradient inversion from a k-dim sketch only (batch {a.batch}, {a.iters} steps)")
    sk_res = attack_sketch_inversion(train, a.sketch_dims, a.batch, a.iters, lr=0.01, seed=a.seed)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "sketch_leakage.csv"), "w") as f:
        f.write("sketch_dim,inversion_psnr_db\n")
        for k, p in sk_res:
            f.write(f"{k},{p:.2f}\n")
    print(f"  saved {OUT}/sketch_leakage.csv  (compare with the sigma=0 PSNR of attack 1 = plaintext update)")
    if a.only_sketch:
        raise SystemExit(0)
    print(f"\nAttack 1: gradient inversion (batch {a.batch}, {a.iters} steps, labels known)")
    inv = attack_inversion(train, a.sigmas, a.batch, a.iters, lr=0.01, seed=a.seed)
    if a.skip_mia:
        print("\nAttack 2 skipped (--skip-mia)")
        mia = [(s_, float("nan"), float("nan")) for s_ in a.sigmas]
    else:
        print(f"\nAttack 2: membership inference (shard {a.shard}, {a.epochs} local epochs)")
        mia = attack_membership(train, test, a.sigmas, a.shard, a.epochs, n_eval=min(a.shard, 500), seed=a.seed)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "inference_attacks.csv")
    with open(path, "w") as f:
        f.write("sigma,epsilon_30rounds,inversion_psnr_db,membership_auc,released_model_acc\n")
        for (s, p), (_, au, acc) in zip(inv, mia):
            f.write(f"{s},{dp_epsilon(s, a.rounds):.2f},{p:.2f},{au:.4f},{acc:.2f}\n")
    print(f"\n{'sigma':>6} {'epsilon':>10} {'PSNR dB':>8} {'MIA AUC':>8} {'acc %':>6}")
    for (s, p), (_, au, acc) in zip(inv, mia):
        e = dp_epsilon(s, a.rounds)
        print(f"{s:6g} {('inf' if e == float('inf') else f'{e:.1f}'):>10} {p:8.2f} {au:8.3f} {acc:6.1f}")
    print(f"\nblind baselines: all-black {blind_black:.2f} dB, average image {blind_avg:.2f} dB")
    print(f"sketch (C1):     " + ", ".join(f"k={k}: {p:.2f} dB" for k, p in sk_res))
    print(f"\nsaved {path}")
    print("Reading: PSNR high = images recovered; AUC 0.5 = no membership leak; "
          "epsilon small = strong formal privacy.")
