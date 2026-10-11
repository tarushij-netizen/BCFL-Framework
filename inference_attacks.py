"""
Step B: do inference attacks really leak client data, and does local DP stop them?

Standalone (no blockchain, no Anvil). Uses the same simpleCNN, FashionMNIST and the same
client-side DP rule as the training pipeline (clip update to C, add N(0, (sigma*C)^2)).

Attack 1  Gradient inversion ("Inverting Gradients", Geiping et al. 2020, cosine loss + TV).
          Worst case for the client: the attacker sees an update computed from ONE batch of
          B images (one SGD step) and knows the labels. Metric: PSNR of the best-matching
          reconstruction per original image (higher = more leak; roughly >18 dB is recognisable).
          Also saves results/inversion_grid.png (row 1 = originals, then one row per sigma).

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
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.sigmas, a.iters, a.shard, a.epochs = [0, 0.1], 200, 200, 5

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    train, test = load_data()
    print(f"\nAttack 1: gradient inversion (batch {a.batch}, {a.iters} steps, labels known)")
    inv = attack_inversion(train, a.sigmas, a.batch, a.iters, lr=0.01, seed=a.seed)
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
    print(f"\nsaved {path}")
    print("Reading: PSNR high = images recovered; AUC 0.5 = no membership leak; "
          "epsilon small = strong formal privacy.")
