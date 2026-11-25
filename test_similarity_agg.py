"""
Standalone checks for similarityAgg.py (v2) and robustAgg.py.
No blockchain needed; torch is optional (the torch checks are skipped without it).

Run from the repo root:   python3 test_similarity_agg.py
"""
import os
import sys
import types
import importlib

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# --- stub ServerAggregator so importing does not pull in torch/brownie -------
for name, path in [("server", "server"), ("server.base", "server/base"),
                   ("server.aggregation_alg", "server/aggregation_alg")]:
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(ROOT, path)]
    sys.modules[name] = pkg

base_mod = types.ModuleType("server.base.baseAggregator")


class ServerAggregator:
    def __init__(self, model=None, args=None):
        self.model, self.id, self.args, self.model_pool = model, 0, args, []

    def receive_upload(self, client_pool):
        for c in client_pool:
            self.model_pool.append(c.get_model_state_dict())

    def aggregate(self, raw=None):
        return self._aggregate_alg(self.model_pool if raw is None else raw)


base_mod.ServerAggregator = ServerAggregator
sys.modules["server.base.baseAggregator"] = base_mod

sa = importlib.import_module("server.aggregation_alg.similarityAgg")
ra = importlib.import_module("server.aggregation_alg.robustAgg")

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# --- toy setup ---------------------------------------------------------------
D1, D2 = 300, 100
D = D1 + D2
N = 10
rng = np.random.default_rng(0)
signal = rng.standard_normal(D)
signal /= np.linalg.norm(signal)


def make_dict(vec, cls=np.asarray):
    return {"layer1": cls(vec[:D1].copy()), "layer2": cls(vec[D1:].copy())}


def flat(d):
    return np.concatenate([np.asarray(d["layer1"]).ravel(), np.asarray(d["layer2"]).ravel()])


prev_vec = rng.standard_normal(D)
prev = make_dict(prev_vec)


def honest_models(n, seed):
    r = np.random.default_rng(seed)
    return [make_dict(prev_vec + signal + r.standard_normal(D) / np.sqrt(D)) for _ in range(n)]


class FakeClient:
    def __init__(self, sd):
        self.sd = sd

    def get_model_state_dict(self):
        return self.sd


passed, total = 0, 0


def check(name, cond, extra=""):
    global passed, total
    total += 1
    passed += bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


def screened(models, **kw):
    agg = sa.similarityAggregator(**kw)
    agg.set_global_model(prev)
    agg.round = 1  # skip warm-up
    out = agg.aggregate(models)
    return agg, out


# 1-2: attackers rejected, result == honest mean
for attack in ["signflip", "noise"]:
    honest = honest_models(N, 1)
    models = list(honest)
    r = np.random.default_rng(5)
    for i in range(2):
        models[i] = sa.poison_state_dict(honest[i], prev, attack, 5.0, r)
    agg, out = screened(models)
    honest_mean = np.mean([flat(m) for m in honest[2:]], axis=0)
    plain_mean = np.mean([flat(m) for m in models], axis=0)
    err = np.linalg.norm(flat(out) - honest_mean)
    check(f"{attack}: attackers 0,1 rejected and nobody else", agg.rejected_last_round == [0, 1],
          f"rejected={agg.rejected_last_round} cos={np.round(agg.last_similarities[:2], 3).tolist()} "
          f"ratio={np.round(agg.last_norm_ratios[:2], 2).tolist()}")
    check(f"{attack}: result equals honest mean", err < 1e-9,
          f"(plain FedAvg off by {np.linalg.norm(plain_mean - honest_mean):.3f})")

# 3: freerider rejected (zero update -> cosine 0)
honest = honest_models(N, 2)
models = list(honest)
models[0] = sa.poison_state_dict(honest[0], prev, "freerider")
agg, _ = screened(models)
check("freerider: client 0 rejected", agg.rejected_last_round == [0], f"rejected={agg.rejected_last_round}")

# 4: no false rejections without attackers
agg, _ = screened(honest_models(N, 3))
check("no attackers: nobody rejected", agg.rejected_last_round == [],
      f"min cos={min(agg.last_similarities):.3f}")

# 5: all rejected -> previous model kept
agg, out = screened(honest_models(N, 4), similarity_threshold=0.999)
check("all rejected keeps previous model", np.allclose(flat(out), prev_vec))

# 6: optional warm-up = plain average
agg = sa.similarityAggregator(warmup_rounds=1)
ms = honest_models(N, 5)
out = agg.aggregate(ms)
check("warm-up round is plain average", np.allclose(flat(out), np.mean([flat(m) for m in ms], axis=0)))

# 7-8: pool reset per round and copies
agg = sa.similarityAggregator()
clients = [FakeClient(m) for m in honest_models(N, 6)]
agg.receive_upload(clients)
agg.receive_upload(clients)
check("pool reset each round", len(agg.model_pool) == N, f"len={len(agg.model_pool)}")
clients[0].sd["layer1"][0] += 100.0
check("pool entries are copies", agg.model_pool[0]["layer1"][0] != clients[0].sd["layer1"][0])

# 9: median aggregator is coordinate-wise median, robust to signflip
honest = honest_models(N, 7)
models = list(honest)
for i in range(2):
    models[i] = sa.poison_state_dict(honest[i], prev, "signflip", 5.0)
med = ra.medianTorchAggregator()
out = med.aggregate(models)
check("median aggregator = coordinate-wise median",
      np.allclose(flat(out), np.median([flat(m) for m in models], axis=0)))

# 10: krum never picks an attacker
kr = ra.krumTorchAggregator(num_byzantine=2)
kr.aggregate(models)
check("krum picks an honest client", set(kr.rejected_last_round) >= {0, 1})

# 11-12: attack wrapper poisons only the malicious indices and tracks the global model
Server = ra.make_server_class("similarity", "signflip", malicious=[0, 1], seed=0)
srv = Server()
srv.set_global_model(prev)
srv.round = 1
honest = honest_models(N, 8)
srv.receive_upload([FakeClient(m) for m in honest])
poisoned_ok = all(not np.allclose(flat(srv.model_pool[i]), flat(honest[i])) for i in [0, 1])
untouched_ok = all(np.allclose(flat(srv.model_pool[i]), flat(honest[i])) for i in range(2, N))
check("wrapper poisons exactly clients 0,1", poisoned_ok and untouched_ok)
out = srv.aggregate()
check("wrapper: similarity rejects the poisoned clients; prev global updated",
      srv.rejected_last_round == [0, 1] and np.allclose(flat(srv._prev_global), flat(out)))

# 13: FedAvg under the same attack is pulled away (shows the attack is real)
Fed = ra.make_server_class("fedavg", "signflip", malicious=[0, 1], seed=0)
fed = Fed()
fed.set_global_model(prev)
fed.receive_upload([FakeClient(m) for m in honest])
fout = fed.aggregate()
hm = np.mean([flat(m) for m in honest], axis=0)
check("fedavg under signflip deviates from honest mean", np.linalg.norm(flat(fout) - hm) > 0.5,
      f"(off by {np.linalg.norm(flat(fout) - hm):.3f})")

# 14: default settings screen from round 0 (no unscreened warm-up round)
Server0 = ra.make_server_class("similarity", "signflip", malicious=[0, 1], seed=0)
s0 = Server0()
s0.set_global_model(prev)
s0.receive_upload([FakeClient(m) for m in honest])
s0.aggregate()
check("default: attackers rejected already in round 0", s0.rejected_last_round == [0, 1],
      f"rejected={s0.rejected_last_round}")

# 15-19: local DP
rng_dp = np.random.default_rng(0)
m = honest_models(1, 11)[0]
out, n = ra.privatize_state_dict(m, prev, clip=1e9, sigma=0.0, rng=rng_dp)
check("DP off (huge clip, sigma 0) leaves update unchanged", np.allclose(flat(out), flat(m)))
out, n = ra.privatize_state_dict(m, prev, clip=0.5, sigma=0.0, rng=rng_dp)
check("DP clip scales update norm to clip", abs(np.linalg.norm(flat(out) - prev_vec) - 0.5) < 1e-9,
      f"(raw norm {n:.3f})")
out, _ = ra.privatize_state_dict(make_dict(prev_vec.copy()), prev, clip=1.0, sigma=2.0,
                                 rng=np.random.default_rng(1))
check("DP noise std = sigma*clip", abs(np.std(flat(out) - prev_vec) - 2.0) < 0.2,
      f"(std {np.std(flat(out) - prev_vec):.3f})")
e1, e2 = ra.dp_epsilon(1.0, 50), ra.dp_epsilon(4.0, 50)
check("epsilon falls as sigma rises; infinite without noise",
      e2 < e1 and ra.dp_epsilon(0, 50) == float("inf"), f"(eps sigma1={e1:.1f}, sigma4={e2:.1f})")
DPS = ra.make_server_class("fedavg", "signflip", malicious=[0, 1], seed=0, dp_clip=0.5, dp_sigma=0.0)
dps = DPS()
dps.set_global_model(prev)
hm = honest_models(N, 12)
dps.receive_upload([FakeClient(x) for x in hm])
hon_ok = all(abs(np.linalg.norm(flat(dps.model_pool[i]) - prev_vec) - 0.5) < 1e-9 for i in range(2, N))
att_ok = all(np.linalg.norm(flat(dps.model_pool[i]) - prev_vec) > 1.0 for i in [0, 1])
check("wrapper: DP on honest clients only, attackers unclipped", hon_ok and att_ok)

# 20-25: C1 prototype (secure aggregation + sketch screening)
sk = importlib.import_module("server.aggregation_alg.sketchSecAgg")
rngm = np.random.default_rng(3)
U = [rngm.standard_normal(D) for _ in range(N)]
h_, s_ = sk.count_sketch_params(D, 64, 5)
pk = [sk.client_package(U[i], i, N, 42, h_, s_, 64) for i in range(N)]
acc_set = [2, 3, 4, 5, 6, 7, 8, 9]
tot = sk.server_unmask_sum([p[0] for p in pk], acc_set, N, 42, D)
check("secure agg: unmasked sum over accepted = true sum (rejected = dropouts)",
      np.allclose(tot, np.sum([U[i] for i in acc_set], axis=0), atol=1e-6))
hidden = min(np.linalg.norm(pk[i][0] - U[i]) / np.linalg.norm(U[i]) for i in range(N))
check("secure agg: a single masked upload looks nothing like the update", hidden > 50,
      f"(mask/update norm ratio >= {hidden:.0f})")
a_, b_ = rngm.standard_normal(D), rngm.standard_normal(D)
a_ = a_ + 0.0; b_ = 0.7 * a_ + 0.3 * b_
cs_true = sk.cosine_similarity(a_, b_)
cs_sk = sk.cosine_similarity(sk.count_sketch(a_, h_, s_, 64), sk.count_sketch(b_, h_, s_, 64))
check("sketch keeps cosine roughly (k=64)", abs(cs_true - cs_sk) < 0.15, f"(true {cs_true:.3f}, sketch {cs_sk:.3f})")
for attack in ["signflip", "noise"]:
    SK = ra.make_server_class("sketch", attack, malicious=[0, 1], seed=0, agg_kwargs={"sketch_dim": 64})
    srv = SK()
    srv.set_global_model(prev)
    hm = honest_models(N, 13)
    srv.receive_upload([FakeClient(x) for x in hm])
    out = srv.aggregate()
    honest_mean = np.mean([flat(m) for m in hm[2:]], axis=0)
    check(f"sketch aggregator ({attack}): rejects 0,1 only, result = honest mean",
          srv.rejected_last_round == [0, 1] and np.allclose(flat(out), honest_mean, atol=1e-6),
          f"rejected={srv.rejected_last_round} cos={np.round(srv.last_similarities[:3], 2).tolist()}")
SK0 = ra.make_server_class("sketch", "none", seed=0, agg_kwargs={"sketch_dim": 64})
srv0 = SK0()
srv0.set_global_model(prev)
srv0.receive_upload([FakeClient(x) for x in honest_models(N, 14)])
srv0.aggregate()
check("sketch aggregator: no attackers -> nobody rejected", srv0.rejected_last_round == [],
      f"(min cos {min(srv0.last_similarities):.2f})")

# 26-29: C3 reputation
R = sk.Reputation(init=0.75, beta=0.7, tau=0.6)
for t in range(10):
    R.update({0: t % 2 == 1, 1: True if t != 3 else False})   # 0 = on-off attacker, 1 = honest, 1 false reject
check("reputation: on-off attacker kept out, honest client with one false rejection stays in",
      R.eligible([0, 1]) == [1], f"(r0={R.get(0):.3f}, r1={R.get(1):.3f})")
check("reputation: new identity starting below tau must earn trust",
      sk.Reputation(init=0.5).eligible([7]) == [])

agg = sk.sketchSecAggAggregator(sketch_dim=64, seed=0, reputation=True)
agg.set_global_model(prev)
seen_ok = True
for t in range(6):
    hm = honest_models(N, 20 + t)
    models = list(hm)
    if t % 2 == 0:  # attackers poison only on even rounds, behave on odd rounds
        for i in [0, 1]:
            models[i] = sa.poison_state_dict(hm[i], ra.snapshot_state_dict(agg.reference_model), "signflip", 5.0)
    agg.aggregate(models)
    if t % 2 == 1 and t >= 3:
        seen_ok = seen_ok and {0, 1} <= set(agg.rejected_last_round)
check("repsketch: on-off attackers excluded even in rounds where they behave", seen_ok,
      f"(last rejected {agg.rejected_last_round}, reputation {agg.last_reputation[:3]})")
check("repsketch: honest clients keep full reputation", min(agg.last_reputation[2:]) > 0.9,
      f"(min honest r {min(agg.last_reputation[2:]):.2f})")

# 30-32: class-profile check for backdoor-like attacks
rb = np.random.default_rng(31)
C, FE = 10, 40
base3 = {"body": rb.standard_normal(500), "head.w": rb.standard_normal((C, FE)), "head.b": rb.standard_normal(C)}
sig3 = {k: 0.3 * rb.standard_normal(v.shape) for k, v in base3.items()}
def toy3(attacker, r):
    m = {k: base3[k] + sig3[k] + 0.3 * r.standard_normal(base3[k].shape) for k in base3}
    if attacker:  # extra push on class 0's output row only (what a backdoor to class 0 does)
        m["head.w"][0] += 1.5 * r.standard_normal(FE)
        m["head.b"][0] += 1.5
    return m
def run_cls(on, attackers=2):
    ag = sk.sketchSecAggAggregator(sketch_dim=64, seed=0, class_check=on)
    ag.set_global_model(base3)
    ag.aggregate([toy3(i < attackers, np.random.default_rng(40 + i)) for i in range(N)])
    return ag.rejected_last_round
check("class check off: class-0 attackers slip through the sketch", run_cls(False) == [],
      f"(rejected {run_cls(False)})")
check("class check on: class-0 attackers rejected, honest kept", run_cls(True) == [0, 1],
      f"(rejected {run_cls(True)})")
check("class check on, no attackers: nobody rejected", run_cls(True, attackers=0) == [],
      f"(rejected {run_cls(True, attackers=0)})")

# probation for re-registered identities (whitewashing fix)
def run_prob(probation, rounds=6):
    Rp = sk.Reputation(probation=probation)
    Rp.reset(0)                            # client 0 re-registers before round 0
    seen = []
    for _ in range(rounds):
        Rp.update({i: True for i in range(N)})  # everyone passes screening every round
        seen.append(0 in Rp.eligible(list(range(N))))
        Rp.tick()
    return seen, Rp
seen3, Rp3 = run_prob(3)
check("probation 3: a fresh identity stays out for 3 rounds even when it passes screening, then joins",
      seen3 == [False, False, False, True, True, True], f"(in aggregate per round {seen3})")
check("probation 3: clients present from the start are never on probation",
      all(i in Rp3.eligible(list(range(N))) for i in range(1, N)))
check("probation 0: fresh identity joins at once (earlier behaviour)", run_prob(0)[0][0] is True)

# collusion check (non-IID backdoor): colluders share one direction, honest clients each have their own
rc = np.random.default_rng(77)
DC = 2000
basec = {"w": rc.standard_normal(DC)}
own = [rc.standard_normal(DC) for _ in range(N)]          # each client's own (non-IID) direction
bd = rc.standard_normal(DC)                               # the shared backdoor direction
def run_col(attackers, rounds=6, forget_at=None):
    ag = sk.sketchSecAggAggregator(sketch_dim=64, seed=0, collusion_check=True,
                                   similarity_threshold=-1.0, magnitude_ratio=1e9)
    ag.set_global_model(basec)
    out = []
    for r in range(rounds):
        if forget_at is not None and r == forget_at:
            ag.forget_client(0)
        rr = np.random.default_rng(100 + r)
        # backdoor weight 3: pairwise cosine ~0.8 among colluders, as measured on FashionMNIST non-IID
        ms = [{"w": ag.reference_model["w"] + 0.05 * (1.0 + own[i] + (3.0 * bd if i < attackers else 0)
                                                     + rr.standard_normal(DC))} for i in range(N)]
        ag.aggregate(ms)
        out.append(list(ag.last_collusion_rejected))
    return out, ag
col3, agc = run_col(3)
check("collusion check: 3 colluders rejected once their history covers 3 rounds, honest kept",
      col3[:2] == [[], []] and all(x == [0, 1, 2] for x in col3[2:]), f"(rejected per round {col3}, max cos {agc.last_collusion_max})")
col0, agc0 = run_col(0)
check("collusion check: non-IID honest clients only -> nobody rejected", all(x == [] for x in col0),
      f"(max cos {agc0.last_collusion_max})")
col2, _ = run_col(2)
check("collusion check: only 2 colluders -> not flagged (needs a group of 3; known limit)",
      all(x == [] for x in col2), f"(rejected per round {col2})")
colf, _ = run_col(3, rounds=6, forget_at=3)
check("collusion check: a re-registered identity starts with empty history (judged again after warm-up)",
      0 not in colf[3] and 0 not in colf[4] and 0 in colf[5], f"(rejected per round {colf})")

# 33-40: C1 with real secure aggregation (Diffie-Hellman + Shamir, secaggDH.py)
dh = importlib.import_module("server.aggregation_alg.secaggDH")
sh = dh.shamir_share(123456789, 4, [1, 2, 3, 4, 5, 6])
check("shamir: any 4 of 6 shares rebuild the secret, 3 do not",
      dh.shamir_reconstruct({x: sh[x] for x in (2, 3, 5, 6)}) == 123456789
      and dh.shamir_reconstruct({x: sh[x] for x in (1, 2, 3)}) != 123456789)
ska, pka = dh.dh_keygen()
skb, pkb = dh.dh_keygen()
check("DH: both sides derive the same pair key", dh.dh_agree(ska, pkb) == dh.dh_agree(skb, pka))
rd = dh.DHSecAggRound(N, D)
Y = {i: rd.client_upload(i, U[i]) for i in range(N)}
tot_dh = rd.server_unmask_sum(Y, acc_set)
check("DH secure agg: unmasked sum over accepted = true sum (rejected 0,1 removed)",
      np.allclose(tot_dh, np.sum([U[i] for i in acc_set], axis=0), atol=1e-5),
      f"(max err {np.abs(tot_dh - np.sum([U[i] for i in acc_set], axis=0)).max():.1e})")
single = dh.decode(Y[0])
corr = abs(np.corrcoef(single, U[0])[0, 1])
check("DH secure agg: a single upload is uniform noise in Z_2^64 (no correlation with update)",
      corr < 5 / np.sqrt(D) and np.abs(single).mean() > 1e9, f"(|corr| {corr:.3f})")
try:
    rd.holder_reveal(5, "sk", 2)        # holder 5 already revealed b of accepted client 2
    refused = False
except PermissionError:
    refused = True
check("DH secure agg: a holder never reveals both secrets of one client", refused)
rd2 = dh.DHSecAggRound(N, D)
Y2 = {i: rd2.client_upload(i, U[i]) for i in range(N)}
try:
    rd2.server_unmask_sum(Y2, acc_set, holders=[2, 3, 4])
    aborted = False
except RuntimeError:
    aborted = True
check("DH secure agg: aborts with fewer than t share holders", aborted)
for attack in ["signflip", "noise"]:
    SKD = ra.make_server_class("sketch", attack, malicious=[0, 1], seed=0,
                               agg_kwargs={"sketch_dim": 64, "secagg": "dh"})
    srvd = SKD()
    srvd.set_global_model(prev)
    hm = honest_models(N, 13)
    srvd.receive_upload([FakeClient(x) for x in hm])
    outd = srvd.aggregate()
    honest_mean = np.mean([flat(m) for m in hm[2:]], axis=0)
    check(f"sketch aggregator, secagg=dh ({attack}): rejects 0,1 only, result = honest mean",
          srvd.rejected_last_round == [0, 1] and np.allclose(flat(outd), honest_mean, atol=1e-5),
          f"(rejected={srvd.rejected_last_round}, {srvd.last_secagg_seconds:.2f}s)")

# --- torch checks -------------------------------------------------------------
if HAVE_TORCH:
    tprev = {k: torch.tensor(v, dtype=torch.float32) for k, v in prev.items()}
    tm = [{k: torch.tensor(v, dtype=torch.float32) for k, v in m.items()} for m in honest_models(N, 9)]
    for m in tm:
        m["bn.num_batches_tracked"] = torch.tensor(7)
    tprev["bn.num_batches_tracked"] = torch.tensor(7)
    tp = sa.poison_state_dict(tm[0], tprev, "noise", 5.0)
    check("torch: poison returns tensors", all(torch.is_tensor(v) for v in tp.values()))
    agg = sa.similarityAggregator()
    agg.set_global_model(tprev)
    agg.round = 1
    out = agg.aggregate([tp] + tm[1:])
    check("torch: dtype kept, int buffer kept, attacker rejected",
          out["layer1"].dtype == torch.float32 and out["bn.num_batches_tracked"].item() == 7
          and agg.rejected_last_round == [0])
    med = ra.medianTorchAggregator().aggregate(tm)
    check("torch: median returns tensors of right shape",
          torch.is_tensor(med["layer1"]) and med["layer1"].shape == tm[0]["layer1"].shape)
    sys.path.insert(0, ROOT)
    from client.poisonedLoader import PoisonedLoader, add_trigger
    xb = torch.zeros(8, 1, 28, 28)
    yb = torch.arange(8)
    lf = list(PoisonedLoader([(xb, yb)], "labelflip"))[0]
    bd = list(PoisonedLoader([(xb, yb)], "backdoor", target=0, poison_frac=0.5))[0]
    check("torch: labelflip maps y -> 9 - y", torch.equal(lf[1], 9 - yb))
    trig_ok = bd[0][:4, 0, -4:-1, -4:-1].min().item() == 1.0 and bd[0][4:].abs().sum().item() == 0
    check("torch: backdoor puts trigger + target label on half the batch",
          trig_ok and torch.equal(bd[1][:4], torch.zeros(4, dtype=yb.dtype)) and torch.equal(bd[1][4:], yb[4:])
          and xb.abs().sum().item() == 0)
else:
    print("(torch not installed: 5 torch checks skipped)")

print(f"\n{passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
