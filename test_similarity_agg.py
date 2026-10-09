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
else:
    print("(torch not installed: 3 torch checks skipped)")

print(f"\n{passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
