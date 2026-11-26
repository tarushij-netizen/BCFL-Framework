"""
Phase 1 checks for the hardened C1 protocol: field secure aggregation, commit-first uploads,
the committee beacon, the per-round sketch and the sum check, plus the adversarial tests of the
design doc (each attack passes the old protocol and is caught by the new one).

No blockchain or torch needed; the Rust extension must be built (see secure/ristretto.py).
Run from the repo root:   python3 test_phase1.py
"""
import os
import sys
import types
import importlib

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
for name, path in [("server", "server"), ("server.base", "server/base"),
                   ("server.aggregation_alg", "server/aggregation_alg")]:
    pkg = types.ModuleType(name)
    pkg.__path__ = [os.path.join(ROOT, path)]
    sys.modules[name] = pkg
base_mod = types.ModuleType("server.base.baseAggregator")


class ServerAggregator:
    def __init__(self, model=None, args=None):
        self.model, self.id, self.args, self.model_pool = model, 0, args, []

    def aggregate(self, raw=None):
        return self._aggregate_alg(self.model_pool if raw is None else raw)


base_mod.ServerAggregator = ServerAggregator
sys.modules["server.base.baseAggregator"] = base_mod

sf = importlib.import_module("server.aggregation_alg.secaggField")
sk = importlib.import_module("server.aggregation_alg.sketchSecAgg")
from chainfl.board import MemoryBoard, PhaseError          # noqa: E402
from chainfl import beacon as bc                           # noqa: E402
from secure import ristretto as rt                         # noqa: E402

passed, total = 0, 0


def check(name, cond, extra=""):
    global passed, total
    total += 1
    passed += bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


N, D, K = 10, 600, 64
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- field secure aggregation
X = [rng.standard_normal(D) for _ in range(N)]
XI = [sf.encode_int(x) for x in X]
rd = sf.FieldSecAggRound(N, D)
Y = {i: rd.client_upload_int(i, XI[i]) for i in range(N)}
acc = [2, 3, 4, 5, 6, 7, 8, 9]
tot = rd.server_unmask_sum_int(Y, acc)
check("field secagg: unmasked sum of accepted = exact sum of their encodings (0, 1 rejected)",
      np.array_equal(tot, np.sum([XI[i] for i in acc], axis=0)))
low = np.array([int(a) % 2 ** 40 for a in Y[0]], dtype=np.float64)     # any view of one upload
check("field secagg: one upload alone is uniform in Z_L (no correlation with the update)",
      abs(np.corrcoef(low, X[0])[0, 1]) < 5 / np.sqrt(D) and np.mean([int(a) for a in Y[0]]) > 2 ** 248)
check("field secagg: fixed point round-trips to 2^-24", np.abs(sf.decode_int(XI[0]) - X[0]).max() <= 2 ** -24)

# ---------------------------------------------------------------- sketch from a seed
h1, s1 = sf.sketch_params_from_seed(b"a" * 32, D, K)
h1b, s1b = sf.sketch_params_from_seed(b"a" * 32, D, K)
h2, s2 = sf.sketch_params_from_seed(b"b" * 32, D, K)
check("sketch: same seed gives the same matrix, another seed a different one",
      np.array_equal(h1, h1b) and np.array_equal(s1, s1b) and (h1 != h2).mean() > 0.9)
check("sketch: exact and linear over the integers",
      np.array_equal(sf.sketch_int(XI[0] + XI[1], h1, s1, K),
                     sf.sketch_int(XI[0], h1, s1, K) + sf.sketch_int(XI[1], h1, s1, K)))
SK = [sf.sketch_int(XI[i], h1, s1, K) for i in range(N)]
check("sum check: honest clients pass", sf.sum_check(tot, SK, acc, h1, s1, K))
lie = list(SK)
lie[3] = sf.sketch_int(sf.encode_int(-X[3] / 5), h1, s1, K)
check("sum check: one client reporting another vector's sketch is caught", not sf.sum_check(tot, lie, acc, h1, s1, K))
z = rng.standard_normal(D)
zn = sf.null_space_project(z, h1, s1, K)
check("null-space projection: invisible to its own sketch, visible to a fresh one",
      np.abs(np.bincount(h1, weights=s1 * zn, minlength=K)).max() < 1e-9
      and np.linalg.norm(np.bincount(h2, weights=s2 * zn, minlength=K)) > 0.5 * np.linalg.norm(zn) * np.sqrt(K / D) * 3,
      f"(kept {np.linalg.norm(zn) / np.linalg.norm(z):.2f} of the norm)")

# ---------------------------------------------------------------- board phases
b = MemoryBoard()
b.post("beacon_deal", 0, 1, 1, {"x": 1})
b.commit_uploads(0, [b"h"] * 3)
try:
    b.post("beacon_deal", 0, 2, 2, {})
    ok1 = False
except PhaseError:
    ok1 = True
try:
    b.post("beacon_reveal", 1, 1, 1, 5)
    ok2 = False
except PhaseError:
    ok2 = True
check("board: no deal after uploads are fixed, no reveal before", ok1 and ok2)

# ---------------------------------------------------------------- beacon


def beacon_round(faults=None, rnd=0):
    brd = MemoryBoard()
    com = bc.BeaconCommittee(brd, 5, 3, faults)
    com.run_open_phase(rnd)
    try:
        bc.seed_from_board(brd, rnd, 5, 3)
        early = True
    except RuntimeError:
        early = False
    brd.commit_uploads(rnd, [b"x" * 32] * N)
    seed = com.run_closed_phase(rnd)
    return brd, com, seed, early


brd, com, seed, early = beacon_round()
qual = bc.qualified_dealers(brd, 0, 5, 3)
again = bc.seed_from_board(brd, 0, 5, 3)
check("beacon: all 5 honest dealers qualify; anyone recomputes the same seed from the board",
      qual == [1, 2, 3, 4, 5] and again == seed and len(seed) == 32)
check("beacon: no seed exists while uploads are still open", not early)
brd2, com2, seed2, _ = beacon_round()
check("beacon: two rounds give different seeds", seed != seed2)


def expected_seed(com, rnd, qual):
    import hashlib
    hh = hashlib.sha256(b"bcfl-beacon|" + int(rnd).to_bytes(8, "big"))
    for j in qual:
        hh.update(bytes([j]) + rt.scalar_bytes(com._secret[(rnd, j)][0]))
    return hh.digest()


brd, com, seed, _ = beacon_round({"withhold": {2}})
check("beacon: a dealer that withholds its reveal is rebuilt from 3 shares; seed unchanged",
      bc.qualified_dealers(brd, 0, 5, 3) == [1, 2, 3, 4, 5] and seed == expected_seed(com, 0, [1, 2, 3, 4, 5])
      and len(brd.entries("beacon_recon", 0)) >= 3)
brd, com, seed, _ = beacon_round({"wrong_reveal": {4}})
check("beacon: a wrong reveal fails its commitment and is replaced by the rebuilt secret",
      seed == expected_seed(com, 0, [1, 2, 3, 4, 5]))
brd, com, seed, _ = beacon_round({"bad_share": {(3, 1)}})
check("beacon: a bad share draws a complaint; the dealer answers in public and stays qualified",
      ((1, 3) in [k for _, k, _ in brd.entries("beacon_complaint", 0)])
      and bc.qualified_dealers(brd, 0, 5, 3) == [1, 2, 3, 4, 5])
brd, com, seed, _ = beacon_round({"bad_share": {(3, 1)}, "ignore_complaints": {3}})
check("beacon: a dealer that leaves a complaint unanswered is disqualified",
      bc.qualified_dealers(brd, 0, 5, 3) == [1, 2, 4, 5] and seed == expected_seed(com, 0, [1, 2, 4, 5]))
brd, com, seed, _ = beacon_round({"false_complaint": {(5, 2)}})
check("beacon: a false complaint is answered with a valid share; the honest dealer stays in",
      bc.qualified_dealers(brd, 0, 5, 3) == [1, 2, 3, 4, 5])
brd, com, seed, _ = beacon_round({"bad_share": {(4, 1), (4, 2), (4, 3)}})
check("beacon: a dealer with t = 3 complaints is disqualified (answering would reveal its secret)",
      4 not in bc.qualified_dealers(brd, 0, 5, 3))
# two corrupt members (1, 2) know their own secrets but not the seed: it changes with dealer 3's secret
brd, com, seed, _ = beacon_round()
import hashlib  # noqa: E402
alt = hashlib.sha256(b"bcfl-beacon|" + (0).to_bytes(8, "big"))
for j in [1, 2, 3, 4, 5]:
    sj = com._secret[(0, j)][0] if j != 3 else (com._secret[(0, 3)][0] + 1) % rt.L
    alt.update(bytes([j]) + rt.scalar_bytes(sj))
check("beacon: the seed depends on every honest dealer's secret (2 corrupt members cannot fix it)",
      alt.digest() != seed)

# ---------------------------------------------------------------- aggregator, adversarial
prev_vec = rng.standard_normal(D)
signal = rng.standard_normal(D)
signal /= np.linalg.norm(signal)
prev = {"w": prev_vec[:500].copy(), "b": prev_vec[500:].copy()}


def mk(vec):
    return {"w": vec[:500].copy(), "b": vec[500:].copy()}


def flat(dct):
    return np.concatenate([dct["w"], dct["b"]])


def round_models(seed, n_mal, base, scale=5.0):
    """Clients train from the current global model `base`."""
    r = np.random.default_rng(seed)
    out = []
    for i in range(N):
        u = signal + r.standard_normal(D) / np.sqrt(D)
        if i < n_mal:
            u = -scale * u                      # sign-flip poisoning, as robustAgg does
        out.append(mk(base + u))
    return out


def run(secagg, c1_attack, n_mal=3, rounds=1, reputation=False, zk=True):
    ag = sk.sketchSecAggAggregator(sketch_dim=K, seed=0, secagg=secagg, c1_attack=c1_attack, reputation=reputation,
                                   secagg_zk=zk)
    ag.malicious, ag.attack, ag.attack_scale = list(range(n_mal)), "signflip", 5.0
    ag.set_global_model(prev) if hasattr(ag, "set_global_model") else None
    ag.reference_model = {k: v.copy() for k, v in prev.items()}
    outs, rej, checks = [], [], []
    for t in range(rounds):
        before = flat(ag.reference_model)
        out = ag.aggregate(round_models(100 + t, n_mal, before))
        outs.append(float(np.dot(flat(out) - before, signal)))
        rej.append(ag.rejected_last_round)
        checks.append(ag.last_sum_check)
    return ag, outs, rej, checks


ag, step, rej, _ = run("dh", "none")
check("baseline (old protocol): plain sign-flip attackers are rejected", rej[0] == [0, 1, 2] and step[0] > 0)
ag, step, rej, _ = run("dh", "sketchlie")
check("gap 3, OLD protocol: benign sketch + poisoned upload passes screening and poisons the model",
      rej[0] == [] and step[0] < 0, f"(rejected {rej[0]}, model step along the signal {step[0]:+.2f})")
ag, step, rej, chk = run("hardened", "sketchlie", zk=False)
check("gap 3, NEW protocol (Phase 1, without the Phase 5 proofs): the sum check fails and the round is "
      "discarded (model unchanged)",
      chk[0] is False and abs(step[0]) < 1e-12 and ag.sum_check_failures == 1,
      f"(sum check {chk[0]}, step {step[0]:+.2e})")
ag, step, rej, _ = run("dh", "nullspace")
check("gap 2, OLD protocol: poison hidden in the public sketch's null space passes and poisons the model",
      rej[0] == [] and step[0] < 0, f"(rejected {rej[0]}, step {step[0]:+.2f})")
ag, step, rej, chk = run("hardened", "nullspace")
check("gap 2, NEW protocol: the fresh sketch shows the poison; attackers rejected; sum check passes",
      rej[0] == [0, 1, 2] and step[0] > 0 and chk[0] is True, f"(rejected {rej[0]}, step {step[0]:+.2f})")
ag, step, rej, chk = run("hardened", "none", n_mal=0, rounds=30)
check("NEW protocol, 30 honest rounds: no false alarm, nobody rejected, every sum check passes",
      all(c is True for c in chk) and all(r == [] for r in rej) and ag.sum_check_failures == 0)
ag_h, s_h, _, _ = run("hardened", "none", n_mal=0)
ag_d, s_d, _, _ = run("dh", "none", n_mal=0)
check("NEW protocol: same aggregate as the old one up to fixed-point precision",
      abs(s_h[0] - s_d[0]) < 1e-4, f"({s_h[0]:.6f} vs {s_d[0]:.6f})")
ag, step, rej, chk = run("hardened", "none", n_mal=3, rounds=5, reputation=True)
check("NEW protocol with reputation weights: plain attackers rejected, sum checks pass every round",
      all(c is True for c in chk) and all(set(r) >= {0, 1, 2} for r in rej))
check("NEW protocol: every round's uploads were committed before its seed (board order)",
      all(ag.board.closed(t) for t in range(5)) and
      all(len(ag.board.upload_commitments(t)) == N for t in range(5)))

print(f"\n{passed}/{total} checks passed")
sys.exit(0 if passed == total else 1)
