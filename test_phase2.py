"""Phase 2 checks: the screening MPC (mpc/bcfl_screen.mpc on MP-SPDZ, 5 committee parties,
malicious-secure Shamir) gives the same verdicts as the plaintext rule.

Needs MP-SPDZ: set BCFL_MPSPDZ to its folder (malicious-shamir-party.x inside).
    python test_phase2.py                 # synthetic rounds (planted outliers and colluders)
    python test_phase2.py DUMP_DIR        # also replay rounds saved with --dump-screening
"""
import glob
import os
import sys
import time

import numpy as np

from server.aggregation_alg.sketchSecAgg import sketchSecAggAggregator
from server.aggregation_alg import secaggField as sf

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def make(screening):
    return sketchSecAggAggregator(sketch_dim=K, secagg="hardened", reputation=True, class_check=True,
                                  collusion_check=True, screening=screening)


def synthetic_round(rng, r, n=10, k=64, c=10):
    base = rng.normal(0, 0.05, k)
    V = base + rng.normal(0, 0.05, (n, k))
    push = rng.normal(0, 1, k)
    push /= np.linalg.norm(push)
    for i in (0, 1, 2):                         # colluders: same extra direction every round
        V[i] += 0.8 * push + rng.normal(0, 0.01, k)
    if r % 3 == 1:
        V[5] *= 6.0                             # norm outlier
    if r % 4 == 2:
        V[7] = -V[7] * 2                        # direction outlier
    P = rng.dirichlet(np.ones(c) * 5, size=n)
    if r % 2 == 0:
        P[8] = np.eye(c)[3] * 0.9 + 0.01        # lopsided class profile
    return V, P


def run_round(agg, V_float_w, P, weights):
    ints = [sf.encode_int(v) for v in V_float_w]
    wts = [max(w, 1e-12) for w in weights]
    floats = [sf.decode_int(ints[i]) / wts[i] for i in range(len(ints))]
    acc, _, _ = agg._decide(floats, list(P), len(ints), ints, weights)
    lists = (list(agg.last_screen_rejected), list(agg.last_head_rejected),
             list(agg.last_collusion_rejected), list(agg.last_rep_excluded))
    return acc, lists, ints


K = 64
if not os.environ.get("BCFL_MPSPDZ"):
    sys.exit("set BCFL_MPSPDZ to the MP-SPDZ folder")

plain, mpc = make("plain"), make("mpc")
rng = np.random.default_rng(7)
times = []
all_same = True
caught_rounds = []
for r in range(8):
    V, P = synthetic_round(rng, r)
    w_p = [plain.reputation.get(i) for i in range(10)]
    w_m = [mpc.reputation.get(i) for i in range(10)]
    Vw = [V[i] * w_p[i] for i in range(10)]     # clients pre-scale by their public weight
    acc_p, lists_p, ints = run_round(plain, Vw, P, w_p)
    t0 = time.perf_counter()
    acc_m, lists_m, _ = run_round(mpc, Vw, P, w_m)
    total = mpc.mpc.sum_accepted(acc_m)
    times.append(time.perf_counter() - t0)
    expect = np.sum([ints[i] for i in acc_m], axis=0) if acc_m else np.zeros(K, dtype=np.int64)
    same = acc_p == acc_m and lists_p == lists_m
    all_same &= same
    check(f"round {r}: MPC verdicts == plaintext", same,
          f"(screen {lists_m[0]}, class {lists_m[1]}, collusion {lists_m[2]}, rep {lists_m[3]})"
          + ("" if same else f" plaintext {lists_p}"))
    check(f"round {r}: revealed sum of accepted sketches is exact", np.array_equal(total, expect))
    if set(lists_m[2]) >= {0, 1, 2}:
        caught_rounds.append(r)
    plain.round += 1
    mpc.round += 1
check("planted colluders 0,1,2 are caught by the collusion check once the window is full",
      len(caught_rounds) > 0, f"(rounds {caught_rounds})")
print(f"MPC time per round (5 parties, n=10, k=64, C=10): mean {np.mean(times):.2f}s, max {max(times):.2f}s")

if len(sys.argv) > 1:
    files = sorted(glob.glob(os.path.join(sys.argv[1], "round_*.npz")))
    replay = make("mpc")
    bad = 0
    for f in files:
        z = np.load(f)
        if z["sketch_ints"].size == 0:
            sys.exit(f"{f}: dump has no integer sketches (made before Phase 2); re-run with --dump-screening")
        w = z["weights"].tolist()
        n = len(w)
        replay._ages = {i: int(a) for i, a in enumerate(z["ages"])}
        floats = list(z["sketches"])
        acc, _, _ = replay._decide(floats, list(z["profiles"]), n, list(z["sketch_ints"]), w)
        replay.mpc.sum_accepted(acc)
        got = (replay.last_screen_rejected, replay.last_head_rejected, replay.last_collusion_rejected)
        want = (z["screen_rej"].tolist(), z["class_rej"].tolist(), z["coll_rej"].tolist())
        bad += got != want
        check(f"replay {os.path.basename(f)}: MPC checks == recorded plaintext checks", got == want,
              "" if got == want else f"mpc {got} plaintext {want}")
        replay.round += 1

print(f"\n{PASS}/{PASS + FAIL} checks passed")
sys.exit(1 if FAIL else 0)
