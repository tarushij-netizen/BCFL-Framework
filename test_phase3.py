"""Phase 3 checks: Feldman VSS on the C1 mask shares (server/aggregation_alg/secaggVSS.py).

    python test_phase3.py
Optional: set BCFL_MPSPDZ to also check MPC screening when a client is left out by VSS.
"""
import importlib
import os
import sys
import types

import numpy as np

# the aggregator package imports brownie through chainfl; a stub is enough here
if "brownie" not in sys.modules:
    try:
        import brownie  # noqa: F401
    except ImportError:
        from unittest.mock import MagicMock
        b = types.ModuleType("brownie")
        b.project, b.network, b.accounts = MagicMock(), MagicMock(), [MagicMock() for _ in range(30)]
        sys.modules["brownie"] = b
        pj = types.ModuleType("brownie.project")
        cs = types.ModuleType("brownie.project.chainServer")
        cs.watermarkNegotiation = cs.clientManager = MagicMock()
        sys.modules["brownie.project"], sys.modules["brownie.project.chainServer"] = pj, cs

sf = importlib.import_module("server.aggregation_alg.secaggField")
sv = importlib.import_module("server.aggregation_alg.secaggVSS")
sk = importlib.import_module("server.aggregation_alg.sketchSecAgg")
from chainfl.board import MemoryBoard, PhaseError  # noqa: E402
from secure import ristretto as rt                  # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


N, D = 10, 40
rng = np.random.default_rng(3)
X = {i: rng.integers(-2 ** 30, 2 ** 30, D, dtype=np.int64) for i in range(N)}


def run(faults=None, accepted=None, rnd=0):
    """Full round on a fresh board; returns (round, board, qualified, sum or exception)."""
    board = MemoryBoard()
    r = sv.VSSSecAggRound(N, D, board, rnd, faults=faults)
    q = r.setup()
    ups = {i: r.client_upload_int(i, X[i]) for i in q}
    board.commit_uploads(rnd, [r.upload_hash(i, ups[i]) if i in ups else b"" for i in range(N)])
    acc = [i for i in (accepted if accepted is not None else q) if i in q]
    r.holders_reveal(acc)
    try:
        out = r.server_unmask_sum_int(ups, acc)
    except Exception as e:  # noqa: BLE001
        out = e
    expect = np.sum([X[i] for i in acc], axis=0)
    return r, board, q, out, expect


# ---------------------------------------------------------------- honest round
r, board, q, out, expect = run(accepted=[0, 1, 2, 4, 5, 6, 8, 9])
check("honest round: everyone qualified", q == list(range(N)))
check("honest round: unmasked sum of accepted == true sum (rejected 3, 7 removed)",
      isinstance(out, np.ndarray) and np.array_equal(out, expect))
check("honest round: no complaints posted", board.entries("secagg_complaint", 0) == [])
deals = sv.deals_on_board(board, 0)
check("deal: C_sk[0] equals the published mask key pk (shared key = published key)",
      all(d["C_sk"][0] == d["pk"] for d in deals.values()))
check("deal: shares travel encrypted (a box is not the plain share)",
      all(len(d["boxes"][j]) == 12 + 64 + 16 for d in deals.values() for j in range(N)))
c = 0
try:
    r.holder_share(1, 0)   # holder 1 reads its own box: fine
    other = sv.VSSSecAggRound(N, D, board, 0)   # fresh keys: cannot open someone else's box
    other.holder_share(1, 0)
    c = other.holder_share(1, 0) is None
except Exception:
    c = True
check("deal: a box opens only with the holder's key", c)
y = r.client_upload_int(4, X[4])
check("upload commitment binds the deal (changing a commitment changes the hash)",
      sv.upload_hash(0, 4, y, deals[4]) != sv.upload_hash(0, 4, y, {**deals[4], "C_b": deals[5]["C_b"]}))

# ---------------------------------------------------------------- dealer faults
r, board, q, out, expect = run(faults={"bad_share": {(2, 5)}})
cpl = [k for _, k, _ in board.entries("secagg_complaint", 0)]
check("bad share to one holder: that holder complains", cpl == [(5, 2)], f"(complaints {cpl})")
check("bad share, dealer answers the complaint: dealer stays qualified, round correct",
      2 in q and isinstance(out, np.ndarray) and np.array_equal(out, expect))

r, board, q, out, expect = run(faults={"bad_share": {(2, 5)}, "ignore_complaints": {2}})
check("bad share, complaint ignored: dealer left out of the round", 2 not in q and len(q) == N - 1, f"(qualified {q})")
check("... and the sum over the rest is exact (its pair masks never entered)",
      isinstance(out, np.ndarray) and np.array_equal(out, expect))

r, board, q, out, expect = run(faults={"bad_share": {(3, j) for j in range(N) if j != 3}})
check("bad shares to every holder, all answered: t or more complaints -> left out", 3 not in q)

r, board, q, out, expect = run(faults={"bad_commitment": {6}})
check("commitment C_sk[0] != pk: deal not well formed -> left out", 6 not in q)

r, board, q, out, expect = run(faults={"false_complaint": {(4, 1)}})
check("false complaint: dealer answers with a valid share and stays qualified",
      1 in q and isinstance(out, np.ndarray) and np.array_equal(out, expect))

check("qualified set is computed from the board alone (anyone gets the same answer)",
      sv.qualified_clients(board, 0, N, r.t) == q)

# ---------------------------------------------------------------- holder faults at unmasking
r, board, q, out, expect = run(faults={"wrong_reveal": {0, 1, 2}}, accepted=[3, 4, 5, 6, 7, 8, 9, 0])
check("3 holders reveal wrong shares: their shares are dropped, sum still exact",
      isinstance(out, np.ndarray) and np.array_equal(out, expect))
check("... and the 3 holders are named", sorted(r.bad_reveals) == [0, 1, 2], f"(bad reveals {sorted(r.bad_reveals)})")

r, board, q, out, expect = run(faults={"wrong_reveal": {0, 1, 2, 3, 4}})
check("5 wrong holders (only n-5-1 < t good shares): unmasking stops with an error, never a wrong sum",
      isinstance(out, RuntimeError), f"({out})")

r, board, q, out, expect = run(faults={"double_reveal": {7}}, accepted=[0, 1, 2, 3, 4, 5, 6, 8])
check("holder revealing both shares of a client: its posts are not used, sum still exact",
      isinstance(out, np.ndarray) and np.array_equal(out, expect) and 7 in r.bad_reveals)

# ---------------------------------------------------------------- board phases
board = MemoryBoard()
r = sv.VSSSecAggRound(N, D, board, 0)
r.setup()
try:
    board.post("secagg_reveal", 0, 1, (1, "b", 0), 5)
    ok = False
except PhaseError:
    ok = True
check("shares cannot be revealed before uploads are fixed (round still open)", ok)
board.commit_uploads(0, [b""] * N)
try:
    board.post("secagg_deal", 0, 1, 1, {})
    ok = False
except PhaseError:
    ok = True
check("no new deals after uploads are fixed (round closed)", ok)

# ---------------------------------------------------------------- Phase 1 without VSS: the gap
f1 = sf.FieldSecAggRound(N, D, faults={"bad_share": {(2, j) for j in range(N)}})
ups = {i: f1.client_upload_int(i, X[i]) for i in range(N)}
try:
    got = f1.server_unmask_sum_int(ups, [i for i in range(N) if i != 7])
    wrong = not np.array_equal(got, np.sum([X[i] for i in range(N) if i != 7], axis=0))
except RuntimeError:
    wrong = True
check("Phase 1 (no VSS): one dealer's bad shares spoil the unmasked sum, and nobody can say who", wrong)
f1 = sf.FieldSecAggRound(N, D, faults={"wrong_reveal": {0}})
ups = {i: f1.client_upload_int(i, X[i]) for i in range(N)}
try:
    got = f1.server_unmask_sum_int(ups, list(range(1, N)))
    wrong = not np.array_equal(got, np.sum([X[i] for i in range(1, N)], axis=0))
except RuntimeError:
    wrong = True
check("Phase 1 (no VSS): one holder's wrong reveal spoils the unmasked sum", wrong)

# ---------------------------------------------------------------- inside the aggregator
DM = 200


def agg_rounds(c1_attack, vss, rounds=3, screening="plain"):
    ag = sk.sketchSecAggAggregator(sketch_dim=16, secagg="hardened", reputation=True, c1_attack=c1_attack,
                                   secagg_vss=vss, screening=screening, class_check=False)
    ag.malicious = [0, 1]
    ag.set_global_model({"w": np.zeros(DM)})
    rr = np.random.default_rng(9)
    base = rr.standard_normal(DM)
    res = []
    for r_ in range(rounds):
        ms = [{"w": ag.reference_model["w"] + 0.01 * (base + 0.3 * rr.standard_normal(DM))} for _ in range(8)]
        before = ag.reference_model["w"].copy()
        ag.aggregate(ms)
        if ag.last_sum_check:
            honest_mean = np.mean([ms[i]["w"] for i in range(8) if i not in ag.rejected_last_round], axis=0)
            model_ok = np.allclose(ag.reference_model["w"], honest_mean, atol=1e-6)
        else:                                   # discarded round: previous model kept
            model_ok = np.array_equal(ag.reference_model["w"], before)
        res.append((list(ag.last_vss_rejected), ag.last_sum_check, list(ag.last_bad_reveals), model_ok))
    return res, ag


res, ag = agg_rounds("badshare", True)
check("aggregator, badshare + VSS: attackers 0,1 left out every round, sum check passes, model = mean of the rest",
      all(r_[0] == [0, 1] and r_[1] is True and r_[3] for r_ in res), f"({res})")
res, ag = agg_rounds("badshare", False)
check("aggregator, badshare without VSS (Phase 1): every round is lost (unmasking or sum check fails)",
      all(r_[1] is False and r_[3] for r_ in res), f"({[(r_[1]) for r_ in res]})")
res, ag = agg_rounds("badreveal", True)
check("aggregator, badreveal + VSS: wrong shares named and skipped, every round kept",
      all(r_[1] is True and set(r_[2]) >= {0, 1} for r_ in res), f"({res})")
res, ag = agg_rounds("badreveal", False)
check("aggregator, badreveal without VSS: rounds lost", all(r_[1] is False for r_ in res))

if os.environ.get("BCFL_MPSPDZ"):
    res, ag = agg_rounds("badshare", True, rounds=3, screening="mpc-check")
    check("MPC screening with clients left out by VSS: same verdicts as plaintext over the present clients",
          ag.mpc_mismatch_rounds == 0 and all(r_[0] == [0, 1] and r_[1] is True for r_ in res),
          f"(mismatch rounds {ag.mpc_mismatch_rounds}, {res})")
    ag.mpc.stop()

print(f"\n{PASS}/{PASS + FAIL} checks passed")
sys.exit(1 if FAIL else 0)
