"""Phase 5 checks: commitments and zero-knowledge proofs tie each client's screening inputs to its
upload (secure/zk.py, server/aggregation_alg/zkSecAgg.py).

    python test_phase5.py
With BCFL_MPSPDZ set it also runs the MPC cases.
"""
import importlib
import os
import sys
import types

import numpy as np

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

sk = importlib.import_module("server.aggregation_alg.sketchSecAgg")
from secure import ristretto as rt, zk          # noqa: E402
from secure.ristretto import L                  # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- the proof itself
rng = np.random.default_rng(0)
d = 300
rows = zk.rows_layout(d, d - 10 * 20 - 10, 10, 20, d - 10)
xs = [int(v) for v in rng.integers(-2 ** 24, 2 ** 24, d)]
x = [v % L for v in xs]
q = [sum(xs[j] ** 2 for j in idx) for idx in rows]
r_x, mu, r_mu = rt.random_scalar(), rt.random_scalar(), rt.random_scalar()
r_q = [rt.random_scalar() for _ in rows]
a = [int(v) for v in rng.integers(0, 2 ** 40, d)]
T = (sum(ai * xi for ai, xi in zip(a, xs)) + mu) % L
Cx, Cmu = zk.vec_commit(x, r_x), zk.scalar_commit(mu, r_mu)
Cq = [zk.scalar_commit(qc, r) for qc, r in zip(q, r_q)]
pf = zk.prove_update(x, r_x, mu, r_mu, r_q, a, T, rows, b"t")
check("proof: an honest statement verifies", zk.verify_update(Cx, Cmu, Cq, a, T, rows, pf, b"t"))
check("proof: a wrong linear target fails", not zk.verify_update(Cx, Cmu, Cq, a, (T + 1) % L, rows, pf, b"t"))
Cq_bad = list(Cq)
Cq_bad[2] = zk.scalar_commit(q[2] + 1, r_q[2])
check("proof: a wrong class-row norm fails", not zk.verify_update(Cx, Cmu, Cq_bad, a, T, rows, pf, b"t"))
x2 = list(x)
x2[5] = (x2[5] + 1) % L
check("proof: a commitment to another update fails",
      not zk.verify_update(zk.vec_commit(x2, r_x), Cmu, Cq, a, T, rows, pf, b"t"))
z_mean = np.mean([int(v) for v in pf["z"]]) / L
check("proof: the response looks uniform (mean of z / L near 0.5)", 0.4 < z_mean < 0.6, f"({z_mean:.3f})")

# ---------------------------------------------------------------- inside the aggregator
N = 10


def rounds(c1_attack="none", zk_on=True, screening="plain", n_rounds=2, class_check=True):
    ag = sk.sketchSecAggAggregator(sketch_dim=16, secagg="hardened", reputation=True, class_check=class_check,
                                   screening=screening, c1_attack=c1_attack, secagg_zk=zk_on)
    ag.malicious, ag.attack, ag.attack_scale = [0, 1, 2], "signflip", 5.0
    ag.set_global_model({"w": np.zeros((10, 20)), "b": np.zeros(10)})
    rr = np.random.default_rng(4)
    base_w, base_b = rr.standard_normal((10, 20)), rr.standard_normal(10)
    res = []
    for _ in range(n_rounds):
        ms = []
        for i in range(N):
            dw = 0.01 * (base_w + 0.3 * rr.standard_normal((10, 20)))
            db = 0.01 * (base_b + 0.3 * rr.standard_normal(10))
            if i in ag.malicious and c1_attack != "none":
                dw, db = -5 * dw, -5 * db             # the poisoned update the attacker uploads
            ms.append({"w": ag.reference_model["w"] + dw, "b": ag.reference_model["b"] + db})
        before = np.concatenate([ag.reference_model["w"].ravel(), ag.reference_model["b"]])
        agg = ag.aggregate(ms)
        after = np.concatenate([agg["w"].ravel(), agg["b"]])
        signal = np.concatenate([base_w.ravel(), base_b])
        res.append(dict(sum=ag.last_sum_check, zk=list(ag.last_zk_rejected), late=list(ag.last_zk_late),
                        rejected=list(ag.rejected_last_round), step=float(np.dot(after - before, signal))))
    return res, ag


res, ag = rounds()
res_off, _ = rounds(zk_on=False)
check("honest clients, plaintext screening: every proof verifies, nobody named, sum check passes, and the "
      "screening verdicts are the same as without Phase 5",
      all(r["sum"] is True and not r["zk"] and not r["late"] for r in res)
      and [r["rejected"] for r in res] == [r["rejected"] for r in res_off], f"{res} vs {res_off}")

res, _ = rounds("sketchlie", zk_on=False)
check("sketch lie WITHOUT Phase 5: screening is fooled, the sum check catches it, the round is lost, nobody named",
      all(r["sum"] is False for r in res), f"{res}")
res, _ = rounds("sketchlie")
check("sketch lie WITH Phase 5: the liars' proofs fail, they are named before screening, the round is kept "
      "and moves the model the honest way", all(r["zk"] == [0, 1, 2] and r["sum"] is True and r["step"] > 0
                                                 for r in res), f"{res}")

res, _ = rounds("commitlie", zk_on=False)
check("commit lie WITHOUT Phase 5: the round is lost", all(r["sum"] is False for r in res), f"{res}")
res, _ = rounds("commitlie")
check("commit lie WITH Phase 5: the liars pass screening with their cover, are caught by the upload relation, "
      "removed from the aggregate; the round is kept",
      all(r["late"] == [0, 1, 2] and r["sum"] is True and r["step"] > 0 for r in res), f"{res}")

res, _ = rounds("pairlie")
check("pair lie WITH Phase 5: the pair dispute names the liars before screening, the honest partner is not blamed",
      all(r["zk"] == [0, 1, 2] and r["sum"] is True and r["step"] > 0 for r in res), f"{res}")

# a colluding pair committing the same fake pair mask: the audit catches the accepted one once
# the other is rejected (the pair check cannot, both sides agree)
from chainfl.board import MemoryBoard                          # noqa: E402
from server.aggregation_alg.secaggVSS import VSSSecAggRound     # noqa: E402
from server.aggregation_alg.zkSecAgg import ZKRound             # noqa: E402
from server.aggregation_alg import secaggField as sf            # noqa: E402
board = MemoryBoard()
fr = VSSSecAggRound(N, d, board, 0)
qual = fr.setup()
zr = ZKRound(N, d, rows, board, 0, fr, faults={"pair_collude": {(1, 2)}})
X = {i: rng.integers(-2 ** 20, 2 ** 20, d) for i in range(N)}
for i in qual:
    zr.commit(i, X[i], X[i], qual)
ups = {i: fr.client_upload_int(i, X[i]) for i in qual}
board.commit_uploads(0, [b"x"] * N)
check("colluding pair with the same fake pair mask: the pair check sees nothing", zr.pair_check(qual) == [])
acc = [i for i in qual if i != 2]
fr.holders_reveal(acc)
sk2 = fr.rebuild("sk", 2)
check("... the audit with the rejected colluder's rebuilt key names the accepted one",
      not zr.audit_ok(1, 2, sk2) and all(zr.audit_ok(i, 2, sk2) for i in acc if i != 1))
b1 = fr.rebuild("b", 1)
check("... and so does its upload relation", not zr.relation_ok(1, ups[1], b1, qual)
      and all(zr.relation_ok(i, ups[i], fr.rebuild("b", i), qual) for i in acc if i != 1))

if os.environ.get("BCFL_MPSPDZ"):
    res, ag = rounds(screening="mpc")
    check("MPC, honest: the binding outputs match every proof; nobody named; sum check passes",
          all(r["sum"] is True and not r["zk"] and not r["late"] for r in res), f"{res}")
    ag.mpc.stop()
    res, ag = rounds("sketchlie", screening="mpc")
    check("MPC, sketch lie: the MPC's binding value does not match the liars' proofs; named before screening",
          all(r["zk"] == [0, 1, 2] and r["sum"] is True and r["step"] > 0 for r in res), f"{res}")
    ag.mpc.stop()
    res, ag = rounds("commitlie", screening="mpc-check")
    check("MPC, commit lie: caught by the upload relation and removed; MPC verdicts equal plaintext",
          all(r["late"] == [0, 1, 2] and r["sum"] is True for r in res) and ag.mpc_mismatch_rounds == 0,
          f"{res} mismatches {ag.mpc_mismatch_rounds}")
    ag.mpc.stop()

print(f"\n{PASS}/{PASS + FAIL} checks passed")
sys.exit(1 if FAIL else 0)
