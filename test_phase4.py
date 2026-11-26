"""Phase 4 checks: a malicious server (chainfl/relay.py, secure/signing.py, sketchSecAgg Phase 4).

    python test_phase4.py
With BCFL_MPSPDZ set it also runs the MPC case (fake_accept, committee attestations).
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
from chainfl.board import MemoryBoard                                       # noqa: E402
from chainfl.relay import EquivocationError, RelayBoard, Registry          # noqa: E402
from secure.signing import Identity, canon, verify_post                    # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


# ---------------------------------------------------------------- signatures and views
ids = {i: Identity() for i in range(3)}
srv = Identity()
reg = Registry({i: k.pk for i, k in ids.items()}, {}, srv.pk)
check("canonical encoding is order-free for dicts and distinguishes types",
      canon({"a": 1, "b": [2, b"x"]}) == canon({"b": [2, b"x"], "a": 1}) and canon(1) != canon("1") != canon(b"1"))
sig = ids[1].sign_post("secagg_complaint", 0, 1, (1, 2), None)
check("a signed post verifies; changing any field breaks it",
      verify_post(ids[1].pk, "secagg_complaint", 0, 1, (1, 2), None, sig)
      and not verify_post(ids[1].pk, "secagg_complaint", 0, 1, (1, 0), None, sig)
      and not verify_post(ids[1].pk, "secagg_complaint", 1, 1, (1, 2), None, sig))

base = MemoryBoard()
relay = RelayBoard(base, reg)
relay.post("secagg_complaint", 0, 1, (1, 2), None, sig)
relay.inject(("client", 0), "secagg_complaint", 0, 2, (2, 0), None, srv.sign_post("secagg_complaint", 0, 2, (2, 0), None))
check("a post forged by the server in a client's name is dropped by the reader",
      relay.view(("client", 0)).entries("secagg_complaint", 0) == [(1, (1, 2), None)])
relay.anchor_phase(0, "setup")
v0, v2 = relay.view(("client", 0)), relay.view(("client", 2))
relay.hide_from(("client", 2), lambda kind, r, a, k, v: a == 1)
ok0 = True
try:
    v0.check(0, "setup")
except EquivocationError:
    ok0 = False
try:
    v2.check(0, "setup")
    ok2 = False
except EquivocationError:
    ok2 = True
check("a reader whose view matches the anchor goes on; one from whom a post was hidden refuses", ok0 and ok2)

base = MemoryBoard()
relay = RelayBoard(base, reg)
base.close(0)
for acc in ([0, 1, 2], [0, 1]):
    relay.post("round_accepted", 0, "server", "accepted", {"accepted": acc},
               srv.sign_post("round_accepted", 0, "server", "accepted", {"accepted": acc}))
relay.anchor_phase(0, "outcome")
try:
    relay.view(("client", 1)).check(0, "outcome")
    conflict = False
except EquivocationError as e:
    conflict = "two different signed" in str(e)
check("two different accepted sets signed by the server are caught as a conflict", conflict)

# ---------------------------------------------------------------- inside the aggregator
N, DM = 10, 300


def rounds(attack, checks, n_rounds=2, screening="plain", sign_flip=False):
    ag = sk.sketchSecAggAggregator(sketch_dim=16, secagg="hardened", reputation=True, class_check=False,
                                   screening=screening, server_checks=checks, server_attack=attack)
    ag.malicious = [0, 1, 2]
    ag.set_global_model({"w": np.zeros(DM)})
    rr = np.random.default_rng(5)
    base = rr.standard_normal(DM)
    res = []
    for _ in range(n_rounds):
        ms = []
        for i in range(N):
            u = 0.01 * (base + 0.3 * rr.standard_normal(DM))
            if sign_flip and i == 0:
                u = -5 * u
            ms.append({"w": ag.reference_model["w"] + u})
        agg = ag.aggregate(ms)
        got = ag.deliver_global(agg, N)
        wrong = [i for i in range(N) if not np.array_equal(got[i]["w"], agg["w"])]
        res.append(dict(alarms=len(ag.last_alarms), sum=ag.last_sum_check, breach=ag.last_privacy_breach,
                        vss=list(ag.last_vss_rejected), model_alarms=list(ag.last_model_alarms), wrong=wrong,
                        rejected=list(ag.rejected_last_round)))
    return res, ag


res, ag = rounds("none", True)
check("honest server, Phase 4 on: rounds complete, no alarms, every client accepts the model",
      all(r["alarms"] == 0 and r["sum"] is True and not r["model_alarms"] and not r["wrong"] for r in res), f"{res}")

res, _ = rounds("split_accepted", False)
check("split accepted set, Phase 4 OFF: the server and the 3 colluding clients rebuild both secrets of an "
      "honest client and recover its individual update", all(r["breach"] is True for r in res), f"{res}")
res, _ = rounds("split_accepted", True)
check("split accepted set, Phase 4 ON: the holders shown the other set refuse; no breach; round stopped",
      all(r["breach"] is False and r["alarms"] > 0 and r["sum"] is False for r in res), f"{res}")

res, _ = rounds("drop_deal", False)
check("hidden deal, Phase 4 OFF: clients disagree on who is in the round and the round is lost",
      all(r["sum"] is False and r["alarms"] == 0 for r in res), f"{res}")
res, _ = rounds("drop_deal", True)
check("hidden deal, Phase 4 ON: caught at the setup anchor, before anyone uploads",
      all(r["alarms"] > 0 and r["sum"] is False for r in res), f"{res}")

res, _ = rounds("forge_complaint", False)
check("forged complaints, Phase 4 OFF: an honest client is thrown out of the round",
      all(3 in r["vss"] for r in res), f"{res}")
res, _ = rounds("forge_complaint", True)
check("forged complaints, Phase 4 ON: the forgeries are ignored, the honest client stays in",
      all(r["vss"] == [] and r["sum"] is True and r["alarms"] == 0 for r in res), f"{res}")

res, _ = rounds("wrong_model", False)
check("wrong global model to one client, Phase 4 OFF: that client trains on it",
      all(r["wrong"] == [3] for r in res), f"{res}")
res, _ = rounds("wrong_model", True)
check("wrong global model to one client, Phase 4 ON: the client catches it and takes the anchored model",
      all(r["model_alarms"] == [3] and r["wrong"] == [] for r in res), f"{res}")

if os.environ.get("BCFL_MPSPDZ"):
    res, ag = rounds("none", True, screening="mpc", sign_flip=True)
    check("MPC screening, honest server: committee attests verdicts and total; clients' public sum check passes",
          all(r["alarms"] == 0 and r["sum"] is True and not r["model_alarms"] and 0 in r["rejected"] for r in res),
          f"{res}")
    ag.mpc.stop()
    res, ag = rounds("fake_accept", False, screening="mpc", sign_flip=True)
    check("fake accept (MPC), Phase 4 OFF: a client the committee rejected gets into the model",
          all(0 not in r["rejected"] for r in res), f"{res}")
    ag.mpc.stop()
    res, ag = rounds("fake_accept", True, screening="mpc", sign_flip=True)
    check("fake accept (MPC), Phase 4 ON: holders compare with the committee's attested verdicts and refuse",
          all(r["alarms"] > 0 and r["sum"] is False for r in res), f"{res}")
    ag.mpc.stop()

print(f"\n{PASS}/{PASS + FAIL} checks passed")
sys.exit(1 if FAIL else 0)
