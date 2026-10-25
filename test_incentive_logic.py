"""Checks the C2 incentive logic (attack timing, Sybil accounts, whitewashing, reward split)
WITHOUT a blockchain: brownie and the contract are replaced by a small Python stand-in that
follows ReputationLedger.sol. Run: python test_incentive_logic.py  (expects 10/10)."""
import os, sys, types, hashlib, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = tempfile.mkdtemp()
class TX: gas_used = 100
class Acct:
    n = 0
    def __init__(s): Acct.n += 1; s.address = f"0x{Acct.n:040x}"
    def transfer(s, *a, **k): pass
class Accounts(list):
    def add(s): return Acct()
class Ledger:
    def __init__(s, committee, thr, init, tau, stake):
        s.thr, s.init, s.tau = thr, init, tau; s.address = "0xLEDGER"; s.acc = {}; s.order = []; s.claim = {}; s.tx = TX(); s.prop = {}
    @classmethod
    def deploy(cls, *a): return cls(*a[:5])
    def register(s, id, payout, opts):
        assert id not in s.acc; s.acc[id] = [s.init, payout]; s.order.append(id); return TX()
    def recordRound(s, r, ps, hs, o):
        assert len(set(ps)) == len(ps); return TX()
    def postVerdicts(s, *a): return TX()
    def threshold(s): return s.thr
    def proposeEpoch(s, e, ids, new, o):
        assert set(ids) <= set(s.order) and len(ids) == len(set(ids))
        s.prop[e] = (list(ids), list(new), 1); return TX()
    def approveEpoch(s, e, ids, new, o):
        i, n, c = s.prop[e]; assert (i, n) == (list(ids), list(new)); c += 1; s.prop[e] = (i, n, c)
        if c >= s.thr:
            for a, r in zip(ids, new): s.acc[a][0] = r
        return TX()
    def distribute(s, e, o):
        el = [a for a in s.order if s.acc[a][0] >= s.tau]; tot = sum(s.acc[a][0] for a in el)
        for a in el: s.claim[s.acc[a][1]] = s.claim.get(s.acc[a][1], 0) + o["value"] * s.acc[a][0] // tot
        return TX()
    def repOf(s, id): return s.acc[id][0]
    def claimable(s, addr): return s.claim.get(addr, 0)
br = types.ModuleType("brownie"); br.accounts = Accounts([Acct()])
br.web3 = types.SimpleNamespace(keccak=lambda b: hashlib.sha3_256(b).digest())
sys.modules["brownie"] = br
for m in ["brownie.project", "brownie.project.chainServer"]:
    sys.modules[m] = types.ModuleType(m)
sys.modules["brownie.project.chainServer"].ReputationLedger = Ledger
ci = types.ModuleType("chainfl.interact"); ci.chain_proxy = None; sys.modules["chainfl.interact"] = ci
import chainfl
ba = types.ModuleType("server.base.baseAggregator")
class ServerAggregator:
    def __init__(s, *a, **k): pass
ba.ServerAggregator = ServerAggregator; sys.modules["server.base.baseAggregator"] = ba
from chainfl.incentive import OnchainIncentives
from server.aggregation_alg.sketchSecAgg import Reputation
from client.poisonedLoader import SCHEDULE, attack_on

ok = 0
def check(c, msg):
    global ok; assert c, msg; ok += 1; print("ok", msg)

SCHEDULE.update(mode="onoff", period=5)
check([attack_on(r) for r in range(12)] == [False]*5 + [True]*5 + [False]*2, "onoff schedule")
SCHEDULE.update(mode="sleeper", period=10)
check([attack_on(r) for r in (0, 9, 10, 29)] == [False, False, True, True], "sleeper schedule")
SCHEDULE.update(mode="always")
check(all(attack_on(r) for r in range(5)), "always schedule")

def sim(reward_tau, whitewash, rounds=30, sybil=0):
    rep = Reputation(); L = OnchainIncentives(10, malicious={0, 1, 2}, reward_tau=reward_tau, sybil=sybil,
                                              out_prefix=os.path.join(OUT, "rt%g_ww%d" % (reward_tau, whitewash)))
    for r in range(rounds):
        acc = [i for i in range(10) if i >= 3]                # screening catches the attackers every round
        rep.update({i: i in acc for i in range(10)}); inc = rep.eligible(acc)
        L.on_round(r, inc, [bytes(32)] * 10, [rep.get(i) for i in range(10)])
        if whitewash:
            for m in (0, 1, 2):
                if rep.get(m) < rep.tau:
                    v = rep.get(m); rep.reset(m); L.rejoin(m, v)
    L.finish({"x": 1}); return L

L = sim(0.6, False, sybil=5)
sy = sum(L.ledger.claimable(x["payout"].address) for x in L.sybils)
check(sy > 0, f"reward tau 0.6 + 5 idle Sybils: Sybils earn {sy/1e18:.2f} ETH -> the Sybil hole")
L = sim(0.9, False, sybil=5)
sy = sum(L.ledger.claimable(x["payout"].address) for x in L.sybils)
check(sy == 0, "reward tau 0.9 + 5 idle Sybils: Sybils earn nothing")
L = sim(0.6, True)
att = sum(L.ledger.claimable(a["payout"].address) for i in (0,1,2) for a in L.owned[i])
check(att == 0, "always-caught whitewasher earns nothing even at tau 0.6")
L = sim(0.9, True)
att = sum(L.ledger.claimable(a["payout"].address) for i in (0,1,2) for a in L.owned[i])
check(att == 0, "reward tau 0.9 + whitewash: attackers earn nothing")
check(sum(L.rejoins) == 90, f"attackers re-registered every round ({sum(L.rejoins)} new identities, stake paid)")
L = sim(0.9, False)
hon = [L.ledger.claimable(L.cur(i)["payout"].address) for i in range(3, 10)]
check(min(hon) > 0 and max(hon) - min(hon) <= 1, "honest clients paid equally at tau 0.9")
import csv
row = list(csv.DictReader(open(os.path.join(OUT, "rt0.9_ww0_summary.csv"))))[0]
check(row["attacker_reward_share"] == "0.0" and row["x"] == "1", "summary CSV written")
print(f"{ok}/{ok} mock checks passed")
