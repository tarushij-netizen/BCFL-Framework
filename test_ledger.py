"""
Checks for ReputationLedger.sol (C2) on the local Anvil chain. No training involved.
Run from the repo root:  python test_ledger.py
"""
from chainfl.interact import chain_proxy  # noqa: F401  connects and compiles the project
from contextlib import contextmanager

from brownie import accounts, web3
from brownie.project.chainServer import ReputationLedger


class _Revert:
    ok = False
    detail = ""


@contextmanager
def reverts(reason):
    """Passes if the transaction inside reverts (works on every Brownie version)."""
    r = _Revert()
    try:
        yield r
    except Exception as e:  # brownie raises VirtualMachineError on revert
        r.ok = True
        msg = str(e)
        r.detail = "reason matched" if reason in msg else f"reverted ({msg.splitlines()[0][:60]})"

passed, total = 0, 0


def hx(x):
    """Same text form for HexBytes, bytes and Brownie HexString values."""
    t = x.hex() if isinstance(x, (bytes, bytearray)) else str(x)
    t = t.lower()
    return t[2:] if t.startswith("0x") else t


def check(name, cond, extra=""):
    global passed, total
    total += 1
    passed += bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


server = accounts[0]
committee = [accounts.add() for _ in range(5)]
payout = [accounts.add() for _ in range(4)]
for a in committee + payout:
    server.transfer(a, "1 ether", silent=True)
STAKE = 10 ** 16
L = ReputationLedger.deploy([m.address for m in committee], 3, 750, 600, STAKE, {"from": server, "silent": True})
ids = [web3.keccak(text=f"secret1-{i}") for i in range(4)]

for i in range(4):
    L.register(ids[i], payout[i].address, {"from": payout[i], "value": STAKE, "silent": True})
check("4 accounts registered at initial reputation 750", L.accountCount() == 4 and L.repOf(ids[0]) == 750)
with reverts("stake too low") as rv:
    L.register(web3.keccak(text="sybil"), payout[0].address, {"from": payout[0], "value": 0, "silent": True})
check("registration without stake is refused (Sybil cost)", rv.ok, f"({rv.detail})")

ps = [web3.keccak(text=f"pseudo-{i}-r0") for i in range(4)]
hs = [web3.keccak(text=f"update-{i}") for i in range(4)]
L.recordRound(0, ps, hs, {"from": server, "silent": True})
L.postVerdicts(0, ps, [True, True, True, False], {"from": server, "silent": True})
check("round commitments and verdicts stored", hx(L.updateHash(0, ps[2])) == hx(hs[2]) and L.verdict(0, ps[3]) == 2)
with reverts("pseudonym reused") as rv:
    L.recordRound(0, [ps[0]], [hs[0]], {"from": server, "silent": True})
check("a pseudonym cannot be reused in the same round", rv.ok, f"({rv.detail})")
with reverts("only server") as rv:
    L.recordRound(1, ps, hs, {"from": committee[0], "silent": True})
check("only the server can record rounds", rv.ok, f"({rv.detail})")

new = [900, 880, 910, 300]  # client 3 was rejected: reputation falls below tau 600
L.proposeEpoch(0, ids, new, {"from": committee[0], "silent": True})
L.approveEpoch(0, ids, new, {"from": committee[1], "silent": True})
check("2 of 3 required approvals: not applied yet", L.repOf(ids[0]) == 750)
with reverts("different update") as rv:
    L.approveEpoch(0, ids, [1000, 1000, 1000, 1000], {"from": committee[2], "silent": True})
check("a member cannot approve a different reputation list", rv.ok, f"({rv.detail})")
L.approveEpoch(0, ids, new, {"from": committee[2], "silent": True})
check("3-of-5 approvals apply the epoch", [L.repOf(i) for i in ids] == new)
with reverts("only committee") as rv:
    L.approveEpoch(0, ids, new, {"from": payout[0], "silent": True})
check("non-members cannot approve", rv.ok, f"({rv.detail})")

L.distribute(0, {"from": server, "value": 10 ** 18, "silent": True})
c = [L.claimable(p.address) for p in payout]
check("client below tau gets no reward", c[3] == 0, f"(claimable {c})")
check("rewards split by reputation among eligible clients", c[2] > c[0] > c[1] > 0 and sum(c) <= 10 ** 18)
before = payout[0].balance()
L.claim({"from": payout[0], "silent": True})
check("claim pays the payout address", payout[0].balance() > before and L.claimable(payout[0].address) == 0)

# ---- hardened C1 (Phase 1): the beacon's phase rules on chain
R = 7
L.postDeal(R, b"\x01" * 40, {"from": committee[0], "silent": True})
check("a member can post a beacon deal while the round is open",
      hx(L.dealHash(R, committee[0].address)) == hx(web3.keccak(b"\x01" * 40)))
with reverts("already dealt") as rv:
    L.postDeal(R, b"\x02" * 40, {"from": committee[0], "silent": True})
check("one deal per member per round", rv.ok, f"({rv.detail})")
with reverts("only committee") as rv:
    L.postDeal(R, b"\x03" * 40, {"from": payout[0], "silent": True})
check("non-members cannot deal", rv.ok, f"({rv.detail})")
with reverts("round open") as rv:
    L.postReveal(R, b"\x05" * 32, {"from": committee[0], "silent": True})
check("no reveal while uploads can still change", rv.ok, f"({rv.detail})")
psR = [web3.keccak(text=f"pseudo-{i}-r{R}") for i in range(4)]
L.recordRound(R, psR, hs, {"from": server, "silent": True})
check("recordRound closes the round", L.roundClosed(R))
with reverts("round closed") as rv:
    L.postDeal(R, b"\x04" * 40, {"from": committee[1], "silent": True})
check("no deal after the uploads are fixed", rv.ok, f"({rv.detail})")
L.postReveal(R, b"\x05" * 32, {"from": committee[0], "silent": True})
check("the dealer reveals after the round is closed", hx(L.revealOf(R, committee[0].address)) == hx(b"\x05" * 32))
with reverts("no deal") as rv:
    L.postReveal(R, b"\x06" * 32, {"from": committee[1], "silent": True})
check("a member that did not deal cannot reveal", rv.ok, f"({rv.detail})")

# ---- hardened C1 (Phase 4): anchors of what the server relayed
R = 9
d0, d2 = web3.keccak(text="setup-view"), web3.keccak(text="outcome-view")
L.postAnchor(R, 0, d0, {"from": server, "silent": True})
check("the server anchors the setup digest while the round is open", hx(L.anchorOf(R, 0)) == hx(d0))
with reverts("already anchored") as rv:
    L.postAnchor(R, 0, web3.keccak(text="other"), {"from": server, "silent": True})
check("an anchor cannot be replaced (one value for every reader)", rv.ok, f"({rv.detail})")
with reverts("only server") as rv:
    L.postAnchor(R, 1, d0, {"from": committee[0], "silent": True})
check("only the server anchors", rv.ok, f"({rv.detail})")
with reverts("round open") as rv:
    L.postAnchor(R, 2, d2, {"from": server, "silent": True})
check("no outcome anchor before the uploads are fixed", rv.ok, f"({rv.detail})")
psR = [web3.keccak(text=f"pseudo-{i}-r{R}") for i in range(4)]
L.recordRound(R, psR, hs, {"from": server, "silent": True})
L.postAnchor(R, 2, d2, {"from": server, "silent": True})
check("outcome anchor after the round is closed", hx(L.anchorOf(R, 2)) == hx(d2))
with reverts("round closed") as rv:
    L.postAnchor(R, 0, d0, {"from": server, "silent": True})
check("no setup anchor after the round is closed", rv.ok, f"({rv.detail})")

print(f"\n{passed}/{total} checks passed")
