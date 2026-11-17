"""
C2: drives ReputationLedger.sol during training.

Per round  : server records each client's pseudonym + hash of its masked upload, then the
             verdicts (included in the aggregate or not).
Per epoch  : the committee (k-of-n) agrees on the new reputation of every account and the
             server splits that epoch's reward pool by reputation.
At the end : prints gas per operation and each client's reward; writes <prefix>_gas.csv,
             <prefix>_rewards.csv and <prefix>_summary.csv.

Two thresholds (Sybil fix, 13 Oct):
- aggregation threshold (repsketch --rep-tau, default 0.6): who takes part in training.
- reward threshold (--reward-tau, default 0.9): who gets paid. It is the contract's `tau`.
  New accounts start at init_rep 0.75: above 0.6 so a new honest client can train at once,
  below 0.9 so a brand-new identity earns nothing until it has been accepted a few rounds.

Idle Sybil accounts (--sybil S): the attacker registers S extra accounts that never upload.
The committee has no verdicts for them, so their reputation stays at init_rep. With a reward
threshold below init_rep they are paid every epoch for nothing (the Sybil hole); with the
threshold above init_rep they earn nothing and the attacker only loses the stakes.

Whitewashing (rejoin): an attacker whose reputation fell can drop its account and register a
new one (fresh secrets, fresh payout address, new stake). `rejoin(i)` simulates that.

Pseudonym -> account link (linkEscrow.py, 2 Nov; verifiable shares 5 Nov): each client escrows
its link secret s2 to the committee as t-of-n Feldman VSS shares (members check their share at
registration), encrypted to each member with Diffie-Hellman. The
server never holds it. At each epoch, t members open the escrow, map the posted pseudonyms to
accounts and recompute every account's reputation from the posted verdicts with the public
reputation rule; that committee-computed list is what goes on chain. The aggregator's own
reputation is only used as a cross-check (a mismatch is printed).

Verdicts posted per round = "passed screening" (sketch + class check), the input of the
reputation rule. Who was finally averaged in also depends on reputation, which is public.

Simulation notes (state these in the paper):
- `self.owned` plays the clients' wallets (they keep s1, s2 and compute their own pseudonyms);
  the escrow boxes and the verdict log are kept in memory instead of being read from the chain.
- The t members that open an epoch learn the links: trust assumption "no t members collude".
"""
import csv
import os
import secrets

from chainfl.interact import chain_proxy  # noqa: F401  (connects to the chain, loads the project)
from brownie import accounts, web3
from brownie.project.chainServer import ReputationLedger

from chainfl.linkEscrow import Committee, escrow


def _keccak(*parts):
    return web3.keccak(b"".join(parts))


class OnchainIncentives:
    def __init__(self, n_clients, malicious=(), n_committee=5, threshold=3, epoch_len=5,
                 init_rep=0.75, reward_tau=0.9, reward_wei=10 ** 18, stake_wei=10 ** 16,
                 out_prefix="results/onchain", sybil=0, rep_params=None):
        self.n = n_clients
        self.malicious = set(malicious)
        self.epoch_len = epoch_len
        self.reward_wei = reward_wei
        self.stake_wei = stake_wei
        self.out_prefix = out_prefix
        self.server = accounts[0]
        self.committee = [accounts.add() for _ in range(n_committee)]
        for m in self.committee:
            self.server.transfer(m, "1 ether", silent=True)
        self.gas = {}
        # off-chain side of the committee: one DH key pair per member (member j <-> committee[j])
        self.escrow = Committee(n_committee, threshold)
        self.rep_params = dict(init=init_rep, beta=0.7, tau=0.6, penalty=0.5)
        self.rep_params.update(rep_params or {})
        self.verdict_log = []            # (round, pseudonyms, passed-screening flags) as posted
        self.next_round = 0
        self.rep_mismatch = 0
        self.ledger = ReputationLedger.deploy([m.address for m in self.committee], threshold,
                                              int(init_rep * 1000), int(reward_tau * 1000), stake_wei,
                                              {"from": self.server, "silent": True})
        self._gas("deploy", self.ledger.tx)
        # per client: list of accounts it has owned; each = dict(s1, s2, id, payout)
        self.owned = [[] for _ in range(n_clients)]
        self.frozen = {}  # id of an abandoned account -> reputation (0-1000) it is left with
        for i in range(n_clients):
            self._new_account(i)
        # idle Sybil accounts, all controlled by the attackers; reputation stays at init_rep
        self.init_rep_int = int(init_rep * 1000)
        self.onchain_rep = {}  # id -> reputation last written on chain (absent = init_rep)
        self.sybils = []
        for _ in range(sybil):
            payout = accounts.add()
            self.server.transfer(payout, "0.1 ether", silent=True)
            sid = _keccak(secrets.token_bytes(32))
            self._gas("register (per account)", self.ledger.register(
                sid, payout.address, {"from": payout, "value": stake_wei, "silent": True}))
            self.sybils.append({"id": sid, "payout": payout})
        self.epoch = 0
        self.history = []
        self.rejoins = [0] * n_clients
        print(f"[C2] ReputationLedger at {self.ledger.address}: {n_clients} accounts, "
              f"committee {threshold}-of-{n_committee}, epoch = {epoch_len} rounds, "
              f"new accounts start at {init_rep}, rewards only at reputation >= {reward_tau}"
              + (f", {sybil} idle Sybil accounts" if sybil else ""))

    # ------------------------------------------------------------------ identities
    def _new_account(self, i):
        payout = accounts.add()  # fresh address, not linked to the client's other accounts
        self.server.transfer(payout, "0.1 ether", silent=True)
        s1, s2 = secrets.token_bytes(32), secrets.token_bytes(32)
        acc = {"s1": s1, "s2": s2, "id": _keccak(s1), "payout": payout}
        tx = self.ledger.register(acc["id"], payout.address,
                                  {"from": payout, "value": self.stake_wei, "silent": True})
        self._gas("register (per account)", tx)
        self.owned[i].append(acc)
        # the client escrows its link secret to the committee; the server never sees it
        complaints = self.escrow.add_box(escrow(acc["id"], s2, self.escrow.public_keys, self.escrow.t),
                                         first_round=self.next_round)
        if complaints:   # every member verifies its share (Feldman VSS); honest clients never hit this
            raise RuntimeError(f"client {i}: escrow refused, members {[c['member'] for c in complaints]} "
                               f"proved their shares invalid")

    def cur(self, i):
        return self.owned[i][-1]

    def rejoin(self, i, rep_now):
        """Client i abandons its account (left at reputation rep_now) and registers a new one."""
        old = self.cur(i)
        self.frozen[old["id"]] = max(0, min(1000, int(round(rep_now * 1000))))
        self._new_account(i)
        self.rejoins[i] += 1

    # ------------------------------------------------------------------ per round / epoch
    def _gas(self, name, tx):
        self.gas.setdefault(name, []).append(tx.gas_used)

    @staticmethod
    def _pseudonym_of(s2, rnd):
        return _keccak(s2, rnd.to_bytes(32, "big"))

    def pseudonym(self, i, rnd):
        """Computed by client i from its own s2 (client side)."""
        return self._pseudonym_of(self.cur(i)["s2"], rnd)

    def on_round(self, rnd, passed, update_hashes, reputation=None):
        """passed: client indices whose update passed screening this round.
        reputation: per-client reputation in [0, 1] from the aggregator, used only to
        cross-check the committee's own tally (None = do not close epochs)."""
        ps = [self.pseudonym(i, rnd) for i in range(self.n)]
        hs = [h if h is not None else bytes(32) for h in update_hashes]
        self._gas(f"recordRound ({self.n} clients)",
                  self.ledger.recordRound(rnd, ps, hs, {"from": self.server, "silent": True}))
        flags = [i in passed for i in range(self.n)]
        self._gas(f"postVerdicts ({self.n} clients)",
                  self.ledger.postVerdicts(rnd, ps, flags, {"from": self.server, "silent": True}))
        self.verdict_log.append((rnd, ps, flags))
        self.next_round = rnd + 1
        if reputation is not None and (rnd + 1) % self.epoch_len == 0:
            self._close_epoch(reputation)

    def _close_epoch(self, reputation):
        """Committee update: t members open the link escrow and recompute every account's
        reputation from the posted verdicts. Only accounts whose reputation changed are sent
        (an abandoned account once, idle Sybil accounts never), so gas does not grow with dead
        accounts."""
        from server.aggregation_alg.sketchSecAgg import Reputation
        k = self.ledger.threshold()
        links = self.escrow.open_links(range(k))
        tally = self.escrow.tally(links, self.verdict_log, self._pseudonym_of, Reputation(**self.rep_params))
        to_int = lambda r: max(0, min(1000, int(round(r * 1000))))
        bad = [i for i in range(self.n) if abs(to_int(tally[self.cur(i)["id"]]) - to_int(reputation[i])) > 1]
        if bad:
            self.rep_mismatch += len(bad)
            print(f"[C2] warning: committee tally differs from the aggregator's reputation for clients {bad}")
        ids, new = [], []
        for box in self.escrow.boxes:
            v = to_int(tally[box["id"]])
            if self.onchain_rep.get(box["id"], self.init_rep_int) != v:
                ids.append(box["id"])
                new.append(v)
        if not ids:  # nothing changed; the contract still needs an applied proposal to pay out
            ids, new = [self.cur(0)["id"]], [self.onchain_rep.get(self.cur(0)["id"], self.init_rep_int)]
        members = self.committee
        self._gas("proposeEpoch", self.ledger.proposeEpoch(self.epoch, ids, new, {"from": members[0], "silent": True}))
        for m in members[1:k]:
            self._gas("approveEpoch", self.ledger.approveEpoch(self.epoch, ids, new, {"from": m, "silent": True}))
        self._gas("distribute", self.ledger.distribute(self.epoch, {"from": self.server, "value": self.reward_wei, "silent": True}))
        for a, v in zip(ids, new):
            self.onchain_rep[a] = v
        onchain = [self.ledger.repOf(self.cur(i)["id"]) for i in range(self.n)]
        self.history.append((self.epoch, onchain))
        print(f"[C2] epoch {self.epoch}: on-chain reputation (current account of each client) {onchain}")
        self.epoch += 1

    # ------------------------------------------------------------------ results
    def finish(self, extra=None):
        if not self.history:
            print("[C2] no epoch was closed: --onchain needs --aggregator repsketch (it supplies reputation)")
        rewards = [sum(self.ledger.claimable(a["payout"].address) for a in self.owned[i]) for i in range(self.n)]
        sybil_reward = sum(self.ledger.claimable(sy["payout"].address) for sy in self.sybils)
        total = (sum(rewards) + sybil_reward) or 1
        d = os.path.dirname(self.out_prefix)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.out_prefix + "_gas.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["operation", "calls", "mean_gas", "max_gas"])
            for k, v in self.gas.items():
                w.writerow([k, len(v), int(sum(v) / len(v)), max(v)])
        final = self.history[-1][1] if self.history else [None] * self.n
        with open(self.out_prefix + "_rewards.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["client", "role", "accounts", "final_reputation", "reward_wei", "reward_share", "stake_wei"])
            for i in range(self.n):
                w.writerow([i, "attacker" if i in self.malicious else "honest", len(self.owned[i]), final[i],
                            rewards[i], f"{rewards[i] / total:.4f}", self.stake_wei * len(self.owned[i])])
            for j, sy in enumerate(self.sybils):
                r = self.ledger.claimable(sy["payout"].address)
                w.writerow([f"sybil{j}", "sybil", 1, self.ledger.repOf(sy["id"]), r, f"{r / total:.4f}", self.stake_wei])
        hon = [rewards[i] for i in range(self.n) if i not in self.malicious]
        att = [rewards[i] for i in range(self.n) if i in self.malicious] + [sybil_reward]
        att_stake = sum(self.stake_wei * len(self.owned[i]) for i in self.malicious) + self.stake_wei * len(self.sybils)
        summary = {
            "honest_reward_share": round(sum(hon) / total, 4),
            "attacker_reward_share": round(sum(att) / total, 4),
            "attacker_reward_eth": round(sum(att) / 1e18, 4),
            "attacker_stake_eth": round(att_stake / 1e18, 4),
            "attacker_new_identities": sum(self.rejoins[i] for i in self.malicious),
            "sybil_accounts": len(self.sybils),
            "sybil_reward_share": round(sybil_reward / total, 4),
            "epochs": len(self.history),
            "committee_tally_mismatches": self.rep_mismatch,
        }
        summary.update(extra or {})
        with open(self.out_prefix + "_summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(summary))
            w.writerow(list(summary.values()))
        print("\n[C2] gas per operation (mean / max):")
        for k, v in self.gas.items():
            print(f"  {k:28s} {int(sum(v) / len(v)):>9,d} / {max(v):>9,d}  ({len(v)} calls)")
        print(f"[C2] reward share: honest clients {sum(hon) / total:.1%} "
              f"(each ~{(sum(hon) / max(len(hon), 1)) / total:.1%}), attackers {sum(att) / total:.1%}")
        if self.sybils:
            print(f"[C2] {len(self.sybils)} idle Sybil accounts got {summary['sybil_reward_share']:.1%} of all rewards")
        print(f"[C2] attackers: {summary['attacker_new_identities']} new identities, stake paid "
              f"{summary['attacker_stake_eth']} ETH, rewards earned {summary['attacker_reward_eth']} ETH")
        for k, v in (extra or {}).items():
            print(f"[C2] {k}: {v}")
        print(f"[C2] wrote {self.out_prefix}_gas.csv, _rewards.csv, _summary.csv")
