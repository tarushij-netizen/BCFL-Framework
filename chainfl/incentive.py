"""
C2: drives ReputationLedger.sol during training.

Per round  : server records each client's pseudonym + hash of its masked upload, then the
             verdicts (included in the aggregate or not).
Per epoch  : the committee (k-of-n) agrees on the new reputation of every account and the
             server splits that epoch's reward pool by reputation.
At the end : prints gas per operation and each client's reward; writes results/onchain_*.csv.

Simulation notes (state these in the paper):
- The pseudonym -> account link is held here in one Python dict standing in for the committee.
  In the design it is secret-shared among the n members; that part is not implemented.
- Reputation values come from the aggregator's own reputation (repsketch). The committee
  would recompute them from the posted verdicts with the same rule.
"""
import csv
import os
import secrets

from chainfl.interact import chain_proxy  # noqa: F401  (connects to the chain, loads the project)
from brownie import accounts, web3
from brownie.project.chainServer import ReputationLedger


def _keccak(*parts):
    return web3.keccak(b"".join(parts))


class OnchainIncentives:
    def __init__(self, n_clients, malicious=(), n_committee=5, threshold=3, epoch_len=5,
                 init_rep=0.75, tau=0.6, reward_wei=10 ** 18, stake_wei=10 ** 16, out_dir="results"):
        self.n = n_clients
        self.malicious = set(malicious)
        self.epoch_len = epoch_len
        self.reward_wei = reward_wei
        self.out_dir = out_dir
        self.server = accounts[0]
        self.committee = [accounts.add() for _ in range(n_committee)]
        for m in self.committee:
            self.server.transfer(m, "1 ether", silent=True)
        self.payouts = [accounts.add() for _ in range(n_clients)]  # fresh, unlinked addresses
        for p in self.payouts:
            self.server.transfer(p, "0.1 ether", silent=True)
        self.gas = {}
        self.ledger = ReputationLedger.deploy([m.address for m in self.committee], threshold,
                                              int(init_rep * 1000), int(tau * 1000), stake_wei,
                                              {"from": self.server, "silent": True})
        self._gas("deploy", self.ledger.tx)
        # each client: secret1 -> reputation account id, secret2 -> per-round pseudonyms
        self.s1 = [secrets.token_bytes(32) for _ in range(n_clients)]
        self.s2 = [secrets.token_bytes(32) for _ in range(n_clients)]
        self.ids = [_keccak(s) for s in self.s1]
        for i in range(n_clients):
            tx = self.ledger.register(self.ids[i], self.payouts[i].address,
                                      {"from": self.payouts[i], "value": stake_wei, "silent": True})
            self._gas("register (per client)", tx)
        self.epoch = 0
        self.history = []
        print(f"[C2] ReputationLedger at {self.ledger.address}: {n_clients} accounts, "
              f"committee {threshold}-of-{n_committee}, epoch = {epoch_len} rounds")

    def _gas(self, name, tx):
        self.gas.setdefault(name, []).append(tx.gas_used)

    def pseudonym(self, i, rnd):
        return _keccak(self.s2[i], rnd.to_bytes(32, "big"))

    def on_round(self, rnd, included, update_hashes, reputation=None):
        """included: client indices whose update went into the aggregate.
        reputation: per-client reputation in [0, 1] from the aggregator (None = keep)."""
        ps = [self.pseudonym(i, rnd) for i in range(self.n)]
        hs = [h if h is not None else bytes(32) for h in update_hashes]
        self._gas("recordRound (10 clients)", self.ledger.recordRound(rnd, ps, hs, {"from": self.server, "silent": True}))
        flags = [i in included for i in range(self.n)]
        self._gas("postVerdicts (10 clients)", self.ledger.postVerdicts(rnd, ps, flags, {"from": self.server, "silent": True}))
        if reputation is not None and (rnd + 1) % self.epoch_len == 0:
            self._close_epoch(reputation)

    def _close_epoch(self, reputation):
        new = [max(0, min(1000, int(round(r * 1000)))) for r in reputation]
        members = self.committee
        self._gas("proposeEpoch", self.ledger.proposeEpoch(self.epoch, self.ids, new, {"from": members[0], "silent": True}))
        k = self.ledger.threshold()
        for m in members[1:k]:
            self._gas("approveEpoch", self.ledger.approveEpoch(self.epoch, self.ids, new, {"from": m, "silent": True}))
        self._gas("distribute", self.ledger.distribute(self.epoch, {"from": self.server, "value": self.reward_wei, "silent": True}))
        onchain = [self.ledger.repOf(i) for i in self.ids]
        self.history.append((self.epoch, onchain))
        print(f"[C2] epoch {self.epoch}: on-chain reputation {onchain}")
        self.epoch += 1

    def finish(self):
        if not self.history:
            print("[C2] no epoch was closed: --onchain needs --aggregator repsketch (it supplies reputation)")
        rewards = [self.ledger.claimable(p.address) for p in self.payouts]
        total = sum(rewards) or 1
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "onchain_gas.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["operation", "calls", "mean_gas", "max_gas"])
            for k, v in self.gas.items():
                w.writerow([k, len(v), int(sum(v) / len(v)), max(v)])
        with open(os.path.join(self.out_dir, "onchain_rewards.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["client", "role", "final_reputation", "reward_wei", "reward_share"])
            final = self.history[-1][1] if self.history else [None] * self.n
            for i in range(self.n):
                w.writerow([i, "attacker" if i in self.malicious else "honest", final[i], rewards[i],
                            f"{rewards[i] / total:.4f}"])
        print("\n[C2] gas per operation (mean / max):")
        for k, v in self.gas.items():
            print(f"  {k:28s} {int(sum(v) / len(v)):>9,d} / {max(v):>9,d}  ({len(v)} calls)")
        hon = [rewards[i] for i in range(self.n) if i not in self.malicious]
        att = [rewards[i] for i in range(self.n) if i in self.malicious]
        print(f"[C2] reward share: honest clients {sum(hon) / total:.1%} "
              f"(each ~{(sum(hon) / max(len(hon), 1)) / total:.1%}), attackers {sum(att) / total:.1%}")
        print(f"[C2] wrote {self.out_dir}/onchain_gas.csv and {self.out_dir}/onchain_rewards.csv")
