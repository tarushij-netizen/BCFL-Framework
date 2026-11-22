"""
ChainBoard: the hardened-C1 bulletin board backed by ReputationLedger (used with --onchain).

Every post is also sent to the contract, which enforces the same OPEN / CLOSED rules as
MemoryBoard (a deal after recordRound, or a reveal before it, reverts on chain). Committee
member m (1..n) posts from committee account m-1; the server posts the upload commitments
(recordRound, which closes the round on chain) and the round outcome.

Data availability: deal data and shares travel in contract events; storage keeps only the
hash of each deal and each revealed secret. `audit(rnd)` checks this copy of the log against
the contract's storage, so a reader that trusts the chain gets the same seed.
"""
from chainfl.board import MemoryBoard


def encode_deal(value, n):
    out = b"".join(value["commitments"])
    for m in range(1, n + 1):
        box = value["boxes"][m]
        out += len(box).to_bytes(2, "big") + box
    return out


class ChainBoard(MemoryBoard):
    def __init__(self, incentives):
        super().__init__()
        self.inc = incentives
        self.ledger = incentives.ledger
        self.members = incentives.committee            # brownie accounts, member m = members[m-1]
        self.n = len(self.members)

    def _acct(self, m):
        return self.members[m - 1]

    def _tx(self, label, fn, *args, sender):
        self.inc._gas(label, fn(*args, {"from": sender, "silent": True}))

    def _on_post(self, kind, rnd, author, key, value):
        L = self.ledger
        if kind == "upload_commit":
            return                                      # sent together by commit_uploads
        if kind == "beacon_deal":
            self._tx("beacon postDeal", L.postDeal, rnd, encode_deal(value, self.n), sender=self._acct(author))
        elif kind == "beacon_complaint":
            m, j = key
            self._tx("beacon postComplaint", L.postComplaint, rnd, self._acct(j).address, sender=self._acct(m))
        elif kind == "beacon_open":
            j, m = key
            self._tx("beacon answerComplaint", L.answerComplaint, rnd, self._acct(m).address,
                     _b32(value), sender=self._acct(j))
        elif kind == "beacon_reveal":
            self._tx("beacon postReveal", L.postReveal, rnd, _b32(value), sender=self._acct(author))
        elif kind == "beacon_recon":
            j, m = key
            self._tx("beacon postReconShare", L.postReconShare, rnd, self._acct(j).address,
                     _b32(value), sender=self._acct(m))
        elif kind == "round_outcome":
            self._tx("postOutcome", L.postOutcome, rnd, bool(value), sender=self.inc.server)

    def commit_uploads(self, rnd, hashes):
        for i, hsh in enumerate(hashes):
            self.post("upload_commit", rnd, "server", i, hsh)
        self.inc.record_commitments(rnd, hashes)         # recordRound: closes the round on chain
        self.close(rnd)

    def audit(self, rnd):
        """Raise if this log disagrees with the contract's storage for round rnd."""
        from brownie import web3
        L = self.ledger
        if not L.roundClosed(rnd):
            raise RuntimeError(f"round {rnd}: closed here but not on chain")
        for author, j, v in self.entries("beacon_deal", rnd):
            if _hx(L.dealHash(rnd, self._acct(author).address)) != _hx(web3.keccak(encode_deal(v, self.n))):
                raise RuntimeError(f"round {rnd}: deal of member {author} differs from the chain")
        for author, j, s in self.entries("beacon_reveal", rnd):
            if _hx(L.revealOf(rnd, self._acct(author).address)) != _hx(_b32(s)):
                raise RuntimeError(f"round {rnd}: reveal of member {author} differs from the chain")


def _b32(x):
    return int(x).to_bytes(32, "little")


def _hx(x):
    t = x.hex() if isinstance(x, (bytes, bytearray)) else str(x)
    t = t.lower()
    return t[2:] if t.startswith("0x") else t
