"""
Bulletin board for the hardened C1 round: one ordered, public log that every party reads.

Each round has two phases, and the board refuses posts in the wrong phase:
  OPEN    upload commitments, beacon deals, complaints, dealers' answers to complaints
  CLOSED  beacon reveals, reconstruction shares, the round's outcome
`close(rnd)` is the moment the server posts all upload commitments: from then on uploads are
fixed, and only then may the beacon be revealed. So no client can know the sketch seed while
its upload can still change.

MemoryBoard keeps the log in Python (runs without a blockchain). ChainBoard (chainfl/chainboard.py)
also writes every post to ReputationLedger, which enforces the same phase rules on chain.
"""

OPEN_KINDS = {"upload_commit", "beacon_deal", "beacon_complaint", "beacon_open",
              "secagg_deal", "secagg_complaint", "secagg_open"}
CLOSED_KINDS = {"beacon_reveal", "beacon_recon", "round_outcome", "secagg_reveal"}


class PhaseError(PermissionError):
    pass


class MemoryBoard:
    def __init__(self):
        self.log = []                 # (seq, kind, round, author, key, value)
        self._closed = set()

    # ----- writing
    def post(self, kind, rnd, author, key, value):
        if kind in OPEN_KINDS and rnd in self._closed:
            raise PhaseError(f"round {rnd} is closed: no more '{kind}' posts")
        if kind in CLOSED_KINDS and rnd not in self._closed:
            raise PhaseError(f"round {rnd} is still open: '{kind}' must wait until uploads are fixed")
        if kind not in OPEN_KINDS | CLOSED_KINDS:
            raise ValueError(f"unknown post kind {kind!r}")
        self.log.append((len(self.log), kind, rnd, author, key, value))
        self._on_post(kind, rnd, author, key, value)

    def commit_uploads(self, rnd, hashes):
        """Server posts every client's upload commitment and closes the round in one step."""
        for i, hsh in enumerate(hashes):
            self.post("upload_commit", rnd, "server", i, hsh)
        self.close(rnd)

    def close(self, rnd):
        if rnd in self._closed:
            raise PhaseError(f"round {rnd} already closed")
        self._closed.add(rnd)
        self._on_close(rnd)

    # hooks for ChainBoard
    def _on_post(self, kind, rnd, author, key, value):
        pass

    def _on_close(self, rnd):
        pass

    # ----- reading (anyone)
    def closed(self, rnd):
        return rnd in self._closed

    def entries(self, kind, rnd):
        """[(author, key, value)] in posting order."""
        return [(a, k, v) for _, kd, r, a, k, v in self.log if kd == kind and r == rnd]

    def upload_commitments(self, rnd):
        return {k: v for _, k, v in self.entries("upload_commit", rnd)}
