"""
C1 Phase 4: the board as a possibly malicious server runs it.

In the earlier phases every party read one honest log (MemoryBoard). In a deployment the posts
travel through the aggregation server, which can (a) drop a post for some readers, (b) forge a
post, (c) show different readers different versions of its own posts (e.g. two accepted sets).
Phase 4 makes each of these either impossible or detected before anyone acts on it:

  signatures   every post is signed by its author (secure/signing.py); keys are in a registry
               that all readers share (on chain in a deployment). A reader ignores any post whose
               signature does not verify, so (b) fails.
  anchors      at four points of the round (board.ANCHOR_PHASES: setup, uploads, outcome, final)
               the server records a digest of the posts it claims to have relayed on the ledger
               (ReputationLedger.postAnchor with --onchain; the ledger shows everyone the same
               value). Before acting, a reader compares the digest of its own view with the
               anchor and refuses on a mismatch, so (a) and (c) are caught before any client
               uploads or any holder reveals a share.
  conflicting  two different signed posts by one author for the same item are themselves a proof
  posts        of misbehaviour; a reader that sees one refuses.

RelayBoard wraps the real board (MemoryBoard or ChainBoard). Writers post through it; every
reader reads through view(reader). The server's attack options (used by the tests and by
--server-attack) only change what individual readers are shown.
"""
import hashlib

from chainfl.board import ANCHOR_PHASES
from secure.signing import canon, verify_post


class EquivocationError(RuntimeError):
    """A reader's view disagrees with the anchored digest, or holds conflicting signed posts."""

    def __init__(self, reader, rnd, phase, why):
        super().__init__(f"round {rnd}: reader {reader} refuses at '{phase}': {why}")
        self.reader, self.rnd, self.phase, self.why = reader, rnd, phase, why


SERVER_KINDS = {"round_accepted", "aggregate", "round_outcome"}
MEMBER_KINDS = {"beacon_deal", "beacon_complaint", "beacon_open", "beacon_reveal", "beacon_recon", "verdict_attest"}


def role_of(kind, author):
    if author == "server" or kind in SERVER_KINDS:
        return "server"
    return "member" if kind in MEMBER_KINDS else "client"


class Registry:
    """Public keys: {'client': {i: pk}, 'member': {m: pk}, 'server': pk}."""

    def __init__(self, clients=None, members=None, server=None):
        self.keys = {"client": dict(clients or {}), "member": dict(members or {}), "server": server}

    def pk(self, kind, author):
        role = role_of(kind, author)
        if role == "server":
            return self.keys["server"]
        return self.keys[role].get(author)


def entries_digest(signed, rnd, kinds):
    """Order-free digest of a set of signed posts."""
    items = sorted(canon([kind, int(rnd), a, k, v, sg]) for kind in kinds for a, k, v, sg in signed(kind, rnd))
    h = hashlib.sha256(b"bcfl-anchor|" + int(rnd).to_bytes(8, "big"))
    for it in items:
        h.update(len(it).to_bytes(4, "big") + it)
    return h.digest()


class RelayBoard:
    def __init__(self, base, registry, verify=True):
        self.base, self.registry = base, registry
        self.verify = verify                    # False = Phase 4 off: readers take what they are shown
        self._hide = {}                         # reader -> [predicate(kind, rnd, author, key, value)]
        self._only = {}                         # (reader, kind, rnd) -> index of the one version shown
        self._inject = {}                       # reader -> [(kind, rnd, author, key, value, sig)]

    # ----- writing (goes to the real board)
    def post(self, kind, rnd, author, key, value, sig=None):
        self.base.post(kind, rnd, author, key, value, sig)

    def commit_uploads(self, rnd, hashes, sigs=None):
        self.base.commit_uploads(rnd, hashes, sigs)

    def close(self, rnd):
        self.base.close(rnd)

    def closed(self, rnd):
        return self.base.closed(rnd)

    def anchor_phase(self, rnd, phase, as_seen_by="server"):
        """Server: anchor the digest of the view it chooses (an honest server anchors its own)."""
        d = self.view(as_seen_by).digest(rnd, phase)
        self.base.anchor(rnd, phase, d)
        return d

    def anchored(self, rnd, phase):
        return self.base.anchored(rnd, phase)

    # server's full view = the real log
    def entries(self, kind, rnd):
        return self.view("server").entries(kind, rnd)

    def signed_entries(self, kind, rnd):
        return self.view("server").signed_entries(kind, rnd)

    def upload_commitments(self, rnd):
        return self.view("server").upload_commitments(rnd)

    # ----- the server's attacks on individual readers
    def hide_from(self, reader, predicate):
        self._hide.setdefault(reader, []).append(predicate)

    def show_version(self, reader, kind, rnd, index):
        """Of several posts of `kind` in round rnd, show `reader` only the index-th one."""
        self._only[(reader, kind, rnd)] = index

    def inject(self, reader, kind, rnd, author, key, value, sig=None):
        self._inject.setdefault(reader, []).append((kind, rnd, author, key, value, sig))

    def view(self, reader):
        return BoardView(self, reader)


class BoardView:
    """What one reader sees: the relayed posts minus anything hidden from it, plus anything
    injected, keeping (if Phase 4 is on) only posts whose signature verifies."""

    def __init__(self, relay, reader):
        self.r, self.reader = relay, reader

    def signed_entries(self, kind, rnd):
        posts = list(self.r.base.signed_entries(kind, rnd))
        if self.reader != "server":
            only = self.r._only.get((self.reader, kind, rnd))
            if only is not None:
                posts = [posts[only]] if only < len(posts) else []
            for pred in self.r._hide.get(self.reader, []):
                posts = [p for p in posts if not pred(kind, rnd, *p[:3])]
            posts += [(a, k, v, sg) for kd, r_, a, k, v, sg in self.r._inject.get(self.reader, [])
                      if kd == kind and r_ == rnd]
        if self.r.verify:
            posts = [p for p in posts if verify_post(self.r.registry.pk(kind, p[0]), kind, rnd, *p)]
        return posts

    def entries(self, kind, rnd):
        return [(a, k, v) for a, k, v, _ in self.signed_entries(kind, rnd)]

    def upload_commitments(self, rnd):
        return {k: v for _, k, v in self.entries("upload_commit", rnd)}

    def closed(self, rnd):
        return self.r.base.closed(rnd)

    def anchored(self, rnd, phase):
        return self.r.base.anchored(rnd, phase)

    def digest(self, rnd, phase):
        return entries_digest(self.signed_entries, rnd, ANCHOR_PHASES[phase])

    def check(self, rnd, phase):
        """Raise EquivocationError unless this view matches the anchor and has no conflicting
        signed posts. With Phase 4 off it never raises (the reader trusts the server)."""
        if not self.r.verify:
            return
        a = self.anchored(rnd, phase)
        if a is None:
            raise EquivocationError(self.reader, rnd, phase, "nothing anchored")
        if a != self.digest(rnd, phase):
            raise EquivocationError(self.reader, rnd, phase, "my view differs from the anchored digest")
        for kind in ANCHOR_PHASES[phase]:
            seen = {}
            for a_, k, v, _ in self.signed_entries(kind, rnd):
                key = (a_, canon(k))
                if key in seen and seen[key] != canon(v):
                    raise EquivocationError(self.reader, rnd, phase,
                                            f"two different signed '{kind}' posts by {a_} for {k}")
                seen[key] = canon(v)


def view_of(board, reader):
    """A reader's view of any board (a plain board has one view for everyone)."""
    return board.view(reader) if hasattr(board, "view") else board
