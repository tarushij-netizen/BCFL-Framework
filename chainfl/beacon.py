"""
Committee randomness beacon for the per-round sketch seed (hardened C1, Phase 1).

Joint Feldman VSS over Ristretto255 among the n committee members (threshold t):
  OPEN phase (while uploads are still being committed)
    1. deal      every member j draws a random scalar s_j, Feldman-shares it (t of n) and posts
                 the commitments C_j plus, for every member m, its share encrypted to m
                 (X25519 + ChaCha20-Poly1305).
    2. complain  every member decrypts its shares and checks them against C_j; a bad or missing
                 share is posted as a complaint (m against j).
    3. answer    a dealer answers each complaint by publishing that share in the clear; anyone
                 can check it against C_j. A dealer that leaves a complaint unanswered, answers
                 it with a bad share, or collects t or more complaints is disqualified.
  CLOSED phase (after the server has posted every upload commitment)
    4. reveal    every qualified dealer publishes s_j; anyone checks s_j * B == C_j[0].
    5. recon     for a dealer that withholds or reveals a wrong value, the members publish
                 their shares of s_j and anyone rebuilds it from t verified shares.
  seed = SHA-256("bcfl-beacon" | round | j | s_j for every qualified dealer j, in order)

Why this is unpredictable and unbiasable: at least t members are honest (t = 3 of 5, at most
2 corrupt), so the sum contains at least one secret that nobody else knows before step 4.
Qualification is decided in the OPEN phase, before any secret is revealed, and a dealer that
refuses to reveal in step 4 cannot change the result because its secret is rebuilt from the
shares. The seed is computed by `seed_from_board` from public posts alone, so every client,
member and the server get the same value and can check it.
"""
import hashlib

from secure import ristretto as rt
from secure import vss
from secure.channel import KeyPair


def _ctx(rnd, j, m):
    return b"bcfl-beacon|" + int(rnd).to_bytes(8, "big") + bytes([j, m])


class BeaconCommittee:
    """The n members (indices 1..n) of the C2 committee, acting as beacon dealers."""

    def __init__(self, board, n=5, t=3, faults=None, identities=None):
        """identities (Phase 4): {m: secure.signing.Identity}; every post is then signed, every
        member reads its own view of the board, and checks the 'setup' anchor before revealing."""
        if not 1 <= t <= n:
            raise ValueError("need 1 <= t <= n")
        self.board, self.n, self.t = board, n, t
        self.ids = identities
        self.alarms = []        # (member, EquivocationError) raised by members that refused
        self.members = list(range(1, n + 1))
        self.keys = {m: KeyPair() for m in self.members}
        self.public_keys = {m: k.pk for m, k in self.keys.items()}
        # faults (test attackers): bad_share {(dealer, member)}, withhold {dealer},
        # wrong_reveal {dealer}, ignore_complaints {dealer}, false_complaint {(member, dealer)}
        self.faults = {k: set(v) for k, v in (faults or {}).items()}
        self._secret = {}       # (rnd, j) -> (s_j, shares) : dealer j's own state

    def _f(self, name):
        return self.faults.get(name, set())

    def _post(self, kind, rnd, author, key, value):
        sig = self.ids[author].sign_post(kind, rnd, author, key, value) if self.ids else None
        self.board.post(kind, rnd, author, key, value, sig)

    def _v(self, m):
        from chainfl.relay import view_of
        return view_of(self.board, ("member", m))

    # ------------------------------------------------------------------ OPEN phase
    def run_open_phase(self, rnd):
        self.deal(rnd)
        self.complain(rnd)
        self.answer(rnd)

    def deal(self, rnd):
        for j in self.members:
            s_j = rt.random_scalar()
            shares, commitments = vss.deal(s_j, self.t, self.members)
            self._secret[(rnd, j)] = (s_j, shares)
            boxes = {}
            for m in self.members:
                y = shares[m]
                if (j, m) in self._f("bad_share"):
                    y = (y + 1) % rt.L
                boxes[m] = self.keys[j].encrypt(self.public_keys[m], _ctx(rnd, j, m), rt.scalar_bytes(y))
            self._post("beacon_deal", rnd, j, j, {"commitments": commitments, "boxes": boxes})

    def _my_share(self, rnd, m, j):
        """Member m's share of dealer j's secret: from its box, else from j's public answer."""
        view = self._v(m)
        deal = deals_on_board(view, rnd).get(j)
        if deal is None:
            return None
        try:
            y = rt.scalar_from_bytes(self.keys[m].decrypt(self.public_keys[j], _ctx(rnd, j, m), deal["boxes"][m]))
            if vss.verify(m, y, deal["commitments"]):
                return y
        except Exception:
            pass
        for author, (jj, mm), y in view.entries("beacon_open", rnd):
            if author == jj == j and mm == m and vss.verify(m, y, deal["commitments"]):
                return y
        return None

    def complain(self, rnd):
        for m in self.members:
            deals = deals_on_board(self._v(m), rnd)
            for j in self.members:
                if j == m:
                    continue
                bad = j not in deals or self._my_share(rnd, m, j) is None
                if bad or (m, j) in self._f("false_complaint"):
                    self._post("beacon_complaint", rnd, m, (m, j), None)

    def answer(self, rnd):
        for j in self.members:
            if j in self._f("ignore_complaints") or (rnd, j) not in self._secret:
                continue
            for author, (m, jj), _ in self._v(j).entries("beacon_complaint", rnd):
                if jj == j and author == m:
                    self._post("beacon_open", rnd, j, (j, m), self._secret[(rnd, j)][1][m])

    # ------------------------------------------------------------------ CLOSED phase
    def run_closed_phase(self, rnd):
        """Reveal, rebuild what was withheld, and return the seed (from public posts only)."""
        self.reveal(rnd)
        self.reconstruct(rnd)
        return seed_from_board(self.board, rnd, self.n, self.t)

    def _refuses(self, m, rnd):
        """Phase 4: member m checks its view of the round's setup against the anchor."""
        from chainfl.relay import EquivocationError
        view = self._v(m)
        if not hasattr(view, "check"):
            return False
        try:
            view.check(rnd, "setup")
            return False
        except EquivocationError as e:
            self.alarms.append((m, e))
            return True

    def reveal(self, rnd):
        for j in qualified_dealers(self.board, rnd, self.n, self.t):
            if j in self._f("withhold") or self._refuses(j, rnd):
                continue
            s_j = self._secret[(rnd, j)][0]
            if j in self._f("wrong_reveal"):
                s_j = (s_j + 1) % rt.L
            self._post("beacon_reveal", rnd, j, j, s_j)

    def reconstruct(self, rnd):
        deals = deals_on_board(self.board, rnd)
        revealed = valid_reveals(self.board, rnd, deals)
        for j in qualified_dealers(self.board, rnd, self.n, self.t):
            if j in revealed:
                continue
            for m in self.members:
                if m == j or self._refuses(m, rnd):
                    continue      # a dealer that withholds does not help rebuild its own secret
                y = self._my_share(rnd, m, j)
                if y is not None:
                    self._post("beacon_recon", rnd, m, (j, m), y)


# ---------------------------------------------------------------------- public verification
def deals_on_board(board, rnd):
    out = {}
    for author, j, v in board.entries("beacon_deal", rnd):
        if author != j or j in out:
            continue          # one deal per dealer, posted by the dealer itself
        out[j] = v
    return out


def qualified_dealers(board, rnd, n, t):
    """Dealers whose deal is well formed and who answered every complaint with a valid share,
    with fewer than t complaints (more would reveal t shares, i.e. the secret)."""
    deals = deals_on_board(board, rnd)
    complaints = {}
    for author, (m, j), _ in board.entries("beacon_complaint", rnd):
        if author == m:
            complaints.setdefault(j, set()).add(m)
    opened = {}
    for author, (j, m), y in board.entries("beacon_open", rnd):
        if author == j and j in deals and vss.verify(m, y, deals[j]["commitments"]):
            opened.setdefault(j, set()).add(m)
    qual = []
    for j in range(1, n + 1):
        d = deals.get(j)
        if d is None or len(d["commitments"]) != t or not all(rt.is_valid_point(c) for c in d["commitments"]):
            continue
        if set(d["boxes"]) != set(range(1, n + 1)):
            continue
        c = complaints.get(j, set())
        if len(c) >= t or not c <= opened.get(j, set()):
            continue
        qual.append(j)
    return qual


def valid_reveals(board, rnd, deals):
    out = {}
    for author, j, s in board.entries("beacon_reveal", rnd):
        if author == j and j in deals and j not in out and rt.g_mul(s) == deals[j]["commitments"][0]:
            out[j] = s
    return out


def seed_from_board(board, rnd, n, t):
    """The round's seed, computed by anyone from the board alone."""
    if not board.closed(rnd):
        raise RuntimeError(f"round {rnd}: the seed is defined only after uploads are fixed")
    deals = deals_on_board(board, rnd)
    qual = qualified_dealers(board, rnd, n, t)
    if not qual:
        raise RuntimeError(f"round {rnd}: no qualified beacon dealer")
    revealed = valid_reveals(board, rnd, deals)
    secrets_ = {}
    for j in qual:
        if j in revealed:
            secrets_[j] = revealed[j]
            continue
        shares = {}
        for author, (jj, m), y in board.entries("beacon_recon", rnd):
            if jj == j and author == m and vss.verify(m, y, deals[j]["commitments"]):
                shares[m] = y
        if len(shares) < t:
            raise RuntimeError(f"round {rnd}: dealer {j} withheld and only {len(shares)} shares were published")
        s = vss.reconstruct(dict(list(shares.items())[:t]))
        if rt.g_mul(s) != deals[j]["commitments"][0]:
            raise RuntimeError(f"round {rnd}: rebuilt secret of dealer {j} does not match its commitment")
        secrets_[j] = s
    h = hashlib.sha256(b"bcfl-beacon|" + int(rnd).to_bytes(8, "big"))
    for j in qual:
        h.update(bytes([j]) + rt.scalar_bytes(secrets_[j]))
    return h.digest()
