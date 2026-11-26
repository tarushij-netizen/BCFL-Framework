"""
Hardened C1, Phase 3: secure aggregation whose Shamir shares are verifiable (Feldman VSS).

Gap 1 (Phase 1 and earlier): a client hands out plain Shamir shares of its mask secrets. A
malicious client can hand some holders wrong shares. Nobody can tell until the server rebuilds
the secret: then a wrong b_i or sk_j makes the unmasked sum garbage (the sum check fails and the
round is lost), and nobody can say whose fault it was. A malicious *holder* can do the same by
returning a wrong share at unmasking time.

Phase 3 protocol (per round; all public posts go to the round's board, chainfl/board.py):
  setup (OPEN phase, before uploads)
    client i: mask key sk_i (Ristretto255 scalar), pk_i = sk_i * B, self-mask seed b_i (scalar);
      Feldman-deals sk_i and b_i to the holders 1..n (holder j has x = j + 1), t of n;
      posts  secagg_deal = {pk, C_sk[0..t-1], C_b[0..t-1], boxes}, where each holder's two
      shares travel in a box encrypted to that holder (X25519 + ChaCha20-Poly1305, secure/channel).
      The deal is well formed only if C_sk[0] == pk: the shared key is the published key.
    holder j: opens its boxes and checks y * B == sum_k x^k C_k for both shares; for every dealer
      whose box is missing, unreadable or fails the check it posts secagg_complaint (j, i).
    client i: answers each complaint by posting that holder's two shares in the clear
      (secagg_open). Anyone can check them against the commitments.
    qualified set Q (computed by anyone from the board): well-formed deal, every complaint
      answered with valid shares, fewer than t complaints (t answered complaints would make t
      shares public). A client outside Q is out of the round: it does not upload, and nobody's
      pair mask includes it.
  upload: client i in Q masks with  PRG(b_i) + sum_{j in Q, j>i} PRG(k_ij) - sum_{j in Q, j<i} PRG(k_ij),
    k_ij = SHA-256("bcfl-pair" | sk_i * pk_j | pk_lo | pk_hi). Its upload commitment also binds
    the hash of its deal, so the on-chain upload hashes pin the commitments too.
  unmask (CLOSED phase, after screening)
    holders post their shares (secagg_reveal): b-shares of accepted clients, sk-shares of
    clients in Q that were rejected. Anyone checks each against the commitments: a wrong share
    is dropped and its holder named (bad_reveals). A holder never reveals both shares of one
    client (that would unmask its individual update): such posts are refused and also visible.
    The server needs t valid shares per secret; rebuilt sk_j is checked against pk_j.

Security assumptions as in Bonawitz et al. (CCS 2017), with t = n//2 + 1: the server plus the
corrupted clients hold fewer than t shares of any honest client's secret. C_0 = b_i * B is
public; b_i is a fresh uniform scalar, so it stays hidden under the discrete-log assumption.
Not yet covered here (Phase 4): a malicious server that shows different clients different
views of the round, and signatures on posts.
"""
import hashlib

import numpy as np

from secure import ristretto as rt
from secure import vss
from secure.channel import KeyPair
from secure.ristretto import L

from . import secaggField as sf

OPEN_KINDS = ("secagg_deal", "secagg_complaint", "secagg_open")
CLOSED_KINDS = ("secagg_reveal",)


def _ctx(rnd, i, j):
    return b"bcfl-secagg-share|" + int(rnd).to_bytes(8, "big") + int(i).to_bytes(4, "big") + int(j).to_bytes(4, "big")


def pair_key(sk_i, pk_i, pk_j):
    lo, hi = sorted([pk_i, pk_j])
    return hashlib.sha256(b"bcfl-pair|" + rt.mul(pk_j, sk_i) + lo + hi).digest()


def deal_digest(deal):
    """Hash of a client's public deal (pk and commitments; boxes excluded): bound into its upload hash."""
    h = hashlib.sha256(b"bcfl-secagg-deal|" + deal["pk"])
    for c in deal["C_sk"] + deal["C_b"]:
        h.update(c)
    return h.digest()


def upload_hash(rnd, i, y_field, deal):
    return hashlib.sha256(sf.upload_hash(rnd, i, y_field) + deal_digest(deal)).digest()


# ---------------------------------------------------------------------- public verification
def deals_on_board(board, rnd):
    out = {}
    for author, i, v in board.entries("secagg_deal", rnd):
        if author == i and i not in out:
            out[i] = v
    return out


def _well_formed(deal, n, t):
    try:
        pts = [deal["pk"]] + list(deal["C_sk"]) + list(deal["C_b"])
        return (len(deal["C_sk"]) == t and len(deal["C_b"]) == t and all(rt.is_valid_point(p) for p in pts)
                and deal["C_sk"][0] == deal["pk"] and set(deal["boxes"]) == set(range(n)))
    except (KeyError, TypeError):
        return False


def opened_shares(board, rnd, deals):
    """{(dealer, holder): (y_sk, y_b)} for public answers that verify."""
    out = {}
    for author, (i, j), v in board.entries("secagg_open", rnd):
        d = deals.get(i)
        if author != i or d is None or (i, j) in out:
            continue
        y_sk, y_b = v
        if vss.verify(j + 1, y_sk, d["C_sk"]) and vss.verify(j + 1, y_b, d["C_b"]):
            out[(i, j)] = (y_sk, y_b)
    return out


def complaints_on_board(board, rnd):
    out = {}
    for author, (j, i), _ in board.entries("secagg_complaint", rnd):
        if author == j:
            out.setdefault(i, set()).add(j)
    return out


def qualified_clients(board, rnd, n, t):
    """Clients whose deal is well formed, who answered every complaint with valid shares, and
    who drew fewer than t complaints. Computable by anyone from the board."""
    deals = deals_on_board(board, rnd)
    opened = opened_shares(board, rnd, deals)
    complaints = complaints_on_board(board, rnd)
    qual = []
    for i in range(n):
        d = deals.get(i)
        if d is None or not _well_formed(d, n, t):
            continue
        c = complaints.get(i, set())
        if len(c) >= t or any((i, j) not in opened for j in c):
            continue
        qual.append(i)
    return qual


def valid_reveals(board, rnd, deals, kind, client, allow_both=False):
    """{x: y} of verified shares of `kind` ('sk' or 'b') of `client`, one per holder, from holders
    that did not also reveal the other kind for that client (unless allow_both: a proven liar,
    Phase 5); and the holders whose share was wrong."""
    posts = board.entries("secagg_reveal", rnd)
    kinds_by_holder = {}
    for author, (j, k, i), _ in posts:
        if author == j and i == client:
            kinds_by_holder.setdefault(j, set()).add(k)
    d = deals[client]
    C = d["C_sk"] if kind == "sk" else d["C_b"]
    good, bad = {}, []
    for author, (j, k, i), y in posts:
        if author != j or i != client or k != kind or (j + 1) in good or j in bad:
            continue
        if len(kinds_by_holder.get(j, ())) > 1 and not allow_both:
            bad.append(j)                       # revealed both kinds: refused
            continue
        if vss.verify(j + 1, y, C):
            good[j + 1] = y
        else:
            bad.append(j)
    return good, bad


# ---------------------------------------------------------------------- the round
class VSSSecAggRound:
    """One round among clients 0..n-1 (each also a share holder). Client and holder steps are
    separate methods; everything that crosses between parties goes through the board or a box.

    faults (test attackers):
      bad_share         {(dealer, holder)}  dealer puts a wrong share in that holder's box
      bad_commitment    {dealer}            dealer's C_sk[0] is not its pk
      ignore_complaints {dealer}            dealer does not answer complaints
      false_complaint   {(holder, dealer)}  holder complains about a good box
      wrong_reveal      {holder}            holder posts wrong shares at unmasking
      double_reveal     {holder}            holder posts both kinds of shares for every client
    """

    def __init__(self, n, d, board, rnd, threshold=None, channel_keys=None, faults=None, identities=None):
        """identities (Phase 4): {i: secure.signing.Identity} of the clients. Every post is then
        signed, and every client reads its own view of the board (chainfl/relay.py)."""
        self.n, self.d, self.board, self.rnd = n, d, board, rnd
        self.ids = identities
        self.alarms = []                        # EquivocationErrors raised by clients that refused
        self.t = threshold or (n // 2 + 1)
        if not 2 <= self.t <= n:
            raise ValueError(f"threshold {self.t} must be in [2, {n}]")
        self.keys = channel_keys or {i: KeyPair() for i in range(n)}     # long-term box keys
        self.box_pk = {i: k.pk for i, k in self.keys.items()}
        self.faults = {k: set(v) for k, v in (faults or {}).items()}
        self._sk, self._b, self._shares = {}, {}, {}
        self.qualified = None
        self.bad_reveals = {}

    def _f(self, name):
        return self.faults.get(name, set())

    def _post(self, kind, author, key, value):
        sig = self.ids[author].sign_post(kind, self.rnd, author, key, value) if self.ids else None
        self.board.post(kind, self.rnd, author, key, value, sig)

    def _v(self, i):
        from chainfl.relay import view_of
        return view_of(self.board, ("client", i))

    def check_views(self, phase, readers=None):
        """Phase 4: each client compares its view with the anchored digest. Returns the clients
        that refuse (and records why); empty when all agree or when the board is a plain one."""
        from chainfl.relay import EquivocationError
        refused = []
        for i in (range(self.n) if readers is None else readers):
            v = self._v(i)
            if not hasattr(v, "check"):
                continue
            try:
                v.check(self.rnd, phase)
            except EquivocationError as e:
                self.alarms.append(e)
                refused.append(i)
        return refused

    def upload_sig(self, i, hsh):
        return self.ids[i].sign_post("upload_commit", self.rnd, i, i, hsh) if self.ids else None

    # ---------------------------------------------------------- OPEN phase
    def setup(self):
        self.deal()
        self.complain()
        self.answer()
        self.qualified = qualified_clients(self.board, self.rnd, self.n, self.t)
        return self.qualified

    def deal(self):
        xs = [j + 1 for j in range(self.n)]
        for i in range(self.n):
            sk, b = rt.random_scalar(), rt.random_scalar()
            sh_sk, C_sk = vss.deal(sk, self.t, xs)
            sh_b, C_b = vss.deal(b, self.t, xs)
            self._sk[i], self._b[i] = sk, b
            self._shares[i] = {j: (sh_sk[j + 1], sh_b[j + 1]) for j in range(self.n)}
            pk = rt.g_mul(sk)
            if i in self._f("bad_commitment"):
                C_sk = [rt.g_mul(rt.random_scalar())] + C_sk[1:]
            boxes = {}
            for j in range(self.n):
                y_sk, y_b = self._shares[i][j]
                if (i, j) in self._f("bad_share"):
                    y_sk, y_b = (y_sk + 1) % L, (y_b + 1) % L
                boxes[j] = self.keys[i].encrypt(self.box_pk[j], _ctx(self.rnd, i, j),
                                                rt.scalar_bytes(y_sk) + rt.scalar_bytes(y_b))
            self._post("secagg_deal", i, i, {"pk": pk, "C_sk": C_sk, "C_b": C_b, "boxes": boxes})

    def holder_share(self, j, i):
        """Holder j's verified (y_sk, y_b) for client i: from its box, else from i's public answer."""
        view = self._v(j)
        deals = deals_on_board(view, self.rnd)
        d = deals.get(i)
        if d is None:
            return None
        try:
            raw = self.keys[j].decrypt(self.box_pk[i], _ctx(self.rnd, i, j), d["boxes"][j])
            y_sk, y_b = rt.scalar_from_bytes(raw[:32]), rt.scalar_from_bytes(raw[32:])
            if vss.verify(j + 1, y_sk, d["C_sk"]) and vss.verify(j + 1, y_b, d["C_b"]):
                return y_sk, y_b
        except Exception:
            pass
        return opened_shares(view, self.rnd, deals).get((i, j))

    def complain(self):
        for j in range(self.n):
            deals = deals_on_board(self._v(j), self.rnd)
            for i in range(self.n):
                if i == j:
                    continue
                bad = i not in deals or self.holder_share(j, i) is None
                if bad or (j, i) in self._f("false_complaint"):
                    self._post("secagg_complaint", j, (j, i), None)

    def answer(self):
        for i in range(self.n):
            if i in self._f("ignore_complaints") or i not in self._shares:
                continue
            for author, (j, ii), _ in self._v(i).entries("secagg_complaint", self.rnd):
                if ii == i and author == j:
                    self._post("secagg_open", i, (i, j), self._shares[i][j])

    # ---------------------------------------------------------- client upload
    def deal_of(self, i):
        return deals_on_board(self.board, self.rnd)[i]

    def _mask(self, i):
        """Client i's mask, from ITS OWN view of the board (its own idea of who is in the round)."""
        if self.qualified is None:
            raise RuntimeError("setup() first")
        view = self._v(i)
        deals = deals_on_board(view, self.rnd)
        qual = qualified_clients(view, self.rnd, self.n, self.t)
        m = sf.prg_field(rt.scalar_bytes(self._b[i]), self.d, b"self")
        for j in qual:
            if j == i:
                continue
            p = sf.prg_field(pair_key(self._sk[i], deals[i]["pk"], deals[j]["pk"]), self.d, b"pair")
            m = (m + p) % L if j > i else (m - p) % L
        return m

    def client_upload_int(self, i, x_int):
        if i not in self.qualified:
            raise PermissionError(f"client {i} is not in the qualified set of round {self.rnd}")
        return (sf.to_field(x_int) + self._mask(i)) % L

    def upload_hash(self, i, y_field):
        return upload_hash(self.rnd, i, y_field, self.deal_of(i))

    # ---------------------------------------------------------- CLOSED phase
    def holders_reveal(self, accepted=None, holders=None, validate=None):
        """Each online holder posts its b-shares of accepted clients and sk-shares of the
        qualified clients that were rejected.

        Phase 4: a holder takes the accepted set from its own view of the board (the server's
        signed 'round_accepted' post), after checking the 'outcome' anchor and, if given,
        validate(j, view) (e.g. the committee's verdict attestations). A holder that finds a
        problem reveals nothing. Without a 'round_accepted' post (plain board) the accepted set
        passed in is used."""
        from chainfl.relay import EquivocationError
        holders = range(self.n) if holders is None else holders
        for j in holders:
            view = self._v(j)
            posted = [v for a, k, v in view.entries("round_accepted", self.rnd) if k == "accepted"]
            try:
                if hasattr(view, "check"):
                    view.check(self.rnd, "outcome")
                if validate is not None:
                    validate(j, view)
            except EquivocationError as e:
                self.alarms.append(e)
                continue
            if posted:
                acc = set(posted[0]["accepted"] if isinstance(posted[0], dict) else posted[0])
            else:
                acc = set(accepted or ())
            qual = qualified_clients(view, self.rnd, self.n, self.t)
            rejected = [i for i in qual if i not in acc]
            for kind, clients in (("b", sorted(acc)), ("sk", rejected)):
                for i in clients:
                    if i == j:
                        continue
                    sh = self.holder_share(j, i)
                    if sh is None:
                        continue
                    y = sh[0] if kind == "sk" else sh[1]
                    if j in self._f("wrong_reveal"):
                        y = (y + 1) % L
                    self._post("secagg_reveal", j, (j, kind, i), y)
                    if j in self._f("double_reveal"):
                        other = sh[1] if kind == "sk" else sh[0]
                        self._post("secagg_reveal", j, (j, "b" if kind == "sk" else "sk", i), other)

    def rebuild(self, kind, i, allow_both=False):
        """Rebuild client i's secret ('b' or 'sk') from the verified shares on the board (anyone can)."""
        deals = deals_on_board(self.board, self.rnd)
        good, bad = valid_reveals(self.board, self.rnd, deals, kind, i, allow_both)
        for j in bad:
            self.bad_reveals.setdefault(j, []).append((kind, i))
        if len(good) < self.t:
            raise RuntimeError(f"round {self.rnd}: only {len(good)} valid {kind}-shares of client {i}, "
                               f"need {self.t}")
        return vss.reconstruct(dict(list(good.items())[:self.t]))

    def holders_reveal_liars(self, liars, check=None, holders=None):
        """Phase 5: a client proven to have lied after it was accepted is taken out of the
        aggregate: each holder that confirms the proof itself (check(j, i)) also reveals its
        sk-share of that client."""
        holders = range(self.n) if holders is None else holders
        for j in holders:
            for i in liars:
                if i == j or (check is not None and not check(j, i)):
                    continue
                sh = self.holder_share(j, i)
                if sh is not None:
                    self._post("secagg_reveal", j, (j, "sk", i), sh[0])

    def server_unmask_sum_int(self, uploads, accepted, liars=()):
        """Unmask sum_{i in accepted} enc(x_i) from the uploads and the board's revealed shares.
        liars: clients removed after acceptance (Phase 5); both of their secrets may be on the board."""
        liars = set(liars)
        acc = sorted(set(accepted))
        if not set(acc) <= set(self.qualified):
            raise ValueError("accepted clients must be qualified")
        rejected = [i for i in self.qualified if i not in acc]
        deals = deals_on_board(self.board, self.rnd)
        self.bad_reveals = {}

        def rebuild(kind, i):
            good, bad = valid_reveals(self.board, self.rnd, deals, kind, i, allow_both=i in liars)
            for j in bad:
                self.bad_reveals.setdefault(j, []).append((kind, i))
            if len(good) < self.t:
                raise RuntimeError(f"round {self.rnd}: only {len(good)} valid {kind}-shares of client {i}, "
                                   f"need {self.t}")
            return vss.reconstruct(dict(list(good.items())[:self.t]))

        total = np.array([0] * self.d, dtype=object)
        for i in acc:
            total = (total + uploads[i]) % L
        for i in acc:
            total = (total - sf.prg_field(rt.scalar_bytes(rebuild("b", i)), self.d, b"self")) % L
        for j in rejected:
            sk_j = rebuild("sk", j)
            if rt.g_mul(sk_j) != deals[j]["pk"]:
                raise RuntimeError(f"round {self.rnd}: rebuilt key of client {j} does not match its pk")
            for i in acc:
                p = sf.prg_field(pair_key(sk_j, deals[j]["pk"], deals[i]["pk"]), self.d, b"pair")
                total = (total - p) % L if j > i else (total + p) % L
        return sf.from_field(total)
