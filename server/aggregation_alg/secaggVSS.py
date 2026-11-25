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


def valid_reveals(board, rnd, deals, kind, client):
    """{x: y} of verified shares of `kind` ('sk' or 'b') of `client`, one per holder, from holders
    that did not also reveal the other kind for that client; and the holders whose share was wrong."""
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
        if len(kinds_by_holder.get(j, ())) > 1:
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

    def __init__(self, n, d, board, rnd, threshold=None, channel_keys=None, faults=None):
        self.n, self.d, self.board, self.rnd = n, d, board, rnd
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
            self.board.post("secagg_deal", self.rnd, i, i, {"pk": pk, "C_sk": C_sk, "C_b": C_b, "boxes": boxes})

    def holder_share(self, j, i):
        """Holder j's verified (y_sk, y_b) for client i: from its box, else from i's public answer."""
        deals = deals_on_board(self.board, self.rnd)
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
        return opened_shares(self.board, self.rnd, deals).get((i, j))

    def complain(self):
        deals = deals_on_board(self.board, self.rnd)
        for j in range(self.n):
            for i in range(self.n):
                if i == j:
                    continue
                bad = i not in deals or self.holder_share(j, i) is None
                if bad or (j, i) in self._f("false_complaint"):
                    self.board.post("secagg_complaint", self.rnd, j, (j, i), None)

    def answer(self):
        for _, (j, i), _ in self.board.entries("secagg_complaint", self.rnd):
            if i in self._f("ignore_complaints") or i not in self._shares:
                continue
            self.board.post("secagg_open", self.rnd, i, (i, j), self._shares[i][j])

    # ---------------------------------------------------------- client upload
    def deal_of(self, i):
        return deals_on_board(self.board, self.rnd)[i]

    def _mask(self, i):
        if self.qualified is None:
            raise RuntimeError("setup() first")
        deals = deals_on_board(self.board, self.rnd)
        m = sf.prg_field(rt.scalar_bytes(self._b[i]), self.d, b"self")
        for j in self.qualified:
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
    def holders_reveal(self, accepted, holders=None):
        """Each online holder posts its b-shares of accepted clients and sk-shares of the
        qualified clients that were rejected."""
        acc = set(accepted)
        rejected = [i for i in self.qualified if i not in acc]
        holders = range(self.n) if holders is None else holders
        for j in holders:
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
                    self.board.post("secagg_reveal", self.rnd, j, (j, kind, i), y)
                    if j in self._f("double_reveal"):
                        other = sh[1] if kind == "sk" else sh[0]
                        self.board.post("secagg_reveal", self.rnd, j, (j, "b" if kind == "sk" else "sk", i), other)

    def server_unmask_sum_int(self, uploads, accepted):
        """Unmask sum_{i in accepted} enc(x_i) from the uploads and the board's revealed shares."""
        acc = sorted(set(accepted))
        if not set(acc) <= set(self.qualified):
            raise ValueError("accepted clients must be qualified")
        rejected = [i for i in self.qualified if i not in acc]
        deals = deals_on_board(self.board, self.rnd)
        self.bad_reveals = {}

        def rebuild(kind, i):
            good, bad = valid_reveals(self.board, self.rnd, deals, kind, i)
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
