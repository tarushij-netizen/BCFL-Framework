"""
Hardened C1, Phase 1: secure aggregation over the Ristretto255 scalar field, commit-first
uploads, a per-round sketch drawn from an unpredictable seed, and the sum-consistency check.

Masking (same structure as secaggDH.py, Bonawitz et al. 2017, but in Z_L instead of Z_2^64):
    y_i = enc(x_i) + PRG(b_i) + sum_{j>i} PRG(k_ij) - sum_{j<i} PRG(k_ij)   (mod L)
L is the Ristretto255 group order, so later phases can commit to enc(x_i) with Pedersen
commitments and prove statements about y_i in the same field. enc() is 24-bit fixed point;
PRG(seed) gives each coordinate from 64 bytes of SHAKE-256 reduced mod L (bias < 2^-259).

Commit first: a client's upload y_i is fixed (and its hash posted) BEFORE anyone knows this
round's sketch seed. The masks are fixed too: pair keys come from the DH keys published at
setup, and b_i is pinned by the Shamir shares the client handed out at setup.

Sketch: CountSketch S_seed, coordinate c -> bucket h[c] with sign s[c], both derived from the
round's beacon seed with SHAKE-256. Clients report sk_i = S_seed enc(x_i) as exact integers.
Because the poison cannot be fitted to a matrix that is unknown when y_i is fixed, a
"hidden direction" (S delta = 0) planned in advance shows up in the real sketch (gap 2).

Sum check (gap 3): CountSketch is linear over the integers, so for the accepted set A
    S_seed( sum_{i in A} enc(x_i) ) == sum_{i in A} sk_i      exactly.
The server unmasks sum_{i in A} enc(x_i) anyway, so it can test this equality. A client that
reports the sketch of a benign update while masking a poisoned one breaks it. The check shows
THAT someone in A lied, not WHO; naming the client needs the Phase 5 proofs.
"""
import hashlib
import secrets

import numpy as np

from secure.ristretto import L
from .secaggDH import _seed_bytes, dh_agree, dh_keygen, shamir_reconstruct, shamir_share, G, P

FRAC_BITS = 24
HALF = L // 2


# ---------------------------------------------------------------------------- field vectors
def encode_int(x):
    """float vector -> signed fixed-point integers (exact, int64)."""
    return np.round(np.asarray(x, dtype=np.float64) * (1 << FRAC_BITS)).astype(np.int64)


def to_field(v_int):
    return np.array([int(a) % L for a in v_int], dtype=object)


def from_field(v_field):
    """field vector -> signed integers (values are known to be small: |v| < L/2)."""
    return np.array([int(a) if a < HALF else int(a) - L for a in v_field], dtype=np.int64)


def decode_int(v_int):
    return np.asarray(v_int, dtype=np.float64) / (1 << FRAC_BITS)


def prg_field(seed: bytes, d, tag=b""):
    buf = hashlib.shake_256(b"bcfl-prg-field|" + tag + seed).digest(64 * d)
    return np.array([int.from_bytes(buf[64 * c:64 * c + 64], "little") % L for c in range(d)], dtype=object)


def upload_hash(rnd, i, y_field):
    """Commitment to a masked upload (binding; the upload itself is already hiding)."""
    h = hashlib.sha256(b"bcfl-upload|" + int(rnd).to_bytes(8, "big") + int(i).to_bytes(4, "big"))
    for a in y_field:
        h.update(int(a).to_bytes(32, "little"))
    return h.digest()


# ---------------------------------------------------------------------------- sketch
def sketch_params_from_seed(seed: bytes, d, k):
    """CountSketch from the round seed: bucket h[c] in [0, k), sign s[c] in {-1, +1}."""
    hb = np.frombuffer(hashlib.shake_256(b"bcfl-sketch-h|" + seed).digest(4 * d), dtype="<u4")
    h = (hb % np.uint32(k)).astype(np.int64)        # k = 64: a power of two, so no modulo bias
    bits = np.unpackbits(np.frombuffer(hashlib.shake_256(b"bcfl-sketch-s|" + seed).digest((d + 7) // 8),
                                       dtype=np.uint8))[:d]
    s = np.where(bits == 1, 1, -1).astype(np.int64)
    return h, s


def sketch_int(v_int, h, s, k):
    """Exact integer CountSketch of an integer vector."""
    out = np.zeros(k, dtype=np.int64)
    np.add.at(out, h, s * np.asarray(v_int, dtype=np.int64))
    return out


def null_space_project(z, h, s, k):
    """Remove from z everything a CountSketch (h, s) can see: the result r has S r = 0.
    Used by the adaptive test attacker of gap 2 (S S^T is diagonal, so this is exact)."""
    z = np.asarray(z, dtype=np.float64)
    seen = np.bincount(h, weights=s * z, minlength=k)
    size = np.bincount(h, minlength=k).astype(np.float64)
    size[size == 0] = 1.0
    return z - s * (seen / size)[h]


# ---------------------------------------------------------------------------- one round
class FieldSecAggRound:
    """One round among clients 0..n-1; share x-coordinate of client j is j + 1."""

    def __init__(self, n, d, threshold=None):
        self.n, self.d = n, d
        self.t = threshold or (n // 2 + 1)
        if not 2 <= self.t <= n:
            raise ValueError(f"threshold {self.t} must be in [2, {n}]")
        self._sk, self.pk, self._b = {}, {}, {}
        self._held = {j: {} for j in range(n)}
        self._revealed = {j: {} for j in range(n)}
        xs = [j + 1 for j in range(n)]
        for i in range(n):
            self._sk[i], self.pk[i] = dh_keygen()
            self._b[i] = secrets.randbits(256)
            for kind, secret in (("sk", self._sk[i]), ("b", self._b[i])):
                for x, y in shamir_share(secret, self.t, xs).items():
                    self._held[x - 1][(kind, i)] = (x, y)

    def _mask(self, i):
        m = prg_field(_seed_bytes(self._b[i]), self.d, b"self")
        for j in range(self.n):
            if j == i:
                continue
            p = prg_field(dh_agree(self._sk[i], self.pk[j]), self.d, b"pair")
            m = (m + p) % L if j > i else (m - p) % L
        return m

    def client_upload_int(self, i, x_int):
        """Client i masks its signed fixed-point update x_int (int64 vector)."""
        return (to_field(x_int) + self._mask(i)) % L

    def holder_reveal(self, holder, kind, client):
        other = "sk" if kind == "b" else "b"
        if self._revealed[holder].get(client) == other:
            raise PermissionError(f"holder {holder} already revealed {other} of client {client}")
        self._revealed[holder][client] = kind
        return self._held[holder][(kind, client)]

    def server_unmask_sum_int(self, uploads, accepted, holders=None):
        """uploads: {i: y_i}; returns sum_{i in accepted} enc(x_i) as signed integers."""
        acc = sorted(set(accepted))
        rejected = [j for j in range(self.n) if j not in acc]
        holders = list(range(self.n)) if holders is None else list(holders)
        if len(holders) < self.t:
            raise RuntimeError(f"only {len(holders)} share holders online, need {self.t}")
        total = np.array([0] * self.d, dtype=object)
        for i in acc:
            total = (total + uploads[i]) % L
        for i in acc:
            shares = dict(self.holder_reveal(h, "b", i) for h in holders if h != i)
            b_i = shamir_reconstruct(dict(list(shares.items())[:self.t]))
            total = (total - prg_field(_seed_bytes(b_i), self.d, b"self")) % L
        for j in rejected:
            shares = dict(self.holder_reveal(h, "sk", j) for h in holders if h != j)
            sk_j = shamir_reconstruct(dict(list(shares.items())[:self.t]))
            if pow(G, sk_j, P) != self.pk[j]:
                raise RuntimeError(f"reconstructed key of client {j} does not match its public key")
            for i in acc:
                p = prg_field(dh_agree(sk_j, self.pk[i]), self.d, b"pair")
                total = (total - p) % L if j > i else (total + p) % L
        return from_field(total)


def sum_check(sum_int, sketches, accepted, h, s, k):
    """True iff the sketch of the unmasked sum equals the sum of the accepted clients' sketches."""
    expected = np.zeros(k, dtype=np.int64)
    for i in accepted:
        expected += np.asarray(sketches[i], dtype=np.int64)
    return bool(np.array_equal(sketch_int(sum_int, h, s, k), expected))
