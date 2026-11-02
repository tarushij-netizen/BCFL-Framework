"""
Real secure aggregation for C1 (Bonawitz et al. 2017, honest-but-curious server), replacing
the float-mask simulation in sketchSecAgg.py.

Per round, every client i
  1. makes a Diffie-Hellman key pair (s_sk_i, s_pk_i) in the RFC 3526 2048-bit MODP group and a
     random self-mask seed b_i,
  2. Shamir-shares s_sk_i and b_i (threshold t) among the other clients,
  3. agrees a pair key with every other client j: k_ij = SHA-256(s_pk_j ^ s_sk_i mod p), which
     both sides compute without ever sending it,
  4. uploads   y_i = enc(x_i) + PRG(b_i) + sum_{j>i} PRG(k_ij) - sum_{j<i} PRG(k_ij)   (mod 2^64)
     where enc() is 24-bit fixed point and PRG is SHAKE-256.

Pair masks cancel in the sum. After screening, the server asks the share holders for
  - shares of b_i for every ACCEPTED client i (to strip its self mask), and
  - shares of s_sk_j for every REJECTED client j (to strip the pair masks between j and the
    accepted clients, since y_j is left out of the sum).
An honest holder never reveals both shares of the same client, so the server can always undo
the masks on the SUM of accepted uploads but never on any single upload, rejected ones included
(a rejected client's y_j still carries PRG(b_j)).

Screening rejection is a logical dropout: rejected clients are still online, so every client
can act as a share holder. Out of scope here (as in the semi-honest Bonawitz protocol): a
server that lies about who was rejected, clients that send wrong shares, and the encrypted,
authenticated channels the shares travel on in the real protocol (here they are handed over
directly).
"""
import hashlib
import secrets

import numpy as np

# RFC 3526, group 14 (2048-bit MODP), generator 2
P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
G = 2
SHAMIR_PRIME = 2 ** 521 - 1          # Mersenne prime, larger than any secret we share
FRAC_BITS = 24                       # fixed-point precision: 2^-24 ~ 6e-8 per coordinate


# ---------------------------------------------------------------------------- primitives
def dh_keygen():
    sk = secrets.randbits(256) | (1 << 255)        # 256-bit exponent: ~128-bit security in group 14
    return sk, pow(G, sk, P)


def dh_agree(sk_i, pk_j):
    if not 1 < pk_j < P - 1:
        raise ValueError("invalid DH public key")
    return hashlib.sha256(pow(pk_j, sk_i, P).to_bytes(256, "big")).digest()


def prg(seed: bytes, d, tag=b""):
    """d pseudo-random uint64 values from a 32-byte seed (SHAKE-256 as a stream cipher)."""
    return np.frombuffer(hashlib.shake_256(tag + seed).digest(8 * d), dtype=np.uint64).copy()


def encode(x):
    """float vector -> fixed point in Z_2^64 (two's complement)."""
    return np.round(np.asarray(x, dtype=np.float64) * (1 << FRAC_BITS)).astype(np.int64).view(np.uint64)


def decode(v):
    return v.view(np.int64).astype(np.float64) / (1 << FRAC_BITS)


def shamir_share(secret, t, xs):
    """Split integer `secret` into shares {x: f(x)}; any t of them rebuild it."""
    coeffs = [secret % SHAMIR_PRIME] + [secrets.randbelow(SHAMIR_PRIME) for _ in range(t - 1)]
    out = {}
    for x in xs:
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % SHAMIR_PRIME
        out[x] = y
    return out


def shamir_reconstruct(shares):
    """shares: {x: y} with at least t entries -> secret (Lagrange at 0)."""
    total = 0
    for xi, yi in shares.items():
        num, den = 1, 1
        for xj in shares:
            if xj != xi:
                num = num * (-xj) % SHAMIR_PRIME
                den = den * (xi - xj) % SHAMIR_PRIME
        total = (total + yi * num * pow(den, -1, SHAMIR_PRIME)) % SHAMIR_PRIME
    return total


def _seed_bytes(b_int):
    return b_int.to_bytes(32, "big")


# ---------------------------------------------------------------------------- one round
class DHSecAggRound:
    """One round of secure aggregation among clients 0..n-1 (share x-coordinate = index + 1)."""

    def __init__(self, n, d, threshold=None):
        self.n, self.d = n, d
        self.t = threshold or (n // 2 + 1)
        if not 2 <= self.t <= n:
            raise ValueError(f"threshold {self.t} must be in [2, {n}]")
        self._sk, self.pk, self._b = {}, {}, {}
        # holder j keeps shares of every other client's s_sk and b: self._held[j][('sk'|'b', i)]
        self._held = {j: {} for j in range(n)}
        self._revealed = {j: {} for j in range(n)}     # what each holder already gave out, per client
        xs = [j + 1 for j in range(n)]
        for i in range(n):
            self._sk[i], self.pk[i] = dh_keygen()
            self._b[i] = secrets.randbits(256)
            for kind, secret in (("sk", self._sk[i]), ("b", self._b[i])):
                for x, y in shamir_share(secret, self.t, xs).items():
                    self._held[x - 1][(kind, i)] = (x, y)
        self.bytes_per_client = 256 * 1 + 2 * (n - 1) * 66   # public key + outgoing shares (rough)

    # ----- client side
    def client_upload(self, i, x):
        """Client i masks its (already weighted) float update x. Uses only its own secrets and
        the other clients' public keys."""
        y = encode(x) + prg(_seed_bytes(self._b[i]), self.d, b"self")
        for j in range(self.n):
            if j == i:
                continue
            m = prg(dh_agree(self._sk[i], self.pk[j]), self.d, b"pair")
            y = y + m if j > i else y - m
        return y

    def holder_reveal(self, holder, kind, client):
        """A holder hands one share to the server, refusing to reveal both secrets of a client."""
        other = "sk" if kind == "b" else "b"
        if self._revealed[holder].get(client) == other:
            raise PermissionError(f"holder {holder} already revealed {other} of client {client}")
        self._revealed[holder][client] = kind
        return self._held[holder][(kind, client)]

    # ----- server side
    def server_unmask_sum(self, uploads, accepted, holders=None):
        """uploads: {i: y_i}; returns the float sum of accepted clients' updates."""
        acc = sorted(set(accepted))
        rejected = [j for j in range(self.n) if j not in acc]
        holders = list(range(self.n)) if holders is None else list(holders)
        if len(holders) < self.t:
            raise RuntimeError(f"only {len(holders)} share holders online, need {self.t}")
        total = np.zeros(self.d, dtype=np.uint64)
        for i in acc:
            total += uploads[i]
        for i in acc:                                    # strip self masks of accepted clients
            shares = dict(self.holder_reveal(h, "b", i) for h in holders if h != i)
            b_i = shamir_reconstruct(dict(list(shares.items())[:self.t]))
            total -= prg(_seed_bytes(b_i), self.d, b"self")
        for j in rejected:                               # strip pair masks with rejected clients
            shares = dict(self.holder_reveal(h, "sk", j) for h in holders if h != j)
            sk_j = shamir_reconstruct(dict(list(shares.items())[:self.t]))
            if pow(G, sk_j, P) != self.pk[j]:
                raise RuntimeError(f"reconstructed key of client {j} does not match its public key")
            for i in acc:
                m = prg(dh_agree(sk_j, self.pk[i]), self.d, b"pair")
                # y_i carried +m if j > i else -m
                total = total - m if j > i else total + m
        return decode(total)
