"""Feldman verifiable secret sharing over Ristretto255 (Feldman 1987).

deal(secret, t, xs) -> (shares {x: y}, commitments [C_0 .. C_{t-1}])
    f(x) = secret + a_1 x + ... + a_{t-1} x^{t-1}  over Z_L,  share y = f(x),  C_k = a_k * B
verify(x, y, commitments)  checks  y * B == sum_k x^k * C_k
reconstruct({x: y})        Lagrange interpolation at 0 (needs t valid shares)

C_0 = secret * B is public: the secret must have enough entropy (here: a fresh random scalar).
"""
from . import ristretto as rt
from .ristretto import L


def deal(secret, t, xs, rng_scalar=rt.random_scalar):
    if t < 1:
        raise ValueError("threshold must be >= 1")
    if len(set(xs)) != len(xs) or any(not 0 < x < L for x in xs):
        raise ValueError("share x-coordinates must be distinct and non-zero")
    coeffs = [secret % L] + [rng_scalar() for _ in range(t - 1)]
    shares = {}
    for x in xs:
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % L
        shares[x] = y
    return shares, [rt.g_mul(c) for c in coeffs]


def verify(x, y, commitments):
    if not commitments or not all(rt.is_valid_point(c) for c in commitments):
        return False
    if not 0 <= y < L:
        return False
    powers = [pow(x, k, L) for k in range(len(commitments))]
    return rt.g_mul(y) == rt.msm(powers, commitments)


def reconstruct(shares):
    total = 0
    for xi, yi in shares.items():
        num, den = 1, 1
        for xj in shares:
            if xj != xi:
                num = num * (-xj) % L
                den = den * (xi - xj) % L
        total = (total + yi * num * pow(den, -1, L)) % L
    return total
