"""Ristretto255 (prime-order group on Curve25519), as Python ints and bytes.

Group operations come from the Rust extension `bcfl_crypto` (curve25519-dalek). Build it once:
    pip install maturin && (cd crypto/bcfl_crypto && maturin develop --release)
Scalars are Python ints mod L; points are 32-byte canonical encodings.
"""
import hashlib
import secrets

try:
    import bcfl_crypto as _rc
except ImportError as e:  # pragma: no cover - message for a missing build
    raise ImportError(
        "the Rust extension bcfl_crypto is not built. Run:\n"
        "  pip install maturin\n"
        "  cd crypto/bcfl_crypto && maturin develop --release && cd ../..") from e

L = 2 ** 252 + 27742317777372353535851937790883648493   # group order (prime)
POINT_BYTES = 32


def scalar_bytes(x: int) -> bytes:
    return (x % L).to_bytes(32, "little")


def scalar_from_bytes(b: bytes) -> int:
    x = int.from_bytes(b, "little")
    if len(b) != 32 or x >= L:
        raise ValueError("not a canonical scalar")
    return x


def random_scalar() -> int:
    return secrets.randbelow(L)


def g_mul(x: int) -> bytes:
    """x * B (B = the standard basepoint)."""
    return _rc.basepoint_mul(scalar_bytes(x))


def mul(p: bytes, x: int) -> bytes:
    return _rc.point_mul(p, scalar_bytes(x))


def add(p: bytes, q: bytes) -> bytes:
    return _rc.point_add(p, q)


def sub(p: bytes, q: bytes) -> bytes:
    return _rc.point_sub(p, q)


def msm(scalars, points) -> bytes:
    """sum_i scalars[i] * points[i]."""
    return _rc.multiscalar_mul([scalar_bytes(s) for s in scalars], list(points))


def identity() -> bytes:
    return _rc.identity()


def hash_to_point(label: bytes) -> bytes:
    """A generator with unknown discrete log relative to B (for Pedersen commitments)."""
    return _rc.from_uniform_bytes(hashlib.sha512(b"bcfl-h2p|" + label).digest())


def is_valid_point(p: bytes) -> bool:
    return isinstance(p, (bytes, bytearray)) and len(p) == 32 and _rc.is_valid_point(bytes(p))
