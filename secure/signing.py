"""Signatures on board posts (C1 Phase 4): Ed25519 identities and a canonical encoding.

Every party (client, committee member, server) has a long-term Ed25519 key whose public half is
registered once (on chain in a deployment: the registry cannot be shown differently to different
readers). A post is (kind, round, author, key, value); its signature covers the canonical
encoding of all five, so a relay (the server) can neither forge a post nor change one.
"""
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canon(obj):
    """Deterministic, type-tagged bytes for the values used on the board."""
    import numpy as np
    if obj is None:
        return b"N"
    if isinstance(obj, (bool, np.bool_)):
        return b"T" if obj else b"F"
    if isinstance(obj, (int, np.integer)):
        v = int(obj)
        raw = v.to_bytes((v.bit_length() + 8) // 8 or 1, "big", signed=True)
        return b"I" + len(raw).to_bytes(4, "big") + raw
    if isinstance(obj, (bytes, bytearray)):
        return b"B" + len(obj).to_bytes(4, "big") + bytes(obj)
    if isinstance(obj, str):
        raw = obj.encode()
        return b"S" + len(raw).to_bytes(4, "big") + raw
    if isinstance(obj, np.ndarray):
        return canon([int(x) for x in obj.ravel()])
    if isinstance(obj, (list, tuple)):
        parts = [canon(x) for x in obj]
        return b"L" + len(parts).to_bytes(4, "big") + b"".join(parts)
    if isinstance(obj, dict):
        items = sorted((canon(k), canon(v)) for k, v in obj.items())
        return b"D" + len(items).to_bytes(4, "big") + b"".join(k + v for k, v in items)
    raise TypeError(f"cannot encode {type(obj).__name__} on the board")


def post_message(kind, rnd, author, key, value):
    return b"bcfl-post|" + canon([kind, int(rnd), author, key, value])


class Identity:
    def __init__(self):
        self._sk = Ed25519PrivateKey.generate()
        self.pk = self._sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def sign(self, msg):
        return self._sk.sign(msg)

    def sign_post(self, kind, rnd, author, key, value):
        return self.sign(post_message(kind, rnd, author, key, value))


def verify(pk, msg, sig):
    if pk is None or sig is None:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def verify_post(pk, kind, rnd, author, key, value, sig):
    return verify(pk, post_message(kind, rnd, author, key, value), sig)
