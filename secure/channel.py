"""Authenticated encryption between two long-term key holders: X25519 key agreement, a key per
(pair, context) derived with SHA-256, and ChaCha20-Poly1305 (random 96-bit nonce, context as
associated data). All primitives come from the `cryptography` package.
"""
import hashlib
import os

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


class KeyPair:
    def __init__(self):
        self._sk = X25519PrivateKey.generate()
        self.pk = self._sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def _key(self, peer_pk: bytes, context: bytes) -> bytes:
        shared = self._sk.exchange(X25519PublicKey.from_public_bytes(peer_pk))
        lo, hi = sorted([self.pk, peer_pk])
        return hashlib.sha256(b"bcfl-channel|" + shared + lo + hi + context).digest()

    def encrypt(self, peer_pk: bytes, context: bytes, plaintext: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + ChaCha20Poly1305(self._key(peer_pk, context)).encrypt(nonce, plaintext, context)

    def decrypt(self, peer_pk: bytes, context: bytes, box: bytes) -> bytes:
        """Raises cryptography.exceptions.InvalidTag if the box was altered or misaddressed."""
        return ChaCha20Poly1305(self._key(peer_pk, context)).decrypt(box[:12], box[12:], context)
