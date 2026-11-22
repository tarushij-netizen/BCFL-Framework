"""Cryptographic building blocks for the hardened C1 protocol (Phase 1 onwards).

ristretto : Ristretto255 group operations (Rust extension bcfl_crypto, curve25519-dalek)
vss       : Feldman verifiable secret sharing over Ristretto255
channel   : authenticated encryption between two parties (X25519 + ChaCha20-Poly1305)
"""
