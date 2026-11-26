"""
C1 Phase 5: Pedersen commitments and the zero-knowledge proofs a client attaches to its upload.

Group: Ristretto255 (secure/ristretto.py), scalars mod L. The update x is a vector in Z_L^d
(the same field encoding the secure aggregation uses, secaggField.to_field).

Commitments
  vector  C_x = sum_j x_j G_j + r H          (G_1..G_d, G, H: hash-to-point, nobody knows a
  scalar  C_v = v G + r H                     discrete-log relation between any of them)

Update proof (one per client per round, Fiat-Shamir):
  public  C_x; C_mu (scalar commitment to a mask mu); C_q[c] for each class row c;
          a in Z_L^d; target T; rows = index sets of the class rows of the last layer
  claims  (1) C_x opens to some x            (2) <a, x> + mu = T  (mod L)
          (3) q_c = sum_{j in row c} x_j^2 for every c, where C_q[c] opens to q_c
  prover  w <- Z_L^d, omega, blinding scalars;  A = Com(w), A_mu = Com(omega),
          t0 = <a, w> + omega,  T1_c = Com(2 <x_c, w_c>),  T2_c = Com(<w_c, w_c>)
          e = H(statement, first messages)
          z = e x + w,  z_mu = e mu + omega,  matching blinding responses
  verify  Com(z; rho) == e C_x + A,  Com(z_mu; rho_mu) == e C_mu + A_mu,
          <a, z> + z_mu == e T + t0,
          Com(<z_c, z_c>; tau_c) == e^2 C_q[c] + e T1_c + T2_c
  This is a sigma protocol of degree 2 in e: three accepting transcripts with one first message
  give x, mu, q (special soundness); z is uniform given e, so it is honest-verifier ZK. The
  proof is O(d) scalars (no succinctness claimed).

DLEQ (Chaum-Pedersen): log_B(P) == log_Q(R), used to show that a revealed pairwise DH point
is the right one.
"""
import hashlib

import numpy as np

from secure import ristretto as rt
from secure.ristretto import L

_GENS = {}


def generators(d):
    """G_1..G_d, plus G (scalars) and H (blinding)."""
    if d not in _GENS:
        G = [rt.hash_to_point(b"bcfl-pedersen-G|" + j.to_bytes(4, "big")) for j in range(d)]
        g, H = rt.hash_to_point(b"bcfl-pedersen-g"), rt.hash_to_point(b"bcfl-pedersen-h")
        _GENS[d] = (G, g, H, rt.PointTable(G + [g, H]))
    return _GENS[d][:3]


def _table(d):
    generators(d)
    return _GENS[d][3]


def _ints(v):
    return [int(a) % L for a in v]


def vec_commit(x, r, d=None, secret=True):
    """sum_j x_j G_j + r H (constant time unless secret=False)."""
    n = len(x)
    d = d or n
    return _table(d).msm(_ints(x) + [r % L], list(range(n)) + [d + 1], secret)


def scalar_commit(v, r, secret=True):
    return _table(1).msm([v % L, r % L], [1, 2], secret)


def hash_scalar(*parts):
    h = hashlib.sha512(b"bcfl-zk|")
    for p in parts:
        b = p if isinstance(p, bytes) else str(p).encode()
        h.update(len(b).to_bytes(4, "big") + b)
    return int.from_bytes(h.digest(), "little") % L


def _dot(a, b):
    return sum(int(x) * int(y) for x, y in zip(a, b)) % L


def _pt_comb(coeffs, pts):
    return rt.msm([c % L for c in coeffs], pts)


# --------------------------------------------------------------------- update proof
def prove_update(x, r_x, mu, r_mu, q_r, a, T, rows, context=b""):
    """x: field ints (len d); r_x, mu, r_mu: scalars; q_r: blinding of each C_q[c];
    a: public coefficients (len d); T: public target with <a,x> + mu = T; rows: [index arrays].
    Returns (statement commitments are computed by the caller) the proof dict."""
    d = len(x)
    G, g, H = generators(d)
    x = _ints(x)
    w = [rt.random_scalar() for _ in range(d)]
    rho_w, omega, rho_om = rt.random_scalar(), rt.random_scalar(), rt.random_scalar()
    A = _table(d).msm(w + [rho_w], list(range(d)) + [d + 1], True)
    A_mu = scalar_commit(omega, rho_om)
    t0 = (_dot(a, w) + omega) % L
    T1, T2, tau1, tau2 = [], [], [], []
    for idx in rows:
        xc, wc = [x[j] for j in idx], [w[j] for j in idx]
        t1, t2 = rt.random_scalar(), rt.random_scalar()
        T1.append(scalar_commit(2 * _dot(xc, wc), t1))
        T2.append(scalar_commit(_dot(wc, wc), t2))
        tau1.append(t1)
        tau2.append(t2)
    e = hash_scalar(context, A, A_mu, t0, *T1, *T2)
    z = [(e * xi + wi) % L for xi, wi in zip(x, w)]
    return {"A": A, "A_mu": A_mu, "t0": t0, "T1": T1, "T2": T2, "z": z,
            "rho": (e * r_x + rho_w) % L, "z_mu": (e * mu + omega) % L, "rho_mu": (e * r_mu + rho_om) % L,
            "tau": [(e * e * qr + e * a1 + a2) % L for qr, a1, a2 in zip(q_r, tau1, tau2)]}


def verify_update(C_x, C_mu, C_q, a, T, rows, proof, context=b""):
    try:
        z = [int(v) % L for v in proof["z"]]
        d = len(z)
        if len(a) != d or len(C_q) != len(rows) or len(proof["T1"]) != len(rows) or len(proof["tau"]) != len(rows):
            return False
        G, g, H = generators(d)
        e = hash_scalar(context, proof["A"], proof["A_mu"], proof["t0"], *proof["T1"], *proof["T2"])
        if _table(d).msm(z + [proof["rho"]], list(range(d)) + [d + 1], False) != rt.add(rt.mul(C_x, e), proof["A"]):
            return False
        if scalar_commit(proof["z_mu"], proof["rho_mu"], False) != rt.add(rt.mul(C_mu, e), proof["A_mu"]):
            return False
        if (_dot(a, z) + proof["z_mu"]) % L != (e * (T % L) + proof["t0"]) % L:
            return False
        for c, idx in enumerate(rows):
            zc = [z[j] for j in idx]
            lhs = scalar_commit(_dot(zc, zc), proof["tau"][c], False)
            rhs = _pt_comb([e * e, e, 1], [C_q[c], proof["T1"][c], proof["T2"][c]])
            if lhs != rhs:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


# --------------------------------------------------------------------- DLEQ
def prove_dleq(x, B, P, Q, R, context=b""):
    """Knowledge of x with P = x*B and R = x*Q."""
    k = rt.random_scalar()
    A1, A2 = rt.mul(B, k), rt.mul(Q, k)
    e = hash_scalar(b"dleq", context, B, P, Q, R, A1, A2)
    return {"A1": A1, "A2": A2, "s": (k + e * x) % L}


def verify_dleq(B, P, Q, R, proof, context=b""):
    try:
        e = hash_scalar(b"dleq", context, B, P, Q, R, proof["A1"], proof["A2"])
        return (rt.mul(B, proof["s"]) == rt.add(proof["A1"], rt.mul(P, e))
                and rt.mul(Q, proof["s"]) == rt.add(proof["A2"], rt.mul(R, e)))
    except (KeyError, TypeError, ValueError):
        return False


def rows_layout(n_flat, w_off, n_cls, row_len, b_off):
    """Index sets of the class rows: row c = its weights w[c, :] and its bias b[c]."""
    return [np.concatenate([np.arange(w_off + c * row_len, w_off + (c + 1) * row_len), [b_off + c]]).tolist()
            for c in range(n_cls)]
