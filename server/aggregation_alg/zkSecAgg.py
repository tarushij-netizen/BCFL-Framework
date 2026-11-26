"""
Hardened C1, Phase 5: every client proves that what it reported for screening comes from the
update it actually uploaded, and a client that lies is named and taken out of the aggregate.

Gap (Phases 1-4): screening sees a client's sketch and class profile, the aggregate sees its
masked upload, and nothing ties the two. A client could report the sketch of a harmless update
and upload a poisoned one. The sum check (Phase 1) catches this only for the round as a whole:
the round is lost and nobody knows who lied.

What each client posts (all signed, Phase 4), with x = its update in Z_L^d (the field encoding
the masks use):
  at upload time ("zk_commit", in the 'uploads' anchor and bound into its upload hash)
    C_x                vector Pedersen commitment to x                       (secure/zk.py)
    C_q[c]             commitment to q_c = sum of squares of class row c of x (the profile input)
    C_mu1, C_mu2       commitments to two masks (MPC screening; zero with plaintext screening)
    Cp[j]              commitment to the pair mask p_ij = PRG(k_ij) it shares with client j, with
                       blinding r_ij = H(k_ij): client j computes the very same point
    s                  r_x + sum_{j>i} r_ij - sum_{j<i} r_ij
  after the coefficients c are fixed ("zk_proof")
    the proof of secure/zk.py: C_x opens to x, C_q opens to the row norms of x, and
    <a, x> + mu1 = T with a = S^T c[:k] (S = this round's CountSketch), where
      plaintext screening:  c = H(seed, round, client, sketch, q) in Z_L, mu1 = 0, T = <c, sketch>
      MPC screening:        c = 40-bit coefficients drawn inside the MPC after the inputs, and
                            T = V1 + kappa1 p, V1 = <c, sketch_input> + mu1_input (mod p) output
                            by the MPC; likewise sum_c c'_c C_q[c] + C_mu2 - (V2 + kappa2 p) G = R2 H
    So the sketch and q that screening used are the ones of the committed x (Schwartz-Zippel
    over the random c; for MPC 2^-40 per coefficient draw, with the inputs range-checked).

How x is tied to the upload y = x + PRG(b_i) + sum_j (+/-) p_ij:
  pairwise   Cp_ij posted by i must equal Cp_ji posted by j. A mismatch starts a dispute: each
             side reveals its DH point sk * pk_other with a DLEQ proof, anyone recomputes the
             true commitment, and the side whose Cp differs is a liar.
  relation   once b_i is revealed (accepted clients), anyone checks
             Com(y_i - PRG(b_i); s_i) == C_x + sum_{j>i} Cp_ij - sum_{j<i} Cp_ij
  audit      for every rejected client j (sk_j is rebuilt anyway), Cp_ij of each accepted i is
             checked against Com(PRG(k_ij); H(k_ij)).
  A client that fails the relation or the audit after it was accepted is a proven liar: the
  holders, having checked the same public evidence, also reveal its sk shares, so the server
  removes it from the aggregate and the round goes on. (A liar loses the privacy of its own
  upload; nobody else does.)

Norm bound: no range proof on x. The sketch coordinates are range-checked (MPC) and S is drawn
after x is committed, so a committed x of large norm gives a large sketch with overwhelming
probability and the norm-ratio screen rejects it (the CountSketch argument of Phase 1).
"""
import hashlib

import numpy as np

from secure import ristretto as rt
from secure import zk
from secure.ristretto import L
from secure.signing import canon

from . import secaggField as sf
from .secaggVSS import deals_on_board, pair_key, qualified_clients

K_BITS = 40


def signed(v):
    v = int(v) % L
    return v if v < L // 2 else v - L


def pair_r(k):
    return zk.hash_scalar(b"bcfl-pair-r", k)


def sketch_coeffs(h, s, c, d):
    """a = S^T c for the CountSketch (h, s): a_j = s_j * c_{h(j)}."""
    return [(int(s[j]) * int(c[int(h[j])])) % L for j in range(d)]


def commit_digest(post):
    return hashlib.sha256(b"bcfl-zk-commit|" + canon(post)).digest()


class ZKRound:
    """One round of Phase 5 for clients 0..n-1. Client-side secrets live in self._sec[i]
    (simulation); everything a verifier uses comes from the board or from the MPC outputs.

    faults (test attackers):
      commit_cover  {i}        commit to (and prove) the cover update it reports, upload another
      pair_lie      {(i, j)}   adjust Cp_ij so its relation still holds for the cover update
      pair_collude  {(i, j)}   i and j both commit the same fake pair mask
      refuse_dispute {i}       does not reveal its DH point in a dispute
      bad_proof     {i}        sends a proof for the wrong statement
    """

    def __init__(self, n, d, rows, board, rnd, frnd, identities=None, mpc_prime=None, faults=None):
        self.n, self.d, self.rows, self.board, self.rnd, self.frnd = n, d, rows, board, rnd, frnd
        self.ids, self.p = identities, mpc_prime
        self.faults = {k: set(v) for k, v in (faults or {}).items()}
        self._sec = {}
        self.liars = {}                 # client -> reason
        self.disputes = []

    def _f(self, name):
        return self.faults.get(name, set())

    def _post(self, kind, author, key, value):
        sig = self.ids[author].sign_post(kind, self.rnd, author, key, value) if self.ids else None
        self.board.post(kind, self.rnd, author, key, value, sig)

    def _v(self, i):
        from chainfl.relay import view_of
        return view_of(self.board, ("client", i))

    # ------------------------------------------------------------------ client: commit
    def q_of(self, x_ints):
        xs = [signed(v) for v in x_ints]
        return [sum(xs[j] * xs[j] for j in idx) for idx in self.rows]

    def pair_mask(self, i, j):
        deals = deals_on_board(self._v(i), self.rnd)
        k = pair_key(self.frnd._sk[i], deals[i]["pk"], deals[j]["pk"])
        return sf.prg_field(k, self.d, b"pair"), pair_r(k), k

    def commit(self, i, x_committed, x_uploaded, qualified):
        """Client i, before uploading. x_committed is what it proves things about (= x_uploaded for
        an honest client). Posts zk_commit and returns its digest (bound into the upload hash)."""
        x = [int(v) % L for v in sf.to_field(np.asarray(x_committed, dtype=np.int64))]
        q = self.q_of(x)
        r_x = rt.random_scalar()
        r_q = [rt.random_scalar() for _ in q]
        mpc = self.p is not None
        mu = [rt.random_scalar() % self.p if mpc else 0, rt.random_scalar() % self.p if mpc else 0]
        r_mu = [rt.random_scalar(), rt.random_scalar()] if mpc else [0, 0]
        Cp, s_i = {}, r_x
        shift = (sf.to_field(np.asarray(x_uploaded, dtype=np.int64)) - sf.to_field(np.asarray(x_committed, dtype=np.int64))) % L
        own_q = qualified_clients(self._v(i), self.rnd, self.n, self.frnd.t)   # its own view, like its mask
        for j in own_q:
            if j == i:
                continue
            p_ij, r_ij, _ = self.pair_mask(i, j)
            if (i, j) in self._f("pair_lie"):
                # put the difference between the upload and the committed update into this pair
                p_ij = (p_ij + shift) % L if j > i else (p_ij - shift) % L
            if (i, j) in self._f("pair_collude") or (j, i) in self._f("pair_collude"):
                lo, hi = min(i, j), max(i, j)
                fake = sf.prg_field(b"collude|%d|%d|%d" % (self.rnd, lo, hi), self.d, b"pair")
                p_ij = fake
            Cp[j] = zk.vec_commit(p_ij, r_ij, self.d)
            s_i = (s_i + r_ij) % L if j > i else (s_i - r_ij) % L
        post = {"C_x": zk.vec_commit(x, r_x, self.d),
                "C_q": [zk.scalar_commit(qc, r) for qc, r in zip(q, r_q)],
                "C_mu": [zk.scalar_commit(m, r) for m, r in zip(mu, r_mu)],
                "Cp": Cp, "s": s_i}
        self._sec[i] = {"x": x, "r_x": r_x, "q": q, "r_q": r_q, "mu": mu, "r_mu": r_mu}
        self._post("zk_commit", i, i, post)
        return commit_digest(post)

    def mpc_masks(self, i):
        return self._sec[i]["mu"]

    def committed_q(self, i):
        return self._sec[i]["q"]

    # ------------------------------------------------------------------ public: read posts
    def commits(self, reader="server"):
        view = self._v(reader) if reader != "server" else self.board
        return {k: v for a, k, v in view.entries("zk_commit", self.rnd) if a == k}

    def proofs(self, reader="server"):
        view = self._v(reader) if reader != "server" else self.board
        return {k: v for a, k, v in view.entries("zk_proof", self.rnd) if a == k}

    # ------------------------------------------------------------------ pairwise check + disputes
    def pair_check(self, qualified):
        """After uploads are fixed: Cp_ij == Cp_ji for every pair; disputes resolved by DLEQ reveals.
        Returns the liars found (also recorded in self.liars)."""
        com = self.commits()
        deals = deals_on_board(self.board, self.rnd)
        found = set()
        for a_ in qualified:
            for b_ in qualified:
                if b_ <= a_:
                    continue
                ca, cb = com.get(a_), com.get(b_)
                if ca is None or cb is None:
                    continue
                if ca["Cp"].get(b_) == cb["Cp"].get(a_):
                    continue
                self.disputes.append((a_, b_))
                for who, other in ((a_, b_), (b_, a_)):        # each side may reveal its DH point
                    if who in self._f("refuse_dispute"):
                        continue
                    K = rt.mul(deals[other]["pk"], self.frnd._sk[who])
                    pf = zk.prove_dleq(self.frnd._sk[who], _B(), deals[who]["pk"],
                                       deals[other]["pk"], K, b"bcfl-dispute|%d" % self.rnd)
                    self._post("zk_dispute", who, (who, other), {"K": K, "dleq": pf})
                truth = None
                for a2, (who, other), v in self.board.entries("zk_dispute", self.rnd):
                    if a2 != who or {who, other} != {a_, b_}:
                        continue
                    if zk.verify_dleq(_B(), deals[who]["pk"], deals[other]["pk"], v["K"], v["dleq"],
                                      b"bcfl-dispute|%d" % self.rnd):
                        lo_pk, hi_pk = sorted([deals[a_]["pk"], deals[b_]["pk"]])
                        k = hashlib.sha256(b"bcfl-pair|" + v["K"] + lo_pk + hi_pk).digest()
                        truth = zk.vec_commit(sf.prg_field(k, self.d, b"pair"), pair_r(k), self.d, secret=False)
                        break
                for who, other in ((a_, b_), (b_, a_)):
                    if truth is None or com[who]["Cp"].get(other) != truth:
                        found.add(who)
                        self.liars.setdefault(who, f"pair commitment with {other} is wrong")
        return sorted(found)

    # ------------------------------------------------------------------ client: prove
    def prove_plain(self, i, sketch, seed_hs):
        """Plaintext screening: the server receives sketch and q in the clear (as before)."""
        h, s = seed_hs
        c = self.plain_coeffs(i, sketch)
        a = sketch_coeffs(h, s, c, self.d)
        T = sum(int(ci) * int(si) for ci, si in zip(c, sketch)) % L
        sec = self._sec[i]
        ctx = self._ctx(i, T)
        pf = zk.prove_update(sec["x"], sec["r_x"], 0, 0, sec["r_q"], a, T, self.rows, ctx)
        if i in self._f("bad_proof"):
            pf["t0"] = (pf["t0"] + 1) % L
        self._post("zk_proof", i, i, {"proof": pf})
        return pf

    def plain_coeffs(self, i, sketch):
        seed = self._seed
        return [zk.hash_scalar(b"bcfl-zk-c", seed, self.rnd, i, canon([int(v) for v in sketch]), t)
                for t in range(len(sketch))]

    def prove_mpc(self, i, sketch, outs):
        """MPC screening: outs = the MPC's {coef, V1, V2} (as every observer received them)."""
        h, s = self._hs
        k = len(sketch)
        c, c2 = outs["coef"][:k], outs["coef"][k:]
        sec = self._sec[i]
        p = self.p
        raw1 = sum(int(ci) * int(si) for ci, si in zip(c, sketch)) + sec["mu"][0]
        raw2 = sum(int(ci) * int(qi) for ci, qi in zip(c2, sec["q"])) + sec["mu"][1]
        kap1, kap2 = (raw1 - outs["V1"][i]) // p, (raw2 - outs["V2"][i]) // p
        T = (outs["V1"][i] + kap1 * p) % L
        a = sketch_coeffs(h, s, c, self.d)
        ctx = self._ctx(i, T)
        pf = zk.prove_update(sec["x"], sec["r_x"], sec["mu"][0], sec["r_mu"][0], sec["r_q"], a, T, self.rows, ctx)
        if i in self._f("bad_proof"):
            pf["t0"] = (pf["t0"] + 1) % L
        R2 = (sum(int(ci) * r for ci, r in zip(c2, sec["r_q"])) + sec["r_mu"][1]) % L
        self._post("zk_proof", i, i, {"proof": pf, "kappa": [kap1, kap2], "R2": R2})
        return pf

    def _ctx(self, i, T):
        com = self.commits().get(i)
        return b"bcfl-zk-ctx|" + canon([self.rnd, i, commit_digest(com) if com else b"", int(T) % L])

    # ------------------------------------------------------------------ verify proofs
    def verify_plain(self, i, sketch, q, r_q):
        """Server side, plaintext screening: the client also handed over q and its openings."""
        com, prf = self.commits().get(i), self.proofs().get(i)
        if com is None or prf is None:
            return False
        if any(zk.scalar_commit(qc, r) != C for qc, r, C in zip(q, r_q, com["C_q"])):
            return False
        if com["C_mu"][0] != rt.identity():
            return False
        h, s = self._hs
        c = self.plain_coeffs(i, sketch)
        T = sum(int(ci) * int(si) for ci, si in zip(c, sketch)) % L
        a = sketch_coeffs(h, s, c, self.d)
        return zk.verify_update(com["C_x"], com["C_mu"][0], com["C_q"], a, T, self.rows, prf["proof"],
                                self._ctx(i, T))

    def verify_mpc(self, i, outs, k, reader="server"):
        """Anyone with the (attested) MPC outputs can run this."""
        com, prf = self.commits(reader).get(i), self.proofs(reader).get(i)
        if com is None or prf is None or not outs["in_range"][i]:
            return False
        kap1, kap2 = prf.get("kappa", [None, None])
        if kap1 not in (-1, 0, 1) or kap2 not in (-1, 0, 1):
            return False
        p = self.p
        c, c2 = outs["coef"][:k], outs["coef"][k:]
        T = (outs["V1"][i] + kap1 * p) % L
        h, s = self._hs
        a = sketch_coeffs(h, s, c, self.d)
        if not zk.verify_update(com["C_x"], com["C_mu"][0], com["C_q"], a, T, self.rows, prf["proof"],
                                self._ctx(i, T)):
            return False
        _, g, H = zk.generators(1)
        lhs = rt.msm([int(ci) % L for ci in c2] + [1, (-(outs["V2"][i] + kap2 * p)) % L],
                     list(com["C_q"]) + [com["C_mu"][1], g])
        return lhs == rt.mul(H, prf["R2"])

    def set_round_params(self, seed, hs):
        self._seed, self._hs = seed, hs

    # ------------------------------------------------------------------ binding to the upload
    def relation_ok(self, i, y_i, b_i, qualified):
        """Com(y_i - PRG(b_i); s_i) == C_x + sum_{j>i} Cp_ij - sum_{j<i} Cp_ij."""
        com = self.commits().get(i)
        if com is None:
            return False
        lhs = zk.vec_commit((np.asarray(y_i, dtype=object) - sf.prg_field(rt.scalar_bytes(b_i), self.d, b"self")) % L,
                            com["s"], self.d, secret=False)
        rhs = com["C_x"]
        for j in qualified:
            if j == i:
                continue
            P = com["Cp"].get(j)
            if P is None:
                return False
            rhs = rt.add(rhs, P) if j > i else rt.sub(rhs, P)
        return lhs == rhs

    def audit_ok(self, i, j, sk_j):
        """Cp_ij of client i against the pair mask rebuilt from rejected client j's key."""
        deals = deals_on_board(self.board, self.rnd)
        com = self.commits().get(i)
        k = pair_key(sk_j, deals[j]["pk"], deals[i]["pk"])
        return com is not None and com["Cp"].get(j) == zk.vec_commit(sf.prg_field(k, self.d, b"pair"), pair_r(k),
                                                                      self.d, secret=False)


def _B():
    return rt.g_mul(1)
