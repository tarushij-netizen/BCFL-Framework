"""
C1 prototype: screening poisoned updates WITHOUT seeing them.

Each client i sends two things instead of its plaintext update u_i:
  1. masked_i = u_i + sum_{j>i} m_ij - sum_{j<i} m_ji     (pairwise-mask secure aggregation,
     Bonawitz et al. 2017). Masks cancel in a sum, so the server learns only sums.
  2. sketch_i = S u_i, a k-dimensional CountSketch (k << d; S is a public random
     sign-hash matrix that changes every round).

The server screens on sketches only (cosine to the coordinate-wise median sketch + norm
ratio, the same rule as similarityAggregator). Rejected clients are treated as dropouts:
the server removes their pairwise masks (in the real protocol the surviving clients reveal
secret shares of those pair seeds) and unmasks ONLY the sum of accepted updates.

What this simulation does and does not do
- The code is split so the server-side function only ever receives (masked, sketch) pairs.
- Masks are real-valued Gaussian vectors, not field elements, and seed recovery is
  simulated (the server is handed the rejected clients' pair seeds). This is enough to
  measure accuracy and detection; it is not a cryptographic implementation.
- The sketch itself leaks some information; inference_attacks.py --sketch-dims measures it.
"""
import logging
from collections import OrderedDict

import numpy as np

from .similarityAgg import (_PoolMixin, _flatten, _float_keys, _to_numpy, cosine_similarity,
                            snapshot_state_dict)
from ..base.baseAggregator import ServerAggregator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- sketch
def count_sketch_params(d, k, seed):
    """Public CountSketch: coordinate c goes to bucket h[c] with sign s[c]."""
    r = np.random.default_rng(seed)
    return r.integers(0, k, size=d), r.choice(np.array([-1.0, 1.0]), size=d)


def count_sketch(v, h, s, k):
    return np.bincount(h, weights=s * v, minlength=k)


# ---------------------------------------------------------------------------- masks
def _pair_mask(seed, i, j, d, scale):
    return np.random.default_rng([seed, i, j]).normal(0.0, scale, size=d)


def client_package(u, i, n, round_seed, h, s, k, mask_scale=100.0):
    """Client side: returns (masked update, sketch). u is this client's flat update."""
    masked = u.astype(np.float64).copy()
    for j in range(n):
        if j > i:
            masked += _pair_mask(round_seed, i, j, len(u), mask_scale)
        elif j < i:
            masked -= _pair_mask(round_seed, j, i, len(u), mask_scale)
    return masked, count_sketch(u, h, s, k)


def server_unmask_sum(masked, accepted, n, round_seed, d, mask_scale=100.0):
    """Server side: sum of accepted clients' updates. Rejected clients are dropouts;
    their masks with accepted clients are rebuilt from (simulated) recovered pair seeds."""
    acc = set(accepted)
    total = np.sum([masked[i] for i in accepted], axis=0)
    for i in accepted:
        for j in range(n):
            if j in acc or j == i:
                continue
            if j > i:
                total -= _pair_mask(round_seed, i, j, d, mask_scale)
            else:
                total += _pair_mask(round_seed, j, i, d, mask_scale)
    return total


def screen(vectors, similarity_threshold, magnitude_ratio):
    """Same rule as similarityAggregator, applied to any set of vectors (here: sketches)."""
    V = np.stack(vectors)
    consensus = np.median(V, axis=0)
    norms = np.linalg.norm(V, axis=1)
    med = float(np.median(norms))
    sims = [cosine_similarity(v, consensus) for v in V]
    ratios = [float(x / med) if med > 0 else 0.0 for x in norms]
    accepted = [i for i in range(len(V)) if sims[i] >= similarity_threshold and ratios[i] <= magnitude_ratio]
    return accepted, sims, ratios


# ---------------------------------------------------------------------------- reputation (C3)
class Reputation:
    """Per-client reputation from screening verdicts, asymmetric (trust is lost faster
    than it is earned):
        accepted:  r <- beta * r + (1 - beta)
        rejected:  r <- penalty * r
    A client takes part in aggregation only if r >= tau, and is weighted by r.
    With beta 0.7, penalty 0.5, tau 0.6: a client rejected every other round is kept out
    from its second round on (r 0.75 -> 0.375 -> 0.56 -> ...), while an honest client with
    one false rejection sits out one round and is back after one accepted round.
    New identities start at `init` (Sybil handling: set it below tau so they must earn trust)."""

    def __init__(self, init=0.75, beta=0.7, tau=0.6, penalty=0.5):
        self.init, self.beta, self.tau, self.penalty = init, beta, tau, penalty
        self.r = {}

    def get(self, i):
        return self.r.get(i, self.init)

    def update(self, verdicts):
        for i, v in verdicts.items():
            r = self.get(i)
            self.r[i] = self.beta * r + (1 - self.beta) if v else self.penalty * r

    def eligible(self, ids):
        return [i for i in ids if self.get(i) >= self.tau]


# ---------------------------------------------------------------------------- aggregator
class sketchSecAggAggregator(_PoolMixin, ServerAggregator):
    def __init__(self, sketch_dim=64, similarity_threshold=0.1, magnitude_ratio=3.0,
                 mask_scale=100.0, seed=0, reputation=False, rep_init=0.75, rep_beta=0.7, rep_tau=0.6,
                 rep_penalty=0.5, class_check=False, class_mad=3.0, model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        # Backdoor signal: each client also reports its "class profile" = the share of its
        # output-layer update that falls on each class row (C numbers, e.g. 10). A backdoor
        # drags the model toward one target class, so the attacker's profile is lopsided.
        # Clients whose profile is far from the median profile (robust z-score > class_mad)
        # are rejected. Privacy cost: the profile hints at the client's label mix.
        self.class_check = class_check
        self.class_mad = class_mad
        self.reputation = Reputation(rep_init, rep_beta, rep_tau, rep_penalty) if reputation else None
        self.last_reputation = []
        self.last_head_rejected = []
        self.k = sketch_dim
        self.similarity_threshold = similarity_threshold
        self.magnitude_ratio = magnitude_ratio
        self.mask_scale = mask_scale
        self.seed = seed
        self.reference_model = None
        self.round = 0
        self.rejected_last_round = []
        self.last_similarities = []
        self.last_norm_ratios = []

    # ----- the only function that plays the server role
    def _server_round(self, packages, d, round_seed, n, weights=None):
        """weights: public per-client weights the clients already multiplied into their
        update (reputation from earlier rounds); None = plain average."""
        sketches = [p[1] for p in packages]
        accepted, sims, ratios = screen(sketches, self.similarity_threshold, self.magnitude_ratio)
        self.last_head_rejected = []
        if self.class_check and packages[0][2] is not None:
            P = np.stack([p[2] for p in packages])
            med = np.median(P, axis=0)
            dist = np.abs(P - med).sum(axis=1)                    # L1 distance to the median profile
            mad = np.median(np.abs(dist - np.median(dist))) + 1e-12
            z = (dist - np.median(dist)) / (1.4826 * mad)
            self.last_class_z = np.round(z, 2).tolist()
            self.last_head_rejected = [i for i in accepted if z[i] > self.class_mad]
            accepted = [i for i in accepted if z[i] <= self.class_mad]
        if self.reputation is not None:
            self.reputation.update({i: (i in accepted) for i in range(n)})
            accepted = self.reputation.eligible(accepted)
        if not accepted:
            return None, accepted, sims, ratios
        total = server_unmask_sum([p[0] for p in packages], accepted, n, round_seed, d, self.mask_scale)
        denom = len(accepted) if weights is None else sum(weights[i] for i in accepted)
        return total / denom, accepted, sims, ratios

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        models = raw_client_model_or_grad_list or self.model_pool
        n = len(models)
        if self.reference_model is None:
            raise RuntimeError("set_global_model() must be called before round 0")
        keys = _float_keys(models[0])
        ref = _flatten(self.reference_model, keys)
        d = len(ref)
        round_seed = int(self.seed * 100003 + self.round)
        h, s = count_sketch_params(d, self.k, round_seed + 7)
        if self.class_check:
            w_key, b_key = keys[-2], keys[-1]
            n_cls = _to_numpy(models[0][b_key]).size

        # public weights from earlier rounds (reputation); clients pre-scale their update by it
        weights = None if self.reputation is None else [self.reputation.get(i) for i in range(n)]

        # client side (each client computes its own package from its own update)
        packages = []
        for i, m in enumerate(models):
            u = _flatten(m, keys) - ref
            masked, _ = client_package(u if weights is None else weights[i] * u, i, n, round_seed,
                                       h, s, self.k, self.mask_scale)
            prof = None
            if self.class_check:
                dw = (_to_numpy(m[w_key]) - _to_numpy(self.reference_model[w_key])).reshape(n_cls, -1)
                db = (_to_numpy(m[b_key]) - _to_numpy(self.reference_model[b_key])).reshape(n_cls, 1)
                rows = np.linalg.norm(np.hstack([dw, db]), axis=1)
                prof = rows / (rows.sum() + 1e-12)
            packages.append((masked, count_sketch(u, h, s, self.k), prof))  # sketch + class profile
        # server side: sees only packages
        mean_update, accepted, sims, ratios = self._server_round(packages, d, round_seed, n, weights)
        if self.reputation is not None:
            self.last_reputation = [round(self.reputation.get(i), 3) for i in range(n)]
            logger.info(f"[sketchSecAgg] round {self.round}: reputation {self.last_reputation}")

        self.rejected_last_round = [i for i in range(n) if i not in accepted]
        self.last_similarities, self.last_norm_ratios = sims, ratios
        logger.info(f"[sketchSecAgg] round {self.round}: k={self.k} cos={np.round(sims, 3).tolist()} "
                    f"ratio={np.round(ratios, 2).tolist()} rejected={self.rejected_last_round}")
        if self.rejected_last_round:
            extra = f" (class-profile check: {self.last_head_rejected})" if self.class_check else ""
            print(f"[sketchSecAgg] round {self.round}: rejected clients (0-based index) "
                  f"{self.rejected_last_round}{extra}")

        from .robustAgg import _unflatten_like
        if mean_update is None:
            print(f"[sketchSecAgg] round {self.round}: all clients rejected, keeping previous global model")
            agg = snapshot_state_dict(self.reference_model)
        else:
            agg = _unflatten_like(ref + mean_update, models[0], keys)
        self.reference_model = snapshot_state_dict(agg)
        self.round += 1
        return agg
