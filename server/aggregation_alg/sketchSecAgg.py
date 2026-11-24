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
- secagg="sim" (default, as in all earlier runs): masks are real-valued Gaussian vectors and
  seed recovery is simulated (the server is handed the rejected clients' pair seeds).
- secagg="dh": the real protocol in secaggDH.py: Diffie-Hellman pair keys, SHAKE-256 masks in
  Z_2^64, self masks, and Shamir shares to remove a rejected client's masks.
- secagg="hardened" (Phase 1, secaggField.py): masks in the Ristretto255 scalar field, uploads
  committed on the bulletin board before the round's sketch seed exists, the seed drawn by the
  committee beacon (chainfl/beacon.py), and the sum check S(sum of accepted) == sum of sketches.
- c1_attack (test attacker against C1 itself, needs attack="signflip" so the attacker knows its
  benign "cover" update -u/scale): "sketchlie" reports the sketch of the cover while masking the
  poisoned update (gap 3); "nullspace" adds the poison only in directions the sketch it expects
  cannot see (gap 2), and reports its sketch honestly.
- The sketch itself leaks some information; inference_attacks.py --sketch-dims measures it.
"""
import hashlib
import logging
import time
from collections import OrderedDict

import numpy as np

from .similarityAgg import (_PoolMixin, _flatten, _float_keys, _to_numpy, cosine_similarity,
                            snapshot_state_dict)
from ..base.baseAggregator import ServerAggregator
from .secaggDH import DHSecAggRound
from . import secaggField as sf

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
    New identities start at `init` (Sybil handling: set it below tau so they must earn trust).
    Probation (whitewashing fix): an identity created during training (reset) is screened and
    scored as usual but kept out of aggregation for its first `probation` rounds, so dropping a
    bad identity for a fresh one no longer buys immediate access. Clients present from round 0
    have no probation. probation=0 = earlier behaviour. The aggregator calls tick() once per
    round, after deciding who is eligible."""

    def __init__(self, init=0.75, beta=0.7, tau=0.6, penalty=0.5, probation=0):
        self.init, self.beta, self.tau, self.penalty = init, beta, tau, penalty
        self.probation = probation
        self.r = {}
        self.on_probation = {}      # id -> rounds of probation left

    def get(self, i):
        return self.r.get(i, self.init)

    def update(self, verdicts):
        for i, v in verdicts.items():
            r = self.get(i)
            self.r[i] = self.beta * r + (1 - self.beta) if v else self.penalty * r

    def eligible(self, ids):
        return [i for i in ids if self.get(i) >= self.tau and self.on_probation.get(i, 0) <= 0]

    def reset(self, i):
        """Client i comes back under a brand-new identity (whitewashing): fresh reputation,
        and `probation` rounds outside aggregation."""
        self.r.pop(i, None)
        if self.probation > 0:
            self.on_probation[i] = self.probation
        else:
            self.on_probation.pop(i, None)

    def tick(self):
        """One round has passed."""
        for i in list(self.on_probation):
            self.on_probation[i] -= 1
            if self.on_probation[i] <= 0:
                del self.on_probation[i]


def collusion_flags(cos_hist, ages, window, threshold, group):
    """Collusion check (aimed at the non-IID backdoor). cos_hist: the last `window` per-round
    matrices of pairwise cosine between clients' centred sketches. A client seen for at least
    `window` rounds is flagged when at least group-1 other such clients have an average cosine
    above `threshold` with it: honest non-IID clients each pull in their own direction, while
    colluders push the same direction round after round. Cosine is compared per round, so it
    does not matter that each round uses a different sketch matrix."""
    n = cos_hist[-1].shape[0] if cos_hist else 0
    judged = [i for i in range(n) if ages.get(i, 0) >= window]
    if len(cos_hist) < window or len(judged) < group:
        return [], 0.0
    avg = np.mean(np.stack(list(cos_hist)[-window:]), axis=0)
    flagged, cmax = [], 0.0
    for i in judged:
        partners = [j for j in judged if j != i and avg[i, j] > threshold]
        cmax = max([cmax] + [float(avg[i, j]) for j in judged if j != i])
        if len(partners) >= group - 1:
            flagged.append(i)
    return flagged, round(cmax, 3)


# ---------------------------------------------------------------------------- aggregator
class sketchSecAggAggregator(_PoolMixin, ServerAggregator):
    def __init__(self, sketch_dim=64, similarity_threshold=0.1, magnitude_ratio=3.0,
                 mask_scale=100.0, seed=0, reputation=False, rep_init=0.75, rep_beta=0.7, rep_tau=0.6,
                 rep_penalty=0.5, class_check=False, class_mad=3.0, secagg="sim", secagg_threshold=None,
                 c1_attack="none", beacon_n=5, beacon_t=3, probation=0, collusion_check=False,
                 collusion_window=3, collusion_threshold=0.35, collusion_group=3, dump_screening=None,
                 screening="plain", model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        if secagg not in ("sim", "dh", "hardened"):
            raise ValueError(f"secagg must be 'sim', 'dh' or 'hardened', got {secagg!r}")
        if c1_attack not in ("none", "sketchlie", "nullspace"):
            raise ValueError(f"c1_attack must be 'none', 'sketchlie' or 'nullspace', got {c1_attack!r}")
        if screening not in ("plain", "mpc", "mpc-check"):
            raise ValueError(f"screening must be 'plain', 'mpc' or 'mpc-check', got {screening!r}")
        if screening != "plain" and secagg != "hardened":
            raise ValueError("screening='mpc' needs secagg='hardened'")
        self.secagg = secagg
        self.screening = screening
        self.mpc = None                 # MPCScreening, started on the first round
        self.mpc_mismatch_rounds = 0
        self.last_mpc_diff = []
        self.last_mpc_seconds = 0.0
        self._coll_rounds = 0
        self.c1_attack = c1_attack
        self.beacon_n, self.beacon_t = beacon_n, beacon_t
        self.board = None            # bulletin board (MemoryBoard, or ChainBoard with --onchain)
        self.beacon = None
        self.last_sum_check = None   # hardened: True / False / None (no accepted client)
        self.sum_check_failures = 0
        self.secagg_threshold = secagg_threshold
        self.last_secagg_seconds = 0.0
        # Backdoor signal: each client also reports its "class profile" = the share of its
        # output-layer update that falls on each class row (C numbers, e.g. 10). A backdoor
        # drags the model toward one target class, so the attacker's profile is lopsided.
        # Clients whose profile is far from the median profile (robust z-score > class_mad)
        # are rejected. Privacy cost: the profile hints at the client's label mix.
        self.class_check = class_check
        self.class_mad = class_mad
        self.reputation = Reputation(rep_init, rep_beta, rep_tau, rep_penalty, probation) if reputation else None
        self.collusion_check = collusion_check
        self.collusion_window, self.collusion_threshold = collusion_window, collusion_threshold
        self.collusion_group = collusion_group
        self._cos_hist = []          # last collusion_window pairwise-cosine matrices
        self._ages = {}              # client -> rounds seen since (re)registration
        self.last_collusion_rejected = []
        self.last_collusion_max = 0.0
        self.last_reputation = []
        self.last_head_rejected = []
        self.last_screen_rejected = []
        self.last_rep_excluded = []
        self.dump_screening = dump_screening   # directory: save each round's screening inputs/verdicts
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
    def _server_round(self, packages, d, round_seed, n, weights=None, dh_round=None):
        """weights: public per-client weights the clients already multiplied into their
        update (reputation from earlier rounds); None = plain average."""
        accepted, sims, ratios = self._decide([p[1] for p in packages], [p[2] for p in packages], n)
        if not accepted:
            return None, accepted, sims, ratios
        if dh_round is not None:
            total = dh_round.server_unmask_sum({i: p[0] for i, p in enumerate(packages)}, accepted)
        else:
            total = server_unmask_sum([p[0] for p in packages], accepted, n, round_seed, d, self.mask_scale)
        denom = len(accepted) if weights is None else sum(weights[i] for i in accepted)
        return total / denom, accepted, sims, ratios

    def _decide(self, sketches, profiles, n, sketch_ints=None, weights=None):
        """Screening on sketches, class-profile check, collusion check, then reputation: the
        accepted set. The three checks run either in plaintext at the server (screening="plain")
        or as an MPC among the committee (screening="mpc", mpc/bcfl_screen.mpc; needs the
        integer sketches of hardened mode). screening="mpc-check" runs both and counts rounds
        where the verdicts differ (a development check: the plaintext side sees the sketches)."""
        ages_before = dict(self._ages)
        if self.collusion_check:
            for i in range(n):
                self._ages[i] = self._ages.get(i, 0) + 1
        sims, ratios = [], []
        if self.screening in ("plain", "mpc-check"):
            screen_ok, class_ok, coll_flag, sims, ratios = self._checks_plain(sketches, profiles, n)
        if self.screening in ("mpc", "mpc-check"):
            if sketch_ints is None:
                raise ValueError("screening='mpc' needs --secagg hardened (integer sketches)")
            m = self._checks_mpc(sketch_ints, profiles, n, weights)
            if self.screening == "mpc-check":
                plain = (screen_ok, class_ok, coll_flag)
                diff = [(name, i) for name, a, b in zip(("screen", "class", "collusion"), plain, m)
                        for i in range(n) if bool(a[i]) != bool(b[i])]
                self.last_mpc_diff = diff
                if diff:
                    self.mpc_mismatch_rounds += 1
                    print(f"[sketchSecAgg] round {self.round}: MPC and plaintext verdicts differ: {diff}")
            screen_ok, class_ok, coll_flag = m
        accepted = [i for i in range(n) if screen_ok[i]]
        self.last_screen_rejected = [i for i in range(n) if not screen_ok[i]]
        self.last_head_rejected = [i for i in accepted if not class_ok[i]]
        accepted = [i for i in accepted if class_ok[i]]
        self.last_collusion_rejected = [i for i in accepted if coll_flag[i]]
        accepted = [i for i in accepted if not coll_flag[i]]
        self.last_rep_excluded = []
        if self.reputation is not None:
            self.reputation.update({i: (i in accepted) for i in range(n)})
            eligible = self.reputation.eligible(accepted)
            self.last_rep_excluded = [i for i in accepted if i not in eligible]
            accepted = eligible
            self.reputation.tick()
        if self.dump_screening:
            self._dump(sketches, profiles, n, ages_before, accepted, sketch_ints, weights)
        return accepted, sims, ratios

    def _collusion_public(self, n):
        """Public inputs of the collusion check: who is judged, and whether it runs this round."""
        self._coll_rounds += 1
        judged = [self._ages.get(i, 0) >= self.collusion_window for i in range(n)]
        active = self._coll_rounds >= self.collusion_window and sum(judged) >= self.collusion_group
        return judged, active

    def _checks_plain(self, sketches, profiles, n):
        accepted, sims, ratios = screen(sketches, self.similarity_threshold, self.magnitude_ratio)
        screen_ok = [i in accepted for i in range(n)]
        class_ok = [True] * n
        if self.class_check and profiles[0] is not None:
            P = np.stack(profiles)
            med = np.median(P, axis=0)
            dist = np.abs(P - med).sum(axis=1)                    # L1 distance to the median profile
            mad = np.median(np.abs(dist - np.median(dist))) + 1e-12
            z = (dist - np.median(dist)) / (1.4826 * mad)
            self.last_class_z = np.round(z, 2).tolist()
            class_ok = [bool(z[i] <= self.class_mad) for i in range(n)]
        coll_flag = [False] * n
        if self.collusion_check:
            V = np.stack([np.asarray(v, dtype=np.float64) for v in sketches])
            C = V - np.median(V, axis=0)                             # remove the common direction
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
            self._cos_hist = (self._cos_hist + [C @ C.T])[-self.collusion_window:]
            flagged, self.last_collusion_max = collusion_flags(
                self._cos_hist, self._ages, self.collusion_window, self.collusion_threshold, self.collusion_group)
            coll_flag = [i in flagged for i in range(n)]
        return screen_ok, class_ok, coll_flag, sims, ratios

    def _checks_mpc(self, sketch_ints, profiles, n, weights):
        if self.mpc is None:
            from mpc.screen_mpc import MPCScreening
            c = len(profiles[0]) if (self.class_check and profiles[0] is not None) else 0
            self.mpc = MPCScreening(n, self.k, c, self.collusion_window if self.collusion_check else 0,
                                    self.similarity_threshold, self.magnitude_ratio, self.class_mad,
                                    self.collusion_threshold, self.collusion_group,
                                    parties=self.beacon_n).start()
        judged, active = self._collusion_public(n) if self.collusion_check else ([False] * n, False)
        prof_ints = [sf.encode_int(p) for p in profiles] if self.mpc.c else None
        s_ok, c_ok, flag = self.mpc.decide(sketch_ints, prof_ints, weights, judged, active)
        return [bool(x) for x in s_ok], [bool(x) for x in c_ok], [bool(x) for x in flag]

    def _dump(self, sketches, profiles, n, ages_before, accepted, sketch_ints, weights):
        import os
        os.makedirs(self.dump_screening, exist_ok=True)
        np.savez(os.path.join(self.dump_screening, f"round_{self.round:03d}.npz"),
                 sketches=np.stack([np.asarray(v, dtype=np.float64) for v in sketches]),
                 sketch_ints=np.zeros((0,)) if sketch_ints is None else np.stack(sketch_ints),
                 weights=np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64),
                 profiles=np.stack(profiles) if profiles[0] is not None else np.zeros((0,)),
                 ages=np.array([ages_before.get(i, 0) for i in range(n)]),
                 screen_rej=np.array(self.last_screen_rejected, dtype=int),
                 class_rej=np.array(self.last_head_rejected, dtype=int),
                 coll_rej=np.array(self.last_collusion_rejected, dtype=int),
                 accepted=np.array(accepted, dtype=int))

    def forget_client(self, i):
        """Client i re-registers under a new identity: fresh reputation (with probation, if set)
        and an empty collusion history, so it is judged again after the warm-up window."""
        if self.reputation is not None:
            self.reputation.reset(i)
        self._ages[i] = 0

    # ----- test attacker against C1 itself (gaps 2 and 3)
    def _attacking(self, i):
        if self.c1_attack == "none" or i not in set(getattr(self, "malicious", [])):
            return False
        if getattr(self, "attack", "none") != "signflip":
            raise ValueError("c1_attack needs attack='signflip' (the attacker's cover update is -u/scale)")
        try:
            from client.poisonedLoader import attack_on
            return bool(attack_on(self.round))
        except ImportError:
            return True

    def _adversary(self, i, u, guessed_hs):
        """Returns (vector the client masks, vector it reports a sketch of)."""
        if not self._attacking(i):
            return u, u
        cover = -u / float(getattr(self, "attack_scale", 5.0))
        if self.c1_attack == "sketchlie":
            return u, cover
        h, s = guessed_hs        # nullspace: poison only where the expected sketch is blind
        x = cover + sf.null_space_project(u - cover, h.astype(np.int64), s, self.k)
        return x, x

    def _ensure_board(self):
        if self.board is None:
            from chainfl.board import MemoryBoard
            self.board = MemoryBoard()
        if self.beacon is None:
            from chainfl.beacon import BeaconCommittee
            self.beacon = BeaconCommittee(self.board, self.beacon_n, self.beacon_t)

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

        if self.secagg == "hardened":
            profiles = [self._profile(m, w_key, b_key, n_cls) if self.class_check else None for m in models]
            mean_update, accepted, sims, ratios = self._hardened_round(models, keys, ref, d, n, weights,
                                                                        profiles, (h, s))
            return self._finish_round(mean_update, accepted, sims, ratios, models, keys, ref, n, None)

        # client side (each client computes its own package from its own update)
        t0 = time.perf_counter()
        dh_round = DHSecAggRound(n, d, self.secagg_threshold) if self.secagg == "dh" else None
        packages = []
        for i, m in enumerate(models):
            u_true = _flatten(m, keys) - ref
            u, u_sk = self._adversary(i, u_true, (h, s))    # the legacy sketch is public: (h, s)
            x = u if weights is None else weights[i] * u
            if dh_round is not None:
                masked = dh_round.client_upload(i, x)
            else:
                masked, _ = client_package(x, i, n, round_seed, h, s, self.k, self.mask_scale)
            prof = self._profile(m, w_key, b_key, n_cls) if self.class_check else None
            packages.append((masked, count_sketch(u_sk, h, s, self.k), prof))  # sketch + class profile
        # commitments of what each client uploaded (for the on-chain ledger, C2)
        self.last_update_hashes = [hashlib.sha256(p[0].tobytes()).digest() for p in packages]
        # server side: sees only packages
        mean_update, accepted, sims, ratios = self._server_round(packages, d, round_seed, n, weights, dh_round)
        self.last_secagg_seconds = time.perf_counter() - t0
        if dh_round is not None:
            logger.info(f"[sketchSecAgg] round {self.round}: DH secure aggregation {self.last_secagg_seconds:.2f}s "
                        f"(n={n}, d={d}, threshold {dh_round.t})")
        return self._finish_round(mean_update, accepted, sims, ratios, models, keys, ref, n, None)

    def _profile(self, m, w_key, b_key, n_cls):
        dw = (_to_numpy(m[w_key]) - _to_numpy(self.reference_model[w_key])).reshape(n_cls, -1)
        db = (_to_numpy(m[b_key]) - _to_numpy(self.reference_model[b_key])).reshape(n_cls, 1)
        rows = np.linalg.norm(np.hstack([dw, db]), axis=1)
        return rows / (rows.sum() + 1e-12)

    def _hardened_round(self, models, keys, ref, d, n, weights, profiles, guessed_hs):
        """Phase 1 round: commit uploads -> beacon seed -> sketches -> screening -> unmask the
        accepted sum -> sum check. Client and server steps are kept apart as in the other modes."""
        t0 = time.perf_counter()
        self._ensure_board()
        rnd = self.round
        self.beacon.run_open_phase(rnd)                 # beacon deals while uploads are open
        frnd = sf.FieldSecAggRound(n, d, self.secagg_threshold)
        uploads, x_ints, cover_ints = {}, {}, {}
        for i, m in enumerate(models):                  # client side, before any seed exists
            u_true = _flatten(m, keys) - ref
            u, u_sk = self._adversary(i, u_true, guessed_hs)
            w = 1.0 if weights is None else weights[i]
            x_ints[i] = sf.encode_int(w * u)
            cover_ints[i] = sf.encode_int(w * u_sk)
            uploads[i] = frnd.client_upload_int(i, x_ints[i])
        hashes = [sf.upload_hash(rnd, i, uploads[i]) for i in range(n)]
        self.board.commit_uploads(rnd, hashes)          # uploads fixed; round closed
        seed = self.beacon.run_closed_phase(rnd)        # only now is the sketch defined
        h, s = sf.sketch_params_from_seed(seed, d, self.k)
        sketches = []
        for i in range(n):                              # client side: report a sketch
            honest = self.c1_attack == "nullspace" or not self._attacking(i)
            sketches.append(sf.sketch_int(x_ints[i] if honest else cover_ints[i], h, s, self.k))
        # server side: check uploads against the board, screen, unmask, check the sum
        posted = self.board.upload_commitments(rnd)
        for i in range(n):
            if sf.upload_hash(rnd, i, uploads[i]) != posted[i]:
                raise RuntimeError(f"round {rnd}: client {i}'s upload does not match its commitment")
        wts = [1.0] * n if weights is None else [max(w, 1e-12) for w in weights]
        float_sketches = [sf.decode_int(sketches[i]) / wts[i] for i in range(n)]
        accepted, sims, ratios = self._decide(float_sketches, profiles, n, sketches, weights)
        self.last_update_hashes = hashes
        self.last_sum_check = None
        mean = None
        if accepted:
            sum_int = frnd.server_unmask_sum_int(uploads, accepted)
            if self.screening in ("mpc", "mpc-check"):
                # the committee reveals only the sum of the accepted sketches
                total = self.mpc.sum_accepted(accepted)
                self.last_sum_check = sf.sum_check_total(sum_int, total, h, s, self.k)
                self.last_mpc_seconds = self.mpc.last_seconds
            else:
                self.last_sum_check = sf.sum_check(sum_int, sketches, accepted, h, s, self.k)
            self.board.post("round_outcome", rnd, "server", "sum_check", self.last_sum_check)
            if self.last_sum_check:
                denom = len(accepted) if weights is None else sum(weights[i] for i in accepted)
                mean = sf.decode_int(sum_int) / denom
            else:
                self.sum_check_failures += 1
                print(f"[sketchSecAgg] round {rnd}: SUM CHECK FAILED: the accepted sketches do not add up "
                      f"to the sketch of the unmasked sum; round discarded, previous global model kept")
        self.last_secagg_seconds = time.perf_counter() - t0
        logger.info(f"[sketchSecAgg] round {rnd}: hardened round {self.last_secagg_seconds:.2f}s "
                    f"(n={n}, d={d}, threshold {frnd.t}, beacon seed {seed.hex()[:16]}..., "
                    f"sum check {self.last_sum_check})")
        return mean, accepted, sims, ratios

    def _finish_round(self, mean_update, accepted, sims, ratios, models, keys, ref, n, _unused):
        if self.reputation is not None:
            self.last_reputation = [round(self.reputation.get(i), 3) for i in range(n)]
            logger.info(f"[sketchSecAgg] round {self.round}: reputation {self.last_reputation}")

        self.rejected_last_round = [i for i in range(n) if i not in accepted]
        discarded = self.secagg == "hardened" and self.last_sum_check is False
        if discarded:   # nothing from this round entered the model
            self.rejected_last_round = list(range(n))
        self.last_similarities, self.last_norm_ratios = sims, ratios
        logger.info(f"[sketchSecAgg] round {self.round}: k={self.k} cos={np.round(sims, 3).tolist()} "
                    f"ratio={np.round(ratios, 2).tolist()} rejected={self.rejected_last_round}")
        if self.rejected_last_round:
            extra = f" (screen: {self.last_screen_rejected}"
            if self.class_check:
                extra += f", class-profile check: {self.last_head_rejected}"
            if self.collusion_check:
                extra += f", collusion check: {self.last_collusion_rejected}"
            if self.reputation is not None:
                extra += f", low reputation: {self.last_rep_excluded}"
            extra += ")"
            print(f"[sketchSecAgg] round {self.round}: rejected clients (0-based index) "
                  f"{self.rejected_last_round}{extra}")

        from .robustAgg import _unflatten_like
        if mean_update is None:
            if not discarded:
                print(f"[sketchSecAgg] round {self.round}: all clients rejected, keeping previous global model")
            agg = snapshot_state_dict(self.reference_model)
        else:
            agg = _unflatten_like(ref + mean_update, models[0], keys)
        self.reference_model = snapshot_state_dict(agg)
        self.round += 1
        return agg
