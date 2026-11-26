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
- secagg_vss (default True, hardened only; Phase 3, secaggVSS.py): the mask secrets are shared
  with Feldman VSS. A client whose shares do not match its public commitments (and who does not
  fix them when a holder complains) is left out of the round, and a holder that returns a wrong
  share at unmasking is caught and named. secagg_vss=False is the Phase 1 round (plain Shamir).
- secagg_zk (default True, hardened + VSS; Phase 5, zkSecAgg.py): each client commits to its
  update with its upload, proves in zero knowledge that the sketch and class input it gives the
  screening come from that update, and binds the commitment to its masked upload; a client
  caught lying is named and kept out of the aggregate, and the round goes on. The class profile
  is then computed from the (proven) squared class-row norms. secagg_zk=False is Phase 4.
- c1_attack "commitlie" / "pairlie" (need attack="signflip"): commit to and report a harmless
  cover update but upload the poisoned one; "pairlie" also bends one pair-mask commitment so the
  upload relation still holds.
- server_checks (default True, hardened only; Phase 4, chainfl/relay.py): every post is signed,
  the server anchors a digest of what it relayed at four points of the round, and every client,
  holder and committee member checks its own view against the anchor before acting; holders
  reveal only per the server's signed accepted set, and with --screening mpc only if it agrees
  with the verdicts the committee itself received and attested. Clients check the global model
  they get against the anchored hash (and, with MPC, the public sum check). A mismatch stops
  the round. server_checks=False is Phase 3 (readers trust the server).
- server_attack (test attacker: a malicious server, colluding with the malicious clients):
  "split_accepted" shows half of the holders an accepted set with a victim and half one
  without it, to collect both of the victim's secrets; "drop_deal" hides a client's deal from
  some clients; "forge_complaint" posts complaints in honest holders' names to throw a client
  out; "fake_accept" (MPC) accepts a client the screening rejected; "wrong_model" sends one
  client a different global model.
- c1_attack "badshare" / "badreveal" (malicious clients, any data attack): hand holders wrong
  shares and ignore complaints / return wrong shares at unmasking (gap 1).
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
                 collusion_window=3, collusion_threshold=0.5, collusion_group=3, dump_screening=None,
                 screening="plain", secagg_vss=True, server_checks=True, server_attack="none",
                 secagg_zk=True, model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        if secagg not in ("sim", "dh", "hardened"):
            raise ValueError(f"secagg must be 'sim', 'dh' or 'hardened', got {secagg!r}")
        if c1_attack not in ("none", "sketchlie", "nullspace", "badshare", "badreveal", "commitlie", "pairlie"):
            raise ValueError(f"c1_attack must be none, sketchlie, nullspace, badshare, badreveal, commitlie or "
                             f"pairlie, got {c1_attack!r}")
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
        self.secagg_vss = secagg_vss
        if server_attack not in ("none", "split_accepted", "drop_deal", "forge_complaint", "fake_accept", "wrong_model"):
            raise ValueError(f"unknown server_attack {server_attack!r}")
        if server_attack == "fake_accept" and screening == "plain":
            raise ValueError("server_attack='fake_accept' needs --screening mpc (with plaintext screening the "
                             "server decides the verdicts itself)")
        self.server_checks = server_checks
        self.secagg_zk = secagg_zk and secagg_vss
        self.last_zk_rejected = []   # clients whose proof failed or who lied in a pair dispute (before screening)
        self.last_zk_late = []       # clients caught after acceptance (upload relation / audit), removed
        self.zk_rejections = 0
        self.last_zk_seconds = 0.0
        self._zk_invalid = set()
        self._zk_hook = None
        self._zk_holder_memo = {}
        self.server_attack = server_attack
        self.relay = None            # RelayBoard (Phase 4) around self.board
        self._ids = None             # {'client': {i: Identity}, 'member': {m: Identity}, 'server': Identity}
        self.last_alarms = []        # equivocation alarms raised this round (Phase 4)
        self.alarm_rounds = 0
        self.last_privacy_breach = None   # split_accepted attack: did the server recover the victim's update?
        self.privacy_breaches = 0
        self.last_model_alarms = []  # clients whose received global model failed the check
        self._verdict_bits = None
        self._attest_total = None
        self._aborted = set()
        self._box_keys = None        # clients' long-term box keys (X25519), made on the first VSS round
        self.last_vss_rejected = []  # clients left out of the round by VSS (Phase 3)
        self.last_bad_reveals = []   # holders caught returning a wrong share
        self.vss_rejections = 0
        self.unmask_failures = 0
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

    def _decide(self, sketches, profiles, n, sketch_ints=None, weights=None, absent=frozenset()):
        """Screening on sketches, class-profile check, collusion check, then reputation: the
        accepted set. The three checks run either in plaintext at the server (screening="plain")
        or as an MPC among the committee (screening="mpc", mpc/bcfl_screen.mpc; needs the
        integer sketches of hardened mode). screening="mpc-check" runs both and counts rounds
        where the verdicts differ (a development check: the plaintext side sees the sketches).
        absent: clients not in this round (left out by VSS): no sketch, rejected, and every
        median / collusion comparison is over the present clients only."""
        absent = set(absent)
        ages_before = dict(self._ages)
        if self.collusion_check:
            for i in range(n):
                self._ages[i] = self._ages.get(i, 0) + 1
        sims, ratios = [], []
        if self.screening in ("mpc", "mpc-check"):
            if sketch_ints is None:
                raise ValueError("screening='mpc' needs --secagg hardened (integer sketches)")
            m = self._checks_mpc(sketch_ints, profiles, n, weights, absent)
            absent = absent | self._zk_invalid           # Phase 5: proofs that did not verify
        if self.screening in ("plain", "mpc-check"):
            screen_ok, class_ok, coll_flag, sims, ratios = self._checks_plain(sketches, profiles, n, absent)
        if self.screening in ("mpc", "mpc-check"):
            if self.screening == "mpc-check":
                plain = (screen_ok, class_ok, coll_flag)
                diff = [(name, i) for name, a, b in zip(("screen", "class", "collusion"), plain, m)
                        for i in range(n) if i not in absent and bool(a[i]) != bool(b[i])]
                self.last_mpc_diff = diff
                if diff:
                    self.mpc_mismatch_rounds += 1
                    print(f"[sketchSecAgg] round {self.round}: MPC and plaintext verdicts differ: {diff}")
            screen_ok, class_ok, coll_flag = m
        accepted = [i for i in range(n) if i not in absent and screen_ok[i]]
        self.last_screen_rejected = [i for i in range(n) if i not in absent and not screen_ok[i]]
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

    def _collusion_public(self, n, absent=frozenset()):
        """Public inputs of the collusion check: who is judged, and whether it runs this round."""
        self._coll_rounds += 1
        judged = [i not in absent and self._ages.get(i, 0) >= self.collusion_window for i in range(n)]
        active = self._coll_rounds >= self.collusion_window and sum(judged) >= self.collusion_group
        return judged, active

    def _checks_plain(self, sketches, profiles, n, absent=frozenset()):
        present = [i for i in range(n) if i not in absent]
        acc_p, sims_p, ratios_p = screen([sketches[i] for i in present], self.similarity_threshold,
                                         self.magnitude_ratio)
        screen_ok, sims, ratios = [False] * n, [0.0] * n, [0.0] * n
        for r, i in enumerate(present):
            screen_ok[i], sims[i], ratios[i] = r in acc_p, sims_p[r], ratios_p[r]
        class_ok = [True] * n
        if self.class_check and profiles[0] is not None:
            P = np.stack([profiles[i] for i in present])
            med = np.median(P, axis=0)
            dist = np.abs(P - med).sum(axis=1)                    # L1 distance to the median profile
            mad = np.median(np.abs(dist - np.median(dist))) + 1e-12
            z = (dist - np.median(dist)) / (1.4826 * mad)
            self.last_class_z = np.round(z, 2).tolist()
            for r, i in enumerate(present):
                class_ok[i] = bool(z[r] <= self.class_mad)
        coll_flag = [False] * n
        if self.collusion_check:
            V = np.stack([np.asarray(v, dtype=np.float64) for v in sketches])
            C = V - np.median(V[present], axis=0)                  # remove the common direction
            C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
            C[sorted(absent)] = 0.0                                 # absent: cosine 0 with everyone
            self._cos_hist = (self._cos_hist + [C @ C.T])[-self.collusion_window:]
            ages = {i: (0 if i in absent else a) for i, a in self._ages.items()}
            flagged, self.last_collusion_max = collusion_flags(
                self._cos_hist, ages, self.collusion_window, self.collusion_threshold, self.collusion_group)
            coll_flag = [i in flagged for i in range(n)]
        return screen_ok, class_ok, coll_flag, sims, ratios

    def _checks_mpc(self, sketch_ints, profiles, n, weights, absent=frozenset()):
        if self.mpc is None:
            c = len(profiles[0]) if (self.class_check and profiles[0] is not None) else 0
            self._ensure_mpc(n, c)
        q_ints = getattr(self, "_q_ints", None)
        if self.mpc.c and q_ints is None:                # outside hardened rounds: profile as q-like ints
            q_ints = [[int(v) for v in sf.encode_int(np.asarray(p) ** 2 * (1 << 24))] for p in profiles]
        present = [i not in absent for i in range(n)]
        zr = getattr(self, "_zk_round", None)
        mus = [zr.mpc_masks(i) if (zr is not None and present[i] and i in zr._sec) else [0, 0] for i in range(n)]
        outs = self.mpc.begin(sketch_ints, q_ints, weights, present, mus if self.mpc.zk else None)
        valid = None
        if self.mpc.zk and self._zk_hook is not None:
            valid = self._zk_hook(outs)
            absent = set(absent) | self._zk_invalid
        judged, active = self._collusion_public(n, absent) if self.collusion_check else ([False] * n, False)
        s_ok, c_ok, flag = self.mpc.finish(judged, active, valid)
        self._verdict_bits = [int(x) for x in s_ok] + [int(x) for x in c_ok] + [int(x) for x in flag]
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
    def _share_attackers(self):
        """Malicious clients running the badshare / badreveal attack this round."""
        if self.c1_attack not in ("badshare", "badreveal"):
            return set()
        try:
            from client.poisonedLoader import attack_on
            if not attack_on(self.round):
                return set()
        except ImportError:
            pass
        return set(getattr(self, "malicious", []))

    def _share_faults(self, n):
        atk = self._share_attackers()
        if self.c1_attack == "badshare":
            return {"bad_share": {(i, j) for i in atk for j in range(n) if j != i}, "ignore_complaints": atk}
        if self.c1_attack == "badreveal":
            return {"wrong_reveal": atk}
        return {}

    def _attacking(self, i):
        if self.c1_attack in ("none", "badshare", "badreveal") or i not in set(getattr(self, "malicious", [])):
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
        if self.c1_attack in ("sketchlie", "commitlie", "pairlie"):
            return u, cover
        h, s = guessed_hs        # nullspace: poison only where the expected sketch is blind
        x = cover + sf.null_space_project(u - cover, h.astype(np.int64), s, self.k)
        return x, x

    def _ensure_phase4(self, n):
        """Identities, registry and the relayed board (Phase 4). Plain board when hardened mode
        runs without VSS (Phase 1)."""
        if self.relay is not None or not self.secagg_vss:
            return
        from chainfl.relay import RelayBoard, Registry
        from secure.signing import Identity
        members = list(self.beacon.members)
        self._ids = {"client": {i: Identity() for i in range(n)},
                     "member": {m: Identity() for m in members}, "server": Identity()}
        reg = Registry({i: k.pk for i, k in self._ids["client"].items()},
                       {m: k.pk for m, k in self._ids["member"].items()}, self._ids["server"].pk)
        self.relay = RelayBoard(self.board, reg, verify=self.server_checks)
        self.beacon.board = self.relay
        self.beacon.ids = self._ids["member"]

    def _server_post(self, kind, rnd, key, value):
        sig = self._ids["server"].sign_post(kind, rnd, "server", key, value)
        self.relay.post(kind, rnd, "server", key, value, sig)

    def _member_post(self, kind, rnd, m, key, value):
        sig = self._ids["member"][m].sign_post(kind, rnd, m, key, value)
        self.relay.post(kind, rnd, m, key, value, sig)

    def _victim(self, n, among=None):
        honest = [i for i in range(n) if i not in set(getattr(self, "malicious", []))]
        pool = [i for i in honest if among is None or i in among]
        return pool[0] if pool else None

    def _install_server_attack(self, rnd, n):
        """Malicious server, before the round's setup: what each reader is shown."""
        if self.relay is None or self.server_attack == "none":
            return
        from secure.signing import Identity
        v = self._victim(n)
        if self.server_attack == "drop_deal":
            honest = [i for i in range(n) if i not in set(getattr(self, "malicious", [])) and i != v]
            for i in honest[len(honest) // 2:]:
                self.relay.hide_from(("client", i), lambda kind, r, a, k, val, v=v, rnd=rnd:
                                     kind == "secagg_deal" and r == rnd and a == v)
        elif self.server_attack == "forge_complaint":
            fake = Identity()                       # the server has no honest holder's key
            others = [j for j in range(n) if j != v][: self.secagg_threshold or (n // 2 + 1)]
            for j in others:                        # complaints "from" t holders against honest client v
                self.relay.post("secagg_complaint", rnd, j, (j, v), None,
                                fake.sign_post("secagg_complaint", rnd, j, (j, v), None))

    def _client_checks(self, rnd, phase, frnd):
        refused = frnd.check_views(phase)
        return refused

    def _attestation_validator(self, rnd, n, qualified):
        """Holder-side check of the accepted set against the committee's attestations (MPC)."""
        from chainfl.relay import EquivocationError
        from secure.signing import canon
        t = self.beacon.t

        def validate(j, view):
            posts = [v for a, k, v in view.entries("round_accepted", rnd) if k == "accepted"]
            if not posts:
                raise EquivocationError(("client", j), rnd, "outcome", "no accepted set posted")
            post = posts[0]
            bits = post.get("verdicts")
            if bits is None:
                return                                  # plaintext screening: server-trusted
            dig = hashlib.sha256(b"bcfl-verdicts|" + canon(bits)).digest()
            att = [m for a, (k, m), v in view.entries("verdict_attest", rnd) if k == "verdicts" and a == m and v == dig]
            if len(set(att)) < t:
                raise EquivocationError(("client", j), rnd, "outcome",
                                        f"verdicts attested by {len(set(att))} committee members, need {t}")
            ok = {i for i in post.get("present", range(n))
                  if bits[i] and bits[n + i] and not bits[2 * n + i]}
            zko = post.get("zk")
            if zko is not None:                         # Phase 5: the binding outputs, attested
                dz = hashlib.sha256(b"bcfl-zkout|" + canon([zko["coef"], zko["V1"], zko["V2"],
                                                           zko["in_range"]])).digest()
                attz = {m for a, (k, m), v in view.entries("verdict_attest", rnd) if k == "zk" and a == m and v == dz}
                if len(attz) < t:
                    raise EquivocationError(("client", j), rnd, "outcome",
                                            f"ZK binding outputs attested by {len(attz)} members, need {t}")
                zr = self._zk_round
                for i in post["accepted"]:              # the holder checks each accepted proof itself
                    key = (rnd, i)
                    if key not in self._zk_holder_memo:  # (same public data for every holder: computed once here)
                        self._zk_holder_memo[key] = zr.verify_mpc(i, zko, self.k, reader=("client", j))
                    if not self._zk_holder_memo[key]:
                        raise EquivocationError(("client", j), rnd, "outcome",
                                                f"accepted client {i} has no valid proof")
            if not set(post["accepted"]) <= ok:
                raise EquivocationError(("client", j), rnd, "outcome",
                                        f"accepted set has clients the screening rejected: "
                                        f"{sorted(set(post['accepted']) - ok)}")
        return validate

    def _try_recover(self, frnd, v, uploads, malicious):
        """split_accepted attack, server side: with the shares the honest holders revealed plus
        everything the colluding clients hold, rebuild both secrets of victim v and unmask its
        individual upload. Returns the recovered integer update or None."""
        from secure import vss as _vss
        from secure import ristretto as _rt
        from .secaggVSS import deals_on_board, pair_key
        board = self.relay
        deals = deals_on_board(board, frnd.rnd)
        shares = {"b": {}, "sk": {}}
        for a, (j, kind, i), y in board.entries("secagg_reveal", frnd.rnd):
            if i == v and a == j:
                shares[kind][j + 1] = y
        for j in malicious:                            # colluding holders give theirs
            sh = frnd.holder_share(j, v)
            if sh is not None:
                shares["sk"][j + 1], shares["b"][j + 1] = sh
        if len(shares["b"]) < frnd.t or len(shares["sk"]) < frnd.t:
            return None
        b = _vss.reconstruct(dict(list(shares["b"].items())[:frnd.t]))
        skv = _vss.reconstruct(dict(list(shares["sk"].items())[:frnd.t]))
        if _rt.g_mul(skv) != deals[v]["pk"]:
            return None
        y = (uploads[v] - sf.prg_field(_rt.scalar_bytes(b), frnd.d, b"self")) % sf.L
        for j in frnd.qualified:
            if j == v:
                continue
            p = sf.prg_field(pair_key(skv, deals[v]["pk"], deals[j]["pk"]), frnd.d, b"pair")
            y = (y - p) % sf.L if j > v else (y + p) % sf.L
        return sf.from_field(y)

    @staticmethod
    def model_hash(state, keys=None):
        """Hash of a global model as clients receive it (every float tensor, in key order)."""
        h = hashlib.sha256(b"bcfl-model|")
        for k in (keys or list(state.keys())):
            a = np.ascontiguousarray(_to_numpy(state[k]))
            h.update(k.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        return h.digest()

    def _post_aggregate(self, agg, keys, updated):
        """Phase 4, server: the round's aggregate (the unmasked sum is public anyway: it is the
        model update) and the hash of the new global model, then the 'final' anchor."""
        if self.relay is None or (self.round - 1) in getattr(self, "_aborted", set()):
            return
        rnd = self.round - 1
        self._model_keys = keys
        value = {"model": self.model_hash(agg, keys), "updated": bool(updated)}
        if updated:
            value["sum"] = [int(x) for x in self._last_sum_int]
            value["denom"] = repr(float(self._last_denom))
            value["total"] = None if self._attest_total is None else [int(x) for x in self._attest_total]
        try:
            self._server_post("aggregate", rnd, "aggregate", value)
            self.relay.anchor_phase(rnd, "final")
        except Exception as e:  # noqa: BLE001 - e.g. the round was stopped before closing
            logger.info(f"[sketchSecAgg] round {rnd}: no aggregate posted ({e})")
        self._attest_total = None

    def deliver_global(self, agg, n):
        """Phase 4: the server sends each client the global model; each client checks it against
        the anchored hash (and, with MPC screening, that the posted sum passes the public sum
        check against the committee-attested total). Returns the model each client ends up with
        and records the clients that caught a wrong one (they then take the anchored model,
        which any honest party can serve)."""
        self.last_model_alarms = []
        rnd = self.round - 1
        if self.relay is None or rnd in self._aborted:
            return [agg] * n                    # stopped round: everyone keeps the previous model
        keys = getattr(self, "_model_keys", None) or list(agg.keys())
        out = []
        for i in range(n):
            sent = agg
            if self.server_attack == "wrong_model" and i == self._victim(n):
                sent = OrderedDict((k, v.clone() if hasattr(v, "clone") else np.array(v, copy=True))
                                   for k, v in agg.items())
                k0 = keys[0]
                sent[k0] = sent[k0] + 0.5
            if self.server_checks and not self._client_accepts_model(i, rnd, sent, keys):
                self.last_model_alarms.append(i)
                sent = agg
            out.append(sent)
        if self.last_model_alarms:
            print(f"[sketchSecAgg] round {rnd}: clients {self.last_model_alarms} received a global model "
                  f"that does not match the anchored hash and refused it")
        return out

    def _client_accepts_model(self, i, rnd, model, keys):
        from chainfl.relay import EquivocationError
        from secure.signing import canon
        view = self.relay.view(("client", i))
        try:
            view.check(rnd, "final")
        except EquivocationError:
            return False
        posts = [v for a, k, v in view.entries("aggregate", rnd) if k == "aggregate"]
        if len(posts) != 1 or posts[0]["model"] != self.model_hash(model, keys):
            return False
        post = posts[0]
        if post.get("updated") and post.get("total") is not None:      # public sum check (MPC)
            dig = hashlib.sha256(b"bcfl-total|" + canon(post["total"])).digest()
            att = {m for a, (k, m), v in view.entries("verdict_attest", rnd) if k == "total" and a == m and v == dig}
            h, s = self._round_hs
            if len(att) < self.beacon.t or not sf.sum_check_total(np.array(post["sum"], dtype=np.int64), post["total"],
                                                                   h, s, self.k):
                return False
        return True

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
            agg = self._finish_round(mean_update, accepted, sims, ratios, models, keys, ref, n, None)
            self._post_aggregate(agg, keys, mean_update is not None)
            return agg

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

    @staticmethod
    def _zk_rows(keys, model, d):
        """Index sets (in the flattened update) of the class rows of the last layer: the weights
        w[c, :] and the bias b[c] of class c."""
        from secure.zk import rows_layout
        if len(keys) < 2:
            return []
        n_cls = _to_numpy(model[keys[-1]]).size
        n_w = _to_numpy(model[keys[-2]]).size
        return rows_layout(d, d - n_cls - n_w, n_cls, n_w // n_cls, d - n_cls)

    @staticmethod
    def _q_of(x_int, rows):
        x = np.asarray(x_int, dtype=object)
        return [int(sum(int(v) * int(v) for v in x[idx])) for idx in rows]

    @staticmethod
    def _profile_from_q(q):
        r = np.sqrt(np.asarray(q, dtype=np.float64))
        return r / (r.sum() + 1e-12)

    def _ensure_mpc(self, n, c):
        if self.mpc is None and self.screening in ("mpc", "mpc-check"):
            from mpc.screen_mpc import MPCScreening
            self.mpc = MPCScreening(n, self.k, c, self.collusion_window if self.collusion_check else 0,
                                    self.similarity_threshold, self.magnitude_ratio, self.class_mad,
                                    self.collusion_threshold, self.collusion_group,
                                    parties=self.beacon_n,
                                    observers=len(self.beacon.members) if self.relay is not None else 0,
                                    zk=self.secagg_zk).start()

    def _zk_mpc_hook(self, zr, outs, present, sketches, rnd):
        """Between the MPC's inputs and its screening: committee members attest what the MPC
        output for the binding, every present client posts its proof, and the proofs are checked.
        Returns the valid bits for the MPC."""
        from secure.signing import canon
        n = self.mpc.n
        self._zk_outs = {"coef": [int(c) for c in outs["coef"]], "V1": [int(v) for v in outs["V1"]],
                         "V2": [int(v) for v in outs["V2"]], "in_range": [bool(b) for b in outs["in_range"]]}
        if self.relay is not None:
            for o, m in enumerate(self.beacon.members):
                seen = self.mpc.last_observed_zk[o]
                dig = hashlib.sha256(b"bcfl-zkout|" + canon([seen["coef"], seen["V1"], seen["V2"],
                                                            [bool(b) for b in seen["in_range"]]])).digest()
                self._member_post("verdict_attest", rnd, m, ("zk", m), dig)
        valid = [False] * n
        self._zk_verified = {}
        for i in present:
            zr.prove_mpc(i, sketches[i], outs)
            ok = zr.verify_mpc(i, outs, self.k)
            self._zk_verified[i] = ok
            valid[i] = ok
            if not ok:
                zr.liars.setdefault(i, "proof does not verify")
        self._zk_invalid = {i for i in present if not valid[i]}
        return valid

    def _zk_late_check(self, zr, frnd, uploads, accepted, qualified):
        """After the b-shares are on the board: the upload relation of every accepted client and
        the audit of its pair commitments with every rejected client."""
        late = []
        rejected = [j for j in qualified if j not in set(accepted)]
        sk = {}
        for j in rejected:
            try:
                sk[j] = frnd.rebuild("sk", j)
            except RuntimeError:
                pass
        for i in accepted:
            b_i = frnd.rebuild("b", i)
            if not zr.relation_ok(i, uploads[i], b_i, qualified):
                zr.liars.setdefault(i, "upload does not match the committed update")
                late.append(i)
                continue
            bad = [j for j in sk if not zr.audit_ok(i, j, sk[j])]
            if bad:
                zr.liars.setdefault(i, f"pair commitment with rejected client {bad[0]} is wrong")
                late.append(i)
        return late

    def _abort_round(self, rnd, alarms, t0, why):
        """Phase 4: honest parties refused to go on; nothing from this round is used."""
        self.last_alarms = [str(a) for a in alarms]
        self.alarm_rounds += 1
        self._aborted.add(rnd)
        if self.mpc is not None:
            self.mpc.finish_round()
        self.last_sum_check = False
        self.sum_check_failures += 1
        print(f"[sketchSecAgg] round {rnd}: SERVER MISBEHAVIOUR DETECTED ({why}); "
              f"{len(alarms)} alarm(s), e.g. {alarms[0]}; round stopped")
        self.last_secagg_seconds = time.perf_counter() - t0
        return None, [], [], []

    def _hardened_round(self, models, keys, ref, d, n, weights, profiles, guessed_hs):
        """Phase 1 + 3 + 4 round: VSS setup -> anchor -> commit uploads -> beacon seed ->
        sketches -> screening -> signed accepted set (+ committee attestations) -> anchor ->
        holders reveal -> unmask -> sum check. Client and server steps are kept apart as in the
        other modes; with Phase 4 every reader goes through its own view of the relayed board."""
        t0 = time.perf_counter()
        self._ensure_board()
        self._ensure_phase4(n)
        rnd = self.round
        self.last_alarms, self.last_privacy_breach = [], None
        self._zk_outs, self._zk_round = None, None
        board = self.relay if self.relay is not None else self.board
        faults = self._share_faults(n)
        self._install_server_attack(rnd, n)
        self.beacon.run_open_phase(rnd)                 # beacon deals while uploads are open
        if self.secagg_vss:
            from .secaggVSS import VSSSecAggRound
            if self._box_keys is None:
                from secure.channel import KeyPair
                self._box_keys = {i: KeyPair() for i in range(n)}
            frnd = VSSSecAggRound(n, d, board, rnd, self.secagg_threshold, self._box_keys, faults,
                                  identities=self._ids["client"] if self._ids else None)
            qualified = frnd.setup()                    # deals, complaints, answers: on the board
        else:
            frnd = sf.FieldSecAggRound(n, d, self.secagg_threshold, faults)
            qualified = list(range(n))
        if self.relay is not None:                      # Phase 4: anchor setup; clients check it
            self.relay.anchor_phase(rnd, "setup")
            if self.server_checks:
                frnd.check_views("setup")
                if frnd.alarms:
                    return self._abort_round(rnd, frnd.alarms, t0, "clients' views of the setup differ from the anchor")
        absent = [i for i in range(n) if i not in qualified]
        self.last_vss_rejected = absent
        self.vss_rejections += len(absent)
        rows = self._zk_rows(keys, models[0], d) if (self.class_check or self.secagg_zk) else []
        n_cls = len(rows) if self.class_check else 0
        self._ensure_mpc(n, n_cls)
        zr = None
        self.last_zk_rejected, self.last_zk_late, self._zk_invalid, self._zk_hook = [], [], set(), None
        if self.secagg_zk:                              # Phase 5
            from .zkSecAgg import ZKRound
            zf = {}
            if self.c1_attack == "pairlie":
                honest_q = [j for j in qualified if j not in set(getattr(self, "malicious", []))]
                zf["pair_lie"] = {(i, honest_q[0]) for i in qualified if self._attacking(i) and honest_q}
            zr = ZKRound(n, d, rows, board, rnd, frnd, self._ids["client"] if self._ids else None,
                         self.mpc.prime if self.mpc is not None else None, zf)
        uploads, x_ints, cover_ints, zdig = {}, {}, {}, {}
        lying = set()
        for i in qualified:                             # client side, before any seed exists
            u_true = _flatten(models[i], keys) - ref
            u, u_sk = self._adversary(i, u_true, guessed_hs)
            w = 1.0 if weights is None else weights[i]
            x_ints[i] = sf.encode_int(w * u)
            cover_ints[i] = sf.encode_int(w * u_sk)
            uploads[i] = frnd.client_upload_int(i, x_ints[i])
            if self._attacking(i) and self.c1_attack in ("sketchlie", "commitlie", "pairlie"):
                lying.add(i)
            if zr is not None:                          # commit to the update (a liar: to its cover)
                committed = cover_ints[i] if (i in lying and self.c1_attack in ("commitlie", "pairlie")) else x_ints[i]
                zdig[i] = zr.commit(i, committed, x_ints[i], qualified)

        def commitment(i):
            if i not in uploads:                        # left out by VSS: nothing uploaded
                return hashlib.sha256(b"bcfl-absent|" + int(rnd).to_bytes(8, "big") + int(i).to_bytes(4, "big")).digest()
            base = frnd.upload_hash(i, uploads[i]) if self.secagg_vss else sf.upload_hash(rnd, i, uploads[i])
            return hashlib.sha256(base + zdig[i]).digest() if i in zdig else base
        hashes = [commitment(i) for i in range(n)]
        sigs = None
        if self._ids is not None:                       # Phase 4: each client signs its own commitment
            sigs = [frnd.upload_sig(i, hashes[i]) if i in uploads else None for i in range(n)]
        board.commit_uploads(rnd, hashes, sigs)         # uploads fixed; round closed
        if self.relay is not None:
            self.relay.anchor_phase(rnd, "uploads")
        try:
            seed = self.beacon.run_closed_phase(rnd)    # only now is the sketch defined
        except RuntimeError as e:
            if self.beacon.alarms:
                alarms, self.beacon.alarms = self.beacon.alarms, []
                return self._abort_round(rnd, [a for _, a in alarms], t0, f"committee refused to reveal ({e})")
            raise
        h, s = sf.sketch_params_from_seed(seed, d, self.k)
        sketches, q_ints = [], []
        for i in range(n):                              # client side: report a sketch (+ class input)
            if i not in uploads:
                sketches.append(np.zeros(self.k, dtype=np.int64))
                q_ints.append([0] * len(rows))
                continue
            honest = self.c1_attack == "nullspace" or not self._attacking(i)
            rep = x_ints[i] if honest else cover_ints[i]
            sketches.append(sf.sketch_int(rep, h, s, self.k))
            q_ints.append(self._q_of(rep, rows) if self.class_check else [])
        self._q_ints = q_ints
        profiles = [self._profile_from_q(q) if i in uploads else np.full(len(rows), 1.0 / len(rows))
                    for i, q in enumerate(q_ints)] if self.class_check else profiles
        zk_out = set()
        if zr is not None:                              # Phase 5: before screening
            tz = time.perf_counter()
            zr.set_round_params(seed, (h, s))
            for i in zr.pair_check(qualified):
                zk_out.add(i)
            present = [i for i in qualified if i not in zk_out]
            if self.mpc is None:                        # plaintext screening: proofs checked now
                for i in present:
                    zr.prove_plain(i, sketches[i], (h, s))
                    sec = zr._sec[i]                    # the client hands q and its openings to the server
                    if not zr.verify_plain(i, sketches[i], q_ints[i], sec["r_q"]):
                        zk_out.add(i)
                        zr.liars.setdefault(i, "proof does not verify")
            else:                                       # MPC: proofs checked between input and screening
                self._zk_hook = lambda outs, zr=zr, present=present, sk=sketches: self._zk_mpc_hook(
                    zr, outs, present, sk, rnd)
            self.last_zk_seconds = time.perf_counter() - tz
        self._zk_round = zr
        # server side: check uploads against the board, screen, unmask, check the sum
        posted = board.upload_commitments(rnd)
        for i in qualified:
            if commitment(i) != posted[i]:
                raise RuntimeError(f"round {rnd}: client {i}'s upload does not match its commitment")
        wts = [1.0] * n if weights is None else [max(w, 1e-12) for w in weights]
        float_sketches = [sf.decode_int(sketches[i]) / wts[i] for i in range(n)]
        self._verdict_bits = None
        accepted, sims, ratios = self._decide(float_sketches, profiles, n, sketches, weights,
                                              absent=set(absent) | zk_out)
        if zr is not None:
            self.last_zk_rejected = sorted(zk_out | self._zk_invalid)
            self.zk_rejections += len(self.last_zk_rejected)
        self.last_update_hashes = hashes
        self.last_sum_check = None
        self.last_bad_reveals = []

        if self.relay is not None:                      # Phase 4: outcome, attestations, anchor
            from secure.signing import canon
            bits = self._verdict_bits
            post = {"accepted": sorted(accepted), "present": sorted(qualified),
                    "verdicts": None if bits is None else [int(b) for b in bits]}
            if getattr(self, "_zk_outs", None) is not None:
                post["zk"] = self._zk_outs
            if bits is not None:                        # committee members attest what they received
                for o, m in enumerate(self.beacon.members):
                    seen = [int(x) for x in self.mpc.last_observed[o]]
                    dig = hashlib.sha256(b"bcfl-verdicts|" + canon(seen)).digest()
                    self._member_post("verdict_attest", rnd, m, ("verdicts", m), dig)
            if self.server_attack == "fake_accept":     # accept a client the screening rejected
                rej = [i for i in qualified if i not in accepted]
                if rej:
                    post["accepted"] = sorted(accepted + rej[:1])
                    accepted = post["accepted"]
            self._server_post("round_accepted", rnd, "accepted", post)
            anchor_view = "server"
            victim = None
            if self.server_attack == "split_accepted":
                victim = self._victim(n, among=accepted)
                if victim is not None:
                    other = dict(post, accepted=[i for i in post["accepted"] if i != victim])
                    self._server_post("round_accepted", rnd, "accepted", other)
                    mal = set(getattr(self, "malicious", []))
                    honest = [j for j in range(n) if j not in mal and j != victim]
                    group_b = honest[len(honest) // 2:]
                    for j in range(n):                  # everyone else (incl. colluders) sees version A
                        self.relay.show_version(("client", j), "round_accepted", rnd, 1 if j in group_b else 0)
                    anchor_view = ("client", honest[0])
            self.relay.anchor_phase(rnd, "outcome", as_seen_by=anchor_view)

        mean = None
        if accepted:
            try:
                if self.secagg_vss:
                    validate = (self._attestation_validator(rnd, n, qualified)
                                if self.relay is not None and self.server_checks else None)
                    frnd.holders_reveal(accepted, validate=validate)   # shares go on the board
                    if self.server_attack == "split_accepted" and victim is not None:
                        rec = self._try_recover(frnd, victim, uploads, set(getattr(self, "malicious", [])))
                        self.last_privacy_breach = rec is not None and np.array_equal(rec, x_ints[victim])
                        self.privacy_breaches += int(self.last_privacy_breach)
                        if self.last_privacy_breach:
                            print(f"[sketchSecAgg] round {rnd}: PRIVACY BREACH: the server rebuilt both secrets "
                                  f"of client {victim} and recovered its individual update")
                    if frnd.alarms and self.server_checks:
                        return self._abort_round(rnd, frnd.alarms, t0, "holders refused to reveal")
                    late = self._zk_late_check(zr, frnd, uploads, accepted, qualified) if zr is not None else []
                    if late:                            # Phase 5: proven liars leave the aggregate
                        self._server_post("zk_liar", rnd, "liars", {"liars": late,
                                          "why": {str(i): zr.liars.get(i, "") for i in late}}) \
                            if self._ids is not None else None
                        verdict = {i: True for i in late}
                        frnd.holders_reveal_liars(late, check=lambda j, i, v=verdict: v.get(i, False))
                        accepted = [i for i in accepted if i not in late]
                        self.last_zk_late = late
                        self.zk_rejections += len(late)
                        print(f"[sketchSecAgg] round {rnd}: clients {late} were accepted but their upload does "
                              f"not match what they proved ({[zr.liars[i] for i in late]}); removed from the aggregate")
                    if not accepted:
                        raise RuntimeError("no client left after removing proven liars")
                    sum_int = frnd.server_unmask_sum_int(uploads, accepted, liars=late)
                    self.last_bad_reveals = sorted(frnd.bad_reveals)
                else:
                    sum_int = frnd.server_unmask_sum_int(uploads, accepted)
            except RuntimeError as e:                   # Phase 1: a wrong share can make unmasking fail
                self.unmask_failures += 1
                self.last_sum_check = False
                self.sum_check_failures += 1
                print(f"[sketchSecAgg] round {rnd}: UNMASKING FAILED ({e}); round discarded")
                sum_int = None
            if sum_int is not None:
                self._last_sum_int = sum_int
                if self.screening in ("mpc", "mpc-check"):
                    # the committee reveals only the sum of the accepted sketches
                    total = self.mpc.sum_accepted(accepted)
                    self._attest_total = total
                    if self.relay is not None:
                        from secure.signing import canon
                        for o, m in enumerate(self.beacon.members):
                            seen = [int(x) for x in self.mpc.last_observed_total[o]]
                            dig = hashlib.sha256(b"bcfl-total|" + canon(seen)).digest()
                            self._member_post("verdict_attest", rnd, m, ("total", m), dig)
                    self.last_sum_check = sf.sum_check_total(sum_int, total, h, s, self.k)
                    self.last_mpc_seconds = self.mpc.last_seconds
                else:
                    self.last_sum_check = sf.sum_check(sum_int, sketches, accepted, h, s, self.k)
                if self.last_sum_check:
                    denom = len(accepted) if weights is None else sum(weights[i] for i in accepted)
                    self._last_denom = denom
                    mean = sf.decode_int(sum_int) / denom
                else:
                    self.sum_check_failures += 1
                    print(f"[sketchSecAgg] round {rnd}: SUM CHECK FAILED: the accepted sketches do not add up "
                          f"to the sketch of the unmasked sum; round discarded, previous global model kept")
            board.post("round_outcome", rnd, "server", "sum_check", bool(self.last_sum_check),
                       *(() if self._ids is None else (self._ids["server"].sign_post(
                           "round_outcome", rnd, "server", "sum_check", bool(self.last_sum_check)),)))
        self._round_hs = (h, s)
        if self.mpc is not None:
            self.mpc.finish_round()                     # unmasking failed: close the MPC round too
        if self.last_bad_reveals:
            print(f"[sketchSecAgg] round {rnd}: holders {self.last_bad_reveals} returned shares that fail "
                  f"the VSS check; their shares were not used")
        self.last_secagg_seconds = time.perf_counter() - t0
        logger.info(f"[sketchSecAgg] round {rnd}: hardened round {self.last_secagg_seconds:.2f}s "
                    f"(n={n}, d={d}, threshold {frnd.t}, vss {self.secagg_vss}, left out {absent}, "
                    f"beacon seed {seed.hex()[:16]}..., sum check {self.last_sum_check})")
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
            if self.secagg == "hardened" and self.last_vss_rejected:
                extra += f", failed VSS: {self.last_vss_rejected}"
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
