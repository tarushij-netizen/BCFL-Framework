"""
similarityAggregator (v2) -- Leash-FL-inspired screening aggregator.

Screens each client's UPDATE (client model - previous global model) against the
coordinate-wise median update of the round, on two signals:
  1. direction : cosine(update_i, median_update) >= similarity_threshold
  2. magnitude : ||update_i|| / median_j ||update_j|| <= magnitude_ratio
Accepted client models are averaged; rejected ones are dropped.

Screening starts in round 0, using the initial global model passed in via
set_global_model() as the reference. (A warm-up round without screening would let
attackers into the very first average, so warmup_rounds defaults to 0.)

This approximates ONE component of Leash-FL (similarity screening). It does not
reproduce Leash-FL's certificateless ECC auth, pseudonym rotation, CRT rekeying,
or blockchain checkpoint/rollback.

Works with torch state_dicts (OrderedDict[str, Tensor]) and with plain dicts of
numpy arrays (used by the standalone unit test, which does not need torch).
"""
import logging
from collections import OrderedDict

import numpy as np

from ..base.baseAggregator import ServerAggregator

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# helpers (work for torch tensors and numpy arrays)
# ----------------------------------------------------------------------------
def _is_tensor(v):
    return hasattr(v, "detach") and hasattr(v, "clone")


def _to_numpy(v):
    if _is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _is_float(v):
    if _is_tensor(v):
        return v.is_floating_point()
    return np.issubdtype(np.asarray(v).dtype, np.floating)


def _float_keys(state_dict):
    """Keys of float parameters/buffers (excludes e.g. BatchNorm num_batches_tracked)."""
    return [k for k, v in state_dict.items() if _is_float(v)]


def _flatten(state_dict, keys):
    return np.concatenate([_to_numpy(state_dict[k]).astype(np.float64).ravel() for k in keys])


def _copy(v):
    if _is_tensor(v):
        return v.detach().clone()
    return np.array(v, copy=True)


def snapshot_state_dict(state_dict):
    """Deep copy of a state_dict. model.state_dict() returns live references to the
    parameters, so without a copy every pooled entry changes when training continues."""
    return OrderedDict((k, _copy(v)) for k, v in state_dict.items())


def _average(models):
    """Mean of float entries; non-float entries (int buffers) copied from the first model."""
    first = models[0]
    out = OrderedDict()
    for k, v in first.items():
        if _is_float(v):
            acc = _copy(v)
            for m in models[1:]:
                acc = acc + m[k]
            out[k] = acc / len(models)
        else:
            out[k] = _copy(v)
    return out


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------------
# attack simulator
# ----------------------------------------------------------------------------
def poison_state_dict(local, prev_global, attack, scale=5.0, rng=None):
    """Return a poisoned COPY of `local`.

    signflip : prev - scale * (local - prev)
    noise    : prev + random Gaussian direction with norm scale * ||local - prev||
    freerider: prev unchanged (client sends back the global model, no training)
    Non-float entries are copied unchanged.
    """
    rng = np.random.default_rng() if rng is None else rng
    keys = _float_keys(local)
    out = snapshot_state_dict(local)
    if attack == "freerider":
        for k in keys:
            out[k] = _copy(prev_global[k])
        return out
    if attack == "signflip":
        for k in keys:
            out[k] = prev_global[k] - scale * (local[k] - prev_global[k])
        return out
    if attack == "noise":
        upd_norm = np.linalg.norm(_flatten(local, keys) - _flatten(prev_global, keys))
        g = {k: rng.standard_normal(_to_numpy(local[k]).shape) for k in keys}
        g_norm = np.sqrt(sum(float((x ** 2).sum()) for x in g.values()))
        factor = 0.0 if g_norm == 0 else scale * upd_norm / g_norm
        for k in keys:
            delta = (factor * g[k]).astype(_to_numpy(local[k]).dtype)
            if _is_tensor(local[k]):
                import torch
                delta = torch.as_tensor(delta, device=local[k].device)
            out[k] = prev_global[k] + delta
        return out
    raise ValueError(f"Unknown attack '{attack}' (use signflip, noise or freerider)")


# ----------------------------------------------------------------------------
# aggregators
# ----------------------------------------------------------------------------
class _PoolMixin:
    """Fixes for the base ServerAggregator when used in the real Task loop:
    - model_pool is reset every round (base class keeps appending forever)
    - uploads are snapshotted (base class stores live references)
    """

    def receive_upload(self, client_pool):
        self.model_pool = [snapshot_state_dict(c.get_model_state_dict()) for c in client_pool]

    def set_global_model(self, state_dict):
        """Called by Task before round 0 with the initial global model."""
        self.reference_model = snapshot_state_dict(state_dict)

    def _on_before_aggregation(self, raw_client_model_or_grad_list=None):
        return raw_client_model_or_grad_list

    def _on_after_aggregation(self, aggregated_model_or_grad=None):
        return aggregated_model_or_grad

    def test(self, test_data=None, device=None, args=None):
        return None


class cleanFedavgAggregator(_PoolMixin, ServerAggregator):
    """Plain unweighted mean, with the per-round pool reset."""

    def __init__(self, model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        self.reference_model = None
        self.rejected_last_round = []

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        models = raw_client_model_or_grad_list or self.model_pool
        agg = _average(models)
        self.reference_model = snapshot_state_dict(agg)
        return agg


class similarityAggregator(_PoolMixin, ServerAggregator):
    def __init__(self, similarity_threshold=0.1, magnitude_ratio=3.0, warmup_rounds=0,
                 model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        self.similarity_threshold = similarity_threshold
        self.magnitude_ratio = magnitude_ratio
        self.warmup_rounds = warmup_rounds
        self.reference_model = None if model is None else snapshot_state_dict(model)
        self.round = 0
        self.rejected_last_round = []
        self.last_similarities = []
        self.last_norm_ratios = []

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        models = raw_client_model_or_grad_list or self.model_pool
        n = len(models)

        # warm-up: plain average, no screening
        if self.reference_model is None or self.round < self.warmup_rounds:
            agg = _average(models)
            self.rejected_last_round, self.last_similarities, self.last_norm_ratios = [], [], []
            logger.info(f"[similarityAgg] round {self.round}: warm-up, plain average of {n} clients")
            self.reference_model = snapshot_state_dict(agg)
            self.round += 1
            return agg

        keys = _float_keys(models[0])
        ref = _flatten(self.reference_model, keys)
        updates = np.stack([_flatten(m, keys) - ref for m in models])
        consensus = np.median(updates, axis=0)
        norms = np.linalg.norm(updates, axis=1)
        med_norm = float(np.median(norms))

        sims = [cosine_similarity(u, consensus) for u in updates]
        ratios = [float(nm / med_norm) if med_norm > 0 else 0.0 for nm in norms]
        accepted = [i for i in range(n)
                    if sims[i] >= self.similarity_threshold and ratios[i] <= self.magnitude_ratio]
        rejected = [i for i in range(n) if i not in accepted]

        self.last_similarities, self.last_norm_ratios = sims, ratios
        self.rejected_last_round = rejected
        logger.info(f"[similarityAgg] round {self.round}: cos={np.round(sims, 3).tolist()} "
                    f"ratio={np.round(ratios, 2).tolist()} rejected={rejected}")
        if rejected:
            print(f"[similarityAgg] round {self.round}: rejected clients (0-based index) {rejected}")

        if not accepted:
            print(f"[similarityAgg] round {self.round}: all clients rejected, keeping previous global model")
            agg = snapshot_state_dict(self.reference_model)
            # keep int buffers from the uploads so load_state_dict still works
            for k, v in models[0].items():
                if k not in agg:
                    agg[k] = _copy(v)
        else:
            agg = _average([models[i] for i in accepted])

        self.reference_model = snapshot_state_dict(agg)
        self.round += 1
        return agg
