"""
Comparison aggregators + the attack wrapper used for the defense experiments.

Why new Krum/median classes: the upstream krum.py and median.py do not work on
real torch state_dicts (median.py calls np.median on a list of tensors and gets
ONE scalar per layer; krum.py builds np.array from tensors of different shapes).
They are left untouched; these versions work on flattened updates instead.
"""
import logging

import numpy as np

from .similarityAgg import (_PoolMixin, _flatten, _float_keys, _to_numpy,
                            poison_state_dict, snapshot_state_dict,
                            cleanFedavgAggregator, similarityAggregator)
from ..base.baseAggregator import ServerAggregator

logger = logging.getLogger(__name__)


def _unflatten_like(vec, template, keys):
    """Write a flat float vector back into a copy of `template` (same keys/shapes/types)."""
    out = snapshot_state_dict(template)
    pos = 0
    for k in keys:
        shape = tuple(np.shape(_to_numpy(template[k])))
        size = int(np.prod(shape)) if shape else 1
        chunk = vec[pos:pos + size].reshape(shape)
        pos += size
        if hasattr(template[k], "detach"):
            import torch
            out[k] = torch.as_tensor(chunk, dtype=template[k].dtype, device=template[k].device)
        else:
            out[k] = chunk.astype(np.asarray(template[k]).dtype)
    return out


class medianTorchAggregator(_PoolMixin, ServerAggregator):
    """Coordinate-wise median of client models (Yin et al., 2018)."""

    def __init__(self, model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        self.reference_model = None
        self.rejected_last_round = []

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        models = raw_client_model_or_grad_list or self.model_pool
        keys = _float_keys(models[0])
        med = np.median(np.stack([_flatten(m, keys) for m in models]), axis=0)
        agg = _unflatten_like(med, models[0], keys)
        self.reference_model = snapshot_state_dict(agg)
        return agg


class krumTorchAggregator(_PoolMixin, ServerAggregator):
    """Krum (Blanchard et al., 2017): pick the client whose n-f-2 nearest
    neighbours are closest. f = assumed number of attackers."""

    def __init__(self, num_byzantine=None, model=None, args=None):
        ServerAggregator.__init__(self, model, args)
        self.num_byzantine = num_byzantine
        self.reference_model = None
        self.rejected_last_round = []

    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        models = raw_client_model_or_grad_list or self.model_pool
        n = len(models)
        f = self.num_byzantine if self.num_byzantine is not None else n // 4
        keys = _float_keys(models[0])
        X = np.stack([_flatten(m, keys) for m in models])
        sq = (X ** 2).sum(1)
        d = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)
        k = max(n - f - 2, 1)
        scores = [np.sort(np.delete(d[i], i))[:k].sum() for i in range(n)]
        sel = int(np.argmin(scores))
        self.rejected_last_round = [i for i in range(n) if i != sel]
        logger.info(f"[krum] selected client index {sel}")
        agg = snapshot_state_dict(models[sel])
        self.reference_model = snapshot_state_dict(agg)
        return agg


AGGREGATORS = {
    "fedavg": cleanFedavgAggregator,
    "similarity": similarityAggregator,
    "median": medianTorchAggregator,
    "krum": krumTorchAggregator,
}


def make_server_class(aggregator="fedavg", attack="none", malicious=(), attack_scale=5.0,
                      seed=0, agg_kwargs=None):
    """Return an aggregator CLASS (Task calls it with no arguments) that
    1. snapshots uploads each round,
    2. replaces the uploads of `malicious` client indices (0-based) with poisoned ones
       built from the previous global model,
    3. runs the chosen aggregator,
    4. records which clients it rejected, for detection metrics.
    """
    base = AGGREGATORS[aggregator]
    agg_kwargs = dict(agg_kwargs or {})
    malicious = sorted(set(malicious))

    class AttackedServer(base):
        def __init__(self):
            base.__init__(self, **agg_kwargs)
            self.attack = attack
            self.malicious = malicious
            self.attack_scale = attack_scale
            self._rng = np.random.default_rng(seed)
            self._prev_global = None

        def set_global_model(self, state_dict):
            base.set_global_model(self, state_dict)
            self._prev_global = snapshot_state_dict(state_dict)

        def receive_upload(self, client_pool):
            base.receive_upload(self, client_pool)
            if self.attack == "none" or not self.malicious:
                return
            if self._prev_global is None:
                raise RuntimeError("set_global_model() must be called before round 0")
            for i in self.malicious:
                if i < len(self.model_pool):
                    self.model_pool[i] = poison_state_dict(self.model_pool[i], self._prev_global,
                                                           self.attack, self.attack_scale, self._rng)

        def aggregate(self, raw_client_model_or_grad_list=None):
            agg = base.aggregate(self, raw_client_model_or_grad_list)
            self._prev_global = snapshot_state_dict(agg)
            return agg

    AttackedServer.__name__ = f"{aggregator}_{attack}"
    return AttackedServer
