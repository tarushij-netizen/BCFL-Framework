from ..base.baseAggregator import ServerAggregator
import numpy as np


class similarityAggregator(ServerAggregator):
    """
    Leash-FL-inspired similarity-screening aggregator.

    Motivation (from the literature review): Krum and the existing
    'balance' aggregator screen clients using raw *distance* to a reference
    model. This aggregator instead screens using *cosine similarity* -- the
    angle between a client's update and the reference update -- which is
    the core idea behind Leash-FL's poisoning/free-rider screening step.
    A client whose update points in a very different direction from the
    consensus is flagged and excluded before averaging.

    Difference from balance.py: the reference model is refreshed after
    every round (see _on_after_aggregation), so it tracks the evolving
    global model rather than always comparing against the very first
    initial model.

    :param similarity_threshold: minimum cosine similarity (range -1 to 1)
        a client's update must have with the reference model to be
        accepted. Start around 0.5-0.7 and tune based on observed values
        (print statements below show the actual similarity per client).
    :param init_model: the initial global model (state_dict-like), used as
        the reference for round 0.
    """

    def __init__(self, similarity_threshold, init_model):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.reference_model = init_model
        self.rejected_last_round = []  # client indices rejected in the most recent round

    def _on_before_aggregation(self, raw_client_model_or_grad_list=None):
        pass

    def _on_after_aggregation(self, aggregated_model_or_grad=None):
        if aggregated_model_or_grad is not None:
            self.reference_model = aggregated_model_or_grad
        return aggregated_model_or_grad

    def test(self, test_data=None, device=None, args=None):
        pass

    @staticmethod
    def _to_numpy(value):
        """Convert a torch tensor OR a plain numpy array/list to a numpy array."""
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.array(value)

    @classmethod
    def _flatten(cls, state_dict_like):
        """Flatten a model state_dict (dict of tensors/arrays) into one 1D numpy vector."""
        pieces = [cls._to_numpy(v).flatten() for v in state_dict_like.values()]
        return np.concatenate(pieces)

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    def _aggregate_alg(self, raw_client_model_or_grad_list=None, t=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool

        if self.reference_model is None:
            # First round, nothing to compare against yet -- accept everyone.
            self.reference_model = raw_client_model_or_grad_list[0]

        ref_vec = self._flatten(self.reference_model)

        accepted = []
        self.rejected_last_round = []
        for idx, model in enumerate(raw_client_model_or_grad_list):
            client_vec = self._flatten(model)
            sim = self._cosine_similarity(client_vec, ref_vec)
            if sim >= self.similarity_threshold:
                accepted.append(model)
            else:
                self.rejected_last_round.append(idx)
                print(f"[similarityAggregator] Client {idx} REJECTED: "
                      f"cosine similarity {sim:.4f} < threshold {self.similarity_threshold}")

        if not accepted:
            print("[similarityAggregator] All clients rejected this round -- "
                  "falling back to previous reference model.")
            return self.reference_model

        # Average the accepted models, parameter-by-parameter.
        keys = accepted[0].keys()
        aggregated_model = {}
        for key in keys:
            values = np.stack([self._to_numpy(model[key]) for model in accepted])
            aggregated_model[key] = np.mean(values, axis=0)

        return aggregated_model
