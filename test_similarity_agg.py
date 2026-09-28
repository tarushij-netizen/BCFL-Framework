"""
Standalone sanity test for similarityAggregator -- no PyTorch, no blockchain,
just numpy dicts shaped like a model state_dict, so this runs instantly.

Simulates:
  - 8 "honest" clients: small random perturbations of a shared true direction
  - 2 "malicious" clients: random noise, uncorrelated with the true direction
        (simulating a poisoning/free-rider attack)

Success criteria:
  1. Both malicious clients get REJECTED (their index printed as rejected)
  2. The final aggregated model is close to the honest clients' average,
     NOT pulled toward the malicious clients' noise
"""
import numpy as np
import sys
import types
import importlib.util

# Stub out the real package chain (which transitively imports torch) so this
# test can run standalone. This does NOT change similarityAgg.py's real code
# -- it only provides a minimal stand-in for its one dependency, ServerAggregator,
# so the actual aggregator logic below is exactly what's in the real file.
fake_base_pkg = types.ModuleType("server.base.baseAggregator")

class ServerAggregator:
    def __init__(self, model=None, args=None):
        self.model = model
        self.id = 0
        self.args = args
        self.model_pool = []

fake_base_pkg.ServerAggregator = ServerAggregator
sys.modules["server"] = types.ModuleType("server")
sys.modules["server.base"] = types.ModuleType("server.base")
sys.modules["server.base.baseAggregator"] = fake_base_pkg
sys.modules["server.aggregation_alg"] = types.ModuleType("server.aggregation_alg")

spec = importlib.util.spec_from_file_location(
    "server.aggregation_alg.similarityAgg", "server/aggregation_alg/similarityAgg.py"
)
similarityAgg_module = importlib.util.module_from_spec(spec)
sys.modules["server.aggregation_alg.similarityAgg"] = similarityAgg_module
spec.loader.exec_module(similarityAgg_module)
similarityAggregator = similarityAgg_module.similarityAggregator

np.random.seed(42)

# A "true" direction that honest clients are all roughly aligned with,
# shaped like a tiny 2-layer model's state_dict.
true_layer1 = np.random.randn(20)
true_layer2 = np.random.randn(10)
reference_model = {"layer1.weight": true_layer1, "layer2.weight": true_layer2}

client_models = []
labels = []

# 8 honest clients: same direction as reference + small noise
for i in range(8):
    client_models.append({
        "layer1.weight": true_layer1 + np.random.normal(0, 0.05, size=20),
        "layer2.weight": true_layer2 + np.random.normal(0, 0.05, size=10),
    })
    labels.append("honest")

# 2 malicious clients: pure random noise, uncorrelated with true direction
for i in range(2):
    client_models.append({
        "layer1.weight": np.random.randn(20) * 3,
        "layer2.weight": np.random.randn(10) * 3,
    })
    labels.append("malicious")

print("Client setup:")
for idx, label in enumerate(labels):
    print(f"  Client {idx}: {label}")
print()

agg = similarityAggregator(similarity_threshold=0.5, init_model=reference_model)
result = agg._aggregate_alg(client_models)

print()
print("Rejected client indices:", agg.rejected_last_round)
print("Expected malicious indices: [8, 9]")
rejected_correctly = set(agg.rejected_last_round) == {8, 9}
print("PASS -- exactly the malicious clients were rejected" if rejected_correctly
      else "FAIL -- rejection did not match expected malicious clients")

# Check the aggregated result is close to the honest-only average,
# not pulled toward the malicious noise.
honest_only_avg_l1 = np.mean([client_models[i]["layer1.weight"] for i in range(8)], axis=0)
diff_from_honest_avg = np.linalg.norm(result["layer1.weight"] - honest_only_avg_l1)
print(f"\nDistance between aggregated result and honest-only average: {diff_from_honest_avg:.6f}")
print("PASS -- aggregated result matches honest-only average closely" if diff_from_honest_avg < 0.1
      else "FAIL -- aggregated result diverges from honest average (malicious clients may have leaked in)")
