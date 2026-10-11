#!/usr/bin/env bash
# C1: screening on sketches under (simulated) secure aggregation vs plaintext screening.
# 30% attackers (3/10). Usage: bash run_c1_experiments.sh [rounds] [seed]
set -e
ROUNDS=${1:-30}
SEED=${2:-1}
COMMON="--benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test"
python test.py $COMMON --aggregator sketch --attack none --sketch-dim 64            # cost of C1 without attack
for ATTACK in signflip labelflip backdoor; do
  python test.py $COMMON --aggregator similarity --attack $ATTACK --num-malicious 3   # plaintext (Leash-FL-style)
  python test.py $COMMON --aggregator sketch     --attack $ATTACK --num-malicious 3 --sketch-dim 64
done
for K in 16 256; do                                                                  # how small can the sketch be?
  python test.py $COMMON --aggregator sketch --attack signflip --num-malicious 3 --sketch-dim $K
done
python3 compare_results.py
