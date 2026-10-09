#!/usr/bin/env bash
# Defense comparison on FashionMNIST: 10 clients, first 2 are attackers.
# Run from the repo root with the venv active:  bash run_defense_experiments.sh [rounds] [seed]
# Results: results/*.csv  ->  summarise with: python3 compare_results.py
set -e
ROUNDS=${1:-50}
SEED=${2:-1}
COMMON="--benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test"

python test.py $COMMON --aggregator fedavg     --attack none
for ATTACK in signflip noise; do
  for AGG in fedavg similarity median krum; do
    python test.py $COMMON --aggregator $AGG --attack $ATTACK --num-malicious 2
  done
done
python3 compare_results.py
