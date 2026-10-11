#!/usr/bin/env bash
# Leash-FL-style attacks: label-flipping and backdoor, 30% malicious (3/10), IID and non-IID.
# Usage: bash run_leash_attacks.sh [rounds] [seed]
set -e
ROUNDS=${1:-30}
SEED=${2:-1}
COMMON="--benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test"
for DATA in "" "--non-iid --alpha 0.5"; do
  for ATTACK in labelflip backdoor; do
    for AGG in fedavg similarity median krum; do
      TAG=$( [ -z "$DATA" ] && echo iid || echo noniid05 )
      python test.py $COMMON $DATA --aggregator $AGG --attack $ATTACK --num-malicious 3 \
                     --results results/FashionMNIST_${AGG}_${ATTACK}3_${TAG}_seed${SEED}.csv
    done
  done
done
python3 compare_results.py
