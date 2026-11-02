#!/usr/bin/env bash
# C1 with real secure aggregation (--secagg dh) next to the simulated-mask "ours" runs of the
# main table. Same settings and file naming, label "oursdh", so compare_results.py shows both.
# Skips any run whose result file already exists.
# Usage: bash run_c1_dh.sh "1" [rounds]      (8 runs per seed)
set -e
SEEDS=${1:-"1"}
ROUNDS=${2:-30}
mkdir -p results/main
for SEED in $SEEDS; do
  for TAG in iid noniid05; do
    for ATT in none signflip labelflip backdoor; do
      NM=3; [ "$ATT" = "none" ] && NM=0
      OUT=results/main/FashionMNIST_oursdh_${ATT}${NM}_${TAG}_seed${SEED}.csv
      if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
      SPLIT=""; [ "$TAG" = "noniid05" ] && SPLIT="--non-iid --alpha 0.5"
      python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test $SPLIT \
                     --attack $ATT --num-malicious $NM --results $OUT \
                     --aggregator repsketch --sketch-dim 64 --class-check --secagg dh
    done
  done
done
python3 compare_results.py results/main
