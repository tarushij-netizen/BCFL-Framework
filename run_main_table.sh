#!/usr/bin/env bash
# Main paper table: every method x attack x data split, for several seeds.
# Skips any run whose result file already exists, so it can be stopped and restarted.
# Usage: bash run_main_table.sh "1 2 3" [rounds]
set -e
SEEDS=${1:-"1 2 3"}
ROUNDS=${2:-30}
mkdir -p results/main
run () {  # $1 label, $2 attack, $3 split tag, $4 seed, rest = flags
  local LABEL=$1 ATT=$2 TAG=$3 SEED=$4; shift 4
  local NM=3; [ "$ATT" = "none" ] && NM=0
  local OUT=results/main/FashionMNIST_${LABEL}_${ATT}${NM}_${TAG}_seed${SEED}.csv
  if [ -f "$OUT" ]; then echo "skip $OUT"; return; fi
  local SPLIT=""; [ "$TAG" = "noniid05" ] && SPLIT="--non-iid --alpha 0.5"
  python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test $SPLIT \
                 --attack $ATT --num-malicious $NM --results $OUT "$@"
}
for SEED in $SEEDS; do
  for TAG in iid noniid05; do
    run fedavg none $TAG $SEED --aggregator fedavg
    run ours   none $TAG $SEED --aggregator repsketch --sketch-dim 64 --class-check
    for ATT in signflip labelflip backdoor; do
      run fedavg     $ATT $TAG $SEED --aggregator fedavg
      run median     $ATT $TAG $SEED --aggregator median
      run krum       $ATT $TAG $SEED --aggregator krum
      run similarity $ATT $TAG $SEED --aggregator similarity
      run ours       $ATT $TAG $SEED --aggregator repsketch --sketch-dim 64 --class-check
    done
  done
done
python3 compare_results.py results/main
