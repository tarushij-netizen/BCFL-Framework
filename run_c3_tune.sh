#!/usr/bin/env bash
# Reduce honest false rejections in non-IID while keeping attackers out.
# Non-IID (alpha 0.5) label-flip, 3/10 attackers, repsketch k64 + class check, 30 rounds.
# Baseline (penalty 0.5, class-mad 3) already exists in results/c3hard.
# Usage: bash run_c3_tune.sh "1 2 3" [rounds]      (~5 min per run, 9 runs)
set -e
SEEDS=${1:-"1 2 3"}
ROUNDS=${2:-30}
OUTDIR=results/c3tune
mkdir -p $OUTDIR
run () {  # $1 label, $2 seed, rest = flags
  local LABEL=$1 SEED=$2; shift 2
  local OUT=$OUTDIR/${LABEL}_seed${SEED}.csv
  if [ -f "$OUT" ]; then echo "skip $OUT"; return; fi
  echo "=== $LABEL seed $SEED ==="
  python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test \
                 --non-iid --alpha 0.5 --attack labelflip --num-malicious 3 \
                 --aggregator repsketch --sketch-dim 64 --class-check --results $OUT "$@"
}
for SEED in $SEEDS; do
  run pen0.7_mad3   $SEED --rep-penalty 0.7  --class-mad 3
  run pen0.85_mad3  $SEED --rep-penalty 0.85 --class-mad 3
  run pen0.5_mad4   $SEED --rep-penalty 0.5  --class-mad 4
done
python3 compare_c2.py $OUTDIR
