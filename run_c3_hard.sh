#!/usr/bin/env bash
# Harder settings where screening alone may miss attackers, to test what reputation (C3) adds.
# Each case is run WITH reputation (repsketch) and WITHOUT (sketch). No chain needed (faster).
# ~5 min per run, 10 runs. Skips runs whose result file exists.
# Usage: bash run_c3_hard.sh [seed] [rounds] [only]
#   only = optional text; runs whose label does not contain it are skipped
#   e.g.  bash run_c3_hard.sh 2 30 noniid_labelflip
set -e
SEED=${1:-1}
ROUNDS=${2:-30}
ONLY=${3:-}
OUTDIR=results/c3hard
mkdir -p $OUTDIR
run () {  # $1 label, rest = flags
  local LABEL=$1; shift
  if [ -n "$ONLY" ] && [[ "$LABEL" != *"$ONLY"* ]]; then return; fi
  for AGG in repsketch sketch; do
    local OUT=$OUTDIR/${LABEL}_${AGG}_seed${SEED}.csv
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    echo "=== $LABEL ($AGG) ==="
    python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test \
                   --num-malicious 3 --sketch-dim 64 --class-check --aggregator $AGG --results $OUT "$@"
  done
}
NONIID="--non-iid --alpha 0.5"
# non-IID: screening is noisier here (honest clients differ more), so attackers may slip through
run noniid_labelflip_always   $NONIID --attack labelflip
run noniid_labelflip_onoff5   $NONIID --attack labelflip --attack-mode onoff --attack-period 5
run noniid_signflip_sleeper10 $NONIID --attack signflip  --attack-mode sleeper --attack-period 10
# stealthy backdoor: only 20% of each batch poisoned, so the class profile is less lopsided
run iid_backdoor_stealth20    --attack backdoor --poison-frac 0.2
run iid_backdoor_stealth20_onoff5 --attack backdoor --poison-frac 0.2 --attack-mode onoff --attack-period 5
python3 compare_c2.py $OUTDIR
