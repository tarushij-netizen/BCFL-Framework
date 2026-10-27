#!/usr/bin/env bash
# Confirm the tuned reputation penalty (0.7) does not let switching attackers back in.
# Non-IID (alpha 0.5) label-flip, 3/10 attackers, sketch k64 + class check, 30 rounds.
#   onoff5 : 5 honest rounds, 5 attack rounds, ...
#   onoff1 : attack every other round (the hardest case for a soft penalty)
# Plus an IID sanity check of penalty 0.7 on sign-flip and backdoor (seed 1).
# Usage: bash run_c3_confirm.sh "1 2 3" [rounds]      (~5 min per run, 17 runs)
set -e
SEEDS=${1:-"1 2 3"}
ROUNDS=${2:-30}
OUTDIR=results/c3confirm
mkdir -p $OUTDIR
run () {  # $1 label, $2 seed, rest = flags
  local LABEL=$1 SEED=$2; shift 2
  local OUT=$OUTDIR/${LABEL}_seed${SEED}.csv
  if [ -f "$OUT" ]; then echo "skip $OUT"; return; fi
  echo "=== $LABEL seed $SEED ==="
  python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test \
                 --num-malicious 3 --sketch-dim 64 --class-check --results $OUT "$@"
}
NL="--non-iid --alpha 0.5 --attack labelflip"
for SEED in $SEEDS; do
  run onoff5_pen0.7    $SEED $NL --attack-mode onoff --attack-period 5 --aggregator repsketch --rep-penalty 0.7
  run onoff1_pen0.7    $SEED $NL --attack-mode onoff --attack-period 1 --aggregator repsketch --rep-penalty 0.7
  run onoff1_pen0.5    $SEED $NL --attack-mode onoff --attack-period 1 --aggregator repsketch --rep-penalty 0.5
  run onoff1_norep     $SEED $NL --attack-mode onoff --attack-period 1 --aggregator sketch
  run onoff1_pen0.85   $SEED $NL --attack-mode onoff --attack-period 1 --aggregator repsketch --rep-penalty 0.85
done
run iid_signflip_pen0.7 1 --attack signflip --aggregator repsketch --rep-penalty 0.7
run iid_backdoor_pen0.7 1 --attack backdoor --aggregator repsketch --rep-penalty 0.7
python3 compare_c2.py $OUTDIR
