#!/usr/bin/env bash
# C2/C3 attack experiments: Sybil accounts, on-off and sleeper attackers, whitewashing.
# IID FashionMNIST, 3/10 attackers, ours = repsketch k64 + class check. ~6 min per run.
# Skips runs whose result file exists, so it can be stopped and restarted.
# Usage: bash run_c2_attacks.sh [seed] [rounds]
set -e
SEED=${1:-1}
ROUNDS=${2:-30}
OUTDIR=results/c2
mkdir -p $OUTDIR
run () {  # $1 label, rest = flags
  local LABEL=$1; shift
  local OUT=$OUTDIR/${LABEL}_seed${SEED}.csv
  if [ -f "$OUT" ]; then echo "skip $OUT"; return; fi
  echo "=== $LABEL ==="
  python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test \
                 --num-malicious 3 --sketch-dim 64 --class-check --results $OUT "$@"
}
OURS="--aggregator repsketch"
# 1-2  Sybil hole and its fix: 5 idle accounts, reward threshold 0.6 vs 0.9
run sybil5_rt0.6        $OURS --attack backdoor --onchain --sybil 5 --reward-tau 0.6
run sybil5_rt0.9        $OURS --attack backdoor --onchain --sybil 5 --reward-tau 0.9
# 3-4  attackers who switch: on-off (5 honest, 5 attack, ...) and sleeper (10 honest, then attack)
run backdoor_onoff5     $OURS --attack backdoor --onchain --attack-mode onoff --attack-period 5
run backdoor_sleeper10  $OURS --attack backdoor --onchain --attack-mode sleeper --attack-period 10
run signflip_onoff5     $OURS --attack signflip --onchain --attack-mode onoff --attack-period 5
# 5-6  whitewashing: an excluded attacker comes back under a new identity
run backdoor_whitewash  $OURS --attack backdoor --onchain --whitewash
run signflip_whitewash  $OURS --attack signflip --onchain --whitewash
# 7-8  same switching attackers WITHOUT reputation (sketch only): what does C3 add?
run backdoor_onoff5_norep     --aggregator sketch --attack backdoor --attack-mode onoff --attack-period 5
run backdoor_sleeper10_norep  --aggregator sketch --attack backdoor --attack-mode sleeper --attack-period 10
python3 compare_c2.py $OUTDIR
