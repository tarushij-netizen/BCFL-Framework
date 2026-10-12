#!/usr/bin/env bash
# Backdoor fix: class-profile check on top of C1 (sketch) and C1+C3 (repsketch).
# 3/10 attackers, 30 rounds. Usage: bash run_backdoor_fix.sh [rounds] [seed]
set -e
ROUNDS=${1:-30}
SEED=${2:-1}
COMMON="--benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test --sketch-dim 64 --class-check"
run () {  # $1 = tag, $2 = aggregator, $3 = attack, $4 = extra flags
  NM=3; [ "$3" = "none" ] && NM=0
  python test.py $COMMON $4 --aggregator $2 --attack $3 --num-malicious $NM \
                 --results results/FashionMNIST_${2}_${3}${NM}_k64_cls3_${1}_seed${SEED}.csv
}
NONIID="--non-iid --alpha 0.5"
run iid sketch backdoor ""
run iid repsketch backdoor ""
run iid repsketch none ""
run iid repsketch labelflip ""
run noniid05 repsketch backdoor "$NONIID"
run noniid05 repsketch none "$NONIID"
run noniid05 repsketch labelflip "$NONIID"
python3 compare_results.py
