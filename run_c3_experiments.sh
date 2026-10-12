#!/usr/bin/env bash
# C3: does reputation (memory across rounds) stop what one-round screening misses?
# 3/10 attackers, sketch k=64. Usage: bash run_c3_experiments.sh [rounds] [seed]
set -e
ROUNDS=${1:-30}
SEED=${2:-1}
COMMON="--benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test --sketch-dim 64"
run () {  # $1 = tag, $2 = aggregator, $3 = attack, $4 = extra flags
  NM=3; [ "$3" = "none" ] && NM=0
  python test.py $COMMON $4 --aggregator $2 --attack $3 --num-malicious $NM \
                 --results results/FashionMNIST_${2}_${3}${NM}_k64_${1}_seed${SEED}.csv
}
run iid repsketch none ""
run iid repsketch backdoor ""
run iid repsketch signflip ""
run iid repsketch labelflip ""
NONIID="--non-iid --alpha 0.5"
python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test $NONIID \
               --aggregator fedavg --attack none --results results/FashionMNIST_fedavg_none0_noniid05_seed${SEED}.csv
run noniid05 sketch none "$NONIID"
run noniid05 repsketch none "$NONIID"
run noniid05 sketch backdoor "$NONIID"
run noniid05 repsketch backdoor "$NONIID"
run noniid05 sketch labelflip "$NONIID"
run noniid05 repsketch labelflip "$NONIID"
python3 compare_results.py
