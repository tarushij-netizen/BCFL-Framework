#!/usr/bin/env bash
# Privacy (local DP) vs poisoning defense trade-off on FashionMNIST, 2/10 signflip attackers.
# Step 1 (calibrate): bash run_dp_experiments.sh calibrate
#   -> read the "[DP] ... honest update norm median X" lines; pick CLIP about equal to that median.
# Step 2 (sweep):     bash run_dp_experiments.sh <CLIP> [rounds] [seed]
set -e
if [ "$1" = "calibrate" ]; then
  python test.py --benchmark FashionMNIST --aggregator fedavg --rounds 3 --seed 1 --no-client-test \
                 --dp-clip 1000 --dp-sigma 0 --results results_calibrate.csv
  exit 0
fi
CLIP=${1:?give the clip value from the calibrate step}
ROUNDS=${2:-30}
SEED=${3:-1}
COMMON="--benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test"
for SIGMA in 0 0.05 0.1 0.5 1; do
  python test.py $COMMON --aggregator fedavg     --attack none --dp-clip $CLIP --dp-sigma $SIGMA
  for AGG in fedavg similarity median; do
    python test.py $COMMON --aggregator $AGG --attack signflip --num-malicious 2 --dp-clip $CLIP --dp-sigma $SIGMA
  done
done
python3 compare_results.py
