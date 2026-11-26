#!/bin/bash
# Phase 5 (commitments + zero-knowledge proofs): the C1 lies with the proofs on and off (--no-zk).
# The 3 malicious clients upload a sign-flipped update and lie about it to the screening.
mkdir -p results/phase5
for cfg in "none on" "none off" "sketchlie on" "sketchlie off" "commitlie on" "commitlie off" "pairlie on"; do
  set -- $cfg; atk=$1; zk=$2
  extra=""; [ $zk = off ] && extra="--no-zk"
  data="--attack signflip --num-malicious 3"; [ $atk = none ] && data="--attack none --num-malicious 0"
  python test.py --benchmark FashionMNIST --rounds 20 --seed 1 --no-client-test $data --aggregator repsketch \
    --sketch-dim 64 --class-check --secagg hardened --c1-attack $atk $extra --results results/phase5/${atk}_${zk}_plain.csv
done
if [ -n "$BCFL_MPSPDZ" ]; then
  python test.py --benchmark FashionMNIST --rounds 20 --seed 1 --no-client-test --attack signflip --num-malicious 3 \
    --aggregator repsketch --sketch-dim 64 --class-check --secagg hardened --c1-attack commitlie --screening mpc \
    --results results/phase5/commitlie_on_mpc.csv
fi
python3 summarize_phase5.py results/phase5
