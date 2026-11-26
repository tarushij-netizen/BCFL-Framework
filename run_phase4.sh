#!/bin/bash
# Phase 4 (malicious server): each server attack with the checks on and off (--no-server-checks).
# The 3 malicious clients collude with the server; fake_accept needs MPC screening (BCFL_MPSPDZ).
mkdir -p results/phase4
for atk in none split_accepted drop_deal forge_complaint wrong_model fake_accept; do
  for chk in on off; do
    extra=""; [ $chk = off ] && extra="--no-server-checks"
    data="--attack none"; scr=""
    [ $atk = fake_accept ] && data="--attack signflip" && scr="--screening mpc"
    nm=3; [ $atk = none ] && nm=0
    python test.py --benchmark FashionMNIST --rounds 20 --seed 1 --no-client-test $data --num-malicious $nm \
      --aggregator repsketch --sketch-dim 64 --secagg hardened --server-attack $atk $extra $scr \
      --results results/phase4/${atk}_${chk}.csv
  done
done
python3 summarize_phase4.py results/phase4
