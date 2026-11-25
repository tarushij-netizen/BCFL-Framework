#!/bin/bash
# Phase 3 (Feldman VSS on the C1 mask shares): old (--no-vss) vs new, under the share attacks.
# Attackers 0-2 train honestly; they attack the secure aggregation itself.
mkdir -p results/phase3
for atk in none badshare badreveal; do
  nm=3; [ $atk = none ] && nm=0
  for v in vss novss; do
    extra=""; [ $v = novss ] && extra="--no-vss"
    python test.py --benchmark FashionMNIST --rounds 30 --seed 1 --no-client-test --attack none \
      --num-malicious $nm --aggregator repsketch --sketch-dim 64 --secagg hardened --c1-attack $atk $extra \
      --results results/phase3/${atk}_${v}.csv
  done
done
