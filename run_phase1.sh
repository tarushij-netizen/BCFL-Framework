#!/usr/bin/env bash
# Phase 1 (hardened C1) adversarial runs: each C1 attack against the OLD protocol (--secagg dh)
# and the NEW one (--secagg hardened), plus clean and plain sign-flip runs of the new protocol.
# Attackers: 3 of 10 sign-flip clients that also cheat on C1 itself (--c1-attack).
#   sketchlie : benign sketch, poisoned masked upload            (gap 3)
#   nullspace : poison hidden where the expected sketch is blind (gap 2)
# Skips any run whose result file already exists.
# Usage: bash run_phase1.sh [seed] [rounds]      (8 runs, about 5 minutes each)
set -e
SEED=${1:-1}
ROUNDS=${2:-30}
mkdir -p results/phase1
run () {  # tag secagg attack nm c1
  OUT=results/phase1/FashionMNIST_$1_seed${SEED}.csv
  if [ -f "$OUT" ]; then echo "skip $OUT"; return; fi
  python test.py --benchmark FashionMNIST --rounds $ROUNDS --seed $SEED --no-client-test \
                 --attack $3 --num-malicious $4 --results $OUT \
                 --aggregator repsketch --sketch-dim 64 --class-check --secagg $2 --c1-attack $5
}
run old_sketchlie   dh       signflip 3 sketchlie
run new_sketchlie   hardened signflip 3 sketchlie
run old_nullspace   dh       signflip 3 nullspace
run new_nullspace   hardened signflip 3 nullspace
run old_signflip    dh       signflip 3 none
run new_signflip    hardened signflip 3 none
run old_none        dh       none     0 none
run new_none        hardened none     0 none
python3 compare_results.py results/phase1
