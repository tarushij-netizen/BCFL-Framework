"""Table for run_phase5.sh: one row per (C1 attack, proofs on/off, screening)."""
import csv
import glob
import os
import sys

print(f"{'attack':10s} {'zk':3s} {'screening':9s} rounds final_acc lost_rounds named_before_screening "
      f"removed_after_acceptance attacker_uploads_in")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_o*_*.csv"))):
    atk, zk_, scr = os.path.basename(f)[:-4].split("_")
    r = list(csv.DictReader(open(f)))
    if not r:
        continue
    lost = sum(int(x.get("round_discarded") or 0) for x in r)
    named = sum(len(x.get("rej_zk", "").split()) for x in r)
    late = sum(len(x.get("zk_late_removed", "").split()) for x in r)
    mal_in = sum(int(x.get("malicious_included") or 0) for x in r)
    print(f"{atk:10s} {zk_:3s} {scr:9s} {len(r):6d} {float(r[-1]['global_acc']):9.2f} {lost:11d} "
          f"{named:22d} {late:24d} {mal_in:19d}")
