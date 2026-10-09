"""Summarise results/*.csv written by test.py --aggregator ... (no extra packages needed)."""
import csv
import glob
import os
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
    with open(path) as f:
        data = list(csv.DictReader(f))
    if not data:
        continue
    accs = [float(r["global_acc"]) for r in data]
    last5 = accs[-5:]
    caught = sum(int(r["malicious_caught"]) for r in data)
    honest_rej = sum(int(r["honest_rejected"]) for r in data)
    rows.append((os.path.basename(path)[:-4], len(data), accs[-1], sum(last5) / len(last5),
                 max(accs), caught, honest_rej))

if not rows:
    print(f"No CSV files in {folder}/")
    sys.exit(0)
print(f"{'run':55s} {'rounds':>6s} {'final':>7s} {'last5':>7s} {'best':>7s} {'caught':>7s} {'honestRej':>9s}")
for r in rows:
    print(f"{r[0]:55s} {r[1]:6d} {r[2]:7.2f} {r[3]:7.2f} {r[4]:7.2f} {r[5]:7d} {r[6]:9d}")
print("\nfinal/last5/best = global model test accuracy (%). caught = attacker uploads rejected "
      "(summed over rounds), honestRej = honest uploads rejected. Krum 'rejects' every client "
      "except the one it picks, so its honestRej is large by design.")
