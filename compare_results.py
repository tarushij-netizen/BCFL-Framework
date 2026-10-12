"""Summarise results/*.csv written by test.py --aggregator ... (no extra packages needed)."""
import csv
import glob
import os
import re
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
    with open(path) as f:
        data = list(csv.DictReader(f))
    if not data or "global_acc" not in data[0]:
        continue  # not a training run (e.g. inference_attacks.csv, sketch_leakage.csv)
    accs = [float(r["global_acc"]) for r in data]
    last5 = accs[-5:]
    caught = sum(int(r["malicious_caught"]) for r in data)
    honest_rej = sum(int(r["honest_rejected"]) for r in data)
    asr = data[-1].get("backdoor_asr") or ""
    rows.append((os.path.basename(path)[:-4], len(data), accs[-1], sum(last5) / len(last5),
                 max(accs), caught, honest_rej, f"{float(asr):.1f}" if asr else "-"))

if not rows:
    print(f"No CSV files in {folder}/")
    sys.exit(0)
print(f"{'run':55s} {'rounds':>6s} {'final':>7s} {'last5':>7s} {'best':>7s} {'caught':>7s} {'honestRej':>9s} {'ASR':>6s}")
for r in rows:
    print(f"{r[0]:55s} {r[1]:6d} {r[2]:7.2f} {r[3]:7.2f} {r[4]:7.2f} {r[5]:7d} {r[6]:9d} {r[7]:>6s}")
print("\nfinal/last5/best = global model test accuracy (%). caught = attacker uploads rejected "
      "(summed over rounds), honestRej = honest uploads rejected, ASR = final backdoor attack success rate (%). Krum 'rejects' every client "
      "except the one it picks, so its honestRej is large by design.")


# ---- mean +- std over seeds (runs whose names differ only in _seedN)
groups = {}
for r in rows:
    m = re.match(r"(.*)_seed(\d+)$", r[0])
    if m:
        groups.setdefault(m.group(1), []).append(r)
multi = {k: v for k, v in groups.items() if len(v) > 1}
if multi:
    def ms(xs):
        mu = sum(xs) / len(xs)
        sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        return f"{mu:6.2f} +- {sd:4.2f}"
    print(f"\n{'run (over seeds)':55s} {'n':>2s} {'final acc':>15s} {'ASR':>15s}")
    for k, v in sorted(multi.items()):
        asrs = [float(x[7]) for x in v if x[7] != "-"]
        print(f"{k:55s} {len(v):2d} {ms([x[2] for x in v]):>15s} {ms(asrs) if len(asrs) > 1 else '-':>15s}")
