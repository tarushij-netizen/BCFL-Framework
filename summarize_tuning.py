"""Table for the collusion-threshold runs: mean ± sd over seeds per (setting, attack).

    python summarize_tuning.py DIR      (files named <setting>_<attack>_s<seed>.csv)
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

ATTACKERS = {0, 1, 2}
rows = defaultdict(list)
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_s*.csv"))):
    m = re.match(r"(.+)_(backdoor|none)_s(\d+)\.csv", os.path.basename(f))
    if not m:
        continue
    data = list(csv.DictReader(open(f)))
    if len(data) < 30:
        continue
    setting, attack = m.group(1), m.group(2)
    atk = ATTACKERS if attack == "backdoor" else set()
    ids = lambda r, c: [int(x) for x in r.get(c, "").split()]
    honest_coll = sum(len([i for i in ids(r, "rej_collusion") if i not in atk]) for r in data)
    honest_rej = sum(int(r["honest_rejected"]) for r in data)
    asr = [float(r["backdoor_asr"]) for r in data if r["backdoor_asr"]]
    rows[(setting, attack)].append(dict(
        acc=float(data[-1]["global_acc"]),
        asr=asr[-1] if asr else np.nan,
        maxasr=max(asr[3:]) if asr else np.nan,
        hrej=honest_rej, hcoll=honest_coll,
        mal_in=sum(int(r["malicious_included"]) for r in data) if atk else np.nan))

fmt = lambda v: "–" if np.all(np.isnan(v)) else f"{np.nanmean(v):.1f} ± {np.nanstd(v):.1f}"
print("setting      attack    seeds  final acc     final ASR     max ASR (r>=3)  honest rej   of which collusion  attacker uploads in")
for (setting, attack), rs in sorted(rows.items()):
    col = lambda k: np.array([r[k] for r in rs], dtype=float)
    print(f"{setting:12s} {attack:9s} {len(rs):5d}  {fmt(col('acc')):13s} {fmt(col('asr')):13s} {fmt(col('maxasr')):15s} "
          f"{fmt(col('hrej')):12s} {fmt(col('hcoll')):19s} {fmt(col('mal_in'))}")
