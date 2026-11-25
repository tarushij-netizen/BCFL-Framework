"""Choose the collusion-check threshold from data.

Replays the collusion rule offline on sketches saved with --dump-screening (runs made with
--collusion-check --collusion-threshold 1.01, so the check never changed training), and reports,
for each threshold, how often honest clients and attackers would be flagged.

    python tune_collusion.py DUMP_DIR [DUMP_DIR ...] --attackers 0 1 2
A dump dir whose name contains "none" is treated as a run without attackers.
"""
import argparse
import glob
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("dirs", nargs="+")
p.add_argument("--attackers", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--window", type=int, default=3)
p.add_argument("--group", type=int, default=3)
p.add_argument("--thresholds", type=float, nargs="*",
               default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
a = p.parse_args()


def window_averages(d):
    hist, out = [], []
    for f in sorted(glob.glob(os.path.join(d, "round_*.npz"))):
        V = np.load(f)["sketches"]
        C = V - np.median(V, axis=0)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        hist = (hist + [C @ C.T])[-a.window:]
        if len(hist) == a.window:
            out.append(np.mean(hist, axis=0))
    return out


honest_pair, coll_pair, honest_max = [], [], []
counts = {t: dict(h_flag=0, h_tot=0, c_flag=0, c_tot=0, rounds_all=0, rounds_tot=0) for t in a.thresholds}
for d in a.dirs:
    atk = [] if "none" in os.path.basename(d.rstrip("/")) else a.attackers
    for A in window_averages(d):
        n = len(A)
        off = ~np.eye(n, dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                (coll_pair if (i in atk and j in atk) else honest_pair).append(A[i, j])
            if i not in atk:
                honest_max.append(A[i][off[i]].max())
        for t in a.thresholds:
            partners = ((A > t) & off).sum(axis=1)
            flagged = partners >= a.group - 1
            c = counts[t]
            for i in range(n):
                if i in atk:
                    c["c_tot"] += 1
                    c["c_flag"] += flagged[i]
                else:
                    c["h_tot"] += 1
                    c["h_flag"] += flagged[i]
            if atk:
                c["rounds_tot"] += 1
                c["rounds_all"] += all(flagged[i] for i in atk)

q = lambda x: np.percentile(x, [50, 90, 99, 100]).round(3).tolist() if len(x) else []
print(f"window-average pairwise cosine, percentiles 50/90/99/max:")
print(f"  honest-honest and honest-attacker pairs: {q(honest_pair)}")
print(f"  honest client's highest partner:         {q(honest_max)}")
print(f"  attacker-attacker pairs:                  {q(coll_pair)}  (min {min(coll_pair):.3f})" if coll_pair else "")
print("\nthreshold | honest flagged | attacker flagged | rounds with all attackers flagged")
for t in a.thresholds:
    c = counts[t]
    h = c["h_flag"] / max(c["h_tot"], 1)
    ca = c["c_flag"] / max(c["c_tot"], 1)
    r = c["rounds_all"] / max(c["rounds_tot"], 1)
    print(f"   {t:4.2f}   |   {h:6.1%} ({c['h_flag']}/{c['h_tot']}) |   {ca:6.1%} ({c['c_flag']}/{c['c_tot']})  |  {r:6.1%}")
