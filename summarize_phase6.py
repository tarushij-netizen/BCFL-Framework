"""Tables (and accuracy curves) for phase6.py.

    python summarize_phase6.py results/phase6        -> prints the tables and writes
                                                        results/phase6/summary.md and figures/*.png
Every number is mean +/- standard deviation over the seeds that finished (n = number of seeds).
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "results/phase6"
ATK = {0, 1, 2}
lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def ms(vals, fmt="{:.1f}"):
    v = np.array([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
    if v.size == 0:
        return "–"
    return (fmt + " ± " + fmt).format(v.mean(), v.std()) if v.size > 1 else fmt.format(v[0])


def num(r, c, default=0.0):
    try:
        return float(r.get(c) or default)
    except ValueError:
        return default


def ids(r, c):
    return [int(x) for x in (r.get(c) or "").split()]


def metrics(rows, attacked):
    asr = [num(r, "backdoor_asr", np.nan) for r in rows if r.get("backdoor_asr")]
    return dict(
        acc=num(rows[-1], "global_acc"),
        asr=asr[-1] if asr else None,
        maxasr=max(asr[3:]) if len(asr) > 3 else None,
        mal_in=sum(num(r, "malicious_included") for r in rows) if attacked else None,
        hrej=sum(num(r, "honest_rejected") for r in rows),
        lost=sum(num(r, "round_discarded") for r in rows),
        round_s=np.mean([num(r, "train_s") + num(r, "aggregate_s") for r in rows]),
        agg_s=np.mean([num(r, "aggregate_s") for r in rows]),
        zk_s=np.mean([num(r, "zk_s") for r in rows]),
        mpc_s=np.mean([num(r, "mpc_s") for r in rows]),
        named=sum(len(ids(r, "rej_zk")) + len(ids(r, "zk_late_removed")) + len(ids(r, "rej_vss")) for r in rows),
        bad_rev=sum(1 for r in rows if r.get("bad_reveal_holders")),
        alarms=sum(1 for r in rows if num(r, "server_alarms") > 0),
        breach=sum(num(r, "privacy_breach") for r in rows),
        model_alarms=sum(len(ids(r, "model_alarms")) for r in rows),
        curve=[num(r, "global_acc") for r in rows])


groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(root, "*", "*_s[0-9]*.csv"))):
    if f.endswith("_gas.csv"):
        continue
    block = os.path.basename(os.path.dirname(f))
    name = os.path.basename(f)[:-4]
    m = re.match(r"(.+)_s(\d+)$", name)
    rows = load(f)
    if not rows or not m:
        continue
    key = m.group(1)
    groups[(block, key)].append(metrics(rows, "_none_" not in "_" + key + "_"))


def col(block, key, k, fmt="{:.1f}"):
    return ms([g[k] for g in groups.get((block, key), [])], fmt)


def nseeds(block, key):
    return len(groups.get((block, key), []))


emit("# Phase 6 results\n")
emit("## Main: accuracy, backdoor success, attacker uploads that reached the model\n")
for split in ("iid", "noniid"):
    emit(f"### {'IID' if split == 'iid' else 'non-IID (Dirichlet 0.5)'}\n")
    emit("| attack | defence | n | final acc (%) | final ASR (%) | max ASR r≥3 (%) | attacker uploads in | honest rejections |")
    emit("|---|---|---|---|---|---|---|---|")
    for atk in ("none", "signflip", "labelflip", "backdoor"):
        for dfn in ("fedavg", "median", "krum", "full"):
            key = f"{split}_{atk}_{dfn}"
            if not nseeds("main", key):
                continue
            emit(f"| {atk} | {dfn} | {nseeds('main', key)} | {col('main', key, 'acc')} | "
                 f"{col('main', key, 'asr') if atk == 'backdoor' else '–'} | "
                 f"{col('main', key, 'maxasr') if atk == 'backdoor' else '–'} | "
                 f"{col('main', key, 'mal_in', '{:.0f}') if atk != 'none' else '–'} | {col('main', key, 'hrej', '{:.0f}')} |")
    emit()

emit("## Screening inside the committee MPC (non-IID)\n")
emit("| attack | system | n | final acc (%) | final ASR (%) | attacker uploads in | honest rejections | MPC s/round |")
emit("|---|---|---|---|---|---|---|---|")
for atk in ("none", "backdoor"):
    for blk, key, lab in (("main", f"noniid_{atk}_full", "full, plaintext screening"),
                          ("mpc", f"noniid_{atk}_fullmpc", "full, MPC screening")):
        if nseeds(blk, key):
            emit(f"| {atk} | {lab} | {nseeds(blk, key)} | {col(blk, key, 'acc')} | "
                 f"{col(blk, key, 'asr') if atk == 'backdoor' else '–'} | "
                 f"{col(blk, key, 'mal_in', '{:.0f}') if atk != 'none' else '–'} | {col(blk, key, 'hrej', '{:.0f}')} | "
                 f"{col(blk, key, 'mpc_s', '{:.2f}') if blk == 'mpc' else '–'} |")
emit()

emit("## Attacks on the protocol (full system, IID)\n")
emit("| attack | n | final acc (%) | rounds lost | clients named / left out | rounds with bad shares named | "
     "server alarms (rounds) | privacy breaches | wrong models caught |")
emit("|---|---|---|---|---|---|---|---|---|")
for pa in ("commitlie", "badshare", "badreveal", "split_accepted", "wrong_model"):
    key = f"iid_{pa}_full"
    if nseeds("protocol", key):
        emit(f"| {pa} | {nseeds('protocol', key)} | {col('protocol', key, 'acc')} | {col('protocol', key, 'lost', '{:.0f}')} | "
             f"{col('protocol', key, 'named', '{:.0f}')} | {col('protocol', key, 'bad_rev', '{:.0f}')} | "
             f"{col('protocol', key, 'alarms', '{:.0f}')} | {col('protocol', key, 'breach', '{:.0f}')} | "
             f"{col('protocol', key, 'model_alarms', '{:.0f}')} |")
emit()

emit("## Cost per round (seconds, this machine, all clients simulated in one process)\n")
emit("| configuration | n | round total | of which aggregation | of which proofs (ZK) | MPC |")
emit("|---|---|---|---|---|---|")
for blk, key in (("main", "iid_none_fedavg"), ("main", "iid_none_median"), ("main", "iid_none_krum"),
                 ("main", "iid_none_full"), ("mpc", "noniid_none_fullmpc")):
    if nseeds(blk, key):
        emit(f"| {key} | {nseeds(blk, key)} | {col(blk, key, 'round_s', '{:.2f}')} | {col(blk, key, 'agg_s', '{:.2f}')} | "
             f"{col(blk, key, 'zk_s', '{:.2f}')} | {col(blk, key, 'mpc_s', '{:.2f}')} |")
emit()

gas = sorted(glob.glob(os.path.join(root, "chain", "*_gas.csv")))
if gas:
    emit("## Gas on the local chain (full system, --onchain)\n")
    emit("| operation | calls | mean gas |")
    emit("|---|---|---|")
    for r in load(gas[0]):
        cells = list(r.values())
        emit("| " + " | ".join(str(c) for c in cells) + " |")
    emit()

with open(os.path.join(root, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(root, "figures"), exist_ok=True)
    for split in ("iid", "noniid"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
        for ax, atk in zip(axes, ("none", "signflip", "labelflip", "backdoor")):
            for dfn in ("fedavg", "median", "krum", "full"):
                runs = groups.get(("main", f"{split}_{atk}_{dfn}"), [])
                if not runs:
                    continue
                c = np.array([r["curve"] for r in runs if len(r["curve"]) == len(runs[0]["curve"])])
                x = np.arange(c.shape[1])
                ax.plot(x, c.mean(0), label=dfn)
                ax.fill_between(x, c.mean(0) - c.std(0), c.mean(0) + c.std(0), alpha=0.2)
            ax.set_title(atk)
            ax.set_xlabel("round")
        axes[0].set_ylabel("test accuracy (%)")
        axes[-1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(root, "figures", f"accuracy_{split}.png"), dpi=150)
        plt.close(fig)
    print(f"\nwrote {os.path.join(root, 'summary.md')} and figures/accuracy_*.png")
except ImportError:
    print(f"\nwrote {os.path.join(root, 'summary.md')} (matplotlib not installed: no figures)")
