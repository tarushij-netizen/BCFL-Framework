"""Table for the C2/C3 attack experiments (run_c2_attacks.sh).
Usage: python compare_c2.py [results/c2]"""
import csv
import glob
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results/c2"
rows = []
for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
    if path.endswith(("_gas.csv", "_rewards.csv", "_summary.csv")):
        continue
    with open(path) as f:
        data = list(csv.DictReader(f))
    if not data or "attack_on" not in data[0]:
        continue
    name = os.path.basename(path)[:-4]
    asr = [float(r["backdoor_asr"]) for r in data if r["backdoor_asr"]]
    asr_late = [float(r["backdoor_asr"]) for r in data[3:] if r["backdoor_asr"]]  # rounds 0-2: untrained model
    on = [r for r in data if r["attack_on"] == "1"]
    inc = sum(int(r["malicious_included"]) for r in on)
    row = {
        "run": name,
        "final_acc": f"{float(data[-1]['global_acc']):.2f}",
        "acc_last5": f"{sum(float(r['global_acc']) for r in data[-5:]) / len(data[-5:]):.2f}",
        "final_asr": f"{asr[-1]:.2f}" if asr else "-",
        "max_asr_r3+": f"{max(asr_late):.2f}" if asr_late else "-",
        "attack_rounds": len(on),
        "attacker_in_while_attacking": f"{inc}/{3 * len(on)}",
        "honest_rejected": sum(int(r["honest_rejected"]) for r in data),
    }
    s = path[:-4] + "_onchain_summary.csv"
    if os.path.exists(s):
        with open(s) as f:
            x = list(csv.DictReader(f))[0]
        row.update({"attacker_reward_%": f"{100 * float(x['attacker_reward_share']):.1f}",
                    "sybil_reward_%": f"{100 * float(x['sybil_reward_share']):.1f}",
                    "new_identities": x["attacker_new_identities"],
                    "attacker_stake_eth": x["attacker_stake_eth"]})
    rows.append(row)
if not rows:
    sys.exit(f"no C2 result files in {d}")
cols = list(dict.fromkeys(k for r in rows for k in r))
w = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in rows:
    print("  ".join(str(r.get(c, "-")).ljust(w[c]) for c in cols))
