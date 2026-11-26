"""Table for run_phase4.sh: one row per (server attack, checks on/off)."""
import csv
import glob
import os
import sys

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_o*.csv"))):
    name = os.path.basename(f)[:-4]
    atk, chk = name.rsplit("_", 1)
    r = list(csv.DictReader(open(f)))
    if not r:
        continue
    num = lambda c: sum(int(x[c]) for x in r if x.get(c, "") not in ("", None))
    cnt = lambda c: sum(len(x.get(c, "").split()) for x in r)
    rows.append((atk, chk, len(r), float(r[-1]["global_acc"]), num("round_discarded"),
                 sum(1 for x in r if int(x.get("server_alarms") or 0) > 0), num("privacy_breach"),
                 cnt("rej_vss"), cnt("model_alarms"), num("malicious_included") if atk == "fake_accept" else ""))
print(f"{'server attack':16s} checks rounds final_acc lost_rounds rounds_with_alarm privacy_breaches "
      f"honest_left_out model_alarms attacker_uploads_in")
for r in rows:
    print(f"{r[0]:16s} {r[1]:6s} {r[2]:6d} {r[3]:9.2f} {r[4]:11d} {r[5]:17d} {r[6]:16d} {r[7]:15d} {r[8]:12d} {r[9]}")
