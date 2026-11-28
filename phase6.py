"""
Phase 6: the full experiment matrix for the thesis, with every C1 phase switched on.

    python phase6.py --list                 # what would run, and how many runs
    python phase6.py -j 3                   # run everything (3 runs at a time), resumable
    python phase6.py -j 3 --only main       # one block: main | mpc | protocol | chain
    python phase6.py --smoke                # every distinct configuration for 2 rounds (seed 1)
    python summarize_phase6.py results/phase6

Blocks
  main      FashionMNIST, 10 clients (3 malicious when attacked), 30 rounds, seeds 1-3,
            IID and non-IID (Dirichlet 0.5), attacks none / signflip / labelflip / backdoor,
            defences FedAvg, Median, Krum and the full system ("full": RepSketch with the class and
            collusion checks over hardened secure aggregation = Phases 1-5, plaintext screening).
  mpc       the full system with screening in the committee MPC (Phase 2), non-IID, none and
            backdoor, seeds 1-3. Needs BCFL_MPSPDZ. Runs one at a time (fixed MPC client port).
  protocol  attacks on the protocol itself against the full system, IID, seeds 1-3, 20 rounds:
            commitlie (C1 lie, Phase 5), badshare and badreveal (Phase 3), split_accepted and
            wrong_model (malicious server, Phase 4).
  chain     the full system on the local chain (--onchain), 5 rounds, seed 1: gas per round.

Every finished run leaves results/phase6/<block>/<name>.csv (and .log); a run whose CSV already has
all its rounds is skipped, so the script can be stopped and started again.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

OUT = "results/phase6"
SEEDS = [1, 2, 3]
FULL = ["--aggregator", "repsketch", "--sketch-dim", "64", "--class-check", "--collusion-check",
        "--secagg", "hardened"]
DEF = {"fedavg": ["--aggregator", "fedavg"], "median": ["--aggregator", "median"],
       "krum": ["--aggregator", "krum"], "full": FULL}


def attack_args(atk):
    return ["--attack", atk, "--num-malicious", "0" if atk == "none" else "3"]


def split_args(split):
    return ["--non-iid", "--alpha", "0.5"] if split == "noniid" else []


def jobs(rounds_main=30, rounds_proto=20, seeds=SEEDS):
    out = []
    for split, atk, dfn, seed in itertools.product(["iid", "noniid"], ["none", "signflip", "labelflip", "backdoor"],
                                                   ["fedavg", "median", "krum", "full"], seeds):
        out.append(dict(block="main", name=f"{split}_{atk}_{dfn}_s{seed}", rounds=rounds_main, mpc=False,
                        args=DEF[dfn] + attack_args(atk) + split_args(split) + ["--seed", str(seed)]))
    for atk, seed in itertools.product(["none", "backdoor"], seeds):
        out.append(dict(block="mpc", name=f"noniid_{atk}_fullmpc_s{seed}", rounds=rounds_main, mpc=True,
                        args=FULL + ["--screening", "mpc"] + attack_args(atk) + split_args("noniid")
                        + ["--seed", str(seed)]))
    proto = {"commitlie": ["--c1-attack", "commitlie", "--attack", "signflip", "--num-malicious", "3"],
             "badshare": ["--c1-attack", "badshare", "--attack", "none", "--num-malicious", "3"],
             "badreveal": ["--c1-attack", "badreveal", "--attack", "none", "--num-malicious", "3"],
             "split_accepted": ["--server-attack", "split_accepted", "--attack", "none", "--num-malicious", "3"],
             "wrong_model": ["--server-attack", "wrong_model", "--attack", "none", "--num-malicious", "3"]}
    for (pa, extra), seed in itertools.product(proto.items(), seeds):
        out.append(dict(block="protocol", name=f"iid_{pa}_full_s{seed}", rounds=rounds_proto, mpc=False,
                        args=FULL + extra + ["--seed", str(seed)]))
    out.append(dict(block="chain", name="iid_none_full_onchain_s1", rounds=5, mpc=False,
                    args=FULL + attack_args("none") + ["--seed", "1", "--onchain"]))
    return out


def done(job):
    path = os.path.join(OUT, job["block"], job["name"] + ".csv")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        return sum(1 for _ in f) - 1 >= job["rounds"]


def command(job):
    csv = os.path.join(OUT, job["block"], job["name"] + ".csv")
    return [sys.executable, "test.py", "--benchmark", "FashionMNIST", "--rounds", str(job["rounds"]),
            "--no-client-test"] + job["args"] + ["--results", csv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", type=int, default=3, help="runs at a time (MPC runs always one at a time)")
    ap.add_argument("--only", choices=["main", "mpc", "protocol", "chain"], action="append")
    ap.add_argument("--match", action="append", help="only runs whose name contains this (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="each distinct configuration, 2 rounds, seed 1")
    a = ap.parse_args()
    global OUT
    todo = jobs()
    if a.smoke:
        OUT = "results/phase6_smoke"
        todo = [dict(j, rounds=2) for j in jobs(2, 2, [1])]
    if a.only:
        todo = [j for j in todo if j["block"] in a.only]
    if a.match:
        todo = [j for j in todo if any(m in j["name"] for m in a.match)]
    if a.list:
        for j in todo:
            print(("done  " if done(j) else "todo  ") + j["block"] + "/" + j["name"])
        print(f"{sum(not done(j) for j in todo)} of {len(todo)} runs still to do")
        return
    if any(j["mpc"] for j in todo) and not os.environ.get("BCFL_MPSPDZ"):
        print("BCFL_MPSPDZ is not set: the mpc block is skipped")
        todo = [j for j in todo if not j["mpc"]]
    todo = [j for j in todo if not done(j)]
    for b in {j["block"] for j in todo}:
        os.makedirs(os.path.join(OUT, b), exist_ok=True)
    parallel = [j for j in todo if not j["mpc"] and j["block"] != "chain"]
    serial = [j for j in todo if j["mpc"] or j["block"] == "chain"]
    print(f"{len(todo)} runs to do ({len(parallel)} in parallel, {len(serial)} one at a time)")
    t0, running, finished, failed = time.time(), {}, 0, []
    free_slots = list(range(a.j))

    def start(job, slot):
        env = dict(os.environ, BCFL_CHAIN_PORT=str(8545 + 10 * slot), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        log = open(os.path.join(OUT, job["block"], job["name"] + ".log"), "w")
        return subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT, env=env), log

    def reap(block):
        nonlocal finished
        while running:
            for slot, (job, proc, log) in list(running.items()):
                if proc.poll() is not None:
                    log.close()
                    finished += 1
                    ok = proc.returncode == 0 and done(job)
                    if not ok:
                        failed.append(job["name"])
                    el = (time.time() - t0) / 60
                    print(f"[{finished}/{len(todo)}] {'ok  ' if ok else 'FAIL'} {job['block']}/{job['name']} "
                          f"({el:.0f} min so far)", flush=True)
                    del running[slot]
                    free_slots.append(slot)
            if not block:
                return
            time.sleep(2)

    for job in parallel:
        while not free_slots:
            reap(False)
            time.sleep(2)
        slot = free_slots.pop(0)
        proc, log = start(job, slot)
        running[slot] = (job, proc, log)
    reap(True)
    for job in serial:
        slot = free_slots.pop(0)
        proc, log = start(job, slot)
        running[slot] = (job, proc, log)
        reap(True)
    print(f"finished in {(time.time() - t0) / 60:.0f} min; failed: {failed or 'none'}")
    if failed:
        print("see the .log next to each failed run's CSV")


if __name__ == "__main__":
    main()
