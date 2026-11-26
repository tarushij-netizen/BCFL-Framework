"""C1 Phase 2: run the screening rule as an MPC among the committee (MP-SPDZ).

The committee members are the MPC parties (malicious-shamir-party.x: honest-majority Shamir,
malicious security, threshold t = (n-1)//2 corrupted). The FL clients and the aggregator server
are MP-SPDZ *external clients*:
  - each FL client secret-shares its integer sketch (K values) and class profile (C values),
    using the client protocol of MP-SPDZ (the parties hand it shares of random values with
    a consistency check; the client sends its input masked by them);
  - the server (coordinator) sends the public per-round data (reputation weights, which clients
    have been around long enough for the collusion check) and receives only the verdict bits,
    then, after the reputation rule, the sum of the accepted sketches for the Phase 1 sum check.

The MPC program is mpc/bcfl_screen.mpc. Locally all parties run on this machine (one process
each); in a deployment each committee member runs its own party on its own host.

Set BCFL_MPSPDZ to the MP-SPDZ folder (with malicious-shamir-party.x built or copied there).
"""
import atexit
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np

FRAC_BITS = 24
_HERE = os.path.dirname(os.path.abspath(__file__))


@contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_client_lib(root):
    """MP-SPDZ's ExternalIO/client.py, loaded under its own name (this repo has a `client` package)."""
    if "mpspdz_client" in sys.modules:
        return sys.modules["mpspdz_client"]
    ext = os.path.join(root, "ExternalIO")
    for name, fname in (("domains", "domains.py"), ("mpspdz_client", "client.py")):
        spec = importlib.util.spec_from_file_location(name, os.path.join(ext, fname))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mpspdz_client"]


class MPCScreening:
    def __init__(self, n, k, c, window, sim, mag, class_mad, coll_thr, group,
                 parties=5, root=None, port=None, party_port=None, binary="malicious-shamir-party.x",
                 log_dir=None, observers=0, zk=False):
        """observers (Phase 4): number of committee observer clients that receive the same
        authenticated outputs as the coordinator (normally one per committee member)."""
        self.n, self.k, self.c, self.window = n, k, c, window
        self.parties = parties
        self.root = os.path.abspath(root or os.environ.get("BCFL_MPSPDZ", ""))
        if not os.path.isfile(os.path.join(self.root, binary)):
            raise FileNotFoundError(f"{binary} not found in {self.root!r}: set BCFL_MPSPDZ to the MP-SPDZ folder")
        # party-to-party ports: not MP-SPDZ's default 5000, which macOS uses for AirPlay Receiver
        self.port = int(port or 14000)        # must match PORT in bcfl_screen.mpc
        self.party_port = int(party_port or os.environ.get("BCFL_MPC_PARTY_PORT", 16000))
        self.binary = binary
        self.args = [str(n), str(k), str(c), str(window), repr(float(sim)), repr(float(mag)),
                     repr(float(class_mad)), repr(float(coll_thr)), str(int(group)), str(int(observers)),
                     str(int(bool(zk)))]
        self.zk = bool(zk)
        self.stage = None           # None | "begun" (waiting for pub2) | "decided" (waiting for the accepted set)
        self.observers = int(observers)
        self.last_observed = []     # what each observer received in the last decide / sum_accepted
        self.prog = "-".join(["bcfl_screen"] + self.args)
        self.log_dir = log_dir or os.path.join(self.root, "logs-bcfl")
        self.procs, self.clients, self.coord, self.obs = [], [], None, []
        self.rounds = 0
        self.last_seconds = 0.0
        self.total_seconds = 0.0

    # ----- setup
    def _compile(self):
        src = os.path.join(_HERE, "bcfl_screen.mpc")
        dst = os.path.join(self.root, "Programs", "Source", "bcfl_screen.mpc")
        if not os.path.exists(dst) or open(src).read() != open(dst).read():
            shutil.copy(src, dst)
        sch = os.path.join(self.root, "Programs", "Schedules", self.prog + ".sch")
        if not os.path.exists(sch) or os.path.getmtime(sch) < os.path.getmtime(dst):
            subprocess.run([sys.executable, "compile.py", "-M", "-F", "128", "bcfl_screen"] + self.args,
                           cwd=self.root, check=True, stdout=subprocess.DEVNULL)

    def _certs(self):
        """TLS certificates for the parties (P*) and the external clients (C*).

        Made here rather than with MP-SPDZ's setup scripts: those put the name only in the
        subject CN, and newer OpenSSL/Boost builds (e.g. Homebrew OpenSSL 4 on macOS) check the
        peer name against subjectAltName only, so every handshake fails. Each certificate here
        carries its name as a DNS subjectAltName too. Regenerated once (marker file) and whenever
        one is missing."""
        pd = os.path.join(self.root, "Player-Data")
        os.makedirs(pd, exist_ok=True)
        names = [f"P{i}" for i in range(self.parties)] + [f"C{i}" for i in range(self.n + 1 + self.observers)]
        marker = os.path.join(pd, ".bcfl-certs-san")
        if os.path.exists(marker) and all(os.path.exists(os.path.join(pd, f"{x}.pem")) for x in names):
            return
        openssl = shutil.which("openssl")
        if openssl is None:
            raise FileNotFoundError("openssl command not found (needed to make the MPC certificates)")
        for x in names:
            subprocess.run([openssl, "req", "-newkey", "rsa:2048", "-nodes", "-x509", "-days", "365",
                            "-out", os.path.join(pd, f"{x}.pem"), "-keyout", os.path.join(pd, f"{x}.key"),
                            "-subj", f"/CN={x}", "-addext", f"subjectAltName=DNS:{x}"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in os.listdir(pd):                # old hash links of the replaced certificates
            if f.endswith(".0") and os.path.islink(os.path.join(pd, f)):
                os.remove(os.path.join(pd, f))
        subprocess.run([openssl, "rehash", pd], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        open(marker, "w").write("certificates with subjectAltName, made by mpc/screen_mpc.py\n")

    def start(self):
        self._compile()
        self._certs()
        lib = _load_client_lib(self.root)
        for p in range(self.parties):
            os.makedirs(self.log_dir, exist_ok=True)
            out = open(os.path.join(self.log_dir, f"party{p}.log"), "w")
            self.procs.append(subprocess.Popen(
                [f"./{self.binary}", "-N", str(self.parties), "-p", str(p), "-pn", str(self.party_port),
                 "-h", "localhost", self.prog], cwd=self.root, stdout=out, stderr=subprocess.STDOUT))
        self._wait_for_parties()
        hosts = ["localhost"] * self.parties
        with _cwd(self.root):
            self.clients = [lib.Client(hosts, self.port, i) for i in range(self.n)]
            self.coord = lib.Client(hosts, self.port, self.n)
            self.obs = [lib.Client(hosts, self.port, self.n + 1 + o) for o in range(self.observers)]
        self._lib = lib
        atexit.register(self.stop)
        return self

    def _ports_listening(self):
        """True when every party listens on its client port. Checked without connecting: a probe
        connection would be taken by the party as a client."""
        ports = [self.port + p for p in range(self.parties)]
        for cmd in (["lsof", "-nP", f"-iTCP:{ports[0]}-{ports[-1]}", "-sTCP:LISTEN"], ["ss", "-ltnH"]):
            if shutil.which(cmd[0]) is None:
                continue
            out = subprocess.run(cmd, capture_output=True, text=True).stdout
            return all(f":{p}" in out for p in ports)
        return None

    def _wait_for_parties(self, timeout=180.0):
        deadline = time.time() + timeout
        while True:
            dead = [p for p, proc in enumerate(self.procs) if proc.poll() is not None]
            if dead:
                code = self.procs[dead[0]].returncode
                log = os.path.join(self.log_dir, f"party{dead[0]}.log")
                self.stop()
                tail = open(log).read()[-2000:] if os.path.exists(log) else ""
                raise RuntimeError(f"MPC party {dead[0]} exited with code {code}; log {log}:\n{tail}")
            ready = self._ports_listening()
            if ready is None:                   # no lsof/ss: give the parties a fixed head start
                time.sleep(5.0)
                return
            if ready:
                return
            if time.time() > deadline:
                self.stop()
                raise RuntimeError(f"MPC parties did not open ports {self.port}..{self.port + self.parties - 1} "
                                   f"within {timeout:.0f}s; see the logs in {self.log_dir}")
            time.sleep(0.5)

    def _send_public(self, values):
        for sock in self.coord.sockets:
            os_ = self._lib.octetStream()
            for v in values:
                os_.store(int(v))
            os_.Send(sock)

    # ----- one round
    @property
    def prime(self):
        return int(self.coord.domain.modulus)

    def begin(self, sketches_int, q_int, weights, present=None, mus=None):
        """Inputs of one round. sketches_int: n x k signed ints (24 fractional bits); q_int: n x c
        squared class-row norms (48 fractional bits); weights: public reputation weights (or None);
        present: clients in this round's qualified set; mus: n x 2 masks (Phase 5, zk=True).
        With zk, returns what the MPC drew and computed for the binding: {"coef": K+C ints,
        "V1": [n], "V2": [n], "in_range": [n]} (V mod p), as the coordinator received it; each
        observer's copy is in self.last_observed_zk."""
        self._t0 = time.perf_counter()
        n = self.n
        self._present = [True] * n if present is None else [bool(p) for p in present]
        if sum(self._present) == 0:
            raise ValueError("no client present")
        w = [1.0] * n if weights is None else [max(float(x), 1e-12) for x in weights]
        winv = [int(round((1 << FRAC_BITS) / x)) for x in w]
        self._send_public([1] + winv + [int(p) for p in self._present])
        for i in range(n):
            if not self._present[i]:
                continue
            vals = [int(x) for x in sketches_int[i]]
            if self.c:
                vals += [int(x) for x in q_int[i]]
            if self.zk:
                vals += [int(m) for m in mus[i]]
            self.clients[i].send_private_inputs(vals)
        self.stage = "begun"
        if not self.zk:
            return None
        cnt = self.k + self.c + 3 * n
        p = self.prime

        def parse(out):
            coef = [int(v) % p for v in out[:self.k + self.c]]
            rest = out[self.k + self.c:]
            return {"coef": coef, "V1": [int(rest[3 * i]) % p for i in range(n)],
                    "V2": [int(rest[3 * i + 1]) % p for i in range(n)],
                    "in_range": [int(rest[3 * i + 2]) % p == 1 for i in range(n)]}
        res = parse(self.coord.receive_outputs(cnt))
        self.last_observed_zk = [parse(o.receive_outputs(cnt)) for o in self.obs]
        return res

    def finish(self, judged, coll_active, valid=None):
        """Screening over the present clients whose proofs verified (valid). Returns three 0/1
        lists (entries of clients not screened are meaningless)."""
        n = self.n
        valid = [True] * n if valid is None else [bool(v) for v in valid]
        live = [p and v for p, v in zip(self._present, valid)]
        m = max(sum(live), 1)
        lo, hi = (m - 1) // 2, m // 2
        self._send_public([int(coll_active), lo, hi] + [int(bool(j) and l) for j, l in zip(judged, live)]
                          + [int(v) for v in valid])
        out = self.coord.receive_outputs(3 * n)
        self.last_observed = [o.receive_outputs(3 * n) for o in self.obs]
        self.stage = "decided"
        self.pending = True             # the program now waits for the accepted set
        return out[:n], out[n:2 * n], out[2 * n:]

    def decide(self, sketches_int, q_int, weights, judged, coll_active, present=None, mus=None):
        self.begin(sketches_int, q_int, weights, present, mus)
        return self.finish(judged, coll_active)

    def finish_round(self):
        """If a round was stopped part way, close it in the MPC (nobody accepted)."""
        if self.stage == "begun":
            self.finish([False] * self.n, False, [False] * self.n)
        if getattr(self, "pending", False):
            self.sum_accepted([])

    def sum_accepted(self, accepted):
        self.pending = False
        self.stage = None
        acc = set(accepted)
        self._send_public([1 if i in acc else 0 for i in range(self.n)])
        total = np.array(self.coord.receive_outputs(self.k), dtype=np.int64)
        self.last_observed_total = [np.array(o.receive_outputs(self.k), dtype=np.int64) for o in self.obs]
        self.last_seconds = time.perf_counter() - self._t0
        self.total_seconds += self.last_seconds
        self.rounds += 1
        return total

    def stop(self):
        if self.coord is not None:
            try:
                self._send_public([0] + [0] * (2 * self.n))
            except OSError:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        for cl in self.clients + ([self.coord] if self.coord else []) + list(getattr(self, "obs", [])):
            for s in cl.sockets:
                s.close()
        self.procs, self.clients, self.coord, self.obs = [], [], None, []
