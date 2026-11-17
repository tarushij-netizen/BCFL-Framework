"""
C2: the pseudonym -> account link, held by the committee as t-of-n secret shares instead of
by one party (this replaces the single Python dict in incentive.py).

A client's account is id = H(s1); its round-r pseudonym is H(s2, r). Whoever knows s2 can
tell which pseudonyms belong to which account, so s2 is the "link". At registration the
client escrows s2:
  - Shamir-splits s2 into n shares, any t of which rebuild it (t = the ledger's k-of-n),
  - encrypts share j to committee member j: key = SHAKE-256(DH(ephemeral_sk, member_pk_j)),
    with one fresh ephemeral DH key per escrow (the same 2048-bit group as secaggDH.py),
  - publishes the box (account id, ephemeral public key, n ciphertexts, tag = H(s2)).
The server never sees s2 or a share in the clear.

At the end of an epoch, t members open the boxes (each decrypts only its own share), rebuild
every s2, check it against the tag, map the posted pseudonyms back to accounts and recompute
every account's reputation from the posted verdicts with the public reputation rule. They
propose and approve that list on the ledger, which applies it only with t approvals.

Verifiable shares (Feldman VSS, 5 Nov): the Shamir polynomial f(x) = s2 + a1 x + ... lives in
Z_q, q = (p-1)/2 the order of g = 2 in the DH group, and the box also carries the commitments
C_j = g^{a_j} mod p (C_0 = g^{s2}). At registration every member decrypts its own share y and
checks g^y == prod_j C_j^(x^j). A member whose share fails files a complaint: the DH value K it
shares with the box plus a Chaum-Pedersen proof that K is really DH(its key, box key). Anyone can
check the complaint, decrypt that one share and see that it does not match the commitments, so
a cheating client is identified (and its registration refused), while a member who complains
falsely is caught by the same check. When the committee opens links, every share is checked
again, so bad shares from members are skipped and any t good ones rebuild s2.

Trust (state this in the paper): fewer than t members learn nothing about any link. The t
members that open an epoch do learn the links, so the assumption is "no t members collude"
(a distributed version of Leash-FL's trusted authority). Removing even that needs the tally to
run inside MPC, which is future work. Out of scope here: posting boxes on chain.
"""
import hashlib
import secrets

from server.aggregation_alg.secaggDH import G, P, dh_agree, dh_keygen

Q = (P - 1) // 2            # P is a safe prime; g = 2 generates the subgroup of prime order Q
SHARE_BYTES = 256           # a share is an element of Z_Q (2047 bits)


def vss_share(secret, t, xs):
    """Feldman VSS: Shamir shares over Z_Q plus public commitments to the polynomial."""
    coeffs = [secret % Q] + [secrets.randbelow(Q) for _ in range(t - 1)]
    shares = {}
    for x in xs:
        y = 0
        for c in reversed(coeffs):
            y = (y * x + c) % Q
        shares[x] = y
    return shares, [pow(G, c, P) for c in coeffs]


def vss_verify(x, y, commitments):
    """Does share (x, y) lie on the committed polynomial?"""
    rhs, xp = 1, 1
    for c in commitments:
        rhs = rhs * pow(c, xp, P) % P
        xp = xp * x % Q
    return pow(G, y % Q, P) == rhs


def vss_reconstruct(shares):
    """{x: y} with >= t entries -> secret (Lagrange at 0 over Z_Q)."""
    total = 0
    for xi, yi in shares.items():
        num, den = 1, 1
        for xj in shares:
            if xj != xi:
                num = num * (-xj) % Q
                den = den * (xi - xj) % Q
        total = (total + yi * num * pow(den, -1, Q)) % Q
    return total


def _in_group(v):
    return 1 < v < P - 1 and pow(v, Q, P) == 1


def _challenge(*vals):
    h = hashlib.sha256(b"dleq" + b"".join(v.to_bytes(256, "big") for v in vals)).digest()
    return int.from_bytes(h, "big") % Q


def dleq_prove(sk, base2):
    """Chaum-Pedersen: prove log_g(g^sk) == log_base2(base2^sk) without revealing sk."""
    A, K = pow(G, sk, P), pow(base2, sk, P)
    r = secrets.randbelow(Q)
    t1, t2 = pow(G, r, P), pow(base2, r, P)
    c = _challenge(G, A, base2, K, t1, t2)
    return K, (t1, t2, (r + c * sk) % Q)


def dleq_verify(A, base2, K, proof):
    t1, t2, z = proof
    c = _challenge(G, A, base2, K, t1, t2)
    return pow(G, z, P) == t1 * pow(A, c, P) % P and pow(base2, z, P) == t2 * pow(K, c, P) % P


def _key_from_dh(K):
    """Same key derivation as secaggDH.dh_agree, from the raw DH value K."""
    return hashlib.sha256(K.to_bytes(256, "big")).digest()


def _keystream(key, n):
    return hashlib.shake_256(b"escrow" + key).digest(n)


def _xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def tag_of(s2):
    return hashlib.sha256(b"link" + s2).digest()


class Member:
    """One committee member: its DH key pair and the shares it has decrypted."""

    def __init__(self, x):
        self.x = x                                  # share coordinate (1..n)
        self._sk, self.pk = dh_keygen()

    def decrypt_share(self, box):
        ct = box["ciphertexts"][self.x - 1]
        y = int.from_bytes(_xor(ct, _keystream(dh_agree(self._sk, box["eph_pk"]), len(ct))), "big")
        return self.x, y

    def check_box(self, box):
        """None if this member's share is valid, otherwise a complaint anyone can verify."""
        x, y = self.decrypt_share(box)
        if vss_verify(x, y, box["commitments"]):
            return None
        K, proof = dleq_prove(self._sk, box["eph_pk"])
        return {"member": self.x, "K": K, "proof": proof}


def complaint_is_valid(box, member_pk, complaint):
    """Public check: the member proved K = DH(member key, box key), and the share it decrypts
    with K does not match the commitments -> the CLIENT cheated. False -> the member lied."""
    K = complaint["K"]
    if not _in_group(K) or not dleq_verify(member_pk, box["eph_pk"], K, complaint["proof"]):
        return False
    ct = box["ciphertexts"][complaint["member"] - 1]
    y = int.from_bytes(_xor(ct, _keystream(_key_from_dh(K), len(ct))), "big")
    return not vss_verify(complaint["member"], y, box["commitments"])


def escrow(account_id, s2, member_pks, t):
    """Client side: the box that lets any t of the members recover s2."""
    eph_sk, eph_pk = dh_keygen()
    shares, commitments = vss_share(int.from_bytes(s2, "big"), t, range(1, len(member_pks) + 1))
    cts = [_xor(shares[x].to_bytes(SHARE_BYTES, "big"), _keystream(dh_agree(eph_sk, pk), SHARE_BYTES))
           for x, pk in enumerate(member_pks, start=1)]
    return {"id": account_id, "eph_pk": eph_pk, "ciphertexts": cts, "commitments": commitments,
            "tag": tag_of(s2)}


class Committee:
    def __init__(self, n_members, t):
        if not 2 <= t <= n_members:
            raise ValueError(f"threshold {t} must be in [2, {n_members}]")
        self.t = t
        self.members = [Member(x) for x in range(1, n_members + 1)]
        self.boxes = []                             # escrow boxes, in registration order

    @property
    def public_keys(self):
        return [m.pk for m in self.members]

    def add_box(self, box, first_round):
        """Every member checks its share. Returns [] and stores the box if all shares are valid;
        otherwise refuses the box and returns the verified complaints (the client cheated).
        first_round: the first round this account can appear in (rounds before it are skipped)."""
        if not _in_group(box["eph_pk"]) or len(box["commitments"]) != self.t \
                or not all(_in_group(c) for c in box["commitments"]):
            raise ValueError("malformed escrow box")
        complaints = [c for m in self.members if (c := m.check_box(box)) is not None]
        proven = [c for c in complaints if complaint_is_valid(box, self.members[c["member"] - 1].pk, c)]
        if proven:
            return proven
        self.boxes.append(dict(box, first_round=first_round))
        return []

    def open_links(self, member_idx):
        """The members in member_idx pool their decrypted shares. Returns {account id: s2}."""
        member_idx = sorted(set(member_idx))
        if len(member_idx) < self.t:
            raise PermissionError(f"{len(member_idx)} members cannot open links, threshold is {self.t}")
        links = {}
        for box in self.boxes:
            # use any t shares that match the commitments (a member sending garbage is skipped)
            shares = {}
            for j in member_idx:
                x, y = self.members[j].decrypt_share(box)
                if vss_verify(x, y, box["commitments"]):
                    shares[x] = y
                if len(shares) == self.t:
                    break
            if len(shares) < self.t:
                raise ValueError(f"escrow of account {box['id'].hex()[:10]}: fewer than {self.t} valid shares")
            secret = vss_reconstruct(shares)
            if secret.bit_length() > 256:
                raise ValueError(f"escrow of account {box['id'].hex()[:10]} does not open (bad shares)")
            s2 = secret.to_bytes(32, "big")
            if tag_of(s2) != box["tag"]:
                raise ValueError(f"escrow of account {box['id'].hex()[:10]} does not match its tag (bad shares)")
            links[box["id"]] = s2
        return links

    def tally(self, links, verdict_log, pseudonym_fn, rule):
        """Recompute every escrowed account's reputation from the posted verdicts.
        verdict_log: list of (round, pseudonyms, accepted flags) as posted on the ledger.
        rule: a fresh sketchSecAgg.Reputation (only its parameters and update rule are used)."""
        by_round = {r: dict(zip(ps, flags)) for r, ps, flags in verdict_log}
        reps = {}
        for box in self.boxes:
            s2, aid = links[box["id"]], box["id"]
            for r in sorted(by_round):
                if r < box["first_round"]:
                    continue
                v = by_round[r].get(pseudonym_fn(s2, r))
                if v is not None:
                    rule.update({aid: bool(v)})
            reps[aid] = rule.get(aid)
        return reps
