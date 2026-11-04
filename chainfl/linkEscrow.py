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

Trust (state this in the paper): fewer than t members learn nothing about any link. The t
members that open an epoch do learn the links, so the assumption is "no t members collude"
(a distributed version of Leash-FL's trusted authority). Removing even that needs the tally to
run inside MPC, which is future work. Out of scope here: verifiable shares (a client sending a
bad share is detected by the tag check, but not attributed) and posting boxes on chain.
"""
import hashlib
import secrets

from server.aggregation_alg.secaggDH import (dh_agree, dh_keygen, shamir_reconstruct,
                                             shamir_share)

SHARE_BYTES = 66            # Shamir field is 2^521 - 1, so a share fits in 66 bytes


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


def escrow(account_id, s2, member_pks, t):
    """Client side: the box that lets any t of the members recover s2."""
    eph_sk, eph_pk = dh_keygen()
    shares = shamir_share(int.from_bytes(s2, "big"), t, range(1, len(member_pks) + 1))
    cts = [_xor(shares[x].to_bytes(SHARE_BYTES, "big"), _keystream(dh_agree(eph_sk, pk), SHARE_BYTES))
           for x, pk in enumerate(member_pks, start=1)]
    return {"id": account_id, "eph_pk": eph_pk, "ciphertexts": cts, "tag": tag_of(s2)}


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
        """first_round: the first round this account can appear in (rounds before it are skipped)."""
        self.boxes.append(dict(box, first_round=first_round))

    def open_links(self, member_idx):
        """The members in member_idx pool their decrypted shares. Returns {account id: s2}."""
        member_idx = sorted(set(member_idx))
        if len(member_idx) < self.t:
            raise PermissionError(f"{len(member_idx)} members cannot open links, threshold is {self.t}")
        links = {}
        for box in self.boxes:
            shares = dict(self.members[j].decrypt_share(box) for j in member_idx[:self.t])
            secret = shamir_reconstruct(shares)
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
