// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/*
 * C2: privacy-preserving reputation and rewards for BCFL.
 *
 * Identities
 *   - A client's long-lived reputation account is an opaque id (e.g. keccak256(secret1)) plus a
 *     payout address the client chose. Neither is linked on-chain to what the client uploads.
 *   - Every round the client uploads under a fresh pseudonym (e.g. keccak256(secret2, round)).
 *     Only commitments (hashes) of the masked update go on-chain, never the update.
 *   - The pseudonym -> account link is known only to a k-of-n committee (held off-chain).
 *
 * Flow per round   : server records {pseudonym, update hash} and the screening verdict.
 * Flow per epoch   : the committee turns verdicts into reputation changes per account; a change is
 *                    applied only when `threshold` members approve the same list (k-of-n). Observers
 *                    see one net change per account per epoch, not which round or pseudonym caused it.
 * Rewards          : each epoch's pool is split in proportion to reputation among accounts at or
 *                    above `tau`. New accounts start at `initRep` (< tau) and must earn trust, so
 *                    making many fake identities (Sybil) earns nothing at first and costs `stake` each.
 * Reputation is an integer in [0, 1000] (1000 = 1.0).
 */
contract ReputationLedger {
    address public server;
    uint256 public threshold;
    uint256 public initRep;
    uint256 public tau;
    uint256 public stake;

    mapping(address => bool) public isMember;
    uint256 public memberCount;

    struct Account {
        bool active;
        uint256 rep;
        address payout;
    }
    mapping(bytes32 => Account) public accounts;
    bytes32[] public accountList;

    // round => pseudonym => commitment / verdict
    mapping(uint256 => mapping(bytes32 => bytes32)) public updateHash;
    mapping(uint256 => mapping(bytes32 => uint8)) public verdict; // 0 none, 1 accepted, 2 rejected

    struct Proposal {
        bytes32 digest;
        uint256 approvals;
        bool applied;
    }
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public approvedBy;

    mapping(address => uint256) public claimable;
    uint256 public totalDistributed;

    event Registered(bytes32 indexed id, address payout);
    event RoundRecorded(uint256 indexed round, uint256 n);
    event VerdictsPosted(uint256 indexed round, uint256 accepted, uint256 rejected);
    event EpochProposed(uint256 indexed epoch, bytes32 digest, address by);
    event EpochApplied(uint256 indexed epoch, uint256 nAccounts);
    event RewardsDistributed(uint256 indexed epoch, uint256 amount, uint256 eligible);
    event Claimed(address indexed payout, uint256 amount);

    modifier onlyServer() {
        require(msg.sender == server, "only server");
        _;
    }
    modifier onlyMember() {
        require(isMember[msg.sender], "only committee");
        _;
    }

    constructor(address[] memory committee, uint256 _threshold, uint256 _initRep, uint256 _tau, uint256 _stake) {
        require(_threshold > 0 && _threshold <= committee.length, "bad threshold");
        require(_initRep <= 1000 && _tau <= 1000, "rep out of range");
        server = msg.sender;
        for (uint256 i = 0; i < committee.length; i++) {
            require(!isMember[committee[i]], "duplicate member");
            isMember[committee[i]] = true;
        }
        memberCount = committee.length;
        threshold = _threshold;
        initRep = _initRep;
        tau = _tau;
        stake = _stake;
    }

    // ------------------------------------------------------------------ accounts
    function register(bytes32 id, address payout) external payable {
        require(!accounts[id].active, "already registered");
        require(msg.value >= stake, "stake too low");
        require(payout != address(0), "no payout address");
        accounts[id] = Account(true, initRep, payout);
        accountList.push(id);
        emit Registered(id, payout);
    }

    function accountCount() external view returns (uint256) {
        return accountList.length;
    }

    function repOf(bytes32 id) external view returns (uint256) {
        return accounts[id].rep;
    }

    // ------------------------------------------------------------------ per round
    function recordRound(uint256 round, bytes32[] calldata pseudonyms, bytes32[] calldata hashes) external onlyServer {
        require(pseudonyms.length == hashes.length, "length mismatch");
        for (uint256 i = 0; i < pseudonyms.length; i++) {
            require(updateHash[round][pseudonyms[i]] == bytes32(0), "pseudonym reused");
            updateHash[round][pseudonyms[i]] = hashes[i];
        }
        emit RoundRecorded(round, pseudonyms.length);
    }

    function postVerdicts(uint256 round, bytes32[] calldata pseudonyms, bool[] calldata accepted) external onlyServer {
        require(pseudonyms.length == accepted.length, "length mismatch");
        uint256 nAcc = 0;
        for (uint256 i = 0; i < pseudonyms.length; i++) {
            require(updateHash[round][pseudonyms[i]] != bytes32(0), "unknown pseudonym");
            require(verdict[round][pseudonyms[i]] == 0, "verdict already set");
            verdict[round][pseudonyms[i]] = accepted[i] ? 1 : 2;
            if (accepted[i]) nAcc++;
        }
        emit VerdictsPosted(round, nAcc, pseudonyms.length - nAcc);
    }

    // ------------------------------------------------------------------ per epoch (k-of-n)
    function _digest(uint256 epoch, bytes32[] calldata ids, uint256[] calldata newReps) internal pure returns (bytes32) {
        return keccak256(abi.encode(epoch, ids, newReps));
    }

    function proposeEpoch(uint256 epoch, bytes32[] calldata ids, uint256[] calldata newReps) external onlyMember {
        require(ids.length == newReps.length, "length mismatch");
        require(proposals[epoch].digest == bytes32(0), "already proposed");
        bytes32 d = _digest(epoch, ids, newReps);
        proposals[epoch] = Proposal(d, 1, false);
        approvedBy[epoch][msg.sender] = true;
        emit EpochProposed(epoch, d, msg.sender);
        if (threshold == 1) _apply(epoch, ids, newReps);
    }

    function approveEpoch(uint256 epoch, bytes32[] calldata ids, uint256[] calldata newReps) external onlyMember {
        Proposal storage p = proposals[epoch];
        require(p.digest != bytes32(0), "no proposal");
        require(!p.applied, "already applied");
        require(!approvedBy[epoch][msg.sender], "already approved");
        require(p.digest == _digest(epoch, ids, newReps), "different update");
        approvedBy[epoch][msg.sender] = true;
        p.approvals += 1;
        if (p.approvals >= threshold) _apply(epoch, ids, newReps);
    }

    function _apply(uint256 epoch, bytes32[] calldata ids, uint256[] calldata newReps) internal {
        for (uint256 i = 0; i < ids.length; i++) {
            require(accounts[ids[i]].active, "unknown account");
            require(newReps[i] <= 1000, "rep out of range");
            accounts[ids[i]].rep = newReps[i];
        }
        proposals[epoch].applied = true;
        emit EpochApplied(epoch, ids.length);
    }

    // ------------------------------------------------------------------ rewards
    function distribute(uint256 epoch) external payable onlyServer {
        require(proposals[epoch].applied, "epoch not applied");
        uint256 total = 0;
        uint256 eligible = 0;
        for (uint256 i = 0; i < accountList.length; i++) {
            uint256 r = accounts[accountList[i]].rep;
            if (r >= tau) {
                total += r;
                eligible++;
            }
        }
        if (total > 0) {
            for (uint256 i = 0; i < accountList.length; i++) {
                Account storage a = accounts[accountList[i]];
                if (a.rep >= tau) claimable[a.payout] += (msg.value * a.rep) / total;
            }
            totalDistributed += msg.value;
        }
        emit RewardsDistributed(epoch, msg.value, eligible);
    }

    function claim() external {
        uint256 amount = claimable[msg.sender];
        require(amount > 0, "nothing to claim");
        claimable[msg.sender] = 0;
        (bool ok, ) = payable(msg.sender).call{value: amount}("");
        require(ok, "transfer failed");
        emit Claimed(msg.sender, amount);
    }
}
