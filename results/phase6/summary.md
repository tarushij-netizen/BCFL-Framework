# Phase 6 results

## Main: accuracy, backdoor success, attacker uploads that reached the model

### IID

| attack | defence | n | final acc (%) | final ASR (%) | max ASR r≥3 (%) | attacker uploads in | honest rejections |
|---|---|---|---|---|---|---|---|
| none | fedavg | 1 | 72.2 | – | – | – | 0 |
| none | median | 2 | 73.5 ± 1.4 | – | – | – | 0 ± 0 |
| none | krum | 3 | 72.2 ± 2.4 | – | – | – | 270 ± 0 |
| none | full | 3 | 74.3 ± 1.4 | – | – | – | 12 ± 4 |
| signflip | fedavg | 3 | 10.0 ± 0.0 | – | – | 90 ± 0 | 0 ± 0 |
| signflip | median | 3 | 71.8 ± 0.4 | – | – | 90 ± 0 | 0 ± 0 |
| signflip | krum | 3 | 71.5 ± 2.3 | – | – | 0 ± 0 | 180 ± 0 |
| signflip | full | 3 | 73.4 ± 1.2 | – | – | 0 ± 0 | 4 ± 2 |
| labelflip | fedavg | 3 | 54.4 ± 4.4 | – | – | 90 ± 0 | 0 ± 0 |
| labelflip | median | 3 | 72.9 ± 1.0 | – | – | 90 ± 0 | 0 ± 0 |
| labelflip | krum | 3 | 71.5 ± 2.3 | – | – | 0 ± 0 | 180 ± 0 |
| labelflip | full | 3 | 73.4 ± 1.2 | – | – | 0 ± 0 | 0 ± 0 |
| backdoor | fedavg | 2 | 74.5 ± 0.1 | 87.3 ± 0.9 | 87.5 ± 0.6 | 90 ± 0 | 0 ± 0 |
| backdoor | median | 3 | 73.7 ± 1.2 | 4.0 ± 0.3 | 5.0 ± 0.3 | 90 ± 0 | 0 ± 0 |
| backdoor | krum | 2 | 73.0 ± 0.9 | 1.2 ± 0.8 | 7.6 ± 0.8 | 0 ± 0 | 180 ± 0 |
| backdoor | full | 3 | 73.4 ± 1.2 | 1.8 ± 0.3 | 3.8 ± 0.5 | 0 ± 0 | 0 ± 0 |

### non-IID (Dirichlet 0.5)

| attack | defence | n | final acc (%) | final ASR (%) | max ASR r≥3 (%) | attacker uploads in | honest rejections |
|---|---|---|---|---|---|---|---|
| none | fedavg | 3 | 69.6 ± 1.5 | – | – | – | 0 ± 0 |
| none | median | 3 | 65.9 ± 1.0 | – | – | – | 0 ± 0 |
| none | krum | 3 | 48.7 ± 1.8 | – | – | – | 270 ± 0 |
| none | full | 3 | 69.8 ± 0.8 | – | – | – | 33 ± 6 |
| signflip | fedavg | 3 | 10.0 ± 0.0 | – | – | 90 ± 0 | 0 ± 0 |
| signflip | median | 3 | 50.7 ± 7.6 | – | – | 90 ± 0 | 0 ± 0 |
| signflip | krum | 3 | 47.4 ± 3.4 | – | – | 0 ± 0 | 180 ± 0 |
| signflip | full | 3 | 66.3 ± 4.2 | – | – | 0 ± 0 | 39 ± 7 |
| labelflip | fedavg | 3 | 53.4 ± 5.2 | – | – | 90 ± 0 | 0 ± 0 |
| labelflip | median | 3 | 58.2 ± 6.8 | – | – | 90 ± 0 | 0 ± 0 |
| labelflip | krum | 3 | 46.2 ± 0.7 | – | – | 0 ± 0 | 180 ± 0 |
| labelflip | full | 3 | 65.8 ± 3.8 | – | – | 5 ± 1 | 23 ± 6 |
| backdoor | fedavg | 3 | 69.7 ± 1.3 | 90.1 ± 0.2 | 90.2 ± 0.4 | 90 ± 0 | 0 ± 0 |
| backdoor | median | 3 | 64.8 ± 2.8 | 8.7 ± 3.9 | 11.4 ± 5.2 | 90 ± 0 | 0 ± 0 |
| backdoor | krum | 3 | 48.7 ± 1.8 | 68.2 ± 43.2 | 81.3 ± 26.2 | 19 ± 12 | 199 ± 12 |
| backdoor | full | 3 | 67.0 ± 3.6 | 3.9 ± 2.6 | 15.7 ± 14.3 | 7 ± 1 | 22 ± 2 |

## Screening inside the committee MPC (non-IID)

| attack | system | n | final acc (%) | final ASR (%) | attacker uploads in | honest rejections | MPC s/round |
|---|---|---|---|---|---|---|---|
| none | full, plaintext screening | 3 | 69.8 ± 0.8 | – | – | 33 ± 6 | – |
| none | full, MPC screening | 3 | 70.3 ± 0.5 | – | – | 30 ± 6 | 0.00 ± 0.00 |
| backdoor | full, plaintext screening | 3 | 67.0 ± 3.6 | 3.9 ± 2.6 | 7 ± 1 | 22 ± 2 | – |
| backdoor | full, MPC screening | 3 | 68.1 ± 3.6 | 4.1 ± 1.9 | 8 ± 1 | 27 ± 4 | 0.00 ± 0.00 |

## Attacks on the protocol (full system, IID)

| attack | n | final acc (%) | rounds lost | clients named / left out | rounds with bad shares named | server alarms (rounds) | privacy breaches | wrong models caught |
|---|---|---|---|---|---|---|---|---|
| commitlie | 3 | 71.1 ± 2.1 | 0 ± 0 | 57 ± 1 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| badshare | 3 | 71.2 ± 1.7 | 0 ± 0 | 60 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| badreveal | 3 | 71.6 ± 1.3 | 0 ± 0 | 0 ± 0 | 20 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| split_accepted | 3 | 8.4 ± 1.3 | 20 ± 0 | 0 ± 0 | 0 ± 0 | 20 ± 0 | 0 ± 0 | 0 ± 0 |
| wrong_model | 3 | 71.6 ± 1.3 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 20 ± 0 |

## Cost per round (seconds, this machine, all clients simulated in one process)

| configuration | n | round total | of which aggregation | of which proofs (ZK) | MPC |
|---|---|---|---|---|---|
| iid_none_fedavg | 1 | 0.00 | 0.00 | 0.00 | 0.00 |
| iid_none_median | 2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| iid_none_krum | 3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| iid_none_full | 3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| noniid_none_fullmpc | 3 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |

## Gas on the local chain (full system, --onchain)

| operation | calls | mean gas |
|---|---|---|
| deploy | 1 | 2220311 | 2220311 |
| register (per account) | 10 | 123736 | 139126 |
| beacon postDeal | 25 | 61111 | 61131 |
| postAnchor | 20 | 50254 | 50806 |
| recordRound (10 clients) | 5 | 285589 | 285611 |
| beacon postReveal | 25 | 52877 | 52883 |
| postOutcome | 5 | 27535 | 27538 |
| postVerdicts (10 clients) | 5 | 289362 | 289414 |
| proposeEpoch | 1 | 105089 | 105089 |
| approveEpoch | 2 | 115646 | 164742 |
| distribute | 1 | 308753 | 308753 |

