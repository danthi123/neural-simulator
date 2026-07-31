---
type: finding
status: live
date: 2026-07-15
---

# On-substrate systematicity RUNG 3 (honest NEGATIVE + the named surpass): the from-scratch BDSP-LEARNED *rate* read-out over the fixed spiking bind hits the documented rate-coded-SNR-wall (small discriminative signal on a large common mode) — but the fully-spiking *placed* read is ALREADY validated (the composer's NEF/phasor spiking cleanup), so the fully-spiking systematicity path is achievable, just not via a from-scratch on-bridge rate read

**Date:** 2026-07-15 · **Runner:** `research/runners/_onsubstrate_bind_onbridge_bdsp_readout_derisk.py` (reuse-by-import RUNG-2's `bound_rates` + the committed `enable_bdsp` machinery `_build_bridge`/`_snapshot_state`; numpy backend; NO `sim/` edit). Follows RUNG 1 (`-coincidence-bind-systematicity-RUNG1-GO`) + RUNG 2 (`-bind-learned-readout-RUNG2-GO`).

## The goal + the result
RUNG 3 aimed for the fully-spiking culmination: learn the read-out over the fixed spiking bind ON THE BRIDGE by the committed BDSP rule (apical credit k*(Y@delta), fixed-random Y = no weight transport), so the WHOLE path — bind AND read-out — is spiking + biologically-learned. **Honest result (1-seed, after 3 systematic-debugging fixes): NEGATIVE.** The on-bridge BDSP-learned read-out does not learn the systematicity read-out: train 0.40 (host RUNG-2 read-out fit 1.00), held 0.357 (≈ MLP-on-raw 0.357), and apical-lesion (0.429) ≥ learn (0.357) — the BDSP credit adds nothing over the Wout delta rule on the fixed random pool.

## Root cause (systematic-debugging, instrumented — NOT guessed)
The diagnostic pinned it: (1) the pool barely fires (mean 0.0049 ≈ 0.5% over the read window); (2) **the coincidence-bind bound rates carry a small discriminative signal on a large common mode** — across-example std of B is 0.012, the channels are all in [0.36, 1.0]. Three fixes each nudged without cracking: dense→sparse W_in (Marr-Albus expansion) 0.286→0.286; common-mode removal (mean-subtracted drive) + longer integration 0.286→0.40. A spiking point-neuron read-out cannot amplify a tiny common-mode-riding signal to the host least-squares/transport-free read-out's fidelity.

## ⇒ This IS the project's documented rate-coded-SNR-wall — with an ALREADY-VALIDATED surpass
This is the SAME wall family the project characterized + surpassed on 2026-06-05 (`-B-opponency-rate-coded-SNR-wall-CONFIRMED`): a rate code physically cannot cleanly remove a small common-mode difference (biology does it ANALOG/pre-spike; Mikulasch-Priesemann point-neuron limit). It is exactly WHY the FHRR pivot moved the composer to phasor coding (info in PHASE, no common mode) and why the composer's spiking read-out is a PLACED/NEF cleanup (Stewart-Tang-Eliasmith), NOT a from-scratch learned rate read. So:
- **The fully-spiking PLACED read over the fixed bind is ALREADY VALIDATED** — the composer's NEF/phasor spiking cleanup, deployed at 320 concepts (`consolidated_320_conversation_demo`). ⇒ the fully-spiking systematicity path (RUNG-1 fixed spiking bind + the composer's spiking placed read) is ACHIEVABLE today.
- **RUNG 2 shows the read-out is biologically LEARNABLE** (host transport-free feedback alignment, 5/6 GO).
- The UNACHIEVED intersection — a from-scratch **on-bridge BDSP-LEARNED rate read** over this small-common-mode signal — is the rung bounded by the rate-coded-SNR-wall. Named surpass (project-validated for the FIXED read, the open build for the LEARNED read): learn the read in PHASOR/population coordinates (no common mode) rather than a rate pool, or add the analog common-mode-removing dendritic stage (Mikulasch-Priesemann; the deferred dendritic substrate).

## ⇒ Net (the RUNG-1/2/3 arc, honestly closed)
Systematicity is realized ON THE SUBSTRATE as **a fixed spiking binding primitive (RUNG 1, 6/6 > learner) + a biologically-learned read-out over it (RUNG 2, host transport-free, 5/6 GO)**; the fully-spiking read is achievable via the project's existing placed/phasor spiking cleanup. The one honest open rung is the from-scratch on-bridge BDSP-learned RATE read, bounded by the documented rate-coded-SNR-wall, whose named surpass (phasor/population-coordinate learning, or the dendritic common-mode stage) is a scoped follow-on, not a mystery. NO `sim/` edit. Runner retained for the phasor-read follow-on. (RUNG-1 GPU/cupy confirm still detached-running; the numpy 6-seed is the standing on-substrate claim.)
