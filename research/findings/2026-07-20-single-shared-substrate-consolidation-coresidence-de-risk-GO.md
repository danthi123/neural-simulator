# Single-shared-substrate consolidation — CO-RESIDENCE de-risk GO (WKV read-out + composer RF phasor on ONE bridge)

**Date:** 2026-07-20 · **Status:** GO, 6-seed — the WKV cortex's read-out state (`cp_ssm_state` + `cp_ssm_readout_w`)
and the composer's RF phasor substrate (`cp_rf_*` + `rf_kick`/`rf_resonate_steps`/`rf_read_phases` on a masked slice)
CO-RESIDE on ONE `SimulationBridge`, each byte-identical to its own isolated bridge, neither corrupting the other.
The cheap-first crux of the owner's "fully-spiking, one brain, single shared substrate" end goal. NO `sim/` edit.

## The end-goal item this de-risks

"Fully closing all gaps INHERENTLY means fully-spiking, one brain, single shared substrate." De-risk 5 had the
composer + the WKV cortex CO-EXECUTING in one PROCESS but on SEPARATE cupy bridges. The consolidation is onto ONE
bridge. This proves the crux — the two persistent spiking states can share a bridge byte-clean.

## Why it holds (read from the step loop + RF ops, `bridge.py`)

- The ssm block (`bridge.py:5958`) + the graded read-out (`:5966`) run UNCONDITIONALLY on array presence, INDEPENDENT
  of `neuron_model_type`, and read `cp_ssm_inject`/`cp_ssm_shunt`/`cp_ssm_state` ONLY (set by the runner) — the
  composer touches none of them.
- The RF ops (`rf_kick`/`_rf_advance_one`/`rf_resonate_steps`) use `v`/`u` (masked to the RF slice) + `cp_rf_*` ONLY,
  do NOT dispatch on `neuron_model_type`, and the composer RE-KICKS its phasor EACH op (De-risk 5b) — so the WKV's
  Izhikevich step between ops is harmless (it corrupts `v`/`u`, but the next composer op re-initializes it).
- ⇒ disjoint persistent arrays (`cp_ssm_*` vs `cp_rf_*`); the only shared array (`v`/`u`) is re-initialized by the
  composer's kick each op.

## Result (`_gap_onebridge_coresidence_derisk.py`, 6-seed 42/43/44/100/101/102)

One bridge C holds BOTH the read-out (a fixed random `cp_ssm_readout_w`) AND a toy RF binding on the last-24-neuron
slice. Interleaved rounds (WKV charge/step/read, then composer kick/resonate/read), compared to ISOLATED bridges
A (read-out only) and B (RF only):

- **WKV read-out co-resident vs isolated: `max|err| = 0.000e+00` (byte-clean) — all 6 seeds.**
- **Composer phase co-resident vs isolated: `max|err| = 0.000e+00` (byte-clean) — all 6 seeds.**
- **ANTI-CHEAT (no-rekick): the composer SKIPS its kick after round 0 → the WKV's Izhikevich step corrupts the shared
  `v`/`u` → phases DIVERGE from isolated by `0.959`–`0.993`** — proving `v`/`u` is genuinely shared and the composer
  re-kick is load-bearing (so the byte-cleanliness is NOT a trivial two-untouched-bridges artifact; the substrate
  really is shared, and co-residence works BECAUSE of the re-kick).

CI: `tests/test_onebridge_coresidence.py` (3 tests, GPU-only, skips on numpy).

## Read-out — honest scope

- **⇒ the WKV cortex's read-out state and the composer's RF phasor CO-RESIDE on ONE bridge byte-clean** — the crux of
  the single-shared-substrate consolidation is proven. The two mechanisms use disjoint persistent arrays and the one
  shared array (`v`/`u`) is safely re-initialized by the composer per op.
- **This is the CRUX de-risk, NOT the full consolidation.** The full end-goal build wires the ACTUAL composer (the RF
  encoder + the fact store + the parser) + the ACTUAL WKV faculty (with its own RF spike-encoder sub-bridge) onto
  ONE bridge and runs a full grounded conversational turn (comprehend → reason → spiking render) on that single
  substrate. This de-risk removes the central risk (do the two persistent spiking states conflict on one bridge? —
  NO) so that build is now an integration/wiring arc, not a research question.
- **Next toward the end goal:** (1) merge the WKV's own RF spike-encoder sub-bridge onto the same bridge (the same
  masked-RF pattern, a second RF slice); (2) run the full De-risk-5 grounded turn on ONE consolidated bridge; (3) the
  on-bridge fluency THROUGHPUT lever (batched stepping) to reach the off-bridge ppl ~40 live.

Runner: `_gap_onebridge_coresidence_derisk.py` (`--seed`, `--rounds`, `--D`, `--n-resonate`).
