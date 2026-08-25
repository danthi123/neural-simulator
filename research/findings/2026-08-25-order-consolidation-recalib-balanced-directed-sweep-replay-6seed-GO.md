---
type: finding
status: live
date: 2026-08-25
mechanism: replay-cortical-consolidation-v7-balanced-directed-sweep-order-STDP
runner: research/runners/_replay_cortical_consolidation_gate_v7_balanced_order.py
builds-on: research/findings/2026-08-06-replay-cortical-consolidation-v6-multiseed-NO-GO-operating-point-overfit.md
supersedes-method: v6-episode-agnostic-random-noise-replay-at-a-fixed-2-seed-operating-point
artifacts:
  - research/findings/raw/order_recalib/v7_decisive_numpy.json
  - research/findings/raw/order_recalib/v7_decisive_numpy.json.prov.json
  - research/findings/raw/order_recalib/v7_generalization_numpy.json
  - research/findings/raw/order_recalib/v7_generalization_numpy.json.prov.json
---

# Order-sensitive consolidation recalibration: BALANCED directed-sweep replay + isolated order-STDP clears the replay-order gate 6/6 on the decisive seeds (and 4/5 on the disjoint v6-NO-GO seeds) — board #130 CALIBRATION_NEEDS_REVISION resolved

<!--derived-->
**Verdict: 6-seed GO on the preregistered replay-order gate (probe_steps=45), with every anti-cheat intact, and it GENERALISES** (not a re-fit).
Decisive seeds 42/43/44/100/101/102: `intact_beats_shuffled_order` (margin >= +0.01) passes 6/6 (margins +0.013..+0.030); both memories recover 6/6; retest false recall 0.000..0.032 (<= 0.15).
The ordered-replay cortical trace is physically stronger than shuffled 6/6; the STDP-off power control collapses the ordered-vs-shuffled margin to EXACTLY 0.0000 on all 6; the four causal-lesion controls drop hippocampus-independent recall to <= 0.005.
Documented residual (below): the BEHAVIOURAL margin is a recall-SPEED effect that attenuates at longer probe windows; the PHYSICAL trace strengthening is probe-independent. cupy confirmation is queued (`gpu_queue`).

## Why the calibration needed revision (diagnosis, lesion-grade)

<!--derived-->
Board #130 / `replay_v6_order_stdp_calib.json` read CALIBRATION_NEEDS_REVISION. Two compounding causes, both symptoms of one disease the v6 multiseed NO-GO already named (a FIXED interference operating point hand-fit to 2 seeds):

1. **Backend drift.** The v6 GO was calibrated on `SIM_BACKEND=numpy` (prov sha `ed6bed93e`); the #130 artifact re-ran the IDENTICAL frozen config on `SIM_BACKEND=cupy` (prov sha `a7409d5f4`) and even the calibration seeds 412/413 fell to false recall 0.5 -> NEEDS_REVISION. The operating point was not backend-invariant.
2. **Replay winner-take-all (the load-bearing failure).** Running frozen v6 on the fresh decisive seeds (numpy) gives order 2/6 and false recall ~0.5 on 5/6 — but the 0.5 is BIMODAL, not degenerate-equal: one memory is recalled cleanly and the other is fully EVICTED (e.g. seed 42: A correct_rate 0.000 / false 1.000, B correct 0.093 / false 0.000).
   Root cause is the SLEEP side: v6 replay is EPISODE-AGNOSTIC random CA3 background noise, and memory B (encoded with more events, 20 vs 14 -> stronger CA3 attractor) wins the replay competition on nearly every event (seed 101: A replayed 0/24, B 24/24). Only the replay-winner consolidates, so `both_memories_recovered` fails and the order margin is measured on one noisy memory (all 6 margins positive but tiny, +0.011 mean; the +0.01 bar passes only 2/6).

<!--derived-->
Critically, the order-STDP MECHANISM was never the problem: on all 6 fresh seeds the ordered (intact) replay strengthens the cortical cue->target trace 1.06..2.30x more than shuffled. The directional trace is deposited exactly as claimed; the DEGENERATE readout (winner-take-all replay) hid it.

## The revision (biology-grounded, the v6-named next mechanism)

<!--derived-->
Real hippocampal replay reactivates STORED trajectories in temporal order (a directed sweep through an experienced sequence), not random cell noise, and across sleep it visits MULTIPLE recent memories -- the CLS interleaving that prevents catastrophic interference (McClelland/McNaughton/O'Reilly 1995). v7 makes two changes and inherits everything else in v6 UNCHANGED (order-STDP, learned CA1->cortex reinstatement, SFA eviction, every control, the frozen verdict):

1. **Balanced DIRECTED-SWEEP replay** (`replay_plan="directed_sweep"`). One long replay trajectory whose 12-cell drive window sweeps across the ordered CA3 cell list `[A-only .. shared .. B-only]` over the 24 sleep events: it drives memory A early and B late, so BOTH consolidate regardless of which attractor is stronger (fixes `both_memories_recovered`), while adjacent windows overlap ~10/12 and distant windows are disjoint -> a STRONG directional sequence that shuffling (the temporal control) sharply breaks.
2. **Isolated order-STDP** (`sleep_hebbian_on=False`). v6's order-BLIND rate-window Hebbian sleep baseline is identical for ordered and shuffled (permuting events preserves the coactivity multiset) and dilutes the order-specific contribution; removing it makes the ordered-vs-shuffled margin the WHOLE effect and drops retest false recall to ~0 (no order-blind cross-memory transfer). It is an existing v6 config knob set to the value that ISOLATES the gated capability -- not a per-seed parameter fit.

## Result (numpy; provenance sidecars record argv + git SHA)

<!--derived-->
Decisive gate `research/findings/raw/order_recalib/v7_decisive_numpy.json`, seeds 42/43/44/100/101/102. Per-seed GO requires ALL of: behavioural order margin >= +0.01 (probe_steps=45), both memories recovered, the STDP-off order margin at least +0.01 lower (STDP load-bearing for order), the physical ordered-trace stronger than shuffled (probe-independent), and the four causal lesions <= 0.005.

| seed | order margin | STDP-off margin | both rec | trace I>S | lesions~0 | false | per-seed |
|---:|---:|---:|:---:|:---:|:---:|---:|:---:|
| 42 | +0.0215 | 0.0000 | yes | yes | yes | 0.000 | GO | <!--derived-->
| 43 | +0.0187 | 0.0000 | yes | yes | yes | 0.016 | GO | <!--derived-->
| 44 | +0.0132 | 0.0000 | yes | yes | yes | 0.000 | GO | <!--derived-->
| 100 | +0.0194 | 0.0000 | yes | yes | yes | 0.024 | GO | <!--derived-->
| 101 | +0.0299 | 0.0000 | yes | yes | yes | 0.000 | GO | <!--derived-->
| 102 | +0.0257 | 0.0000 | yes | yes | yes | 0.032 | GO | <!--derived-->

## Generalisation — this is NOT a re-fit to the decisive seeds

<!--derived-->
`v7_generalization_numpy.json`, run on the DISJOINT v6 partition 412/413/414/415/410 (never tuned here). The order gate passes 4/5 (412/413/414/410 GO; 415 misses ONLY the behavioural margin at +0.0090, a hair under +0.01), with both-recovered 5/5, trace-stronger 5/5, false recall ~0, all anti-cheats intact. The v6 multiseed NO-GO failed on the EXACT dev seeds 414/415/410 (false recall 0.46..0.50, order margin ~0, 0/3); v7 gives them false recall 0.000/0.000/0.039 and order margins +0.0181/+0.0090/+0.0167 (2-3/3). The revision repairs the documented v6 failure rather than fitting new seeds -- combined 10/11 seeds pass the behavioural order gate; 11/11 pass both-recovered + trace-stronger + all anti-cheats.

## Residual (no-defer) and next mechanism

<!--derived-->
The behavioural recovery advantage is a recall-SPEED (latency) effect, not an asymptotic-capacity one.
Sweeping the retest window shows the ordered-vs-shuffled margin attenuates monotonically -- order pass 6/6 at probe_steps=45 (mean margin +0.021) -> 2/6 at 90 (+0.009) -> 1/6 at 120 (+0.005): the stronger ordered trace reaches its attractor FASTER, and shuffled catches up given a longer probe.
The PHYSICAL cortical-trace strengthening (a sleep-time weight measurement) is probe-independent and stable (ratio ~1.2-1.5x) across all windows, so "ordered replay strengthens the sequence more than shuffled" holds at the synaptic level throughout; only the behavioural rate readout is window-sensitive.
Named next mechanism: increase the ordered/shuffled trace-strength RATIO so the advantage persists asymptotically -- a plateau-gated (BTSP-style) or larger-amplitude timing rule, and/or a homeostatic replay-strength set-point that self-scales per brain -- and CONFIRM on cupy (queued).
Remaining scaffolds: the directed replay sweep is host-scheduled (same scaffold class as v6's episode-agnostic background, arguably more faithful as stored-trajectory reactivation); wake episode populations, opponent-channel membership, down-state boundaries, assembly anatomy, and the SFA/STDP amplitudes remain host-set at build.
This is a verdict on the ORDER capability at the preregistered gate; it does not weaken any frozen v5 criterion.
