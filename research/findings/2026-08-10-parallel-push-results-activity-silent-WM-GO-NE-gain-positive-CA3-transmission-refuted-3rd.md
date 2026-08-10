---
type: finding
status: contributing
date: 2026-08-10
mechanism: buildable-now-faculties
lane: A5 / EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: each result carries its own control-arm decomposition — WM: test (STP on) vs FAIR-ctrl (tau_f=5) vs stp-OFF diagnostic; NE-gain: multiplicative gain vs additive-offset control + byte-identical-when-off; CA3: g_e delivered vs weight (w=0 floor as the null).
---

# Parallel-push results: ACTIVITY-SILENT working memory is a 6/6 GO (Mongillo STP), NE-gain vigilance is an idealized-probe POSITIVE (real-substrate build queued), and the CA3 "transmission wall" is REFUTED a 3rd time

Three of the four aggressive-parallel-push frontiers (each RAG-checked the record first). The CA3 result also
caught + fixed the doc-drift corrected this cycle (a refuted claim I had re-cited).

## 1. ACTIVITY-SILENT working memory (Mongillo 2008) — 6/6 GO

<!--derived-->

New runner `research/runners/_activity_silent_wm_ping_derisk.py` (config-only; NO `sim/` edit). K=4 isolated
excitatory assemblies with within-assembly recurrent E->E (STP ON, `stp_tau_f=1500`, `stp_tau_d=200`). Protocol:
LOAD one assembly (its recurrent synapses facilitate, `cp_stp_u` rises) -> DELAY 400 ms zero-drive -> NONSPECIFIC
uniform ping -> read per-assembly firing. **6/6 GO** (seeds 42/43/44/100/101/102): reactivation via the nonspecific
ping = 0.72-0.90 (chance 0.25; margin +12.6..+19.7 spikes); **delay firing ~0.0004 spikes/neuron/step = genuinely
SILENT** (not persistent activity). Controls: the FAIR control (`tau_f=5`, facilitation minimal) drops reactivation
to 0.15-0.40 -> the FACILITATION is load-bearing; an honest diagnostic flags that with STP fully OFF a persistent
attractor path exists (delayfire 0.068, acc 1.0) — but the TEST condition is verified silent, so the memory is held
in the facilitated `cp_stp_u`, reactivated by a ping = genuine activity-silent WM. **Next: write as a standalone
faculty finding; the persistent-attractor diagnostic is the honest caveat (the recurrent weight sits in a regime a
persistent attractor is possible, suppressed by STP depression in the test).** A5 buildable-now faculty realized.

## 2. NE-gain vigilance (Aston-Jones-Cohen) — idealized-probe POSITIVE, real-substrate build queued

<!--derived-->

Runner `research/runners/_ne_lc_gain_vigilance_derisk.py`. A slow-decay norepinephrine-like MULTIPLICATIVE gain on a
200-LIF population improves weak-signal detection: population spike-count d-prime 2.04 (g=1) -> 3.14 (g=1.5) -> 4.25
(g=2.0), monotone. Controls PASS: byte-identical-when-off (g=1.0 hits the guarded no-op path); the multiplicative
gain (not an additive offset) is what improves d-prime. **Bound: this is an IDEALIZED LIF probe, not the real
bridge.** NEXT = a real-substrate test (`enable_neuromodulator_subsystem=True`, gain on a bridge population) before
a GO — queued.

> **⤷ RESOLVED (2026-08-10, `6ecba7b69`) — the real-substrate test is an HONEST NEGATIVE, the idealized POSITIVE does NOT robustly transfer.** With the gain delivered by the REAL `NeuromodulatorManager.compute_synaptic_gain_multiplier()` (`bridge.py:8167`, scope=all — not a host multiply), d' rises with gain on only **3/6 seeds** (42/43/102); 44/100/101 go flat/negative. GO_ALL=False; mult-beats-additive 3/6. The homogeneous probe replaced per-neuron heterogeneity + OU + adaptive thresholds with constants, so one global operating point lands half the seeds outside the sensitive f-I band → next lever = pair the gain with a rate/threshold homeostat that first places each neuron on its sensitive curve, then apply the gain. (byte-identical-when-off holds 5/6 exactly + 1 harness state-reset residual — a fresh subsystem-ON bridge at g=1.0 is bit-identical to OFF including seed 44, so the guarded-off path does NOT leak.) See `2026-08-10-NE-LC-gain-vigilance-REAL-SUBSTRATE-does-not-robustly-transfer-3of6.md`.

## 3. CA3 recurrents "functionally silent" — REFUTED a 3rd time (+ the doc-drift it caused, fixed)

<!--derived-->

Runner `research/runners/_ca3_recurrent_transmission_scale_probe_SMOKE.py` (direct g_e delivery instrument; NO `sim/`
edit). The "weight-120 recurrent delivers ~0.2 mV / ~1000x too weak / weight-invariant" premise (`2026-07-08-riii-CORRECTION`)
is REFUTED: peak g_e delivered to non-driven CA3 targets (which receive current ONLY via ca3->ca3) scales cleanly
~linearly with weight — w=0 -> 0.0153, w=1 -> 0.104, w=10 -> 1.07, w=100 -> 43.9, w=1000 -> 469.3 (monotone,
~30,000x span; recruits non-cued members at w>=100). The "0.2 mV" was a WEAK-DRIVE floor artifact. This confirms the
2026-07-17 + 2026-07-25 refutations a 3rd time. **The episodic recall residual is ATTRACTOR STRENGTH / SPECIFICITY
(does trained recurrent LTP yield pattern-SELECTIVE completion — a tractable weight×density sweep) OR the
dendritic-plateau completion readout (6-seed GO), a config lever — NOT a sim transmission fix.** The RAG check caught
that I had re-cited the refuted claim as the episodic sim wall across 4 governed files; corrected this cycle
(`29e73b0b`).

Artifact: `research/findings/raw/_activity_silent_wm_ping.json` (WM 6/6 per-seed rows). Reproducers: the three
runners above. NO `sim/` edit in any. SIM_BACKEND=numpy.
