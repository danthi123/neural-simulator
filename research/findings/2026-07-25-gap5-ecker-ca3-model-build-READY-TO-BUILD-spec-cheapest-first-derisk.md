# gap#5 Ecker-2022 CA3 replay model-build — READY-TO-BUILD spec (research-gate workflow output): the replay-ignition boundary is 3 coupled ingredients, each isolated by a control; cheapest-first de-risk fully specified (2026-07-25)

## Where gap#5 stands (RAG-verified, not re-derived)
- **ENCODE = WIN, 6/6** (`fe12ce2c`, `2026-07-24-gap5-moving-bump-replay-decode-encode-WIN-replay-BOUNDARY.md`): a sharp
  near-diagonal banded ca3→ca3 matrix via the theta-adjacent-pair encode (adj/skip1 ratio ~8–10). Ports to AdEx.
- **REPLAY-IGNITION = BOUNDARY**, pinned to ONE residual: a WEAK noise seed must SELECTIVELY ignite the right attractor
  and travel A→B→C, WITHOUT a strong drive that non-selectively detonates every assembly (the banked [3,3,3] co-fire).
  SWR-envelope Option 2 confirmed **3/4 ingredients** (discrete ignition ✓, self-termination ✓, noise-seeded ✓; MISSING
  = the attractor-selective forward hand-off). Ordering GIVEN ignition is solved at the rate level (gamma-WTA + post-fire
  silence → 1.000 forward). The Izhikevich boundary was a substrate property (SFA *suppresses* rather than travels), NOT
  a decoder bug (the Bayesian decoder is independently validated: synthetic bump forward r=+0.985).
- **The decoder + encode are already built + unit-tested** (`_gap5_moving_bump_replay_decode.py`,
  `_gap5_sequence_replay_derisk.py`). This build reuses them; the only NEW pieces are an AdEx PC preset + a PVBC pool.

## Ecker 2022 (eLife e71850) — the exact recipe (our substrate class, point neurons, NOT dendritic)
Replay = a moving population bump on the banded matrix, read by Bayesian population decode (Davidson 2009). Three coupled
ingredients fix our boundary, each isolated by one control:
1. **NEGATIVE subthreshold adaptation a=−0.27 nS + LARGE spike-triggered b=206.84 pA** converts a STATIONARY bump into a
   TRAVELING one (isolated by ADAPT-LESION → stationary). This is the mechanistic crux vs Izhikevich SFA (which suppressed).
2. **nS-calibrated band** (PC→PC 0.1–6.3 nS, E-E scaling ≥0.9×) strong enough to re-ignite the next assembly before
   b-adaptation extinguishes the current bump (isolated by NO-BAND/structure-shuffle → abolishes replay).
3. **PVBC→PC feedback inhibition** localizes the bump so a WEAK seed ignites ONE location and travels, rather than a
   strong drive detonating all (isolated by NO-PVBC → fails to localize, the [3,3,3] residual).

Ecker PC AdEx preset: a=−0.27, b=206.84, tau_w=84.93, C=180.13, gL=4.31, EL=−75.19, DeltaT=4.23, VT=−24.42, Vr=−29.74,
Vpeak=−3.25.

## STEP 0 (config, ~30 min) — the ECKER_CA3_PC preset + verification
Add `ECKER_CA3_PC` to `sim/enums.py DefaultAdExParamsManager` (additive; a preset dict — the ONLY `sim/` touch on the
cheap path). CRITICAL verifies: (a) `cfg.adex_a` default is **+4.0 (WRONG SIGN)** — print the scalar the kernel
(`bridge.py:7537`) actually receives and **assert a<0 end-to-end** (the negative-a is the crux; risk it's silently not
applied). (b) Seed: set `cfg.seed` (NEVER `actual_seed_used` — the documented no-op-field gotcha); build twice, hash
`cp_neuron_firing_thresholds` identical to prove the seed controls the substrate.

## STEP 1 = THE CHEAP DE-RISK (single seed 42, ~1–2 GPU-hours over the E-E scaling sweep) — the go/no-go
New thin driver `research/runners/_gap5_ecker_ca3_replay_derisk.py`, minimal 2-region model:
- **PC region** n=2000, `NeuronModel.ADEX`, `ECKER_CA3_PC`; band = the WON theta-adjacent-pair sharp matrix (reuse
  `_prepare_sequence` + `chain_adjacent_pairs`, `chain_rule='hebb_sym'`) recalibrated to Ecker nS (E-E scaling swept
  0.90/0.95/1.00/1.05×, tau_d=9.5 ms).
- **PVBC pool** 40 FS-Izhikevich (`IZH2007_FS_CORTICAL_INTERNEURON` — non-adapting, b≈0.91 pA≈0, a faithful stand-in);
  pathways PC→PVBC (0.85 nS, 0.10) + PVBC→PC (0.65 nS, 0.25, Einh=−70). Drop PVBC→PVBC + the 15 Hz spontaneous mode for
  this pass.
- **Ignition**: a 200 ms / 20 Hz Poisson train (`POISSON_SPIKE_TRAIN`) into one 100-cell location, plasticity frozen.
- **Read**: `_decode_replay` on PC spikes; `_decoder_unit_test` must pass in the SAME run (guards the decoder).
- **GO (this cheap pass)**: the cued event produces a decodable localized traveling trajectory |r|>0.6 whose sign
  matches the cued location, AND the adapt-lesion arm (a=0,b=0 ExpIF, mossy doubled ~38 nS) produces a STATIONARY bump
  (velocity~0). GO on ANY E-E scaling point → the Ecker recipe fixes ignition → proceed to the 6-seed GO; NO across the
  whole sweep → the two controls (adapt-lesion / no-PVBC) isolate WHICH ingredient is short BEFORE any 8000-neuron spend.

## KEY ENGINEERING CONSTRAINT (verified from the code)
The AdEx kernel is **GLOBAL-SCALAR** (`bridge.py:7537` reads scalar `cfg.adex_a/b/tau_w`; the per-region overlay at 1801
writes the same globals → last-region-wins). **One bridge cannot host PC (a=−0.27) AND PVBC (a=3.05) as two AdEx types.**
MITIGATION (cheap + faithful): PC = global AdEx (Ecker params), PVBC = FS-Izhikevich/LIF (b≈0 → non-adapting anyway).
Only a truly two-AdEx model later needs per-neuron `cp_adex_a/b/tau_w` heterogeneity arrays (additive `sim/` edit, NOT on
the cheap path).

## 6-SEED GO GATE (at Ecker scale 8000 PC + 150 PVBC, after the cheap pass GOes)
Cued-ignition (200ms/20Hz into a 100-cell location, plasticity frozen), Bayesian decode (dt=10ms, 50 bins, place cells).
An event counts iff Rmax (constant-velocity band-fit, |v| 0.3–18 m/s) beats the 95th pct of 200 cell-identity
column-shuffles (≡ |r|>0.6, per-event position-shuffle p<0.05). GO requires ALL: (1) ≥3 significant traveling events/seed
in ≥5/6 seeds, decoded direction matches the cue on ≥80% of cued trials; (2) in-run decoder unit test passes; (3) seed
verification passes; (4) all THREE controls collapse to ~0. **Report per-seed clean-event vectors** (e.g. [4,3,5,3,6,4]),
NOT a mean — directly comparable to the banked Izhikevich boundary [0,0,0,0,0,0] and its control-outperforms-real signature.

## ANTI-CHEATS (the result IS the controls)
NO-BAND/structure-shuffle → 0 events (Ecker's own); NO-PVBC → fails to localize (targets the [3,3,3] residual);
ADAPT-LESION → stationary bump (isolates the neg-a/large-b traveling crux — MUST be shown necessary); NO-ENCODE
ignition-selectivity gate (cue a bandless store → no trajectory, guards detonator-not-seed); per-event cell-identity +
position shuffle null; in-run decoder POSITIVE control; reverse-store + interior-seed-invariance; seed-actually-controls
(hash thresholds); **control-outperforms-real check** (the banked boundary FAILED here — adapt-lesion [0,6,6,0,11,0] beat
real [0,0,0,0,0,0]; a GO is invalid if any control matches/beats real).

## REUSE (almost everything exists; NO `sim/` edit beyond the preset)
Decoder DONE + unit-tested: `_gap5_moving_bump_replay_decode.py` (`_bayes_decode` Zhang-1998, `_weighted_corr`
Davidson-2009, `_decode_replay` 200-shuffle 95th-pct, `_decoder_unit_test`/`_synthetic_bump` forward r=+0.985).
Encode (the 6/6 WIN): `_gap5_sequence_replay_derisk.py` (`_prepare_sequence`, `chain_adjacent_pairs`,
`chain_rule='hebb_sym'` = Ecker symmetric-STDP). SWR envelope + 9 anti-cheats: `_gap5_swr_envelope_replay_derisk.py`.
Cue: `experiment/stimulus.py POISSON_SPIKE_TRAIN`. Wiring: `sim/regions.py` + `inject_explicit_wiring`. Ordering assist
if needed: `_gap5_gamma_wta_replay_derisk.py`.

## Verdict + next
Ready-to-build. **NEXT ACTION:** STEP 0 (add `ECKER_CA3_PC` + verify a<0 reaches the kernel + seed-hash), then STEP 1
(build `_gap5_ecker_ca3_replay_derisk.py`, run the single-seed cued-ignition decode over the E-E scaling sweep). GO → the
Ecker recipe fixes the ignition boundary → 6-seed. Source: research-gate workflow `wf_cae24e10-13f` (3 agents, journal.jsonl).
