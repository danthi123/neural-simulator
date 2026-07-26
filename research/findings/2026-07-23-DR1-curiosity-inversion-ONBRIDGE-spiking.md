# DR-1 curiosity inversion — the ON-BRIDGE SPIKING realization (2026-07-23)

The numpy cheap-first (`_curiosity_seek_learn_cheap_first_probe.py`, 6-seed GO,
`2026-07-23-DR1-curiosity-inversion-6seed-GO.md`) proxied the curiosity modulator + the value at RATE level.
This promotes it to a REAL spiking `SimulationBridge`: the curiosity drive is spiking (the reserved
`from_novelty` neuromodulator hook, now filled), the learning-progress reward runs through the spiking-SNc RPE
machinery, and the epistemic gap is the real reused Bogacz-Brown familiarity gate. Roadmap §2.8 (DR-1 Phase-0
→ on-bridge). Runner `research/runners/_curiosity_seek_learn_onbridge_derisk.py`.

## What was built (the three deliverables)
1. **Curiosity drive = SPIKING (fills `from_novelty`).** A `curiosity` `NeuromodulatorConfig` whose production
   rule is `from_novelty` reads the brain's epistemic-gap scalar `core_config.current_novelty_signal` (the
   familiarity-gate novelty of the concept under consideration — the SAME signal that drives the no-confab
   moat) and drives an `excitability_drive` on a spiking ASK pool (`scope="group:ask"`). HIGH novelty → higher
   curiosity concentration → more ASK-pool SPIKING. The WANTING is read from real `cp_firing_states[ask]` (Hz),
   so gate (a) `corr(gap, wanting)` is measured ON SPIKES.
2. **Learning-progress reward via the spiking-SNc RPE machinery** (the `_limbic_core_rpe_battery` organ:
   `reward_us → snc ← striosome_value(GABA_B)`, `cue → striosome` plastic). Each ASK delivers r = learning
   progress `g_before − g_after` as a reward_us drive; the SNc computes the RPE on spikes; the DA teaching
   signal is **read directly from the SNc firing** (DA release ∝ SNc rate; Schultz) and routes the three-factor
   plasticity so the striosome critic LEARNS a value. The **veto** value is a per-concept expected-learning-
   progress `ELP` (the numpy probe's TD tracker, optimistic init) fed by the spiking reward read =
   `SNc_with_reward − SNc_without_reward` for the same concept (so the learned V cancels and the read isolates
   r). A NOISY/un-learnable concept evokes r≈0 → reward_read≈0 → ELP decays below threshold → the policy STOPS
   asking it, WHILE its gap stays HIGH (never spuriously learned). Curious AND honest (Oudeyer/Schmidhuber
   learning-progress; the noisy-TV cure).
3. **Epistemic gap = the real on-bridge Bogacz-Brown familiarity gate** (`RealAntiHebbianFamiliarity`, catalog
   D.04) — reused-by-import, verbatim. INGEST = imprint the teacher render. Moat-by-construction (only asks
   when NOVEL; confident set ⊆ ingested set).

## The ONE sim/ edit (additive, default-off, byte-identical when off)
- `sim/neuromodulators.py` — filled the reserved `from_novelty` production rule (was `return 0.0`): reads
  `core_config.current_novelty_signal - novelty_baseline`, the exact sibling of `current_reward_signal` for
  `from_reward`.
- `sim/config.py` — added the additive fields `current_novelty_signal: float = 0.0` and
  `novelty_baseline: float = 0.0` (siblings of `current_reward_signal`).
- **Byte-identical proof:** the `from_novelty` branch is only reachable for a registered `from_novelty`
  production rule, and NO config in the repo registered one (grep). Even if one existed, with no novelty signal
  written (default 0.0) it returns `sensitivity*(0−0)=0.0` — identical to the old stub. Verified: (a)
  `tests/test_neuromodulators.py` pass/fail set is BYTE-IDENTICAL before/after (39 pass, 6 pre-existing
  numpy-backend `.get()` harness failures, unchanged); (b) `tests/test_from_novelty_curiosity.py` (6 new tests)
  pins the byte-identical-when-off behavior + the new drive. `git diff --ignore-cr-at-eol --stat sim/`:
  `config.py +8`, `neuromodulators.py +25 -2`.

## Key on-bridge modeling decisions (each a diagnosed substrate issue → fix)
- **DA teaching signal READ from SNc firing, not the autonomous `from_region_firing_signed` integrator.** The
  autonomous da concentration drifts below baseline during the silent inter-ask gap (snc=0 ≪ threshold), mis-
  signing the teacher under the pulsed ask protocol. Reading the SNc firing (DA ∝ rate) is more faithful AND
  robust.
- **SNc uses a rebound-free RS integrator, NOT the IZH2007_DOPAMINE preset.** The dopamine preset post-
  inhibitory-rebound-bursts: the strio's hyperpolarizing GABA_B value-subtraction deinactivates its T-current →
  ~400 Hz runaway once the critic has learned any V (strio just 20 Hz drives snc to 421 Hz). The RS integrator
  computes `snc = tonic + r − V` cleanly.
- **A wash-out (EMERGE-61 pattern) + neuromodulator-concentration reset before each op.** Izhikevich slow
  adaptation `cp_recovery_variable_u` accumulates across asks (the SAME concept reads V=23 early / V=125 late,
  weights unchanged); the curiosity concentration accumulates likewise (the SAME novelty reads a rising want on
  successive ops). Restoring the clean post-init dynamic state + resetting NM concentrations to baseline makes
  every op a drift-free function of the learned weights / novelty signal alone (want-vs-novelty reproducibility
  0.994).
- **The value subtraction (strio→snc) is kept GENTLE** (`strio_to_snc_weight=2`); a strong subtraction over-
  suppresses the fresh SNc to 0 → inverts the RPE sign → runaway.
- **The veto is a spiking-reward-fed ELP TD tracker, not the striosome rate read.** The striosome LEARNS V (it
  is the actor-critic organ, strio value rises for learnable), but the on-substrate striosome RATE read is too
  drift/transient-sensitive at 12 concepts to VETO on directly; the ELP (numpy-faithful) fed by the robust
  spiking reward read is the veto. Honest: this keeps the per-concept expected-LP a TD tracker exactly as the
  numpy probe did, with the DRIVE and the REWARD/RPE now spiking.

## Result — full config (8 learnable + 4 noisy, D=1024, ask-budget 30)
- **seed 42 GO (CPU + GPU):** corr(gap,SPIKING-want) **+0.99** (≥0.9); ask unknown ≫ 2× known; conf-rise
  **+0.55** above the abstain floor (8/8 learnable mastered); NOISY early-rate 0.03 → late 0.00 (g stays 0.97,
  ELP 0.07 ≤ thr 0.12 → vetoed) — curious AND honest; moat holds. Controls collapse: lesion 0 asks; yoked
  masters 7 < real 8; permuted corr −0.08.
- **GPU 6-seed (SIM_BACKEND=cupy, seeds 42/43/44/100/101/102):** _[PENDING — filled on completion]_

## Honest scope
- On-bridge spiking realization validated at 12 concepts (8 learnable + 4 noisy). The two headline properties
  hold on spikes: curiosity drives asking (corr gap↔spiking-want), and the noisy/un-learnable concept STOPS
  being asked (learning-progress reward via the spiking SNc, not novelty) WITHOUT being spuriously learned.
- The per-concept expected-LP VALUE is a host TD tracker (as in the numpy probe) fed by the spiking SNc reward;
  the on-substrate striosome learns a value too but its rate-read is not the veto signal (drift-sensitive at
  scale). The DRIVE and the REWARD/RPE are spiking; the SNc RPE machinery is the reward path.
- NO `sim/` edit beyond the single additive `from_novelty` fill (+ its config sibling fields), byte-identical
  when off.
