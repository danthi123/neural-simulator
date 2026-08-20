---
type: finding
status: live
date: 2026-08-20
mechanism: idle-tick-replay-reactivation
lane: continuous-substrate
seeds: [42]
seed-waiver: A single-seed TRANSFER/LOCUS probe with WITHIN-RUN adversarial controls (reproducibility ×6, volley-robustness ×4 content-blind draws, lesion-teeth) that each refute the one apparent hit — the decisiveness is the controls, not a seed population; per the runner's own rule 6-seed is the natural next rung ONLY if single-seed is GO, and it is NO-GO.
instrument: research/runners/_idle_replay_dgec_afferent_d5_derisk.py — content-blind DG/EC afferent (dg->ca3 mossy) volley into the UNCHANGED production EpisodicDapMemory, structure-awareness measured (not assumed) against a never-stored control
runner: research/runners/_idle_replay_dgec_afferent_d5_derisk.py
external: NO-EXTERNAL-NEEDED — the next-mechanism biology (the SWR brain state) is ALREADY established in [[2026-07-24-gap5-SWR-state-readout-research-gate-the-missing-biology]] (Buzsáki 2006; Ecker 2022 eLife e71850). This NO-GO re-confirms that arc from a new locus; it does not open a new literature question.
artifacts:
  - research/findings/raw/_idle_replay_dgec_afferent_d5/seed42.json
---
# NO-GO: content-blind DG/EC-afferent replay does NOT reactivate the real D5 store — the afferent locus is wrong; the SWR brain state is the path

Artifact: research/findings/raw/_idle_replay_dgec_afferent_d5/seed42.json

**One line.** To wire LEARN-THROUGH-USE live, the emergent pattern-completion replay must reactivate the REAL episodic
store (`EpisodicDapMemory`) during idle. The prior probe ([[2026-08-20-idle-replay-on-d5-episodic-transfer-UNDEFINED-substrate-present-instrument-first]])
drove untargeted CA3 noise (0/0) and named the biologically-principled successor: drive the DG/EC AFFERENT (mossy
detonator), content-blind, so the projection concentrates on the stored assembly (SWR-initiation biology). This de-risk
BUILT + TESTED that successor on the unchanged production organ. **Result: NO-GO — the afferent locus does not carry
structure-aware reactivation, for two identified reasons — and it re-derives, from a new locus, the SAME conclusion a
prior arc already reached: the missing biology is the SHARP-WAVE-RIPPLE (SWR) brain STATE, not a better driver.**

## The measurement + its self-refutation (the controls are the finding)
<!--derived-->
A gentle→strong content-blind mossy-scale sweep produced ONE apparent hit (mb=10, dg=900pA: dog apical 0.578, cat 0.045;
dw dog 7.48 vs cat 1.85) that looked like specific pattern-completion + a specific gated write. THREE within-run
adversarial controls, all built into the runner, demolished it:
- **REPRODUCIBLE=False** — at the identical dose + IDENTICAL content-blind DG pattern, the completion fired **1/6** re-drives (repro_fire_rate 0.167). A reliable transfer does not fire one time in six on a fixed input; it is a state/phase-dependent transient.
- **VOLLEY_ROBUST=False** — across **4** different content-blind DG draws at the same dose, dog>>cat held on **0/4** (all apical gaps 0.000). The specificity was one lucky draw, not a store-carried effect.
- **LESION_COLLAPSES=False** — under UNFORMED baseline recurrent weights the afferent still drove dog apical to **0.850** (≈ cat 0.854). When the afferent fires at all it excites both assemblies non-specifically; the apparent dog-specificity was NOT carried by the stored recurrence.
Runner's own verdict: `NO-GO / HONEST-TRANSFER-NEGATIVE (instrument validated, afferent engaged)`, `GO=False`.

## The instrument was FIXED first (a precondition for trusting any negative)
<!--derived-->
The prior probe was UNDEFINED because its `recall_reproducible` gate required per-draw std ≤ 0.10 — impossible on dog's
7-held-cell read (quantization floor 1/7 ≈ 0.14). Replaced with a **signal-detectability** criterion: K=5 averaged
draws, stored-vs-unstored separation must exceed the measured read jitter. Separation 0.600 ≫ noise floor 0.239 →
`instr_ok=True` (baseline dog 0.600, cat 0.000, lesion→0.000, quiet inert, recall reproducible). So this is a validated
negative, not an instrument artifact.

## WHY it fails — two root causes, both diagnosed (not hand-waved)
1. **Architectural (structure-blindness).** Assembly MEMBERSHIP is selected on a separate, discarded bridge
   (`_gap5_emergent_dg_selection_derisk._build_bridge`, ca3_density=0.05, mossy_weight≈3000). The READOUT bridge we drive
   is `EpisodicDapMemory`'s own `_build` (ca3_density=0.5, default mossy) — an INDEPENDENT RNG draw. So the readout
   bridge's dg->ca3 mossy wiring does not structurally target the cells the store was written into; a content-blind DG
   volley cannot concentrate on dog's assembly (the non-collapsing lesion proves it hits dog and cat alike). The store's
   ONLY structural signature on the readout bridge is its potentiated RECURRENCE — which the afferent locus does not reach.
2. **Mechanistic (wrong brain state / missing companion process).** When completion did occur it was arbitrary-phase-
   dependent (the sweep hit vs the identical-pattern 1/6 miss). Real offline reactivation is a SHARP-WAVE-RIPPLE
   phenomenon — ripple-trough-timed, ~2× compressed, riding spike TIMING — a different interneuron regime than the
   arbitrary-phase drive here. This is the CLAUDE.md wall-reframe exactly: **the process the real system runs alongside
   the driver, that we replaced with a constant, is the SWR oscillatory state.**

## This re-confirms an existing arc — do NOT re-derive it (the stale-pointer catch)
<!--derived-->
A RAG check after the run surfaced that root-cause-2 was ALREADY established four weeks earlier:
[[2026-07-24-gap5-SWR-state-readout-research-gate-the-missing-biology]] — "the missing biology is the SHARP-WAVE-RIPPLE
(SWR) state … point-neuron-realizable" (Buzsáki 2006 SWR section; Ecker 2022 eLife e71850: a spiking CA3 of point-ish
neurons with structured recurrence AUTONOMOUSLY generates SWRs + replays from a NON-SPECIFIC drive). SWR ignition = a
TRANSIENT E>I imbalance in the CA3 recurrent circuit (~100-200 ms envelope), self-terminating. The recovered
`episodic_completion` probe corroborates from yet another organ: its "completion load-bearing (full vs zero-recurrent)"
control tied (0.625==0.625) — the recurrent completion is NOT load-bearing under a feedforward cue, only a state-driven
recurrent transient recruits it. **Three independent loci now agree: a better/targeted/theta DRIVER is the wrong lever;
the SWR-state E>I transient is the mechanism.**

## The next mechanism (already built, stuck in a tuning band — NOT a fresh derivation)
<!--derived-->
`research/runners/_gap5_swr_envelope_replay_derisk.py` (built 2026-07-24, NO finding yet) implements exactly this — rest
in the SWR down-state + weak non-specific noise + a transient E>I envelope — and lands `SWR-ENVELOPE PARTIAL/NEGATIVE
0/1` (`research/findings/raw/gap5_r4/swr_envelope_seed42.json`): forward_frac 0.000 vs reverse 0.333 vs chance 0.167,
per_asm_active~[3,3,3] (over-driven co-fire), all anti-cheats 0.000. Its own verdict names the residual as the
**envelope depth × duration × noise-σ TUNING band** (over-drive→[3,3,3], under-drive→[0,0,0]) with the knobs
`env_exc_pa / env_basket_drop / swr_period / env_dur / noise_pa / self_regen_read / --sel-inhib-spare`. So the next step
is NOT another afferent build — it is a **tuning sweep of that existing runner** (mechanical → the mini-PC pool, 0
tokens; n_ca3=1000 numpy locates the op-point band, confirm the winner at n_ca3=2000 on GPU). A parallel structural fix
for root-cause-1 (make the readout bridge SHARE the selection bridge's dg->ca3 wiring, or potentiate the encoding mossy
synapses — biologically: the same fibers that select an assembly should detonate it) would make a content-blind afferent
volley structure-aware — but the SWR-envelope tuning is the cheaper, already-built path. Not wired live; the D5
learn-through-use transfer stays blocked on the SWR-state op-point. (Agent-built + launched; parent verified the
runner's OWN verdict from the artifact, then RAG-checked and connected it to the existing SWR arc rather than banking a
rediscovery.)
