---
type: finding
status: smoke
date: 2026-08-08
mechanism: episodic-cortical-cue-recall
lane: EPISODIC
---

# Episodic cortical readout IGNITES (fan-in fix) — Wave-1 silence SURPASSED; recall is heteroassociative, NOT completion (SMOKE, seed 42)

Status: SMOKE (single seed 42; a PARTIAL surpass + teeth-backed honest negatives). Needs the 6-seed run below
before any generalization. Backend: numpy (CPU) — CORRECTED 2026-08-08: the committed provenance sidecar records
`sim_backend=numpy` (the smoke ran on numpy, not CuPy; an earlier draft mislabeled it CuPy). `cfg.seed`-seeded substrate (build-twice threshold hash
IDENTICAL). Runner: `research/runners/_riii_ca3_cortical_episodic_wta_derisk.py`. Artifact:
`research/findings/raw/cortical_episodic_wta/_episodic_readout_ignition_SMOKE_s42.json`. NO `sim/` edit
(reuse-by-import; all region/pathway construction + config runner-side).

## The law applied: the Wave-1 NEGATIVE launched this search, and it moved the wall

Wave-1 (worktree `wf_4661aab6-071-3`) died with the cortical readout SILENT — max cortical rate 0.000 — and
the only "pass" was `np.argmax` over an all-zero rate vector (returns index 0 → 0.25 mechanically at k=4).
Re-running the as-shipped runner reproduces it EXACTLY: every condition = chance 0.25, `ca3_compl`=0.00,
`sep`=0.000. That is a host tiebreak, not a neural decision. **The readout had to IGNITE before any B claim
is admissible.**

## What moved the wall: readout FAN-IN (the ignition lever), isolated by an instrument sweep

The ignition bottleneck was localised by a ceiling/instrument sweep (reproducible: runner `--ceiling`, plus the
per-stage diagnostics):

- The learned per-synapse CA3→cortex weight caps at **~11.6** (max ~14.8) under the committed rate-window
  Hebbian — the soft bound `hebbian_max`=2000 is far off, so the cap is the co-activity product, not the bound.
- A cortical RS pyramidal needs an **effective weight ~60** to ignite (manual-weight ceiling: W=60 → readout
  0.15, W=150 → 0.30). The learned weight is ~5× too weak per synapse.
- The gap does **not** close by (a) higher learning rate — lr≥0.05 collapses CA3 (`ca3_compl` 1.00→0.14)
  before the readout fires; (b) within-item recurrent amplification — fixed or plastic, an attractor cannot
  bootstrap from a **zero** seed; (c) ACh/theta recall-mode depolarization — the bias needed to seed firing
  (~180 pA) is within noise of self-ignition (no-cue ≈ cued).
- It DOES close by **fan-in**: raising `ca3_cortex_density` 0.5→1.0 over a larger assembly SUMMATES the many
  weak-but-SPECIFIC learned weights over threshold, **without** touching CA3 stability (adding CA3→cortex
  synapses does not feed back into CA3). Isolated stage-B test: acc 0.83–1.00, no-cue silent 0.000.

The biological reframe that found it (per CLAUDE.md's wall question): the cortical target is not a passive
readout gained by a bigger per-synapse weight — reinstatement rides on **convergent afferent fan-in** onto the
cortical cell, which is what a larger, denser CA3→cortex projection supplies.

## Result (SMOKE, seed 42, n_ca3=1500, k=4, assembly 0.10, ca3_cortex_density 1.0, partial-CA3 cue 0.5)

<!--derived-->
| condition | winner_overall | max_cortex_rate | ca3_compl | reads as |
|---|---|---|---|---|
| full | 0.75 | 0.042 | 0.00 | readout IGNITES, correct WHAT/WHEN wins (chance 0.25) |
| permute_cue | 0.00 | 0.042 | 0.00 | wrong-assembly cue → wrong readout (specificity) |
| lesion_real | 0.25 | 0.000 | 0.00 | ablated cue-assembly → silent readout, chance |
| lesion_sham | 0.75 | 0.042 | 0.00 | unrelated ablation → recall PRESERVED |
| untrained | 0.25 | 0.000 | 0.00 | no engram → silent readout, chance |
| zero_recurrent | 0.75 | 0.042 | 0.00 | ca3→ca3 zeroed → recall UNCHANGED (see negative #1) |
| wta_off | 0.75 | 0.042 | 0.00 | lateral inhibition off → unchanged (see negative #2) |

(Values from the artifact above; chance=0.25.)

## What is EARNED (the surpass)

The cortical readout **ignites** (max cortical rate 0.042 > 0, vs Wave-1's 0.000) and selects the correct <!--derived-->
WHAT/WHEN attribute above chance (0.75 vs 0.25) by a **neural** heteroassociative CA3→cortex map, with a teeth
panel that PASSES: permuted-cue → chance + wrong readout; real-lesion → silent + chance; sham-lesion →
preserved; untrained → silent + chance. This kills the "silent readout + argmax-over-zeros" death and corrects
the Wave-1 anti-cheat (1) mislabel (a host argmax dressed as a lateral-inhibition WTA).

## Honest NEGATIVES (mapped with teeth — the deliverable, not a caveat)

1. **CA3 pattern COMPLETION is NOT load-bearing.** `ca3_compl`=0.00 in every condition and the zero-recurrent
   control is INERT (0.75 == full). The recall is FEEDFORWARD heteroassociation from the CUED CA3 cells, not
   recurrent attractor completion. Per `docs/TERMS.md` the term **"completion" is NOT earned** here; this is
   cue-recall via a learned CA3→cortex map. An 8-cell operating-point sweep (assembly 0.04–0.06 × recall_k
   18–28 × cue-frac 0.5–0.6) found NO point where completion becomes load-bearing without silencing the
   readout — the requirements oppose (below).
2. **The neural WTA lateral inhibition is NOT load-bearing** (`wta_off` == full). At this sparse operating
   point the selector is heteroassociative specificity, not lateral inhibition.
3. **The cortical feedforward cue cannot ignite CA3** (`ca3_compl`=0.00 from a cortex_who cue): the admissible
   cue is a PARTIAL CA3 cue delivered INTO CA3 (as EC/mossy delivers it). A cortex→CA3 detonator weight up to
   80 did not trigger the CA3 dendritic plateau.
4. **Opposing assembly-size requirements.** Readout ignition needs a LARGE assembly (fan-in); clean completion
   needs a SMALL sparse assembly. A large assembly lets the cued half feedforward-drive the readout (completion
   redundant); a small assembly starves fan-in → silent readout. Resolving BOTH needs a `sim/` change — a
   cortical-stage bistable dendritic amplifier (transplant `fused_coincidence_plateau` to the cortical pools so
   a weak specific CA3 seed triggers a plateau-sustained full item), or a target-specific readout plasticity
   gain (so CA3→cortex potentiates to ~60 while the cortex→CA3 encoder stays weak, preserving CA3 stability).

## Levers spent on the one defect (readout ignition), for the gate

per-synapse weight · learning-rate (0.005–0.5) · within-item recurrence (plastic + fixed 8–60) · ACh <!--derived-->
recall-mode bias (60–180) · cortex→CA3 detonator (40–80) · **fan-in (density×assembly) = the one that worked**.
Six levers; fan-in resolved it. Completion remains open and is routed to the two `sim/`-level surpasses above.

## 6-seed command (parent dispatches survivors to the pool)

```
python -m research.runners._riii_ca3_cortical_episodic_wta_derisk --smoke \
  --seeds 42,43,44,100,101,102 --n-ca3 1500 --k-items 4 --train-events 30 \
  --assembly-frac 0.10 --ca3-cortex-density 1.0 --ca3-cue-frac 0.5 \
  --out research/findings/raw/cortical_episodic_wta/<6seed>.json   # placeholder — run mints this artifact
```

Verify readout ignition first: `--verify-seed` (build-twice threshold hash) then `--ceiling` (architecture
conducts). GO on the readout-ignition capability requires, per seed: max_cortex_rate > 0 AND winner_overall >
chance AND permute/lesion_real/untrained at chance AND sham preserved. The completion + WTA claims stay
NEGATIVE until a `sim/`-level cortical bistable amplifier is built.
