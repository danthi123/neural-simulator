---
type: finding
status: negative
date: 2026-08-09
mechanism: spiking-pattern-completing-attractor-engram-store-for-sleep-replay-consolidation
lane: breadth / teacher-loop / memory
runner: research/runners/_teacher_loop_spiking_engram_consolidation_derisk.py
builds-on: research/runners/_teacher_loop_sleep_replay_consolidation_derisk.py
attacks-baseline: teacher-loop SLEEP-REPLAY consolidation with a HOST MEAN-VECTOR engram store (6-seed replay frac_recalled mean 0.55, range 0.20-0.90; finding 2026-08-08-teacher-loop-sleep-replay-consolidation-self-replay-beats-catastrophic-forgetting.md)
biological-pattern: Hopfield/CA3 sparse autoassociative pattern completion (Marr 1971; Hopfield 1982; Tsodyks-Feigelman 1988 sparse-covariance rule; de Almeida-Idiart-Lisman 2009 E%-max feedback inhibition); hippocampal->cortical systems consolidation (McClelland/McNaughton/O'Reilly 1995)
artifacts:
  - research/findings/raw/teacher_loop_spiking_engram_s42.json
  - research/findings/raw/teacher_loop_spiking_engram_s43.json
  - research/findings/raw/teacher_loop_spiking_engram_s44.json
  - research/findings/raw/teacher_loop_spiking_engram_s45.json
  - research/findings/raw/teacher_loop_spiking_engram_s46.json
  - research/findings/raw/teacher_loop_spiking_engram_s47.json
  - research/findings/raw/teacher_loop_spiking_engram_6seed_agg.json
---

# Spiking pattern-completing engram store MATCHES (does not beat) the host mean-vector for sleep-replay consolidation

## Claim (honest NEGATIVE = NO-GO, with a fully-passing mechanism panel)

<!--derived-->

Replacing the sleep-replay loop's **host mean-vector engram store** (`Hippocampus.encode` = `X.mean(axis=0)`, a
Python list) with a genuinely **spiking, pattern-completing attractor** (a minimal Hopfield/CA3-style store: sparse
binary assemblies, sparse-covariance recurrent synapses, E%-max feedback inhibition, a Hebbian assembly->feature
readout) **reproduces the baseline retention but does NOT raise it toward the 0.8 ceiling**: 6-seed replay
frac_recalled@N=10 **spk 0.533 vs host 0.517** (like-for-like, same net/seed/epochs) — a **+0.017 tie** (per-seed
signs mixed: 0/0/0/-0.10/+0.30/-0.10; the attribution tool reads 96.9% of the total as present in the control),
**below the 0.10 real-effect bar -> NO-GO** on the headline. The reconstruction is **perfect (fidelity cosine =
1.000 every seed)** — which is exactly why it cannot win: in this **unimodal** teacher-world (one perceptual
prototype per referent) the host mean is already a **sufficient statistic** of the engram, so a higher-mechanism
store has no fidelity headroom to recover.

**The hypothesis under test — "a higher-fidelity neural engram raises replay retention" — is FALSIFIED for this
world.** Store fidelity is **NOT** the 0.55 consolidation bottleneck. The mechanism panel (b/c/d) all PASS — the
store is real, neural, load-bearing and self-generated — so this is an EARNED negative (a live instrument that
detected no headline effect), not an instrument failure.

<!--derived-->

## Numbers (6 seeds 42-47, N=10, chance 0.10; frac_recalled@N=10)

| seed | host_noreplay | host_replay (the 0.55 baseline) | spk_replay | spk_scramble | spk_lesion |
|-----:|--------------:|--------------------------------:|-----------:|-------------:|-----------:|
| 42 | 0.10 | 0.90 | 0.90 | 0.10 | 0.50 |
| 43 | 0.10 | 0.20 | 0.20 | 0.20 | 0.10 |
| 44 | 0.10 | 0.40 | 0.40 | 0.20 | 0.30 |
| 45 | 0.10 | 0.20 | 0.10 | 0.10 | 0.20 |
| 46 | 0.10 | 0.60 | 0.90 | 0.00 | 0.30 |
| 47 | 0.10 | 0.80 | 0.70 | 0.10 | 0.50 |
| **mean** | **0.10** | **0.517** | **0.533** | **0.117** | **0.317** | <!--derived-->

<!--derived-->
Reconstruction fidelity (cosine to true engram): **replay 1.000, lesion 0.973**. Immediate acquisition
(spk_replay): **0.993**. host_replay reproduces the declared baseline **mean 0.517, range 0.20-0.90** (same as the
2026-08-08 store, seed-for-seed) — the like-for-like harness is validated.

## Teeth verdict (NO-GO on the headline; b/c/d instrument-validity all PASS)

<!--derived-->

- **(a) retention rise vs the host store — NOT MET (the honest negative):** spk 0.533 vs host 0.517 = +0.017,
  below the 0.10 real-effect bar. NO-GO.
- **(b) attractor completion load-bearing — MET:** spk_replay 0.533 vs spk_lesion 0.317 (+0.217), and fidelity
  1.000 -> 0.973 when the recurrents are lesioned. The neural engram + its completion genuinely carry retention
  WITHIN the spiking store (lesioning the recurrents lets the degraded/contaminated cue through -> both fidelity
  and retention drop). The engram is neural and load-bearing, not an inert wrapper.
- **(c) self-generated — MET:** spk_scramble 0.117 ~= host_noreplay 0.10 (content-lesioned store forgets like the
  wall; the retention comes from the STORED content, not the extra compute or the teacher).
- **(d) immediate acquisition stays perfect — MET:** 0.993.

## Why (the mechanism, and what consolidation actually needs)

The baseline finding called the mean-vector "lossy". It is lossy only in the sense of being a **single prototype**
— but for a **unimodal Gaussian** referent it is the **maximum-likelihood, sufficient statistic**: nothing a
higher-fidelity store recovers changes the replay draws' distribution. The spiking attractor completes to exactly
that mean (fidelity 1.000), so it TIES. This is **structural to the world, not a tuning failure** — the mechanism
is engaged (fidelity is perfect, the completion runs, the lesion/scramble teeth move as designed) and inert on the
host-vs-spk retention comparison.

**The 0.55 under-consolidation is therefore a REPLAY-BUDGET / SHARED-READOUT-CAPACITY wall, not a store-fidelity
wall.** The mapping this de-risk delivers: consolidation here needs (i) more replay interleaving against the
single shared leaky-readout, or (ii) a genuinely **lossy** store to exercise — i.e. a **multimodal** category world
where the mean averages across modes and a multi-assembly attractor would preserve them. The spiking
pattern-completing store built here is the right substrate for (ii); this world does not exercise it.

## Reproduce

Single-seed SMOKE:
```
SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_spiking_engram_consolidation_derisk --seeds 42 \
    --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
    --out research/findings/raw/teacher_loop_spiking_engram_s42.json
```
6-SEED (the decisive like-for-like; ~20 min, run one process per seed in parallel):
```
for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_spiking_engram_consolidation_derisk --seeds $s \
    --milestones 1 5 10 --n-max 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
    --out research/findings/raw/teacher_loop_spiking_engram_s$s.json & done; wait
```

## Scope / honesty

- **Genuinely spiking, minimal:** the store is a binary-threshold (McCulloch-Pitts) recurrent attractor with
  E%-max (k-active) feedback inhibition — the completion is done by neurons + recurrent synapses, NOT a host mean.
  It is **NOT** the Izhikevich `SimulationBridge` substrate; it is the "minimal Hopfield/CA3-style spiking
  attractor" the spec permits. The class label bound to each assembly is bookkeeping in BOTH stores (like-for-like).
- **No host mean-vector on the replay path** (grep-verified: `generate_replay` / `_complete` / `_cue_for` /
  `_readout` contain no `.mean(axis=0)` and no `env`). The true engram mean is computed MEASUREMENT-ONLY (fidelity
  reference), never fed to e-prop.
- NO sim/ edit (reuse-by-import of the sleep-replay + scaling machinery). cfg.seed seeds the substrate.
