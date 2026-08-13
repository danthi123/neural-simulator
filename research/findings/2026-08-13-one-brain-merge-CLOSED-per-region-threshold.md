---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# One-brain 2-organ MERGE: per-region threshold seeding CLOSES the INIT-invariance byte-identity (0/6 → 6/6); the production trained-read residual is a NEWLY-isolated homeostatic companion process (not the init RNG)

**Date:** 2026-08-13 · **Runner:** `research/runners/_one_brain_merge_2organ_derisk.py` · **Artifact:**
`research/findings/raw/_one_brain_merge_2organ_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`).
**`sim/` EDIT (owner green-lit): additive + guarded + byte-identical-when-off** — a new opt-in config flag
`cfg.per_region_threshold_heterogeneity` (default `False`). Supersedes the BOUNDARY axis of
`2026-08-13-one-brain-merge-2organ-BOUNDARY.md` (structural merge was already GO 6/6 there).

## What shipped (the `sim/` fix)

`sim/config.py`: `per_region_threshold_heterogeneity: bool = False`. `sim/bridge.py`: after the existing global
`cp_neuron_firing_thresholds` draw (the `cp.random.uniform(size=n)` at what was `:2307`), a NEW block —
**skipped entirely when the flag is off** — OVERWRITES each brain region's threshold slice with a draw from a
REGION-SCOPED RNG (`_overwrite_region_scoped_thresholds`), keyed on a STABLE `zlib.crc32` hash of the region name
(process-independent, unlike the salted builtin `hash`), each substream reset to position 0. This mirrors the
engine's existing `per_parameter_heterogeneity_seed` / `_HET_PER_PARAM_SEED_STRIDE` machinery. Because the legacy
`size=n` draw still runs first, global-RNG consumption is preserved bit-for-bit (any later global draw is
unperturbed) and any neuron not owned by a region keeps its legacy value.

**Why default-off is byte-identical (verified in data, not inferred):** a default Izhikevich bridge built at a
fixed seed with the flag absent-vs-present-but-off hashes IDENTICALLY on `cp_neuron_firing_thresholds`,
`cp_membrane_potential_v`, `cp_izh_a`, `cp_connections.data` (git-stash A/B of `sim/` only). `tests/test_determinism.py`
passes 9/9 incl. `TestSubstrateActuallySeeded`; the production `brain_chat_tui --smoke` runs unchanged (the flag
is never set, so the block is unreached). Determinism WITH the flag ON is 6/6 (build-twice-same-seed hashes of
`cp_neuron_firing_thresholds` are identical — now part of the runner's determinism check).

## Result — the INIT-invariance axis CLOSES; the structural merge stays GO

| criterion | fix ON (6 seeds) | verdict |
|---|---|---|
| ONE shared spiking pool (both organs in one `cp_` array) | 6/6 | **GO** |
| determinism (`cfg.seed`; build-twice incl. thresholds byte-identical) | 6/6 | **GO** |
| CROSS-ORGAN synapse LOAD-BEARING (intact vs lesion) | 6/6 | **GO** |
| **INIT byte-identity — every per-neuron array of each organ, merged-vs-standalone, BEFORE training** | **6/6 (max err 0.0)** | **GO (CLOSED)** |
| byte-identical, HOMEOSTASIS-OFF (static per-region-heterogeneous thresholds) | 6/6 | **GO** |
| byte-identical, PRODUCTION (homeostasis ON — the fully adapted trained read) | 0/6 | **BOUNDARY** |
| **STRUCTURAL MERGE (pool + determinism + load-bearing + INIT byte-id + homeo-off byte-clean)** | **6/6** | **GO** |

- **INIT byte-identity is EXACT 0.0 for BOTH organs, all 12 per-neuron arrays** (thresholds + the 9 Izhikevich
  params + v + u), 6/6. Under `--legacy-global-thresh` the same gate reads `init_maxerr ≈ 24.6` (the threshold
  divergence) and STRUCTURAL falls to BOUNDARY — the fix is the discriminator. This is the mission's literal
  ask ("a merged organ's init is invariant to its co-residents, byte-identity 0/6 → 6/6"): **CLOSED.**
- **The cross-organ synapse stays DECISIVELY load-bearing** (6/6): organ-B recall rises `+20.2…+33.6 Hz`
  intact when organ A is surprised vs `|lesion| ≤ 0.99 Hz` after zeroing the `surprise_A→cue_B` edges. The
  surprise faculty stays FUNCTIONAL on the merged bridge (contradict/confirm separation `7.2×…49.9×`).

## The honest boundary — a SECOND cause the BOUNDARY finding did not isolate (homeostasis)

The BOUNDARY finding root-caused the residual to ONE line (the init threshold RNG). That was **INCOMPLETE.**
The init RNG is one cause and it is now CLOSED. The production trained-read residual (recall err ≤ 1.63 Hz,
surprise err ≤ 3.59 Hz, 0/6) is a DIFFERENT, load-bearing mechanism: **homeostatic threshold adaptation**
(intrinsic plasticity; `enable_homeostasis=True` by config default — the runner's own docstring wrongly assumed
it OFF). Homeostasis REASSIGNS `cp_neuron_firing_thresholds` every step (`bridge.py:~10525`), so each organ's
operating point is coupled to the WHOLE pool's activity history: under the toy's sequential protocol (train organ
A, then organ B) each organ **idle-drifts** while the other trains. Decisive controls (in the artifact):

- **Homeostasis OFF** (static, still per-region-HETEROGENEOUS thresholds) → the full trained+read pipeline is
  byte-EXACT 6/6. With INIT already exact, this isolates the production residual to the homeostatic DYNAMICS.
- Homeostasis is **load-bearing**: OFF collapses the surprise separation to 1.0× (it SETS the operating point).
  So it cannot simply be disabled — this is a genuine companion process, not a bug.
- **In-session diagnostic** (per-organ homeostatic-state isolation, cross inert during training): organ A goes
  EXACT 0.0 and organ B collapses to a sub-0.1 Hz near-threshold floor <!--derived--> — a single near-threshold
  fact flipping by ~1 spike, a shared-numerical-context (one sparse matrix, one step) floating-point effect, NOT
  the init RNG. This is the deepest residual and is irreducible without deterministic-summation arithmetic.

**This is a textbook instance of the project's companion-process doctrine** ("what else does the real system run
alongside this, that we replaced with a constant?"): the init RNG was the visible cause; homeostasis — the
process that sets the operating point — owned the rest, and only became measurable once the init cause was removed.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._one_brain_merge_2organ_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_one_brain_merge_2organ_6seed.json
# BOUNDARY reproduction (legacy single global stream -> INIT byte-id fails, STRUCTURAL BOUNDARY):
SIM_BACKEND=numpy python -m research.runners._one_brain_merge_2organ_derisk --seed 42 --legacy-global-thresh
```

## Honest scope / non-claims

- **CLOSED = the INIT-invariance axis + the init-RNG cause** (6/6 byte-exact) and the STRUCTURAL merge (6/6 GO).
  **NOT claimed:** exact byte-identity of the fully homeostatically-adapted PRODUCTION read — that is bounded by
  the homeostatic companion process (+ a shared-numerical-context FP floor), honestly mapped above, not the init.
- **Not integrated into production.** This is a substrate-init correctness fix behind an opt-in flag; no
  production organ set is merged onto one pool by default. "One substrate" for the production organs remains
  CO-RESIDENCY.
- **Two INSTANCES of one builder** (surprise + recall). Merging DIFFERENT builders (e.g. + the Wong-Wang
  comprehension monitor) needs a config SUPERSET (GABA_B + NMDA-accumulator coexistence) and 2→N scaling — the
  named next step, not attempted here.
- **Functional read-outs only**; no phenomenal claim.
