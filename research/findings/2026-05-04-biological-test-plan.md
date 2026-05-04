# Biological-scale test plan — what to run after reboot

**Date:** 2026-05-04
**Context:** B1 sanity check (2026-05-04 ~10:00 EDT) found that hand-built
perfect language→motor weights at N=25 motor + no recurrence give 25%
TRUE accuracy (chance). The minimal architecture intentionally strips
the cortical canon (recurrence, E/I balance, NMDA) along with the
cascade. That may have been too much.

This plan tests the hypothesis: **"the eval was broken because the
architecture was too minimal, not because the eval logic itself is
wrong."**

## Hardware budget

- RTX 3090 / 24 GB VRAM (target ~18-20 GB usable peak)
- 48 GB RAM
- i7-12700 (12 cores, 8P+4E)
- Single-process per run (parallel=1) for max scale per single run

## Test 1: bio_sanity_check (~3 hours)

Same 4 modes as the broken-eval B1, now at biological scale.

**Architecture (`build_biological_brain_regions`):**
- `language_input`: 2048 neurons (4× current Wernicke), `internal_density=0.05`,
  `exc_fraction=0.8`
- `motor_X` per action: 500 neurons (Schieber 2001 / Rathelot 2009 motor
  sub-pool), `internal_density=0.10` (Lefort 2009 cortical recurrence),
  `exc_fraction=0.8` (real cortex 80E/20I), `exc_weight=2.0`, `inh_weight=4.0`
- NMDA enabled (`cfg.enable_nmda=True`, Wang 2002 bistability)
- Total: ~4048 neurons, ~1.5M synapses, ~1-2 GB GPU peak

**Conditions (4 × 6 seeds = 24 runs, parallel=1):**
- `bio_density030` — perfect weights, density 0.30 (matches biology sweep)
- `bio_density100` — perfect weights, density 1.0 (full connectivity)
- `bio_density030_wrong` — rotated weights (control)
- `bio_density030_random` — random U[0, 8.0] (control)

**Expected outcomes:**
- **perfect mode aligned ≥ 4/6:** eval works at bio scale → proceed to
  Test 2 (full STDP training at bio scale).
- **perfect mode aligned 0-1/6:** even biological canon doesn't unlock
  the eval. Deeper investigation required: drive currents, measurement
  window, baseline subtraction in `evaluate_word_to_action`.

## Test 2: bio_proof_of_concept (~2.5 hours, CONDITIONAL on Test 1 passing)

Single-seed proof-of-concept of full STDP training at biological scale.

**Conditions (2 × 1 seed = 2 runs, parallel=1):**
- `bio_baseline` — cortical canon only, no biology fix. Tests
  "does cortical canon ALONE enable W→A learning?"
- `bio_topo_fs` — cortical canon + Pulvermüller topographic prior +
  Vogels motor PV-FSI. Tests "is biology fix additive on canon?"

**Expected outcomes:**
- **Both conditions aligned at TRUE labels (4/4 trials):** dramatic
  improvement vs minimal scale. Cortical canon was the missing piece.
  Multi-seed validation becomes the next experiment.
- **Both 0-1/4:** plasticity rule itself is the bottleneck. Next
  experiment is B3 (supervised gradient).
- **bio_baseline 0/4 but bio_topo_fs ≥ 3/4:** biology fix matters at
  bio scale (it didn't at minimal scale because there was no canon for
  it to amplify). 

## Decision tree

```
bio_sanity_check
    │
    ├── perfect ≥ 4/6 ──► bio_proof_of_concept ──► [multi-seed validation]
    │                          │
    │                          ├── ≥ 3/4: cortical canon enables W→A → SHIPPED
    │                          ├── ≥ 3/4 only with topo_fs: biology fix is needed
    │                          └── 0/4 both: plasticity bottleneck → B3 next
    │
    └── perfect 0-1/6 ──► investigate evaluate_word_to_action directly:
                          drive currents, stim window, baseline subtraction.
                          Possibly instrument bridge to dump motor pool
                          firing rates per word.
```

## Memory budget verification

Estimated GPU usage at bio scale:
- CUDA context + CuPy memory pool: ~500-700 MB
- 1.5M synapses × ~50 B/synapse: ~75 MB
- 4048 neurons × ~500 B/neuron: ~2 MB
- STDP arrays per plastic synapse: ~32 MB
- Eligibility traces, refractory timers, etc: negligible
- **Total estimated: ~700-800 MB peak**

Plenty of headroom in 24 GB. No risk of OOM. The 1.3 GB figure quoted
in CLAUDE.md for the 5K-neuron flagship is dominated by CUDA context,
not data; this scale is comparable.

## How to run (after reboot)

```powershell
# In E:\Documents\Projects\sim
powershell -ExecutionPolicy Bypass -File scripts/launch_bio_chain.ps1
```

This:
1. Cleans stale python processes
2. Launches webapp on port 8765 (clean uvicorn)
3. Launches `bio_sanity_check` (24 runs, parallel=1, ~3 hours)
4. Waits for completion, aggregates results
5. If perfect mode aligns ≥ 4/6: launches `bio_proof_of_concept`
   (2 runs, parallel=1, ~2.5 hours)
6. Aggregates final results

Total runtime: ~3 hours (sanity only) or ~5.5 hours (full chain).

## How to monitor

- Live: `http://127.0.0.1:8765` (webapp dashboard)
- CLI: `python -m research.runners.morning_briefing`
- Chain log: `research/findings/raw/g11_bg/launch_bio_chain.log`
- Per-experiment: `research/findings/raw/g11_bg/{bio_sanity_check,bio_proof_of_concept}.{stdout,stderr}.log`

## How to abort

```powershell
# Find the launcher orchestrator
Get-Process powershell | Where-Object { $_.MainWindowTitle -like '*bio*' }
# Stop everything
Get-Process python | Stop-Process -Force
```

## Why this is the right next step

Three reasons:

1. **The minimum-arch B1 result (2026-05-04 ~10:00 EDT) was suggestive
   of eval methodology issues, but the architecture was unprecedented**
   (no recurrence + N=25 + no E/I + no NMDA). Real motor cortex doesn't
   look anything like that. Testing the eval on a biologically-faithful
   architecture is more honest.

2. **All prior W→A evals in this project ran on architectures that ALSO
   lacked recurrence + NMDA on the motor pools.** Even the v2 flagship
   used motor pools without these features. So the broken-eval finding
   may explain not just the minimal-iso result but the entire 0/N
   alignment streak across biology + cascade + arch sweeps.

3. **Memory permits it.** RTX 3090's 24 GB is way more than the bio
   scale needs (~1 GB peak). No reason to stay at minimal scale.

## Skips and trade-offs

- **No multi-seed validation in Test 2 yet** — single-seed
  proof-of-concept first because each run is ~70 min at bio scale.
  If results look promising, follow with 6-seed validation (~7 hours).
- **OU noise stays at default `tau_ms=15.0`** — could be slower (50-100ms
  for biological cortical noise) but conservative for first test.
- **Token sparsity unchanged at 0.1** — overlap is the same; if Test 1
  passes but Test 2 fails, sparsity 0.05 (~2-3 word overlap vs ~6-9)
  is the next variable.
