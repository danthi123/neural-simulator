---
status: live
type: finding
lane: integration
date: 2026-08-27
---

# Per-neuron-seeded OU stream (sim/ edit) closes `curiosity`'s organ-read → 7/7 Group-A organ-read

Status: GO. Two coupled results. (1) A guarded, DEFAULT-OFF `sim/bridge.py` edit adds a PER-NEURON-seeded
Ornstein-Uhlenbeck noise stream: with the flag OFF the OU draw is BYTE-IDENTICAL to today; with it ON each neuron's
OU background drive is keyed on a STABLE per-neuron id (region-name crc32 × within-region rank), so it is
co-residence-INVARIANT (independent of the neuron's absolute pool index / how many neurons precede it). This burns
down the "OU-off" shortcut the one-brain merge framework had to take (its organ-read migration turned global
OU/conductance noise OFF because the index-order OU RNG is co-residence-DEPENDENT). (2) With that stream, `curiosity`
— the ONE Group-A organ whose read was still deferred — closes its ORGAN-READ byte-identity, 6-seed GO, co-resident
with `comprehension` on ONE `SimulationBridge`. No other Group-A organ remains organ-read-deferred.

Artifact (carries a `tools.verdict.Verdict` `preconditions` block):
`research/findings/raw/_onebrain_merge_curiosity_organread_6seed.json`.

## The sim/ edit (additive, DEFAULT-OFF, byte-identical-when-off, guarded)

- `sim/config.py:290` — `per_neuron_ou_seed: bool = False` (a new opt-in flag; the sibling of `per_region_ou_seed`,
  at neuron granularity). Off by default.
- `sim/bridge.py:183` — module helper `_splitmix64(x)`: a vectorized, stateless SplitMix64 finalizer over host
  uint64 (wraps mod 2**64). Used ONLY by the per-neuron path.
- `sim/bridge.py:3869` — in `_initialize_ou_process_state`, when `cfg.per_neuron_ou_seed` is ON and a region_manager
  exists, build per-neuron keys via `_build_per_neuron_ou_keys` (`sim/bridge.py:3873`): for each region, each
  neuron's key = `_splitmix64(base_seed ^ (crc32(region_name)×stride) ^ (within_region_rank×odd))`. Keyed on
  `cfg.ou_seed` (else `cfg.seed`). The within-region RANK is the co-residence-invariant per-neuron id (a region is a
  contiguous block whose j-th logical member is always at sorted rank j, alone or offset in a merged pool — the same
  validity basis the per_region seams rest on).
- `sim/bridge.py:3917` — `_per_neuron_ou_gaussians(keys, step)`: per-neuron N(0,1) via two SplitMix64 hashes of
  `(key, step_counter)` + a Box-Muller transform, in host NumPy. Stateless/counter-based ⇒ a neuron's per-step draw
  is a pure function of (its stable key, the step counter), hence co-residence-invariant.
- `sim/bridge.py:3999` — in `_draw_ou_noise_samples`, the legacy global `cp.random.randn(n)` draw still runs FIRST
  (global-RNG consumption preserved bit-for-bit; any neuron NOT owned by a region keeps its legacy value), then each
  region-owned neuron's slot is OVERWRITTEN from its own hashed stream. The step counter advances once per draw call.

When OFF (default) the per-neuron branch is never entered ⇒ byte-identical to today. It does NOT reshape any core
array. The deterministic-matvec path was NOT touched (a separate, higher-risk edit).

## The three determinism / byte-identity proofs (all measured, not asserted; tiny numpy region bridges)

Reference legacy behavior (pristine code, flag absent): a 2-region net (cue=30, ask=20), OU on, 12 steps.

1. DEFAULT-OFF byte-identity — flag OFF, my code vs pristine: `sha256(cp_ou_current)=3c80c12d6edae109` and
   `sha256(cp_membrane_potential_v)=b400d74b8b7796ca` — EXACT match to the pristine hashes. (Pristine legacy is itself
   co-residence-DEPENDENT: the "ask" region's OU trajectory differs by 182.95 pA alone-vs-coresident — the bug the
   seam fixes.)
2. Determinism (the cfg.seed trap) — flag ON, build TWICE at seed 42: `cp_ou_current` hash identical
   (`c0690216484b58f1` both builds). And seed 42 ≠ seed 43 (`4d463d690e828bb9`) ⇒ `cfg.seed` genuinely controls the
   per-neuron OU (not a re-draw from a global stream).
3. Co-residence invariance — the "ask" region ALONE (indices 0..19) vs co-resident BEHIND a "cue" region (indices
   30..49), flag ON: `cp_ou_current[ask]` trajectory over 12 steps is IDENTICAL, `coresidence_maxerr = 0.0`. Legacy
   (flag OFF) gives `182.95`. (per_region_ou_seed, the sibling seam, also gives 0.0 — a cross-check.)

Regression: `tests/test_determinism.py` 9 passed / 2 skipped; `tests/test_from_novelty_curiosity.py` 6 passed.

## Curiosity's organ-read closure (6-seed = 42,43,44,100,101,102)

Curiosity's ASK pool has NO afferent pathway — it fires ONLY via the `from_novelty` → excitability_drive `curiosity`
neuromodulator plus its OU background drive. Its read is a FROZEN forward pass of the spiking ASK-pool WANT (Hz) at a
NOVEL (0.95) vs FAMILIAR (0.0) epistemic gap, averaged over 4 drift-free reps. Conductance noise does NOT touch the
Izhikevich ASK pool (it applies only to HH g_Na/g_K, `sim/bridge.py:9488`), so OU was its ONE co-residence-dependent
input — now closed.

The read organ (`_CuriosityReadOrgan`, `research/runners/onebrain_merge_framework.py`) uses the LOCAL-INIT pattern
(the source_provenance / causal_whatif toggle-then-restore seam, adapted to OU + neuromod): the pool config keeps
`enable_ou_process` and the neuromodulator subsystem OFF (so it unions cleanly with the OU-off frozen organs), and
the read BUILDS both locally on the pool slice — a per-neuron OU stream via `_initialize_ou_process_state`, and the
curiosity `NeuromodulatorManager` — inside `sequence_isolation` (which restores the full per-neuron/per-synapse state
+ the RNG cursor on exit), then TEARS THEM DOWN. So curiosity's OU + neuromod needs are confined to its own read and
every co-resident slice stays byte-identical.

Result (`--keys comprehension,curiosity`, curiosity's ASK pool genuinely index-OFFSET — 1322 co-resident vs 290
alone): read_byte_identical 6/6 (read_maxerr = 0.0 every seed), answer_same 6/6 (the honest follow-up question is
preserved), substrate-byte-identical 6/6, param-het reconciliation load-bearing 6/6, legacy discriminator diverges
6/6, POOL ALL-GO. NON-DEGENERATE: want(novel) = 13.5–17.0 Hz vs want(familiar) = 0.17–2.08 Hz, margin 12.0–16.0 Hz
across all 6 seeds — the ASK drive genuinely tracks novelty, not a hollow all-equal read.

The legacy (per_neuron OFF) discriminator on the SAME local-init read diverges 6/6 (read_maxerr up to 6.42 Hz
merged-vs-coresident) — the per-neuron OU seam is LOAD-BEARING for the closure, not vacuous.

## The other organs do NOT regress

`--keys self_schema,d6_multiref_wm,comprehension,source_provenance,causal_whatif` 6-seed: all 5 previously-closed
organ-reads STILL GO 6/6 (read_byte 6/6 each), POOL ALL-GO. The sim/ edit is a no-op for them (they run
`enable_ou_process=False`, so `per_neuron_ou_seed` — default OFF, and unset in their configs — never fires), and the
added curiosity plumbing does not touch their code paths. `prospective_memory`'s multi-turn read is verified by its
own harness and is likewise untouched.

## Files changed

- `sim/config.py` — `per_neuron_ou_seed: bool = False` (additive flag).
- `sim/bridge.py` — `_splitmix64` module helper; `_build_per_neuron_ou_keys` + `_per_neuron_ou_gaussians` methods;
  the per-neuron init block in `_initialize_ou_process_state` and the overwrite branch in `_draw_ou_noise_samples`.
  All guarded by `cfg.per_neuron_ou_seed` (default OFF ⇒ byte-identical). No core array reshaped.
- `research/runners/_curiosity_seek_learn_onbridge_derisk.py` — additive `per_neuron_ou_seed=False` kwarg on
  `build_curiosity_bridge` (byte-identical when False).
- `research/runners/onebrain_merge_framework.py` — the `_CuriosityReadOrgan` read plumbing + the curiosity
  descriptor's organ_cls/read_fn/answer_fn/supports_shared; `GROUP_A_ORGANREAD_DEFERRED` emptied (no Group-A organ
  remains organ-read-deferred).

## Honest boundary — unchanged: still the MIGRATION gate, not INTEGRATION

Byte-identity-in-isolation forbids the cross-region interaction that IS the one-brain goal: a pool with zero
cross-synapses is MIGRATED, not INTEGRATED. The co-resident-alone baseline is the pool-built-alone organ on the
superset config, NOT curiosity's pre-migration standalone — the claim is co-residence-invariance of the read at the
shared pool operating point. Curiosity's declared scaffold residuals are unchanged: the novelty scalar is a
host-derived epistemic-gap (the abstain; a graded familiarity-gate novelty is the next rung), the wh-frame of the
follow-up is a fixed host language scaffold, and the learning-progress selector + noisy-TV veto are not wired (a
single-topic follow-up needs neither). This finding does NOT flip any production default and does not do the
production integration.
