---
status: live
type: finding
lane: integration
date: 2026-09-05
mechanism: da-write-gain-spiking-population-read
integration_faculty: da-gated-encoding
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
runner: research/runners/_da_write_gain_spiking_derisk.py
instrument: lesion control (excitability_drive sensitivity pinned to 0.0 at build time, a static config value)
  collapses the intact population's 1.35-1.52 gain span to 0.0000 span on every seed, plus a per-seed
  correlation against the pre-existing host formula (0.9958-0.9999) -- both reported per-seed below, not a
  ratio alone.
artifacts:
  - research/findings/raw/_da_write_gain_spiking/6seed.json
  - research/findings/raw/_da_write_gain_spiking/hook_verify.json
  - research/findings/raw/_da_encoding_wired/verify.json
external: NO-EXTERNAL-NEEDED -- reuses the ALREADY-established DA/neuromodulator machinery in-repo
  (`sim.neuromodulators`'s `excitability_drive` ModulatorTarget, the SAME target_type/scope idiom
  `_neuromod_spiking_da_mode_derisk` already validated 6/6-seed for DA->str_D1/D2; the Lisman-Grace
  hippocampal-VTA-loop / Kandel D.16 biology citation is reused verbatim from `da_encoding_drives_chat.py`'s
  own pre-existing docstring, not newly researched).
---

# DA -> write-magnitude, the remaining LEAF host linear map, retired by a spiking population read (rank-16, 6/6-seed GO, wired default-OFF)

**Verdict: GO (mechanism de-risk, 6/6 seeds) + wired (default-OFF) production hook, byte-identical-off verified in the data on both leaf branches, load-bearing on its own lesion.** `research/coordination/scaffold_retirement_backlog.md` rank-16 named the dopamine-to-write-magnitude mapping "MED - partial: homeostasis-half retired" — LEVER-3 (`da_encoding_substrate_enabled`, default ON since 2026-08-25) already moved the POPULATION-level homeostatic regulation onto the substrate (`OneBrainComposer.apply_homeostatic_scaling`, a genuine synaptic-scaling rule read from measured neural activity). What LEVER-3 left untouched is the PER-WRITE LEAF: "given the live self-produced DA, how much gain does THIS fact get" was still `_gain_map()`'s closed-form `g = clip(g_min, g_max, 1 + k_DA*(DA - DA_baseline))` — host arithmetic on a scalar, not a neuron or synapse. This session builds a small spiking population that reads the SAME live DA through the SAME neuromodulator machinery other DA couplings in this codebase already use, and reads the write gain off that population's OWN firing rate instead.

**Not flipped.** `BRAIN_DA_ENCODING_SPIKING_GAIN` (new, DEFAULT OFF) gates the swap; `_gain_map()` remains the production default. Per `docs/TERMS.md`, this is **wired (default-OFF)**, not `closed` or `integrated` — the host shortcut is still what a default turn runs.

## The premise check (before building)

`bash tools/before_you_build.sh "DA write-magnitude host linear spiking gain"` and
`.venv-rag/bin/python tools/rag/rag_search.py "dopamine write magnitude encoding gain spiking neuromodulator" 5 --corpus all`
surfaced the retirement history directly: `2026-06-19-dopamine-encoding-gain-derisk.md` (Tier-2 #6, the original host gain), `2026-08-19-neuromod-spiking-da-mode-GO.md` (the spiking SNc nucleus that produces the live DA level this coupling reads), and `2026-08-21-da-gated-encoding-wired-into-chat-GO.md` (the production wire-in). Reading `webapp/da_encoding_drives_chat.py` in full confirmed the exact CURRENT split the backlog line summarizes: LEVER-2 (`homeostatic_step`, host EMA + multiplicative scale) is the pre-2026-08-25 attempt, explicitly documented in its own comment block as "a documented PROXY for an emergent spiking homeostatic-plasticity rule on the substrate synapses"; LEVER-3 (`da_encoding_substrate_enabled`, default ON) is what actually retired that proxy onto `OneBrainComposer.apply_homeostatic_scaling()` — a real synaptic-scaling consolidation pass. Both LEVER-2 and LEVER-3's own per-write paths, however, still call the identical `_gain_map()`/`da_to_encoding_gain` closed form for the ONE remaining leaf computation. No existing finding retires that leaf; this is a fresh mechanism, not a re-derivation.

## The mechanism

A new population, **write_gain** (40->80 excitatory `IZH2007_HIPPO_PYRAMIDAL` neurons — the SAME CA1/CA3-pyramidal cell class the coupling's own biology citation names, Lisman & Grace 2005's hippocampal-VTA loop / Kandel D.16 dopamine-gated memory encoding), is built on its own minimal bridge (`research/runners/_da_write_gain_spiking_derisk.py::_build_write_gain_bridge`, NO `sim/` edit — region + `NeuromodulatorConfig` only). It receives:

1. a **fixed background "write-event" current** (260pA, comfortably above the preset's ~109pA rheobase) — the host-legitimate environmental trigger "a fact is being taught right now," the same class of boundary input every other faculty in this codebase uses to drive a population (`da_mode_drives_chat`'s engagement afferent, the GNW stop-trigger's relay pools);
2. a **DA-modulated excitability drive** via `sim.neuromodulators.ModulatorTarget(target_type="excitability_drive", scope="group:write_gain", sensitivity=+260)` — the IDENTICAL target_type/scope idiom `_neuromod_spiking_da_mode_derisk.da_nucleus_config()` already uses for `scope="group:str_D1"` (sensitivity positive, D1R Gs-excitatory), reused here for a hippocampal- rather than striatal-gated readout of the SAME live DA concentration `da_level_of(chat)` already produces (the #76/#79 spiking-SNc-derived level — UNCHANGED; this file does not touch how DA is produced, only how a gain is read FROM it).

The population's own membrane integration and spiking response to that DA-modulated current — not a python formula — decides its firing rate; the rate (read over a 200ms settle + 600ms window) is what the write gain is computed from. The **only remaining host arithmetic** is a two-point affine unit conversion (Hz -> gain units), calibrated against the SAME two DA reference points `da_mode_drives_chat.py` already established (0.05 "rest floor", 1.24 "arousal ceiling") — the same class of rate<->concentration/current transduction this codebase already treats as legitimate neural plumbing throughout the DA family (DA concentration ITSELF is produced from SNc firing via an analogous linear `from_region_firing_signed` transduction).

**Lesion (this mechanism's own).** `lesion=True` builds write_gain with the excitability_drive target's sensitivity pinned to **0.0 at build time** — a static config value, not a plastic weight (this population has no STDP/Hebbian/structural plasticity of any kind: `enable_stdp=False`, `plastic_internal=False`, `enable_structural_plasticity=False`), so per `docs/TERMS.md`'s lesion condition it cannot regrow within the read window. With the target severed, write_gain's rate is IDENTICAL to its rate at DA==baseline for every DA level (the driving current at baseline is already zero either way: `sensitivity*(baseline-baseline)=0`), collapsing the gain to the DA-independent floor.

## The 6-seed gate (42/43/44/100/101/102; numpy-CPU; `research/findings/raw/_da_write_gain_spiking/6seed.json`)

Each seed builds its own population + reads a real spiking SNc's own (da_low, da_high, da_baseline) via `_da_composer_salience_cleanup_derisk.measure_da_levels` (reused, not hand-picked) to source the DA sweep. **All 6 seeds pass all gates:**

| seed | intact gain span | lesioned gain span | monotonic (DA-tolerant) | corr(spiking, host formula) | seeded (cfg.seed) | GO |
|---|---|---|---|---|---|---|
| 42 | 1.4718 | 0.0000 | yes | 0.9976 | yes | yes |
| 43 | 1.5187 | 0.0000 | yes | 0.9997 | yes | yes |
| 44 | 1.3525 | 0.0000 | yes | 0.9999 | yes | yes |
| 100 | 1.5105 | 0.0000 | yes | 0.9965 | yes | yes |
| 101 | 1.4810 | 0.0000 | yes | 0.9960 | yes | yes |
| 102 | 1.4832 | 0.0000 | yes | 0.9958 | yes | yes |

- **LOAD-BEARING**: the intact population's gain varies by 1.35-1.52 (gain units) across the DA sweep on every seed (gate threshold 0.15) — a genuine, non-vacuous differential.
- **LESION collapses the span to exactly 0.0000 on every seed** (gate threshold 0.05) and lands within 0.05 of the floor (1.0) — the "coupling severed" signature every other lesion in this codebase produces, here reproduced from a STRUCTURAL severance of a neuromodulator target rather than a host `if`.
- **MONOTONIC** (DA-tolerant: a decrease smaller than the measured single-read OU-noise floor, 0.08 gain units, does not fail the check — two DA points ~0.007 units apart (an exploratory-sweep gap, not a stored field) were measured genuinely indistinguishable by this instrument, not a bug; see "instrument notes" below) on every seed. <!--derived-->
- **PARITY** with the pre-existing host formula: correlation 0.9958-0.9999 across the 6 seeds. Reported as correlation, not claimed exact — a genuinely different, spiking-derived mechanism is not required to bit-match the closed form it replaces (`docs/TERMS.md`'s `selective` discipline: the raw per-seed numbers are reported above, not a single pooled ratio). The correlation carries a small run-to-run jitter (a few parts in a thousand) from the substrate's own OU background-noise draws (see "instrument notes"); the gain SPAN, lesion collapse, monotonicity and determinism checks do not.
- **DETERMINISM** (the `cfg.seed` trap CLAUDE.md's Reproducibility section names): each seed's population was built TWICE and `cp_neuron_firing_thresholds` hashed; identical on all 6 seeds, confirming `cfg.seed` (not `actual_seed_used`) actually controls this substrate's heterogeneity, per `tests/test_determinism.py::TestSubstrateActuallySeeded`'s own standard.

### Instrument notes (an OU-noise floor found and handled, not hidden)

The substrate's own Ornstein-Uhlenbeck background-current process (`cfg.enable_ou_process`, default True, `ou_std_current_pA=100`) was deliberately left ON (biological realism, matching precedent) rather than suppressed. A single short read gives a per-point noise std of ~0.2-0.3Hz on a ~20Hz calibration span; two DA points closer than ~0.01 units are genuinely indistinguishable by a single read. Fixed by (a) averaging 3 independent reads for calibration and for the gate's own per-point comparisons (a validation-only cost; the cheap production read stays a single 800-step window), and (b) a monotonicity tolerance (0.08 gain units) set a small factor above the measured noise floor, not chosen to paper over a real reversal — both are documented inline in `_da_write_gain_spiking_derisk.py` with the measured numbers that justify them.

## The production hook (wired, default-OFF; `research/findings/raw/_da_write_gain_spiking/hook_verify.json`)

`webapp/da_encoding_drives_chat.py`'s `encoding_gain_for()` now calls a new `_leaf_gain(da, g_min, g_max)` helper at both its existing leaf call sites (the LEVER-3 substrate/default branch, floor=1.0; the raw/ablation branch, floor=0.5) instead of inlining `_gain_map()` directly. `_leaf_gain` dispatches to `spiking_write_gain` (LAZY import) only when `da_encoding_spiking_gain_enabled()` (`BRAIN_DA_ENCODING_SPIKING_GAIN`, DEFAULT OFF) is armed; otherwise it is a bare `_gain_map()` call, byte-for-byte what ran before this change.

- **(A) OFF byte-identical**, asserted in the data (exact float equality, not inferred): identical on BOTH leaf branches.
- **(B) ON load-bearing + parity**: span 1.440 across the DA sweep, correlation 0.9994 with the host formula.
- **(C) this mechanism's OWN inner lesion** (`BRAIN_DA_ENCODING_SPIKING_GAIN_LESION`, new, distinct from the pre-existing `BRAIN_DA_ENCODING_LESION` and `BRAIN_DA_DRIVES_LESION`): span 0.0000, lands at the floor.
- **(D) the pre-existing outer `da_encoding_lesioned()` gate still pins g=1.0 regardless of the new flag** — LEVER-4 does not touch, weaken, or bypass it (`encoding_gain_for`'s first line still short-circuits before `_leaf_gain` is ever reached).
- **(E) the new spiking module is never imported while the flag is off**, confirmed in a fresh subprocess (`sys.modules` checked directly) — the substrate is never even built when off, not merely computed and discarded.
- **Regression check on the PRE-EXISTING coupling**: `research/runners/_da_encoding_wired_verify.py` (the full `brain_chat`-handler-level proof for the whole LEVER-1/2/3 coupling, unmodified by this session) was re-run through the real handler; its own artifact `research/findings/raw/_da_encoding_wired/verify.json` reproduces the SAME numbers as before this session's edit (`g_high=2.4773555339723594`, `g_low=1.0`, both lesion arms `1.0`, `attribution_to_live_DA_read=1.0`) — this session's edit to `encoding_gain_for()` introduces no regression on the default (flag-off) path.

## Honest residuals (named, not claimed closed)

1. **The two-point Hz-to-gain rescale is host arithmetic** — a unit conversion, not the decision-bearing step (which is now the population's own f-I response to a DA-modulated current), but still arithmetic on a scalar. Structurally identical to how this codebase already treats the DA-concentration<->pA and Hz<->DA-concentration transductions elsewhere as legitimate neural plumbing, named here rather than left implicit.
2. **The write-event background current (260pA) is a host constant**, not itself derived from "a fact is being written" via any neural detector — the same class of environmental-boundary input already accepted throughout this codebase (a message arriving, a turn's content-token count). A neural "is this a write event" detector is a separate, unscoped faculty.
3. **Not flipped to production default.** `_gain_map()` remains what a real chat turn runs. Flipping requires its own default-on flip-gate soak (mirroring `2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP.md`'s own precedent for LEVER-3), out of scope for this de-risk per the task's explicit instruction.
4. **A pre-existing, unrelated latent bug was found and NOT fixed here** (logged `research/FAILURE_LOG.md`, 2026-09-05, NOT-GATEABLE): `_da_composer_salience_cleanup_derisk.py::_build_snc_bridge` (reused for this de-risk's real-DA-reference sourcing) never disables `cfg.enable_structural_plasticity` (default True), which crashes (caught, logged CRITICAL, non-corrupting) on any zero-synapse bridge. Fixed in THIS session's OWN new bridge only; retrofitting the shared, already-GO'd function risks perturbing the RNG trajectory other shipped findings (I-7-b) were validated against, and needs its own re-verification pass.

## Files

- `research/runners/_da_write_gain_spiking_derisk.py` (NEW) — the write_gain population build, the rate->gain calibration, the 6-seed gate.
- `research/runners/_da_write_gain_spiking_hook_verify.py` (NEW) — the production-hook-level byte-identical/load-bearing/lesion/lazy-import proof.
- `webapp/da_encoding_drives_chat.py` (EDIT, additive) — `da_encoding_spiking_gain_enabled()`, `da_encoding_spiking_gain_lesioned()`, `_leaf_gain()`; `encoding_gain_for()`'s two `_gain_map()` call sites now route through `_leaf_gain()`. `git diff sim/` is empty.
- `research/FAILURE_LOG.md` (EDIT) — the `_build_snc_bridge` structural-plasticity crash, logged NOT-GATEABLE.
