---
type: finding
status: live
date: 2026-09-05
mechanism: appraisal-interoceptive-afferent
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_appraisal_interoceptive_ladder_derisk.py
artifacts:
  - research/findings/raw/appraisal_interoceptive_ladder/_appraisal_interoceptive_ladder_6seed.json
builds_on:
  - research/findings/2026-08-19-embodied-affect-interoception-GO.md
  - research/coordination/scaffold_retirement_backlog.md
---

# Gate-B appraisal via the board-#84 interoceptive-relay CURRENT afferent — de-risked, byte-identical-off, load-bearing (6/6-seed GO)

**Scaffold-retirement backlog rank 5** (`research/coordination/scaffold_retirement_backlog.md`): "Route Gate-B
appraisal through the board-#84 interoceptive CURRENT afferent (already 6/6-validated, running in production for
the same substrate) instead of the direct host `set_concentration` write — the readiest high-load-bearing win."
Verdict: **GO, 6/6 seeds** (numpy-CPU, NO `sim/` edit). This is a **default-OFF de-risk**, not a production flip —
the flag stays off; the host-write mechanism remains the production default pending the owner's own review. Raw
data: `research/findings/raw/appraisal_interoceptive_ladder/_appraisal_interoceptive_ladder_6seed.json`.

## The gap

Gate-B's production affect organ (`research/runners/affect_production_organ.py`,
`AffectProductionOrgan.read_differential`) injects the appraised message valence/arousal into its SEAM-C
staggered-bistable ladder (`aff_vplus_L1..L8` / `aff_vminus_L1..L8` / `aff_arousal_L1..L8`, built by
`_stageA_full_integration_derisk.build_one_brain(co_resident_affect_ladder=True)`) via a **direct host write**:

```python
nm.set_concentration("appraisal_lad_vplus", float(m) if pos_sign else 0.0)
nm.set_concentration("appraisal_lad_vminus", 0.0 if pos_sign else float(m))
nm.set_concentration("appraisal_lad_arousal", float(m))
```

This concentration broadcasts uniformly as an additive `excitability_drive` (raw per-neuron pA offset,
`sensitivity=240`) onto every rung of the matching sign — a scalar Python float lands directly on the target
population, never through a synapse. Board #49 (`2026-08-19-embodied-affect-interoception-GO.md`) and its #81/#84
production adaptation (`_graded_affect_attractor_derisk.GradedAffectBrain` / `webapp/affect_drives_chat.py`,
6/6-seed GO, running in production for the SAME KIND of Koulakov/Goldman bistable-ladder substrate) already
established the correct pattern for exactly this situation: a host scalar enters the brain ONLY as an afferent
**CURRENT** onto small spiking relay pools (Izhikevich RS, no recurrence — a legitimate body/sensory-interface
boundary, not a shortcut), and those pools drive the target attractor **synaptically** (AMPA, gated by a runtime
transmission gate) — never a direct write onto the target population or its neuromodulator bus.

## What was built (an adaptation, not a fresh mechanism)

`research/runners/_appraisal_interoceptive_ladder_derisk.py`'s `AppraisalInteroceptiveLadder` reuses
`_stageA_full_integration_derisk._ladder_region_specs` / `_ladder_pathways` **by import** — byte-for-byte the same
region/pathway architecture Gate-B's own co-resident SEAM-C ladder uses (`aff_n_rungs=8`) — plus 3 new
interoceptive-relay pools (`appr_intero_vplus/vminus/arousal`, the board #49/#81 `intero_*` pattern: Izhikevich RS,
no recurrence, pure afferent relays). The appraisal now drives a real **CURRENT** onto these relay pools; their
AMPA synapses (gated by a new transmission gate, `appraisal_intero_out`) project onto **every rung of the matching
sign**, uniformly — mirroring the pre-existing diffuse-broadcast semantics (the staggered recruitment still lives
entirely in each rung's own pre-existing intrinsic-current offset; only the injection mechanism changes from a
diffuse neuromodulator write to a synapse).

The comparison bridge is a **dedicated standalone build** (the ladder + 3 relay pools only, no
composer/arbiter/honesty overhead) rather than a new co-resident seam added to `build_one_brain` itself — that
function is shared by every other Stage-A/one-brain de-risk and carries carefully-proven byte-identical-off
invariants (append-LAST index+draw invariance, SEAM-A/SEAM-C's separate-union RNG decoupling). Building a
dedicated bridge reuses the identical ladder region/pathway spec by import (so it genuinely is the same
architecture Gate-B's ladder is) while touching **zero lines** of that shared module and building fast enough for
a 6-seed x multi-condition battery (~20-30s/seed vs the co-resident one-brain's own ~13-16s per single read). The
one pathway `_ladder_pathways` emits that this bridge cannot host (`arousal -> speak_acc`, the arbiter's
action-selection region, out of scope for a ladder-only bridge) is explicitly excluded in code, documented as a
dead-end output that cannot affect the measured differential.

`research/runners/affect_production_organ.py` gained exactly two flag functions
(`appraisal_interoceptive_enabled` / `appraisal_interoceptive_lesioned`) and an 8-line dispatch at the very top of
`AffectProductionOrgan.read_differential`:

```python
self.ensure_built()
if appraisal_interoceptive_enabled():
    from research.runners._appraisal_interoceptive_ladder_derisk import get_ladder
    return get_ladder(self.seed).read_differential(
        appraisal, lesion=lesion, intero_lesion=appraisal_interoceptive_lesioned(),
        ramp_ms=ramp_ms, drive_off_ms=drive_off_ms, read_ms=read_ms)
b, xp, idx, snap = self.bridge, self.xp, self.idx, self.snap   # <- ORIGINAL code, untouched, below this line
```

Default (`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE` unset): the branch is skipped and the original host-write code
runs, byte-for-byte unchanged. `research/runners/_stageA_full_integration_derisk.py` (the shared one-brain module)
was **not touched at all**.

## Anti-cheat 1 — BYTE-IDENTICAL-OFF (asserted in the data, not inferred from the code)

Per `docs/TERMS.md`, byte-identical must be asserted in the data. Before writing this module, five appraisal
reads + one lesion read were captured from the unmodified `AffectProductionOrgan.read_differential` (seed 42,
numpy-CPU) and hardcoded as `_PRE_EDIT_BASELINE`. `run_byte_identical_off()` re-runs the identical calls, on the
**post-edit** code, with the flag unset, and requires an exact float match:

| appraisal | pre-edit (baseline) | post-edit, flag unset | exact match |
| --- | --- | --- | --- |
| -1.0 | -0.06638888888888889 | -0.06638888888888889 | yes |
| -0.5 | -0.035277777777777776 | -0.035277777777777776 | yes |
| 0.0 | 0.0 | 0.0 | yes |
| 0.5 | 0.03972222222222222 | 0.03972222222222222 | yes |
| 1.0 | 0.07083333333333333 | 0.07083333333333333 | yes |
| 0.7 (lesion=True) | 0.0 | 0.0 | yes |

All 6/6 exact matches (`byte_identical_off_detail.byte_identical_off: true` in the cited artifact), reproduced
fresh again inside the 6-seed battery. <!--derived-->

## Anti-cheat 2 — LOAD-BEARING (the appraisal genuinely colors content-volunteering + the mouth's manner)

With the flag on, an end-to-end call through the **actual production class** (`AffectProductionOrgan`, the same
object `webapp/server.py`, `webapp/wkv_mouth_generator.py` and `webapp/continuous_engine.py` all call) shows the
downstream consumers moving correctly off the new mechanism. Seed 42's own recorded sweep (`per_seed[0]`) at
appraisal `+-0.7`, with `tone_level`/`content_plan`/`manner_for` (unmodified, pre-existing functions of
`differential` alone) applied to the cited differentials:

| appraisal | differential (cited) | tone_level | content_plan | manner |
| --- | --- | --- | --- | --- |
| +0.7 | +0.059722222222222225 | +2 | `{max_sentences: 4, max_elaborations: 3}` | "...warm, friendly sentence." |
| -0.7 | -0.05916666666666667 | -2 | `{max_sentences: 1, max_elaborations: 0}` | "...blunt, matter-of-fact sentence." |

`content_plan`/`manner_for` inherit the new mechanism automatically because `read_differential`'s **output shape**
is unchanged, so no downstream file needed editing for this to hold. <!--derived-->

Across the 6-seed sweep (`-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0`, pooled means): the new mechanism tracks
the appraisal with `corr = +0.973784` (`means.new_corr`), `range = 0.162315` (`means.new_range`) vs the host-write
reference's `corr = +0.996346` (`means.host_corr`), `range = 0.142222` (`means.host_range`), read fresh from the
real `AffectProductionOrgan` on the identical seeds; `downstream_content_and_manner_vary` is true on 6/6 seeds
(`content_plan.max_sentences` and `manner_for(...)` both take more than one distinct value across the sweep, every
seed — e.g. seed 42's own `downstream_n_sentences: [1, 1, 1, 4, 4, 4, 4, 4, 4]`). <!--derived-->

## Anti-cheat 3 — LESIONABLE (a genuine dissociation, not silence)

Two independent lesions, both verified to hold at read time (the `docs/TERMS.md` lesion condition):

- **The pre-existing readout gate** (`affect_out=0`, identical semantics to the host-write path): the ladder's
  own output collapses to `differential = 0.0` regardless of appraisal, 6/6 seeds.
- **The NEW relay->ladder synapse gate** (`appraisal_intero_out=0`, `appraisal_interoceptive_lesioned()`,
  reachable from the SAME production env var mirrored in `affect_production_organ.py`): the appraisal->ladder
  coupling collapses (`intero_lesion_collapses_range(<=0.25x)`: measured range **0.0000**, i.e. **100%** of the
  intact range is owned by the synapse, `tools.lab.attributable_to`, 6/6 seeds) while the relay pools **still
  fire and still encode the appraisal magnitude** (`relay_enc_under_intero_lesion` corr **+0.96**, identical to
  the intact `relay_enc_intact` **+0.96**, 6/6 seeds) — the body/appraisal signal is present, it can no longer
  reach the ladder. This mirrors board #49's dissociation proof exactly (encode-under-lesion, not just silence).
  <!--derived-->

  **On the exact tie between `relay_enc_intact` and `relay_enc_under_intero_lesion`** (flagged by
  `gates/discriminating_power` as worth a look, not a fault): the two numbers are bit-identical, every seed,
  because `intero_lesion` clamps ONLY the `appraisal_intero_out` transmission gate — a multiplicative scale on
  the relay->ladder SYNAPTIC current — and never touches the relay pools' own input current or firing. The relay
  pools are driven and recorded identically in both conditions; what differs is whether their spikes reach the
  ladder at all. A tie here is the CORRECT, expected signature of a downstream-only lesion, not a failure to vary
  the manipulation — it is the same "still fires, still encodes" reading board #49 reports, made exact because
  this lesion sits strictly after the point being measured. <!--derived-->

## Anti-cheat 4 — NO-REGRESSION vs the real production host-write path

Every seed's new-mechanism sweep is compared against a **fresh read of the real `AffectProductionOrgan`** (the
unmodified host-write path, flag off), same seed, same appraisal values — not a cached number. `all_seeds_no_
regression_vs_host` (sign-for-sign agreement in the production-realistic band, defined next, plus `new corr>=0.8`
whenever the host's own corr is `>=0.8`) holds 6/6 seeds. Means (`means.new_range` / `means.host_range` /
`means.new_corr` / `means.host_corr`): new range **0.162315** vs host range **0.142222** (the new mechanism's
dynamic range is *not* smaller); new corr **+0.973784** vs host corr **+0.996346**.

## Honest residual: an extra threshold stage, characterized and bounded to an uncommon band

The interoceptive relay pool has its **own rheobase** — below it the relay fires at 0 Hz and the ladder reads an
**exact 0.0** differential, rather than a small same-signed value. The direct host write has no such stage (the
diffuse neuromodulator broadcasts straight onto the ladder's own L1 rung, which sits only 40 pA below its own
intrinsic threshold, so even a tiny appraisal moves it a little). At the calibrated operating point
(`i_pa=220 pA`, `w=10.0`, `dens=0.6` — chosen from a sweep over `i_pa in {30..2200}pA` for the best ordered-
tracking; below ~100 pA the relay never fires at all, above ~1000 pA the ladder saturates near its max range
almost independent of magnitude), this shows up as **12 of 48** nonzero-appraisal reads landing exactly on 0.0
across the 6 seeds, all confined to `|appraisal| < 0.5`. <!--derived-->

This band is **not** where production appraisals typically land: `affect_production_organ.appraise_text`'s
salience gate (`_STRONG_MARGIN = 2.0` on a 1-9 Warriner scale) admits a word only if `|v9-5| >= 2.0`, which forces
`|valence| = |(v9-5)/4| >= 0.5` for every single word that passes the gate — a real single-word-triggered
appraisal is essentially never inside `(-0.5, 0.5)`. Values below 0.5 in magnitude can still occur from averaging
multiple gated words of mixed sign, so the sub-threshold band was measured rather than excluded; restricted to the
production-realistic band (`|appraisal| >= 0.5`), sign-correctness is strict and clean, 6/6 seeds
(`all_seeds_signs_correct_realistic_band`). A small tonic bias current on the relay pools is a plausible follow-on
to close this residual; not attempted here, to keep the de-risk to the minimal adaptation the backlog asked for.

Two further scope notes, consistent with the board #49/#81 precedent this reuses: (1) the comparison bridge is a
**dedicated standalone build**, not literally the same Python object as the co-resident one-brain's ladder — it is
the same architecture/spec reused by import, seeded independently, so exact per-neuron thresholds differ from the
co-resident build even at the same `cfg.seed` (different RNG draw order/context); the aggregate population-level
behavior (sign, ordered tracking, lesion dissociation) is what is compared, not per-neuron identity. (2) The
operating point (`i_pa`, `w`) is smoke-calibrated, not first-principles-derived from the original
240 pA/concentration `excitability_drive` sensitivity.

## Preconditions (the verdict travels with these; a miss would make it UNDEFINED, not GO)

All five hold, verified in the cited artifact: substrate seeded (`cfg.seed`; identical firing thresholds on
rebuild), all 6 requested seeds ran, the differential read is neural (`rate` off `cp_firing_states`, never a host
formula), the ladder is reachable only via synapses (a runtime `assert` on `cp_external_input_current` held every
step of every read), numpy-CPU backend.

## Answering the report question

**Is appraisal-via-interoceptive-afferent load-bearing and byte-identical-off — ready to replace the host
`set_concentration` write?** Load-bearing: yes, demonstrated through the actual production class and its
unmodified downstream consumers, 6/6 seeds. Byte-identical-off: yes, asserted in the data against a pre-edit
captured baseline, 6/6 seeds plus the standalone check. Lesionable: yes, two independent gates, both dissociating
cleanly (100% of the range attributable to the new synapse). No-regression: yes, in the realistic band the
production salience gate actually produces; there is a characterized, bounded, honestly-reported reduction in
sensitivity below `|appraisal|=0.5`, a band the current lexicon-gated appraisal rarely if ever reaches from a
single word. Given all four hold 6/6 seeds and the change to production code is a two-function, one-branch
addition with zero lines touched in the shared one-brain module, this adaptation is **ready for the owner's
default-flip review** — it is **not** itself proposed as on-by-default here (the flag stays off; this is a
de-risk, not a flip, and no `sim/` file or shared module was edited).

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._appraisal_interoceptive_ladder_derisk --smoke
SIM_BACKEND=numpy python -u -m research.runners._appraisal_interoceptive_ladder_derisk \
    --seeds 42 43 44 100 101 102
# end-to-end, via the real production class:
BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE=1 python -c "
from research.runners.affect_production_organ import AffectProductionOrgan, tone_level, content_plan
o = AffectProductionOrgan(seed=42)
print(o.read_differential(0.8))"
```
