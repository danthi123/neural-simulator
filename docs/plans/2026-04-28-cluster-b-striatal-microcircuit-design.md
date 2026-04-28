# Cluster B — Striatal microcircuit (Design)

**Status:** design draft. Citation-grounding deferred until the textbook catalog session lands; will be revised against Kandel 6e references before implementation begins.

**Strategy context:** [`2026-04-28-cheat5-real-options-survey.md`](2026-04-28-cheat5-real-options-survey.md) (cluster B is one of 5 in the cheat-5 buildout). [`2026-04-28-cheat5-post-v4-reframe.md`](../../research/findings/2026-04-28-cheat5-post-v4-reframe.md) (current cheat-5 status).

## Goal

Implement the three biological mechanisms inside striatum that interact strongly with cross-projection refinement:

1. **D1/D2 plasticity asymmetry** — D1 MSNs LTP under +DA / LTD under −DA; D2 MSNs invert both.
2. **Striatal fast-spiking interneurons (FSIs)** — dedicated interneuron population providing fast (millisecond-scale) broadcast inhibition; sharper WTA than MSN-MSN lateral.
3. **Cholinergic interneurons (TANs / Tonically Active Neurons)** — release ACh in response to salient events, gating corticostriatal plasticity in temporal windows.

These together form a "complete striatal subsystem" — testing them as a unit lets us validate microcircuit dynamics independent of cheat-5, and any cross-projection improvement gets attributed to having the right scaffolding.

## Why this cluster, why now

Multi-goal patch-matrix showed cross-projections aren't fundamentally broken — they're under-constrained. Cluster B addresses three of the most direct constraints:

- **D1/D2 asymmetry** lets cross-projections to D1 vs D2 encode complementary "do X" / "don't do Y" signals. Without it, every cross-projection gets the same teaching signal, which is what makes patch-matrix variance-prone.
- **FSIs** provide a millisecond-scale gate that's faster + more selective than MSN-MSN lateral inhibition. Cross-projections fight against the wrong action choice; FSIs help suppress that fight.
- **TANs** ensure plasticity only consolidates during salient events. Random uncorrelated firing won't slowly drift cross-projection weights toward useless configurations.

Cluster B has the highest cheat-5 cost-effectiveness of the five clusters: smallest implementation cost, most direct interaction with cross-projection refinement.

## Three sub-features

### B.1 — D1/D2 plasticity asymmetry

**Biology (sketched, to be cited from Kandel):**
- D1 MSNs express D1 receptors → coupled to Gαs → cAMP → PKA. +DA increases cAMP → potentiation of corticostriatal synapses.
- D2 MSNs express D2 receptors → coupled to Gαi → reduces cAMP → potentiation of corticostriatal synapses under −DA (reduced inhibition of cAMP).
- Net effect: D1 LTPs under +DA / LTDs under −DA. D2 LTPs under −DA / LTDs under +DA. (Shen et al. 2008 Science is the canonical reference; Kreitzer & Malenka 2008.)

**Implementation:**
- Existing `cp_synapse_action_tag`-style mechanism (or similar) to mark which synapses terminate on D1 vs D2. We already have D1 / D2 region splits in the BG cascade; just need to surface a per-synapse D1/D2 tag.
- Modify the reward + eligibility-driven plasticity rule:
  - Currently: `Δw = eligibility * reward * gain`
  - With asymmetry: `Δw = eligibility * (reward * sign[i]) * gain`, where `sign[i] = +1` for D1-terminating synapses and `-1` for D2-terminating.
- Single new flag: `--enable-d1-d2-asymmetry` (default off; flagship behavior unchanged when off).

**New unit tests:**
- `test_d1_d2_asymmetry_tag_assignment` — every synapse with `to_region` matching `str_D1_*` gets sign=+1; `str_D2_*` gets sign=-1; others sign=0 (no asymmetry).
- `test_d1_d2_asymmetry_inverts_weight_change` — with same eligibility + reward, D1 synapses move opposite direction from D2 synapses.

### B.2 — Striatal FSIs

**Biology:**
- ~1% of striatal cells; parvalbumin-positive; very fast firing (>200 Hz transient bursts).
- Receive convergent corticostriatal excitation; inhibit MSNs broadly via GABAergic outputs.
- Effective WTA gate at the millisecond scale — faster than the seconds-scale lateral MSN-MSN inhibition we already have via v3.
- Anatomically positioned to bias which action's MSN pool wins on a per-trial basis.

**Implementation:**
- New region per action: `str_FS_X` with ~5 neurons each (4 actions × 5 = 20 FSIs total). Use an Izhikevich preset that reproduces fast-spiking dynamics — `IZH2007_FS_CORTICAL_INTERNEURON` is close but FSI-specific tuning may be needed.
- New pathways:
  - `cortex_X → str_FS_X` (excitatory, dense, plastic=False) — drive FSIs from cortex.
  - `str_FS_X → str_D1_Y` and `str_FS_X → str_D2_Y` for ALL Y including X (each FSI broadcasts inhibition across the whole striatum, not just same-action).
  - These are GABAergic (FSIs are inhibitory); use existing inh-region machinery.
- Single new flag: `--enable-striatal-fsis` (default off).

**New unit tests:**
- `test_striatal_fsi_pathways_built` — when flag on, 4 new FS regions + 4 cortex→FS + 32 FS→MSN pathways present.
- `test_striatal_fsis_disabled_by_default` — flag off → no FS regions or pathways.

### B.3 — Cholinergic interneurons (TANs)

**Biology:**
- ~1-2% of striatal cells; tonically active (~5 Hz baseline) but pause briefly on salient events (reward, novel stimuli).
- ACh release at corticostriatal synapses modulates both LTP and LTD via M1/M4 muscarinic receptors.
- Net effect: ACh creates "windows" during which corticostriatal plasticity is enhanced or suppressed. Real BG only consolidates synapses when ACh says "now's the time."

**Implementation:**
- Extend the neuromodulator subsystem with a new neuromodulator: `acetylcholine`. Concentration baseline ~1.0; pauses on salient events.
- New target type for the existing `ModulatorTarget` framework: `plasticity_window_gate`. Synapses tagged with this target have their plasticity gain multiplied by `(1 - acetylcholine_concentration)` — high ACh suppresses, low ACh (during pause) permits.
- Production rule: ACh tracks reward magnitude with a "pause on reward" dynamics (concentration drops below baseline briefly when reward arrives, recovers slowly).
- Single new flag: `--enable-tans` (default off).

**New unit tests:**
- `test_tan_acetylcholine_pause_on_reward` — driving reward signal causes ACh concentration to drop briefly.
- `test_tan_plasticity_window_gating` — corticostriatal synapses' effective plasticity gain is highest when ACh is paused (low), lowest when ACh is at baseline.

## Validation

### Standalone biological correctness (NOT cheat-5 specific)

For each sub-feature, verify the cascade behaves more like real BG:

- **D1/D2 asymmetry**: in a standalone reward-prediction probe, D1 weights should grow under positive reward; D2 weights should grow under punishment. Diverging directions at the synapse level.
- **FSIs**: when two action-pools fire simultaneously, FSI activity should broaden out and gate the loser's motor pool faster than MSN-MSN lateral inhibition does. Measure: time-to-suppression difference.
- **TANs**: pause-on-reward dynamics; plasticity events cluster around the pause window.

These are *biology* validations, not behavior validations. Each is its own probe script.

### Integrated multi-goal eval

After all three sub-features land:

1. Re-baseline v3 + Cluster B (no cross-projections) under multi-goal. Should be ≤ v3 baseline 7.08 (cluster B shouldn't hurt).
2. Re-test patch-matrix (density 0.25) + Cluster B under multi-goal, n=3. Expected: variance drops; mean improves toward baseline. The "topology luck" signature should weaken because cluster B now provides scaffolding to consistently use the cross-pairs.
3. If patch-matrix + Cluster B beats baseline, consider running structural pruning (option 1) + Cluster B too.

If patch-matrix + Cluster B is consistently ≤ 6.0, that's the first real cheat-5 partial closure signal.

## Implementation order

Within Cluster B:

1. **D1/D2 asymmetry first** (smallest, most testable, most directly impacts plasticity). Standalone biological probe + unit tests + multi-goal eval = ~2 days.
2. **Striatal FSIs second** (medium scope; new region + pathways + dynamics). ~3-4 days.
3. **TANs third** (largest scope; extends neuromodulator subsystem; risk of subtle interaction with existing reward modulation). ~3-4 days.

Each sub-feature is its own commit + tests + cheat-5 re-eval. Cluster B is "done" when all three are integrated AND the integrated multi-goal result is reported.

## Wait-for-textbook checklist

Before starting implementation, the textbook catalog session should be at least past Section IV (synaptic plasticity) so the design doc can be revised with citations:

- [ ] Cite Kandel chapter on D1/D2 differential plasticity
- [ ] Cite chapter on FSI properties + corticostriatal dynamics
- [ ] Cite chapter on cholinergic interneurons + ACh receptor types
- [ ] Cross-check the proposed implementation against the textbook descriptions; flag any deviations as `[implementation simplification: ...]`

## Done criteria

- [ ] D1/D2 asymmetry implemented + biology probe passes + unit tests pass
- [ ] FSIs implemented + biology probe passes + unit tests pass
- [ ] TANs implemented + biology probe passes + unit tests pass
- [ ] Integrated multi-goal eval (v3 + Cluster B, no cross): ≤ 7.08 baseline
- [ ] Integrated multi-goal eval (patch-matrix + Cluster B): variance reduction + mean improvement reported
- [ ] Findings doc + propagation per the standard template
- [ ] CLAUDE.md cheat-5 section + SCIENCE_ROADMAP §4.7 + INDEX + CHANGELOG + memory updated

## Out of scope

- D1/D2 differential intrinsic excitability (different baseline firing rates) — possibly relevant but probably second-order; defer.
- Specific receptor subtypes (D1A/D1B/D2L/D2S, M1/M2/M3/M4/M5) — we model the net effect, not receptor pharmacology. Defer to a hypothetical Cluster H "receptor pharmacology" if ever needed.
- Striatal SOM/NPY interneurons (a third interneuron class) — less relevant for the cross-projection question; could be added later.
- Patch-matrix anatomical compartments (real striatum has patch + matrix sub-regions with different connectivity rules) — that's a Cluster E refinement, not Cluster B.
