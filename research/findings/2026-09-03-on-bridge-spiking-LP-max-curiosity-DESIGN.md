---
type: finding
status: design
date: 2026-09-03
mechanism: curiosity-learning-progress-maximizing-selection-onbridge
builds_on: research/findings/2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md
reuses:
  - research/runners/_curiosity_seek_learn_onbridge_derisk.py
  - research/runners/_affect_marker_wta_derisk.py
  - research/runners/bg_action_selection_production_organ.py
artifacts:
  - research/findings/raw/lanes/curiosity/lp_max_selection_6seed.json
---

# DESIGN — the on-bridge SPIKING realization of learning-progress-MAXIMIZING curiosity selection

**This is a DESIGN doc (research + spec), not a build.** It specifies how the validated CPU-proxy mechanism —
learning-progress (LP) MAXIMIZING ask selection, 6-seed GO in
`2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md` (runner
`_laneB_curiosity_lp_max_selection_derisk.py`, artifact
`research/findings/raw/lanes/curiosity/lp_max_selection_6seed.json`) — becomes a
NEURAL selection on one `SimulationBridge`, per the project's brain-based-only standard (a host-computed
selection is a shortcut; the genuine next rung is neurons/synapses selecting the max-LP option). It names the
exact host shortcuts to convert, the reusable substrate assets, the spiking mechanism (biology-cited), the
cheapest CPU de-risk with a GO gate that matches the proxy's bar, and the expected failure modes. A follow-up
agent builds it. NO `sim/` edit is anticipated (every hook below already exists); RSS trivial.

## 1. The EXACT host-computed shortcut(s) in the current LP-max proxy

The CPU proxy (`_laneB_curiosity_lp_max_selection_derisk.py`) is entirely numpy. Two pieces are the
load-bearing shortcuts under the brain-based-only rule; a third is secondary.

**Shortcut A — the per-option LP estimate is a host numpy EMA.** `class Trace` holds `fast` and `tonic`
scalars updated by `fast += fast_alpha*(x-fast)` / `tonic += slow_alpha*(x-tonic)` where `x = 1 -
post-ask-novelty`; `slope = fast - tonic` (runner lines 103-119). The LP signal every arm consumes is this
host phasic-minus-tonic difference, not a neural read.

**Shortcut B — the MAX-selection over options is a host `argmax`.** Selection is `scores = {c: exploit(c) +
explore_beta/sqrt(count+1) ...}`; then `mx = max(scores.values()); c_ask = rng.choice([c for c in concepts
if scores[c] >= mx-1e-12])` (runner lines 165-174). The exploitation term for the real arm is `max(0,
traces[c].slope)` (line 159). So the whole point of the mechanism — "pick the option whose learning progress
is greatest" — is realized by `max(dict.values())` in Python. This host argmax is present in EVERY curiosity
runner including the DR-1 ON-BRIDGE runner (`_curiosity_seek_learn_onbridge_derisk.py` lines 527-531 does
`mx = max(want[c] for c in cands)` even though `want` is read from spikes). **Neural max-selection over the
option set has NEVER been built here — this is the true un-built core of the LP-max realization.**

**Shortcut C (secondary) — the count-based exploration bonus** `explore_beta/sqrt(count+1)` is host, but it
is identical across all arms (novelty-agnostic bookkeeping), so it is not the LP-specific mechanism. It maps
to an ACC/novelty exploration afferent; see §5 (carry it as a scoped shortcut, do not block on it).

**Which parts are already neural (in the DR-1 precedent, reused here):** the epistemic-gap/progress read
(the real Bogacz-Brown `RealAntiHebbianFamiliarity` gate, catalog D.04), and — crucially — the per-concept
LP *reward magnitude* as a spiking-SNc RPE read (`deliver_reward` -> `snc_B - snc_A` isolates r on spikes).
So Shortcut A is HALF-solved already: the DR-1 machinery produces a spiking per-concept reward that IS the
learning-progress magnitude. What the DR-1 runner then does with it (feed a host ELP veto, argmax novelty) is
the wrong consumer for LP-MAX. The design below re-points that spiking LP signal from a veto into the
SELECTION drive, and adds the missing neural max.

## 2. Reusable on-bridge assets (do NOT reinvent)

The breadth-scoping note said "no existing runner exists — would require inventing one." That is true of the
*integrated LP-max selector*, but every PRIMITIVE it needs is already GO on this substrate:

| need | reusable asset | what it gives |
|---|---|---|
| per-concept LP as spikes | `_curiosity_seek_learn_onbridge_derisk.py` `build_curiosity_bridge` + `deliver_reward` | `reward_us->snc<-striosome_value(GABA_B)` RPE critic; `snc_B - snc_A` = spiking learning-progress r in [0,1]; SNc learn-burst 14.7 Hz vs noisy 0.0 Hz (DR-1 6/6 GO) |
| N-channel spiking MAX-selection | `_affect_marker_wta_derisk.py` (biology `affective-marker-lateral-inhibition-wta.md`, 6/6 GO, default-ON) | N excitatory assemblies, each with a cross-inhibiting FSI sub-pool; per-channel drive current -> the winning assembly is read from `cp_firing_states`; includes a mis-routing (permuted) anti-cheat |
| BG selection-by-disinhibition (alt.) | `bg_action_selection_production_organ.py` / `build_selector_bridge` (`BRAIN_BG_SELECT`, 6/6 flip-soak GO) | D1->GPi direct-path disinhibition, GPe/STN indirect path, GPi->thalamus commit, cross-channel commit inhibition; per-candidate salience bias + shared enabling barrage; a real race, no host max |
| drift-free reads | DR-1 `_snapshot_state`/`_restore_state` wash-out (EMERGE-61 fix) | every want/value/LP read is a function of learned weights alone, not prior-ask adaptation drift |
| curiosity drive | `from_novelty` neuromodulator + `excitability_drive` scope=group:X (`sim/neuromodulators.py:732`) | already filled; here re-used only as the moat/veto gate, NOT as the selection drive |

The design is therefore a COMPOSITION: DR-1's spiking LP signal -> drives -> the affect-marker-style
N-channel WTA (or the BG selector) -> the winner is the ask. The novel wiring is the LP->channel-drive map
and the multi-option competition; both primitives are proven.

## 3. The spiking-realization design

### (a) Per-option learning-progress as a spiking/synaptic signal

Each concept/option `c` carries a scalar LP-slope. Realize it as the DR-1 spiking reward read, low-passed
into a phasic-minus-tonic contrast:

- `lp_read(c)` = the DR-1 `deliver_reward(c, LP)` -> `reward_read` in [0,1] (the SNc burst the LP reward
  evokes, paired against the same concept's no-reward burst so the learned striosome V cancels; already on
  spikes, already drift-free).
- `lp_fast(c)`, `lp_tonic(c)` = two small RS pools per concept whose input current is driven by
  `lp_read(c)` at fast vs slow effective gains (equivalently, two TD low-passes realized as pool activity
  with different leak); the SELECTION drive is `max(0, rate(lp_fast(c)) - rate(lp_tonic(c)))`, matching the
  proxy's `max(0, slope)`. A learnable-while-improving concept has fast>tonic (positive slope -> strong
  drive); a mastered concept has fast~tonic~high (slope ~0); a NOISY concept has fast~tonic~0 (r~0 every ask
  -> no drive) — noisy-TV immunity BY CONSTRUCTION, now on spikes.

Cheapest first realization: skip the two-pool fast/tonic split and drive each channel directly with
`lp_read(c)` gated by a positivity threshold — the proxy's slope and the DR-1 reward-read are both
"how much did asking pay off"; the fast/tonic split only matters for the mastered-vs-improving distinction,
which the de-risk (§4) tests explicitly. Start simple, add the split only if the mastered case leaks.

### (b) Neural MAX-selection (the un-built core)

Route the N per-option LP drives into an N-channel competitive selector and read the winner from spikes:

**Primitive 1 (recommended for the de-risk) — FSI lateral-inhibition WTA** (the affect-marker recipe,
generalized from its 6 fixed marker assemblies to N=n_concepts option assemblies). Each option assembly `i`
receives `drive_pa[i] ∝ LP-slope(i)`; each assembly recruits a fast-spiking-interneuron sub-pool that
cross-inhibits every other assembly (Grossberg on-center/off-surround; Douglas & Martin canonical
microcircuit). The assembly whose LP drive wins the race fires; `cp_firing_states` names the winner. This is
the lightest, already-6/6-GO N-channel selector and its drive-swap (felt mood -> LP-slope) is a
one-quantity change.

**Primitive 2 (higher-fidelity, optional) — BG selection-by-disinhibition** (the `bg_action_selection`
organ, generalized from 2 channels to N). Each option = a striatal D1 channel; per-channel LP-slope is the
salience bias on that channel's MSN pool; a shared enabling barrage brings all channels toward threshold;
the first channel to cross the GPi->thalamus disinhibition commit threshold is selected (catalog A.04:
"selected channel = strongest inhibitory input from striatum -> GPi/SNr silenced -> thalamic target
released; selection is an emergent property of the entire reentrant network"). More faithful to "the brain
selects an action", and it already sits next to the striosome/SNc the LP critic uses. Heavier to stand up at
N channels; make it the second build, gated on Primitive 1 passing.

**The critical design constraint (from the 2026-06-06 action-selection deep research,
`2026-06-06-action-selection-readout-deep-research.md`).** A NAIVE INSTANTANEOUS WTA is a weak, unreliable
selector: (catalog B.04, Wilson 2007 / Tepper-Koós 2017) symmetric MSN->MSN mutual inhibition "does not
produce strong competitive interactions"; the real biological WTA substrate is FEEDFORWARD FSI inhibition,
not symmetric collateral feedback — which is exactly why Primitive 1 uses FSI sub-pools, not MSN
collaterals. And when competing drives are CLOSE, the faithful selector needs recurrent self-excitation +
integrate-to-threshold (Wang 2002 slow-NMDA reverberation attractor; Lo & Wang 2006 BG->SC commit
threshold; Stine et al. 2023 LIP-accumulate / SC-threshold division of labor). The affect-marker WTA is
honestly indecisive at a tuning boundary (reports "no clean winner"); LP-slopes of two co-improving
learnable options can be equally close. Mitigations, in order of preference: (i) keep the proxy's eps-greedy
+ tie-break tolerance (a "no clean winner" step just picks among near-winners — behaviorally fine); (ii) add
recurrent self-excitation to each option assembly so a weak-but-consistent LP lead integrates to a bound
before the WTA commits (Wang/Douglas-Martin); (iii) only if needed, a full accumulate-to-threshold commit.
Start with (i); it matches the proxy, which already uses eps-greedy.

### (c) Composition with the substrate + reused bridge hooks

ONE `SimulationBridge`, built by extending DR-1's `build_curiosity_bridge`:

- KEEP: `cue` (per-concept slices), `striosome_value`, `reward_us`, `snc`, the GABA_B critic pathways, the
  `curiosity`/`from_novelty` neuromodulator (now used ONLY for the moat gate: only ask concepts the gate
  reads NOVEL — unchanged), the wash-out `_snapshot_state`/`_restore_state`, `_measure_snc_neutral`,
  `deliver_reward`, `read_value`.
- ADD (the new selection layer): `n_concepts` `option` assemblies (RS) + their FSI cross-inhibition
  (Primitive 1) OR the D1/GPe/STN/GPi/thal channels (Primitive 2). Per step, set
  `cp_external_input_current[option_slice[c]] ∝ lp_slope_read(c)` (the §3a spiking LP), run a settle window,
  read the winner from `cp_firing_states[option_all]` (highest-rate assembly, drift-free via wash-out).
- REPLACE: the host `mx = max(want[c])` selection (DR-1 line 528 / proxy line 172) with the winner read from
  the selection layer. The novelty-driven `want` read is retained ONLY as the moat/candidate filter, not the
  selector.

No new `sim/` mechanism is required: `excitability_drive`, per-region external current, GABA_B, plastic
cue->striosome, and the FSI/BG region+pathway framework all exist. If the two-pool fast/tonic split (3a) is
adopted it is still additive runner-local regions, no `sim/` edit.

### (d) Flags + code-level sketch + where the runner slots

New runner: `research/runners/_laneB_curiosity_lp_max_onbridge_derisk.py` (mirrors the DR-1 naming). Flags:

```
--seeds 42 43 44 100 101 102     # 6-seed (matches the proxy + DR-1)
--smoke                          # tiny CPU smoke (few concepts, short budget), SIM_BACKEND=numpy
--selection {wta,bg}             # wta = FSI lateral-inhibition (default); bg = disinhibition (Primitive 2)
--fast-tonic                     # opt-in: two-pool phasic-minus-tonic LP drive (else direct lp_read drive)
--out <path>
```

Selection-layer sketch (Primitive 1, additive to `build_curiosity_bridge`):

```python
# regions: N option assemblies + one shared FSI sub-pool per assembly (EMERGE-11 / affect-marker recipe)
for i in range(n_concepts):
    regions.append(BrainRegion(name=f"opt_{i}", n_neurons=N_OPT, exc_fraction=1.0, ..., izh=RS))
    regions.append(BrainRegion(name=f"fsi_{i}", n_neurons=N_FSI, exc_fraction=0.0, ...,
                               izh=IZH2007_FS_CORTICAL_INTERNEURON))
for i in range(n_concepts):
    pathways.append(RegionPathway(f"opt_{i}", f"fsi_{i}", density=1.0, weight_mean=OPT_FSI_W))   # drive own FSI
    for j in range(n_concepts):
        if j != i:
            pathways.append(RegionPathway(f"fsi_{i}", f"opt_{j}", density=1.0,
                                          weight_mean=FSI_OPT_W))   # inhibit every OTHER assembly

def select_ask(bridge, cands, lp_slope):          # lp_slope[c] from the spiking DR-1 reward read (§3a)
    _restore_state(bridge, snap0)                  # drift-free
    for c in cands:
        bridge.cp_external_input_current[opt_slice[c]] = xp.float32(OPT_GAIN * max(lp_slope[c], 0.0))
    rate = {c: 0 for c in cands}
    for _ in range(W_SELECT):
        _advance(bridge)
        for c in cands: rate[c] += int(bridge.cp_firing_states[opt_slice[c]].sum())
    mx = max(rate.values())                        # read the SPIKING winner (not a host score-argmax)
    winners = [c for c in cands if rate[c] >= mx]  # ties -> eps/random pick (proxy-matched)
    return int(rng.choice(winners)) if rng.random() >= EPS else int(rng.choice(cands))
```

The `max(rate.values())` here is a READOUT of which assembly the neural competition left firing (permitted —
same status as `name_from_spikes` reading a word-pool winner), NOT a computation of the selection value; the
selection VALUE (which assembly wins) is set by the synaptic FSI race. (If a future rung wants the readout
itself neural, add a committed-burst/thalamic latch per the deep-research menu — out of scope for the
de-risk.) `cands` is the moat-and-drive-filtered candidate set from DR-1 (gate-novel AND drive-active),
unchanged.

## 4. The cheapest CPU de-risk (pool-runnable) + GO gate + failure modes

**The core assumption to validate before any full build:** a spiking FSI-WTA driven by per-option currents
∝ LP-slope selects the max-LP option as reliably as the host argmax, AND a noisy option (LP≈0) never wins
even when its NOVELTY is maximal. This can be tested WITHOUT the full curiosity loop — isolating the NEW
neural-selection piece from the (known-fragile) LP-estimate, so the two failure sources do not confound.

**Experiment (CPU, `SIM_BACKEND=numpy`, no GPU):** build ONLY the selection layer (N `opt`/`fsi` assemblies,
§3d). On each trial, feed each assembly a current from a KNOWN synthetic LP-slope vector (one clear
max-learnable, several mid, one noisy=0), plus — on a separate would-be-novelty channel — a HIGH novelty
current to the noisy option, to prove novelty cannot leak into the LP-driven race. Read the spiking winner.
Sweep many random LP vectors x 6 seeds.

**GO gate (matches the proxy's bar — selection correctness, noise avoidance, LP load-bearing, 6-seed):**

- **g_select** — neural WTA winner == host `argmax(LP-slope)` on >= 90% of trials, each of 6 seeds
  (42/43/44/100/101/102). (Selects the high-LP option, the proxy's g1/g3 analog.)
- **g_noisy** — the noisy option (LP≈0, novelty≈max) win-rate ≈ 0 across all trials/seeds (proxy g2
  noise-avoidance, now BY CONSTRUCTION on spikes; reported with raw per-option win counts, not a ratio
  alone, per `docs/TERMS.md` "selective").
- **g_loadbearing (lesion, verified to hold at measurement)** — replace the LP drive with a novelty drive
  (or equalize all LP currents): the winner must then track NOVELTY / go uniform, NOT the max-LP option — so
  the noisy high-novelty option now WINS. This proves the LP drive is causally required for noise-avoidance
  (proxy g4). The lesion (zeroed/redirected LP current) is static, so persistence is trivially verified.
- **g_specificity (permuted anti-cheat)** — reuse the affect-marker mis-routing: assembly `i` receives
  option `perm(i)`'s LP current. The reported winner must change / stop tracking the true max-LP option
  (proxy g5 LP-specificity).

**A GO here de-risks the neural max; the FULL build then integrates the DR-1 spiking LP into this layer and
re-runs the proxy's g1-g5 on spikes** (mastery, noise-avoidance, efficiency, LP-load-bearing,
LP-specificity) as the real closure gate.

**Expected failure modes (and mitigations):**

1. **WTA gives no clean winner / multiple fire** when LP-slopes are close (the affect-marker boundary case;
   EMERGE-11 showed WTA can burst or go silent). Mitigation: eps-greedy tie-tolerance (proxy-matched, cheap);
   then recurrent self-excitation + integrate-to-bound (Wang 2002) if the margin is too tight. This is the
   single most likely blocker and the reason the de-risk uses SYNTHETIC well-separated LP first.
2. **LP-signal drift/saturation** — the known lane-B fragility: the substrate-memory LP promotion was 1/6
   seeds (the 2026-08-02 finding's open substrate-expressivity question). Mitigation: the de-risk FEEDS
   synthetic LP so this cannot confound g_select/g_noisy; the full build inherits DR-1's drift-free wash-out
   read and, if still fragile, a homeostatic/normalized LP readout (flagged, not assumed).
3. **Novelty leakage into selection** — if option pools also receive novelty drive (as DR-1's `want` does),
   selection is captured by novelty and the whole LP-max point is lost. Mitigation (structural): option
   assemblies are driven by LP ONLY; novelty stays on the moat/candidate filter. g_noisy tests this directly.
4. **Exploration insufficiency** — dropping the host `beta/sqrt(count+1)` bonus (Shortcut C) may starve
   early exploration so mastery stalls. Mitigation: carry it as a scoped host shortcut for the de-risk; map
   it to an ACC/novelty exploration afferent in a later rung. Do not block the LP-max realization on it.
5. **FSI over-inhibition collapses all assemblies to silence** (the nav TRN-WTA 20.0 failure analog).
   Mitigation: tune `OPT_FSI_W`/`FSI_OPT_W` on the synthetic de-risk before wiring the loop; the affect-
   marker organ's working weights are the starting point.

## 5. Honest scope, terms, and open questions

- This doc claims NO new measurement; it cites the CPU-proxy 6-seed GO
  (`research/findings/raw/lanes/curiosity/lp_max_selection_6seed.json`) as the mechanism this realizes, and
  the DR-1 on-bridge 6/6 GO + the
  affect-marker/BG WTA GOs as the reused primitives. Per `docs/TERMS.md`: "closed" is NOT claimed (that
  requires integrated + default-ON + scaffold-retired); this is a design toward an on-bridge de-risk.
- The residual genuinely-new engineering is small and isolated: the LP->channel-drive map + an N-channel
  competition, both from proven primitives. The "would require inventing a runner" framing overstated it —
  the runner is a composition, and the de-risk (§4) is a ~minutes CPU run like the DR-1 smoke.
- Open question carried forward from lane B: the seed-robustness of the LP MEMORY read (1/6 substrate-
  memory). The de-risk deliberately isolates the selection layer from it; if the full loop then fails on the
  LP-estimate (not the selector), that is the LP-readout homeostasis sub-problem, not the max-selection one,
  and is scoped separately.

## Citations

- CPU proxy (this builds on): `2026-08-07-laneB-curiosity-learning-progress-MAXIMIZING-selection-CPU-proxy-6seed-GO.md`;
  LP-slope precursor `2026-08-02-laneB-curiosity-learning-progress-slope-CPU-proxy-6seed-GO-next-onbridge-realization.md`.
- On-bridge curiosity precedent (reused): `2026-07-30-lane-B-curiosity-DR1-onbridge-6seed-GO.md`,
  runner `_curiosity_seek_learn_onbridge_derisk.py`.
- N-channel spiking WTA (reused primitive): `research/biology/affective-marker-lateral-inhibition-wta.md`,
  runner `_affect_marker_wta_derisk.py`.
- BG selection-by-disinhibition (alt. primitive): `bg_action_selection_production_organ.py`.
- Action-selection biology + literature synthesis (the WTA-fidelity constraints):
  `2026-06-06-action-selection-readout-deep-research.md` — catalog A.04/A.05/A.07 (BG disinhibition
  selection), B.04 (Wilson 2007; Tepper-Koós 2017 — FSI feedforward, not MSN collateral, is the real WTA
  substrate), G.16 (accumulate-to-bound); Wang 2002 (Neuron 36:955-968, slow-NMDA reverberation attractor);
  Lo & Wang 2006 (Nat. Neurosci. 9:956-963, BG->SC commit threshold); Stine, Trautmann, Jeurissen & Shadlen
  2023 (Neuron, LIP-accumulate / SC-threshold); Douglas & Martin (canonical cortical microcircuit).
- Intrinsic-motivation thesis: Oudeyer & Kaplan learning-progress maximization (the noisy-TV cure) — the
  drive the CPU proxy validated and this realizes on spikes.
</content>
</invoke>
