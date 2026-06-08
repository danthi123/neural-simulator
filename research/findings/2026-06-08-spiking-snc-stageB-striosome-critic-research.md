# Stage B — striosome value critic for the spiking-SNc actor-critic: deep-research + catalog review

**Date:** 2026-06-08
**Type:** Deep-research / scoping review (READ-ONLY; no code changed, no GPU run). Produced to scope
the NEXT protected `sim/` edit before it is built, per the project's standing practice ("deep
research + catalog review FIRST at roadblocks and new directions").
**Owner standard (CLAUDE.md "BRAIN-BASED ONLY"; MEMORY.md "proper brain analogue"):** the value
prediction `V(s)` and the reward-prediction-error (RPE) are *cognition* and must be computed by
neurons/synapses, not host code. An honest negative (the neural critic underperforming the host
scaffold) is itself the scientific deliverable.
**Reads:** code from the clean worktree `E:\Documents\Projects\sim-derisk`; catalog from
`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`.

---

## 1. Executive summary + recommendation

**The Stage B question.** Stage A (already shipped) makes the dopamine broadcast *spiking*: the
`snc` pool's windowed firing rate encodes `delta = r − V`, and the new signed neuromodulator rule
`from_region_firing_signed` (the one protected Stage-A edit) turns that rate into the signed plasticity
signal the bridge already consumes. But the value `V` it subtracts is still the **host** running
reward average (`_V_scaffold = max(0, reward_ema_pre)` at `g11_bg_runner.py:5025`, fed as inhibitory
current at `:5029`). Stage B must replace that host scalar with a **neural value critic**: a
striosome/patch population that learns `V(s)` from the perceived state and projects GABAergic
(inhibitory) to the SNc, so the subtraction `r − V` happens at the SNc *membrane* via opposing
synaptic currents, and the critic is trained by the dopamine RPE the SNc itself broadcasts.

**The single most important finding of this review:** the project **already has a striosome
population and the canonical striosome→SNc GABAergic pathway wired** — they are built unconditionally
in the navigation runner. `build_bg_brain_regions` (called for the moving-goal path at
`g11_bg_runner.py:2998`) creates four per-action striosome pools `str_striosome_{N,E,S,W}` (D1-MSN
typed, GABAergic, `g11_bg_runner.py:764–774`) and the projection `str_striosome_X → snc`
(`:1327–1330`, GABAergic, `plastic=False`, weight 2.5). The biology Stage B wants — "striosomes are the
value critic projecting GABA to SNc dopamine cells" — is **anatomically present already** (added in the
2026-04-29 R3.11 catalog pass, citing Tepper & Lee PBR-160 ch 11 p 191). What is *missing* is purely
functional: (a) those striosomes are driven by **action-cortex** (`cortex_X`), not the perceived
state, so they encode an action-tied signal, not a learned state value; (b) the `str_striosome_X → snc`
weight is **frozen** (`plastic=False`) and (c) the host `_V_scaffold` term at `:5029` is still injected
*in addition*, so even if the striosomes did carry `V` it is double-counted.

**Recommendation (ranked in §6): build a dedicated single `striosome_value` region** driven by the
perceived state (the option the Stage-A design already sketches in §2.3), rather than re-purposing the
four per-action `str_striosome_*` pools. Reasons: (i) the existing pools are per-action `Q(s,a)`-shaped
and action-cortex-driven — re-pointing them to perceived-state input and merging them into one
state-value `V(s)` is *more* code churn and semantically muddier than adding one clean region; (ii) a
dedicated region cleanly isolates the anti-cheat assertion ("the critic reads only perceived-state
regions, never coordinates"); (iii) it leaves the existing per-action striosome→SNc/→SNr anatomy
(which serves the R3.11 limbic-gates-motor story) untouched, so Stage B is additive and reversible.
The existing pools then stand as a *second, cheaper* option (§6 Option B) and a useful sanity cross-check.

**The protected-`sim/` surface for Stage B is ZERO new protected edits.** Everything Stage B needs —
a new `BrainRegion`, plastic perceived-state→critic pathways, a GABAergic critic→SNc pathway that
*replaces* the host value term, and δ-training of the critic synapses — is realizable with the
**runner-side** region/pathway framework + the bridge machinery that already exists (per-region
inhibitory reversal potentials, the global eligibility trace, the `da_signal`-driven three-factor
update, `cp_d1_d2_sign` defaulting to +1 for non-D2 regions). I verified each of these against the
source (§4). The only `sim/` edit in the whole arc remains Stage A's `from_region_firing_signed` rule,
which is **already merged** (`sim/neuromodulators.py:774–817`). Stage B rides entirely on top.

**Cheap-first de-risk (§5):** extend the existing `snc_pavlovian_probe.py` with a **cue-shift** test
on a `striosome_value` critic, run under `SIM_BACKEND=numpy` (no GPU), *before* any nav run. Cue-shift
is the unique discriminator: a host global-EMA value provably cannot transfer the burst from US to CS
(the probe already says so in its docstring, `snc_pavlovian_probe.py:14–17`), so passing cue-shift is
positive proof the value became neural and state-dependent. The anti-cheat control is a **striosome
lesion**: zero the critic population (or its →SNc weight) and confirm the value subtraction vanishes —
proving `V` comes from striosome firing → GABA current, not a host formula in disguise.

**Honest uncertainty flagged up front.** Two things are biology-established and two are modeling
choices: (established) striosomes carry reward-value-correlated activity and project monosynaptically
to SNc DA cells (Yoshizawa-Ito-Doya 2018; Crittenden 2016); (established) the actor-critic anatomical
mapping SNc=δ / striosome=V(s) / matrix=actor (Houk-Adams-Barto 1995; Schultz 1998 Fig 9C; catalog
C.30). (Modeling choice) that the *whole* value is computed in striosomes — the ventral striatum/NAcc
is the classical "critic" locus and the literature explicitly debates whether the split is real
(§2.4). (Open challenge) a 2025 result argues the striosome-dopamine circuit signals *policy
information gain*, not a prediction error (§2.4) — which does not block Stage B but caveats the
"striosome = textbook TD critic" framing.

---

## 2. Biology validation — is "striosome = value critic projecting GABA to SNc" defensible?

### 2.1 The striosome/matrix compartmentalization of striatum
The striatum is not homogeneous: it is a mosaic of **striosomes (patches)** embedded in a surrounding
**matrix**, developmentally and neurochemically distinct (identified by µ-opioid receptor and calbindin
staining). Catalog **B.07** (`feature-catalog.md:505–519`) records this: *"patch (striosome) ↔ ventral
midbrain DA neurons (limbic) vs matrix ↔ thalamus/cortex (sensorimotor)… patch implicated in
habit/OCD; matrix in motor learning."* Crucially the catalog already flags the **output-level
alignment**: *"major input to SNc dopaminergic neurons arises from striatal patch (striosome)
compartment while GABAergic SNr neurons receive input from striatal matrix — the patch/matrix split
aligns with the SNc/SNr split at the output level"* (`:519`, citing PBR-160 ch 11 p 191, Tepper & Lee;
the project's own R3.11 source). This is the anatomical premise Stage B rests on.

### 2.2 The striosome→SNc direct projection (the relevant tract)
The canonical striatonigral projection from striosomes targets the SNc dopamine cells directly. The
anatomical substrate is the **"striosome–dendron bouquet"**: striosomal striatonigral axons form
bouquet-like arborizations that wrap clusters of tightly bundled DA-neuron dendrites descending into
the SNr (Crittenden, Tillberg, Riad, Shima, Gerfen, Curry, Housman, Nelson, Boyden & Graybiel, **PNAS
2016**, "Striosome–dendron bouquets highlight a unique striatonigral circuit targeting
dopamine-containing neurons", `pnas.org/doi/10.1073/pnas.1613337113`). Yoshizawa, Ito & Doya (**2018**,
"Reward-Predictive Neural Activities in Striatal Striosome Compartments", PMC5804148) report that
*"almost all striatal neurons directly projecting to midbrain dopaminergic neurons belong to striosome
compartments"* and that *"83.2% of Cre-expressing [striosomal] neurons were D1 medium spiny neurons,
projecting monosynaptically to dopaminergic neurons in the SNc."* So: a **monosynaptic, striosome→SNc-DA**
projection from D1-type MSNs is established. The project models exactly this (`str_striosome_X → snc`,
D1-MSN-typed, `g11_bg_runner.py:764–774, 1327–1330`).

**Is it GABAergic/inhibitory?** MSNs are GABAergic projection neurons (the dominant striatal output
phenotype), so the striosome→SNc projection is inhibitory. *Honest caveat:* Yoshizawa-2018 itself does
**not** directly demonstrate the GABAergic transmitter at that synapse (it characterizes value-coding +
the monosynaptic projection); the GABAergic-ness is inferred from the well-established MSN class. The
project encodes this inference (the striosome region is `IZH2007_STRIATAL_MSN_D1` with
`exc_fraction=0.05`, i.e. essentially all-GABAergic, and `syn_reversal_potential_i_override=-60.0`,
`:767–772`). **Functional sign at the SNc — the subtle point (catalog B.07 supplemental,
`feature-catalog.md:434`):** the SNc lacks the KCC2 chloride exporter, so its GABA_A reversal is
depolarized (~−55 mV, encoded as `syn_reversal_potential_i_override=-55.0` on the `snc` region,
`g11_bg_runner.py:859`). GABA onto SNc is therefore *depolarizing near rest but shunting/hyperpolarizing
near threshold*. The bridge computes the inhibitory current as `g_i·(V − E_inh)`
(`sim/bridge.py:5279–5282`); when a DA neuron is depolarized toward firing (`V > −55 mV`),
`(V − E_inh) > 0`, so striosomal GABA **reduces** firing — the value subtraction works in the regime
that matters (when the SNc is being driven to burst). This is biologically faithful (shunting
inhibition gating the burst) but means the critic→SNc gain interacts nonlinearly with the SNc operating
point — a calibration item flagged in §5/§6, not a blocker.

### 2.3 Do striosomes encode value / reward expectation?
Yes, with direct electrophysiological evidence:
- **Yoshizawa, Ito & Doya 2018** (PMC5804148): striosomal neurons in dorsomedial striatum showed
  cue-evoked predictive activity *"proportional to the expected reward"*, with **positive** value
  correlation (late-stage ~93% of striosomal, 100% of control reward-predictive cells had positive
  coefficients). Population decoding showed striosomes represented expected reward **more robustly than
  surrounding tissue** in late learning. The signal was **learning-stage-specific** (it emerged with
  learning and faded with overtraining) — consistent with a value/prediction signal, not a raw sensory
  or motor one.
- **Graybiel-lab single-unit work** (Friedman et al.; the Crittenden & Graybiel reviews) places
  striosomes in the value/cost–benefit evaluation and reinforcement circuitry, with the striosome→SNc
  arm positioned to shape the DA teaching signal.
- The **GABAergic-VTA analogue** is informative for the *mechanism* Stage B uses: in VTA, the GABAergic
  interneurons *"encode expectation about rewards… represent only the predicted reward"*, while the DA
  neurons compute the difference (observed − predicted) (Cohen et al. 2012, surfaced in the search).
  That is precisely the Stage-B decomposition: an inhibitory population carrying the *prediction* `V`,
  subtracted at the DA membrane to yield the *error* `δ`. The striosome→SNc projection is the dorsal
  homologue of that inhibitory-prediction arm.

### 2.4 The actor-critic mapping, and the honest "where is V" debate
**The chosen mapping (catalog C.30, `feature-catalog.md:592–599`):** *"VTA/SNc DA = critic δ output;
striosome-patch (limbic striatum) = critic state-value; striatal matrix (sensorimotor striatum) = actor
preferences; corticostriatal synapses on matrix = actor weights modified by δ"* (Schultz 1998 Fig 9C;
Houk-Adams-Barto 1995; Barto 1995). C.30's sim-status is the exact gap Stage B closes: *"partial — actor
implemented, critic missing… there is no separable population that outputs a learned V(s), and
consequently no bootstrapping."* Catalog **O.20** (`:525–531`) frames the same gap in RL terms: the
project does policy *improvement* (DA-gated STDP every step) but has no policy *evaluation* (no critic),
so *"it is policy-improvement-only, which can converge to local optima that an evaluator would have moved
past."* Adding the striosome critic makes it a full generalized-policy-iteration loop.

**Important nuance C.30 itself raises (`:595`):** the project's D1/D2 split is *already* a partial
actor-critic of a different flavor — *"D1 = Go = positive-affect actor, D2 = NoGo = negative-affect
actor… a two-actor, no-critic architecture, more closely matching Frank's BG models than canonical
actor-critic."* So Stage B is **adding a canonical state-value critic on top of a Frank-style two-actor
matrix** — a defensible hybrid, but worth stating plainly so the result isn't overclaimed as "the"
unique BG-RL architecture.

**Competing critic-locus accounts (be honest about these):**
1. **Ventral striatum / nucleus accumbens as the classical critic.** The mainstream actor-critic-of-BG
   literature (Joel-Niv-Ruppin 2002; O'Doherty 2004) localizes the *critic* to **ventral** striatum /
   NAcc, with **dorsal** striatum as the actor. The review "Ventral striatum: a critical look at models
   of learning and evaluation" (van der Meer & Redish-adjacent, PMC3134536) is explicitly skeptical that
   the clean dorsal-actor/ventral-critic split holds. Recent large-scale recordings find prediction-error
   coding spanning **both** dorsal and ventral striatum, *"challenging the functional separation of the
   striatum into a dorsal 'actor' and a ventral 'critic'."* **Why striosomes are nonetheless the chosen
   locus here:** the project's navigation circuit is dorsal/sensorimotor (it has no NAcc region), and the
   *dorsal* compartment that carries the value-to-DA signal **is** the striosome (Yoshizawa 2018; the
   striosome→SNc arm). So within the modeled anatomy, striosomes are the correct dorsal value-to-DA
   conduit even if a full brain would also have a ventral critic. This should be documented as a scoping
   choice, not asserted as the only critic.
2. **"Information gain, not prediction error."** A 2025 result (Jeong/Namboodiri-adjacent; "Striosome–
   dopamine circuit signals information gain, not prediction error", researchgate 393865764) argues the
   striosome-dopamine arm specifically carries a *policy information-gain* signal rather than a scalar
   RPE. This **does not block Stage B** (the project's δ is still the SNc output; the striosome supplies
   the *prediction* term, not the error), but it caveats the strong claim "the striosome IS the textbook
   TD value." Recommend: build the value-critic version (it is the cleanest, best-supported first step),
   and note information-gain as a future alternative-hypothesis probe.

**Verdict:** "striosome = the value critic projecting GABA to SNc DA neurons" is a **defensible,
literature-grounded modeling choice** — the anatomy (monosynaptic striosome→SNc-DA, GABAergic-by-MSN-
class), the physiology (positive reward-value coding, Yoshizawa 2018), and the computational mapping
(C.30 / Houk-Adams-Barto) all support it. The honest qualifications are: GABAergic transmitter inferred
not directly shown in the value paper; the ventral-striatum critic is the classical alternative and the
dorsal/ventral split is debated; and a recent account reframes the striosome-DA signal as information
gain. None of these blocks the build; all should be written into the result's "modeling choices" table.

---

## 3. How V(s) is learned, and what project machinery is reusable as-is

### 3.1 The learning rule: three-factor δ-modulated plasticity on cortico-striosomal synapses
The critic learns by the **same dopamine δ it helps generate** — the textbook actor-critic bootstrap
(Sutton-Barto Ch 11; Houk-Adams-Barto 1995). Concretely, the perceived-state→striosome synapses are
plastic and reward-modulated; their weight change is the three-factor product:

```
Δw(state→striosome) = lr · da_signal · eligibility(pre=perceived-state spikes, post=striosome spikes)
```

When the perceived state reliably precedes reward, early `da_signal = δ` is positive → LTP raises the
state→value weights → `V` rises → subsequent `δ` shrinks toward zero. That is the TD fixed point
`V → E[r]`. The critic and the actor (cortico-striatal matrix) consume the *same* `δ`, which is exactly
the C.30 "shared scalar TD error" property. Catalog C.28 (`feature-catalog.md:574–579`) and C.31
(bootstrapping) name this; the minimal first version is Rescorla-Wagner (`δ = r − V`), the deeper
version is full bootstrap (`δ = r + γV(s′) − V(s)`).

### 3.2 What the project ALREADY has (reusable as-is) — verified file:line
The reason Stage B needs **no protected `sim/` edit** is that the entire three-factor pipeline already
runs over arbitrary plastic region pathways:

| Needed for the critic | Already exists | file:line (sim-derisk) |
|---|---|---|
| A signed dopamine teaching signal `δ` read from SNc firing | `from_region_firing_signed` rule (the merged Stage-A edit) → `da_conc`; bridge reads `da_signal = da_conc − baseline` (signed) | `sim/neuromodulators.py:774–817`; `sim/bridge.py:5894–5904` |
| A global eligibility trace over ALL synapses (so critic synapses get it automatically) | `cp_eligibility_trace = cp.zeros(capacity)`, accumulated for every plastic synapse | `sim/bridge.py:616`, allocation `:2242` |
| The three-factor update applied to every reward-eligible synapse | `weight_updates = effective_reward_lr · effective_signal · cp_eligibility_trace` | `sim/bridge.py:5952` |
| Correct sign for the critic (V rises on +RPE) | `cp_d1_d2_sign` = +1 everywhere except `str_D2_*` post-neurons; a `striosome_value` region is not `str_D2_*`, so default +1 | `sim/bridge.py:2163–2178`, applied `:5964` |
| Per-pathway plasticity gating to stage/freeze the critic | `RegionPathway.plasticity_gate` + `bridge.set_plasticity_gate(name, value)` → `cp_plasticity_rate_gain` | `sim/regions.py:201–218`; applied `sim/bridge.py:5958–5959` |
| Per-region inhibitory reversal (so critic→SNc GABA subtracts at the membrane) | `BrainRegion.syn_reversal_potential_i_override`; bridge builds `cp_syn_reversal_potential_i_per_neuron`; inhibitory current `g_i·(V−E_inh)` | `sim/regions.py:98`; `sim/bridge.py:1045–1062, 5279–5282` |
| A declarative region + cross-region pathway with chosen density/weight/plasticity | `BrainRegion`, `RegionPathway`, wiring via `build_wiring_plan`/`inject_explicit_wiring` | `sim/regions.py:32, 171, 461–506` |
| **An existing striosome population + the canonical striosome→SNc GABAergic projection** | `str_striosome_{N,E,S,W}` (D1-MSN, GABAergic) + `str_striosome_X → snc` | `g11_bg_runner.py:764–774, 1327–1330` |
| The host value scaffold to swap out, and the swap-point | `_V_scaffold = max(0, reward_ema_pre)`; `I_snc = … − snc_value_gain·_V_scaffold` written to `cp_external_input_current[snc]` | `g11_bg_runner.py:5025, 5026–5033` |

**Net:** the critic's *learning* is entirely covered by existing machinery — a plastic, reward-eligible
`perceived-state → striosome_value` pathway is trained by the SNc-derived `δ` "for free" via the line
`:5952` update, with the correct +1 sign by default. **Nothing must be added to `sim/` for the critic
to learn.** This was the load-bearing claim of the Stage-A design §2.3 and it checks out against source.

### 3.3 What must be ADDED (all runner-side data/config)
Only the *data* of the critic is new: the region object, its afferent pathways, the inhibitory
critic→SNc pathway, the gate plumbing, the anti-cheat assertion, and the removal of the now-redundant
host value injection. Enumerated in §4.

---

## 4. The minimal protected surface — every edit, classified

**Headline: Stage B requires ZERO new protected `sim/` edits.** The only `sim/` change in the entire
spiking-SNc arc is Stage A's `from_region_firing_signed` rule, already merged
(`sim/neuromodulators.py:774–817`). Stage B is **runner-side / data / config only**, verified below.

### 4.1 Runner-side edits (unprotected) — `research/runners/g11_bg_runner.py`

**(E1) New CLI flag + builder param `--enable-neural-critic`.** Master switch for Stage B. Registered
beside `--spiking-snc` (`:5961`) and threaded into `build_bg_brain_regions`. Runner-only.
*Classification:* unprotected.

**(E2) The `striosome_value` region (data).** Append one `BrainRegion` in `build_bg_brain_regions`
when `--enable-neural-critic`:
```python
BrainRegion(name="striosome_value", n_neurons≈80,
            exc_fraction≈0.05,                 # GABAergic projection neurons (MSN), few glut spillover
            internal_density≈0.05,             # mild recurrent smoothing
            izh_neuron_type=IZH2007_STRIATAL_MSN_D1.name,   # striosomes are D1-MSN-rich
            syn_reversal_potential_i_override=-60.0)        # MSN GABA_A reversal (matches str_striosome_*)
```
*Classification:* unprotected (pure data; mirrors the existing `str_striosome_*` region recipe at
`:764–774`). *Note on sizing:* the design suggested `exc_fraction≈0.8`; that is wrong for a striosome —
the existing striosome pools use `exc_fraction=0.05` (GABAergic). For the **critic→SNc projection to be
inhibitory, the projecting neurons must be inhibitory**, so use `exc_fraction≈0.05` (or carve a
dedicated GABAergic sub-population). This is the one place the design's draft parameters need correction;
copy the shipped `str_striosome_*` values.

**(E3) Afferent pathways: perceived state → critic (plastic, data).** Append `RegionPathway`(s),
`plastic=True`, `plasticity_gate="value_input"`, with the source chosen by what perception is on
(mirrors the cerebellum `mossy_state` union at `:1775–1786`):
- `--enable-visual-cortex` → `cortex_it → striosome_value` (ventral object code; `cortex_it` exists);
- else `--enable-learned-perception` → `sensory → striosome_value`;
- else perception-arc default → `ppc_goal_input → striosome_value` and/or
  `sensor_place_readout → striosome_value` (the perceived goal-vector / place code).
*Classification:* unprotected (data). These synapses ride the existing eligibility + δ update (`:5952`)
and train `V` with no new mechanism (§3.2).

**(E4) The critic→SNc inhibitory pathway that REPLACES the host value term (data).** Append
```python
RegionPathway(from_region="striosome_value", to_region="snc",
              density≈0.4, weight_mean≈2.5, weight_jitter=0.2, plastic=False)
```
This is **structurally identical to the existing `str_striosome_X → snc` pathway** (`:1327–1330`) —
proof the framework already supports exactly this GABAergic critic→SNc projection with no kernel work.
Because `striosome_value` neurons are inhibitory (E2) and the SNc carries `E_inh=−55 mV`, the resulting
`g_i·(V−E_inh)` current subtracts from the SNc drive when the SNc is depolarized (§2.2). With this
present, **the runner DROPS the host value injection**: the `− snc_value_gain·_V_scaffold` term at
`:5029` is removed (the inhibition is now synaptic). The SNc soma then integrates
`I_tonic + k_r·max(0,r) − (synaptic inhibition from striosome_value)` and fires `δ = r − V` — the
maximally brain-based form (the subtraction is at the membrane, no host reads `V`).
*Classification:* unprotected (data + deleting one host arithmetic line).

**(E5) Gate plumbing + δ-training stays on (config).** Register `"value_input"` in the runner's gate
set (mirror the existing `"corticostriatal"` handling at `:3739`) so the critic can be staged/frozen via
`bridge.set_plasticity_gate("value_input", …)`. Keep the critic afferents plastic during the reward
window so the existing `:5952` update trains them. *Classification:* unprotected (config).

**(E6) Anti-cheat assertion (code).** In `build_bg_brain_regions`, assert the `striosome_value` afferent
set contains only perceived-state region names — `cortex_it`, `sensory`, `ppc_goal_input`,
`sensor_place_readout` — and **never** `goal_cells`/`ppc_goal_input`-as-raw-coords, and that `V` is never
seeded from `(gx,gy)`/distance. (With N5 perceived reward already in place, the whole RPE loop is then
coordinate-free.) *Classification:* unprotected (code).

**(E7) Logging (config).** Surface `V` (the `striosome_value` windowed rate, read with the same
accumulator as the SNc rate at `:5053–5056`), `delta`, and the SNc rate in the run JSON for the probe +
webapp. *Classification:* unprotected.

### 4.2 Why NO protected `sim/` edit is required (the explicit check)
The bridge already (a) injects per-region current via `cp_external_input_current`
(`:5031–5033`); (b) builds the inhibitory-reversal array and computes `g_i·(V−E_inh)` honoring per-region
overrides (`:1045–1062, 5279–5282`); (c) maintains a global eligibility trace for all plastic synapses
(`:616`); (d) consumes the signed `da_signal` and applies the three-factor update with the `+1` default
sign to every reward-eligible synapse (`:5894–5904, 5952, 5963–5964`); (e) supports declarative
regions/pathways with plasticity gates (`sim/regions.py`). A new inhibitory region projecting to the SNc,
trained by δ, is **entirely expressible in that vocabulary** — and the existing `str_striosome_X → snc`
pathway is the existence proof. **There is no new kernel, no new `cp_*` array, no new bridge branch.**

### 4.3 If the owner instead chooses to REUSE the existing striosome pools (Option B in §6)
That path also needs **no protected edit**, but it requires *changing existing data* rather than adding:
re-point `cortex_X → str_striosome_X` afferents to a perceived-state source (or add perceived-state
afferents), set `str_striosome_X → snc` to participate in the value subtraction (and remove the host
`_V_scaffold` term), and reconcile the per-action structure (4 pools) with a single state-value (sum
their →SNc inhibition). It is *less* additive (mutates the shipped R3.11 anatomy used by other
behaviors) and semantically `Q(s,a)`-shaped, which is why it is ranked second.

---

## 5. Cheap-first de-risk + anti-cheat

### 5.1 The probe (reuse `snc_pavlovian_probe.py`, extend with cue-shift + the critic)
The Stage-A harness `research/runners/snc_pavlovian_probe.py` already builds a minimal `snc` bridge +
the signed `dopamine` modulator and runs the **omission-dip** falsifier under `SIM_BACKEND=numpy` (CPU,
no GPU). Stage B extends it minimally:
1. Add a `striosome_value` region + a `cue` (CS) input region + `cue → striosome_value` (plastic,
   `plasticity_gate="value_input"`) + `striosome_value → snc` (inhibitory, the value drive) to the
   probe's `_build_snc_bridge` (it already constructs `BrainRegion`/`RegionPathway` lists).
2. Enable the δ-training path in the probe (the harness currently sets `enable_reward_modulation=False`
   at `:77`; for Stage B turn it on so the cortico-striosome synapses learn from `da_signal`).
3. Drive the Schultz 2-cue Pavlovian schedule the probe already scaffolds (CS → delay → US into the SNc
   reward afferent).

### 5.2 PASS / FAIL criteria
- **(i) CUE-SHIFT (the unique Stage-B discriminator).** Time-lock the SNc burst (windowed rate above
  tonic) to **US** vs **CS** across training.
  **PASS:** early trials burst at US, not CS; late trials (after the critic learns) burst **shifts to
  CS** and the US burst **shrinks toward zero**. Quantitative gate (from the design §4.1):
  `US-burst(late) < 0.5 × US-burst(early)` **AND** `CS-burst(late) > 2 × CS-burst(early)`, sign-
  consistent across **≥3 seeds**.
  **Why it is the discriminator:** cue-shift requires the value to be **state-dependent** (the CS state
  must acquire value). A host global-EMA value *cannot* produce it — the probe's own docstring states
  this (`snc_pavlovian_probe.py:14–17`: cue-shift is OUT OF SCOPE for Stage A's host-EMA value).
  Therefore **passing cue-shift is positive proof the value became neural and state-dependent** — the
  central Stage-B claim. (Schultz 1998 §TD-learning; Hollerman-Schultz 1998 graded cue-shift; catalog
  C.30 acceptance metric (a), `feature-catalog.md:599`; C.22 supplemental HS98 criterion, `:918`.)
- **(ii) OMISSION DIP (regression guard, already passing at Stage A).** On a probe trial, withhold the
  expected US after the CS. **PASS:** SNc rate dips below tonic at the expected-reward time
  (`rate(omission, expected-US-window) < tonic_rate − margin`, ≥3 seeds). Stage B must not lose this.
- **(iii) Monotone-in-δ calibration smoke (`--snc-probe`, `:301`).** The existing sweep confirms the SNc
  windowed rate is monotone in `delta = r − V` (burst on +RPE, tonic at 0, dip on −RPE) **before** the
  gains are trusted. For Stage B, additionally verify the *neural* `V` (striosome rate) tracks expected
  reward monotonically.

### 5.3 Anti-cheat — proving V is computed by striosome NEURONS, not a host formula
Three controls, all in the probe (no GPU, no nav run):
1. **Provenance assertion.** Under `--enable-neural-critic`, assert the `_V_scaffold` host term at
   `:5029` is **removed** and the SNc inhibition arrives *only* via the `striosome_value → snc` synaptic
   current. Grep-assert no host `V`/`reward_ema` value reaches the SNc drive. (Mirrors the Stage-A
   anti-cheat checklist item, design §4.3.1.)
2. **Lesion test (the decisive one).** Zero the `striosome_value` population (or set the
   `striosome_value → snc` weight to 0) and re-run: the value subtraction must **vanish** — the SNc
   should burst to *every* reward regardless of prediction (no omission dip, no cue-shift). If `V` were a
   host formula in disguise, lesioning the neurons would leave the subtraction intact; it must not. This
   is the clean "the value comes from striosome firing → GABA current" proof.
3. **Coordinate-freedom assertion.** Assert the critic's afferents are perceived-state regions only
   (E6); combined with N5's perceived reward, the entire RPE loop references no coordinate.

### 5.4 Nav-score regression gate (necessary, not sufficient)
Flagship multi-goal-deterministic 6-seed (the A+E+G v2.5 stack) with `--spiking-snc --enable-neural-
critic`: summed reward **≥ Stage A** (which is ≥ the raw-reward `--rpe-dopamine` baseline). A correct
critic should match or beat actor-only (C.30/O.20 predict the evaluator escapes local optima). **An
honest negative here is a valid deliverable** (it would map a limit of the neural critic vs the host
scaffold). The Pavlovian cue-shift (5.2) is what proves the *mechanism* is the real biology; the nav
score only proves it didn't break navigation.

---

## 6. Ranked options + recommendation + open questions

### Option A (RECOMMENDED) — a dedicated `striosome_value` state-value critic
A single new `striosome_value` region (GABAergic, MSN-typed) driven by the **perceived state**,
projecting GABA to the SNc, trained by δ; host value term removed.
- **Biological fidelity:** high. Matches C.30 (striosome = state-value V(s)) and Yoshizawa-2018
  (striosomes carry positive reward-value, project monosynaptically to SNc). A *single* state-value
  critic is the canonical actor-critic form (vs the per-action `Q(s,a)` the existing pools imply).
- **Implementation cost:** low. One region + 1–2 plastic afferent pathways + one inhibitory →SNc pathway
  + gate plumbing + delete one host line. **Zero protected `sim/` edits.** ~2–3 days incl. the cue-shift
  probe extension and nav 6-seed (matches the design estimate).
- **Risk:** moderate, bounded. Chief risks: (1) the critic→SNc inhibitory **gain calibration** against
  the depolarized SNc reversal (the value must cancel the reward drive at `r=V` without over-shunting) —
  mitigated by the host-readout fallback as a *calibration crutch* (design §2.3) and the `--snc-probe`
  sweep; (2) **TD-bootstrap sign/stability** — mitigated because `cp_d1_d2_sign` defaults +1 for the
  non-D2 critic (verified `:2163–2178`) and the cue-shift test *is* the sign check; (3) **rate-coding
  noise** on a small striosome pool — mitigated by `n≈80`, the 200 ms DA-conc EMA, and a wider readout
  window (Potjans-Diesmann-Morrison 2011: an imperfect spiking RPE still drives TD learning).
- **Why recommended:** maximally additive (touches no existing behavior), cleanest anti-cheat boundary,
  canonical state-value semantics, and the cheapest correct first step.

### Option B — reuse the existing `str_striosome_{N,E,S,W}` pools as the critic
Re-point/augment their afferents to perceived state, activate their →SNc inhibition as the value term,
remove the host scaffold, reconcile 4 per-action pools into one value.
- **Biological fidelity:** medium-high but *different shape*. The existing pools are per-action and the
  →SNc anatomy is already wired (a plus). But per-action striosomes driven by action-cortex are closer
  to **action-value `Q(s,a)`** than the canonical single state-value `V(s)`; merging them muddies the
  actor-critic story and overlaps the R3.11 "limbic gates motor" function those pools were added for.
- **Implementation cost:** medium. **Zero protected edits**, but it *mutates shipped data* (the R3.11
  pathways used by other behaviors), so it is less reversible and needs regression care on non-Stage-B
  runs.
- **Risk:** higher coupling (changing data other behaviors depend on); the per-action→state-value
  reconciliation is an extra design question.
- **When this wins:** if the owner specifically wants **action-value `Q(s,a)`** (a two-actor + per-action
  critic, closer to the Frank-style hybrid C.30 notes), reusing these pools is the natural substrate.
  Otherwise Option A is cleaner.

### Option C — keep an algorithmic critic but make it a *neural* readout (the design's host-readout fallback)
Read the `striosome_value` population's windowed firing rate and inject `I_value = k_v·rate` as
hyperpolarizing host current (value computed by neurons, subtraction done in host arithmetic).
- **Biological fidelity:** lower — the *subtraction* is host code (a weaker claim under the BRAIN-BASED
  standard), though the *value* is neural.
- **Implementation cost:** lowest (no inhibitory-gain calibration).
- **Use:** **only as a calibration crutch / de-risk waypoint**, not the deliverable. The design itself
  says "prefer the inhibitory projection; document the fallback only as a calibration crutch" (§2.3). If
  Option A's inhibitory gain proves hard to tune, ship C transiently and label it honestly as a partial
  (value-neural, subtraction-host) result, then convert to A.

**Recommendation:** **Option A.** Build the dedicated `striosome_value` state-value critic, validate with
the cue-shift + omission-dip + lesion controls on the extended `snc_pavlovian_probe.py` (CPU, no GPU)
*before* any nav run, then gate on the flagship 6-seed nav score. Keep Option C in the back pocket as a
calibration crutch. Keep Option B documented as the action-value alternative if the owner wants `Q(s,a)`.

**What would change the recommendation:**
- If the owner wants **action-value `Q(s,a)`** semantics (per-action critic, Frank-style), switch to
  **Option B** (reuse the existing per-action striosome pools).
- If the inhibitory critic→SNc **gain calibration** turns out intractable on the small SNc pool, fall
  back to **Option C** transiently (neural value, host subtraction) and label the partial honestly.
- If the **"information-gain not RPE"** account (Jeong/Namboodiri 2025, §2.4) is something the owner
  wants to test, that is a *follow-on probe* on top of Option A, not a change to the build.

### Open questions (for the owner to weigh before building)
1. **State-value V(s) vs action-value Q(s,a).** Option A = single V(s) (canonical actor-critic); Option B
   = per-action Q (matches the existing pools + the Frank two-actor framing C.30 already notes). Which
   semantics does the project want? (Recommendation assumes V(s).)
2. **Bootstrap depth.** Rescorla-Wagner (`δ = r − V`, no γ) is the cheap first step and is all the
   Pavlovian probe needs. Full TD bootstrap (`δ = r + γV(s′) − V(s)`, catalog C.28) requires a `V(s′)`
   read at the next state and is a deeper (later) increment. Start with R-W.
3. **Inhibitory gain learning.** Start `striosome_value → snc` `plastic=False` (fixed gain like the
   existing striosome→SNc). If the cancellation at `r=V` is too sensitive, make that weight plastic too
   (a second learning loop) — but only if needed; it adds a stability question.
4. **Depolarized-SNc-reversal interaction.** The −55 mV SNc GABA reversal means striosomal inhibition is
   shunting/depolarizing near rest and only clearly hyperpolarizing near burst threshold (§2.2). Confirm
   in the `--snc-probe` sweep that the value subtraction is monotone across the SNc operating range used
   in nav; if not, consider tonic-drive recalibration so the SNc sits where GABA is hyperpolarizing.
5. **Existing per-action striosome→SNc pathway interaction.** Under Option A, the four existing
   `str_striosome_X → snc` inhibitory pathways (`:1327–1330`, driven by action-cortex) are *still present*
   and inject action-correlated inhibition into the SNc. Decide whether to (a) leave them (they add an
   action-tied DA modulation that may be biologically reasonable but confounds the clean δ), (b) zero
   their weight under `--enable-neural-critic`, or (c) fold them into the critic. Cleanest for the
   isolated cue-shift test is (b); document the choice.

---

## 7. Sources

### Project code (verified file:line, sim-derisk clean worktree)
- Stage-A signed DA rule (the merged protected edit): `sim/neuromodulators.py:774–817`
  (`from_region_firing_signed`); the one-sided template it mirrors `:736–772` (`from_region_firing`);
  `from_reward` baseline-subtraction `:647–653`.
- Bridge DA consumption + three-factor update: `sim/bridge.py:5894–5904` (`da_signal = da_conc −
  baseline`, signed), `:5952` (`Δw = lr · effective_signal · eligibility`), `:5963–5964` (`cp_d1_d2_sign`),
  `:5958–5959` (`cp_plasticity_rate_gain` gate).
- Eligibility trace (global, all synapses): `sim/bridge.py:616`, `:2242`.
- `cp_d1_d2_sign` construction (+1 default, −1 only for `str_D2_*` post): `sim/bridge.py:2163–2178`.
- Per-region inhibitory reversal + inhibitory current: `sim/regions.py:98`
  (`syn_reversal_potential_i_override`); `sim/bridge.py:1045–1062` (array build), `:5279–5282`
  (`g_i·(V−E_inh)` via `fused_conductance_decay_and_current`).
- Region/pathway framework: `sim/regions.py:32` (BrainRegion), `:171–248` (RegionPathway: `plastic`,
  `plasticity_gate`, `transmission_gate`), `:461–612` (wiring plan / `_build_pathway`).
- **Existing striosome population + striosome→SNc GABAergic pathway:**
  `research/runners/g11_bg_runner.py:136` (`n_str_striosome_per_action=8`), `:755–774` (region build,
  D1-MSN, `exc_fraction=0.05`, `E_inh=−60`), `:1311–1334` (`cortex_X→str_striosome_X` plastic,
  `str_striosome_X→snc` GABAergic `plastic=False`, `str_striosome_X→gpi_X`); R3.11 note `:755–763`.
- The silent SNc pool: `g11_bg_runner.py:851–860` (`name="snc"`, `IZH2007_DOPAMINE`, `E_inh=−55`).
- Stage-A swap-point (the host value scaffold to replace): `g11_bg_runner.py:5024–5033`
  (`_V_scaffold = max(0, reward_ema_pre)`; `I_snc = I_tonic + k_r·max(0,r) − k_v·_V_scaffold` → written
  to `cp_external_input_current[snc]`); SNc rate accumulator `:5046–5058`; `current_reward_signal=0`
  under spiking-SNc `:5059`.
- Stage-A DA modulator registration + precedence: `g11_bg_runner.py:3223–3284` (`--spiking-snc` owns the
  `dopamine` modulator via `from_region_firing_signed`; mutually exclusive with compartmentalized DA;
  tonic-DA skipped); CLI flags + defaults `:5961–5985` (`--spiking-snc`, `--snc-tonic-pa 220`,
  `--snc-reward-gain 400`, `--snc-value-gain 400`); builder params `:2590–2592`.
- Stage-A Pavlovian probe: `research/runners/snc_pavlovian_probe.py` (omission-dip falsifier `:193–287`;
  cue-shift OUT-OF-SCOPE note `:14–17`; `--snc-probe` calibration `:152–190, 301`; `enable_reward_
  modulation=False` to flip on for Stage B `:77`).
- Stage A+B design doc: `docs/plans/2026-06-08-spiking-snc-actor-critic-design.md` (Stage B §2.3, §3,
  §4); N9 research: `research/findings/2026-06-08-remaining-nav-cheats-full-biologization-research.md` §N9.

### Project feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **C.30** Actor-critic mapping (SNc=δ / striosome-patch=V(s) / matrix=actor; "actor implemented, critic
  missing"; D1/D2 = two-actor-no-critic / Frank-style; acceptance = cue-shift + omission dip): `:592–599`.
- **C.28** TD error `δ = r + γV(s′) − V(s)` ("project's `current_reward_signal = r(t)`… cannot produce
  cue-shift… requires a critic population"): `:574–579`. **C.29** eligibility traces `:583–590`.
- **C.22** Schultz RPE + **HS98 cue-shift/omission validation criterion** ("reproducing it requires the
  value-function critic of an actor-critic architecture"): `:912–918`.
- **O.20** Generalized Policy Iteration (actor-only → "policy-improvement-only, can converge to local
  optima a critic would have moved past"): `:525–531`. **O.21** average-reward R-learning (the R̄ scaffold)
  `:447–448`.
- **B.07** Striatal patch/matrix compartments (patch↔ventral-midbrain-DA/limbic; striosome→SNc canonical;
  striosome→SNr correction; "major input to SNc DA arises from striatal patch/striosome"): `:505–519`.
- GABA reversal supplements (MSN E_GABA ≈ −60 mV depolarizing-but-shunting; SNc DA lacks KCC2 → ECl
  ≈ −55 mV, GABA depolarizing/excitatory at rest): `:343, 434`.

### Peer-reviewed literature (verified this session where fetched)
- **Yoshizawa T., Ito M., Doya K. (2018)** "Reward-Predictive Neural Activities in Striatal Striosome
  Compartments", eNeuro / PMC5804148. Striosomal neurons encode expected reward (positive correlation,
  late-stage ~93–100% positive coefficients); ~83% are D1 MSNs projecting **monosynaptically** to SNc DA;
  learning-stage-specific. https://pmc.ncbi.nlm.nih.gov/articles/PMC5804148/  *(GABAergic transmitter at
  the striosome→SNc synapse is inferred from MSN class, not directly shown in this paper.)*
- **Crittenden J.R., …, Graybiel A.M. (2016)** "Striosome–dendron bouquets highlight a unique
  striatonigral circuit targeting dopamine-containing neurons", *PNAS* 113:11318.
  https://www.pnas.org/doi/10.1073/pnas.1613337113 — anatomical substrate of the striosome→SNc-DA
  projection (bouquet arborizations wrapping clustered DA dendrites).
- **Houk J.C., Adams J.L., Barto A.G. (1995)** "A model of how the basal ganglia generate and use neural
  signals that predict reinforcement" — striosome = critic V(s); SNc = δ; matrix = actor (the C.30
  mapping). *(Catalog C.30 primary source; not re-fetched.)*
- **Schultz W. (1998)** "Predictive reward signal of dopamine neurons", *J. Neurophysiol.* 80:1
  (δ = r − P; cue-shift; omission dip; Fig 9C TD implementation).
  https://journals.physiology.org/doi/full/10.1152/jn.1998.80.1.1
- **Hollerman J.R., Schultz W. (1998)** "Dopamine neurons report an error in the temporal prediction of
  reward during learning", *Nat. Neurosci.* 1:304 — graded cue-shift + omission dip (catalog C.22
  validation criterion).
- **Joel D., Niv Y., Ruppin E. (2002)** "Actor–critic models of the basal ganglia", *Neural Networks*
  15:535 — the standard actor-critic-of-BG review; classical critic = ventral striatum/NAcc (the
  competing-locus account). https://www.sciencedirect.com/science/article/abs/pii/S0893608002000473
- **van der Meer M.A.A. & Redish A.D. (and review)** "Ventral striatum: a critical look at models of
  learning and evaluation", *Curr. Opin. Neurobiol.* (PMC3134536) — skepticism that the dorsal-actor /
  ventral-critic split is clean; recent recordings find RPE across dorsal+ventral striatum.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3134536/
- **Cohen J.Y., Haesler S., Vong L., Lowell B.B., Uchida N. (2012)** "Neuron-type-specific signals for
  reward and punishment in the VTA", *Nature* 482:85 — VTA GABAergic neurons encode the *expected*
  reward (the prediction), DA encodes the difference — the inhibitory-prediction-arm mechanism Stage B
  mirrors. *(Surfaced via search; not re-fetched.)*
- **Jeong H., …, Namboodiri V.M.K. (2025)** "Striosome–dopamine circuit signals information gain, not
  prediction error" (preprint, researchgate 393865764) — alternative-hypothesis caveat to "striosome =
  textbook TD value"; does not block Stage B. https://www.researchgate.net/publication/393865764
- **Frémaux N., Sprekeler H., Gerstner W. (2013)** "Reinforcement Learning Using a Continuous Time
  Actor-Critic Framework with Spiking Neurons", *PLoS Comput. Biol.* 9(4):e1003024 — canonical spiking
  actor-critic (spiking critic estimates V; TD error modulates reward-STDP).
  https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003024
- **Potjans W., Diesmann M., Morrison A. (2011)** "An imperfect dopaminergic error signal can drive
  temporal-difference learning", *Front. Comput. Neurosci.* (PMC3093351) — de-risks the small-pool
  rate-coding noise. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3093351/
- **Sutton R.S. & Barto A.G.** *Reinforcement Learning* (2nd ed.) — Ch 6 (TD), Ch 11 (actor-critic,
  average-reward R-learning), Ch 15 (GPI).
- **Kandel et al.** *Principles of Neural Science* 6e — Ch 43 (dopamine/reward), Ch 16 of PBR-160
  (Tepper & Lee, striosome→SNc), as cited in the catalog above.

---

**Deliverable path:** `E:\Documents\Projects\sim\research\findings\2026-06-08-spiking-snc-stageB-striosome-critic-research.md`
