# Spiking substantia-nigra (SNc) actor-critic — replace the host scalar dopamine

**Date:** 2026-06-08
**Type:** DESIGN (READ-ONLY produced; no code edited, no GPU run)
**Owner standard (CLAUDE.md "Standing standard: BRAIN-BASED ONLY"; MEMORY.md "artificial
life with a proper brain analogue"):** cognition must be done by NEURONS / SYNAPSES / their
communication. A host-side calculation — even a biologically *correct* one — is a shortcut.
Host code is legitimate only for the **environment** and the **body**. The dopamine
reward-prediction-error (RPE) and the value/prediction are *cognition*, so under the strict
standard they must be computed by spiking neurons.
**Scope:** N9 of `research/findings/2026-06-08-remaining-nav-cheats-full-biologization-research.md`.
Builds on N5 (perceived, coordinate-free reward `r`) so the whole RPE loop is coordinate-free.

---

## 1. EXECUTIVE SUMMARY (read this first)

**The shortcut.** Today the dopamine teaching signal is a Python scalar. `g11_bg_runner.py`
computes `reward` (host), and with `--rpe-dopamine` a prediction error `delta = reward −
reward_ema_pre` (`:4867`, a host formula; `reward_ema` is a host EMA at `:4837`), then sets
`bridge.core_config.current_reward_signal = delta` (`:4879`). The bridge multiplies that signed
scalar into the reward-modulated cortico-striatal STDP (`sim/bridge.py:5856`, `:5952`). Under the
strict standard **both** the RPE and the value are host computation → shortcuts. A spiking `snc`
region already exists but is a **silent placeholder** — `g11_bg_runner.py:851`, `name="snc"`,
`IZH2007_DOPAMINE`, `n_dopamine=10` neurons, `internal_density=0`, **no afferents** (verified;
`exc_weight_mean=0`, `inh_weight_mean=0`, `plastic_internal=False`). The teaching signal bypasses
it entirely.

**The target.** The RPE must be the **firing of spiking dopamine neurons** (the `snc` pool), and
the prediction must be a **neural critic** `V(s)` driven by the project's existing neural
perception — never by coordinates or a host EMA. The SNc fires a phasic **burst** when reward
exceeds prediction and a **dip/pause** below its **tonic baseline** when reward falls short
(Schultz 1998; catalog C.22/C.28/C.30; Houk-Adams-Barto 1995). We drive the existing `snc`
IZH2007_DOPAMINE pool with **excitatory reward drive + inhibitory value drive + a tonic baseline**
so its rate encodes `delta = r − V`; a striosome/patch-analogue **value population** supplies the
inhibitory `V`, learned by `delta` itself; and the **DA broadcast** is produced *from SNc firing*
(via the neuromodulator subsystem's `from_region_firing` rule) rather than a scalar.

**Staged build (each stage independently falsifiable).**

| Stage | What | Value source | DA source | `sim/` edits | Falsifier |
|---|---|---|---|---|---|
| **A0** | *(already shipped)* algorithmic `delta = r − R̄` → `current_reward_signal` | host EMA | host scalar | none | nav-score (baseline) |
| **A** | **Spiking SNc fires `delta`**: excitatory reward current + inhibitory value current + tonic baseline drive the `snc` pool; **DA conc produced from SNc firing** (signed). Value still **host-fed as a scaffold** (the inhibitory drive is proportional to the host `R̄`). | host EMA (scaffold) | **SNc firing** | **2 protected** (see §3) | SNc rate tracks `delta` (burst on +RPE, dip on −RPE) |
| **B** | **Neural critic** replaces the host value: a striosome/patch **value population** reads the perceived state, its readout becomes the inhibitory drive to SNc, trained by `delta` (TD/Rescorla-Wagner). | **neural `V(s)`** | SNc firing | reuses Stage A edits + 1 small additive readout edit (see §3) | **Pavlovian cue-shift + omission dip** (only a state-dependent neural critic can produce cue-shift) |

**Headline.** Stage A makes the DA broadcast *spiking* (the literal C.22/C.28 biology). Stage B
makes the prediction *neural and state-dependent* — the actor-critic proper (C.30). The de-risk is
the canonical Schultz **2-cue Pavlovian** falsifier (cue-shift + omission dip), run as a tiny
instrumentation harness *separate from the nav runner*, plus a nav-score regression gate (flagship
multi-goal-deterministic 6-seed must not regress vs the raw-reward baseline). A host global-EMA
value **cannot** produce cue-shift — only a state-dependent neural critic can — so the Pavlovian
test is the real "is the RPE neural and real?" discriminator.

**Protected `sim/` edits (the byte-level-reviewed set; full detail in §3).**
1. **`sim/neuromodulators.py`** — add **ONE new production rule** `from_region_firing_signed`
   (a signed, two-sided variant of the existing `from_region_firing` at `:719`). Required because
   the existing rule is **one-sided** (`max(0, ema − threshold)`) so it can only burst, never dip —
   it cannot encode a *negative* RPE. ~25 lines, additive, opt-in (new `rule_type` string;
   unknown-rule path already no-ops, so every existing config is byte-unaffected).
2. **`sim/neuromodulators.py`** — add **ONE new default-config factory** `_default_snc_dopamine_config()`
   (a `dopamine`-named modulator whose production rule is `from_region_firing_signed` over
   `["snc"]`, tonic baseline = SNc tonic rate). ~20 lines, additive, never imported unless the new
   `--enable-spiking-snc` flag is set. (Could *almost* live runner-side, but the runner constructs
   `NeuromodulatorConfig` objects from `sim.neuromodulators` already — keeping the factory beside
   the others is consistent and testable; if the owner prefers zero new `sim/` symbols, the runner
   can build the identical `NeuromodulatorConfig` inline — see §3 note.)

Everything else — the SNc afferent current injection, the value population + its pathways, the
value readout, the per-step `delta` plumbing — is **runner-side / data / config** (new
`BrainRegion` + `RegionPathway` objects built in `g11_bg_runner.py`, new CLI flags, current writes
to `bridge.cp_external_input_current[region_indices_cp["snc"]]`). **No protected edit is needed for
Stage A's circuit wiring or Stage B's critic** beyond the two neuromodulator additions above. This
is the minimal `sim/` footprint and is the whole point of reusing `from_region_firing` /
`cp_external_input_current` / `RegionPathway`.

**Cost:** Stage A ≈ 1.5 days; Stage B ≈ 2–3 days (incl. the Pavlovian harness). Risk is moderate
and bounded: the chief technical risk is **rate-coding noise on a 10-neuron SNc pool** (de-risked
by Potjans-Diesmann-Morrison 2011: an imperfect spiking RPE still drives TD learning; mitigation =
larger `n_dopamine`, an EMA on the rate, and a paired-opponent read — see §4).

---

## 2. THE NEURAL CIRCUIT

### 2.0 Naming / sign conventions used throughout
- `r` = the per-trial reward (Stage ≥ N5: the perceived approach reward at `g11_bg_runner.py:4744`,
  ∈ {−1, 0, +1}; coordinate-free).
- `V` = the value/prediction (Stage A: host `R̄`; Stage B: neural readout of the value population).
- `delta = r − V` = the RPE the SNc must encode in its firing.
- **SNc tonic** = baseline firing the pool sustains with no reward/value signal (IZH2007_DOPAMINE
  is parameterized for "slow tonic firing 1–5 Hz spontaneously, bursts > 15 Hz", `sim/enums.py:665–670`,
  Grace & Bunney 1984). The burst/dip modulates *around* this tonic, exactly as in biology.

### 2.1 Driving the spiking `snc` pool so its rate encodes `delta` (Stage A)

The `snc` region exists but receives nothing. We give it **three additive current sources**, each
written into `bridge.cp_external_input_current` at the SNc indices once per trial (the same
mechanism every other region in the runner already uses, e.g. `:2256`, `:4792`; indices come from
`region_indices_cp["snc"]`, built at `:3284`):

1. **Tonic baseline drive `I_tonic` (constant, every step).** A small positive current that holds
   the pool at its spontaneous rate so there is *headroom to dip*. Without a tonic floor a negative
   RPE cannot be represented (firing can't go below zero). Set `I_tonic` so the no-signal SNc rate
   sits mid-range (target ≈ 4–6 Hz, i.e. a few spikes across the 10-neuron pool per readout window).
   Calibrated empirically in the harness (§4), not hand-asserted.

2. **Excitatory reward drive `I_reward = k_r · max(0, r)`** (phasic, during the reward-hold window).
   Reward *above zero* depolarizes SNc → burst. (With N5, `r ∈ {−1,0,+1}`; `max(0,r)` is the
   appetitive/US drive.) Biologically: glutamatergic reward input (PPTg/LDT, lateral habenula-relayed
   excitation) onto DA neurons (Schultz 1998 Fig 9C; catalog C.22).

3. **Inhibitory value drive `I_value = k_v · V`** (phasic, during the reward-hold window),
   delivered as **hyperpolarizing current** (or, in Stage B, as a real GABAergic projection — see
   §2.3). Prediction *suppresses* the DA response — the canonical "expected reward elicits no
   burst" result. Biologically: the striosome/patch GABAergic projection to SNc that carries `V`
   (Houk-Adams-Barto 1995; catalog C.30 "striosome-patch = critic state-value V(s)").
   - **Negative-RPE realization (the dip).** When `r < V` (worse than predicted), `I_value`
     dominates `I_reward + I_tonic` and the pool's rate drops **below tonic** → the omission/dip
     signature. The aversive branch `r < 0` *additionally* removes the excitatory drive. We do
     **not** inject a separate "punishment excitation"; biology encodes negative RPE as a *pause*,
     and the asymmetry (dips smaller in rate-range than bursts) is automatic because tonic is low —
     matching `cfg.reward_aversive_scale` semantics already in the bridge (`:5858`).

Net at the SNc soma during the reward window:
`I_snc = I_tonic + k_r·max(0,r) − k_v·V`.
The pool's **windowed firing rate** is a monotone function of `delta = r − V` around tonic: high
`delta` → burst, `delta ≈ 0` → tonic, `delta < 0` → dip. This is the spiking RPE.

**Reading the SNc rate.** The runner already reads windowed population spike counts by summing
`bridge.cp_firing_states` over a window (the readout loop at `:4595–4632`). Stage A adds an
identical small accumulator for `snc` over the reward-hold window (`for _ in range(reward_hold_steps)`
at `:4907`): `snc_spikes += int(firing[snc_idx_host].sum())`. The **firing rate** (spikes /
neurons / window-ms) is the spiking RPE estimate. This is *measured from spikes*, not a formula —
the anti-cheat requirement.

**Why `k_r`, `k_v`, `I_tonic` are not a cheat.** They are *synaptic-weight / drive constants* of a
fixed circuit (the gain of the reward afferent, the gain of the striosome inhibition, the tonic
pacemaker drive), exactly analogous to every other `weight_mean` / drive constant in the runner
(e.g. `cortex→str` weight, the heuristic drive `:2302`). They are properties of the *body/circuit*,
not a per-trial host computation of cognition. The cognition — "is this better than I predicted?" —
is computed by the SNc neurons integrating these drives and **firing**.

### 2.2 The DA broadcast: SNc firing → neuromodulator concentration → plasticity gain (Stage A)

The DA "concentration" must be produced **from SNc firing**, not set to a scalar. The cleaner,
more biological path (option **(b)** in the task) reuses the neuromodulator subsystem:

- Register a **`dopamine`** modulator whose **production rule reads `snc` firing** and whose
  **target is `plasticity_rate` (scope=all)** — i.e. the DA concentration's deviation from baseline
  becomes the cortico-striatal plasticity signal. The bridge **already consumes exactly this**:
  `sim/bridge.py:5894–5904` reads `da_conc = get_concentration("dopamine")` and uses
  `da_signal = da_conc − da_baseline` as the **signed** `effective_signal` that scales the
  eligibility × STDP update (`:5904`, `:5952`). So if we make the `dopamine` concentration track
  `(SNc_rate − tonic_rate)`, the bridge's existing reward-modulation path *is* SNc-driven DA — **no
  bridge edit needed for the broadcast.**

- **The catch (this is the one real `sim/` addition).** The existing `from_region_firing` rule
  (`sim/neuromodulators.py:719`) is **one-sided**: it returns `sensitivity·(rate_ema − threshold)`
  only when `rate_ema > threshold`, else **0** (`:753–755`). It can make DA *rise* on a burst but
  **cannot make DA fall below baseline on a dip** — so it cannot represent a negative RPE. We add a
  **signed** sibling rule `from_region_firing_signed` (§3 edit #1) that returns
  `sensitivity·(rate_ema − threshold)` with **no `max(0,·)` clamp**, so `rate_ema < threshold`
  (a dip) drives the concentration *below* baseline. Set `threshold = tonic_rate` and
  `baseline = 0.5` (matching `_default_dopamine_config`'s tonic). Then:
  - SNc burst (rate > tonic) → conc > baseline → `da_signal > 0` → LTP-direction plasticity.
  - SNc tonic (rate ≈ tonic) → conc ≈ baseline → `da_signal ≈ 0` → no net plasticity (expected
    reward teaches nothing — the RPE-is-zero result).
  - SNc dip (rate < tonic) → conc < baseline → `da_signal < 0` → LTD-direction plasticity.

  The `decay_tau_ms ≈ 200 ms` (phasic DA timescale, already the `dopamine` default) gives the
  concentration a short memory so the windowed rate is smoothed into a clean signal.

**Composing with `--enable-tonic-da`.** `--enable-tonic-da` (`:3196`) registers the *constant-baseline*
`_default_dopamine_config()` whose production rule is `from_reward` (consumes `current_reward_signal`).
Our new `--enable-spiking-snc` registers a `dopamine` modulator whose production rule is
`from_region_firing_signed` over `["snc"]`. **These must not both register a `dopamine` modulator**
(the manager keys by name; two would be a config error). The runner enforces **precedence**
identical to the existing tonic-vs-compartmentalized guard at `:3196` ("`if enable_tonic_da and not
enable_compartmentalized_da`"): **when `--enable-spiking-snc` is set, it owns the `dopamine`
modulator; `--enable-tonic-da`'s registration is skipped.** Both still set
`cfg.enable_neuromodulator_subsystem = True`. The tonic *baseline* (0.5) is preserved, so ACh
window-gating (`--enable-tans`) and the rest compose unchanged. The runner-side `--rpe-dopamine`
host formula (`:4868`) is **disabled** when `--enable-spiking-snc` is on (don't double-count: the
RPE now lives in SNc firing → DA conc, not in `current_reward_signal`). See the compose check in §4.

**The legacy scalar path is bypassed, not broken.** When the `dopamine` conc supplies `da_signal`,
the bridge ignores `reward_prediction_error = current_reward_signal − reward_baseline` (`:5904`
prefers `da_signal`). The runner still sets `current_reward_signal` during the reward window
**only** as the *excitatory drive computation input* is unaffected — actually, under
`--enable-spiking-snc` the runner should leave `current_reward_signal = 0` (the SNc drive is
injected as current, not as the scalar) so nothing downstream reads a stale scalar. Documented in §3.

### 2.3 The neural value critic `V(s)` (Stage B)

Stage A scaffolds `V` from the host `R̄` (the inhibitory drive `I_value = k_v·R̄`). Stage B replaces
that host value with a **spiking value population** whose readout *is* `V`.

**Population (a new `BrainRegion`, runner-side data).**
- `name="striosome_value"` (the striosome/patch analogue; C.30 maps the critic's state-value to
  striosome-patch). Size ≈ 60–100 neurons (a small population; matches `dlpfc_wm`=60, `ec`=80
  scales). `IZH2007_RS_CORTICAL_PYRAMIDAL` (or `IZH2007_STRIATAL_MSN` for striosome authenticity —
  pick MSN; striosomes are MSN-typed). `internal_density` small (≈0.05) for mild recurrent
  smoothing. **Excitatory** projection neurons for the readout; a GABAergic sub-population (or a
  fixed inhibitory pathway) carries the inhibition to SNc.

**Afferent: the PERCEIVED STATE (anti-cheat — no coordinates).** The value population is driven by
the project's existing **neural perception**, declared as `RegionPathway`s (runner-side data,
`plastic=True`, `plasticity_gate="value_input"` so the critic can be staged/frozen):
- When `--enable-visual-cortex`: `cortex_it → striosome_value` (the ventral-stream object code;
  `cortex_it` exists at `:1901`). IT carries "what/where the goal looks like" — the perceived state.
- Else when `--enable-learned-perception`: `sensory → striosome_value`.
- Else (perception-arc default): `ppc_goal_input → striosome_value` and/or `place_cells →
  striosome_value` (the same union the cerebellum's `mossy_state` uses, `:1775–1786`) — the
  perceived goal-vector / place code.
- **Assertion (anti-cheat):** the afferent list is **exactly** these perceived-state regions; the
  runner asserts `("goal_cells" not in afferents) and no raw (gx,gy)/(x,y)` enters — i.e. the
  critic reads only what the *visual/perception* code produced. (Under N5 the reward is already
  coordinate-free, so combined the entire loop is coordinate-free.)

**Readout: `V` as a neural quantity.** Two equivalent realizations (pick the spiking one for the
strict standard; the host-readout is a fallback the standard discourages):
- **(Preferred, fully neural) inhibitory projection.** Add a fixed (or learned) **inhibitory**
  pathway `striosome_value → snc` (`RegionPathway`, runner-side data; GABAergic — set
  `inh_weight_mean` via the region's `exc_fraction`, or use a dedicated inhibitory value
  sub-population). Then `I_value` of §2.1 is **not** injected as host current at all — it is the
  **synaptic current from the value population**, exactly the biology (striosome GABA → SNc). `V` is
  implicitly the value population's firing rate; the SNc integrates `I_reward − (synaptic
  inhibition from striosome_value)` and fires `delta`. **This is the maximally brain-based form: the
  RPE subtraction `r − V` happens at the SNc membrane via opposing synaptic currents, not in host
  code.** No host reads `V` at all.
- **(Fallback, host-readout) population-rate readout.** If the inhibitory-projection gain is hard
  to calibrate, the runner can read the value population's windowed firing rate (same accumulator
  pattern as §2.1) and inject `I_value = k_v · rate(striosome_value)` as hyperpolarizing current.
  This still has the *value* computed by neurons (the population's firing), but the *subtraction*
  is host arithmetic on the injected current — a weaker claim. **Prefer the inhibitory projection;
  document the fallback only as a calibration crutch.**

**Training the critic by `delta` (TD / Rescorla-Wagner bootstrapping).** The value-input pathway
(`*→striosome_value`) is **plastic and reward-modulated by the same `delta`** that the SNc encodes —
i.e. the critic learns to predict reward from the perceived state:
- The bridge's reward-modulation path already moves *all* plastic, reward-eligible synapses by
  `da_signal × eligibility` (`:5952`). Because the value-input synapses are `plastic=True` and carry
  eligibility (pre=perceived-state spikes, post=striosome_value spikes), **they are automatically
  trained by the SNc-derived DA signal** — no extra mechanism. When the perceived state reliably
  precedes reward, `da_signal` (= `delta`) is positive early, LTP raises the state→value weights,
  `V` rises, and subsequent `delta` shrinks toward zero — the **TD fixed point** `V → E[r]`. This is
  the bootstrapping the actor-critic needs and it falls out of the existing three-factor rule
  operating on the value pathway. (Sutton-Barto Ch 6/11; Houk-Adams-Barto 1995.)
- **Sign caveat (must verify in the harness):** the value pathway must move *with* `delta` so that
  positive RPE *raises* `V`. The corticostriatal `cp_d1_d2_sign` (`:5963`) flips D2 synapses; the
  value pathway must be tagged as **D1-like / unsigned** (default sign +1) so `V` increases on
  positive RPE. The runner ensures `striosome_value` is **not** tagged D2 (it isn't a `str_D2_*`
  region, so `cp_d1_d2_sign` leaves it +1 by default — verified by how the sign array is built from
  region names). This is the one numerically delicate coupling; the Pavlovian harness (§4) is
  precisely the test that it has the right sign (cue-shift only happens if `V` rises with predicted
  reward).

**Why Stage B is the actor-critic proper.** Actor = the existing cortico-striatal matrix
(`cortex→str_D1/D2→…→motor`), its weights moved by `delta`. Critic = `striosome_value`, its readout
`V`, trained by `delta`. SNc = the `delta` generator. This is the C.30 anatomical mapping realized
in spikes (SNc δ output; striosome-patch V(s); striatal matrix actor; corticostriatal synapses =
actor weights modified by δ).

### 2.4 Circuit diagram (Stage B, the full actor-critic)

```
        perceived state (NO coords)
   cortex_it / sensory / ppc_goal_input / place_cells
        │ (plastic, gate="value_input",            │ (existing actor afferents,
        │  reward-modulated by delta)              │  reward-modulated by delta)
        ▼                                          ▼
  ┌───────────────────┐   GABAergic        ┌──────────────────────────────┐
  │  striosome_value  │── inhibitory ────► │            snc               │
  │  (critic V(s),    │   (= −k_v·V,       │  IZH2007_DOPAMINE, n=10..30   │
  │   MSN-typed)      │    the value drive)│  tonic baseline + reward exc  │
  └───────────────────┘                    │  → fires rate ∝ delta = r−V  │
        ▲                                   └──────────────┬───────────────┘
        │ trained by delta (TD bootstrap)                  │ from_region_firing_signed
        │                                                  ▼
        │                                   dopamine NM concentration
        │                                   conc − baseline = da_signal (signed)
   reward r (N5 perceived, coord-free)               │
        │  excitatory drive I_reward = k_r·max(0,r)  ▼
        └──────────────────────────────►   bridge: Δw = lr · da_signal · eligibility
                                            (cortico-striatal ACTOR weights AND
                                             striosome_value CRITIC weights)
```

The only host involvement is: render the environment image (body/world), compute the perceived
reward `r` from pixels (N5, body/world), and **inject `I_reward` as current into the `snc` pool**
(the reward afferent — a *body* signal entering the brain, like a sensory drive). Everything labeled
"cognition" — predicting value, computing the prediction error, broadcasting it — is spikes.

---

## 3. EXACTLY which edits are runner-side/additive vs PROTECTED `sim/` edits

The owner reviews the byte-level diff of every protected `sim/` file before it lands, so this list
is precise. **Two** protected edits, both in `sim/neuromodulators.py`, both purely additive and
opt-in. Everything else is runner-side data/config.

### PROTECTED `sim/` edit #1 — `sim/neuromodulators.py`: add `from_region_firing_signed` production rule
- **File / location:** `sim/neuromodulators.py`, inside `_compute_production` (after the existing
  `from_region_firing` block at `:719–755`), plus a one-line docstring entry in the
  `ProductionRule.rule_type` docstring (`:88–148`).
- **What:** a new `rule_type == "from_region_firing_signed"` branch. Identical to
  `from_region_firing` (reads mean firing across `source_regions`, EMA over `window_ms` using the
  same `rate_ema` state key) **except** it returns the signed value with **no `max(0,·)` clamp**:
  `return rule.sensitivity * (ema - rule.threshold) * (self.dt_ms / 1000.0)` for **all** `ema`
  (so `ema < threshold` yields a negative contribution → DA below baseline). ~20–25 lines.
- **Why it CANNOT be data/config:** production rules are dispatched by a hard-coded `if rt == …`
  ladder inside `_compute_production` (`:626–759`). A *new behavior* (a two-sided/signed firing-rate
  rule) is new code in that ladder — there is no config knob that makes the existing one-sided rule
  emit negative contributions. The existing rule's `max(0,·)` is structural (it models one-sided
  neuropeptide co-release). We need a *signed* rule for a bidirectional RPE; that is genuinely new
  `sim/` logic.
- **Backward-compat / safety:** additive `rt` string; the dispatcher's final `return 0.0`
  (unknown-rule no-op, `:757–759`) means **every existing config is byte-for-byte unaffected** —
  no current config uses the new string. 0 behavior change for all flagship configs.
- **Test (CPU-only, additive to `tests/test_neuromodulators.py`):** drive a fake `bridge` with a
  stub `region_manager.indices("snc")` + `cp_firing_states` at high vs low fractions; assert conc
  rises above baseline on high firing and **falls below baseline** on low firing (the property the
  one-sided rule lacks).

### PROTECTED `sim/` edit #2 — `sim/neuromodulators.py`: add `_default_snc_dopamine_config()` factory
- **File / location:** `sim/neuromodulators.py`, beside the other `_default_*_config` factories
  (after `_default_dopamine_config` at `:914–967`).
- **What:** returns a `NeuromodulatorConfig(name="dopamine", baseline=0.5, decay_tau_ms=200,
  concentration_min=0, concentration_max=2.0, targets=[ModulatorTarget("plasticity_rate",
  scope="all", sensitivity=+1.0)], production_rules=[ProductionRule("from_region_firing_signed",
  sensitivity=<gain>, threshold=<tonic_rate>, window_ms=200, source_regions=["snc"])])`. ~18 lines.
- **Why it is in `sim/` (soft — see note):** the runner imports `_default_dopamine_config` /
  `_default_per_action_dopamine_config` from `sim.neuromodulators` already (`:3197`, `:3212`).
  Keeping the new factory beside them is consistent, unit-testable, and documents the canonical SNc
  config. **It carries no new *behavior*** beyond edit #1 — it is data assembly.
- **NOTE — can be avoided entirely:** if the owner prefers **zero new `sim/` symbols beyond the rule
  in edit #1**, the runner can build this exact `NeuromodulatorConfig` inline (it constructs
  `NeuromodulatorConfig`/`ModulatorTarget`/`ProductionRule` from `sim.neuromodulators` already).
  Then **edit #2 disappears and only edit #1 remains protected.** Recommended default: **inline it
  in the runner** to minimize the protected surface to a single ~25-line rule. The factory is listed
  here only so the owner can choose the trade-off; the design works with one OR two protected edits.

### Runner-side / data / config (NO protected edit) — `research/runners/g11_bg_runner.py`
All of the following are additive in the runner (new flags, new `BrainRegion`/`RegionPathway`
objects in `build_bg_brain_regions`, new current writes in the trial loop). None touches `sim/`.

1. **New CLI flags:** `--enable-spiking-snc` (master), `--snc-tonic-pA`, `--snc-reward-gain-pA`
   (`k_r`), `--snc-value-gain` (`k_v`), `--enable-neural-critic` (Stage B; without it, Stage A uses
   the host-`R̄` value scaffold), `--critic-value-input` (which perceived-state source). Registered
   beside `--rpe-dopamine` (`:5804`).
2. **Stage A circuit wiring (data + current writes):**
   - **Tonic + reward + value drive into `snc`:** in the reward-hold block (`:4907`), write
     `bridge.cp_external_input_current[region_indices_cp["snc"]] = cp.float32(I_tonic +
     k_r*max(0,r) − k_v*V)` (Stage A: `V = reward_ema_pre`, the host scaffold; this is the *only*
     host use of the value, and it is explicitly the *scaffold* stage). `region_indices_cp["snc"]`
     exists (`:3284`). Also write `I_tonic` every step (outside the reward window) for the tonic
     floor, or precompute a constant `cp_external_input_current` offset for the SNc indices at init.
   - **DA modulator registration:** when `--enable-spiking-snc`, set
     `cfg.enable_neuromodulator_subsystem = True` and append the SNc `dopamine` config (factory or
     inline per edit #2 note). Enforce precedence vs `--enable-tonic-da` (skip its registration).
   - **Disable the host RPE formula:** when `--enable-spiking-snc`, do **not** take the
     `rpe_dopamine` branch at `:4868`; leave `current_reward_signal = 0` (the RPE is in SNc firing /
     DA conc now, not the scalar).
   - **SNc rate accumulator:** sum `bridge.cp_firing_states[snc_idx_host]` over the reward-hold
     window for logging / the harness (`snc_idx_host = region_indices_cp["snc"].get()`).
3. **Stage B neural critic (data only):**
   - **New region:** `BrainRegion(name="striosome_value", n_neurons≈80, exc_fraction≈0.8,
     internal_density≈0.05, izh_neuron_type=IZH2007_STRIATAL_MSN)`. Appended in
     `build_bg_brain_regions` only when `--enable-neural-critic`.
   - **Afferent pathways (plastic, perceived-state):** `RegionPathway(from_region=<perceived state>,
     to_region="striosome_value", density≈0.3, weight_mean small, plastic=True,
     plasticity_gate="value_input")` for the chosen perceived-state source(s) (§2.3 union).
   - **Value→SNc inhibitory pathway:** `RegionPathway(from_region="striosome_value",
     to_region="snc", density≈0.5, weight_mean=<k_v-equivalent>, plastic=False)` with GABAergic sign
     (via a value inhibitory sub-pop or the region's `exc_fraction`). With this present, the runner
     drops the host `I_value` injection (the inhibition is now synaptic — the strict-standard form).
   - **Anti-cheat assertion:** assert the critic's afferent set contains only perceived-state region
     names (no `goal_cells`, no raw-coordinate region) and that `V` is never seeded from
     `(gx,gy)`/distance.
4. **Webapp / preset:** add `--enable-spiking-snc` (+ `--enable-neural-critic`) to the relevant
   flagship preset string and surface the SNc rate / `V` / `delta` in the run JSON (the
   `keep-webapp-current` skill territory — flagged, not required for the science).

**Summary of the protected surface:** **1 (or optionally 2) additive function(s) in
`sim/neuromodulators.py`.** No edit to `sim/bridge.py`, `sim/regions.py`, `sim/config.py`,
`sim/enums.py`, or `sim/kernels.py` — the bridge **already** (a) injects per-region current via
`cp_external_input_current`, (b) consumes `dopamine` concentration as the signed plasticity signal
(`:5894–5904`), (c) trains plastic reward-eligible pathways by that signal (`:5952`), and (d)
supports declarative regions/pathways with plasticity gates. The whole circuit is buildable on top
of that.

---

## 4. DE-RISK — falsifiers and acceptance criteria

### 4.1 The canonical Pavlovian falsifier (the real "is the RPE neural and real?" test)
**A tiny instrumentation harness, NOT the nav runner.** New file
`research/runners/snc_pavlovian_probe.py` (runner-side; builds a minimal bridge with the `snc` pool
+ the value population + one or two "cue" input regions; no gridworld). This is the strongest,
most diagnostic test and it gates a **real metric**, not a vibe.

**Schedule (2-cue conditioning, Schultz 1998; catalog C.22/C.28):**
- A **cue (CS)** region drives the value population; after a fixed delay a **reward (US)** current
  is injected into the SNc reward afferent. Run many trials.
- **(i) Cue-shift (CS acquisition).** Measure the SNc burst (windowed rate above tonic) time-locked
  to **US** vs to **CS** across training.
  **Acceptance:** early trials — SNc bursts at **US**, not CS. Late trials (after the critic learns)
  — SNc burst **shifts to CS** and the US burst **shrinks toward zero** (reward fully predicted →
  no RPE at US). Quantitative gate: `US-burst(late) < 0.5 × US-burst(early)` AND
  `CS-burst(late) > 2 × CS-burst(early)`, multi-seed (≥3 seeds), sign-consistent.
- **(ii) Omission dip.** On a probe trial, **withhold** the expected US after the CS.
  **Acceptance:** the SNc rate **dips below tonic** at the expected-reward time (the omission/dip
  signature). Gate: `rate(omission, expected-US-window) < tonic_rate − margin`, multi-seed.
- **Why this is the discriminator:** a **host global-EMA value cannot produce cue-shift** — cue-shift
  requires the value to be **state-dependent** (the CS state must acquire value). Only the **neural
  critic** `V(s)` driven by the cue (Stage B) can pass (i). The omission dip (ii) requires the
  **signed** DA rule (edit #1) — a one-sided rule cannot dip. So passing both proves the RPE is
  (a) neural, (b) state-dependent, and (c) bidirectional. **Stage A is expected to PASS (ii) but
  PARTIALLY pass (i)** (host-`R̄` value gives a generic "reward got predictable" dip but not a
  CS-specific shift); **Stage B is required for (i).** That expected Stage-A-fails-cue-shift result
  is itself a clean falsifiable prediction.

### 4.2 Nav-score regression gate
- **Flagship multi-goal-deterministic 6-seed** (`--moving-goal --goal-schedule multi --deterministic`
  + the A+E+G v2.5 stack, the documented best). Run with `--enable-spiking-snc` (Stage A) and again
  with `--enable-neural-critic` (Stage B).
- **Acceptance:** summed reward **≥ the raw-reward baseline** (a correct critic should *match or
  beat* actor-only — C.30/O.20 predict the evaluator escapes local optima). Specifically Stage A
  must not regress vs `--rpe-dopamine` (A0); Stage B must not regress vs Stage A. 6-seed mean,
  honest std, no seed catastrophe.
- This is necessary but **not sufficient** — the Pavlovian test (4.1) is what proves the mechanism
  is the *real* biology rather than a cosmetic re-routing that happens to preserve nav score.

### 4.3 Anti-cheat checklist (assert in code / verify in the harness)
1. **The RPE is the SNc's FIRING, not a host formula.** Under `--enable-spiking-snc` the
   `rpe_dopamine` host branch (`:4868`) is OFF and `current_reward_signal` stays 0; the plasticity
   signal `da_signal` is produced by `from_region_firing_signed` over `["snc"]` — i.e. read from
   `cp_firing_states`. Grep-assert no host `delta` reaches `current_reward_signal`.
2. **The value is neural and learned (Stage B).** `V` is the `striosome_value` population's
   readout; its input synapses are `plastic=True` and trained by `delta`. **Never** seeded with
   `(gx,gy)` or true distance — assert the afferent set is perceived-state regions only.
3. **The reward is perceived (with N5).** `r` is `sc_salience_offset_from_image`-derived
   (`:4744`), coordinate-free. Combined with (1)+(2), the **entire** RPE loop is coordinate-free.
4. **No double-counting (compose check).** With `--enable-tonic-da` also set: assert only ONE
   `dopamine` modulator is registered (spiking-SNc owns it; tonic-da skipped). With
   `--enable-compartmentalized-da`: spiking-SNc is mutually exclusive (per-action channels are a
   different DA decomposition; document as not-combinable in v1, mirroring the existing
   tonic-vs-compartmentalized guard `:3196`).
5. **Stage A value is honestly a scaffold.** When `--enable-spiking-snc` *without*
   `--enable-neural-critic`, the run JSON / docstring states `V = host R̄ (scaffold)` so the result
   is not overclaimed as a neural critic. Only `--enable-neural-critic` may be reported as
   actor-critic-proper.

### 4.4 Calibration smoke (before the real runs)
A `--snc-probe` mode (in the Pavlovian harness): sweep `I_tonic`, `k_r`, `k_v` and report the SNc
windowed rate vs known `(r, V)` to confirm the rate is **monotone in `delta`** (burst on +RPE, tonic
at 0, dip on −RPE) before trusting it as a teaching signal. This replaces hand-asserting the gains;
it *measures* that the spiking RPE is well-formed (the project's "probes must match deployed config"
rule — the harness builds the SAME `snc` region the runner does).

---

## 5. COST / RISK + reusable machinery

### 5.1 Cost
| Stage | Work | Est. |
|---|---|---|
| **A** | edit #1 (signed rule) + tests; runner flags; SNc current injection + DA registration + precedence; SNc rate accumulator; calibration smoke | **~1.5 days** |
| **B** | `striosome_value` region + perceived-state afferents + inhibitory value→SNc pathway; verify TD bootstrap sign; Pavlovian harness (cue-shift + omission); nav 6-seed | **~2–3 days** |
| **Total** | | **~3.5–4.5 days** (matches the N9 research estimate of 3–5 days) |

### 5.2 Risk (honest)
- **Rate-coding noise on a 10-neuron SNc pool (chief risk).** A windowed rate over 10 neurons is
  coarse; `delta` may be jittery. **De-risked by Potjans-Diesmann-Morrison 2011** — an *imperfect*
  spiking dopaminergic error still drives TD learning. **Mitigations (all cheap, runner-side):**
  raise `n_dopamine` (10 → 30–50; it's a free `BrainRegion` size), use the `decay_tau_ms≈200ms` EMA
  on the DA concentration (already smooths), widen the reward-hold window, and optionally a
  paired-opponent read (burst pool vs a baseline pool) if single-pool SNR is inadequate.
- **TD-bootstrap sign / stability (Stage B).** The value pathway must move *with* `delta` (positive
  RPE raises `V`); if mis-signed, `V` diverges and `delta` never converges. **Mitigation:** the
  Pavlovian cue-shift test is exactly the sign check; calibrate on the harness before nav. The
  `cp_d1_d2_sign` default (+1 for non-D2 regions) is correct for `striosome_value` but must be
  verified (it's MSN-typed but not `str_D2_*`).
- **Calibration of `I_tonic`/`k_r`/`k_v`.** Three constants to set so the rate spans burst↔dip
  cleanly. **Mitigation:** the `--snc-probe` sweep (4.4) measures the monotone-in-`delta` property
  rather than guessing.
- **Inhibitory value→SNc pathway gain (Stage B preferred form).** Getting the synaptic inhibition
  to exactly cancel the reward drive at `r = V` is delicate. **Mitigation:** the host-readout
  fallback (§2.3) is a calibration crutch; or learn the `striosome_value→snc` weight too (start
  `plastic=False`, enable if needed).
- **Bounded blast radius.** Every change is opt-in behind `--enable-spiking-snc` /
  `--enable-neural-critic`; the single protected rule no-ops for all existing configs. A regression
  cannot touch any flagship run that doesn't set the new flags.

### 5.3 Reusable machinery (with `file:line`)
- **Existing silent SNc region** — `research/runners/g11_bg_runner.py:851` (`name="snc"`,
  `IZH2007_DOPAMINE`, `n_dopamine=10`, `internal_density=0`, the placeholder to wire up).
- **IZH2007_DOPAMINE tonic/burst params** — `sim/enums.py:665–670` (tonic 1–5 Hz, burst > 15 Hz,
  Grace & Bunney 1984 — the SNc fires the way we need *out of the box*).
- **Per-region current injection** — `bridge.cp_external_input_current[region_indices_cp[...]]`
  (pattern at `:2256`, `:4792`; SNc indices built at `:3284`). This is how `I_tonic/I_reward/I_value`
  enter the SNc with **no `sim/` edit**.
- **Windowed population spike-count readout** — the readout loop summing `bridge.cp_firing_states`
  at `:4595–4632`; the reward-hold loop at `:4907`. The SNc-rate / value-rate accumulators copy
  this pattern.
- **DA-concentration → signed plasticity signal (the broadcast, already consumed)** —
  `sim/bridge.py:5894–5904` (`da_signal = get_concentration("dopamine") − baseline`), used as the
  signed `effective_signal` at `:5904` and applied to eligibility × STDP at `:5952`. **This is why
  Stage A needs no bridge edit** — make the `dopamine` conc track SNc firing and the existing path
  *is* SNc-driven DA.
- **`from_region_firing` rule (the template for edit #1)** — `sim/neuromodulators.py:719–755`
  (reads region firing, EMA over `window_ms`); edit #1 is its signed sibling.
- **`_default_dopamine_config` (the template for the SNc DA config)** — `sim/neuromodulators.py:914–967`
  (baseline 0.5, decay 200 ms, plasticity_rate scope=all).
- **Declarative regions/pathways + plasticity gates** — `sim/regions.py:32` (BrainRegion), `:171`
  (RegionPathway, `plastic`, `plasticity_gate`); `bridge.set_plasticity_gate` (`:2622`) to
  stage/freeze the critic. The `striosome_value` region + its afferents are pure data.
- **Reward-modulated plasticity training the critic** — `sim/bridge.py:5838–5982` (eligibility
  decay + `Δw = lr · da_signal · eligibility`, with `cp_d1_d2_sign` and gates). The value pathway
  rides this for free.
- **Precedence pattern to copy** — `g11_bg_runner.py:3196` (`if enable_tonic_da and not
  enable_compartmentalized_da`) is the exact template for the spiking-SNc-vs-tonic-da guard.
- **Host-`R̄` value scaffold (Stage A)** — the existing `reward_ema` / `reward_ema_pre` (`:3423`,
  `:4827`, `:4837`) is reused verbatim as the Stage-A inhibitory-drive value (then *replaced* by the
  neural critic in Stage B).
- **N5 perceived reward** — `g11_bg_runner.py:4744` (`sc_salience_offset_from_image`-derived `r`),
  making the loop coordinate-free.

---

### References (grounding)
- Schultz W. (1998) "Predictive reward signal of dopamine neurons." *J Neurophysiol* 80:1–27
  (RPE, cue-shift, omission dip — catalog C.22/C.28).
- Houk, Adams & Barto (1995) — striosome = critic V(s); SNc δ; matrix = actor (catalog C.30).
- Joel, Niv & Ruppin (2002) "Actor–critic models of the basal ganglia." *Neural Networks* 15:535.
- **Frémaux, Sprekeler & Gerstner (2013)** "Reinforcement Learning Using a Continuous Time
  Actor-Critic Framework with Spiking Neurons." *PLoS Comput Biol* 9(4):e1003024 — the canonical
  spiking actor-critic blueprint.
- **Potjans, Diesmann & Morrison (2011)** "An imperfect dopaminergic error signal can drive
  temporal-difference learning." *Front Comput Neurosci* (PMC3093351) — a noisy spiking SNc RPE
  still drives TD learning (de-risks the 10-neuron-pool noise).
- Grace & Bunney (1984) — SNc tonic/burst firing (the IZH2007_DOPAMINE parameterization).
- Sutton & Barto, *Reinforcement Learning* Ch 6 (TD), Ch 11 (actor-critic / average-reward).
- N9 research: `research/findings/2026-06-08-remaining-nav-cheats-full-biologization-research.md` §N9.
