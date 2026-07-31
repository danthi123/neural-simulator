---
type: plan
status: live
date: 2026-06-10
---

# N9 — temporal-difference cue-shift on the spiking SNc: migrating the dopamine burst from reward onto the predictive cue

**Date:** 2026-06-10
**Type:** READ-ONLY deep-research + DESIGN. No code edited (the only file written is this doc). Standing practice: deep research + catalog review BEFORE committing build/GPU resources.
**Scope:** design how to add a **temporal-difference (TD)** cue-shift to the project's already-built spiking substantia-nigra-pars-compacta (SNc) reward-prediction-error (RPE) circuit, so that after cue→reward learning the SNc phasic dopamine burst **migrates** from the reward onto the earlier predictive cue (the iconic Schultz 1997 signature — the one canonical dopamine signature the current circuit does NOT yet show).

**Terms, defined once.**
- **RPE** = reward-prediction-error, the dopamine teaching signal.
- **TD error** δ_t = r_t + γ·V(s_{t+1}) − V(s_t): reward now, plus the *discounted value of the next state*, minus the *value of this state*. γ ∈ (0,1) is the discount. The bracketed `γ·V(s_{t+1}) − V(s_t)` is the **value derivative / bootstrap** term that Rescorla-Wagner (R-W, δ = r − V) lacks.
- **R-W** = Rescorla-Wagner: δ = r − V, a one-step prediction error with NO time axis (what the circuit computes today).
- **SNc** = substantia nigra pars compacta, the midbrain dopamine-cell pool; its windowed firing IS the dopamine signal here.
- **Critic** = the population that learns and represents V(state): the `striosome_value` medium-spiny-neuron (MSN) pool.
- **CS / US** = conditioned stimulus (the predictive cue) / unconditioned stimulus (the primary reward).
- **GABA_B / GIRK** = a slow (tau ≈ 150 ms) metabotropic inhibitory conductance through a G-protein inwardly-rectifying potassium channel (reversal ≈ −90 mV); the engine's strong, sign-correct way to subtract V onto the SNc (already shipped, owner-approved).
- **CSC** = complete serial compound: a tapped-delay-line state code in which a cue is represented not as one event but as a *sequence* of sub-states (cue@t0, cue@t0+Δ, …) — the representation that makes TD reproduce the cue-shift.
- **Eligibility trace** = a per-synapse decaying memory of recent pre/post co-firing (the project's `cp_eligibility_trace`); already implemented (catalog C.29).
- **Host** = computed in Python (the runner), not by simulated neurons — a shortcut under the project's BRAIN-BASED-ONLY standard even when the formula is biologically correct.

**Load-bearing standard (restated).** The TD error must be computed by the **spiking circuit**, not host code. A Python `δ = r + γ·V' − V` is a shortcut. An **honest NEGATIVE** (the spiking TD failing to show a clean cue-shift — e.g. the bootstrap is too noisy on a rate-coded critic) maps a substrate limit and **IS a valid deliverable** (catalog C.28/C.30 acceptance is explicitly "currently failing").

**Grounded in the EXISTING circuit (read first; this design reuses, it does not reinvent):**
- `research/findings/2026-06-10-N9-spiking-snc-current-state-assessment.md` — the current-state map: the loop computes **R-W δ = r − V** fully in spikes; signature **(a) cue-shift is the one NOT validated** and is flagged as "a deeper, orthogonal later increment."
- `research/findings/2026-06-10-N9-fully-spiking-reward-loop-MILESTONE.md` — both reward pieces are now spiking: **r** = `reward_us` (pedunculopontine-nucleus-like, PPN) excitation onto the SNc; **V** = the FS-clamped `striosome_value` critic's GABA_B subtraction at the SNc membrane.
- `research/findings/2026-06-08-spiking-snc-stageB-Bprime-value-subtraction-circuit-research.md` + `research/runners/snc_stageb_critic_probe.py` — the value subtraction circuit and its 4-gate CPU falsifier (the test harness this design extends).
- Catalog `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`: **C.22** (Schultz RPE, the cue-shift criterion + HS98 quantitative numbers), **C.28** (TD error δ = r + γV(s′) − V(s), "partial — gap is measurable"), **C.29** (eligibility traces, "implemented" = TD(λ) in all but name), **C.30** (actor-critic, "actor implemented, critic missing"; acceptance = cue-shift + omission dip), **C.31** (bootstrapping vs Monte Carlo — why phasic DA *must* bootstrap), **C.33** (PPN→DA, the cue-shift driver), Kandel 6e Ch 43.

---

## 0. Executive summary + recommendation

**The single decisive finding.** The circuit already has every piece TD needs **except two**, and both are addable as *one* new spiking idea: (1) a **temporally-extended cue representation** (a sustained cue trace so V can predict from cue-onset forward), and (2) the **bootstrap** `γ·V(s_{t+1}) − V(s_t)` delivered to the SNc as the **temporal derivative of the value channel**, not just `−V`. Crucially, the engine's GABA_B/GIRK machinery is *already* a slow, decaying conductance with a configurable time constant — so a **difference of two GABA_B-like value channels at different latencies** computes the value derivative *at the SNc membrane, in conductance, with no host arithmetic*. The critic, the place/cue code, the eligibility trace, the SNc, and the dopamine-from-firing rule are all reused unchanged.

**Recommendation — build `TD-DERIV-CONDUCTANCE` (ranked #1 in §6): a two-latency value channel whose difference IS the bootstrap.** The critic projects to the SNc through *two* slow inhibitory conductances of the SAME striosome→SNc pathway:
- a **fast/leading** value channel (short rise, the current value `V(s_t)`), delivered as a **hyperpolarizing** GABA_B current (the existing `−V` term);
- a **slow/lagging** value channel (a delayed copy of the value, `V(s_{t−Δ})` ≈ a one-step-back value), delivered as a **de-inhibiting** (positive) drive of equal strength.
At the SNc, the **net** inhibitory current is proportional to `V(s_t) − V(s_{t−Δ})`, which (running forward in trial time, with the value trace acquiring power on the cue) is the **positive value derivative** that, *added to the reward `r`*, makes the SNc burst when value RISES (cue onset, value jumps 0→V) and fall back to baseline once value plateaus (between cue and reward) — exactly the bootstrap that migrates the burst onto the cue and produces the omission dip at reward time. This is realized by the **existing GABA_B conductance** plus **one new, additive, default-OFF protected edit**: a *second* slow conductance channel (the lagging value), or — the strictly-minimal form — a **single GABA_B conductance read as a leaky derivative** (`g_now − g_delayed`). See §2–§3.

**Why this and not a tapped-delay-line (CSC) of cue sub-states.** The textbook way to make TD reproduce the cue-shift is a CSC: represent the cue as a chain of time-tagged sub-states and let TD back-propagate value one tap per trial (catalog C.28; Montague-Dayan-Sejnowski 1996). That is the **highest-fidelity** option and is on-substrate (a `cue_delay_chain` of relay populations). But it requires a multi-population delay chain + per-tap critic weights and a many-trial back-propagation that a small rate-coded critic learns noisily. The **conductance-derivative** route gets the *same observable* (burst migrates to cue, dip at reward) from machinery the engine already has (a slow conductance with a time constant), at a fraction of the build + de-risk cost, and is the cheaper-first de-risk. **CSC is ranked #2 — the fidelity comparator if the owner wants the literal tapped-delay account.** A pragmatic hybrid (a short 2–3-tap cue trace feeding the conductance-derivative) is the recommended *first nav build* if the single-step derivative under-migrates (§6).

**Protected `sim/` edit surface: ONE additive, guarded, default-OFF byte-identical edit** — a second slow inhibitory conductance channel (the "lagging value"), an exact structural mirror of the already-shipped GABA_B/GIRK block (the same NMDA-pattern-inverted code, a second `cp_conductance_g_*` + decay + current term), gated by a new `cfg.enable_td_value_derivative` (default False → the block is unreached → `total_input_current_pA` is byte-identical). Everything else — the cue trace, the second pathway tag, the SNc drive, the validation — is runner-side. The byte-identity argument + proof are in §3.

**Cheap-first de-risk (§5): extend `snc_stageb_critic_probe.py`** into a Pavlovian cue→reward protocol over many trials (CPU, `SIM_BACKEND=numpy`, no GPU, no nav build) and measure **burst migration**: early trials → SNc bursts at the US; late trials → SNc bursts at the **CS**, not the US; omission → the dip stays at **reward time**. The decisive metric is catalog C.22's: **correlation r > 0.7 between trial number and the SNc burst's time-of-peak shifting from US-onset toward CS-onset**, multi-seed (≥3). Anti-cheats: lesion the cue→critic pathway → no transfer; an **unpaired** (random CS/US timing) control → no transfer; a provenance assertion that **no host TD term** reaches the SNc.

**Honest could-be-NEGATIVE, flagged up front.** A rate-coded spiking critic estimates V noisily; the bootstrap `γV(s_{t+1}) − V(s_t)` is a *difference of two noisy estimates*, which can be dominated by noise and is the classic TD instability (Potjans-Diesmann-Morrison 2011 studied exactly "an imperfect dopaminergic error signal driving TD"). The honest outcomes (§6): (i) clean migration → ship; (ii) **partial** migration (a *graded shift* of the burst toward the cue without fully vacating the reward, which is *also* what HS98 measured for slow-learned pairs — a defensible partial, not a failure); (iii) **no** migration (the conductance-derivative bootstrap is too noisy / the single-step Δ is too short) → the deliverable is the negative + the CSC/longer-trace recommendation. All three are valid findings.

---

## 1. WHAT IS MISSING for TD vs the current Rescorla-Wagner

### 1.1 Where the circuit is today (verified in code)

The live nav-loop SNc drive (`research/runners/g11_bg_runner.py`, the `_I_snc` block, ~lines 7107–7143, with `--enable-neural-critic --spiking-snc --spiking-reward-us`):

```python
# reward burst r = reward_us FIRING into the SNc (spiking US afferent)
bridge.cp_external_input_current[reward_us_idx] = reward_us_drive_pa * max(0, reward)   # the `r` term, neural
_I_snc = float(snc_tonic_pa)                  # tonic pacemaker ONLY
bridge.cp_external_input_current[snc_idx] = _I_snc
#  ...the value V is subtracted at the SNc MEMBRANE by the striosome_value GABA_B current
#     (the striosome_value -> snc pathway, receptor="gaba_b") — neural, not a host term.
```

So the SNc integrates, **at reward-delivery time**: `(tonic) + (reward_us excitation = r) − (striosome GABA_B = V)`, and its windowed firing encodes **δ = r − V**. The value V is read from the perceived *state* (the place/cue code) through the plastic `place → striosome_value` critic and subtracted via GABA_B (catalog B.15 / Eshel 2015 subtraction). **This is R-W** (the probe's own docstring says so) and it produces:
- US-burst **shrink** as V cancels r across training,
- **omission dip** (CS but no US → SNc dips below tonic),
- a **state-specific gap** (predicted < unpredicted) — the host-EMA-impossible discriminator that the Stage-B gate proves.

What it does **NOT** produce: **the burst never moves onto the cue.** At cue onset (CS, no reward yet) the SNc does *not* burst above tonic, because the reward term `r` is zero until the US and the value term only ever *subtracts*. The cue acquires no DA-burst-evoking power. **This is the C.22 gap verbatim** ("the predictive cue itself never acquires DA-burst-evoking power… reproducing it requires the value-function critic of an actor-critic architecture").

### 1.2 The two missing ingredients (the precise delta)

For TD δ_t = r_t + γ·V(s_{t+1}) − V(s_t), two things must be added on top of the existing `r − V`:

**(i) A temporally-extended cue/state representation.** Today the place/cue code is presented per-event; V is read at the moment the state is rendered. For the burst to appear *at the cue* and persist *until the reward*, the critic must represent the cue as a state with **temporal extent** — a sustained cue trace (or a CSC chain) so that V(cue-state) is non-zero from cue-onset onward, *before* the reward arrives. Without a cue that persists in the critic's input, there is no `V(s_t)` to be non-zero at cue time and hence nothing for the derivative to detect. (Catalog C.28: the cue-shift "emerges naturally because P(t) is itself learned over trials" — but only if P(t) is *defined over time*, i.e. the cue state spans the CS→US interval.)

**(ii) The bootstrap term `γ·V(s_{t+1}) − V(s_t)` realized in spikes.** This is the heart of TD and the one thing R-W structurally cannot do: the SNc must receive **the temporal change in value**, not just `−V`. When value *rises* (a new predictor turns on: 0 → V at cue onset), `V(s_{t+1}) − V(s_t) > 0` → the SNc bursts **at the cue**. When value is *flat* (between an established cue and its reward), the term is ≈ 0 → no burst. When the predicted reward is *omitted*, value *drops* (V → 0 at the expected reward time) → `V(s_{t+1}) − V(s_t) < 0` → the SNc **dips at reward time** even with no reward signal. Bootstrapping is *required* by the empirical data (catalog C.31: DA shifts on a *single trial* with no episode-end wait — Monte Carlo cannot do this). **The minimal biologically-grounded addition is therefore: deliver the value DERIVATIVE (not just −V) to the SNc, in spikes/conductance.**

### 1.3 The minimal addition (named)

> **Add a value-DERIVATIVE channel to the SNc, plus a sustained cue trace, both spiking.** The derivative = `V(s_t) − V(s_{t−Δ})` (a forward-time discrete derivative of the critic's value output). Realize it as a **difference of two slow inhibitory conductances at two latencies** (the conductance-derivative, §2.2) — reusing the GABA_B/GIRK machinery — so the SNc membrane *computes* the bootstrap. The cue trace is a sustained/relayed cue population so V(cue-state) is non-zero across the CS→US interval. Nothing else in the loop changes: r is still `reward_us`, V is still the striosome critic, dopamine is still SNc firing, plasticity is still three-factor (eligibility × DA). **This is the smallest change that turns δ = r − V into δ = r + γV(s_{t+1}) − V(s_t).**

---

## 2. THE SPIKING REALIZATION on the existing circuit

Goal restated precisely: the SNc windowed firing must encode `δ_t = r_t + (γ·V(s_{t+1}) − V(s_t))`, where every term is produced by neurons/synapses. r and V are already spiking. The new work is (A) a cue trace and (B) the value-derivative delivery.

### 2.1 (A) The sustained cue trace (runner-side; no protected edit)

The cue/state that feeds the critic must have temporal extent over the CS→US interval so V(cue-state) is non-zero from cue-onset to reward. Two on-substrate ways, both reusing existing machinery:

- **A-trace (recommended, minimal): a sustained cue population.** Present the cue (the goal-predicting perceptual state — the place/`vs_place_context`/`cue` code) and let it **persist** across the CS→US window. Mechanically this is a tonic drive on the cue/place region held from CS-onset to US (the same `cp_external_input_current` write the probe already uses), OR — more faithfully — a **self-sustaining cue trace** via the engine's **NMDA-recurrent** machinery (`enable_nmda_recurrent`, the slow excitatory mirror of GABA_B already in the engine, `sim/bridge.py:5731`): a recurrent `cue → cue` slow-NMDA loop holds a short working-memory trace of the cue after its offset (the standard "stimulus trace" of TD; Sutton-Barto Ch 12). The critic then reads a value that is non-zero across the interval. **No protected edit — `enable_nmda_recurrent` and the cue/place regions already exist.**
- **A-CSC (fidelity option): a tapped-delay chain.** A `cue_delay_chain` of small relay populations (cue@Δ, cue@2Δ, …) each driving the critic through its own plastic synapse — the complete-serial-compound. This is the literal CSC that makes TD back-propagate value one tap per trial (Montague-Dayan-Sejnowski 1996; catalog C.28). Higher fidelity, more build + slower learning; **the §6 #2 comparator.**

For the **first de-risk and nav build, use A-trace** (sustained cue, optionally NMDA-held). It is the cheapest representation that gives V temporal extent.

### 2.2 (B) The value DERIVATIVE at the SNc — the one genuinely new mechanism

This is where the bootstrap lives, and where the existing GABA_B/GIRK subtraction is **reused and extended**. The insight: a *single* slow conductance already gives a *low-pass-filtered* copy of the critic's value. A **difference of two such conductances at different time constants/latencies** is a band-pass = a **temporal derivative**. The SNc then sees `d/dt V`, the bootstrap.

**B-1 ★ RECOMMENDED — two-latency value channel (the conductance-derivative).** The striosome→SNc value pathway drives **two** slow inhibitory conductances on the SNc:
- the **leading** channel = the existing GABA_B/GIRK conductance `g_gabab` (rise+decay ~150 ms): a **hyperpolarizing** current `I_lead = g_gabab·(E_K − V)`, E_K ≈ −90 mV → this is the current `−V(s_t)` subtraction (unchanged).
- a **lagging** channel = a *new* slow conductance `g_gabab_lag` driven by the SAME striosome firing but with a **longer time constant** (e.g. tau ≈ 300–500 ms) so it tracks a *delayed* copy of the value, `≈ V(s_{t−Δ})`. Its current is delivered with **opposite sign** (de-inhibition / a small excitatory-equivalent), `I_lag = +k·g_gabab_lag·(something)` — concretely the *cleanest* implementation makes the lagging channel **reduce** the leading subtraction: the net inhibitory current onto the SNc is `g_gabab·(E_K − V) − g_gabab_lag·(E_K − V) = (g_gabab − g_gabab_lag)·(E_K − V)`, which is **proportional to the difference of the two value channels = the value derivative** `V(s_t) − V(s_{t−Δ})`.

**Sign analysis (the bootstrap falls out).** Define the net value-inhibition `I_net = (g_gabab − g_gabab_lag)·(E_K − V)`. (E_K − V) < 0, so `I_net` hyperpolarizes proportionally to `(g_gabab − g_gabab_lag)`.
- **Value RISING (cue onset, 0 → V):** the fast channel `g_gabab` rises first; the slow `g_gabab_lag` lags → `g_gabab > g_gabab_lag` → `I_net` is *strongly* hyperpolarizing. **But the SNc reads the bootstrap as a POSITIVE δ — so the sign must be inverted at delivery** (see the critical sign note below): the derivative of value should *excite* the SNc when value rises (the burst migrates *onto* the cue). The correct wiring makes the **rising value DE-INHIBIT (disinhibit) the SNc** — exactly the B′-DISINHIBIT logic already researched (`2026-06-08-Bprime-*`): route the value-derivative so a value *increase* *releases* the SNc (a transient disinhibition burst at the cue), and a value *decrease* (omission) *adds* inhibition (the dip).
- **Value FLAT (between cue and reward):** `g_gabab ≈ g_gabab_lag` → `I_net ≈ 0` → no net effect → the SNc sits at tonic (no burst between cue and reward, as in Schultz). ✅
- **Value FALLING (omission, V → 0 at reward time):** `g_gabab < g_gabab_lag` → the net flips → the SNc is *extra-inhibited* → **dip at reward time**. ✅ And because this is driven by the *value* channel (the cue's learned prediction), the dip is at the *expected reward time*, not the cue. ✅

**The critical sign realization (load-bearing — same lesson as B′).** The value *level* must SUBTRACT from the SNc (more V → less SNc at the reward: the R-W `−V`, which the existing GABA_B already does). The value *derivative* must ADD to the SNc when value rises (more dV/dt → more SNc at the cue: the bootstrap burst). These are two different signs on two different quantities. The clean way to get both without fighting the depolarized SNc membrane is to keep the **existing `−V` GABA_B as-is** (the subtraction at reward) and deliver the **derivative via the B′-DISINHIBIT relay** that the project already designed: the value-derivative gates a normal-reversal **excitatory** relay `snc_drive`, so a value *rise* (positive derivative) *increases* the relay's excitation of the SNc (burst at cue) and a value *fall* (negative derivative) *decreases* it (dip). **Reuse `2026-06-08-Bprime-value-subtraction-circuit-research.md`'s `snc_drive` relay verbatim — but drive it with the value DERIVATIVE rather than the value LEVEL.** The derivative itself is the two-latency conductance difference (B-1) read off the critic. This composes: `−V` (level, GABA_B, at reward) + `+dV/dt` (derivative, via the disinhibitory relay, at the cue) = the full TD bootstrap, all in conductance/spikes.

**B-2 — strictly-minimal single-conductance leaky-derivative (the cheapest protected edit).** Instead of a *second* conductance, read the existing `g_gabab` as a **leaky derivative**: maintain a slow EMA of `g_gabab` (one extra `cp_conductance_g_gabab_slow` that decays slower) and let the SNc current term use `(g_gabab − g_gabab_slow)`. This is B-1 collapsed to one extra array (the slow EMA of the same channel) — the **minimal** additive edit. Functionally identical observable (band-passed value = derivative). Recommended if the owner wants the smallest possible `sim/` surface.

**B-3 — pure runner-side derivative via the eligibility/EMA (NO protected edit, the fallback).** If a protected edit is undesirable for the first de-risk, the *value trace itself* can be made to rise-and-decay so its **GABA_B onto the SNc is already a transient at value onset**: drive the critic with a **phasic** cue trace (rise at cue, decay) so the critic *fires a burst at cue onset and then adapts* (MSN spike-frequency adaptation / the FS clamp already does this) — the critic's GABA_B is then naturally larger *at the moment value turns on* and smaller once it plateaus, approximating a derivative *without* a second conductance. This is lower-fidelity (it leans on adaptation, not a clean derivative) but it is **zero protected edit** and a legitimate cheap-first probe. Use it to *establish feasibility* before paying for B-1/B-2.

### 2.3 How the critic still LEARNS the cue value (unchanged three-factor)

The critic learns V exactly as today: DA-gated STDP on `place/cue → striosome_value` (eligibility × the SNc-derived `da_signal`). The TD addition changes *what δ the SNc emits* (now r + dV/dt − V instead of r − V), which changes the *teaching signal* the critic itself consumes — which is the actor-critic consistency (catalog C.30: critic and actor consume the same δ). Concretely, as the cue→value weight grows over trials, V(cue-state) rises from 0 toward the reward magnitude; the derivative-at-cue grows (burst migrates onto the cue) and the residual-at-reward shrinks (r − V → 0). **The migration is driven by the critic learning V from the bootstrapped δ — the standard TD(0) dynamic, now in spikes.** The eligibility trace (catalog C.29, already TD(λ) in all but name) supplies the temporal credit assignment across the CS→US interval.

### 2.4 What stays host (and is legitimate)

- **The environment** rendering the cue + the goal-contact event (the world telling the retina "the cue is on" / "the goal is reached") — legitimate (environment/body boundary).
- **The trial clock** that the *protocol* (CS at t0, US at t0+ISI) uses to *schedule* the cue and reward drives — legitimate (this is the world's event timing, exactly like a real conditioning apparatus; the *brain's* job is to learn the prediction, which it does in neurons). Do NOT make the trial clock "spiking" — that would make the apparatus a brain.
- **Everything cognitive** — the value, the derivative, the burst, the dip, the credit assignment — is neural. No host `δ`, no host `γV' − V`, no host EMA of value reaching the SNc.

---

## 3. THE PROTECTED `sim/` EDIT SCOPE (byte-review-ready)

**Headline: ONE additive, guarded, default-OFF byte-identical protected edit** — a second slow inhibitory conductance (the lagging value channel of B-1), OR the single-array slow-EMA of B-2. Everything else (cue trace, pathway tags, SNc/relay drive, the protocol, the validation) is runner-side. (B-3 needs **zero** protected edit and is the first probe.)

### 3.1 The edit (B-1 form), as an exact mirror of the shipped GABA_B block

The GABA_B/GIRK conductance is already an additive, guarded, owner-approved edit (`enable_gabab`, `cp_conductance_g_gabab`, `fused_gabab_decay_and_current`, the per-synapse mask, the guarded current block at `sim/bridge.py:5832-5872`). The TD edit is the **same pattern, a second instance**:

| File / method | What it adds | Mirror of |
|---|---|---|
| `sim/config.py` (+~4 lines) | `enable_td_value_derivative: bool = False`; `td_lag_tau_decay: float = 400.0` (the lagging channel's slower time constant, ms); `td_derivative_gain: float = 1.0` (how strongly the derivative drives the SNc/relay) | the `enable_gabab` / `gabab_tau_decay` / `gabab_propagation_strength` block (`config.py:197-200`) |
| `sim/bridge.py` `__init__` (+~2 lines) | `self.cp_conductance_g_gabab_lag = None` (the lagging value conductance) | `self.cp_conductance_g_gabab = None` (`bridge.py:240`) |
| `sim/bridge.py` alloc (+~3 lines, guarded by `enable_td_value_derivative`) | allocate `cp_conductance_g_gabab_lag = cp.zeros(n)` | the `enable_gabab` alloc (`bridge.py:1223-1228`) |
| `sim/bridge.py` per-step (+~8 lines, guarded) | increment `g_gabab_lag` from the SAME GABA_B-tagged synapses (reuse `cp_gabab_synapse_mask`) with a slower decay; compute the derivative current `I_deriv = td_derivative_gain·(g_gabab − g_gabab_lag)·(E_K − V)` and add it to `total_input_current_pA` (or route it to the disinhibitory relay) | the GABA_B current block (`bridge.py:5838-5872`) + `fused_gabab_decay_and_current` |
| `sim/kernels.py` (0 new; reuse) | reuse `fused_gabab_decay_and_current` for the lagging channel (different `decay` arg) | — |

(B-2 is even smaller: drop the `_lag` increment; instead keep a slow-decaying EMA copy of `g_gabab` and use `(g_gabab − g_gabab_slow)`. One array, one decay constant, one current term.)

### 3.2 The byte-identity argument + how to prove it

**Argument.** The entire new block is gated by `if getattr(cfg, "enable_td_value_derivative", False) and self.cp_conductance_g_gabab_lag is not None:`. The default is `False` and the array is `None` for every run that does not opt in, so the block is **unreached** and `total_input_current_pA` is computed by exactly the same operations as today → bit-for-bit identical neuron dynamics. This is the *same* guard structure already proven byte-identical for `enable_gabab`, `enable_nmda_recurrent`, and `enable_coincidence_detection` (all owner-approved on that argument). The new config fields are read **only** inside the guard.

**Proof (the standard the project uses).** (1) **Static**: the new code is one guarded block + guarded alloc + dataclass defaults; grep shows the new `cfg.*` fields are referenced only inside `if enable_td_value_derivative`. (2) **Dynamic byte-identity**: run a fixed-seed bridge (e.g. the Izhikevich kernel smoke + a small nav warm-up) with the edit present-but-OFF and assert the per-step `cp_membrane_potential_v` / `cp_firing_states` / `cp_connections.data` are bit-identical to the pre-edit commit (a hash over the arrays each step, the exact protocol used for the GABA_B and determinism-transpose edits in `2026-06-10-N9-deterministic-transpose-matvec-byte-review.md`). (3) **R-A (regression-absent)**: the existing test suite (`pytest tests/`) passes unchanged with the edit present + OFF. Commit the edit tagged **FOR OWNER BYTE-REVIEW** with the off==baseline hash evidence, mirroring the GABA_B / GIRK-cap / determinism commits.

### 3.3 Prefer reuse over new arrays (explicit)

- **Reuse** the `cp_gabab_synapse_mask` (the lagging channel is driven by the *same* striosome→SNc synapses — no new mask).
- **Reuse** `fused_gabab_decay_and_current` (the lagging channel is the same decay+current op with a different time constant — no new kernel).
- **Reuse** the eligibility trace + `from_region_firing_signed` DA rule + the three-factor update (the critic learns V exactly as today — no plasticity change).
- **Reuse** the B′-DISINHIBIT `snc_drive` relay (runner-side region/pathway) for the sign-correct derivative-burst delivery (no new protected mechanism).
- The **only** genuinely new state is the one lagging conductance array (B-1) or the one slow-EMA array (B-2).

---

## 4. THE VALIDATION (the decisive test): burst MIGRATION

The validation is a **Pavlovian cue→reward protocol on the spiking circuit**, instrumented for the SNc's time-of-burst across learning. This is catalog C.22/C.28/C.30's acceptance criterion and HS98's experiment.

### 4.1 Protocol (extend `snc_stageb_critic_probe.py` → a `--td` Pavlovian mode)

- Build the minimal bridge: `cue → striosome_value` (plastic critic) + `striosome_value → snc` (GABA_B `−V`) + the **new** value-derivative channel (B-1/B-2) + (for sign) the `snc_drive` disinhibitory relay + `reward_us → snc` (the spiking US) + the DA-from-SNc-firing rule. (All of this is the probe's existing recipe plus the one derivative channel and the sustained cue trace.)
- **A trial** = cue ON at t0 (sustained for the CS→US interval, the A-trace), reward (US via `reward_us`) at t0+ISI (a fixed inter-stimulus interval, e.g. 200–400 ms), then an inter-trial gap. Learning ON (the critic learns V via the SNc δ).
- Run N trials (e.g. 40–80). **Each trial, record the SNc firing-rate time-course** (per-substep windowed rate across the whole CS→US→post window) so the *time-of-peak* is measurable.
- Then a **frozen-learning test block**: predicted (CS+US), unpredicted (US alone), omission (CS, no US), baseline.

### 4.2 Metrics + pass bars

| Signature (Schultz/HS98) | Metric | Pass bar |
|---|---|---|
| **Burst MIGRATES cue← reward** (the headline) | correlation r between **trial number** and **time-of-SNc-peak** (peak should move from US-onset toward CS-onset across learning) | **r > 0.7** (catalog C.22/C.28), sign = peak moves earlier (toward CS) |
| **Early: burst at US** | trial-1–2 SNc peak time ≈ US-onset; rate at US ≫ tonic | peak within the US window early |
| **Late: burst at CS, not US** | late-trial SNc peak at CS-onset; rate at CS ≫ tonic AND rate at US ≈ tonic (the US burst has *transferred*, not just shrunk) | late CS-rate > tonic AND late US-rate ≈ tonic (within noise) |
| **Omission dip stays at REWARD time** | in the omission test, SNc dips below tonic at **t0+ISI** (the expected-reward time), NOT at the cue | dip depth > 0 at reward-time bin; no dip at cue |
| **No burst between cue and reward** | SNc ≈ tonic in the CS→US gap (value flat → derivative ≈ 0) | gap-rate ≈ tonic |
| (regression guard) **state-specific gap retained** | predicted < unpredicted (the R-W `−V` still works) | unpred > 1.30·pred (the existing Stage-B gate) |
| (regression guard) **V learned, cue-gated** | striosome rate on CS rises over trials; cue-gated | v_late > 1.20·v_early; predicted/omission ≫ unpredicted/baseline |

**Graded-partial credit (HS98-faithful).** HS98 found the transfer is **graded with learning rate, not binary** — slow-learned pairs retain reward responses for tens of trials. So a **partial migration** (the burst *shifts toward* the cue and the US burst *shrinks but does not fully vacate*) is a *defensible partial PASS* mapping the slow-learning regime, NOT a failure — report it as such with the r-value and the residual US-burst fraction.

### 4.3 Multi-seed plan

- **De-risk: ≥3 seeds (42/43/44)** on the CPU Pavlovian probe; the migration r-value and the omission-dip-at-reward must hold sign-consistently across all three. (The probe already pins per-seed RNG — harness fix #5.)
- **If de-risk passes: nav 6-seed regression** (flagship A+E+G v2.5 with `--spiking-snc --enable-neural-critic --enable-td-value-derivative …`): the online TD δ must not break nav (summed reward ≥ the R-W Stage-B), and — separately, on a **harder reward-load-bearing task** if available — the cue-shift should be visible in vivo. (Per the current-state doc, the orient-solvable gridworld is *insensitive* to the reward pathway, so the **probe is the sensitive test**; the nav 6-seed is the necessary-not-sufficient regression gate.)

---

## 5. ANTI-CHEAT controls

The whole point of the BRAIN-BASED-ONLY standard: prove the TD error is computed by **neurons**, not host code.

1. **Cue-pathway lesion → no transfer (decisive).** After training, zero the `cue/place → striosome_value` weights (extend the probe's existing `_lesion_pathway`). The migration MUST vanish: the SNc bursts to the US (the innate `reward_us` reflex) but no longer at the cue, and the omission dip disappears. If the transfer survived a cue-pathway lesion, it was host arithmetic in disguise. (Mirror: the existing Stage-B lesion already proves the `−V` subtraction is the synaptic GABA_B.)
2. **Derivative-channel lesion → the cue burst + dip vanish, the `−V` gap remains.** Zero the lagging conductance (or the `snc_drive` relay edge) so only the existing GABA_B `−V` survives. The circuit must revert to **R-W** (state-specific gap + omission-dip-by-subtraction, but NO cue-burst, NO migration). This localizes the cue-shift specifically to the new derivative channel — proving the bootstrap is what the new edit adds.
3. **No host TD computation in the path (provenance assertion).** Assert, under `--enable-td-value-derivative`, that the SNc current is `tonic + reward_us excitation + (synaptic GABA_B −V) + (synaptic derivative via the conductance/relay)` ONLY — no host `δ`, no host `γ·V' − V`, no host EMA of value, no `reward_ema` reaching the SNc. (Grep the SNc-drive block for any value/EMA host term; the only host inputs allowed are the tonic and the *protocol's* cue/US scheduling.)
4. **Unpaired (random-timing) control → no transfer.** Run the protocol with the CS and US **decoupled in time** (US at random offsets unrelated to the CS, or CS and US uncorrelated across trials). The cue must then acquire **no** predictive value → **no** burst migration, **no** omission dip. This proves the transfer rides on the genuine CS→US *contingency* (the critic learning the prediction), not on any cue-present back-channel. (Analogue of the place-shuffle / eccentricity-permute controls already used.)
5. **Coordinate-freedom (inherited).** The cue/state the critic reads is the perceived state (place/pixels), never `(x,y)/(gx,gy)`; combined with the N5 perceived-reward, the whole TD loop references no coordinate. (Carry the existing N9 provenance bar.)
6. **GABA_B/`−V` still synaptic (inherited).** The existing GABA_B-mask lesion (the probe's `_lesion_gabab_mask`) must still collapse the `−V` gap — the TD edit must not silently replace the synaptic subtraction with anything host.

---

## 6. HONEST could-be-NEGATIVE framing + ranked options

### 6.1 The honest risk

A rate-coded spiking critic estimates V **noisily** (the small striosome pool fires a windowed rate, draw-variable — the very non-determinism the determinism edit addresses). The bootstrap `γ·V(s_{t+1}) − V(s_t)` is a **difference of two noisy value estimates**, the classic TD-instability regime. Potjans-Diesmann-Morrison (2011, *Front. Comput. Neurosci.*) studied exactly "an imperfect dopaminergic error signal driving temporal-difference learning" and found it *can* work but is sensitive to the error signal's fidelity. The conductance-derivative (B-1/B-2) further depends on the two time constants being well-separated enough to read a clean derivative without amplifying noise. **So the cue-shift may be noisy, under-migrate, or be unstable.** This is anticipated, not a surprise.

### 6.2 The three honest outcomes (all valid deliverables)

- **(i) Clean migration** (r > 0.7, US burst fully transfers, dip at reward) → **ship** the TD edit; navigation gains the one missing canonical dopamine signature.
- **(ii) Partial / graded migration** (the burst shifts *toward* the cue, the US burst shrinks but does not fully vacate) → **a defensible PASS** that maps the *slow-learning* regime HS98 actually measured (graded transfer, not binary). Report the r-value + residual-US fraction; recommend the **CSC / longer cue trace** (A-CSC, §2.1) or **more trials** to push toward full migration.
- **(iii) No migration** (the conductance-derivative bootstrap is too noisy, or the single-step Δ is too short to register a cue-time burst) → **the negative is the deliverable**: it maps a substrate limit (a small rate-coded critic's value estimate is too noisy for a clean spiking bootstrap; the derivative-of-noise dominates). The recommendation is then the **CSC tapped-delay-line** (back-propagating value one tap per trial is more robust than a single-step derivative) and/or a **denser/slower critic** + the determinism fix for a cleaner V. Per the project standard, "the spiking TD failing to show a clean cue-shift maps a substrate limit and IS a valid deliverable."

### 6.3 Ranked options

| Option | What | Fidelity | P(works) | Protected edit | Verdict |
|---|---|---|---|---|---|
| **TD-DERIV-CONDUCTANCE (B-1)** | two-latency value channel; difference = bootstrap; `−V` via GABA_B + `+dV/dt` via the B′ disinhibitory relay | medium-high (a real conductance derivative; abstracts the value-derivative) | medium | **ONE** (2nd slow conductance, default-OFF, GABA_B mirror) | **RECOMMENDED build** |
| TD-DERIV-EMA (B-2) | single conductance read as leaky derivative (`g − g_slow`) | medium | medium | ONE (1 slow-EMA array) — the **minimal** edit | recommended if smallest surface wanted |
| TD-DERIV-ADAPT (B-3) | phasic cue trace → critic adaptation makes its GABA_B a transient ≈ derivative | low-medium (leans on adaptation) | low-medium | **ZERO** | the cheap-first **first probe** (establish feasibility before any edit) |
| **CSC tapped-delay (A-CSC)** | complete-serial-compound cue chain; TD back-propagates value one tap/trial | **highest** (the literal Montague-Dayan-Sejnowski TD) | medium (more robust migration; harder to learn) | zero protected (runner-side relay chain) but bigger build + slower | **#2 — fidelity comparator / the fix if B-* under-migrates** |
| (NOT TD) host δ = r+γV'−V | compute the bootstrap in Python | n/a | n/a | n/a | **DISQUALIFIED** — a host shortcut; the standard forbids it |

**Recommended path:** run **B-3** (zero-edit, CPU) first to establish that *any* derivative-of-value at the cue produces *any* migration on this critic; if yes, build **B-1/B-2** (the clean conductance derivative, one protected edit) and validate the migration r-value multi-seed; if the single-step derivative under-migrates, escalate to **A-CSC** (the tapped-delay chain) which is the most robust route to a full cue-shift. Present **B-1 (with B-2 as the minimal variant, B-3 as the first probe, A-CSC as the fidelity escalation)** to the owner; run the CPU Pavlovian de-risk before any nav build.

### 6.4 Open questions for the owner

1. **Conductance-derivative (B-1/B-2) vs CSC (A-CSC) first.** Recommendation: B-3 probe → B-1/B-2 → A-CSC only if needed. Confirm the owner accepts the conductance-derivative as a *defensible abstraction* of the value-derivative (it is not a literal tapped-delay reconstruction).
2. **The trial clock stays host.** The protocol scheduling the CS/US in time is the world's event timing (legitimate environment/body). Confirm this is accepted (it is the same status as the goal-contact event in N5).
3. **R-W now SHIPS; TD is the increment.** The existing `−V` GABA_B subtraction is unchanged; TD *adds* the derivative channel. Confirm the TD edit composes on top (it does — `−V` at reward + `+dV/dt` at cue).
4. **Acceptable partial.** Confirm a *graded* migration (HS98 slow-learning regime) counts as a PASS (it should — it is what the biology shows), so a clean-but-incomplete transfer is not scored as a failure.
5. **Sign delivery.** Confirm using the already-designed B′-DISINHIBIT `snc_drive` relay to deliver the *positive* derivative (burst at cue) is acceptable (it reuses owner-approved-research machinery and sidesteps the depolarized-SNc-membrane sign problem).

---

## 7. Sources

### Project code (verified file:line this session)
- Live nav-loop SNc drive (R-W δ = r − V; the `_I_snc` block to extend): `research/runners/g11_bg_runner.py:7107-7143` (`reward_us` r at `:7124-7125`, tonic-only `_I_snc` at `:7126`, host `−k_v·V` already dropped under `enable_neural_critic`).
- Stage-B critic + GABA_B `−V` subtraction + value-leads-reward read window: `g11_bg_runner.py:5543-5618` (`_critic_rate`, `_snc_burst_rate`, `critic_lead_steps`); the `striosome_value → snc` GABA_B pathway `:1880-1905`.
- `reward_us` PPN-like US afferent (the spiking r): `g11_bg_runner.py:1144-1166, 1897-1905`; nav-loop drive `:7109-7125, 7184-7185`.
- Place self-org + `place_fs` + critic FS-clamp: `g11_bg_runner.py:1175-1232, 1783-1841`.
- GABA_B/GIRK conductance (the mechanism to mirror): `sim/bridge.py:240-242` (state), `:1223-1228` (alloc), `:2356-2386` (mask), `:5832-5872` (per-step current); kernel `fused_gabab_decay_and_current` (`sim/kernels.py`, imported `bridge.py:89`); config `sim/config.py:197-200, 246` (`enable_gabab`, `gabab_tau_decay`, `gabab_propagation_strength`, `gabab_conductance_max`).
- NMDA-recurrent slow excitatory channel (the cue-trace option A-trace): `sim/bridge.py:5731-5804`; config `coincidence_*` + `enable_nmda_recurrent` neighbours `sim/config.py:150-200`.
- Eligibility trace + three-factor + signed-DA rule: `sim/bridge.py:616` (trace); `sim/neuromodulators.py:774-817` (`from_region_firing_signed`, the EMA the SNc-DA broadcast uses).
- The CPU de-risk probe to extend (4-gate falsifier, lesion anti-cheats, per-seed RNG pin): `research/runners/snc_stageb_critic_probe.py` (R-W-vs-TD scope note `:11-34`; gates `:548-554`; `_lesion_pathway` `:358-374`; `_lesion_gabab_mask` `:377-394`).

### Project feature catalog (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`)
- **C.22** Dopamine RPE / Schultz cue-shift — the migration is "the canonical signature… project reproduces only sign (a)"; HS98 quantitative criteria (cue-shift r, omission dip 99±29 ms, late/early temporal-PE); the math `r̂(t)=r(t)+γP(t)−P(t−1)`: `:907-921`.
- **C.28** TD error δ = r + γV(s′) − V(s) — "partial, gap is measurable"; the project EMAs r but "never bootstraps from a learned V(s′)"; acceptance = monotone cue-shift (r>0.7) + omission dip: `:565-575`.
- **C.29** Eligibility traces / TD(λ) — "implemented" (TD(λ) in all but name); the temporal credit-assignment substrate: `:577-585`.
- **C.30** Actor-critic — "actor implemented, critic missing"; acceptance = cue-shift + omission dip; striosome=V(s), SNc=δ, matrix=actor: `:587-599`.
- **C.31** Bootstrapping vs Monte Carlo — why phasic DA MUST bootstrap (single-trial shift, no episode-end wait): `:601-611`.
- **C.33** PPN→DA — the cue-shift driver; "adding a small PPN region… would let the project model the cue-shift transfer dynamic" (already built as `reward_us`): `:627-637`.

### Peer-reviewed literature
- **Schultz W., Dayan P., Montague P.R. (1997)** "A neural substrate of prediction and reward", *Science* 275:1593 — the cue-shift / TD-dopamine result.
- **Montague P.R., Dayan P., Sejnowski T.J. (1996)** "A framework for mesencephalic dopamine systems based on predictive Hebbian learning", *J. Neurosci.* 16:1936 — the TD model of dopamine + the complete-serial-compound (CSC) state representation that makes TD reproduce the cue-shift.
- **Hollerman J.R. & Schultz W. (1998)** "Dopamine neurons report an error in the temporal prediction of reward during learning", *Nat. Neurosci.* 1:304 — the graded cue-shift + omission dip + temporal-prediction-error quantitative criteria (the catalog C.22 validation numbers).
- **Schultz W. (1998)** "Predictive reward signal of dopamine neurons", *J. Neurophysiol.* 80:1 — `r̂(t)=r(t)+γP(t)−P(t−1)`; cue-shift; omission dip.
- **Sutton R.S. & Barto A.G.** *Reinforcement Learning* 2e — Ch 6 (TD prediction / bootstrapping), Ch 7 (eligibility traces / TD(λ)), Ch 11 (actor-critic), Ch 12 (the stimulus-trace / CSC for the cue-shift).
- **Eshel N. et al. (2015)** *Nature* 525:243 — the SNc computes δ by subtractive inhibition (the project's `−V` GABA_B basis; the value subtraction the derivative composes on top of).
- **Potjans W., Diesmann M., Morrison A. (2011)** "An imperfect dopaminergic error signal can drive temporal-difference learning", *Front. Comput. Neurosci.* — the honest-negative reference: a noisy spiking RPE *can* drive TD but is fidelity-sensitive (the §6.1 risk).
- **Frémaux N., Sprekeler H., Gerstner W. (2013)** "Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons", *PLoS Comput. Biol.* 9:e1003024 — the canonical spiking actor-critic (a spiking critic estimates V; the TD error modulates reward-STDP) the design follows.
- **Kandel et al.** *Principles of Neural Science* 6e — Ch 43 (dopamine/reward, Fig 43-2, the Schultz cue-shift figure).

---

**Deliverable path:** `E:\Documents\Projects\sim\docs\plans\2026-06-10-N9-TD-cue-shift-design.md`
