# Striatal value-critic FIRING + LEARNING in deterministic nav — deep-research scoping

**Date:** 2026-06-08
**Type:** read-only deep-research findings (standing practice: research + catalog review BEFORE committing build/GPU resources). NO `sim/` edits; two short CPU read-only diagnostics (`SIM_BACKEND=numpy`).
**Roadblock being scoped:** `2026-06-08-nav-placecritic-calibration-NEGATIVE.md` — the `striosome_value` MSN-D1 critic cannot be made to both FIRE and LEARN a place-graded value in deterministic nav while keeping the actor sane.
**Predecessors:** `2026-06-08-gabab-girk-stageB-derisk-GO.md` (the GABA_B/GIRK subtraction — SHIPPED, validated, NOT in question); `snc_stageb_critic_probe_place.py` (the place-code/value-leads de-risk that PASSED on CPU).
**Sources reviewed:** catalog `feature-catalog.md` B.02 / B.14 / B.15 / B.17 / B.07 / C.30 / C.31 / O.03 / O.16 / O.18 (sim-catalog worktree); Kandel 6e ch 38/43; the runner's place/critic code; the de-risk probe; current literature (van der Meer & Redish 2009; Lansink 2009; van der Meer 2011 theta-precession).

---

## TL;DR (the decision this doc forces)

The NEGATIVE finding framed the wall as "three coupled layers (firing / learning / collateral) with a determinism-vs-up-state tension." **The two read-only diagnostics below relocate the dominant wall: it is NOT the MSN intrinsic rheobase, NOT fundamentally the missing OU noise, and NOT the determinism constraint. It is the AFFERENT CONDUIT — the sparse `sensor_place_readout` place code (~1–3 active cells) delivers only ~2–5 Hz into the critic, no matter how the neuron or noise is tuned.** And the **faithful** biology (catalog B.02: the MSN up-state is *excitation-driven by convergent input*, E/I 2–5× excitation-dominant — NOT noise-driven; Wilson & Kawaguchi 1996) says the OU crutch was always a non-faithful stand-in. The diagnostics show a **dedicated dense place afferent fires the MSN-D1 critic at 22–49 Hz with OU OFF**, no actor-perturbing 1500 pA drive, no noise. So the recommended fix needs no determinism relaxation and no per-region noise edit: it is a **dedicated dense ventral/dorsal place input that feeds ONLY the critic**, plus the already-validated value-leads-reward timing. This is reusable runner-side machinery on top of the shipped GABA_B substrate. An honest negative remains possible and the de-risk is designed to fail gracefully under the exact deterministic regime if the dense afferent does not in fact carry it.

---

## 1. Diagnosis — the three coupled layers, re-grounded in the biology + the diagnostics

The NEGATIVE finding's three layers are real, but the diagnostics re-rank which is load-bearing.

### Layer 1 — FIRING. (Re-diagnosed: the wall is the afferent, not the neuron, not the noise.)

The MSN-D1 critic (`striosome_value`, `IZH2007_STRIATAL_MSN_D1`: `vr=−80, vt=−25` — a 55 mV rest-to-threshold gap, `b=−20` mimicking the KIR2 hyperpolarized clamp; `sim/enums.py:671`) is intentionally hard to fire — that is the B.02/B.14 biology (KIR2 holds RMP at −80…−95 mV; the cell is silent at rest and thresholds only strong consensual input). The NEGATIVE finding attributed the silence to "rheobase ~700 pA unreachable + OU off." The diagnostic (`_strio_critic_firing_diag.py`, OU OFF) refines this:

| Drive route (OU OFF) | striosome firing |
|---|---|
| **Direct** constant current into the critic | 200 pA → 13 Hz, 700 pA → 45 Hz, 1500 pA → 95 Hz (graded from ~150 pA) |
| Through the **w=3.0 / density 0.6 afferent**, place region @2500 pA (saturated 496 Hz) | **only 25.9 Hz** |
| Through that afferent with only **1–3 active place cells** (the nav code) | **2.2 / 3.7 / 5.4 Hz** |

So the MSN itself fires gradedly from ~150 pA of *effective* current; the **"~700 pA" is the current needed AT the soma, and the afferent at w=3.0 with a handful of active cells delivers far less than that.** The decisive variable is the conduit (`_strio_critic_afferent_diag.py`, OU OFF):

| Lever (OU OFF) | striosome firing |
|---|---|
| **#active presynaptic cells** (each @1500 pA, w=10): 1 / 3 / 10 / 20 / 40 | 2.2 / 5.4 / 16 / 32 / **61 Hz** |
| **afferent weight** (all 40 active @1500 pA): 3 / 6 / 10 / 25 | 20 / 38 / 61 / **151 Hz** |

The sparse nav place code (`sensor_place_readout`: σ=0.5 at grid 32, ~1–3 cells active, `IZH2007_HIPPO_PYRAMIDAL`, instrumented ~0.57 Hz in the NEGATIVE finding) sits at the **2–5 Hz floor of that table.** OU's role is now clear from the diagnostic: OU at 0 direct drive gives only **1.67 Hz** — OU alone is NOT the up-state; in the probe, OU added a small lift on top of a strong dense+grown drive. **The up-state in biology is convergent excitation, not noise (B.02 supplemental: "striatal Up states are excitation-driven; inhibition in striatum is *not* the gate for entering Up state"). The OU crutch was a non-faithful surrogate; removing it for determinism was correct, and a dense excitatory afferent is the faithful replacement.**

### Layer 2 — LEARNING. (The LTD is a timing artifact of continuous post-before-pre; biology gates it.)

When the critic was forced to fire under strong continuous drive, the `sensor_place_readout → striosome_value` weight underwent net **LTD** (NEGATIVE finding v2: 25.10 → 24.44). Cause: a continuously-active afferent into a fast-firing post gives predominantly post-before-pre STDP ordering → depression. The biology (O.03 / C.30 / Schultz98 three-factor): corticostriatal plasticity is **dopamine-gated and eligibility-trace-based** — `Δw = η · r̂ · h(i,o)`, where the sign is set by the DA signal and the trace is a *fast-decaying coincidence tag*. The place→value association forms specifically because **the place LEADS reward** (the value-leads-reward finding `d0416fc3`): the place ensemble is active for many steps *approaching* the reward site, builds eligibility, and the SNc-derived δ converts that tag to LTP at reward. The de-risk's `_clear_eligibility` + held-out-FAR design reproduces exactly this: it clears the tag at the start of each NEAR window so only the near co-firing × positive δ potentiates (location-selective LTP), and the FAR ensemble is never paired with reward → stays low. **The learning layer is solved by the de-risk's protocol (eligibility-clear + place-leads-reward), provided Layer 1 lets the critic fire enough to co-spike with the afferent.** It is downstream of the firing fix.

### Layer 3 — COLLATERAL. (Perturbation came from sharing the actor's afferent + raising its drive.)

Nav regressed 2.16 → 3.24 when the critic was fired by **doubling the shared place+goal drive to 1500 pA** — i.e., the actor and critic read the SAME `sensor_place_readout`, so cranking it to fire the critic also over-drove the actor's `sensor_place_readout → cortex_X` pathways. This vanishes if the critic gets its **own dedicated afferent** (a separate region driving ONLY `striosome_value`), leaving the actor's place input untouched. Diagnostic (G) confirms a dedicated dense afferent fires the critic to 22–49 Hz with **no direct drive on the actor's pathway at all.**

**Re-ranked:** Layer 1 (afferent conduit) is the load-bearing wall; Layers 2 and 3 are consequences of how Layer 1 was forced (shared sparse afferent + brute drive). Fix Layer 1 with a *dedicated dense* afferent and Layers 2–3 are addressed by construction + the de-risk protocol.

---

## 2. Ranked biology-grounded options per layer, and how they COMPOSE

Notation: **fidelity** (how biologically faithful) · **P(works)** (likelihood, given diagnostics) · **surface** (runner-side vs protected `sim/` edit).

### Layer 1 — FIRING (give the MSN-D1 a faithful up-state)

| # | Option | Fidelity | P(works) | Surface |
|---|---|---|---|---|
| **1A ★** | **Dedicated dense place afferent feeding ONLY the critic.** A new region (e.g. `vs_place_context`, 150–200 cells, σ widened for grid 32 so 30–80 cells fire per position, a denser/less-adapting type than HIPPO_PYRAMIDAL), with a `vs_place_context → striosome_value` plastic afferent (density ~0.5, w ~6). Does NOT touch the actor's `sensor_place_readout`. | **High** — this IS the biological up-state (convergent excitation; B.02) AND the correct anatomy (dense HC/subiculum → ventral striatum value cells; van der Meer & Redish 2009; Lansink 2009). | **High** — diagnostic (G): 22–49 Hz with OU OFF, no actor drive. | Runner-side (new region + pathway + a drive injection mirroring the existing place injection). NO `sim/` edit. |
| 1B | **Widen + densify the EXISTING `sensor_place_readout`** (raise n, σ for grid 32, swap to a less-sparse type) and let the critic read it. | Medium — fixes sparsity but re-couples actor+critic (Layer 3 risk) and changes the actor's perception. | Medium | Runner-side, but perturbs the validated actor. |
| 1C | **Per-region OU / background-noise knob** (OU-on-for-the-critic-only, or a `background_drive_per_region`). | Low–Medium — a *noisy* up-state is the LESS faithful model (B.02: noise is not the Up-state gate); also sacrifices strict determinism OR needs a protected `sim/` edit. | Medium | Protected `sim/` edit (per-region noise) OR relax determinism. **Not recommended** — the diagnostics show it is unnecessary. |
| 1D | **More-excitable critic neuron type** (`--critic-neuron-type`, already wired): HIPPO_PYRAMIDAL / RS / THALAMIC_RELAY fire 2–4× harder per pA (diagnostic D). | Low — a striatal value cell is an MSN; making it RS/relay is a fidelity regression (loses the KIR2 thresholding that is the point of B.02). | Medium (helps, doesn't fix the sparse-afferent floor alone) | Runner-side flag (exists). Use only as a *secondary* knob, not the primary fix. |

### Layer 2 — LEARNING (LTP not LTD; place-leads-reward, DA-gated)

| # | Option | Fidelity | P(works) | Surface |
|---|---|---|---|---|
| **2A ★** | **Value-leads-reward eligibility protocol** (the de-risk's design): the dense place ensemble is active for a LEAD window approaching the reward site; clear the eligibility tag at window start; SNc-derived δ converts near-co-firing to LTP at reward; FAR/low-value locations never pair with reward → stay low. | **High** — exactly Schultz98 `r̂ · h(i,o)` + the anticipatory-ramp value-cell physiology (van der Meer & Redish 2009: ramps precede reward; Lansink 2009: HC leads VS). | **High** — the de-risk PASSED 3/3 with this; the wall was only the *port*. | Runner-side (the eligibility-clear + LEAD already exist as probe machinery; the nav loop needs a place-leads-reward window). |
| 2B | **Three-factor as-is, no lead** (place + reward simultaneous). | Medium | Low — this is what produced the LTD (post-before-pre under continuous drive). | Runner-side. **Rejected** — reproduces the NEGATIVE. |
| 2C | **Opposite-sign D2 plasticity / explicit TD bootstrap** (O.03 supplemental; C.31). | High (longer-term fidelity) | n/a for THIS gate | Larger; orthogonal future increment, not needed to clear firing+learning. |

### Layer 3 — COLLATERAL (don't perturb the actor)

| # | Option | Fidelity | P(works) | Surface |
|---|---|---|---|---|
| **3A ★** | **Dedicated afferent (= 1A) ⇒ the actor's place code is untouched.** | High | High (by construction — the fire-drive never lands on the actor's pathway). | Runner-side (falls out of 1A). |
| 3B | Keep shared afferent but cap/scale the actor's place→cortex via a transmission gate during critic-drive. | Medium | Medium (fragile coupling) | Runner-side; more moving parts. |

### ★ THE SINGLE RECOMMENDED COMBINATION: **1A + 2A (+ 3A free)**

A **dedicated dense place context input** (`vs_place_context`, ~200 cells, grid-32 tuning, density ~0.5, w ~6 onto `striosome_value`) that feeds ONLY the critic (1A) → fires the MSN-D1 up-state by convergent excitation with OU OFF (no noise, no determinism break), and by construction leaves the actor's `sensor_place_readout` untouched (3A). Train it with the **value-leads-reward eligibility protocol** (2A) so the place→value weight undergoes location-selective LTP, not LTD. The critic→SNc subtraction is the already-shipped, already-validated **GABA_B/GIRK** route through `critic_snc_window`. This combination is entirely runner-side on top of protected machinery that already exists — **no new `sim/` edit is required for the recommended path.**

---

## 3. Reusable project machinery (exists) vs genuinely new

**Exists / reuse (the heavy lifting is done):**
- **GABA_B/GIRK conductance** (`enable_gabab`, `receptor="gaba_b"`, `cp_conductance_g_gabab`) — SHIPPED, byte-identity-verified, the value-subtraction substrate. `striosome_value → snc` is already wired `receptor="gaba_b"` (`g11_bg_runner.py:1435–1442`).
- **`transmission_gate="critic_snc_window"`** — the bounded value-leads-reward LEAD window on the critic→SNc route already exists; the runner opens it ~1τ before reward (`d0416fc3`). This is the timing primitive 2A needs.
- **The three-factor pipeline** — STDP eligibility × SNc-firing-derived δ (via the `from_region_firing_signed` dopamine rule) → DA-gated weight change. Reused verbatim; the critic learns through it (the de-risk proves it).
- **The de-risk probe `snc_stageb_critic_probe_place.py`** — already implements the place-population-code drive, the eligibility-clear, the held-out-FAR location-selective LTP, the LEAD sweep, the lesion + A/B + host-EMA anti-cheats, AND a graceful PASS/FAIL gate. **It is the de-risk harness; the only change needed is to run it in the deterministic-nav regime (§4).**
- **The place-drive injection pattern** (`g11_bg_runner.py:4758–4760`: per-neuron Gaussian over preferred position) — copy it for `vs_place_context` with grid-32 tuning.
- **The calibration knobs** (`--critic-neuron-type`, `--critic-afferent-weight`, `--hippocampus-drive-sigma`, `--hippocampus-drive-max-pa`) — exist; 1A adds an analogous `--vs-place-*` set or reuses these on the new region.
- **`BrainRegion` / `RegionPathway`** with `plasticity_gate`, `transmission_gate`, `syn_reversal_potential_i_override` — all the declarative wiring 1A needs.

**Genuinely new (runner-side only):**
- The `vs_place_context` region + `vs_place_context → striosome_value` pathway in `build_bg_brain_regions` (dense, grid-32-tuned, feeds ONLY the critic).
- A drive-injection block in the nav loop that renders the agent's perceived position into the dense place code each step (mirrors the existing `sensor_place_readout` injection; this is legitimate host code under the BRAIN-BASED-ONLY rule — it is the *sensory rendering of the world's state into the neural place input*, exactly like the existing place injection).
- A place-leads-reward eligibility window in the nav loop (open the gate / clear eligibility on goal-approach; the dwell-before-reward provides the LEAD).

**NOT needed for the recommended path:** any `sim/` edit (no per-region noise knob, no determinism change). A per-region noise knob (1C) is a *fallback only if* the dense-afferent de-risk fails — and the diagnostics indicate it will not.

---

## 4. Recommended cheap-first de-risk — REPLICATING THE DETERMINISTIC-NAV REGIME

**This is the load-bearing methodological requirement.** The arc has now hit THREE probe-vs-deployment gaps (the GABA_B mask bug, the simultaneous-timing artifact, and this OU-on / dense-drive gap) — every one because the de-risk diverged from deployment. The de-risk MUST replicate the deterministic-nav regime exactly on the axes that bit us.

### What it must replicate (and why that closes the gap)

| Deployment condition (nav, deterministic) | De-risk MUST set | Why (which gap it closes) |
|---|---|---|
| `enable_ou_process = False` (runner line 3342) | **OU OFF** in the probe's `CoreSimConfig` (the probe currently leaves it ON by default — THE gap) | The probe's critic fired partly on OU lift; deployment has none. This is the gap the NEGATIVE finding identified. |
| `enable_conductance_noise = False` (line 3343) | conductance noise OFF | same background-depolarization class |
| `enable_homeostasis/STP/heterogeneity = False` | match | the probe already disables STP/heterogeneity; add homeostasis-off |
| Critic afferent is the **fix-under-test**, NOT the sparse `sensor_place_readout` | the **dedicated dense `vs_place_context`** (N≈200, density 0.5, w≈6, grid-32 σ giving 30–80 active cells) — i.e. de-risk the ACTUAL proposed afferent, not the probe's tuned 40-cell/2500 pA bump | The probe's dense 2500 pA bump is not what nav delivers; de-risk the dense *dedicated* afferent that IS the proposed deployment, with OU OFF. |
| Actor reads its own `sensor_place_readout`, undisturbed | the de-risk **must include the actor's place pathway and assert it is NOT perturbed** (place→cortex firing within tolerance of the no-critic baseline) | closes Layer 3 — proves the dedicated afferent doesn't leak onto the actor. |
| GABA_B physiological operating point | `--nav-derisk` preset (`gabab_propagation_strength≈0.02`, tonic 180 / reward 300, lead sweep) | already the validated live-SNc operating point. |

### The de-risk, concretely (CPU, `SIM_BACKEND=numpy`, serial — leave the webapp GPU PID 29576 alone)

Extend `snc_stageb_critic_probe_place.py` (or a thin sibling) so the bridge it builds:
1. Sets `enable_ou_process=False` + `enable_conductance_noise=False` + homeostasis off — **the deterministic-nav regime.** (One-line change; the probe already pins seeds.)
2. Replaces the 40-cell place region with the **dedicated dense `vs_place_context`** (N≈200, density 0.5, w≈6, grid-32-realistic σ → 30–80 active cells per location), and ALSO instantiates an actor stub (`sensor_place_readout → cortex_X`) reading the *separate* sparse code, so the no-perturbation assertion is testable.
3. Runs the existing value-leads-reward LEAD sweep (`--lead-sweep 0,100,150,200,300,400,500`) + the existing gates:
   - **(1) V-learned-spatial** — V(near) rises across training AND ends > V(far) (graded value-of-location), OU OFF.
   - **(2) state-specific RPE above floor** — far(unpredicted) burst > 1.30× near(predicted) AND far ≥ 10 Hz, at a nav-realistic lead.
   - **(3) location-selective LTP** — near-ensemble place→strio weight grows from init AND grows more than the held-out far ensemble (the value-leads protocol; proves LTP not LTD).
   - **(NEW 4) actor-not-perturbed** — the actor's `sensor_place_readout → cortex_X` firing with the critic present is within tolerance (e.g. ±10%) of the critic-absent baseline.

### Graceful FAIL contract (the honest-negative requirement)

The de-risk **must FAIL under the deterministic regime if the dense afferent doesn't actually carry it.** Concretely: if, with OU OFF and the dense dedicated afferent, gate (1) or (3) fails (V stays place-blind, or the weight undergoes LTD), the verdict is **FAIL** and the conclusion becomes "the deterministic-nav constraint and the MSN up-state are in genuine tension; the faithful fix requires a protected per-region noise/up-state `sim/` edit (1C) or relaxing determinism (run more seeds)." That itself decides the path and is a valid BRAIN-BASED-ONLY deliverable (it maps what the substrate can/can't do on its own). The gate is already wired to print PASS/FAIL — the requirement is to run it in the OFF regime and on the dense *dedicated* afferent, not the probe's tuned bump.

**Decision rule:** PASS (≥3 seeds robust gates 1–3 + gate 4) → proceed to the nav 6-seed regression gate (flagship A+E+G v2.5 + `--spiking-snc --enable-neural-critic` + the dedicated `vs_place_context`, acceptance = summed reward ≥ Stage A; an honest negative there is still a deliverable). FAIL → bank the negative and escalate to 1C (protected per-region up-state edit) or determinism relaxation as a separately-scoped, owner-steered decision.

---

## 5. Anti-cheat controls (carry the de-risk's, add the deterministic-regime ones)

The probe already implements three; keep all and add two:

1. **Place is a POPULATION CODE, not a coordinate** (carry from probe, anti-cheat (a)): the critic's afferent is a K-cell place ensemble; different locations activate different ensembles (asserted via Jaccard overlap < 0.5). The dense `vs_place_context` must satisfy the same provenance check — a per-neuron Gaussian over preferred position, NOT a scalar/coordinate handed to a formula. (Under BRAIN-BASED-ONLY: rendering position → place-cell current is legitimate sensory rendering, identical in kind to the existing `sensor_place_readout` injection.)
2. **Conductance / critic lesion** (carry, anti-cheat (b)): zero `cp_gabab_synapse_mask` after training → the state-specific SNc gap must VANISH (SNc bursts to reward at both near and far). Proves the subtraction is carried by the GABA_B conductance, not host arithmetic. `current_reward_signal` stays 0.0 throughout (no host reward reaches the SNc).
3. **A/B vs host-value Stage A** (carry, anti-cheat (c)): (i) same circuit `receptor="gaba_a"` must FAIL the gap (reproduce the depolarized-SNc wall); (ii) a global reward-EMA value is place-BLIND (identical near vs far → host gap = 1.0 by construction) — the neural place gap is the discriminator that proves the value is BOTH neural AND spatial.
4. **(NEW) Deterministic-regime fidelity** — assert OU + conductance noise are OFF in the de-risk bridge (the exact knobs nav disables). If a future edit re-enables them, the de-risk no longer replicates deployment and the gate is void. (Guards against the gap recurring.)
5. **(NEW) Actor-not-perturbed** — the actor's place→cortex firing with vs without the critic must match within tolerance (gate 4 above). This is the anti-cheat against "fired the critic by secretly over-driving the actor's perception" — the Layer-3 collateral that degraded nav 2.16→3.24.

**Graceful-failure clause (explicit):** under the deterministic regime, if the dense afferent does not produce a learned place-graded value, the de-risk reports FAIL — it must not "rescue" the result by re-enabling OU, raising the drive onto the actor's pathway, or driving the critic directly. Any of those reintroduces a probe-vs-deployment gap and is disallowed.

---

## Honest bottom line

The NEGATIVE finding was correct that the critic-as-built doesn't port — but the diagnostics show the dominant cause is the **sparse afferent conduit**, not an irreducible determinism-vs-up-state tension. The faithful biology (excitation-driven MSN up-state; dense HC→ventral-striatum value coding with anticipatory ramps) and the diagnostics agree that a **dedicated dense place afferent feeding only the critic, trained with the value-leads-reward eligibility protocol, fires and learns a place-graded value with OU OFF and no actor perturbation** — entirely runner-side on top of the shipped GABA_B substrate, no `sim/` edit required. The cheap-first de-risk is the existing place probe run in the deterministic-nav regime on the dense *dedicated* afferent, with a graceful-FAIL contract that, if it fails, decides for the protected-noise-edit / determinism-relaxation path instead. Recommend the controller present 1A+2A and run the deterministic-regime de-risk before any nav build.

---

### Diagnostic artifacts (read-only, CPU)
- `research/findings/raw/_strio_critic_firing_diag.py` — MSN-D1 rheobase + OU-on/off + steady-excitatory-afferent + neuron-type comparison.
- `research/findings/raw/_strio_critic_afferent_diag.py` — afferent weight sweep + #active-presynaptic-cells sweep + dedicated-dense-afferent (the recommended-fix proof: 22–49 Hz, OU OFF).

### Sources (literature)
- [Triple Dissociation / hippocampus→ventral striatum place–reward (Lansink 2009, *PLoS Biol*): Hippocampus Leads Ventral Striatum in Replay of Place-Reward Information](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2717326/)
- [van der Meer & Redish 2009, covert expectation-of-reward at decision points in rat ventral striatum](https://frontiersin.org/articles/10.3389/neuro.07.001.2009/full)
- [van der Meer et al. 2011, Theta phase precession in rat ventral striatum links place and reward (*J Neurosci*)](https://www.jneurosci.org/content/31/8/2843.full)
- [Ventral striatum: a critical look at models of learning and evaluation (review, *PMC*)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134536/)
- [Hippocampal projections to the ventral striatum: from spatial memory to motivated behavior](https://link.springer.com/chapter/10.1007/978-3-7091-1292-2_18)
- Catalog (sim-catalog `references/feature-catalog.md`): B.02 (MSN bistability + up-state E/I budget), B.14 (MSN depolarized E_GABA / shunting), B.15 (SNc lacks KCC2), B.17 (Sp-Sp dendritic linearization), B.07 (patch/striosome ↔ limbic DA), C.30 (actor-critic: striosome = V(s)), O.03 (DA three-factor), O.18 (the explicit "50-neuron striosome V(s)" prescription). Kandel 6e ch 38, 43.
