# N9 — making the two remaining HOST pieces of the reward loop spiking: (A) a US/reward→SNc chain driven by PERCEPTION, and (B) physiological-range critic firing by pooled FS inhibition instead of the GIRK cap

**Date:** 2026-06-10
**Type:** read-only deep-research + catalog/Kandel/literature review. NO code edited, NO runs launched (CPU or GPU). Standing practice: research BEFORE committing build/GPU resources.
**Scope:** the two host-computed pieces the owner flagged in the otherwise-all-neural N9 RPE loop (`build_bg_brain_regions(enable_neural_critic=True, spiking_snc=True, neural_place_selforg=True)` in `research/runners/g11_bg_runner.py`):
- **(A)** the reward burst `r` onto the SNc is **host-injected** (`cp_external_input_current[snc] = snc_tonic + snc_reward_gain*max(0,reward)`, `g11_bg_runner.py:6733-6750`). `reward` is the environment's goal-reach scalar. Make the **US→DA drive** spiking, riding on the agent's PERCEPTION of the goal (composing with N5).
- **(B)** the MSN-D1 value critic **over-fires (~125 Hz)** on some place-code draws → over-clamps the SNc → binary (not graded) δ. The current patch is a **GIRK conductance cap** (`cfg.gabab_conductance_max`) — legitimate biophysics but it masks the symptom. Keep the critic **physiological (~1–20 Hz) and GRADED** via real spiking inhibition.

**Sources reviewed:** catalog `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (**C.33** PPN sensory+reward→DA — the keystone for A; **C.23** heterogeneous DA; **O.16** NAc reward hub; **B.06** striatal PV-FSI feedforward inhibition; **B.07** patch/striosome→SNc; **C.16/C.22/C.30** DA/RPE/actor-critic; the **C.30/O.18** striosome-V(s) prescription; **B.02/B.14/B.15** MSN up-state + SNc-lacks-KCC2); Kandel 6e (`E:\Documents\Projects\sim\references\textbooks\kandel-pns-6e\full-book.pdf` present) ch 38 (BG), ch 43 (reward/DA); the runner's reward + critic + place code; the two sibling docs that already cover adjacent ground (see next); web literature (Watabe-Uchida 2012 inputome; Tian-Uchida 2016 distributed RPE; Eshel 2015 arithmetic subtraction; Hong-Hikosaka LHb→RMTg; LH→VTA; Lee 2017 / Owen 2018 FS feedforward control; Carandini-Heeger 2012 normalization).

> **Relationship to the two sibling docs (read them first — this doc is the COMPLEMENT, not a rehash).**
> - `2026-06-08-striatal-value-critic-firing-research.md` solved the **under-firing / learning** problem (dedicated dense place afferent + value-leads-reward eligibility). That doc is about getting V to *fire and learn at all*.
> - `2026-06-10-N9-placecode-reproducibility-robustness-research.md` is about the **run-to-run draw variance** (the transpose-SpMV non-determinism) and ranked **B1 per-region homeostasis / B2 synaptic scaling / B5 divisive normalization** to make the critic robust to draw strength. Its **B5 (Carandini-Heeger divisive normalization via pooled inhibition) was explicitly DEFERRED** with the note "the FS-PING pool is already a normalization substrate."
> - **This doc opens the two pieces neither sibling addressed head-on:** (A) the **US/reward→SNc spiking chain** (entirely new — neither sibling touches the reward *afferent*; both assume the host reward burst), and (B) the **specific FS→critic divisive-inhibition design** that the robustness doc deferred as B5, with the load-bearing skeptical question the prompt poses: *can pooled inhibition actually normalize an all-or-none coincidence-plateau-driven MSN, or does the plateau defeat rate normalization?*

---

## 0. TL;DR / recommendation

**(A) — feasible and biologically prescribed; runner-side; do it.** The catalog entry **C.33 literally prescribes the implementation**: "Adding a small PPN region (e.g. 30–50 neurons) projecting to the dopamine pool, receiving sensory cue inputs, with plastic synapses gated by reward delivery." The faithful US→DA biology (PPN/PBN glutamate + LH; Watabe-Uchida 2012: SNc gets strong **excitatory** drive from sensorimotor cortex + STN, VTA from LH) is an **excitatory afferent population that fires on the SENSED primary reward** and projects to the SNc — which is *exactly* the shape of the existing host current write. Replace the host `snc_reward_gain*max(0,reward)` term with a spiking **`reward_us` region** (PPN-like, ~40 cells) that is driven when the agent **perceives goal-contact** — read from the **same pixel-sourced goal eccentricity N5 already computes** (eccentricity ≈ 0 = on/at goal). The drive into `reward_us` is legitimate host code **only** because it is *sensory rendering of the world into a neural input* (identical in kind to the existing place/retina injection); the **reward VALUE, the burst, and the r−V subtraction are then all neural**. Anti-cheat: no coordinate/distance enters the US logic — it rides on `sc_salience_offset_from_image` (pixels), and a **goal-cue-shuffle / eccentricity-permute control** must abolish the US burst. **One honest residual that may stay host: the exact *instant* of goal-contact** (the environment's collision check) — but that is the *body/environment boundary* (the world telling the retina "the goal cue is now centred"), not a cognitive computation, so it is legitimately host (see §A.4).

**(B) — partially feasible, with a real caveat the prompt anticipated; mostly runner-side; pursue the FS→critic route but expect the HONEST answer to be a HYBRID.** A spiking **`place_fs → striosome_value` feedforward-inhibition pathway** (the FS-PING pool `place_fs` already exists and already gamma-synchronizes the volley; it currently inhibits **only `place`, not the critic** — adding the critic edge is a one-line `RegionPathway`, **runner-side, no `sim/` edit**) is the biologically correct gain-control substrate (B.06; Carandini-Heeger 2012 via pooled inhibition; the FS pool scales with the volley size, so more-active draws → more inhibition → a **divisive** effect). **BUT the skeptical question has a real edge:** the critic's drive is the **coincidence dendritic plateau**, which the engine implements as an explicit **"SUPRALINEAR all-or-none switch"** on the coincident-input count (`sim/bridge.py:5749`, `fused_coincidence_plateau`). Pooled somatic GABA_A inhibition can **raise the effective threshold** (gate WHETHER the plateau triggers, and clamp the *post-plateau* spike rate via shunting + hyperpolarization) — that **does** convert a 125 Hz runaway into a bounded rate — but it **cannot smoothly divide an already-fully-triggered all-or-none plateau current** the way classic divisive normalization divides a graded drive. So the faithful, defensible outcome is: **(i)** FS→critic inhibition holds the critic in a physiological *rate* range across draws (the over-firing fix — likely works), **but (ii)** the *graded-δ* requirement is carried not by dividing the plateau but by the **WEIGHTED coincidence drive** (`cfg.coincidence_weighted_drive=True`, the Poirazi-Brannon-Mel subunit already in the engine) **+** the FS rate-clamp **+** the per-region homeostasis target rate (sibling B1). The GIRK cap is then no longer *masking* a runaway — it becomes a redundant safety bound that should rarely bind. **Recommend the FS→critic de-risk; be honest that the cleanest result is FS-clamp + weighted-plateau + homeostasis-target replacing the *role* of the GIRK cap, not pooled inhibition alone perfectly dividing the plateau.** `critic_teacher_pa` (B's sub-question) is an **acceptable teaching scaffold** (§B.5).

---

# PART (A): the host reward burst → a spiking, perception-driven US/reward → SNc chain

## A.1 Diagnosis (crisp)

The N9 loop is all-neural **except the reward afferent**. With `enable_neural_critic`, the SNc input current is (`g11_bg_runner.py:6735-6740`):

```python
_I_snc = float(snc_tonic_pa) + float(snc_reward_gain) * max(0.0, float(reward))
#         (tonic pacemaker)     (HOST-INJECTED reward/US drive)   ← piece (A)
# (the −k_v·V term is already DROPPED: the striosome GABA_B subtracts V at the membrane — neural)
bridge.cp_external_input_current[region_indices_cp["snc"]] = cp.float32(_I_snc)
```

`reward ∈ {−1, 0, +1}` is the environment's outcome. With **N5** (`--perceived-approach-reward`, `g11_bg_runner.py:6515-6539`) the reward **LOGIC** is already coordinate-free — it is `sign(Δ eccentricity)` read from `sc_salience_offset_from_image` on the rendered image (pixels only; coords never enter the reward branch). **But the value is still a Python scalar that the host multiplies by a gain and writes as a current.** Biologically this is the **unconditioned-stimulus → VTA/SNc pathway** collapsed into one host write. Under BRAIN-BASED-ONLY: the *world rendering the goal cue* is legitimate (environment); *a number becoming a dopamine-driving current without a neuron in between* is the shortcut. **The brain is not detecting/representing the reward — the host arithmetic is.**

## A.2 The faithful biology (what actually drives DA at reward delivery)

DA neurons do **not** compute reward internally; they are **driven** to burst by afferents carrying the primary (unconditioned) reward signal, and the VALUE that gets subtracted (the −V) comes from a *separate* GABAergic source (already modelled here as the striosome→SNc GABA_B). The afferent sources of the **positive reward (US) component**:

| Source | Sign / transmitter | Role | Catalog / lit |
|---|---|---|---|
| **PPN (pedunculopontine n.)** | **glutamate + ACh, EXCITATORY** → SNc/VTA | Sensory + reward-magnitude drive; PPN inactivation **degrades the cue-evoked DA burst**; electrical PPN stim activates 20–40% of DA cells. **PPN does NOT itself compute RPE** — it is a *driver/contributor* of the early detection + reward component (latency *shorter* than DA). | **catalog C.33** (Schultz 2016 JNT pp 684–686; Pan & Hyland 2005; Hong & Hikosaka 2014) |
| **Parabrachial n. (PBN)** | glutamate, excitatory | Relays gustatory/visceral US (taste reward, the canonical primary reward) toward the DA system | Kandel 6e ch 43 (taste→reward); ascending-arousal catalog N.01 |
| **Lateral hypothalamus (LH)** | mixed: **GABAergic arm DISINHIBITS DA** (→ NAc DA, place preference, positive reinforcement); glutamatergic arm is aversive | LH is the **major excitatory/disinhibitory input to VTA for VALUE coding** (Watabe-Uchida 2012 inputome). The GABA arm (LH→VTA-GABA→DA disinhibition) is the reward-positive route. | Watabe-Uchida 2012; Nieh/Stamatakis 2016 (LH→VTA) |
| **SNc itself** | dopamine | Performs **subtraction** r − V; the −V comes from VTA/RMTg GABA. The US burst is the **r** term. | **Eshel 2015** (VTA GABA = the subtraction); **Tian-Uchida 2016** (distributed inputs, common response fn) |
| (the −V side, already modelled) | LHb→**RMTg** GABA→DA; striosome→SNc GABA | Reward-NEGATIVE / expected-value subtraction. **Already in the sim** as `striosome_value →(GABA_B) snc`. | Hong-Hikosaka 2011; catalog B.07 |

**The clean takeaway for design:** the **positive reward (US) component is an EXCITATORY afferent population that fires on the SENSED primary reward and projects to the SNc.** This is precisely the shape of the host current write being replaced. The −V subtraction is a *different* (GABAergic) population — and that one is **already spiking** in the sim. So (A) is "make the *excitatory US afferent* a real population," nothing more exotic.

## A.3 Ranked, biology-grounded SPIKING design for (A)

Notation: **fidelity** · **P(works)** · **surface** (runner-side vs protected `sim/` edit).

### A-1 ★ RECOMMENDED — a spiking `reward_us` (PPN-like) excitatory population, fired by PERCEIVED goal-contact, projecting to the SNc. **Runner-side. No `sim/` edit.**

This is the catalog C.33 prescription verbatim.

**Wiring (in `build_bg_brain_regions`, gated by a new `spiking_reward_us` flag):**
- A `BrainRegion(name="reward_us", n_neurons≈40, exc_fraction=1.0, izh_neuron_type=IZH2007_RS_CORTICAL_PYRAMIDAL)` — an excitable glutamatergic relay (PPN is glutamate+ACh; RS pyramidal is the closest existing excitable-relay preset; a dedicated PPN preset is optional polish, not required).
- A `RegionPathway(from_region="reward_us", to_region="snc", density≈0.6, weight_mean≈ tuned so a full US volley delivers ≈ the current `snc_reward_gain` worth of drive, plastic=False, receptor default excitatory)`. This is the PPN→SNc glutamatergic drive (C.33). `plastic=False` keeps it an *innate* US→DA reflex arc (the unconditioned response is hard-wired; only the *cue→US-prediction* learning is plastic — that lives in the actor/critic, not here).

**Drive (in the nav loop, replacing the host SNc reward write):**
- Each step, compute the **perceived goal eccentricity** `e = ‖sc_salience_offset_from_image(rendered_image)‖` (pixels; the **exact** quantity N5 already computes at `g11_bg_runner.py:6530-6533`). Map it to a US drive current onto `reward_us`: e.g. `I_us = us_gain · g(e)` where `g` is a **monotone-decreasing** gate of eccentricity (max when `e≈0` = on/at the goal; ~0 when the goal is far in the visual field). The simplest faithful form is **on-goal detection**: `I_us = us_drive_pA if e < e_contact else 0` (a US that fires at goal *attainment*, the unconditioned reward = "I am at the goal"). A graded form (`I_us ∝ max(0, 1 − e/e0)`) gives the appetitive/incentive-salience approach drive (Berridge wanting) and composes naturally with N5's approach reward.
- **Remove the `snc_reward_gain*max(0,reward)` term from `_I_snc`.** The SNc now reads `tonic + (synaptic drive from reward_us)`; the burst is produced by `reward_us` *firing into the SNc*, and the striosome GABA_B subtracts V — **the whole δ = r − V is now neural**, with both r and V carried by spiking afferents.

**Why this is the right call:** it is the lowest-fidelity-risk change (an excitatory afferent firing the SNc is uncontroversial DA physiology), it is **prescribed by the catalog**, it reuses the entire existing SNc/`from_region_firing_signed`/three-factor machinery downstream, and the US drive **rides on the same pixel signal N5 already validated** so the anti-cheat is inherited.

### A-2 — drive the DA neuromodulator concentration from `reward_us` firing via the existing `from_region_firing` rule (instead of, or in addition to, the synaptic SNc drive). **Runner-side.**

The neuromodulator subsystem **already** supports `rule_type="from_region_firing"` (`sim/neuromodulators.py:736`): a modulator concentration tracks the EMA firing fraction of named `source_regions`. One could add `reward_us` as a positive source so the **DA concentration itself** is partly produced by the perceived-US population firing. *However* — the current design (Stage A) deliberately produces the DA broadcast from the **SNc firing** via `from_region_firing_signed`, so the cleanest composition is **A-1** (US fires SNc → SNc firing produces DA), keeping a single DA-source-of-truth (the SNc), exactly as the project intends ("the dopamine signal IS the SNc firing"). A-2 is a viable alternative if you want the US to bypass the SNc membrane, but it **double-counts** the reward unless you also drop the synaptic arm. **Prefer A-1; A-2 is a fallback / not recommended to stack.**

### A-3 — full LH/RMTg disinhibition circuit (US→VTA-GABA→DA). **`sim/`-adjacent; defer.**

The most anatomically complete US route is *disinhibitory* (LH-GABA → VTA-GABA → DA, Watabe-Uchida/Nieh 2016): the US **silences** a tonic GABA brake on DA. This is higher fidelity but adds a population (a tonic VTA-GABA interneuron pool) and is **strictly more than needed** to make (A) neural. **Defer** — A-1's direct excitatory PPN→SNc drive is the catalog-endorsed minimal faithful form. (Note the sim already has the *complementary* GPi/SNr→SNc disinhibition collateral at `g11_bg_runner.py:1726-1729` and the striosome→SNc GABA_B, so the disinhibition motif is partly present for the −V side.)

**Recommended: A-1.** (A-2 fallback; A-3 deferred.)

## A.4 Honest: what genuinely stays host (and why it's legitimate)

- **The rendering of the goal cue into the agent's retina** — legitimate (environment: the world is visible). Already host (`render_gridworld_to_image`).
- **The reading of goal eccentricity from those pixels** — this is the agent's *perception*; in the sim it's a numpy centroid read (`sc_salience_offset_from_image`). **This is itself a host shortcut for what a spiking visual/SC salience map would compute** — and the project already acknowledges this (the N1 SC reflex is "biologically-shaped but host-computed → a shortcut," CLAUDE.md re-classification). For (A) specifically, the *reward* becoming neural does **not** require the *perception* to be neural first — but be honest that the US drive currently rides on a host perception. The fully-faithful end state routes the US gate off a **spiking SC salience map** (a separate, larger effort; out of (A)'s scope). For (A), the deliverable is: **the reward VALUE/burst/subtraction is neural; the perception it rides on is the same host-perception N5 already uses** (so no *new* shortcut is introduced — the reward shortcut is removed).
- **The instant of goal-contact** (the environment's "agent is on the goal" collision check) — **legitimate host (the body/environment boundary).** The world telling the retina "the goal cue is now centred / attained" is environment state, exactly like a real animal's mouth contacting food. The US neuron firing *in response to that sensed contact* is the neural part; the contact *event* is the world's. Do **not** try to make the collision check itself "spiking" — that would be making the environment a brain, which the standard explicitly excludes.

## A.5 Reusable machinery for (A)

| Need | Reuse | Where |
|---|---|---|
| Perceived goal eccentricity (pixels, no coords) | `sc_salience_offset_from_image` | `g11_bg_runner.py:158`; already used by N5 at `:6530` |
| The rendered goal cue | `render_gridworld_to_image` | `sim/visual_cortex.py`; used at `:6524-6529` |
| N5 coordinate-free approach reward (compose with US) | `--perceived-approach-reward` | `g11_bg_runner.py:6515`, flag `:7254` |
| SNc pool + tonic + DA-from-firing | `snc` region, `snc_tonic_pa`, `from_region_firing_signed` | `:3946-3954`, `:6733` |
| Region/pathway declaration | `BrainRegion`, `RegionPathway` | `sim/regions.py` |
| Modulator-from-firing (A-2 fallback) | `rule_type="from_region_firing"` | `sim/neuromodulators.py:736` |
| The host reward write being replaced | `_I_snc` block | `:6733-6750` |

**Genuinely new (runner-side):** the `reward_us` region + `reward_us→snc` pathway in `build_bg_brain_regions`; a US-drive injection block in the nav loop (`I_us = f(perceived eccentricity)`); and **dropping** the `snc_reward_gain*max(0,reward)` term. **No `sim/` edit.**

## A.6 Cheap-first de-risk for (A)

**One CPU experiment (`SIM_BACKEND=numpy`, serial, leave the GPU webapp alone), Pavlovian-style, BEFORE any nav build:**

Reuse the existing SNc Pavlovian probe pattern (`research/runners/snc_pavlovian_probe.py`, referenced at `:3929`). Build a minimal bridge: `reward_us → snc` + the striosome critic + the DA rule. Then:
1. **US drives the SNc burst.** Render the agent on the goal (`e≈0`) → `reward_us` fires → SNc bursts above tonic. Render it far (`e` large) → `reward_us` silent → SNc at tonic. **Gate: SNc burst on perceived-contact, tonic otherwise** — purely from `reward_us` firing, with the host `snc_reward_gain` term removed.
2. **δ = r − V still works with the neural US.** With a trained striosome V, the SNc burst at the (now-neural) US must SHRINK toward predicted (the GABA_B subtraction cancels the neural reward drive) — i.e. the existing Stage-B gaps reproduce with the US spiking instead of host-injected.
3. **Graceful FAIL contract:** if the `reward_us` volley can't deliver enough current to burst the SNc at any sane weight (mirror the MSN-rheobase lesson from the critic doc — check the SNc, which lacks KCC2 so is *easier* to drive, but verify), report FAIL and the verdict is "the US afferent needs a denser/stronger drive or a dedicated preset" — a valid finding, not a rescue-by-host-injection.

## A.7 Anti-cheat controls for (A) — *prove no coordinate/distance enters the reward*

1. **(US rides on PIXELS only)** — assert the US drive is a function **only** of `sc_salience_offset_from_image(rendered_image)`; `gx, gy, x, y, manhattan(...)` must NOT appear in the US-drive computation. (Same provenance bar N5 already meets; grep the US block for coord symbols.)
2. **(Goal-cue-shuffle / eccentricity-permute control)** — render the goal cue at a *random* pixel location decoupled from the true goal (or permute the eccentricity→US map). The US burst MUST then fire at the **wrong** times / the agent must FAIL to learn — proving the US is genuinely reading the *perceived goal*, not a back-channel. (Analogue of the place-shuffle control for the critic.)
3. **(No host reward to the SNc)** — assert `snc_reward_gain*max(0,reward)` is removed from `_I_snc`; the SNc reward burst must come **only** from `reward_us` synaptic drive. If the agent still navigates with that term deleted, the US chain is carrying the reward. (Lesion test: zero the `reward_us→snc` weight → SNc never bursts → no learning, proving the chain is load-bearing.)
4. **(`current_reward_signal` honesty)** — note that `current_reward_signal` (the host scalar) is still used by the three-factor *plasticity* path elsewhere; (A) only removes its role as the *SNc reward drive*. Be explicit which uses remain host (the three-factor reward-modulation gate reads it; that's a separate, documented item — making *that* neural is the deeper Stage-B work, not (A)).
5. **(Compose with N5, don't regress it)** — the N5 approach reward and the US-at-contact must be the same pixel signal at two eccentricity bands (approach = Δe<0; contact = e≈0); assert they don't double-count or fight.

---

# PART (B): physiological-range, GRADED critic firing via spiking pooled inhibition (not the GIRK cap)

## B.1 Diagnosis (crisp)

The MSN-D1 `striosome_value` critic is driven by the **FS-PING-synchronized place volley** through a **coincidence dendritic plateau** (`place → striosome_value`, `coincidence_detector=True`, `g11_bg_runner.py:1652-1658`). The place-code self-org is CuPy-non-deterministic (transpose-SpMV atomic scatter — the sibling robustness doc's Axis A), so the volley strength varies **28–118 Hz run-to-run** (`g11_bg_runner.py:1048-1049` comment), and on a *hot* draw the critic over-fires (~125 Hz). An over-firing critic delivers a saturating GABA_B/GIRK conductance onto the SNc → the value subtraction is **all-or-nothing** (clamp the SNc to E_K ≈ −90 mV) rather than **graded** (subtract an amount proportional to V) → the δ goes binary. The current mitigation is **`cfg.gabab_conductance_max`** (a finite-channel GIRK saturation cap, `g11_bg_runner.py:3829-3831`): it bounds how hard the critic can clamp the SNc. **This is legitimate biophysics (GIRK channels are finite) but it bounds the SYMPTOM at the SNc**, not the cause (the critic firing 125 Hz). The owner wants the **root** fixed: keep the MSN critic in a **physiological 1–20 Hz** range so V — and hence the GABA_B current it injects and the δ that results — is **graded across the draw-variable drive**, via **real spiking inhibition** rather than a conductance bound.

**Why "physiological MSN rate" is the right target:** in vivo MSNs fire at **<1–10 Hz** even in their active up-state (B.02; Wilson & Kawaguchi). A 125 Hz MSN is unphysiological and is a symptom of the all-or-none plateau triggering with no inhibitory brake. The biological brake is **feedforward FS-PV inhibition** (B.06) + lateral MSN inhibition (B.52) + the pooled-inhibition substrate of **divisive normalization** (Carandini-Heeger 2012).

## B.2 The faithful biology of striatal MSN firing-rate control

| Mechanism | What it does | Catalog / lit | Caveat for *this* problem |
|---|---|---|---|
| **FS-PV feedforward inhibition** (`place_fs`-type) | A few FS cells, each contacting hundreds of MSNs perisomatically, fire 1–3 ms after the cortical/afferent volley and **clamp MSN bursting, Ca²⁺, and plasticity** | **B.06** (Kandel 6e ch 38 p 935; TK-2017; Tepper-2018); **Lee 2017** ("FSIs supply feedforward control of bursting, calcium, and plasticity"); Owen 2018 (silencing FSIs → MSNs disinhibit, rate↑) | FS inhibition is **perisomatic / shunting** → it most directly controls **somatic spike output and bursting**, *gating* the plateau's expression — exactly the lever to bound a 125 Hz runaway. But it acts at the soma, **downstream of the dendritic plateau current** (see B.4). |
| **Lateral MSN collateral inhibition** (B.52) | MSNs inhibit each other (WTA/sparsification) | catalog B.52; already partly in the sim as v3 `--bg-lateral-inhibition` | Weak per-synapse, slow; not the primary rate-control for a *single* value cell. |
| **Divisive normalization** (Carandini-Heeger) | Output ∝ drive / (σ + pooled activity) → invariant to input *gain*, sensitive to *pattern* | **Carandini & Heeger 2012** (Nat Rev Neurosci 13:51); the FS pool **is** a normalization substrate | The canonical mechanism for "make the readout robust to drive magnitude." **But its clean divisive form assumes a GRADED drive in the denominator and numerator** — see the all-or-none caveat (B.4). |
| **Intrinsic homeostasis** (threshold) | Cell adapts its own threshold to defend a target rate | sibling doc **B1**, committed `89b8d909`, `--enable-critic-homeostasis` | Already shipped + de-risked 3/3; SLOW (defends a *target rate*, complementary to fast FS gain control). |

## B.3 Ranked, biology-grounded SPIKING design for (B)

### B-1 ★ RECOMMENDED (first lever) — add a `place_fs → striosome_value` feedforward GABA_A inhibition pathway (pooled, divisive-leaning). **Runner-side. ONE `RegionPathway`. No `sim/` edit.**

**The key reusable fact:** the FS-PING pool **`place_fs` already exists** and is reciprocally wired to `place` (`g11_bg_runner.py:1638-1648`) — it gamma-synchronizes the volley that fires the critic. **It currently inhibits ONLY `place`** (the `place_fs → place` edge, `transmission_gate="place_fs_gate"`). **There is NO `place_fs → striosome_value` edge today** — that is the missing brake on the critic. Add it:

```python
# NEW (runner-side, in build_bg_brain_regions, the neural_place_selforg branch):
RegionPathway(from_region="place_fs", to_region="striosome_value",
              density≈0.6, weight_mean≈ tuned, weight_jitter=0.2, plastic=False,
              receptor default GABA_A,            # perisomatic shunt on the MSN soma
              transmission_gate="critic_fs_gate") # optional: hold open during read-out
```

**Why this is divisive-LEANING (the good part):** `place_fs` firing scales with the **size of the place volley** (more active place cells → more FS drive → more FS spikes). So a *hot* draw (118 Hz volley) recruits *more* FS inhibition than a *weak* draw (28 Hz volley) → the inhibition **subtracts more when the drive is larger** → the critic's *somatic output rate* is compressed toward a common range across draws. That is the pooled-inhibition route to gain control (Carandini-Heeger via real inhibition), and it is the **root fix the owner wants**: the critic stays bounded because a *neuron* (the FS pool) is inhibiting it, not because a conductance cap clips the GIRK at the SNc.

**Expected outcome (honest):** this **very likely fixes the over-firing** (125 Hz → physiological) — perisomatic FS shunt + hyperpolarization is exactly what bounds MSN somatic rate (Lee 2017). The GIRK cap then rarely binds (becomes a redundant safety bound, not the operative mechanism).

### B-2 (compose with B-1) — carry the GRADED-δ requirement with the WEIGHTED coincidence plateau + homeostasis target, NOT by dividing the plateau. **Runner-side flag (`coincidence_weighted_drive`) + the shipped homeostasis.**

This is the load-bearing honesty (see B.4): pooled inhibition bounds the *rate* but you still need V to be **graded by value**, and the plateau is all-or-none on *count*. The engine **already** has the fix — the **WEIGHTED** coincidence drive (`cfg.coincidence_weighted_drive=True`, `sim/bridge.py:5782`, the Poirazi-Brannon-Mel subunit): `c_drive = Σ_j w_eff_j · x_j`, so the plateau crosses its supralinear switch **as a function of the LEARNED synaptic value**, not just the count. A high-V location (grown weights) crosses harder/earlier; a low-V location may not cross at all. Combined with B-1's FS rate-clamp and the sibling-doc **B1 per-region homeostasis** (defends a *target* critic rate independent of raw drive), the result is: **V graded by location-value, critic rate physiological, δ graded — all neural.** The comment at `:3839-3846` notes the weighted form is "a READ-OUT-only toggle, applied in Phase 3 value-learning" — so it's already intended for the value-read; B-2 is "turn it on for the read-out and verify it grades."

### B-3 — keep the GIRK cap as a redundant SAFETY bound (don't delete it). **Already shipped.**

`cfg.gabab_conductance_max` is **legitimate biophysics** (finite GIRK channels) — the recommendation is NOT to remove it but to **relegate it**: with B-1+B-2 the critic shouldn't reach the rates that make the cap bind, so it becomes a guardrail, not the operative gain control. Report (anti-cheat) that with B-1+B-2 on, the cap **rarely or never binds** (log how often `g_gabab` hits `gabab_conductance_max`) — if it still binds constantly, B-1 didn't actually clamp the rate and you've learned that (honest negative).

### B-4 — (the skeptical core) can pooled inhibition NORMALIZE an all-or-none plateau? Honest analysis.

**The prompt's exact worry, answered.** The critic's drive is **not** a graded current that pooled inhibition can cleanly divide. It is (`sim/bridge.py:5749`): *"A supralinear **all-or-none** switch on c_drive (≥ cfg.coincidence_k_threshold) injects a regenerative, Mg²⁺-self-limiting, slow-decaying plateau current."* So:

- **What FS inhibition CAN do to an all-or-none plateau:**
  1. **Gate WHETHER it triggers** — somatic/perisomatic hyperpolarization + shunt raises the *effective* coincident-input count needed to cross `coincidence_k_threshold` (fewer NMDA/AMPA inputs reach the dendritic threshold). So strong FS inhibition can **prevent** the plateau on weak/ambiguous draws and **permit** it on strong/high-value ones — a *gating* (near-threshold sharpening), which is biologically real and useful (it's how FS cells enforce "only consensual input fires the MSN," B.02/B.06).
  2. **Clamp the post-plateau SPIKE RATE** — once the plateau is up (a sustained depolarizing current), the number of *spikes* it produces is set by the balance of (plateau depolarization) vs (FS shunt + AHP). More FS inhibition → fewer spikes per plateau → lower *output rate*. **This is the 125 Hz → physiological fix**, and it works *even though the plateau current itself is all-or-none*, because the **spike rate** is graded by the inhibition even when the **plateau** is not.
- **What FS inhibition CANNOT do:**
  3. **Smoothly DIVIDE an already-fully-triggered plateau current** the way Carandini-Heeger divides a graded feedforward drive. The plateau is regenerative/Mg²⁺-self-limiting; once over threshold it delivers its (roughly fixed) current. So you **cannot** get "V ∝ drive / pooled-activity" as a clean analog division of the plateau amplitude. The *grading of V by value* therefore must come from the **WEIGHTED plateau** (B-2: which locations cross, and how hard) **+ the spike-rate clamp** (B-1: how many spikes the up-plateau emits), **not** from dividing the plateau.

**Therefore the HONEST design is a HYBRID, and that is the correct answer, not a cop-out:**
- **Rate normalization** (125 Hz runaway → 1–20 Hz): **B-1 FS→critic pooled inhibition** (clamps the *spike output*) **+ sibling B1 homeostasis** (defends a *target* rate). ✅ spiking, root-cause.
- **Value grading** (δ proportional to V, not binary): **B-2 weighted coincidence plateau** (which/how-hard the plateau triggers, by learned value) + the spike-rate clamp grading the output. ✅ spiking.
- **Safety**: **B-3 GIRK cap** relegated to a rarely-binding guardrail. ✅ legitimate biophysics.

If the de-risk shows FS inhibition **cannot** hold the critic graded across draws (e.g. the all-or-none plateau means the critic is either ~0 Hz or saturated with no middle band, so FS just flips it off/on), then the **honest answer is that the GIRK cap (or a physiological plateau strength — lowering `coincidence_plateau_strength` so the up-state itself is gentler) is the right tool**, and pooled inhibition alone does not rate-normalize an all-or-none plateau. **That negative is a valid deliverable** (it maps a substrate limit: rate normalization presupposes a gradable drive; an all-or-none dendritic plateau is intrinsically a poor target for divisive normalization — the grading must live in the plateau's *weighted trigger*, not in post-hoc division).

### B-5 — alternative locus: reduce the plateau strength itself (physiological up-state). **Runner-side knob (`coincidence_plateau_strength`).**

If B-1+B-2 don't tame it, the *most direct* root fix is that the plateau is simply **too strong** (`coincidence_plateau_strength=80.0`, `:3845`). A gentler plateau (lower strength) produces a physiological MSN up-state firing fewer spikes — no inhibition needed. This is **tuning the dendritic biophysics to the physiological range** (defensible: the plateau strength is a free parameter that was set for *bootstrapping the post-spike*, `:3838`). It's the honest fallback if pooled inhibition can't grade an all-or-none current: **make the current itself smaller/gentler.** Lower fidelity than "FS controls the rate" but it's a legitimate biophysical setting, not a host shortcut.

**Recommended for (B): B-1 + B-2 (+ keep B-3 as guardrail), with B-4's honest expectation that the result is a HYBRID (FS clamps rate, weighted-plateau grades value), and B-5 as the fallback if FS can't grade the all-or-none plateau.**

## B.5 Is `critic_teacher_pa` an acceptable teaching scaffold?

**Yes — it is an acceptable teaching scaffold, consistent with the project's "innate-reflex-teaches-a-learned-circuit" pattern, PROVIDED it is removed before nav (it is) and the value is RE-VALIDATED to fire from the place volley alone afterward.** `critic_teacher_pa` (default 300 pA, `g11_bg_runner.py:3081`) is a **sub-threshold** phase-locked current on `striosome_value` applied **ONLY during the PAIR phase of pre-nav value-training** and **removed before the read-out / nav** (`:3073-3075`, `:5051-5052`). Its role is to make the weak-drive place volley fire the critic *phase-locked* during training so DA-gated STDP can grow the `place→striosome_value` weights (the LTP-bootstrap deadlock-breaker, mirroring the de-risk). This is **exactly** the legitimate scaffold class: an external current that *teaches* a circuit which then operates without it (like the N1 SC reflex teaching a learned dorsal read-out). **Conditions for it to stay legitimate:**
1. It is **sub-threshold** (it does not by itself fire the critic — it phase-locks an otherwise-marginal volley). The de-risk uses 300 pA precisely so post-before-pre LTD is avoided and the *place* drive supplies the suprathreshold coincidence (the `:2994-2996` note: a *supra*-threshold teacher is counter-productive — it drives post≫pre → LTD → weight collapse). **Keep it sub-threshold.**
2. It is **removed before nav** (it is — `value_input` is frozen and the teacher is off for the read-out).
3. **Post-training, the critic must fire from the place volley ALONE** (no teacher) and stay GRADED near≫far — which is exactly Stage-B gate 2. If it only fires *with* the teacher, the scaffold became a crutch and that's a FAIL.

**Should it ALSO be spiking?** It could be upgraded to a spiking "training-context" drive (e.g. a tonic excitatory teacher population), but there is **little fidelity gain**: a sub-threshold phase-locking current that is removed before deployment is the textbook teaching-scaffold, and the project explicitly endorses these. **Recommendation: keep `critic_teacher_pa` as-is (host sub-threshold teacher, removed pre-nav) — it is an acceptable scaffold, not a shortcut, by the project's own standard.** Do not spend effort spiking it unless a future audit wants *zero* host current anywhere in training; if so, the minimal upgrade is a small tonic excitatory "training-context" region active only during the PAIR phase.

## B.6 Reusable machinery for (B)

| Need | Reuse | Where |
|---|---|---|
| **FS-PING pool (the inhibitor)** | **`place_fs`** (already exists, gamma-synchronizes the volley) | `g11_bg_runner.py:1038-1043` (region), `:1638-1648` (wired to `place` only — the critic edge is what's missing) |
| FS interneuron preset | `IZH2007_FS_CORTICAL_INTERNEURON` | `sim/enums.py` |
| **WEIGHTED coincidence plateau (value grading)** | `cfg.coincidence_weighted_drive=True` + the Poirazi-Brannon-Mel subunit | `sim/bridge.py:5782`, kernel `fused_coincidence_plateau` `:5793`; runner `:3839-3846` |
| Plateau strength knob (B-5 fallback) | `cfg.coincidence_plateau_strength`, `coincidence_k_threshold` | `:3845`, `:3843`; runner kwarg `coincidence_plateau` `:3053` |
| **Intrinsic homeostasis (target rate)** | `--enable-critic-homeostasis` (committed `89b8d909`) | `:1067/:1089/:1120`; `cp_homeostasis_neuron_mask` `sim/bridge.py` |
| GIRK saturation cap (B-3 guardrail) | `cfg.gabab_conductance_max` (`--critic-gabab-max`) | `:3831` |
| GABA_A inhibition / receptor routing | `RegionPathway(receptor=...)`, `syn_reversal_potential_i_override` | `sim/regions.py`; MSN E_GABA already −60 at `:1065` |
| Transmission gate (hold FS open during read) | `transmission_gate=...` + `set_transmission_gate` | `:1647` (the `place_fs_gate` precedent) |
| Striatal PV-FSI machinery (alt inhibitor) | `--enable-striatal-fsis` (`str_fs`, B.06) | `:836`, `:1482-1504` |
| Place-shuffle anti-cheat (must survive) | `--shuffle` permuted place→value control | the Stage-B / de-risk harness |

**Genuinely new (runner-side):** the single `place_fs → striosome_value` `RegionPathway`. **Everything else (weighted plateau, homeostasis, GIRK cap, plateau strength) is existing flags.** **No `sim/` edit** for B-1/B-2/B-3/B-5. (If you wanted a *dedicated* feedforward FS pool for the critic distinct from the place-PING `place_fs`, that's still just regions+pathways, runner-side.)

## B.7 Cheap-first de-risk for (B)

**One CPU experiment (`SIM_BACKEND=numpy`), on the isolated critic, BEFORE nav** — extend the Stage-B critic probe (`snc_stageb_critic_probe_*.py`):

1. **Sweep the draw strength** (emulate the 28–118 Hz volley variance by scaling the place-volley drive across a range) and measure `striosome_value` output rate **with vs without** the `place_fs → striosome_value` edge:
   - **without** (today): rate runs away to ~125 Hz at the hot end.
   - **with B-1** (FS→critic): rate should compress into a physiological band (target ~1–20 Hz) across the whole draw range. **Gate B-i: critic rate < ~25 Hz at the hottest draw AND > ~3 Hz at the weakest** (bounded both ways, not flipped off).
2. **With `coincidence_weighted_drive=True` (B-2)**, verify V is **graded by value**: near (high-grown-weight) ≫ far (low-weight) in *both* firing rate **and** the GABA_B current onto the SNc, across draws. **Gate B-ii: near/far ratio ≥ ~2× preserved at every draw strength** (grading survives the FS clamp — the all-or-none worry didn't materialize).
3. **δ is graded:** drive the SNc with `reward_us`/tonic and the critic's GABA_B; sweep V and confirm the SNc burst **shrinks proportionally to V** (not binary). **Gate B-iii: SNc burst is a monotone-graded function of V** (the binary-δ symptom is gone).
4. **GIRK-cap relegation:** log how often `g_gabab` hits `gabab_conductance_max` with B-1+B-2 on. **Gate B-iv: the cap binds rarely (< some small %)** — proving the *neuron* (FS), not the *cap*, is now controlling the rate.
5. **Graceful FAIL contract:** if Gate B-ii fails (FS clamp makes the critic either ~0 or saturated with no graded middle — the all-or-none plateau defeating rate-normalization), the verdict is **"pooled inhibition cannot grade the all-or-none plateau; the honest fix is B-5 (gentler `coincidence_plateau_strength`) or retaining the GIRK cap as the operative bound."** That negative **maps a real substrate limit** and is the deliverable.

**Decision rule:** B-i..B-iv pass ≥3 draw strengths → wire `place_fs→striosome_value` into nav and run the nav 6-seed regression (acceptance = the online δ is graded AND summed reward ≥ the GIRK-cap-only Stage-B). FAIL Gate B-ii → adopt B-5 / keep the cap, bank the negative.

## B.8 Anti-cheat controls for (B) — *prove the critic stays GRADED and the place-shuffle still breaks V*

1. **(Place-shuffle still breaks V)** — the existing `--shuffle` permuted `place→value` control MUST drop the near/far ratio below ~2× under the new FS+weighted protocol. FS inhibition + homeostasis lift/clamp *rate*; they must NOT let the critic learn V from "fired-on-any-drive." V must ride on weights learned at the *rewarded* location. (Carry verbatim from the sibling docs.)
2. **(Grading survives the inhibition)** — after adding FS→critic, assert near ≫ far in firing **and** GABA_B current at *every* tested draw strength (Gate B-ii). The fix must NOT flatten V into place-blindness (the failure mode an over-aggressive clamp would cause).
3. **(The inhibition is NEURAL, pooled, not a host divide)** — assert the rate control is the `place_fs → striosome_value` *synaptic* current (a spiking pooled inhibition), NOT a Python `V / (σ + Σ)` normalization. If anyone implements divisive normalization as a host formula on the critic's output, that is a **shortcut** and disqualified — it must be the FS pool's GABA_A current.
4. **(GIRK cap is a guardrail, not the mechanism)** — Gate B-iv: log cap-binding frequency; the claim "the root cause is fixed" requires the cap to rarely bind. If the cap still binds every step, the FS route didn't actually control the rate — say so (honest negative; the GIRK cap remained the operative bound).
5. **(δ graded, not binary)** — Gate B-iii: the SNc burst must be a monotone function of V across a V-sweep (the deliverable's headline). A binary SNc burst means the symptom persists.
6. **(Determinism honesty)** — note this composes with the sibling robustness doc's Axis A: FS rate-control makes the critic robust to draw *strength*, but the *place code itself* still varies run-to-run until the transpose-SpMV non-determinism is fixed (sibling A1/A3). Report whether the FS clamp alone makes the *online δ* draw-robust enough for nav, or whether the determinism fix is also needed (likely both — FS for rate, determinism for reproducibility, homeostasis for target). The honest framing: **(B) fixes the over-firing/binary-δ; it does not by itself fix the place-code non-determinism (that's the sibling doc).**
7. **(Actor untouched)** — the `place_fs → striosome_value` edge must inhibit ONLY the critic; assert the actor (`cortex_X`/`motor_X`) firing is unchanged (the FS pool is the place-PING pool, location-blind by design `:1035-1037`, so this should hold — verify).

---

## Honest bottom line

**(A) is the clean, catalog-prescribed win:** an excitatory `reward_us` (PPN-like) population firing the SNc on **perceived** goal-contact, replacing the host `snc_reward_gain*reward` write — entirely runner-side, reusing the SNc/DA/three-factor stack, riding on the **same pixel signal N5 validated** so the anti-cheat is inherited. The only honest residual is that the *perception* it rides on is still a host centroid read (a pre-existing, separately-scoped shortcut) and the *goal-contact event* legitimately stays host (the body/environment boundary). **(B) is feasible at the rate-control level but the prompt's skepticism is warranted:** the FS-PING pool `place_fs` already exists and adding a `place_fs → striosome_value` edge (one `RegionPathway`, runner-side) is the biologically-correct pooled-inhibition brake that will **very likely** clamp the 125 Hz runaway — **but** because the critic's drive is an **all-or-none coincidence plateau**, pooled inhibition can clamp the *spike rate* and *gate the trigger* yet **cannot smoothly divide the plateau** into a graded value; the **grading must come from the WEIGHTED plateau (already in the engine) + the rate clamp + homeostasis**, and the GIRK cap is relegated to a rarely-binding guardrail. If the de-risk shows FS can't keep V graded across draws, the honest fix is a gentler plateau strength (B-5) or keeping the GIRK cap — and **that negative is itself the deliverable** (it maps the limit: an all-or-none dendritic plateau is a poor target for divisive rate-normalization). **`critic_teacher_pa` is an acceptable teaching scaffold** (sub-threshold, removed pre-nav, with the critic re-validated to fire from the volley alone) and does not need to be spiking by the project's own standard. Recommend the controller present **A-1** and the **B-1+B-2 hybrid (keep B-3 as guardrail; B-5 fallback)**, and run the two CPU de-risks (Pavlovian US-fires-SNc; draw-sweep critic-rate-graded) before any nav build.

---

### Load-bearing citations

| Claim | Source |
|---|---|
| Host reward burst onto SNc (piece A) | `g11_bg_runner.py:6735-6750` |
| N5 reward LOGIC is pixel-sourced (coord-free), still host-injected onto SNc | `:6515-6539` (logic) + `:6737` (injection); `sc_salience_offset_from_image` `:158` |
| **PPN supplies sensory+reward drive to DA; small PPN region → DA pool prescription** | **catalog C.33** (Schultz 2016 JNT pp 684–686; Pan & Hyland 2005; Hong & Hikosaka 2014) |
| SNc gets strong EXCITATORY input (sensorimotor cortex + STN); VTA from LH (value) | Watabe-Uchida et al. 2012, *Neuron* (whole-brain inputome) |
| DA neurons perform SUBTRACTION; VTA GABA = the subtraction source | Eshel et al. 2015, *Nature* 525:243 (already the project's −V basis) |
| Distributed inputs, common DA response function | Tian, Uchida et al. 2016, *Nat Neurosci* 19; Tian & Uchida 2016 *Neuron* |
| LHb→RMTg→DA GABA = reward-NEGATIVE subtraction (the −V side, already in sim) | Hong & Hikosaka 2011, *J Neurosci* 31:11457; catalog B.07 |
| LH→VTA: GABA arm disinhibits DA / positive reinforcement; glutamate arm aversive | Nieh/Stamatakis 2016, *Neuron* (LH→VTA) |
| Critic drive is an **all-or-none supralinear** coincidence plateau | `sim/bridge.py:5749`, `fused_coincidence_plateau` `:5793-5804` |
| **WEIGHTED** coincidence drive (Poirazi-Brannon-Mel) grades the plateau by learned value | `sim/bridge.py:5782`; runner `:3839-3846` |
| **`place_fs` FS-PING pool exists, inhibits ONLY `place`** (the missing critic-brake) | `:1038-1043` (region), `:1638-1648` (wiring) |
| Striatal PV-FSI feedforward inhibition controls MSN bursting/Ca²⁺/rate | **catalog B.06**; Kandel 6e ch 38 p 935; Lee et al. 2017 (*Cell* 171:1532); Owen et al. 2018 |
| MSN physiological up-state rate is <1–10 Hz | catalog B.02; Wilson & Kawaguchi 1996 |
| Divisive normalization via pooled inhibition (canonical computation) | Carandini & Heeger 2012, *Nat Rev Neurosci* 13:51 |
| Per-region homeostasis (target-rate, committed) | commit `89b8d909`; `--enable-critic-homeostasis` |
| GIRK conductance cap (the symptom-bounding patch) | `cfg.gabab_conductance_max`, `:3829-3831` |
| `critic_teacher_pa` sub-threshold, removed pre-nav (acceptable scaffold) | `:3073-3081`, `:5051-5052`; the supra-threshold-is-counterproductive note `:2994-2996` |
| BRAIN-BASED-ONLY standard (host computation = shortcut; honest negatives = deliverable) | CLAUDE.md "Standing standard: BRAIN-BASED ONLY"; owner directive 2026-06-08 |

**Sources (web):**
- [Whole-Brain Mapping of Direct Inputs to Midbrain Dopamine Neurons — Watabe-Uchida 2012, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(12)00281-4)
- [Arithmetic and local circuitry underlying dopamine prediction errors — Eshel 2015, Nature](https://www.nature.com/articles/nature14855)
- [Dopamine neurons share common response function for reward prediction error — Eshel/Tian/Uchida 2016, Nat Neurosci](https://www.nature.com/articles/nn.4239)
- [Neural Circuitry of Reward Prediction Error — Watabe-Uchida & Uchida 2017 (review; PPN/LHb/RMTg afferents)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6721851/)
- [Negative Reward Signals from the Lateral Habenula to Dopamine Neurons Are Mediated by RMTg — Hong & Hikosaka 2011, J Neurosci](https://www.jneurosci.org/content/31/32/11457)
- [Inhibitory Input from the Lateral Hypothalamus to the VTA Disinhibits Dopamine Neurons — Nieh 2016, Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(16)30122-2)
- [Fast-spiking interneurons supply feed-forward control of bursting, calcium, and plasticity — Lee 2017, Cell (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5810594/)
- [Functional Properties of Striatal Fast-Spiking Interneurons — review, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3121016/)
- [Carandini & Heeger 2012, Normalization as a canonical neural computation, Nat Rev Neurosci 13:51](https://www.nature.com/articles/nrn3136)
