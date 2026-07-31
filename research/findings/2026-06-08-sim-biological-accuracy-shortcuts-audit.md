---
type: finding
status: contributing
date: 2026-06-08
---

# Biophysical-accuracy SHORTCUTS audit (the substrate, not the cheats)

**Date:** 2026-06-08
**Type:** read-only audit (no code changed, no jobs run)
**Question (owner):** "Do we have other instances in the sim that are similar [to the
GABA_A-only inhibition]? Places we picked convenience over biological accuracy? While
not outright cheats — as long as the sim is handling it directly — I'd consider such
instances SHORTCUTS rather than cheats, but may still want to address them."

## Definitions used here

- **CHEAT** = a *cognitive* function done by HOST code instead of by neurons/synapses (a
  Python RPE formula, a pixel-reading reflex, an `argmax` over spike counts). **Out of scope.**
- **SHORTCUT** = a biophysical/neural mechanism the **sim handles directly** (in neurons,
  synapses, conductances, plasticity) but **simplifies/approximates** for convenience,
  trading fidelity. **This is the catalogue.** The brain is doing the work; just less
  faithfully than real biology.

The **exemplar** (do not re-derive): the engine models only **GABA_A** inhibition — a single
`g_i·(E_inh − V)` current (`sim/kernels.py:208` `fused_conductance_decay_and_current`, with
`E_inh = -75 mV`, `sim/config.py:129`) — and lacks the **GABA_B→GIRK** slow, K⁺-reversal
(E_K ≈ −90 mV), non-chloride hyperpolarizing arm. The sim IS doing inhibition in neurons; it
just collapses two receptor classes into one. (Catalog C.02, line 678: *"GABA-B … is missing
— would need an additional slower inhibitory channel pathway."*) This audit finds the same
*pattern* in ~14 other places.

Sources verified at `file:line`: `sim/kernels.py`, `sim/config.py`, `sim/bridge.py`,
`sim/neuromodulators.py`, `sim/enums.py`; the reference catalog
`sim-catalog/references/feature-catalog.md` (which already tracks per-feature *sim-status* +
*discrepancy* notes — the richest source); cross-referenced findings docs.

A guiding principle for the ranking: rank by **impact on the project's actual goals** — a
biology-faithful *learning agent*, and the *conversational* + *navigation* arcs. A shortcut
that has already blocked a result ranks above an anatomically-glaring one that has never
bitten.

---

## TIER 1 — Already bit us / high impact

### 1. Inhibition is GABA_A-only — no GABA_B→GIRK slow arm  *(the exemplar)*

- **Mechanism + code:** single inhibitory conductance `g_i` with one reversal potential.
  `fused_conductance_decay_and_current` (`sim/kernels.py:208`): `I_syn = g_e·(E_e−v) +
  g_i·(E_i−v)`. `E_inh = -75 mV` (`sim/config.py:129`); single-exponential decay
  `syn_tau_g_i = 10 ms` (`sim/config.py:131`). The step applies one `E_inh_to_use`
  (`sim/bridge.py:5322`).
- **Simplification:** all inhibition is fast ionotropic Cl⁻ at one τ and one reversal. No
  metabotropic GABA_B (Gi/Go → GIRK K⁺ channel, E_K ≈ −90 mV, ~150–300 ms IPSP, presynaptic
  autoreceptor that also suppresses release).
- **Biology:** GABA_B is a *distinct* receptor with a *more hyperpolarized* reversal than
  GABA_A and a much slower time course (Kandel 6e Ch 16 p 365–367; catalog C.02). The deeper
  K⁺ reversal is exactly what makes GABA_B a strong *subtractive/divisive value brake* where
  GABA_A (shunting near −75 mV) is weak.
- **Impact (already bit):** this directly blocked a biologically-strong value subtraction onto
  dopamine cells in the spiking-SNc actor-critic work (the triggering observation;
  `docs/plans/2026-06-08-spiking-snc-actor-critic-design.md` and the `from_region_firing_signed`
  rule added 2026-06-08, `sim/neuromodulators.py:774`). The SNc dip (negative RPE) wants a
  hyperpolarizing pull below tonic that a −75 mV chloride current cannot deliver. Latently it
  also limits slow-IPSP rhythms, K⁺-driven post-inhibitory rebound timing, and any circuit
  where presynaptic GABA_B gain control matters.
- **Worth addressing? YES — top-3.** Scope: **medium**, a *protected* `sim/` edit. Add a
  third conductance channel `g_gabab` with its own slow τ and `E_gabab ≈ -90 mV`, driven by
  the same inhibitory spikes scaled by a `gabab_ratio` (mirrors exactly how NMDA was added
  alongside AMPA — `fused_nmda_update_and_current`, `sim/kernels.py:218`, is the template).
  Default ratio 0 = byte-identical to today. Anti-cheat: a paired-pulse high-rate IPSP assay
  (slow component appears only at high presynaptic rates) per catalog C.02 behavioral
  validation, plus confirm Izhikevich/HH/AdEx unchanged at ratio 0.
- **Honest note:** for *cortical PING/gamma and E/I balance*, GABA_A-only is fine — those
  benchmarks pass. The shortcut bites specifically where the *slow, deep-K⁺* arm is the
  functional substrate (value subtraction, slow rebound, presynaptic gain).

### 2. Single-compartment (point) neurons everywhere — no dendrites

- **Mechanism + code:** every model integrates all input in **one** membrane equation:
  Izhikevich (`fused_izhikevich2007_dynamics_update`, `sim/kernels.py:32`), HH
  (`...:48`), AdEx (`...:184`). Synaptic current sums at a single `v`
  (`sim/bridge.py:5328`). No apical/basal split, no spine heads/necks, no cable.
- **Simplification:** dendrites collapsed to a point. No dendritic nonlinearity (NMDA
  spikes, Ca²⁺ plateaus), no apical-tuft / basal coincidence detection, no per-spine gating,
  no distance-dependent attenuation.
- **Biology:** *"Single-compartment everywhere. This is one of the largest abstractions in
  the simulator"* (catalog J.05/J.07, lines 2638, 2648); L5 pyramidal apical-basal coincidence
  (catalog, line 2648); MSN spine head(cortex/AMPA+NMDA)/neck(DA, GABA) three-input gating
  (catalog B.02 supplemental, lines 427, 97); Kandel 6e Ch 6–7 (cable/compartments), Ch 10
  (dendritic integration). E.06 (line 3433): *"not-applicable — no dendrites, no λ."*
- **Impact (already bit, hard):** the **dendritic-learning arc is a multiply-confirmed
  NEGATIVE** — `2026-05-17-dendritic-credit-assignment-NEGATIVE.md`,
  `-faithful-instrument-TERMINUS.md`, `2026-05-18-dendritic-fairscale-…-VOID*.md`. The whole
  apical-basal credit-assignment alternative to global-scalar feedback (the W→A bottleneck,
  CLAUDE.md item 5) cannot be expressed because there is no second compartment. The
  composer-as-idealization limitation (`2026-06-06-composer-vsa-idealization-known-
  limitation.md`) — "a real cortex has LEARNED, lossy, redundant read-outs" — is partly
  downstream of this: point neurons with one soma can't host the dendritic read-out machinery
  a genuine cortex uses.
- **Worth addressing? PARTIALLY / strategically — but NOT a quick win.** Scope: **very large**
  (≥1.5–2 months; ~10× compute/neuron; protected `sim/` core rewrite). The prior dendritic
  design doc (`docs/plans/2026-05-05-dendritic-learning-design.md`) scoped it. Recommendation:
  keep **benched** unless a specific capability (apical-basal learning, perceptual inference)
  is the next deliberate target. A cheaper *two-compartment* (soma + one apical) AdEx variant
  is the minimal viable step if/when it is.
- **Honest note:** for the *conversational* arc, the FHRR/VSA binding is already on-substrate
  (resonate-and-fire + complex synapses) and point neurons suffice for it. Compartments are
  not on the critical path for *current* conversational goals — they are for the
  *genuine-cortex* conversion and for dendritic learning.

### 3. Uniform single-step (1×dt) synaptic delay — no per-pathway conduction delays

- **Mechanism + code:** propagation reads **`cp_prev_firing_states`** only — last step's
  spikes (`sim/bridge.py:5335`, `:5360`). `max_synaptic_delay_ms` (`sim/config.py:140`) is
  converted to `runtime_state.max_delay_steps` (`sim/bridge.py:1951`) **but that value is never
  used as a delay buffer anywhere in the step** (grep-confirmed: only set, never read in
  `_run_one_simulation_step`). Every synapse therefore fires with the *same* 1×dt latency.
- **Simplification:** no axonal conduction-delay state; no distance- or pathway-specific
  delays; no delay distribution.
- **Biology:** conduction velocities differ several-fold across projections — e.g.
  striatonigral ~1.4 m/s vs pallidonigral ~4 m/s (catalog B.16, lines 369, 99); the in-vivo
  BG three-phase response (early STN → striatal inhibition → late STN) *requires* the slow
  D1 gate arriving after fast GPe (catalog A.01 supplemental, line 99). General: catalog
  (line 369) *"missing — no axonal conduction delays … all pathway transmission is one-step";*
  Kandel 6e Ch 4 (conduction). P.11 behavioral note (line 5198): *"Increase per-pathway
  conduction delay → spike-coincidence falls → STDP windows mistime."*
- **Impact (latent-to-real):** caps temporal-code fidelity. The dlPFC dialogue-planning tie-
  break already shows rank-order (latency) coding is *dt-bound* (CLAUDE.md "ONE-BRIDGE
  UNIFICATION … rank-order coding RESOLUTION is dt-bound"). Phasor/FHRR phase precision and
  any polychronization-style sequence learning are limited by having a single delay. It also
  blocks faithful BG temporal sequencing.
- **Worth addressing? YES — top-3.** Scope: **medium** (protected `sim/` edit, but well-bounded
  and additive). Add an optional `RegionPathway.conduction_delay_ms` realized by a small
  per-pathway ring buffer of recent firing vectors (index `delay_steps` back instead of always
  `prev`). Default 1 step = byte-identical. This is the single highest *fidelity-per-effort*
  item: it unlocks temporal-code experiments the project keeps brushing against, and the
  catalog explicitly flags it as the missing piece for BG three-phase dynamics.
- **Honest note:** for rate-coded readouts (most current conversational + nav results), a
  uniform 1-step delay is fine. It bites only temporal/phase codes and multi-pathway timing.

---

## TIER 2 — Latent / medium impact

### 4. Single-exponential synaptic conductance (no dual-exp AMPA; NMDA is dual-exp but shared with AMPA drive)

- **Code:** `fused_conductance_decay_and_current` decays `g_e`, `g_i` by a single factor each
  (`sim/kernels.py:211–214`); `syn_tau_g_e = 5 ms`, `syn_tau_g_i = 10 ms`
  (`sim/config.py:130–131`). AMPA has **no rise time** (instantaneous jump, single-exp decay).
  NMDA *does* use a biologically-correct dual-exponential + Jahr-Stevens Mg block
  (`fused_nmda_update_and_current`, `sim/kernels.py:218`) — good — but NMDA receives the *same*
  excitatory drive as AMPA scaled by `nmda_ratio` (`sim/bridge.py:5416`), i.e. AMPA and NMDA
  share one presynaptic event rather than having independent receptor pools.
- **Simplification:** instantaneous-rise, single-τ AMPA/GABA_A kinetics; no separate
  kainate; AMPA and NMDA glued to one conductance increment.
- **Biology:** real AMPA EPSCs have a finite rise (~0.2–1 ms) and bi-exponential decay;
  GABA_A similarly (Kandel 6e Ch 10–11; catalog C.01/C.02). The single-exp is the standard
  reduced-model choice (Dayan & Abbott; Brette et al. 2007).
- **Impact:** minor for rate/firing dynamics; slightly mistimes very-fine coincidence windows.
  Has **not** demonstrably bitten any result.
- **Worth addressing? LOW priority.** A dual-exp AMPA is cheap (the NMDA dual-exp code is the
  template) but the payoff is small. Reasonable to leave.
- **Honest note:** this is a **reasonable simplification.** Single-exponential conductances are
  the field-standard reduction; the project's STDP/gamma/E-I benchmarks all pass with it.

### 5. Neuromodulators as global concentration scalars — no spatial volume transmission, no receptor-subtype machinery

- **Code:** `NeuromodulatorManager` holds **one scalar concentration per modulator**,
  broadcast globally (`sim/neuromodulators.py:211`, `compute_*_multiplier` apply scope=`all`
  uniformly, `:301`, `:323`). Effects are three phenomenological target types — `synaptic_gain`,
  `plasticity_rate`, `excitability_drive` (`sim/neuromodulators.py:31–80`).
- **Simplification:** no diffusion field / distance-dependent concentration gradient; receptor
  *subtypes* (D1 vs D2, β1 vs β2, GABA_B-as-NM, mAChR M1–M5) are collapsed into a per-target
  sensitivity rather than modeled as distinct receptors with opposite signs in the same region.
- **Biology:** monoamines act largely by **volume transmission** — non-synaptic diffuse release
  reaching many targets with a spatial gradient (catalog C.21, lines 894, 907); receptor
  subtypes have *opposing* G-protein coupling (D1 Gs vs D2 Gi; Kandel 6e Ch 14, Ch 16 p 360–371).
- **Impact:** the catalog judges the *abstraction* sound for volume transmission (line 900:
  *"the right abstraction for volume transmission"*) and J.13 (line 3647) calls it a deliberate
  phenomenological shortcut. It bites only where a *spatial gradient* of NM or a *within-region
  D1/D2 sign split at the molecular level* is the substrate. The project already works around
  the D1/D2 split structurally (`enable_d1_d2_asymmetry`, `sim/config.py:336`; per-action DA,
  `sim/neuromodulators.py:1032`) — i.e. the functional need is met by other means.
- **Worth addressing? LOW–MEDIUM.** Only if a volume-gradient experiment is wanted. The
  global-scalar choice is explicitly endorsed in the catalog.
- **Honest note:** **reasonable simplification**, and documented as such. Don't churn it.

### 6. Phenomenological AHP via M-current — no Ca²⁺-gated SK/BK channel, no intracellular Ca²⁺ pool

- **Code:** spike-frequency adaptation in the HH path is an **M-current** (`g_M`,
  `fused_hh_m_current_update`, `sim/kernels.py:120`); presets set `g_M_max` "AHP"
  (`sim/enums.py:260,279,294,331`). `g_CaT` produces a Ca²⁺ *current* (`...:139`) but there is
  **no intracellular [Ca²⁺] state variable that accumulates and gates an SK/BK channel** (grep:
  no `cp_calcium`/SK/AHP-Ca array). AdEx uses its `w` adaptation variable; Izhikevich uses `u`.
- **Simplification:** adaptation is voltage-gated (M-type) or a generic recovery variable, not
  true **Ca²⁺-activated K⁺** (SK = apamin-sensitive mAHP; BK = fast AHP).
- **Biology:** SK/BK channels are gated by [Ca²⁺]_i, not voltage; they couple firing history to
  AHP via Ca²⁺ accumulation (Kandel 6e Ch 7, Ch 10). The comment at `sim/enums.py:294` even
  notes "BK-like" — acknowledging the surrogate.
- **Impact:** the *functional* AHP (rate adaptation, burst termination) is present and tuned,
  so most dynamics are fine. Missing the Ca²⁺-dependence means adaptation doesn't scale with
  Ca²⁺ load / can't be selectively blocked (apamin experiments), and Ca²⁺-dependent plasticity
  rules that read [Ca²⁺]_i are impossible.
- **Worth addressing? LOW (now); MEDIUM if Ca²⁺-based plasticity is pursued.** Adding a leaky
  [Ca²⁺]_i integrator gated by spikes + `g_CaT`, feeding an SK conductance, is a moderate
  protected `sim/` edit. Only worth it alongside a Ca²⁺-plasticity goal.
- **Honest note:** **reasonable** as a phenomenological stand-in for SFA; the surrogate is even
  self-documented in the presets.

### 7. STP (Tsodyks-Markram) lacks Ca²⁺-4th-power release and asynchronous release; no quantal variability

- **Code:** `fused_stp_decay_recovery` (`sim/kernels.py:242`) + per-type U/τ_d/τ_f
  (`sim/config.py:148–157`). Release probability `U` is a fixed scalar per connection type.
- **Simplification:** `U` is not derived from a Ca²⁺ signal (no 4th-power Ca²⁺ dependence);
  only synchronous release; one conductance quantum per spike (no binomial quantal noise);
  no Syt-7 asynchronous tail.
- **Biology:** P_release ∝ [Ca²⁺]⁴ (cooperative synaptotagmin); async release via low-affinity
  Syt-7 (catalog J.20, line 3717; Kandel 6e Ch 15; Dodge & Rahamimoff). Catalog: *"partial …
  not explicitly Ca²⁺-fourth-power … asynchronous release not modeled."*
- **Impact:** the paired-pulse STP benchmark passes (PPR is reproduced), so the *macroscopic*
  facilitation/depression is faithful. The shortcut bites only deep Ca²⁺-manipulation assays
  and async-release timing. Has not bitten a project result.
- **Worth addressing? LOW.** Leave unless a Ca²⁺-channel-blocker experiment is the goal.
- **Honest note:** **reasonable** — STP captures the behavior the benchmarks check.

### 8. Default neuron model is Izhikevich (phenomenological), not biophysical, for almost all runs

- **Code:** `neuron_model_type` defaults to `IZHIKEVICH` (`sim/config.py:35`); essentially all
  conversational + nav runners use Izhikevich-2007 presets (`sim/enums.py`
  `DefaultIzhikevichParamsManager`). HH is available but `dt` must drop to 0.05 ms (10–20×
  slower; `sim/config.py:354`).
- **Simplification:** spikes are reset-and-recover abstractions; `v`/`u` are not ion
  conductances. No real Na/K gating, no true AP shape, no biophysical channel knobs during a
  run.
- **Biology:** Izhikevich (2003/2007) is an explicitly *reduced* model — it reproduces firing
  *patterns* cheaply, not membrane biophysics (HH = the faithful model).
- **Impact:** this is a *deliberate speed/fidelity trade*, and the right one for large networks
  and long training. It "bites" only when a result hinges on biophysical detail (channel
  pharmacology, exact AP-shape-dependent ephaptic/threshold effects) — none of the current
  arcs do. HH presets exist precisely for when that matters.
- **Worth addressing? NO (as a default).** Keep Izhikevich default; use HH selectively. Worth
  *documenting* prominently that "biology-grounded" results below the spike level are
  Izhikevich-level, not HH-level.
- **Honest note:** **reasonable / correct** engineering choice. Flagged only for honesty about
  what "biophysical" means in the project's claims.

### 9. Forward Euler integration (not exact/RK) for membrane dynamics

- **Code:** all dynamics kernels use explicit Euler (`v_new = v + dv·dt`,
  `sim/kernels.py:43,116,203`). HH compensates with `dt = 0.05 ms`; Izhikevich/AdEx run at
  `dt = 0.5–1.0 ms`. (Gating variables *do* use the exact first-order analytic update —
  `sim/kernels.py:95–103` — which is good.)
- **Simplification:** first-order integration; truncation error scales with `dt`. At
  `dt = 1.0 ms` (the conversational default) Izhikevich AP timing has ~ms-level error.
- **Biology:** n/a (numerics, not biology) — but it *interacts* with the temporal-code
  shortcuts (#3) to set the effective timing resolution.
- **Impact:** mostly absorbed by the per-model `dt` choices; the determinism work and `dt=1.0`
  conversational results are stable. Fine.
- **Worth addressing? NO.** Standard practice; the `dt` schedule already manages stability.
- **Honest note:** **reasonable.** Listed for completeness because it bounds timing fidelity.

---

## TIER 3 — Reasonable simplification / low (catalogued for completeness)

### 10. No glia at all (astrocytes, microglia, oligodendrocytes) — entire Cluster Q absent
- No astrocyte K⁺ buffering / glutamate uptake (Q.02), no tripartite synapse / gliotransmission
  / D-serine (Q.04, line 5577: *"missing — no astrocyte compartment, no D-serine, no Ca²⁺
  waves; NMDA-block formula is purely neuronal-side"*), no microglial pruning (Q.03), no myelin
  (Q.01). Kandel 6e Ch 6. **Reasonable** — well below the project's level of abstraction; the
  *functional* outcomes glia produce (weak-synapse elimination, K⁺ homeostasis) are either
  captured by structural plasticity / homeostasis or irrelevant to current goals. Only the
  tripartite-synapse D-serine modulation of NMDA could matter someday.

### 11. Homeostasis / synaptic scaling timescales compressed
- Homeostatic threshold EMA τ ≈ 5 s, adapt-rate slowed to ~0.5 mV/s (`sim/config.py:160–161`);
  synaptic scaling at "seconds" (`:167`). Real Turrigiano scaling is **hours–days** (Kandel 6e
  Ch 49). **Reasonable / necessary** — wall-clock forces compression; the *direction* and
  *sign* of homeostasis are correct, only the clock is sped up. Worth a one-line caveat when
  claiming homeostatic realism.

### 12. No transcription / late-LTP (gene-expression-dependent) consolidation tier
- No CREB/protein-synthesis weight tier; weights are single-timescale (catalog J.18, line 3697:
  *"one of the more important missing mechanisms for long-horizon memory"*). The project instead
  uses hippocampal SWR consolidation + lineage persistence. **Reasonable for now**, but the
  *closest real candidate* to add if very-long-horizon memory stability becomes a goal (a slow
  per-synapse "consolidation" floor resisting LTD).

### 13. AdEx/Izhikevich adaptation is the only SFA channel for non-HH runs
- For the default Izhikevich path, adaptation is the abstract `u` variable (one timescale);
  AdEx has one `w`. No multi-timescale adaptation (fast BK + slow sAHP + M). **Reasonable** for
  reduced models; subsumed by #6/#8.

### 14. Mg²⁺ block uses the standard static Jahr-Stevens sigmoid (no use-dependent unblock kinetics)
- `fused_nmda_update_and_current` (`sim/kernels.py:236`) applies `B(V)` instantaneously per
  step. Real Mg²⁺ unblock has finite kinetics. **Reasonable** — Jahr-Stevens is the field-
  standard; the instantaneous approximation is accurate at these `dt`. Listed only because it's
  a (small) kinetic shortcut on an otherwise-faithful NMDA model. **Genuinely fine.**

### 15. Inhibitory driving-force compensation via a propagation-strength scalar
- `inhibitory_propagation_strength = 0.105` is hand-scaled "for E_inh=−75 mV"
  (`sim/config.py:139`) rather than emerging from conductance units; similarly the historical
  "0.7× propagation scaling" (catalog J.10, line 3617). This is a **units/calibration shortcut**
  (a global gain knob standing in for proper nS→pA conversion), not a missing mechanism. **Mostly
  fine**, but it means absolute conductances aren't physical — worth knowing when comparing to
  literature nS values.

---

## Top 3 worth addressing (recommendation)

Ranked by **fidelity gained per unit effort, weighted by how much it bites the project's goals**:

1. **GABA_B→GIRK slow inhibitory arm (#1).** *Why:* it has **already** blocked a concrete,
   in-flight result (the spiking-SNc value subtraction, 2026-06-08) — the exact trigger for
   this audit. *Effort:* medium, additive protected `sim/` edit, with NMDA-alongside-AMPA as a
   proven template and a default that's byte-identical. Highest "already-bit + cheap" score.

2. **Per-pathway conduction delays (#3).** *Why:* the single highest *fidelity-per-effort* fix —
   the sim currently has *no* axonal delay at all (a hard 1-step), and the catalog flags it as
   the missing piece for both BG three-phase timing and any temporal/phase code (which the
   phasor/FHRR and rank-order work keep bumping into). *Effort:* medium, well-bounded ring-buffer
   per pathway, default 1-step = identical. Unlocks a whole class of temporal-code experiments
   cheaply.

3. **Two-compartment (soma + apical) neurons (#2) — strategic, not quick.** *Why:* this is the
   *most consequential* shortcut for the project's deepest goals — it is why dendritic credit
   assignment is a confirmed dead end and a load-bearing part of the composer-idealization
   limitation. *Effort:* large (months); recommend **scoping a minimal 2-compartment AdEx**
   rather than full multi-compartment, and only committing when apical-basal learning or
   perceptual inference is the deliberate next target. Listed third because impact is high but it
   is explicitly *not* a near-term win.

Everything in Tiers 2–3 is, on balance, a **reasonable reduced-model choice** that the project
should keep — with brief honesty caveats where "biological/biophysical" claims are made
(notably: results below the spike level are Izhikevich-level, not HH-level; homeostasis clocks
are compressed; absolute conductances are calibrated, not physical).

---

## Appendix — quick reference table

| # | Shortcut | Code anchor | Catalog | Tier | Address? |
|---|----------|-------------|---------|------|----------|
| 1 | GABA_A-only (no GABA_B/GIRK) | kernels.py:208; config.py:129 | C.02 (678) | 1 | **YES** |
| 2 | Point neurons (no dendrites) | kernels.py:32/48/184 | J.05/J.07 (2638,2648) | 1 | strategic |
| 3 | Uniform 1-step delay (no conduction delay) | bridge.py:5335,1951 | B.16 (369) | 1 | **YES** |
| 4 | Single-exp AMPA/GABA_A; AMPA+NMDA shared drive | kernels.py:211; bridge.py:5416 | C.01/C.02 | 2 | low |
| 5 | NM = global scalar (no volume gradient/subtypes) | neuromodulators.py:211,301 | C.21/J.13 (900,3647) | 2 | low |
| 6 | AHP = M-current (no Ca²⁺-gated SK/BK, no [Ca]i) | kernels.py:120; enums.py:294 | J (Ch 7/10) | 2 | low/med |
| 7 | STP: no Ca⁴ release, no async/quantal | kernels.py:242 | J.20 (3717) | 2 | low |
| 8 | Izhikevich default (phenomenological) | config.py:35; enums.py | — | 2 | no (doc) |
| 9 | Forward Euler integration | kernels.py:43,116,203 | — | 2 | no |
| 10 | No glia (Cluster Q) | (absent) | Q.01–Q.08 (5577) | 3 | no |
| 11 | Compressed homeostasis timescales | config.py:160,167 | Ch 49 | 3 | no (doc) |
| 12 | No late-LTP / transcription tier | (absent) | J.18 (3697) | 3 | maybe later |
| 13 | Single-timescale SFA (non-HH) | kernels.py:32,184 | — | 3 | no |
| 14 | Static Jahr-Stevens Mg block | kernels.py:236 | — | 3 | fine |
| 15 | Inhib driving-force via gain scalar | config.py:139 | J.10 (3617) | 3 | no (doc) |
