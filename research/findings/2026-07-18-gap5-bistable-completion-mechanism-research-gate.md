# Deep-research gate — a ROBUST bistable + specific CA3 pattern-completion attractor (gap #5)

**2026-07-18. READ-ONLY research gate at a CONFIRMED BOUNDARY.** Verdict up front: **robust bistable + specific
completion is ACHIEVABLE on this substrate — it is a verdict on the METHOD, not the capability. The weak-magnitude
boundary is a consequence of using the wrong recurrent element (the dendritic dAP *coincidence READOUT*) as the
attractor, without the *somatic slow-NMDA reverberatory excitation* + E/I working point that every canonical robust
attractor (Wang 2002, Amit–Brunel 1997, Kopsick–Ascoli 2024) uses. The cheapest fix is ALREADY BUILT and ALREADY
PROVEN bistable on this exact substrate (the D3 persistent-slot work). #1 = flip the `ca3→ca3` recurrent from
`coincidence_detector` to `exc_receptor="nmda_slow"` (Wang 2002), reuse-by-import, NO `sim/` edit.**

---

## 1. The boundary (restated)
From a PARTIAL cue (50% of a stored sparse assembly), the held-out members must strongly REACTIVATE, while
(a) the network is SILENT at rest (no always-on limit cycle) and (b) a PERMUTED cue (random non-assembly cells) does
NOT complete. Current status (`2026-07-18-gap5-ca3-functional-completion-CLOSED-...` retraction + boundary blocks):
genuine bistable+specific completion is DEMONSTRATED but WEAK (~5% held reactivation: cue 0.050 / perm 0.004 /
nocue 0.000 / rest 0.000). Every magnitude lever breaks the regime — a **magnitude-vs-bistability-vs-specificity
trilemma** with no wide window. The `ca3→ca3` recurrents already route through the dendritic dAP NMDA-spike plateau
(`coincidence_detector=True`), so it is NOT a pure-AMPA limitation.

---

## 2. What the primary sources actually say (read in depth)

### 2a. Wang 2002 / Brunel–Wang 2001 / Amit–Brunel 1997 — the bistability principle
(Wang, *Neuron* 2002 "Probabilistic decision making by slow reverberation…"; Amit & Brunel *Cereb. Cortex* 1997;
Brunel & Wang *J. Comput. Neurosci.* 2001. Parameters cross-checked against the NEST `Brunel_Wang_2001` reference
model.)

- **Two co-existing attractors (the definition of bistability).** Amit–Brunel 1997: a recurrent E/I network has
  **a spontaneous state at an arbitrarily LOW rate** (sitting on the *expansive* part of the neuron's f–I curve,
  fluctuation-driven, irregular firing) **coexisting with a sustained-activity attractor** at a higher rate (on the
  *contractive/saturating* part of the f–I curve). Memory = the network resting in the low state and being switched
  to the high state by a cue. **Both states must be stable simultaneously.** The project's retracted "self-sustaining"
  artifact is precisely the failure mode where the LOW state does not exist (mono-stable / always-on): weights grown
  to "completion scale" collapse the spontaneous fixed point.
- **Why SLOW NMDA is required (the load-bearing claim).** Wang 2002: *"recurrent excitation is dominated by a slow
  (and saturating) recurrent synaptic dynamics of NMDA receptors."* With ONLY fast AMPA (τ < 5 ms) at the recurrent
  synapses the network "can still show attractor dynamics, but these faster synapses cannot sustain the temporal
  integration" — and worse, fast synchronous AMPA recurrence produces oscillatory/synchronous runaway instead of a
  stable graded plateau. **The slow NMDA (τ_decay ≈ 100 ms) gives (i) a WIDE bistable window and (ii) temporal
  integration that REJECTS transient / non-specific input** — a brief random (permuted) cue does not build the NMDA
  conductance past the ignition threshold, so it cannot ignite the wrong assembly. This is *exactly* the
  specificity + silent-rest the boundary asks for, delivered by the synaptic time constant, not by a readout threshold.
- **Exact Wang 2002 synaptic parameters** (NEST `Brunel_Wang_2001` reference): **NMDA τ_rise 2.0 ms, NMDA τ_decay
  100.0 ms, AMPA τ 2.0 ms, GABA τ 5.0 ms, [Mg²⁺] 1.0 mM, C_m 500 pF, g_L 25 nS, E_ex 0 mV, E_in −70 mV, recurrent
  inhibitory weight w_in ≈ 15.** The persistent state is only stable when the potentiated within-pool weight w+ is in
  a band: strong enough that the high fixed point exists, but not so strong that the low fixed point disappears. The
  feedback inhibition sets the working point that keeps the spontaneous state stable and prevents runaway.

  > **The project's `nmda_recurrent` defaults ARE the Wang values.** `sim/config.py`: `nmda_recurrent_tau_rise_ms
  > = 2.0`, `nmda_recurrent_tau_decay_ms = 100.0`, `nmda_mg_concentration = 1.0`; and `ca3_fb_inhib` has been run at
  > 15–20 ≈ Wang's w_in 15. The substrate is already parameterised for the Wang attractor — it is simply not wired
  > into the `ca3→ca3` recurrent.

### 2b. Kopsick, Kilgore, Adam & Ascoli 2024 — the most realistic spiking CA3, robust completion WITH a silent rest
(PMC10996657, DOI [10.1101/2024.03.27.586909](https://doi.org/10.1101/2024.03.27.586909); *per PubMed / PMC.* 84,053
neurons, 176 M synapses, 8 Hippocampome-derived neuron types.) This is the decisive on-point reference and it
**directly reframes the project's trilemma.** How it gets ROBUST completion (50–70 % degraded cues) with a silent rest:

1. **The "silent rest" is a stable ASYNCHRONOUS LOW-RATE background, not a dead network.** *"The full-scale network
   exhibited asynchronous population activity while patterns were not presented, with each neuron type firing at rates
   consistent with those observed in vivo."* Completion happens ON TOP of this background = the Amit–Brunel spontaneous
   state. **The project hard-silences to a DEAD (zero) state and drives the cue into a network with no working point —
   there is no bistable window around a dead fixed point.**
2. **Moderate-SNR weights via homeostatic DIVISIVE DOWNSCALING — the direct answer to the trilemma.** After STDP grows
   the assembly, *"all PC-PC synaptic weights were re-normalized via synaptic divisive downscaling"* (synaptic
   homeostasis, ~slow-wave sleep) so the MEAN returns to baseline while the assembly **SNR** (within/between weight
   ratio) is preserved. Explicitly: *"effective learning… does not require synaptic saturation"* — reconstruction is
   near-optimal *"when only 10 % of assembly synapses had reached their maximum weight."* **The SNR does the completion,
   NOT raw magnitude.** The project grows the ABSOLUTE within-assembly weight to ~200× baseline ("completion scale") →
   that is the self-sustaining regime. Kopsick keeps the absolute weight moderate (90 % of synapses below max) and
   lets the *ratio* + inhibition + background do the work → silent rest AND robust completion, no trilemma.
3. **Per-cell recurrent drive is bounded: `w_max ∝ 1/assembly_size`** (300→~20, 150→40, 600→10). The convergent
   integration onto each PC is held constant regardless of assembly size — a built-in divisive normalization the
   project does not apply.
4. **Absolute assembly size 150–300 is the robust range; < 150 suffers pattern interference / weak completion; best =
   275 ≈ √N_PC.** **The project's `n_ca3=2000 × assembly_frac 0.008 = 16 neurons` is far below this** — 16 members is
   too few for redundant recurrent completion. Sparse (γ < 1 %) AND absolute-large (150+) cannot BOTH hold at
   n_ca3=2000 (0.8 % of 2000 = 16). This is a genuine scale factor.
5. **Diverse E/I inhibition confines completion in space AND time.** During completion *"the activity of each
   interneuron type remained similar to non-presentation periods"* → the inhibition keeps non-members at background
   (specificity) and the assembly reactivation is a **transient within one theta cycle** (cue in the first gamma
   window, completion read in the second half of the theta cycle via recurrent PC connections — "simple recall"),
   decaying back to background afterward. Silent rest is then trivial (there is no persistent high state to leak).
6. **Specificity = SNR, not a coincidence threshold.** A permuted cue's cells lack the strong within-assembly recurrent
   weights, so their recurrent drive cannot recruit held members. No dAP threshold is needed for the specificity.

**Kopsick's completion is achieved WITHOUT Wang slow-NMDA bistability** — it is an E/I-balanced *transient* completion
(moderate-SNR weights + background + inhibition + a read window). So there are TWO valid routes: a *persistent* bistable
attractor (Wang, §2a) and a *transient* E/I-balanced completion (Kopsick). **Both satisfy the project's gate**
(no-cue→silent, cue→held fires during the read window, perm→silent).

### 2c. The somatic-NMDA-recurrent vs dendritic-dAP-plateau distinction (the diagnosed root cause)
- **Dendritic dAP coincidence plateau** (`coincidence_detector=True`, `coincidence_tau_decay_ms=80`, Major-Larkum-
  Schiller / Poirazi-Mel): a supralinear **READOUT** — for each post cell, if ≥ k of its routed inputs COINCIDE in ONE
  step, inject an all-or-none plateau ADDITIVELY on top of AMPA. It is a coincidence detector, not reverberatory drive:
  it detects that the assembly is co-active, but it does not provide the *sustained, graded, self-consistent* recurrent
  EXCITATION that holds a stable high fixed point. Forcing it to also do the sustain is what creates the trilemma
  (raise k_thresh → specific but collapses; lower → floods; raise recurrent cap → self-sustains).
- **Somatic slow-NMDA recurrent** (`exc_receptor="nmda_slow"`, `enable_nmda_recurrent=True`, τ_decay 100 ms): the
  routed recurrent synapses' fast-AMPA is SUPPRESSED and REPLACED by a separate slow, Mg²⁺-self-limiting NMDA
  conductance (`bridge.py:6404` block). This IS the Wang reverberatory excitation for the attractor itself. It gives
  the graded, non-synchronous, temporally-integrating recurrent drive that has a stable low state AND a stable high
  state. **This is the element the boundary correctly identifies as missing from the `ca3→ca3` recurrent.**

---

## 3. Substrate inventory — the mechanism is ALREADY BUILT and ALREADY PROVEN BISTABLE
- **`exc_receptor="nmda_slow"` + `enable_nmda_recurrent`** (`sim/regions.py:330-341`, `sim/config.py:138-156`,
  `sim/bridge.py:6404-6439`): the Wang somatic slow-NMDA recurrent, per-pathway routed, byte-identical when off.
  Defaults = Wang values (τ 2/100, Mg 1.0).
- **`build_persistent_slot`** (`research/runners/_d3_persistent_slot_derisk.py`) — a **fully-proven, reusable
  exemplar** of this exact mechanism on this exact substrate. It EXPLICITLY abandons `internal_density` (AMPA, *"it
  cannot hold — decays in ~5 ms"*) in favour of a `exc_receptor="nmda_slow"` self-pathway (`recur=25.0`), + a shared
  FS pool (`exc_to_fs=1.4`, `fs_to_exc=10.0`) for the E/I working point. Its de-risk PROVES the bistability the boundary
  needs: the slot **HOLDS its winner with external input identically ZERO** (persistent state; Amit-Brunel/Wang 2002),
  is a **clean one-of-K** (selectivity > 0.8), the **NO-RECURRENCE control cannot hold** (norecur_hold ≈ 0), input
  alone does NOT overwrite it, and a reset SHORTER than τ_NMDA does not either (the residual conductance re-ignites) —
  i.e. genuine hysteresis. The `Reset...` helper already knows to clear `cp_conductance_g_nmda_recurrent` for a true
  silence. **This is the CA3-completion mechanism minus the "distributed assembly instead of one-of-K pool" wrapper.**
- The current CA3 derisk (`_riii_ca3_synchronous_assembly_derisk.py` → `_build(..., coincidence=True)`) flips
  `coincidence_detector=True` on `ca3→ca3` (dendritic dAP), NOT `exc_receptor="nmda_slow"`. **The one-line change is
  the whole #1 recommendation.** dt = 1.0 ms; `recall_steps ≈ 60–100` ⇒ the read window is only **~0.6–1 × the NMDA
  τ_decay (100 ms)** — borderline too short for the slow conductance to build up (a concrete untried lever).

---

## 4. Is the weak-magnitude a fundamental boundary, or a missed lever? → **MISSED LEVERS (concrete, multiple)**
Every one of these is untried in the current CA3 derisk and each is load-bearing in the canonical robust models:

| # | Missed lever | Source | Why it caps the current magnitude |
|---|---|---|---|
| A | Recurrent is a **coincidence READOUT**, not **reverberatory excitation** | Wang 2002 §2a | a readout detects co-activity; it can't hold a stable high fixed point → the trilemma |
| B | **Absolute assembly = 16**, robust range **150–300** | Kopsick §2b(4) | too few members for redundant completion; < 150 = interference/weak |
| C | Grows **absolute** weight to self-sustain scale; should keep **moderate absolute + high SNR** via **homeostatic divisive downscaling** | Kopsick §2b(2) | the direct trilemma resolution — SNR completes, magnitude self-sustains |
| D | **No background working point** (hard-silence to DEAD, then drive cue) | Amit-Brunel / Kopsick §2b(1) | the bistable window exists only around the fluctuation-driven spontaneous state |
| E | **Read window ~0.6–1 × τ_NMDA** (60–100 ms) | Wang §2a temporal integration | slow NMDA needs longer than 1 τ to build; extend `recall_steps` to 250–400 |
| F | No **`w_max ∝ 1/assembly_size`** per-cell normalization | Kopsick §2b(3) | unbounded per-cell recurrent drive → runaway at the scale that completes |
| G | Feedback inhibition is **proportional-to-firing / zero-at-rest**, not an **Amit-Brunel inhibitory set-point** | Wang §2a, Kopsick §2b(5) | a set-point stabilizes the spontaneous state AND caps completion spread |

⇒ the "no wide window" is an artifact of fighting the trilemma with the WRONG element (a readout) at the WRONG scale
(16-cell assembly) with the WRONG weight regime (absolute-magnitude, no downscaling) around NO working point. It is
not an irreducible substrate limit.

---

## 5. Ranked mechanisms — cheapest-first

### ⭐ #1 (CHEAPEST, reuse-proven, NO `sim/` edit): Somatic slow-NMDA recurrent on `ca3→ca3` (Wang 2002)
**The mechanism.** Replace the dendritic-dAP coincidence recurrent with the somatic slow-NMDA reverberatory
excitation — the element Wang uses for the attractor itself. Slow NMDA (τ 100 ms) gives the wide bistable window +
temporal integration that rejects the permuted transient; AMPA is suppressed on those synapses so there is no
synchronous runaway.
- **Reusable machinery (already built + proven):** `exc_receptor="nmda_slow"`, `enable_nmda_recurrent=True`,
  `nmda_recurrent_tau_decay_ms=100` (all defaults = Wang); the exact recipe + operating point from
  `build_persistent_slot` (`recur≈25`, shared FS pool `exc_to_fs≈1.4`, `fs_to_exc≈10`); the existing `ca3_fb_inhib`
  basket wiring (set to ≈ Wang w_in 15); the existing hard-silence-that-clears-`g_nmda_recurrent`; the existing
  bistable gate. The change vs the current derisk: in `_build`, tag the returned `ca3→ca3` `RegionPathway` with
  `exc_receptor="nmda_slow"` (and `cfg.enable_nmda_recurrent=True`) INSTEAD of `coincidence_detector=True`. Runner-side
  flip; **NO `sim/` edit** (mirrors the coincidence flip already there).
- **The learned attractor still applies:** the within-assembly weights the formation recipe already grows (continuous
  drive + coact_thresh + heterosynaptic competition) become the potentiated w+ of the NMDA recurrent — now they
  reverberate gradedly instead of feeding a coincidence plateau. Keep the assembly-selective `ca3_fb_inhib` (it is the
  E/I set-point that caps completion spread and keeps non-members silent — the Kim-Kim 2025 role, doubling as Wang's
  inhibitory working point).
- **Cheap-first de-risk (the existing `bistable=True` gate, 6-seed 42/43/44/100/101/102):**
  1. NO-CUE → held SILENT (rest ≤ 0.05): the LOW attractor is stable (the D3 slot already shows the NMDA recurrent has
     a stable low state until ignited).
  2. CORRECT 50 % cue → held FIRES STRONGLY (target ≫ the 5 % floor; aim held ≥ 0.20 and ≥ 3× nocue and ≥ 3× perm).
  3. PERMUTED cue → held SILENT (the slow NMDA rejects the transient; ≥ 3× specificity).
  4. Sweep only w+ (`ca3w`/hebb scale) × `ca3_fb_inhib` × `recall_steps` (extend to 250–400 = 2.5–4 τ_NMDA, lever E).
- **Anti-cheats:** (a) NO-ENCODING (encode_drive=0) → held 0 (attractor load-bearing); (b) NO-RECURRENCE
  (`enable_nmda_recurrent=False`) → completion collapses (the reverberation, not the cue drive, completes); (c)
  PERMUTED-cue (already in the gate) → held silent; (d) NO-CUE (already) → held silent; (e) hard-silence must clear
  `cp_conductance_g_nmda_recurrent` (the D3 lesson — else a bistable attractor re-ignites and looks self-sustaining).
- **Risk & fallback:** if a *distributed* assembly (vs D3's one-of-K discrete pool) does not settle into a clean high
  state at recur≈25 — because a sparse assembly has fewer within-pool synapses than a dense pool — the fix is to raise
  w+ / density modestly (still moderate absolute via downscaling, lever C) and/or enlarge the absolute assembly
  (lever B). That is exactly #2/#3.

### #2 (Kopsick E/I-balanced transient completion — reframes the trilemma; more levers, needs a scale bump)
Keep an AMPA (or nmda_slow) recurrent but adopt Kopsick's robust-completion recipe: (a) **homeostatic divisive
downscaling** of all `ca3→ca3` weights after formation so the mean returns to baseline while the assembly SNR is
preserved (moderate absolute weight → no self-sustain; SNR → completion); (b) **`w_max ∝ 1/assembly_size`** (bounded
per-cell drive); (c) a **stable asynchronous background** working point (a small lognormal background current to CA3 +
the FS pool) instead of hard-silence-to-dead; (d) a **larger absolute assembly (150–300)** — needs `n_ca3` ≈ 19k–40k to
keep γ < 1 % AND size ≥ 150 (a GPU/scale bump, per the "long local runs OK" discipline; measure VRAM first).
- **Reusable:** the existing formation recipe (STDP/rate-window LTP), the `ca3_fb_inhib` basket; the divisive
  downscaling is a one-line renormalize of the learned `ca3→ca3` weights between train and test (Kopsick does exactly
  this); `n_ca3` scaling is a parameter.
- **De-risk / anti-cheats:** the same bistable gate; ADD a control that the downscaling is load-bearing (skip it →
  self-sustain returns) and that the SNR (not absolute magnitude) predicts completion (Kopsick's SNR-plateau finding:
  completion saturates at ~94 % SNR with 90 % of synapses below max).
- **Why #2 not #1:** more tuning knobs + a scale bump, but it is the field's most-realistic proven route and it
  RESOLVES the trilemma at the concept level (magnitude self-sustains; SNR completes). Run it if #1 alone underpowers.

### #3 (the robust target — COMBINE: Wang NMDA reverberation + Kopsick E/I working point + downscaling)
The "proper E-I-balanced attractor network" the boundary names. Somatic slow-NMDA recurrent (#1) for the wide bistable
window and transient-rejection, PLUS Kopsick's moderate-SNR downscaled weights + `w_max ∝ 1/size` + background working
point + Amit-Brunel inhibitory set-point (#2) for the silent background and specificity. Highest expected robustness;
do #1 first (it may suffice), fold in #2's pieces only for the levers #1 leaves on the table.

### Pre-step (near-zero cost, run on the EXISTING derisk before building anything)
On the current coincidence-plateau derisk, first test levers D + E alone: (i) don't hard-silence to dead — settle to a
low background under a small tonic input, then cue; (ii) extend `recall_steps` to 250–400. If completion strengthens
even a little, it confirms the working-point + integration-time diagnosis and de-risks #1 for near-free. (Expected:
partial — the coincidence readout still can't hold a high state — but it isolates D/E from the recurrent-element change.)

---

## 6. Verdict
**Robust bistable + specific CA3 completion is ACHIEVABLE on this point-neuron substrate — NOT an irreducible limit.**
The weak-magnitude boundary is fully explained by a mechanism mismatch (a dendritic-dAP *coincidence readout* standing
in for *somatic reverberatory excitation*) compounded by scale (16-cell assembly vs the robust 150–300), weight regime
(absolute-magnitude vs moderate-SNR-with-downscaling), read window (< 1 τ_NMDA), and the absence of a background
working point — every one a lever the canonical robust models (Wang 2002, Amit-Brunel 1997, Kopsick-Ascoli 2024) use
and the current derisk omits.

**Cheapest path: #1 — flip the `ca3→ca3` recurrent to `exc_receptor="nmda_slow"` (Wang 2002).** The mechanism is
already built, its defaults are already the Wang values, and it is already PROVEN bistable on this exact substrate by
the D3 persistent-slot de-risk (holds with zero input, clean state, no-recurrence-control collapses). It is a
runner-side one-line pathway flip, NO `sim/` edit, de-risked by the existing 6-seed bistable gate with the mandatory
no-cue + permuted anti-cheats. If a distributed sparse assembly underpowers at the D3 operating point, fold in #2's
moderate-SNR downscaling + larger absolute assembly (#3). Recommend building #1 next.

---

### Sources
- Wang X-J. *Probabilistic decision making by slow reverberation in cortical circuits.* **Neuron** 36:955–968 (2002).
  PMID 12467598. [cns.nyu.edu PDF](https://www.cns.nyu.edu/wanglab/publications/pdf/wang2002_decision.pdf).
- Amit D.J., Brunel N. *Model of global spontaneous activity and local structured activity during delay periods in the
  cerebral cortex.* **Cereb. Cortex** 7:237–252 (1997). — the coexisting low (spontaneous) / high (persistent)
  attractor working point.
- Brunel N., Wang X-J. (2001) synaptic/network parameters; cross-checked vs the NEST `Brunel_Wang_2001` reference
  model ([NMDA τ 2/100 ms, AMPA 2, GABA 5, w_in 15, Mg 1.0](https://nest-simulator.readthedocs.io/en/stable/model_details/Brunel_Wang_2001_Model_Approximation.html)).
- Kopsick J.D., Kilgore J.A., Adam G.C., Ascoli G.A. *Formation and Retrieval of Cell Assemblies in a Biologically
  Realistic Spiking Neural Network Model of Area CA3 in the Mouse Hippocampus.* (2024) *per PubMed / PMC*
  PMC10996657, DOI [10.1101/2024.03.27.586909](https://doi.org/10.1101/2024.03.27.586909). — robust completion from
  50–70 % degraded cues with a silent (asynchronous-background) rest; homeostatic downscaling, moderate SNR,
  `w_max ∝ 1/size`, assembly 150–300, diverse E/I inhibition.
- Marr D. (1971) CA3 recurrent autoassociator; Kandel 6e Ch. 54 (CA3 pattern completion, NMDA role).
- Project reuse: `research/runners/_d3_persistent_slot_derisk.py` (`build_persistent_slot` — the proven on-substrate
  Wang NMDA bistable slot); `sim/{config,regions,bridge}.py` `exc_receptor="nmda_slow"` / `enable_nmda_recurrent`
  machinery; `research/runners/_riii_ca3_{coincidence_completion,synchronous_assembly}_derisk.py` (the current CA3
  derisk + bistable gate).
