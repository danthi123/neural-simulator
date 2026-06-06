# Biologizing the action-selection READOUT (cheat N6): catalog + literature synthesis and a ranked, cited mechanism menu — 2026-06-06

**Type:** RESEARCH ONLY (reading + literature synthesis). No code edited, nothing run on the GPU,
`research/runners/g11_bg_runner.py` not touched (a parallel implementation de-risk owns it). This doc is
the foundation for converting cheat **N6** — the host-side argmax over spiking-pool firing rates — into a
decision that EMERGES from spiking dynamics.

## The problem, restated precisely

The navigation agent's BG cascade already runs genuine GPi→thalamus disinhibition (N8 resolved, multi-seed
GO; `2026-06-06-N8N6-combined-readout-GO.md`). Under disinhibition the **released thalamus is the cleanly
selective signal** — only the chosen action's `thal_X` fires, the other three sit at exactly 0.000. The OPEN
residual is N6: committing that released-thalamus winner to a motor action **as a spiking decision**, not a
host `argmax` over pool rates.

The two facts that constrain every candidate:

1. **The released signal is CLEAN but WEAK.** The chosen motor fires ~0.016 spikes/neuron/step (one synapse
   downstream of thal, 10 neurons). The thalamus itself is the strong-and-clean point; motor is the weak end.
2. **Naive instantaneous spiking WTAs FAILED** (cheat-5 multi-goal nav distance, LOWER is better; thal
   host-argmax target = **2.3**):
   - motor-pool WTA → **14.7** (drove competition from the weak motor counts).
   - thalamic-reticular (TRN) WTA on the relay → **20.0** (put the competition ON `thal_X`, corrupting the
     same signal that drives the thal→motor cascade and navigation).
   - The current opt-in `--readout-source spiking_wta` (a read-only `thal_X → sel_X → sel_FS_X → sel_Y`
     soft-WTA) is the parallel de-risk's subject; its score was PENDING at the time of this research.

**Key code fact (verified `g11_bg_runner.py:1421–1470`):** the existing `sel_X` selection pools are declared
with `internal_density=0.0` and `exc_weight_mean=0.0` — i.e. **NO recurrent self-excitation within a
selection pool**. The layer is a pure *feedforward-driven, instantaneously-competing* soft-WTA. As the
literature below makes decisive, that missing recurrent self-excitation is precisely the ingredient biology
uses to (a) amplify a weak input and (b) integrate it over time to a committed threshold. This is the single
highest-leverage change and it is purely additive.

---

## STEP 1 — The catalog (`references/feature-catalog.md`, `catalog-build` branch)

The project's reference catalog is `references/feature-catalog.md` on the `catalog-build` branch (~323
entries, Kandel 6e + supplements; `docs/biology.md` is the curated subset). The entries that bear directly
on committing a BG/thalamo-cortical winner to a motor decision:

### Basal-ganglia action selection / commitment (Cluster A)

- **A.04 — BG output disinhibition is selective: competitive WTA at GPi/SNr.** *"'Selected' channel =
  strongest inhibitory input from striatum → GPi/SNr neuron silenced → thalamic / SC target released.
  Non-selected channels remain inhibited.* **Selection is an emergent property of the entire reentrant
  network.**" Cites Kandel 6e Ch 38 p 939–942 (Mink 1996; Redgrave selection model). **This is the catalog's
  charter for N6: selection should EMERGE from the reentrant network, not be computed by a host argmax.** Sim
  status: implemented as the per-action cascade + MSN lateral inhibition.

- **A.05 — Reentrant cortico-BG-thalamo-cortical loops.** The selected action is committed via positive
  feedback through `thal_X → cortex_X` reentry (already partially modeled via cluster-A `--enable-cluster-a-
  closed-loop`). Kandel 6e Ch 38 p 943–948. Mechanism: the disinhibited thalamus re-excites its source
  cortex, sustaining the chosen action over short delays — the biological "latch" on the decision.

- **A.07 — Subcortical BG loops → superior colliculus / MLR.** *"Not all BG output goes to cortex. Direct
  projections to SC release saccades; to MLR release locomotion."* The BG output gate's most ancient and
  direct commitment target is a brainstem **burst** structure, not cortex. Kandel 6e Ch 38 p 938–942.

### Decision-making cluster (Cluster G, Kandel Ch 56) — the accumulator

- **G.15 — Signal-detection decision rule.** Threshold on noisy evidence; criterion encodes priors + costs.
  Sim status: *"BG cascade implements something close to a criterion (winner-take-all over per-action
  striatum)."* Kandel 6e Ch 56 p 1393–1395.

- **G.16 — Drift-diffusion / bounded evidence accumulation.** *"accumulator integrates difference (right −
  left) over time; decision terminates when accumulator hits ±bound … Two anti-correlated accumulators (one
  per choice) terminate at first-bound-crossing."* **The catalog explicitly flags the sim status: "BG cascade
  with motor-output thresholding is *functionally equivalent to* a bounded accumulator … No explicit RT
  analysis or coherence-vs-accuracy curve exists."** Kandel 6e Ch 56 p 1399–1404. **This is the missing
  piece for the weak signal: integrate over time to a bound rather than decide instantaneously.**

- **G.17 — LIP / parietal accumulator.** *"Persistent ramping firing during evidence accumulation; threshold
  reached just before saccade … Firing rate ≈ accumulated logLR … the unchosen-direction cell's firing
  terminates without crossing bound."* Sim status: `cortex_X` pools "ramp toward action threshold; not yet
  validated against ramp-with-coherence signature." Kandel 6e Ch 56 p 1402–1404. **The biological readout is
  a ramp-to-threshold, not an argmax-of-rates.**

- **G.18 — logLR accumulation in LIP** (evidence need not be sensory; any reliability-weighted evidence
  integrates additively). G.19 — affordance/Gibson-Shadlen ("knowing = provisional commitment to action").

### Motor commitment / burst generators (Cluster H)

- **H.24 — Saccade generator (pontine reticular burst circuit).** *"Saccade is initiated when OPNs
  [omnipause neurons] are silenced, releasing EBNs [excitatory burst neurons] to drive a high-velocity burst
  … Pulse-step waveform; pulse drives the saccade, step holds gaze. Burst duration determines amplitude."*
  Kandel 6e Ch 35 p 868–880. **A tonically-inhibited burst circuit that, when released (disinhibited),
  produces an all-or-none committed motor output — exactly the project's released-thalamus → commit shape.**

- **H.25 — Superior colliculus saccade map.** *"SC integrates visual + auditory + cognitive inputs into a
  'where to look next' decision; output → pontine reticular saccade generator. Receives BG (SNr) tonic
  inhibition; selection by SNr disinhibition (A.07)."* Kandel 6e Ch 35 p 875–882. **The SC is the canonical
  structure that converts BG disinhibition into a committed, thresholded, all-or-none motor burst.**

- **H.08 — Renshaw recurrent inhibition** / **H.14 — half-center (mutual inhibition + adaptation)** /
  **H.07 — reciprocal inhibition**: the spinal/brainstem inhibitory-geometry primitives. The catalog notes
  v3 lateral inhibition is at the striatal level, "different mechanism" from motoneuron-level Renshaw.

### Striatal WTA reality (Cluster B) — the cautionary note

- **B.04 — MSN lateral inhibition.** The catalog's deepest WTA correction, citing Wilson 2007 (PBR-160 ch 6)
  and Tepper-Koós 2017: *"Mutual inhibition does not produce strong competitive interactions"* in striatum;
  MSN→MSN reciprocal coupling is at chance level (~1/38), small unitary IPSPs (<0.5 mV at soma), high
  failure. **The dominant biological WTA substrate is FEEDFORWARD FSI inhibition, not symmetric collateral
  feedback.** This is the catalog's empirical warning that mirrors the Rutishauser stability result below: a
  naive symmetric mutual-inhibition WTA is a weak, unreliable selector.

**Catalog takeaway:** the catalog does NOT contain a recurrent-attractor decision model or the
Rutishauser-Douglas-Slotine stability conditions explicitly — it flags the BG cascade as a *functional*
accumulator (G.16) and points to SC burst commitment (H.24/H.25), but the *explicit accumulate-to-threshold
and the stable-WTA conditions* live in the primary literature. The literature search (Step 2) fills exactly
that gap, and it converges hard on a single mechanism.

---

## STEP 2 — How the brain SELECTS + COMMITS an action (literature synthesis)

The primate decision-making literature describes a **two-stage** architecture that is a near-perfect match for
the project's situation (clean-but-weak upstream signal + need to commit). The stages are:

### Stage 1 — Integrate weak evidence over time via a recurrent attractor (the accumulator)

**Wang 2002, "Probabilistic decision making by slow reverberation in cortical circuits"** (Neuron 36:955–968)
is the canonical spiking model. Two selective excitatory populations, each with **recurrent self-excitation**,
compete via a **shared inhibitory interneuron pool**. The decisive properties:

- *"slow recurrent excitation and feedback inhibition produce attractor dynamics that amplify the difference
  between conflicting inputs and generate a binary choice."*
- *"A long integration time constant is achieved biophysically … by a combination of slow synapses dominated
  by NMDA receptors and strong network recurrency"* — network time constant ≈ τ_syn / |1 − w_rec|, reaching
  ~1 s, enabling ramping over hundreds of ms before divergence.
- *"sensory data are first integrated over time in a graded fashion, followed by winner-take-all competition
  leading to a binary choice."* (Wang review, *Decision Making in Recurrent Neuronal Circuits*, Neuron 2008.)

**Why this is THE answer to "clean but weak":** recurrent self-excitation is an *amplifier and integrator*. A
small, weak, but consistent input is accumulated over time and amplified by re-excitation until it crosses a
bound — exactly the regime an instantaneous WTA cannot win. The empirical correlate is **Mazurek-Roitman-
Ditterich-Shadlen 2003** (Cerebral Cortex 13:1257–1269): LIP firing ramps as evidence accumulates and *"a
threshold level of LIP activity (≈55 spikes/s) appears to mark the completion of the decision process and to
govern the tradeoff between accuracy and speed."*

**The canonical-microcircuit grounding (Douglas & Martin):** this is not special to decision cortex — it is
the general cortical motif, and the framing is uncannily exact for our thalamus problem: *"The thalamic input
is relatively limited in terms of numbers of synapses, but is amplified by strong recurrent excitatory
connections … selectively amplifying the thalamic inputs. Inhibition is needed to oppose or modulate this
potentially strong excitation."* The brain's answer to a weak thalamic drive is **recurrent excitatory
amplification restrained by inhibition** — precisely what the current `sel_X` pools lack.

### Stage 2 — Commit the accumulated signal to a motor burst via a downstream threshold (BG→SC)

**Lo & Wang 2006, "Cortico-basal ganglia circuit mechanism for a decision threshold in reaction time tasks"**
(Nat. Neurosci. 9:956–963) is the spiking model of the COMMIT step, and it is the single most on-point paper
for this conversion:

- *"local dynamics in the superior colliculus gives rise to an all-or-none burst response that signals
  threshold crossing in upstream cortical neurons."*
- The SNr maintains tonic inhibition of SC movement neurons; cortical drive activates caudate → caudate
  silences SNr → **SC is disinhibited and produces an all-or-none burst** (this is the project's exact
  released-thalamus geometry, one structure over).
- *"the decision threshold is much more readily adjustable by tuning the synaptic strength of cortico-
  striatal pathway"* — the threshold is set upstream (DA-modulated cortico-striatal weights), giving a
  biological speed-accuracy knob.

**Modern causal confirmation — Stine, Trautmann, Jeurissen, Shadlen 2023, "A neural mechanism for terminating
decisions"** (Neuron; PMC10565788): simultaneous LIP + SC recording during decisions shows a clean **division
of labor** that is decisive for our design:

- *"Single-trial activity in LIP approximates drift-diffusion — the accumulation of noise plus signal"* while
  *"Single-trial activity in SC is qualitatively different,"* showing bursts that *"manifest a threshold
  mechanism applied to signals represented in LIP to terminate the decision."*
- *"SC applies a threshold to the drift-diffusion signal in LIP using a combination of the firing rate and
  its derivative."* When the accumulator is weak, SC inhibition is strong (premature bursts are suppressed);
  when strong, SC is *"unleashed to generate a large saccadic burst."*
- **SC inactivation** dissociates the upticks from termination: *"SC inactivation had little to no effect on
  the magnitude and frequency of upticks in LIP activity. Inactivation simply dissociates upticks from
  decision termination"* and *"led to an increase in LIP firing rate … more gradual"* buildup (i.e. a raised
  threshold). **The accumulator keeps accumulating; only the COMMITMENT fails.**

**Synthesis of Stage 1 + Stage 2 onto the project:** the released `thal_X` is the *input to* an accumulator,
not the decision. Biology (i) re-excites a decision pool to amplify and integrate the weak released drive
(Wang 2002 / LIP), then (ii) hands the accumulated ramp to a *separate* downstream burst/threshold structure
that fires all-or-none when a rate-plus-derivative bound is crossed (Lo-Wang SC / Stine 2023). The host argmax
is standing in for BOTH stages at once, instantaneously — which is why an instantaneous spiking WTA on a weak
signal cannot replace it. **Add the integration; add the threshold-burst.**

### The stability constraint on any WTA — Rutishauser-Douglas-Slotine

**Rutishauser, Douglas, Slotine 2011, "Collective stability of networks of winner-take-all circuits"** (Neural
Computation 23:735–773) and the companion **Rutishauser et al., "Competition through selective inhibitory
synchrony"** establish the conditions for a *stable* spiking soft-WTA:

- A WTA needs **recurrent self-excitation** (gain α) for *selective amplification, signal restoration, and
  decision-making* — the same recurrent excitation Wang 2002/Douglas-Martin require. There is *"a critical
  trade-off between providing feedback strong enough to support sophisticated computations while maintaining
  overall circuit stability."*
- Via **contraction theory**, the network converges exponentially to its fixed point iff its generalized
  Jacobian is negative definite. The self-excitation gain α gates the behavior: **α < 1 permits a stable soft
  WTA; α > 1 (hard WTA, a single forced winner) imposes stricter constraints** and is where naive designs go
  unstable (runaway winner or oscillation).
- The competition must be mediated by inhibition that is **structured/selective**, not a naive symmetric
  global blanket — *"selective inhibitory synchrony."* This is the same lesson the catalog's B.04 teaches
  empirically (symmetric MSN→MSN mutual inhibition is a weak, unreliable selector; feedforward FSI inhibition
  is the real substrate). **The project's conversational side already hit this exact wall** (a naive symmetric
  soft-inhibition WTA was unstable), so it is a known, real risk here.

**Constraint takeaway:** whatever the chosen mechanism, it must include *recurrent self-excitation tuned below
the hard-WTA instability* (soft-WTA, α<1) and *structured* (per-pool, feedforward-style) inhibition. The
current `sel_X` layer satisfies the inhibition geometry (per-pool `sel_FS_X`) but has **α = 0** (no self-
excitation), so it is the degenerate "all-feedforward, no amplification, no integration" case — robust but
unable to manufacture a winner from a weak input. Adding modest self-excitation moves it from a passive
comparator into a genuine amplifying integrator while staying inside the stable soft-WTA regime.

---

## STEP 3 — Ranked, cited menu (biological grounding × likelihood of committing a CLEAN-but-WEAK signal)

Ranking axis: **biological grounding × probability of converting a clean-but-weak released signal into a
committed spiking decision** (and not re-failing the two known failure modes: driving from the weak end;
corrupting the relay).

### ★ #1 (RECOMMENDED) — Thalamus-driven recurrent ACCUMULATOR + downstream burst-threshold readout (Wang-2002 attractor → Lo-Wang/SC commit)

The two-stage biology, realized as a **read-only** layer fed by the strong clean thalamus. This is the
existing `spiking_wta` `sel_X` layer with the **one missing ingredient added: recurrent self-excitation**
inside each `sel_X` pool (turning the instantaneous comparator into an accumulator), optionally followed by a
dedicated all-or-none burst pool that reads the ramp.

- **Biology:** Wang 2002 (Neuron) attractor decision network; Lo & Wang 2006 (Nat. Neurosci.) BG→SC decision
  threshold; Stine-Shadlen 2023 (Neuron) LIP-accumulate/SC-commit division of labor; Douglas-Martin canonical
  microcircuit (recurrent amplification of weak thalamic input); Rutishauser-Douglas-Slotine 2011 (soft-WTA
  stability, α<1). Catalog A.04, A.07, G.16, G.17, H.24, H.25.
- **Why it handles a weak clean signal (the crux):** recurrent self-excitation *amplifies and integrates*
  the weak released drive over time (network τ = τ_syn/|1−w_rec|), so a small consistent input ramps to a
  bound that an instantaneous WTA can never reach. The downstream burst pool only fires when the ramp crosses
  threshold (rate + derivative; Stine 2023), giving a decisive all-or-none commit. The two prior failures are
  both avoided: (a) the competition is driven from the STRONG thalamus, not the weak motor end; (b) it is a
  pure read-only tap (no back-projection to `thal_X`), so the relay and navigation dynamics are byte-identical
  to the thal-readout run.
- **On-substrate realization (additive, runner-side, NO `sim/` edit):** extend the existing `sel_X`/`sel_FS_X`
  layer in `build_bg_brain_regions`:
  - Give each `sel_X` **recurrent self-excitation**: set `internal_density ≈ 0.2–0.4` and `exc_weight_mean`
    to a modest value (the amplification gain — tune to soft-WTA, α<1, i.e. enough to ramp/hold but not self-
    ignite without thalamic drive). Ideally route this recurrence through **NMDA** (the bridge already has
    per-region NMDA via `--enable-pfc-nmda`/`enable_nmda`; the cortex/PFC NMDA mask pattern is the precedent)
    so the integration time constant is biological (slow reverberation, Wang 2002) rather than AMPA-fast.
  - Keep `thal_X → sel_X` feedforward (the evidence input) and the per-pool `sel_X → sel_FS_X → sel_Y≠X`
    inhibition (the structured competition, Rutishauser selective inhibition; same idiom as MSN/motor WTA).
  - **Add a dedicated burst/commit pool** `commit_X` (the SC/EBN analogue): `sel_X → commit_X` excitatory,
    with `commit_X` held near-threshold and *tonically inhibited* (a shared `commit_OPN` omnipause-style pool,
    H.24) so it stays silent until `sel_X` ramps past the bound, then fires an all-or-none burst. The host
    reads which `commit_X` burst — a thresholded event, not an argmax of graded rates. (Minimal variant:
    skip `commit_X` and simply threshold the winning `sel_X` rate; the dedicated burst pool is the more
    faithful Lo-Wang/SC realization and is more decisive on weak signals.)
  - **Optional reentry (A.05):** a weak `sel_X → cortex_X` (or `thal_X → cortex_X`, the cluster-A closed loop
    that already exists) latches the chosen action over the readout window — the biological hold that the
    conversational side found is needed when a bursty winner must persist for a downstream read.
- **Speed-accuracy knob (free, biological):** the bound is set by the cortico-striatal weight / `sel`
  excitability (Lo-Wang) — a single tunable that trades commit latency vs reliability, and the existing DA
  modulation is the biological controller for it.
- **Risk:** tuning the self-excitation gain (must stay α<1 soft-WTA per RDS; too high → self-igniting winner
  independent of thalamus; too low → no amplification, reverts to the present passive comparator). NMDA-slow
  recurrence is the stabilizer (Wang 2002 shows the slow time constant is what makes the integration robust).
  This is the *only* candidate whose mechanism is specifically designed for weak evidence — it is the
  recommendation.

### #2 — Rutishauser-conditioned soft-WTA (self-excitation + structured inhibition), instantaneous

The same recurrent-self-excitation + structured-inhibition circuit, but tuned as an *instantaneous* amplifier
(fast/AMPA recurrence, no explicit temporal integration or burst pool). Essentially #1 minus the slow-NMDA
integration and the dedicated commit burst.

- **Biology:** Rutishauser-Douglas-Slotine 2011; Douglas-Martin canonical microcircuit; catalog A.04, B.04.
- **Why it might handle the weak signal:** self-excitation provides *signal restoration / selective
  amplification* even without temporal integration — it can sharpen a weak-but-clean winner that a gain-0
  comparator passes through marginally. **Why it might not:** with no integration, a *weak* input may not
  reach the basin of the winning attractor within a single step; amplification of an instantaneously-weak
  signal is less robust than amplification-plus-integration. This is the natural next step *if* the current
  gain-0 `spiking_wta` (the de-risk's subject) underperforms but full temporal integration is deemed
  premature — it is strictly a subset of #1.
- **On-substrate realization:** identical to #1's `sel_X` recurrence (add `internal_density`/`exc_weight_mean`
  to the existing layer, AMPA-fast), keep the existing inhibition, **no** `commit_X` burst pool. Lowest-effort
  delta over the shipped flag.

### #3 — Race-to-threshold / drift-diffusion accumulator (explicit bounded integrators, per action)

Four independent leaky integrators (one per `sel_X`), each accumulating its `thal_X` drive; first to cross a
fixed bound wins and is committed; a reset follows. The "two anti-correlated accumulators terminate at
first-bound-crossing" of G.16, generalized to four.

- **Biology:** drift-diffusion / bounded accumulation (catalog G.16); LIP accumulator (G.17, Mazurek-Roitman
  2003, 55 spikes/s bound); Lo-Wang bound. Race/LCA models (Usher-McClelland 2001) are the algorithmic form.
- **Why it handles the weak signal:** *integration over time is the textbook fix for weak/noisy evidence* —
  a small consistent drive crosses a bound given enough time, and the bound height is the explicit speed-
  accuracy control. **Caveat:** a pure independent race (no cross-inhibition) is less robust to correlated
  noise than the mutually-inhibiting attractor (#1); the attractor *is* the biologically-preferred way to
  build the integrator (Wang 2002 argues the recurrent network IS the integrator, with the bound emerging
  from the attractor rather than imposed). So #3 is essentially #1 *without* the mutual inhibition — keep it
  as the conceptual frame, but realize it as the inhibition-coupled attractor (#1), which is why #1 outranks
  it.
- **On-substrate realization:** `sel_X` with strong recurrent self-excitation (near-integrator gain) + a
  per-pool threshold/burst readout, *minus* the cross-inhibition. Most cleanly done as the #1 layer with the
  `sel_FS` cross-inhibition weakened — i.e. it lives on the same code path.

### #4 — Boost-the-released-signal-then-WTA (pre-amplify `thal_X`/feedforward gain, then instantaneous WTA)

Increase the feedforward gain so the weak released signal arrives strong at the WTA (e.g. raise
`thal_to_sel_weight`, or interpose an amplifier pool), then run the existing instantaneous soft-WTA.

- **Biology:** thalamo-cortical amplification (Douglas-Martin: thalamic input amplified by recurrent
  excitation) — but note the biology amplifies via *recurrence*, not via a bigger feedforward weight. A pure
  feedforward boost is the *engineering* shortcut version of the biological recurrent amplifier.
- **Why it partially works / why it is limited:** a higher feedforward gain does make a clean winner more
  separable, and it is trivial to try. But it amplifies noise as much as signal (no integration, no signal
  restoration), and it does nothing about *temporal* robustness over a multi-goal run — the exact axis on
  which motor-pool WTA failed (14.7). It is a cheap diagnostic ("is the problem just gain?") and a fallback,
  not a faithful mechanism. **It is already partially present** (the de-risk uses `thal_to_sel_weight=60`).
- **On-substrate realization:** bump `--thal-to-sel-weight` and/or add a relay-gain pool; no new structure.

### #5 — Thalamo-cortical reentrant amplification (A.05 closed loop as the commitment)

Use the existing cluster-A `thal_X → cortex_X` reentry (plus `cortex_X → stn` hyperdirect) so the released
thalamus re-excites its cortex, the loop reverberates, and the sustained reentrant winner IS the committed
decision (read from `cortex_X`).

- **Biology:** A.05 reentrant loops; Kandel 6e Ch 38 p 943–948; the reentry is the biological "hold" on a
  selected action. Composes with the existing `--enable-cluster-a-closed-loop`.
- **Why it could help the weak signal:** positive feedback through the loop amplifies and *sustains* the weak
  winner (latch), addressing the persistence problem the conversational side documented. **Why it is riskier
  here:** it is NOT a read-only tap — it feeds back into the forward cascade, so (like the TRN-on-relay
  failure, 20.0) a mistuned loop can corrupt the very signal it reads or runaway-excite. It also needs a
  separate threshold/commit step to convert "sustained cortical winner" into a discrete action. Best used as
  the *optional reentry latch* inside #1 (a weak, carefully-bounded loop), not as the primary selector.
- **On-substrate realization:** enable cluster-A reentry + add a `sel`/`commit` threshold readout on
  `cortex_X`; tune loop gain conservatively (contraction/soft-WTA regime).

### #6 — E-I two-pool-per-action WTA (paired excitatory + inhibitory pool per action, mutual inhibition)

A symmetric two-pool-per-action competitive circuit (each action: an E pool + an I pool; I pools cross-inhibit
other actions' E pools). This is the generic "balanced E-I WTA."

- **Biology:** generic cortical/striatal E-I competition; but the catalog's B.04 + Rutishauser both warn that
  *symmetric* mutual inhibition without structured (feedforward/selective) inhibition and without self-
  excitation is a weak, unstable selector. It is essentially what the failed TRN-WTA was (cross-inhibiting
  pools), and without recurrent self-excitation it inherits the weak-signal problem.
- **Why it is ranked last:** it lacks the two ingredients the literature says are decisive (recurrent self-
  excitation for amplification/integration; structured rather than symmetric inhibition), and the project has
  *already empirically failed* the symmetric-mutual-inhibition variant twice (TRN 20.0; and on the
  conversational side). Included for completeness; not recommended.
- **On-substrate realization:** per-action E+I `BrainRegion`s with cross-action inhibitory `RegionPathway`s —
  but this is strictly dominated by #1 (add self-excitation + structured inhibition + read-only tap).

### Mechanisms the texts point to that the candidate list was missing

- **(NEW, folded into #1) A dedicated downstream all-or-none BURST/threshold pool (SC / saccade-generator
  analogue), tonically inhibited, released by the accumulator** — H.24 (OPN→EBN), H.25 (SC), Lo-Wang 2006,
  Stine 2023. This is the *commit* half of the two-stage mechanism and was absent from the original six-item
  menu (all six were "selection" mechanisms; none was a "termination/threshold" mechanism). Stine 2023's SC-
  inactivation result is the direct evidence that selection-without-a-threshold-structure fails to commit —
  the project's instantaneous WTAs are precisely "selection without a termination stage." **This is why #1
  pairs the accumulator with a burst pool, not just a WTA.**
- **(NEW) Threshold set by cortico-striatal weight (Lo-Wang) as the biological speed-accuracy knob** — the
  bound is not a hand-tuned constant but a DA-modulated synaptic strength, giving the conversion a principled,
  already-present tuning locus (the existing R-STDP/DA on `cortex_X → str_d1_X`).
- **(framing) The catalog's own G.16 verdict** — that the BG cascade is *already* a functional bounded
  accumulator missing only the explicit integration/threshold — means the conversion is *completing an
  existing mechanism*, not bolting on a foreign one.

---

## RECOMMENDED mechanism + fallback order

**RECOMMENDED: #1 — thalamus-driven recurrent accumulator (Wang-2002 attractor) + downstream burst-threshold
commit (Lo-Wang/SC), realized as a read-only extension of the existing `spiking_wta` `sel_X` layer.** The
single concrete change that the whole literature converges on is *add recurrent self-excitation (ideally
NMDA-slow) inside each `sel_X` pool* — the current `internal_density=0.0` is the degenerate, un-amplified,
un-integrating case. This converts a passive instantaneous comparator (which provably cannot manufacture a
winner from a weak input) into a biological amplifying integrator that ramps the weak-but-clean released
thalamus to a committed threshold, with the existing per-pool inhibition providing the (structured, stable,
soft-WTA) competition and an optional `commit_X` burst pool / closed-loop latch providing the all-or-none
termination. It avoids both prior failure modes by construction (driven from the strong thalamus; read-only,
no relay corruption), and its mechanism is the only one in the menu specifically designed for weak evidence.

**Fallback order (each a strict reduction of #1, on the same code path):**
1. **#1** full two-stage (NMDA-slow recurrent accumulator + `commit_X` burst). ← primary.
2. **#2** Rutishauser soft-WTA (AMPA-fast recurrent self-excitation, no burst pool) — if temporal integration
   is deemed premature; smallest delta over the shipped flag.
3. **#3** explicit race-to-threshold (strong recurrence ≈ integrator, weak/no cross-inhibition) — conceptual
   frame; realize on #1's path.
4. **#4** feedforward gain boost — cheap diagnostic / fallback (already partially in place).
5. **#5** reentrant latch (A.05) — use only as the optional hold inside #1, not standalone (relay-corruption
   risk).
6. **#6** symmetric E-I two-pool WTA — not recommended (already failed empirically; lacks both decisive
   ingredients).

**The single highest-leverage experiment:** add NMDA-slow recurrent self-excitation to the `sel_X` pools
(soft-WTA gain, α<1) and gate the readout on a thresholded `commit_X` burst; benchmark against the thal host-
argmax (target 2.3) on cheat-5 multi-goal, multi-seed. This is purely additive (no `sim/` edit; reuses
`BrainRegion`/`RegionPathway`/existing per-region NMDA), and it completes the mechanism the catalog (G.16)
already says the cascade is 90% of the way to.

---

## Key citations (text + page where available)

**Catalog (`references/feature-catalog.md`, `catalog-build` branch):** A.04 (BG WTA at GPi/SNr, Kandel 6e Ch
38 p 939–942), A.05 (reentrant loops, p 943–948), A.07 (BG→SC/MLR, p 938–942), G.16 (drift-diffusion, Ch 56 p
1399–1404), G.17 (LIP accumulator, p 1402–1404), H.24 (saccade burst generator, Ch 35 p 868–880), H.25 (SC
saccade map / SNr disinhibition, p 875–882), B.04 (striatal WTA is feedforward-FSI, Wilson 2007 / TK-2017).

**Primary papers:**
- Wang, X-J. (2002). Probabilistic decision making by slow reverberation in cortical circuits. *Neuron*
  36:955–968. — recurrent NMDA attractor integrates weak evidence into a binary choice.
- Lo, C-C. & Wang, X-J. (2006). Cortico–basal ganglia circuit mechanism for a decision threshold in reaction
  time tasks. *Nat. Neurosci.* 9(7):956–963. — SC all-or-none burst signals threshold crossing; SNr
  disinhibition gates it; threshold set by cortico-striatal weight. **The most on-point paper.**
- Stine, G.M., Trautmann, E.M., Jeurissen, D., Shadlen, M.N. (2023). A neural mechanism for terminating
  decisions. *Neuron* (PMC10565788). — LIP accumulates (drift-diffusion), SC commits (all-or-none burst,
  rate+derivative threshold); SC inactivation removes commitment but not accumulation.
- Mazurek, M.E., Roitman, J.D., Ditterich, J., Shadlen, M.N. (2003). A role for neural integrators in
  perceptual decision making. *Cereb. Cortex* 13:1257–1269. — LIP ramp-to-threshold (~55 spikes/s) accumulator.
- Rutishauser, U., Douglas, R.J., Slotine, J-J. (2011). Collective stability of networks of winner-take-all
  circuits. *Neural Computation* 23:735–773 (+ companion "Competition through selective inhibitory
  synchrony", arXiv:1201.2845). — stable soft-WTA needs recurrent self-excitation (gain α<1) + structured
  inhibition; contraction-theory stability; symmetric naive WTA is unstable.
- Douglas, R.J. & Martin, K.A.C. — canonical cortical microcircuit: weak thalamic input amplified by recurrent
  excitation, restrained by inhibition (selective amplification / signal restoration).
- Wang, X-J. (2008). Decision making in recurrent neuronal circuits. *Neuron* 60:215–234 (review, PMC2710297)
  — the τ_network = τ_syn/|1−w_rec| integration formula; ties the attractor accumulator to the BG/SC threshold.

**Sources (URLs):**
- Lo & Wang 2006: https://www.nature.com/articles/nn1722 · https://pubmed.ncbi.nlm.nih.gov/16767089/
- Stine et al. 2023: https://pmc.ncbi.nlm.nih.gov/articles/PMC10565788/
- Wang 2002: https://www.cns.nyu.edu/wanglab/publications/pdf/wang2002_decision.pdf
- Wang 2008 review: https://pmc.ncbi.nlm.nih.gov/articles/PMC2710297/
- Mazurek et al. 2003: https://academic.oup.com/cercor/article/13/11/1257/274091
- Rutishauser-Douglas-Slotine 2011: https://direct.mit.edu/neco/article-abstract/23/3/735/7647/ · http://web.mit.edu/nsl/www/preprints/competition_2010.pdf
- Rutishauser companion (selective inhibitory synchrony): https://arxiv.org/pdf/1201.2845
