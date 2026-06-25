# LEARNED TALKATIVENESS — the speak-policy LEARNED from interaction feedback — SCOPING (read-only, 2026-06-24)

**Role:** read-only deep-research + scoping subagent. NO edits, NO runs, NO webapp. No `tail`/`grep -f`/`while-sleep` watchers.
**Owner directive (verbatim intent):** the speak-threshold (the choose-to-speak appraisal's emit decision) should be a **LEARNED,
CONTEXT-DEPENDENT value-policy shaped by conversational feedback** — when the brain gives a short reply and the owner asks it to
ELABORATE, that feedback should TEACH it to speak more in SIMILAR situations over time. The memory
`project_communicable_brain_not_rag` REFINEMENT (2026-06-24): *"the talkativeness / speak-threshold should be LEARNED from
interaction, NOT a hardcoded knob I set… reward-modulated learning of a context-dependent speak-VALUE-POLICY… the ask-to-elaborate
is a perceived conversational reward/teaching signal -> the DA/limbic reward system raises the 'worth-saying' value for that context
-> it speaks more there next time… this dissolves the 'values call' (the brain learns its own talkativeness from the owner) and ties
into the artificial-life develop loop."*
**Research-gate trigger (d) NEW-MECHANISM-CLASS + (c) BLOCKS-A-GOAL:** the learned-from-interaction policy is the *goal* of the
communicable-brain frontier; the fixed-threshold appraisal de-risk (in flight, `_value_salience_appraisal_*`) is only the stepping
stone (prove the appraisal mechanism). **Hard constraints honored throughout:** BIOLOGICAL (neurons/synapses/their communication);
SINGLE BRAIN, SINGLE SUBSTRATE; the brain does the cognition; the reward is **BRAIN-COMPUTED** (not a host `+1`); the moat **RELAXES**
(speak-while-flagging) — NOT removed.

> **HEADLINE (load-bearing for the controller).** This scoping sits one layer ABOVE the choose-to-speak appraisal de-risk. The
> appraisal makes the brain decide *whether/which* to speak via a value-grounded threshold; THIS makes that threshold (the
> appraisal-pool gain / the speak-value) **LEARNED from feedback** rather than set by the owner. **The decisive finding from the
> catalog + project review: this is NOT a new mechanism — it is a near-pure COMPOSITION of three already-GO project primitives that
> have never been wired together for this purpose.** (1) The **brain-computed reward** = a PERCEIVED "elaborate"/follow-up token
> processed as a **conditioned social reinforcer** → the already-shipped spiking SNc reward-burst (`co_resident_nav_critic`'s
> `reward_us` US-afferent + `td_snc` → the signed `dopamine` broadcast). (2) The **context-conditioning** = the speak-value is held
> on a **context→appraisal synapse** updated by **three-factor reward-modulated plasticity** (the shipped
> `cp_eligibility_trace × (reward − baseline) × STDP` path, bridge.py:7075-7190), with the **`from_action_specific_reward` rule** as
> the exact precedent for "credit only the context that was active." (3) The **policy = the appraisal threshold itself** (the
> in-flight de-risk), now read from the LEARNED context→appraisal weight instead of a fixed gain. The genuine new *science* is
> only the **wiring** (perceived-feedback → reward → context-gated speak-value plasticity → appraisal drive) + the **demonstration**
> that repeated "elaborate" RAISES the speak-rate IN THOSE CONTEXTS while leaving others unchanged, with the reward brain-computed
> and the moat preserved. The catalog confirms the value/RPE/eligibility/three-factor half is **implemented/partial-but-reusable**;
> the **conditioned-social-reinforcer framing of a conversational cue** and the **context-conditioned speak-value** are the
> un-catalogued composition.

---

## 1. DIAGNOSIS — decompose the three sub-problems; what we HAVE vs the genuine new part

The owner's request decomposes into three cooperating sub-mechanisms (a)/(b)/(c). I treat each separately because each maps to a
DIFFERENT reusable primitive, and the "genuine new part" is the wiring between them, not any one piece.

### 1.1 (a) How a perceived "elaborate" becomes a BRAIN-COMPUTED reward (not a host counter)

**Biology.** An "elaborate / tell me more / a follow-up question" is a **social/communicative cue**. Two converging biological
facts make it a legitimate *brain-computed* teaching signal:
- **Conditioned (secondary) reinforcement** (catalog C.22/C.28/C.31; Schultz 1998; cue-evoked DA literature): a stimulus that
  reliably PREDICTS reward acquires the ability to drive a phasic DA burst itself (the TD cue-shift). A conversational cue the brain
  has learned to associate with "I did the right thing" therefore drives DA via the SAME midbrain machinery as a primary reward —
  the burst is COMPUTED by the brain's reward circuit, not asserted by Python. The "elaborate" need not even be pre-defined as
  rewarding: it can be a *primary social reinforcer* (the rewarding nature of positive social interaction is mediated by NAc /
  striatum / thalamic reward nuclei — Krach et al. 2010 *Front. Behav. Neurosci.*; the social-media-dopamine literature confirms
  positive conversational feedback activates NAc/putamen reward synapses).
- **The boundary that keeps it brain-based.** Per `feedback_brain_based_only_standard`: the host code is legitimate ONLY for **the
  environment** (the owner's reply text is part of the world the brain senses) and **the body** (rendering/emitting). So the host
  legitimately (i) DELIVERS the owner's reply token to the brain's sensory input, exactly as it delivers any heard sentence (the
  existing parser front-end already does this), and (ii) at most fires a brief **US-afferent drive** the way `reward_us` is driven by
  the world in nav (a "reward delivered" sensory event = the body/world signaling). The brain then COMPUTES the RPE: the
  `reward_us` afferent → spiking SNc burst → the signed `dopamine` concentration → the per-synapse three-factor update. **A host
  `if "elaborate" in reply: reward = +1` written directly into the plasticity rule is the shortcut; a host "the owner said
  elaborate → fire the reward-US sensory afferent, let the SNc compute the burst" is the legitimate environment/body boundary** (it
  is the analogue of "the agent reached the goal → fire the reward-US"). The lesion anti-cheat (§4) is what enforces this: lesion the
  SNc/DA → the host afferent fires but NO learning happens, proving the brain computes the teaching signal.

**What the project HAS (reusable, all GO):**
| Piece | Where | Reuse |
|---|---|---|
| **Spiking SNc reward-burst + `reward_us` US-afferent + signed `dopamine` broadcast** (`co_resident_nav_critic`) | `nav_conv_merged_bridge.py:553/621-712`; `from_region_firing_signed` over `[td_snc]` (`neuromodulators.py:774`) | the BRAIN-COMPUTED reward path: drive `reward_us` from a perceived-feedback sensory event → SNc fires → DA. Validated Schultz battery 6/6 (`2026-06-18-limbic-core-rpe-battery-GO.md`). |
| **`from_reward` production rule** (phasic DA from `current_reward_signal − baseline`) | `neuromodulators.py:95/647` | the simplest DA-from-reward path if a full spiking SNc isn't co-resident (the CPU de-risk stand-in). |
| **TD cue-shift critic** (a cue ACQUIRES the DA via learned V) | `co_resident_td_cueshift`; `sim/td_value_critic.py` (oracle); `2026-06-22-shortcut5b-td-read-derisk.md` | the conditioned-reinforcer dynamic if we want the "elaborate" cue itself to become predictive over time (deeper; not needed for the cheap de-risk). |

**Genuine new part for (a):** **framing a conversational cue as the reward-US** (the perception→reward wiring). Small — it reuses the
exact `reward_us` afferent pattern; the only new code is "a perceived feedback token fires the reward-US sensory drive."

### 1.2 (b) The context-conditioning — speak-value learned PER-SITUATION, not a global gain

**Biology.** Context-dependent value/policy learning is a core BG/PFC-thalamus function (catalog O.22 striatal action-value:
*subgroups of MSNs fire for the value of one specific action regardless of choice*; PMC10348919 *computational and neural bases of
context-dependent learning*; thalamo-cortical context gating reconfigures cortical representations — ScienceDirect S0166223625001924).
The "context" for talkativeness is the **conversational situation** — e.g. the topic neighborhood, whether the owner is in a
follow-up exchange, the question type. The speak-value `Q(context, speak)` must be coded **per-context** so that learning to elaborate
about TOPIC-A raises speaking about A-similar topics WITHOUT raising it everywhere (the owner's exact requirement: *"speak more in
SIMILAR situations… leaving others unchanged"*). The "similar" generalization is exactly what the PPMI-cortex code-similarity buys
(a learned-from-conversation context representation where related topics have overlapping codes — CYCLE 88-96 stream cortex).

**What the project HAS (reusable, all GO):**
| Piece | Where | Reuse |
|---|---|---|
| **`from_action_specific_reward` rule** — reward credited ONLY when `last_selected_action == source_action` | `neuromodulators.py:148-164` | the EXACT precedent for "credit only the active context": replace "action" with "context-id." A per-context DA channel, or (cheaper) a single DA × per-context eligibility. |
| **Three-factor reward-modulated plasticity** — `Δw = lr × (reward − baseline) × eligibility_trace`, gated per-pathway | `bridge.py:7075-7190` (`cp_eligibility_trace`, `fused_eligibility_trace_decay`); catalog C.29 (implemented), C.30 actor (implemented) | THE learning rule that adjusts a **context→appraisal synapse**: the context assembly that was active leaves an eligibility trace; the feedback-DA converts it to a weight change on JUST that synapse → the speak-value rises for that context only. |
| **PPMI stream-cortex context codes** (similar topics → overlapping codes) | CYCLE 88-96 (`2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`); the develop-loop cortex | the "similar situations" generalization is FREE: a weight learned for context-A's code partially activates for A-similar codes (graded population overlap), so elaborating about A raises speaking about related topics, not unrelated ones. |
| **Per-pathway plasticity / transmission gates** | `regions.py:281-316`; `set_plasticity_gate`/`set_transmission_gate` | freeze everything EXCEPT the context→appraisal synapse during the talkativeness-learning probe (isolate the learned signal; anti-cheat). |

**Genuine new part for (b):** a **context→speak-value (context→appraisal) synapse that the three-factor rule updates** — i.e. the
speak-value is a *learned weight indexed by the conversational context code*, not a scalar gain. The machinery (eligibility,
context-specific credit, similarity codes) all exist; the new part is *pointing them at the appraisal pool*.

### 1.3 (c) The reward-modulated plasticity that ADJUSTS the speak-policy

**Biology.** Actor-critic / policy improvement (catalog C.30, O.20): the DA-RPE updates the actor's action preferences
`H(s,a) ← H(s,a) + αδ` (S&B p. 259). Here the "action" is *speak-vs-stay-silent* and the "state" is the conversational context, so
the policy `π(speak | context)` is improved by the feedback-DA exactly as the BG cascade improves a motor policy. **Tonic-DA vigor**
(Niv 2007 — tonic DA = average-reward opportunity cost → response RATE/vigor) gives the *global* talkativeness set-point: a brain
that has been rewarded for speaking a lot runs a higher tonic DA → speaks more readily everywhere. So the learned policy has TWO
knobs the feedback shapes: a **per-context speak-value** (phasic, the §1.2 synapse) and a **global willingness/vigor** (tonic DA
set-point) — both already representable in the project (phasic via three-factor on the synapse; tonic via a slow `from_reward`/EMA
modulator concentration that scales the appraisal-pool excitability, the catalog-O.21 average-reward `R̄`).

**What the project HAS:** the three-factor rule (1.2), the neuromodulator concentration dynamics (tonic set-point via slow decay +
`from_reward`), and the **appraisal pool itself** (the in-flight de-risk's speak/silence sel→commit accumulator, whose drift is the
speak-value).

**Genuine new part for (c):** nothing mechanically new — it is the three-factor rule (phasic, per-context) + a slow DA EMA (tonic,
global vigor) both feeding the appraisal threshold. The science is the *demonstration* that this closes the loop:
feedback → DA → speak-value↑ → speaks more there.

### 1.4 Summary of the diagnosis

| Sub-problem | Genuine new part | Reused (GO) primitive |
|---|---|---|
| (a) perceived "elaborate" → brain-computed reward | conversational cue framed as the **reward-US sensory afferent** (perception→reward wiring) | spiking SNc + `reward_us` + signed `dopamine` (`co_resident_nav_critic`); `from_reward` (CPU stand-in) |
| (b) context-conditioning (per-situation) | a **context→appraisal synapse** the three-factor rule updates; "similar" via PPMI overlap | `from_action_specific_reward` precedent; three-factor plasticity (bridge.py:7075); PPMI codes; per-pathway gates |
| (c) reward-modulated speak-policy update | (none mechanically) — compose phasic three-factor + tonic-DA vigor into the appraisal threshold | three-factor rule (C.30 actor); tonic DA set-point (Niv 2007 / O.21 R̄); the appraisal accumulator (in-flight de-risk) |

**⇒ Verdict-in-one-line:** **a COMPOSITION of proven pieces (not a new mechanism class)** — the only genuinely new science is the
*wiring* (perceived-feedback → SNc reward → context-gated speak-value plasticity → appraisal drive) and the *demonstration* of
context-specific talkativeness learning. The catalog's reward/RL cluster (C.28-C.34, O.19-O.23) is implemented/partial-but-reusable;
the conditioned-social-reinforcer framing of a conversational cue and the context-conditioned speak-value are un-catalogued but are
straightforward instances of catalogued primitives.

---

## 2. RANKED biology-grounded options (cheap-first)

> Every option is single-substrate, the brain computes the reward (the host only delivers the perceived-feedback sensory event and
> fires the reward-US afferent — the environment/body boundary), and the moat RELAXES-NOT-REMOVED (the speak channel only changes
> *whether/which* to volunteer a FLAGGED view; the known-fact channel stays hard-gated). Ordered cheapest/most-reusing → deepest.
> **All depend on the choose-to-speak appraisal existing** (the in-flight `_value_salience_appraisal_*` de-risk supplies the speak-VALUE
> + the threshold this scoping makes LEARNED).

### Option A [CHEAPEST — the recommended de-risk; "feedback-modulated context→speak-value synapse" on the existing three-factor rule]
**Mechanism.** Add ONE plastic **context→appraisal synapse** (a weight indexed by the conversational-context code, read into the
appraisal pool's drive). After the brain emits (or stays silent), if the owner's reply is a perceived **"elaborate"** (delivered as a
sensory token → fires a `reward_us`-style afferent → the SNc burst → a phasic positive `dopamine`), the **three-factor rule**
(`Δw = lr × (DA − baseline) × eligibility`) updates JUST the context→appraisal synapse that was active (its eligibility trace is the
only one elevated) → the speak-value for that context RISES → next time a SIMILAR context appears (PPMI code overlap), the appraisal
pool's drive is higher → it crosses the speak threshold more readily → it speaks/elaborates more THERE. Conversely an "ok, stop /
that's enough" delivers a negative-RPE (sub-tonic SNc → DA dips below baseline via `from_region_firing_signed`) → the speak-value for
that context falls. The **global talkativeness** rides a slow DA EMA (tonic vigor, Niv 2007) scaling the appraisal-pool excitability.
**Why cheapest:** reuses the GO three-factor plasticity (bridge.py:7075-7190), the GO `from_action_specific_reward`/eligibility
context-credit pattern, the GO spiking SNc reward path (or the `from_reward` CPU stand-in), the GO PPMI context codes, and the
in-flight appraisal pool. The ONLY new pieces are (i) the context→appraisal plastic synapse and (ii) the "perceived-feedback fires
the reward-US" wiring. **NO `sim/` edit expected** (the three-factor rule, eligibility, neuromodulator subsystem, and per-pathway
gates are all shipped; the feedback-US is `cp_external_input_current` on the reward-US afferent; the speak-value is an ordinary
plastic pathway).
**Biology:** conditioned social reinforcer → DA RPE (Schultz 1998; C.28/C.31); actor policy improvement `H(s,a)+=αδ` (C.30); striatal
context/action-value (O.22); tonic-DA vigor set-point (Niv 2007); generalization via overlapping learned codes (PPMI cortex).
**Risk:** the context code must be a clean, similarity-structured handle (PPMI gives this); the credit must be context-specific
(eligibility + the `from_action_specific_reward` gating ensures it). Both mitigated by reuse.

### Option B [per-context DA channel — the compartmentalized-DA / `from_action_specific_reward` pattern made explicit]
**Mechanism.** Instead of a single DA × per-context eligibility, declare a **per-context (or per-context-cluster) DA modulator**
(`from_action_specific_reward` with `source_action` = the context-id), so the feedback-DA is delivered ONLY to the active context's
speak-value. Mechanically equivalent to Option A's eligibility-gated credit but makes the context-specificity structural rather than
trace-timing-dependent.
**Why next:** also reuses-by-import (the rule exists, `neuromodulators.py:148`); slightly more wiring (N context channels). Best when
the context set is small + discrete (e.g. a handful of topic clusters); Option A's single-DA + eligibility scales better to a
continuous PPMI context space. Good as a **cross-check** that the context-specificity is genuine (do A and B agree on which contexts
get more talkative?).
**Biology:** compartmentalized DA (catalog C, Cluster-C-v2 per-action DA); striatal action-value coded by separate neurons (O.22).
**Risk:** the project's own Cluster-C-v2 found per-action DA channels NEGATIVE for nav (desync across phases) — but that was for
fast-switching motor selection; here the contexts are slow + the credit is unambiguous (one context active per turn), so the failure
mode likely doesn't bite. Flag: validate it doesn't add variance vs Option A.

### Option C [tonic-DA vigor as an explicit learned set-point — the global "talkativeness" knob the owner stops setting]
**Mechanism.** A slow **DA EMA modulator** (catalog O.21 average-reward `R̄`; `from_reward` with a long `decay_tau_ms`) whose
concentration tracks the long-run rate of positive conversational feedback, scaling the appraisal-pool **excitability** (more
positive feedback over a session → higher tonic DA → speaks more readily everywhere; a session of "stop/too much" → lower tonic →
quieter). This is the GLOBAL half of the policy (Niv 2007), complementary to A/B's per-context phasic half.
**Why deeper-but-cheap:** mechanically trivial (a slow modulator + an `excitability_drive` target on the appraisal pool — both
shipped in the NM subsystem), but it's the piece that most directly "dissolves the values call" — the owner no longer sets a
talkativeness gain; the brain's tonic DA *becomes* it, shaped by the running feedback. Recommend folding it in AFTER Option A proves
the per-context phasic learning (so the two knobs are validated separately).
**Biology:** Niv 2007 tonic-DA vigor / opportunity cost; O.21 R-learning average reward.
**Risk:** tonic and phasic must not double-count the same feedback; mitigated by the standard EMA-vs-phasic separation (C.32: Component
1 raw, Component 2 EMA) — give the tonic set-point a much slower tau than the phasic burst.

### Option D [DEEP / DEFER — the "elaborate" cue itself becomes a learned conditioned reinforcer via the TD critic]
**Mechanism.** Rather than the host always firing the reward-US on "elaborate," let a **TD critic** (catalog C.28/C.30;
`co_resident_td_cueshift`) LEARN that the "elaborate"-context PREDICTS reward, so over trials the DA burst shifts onto the cue itself
(the cue-shift signature). This makes the conditioned-reinforcer status of conversational feedback genuinely *learned* rather than
*given*. **DEFER** — the on-bridge TD value-read is contaminated by place-code magnitude (`2026-06-22-shortcut5b-td-read-derisk.md`),
and it's not needed for the cheap demonstration (a primary-social-reinforcer framing of "elaborate" via `reward_us` is sufficient and
honest). Flag for a later purity arc once the TD-read contamination is resolved.

---

## 3. RECOMMENDED cheap-first de-risk (smallest falsification)

**Probe: "Does a repeated 'elaborate' feedback in SIMILAR contexts RAISE the brain's speak-rate IN THOSE CONTEXTS while leaving
others unchanged — via reward-modulated plasticity of a context→speak-value synapse, with the reward BRAIN-COMPUTED, the moat
preserved, and the learning lesion-confirmed-as-the-BRAIN's-value-system?"** (Option A; CPU-first on the Probe-1 standalone brain,
reusing `_communicable_brain_probe1_whatdoyouthink.py` + the in-flight appraisal de-risk verbatim and extending it.)

- **Build (reuse-by-import, NO `sim/` edit):** the Probe-1 brain (PPMI cortex + `RFPhasorComposer` store + `GenerativeReplayProposer`
  + `BrainConversationalAgent` parse + familiarity gate) + the in-flight choose-to-speak appraisal (the speak/silence accumulator
  whose drift = {value, plausibility, familiarity}). Add ONE LEARNED element: a **context→speak-value weight** (indexed by the topic's
  PPMI context code) read into the appraisal's value-drive. **Training (the "develop loop in miniature"):** present a set of topics
  split into a TAUGHT subset (the owner "asks to elaborate" — fires the reward-US → DA-positive) and an UNTAUGHT subset (no feedback);
  after each emit, run the three-factor update on the active context→speak-value synapse. The **value/reward axis** on the standalone
  brain is the spiking-SNc stand-in (the CPU `from_reward` DA proxy or a transparent DA scalar; the GPU follow-on uses the REAL shared
  `dopamine` from `co_resident_nav_critic`). The DA proxy MUST be a DISTINCT signal from the plausibility axis (else the test is
  circular — see anti-cheat 4).
- **GO bars (multi-seed ≥3; controller runs 6-seed if GO):**
  1. **TALKATIVENESS RISES WHERE TAUGHT (the core claim).** After N feedback rounds, the speak-rate on the TAUGHT-context cluster is
     SIGNIFICANTLY higher than its pre-training rate AND higher than the UNTAUGHT cluster's (which is ~unchanged). Quantify: Δspeak-rate
     (taught) ≫ Δspeak-rate (untaught), with the taught/untaught gap growing monotonically with feedback rounds (the *learning curve* —
     the analogue of the Schultz acquisition curve). Generalization check: A-SIMILAR untaught topics (high PPMI overlap with a taught
     topic) rise PARTIALLY; A-DISSIMILAR topics do not (the owner's "similar situations" requirement, measured by code-similarity bins).
  2. **CONTEXT-SPECIFIC, NOT GLOBAL.** A control where the SAME total amount of feedback-DA is delivered but DECORRELATED from context
     (shuffled context-credit) raises the GLOBAL rate flatly with NO taught-vs-untaught gap → confirms the gap is *context-conditioned*
     learning, not a global vigor shift. (This separates Option-A's phasic per-context learning from Option-C's tonic global knob.)
  3. **MOAT RELAXED-NOT-REMOVED (HARD).** 0 known-fact-channel leaks throughout (a who/what query on every emitted un-stored
     proposition still ABSTAINS; every emission flagged as a hypothesis). Higher talkativeness must NOT mean confabulated facts — it
     means MORE flagged hypotheses on supported topics. The known-fact channel is byte-unchanged.
- **The decisive anti-cheat (LESION the value system):** **pin the DA / value axis to baseline (lesion the SNc / the feedback-reward
  path) → the feedback produces NO change in speak-rate** (taught == untaught == pre-training). This proves the talkativeness change
  is driven by the BRAIN's reward-modulated plasticity, not a host counter incrementing a threshold. (Mirrors the Probe-1 lesion 46/46
  + the familiarity-gate `lesion()` + the limbic-core SNc-lesion precedent.) **This is the load-bearing anti-cheat** — it is what makes
  the learning the brain's, not a Python `if elaborate_count > k: threshold -= ε` (the `feedback_brain_based_only_standard` bar).
- **Cost:** CPU, minutes (the standalone-brain pre-pass). The GPU follow-on wires the SAME context→speak-value synapse onto the merged
  "one brain" so the feedback-DA is the REAL spiking SNc (`co_resident_nav_critic`'s `reward_us` → `td_snc` → `dopamine`), the credit
  is the real eligibility trace, and the lesion is the real SNc-pinning; re-runs the bars + the live-nav co-residence gate (byte-identity
  of the conversational reads at REST, nav score Δ=0).

**If NEGATIVE** (the feedback doesn't raise taught talkativeness / the gap isn't context-specific / a control fails / the moat breaks),
report it PRECISELY — that localizes whether the gap is the **reward SIGNAL** (the on-substrate DA-read contamination,
`2026-06-22-shortcut5b`), the **context handle** (the PPMI code isn't a clean enough credit target), or the **credit assignment** (the
eligibility doesn't isolate the active context). An honest negative IS the deliverable under BRAIN-BASED-ONLY — it maps what the
substrate can/can't learn about its own communicative style.

---

## 4. Anti-cheats (every probe inherits these)

1. **The learning is the BRAIN's reward system, not a host counter — LESION the DA/reward.** Pin DA / the feedback-reward path to
   baseline (lesion the SNc) → the feedback produces NO talkativeness change. **Load-bearing** — it is what distinguishes "the brain
   learned its speak-policy via reward-modulated plasticity" from "a Python counter lowered a threshold."
2. **The "elaborate"→reward is PERCEIVED, not hardcoded into the rule.** The feedback enters as a SENSORY event (a heard token
   delivered to the brain's input, the environment boundary) that fires a `reward_us`-style afferent; the brain COMPUTES the RPE (SNc
   burst → DA). No `reward = +1` is written into the three-factor update directly. (Provenance check on the GPU version: no
   `cp_..._reward_signal = f("elaborate" in reply)` host write into the plasticity rule; the reward enters via the SNc-driven `dopamine`
   concentration only — the legitimate neuromodulatory-broadcast boundary.)
3. **CONTEXT-SPECIFICITY (learns per-context, not globally).** The decorrelated-context-credit control (anti-cheat for GO-bar 2) must
   collapse the taught-vs-untaught gap → the learned change is conditioned on the active context, not a flat global gain. Plus the
   similarity-generalization gradient (taught > similar-untaught > dissimilar-untaught) must hold.
4. **NON-CIRCULAR value axis.** The DA/reward axis (what the feedback raises) MUST be a DISTINCT quantity from the plausibility axis
   (the content support) — else "feedback raises the speak-value" is tautological with "plausible topics get spoken." Use the
   feedback-DA as a separate, learned reward signal; verify the taught/untaught gap appears even when plausibility is held constant
   across the two clusters (taught and untaught topics matched for graph support).
5. **MOAT RELAXED-NOT-REMOVED (HARD).** 0 known-fact-channel leaks at every step; every emission flagged; the speak channel is additive
   (it changes *whether/which* to volunteer a flagged view); the `what_does`/`is_it_true` known-fact channel stays hard-gated +
   byte-identical. (Per `feedback_moat_not_hard_lossy_memory_ok`: kept-where-free; here it is free.)
6. **GROUNDEDNESS preserved.** The shuffled-PPMI-graph control (Probe-1) must still collapse the emission set's worth ≥3× — higher
   talkativeness must not mean speaking on ungrounded noise; it means more flagged hypotheses on GENUINELY supported topics.
7. **SINGLE-SUBSTRATE / BYTE-IDENTITY (GPU).** The PPMI cortex + RF composer + spiking SNc/`striosome_value` critic + the appraisal
   accumulator + the context→speak-value synapse are the brain; the host does only the perceived-feedback sensory delivery + emission
   bookkeeping. Adding the learned synapse must not regress the conversational reads (byte-identical at REST / DA==baseline) or the nav
   score (Δ=0; the composer's complex synapses are array-disjoint from `cp_connections`).

---

## 5. VERDICT + how it composes (appraisal + develop loop) + owner-steer flags

**Verdict: COMPOSE-PROVEN-PIECES (not a new mechanism class, not a deep frontier).** This is the cleanest kind of work the
research-gate flags as "proceed-directly-once-scoped": the reward half (DA-RPE, eligibility, three-factor, actor policy improvement,
tonic-DA vigor) is **implemented/partial-but-reusable** in the project AND well-grounded in the catalog (C.28-C.34, O.19-O.23); the
context-conditioning has an EXACT precedent (`from_action_specific_reward`); the policy IS the in-flight appraisal threshold. The
genuine new *science* is narrow: (1) framing a perceived conversational cue as a **conditioned/primary social reinforcer** that fires
the reward-US (the perception→reward wiring), and (2) a **context→speak-value synapse** the three-factor rule updates per-context, with
PPMI overlap giving the "similar situations" generalization. The smallest falsification is CPU, reuse-by-import, NO `sim/` edit, with a
sharp lesion anti-cheat (pin the DA → no learning) that makes it the BRAIN's reward system. **It is also the piece that DISSOLVES the
"values call"** flagged in the appraisal scoping (owner-steer #1 there): the owner stops setting a talkativeness gain; the brain learns
its own from the owner's feedback.

**How it composes with the choose-to-speak APPRAISAL (it sits directly on top of it):** the appraisal de-risk (in flight) supplies the
**speak-VALUE + the threshold** (the speak/silence sel→commit accumulator whose drift = {value, plausibility, familiarity}). THIS
scoping makes the **value/threshold LEARNED** — specifically, the appraisal's *value-drive* is read from the **context→speak-value
synapse** that the feedback-DA updates. So: **the appraisal decides whether/which to speak given a value; this makes that value a
context-conditioned, feedback-shaped LEARNED quantity.** Cleanest sequencing: (i) land the in-flight fixed-threshold appraisal de-risk
(prove the speak-decision mechanism); (ii) the CPU learned-talkativeness de-risk on the standalone brain (this doc's §3); (iii) wire it
onto the merged one-brain (the real spiking SNc feedback-DA + the real eligibility), validating the lesion anti-cheat against the real
SNc.

**How it composes with the DEVELOP LOOP (it IS the conversational-style development):** the artificial-life develop loop
(`_longitudinal_develop_loop_gpu.py`) already runs day-by-day learning with reward/consolidation hooks; the brain DEVELOPS its
vocabulary/facts over a simulated week. Learned talkativeness is the natural EXTENSION — over the develop loop's days, the brain
develops its **conversational STYLE** (how talkative, in which contexts) from the accumulated feedback, exactly as it develops vocab.
The context→speak-value synapse persists + consolidates with the rest of the brain state (lineage save/load), so the talkativeness
learned on day-1 carries to day-7 (and the self-replay consolidation should preserve it — a develop-loop GO bar: zero forgetting of
the learned speak-policy). This is the owner's *"the brain develops its conversational style via interaction"* realized.

**Owner-steer flags (genuine forks):**
1. **CONTEXT GRANULARITY (the main design fork).** What is "the context" the speak-value conditions on? Cheapest = the topic's PPMI
   code (continuous, similarity-structured, generalizes for free — recommend this). Alternatives = the question-type (who/what/opinion),
   or the discourse-state (in-a-follow-up vs new-topic), or a coarse topic-cluster id (discrete → favors Option B's per-context DA
   channel). Recommend PPMI-code context first (it directly delivers "similar situations"), with question-type as a second factor if
   the owner wants "elaborate more on opinion questions specifically."
2. **PHASIC vs TONIC (which knob to demonstrate first).** Option A (phasic, per-context speak-value) is the core claim and the
   recommended cheap de-risk; Option C (tonic-DA global vigor set-point) is the "dissolve the values call globally" knob. Recommend A
   first (it carries the context-specificity), then fold in C (so the owner-set talkativeness gain is fully replaced by a learned
   tonic set-point). The owner may want C demonstrated too (it's the most direct "the brain learns HOW talkative to be overall").
3. **THE VALUES CALL — now LEARNED, but the owner still sets the REWARD POLICY (the meta-level).** The owner no longer sets a
   talkativeness gain, but DOES implicitly set what counts as positive feedback ("elaborate" = reward, "stop" = negative). That is the
   correct, honest residual of owner control — it is *teaching*, exactly the develop-loop framing. Flag for the owner: is "elaborate"
   the only positive signal, or also other engagement cues (a follow-up question, a long owner reply)? Recommend a small, transparent,
   PERCEIVED set (elaborate/tell-me-more = +; stop/too-much = −; the rest = neutral) for the de-risk, expandable later.
4. **STANDALONE value stand-in vs real SNc (CPU vs GPU).** The CPU de-risk needs a DA proxy (a distinct, learned reward signal, NOT the
   plausibility axis — anti-cheat 4); the GPU follow-on uses the REAL shared `dopamine` from `co_resident_nav_critic` so the lesion is
   genuine (pin the real SNc). Recommend a transparent CPU stand-in for the smoke + the real spiking SNc for the GPU GO bar.
5. **DEEP/DEFER — Option D (the "elaborate" cue becomes a LEARNED conditioned reinforcer via the TD critic).** The only deeper
   follow-on; deferred behind the cheap GO (a primary-social-reinforcer framing is sufficient + honest) and behind resolving the
   on-bridge TD-read contamination (`2026-06-22-shortcut5b`). Owner call on priority vs. the develop-loop horizon.

---

### Key references

- **Owner intent / dependency:** `project_communicable_brain_not_rag` memory (the 2026-06-24 REFINEMENT, verbatim); the in-flight
  choose-to-speak appraisal de-risk `research/findings/raw/_value_salience_appraisal_scoping.md` (+ `_value_salience_appraisal_derisk.json`)
  — this scoping makes that appraisal's threshold/value LEARNED.
- **Probe-1 harness (reuse for the de-risk):** `research/runners/_communicable_brain_probe1_whatdoyouthink.py` (`WhatDoYouThinkTurn`,
  `plausibility_score`, `hedge_for`, the lesion + shuffled-graph controls); `research/findings/2026-06-24-communicable-brain-probe1-GO.md`.
- **Brain-computed reward (project):** `research/runners/nav_conv_merged_bridge.py` (`co_resident_nav_critic` `:553/621-712`: spiking
  SNc `reward_us` US-afferent + `striosome_value` MSN-D1 critic + `td_snc`→`dopamine`); `sim/neuromodulators.py`
  (`from_region_firing_signed` `:774` = signed SNc→DA; `from_reward` `:95/647`; `from_action_specific_reward` `:148-164` = the
  per-context credit precedent; `pause_on_reward`); `research/findings/2026-06-18-limbic-core-rpe-battery-GO.md` (Schultz 6/6).
- **Reward-modulated plasticity (project):** `sim/bridge.py:7075-7190` (three-factor `Δw = lr × (reward − baseline) × eligibility`,
  `cp_eligibility_trace`, `fused_eligibility_trace_decay`); `sim/td_value_critic.py` (TD(λ) oracle); `research/runners/td_critic_gate.py`;
  `research/findings/2026-06-22-shortcut5b-td-read-derisk.md` (on-bridge value-read contamination — the deferred Option-D caveat).
- **Context handle (project):** PPMI stream-cortex `research/findings/2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`
  (similar topics → overlapping codes = the "similar situations" generalization); per-pathway gates `sim/regions.py:281-316`.
- **Develop loop (composes):** `research/runners/_longitudinal_develop_loop_gpu.py` (`develop_gpu`, `consolidate`, per-day lineage
  save/load); the develop-loop GO `2026-06-23-longitudinal-develop-loop-GPU-GO.md` + the week-1 console capstone
  `2026-06-24-week1-develop-loop-console-capstone.md`.
- **Catalog (`sim-catalog/references/feature-catalog.md`):** C.28 (TD error = phasic DA, `partial`), C.29 (eligibility traces,
  `implemented`), C.30 (actor-critic / policy improvement `H(s,a)+=αδ`, actor `implemented`/critic `partial`), C.31 (bootstrapping),
  C.32 (two-component DA: Component-1 salience + Component-2 utility-RPE, `partial`), C.34 (DA codes utility), O.19 (vmPFC/OFC
  subjective value modulates accumulator drift), O.20 (GPI policy-improvement), O.21 (average-reward `R̄` = tonic-DA vigor homologue),
  O.22 (striatal action-value coded per-action, `partial`), O.23 (reward function 1 = positive reinforcer, `implemented`). **Closest
  primitives to the social-reinforcer framing:** C.22/C.28/C.31 conditioned-reinforcer cue-shift; un-catalogued: the
  conditioned-SOCIAL-reinforcer of a CONVERSATIONAL cue + the context-conditioned SPEAK-value (both straightforward instances).
- **Literature:** Niv, Daw, Joel & Dayan 2007 *Psychopharmacology* "Tonic dopamine: opportunity costs and the control of response
  vigor" (tonic DA = average-reward → response RATE/willingness = the talkativeness set-point); Schultz 1998 *J. Neurophysiol.*
  "Predictive reward signal of dopamine neurons" (cue acquires DA = conditioned reinforcer); Krach et al. 2010 *Front. Behav. Neurosci.*
  "The rewarding nature of social interactions" (social reward via NAc/striatum); "The computational and neural bases of
  context-dependent learning" (PMC10348919); thalamo-cortical context gating (ScienceDirect S0166223625001924); cue-evoked DA promotes
  conditioned responding (Sciencedirect S089662732030012X); social-feedback → NAc/putamen reward-synapse activation (social-media-DA
  literature).
