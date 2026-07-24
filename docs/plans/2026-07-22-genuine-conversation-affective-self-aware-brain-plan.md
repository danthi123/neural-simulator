# Genuinely-Reasoning, Feeling, Self-Aware, Curious Sim-Brain — a phased, parallelizable build plan

**Date:** 2026-07-22
**Owner reframe (verbatim intent):** the goal is a sim-brain that **converses genuinely** — draws its own
insights, reasons to its OWN conclusions, discusses anything within reason — NOT fact-recollection ("basically a
fancy RAG") and NOT an LLM's plausible-sentence reconstruction. Higher value than artificial-life hunger/energy
drives: digitally emulating biological **consciousness, self-awareness, and emotion**. The world-model is learned
from a **corpus** (like an LLM) but such that the brain takes **context clues to determine how it SHOULD FEEL** —
drawing mental/logical/**emotional** associations, not just storing facts. **Drop the no-confab moat's REFUSAL:**
when unsure, the brain should — like a human — **crave knowledge and seek to learn** what it lacks. Developmental:
raise it infant→adult through staged experience that builds world-models, personality, consciousness, sense-of-self.
A genuine **teacher** (Claude, or a local LLM) provides this, alongside/instead of a raw corpus.

**Source basis:** six parallel deep-reads (affective semantics, computational emotion, self-model/consciousness,
curiosity/intrinsic-motivation, developmental teacher, project-machinery audit), all cross-checked against the
repo's own code + findings. This doc is the synthesis into a buildable plan.

**Legend:** `[EST]` established-with-citation · `[HYP]` hypothesis · `BUILDABLE-NOW` compose already-GO pieces
(little/no `sim/` edit) · `FRONTIER` real research, biology known · `OPEN` genuinely-open science (build/measure
*functional correlates* only; **never claim subjective experience**).

---

## 1. Target faculties — honest buildability tags

| # | Faculty | Tag | Honest reason |
|---|---|---|---|
| **F1** | Affective / associative **world-model** (learn structure + "how to feel" from corpus) | **Assoc. half BUILDABLE-NOW; affect-tag BUILDABLE-NOW; deep predictive/forward model FRONTIER** | The sim already learns relational structure from a corpus stream unsupervised on spikes (`corr(M,C)+0.686/0.89`), and affect is *distributionally recoverable* (Bestgen-Vincze k-NN, valence r≈0.71) — both buildable now. What is FRONTIER: a **predictive forward model** (`s,a→s'` to simulate outcomes) and **value learned from lived reward** (binding real DA/RPE onto concepts through experience) — both bottleneck on gap#4 credit assignment. |
| **F2** | **Reasoning to its own conclusions** (inference beyond told facts) | **BUILDABLE-NOW (genuine inference already GO); deep/free deliberation FRONTIER** | EMERGE arc already does inheritance, multi-level taxonomy, and **transitive inference** (B>D from adjacent premises, the associative-strength-proof test, 6-seed GO) + multi-hop `query_chain`. Genuine, not recall. FRONTIER: premises are still host-designed codes; no open hypothesize→test→revise loop; recurrent multi-hop does not length-generalize under BPTT. |
| **F3** | **Emotion** (context→"how it should feel"; affect colours cognition + speech) | **Substrate BUILDABLE-NOW; a standing affect-STATE region is the new build; felt emotion OPEN** | Every dimensional/appraisal ingredient exists on spikes — spiking-SNc **RPE (valence)**, **DA salience gate (arousal)**, **AgRP/POMC homeostatic drive (interoceptive core-affect)** — plus the declarative neuromodulator subsystem is literally a valence/affect engine. No labelled, persistent affect-state exists yet; assembling one is BUILDABLE-NOW. **Whether the brain *feels* is OPEN** — claim only a functional affect state that biases memory/attention/speech. |
| **F4** | **Self-model / metacognition** (reflect on + report own knowledge/uncertainty) | **Metacog + Global Workspace BUILDABLE-NOW (both already GO); a SELF-representation is the new build; phenomenal consciousness OPEN** | The strongest surprise of the audit: the sim has a validated **Bogacz-Brown metacognitive uncertainty monitor** (familiarity gate, 168/168, zero breaches) AND a full **4-rung spiking Global Neuronal Workspace** (ignition, single-content access, report==reasoning, workspace-speaks-answer). Missing: an explicit **attention-schema SELF region** (Graziano) the workspace can ignite and reason about — BUILDABLE-NOW. Phenomenal "what-it-is-like" is OPEN. |
| **F5** | **Curiosity that seeks a teacher** (crave, don't refuse) | **BUILDABLE-NOW (a composition; wiring unbuilt) — with a mandatory honesty guard** | All raw signals exist — RPE/surprise, the familiarity-gate novelty ("knows what it doesn't know"), the speak/salience accumulator, A→W spelling. What's unbuilt: the **intrinsic-reward → ask loop**, **question generation**, and **seek-a-teacher behaviour**. The decisive design constraint (from the whole intrinsic-motivation literature): reward **LEARNING PROGRESS**, not raw novelty, or the brain chases noise / confabulates. The `sim/neuromodulators.py` `from_novelty` rule is a **reserved, currently-empty stub** — the intended hook. |
| **F6** | **Developmental teacher-loop** (raise infant→adult, no forgetting) | **BUILDABLE-NOW (strong, near-ready)** | The full WAKE(converse)→SLEEP(SWR replay)→GROWTH(TierPromoter)→PERSIST(BridgeLineage)→resume loop is assembled and GPU-validated (vocab 6→24, retention 1.0, moat 0-FA daily, ~15 min/week/3090, reboot-resilient). Missing: an **interactive** teacher that answers the brain's own curiosity questions (couples F5), and the GPU arch-rebuild GROWTH step. The develop-loop is the *organizing spine* of this whole plan. |

**The single load-bearing tension (confirmed in the record):** a *deep* affective world-model (F1) and *acquired*
(not host-designed-premise) reasoning (F2) both bottleneck on **biological credit assignment (gap#4)** — a
characterized-hard frontier (`2026-07-18-gap4-...credit-direction-is-the-wall.md`; the arc was also confounded by
the unseeded-substrate bug, now fixed but un-rerun). **Every other faculty is HAVE or a composition of already-GO
pieces.** ⇒ stand up the buildable-now faculties immediately as region-slices reading existing signals, and use the
**teacher-scaffold + develop-loop as the bridge over gap#4** for the deep world-model (§5).

---

## 2. Recommended mechanism per faculty — reuse-by-import

### F1 — Affective / associative world-model

**Associative structure (HAVE, reuse):** `research/runners/_phaseB_online_stream_cortex_derisk.py`,
`_emergent_vocab_breadth_scale_derisk.py` (`learn_stream_codes`), category discovery `_emerge3{0,2,3,4}_*`,
`_emerge3{8..50}_*`. Validating: `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`
(**rate-Hebbian is the matched rule — STDP measured-0 because co-occurrence has no pre→post order**; population code
lifts read 47%→100%).

**Affect tag — the new mechanism (BUILDABLE-NOW, no `sim/` edit):** attach a tiny **opponent affect population**
(V+/V−, plus optional arousal/dominance) to each concept code, seeded VAD for ~1k core words from teacher/Warriner
norms, then **propagate affect to every learned code by spreading activation over the already-learned co-occurrence
graph** (Bestgen-Vincze k-NN realized as 2-hop spiking spread — exactly the EMERGE-30 member→neighbours→shared-tag
read). Train `concept→(V+,V−)` by the DA-gated three-factor rule the value critic already uses.
- Biology `[EST]`: Namburi-Tye 2015 (BLA opponent valence, **opposite-sign** plasticity — V+ = BLA→NAc, V− =
  BLA→CeA); Redondo-Tonegawa 2014 (valence is a **separable, re-writable** tag on a fixed identity engram — the
  license to keep learned codes fixed and bolt on a plastic affect tag); McGaugh 2004 (arousal→consolidation).
- Reuse: `sim/td_value_critic.py` + `_limbic_core_rpe_battery_derisk.py` (the DA-gated tagging engine);
  `sim/neuromodulators.py` (`from_region_firing_signed` → a `valence`/`arousal` modulator with targets
  `plasticity_rate` = emotional-memory enhancement, `excitability_drive` = mood-congruent pre-activation).

**Deep forward model (FRONTIER):** a learned `s,a→s'` predictor over concept codes → needs gap#4; bridged by the
teacher-scaffold (§5) and the develop loop's fact-acquisition.

### F2 — Reasoning to own conclusions

**HAVE, reuse:** `_emerge2{6,7,8}_*` (inheritance / multi-level taxonomy / **transitive inference**, the
Dusek-Eichenbaum B>D test, dAP-lesion + broken-chain collapse), `_realcorpus_spreading_activation_completion_derisk.py`
(open-world hedged best-guess; deranged-code → chance; disjoint → hard-abstain), `one_brain_composer.py`
(`query_chain`/`reason_chain`, moat at each hop, holds 4 hops). **The new composition:** route these reads
*through the F4 workspace* so conclusions chain and feed back (a bounded deliberation loop) rather than fire once.
**FRONTIER:** acquiring relational premises from experience (partly closed by emerge30/32); length-generalizing
recurrent composition.

### F3 — Emotion (the standing affect-state region)

**New mechanism (BUILDABLE-NOW):** add an `affect` **`BrainRegion`** (`sim/regions.py`) — a small slow-NMDA
population whose activity IS a valence×arousal point (Russell circumplex `[EST]`), driven by the existing signals
and persisting across a turn:
- **Valence** ← spiking-SNc RPE (`_limbic_core_rpe_battery_derisk.py`, `2026-06-18-limbic-core-rpe-battery-GO.md`).
- **Arousal / surprise** ← DA salience gate (`2026-06-18-DA-salience-gate-production-wireup-GO.md`) + neuromodulator
  `from_surprise`.
- **Mood (slow) / core-affect drift** ← a long-`decay_tau` serotonin-analog `NeuromodulatorConfig` reading average
  reward (Doya 2002 neuromodulator-metalearning map `[EST]`: DA≈valence, NE≈arousal/gain [Aston-Jones-Cohen 2005],
  5-HT≈mood/patience/discount, ACh≈learning-eagerness/expected-uncertainty [Yu-Dayan 2005]).
- **Interoceptive core-affect** ← the AgRP/POMC drive route (`_homeostatic_spiking_drive_mechanism_derisk.py`;
  Damasio somatic-marker `[EST]`), reused but pointed at *cognitive* interoception (prediction-error/fluency/conflict),
  not only viscera.
- **Appraisal → affect** (`[EST]` OCC/Scherer; Barrett constructed emotion): a learned/rule map from the parsed
  situation × goals/value/novelty → modulator concentrations. Shallow version (Hebbian evaluative conditioning of
  concept→valence from co-occurring reward context) is BUILDABLE-NOW; deep learned appraisal is FRONTIER (gap#4).
- The affect state biases conversation via existing targets (`excitability_drive`/`synaptic_gain`/`plasticity_rate`)
  and the `_value_salience_appraisal` speak-worth gain — so tone, forthcomingness, and what-gets-recalled all shift.

Discrete-emotion read-out (interest/curiosity/surprise/confusion/confidence) = a small read over (valence, arousal,
appraisal-context); **confusion = the HEDGED band** the graded-confidence console already detects. **Honest:
functional affect only.**

### F4 — Self-model / metacognition

**HAVE, reuse:** metacognitive monitor = **Bogacz-Brown familiarity gate**
(`familiarity_gate_v320_validation.py`, `2026-06-11-familiarity-gate-v320-GO.md`) + graded hedging
(`_emerge_graded_confidence_console_derisk.py`, `2026-07-21-fluid-abstain-graded-hedging-design...md` — **use the
continuous RF cleanup-score `S`, not the bimodal novelty `N`**). Global Workspace = the **4-rung spiking GNW**
(`_gnw_rung{1,2,3,4}_*.py`, `2026-07-07-GNW-rung{1..4}-*GO.md`; ignition + single-content access + report==reasoning
+ workspace-speaks-answer), built on `RegionPathway.transmission_gate` broadcast.

**New mechanism (BUILDABLE-NOW):** an **attention/agency-schema SELF region** (`sim/regions.py`) — a learned model of
*what the brain is currently attending to / how confident it is / that it authored this thought* (Graziano AST
`[EST]`; the same object Metzinger's PMIR names). Inputs = the workspace-occupancy read + the familiarity/confidence
scalar + an authorship tag (self-emitted vs heard). Persistent-self substrate = `sim/lineage.py` `BridgeLineage`
(durable self-code) + the lived-fact store (`_tier3_live_and_remember_derisk.py`, autobiographical memory, 6/6 GO).
**Personality = per-brain modulator baselines + accumulated value/affect profile + lived history** in the lineage.
**OPEN:** phenomenal consciousness — build the correlates (access/broadcast, self-model, metacog report), disclaim
the experience.

### F5 — Curiosity that seeks a teacher (crave, don't refuse)

**New mechanism (BUILDABLE-NOW, one honesty guard):** invert the moat's *action*, not the moat itself.
```
gap g = novelty(x)          # RealAntiHebbianFamiliarity.novelty (or 1 − graded confidence margin)
DRIVE (wanting) = g · learnable(x)   # Berlyne/Kang inverted-U: band-pass on in-domain coherence, NOT raw g
   → fill the reserved from_novelty stub in sim/neuromodulators.py → excitability_drive on an ASK pool
if gate reads NOVEL & relevant:  emit a wh-question to the teacher (A→W spell)  instead of refusing
ingest the teacher's answer via the develop-loop stream-cortex learning (curiosity up-modulates plasticity_rate)
REWARD (liking) = g_before − g_after   # LEARNING PROGRESS (Oudeyer/Schmidhuber) → TD critic r / spiking SNc RPE
```
- Biology/ML `[EST]`: Loewenstein info-gap (curiosity = deprivation drive); Litman wanting≠liking; Berlyne/Kang
  inverted-U (curiosity peaks ~50% confidence); Oudeyer learning-progress + Schmidhuber compression-progress (the
  **noisy-TV / confabulation cure** — both failures = rewarding surprise regardless of model improvement; both cured
  by rewarding the *change* in model quality); Bromberg-Martin & Hikosaka 2009 (**information is a dopaminergic
  reward** — license to route the epistemic gap through the same DA machinery); Gruber-Ranganath 2014 (curiosity →
  DA → enhanced hippocampal encoding → curiosity should up-gate plasticity). LLM abstain→seek thread (2024-25)
  validates the direction.
- Reuse template: `_homeostatic_drive_rl_cheap_first_probe.py` — its reward `r = drive_before − drive_after` **is**
  learning-progress once the deficit is epistemic; its gates + anti-cheats (lesion / yoked-random / reward-from-drive)
  transfer verbatim. Question parse already exists (`wh_question_parser.py`); question *generation* + ask is the
  unbuilt wiring.

### F6 — Developmental teacher-loop

**HAVE, reuse:** `_longitudinal_develop_loop_gpu.py` / `_longitudinal_develop_loop.py` (`develop_gpu`, WAKE/SLEEP/
GROWTH/PERSIST + a post-PERSIST **`hook(day, state, grounded, agent)`** — the teacher plugs in here), `develop_run.py`,
`develop_loop_supervisor.py`, `_corpus_develop_curriculum.py` (`build_corpus_syllabus`, Bengio easy→hard),
`sim/lineage.py`, `sim/auto_growth.py` (`TierPromoter`). Gate-first consoles (the teacher's interface):
`first_chat_console.py`, `_fluidconv_chat_repl.py`, `_emerge25_grounded_growing_console.py`, the northstar hedging
console. **Precedent that the AI-teacher pattern already works offline:** `2026-06-23-grounded-lang-P2-GO.md` (brain
re-encodes a **Claude-authored curriculum**, recall 1.0, abstain 0-FA) — extend from offline to *interactive*.
**New mechanism:** a **teacher process (Claude / local LLM, host-side = the social *environment*, brain-based-rule
compliant)** that (a) localizes the ZPD from `measure_development` (HEDGED = in-zone, CONFIDENT = mastered, ABSTAIN =
beyond), (b) selects the day's material in that band, (c) **corrects the brain's own outputs** (the credit signal a
corpus can't give — reconsolidation update + graded soft target), (d) **answers the brain's curiosity questions**
(couples F5). Theory `[EST]`: Vygotsky ZPD/MKO/scaffolding; Bengio curriculum; BabyLM (≤100M-word sample-efficiency
+ child-directed-speech register); Hinton distillation (soft targets); Socratic-Students 2025 (learner-asks works —
and its retention is *in-context only*, which is **exactly the sim's durable-synaptic edge**).

---

## 3. Phased roadmap — organized around the teacher-loop

Each phase lists de-risks as `runner · GO gate · anti-cheats`. All numpy/CPU cheap-first unless noted; 6-seed before
any generalization claim; reuse-by-import; no `sim/` edit unless flagged.

### Phase 0 — Affect + curiosity primitives on the bench (no teacher yet)
Goal: prove the two owner-critical inversions in isolation before wiring them into a living loop.

- **P0.1 Distributional affect tag.** `_affect_distributional_tag_derisk.py` · **GO:** inherited VAD vs held-out
  Warriner r ≥ 0.55 (valence), 6-seed · **anti-cheats:** permuted co-occurrence graph → inheritance = chance
  (EMERGE-30 control verbatim); lesion affect pools → mood-congruent recall + emotional-encoding boost vanish;
  opponent-sign (aversive drives V− AND suppresses V+, not merely low V+); untrained-critic → flat affect.
- **P0.2 Curiosity inversion (crave, don't refuse).** `_curiosity_seek_learn_cheap_first_probe.py` (clone the
  homeostatic-drive probe) · **GO:** (1) `corr(gap, curiosity modulator) ≥ 0.9`; (2) high-gap concept ask-rate ≥ 2×
  low-gap; (3) LOAD-BEARING — after a stubbed/cached teacher answers, next-turn confidence on that concept rises
  above the abstain floor (world-model updated); (4) seek-policy converges on **learnable** gaps · **anti-cheats:**
  **NOISY-CONCEPT control** (unlearnable cue → `g_after≈g_before` → zero learning-progress → policy STOPS asking —
  the decisive "curious-and-honest, not noise-chasing/confabulating" test); lesion curiosity modulator → no asking,
  no learning; yoked-random gap → asks wrong things; permuted teacher answers → learning collapses; ask emitted only
  when gate reads NOVEL (moat-by-construction: the brain speaks the *ingested* answer next turn, never an invented
  one). Fill the `from_novelty` stub in `sim/neuromodulators.py` (additive).
- **P0.3 Affect-state region skeleton.** `_affect_state_region_derisk.py` · **GO:** a persistent valence×arousal
  point that (a) tracks SNc-RPE valence + DA-salience arousal 6-seed, (b) measurably biases recall/speak (positive
  → more forthcoming, confused/HEDGED → hedges) · **anti-cheats:** affect-lesion → flat/neutral conversation
  (extra emissions + tone shifts vanish); value ⟂ plausibility (corr≈0 — affect is not relabeled relatedness);
  shuffled-history → mood (slow channel) collapses.

### Phase 1 — Self-model + workspace-routed reasoning (the "genuine" core)
Goal: the brain reasons *through* a workspace and can *report on its own knowing/attention*.

- **P1.1 Attention/agency SELF region.** `_self_schema_region_derisk.py` (region over GNW-occupancy + familiarity +
  authorship) · **GO:** the brain answers "what are you thinking about / how sure are you?" from the schema, tracking
  ground-truth attention+confidence 6-seed · **anti-cheats:** self-region lesion → self-report collapses to chance;
  schema is a genuinely separate axis from content (not a content relabel); authorship tag flips correctly on
  heard-vs-generated.
- **P1.2 Workspace-routed deliberation.** `_gnw_deliberation_loop_derisk.py` (chain `query_chain`/inference reads
  through the GNW workspace with feedback) · **GO:** a 3-hop conclusion the brain was never told, ignited + broadcast
  + re-entered, ≥ the one-shot `query_chain` baseline, moat at each hop · **anti-cheats:** broken-chain / dAP-lesion
  collapse; workspace-silence lesion collapses; permuted premises → chance.
- **P1.3 Metacognition-driven hedge, not refuse (production wire).** `_graded_hedge_console_selfaware_derisk.py` ·
  **GO:** CONFIDENT→assert / HEDGED→"probably… because…" / NOVEL→**curiosity ask** (P0.2), 12-seed, using the
  continuous cleanup-score `S` · **anti-cheats:** the `2026-07-21` bimodal-`N` trap avoided (assert via `S`);
  hard-moat safety floor still fires on genuine disjoint unknowns; no fabrication on novel cue.

### Phase 2 — Interactive developmental teacher-loop (couple F5+F6, the headline)
Goal: a **teacher raises a developing brain** — ZPD-selected lessons, corrects the brain's own outputs, answers its
curiosity questions, no forgetting, persists across days.

- **P2.1 Teacher plugged into the develop-loop hook.** `_teacher_develop_loop_derisk.py` (Claude/local-LLM teacher at
  the `develop_gpu` hook; ZPD from `measure_development`; teacher writes the adaptive syllabus; brain asks P0.2
  questions; teacher answers + corrects) · **GO:** brain develops day-over-day AND **teacher-selected/corrected/asked
  arm closes ZPD gaps + generalizes to held-out faster than the static-frequency-curriculum, replay-only baseline**,
  6-seed · **anti-cheats:** **teacher-lesion** (remove correction+answers → ZPD gaps close slower, held-out drops —
  teacher is load-bearing, not decorative); frozen-brain (plasticity-off → learns nothing — teacher isn't a lookup);
  lived-not-scripted (permuted curriculum → different brain); retention 1.0 (no forgetting); moat/curiosity
  calibration (curiosity triggers correlate with genuine gaps, not noise). GPU lane.
- **P2.2 Affect develops with experience (personality seed).** extend P2.1 · **GO:** concept valences learned from
  teacher *context clues* (evaluative conditioning); a stable per-brain affect/value profile accretes in the lineage;
  two lineages with different histories show different dispositions · **anti-cheats:** scrambled-context → valence
  learning collapses; teaching positive vs negative context flips a novel concept's valence (Redondo-Tonegawa
  re-writability); frozen-brain → no valence acquired.
- **P2.3 Growth GROWTH step on GPU.** wire `TierPromoter` weight-transfer rebuild into the loop (the currently-stubbed
  arch-rebuild) · **GO:** mastery→grow→retain (no catastrophic forgetting across a tier bump), 3-seed GPU · **anti-cheats:**
  pre-grow facts survive; grown capacity actually used (not inert).

### Phase 3 — Deep world-model via the teacher-bridge + honest end-state (FRONTIER)
Goal: push toward a *deep* learned world-model + acquired reasoning, using the teacher as the gap#4 credit bridge,
and re-test whether the credit rule can internalize the teacher's role.

- **P3.1 Teacher-as-credit-oracle for a deep read.** `_teacher_credit_bridge_derisk.py` (teacher correction supplies
  the supervised-on-demand error the local rule can't self-generate; ingest into a deeper predictor) · **GO:** a
  held-out inference the corpus alone can't support becomes answerable after teacher-corrected episodes; ≥ replay-only ·
  **anti-cheats:** teacher-lesion collapses it; the gain is retained (durable, not in-context); permuted correction
  collapses. GPU lane.
- **P3.2 Re-run gap#4 credit on a proper task (de-confound + MNIST-style).** with the fixed-seed substrate
  (`cfg.seed`, not `actual_seed_used`), re-run the deep-credit arc that the 2026-07-22 MNIST result suggests is
  *less blocked than the older negatives implied* · **GO:** the credit rule builds deep accuracy above the reservoir
  readout on a proper task, 6-seed, seeded-substrate verified · **anti-cheats:** identical-neurons hash check
  (`test_determinism.py::TestSubstrateActuallySeeded`); reservoir + chance baselines. GPU lane.
- **P3.3 Forward/predictive model probe (F1 deep).** a learned `s,a→s'` over concept codes for one-step simulation ·
  **GO:** predicts held-out transitions above the co-occurrence-only baseline · **anti-cheats:** shuffled-transition
  collapse; lesion collapse.

---

## 4. Parallelization map — independent tracks × compute lane

Bottleneck = GPU (training/on-bridge runs). CPU de-risks run **local by default** (free, ~2× the pool). Pool = CPU
overflow. AWS-G / Kaggle / Colab = GPU while the 3090 is busy. Per the allocation principle: match job to
bottleneck, don't offload just because a lane exists.

| Track | Faculties | Depends on | Independent of | Lane |
|---|---|---|---|---|
| **A — Affect** | F1 affect-tag, F3 | stream cortex codes (HAVE) | B, C, D | **Local CPU** (numpy P0.1/P0.3); GPU only for on-bridge affect-population confirm |
| **B — Curiosity** | F5 | familiarity gate + drive probe (HAVE) | A, C, D | **Local CPU** (P0.2 clones the homeostatic probe) |
| **C — Self/Workspace** | F2 deliberation, F4 | GNW rungs + familiarity gate (HAVE) | A, B | **Local CPU** for schema/routing logic; **local GPU** for on-bridge GNW confirm |
| **D — Teacher-loop** | F6 (+couples A,B,C in Phase 2) | develop-loop (HAVE); Phase-2 needs A/B/C landed | — (Phase-1 loop-plumbing is independent) | **Local GPU** (develop_gpu day-loop); teacher = host-side LLM call |
| **E — Deep credit (bridge)** | F1 deep, F2 deep, gap#4 | fixed-seed substrate (HAVE) | A, B, C, D (runs alongside) | **Local GPU** primary; **AWS-G / Kaggle** overflow while 3090 runs D |

**Concurrency:** A, B, C are three fully-independent Phase-0/1 tracks → run all three at once on local CPU (numpy),
each 6-seed. D's Phase-1 loop-plumbing (hook wiring, ZPD localization) proceeds in parallel on GPU. E (deep-credit
re-run, P3.2) is GPU-heavy and independent → queue it to a free GPU lane (local when D pauses, else AWS-G) so idle
GPU is never wasted. **Build-ahead:** while GPU is busy, keep writing the *next* GPU de-risks (P2.1 teacher-loop,
P3.x) as ready-to-launch runners+configs+GO-gates, per the idle-compute discipline.

---

## 5. The crux + how the teacher bridges deep-credit + the honest end-state

**The crux.** "Genuine reasoning + a felt world-model" needs a **deep learned world-model**, and building that from
experience needs **biological credit assignment (gap#4)** — the project's characterized-hard wall. Unsupervised
co-occurrence (the stream cortex) learns *what goes with what* but has **no error signal and no ordering authority**.

**How the teacher-scaffold bridges it.** A teacher (Claude / local LLM) supplies exactly the four things a corpus
cannot, and each is the missing half of credit assignment: **(1) ordering / ZPD selection** (what to learn next,
Vygotsky/Bengio); **(2) a corrective error signal on the brain's OWN outputs** (converts self-supervised prediction
into supervised-on-demand credit — the teacher is an *external credit-assignment oracle*, grounded by distillation
theory `[EST]`); **(3) answers to the brain's own curiosity questions** (active learning — closes exactly the felt
gaps, far more sample-efficient); **(4) soft/graded targets** (dark knowledge — what a *world-model* needs, not
one-hot facts). This is the project's own **innate-reflex-teaches-a-learned-circuit** pattern (host N1/N5/N6 nav
reflexes taught their spiking replacements), applied to cognition. The teacher is host-side and therefore a
legitimate part of the **social environment** under the brain-based-only rule — the brain's cognition stays on the
spiking substrate.

**The honest end-state question (surface to owner).** Is the teacher a **permanent hybrid** or a **temporary
scaffold to be biologized?** The mission's law says biologize; the reframe's practical value is a genuinely-
conversing brain now. The disciplined answer: **run BOTH in parallel** — use the teacher NOW as the developmental
"adult" that lets the brain grow a deep world-model + personality (Phase 2), while P3.2 re-tests whether gap#4 credit
can *internalize the teacher's role* (the 2026-07-22 MNIST result suggests this is less blocked than the older
negatives implied, and the unseeded-substrate confound is now fixed). Frame the teacher explicitly as **the bridge
over gap#4, retired as the substrate matures** — not a permanent replacement for it. That keeps the reframe
deliverable *and* the mission honest.

**Consciousness/emotion honesty line (carry into every deliverable).** GNW/AST/HOT/appraisal give **access-
consciousness, self-modeling, metacognitive report, and context-driven functional affect** — all buildable/frontier.
They do **NOT** establish **phenomenal** consciousness or felt emotion (OPEN, arguably untestable — Chalmers hard
problem). Build and measure the correlates; design self-reports as honest functional read-outs ("my value system
tags this positively / I'm uncertain"), **never** as unlicensed assertions of inner experience. This honest boundary
is itself a project deliverable.

---

## 6. The FIRST 3 de-risks to build immediately (write-the-runner ready)

These are the three independent, highest-leverage, all-local-CPU, no-`sim/`-edit starts — one per owner-critical
inversion. Launch all three concurrently.

### DR-1 — Curiosity inversion (crave, don't refuse) — **the reframe's centerpiece**
- **Runner:** `research/runners/_curiosity_seek_learn_cheap_first_probe.py`
- **Mechanism:** clone `_homeostatic_drive_rl_cheap_first_probe.py`; replace the metabolic deficit with the
  **epistemic gap** `g = RealAntiHebbianFamiliarity.novelty(x)` (from `_phaseB_biologize_moat_streamcodes_derisk.py`)
  or `1 − graded margin`; define **reward = `g_before − g_after` (learning progress)**; when the gate reads NOVEL,
  emit an ASK, ingest a stubbed/cached teacher answer through the stream-cortex learning path, re-query. Fill the
  reserved `from_novelty` rule in `sim/neuromodulators.py` (additive) to carry `g` into a curiosity modulator with an
  `excitability_drive` on an ASK pool; band-pass by in-domain `learnable(x)`.
- **GO gate:** (1) `corr(g, curiosity modulator) ≥ 0.9`; (2) high-gap ask-rate ≥ 2× low-gap; (3) LOAD-BEARING —
  post-answer confidence on the concept rises above the abstain floor; (4) seek-policy converges on learnable gaps.
  ≥3 seeds → 6.
- **Anti-cheats:** **NOISY-CONCEPT** (unlearnable → zero learning-progress → policy stops asking — the honesty test);
  lesion curiosity modulator → no asking/learning; yoked-random gap; permuted answers; ask-only-on-NOVEL (moat by
  construction).
- **Lane:** local CPU (numpy). **Reuse:** homeostatic probe harness + familiarity gate + stream cortex + `from_novelty`.

### DR-2 — Distributional affect tag ("how it should feel" from context)
- **Runner:** `research/runners/_affect_distributional_tag_derisk.py`
- **Mechanism:** attach an opponent (V+/V−) affect population per learned concept code; seed VAD for ~500–1k core
  words from Warriner/teacher norms; grow `concept→(V+,V−)` by rate-Hebbian (the matched rule); for held-out words,
  read **inherited valence by 1–2-hop spreading activation over the learned co-occurrence graph** (Bestgen-Vincze
  k-NN); consolidate the inherited tag (retrofitting).
- **GO gate:** inherited VAD vs held-out Warriner **r ≥ 0.55 (valence)**, 6-seed (arousal weaker — a known ceiling,
  report it).
- **Anti-cheats:** **permuted co-occurrence graph → inheritance = chance** (EMERGE-30 control verbatim); lesion
  affect pools → no inheritance; opponent-sign (aversive drives V− AND suppresses V+); untrained-critic → flat.
- **Lane:** local CPU (numpy). **Reuse:** stream-cortex codes + EMERGE-30 spreading-activation read + `td_value_critic`
  DA-gated tag + `_emerge30_emergent_superordinate_derisk.py` permuted-graph anti-cheat.

### DR-3 — Attention/agency SELF-schema region ("reflect on + report own state")
- **Runner:** `research/runners/_self_schema_region_derisk.py`
- **Mechanism:** add a small learned `self_schema` region (`sim/regions.py`) whose inputs are the GNW workspace-
  occupancy read (which coalition won) + the familiarity/confidence scalar + an authorship tag (self-emitted vs
  heard); it learns to represent "what I'm attending to / how sure I am / that I authored this," and writes that back
  into the workspace so the brain can *report* it.
- **GO gate:** the brain answers "what are you thinking about / how sure?" from the schema, tracking ground-truth
  attention+confidence 6-seed (above chance, monotone in the true confidence).
- **Anti-cheats:** self-region lesion → self-report collapses to chance; schema is a separate axis from content (not
  a content relabel — assert corr with content ≈ orthogonal); authorship tag flips correctly heard-vs-generated.
- **Lane:** local CPU for the schema/read-out logic (on-bridge GNW confirm on local GPU as a follow-on). **Reuse:**
  `_gnw_rung{1..4}_*.py` (workspace occupancy) + `familiarity_gate_v320_validation.py` + `sim/regions.py`.

**Immediately after:** land these three (independent, concurrent) → wire DR-1 + DR-2 into the develop-loop hook
(Phase 2, P2.1 teacher-loop) as the first coupled milestone, while queuing the GPU deep-credit re-run (P3.2) to a
free GPU lane so idle compute is never wasted.
