# Robust multi-cue competition parser (Phase 1) — deep-research GATE: the gate finds Phase 1 ALREADY BUILT END-TO-END

**Date:** 2026-06-20
**Type:** Standing deep-research + catalog-review gate for the owner's stated PRIMARY next capability
(robust, language-agnostic comprehension via a Bates-MacWhinney multi-cue Competition Model parser; memory
`project_conversational_primary_robust_multicue_parser`). READ-ONLY; no code built/edited/run except running the
existing CI guard to verify the claimed state.
**Scope I was asked to gate:** **Phase 1** = robust-to-imperfect-English multi-cue competition, reusing the
existing #2 biased-competition mechanism.
**The honest top-line up front:** **the gate fires AGAINST a re-scope — Phase 1 is not a future build, it was
already designed → de-risked (numpy + spiking) → production-wired behind a default-OFF flag → CI-guarded, all on
2026-06-19. Phase 2 (case-marking cue, cross-language) is ALSO already built the same way. The remaining Phase-1
work is small, identified, and below.** This document verifies that state against the actual source/catalog text
(trust-but-verify), states what genuinely remains, and recommends the highest-leverage next move.

---

## 0. TL;DR

- **The direction is correct and the mechanism is the right one.** The current `BridgeParser` is brittle because it
  assigns thematic roles by **word position alone** (`(position × voice) → role`); that brittleness has two faces —
  (1) it dies on imperfect English (dropped/scrambled/object-fronted words corrupt the position cue), and (2) it is
  an English-only parser (case-marked languages signal roles by morphology, not order). These are **one** structural
  fact (everything staked on one cue), and the **Competition Model** (Bates-MacWhinney: order/animacy/agreement/case
  as graded, competing, *learned-validity* cues) is the verified fix. **All confirmed against the actual parser
  code + the catalog (G.18, G.12) + the literature.**
- **Phase 1 is DONE (not to-be-built).** Verified, in this order, on 2026-06-19:
  1. **Scoping** — `2026-06-19-multicue-competition-parser-scoping.md` (the diagnosis, mechanism map, reuse-vs-new,
     GO bar + controls; controller-reviewed, commit `0ecdb628`).
  2. **Numpy MECHANISM de-risk — GO 6/6 seeds** — `2026-06-19-multicue-competition-derisk.md`. Multi-cue with
     LEARNED validities recovers degraded English at **0.95** role accuracy where position-only collapses to **0.26
     (object-front 0.00)**; cue-lesion + no-learning + permuted-cue controls all collapse; moat 0/240.
  3. **Spiking-substrate de-risk — GO** — `2026-06-19-multicue-competition-spiking-derisk.md`. The competition is
     real `cp_firing_states` on a `SimulationBridge` (re-pointed `biased_competition_buffer.py` Wong-Wang/Rutishauser
     WTA over thematic ROLES + plastic cue→role projections). Install path **5/6 seeds**; the load-bearing learning
     finding (plain Hebbian is the WRONG rule; three-factor/reward is load-bearing) was found and fixed here.
  4. **On-substrate LEARNING firmed + REWARD neuralized** — `2026-06-19-multicue-learning-firm-and-neural-reward.md`.
     The three-factor learner produces the correct cue-validity SIGNATURE on **6/6 seeds**, and the reward term moved
     from a host formula to a **spiking SNc dopamine RPE** (signature recovered 6/6 on real spikes; moat 0/6).
  5. **Production WIRE-IN — DONE** — `2026-06-19-multicue-competition-agent-wirein.md`.
     `BrainConversationalAgent(enable_multicue_competition=True, multicue_verbs=…)` routes `hear()` through
     `MultiCueRoleParser`; the agent answers who/what CORRECTLY on object-fronted English where the default inverts
     the roles; flag-OFF byte-identical; CI guard `tests/test_multicue_competition_agent.py` (7 tests). **I ran it:
     7 passed in 30.6 s** — the wire-in is live and real, not stale.
- **Phase 2 (case cue) is ALSO DONE** the same way: `2026-06-19-case-cue-crosslanguage-derisk.md` (GO 5/6; the
  cross-linguistic dissociation — same code, English `w_case`→floor / Japanese `w_case`→top — **6/6 seeds**) +
  `2026-06-19-case-cue-crosslanguage-agent-wirein.md` (`enable_case_competition` flag + `tests/test_case_cue_crosslanguage_agent.py`).
- **What genuinely REMAINS for Phase 1** (small, bounded, all documented as honest residuals in the existing
  findings — none is a re-scope): **(R1)** the **production-default FLIP** (both flags are default-OFF; nothing routes
  the production demos through the multi-cue parser yet) — but this needs a decision because it is **NOT free**
  (it forces GPU + costs latency + narrows the validated scope to 2-noun transitive); **(R2)** the **end-to-end
  spiking-readout operating-point friction on object-fronted items** (the documented seed-variance; a Wong-Wang WTA
  calibration study, NOT a learning failure); **(R3)** extend the competition beyond the **2-noun transitive** clause
  (1-noun, 3+-noun, attributed adj+noun, multi-clause) — currently a surface-order fallback; **(R4)** neuralize the
  **host feature lexicons** (animacy / verb-fit) into a learned synaptic lexical-feature map — the last cognitive
  host scaffold in the path (the moat-style boundary, already flagged).
- **The recommended next move is NOT "build Phase 1."** It is: present the DONE state to the owner and pick among
  **(a)** the production-default flip with its honest cost (R1), **(b)** the highest-reward capability extension
  (R3 — multi-clause/attributed multi-cue, which is where the parser's *coverage* — not its robustness — is now the
  bottleneck), or **(c)** proceeding to Phase 3 / the next roadmap tier. My ranked recommendation is in §7.
- **No known wall hit for Phase 1.** Cue-integration-to-a-winner is **rate-coded reliability-weighted accumulation**
  (catalog G.18; attractor accumulators do this near-optimally with rate codes), categorically NOT the analog/pre-
  spike decorrelation that walled before (Mikulasch-Priesemann). The one residual (R2) is a WTA *operating-point*
  friction, not a substrate limit. The genuinely-dendritic frontier (R4's deepest form / generalization across
  similar concepts) is the separate, already-mapped arc, NOT Phase 1's robustness goal.

---

## 1. Diagnosis — why the current `BridgeParser` is brittle (verified against its code)

### 1.1 What the parser actually does (read from source, `brain_conversational_agent.py:28`)
`BridgeParser` builds **6 conjunction units** indexed `k = position*2 + voice` (voice 0=active, 1=passive) and a
hard-coded ground-truth map `_GT = {0:agent, 1:patient, 2:action, 3:action, 4:patient, 5:agent}` (line 25). Each
conjunction unit projects (plastic, Hebbian) to **3 role ensembles** (agent/action/patient, `R=40` neurons each).
Training (`_train`, lines 110-120) teacher-drives the conjunction unit **and** its correct role simultaneously, so
Hebbian co-firing grows that one (position×voice)→role synapse. At comprehension, `role_of(position, voice)` drives
the conjunction and reads the max-firing role; `parse(words, voice)` assigns each word the role its
`(position, voice)` conjunction reads out.

**The load-bearing fact:** the ONLY input feature is **serial position** (voice merely selects *which* position→role
table — active 1st→agent vs passive 1st→patient). There is **no semantic cue, no agreement cue, no case cue, and no
competition** — the role is a deterministic function of position. `FrameParser` / `MultiFrameParser`
(`frame_parser.py`, `_phaseB_multiframe_comprehension_derisk.py`) generalize the structural cue to
verb-position→frame (SVO/VSO/OSV), but that is still **one structural cue family** (which position→role table to
apply), confirmed by reading them.

### 1.2 Why that is brittle — and why it is ONE problem (both failure modes)
- **Imperfect-English failure.** Real users drop words ("dog north"), scramble ("north the dog goes"), front objects
  ("the bone, the dog ate"). Each **corrupts the position cue**. With position as the *sole* cue, a corrupted
  position → a corrupted role. There is no fallback — the verified spiking de-risk measured the position-only parser
  at **0.000** on object-fronted input (the position table maps the fronted patient → agent).
- **Cross-language failure.** Position-only comprehension is a *typological commitment to a fixed-word-order
  language*. Case-marked languages (Japanese ga/wo, Russian/Turkish case, agreement-heavy German/Italian) signal
  roles by **morphology, not order** — the very cue the parser ignores. A position-only parser IS an English-only
  parser.

**Both are the same structural fact:** everything is staked on one cue, so the parser dies whenever that cue is
unavailable — whether the *utterance* degraded it (imperfect input) or the *language* doesn't use it (cross-language).
The Competition Model's answer is identical in both cases: **carry the role assignment on whatever OTHER cues
survive, weighted by their learned validity.**

---

## 2. The mechanism (verified) + the reuse-vs-new split (as actually built)

### 2.1 The Competition Model, anchored to the catalog (trust-but-verify, confirmed)
A finite inventory of **cues** (word order, animacy, agreement, case, selectional plausibility) each carries
information about thematic roles; each has a **cue validity** (availability × reliability in the language); the
comprehender's cue **weights are learned to track those validities**; at comprehension the cues **compete** and the
role assignment is the **settled winner of the weighted competition**. Degrade a high-weight cue → lower-weight cues
carry it (graceful degradation). Different languages → different learned weights (English=order high, Italian/German=
agreement high) — **same architecture, different learned weights** = the "adapt by re-learning weights, not re-coding"
property the owner wants.

**Neural realization — verified against the actual catalog text** (`sim-catalog/references/feature-catalog.md`):
- **G.18** (line 2850, "Probabilistic reasoning from symbols — logLR accumulation in LIP"): the catalog states
  *verbatim* that "LIP's accumulator is **not specific to perceptual evidence; it integrates *any* evidence weighted
  by reliability**. Each shape contributes its known logLR **additively**." This is **exactly cue-validity-weighted
  integration as a firing-rate accumulator** — the cue weight = the cue's reliability (logLR scaling), the role
  decision = the accumulator's bounded winner. **Confirmed the scoping cited this faithfully.**
- **G.12** (line 2774, Broca's area): the catalog states it "supports comprehension of grammatically complex
  (**non-canonical**) sentences." This is the documented neural locus for the case where the order/syntactic cue is
  hard and other cues must carry comprehension. (The scoping's paraphrase — "semantic cues carry comprehension when
  syntax is hard" — is the standard Competition-Model behavioral reading of this entry; the bare catalog line
  supports the non-canonical-comprehension role directly.)
- **G.16 / G.17** (drift-diffusion / LIP accumulator) and **G.06 / G.08** (PFC WM attractor) supply the competitive
  WTA that settles the decision and the recurrent substrate that holds the competing role hypotheses.

### 2.2 The architecture, as actually built (not as proposed)
The built system (`_phaseB_multicue_competition_spiking_derisk.py` → `multicue_role_parser.py`) realizes:
```
   cue populations (signed pairs)        role assemblies (Wong-Wang accumulators)
   position  --w_pos----\                ┌──────────────┐
   animacy   --w_anim----+--> evidence-->│  sel_agent   │◄─┐ mutual inhibition
   verbfit   --w_vfit----+               │  sel_patient │◄─┘ (sel_r → sel_FS_r → sel_(s≠r),
   (P2) case --w_case----/               └──────────────┘     Rutishauser selective FS)
   lexbias   --w_dist---x (chance cue the learner drives to ~0)
       plastic cue→role weights = learned cue VALIDITY (three-factor reward-modulated)
```
- **Cue populations** — each cue is a small signed population whose firing is *that cue's vote* (agent vs patient) for
  a noun. Position = the existing conjunction signal; animacy / verb-fit = the lexical-feature lookups.
- **Learned cue→role weights = cue validity** — plastic synapses; **three-factor reward-modulated** learning
  (spike-eligibility × spiking-SNc-RPE × vote sign). The de-risk's load-bearing finding: **plain Hebbian co-firing
  cannot learn validity** (it has no error term, so it cannot down-weight a high-co-occurrence-but-unreliable cue);
  the error/reward term is what's load-bearing. The learned signature is `w_position ≪ w_animacy ≈ w_verbfit`,
  distractor → 0.
- **Biased competition + accumulation = the settle** — `sel_agent`/`sel_patient` Wong-Wang accumulators in mutual
  inhibition (the re-pointed `biased_competition_buffer.py` WTA — the #2 mechanism the owner cited). Degrading a cue
  removes one summand from the reliability-weighted vote; if the surviving cues still separate the roles, the
  accumulator still settles correctly. **This is where robustness comes from.**

### 2.3 Reuse-vs-new — what was actually reused, what was actually new (verified)
| Piece | Reused from | Built new |
|---|---|---|
| Mutual-inhibition role WTA (`sel_agent`/`sel_patient` + `sel_FS_*`) | `biased_competition_buffer.py` (the #2 mechanism: Wong-Wang accumulators + Rutishauser selective FS) — **re-pointed referent→ROLE** | the role re-pointing (a mechanical rename) |
| Animacy + verb-selectional-fit lexicons | `biased_competition_buffer.py` `ANIMACY` / `VERB_SELECTS` (imported with a drift assertion) | — |
| Position cue + Hebbian training loop | `BridgeParser` conjunction units + role ensembles + the v16 `_train` loop | — |
| The no-confab moat | the buffer's abstain-on-no-decisive-winner logic | the content-gate margin from learned semantic weights (`_default_margin`) |
| The reward / RPE | nav g11 `spiking_snc` (`I_snc = tonic + reward_gain*r − value_gain*V`) | the three-factor cue-validity learner |
| **NEW (the minimal piece):** plastic `{animacy,verbfit,(case)}_cue → {agent,patient}` projections + a per-word cue driver | — | **yes — this is what Phase 1 added: 2-3 plastic cue projections + the driver. Confirmed ~90% reuse, NO `sim/` edit.** |

**Net (verified):** Phase 1 was **assembly of validated parts + a few plastic cue projections + a cue driver**, all
reuse-by-import / additive, **NO `sim/` edit**, exactly as the scoping predicted.

---

## 3. The evidence the built system meets the GO bar (the controls actually passed)

Reproduced from the verified findings (numpy mechanism, 6 seeds; + the spiking install path, 6 seeds):

| Control (the scoping's GO bar) | Numpy result | Spiking install | What it proves |
|---|---|---|---|
| MULTI-CUE on degraded battery (≥0.80) | **0.950** (6/6) | **0.896** (5/6) | the cues carry degraded input |
| POSITION-ONLY collapses on the SAME battery (≤0.45) | **0.258** (object-front **0.000**) | **0.281** | the battery genuinely degrades position — the win is the *added cues*, not a generically better parser |
| CUE-LESION (zero animacy+verbfit) → ≈ position-only | **0.258/0.267** | **0.281** | the semantic cues are **load-bearing** |
| NO-LEARNING (weights frozen) collapses | **0.610** (and a later fix made the *naive-prior* control collapse cleanly) | — | the validities are **LEARNED, not hand-set** |
| PERMUTED-CUE (scrambled cue tags) collapses | **0.258** | — | the cues carry **real** role info, not a relabelled position signal |
| Held-out fillers + verbs | held-out 0.950 | — | **not memorizing** examples |
| no-confab MOAT | **0/240 breaches** | **0 on every seed** | the moat is **not weakened** — a content-tie ABSTAINS |
| Learned signature | `pos 0.34 ≪ anim 0.76, vfit 0.72, distractor 0.03` | `pos 4.7 ≪ sem ~20, distr 2.2` | cue-validity learning ON the substrate |

**The clincher (verified):** the learner ends every seed with `w_position` driven **below** the semantic weights and
the chance-validity distractor driven to **~0** — that is cue-validity learning (down-weight the unreliable cue,
discard the chance cue), the Competition Model's central claim, realized as three-factor synaptic learning. The
production wire-in's CI guard (which **I ran: 7/7 passed**) asserts the agent answers who/what correctly on
object-fronted English where the default inverts the roles, with the moat intact and flag-OFF byte-identical.

---

## 4. What GENUINELY remains for Phase 1 (the honest residuals — none is a re-scope)

Every item below is already named as an honest residual in the existing findings; I am consolidating + ranking them.

- **(R1) Production-default FLIP — a DECISION, not a build.** Both flags (`enable_multicue_competition`,
  `enable_case_competition`) default **OFF**; no production demo/agent routes through the multi-cue parser yet (the
  rf/onebrain default comprehension path is byte-unchanged). This was deliberate (banked behind a flag, like
  `enable_attributed`/`enable_biased_competition`). **The honest cost of flipping** (so it is a decision, not free):
  (i) the spiking competition needs `SIM_BACKEND=cupy` for production scale (the tiny bridge is CPU-fine, but the
  full pipeline is GPU) — flipping the *library default* would force GPU on every default agent + break numpy-CPU
  portability (the exact reason the onebrain 320 flip was scoped to the *demo* only, not the library constructor);
  (ii) per-op latency cost (a spiking WTA settle per parse); (iii) the validated scope is the **2-noun transitive**
  clause — a 1- or 3+-noun input falls back to a surface-order read (the moat still abstains, so it is safe but not
  robust). **Recommendation:** flip it on the *specific demo(s)* that target robust/imperfect English (the owner's
  use case), NOT the library constructor default — mirroring the onebrain-320 precedent.
- **(R2) End-to-end spiking-readout friction on object-fronted items** (the documented seed-variance). The validity
  LEARNING is robust (correct signature 6/6, host or spiking-RPE); the END-TO-END accuracy is seed-variable on the
  hardest (object_front) items because the **tiny-scale Wong-Wang WTA's object_front resolution has per-seed
  operating-point friction**. Verified NOT a learning failure (signature correct on every seed) and NOT fixed by
  naive levers (more epochs/read-steps: no change; bigger sel pools: *worse* — mis-calibrated WTA). **This is a
  genuine WTA calibration study (selective-inhibition gain vs pool size), correctly flagged-not-escalated. The
  install path is the robust 5/6 deployment regardless.** A real, bounded follow-on — not a wall (it's an
  operating-point, not a substrate limit).
- **(R3) Coverage beyond the 2-noun transitive clause.** The drop-in's validated scope is a single 2-noun transitive
  (the de-risk scope). Real conversation has **1-noun ("dog runs"), 3+-noun, attributed (adj+noun: "big red apple"),
  and multi-clause** inputs. Today those fall back to a surface-order read. The project already has the pieces to
  extend (the `enable_attributed` adj-noun parser, the `enable_multiframe` parser, the embedded-clause work
  `2026-06-19-embedded-clause-parse-derisk.md`) — extending the multi-cue competition to compose with these is the
  natural **coverage** frontier. **This is where the parser's bottleneck has now MOVED**: robustness (the headline
  brittleness) is solved; *coverage* of richer inputs is the remaining gap.
- **(R4) Neuralize the host feature lexicons** (animacy / verb-selectional-fit → a learned synaptic lexical-feature
  map). This is the last cognitive host scaffold in the comprehension path — the lexicons supply each cue's VALUE
  (which cue population to light), they do NOT supply the role decision (the PERMUTED-CUE + NO-LEARNING controls
  guard against the lexicon doing the discrimination, and it is the same legitimate lexical-front-end boundary that
  `FrameParser` and the buffer's `content_bias_target` occupy). The conversion target — a learned lexical-feature map
  — is the buffer's already-documented boundary. **In its deepest form (similarity-structured features so "dog"/"cat"
  are related cue-bearers) this touches the generalization/dendritic arc; the near-binary animacy/verb-fit cues do
  NOT need that — they are a learnable lexical-feature lookup.**

---

## 5. Anti-cheats — already satisfied + the standard for any follow-on

The mechanism was validated with exactly the decisive controls the gate requires (all reproduced in §3, all PASS):
**position-only collapse** (the load-bearing control — the win is the added cues, not a better parser),
**cue-lesion** (the cues are load-bearing), **no-learning** (the weights are learned, not hand-set — the primary
"hand-tuned cues masquerading as a learned model" mislead, directly excluded), **permuted-cue** (the cues carry real
role info), **held-out fillers/verbs** (not memorizing), and the **no-confab moat** (0 breaches everywhere — a
content-tie abstains, never confabulates). **Any follow-on (R1-R4) MUST keep these:** in particular the moat (0
false role-commitments on genuinely ambiguous input) and a baseline the new piece must BEAT (for R3, the
current 2-noun parser on multi-clause input; for R4, the host-lexicon path the learned map must match). The standard
is the project's load-bearing-controls-first discipline that retracted the 2026-05-14 compose-concept and
transitive-inference claims.

---

## 6. Honest top-line — is robust Phase-1 parsing achievable? It is ACHIEVED; any wall?

**Robust Phase-1 parsing is achieved, on the point-neuron substrate, reusing #2 biased-competition + a small
cue-integration piece — exactly as the gate's question framed it.** No part of Phase 1 hit a known wall:
- **The cue competition itself is rate-coded** reliability-weighted accumulation (catalog G.18; attractor
  accumulators do this near-optimally with firing-rate codes), categorically **NOT** the analog/pre-spike
  decorrelation/whitening that walled before (Mikulasch-Priesemann point-neuron limit). Confirmed: the WTA settles
  fine; the verified de-risks explicitly checked and the point-neuron risk is LOW.
- **The one residual that looks like friction (R2) is an operating-point**, not a substrate limit — a Wong-Wang WTA
  calibration on the hardest items, with the install path robust at 5/6 regardless.
- **The only genuinely-dendritic-adjacent piece** is R4's *deepest* form (similarity-structured lexical features) and
  generalization across similar concepts — but that is the **separate, already-mapped generalization arc**, NOT
  Phase 1's robustness goal, and the near-binary animacy/verb-fit cues Phase 1 uses do not need it.

**The phased build, corrected to reality:**
| Phase | Stated scope | ACTUAL status (verified) |
|---|---|---|
| **Phase 1** — robust imperfect-English multi-cue competition | "DO FIRST; reuse #2" | **DONE** — scoped → numpy GO 6/6 → spiking GO 5/6 → learning firmed 6/6 → reward neuralized → production-wired behind a flag + CI guard (7/7 ran green). Residuals R1-R4 (small, bounded). |
| **Phase 2** — case/agreement cue + non-English toy (true cross-language) | "next; modest incremental effort" | **DONE** — case-cue de-risk GO 5/6; the cross-linguistic dissociation (same code, English `w_case`→floor / Japanese `w_case`→top) **6/6 seeds**; production-wired behind `enable_case_competition` + CI guard. Residuals: toy-calibration polish + the shared R2/R4. |
| **Phase 3** — sub-word morphology (agglutinative langs; fused/portmanteau case) | "DEFERRED — months-scale new layer" | correctly DEFERRED — needs a morpheme-segmentation tier (a new representational layer, plausibly the dendritic/compositional arc). |

---

## 7. The recommended next move (ranked) — because the scoped build is already done

The gate's job is to point the next action at the highest reward-per-effort. Since Phase 1 is built, the next move is
**not "build Phase 1"** — it is one of:

1. **★ FIRST: present the DONE state + flip the production default on the imperfect-English demo (R1, scoped, not the
   library constructor).** Reward HIGH / effort LOW. The owner's stated primary ("robust-to-imperfect-English
   comprehension") is *already validated and wired* — the only thing between it and being the live default is a
   deliberate flip, which (per the onebrain-320 precedent) should be scoped to the demo(s) that target this use case
   (NOT the library default, to preserve numpy-CPU portability). This converts a banked capability into a shipped
   one and surfaces the honest 2-noun-scope + GPU cost for the owner. **This is the cleanest immediate win.**
2. **THEN: R3 — extend the multi-cue competition to richer inputs (attributed adj+noun + multi-clause).** Reward HIGH
   / effort MEDIUM. This is where the parser's bottleneck has genuinely **moved**: robustness is solved; *coverage*
   of real conversational inputs (3+-noun, adjective attribution, embedded clauses) is the remaining gap, and the
   pieces exist (`enable_attributed`, `enable_multiframe`, the embedded-clause de-risk). Composing them with the
   multi-cue competition is the natural follow-on that keeps extending the conversational primary.
3. **OPTIONAL polish: R2 (WTA object-front calibration study) + R4 (neuralize the feature lexicons).** Reward MODERATE
   / effort MEDIUM. R2 is a genuine operating-point study (selective-inhibition gain vs pool size) that would lift the
   end-to-end ceiling on the hardest items; R4 closes the last cognitive host scaffold (the near-binary version is
   learnable without the dendritic arc). Neither blocks (1) or (2); the install path is already the robust deployment.
4. **OR: proceed to the next roadmap tier.** Per `project_post_conversational_roadmap_tiers`, after the conversational
   loose ends comes **Tier 2 (TRUE ONE BRAIN — the limbic→composer wire + the persistent integrated spiking loop)**.
   If the owner judges Phase-1 robustness "banked behind a flag" sufficient for now, the higher-leverage frontier may
   be Tier 2 rather than the Phase-1 polish. **This is an owner fork** (deepen the conversational primary vs advance
   the one-brain integration); both are legitimate and I cannot resolve it from context.

**My single recommendation if forced to one:** do (1) now (flip the imperfect-English demo to the validated
multi-cue parser — it is the owner's stated primary, already validated, and one decision away from live), then (2)
(extend coverage to attributed + multi-clause inputs, the moved bottleneck). Defer (3) as polish and treat (4) as the
owner's call.

---

## 8. Provenance (verified this session)
- **Project machinery (read + verified):** `research/runners/brain_conversational_agent.py:28` (`BridgeParser`:
  position-only, `_GT` map, `_train` Hebbian loop — confirmed single-cue); `:154-358` (the `enable_multicue_competition`
  / `enable_case_competition` flags, default-OFF, lazy, precedence, byte-identical `hear()` routing — confirmed);
  `research/runners/multicue_role_parser.py` (`MultiCueRoleParser`: install-path validities, frozen plasticity,
  `parse_decisive` moat gate — confirmed); `research/runners/biased_competition_buffer.py` (the #2 Wong-Wang /
  Rutishauser WTA + `ANIMACY`/`VERB_SELECTS` lexicons — confirmed the reuse substrate); `research/runners/case_aware_role_parser.py`
  (Phase-2 drop-in). **CI guard RUN: `tests/test_multicue_competition_agent.py` 7/7 passed (30.6 s, numpy).**
- **The DONE-state findings (read + cross-checked):** `2026-06-19-multicue-competition-parser-scoping.md`,
  `-multicue-competition-derisk.md` (numpy GO 6/6), `-multicue-competition-spiking-derisk.md` (spiking GO 5/6 +
  the Hebbian-vs-three-factor finding), `-multicue-learning-firm-and-neural-reward.md` (learning firmed 6/6 + spiking
  SNc RPE), `-multicue-competition-agent-wirein.md` (production wire-in DONE), `-case-cue-crosslanguage-derisk.md`
  (Phase-2 GO + dissociation 6/6), `-case-cue-crosslanguage-agent-wirein.md` (Phase-2 wire-in DONE),
  `2026-06-20-shortcut-burndown-status.md` (confirms the gate-to-capability is cleared; the primary "teed up").
- **Catalog (verified verbatim, `sim-catalog/references/feature-catalog.md`):** **G.18** (line 2850 — "integrates
  *any* evidence weighted by reliability... additively" = reliability-weighted cue integration); **G.12** (line 2774,
  Broca — "comprehension of grammatically complex (non-canonical) sentences"); G.16/G.17 (drift-diffusion/LIP
  accumulator), G.06/G.08 (PFC WM attractor).
- **Literature (cited in the de-risks; consistent with the catalog):** Competition Model — Bates & MacWhinney
  1982/1989; MacWhinney-Bates-Kliegl 1984 (English=order, German/Italian=agreement). Biased competition — Desimone &
  Duncan 1995; Wong & Wang 2006 (the project's WTA core). Good-enough/NVN — Ferreira 2003. Dopamine-as-RPE — Schultz
  1998. Biologically-plausible-parser precedent + the single-cue gap — NEMO / Assembly Calculus
  (Papadimitriou-Vempala-Dabagia-Mitropolsky; object-initial weakness; no animacy/case integration). Point-neuron
  limit (why it does NOT apply to cue competition) — Mikulasch & Priesemann 2021.

---

**Gate confirmation:** stayed on `main`, READ-ONLY (the only execution was running the existing CI guard
`tests/test_multicue_competition_agent.py` to verify the claimed DONE state — no code written/edited, no experiment
designed/run). Load-bearing claims verified against the actual `BridgeParser`/agent/parser source + the actual
catalog text + the CI guard, not assumed.
