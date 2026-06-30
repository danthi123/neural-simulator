# Tier 1 (conversational loose ends) — close-out assessment (READ-ONLY, 2026-06-30)

**Type:** read-only status assessment (no code edits, no GPU run), per the standing deep-research-first practice.
**Scope:** the three Tier-1 items of the owner-accepted post-conversational roadmap (memory
`project_post_conversational_roadmap_tiers`, 2026-06-19): (1) wire the proven Phase-2 cross-language case-cue
mechanism into the PRODUCTION conversational agent; (2) firm the seed-variable on-substrate cue-validity LEARNING +
neuralize its host reward scaffold; (3) minor polish — #2 content-graded bias (the seed-100 abstention case) and #3
embedded-clause 0.02 gap.
**Method:** read the code (`brain_conversational_agent.py`, `multi_turn_agent.py`, `biased_competition_buffer.py`,
`case_aware_role_parser.py`, `multicue_role_parser.py`, `g11_bg_runner.py`), the 2026-06-19/20 findings, the
2026-06-27 burndown-arc findings, the git log, and `AUTONOMOUS_STATE.md` (CYCLE 270–281, 314–315, 5548).

---

## TL;DR — per-item status table

| # | Tier-1 item | Mechanism (de-risk) | Production wiring | Net verdict |
|---|---|---|---|---|
| **1** | Cross-language **case-cue** + Phase-1 **multicue** role-assignment wired into the production agent | **DONE** (case 5/6 seeds + dissociation 6/6; multicue install 5/6; all anti-cheats; moat 0/240) | **DONE** — `BrainConversationalAgent(enable_case_competition=…/enable_multicue_competition=…)`, default-OFF flags, `hear()` routes through them; CI-guarded | **DONE** (deliberately **opt-in**, not default-on — documented un-defaultable carve-out, not a loose end) |
| **2** | Firm the seed-variable cue-validity **LEARNING** + **neuralize the reward** (host→spiking RPE) | **DONE** — learned-validity **signature 6/6** (host AND spiking-SNc RPE); reward now neural; moat 0/6 | de-risk runner only (production deploys the robust **install path**, by design) | **DONE** (the genuine residual is a *characterized* tiny-scale WTA readout operating-point boundary on `object_front`, **not** a learning failure — install path is the robust headline) |
| **3a** | Polish **#2** — content-**graded** bias (close the seed-100 extreme-asymmetry abstention) | **DONE** — GO 6/6, bias-lesion-breaks 6/6 (load-bearing), moat 6/6 | **OPEN** — production `MultiTurnAgent` still uses the **fixed** 2500 pA bias (`_resolve_biased`); graded version lives only in the de-risk runner | **PARTIAL** (de-risk closed; production wire-in is the recommended-but-undone follow-on) |
| **3b** | Polish **#3** — close the embedded-clause **0.02 gap** (depth-1) | **DONE** — redundancy lever lifts both 0.88 seeds to **1.000**, GO 6/6, controls collapse, moat 6/6 | **OPEN** — no `parse_nested`/`hear_nested` on any production agent; lives only in the de-risk runner | **PARTIAL** (de-risk closed; the `parse_nested` production opt-in is the recommended-but-undone follow-on) |

**Bottom line:** every Tier-1 item is **mechanistically de-risked GO** and item 1 is **production-wired**. Two
items remain OPEN *only at the production-wiring level* — both are trivial, low-risk wire-ins (an `enable_*` flag in
the same "bank behind a flag" pattern already used 4× this arc), and both are **CPU-only** to land + guard. The
"firm the learning / neuralize the reward" residual is a genuine but **characterized** substrate boundary that
should NOT be chased (naive levers were tried and made it worse); the robust deployment is the install path, which
is already what ships. **The 2026-06-27 spiking-substrate burndown arc did NOT touch any Tier-1 item** (it worked a
different conversational layer — entity instances, transitive/analogy, tense/aspect — and its own deep-research gate
at CYCLE 315 explicitly *confirmed* the multicue/case parser "was ALREADY built 2026-06-19" and prevented a
redundant rebuild). So nothing below has been silently superseded.

---

## Item 1 — Cross-language case-cue + multicue wire-in into the PRODUCTION agent

### Verdict: **DONE** (wired into `BrainConversationalAgent`; deliberately opt-in)

**Which agent + exactly where.** The production wire-in is on **`BrainConversationalAgent`** (the production
who/what/yes-no/describe/reason turn agent), NOT just a standalone de-risk runner:

- Constructor flags (`research/runners/brain_conversational_agent.py:179-180`):
  `enable_multicue_competition=False, multicue_verbs=None` and
  `enable_case_competition=False, case_verbs=None, case_lexicon=None`.
- `hear()` (the production turn entry point) routes the agent/patient decision through the spiking competition
  (`brain_conversational_agent.py:508-511`): if `enable_multicue_competition` → `hear_multicue()`; elif
  `enable_case_competition` → `hear_case()` (multicue takes precedence if both set — they are alternative
  comprehension front-ends).
- Lazy-built drop-ins: `MultiCueRoleParser` (`research/runners/multicue_role_parser.py`, built at
  `brain_conversational_agent.py:465-468`) and `CaseAwareRoleParser`
  (`research/runners/case_aware_role_parser.py`, built at `:433-437`). Both wrap the validated
  `SpikingRoleCompetition` (Wong-Wang/Rutishauser WTA over thematic roles + plastic cue→role projections, real
  `cp_firing_states`) with the validated **install-path** cue validities; `CaseAwareRoleParser` adds the 5th `case`
  cue (が→+1 agent / を→−1 patient).
- CI guards (CPU/numpy-runnable, use the rf composer with an explicit vocab so the `denoise64` cache is not
  needed): `tests/test_multicue_competition_agent.py` (7 tests) and `tests/test_case_cue_crosslanguage_agent.py`
  (9 tests). Flag-OFF byte-identity asserted by `tests/test_brain_conversational_agent.py` (passes verbatim with
  the flags present).

**The load-bearing wins (both reproduced in the production path).**
- Multicue (robust English): object-fronted "apple eat dog" → the default position-only parser stores it
  **backwards** (agent=apple); flag-ON the agent answers `who_does('eat','apple')=='dog'` and returns None for the
  inverted answer. Battery: MULTICUE **0.950** vs POSITION-ONLY **0.225** (object-front 0.950 vs 0.000), moat 0
  breaches.
- Case (free word order): "wolf wo dog ga chase" (object-fronted, case-marked) → flag-ON resolves agent=dog,
  patient=wolf (case overrides surface order; free-order 40/40 = 1.000), and the cross-linguistic dissociation
  holds in production (case DECIDES on the case-toy, is SILENT on English, flips abstain↔decide by particle
  presence).

**Evidence:** `2026-06-19-multicue-competition-agent-wirein.md`, `2026-06-19-case-cue-crosslanguage-agent-wirein.md`
(both verdict **DONE**); mechanism de-risks `2026-06-19-multicue-competition-spiking-derisk.md`,
`2026-06-19-case-cue-crosslanguage-derisk.md`. Commits `e571ad49`/`711a9b95`/`447e84b8`/`97436525` (multicue) and
`6537cb4e`/`e86ce322`/`a796792e`/`063a94b5` (case). AUTONOMOUS_STATE CYCLE 272 (controller-verified 16 passed / 5
GPU-skipped). Burndown re-confirmation: AUTONOMOUS_STATE CYCLE 315 + commits `a68b20c8`/`5865fe9c` (the burndown's
own deep-research gate found "the robust multi-cue parser (Phase 1+2) was ALREADY built 2026-06-19 … moat 0/240").

### Why this is DONE-but-opt-in, not a loose end

The roadmap item asked to "wire the proven Phase-2 mechanism into the PRODUCTION conversational agent." That is done
— the agent constructs and routes through it behind a flag, exactly the "add → validate-behind-flag-byte-identical
→ deliberately flip" staging pattern this arc used for the other 4 capabilities. It is **deliberately NOT flipped
default-ON**, for documented reasons (`2026-06-19-default-on-consolidation.md`, the "stays OPT-IN" table):

1. it requires a hand-curated verb lexicon (animacy + selectional restrictions, plus the case-particle lexicon)
   that the agent's plain `{word: code}` vocab cannot supply; and
2. it **replaces** rather than composes with the position/frame parser path.

So "fully close" here means **leave it as the validated opt-in it already is** (the honest carve-out), OR — if a
production-default flip is desired — that is gated on giving the agent a way to supply the verb/animacy/case lexicons
(a learned lexical-feature map is the named BRAIN-BASED follow-on; that is a Tier-2/Tier-4-flavored build, not a
Tier-1 loose end). **No action recommended for Tier-1 close-out beyond confirming the carve-out is intended.**

---

## Item 2 — Firm the cue-validity LEARNING + neuralize the reward

### Verdict: **DONE** (learning robust 6/6; reward neuralized 6/6; residual is a characterized boundary, not a hole)

**The learning is robust.** The prior end-to-end strict GO was seed-variable (3/6→4/6); re-examination found the
**learned-weight SIGNATURE was correct on ALL 6 seeds** (position driven materially below the semantic cues), i.e.
the LEARNING was not failing. The 3/6 was *partly the test* (two real test-validity holes: the NO-LEARNING control
was uniform-init so the semantic cues alone carried the degraded battery; the PERMUTE control's position weight ran
away to ~6082 via three-factor positive feedback) and *partly* a real readout boundary. Both test holes were fixed
(naive-canonical-prior no-learn control + an inert-on-real-data per-weight cap + a `--hard-battery` validity-stress
mode):
- learned-validity **signature correct 6/6**, position-only baseline collapses 6/6 (battery is valid), permute
  control now collapses 6/6 (cap fixed the runaway), no-confab **moat 0 breaches**;
- learned end-to-end role accuracy **≥0.80 on 5/6** (soft battery) / 4/6 (hard battery).

**The reward is neuralized (the TRUE-ONE-BRAIN spike-ification).** Part 1's learner already spike-measured the cue
ELIGIBILITY; its remaining host scaffold was the reward scalar `err = target − tanh(pred·8)`. Part 2 replaces it
with a **spiking SNc dopamine pool** (`IZH2007_DOPAMINE`, `n_snc=40`) driven by
`I_snc = tonic + reward_gain·target − value_gain·pred`, whose windowed firing-rate deviation from tonic IS the RPE
`δ = r − V` (reusing the nav g11 `spiking_snc` pattern verbatim in spirit). The dopamine firing **recovers the
validity signature on 6/6 seeds on real spikes** (position 59–89% below the semantic weight), end-to-end comparable
to host (4/6 vs 5/6 ≥0.80), **moat 0/6**. The probe confirms the spiking RPE sign-tracks the host `err` on all 8
target/pred cases, graded magnitude (under-predict → +0.76, matched → ≈0, over-predict → −0.93).

**The genuine residual — characterized, NOT a hole to chase.** The end-to-end per-seed variance on the hardest
items (`object_front`) is a **tiny-scale Wong-Wang WTA operating-point/calibration boundary**, not a learning
failure (the signature is correct on every seed) and not a reward-precision wall (host and spiking-RPE give the
identical 6/6 signature). Naive robustness levers were tried and rejected: more epochs/read-steps (no change),
population redundancy n_sel 24→48→64 (made it WORSE non-monotonically — a mis-calibrated WTA). Re-calibrating the
WTA selective-inhibition gain vs pool size is a genuine operating-point study (flagged, not escalated, per the
standing "don't escalate into a config search" guidance).

**Evidence:** `2026-06-19-multicue-learning-firm-and-neural-reward.md` (Parts 1+2, verdict GO on the brain-based
claim). Commits `37afbcd6`/`0c422799`/`488ae1f5`. AUTONOMOUS_STATE CYCLE 273 (controller-verified). The nav SnC
pattern reused is real: `research/runners/g11_bg_runner.py:3536` (`I_snc = snc_tonic_pa + snc_reward_gain·max(0,r)
− snc_value_gain·V`), `spiking_reward_us` afferent at `:527`/`:2153`/`:2808`.

### Is the reward still a host scaffold?

**No — it is neuralized.** Both the eligibility AND the reward (`r − V` subtraction + RPE magnitude) are now
computed by neuron firing (the SNc pool). The ONLY host scaffold left is the **gold→target lookup**, which is the
**legitimate teaching/environment boundary** — exactly as the nav `reward_us` rides on a host-supplied perceived
reward (memory `feedback_brain_based_only_standard`: host code is legitimate for the environment + body). So there
is no further neuralization owed here for Tier-1 purposes. (The fully-spiking *live-SNc-driven* deployment — wiring
the SNc pool into the merged-bridge step loop rather than the de-risk harness — is a Tier-2 deployment follow-on,
overlapping Tier-2 #6 limbic→composer; not a Tier-1 loose end.)

**The nav dopamine/SnC neuralization pattern is reusable here, and was reused.** The pattern is: drive a dedicated
DA pool with `I = tonic + reward_gain·r − value_gain·V`; read the RPE as the firing-rate deviation from tonic; gate
the third (reward) factor of the learning rule with that deviation. Item 2 Part 2 already imports this pattern from
g11. The only remaining reuse would be the live in-loop deployment, which depends on the limbic core being on the
merged bridge (Tier-2 #6).

---

## Item 3 — minor polish (#2 content-graded bias, #3 embedded-clause gap)

### #2 content-graded bias (seed-100 abstention case) — **PARTIAL** (de-risk DONE; production wire-in OPEN)

**De-risk: DONE.** A content-graded bias (scale the steer by the favored referent's intrinsic accumulator deficit
vs its rival, `bias_pA = min(cap, base·(1 + gain·deficit/ref))`, injected only into the favored sel pool) closes
the one pre-registered miss (seed 100, extreme intrinsic asymmetry): GO-arm **6/6** (was 5/6), bias-LESION still
breaks **6/6** (`graded(0)=0` → reverts to intrinsic winner → load-bearing, NOT a relabelled global gain),
no-confab **moat 6/6**, recency + salience-4× baselines still FAIL 6/6, 3-referent scale 6/6. Evidence:
`2026-06-19-multireferent-graded-bias-polish.md` (verdict GO); commits `19edb54e`/`9abb8fed`; AUTONOMOUS_STATE
CYCLE 275.

**Production wiring: OPEN.** The production agent that resolves multi-referent pronouns is **`MultiTurnAgent`**
(`research/runners/multi_turn_agent.py`), and `enable_biased_competition` is already **default-ON** there
(`multi_turn_agent.py:49`, flipped by commit `41389c93`). **BUT** the production path uses the **FIXED** bias, not
the graded one: `_resolve_biased` (`multi_turn_agent.py:164-180`) calls `content_bias_target(held, query_verb)` then
`self.bcw.read(window=…, bias_concept=fav, bias_pA=self._bc_bias_pA)` with `self._bc_bias_pA = 2500.0`
(`multi_turn_agent.py:50,99`) — there is **no unbiased deficit-probe-read and no deficit-scaled magnitude**. The
graded mechanism lives only in the de-risk runner `_phaseB_biased_competition_graded_derisk.py`. The polish's own
recommendation (`2026-06-19-multireferent-graded-bias-polish.md` §6) is to "update the production `MultiTurnAgent`
biased-competition to the content-graded bias (a small follow-on)" — that wire-in is **not done**. So the seed-100
extreme-asymmetry case still **abstains** (not confabulates — the moat is intact) in production today.

### #3 embedded-clause 0.02 gap (depth-1) — **PARTIAL** (de-risk DONE; production wire-in OPEN)

**De-risk: DONE.** The population-redundancy read-out lever (`RedundantEmbeddedReadout` — R independent
`RFPhasorComposer` replicas with distinct codebooks, per-slot majority vote) lifts both 0.88 marginal seeds (43,
101) to **1.000** on GPU 6-seed (all 6 at 1.000) and CPU, every anti-cheat still collapses (no-seg 0.500, scramble
0.000, head-attach 0.000), no-confab **moat 6/6**. `--readout-redundancy 1` is byte-identical; R=3 recommended as
the de-risk default. Evidence: `2026-06-19-embedded-clause-redundancy-polish.md` (verdict GO),
`2026-06-19-embedded-clause-parse-derisk.md` (the prior near-GO mean 0.951); commits `2e2faf94`/`84e3c9a3`/
`7699490b`; AUTONOMOUS_STATE CYCLE 281 ("TIER 1 FULLY CLOSED" — at the de-risk level).

**Production wiring: OPEN.** There is **no `parse_nested`/`hear_nested`/`EmbeddedClauseParser` on any production
agent** (grep over `research/runners/`: `EmbeddedClauseParser`/`parse_nested` appear ONLY in
`_phaseB_embedded_clause_parse_derisk.py`; `brain_conversational_agent.py` has none). The existing
`hear_clause_fact` (`brain_conversational_agent.py:264-266`) still HOST-constructs the `Clause` operand ("nested
input parsing is future work, so the clause is provided structurally here"). The polish's recommendation
(`2026-06-19-embedded-clause-redundancy-polish.md` §"Recommendation" / the parse de-risk §"Recommended production
wire-in") is the "production `parse_nested` opt-in (mirroring `enable_attributed`/`enable_multiframe`)" — that is
explicitly listed as **deferred**, and is not done.

### Reconciling the "TIER 1 FULLY CLOSED" claim (CYCLE 281) with the OPEN production wirings

CYCLE 281 and commit `c1567cbf` declare "TIER 1 (conversational loose ends) FULLY CLOSED." That claim is true **at
the de-risk/mechanism level** — all four sub-items (item-1 wire-in, item-2 firm+neuralize, #2 graded bias, #3
embedded redundancy) are validated GO. What it does NOT mean is that #2 and #3 were **wired into production**: the
graded bias and the `parse_nested` parser both remain runner-only, by the de-risks' own "recommended follow-on"
framing. CYCLE 273 was more precise ("item 3 = MINOR characterized backlog … low-value polish on working
capabilities; deferred unless owner wants"). For a controller asked to **fully close** these, the two production
wire-ins are the remaining concrete work.

---

## Did the 2026-06-27 spiking-substrate burndown arc close any Tier-1 item? — No

The burndown arc (CYCLE 672–691, `2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md`,
`2026-06-27-conversation-depth-brain-based-audit-and-burndown.md`) worked a **different conversational layer** —
entity instances / file-cards, transitive & ordinal maps, factored-relation analogy, chain-of-thought, tense/aspect,
common-ground — plus nav close-out (R1a spiking default, R2/R5 value). It did **not** touch multicue/case wire-in,
cue-validity learning, the graded bias, or embedded-clause parsing. The only Tier-1-adjacent mention is bucket-C item
6 ("Tier-1 which-X candidate SCORING → the de-risked spiking biased-competition WTA `biased_competition_buffer.py`")
— a *future* burndown target, not done, and it points at the SAME `biased_competition_buffer.py` the graded-bias
polish recommends upgrading. And the burndown's own deep-research gate (AUTONOMOUS_STATE CYCLE 315, commits
`a68b20c8`/`5865fe9c`) explicitly found the multicue/case parser "was ALREADY built 2026-06-19" and *prevented a
redundant rebuild*. **⇒ nothing in this assessment has been superseded; nothing here is a redo.**

---

## Cheapest biology-grounded close path for each still-OPEN item

Both OPEN items are **production wire-ins of an already-validated GO mechanism** — i.e. the "(i) CLOSEABLE-CHEAP"
category in the burndown taxonomy (a deployment flip / wiring, not research). Neither needs a new mechanism, a new
de-risk, or any `sim/` edit. The standing research gate does NOT fire (these compose already-de-risked machinery;
they are mechanical wirings, per the CLAUDE.md "Does NOT fire" list).

### Close #2 (content-graded bias) into production `MultiTurnAgent` — CPU, ~1 short session

- **Mechanism (reusable as-is):** the de-risk's probe-read + deficit-scaling already lives in
  `_phaseB_biased_competition_graded_derisk.py` and reuses the production `BiasedCompetitionContextBuffer` +
  `content_bias_target` + `resolve_referent` verbatim. The wire-in promotes the deficit-graded magnitude into
  `MultiTurnAgent._resolve_biased` (replace the single `read(bias_pA=self._bc_bias_pA)` with: a non-destructive
  unbiased probe read → measure `fav_sel`/`rival_sel` → `deficit = max(0, rival_sel − fav_sel)` →
  `bias_pA = min(cap, base·(1 + gain·deficit/ref))` → the biased read), behind an additive default-preserving knob
  (e.g. `biased_competition_graded=False` so `False` = byte-identical to today's fixed 2500 pA path). The de-risk's
  validated constants (base 2500, gain 1.0, ref 0.20, cap 8000) install directly.
- **Cheap-first de-risk:** none needed beyond re-running the existing graded de-risk through the agent path — the
  6-seed GO already exists. A 1-seed agent-level capability test (the seed-100 `roll` cases resolve to ball via the
  agent's `what_does('it','roll')`) confirms the wiring.
- **Anti-cheats (carry forward verbatim):** bias-LESION must still break resolution 6/6 (proves graded ≠ global
  gain); no-confab moat 6/6 (empty WM → abstain, content-silent → abstain); recency + salience-4× baselines still
  FAIL; `graded=False` byte-identical (`tests/test_multireferent_biased_competition.py` +
  `tests/test_multi_turn_agent.py` pass verbatim).
- **Reusable machinery:** `biased_competition_buffer.py` (`read` already supports a `bias_pA` argument and a
  non-destructive re-presented read; the probe read is the same `read()` path with `bias_pA=0`).
- **Honest residual carried forward (NOT closed by this):** the **all-compatible-referent** case (two
  same-animacy/number candidates, agreement silent) still abstains — that needs finer cues composed on top of the
  competition (a separate, named, non-Tier-1 follow-on). The content scoring (`content_bias_target`) remains a
  flagged host scaffold for a learned synaptic feature-compatibility map (BRAIN-BASED follow-on, also non-Tier-1).

### Close #3 (embedded-clause `parse_nested`) into the production agent — CPU to land + guard; GPU for the 6-seed re-confirm only

- **Mechanism (reusable as-is):** `EmbeddedClauseParser.parse_nested` + `RedundantEmbeddedReadout` already exist in
  `_phaseB_embedded_clause_parse_derisk.py` and reuse `AttributedBridgeParser`, `OrderedPositionWM`,
  `RFPhasorComposer`/`Clause`/`_decode_clause` verbatim (NO `sim/` edit). The wire-in adds
  `BrainConversationalAgent.hear_nested(flat_sentence, verbs)` behind an additive default-OFF `enable_embedded_clause`
  flag (mirroring `enable_attributed`/`enable_multiframe`), replacing the host-constructed `Clause` in
  `hear_clause_fact` with the parsed one, with `--readout-redundancy 3` as the default.
- **Cheap-first de-risk:** none needed beyond an agent-level capability test — the parser's 6-seed GO at 1.000
  already exists. The CPU smoke (subject- + object-relatives round-trip through `hear_nested` → `query_patient`)
  confirms the wiring; the GPU 6-seed re-confirm is only to re-bless the redundancy default on the agent path.
- **Anti-cheats (carry forward verbatim):** the NO-SEGMENTATION baseline must FAIL (0.500 — the load-bearing
  control, decisive because object-relatives can't be segmented by a flat reader); scramble → 0.000; permuted-head
  → 0.000; held-out leakage 0; no-confab moat intact (garbled/unknown/never-stored → abstain); `enable=False`
  byte-identical.
- **Reusable machinery:** `attributed_parser.py`, `ordered_position_wm.py`, `rf_phasor_composer.py`.
- **Honest residual carried forward (NOT closed by this):** the **fully-neural relativizer/verb detector** (the
  closed-class lexical tag is the legitimate environment/lexicon front end, same as `FrameParser`); a transitive
  matrix clause with its own object; and **depth-2 center-embedding** = the EXPECTED catalog G.12 boundary (the
  human ~2-level limit; an honest NEGATIVE there is the deliverable, NOT a defect to brute-force). All non-Tier-1.

### Item 1 — no close action; confirm the carve-out

Item 1 is DONE. The only open question is whether the owner wants a **production-default flip** (vs the current
deliberate opt-in). That flip is blocked by the un-defaultable verb/animacy/case lexicon (the agent's plain vocab
can't supply it) and the parser-replacement design — so the cheap close is **"none, it is correctly opt-in."** A
default-on flip would require a learned lexical-feature map (the named BRAIN-BASED follow-on), which is a
Tier-2/Tier-4-flavored build, not a Tier-1 loose end.

### Item 2 — no close action; the residual is a characterized boundary

Item 2 is DONE (learning robust 6/6, reward neuralized 6/6, moat 0/6). The genuine residual (tiny-scale WTA
`object_front` readout operating-point variance) is **characterized, not a learning hole**, and was shown
*resistant to naive levers* (more epochs/redundancy made it WORSE). Per the standing guidance it should be **flagged,
not chased** in Tier 1. The robust deployment is the **install path**, which is already what production uses. The
only Tier-2-overlapping follow-on is the live-SNc-in-loop deployment (depends on the limbic core on the merged
bridge — Tier-2 #6).

---

## Recommended close-out order

The two genuinely-open wire-ins are independent, cheap, CPU-landable, and follow a pattern executed 4× this arc.
Recommended order (cheapest → most valuable):

1. **#2 content-graded bias → `MultiTurnAgent`** (CPU; ~1 short session). Smallest diff (swap one `read()` call for
   probe-read + deficit-scale behind a default-preserving knob), reuses the already-extracted production buffer +
   helpers, lifts the production multi-referent path from "abstain on seed-100-style extreme asymmetry" to "resolve
   correctly" with the lesion-load-bearing + moat guarantees intact. CI guard: extend
   `tests/test_multireferent_biased_competition.py`.
2. **#3 `parse_nested` opt-in → `BrainConversationalAgent.hear_nested`** (CPU to land + guard; one GPU 6-seed
   re-confirm of the redundancy default). Bigger surface (a new `hear_nested` path + an `enable_embedded_clause`
   flag), but still pure reuse-by-import, NO `sim/` edit; replaces the last host-constructed `Clause` with a parsed
   one. CI guard: a new agent-level capability test mirroring `test_multicue_competition_agent.py`.
3. **Item 1 carve-out + item 2 residual — confirm-and-document, no build.** Record that the case/multicue wire-in is
   correctly opt-in (default-flip blocked on the un-defaultable lexicon → a learned lexical-feature map is the
   non-Tier-1 follow-on), and that the cue-validity end-to-end residual is a characterized WTA operating-point
   boundary (install path is the robust production headline; live-SNc-in-loop is the Tier-2 #6 overlap). Then **Tier
   1 is fully closed in fact** (de-risks GO + the two CPU wire-ins landed + the two boundaries honestly documented),
   and the path is clear to Tier 2 (#6 limbic→composer + the persistent integrated spiking loop).

**GPU vs CPU summary:**

| Close action | CPU / GPU |
|---|---|
| #2 graded-bias wire-in + CI guard | **CPU** (numpy; the de-risk is CPU) |
| #3 `parse_nested` wire-in + CPU smoke + CI guard | **CPU** to land + guard |
| #3 6-seed redundancy re-confirm on the agent path | **GPU** (one A/B re-bless, optional — the de-risk's 6-seed GPU GO already exists) |
| Item 1 / Item 2 confirm-and-document | **CPU** (no run) |

---

## Files cited (all absolute under `E:\Documents\Projects\sim\`)

**Code (current state):**
- `research/runners/brain_conversational_agent.py` — flags `:179-180`; `hear()` routing `:508-511`; case parser
  build `:433-437` + `hear_case` `:441`; multicue parser build `:465-468` + `hear_multicue` `:471`; host-constructed
  clause `:264-266`.
- `research/runners/multi_turn_agent.py` — `enable_biased_competition=True` default `:49`, fixed `bias_pA=2500.0`
  `:50,99`; `_resolve_biased` (fixed, not graded) `:164-180`; `_resolve` routing `:182-192`.
- `research/runners/biased_competition_buffer.py` — `read(bias_pA=…)` `:274`, `resolve_referent` `:328` (no graded
  path present).
- `research/runners/case_aware_role_parser.py`, `research/runners/multicue_role_parser.py` — the production drop-ins.
- `research/runners/g11_bg_runner.py` — nav SnC pattern `:3536` (`I_snc`), `spiking_reward_us` `:527`/`:2153`/`:2808`.
- `research/runners/_phaseB_embedded_clause_parse_derisk.py` — the ONLY home of `EmbeddedClauseParser`/`parse_nested`/
  `RedundantEmbeddedReadout` (not in production).
- `research/runners/_phaseB_biased_competition_graded_derisk.py` — the ONLY home of the graded bias (not in
  production).

**Findings:**
- `2026-06-19-multicue-competition-agent-wirein.md`, `2026-06-19-case-cue-crosslanguage-agent-wirein.md` (item 1
  wire-in DONE).
- `2026-06-19-multicue-learning-firm-and-neural-reward.md` (item 2 DONE — learning 6/6, spiking-RPE 6/6).
- `2026-06-19-multireferent-graded-bias-polish.md` (#2 de-risk GO 6/6).
- `2026-06-19-embedded-clause-redundancy-polish.md`, `2026-06-19-embedded-clause-parse-derisk.md` (#3 de-risk GO
  6/6 @ 1.000).
- `2026-06-19-default-on-consolidation.md` (the 4-capability default-on flip + the multicue/case opt-in carve-out).
- `2026-06-19-multireferent-integration-multiturnagent.md` (the fixed-bias `enable_biased_competition` integration).
- `2026-06-27-comprehensive-shortcut-inventory-burndown-plan.md`,
  `2026-06-27-conversation-depth-brain-based-audit-and-burndown.md` (the burndown arc — did NOT touch Tier 1).
- `research/findings/AUTONOMOUS_STATE.md` CYCLE 270–281 (the Tier-1 close arc), 314–315 (burndown gate re-confirm),
  line 5548 (this task's dispatch).

**Cited science:** Bates & MacWhinney 1982/1989 (Competition Model + cross-linguistic cue validity); Desimone &
Duncan 1995, Wong & Wang 2006, Rutishauser-Douglas-Slotine 2011 (biased-competition WTA + α<1 stability); Schultz
1998 (dopamine RPE); Lisman-Grace (VTA-hippo, DA stabilizes a trace); Chomsky-Miller (center-embedding ~2-level
limit, catalog G.12).
