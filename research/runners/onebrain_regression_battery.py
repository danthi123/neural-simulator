"""The SHIPPED-FACULTY REGRESSION BATTERY — the cross-faculty no-regression instrument the one-brain INTEGRATION
program needs and that no per-faculty flip-verify has ever had.

THE GAP (integration program, Phase 1, item 2). Every existing flip-verify's ARM C ("no regression") checks only ITS
OWN faculty's fixed items. NOTHING asserts that flipping flag X does not silently break one of the OTHER ~29 default-ON
faculties on the roster (the seam-taxonomy killers — a MergeConflict is NOT raised; the union accepts a default and a
faculty dies quietly). This battery is that missing test: given a flag flipped ON-vs-OFF, it runs a representative
deterministic probe for EACH default-ON faculty through the REAL `webapp.server.brain_chat`, and asserts each still
DECIDES identically — or reports exactly which regressed. Every future merge/flip in the program gates on it.

HOW. A small set of deterministic PROBE TURNS is run through a fresh brain in the flag-ON arm and again in the flag-OFF
arm (each arm a FRESH subprocess build at the same seed, so the shared background-noise trajectory is identical between
arms — the reference `_xedge_flip_production_verify` model; comparing two sequential in-process arms would diverge on
noise). Each faculty is mapped to (the probe turn that exercises it, the DECISION fields it exposes in the response).
Only categorical DECISION variables are compared (booleans / labels / ids); continuous measurements (rates, levels,
margins, firing, mood, seconds, pA, ema_*) are EXCLUDED — a background process advances between reads, so the
reproducible claim is the DECISION, not the number (the same instrument choice ARM A makes: answer-string + decision
equality, never numeric margin identity).

OFF-ARM DISCIPLINE (2026-08-27 staleness class, gated by tools/gates/flip_offarm_staleness.py). The OFF arm ALWAYS sets
the flag EXPLICITLY to "0" — never `os.environ.pop` — so it stays OFF even after the flag's own default flips ON.

HONEST BOUNDARY. This is a REACHABILITY + DECISION-STABILITY instrument, not a proof of each faculty's correctness. A
faculty whose decision fields are None/absent on the probe set (it needs a trigger this set does not supply) is
reported as `not-exercised` (a THIN probe: counted, honest, not claimed as covered). The battery catches a flip that
changes a faculty's DECIDED output on a turn the set already drives; it cannot catch a regression a probe never
reaches.

THIN-PROBE LIFT (2026-09-02, the mechanical follow-on this paragraph named). Of the original 38 rows, 16 were driving
and 22 were thin. 20 of the 22 are now driving (comprehension-learned-animacy-cue/-verb-selects, affect-marker-
spiking-wta, confidence-forthcomingness, prospective-memory [formation half only], pragmatic-implicature [field-path
fix], surprise-monitor, metacog-monitor, worldmodel-forward, curiosity-followup, reconsolidation, episodic-memory,
discourse-register, open-ended-generation, discourse-planner, gnw-multistep-deliberation, self-initiated-utterance,
vision-identity-spiking-hmax, bg-action-selection, selective-attention-biased-competition) — each via either a new
PROBE_TURNS entry (a genuine trigger this set never supplied: a contradicting assertion, an expectation query, a
referential turn, a visual percept, a content-empty turn, a chase-form question, an idle/empty turn, a rich=True
override, ...) or a field-path fix (several thin rows pointed at a response key that never existed — e.g.
`pmem.armed`/`reconsolidation.revised` are not real keys; the real ones are `prospective.held`/
`reconsolidation.action`). 2 rows (gnw-deliberation, value-driven-choice) stay thin=True — both need a genuine
>=2-distinct-patient (agent,action) ambiguity that brain_chat-only conversational teaching CANNOT construct: the
default-ON reconsolidation organ rewrites a contradicting assertion IN PLACE rather than leaving two candidates
(verified live via the `contra` probe). See research/findings/2026-09-02-regression-battery-thin-probes-lifted.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


# ── the probe turns (deterministic; each populates several faculties' decision fields) ───────────────────────────
# (label, message, session, reset, percept, rich). A shared-session pair (hold -> held, or dr_a -> dr_b -> dr_c, or
# bc_a -> bc_b) sets discourse/WM state for the later read in that SAME group — PROBE_TURNS declaration order IS
# the execution order (the worker iterates it top-to-bottom), so a dependent group must stay declared consecutively
# in its dependency order. `percept` (None -> omitted from the request) and `rich` (default False -> the single-fact
# path) extend the original 4-tuple; every pre-existing row keeps `(None, False)` so it is byte-identical to before.
PROBE_TURNS = [
    ("well",     "the wolf bites the apple", "well", True,  None,   False),  # comprehensible transitive: recall/affect/da/provenance
    ("question", "what does the wolf bite",  "q",    True,  None,   False),  # a question -> None comprehension
    ("unknown",  "what is the capital of france", "u", True, None,  False),  # the no-confab MOAT -> abstain
    ("hold",     "the fox and the wolf walked in", "d", True, None, False),  # >=2 referents -> d6 multiref sets focus
    ("held",     "the wolf watches the owl", "d", False, None, False),       # same session: multiref/swap/anaphora on the held read
    ("scalar",   "some of the dogs ran",     "s",    True,  None,   False),  # a scalar-quantity turn -> pragmatic implicature
    ("open",     "what might a dog chase",   "o",    True,  None,   False),  # open-ended -> generation channel (single-fact path)
    # ── lifted 2026-09-02 (thin-probe follow-on to the Phase-1 battery): each turn below is a DRIVING trigger for a
    # faculty that the original 7 turns above never reached (see FAILURE_LOG-style rationale in the finding). Every
    # new turn uses its OWN fresh session so it cannot contaminate any other turn's per-session ChatBrain/state.
    ("confirm",  "the dog chase the cat",    "surp", True,  None,   False),  # (dog,chase) already known, SAME patient -> surprise CONFIRM (not surprised) + a genuine metacog confidence read on a real recall
    ("contra",   "the dog chase the fish",   "surp2", True, None,   False),  # (dog,chase) already known, DIFFERENT patient -> surprise CONTRADICT (surprised) + reconsolidation in-place rewrite
    ("expect_q", "what do you expect",       "wm",   True,  None,   False),  # E2 world-model QUERYABLE-expectation short-circuit (is_expectation_query)
    ("episodic", "did we discuss the dog",   "epi",  True,  None,   False),  # D5 referential-recall short-circuit (is_referential); honest not-in-memory on a fresh session
    ("vision",   "what do you see",          "vis",  True,  "bird", False),  # vision-identity: a visual query WITH a percept (BrainChatRequest.percept)
    ("bgdots",   "...",                      "bg",   True,  None,   False),  # a content-empty turn -> the BG SPEAK-vs-STAY-SILENT race is consulted
    ("dr_a",     "dog chase cat",            "dr",   True,  None,   False),  # D3 discourse fold #1 (bare 3-token clause, no connective) -> the CURRENT event
    ("dr_b",     "then bird chase worm",     "dr",   False, None,   False),  # D3 discourse fold #2 (connective-led) -> SHIFT: current->prev, new current
    ("dr_c",     "who was doing it before",  "dr",   False, None,   False),  # D3 before-query -> reads the held PREV slot (needs dr_a+dr_b run first, same session)
    ("chase",    "what does the dog chase all the way", "ch", True, None, False),  # gnw-multistep: an explicit chase-form question over the built-in dog->cat->fish chain
    ("selfinit", "",                         "si",   True,  None,   False),  # the idle/empty-turn self-initiated-utterance short-circuit (is_selfinit_trigger)
    ("animacy",  "the monkey carries the cup", "anim", True, None,  False),  # a hand-ANIMACY-table-OOV noun ('monkey', 19-noun table) covered only by the learned animacy cue; 'carry'/'cup' are hand-covered
    ("verbsel",  "the dog cleans the cup",   "vsel", True,  None,   False),  # a hand-VERB_SELECTS-table-OOV verb ('clean', 8-verb table) covered only by the learned verb-selects cue; 'dog'/'cup' are hand-covered
    ("emo",      "Wonderful! I am so happy and delighted, this is fantastic and amazing!", "emo", True, None, False),  # a strongly-affective turn -> a non-neutral mood LEVEL, so the affect-marker WTA actually has a marker to select (level==0 is an unconditional '' regardless of the flag)
    ("bc_a",     "the cat and the ball walked in", "bc", True, None, False),  # 2 held referents of OPPOSING animacy (cat=animate, ball=inanimate) for the selective-attention race
    ("bc_b",     "what does it eat",         "bc",   False, None,   False),  # a pronoun+verb-selectional query -> biased-competition content-bias resolves 'it' (needs bc_a run first, same session)
    ("rich_well","the wolf bites the apple", "richw", True, None,   True),   # the SAME well-formed transitive, explicitly rich=True -> the multi-sentence path (discourse-planner + confidence-forthcoming both live only there)
    ("rich_open","what might a dog chase",   "ropen", True, None,   True),   # the SAME open-ended prompt, explicitly rich=True -> the rich composer's own hypothesis-generation branch (resp['hypothesis'])
    ("pmem_form","remind me to feed the dog when the bird sings", "pmem", True, None, False),  # an intention-FORMATION utterance -> the prospective-memory latch (tests formation only, not the later cue-fire half)
]
_TURN_BY_LABEL = {t[0]: t for t in PROBE_TURNS}

# ── EXTRA turns reachable BY LABEL ONLY, deliberately NOT in the default PROBE_TURNS roster ───────────────────────
# Rationale: the full roster is iterated by run_regression_battery + every flip-verify harness that imports it, so a
# turn added to PROBE_TURNS runs (and on cupy would BTSP-write) in ALL of them. These turns are needed only by the
# load_bearing runner's driving remaps (LB_EPISODIC_DRIVE_PROBE / LB_DISCOURSE_REGISTER_DRIVE_PROBE /
# LB_NONCONTRADICTION_DRIVE_PROBE), so they live
# here — merged into _TURN_BY_LABEL (the worker resolves turns by label from it) but OUT of PROBE_TURNS -> the default
# roster, the regression battery, and every flip-verify harness are BYTE-IDENTICAL.
#   EPISODIC DRIVING PAIR ('epi2'): a STORE turn then a RECALL turn in ONE isolated session (declared store-first), so
#   the referential recall has a memory to COMPLETE (intact in_memory=True -> disclosure; lesion in_memory=False ->
#   "I don't recall") — the load-bearing recall path the lone-fresh-session `episodic` turn can NEVER exercise
#   (nothing stored -> intact reads not-in-memory, identical to the lesion).
#   DISCOURSE-REGISTER DRIVING TRIPLE ('dr2'): the default `dr_c` probe ('dog chase cat' -> 'then bird chase worm' ->
#   before?) has its correct before-agent be 'dog' = referents[0] = the register's identity index (0). The LESION
#   (_PrevSilencePairRegister.observe) forces the held prev slots to that SAME identity index, so intact ('dog' via the
#   learned RNN shift) and lesion ('dog' via forced-identity) return the IDENTICAL agent -> zero diff -> hollow, purely
#   an index collision. This triple SWAPS the roles so the correct before-agent is 'bird' = referents[3] != identity:
#   'bird chase worm' (bare clause -> CURRENT event) -> 'then dog chase cat' (connective -> SHIFT: bird/worm to prev,
#   dog/cat current) -> 'who was doing it before' -> intact reads the held PREV agent 'bird', lesion still forces 'dog'
#   -> discourse_register.agent FLIPS 'bird' vs 'dog' -> LOAD-BEARING. No forced env needed (the register defaults
#   spiking=True on any backend). See research/runners/load_bearing_fraction.py.
_EXTRA_TURNS = [
    ("epi_store", "the dog chase the cat",    "epi2", True,  None,   False),  # stores 'dog' (Hook B verified-SVO BTSP write; needs BRAIN_EPISODIC_STORE=1 or a cupy backend to execute)
    ("epi_recall","did we discuss the dog",   "epi2", False, None,   False),  # recalls 'dog' (Hook A dendritic-dAP completion) in the SAME session -> in_memory True intact / False lesion
    ("dr2_a",     "bird chase worm",          "dr2",  True,  None,   False),  # D3 fold #1 (bare clause, no connective) -> CURRENT event agent=bird (referents[3])
    ("dr2_b",     "then dog chase cat",       "dr2",  False, None,   False),  # D3 fold #2 (connective-led) -> SHIFT: bird/worm -> prev slots, dog/cat -> current
    ("dr2_c",     "who was doing it before",  "dr2",  False, None,   False),  # D3 before-query -> held PREV agent = 'bird' (index 3 != identity 0); lesion forces 'dog' (identity 0)
    # ── COMMON-GROUND DRIVING PAIR (label-only; NOT in PROBE_TURNS) ──────────────────────────────────────────────
    # The load_bearing runner's LB_CG_DRIVE_PROBE remap uses these so common-ground-drives exercises its actual
    # load-bearing axis: a REDUCE-vs-INTRODUCE flip on the RE-MENTION of an already-grounded referent. 'dog' is a
    # BUILD-TIME KB agent (brain_chat_tui.py facts: dog->chase->cat), so gnw_thought_swap._extract_topic finds it as
    # the grounded topic immediately -- no gate()-ordering / OOV problem (unlike 'wolf' in the lone default 'well'
    # probe). mention1 (first mention this session) -> intact & lesion both read UNGROUNDED -> decision=introduce, and
    # the organ then GROUNDS the slot (ignite + NMDA self-sustain). mention2 (same session 'cg2', re-mention) ->
    # was_grounded=True: the INTACT ledger's self-sustaining recurrence still holds the slot -> substrate reads
    # grounded -> decision=REDUCE; the BRAIN_CG_DRIVES_LESION=1 ledger built its recurrence at weight 0 (common_ground_
    # drives_chat.cg_drives_lesioned) so the slot decayed by read-time -> decision stays INTRODUCE -- a genuine
    # categorical flip on `common_ground_drives.decision`. The lone fresh-session first-mention 'well' probe can NEVER
    # construct this (nothing grounded -> intact==lesion==introduce). See research/runners/load_bearing_fraction.py.
    ("cg_mention1", "the dog runs fast",  "cg2", True,  None, False),   # first mention of 'dog' -> introduce; grounds the slot
    ("cg_mention2", "the dog runs again", "cg2", False, None, False),   # re-mention SAME session -> intact reduce / lesion introduce
    # ── NON-CONTRADICTION DRIVING turn (label-only; NOT in PROBE_TURNS) ──────────────────────────────────────────
    # LB_NONCONTRADICTION_DRIVE_PROBE: a single fresh-session turn that ASSERTS the NEGATED form of a fact the tiny-demo
    # brain already holds AFFIRM at BUILD time ((dog,chase,cat), brain_chat_tui `_build*` hear-loop; a build-time store,
    # not a cupy-gated BTSP write, so it is present on ANY backend with no forced-write flag needed). The default
    # `noncontradiction-gate` probe rides the `well` teach turn ("the wolf bites the apple" = brand-new vocab), whose
    # recall is "unknown" on the INTACT substrate too — identical to the lesion's forced-"unknown" — so the gate reads
    # integrated-HOLLOW there for a PROBE reason, not a wiring reason. This turn constructs the driving condition: intact
    # recalls "yes"/AFFIRM -> stored != asserted -> REJECT; lesion forces "unknown" -> ACCEPT — so reject / recalled_yn /
    # stored_polarity all diverge. See research/runners/load_bearing_fraction.py.
    ("noncontra_neg", "the dog does not chase the cat", "ncontra", True, None, False),  # NEGATE assertion of the AFFIRM boot fact (dog,chase,cat) -> intact reject=True (recall 'yes'); lesion accept (forced 'unknown')
    # ── SWAP-DRIVES DRIVING GROUP (label-only; used only by load_bearing_fraction's LB_SWAP_DRIVE_PROBE) ─────────────
    # The default swap-drives probe rides `held` ('the wolf watches the owl'), which (a) names no BUILD-TIME KB concept
    # (wolf/owl are not tiny-demo agents/patients -> gnw_thought_swap._extract_topic returns None -> no_topic_hold) and
    # (b) is answered by the role-binding REPAIR short-circuit, which returns before `swap_drives` is attached -> the
    # field is absent in BOTH arms -> NOT-EXERCISED on every seed. This group is an ordinary topic conversation over the
    # boot facts (dog,chase,cat)/(cat,eat,fish): OPEN establishes the held topic 'dog' (first thought), HOLD re-asks on
    # the SAME topic (the within-session NULL-CONTRAST turn: no swap is due, so the lesion must NOT change it), SWITCH
    # asks about a DIFFERENT grounded concept 'cat' (the salient competing topic: the neural mismatch detector should
    # fire -> evict 'dog' -> admit 'cat' -> the reply leads "On cat, then -- ..."; BRAIN_SWAP_DRIVES_LESION silences
    # the detector -> no swap -> no lead). Kept OUT of PROBE_TURNS -> the default roster + every flip-verify harness
    # stay BYTE-IDENTICAL. Pre-registration: research/findings/2026-09-23-swap-drives-adequate-probe-PREREGISTRATION.md.
    ("sw_open",   "what does the dog chase", "sw2", True,  None, False),   # establish the held topic 'dog' (first thought)
    ("sw_hold",   "what does the dog chase", "sw2", False, None, False),   # SAME topic -> hold (contrast turn: lesion must not change it)
    ("sw_switch", "what does the cat eat",   "sw2", False, None, False),   # DIFFERENT grounded topic 'cat' -> intact swap + lead / lesion no swap
    # ── PROSPECTIVE-MEMORY DRIVING GROUP (label-only; used only by load_bearing_fraction's LB_PMEM_DRIVE_PROBE) ──────
    # Prospective memory is, by definition, an intention held ACROSS INTERVENING ACTIVITY and released at a LATER cue
    # (McDaniel & Einstein 2000 multiprocess framework). The held-intention x cue coincidence in the SFA/NMDA substrate
    # only reaches its operating point (rel_A crosses FIRE_THR=0.2) AFTER the hold has been advanced by intervening
    # turns: measured organ-level, intact rel_A ramps 0.163(n=0)->0.221(n=1)->0.256(n=2)->0.340(n=3) while the
    # BRAIN_PMEM_LESION arm stays ~0.04 at every n. A zero-delay formation->cue (n=0) does NOT fire even intact (0.163
    # < 0.2) -> intact==lesion==not-fired -> HOLLOW: that is exactly why the prior 2-turn [pmem_form2,pmem_cue] driving
    # group read treat=0. THREE intervening distractor turns (matching the validated isolated-verify + de-risk protocol,
    # fire_on_cue 6/6) put the intact arm at rel_A~0.34 (70% over threshold) so it FIRES, while the lesioned latch stays
    # silent -> `prospective.fired` FLIPS True/False. Session 'pmem2', declared formation-first; the distractors carry
    # neither cue keyword ('bird'/'sings') nor a formation phrasing, and NONE is in PROBE_TURNS -> the default roster,
    # the regression battery and every flip-verify harness stay BYTE-IDENTICAL. See research/runners/load_bearing_fraction.py.
    ("pmem_form2","remind me to feed the dog when the bird sings", "pmem2", True,  None,   False),  # FORMATION: latch the deferred intention (one-shot Hebbian cue->action binding)
    ("pmem_d0",   "what does the cat eat",     "pmem2", False, None,   False),  # intervening turn 1: advances the hold (real competing WM load), cue stays silent
    ("pmem_d1",   "how is the weather today",  "pmem2", False, None,   False),  # intervening turn 2
    ("pmem_d2",   "tell me about the sky",     "pmem2", False, None,   False),  # intervening turn 3 -> the held x cue coincidence is now at its operating point
    ("pmem_cue",  "the bird sings",            "pmem2", False, None,   False),  # CUE: the held x cue coincidence fires (intact) / collapses silent (lesion)
    # ── OPEN-ENDED-GENERATION DRIVING GROUP v2 (load_bearing_fraction's LB_OPEN_ENDED_DRIVE_PROBE) ────────────────
    # WHY (finding 2026-09-20-gap-open-ended-generation-v2): the default open-ended probe (`rich_open` = "what might a
    # dog chase" on a FRESH tiny-demo brain) is integrated-HOLLOW: the tiny KB has ONE 'chase' fact -- (dog,chase,cat)
    # -- already stored, so the ONLY reachable (dog,chase,?) patient is novelty-excluded -> _generate_hypothesis
    # abstains in BOTH arms. The v1 fix (branch hollow-open-ended-generation-drive) taught 9 chase facts but read
    # treat=0 on the real brain for a MECHANISM reason this v2 diagnosed: the stored 'cat' has co-occurrence weight 2
    # with (dog,chase) and, tied with the twice-taught 'rabbit', WON the intact spiking-WTA argmax -> the intact draw
    # FIXATED on 'cat' (novelty-excluded) and dead-ended to abstain, so the likelihood ablation could not show. This
    # v2 group teaches a NATURAL predator-prey chase KB where 'rabbit' is chased by FOUR predators (wolf/fox/hawk/
    # eagle) -> chase~rabbit co-occurrence 4 -> weight(dog,chase,rabbit)=4 STRICTLY dominates the stored cat's 2, so
    # the INTACT likelihood-weighted draw peaks the NOVEL 'rabbit' (volunteers "a dog might chase a rabbit"); the
    # LESION's UNIFORM draw (BRAIN_SPIKING_DRAW_LESION -> the honored ablate on draw_from_weights) has no likelihood
    # bias and selects among ALL novel plausible patients {rabbit,deer,boar,mouse,minnow,beetle} -> a hypothesis_svo/
    # answer diff. Kept OUT of PROBE_TURNS (label-only) -> the default roster + every flip-verify harness stay
    # BYTE-IDENTICAL. See research/runners/load_bearing_fraction.py.
    # NATURAL predator-prey chase KB: 'rabbit' is chased by FOUR predators (wolf/fox/hawk/eagle) so chase~rabbit
    # co-occurrence = 4 -> weight(dog,chase,rabbit) STRICTLY dominates the stored cat's 2 -> the INTACT likelihood-
    # weighted spiking draw peaks the NOVEL 'rabbit'; five other predator->prey singletons give novel plausible
    # ALTERNATIVES the LESION's uniform draw selects among. NB: the measurement (load_bearing_fraction) applies
    # base_env BRAIN_SPIKING_PLAUSIBILITY=0 to BOTH arms so the (co-occurrence-1) #3E gate admits the candidates --
    # WITHOUT that the default-ON spiking plausibility read is too conservative on the tiny KB's weak agent-action
    # edge to admit ANY, and both arms abstain (the v2 masking finding); the ONLY inter-arm difference stays the draw
    # lesion, so the treat diff is attributable to the draw.
    ("oe_t1",  "the wolf chase the rabbit",   "oe2", True,  None,   False),   # NOVEL: rabbit (chase-cooc #1)
    ("oe_t2",  "the fox chase the rabbit",    "oe2", False, None,   False),   # rabbit (chase-cooc #2)
    ("oe_t3",  "the hawk chase the rabbit",   "oe2", False, None,   False),   # rabbit (chase-cooc #3)
    ("oe_t4",  "the eagle chase the rabbit",  "oe2", False, None,   False),   # rabbit (chase-cooc #4) -> STRICT intact likelihood peak
    ("oe_t5",  "the lion chase the deer",     "oe2", False, None,   False),   # NOVEL alternative: deer
    ("oe_t6",  "the bear chase the mouse",    "oe2", False, None,   False),   # NOVEL alternative: mouse
    ("oe_t7",  "the owl chase the beetle",    "oe2", False, None,   False),   # NOVEL alternative: beetle
    ("oe_t8",  "the pike chase the minnow",   "oe2", False, None,   False),   # NOVEL alternative: minnow
    ("oe_t9",  "the crow chase the boar",     "oe2", False, None,   False),   # NOVEL alternative: boar
    ("oe_ask", "what might a dog chase",      "oe2", False, None,   True),    # rich=True -> the generation branch (resp['hypothesis_svo']); draws (dog,chase,?) over the now-rich chase graph
    # ── WM-BINDING HOLD-QUERY PAIRS (label-only; NOT in PROBE_TURNS; used only by LB_WMB_HOLDQUERY_PROBE) ────────
    # The default wm-binding-advanced probe ('held' = 'the wolf watches the owl') names ONE referent the D6 organ's
    # hand lexicon admits ('owl' is not on _REFERENT_NOUNS) and exits through the comprehension-repair return, so the
    # organ never reaches the reply. The organ's reply path is its HOLD-QUERY read-out: after a turn introduces >=2
    # referents it knows ('fox', 'wolf' are both on _REFERENT_NOUNS), "who are we talking about" is answered by reading
    # every held referent back off the spiking buffer. 'wmb' = the driving pair (2 referents -> the read-out answers);
    # 'wmb1' = the SPECIFICITY control (1 referent -> the organ is out of scope, the ask falls through to the normal
    # path, so the lesion must NOT change that reply). Pre-registration:
    # research/findings/2026-09-23-wm-binding-holdquery-adequate-probe-PREREGISTRATION.md.
    ("wmb_intro",  "the fox and the wolf walked in", "wmb",  True,  None, False),  # 2 lexicon referents -> the organ LOADs fox+wolf (maintain)
    ("wmb_ask",    "who are we talking about",       "wmb",  False, None, False),  # hold-query -> reply = read-back off the spiking buffer
    ("wmb1_intro", "the fox walked in",              "wmb1", True,  None, False),  # 1 referent -> organ out of scope (judge returns None)
    ("wmb1_ask",   "who are we talking about",       "wmb1", False, None, False),  # same ask, organ out of scope -> falls through; lesion must not change it
    # ── WM-BINDING ORDINARY-CONTENT PAIRS (label-only; used only by LB_WMB_CONTENT_PROBE) ─────────────────────────
    # The hold-query above is PASS-BY-CONSTRUCTION (its reply template's only input is the lesioned buffer; review
    # v2:7a3b94367) -> an INTEGRITY smoke. These pairs ask whether the organ's held state changes an ORDINARY content
    # reply: the organ LOADs two referents on the intro, then an ordinary transitive that the comprehension organ reads
    # WITH this session's held WM focus co-driven through the one-brain cross-edge (the organ's only path into an
    # ordinary reply). 'wmc' = fox/wolf; 'wmcx' = the content-swapped in-scope control (cat/dog, same structure).
    ("wmc_intro",  "the fox and the wolf walked in", "wmc",  True,  None, False),  # 2 lexicon referents -> LOAD fox+wolf
    ("wmc_drive",  "the wolf watches the owl",       "wmc",  False, None, False),  # ordinary transitive (no hold-query)
    ("wmcx_intro", "the cat and the dog walked in",  "wmcx", True,  None, False),  # content-swapped: LOAD cat+dog
    ("wmcx_drive", "the dog watches the owl",        "wmcx", False, None, False),  # same structure, swapped content
    # ── WM REFERENT->FOCUS BIND ANAPHOR PAIRS (label-only; used only by LB_WMB_FOCUS_PROBE) ─────────────────────
    # Each pair = two sessions that introduce the SAME two lexicon referents in SWAPPED order (so which referent sits
    # in which register differs; the words do not), then the SAME ordinary anaphor question. With
    # BRAIN_MULTIREF_FOCUS_BIND=1 the organ resolves 'it' to the register that wins its retrieval competition, so the
    # two sessions must answer differently; a positional route cannot. Pre-registration:
    # research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md.
    ("wmf_a1_intro", "the dog and the cat walked in",  "wmfa1", True,  None, False),  # LOAD dog (reg0) + cat (reg1)
    ("wmf_a1_ask",   "what does it chase",             "wmfa1", False, None, False),  # anaphor -> the retrieved referent
    ("wmf_a2_intro", "the cat and the dog walked in",  "wmfa2", True,  None, False),  # SWAPPED: cat (reg0) + dog (reg1)
    ("wmf_a2_ask",   "what does it chase",             "wmfa2", False, None, False),  # identical question
    ("wmf_b1_intro", "the cat and the bird walked in", "wmfb1", True,  None, False),  # second pair, other referents
    ("wmf_b1_ask",   "what does it eat",               "wmfb1", False, None, False),
    ("wmf_b2_intro", "the bird and the cat walked in", "wmfb2", True,  None, False),  # SWAPPED
    ("wmf_b2_ask",   "what does it eat",               "wmfb2", False, None, False),
]
# ── DA TAG-AND-CAPTURE NEXT-DAY GROUPS (label-only; load_bearing_fraction's LB_DA_TAG_CAPTURE_PROBE + the dedicated
# research/runners/_da_tag_capture_chat_probe.py) ─────────────────────────────────────────────────────────────────
# da-gated-encoding acts on PERSISTENCE, not on an immediate read (Bethus, Tse & Morris 2010: D1/D5 blockade spares
# encoding + immediate recall and changes ~24 h retention). Its default probe (`well`, one fresh turn) can never show
# that, and the battery had no next-day turn. Each group below tells ONE fact inside a short conversation, lets a
# night pass through the brain's own idle/sleep tick (the `_WORLD_STEPS` pseudo-turn below: the ENVIRONMENT's clock
# jumps 24 h, then continuous_engine.tick_idle_sessions runs at that time exactly as the server loop calls it), then
# asks for the fact. SALIENT: the fact sits inside surprising news; NEUTRAL: the same fact told plainly after its words
# were introduced (habituated). *_imm groups ask at once (no night) -- the immediate-recall precondition + the
# lesion's spare-encoding contrast. Separate sessions per group; none is in PROBE_TURNS -> the default roster, the
# regression battery and every flip-verify harness stay BYTE-IDENTICAL.
_DATC_SALIENT = [
    "Guess what, something unbelievable happened at the circus today!",
    "You will never believe this crazy story, it is absolutely amazing!",
    "the cat chases the ball",
    "Everyone in the audience was screaming and laughing in total shock!",
    "Honestly it was the most astonishing spectacle anybody has ever witnessed!",
]
_DATC_NEUTRAL = [
    "the cat is here",
    "the ball is here",
    "the cat chases the ball",
    "the cat is here",
    "the ball is here",
]
_DATC_RECALL = "what does the cat chase"
_WORLD_NIGHT = "__world_step:overnight_24h__"   # never sent to brain_chat: the worker runs the world step instead


def _datc_group(prefix, session, texts, night):
    rows = [("%s_t%d" % (prefix, i + 1), txt, session, i == 0, None, False) for i, txt in enumerate(texts)]
    if night:
        rows.append(("%s_night" % prefix, _WORLD_NIGHT, session, False, None, False))
    rows.append(("%s_recall" % prefix, _DATC_RECALL, session, False, None, False))
    return rows


_EXTRA_TURNS += (_datc_group("datc", "datc", _DATC_SALIENT, True) + _datc_group("datn", "datn", _DATC_NEUTRAL, True)
                 + _datc_group("datci", "datci", _DATC_SALIENT, False)
                 + _datc_group("datni", "datni", _DATC_NEUTRAL, False))

# ── SLEEP-ROUTE r2 GROUPS (label-only; research/runners/_da_tag_capture_chat_probe.py --family r2, branch
# research/sleep-replay-capture-r2; gates in the sleep-replay-capture PREREGISTRATION, Amendment 1) ────────────────
# 'datl' LONG DELAY: the neutral telling, then the brain stays AWAKE 4 h with no conversation (`_WORLD_AWAKE`: the
#   environment clock moves, the body's awake mark is set, no idle tick runs), then the usual night, then recall. The
#   fact is ~4 h old at sleep onset, past the ~2-3 h capture window.
# 'd3w' / 'd3c' / 'd3r' THREE NIGHTS: a WEAK telling (the fact said last, after its words have habituated the brain's
#   DA), a salient telling, and the weak telling RE-MENTIONED once on each of the two following days; three nights;
#   recall after the third.
_WORLD_AWAKE = "__world_step:awake_4h__"   # never sent to brain_chat: the worker runs the world step instead
_DATC_WEAK = [
    "the cat is here",
    "the ball is here",
    "the cat is here",
    "the ball is here",
    "the cat chases the ball",
]
_EXTRA_TURNS += ([("datl_t%d" % (i + 1), txt, "datl", i == 0, None, False) for i, txt in enumerate(_DATC_NEUTRAL)]
                 + [("datl_awake", _WORLD_AWAKE, "datl", False, None, False),
                    ("datl_night", _WORLD_NIGHT, "datl", False, None, False),
                    ("datl_recall", _DATC_RECALL, "datl", False, None, False)])


def _d3_group(prefix, texts, remention):
    rows = [("%s_t%d" % (prefix, i + 1), txt, prefix, i == 0, None, False) for i, txt in enumerate(texts)]
    for n in (1, 2, 3):
        rows.append(("%s_night%d" % (prefix, n), _WORLD_NIGHT, prefix, False, None, False))
        if remention and n < 3:
            rows.append(("%s_remention%d" % (prefix, n), "the cat chases the ball", prefix, False, None, False))
    rows.append(("%s_recall" % prefix, _DATC_RECALL, prefix, False, None, False))
    return rows


_EXTRA_TURNS += (_d3_group("d3w", _DATC_WEAK, False) + _d3_group("d3c", _DATC_SALIENT, False)
                 + _d3_group("d3r", _DATC_WEAK, True))
# 'd10w' HORIZON (r2 Amendment 3, REPORTED): the weak telling, then TEN nights, each followed by the recall question
# (a read-only probe in this model: the store read writes nothing), so the night a fact stops being recalled is measured.
_EXTRA_TURNS += ([("d10w_t%d" % (i + 1), txt, "d10w", i == 0, None, False) for i, txt in enumerate(_DATC_WEAK)]
                 + [row for n in range(1, 11) for row in (("d10w_night%d" % n, _WORLD_NIGHT, "d10w", False, None, False),
                                                          ("d10w_recall%d" % n, _DATC_RECALL, "d10w", False, None,
                                                           False))])

# ── FORGETTING-INTERFERENCE GROUPS (label-only; research/runners/_da_tag_capture_chat_probe.py --family fi, branch
# research/sleep-forgetting-interference; gates in the sleep-replay-capture PREREGISTRATION, Amendment 6) ───────────────
# The weak telling (as d3w / d10w), then SEVEN nights, each followed by the recall question (a read-only probe in this
# model). On each later day, after that morning's question, the ENVIRONMENT tells k other facts that share no content
# word with the target (no cat / chase / ball): 'fiv' k=0 (nothing else learned), 'fil' k=1, 'fih' k=3; 'fis' the
# salient telling with k=3; 'fir' the weak telling re-mentioned after nights 1 and 2 (before that day's facts), k=3.
# An environment check (tag-capture ledger on, no night, no recall of the fact) found every sentence below STORED as
# one new block by the tiny-demo brain, in this order, at seed 42 (all 18) and seed 7 (13 of them; the rest untried).
# The brain's spiking comprehension gate does NOT store 'eat' / 'learn' sentences, animate patients, or most 'words'
# patients (its role binding does not resolve), so every fact here is animate agent + use/store + inanimate patient.
# k=1 tells the first six, one a day; k=3 tells three a day in order.
_FI_NIGHTS = 7
_FI_FACTS = [
    "the bird uses the river", "the dog stores the memory", "the fish stores the spikes",     # (k=3) day 2
    "the worm uses the river", "the fish stores the memory", "the dog uses the spikes",       # day 3
    "the bird stores the spikes", "the worm stores the memory", "the dog uses the river",     # day 4
    "the fish uses the river", "the bird stores the memory", "the dog stores the spikes",     # day 5
    "the bird uses the spikes", "the fish uses the spikes", "the worm stores the spikes",     # day 6
    "the dog uses the memory", "the bird uses the memory", "the fish uses the memory",        # day 7
]


def _fi_group(prefix, texts, k, remention_after=()):
    rows = [("%s_t%d" % (prefix, i + 1), txt, prefix, i == 0, None, False) for i, txt in enumerate(texts)]
    for n in range(1, _FI_NIGHTS + 1):
        rows.append(("%s_night%d" % (prefix, n), _WORLD_NIGHT, prefix, False, None, False))
        rows.append(("%s_recall%d" % (prefix, n), _DATC_RECALL, prefix, False, None, False))
        if n < _FI_NIGHTS:
            if n in remention_after:
                rows.append(("%s_remention%d" % (prefix, n), "the cat chases the ball", prefix, False, None, False))
            for j in range(k):
                rows.append(("%s_d%df%d" % (prefix, n + 1, j + 1), _FI_FACTS[(n - 1) * k + j], prefix, False, None,
                             False))
    return rows


_EXTRA_TURNS += (_fi_group("fiv", _DATC_WEAK, 0) + _fi_group("fil", _DATC_WEAK, 1) + _fi_group("fih", _DATC_WEAK, 3)
                 + _fi_group("fis", _DATC_SALIENT, 3) + _fi_group("fir", _DATC_WEAK, 3, (1, 2)))

# ── AWAKE-REST GROUPS (label-only; research/runners/_da_tag_capture_chat_probe.py --family arc, branch
# research/awake-replay-capture; gates in the sleep-replay-capture PREREGISTRATION, Amendment 4) ────────────────────
# The long-delay telling of 'datl' (a fact ~4 h old at sleep onset), but the waking interval now contains QUIET REST:
# the body stays awake (the awake mark moves with the clock) and the continuous engine runs a LIGHT idle tick (idle
# == IDLE_SEC, never sleep-depth) at the end of each rest period, so any idle-tick process -- and the awake-replay
# bout when BRAIN_AWAKE_REPLAY_CAPTURE is armed -- runs there. Every arm of a group gets the SAME ticks; only flags differ.
# 'datr'  neutral telling, 4 h of rest (a tick every 5 min = AWAKE_BOUT_H: 48), night, recall.
# 'datcr' the SALIENT telling, the same 4 h of rest, night, recall.
# 'datq'  neutral telling, 4 h awake with one rest tick per hour (4), night, recall (the rest-dose point).
# 'datz'  neutral telling, 3 h awake WITHOUT rest, then 1 h of rest (12 ticks), night, recall (late rest: regrowth).
_WORLD_AWAKE_REST = "__world_step:awake_rest_4h__"
_WORLD_AWAKE_REST_HOURLY = "__world_step:awake_rest_hourly_4h__"
_WORLD_AWAKE_3H = "__world_step:awake_3h__"
_WORLD_AWAKE_REST_1H = "__world_step:awake_rest_1h__"


def _arc_group(prefix, texts, steps):
    rows = [("%s_t%d" % (prefix, i + 1), txt, prefix, i == 0, None, False) for i, txt in enumerate(texts)]
    rows += [("%s_%s" % (prefix, name), step, prefix, False, None, False) for name, step in steps]
    rows += [("%s_night" % prefix, _WORLD_NIGHT, prefix, False, None, False),
             ("%s_recall" % prefix, _DATC_RECALL, prefix, False, None, False)]
    return rows


_EXTRA_TURNS += (_arc_group("datr", _DATC_NEUTRAL, [("rest", _WORLD_AWAKE_REST)])
                 + _arc_group("datcr", _DATC_SALIENT, [("rest", _WORLD_AWAKE_REST)])
                 + _arc_group("datq", _DATC_NEUTRAL, [("rest", _WORLD_AWAKE_REST_HOURLY)])
                 + _arc_group("datz", _DATC_NEUTRAL, [("awake", _WORLD_AWAKE_3H), ("rest", _WORLD_AWAKE_REST_1H)]))
# world-step kind -> (hours, rest period in hours or None = awake without rest). 5 min = AWAKE_BOUT_H (reused).
_AWAKE_STEP_KINDS = {"awake_4h": (4.0, None), "awake_3h": (3.0, None), "awake_rest_4h": (4.0, 5.0 / 60.0),
                     "awake_rest_hourly_4h": (4.0, 1.0), "awake_rest_1h": (1.0, 5.0 / 60.0)}

# ── D5-CONSOLIDATE / SLEEP-REPLAY DRIVING GROUPS (label-only; used only by load_bearing_fraction's new
# lbf_rows/learning.py EXTRA_PROBES for "d5-consolidate" / "sleep-replay") ──────────────────────────────────────
# Both faculties are gated on the SAME idle tick the DA tag-capture groups above already exercise (_WORLD_NIGHT ->
# _run_world_step("overnight_24h") -> webapp.continuous_engine.tick_idle_sessions); no new world-step kind is
# introduced. 24h idle trivially clears both IDLE_SEC (20s, gates D5) and SLEEP_IDLE_SEC (300s, gates sleep-replay).
#
# 'd5c' (D5 learn-through-use): teach 'wolf' -> a referential recall (Hook A completion arms the consolidation
# budget via continuous_engine.mark_recall) -> the idle tick (intact: consolidate_used_memory strengthens the
# 'wolf' assembly via the substrate's own plateau-gated BTSP; BRAIN_D5_CONSOLIDATE=0: no-op) -> the SAME referential
# recall again. The DRIVING field is the graded apical magnitude (episodic.graded_cue.depth_hold), which rises
# ONLY on the intact post-tick recall (recall_disclosure surfaces it only for a topic actually consolidated this
# conversation) -- NOT episodic.in_memory, which is True in both arms on both recalls (the completion gate is not
# what this faculty lesions). d5c_recall1 is the PRECONDITION turn (must read identical intact vs lesion -- no tick
# has run yet); it is deliberately NOT the row's compared turn (see research/runners/lbf_rows/learning.py).
_EXTRA_TURNS += [
    ("d5c_teach", "the wolf chase the rabbit", "d5c", True, None, False),
    ("d5c_recall1", "you mentioned the wolf", "d5c", False, None, False),
    ("d5c_tick", _WORLD_NIGHT, "d5c", False, None, False),
    ("d5c_recall2", "you mentioned the wolf", "d5c", False, None, False),
    # 'slp' (offline sleep-replay): store 3 episodes (fox/owl/hawk), a genuine sleep-depth idle (>=SLEEP_IDLE_SEC
    # via the same 24h step), then recall the MIDDLE-stored one ('owl'). Intact: consolidate_sleep_replay batch-
    # reactivates all 3 in store order (BRAIN_SLEEP_REPLAY default-ON) -> the recall reads a risen depth_hold + the
    # "replayed it offline" clause; BRAIN_SLEEP_REPLAY=0: no-op, un-replayed baseline. episodic.in_memory is True
    # in both arms (the batch-replay lesion never touches whether the topic completes, only how strong it reads).
    ("slp_teach1", "the fox chase the hare", "slp", True, None, False),
    ("slp_teach2", "the owl chase the mouse", "slp", False, None, False),
    ("slp_teach3", "the hawk chase the vole", "slp", False, None, False),
    ("slp_tick", _WORLD_NIGHT, "slp", False, None, False),
    ("slp_recall", "you mentioned the owl", "slp", False, None, False),
]


def _lbf_rows_extra_turns():
    """Turns that research/runners/lbf_rows/<module>.py declares as a LITERAL `EXTRA_TURNS` list (same 6-tuple shape as
    _EXTRA_TURNS), read with ast and NEVER imported: a row module may set env defaults at import, and a battery worker
    must not inherit them. A label already defined here wins; a malformed entry is skipped. This is what lets a row's
    probe turn -- and the same-session turns before it (turn_group) -- reach the workers (2026-09-24: live_organs'
    affective-tom and causal-whatif were parked because the battery could not run lbf_tom1 / lbf_cau_whatif)."""
    import ast
    import glob
    found = []
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lbf_rows", "*.py"))):
        if os.path.basename(path).startswith("_"):
            continue
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "EXTRA_TURNS" for t in node.targets):
                try:
                    value = ast.literal_eval(node.value)
                except ValueError:
                    continue
                found += [tuple(t) for t in value
                          if isinstance(t, (tuple, list)) and len(t) == 6 and isinstance(t[0], str)]
    return found


_known_labels = {t[0] for t in PROBE_TURNS} | {t[0] for t in _EXTRA_TURNS}
for _t in _lbf_rows_extra_turns():
    if _t[0] not in _known_labels:
        _EXTRA_TURNS.append(_t)          # in place: load_bearing_fraction imports this list by reference
        _known_labels.add(_t[0])
_WORLD_STEPS = {t[0]: "overnight_24h" for t in _EXTRA_TURNS if t[1] == _WORLD_NIGHT}
_WORLD_STEPS.update({t[0]: "awake_4h" for t in _EXTRA_TURNS if t[1] == _WORLD_AWAKE})   # r2 (label-only groups)
_WORLD_STEPS.update({t[0]: {_WORLD_AWAKE_REST: "awake_rest_4h", _WORLD_AWAKE_REST_HOURLY: "awake_rest_hourly_4h",
                            _WORLD_AWAKE_3H: "awake_3h", _WORLD_AWAKE_REST_1H: "awake_rest_1h"}[t[1]]
                     for t in _EXTRA_TURNS
                     if t[1] in (_WORLD_AWAKE_REST, _WORLD_AWAKE_REST_HOURLY, _WORLD_AWAKE_3H, _WORLD_AWAKE_REST_1H)})
_TURN_BY_LABEL.update({t[0]: t for t in _EXTRA_TURNS})


def _run_world_step(kind):
    """The ENVIRONMENT passes time (host = world, legitimate): a night of `hours` -- the world clock jumps, then the
    brain's own between-turn process runs once at that time, exactly as the server's background loop calls it
    (webapp/server.py: continuous_engine.tick_idle_sessions with the same four getters). Being idle >= SLEEP_IDLE_SEC
    makes it a sleep-depth tick (sleep replay, the Turrigiano pass, the tag-and-capture ledger when armed). Returns a
    trace (no reply: nothing is said)."""
    import time as _time
    hours = {"overnight_24h": 24.0, "awake_4h": 4.0}[kind] if kind in ("overnight_24h", "awake_4h") \
        else _AWAKE_STEP_KINDS[kind][0]
    from webapp import server as _S
    from webapp import continuous_engine as _CE
    if kind in _AWAKE_STEP_KINDS and _AWAKE_STEP_KINDS[kind][1] is not None:
        # (awake-replay-capture) QUIET REST while awake: in each rest period the environment clock moves, every live
        # ledger is told the body was awake through world-now, then the engine runs ONE light idle tick (idle ==
        # IDLE_SEC after the last request: never sleep-depth), exactly the tick the server loop would run in a pause.
        from webapp import da_tag_capture_chat as _DTC
        period = _AWAKE_STEP_KINDS[kind][1]
        n_steps = int(round(hours / period))
        n_ticked = 0
        for _k in range(n_steps):
            _DTC.advance_world_clock_h(period)
            for _c in list(_S._BRAIN_CHATS.values()):
                _DTC.mark_awake(_c)
            _last = [v for v in _CE._LAST_REQUEST.values() if v is not None]
            now = (max(_last) if _last else _time.time()) + _CE.IDLE_SEC
            n_ticked += int(_CE.tick_idle_sessions(_S._SESSION_MOOD, _S._get_affect_organ, now=now,
                                                   selfinit_getter=_S._get_selfinit_organ,
                                                   episodic_getter=_S._get_episodic_organ_existing,
                                                   chat_getter=_S._get_chat_existing) or 0)
        return {"world_step": kind, "hours": hours, "rest_period_h": period, "n_rest_ticks": n_steps,
                "n_session_ticks": n_ticked}
    if kind in ("awake_4h", "awake_3h"):
        # (r2) the body stays AWAKE: only the environment clock moves and every live session's ledger is told the
        # brain was awake through world-now. No idle tick runs (the engine would read >= SLEEP_IDLE_SEC of idle as
        # sleep). A session without a ledger is untouched.
        from webapp import da_tag_capture_chat as _DTC
        _DTC.advance_world_clock_h(hours)
        n_marked = sum(1 for _c in list(_S._BRAIN_CHATS.values()) if _DTC.mark_awake(_c) is not None)
        return {"world_step": kind, "hours": hours, "n_sessions_marked_awake": int(n_marked)}
    try:
        from webapp import da_tag_capture_chat as _DTC
        _DTC.advance_world_clock_h(hours)
    except ImportError:
        _DTC = None
    now = _time.time() + hours * 3600.0
    n = _CE.tick_idle_sessions(_S._SESSION_MOOD, _S._get_affect_organ, now=now,
                               selfinit_getter=_S._get_selfinit_organ,
                               episodic_getter=_S._get_episodic_organ_existing,
                               chat_getter=_S._get_chat_existing)
    return {"world_step": kind, "hours": hours, "n_sessions_ticked": int(n or 0)}

# ── continuous fields to NEVER compare (a background process advances between builds; decisions are stable, not these)
_NOISE_FIELDS = {
    "rate_perceived", "rate_generated", "neg_rate", "pos_rate", "vminus_rate", "vplus_rate", "mood", "differential",
    "appraisal_valence", "appraisal_arousal", "felt_arousal", "ema_arousal", "ema_valence", "ema_engagement",
    "da_level", "snc_firing", "afferent_pA", "turn_engagement", "g", "d", "n_facts_scanned", "wm_margin",
    "gen_seconds", "body_a", "body_h", "confidence", "tone_level", "level", "appraisal_hits",
}


def _get_path(d, path):
    """Fetch a dotted path (e.g. 'affect.valence_sign'); returns (present, value)."""
    cur = d
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return (False, None)
    return (True, cur)


# ── the faculty registry: faculty -> (probe turn label, [decision field paths], thin?) ───────────────────────────
# `thin=True` marks a faculty whose driving decision fields are not reliably populated by this probe set (it rides the
# shared top-level decision on its turn); reported as `not-exercised` when its fields are absent. Aligned to the
# PRODUCTION_INTEGRATION_LEDGER on-by-default rows.
FACULTY_PROBES = [
    # (faculty_key, turn_label, decision_field_paths, thin)
    ("content-selection",       "well",     ["answer", "abstained", "recalled_svo", "activity.matched_fact_index"], False),
    ("semantic-recall",         "well",     ["recalled_svo", "activity.composer", "verified"], False),
    ("one-brain-substrate",     "well",     ["activity.composer"], False),
    ("moat-verify",             "unknown",  ["abstained", "answer"], False),
    ("in-loop-learning",        "well",     ["answer", "recalled_svo"], False),
    ("comprehension-monitor",   "well",     ["comprehension.on", "comprehension.comprehended"], False),
    # LIFTED 2026-09-02: "well" ('the wolf bites the apple') never exercises the LEARNED cue extension -- every
    # word in it is hand-table-covered, so the learned lexicon is never consulted. `animacy`/`verbsel` use a noun
    # ('monkey') / verb ('clean') the ~19-noun / 8-verb HAND table misses but the learned lexicon covers (the exact
    # examples the ledger's own lesion_note uses); with the (default-ON) learned cue enabled, `competent()` passes
    # and `comprehension.on` is populated -- with it OFF, `judge()` returns None and the whole key is absent, so a
    # flip of EITHER learned-cue flag is a presence/absence swing this field genuinely catches.
    ("comprehension-learned-animacy-cue",  "animacy", ["comprehension.on"], False),
    ("comprehension-learned-verb-selects", "verbsel", ["comprehension.on"], False),
    ("noncontradiction-gate",   "well",     ["noncontradiction.on", "noncontradiction.reject",
                                             "noncontradiction.recalled_yn", "noncontradiction.asserted_polarity"], False),
    ("affect-coloring",         "well",     ["affect.on", "affect.valence_sign", "affect.tone_token"], False),
    ("affect-drives-response",  "well",     ["affect_drives.on", "affect_drives.acted", "affect_drives.high_arousal",
                                             "affect_drives.reason"], False),
    # LIFTED 2026-09-02: `expression_lead()` returns '' UNCONDITIONALLY at mood level 0 (checked BEFORE either
    # selection path even runs), and "well" 's mood stays neutral (level 0) -- so the field could never discriminate
    # the spiking marker-WTA from the host `_LEAD_WORD` dict lookup it replaces. `emo` is strongly-affective (crosses
    # the ~0.045 L2 mood-level threshold in one turn) so a marker word is actually SELECTED; the WTA's own choice is
    # `affect_drives.lead` (the marker string), not `affect.valence_sign` (a DIFFERENT, Gate-B-only ladder read the
    # affect-coloring row already covers).
    ("affect-marker-spiking-wta", "emo",     ["affect_drives.lead"], False),
    ("da-mode-drives-response", "well",     ["da_drives.on", "da_drives.acted", "da_drives.mode", "da_drives.reason"], False),
    ("da-gated-encoding",       "well",     ["da_encoding.on"], False),
    ("source-provenance-honesty", "well",   ["provenance.known", "provenance.label", "provenance.agrees_with_encoded",
                                             "provenance.encoded_as"], False),
    ("common-ground-drives",    "well",     ["common_ground_drives.on", "common_ground_drives.decision",
                                             "common_ground_drives.reason"], False),
    # LIFTED 2026-09-02: `resp["confidence_forthcoming"]` is only ATTACHED on the rich (multi-sentence) path -- the
    # battery's probes historically hardcoded rich=False, so this key never appeared. `rich_well` is the SAME
    # well-formed transitive with rich=True. The OLD field path was also wrong: `affect.forthcomingness` is the
    # MOOD-set floor (max_sentences/max_elaborations, a different coupling, #81/#84), not this organ's own
    # granted/reason trace.
    ("confidence-forthcomingness", "rich_well", ["confidence_forthcoming.granted", "confidence_forthcoming.reason"], False),
    ("swap-drives-response",    "held",     ["swap_drives.on", "swap_drives.acted", "swap_drives.swapped",
                                             "swap_drives.reason"], False),
    ("anaphora-wm",             "held",     ["activity.roles"], False),
    ("wm-binding-advanced",     "held",     ["multiref.n_referents"], False),
    # LIFTED 2026-09-02: "well" (a fresh TEACH -- 'wolf'/'bite'/'apple' are new vocabulary) never forms an intention.
    # `pmem_form` ('remind me to feed the dog when the bird sings') matches the FORMATION regex -- a disjoint
    # short-circuit that latches the intention and returns `resp["prospective"]` (not `resp["pmem"]`, the old path
    # was also the wrong top-level key). This exercises FORMATION only, not the later cue-fire half (a 3rd turn);
    # honest partial coverage, not a fake full-cycle claim.
    ("prospective-memory",      "pmem_form", ["prospective.held"], False),
    # LIFTED 2026-09-02: field-path fix only (the turn already drove it) -- `pragmatic_production_organ.interpret()`
    # returns "implicature_margin"/"enriched_interpretation", never "implicature"; "on" DOES exist (`"pragmatic.on"`
    # was actually fine, kept).
    ("pragmatic-implicature",   "scalar",   ["pragmatic.on", "pragmatic.enriched_interpretation"], False),
    # LIFTED 2026-09-02: "well" is a fresh TEACH -- `extract_assertion` finds no PRIOR `what_does(agent,action)` to
    # compare against (nothing was stored before this turn), so `surprise_info` stays null structurally, regardless
    # of the flag. `contra` asserts a DIFFERENT patient for an (agent,action) pair the tiny-demo brain already knows
    # from BUILD time (dog,chase,cat) -> a genuine CONTRADICT (surprised=True). Field path unchanged ("on" is
    # absent from `judge()`'s own dict and is simply skipped by `compare()`; "surprised" is real and populated).
    ("surprise-monitor",        "contra",   ["surprise.surprised", "surprise.on"], False),
    # LIFTED 2026-09-02: on "well" the rf trace shows `matched_fact_index: null` / every role `confidence: null`
    # (a TEACH, not a recall -- nothing MATCHED, so `mean_role_confidence` has nothing to average, and #184's own
    # guard logs a WARNING and returns None) -- metacog is out of scope BY CONSTRUCTION on a teach turn. `confirm`
    # is a genuine RECALL of an already-known fact (dog,chase,cat): the rf composer actually MATCHES, roles carry
    # real confidences, and the metacog read populates for real. Field path unchanged (already correct).
    ("metacog-monitor",         "confirm",  ["metacog.confident", "metacog.on"], False),
    # LIFTED 2026-09-02: E2's QUERYABLE-expectation short-circuit (`is_expectation_query`) needs an explicit "what
    # do you expect / how is this going" turn -- "well" never matches it. Field path unchanged (`exp["pred_sign"]`
    # / `exp["on"]` are both real keys on `WorldModelProductionOrgan.expectation()`'s return dict).
    ("worldmodel-forward",      "expect_q", ["worldmodel.pred_sign", "worldmodel.on"], False),
    # LIFTED 2026-09-02: curiosity only reads on an ABSTAIN (`_curiosity_followup(abstained)` -- out of scope on
    # any non-abstain turn including "well"). `unknown` ('what is the capital of france') already abstains for the
    # moat-verify row -- reusing it drives curiosity too. Field name was ALSO wrong: `judge()` returns "curious",
    # never "crave".
    ("curiosity-followup",      "unknown",  ["curiosity.curious", "curiosity.on"], False),
    # LIFTED 2026-09-02: reconsolidation only fires INSIDE the surprise block on a genuine contradiction (shares
    # `contra`'s trigger + the SAME spiking surprise read, zero extra cost). Field path was also wrong: the
    # `reconsolidate()` return dict has no "revised"/"on" keys -- the real categorical decision is "action"
    # (rewrite / restabilize / abstain / lesioned_nowrite).
    ("reconsolidation",         "contra",   ["reconsolidation.action"], False),
    # LIFTED 2026-09-02: Hook A (`is_referential`) needs a "did we discuss X" / "you mentioned X" -class turn --
    # "well" never matches it. `episodic` is referential on a FRESH session, so `in_memory` reads False (an honest
    # not-in-memory disclosure) -- still a real, non-null, DECISION-STABLE field. Field path was also wrong: the
    # `recall()` dict key is "in_memory", never "stored"/"on".
    ("episodic-memory",         "episodic", ["episodic.in_memory"], False),
    # LIFTED 2026-09-02: "held" ('the wolf watches the owl') is not a before/now QUERY, so `maybe_answer` returns
    # None and the whole `discourse_register` key never appears -- and the old field path ("discourse.event") was
    # never a real key either (the response key is "discourse_register", not "discourse"). `dr_a` folds a bare
    # (no-connective) clause -> the CURRENT event; `dr_b` folds a CONNECTIVE-led clause -> SHIFT (current->prev);
    # `dr_c` is the actual before-query, reading the held PREV slot off cp_firing_states.
    ("discourse-register",      "dr_c",     ["discourse_register.abstained", "discourse_register.agent"], False),
    # LIFTED 2026-09-02: the hypothesis-generation branch (`resp["hypothesis"]`) lives INSIDE the rich composer's
    # own answer path (`is_hyp = bool(r.get("hypothesis"))`), which every existing probe turn bypasses by hardcoding
    # rich=False. `rich_open` is the SAME open-ended prompt with rich=True explicitly requested.
    ("open-ended-generation",   "rich_open", ["hypothesis", "answer"], False),
    # LIFTED 2026-09-02: `resp["n_sentences"]` / a genuine `resp["rich"]=True` are single-fact-path-False by
    # construction (every existing probe hardcodes rich=False) -- `rich_well` requests rich=True on the SAME
    # well-formed transitive.
    ("discourse-planner",       "rich_well", ["rich", "n_sentences"], False),
    # NOT LIFTED (2026-09-02, investigated, genuinely not constructible through this harness): gnw-deliberation's
    # trigger needs >=2 DISTINCT stored patients for the SAME (agent,action) pair (a genuine multi-candidate
    # conflict the substrate must arbitrate). The tiny-demo brain's fixed 5-fact KB has no such duplicate, and
    # `contra` (above) empirically PROVES the live-teach route cannot construct one either: asserting a
    # contradicting patient for an already-known (agent,action) does not create a SECOND candidate, it triggers the
    # default-ON reconsolidation organ to REWRITE the stored patient IN PLACE (`reconsolidation.action=="rewrite"`,
    # verified live) -- so at most one patient is ever stored per (agent,action) key through `/api/brain-chat`. The
    # de-risk's own "dog->chase->{cat,ball}" ambiguity fixture is built by directly constructing a composer with two
    # KB rows, bypassing conversational teaching entirely -- a construction this brain_chat-only battery cannot
    # reach without either a second brain bundle with a genuine pre-existing duplicate (not verified to exist) or
    # forcing BRAIN_RECONSOLIDATION=0 in the probe env (which would falsify the ADJACENT reconsolidation-monitor
    # probe by disabling its own default-ON mechanism for every turn in the same arm build). Left thin=True.
    ("gnw-deliberation",        "well",     ["activity.composer"], True),
    # LIFTED 2026-09-02: the multi-step gate wraps `chat.gate` itself (no dedicated response key) -- an explicit
    # chase-form question ("... all the way") over the tiny-demo's own dog->chase->cat / cat->eat->fish chain drives
    # the re-entrant workspace to the CHAIN TERMINAL ('fish'), which surfaces through the ALREADY-tracked
    # `recalled_svo` field (a single-hop turn would instead recall/abstain on 'cat'). "well" never asks a chase-form
    # question, so this never engaged before.
    ("gnw-multistep-deliberation", "chase", ["recalled_svo"], False),
    # LIFTED 2026-09-02: the self-initiation short-circuit is a DISJOINT idle/empty-turn class
    # (`is_selfinit_trigger`) -- "well" (real content) never matches it. `selfinit` is the empty-string message; the
    # top-level `abstained` field (not spoke) is the simplest robust categorical read (the nested `self_initiated`
    # dict carries continuous want-rate fields alongside it, so comparing the WHOLE dict risks a noise-driven
    # false-regressed verdict this dedicated field avoids).
    ("self-initiated-utterance", "selfinit", ["abstained"], False),
    # LIFTED 2026-09-02: the block only fires when the turn CARRIES a percept AND matches a visual-query pattern --
    # "well" has neither. `vision` supplies `percept="bird"` on a "what do you see" turn (BrainChatRequest.percept,
    # the only production consumer that ever populates it today). Field path was also wrong: the response key is
    # "vision_identity", never "vision".
    ("vision-identity-spiking-hmax", "vision", ["vision_identity.recognized_category"], False),
    # NOT LIFTED (2026-09-02, same root cause as gnw-deliberation above): value-driven-choice resolves the IDENTICAL
    # >=2-distinct-patient (agent,action) ambiguity gnw-deliberation arbitrates (it installs its wrapper INSIDE the
    # same conflict scope, "OUTSIDE the GNW deliberation gate... INSIDE the multistep gate"). The SAME `contra`
    # evidence applies: reconsolidation's default in-place rewrite means brain_chat-only conversational teaching
    # can never leave two candidate patients stored under one (agent,action) key for this organ to choose between.
    # Left thin=True.
    ("value-driven-choice",     "well",     ["value_choice"], True),
    # LIFTED 2026-09-02: the selector is CONSULTED only on a content-empty turn (a normal message always favors
    # SPEAK without even calling the organ) -- "well" is real content. `bgdots` ('...') is the doc's own worked
    # example of a turn where STAY-SILENT is a genuine contender. The top-level `abstained` field is used instead
    # of the old whole-dict `bg_select` path (the same noise-risk reasoning as self-initiated-utterance above).
    ("bg-action-selection",     "bgdots",   ["abstained"], False),
    # LIFTED 2026-09-02 (medium confidence -- verify via the self-test before trusting in production): the WTA race
    # only engages on a BARE PRONOUN query over >=2 held referents of OPPOSING animacy; "held" ('the wolf watches
    # the owl') restates referents directly (no pronoun) and 'wolf'/'owl' are BOTH animate (no opposing-animacy
    # conflict for content_bias_target to resolve). `bc_a` holds one animate (cat) + one inanimate (ball) referent
    # (mirrors the row's own lesion_note worked example exactly); `bc_b` ('what does it eat') is the pronoun+verb
    # query whose content-bias should resolve 'it'->cat (cat is the brain's only known eater). `recalled_svo` is the
    # visible effect of which referent 'it' resolved to.
    ("selective-attention-biased-competition", "bc_b", ["recalled_svo"], False),
]


def faculty_list():
    return [f[0] for f in FACULTY_PROBES]


# ── worker: build ONE fresh brain with a given env, run the probe turns, dump responses ──────────────────────────
def _collect_worker(env_json, turn_labels, out_path):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        # OFF-ARM DISCIPLINE: an explicit value ("0"/"1"), never a pop -> the OFF arm stays OFF post-flip.
        os.environ[k] = v
    from webapp.server import brain_chat, BrainChatRequest
    responses = {}
    for label in turn_labels:
        if label in _WORLD_STEPS:              # label-only environment step (never a brain_chat turn)
            try:
                responses[label] = _run_world_step(_WORLD_STEPS[label])
            except Exception as e:
                responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
            continue
        _, msg, session, reset, percept, rich = _TURN_BY_LABEL[label]
        try:
            kwargs = dict(session=session, message=msg, brain="tiny-demo",
                          renderer="stub", rich=bool(rich), reset=reset)
            if percept is not None:
                kwargs["percept"] = percept
            r = brain_chat(BrainChatRequest(**kwargs))
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(responses, open(out_path, "w"), indent=2, default=str)
    print("[battery worker] env=%s -> %d turns -> %s" % (env, len(responses), out_path), flush=True)
    return 0


def _spawn_arm(env, turn_labels, out_path):
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.onebrain_regression_battery",
                        "--worker", "--env", json.dumps(env), "--turns", ",".join(turn_labels),
                        "--out", out_path], env=dict(os.environ))
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    return json.load(open(out_path))


# ── the comparison: per-faculty decision equality across the two arms ────────────────────────────────────────────
def compare(on_responses, off_responses, faculties=None):
    """For each faculty, compare its DECISION fields (categorical only) between the ON and OFF arms.

    verdict per faculty: 'pass' (fields present in >=1 arm and equal), 'regressed' (a field differs),
    'not-exercised' (all fields absent/None in BOTH arms -> a thin probe the set does not drive)."""
    faculties = faculties or FACULTY_PROBES
    per, n_pass, n_regress, n_thin = [], 0, 0, 0
    for key, turn_label, fields, thin in faculties:
        on_r = (on_responses or {}).get(turn_label) or {}
        off_r = (off_responses or {}).get(turn_label) or {}
        diffs, any_present = [], False
        for path in fields:
            leaf = path.split(".")[-1]
            if leaf in _NOISE_FIELDS:
                continue                                   # never compare a continuous measurement
            on_present, on_val = _get_path(on_r, path)
            off_present, off_val = _get_path(off_r, path)
            if not on_present and not off_present:
                continue
            if (on_val is None) and (off_val is None):
                continue
            any_present = True
            if on_val != off_val:
                diffs.append({"field": path, "on": on_val, "off": off_val})
        if diffs:
            verdict = "regressed"; n_regress += 1
        elif not any_present:
            verdict = "not-exercised"; n_thin += 1
        else:
            verdict = "pass"; n_pass += 1
        per.append({"faculty": key, "turn": turn_label, "verdict": verdict, "thin_probe": thin, "diffs": diffs})
    return {
        "all_pass": (n_regress == 0),
        "n_faculties": len(faculties),
        "n_pass": n_pass, "n_regressed": n_regress, "n_not_exercised": n_thin,
        "regressed": [p["faculty"] for p in per if p["verdict"] == "regressed"],
        "not_exercised": [p["faculty"] for p in per if p["verdict"] == "not-exercised"],
        "per_faculty": per,
    }


# ── the production entry: flag ON vs flag OFF, through the real handler ───────────────────────────────────────────
def run_regression_battery(flag, out_dir="research/findings/raw/_regression_battery",
                           on_value="1", probe_subset=None, base_env=None):
    """Flip `flag` ON-vs-OFF and assert every default-ON faculty decides identically. Returns the compare() dict.

    NOTE this is a DECISION-STABILITY comparison (a metamorphic no-op-preservation relation), NOT a lesion-attribution
    experiment — it compares the flag's ON vs OFF arms, it does not compute a lesion control to attribute a difference
    to (that is ARM B's job in the harness). So it deliberately makes no `tools.lab.attributable_to` call."""
    os.makedirs(out_dir, exist_ok=True)
    labels = probe_subset or [t[0] for t in PROBE_TURNS]
    base = dict(base_env or {})
    on_env = dict(base); on_env[flag] = on_value
    off_env = dict(base); off_env[flag] = "0"           # EXPLICIT off (never pop)
    on_out = os.path.join(out_dir, "arm_on_%s.json" % flag)
    off_out = os.path.join(out_dir, "arm_off_%s.json" % flag)
    print("[battery] %s: ON(%s=%s) vs OFF(%s=0) over %d probe turns" % (flag, flag, on_value, flag, len(labels)),
          flush=True)
    on_resp = _spawn_arm(on_env, labels, on_out)
    off_resp = _spawn_arm(off_env, labels, off_out)
    # only compare faculties whose turn is in the subset
    facs = [f for f in FACULTY_PROBES if f[1] in labels]
    result = compare(on_resp, off_resp, faculties=facs)
    result["flag"] = flag
    result["probe_turns"] = labels
    result["arms_built"] = {"on": on_resp is not None, "off": off_resp is not None}
    json.dump(result, open(os.path.join(out_dir, "battery_%s.json" % flag), "w"), indent=2, default=str)
    return result


# ── the de-risk DEMO: a no-op flip -> all pass, AND a deliberately-broken probe -> caught ────────────────────────
def demo(no_op_flag="BRAIN_REGRESSION_BATTERY_NOOP", probe_subset=None, skip_real=False):
    """(1) real no-op flip -> every exercised faculty passes; (2) synthetic broken probe -> caught. Numpy/CPU.

    The no-op flip uses an UNUSED SENTINEL flag by default (nothing reads BRAIN_REGRESSION_BATTERY_NOOP), so the ON
    and OFF arms build byte-identically at the same seed and every exercised faculty MUST decide identically — a
    guaranteed-no-op that isolates the battery's real two-arm brain_chat plumbing + its all-pass reporting from any
    real faculty change. (In production the harness ARM C calls run_regression_battery with the REAL edge flag; a
    genuine answer-preserving flip like BRAIN_ONEBRAIN_MERGE also exercises it, at the cost that its RNG-trajectory
    shift can flip a borderline decision — which, if it happens, is a real finding the battery correctly surfaces.)

    Default `probe_subset=None` -> the FULL PROBE_TURNS roster (matching `run_regression_battery`'s own default,
    what the production flip-verify harness actually calls) — every default-ON faculty this file drives at all,
    not just the original 4-turn fast subset. Slower (more turns -> more organs to build once each); pass an
    explicit `probe_subset` for the old fast smoke (e.g. `["well", "unknown", "hold", "held"]`)."""
    out_dir = "research/findings/raw/_regression_battery"
    os.makedirs(out_dir, exist_ok=True)
    labels = probe_subset or [t[0] for t in PROBE_TURNS]
    report = {"no_op_flag": no_op_flag, "probe_turns": labels}

    if not skip_real:
        # (1) REAL no-op flip through the real handler.
        real = run_regression_battery(no_op_flag, out_dir=out_dir, probe_subset=labels)
        report["real_no_op"] = {k: real[k] for k in ("all_pass", "n_faculties", "n_pass", "n_regressed",
                                                      "n_not_exercised", "regressed", "not_exercised")}
        real_on = json.load(open(os.path.join(out_dir, "arm_on_%s.json" % no_op_flag)))
        real_off = json.load(open(os.path.join(out_dir, "arm_off_%s.json" % no_op_flag)))
    else:
        real_on = real_off = None

    # (2) SYNTHETIC broken-probe catch: take the ON arm as both arms (identical -> all pass), then deliberately
    # BREAK ONE faculty's decision field in the OFF copy and require compare() to flag exactly that faculty.
    if real_on is not None:
        base = real_on
    else:
        # no real arms (skip_real): synthesize a minimal well-turn response covering a few faculties.
        base = {"well": {"answer": "the wolf bites the apple.", "abstained": False, "recalled_svo": ["wolf", "bite", "apple"],
                         "verified": True, "comprehension": {"on": True, "comprehended": True},
                         "affect": {"on": True, "valence_sign": "0", "tone_token": ""},
                         "da_drives": {"on": True, "acted": True, "mode": "focus", "reason": "engaged"},
                         "activity": {"composer": "onebrain", "matched_fact_index": 5},
                         "noncontradiction": {"on": True, "reject": False, "recalled_yn": "unknown",
                                              "asserted_polarity": "AFFIRM"}},
                "unknown": {"answer": "I don't know about that.", "abstained": True},
                "hold": {}, "held": {"swap_drives": {"on": True, "acted": False, "swapped": False, "reason": "x"},
                                      "activity": {"roles": []}, "multiref": {"n_referents": 2}}}
    facs = [f for f in FACULTY_PROBES if f[1] in labels]
    identical = compare(base, base, faculties=facs)
    # break the affect faculty's valence_sign in a deep copy of the OFF arm
    broken = json.loads(json.dumps(base))
    tgt_faculty = "da-mode-drives-response"
    if "well" in broken and isinstance(broken["well"].get("da_drives"), dict):
        broken["well"]["da_drives"]["mode"] = "__BROKEN_MODE__"
    else:                                               # fallback: break the top-level answer on the well turn
        tgt_faculty = "content-selection"
        broken.setdefault("well", {})["answer"] = "__BROKEN_ANSWER__"
    caught = compare(base, broken, faculties=facs)
    report["synthetic_identical_all_pass"] = bool(identical["all_pass"])
    report["synthetic_broken_target"] = tgt_faculty
    report["synthetic_broken_caught"] = bool(not caught["all_pass"] and tgt_faculty in caught["regressed"])
    report["synthetic_broken_regressed_list"] = caught["regressed"]

    json.dump(report, open(os.path.join(out_dir, "battery_demo.json"), "w"), indent=2, default=str)
    print("\n===== REGRESSION BATTERY DEMO =====", flush=True)
    if "real_no_op" in report:
        r = report["real_no_op"]
        print("  REAL no-op flip (%s ON vs OFF): all_pass=%s  pass=%d regressed=%d not_exercised=%d"
              % (no_op_flag, r["all_pass"], r["n_pass"], r["n_regressed"], r["n_not_exercised"]), flush=True)
        if r["regressed"]:
            print("    REGRESSED: %s" % r["regressed"], flush=True)
    print("  SYNTHETIC identical->all_pass=%s ; broken(%s)->caught=%s (regressed=%s)"
          % (report["synthetic_identical_all_pass"], report["synthetic_broken_target"],
             report["synthetic_broken_caught"], report["synthetic_broken_regressed_list"]), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help="internal: build one arm + run the turns")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--turns", default="")
    ap.add_argument("--out", default="research/findings/raw/_regression_battery/arm.json")
    ap.add_argument("--flag", default=None, help="run the battery flipping this flag ON vs OFF")
    ap.add_argument("--demo", action="store_true", help="no-op-all-pass + broken-catch de-risk demo")
    ap.add_argument("--skip-real", action="store_true", help="demo: skip the real brain arms (synthetic-only)")
    ap.add_argument("--noop-flag", default="BRAIN_REGRESSION_BATTERY_NOOP", help="demo: the no-op flip flag")
    ap.add_argument("--subset", default=None, help="comma-separated probe turn labels to restrict to")
    args = ap.parse_args()
    subset = args.subset.split(",") if args.subset else None
    if args.worker:
        return _collect_worker(args.env, [t for t in args.turns.split(",") if t], args.out)
    if args.demo:
        demo(no_op_flag=args.noop_flag, probe_subset=subset, skip_real=args.skip_real)
        return 0
    if args.flag:
        r = run_regression_battery(args.flag, probe_subset=subset)
        print(json.dumps({k: r[k] for k in ("all_pass", "n_faculties", "n_regressed", "regressed",
                                            "n_not_exercised")}, indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
