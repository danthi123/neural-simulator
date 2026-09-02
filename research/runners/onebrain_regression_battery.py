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
