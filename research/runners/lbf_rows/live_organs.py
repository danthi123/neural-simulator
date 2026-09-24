"""LOAD-BEARING-FRACTION ROWS for PRODUCTION-WIRED organs that already ship a dedicated `BRAIN_<X>_LESION` knob
but have no row in `research/runners/load_bearing_fraction.py::FACULTY_LESIONS`/`FACULTY_PROBES` yet.

2026-09-24 midnight plan, lane A1, step S12. See the pre-registration this module implements:
`research/findings/2026-09-24-lbf-rows-live-organs-PREREGISTRATION.md` (committed BEFORE this file, and again as
an amendment after the seed-7 smoke below).

INTERFACE (research/runners/lbf_rows/__init__.py): `EXTRA_LESIONS` / `EXTRA_PROBES` are merged into
`FACULTY_LESIONS`/`FACULTY_PROBES` by AG-REG's import hook (S08/S26) -- this module does NOT edit either literal
registry. `EXTRA_TURNS` (this module's own addition to the interface, mirroring the sibling A2 lane's
`research/runners/lbf_rows/learning.py`, 2026-09-24-learning-rows-d5-consolidate-sleep-replay-lbf-PREREGISTRATION
.md "added to `_EXTRA_TURNS`, NOT to `PROBE_TURNS`") carries the two NEW turn tuples this module needs; AG-REG's
hook folds it into `onebrain_regression_battery._EXTRA_TURNS` (label-only, so it resolves via `_TURN_BY_LABEL`
without growing the default `PROBE_TURNS` roster any existing runner iterates).

TEN CANDIDATE ROWS, per the S12 action list. Each was checked against source (the flag genuinely resolves in
`webapp/` or `research/runners/`, confirmed by `_flag_resolves`-equivalent grep -- see the PREREG's "flag
resolves" table) and reasoned from the organ's own docstring/wiring in `webapp/server.py` for which turn drives
it and which response field the lesion should flip. SIX register as `neural-lesion` (a genuine dedicated cut,
default-ON, a plausible driving turn identified); FOUR register `kind="thin"` after two probe-design attempts
each found no field this env-flag harness can drive without a capability this interface cannot express (a
companion enable-flag the single flag/value row shape has no slot for, or no observable decision field at all)
-- per the plan's own fallback ("a row still unexercisable after 2 probe designs is registered kind='thin' with
its reason"), never silently folded into "not load-bearing".

RESOURCE RESIDUAL (declared, not a design gap): the 2026-09-24 midnight plan dispatches ~13 concurrent agent
lanes on ONE 46 GB / 20-core box (`tools/parallel_audit.py`'s own compute-lanes map); at S12's build time this
box measured load average ~22 on 20 cores and 19+ GB swap in use from sibling lanes' own memcapped brain builds
(confirmed via `systemctl --user list-units --type=scope` showing concurrent scopes from A2/AG-REG/wm-focus-bind
runners; `ssh pool2` did not resolve from this worktree). A full-brain build under `tools/memcap.sh 2` therefore
spent minutes in D-state (uninterruptible I/O wait under system-wide swap pressure) rather than computing. The
"question"/gnw-bus row's INTACT arm was confirmed actually progressing (organ-by-organ build log advancing,
RSS growing normally) before contention stalled it again; no row's smoke reached a completed intact+lesion pair
locally in the time available to this lane. See the PREREG amendment for the exact staged commands (ready to run
the moment mem_ok.sh clears, or on pool2 once reachable) -- this module's row DESIGNS are not blocked on that,
only their FIRST empirical confirmation is, which is what B2b's 6-seed pool run is for regardless.
"""
from __future__ import annotations

# Attribution discipline (tools.lab, 2026-07-31): `assert_lesion_holds` below is exactly the treatment/control
# pair the gate class AT exists for (an intact arm vs a lesion arm) -- `lever()` makes the "did it actually MOVE"
# question execute instead of being eyeballed from two JSON blobs one key apart (the gap#5 lesson: two banked
# numbers with nobody subtracting them hid a 97%-clamp-owned effect for weeks).
from tools.lab import lever

# ── EXTRA_TURNS: the two NEW turn tuples this module's rows need (label, message, session, reset, percept, rich).
# Both use a session name namespaced to this module (never shared with PROBE_TURNS/_EXTRA_TURNS' own sessions),
# so merging is additive and cannot alter any existing turn-group's session history.
EXTRA_TURNS = [
    # affective-tom: a third-party affectively-charged situation -- the organ's OWN docstring worked example
    # (research/runners/affective_tom_production_organ.py `detect_other_agent`'s module comment: "Maria is
    # devastated"). Own fresh session, single turn (no dependency).
    ("lbf_tom1", "Maria is devastated", "lbf_tom", True, None, False),
    # causal-whatif: teach the canonical causal chain + confound THROUGH CHAT (the disjoint `_maybe_acquire`
    # 3-content-word acquisition path, matching the existing "well"/"unknown" teach convention -- NOT the
    # production verify script's direct `composer.store()`, which this brain_chat-only harness cannot reach),
    # then ask the validated what-if query. FACTS (research/runners/_causal_forward_model_grounded_derisk.py):
    # A=(dog,go,east) -> B=(dog,reach,river) -> D=(dog,drink,water) [the chain]; C=(sun,rise,sky) is the confound
    # common cause of X=(bird,sing,dawn) and Y=(dog,wake,morning) [not needed for what-if, taught for symmetry
    # with a future why-did row]. Verb surface forms are 3rd-person ("goes"/"reaches"/... ) so the SAME
    # store-write lemmatization the existing "hunts"->"hunt" fix already canonicalizes
    # (research/runners/brain_chat_tui.py _maybe_acquire) collapses them to the FACTS base lemmas; the patient
    # nouns already match verbatim. Own fresh session ('lbf_cau'), each teach turn is EXACTLY 3 content words
    # after the "the" stopword strip, matching the acquisition path's SVO-assertion gate.
    ("lbf_cau_teach1", "the dog goes east", "lbf_cau", True, None, False),
    ("lbf_cau_teach2", "the dog reaches river", "lbf_cau", False, None, False),
    ("lbf_cau_teach3", "the dog drinks water", "lbf_cau", False, None, False),
    ("lbf_cau_teach4", "the sun rises sky", "lbf_cau", False, None, False),
    ("lbf_cau_teach5", "the bird sings dawn", "lbf_cau", False, None, False),
    ("lbf_cau_teach6", "the dog wakes morning", "lbf_cau", False, None, False),
    ("lbf_cau_whatif", "what happens if the dog goes east", "lbf_cau", False, None, False),
]

# ── EXTRA_LESIONS: same entry shape as FACULTY_LESIONS ──────────────────────────────────────────────────────────
EXTRA_LESIONS = {
    "self-schema": dict(
        flag="BRAIN_SELF_SCHEMA_LESION", value="1", kind="neural-lesion",
        note="webapp/server.py:6357-6381 (DR-3 authorship, default-ON since the wave-1/2 flip). Under lesion the "
             "author-pool access is severed (schema_access=False) -> the read collapses to 'heard' regardless of "
             "the turn actually being a self-generated hypothesis. Only fires on an is_hyp turn (a generated "
             "hypothesis), so the driving turn is 'rich_open' (open-ended, rich=True), already in PROBE_TURNS."),
    "affective-tom": dict(
        flag="BRAIN_AFFECTIVE_TOM_LESION", value="1", kind="neural-lesion",
        note="research/runners/affective_tom_production_organ.py (W5, default-ON since 2026-08-26). "
             "BRAIN_AFFECTIVE_TOM_GRADED defaults ON (unset->'1'), so observe_turn() takes the GRADED-circumplex "
             "branch and populates 'tone_level'/'reason', NOT the bistable 'tone_sign' (which stays at its "
             "unused init value 0 on that branch) -- the compared field must be 'reason'/'tone_level', not "
             "'tone_sign'. On 'Maria is devastated' (a genuine third-party negative-affect trigger, "
             "detect_other_agent's own worked example), intact reads reason='empathic' with a non-zero "
             "tone_level; BRAIN_AFFECTIVE_TOM_LESION=1 clamps the OTHER-model's affect_out -> the differential "
             "collapses -> reason='lesion_collapsed', tone_level back to the neutral bucket."),
    "causal-whatif": dict(
        flag="BRAIN_CAUSAL_LESION", value="1", kind="neural-lesion",
        note="research/runners/causal_whatif_production_organ.py (T1-4, default-ON). Zeroing the learned forward "
             "edges removes the A->B->D rollout, so the moat-confirmed what-if consequence can no longer be "
             "produced and the turn collapses to the honest _honest_causal_answer abstain. Driving turn "
             "'lbf_cau_whatif', reached only after the chain is taught through chat (EXTRA_TURNS above); compare "
             "'abstained' + 'causal.confirmed'. RESIDUAL (declared, not independently confirmed by a completed "
             "local smoke -- see the module docstring's resource note): the production VERIFY script for this "
             "organ (research/runners/_causal_whatif_production_organ_verify.py) teaches via a direct "
             "`composer.store()` call, bypassing conversational acquisition; this row is the first design to "
             "drive the SAME organ through chat-only teaching, so the exact store-write path (verb "
             "lemmatization on 'goes'/'reaches'/'rises'/'sings'/'wakes', vocabulary code allocation for "
             "'east'/'river'/'sky'/'dawn'/'morning') has not yet been empirically confirmed end-to-end."),
    "spiking-anaphor": dict(
        flag="BRAIN_SPIKING_ANAPHOR_LESION", value="1", kind="neural-lesion",
        note="research/runners/spiking_anaphor_detection_organ.py. Since the 2026-09-16 scaffold-retirement the "
             "CA3 pattern-completion organ is the SOLE anaphor-token detector in "
             "brain_chat_tui.py::ChatBrain._is_anaphor_token (the host `tl in {...}` set test was DELETED, no "
             "flag needed to opt in -- BRAIN_SPIKING_ANAPHOR_LESION alone governs it). Reuses the EXISTING "
             "bc_a->bc_b pair already in PROBE_TURNS (selective-attention-biased-competition's own turns: "
             "'the cat and the ball walked in' then the pronoun query 'what does it eat'); under lesion "
             "(attractor_weight=0.0) 'it' no longer completes to an ignited assembly -> is_anaphor('it')=False -> "
             "the pronoun is never substituted with the held referent -> the query cannot resolve an "
             "(agent,action) pair -> abstain, vs intact's resolved recall. Compare 'abstained' + 'recalled_svo'."),
    "gnw-bus": dict(
        flag="BRAIN_GNW_BUS_LESION", value="1", kind="neural-lesion",
        note="webapp/gnw_bus_shadow.py (default-ON combination since 2026-08-13's flip finding "
             "2026-08-13-gnw-bus-default-flip-substrate-authors-organ-combination.md). That finding's own "
             "production verify already showed the load-bearing effect on the ANSWER (not just the debug trace "
             "BRAIN_GNW_BUS=1 attaches): 'with the bus authoring, a lesion ... makes the combined ANSWER "
             "collapse to abstain (intact commits the patient), while the forward-recall reflex ... still "
             "returns it.' Reuses the EXISTING 'question' turn ('what does the wolf bite', a stored build-time "
             "fact, already in PROBE_TURNS) -- no BRAIN_GNW_BUS=1 needed, since the field compared is the "
             "top-level 'abstained' the bus's combination decides directly, not the opt-in observability block."),
    "multiref-competition": dict(
        flag="BRAIN_MULTIREF_COMPETITION_LESION", value="1", kind="neural-lesion",
        note="research/runners/d6_multiref_wm_production_organ.py (board #196, default-ON since 2026-09-02). "
             "Reuses the EXISTING 'hold' turn ('the fox and the wolf walked in', already in PROBE_TURNS as the "
             "first turn of the wm-binding-advanced hold->held group -- turn_group('hold') is ['hold'] alone, so "
             "no dependency on 'held'). Under lesion every referent routes to register 0 regardless of the "
             "occupancy probe -> fox+wolf collide in one register (the already-validated superposed-collide "
             "regime) -> 'multiref.all_recovered' flips True (intact, distinct registers) -> False (lesioned, "
             "collision) on the SAME turn multiref_info already populates for the existing wm-binding-advanced "
             "row (whose own field, n_referents, is a COUNT and does not distinguish this lesion -- this row "
             "adds the complementary field the count cannot show)."),
    "affect-appraisal-interoceptive": dict(
        flag="BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION", value="1", kind="neural-lesion",
        note="research/runners/affect_production_organ.py (board #49/#81 pattern, default-ON since the "
             "2026-09-05 flip). webapp/server.py:4842 calls `organ.read_differential(mood['valence'], "
             "lesion=affect_lesioned())`, which internally passes `intero_lesion=appraisal_interoceptive_"
             "lesioned()` as a SEPARATE, independently-settable cut (distinct from the existing affect-coloring "
             "row's BRAIN_AFFECT_LESION, which clamps the ladder's OWN affect_out gate). Reuses the EXISTING "
             "'emo' turn (already in PROBE_TURNS for affect-marker-spiking-wta -- a strongly-affective message "
             "so the appraisal is genuinely non-neutral, unlike the affect-coloring row's neutral 'well' turn). "
             "Compare 'affect.valence_sign' ('+' intact under a positive appraisal -> '0' when the relay->ladder "
             "synapse is cut and the ladder never receives the interoceptive drive)."),
    # ── kind="thin": a dedicated lesion flag exists and resolves in source, but after two probe designs each,
    # no turn/field this SINGLE-flag row harness can express reaches a genuine categorical decision diff. Per the
    # plan's own fallback: registered honestly, never folded into a false "not load-bearing" (a flag that never
    # got the chance to bite would read identically on both arms and silently count as "pass").
    "spiking-qroute": dict(
        flag="BRAIN_SPIKING_QROUTE_LESION", value="1", kind="thin",
        note="research/runners/spiking_qroute_selection_organ.py. Since 2026-09-16 the sole DISPATCH among the "
             "four comprehension routes (RELFRONT/KBREL/DEFCOP/GENERIC) in "
             "brain_chat_tui.py::ChatBrain._spiking_route_decision -- no companion enable flag, so the lesion "
             "alone should in principle be drivable. DESIGN 1 (RELFRONT/KBREL): both need a Wikidata-style "
             "underscored KB relation (research/runners/_knowledge_core_curate.py's curated core); the tiny-demo "
             "brain_chat build carries no such KB, so `_relf`/`_kbrel` are always None -- these two candidates "
             "are structurally unreachable through this harness. DESIGN 2 (DEFCOP-vs-GENERIC): 'what is a dog' "
             "makes `_defo=['dog','isa']` non-None (content<=1) with `_relf`/`_kbrel` both None, so DEFCOP is the "
             "lone non-GENERIC candidate -- but the WTA decision itself is never attached to any response field "
             "(no route-name key exists anywhere in `resp`), and since 'dog isa ?' was never taught either way, "
             "both the DEFCOP path (isa-relation lookup) and a lesioned fallback to GENERIC (positional "
             "agent-only parse) plausibly ABSTAIN identically -- not confirmed to diverge on any compared field. "
             "A genuine probe needs either a response field naming the winning route (a harness capability this "
             "single-flag interface has no way to request) or a taught 'X isa Y' fact whose GENERIC-path "
             "interpretation would answer DIFFERENTLY from its DEFCOP-path interpretation, which the current "
             "PROBE_TURNS/EXTRA_TURNS vocabulary does not construct."),
    "learned-referent": dict(
        flag="BRAIN_LEARNED_REFERENT_LESION", value="1", kind="thin",
        note="research/runners/d6_multiref_wm_production_organ.py `learned_referent_lesioned()`. Genuinely "
             "load-bearing ONLY when its companion enable flag `BRAIN_LEARNED_REFERENT_LEXICON=1` is ALSO set "
             "(default OFF -- `_flag_learned_referent_lexicon()` returns None immediately when unset, making the "
             "lesion flag a no-op under the shipped default). The single flag/value `EXTRA_LESIONS` row shape "
             "has no slot for a companion env var, so this row cannot express its own trigger condition through "
             "the generic harness without silently reading 'pass' (both arms out-of-scope, n_referents=1, "
             "identical) -- exactly the false-negative the module's own docstring warns a no-op lesion produces. "
             "DESIGN 1 (companion env baked into this dict as an ad-hoc extra key): rejected -- the interface "
             "contract is the exact FACULTY_LESIONS shape, and an unhonored extra key is worse than an honest "
             "thin (silent no-op if the merge hook ignores it). DESIGN 2 (measure via the dedicated mechanism "
             "PREREG instead): the concurrent, already-preregistered "
             "research/findings/2026-09-24-d6-multiref-wm-learned-referent-env-flag-route-PREREGISTERED.md "
             "(a sibling lane, branch research/wm-referent-focus-bind) already measures this EXACT flag pair "
             "properly with BOTH env vars set, over a validated held-out word ('owl') and a 6-seed population "
             "gate -- the correct, already-in-flight instrument for this capability; this LBF row should be "
             "revisited once (if) `BRAIN_LEARNED_REFERENT_LEXICON` itself ships default-ON, at which point a "
             "single-flag row becomes expressible."),
    "onebrain-xedge": dict(
        flag="BRAIN_ONEBRAIN_XEDGE_LESION", value="1", kind="thin",
        note="research/runners/onebrain_xedge_production.py (default-ON since 2026-08-28: BRAIN_ONEBRAIN_XEDGE "
             "and BRAIN_ONEBRAIN_XEDGE_LEARN both default True, so unlike learned-referent this needs NO "
             "companion flag). DESIGN 1: `comprehension.xedge_live_learn` (attached on 'held' after 'hold', the "
             "existing wm-binding-advanced hold->held group) is a per-turn CREDIT trace, not the lesion's target "
             "-- `credit_live_turn_from_comprehension`'s early-return conditions gate on xedge_enabled/"
             "xedge_learn_enabled only, so the trace is PRESENT under lesion too, and its 'direction' field is "
             "the sign of the held referent's own per-noun agent-evidence (a0), which is a property of the SVO "
             "structure itself, not of the cross-edge -- not expected to flip under this lesion. DESIGN 2: the "
             "cross-edge's real target is the shared d6-WM->comprehension MARGIN (whether `comprehension.on`/"
             "`comprehended` clears the confidence threshold) on a near-borderline OOV case, which needs a "
             "probe deliberately tuned to sit AT the pre-cross-edge threshold so the xedge nudge is what tips "
             "it -- the existing 'animacy'/'verbsel' OOV turns are tuned for the LEARNED-CUE rows, not this "
             "margin, and neither was confirmed (nor refuted) to sit at the right operating point without a "
             "run this lane's compute could not complete (see the module docstring's resource note). Left thin "
             "rather than guessed at."),
}

# ── EXTRA_PROBES: same entry shape as FACULTY_PROBES (a list of (key, turn_label, fields, thin) 4-tuples) ────────
# PARKED (2026-09-24, Amendment 3 of research/findings/2026-09-24-lbf-rows-live-organs-PREREGISTRATION.md): the
# seed-7 smoke raised KeyError('authorship') -- the registered turn `rich_open` never attaches an `authorship` block in
# the live brain_chat response, so this row's lesion check reads nothing and its integrity note describes a different
# turn (`is_hyp`). The row stays here for the record; the registry hook leaves it out until it is redesigned.
PARKED = {
    "self-schema": "retracted by Amendment 3: probe turn rich_open never attaches 'authorship' (KeyError at seed 7)",
}

EXTRA_PROBES = [
    ("self-schema", "rich_open", ["authorship.is_self", "authorship.label"], False),
    ("affective-tom", "lbf_tom1", ["affective_tom.reason", "affective_tom.tone_level"], False),
    ("causal-whatif", "lbf_cau_whatif", ["abstained", "causal.confirmed"], False),
    ("spiking-anaphor", "bc_b", ["abstained", "recalled_svo"], False),
    ("gnw-bus", "question", ["abstained"], False),
    ("multiref-competition", "hold", ["multiref.all_recovered"], False),
    ("affect-appraisal-interoceptive", "emo", ["affect.valence_sign"], False),
    # thin rows still get a probe entry (matching the existing gnw-deliberation/value-driven-choice convention in
    # FACULTY_PROBES: thin=True, a placeholder turn/field so `_faculty_row()` resolves and the row appears in the
    # coverage table with an honest not-covered verdict rather than being invisible).
    ("spiking-qroute", "well", ["activity.composer"], True),
    ("learned-referent", "held", ["multiref.n_referents"], True),
    ("onebrain-xedge", "held", ["comprehension.xedge_live_learn"], True),
]


# ── per-row DATA-CHECK the lesion "held" (the read-time assertion the plan asks for; see the module docstring's
#    resource residual for why this runs on the SMOKE's two artifacts rather than as an in-process runtime assert
#    -- each arm is a fresh subprocess build, exactly the constraint the sibling A2/learning.py row hit first) ──
def assert_lesion_holds(key: str, intact: dict, lesion: dict) -> dict:
    """Given the two arms' response dict for this row's turn (already reduced to the turn label the row probes),
    return {"held": bool, "moved": bool|None, "detail": str} -- a DATA check, not a claim. Never raises; a
    KeyError/TypeError reading an unexpected shape is reported as held=False with the exception text, never
    silently swallowed as True. `lever(..., required=False)` (tools.lab) makes the intact-vs-lesion MOVE itself
    an executed check (not just an eyeballed diff of two dicts one key apart) alongside the row's own directional
    expectation (`held`, e.g. "abstained flips False->True", which a bare 'moved' cannot distinguish from a move
    in the WRONG direction)."""
    try:
        if key == "self-schema":
            moved = lever("self-schema authorship.label", intact["authorship"]["label"],
                          lesion["authorship"]["label"], required=False)
            return {"held": intact["authorship"]["label"] == "self" and lesion["authorship"]["label"] == "heard",
                    "moved": moved,
                    "detail": "intact.label=%r lesion.label=%r" % (
                        intact["authorship"]["label"], lesion["authorship"]["label"])}
        if key == "affective-tom":
            moved = lever("affective-tom affective_tom.reason", intact["affective_tom"]["reason"],
                          lesion["affective_tom"]["reason"], required=False)
            return {"held": intact["affective_tom"]["reason"] == "empathic"
                            and lesion["affective_tom"]["reason"] == "lesion_collapsed",
                    "moved": moved,
                    "detail": "intact.reason=%r lesion.reason=%r" % (
                        intact["affective_tom"]["reason"], lesion["affective_tom"]["reason"])}
        if key == "causal-whatif":
            moved = lever("causal-whatif abstained", intact["abstained"], lesion["abstained"], required=False)
            return {"held": (intact["abstained"] is False and intact["causal"]["confirmed"] is True
                            and lesion["abstained"] is True and lesion["causal"]["confirmed"] is not True),
                    "moved": moved,
                    "detail": "intact.abstained=%r lesion.abstained=%r" % (intact["abstained"], lesion["abstained"])}
        if key == "spiking-anaphor":
            moved = lever("spiking-anaphor abstained", intact["abstained"], lesion["abstained"], required=False)
            return {"held": intact["abstained"] is False and lesion["abstained"] is True, "moved": moved,
                    "detail": "intact.abstained=%r lesion.abstained=%r" % (intact["abstained"], lesion["abstained"])}
        if key == "gnw-bus":
            moved = lever("gnw-bus abstained", intact["abstained"], lesion["abstained"], required=False)
            return {"held": intact["abstained"] is False and lesion["abstained"] is True, "moved": moved,
                    "detail": "intact.abstained=%r lesion.abstained=%r" % (intact["abstained"], lesion["abstained"])}
        if key == "multiref-competition":
            moved = lever("multiref-competition multiref.all_recovered", intact["multiref"]["all_recovered"],
                          lesion["multiref"]["all_recovered"], required=False)
            return {"held": intact["multiref"]["all_recovered"] is True
                            and lesion["multiref"]["all_recovered"] is False,
                    "moved": moved,
                    "detail": "intact.all_recovered=%r lesion.all_recovered=%r" % (
                        intact["multiref"]["all_recovered"], lesion["multiref"]["all_recovered"])}
        if key == "affect-appraisal-interoceptive":
            moved = lever("affect-appraisal-interoceptive affect.valence_sign", intact["affect"]["valence_sign"],
                          lesion["affect"]["valence_sign"], required=False)
            return {"held": intact["affect"]["valence_sign"] != "0" and lesion["affect"]["valence_sign"] == "0",
                    "moved": moved,
                    "detail": "intact.valence_sign=%r lesion.valence_sign=%r" % (
                        intact["affect"]["valence_sign"], lesion["affect"]["valence_sign"])}
        return {"held": None, "moved": None, "detail": "kind=thin -- no data-check registered (see EXTRA_LESIONS note)"}
    except Exception as e:
        return {"held": False, "moved": None, "detail": "%s: %s" % (type(e).__name__, e)}


# ── blocking pass-by-construction audit (per the plan: "if the lesion removes the reply template's only input,
#    the row is an integrity smoke, excluded"). Checked by REASONING here (no completed local run to check it
#    against, see the resource residual): for every neural-lesion row above, the response block that carries the
#    compared field(s) is ALWAYS attached when the organ's own default-ON master switch is on and the turn is
#    in-scope -- the LESION never removes the block itself (an info dict is always built, e.g. `affective_tom`'s
#    inert no-lead info, `authorship` on every is_hyp turn, `causal` on every matched causal query, `abstained` on
#    every turn) -- only an INTERNAL field WITHIN that block changes. None of these six rows is an integrity
#    smoke (the compared field is not the reply template's only input; the reply's OTHER content -- the recalled
#    fact, the base answer string structure -- stays present under lesion too, only the specific decision flips).
INTEGRITY_SMOKE_AUDIT = {
    "self-schema": "not an integrity smoke: 'authorship' is attached on every is_hyp turn regardless of lesion; "
                   "only 'label'/'is_self' inside it changes.",
    "affective-tom": "not an integrity smoke: observe_turn() always returns an info dict (even the inert "
                     "no-lead case carries 'acted'/'reason'); the lesion only changes 'reason'/'tone_level'.",
    "causal-whatif": "not an integrity smoke: the causal block always returns EITHER a confirmed consequence OR "
                     "the honest _honest_causal_answer disclaimer -- both are real, non-empty replies; the "
                     "lesion selects which of the two, it does not remove the block.",
    "spiking-anaphor": "not an integrity smoke: bc_b's comprehension path always runs (a normal question); the "
                       "lesion changes only whether the pronoun resolves before comprehension is attempted.",
    "gnw-bus": "not an integrity smoke: chat.gate() always runs and always returns SOME (agent,action) combination "
              "(bus-authored or, structurally, the same shape the host cascade produced) -- the lesion changes "
              "which combination it commits to, not whether one is attempted.",
    "multiref-competition": "not an integrity smoke: the multiref MAINTAIN block always runs on a >=2-referent "
                            "turn; the lesion changes the register-allocation decision inside it, not whether "
                            "referents are loaded at all.",
    "affect-appraisal-interoceptive": "not an integrity smoke: the affect ladder read always runs on every turn "
                                      "(default-ON); the lesion cuts one specific afferent synapse, not the read.",
}
