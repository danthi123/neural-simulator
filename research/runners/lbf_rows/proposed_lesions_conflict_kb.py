"""Plan step S14 (midnight plan, 2026-09-24), lane A11: five default-inert lesion knobs converting
`FACULTY_LESIONS` rows content-selection / semantic-recall / moat-verify / in-loop-learning / discourse-planner /
selective-attention-biased-competition from in-process/proposed to an env-driven neural-lesion, plus an
LB_CONFLICT_KB_PROBE fixture-state check for gnw-deliberation / value-driven-choice.

PRE-REGISTRATION: research/findings/2026-09-24-lbf-proposed-lesion-knobs-conflict-kb-PREREGISTRATION.md
(committed before this module and before any run it governs).

LBF ROW INTERFACE: this module exposes `EXTRA_LESIONS` (dict, same shape as
`load_bearing_fraction.FACULTY_LESIONS`) and `EXTRA_PROBES` (list, same shape as
`onebrain_regression_battery.FACULTY_PROBES`) for AG-REG's import hook to merge. It does NOT edit
`FACULTY_LESIONS`/`FACULTY_PROBES` directly; the smoke functions below merge a LOCAL COPY in-process (restored
after use) purely to reuse `load_bearing_fraction.measure_faculty`'s existing, validated per-faculty measurement
machinery for this lane's own pre-merge smoke -- the exact "REUSE, NOT REINVENT" convention that module's own
docstring states.

Run (seed 7 is a DEV/SMOKE seed only -- CLAUDE.md; never a validation seed):
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -m research.runners.lbf_rows.proposed_lesions_conflict_kb \
        --smoke --out research/findings/raw/_lbf_rows_conflict_kb/smoke_s7.json
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -m research.runners.lbf_rows.proposed_lesions_conflict_kb \
        --conflict-kb --out research/findings/raw/_lbf_rows_conflict_kb/conflict_kb_probe_s7.json
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -m research.runners.lbf_rows.proposed_lesions_conflict_kb \
        --byte-identity --out research/findings/raw/_lbf_rows_conflict_kb/byte_identity_s7.json
Wrap every invocation: `bash tools/mem_ok.sh <n> 4 && bash tools/memcap.sh <n> -- <command above>` (RAM-tight box).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

from tools.lab import lever, void_if   # noqa: E402  (after the env defaults above, this codebase's convention)


# ── EXTRA_LESIONS (merged into FACULTY_LESIONS by AG-REG's import hook; not edited directly here) ─────────────────
EXTRA_LESIONS = {
    # ── the four already-driving FACULTY_PROBES rows (no new turn needed): upgraded in-process -> neural-lesion ──
    "content-selection": dict(
        flag="BRAIN_SUBSTRATE_PARSE_LESION", value="1", kind="neural-lesion",
        note="brain_chat_tui.py::ChatBrain._neural_question_parse -- BRAIN_SUBSTRATE_PARSE_LESION=1 skips the "
             "on-brain BridgeParser.role_of (position,voice)->role read entirely (returns None instead of "
             "sampling it), the same cut _production_lesion_probe already applies in-process. Measured on the "
             "existing 'well' turn (FACULTY_PROBES, unchanged)."),
    "semantic-recall": dict(
        flag="BRAIN_COMPOSER_RECALL_LESION", value="1", kind="neural-lesion",
        note="one_brain_composer.py::OneBrainComposer.query_patient -- BRAIN_COMPOSER_RECALL_LESION=1 returns "
             "None before the spiking K-way block-selection runs, mirroring _production_lesion_probe's own "
             "composer.query_patient monkeypatch. Measured on the existing 'well' turn (FACULTY_PROBES, unchanged)."),
    "moat-verify": dict(
        flag="BRAIN_SUBSTRATE_RECALL_LESION", value="1", kind="neural-lesion",
        note="brain_chat_tui.py::ChatBrain._substrate_recall -- BRAIN_SUBSTRATE_RECALL_LESION=1 returns None "
             "unconditionally, mirroring _production_lesion_probe's chat._substrate_recall monkeypatch. Measured "
             "on the existing 'unknown' turn (FACULTY_PROBES, unchanged)."),
    "in-loop-learning": dict(
        flag="BRAIN_SUBSTRATE_RECALL_LESION", value="1", kind="neural-lesion",
        note="SAME flag + cut as moat-verify (FACULTY_LESIONS's own note already names the identical minimal add "
             "for both keys: brain_chat_tui.py::ChatBrain._substrate_recall). Measured on the existing 'well' turn "
             "(FACULTY_PROBES, unchanged)."),
    "discourse-planner": dict(
        flag="BRAIN_DISCOURSE_PLANNER_LESION", value="1", kind="neural-lesion",
        note="rich_answer_composer.py::NeuralDiscoursePlanner.ordered_associates -- BRAIN_DISCOURSE_PLANNER_LESION=1 "
             "returns [] unconditionally, the SAME cut the burndown-3G de-risk already applies by monkeypatch "
             "(planner.ordered_associates = lambda topic, avoid=(): []). Measured on the existing 'rich_well' turn "
             "(FACULTY_PROBES, unchanged)."),
    # ── the one HONEST NEGATIVE: wired + isolation-verified, but NOT yet exercised by the shipped bc_b probe ──────
    "selective-attention-biased-competition": dict(
        flag="BRAIN_BIASED_COMPETITION_LESION", value="1", kind="proposed",
        note="_gap3_spiking_feature_compat_derisk.py::SpikingFeatureCompat.bias_target -- BRAIN_BIASED_COMPETITION_"
             "LESION=1 returns None unconditionally (the weights-cleared twin). VERIFIED in isolation, seed 7 "
             "(research/findings/raw/_lbf_rows_conflict_kb/biased_competition_isolated_s7.json): unlesioned "
             "bias_target(['worm','rock'], 'eat') = 'worm'; lesioned = None (abstain); tools.lab.lever confirms "
             "MOVED. Kept kind='proposed' (NOT upgraded to neural-lesion) "
             "because the tiny-demo brain every production build site constructs hears only 5 baked-in facts, "
             "below build_referent_bias_from_experience's own min_facts=40 floor (multi_turn_agent.py:299-310; "
             "brain_chat_tui.py:2081-2083's own comment already names this exact gap) -- agent._feat_compat_source "
             "stays None on the shipped 'bc_b' probe and MultiTurnAgent._resolve_biased falls through to "
             "_focus_bias_source instead, which this flag does not touch. Running this flag through the standard "
             "bc_b probe as-is would read a FALSE not-load-bearing (hollow), not a true negative -- declared, not "
             "shipped. Next rung: a driving-remap probe (mirroring LB_EPISODIC_DRIVE_PROBE's pattern) that first "
             "teaches >=40 facts on a fresh session before bc_b."),
    # ── Phase 2: note-only amendment (flag/kind UNCHANGED from the existing FACULTY_LESIONS entries) ─────────────
    "gnw-deliberation": dict(
        flag="BRAIN_GNW_DELIBERATE_LESION", value="1", kind="thin",
        note="UNCHANGED lesion knob/kind. LB_CONFLICT_KB_PROBE (this module, --conflict-kb) confirms the >=2-"
             "distinct-patient (agent,action) conflict the standard 'well' probe cannot reach IS directly "
             "constructible by reusing _value_choice_flip_soak._build_chat's own dog->chase->{cat,ball} "
             "construction (inner.hear bypassing conversational teaching) -- a fixture-state case: production "
             "chat cannot reach this KB state through ordinary teaching, but a directly-constructed composer can. "
             "Stays kind='thin' (out of load_bearing_fraction's numerator/denominator, unchanged code path); the "
             "construction is cited here as the working next-rung starting point, not wired into measure_faculty."),
    "value-driven-choice": dict(
        flag="BRAIN_VALUE_CHOICE_LESION", value="1", kind="thin",
        note="UNCHANGED lesion knob/kind; same fixture-state reasoning as gnw-deliberation above. The identical "
             "construction already has a passing 6-seed soak (research/findings/raw/_value_choice_prodflip/"
             "soak_summary_6seed.json, verdict GO) through this exact organ, at the organ-soak level (not through "
             "load_bearing_fraction's own per-faculty dispatch, which is what stays kind='thin' here)."),
}

# All six keys above already carry a real, non-thin FACULTY_PROBES row (content-selection/semantic-recall/
# in-loop-learning: 'well'; moat-verify: 'unknown'; discourse-planner: 'rich_well') or a deliberately-thin one
# (gnw-deliberation/value-driven-choice: 'well', thin=True, unchanged) -- no new turn is needed, so EXTRA_PROBES
# is empty. (Keeping the name for interface-shape parity with the sibling lbf_rows modules.)
EXTRA_PROBES = []


# ── Phase 1 smoke: reuse load_bearing_fraction.measure_faculty over a LOCAL-COPY-merged FACULTY_LESIONS ───────────
_FIVE_DRIVING_KEYS = ["content-selection", "semantic-recall", "moat-verify", "in-loop-learning", "discourse-planner"]


def smoke_five_lesions(seed=7, out_dir="research/findings/raw/_lbf_rows_conflict_kb"):
    """Build the SAME per-faculty measurement `load_bearing_fraction.measure_faculty` runs in production, with a
    LOCAL, restored-after-use merge of this module's EXTRA_LESIONS into `FACULTY_LESIONS` (mirrors what AG-REG's
    import hook will do permanently once merged; this lane does not wait on that merge to smoke its own rows).
    Shares one `intact_cache` across the 5 keys so faculties on the SAME turn ('well': content-selection,
    semantic-recall, in-loop-learning) reuse one intact arm pair instead of rebuilding it three times."""
    from research.runners import load_bearing_fraction as lbf
    orig = lbf.FACULTY_LESIONS
    patched = dict(orig)
    patched.update(EXTRA_LESIONS)
    lbf.FACULTY_LESIONS = patched
    results = {}
    lever_checks = {}
    try:
        cache = {}
        for key in _FIVE_DRIVING_KEYS:
            res = lbf.measure_faculty(key, out_dir, repeats=1, intact_cache=cache, seed=seed)
            results[key] = res
            # READ-TIME ASSERTION (tools.lab.lever): the lesion must actually have moved something -- a lesion that
            # cannot bite is the exact false-negative this codebase's own instrument-failure lessons warn against.
            # `required=False` here (not raise-on-fail) because the smoke's JOB is to REPORT which knobs bite, not
            # to abort on the first one that (unexpectedly) does not; the printed MOVED/UNCHANGED tag + the raw
            # before/after are what get recorded, and the caller (main()) still fails loudly if a key expected to
            # bite reads UNCHANGED.
            moved = lever(key, before=0, after=res.get("treatment_diffs", 0) or 0,
                          required=False, continuous=res.get("diffs"))
            lever_checks[key] = {"moved": bool(moved), "treatment_diffs": res.get("treatment_diffs"),
                                 "control_diffs": res.get("control_diffs"), "verdict": res.get("verdict"),
                                 "load_bearing": res.get("load_bearing"), "diffs": res.get("diffs")}
    finally:
        lbf.FACULTY_LESIONS = orig
    return {"seed": seed, "keys": _FIVE_DRIVING_KEYS, "results": results, "lever_checks": lever_checks}


def smoke_biased_competition_isolated(seed=7):
    """Isolated proof that BRAIN_BIASED_COMPETITION_LESION correctly cuts SpikingFeatureCompat.bias_target, built
    DIRECTLY (bypassing the tiny-demo's own min_facts=40 floor -- see EXTRA_LESIONS's note) so the mechanism is
    verified even though the shipped bc_b probe cannot yet reach it. Uses the SAME corpus/candidate shape the
    faculty's own de-risk (`_gap3_spiking_feature_compat_derisk.main`) validates against, at seed 7."""
    from research.runners._gap3_spiking_feature_compat_derisk import SpikingFeatureCompat, ANIMATE, INANIM, VERBS
    import numpy as np
    fc = SpikingFeatureCompat(seed=seed)
    rng = np.random.default_rng(seed)
    # find a (verb, [animate, inanimate]) pair the UNLESIONED chooser actually resolves (matches the de-risk's own
    # search loop, `run_seed` in that module) -- an abstain-by-construction pair would make this smoke vacuous.
    unlesioned = lesioned = None
    verb_used = cand_used = None
    os.environ.pop("BRAIN_BIASED_COMPETITION_LESION", None)
    for verb in VERBS:
        for _ in range(8):
            a = rng.choice(ANIMATE); i = rng.choice(INANIM)
            cands = [a, i]; rng.shuffle(cands)
            top = fc.bias_target(cands, verb)
            if top is not None:
                unlesioned, verb_used, cand_used = top, verb, list(cands)
                break
        if unlesioned is not None:
            break
    os.environ["BRAIN_BIASED_COMPETITION_LESION"] = "1"
    try:
        lesioned = fc.bias_target(cand_used, verb_used) if cand_used is not None else fc.bias_target(
            [ANIMATE[0], INANIM[0]], VERBS[0])
    finally:
        os.environ.pop("BRAIN_BIASED_COMPETITION_LESION", None)
    moved = lever("selective-attention-biased-competition:isolated", before=unlesioned, after=lesioned,
                  required=(unlesioned is not None))
    return {"seed": seed, "verb": verb_used, "candidates": cand_used, "unlesioned_choice": unlesioned,
            "lesioned_choice": lesioned, "moved": bool(moved),
            "note": "isolated proof only -- NOT exercised by the shipped bc_b probe on the tiny-demo build; "
                    "see EXTRA_LESIONS['selective-attention-biased-competition']"}


# ── Phase 2: LB_CONFLICT_KB_PROBE (fixture-state; gnw-deliberation / value-driven-choice) ──────────────────────────
def conflict_kb_probe(seed=7):
    """Reuses `_value_choice_flip_soak._build_chat`'s own construction (import, not reimplementation): build the
    tiny-demo brain, then `inner.hear('dog chase ball', polarity='AFFIRM')` DIRECTLY on the agent (bypassing
    conversational teaching / reconsolidation entirely, exactly as that soak's own docstring describes) to leave
    TWO distinct stored patients under (dog, chase). Confirms the conflict is visible in the composer's own KB
    dump, and -- where the optional GNW deliberation / value-choice installs succeed -- that the ambiguous
    question routes into them instead of a random single-patient recall."""
    from research.runners._value_choice_flip_soak import _build_chat, AMBIGUOUS, _answer, TRIGGER_TURN
    chat, VC, ctx_holder = _build_chat(seed, composer_kind="onebrain")
    inner = getattr(chat.inner, "agent", chat.inner)
    facts = [tuple(f) for f in chat.list_facts()]
    matching = [f for f in facts if f[0] == AMBIGUOUS[0] and f[1] == AMBIGUOUS[1]]
    distinct_patients = sorted({f[2] for f in matching})
    conflict_visible = len(distinct_patients) >= 2 and set(distinct_patients) <= set(AMBIGUOUS[2])
    trigger_reply = _answer(chat, TRIGGER_TURN)
    trigger_names_stored_patient = any(p in trigger_reply for p in AMBIGUOUS[2])
    # organ reachability: did the optional deliberation install succeed (`_build_chat` wraps it in try/except, so
    # its absence degrades silently there); value-choice is unconditional in `_build_chat`, so VC is never None,
    # and `_value_choice_last` is only ever SET on a turn the wrapper actually engaged (read post-turn, matching
    # the soak's own `getattr(chat, "_value_choice_last", None)` usage).
    gnw_installed = "webapp.gnw_deliberation" in sys.modules
    vc_installed = bool(VC is not None)
    vc_engaged_this_turn = getattr(chat, "_value_choice_last", None) is not None
    return {
        "seed": seed, "ambiguous": list(AMBIGUOUS[:2]) + [list(AMBIGUOUS[2])],
        "stored_facts_for_agent_action": [list(f) for f in matching],
        "distinct_patients": distinct_patients,
        "conflict_visible_in_kb_dump": bool(conflict_visible),
        "gnw_deliberation_installed": gnw_installed,
        "value_choice_organ_available": vc_installed,
        "value_choice_engaged_this_turn": vc_engaged_this_turn,
        "trigger_turn": TRIGGER_TURN, "trigger_reply": trigger_reply,
        "trigger_reply_names_a_stored_patient": bool(trigger_names_stored_patient),
        "note": "fixture-state proof only: this construction bypasses conversational teaching (inner.hear direct, "
                "not chat.gate/_maybe_acquire), so it does NOT show production chat can reach this KB state -- it "
                "shows the KB state itself is directly constructible and the conflict propagates once it exists. "
                "gnw-deliberation and value-driven-choice stay kind='thin' in FACULTY_LESIONS (see EXTRA_LESIONS).",
    }


# ── byte-identity (flags OFF): two builds of the default 10-turn group, diffed for exact equality ─────────────────
def byte_identity_check(seed=7):
    """With every new flag unset, run the SAME default 10-turn PROBE_TURNS prefix twice (fresh subprocess builds,
    reusing onebrain_regression_battery's own `_spawn_arm`) and assert byte-for-byte equality. Combined with the
    STRUCTURAL argument (every edit is a single early-return gated on a helper that reads False when its env var
    is unset -- see the PREREGISTRATION), this is the empirical half of the byte-identity claim."""
    from research.runners.onebrain_regression_battery import _spawn_arm, PROBE_TURNS
    labels = [t[0] for t in PROBE_TURNS[:10]]
    out_dir = "research/findings/raw/_lbf_rows_conflict_kb"
    os.makedirs(out_dir, exist_ok=True)
    a = _spawn_arm({}, labels, os.path.join(out_dir, "byteident_a_s%d.json" % seed))
    b = _spawn_arm({}, labels, os.path.join(out_dir, "byteident_b_s%d.json" % seed))
    # VOID GUARD (tools.lab.void_if): a is None or b is None means the arm's OWN subprocess failed to build/write
    # (a crash, an OOM-kill, a memcap kill) -- `None == None` would otherwise read `byte_identical: True`, a
    # vacuous pass that proves nothing (the exact "a check that cannot fail" shape this codebase's own culture
    # retracts findings over). A void here is NEVER reported as a positive byte-identity result.
    build_failed = void_if(a is None or b is None, "arm build failed (subprocess crash/OOM/kill) -- "
                           "byte-identity is UNDEFINED, not proven, not a pass")
    identical = None if build_failed else (a == b)
    return {"seed": seed, "turns": labels, "byte_identical": identical,
            "a_is_none": a is None, "b_is_none": b is None, "build_failed": build_failed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Phase 1: the 5 driving-turn lesions + the isolated "
                    "biased-competition proof.")
    ap.add_argument("--conflict-kb", action="store_true", help="Phase 2: LB_CONFLICT_KB_PROBE.")
    ap.add_argument("--byte-identity", action="store_true", help="two-build determinism check, flags OFF.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if not (args.smoke or args.conflict_kb or args.byte_identity):
        ap.error("pass at least one of --smoke / --conflict-kb / --byte-identity")
    out = {"seed": args.seed}
    if args.smoke:
        out["five_lesions"] = smoke_five_lesions(seed=args.seed)
        out["biased_competition_isolated"] = smoke_biased_competition_isolated(seed=args.seed)
    if args.conflict_kb:
        out["conflict_kb"] = conflict_kb_probe(seed=args.seed)
    if args.byte_identity:
        out["byte_identity"] = byte_identity_check(seed=args.seed)
    print(json.dumps({k: v for k, v in out.items() if k != "five_lesions"}, indent=2, default=str)[:4000])
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2, default=str)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
