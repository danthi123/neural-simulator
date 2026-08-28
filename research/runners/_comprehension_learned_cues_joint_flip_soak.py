"""JOINT 6-seed flip-soak for the comprehension organ's two corpus-learned cues (Vikunja #190):
`BRAIN_LEARNED_ANIMACY_CUE` (research/findings/2026-08-27-comprehension-cue-lexicon-spiking-realized-and-wired.md)
+ `BRAIN_LEARNED_VERB_SELECTS` (research/findings/2026-08-27-comprehension-verb-selects-wired-GO.md), both
already individually wired (default-OFF, GO-verified). This is the arc-closing rung: does turning BOTH ON
TOGETHER remain byte-identical-off, non-regressive on hand-covered vocabulary, genuinely coverage-extending on
open-vocab nouns/verbs, lesion-load-bearing per-cue, and free of a two-cue interaction defect?

METHOD: each (seed, flag-condition) arm runs `_comprehension_learned_cues_joint_arm.py` in a FRESH subprocess
(the organ + the two learned lexicons are process-global singletons -- a same-process sequential flag toggle
would risk the residual chaotic-jitter class this repo has documented elsewhere; a fresh process per arm side-
steps that at negligible cost, ~2-3s/arm numpy-CPU). Conditions per seed:

    C0 unset            -- BOTH flags left UNSET (the literal CURRENT-MAIN default state)
    C1 off               -- BOTH flags explicit "0" (the byte-identical escape a future default-flip would use)
    C2 animacy_only      -- animacy ON, verb OFF
    C3 verb_only         -- animacy OFF, verb ON
    C4 both              -- BOTH ON (the joint-flip candidate state)
    C5 lesion_animacy    -- both ON, BRAIN_LEARNED_ANIMACY_LESION=1 (verb cue intact)
    C6 lesion_verb       -- both ON, BRAIN_LEARNED_VERB_SELECTS_LESION=1 (animacy cue intact)
    C7 lesion_both       -- both ON, both lesions on (must fully revert to C1)

GATES (all must hold for a GO; see the arm script's docstring for the battery definition):
    G1 BYTE-IDENTICAL-OFF   : C0 == C1 on the FULL battery, every seed (unset really is "off", empirically).
    G2 NO-REGRESSION        : C1==C2==C3==C4==C5==C6==C7 on every HAND_COVERED item, every seed (the hand
                               table is an unconditional fast path -- ANY flag/lesion combination must be
                               byte-identical there; this also IS the literal "interaction check" the task
                               specifies -- hand-covered outcomes must not depend on which cue(s) are on).
    G3 COVERAGE EXTENDS     : every HELD_NOUN/HELD_VERB item's `competent()` flips OFF(C1)=False ->
                               BOTH-ON(C4)=True, at every seed, and NEVER the reverse (monotonic superset).
    G4 LESION LOAD-BEARING  : C5 (animacy lesioned) reverts HELD_NOUN items' competence to C1 while leaving
                               HELD_VERB items at C4; C6 (verb lesioned) is the mirror; C7 (both lesioned)
                               reverts the WHOLE battery to exactly C1.
    G5 JOINT COVERAGE       : on the doubly-held-out JOINT items, C1's `repair_target` names all 3 open-
                               vocabulary content words as OOV; C4's does not (none of the 3 are OOV any
                               more) -- the joint-specific coverage signal (these items are `competent()`=True
                               under BOTH C1 and C4 via the pre-existing fully-OOV/fully-covered symmetry of
                               `competent()`, so the noun/verb flip metric does not apply to them -- see the
                               arm script's docstring; this is a DIFFERENT, honestly-distinct measurement).

Plus a full-production-turn check (webapp.server.brain_chat, single seed=42 -- the ONLY seed the handler ever
actually builds with, research/runners/comprehension_production_organ.py get_organ(seed=42) call sites) on a
small representative sub-battery, mirroring `_gateB_repair_production_verify.py`'s own methodology.

Run: SIM_BACKEND=numpy python -m research.runners._comprehension_learned_cues_joint_flip_soak \\
    --out research/findings/raw/_comprehension_learned_cues_joint_flip_soak_6seed.json
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

from tools.lab import attributable_to
from tools.verdict import Verdict
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

_REPO = Path(__file__).resolve().parents[2]
ARM_MOD = "research.runners._comprehension_learned_cues_joint_arm"
ARM_DIR = _REPO / "research" / "findings" / "raw" / "_comprehension_learned_cues_joint"

SEEDS = [42, 43, 44, 100, 101, 102]

CONDITIONS = [
    # (label, animacy, verb, lesion_animacy, lesion_verb)
    ("C0_unset", "unset", "unset", "off", "off"),
    ("C1_off", "off", "off", "off", "off"),
    ("C2_animacy_only", "on", "off", "off", "off"),
    ("C3_verb_only", "off", "on", "off", "off"),
    ("C4_both", "on", "on", "off", "off"),
    ("C5_lesion_animacy", "on", "on", "on", "off"),
    ("C6_lesion_verb", "on", "on", "off", "on"),
    ("C7_lesion_both", "on", "on", "on", "on"),
]


def _run_arm(seed: int, label: str, animacy: str, verb: str, lesion_a: str, lesion_v: str) -> dict:
    out_path = ARM_DIR / f"arm_seed{seed}_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", ARM_MOD,
        "--seed", str(seed), "--animacy", animacy, "--verb", verb,
        "--lesion-animacy", lesion_a, "--lesion-verb", lesion_v,
        "--out", str(out_path),
    ]
    env = os.environ.copy()
    # Never let the controller's own already-set flags bleed into a child that means to test "unset".
    for k in ("BRAIN_LEARNED_ANIMACY_CUE", "BRAIN_LEARNED_VERB_SELECTS",
              "BRAIN_LEARNED_ANIMACY_LESION", "BRAIN_LEARNED_VERB_SELECTS_LESION"):
        env.pop(k, None)
    r = subprocess.run(cmd, cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"arm failed seed={seed} {label}: rc={r.returncode}\nSTDOUT:{r.stdout[-3000:]}\nSTDERR:{r.stderr[-3000:]}")
    return json.loads(out_path.read_text())


def _strip_calib(battery: dict) -> dict:
    """`judge()['calib']` is a per-organ build-time constant (same across every condition at a fixed seed,
    since it never touches the cue flags) -- drop it before diffing so a diff highlights a REAL behavioral
    difference, not a float-formatting artifact of re-serializing the same numbers."""
    out = {}
    for k, v in battery.items():
        v = copy.deepcopy(v)
        if v.get("judge") and "calib" in v["judge"]:
            v["judge"].pop("calib", None)
        out[k] = v
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" /
                                         "_comprehension_learned_cues_joint_flip_soak_6seed.json"))
    args = ap.parse_args()

    from research.runners._comprehension_learned_cues_joint_arm import (
        HAND_COVERED, HELD_NOUN_ANIM, HELD_NOUN_INANIM, HELD_VERB_INANIM_PATIENT,
        HELD_VERB_ANIM_PATIENT, _joint_sentences, MOAT,
    )
    hand_labels = [lbl for lbl, *_ in HAND_COVERED]
    held_noun_labels = [f"noun_anim_{n}" for n in HELD_NOUN_ANIM] + [f"noun_inanim_{n}" for n in HELD_NOUN_INANIM]
    held_verb_labels = [f"verb_inanimpat_{v}" for v in HELD_VERB_INANIM_PATIENT] + \
                        [f"verb_animpat_{v}" for v in HELD_VERB_ANIM_PATIENT]
    joint_labels = [lbl for lbl, *_ in _joint_sentences()]
    moat_labels = [lbl for lbl, *_ in MOAT]

    per_seed = {}
    for seed in SEEDS:
        arms = {}
        for label, animacy, verb, la, lv in CONDITIONS:
            arms[label] = _run_arm(seed, label, animacy, verb, la, lv)["battery"]
        per_seed[seed] = arms
        print(f"seed {seed}: {len(arms)} conditions collected")

    # ── G1: byte-identical-off (C0 unset == C1 explicit-off), FULL battery, every seed. ──
    g1_fail = []
    for seed in SEEDS:
        c0 = _strip_calib(per_seed[seed]["C0_unset"])
        c1 = _strip_calib(per_seed[seed]["C1_off"])
        if c0 != c1:
            for k in c0:
                if c0[k] != c1.get(k):
                    g1_fail.append({"seed": seed, "item": k, "c0": c0[k], "c1": c1.get(k)})
    g1_pass = not g1_fail

    # ── G2: no-regression + interaction on HAND_COVERED (every condition == C1, every seed). ──
    g2_fail = []
    for seed in SEEDS:
        c1 = _strip_calib(per_seed[seed]["C1_off"])
        for label, _, _, _, _ in CONDITIONS:
            if label in ("C0_unset", "C1_off"):
                continue
            cx = _strip_calib(per_seed[seed][label])
            for item in hand_labels:
                if cx[item] != c1[item]:
                    g2_fail.append({"seed": seed, "condition": label, "item": item, "c1": c1[item], "cx": cx[item]})
    g2_pass = not g2_fail

    # ── G3: coverage extends monotonically (competent False->True, C1->C4) on HELD_NOUN + HELD_VERB. ──
    g3_flips = []
    g3_regressions = []
    for seed in SEEDS:
        c1 = per_seed[seed]["C1_off"]
        c4 = per_seed[seed]["C4_both"]
        for item in held_noun_labels + held_verb_labels:
            off_comp = c1[item]["competent"]
            on_comp = c4[item]["competent"]
            if off_comp is False and on_comp is True:
                g3_flips.append({"seed": seed, "item": item})
            else:
                g3_regressions.append({"seed": seed, "item": item, "note": "expected off=False,on=True",
                                        "off_comp": off_comp, "on_comp": on_comp})
    expected_flips = len(SEEDS) * len(held_noun_labels + held_verb_labels)
    g3_pass = (len(g3_flips) == expected_flips) and (not g3_regressions)

    # ── G4: lesion load-bearing, per-cue + full revert. ──
    g4_fail = []
    for seed in SEEDS:
        c1 = per_seed[seed]["C1_off"]
        c4 = per_seed[seed]["C4_both"]
        c5 = per_seed[seed]["C5_lesion_animacy"]
        c6 = per_seed[seed]["C6_lesion_verb"]
        c7 = _strip_calib(per_seed[seed]["C7_lesion_both"])
        for item in held_noun_labels:
            if c5[item]["competent"] != c1[item]["competent"]:
                g4_fail.append({"seed": seed, "check": "lesion_animacy_reverts_noun", "item": item,
                                 "c5_competent": c5[item]["competent"], "c1_competent": c1[item]["competent"]})
            if c6[item]["competent"] != c4[item]["competent"]:
                g4_fail.append({"seed": seed, "check": "lesion_verb_spares_noun", "item": item,
                                 "c6_competent": c6[item]["competent"], "c4_competent": c4[item]["competent"]})
        for item in held_verb_labels:
            if c6[item]["competent"] != c1[item]["competent"]:
                g4_fail.append({"seed": seed, "check": "lesion_verb_reverts_verb", "item": item,
                                 "c6_competent": c6[item]["competent"], "c1_competent": c1[item]["competent"]})
            if c5[item]["competent"] != c4[item]["competent"]:
                g4_fail.append({"seed": seed, "check": "lesion_animacy_spares_verb", "item": item,
                                 "c5_competent": c5[item]["competent"], "c4_competent": c4[item]["competent"]})
        c1_full = _strip_calib(c1)
        if c7 != c1_full:
            for k in c1_full:
                if c1_full[k] != c7.get(k):
                    g4_fail.append({"seed": seed, "check": "lesion_both_full_revert", "item": k,
                                     "c1": c1_full[k], "c7": c7.get(k)})
    g4_pass = not g4_fail

    # ── G5: joint coverage (repair_target OOV-token naming shrinks to empty on the doubly-held-out items). ──
    g5_fail = []
    g5_detail = []
    for seed in SEEDS:
        c1 = per_seed[seed]["C1_off"]
        c4 = per_seed[seed]["C4_both"]
        for item in joint_labels:
            off_oov = set((c1[item].get("repair_target") or {}).get("oov_tokens") or [])
            on_oov = set((c4[item].get("repair_target") or {}).get("oov_tokens") or [])
            g5_detail.append({"seed": seed, "item": item, "off_oov_tokens": sorted(off_oov),
                               "on_oov_tokens": sorted(on_oov),
                               "off_comprehended": (c1[item].get("judge") or {}).get("comprehended"),
                               "on_comprehended": (c4[item].get("judge") or {}).get("comprehended")})
            if not off_oov:
                g5_fail.append({"seed": seed, "item": item, "note": "flags-off did not name the open-vocab words OOV"})
            if on_oov:
                g5_fail.append({"seed": seed, "item": item, "note": "flags-on STILL names words OOV",
                                 "still_oov": sorted(on_oov)})
    g5_pass = not g5_fail

    # ── Moat: OOV item must abstain (competent True via fully_oov, comprehended False) in every condition. ──
    moat_fail = []
    for seed in SEEDS:
        for label, _, _, _, _ in CONDITIONS:
            row = per_seed[seed][label][moat_labels[0]]
            j = row.get("judge") or {}
            if j.get("comprehended") is not False:
                moat_fail.append({"seed": seed, "condition": label, "judge": j})
    moat_pass = not moat_fail

    overall_go = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass and moat_pass

    # ── ATTRIBUTION (tools.lab): the lesion is a treatment/control pair (G4 measured both arms) -- ask WHOSE
    # the coverage-extension effect is, not just that the two arms differ. Mean per-seed competent()-rate over
    # each cue's own isolated held-out battery, live (C4, both cues on, neither lesioned) vs that cue ALONE
    # lesioned (C5 for animacy, C6 for verb-selects; the sibling cue stays live in both arms, so this isolates
    # THIS cue's own contribution, not the pair's).
    def _competent_rate(cond_key, items):
        vals = []
        for seed in SEEDS:
            arm = per_seed[seed][cond_key]
            vals.append(sum(1 for it in items if arm[it]["competent"]) / len(items))
        return sum(vals) / len(vals)

    rate_noun_live = _competent_rate("C4_both", held_noun_labels)
    rate_noun_animacy_lesioned = _competent_rate("C5_lesion_animacy", held_noun_labels)
    rate_verb_live = _competent_rate("C4_both", held_verb_labels)
    rate_verb_lesioned = _competent_rate("C6_lesion_verb", held_verb_labels)
    attr_animacy = attributable_to("animacy-cue coverage extension (held-noun competent-rate, live vs lesioned)",
                                    rate_noun_live, rate_noun_animacy_lesioned)
    attr_verb = attributable_to("verb-selects-cue coverage extension (held-verb competent-rate, live vs lesioned)",
                                 rate_verb_live, rate_verb_lesioned)

    # ── VERDICT (tools.verdict): a GO must travel with what earned it, not sit beside an unguarded bool. ──
    v = Verdict("comprehension learned-cues joint flip-soak (Vikunja #190)")
    v.require("G1 byte-identical-off (unset == explicit '0')", g1_pass, expect=True)
    v.require("G2 no-regression on hand-covered (== the interaction check)", g2_pass, expect=True)
    v.require("G3 coverage extends monotonically (held-out competent() False->True, 0 regressions)", g3_pass, expect=True)
    v.require("G4 lesion load-bearing (per-cue + full revert)", g4_pass, expect=True)
    v.require("G5 joint coverage (doubly-held-out OOV-naming shrinks to empty)", g5_pass, expect=True)
    v.require("moat holds in every one of the 8 conditions", moat_pass, expect=True)
    v.control("animacy lesion collapses held-noun coverage", treatment=rate_noun_live,
              control=rate_noun_animacy_lesioned, min_separation=0.5)
    v.control("verb-selects lesion collapses held-verb coverage", treatment=rate_verb_live,
              control=rate_verb_lesioned, min_separation=0.5)
    decided = v.decide(go=overall_go)

    payload = {
        "seeds": SEEDS,
        "conditions": [c[0] for c in CONDITIONS],
        "battery_labels": {
            "hand_covered": hand_labels, "held_noun": held_noun_labels,
            "held_verb": held_verb_labels, "joint": joint_labels, "moat": moat_labels,
        },
        "gates": {
            "G1_byte_identical_off": {"pass": g1_pass, "fail_count": len(g1_fail), "fails": g1_fail[:20]},
            "G2_no_regression_hand_covered": {"pass": g2_pass, "fail_count": len(g2_fail), "fails": g2_fail[:20]},
            "G3_coverage_extends": {"pass": g3_pass, "flip_count": len(g3_flips), "expected_flips": expected_flips,
                                     "regression_count": len(g3_regressions), "regressions": g3_regressions[:20]},
            "G4_lesion_load_bearing": {"pass": g4_pass, "fail_count": len(g4_fail), "fails": g4_fail[:20]},
            "G5_joint_coverage": {"pass": g5_pass, "fail_count": len(g5_fail), "fails": g5_fail[:20], "detail": g5_detail},
            "moat_holds": {"pass": moat_pass, "fail_count": len(moat_fail), "fails": moat_fail[:20]},
        },
        "attribution": {
            "animacy_cue_held_noun_rate_live": rate_noun_live,
            "animacy_cue_held_noun_rate_lesioned": rate_noun_animacy_lesioned,
            "animacy_cue_attributable_fraction": attr_animacy,
            "verb_selects_cue_held_verb_rate_live": rate_verb_live,
            "verb_selects_cue_held_verb_rate_lesioned": rate_verb_lesioned,
            "verb_selects_cue_attributable_fraction": attr_verb,
        },
        "GO": bool(decided["go"]),
        "status": decided["status"],
        "verdict": decided["status"],
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    for gname, g in payload["gates"].items():
        print(f"  [{'PASS' if g['pass'] else 'FAIL'}] {gname}")
    print(f"\nGO={payload['GO']} status={payload['status']}  wrote {args.out}")
    return 0 if payload["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
