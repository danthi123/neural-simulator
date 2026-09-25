#!/usr/bin/env python3
"""Independent re-derivation of the awake-replay `--family arc` 6-seed verdict, from RAW per-arm JSON only.

Does NOT import research/runners/_da_tag_capture_chat_probe.py or call grade_seed_arc/aggregate_arc. Field
names (recalled_svo, abstained, da_tag_capture, awake_replay_capture.bouts, sleep_replay_capture.epochs, R,
R_eff, early_before/after, n_drive_entries, p_at_bout, lesioned, replay_lesioned, da_swr, da_seen_by_d1,
gamma, blocks) were confirmed by directly inspecting the raw per-arm JSON files themselves, cross-checked
against the grader source only to be sure the reader is looking at the field the grader means. All gate LOGIC
below is re-derived from research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md Amendments 4-5
(the prereg's own prose: G0/P1/I1/I2/I3/gamma UNDEFINED rules, ARC1-ARC7 GO conditions, the 6-seed combine,
and the arm/env table), not copied from grade_seed_arc.
"""
import glob
import itertools
import json
import math
import os

RAW = "research/findings/raw/_awake_replay_capture"
SEEDS = [42, 43, 44, 100, 101, 102]
FACT = ["cat", "chase", "ball"]
EXPECTED_SHA = "30ba29d4b4fa5c5d9b069a83ef9e317f75ba67c5"
DA_TONIC = 0.5  # stated in the finding's own prose ("da_swr=0.5 (the tonic reference)", "da_seen_by_d1=0.5 (tonic)")
AWAKE_H = 4.0   # prereg: "an awake mark of at least 4 h" / Amendment 4 "AWAKE_H"

# Arm -> (group, label, role, env flags expected ON), taken from the PREREG's own "### Arms" table (Amendment 4),
# not from the grader's ARC_ARMS constant. ON=BRAIN_DA_TAG_CAPTURE=1(+CLOCK=turn), RC=BRAIN_SLEEP_REPLAY_CAPTURE=1,
# ARC=BRAIN_AWAKE_REPLAY_CAPTURE=1, plus each arm's named lesion flag.
ARM_TABLE = {
    "lr_arc_a":            dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1"}),
    "lr_arc_b":            dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1"}),
    "lr_noarc":            dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1"}),
    "lr_arc_lesion":       dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1", "BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1"}),
    "ln_arc":              dict(group="datl",  label="datl_recall",  n_bouts=0,  has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1"}),
    "lr_arc_sleeplesion":  dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"}),
    "lr_arc_dalesion":     dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1", "BRAIN_DA_ENCODING_LESION": "1"}),
    "lsr_arc_sleeplesion": dict(group="datcr", label="datcr_recall", n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"}),
    "neu_imm_arc":         dict(group="datni", label="datni_recall", n_bouts=0,  has_night=False,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1"}),
    "lq_arc":              dict(group="datq",  label="datq_recall",  n_bouts=4,  has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1"}, reported=True),
    "lz_arc":              dict(group="datz",  label="datz_recall",  n_bouts=12, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_SLEEP_REPLAY_CAPTURE": "1",
                                        "BRAIN_AWAKE_REPLAY_CAPTURE": "1"}, reported=True),
    "lr_ledger_off":       dict(group="datr",  label="datr_recall",  n_bouts=48, has_night=True,
                                 flags={"BRAIN_DA_TAG_CAPTURE": "0"}, reported=True),
}
REPORTED_ARMS = {k for k, v in ARM_TABLE.items() if v.get("reported")}
GATED_ARMS = [k for k in ARM_TABLE if k not in REPORTED_ARMS]
ALL_ARMS = list(ARM_TABLE)


def outcome(rec):
    """correct / abstain / confab / undefined -- re-derived from the prereg's own def of a recall turn's result,
    not copied from research.runners._da_tag_capture_chat_probe.outcome (independently reads the same two raw
    fields, recalled_svo and abstained, that the finding's own tables cite)."""
    if not isinstance(rec, dict) or rec.get("_error"):
        return "undefined"
    svo = rec.get("recalled_svo")
    if svo is not None and list(svo) == FACT:
        return "correct"
    if svo is None and rec.get("abstained"):
        return "abstain"
    if svo is not None:
        return "confab"
    return "undefined"


def tc_of(rec):
    return (rec or {}).get("da_tag_capture") or {}


def bouts_of(rec):
    return (tc_of(rec).get("awake_replay_capture") or {}).get("bouts") or []


def epochs_of(rec):
    return (tc_of(rec).get("sleep_replay_capture") or {}).get("epochs") or []


def load_arm(seed, arm):
    path = os.path.join(RAW, "seed%d" % seed, "%s.json" % arm)
    with open(path) as f:
        raw = json.load(f)
    label = ARM_TABLE[arm]["label"]
    rec = raw.get(label)
    n_err = sum(1 for v in raw.values() if isinstance(v, dict) and v.get("_error"))
    return raw, rec, n_err


def check_provenance(seed, arm):
    prov_path = os.path.join(RAW, "seed%d" % seed, "%s.json.prov.json" % arm)
    with open(prov_path) as f:
        p = json.load(f)
    problems = []
    if p.get("git_sha") != EXPECTED_SHA:
        problems.append("git_sha=%r" % p.get("git_sha"))
    if p.get("source_kind") != "git_archive":
        problems.append("source_kind=%r" % p.get("source_kind"))
    if p.get("git_dirty") is not False:
        problems.append("git_dirty=%r" % p.get("git_dirty"))
    if p.get("source_manifest_verified_at_start") is not True:
        problems.append("manifest_start=%r" % p.get("source_manifest_verified_at_start"))
    if p.get("source_manifest_verified_at_exit") is not True:
        problems.append("manifest_exit=%r" % p.get("source_manifest_verified_at_exit"))
    # cross-check the actual argv --env against the expected flags for this arm (independent of the prereg prose:
    # this is what the worker subprocess was ACTUALLY launched with, recorded by research/runners/__init__.py)
    argv = p.get("argv") or []
    env_json = None
    for i, a in enumerate(argv):
        if a == "--env" and i + 1 < len(argv):
            env_json = argv[i + 1]
    actual_env = json.loads(env_json) if env_json else {}
    expected = ARM_TABLE[arm]["flags"]
    for k, v in expected.items():
        if actual_env.get(k) != v:
            problems.append("env[%s]=%r (expected %r)" % (k, actual_env.get(k), v))
    # also: no UNEXPECTED lesion/flag beyond what's expected (catches a mixed-up arm)
    lesion_keys = {"BRAIN_AWAKE_REPLAY_CAPTURE_LESION", "BRAIN_SLEEP_REPLAY_CAPTURE_LESION",
                   "BRAIN_DA_ENCODING_LESION"}
    for k in lesion_keys:
        if k not in expected and actual_env.get(k) == "1":
            problems.append("unexpected env[%s]=1" % k)
    return problems


def signflip_p_one_sided(diffs):
    """Exact one-sided sign-flip test: drop zeros, p = P(all remaining signs land as observed | random sign)."""
    nz = [d for d in diffs if d != 0]
    if not nz:
        return None
    n = len(nz)
    observed_pos = sum(1 for d in nz if d > 0)
    # one-sided: direction is "positive" (rescue effect), p = P(X >= observed_pos), X ~ Binomial(n, 0.5)
    p = sum(math.comb(n, k) for k in range(observed_pos, n + 1)) / (2 ** n)
    return p


def grade_seed(seed):
    raw = {}
    rec = {}
    nerr = {}
    for arm in ALL_ARMS:
        raw[arm], rec[arm], nerr[arm] = load_arm(seed, arm)
    o = {arm: outcome(rec[arm]) for arm in ALL_ARMS}
    errs = sum(nerr[arm] for arm in GATED_ARMS)

    problems = []

    # G0: lr_arc_a vs lr_arc_b null-rebuild -- full da_tag_capture record must agree exactly, plus the recall
    # decision, per the prereg's "outcome, recalled_svo, abstained, ... the sleep record or the awake record"
    a, b = rec["lr_arc_a"], rec["lr_arc_b"]
    g0 = (o["lr_arc_a"] == o["lr_arc_b"]
          and a.get("recalled_svo") == b.get("recalled_svo")
          and a.get("abstained") == b.get("abstained")
          and json.dumps(tc_of(a), sort_keys=True) == json.dumps(tc_of(b), sort_keys=True))

    # P1: immediate recall precondition
    p1 = o["neu_imm_arc"] == "correct"

    # gamma consistency across every companion-ON (BRAIN_DA_TAG_CAPTURE=1) gated arm
    gammas = [tc_of(rec[k]).get("gamma") for k in GATED_ARMS
              if ARM_TABLE[k]["flags"].get("BRAIN_DA_TAG_CAPTURE") == "1" and tc_of(rec[k]).get("gamma") is not None]
    gamma_ok = (not gammas) or all(abs(x - gammas[0]) < 1e-6 for x in gammas)

    # I1: awake branch ran exactly as scheduled
    def inst_ok(arm):
        spec = ARM_TABLE[arm]
        bo = bouts_of(rec[arm])
        ep = epochs_of(rec[arm])
        if spec["flags"].get("BRAIN_AWAKE_REPLAY_CAPTURE") != "1":
            # flag off (lr_noarc, lr_ledger_off): no awake record should exist at all
            return tc_of(rec[arm]).get("awake_replay_capture") is None
        if len(bo) != spec["n_bouts"]:
            return False
        if any((x.get("R") is None or None in (x.get("R") or [None])) for x in bo):
            return False
        aw = tc_of(rec[arm]).get("awake_until_h")
        if not spec["has_night"]:
            return len(ep) == 0
        if len(ep) != 1 or aw is None or aw < AWAKE_H:
            return False
        return all(x["t_h"] <= aw + 1e-9 and x["t_h"] < ep[0]["t_h"] for x in bo)

    i1_gated = {arm: inst_ok(arm) for arm in GATED_ARMS}
    i1 = all(i1_gated.values())
    i1_reported = {arm: inst_ok(arm) for arm in REPORTED_ARMS if arm != "lr_ledger_off"}
    # lr_ledger_off is flag-off entirely (ledger off): no awake/sleep record makes sense to check the same way;
    # report separately, never gating.
    i1_reported["lr_ledger_off"] = tc_of(rec["lr_ledger_off"]).get("awake_replay_capture") is None

    # I2: every lesion held AT MEASUREMENT (read off the record, not the flag)
    held = True
    for arm in GATED_ARMS:
        spec = ARM_TABLE[arm]
        awake_lesioned = spec["flags"].get("BRAIN_AWAKE_REPLAY_CAPTURE_LESION") == "1"
        for x in bouts_of(rec[arm]):
            if awake_lesioned:
                held &= bool(x.get("lesioned") is True and all(v == 0.0 for v in (x.get("R_eff") or []))
                             and x.get("early_after") == x.get("early_before"))
            else:
                held &= x.get("lesioned") is not True
        if awake_lesioned:
            blocks = tc_of(rec[arm]).get("blocks") or []
            held &= all("e_rep" not in bl for bl in blocks)
        sleep_lesioned = spec["flags"].get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1"
        da_lesioned = spec["flags"].get("BRAIN_DA_ENCODING_LESION") == "1"
        for e in epochs_of(rec[arm]):
            if sleep_lesioned:
                held &= bool(e.get("replay_lesioned") is True and all(v == 0.0 for v in (e.get("R_eff") or []))
                             and abs(e.get("da_swr", -1.0) - DA_TONIC) < 1e-9)
            else:
                held &= e.get("replay_lesioned") is not True
            if da_lesioned:
                held &= abs(e.get("da_seen_by_d1", -1.0) - DA_TONIC) < 1e-9
    i2 = held

    # I3: awake bouts add no PRP -- constant n_drive_entries, non-increasing p_at_bout, across >=2-bout gated arms
    noprp = True
    for arm in GATED_ARMS:
        bo = bouts_of(rec[arm])
        if len(bo) >= 2:
            noprp &= len({x.get("n_drive_entries") for x in bo}) == 1
            noprp &= all(y.get("p_at_bout", 0.0) <= x.get("p_at_bout", 0.0) + 1e-15 for x, y in zip(bo, bo[1:]))
    i3 = noprp

    # behavioural gates
    arc1 = (o["lr_arc_a"] == "correct" and o["lr_noarc"] == "abstain")
    arc2 = (o["lr_arc_lesion"] == "abstain")
    arc3 = (o["ln_arc"] == "abstain")
    arc4 = (o["lr_arc_sleeplesion"] == "abstain")
    arc5 = (o["lr_arc_dalesion"] == "abstain")
    arc6 = (o["lsr_arc_sleeplesion"] == "correct")
    arc7 = all(v != "confab" for v in o.values())  # every arm, REPORTED included

    undefined = (not g0) or (not p1) or (not gamma_ok) or (not i1) or (not i2) or (not i3) or errs > 0 \
        or any(o[k] == "undefined" for k in GATED_ARMS)
    core = all([arc1, arc2, arc3, arc4, arc5, arc6, arc7])
    verdict = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")

    return dict(seed=seed, outcomes=o, n_arm_errors=errs,
                G0=g0, P1=p1, gamma=gamma_ok, I1=i1, I1_reported=i1_reported, I2=i2, I3=i3,
                ARC1=arc1, ARC2=arc2, ARC3=arc3, ARC4=arc4, ARC5=arc5, ARC6=arc6, ARC7=arc7,
                undefined=undefined, verdict=verdict, rec=rec)


def main():
    print("=" * 100)
    print("PROVENANCE CHECK (72 arm files, expect git_sha=%s, source_kind=git_archive)" % EXPECTED_SHA)
    print("=" * 100)
    prov_problems = {}
    n_checked = 0
    for seed in SEEDS:
        for arm in ALL_ARMS:
            n_checked += 1
            probs = check_provenance(seed, arm)
            if probs:
                prov_problems[(seed, arm)] = probs
    print("checked %d arm files" % n_checked)
    if prov_problems:
        print("PROVENANCE PROBLEMS:")
        for k, v in prov_problems.items():
            print(" ", k, v)
    else:
        print("provenance: ALL 72 arm files clean (git_sha exact, source_kind=git_archive, git_dirty=False,"
              " manifest verified start+exit, argv --env matches the expected role table)")

    print()
    print("=" * 100)
    print("PER-SEED GATE RE-DERIVATION (independent script, raw JSON only)")
    print("=" * 100)
    results = {}
    for seed in SEEDS:
        g = grade_seed(seed)
        results[seed] = g
        print("seed %3d: G0=%s P1=%s gamma=%s I1=%s I2=%s I3=%s errs=%d | ARC1=%s ARC2=%s ARC3=%s ARC4=%s ARC5=%s "
              "ARC6=%s ARC7=%s -> %s"
              % (seed, g["G0"], g["P1"], g["gamma"], g["I1"], g["I2"], g["I3"], g["n_arm_errors"],
                 g["ARC1"], g["ARC2"], g["ARC3"], g["ARC4"], g["ARC5"], g["ARC6"], g["ARC7"], g["verdict"]))

    n_go = sum(1 for g in results.values() if g["verdict"] == "GO")
    n_undef = sum(1 for g in results.values() if g["verdict"] == "UNDEFINED")
    complete = sorted(results) == sorted(SEEDS)
    if not complete:
        overall = "INCOMPLETE"
    elif n_undef > 0:
        overall = "UNDEFINED present (%d) -- prereg's 6-seed rule does not cover this case explicitly; treating as" \
                  " NOT a clean GO/NO-GO" % n_undef
    elif n_go == 6:
        overall = "GO"
    else:
        overall = "NO-GO"
    print()
    print("6-seed combine (prereg Amendment-4 rule: GO iff all six GO; INCOMPLETE if a seed missing; else NO-GO):")
    print("  n_go=%d/6, complete=%s, n_undefined=%d -> %s" % (n_go, complete, n_undef, overall))

    # zero-rest / rescue-arm bout sanity, read straight off raw JSON (not from any gate boolean)
    print()
    print("=" * 100)
    print("AWAKE-BOUT SANITY: rescue arms (48 bouts expected) vs the zero-idle-tick arm ln_arc (0 bouts expected)")
    print("=" * 100)
    for seed in SEEDS:
        g = results[seed]
        row = {}
        for arm in ("lr_arc_a", "lr_arc_b", "lr_arc_lesion", "lr_arc_sleeplesion", "lr_arc_dalesion",
                    "lsr_arc_sleeplesion", "ln_arc", "lr_noarc", "neu_imm_arc"):
            row[arm] = len(bouts_of(g["rec"][arm]))
        print("seed %3d: %s" % (seed, row))

    # sign-flip p, both prereg-registered contrasts
    print()
    print("=" * 100)
    print("SIGN-FLIP p (one-sided, exact), independently recomputed from per-seed outcomes")
    print("=" * 100)
    diffs_on_off = [int(results[s]["outcomes"]["lr_arc_a"] == "correct")
                    - int(results[s]["outcomes"]["lr_noarc"] == "correct") for s in SEEDS]
    diffs_on_lesion = [int(results[s]["outcomes"]["lr_arc_a"] == "correct")
                       - int(results[s]["outcomes"]["lr_arc_lesion"] == "correct") for s in SEEDS]
    print("seeds:                       ", SEEDS)
    print("diffs lr_arc_a - lr_noarc:   ", diffs_on_off, "-> p =", signflip_p_one_sided(diffs_on_off))
    print("diffs lr_arc_a - lr_arc_lesion:", diffs_on_lesion, "-> p =", signflip_p_one_sided(diffs_on_lesion))

    # reported-arm correct counts (not gating; cross-check the finding's table)
    print()
    print("=" * 100)
    print("REPORTED arms (never gating) -- correct-count out of 6")
    print("=" * 100)
    for arm in ("lr_ledger_off", "lq_arc", "lz_arc"):
        n_correct = sum(1 for s in SEEDS if results[s]["outcomes"][arm] == "correct")
        print("  %-14s %d/6 correct  (per-seed: %s)" % (arm, n_correct,
              [results[s]["outcomes"][arm] for s in SEEDS]))

    # seed-101 R-decay table, straight from raw JSON (cross-check of the finding's own numeric table)
    print()
    print("=" * 100)
    print("lr_arc_a reactivation read R: first bout / last bout / sleep onset, per seed")
    print("=" * 100)
    for seed in SEEDS:
        rec = results[seed]["rec"]["lr_arc_a"]
        bo = bouts_of(rec)
        ep = epochs_of(rec)
        r_first = bo[0]["R"][0] if bo else None
        r_last = bo[-1]["R"][0] if bo else None
        r_sleep = ep[0]["R"][0] if ep else None
        print("  seed %3d: R_first=%.9f  R_last=%.9f  R_sleep_onset=%.9f  outcome=%s"
              % (seed, r_first, r_last, r_sleep, results[seed]["outcomes"]["lr_arc_a"]))


if __name__ == "__main__":
    main()
