"""THE PAIR ON ITS PRODUCTION PATH: DA tag-capture + sleep-replay capture on the wall clock, inside one family, on cupy.

WHY (review items B3, B4 and D3 of research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md, and
the opus re-review's missed angle). The pair (BRAIN_DA_TAG_CAPTURE + BRAIN_SLEEP_REPLAY_CAPTURE, both default OFF) has
three registered GOs, all on the scripted TURN clock, numpy, one fact per conversation and one SWR epoch per protocol.
Production runs a different clock (wall), a different sleep trigger (any idle of 5 min or more), several SWR epochs a
day, the episodic organ (backend-gated OFF on numpy, so inert in every earlier arm) and cupy. Gates and predictions are
pre-registered in research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md, Amendment 7; the functions
below implement them verbatim.

FAMILIES (each arm a fresh tiny-demo brain in its own subprocess through the REAL `webapp.server.brain_chat`; groups
in research/runners/onebrain_regression_battery.py; LTM off as in every earlier family):
  pp  (B3 + the episodic angle) the 'wd' day on the WALL clock (the ledger's default clock, fed by the environment's
      virtual wall clock through `webapp.da_tag_capture_chat.set_wall_clock`): three facts told at different times,
      three pauses of >= 5 min, a 3-h stretch with no pause, an 11-h night, three idle days. Sub-verdicts WD (ordinary
      and salient outcomes, the replay edge, no confab), NR (no resurrection of a decayed fact), EP (episodic vs composer
      agreement next day, BRAIN_EPISODIC_STORE=1 on numpy).
  sn  (B4 + the weak telling) the salient vs neutral long-delay contrast inside ONE family with both flags intact, a
      WAKING-ONLY DA lesion (BRAIN_DA_ENCODING_LESION + BRAIN_DA_ENCODING_LESION_SPARE_SWR: the SWR DA edge intact), and
      REPORTED arms: the waking-only lesion at short delay, the full DA lesion, today's default, and the weak telling
      read at once and next day with the pair on and off.
  cu  (D3) the reference 3090: gamma and d1_a_go on cupy, the capture edge on cupy, and no drift in the other organs'
      per-turn DA from the ledger's private RNG (fixed: `_private_rng` now restores the cupy RandomState); a REPORTED
      arm with the pre-fix reseed (BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE=1) and the 'wd' day at the cupy defaults.

Run one seed (numpy; ALWAYS memcap, gate with mem_ok; data/corpus must be present):
  bash tools/mem_ok.sh 4 4 && bash tools/memcap.sh 4 -- .venv/bin/python -u -m research.runners._pair_production_path_probe \
      --family sn --seed 7 --out research/findings/raw/_pair_production_path_smoke
  ... --family pp --seed N ; ... --family cu --seed N (SIM_BACKEND=cupy, one arm at a time)
Aggregate: ... --family pp --aggregate research/findings/raw/_pair_production_path
Selftest (no brain, every gate through its failing directions): ... --selftest
Fake-substrate design day (no brain; not a gate row): ... --design-fake <out.json>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._da_tag_capture_chat_probe import (AWAKE_H, LES, RC, RC_LES, SEEDS, _DA_TONIC_REF,  # noqa: E402
                                                         _tc, corpus_missing, seed_signflip_p)

TAU_EARLY_H = 1.5                     # == webapp.da_tag_capture.TAU_EARLY_H (asserted in the selftest)
SLEEP_ONSET_H = 300.0 / 3600.0        # == continuous_engine.SLEEP_IDLE_SEC (asserted in the selftest)
NIGHT_PERIOD_H = 24.0                 # == sleep_replay_capture.NIGHT_PERIOD_H (asserted in the selftest)
WALL_ON = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "wall"}
TURN_ON = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
BOTH_OFF_WALL = {"BRAIN_DA_TAG_CAPTURE": "0", "BRAIN_SLEEP_REPLAY_CAPTURE": "0", "BRAIN_DA_TAG_CAPTURE_CLOCK": "wall"}
BOTH_OFF_TURN = {"BRAIN_DA_TAG_CAPTURE": "0", "BRAIN_SLEEP_REPLAY_CAPTURE": "0", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
EPI = {"BRAIN_EPISODIC_STORE": "1"}
WAKE_LES = {"BRAIN_DA_ENCODING_LESION": "1", "BRAIN_DA_ENCODING_LESION_SPARE_SWR": "1"}
NO_RESTORE = {"BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE": "1"}
WD_LABEL = "wd_rC_d5"                 # the 'wd' group's last turn (turn_group(WD_LABEL) = the whole day)
WD_FACTS = {"A": ["cat", "chase", "ball"], "B": ["dog", "store", "memory"], "C": ["bird", "use", "river"]}
WD_TELL_TEXT = {"A": "the cat chases the ball", "B": "the dog stores the memory", "C": "the bird uses the river"}
WD_SETS = ("imm", "decay", "preB", "eve", "morn", "d5")
FACT = ["cat", "chase", "ball"]        # the one fact of every turn-clock group (datc / datn / datl / datcl / dw*)

PP_ARMS = [
    ("wd_a", WD_LABEL, {**WALL_ON, **RC}),
    ("wd_b", WD_LABEL, {**WALL_ON, **RC}),                              # G0 null-control rebuild
    ("wd_replaylesion", WD_LABEL, {**WALL_ON, **RC, **RC_LES}),         # WD5
    ("wd_ledger_off", WD_LABEL, dict(BOTH_OFF_WALL)),                   # REPORTED: today's default, same day
    ("wd_epi", WD_LABEL, {**WALL_ON, **RC, **EPI}),                     # EP: the episodic write forced on (numpy)
]
PP_REPORTED_ARMS = ("wd_ledger_off",)
SN_ARMS = [
    ("lsal_rc_a", "datcl_recall", {**TURN_ON, **RC}),                   # SN1
    ("lsal_rc_b", "datcl_recall", {**TURN_ON, **RC}),                   # G0 null-control rebuild
    ("lneu_rc", "datl_recall", {**TURN_ON, **RC}),                      # SN1
    ("lsal_rc_wakelesion", "datcl_recall", {**TURN_ON, **RC, **WAKE_LES}),   # SN2
    ("sal_imm_rc", "datci_recall", {**TURN_ON, **RC}),                  # P1
    ("neu_imm_rc", "datni_recall", {**TURN_ON, **RC}),                  # P1
    ("lsal_rc_dalesion", "datcl_recall", {**TURN_ON, **RC, **LES}),     # REPORTED (the three-edge lesion)
    ("sal_rc_wakelesion", "datc_recall", {**TURN_ON, **RC, **WAKE_LES}),     # REPORTED (review I-3, short delay)
    ("lsal_ledger_off", "datcl_recall", dict(BOTH_OFF_TURN)),           # REPORTED (today's default)
    ("wk_imm_rc", "dwi_recall", {**TURN_ON, **RC}),                     # REPORTED (weak telling, read at once)
    ("wk_imm_off", "dwi_recall", dict(BOTH_OFF_TURN)),                  # REPORTED
    ("wk_night_rc", "dwn_recall", {**TURN_ON, **RC}),                   # REPORTED (weak telling, next day)
    ("wk_night_off", "dwn_recall", dict(BOTH_OFF_TURN)),                # REPORTED (the ledger-off next-day arm)
]
SN_REPORTED_ARMS = ("lsal_rc_dalesion", "sal_rc_wakelesion", "lsal_ledger_off", "wk_imm_rc", "wk_imm_off",
                    "wk_night_rc", "wk_night_off")
CU_ARMS = [
    ("cu_off_a", "datc_recall", dict(BOTH_OFF_TURN)),                   # G0 on cupy
    ("cu_off_b", "datc_recall", dict(BOTH_OFF_TURN)),
    ("cu_on", "datc_recall", {**TURN_ON, **RC}),                        # CU1 / CU2
    ("cu_neu_rc", "datn_recall", {**TURN_ON, **RC}),                    # CU3
    ("cu_neu_norc", "datn_recall", dict(TURN_ON)),                      # CU3
    ("cu_on_norestore", "datc_recall", {**TURN_ON, **RC, **NO_RESTORE}),     # REPORTED: the pre-fix reseed
    ("cu_wd", WD_LABEL, {**WALL_ON, **RC}),                             # REPORTED: the day at the cupy defaults
]
CU_REPORTED_ARMS = ("cu_on_norestore", "cu_wd")
FAMILIES = {"pp": PP_ARMS, "sn": SN_ARMS, "cu": CU_ARMS}
OUT_DEFAULT = "research/findings/raw/_pair_production_path"
NUMPY_GAMMA_REF = 32.7735             # the numpy gamma on every arm of every earlier family (review section 1)


# ── outcome reads ─────────────────────────────────────────────────────────────────────────────────────────────────
_GUESS_MARKERS = ("a guess from what i've learned", "that's a guess")
NOT_RECALLED = ("abstain", "guess")


def outcome_for(resp, fact):
    """correct / abstain / guess / confab / undefined for one recall-question reply, against `fact`. A reply the brain
    itself FLAGS as a guess (the generated-hypothesis path: `hypothesis` true, or the "a guess from what I've learned"
    disclaimer, whose `recalled_svo` carries the guessed triple) is "guess" whatever its content: not a recall, and
    not an unflagged confabulation. (The earlier families' `outcome` would score it confab; none of their recall
    probes produced one.)"""
    if not isinstance(resp, dict) or resp.get("_error"):
        return "undefined"
    if resp.get("hypothesis") or any(m in str(resp.get("answer") or "").lower() for m in _GUESS_MARKERS):
        return "guess"
    svo = resp.get("recalled_svo")
    if svo is not None and list(svo) == list(fact):
        return "correct"
    if svo is None and resp.get("abstained"):
        return "abstain"
    if svo is not None:
        return "confab"
    return "undefined"


def _probe_meta(label):
    m = re.match(r"^wd_([qr])([ABC])_(imm|decay|preB|eve|morn|d5)$", label)
    return None if m is None else (m.group(1), m.group(2), m.group(3))


def expr_frac(blk, t_now):
    """A managed block's mean expressed fraction of its increment at world time t_now: e + z_mean (1 - e), e the
    write's early phase (no awake replay in these arms). 1 = fully expressed, ~0 = at baseline."""
    if not blk or blk.get("t_w") is None or blk.get("z_mean") is None or t_now is None:
        return None
    e = math.exp(-max(0.0, float(t_now) - float(blk["t_w"])) / TAU_EARLY_H)
    return round(e + float(blk["z_mean"]) * (1.0 - e), 9)


def expected_epoch_times(steps):
    """The SWR epochs production runs on this day: for every idle interval of >= SLEEP_IDLE_SEC after a turn, night k
    at (last turn) + onset + 24 k h, while that is not after the interval's end (the next turn's catch-up runs it)."""
    out = []
    for st in steps:
        if st.get("kind", "").startswith("vclock:idle") and st.get("seconds") is not None:
            t0 = float(st["virtual_t_start_s"]) / 3600.0
            t_end = t0 + float(st["seconds"]) / 3600.0
            k = 0
            while t0 + SLEEP_ONSET_H + k * NIGHT_PERIOD_H <= t_end + 1e-12:
                out.append(round(t0 + SLEEP_ONSET_H + k * NIGHT_PERIOD_H, 9))
                k += 1
    return out


def wd_record(r, turns):
    """Everything the pp / cu_wd graders read from one 'wd' arm's responses (a pure function of the stored replies)."""
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, _WORLD_STEPS
    probes = {}
    for lab in turns:
        meta = _probe_meta(lab)
        if meta is None:
            continue
        kind, fact, ps = meta
        resp = r.get(lab) or {}
        tc = _tc(resp)
        rec = {"kind": kind, "fact": fact, "set": ps, "answer": (resp.get("answer") or "")[:240],
               "abstained": resp.get("abstained"), "recalled_svo": resp.get("recalled_svo"),
               "error": (str(resp.get("_error"))[:300] if resp.get("_error") else None)}
        if kind == "q":
            rec["outcome"] = outcome_for(resp, WD_FACTS[fact])
            rec["world_t_h"] = tc.get("world_t_h")
            rec["n_managed_blocks"] = tc.get("n_managed_blocks")
            rec["blocks"] = tc.get("blocks")
            rec["expr_frac"] = [expr_frac(b, tc.get("world_t_h")) for b in (tc.get("blocks") or [])]
            rec["clock"] = tc.get("clock")
        else:
            ep = resp.get("episodic") or {}
            rec["episodic"] = {k: ep.get(k) for k in ("topic", "formed", "in_memory", "reason", "kind", "slot")}
            rec["referential"] = resp.get("referential")
            rec["outcome"] = ("undefined" if resp.get("_error") or not isinstance(resp, dict)
                              else ("in_memory" if ep.get("in_memory") else "not_in_memory"))
        probes[lab] = rec
    tell = {}
    for f, txt in WD_TELL_TEXT.items():
        labs = [lab for lab in turns if lab not in _WORLD_STEPS and (_TURN_BY_LABEL.get(lab) or (None, None))[1] == txt]
        lab = labs[0] if len(labs) == 1 else None
        resp = (r.get(lab) or {}) if lab else {}
        tell[f] = {"label": lab, "n_labels": len(labs), "new_blocks": _tc(resp).get("new_blocks_this_turn"),
                   "da_level": ((resp.get("da_drives") or {}).get("da_level")),
                   "recalled_svo": resp.get("recalled_svo")}
    steps = []
    for lab in turns:
        if lab in _WORLD_STEPS and str(_WORLD_STEPS[lab]).startswith("vclock"):
            s = r.get(lab) or {}
            steps.append({"label": lab, "kind": _WORLD_STEPS[lab], "seconds": s.get("seconds"),
                          "virtual_t_start_s": s.get("virtual_t_start_s"), "virtual_t_end_s": s.get("virtual_t_end_s"),
                          "n_ticks": s.get("n_ticks"), "n_session_ticks": s.get("n_session_ticks"),
                          "n_sleep_depth_ticks": s.get("n_sleep_depth_ticks"),
                          "error": (str(s.get("_error"))[:300] if s.get("_error") else None)})
    last_q = "wd_qC_d5"
    tc_last = _tc(r.get(last_q))
    src = tc_last.get("sleep_replay_capture") or {}
    return {"probes": probes, "tell": tell, "steps": steps,
            "steps_ok": bool(steps) and all(s["error"] is None for s in steps)
            and all(s["kind"] == "vclock:start" or (s["seconds"] is not None and s["virtual_t_end_s"] is not None)
                    for s in steps),
            "expected_epoch_t_h": expected_epoch_times(steps),
            "epochs": src.get("epochs") or [], "clock": tc_last.get("clock"),
            "final_n_managed_blocks": tc_last.get("n_managed_blocks"), "final_blocks": tc_last.get("blocks"),
            "final_tc": {k: tc_last.get(k) for k in ("world_t_h", "n_turns", "p", "p_max", "n_managed_blocks",
                                                      "external_rescales", "external_rewrites", "gamma", "d1_a_go")},
            "ledger_key_present": any("da_tag_capture" in (r.get(t) or {}) for t in turns),
            "turn_da": {t: ((r.get(t) or {}).get("da_drives") or {}).get("da_level") for t in turns
                        if t not in _WORLD_STEPS}}


def turn_record(r, turns, label):
    """The turn-clock arms' read (sn / cu): the recall outcome, the ledger + sleep record at recall, per-turn DA as the
    brain read it, as the D1 pool saw it, and the DA-encoding write gain."""
    from research.runners.onebrain_regression_battery import _WORLD_STEPS
    rec = r.get(label)
    tc = _tc(rec)
    text_turns = [t for t in turns if t not in _WORLD_STEPS]
    return {
        "recall_outcome": outcome_for(rec, FACT), "recalled_svo": (rec or {}).get("recalled_svo"),
        "abstained": (rec or {}).get("abstained"), "answer": ((rec or {}).get("answer") or "")[:240],
        "tag_capture_at_recall": {k: tc.get(k) for k in ("world_t_h", "n_turns", "p", "p_max", "n_managed_blocks",
                                                          "external_rescales", "external_rewrites", "gamma",
                                                          "d1_a_go", "clock")},
        "blocks_at_recall": tc.get("blocks"), "sleep_replay_at_recall": tc.get("sleep_replay_capture"),
        "awake_until_h": tc.get("awake_until_h"),
        "turn_da": [((r.get(t) or {}).get("da_drives") or {}).get("da_level") for t in text_turns],
        "turn_answer": [((r.get(t) or {}).get("answer") or "")[:240] for t in text_turns],
        "turn_da_seen": [((_tc(r.get(t)).get("last_turn") or {}).get("da_seen_by_d1")) for t in text_turns],
        "turn_da_enc": [{k: ((r.get(t) or {}).get("da_encoding") or {}).get(k) for k in ("g", "lesioned")}
                        for t in text_turns],
        "ledger_key_present": any("da_tag_capture" in (r.get(t) or {}) for t in text_turns),
        "text_turns": text_turns}


# ── arms ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, out_dir, family, workers=1, only=None):
    _miss = corpus_missing()
    if _miss and os.environ.get("LB_ALLOW_NO_CORPUS") != "1":
        print("⛔ _pair_production_path_probe: data/corpus/ is missing %s -- the brain would build without its "
              "cross-edge and corpus-learned organs. Link the corpus first." % _miss, file=sys.stderr)
        raise SystemExit(3)
    arm_list = FAMILIES[family]
    grader = {"pp": grade_seed_pp, "sn": grade_seed_sn, "cu": grade_seed_cu}[family]
    if family == "cu":
        os.environ["SIM_BACKEND"] = "cupy"
        workers = 1                                     # one brain on the one GPU at a time
    else:
        os.environ.setdefault("SIM_BACKEND", "numpy")
    if only:                                   # a de-risk subset (never a gate row): run only these arms, do not grade
        arm_list = [a for a in arm_list if a[0] in set(only)]
        grader = lambda _res: {"partial": True, "arms_run": [a[0] for a in arm_list]}   # noqa: E731
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"          # LTM off, as in every earlier family (declared)
    from research.runners.load_bearing_fraction import _spawn_arm, turn_group
    sdir = os.path.join(out_dir, "%s_seed%d" % (family, int(seed)))
    os.makedirs(sdir, exist_ok=True)

    def _one(spec):
        name, label, env = spec
        grp = turn_group(label)
        r = _spawn_arm(dict(env), grp, os.path.join(sdir, "%s.json" % name)) or {}
        print("[%s seed %d] arm %s done (%d replies)" % (family, seed, name, len(r)), flush=True)
        return name, label, env, grp, r

    import concurrent.futures as cf
    res = {"seed": int(seed), "family": family, "arms": {}, "backend": os.environ.get("SIM_BACKEND"),
           "argv": list(sys.argv), "workers": int(workers), "ltm": "off"}
    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for name, label, env, grp, r in ex.map(_one, arm_list):
            errs = [str(v.get("_error"))[:300] for v in r.values() if isinstance(v, dict) and v.get("_error")]
            if not r:
                errs.append("arm produced no replies (worker failed)")
            arm = {"label": label, "env": env, "turns": grp, "artifact": os.path.join(sdir, name + ".json"),
                   "errors": errs}
            if label == WD_LABEL:
                arm["wd"] = wd_record(r, grp)
                arm["tag_capture_at_recall"] = arm["wd"]["final_tc"]
            else:
                arm.update(turn_record(r, grp, label))
            res["arms"][name] = arm
    res["gates"] = grader(res)
    json.dump(res, open(os.path.join(out_dir, "%s_seed%d.json" % (family, int(seed))), "w"), indent=2, default=str)
    print(json.dumps(res["gates"], indent=2, default=str)[:6000], flush=True)
    return res


# ── pp grader (B3 + the episodic angle) ────────────────────────────────────────────────────────────────────────────
def _wd_q(arm, fact, ps):
    return ((arm.get("wd") or {}).get("probes") or {}).get("wd_q%s_%s" % (fact, ps), {}).get("outcome")


def _wd_r(arm, fact, ps):
    return (((arm.get("wd") or {}).get("probes") or {}).get("wd_r%s_%s" % (fact, ps), {}).get("episodic") or {})


def _wd_seq(arm, fact):
    """The fact's recall-question outcomes in protocol order (imm, [decay], eve, morn, d5)."""
    P = (arm.get("wd") or {}).get("probes") or {}
    return [P["wd_q%s_%s" % (fact, s)]["outcome"] for s in WD_SETS if "wd_q%s_%s" % (fact, s) in P]


def _wd_inst(arm, lesion=False):
    """The production path ran as registered on one ledger-ON 'wd' arm."""
    w = arm.get("wd") or {}
    ep, want = w.get("epochs") or [], w.get("expected_epoch_t_h") or []
    ok = bool(w.get("clock") == "wall" and w.get("steps_ok")
              and all((w.get("tell") or {}).get(f, {}).get("n_labels") == 1
                      and (w.get("tell") or {}).get(f, {}).get("new_blocks") == 1 for f in "ABC")
              and w.get("final_n_managed_blocks") == 3
              and len(want) >= 7 and len(ep) == len(want)
              and all(abs(float(e.get("t_h", -1.0)) - x) < 1e-6 for e, x in zip(ep, want))
              and all(not e.get("no_reader") and e.get("R") is not None and len(e.get("R") or []) >= 1 for e in ep))
    for e in ep:
        if lesion:
            ok &= bool(e.get("replay_lesioned") and all(x == 0.0 for x in e.get("R_eff") or [])
                       and abs(e.get("da_swr", -1) - _DA_TONIC_REF) < 1e-9)
        else:
            ok &= not e.get("replay_lesioned")
    return bool(ok)


def grade_seed_pp(res):
    """Amendment 7 (pp): sub-verdicts WD, NR, EP. Pure function of res["arms"]."""
    A = res["arms"]
    g = {}
    on = [k for k in ("wd_a", "wd_b", "wd_replaylesion", "wd_epi") if k in A]
    a, b = A["wd_a"].get("wd") or {}, A["wd_b"].get("wd") or {}
    pa, pb = a.get("probes") or {}, b.get("probes") or {}
    g["G0_null_clean"] = bool(pa and set(pa) == set(pb)
                              and all(pa[k].get("outcome") == pb[k].get("outcome")
                                      and pa[k].get("recalled_svo") == pb[k].get("recalled_svo")
                                      and pa[k].get("abstained") == pb[k].get("abstained")
                                      and pa[k].get("episodic") == pb[k].get("episodic") for k in pa)
                              and a.get("epochs") == b.get("epochs") and a.get("final_blocks") == b.get("final_blocks")
                              and a.get("final_tc") == b.get("final_tc"))
    g["P1_each_fact_recalled_at_once"] = bool(all(_wd_q(A["wd_a"], f, "imm") == "correct" for f in "ABC"))
    inst = {k: _wd_inst(A[k], lesion=(A[k].get("env") or {}).get("BRAIN_SLEEP_REPLAY_CAPTURE_LESION") == "1")
            for k in on}
    off = A.get("wd_ledger_off") or {}
    inst_off = bool((off.get("wd") or {}).get("steps_ok") and not (off.get("wd") or {}).get("ledger_key_present"))
    g["I1_production_path_as_registered"] = bool(all(inst.values()) and inst_off)
    g["I1_detail"] = dict(inst, wd_ledger_off=inst_off)
    gam = [(A[k].get("tag_capture_at_recall") or {}).get("gamma") for k in on]
    gam = [x for x in gam if x is not None]
    g["G_isolation_gamma_consistent"] = bool(not gam or all(abs(x - gam[0]) < 1e-6 for x in gam))
    gated = ("wd_a", "wd_b", "wd_replaylesion")
    errs = sum(len(A[k].get("errors") or []) for k in gated) + len(off.get("errors") or [])
    undef_out = any(v.get("outcome") == "undefined" for k in gated + ("wd_ledger_off",) if k in A
                    for v in ((A[k].get("wd") or {}).get("probes") or {}).values())
    # WD: the ordinary and salient outcomes, the replay edge, no confab
    g["WD1_salient_B_kept"] = bool(_wd_q(A["wd_a"], "B", "morn") == "correct" and _wd_q(A["wd_a"], "B", "d5") == "correct")
    g["WD2_ordinary_A_kept"] = bool(all(_wd_q(A["wd_a"], "A", ps) == "correct" for ps in ("preB", "morn", "d5")))
    g["WD4_no_confab"] = bool(not any(v.get("outcome") == "confab" for k in ("wd_a", "wd_b", "wd_replaylesion",
                                                                            "wd_ledger_off", "wd_epi") if k in A
                                      for v in ((A[k].get("wd") or {}).get("probes") or {}).values()))
    g["WD5_replay_edge_carries_A"] = bool(_wd_q(A["wd_replaylesion"], "A", "preB") in NOT_RECALLED)
    wd_undef = (not g["G0_null_clean"]) or (not g["P1_each_fact_recalled_at_once"]) or (not g["I1_production_path_as_registered"]) \
        or (not g["G_isolation_gamma_consistent"]) or errs > 0 or undef_out
    wd_core = all(g[k] for k in ("WD1_salient_B_kept", "WD2_ordinary_A_kept", "WD4_no_confab",
                                 "WD5_replay_edge_carries_A"))
    g["WD_verdict"] = "UNDEFINED" if wd_undef else ("GO" if wd_core else "NO-GO")
    # NR: no resurrection of a decayed fact on the production configuration (wd_a, wd_b; vacuous -> UNDEFINED). The
    # replay-lesion arm's trajectory is REPORTED (after B's news its decayed A may be captured by B's PRP: a lesion arm).
    resur, decayed = {}, False
    for k in ("wd_a", "wd_b", "wd_replaylesion"):
        for f in "ABC":
            seq = _wd_seq(A[k], f)
            resur["%s:%s" % (k, f)] = bool(any(seq[i] in NOT_RECALLED and "correct" in seq[i + 1:]
                                               for i in range(len(seq))))
            if k == "wd_a" and any(x in NOT_RECALLED for x in seq[:-1]):
                decayed = True
    g["NR_decayed_fact_exists"] = bool(decayed)
    g["NR1_no_resurrection"] = bool(not any(v for key, v in resur.items() if not key.startswith("wd_replaylesion")))
    g["NR_detail"] = resur
    g["NR_verdict"] = "UNDEFINED" if (wd_undef or not decayed) else ("GO" if g["NR1_no_resurrection"] else "NO-GO")
    # EP: episodic vs composer next day (wd_epi, BRAIN_EPISODIC_STORE=1 on numpy)
    ep_arm = A.get("wd_epi")
    if ep_arm is None:
        g["EP_verdict"] = "UNDEFINED"
        g["EP_I_episodic_wrote"] = False
    else:
        g["EP_I_episodic_wrote"] = bool(_wd_r(ep_arm, "A", "eve").get("formed") is True and inst.get("wd_epi"))
        agree = {}
        for ps in ("morn", "d5"):
            for f in "ABC":
                q, rr = _wd_q(ep_arm, f, ps), _wd_r(ep_arm, f, ps)
                agree["%s:%s" % (ps, f)] = {"composer": q, "episodic_in_memory": rr.get("in_memory"),
                                            "agree_where_composer_answers": (True if q != "correct"
                                                                             else rr.get("in_memory") is True)}
        g["EP1_episodic_agrees_where_composer_answers"] = bool(all(v["agree_where_composer_answers"]
                                                                   for v in agree.values()))
        g["EP_detail"] = agree
        g["EP2_reported_composer_same_as_wd_a"] = {
            k: (v.get("outcome") == ((A["wd_a"].get("wd") or {}).get("probes") or {}).get(k, {}).get("outcome"))
            for k, v in ((ep_arm.get("wd") or {}).get("probes") or {}).items() if v.get("kind") == "q"}
        ep_undef = (not g["EP_I_episodic_wrote"]) or len(ep_arm.get("errors") or []) > 0 \
            or any(v.get("outcome") == "undefined" for v in ((ep_arm.get("wd") or {}).get("probes") or {}).values())
        g["EP_verdict"] = "UNDEFINED" if ep_undef else ("GO" if g["EP1_episodic_agrees_where_composer_answers"] else "NO-GO")
    # REPORTED
    g["reported"] = {k: {"q_outcomes": {f: _wd_seq(A[k], f) for f in "ABC"},
                         "epochs_R": [e.get("R") for e in (A[k].get("wd") or {}).get("epochs") or []],
                         "epochs_da_swr": [e.get("da_swr") for e in (A[k].get("wd") or {}).get("epochs") or []],
                         "epochs_t_h": [e.get("t_h") for e in (A[k].get("wd") or {}).get("epochs") or []],
                         "expr_frac_by_probe": {lab: v.get("expr_frac") for lab, v in
                                                ((A[k].get("wd") or {}).get("probes") or {}).items()
                                                if v.get("kind") == "q"},
                         "errors": A[k].get("errors")}
                     for k in ("wd_a", "wd_b", "wd_replaylesion", "wd_ledger_off", "wd_epi") if k in A}
    # ATTRIBUTION (REPORTED; tools.lab): the replay lesion's change to A's probes vs the null rebuild's
    from tools.lab import attributable_to

    def _ndiff(x, y):
        return sum(1 for ps in WD_SETS if _wd_q(x, "A", ps) is not None and _wd_q(x, "A", ps) != _wd_q(y, "A", ps))
    treat = _ndiff(A["wd_a"], A["wd_replaylesion"])
    g["attribution"] = {"replay_lesion_vs_null": attributable_to("replay-edge lesion (A's probes) vs null rebuild",
                                                                treat, _ndiff(A["wd_a"], A["wd_b"])),
                        "treatment_diffs": treat}
    g["seed_verdict"] = {"WD": g["WD_verdict"], "NR": g["NR_verdict"], "EP": g["EP_verdict"]}
    return g


# ── sn grader (B4 + the weak telling) ──────────────────────────────────────────────────────────────────────────────
def _ep(arm):
    return ((arm or {}).get("sleep_replay_at_recall") or {}).get("epochs") or []


def _lesion_held(arm):
    """Every lesion on this arm held on the record itself (the turns' D1 input, the write gain, the SWR edge)."""
    env = arm.get("env") or {}
    les = env.get("BRAIN_DA_ENCODING_LESION") == "1"
    spare = env.get("BRAIN_DA_ENCODING_LESION_SPARE_SWR") == "1"
    seen = [x for x in arm.get("turn_da_seen") or [] if x is not None]
    enc = [x for x in arm.get("turn_da_enc") or [] if x.get("lesioned") is not None]
    ok = bool(seen) and bool(enc)
    if les:
        ok &= all(abs(x - _DA_TONIC_REF) < 1e-9 for x in seen)
        ok &= all(x.get("lesioned") is True and x.get("g") is not None and abs(float(x["g"]) - 1.0) < 1e-9 for x in enc)
    else:
        ok &= all(x.get("lesioned") is False for x in enc)
    for e in _ep(arm):
        ok &= not e.get("replay_lesioned")
        if les and spare:
            ok &= bool(e.get("da_lesion_spares_swr") is True and abs(e.get("da_seen_by_d1", -1) - e.get("da_swr", -2)) < 1e-9)
        elif les:
            ok &= bool(abs(e.get("da_seen_by_d1", -1) - _DA_TONIC_REF) < 1e-9 and "da_lesion_spares_swr" not in e)
        else:
            ok &= bool("da_lesion_spares_swr" not in e and abs(e.get("da_seen_by_d1", -1) - e.get("da_swr", -2)) < 1e-9)
    return bool(ok)


def _one_epoch_after_awake(arm):
    ep, aw = _ep(arm), arm.get("awake_until_h")
    return bool(len(ep) == 1 and aw is not None and aw >= AWAKE_H and ep[0].get("t_h", -1) > aw
                and not ep[0].get("no_reader"))


def grade_seed_sn(res):
    """Amendment 7 (sn). Pure function of res["arms"]. SN_REPORTED_ARMS enter no gate, error count, gamma check or
    UNDEFINED rule, except SN3 (no confab), which reads every arm."""
    A = res["arms"]
    o = {k: v.get("recall_outcome") for k, v in A.items()}
    gated = [k for k in A if k not in SN_REPORTED_ARMS]
    g = {}
    a, b = A["lsal_rc_a"], A["lsal_rc_b"]
    g["G0_null_clean"] = bool(o["lsal_rc_a"] == o["lsal_rc_b"] and a.get("recalled_svo") == b.get("recalled_svo")
                              and a.get("abstained") == b.get("abstained")
                              and a.get("tag_capture_at_recall") == b.get("tag_capture_at_recall")
                              and a.get("blocks_at_recall") == b.get("blocks_at_recall")
                              and a.get("sleep_replay_at_recall") == b.get("sleep_replay_at_recall"))
    g["P1_immediate_precondition"] = bool(o["sal_imm_rc"] == "correct" and o["neu_imm_rc"] == "correct")
    long_on = [k for k in ("lsal_rc_a", "lsal_rc_b", "lneu_rc", "lsal_rc_wakelesion") if k in A]
    inst = all(_one_epoch_after_awake(A[k]) for k in long_on)
    inst &= all(len(_ep(A[k])) == 0 for k in ("sal_imm_rc", "neu_imm_rc"))
    inst &= all(_lesion_held(A[k]) for k in gated)
    g["I1_one_epoch_after_waking_and_lesions_held"] = bool(inst)
    g["I1_reported_arms"] = {k: {"lesion_held": _lesion_held(A[k]), "n_epochs": len(_ep(A[k]))}
                             for k in SN_REPORTED_ARMS if k in A and A[k].get("ledger_key_present")}
    gam = [(A[k].get("tag_capture_at_recall") or {}).get("gamma") for k in gated
           if (A[k].get("env") or {}).get("BRAIN_DA_TAG_CAPTURE") == "1"]
    gam = [x for x in gam if x is not None]
    g["G_isolation_gamma_consistent"] = bool(not gam or all(abs(x - gam[0]) < 1e-6 for x in gam))
    errs = sum(len(A[k].get("errors") or []) for k in gated)
    g["SN1_salient_vs_neutral_long_delay"] = bool(o["lsal_rc_a"] == "correct" and o["lneu_rc"] in NOT_RECALLED)
    g["SN2_waking_salience_load_bearing"] = bool(o["lsal_rc_wakelesion"] in NOT_RECALLED)
    g["SN3_no_confab"] = bool(all(v != "confab" for v in o.values()))

    def _rep(k):
        arm = A.get(k)
        if arm is None:
            return None
        ep = _ep(arm)
        return {"outcome": o.get(k), "R": (ep[0].get("R") if ep else None), "da_swr": (ep[0].get("da_swr") if ep else None),
                "da_seen_by_d1": (ep[0].get("da_seen_by_d1") if ep else None),
                "frac_z_gt_half_at_onset": (ep[0].get("pre_frac_z_gt_half") if ep else None),
                "frac_z_gt_half_at_recall": [bl.get("frac_synapses_z_gt_half") for bl in arm.get("blocks_at_recall") or []],
                "turn_da": arm.get("turn_da"), "errors": arm.get("errors")}
    g["reported"] = {k: _rep(k) for k in A}
    undefined = (not g["G0_null_clean"]) or (not g["P1_immediate_precondition"]) \
        or (not g["I1_one_epoch_after_waking_and_lesions_held"]) or (not g["G_isolation_gamma_consistent"]) \
        or errs > 0 or any(o[k] == "undefined" for k in gated)
    from tools.lab import attributable_to

    def _nd(x, y):
        return int(x.get("recalled_svo") != y.get("recalled_svo")) + int(x.get("abstained") != y.get("abstained"))
    treat = _nd(A["lsal_rc_a"], A["lsal_rc_wakelesion"])
    g["attribution"] = {"wake_lesion_vs_null": attributable_to("waking-only DA lesion vs null rebuild", treat,
                                                               _nd(A["lsal_rc_a"], A["lsal_rc_b"])),
                        "treatment_diffs": treat}
    core = all(g[k] for k in ("SN1_salient_vs_neutral_long_delay", "SN2_waking_salience_load_bearing", "SN3_no_confab"))
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


# ── cu grader (D3, the reference 3090) ─────────────────────────────────────────────────────────────────────────────
def grade_seed_cu(res):
    """Amendment 7 (cu). Pure function of res (its arms and backend)."""
    A = res["arms"]
    o = {k: v.get("recall_outcome") for k, v in A.items() if "recall_outcome" in v}
    gated = [k for k in A if k not in CU_REPORTED_ARMS]
    g = {}
    a, b = A["cu_off_a"], A["cu_off_b"]
    g["G0_cupy_rebuild_identical"] = bool(o["cu_off_a"] == o["cu_off_b"] and a.get("turn_da") == b.get("turn_da")
                                          and a.get("turn_answer") == b.get("turn_answer")
                                          and a.get("recalled_svo") == b.get("recalled_svo"))
    g["I_backend_cupy"] = bool(res.get("backend") == "cupy")
    on = [k for k in ("cu_on", "cu_neu_rc", "cu_neu_norc") if k in A]
    gam = [(A[k].get("tag_capture_at_recall") or {}).get("gamma") for k in on]
    d1 = [(A[k].get("tag_capture_at_recall") or {}).get("d1_a_go") for k in on]
    g["CU1_calibration_on_cupy"] = {"gamma": gam, "d1_a_go": d1, "numpy_gamma_ref": NUMPY_GAMMA_REF}
    g["G_isolation_gamma_consistent"] = bool(gam and None not in gam and all(abs(x - gam[0]) < 1e-6 for x in gam)
                                             and None not in d1 and all(abs(x - d1[0]) < 1e-9 for x in d1))
    on_da, off_da = A["cu_on"].get("turn_da") or [], a.get("turn_da") or []
    g["CU2_no_drift_in_other_organs"] = bool(on_da and len(on_da) == len(off_da) and None not in on_da
                                             and all(x == y for x, y in zip(on_da, off_da))
                                             and (A["cu_on"].get("turn_answer") or [])[:2] == (a.get("turn_answer") or [])[:2])
    g["CU3_capture_edge_on_cupy"] = bool(o["cu_neu_rc"] == "correct" and o["cu_neu_norc"] in NOT_RECALLED)
    g["CU4_no_confab"] = bool(all(v != "confab" for v in o.values()))
    nr = A.get("cu_on_norestore") or {}
    nr_da = nr.get("turn_da") or []
    ep = _ep(A["cu_neu_rc"])
    g["reported"] = {
        "cu_on_outcome": o.get("cu_on"),
        "norestore_turns_with_da_diff_vs_off": (sum(1 for x, y in zip(nr_da, off_da) if x != y) if nr_da else None),
        "fixed_turns_with_da_diff_vs_off": sum(1 for x, y in zip(on_da, off_da) if x != y),
        "cu_neu_rc_R_epoch0": (ep[0].get("R") if ep else None), "cu_neu_rc_da_swr": (ep[0].get("da_swr") if ep else None),
        "cu_wd": ({"q_outcomes": {f: _wd_seq(A["cu_wd"], f) for f in "ABC"},
                   "episodic": {"%s:%s" % (ps, f): _wd_r(A["cu_wd"], f, ps).get("in_memory")
                                for ps in ("eve", "morn", "d5") for f in "ABC"},
                   "inst": _wd_inst(A["cu_wd"]), "errors": A["cu_wd"].get("errors")} if "cu_wd" in A else None)}
    from tools.lab import attributable_to
    g["attribution"] = {"norestore_vs_fixed_da_drift": (attributable_to(
        "pre-fix cupy reseed vs the fixed restore (turns whose DA differs from OFF)",
        g["reported"]["norestore_turns_with_da_diff_vs_off"], g["reported"]["fixed_turns_with_da_diff_vs_off"])
        if g["reported"]["norestore_turns_with_da_diff_vs_off"] is not None else None)}
    errs = sum(len(A[k].get("errors") or []) for k in gated)
    undefined = (not g["G0_cupy_rebuild_identical"]) or (not g["I_backend_cupy"]) \
        or (not g["G_isolation_gamma_consistent"]) or errs > 0 or any(o.get(k) == "undefined" for k in gated)
    core = all(g[k] for k in ("CU2_no_drift_in_other_organs", "CU3_capture_edge_on_cupy", "CU4_no_confab"))
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


# ── 6-seed combine ────────────────────────────────────────────────────────────────────────────────────────────────
def aggregate(d, family):
    grader = {"pp": grade_seed_pp, "sn": grade_seed_sn, "cu": grade_seed_cu}[family]
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "%s_seed*.json" % family))):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if r.get("family") == family and int(r.get("seed", -1)) in SEEDS:
            rows.append(r)
    for r in rows:
        r["gates"] = grader(r)                          # re-graded with the current code
    complete = sorted(r["seed"] for r in rows) == sorted(SEEDS)
    out = {"family": family, "seeds": sorted(r["seed"] for r in rows)}
    subs = ("WD", "NR", "EP") if family == "pp" else ("seed",)
    for s in subs:
        v = {r["seed"]: (r["gates"]["seed_verdict"][s] if family == "pp" else r["gates"]["seed_verdict"]) for r in rows}
        n_go = sum(1 for x in v.values() if x == "GO")
        out["%s_seed_verdicts" % s] = v
        out["%s_n_go" % s] = n_go
        out["%s_verdict" % s] = "INCOMPLETE" if not complete else ("GO" if n_go == 6 else "NO-GO")
    if family == "sn" and rows:
        diffs = [int(r["gates"]["outcomes"]["lsal_rc_a"] == "correct")
                 - int(r["gates"]["outcomes"]["lsal_rc_wakelesion"] == "correct") for r in rows]
        out["signflip_p_salient_intact_minus_wakelesion"] = seed_signflip_p(diffs)
        out["diffs_salient_intact_minus_wakelesion"] = diffs
    if family == "pp" and rows:
        diffs = [int(_wd_q(r["arms"]["wd_a"], "A", "morn") == "correct")
                 - int(_wd_q(r["arms"]["wd_replaylesion"], "A", "morn") == "correct") for r in rows]
        out["signflip_p_A_intact_minus_replaylesion"] = seed_signflip_p(diffs)
        out["diffs_A_intact_minus_replaylesion"] = diffs
    json.dump(out, open(os.path.join(d, "aggregate_%s.json" % family), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


# ── fake-substrate design day (no brain; not a gate row) ───────────────────────────────────────────────────────────
class _FakeD1:
    """a(DA) linear from tonic to the D1 pool's own ceiling; deterministic (as tests/test_sleep_replay_capture.py)."""

    def __init__(self, *a, **kw):
        from webapp import da_tag_capture as T
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        import numpy as np
        from webapp import da_tag_capture as T
        from webapp import sleep_replay_capture as S
        return float(np.clip((float(da) - T._DA_TONIC) / (S.DA_SWR_FULL - T._DA_TONIC), 0.0, 1.0)), None


class _FakeComposer:
    """Block-major store with a substrate read-back: the coherence of a block's current weights with the phasor pattern
    it was written with (high while the increment is expressed, ~1/sqrt(D) at baseline). As the tests' fake."""

    def __init__(self, D=64, seed=0):
        import numpy as np
        self.D = D
        self.store_conns = []
        self.patterns = []
        self._rng = np.random.default_rng(seed)

    def store(self, g=1.0):
        import numpy as np
        u = np.exp(2j * np.pi * self._rng.random(self.D))
        i = len(self.patterns)
        self.patterns.append(u)
        trig = 1000 + i * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(g * u[k])) for k in range(self.D)]

    def coherence(self, i):
        import numpy as np
        w = np.array([complex(x[2]) for x in self.store_conns[i * self.D:(i + 1) * self.D]])
        return float(abs(np.mean(np.conj(self.patterns[i]) * w)) / max(1e-12, float(np.mean(np.abs(w)))))

    def _block_role_scores(self, i):
        c = self.coherence(i)
        return {"agent": ("a", 1.0, c, None), "action": ("b", 1.0, c, None), "patient": ("c", 1.0, c, None)}


def design_fake(out_path=None, lesion_replay=False):
    """The registered 'wd' day's clock and sleep schedule on the FAKE substrate (the tests' fake composer and linear D1),
    driven through the real ledger + sleep route with the wall clock fed by `set_wall_clock`. Neutral turns read DA 0.5,
    the salient frame 1.0 / 1.1 / fact 0.5 / 1.1 / 1.0 (the tests' SALIENT). Not brain evidence; not a gate row."""
    from webapp import da_tag_capture_chat as W
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL, _WORLD_STEPS, _DATC_SALIENT
    from research.runners.load_bearing_fraction import turn_group
    keys = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_SLEEP_REPLAY_CAPTURE",
            "BRAIN_SLEEP_REPLAY_CAPTURE_LESION")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "wall",
                       "BRAIN_SLEEP_REPLAY_CAPTURE": "1"})
    if lesion_replay:
        os.environ["BRAIN_SLEEP_REPLAY_CAPTURE_LESION"] = "1"
    else:
        os.environ.pop("BRAIN_SLEEP_REPLAY_CAPTURE_LESION", None)
    prev_d1, prev_clock, prev_off = W.SpikingD1Activation, W._WALL_CLOCK, W._WORLD_OFFSET_H
    vt = [0.0]
    W.SpikingD1Activation = _FakeD1
    W.set_wall_clock(lambda: vt[0])
    W._WORLD_OFFSET_H = 0.0
    try:
        comp = _FakeComposer()
        for _ in range(2):
            comp.store()                                  # build-time knowledge (unmanaged)

        class _Inner:
            pass

        class _Chat:
            pass
        chat = _Chat()
        chat.inner = _Inner()
        chat.inner.composer = comp
        salient = {_DATC_SALIENT[0]: 1.0, _DATC_SALIENT[1]: 1.1, _DATC_SALIENT[3]: 1.1, _DATC_SALIENT[4]: 1.0}
        tell = {v: k for k, v in WD_TELL_TEXT.items()}
        block_of = {}
        rows = []
        for lab in turn_group(WD_LABEL):
            if lab in _WORLD_STEPS:
                kind = _WORLD_STEPS[lab]
                if kind.startswith("vclock:idle"):
                    vt[0] += float(kind.rsplit(":", 1)[1])
                    W.tick_chat(chat)
                continue
            text = _TURN_BY_LABEL[lab][1]
            chat._last_da_drives = {"da_level": salient.get(text, 0.5)}
            W.observe_chat_turn(chat, seed=7)
            if text in tell:
                block_of[tell[text]] = len(comp.patterns)
                comp.store(g=1.0)
            W.after_store_chat(chat)
            meta = _probe_meta(lab)
            if meta is not None and meta[0] == "q":
                cap = chat._da_tag_capture
                t = cap.ledger.t
                summ = cap.ledger.summary()
                rows.append({"label": lab, "t_h": round(t, 6),
                             "coherence": {f: round(comp.coherence(i), 6) for f, i in block_of.items()},
                             "expr_frac": {f: expr_frac(summ[i - 2], t) for f, i in block_of.items()},
                             "frac_z_gt_half": {f: summ[i - 2]["frac_synapses_z_gt_half"]
                                                for f, i in block_of.items()}})
        src = chat._da_tag_capture._src.summary()
        out = {"what": "the registered 'wd' day on the FAKE substrate (not brain evidence, not a gate row)",
               "replay_lesioned": bool(lesion_replay), "probes": rows,
               "epochs": [{k: e.get(k) for k in ("t_h", "R", "sum_R_eff", "da_swr", "a_eff_mean",
                                                 "pre_frac_z_gt_half")} for e in src["epochs"]]}
    finally:
        W.SpikingD1Activation, W._WORLD_OFFSET_H = prev_d1, prev_off
        W.set_wall_clock(prev_clock)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump(out, open(out_path, "w"), indent=1)
    return out


# ── selftest (no brain): every gate through its failing directions ─────────────────────────────────────────────────
def _syn_wd(q=None, epi_in=True, epi_formed=True, n_epochs=7, lesion=False, clock="wall", new_blocks=1,
            n_managed=3, steps_ok=True, ledger=True, gamma=32.7735, errors=None, epoch_lesion=None):
    """A synthetic 'wd' arm record in the shape `wd_record` writes. `q`: {(fact, set): outcome} overrides."""
    base = {("C", "imm"): "correct", ("C", "decay"): "abstain", ("C", "eve"): "abstain", ("C", "morn"): "abstain",
            ("C", "d5"): "abstain", ("A", "imm"): "correct", ("A", "preB"): "correct", ("A", "eve"): "correct",
            ("A", "morn"): "correct", ("A", "d5"): "correct", ("B", "imm"): "correct", ("B", "eve"): "correct",
            ("B", "morn"): "correct", ("B", "d5"): "correct"}
    base.update(q or {})
    probes = {}
    for (f, s), out in base.items():
        probes["wd_q%s_%s" % (f, s)] = {"kind": "q", "fact": f, "set": s, "outcome": out,
                                        "recalled_svo": (WD_FACTS[f] if out == "correct" else
                                                         (["x", "y", "z"] if out == "confab" else None)),
                                        "abstained": out == "abstain"}
        if s in ("eve", "morn", "d5"):
            probes["wd_r%s_%s" % (f, s)] = {"kind": "r", "fact": f, "set": s,
                                            "outcome": "in_memory" if epi_in else "not_in_memory",
                                            "episodic": {"formed": epi_formed, "in_memory": epi_in}}
    want = [1.0 + i for i in range(7)]
    el = lesion if epoch_lesion is None else epoch_lesion
    ep = [{"t_h": x, "R": [0.3, 0.05, 0.6], "R_eff": ([0.0, 0.0, 0.0] if el else [0.3, 0.05, 0.6]),
           "replay_lesioned": el, "da_swr": (_DA_TONIC_REF if el else 1.0), "no_reader": False}
          for x in want[:n_epochs]]
    w = {"probes": probes, "tell": {f: {"n_labels": 1, "new_blocks": new_blocks} for f in "ABC"},
         "steps": [], "steps_ok": steps_ok, "expected_epoch_t_h": want, "epochs": ep, "clock": clock,
         "final_n_managed_blocks": n_managed, "final_blocks": [{"z_mean": 1.0}], "ledger_key_present": ledger,
         "final_tc": {"gamma": gamma}}
    return {"wd": w, "errors": list(errors or []), "env": {"BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"} if lesion else {},
            "tag_capture_at_recall": {"gamma": gamma if ledger else None}}


def _syn_pp(**over):
    lesq = {("A", "preB"): "abstain", ("A", "eve"): "abstain", ("A", "morn"): "abstain", ("A", "d5"): "abstain"}
    arms = {"wd_a": _syn_wd(), "wd_b": _syn_wd(), "wd_replaylesion": _syn_wd(q=lesq, lesion=True),
            "wd_ledger_off": _syn_wd(q={("C", "decay"): "correct", ("C", "eve"): "correct", ("C", "morn"): "correct",
                                        ("C", "d5"): "correct"}, ledger=False),
            "wd_epi": _syn_wd()}
    for k, kw in over.items():
        arms[k] = _syn_wd(**kw)
    return {"arms": arms}


def _syn_turn(outcome="correct", env=None, n_epochs=1, epoch_t=4.2, awake=4.0, seen=0.7, enc_les=False, g=1.3,
              spare=False, da_seen_epoch=None, da_swr=0.9, gamma=32.7735, errors=None, ledger=True, turn_da=None,
              answers=None, svo=None):
    env = dict(env or {})
    ep = []
    for _ in range(n_epochs):
        e = {"t_h": epoch_t, "R": [0.4], "R_eff": [0.4], "replay_lesioned": False, "da_swr": da_swr,
             "da_seen_by_d1": (da_swr if da_seen_epoch is None else da_seen_epoch), "no_reader": False}
        if spare:
            e["da_lesion_spares_swr"] = True
        ep.append(e)
    return {"recall_outcome": outcome, "recalled_svo": (svo if svo is not None else (FACT if outcome == "correct" else None)),
            "abstained": outcome == "abstain", "env": env,
            "tag_capture_at_recall": {"gamma": gamma if ledger else None, "d1_a_go": 0.4 if ledger else None},
            "blocks_at_recall": [{"z_mean": 1.0}], "sleep_replay_at_recall": {"epochs": ep} if ledger else None,
            "awake_until_h": awake, "turn_da": list(turn_da or [0.6, 0.7, 0.5, 0.4, 0.3, 0.5]),
            "turn_answer": list(answers or ["a", "b", "c", "d", "e", "f"]),
            "turn_da_seen": ([seen] * 6 if ledger else [None] * 6),
            "turn_da_enc": [{"g": (1.0 if enc_les else g), "lesioned": enc_les}] * 6,
            "ledger_key_present": ledger, "errors": list(errors or [])}


def _syn_sn(**over):
    wl = dict(env=WAKE_LES, seen=_DA_TONIC_REF, enc_les=True, spare=True)
    arms = {"lsal_rc_a": _syn_turn(), "lsal_rc_b": _syn_turn(), "lneu_rc": _syn_turn("abstain"),
            "lsal_rc_wakelesion": _syn_turn("abstain", **wl),
            "sal_imm_rc": _syn_turn(n_epochs=0, awake=None), "neu_imm_rc": _syn_turn(n_epochs=0, awake=None),
            "lsal_rc_dalesion": _syn_turn("abstain", env=LES, seen=_DA_TONIC_REF, enc_les=True,
                                          da_seen_epoch=_DA_TONIC_REF),
            "sal_rc_wakelesion": _syn_turn(n_epochs=1, epoch_t=0.2, awake=None, **wl),
            "lsal_ledger_off": _syn_turn(ledger=False, n_epochs=0),
            "wk_imm_rc": _syn_turn(n_epochs=0, awake=None), "wk_imm_off": _syn_turn(ledger=False, n_epochs=0),
            "wk_night_rc": _syn_turn(awake=None), "wk_night_off": _syn_turn(ledger=False, n_epochs=0)}
    for k, kw in over.items():
        arms[k] = _syn_turn(**kw)
    return {"arms": arms}


def _syn_cu(backend="cupy", **over):
    arms = {"cu_off_a": _syn_turn(ledger=False, n_epochs=0), "cu_off_b": _syn_turn(ledger=False, n_epochs=0),
            "cu_on": _syn_turn(awake=None), "cu_neu_rc": _syn_turn(awake=None),
            "cu_neu_norc": _syn_turn("abstain", awake=None),
            "cu_on_norestore": _syn_turn(awake=None, turn_da=[0.6, 0.71, 0.52, 0.4, 0.3, 0.5])}
    for k, kw in over.items():
        arms[k] = _syn_turn(**kw)
    return {"arms": arms, "backend": backend}


def selftest():
    import tempfile
    checks = {}
    from webapp import da_tag_capture as T
    from webapp import sleep_replay_capture as S
    from webapp import continuous_engine as CE
    import research.runners.onebrain_regression_battery as B
    from research.runners.load_bearing_fraction import turn_group
    checks["constants: TAU_EARLY_H / sleep onset / night period mirror the modules"] = \
        (TAU_EARLY_H == T.TAU_EARLY_H and abs(SLEEP_ONSET_H - CE.SLEEP_IDLE_SEC / 3600.0) < 1e-12
         and NIGHT_PERIOD_H == S.NIGHT_PERIOD_H and B.WD_FACTS == WD_FACTS)
    # the registered day: structure, clock steps, epoch schedule
    grp = turn_group(WD_LABEL)
    steps, t = [], 0.0
    for lab in grp:
        kind = B._WORLD_STEPS.get(lab)
        if kind is not None and kind.startswith("vclock:idle"):
            sec = float(kind.rsplit(":", 1)[1])
            steps.append({"kind": kind, "seconds": sec, "virtual_t_start_s": t})
            t += sec
    probes = [lab for lab in grp if _probe_meta(lab)]
    sleepish = [s for s in steps if s["seconds"] >= CE.SLEEP_IDLE_SEC]
    checks["wd day: vclock start first, 23 probes, each fact told once, 3 day pauses + night + 3-day idle"] = \
        (grp[0] == "wd_vclock" and B._WORLD_STEPS[grp[0]] == "vclock:start" and len(probes) == 23
         and all(sum(1 for lab in grp if B._TURN_BY_LABEL[lab][1] == txt) == 1 for txt in WD_TELL_TEXT.values())
         and len(sleepish) == 5 and [s["seconds"] for s in sleepish] == [1500.0, 420.0, 600.0, 39600.0, 259200.0])
    checks["wd day: every other gap is below the sleep threshold (no hidden epoch)"] = \
        all(s["seconds"] < CE.SLEEP_IDLE_SEC for s in steps if s not in sleepish)
    exp = expected_epoch_times(steps)
    checks["wd day: production runs exactly 7 SWR epochs (3 pauses + night + 3 idle nights)"] = len(exp) == 7
    first_turn = [lab for lab in grp if lab not in B._WORLD_STEPS][0]
    checks["wd day: the first text turn resets the session, no other does"] = \
        (B._TURN_BY_LABEL[first_turn][3] is True
         and not any(B._TURN_BY_LABEL[lab][3] for lab in grp if lab not in B._WORLD_STEPS and lab != first_turn))
    tC = next(s for s in steps if s["seconds"] == 1500.0)["virtual_t_start_s"] / 3600.0
    checks["wd day: C is ~3 h old at its decay probe (2.9-3.2 h)"] = 2.9 <= tC <= 3.2
    checks["tick offsets: 40 s -> 1 tick; 7 min -> 20 incl. the 300-s sleep tick; 11 h -> 189; 72 h -> 250"] = \
        (B._vidle_tick_offsets(40) == [20.0] and len(B._vidle_tick_offsets(420)) == 20
         and 300.0 in B._vidle_tick_offsets(420) and len(B._vidle_tick_offsets(39600)) == 189
         and len(B._vidle_tick_offsets(259200)) == 250)
    checks["expr_frac: fresh 1, decayed ~0, captured 1"] = \
        (expr_frac({"t_w": 0.0, "z_mean": 0.0}, 0.0) == 1.0 and expr_frac({"t_w": 0.0, "z_mean": 0.0}, 24.0) < 1e-6
         and abs(expr_frac({"t_w": 0.0, "z_mean": 1.0}, 24.0) - 1.0) < 1e-9)
    checks["outcome_for: correct / abstain / confab / undefined"] = \
        (outcome_for({"recalled_svo": ["dog", "store", "memory"]}, WD_FACTS["B"]) == "correct"
         and outcome_for({"recalled_svo": None, "abstained": True}, WD_FACTS["B"]) == "abstain"
         and outcome_for({"recalled_svo": ["dog", "chase", "cat"]}, WD_FACTS["B"]) == "confab"
         and outcome_for({"_error": "x"}, WD_FACTS["B"]) == "undefined"
         and outcome_for({"recalled_svo": ["dog", "chase", "ball"], "answer": "perhaps the dog chases the ball  [a guess "
                          "from what I've learned -- not something I was taught]"}, WD_FACTS["B"]) == "guess"
         and outcome_for({"recalled_svo": ["dog", "store", "memory"], "hypothesis": True}, WD_FACTS["B"]) == "guess")
    # ── pp grader
    sv = lambda res: grade_seed_pp(res)["seed_verdict"]   # noqa: E731
    checks["pp: designed pattern -> WD GO, NR GO, EP GO"] = sv(_syn_pp()) == {"WD": "GO", "NR": "GO", "EP": "GO"}
    for name, over, key, want in (
            ("salient B lost next morning", {"wd_a": {"q": {("B", "morn"): "abstain"}},
                                            "wd_b": {"q": {("B", "morn"): "abstain"}}}, "WD", "NO-GO"),
            ("ordinary A lost next morning", {"wd_a": {"q": {("A", "morn"): "abstain", ("A", "d5"): "abstain"}},
                                             "wd_b": {"q": {("A", "morn"): "abstain", ("A", "d5"): "abstain"}}},
             "WD", "NO-GO"),
            ("A kept with the replay edge cut", {"wd_replaylesion": {"lesion": True}}, "WD", "NO-GO"),
            ("today's default confabulates", {"wd_ledger_off": {"q": {("C", "d5"): "confab"}, "ledger": False}},
             "WD", "NO-GO"),
            ("rebuild differs", {"wd_b": {"q": {("C", "eve"): "correct"}}}, "WD", "UNDEFINED"),
            ("a fact not recalled at once", {"wd_a": {"q": {("B", "imm"): "abstain"}},
                                            "wd_b": {"q": {("B", "imm"): "abstain"}}}, "WD", "UNDEFINED"),
            ("six epochs ran, not seven", {"wd_a": {"n_epochs": 6}, "wd_b": {"n_epochs": 6}}, "WD", "UNDEFINED"),
            ("a telling stored two blocks", {"wd_a": {"new_blocks": 2}, "wd_b": {"new_blocks": 2}}, "WD", "UNDEFINED"),
            ("the ledger ran on the turn clock", {"wd_a": {"clock": "turn"}, "wd_b": {"clock": "turn"}},
             "WD", "UNDEFINED"),
            ("the off arm carried a ledger", {"wd_ledger_off": {"ledger": True}}, "WD", "UNDEFINED"),
            ("the replay lesion did not hold", {"wd_replaylesion": {"q": {("A", "preB"): "abstain"}, "lesion": True,
                                                                    "epoch_lesion": False}}, "WD", "UNDEFINED"),
            ("a world step failed", {"wd_a": {"steps_ok": False}, "wd_b": {"steps_ok": False}}, "WD", "UNDEFINED"),
            ("gamma differs", {"wd_replaylesion": {"q": {("A", "preB"): "abstain", ("A", "eve"): "abstain",
                                                         ("A", "morn"): "abstain", ("A", "d5"): "abstain"},
                                                   "lesion": True, "gamma": 30.0}}, "WD", "UNDEFINED"),
            ("a gated arm errs", {"wd_a": {"errors": ["boom"]}, "wd_b": {"errors": ["boom"]}}, "WD", "UNDEFINED"),
            ("A flagged guess where the lesion arm forgets A still reads not-recalled",
             {"wd_replaylesion": {"q": {("A", "preB"): "guess", ("A", "eve"): "abstain", ("A", "morn"): "abstain",
                                        ("A", "d5"): "abstain"}, "lesion": True}}, "WD", "GO"),
            ("C comes back after a flagged guess (still a resurrection)",
             {"wd_a": {"q": {("C", "decay"): "guess", ("C", "d5"): "correct"}},
              "wd_b": {"q": {("C", "decay"): "guess", ("C", "d5"): "correct"}}}, "NR", "NO-GO"),
            ("C resurrected after its decay probe", {"wd_a": {"q": {("C", "morn"): "correct", ("C", "d5"): "correct"}},
                                                    "wd_b": {"q": {("C", "morn"): "correct",
                                                                   ("C", "d5"): "correct"}}}, "NR", "NO-GO"),
            ("no fact ever decayed (vacuous)", {"wd_a": {"q": {("C", "decay"): "correct", ("C", "eve"): "correct",
                                                               ("C", "morn"): "correct", ("C", "d5"): "correct"}},
                                                "wd_b": {"q": {("C", "decay"): "correct", ("C", "eve"): "correct",
                                                               ("C", "morn"): "correct", ("C", "d5"): "correct"}}},
             "NR", "UNDEFINED"),
            ("the lesion arm's A comes back after B (REPORTED, not NR)",
             {"wd_replaylesion": {"q": {("A", "preB"): "abstain", ("A", "eve"): "correct", ("A", "morn"): "correct",
                                        ("A", "d5"): "correct"}, "lesion": True}}, "NR", "GO"),
            ("episodic says not-in-memory where the composer answers", {"wd_epi": {"epi_in": False}}, "EP", "NO-GO"),
            ("the episodic organ never wrote", {"wd_epi": {"epi_formed": False}}, "EP", "UNDEFINED"),
            ("episodic arm errs", {"wd_epi": {"errors": ["boom"]}}, "EP", "UNDEFINED")):
        checks["pp: %s -> %s %s" % (name, key, want)] = sv(_syn_pp(**over))[key] == want
    # ── sn grader
    svs = lambda res: grade_seed_sn(res)["seed_verdict"]   # noqa: E731
    wl = dict(env=WAKE_LES, seen=_DA_TONIC_REF, enc_les=True, spare=True)
    checks["sn: designed pattern -> GO"] = svs(_syn_sn()) == "GO"
    for name, over, want in (
            ("neutral kept at long delay", {"lneu_rc": {}}, "NO-GO"),
            ("salient lost at long delay", {"lsal_rc_a": {"outcome": "abstain"}, "lsal_rc_b": {"outcome": "abstain"}},
             "NO-GO"),
            ("waking-only lesion keeps the salient fact", {"lsal_rc_wakelesion": dict(outcome="correct", **wl)}, "NO-GO"),
            ("waking-only lesion pinned the SWR read", {"lsal_rc_wakelesion": dict(outcome="abstain",
                                                                                  da_seen_epoch=_DA_TONIC_REF, **wl)},
             "UNDEFINED"),
            ("waking-only lesion left a turn's D1 read live", {"lsal_rc_wakelesion": dict(
                outcome="abstain", env=WAKE_LES, seen=0.8, enc_les=True, spare=True)}, "UNDEFINED"),
            ("waking-only lesion left the write gain live", {"lsal_rc_wakelesion": dict(
                outcome="abstain", env=WAKE_LES, seen=_DA_TONIC_REF, enc_les=False, spare=True)}, "UNDEFINED"),
            ("the spare knob never reached the epoch", {"lsal_rc_wakelesion": dict(
                outcome="abstain", env=WAKE_LES, seen=_DA_TONIC_REF, enc_les=True, spare=False)}, "UNDEFINED"),
            ("rebuild differs", {"lsal_rc_b": {"outcome": "abstain"}}, "UNDEFINED"),
            ("salient not recalled at once", {"sal_imm_rc": {"outcome": "abstain", "n_epochs": 0, "awake": None}},
             "UNDEFINED"),
            ("two epochs on a long-delay arm", {"lsal_rc_a": {"n_epochs": 2}, "lsal_rc_b": {"n_epochs": 2}},
             "UNDEFINED"),
            ("the epoch came before the awake mark", {"lneu_rc": {"outcome": "abstain", "epoch_t": 3.0}}, "UNDEFINED"),
            ("a gated arm errs", {"lneu_rc": {"outcome": "abstain", "errors": ["boom"]}}, "UNDEFINED"),
            ("a REPORTED arm confabulates", {"wk_night_off": {"outcome": "confab", "ledger": False, "n_epochs": 0,
                                                             "svo": ["dog", "chase", "cat"]}}, "NO-GO"),
            ("a REPORTED arm errs (not gated)", {"wk_night_off": {"ledger": False, "n_epochs": 0,
                                                                 "errors": ["boom"]}}, "GO")):
        checks["sn: %s -> %s" % (name, want)] = svs(_syn_sn(**over)) == want
    # ── cu grader
    svc = lambda res: grade_seed_cu(res)["seed_verdict"]   # noqa: E731
    checks["cu: designed pattern -> GO"] = svc(_syn_cu()) == "GO"
    for name, res, want in (
            ("not on cupy", _syn_cu(backend="numpy"), "UNDEFINED"),
            ("cupy rebuild not identical", _syn_cu(cu_off_b=dict(ledger=False, n_epochs=0,
                                                                 turn_da=[0.6, 0.7, 0.5, 0.4, 0.3, 0.51])), "UNDEFINED"),
            ("the ledger drifted another organ's DA", _syn_cu(cu_on=dict(awake=None,
                                                                         turn_da=[0.6, 0.7, 0.52, 0.4, 0.3, 0.5])),
             "NO-GO"),
            ("the ledger changed an early answer", _syn_cu(cu_on=dict(awake=None,
                                                                      answers=["a", "X", "c", "d", "e", "f"])), "NO-GO"),
            ("no rescue on cupy", _syn_cu(cu_neu_rc=dict(outcome="abstain", awake=None)), "NO-GO"),
            ("gamma differs on cupy", _syn_cu(cu_neu_rc=dict(awake=None, gamma=31.0)), "UNDEFINED"),
            ("a gated arm confabulates", _syn_cu(cu_on=dict(outcome="confab", awake=None,
                                                            svo=["dog", "chase", "cat"])), "NO-GO"),
            ("the REPORTED pre-fix arm errs (not gated)", _syn_cu(cu_on_norestore=dict(awake=None, errors=["x"])),
             "GO")):
        checks["cu: %s -> %s" % (name, want)] = svc(res) == want
    r = grade_seed_cu(_syn_cu())
    checks["cu: the pre-fix drift is REPORTED (2 turns differ), the fixed arm 0"] = \
        (r["reported"]["norestore_turns_with_da_diff_vs_off"] == 2 and r["reported"]["fixed_turns_with_da_diff_vs_off"] == 0)
    # ── aggregates
    with tempfile.TemporaryDirectory() as td:
        for s in SEEDS:
            json.dump(dict(_syn_pp(), seed=s, family="pp"), open(os.path.join(td, "pp_seed%d.json" % s), "w"))
            json.dump(dict(_syn_sn(), seed=s, family="sn"), open(os.path.join(td, "sn_seed%d.json" % s), "w"))
        app, asn = aggregate(td, "pp"), aggregate(td, "sn")
        checks["aggregate pp: 6 designed seeds -> WD/NR/EP GO, sign-flip p = 1/64"] = \
            (app["WD_verdict"] == "GO" and app["NR_verdict"] == "GO" and app["EP_verdict"] == "GO"
             and abs(app["signflip_p_A_intact_minus_replaylesion"] - 1 / 64.0) < 1e-12)
        checks["aggregate sn: 6 designed seeds -> GO, p = 1/64"] = \
            asn["seed_verdict"] == "GO" and abs(asn["signflip_p_salient_intact_minus_wakelesion"] - 1 / 64.0) < 1e-12
        os.remove(os.path.join(td, "sn_seed102.json"))
        checks["aggregate sn: a missing seed -> INCOMPLETE"] = aggregate(td, "sn")["seed_verdict"] == "INCOMPLETE"
    # ── the production seams the families depend on (no brain)
    from webapp import da_tag_capture_chat as W
    prev = W.set_wall_clock(lambda: 7200.0)
    try:
        class _Cap:
            mode, t0_wall, t0_offset = "wall", 0.0, W._WORLD_OFFSET_H
        checks["clock seam: the wall clock reads the installed source"] = abs(W.ChatTagCapture.now_h(_Cap()) - 2.0) < 1e-12
    finally:
        W.set_wall_clock(prev)
    checks["clock seam: default source is time.time"] = W._WALL_CLOCK is None and abs(W._wall_now() - __import__("time").time()) < 5.0
    saved = {k: os.environ.get(k) for k in ("BRAIN_DA_ENCODING_LESION", "BRAIN_DA_ENCODING_LESION_SPARE_SWR",
                                            "BRAIN_DA_CAPTURE_LESION")}
    try:
        for k in saved:
            os.environ.pop(k, None)
        a0 = S.swr_prp_da(0.9)
        os.environ["BRAIN_DA_ENCODING_LESION"] = "1"
        a1 = S.swr_prp_da(0.9)
        os.environ["BRAIN_DA_ENCODING_LESION_SPARE_SWR"] = "1"
        a2 = S.swr_prp_da(0.9)
        os.environ["BRAIN_DA_CAPTURE_LESION"] = "1"
        a3 = S.swr_prp_da(0.9)
        checks["waking-only lesion: intact 0.9, full lesion tonic, spare-SWR 0.9, capture lesion still tonic"] = \
            (a0 == 0.9 and a1 == T._DA_TONIC and a2 == 0.9 and a3 == T._DA_TONIC)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    ok = all(checks.values())
    print("VERDICT:", "PASS" if ok else "FAIL", "(%d checks)" % len(checks))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--only", default=None, help="comma list of arm names: run only these, ungraded (a de-risk subset)")
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--design-fake", default=None, help="write the fake-substrate design day (intact + replay lesion)")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest() else 1
    if a.design_fake:
        out = {"runner": "research/runners/_pair_production_path_probe.py", "argv": list(sys.argv),
               "not_a_gate_row": "fake substrate (the tests' fake composer + linear D1); no brain",
               "intact": design_fake(None, False), "replay_lesion": design_fake(None, True)}
        os.makedirs(os.path.dirname(a.design_fake) or ".", exist_ok=True)
        json.dump(out, open(a.design_fake, "w"), indent=1)
        print("wrote", a.design_fake)
        return 0
    if a.family is None:
        ap.error("--family required")
    if a.aggregate:
        aggregate(a.aggregate, a.family)
        return 0
    if a.seed is None:
        ap.error("--seed required")
    run_seed(a.seed, a.out, a.family, workers=a.workers,
             only=([x.strip() for x in a.only.split(",") if x.strip()] if a.only else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
