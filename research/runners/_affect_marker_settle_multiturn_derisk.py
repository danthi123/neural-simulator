"""AFFECT-MARKER SETTLE — MULTI-TURN full-brain contrast (research/settle-multiturn-contrast, 2026-09-24).

WHY THIS EXISTS (the gap the single-turn gate left). `_affect_marker_settle_derisk.py --score-fullbrain`
(2026-09-23) established that `BRAIN_AFFECT_MARKER_SETTLE` makes the affect-marker WTA load-bearing on
6/6 seeds on a SINGLE isolated 'emo' turn, each seed its own fresh session. Re-reading that gate today found
it tests the WRONG hypothesis for a DEFAULT FLIP: production sessions carry MULTIPLE turns, the affect-marker
WTA's own reader is process-warm and CACHED ACROSS TURNS (`_affect_marker_wta_derisk._READERS`, keyed by
seed), and with SETTLE off every seed has a NARROW no-commit band near register boundaries that a REPEATED
read (the previous read's slow state, only `WASHOUT_STEPS`=40 ms of relaxation) can fall into even when the
FIRST read of a session committed cleanly. A same-day investigation (this arc) found: with SETTLE off, a
warm reader re-read at a held boundary mood emits the marker on 17/30 trials vs 30/30 with SETTLE on; the
single-turn gate cannot see this because every one of its 6 seeds is its OWN fresh session (one read each).

THE PROBE. A SINGLE session ("mt") of 5 turns run through the REAL production path
(`webapp.server.brain_chat`, numpy backend, stub renderer, LLM disabled — exactly `onebrain_regression_
battery._collect_worker`'s worker discipline, reused, not reinvented):
  1. mt_neutral  "the wolf bites the apple"                                        -- mood-neutral (n_hits=0);
                 NEVER touches the WTA (the neutral gate at `expression_lead` level==0 returns '' before any
                 spiking circuit is invoked) -- establishes the session/cache, nothing else.
  2. mt_emo1     EMO_TEXT (verbatim from `_lbf_borderline_operating_point.EMO_TEXT` / the battery's 'emo'
                 turn) -- the FIRST affective read: this is what builds+warms the WTA reader for the process.
  3. mt_emo2     a DIFFERENT text, the SAME (strongly positive) register (`appraise_text` valence 0.787 vs
                 emo1's 0.787, `n_hits`=4) -- immediately consecutive, so this is exactly the "read #2 on a
                 40ms-washed-out warm reader" case the mechanism probe measured degrading.
  4. mt_neg1     a strongly NEGATIVE text (a different register, valence -0.727) -- a register switch.
  5. mt_emo3     EMO_TEXT again (identical to turn 2) -- a repeated read of the SAME operating point AFTER an
                 intervening register switch, i.e. does the marker recover to its own turn-2 answer.
Every non-turn-1 row shares the ONE session (`reset=False`), so `AffectDrivesWorkspace`
(`webapp.affect_drives_chat.get_workspace`) and the `AffectMarkerWTA` reader persist across turns exactly as
in a real multi-turn conversation. `mt_neutral` resets first so the session starts clean.

PER SEED (BRAIN_CHAT_SEED = 42,43,44,100,101,102), SIX arms — each ONE fresh subprocess (`_spawn_arm`,
identical mechanism to the battery's), so noise trajectories are shared within an arm and independent
between arms exactly as the battery/lbf model requires:
  off_a, off_b    SETTLE=0, lesion=0  (off_a vs off_b: determinism)
  off_lesion      SETTLE=0, lesion=1  (BRAIN_AFFECT_MARKER_SPIKING_LESION=1 -- the SAME dedicated neural-cut
                                       knob `load_bearing_fraction.FACULTY_LESIONS["affect-marker-spiking-wta"]`
                                       already uses)
  on_a, on_b      SETTLE=1, lesion=0
  on_lesion       SETTLE=1, lesion=1
OFF-ARM DISCIPLINE (2026-08-27 class): every arm sets BOTH flags EXPLICITLY ("0" or "1"), never a pop.

PRE-REGISTERED HYPOTHESIS for a DEFAULT FLIP (committed BEFORE any evaluation row of this runner exists).
On EVERY seed and EVERY turn:
  H1 SUPERSET   : wherever OFF emits a marker (lead != ''), ON also emits one (ON never REGRESSES a turn
                  OFF already got right).
  H2 REGISTER   : wherever BOTH emit, the register (the marker WORD, punctuation-independent -- 'Gladly!' and
                  'Gladly —' are the SAME register) is IDENTICAL.
  H3 LOAD-BEARING: wherever ON's intact arm emits a marker, ON's LESION arm is silent on that same turn (lead
                  == '') -- the marker is provably the spiking WTA's contribution, not a host fallback.
  H4 ISOLATION  : nothing else in a turn's record differs between OFF and ON beyond the lead itself (and the
                  answer text's lead PREFIX, which is a deterministic function of the lead) -- every other
                  response field (after excluding the SAME continuous `_NOISE_FIELDS`
                  `onebrain_regression_battery` already excludes: mood/felt_arousal/ema_*/body_*/level/... --
                  a background process legitimately advances between builds) is byte-equal.
  H5 DETERMINISM: off_a==off_b and on_a==on_b on every turn (ignoring the same noise fields) -- the harness
                  itself is deterministic, so any OFF-vs-ON difference is attributable to the flag.
REPORTED, NOT GATED (this is the DEFECT SETTLE targets, not a flip precondition on any ONE seed):
  D1 DEFECT REPRODUCED: OFF loses the marker (lead=='') on at least one turn, ACROSS THE WHOLE 6-seed x
                  5-turn matrix, where the SAME turn's ON arm emits one. A design that never reproduces the
                  known defect would make this whole probe moot (an all-pass H1..H5 with D1 False would mean
                  the probe never stressed the failure mode it exists to test) -- reported honestly either way.
GO iff H1..H5 hold on EVERY (seed, turn) AND every arm is a valid, error-free read on every seed (6/6 seeds
present). ANY H1-H4 violation on ANY single (seed,turn) -> NO-GO (an ON arm that changes anything else, or a
lesion that fails to silence, must fail this gate -- verified in the failing direction by `--selftest`,
BEFORE any real row exists). A missing/errored arm, or an H5 determinism failure, -> UNDEFINED (the harness
itself is not trustworthy on that seed, never silently "not load-bearing").

TERMINOLOGY (docs/TERMS.md). A GO here means: under the ADEQUATE multi-turn probe, `BRAIN_AFFECT_MARKER_
SETTLE` (opt-in, default-OFF) reproduces the DEFAULT-flip criterion this arc was commissioned to derive
evidence for. It is NOT itself an owner decision to flip the default -- that is reported separately, paired
with the GPU-latency de-risk (`_affect_marker_settle_gpu_timing.py`) and the congruence measurement
(`_affect_marker_settle_congruence.py`).

NAMED HOST SHORTCUTS on this path (unchanged, inherited from `_affect_marker_wta_derisk.py` /
`_affect_marker_settle_derisk.py`, not re-derived here): S1 the `_select` argsort+DEAD_MARGIN readout: S2 the
Gaussian population-code DRIVE built by a host formula from the mood float; S3 the deliberation/rest CLOCK
(the legitimate host role). This runner adds a fourth, purely instrumental one: S5 the register-equality
check (H2) strips the trailing '!'/'—' punctuation with a host string op -- reading the register, not
selecting it (the selection remains the spiking competition).

Run (worker; a single fresh brain, one seed, one flag combination — this is what `_spawn_arm` launches):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -u -m \\
    research.runners._affect_marker_settle_multiturn_derisk --worker \\
    --env '{"BRAIN_AFFECT_MARKER_SETTLE": "1", "BRAIN_AFFECT_MARKER_SPIKING_LESION": "0"}' \\
    --out research/findings/raw/_affect_marker_settle_multiturn/s42/on_a.json

Run (one seed, all 6 arms, scored):
  SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 tools/memcap.sh 10 -- .venv/bin/python -u -m \\
    research.runners._affect_marker_settle_multiturn_derisk --run --seeds 42 \\
    --out-dir research/findings/raw/_affect_marker_settle_multiturn

Score already-collected raw arm files (e.g. harvested from the pool):
  .venv/bin/python -m research.runners._affect_marker_settle_multiturn_derisk --score \\
    --raw-dir research/findings/raw/_affect_marker_settle_multiturn --seeds "42 43 44 100 101 102" \\
    --out research/findings/raw/_affect_marker_settle_multiturn/verdict.json

Self-test (no brain build; drives the PURE scorer through every failing direction):
  .venv/bin/python -m research.runners._affect_marker_settle_multiturn_derisk --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

VERIFY_SEEDS = (42, 43, 44, 100, 101, 102)

EMO_TEXT = "Wonderful! I am so happy and delighted, this is fantastic and amazing!"
EMO2_TEXT = "This is absolutely wonderful, I am thrilled and overjoyed, what fantastic news!"
NEG_TEXT = "This is terrible, I am so sad and upset, everything is awful and horrible."
NEU_TEXT = "the wolf bites the apple"

# (label, message, session, reset, kind). kind is this runner's OWN bookkeeping (not read by brain_chat).
MT_TURNS = [
    ("mt_neutral", NEU_TEXT, "mt", True, "neutral"),
    ("mt_emo1", EMO_TEXT, "mt", False, "emo"),
    ("mt_emo2", EMO2_TEXT, "mt", False, "emo_repeat"),
    ("mt_neg1", NEG_TEXT, "mt", False, "negative"),
    ("mt_emo3", EMO_TEXT, "mt", False, "emo_repeat"),
]
MT_LABELS = [t[0] for t in MT_TURNS]
MT_KIND = {t[0]: t[4] for t in MT_TURNS}

# the SAME continuous fields `onebrain_regression_battery` already excludes from decision-equality (reuse by
# literal copy so a change there is a conscious decision here too -- this module has no import-time dependency
# on that file beyond this list, so it never builds a brain just to read a constant).
NOISE_FIELDS = {
    "rate_perceived", "rate_generated", "neg_rate", "pos_rate", "vminus_rate", "vplus_rate", "mood",
    "differential", "appraisal_valence", "appraisal_arousal", "felt_arousal", "ema_arousal", "ema_valence",
    "ema_engagement", "da_level", "snc_firing", "afferent_pA", "turn_engagement", "g", "d",
    "n_facts_scanned", "wm_margin", "gen_seconds", "body_a", "body_h", "confidence", "tone_level", "level",
    "appraisal_hits", "seed", "turn",
}

SETTLE_ENV = "BRAIN_AFFECT_MARKER_SETTLE"
LESION_ENV = "BRAIN_AFFECT_MARKER_SPIKING_LESION"


# ───────────────────────────────────────────── worker (subprocess) ─────────────────────────────────────────────
def _worker(env_json: str, out_path: str) -> int:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v            # OFF-ARM DISCIPLINE: explicit value, never a pop
    from webapp.server import brain_chat, BrainChatRequest
    responses = {}
    for label, msg, session, reset, kind in MT_TURNS:
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                            renderer="stub", rich=False, reset=reset))
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
    rec = {"env": env, "seed": os.environ.get("BRAIN_CHAT_SEED"), "turns": responses}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("[multiturn worker] env=%s seed=%s -> %d turns -> %s"
          % (env, rec["seed"], len(responses), out_path), flush=True)
    return 0


def _spawn_arm(seed: int, settle: bool, lesion: bool, out_path: str):
    env = {SETTLE_ENV: "1" if settle else "0", LESION_ENV: "1" if lesion else "0"}
    host_env = dict(os.environ)
    host_env["BRAIN_CHAT_SEED"] = str(int(seed))
    host_env.setdefault("SIM_BACKEND", "numpy")
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners._affect_marker_settle_multiturn_derisk",
                        "--worker", "--env", json.dumps(env), "--out", out_path], env=host_env)
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)


ARM_SPECS = {   # arm name -> (settle, lesion)
    "off_a": (False, False), "off_b": (False, False), "off_lesion": (False, True),
    "on_a": (True, False), "on_b": (True, False), "on_lesion": (True, True),
}


def run_seed(seed: int, out_dir: str) -> dict:
    sdir = os.path.join(out_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    arms = {}
    for name, (settle, lesion) in ARM_SPECS.items():
        arms[name] = _spawn_arm(seed, settle, lesion, os.path.join(sdir, "%s.json" % name))
        got = "OK" if arms[name] is not None else "FAILED"
        print("  s%d %-10s settle=%d lesion=%d -> %s" % (seed, name, settle, lesion, got), flush=True)
    return arms


# ───────────────────────────────────────────────── scoring (pure) ──────────────────────────────────────────────
def _get(resp, path):
    cur = resp
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def _lead(resp) -> str:
    return str(_get(resp, "affect_drives.lead") or "") if isinstance(resp, dict) else ""


def register_of(lead: str) -> str:
    """The marker WORD, punctuation-independent ('Gladly! ' and 'Gladly — ' are the SAME register)."""
    s = (lead or "").strip()
    if not s:
        return ""
    return s.rstrip("!—-").strip()


def _diff_ignoring(a, b, ignore_leaf: set, path: str = "") -> list:
    """Recursive dict diff; a path whose LAST segment is in `ignore_leaf` is never reported. Returns a list of
    (path, a_value, b_value) triples for every remaining mismatch (both dicts, elementwise)."""
    out = []
    if not isinstance(a, dict) or not isinstance(b, dict):
        if a != b:
            out.append((path, a, b))
        return out
    for k in sorted(set(a) | set(b)):
        if k in ignore_leaf:
            continue
        p = "%s.%s" % (path, k) if path else k
        av, bv = a.get(k, "<absent>"), b.get(k, "<absent>")
        if isinstance(av, dict) and isinstance(bv, dict):
            out.extend(_diff_ignoring(av, bv, ignore_leaf, p))
        elif av != bv:
            out.append((p, av, bv))
    return out


def _strip_answer_lead(resp: dict) -> dict:
    """A copy of `resp` with the affect-drives lead's own PREFIX removed from `answer` (H4 declares the lead
    prefix a LEGITIMATE difference: `webapp/server.py` does `resp['answer'] = affect_drives_lead + resp['answer']`
    verbatim) and `affect_drives.lead` itself blanked (compared separately, by H1/H2)."""
    if not isinstance(resp, dict):
        return resp
    out = dict(resp)
    lead = _lead(resp)
    answer = str(out.get("answer", "") or "")
    out["answer"] = answer[len(lead):] if lead and answer.startswith(lead) else answer
    ad = dict(out.get("affect_drives") or {})
    ad.pop("lead", None)
    out["affect_drives"] = ad
    return out


def compare_turn(off_resp, on_resp) -> dict:
    """Everything the scorer needs about ONE turn's OFF-vs-ON pair."""
    off_err, on_err = ("_error" in (off_resp or {})), ("_error" in (on_resp or {}))
    off_lead, on_lead = _lead(off_resp), _lead(on_resp)
    diffs = [] if (off_err or on_err) else _diff_ignoring(_strip_answer_lead(off_resp), _strip_answer_lead(on_resp),
                                                          NOISE_FIELDS)
    return {
        "off_lead": off_lead, "on_lead": on_lead,
        "off_register": register_of(off_lead), "on_register": register_of(on_lead),
        "off_level": _get(off_resp, "affect_drives.level"), "on_level": _get(on_resp, "affect_drives.level"),
        "off_error": off_err, "on_error": on_err,
        "superset_ok": bool(off_lead == "" or on_lead != ""),
        "register_ok": bool(off_lead == "" or on_lead == "" or register_of(off_lead) == register_of(on_lead)),
        "other_diffs": diffs, "isolation_ok": bool(not diffs) and not (off_err or on_err),
    }


def compare_lesion(on_resp, on_lesion_resp) -> dict:
    on_lead = _lead(on_resp)
    les_lead = _lead(on_lesion_resp)
    on_err = "_error" in (on_resp or {})
    les_err = "_error" in (on_lesion_resp or {})
    return {"on_lead": on_lead, "on_lesion_lead": les_lead,
            "load_bearing_ok": bool(on_err or les_err or on_lead == "" or les_lead == "")}


def compare_determinism(a_resp, b_resp) -> dict:
    diffs = [] if ("_error" in (a_resp or {}) or "_error" in (b_resp or {})) else \
        _diff_ignoring(a_resp, b_resp, NOISE_FIELDS)
    ok = (not diffs) and "_error" not in (a_resp or {}) and "_error" not in (b_resp or {})
    return {"deterministic": bool(ok), "diffs": diffs}


def score_seed(seed: int, arms: dict) -> dict:
    """`arms` = {arm_name: worker record-or-None}. Returns the per-seed scored record (pure)."""
    missing = [a for a in ARM_SPECS if arms.get(a) is None]
    valid = not missing
    per_turn = {}
    if valid:
        for label in MT_LABELS:
            off_t = arms["off_a"]["turns"].get(label)
            on_t = arms["on_a"]["turns"].get(label)
            les_t = arms["on_lesion"]["turns"].get(label)
            off_b_t = arms["off_b"]["turns"].get(label)
            on_b_t = arms["on_b"]["turns"].get(label)
            row = compare_turn(off_t, on_t)
            row.update(compare_lesion(on_t, les_t))
            row["off_determinism"] = compare_determinism(off_t, off_b_t)
            row["on_determinism"] = compare_determinism(on_t, on_b_t)
            row["kind"] = MT_KIND[label]
            row["defect_instance"] = bool(row["off_lead"] == "" and row["on_lead"] != "")
            per_turn[label] = row
    return {"seed": seed, "valid": valid, "missing_arms": missing, "per_turn": per_turn}


def score_all(per_seed_arms: dict, seeds=VERIFY_SEEDS) -> dict:
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    scored = {s: score_seed(s, per_seed_arms.get(s, {})) for s in seeds}
    n = len(seeds)
    n_valid = sum(1 for s in seeds if scored[s]["valid"])
    bad = []
    h1 = h2 = h3 = h4 = h5 = True
    defect_instances = []
    # ATTRIBUTION (tools.lab discipline, gap#5): H3 measures BOTH the intact-ON marker count and the count that
    # SURVIVES the lesion -- ask whose the marker count is, do not just report both numbers one key apart.
    # treatment = how many (seed,turn) cells ON's intact arm names a marker; control = how many of THOSE SAME
    # cells the lesion arm ALSO still names one (should be ~0 if the marker is the spiking circuit's own).
    n_on_emits = sum(1 for s in seeds if scored[s]["valid"] for row in scored[s]["per_turn"].values()
                     if row["on_lead"] != "")
    n_on_emits_survive_lesion = sum(1 for s in seeds if scored[s]["valid"] for row in scored[s]["per_turn"].values()
                                    if row["on_lead"] != "" and row["on_lesion_lead"] != "")
    lesion_attribution = attributable_to("affect-marker multiturn: ON marker count vs surviving-under-lesion count",
                                         float(n_on_emits), float(n_on_emits_survive_lesion))
    for s in seeds:
        rec = scored[s]
        if not rec["valid"]:
            bad.append("s%d missing arms: %s" % (s, rec["missing_arms"]))
            h1 = h2 = h3 = h4 = h5 = False
            continue
        for label, row in rec["per_turn"].items():
            if row["off_error"] or row["on_error"]:
                bad.append("s%d %s: arm error (off_error=%s on_error=%s)" % (s, label, row["off_error"], row["on_error"]))
                h1 = h2 = h3 = h4 = False
            if not row["superset_ok"]:
                h1 = False
                bad.append("s%d %s: H1 SUPERSET violated (off=%r on=%r)" % (s, label, row["off_lead"], row["on_lead"]))
            if not row["register_ok"]:
                h2 = False
                bad.append("s%d %s: H2 REGISTER mismatch (off=%r on=%r)" % (s, label, row["off_register"], row["on_register"]))
            if not row["load_bearing_ok"]:
                h3 = False
                bad.append("s%d %s: H3 LESION did not silence ON (on=%r lesion=%r)" % (s, label, row["on_lead"], row["on_lesion_lead"]))
            if not row["isolation_ok"]:
                h4 = False
                bad.append("s%d %s: H4 ISOLATION violated, extra diffs=%s" % (s, label, row["other_diffs"]))
            if not (row["off_determinism"]["deterministic"] and row["on_determinism"]["deterministic"]):
                h5 = False
                bad.append("s%d %s: H5 DETERMINISM failed off=%s on=%s"
                           % (s, label, row["off_determinism"]["diffs"], row["on_determinism"]["diffs"]))
            if row["defect_instance"]:
                defect_instances.append("s%d/%s(%s)" % (s, label, row["kind"]))
    vd = Verdict("affect_marker_settle_multiturn_contrast")
    vd.require("full 6-seed set, every arm present", n_valid, expect=lambda x: x == n, note="; ".join(bad[:5]))
    vd.require("H5 determinism holds on every seed/turn", h5, expect=True)
    go = bool(h1 and h2 and h3 and h4 and n_valid == n)
    decided = vd.decide(go)
    return {"probe": "affect_marker_settle_multiturn_contrast", "seeds": list(seeds),
            "go": bool(decided["go"]), "status": decided["status"],
            "H1_superset": h1, "H2_register": h2, "H3_load_bearing": h3, "H4_isolation": h4, "H5_determinism": h5,
            "D1_defect_reproduced": bool(defect_instances), "defect_instances": defect_instances,
            "lesion_attribution": {"n_on_emits": n_on_emits, "n_on_emits_survive_lesion": n_on_emits_survive_lesion,
                                   "fraction_attributable_to_lesion": lesion_attribution},
            "failures": bad, "verdict": decided, "preconditions": decided["preconditions"],
            "per_seed": {str(s): v for s, v in scored.items()},
            "rule": "GO iff H1 (ON marker-superset over OFF), H2 (register identical wherever both emit), "
                    "H3 (ON's lesion silences ON's marker), H4 (no other field differs beyond the lead/answer-"
                    "prefix) hold on EVERY seed and turn, and every arm on every seed is present + error-free. "
                    "D1 (OFF loses the marker on >=1 turn ON keeps) is REPORTED, never gated -- it is the "
                    "defect this probe exists to reproduce, not a flip precondition."}


# ─────────────────────────────────────────────────── selftest ──────────────────────────────────────────────────
def _resp(lead="", level=2, answer="hi", extra=None, error=None):
    if error:
        return {"_error": error}
    r = {"answer": (lead + answer) if lead else answer,
         "affect_drives": {"lead": lead, "level": level, "mood": 0.07, "acted": True}, "abstained": False}
    if extra:
        r.update(extra)
    return r


def _arms(off_leads, on_leads, on_lesion_leads=None, extra_on=None, off_b_leads=None, on_b_leads=None) -> dict:
    """Build a synthetic arms dict for ONE seed from per-turn lead lists (len == len(MT_LABELS))."""
    on_lesion_leads = on_lesion_leads or [""] * len(MT_LABELS)
    off_b_leads = off_b_leads or off_leads
    on_b_leads = on_b_leads or on_leads
    turns_off = {lab: _resp(off_leads[i]) for i, lab in enumerate(MT_LABELS)}
    turns_off_b = {lab: _resp(off_b_leads[i]) for i, lab in enumerate(MT_LABELS)}
    turns_on = {lab: _resp(on_leads[i], extra=(extra_on[i] if extra_on else None)) for i, lab in enumerate(MT_LABELS)}
    # NOTE: on_b carries the SAME `extra` as on_a (when on_b_leads was not explicitly overridden) so an
    # H4-only synthetic case (an unrelated field that differs OFF-vs-ON) does not ALSO trip H5 determinism as
    # a side effect -- H5 is exercised deliberately only via `off_b_leads`/`on_b_leads` overrides (case (f)).
    turns_on_b = {lab: _resp(on_b_leads[i], extra=(extra_on[i] if (extra_on and on_b_leads is on_leads) else None))
                  for i, lab in enumerate(MT_LABELS)}
    turns_lesion = {lab: _resp(on_lesion_leads[i]) for i, lab in enumerate(MT_LABELS)}
    return {"off_a": {"turns": turns_off}, "off_b": {"turns": turns_off_b}, "off_lesion": {"turns": turns_lesion},
            "on_a": {"turns": turns_on}, "on_b": {"turns": turns_on_b}, "on_lesion": {"turns": turns_lesion}}


def _selftest_scorer() -> bool:
    ok = True
    n = len(MT_LABELS)
    # (a) clean case: OFF misses turn 3 ('mt_emo2'), ON gets everything, lesion silences -> GO, D1 True
    off = ["", "Gladly! ", "", "Frankly — ", "Gladly! "]
    on = ["", "Gladly! ", "Gladly! ", "Frankly — ", "Gladly! "]
    r = score_all({42: _arms(off, on)}, seeds=(42,))
    ok = ok and r["go"] and r["D1_defect_reproduced"]
    print("  clean+defect -> GO=%s D1=%s (want True True) %s" % (r["go"], r["D1_defect_reproduced"], "ok" if r["go"] and r["D1_defect_reproduced"] else "FAIL"))
    # (b) H1 violated: ON drops a marker OFF had -> must FAIL
    off2 = ["", "Gladly! ", "Gladly! ", "Frankly — ", "Gladly! "]
    on2 = ["", "", "Gladly! ", "Frankly — ", "Gladly! "]
    r = score_all({42: _arms(off2, on2)}, seeds=(42,))
    got = (not r["go"]) and (not r["H1_superset"])
    ok = ok and got
    print("  H1 superset violated -> go=%s H1=%s (want False False) %s" % (r["go"], r["H1_superset"], "ok" if got else "FAIL"))
    # (c) H2 violated: both emit, different register -> must FAIL
    on3 = ["", "Wonderful! ", "Gladly! ", "Frankly — ", "Gladly! "]
    r = score_all({42: _arms(off, on3)}, seeds=(42,))
    got = (not r["go"]) and (not r["H2_register"])
    ok = ok and got
    print("  H2 register mismatch -> go=%s H2=%s (want False False) %s" % (r["go"], r["H2_register"], "ok" if got else "FAIL"))
    # (d) H3 violated: ON emits, lesion does NOT silence -> must FAIL
    lesion_bad = ["", "Gladly! ", "", "", ""]
    r = score_all({42: _arms(off, on, on_lesion_leads=lesion_bad)}, seeds=(42,))
    got = (not r["go"]) and (not r["H3_load_bearing"])
    ok = ok and got
    print("  H3 lesion fails to silence -> go=%s H3=%s (want False False) %s" % (r["go"], r["H3_load_bearing"], "ok" if got else "FAIL"))
    # (e) H4 violated: an ON arm changes an unrelated field (abstained flips) -> must FAIL
    extra_on = [None] * n
    extra_on[1] = {"abstained": True}
    r = score_all({42: _arms(off, on, extra_on=extra_on)}, seeds=(42,))
    got = (not r["go"]) and (not r["H4_isolation"])
    ok = ok and got
    print("  H4 isolation violated (unrelated field changed) -> go=%s H4=%s (want False False) %s" % (r["go"], r["H4_isolation"], "ok" if got else "FAIL"))
    # (f) H5 violated: off_a vs off_b disagree (harness non-deterministic) -> must FAIL (and never silently GO)
    off_b_bad = ["", "Gladly! ", "Frankly — ", "Frankly — ", "Gladly! "]
    r = score_all({42: _arms(off, on, off_b_leads=off_b_bad)}, seeds=(42,))
    got = (not r["go"]) and (not r["H5_determinism"])
    ok = ok and got
    print("  H5 determinism failed -> go=%s H5=%s (want False False) %s" % (r["go"], r["H5_determinism"], "ok" if got else "FAIL"))
    # (g) missing arm -> UNDEFINED, never a silent pass
    arms_missing = _arms(off, on)
    arms_missing["on_lesion"] = None
    r = score_all({42: arms_missing}, seeds=(42,))
    got = (not r["go"]) and r["status"] == "UNDEFINED"
    ok = ok and got
    print("  missing arm -> go=%s status=%s (want False UNDEFINED) %s" % (r["go"], r["status"], "ok" if got else "FAIL"))
    # (h) D1 never reproduced (OFF never loses a marker ON keeps) -> still a GO if H1-H5 hold (REPORTED not GATED)
    off_full = ["", "Gladly! ", "Gladly! ", "Frankly — ", "Gladly! "]  # OFF matches ON everywhere
    r = score_all({42: _arms(off_full, on)}, seeds=(42,))
    got = r["go"] and not r["D1_defect_reproduced"]
    ok = ok and got
    print("  D1 not reproduced but H1-H5 hold -> go=%s D1=%s (want True False, D1 is advisory) %s"
          % (r["go"], r["D1_defect_reproduced"], "ok" if got else "FAIL"))
    # (i) register_of punctuation independence
    ok_reg = (register_of("Gladly! ") == register_of("Gladly — ") == "Gladly") and register_of("") == ""
    ok = ok and ok_reg
    print("  register_of punctuation-independent -> %s" % ("ok" if ok_reg else "FAIL"))
    return ok


def selftest() -> bool:
    ok = _selftest_scorer()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", default=" ".join(str(s) for s in VERIFY_SEEDS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="research/findings/raw/_affect_marker_settle_multiturn")
    ap.add_argument("--raw-dir", default="research/findings/raw/_affect_marker_settle_multiturn")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.worker:
        sys.exit(_worker(a.env, a.out or "research/findings/raw/_affect_marker_settle_multiturn/worker.json"))
    seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
    if a.run:
        per_seed_arms = {s: run_seed(s, a.out_dir) for s in seeds}
        rec = score_all(per_seed_arms, seeds)
        out = a.out or os.path.join(a.out_dir, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("STATUS=%s GO=%s H1=%s H2=%s H3=%s H4=%s H5=%s D1=%s -> %s"
              % (rec["status"], rec["go"], rec["H1_superset"], rec["H2_register"], rec["H3_load_bearing"],
                 rec["H4_isolation"], rec["H5_determinism"], rec["D1_defect_reproduced"], out))
        return
    if a.score:
        per_seed_arms = {}
        for s in seeds:
            sdir = os.path.join(a.raw_dir, "s%d" % s)
            arms = {}
            for name in ARM_SPECS:
                p = os.path.join(sdir, "%s.json" % name)
                arms[name] = json.load(open(p)) if os.path.exists(p) else None
            per_seed_arms[s] = arms
        rec = score_all(per_seed_arms, seeds)
        out = a.out or os.path.join(a.raw_dir, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("STATUS=%s GO=%s H1=%s H2=%s H3=%s H4=%s H5=%s D1=%s -> %s"
              % (rec["status"], rec["go"], rec["H1_superset"], rec["H2_register"], rec["H3_load_bearing"],
                 rec["H4_isolation"], rec["H5_determinism"], rec["D1_defect_reproduced"], out))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
