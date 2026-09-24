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

AMENDMENT A1 (2026-09-24, fix round after an independent review; committed BEFORE the amended scorer is run on
any seed; the governing document is research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-
PREREG.md). The worker (`_worker`, `_spawn_arm`, `ARM_SPECS`, `run_seed`) is UNCHANGED from the pinned revision
49a089d8d that produced every arm; only the scorer changes. What changed and why:
  A1.1 H4 reads the lead WHEREVER it sits in the reply. A later stage prepends a clause ("That's absolutely
       thrilling for This -- Wonderful! I don't know..."), so the old prefix-only strip scored a mid-reply marker
       rescue as an H4 isolation failure. H4 now holds iff removing each arm's own lead at one of its verbatim
       occurrences leaves the two replies equal. A lead the reply does not contain fails H4.
  A1.2 VALIDITY is checked, not assumed. Every arm file of every seed must name the right seed and the right
       SETTLE/LESION env, carry every MT turn, and every turn must be a real reply: no `_error`, an `answer`
       string, an `affect_drives` record with acted=True and a `reason` that is not `error:*`. Any failure makes
       that seed invalid and the verdict UNDEFINED. An errored lesion turn no longer counts as "silenced", and
       `off_lesion` is validated and scored (H3-OFF is REPORTED, not gated).
  A1.3 LEVER: SETTLE must be shown to reach the reader. `expression_lead` swallows reader exceptions and returns
       '', so a crashed WTA and a real cut look the same in a reply. The scorer therefore REPLAYS every arm's own
       recorded (mood, felt_arousal) sequence through the production `expression_lead` in this process, with a
       fresh reader built from the arm's own env, recording every reader exception. Precondition (UNDEFINED
       if unmet on any seed): the env builds the expected reader (SETTLE: 500/1000 ms, OFF: 60/40 ms), the
       replay raises nothing, and it reproduces every recorded lead of every arm. A counterfactual replay
       (the other SETTLE setting, and the lesion arms with the cut undone) is REPORTED: it counts the turns on
       which the SETTLE setting, or the lesion, alone decides the lead.
  A1.4 NON-VACUOUS: ON must differ from OFF on at least one (seed, turn), else UNDEFINED. A SETTLE that changes
       nothing no longer reads GO (the old selftest case (h) asserted that it did).
  A1.5 D1 is split and still REPORTED, never gated: D1_first_read (mt_emo1, the single-turn defect the
       2026-09-23 gate already showed) and D1_warm_read (mt_emo2, mt_neg1, mt_emo3 -- the warm-reader repeat-read
       defect this probe was built to reproduce). A GO with D1_warm_read False certifies that the flip is SAFE on
       multi-turn sessions, not that it fixes a multi-turn defect; `interpretation` says so in the verdict.
  A1.6 Rates are reported per AFFECTIVE turn (4 per seed) as well as per turn (5 per seed).
GO/NO-GO: GO iff every precondition holds (6/6 valid seeds, H5, A1.3 lever, A1.4 non-vacuous) AND H1-H4 hold on
every (seed, turn). NO-GO iff every precondition holds and any H1-H4 cell fails. Otherwise UNDEFINED.
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
AFFECTIVE_LABELS = tuple(t[0] for t in MT_TURNS if t[4] != "neutral")          # 4 per seed
FIRST_READ_LABELS = ("mt_emo1",)                                                 # the reader's first read
WARM_READ_LABELS = tuple(lab for lab in AFFECTIVE_LABELS if lab not in FIRST_READ_LABELS)   # repeat reads


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


def _lead_removals(answer: str, lead: str) -> set:
    """Every string `answer` becomes when ONE verbatim occurrence of `lead` is removed, wherever it sits (A1.1).
    No lead -> {answer}. A non-empty lead the answer does not contain -> the EMPTY set (the lead field names a
    marker the user never sees, which no OFF/ON pairing can explain)."""
    if not lead:
        return {answer}
    outs, i = set(), answer.find(lead)
    while i >= 0:
        outs.add(answer[:i] + answer[i + len(lead):])
        i = answer.find(lead, i + 1)
    return outs


def answers_match_modulo_leads(off_ans: str, off_lead: str, on_ans: str, on_lead: str) -> bool:
    """True iff removing each arm's OWN lead at one of its occurrences leaves the two replies byte-equal (A1.1)."""
    return bool(_lead_removals(off_ans, off_lead) & _lead_removals(on_ans, on_lead))


def _without_lead(resp: dict, answer_placeholder=None) -> dict:
    """A copy of `resp` with `affect_drives.lead` removed (compared separately by H1/H2) and, when a placeholder is
    given, `answer` replaced by it (the answers were already compared modulo their leads)."""
    out = dict(resp)
    if answer_placeholder is not None:
        out["answer"] = answer_placeholder
    ad = dict(out.get("affect_drives") or {})
    ad.pop("lead", None)
    out["affect_drives"] = ad
    return out


def compare_turn(off_resp, on_resp) -> dict:
    """Everything the scorer needs about ONE turn's OFF-vs-ON pair. Only ever called on VALIDATED turns."""
    off_lead, on_lead = _lead(off_resp), _lead(on_resp)
    off_ans, on_ans = str(off_resp.get("answer", "")), str(on_resp.get("answer", ""))
    ans_ok = answers_match_modulo_leads(off_ans, off_lead, on_ans, on_lead)
    ph = "<answer equal modulo leads>"
    diffs = _diff_ignoring(_without_lead(off_resp, ph), _without_lead(on_resp, ph), NOISE_FIELDS)
    if not ans_ok:        # includes a lead the reply does not contain (then the raw answers may even be equal)
        diffs.insert(0, ("answer (modulo each arm's own lead)", off_ans, on_ans))
    return {
        "off_lead": off_lead, "on_lead": on_lead,
        "off_register": register_of(off_lead), "on_register": register_of(on_lead),
        "off_level": _get(off_resp, "affect_drives.level"), "on_level": _get(on_resp, "affect_drives.level"),
        "superset_ok": bool(off_lead == "" or on_lead != ""),
        "register_ok": bool(off_lead == "" or on_lead == "" or register_of(off_lead) == register_of(on_lead)),
        "answer_equal_modulo_leads": ans_ok,
        "other_diffs": diffs, "isolation_ok": bool(not diffs),
    }


def compare_lesion(intact_resp, lesion_resp) -> dict:
    """H3 on VALIDATED turns only (A1.2): an errored lesion turn never reaches here -- it invalidates the seed."""
    i_lead, l_lead = _lead(intact_resp), _lead(lesion_resp)
    return {"intact_lead": i_lead, "lesion_lead": l_lead, "load_bearing_ok": bool(i_lead == "" or l_lead == "")}


def compare_determinism(a_resp, b_resp) -> dict:
    diffs = _diff_ignoring(a_resp, b_resp, NOISE_FIELDS)
    return {"deterministic": not diffs, "diffs": diffs}


def expected_env(arm: str) -> dict:
    settle, lesion = ARM_SPECS[arm]
    return {SETTLE_ENV: "1" if settle else "0", LESION_ENV: "1" if lesion else "0"}


def validate_turn(resp) -> list:
    """Why this turn is not a real, error-free reply (empty list = valid). A1.2."""
    if not isinstance(resp, dict):
        return ["turn is %s, not a response dict" % type(resp).__name__]
    probs = []
    if "_error" in resp:
        probs.append("worker exception: %s" % str(resp["_error"])[:120])
    if not isinstance(resp.get("answer"), str):
        probs.append("no answer string")
    ad = resp.get("affect_drives")
    if not isinstance(ad, dict):
        probs.append("no affect_drives record")
    else:
        if ad.get("acted") is not True:
            probs.append("affect_drives.acted=%r" % ad.get("acted"))
        reason = ad.get("reason")
        if isinstance(reason, str) and reason.startswith("error"):
            probs.append("affect_drives.reason=%s" % reason[:120])
        if not isinstance(ad.get("lead", ""), str):
            probs.append("affect_drives.lead is not a string")
    return probs


def validate_arm(seed: int, arm: str, rec) -> list:
    """Why this arm file cannot be scored (empty list = valid). A1.2."""
    if rec is None:
        return ["%s: arm file missing" % arm]
    if not isinstance(rec, dict) or not isinstance(rec.get("turns"), dict):
        return ["%s: no turns record" % arm]
    probs = []
    if str(rec.get("seed")) != str(seed):
        probs.append("%s: records seed %r, expected %d" % (arm, rec.get("seed"), seed))
    if rec.get("env") != expected_env(arm):
        probs.append("%s: env %r != expected %r" % (arm, rec.get("env"), expected_env(arm)))
    for label in MT_LABELS:
        if label not in rec["turns"]:
            probs.append("%s/%s: turn missing" % (arm, label))
            continue
        for p in validate_turn(rec["turns"][label]):
            probs.append("%s/%s: %s" % (arm, label, p))
        ad_seed = _get(rec["turns"][label], "affect_drives.seed")
        if ad_seed is not None and str(ad_seed) != str(seed):
            probs.append("%s/%s: affect_drives.seed=%r, expected %d" % (arm, label, ad_seed, seed))
    return probs


def score_seed(seed: int, arms: dict) -> dict:
    """`arms` = {arm_name: worker record-or-None}. Returns the per-seed scored record (pure)."""
    problems = []
    for a in ARM_SPECS:
        problems.extend(validate_arm(seed, a, (arms or {}).get(a)))
    valid = not problems
    per_turn = {}
    if valid:
        T = {a: arms[a]["turns"] for a in ARM_SPECS}
        for label in MT_LABELS:
            row = compare_turn(T["off_a"][label], T["on_a"][label])
            les = compare_lesion(T["on_a"][label], T["on_lesion"][label])
            les_off = compare_lesion(T["off_a"][label], T["off_lesion"][label])
            row.update({"on_lesion_lead": les["lesion_lead"], "load_bearing_ok": les["load_bearing_ok"],
                        "off_lesion_lead": les_off["lesion_lead"], "off_load_bearing_ok": les_off["load_bearing_ok"]})
            row["off_determinism"] = compare_determinism(T["off_a"][label], T["off_b"][label])
            row["on_determinism"] = compare_determinism(T["on_a"][label], T["on_b"][label])
            row["kind"] = MT_KIND[label]
            row["affective"] = label in AFFECTIVE_LABELS
            row["read"] = ("first" if label in FIRST_READ_LABELS else "warm" if label in WARM_READ_LABELS
                           else "none")
            row["defect_instance"] = bool(row["off_lead"] == "" and row["on_lead"] != "")
            row["settle_changed_lead"] = bool(row["off_lead"] != row["on_lead"])
            per_turn[label] = row
    return {"seed": seed, "valid": valid, "problems": problems,
            "missing_arms": [a for a in ARM_SPECS if (arms or {}).get(a) is None], "per_turn": per_turn}


# ──────────────────────────────────────────── replay lever (impure, cheap) ─────────────────────────────────────
def replay_arm(seed: int, arm: str, rec: dict, *, settle=None, lesion=None) -> dict:
    """A1.3. Re-drive this arm's OWN recorded (level, high_arousal, mood, felt_arousal) sequence, in MT order,
    through the PRODUCTION `webapp.affect_drives_chat.expression_lead`, with a FRESH reader built from the arm's own
    env (or the `settle`/`lesion` override, for a counterfactual). The reader is the tiny private WTA bridge only
    (no brain build; ~0.2 s per read). Every reader exception is recorded here BEFORE `expression_lead` swallows
    it. Deterministic: the reader is seeded by cfg.seed and does not draw from the process-global RNG (checked on
    the dev seed before this amendment was written)."""
    import research.runners._affect_marker_wta_derisk as W
    import webapp.affect_drives_chat as ADC
    s_flag, l_flag = ARM_SPECS[arm]
    s_flag = s_flag if settle is None else bool(settle)
    l_flag = l_flag if lesion is None else bool(lesion)
    env = {SETTLE_ENV: "1" if s_flag else "0", LESION_ENV: "1" if l_flag else "0"}
    saved_env = {k: os.environ.get(k) for k in list(env) + ["BRAIN_AFFECT_MARKER_SPIKING",
                                                            "BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE"]}
    saved_get_reader = W.get_reader
    readers, calls, excs = {}, [], []

    def _instrument(r):
        for meth in ("select_valence", "select_arousal"):
            orig = getattr(r, meth)

            def wrapped(*a, _orig=orig, _meth=meth, **k):
                calls.append(_meth)
                try:
                    return _orig(*a, **k)
                except Exception as e:           # recorded, then re-raised into expression_lead's own handler
                    excs.append("%s: %s: %s" % (_meth, type(e).__name__, e))
                    raise
            setattr(r, meth, wrapped)
        return r

    def fresh_get_reader(seed=42):
        if int(seed) not in readers:
            readers[int(seed)] = _instrument(W.AffectMarkerWTA(seed=int(seed)))   # settle read from the env
        return readers[int(seed)]
    try:
        os.environ.update(env)
        os.environ.pop("BRAIN_AFFECT_MARKER_SPIKING", None)            # production default: the spiking selector
        os.environ.pop("BRAIN_AFFECT_MARKER_SPIKING_SHUFFLE", None)
        W.get_reader = fresh_get_reader
        turns = {}
        for label in MT_LABELS:
            ad = rec["turns"][label]["affect_drives"]
            n0, e0 = len(calls), len(excs)
            lead = ADC.expression_lead(int(ad.get("level", 0)), bool(ad.get("high_arousal")),
                                       mood=ad.get("mood"), felt_arousal=ad.get("felt_arousal"),
                                       seed=int(ad.get("seed", seed)))
            turns[label] = {"recorded": str(ad.get("lead", "") or ""), "replayed": lead,
                            "n_reader_calls": len(calls) - n0, "exceptions": excs[e0:]}
        cfg = [{"seed": k, "settle": r.settle, "warmup": r.warmup, "washout": r.washout} for k, r in readers.items()]
    finally:
        W.get_reader = saved_get_reader
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    exp = ({"settle": True, "warmup": W.DELIBERATION_MS, "washout": W.INTERTURN_REST_MS} if s_flag else
           {"settle": False, "warmup": W.WARMUP_STEPS, "washout": W.WASHOUT_STEPS})
    return {"arm": arm, "settle": s_flag, "lesion": l_flag, "expected_reader": exp, "readers": cfg, "turns": turns}


def replay_seed(seed: int, arms: dict) -> dict:
    """Factual replay of all six arms, plus the counterfactuals A1.3 reports (only on a seed whose arms validated)."""
    out = {"factual": {a: replay_arm(seed, a, arms[a]) for a in ARM_SPECS}}
    out["cf_settle_flipped"] = {a: replay_arm(seed, a, arms[a], settle=not ARM_SPECS[a][0])
                                for a in ("off_a", "on_a")}
    out["cf_lesion_undone"] = {a: replay_arm(seed, a, arms[a], lesion=False) for a in ("off_lesion", "on_lesion")}
    return out


def lever_from_replay(rep) -> dict:
    """Pure: did SETTLE reach the reader, and do the recorded leads come from the reader as configured (A1.3)."""
    if not isinstance(rep, dict) or "factual" not in rep:
        return {"measured": False, "ok": None, "problems": ["no replay record"]}
    probs = []
    for arm, fr in rep["factual"].items():
        exp = fr.get("expected_reader") or {}
        for r in fr.get("readers") or []:
            got = {k: r.get(k) for k in ("settle", "warmup", "washout")}
            if got != exp:
                probs.append("%s: reader built as %s, expected %s" % (arm, got, exp))
        for label, t in (fr.get("turns") or {}).items():
            if t.get("exceptions"):
                probs.append("%s/%s: reader raised %s" % (arm, label, t["exceptions"]))
            if t.get("replayed") != t.get("recorded"):
                probs.append("%s/%s: replay %r != recorded %r" % (arm, label, t.get("replayed"), t.get("recorded")))
        if any((t.get("n_reader_calls") or 0) > 0 for t in (fr.get("turns") or {}).values()) and not fr.get("readers"):
            probs.append("%s: reader calls recorded but no reader config" % arm)
    ident_settle = []
    for arm, cf in (rep.get("cf_settle_flipped") or {}).items():
        fact = rep["factual"].get(arm, {}).get("turns", {})
        for label, t in (cf.get("turns") or {}).items():
            if label in fact and t.get("replayed") != fact[label].get("replayed"):
                ident_settle.append("%s/%s" % (arm, label))
    ident_lesion = []
    for arm, cf in (rep.get("cf_lesion_undone") or {}).items():
        fact = rep["factual"].get(arm, {}).get("turns", {})
        for label, t in (cf.get("turns") or {}).items():
            if label in fact and fact[label].get("replayed") == "" and t.get("replayed") != "":
                ident_lesion.append("%s/%s" % (arm, label))
    return {"measured": True, "ok": not probs, "problems": probs,
            "settle_decides_lead_turns": ident_settle, "lesion_decides_lead_turns": ident_lesion}


def score_all(per_seed_arms: dict, seeds=VERIFY_SEEDS, replay=None) -> dict:
    """`replay` = {seed: replay_seed(...) record}; a seed without one reads UNDEFINED (A1.3)."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    replay = replay or {}
    scored = {s: score_seed(s, per_seed_arms.get(s, {})) for s in seeds}
    # "full 6-seed set" counts the pre-registered VERIFY_SEEDS that were scored AND valid (19c8ed596), with
    # validity now meaning every arm validated turn by turn (A1.2), not merely that its file exists.
    n = len(VERIFY_SEEDS)
    n_valid = sum(1 for s in VERIFY_SEEDS if s in scored and scored[s]["valid"])
    unscored = [s for s in VERIFY_SEEDS if s not in scored]
    bad = ["s%d not scored (not in --seeds)" % s for s in unscored]
    h1 = h2 = h3 = h4 = h5 = True
    h3_off = True
    d1_first, d1_warm, changed = [], [], []
    levers = {}
    for s in seeds:
        rec = scored[s]
        if not rec["valid"]:
            bad.append("s%d invalid: %s" % (s, "; ".join(rec["problems"][:4])))
            continue
        levers[s] = lever_from_replay(replay.get(s))
        for label, row in rec["per_turn"].items():
            if not row["superset_ok"]:
                h1 = False
                bad.append("s%d %s: H1 SUPERSET violated (off=%r on=%r)" % (s, label, row["off_lead"], row["on_lead"]))
            if not row["register_ok"]:
                h2 = False
                bad.append("s%d %s: H2 REGISTER mismatch (off=%r on=%r)" % (s, label, row["off_register"], row["on_register"]))
            if not row["load_bearing_ok"]:
                h3 = False
                bad.append("s%d %s: H3 LESION did not silence ON (on=%r lesion=%r)" % (s, label, row["on_lead"], row["on_lesion_lead"]))
            if not row["off_load_bearing_ok"]:
                h3_off = False
            if not row["isolation_ok"]:
                h4 = False
                bad.append("s%d %s: H4 ISOLATION violated, extra diffs=%s" % (s, label, row["other_diffs"][:3]))
            if not (row["off_determinism"]["deterministic"] and row["on_determinism"]["deterministic"]):
                h5 = False
                bad.append("s%d %s: H5 DETERMINISM failed off=%s on=%s"
                           % (s, label, row["off_determinism"]["diffs"][:2], row["on_determinism"]["diffs"][:2]))
            if row["defect_instance"]:
                (d1_first if row["read"] == "first" else d1_warm).append("s%d/%s(%s)" % (s, label, row["kind"]))
            if row["settle_changed_lead"]:
                changed.append("s%d/%s" % (s, label))
    valid_seeds = [s for s in seeds if scored[s]["valid"]]
    rows = [row for s in valid_seeds for row in scored[s]["per_turn"].values()]
    aff = [row for row in rows if row["affective"]]
    n_on_emits = sum(1 for row in rows if row["on_lead"] != "")
    n_on_emits_survive_lesion = sum(1 for row in rows if row["on_lead"] != "" and row["on_lesion_lead"] != "")
    lesion_attribution = attributable_to("affect-marker multiturn: ON marker count vs surviving-under-lesion count",
                                         float(n_on_emits), float(n_on_emits_survive_lesion))
    lever_ok_all = bool(valid_seeds) and all(levers[s]["ok"] is True for s in valid_seeds)
    lever_unmeasured = [s for s in valid_seeds if not levers[s]["measured"]]
    lever_notes = ["s%d: %s" % (s, "; ".join(levers[s]["problems"][:3])) for s in valid_seeds
                   if levers[s]["ok"] is not True]
    vd = Verdict("affect_marker_settle_multiturn_contrast")
    vd.require("full 6-seed set, every arm present and every turn a valid error-free reply (A1.2)", n_valid,
               expect=lambda x: x == n, note="; ".join(bad[:5]))
    vd.require("H5 determinism holds on every seed/turn", h5, expect=True)
    vd.require("SETTLE reached the reader: env builds the expected reader, replay raises nothing and reproduces "
               "every recorded lead (A1.3)", None if (lever_unmeasured or not valid_seeds) else lever_ok_all,
               expect=True, note="; ".join(lever_notes[:4]) or
               ("replay missing for %s" % lever_unmeasured if lever_unmeasured else ""))
    vd.require("non-vacuous: ON differs from OFF on >=1 (seed, turn) (A1.4)", len(changed), expect=lambda x: x >= 1)
    go = bool(h1 and h2 and h3 and h4 and n_valid == n)
    decided = vd.decide(go)
    if decided["status"] == "GO":
        interpretation = ("SAFE and BENEFICIAL on warm reads: H1-H4 hold everywhere and OFF lost a warm-read marker "
                          "that ON kept." if d1_warm else
                          "SAFE on multi-turn sessions (H1-H4 hold everywhere), but the warm-reader repeat-read "
                          "defect was NOT reproduced: the only OFF losses ON repairs are first reads (the single-turn "
                          "defect). This GO is not evidence of a multi-turn benefit.")
    else:
        interpretation = "status %s: see failures / preconditions" % decided["status"]
    return {"probe": "affect_marker_settle_multiturn_contrast", "amendment": "A1 (2026-09-24)", "seeds": list(seeds),
            "n_verify_seeds_valid": n_valid, "verify_seeds_unscored": unscored,
            "go": bool(decided["go"]), "status": decided["status"], "interpretation": interpretation,
            "H1_superset": h1, "H2_register": h2, "H3_load_bearing": h3, "H4_isolation": h4, "H5_determinism": h5,
            "H3_off_load_bearing_reported": h3_off,
            "D1_defect_reproduced": bool(d1_first or d1_warm),
            "D1_first_read": bool(d1_first), "D1_warm_read": bool(d1_warm),
            "defect_instances_first_read": d1_first, "defect_instances_warm_read": d1_warm,
            "benefit_on_warm_reads": "SHOWN" if d1_warm else "NOT_REPRODUCED",
            "settle_changed_lead_cells": changed,
            "lever": {str(s): v for s, v in levers.items()},
            "rates": {"n_turn_cells": len(rows), "n_affective_cells": len(aff),
                      "off_markers_per_affective_turn": "%d/%d" % (sum(1 for r in aff if r["off_lead"]), len(aff)),
                      "on_markers_per_affective_turn": "%d/%d" % (sum(1 for r in aff if r["on_lead"]), len(aff)),
                      "on_markers_surviving_lesion": "%d/%d" % (n_on_emits_survive_lesion, n_on_emits)},
            "lesion_attribution": {"n_on_emits": n_on_emits, "n_on_emits_survive_lesion": n_on_emits_survive_lesion,
                                   "fraction_attributable_to_lesion": lesion_attribution},
            "failures": bad, "verdict": decided, "preconditions": decided["preconditions"],
            "per_seed": {str(s): v for s, v in scored.items()},
            "rule": "GO iff every precondition holds (6/6 valid seeds turn by turn, H5 determinism, the replay lever "
                    "A1.3, non-vacuous A1.4) AND H1 (ON marker-superset over OFF), H2 (register identical wherever "
                    "both emit), H3 (ON's lesion silences ON's marker), H4 (no other field differs; replies equal "
                    "modulo each arm's own lead wherever it sits) hold on EVERY seed and turn. NO-GO iff the "
                    "preconditions hold and any H1-H4 cell fails. D1 (split first/warm read) and H3-OFF are "
                    "REPORTED, never gated."}


# ─────────────────────────────────────────────────── selftest ──────────────────────────────────────────────────
_PREFIX = "That's absolutely thrilling for This -- "


def _resp(lead="", level=2, answer="hi", extra=None, error=None, seed=42, prefix=""):
    if error:
        return {"_error": error}
    r = {"answer": prefix + lead + answer,
         "affect_drives": {"lead": lead, "level": level, "mood": 0.07, "felt_arousal": 0.07, "high_arousal": True,
                           "acted": True, "reason": "graded_affect", "seed": seed},
         "abstained": False}
    if extra:
        r.update(extra)
    return r


def _arms(off_leads, on_leads, on_lesion_leads=None, extra_on=None, off_b_leads=None, on_b_leads=None, seed=42,
          prefixes=None) -> dict:
    """A synthetic arms dict for ONE seed from per-turn lead lists (len == len(MT_LABELS)); `prefixes` puts a
    per-turn clause IN FRONT of the lead in every arm (the real mt_emo2/mt_neg1 answer shape)."""
    on_lesion_leads = on_lesion_leads or [""] * len(MT_LABELS)
    off_b_leads = off_b_leads or off_leads
    on_b_leads = on_b_leads or on_leads
    pre = prefixes or [""] * len(MT_LABELS)

    def turns(leads, extra=None):
        return {lab: _resp(leads[i], extra=(extra[i] if extra else None), seed=seed, prefix=pre[i])
                for i, lab in enumerate(MT_LABELS)}
    # on_b carries the SAME `extra` as on_a (unless on_b_leads was overridden) so an H4-only case does not ALSO
    # trip H5 as a side effect -- H5 is exercised deliberately only via off_b_leads/on_b_leads (case (f)).
    recs = {"off_a": turns(off_leads), "off_b": turns(off_b_leads), "off_lesion": turns([""] * len(MT_LABELS)),
            "on_a": turns(on_leads, extra_on), "on_b": turns(on_b_leads, extra_on if on_b_leads is on_leads else None),
            "on_lesion": turns(on_lesion_leads)}
    return {a: {"env": expected_env(a), "seed": str(seed), "turns": t} for a, t in recs.items()}


def _replay_ok(arms: dict, *, settle_ident=True) -> dict:
    """A synthetic replay record in which every arm reproduces its recorded leads with the expected reader."""
    def fact(a):
        settle = ARM_SPECS[a][0]
        exp = ({"settle": True, "warmup": 500, "washout": 1000} if settle else
               {"settle": False, "warmup": 60, "washout": 40})
        return {"arm": a, "expected_reader": exp, "readers": [dict(exp, seed=42)],
                "turns": {lab: {"recorded": _lead(arms[a]["turns"][lab]), "replayed": _lead(arms[a]["turns"][lab]),
                                "n_reader_calls": 1, "exceptions": []} for lab in MT_LABELS}}
    return {"factual": {a: fact(a) for a in ARM_SPECS}, "cf_settle_flipped": {}, "cf_lesion_undone": {}}


def _six(clean_fn, bad_fn=None, bad_seed: int = 101):
    """All six VERIFY_SEEDS carry `clean_fn(seed)` arms; `bad_fn(seed)` replaces ONE seed (default s101), so every
    failing-direction case proves a single bad seed among five clean ones sinks the verdict. Returns
    (per_seed_arms, replay) with a clean factual replay for every seed."""
    arms = {s: (bad_fn(s) if (bad_fn is not None and s == bad_seed) else clean_fn(s)) for s in VERIFY_SEEDS}
    # like `_score_and_write`, a replay exists only for a seed whose arms validated
    return arms, {s: _replay_ok(a) for s, a in arms.items() if score_seed(s, a)["valid"]}


def _selftest_scorer() -> bool:
    ok = True
    n = len(MT_LABELS)

    def check(name, got):
        nonlocal ok
        ok = ok and bool(got)
        print("  %-86s %s" % (name, "ok" if got else "FAIL"))

    # (a) clean case: OFF misses a WARM read (mt_emo2), ON gets it, lesion silences -> GO, D1 warm True
    off = ["", "Gladly! ", "", "Frankly — ", "Gladly! "]
    on = ["", "Gladly! ", "Gladly! ", "Frankly — ", "Gladly! "]
    arms, rep = _six(lambda s: _arms(off, on, seed=s))
    r = score_all(arms, replay=rep)
    check("(a) clean + warm-read defect -> GO, D1_warm, benefit SHOWN",
          r["status"] == "GO" and r["D1_warm_read"] and r["benefit_on_warm_reads"] == "SHOWN")
    # (b) H1 violated: ON drops a marker OFF had -> NO-GO
    off2 = ["", "Gladly! ", "Gladly! ", "Frankly — ", "Gladly! "]
    on2 = ["", "", "Gladly! ", "Frankly — ", "Gladly! "]
    arms, rep = _six(lambda s: _arms(off, on, seed=s), lambda s: _arms(off2, on2, seed=s))
    r = score_all(arms, replay=rep)
    check("(b) H1 superset violated -> NO-GO", r["status"] == "NO-GO" and not r["H1_superset"])
    # (c) H2 violated: both emit, different register -> NO-GO
    on3 = ["", "Wonderful! ", "Gladly! ", "Frankly — ", "Gladly! "]
    arms, rep = _six(lambda s: _arms(off, on, seed=s), lambda s: _arms(off, on3, seed=s))
    r = score_all(arms, replay=rep)
    check("(c) H2 register mismatch -> NO-GO", r["status"] == "NO-GO" and not r["H2_register"])
    # (d) H3 violated: ON emits, lesion does NOT silence -> NO-GO
    lesion_bad = ["", "Gladly! ", "", "", ""]
    arms, rep = _six(lambda s: _arms(off, on, seed=s), lambda s: _arms(off, on, on_lesion_leads=lesion_bad, seed=s))
    r = score_all(arms, replay=rep)
    check("(d) H3 lesion fails to silence -> NO-GO", r["status"] == "NO-GO" and not r["H3_load_bearing"])
    # (e) H4 violated: an ON arm changes an unrelated field (abstained flips) -> NO-GO
    extra_on = [None] * n
    extra_on[1] = {"abstained": True}
    arms, rep = _six(lambda s: _arms(off, on, seed=s), lambda s: _arms(off, on, extra_on=extra_on, seed=s))
    r = score_all(arms, replay=rep)
    check("(e) H4 isolation violated (unrelated field changed) -> NO-GO", r["status"] == "NO-GO" and not r["H4_isolation"])
    # (f) H5 violated: off_a vs off_b disagree -> UNDEFINED, never GO
    off_b_bad = ["", "Gladly! ", "Frankly — ", "Frankly — ", "Gladly! "]
    arms, rep = _six(lambda s: _arms(off, on, seed=s), lambda s: _arms(off, on, off_b_leads=off_b_bad, seed=s))
    r = score_all(arms, replay=rep)
    check("(f) H5 determinism failed -> UNDEFINED", r["status"] == "UNDEFINED" and not r["H5_determinism"])
    # (g) missing arm -> UNDEFINED
    def missing(s):
        a = _arms(off, on, seed=s)
        a["on_lesion"] = None
        return a
    arms, rep = _six(lambda s: _arms(off, on, seed=s), missing)
    r = score_all(arms, replay=rep)
    check("(g) missing arm -> UNDEFINED", r["status"] == "UNDEFINED" and not r["go"])
    # (h) A1.4: SETTLE that changed nothing (OFF==ON everywhere) -> UNDEFINED, never GO (was GO before A1)
    arms, rep = _six(lambda s: _arms(off2, off2, seed=s))
    r = score_all(arms, replay=rep)
    check("(h) OFF==ON on every seed/turn (vacuous contrast) -> UNDEFINED", r["status"] == "UNDEFINED" and not r["go"])
    # (h2) A1.5: the only OFF loss ON repairs is a FIRST read -> GO, but benefit on warm reads NOT_REPRODUCED
    off_first = ["", "", "Gladly! ", "Frankly — ", "Gladly! "]
    arms, rep = _six(lambda s: _arms(off_first, on, seed=s))
    r = score_all(arms, replay=rep)
    check("(h2) D1 only at the first read -> GO with D1_first, D1_warm False, benefit NOT_REPRODUCED",
          r["status"] == "GO" and r["D1_first_read"] and not r["D1_warm_read"]
          and r["benefit_on_warm_reads"] == "NOT_REPRODUCED" and "NOT reproduced" in r["interpretation"])
    # (j) SUBSET SCORE (19c8ed596): a clean seed 42 (and 42+43) scored ALONE must NOT read GO
    for sub in ((42,), (42, 43)):
        arms = {s: _arms(off, on, seed=s) for s in sub}
        r = score_all(arms, seeds=sub, replay={s: _replay_ok(a) for s, a in arms.items()})
        pre = [p for p in r["preconditions"] if p["name"].startswith("full 6-seed set")]
        check("(j) subset score seeds=%s -> UNDEFINED" % list(sub),
              (not r["go"]) and r["status"] == "UNDEFINED" and len(pre) == 1 and pre[0]["ok"] is False)
    # (k) A1.1 BLOCKER: a warm-read rescue with the marker MID-reply (the real mt_emo2 / mt_neg1 shape) -> GO, H4 ok
    pre = ["", "", _PREFIX, "That sounds devastating for This -- ", ""]
    arms, rep = _six(lambda s: _arms(off, on, seed=s, prefixes=pre))
    r = score_all(arms, replay=rep)
    check("(k) mid-reply marker rescue (clause before the lead) -> GO, H4 holds",
          r["status"] == "GO" and r["H4_isolation"] and r["D1_warm_read"])
    # (k2) the reviewer's reproduction: the PRE-A1 prefix-only strip on this shape -> H4 False (the bug), now True
    row = compare_turn(_resp("", prefix=_PREFIX), _resp("Wonderful! ", prefix=_PREFIX))
    check("(k2) compare_turn on the real answer shape: isolation_ok", row["isolation_ok"] and row["answer_equal_modulo_leads"])
    # (k3) the lead field names a marker the reply does not contain -> H4 fails (NO-GO)
    def ghost(s):
        a = _arms(off, on, seed=s)
        a["on_a"]["turns"]["mt_emo2"]["answer"] = "hi"
        a["on_b"]["turns"]["mt_emo2"]["answer"] = "hi"
        return a
    arms, rep = _six(lambda s: _arms(off, on, seed=s), ghost)
    r = score_all(arms, replay=rep)
    check("(k3) lead field set but absent from the reply -> NO-GO (H4)", r["status"] == "NO-GO" and not r["H4_isolation"])
    # (m) A1.2 BLOCKER: on_lesion errors on EVERY turn of ALL 6 seeds -> UNDEFINED (was GO with H3=True)
    def les_err(s):
        a = _arms(off, on, seed=s)
        a["on_lesion"]["turns"] = {lab: {"_error": "RuntimeError: boom"} for lab in MT_LABELS}
        return a
    arms, rep = _six(les_err)
    r = score_all(arms, replay=rep)
    check("(m) lesion arm errors on every turn, all seeds -> UNDEFINED", r["status"] == "UNDEFINED" and not r["go"])
    # (n) every arm has ZERO turns, all seeds -> UNDEFINED (was GO)
    def empty(s):
        return {a: {"env": expected_env(a), "seed": str(s), "turns": {}} for a in ARM_SPECS}
    arms, rep = _six(empty)
    r = score_all(arms, replay=rep)
    check("(n) every arm has zero turns, all seeds -> UNDEFINED", r["status"] == "UNDEFINED" and r["n_verify_seeds_valid"] == 0)
    # (o) every turn is an {'error': ...} body (a failed HTTP turn) -> UNDEFINED (was GO)
    def err_body(s):
        return {a: {"env": expected_env(a), "seed": str(s), "turns": {lab: {"error": "400"} for lab in MT_LABELS}}
                for a in ARM_SPECS}
    arms, rep = _six(err_body)
    r = score_all(arms, replay=rep)
    check("(o) every turn an {'error':...} body -> UNDEFINED", r["status"] == "UNDEFINED" and not r["go"])
    # (p) the affect read itself crashed in the lesion arm (reason 'error:...') -> UNDEFINED
    def reason_err(s):
        a = _arms(off, on, seed=s)
        a["on_lesion"]["turns"]["mt_emo2"]["affect_drives"]["reason"] = "error:RuntimeError: ladder"
        return a
    arms, rep = _six(lambda s: _arms(off, on, seed=s), reason_err)
    r = score_all(arms, replay=rep)
    check("(p) affect_drives.reason error:* in one lesion turn -> UNDEFINED", r["status"] == "UNDEFINED")
    # (q) an arm file whose env does not match its name (a mislabelled harvest) -> UNDEFINED
    def mislabel(s):
        a = _arms(off, on, seed=s)
        a["on_a"]["env"] = expected_env("off_a")
        return a
    arms, rep = _six(lambda s: _arms(off, on, seed=s), mislabel)
    r = score_all(arms, replay=rep)
    check("(q) arm env does not match its name -> UNDEFINED", r["status"] == "UNDEFINED")
    # (r) A1.3 lever: no replay / replay that does not reproduce a lead / reader built with the wrong config /
    # a reader exception -> UNDEFINED each
    arms, rep = _six(lambda s: _arms(off, on, seed=s))
    r = score_all(arms, replay=None)
    check("(r1) replay missing -> UNDEFINED", r["status"] == "UNDEFINED")
    rep_bad = {s: _replay_ok(a) for s, a in arms.items()}
    rep_bad[43]["factual"]["on_a"]["turns"]["mt_emo2"]["replayed"] = ""
    check("(r2) replay does not reproduce one recorded lead -> UNDEFINED",
          score_all(arms, replay=rep_bad)["status"] == "UNDEFINED")
    rep_bad = {s: _replay_ok(a) for s, a in arms.items()}
    rep_bad[44]["factual"]["on_a"]["readers"] = [{"seed": 44, "settle": False, "warmup": 60, "washout": 40}]
    check("(r3) ON env built an OFF-configured reader (SETTLE never reached it) -> UNDEFINED",
          score_all(arms, replay=rep_bad)["status"] == "UNDEFINED")
    rep_bad = {s: _replay_ok(a) for s, a in arms.items()}
    rep_bad[100]["factual"]["on_lesion"]["turns"]["mt_emo1"]["exceptions"] = ["select_valence: RuntimeError: x"]
    check("(r4) the reader raised during replay (a crash, not a cut) -> UNDEFINED",
          score_all(arms, replay=rep_bad)["status"] == "UNDEFINED")
    # (s) helpers
    check("(s1) register_of punctuation-independent",
          (register_of("Gladly! ") == register_of("Gladly — ") == "Gladly") and register_of("") == "")
    check("(s2) lead removal anywhere; absent lead -> empty set",
          _lead_removals("A -- Gladly! B", "Gladly! ") == {"A -- B"} and _lead_removals("AB", "Gladly! ") == set())
    return ok


def _selftest_replay() -> bool:
    """The replay machinery against the REAL production expression_lead on the DEV seed (7) only: a recorded
    sequence produced by the production path must be reproduced exactly, and its counterfactual must differ where
    SETTLE decides (no evaluation seed is touched; ~5 s)."""
    import research.runners._affect_marker_wta_derisk as W
    import webapp.affect_drives_chat as ADC
    moods = [(0, 0.0, 0.0), (2, 0.0685, 0.0508), (3, 0.0752, 0.0743), (-1, -0.0376, 0.0730), (2, 0.0659, 0.0741)]
    recs = {}
    for arm, (settle, lesion) in ARM_SPECS.items():
        saved = {k: os.environ.get(k) for k in (SETTLE_ENV, LESION_ENV)}
        saved_gr = W.get_reader
        try:
            os.environ[SETTLE_ENV], os.environ[LESION_ENV] = ("1" if settle else "0"), ("1" if lesion else "0")
            reader = W.AffectMarkerWTA(seed=7)
            W.get_reader = lambda seed=42, _r=reader: _r
            turns = {}
            for (lab, (lvl, m, fa)) in zip(MT_LABELS, moods):
                lead = ADC.expression_lead(lvl, True, mood=m, felt_arousal=fa, seed=7)
                turns[lab] = {"answer": lead + "x", "affect_drives": {"lead": lead, "level": lvl, "mood": m,
                              "felt_arousal": fa, "high_arousal": True, "acted": True, "reason": "graded_affect",
                              "seed": 7}}
        finally:
            W.get_reader = saved_gr
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        recs[arm] = {"env": expected_env(arm), "seed": "7", "turns": turns}
    rep = replay_seed(7, recs)
    lev = lever_from_replay(rep)
    ok1 = lev["ok"] is True
    print("  replay reproduces a production-path sequence on dev seed 7 (lever ok) -> %s %s"
          % ("ok" if ok1 else "FAIL", "" if ok1 else lev["problems"][:3]))
    tampered = json.loads(json.dumps(recs))
    tampered["on_a"]["turns"]["mt_emo1"]["affect_drives"]["lead"] = "Frankly — "
    lev2 = lever_from_replay(replay_seed(7, tampered))
    ok2 = lev2["ok"] is False
    print("  a recorded lead the reader could not have produced is CAUGHT by the replay -> %s" % ("ok" if ok2 else "FAIL"))
    ok3 = bool(lev["lesion_decides_lead_turns"])
    print("  the counterfactual (lesion undone) restores a marker the cut removed -> %s (%s)"
          % ("ok" if ok3 else "FAIL", lev["lesion_decides_lead_turns"][:3]))
    return ok1 and ok2 and ok3


def selftest(with_replay: bool = True) -> bool:
    ok = _selftest_scorer()
    if with_replay:
        ok = _selftest_replay() and ok
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


def _load_arms(raw_dir: str, seed: int) -> dict:
    sdir = os.path.join(raw_dir, "s%d" % seed)
    arms = {}
    for name in ARM_SPECS:
        p = os.path.join(sdir, "%s.json" % name)
        if os.path.exists(p):
            with open(p) as f:
                arms[name] = json.load(f)
        else:
            arms[name] = None
    return arms


def _score_and_write(per_seed_arms: dict, seeds, out: str, do_replay: bool) -> dict:
    replay = {}
    if do_replay:
        for s in seeds:
            if score_seed(s, per_seed_arms.get(s, {}))["valid"]:
                replay[s] = replay_seed(s, per_seed_arms[s])
    rec = score_all(per_seed_arms, seeds, replay=replay)
    rec["replay"] = {str(s): v for s, v in replay.items()}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("STATUS=%s GO=%s H1=%s H2=%s H3=%s H4=%s H5=%s D1_first=%s D1_warm=%s valid=%d/6 -> %s"
          % (rec["status"], rec["go"], rec["H1_superset"], rec["H2_register"], rec["H3_load_bearing"],
             rec["H4_isolation"], rec["H5_determinism"], rec["D1_first_read"], rec["D1_warm_read"],
             rec["n_verify_seeds_valid"], out))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-replay", action="store_true", help="skip the A1.3 replay (the verdict then reads UNDEFINED)")
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
        _score_and_write(per_seed_arms, seeds, a.out or os.path.join(a.out_dir, "verdict.json"), not a.no_replay)
        return
    if a.score:
        per_seed_arms = {s: _load_arms(a.raw_dir, s) for s in seeds}
        _score_and_write(per_seed_arms, seeds, a.out or os.path.join(a.raw_dir, "verdict.json"), not a.no_replay)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
