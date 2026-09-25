"""AFFECT-MARKER CONGRUENCE — an opt-in, default-OFF policy that suppresses the opening word when it would
CONTRADICT the reply it decorates (research/settle-multiturn-contrast, 2026-09-24).

WHY. The multi-turn contrast (`_affect_marker_settle_multiturn_derisk.py`) and the earlier single-turn battery
both observe the SAME anti-pattern: on a strongly-affective turn the felt-mood-driven marker fires ("Gladly! "/
"Wonderful! ") while the CONTENT path abstains (no matching fact for "Wonderful, I am so happy..."), so the
surface reads "Gladly! I don't know." -- a warm marker glued onto a non-answer. This module names that
INCONGRUENCE explicitly and gives an opt-in policy that suppresses the marker exactly there, instead of leaving
it to whatever a future reader infers.

TWO INCONGRUENCE CONDITIONS (either suppresses the lead):
  (1) ABSTENTION CONFLICT -- the turn's top-level `abstained` field is True. `abstained` is REUSED VERBATIM, not
      recomputed: it is the SAME decision field `onebrain_regression_battery.FACULTY_PROBES` already treats as
      the production decision boundary for `moat-verify` (the no-confab abstain) and `bg-action-selection` (the
      SPEAK-vs-STAY-SILENT race) -- this module adds no new abstention logic.
  (2) VALENCE CONFLICT -- the marker's register SIGN (positive: Wonderful/Gladly/Sure; negative: Hm/Honestly/
      Frankly) disagrees with `affect.valence_sign`, an INDEPENDENT neural read: the Gate-B co-resident affect
      ladder's OWN differential sign (`webapp/server.py`'s `_AO.tone_level(diff)` block, `read_differential`
      off `cp_firing_states` -- a DIFFERENT organ/pathway than the #84 ladder the marker itself reads). Two
      independently-computed spiking valence reads disagreeing is a spiking-grounded congruence signal, not a
      host sentiment recomputation; the comparison itself (equal signs?) is a host `if`, exactly as every
      existing decision threshold in this repo (DEAD_MARGIN, argsort) is a host READOUT of neural quantities.

HOST RESIDUAL, DECLARED (not hidden): the register-sign table (`POS_REGISTERS`/`NEG_REGISTERS`) is a host
lookup from the marker WORD to its sign -- the SAME table `webapp/affect_drives_chat._LEAD_WORD` already fixes
by construction (registers -3..-1 are named negative words, +1..+3 positive; this is a re-read of that existing
fixed mapping, not a new one). No sentiment is computed by this module; both signs it compares are ALREADY
present on the response the production path already produced.

SCOPE. Additive; NO `sim/`, `webapp/`, or existing-runner edit. `apply_policy()` is a pure function of an
already-collected `webapp.server.brain_chat` response dict -- it can be applied POST-HOC to any already-
collected arm (this de-risk reuses `_affect_marker_settle_multiturn_derisk`'s raw "mt" session arms so no
additional brain build is needed for the measurement) or wired as a live outermost decorator in a future
production change (not done here -- that is an owner-reserved wiring decision, same as the SETTLE flag itself).

MEASURING "lead-reply agreement OFF/ON": for a set of already-collected turns, OFF = the response AS SHIPPED
(the raw `affect_drives.lead`); ON = `apply_policy(resp, enabled=True)`'s lead. `congruent(resp)` is True iff
lead=="" (nothing to disagree with) OR (not abstained AND (no independent sign read OR signs agree)). Agreement
rate = congruent count / total. By construction ON's rate on the FLAGGED reasons is 1.0 (the policy suppresses
exactly the incongruent cases) -- the number that matters is (a) how often OFF is ALREADY incongruent (the size
of the defect) and (b) that ON never touches an ALREADY-congruent turn's lead (no over-suppression).

Run (spawn a small SELF-CONTAINED battery -- 3 single-turn probes reused verbatim from the regression battery
["well","unknown","emo"] -- across all 6 seeds; settle is read from the SAME BRAIN_AFFECT_MARKER_SETTLE the
multi-turn contrast candidate would ship with):
  SIM_BACKEND=numpy OMP_NUM_THREADS=1 tools/memcap.sh 10 -- .venv/bin/python -u -m \\
    research.runners._affect_marker_settle_congruence --run --settle 1 --seeds 42 \\
    --out-dir research/findings/raw/_affect_marker_settle_congruence

Score turns ALREADY collected by the multi-turn contrast (no extra brain build):
  .venv/bin/python -m research.runners._affect_marker_settle_congruence --score-multiturn \\
    --raw-dir research/findings/raw/_affect_marker_settle_multiturn --arm on_a \\
    --seeds "42 43 44 100 101 102" --out research/findings/raw/_affect_marker_settle_congruence/multiturn.json

Self-test (no brain build):
  .venv/bin/python -m research.runners._affect_marker_settle_congruence --selftest

AMENDMENT A2 (2026-09-24, fix round after an independent review; committed BEFORE any amended score is computed;
governing document research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md).
  A2.0 HISTORY, stated plainly. The "rule holds" GO/NO-GO (8d4605b74) and the surface fix were written AFTER the
       lane had seen s42-s44 data. That criterion carries NO pre-registered weight and is withdrawn as a
       flip criterion; the numbers it produced are descriptive only.
  A2.1 NOT A PRODUCTION PROPERTY. `apply_policy` edits already-recorded replies. Nothing in webapp/ reads
       BRAIN_AFFECT_MARKER_CONGRUENCE, so the policy is not `wired` (docs/TERMS.md) and "the rule holds" says
       nothing about what a SETTLE flip would ship. `score_all` therefore requires the policy to have been applied
       BY the production path; every mode in this runner is post-hoc, so the rule's status is UNDEFINED until a
       production wiring exists and is scored as shipped.
  A2.2 THE VALENCE HALF MUST BE EXERCISED. Condition (2) can only fire when the Gate-B read takes the sign opposite
       to the marker. On s42-s44 `affect.valence_sign` never read '-' (even the strongly negative mt_neg1 read
       '0'), so "0 markers on an opposite-sign read" was empty by construction. The rule's status now requires
       Gate-B to read '-' on >=1 scored turn AND '+' on >=1; otherwise the valence half is UNTESTED -> UNDEFINED.
  A2.3 NAMED HOST SHORTCUT S6. The policy is a host string edit (abstained, so delete the marker word from the
       reply). Under the brain-based-only standard it is a declared host shortcut, not a brain mechanism. A
       production congruence mechanism would be a separate design decision (for example, gating the marker by the
       same spiking speak/abstain race that decides abstention), not this edit.
  A2.4 DENOMINATORS. Rates are reported per AFFECTIVE turn (appraisal hits or a non-zero affect level), not per
       turn: the neutral turn cannot carry a marker and was diluting the denominator (12/15 is 12/12).
  A2.5 --score-settle-contrast (DESCRIPTIVE, no threshold, no GO/NO-GO): off_a vs on_a AS SHIPPED (no policy),
       per affective turn: markers, markers attached to an abstention, markers on an opposite-sign Gate-B read;
       and the markers that would SURVIVE the policy in each arm (SETTLE's visible effect if the policy shipped).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

VERIFY_SEEDS = (42, 43, 44, 100, 101, 102)
CONGRUENCE_ENV = "BRAIN_AFFECT_MARKER_CONGRUENCE"
SETTLE_ENV = "BRAIN_AFFECT_MARKER_SETTLE"

# the SAME fixed word->sign mapping `webapp/affect_drives_chat._LEAD_WORD` already carries (levels -3..-1 are
# the negative words, +1..+3 the positive ones) -- re-read here, not re-derived.
POS_REGISTERS = {"Wonderful", "Gladly", "Sure"}
NEG_REGISTERS = {"Hm", "Honestly", "Frankly"}
_SIGN_MAP = {"+": 1, "-": -1, "0": 0}


def congruence_enabled(explicit=None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(CONGRUENCE_ENV, "0").strip().lower() in ("1", "true", "on", "yes")


def congruence_wired_in_webapp() -> bool:
    """A2.1, REPORTED: does any webapp/*.py module read the policy flag at all? (False today: the policy exists only
    in this research runner.) A static read, so it is reported beside the status, never used to grant GO."""
    import glob
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for p in glob.glob(os.path.join(root, "webapp", "*.py")):
        try:
            with open(p, encoding="utf-8") as f:
                if CONGRUENCE_ENV in f.read():
                    return True
        except OSError:
            continue
    return False


def register_word(lead: str) -> str:
    from research.runners._affect_marker_settle_multiturn_derisk import register_of
    return register_of(lead)


def register_sign(word: str) -> int:
    if word in POS_REGISTERS:
        return 1
    if word in NEG_REGISTERS:
        return -1
    return 0


def diagnose(resp: dict) -> dict:
    """The read-only diagnosis (never mutates `resp`): the two candidate incongruence conditions, computed from
    fields the production response ALREADY carries."""
    ad = resp.get("affect_drives") or {}
    lead = str(ad.get("lead", "") or "")
    word = register_word(lead)
    rsign = register_sign(word)
    abstained = bool(resp.get("abstained"))
    vsign_raw = (resp.get("affect") or {}).get("valence_sign")
    vsign = _SIGN_MAP.get(vsign_raw)
    abstention_conflict = bool(lead and abstained)
    valence_conflict = bool(lead and vsign is not None and rsign != 0 and vsign != 0 and rsign != vsign)
    congruent = bool((not lead) or (not abstention_conflict and not valence_conflict))
    return {"lead": lead, "register_word": word, "register_sign": rsign, "abstained": abstained,
            "gateb_valence_sign_raw": vsign_raw, "gateb_valence_sign": vsign,
            "abstention_conflict": abstention_conflict, "valence_conflict": valence_conflict,
            "incongruent": not congruent, "congruent": congruent}


def apply_policy(resp: dict, *, enabled=None) -> tuple:
    """Returns (possibly-modified copy of resp, diagnosis). Enabled defaults to reading BRAIN_AFFECT_MARKER_
    CONGRUENCE from the environment (byte-identical passthrough when unset/false); pass enabled=True/False to
    score both arms from ONE already-collected response without touching the environment."""
    diag = diagnose(resp)
    on = congruence_enabled(enabled)
    if not on or not diag["incongruent"]:
        return dict(resp), diag
    out = dict(resp)
    lead = diag["lead"]
    answer = str(out.get("answer", "") or "")
    # SURFACE FIX (2026-09-24): the lead is inserted VERBATIM (`resp['answer'] = lead + answer`), but a later stage
    # can PREPEND text ("That's absolutely thrilling for This -- Wonderful! I don't know..."), so a prefix-only strip
    # left the marker in the reply on 6/12 incongruent real turns (every mt_emo2 / mt_neg1, s42-s44) while the
    # `lead` FIELD read blank. Remove the lead's first verbatim occurrence wherever it sits.
    if lead and answer.startswith(lead):
        out["answer"] = answer[len(lead):]
    elif lead and lead in answer:
        out["answer"] = answer.replace(lead, "", 1)
    else:
        out["answer"] = answer
    ad = dict(out.get("affect_drives") or {})
    ad["lead"] = ""
    ad["congruence_suppressed"] = True
    ad["congruence_suppressed_reason"] = "abstention" if diag["abstention_conflict"] else "valence_mismatch"
    out["affect_drives"] = ad
    return out, diag


# ─────────────────────────────────────────── measurement: a small self-contained battery ───────────────────────
RUN_TURN_LABELS = ("well", "unknown", "emo")   # reused VERBATIM from onebrain_regression_battery._TURN_BY_LABEL


def _spawn(seed: int, settle: bool, out_path: str):
    from research.runners.onebrain_regression_battery import _spawn_arm
    env = {SETTLE_ENV: "1" if settle else "0"}
    host_env = dict(os.environ)
    host_env["BRAIN_CHAT_SEED"] = str(int(seed))
    host_env.setdefault("SIM_BACKEND", "numpy")
    host_env.setdefault("BRAIN_CHAT_RENDERER", "stub")
    host_env.setdefault("SIM_DISABLE_LLM", "1")
    # _spawn_arm shells out with `env=dict(os.environ)` internally; set the seed/flag on THIS process's env for
    # the duration of the call so the child inherits them (mirrors load_bearing_fraction's own seed threading).
    old = {k: os.environ.get(k) for k in host_env}
    os.environ.update(host_env)
    try:
        return _spawn_arm(env, list(RUN_TURN_LABELS), out_path)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_seed(seed: int, settle: bool, out_dir: str) -> dict:
    """Returns {label: response_dict} for this seed (or {} if the arm build failed) -- `_spawn_arm`'s own
    return shape (a bare label->response map), never re-wrapped."""
    sdir = os.path.join(out_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    turns = _spawn(seed, settle, os.path.join(sdir, "responses.json"))
    return turns or {}


# ─────────────────────────────────────────────────────── scoring ───────────────────────────────────────────────
def score_turns(turns: dict) -> dict:
    """`turns` = {label: response_dict}. Returns per-turn OFF vs ON diagnosis + the ON-arm's own re-check."""
    per = {}
    for label, resp in turns.items():
        if not isinstance(resp, dict) or "_error" in resp:
            per[label] = {"error": True}
            continue
        off_resp, off_diag = apply_policy(resp, enabled=False)
        on_resp, on_diag = apply_policy(resp, enabled=True)
        on_recheck = diagnose(on_resp)
        # SURFACE RE-CHECK (2026-09-24): `diagnose` reads the `lead` FIELD, which the policy blanks -- so a marker
        # left in the reply TEXT scored as removed (the instrument read the field, not what the user sees). An
        # incongruent turn is only resolved ON if the ON reply carries FEWER verbatim copies of the lead than OFF.
        lead_off = off_diag["lead"]
        surface_residual = bool(off_diag["incongruent"] and lead_off and
                                str(on_resp.get("answer", "") or "").count(lead_off)
                                >= str(off_resp.get("answer", "") or "").count(lead_off))
        # NO OVER-SUPPRESSION: a turn OFF already congruent must have an IDENTICAL lead ON.
        unchanged_when_congruent = bool(off_diag["congruent"] and off_resp.get("affect_drives", {}).get("lead")
                                        == on_resp.get("affect_drives", {}).get("lead"))
        per[label] = {"off_diag": off_diag, "on_diag": on_diag, "on_recheck_incongruent": bool(on_recheck["incongruent"] or surface_residual),
                     "on_surface_residual": surface_residual,
                     "off_lead": off_diag["lead"], "on_lead": on_diag["lead"] if not on_diag["incongruent"] else "",
                     "no_over_suppression_ok": bool((not off_diag["congruent"]) or unchanged_when_congruent)}
    return per


def is_affective(resp: dict) -> bool:
    """A2.4: a turn that can carry a marker -- the appraisal heard affect words, or the felt level is non-zero."""
    if not isinstance(resp, dict):
        return False
    hits = (resp.get("affect") or {}).get("appraisal_hits")
    level = (resp.get("affect_drives") or {}).get("level")
    return bool(hits) or (level not in (None, 0))


def score_all(per_seed_turns: dict, seeds=VERIFY_SEEDS, expected_turns=None, post_hoc=True) -> dict:
    """`expected_turns` = the turn labels every seed must carry (MT_LABELS for --score-multiturn, RUN_TURN_LABELS
    for --run). When given, the verdict below requires EVERY VERIFY_SEED to be scored with every expected turn
    present and error-free -- the 2026-09-24 fix for a missing seed file silently contributing zero turns (a
    3-seed score read as if it were the whole set; the same class as the multiturn scorer's subset-GO bug).
    `post_hoc` (A2.1): True when this function APPLIES the policy to recorded replies -- every mode in this runner.
    The status can only be GO for replies the production path itself produced with the policy applied."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    n_total = n_off_incongruent = n_on_incongruent = n_over_suppress_violations = n_errors = 0
    n_off_leads = n_on_leads_after_policy = 0
    n_affective = n_gateb_neg = n_gateb_pos = 0
    detail = []
    incomplete = []
    for s in VERIFY_SEEDS:
        if s not in seeds:
            incomplete.append("s%d not scored (not in --seeds)" % s)
    for s in seeds:
        turns = per_seed_turns.get(s) or {}
        if expected_turns is not None:
            absent = [lab for lab in expected_turns if lab not in turns]
            if absent:
                incomplete.append("s%d missing turns %s" % (s, absent))
        scored = score_turns(turns)
        for label, row in scored.items():
            n_total += 1
            if row.get("error"):
                n_errors += 1
                incomplete.append("s%d %s: arm error" % (s, label))
                continue
            resp = turns.get(label)
            row["affective"] = is_affective(resp)
            n_affective += int(row["affective"])
            vs = row["off_diag"]["gateb_valence_sign_raw"]
            n_gateb_neg += int(vs == "-")
            n_gateb_pos += int(vs == "+")
            if row["off_diag"]["incongruent"]:
                n_off_incongruent += 1
            if row["on_recheck_incongruent"]:
                n_on_incongruent += 1
            if not row["no_over_suppression_ok"]:
                n_over_suppress_violations += 1
            if row["off_lead"]:
                n_off_leads += 1
            if row["on_lead"]:
                n_on_leads_after_policy += 1
            detail.append({"seed": s, "turn": label, **row})
    denom = max(n_total - n_errors, 0)
    off_rate = (n_off_incongruent / denom) if denom else None
    on_rate = (n_on_incongruent / denom) if denom else None
    attribution = None
    if n_off_incongruent > 0:
        attribution = attributable_to("congruence policy: incongruent-turn count OFF vs ON",
                                      float(n_off_incongruent), float(n_on_incongruent))
    # THE RULE (the invariants prereg b2e50bd37 already names, made a verdict 2026-09-24): with the policy ON, no
    # scored turn is left incongruent (no marker glued onto an abstention or onto an opposite-sign Gate-B read),
    # and no already-congruent lead is touched (no over-suppression). Scored over the WHOLE 6-seed set only.
    complete = bool(expected_turns is not None and not incomplete and denom > 0)
    vd = Verdict("affect_marker_settle_congruence_rule")
    vd.require("all 6 verification seeds scored, every expected turn present and error-free",
               complete, expect=True, note="; ".join(incomplete[:6]))
    vd.require("the policy was applied BY the production path, not post-hoc to recorded replies (A2.1)",
               not post_hoc, expect=True,
               note="post-hoc: nothing in webapp/ reads %s; the rule is not wired" % CONGRUENCE_ENV if post_hoc else "")
    valence_exercised = bool(n_gateb_neg >= 1 and n_gateb_pos >= 1)
    vd.require("valence half exercised: Gate-B read '-' on >=1 scored turn and '+' on >=1 (A2.2)",
               valence_exercised, expect=True,
               note="Gate-B '-' reads=%d, '+' reads=%d; without both, condition (2) cannot fire and is UNTESTED"
                    % (n_gateb_neg, n_gateb_pos))
    rule_holds = bool(n_on_incongruent == 0 and n_over_suppress_violations == 0)
    decided = vd.decide(bool(complete and rule_holds))
    return {"probe": "affect_marker_settle_congruence", "amendment": "A2 (2026-09-24)", "seeds": list(seeds),
            "n_total": n_total, "post_hoc": bool(post_hoc),
            "wired_in_webapp": congruence_wired_in_webapp(),
            "rule_holds": rule_holds, "status": decided["status"], "go": bool(decided["go"]),
            "preconditions": decided["preconditions"], "incomplete": incomplete,
            "n_off_leads": n_off_leads, "n_on_leads_after_policy": n_on_leads_after_policy,
            "n_affective_turns": n_affective, "gateb_minus_reads": n_gateb_neg, "gateb_plus_reads": n_gateb_pos,
            "valence_half": "EXERCISED" if valence_exercised else "UNTESTED",
            "off_incongruent_per_affective_turn": "%d/%d" % (n_off_incongruent, n_affective),
            "rule": "status GO iff, over all 6 verification seeds with every turn present and error-free, the policy "
                    "as APPLIED BY PRODUCTION leaves 0 incongruent turns AND never changes an already-congruent lead, "
                    "with both halves exercised (A2.1/A2.2). Post-hoc scoring (every current mode) reads UNDEFINED; "
                    "rule_holds is then descriptive only. n_on_leads_after_policy is REPORTED: how many markers "
                    "still reach the reply once the rule is applied.",
            "n_errors": n_errors, "n_off_incongruent": n_off_incongruent, "n_on_incongruent": n_on_incongruent,
            "off_incongruent_rate": off_rate, "on_incongruent_rate": on_rate,
            "off_agreement_rate": (1.0 - off_rate) if off_rate is not None else None,
            "on_agreement_rate": (1.0 - on_rate) if on_rate is not None else None,
            "n_over_suppression_violations": n_over_suppress_violations,
            "over_suppression_clean": n_over_suppress_violations == 0,
            "fraction_incongruence_removed_by_policy": attribution, "detail": detail}


# ─────────────────────────── A2.5: SETTLE's effect on congruence AS SHIPPED (descriptive, no threshold) ─────────
def _arm_counts(turns: dict) -> dict:
    """Counts for ONE arm's turns, AS SHIPPED (no policy) and with the policy applied post-hoc."""
    c = {"n_affective": 0, "markers": 0, "abstention_attached": 0, "opposite_sign_attached": 0,
         "markers_surviving_policy": 0, "errors": 0}
    for label, resp in (turns or {}).items():
        if not isinstance(resp, dict) or "_error" in resp:
            c["errors"] += 1
            continue
        if not is_affective(resp):
            continue
        c["n_affective"] += 1
        d = diagnose(resp)
        c["markers"] += int(bool(d["lead"]))
        c["abstention_attached"] += int(d["abstention_conflict"])
        c["opposite_sign_attached"] += int(d["valence_conflict"])
        on_resp, _ = apply_policy(resp, enabled=True)
        c["markers_surviving_policy"] += int(bool((on_resp.get("affect_drives") or {}).get("lead")))
    return c


def settle_contrast(off_by_seed: dict, on_by_seed: dict, expected_turns) -> dict:
    """off_by_seed / on_by_seed = {seed: {label: response}} from the multi-turn off_a / on_a arms. DESCRIPTIVE:
    what flipping SETTLE changes in the shipped reply's marker/abstention congruence, and what would remain of
    SETTLE's visible effect if the post-hoc policy were shipped. No threshold, no verdict (A2.5)."""
    per_seed, missing = {}, []
    tot = {"off": {}, "on": {}}
    for s in VERIFY_SEEDS:
        row = {}
        for arm, src in (("off", off_by_seed), ("on", on_by_seed)):
            turns = src.get(s) or {}
            absent = [lab for lab in expected_turns if lab not in turns]
            if absent:
                missing.append("s%d %s missing turns %s" % (s, arm, absent))
            c = _arm_counts(turns)
            if c["errors"]:
                missing.append("s%d %s: %d errored turns" % (s, arm, c["errors"]))
            row[arm] = c
            for k, v in c.items():
                tot[arm][k] = tot[arm].get(k, 0) + v
        per_seed[str(s)] = row
    complete = not missing

    def frac(arm, k):
        return "%d/%d" % (tot[arm].get(k, 0), tot[arm].get("n_affective", 0))
    return {"probe": "affect_marker_settle_congruence_settle_contrast", "amendment": "A2.5 (2026-09-24)",
            "descriptive_only": True, "complete": complete, "incomplete": missing,
            "per_affective_turn": {arm: {k: frac(arm, k) for k in ("markers", "abstention_attached",
                                                                     "opposite_sign_attached",
                                                                     "markers_surviving_policy")}
                                   for arm in ("off", "on")},
            "settle_delta_as_shipped": {k: tot["on"].get(k, 0) - tot["off"].get(k, 0)
                                        for k in ("markers", "abstention_attached", "opposite_sign_attached")},
            "settle_delta_if_policy_shipped": tot["on"].get("markers_surviving_policy", 0)
                                              - tot["off"].get("markers_surviving_policy", 0),
            "totals": tot, "per_seed": per_seed,
            "note": "abstention_attached = a marker on a turn whose top-level `abstained` is True, AS SHIPPED today "
                    "(no policy is wired). settle_delta_if_policy_shipped = 0 means the policy would erase every "
                    "marker SETTLE adds."}


# ══════════════════ AMENDMENT 2 (2026-09-25): the PRODUCTION-PATH battery/probe hook for A2 ══════════════════
# WHY. A2.1 above says the policy is not `wired` and every mode in THIS file is post-hoc. The 2026-09-25 amendment
# (research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md, "Amendment 2") wires a
# congruence GATE into `webapp/affect_drives_chat.py` (`congruence_gate`, `BRAIN_AFFECT_MARKER_CONGRUENCE`,
# default OFF) called at the two live `/api/brain-chat` sites, BEFORE the marker is ever prepended -- not this
# file's post-hoc `apply_policy`. This section measures THAT wiring, on a real production `brain_chat` call, over
# the SAME 5-turn sequence A1 uses (`_affect_marker_settle_multiturn_derisk.MT_TURNS`) so the SAME turns already
# known to contain an abstention candidate (mt_emo1) and a Gate-B-negative candidate (mt_neg1) exercise A2's own
# preconditions, instead of a new hand-built battery.
WIRING_TURN_SOURCE = "_affect_marker_settle_multiturn_derisk.MT_TURNS"


def _worker_wiring(env_json: str, out_path: str) -> int:
    """Subprocess entry (`--worker-wiring`): ONE fresh tiny-demo brain, the SAME 5-turn `MT_TURNS` sequence A1
    replays, on the numpy/stub/no-LLM harness (`onebrain_regression_battery`'s convention -- fast + deterministic;
    the GPU/Qwen production-latency question is A3's, not this one's)."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v            # explicit "0"/"1"; a key absent from `env` is left UNSET (the "unset" arm)
    from webapp.server import brain_chat, BrainChatRequest
    from research.runners._affect_marker_settle_multiturn_derisk import MT_TURNS
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
    print("[wiring worker] env=%s seed=%s -> %d turns -> %s" % (env, rec["seed"], len(responses), out_path),
          flush=True)
    return 0


WIRING_ARMS = {"unset": None, "off": "0", "on": "1"}    # env value to set; None = key left OUT of `env` entirely


def _spawn_wiring_arm(seed: int, arm: str, out_path: str):
    value = WIRING_ARMS[arm]
    env = {} if value is None else {CONGRUENCE_ENV: value}
    host_env = dict(os.environ)
    host_env.pop(CONGRUENCE_ENV, None)          # the "unset" arm must not inherit the CALLER's shell env either
    host_env["BRAIN_CHAT_SEED"] = str(int(seed))
    host_env.setdefault("SIM_BACKEND", "numpy")
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners._affect_marker_settle_congruence",
                        "--worker-wiring", "--env", json.dumps(env), "--out", out_path], env=host_env)
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)


def run_seed_wiring(seed: int, out_dir: str) -> dict:
    sdir = os.path.join(out_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    arms = {}
    for name in WIRING_ARMS:
        arms[name] = _spawn_wiring_arm(seed, name, os.path.join(sdir, "%s.json" % name))
        print("  s%d %-6s -> %s" % (seed, name, "OK" if arms[name] is not None else "FAILED"), flush=True)
    return arms


def _deep_diff(a, b, path: str = "") -> list:
    """Every leaf where `a` and `b` disagree, as `path: a_value != b_value` -- the exact-compare byte-identical
    check (docs/TERMS.md: asserted IN THE DATA, never inferred). No ignore-list: two production runs of the SAME
    turns at the SAME seed with the flag off/unset are expected to be IDENTICAL, not merely close (this project's
    own determinism discipline, tests/test_determinism.py::TestSubstrateActuallySeeded)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out.extend(_deep_diff(a.get(k, "<absent>"), b.get(k, "<absent>"), "%s.%s" % (path, k)))
        return out
    if isinstance(a, list) and isinstance(b, list):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(_deep_diff(x, y, "%s[%d]" % (path, i)))
        if len(a) != len(b):
            out.append("%s: length %d != %d" % (path, len(a), len(b)))
        return out
    if a != b:
        return ["%s: %r != %r" % (path, a, b)]
    return []


def score_wiring(per_seed_arms: dict, seeds=VERIFY_SEEDS) -> dict:
    """The Amendment-2 rule (research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md).
    `per_seed_arms[seed]` = {"unset": turns, "off": turns, "on": turns} from `run_seed_wiring`, `turns` = {label:
    response}. GO iff (1) holds on all valid seeds AND (2)+(3) hold across the set. NO-GO iff (1) holds but (2) or
    (3) fails. UNDEFINED if (1) fails on any seed, a process errored, or fewer than 6 seeds are valid."""
    from research.runners._affect_marker_settle_multiturn_derisk import MT_LABELS
    per_seed_report, byte_identical_off_ok = {}, True
    n_valid = 0
    neg_seen = pos_seen = False
    non_vacuous_instances, attribution_problems = [], []
    for s in seeds:
        arms = per_seed_arms.get(s) or {}
        unset, off, on = arms.get("unset"), arms.get("off"), arms.get("on")
        row = {"seed": s, "valid": False, "problems": []}
        if not (isinstance(unset, dict) and isinstance(off, dict) and isinstance(on, dict)):
            row["problems"].append("one or more arms missing/failed to spawn")
            per_seed_report[str(s)] = row
            continue
        u_turns, o_turns, n_turns = unset.get("turns", {}), off.get("turns", {}), on.get("turns", {})
        missing = [lab for lab in MT_LABELS if lab not in u_turns or lab not in o_turns or lab not in n_turns]
        if missing:
            row["problems"].append("missing turns: %s" % missing)
            per_seed_report[str(s)] = row
            continue
        errored = [lab for lab in MT_LABELS if any("_error" in (t.get(lab) or {}) for t in (u_turns, o_turns, n_turns))]
        if errored:
            row["problems"].append("errored turns: %s" % errored)
            per_seed_report[str(s)] = row
            continue
        # (1) byte-identical-off: unset == off, EVERY field, every turn.
        diffs = _deep_diff(u_turns, o_turns)
        row["byte_identical_off"] = (not diffs)
        row["byte_identical_off_diffs"] = diffs[:20]
        if diffs:
            byte_identical_off_ok = False
            row["problems"].append("unset != off (%d field diff(s))" % len(diffs))
        # (2) both Gate-B signs
        for lab in MT_LABELS:
            sign = (n_turns[lab].get("affect") or {}).get("valence_sign")
            neg_seen = neg_seen or (sign == "-")
            pos_seen = pos_seen or (sign == "+")
        # (3)+(4) non-vacuous + attribution
        for lab in MT_LABELS:
            off_lead = str((o_turns[lab].get("affect_drives") or {}).get("lead", "") or "")
            on_turn = n_turns[lab]
            cg = on_turn.get("affect_marker_congruence")
            suppressed = bool(cg and cg.get("suppressed"))
            if off_lead and suppressed:
                non_vacuous_instances.append("s%d/%s" % (s, lab))
                reason = cg.get("reason")
                if reason == "abstention" and not bool(on_turn.get("abstained")):
                    attribution_problems.append("s%d/%s: reason=abstention but abstained is False" % (s, lab))
                if reason == "valence_mismatch":
                    off_sign = _register_sign_of(_register_word_of(off_lead))
                    on_sign_raw = (on_turn.get("affect") or {}).get("valence_sign")
                    on_sign = {"+": 1, "-": -1, "0": 0}.get(on_sign_raw)
                    disagree = bool(off_sign != 0 and on_sign not in (0, None) and on_sign != off_sign)
                    if not disagree:
                        attribution_problems.append("s%d/%s: reason=valence_mismatch but signs do not disagree "
                                                    "(off_sign=%s on_sign=%s)" % (s, lab, off_sign, on_sign_raw))
                if off_lead in str(on_turn.get("answer", "")):
                    attribution_problems.append("s%d/%s: marker claimed suppressed but its text is still in the "
                                                "reply" % (s, lab))
        row["valid"] = True
        n_valid += 1
        per_seed_report[str(s)] = row
    valence_half = "EXERCISED" if (neg_seen and pos_seen) else "UNTESTED"
    preconditions_ok = (n_valid == len(seeds) and byte_identical_off_ok)
    non_vacuous = bool(non_vacuous_instances)
    if not preconditions_ok or valence_half != "EXERCISED" or not non_vacuous:
        status, go = "UNDEFINED", False
    elif attribution_problems:
        status, go = "NO-GO", False
    else:
        status, go = "GO", True
    return {"probe": "affect_marker_settle_congruence_wiring", "amendment": "Amendment 2 (2026-09-25)",
            "seeds": list(seeds), "n_valid": n_valid, "byte_identical_off": byte_identical_off_ok,
            "valence_half": valence_half, "non_vacuous": non_vacuous,
            "non_vacuous_instances": non_vacuous_instances, "attribution_problems": attribution_problems,
            "status": status, "go": go, "per_seed": per_seed_report,
            "rule": "GO iff byte-identical-off holds on every valid seed AND both Gate-B signs are exercised AND "
                    "the gate suppresses >=1 real production marker AND every suppression's declared reason "
                    "matches its turn's own abstained/valence-sign fields. NO-GO iff preconditions+non-vacuous+"
                    "valence-half hold but an attribution mismatch is found. UNDEFINED otherwise."}


def _register_word_of(lead: str) -> str:
    from webapp.affect_drives_chat import _register_word
    return _register_word(lead)


def _register_sign_of(word: str) -> int:
    from webapp.affect_drives_chat import _register_sign
    return _register_sign(word)


def _selftest_wiring() -> bool:
    """Pure scorer checks (no brain build): drives `score_wiring` through every failing direction."""
    ok = True

    def turn(lead="", abstained=False, vsign=None, answer="hi", cg=None):
        d = {"answer": (lead + answer) if lead else answer, "abstained": abstained,
             "affect_drives": {"lead": lead}, "affect": ({"valence_sign": vsign} if vsign is not None else {})}
        if cg is not None:
            d["affect_marker_congruence"] = cg
        return d

    from research.runners._affect_marker_settle_multiturn_derisk import MT_LABELS
    base_off = {lab: turn(lead=("Gladly! " if lab == "mt_emo1" else ""), abstained=(lab == "mt_emo1"))
               for lab in MT_LABELS}
    base_unset = dict(base_off)  # identical dict contents -> byte-identical
    base_on = {lab: turn(abstained=(lab == "mt_emo1"), vsign=("-" if lab == "mt_neg1" else "+"),
                        cg=({"suppressed": True, "reason": "abstention"} if lab == "mt_emo1" else None))
              for lab in MT_LABELS}
    good = {s: {"unset": {"turns": base_unset}, "off": {"turns": base_off}, "on": {"turns": base_on}}
           for s in VERIFY_SEEDS}
    r1 = score_wiring(good, VERIFY_SEEDS)
    got1 = r1["status"] == "GO" and r1["byte_identical_off"] and r1["valence_half"] == "EXERCISED" and r1["non_vacuous"]
    ok = ok and got1
    print("  clean 6-seed set, both signs seen, one real suppression, attribution matches -> GO ->",
          "ok" if got1 else "FAIL (%r)" % r1["status"])
    # FAILING DIRECTION: unset != off (the flag is NOT truly a no-op) -> UNDEFINED, never GO
    bad_off = dict(base_off, mt_emo3=turn(lead="Sure — "))   # off differs from unset on one field
    bad = {s: {"unset": {"turns": base_unset}, "off": {"turns": bad_off}, "on": {"turns": base_on}}
          for s in VERIFY_SEEDS}
    r2 = score_wiring(bad, VERIFY_SEEDS)
    got2 = r2["status"] == "UNDEFINED" and not r2["byte_identical_off"]
    ok = ok and got2
    print("  unset != off (flag is not a no-op) -> UNDEFINED ->", "ok" if got2 else "FAIL (%r)" % r2["status"])
    # FAILING DIRECTION: Gate-B never reads '-' -> valence half UNTESTED -> UNDEFINED
    plus_only_on = {lab: turn(abstained=(lab == "mt_emo1"), vsign="+",
                              cg=({"suppressed": True, "reason": "abstention"} if lab == "mt_emo1" else None))
                   for lab in MT_LABELS}
    r3 = score_wiring({s: {"unset": {"turns": base_unset}, "off": {"turns": base_off},
                          "on": {"turns": plus_only_on}} for s in VERIFY_SEEDS}, VERIFY_SEEDS)
    got3 = r3["status"] == "UNDEFINED" and r3["valence_half"] == "UNTESTED"
    ok = ok and got3
    print("  Gate-B never reads '-' -> valence half UNTESTED -> UNDEFINED ->", "ok" if got3 else "FAIL (%r)" % r3["status"])
    # FAILING DIRECTION: the gate never suppresses anything real (vacuous) -> UNDEFINED, never GO
    never_on = {lab: turn(abstained=(lab == "mt_emo1"), vsign=("-" if lab == "mt_neg1" else "+")) for lab in MT_LABELS}
    r4 = score_wiring({s: {"unset": {"turns": base_unset}, "off": {"turns": base_off}, "on": {"turns": never_on}}
                       for s in VERIFY_SEEDS}, VERIFY_SEEDS)
    got4 = r4["status"] == "UNDEFINED" and not r4["non_vacuous"]
    ok = ok and got4
    print("  ON never actually suppresses a real OFF marker -> UNDEFINED (vacuous) ->",
          "ok" if got4 else "FAIL (%r)" % r4["status"])
    # FAILING DIRECTION: attribution mismatch -- claims "abstention" but the turn's own field says not abstained
    bad_attr_on = dict(base_on)
    bad_attr_on["mt_emo1"] = turn(abstained=False, cg={"suppressed": True, "reason": "abstention"})
    r5 = score_wiring({s: {"unset": {"turns": base_unset}, "off": {"turns": base_off}, "on": {"turns": bad_attr_on}}
                       for s in VERIFY_SEEDS}, VERIFY_SEEDS)
    got5 = r5["status"] == "NO-GO" and bool(r5["attribution_problems"])
    ok = ok and got5
    print("  suppression claims abstention but abstained=False -> attribution mismatch -> NO-GO ->",
          "ok" if got5 else "FAIL (%r)" % r5["status"])
    # FAILING DIRECTION: fewer than 6 valid seeds -> UNDEFINED
    partial = {s: {"unset": {"turns": base_unset}, "off": {"turns": base_off}, "on": {"turns": base_on}}
              for s in VERIFY_SEEDS[:5]}
    partial[VERIFY_SEEDS[5]] = {"unset": None, "off": None, "on": None}
    r6 = score_wiring(partial, VERIFY_SEEDS)
    got6 = r6["status"] == "UNDEFINED" and r6["n_valid"] == 5
    ok = ok and got6
    print("  5/6 seeds valid -> UNDEFINED ->", "ok" if got6 else "FAIL (%r)" % r6["status"])
    return ok


# ─────────────────────────────────────────────────────── selftest ──────────────────────────────────────────────
def _r(lead="", abstained=False, vsign=None, answer="hi", level=2):
    return {"answer": (lead + answer) if lead else answer, "abstained": abstained,
            "affect_drives": {"lead": lead, "level": level},
            "affect": ({"valence_sign": vsign} if vsign is not None else {})}


def _selftest_policy() -> bool:
    ok = True
    # (1) abstention conflict: marker present + abstained -> incongruent OFF, suppressed ON
    r = _r(lead="Gladly! ", abstained=True)
    off, doff = apply_policy(r, enabled=False)
    on, _don = apply_policy(r, enabled=True)
    got = doff["incongruent"] and doff["abstention_conflict"] and off["affect_drives"]["lead"] == "Gladly! " \
        and on["affect_drives"]["lead"] == "" and on["answer"] == "hi" and diagnose(on)["incongruent"] is False
    ok = ok and got
    print("  abstention conflict suppressed ON, flagged OFF ->", "ok" if got else "FAIL")
    # (2) valence conflict: positive marker but Gate-B says negative -> incongruent, suppressed ON
    r2 = _r(lead="Wonderful! ", abstained=False, vsign="-")
    _, d2 = apply_policy(r2, enabled=False)
    on2, don2 = apply_policy(r2, enabled=True)
    got2 = d2["incongruent"] and d2["valence_conflict"] and on2["affect_drives"]["lead"] == ""
    ok = ok and got2
    print("  valence conflict suppressed ON, flagged OFF ->", "ok" if got2 else "FAIL")
    # (3) congruent: marker present, not abstained, signs agree -> UNCHANGED on ON (no over-suppression)
    r3 = _r(lead="Gladly! ", abstained=False, vsign="+")
    off3, d3 = apply_policy(r3, enabled=False)
    on3, don3 = apply_policy(r3, enabled=True)
    got3 = (not d3["incongruent"]) and on3["affect_drives"]["lead"] == "Gladly! " and on3["answer"] == off3["answer"]
    ok = ok and got3
    print("  congruent case UNCHANGED by the policy (no over-suppression) ->", "ok" if got3 else "FAIL")
    # (4) congruent: no lead at all -> vacuously congruent, untouched either way
    r4 = _r(lead="", abstained=True, vsign="-")
    _, d4 = apply_policy(r4, enabled=False)
    got4 = not d4["incongruent"]
    ok = ok and got4
    print("  no-lead turn vacuously congruent regardless of abstain/sign ->", "ok" if got4 else "FAIL")
    # (5) missing Gate-B sign (affect off) -> no valence_conflict can be raised (never fabricate a disagreement)
    r5 = _r(lead="Frankly — ", abstained=False, vsign=None)
    _, d5 = apply_policy(r5, enabled=False)
    got5 = (not d5["valence_conflict"]) and (not d5["incongruent"])
    ok = ok and got5
    print("  missing independent sign read -> no valence_conflict fabricated ->", "ok" if got5 else "FAIL")
    # (6) default OFF (env unset) is a byte-identical passthrough even on an incongruent turn
    os.environ.pop(CONGRUENCE_ENV, None)
    r6 = _r(lead="Gladly! ", abstained=True)
    out6, _ = apply_policy(r6)          # enabled=None -> reads the (unset) env -> OFF
    got6 = out6 == dict(r6)
    ok = ok and got6
    print("  default-OFF (env unset) is byte-identical passthrough ->", "ok" if got6 else "FAIL")
    # (7) aggregate scorer: OFF has 2 incongruent turns, ON removes them, no over-suppression, attribution=1.0
    seed_turns = {42: {"a": _r(lead="Gladly! ", abstained=True), "b": _r(lead="Gladly! ", abstained=False, vsign="+"),
                       "c": _r(lead="Wonderful! ", abstained=False, vsign="-")}}
    rec = score_all(seed_turns, seeds=(42,))
    got7 = (rec["n_off_incongruent"] == 2 and rec["n_on_incongruent"] == 0 and rec["over_suppression_clean"]
            and rec["fraction_incongruence_removed_by_policy"] == 1.0)
    ok = ok and got7
    print("  aggregate scorer: OFF=2 incongruent, ON=0, no over-suppression, attribution=1.0 ->", "ok" if got7 else "FAIL")
    # (8) FAILING DIRECTION: an over-suppression bug (policy blanks an ALREADY-congruent lead) must be CAUGHT
    bad_turns = {42: {"a": _r(lead="Gladly! ", abstained=False, vsign="+")}}
    scored = score_turns(bad_turns[42])
    # simulate a buggy policy result by hand: on_lead differs from off_lead though off was congruent
    scored["a"]["on_lead"] = ""  # tamper, as a broken policy would produce
    scored["a"]["no_over_suppression_ok"] = bool(scored["a"]["off_lead"] == scored["a"]["on_lead"]
                                                 or not scored["a"]["off_diag"]["congruent"])
    got8 = scored["a"]["no_over_suppression_ok"] is False
    ok = ok and got8
    print("  tampered over-suppression is CAUGHT by no_over_suppression_ok ->", "ok" if got8 else "FAIL")
    # (9) FAILING DIRECTION (2026-09-24 silent-skip fix): a score missing seeds -- or missing a turn on one seed --
    # must read UNDEFINED, never "rule holds" GO, even when every turn it DID see is clean.
    labs = ("t1", "t2")
    clean = {"t1": _r(lead="Gladly! ", abstained=True), "t2": _r(lead="", abstained=True)}
    three = {s: clean for s in (42, 43, 44)}
    r9 = score_all(three, seeds=(42, 43, 44), expected_turns=labs)
    six_missing_turn = {s: clean for s in VERIFY_SEEDS}
    six_missing_turn[101] = {"t1": clean["t1"]}
    r9b = score_all(six_missing_turn, seeds=VERIFY_SEEDS, expected_turns=labs)
    got9 = (r9["status"] == "UNDEFINED" and not r9["go"] and r9b["status"] == "UNDEFINED" and not r9b["go"])
    ok = ok and got9
    print("  missing seeds / a missing turn -> UNDEFINED (got %s, %s) ->" % (r9["status"], r9b["status"]),
          "ok" if got9 else "FAIL")
    # (10) A2.1: complete 6-seed set, policy removes every incongruence, nothing over-suppressed -> rule_holds, but
    # the policy was applied POST-HOC (every mode here) -> status UNDEFINED, never GO (was GO before A2).
    r10 = score_all({s: clean for s in VERIFY_SEEDS}, seeds=VERIFY_SEEDS, expected_turns=labs)
    got10 = (r10["status"] == "UNDEFINED" and not r10["go"] and r10["rule_holds"] and r10["n_off_incongruent"] == 6
             and r10["n_on_leads_after_policy"] == 0)
    ok = ok and got10
    print("  complete set, rule holds, but post-hoc -> UNDEFINED (got %s) ->" % r10["status"], "ok" if got10 else "FAIL")
    # (10b) the same, applied BY production (post_hoc=False) with Gate-B reading both signs -> GO
    both = {"t1": _r(lead="Gladly! ", abstained=True, vsign="+"), "t2": _r(lead="", abstained=True, vsign="-")}
    r10b = score_all({s: both for s in VERIFY_SEEDS}, seeds=VERIFY_SEEDS, expected_turns=labs, post_hoc=False)
    got10b = r10b["status"] == "GO" and r10b["valence_half"] == "EXERCISED"
    ok = ok and got10b
    print("  production-applied, both Gate-B signs seen, rule holds -> GO (got %s) ->" % r10b["status"],
          "ok" if got10b else "FAIL")
    # (10c) A2.2: production-applied but Gate-B never reads '-' -> the valence half is UNTESTED -> UNDEFINED
    plus_only = {"t1": _r(lead="Gladly! ", abstained=True, vsign="+"), "t2": _r(lead="", abstained=True, vsign="0")}
    r10c = score_all({s: plus_only for s in VERIFY_SEEDS}, seeds=VERIFY_SEEDS, expected_turns=labs, post_hoc=False)
    got10c = r10c["status"] == "UNDEFINED" and r10c["valence_half"] == "UNTESTED"
    ok = ok and got10c
    print("  Gate-B never reads '-' -> valence half UNTESTED -> UNDEFINED (got %s) ->" % r10c["status"],
          "ok" if got10c else "FAIL")
    # (12) A2.5 settle_contrast: ON adds one marker on an abstention -> as-shipped delta +1; the policy erases both
    off_t = {"t1": _r(lead="Gladly! ", abstained=True), "t2": _r(lead="", abstained=True)}
    on_t = {"t1": _r(lead="Gladly! ", abstained=True), "t2": _r(lead="Wonderful! ", abstained=True)}
    sc = settle_contrast({s: off_t for s in VERIFY_SEEDS}, {s: on_t for s in VERIFY_SEEDS}, labs)
    got12 = (sc["complete"] and sc["settle_delta_as_shipped"]["abstention_attached"] == 6
             and sc["settle_delta_if_policy_shipped"] == 0 and sc["per_affective_turn"]["on"]["markers"] == "12/12")
    sc_missing = settle_contrast({42: off_t}, {42: on_t}, labs)
    got12 = got12 and not sc_missing["complete"]
    ok = ok and got12
    print("  settle_contrast: +6 abstention-attached markers as shipped, 0 if the policy shipped; missing seeds "
          "-> incomplete ->", "ok" if got12 else "FAIL")
    # (11) SURFACE: a lead that sits AFTER a prepended clause must leave the reply TEXT, not just the field; and a
    # policy that only blanks the field (the pre-fix behaviour) must be CAUGHT by the surface re-check.
    emb = {"answer": "That's thrilling for This -- Gladly! I don't know about that.", "abstained": True,
           "affect_drives": {"lead": "Gladly! "}, "affect": {"valence_sign": "+"}}
    on11, _ = apply_policy(emb, enabled=True)
    got11a = "Gladly!" not in on11["answer"] and score_turns({"x": emb})["x"]["on_recheck_incongruent"] is False
    # run score_turns against the PRE-FIX policy (blank the field, prefix-only text strip) swapped in for real
    real_policy = globals()["apply_policy"]

    def _prefix_only_policy(resp, *, enabled=None):
        d = diagnose(resp)
        if not congruence_enabled(enabled) or not d["incongruent"]:
            return dict(resp), d
        o = dict(resp)
        ans, ld = str(o.get("answer", "") or ""), d["lead"]
        o["answer"] = ans[len(ld):] if ld and ans.startswith(ld) else ans
        o["affect_drives"] = dict(o.get("affect_drives") or {}, lead="")
        return o, d
    globals()["apply_policy"] = _prefix_only_policy
    try:
        row_bad = score_turns({"x": emb})["x"]
    finally:
        globals()["apply_policy"] = real_policy
    got11b = row_bad["on_recheck_incongruent"] is True and row_bad["on_surface_residual"] is True
    ok = ok and got11a and got11b
    print("  embedded lead removed from the reply text; field-only suppression caught by the surface re-check ->",
          "ok" if (got11a and got11b) else "FAIL")
    return ok


def selftest() -> bool:
    ok = _selftest_policy()
    ok2 = _selftest_wiring()
    print("SELFTEST", "PASS" if (ok and ok2) else "FAIL")
    return bool(ok and ok2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score-multiturn", action="store_true")
    ap.add_argument("--score-settle-contrast", action="store_true",
                    help="A2.5: off_a vs on_a as shipped, per affective turn (descriptive, no verdict)")
    ap.add_argument("--worker-wiring", action="store_true", help=argparse.SUPPRESS)   # subprocess entry only
    ap.add_argument("--run-wiring", action="store_true",
                    help="Amendment 2: spawn real production brain_chat turns (unset/off/on) per seed")
    ap.add_argument("--score-wiring", action="store_true",
                    help="Amendment 2: score the --run-wiring output (GO/NO-GO/UNDEFINED)")
    ap.add_argument("--env", default="{}", help="--worker-wiring only: JSON env overrides for the subprocess")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--settle", type=int, default=1, choices=(0, 1))
    ap.add_argument("--seeds", default=" ".join(str(s) for s in VERIFY_SEEDS))
    ap.add_argument("--out-dir", default="research/findings/raw/_affect_marker_settle_congruence")
    ap.add_argument("--raw-dir", default="research/findings/raw/_affect_marker_settle_multiturn")
    ap.add_argument("--arm", default="on_a", help="which multiturn arm file to score (on_a/off_a/...)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.worker_wiring:
        sys.exit(_worker_wiring(a.env, a.out or "wiring_worker_out.json"))
    seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
    if a.run_wiring:
        per_seed = {}
        for s in seeds:
            per_seed[s] = run_seed_wiring(s, a.out_dir)
        rec = score_wiring(per_seed, seeds)
        out = a.out or os.path.join(a.out_dir, "wiring_arms_manifest.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump({"seeds": list(seeds), "out_dir": a.out_dir}, f, indent=1, default=str)
        print("run-wiring: spawned unset/off/on for %d seed(s) under %s (score with --score-wiring)"
              % (len(seeds), a.out_dir))
        return
    if a.score_wiring:
        per_seed = {}
        for s in seeds:
            sdir = os.path.join(a.raw_dir, "s%d" % s)
            row = {}
            for name in WIRING_ARMS:
                p = os.path.join(sdir, "%s.json" % name)
                row[name] = json.load(open(p)) if os.path.exists(p) else None
            per_seed[s] = row
        rec = score_wiring(per_seed, seeds)
        out = a.out or os.path.join(a.raw_dir, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("byte_identical_off=%s valence_half=%s non_vacuous=%s attribution_problems=%d -> status=%s -> %s"
              % (rec["byte_identical_off"], rec["valence_half"], rec["non_vacuous"],
                 len(rec["attribution_problems"]), rec["status"], out))
        return
    if a.run:
        per_seed = {}
        for s in seeds:
            turns = run_seed(s, bool(a.settle), a.out_dir)
            per_seed[s] = turns
            print("  s%d settle=%d -> %s" % (s, a.settle, "OK" if turns else "FAILED"), flush=True)
        rec = score_all(per_seed, seeds, expected_turns=RUN_TURN_LABELS)
        out = a.out or os.path.join(a.out_dir, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("OFF incongruent %d/%d (%.3f)  ON incongruent %d/%d (%.3f)  over_suppression_clean=%s -> %s"
              % (rec["n_off_incongruent"], rec["n_total"] - rec["n_errors"], rec["off_incongruent_rate"] or -1,
                 rec["n_on_incongruent"], rec["n_total"] - rec["n_errors"], rec["on_incongruent_rate"] or -1,
                 rec["over_suppression_clean"], out))
        return
    if a.score_settle_contrast:
        from research.runners._affect_marker_settle_multiturn_derisk import MT_LABELS

        def load(arm):
            got = {}
            for s in seeds:
                p = os.path.join(a.raw_dir, "s%d" % s, "%s.json" % arm)
                if os.path.exists(p):
                    with open(p) as f:
                        got[s] = json.load(f).get("turns", {})
            return got
        rec = settle_contrast(load("off_a"), load("on_a"), MT_LABELS)
        out = a.out or os.path.join("research/findings/raw/_affect_marker_settle_congruence", "settle_contrast.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("complete=%s  per affective turn: OFF %s  ON %s  | SETTLE delta as shipped %s | if policy shipped %+d -> %s"
              % (rec["complete"], rec["per_affective_turn"]["off"], rec["per_affective_turn"]["on"],
                 rec["settle_delta_as_shipped"], rec["settle_delta_if_policy_shipped"], out))
        return
    if a.score_multiturn:
        per_seed = {}
        for s in seeds:
            p = os.path.join(a.raw_dir, "s%d" % s, "%s.json" % a.arm)
            if os.path.exists(p):
                rec = json.load(open(p))
                per_seed[s] = rec.get("turns", {})
            else:
                per_seed[s] = {}
        from research.runners._affect_marker_settle_multiturn_derisk import MT_LABELS
        rec = score_all(per_seed, seeds, expected_turns=MT_LABELS)
        out = a.out or os.path.join("research/findings/raw/_affect_marker_settle_congruence",
                                    "multiturn_%s.json" % a.arm)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("OFF incongruent %d/%d (%s)  ON incongruent %d/%d (%s)  over_suppression_clean=%s  markers reaching "
              "the reply OFF=%d ON=%d  RULE status=%s -> %s"
              % (rec["n_off_incongruent"], rec["n_total"] - rec["n_errors"], rec["off_incongruent_rate"],
                 rec["n_on_incongruent"], rec["n_total"] - rec["n_errors"], rec["on_incongruent_rate"],
                 rec["over_suppression_clean"], rec["n_off_leads"], rec["n_on_leads_after_policy"], rec["status"], out))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
