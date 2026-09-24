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
    out["answer"] = answer[len(lead):] if lead and answer.startswith(lead) else answer
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
        # NO OVER-SUPPRESSION: a turn OFF already congruent must have an IDENTICAL lead ON.
        unchanged_when_congruent = bool(off_diag["congruent"] and off_resp.get("affect_drives", {}).get("lead")
                                        == on_resp.get("affect_drives", {}).get("lead"))
        per[label] = {"off_diag": off_diag, "on_diag": on_diag, "on_recheck_incongruent": on_recheck["incongruent"],
                     "off_lead": off_diag["lead"], "on_lead": on_diag["lead"] if not on_diag["incongruent"] else "",
                     "no_over_suppression_ok": bool((not off_diag["congruent"]) or unchanged_when_congruent)}
    return per


def score_all(per_seed_turns: dict, seeds=VERIFY_SEEDS) -> dict:
    from tools.lab import attributable_to
    n_total = n_off_incongruent = n_on_incongruent = n_over_suppress_violations = n_errors = 0
    detail = []
    for s in seeds:
        turns = per_seed_turns.get(s) or {}
        scored = score_turns(turns)
        for label, row in scored.items():
            n_total += 1
            if row.get("error"):
                n_errors += 1
                continue
            if row["off_diag"]["incongruent"]:
                n_off_incongruent += 1
            if row["on_recheck_incongruent"]:
                n_on_incongruent += 1
            if not row["no_over_suppression_ok"]:
                n_over_suppress_violations += 1
            detail.append({"seed": s, "turn": label, **row})
    denom = max(n_total - n_errors, 0)
    off_rate = (n_off_incongruent / denom) if denom else None
    on_rate = (n_on_incongruent / denom) if denom else None
    attribution = None
    if n_off_incongruent > 0:
        attribution = attributable_to("congruence policy: incongruent-turn count OFF vs ON",
                                      float(n_off_incongruent), float(n_on_incongruent))
    return {"probe": "affect_marker_settle_congruence", "seeds": list(seeds), "n_total": n_total,
            "n_errors": n_errors, "n_off_incongruent": n_off_incongruent, "n_on_incongruent": n_on_incongruent,
            "off_incongruent_rate": off_rate, "on_incongruent_rate": on_rate,
            "off_agreement_rate": (1.0 - off_rate) if off_rate is not None else None,
            "on_agreement_rate": (1.0 - on_rate) if on_rate is not None else None,
            "n_over_suppression_violations": n_over_suppress_violations,
            "over_suppression_clean": n_over_suppress_violations == 0,
            "fraction_incongruence_removed_by_policy": attribution, "detail": detail}


# ─────────────────────────────────────────────────────── selftest ──────────────────────────────────────────────
def _r(lead="", abstained=False, vsign=None, answer="hi"):
    return {"answer": (lead + answer) if lead else answer, "abstained": abstained,
            "affect_drives": {"lead": lead}, "affect": ({"valence_sign": vsign} if vsign is not None else {})}


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
    return ok


def selftest() -> bool:
    ok = _selftest_policy()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--score-multiturn", action="store_true")
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
    seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
    if a.run:
        per_seed = {}
        for s in seeds:
            turns = run_seed(s, bool(a.settle), a.out_dir)
            per_seed[s] = turns
            print("  s%d settle=%d -> %s" % (s, a.settle, "OK" if turns else "FAILED"), flush=True)
        rec = score_all(per_seed, seeds)
        out = a.out or os.path.join(a.out_dir, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("OFF incongruent %d/%d (%.3f)  ON incongruent %d/%d (%.3f)  over_suppression_clean=%s -> %s"
              % (rec["n_off_incongruent"], rec["n_total"] - rec["n_errors"], rec["off_incongruent_rate"] or -1,
                 rec["n_on_incongruent"], rec["n_total"] - rec["n_errors"], rec["on_incongruent_rate"] or -1,
                 rec["over_suppression_clean"], out))
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
        rec = score_all(per_seed, seeds)
        out = a.out or os.path.join("research/findings/raw/_affect_marker_settle_congruence",
                                    "multiturn_%s.json" % a.arm)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("OFF incongruent %d/%d (%s)  ON incongruent %d/%d (%s)  over_suppression_clean=%s -> %s"
              % (rec["n_off_incongruent"], rec["n_total"] - rec["n_errors"], rec["off_incongruent_rate"],
                 rec["n_on_incongruent"], rec["n_total"] - rec["n_errors"], rec["on_incongruent_rate"],
                 rec["over_suppression_clean"], out))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
