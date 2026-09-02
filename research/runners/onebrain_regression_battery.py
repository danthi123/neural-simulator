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
faculty whose decision fields are None/absent on the probe set (it needs a trigger this set does not supply — a
mismatch turn for surprise, a scalar-quantity turn for pragmatic, a 2-turn intention for prospective memory, a visual
percept for vision-identity, a between-turn tick for self-initiation) is reported as `not-exercised` (a THIN probe:
counted, honest, not claimed as covered). Extending a thin probe to a driving one is a mechanical follow-on. The battery
catches a flip that changes a faculty's DECIDED output on a turn the set already drives; it cannot catch a regression a
probe never reaches.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


# ── the probe turns (deterministic; each populates several faculties' decision fields) ───────────────────────────
# (label, message, session, reset). A shared-session pair (hold -> held) sets discourse/WM state for the held read.
PROBE_TURNS = [
    ("well",     "the wolf bites the apple", "well", True),     # comprehensible transitive: recall/affect/da/provenance
    ("question", "what does the wolf bite",  "q",    True),     # a question -> None comprehension
    ("unknown",  "what is the capital of france", "u", True),   # the no-confab MOAT -> abstain
    ("hold",     "the fox and the wolf walked in", "d", True),  # >=2 referents -> d6 multiref sets focus
    ("held",     "the wolf watches the owl", "d", False),       # same session: multiref/swap/anaphora on the held read
    ("scalar",   "some of the dogs ran",     "s",    True),     # a scalar-quantity turn -> pragmatic implicature
    ("open",     "what might a dog chase",   "o",    True),     # open-ended -> generation channel
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
    ("comprehension-learned-animacy-cue",  "well", ["comprehension.on"], True),
    ("comprehension-learned-verb-selects", "well", ["comprehension.on"], True),
    ("noncontradiction-gate",   "well",     ["noncontradiction.on", "noncontradiction.reject",
                                             "noncontradiction.recalled_yn", "noncontradiction.asserted_polarity"], False),
    ("affect-coloring",         "well",     ["affect.on", "affect.valence_sign", "affect.tone_token"], False),
    ("affect-drives-response",  "well",     ["affect_drives.on", "affect_drives.acted", "affect_drives.high_arousal",
                                             "affect_drives.reason"], False),
    ("affect-marker-spiking-wta", "well",   ["affect.valence_sign"], True),
    ("da-mode-drives-response", "well",     ["da_drives.on", "da_drives.acted", "da_drives.mode", "da_drives.reason"], False),
    ("da-gated-encoding",       "well",     ["da_encoding.on"], False),
    ("source-provenance-honesty", "well",   ["provenance.known", "provenance.label", "provenance.agrees_with_encoded",
                                             "provenance.encoded_as"], False),
    ("common-ground-drives",    "well",     ["common_ground_drives.on", "common_ground_drives.decision",
                                             "common_ground_drives.reason"], False),
    ("confidence-forthcomingness", "well",  ["affect.forthcomingness.forthcoming"], True),
    ("swap-drives-response",    "held",     ["swap_drives.on", "swap_drives.acted", "swap_drives.swapped",
                                             "swap_drives.reason"], False),
    ("anaphora-wm",             "held",     ["activity.roles"], False),
    ("wm-binding-advanced",     "held",     ["multiref.n_referents"], False),
    ("prospective-memory",      "well",     ["pmem.armed"], True),
    ("pragmatic-implicature",   "scalar",   ["pragmatic.implicature", "pragmatic.on"], True),
    ("surprise-monitor",        "well",     ["surprise.surprised", "surprise.on"], True),
    ("metacog-monitor",         "well",     ["metacog.confident", "metacog.on"], True),
    ("worldmodel-forward",      "well",     ["worldmodel.pred_sign", "worldmodel.on"], True),
    ("curiosity-followup",      "well",     ["curiosity.crave", "curiosity.on"], True),
    ("reconsolidation",         "well",     ["reconsolidation.revised", "reconsolidation.on"], True),
    ("episodic-memory",         "well",     ["episodic.stored", "episodic.on"], True),
    ("discourse-register",      "held",     ["discourse.event"], True),
    ("open-ended-generation",   "open",     ["hypothesis", "answer"], True),
    ("discourse-planner",       "well",     ["rich", "n_sentences"], True),
    ("gnw-deliberation",        "well",     ["activity.composer"], True),
    ("gnw-multistep-deliberation", "well",  ["activity.composer"], True),
    ("self-initiated-utterance", "well",    ["self_initiated"], True),
    ("vision-identity-spiking-hmax", "well", ["vision"], True),
    ("value-driven-choice",     "well",     ["value_choice"], True),
    ("bg-action-selection",     "well",     ["bg_select"], True),
    ("selective-attention-biased-competition", "held", ["activity.roles"], True),
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
        _, msg, session, reset = _TURN_BY_LABEL[label]
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                            renderer="stub", rich=False, reset=reset))
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
    shift can flip a borderline decision — which, if it happens, is a real finding the battery correctly surfaces.)"""
    out_dir = "research/findings/raw/_regression_battery"
    os.makedirs(out_dir, exist_ok=True)
    labels = probe_subset or ["well", "unknown", "hold", "held"]
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
