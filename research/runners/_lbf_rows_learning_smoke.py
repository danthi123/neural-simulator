"""Seed-7 DEV smoke for the d5-consolidate / sleep-replay load-bearing rows (A2, 2026-09-24 midnight plan S13).

Pre-registration: research/findings/2026-09-24-learning-rows-d5-consolidate-sleep-replay-lbf-PREREGISTRATION.md

Merges research/runners/lbf_rows/learning.py's EXTRA_LESIONS/EXTRA_PROBES into the live
load_bearing_fraction.FACULTY_LESIONS / onebrain_regression_battery.FACULTY_PROBES **in this process only**
(never edits either source file -- a stand-in for AG-REG's not-yet-landed import hook, S08), then runs the real
load_bearing_fraction measurement over both new rows through the actual webapp.server.brain_chat path, plus a
diagnostic "held-through-the-tick" check and a knob-off (byte-identity) probe.

Seed 7 is a DEV/calibration seed (this lane's scope: seed 7 for smokes/calibration only, never 42/43/44/100/101/
102). No GO/NO-GO capability claim is made by this script; it only proves the rows are wired and reachable.

Run (memcap-wrapped, single process at a time):
  bash tools/mem_ok.sh 10 4 && OMP_NUM_THREADS=1 bash tools/memcap.sh 10 -- \\
      .venv/bin/python -u -m research.runners._lbf_rows_learning_smoke --seed 7 \\
      --out research/findings/raw/_lbf_rows_learning/smoke_s7.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from tools.lab import lever


def _merge_rows():
    """In-process stand-in for AG-REG's not-yet-landed import hook. FACULTY_LESIONS is a module-global dict in
    load_bearing_fraction.py (updated in place); FACULTY_PROBES is a list object load_bearing_fraction.py
    imported BY REFERENCE from onebrain_regression_battery.py, so appending to either module's binding mutates
    the SAME list every other importer already holds. Idempotent (checks membership before appending)."""
    import research.runners.load_bearing_fraction as _lbf
    from research.runners.onebrain_regression_battery import FACULTY_PROBES
    from research.runners.lbf_rows import learning as _rows

    _lbf.FACULTY_LESIONS.update(_rows.EXTRA_LESIONS)
    existing = {row[0] for row in FACULTY_PROBES}
    for row in _rows.EXTRA_PROBES:
        if row[0] not in existing:
            FACULTY_PROBES.append(row)
    return _lbf, _rows


def _held_through_tick(lbf, key, out_dir, seed):
    """Diagnostic (NOT part of the registered row's own compare()): build the INTACT and LESION arms directly
    (bypassing measure_faculty's caching so the exact turn-by-turn trace is available here) and read the graded
    field immediately before vs after the tick, in EACH arm separately. A tick that silently did not run at all
    would leave BOTH arms' pre==post; the driving-turn comparison (intact CHANGED, lesion DID NOT) rules that out
    only if intact_changed is True here."""
    from research.runners.onebrain_regression_battery import _spawn_arm as _spawn_arm_raw, _get_path
    from research.runners.load_bearing_fraction import turn_group, _seed_suffix

    spec = lbf.FACULTY_LESIONS[key]
    row = lbf._faculty_row(key)
    grp = turn_group(row[1])
    sfx = _seed_suffix(seed)
    fname = key.replace("-", "_")
    intact = _spawn_arm_raw({}, grp, os.path.join(out_dir, "held_intact_%s%s.json" % (fname, sfx)))
    lesion = _spawn_arm_raw({spec["flag"]: spec["value"]}, grp,
                             os.path.join(out_dir, "held_lesion_%s%s.json" % (fname, sfx)))

    def _dh(resp, label):
        return _get_path((resp or {}).get(label) or {}, "episodic.graded_cue.depth_hold")[1]

    def _ticked(resp):
        for label in grp:
            t = (resp or {}).get(label) or {}
            if isinstance(t, dict) and "n_sessions_ticked" in t:
                return t.get("n_sessions_ticked")
        return None

    if key == "d5-consolidate":
        pre_label, post_label = "d5c_recall1", "d5c_recall2"
    else:  # sleep-replay: no same-topic pre-tick recall exists (the faculty is defined on the batch, not one topic)
        pre_label, post_label = None, "slp_recall"

    out = {
        "group": grp,
        "flag": spec["flag"], "lesion_value": spec["value"],
        "n_sessions_ticked_intact": _ticked(intact), "n_sessions_ticked_lesion": _ticked(lesion),
        "post_depth_hold_intact": _dh(intact, post_label), "post_depth_hold_lesion": _dh(lesion, post_label),
    }
    if pre_label is not None:
        pre_i, post_i = _dh(intact, pre_label), _dh(intact, post_label)
        pre_l, post_l = _dh(lesion, pre_label), _dh(lesion, post_label)
        # ATTRIBUTION (tools.lab.lever): the manipulation under test is the idle TICK, not the flag directly -- the
        # flag decides whether the tick's own re-activation loop fires. So the lever call is per-arm: the INTACT
        # arm's depth_hold MUST move across the tick (required=True -- if it doesn't, the whole probe is void, the
        # 2026-07-28 "two identical arms" failure mode); the LESION arm's depth_hold must NOT move (required=False
        # -- "UNCHANGED" here is the claimed finding, not an instrument failure, so it must not raise).
        lever("d5-consolidate depth_hold (INTACT, pre-tick -> post-tick)", pre_i, post_i, required=True)
        lever("d5-consolidate depth_hold (LESION, pre-tick -> post-tick)", pre_l, post_l, required=False)
        out.update({
            "pre_depth_hold_intact": pre_i, "pre_depth_hold_lesion": pre_l,
            "intact_recall1_answer": (intact or {}).get(pre_label, {}).get("answer"),
            "lesion_recall1_answer": (lesion or {}).get(pre_label, {}).get("answer"),
            "precondition_spared": (intact or {}).get(pre_label, {}).get("answer")
                == (lesion or {}).get(pre_label, {}).get("answer"),
            "intact_changed_pre_to_post": pre_i != post_i,
            "lesion_held_pre_to_post": pre_l == post_l,
        })
    else:
        # sleep-replay has no pre-tick recall of the SAME topic to lever against within one arm; the attribution
        # here is ACROSS arms instead (lever the post-tick reads intact-vs-lesion directly).
        lever("sleep-replay depth_hold (post-tick, INTACT vs LESION)",
              out["post_depth_hold_lesion"], out["post_depth_hold_intact"], required=True)
    out["intact_post_answer"] = (intact or {}).get(post_label, {}).get("answer")
    out["lesion_post_answer"] = (lesion or {}).get(post_label, {}).get("answer")
    out["reply_differs_intact_vs_lesion"] = out["intact_post_answer"] != out["lesion_post_answer"]
    return out


def _knob_off_probe(seed):
    """A stock 10-turn conversation over the DEFAULT PROBE_TURNS roster (unaffected by this branch's additions,
    which are new list entries kept OUT of PROBE_TURNS + a new module nothing default-imports), hashed. The
    static half of byte-identity (this branch's diff against origin/main touches only ADDED lines in
    onebrain_regression_battery.py plus two wholly new files) is checked by the caller via `git diff --stat`;
    this is the data half -- the same 10 turns run twice at this seed must hash identically to each other
    (this branch's own determinism), and this hash is recorded for a separate cross-revision diff against the
    identical command run on origin/main."""
    from research.runners.onebrain_regression_battery import PROBE_TURNS, _spawn_arm as _spawn_arm_raw
    labels = [t[0] for t in PROBE_TURNS[:10]]
    out_dir = "research/findings/raw/_lbf_rows_learning"
    os.makedirs(out_dir, exist_ok=True)
    resp_a = _spawn_arm_raw({}, labels, os.path.join(out_dir, "knob_off_a_s%d.json" % seed))
    resp_b = _spawn_arm_raw({}, labels, os.path.join(out_dir, "knob_off_b_s%d.json" % seed))
    ha = hashlib.sha256(json.dumps(resp_a, sort_keys=True, default=str).encode()).hexdigest()
    hb = hashlib.sha256(json.dumps(resp_b, sort_keys=True, default=str).encode()).hexdigest()
    return {"labels": labels, "sha256_a": ha, "sha256_b": hb, "self_reproducible": ha == hb}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="research/findings/raw/_lbf_rows_learning/smoke_s7.json")
    args = ap.parse_args()

    if args.seed in (42, 43, 44, 100, 101, 102):
        raise SystemExit("this smoke is a DEV/calibration probe; use a seed outside 42/43/44/100/101/102 "
                          "(the plan's own rule) -- the 6-seed gate is staged in the prereg, not run here.")

    os.environ["BRAIN_CHAT_SEED"] = str(args.seed)
    lbf, rows = _merge_rows()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    lbf_report = lbf.run(out_dir=out_dir, only=["d5-consolidate", "sleep-replay"], repeats=2, seed=args.seed)
    held = {k: _held_through_tick(lbf, k, out_dir, args.seed) for k in ("d5-consolidate", "sleep-replay")}
    knob_off = _knob_off_probe(args.seed)

    result = {"seed": args.seed, "lbf": lbf_report, "held_through_tick": held, "knob_off": knob_off}
    json.dump(result, open(args.out, "w"), indent=2, default=str)
    print("wrote", args.out)
    for k in ("d5-consolidate", "sleep-replay"):
        pf = next(p for p in lbf_report["per_faculty"] if p["faculty"] == k)
        print("  %-16s verdict=%-20s load_bearing=%s null_control_clean=%s diffs=%s"
              % (k, pf["verdict"], pf["load_bearing"], pf["null_control_clean"], pf["diffs"]))
        print("      held-through-tick:", {kk: vv for kk, vv in held[k].items() if kk != "group"})
    print("  knob-off self-reproducible:", knob_off["self_reproducible"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
