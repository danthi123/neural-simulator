#!/usr/bin/env python3
"""One BLOCKING command to sweep a runner across seeds and collapse the result to ONE verdict.

WHY THIS EXISTS (2026-08-06, two problems, one tool):

  (1) FRUGALITY. The plan-usage rule (feedback_minimize_plan_usage_via_nonclaude_machinery) is: Claude touches
      only the ENDPOINTS — launch an experiment, read its aggregate — and the middle (the per-seed sweep) runs
      on non-Claude compute. Before this, that pattern existed only as a bespoke per-lane aggregator
      (aggregate_source_monitor_seeds.py, hand-written each time). This generalises it: point it at any
      seed-parameterised runner and it self-sweeps + writes one earned verdict artifact. Launch once, read once.

  (2) ORPHANED SWEEPS. Sub-agents repeatedly wrote their own seed-loop runner, launched it in the BACKGROUND,
      and ended their turn "awaiting" a notification a terminated sub-agent never receives — orphaning the sweep
      (Stage-2c, Stage-2f, Stage-2g, each needing manual rescue). The ROOT CAUSE is structural and no tool an
      agent runs INSIDE ITS TURN can fix it: a sub-agent can background anything (including this tool), and a
      sub-agent — unlike the MAIN LOOP — is NOT reliably re-invoked by its own background job. So the sweep must
      not live in an agent's turn at all:

        - a lane AGENT builds the runner and runs `--smoke 1` (ONE seed, bounded, in-turn) to prove it runs,
          then RETURNS. It never runs the multiseed sweep.
        - the PARENT (main loop) runs the full multiseed sweep with THIS tool, launched as a harness-tracked
          background job. The main loop IS reliably re-invoked on completion, so nothing is orphaned, and the
          verdict artifact lands under parent control.

      That division — build in the agent, sweep in the parent — is what actually removes the failure mode.
      `--smoke N` exists for the agent half; the full run (no --smoke) is the parent half.

CONTRACT ON THE RUNNER. Each seed is a separate `python -m <runner> [extra] --<seed-flag> <S> --<out-flag> <tmp>`
invocation (defaults `--seed` / `--json`). The runner must write a JSON object to <tmp>. A seed PASSES when that
object's pass field is truthy — by default `verdict_status == "GO"`, else a truthy `pass`/`go`, overridable with
--pass-field. A runner whose per-seed exit code is nonzero AND wrote no JSON is recorded as an ERROR seed (not a
silent pass — the silent-failure rule).

    python -m tools.run_and_aggregate --runner research.runners.my_gate --seeds 652-654 \
        --extra "--phase development" --out research/findings/raw/my_gate/dev_verdict.json

Exit 0 => GO (all seeds pass). Exit 2 => NO-GO. Exit 1 => UNDEFINED (a seed errored / produced no readable JSON).
Reserving exit 1 for undefined-evidence, not negative science, matches the runners' own convention.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def seed_passes(obj: dict, pass_field: str | None) -> bool:
    """Truthy => this seed passed. Explicit field wins; else the runners' verdict convention."""
    if pass_field:
        return bool(obj.get(pass_field))
    if "verdict_status" in obj:
        return obj.get("verdict_status") == "GO"
    for k in ("go", "pass", "all_pass"):
        if k in obj:
            return bool(obj[k])
    raise KeyError(
        "no pass field in seed output: expected verdict_status/go/pass/all_pass, or pass --pass-field NAME")


def aggregate(rows: list[dict]) -> dict:
    """Collapse per-seed rows to ONE earned verdict. Pure function — this is the testable core.

    GO iff every seed produced a readable result AND passed. Any errored seed => UNDEFINED (never a silent
    GO/NO-GO over incomplete evidence). No seeds => UNDEFINED.
    """
    if not rows:
        return {"outcome": "UNDEFINED", "reason": "no seeds", "n_seeds": 0, "n_pass": 0, "rows": rows}
    errored = [r for r in rows if r.get("error")]
    if errored:
        return {"outcome": "UNDEFINED", "reason": "%d seed(s) errored: %s" % (
            len(errored), ", ".join(str(r["seed"]) for r in errored)),
            "n_seeds": len(rows), "n_pass": sum(1 for r in rows if r.get("pass")), "rows": rows}
    n_pass = sum(1 for r in rows if r.get("pass"))
    go = n_pass == len(rows)
    return {"outcome": "GO" if go else "NO-GO", "go": go,
            "n_seeds": len(rows), "n_pass": n_pass, "rows": rows}


def run_seed(runner: str, seed: int, extra: list[str], seed_flag: str, out_flag: str,
             pass_field: str | None, env: dict, timeout_s: int) -> dict:
    """Run ONE seed to completion (blocking) and return its pass/fail row."""
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as tf:
        tmp = tf.name
    try:
        cmd = [sys.executable, "-m", runner, *extra, "--%s" % seed_flag, str(seed), "--%s" % out_flag, tmp]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout_s)
        obj = None
        try:
            with open(tmp) as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError):
            obj = None
        if obj is None:
            return {"seed": seed, "error": "no readable JSON (exit %d): %s"
                    % (proc.returncode, (proc.stderr or proc.stdout or "")[-300:].strip())}
        try:
            passed = seed_passes(obj, pass_field)
        except KeyError as e:
            return {"seed": seed, "error": str(e)}
        return {"seed": seed, "pass": bool(passed),
                "verdict_status": obj.get("verdict_status"), "artifact": obj.get("out") or tmp}
    except subprocess.TimeoutExpired:
        return {"seed": seed, "error": "timeout after %ds" % timeout_s}
    finally:
        # keep the tmp only if the runner did not record its own --out; harmless either way
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Blocking seed-sweep + aggregate to one verdict.")
    ap.add_argument("--runner", required=True, help="module path, e.g. research.runners.my_gate")
    ap.add_argument("--seeds", required=True, help="e.g. 652-654 or 42,43,44")
    ap.add_argument("--out", required=True, help="aggregate verdict JSON path")
    ap.add_argument("--extra", default="", help="extra args passed to every seed invocation (one string)")
    ap.add_argument("--seed-flag", default="seed")
    ap.add_argument("--out-flag", default="json")
    ap.add_argument("--pass-field", default=None, help="override the pass field (default: verdict_status==GO)")
    ap.add_argument("--backend", default=None, help="sets SIM_BACKEND for every seed (numpy/cupy)")
    ap.add_argument("--per-seed-timeout", type=int, default=1800)
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="AGENT half: run only the first N seeds as a bounded in-turn CHECK. The result is "
                         "marked SMOKE and can NEVER be read as a verdict — the full sweep is the parent's job.")
    args = ap.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    if args.smoke:
        seeds = seeds[:args.smoke]
    extra = args.extra.split() if args.extra else []
    env = dict(os.environ)
    if args.backend:
        env["SIM_BACKEND"] = args.backend

    started = time.perf_counter()
    rows = []
    for i, s in enumerate(seeds, 1):
        print("  [%d/%d] seed %d ..." % (i, len(seeds), s), flush=True)
        row = run_seed(args.runner, s, extra, args.seed_flag, args.out_flag,
                       args.pass_field, env, args.per_seed_timeout)
        print("      -> %s" % ({k: row[k] for k in row if k != "artifact"}), flush=True)
        rows.append(row)

    verdict = aggregate(rows)
    if args.smoke:
        # A SMOKE is a build-check, never a verdict. Relabel so no finding can quote it as GO/NO-GO.
        verdict["smoke_outcome"] = verdict.get("outcome")
        verdict["outcome"] = "SMOKE"
        verdict["smoke_n"] = args.smoke
    verdict.update({"runner": args.runner, "seeds": seeds, "extra": extra,
                    "backend": args.backend, "elapsed_seconds": round(time.perf_counter() - started, 1)})
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps({k: verdict[k] for k in ("outcome", "smoke_outcome", "n_pass", "n_seeds") if k in verdict},
                     indent=2))
    print("  verdict -> %s" % out)
    if args.smoke:                       # smoke exit: 0 if it ran+passed (build works), 2 if a seed failed
        return 0 if verdict.get("smoke_outcome") == "GO" else 2
    return {"GO": 0, "NO-GO": 2}.get(verdict["outcome"], 1)


def _selfcheck() -> None:
    """FAILING DIRECTION FIRST: prove the aggregation cannot silently pass incomplete/failing evidence."""
    bad = []
    if aggregate([{"seed": 1, "pass": True}, {"seed": 2, "pass": True}])["outcome"] != "GO":
        bad.append("all-pass did not read as GO")
    if aggregate([{"seed": 1, "pass": True}, {"seed": 2, "pass": False}])["outcome"] != "NO-GO":
        bad.append("one failing seed did not read as NO-GO")
    if aggregate([{"seed": 1, "pass": True}, {"seed": 2, "error": "boom"}])["outcome"] != "UNDEFINED":
        bad.append("an ERRORED seed was NOT undefined — a silent pass over incomplete evidence")
    if aggregate([])["outcome"] != "UNDEFINED":
        bad.append("no seeds did not read as UNDEFINED")
    if not seed_passes({"verdict_status": "GO"}, None) or seed_passes({"verdict_status": "NO-GO"}, None):
        bad.append("verdict_status convention wrong")
    if not seed_passes({"strict": True}, "strict"):
        bad.append("explicit --pass-field not honoured")
    if parse_seeds("652-654,42") != [652, 653, 654, 42]:
        bad.append("seed spec parse wrong")
    if bad:
        raise SystemExit("⛔ run_and_aggregate selfcheck FAILED: " + "; ".join(bad))
    print("run_and_aggregate selfcheck PASSED: GO/NO-GO/UNDEFINED(errored)/UNDEFINED(empty), "
          "verdict_status + --pass-field conventions, seed-spec parse.")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        raise SystemExit(main())
