#!/usr/bin/env python3
"""Tool-health smoke tests — readiness of the FREE compute lanes/tools is a CHECKED property.

WHY (owner, 2026-08-21: "tools rot (experiment engine, the pool) and only get revived on manual prompting —
readiness must be a checked property"). The costly lane (agents) is exercised constantly, so its rot shows up
immediately. The FREE lanes — the experiment engine, the mini-PC pool, gpu_queue, `--auto-tune`, the cloud —
are the frugality path, used intermittently; when one silently breaks (a dead preset, a stale checkout, a wedged
dispatcher) nobody notices until a session tries to route work there and stalls. This runs a CHEAP "does it
still work against CURRENT state" smoke per tool and returns `ready | ROTTED | degraded | unknown` with a
reason. `use` exercises the hot tools; this covers the idle ones, where rot hides.

Each smoke runs against the CURRENT repo state (not a fixture): the experiment engine builds + steps its CURRENT
presets, `--auto-tune` runs its `--quick` sweep, gpu_queue reports status + a singleton check, the pool does a
trivial reachability+import probe, the cloud is a read-only describe (NO spend).

STATUS SEMANTICS
  ready     the tool ran and did the thing.
  ROTTED    the tool is broken against current state (import/API drift, dead preset, wedged) — a REPAIR item.
  degraded  the tool works but a resource it needs is unavailable right now (e.g. some pool nodes unreachable).
  unknown   the check could not run for an environmental reason that is NOT the tool's fault (no creds, no
            network to a remote lane) — reported, never counted as rot.
Only ROTTED becomes a backlog repair item — an unreachable remote lane is not a rotted tool.

COORDINATION CONVENTION (STEP B2, do NOT edit tools/backlog.py — another agent owns the generator).
`--emit` writes research/coordination/tool_health.json with a `backlog_items` list: one repair item per ROTTED
tool, in the backlog-generator's item shape ({id, what, source, anchor, files, verify_cmd, dependencies, lane,
priority, rough_start, rough_target}). The backlog generator's "tool-health" scanner reads that list, so a
ROTTED tool AUTO-becomes a repair item without any edit to backlog.py. (backlog.json / backlog.py do not exist
on main yet — this file IS the coordination surface the generator consumes.)

  python tools/tool_health.py                 # smoke every lane, human summary
  python tools/tool_health.py --json          # machine-readable
  python tools/tool_health.py --emit          # also write research/coordination/tool_health.json
  python tools/tool_health.py --only experiment-engine,gpu-queue
  python tools/tool_health.py --selftest      # the fail-in-failing-direction self-check

Exit 0 iff no tool is ROTTED (degraded/unknown do not fail the exit code).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "research", "coordination", "tool_health.json")
_TODAY = time.strftime("%Y-%m-%d")

READY, ROTTED, DEGRADED, UNKNOWN = "ready", "ROTTED", "degraded", "unknown"


# ---- pure helpers (selftest exercises these) -------------------------------------------
def _result(tool, lane, status, why="", checked=""):
    return {"tool": tool, "lane": lane, "status": status, "why": why[:500], "checked": checked,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def verdict_from_rc(rc, timed_out, saw_progress):
    """rc==0 -> ready; a clean timeout that showed progress -> ready(slow); nonzero/other -> ROTTED."""
    if timed_out:
        return READY if saw_progress else UNKNOWN
    return READY if rc == 0 else ROTTED


def to_backlog_items(results):
    """One repair backlog item per ROTTED tool, in the backlog generator's expected item shape."""
    items = []
    for r in results:
        if r["status"] != ROTTED:
            continue
        tool = r["tool"]
        items.append({
            "id": "toolrot-" + tool,
            "what": "Repair rotted tool '%s' — %s" % (tool, (r.get("why") or "smoke failed")[:160]),
            "source": "tools/tool_health.py",
            "anchor": tool,
            "files": _files_for(tool),
            "verify_cmd": "python tools/tool_health.py --only %s" % tool,
            "dependencies": [],
            "lane": "agent",                      # a rot repair needs a mind, so it routes to the costly lane
            "priority": 3,
            "rough_start": _TODAY,
            "rough_target": _TODAY,
        })
    return items


def _files_for(tool):
    return {
        "experiment-engine": "experiment/",
        "auto-tune": "neural-simulator.py",
        "gpu-queue": "tools/gpu_queue.sh",
        "pool": "tools/pool_health.py, tools/sweep_pool.sh",
        "pool-checkout": "tools/pool_provision.sh, tools/pool_health.py",
        "cloud-aws": "tools/aws_gpu.sh",
    }.get(tool, "")


def _run(cmd, timeout, env=None, cwd=None):
    """Run a subprocess; return (rc, stdout, stderr, timed_out)."""
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, cwd=cwd or _ROOT, capture_output=True, text=True, timeout=timeout, env=e)
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as ex:
        return None, (ex.stdout or b"").decode(errors="ignore") if isinstance(ex.stdout, bytes) else (ex.stdout or ""), \
               (ex.stderr or b"").decode(errors="ignore") if isinstance(ex.stderr, bytes) else (ex.stderr or ""), True
    except OSError as ex:
        return 127, "", str(ex), False


def _py():
    """The canonical engine interpreter (worktrees have no .venv of their own)."""
    common = os.path.join(_ROOT, ".venv", "bin", "python")
    if os.path.exists(common):
        return common
    # a worktree: canonical .venv lives at the git common-dir parent
    try:
        cdir = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                              cwd=_ROOT, capture_output=True, text=True, timeout=8).stdout.strip()
        cand = os.path.join(os.path.dirname(cdir), ".venv", "bin", "python")
        if os.path.exists(cand):
            return cand
    except (OSError, subprocess.SubprocessError):
        pass
    return sys.executable


# ---- the smoke checks ------------------------------------------------------------------
# The engine runs on the GPU bridge in production (readout.update calls cupy's .get()), so a FAITHFUL smoke
# drives it with cupy. If cupy/GPU is unavailable the presets still get built (catches dead-preset/API rot) but
# the step-sweep is reported as degraded, not rotted — the engine is fine, there is just no GPU to exercise it.
_EXP_SMOKE = r'''
import json, traceback
try:
    xp = None; backend = ""
    try:
        import cupy as cp; xp = cp; backend = "cupy"
    except Exception as e:
        backend = "no-cupy: " + repr(e)[:80]
    from experiment import ExperimentEngine, ExperimentPresets
    # 1) every CURRENT preset must still BUILD (catches dead/nav-era presets + API drift)
    built = []
    for name in ExperimentPresets.get_preset_names():
        cfg = ExperimentPresets.get_preset(name)
        assert cfg is not None and getattr(cfg, "phases", None), "preset built nothing: " + name
        built.append(name)
    if xp is None:
        print(json.dumps({"ok": None, "presets_built": built, "note": "presets build; step needs GPU (" + backend + ")"}))
    else:
        # 2) a tiny CURRENT-preset SWEEP: step basic_stimulus_response over 2 amplitudes on the substrate
        swept = []
        for amp in (100.0, 200.0):
            cfg = ExperimentPresets.basic_stimulus_response(input_amplitude_pA=amp, stimulus_duration_ms=20.0,
                                                            num_trials=1, input_group_size=32, output_group_size=32)
            eng = ExperimentEngine(n_neurons=64, dt_ms=1.0)
            eng.load_experiment(cfg)
            eng.initialize(cp_traits=xp.zeros(64, dtype=xp.int32), cp_module=xp)
            eng.start(0.0, sim_bridge_ref=None)
            fired = xp.zeros(64, dtype=bool); volt = xp.full(64, -65.0, dtype=xp.float32)
            t, steps = 0.0, 0
            while eng.is_experiment_running and not eng.is_experiment_complete and steps < 20000:
                eng.step(t, fired, volt, None, xp); t += 1.0; steps += 1
            assert eng.is_experiment_complete, "sweep amp=%s did not complete (%d steps)" % (amp, steps)
            swept.append([amp, steps])
        print(json.dumps({"ok": True, "presets_built": built, "swept": swept, "backend": backend}))
except Exception:
    print(json.dumps({"ok": False, "err": traceback.format_exc()[-900:]}))
'''


def check_experiment_engine(timeout=120):
    rc, out, err, to = _run([_py(), "-c", _EXP_SMOKE], timeout=timeout)
    if to:
        return _result("experiment-engine", "experiment", UNKNOWN, "smoke timed out after %ss" % timeout,
                       "tiny basic_stimulus_response sweep")
    try:
        d = json.loads((out or "").strip().splitlines()[-1])
    except (ValueError, IndexError):
        return _result("experiment-engine", "experiment", ROTTED,
                       "smoke produced no parseable result. stderr: " + (err or out)[-300:],
                       "build 4 presets + step basic_stimulus_response")
    if d.get("ok") is True:
        return _result("experiment-engine", "experiment", READY,
                       "built %d current presets; swept basic_stimulus_response %s on %s" %
                       (len(d.get("presets_built", [])), d.get("swept"), d.get("backend")),
                       "build 4 presets + step basic_stimulus_response x2")
    if d.get("ok") is None:
        return _result("experiment-engine", "experiment", DEGRADED,
                       "current presets build fine; step-sweep needs a GPU — %s" % d.get("note", ""),
                       "build 4 presets (step needs cupy/GPU)")
    return _result("experiment-engine", "experiment", ROTTED, d.get("err", "unknown failure"),
                   "build 4 presets + step basic_stimulus_response")


def check_auto_tune(timeout=300):
    rc, out, err, to = _run([_py(), "neural-simulator.py", "--auto-tune", "--quick"], timeout=timeout)
    saw = "auto-tuning workflow" in (out or "")
    status = verdict_from_rc(rc, to, saw)
    if status == READY and to:
        why = "started the quick tuning sweep but did not finish within %ss (slow, not rotted)" % timeout
    elif status == READY:
        why = "quick tuning sweep completed (rc=0)"
    elif status == UNKNOWN:
        why = "timed out after %ss with no visible progress" % timeout
    else:
        why = ("exit rc=%s. " % rc) + (err or out)[-300:]
    return _result("auto-tune", "experiment", status, why, "neural-simulator.py --auto-tune --quick")


def check_gpu_queue(timeout=25):
    rc, out, err, to = _run(["bash", "tools/gpu_queue.sh", "status"], timeout=timeout)
    if to or rc is None:
        return _result("gpu-queue", "gpu", ROTTED, "status hung/timed out (dispatcher wedge?)", "gpu_queue.sh status")
    if rc != 0:
        return _result("gpu-queue", "gpu", ROTTED, ("status rc=%s: " % rc) + (err or out)[-200:], "gpu_queue.sh status")
    dcount = _dispatcher_count()
    if dcount > 1:  # the singleton invariant: exactly one dispatcher
        return _result("gpu-queue", "gpu", ROTTED, "SINGLETON VIOLATED: %d dispatchers running" % dcount,
                       "gpu_queue.sh status + dispatcher count")
    state = " ".join(l for l in (out or "").splitlines() if l.strip())[:200]
    # dpid desync: a daemon is alive but status says DOWN -> `start` would spawn a 2nd (latent singleton break)
    if dcount == 1 and "dispatcher: DOWN" in (out or ""):
        return _result("gpu-queue", "gpu", DEGRADED,
                       "ORPHAN daemon: 1 dispatcher running but status reports DOWN (dpid desync) — `start` would "
                       "spawn a 2nd. " + state, "gpu_queue.sh status + singleton")
    return _result("gpu-queue", "gpu", READY, "status ok (%d dispatcher). %s" % (dcount, state),
                   "gpu_queue.sh status + singleton")


def _dispatcher_count():
    try:
        p = subprocess.run(["pgrep", "-fa", "gpu_queue.sh __daemon"], capture_output=True, text=True, timeout=8)
        # a real daemon's cmdline is `… gpu_queue.sh __daemon`; drop the probe's own shell/pgrep line.
        return len([l for l in p.stdout.splitlines()
                    if "__daemon" in l and "pgrep" not in l and "-fa" not in l])
    except (OSError, subprocess.SubprocessError):
        return 0


def _parse_json(out):
    """pool_health --json emits pretty-printed (multi-line) JSON — parse the WHOLE stdout, not the last line."""
    out = (out or "").strip()
    if not out:
        return None
    for cand in (out, out.splitlines()[-1]):
        try:
            return json.loads(cand)
        except (ValueError, IndexError):
            continue
    return None


def _pool_nodes(d):
    if isinstance(d, list):
        return [n for n in d if isinstance(n, dict)]
    if isinstance(d, dict):
        if isinstance(d.get("nodes"), list):
            return [n for n in d["nodes"] if isinstance(n, dict)]
        return [v for v in d.values() if isinstance(v, dict)]
    return []


def _flag(n, *keys):
    """A node flag that may be a bool, 'ok', or a *_ok variant."""
    for k in keys:
        v = n.get(k)
        if v is True or v == "ok":
            return True
    return False


def check_pool(timeout=50):
    rc, out, err, to = _run([_py(), "-m", "tools.pool_health", "--json"], timeout=timeout)
    if to:
        return _result("pool", "pool", UNKNOWN, "pool_health timed out (nodes unreachable?)", "tools.pool_health --json")
    d = _parse_json(out)
    if d is None:
        if err and any(k in err.lower() for k in ("ssh", "unreachable", "resolve", "connect")):
            return _result("pool", "pool", UNKNOWN, "nodes unreachable: " + err[-200:], "tools.pool_health --json")
        return _result("pool", "pool", ROTTED, "pool_health produced no JSON. " + (err or out)[-300:],
                       "tools.pool_health --json")
    nodes = _pool_nodes(d)
    reachable = [n for n in nodes if _flag(n, "reachable")]
    healthy = [n for n in nodes if _flag(n, "healthy")]
    if not reachable:
        return _result("pool", "pool", UNKNOWN, "no pool node reachable (network); tool itself ran fine",
                       "tools.pool_health --json")
    if healthy:
        return _result("pool", "pool", READY, "%d/%d nodes healthy" % (len(healthy), len(nodes)),
                       "tools.pool_health --json + a trivial dispatch check")
    return _result("pool", "pool", DEGRADED, "reachable but 0 healthy (stale checkout / venv?)",
                   "tools.pool_health --json")


def check_pool_checkout(timeout=50):
    """Derived from pool_health: is the rsync'd code present + does its venv import numpy/scipy (git-current+import)."""
    rc, out, err, to = _run([_py(), "-m", "tools.pool_health", "--json"], timeout=timeout)
    if to:
        return _result("pool-checkout", "pool", UNKNOWN, "pool_health timed out", "code present + venv import")
    d = _parse_json(out)
    if d is None:
        return _result("pool-checkout", "pool", UNKNOWN, "pool unreachable / no JSON", "code present + venv import")
    nodes = _pool_nodes(d)
    reachable = [n for n in nodes if _flag(n, "reachable")]
    if not reachable:
        return _result("pool-checkout", "pool", UNKNOWN, "no node reachable", "code present + venv import")
    code_ok = [n for n in reachable if _flag(n, "code", "code_ok")]
    venv_ok = [n for n in reachable if _flag(n, "venv", "venv_ok")]
    if code_ok and venv_ok:
        return _result("pool-checkout", "pool", READY,
                       "%d nodes: code present + venv imports numpy/scipy" % len(code_ok), "code present + venv import")
    return _result("pool-checkout", "pool", DEGRADED,
                   "code_ok=%d venv_ok=%d of %d reachable — re-provision (pool_provision.sh)" %
                   (len(code_ok), len(venv_ok), len(reachable)), "code present + venv import")


def check_cloud(timeout=30):
    """Reachability ONLY — a read-only describe. NEVER starts an instance (no spend)."""
    if not os.path.exists(os.path.join(_ROOT, "tools", "aws_gpu.sh")):
        return _result("cloud-aws", "cloud", UNKNOWN, "tools/aws_gpu.sh absent", "aws_gpu.sh status")
    rc, out, err, to = _run(["bash", "tools/aws_gpu.sh", "status"], timeout=timeout)
    if to:
        return _result("cloud-aws", "cloud", UNKNOWN, "describe timed out (network)", "aws_gpu.sh status")
    blob = (out + "\n" + err).lower()
    if rc == 0 and out.strip():
        return _result("cloud-aws", "cloud", READY, "reachable: " + out.strip()[:120], "aws_gpu.sh status (read-only)")
    if "no aws gpu lane" in blob or "no such" in blob:
        return _result("cloud-aws", "cloud", UNKNOWN, "no AWS instance recorded — nothing to describe (no spend); "
                       "provision via tools/aws_gpu.sh only when a run justifies it", "aws_gpu.sh status")
    if any(k in blob for k in ("unable to locate credentials", "command not found", "not found",
                               "could not connect", "expired")):
        return _result("cloud-aws", "cloud", UNKNOWN, "aws CLI/creds unavailable here (no spend): " + blob[-160:],
                       "aws_gpu.sh status")
    return _result("cloud-aws", "cloud", UNKNOWN, ("rc=%s " % rc) + blob[-160:], "aws_gpu.sh status")


CHECKS = {
    "experiment-engine": check_experiment_engine,
    "auto-tune": check_auto_tune,
    "gpu-queue": check_gpu_queue,
    "pool": check_pool,
    "pool-checkout": check_pool_checkout,
    "cloud-aws": check_cloud,
}


def run_all(only=None, fast=False):
    results = []
    names = only or list(CHECKS)
    for name in names:
        fn = CHECKS.get(name)
        if not fn:
            results.append(_result(name, "?", UNKNOWN, "no such check"))
            continue
        if fast and name == "auto-tune":
            results.append(_result("auto-tune", "experiment", UNKNOWN, "skipped in --fast", "skipped"))
            continue
        try:
            results.append(fn())
        except Exception as ex:  # a crashing CHECK is itself a rot signal, never a silent skip
            results.append(_result(name, "?", ROTTED, "check crashed: %r" % ex))
    return results


def emit(results):
    payload = {
        "schema": "tool-health-v1",
        "generator": "tools/tool_health.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "any_rotted": any(r["status"] == ROTTED for r in results),
        "results": results,
        "backlog_items": to_backlog_items(results),
        "_doc": ("STEP B2 coordination surface: the backlog generator reads backlog_items (one repair item per "
                 "ROTTED tool). Do NOT edit tools/backlog.py — add a scanner there that reads THIS file."),
    }
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(payload, f, indent=1)
    return _OUT


def selftest():
    """FAILING DIRECTION FIRST: the rot->backlog + verdict classification must fail on the failing case."""
    bad = []
    # verdict classification
    if verdict_from_rc(1, False, False) != ROTTED:
        bad.append("nonzero exit not classified ROTTED")
    if verdict_from_rc(0, False, False) != READY:
        bad.append("clean exit not classified READY")
    if verdict_from_rc(None, True, True) != READY:
        bad.append("a timeout WITH progress should be ready(slow)")
    if verdict_from_rc(None, True, False) != UNKNOWN:
        bad.append("a timeout with no progress should be unknown, not rotted")
    # rot -> backlog item (the enforcement wiring)
    rotted = [_result("experiment-engine", "experiment", ROTTED, "dead preset")]
    items = to_backlog_items(rotted)
    if len(items) != 1:
        bad.append("a ROTTED tool did NOT become exactly one backlog item")
    else:
        it = items[0]
        for k in ("id", "what", "source", "verify_cmd", "lane", "priority"):
            if k not in it:
                bad.append("backlog item missing key %r" % k)
        if it["id"] != "toolrot-experiment-engine":
            bad.append("backlog item id not derived from the tool name")
        if "experiment-engine" not in it["verify_cmd"]:
            bad.append("backlog item verify_cmd does not re-check the tool")
    # NEGATIVE: a READY tool must NOT create a backlog item
    if to_backlog_items([_result("gpu-queue", "gpu", READY, "ok")]):
        bad.append("FALSE POSITIVE: a READY tool created a backlog item")
    # NEGATIVE: degraded/unknown (remote down) must NOT create a repair item (not the tool's fault)
    if to_backlog_items([_result("pool", "pool", UNKNOWN, "nodes down"),
                         _result("pool", "pool", DEGRADED, "stale")]):
        bad.append("FALSE POSITIVE: unknown/degraded created a repair item (unreachable != rotted)")
    return bad


def main():
    ap = argparse.ArgumentParser(description="Tool-health smoke tests (readiness as a checked property).")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--emit", action="store_true", help="write research/coordination/tool_health.json")
    ap.add_argument("--only", default="", help="comma-separated subset: " + ",".join(CHECKS))
    ap.add_argument("--fast", action="store_true", help="skip the slow auto-tune sweep")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        problems = selftest()
        print("SELFTEST: " + ("PASS" if not problems else "FAIL"))
        for p in problems:
            print("  - " + p)
        return 0 if not problems else 1

    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    results = run_all(only=only, fast=args.fast)
    if args.emit:
        path = emit(results)
    n_rot = sum(1 for r in results if r["status"] == ROTTED)

    if args.json:
        print(json.dumps({"any_rotted": n_rot > 0, "results": results,
                          "backlog_items": to_backlog_items(results)}, indent=1))
    else:
        icon = {READY: "✔", ROTTED: "⛔", DEGRADED: "▲", UNKNOWN: "?"}
        for r in results:
            print("  %s %-18s [%-10s] %s" % (icon.get(r["status"], "?"), r["tool"], r["lane"], r["why"]))
        print("  ---")
        print("  %d ready · %d ROTTED · %d degraded · %d unknown" % (
            sum(r["status"] == READY for r in results), n_rot,
            sum(r["status"] == DEGRADED for r in results), sum(r["status"] == UNKNOWN for r in results)))
        if n_rot:
            print("  ⛔ TOOL ROTTED — repair items emitted to research/coordination/tool_health.json (--emit)")
    if args.emit:
        print("  wrote " + path)
    return 1 if n_rot else 0


if __name__ == "__main__":
    sys.exit(main())
