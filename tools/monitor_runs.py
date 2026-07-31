"""monitor_runs — a COVERAGE-COMPLETE status poller for long background runs (GPU sweeps, de-risks, training).

WHY THIS EXISTS (three real failures on 2026-07-16, all in one session):
  1. A `tail -F | grep "==> seed"` monitor stayed SILENT through a run that crash-looped 10023 times -- the filter
     matched only the happy path, so a dying run looked identical to a working one.
  2. Widening that filter to catch "Traceback" then spammed one event per line until the monitor was AUTO-STOPPED
     for excessive output -- so the run became UNwatched precisely when it was failing.
  3. A run silently no-op'd forever (swallowed init -> is_initialized False -> every step an early return). It
     logged NOTHING and never exited. No log-tailing monitor can ever detect that; only "the log stopped growing
     while the process is still alive" can.

THE FIX: do not tail. POLL STATE -- and use CPU-TIME DELTA, not log growth, as the liveness signal. Every tick each
run is classified into exactly one of five states. Note that SILENCE IS A STATE, not the absence of one -- but note
equally that silence is NOT automatically bad:

  RUNNING  proc alive + CPU rising + log fresh
  QUIET    proc alive + CPU RISING + log stale for --stall-min   -> INFORMATIONAL: working, just between log lines
  HUNG     proc alive + CPU FLAT across a tick                   -> the REAL alarm (deadlock / blocked / sleeping)
  DONE     proc gone  + a success marker present
  CRASHED  proc gone  + an error marker present, OR no success marker (an exit without a marker is a FAILURE,
           never a pass -- "it stopped" must never be read as "it finished")

WHY CPU-DELTA AND NOT LOG-STALENESS (learned immediately, the hard way): the first version used log growth alone and
CRIED WOLF within one tick -- a perfectly healthy e-prop arm pinned at 99.8% CPU logs only once per SEED (30-60 min),
so a 20-min log gap flagged STALLED while its utime was rising a full core per second. A FALSE alarm is a lie in the
other direction: it trains the reader to ignore the alerts, which is exactly how a real crash gets missed. Log
staleness cannot distinguish "computing hard but quiet" from "hung"; CPU-time delta can, so staleness is demoted to
an informational QUIET and only CPU-flat raises an alarm.

Emits ONE compact line per run, only when something a human would act on changes (state flips, or new progress),
plus a periodic heartbeat so a healthy long run still reports in. Exits when every run reaches a terminal state,
printing a final summary -> as a Monitor command it self-terminates instead of idling until timeout.

RUN<->PROC MAPPING: a shell redirect (`> x.log`) does not appear in the proc cmdline, so a log cannot be mapped to
its process directly. We map via the run's `--out` JSON, which IS in the cmdline, using the convention that the log
and the json share a stem (foo.log <-> foo.json). If no proc matches, staleness alone decides -- so the tool still
works for runs it did not launch.

USAGE (as a Monitor command; each stdout line becomes one notification):
    python tools/monitor_runs.py --logs "research/findings/raw/_eprop_*k8.log" --interval 300 --heartbeat 3
    python tools/monitor_runs.py --logs "a.log" "b.log" --progress-re "==> seed \\d+" --stall-min 20

  --interval    seconds between polls (default 300 = 5 min; a 2h GPU run does not need 30s polling)
  --heartbeat   emit a status line every N ticks even when nothing changed (default 3 => ~15 min at 300s)
  --stall-min   minutes of no log growth before a live, CPU-rising run is reported QUIET (default 20;
                raise it for runners that only log per-seed -- it is informational, not an alarm)
  --progress-re regex whose LAST match is shown as the run's progress (default: seed/epoch/step lines)
  --max-ticks   safety bound so the poller cannot outlive the session (default 288 = 24h at 300s)
"""
import argparse, glob, os, re, subprocess, sys, time

SUCCESS_RE = re.compile(
    r"wrote .*\.json|\[\w[\w-]*\]\s+\d+/\d+ GO|ALL DONE|=== DONE|SIGNAL|HONEST NEGATIVE|GO=(True|False)",
    re.I)
ERROR_RE = re.compile(
    r"Traceback \(most recent call last\)|^\s*\w*Error\b|ERROR --|MemoryError|CUDA_ERROR|out of memory|"
    r"Killed|Segmentation fault|AssertionError|could not be broadcast",
    re.M)
DEFAULT_PROGRESS_RE = r"==> seed \d+.*|\[seed \d+\].*|epoch\s+\d+.*|step\s+\d+.*|\d+/\d+ GO.*"


_SHELL_WRAPPER = re.compile(r"^\S*/?(zsh|bash|sh|fish)\s+-c\b")


def _procs():
    """cmdlines of live python WORKER procs.

    MUST exclude shell wrappers: a launcher runs as `zsh -c "... python -m runner --out foo.json ... & wait"`, so the
    SHELL's cmdline contains both "python" and the json. Counting it would keep a CRASHED run looking alive forever
    -> it would be reported STALLED instead of CRASHED, never reach a terminal state, and the poller would never
    exit. (Caught by testing the poller against a run I already knew had crashed -- the reason this tool is tested
    against known-bad runs, not just happy ones.)"""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return [l for l in out.splitlines()
            if "python" in l and not _SHELL_WRAPPER.match(l.strip()) and " -c " not in l[:200]]


def _pids():
    """(pid, cmdline) of live python WORKER procs -- same shell-wrapper exclusion as _procs()."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines()[1:]:
        line = line.strip()
        if not line or "python" not in line:
            continue
        pid, _, args = line.partition(" ")
        if _SHELL_WRAPPER.match(args.strip()) or " -c " in args[:200]:
            continue
        if pid.isdigit():
            rows.append((int(pid), args))
    return rows


def _gpu_pids():
    """PIDs currently registered as CUDA compute apps.

    WHY (2026-07-16, learned the hard way): a 4-arm sweep ran ~50 min on the CPU while this poller happily reported
    RUNNING -- and it was RIGHT: processes alive, CPU climbing, logs growing, all true. It simply could not see WHICH
    DEVICE. The cause was a runner doing os.environ.setdefault("SIM_BACKEND","numpy"), harmless for months only
    because a MISSING scipy made numpy mode throw and fall back to cupy; installing scipy made that default REAL.
    "Running on the wrong accelerator" is exactly the silent-wrongness this tool exists to catch, so it must look."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None                      # no nvidia-smi => cannot judge; report unknown rather than a false alarm
    return {int(x.strip()) for x in out.split() if x.strip().isdigit()}


def _cpu_ticks(pid):
    """utime+stime from /proc/<pid>/stat. RISING => the proc is genuinely computing, however quiet its log is."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            parts = f.read().decode("utf-8", "replace").rsplit(")", 1)[1].split()
        return int(parts[11]) + int(parts[12])       # utime, stime (fields 14,15; 1-indexed pre-split)
    except Exception:
        return None


def _find(stem, pids, log_path=None):
    """Map a run to its worker pid.

    AUTHORITATIVE FIRST (fix 2026-07-31): the process whose STDOUT *is* this log file. A `> x.log` redirect never
    appears in a cmdline, so the original heuristic matched on `basename(stem) + ".json"` instead -- i.e. it
    required the --out json to be named after the log. That is a NAMING CONVENTION, not a fact: a run whose
    --out differs from its log name reads as pid=None -> alive=False -> reported DONE/CRASHED while it is
    happily computing, and a coincidental name match on another process reports the wrong pid (so the CPU-tick
    liveness delta is then read off an unrelated process). tools/device_check.sh already resolves this correctly
    via /proc/<pid>/fd/1; this brings the monitor in line.
    """
    if log_path:
        try:
            target = os.path.realpath(log_path)
            for pid, _args in pids:
                try:
                    if os.path.realpath("/proc/%s/fd/1" % pid) == target:
                        return pid
                except (OSError, PermissionError):
                    continue
        except OSError:
            pass
    needle = os.path.basename(stem) + ".json"          # fallback: the original --out-name heuristic
    for pid, args in pids:
        if needle in args:
            return pid
    return None


def _tail(path, n=400_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def classify(log, pids, stall_s, prog_re, prev, gpu_pids=None):
    stem = log[:-4] if log.endswith(".log") else log
    exists = os.path.exists(log)
    size = os.path.getsize(log) if exists else 0
    age = (time.time() - os.path.getmtime(log)) if exists else 0.0
    txt = _tail(log)
    pid = _find(stem, pids, log_path=log)
    alive = pid is not None
    ticks = _cpu_ticks(pid) if alive else None
    err = ERROR_RE.search(txt)
    ok = SUCCESS_RE.search(txt)
    n_err = len(ERROR_RE.findall(txt))

    # LIVENESS = CPU-TIME DELTA, not log growth. Log-staleness alone CANNOT tell "computing hard but quiet" from
    # "hung" -- and it cried wolf immediately in practice: a healthy e-prop arm at 99.8% CPU logs only once per
    # SEED (30-60 min), so a 20-min log gap flagged STALLED while utime was rising a full core's worth per second.
    # A false STALLED is a lie in the other direction -- it trains the reader to ignore the alerts. So:
    #   HUNG  = alive but CPU FLAT across a tick   -> a REAL alarm (deadlock / blocked forever / sleeping)
    #   QUIET = alive, CPU RISING, log stale       -> INFORMATIONAL: working, just between log lines
    cpu_moved = None
    if (alive and ticks is not None and prev is not None and prev.get("ticks") is not None
            and prev.get("pid") == pid):
        # `and prev.get("pid") == pid` is LOAD-BEARING: without it, a relaunch (new pid, tick counter restarting
        # near 0) compares against the DEAD proc's accumulated ticks, reads as CPU-flat, and fires a FALSE HUNG on
        # a healthy run. Observed 2026-07-16 on all four arms at once. A monitor that cries wolf is as corrosive as
        # one that stays silent -- it trains the reader to ignore it.
        cpu_moved = ticks > prev["ticks"]

    if alive:
        if cpu_moved is False:
            state = "HUNG"
        elif age > stall_s:
            state = "QUIET"
        else:
            state = "RUNNING"
    else:
        # An exit WITHOUT a success marker is a FAILURE, never a pass. Silence is not success.
        state = "DONE" if (ok and not err) else "CRASHED"

    m = list(re.finditer(prog_re, txt))
    prog = m[-1].group(0).strip()[:100] if m else ("(no progress lines yet)" if exists else "(no log yet)")
    detail = ""
    if err:
        first = ERROR_RE.search(txt).group(0).strip()[:60]
        detail = f" | {n_err} error-hit(s), first: {first}"
    # DEVICE: a live worker that is NOT a registered CUDA compute app is running on the CPU. Flag it -- a run on
    # the wrong device looks identical to a healthy one by every other signal (alive, CPU rising, log growing).
    dev = ""
    if alive and gpu_pids is not None:
        dev = " [GPU]" if pid in gpu_pids else "  <-- ON CPU, not the GPU (SIM_BACKEND / setdefault?)"
    return {"state": state, "prog": prog, "size": size, "age": age, "detail": detail + dev,
            "ticks": ticks, "pid": pid, "json": os.path.exists(stem + ".json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="log paths or globs")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--heartbeat", type=int, default=3, help="emit status every N ticks even if unchanged")
    ap.add_argument("--stall-min", type=float, default=20.0)
    ap.add_argument("--progress-re", default=DEFAULT_PROGRESS_RE)
    ap.add_argument("--max-ticks", type=int, default=288)
    a = ap.parse_args()

    prog_re = re.compile(a.progress_re)
    stall_s = a.stall_min * 60.0
    last, t0, tick = {}, time.time(), 0

    while tick < a.max_ticks:
        tick += 1
        logs = sorted({p for pat in a.logs for p in (glob.glob(pat) or [pat])})
        pids = _pids()
        gpu = _gpu_pids()
        rows = {log: classify(log, pids, stall_s, prog_re, last.get(log), gpu) for log in logs}
        beat = (tick % a.heartbeat == 0)
        mins = (time.time() - t0) / 60.0

        for log, r in rows.items():
            prev = last.get(log)
            changed = (prev is None or prev["state"] != r["state"] or prev["prog"] != r["prog"])
            if changed or beat:
                name = os.path.basename(log)[:-4]
                flag = {"RUNNING": "",
                        "QUIET": f"  (working: CPU rising; no log line for {r['age']/60:.0f}m -- normal for runners that log per-seed)",
                        "HUNG": "  <-- ALARM: alive but CPU FLAT (deadlock / blocked / sleeping)",
                        "DONE": "", "CRASHED": "  <-- CRASHED"}[r["state"]]
                print(f"[{mins:5.1f}m] {r['state']:<8} {name:<34} {r['prog']}{r['detail']}{flag}", flush=True)
            last[log] = r

        if rows and all(r["state"] in ("DONE", "CRASHED") for r in rows.values()):
            done = sum(r["state"] == "DONE" for r in rows.values())
            bad = [os.path.basename(l)[:-4] for l, r in rows.items() if r["state"] == "CRASHED"]
            print(f"[{mins:5.1f}m] ALL TERMINAL -- {done}/{len(rows)} DONE"
                  + (f"; CRASHED: {', '.join(bad)}" if bad else "") + " -- poller exiting.", flush=True)
            return
        time.sleep(a.interval)

    print(f"[{(time.time()-t0)/60:5.1f}m] max-ticks reached; poller exiting (runs may still be live).", flush=True)


if __name__ == "__main__":
    main()
