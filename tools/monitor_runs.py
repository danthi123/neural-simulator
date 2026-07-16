"""monitor_runs — a COVERAGE-COMPLETE status poller for long background runs (GPU sweeps, de-risks, training).

WHY THIS EXISTS (three real failures on 2026-07-16, all in one session):
  1. A `tail -F | grep "==> seed"` monitor stayed SILENT through a run that crash-looped 10023 times -- the filter
     matched only the happy path, so a dying run looked identical to a working one.
  2. Widening that filter to catch "Traceback" then spammed one event per line until the monitor was AUTO-STOPPED
     for excessive output -- so the run became UNwatched precisely when it was failing.
  3. A run silently no-op'd forever (swallowed init -> is_initialized False -> every step an early return). It
     logged NOTHING and never exited. No log-tailing monitor can ever detect that; only "the log stopped growing
     while the process is still alive" can.

THE FIX: do not tail. POLL STATE. Every tick, classify each run into exactly one of four states -- and note that
SILENCE IS A STATE (`STALLED`), not the absence of one:

  RUNNING  proc alive + log grew since the last tick
  STALLED  proc alive + log has NOT grown for --stall-min minutes   <- the hang/silent-no-op case
  DONE     proc gone  + a success marker present
  CRASHED  proc gone  + an error marker present, OR no success marker (an exit without a marker is a FAILURE,
           never a pass -- "it stopped" must never be read as "it finished")

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
  --stall-min   minutes of no log growth (proc still alive) before declaring STALLED (default 20)
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


def _alive(stem, procs):
    """A run is alive if some python proc's cmdline mentions its json (the --out arg). The log itself is a shell
    redirect and never appears in a cmdline, so the json stem is the only reliable link."""
    needle = os.path.basename(stem) + ".json"
    return any(needle in p for p in procs)


def _tail(path, n=400_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def classify(log, procs, stall_s, prog_re):
    stem = log[:-4] if log.endswith(".log") else log
    exists = os.path.exists(log)
    size = os.path.getsize(log) if exists else 0
    age = (time.time() - os.path.getmtime(log)) if exists else 0.0
    txt = _tail(log)
    alive = _alive(stem, procs)
    err = ERROR_RE.search(txt)
    ok = SUCCESS_RE.search(txt)
    n_err = len(ERROR_RE.findall(txt))

    if alive:
        state = "STALLED" if age > stall_s else "RUNNING"
    else:
        # An exit WITHOUT a success marker is a FAILURE, never a pass. Silence is not success.
        state = "DONE" if (ok and not err) else "CRASHED"

    m = list(re.finditer(prog_re, txt))
    prog = m[-1].group(0).strip()[:100] if m else ("(no progress lines yet)" if exists else "(no log yet)")
    detail = ""
    if err:
        first = ERROR_RE.search(txt).group(0).strip()[:60]
        detail = f" | {n_err} error-hit(s), first: {first}"
    return {"state": state, "prog": prog, "size": size, "age": age, "detail": detail,
            "json": os.path.exists(stem + ".json")}


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
        procs = _procs()
        rows = {log: classify(log, procs, stall_s, prog_re) for log in logs}
        beat = (tick % a.heartbeat == 0)
        mins = (time.time() - t0) / 60.0

        for log, r in rows.items():
            prev = last.get(log)
            changed = (prev is None or prev["state"] != r["state"] or prev["prog"] != r["prog"])
            if changed or beat:
                name = os.path.basename(log)[:-4]
                flag = {"RUNNING": "", "STALLED": "  <-- NO LOG GROWTH, proc still alive (hang? silent no-op?)",
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
