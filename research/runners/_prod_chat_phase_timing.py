"""Per-phase wall-clock + memory-pressure timing of the DEFAULT PRODUCTION chat path, on the GPU (2026-09-24, S01/G1).

WHY. `tests/test_production_chat_gpu_smoke.py` is a class guard: it proves the default `/api/brain-chat` turn
(tiny-demo + the spiking Qwen mouth) answers 200 on the cupy backend instead of 400-crashing. It does not say
HOW LONG anything took or whether the run was throttled by the RAM cap. Before BRAIN_AFFECT_MARKER_SETTLE (which
lengthens the affect-marker circuit's deliberation window 60->500 ms and inter-turn rest 40->1000 ms simulated,
research/runners/_affect_marker_wta_derisk.py) can join a default-on flip batch, its per-phase cost on the ACTUAL
production GPU chat path has to be on record -- not just the isolated affect-marker circuit in isolation
(research/runners/_settle_turn_cost_probe.py measures that; this measures the same flag through the real
`webapp.server.brain_chat` entrypoint the class-guard test exercises).

This is a HOST-SIDE INSTRUMENT (timers + /proc reads around an unmodified production call), not a cognitive
mechanism -- it changes nothing about what the brain computes; see the brain-based-only boundary in CLAUDE.md.
It ADDS NO NEW FLAG: BRAIN_AFFECT_MARKER_SETTLE is read from the environment exactly as production reads it,
this runner does not set or default it, so it changes nothing when SETTLE itself is off.

FOUR PHASES per run, each stamped with /proc/loadavg and this process's own cgroup memory.events 'high' count
(memcap.sh sets MemoryHigh to 90% of its hard cap; a rising 'high' count is reclaim-throttling BEFORE any hard
OOM-kill -- see tools/memcap.sh):
  1. brain_build             -- constructing the tiny-demo spiking substrate (webapp.server._build_chat_brain,
                                 MINUS the Qwen sub-span below). Only paid once, inside the turn-1 call.
  2. qwen_weight_load         -- the off-bridge Qwen-0.5B model load + the P1b calibration pass + spiking-op
                                 install (webapp.server._get_warm_qwen_renderer -> QwenRenderer.__init__ ->
                                 SpikingQwenFaculty.__init__). A process-wide singleton: paid ONCE per process,
                                 inside the turn-1 call, never again on turn 2.
  3. turn_1                   -- the first answer, wall time MINUS the brain-build+Qwen-load span above (so it
                                 reads the actual gate/compose/render cost of one turn, not construction).
  4. turn_2 (total / qwen_cuda_generation / non_generation) -- a second, WARM turn on the same session (no
                                 build, no weight load): total wall time, the isolated `model.generate()` CUDA
                                 span (patched at SpikingQwenFaculty._generate / _generate_batch, whichever the
                                 production `rich` default routes through), and the remainder.

Phase separation is done by WRAPPING (not editing) four existing functions for the run's duration:
webapp.server._build_chat_brain, webapp.server._get_warm_qwen_renderer, and
research.runners._grounded_lang_integration_derisk.SpikingQwenFaculty.{_generate,_generate_batch}. No production
module is modified on disk; the wraps are undone (best-effort) before the process exits.

USAGE (queued -- do not run this directly against the shared 3090; see the PREREGISTRATION for the exact
mem_ok/memcap-wrapped command):
    SIM_BACKEND=cupy OMP_NUM_THREADS=1 BRAIN_AFFECT_MARKER_SETTLE=1 .venv/bin/python -u \\
        -m research.runners._prod_chat_phase_timing --out research/findings/raw/_settle_cost/prod_chat_phase_settle_on.json
    SIM_BACKEND=cupy OMP_NUM_THREADS=1 BRAIN_AFFECT_MARKER_SETTLE=0 .venv/bin/python -u \\
        -m research.runners._prod_chat_phase_timing --out research/findings/raw/_settle_cost/prod_chat_phase_settle_off.json

Dev/calibration smoke (CPU, no GPU, seed 7 ONLY -- never 42/43/44/100/101/102, see CLAUDE.md):
    SIM_BACKEND=numpy BRAIN_CHAT_SEED=7 .venv/bin/python -m research.runners._prod_chat_phase_timing \\
        --renderer stub --seed 7 --out /tmp/smoke.json

Compare two runs against pre-registered criterion L (SETTLE warm-turn delta <= +0.3 s on GPU):
    .venv/bin/python -m research.runners._prod_chat_phase_timing --compare-on ON.json --compare-off OFF.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

_TIMEOUT_S = 1800.0   # mirrors the 1800 s timeout in the plan step (a full cold build + Qwen load + 2 turns
                       # is expected to be well under this; see the PREREGISTRATION for the observed budget).


def _loadavg():
    try:
        with open("/proc/loadavg") as f:
            return [float(x) for x in f.read().split()[:3]]
    except Exception:
        return None


def _cgroup_memory_high_count():
    """This process's own cgroup-v2 memory.events 'high' counter (incremented every time usage crosses the
    scope's MemoryHigh throttling threshold -- memcap.sh sets MemoryHigh=90% of its hard MemoryMax cap). Returns
    None, NOT 0, when memory.events is not readable (no cgroup v2, not inside a memcap.sh scope, permission) --
    absence of the file is a different fact than a genuine zero count (tools.lab's undefined-not-zero discipline)."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().strip().splitlines()
        path = None
        for line in lines:
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[0] == "0":
                path = parts[2]
                break
        if not path:
            return None
        with open("/sys/fs/cgroup" + path + "/memory.events") as f:
            for line in f:
                k, v = line.split()
                if k == "high":
                    return int(v)
    except Exception:
        return None
    return None


def _snapshot(phase, t0):
    return {"phase": phase, "wall_s_since_start": round(time.perf_counter() - t0, 4),
            "loadavg": _loadavg(), "cgroup_memory_high": _cgroup_memory_high_count()}


class _HardTimeout:
    """A watchdog THREAD (not signal.alarm) that force-exits the process after `secs` unless cancelled: a
    background thread keeps running (and os._exit() needs no GIL cooperation) even if the main thread is
    blocked inside a long C/CUDA call, where a SIGALRM handler can be deferred until the call returns."""

    def __init__(self, secs, out_path):
        self._done = threading.Event()
        self._t = threading.Thread(target=self._watch, args=(secs, out_path), daemon=True)

    def _watch(self, secs, out_path):
        if self._done.wait(secs):
            return
        sys.stderr.write("[prod_chat_phase_timing] HARD TIMEOUT at %ss -- writing a TIMED_OUT marker to %s "
                          "and force-exiting so the GPU queue is not held forever.\n" % (secs, out_path))
        sys.stderr.flush()
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({"status": "TIMED_OUT", "timeout_s": secs}, f, indent=1)
        except Exception:
            pass
        os._exit(124)   # conventional timeout exit code

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        return False


def _run(a):
    t0 = time.perf_counter()
    phases = []

    backend = os.environ.get("SIM_BACKEND", "numpy")
    settle_env = os.environ.get("BRAIN_AFFECT_MARKER_SETTLE")
    if a.renderer == "qwen" and backend != "cupy":
        print("[prod_chat_phase_timing] WARNING: --renderer qwen with SIM_BACKEND=%r (not 'cupy'). "
              "QwenRenderer needs cupy + CUDA torch; this will likely raise. Use --renderer stub for a "
              "CPU/numpy dev smoke." % backend, file=sys.stderr)
    if a.seed is not None:
        os.environ["BRAIN_CHAT_SEED"] = str(a.seed)   # webapp.server._brain_chat_seed(); unset -> 42

    # Import AFTER env is set: SIM_BACKEND and BRAIN_CHAT_SEED are both read at/around import or first-build time.
    import webapp.server as ws

    # ---- wrap (1) brain build and (2) Qwen weight load, so their SHARED call (_build_chat_brain, invoked once
    # on cache-miss inside brain_chat) can be reported as two separate numbers without editing either function.
    _qwen_load_s = {"v": 0.0}
    _orig_get_warm_qwen = ws._get_warm_qwen_renderer

    def _timed_get_warm_qwen():
        t = time.perf_counter()
        r = _orig_get_warm_qwen()
        _qwen_load_s["v"] += time.perf_counter() - t
        return r
    ws._get_warm_qwen_renderer = _timed_get_warm_qwen

    _build_total_s = {"v": 0.0}
    _orig_build_chat_brain = ws._build_chat_brain

    def _timed_build_chat_brain(*args, **kwargs):
        t = time.perf_counter()
        r = _orig_build_chat_brain(*args, **kwargs)
        _build_total_s["v"] += time.perf_counter() - t
        return r
    ws._build_chat_brain = _timed_build_chat_brain

    # ---- wrap the Qwen CUDA generation span (both the single-item and batched launch paths -- the production
    # `rich` default can route through either) so a turn's generation cost is isolated from its non-generation
    # (gate/compose/VERIFY/host) cost.
    _gen_spans = []
    _gen_patch_ok = False
    try:
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty as _SQF
        _orig_generate = _SQF._generate
        _orig_generate_batch = _SQF._generate_batch

        def _timed_generate(self, *args, **kwargs):
            r = _orig_generate(self, *args, **kwargs)
            _gen_spans.append(("_generate", r[-1]))   # (first_line, full_text, seconds)
            return r
        _SQF._generate = _timed_generate

        def _timed_generate_batch(self, *args, **kwargs):
            r = _orig_generate_batch(self, *args, **kwargs)
            _gen_spans.append(("_generate_batch", r[-1]))   # (results, batch_seconds)
            return r
        _SQF._generate_batch = _timed_generate_batch
        _gen_patch_ok = True
    except Exception as e:
        print("[prod_chat_phase_timing] could not patch SpikingQwenFaculty generation methods (%s: %s) -- "
              "the turn-level Qwen-generation span will read null; phase totals are unaffected."
              % (type(e).__name__, e), file=sys.stderr)

    from webapp.server import brain_chat, BrainChatRequest

    session = "prod_chat_phase_timing_%d_%s" % (os.getpid(), a.mode or "run")

    # ---- TURN 1: cache-miss -> pays brain_build + qwen_weight_load + turn-1 processing in one brain_chat() call.
    n_gen_before = len(_gen_spans)
    t_turn1 = time.perf_counter()
    resp1 = brain_chat(BrainChatRequest(session=session, message=a.message1, brain="tiny-demo",
                                        renderer=a.renderer))
    turn1_wall_s = time.perf_counter() - t_turn1
    code1 = int(getattr(resp1, "status_code", 0))
    body1 = json.loads(bytes(resp1.body))
    turn1_gen_s = round(sum(s for _, s in _gen_spans[n_gen_before:]), 4) if _gen_patch_ok else None

    brain_build_s = round(max(_build_total_s["v"] - _qwen_load_s["v"], 0.0), 4)
    qwen_weight_load_s = round(_qwen_load_s["v"], 4)
    turn1_processing_s = round(max(turn1_wall_s - _build_total_s["v"], 0.0), 4)

    phases.append(dict(_snapshot("brain_build", t0), elapsed_s=brain_build_s))
    phases.append(dict(_snapshot("qwen_weight_load", t0), elapsed_s=qwen_weight_load_s,
                        note="includes the one-time P1b calibration pass + spiking-op install, not model "
                             "weight I/O alone; see webapp.server._get_warm_qwen_renderer"))
    phases.append(dict(_snapshot("turn_1", t0), elapsed_s=turn1_processing_s, qwen_cuda_generation_s=turn1_gen_s,
                        wall_s=round(turn1_wall_s, 4), http_status=code1, abstained=bool(body1.get("abstained"))))

    # ---- TURN 2: cache-hit -> no build, no weight load. This is the WARM turn criterion L compares.
    n_gen_before2 = len(_gen_spans)
    t_turn2 = time.perf_counter()
    resp2 = brain_chat(BrainChatRequest(session=session, message=a.message2, brain="tiny-demo",
                                        renderer=a.renderer))
    turn2_wall_s = round(time.perf_counter() - t_turn2, 4)
    code2 = int(getattr(resp2, "status_code", 0))
    body2 = json.loads(bytes(resp2.body))
    turn2_gen_s = round(sum(s for _, s in _gen_spans[n_gen_before2:]), 4) if _gen_patch_ok else None
    turn2_non_gen_s = round(max(turn2_wall_s - (turn2_gen_s or 0.0), 0.0), 4)

    phases.append(dict(_snapshot("turn_2_total", t0), elapsed_s=turn2_wall_s, http_status=code2,
                        abstained=bool(body2.get("abstained"))))
    phases.append(dict(_snapshot("turn_2_qwen_cuda_generation", t0), elapsed_s=turn2_gen_s))
    phases.append(dict(_snapshot("turn_2_non_generation", t0), elapsed_s=turn2_non_gen_s))

    result = {
        "backend": backend, "renderer": a.renderer,
        "brain_chat_seed_env": os.environ.get("BRAIN_CHAT_SEED"),
        "settle_env_BRAIN_AFFECT_MARKER_SETTLE": settle_env,
        "mode": a.mode,
        "message1": a.message1, "message2": a.message2,
        "gen_patch_ok": _gen_patch_ok,
        "phases": phases,
        "warm_turn_total_s": turn2_wall_s,   # the quantity criterion L compares on/off vs.
        "total_wall_s": round(time.perf_counter() - t0, 4),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"phase_elapsed_s": {p["phase"]: p.get("elapsed_s") for p in phases},
                       "warm_turn_total_s": turn2_wall_s, "turn1_http": code1, "turn2_http": code2}))

    # best-effort unwind (this process is about to exit either way; harmless if it doesn't fully restore)
    try:
        ws._get_warm_qwen_renderer = _orig_get_warm_qwen
        ws._build_chat_brain = _orig_build_chat_brain
    except Exception:
        pass


def _compare(a):
    """Evaluate PREREGISTRATION criterion L: the SETTLE warm-turn delta (on - off) must be <= +0.3 s on GPU."""
    with open(a.compare_on) as f:
        on = json.load(f)
    with open(a.compare_off) as f:
        off = json.load(f)
    delta = on["warm_turn_total_s"] - off["warm_turn_total_s"]
    verdict = "PASS" if delta <= 0.3 else "FAIL"
    out = {"criterion": "L: SETTLE warm-turn delta <= +0.3 s on GPU",
           "on_warm_turn_total_s": on["warm_turn_total_s"], "off_warm_turn_total_s": off["warm_turn_total_s"],
           "delta_s": round(delta, 4), "verdict": verdict,
           "on_backend": on.get("backend"), "off_backend": off.get("backend"),
           "on_settle_env": on.get("settle_env_BRAIN_AFFECT_MARKER_SETTLE"),
           "off_settle_env": off.get("settle_env_BRAIN_AFFECT_MARKER_SETTLE")}
    if out["on_backend"] != "cupy" or out["off_backend"] != "cupy":
        out["caveat"] = "at least one run's backend was not 'cupy' -- criterion L is a GPU criterion; this is de-risk only"
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="where to write this run's phase-timing JSON (required unless --compare-*)")
    ap.add_argument("--seed", type=int, default=None,
                    help="sets BRAIN_CHAT_SEED (webapp.server._brain_chat_seed(); unset -> production default 42). "
                         "Use 7 (never 42/43/44/100/101/102) for a dev/calibration smoke only.")
    ap.add_argument("--renderer", default="qwen", choices=["qwen", "stub", "raw"],
                    help="'qwen' is the production GPU path this runner exists to time; 'stub' is a GPU-free "
                         "dev smoke that skips phases 2 and the generation span entirely.")
    ap.add_argument("--message1", default="what does the dog chase?")
    ap.add_argument("--message2", default="what does the cat eat?")
    ap.add_argument("--mode", default=None, help="free-text tag echoed into the output JSON (e.g. 'settle_on')")
    ap.add_argument("--timeout-s", type=float, default=_TIMEOUT_S)
    ap.add_argument("--compare-on", help="compare mode: path to a settle-ON run's JSON (no brain build; instant)")
    ap.add_argument("--compare-off", help="compare mode: path to a settle-OFF run's JSON")
    a = ap.parse_args()

    if a.compare_on or a.compare_off:
        if not (a.compare_on and a.compare_off):
            ap.error("--compare-on and --compare-off must be given together")
        return _compare(a)

    if not a.out:
        ap.error("--out is required (unless using --compare-on/--compare-off)")
    with _HardTimeout(a.timeout_s, a.out):
        _run(a)


if __name__ == "__main__":
    sys.exit(main())
