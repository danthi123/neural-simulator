"""AFFECT-MARKER SETTLE — GPU per-turn latency, criterion 3 of the SETTLE default-flip (research/settle-multiturn-contrast).

WHY. `BRAIN_AFFECT_MARKER_SETTLE` adds simulated TIME to the affect-marker WTA's own tiny private bridge: per
axis (valence, then -- only when a word was actually selected -- arousal), `WARMUP_STEPS` 60->`DELIBERATION_MS`
500 and `WASHOUT_STEPS` 40->`INTERTURN_REST_MS` 1000 (dt_ms=1.0, so step count == simulated ms). A full
non-neutral turn calls `_select` TWICE, so SETTLE adds up to ~2 x ((500-60)+(1000-40)) = 2800 simulated ms of
extra steps per AFFECTIVE turn. A neutral turn never reaches the circuit (`expression_lead` returns '' at level 0
before any spiking read), so SETTLE's cost exists only on affective turns.

AMENDMENT A3 (2026-09-24, fix round after an independent review; this runner is REWRITTEN and committed BEFORE its
first run; governing document research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md).
The pre-A3 runner (49a089d8d) had no threshold and no verdict, timed the stub renderer with the LLM disabled (not
the quantity criterion L defines), ran n=1 with OFF always before ON, never checked that SETTLE reached the reader,
and its ON arm ran on a dirty tree. Its s42 artifacts (research/findings/raw/_affect_marker_settle_gpu_timing/s42*)
are kept as the record of that run and CANNOT support a verdict. What A3 measures instead:

  QUANTITY (criterion L, 228ba16f0, carried over): the WARM-turn total wall time of `webapp.server.brain_chat` on
  the DEFAULT production path -- `SIM_BACKEND=cupy`, brain 'tiny-demo', renderer 'qwen' (the off-bridge Qwen-0.5B
  mouth, LLM enabled), `rich` left at the production default -- SETTLE ON minus OFF, bound +0.3 s. ONE CHANGE to
  criterion L: its default message pair ("what does the dog chase?" / "what does the cat eat?") is NEUTRAL, so the
  WTA never runs and the delta is zero by construction. Here every warm turn is AFFECTIVE.

  PROTOCOL. Fresh subprocess per process, sequential (never concurrent). Order counterbalanced ABBA:
  off, on, on, off (>= 2 processes per arm). Each process: a build turn (NEU_TEXT, reset) timed separately; a first
  affective turn (EMO_TEXT; builds the WTA reader lazily; reported, not scored); then N_WARM=4 warm affective turns
  alternating EMO2_TEXT / EMO_TEXT (the scored turns). Inside the process, `AffectMarkerWTA._select` is wrapped by a
  timer, so each warm turn also reports the wall time spent in the spiking WTA itself (M2), the reader config it
  ran with, and how many reads it made.

  METRICS. Per process: the median over its warm turns. Per arm: the median of its process medians.
    M1 = arm(ON) - arm(OFF) of the warm-turn TOTAL wall time        (criterion L's quantity)
    M2 = arm(ON) - arm(OFF) of the warm-turn WTA wall time          (the flag's own attributable cost)
    NOISE = max over arms of (max - min) of that arm's process medians of the warm-turn total
    M3 = arm(ON) - arm(OFF) of (total - WTA), REPORTED (should be ~0: SETTLE touches nothing else)

  PRECONDITIONS (any unmet -> UNDEFINED): every process completed; >= 2 processes per arm; the order is balanced
  (equal mean position per arm); one code revision across processes, with no tracked file modified other than the
  provenance log; backend cupy; the reply's renderer is the Qwen renderer; every warm turn is affective (level != 0)
  and made >= 1 WTA read; every WTA read and every cached reader ran with the arm's expected config
  (ON 500/1000 ms, OFF 60/40 ms) -- the lever check; and the result is RESOLVABLE at the bound (below).

  RULE. bound = 0.3 s.  PASS region: M2 <= bound AND M1 + NOISE <= bound.  FAIL region: M2 > bound OR
  M1 - NOISE > bound.  Resolvable iff in one region.  GO = PASS; NO-GO = FAIL; otherwise UNDEFINED (the
  whole-turn delta cannot be told apart from the bound at this noise). Seed 42 only (production default): a latency
  de-risk, not a capability claim, exactly as criterion L's pre-registration scopes it.

Queue (GPU, one job at a time; run from a clean checkout pinned at the commit that carries this file, with
data/corpus/tinystories.txt present -- the Qwen renderer reads it at load):
  bash tools/gpu_queue.sh add 'cd <pinned checkout> && until bash tools/mem_ok.sh 16 4 >/dev/null 2>&1; do sleep 60;
    done; SIM_BACKEND=cupy OMP_NUM_THREADS=1 bash tools/memcap.sh 20 -- <venv>/bin/python -u -m
    research.runners._affect_marker_settle_gpu_timing --run --seeds 42 --order off,on,on,off
    --out-dir <abs>/research/findings/raw/_affect_marker_settle_gpu_timing/a3
    --out <abs>/research/findings/raw/_affect_marker_settle_gpu_timing/a3/verdict.json'

Self-test (no brain build, no GPU; drives the pure verdict through every failing direction):
  .venv/bin/python -m research.runners._affect_marker_settle_gpu_timing --selftest

AMENDMENT 3 (2026-09-25; written AFTER the A3 run above read UNDEFINED -- M2 +0.133 s, M1 -3.46 s, NOISE 4.72 s --
and BEFORE any new data; governing section: "Amendment 3 -- A3" in the PREREG named above; decomposition of that
run: research/findings/2026-09-25-affect-marker-settle-a3-gpu-timing-UNDEFINED-noise.md). The A3 functions above
are kept unchanged: they reproduce the A3 verdict from its committed artifacts. The instrument that decides A3 from
now on is the WITHIN-PROCESS CROSSOVER below (`--xo-run` / `--xo-score`, pure verdict `decide_xo`).

  SAME QUANTITY: warm affective turn total wall time of `webapp.server.brain_chat` on the production path (cupy,
  tiny-demo, Qwen renderer with the LLM on, `rich` default), SETTLE ON minus OFF, bound +0.3 s.

  WHY A NEW DESIGN. A3's noise was between PROCESSES (arm medians 30.2/28.8 vs 35.3/30.6 s; the SETTLE-free build
  turn was slow in the same processes), and its NOISE was a RANGE of process medians, which grows with the number
  of processes -- no number of A3 processes resolves 0.3 s. Here every process runs BOTH arms, so process-level
  speed cancels inside each process.

  PROTOCOL. `BRAIN_AFFECT_MARKER_SETTLE` is toggled per turn inside one process. `get_reader` keys its cache by the
  flag, so each arm keeps its own process-warm reader (nothing is rebuilt between turns; a rebuild would add the
  reader build to every turn, which a production warm turn never pays). The Qwen renderer is warmed before the
  build turn (what the server's startup warm does). Per process: a build turn (NEU_TEXT, reset); two unscored
  warm-up affective turns (one per arm, so both readers exist before scoring); then XO_RUNS runs of XO_RUN_LEN
  same-arm turns. Run arms follow ABBA repeated (A = the process's orientation), so linear drift cancels inside
  each process. The first turn of every run is a WASHOUT (unscored): every scored turn follows a turn of its own
  arm, so a lag-1 carry-over of SETTLE into the next turn counts toward the arm that causes it, as it would in an
  all-ON session. Turn k gets the same message in every process ((EMO_TEXT, EMO2_TEXT)[k % 2]). Four processes at
  seed 42, orientations off,on,on,off (the A3 counterbalance, now as mirrored orientations): at every run slot two
  processes are ON and two are OFF, so the run-slot fixed effect removes the turn-to-turn work variation all
  processes share (the A3 turn means ranged 28.7-33.8 s by turn index alone).

  INSTRUMENT (no production code changes; wrappers only time and count): per turn, the spiking WTA `_select` wall
  time + config + read count + exceptions; every `generate` call on the warm Qwen model (wall time, batch, new
  tokens, prompt tokens); the reply with the lead removed (sha1) and its length; process CPU seconds; 1-min load
  average at turn start; the CuPy pool bytes in use; max RSS.

  ESTIMATE. Run level: y(p, r) = mean over the run's kept scored turns. OLS with process and run-slot fixed effects
  and an ON indicator; delta = the ON coefficient; one-sided 95% bounds delta +/- t(0.95, df) * SE. M1 = delta on
  the total wall time, M2 = delta on the WTA time. Reported, not gated: delta on Qwen render time and on the rest,
  work identity (render calls / tokens / reply-without-lead, ON vs OFF, per kept index), a lag-1 carry-over
  estimate from all affective run turns, a process-demeaned per-slot median, the load average by arm.

  PRECONDITIONS (any unmet -> UNDEFINED): every process completed and validated (cupy, Qwen renderer on every
  turn, HTTP 200, clean tree, the planned arm on every turn, both readers cached with their configs, every
  affective run turn read the WTA with its arm's config and no reader exception); >= 2 processes per orientation,
  equal counts; one code revision; one plan; the same message at every index in every process; kept scored
  indices >= 80% of planned (an index is dropped for ALL processes iff any process's turn there is neutral --
  level is set before the WTA reads, so the drop cannot depend on the arm); and resolvable (below).

  RULE. bound = 0.3 s. PASS region: U(M2) <= bound AND U(M1) <= bound -> GO. FAIL region: L(M2) > bound OR
  L(M1) > bound -> NO-GO. Otherwise UNDEFINED.

  SIZE. XO_RUNS=48 x XO_RUN_LEN=4 per process = 144 scored turns/process, 576 total. From the A3 two-way residual
  (0.80 s per turn without the contended process 00_off, 1.72 s with it) the expected one-sided half-width is
  ~0.11 s (80% power to read GO if the true cost is the measured WTA cost) at 0.80 s, ~0.24 s at 1.72 s. ~7.5 h.

  Queue (GPU; clean checkout pinned at the commit carrying this file, data/corpus/tinystories.txt symlinked in):
    --xo-run --seeds 42 --orient off,on,on,off --runs 48 --run-len 4 --out-dir <abs a3x dir> --out <abs a3x>/verdict.json
  Score an existing raw dir again: --xo-score --raw-dir <a3x dir> --orient off,on,on,off --out <json>
  Decompose the A3 run: --decompose-a3 --raw-dir <a3 dir> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

VERIFY_SEEDS = (42,)          # production default seed; criterion L is a 1-seed latency de-risk
SETTLE_ENV = "BRAIN_AFFECT_MARKER_SETTLE"
GPU_SESSION = "gpu_settle_timing"
BOUND_S = 0.3                 # criterion L (228ba16f0), unchanged
DEFAULT_ORDER = "off,on,on,off"
N_WARM = 4
PROC_TIMEOUT_S = 3600
# Amendment 3 (within-process crossover); see the docstring block of that name
XO_DEFAULT_ORIENT = "off,on,on,off"   # first run's arm per process: mirrored, the A3 counterbalance
XO_RUNS = 48                  # runs per process, a multiple of 4 (ABBA quads)
XO_RUN_LEN = 4                # turns per run; the first is a washout
XO_MIN_KEPT_FRAC = 0.8        # kept scored indices / planned scored indices
XO_ALPHA = 0.05               # one-sided, each bound
XO_PROC_TIMEOUT_S = 4 * 3600
XO_EST_TURN_S = 30.0          # A3 warm-turn median, for the cost projection only
XO_EST_BUILD_S = 700.0 + 95.0  # A3 build turn (<= 706 s) + the two warm-up turns
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROV_LOG = "research/findings/raw/_provenance/runs.jsonl"


def _code_state() -> dict:
    def git(*a):
        r = subprocess.run(["git", *a], cwd=_ROOT, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)%s" % _PROV_LOG)
    return {"sha": git("rev-parse", "HEAD"), "dirty_tracked": [ln for ln in (dirty or "").splitlines() if ln],
            "git_ok": dirty is not None}


# ───────────────────────────────────────────── worker (subprocess) ─────────────────────────────────────────────
def _worker(env_json: str, out_path: str, renderer: str, n_warm: int) -> int:
    """Subprocess entry: build ONE tiny-demo brain, one first affective turn, then `n_warm` timed warm ones."""
    os.environ.setdefault("SIM_BACKEND", "cupy")
    if renderer == "stub":                       # dev smoke only; the verdict requires the Qwen renderer
        os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
        os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v                        # OFF-ARM DISCIPLINE: explicit "0"/"1", never a pop
    from research.runners._affect_marker_settle_multiturn_derisk import EMO_TEXT, EMO2_TEXT, NEU_TEXT
    import research.runners._affect_marker_wta_derisk as W

    trace = []
    orig_select = W.AffectMarkerWTA._select

    def timed_select(self, *a, **k):             # instrument only: times the spiking WTA read, changes nothing
        t0 = time.perf_counter()
        try:
            return orig_select(self, *a, **k)
        finally:
            trace.append((time.perf_counter() - t0, bool(self.settle), int(self.warmup), int(self.washout)))
    W.AffectMarkerWTA._select = timed_select
    from webapp.server import brain_chat, BrainChatRequest

    def _turn(msg, reset):
        n0 = len(trace)
        t0 = time.perf_counter()
        r = brain_chat(BrainChatRequest(session=GPU_SESSION, message=msg, brain="tiny-demo", renderer=renderer,
                                        reset=reset))
        wall = time.perf_counter() - t0
        body = json.loads(bytes(r.body))
        sel = trace[n0:]
        ad = body.get("affect_drives") or {}
        return {"wall_s": wall, "http_status": int(getattr(r, "status_code", 0) or 0),
                "wta_s": sum(x[0] for x in sel), "n_wta_reads": len(sel),
                "wta_configs": sorted({(x[1], x[2], x[3]) for x in sel}),
                "level": ad.get("level"), "lead": ad.get("lead", ""), "reason": ad.get("reason"),
                "renderer": body.get("renderer"), "abstained": body.get("abstained")}

    build = _turn(NEU_TEXT, True)
    first = _turn(EMO_TEXT, False)
    warm_texts = (EMO2_TEXT, EMO_TEXT)
    warm = [_turn(warm_texts[i % 2], False) for i in range(int(n_warm))]
    try:
        from sim.backend import get_backend
        backend = get_backend()[1]
    except Exception as e:
        backend = "unknown (%s)" % type(e).__name__
    readers = [{"key": str(k), "settle": bool(r.settle), "warmup": int(r.warmup), "washout": int(r.washout)}
               for k, r in W._READERS.items()]
    rec = {"env": env, "seed": os.environ.get("BRAIN_CHAT_SEED"), "backend": backend, "renderer_requested": renderer,
           "sim_disable_llm": os.environ.get("SIM_DISABLE_LLM"), "code": _code_state(),
           "build_turn": build, "first_affective_turn": first, "warm_turns": warm, "readers": readers}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("[gpu timing worker] env=%s backend=%s build=%.1fs first=%.2fs warm=%s -> %s"
          % (env, backend, build["wall_s"], first["wall_s"], ["%.2f" % w["wall_s"] for w in warm], out_path),
          flush=True)
    return 0


def _spawn(seed: int, arm: str, out_path: str, renderer: str, n_warm: int):
    env = {SETTLE_ENV: "1" if arm == "on" else "0"}
    host_env = dict(os.environ)
    host_env["BRAIN_CHAT_SEED"] = str(int(seed))
    host_env.setdefault("SIM_BACKEND", "cupy")
    try:
        p = subprocess.run([sys.executable, "-u", "-m", "research.runners._affect_marker_settle_gpu_timing",
                            "--worker", "--env", json.dumps(env), "--renderer", renderer, "--n-warm", str(n_warm),
                            "--out", out_path], env=host_env, timeout=PROC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        return json.load(f)


def run_seed(seed: int, order, out_dir: str, renderer: str, n_warm: int) -> list:
    sdir = os.path.join(out_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    procs = []
    for i, arm in enumerate(order):
        rec = _spawn(seed, arm, os.path.join(sdir, "%02d_%s.json" % (i, arm)), renderer, n_warm)
        print("  s%d #%d %s -> %s" % (seed, i, arm, "OK" if rec else "FAILED"), flush=True)
        procs.append({"pos": i, "arm": arm, "rec": rec})
    return procs


# ─────────────────────────────────────────────────── verdict (pure) ────────────────────────────────────────────
def expected_config(arm: str) -> tuple:
    import research.runners._affect_marker_wta_derisk as W
    return ((True, W.DELIBERATION_MS, W.INTERTURN_REST_MS) if arm == "on" else
            (False, W.WARMUP_STEPS, W.WASHOUT_STEPS))


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def check_process(p: dict, renderer_required: str) -> dict:
    arm, rec = p["arm"], p.get("rec")
    if not rec:
        return {"pos": p["pos"], "arm": arm, "valid": False, "problems": ["process failed, timed out or wrote nothing"]}
    probs = []
    exp = expected_config(arm)
    if rec.get("backend") != "cupy":
        probs.append("backend %r, not cupy" % rec.get("backend"))
    code = rec.get("code") or {}
    if not code.get("git_ok") or code.get("dirty_tracked"):
        probs.append("code not clean: %s" % (code.get("dirty_tracked") or "git state unreadable"))
    turns = [rec.get("build_turn") or {}, rec.get("first_affective_turn") or {}] + list(rec.get("warm_turns") or [])
    for t in turns:
        if t.get("http_status") != 200:
            probs.append("a turn returned HTTP %r" % t.get("http_status"))
        if renderer_required not in str(t.get("renderer") or "").lower():
            probs.append("reply renderer %r is not %r" % (t.get("renderer"), renderer_required))
    warm = list(rec.get("warm_turns") or [])
    if not warm:
        probs.append("no warm turns")
    for i, t in enumerate(warm):
        if t.get("level") in (None, 0):
            probs.append("warm turn %d is not affective (level %r)" % (i, t.get("level")))
        if (t.get("n_wta_reads") or 0) < 1:
            probs.append("warm turn %d made no WTA read" % i)
        cfgs = [tuple(c) for c in (t.get("wta_configs") or [])]
        if cfgs and cfgs != [exp]:
            probs.append("warm turn %d read with config %s, expected %s" % (i, cfgs, exp))
    for r in rec.get("readers") or []:
        if (r.get("settle"), r.get("warmup"), r.get("washout")) != exp:
            probs.append("cached reader %s config %s, expected %s" % (r.get("key"), r, exp))
    if not rec.get("readers"):
        probs.append("no cached WTA reader recorded")
    out = {"pos": p["pos"], "arm": arm, "valid": not probs, "problems": probs, "sha": code.get("sha")}
    if warm:
        out["median_total_s"] = _median([t["wall_s"] for t in warm])
        out["median_wta_s"] = _median([t["wta_s"] for t in warm])
        out["median_non_wta_s"] = _median([t["wall_s"] - t["wta_s"] for t in warm])
    return out


def decide(procs: list, *, renderer_required: str = "qwen", bound: float = BOUND_S) -> dict:
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    checked = [check_process(p, renderer_required) for p in procs]
    by_arm = {a: [c for c in checked if c["arm"] == a] for a in ("off", "on")}
    all_valid = bool(checked) and all(c["valid"] for c in checked)
    n_min = min(len(by_arm["off"]), len(by_arm["on"]))
    mean_pos = {a: (sum(c["pos"] for c in v) / len(v)) if v else None for a, v in by_arm.items()}
    balanced = bool(n_min >= 1 and mean_pos["off"] == mean_pos["on"])
    shas = {c.get("sha") for c in checked if c.get("valid")}
    m = {}
    if all_valid and n_min >= 1:
        def arm_level(a, key):
            return _median([c[key] for c in by_arm[a]])
        m["M1_total_delta_s"] = arm_level("on", "median_total_s") - arm_level("off", "median_total_s")
        m["M2_wta_delta_s"] = arm_level("on", "median_wta_s") - arm_level("off", "median_wta_s")
        m["M3_non_wta_delta_s"] = arm_level("on", "median_non_wta_s") - arm_level("off", "median_non_wta_s")
        m["noise_s"] = max(max(c["median_total_s"] for c in v) - min(c["median_total_s"] for c in v)
                           for v in by_arm.values())
        m["arm_total_s"] = {a: arm_level(a, "median_total_s") for a in by_arm}
        m["arm_wta_s"] = {a: arm_level(a, "median_wta_s") for a in by_arm}
        # whose is the whole-turn delta? the part the WTA's own time accounts for, vs the rest
        m["fraction_of_total_delta_in_wta"] = attributable_to(
            "SETTLE warm-turn delta: total vs non-WTA remainder", m["M1_total_delta_s"], m["M3_non_wta_delta_s"])
    clear_pass = bool(m) and m["M2_wta_delta_s"] <= bound and m["M1_total_delta_s"] + m["noise_s"] <= bound
    clear_fail = bool(m) and (m["M2_wta_delta_s"] > bound or m["M1_total_delta_s"] - m["noise_s"] > bound)
    vd = Verdict("affect_marker_settle_gpu_timing_A3")
    vd.require("every process completed and validated: cupy, Qwen renderer, affective warm turns, WTA ran with the "
               "arm's expected config (lever), clean code", all_valid, expect=True,
               note="; ".join("#%d %s: %s" % (c["pos"], c["arm"], c["problems"][:2]) for c in checked if not c["valid"]))
    vd.require(">= 2 processes per arm", n_min, expect=lambda x: x >= 2)
    vd.require("order counterbalanced (equal mean position per arm)", balanced, expect=True, note=str(mean_pos))
    vd.require("one code revision across processes", len(shas), expect=1, note=str(sorted(s for s in shas if s)))
    vd.require("resolvable at the %.2f s bound given the measured noise" % bound,
               (clear_pass or clear_fail) if m else None, expect=True,
               note=("M1=%+.3f M2=%+.3f noise=%.3f" % (m["M1_total_delta_s"], m["M2_wta_delta_s"], m["noise_s"])
                     if m else ""))
    decided = vd.decide(clear_pass)
    return {"probe": "affect_marker_settle_gpu_timing", "amendment": "A3 (2026-09-24)", "bound_s": bound,
            "renderer_required": renderer_required, "status": decided["status"], "go": bool(decided["go"]),
            "metrics": m, "processes": checked, "verdict": decided, "preconditions": decided["preconditions"],
            "rule": "GO iff every precondition holds and M2 <= bound and M1 + noise <= bound; NO-GO iff they hold and "
                    "M2 > bound or M1 - noise > bound; otherwise UNDEFINED."}


# ─────────────────────────────────────────────────────── selftest ──────────────────────────────────────────────
def _proc(pos, arm, totals, wta, *, level=2, renderer="off-bridge Qwen-0.5B (spiking forward)", backend="cupy",
          cfg=None, dirty=(), sha="abc", n_reads=2, reader_cfg=None, http=200):
    cfg = cfg or expected_config(arm)
    reader_cfg = reader_cfg or cfg
    warm = [{"wall_s": t, "wta_s": w, "n_wta_reads": n_reads, "wta_configs": [list(cfg)] if n_reads else [],
             "level": level, "renderer": renderer, "http_status": http} for t, w in zip(totals, wta)]
    base = {"wall_s": 1.0, "renderer": renderer, "http_status": 200}
    return {"pos": pos, "arm": arm, "rec": {"backend": backend, "code": {"sha": sha, "dirty_tracked": list(dirty),
                                                                      "git_ok": True},
                                           "build_turn": base, "first_affective_turn": base, "warm_turns": warm,
                                           "readers": [{"key": "42", "settle": reader_cfg[0], "warmup": reader_cfg[1],
                                                        "washout": reader_cfg[2]}]}}


def _abba(off_t, on_t, off_w, on_w, **kw):
    """ABBA processes; *_t / *_w = per-process lists of 4 warm totals / WTA times."""
    return [_proc(0, "off", off_t[0], off_w[0], **kw), _proc(1, "on", on_t[0], on_w[0], **kw),
            _proc(2, "on", on_t[1], on_w[1], **kw), _proc(3, "off", off_t[1], off_w[1], **kw)]


def selftest() -> bool:
    ok = True

    def check(name, got):
        nonlocal ok
        ok = ok and bool(got)
        print("  %-84s %s" % (name, "ok" if got else "FAIL"))
    off_w = [[0.02] * 4, [0.02] * 4]
    on_w = [[0.16] * 4, [0.16] * 4]
    clean = _abba([[29.5] * 4, [29.6] * 4], [[29.6] * 4, [29.7] * 4], off_w, on_w)
    r = decide(clean)
    check("clean: M1=+0.10 noise=0.10 M2=+0.14 -> GO", r["status"] == "GO"
          and abs(r["metrics"]["M2_wta_delta_s"] - 0.14) < 1e-9)
    r = decide(_abba([[29.5] * 4, [29.6] * 4], [[29.6] * 4, [29.7] * 4], off_w, [[0.5] * 4, [0.5] * 4]))
    check("the WTA's own cost exceeds the bound (M2=+0.48) -> NO-GO", r["status"] == "NO-GO")
    r = decide(_abba([[29.5] * 4, [29.55] * 4], [[31.0] * 4, [31.05] * 4], off_w, on_w))
    check("whole warm turn clearly slower (M1=+1.50, noise 0.05) -> NO-GO", r["status"] == "NO-GO")
    r = decide(_abba([[28.0] * 4, [30.0] * 4], [[28.5] * 4, [30.5] * 4], off_w, on_w))
    check("noise 2.0 s swamps the bound -> UNDEFINED, never GO", r["status"] == "UNDEFINED")
    bad = list(clean)
    bad[1] = _proc(1, "on", [29.6] * 4, [0.02] * 4, cfg=expected_config("off"), reader_cfg=expected_config("off"))
    r = decide(bad)
    check("an ON process read with the OFF config (SETTLE never reached the reader) -> UNDEFINED",
          r["status"] == "UNDEFINED")
    bad = list(clean)
    bad[2] = _proc(2, "on", [29.7] * 4, [0.0] * 4, level=0, n_reads=0)
    check("a neutral warm turn (no WTA read, SETTLE cannot cost anything) -> UNDEFINED",
          decide(bad)["status"] == "UNDEFINED")
    check("stub renderer when the Qwen renderer is required -> UNDEFINED",
          decide(_abba([[29.5] * 4, [29.6] * 4], [[29.6] * 4, [29.7] * 4], off_w, on_w,
                       renderer="template-stub (GPU-free)"))["status"] == "UNDEFINED")
    check("dirty tracked file -> UNDEFINED",
          decide(_abba([[29.5] * 4, [29.6] * 4], [[29.6] * 4, [29.7] * 4], off_w, on_w,
                       dirty=[" M webapp/server.py"]))["status"] == "UNDEFINED")
    check("numpy backend -> UNDEFINED",
          decide(_abba([[29.5] * 4, [29.6] * 4], [[29.6] * 4, [29.7] * 4], off_w, on_w,
                       backend="numpy"))["status"] == "UNDEFINED")
    bad = list(clean)
    bad[3] = {"pos": 3, "arm": "off", "rec": None}
    check("a process that failed -> UNDEFINED", decide(bad)["status"] == "UNDEFINED")
    unbal = [_proc(0, "off", [29.5] * 4, [0.02] * 4), _proc(1, "off", [29.6] * 4, [0.02] * 4),
             _proc(2, "on", [29.6] * 4, [0.16] * 4), _proc(3, "on", [29.7] * 4, [0.16] * 4)]
    check("order OFF,OFF,ON,ON (not counterbalanced) -> UNDEFINED", decide(unbal)["status"] == "UNDEFINED")
    check("one process per arm -> UNDEFINED", decide(clean[:2])["status"] == "UNDEFINED")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


# ═════════════════════════════════ AMENDMENT 3: within-process crossover (see the docstring) ═════════════════════
def xo_arm_pattern(orient: str, runs: int) -> list:
    """Run arms of one process: ABBA repeated, A = `orient` (its first run's arm)."""
    other = "on" if orient == "off" else "off"
    return [orient if (r % 4) in (0, 3) else other for r in range(int(runs))]


def xo_turn_arms(orient: str, runs: int, run_len: int) -> list:
    pat = xo_arm_pattern(orient, runs)
    return [pat[k // int(run_len)] for k in range(int(runs) * int(run_len))]


def xo_message(k: int) -> str:
    """Turn k's message -- identical in every process, so every index carries the same input in both arms."""
    from research.runners._affect_marker_settle_multiturn_derisk import EMO_TEXT, EMO2_TEXT
    return (EMO_TEXT, EMO2_TEXT)[int(k) % 2]


def _sha1(s) -> str:
    return hashlib.sha1(str(s or "").encode("utf-8")).hexdigest()


def _worker_xo(orient: str, out_path: str, renderer: str, runs: int, run_len: int) -> int:
    """Subprocess entry: ONE tiny-demo brain; SETTLE toggled per turn; every turn timed and its work recorded."""
    os.environ.setdefault("SIM_BACKEND", "cupy")
    if renderer == "stub":                       # dev smoke only; the verdict requires the Qwen renderer
        os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
        os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ[SETTLE_ENV] = "0"                 # OFF-ARM DISCIPLINE: explicit "0"/"1" on every turn, never a pop
    import resource
    from research.runners._affect_marker_settle_multiturn_derisk import EMO_TEXT, EMO2_TEXT, NEU_TEXT
    import research.runners._affect_marker_wta_derisk as W
    t_proc0 = time.perf_counter()
    other = "on" if orient == "off" else "off"
    trace, gen_trace = [], []
    orig_select = W.AffectMarkerWTA._select

    def timed_select(self, *a, **k):             # instrument only: times the spiking WTA read, changes nothing
        t0 = time.perf_counter()
        err = False
        try:
            return orig_select(self, *a, **k)
        except Exception:
            err = True
            raise
        finally:
            trace.append((time.perf_counter() - t0, bool(self.settle), int(self.warmup), int(self.washout), err))
    W.AffectMarkerWTA._select = timed_select
    import webapp.server as S
    qwen_wrapped = False
    if renderer == "qwen":
        rend = S._get_warm_qwen_renderer()       # what the server's startup warm does; the model loads once
        model = rend._fac.model
        orig_gen = model.generate

        def timed_generate(*a, **k):             # instrument only: times + counts every Qwen generate call
            t0 = time.perf_counter()
            out = orig_gen(*a, **k)
            dt = time.perf_counter() - t0
            ids = k.get("input_ids")
            if ids is None and a:
                ids = a[0]
            try:
                in_len, batch = int(ids.shape[1]), int(out.shape[0])
                new_len = int(out.shape[1]) - in_len
            except Exception:
                in_len = batch = new_len = -1
            gen_trace.append((dt, batch, new_len, in_len))
            return out
        model.generate = timed_generate
        qwen_wrapped = True
    try:
        import cupy as _cp
        pool = _cp.get_default_memory_pool()
    except Exception:
        pool = None

    def _turn(msg, reset, arm):
        os.environ[SETTLE_ENV] = "1" if arm == "on" else "0"
        n0, g0 = len(trace), len(gen_trace)
        load1 = os.getloadavg()[0]
        c0 = time.process_time()
        t0 = time.perf_counter()
        r = S.brain_chat(S.BrainChatRequest(session=GPU_SESSION, message=msg, brain="tiny-demo", renderer=renderer,
                                            reset=reset))
        wall = time.perf_counter() - t0
        cpu = time.process_time() - c0
        body = json.loads(bytes(r.body))
        sel, gens = trace[n0:], gen_trace[g0:]
        ad = body.get("affect_drives") or {}
        lead = ad.get("lead", "") or ""
        ans = body.get("answer") or ""
        core = ans.replace(lead, "", 1) if lead else ans      # the reply minus this turn's affect lead
        return {"arm": arm, "msg_sha": _sha1(msg), "wall_s": wall, "cpu_s": cpu, "load1_start": load1,
                "http_status": int(getattr(r, "status_code", 0) or 0),
                "wta_s": sum(x[0] for x in sel), "n_wta_reads": len(sel),
                "wta_configs": sorted({(x[1], x[2], x[3]) for x in sel}), "wta_errors": sum(1 for x in sel if x[4]),
                "render_s": sum(g[0] for g in gens), "n_gen_calls": len(gens),
                "gen_calls": [[g[1], g[2], g[3]] for g in gens],
                "level": ad.get("level"), "lead": lead, "reason": ad.get("reason"),
                "renderer": body.get("renderer"), "abstained": body.get("abstained"),
                "answer_len": len(ans), "answer_core_sha": _sha1(core),
                "cupy_pool_used_bytes": int(pool.used_bytes()) if pool is not None else None,
                "maxrss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}

    arms = xo_turn_arms(orient, runs, run_len)
    rec = {"amendment": "A3 amendment 3 (2026-09-25): within-process crossover", "orient": orient,
           "seed": os.environ.get("BRAIN_CHAT_SEED"), "renderer_requested": renderer,
           "sim_disable_llm": os.environ.get("SIM_DISABLE_LLM"), "code": _code_state(),
           "plan": {"orient": orient, "runs": int(runs), "run_len": int(run_len), "arms": arms,
                    "msg_sha": [_sha1(xo_message(k)) for k in range(len(arms))]},
           "qwen_generate_wrapped": qwen_wrapped,
           "projected_seconds": XO_EST_BUILD_S + len(arms) * XO_EST_TURN_S,
           "complete": False, "build_turn": None, "warmup_turns": [], "run_turns": [], "readers": []}

    def _dump():
        try:
            from sim.backend import get_backend
            rec["backend"] = get_backend()[1]
        except Exception as e:
            rec["backend"] = "unknown (%s)" % type(e).__name__
        rec["readers"] = [{"key": str(k), "settle": bool(v.settle), "warmup": int(v.warmup), "washout": int(v.washout)}
                          for k, v in W._READERS.items()]
        rec["process_wall_s"] = time.perf_counter() - t_proc0
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        os.replace(tmp, out_path)

    try:
        rec["build_turn"] = _turn(NEU_TEXT, True, "off")
        rec["warmup_turns"] = [_turn(EMO_TEXT, False, orient), _turn(EMO2_TEXT, False, other)]
        _dump()
        for k, arm in enumerate(arms):
            t = _turn(xo_message(k), False, arm)
            t.update({"index": k, "run": k // run_len, "pos_in_run": k % run_len,
                      "scored_planned": (k % run_len) != 0})
            rec["run_turns"].append(t)
            if k % run_len == run_len - 1:
                _dump()                          # outside the timed turn; a crash keeps what ran
                print("[xo worker %s] run %d/%d %s wall=%s" % (orient, k // run_len + 1, runs, arm,
                      ["%.2f" % x["wall_s"] for x in rec["run_turns"][-run_len:]]), flush=True)
        rec["complete"] = True
    except Exception as e:                       # the partial record is the diagnosis; the verdict rejects it
        rec["aborted"] = "%s: %s" % (type(e).__name__, e)
    finally:
        os.environ[SETTLE_ENV] = "0"
        _dump()
    print("[xo worker %s] complete=%s -> %s" % (orient, rec["complete"], out_path), flush=True)
    return 0 if rec["complete"] else 1


def _xo_reusable(path: str, orient: str, runs: int, run_len: int, head_sha) -> bool:
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception:
        return False
    plan = rec.get("plan") or {}
    return bool(rec.get("complete") and head_sha and (rec.get("code") or {}).get("sha") == head_sha
                and plan.get("orient") == orient and plan.get("runs") == runs and plan.get("run_len") == run_len)


def _spawn_xo(seed: int, orient: str, out_path: str, renderer: str, runs: int, run_len: int):
    if os.path.exists(out_path):                 # never read a stale record as this run's
        os.replace(out_path, out_path + ".stale")
    host_env = dict(os.environ)
    host_env["BRAIN_CHAT_SEED"] = str(int(seed))
    host_env.setdefault("SIM_BACKEND", "cupy")
    try:
        p = subprocess.run([sys.executable, "-u", "-m", "research.runners._affect_marker_settle_gpu_timing",
                            "--xo-worker", "--orient", orient, "--renderer", renderer, "--runs", str(runs),
                            "--run-len", str(run_len), "--out", out_path], env=host_env, timeout=XO_PROC_TIMEOUT_S)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    if not os.path.exists(out_path):
        return None
    with open(out_path) as f:
        rec = json.load(f)
    rec["worker_returncode"] = rc
    return rec


def run_xo(seed: int, orients, out_dir: str, renderer: str, runs: int, run_len: int) -> list:
    sdir = os.path.join(out_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    head = _code_state().get("sha")
    procs = []
    for i, o in enumerate(orients):
        path = os.path.join(sdir, "xo_%02d_%s.json" % (i, o))
        if _xo_reusable(path, o, runs, run_len, head):
            with open(path) as f:
                rec = json.load(f)
            rec.setdefault("worker_returncode", 0)
            print("  s%d #%d orient=%s -> REUSED complete record at HEAD %s" % (seed, i, o, head), flush=True)
        else:
            rec = _spawn_xo(seed, o, path, renderer, runs, run_len)
            print("  s%d #%d orient=%s -> %s" % (seed, i, o, "OK" if rec and rec.get("complete") else "FAILED"),
                  flush=True)
        procs.append({"pos": i, "orient": o, "rec": rec})
    return procs


def load_xo(raw_dir: str, seed: int, orients) -> list:
    procs = []
    for i, o in enumerate(orients):
        path = os.path.join(raw_dir, "s%d" % seed, "xo_%02d_%s.json" % (i, o))
        rec = None
        if os.path.exists(path):
            with open(path) as f:
                rec = json.load(f)
        procs.append({"pos": i, "orient": o, "rec": rec})
    return procs


def check_process_xo(p: dict, renderer_required: str) -> dict:
    orient, rec = p["orient"], p.get("rec")
    if not rec:
        return {"pos": p["pos"], "orient": orient, "valid": False, "problems": ["process failed or wrote nothing"]}
    probs = []
    other = "on" if orient == "off" else "off"
    if not rec.get("complete") or rec.get("aborted"):
        probs.append("record incomplete (aborted=%r)" % rec.get("aborted"))
    if rec.get("worker_returncode") not in (0, None):
        probs.append("worker exit %r" % rec.get("worker_returncode"))
    if rec.get("backend") != "cupy":
        probs.append("backend %r, not cupy" % rec.get("backend"))
    code = rec.get("code") or {}
    if not code.get("git_ok") or code.get("dirty_tracked"):
        probs.append("code not clean: %s" % (code.get("dirty_tracked") or "git state unreadable"))
    plan = rec.get("plan") or {}
    runs, run_len = plan.get("runs"), plan.get("run_len")
    if plan.get("orient") != orient or not isinstance(runs, int) or runs < 4 or runs % 4 \
            or not isinstance(run_len, int) or run_len < 2:
        probs.append("plan invalid for orient %s: %s" % (orient, {k: plan.get(k) for k in ("orient", "runs", "run_len")}))
        exp_arms = []
    else:
        exp_arms = xo_turn_arms(orient, runs, run_len)
    turns = list(rec.get("run_turns") or [])
    if exp_arms and len(turns) != len(exp_arms):
        probs.append("%d run turns, planned %d" % (len(turns), len(exp_arms)))
    warm = list(rec.get("warmup_turns") or [])
    if [t.get("arm") for t in warm] != [orient, other]:
        probs.append("warm-up arms %s, expected %s" % ([t.get("arm") for t in warm], [orient, other]))
    for t in [rec.get("build_turn") or {}] + warm + turns:
        if t.get("http_status") != 200:
            probs.append("a turn returned HTTP %r" % t.get("http_status"))
        if renderer_required not in str(t.get("renderer") or "").lower():
            probs.append("reply renderer %r is not %r" % (t.get("renderer"), renderer_required))
    for k, t in enumerate(turns):
        if exp_arms and k < len(exp_arms) and (t.get("arm") != exp_arms[k] or t.get("index") != k):
            probs.append("turn %d arm %r index %r, planned arm %r" % (k, t.get("arm"), t.get("index"), exp_arms[k]))
        if t.get("level") not in (None, 0):
            exp = expected_config(t.get("arm"))
            if (t.get("n_wta_reads") or 0) < 1:
                probs.append("affective turn %d made no WTA read" % k)
            cfgs = {tuple(c) for c in (t.get("wta_configs") or [])}
            if cfgs and cfgs != {exp}:
                probs.append("turn %d (%s) read with config %s, expected %s" % (k, t.get("arm"), sorted(cfgs), exp))
            if t.get("wta_errors"):
                probs.append("turn %d: %d WTA reader exception(s)" % (k, t.get("wta_errors")))
    rcfg = sorted((bool(r.get("settle")), r.get("warmup"), r.get("washout")) for r in (rec.get("readers") or []))
    if rcfg != sorted([expected_config("on"), expected_config("off")]):
        probs.append("cached readers %s, expected one per arm %s" % (rcfg, [expected_config("off"), expected_config("on")]))
    uniq = []
    for q in probs:                              # a failure repeated on 190 turns is one problem
        if q not in uniq:
            uniq.append(q)
    return {"pos": p["pos"], "orient": orient, "valid": not uniq, "problems": uniq[:20], "n_problems": len(uniq),
            "sha": code.get("sha"), "plan": (plan.get("runs"), plan.get("run_len"))}


def _fe_fit(rows: list, key: str, *, alpha: float = XO_ALPHA, extra=None):
    """OLS  y = process FE + slot FE + delta*ON (+ gamma*extra).  None if ON is not identifiable or no df is left."""
    import numpy as np
    from scipy import stats
    if not rows:
        return None
    procs = sorted({r["proc"] for r in rows})
    slots = sorted({r["slot"] for r in rows})
    n_fe = len(procs) + len(slots) - 1
    cols = n_fe + 1 + (1 if extra else 0)
    X = np.zeros((len(rows), cols))
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        X[i, procs.index(r["proc"])] = 1.0
        j = slots.index(r["slot"])
        if j > 0:
            X[i, len(procs) + j - 1] = 1.0
        X[i, n_fe] = 1.0 if r["on"] else 0.0
        if extra:
            X[i, n_fe + 1] = 1.0 if r[extra] else 0.0
        y[i] = float(r[key])
    rank = int(np.linalg.matrix_rank(X))
    if rank < cols or rank == int(np.linalg.matrix_rank(np.delete(X, n_fe, axis=1))):
        return None                              # the ON (or extra) column is collinear with the fixed effects
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    df = len(rows) - rank
    if df < 1:
        return None
    s2 = float(resid @ resid) / df
    cov = s2 * np.linalg.pinv(X.T @ X)
    t = float(stats.t.ppf(1.0 - alpha, df))
    out = {"df": df, "t": t, "n": len(rows), "resid_sd": math.sqrt(s2)}
    for name, idx in [("delta", n_fe)] + ([("gamma", n_fe + 1)] if extra else []):
        est, se = float(beta[idx]), math.sqrt(max(float(cov[idx, idx]), 0.0))
        out[name] = {"est": est, "se": se, "lower": est - t * se, "upper": est + t * se}
    return out


def decide_xo(procs: list, *, renderer_required: str = "qwen", bound: float = BOUND_S,
              min_kept: float = XO_MIN_KEPT_FRAC) -> dict:
    from tools.verdict import Verdict
    checked = [check_process_xo(p, renderer_required) for p in procs]
    all_valid = bool(checked) and all(c["valid"] for c in checked)
    n_orient = {o: sum(1 for p in procs if p["orient"] == o) for o in ("off", "on")}
    balanced = n_orient["off"] == n_orient["on"] and n_orient["off"] >= 2
    shas = {c.get("sha") for c in checked if c["valid"]}
    plans = {c.get("plan") for c in checked if c["valid"]}
    m, diag = {}, {}
    same_inputs = kept_frac = None
    if all_valid and len(plans) == 1:
        recs = [p["rec"] for p in procs]
        runs, run_len = next(iter(plans))
        K = runs * run_len
        T = [r["run_turns"] for r in recs]
        same_inputs = all(len({T[i][k].get("msg_sha") for i in range(len(recs))}) == 1 for k in range(K)) and \
            len({tuple((r.get("plan") or {}).get("msg_sha") or ()) for r in recs}) == 1
        affective = [k for k in range(K) if all(T[i][k].get("level") not in (None, 0) for i in range(len(recs)))]
        aff_set = set(affective)
        planned = [k for k in range(K) if k % run_len != 0]
        kept = [k for k in planned if k in aff_set]
        kept_frac = len(kept) / float(len(planned)) if planned else 0.0
        diag["n_planned_scored"], diag["n_kept_scored"] = len(planned), len(kept)
        diag["indices_level_differs_across_processes"] = [
            k for k in range(K) if len({T[i][k].get("level") for i in range(len(recs))}) > 1][:50]
        rows = []
        for i in range(len(recs)):
            for run in range(runs):
                ks = [k for k in kept if k // run_len == run]
                if not ks:
                    continue
                ts = [T[i][k] for k in ks]
                tot = [t["wall_s"] for t in ts]
                wta = [t["wta_s"] for t in ts]
                ren = [float(t.get("render_s") or 0.0) for t in ts]
                rows.append({"proc": i, "slot": run, "on": ts[0]["arm"] == "on",
                             "total": sum(tot) / len(ts), "wta": sum(wta) / len(ts), "render": sum(ren) / len(ts),
                             "rest": sum(a - b - c for a, b, c in zip(tot, wta, ren)) / len(ts)})
        fits = {key: _fe_fit(rows, key) for key in ("total", "wta", "render", "rest")}
        if fits["total"] and fits["wta"]:
            m["M1_total"] = fits["total"]["delta"]
            m["M2_wta"] = fits["wta"]["delta"]
            m["M4_render"] = (fits["render"] or {}).get("delta")
            m["M3_rest"] = (fits["rest"] or {}).get("delta")
            m["fit"] = {k: {kk: v[kk] for kk in ("df", "t", "n", "resid_sd")} for k, v in fits.items() if v}
            m["arm_mean_total_s"] = {a: sum(r["total"] for r in rows if r["on"] == (a == "on")) /
                                     max(1, sum(1 for r in rows if r["on"] == (a == "on"))) for a in ("off", "on")}
        # ---- diagnostics (reported, never gated) ----
        sig_on, sig_off, differs, nondet = {}, {}, 0, 0
        tok = {"off": 0, "on": 0}
        for k in kept:
            s = {"on": set(), "off": set()}
            for i in range(len(recs)):
                t = T[i][k]
                s[t["arm"]].add((t.get("n_gen_calls"), json.dumps(t.get("gen_calls")), t.get("answer_core_sha"),
                                 t.get("reason"), t.get("abstained")))
                tok[t["arm"]] += sum(max(0, c[0]) * max(0, c[1]) for c in (t.get("gen_calls") or []) if len(c) >= 2)
            differs += int(s["on"] != s["off"])
            nondet += int(len(s["on"]) > 1 or len(s["off"]) > 1)
        diag["work_identity"] = {"n_kept_indices": len(kept), "n_indices_on_vs_off_work_differs": differs,
                                 "n_indices_work_differs_within_an_arm": nondet,
                                 "generated_tokens_total_by_arm": tok}
        aff_rows = []
        for i in range(len(recs)):
            prev = "on" if procs[i]["orient"] == "off" else "off"   # the second warm-up turn's arm
            for k in range(K):
                t = T[i][k]
                if k in aff_set:
                    aff_rows.append({"proc": i, "slot": k, "on": t["arm"] == "on", "prev_on": prev == "on",
                                     "total": t["wall_s"]})
                prev = t["arm"]
        co = _fe_fit(aff_rows, "total", extra="prev_on")
        diag["carryover_lag1_all_affective_turns"] = (
            {"delta_same_turn": co["delta"], "gamma_prev_turn_on": co["gamma"], "df": co["df"]} if co else None)
        by_proc = {}
        for r in rows:
            by_proc.setdefault(r["proc"], []).append(r["total"])
        pm = {pp: sum(v) / len(v) for pp, v in by_proc.items()}
        slot_d = []
        for run in sorted({r["slot"] for r in rows}):
            on_v = [r["total"] - pm[r["proc"]] for r in rows if r["slot"] == run and r["on"]]
            off_v = [r["total"] - pm[r["proc"]] for r in rows if r["slot"] == run and not r["on"]]
            if on_v and off_v:
                slot_d.append(sum(on_v) / len(on_v) - sum(off_v) / len(off_v))
        diag["slot_median_total_delta_process_demeaned_s"] = _median(slot_d)
        loads = {"off": [], "on": []}
        for i in range(len(recs)):
            for k in kept:
                t = T[i][k]
                if t.get("load1_start") is not None:
                    loads[t["arm"]].append(float(t["load1_start"]))
        diag["mean_load1_by_arm"] = {a: (sum(v) / len(v) if v else None) for a, v in loads.items()}
        diag["process_mean_total_s"] = {str(procs[pp]["pos"]): v for pp, v in pm.items()}
    resolvable = None
    clear_pass = clear_fail = False
    if m:
        U1, L1 = m["M1_total"]["upper"], m["M1_total"]["lower"]
        U2, L2 = m["M2_wta"]["upper"], m["M2_wta"]["lower"]
        clear_pass = U2 <= bound and U1 <= bound
        clear_fail = L2 > bound or L1 > bound
        resolvable = bool(clear_pass or clear_fail)
    vd = Verdict("affect_marker_settle_gpu_timing_A3_amendment3")
    vd.require("every process completed and validated: cupy, Qwen renderer, HTTP 200, clean code, planned arm on "
               "every turn, one cached reader per arm, every affective turn read the WTA with its arm's config "
               "(lever) and no reader exception", all_valid, expect=True,
               note="; ".join("#%d %s: %s" % (c["pos"], c["orient"], c["problems"][:2]) for c in checked
                              if not c["valid"]))
    vd.require(">= 2 processes per orientation, equal counts (mirrored)", balanced, expect=True, note=str(n_orient))
    vd.require("one code revision across processes", len(shas), expect=1, note=str(sorted(s for s in shas if s)))
    vd.require("one plan (runs, run length) across processes", len(plans), expect=1, note=str(sorted(plans)))
    vd.require("the same message at every turn index in every process", same_inputs, expect=True)
    vd.require("kept affective scored indices >= %.0f%% of planned" % (100 * min_kept), kept_frac,
               expect=lambda x: x >= min_kept, note=str(diag.get("n_kept_scored")))
    vd.require("resolvable at the %.2f s bound (both one-sided 95%% bounds in the PASS or the FAIL region)" % bound,
               resolvable, expect=True,
               note=("M1 %+.3f [%+.3f, %+.3f]  M2 %+.3f [%+.3f, %+.3f]" % (
                   m["M1_total"]["est"], m["M1_total"]["lower"], m["M1_total"]["upper"],
                   m["M2_wta"]["est"], m["M2_wta"]["lower"], m["M2_wta"]["upper"]) if m else ""))
    decided = vd.decide(clear_pass)
    backends = sorted({str((p.get("rec") or {}).get("backend")) for p in procs})
    return {"probe": "affect_marker_settle_gpu_timing", "amendment": "A3 amendment 3 (2026-09-25)",
            "design": "within-process crossover", "bound_s": bound, "renderer_required": renderer_required,
            "sim_backend": backends, "status": decided["status"], "go": bool(decided["go"]), "metrics": m,
            "diagnostics": diag, "processes": checked, "verdict": decided, "preconditions": decided["preconditions"],
            "rule": "GO iff every precondition holds and U(M2) <= bound and U(M1) <= bound; NO-GO iff they hold and "
                    "L(M2) > bound or L(M1) > bound; otherwise UNDEFINED. U/L = one-sided 95% bounds of the ON "
                    "coefficient of a run-level OLS with process and run-slot fixed effects."}


def _xo_proc(pos, orient, *, runs=8, run_len=4, proc_off=0.0, idx_eff=None, wta_on=0.15, wta_off=0.02,
             on_extra=0.0, carry=0.0, noise=0.1, seed=0, neutral=(), renderer="off-bridge Qwen-0.5B (spiking forward)",
             backend="cupy", dirty=(), sha="abc", bad_cfg_turn=None, zero_read_turn=None, msg_tag="",
             readers=None, complete=True, flip_arm_turn=None, err_turn=None):
    """A synthetic process record for the selftest. wall = 30 + proc_off + idx_eff[k] + WTA + ON-extra
    + carry (if the previous turn was ON) + noise. Levels are 3 except the `neutral` indices (0)."""
    import numpy as np
    rng = np.random.default_rng(1000 + seed)
    arms = xo_turn_arms(orient, runs, run_len)
    other = "on" if orient == "off" else "off"
    prev_on = other == "on"
    turns = []
    for k, arm in enumerate(arms):
        on = arm == "on"
        lvl = 0 if k in neutral else 3
        wta = (wta_on if on else wta_off) if lvl else 0.0
        eff = idx_eff[k] if idx_eff is not None else 0.0
        wall = 30.0 + proc_off + eff + wta + (on_extra if on else 0.0) + (carry if prev_on else 0.0) + \
            noise * float(rng.standard_normal())
        cfg = expected_config(("off" if on else "on") if bad_cfg_turn == k else arm)
        n_reads = 0 if (not lvl or zero_read_turn == k) else 2
        turns.append({"index": k, "run": k // run_len, "pos_in_run": k % run_len, "scored_planned": k % run_len != 0,
                      "arm": ("off" if on else "on") if flip_arm_turn == k else arm,
                      "msg_sha": _sha1("m%d%s" % (k % 2, msg_tag if k == 5 else "")),
                      "wall_s": wall, "wta_s": wta, "n_wta_reads": n_reads,
                      "wta_configs": [list(cfg)] if n_reads else [], "wta_errors": 1 if err_turn == k else 0,
                      "render_s": 1.0, "n_gen_calls": 1, "gen_calls": [[1, 12, 40]], "level": lvl,
                      "lead": "Wonderful! " if lvl else "", "reason": "graded_affect", "abstained": True,
                      "renderer": renderer, "http_status": 200, "answer_core_sha": "x", "load1_start": 1.0})
        prev_on = on
    base = {"http_status": 200, "renderer": renderer, "level": 0, "n_wta_reads": 0, "wta_configs": []}
    rd = readers if readers is not None else [
        {"key": "42", "settle": False, "warmup": expected_config("off")[1], "washout": expected_config("off")[2]},
        {"key": "(42, 'settle')", "settle": True, "warmup": expected_config("on")[1], "washout": expected_config("on")[2]}]
    rec = {"orient": orient, "backend": backend, "code": {"sha": sha, "dirty_tracked": list(dirty), "git_ok": True},
           "plan": {"orient": orient, "runs": runs, "run_len": run_len, "arms": arms,
                    "msg_sha": [_sha1("m%d" % (k % 2)) for k in range(len(arms))]},
           "complete": complete, "worker_returncode": 0, "qwen_generate_wrapped": True,
           "build_turn": dict(base, arm="off"), "warmup_turns": [dict(base, arm=orient), dict(base, arm=other)],
           "run_turns": turns, "readers": rd}
    return {"pos": pos, "orient": orient, "rec": rec}


def _xo_quad(orients=("off", "on", "on", "off"), offs=(4.7, -0.4, -1.8, 0.0), **kw):
    """Four synthetic processes sharing the same per-index work (idx_eff), A3-sized process offsets by default."""
    runs, run_len = kw.get("runs", 8), kw.get("run_len", 4)
    idx = kw.pop("idx_eff", None)
    if idx is None:
        idx = [2.0 * math.sin(0.7 * k) for k in range(64 * run_len)]   # long enough for any plan here
    per = kw.pop("per_proc", {})
    out = []
    for i, o in enumerate(orients):
        a = dict(kw)
        a.update(per.get(i, {}))
        out.append(_xo_proc(i, o, proc_off=offs[i % len(offs)], idx_eff=idx, seed=i, **a))
    return out


def selftest_xo() -> bool:
    ok = True

    def check(name, got):
        nonlocal ok
        ok = ok and bool(got)
        print("  %-92s %s" % (name, "ok" if got else "FAIL"))

    def st(procs, **kw):
        r = decide_xo(procs, **kw)
        return r["status"], r

    s, r = st(_xo_quad())
    check("clean, A3-sized process offsets (4.7 s spread): M2~0.13, no other cost -> GO",
          s == "GO" and abs(r["metrics"]["M2_wta"]["est"] - 0.13) < 1e-6
          and abs(r["metrics"]["M1_total"]["est"] - 0.13) < 0.1)
    s, _ = st(_xo_quad(offs=(9.0, -6.0, 3.0, -2.0)))
    check("process offsets of 15 s cancel inside each process -> still GO", s == "GO")
    s, _ = st(_xo_quad(wta_on=0.6))
    check("the WTA's own cost exceeds the bound (M2 +0.58) -> NO-GO", s == "NO-GO")
    # M2 is gated on its own, not only through M1: a WTA over the bound whose ON turns are otherwise faster keeps M1
    # inside the bound, so only the M2 comparisons can read NO-GO here. (Added 2026-09-25 after a mutation check:
    # dropping U2 from PASS or L2 from FAIL left every earlier case passing, because each M2 case also moved M1.)
    s, r = st(_xo_quad(wta_on=0.6, on_extra=-0.6))
    check("the WTA costs +0.58 but the rest of the ON turn is 0.6 s faster (M1 ~-0.02) -> NO-GO, never GO",
          s == "NO-GO" and r["metrics"]["M1_total"]["upper"] <= BOUND_S)
    s, _ = st(_xo_quad(on_extra=1.0))
    check("ON turns carry +1.0 s outside the WTA (M1 +1.13) -> NO-GO", s == "NO-GO")
    s, r = st(_xo_quad(carry=0.8))
    check("SETTLE adds +0.8 s to the NEXT turn: the washout charges it to ON (M1 ~+0.93) -> NO-GO",
          s == "NO-GO" and r["metrics"]["M1_total"]["est"] > 0.6)
    s, _ = st(_xo_quad(on_extra=0.17, noise=0.3))
    check("true cost at the bound (0.30 s) with 0.3 s turn noise -> UNDEFINED, never GO", s == "UNDEFINED")
    s, _ = st(_xo_quad(noise=5.0))
    check("5 s turn noise swamps the bound -> UNDEFINED, never GO", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={1: {"bad_cfg_turn": 2}}))           # process 1 runs ON first
    check("an ON turn read with the OFF config (SETTLE did not reach the reader) -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={2: {"zero_read_turn": 6}}))
    check("an affective turn made no WTA read -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={0: {"err_turn": 7}}))
    check("a WTA reader exception on a turn -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={3: {"flip_arm_turn": 10}}))
    check("a turn ran the wrong arm for the plan -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={1: {"readers": [{"key": "42", "settle": False, "warmup": 60, "washout": 40}]}}))
    check("only one cached reader (an arm never built its own) -> UNDEFINED", s == "UNDEFINED")
    s, r = st(_xo_quad(neutral=(1, 2)))
    check("2 of 24 scored indices neutral in every process: dropped for all, still decides -> GO",
          s == "GO" and r["diagnostics"]["n_kept_scored"] == 22)
    s, _ = st(_xo_quad(neutral=tuple(range(1, 32, 3))))
    check(">20% of scored indices neutral -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(renderer="template-stub (GPU-free)"))
    check("stub renderer when the Qwen renderer is required -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(dirty=[" M webapp/server.py"]))
    check("dirty tracked file -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(backend="numpy"))
    check("numpy backend -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(complete=False))
    check("incomplete (aborted) process records -> UNDEFINED", s == "UNDEFINED")
    bad = _xo_quad()
    bad[3] = {"pos": 3, "orient": "off", "rec": None}
    check("a process that failed -> UNDEFINED", st(bad)[0] == "UNDEFINED")
    s, _ = st(_xo_quad(orients=("off", "off", "off", "on")))
    check("orientations off,off,off,on (not mirrored) -> UNDEFINED", s == "UNDEFINED")
    check("one process per orientation -> UNDEFINED", st(_xo_quad()[:2])[0] == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={2: {"sha": "def"}}))
    check("two code revisions -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={1: {"msg_tag": "x"}}))
    check("a different message at one index in one process -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(per_proc={0: {"runs": 12}}))
    check("processes ran different plans -> UNDEFINED", s == "UNDEFINED")
    s, _ = st(_xo_quad(orients=("on", "off", "off", "on"), wta_on=0.02))
    check("SETTLE costs nothing at all (orientations reversed) -> GO", s == "GO")
    print("SELFTEST (amendment 3)", "PASS" if ok else "FAIL")
    return bool(ok)


# ─────────────────────────────── decomposition of the A3 run (what made it UNDEFINED) ───────────────────────────
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 8: 2.847, 10: 3.078}   # E[range]/sigma, normal samples


def decompose_a3(raw_dir: str, seed: int = 42, order=None, bound: float = BOUND_S) -> dict:
    """Read the A3 per-process records and split their noise: process level vs turn level vs turn index. Pure."""
    import statistics as stt
    from scipy import stats
    import research.runners._affect_marker_wta_derisk as W
    order = order or [x for x in DEFAULT_ORDER.split(",")]
    recs = []
    for i, arm in enumerate(order):
        with open(os.path.join(raw_dir, "s%d" % seed, "%02d_%s.json" % (i, arm))) as f:
            recs.append((i, arm, json.load(f)))
    procs = []
    X = []
    for i, arm, r in recs:
        w = r["warm_turns"]
        tot = [t["wall_s"] for t in w]
        wta = [t["wta_s"] for t in w]
        X.append(tot)
        procs.append({"pos": i, "arm": arm, "build_turn_s": r["build_turn"]["wall_s"],
                      "first_affective_turn_s": r["first_affective_turn"]["wall_s"], "warm_total_s": tot,
                      "warm_wta_s": wta, "warm_non_wta_s": [a - b for a, b in zip(tot, wta)],
                      "median_total_s": stt.median(tot), "mean_total_s": stt.mean(tot), "sd_total_s": stt.stdev(tot),
                      "median_wta_s": stt.median(wta), "wta_s_per_read": [t["wta_s"] / t["n_wta_reads"] for t in w],
                      "n_wta_reads": [t["n_wta_reads"] for t in w], "leads": [t["lead"] for t in w],
                      "first_minus_warm_median_s": r["first_affective_turn"]["wall_s"] - stt.median(tot),
                      "warm_median_over_build": stt.median(tot) / r["build_turn"]["wall_s"]})

    def two_way(rows):
        p_, t_ = len(rows), len(rows[0])
        g = sum(map(sum, rows)) / (p_ * t_)
        pm = [sum(x) / t_ for x in rows]
        tm = [sum(rows[a][b] for a in range(p_)) / p_ for b in range(t_)]
        res = [[rows[a][b] - pm[a] - tm[b] + g for b in range(t_)] for a in range(p_)]
        ss = sum(v * v for x in res for v in x)
        df = (p_ - 1) * (t_ - 1)
        return {"process_means_s": pm, "turn_index_means_s": tm, "grand_mean_s": g, "residual_ss": ss,
                "residual_df": df, "sigma_turn_s": math.sqrt(ss / df),
                "residual_ss_by_process": [sum(v * v for v in x) for x in res]}
    tw_all = two_way(X)
    tw_wo0 = two_way(X[1:])
    med = {p["pos"]: p["median_total_s"] for p in procs}
    by_arm = {a: [med[p["pos"]] for p in procs if p["arm"] == a] for a in ("off", "on")}
    sd_arm = {a: abs(v[0] - v[1]) / math.sqrt(2) for a, v in by_arm.items()}
    sd_pool = math.sqrt((sd_arm["off"] ** 2 + sd_arm["on"] ** 2) / 2)
    with open(os.path.join(raw_dir, "verdict.json")) as f:
        vd = json.load(f)
    m2 = vd["metrics"]["M2_wta_delta_s"]
    margin = bound - m2
    z95, z80 = float(stats.norm.ppf(0.95)), float(stats.norm.ppf(0.80))
    between = {"%s_margin_%.3f" % (lab, mg): 2 * (z * sd_pool / mg) ** 2
               for mg in (bound, margin) for lab, z in (("z95", z95), ("z95_power80", z95 + z80))}
    xo = {}
    for lab, sig in (("sigma_all", tw_all["sigma_turn_s"]), ("sigma_without_pos0", tw_wo0["sigma_turn_s"])):
        xo[lab] = {"sigma_turn_s": sig,
                   "scored_turns_needed_power80": 4 * sig ** 2 * ((z95 + z80) / margin) ** 2,
                   "half_width_at_%d_scored" % (4 * XO_RUNS * (XO_RUN_LEN - 1)):
                       z95 * 2 * sig / math.sqrt(4 * XO_RUNS * (XO_RUN_LEN - 1))}
    steps = {"off": W.WARMUP_STEPS + W.RUN_STEPS + W.WASHOUT_STEPS,
             "on": W.DELIBERATION_MS + W.RUN_STEPS + W.INTERTURN_REST_MS}
    per_read = {a: stt.mean([x for p in procs if p["arm"] == a for x in p["wta_s_per_read"]]) for a in ("off", "on")}
    keys = sorted({k for _i, _a, r in recs for t in r["warm_turns"] for k in t})
    return {"probe": "affect_marker_settle_gpu_timing", "analysis": "A3 decomposition (2026-09-25)",
            "runner": "research/runners/_affect_marker_settle_gpu_timing.py --decompose-a3", "seeds": [seed],
            "sim_backend": "none: pure analysis of recorded artifacts, no simulation ran",
            "analyzed_run_backend": sorted({r.get("backend") for _i, _a, r in recs}), "bound_s": bound,
            "M2_wta_delta_s": m2,
            "margin_for_non_wta_s": margin, "processes": procs, "two_way_all": tw_all,
            "two_way_without_pos0": tw_wo0, "within_arm_process_median_sd_s": sd_arm,
            "pooled_process_median_sd_s": sd_pool,
            "a3_noise_is_a_range_expected_s": {str(n): d * sd_pool for n, d in _D2.items()},
            "between_process_n_per_arm_if_noise_were_an_se": between, "within_process_crossover": xo,
            "wta_steps_per_read": steps, "wta_s_per_read_by_arm": per_read,
            "wta_s_per_step_by_arm": {a: per_read[a] / steps[a] for a in steps},
            "per_turn_fields_recorded": keys,
            "render_fields_recorded": any(k in keys for k in ("render_s", "n_gen_calls", "gen_calls", "answer_len"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", default=" ".join(str(s) for s in VERIFY_SEEDS))
    ap.add_argument("--order", default=DEFAULT_ORDER, help="comma list of off/on, one fresh process each")
    ap.add_argument("--renderer", default="qwen", choices=("qwen", "stub"),
                    help="'qwen' is the production default path the verdict requires; 'stub' is a dev smoke")
    ap.add_argument("--n-warm", type=int, default=N_WARM)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="research/findings/raw/_affect_marker_settle_gpu_timing/a3")
    # Amendment 3 (within-process crossover)
    ap.add_argument("--xo-run", action="store_true")
    ap.add_argument("--xo-score", action="store_true")
    ap.add_argument("--xo-worker", action="store_true")
    ap.add_argument("--orient", default=XO_DEFAULT_ORIENT,
                    help="--xo-run/--xo-score: comma list, first run's arm per process; --xo-worker: one of off/on")
    ap.add_argument("--runs", type=int, default=XO_RUNS)
    ap.add_argument("--run-len", type=int, default=XO_RUN_LEN)
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--decompose-a3", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        ok_a3 = selftest()
        ok_xo = selftest_xo()
        sys.exit(0 if (ok_a3 and ok_xo) else 1)
    if a.decompose_a3:
        raw = a.raw_dir or "research/findings/raw/_affect_marker_settle_gpu_timing/a3"
        rec = decompose_a3(raw)
        out = a.out or os.path.join(raw, "decomposition.json")
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        print("A3 decomposition -> %s" % out)
        return
    if a.xo_worker:
        if a.orient not in ("off", "on"):
            ap.error("--xo-worker takes --orient off|on")
        if a.runs % 4 or a.run_len < 2:
            ap.error("--runs must be a multiple of 4 and --run-len >= 2")
        sys.exit(_worker_xo(a.orient, a.out or "research/findings/raw/_affect_marker_settle_gpu_timing/xo_worker.json",
                            a.renderer, a.runs, a.run_len))
    if a.xo_run or a.xo_score:
        seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
        orients = [x.strip() for x in a.orient.split(",") if x.strip()]
        if any(x not in ("off", "on") for x in orients):
            ap.error("--orient takes only off/on")
        if a.runs % 4 or a.run_len < 2:
            ap.error("--runs must be a multiple of 4 and --run-len >= 2")
        raw = a.raw_dir or a.out_dir
        procs = []
        for s in seeds:
            procs.extend(run_xo(s, orients, raw, a.renderer, a.runs, a.run_len) if a.xo_run
                         else load_xo(raw, s, orients))
        rec = decide_xo(procs)
        rec["seeds"] = list(seeds)
        rec["orient"] = orients
        rec["plan"] = {"runs": a.runs, "run_len": a.run_len}
        rec["cost_projection"] = {"projected_total_hours": len(orients) * len(seeds) * (
            XO_EST_BUILD_S + a.runs * a.run_len * XO_EST_TURN_S) / 3600.0}
        out = a.out or os.path.join(raw, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        mm = rec["metrics"]
        print("STATUS=%s  M1=%s  M2=%s -> %s" % (rec["status"], mm.get("M1_total"), mm.get("M2_wta"), out))
        return
    if a.worker:
        sys.exit(_worker(a.env, a.out or "research/findings/raw/_affect_marker_settle_gpu_timing/worker.json",
                         a.renderer, a.n_warm))
    if a.run:
        seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
        order = [x.strip() for x in a.order.split(",") if x.strip()]
        if any(x not in ("off", "on") for x in order):
            ap.error("--order takes only off/on")
        procs = []
        for s in seeds:
            procs.extend(run_seed(s, order, a.out_dir, a.renderer, a.n_warm))
        rec = decide(procs)
        rec["seeds"] = list(seeds)
        rec["order"] = order
        out = a.out or os.path.join(a.out_dir, "verdict.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        m = rec["metrics"]
        print("STATUS=%s  M1=%s M2=%s noise=%s -> %s" % (rec["status"], m.get("M1_total_delta_s"),
                                                         m.get("M2_wta_delta_s"), m.get("noise_s"), out))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
