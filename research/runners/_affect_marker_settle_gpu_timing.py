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
"""
from __future__ import annotations

import argparse
import json
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
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
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
