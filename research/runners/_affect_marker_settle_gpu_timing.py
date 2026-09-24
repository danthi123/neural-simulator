"""AFFECT-MARKER SETTLE — GPU per-turn latency de-risk (research/settle-multiturn-contrast, 2026-09-24).

WHY. `BRAIN_AFFECT_MARKER_SETTLE` adds simulated TIME to the affect-marker WTA's own tiny private bridge: per
axis (valence, then -- only when a word was actually selected -- arousal), `WARMUP_STEPS` 60->`DELIBERATION_MS`
500 and `WASHOUT_STEPS` 40->`INTERTURN_REST_MS` 1000 (dt_ms=1.0, so step count == simulated ms). A full
non-neutral turn calls `_select` TWICE (valence then arousal), so SETTLE adds up to ~2 x ((500-60)+(1000-40)) =
2800 simulated ms of EXTRA steps per affective turn versus the byte-identical-OFF read. Whether that is
acceptable depends on the WALL-CLOCK cost per step on the production GPU path (`SIM_BACKEND=cupy`), which is
dominated by per-step CUDA kernel-launch overhead on a network this small (~216 valence-circuit neurons + ~72
arousal-circuit neurons), not by FLOPs -- an empirical question this runner measures, not derives.

THE PROBE. TWO fresh, sequential, single-process arms (never concurrent -- "one brain-loading GPU process at a
time", `tools/gpu_queue.sh`'s own rule) per seed: SETTLE=0, then SETTLE=1. Each arm:
  1. builds the SAME tiny-demo brain (`webapp.server.brain_chat`, brain='tiny-demo', renderer='stub', LLM
     disabled -- the production GPU path, `SIM_BACKEND=cupy`) via a WARMUP turn (`NEU_TEXT`, a fresh session,
     `reset=True`) -- this pays the one-time brain-build cost, timed and reported SEPARATELY (`build_and_warmup_s`)
     so it never leaks into the per-turn numbers below;
  2. times the FIRST affective turn (`EMO_TEXT`, same session, `reset=False`) alone -- `first_affective_turn_s`;
  3. times a SECOND, back-to-back affective turn (`EMO_TEXT` again, same session) -- `repeat_affective_turn_s` --
     this is the WARM-READER case (matching `_affect_marker_settle_multiturn_derisk`'s mt_emo2/mt_emo3): with
     SETTLE off the reader's own 40 ms washout is the SAME whether the previous read was 1 ms or 1 second of
     wall-clock ago (the bridge only advances when READ), so this number isolates the recurring per-turn cost a
     production session actually pays on every subsequent affective turn, not just the first one.
Reuses `_affect_marker_settle_multiturn_derisk.EMO_TEXT` / `NEU_TEXT` verbatim (no new text constants) and
`onebrain_regression_battery._spawn_arm` for the fresh-subprocess-per-arm mechanism (the SAME mechanism the
multi-turn contrast and every load-bearing arm in this repo already use), so noise trajectories are independent
between the two arms exactly as that model requires.

NOT a correctness re-check (the multi-turn contrast already covers H1-H5); this runner ONLY times.

Queue (GPU, one job at a time, gated on host RAM at DISPATCH time -- `tools/gpu_queue.sh`'s dispatcher runs the
mem_ok check when it actually pops the job, not when it is enqueued):
  bash tools/gpu_queue.sh add 'bash tools/mem_ok.sh 14 4 && SIM_BACKEND=cupy OMP_NUM_THREADS=1 tools/memcap.sh 14 \\
      -- .venv/bin/python -u -m research.runners._affect_marker_settle_gpu_timing --run --seeds 42 \\
      --out research/findings/raw/_affect_marker_settle_gpu_timing/s42.json'

Self-test (no brain build, no GPU):
  .venv/bin/python -m research.runners._affect_marker_settle_gpu_timing --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

VERIFY_SEEDS = (42,)   # the primary queued run; --seeds extends (amendment convention, e.g. slotbinder L3)
SETTLE_ENV = "BRAIN_AFFECT_MARKER_SETTLE"
GPU_SESSION = "gpu_settle_timing"


def _worker(env_json: str, out_path: str) -> int:
    """Subprocess entry: build ONE tiny-demo brain (warmup turn), then time the affective turn twice."""
    os.environ.setdefault("SIM_BACKEND", "cupy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v            # OFF-ARM DISCIPLINE: explicit "0"/"1", never a pop
    from research.runners._affect_marker_settle_multiturn_derisk import EMO_TEXT, NEU_TEXT
    from webapp.server import brain_chat, BrainChatRequest

    def _turn(msg, reset):
        t0 = time.time()
        r = brain_chat(BrainChatRequest(session=GPU_SESSION, message=msg, brain="tiny-demo",
                                        renderer="stub", rich=False, reset=reset))
        dt = time.time() - t0
        return dt, json.loads(r.body)

    t_build, resp_warm = _turn(NEU_TEXT, True)
    t_first, resp_first = _turn(EMO_TEXT, False)
    t_repeat, resp_repeat = _turn(EMO_TEXT, False)
    rec = {
        "env": env, "seed": os.environ.get("BRAIN_CHAT_SEED"), "sim_backend": os.environ.get("SIM_BACKEND"),
        "build_and_warmup_s": t_build, "first_affective_turn_s": t_first, "repeat_affective_turn_s": t_repeat,
        "first_lead": (resp_first.get("affect_drives") or {}).get("lead", ""),
        "repeat_lead": (resp_repeat.get("affect_drives") or {}).get("lead", ""),
        "first_level": (resp_first.get("affect_drives") or {}).get("level"),
        "warmup_lead": (resp_warm.get("affect_drives") or {}).get("lead", ""),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print("[gpu timing worker] env=%s seed=%s build=%.3fs first=%.3fs repeat=%.3fs -> %s"
          % (env, rec["seed"], t_build, t_first, t_repeat, out_path), flush=True)
    return 0


def _spawn_arm(seed: int, settle: bool, out_path: str):
    """A fresh subprocess per arm (the SAME independent-noise-trajectory mechanism
    `onebrain_regression_battery._spawn_arm` uses; not reused directly because that helper iterates a fixed
    `turn_labels` roster against `_TURN_BY_LABEL`, and this runner needs the bespoke build/first/repeat timing
    sequence `_worker` above implements)."""
    import subprocess
    env = {SETTLE_ENV: "1" if settle else "0"}
    old = {k: os.environ.get(k) for k in ("BRAIN_CHAT_SEED", "SIM_BACKEND", "BRAIN_CHAT_RENDERER", "SIM_DISABLE_LLM")}
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    os.environ.setdefault("SIM_BACKEND", "cupy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    try:
        p = subprocess.run(
            [sys.executable, "-u", "-m", "research.runners._affect_marker_settle_gpu_timing",
             "--worker", "--env", json.dumps(env), "--out", out_path], env=dict(os.environ))
        if p.returncode != 0 or not os.path.exists(out_path):
            return None
        with open(out_path) as f:
            return json.load(f)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_seed(seed: int, out_dir: str) -> dict:
    sdir = os.path.join(out_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    off = _spawn_arm(seed, False, os.path.join(sdir, "off.json"))
    on = _spawn_arm(seed, True, os.path.join(sdir, "on.json"))
    return summarize(seed, off, on)


def summarize(seed: int, off: dict, on: dict) -> dict:
    ok = bool(off and on)
    row = {"seed": seed, "off": off, "on": on, "valid": ok}
    if ok:
        row["delta_first_affective_turn_s"] = on["first_affective_turn_s"] - off["first_affective_turn_s"]
        row["delta_repeat_affective_turn_s"] = on["repeat_affective_turn_s"] - off["repeat_affective_turn_s"]
        row["off_marker_present"] = bool(off.get("first_lead") or off.get("repeat_lead"))
        row["on_marker_present"] = bool(on.get("first_lead") or on.get("repeat_lead"))
    return row


def aggregate(rows: list) -> dict:
    valid = [r for r in rows if r["valid"]]
    out = {"n_seeds": len(rows), "n_valid": len(valid), "rows": rows}
    if valid:
        out["mean_off_first_affective_turn_s"] = sum(r["off"]["first_affective_turn_s"] for r in valid) / len(valid)
        out["mean_on_first_affective_turn_s"] = sum(r["on"]["first_affective_turn_s"] for r in valid) / len(valid)
        out["mean_off_repeat_affective_turn_s"] = sum(r["off"]["repeat_affective_turn_s"] for r in valid) / len(valid)
        out["mean_on_repeat_affective_turn_s"] = sum(r["on"]["repeat_affective_turn_s"] for r in valid) / len(valid)
        out["mean_delta_first_s"] = sum(r["delta_first_affective_turn_s"] for r in valid) / len(valid)
        out["mean_delta_repeat_s"] = sum(r["delta_repeat_affective_turn_s"] for r in valid) / len(valid)
    return out


# ─────────────────────────────────────────────────────── selftest ──────────────────────────────────────────────
def _row(seed, off_first, on_first, off_repeat, on_repeat):
    off = {"first_affective_turn_s": off_first, "repeat_affective_turn_s": off_repeat, "first_lead": "Gladly! ",
          "repeat_lead": ""}
    on = {"first_affective_turn_s": on_first, "repeat_affective_turn_s": on_repeat, "first_lead": "Gladly! ",
         "repeat_lead": "Gladly! "}
    return summarize(seed, off, on)


def selftest() -> bool:
    ok = True
    r = _row(42, 0.05, 0.20, 0.04, 0.18)
    got = (abs(r["delta_first_affective_turn_s"] - 0.15) < 1e-9 and abs(r["delta_repeat_affective_turn_s"] - 0.14) < 1e-9
          and r["off_marker_present"] and r["on_marker_present"])
    ok = ok and got
    print("  delta computation ->", "ok" if got else "FAIL")
    # a missing arm (build failed) -> UNDEFINED-shaped (valid=False), never a fabricated 0 delta
    bad = summarize(42, None, {"first_affective_turn_s": 0.2, "repeat_affective_turn_s": 0.2, "first_lead": "", "repeat_lead": ""})
    got2 = (not bad["valid"]) and "delta_first_affective_turn_s" not in bad
    ok = ok and got2
    print("  missing arm -> no fabricated delta ->", "ok" if got2 else "FAIL")
    agg = aggregate([r, bad])
    got3 = agg["n_seeds"] == 2 and agg["n_valid"] == 1 and abs(agg["mean_delta_first_s"] - 0.15) < 1e-9
    ok = ok and got3
    print("  aggregate skips the invalid seed ->", "ok" if got3 else "FAIL")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", default=" ".join(str(s) for s in VERIFY_SEEDS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default="research/findings/raw/_affect_marker_settle_gpu_timing")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.worker:
        sys.exit(_worker(a.env, a.out or "research/findings/raw/_affect_marker_settle_gpu_timing/worker.json"))
    seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
    if a.run:
        rows = [run_seed(s, a.out_dir) for s in seeds]
        rec = aggregate(rows)
        out = a.out or os.path.join(a.out_dir, "summary.json")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, indent=1, default=str)
        if rec["n_valid"]:
            print("n=%d/%d  mean OFF first=%.3fs repeat=%.3fs | mean ON first=%.3fs repeat=%.3fs | "
                  "mean delta first=%+.3fs repeat=%+.3fs -> %s"
                  % (rec["n_valid"], rec["n_seeds"], rec["mean_off_first_affective_turn_s"],
                     rec["mean_off_repeat_affective_turn_s"], rec["mean_on_first_affective_turn_s"],
                     rec["mean_on_repeat_affective_turn_s"], rec["mean_delta_first_s"], rec["mean_delta_repeat_s"], out))
        else:
            print("NO VALID SEEDS (every arm build failed) -> %s" % out)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
