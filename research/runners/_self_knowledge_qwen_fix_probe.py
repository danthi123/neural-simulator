"""PART 2 — diagnose + fix the off-bridge Qwen hard-crash (the fluency faculty).

STRONG HYPOTHESIS (from the bridge log 'mempool limit: 80%'): the live cupy bridge's default memory pool
reserves 80% of VRAM (sim/bridge.py:595 `mempool.set_limit(size=total*0.8)`), so when the PyTorch Qwen
(~1GB fp16 weights + CUDA context + activation buffers) is loaded ALONGSIDE a live cupy bridge, PyTorch's
separate allocator can OOM/fault because cupy's pool has already grabbed (and CACHES, not frees-to-OS) most
of the device.

This probe, on GPU, runs three arms (each in its OWN subprocess so a hard-crash in one does NOT take down the
others, and a fault is captured as a non-zero exit + stderr):

  (A) BASELINE  — load Qwen FIRST, NO bridge. Proves Qwen itself loads + renders a few SVO->sentences.
  (B) REPRO     — build a live cupy bridge (default mempool 80%), THEN load Qwen alongside. The crash repro.
  (C) FIX       — same, but (1) cap the cupy mempool to a smaller fraction at bridge-build (GPUConfig
                  memory_pool_limit_fraction=0.5) AND (2) free the cupy pool's cached blocks
                  (`cp.get_default_memory_pool().free_all_blocks()`) BEFORE loading Qwen, so PyTorch has room.
                  Proves Qwen renders SVO->sentences WITHOUT crashing alongside a live bridge.

Each arm prints a single JSON line; the parent aggregates. The render uses SpikingQwenFaculty.render_svo on a
few self-knowledge SVOs.

    SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_qwen_fix_probe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

OUT = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_qwen_fix_probe.json")

# a few self-knowledge SVOs to render (the faculty's job: fluent prose for a recalled brain fact)
SVOS = [("brain", "uses", "spikes"), ("moat", "prevents", "confabulation"), ("brain", "learns", "words")]


# ============================================================================================================
# The CHILD entry: run ONE arm in its own process. arm in {baseline, repro, fix}.
# ============================================================================================================
def _child(arm, T):
    import numpy as np  # noqa
    out = {"arm": arm, "ok": False}
    t0 = time.time()
    try:
        import cupy as cp  # noqa
    except Exception as e:
        out["error"] = f"cupy import failed: {e!r}"
        print("RESULT " + json.dumps(out), flush=True)
        return 0

    def _vram():
        free, total = cp.cuda.runtime.memGetInfo()
        return {"free_mb": round(free / 1024**2, 1), "total_mb": round(total / 1024**2, 1),
                "used_mb": round((total - free) / 1024**2, 1)}

    bridge = None
    if arm in ("repro", "fix"):
        # build a live cupy bridge exactly like the composer/develop loop do
        from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
        from sim.enums import NeuronModel
        from sim.bridge import SimulationBridge
        cfg = CoreSimConfig()
        cfg.num_neurons = 53248          # the firewall agent's ~54K-neuron scale (D=128 composer build)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = 42
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                  "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
            if hasattr(cfg, f):
                setattr(cfg, f, False)
        cfg.ou_std_current_pA = 0.0
        gpu = GPUConfig()
        if arm == "fix":
            gpu.memory_pool_limit_fraction = 0.5     # FIX (1): cap the cupy pool so PyTorch has room
        bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                  runtime_state=RuntimeState(), gpu_config=gpu)
        bridge._initialize_simulation_data(called_from_playback_init=False)
        # exercise the pool a bit (a few steps) so it actually reserves device memory
        for _ in range(5):
            bridge._run_one_simulation_step()
        out["vram_after_bridge"] = _vram()
        out["mempool_limit_mb"] = round(cp.get_default_memory_pool().get_limit() / 1024**2, 1)
        out["mempool_used_after_bridge_mb"] = round(cp.get_default_memory_pool().used_bytes() / 1024**2, 1)
        if arm == "fix":
            # FIX (2): release the cupy pool's CACHED blocks back to the device so PyTorch can allocate.
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream.null.synchronize()
            out["vram_after_freeall"] = _vram()
            out["mempool_used_after_freeall_mb"] = round(cp.get_default_memory_pool().used_bytes() / 1024**2, 1)

    # load the Qwen faculty + render a few SVOs
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fac = SpikingQwenFaculty(T=T, max_new_tokens=24, seed=42, device=dev)
        out["faculty_load_seconds"] = fac.load_seconds
        out["faculty_device"] = str(fac.device)
        out["vram_after_qwen"] = _vram()
        renders = []
        for (a, v, p) in SVOS:
            surface, full, gen_s = fac.render_svo(a, v, p)
            renders.append({"svo": [a, v, p], "surface": surface, "gen_seconds": gen_s})
        out["renders"] = renders
        out["ok"] = all(r["surface"] for r in renders)
    except Exception as e:
        import traceback
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()[-1500:]
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(out), flush=True)
    return 0


# ============================================================================================================
# The PARENT entry: spawn each arm subprocess (isolating crashes), aggregate, write the JSON.
# ============================================================================================================
def _run_arm(arm, T):
    env = dict(os.environ)
    env["SIM_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "research.runners._self_knowledge_qwen_fix_probe", "--child", arm, "--T", str(T)],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=600)
    rec = {"arm": arm, "returncode": proc.returncode}
    # find the RESULT line
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            try:
                result = json.loads(line[len("RESULT "):])
            except Exception:
                pass
    if result is not None:
        rec.update(result)
    else:
        # a hard crash: no RESULT line. Capture the tail of stderr (the fault signature).
        rec["ok"] = False
        rec["hard_crash"] = True
        rec["stderr_tail"] = (proc.stderr or "")[-1500:]
        rec["stdout_tail"] = (proc.stdout or "")[-500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, help="(internal) run one arm in-process")
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    if a.child is not None:
        return _child(a.child, a.T)

    print("=" * 100, flush=True)
    print("[QWEN-FIX PROBE] baseline (Qwen only) | repro (bridge@80% then Qwen) | fix (bridge@50%+free then Qwen)",
          flush=True)
    print("=" * 100, flush=True)
    res = {"T": a.T, "arms": []}
    for arm in ("baseline", "repro", "fix"):
        print(f"\n[arm: {arm}] launching subprocess ...", flush=True)
        rec = _run_arm(arm, a.T)
        res["arms"].append(rec)
        tag = "OK" if rec.get("ok") else ("HARD-CRASH" if rec.get("hard_crash") else "ERROR")
        extra = ""
        if rec.get("vram_after_bridge"):
            extra += f" vram_after_bridge_used={rec['vram_after_bridge']['used_mb']}MB"
        if rec.get("vram_after_qwen"):
            extra += f" vram_after_qwen_used={rec['vram_after_qwen']['used_mb']}MB"
        print(f"    -> {tag} (rc={rec['returncode']}){extra}", flush=True)
        if rec.get("error"):
            print(f"       error: {rec['error'][:200]}", flush=True)
        if rec.get("stderr_tail"):
            print(f"       stderr_tail: ...{rec['stderr_tail'][-300:]}", flush=True)
        if rec.get("renders"):
            for r in rec["renders"]:
                print(f"       render {r['svo']} -> {r['surface']!r}", flush=True)

    base = next(r for r in res["arms"] if r["arm"] == "baseline")
    repro = next(r for r in res["arms"] if r["arm"] == "repro")
    fix = next(r for r in res["arms"] if r["arm"] == "fix")
    res["summary"] = {
        "baseline_ok": bool(base.get("ok")),
        "repro_crashed": bool(not repro.get("ok")),
        "fix_ok": bool(fix.get("ok")),
        "diagnosis": ("CONFIRMED: a live cupy bridge at the default 80% mempool reserves the device so the "
                      "PyTorch Qwen faults; capping the mempool fraction to 0.5 + free_all_blocks() before "
                      "loading Qwen fixes it." if (base.get("ok") and not repro.get("ok") and fix.get("ok"))
                      else "see arms for the actual outcome (the hypothesis may need refining)."),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\n[saved] {a.out}", flush=True)
    print(f"[SUMMARY] baseline_ok={res['summary']['baseline_ok']} repro_crashed={res['summary']['repro_crashed']} "
          f"fix_ok={res['summary']['fix_ok']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
