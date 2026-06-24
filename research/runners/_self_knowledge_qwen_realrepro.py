"""PART 2 (faithful repro) — reproduce the REAL demo's pre-Qwen VRAM state, then load Qwen.

The isolated bridge+Qwen probe did NOT crash, because cupy's set_limit is a CAP not a reservation (the pool
only grows as it allocates). The real demo loads Qwen AFTER: (1) the full develop loop (StreamCortex bridge +
per-day MultiTurnAgent composer bridges + read_codes densify), and (2) building the firewall agent
(BrainConversationalAgent + RFPhasorComposer) and RUNNING the firewall batteries -- which create + CACHE many
RF composer bridges (self._bridge_cache) + Izhikevich cleanup banks that stay ALIVE holding pool memory.

This arm builds the firewall agent on the developed grounded codes (cupy backend), runs the firewall (so the
composer caches its bridges), keeps it ALL alive, then loads Qwen -- the real pre-Qwen state.

Two arms (subprocess-isolated): 'as_is' (load Qwen with everything alive, no mempool tweak) and 'fixed'
(cap the composer/agent cupy bridges' mempool to 0.5 via GPUConfig + free_all_blocks() right before Qwen).

    SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_qwen_realrepro
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

OUT = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_qwen_realrepro.json")
SVOS = [("brain", "uses", "spikes"), ("moat", "prevents", "confabulation"), ("brain", "learns", "words")]


def _child(arm, T):
    out = {"arm": arm, "ok": False}
    t0 = time.time()
    import numpy as np
    import cupy as cp
    from research.runners._self_knowledge_demo import (
        _load_curriculum, _qa_vocab, build_qa_agent, run_firewall, _all_facts_svo)

    def _vram():
        free, total = cp.cuda.runtime.memGetInfo()
        return {"free_mb": round(free / 1024**2, 1), "used_mb": round((total - free) / 1024**2, 1)}

    cur = _load_curriculum()
    facts = _all_facts_svo(cur)
    action_words = {v for (_a, v, _p) in facts}
    vocab = _qa_vocab(cur)
    # dev-random grounded codes stand-in (the VRAM behavior is identical to the real grounded codes; this avoids
    # the 50-min develop loop). The point is the WM-loop + composer bridge VRAM growth, not the code values.
    rng = np.random.default_rng(42)
    grounded = {w: rng.uniform(0, 1, 128) for w in vocab}

    # (optionally) cap the agent's composer cupy bridges' mempool by patching GPUConfig default for THIS process
    if arm == "fixed":
        import sim.config as scfg
        _orig = scfg.GPUConfig

        class _CappedGPUConfig(_orig):
            def __init__(self, *a2, **k2):
                super().__init__(*a2, **k2)
                self.memory_pool_limit_fraction = 0.5
        scfg.GPUConfig = _CappedGPUConfig
        # the composer imports GPUConfig at call time from sim.config, so the patch takes effect for new bridges

    # build the REAL MultiTurnAgent (the heavy 8640-neuron WM-loop dlPFC bridge -- the run.log's 12.4GB jump) so
    # this repro matches the real demo's pre-Qwen VRAM state, then exercise recall through it.
    from research.runners._longitudinal_develop_loop import build_agent, _teach_fact
    referent_nouns = sorted({f[0] for f in facts} | {f[2] for f in facts})
    mt = build_agent(vocab, 42, plastic=True, use_multiturn=True, enable_neural_render=False,
                     referent_nouns=referent_nouns)
    from research.runners._longitudinal_develop_loop_gpu import _inject_grounded
    _inject_grounded(mt, grounded)
    for f in facts:
        _teach_fact(mt, f)
    out["vram_after_wm_build"] = _vram()
    # trigger the composer's resonate ops (the 12GB growth happens here, during recall)
    mt_recall = sum(1 for a, v, p in facts if mt.what_does(a, v) == p)
    out["mt_recall"] = round(mt_recall / len(facts), 4)
    out["vram_after_wm_recall"] = _vram()
    out["mempool_used_after_wm_recall_mb"] = round(cp.get_default_memory_pool().used_bytes() / 1024**2, 1)

    agent, n_taught = build_qa_agent(cur, vocab, grounded, 42)
    fw = run_firewall(agent, cur, action_words)          # exercises + caches the composer's RF bridges
    out["n_taught"] = n_taught
    out["firewall_project"] = f"{fw['positive_answered']}/{fw['positive_total']}"
    out["n_cached_composer_bridges"] = len(getattr(agent.composer, "_bridge_cache", {}))
    out["vram_after_agent"] = _vram()
    out["mempool_used_after_agent_mb"] = round(cp.get_default_memory_pool().used_bytes() / 1024**2, 1)
    out["mempool_limit_mb"] = round(cp.get_default_memory_pool().get_limit() / 1024**2, 1)

    if arm == "fixed":
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        out["vram_after_freeall"] = _vram()
        out["mempool_used_after_freeall_mb"] = round(cp.get_default_memory_pool().used_bytes() / 1024**2, 1)

    # now load Qwen alongside the live agent
    try:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fac = SpikingQwenFaculty(T=T, max_new_tokens=24, seed=42, device=dev)
        out["faculty_load_seconds"] = fac.load_seconds
        out["vram_after_qwen"] = _vram()
        renders = [{"svo": [a, v, p], "surface": fac.render_svo(a, v, p)[0]} for (a, v, p) in SVOS]
        out["renders"] = renders
        out["ok"] = all(r["surface"] for r in renders)
    except Exception as e:
        import traceback
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()[-1500:]
    out["elapsed_seconds"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(out), flush=True)
    return 0


def _run_arm(arm, T):
    env = dict(os.environ)
    env["SIM_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "research.runners._self_knowledge_qwen_realrepro", "--child", arm, "--T", str(T)],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=900)
    rec = {"arm": arm, "returncode": proc.returncode}
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
        rec["ok"] = False
        rec["hard_crash"] = True
        rec["stderr_tail"] = (proc.stderr or "")[-2000:]
        rec["stdout_tail"] = (proc.stdout or "")[-500:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
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
    print("[QWEN REAL-REPRO] firewall agent alive (composer bridges cached) -> load Qwen. as_is vs fixed.",
          flush=True)
    print("=" * 100, flush=True)
    res = {"T": a.T, "arms": []}
    for arm in ("as_is", "fixed"):
        print(f"\n[arm: {arm}] launching subprocess ...", flush=True)
        rec = _run_arm(arm, a.T)
        res["arms"].append(rec)
        tag = "OK" if rec.get("ok") else ("HARD-CRASH" if rec.get("hard_crash") else "ERROR")
        print(f"    -> {tag} (rc={rec['returncode']})", flush=True)
        for k in ("n_cached_composer_bridges", "vram_after_agent", "mempool_used_after_agent_mb",
                  "vram_after_freeall", "vram_after_qwen", "faculty_load_seconds"):
            if k in rec:
                print(f"       {k}: {rec[k]}", flush=True)
        if rec.get("error"):
            print(f"       error: {rec['error'][:300]}", flush=True)
        if rec.get("stderr_tail"):
            print(f"       stderr_tail: ...{rec['stderr_tail'][-400:]}", flush=True)
        for r in rec.get("renders", []):
            print(f"       render {r['svo']} -> {r['surface']!r}", flush=True)

    as_is = next(r for r in res["arms"] if r["arm"] == "as_is")
    fixed = next(r for r in res["arms"] if r["arm"] == "fixed")
    res["summary"] = {"as_is_ok": bool(as_is.get("ok")), "fixed_ok": bool(fixed.get("ok"))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\n[saved] {a.out}", flush=True)
    print(f"[SUMMARY] as_is_ok={res['summary']['as_is_ok']} fixed_ok={res['summary']['fixed_ok']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
