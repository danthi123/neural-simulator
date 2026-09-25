"""CHAT-TIME PLASTICITY AUDIT (2026-09-24): which synapses in the PRODUCTION brain can actually change DURING a
real conversation, and is any chat-time-learning faculty SILENTLY FROZEN by a mechanism unrelated to its own
design?

TRIGGER: research/findings/2026-09-24-plastic-mask-instrument-and-production-reachability.md found that
`WorldModelProductionOrgan._build_one` (research/runners/worldmodel_production_organ.py) trains its own
state->pred transition then sets `cfg.enable_hebbian_learning = False` on the ONE cfg object the DEFAULT-ON
11-organ wave3 pool shares (`onebrain_wave3_pool_production.get_merged_cortical_pool`, default-ON since
2026-09-17) -- a bridge-WIDE kill switch, not a per-organ freeze, that (per that finding's own honest scope)
"a live question this document does not resolve". This runner answers it empirically, through the REAL
production entry point (`webapp.server.brain_chat`), not a docstring read.

WHAT THIS MEASURES, in order:
  1. WARMUP, replicating `webapp/server.py::_warm_chat_brain`'s inner `_warm()` attach order EXACTLY (affect ->
     comprehension -> surprise -> metacog -> world-model -> pragmatic; value-choice is SKIPPED -- its own
     docstring warns its first build can cost ~4 minutes, and it attach-orders AFTER every organ this bug can
     touch, so skipping it does not change what this runner measures). After EACH step: the shared wave3 pool's
     `cfg.enable_hebbian_learning` / `cfg.enable_stdp`, so "when does Hebbian go off, and who turns it off" is
     read off the real objects, not inferred from source.
  2. SIX real chat turns through `webapp.server.brain_chat` (a teach/recall/prospective-memory/novel-topic
     conversation, seed 42, numpy, stub renderer, no LLM) -- the SAME handler `/api/brain-chat` calls. After
     EVERY turn: cfg state + whole-bridge max|dw| for the shared wave3 pool AND for every organ confirmed
     STANDALONE (own bridge, immune to the shared-cfg mechanism): affect, source_provenance, prospective_memory
     (built lazily on its own formation turn). Weight drift is bucketed by NAMED plasticity gate (meant frozen)
     vs UNGATED (meant plastic, i.e. would-be chat-time-plastic if `enable_hebbian_learning` were ever True).

HONEST SCOPE: this measures the DEFAULT production configuration (every `BRAIN_*_LOCAL_FREEZE` flag OFF, i.e.
today's shipped mechanism) at seed 42, numpy/CPU. It is a single-seed DESCRIPTIVE instrument (see docs/TERMS.md
on 'works'/'selective'): the qualitative claims here (which cfg is shared, which pathway is gated, whether a
drift is exactly 0.0 or nonzero) do not depend on the seed; a magnitude claim would.

Run (numpy/CPU, memcapped):
  bash tools/mem_ok.sh 8 && bash tools/memcap.sh 8 -- .venv/bin/python -m research.runners.chat_time_plasticity_audit \
      --seed 42 --out research/findings/raw/_chat_time_plasticity_audit/s42.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _to_host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def _bridge_cfg_state(bridge):
    cfg = bridge.core_config
    return {"enable_hebbian_learning": bool(cfg.enable_hebbian_learning),
            "enable_stdp": bool(cfg.enable_stdp),
            "gates": {n: float(bridge.get_plasticity_gate_value(n)) for n in bridge.list_plasticity_gates()}}


def _drift(bridge, w0):
    """Whole-bridge max|dw| since `w0`, bucketed by NAMED gate (meant frozen/controlled) vs UNGATED (meant
    plastic under the ambient `enable_hebbian_learning` switch alone). Mirrors the instrument
    research/findings/raw/_plastic_mask_flip_prep/other_organs_drift_probe.py already validated (2026-09-24),
    extended with the gate-name bucketing this audit's task needs."""
    import numpy as np
    nnz = bridge.cp_connections.nnz
    if len(w0) != nnz:
        return {"error": "nnz changed since baseline", "nnz0": len(w0), "nnz1": nnz}
    w1 = np.asarray(_to_host(bridge.cp_connections.data))[:nnz]
    dw = np.abs(w1 - w0)
    out = _bridge_cfg_state(bridge)
    out["whole_bridge_max_dw"] = float(dw.max()) if dw.size else 0.0
    gated_idx = set()
    per_gate = {}
    for name in bridge.list_plasticity_gates():
        idx = np.asarray(bridge._plasticity_gate_to_synapses.get(name, []), dtype=np.int64)
        idx = idx[idx < dw.size]
        gated_idx.update(int(i) for i in idx.tolist())
        per_gate[name] = float(dw[idx].max()) if idx.size else 0.0
    out["per_gate_max_dw"] = per_gate
    if gated_idx:
        gi = np.fromiter(gated_idx, dtype=np.int64)
        ungated_mask = np.ones(dw.size, dtype=bool)
        ungated_mask[gi] = False
    else:
        ungated_mask = np.ones(dw.size, dtype=bool)
    out["ungated_max_dw"] = float(dw[ungated_mask].max()) if ungated_mask.any() else 0.0
    out["n_gated_synapses"] = int(len(gated_idx))
    out["n_ungated_synapses"] = int(ungated_mask.sum())
    return out


def _w0(bridge):
    import numpy as np
    nnz = bridge.cp_connections.nnz
    return np.asarray(_to_host(bridge.cp_connections.data))[:nnz].copy()


def run(seed: int) -> dict:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    os.environ["SIM_DISABLE_LLM"] = "1"
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    # D5 episodic's WRITE is latency-DEFERRED on numpy by design (`webapp/server.py::_episodic_store_ok`: "a
    # BTSP store is ~seconds on cupy (the production substrate) but ~510s/topic on numpy@2000"). An earlier
    # version of this runner forced `BRAIN_EPISODIC_STORE=1` to observe it anyway; on this shared, contended
    # box that made the 6-turn audit take >10 minutes per turn and it was killed unfinished. D5 is NOT part of
    # the wave3-shared-cfg mechanism this audit investigates (its own EpisodicDapMemory substrate is not one of
    # the wave3 pool's 11 organs), so this runner leaves the numpy default (deferred) in place and reports that
    # honestly rather than forcing an expensive, off-topic measurement.

    from webapp import server as S

    out = {"seed": seed, "backend": os.environ.get("SIM_BACKEND"), "warmup": [], "turns": []}

    renderer = "stub"
    chat, source = S._build_chat_brain("tiny-demo", renderer)
    chat._brain_chat_source = source
    cache_key = ("audit", "tiny-demo", renderer)
    S._BRAIN_CHATS[cache_key] = chat

    def _pool_state(step):
        try:
            from research.runners.onebrain_wave3_pool_production import get_merged_cortical_pool
            pool = get_merged_cortical_pool(seed, min_wave=1)
            if pool is None:
                out["warmup"].append({"step": step, "wave3_pool": None})
                return
            state = _bridge_cfg_state(pool.bridge)
            state["step"] = step
            out["warmup"].append(state)
        except Exception as e:
            out["warmup"].append({"step": step, "error": f"{type(e).__name__}: {e}"})

    _pool_state("before_any_organ")

    from research.runners.affect_production_organ import affect_enabled
    if affect_enabled():
        S._get_affect_organ().ensure_built()
    _pool_state("after_affect")   # affect is standalone -- expect no change to the wave3 pool's own state

    from research.runners.comprehension_production_organ import comprehension_enabled
    if comprehension_enabled():
        S._get_comprehension_organ().ensure_built()
    _pool_state("after_comprehension")

    from research.runners.surprise_production_organ import surprise_enabled
    if surprise_enabled():
        S._get_surprise_organ().ensure_built()
    _pool_state("after_surprise")

    from research.runners.metacog_production_organ import metacog_enabled
    if metacog_enabled():
        S._get_metacog_organ().ensure_built()
    _pool_state("after_metacog")

    from research.runners.worldmodel_production_organ import worldmodel_enabled
    if worldmodel_enabled():
        S._get_worldmodel_organ().ensure_built()
    _pool_state("after_worldmodel")

    from research.runners.pragmatic_production_organ import pragmatic_enabled
    if pragmatic_enabled():
        S._get_pragmatic_organ().ensure_built()
    _pool_state("after_pragmatic")
    # value-choice SKIPPED (own docstring: first build can cost ~4 min; attach-orders after every organ this
    # bug can touch, so skipping it does not change what this runner measures -- see module docstring).

    # ── baseline snapshots (post-warmup == "chat is about to start") ──
    from research.runners.onebrain_wave3_pool_production import get_merged_cortical_pool
    wave3 = get_merged_cortical_pool(seed, min_wave=1)
    w0_wave3 = _w0(wave3.bridge) if wave3 is not None else None
    affect_bridge = getattr(S._get_affect_organ(), "bridge", None)
    w0_affect = _w0(affect_bridge) if affect_bridge is not None else None

    TURNS = [
        ("teach_dog", "the dog chases the cat"),
        ("teach_bird", "the bird eats the worm"),
        ("recall_dog", "what does the dog chase"),
        ("pmem_form", "remind me to check the oven when the timer rings"),
        ("intervening", "what does the bird eat"),
        ("pmem_cue", "the timer rings"),
    ]
    for i, (label, msg) in enumerate(TURNS):
        r = S.brain_chat(S.BrainChatRequest(session="audit", message=msg, brain="tiny-demo", renderer=renderer,
                                            rich=False, reset=(i == 0 and False)))
        turn_out = {"label": label, "message": msg, "response": json.loads(r.body)}
        turn_out["wave3_pool"] = _drift(wave3.bridge, w0_wave3) if wave3 is not None else None
        turn_out["affect_bridge"] = _drift(affect_bridge, w0_affect) if affect_bridge is not None else None
        # source_provenance: own standalone bridge, built lazily the first time a recalled fact is judged.
        try:
            import research.runners.source_provenance_production_organ as SP
            if SP.source_provenance_enabled():
                sp_organ = S._get_source_provenance_organ()
                sp_bridge = getattr(getattr(sp_organ, "_brain", None), "_bridge", None)
                if sp_bridge is not None:
                    key = f"sp_w0_{i}"
                    if "sp_w0" not in out:
                        out["sp_w0"] = _w0(sp_bridge)
                    turn_out["source_provenance_bridge"] = _drift(sp_bridge, out["sp_w0"])
        except Exception as e:
            turn_out["source_provenance_bridge"] = {"error": f"{type(e).__name__}: {e}"}
        # prospective memory: PER-SESSION, own standalone bridge, built lazily on the formation turn.
        try:
            pm_organ = S._SESSION_PMEM.get(cache_key)
            pm_bridge = getattr(getattr(pm_organ, "_pm", None), "bridge", None) if pm_organ is not None else None
            if pm_bridge is not None:
                if "pm_w0" not in out:
                    out["pm_w0"] = _w0(pm_bridge)
                turn_out["prospective_memory_bridge"] = _drift(pm_bridge, out["pm_w0"])
            else:
                turn_out["prospective_memory_bridge"] = None
        except Exception as e:
            turn_out["prospective_memory_bridge"] = {"error": f"{type(e).__name__}: {e}"}
        out["turns"].append(turn_out)

    # numpy arrays are not JSON-serializable -- drop the raw baselines before dumping.
    out.pop("sp_w0", None)
    out.pop("pm_w0", None)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    result = run(a.seed)
    js = json.dumps(result, indent=2, default=str)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(js)
        print(f"[chat_time_plasticity_audit] wrote {a.out}", flush=True)
    else:
        print(js)
