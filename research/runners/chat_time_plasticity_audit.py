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

`--real-pool-freeze-check` (coordinator review round 2, 2026-09-24): `run_real_pool_freeze_check()` builds the
REAL wave3 production pool with BOTH `BRAIN_WORLDMODEL_LOCAL_FREEZE=1` and `BRAIN_SURPRISE_LOCAL_FREEZE=1`,
drives a few real chat turns, and asserts every co-resident organ's gain0-frozen edges
(`onebrain_merge_framework._apply_gain0_freeze`, a `freeze_regions`-driven direct-array freeze independent of
`cfg.enable_hebbian_learning` -- the real safety net the first round of this audit omitted) stay at max|dw|==0,
in addition to re-checking world-model/surprise's own new named gates. See that function's docstring for the
full mechanism explanation.

Run (numpy/CPU, memcapped):
  bash tools/mem_ok.sh 12 && bash tools/memcap.sh 12 -- .venv/bin/python -m research.runners.chat_time_plasticity_audit \
      --seed 42 --out research/findings/raw/_chat_time_plasticity_audit/s42.json
  bash tools/mem_ok.sh 12 && bash tools/memcap.sh 12 -- .venv/bin/python -m research.runners.chat_time_plasticity_audit \
      --real-pool-freeze-check --seed 42 --out research/findings/raw/_chat_time_plasticity_audit/real_pool_freeze_s42.json
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


def run_real_pool_freeze_check(seed: int) -> dict:
    """Coordinator review round 2 (2026-09-24): the FIRST audit's fix (`BRAIN_WORLDMODEL_LOCAL_FREEZE` /
    `BRAIN_SURPRISE_LOCAL_FREEZE`) was verified only on a SYNTHETIC 3-organ pool
    (`tests/_chat_time_plasticity_local_freeze_scenario.py`), never on the REAL wave3 production pool -- and the
    finding OMITTED the real safety net that protects the other 6 co-resident organs: `onebrain_merge_framework.
    py`'s `_apply_gain0_freeze` (pool-build step 7) sets `cp_plasticity_rate_gain = 0.0` DIRECTLY (no gate NAME,
    so it is invisible to `set_plasticity_gate`/`list_plasticity_gates`) on every edge with BOTH endpoints inside
    a descriptor's declared `freeze_regions` -- INDEPENDENT of `cfg.enable_hebbian_learning`. Comprehension,
    metacog, pragmatic, self_schema, curiosity, causal_whatif and source_provenance's wave3-pool regions ALL
    declare `freeze_regions` (`_onebrain_wave3_organread_verify._wave3_descriptors`, reuse-by-import below);
    world-model and surprise are the ONLY two pool-#1 organs that do NOT (`_onebrain_twopool_merge_organread_
    verify._recon_descriptors`'s own comment: "match pool-1 global; pool-2 edges gain-0 frozen" -- the pool's
    original designers built this distinction on purpose). So flipping the two new flags on can ONLY ever affect
    world-model/surprise's own now-named-gated pathways; every other organ's edges are hard-frozen at pool BUILD
    time regardless of what `cfg.enable_hebbian_learning` reads afterward. This function measures that directly
    on the REAL pool instead of arguing it from source: builds the actual wave3 pool with BOTH flags on, drives
    a few real chat turns through `webapp.server.brain_chat`, and asserts the gain0-frozen union's weights are
    UNCHANGED, in addition to re-checking world-model/surprise's own named gates (as `run()` already does on the
    default/flags-off configuration)."""
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    os.environ["SIM_DISABLE_LLM"] = "1"
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    os.environ["BRAIN_WORLDMODEL_LOCAL_FREEZE"] = "1"
    os.environ["BRAIN_SURPRISE_LOCAL_FREEZE"] = "1"

    import numpy as np
    from webapp import server as S
    from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
    from research.runners._onebrain_wave2_organread_verify import _frozen_edge_weights

    out = {"seed": seed, "flags": {"BRAIN_WORLDMODEL_LOCAL_FREEZE": "1", "BRAIN_SURPRISE_LOCAL_FREEZE": "1"}}

    descs = _wave3_descriptors()
    frozen_owners = {d.key: list(d.freeze_regions) for d in descs if d.freeze_regions}
    frozen_regions = sorted({r for regs in frozen_owners.values() for r in regs})
    out["frozen_regions_by_organ"] = frozen_owners
    out["organs_with_no_freeze_regions"] = sorted(d.key for d in descs if not d.freeze_regions)

    renderer = "stub"
    chat, source = S._build_chat_brain("tiny-demo", renderer)
    chat._brain_chat_source = source
    cache_key = ("real_pool_freeze_check", "tiny-demo", renderer)
    S._BRAIN_CHATS[cache_key] = chat

    # warmup order (mirrors _warm_chat_brain / run()'s own replication above; value-choice skipped, same reason).
    from research.runners.affect_production_organ import affect_enabled
    if affect_enabled():
        S._get_affect_organ().ensure_built()
    from research.runners.comprehension_production_organ import comprehension_enabled
    if comprehension_enabled():
        S._get_comprehension_organ().ensure_built()
    from research.runners.surprise_production_organ import surprise_enabled
    if surprise_enabled():
        S._get_surprise_organ().ensure_built()
    from research.runners.metacog_production_organ import metacog_enabled
    if metacog_enabled():
        S._get_metacog_organ().ensure_built()
    from research.runners.worldmodel_production_organ import worldmodel_enabled
    if worldmodel_enabled():
        S._get_worldmodel_organ().ensure_built()
    from research.runners.pragmatic_production_organ import pragmatic_enabled
    if pragmatic_enabled():
        S._get_pragmatic_organ().ensure_built()

    from research.runners.onebrain_wave3_pool_production import get_merged_cortical_pool
    wave3 = get_merged_cortical_pool(seed, min_wave=1)
    if wave3 is None:
        out["error"] = "wave3 pool did not build (wave3_pool_enabled() False? check BRAIN_ONEBRAIN_WAVE3_POOL)"
        return out
    bridge = wave3.bridge
    out["enable_hebbian_learning_after_warmup"] = bool(bridge.core_config.enable_hebbian_learning)
    gates_after_warmup = {n: float(bridge.get_plasticity_gate_value(n)) for n in bridge.list_plasticity_gates()}
    out["gates_after_warmup"] = gates_after_warmup

    from research.runners._affective_world_model_derisk import WORLDMODEL_FREEZE_GATE
    from research.runners._spiking_expectation_rpe_derisk import SURPRISE_FREEZE_GATE
    out["worldmodel_gate_value"] = gates_after_warmup.get(WORLDMODEL_FREEZE_GATE)
    out["surprise_gate_value"] = gates_after_warmup.get(SURPRISE_FREEZE_GATE)

    frozen_w0 = _frozen_edge_weights(bridge, frozen_regions)
    nnz = bridge.cp_connections.nnz
    whole_w0 = np.asarray(_to_host(bridge.cp_connections.data))[:nnz].copy()

    TURNS = [
        ("teach_dog", "the dog chases the cat"),
        ("teach_bird", "the bird eats the worm"),
        ("recall_dog", "what does the dog chase"),
        ("novel", "what does the elephant drink"),
    ]
    responses = []
    for i, (label, msg) in enumerate(TURNS):
        r = S.brain_chat(S.BrainChatRequest(session="real_pool_freeze_check", message=msg, brain="tiny-demo",
                                            renderer=renderer, rich=False, reset=False))
        responses.append({"label": label, "response": json.loads(r.body)})
    out["responses"] = responses

    frozen_w1 = _frozen_edge_weights(bridge, frozen_regions)
    frozen_max_dw = (float(np.max(np.abs(frozen_w1 - frozen_w0)))
                     if frozen_w0.shape == frozen_w1.shape else float("inf"))
    out["gain0_frozen_regions_max_dw"] = frozen_max_dw
    out["gain0_frozen_regions_n_synapses"] = int(frozen_w0.size)
    out["gain0_frozen_regions_ok"] = bool(frozen_max_dw == 0.0)

    # WHOLE-BRIDGE cross-check: everything protected (gain0-frozen union + world-model/surprise's own named
    # gates) plus anything NOT protected. `run()`'s synthetic-probe test already proved the MECHANISM moves an
    # ungated synapse when the switch is genuinely True; this checks whether any such unprotected synapse
    # actually EXISTS on the REAL pool today. `gain0_frozen_regions_max_dw` above (computed on the exact index
    # set `_frozen_edge_weights` derives from `frozen_regions`) is the authoritative per-mechanism verdict for
    # the other 6 organs; this is only an order-of-magnitude cross-check over the whole bridge.
    whole_w1 = np.asarray(_to_host(bridge.cp_connections.data))[:nnz]
    dw_all = np.abs(whole_w1 - whole_w0)
    out["whole_bridge_max_dw"] = float(dw_all.max()) if dw_all.size else 0.0
    out["residual_note"] = ("whole_bridge_max_dw covers every synapse (gain0-frozen + named-gated + anything "
                            "else); gain0_frozen_regions_max_dw and the world-model/surprise gate values above "
                            "are the authoritative per-mechanism checks.")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--real-pool-freeze-check", action="store_true",
                    help="Coordinator round 2: verify the REAL wave3 pool with both BRAIN_*_LOCAL_FREEZE=1 flags "
                         "holds every co-resident organ's gain0-frozen edges at max|dw|==0 over a few chat turns.")
    a = ap.parse_args()
    result = run_real_pool_freeze_check(a.seed) if a.real_pool_freeze_check else run(a.seed)
    js = json.dumps(result, indent=2, default=str)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(js)
        print(f"[chat_time_plasticity_audit] wrote {a.out}", flush=True)
    else:
        print(js)
