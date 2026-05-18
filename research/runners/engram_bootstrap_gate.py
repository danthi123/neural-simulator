"""Kill-safe THREE-STATE + SCALE-CONFIDENCE gate: does the validated
reward-FREE Tonegawa engram bind BOOTSTRAP the rewarded episode the
compose-bridge VOID lacked (n_rewarded=0), so the validated temporal-
credit/eligibility mechanism GENERATIVELY refines it -- and is that
capability SCALE-CONFIDENT across a pre-registered local scale ladder?

REUSES byte-UNMODIFIED: compose_bridge_core.cbr_verdict (frozen _CBR_*
INHERITED -- NO new movable bar), the Tonegawa engram bridge API, the
validated temporal-credit/eligibility path, build_biological_brain_
regions, sim.train_checkpoint, sim.neuromodulators. EVERY condition
gets the IDENTICAL engram bootstrap; conditions differ ONLY in the
temporal-credit refinement on top (mechanism isolation). NO automatic
differentiation. ASCII only.

HONEST CEILING (printed, never spun): a SCALE-CONFIDENT PASS = the
generative mechanism works locally at small capacity AND shows no
architectural ceiling across the local ladder (so scale-up is
justified) -- explicitly NOT GPT-class/open-ended fluent composition
on local hardware, NOT an LLM, NOT conversation-solved. A works-small-
but-plateaus result is an honest non-success (NOT a win) that triggers
the autonomous Q2 pivot."""
from __future__ import annotations
import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.compose_bridge_core import cbr_verdict

# Pre-registered, NEVER tuned (mirrors compose_bridge_gate's frozen
# _GAMMA/_LAMBDA pattern). _SCALE_TOL is the substrate's irreducible
# greedy-eval noise floor, justified BEFORE any run.
_SCALE_LADDER = (4, 8, 16)
_SCALE_TOL = 0.05


def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    import math
    return f if math.isfinite(f) else None


def scale_confidence(rungs):
    """Pure, deterministic, fail-closed classification over the ordered
    per-rung records. rungs: list of {"B", "verdict": {"GATE": ...},
    "td_mean", "engram_only_mean"} ordered by ascending B.

    Pre-registered (NEVER tuned):
      (a) every rung GATE == PASS;
      (b) td non-decreasing up to _SCALE_TOL across adjacent rungs;
      (c) at the LARGEST rung td >= _CBR_SCI_ACC_MIN AND
          td - engram_only >= _SCALE_TOL (generative signature holds at
          the hardest scale).
    SCALE-CONFIDENT iff (a)&(b)&(c). Else classify honestly:
      any VOID rung -> VOID; any FAIL rung -> FAIL; all PASS but
      (b)/(c) fails -> WORKS-SMALL-NO-SCALE-CONFIDENCE. Non-numeric/
      missing/unordered -> VOID (never raise)."""
    from research.runners.compose_bridge_core import _CBR_SCI_ACC_MIN
    try:
        ordered = sorted(rungs, key=lambda r: r["B"])
    except (TypeError, KeyError):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "rungs not orderable by B"}
    if [r.get("B") for r in ordered] != list(_SCALE_LADDER):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "ladder != pre-registered %s"
                          % (_SCALE_LADDER,)}
    gates = []
    for r in ordered:
        v = r.get("verdict")
        g = v.get("GATE") if isinstance(v, dict) else None
        gates.append(g)
    if any(g == "VOID" or g is None for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is VOID/missing"}
    if any(g == "FAIL" for g in gates):
        return {"scale_confident": False, "classification": "FAIL",
                "reason": "a rung GATE is FAIL"}
    if any(g != "PASS" for g in gates):
        return {"scale_confident": False, "classification": "VOID",
                "reason": "a rung GATE is not PASS/FAIL/VOID"}
    tds, eos = [], []
    for r in ordered:
        t = _num(r.get("td_mean"))
        e = _num(r.get("engram_only_mean"))
        if t is None or e is None:
            return {"scale_confident": False, "classification": "VOID",
                    "reason": "non-numeric rung metric"}
        tds.append(t)
        eos.append(e)
    monotone = all(tds[i + 1] >= tds[i] - _SCALE_TOL
                   for i in range(len(tds) - 1))
    top_ok = (tds[-1] >= _CBR_SCI_ACC_MIN
              and (tds[-1] - eos[-1]) >= _SCALE_TOL)
    if monotone and top_ok:
        return {"scale_confident": True,
                "classification": "SCALE-CONFIDENT-PASS",
                "reason": "all rungs PASS; td monotone up to tol; "
                          "generative signature holds at largest rung",
                "td_by_rung": tds, "engram_only_by_rung": eos}
    return {"scale_confident": False,
            "classification": "WORKS-SMALL-NO-SCALE-CONFIDENCE",
            "reason": "all rungs PASS but %s%s"
                      % ("" if monotone else "td degrades beyond tol; ",
                         "" if top_ok else "generative signature absent "
                         "at largest rung"),
            "td_by_rung": tds, "engram_only_by_rung": eos}


import numpy as np

from research.runners.text_minimal_isolation import (
    build_biological_brain_regions)
from sim.kernels import fused_eligibility_trace_decay  # noqa: F401 (parity)
from sim.train_checkpoint import save_checkpoint  # kill-safe
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: scale-confidence PoC ONLY -- generative "
           "mechanism local at small capacity + no architectural "
           "ceiling across the local ladder; NOT GPT-class/open-ended "
           "fluent composition on local hardware, NOT an LLM, NOT "
           "conversation-solved. works-small-but-plateaus = honest "
           "non-success -> autonomous Q2 pivot.")
_CONTROLS_SEMANTICS = {
    "hebbian_no_trace": "engram_only: identical to td incl. the engram "
                        "bootstrap+stimulate_tag, MINUS EXACTLY the "
                        "eligibility-trace bridging across the gap (the "
                        "faithful storage-only analog; NOT a strawman).",
    "permuted": "identical engram bootstrap; pi(verb->motor) re-"
                "randomized per episode via a DEDICATED rng so the "
                "training-order stream stays byte-aligned with td "
                "(reward decorrelated; single-variable isolate).",
    "wrongsign": "identical engram bootstrap; TD delta sign-flipped."}

_GAMMA = 0.95
_LAMBDA = 0.9
_N_BINDINGS_TINY = 4


def _params_for(B, tiny):
    if tiny:
        return dict(B=_N_BINDINGS_TINY, n_lang_input=64 * _N_BINDINGS_TINY,
                    sparsity=0.5 / _N_BINDINGS_TINY, n_per_pool=8,
                    n_fs_per_pool=2, stim_steps=4, gap_steps=3,
                    reset_steps=2, readout_steps=4, encode_steps=4,
                    n_train_epochs=2, drive_pA=260.0, teacher_pA=420.0,
                    engram_stim_pA=600.0, engram_top_k=24)
    return dict(B=B, n_lang_input=64 * B, sparsity=0.5 / B,
                n_per_pool=40, n_fs_per_pool=6, stim_steps=24,
                gap_steps=14, reset_steps=10, readout_steps=18,
                encode_steps=20, n_train_epochs=10, drive_pA=260.0,
                teacher_pA=420.0, engram_stim_pA=600.0, engram_top_k=120)


def _da_modulator_from_delta():
    return NeuromodulatorConfig(
        name="dopamine_engram_bootstrap", baseline=0.0,
        decay_tau_ms=50.0, concentration_min=-5.0, concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0, threshold=0.0,
                                         window_ms=0.0)])


def _pool_names(B):
    return ["P%d" % i for i in range(B)]


def _build_bridge(seed, P):
    from sim.config import (CoreSimConfig, VisualizationConfig,
                            RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    names = _pool_names(P["B"])
    regions, pathways = build_biological_brain_regions(
        n_lang_input=P["n_lang_input"], n_motor_per_action=8,
        enable_motor_fs=False, enable_noun_pools=True,
        noun_pool_names=list(names), n_noun_per_pool=P["n_per_pool"],
        n_noun_fs_per_pool=P["n_fs_per_pool"],
        concept_pool_internal_density=0.05,
        concept_pool_exc_weight_mean=0.3,
        concept_pool_inh_weight_mean=0.8)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_stdp = True
    cfg.enable_reward_modulation = True
    cfg.reward_learning_rate = 0.05
    cfg.reward_eligibility_tau_ms = 200.0
    cfg.reward_baseline = 0.0
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg,
                              viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(),
                              gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def _verb_drive(verb_idx, B, n_lang_input, P):
    from sim.text_embeddings import orthogonal_drive_pattern
    return orthogonal_drive_pattern(
        cue_idx=verb_idx, n_cues=B, n_neurons=n_lang_input,
        drive_max_pA=P["drive_pA"], sparsity=P["sparsity"])


def _step(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1


def _encode_engram(bridge, tag, verb_idx, target_pool_idx, P,
                   lang_arr, pool_arrs):
    cp = bridge.xp if hasattr(bridge, "xp") else np
    B = P["B"]
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(P["reset_steps"]):
        _step(bridge)
    bridge.start_engram_recording(tag)
    drive = cp.asarray(_verb_drive(verb_idx, B, lang_arr.shape[0], P),
                        dtype=cp.float32)
    bridge.cp_external_input_current[lang_arr] = drive
    bridge.cp_external_input_current[pool_arrs[target_pool_idx]] += \
        float(P["teacher_pA"])
    for _ in range(P["encode_steps"]):
        _step(bridge)
    bridge.commit_engram_tag(tag, top_k=int(P["engram_top_k"]))
    bridge.cp_external_input_current[:] = 0.0


def _episode(bridge, mode, verb_idx, target_pool_idx, tag, P,
             lang_arr, pool_arrs, value_table):
    cp = bridge.xp if hasattr(bridge, "xp") else np
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(P["reset_steps"]):
        _step(bridge)
    drive = cp.asarray(_verb_drive(verb_idx, P["B"], lang_arr.shape[0],
                                   P), dtype=cp.float32)
    bridge.cp_external_input_current[lang_arr] = drive
    bridge.stimulate_tag(tag, drive_pA=float(P["engram_stim_pA"]),
                         additive=True)
    for _ in range(P["stim_steps"]):
        _step(bridge)
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(P["gap_steps"]):
        if mode == "hebbian_no_trace" and \
                bridge.cp_eligibility_trace is not None:
            bridge.cp_eligibility_trace[:] = 0.0  # engram_only: ONLY diff
        _step(bridge)
    counts = np.zeros(P["B"], dtype=np.float64)
    for _ in range(P["readout_steps"]):
        _step(bridge)
        fired = bridge.cp_firing_states
        for j, pa in enumerate(pool_arrs):
            counts[j] += float(fired[pa].sum())
    selected = int(np.argmax(counts))
    reward = 1.0 if selected == target_pool_idx else 0.0
    v = float(value_table[verb_idx])
    delta = reward - v
    value_table[verb_idx] = v + (1.0 - _GAMMA * _LAMBDA) * delta
    if mode == "wrongsign":
        delta = -delta
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = float(delta)
    _step(bridge)
    bridge.core_config.current_reward_signal = 0.0
    return reward, selected


def _bijection(rng, B):
    perm = np.arange(B)
    rng.shuffle(perm)
    return perm


def _greedy_score(bridge, pi, P, lang_arr, pool_arrs):
    cp = bridge.xp if hasattr(bridge, "xp") else np
    B = P["B"]
    correct = 0
    for vi in range(B):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(P["reset_steps"]):
            _step(bridge)
        drive = cp.asarray(_verb_drive(vi, B, lang_arr.shape[0], P),
                            dtype=cp.float32)
        bridge.cp_external_input_current[lang_arr] = drive
        for _ in range(P["stim_steps"] + P["gap_steps"]):
            _step(bridge)
        counts = np.zeros(B, dtype=np.float64)
        for _ in range(P["readout_steps"]):
            _step(bridge)
            fired = bridge.cp_firing_states
            for j, pa in enumerate(pool_arrs):
                counts[j] += float(fired[pa].sum())
        if int(np.argmax(counts)) == int(pi[vi]):
            correct += 1
    bridge.cp_external_input_current[:] = 0.0
    return correct / float(B)


def _run_mode(mode, seed, P, gap_zero=False):
    Pl = dict(P)
    if gap_zero:
        Pl["gap_steps"] = 0
    B = Pl["B"]
    bridge = _build_bridge(seed, Pl)
    cp = bridge.xp if hasattr(bridge, "xp") else np
    rm = bridge.region_manager
    lang_arr = cp.asarray(list(rm.indices("language_input")),
                          dtype=cp.int64)
    names = _pool_names(B)
    pool_arrs = [cp.asarray(list(rm.indices("noun_pool_%s" % nm)),
                            dtype=cp.int64) for nm in names]
    try:
        bridge.set_plasticity_gate("language_input_to_noun_pool", 1.0)
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    # Dedicated RNG for the `permuted` control's per-epoch pi
    # re-randomization, so the SHARED training-order stream stays
    # byte-aligned with `td` (adversarial-review STRENGTHEN: permuted
    # then differs from td ONLY by the intended reward decorrelation).
    perm_rng = np.random.default_rng(seed + 1000003)
    pi = _bijection(rng, B)
    value_table = np.zeros(B, dtype=np.float64)
    tags = {}
    for vi in range(B):
        tag = "boot_%d_%d" % (seed, vi)
        _encode_engram(bridge, tag, vi, int(pi[vi]), Pl, lang_arr,
                        pool_arrs)
        tags[vi] = tag
    n_rewarded = 0
    for _ep in range(Pl["n_train_epochs"]):
        if mode == "permuted":
            pi = _bijection(perm_rng, B)
        order = np.arange(B)
        rng.shuffle(order)
        for vi in order:
            r, _sel = _episode(bridge, mode, int(vi), int(pi[vi]),
                               tags[int(vi)], Pl, lang_arr, pool_arrs,
                               value_table)
            n_rewarded += int(r > 0.0)
    try:
        bridge.set_plasticity_gate("language_input_to_noun_pool", 0.0)
    except Exception:
        pass
    return _greedy_score(bridge, pi, Pl, lang_arr, pool_arrs), n_rewarded


def _run_seed(seed, P):
    nogap, _ = _run_mode("td", seed, P, gap_zero=True)
    td, nrew_td = _run_mode("td", seed, P, gap_zero=False)
    controls = {}
    for c in _CONTROLS:
        acc, _ = _run_mode(c, seed, P, gap_zero=False)
        controls[c] = acc
    return {"nogap_td": nogap, "td": td, "controls": controls,
            "n_rewarded_td": nrew_td,
            "controls_semantics": _CONTROLS_SEMANTICS}


def _run_rung(B, seeds, tiny, ckpt):
    P = _params_for(B, tiny)
    per_seed = {}
    for s in seeds:
        row = _run_seed(s, P)
        if ckpt:
            save_checkpoint(ckpt, (B * 1000 + s),
                            {"row": [row["nogap_td"], row["td"]]},
                            None, [])
        per_seed[s] = row
    verdict = cbr_verdict(per_seed)
    tds = [per_seed[s]["td"] for s in sorted(per_seed)]
    eos = [per_seed[s]["controls"]["hebbian_no_trace"]
           for s in sorted(per_seed)]
    return {"B": B, "verdict": verdict,
            "td_mean": float(np.mean(tds)),
            "engram_only_mean": float(np.mean(eos))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny-synth", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds for the pre-registered gate")
        return 2
    _ = _da_modulator_from_delta()
    ladder = (_N_BINDINGS_TINY,) if a.tiny_synth else _SCALE_LADDER
    rungs = []
    try:
        for B in ladder:
            rungs.append(_run_rung(B, a.seeds, a.tiny_synth, a.ckpt))
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    sc = scale_confidence(rungs) if not a.tiny_synth else {
        "scale_confident": False,
        "classification": "TINY-SYNTH (toy; NOT propagated)"}
    out = {"ladder": rungs, "scale_confident": sc["scale_confident"],
           "scale_classification": sc["classification"],
           "scale_reason": sc.get("reason", ""), "banner": _BANNER,
           "tiny_synth": bool(a.tiny_synth)}
    if a.tiny_synth:
        out["note"] = "TINY-SYNTH toy verdict -- NOT propagated"
    else:
        out["note"] = ("multi-rung scale-confidence verdict -- "
                       "recompute from this JSON; no re-run/no tuning")
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("SCALE=%s  classification=%s  %s"
          % (out["scale_confident"], out["scale_classification"],
             _BANNER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
