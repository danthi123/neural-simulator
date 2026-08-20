"""DE-RISK: the FAITHFUL burn-down of the 2026-08-20 inhibition-of-return host-gain scaffold
(`_continuous_wander_ior_derisk.py`, GO
`research/findings/2026-08-20-inhibition-of-return-breaks-the-degenerate-wander-GO.md`): PER-NEURON
spike-frequency-adaptation (SFA) instead of a host multiply on the curiosity neuromod drive (`gains_on`).

THE SCAFFOLD BEING REPLACED: the IOR de-risk fatigued the just-won BASIN by multiplying its `gains_on` entry -- a
python-side scalar fed into `_scale_within_assembly`, which rescales the WITHIN-assembly recurrent SYNAPTIC weights
(`conn.data[idxs] *= factor`). That is a host dial on a population-level neuromodulatory gain (a "transient
recurrent-gain tag" applied ONCE before rest, per `_self_initiated_spontaneous_thought_derisk._scale_within_assembly`'s
own docstring) -- not a property of any individual neuron. The finding's own "Honest scope / residual" names the
faithful next step: "a per-neuron SFA current on the just-ignited CA3 basin ... reusing the 2026-08-14 SFA machinery".

THE FAITHFUL FORM built here: after a wander, EVERY CA3 neuron's OWN adaptation state updates from ITS OWN spike
count during that wander -- real spike-frequency adaptation is intrinsically per-cell (the Izhikevich recovery
variable `u` / an AHP current; the 2026-08-14 SFA-eviction precedent used exactly this class of mechanism via
`cp_izh_d_increment`/`cp_izh_a`, `research/findings/2026-08-14-gnw-rung2b-sfa-workspace-eviction-BOUNDARY.md`). No
basin bookkeeping is required at all: whichever neurons fired a lot (the just-ignited basin's members) get fatigued;
neurons that stayed silent do not -- the basin-level IOR effect EMERGES from purely per-cell state.

THE PERSISTENCE PROBLEM (why this needs its OWN wander wrapper, not `SelfInitiationOrgan.speak()` unmodified):
each `speak()` call (`_wander_speak` -> `_dmn_per_basin_encode_equalization_derisk._run_wander`) builds a FRESH
bridge from scratch (`_prepare_equalized` -> `_build`), and the wander's own `_steered_rest` calls `_hard_silence`,
which explicitly ZEROS the Izhikevich recovery variable `cp_recovery_variable_u` every call (verified by reading
`research/runners/_gap5_spontaneous_reactivation_derisk.py::_hard_silence`: it resets v / u / conductances / apical-v,
then runs `settle` steps). So the native Izhikevich `u` CANNOT carry adaptation across separate `speak()` calls --
whatever accumulated during wander N is wiped before wander N+1 even starts. Two fields survive rebuild-to-rebuild
unaffected by `_hard_silence`: (a) nothing on the OLD bridge (it's discarded, a fresh bridge object each call) and
(b) an EXTERNALLY-OWNED array this runner keeps in Python and RE-INJECTS into the new bridge after each build, via
`bridge.cp_intrinsic_current_pA` -- an existing additive per-neuron current field (`sim/bridge.py`, added into
`total_input_current_pA` every step) that `_hard_silence` never touches and `_steered_rest`'s per-step loop never
overwrites (it only rewrites `cp_external_input_current`, a DIFFERENT field). `_clamped_encode` in the equalization
de-risk already hyperpolarises competitor assemblies through the sibling field `cp_external_input_current` the same
way during encode -- this is the same class of host-injected CURRENT, not a re-derivation. The LOCAL CA3 index
ordering is deterministic given the seed (`rng = np.random.default_rng(seed*17+3)` inside `_prepare_equalized`), so
LOCAL index j always names the SAME neuron across separate builds -- carrying a length-n_ca3 array indexed by j is
valid.

MECHANISM: `sfa_state[j]` in [0,1] per LOCAL CA3 neuron j (0 = fully rested). Each wander:
  1. RECOVERY: sfa_state *= SFA_PERSIST (a fraction is retained; the rest decays toward rest -- an AHP/K+ channel
     relaxing over the inter-wander interval).
  2. OWN-ACTIVITY FATIGUE: sfa_state = max(sfa_state, this_wander's_own_spike_count / max_spike_count) -- a neuron
     that fired heavily THIS wander (an ignited-basin member) becomes maximally fatigued regardless of its prior
     state; a neuron that never fired is untouched by this step (only recovers).
  3. INJECT (start of the NEXT wander, on the freshly-built bridge): bridge.cp_intrinsic_current_pA[global id of
     neuron j] = -SFA_STRENGTH_PA * sfa_state[j] -- a per-neuron hyperpolarizing bias sized by that neuron's OWN
     recent activity, persisting through the whole wander (untouched by `_hard_silence` or the per-step loop).

TWO ARMS (same organ/seed, N successive wanders each):
  BASELINE : the byte-clean production path (`_dmn_per_basin_encode_equalization_derisk._run_wander`, UNMODIFIED,
             reused-by-import) -- expected to reproduce the 2026-08-20 negative (fixed concept every wander).
  SFA      : the wrapper above -- expect VARIETY (n_distinct > 1), and ideally full coverage across the n_mem
             stored concepts (the IOR host-gain scaffold only ever reached a 2-cycle).

Numpy smoke (TINY -- import + mechanics sanity only, NOT a validity claim; production is cupy/n_ca3=2000):
  SIM_BACKEND=numpy WANDER_N=2 SFA_N_CA3=200 SFA_REST_STEPS=150 SFA_SETTLE_STEPS=80 \
    .venv/bin/python -m research.runners._continuous_wander_perneuron_sfa_derisk
Cupy verification (production scale -- QUEUE this, do not run cupy directly; see the GPU queue in CLAUDE.md):
  SIM_BACKEND=cupy BRAIN_SELF_INITIATE_STORE=1 \
    .venv/bin/python -m research.runners._continuous_wander_perneuron_sfa_derisk
Writes research/findings/raw/_continuous_live_cupy/wander_perneuron_sfa.json ; exit 0 iff SFA yields MORE variety
than baseline (n_distinct > 1 and > baseline).
"""
import os
import sys
import json
import time

import numpy as np

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")

N = int(os.environ.get("WANDER_N", "6"))
SFA_STRENGTH_PA = float(os.environ.get("SFA_STRENGTH_PA", "500.0"))    # hyperpolarizing bias (pA) at full fatigue
SFA_PERSIST = float(os.environ.get("SFA_PERSIST", "0.5"))              # fraction of fatigue RETAINED each wander
N_CA3_OVERRIDE = os.environ.get("SFA_N_CA3")                           # smoke-only override (prod == GO_CFG's 2000)
REST_STEPS_OVERRIDE = os.environ.get("SFA_REST_STEPS")                 # smoke-only override (prod == 4000 on cupy)
SETTLE_STEPS_OVERRIDE = os.environ.get("SFA_SETTLE_STEPS")             # smoke-only override (prod == 600)
_SUF = os.environ.get("SFA_OUT_SUFFIX", "")                            # distinct artifact per sweep config
OUT = os.path.join("research", "findings", "raw", "_continuous_live_cupy", "wander_perneuron_sfa%s.json" % _SUF)


def _run_wander_with_sfa(seed, cfg, rest_steps, gains, encode_mode, sfa_state_local, sfa_strength_pa):
    """Reuse-by-import `_prepare_equalized` + `_scale_within_assembly` + `_steered_rest` (all UNMODIFIED -- the
    exact production wander machinery `_wander_speak` calls). The ONLY addition: inject the PERSISTED per-neuron
    adaptation as a hyperpolarizing bias on `bridge.cp_intrinsic_current_pA` before the noise-driven rest -- a field
    `_hard_silence` never zeros (verified by reading it) and the rest loop never overwrites, so it stays live for
    the whole wander. Returns F, prep, diag exactly like the reused `_run_wander`, so every downstream analysis call
    (`_selection`, `_utterance_stream`) is byte-identical to the production dominant-basin read-out."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from research.runners._dmn_per_basin_encode_equalization_derisk import _prepare_equalized, _scale_within_assembly
    from research.runners._self_initiated_spontaneous_thought_derisk import _steered_rest

    n_mem = int(cfg["n_mem"])
    prep = _prepare_equalized(seed, cfg, do_encode=True, encode_mode=encode_mode)
    if gains is not None:
        for i in range(n_mem):
            _scale_within_assembly(prep, i, float(gains[i]))

    bridge = prep["bridge"]
    n_all = int(bridge.core_config.num_neurons)
    if getattr(bridge, "cp_intrinsic_current_pA", None) is None:
        bridge.cp_intrinsic_current_pA = cp.zeros(n_all, dtype=cp.float32)
    if sfa_state_local is not None and float(np.max(sfa_state_local)) > 0.0:
        ca3_global = cp.asarray(np.asarray(prep["ca3_arr_host"], dtype=np.int64))
        bias_host = (-float(sfa_strength_pa) * np.asarray(sfa_state_local, dtype=np.float64)).astype(np.float32)
        bridge.cp_intrinsic_current_pA[ca3_global] = cp.asarray(bias_host)

    F, diag = _steered_rest(prep, [0.0] * n_mem, rest_steps, seed, noise_on=True)
    return F, prep, diag


def _dominant_concept(F, prep, org, min_frac):
    from research.runners._self_initiation_multibasin_derisk import _selection
    from research.runners._self_initiated_utterance_derisk import _utterance_stream
    ident = list(range(org.n_mem))
    sel = _selection(F, prep["assemblies_local"], org.seed, min_frac)
    st = _utterance_stream(F, prep["assemblies_local"], org.agents, org.utt_by_agent, org.decode_ok, min_frac, ident)
    counts = np.asarray(st["counts"], dtype=float)
    concept = org.agents[int(np.argmax(counts))] if counts.sum() > 0 else None
    return concept, sel, st


def _arm_cfg(org, n_ca3, settle_steps):
    from research.runners._gap5_spontaneous_reactivation_derisk import GO_CFG
    cfg = dict(GO_CFG)
    cfg["n_ca3"] = int(n_ca3)
    cfg["n_mem"] = int(org.n_mem)
    cfg["settle_steps"] = int(settle_steps)
    return cfg


def _run_baseline(org, n_ca3, rest_steps, settle_steps, min_frac):
    """The byte-clean production path (`_run_wander`, unmodified) -- reproduces the 2026-08-20 negative."""
    from research.runners._dmn_per_basin_encode_equalization_derisk import _run_wander
    cfg = _arm_cfg(org, n_ca3, settle_steps)
    seq, times = [], []
    for _ in range(N):
        t0 = time.time()
        F, prep, diag = _run_wander(org.seed, cfg, rest_steps, True, gains=org.gains_on,
                                    encode_mode="consolidated", do_encode=True)
        c, _sel, _st = _dominant_concept(F, prep, org, min_frac)
        times.append(round(time.time() - t0, 1))
        seq.append(c)
    valid = [c for c in seq if c]
    return {"sequence": seq, "per_wander_s": times, "n_distinct": len(set(valid)), "distinct": sorted(set(valid))}


def _run_sfa(org, n_ca3, rest_steps, settle_steps, min_frac):
    """The FAITHFUL per-neuron SFA arm -- a persisted per-neuron fatigue array carried (in this runner's own
    Python state, across separate `_run_wander_with_sfa` calls / fresh bridge builds)."""
    cfg = _arm_cfg(org, n_ca3, settle_steps)
    sfa_state = np.zeros(int(n_ca3), dtype=np.float64)
    seq, times, fatigue_trace = [], [], []
    for _ in range(N):
        t0 = time.time()
        F, prep, diag = _run_wander_with_sfa(org.seed, cfg, rest_steps, org.gains_on, "consolidated",
                                             sfa_state, SFA_STRENGTH_PA)
        c, _sel, _st = _dominant_concept(F, prep, org, min_frac)
        times.append(round(time.time() - t0, 1))
        seq.append(c)
        # per-neuron own-activity fatigue update (recovery, then own-activity max) -- see MECHANISM above.
        spike_counts = F.sum(axis=0).astype(np.float64)
        mx = float(spike_counts.max())
        fatigue_now = (spike_counts / mx) if mx > 0 else spike_counts
        sfa_state = sfa_state * SFA_PERSIST
        sfa_state = np.maximum(sfa_state, fatigue_now)
        fatigue_trace.append(round(float(sfa_state.mean()), 4))
    valid = [c for c in seq if c]
    return {"sequence": seq, "per_wander_s": times, "n_distinct": len(set(valid)), "distinct": sorted(set(valid)),
            "mean_fatigue_after_wander": fatigue_trace}


def main() -> int:
    from sim.backend import get_backend
    _, backend = get_backend()
    from research.runners.self_initiated_production_organ import SelfInitiationOrgan, _settle_steps, _wander_rest_steps

    org = SelfInitiationOrgan(seed=42)
    org._ensure_mouth()
    min_frac = org.min_frac
    n_ca3 = int(N_CA3_OVERRIDE) if N_CA3_OVERRIDE else 2000
    rest_steps = int(REST_STEPS_OVERRIDE) if REST_STEPS_OVERRIDE else _wander_rest_steps()
    settle_steps = int(SETTLE_STEPS_OVERRIDE) if SETTLE_STEPS_OVERRIDE else _settle_steps()

    print(f"=== BASELINE (byte-clean production _run_wander, fixed gains) === "
          f"n_ca3={n_ca3} rest={rest_steps} settle={settle_steps} n_mem={org.n_mem}", flush=True)
    baseline = _run_baseline(org, n_ca3, rest_steps, settle_steps, min_frac)
    print(baseline["sequence"], "-> n_distinct", baseline["n_distinct"], flush=True)

    print(f"=== PER-NEURON SFA (persisted per-neuron fatigue current, strength={SFA_STRENGTH_PA}pA "
          f"persist={SFA_PERSIST}) ===", flush=True)
    sfa = _run_sfa(org, n_ca3, rest_steps, settle_steps, min_frac)
    print(sfa["sequence"], "-> n_distinct", sfa["n_distinct"], flush=True)

    result = {
        "runner": "research/runners/_continuous_wander_perneuron_sfa_derisk.py",
        "seed": org.seed, "backend": backend, "n_wanders": N, "n_mem": org.n_mem,
        "n_ca3": n_ca3, "rest_steps": rest_steps, "settle_steps": settle_steps,
        "sfa_strength_pA": SFA_STRENGTH_PA, "sfa_persist": SFA_PERSIST,
        "baseline": baseline, "sfa": sfa,
        "baseline_distinct": baseline["n_distinct"], "sfa_distinct": sfa["n_distinct"],
        "full_coverage": bool(sfa["n_distinct"] >= org.n_mem),
        # A SFA arm BYTE-IDENTICAL to baseline means the injection had NO measurable effect on the trajectory, which
        # cannot be distinguished from 'the current was never engaged' WITHOUT an injection-effect diagnostic -> that
        # is UNDEFINED, never a validated NO-GO (the silent-failure discipline: verify the instrument before trusting
        # its output). A GENUINE NO-GO needs the injection to demonstrably perturb the run yet not increase variety.
        "seqs_byte_identical_to_baseline": bool(baseline["sequence"] == sfa["sequence"]),
        "VERDICT": ("GO" if sfa["n_distinct"] > baseline["n_distinct"] and sfa["n_distinct"] > 1
                    else ("UNDEFINED" if baseline["sequence"] == sfa["sequence"] else "NO-GO")),
        "interpretation": (
            "per-neuron SFA (a persisted hyperpolarizing current on cp_intrinsic_current_pA, sized by each "
            "neuron's OWN spike count in the prior wander) breaks the degenerate wander WITHOUT a host multiply "
            "on the curiosity neuromod drive -- replaces the 2026-08-20 IOR gain-scaffold with the faithful form"
            if sfa["n_distinct"] > baseline["n_distinct"] and sfa["n_distinct"] > 1 else
            "UNDEFINED: the SFA arm is BYTE-IDENTICAL to baseline -> the injection had no measurable effect on the "
            "wander, so we cannot tell 'not engaged' (cp_intrinsic_current_pA not read by _steered_rest, or sfa_state "
            "stayed 0) from 'too weak'. NEEDS an injection-effect diagnostic before a verdict: confirm (a) sfa_state>0 "
            "after wander 1, (b) cp_intrinsic_current_pA nonzero after inject, (c) it is added into total_input_current "
            "during the noise rest. The host-gain IOR scaffold (2026-08-20, wired live) remains the working form."
            if baseline["sequence"] == sfa["sequence"] else
            "per-neuron SFA perturbed the run but did not increase variety -- sweep SFA_STRENGTH_PA / SFA_PERSIST"
        ),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("baseline_distinct", "sfa_distinct", "full_coverage", "VERDICT")},
                     indent=2), flush=True)
    return 0 if result["VERDICT"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
