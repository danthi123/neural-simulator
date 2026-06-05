"""Step 3 — Task 1 DECISION de-risk probe: does the dlPFC working-memory loop's PERSISTENT ACTIVITY survive
at dt=1.0 (the parser+composer timestep) or does it require its tuned dt=0.5?

THE ONE QUESTION that decides step 3 (one-bridge unification): a single SimulationBridge has ONE integration
timestep `dt`. The parser + composer (steps 1–2) run at dt=1.0 ms with NMDA OFF. The dlPFC dialogue-planning
working memory (`content_selection_spiking.build_loop_wm_bridge`: a `cortex_ctx ↔ dlpfc_wm` reverberatory loop,
NMDA ON) is tuned to dt=0.5 ms — its PERSISTENT ACTIVITY (the working-memory "latch": the driven neurons keep
firing AFTER the input drive is removed, holding a concept active for spreading-activation dialogue planning) is
sustained by NMDA + loop reverberation, which may need the finer timestep.

  * MERGE   → if the loop's persistence survives at dt=1.0, the dlPFC can merge onto the unified bridge at
              dt=1.0 (step-3 Task 2 proceeds).
  * BOUNDARY→ if persistence collapses to baseline at dt=1.0, the honest result is "working-memory timescale ≠
              binding timescale": the dlPFC stays a SEPARATE-timing region. NOT a failure — a real
              biology-translatable finding. The validated dialogue planning is NOT weakened to force a merge.

WHAT THIS PROBE MEASURES (per dt ∈ {0.5 baseline, 1.0 decision}):
  1. driven_during_drive  — the driven concept-assembly's firing rate WHILE its cortex pattern is driven
     (sanity: the drive lands).
  2. driven_post_drive    — the driven assembly's firing rate AFTER the drive is removed (the PERSISTENCE — the
     working-memory latch; THIS is the quantity that decides MERGE vs BOUNDARY).
  3. control_post_drive   — an UN-driven concept-assembly's post-drive rate in the SAME run (specificity control:
     persistence should be of the driven concept, not a generic global excitation).
  4. no_drive_baseline    — the driven assembly's firing rate in a NO-DRIVE run (the floor the persistence must
     clear; sustained activity must be ABOVE the network's resting/OU spontaneous rate).

MECHANISM (the module's VALIDATED persistence path, reused exactly): the bare random loop does not reliably
latch a SPECIFIC concept (the module's single-region standalone was a negative). Persistence is installed as a
pattern-specific ATTRACTOR — outer-product cortex_A↔dlPFC_A assembly weights (`SpikingLoopContextBuffer` does
exactly this: 220x specificity, holds the driven concept). So this probe installs ONE concept's attractor, drives
that concept's cortex pattern, removes the drive, and measures whether the attractor self-sustains. The ONLY
variable swept is `dt`; everything else (attractor weights, drive magnitude, windows, OU off for a clean hold) is
held fixed across the two timesteps so the comparison isolates the timestep.

WHY A LOCAL `dt`-PARAMETERIZED BUILDER (and not editing content_selection_spiking): the shipped
`build_loop_wm_bridge` hardcodes `cfg.dt_ms = 0.5`. Per the step-3 plan + the standing "no protected edits, and
prefer not to edit content_selection_spiking" constraint, this probe carries a runner-side COPY of that builder
with a `dt` kwarg (otherwise byte-identical region/pathway/config construction — same regions, same loop
pathways, same NMDA-on, same stdp_w_max, same fast_spike_reset). No `sim/` edit; no content_selection_spiking
edit. The attractor install + drive/read reuse the same `set_pathway_weights` / `cp_external_input_current` /
`_run_one_simulation_step` API the validated `SpikingLoopContextBuffer` uses.

Runs on the validated production (CuPy/GPU) backend — the loop's persistent-activity dynamics are GPU-bound; on
NumPy the loop dynamics diverge from the validated behavior.

  python -m research.findings.raw._step3_dlpfc_dt_probe --seed 42
"""
from __future__ import annotations

import argparse

import numpy as np


# Persistence operating point. drive magnitude + windows mirror the validated SpikingLoopContextBuffer
# (drive_pA≈2500, stim/settle windows). The attractor weight is chosen to put the loop in the GENUINELY
# NMDA-DEPENDENT bistability regime — the real dlPFC working-memory mechanism — NOT trivial strong AMPA
# recurrence.
#
# >>> WHY attractor weight 30 (NOT the module's installed-attractor 50): a control sweep (this probe's
# development; recorded in the finding) showed that at the module's outer-product weight 50, the post-drive
# "persistence" SURVIVES even with NMDA OFF (post≈0.33 NMDA-off) — i.e. it is trivial saturated AMPA
# ping-pong, NOT the NMDA-dependent latch the dlPFC dialogue planning relies on. A probe at weight 50 would
# answer the WRONG question (it would report MERGE for AMPA recurrence that has nothing to do with the dlPFC's
# actual bistability mechanism, and the dt sweep would be meaningless). At weight 30 the loop sits in the clean
# NMDA-dependent regime: at dt=0.5 the latch persists with NMDA ON (post≈0.10) and COLLAPSES to baseline with
# NMDA OFF (post≈0.00). THAT is the real working-memory mechanism, and the regime where the dt question is
# meaningful. (The probe asserts this NMDA-dependence at dt=0.5 as a non-vacuity guard.)
PATTERN_SIZE = 50
ATTRACTOR_WEIGHT = 30.0
DRIVE_PA = 2500.0
DRIVE_STEPS = 60          # DRIVE window: hold the concept's cortex pattern (steps; identical step COUNT at both dt)
POST_DRIVE_STEPS = 80     # POST-DRIVE window: drive removed, measure the latch (the persistence)


def build_loop_wm_bridge_dt(n=600, density=0.0, loop_weight=0.0, loop_density=0.05, seed=42, dt=0.5,
                            enable_ou=False, nmda=True, verbose=False):
    """Runner-side COPY of `content_selection_spiking.build_loop_wm_bridge` with `dt` + `nmda` kwargs (the
    shipped one hardcodes dt_ms=0.5 and NMDA on). Two mutually-exciting regions forming the cortex_ctx ↔
    dlpfc_wm LOOP. All construction is otherwise faithful to the shipped builder; the swept variables are `dt`
    (the decision) and `nmda` (the non-vacuity control: with NMDA OFF the latch must collapse, proving the
    dt=0.5 persistence is the NMDA-dependent working-memory mechanism, not trivial AMPA recurrence).

    Defaults match the validated CLEAN-WM config (`SpikingLoopContextBuffer` / `SpikingController`):
    loop_weight=0 (the installed concept attractors are the ONLY loop connections — no generic random
    reverberation to bleed the driven pattern into undriven ones), internal density=0 (no random within-region
    recurrence coupling separate attractors), enable_ou=False (a quiet, noise-robust hold).
    """
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    def reg(name):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=density,
                           exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                           plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=bool(nmda))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [reg("cortex_ctx"), reg("dlpfc_wm")]
    cfg.region_pathways = [
        RegionPathway(from_region="cortex_ctx", to_region="dlpfc_wm", density=loop_density,
                      weight_mean=loop_weight, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="dlpfc_wm", to_region="cortex_ctx", density=loop_density,
                      weight_mean=loop_weight, weight_jitter=0.2, plastic=False),
    ]
    cfg.dt_ms = float(dt)          # <-- swept (decision); shipped builder hardcodes 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = bool(nmda)   # <-- swept (non-vacuity control); shipped builder hardcodes True
    cfg.enable_ou_process = bool(enable_ou)
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 30.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if verbose:
        print(f"[loop WM bridge dt={dt}] cortex_ctx<->dlpfc_wm loop, {n} neurons each, "
              f"NMDA={'on' if nmda else 'off'}, OU={'on' if enable_ou else 'off'}", flush=True)
    return bridge


def _install_attractor(bridge, cpat, dpat, weight=ATTRACTOR_WEIGHT):
    """Install one concept's pattern-specific attractor: outer-product cortex_assembly ↔ dlPFC_assembly weights
    (cortex_A drives dlPFC_A and dlPFC_A drives cortex_A), exactly as `SpikingLoopContextBuffer` does. This is
    the loop's only excitatory coupling (loop_weight=0), so the attractor IS the persistence mechanism."""
    ps = len(cpat)
    pre1 = np.repeat(cpat, ps).astype(np.int64)
    post1 = np.tile(dpat, ps).astype(np.int64)
    pre2 = np.repeat(dpat, ps).astype(np.int64)
    post2 = np.tile(cpat, ps).astype(np.int64)
    ww = np.full(ps * ps, float(weight), np.float32)
    bridge.set_pathway_weights("c2d", pre_indices=pre1, post_indices=post1, weights=ww, add_missing=True)
    bridge.set_pathway_weights("d2c", pre_indices=pre2, post_indices=post2, weights=ww, add_missing=True)


def _assembly_rate(bridge, idx, xp, to_host):
    """Per-neuron firing fraction of the assembly `idx` on the just-stepped bridge (one step's firing states)."""
    return float(to_host(bridge.cp_firing_states[idx].astype(xp.float64).mean()))


def _run_persistence(bridge, driven_c, control_c, xp, to_host):
    """Drive the DRIVEN concept's cortex assembly for DRIVE_STEPS, accumulating its during-drive rate; then
    REMOVE the drive and run POST_DRIVE_STEPS with NO input, accumulating the driven assembly's post-drive rate
    (the PERSISTENCE) and an UN-driven control assembly's post-drive rate (specificity). Returns the three rates.
    Advances the clock each step (CLAUDE.md gotcha: _run_one_simulation_step does NOT advance current_time_ms)."""
    cfg = bridge.core_config
    drv = driven_c["cortex"]

    # DRIVE window — hold the concept's cortex pattern; the attractor + loop reverberation build the latch.
    acc_drive = 0.0
    for _ in range(DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[drv] = DRIVE_PA
        bridge.runtime_state.current_time_ms += cfg.dt_ms
        bridge._run_one_simulation_step()
        acc_drive += _assembly_rate(bridge, driven_c["cortex"], xp, to_host)

    # POST-DRIVE window — drive removed; does the assembly keep firing (working-memory persistence)?
    bridge.cp_external_input_current[:] = 0.0
    acc_post_driven = 0.0
    acc_post_control = 0.0
    for _ in range(POST_DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.runtime_state.current_time_ms += cfg.dt_ms
        bridge._run_one_simulation_step()
        acc_post_driven += _assembly_rate(bridge, driven_c["cortex"], xp, to_host)
        acc_post_control += _assembly_rate(bridge, control_c["cortex"], xp, to_host)
    return (acc_drive / DRIVE_STEPS, acc_post_driven / POST_DRIVE_STEPS, acc_post_control / POST_DRIVE_STEPS)


def _run_no_drive_baseline(bridge, driven_c, xp, to_host):
    """The no-drive floor: run POST_DRIVE_STEPS with NO input from rest, measuring the driven assembly's
    spontaneous rate. The persistence (driven_post_drive) must clear THIS baseline to count as a latch."""
    cfg = bridge.core_config
    acc = 0.0
    for _ in range(POST_DRIVE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge.runtime_state.current_time_ms += cfg.dt_ms
        bridge._run_one_simulation_step()
        acc += _assembly_rate(bridge, driven_c["cortex"], xp, to_host)
    return acc / POST_DRIVE_STEPS


def _make_concepts(bridge, n, pattern_size, seed):
    """Two disjoint concept assemblies (driven + control), each a sparse subset of the cortex_ctx and dlpfc_wm
    index ranges, laid out exactly as `SpikingLoopContextBuffer` does (a shared permutation indexing the
    region's cortex and dlPFC index arrays so cortex_A and dlPFC_A are aligned for the outer-product attractor).
    """
    import sim.backend as B
    rm = bridge.region_manager
    cidx = np.asarray(rm.indices("cortex_ctx"))
    didx = np.asarray(rm.indices("dlpfc_wm"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    out = []
    for i in range(2):
        p = perm[i * pattern_size:(i + 1) * pattern_size]
        out.append({"cortex": cidx[p], "dlpfc": didx[p]})
    xp, _ = B.get_backend()
    for c in out:
        c["cortex"] = xp.asarray(c["cortex"])
        c["dlpfc"] = xp.asarray(c["dlpfc"])
        c["cortex_host"] = B.to_host(c["cortex"])
        c["dlpfc_host"] = B.to_host(c["dlpfc"])
    return out


def run_one_dt(dt, seed=42, n=600, pattern_size=PATTERN_SIZE, nmda=True, verbose=False):
    """Build the loop bridge at `dt` (and `nmda`), install the DRIVEN concept's attractor, and measure the four
    rates (driven during drive / driven post-drive / control post-drive / no-drive baseline). The no-drive
    baseline is measured on a FRESH bridge (same dt/seed/nmda/attractor) so a leftover latch from the
    persistence run cannot contaminate the floor."""
    import sim.backend as B
    xp, _ = B.get_backend()
    to_host = B.to_host

    # --- persistence run: drive the concept, remove drive, measure the latch ---
    bridge = build_loop_wm_bridge_dt(n=n, seed=seed, dt=dt, nmda=nmda, verbose=verbose)
    concepts = _make_concepts(bridge, n, pattern_size, seed)
    driven_c, control_c = concepts[0], concepts[1]
    _install_attractor(bridge, driven_c["cortex_host"], driven_c["dlpfc_host"])   # only the DRIVEN concept latches
    during, post_driven, post_control = _run_persistence(bridge, driven_c, control_c, xp, to_host)

    # --- no-drive baseline: fresh identical bridge, never driven ---
    bridge2 = build_loop_wm_bridge_dt(n=n, seed=seed, dt=dt, nmda=nmda, verbose=False)
    concepts2 = _make_concepts(bridge2, n, pattern_size, seed)
    _install_attractor(bridge2, concepts2[0]["cortex_host"], concepts2[0]["dlpfc_host"])
    baseline = _run_no_drive_baseline(bridge2, concepts2[0], xp, to_host)

    return {
        "dt": float(dt),
        "nmda": bool(nmda),
        "driven_during_drive": during,
        "driven_post_drive": post_driven,       # THE PERSISTENCE (decides MERGE vs BOUNDARY)
        "control_post_drive": post_control,      # specificity control (un-driven assembly)
        "no_drive_baseline": baseline,           # the floor the persistence must clear
    }


def run_dlpfc_dt_probe(seed=42, n=600, pattern_size=PATTERN_SIZE, verbose=False):
    """Run the decision: measure the loop's persistence at dt=0.5 (tuned baseline) AND dt=1.0 (parser/composer
    timestep). ALSO measures the dt=0.5 NMDA-OFF control — the non-vacuity guard that the dt=0.5 latch is the
    GENUINELY NMDA-DEPENDENT working-memory mechanism (with NMDA off it must collapse to baseline), not trivial
    AMPA recurrence. Returns {0.5: {...}, 1.0: {...}, "0.5_nmda_off": {...}} consumed by
    `test_step3_dlpfc_bistability_survives_dt1`."""
    return {
        0.5: run_one_dt(0.5, seed=seed, n=n, pattern_size=pattern_size, nmda=True, verbose=verbose),
        1.0: run_one_dt(1.0, seed=seed, n=n, pattern_size=pattern_size, nmda=True, verbose=verbose),
        "0.5_nmda_off": run_one_dt(0.5, seed=seed, n=n, pattern_size=pattern_size, nmda=False, verbose=verbose),
    }


def _decision(res):
    """Plain-language decision line from the measured rates (MERGE bar: dt=1.0 post-drive ≥ 70% of dt=0.5
    post-drive AND clearly above the no-drive baseline)."""
    r05, r10 = res[0.5], res[1.0]
    bar = 0.70 * r05["driven_post_drive"]
    merge = (r10["driven_post_drive"] >= bar and
             r10["driven_post_drive"] > r05["no_drive_baseline"] + 0.05)
    pct = (100.0 * r10["driven_post_drive"] / r05["driven_post_drive"]) if r05["driven_post_drive"] > 0 else 0.0
    return ("MERGE" if merge else "BOUNDARY"), pct, bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--pattern-size", type=int, default=PATTERN_SIZE)
    a = ap.parse_args()

    res = run_dlpfc_dt_probe(seed=a.seed, n=a.n, pattern_size=a.pattern_size, verbose=True)
    print("\n=== dlPFC loop persistence vs dt (step-3 Task-1 decision) ===")
    hdr = f"{'config':>16}  {'driven_during':>14}  {'driven_post(PERSIST)':>20}  {'control_post':>13}  {'no_drive_base':>14}"
    print(hdr)
    print("-" * len(hdr))
    for key, label in ((0.5, "dt=0.5 NMDA-on"), (1.0, "dt=1.0 NMDA-on"),
                       ("0.5_nmda_off", "dt=0.5 NMDA-OFF")):
        r = res[key]
        print(f"{label:>16}  {r['driven_during_drive']:>14.4f}  {r['driven_post_drive']:>20.4f}  "
              f"{r['control_post_drive']:>13.4f}  {r['no_drive_baseline']:>14.4f}")

    off = res["0.5_nmda_off"]
    print(f"\nNon-vacuity (NMDA-dependence) check at dt=0.5: NMDA-on post={res[0.5]['driven_post_drive']:.4f} "
          f"vs NMDA-OFF post={off['driven_post_drive']:.4f} -> the latch is "
          f"{'GENUINELY NMDA-DEPENDENT' if off['driven_post_drive'] < res[0.5]['driven_post_drive'] * 0.5 else 'NOT clearly NMDA-dependent (suspect trivial AMPA recurrence)'}.")

    decision, pct, bar = _decision(res)
    print(f"\nMERGE bar (>=70% of dt=0.5 persist AND > baseline): {bar:.4f}")
    print(f"dt=1.0 retains {pct:.1f}% of the dt=0.5 post-drive persistence.")
    print(f"\nDECISION: {decision}")
    if decision == "MERGE":
        print("  -> the dlPFC working-memory latch survives dt=1.0; it can merge onto the unified bridge "
              "(step-3 Task 2). Write 2026-06-04-step3-dlpfc-dt-survives.md.")
    else:
        print("  -> the dlPFC working-memory latch COLLAPSES at dt=1.0 (working-memory timescale != binding "
              "timescale). The dlPFC stays a SEPARATE-timing region; do NOT force a merge. Write "
              "2026-06-04-step3-dlpfc-dt-BOUNDARY.md.")


if __name__ == "__main__":
    main()
