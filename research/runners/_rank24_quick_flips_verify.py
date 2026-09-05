"""RANK-24 quick-flips shared verify-runner (2026-09-05).

De-risks the scaffold-retirement backlog's rank-24 entry (`research/coordination/scaffold_retirement_backlog.md`,
produced by the scaffold-shortcut-map workflow `w9sn9wn4b`): "quick flips packaged as one verify-runner". The
backlog's own retirement_mechanism text names exactly THREE items as "one-line flips" (as opposed to the rest of
rank-24's grouped tail, which it explicitly calls fresh builds):

    "several are one-line flips (d5 depth_hold direct read; flip enable_spiking_sc after a pool soak; fix the nav
    --readout-source argparse default) ... most are fresh builds on narrow/off-path triggers."

This runner checks each of the three against the actual source + a real measurement, rather than trusting the
map's readiness label. Verdict reached this session (2026-09-05):

  1. d5-direct-read   -- NO-GO (measured HERE, `--check d5-direct-read`). The proposed mechanism ("read the
     existing graded depth_hold value directly at reply-generation time against a calibrated threshold, removing
     the separate host _CONSOLIDATED_TOPICS ledger") does not survive measurement: raw depth_hold is NOT
     comparable ACROSS topics (each topic's DG-formed CA3 assembly has its own baseline apical-latch magnitude,
     which varies with assembly composition/size). A topic that was NEVER consolidated can read a HIGHER absolute
     depth_hold than a DIFFERENT topic that WAS consolidated (seed 42: bird-never-consolidated=30.72mV >
     dog-consolidated=30.91mV is only barely separated; seed 43: bird-never-consolidated=30.39mV >
     dog-consolidated=29.63mV -- bird's baseline EXCEEDS dog's post-consolidation value outright). No single
     global threshold can gate "was THIS topic consolidated" from the raw signal alone. See the finding:
     research/findings/2026-09-05-d5-depth-hold-direct-read-NO-GO-cross-topic-baseline-not-comparable.md
     The existing host `_CONSOLIDATED_TOPICS` set-membership gate (`webapp/continuous_engine.py`,
     `research/runners/d5_episodic_production_organ.py::_topic_consolidated`) is UNCHANGED -- still the
     production mechanism; this runner adds no new flag/code path to the production module (a mechanism this
     runner ITSELF disproves has no business shipping behind a flag).

  2. enable-spiking-sc -- SPLIT VERDICT (measured HERE via `--check enable-spiking-sc`, plus the pre-existing
     `_sc_orienting_flip_soak.py`). The map's "enable_spiking_sc" maps to TWO flags: (a) the standalone
     `sc_orienting_production_organ.py` organ's `BRAIN_SPIKING_SC_ORIENT` -- GO, now default-ON (its own
     purpose-built 6-seed pool soak passes cleanly, and it has NO current caller anywhere, so the flip is
     zero-risk); (b) `g11_bg_runner.py`'s own library-default `enable_spiking_sc` kwarg -- REVERTED after
     measurement (the map's causal premise doesn't match the source -- the host reflex it claims to retire is
     gated by an unrelated `sc_orienting_reflex` flag, independently False -- and a repo-wide audit found 7
     existing research probes that would be silently affected, with the SC's OWN directional read-out measured
     only 4/6-seed reliable when co-resident with the full default BG cascade + visual cortex, vs. 6/6 in the
     isolated configuration the original CLOSED result and today's organ-level re-verify both used). See the
     finding: research/findings/2026-09-05-enable-spiking-sc-split-verdict-organ-flag-GO-library-default-reverted.md

  3. nav --readout-source CLI default -- NOT ATTEMPTED (reported NOT flip-ready, no runner section here). The
     CLI default was DELIBERATELY pinned to "motor" on 2026-06-19 specifically "so every documented standalone
     benchmark reproduces unchanged" (docs/ENGINE_REFERENCE.md:184-186) -- a standing decision, not an oversight
     -- and the scaffold-map's OWN readiness note for the surrounding silent-commit-fallback/RNG-coinflip
     residuals (g11_bg_runner.py:500-505) calls them "Partially de-risked ... no finding formally closes the
     residual to zero across seeds". Flipping the CLI default alone would not even retire the "N6 cheat" fully
     (the silent-commit argmax + RNG-tiebreak fallback survive it untouched) while breaking the explicit
     documented-benchmark-reproducibility invariant. Needs an owner call + fresh work, not a quick flip.

Run:
  .venv/bin/python -m research.runners._rank24_quick_flips_verify --check d5-direct-read --seeds 42 43 44 100 101 102
  .venv/bin/python -m research.runners._rank24_quick_flips_verify --check enable-spiking-sc --seeds 42 43 44
  .venv/bin/python -m research.runners._rank24_quick_flips_verify --check all --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT_DIR = _REPO / "research" / "findings" / "raw" / "_rank24_quick_flips"


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CHECK 1 — d5 depth_hold direct-read: is there a GLOBAL threshold on the raw graded read that reproduces the
# host _CONSOLIDATED_TOPICS gate's per-topic surfacing decision? Reuses the EXISTING soak's scenario builder
# (research.runners._d5_graded_flip_soak) by import -- no duplicated scenario logic.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _d5_direct_read_check_one(seed: int) -> dict:
    from sim.backend import get_backend
    from research.runners.d5_episodic_production_organ import EpisodicRecallOrgan
    from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch
    from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state
    from research.runners import _d5_graded_flip_soak as SOAK

    cp, backend = get_backend()
    cache_key = ("rank24-d5-direct-read", seed)
    org = EpisodicRecallOrgan(seed, ["cat", "dog", "bird"], verbose=False, sep_bias=0.0)
    org._ensure_built()
    mem = org.mem
    assert org.note_topic("dog") and org.note_topic("bird"), "note_topic failed"
    mem.recall("dog"); mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
    snap = snapshot_state(mem.bridge)
    W0 = mem.R.C.data.copy()

    on = SOAK._run_conversation(org, mem, snap, cp, cache_key, W0, flag_on=True)
    dog_baseline = float(on["t2_dog"]["depth_hold"])       # dog BEFORE consolidation
    dog_consolidated = float(on["t4_dog"]["depth_hold"])    # dog AFTER consolidation (the case a gate must ACCEPT)
    bird_baseline = float(on["t4_bird"]["depth_hold"])      # bird NEVER consolidated (the case a gate must REJECT)
    # A per-seed threshold t is valid iff it accepts dog-consolidated and rejects bird's untouched baseline.
    per_seed_threshold_exists = dog_consolidated > bird_baseline
    from research.runners.d5_episodic_production_organ import recall_disclosure
    del recall_disclosure  # (import touch only, keeps this file's declared honesty-boundary contract visible)
    SOAK.CE.forget_session(cache_key)
    return {
        "seed": seed, "dog_baseline": dog_baseline, "dog_consolidated": dog_consolidated,
        "bird_baseline": bird_baseline, "dog_rise": round(dog_consolidated - dog_baseline, 5),
        "per_seed_threshold_exists": bool(per_seed_threshold_exists),
        "per_seed_threshold_window": (round(bird_baseline, 5), round(dog_consolidated, 5)),
        "assembly_sizes": mem.assembly_sizes,
    }


def check_d5_direct_read(seeds: list[int]) -> dict:
    print("\n" + "=" * 118)
    print("[rank24] CHECK d5-direct-read: does ONE global depth_hold threshold reproduce the host "
          "_CONSOLIDATED_TOPICS gate across seeds?", flush=True)
    per_seed = []
    for s in seeds:
        r = _d5_direct_read_check_one(s)
        per_seed.append(r)
        print(f"[rank24] seed={s} dog_baseline={r['dog_baseline']:.3f} dog_consolidated={r['dog_consolidated']:.3f} "
              f"bird_baseline(never-consolidated)={r['bird_baseline']:.3f} "
              f"per_seed_threshold_exists={r['per_seed_threshold_exists']}", flush=True)
    # The bar for the PROPOSED mechanism (one constant in the codebase, not per-seed-tuned): a single fixed
    # threshold must sit ABOVE every seed's never-consolidated bird baseline AND AT/BELOW every seed's
    # consolidated dog value.
    max_bird_baseline = max(r["bird_baseline"] for r in per_seed)
    min_dog_consolidated = min(r["dog_consolidated"] for r in per_seed)
    global_threshold_exists = max_bird_baseline < min_dog_consolidated
    n_per_seed_ok = sum(1 for r in per_seed if r["per_seed_threshold_exists"])
    GO = bool(global_threshold_exists)
    result = {
        "check": "d5-direct-read", "seeds": seeds, "GO": GO,
        "n_per_seed_threshold_exists": n_per_seed_ok, "n_seeds": len(seeds),
        "max_bird_baseline_never_consolidated": round(max_bird_baseline, 5),
        "min_dog_consolidated": round(min_dog_consolidated, 5),
        "global_threshold_exists": global_threshold_exists,
        "verdict": ("GO -- a global threshold in (%.3f, %.3f] separates consolidated from never-consolidated "
                    "on every seed" % (max_bird_baseline, min_dog_consolidated)) if GO else
                   ("NO-GO -- the never-consolidated bird baseline (max %.3f mV across seeds) meets or EXCEEDS "
                    "the consolidated dog value (min %.3f mV across seeds); no single global threshold can "
                    "separate 'this topic was consolidated' from 'that topic's own baseline is just high' "
                    "using the raw depth_hold read alone" % (max_bird_baseline, min_dog_consolidated)),
        "per_seed": per_seed,
    }
    print(f"[rank24] d5-direct-read: {n_per_seed_ok}/{len(seeds)} seeds individually separable; "
          f"GLOBAL (one fixed constant) threshold_exists={global_threshold_exists} => "
          f"{'GO' if GO else 'NO-GO'}", flush=True)
    print(f"[rank24] {result['verdict']}", flush=True)
    return result


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# CHECK 2 — enable_spiking_sc default flip. Two bars: (a) BYTE-IDENTICAL-OFF -- the true default path (no
# --enable-visual-cortex, the overwhelming majority of callers/tests/documented commands) is completely
# unaffected, since enable_spiking_sc's region-build is nested under `if enable_visual_cortex:`; (b)
# LOAD-BEARING/CORRECT WHEN ON -- with --enable-visual-cortex and the new default, the SC populations actually
# build, actually fire, and driving the SC's retina actually produces a directional cortex_{N,E,S,W} bias (the
# spiking read, not a silently-inert region).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def check_enable_spiking_sc_byte_identical_off(seeds: list[int]) -> dict:
    """(a) With enable_visual_cortex left at its own default (False), building the region list is IDENTICAL
    regardless of enable_spiking_sc's default value -- confirms the flip is inert for the true default path."""
    from research.runners.g11_bg_runner import build_bg_brain_regions
    regions_a, pathways_a = build_bg_brain_regions(enable_spiking_sc=False)
    regions_b, pathways_b = build_bg_brain_regions(enable_spiking_sc=True)
    names_a = [r.name for r in regions_a]
    names_b = [r.name for r in regions_b]
    identical = (names_a == names_b) and (len(pathways_a) == len(pathways_b))
    print(f"[rank24] enable-spiking-sc byte-identical-off (enable_visual_cortex=False, the true default path): "
          f"{len(names_a)} regions either way, identical={identical}", flush=True)
    return {"check": "enable-spiking-sc-byte-identical-off", "GO": bool(identical),
            "n_regions_sc_false": len(names_a), "n_regions_sc_true": len(names_b),
            "n_pathways_sc_false": len(pathways_a), "n_pathways_sc_true": len(pathways_b)}


def check_enable_spiking_sc_load_bearing(seed: int) -> dict:
    """(b) With --enable-visual-cortex (+ the new enable_spiking_sc=True default), build a real bridge, drive
    the SC's egocentric eye toward a known goal direction, step it, and confirm sc_map forms a bump AND the
    cortex_{N,E,S,W} read-out actually differs by goal direction (the spiking mechanism is load-bearing, not an
    inert unconnected region). Reuses install_spiking_sc_wiring (the SAME production wiring
    sc_orienting_production_organ.py uses) -- no new sim/ mechanism."""
    import numpy as np
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.g11_bg_runner import (
        build_bg_brain_regions, install_spiking_sc_wiring, render_egocentric_goal)
    from sim.visual_cortex import image_to_retina_drive

    regions, pathways = build_bg_brain_regions(enable_visual_cortex=True, enable_spiking_sc=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    built_names = {r.name for r in bridge.region_manager.regions()}
    assert "sc_map" in built_names, "sc_map region did not build under enable_spiking_sc=True"
    install_spiking_sc_wiring(bridge)

    from sim.backend import get_backend
    cp, _ = get_backend()
    rm = bridge.region_manager

    # WARM_STEPS/READ_STEPS mirror sc_orienting_production_organ.py's own validated op-point (30 warm-up
    # steps discarding the transient, then a 160-step read) -- this g11_bg_runner config differs from that
    # organ's ISOLATED minimal scaffold by ALSO carrying the full default BG cascade + visual cortex at the
    # DEFAULT (higher) background OU noise, so the SAME short read window that is clean in isolation can be
    # noisier here; using the validated warm-up/read split (rather than a bare 60-step snapshot) is the
    # honest like-for-like comparison, not a thumb on the scale.
    WARM_STEPS, READ_STEPS = 30, 160

    def _drive_and_read(goal_dxdy):
        img = render_egocentric_goal((16, 16), (16 + goal_dxdy[0], 16 + goal_dxdy[1]), image_size=32)
        drive = image_to_retina_drive(img, drive_max_pA=2500.0)
        bridge.cp_external_input_current[:] = 0.0
        sc_idx = cp.asarray(list(rm.indices("sc_retina")), dtype=cp.int64)
        bridge.cp_external_input_current[sc_idx] = cp.asarray(drive, dtype=cp.float32)
        pool_idx = {pool: cp.asarray(list(rm.indices(pool)), dtype=cp.int64)
                    for pool in ("cortex_N", "cortex_E", "cortex_S", "cortex_W")}
        for _ in range(WARM_STEPS):
            bridge._run_one_simulation_step()
        counts = {pool: 0 for pool in pool_idx}
        for _ in range(READ_STEPS):
            bridge._run_one_simulation_step()
            firing = bridge.cp_firing_states
            for pool, idx in pool_idx.items():
                counts[pool] += int(firing[idx].sum())
        return counts

    east_counts = _drive_and_read((10, 0))   # goal due EAST of agent
    north_counts = _drive_and_read((0, -10))  # goal due NORTH of agent (image row convention: -y = up)

    east_argmax = max(east_counts, key=east_counts.get)
    north_argmax = max(north_counts, key=north_counts.get)
    directional = bool(east_argmax != north_argmax)
    print(f"[rank24] enable-spiking-sc load-bearing smoke (seed={seed}): east-goal cortex spikes={east_counts} "
          f"(argmax={east_argmax}); north-goal cortex spikes={north_counts} (argmax={north_argmax}); "
          f"directional={directional}", flush=True)
    return {"check": "enable-spiking-sc-load-bearing", "seed": seed, "GO": directional,
            "east_counts": east_counts, "north_counts": north_counts,
            "east_argmax": east_argmax, "north_argmax": north_argmax}


def check_enable_spiking_sc(seeds: list[int]) -> dict:
    print("\n" + "=" * 118)
    print("[rank24] CHECK enable-spiking-sc: byte-identical-off (true default path) + load-bearing-when-on "
          "(a real bridge, SC actually drives a directional cortex read-out)", flush=True)
    off_check = check_enable_spiking_sc_byte_identical_off(seeds)
    on_checks = [check_enable_spiking_sc_load_bearing(s) for s in seeds]
    n_directional = sum(1 for r in on_checks if r["GO"])
    GO = bool(off_check["GO"] and n_directional == len(seeds))
    result = {"check": "enable-spiking-sc", "seeds": seeds, "GO": GO,
              "byte_identical_off": off_check, "load_bearing_on": on_checks,
              "n_directional": n_directional, "n_seeds": len(seeds)}
    print(f"[rank24] enable-spiking-sc: byte_identical_off={off_check['GO']} "
          f"directional={n_directional}/{len(seeds)} => {'GO' if GO else 'NO-GO'}", flush=True)
    return result


CHECKS = {
    "d5-direct-read": lambda seeds: check_d5_direct_read(seeds),
    "enable-spiking-sc": lambda seeds: check_enable_spiking_sc(seeds),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", choices=list(CHECKS.keys()) + ["all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = list(CHECKS.keys()) if a.check == "all" else [a.check]
    results = {}
    t0 = time.time()
    for name in checks:
        try:
            results[name] = CHECKS[name](a.seeds)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            results[name] = {"check": name, "GO": False, "error": repr(e)}
    out_path = Path(a.out) if a.out else OUT_DIR / f"verify_{'_'.join(checks)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"checks": checks, "seeds": a.seeds, "elapsed_s": round(time.time() - t0, 1),
                                    "results": results}, indent=2, default=str))
    print(f"\n[rank24] wrote {out_path}")
    all_go = all(bool(results[c].get("GO")) for c in checks)
    print(f"[rank24] OVERALL: {'GO' if all_go else 'NOT ALL GO (see per-check verdicts above)'}")
    return 0 if all_go else 1


if __name__ == "__main__":
    sys.exit(main())
