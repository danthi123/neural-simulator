"""Profile the per-step spiking-simulation kernel bottleneck at composer-representative scale
(n_neurons=64,324, SIM_BACKEND=cupy) and measure the REAL, at-scale speedup of the existing-but-
never-measured-at-this-scale `enable_branchless_plasticity` fast path against today's default
STDP-enabled learning-path cost.

WHY / context (see docs/TERMS.md before citing any term from this runner's output as settled):
research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md named the
per-step `_run_one_simulation_step()` cost at `n_neurons=64,324` as the dominant, unaddressed latency driver
behind the composer's 76x latency vs FHRR, and explicitly flagged a GPU/cupy re-verify as NOT attempted there
(that finding measured CPU/numpy only, per its own S7/cost-routing note). This runner IS that re-verify, plus:

  1. A phase-attribution ABLATION (pure inference vs +STDP, fused-megakernel vs unfused python) at the exact
     composer scale, isolating STDP's own marginal cost from the Izhikevich-update + synaptic-matvec cost.
  2. A scale SWEEP reproducing this project's own established launch-bound diagnostic
     (research/findings/2026-07-24-gap4-onbridge-spiking-6seed-nothing-learns-LAUNCH-BOUND-compute-wall.md's
     "steps/sec flat across a network-size increase" method) up through n=64,324 -- prior art only measured
     this up to ~2,400 neurons; docs/plans/2026-07-23-general-step-megakernel-design.md's own prediction that
     the fused-inference win "tapers to ~1x past ~50-100K where CSR compute dominates" was never tested at the
     real composer scale. This measures it directly instead of assuming it.
  3. A real at-scale measurement of `cfg.enable_branchless_plasticity` (sim/bridge.py `_apply_branchless_stdp`)
     -- an ADDITIVE, default-OFF, already-byte-identical-tested (tests/test_branchless_plasticity.py) fast path
     that removes the STDP compacting sync, whose only prior speedup evidence is a 100K-1M-nnz microbench
     estimate in research/findings/2026-07-23-perf-scoping-adex-hh-learning-megakernel.md -- NOT a full bridge
     run at the true ~28.6M-nnz composer scale. (Commit history -- 3f6bf4354/37f34530c -- already retracted one
     prior claim about this exact flag for citing the code without checking it actually dispatches; this runner
     asserts non-vacuousness and dispatch state directly rather than citing the source.)
  4. GPU utilization sampling (nvidia-smi) during the learning-path config, against this task's own cited ~30%
     GPU-util figure.
  5. A cProfile function-level pass over the default learning-path config as corroborating (not authoritative)
     evidence for which phase dominates.

Run (GPU-exclusive -- MUST go via tools/gpu_queue.sh, never invoked directly):
  SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._perf_step_kernel_profile \
      --out research/findings/raw/_perf_step_kernel_profile/result.json

CPU smoke test (catches Python bugs cheaply before spending a GPU slot -- runs the SAME ablation code path on
tiny numpy-backend networks; the megakernel guard always fails on numpy, so this does NOT exercise the real
speedup, only correctness/non-crash of the harness itself):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._perf_step_kernel_profile --smoke
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
import time

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim import (  # noqa: E402
    SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig, NeuronModel,
)
from sim.backend import is_gpu_backend, get_backend, to_host, synchronize  # noqa: E402

# Composer-representative scale: the L3 finding's S3/S7 name n_neurons=64,324 / ~28.6M synapses as the real
# K=2020/KF=1195 production topology's neuron/synapse count. connections_per_neuron=445 reproduces the SAME
# edge/neuron ratio (28.6M / 64,324 = 444.6) with a generic random sparse Izhikevich network, instead of paying
# SlotBinderComposer's own ~14 CPU-hour teach-cost setup (L3 S5) just to stress the identical
# `_run_one_simulation_step()` cost this task is actually about.
N_COMPOSER = 64324
FANIN_COMPOSER = 445

_READONLY = dict(
    neuron_model_type=NeuronModel.IZHIKEVICH.name,
    fast_spike_reset=True,
    read_only_fast_step=True,
    enable_hebbian_learning=False,
    enable_short_term_plasticity=False,
    enable_homeostasis=False,
    enable_stdp=False,
    enable_structural_plasticity=False,
    enable_reward_modulation=False,
)


def _build(n, fan_in, seed=42, **overrides):
    cfg = CoreSimConfig(
        num_neurons=n, connections_per_neuron=fan_in, seed=seed, dt_ms=1.0,
        **{**_READONLY, **overrides},
    )
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig(enable_profiling=False))
    b._initialize_simulation_data()
    return b


def _drive(b, n, frac=0.2, current_pA=2500.0):
    """Sustained strong depolarizing current to a FRACTION of neurons -- drives genuine, repeated firing (not a
    synthetic last_spike_time hack), so STDP's candidate_mask reflects real (pre,post) activity each step."""
    xp, _ = get_backend()
    d = xp.zeros(n, dtype=xp.float32)
    k = max(1, int(n * frac))
    d[:k] = current_pA
    b.cp_external_input_current[:] = d


def _step(b):
    b._run_one_simulation_step()
    # STDP correctness precondition (documented guard, bridge.py ~10160): _run_one_simulation_step() does NOT
    # advance the clock -- only step_simulation() does. Every research runner that drives the low-level step
    # directly must advance it itself, or every spike shares one timestamp, delta_t==0 for every pair, and
    # fused_stdp_weight_update returns exactly 0.0 (the "656k STDP events, 0 weight change" instrument-artifact
    # bug this project already banked and fixed a guard for). This is a TIMING run, not a correctness run, so
    # the launch/sync pattern would be identical either way -- but advancing it removes any doubt and keeps
    # this runner's own STDP non-vacuousness assertions meaningful.
    b.runtime_state.current_time_ms += b.core_config.dt_ms


def _timed_run(b, n, n_steps, warmup, frac=0.2):
    _drive(b, n, frac=frac)
    for _ in range(warmup):
        _step(b)
    synchronize()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _step(b)
    synchronize()
    dt = time.perf_counter() - t0
    return dt / n_steps * 1e3, n_steps / dt  # (ms/step, steps/sec)


def _gpu_util_during(b, n, n_steps, frac=0.2):
    """Sample nvidia-smi GPU/mem utilization concurrently with a step loop. Returns
    (avg_util_pct, avg_mem_util_pct, n_samples, wall_seconds)."""
    _drive(b, n, frac=frac)
    for _ in range(10):
        _step(b)
    synchronize()
    proc = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
         "--format=csv,noheader,nounits", "-lms", "200"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _step(b)
    synchronize()
    wall = time.perf_counter() - t0
    time.sleep(0.3)  # let one more sample land before killing the sampler
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        out = ""
    samples = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            try:
                samples.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    if not samples:
        return None, None, 0, wall
    util = [s[0] for s in samples]
    mem = [s[1] for s in samples]
    return sum(util) / len(util), sum(mem) / len(mem), len(samples), wall


def _nonvacuous_weight_change(b):
    xp, _ = get_backend()
    return to_host(b.cp_connections.data).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--skip-util", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU/numpy correctness smoke test, no GPU")
    args = ap.parse_args()

    global N_COMPOSER, FANIN_COMPOSER
    if args.smoke:
        N_COMPOSER, FANIN_COMPOSER = 300, 40
        args.n_steps, args.warmup = 20, 5
        args.skip_util = True

    result = {"backend_is_gpu": is_gpu_backend(), "smoke": args.smoke}
    if not args.smoke and not is_gpu_backend():
        print("NOT a GPU backend -- SIM_BACKEND=cupy required (or pass --smoke). Aborting.", flush=True)
        result["error"] = "not_gpu_backend"
        _dump(result, args.out)
        return

    print(f"[perf-step-profile] backend_is_gpu={is_gpu_backend()} smoke={args.smoke}. "
          f"Composer-representative scale: N={N_COMPOSER}, fan_in={FANIN_COMPOSER} "
          f"({N_COMPOSER * FANIN_COMPOSER:,} nnz)", flush=True)

    # ---------------------------------------------------------------------------------------------
    # PART 1: scale sweep at a FIXED modest fan_in -- reproduces the established launch-bound
    # diagnostic up through the actual composer scale (prior art stopped at ~2,400 neurons).
    # ---------------------------------------------------------------------------------------------
    sweep = []
    if not args.skip_sweep:
        print("\n=== PART 1: scale sweep (fan_in=100, inference-only): "
              "fused megakernel-v2 (default-on) vs forced-off python ===", flush=True)
        sizes = (300,) if args.smoke else (1000, 4000, 16000, N_COMPOSER)
        for n in sizes:
            fan_in = 40 if args.smoke else 100
            row = {"n_neurons": n, "connections_per_neuron": fan_in, "nnz": n * fan_in}

            b_off = _build(n, fan_in, enable_step_megakernel=False, enable_step_megakernel_v2=False)
            assert not b_off._step_megakernel_can_dispatch(), "sanity: forced-off config must NOT dispatch"
            ms_off, sps_off = _timed_run(b_off, n, args.n_steps, args.warmup)
            b_off.clear_simulation_state_and_gpu_memory()

            b_on = _build(n, fan_in)  # v2 default True
            can_dispatch_on = b_on._step_megakernel_can_dispatch()
            if not args.smoke:
                assert can_dispatch_on, "sanity: default read-only GPU config MUST dispatch v2"
            ms_on, sps_on = _timed_run(b_on, n, args.n_steps, args.warmup)
            b_on.clear_simulation_state_and_gpu_memory()

            row.update(python_ms_step=ms_off, python_steps_sec=sps_off,
                       megakernel_v2_ms_step=ms_on, megakernel_v2_steps_sec=sps_on,
                       megakernel_v2_dispatched=can_dispatch_on,
                       speedup=sps_on / sps_off)
            print(f"  N={n:>7,}  python={ms_off:9.5f} ms/step ({sps_off:10.1f} sps)   "
                  f"megakernel_v2={ms_on:9.5f} ms/step ({sps_on:10.1f} sps)   "
                  f"speedup={row['speedup']:.2f}x  dispatched={can_dispatch_on}", flush=True)
            sweep.append(row)
    result["scale_sweep_inference_only"] = sweep

    # ---------------------------------------------------------------------------------------------
    # PART 2: phase-attribution ablation AT THE COMPOSER-REPRESENTATIVE SCALE.
    #   A = pure inference, unfused python path (forced-off megakernel)
    #   B = pure inference, fused megakernel v2 (today's shipped DEFAULT for any plasticity-free bridge)
    #   C = STDP-only learning, unfused python path (today's shipped DEFAULT the instant enable_stdp=True
    #       -- the megakernel guard fails on enable_stdp regardless of the flag's own value, so C IS
    #       today's real production behavior for any learning bridge)
    #   D = STDP-only learning + enable_branchless_plasticity=True (existing, additive, default-OFF,
    #       byte-identical-tested flag -- never measured at this scale before; the STEP-2 prototype
    #       measurement this task asked for)
    # ---------------------------------------------------------------------------------------------
    print(f"\n=== PART 2: phase ablation at N={N_COMPOSER:,}, fan_in={FANIN_COMPOSER} "
          f"({N_COMPOSER * FANIN_COMPOSER:,} nnz) ===", flush=True)
    n, fan_in = N_COMPOSER, FANIN_COMPOSER
    timings = {}
    dispatch = {}
    nonvacuous = {}

    b = _build(n, fan_in, enable_step_megakernel=False, enable_step_megakernel_v2=False)
    dispatch["A_inference_python"] = b._step_megakernel_can_dispatch()
    assert not dispatch["A_inference_python"]
    timings["A_inference_python"] = _timed_run(b, n, args.n_steps, args.warmup)
    b.clear_simulation_state_and_gpu_memory()

    b = _build(n, fan_in)
    dispatch["B_inference_megakernel_v2"] = b._step_megakernel_can_dispatch()
    if not args.smoke:
        assert dispatch["B_inference_megakernel_v2"], "megakernel v2 must dispatch on the plain read-only build"
    timings["B_inference_megakernel_v2"] = _timed_run(b, n, args.n_steps, args.warmup)
    b.clear_simulation_state_and_gpu_memory()

    b = _build(n, fan_in, enable_stdp=True, enable_step_megakernel=False, enable_step_megakernel_v2=False)
    dispatch["C_stdp_compacting"] = b._step_megakernel_can_dispatch()
    assert not dispatch["C_stdp_compacting"], "sanity: STDP-on must NOT dispatch the megakernel"
    w0 = _nonvacuous_weight_change(b)
    timings["C_stdp_compacting"] = _timed_run(b, n, args.n_steps, args.warmup)
    w1 = _nonvacuous_weight_change(b)
    nonvacuous["C_stdp_compacting"] = bool((abs(w1 - w0) > 1e-9).sum() > 0)
    b.clear_simulation_state_and_gpu_memory()

    b = _build(n, fan_in, enable_stdp=True, enable_branchless_plasticity=True,
               enable_step_megakernel=False, enable_step_megakernel_v2=False)
    dispatch["D_stdp_branchless"] = b._step_megakernel_can_dispatch()
    assert not dispatch["D_stdp_branchless"]
    w0 = _nonvacuous_weight_change(b)
    timings["D_stdp_branchless"] = _timed_run(b, n, args.n_steps, args.warmup)
    w1 = _nonvacuous_weight_change(b)
    nonvacuous["D_stdp_branchless"] = bool((abs(w1 - w0) > 1e-9).sum() > 0)
    b.clear_simulation_state_and_gpu_memory()

    for k, (ms, sps) in timings.items():
        print(f"  {k:28s}: {ms:10.5f} ms/step  {sps:12.1f} steps/sec  dispatched={dispatch[k]}", flush=True)

    a_ms = timings["A_inference_python"][0]
    b_ms = timings["B_inference_megakernel_v2"][0]
    c_ms = timings["C_stdp_compacting"][0]
    d_ms = timings["D_stdp_branchless"][0]

    print(f"\n  inference fused speedup (A -> B):                       {a_ms / b_ms:.2f}x", flush=True)
    print(f"  STDP marginal cost over pure inference (python, C - A): {c_ms - a_ms:.5f} ms/step "
          f"({(c_ms - a_ms) / a_ms * 100:.1f}% of A)", flush=True)
    print(f"  branchless-plasticity speedup on the STDP path (C -> D): {c_ms / d_ms:.2f}x", flush=True)
    print(f"  STDP non-vacuous (weights actually moved)? compacting={nonvacuous['C_stdp_compacting']} "
          f"branchless={nonvacuous['D_stdp_branchless']}", flush=True)

    result["ablation_at_composer_scale"] = {
        "n_neurons": n, "connections_per_neuron": fan_in, "nnz": n * fan_in,
        "dispatch": dispatch,
        "nonvacuous": nonvacuous,
        "A_inference_python_ms_step": a_ms, "A_inference_python_steps_sec": timings["A_inference_python"][1],
        "B_inference_megakernel_v2_ms_step": b_ms,
        "B_inference_megakernel_v2_steps_sec": timings["B_inference_megakernel_v2"][1],
        "C_stdp_compacting_ms_step": c_ms, "C_stdp_compacting_steps_sec": timings["C_stdp_compacting"][1],
        "D_stdp_branchless_ms_step": d_ms, "D_stdp_branchless_steps_sec": timings["D_stdp_branchless"][1],
        "inference_fused_speedup_A_to_B": a_ms / b_ms,
        "stdp_marginal_cost_ms_C_minus_A": c_ms - a_ms,
        "stdp_marginal_cost_pct_of_A": (c_ms - a_ms) / a_ms * 100,
        "branchless_speedup_on_stdp_path_C_to_D": c_ms / d_ms,
    }

    # ---------------------------------------------------------------------------------------------
    # PART 3: GPU utilization sample during today's-default learning path (C) and the fused
    # inference path (B) at scale -- checked against this task's own cited ~30% GPU-util figure.
    # ---------------------------------------------------------------------------------------------
    if not args.skip_util:
        print(f"\n=== PART 3: GPU utilization sampling during config C (STDP-compacting, N={n:,}) ===", flush=True)
        b = _build(n, fan_in, enable_stdp=True, enable_step_megakernel=False, enable_step_megakernel_v2=False)
        util_steps = max(args.n_steps * 8, 3000)
        avg_util, avg_mem, n_samples, wall = _gpu_util_during(b, n, util_steps)
        b.clear_simulation_state_and_gpu_memory()
        print(f"  {n_samples} nvidia-smi samples over {wall:.2f}s ({util_steps} steps): "
              f"avg GPU util={avg_util}%  avg mem-util={avg_mem}%", flush=True)
        result["gpu_util_during_stdp_learning"] = {
            "n_steps": util_steps, "wall_s": wall, "n_samples": n_samples,
            "avg_gpu_util_pct": avg_util, "avg_mem_util_pct": avg_mem,
        }

        print(f"\n=== PART 3b: GPU utilization sampling during config B (fused inference, N={n:,}) ===", flush=True)
        b = _build(n, fan_in)
        avg_util_b, avg_mem_b, n_samples_b, wall_b = _gpu_util_during(b, n, util_steps)
        b.clear_simulation_state_and_gpu_memory()
        print(f"  {n_samples_b} nvidia-smi samples over {wall_b:.2f}s ({util_steps} steps): "
              f"avg GPU util={avg_util_b}%  avg mem-util={avg_mem_b}%", flush=True)
        result["gpu_util_during_fused_inference"] = {
            "n_steps": util_steps, "wall_s": wall_b, "n_samples": n_samples_b,
            "avg_gpu_util_pct": avg_util_b, "avg_mem_util_pct": avg_mem_b,
        }

    # ---------------------------------------------------------------------------------------------
    # PART 4: cProfile function-level breakdown of config C (today's default learning path) --
    # corroborating (not authoritative) evidence for which named phase dominates wall time.
    # ---------------------------------------------------------------------------------------------
    if not args.skip_profile:
        print(f"\n=== PART 4: cProfile breakdown of config C (STDP-compacting, N={n:,}) ===", flush=True)
        b = _build(n, fan_in, enable_stdp=True, enable_step_megakernel=False, enable_step_megakernel_v2=False)
        _drive(b, n)
        for _ in range(args.warmup):
            _step(b)
        synchronize()
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(100 if not args.smoke else 10):
            _step(b)
        synchronize()
        pr.disable()
        b.clear_simulation_state_and_gpu_memory()
        buf = io.StringIO()
        ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
        ps.print_stats(25)
        profile_text = buf.getvalue()
        print(profile_text, flush=True)
        result["cprofile_config_C_top25_cumulative"] = profile_text

    _dump(result, args.out)
    print("\n[perf-step-profile] DONE.", flush=True)


def _dump(result, out_path):
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[perf-step-profile] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
