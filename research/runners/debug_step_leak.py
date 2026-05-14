"""Minimal repro to find the bridge step leak.

Load a bridge, run N steps in a tight loop (no engram, no encoding),
measure RAM growth periodically.
"""
import argparse, gc, os, psutil, time
import research.runners.concept_pool_demo as cpd


def get_rss_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-bridge", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--check-every", type=int, default=500)
    ap.add_argument("--free-pool-every", type=int, default=0,
                     help="Call cp memory_pool.free_all_blocks() every N steps (0=off)")
    ap.add_argument("--gc-every", type=int, default=0,
                     help="Call gc.collect() every N steps (0=off)")
    ap.add_argument("--engram-cycle", type=int, default=0,
                     help="Encode an engram every N steps (0=off; tests if "
                     "the leak is in engram tagging)")
    ap.add_argument("--pattern-accum", action="store_true",
                     help="Also accumulate pattern_accum like compose_engram_retrieval")
    ap.add_argument("--to-host-every", type=int, default=0,
                     help="Call to_host(pattern_accum) every N steps (0=off)")
    ap.add_argument("--drive-lang", action="store_true",
                     help="Drive lang_input per step (like compose_engram_retrieval)")
    ap.add_argument("--use-retrieval-funcs", action="store_true",
                     help="Call actual encode_with_pattern in a loop (test if THAT leaks)")
    args = ap.parse_args()

    print(f"Loading bridge (seed={args.seed})...", flush=True)
    bridge = cpd.build_concept_bridge(
        seed=args.seed, n_lang_input=2048, n_per_pool=200, n_fs_per_pool=24,
        enable_adjective=True, weak_dynamics=True,
        enable_direct_verb_to_motor=True, verbose=False,
    )
    bridge.load_checkpoint(args.load_bridge)
    print(f"Bridge loaded. RAM after load: {get_rss_mb():.1f} MB", flush=True)

    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
    except Exception:
        pool = None

    # If --use-retrieval-funcs: simulate full retrieval flow:
    # 1) encode 48 engrams, store patterns
    # 2) measure 48 firing patterns during drive (the retrieval phase)
    if args.use_retrieval_funcs:
        import numpy as np
        from research.runners.compose_engram_retrieval import (
            encode_with_pattern, measure_firing_pattern_during_drive,
        )
        n_total = bridge.cp_external_input_current.shape[0]
        region_filter = (
            [f"verb_pool_{v}" for v in ["GO", "COME", "STOP", "LOOK"]]
            + [f"noun_pool_{n}" for n in ["APPLE", "RIVER", "DOG", "CAT"]]
            + [f"adjective_pool_{a}" for a in ["BIG", "SMALL", "HOT", "COLD"]]
            + [f"motor_{a}" for a in ["N", "E", "S", "W"]]
        )
        rf_mask = np.zeros(n_total, dtype=bool)
        for rname in region_filter:
            try:
                rf_mask[list(bridge.region_manager.indices(rname))] = True
            except Exception:
                pass

        N = 48
        encoded = {}
        t0 = time.time()
        base_rss = get_rss_mb()
        print(f"Encoding {N} engrams + storing patterns...", flush=True)
        for i in range(N):
            verb = ["go", "come", "stop", "look"][i % 4]
            motor = ["north", "east", "south", "west"][(i // 4) % 4]
            tag = f"enc_{i}"
            _, pattern = encode_with_pattern(
                bridge, verb, motor, tag,
                encoding_steps=200, sparsity=0.05,
                n_lang_input=2048, region_filter=region_filter,
                top_k=100, verbose=False,
            )
            encoded[tag] = pattern  # store pattern (like compose_engram_retrieval)
            if (i + 1) % 8 == 0:
                rss = get_rss_mb()
                print(f"  encode {i+1}/{N} RAM={rss:.1f} MB (+{rss-base_rss:.1f}) "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"\nNow running {N} retrieval queries...", flush=True)
        for i in range(N):
            verb = ["go", "come", "stop", "look"][i % 4]
            motor = ["north", "east", "south", "west"][(i // 4) % 4]
            query_pattern = measure_firing_pattern_during_drive(
                bridge, verb, motor, rf_mask,
                drive_steps=200, sparsity=0.05, n_lang_input=2048,
            )
            if (i + 1) % 8 == 0:
                rss = get_rss_mb()
                print(f"  query  {i+1}/{N} RAM={rss:.1f} MB (+{rss-base_rss:.1f}) "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
        print(f"Done. Total RAM growth: {get_rss_mb()-base_rss:.1f} MB")
        return

    print(f"Running {args.n_steps} bridge steps "
          f"(engram_cycle={args.engram_cycle}, pattern_accum={args.pattern_accum})...",
          flush=True)
    t0 = time.time()
    base_rss = get_rss_mb()
    engram_count = 0
    n_total = bridge.cp_external_input_current.shape[0]
    pattern_accum = None
    saved_patterns = []
    if args.pattern_accum:
        import cupy as cp_local
        pattern_accum = cp_local.zeros(n_total, dtype=cp_local.float32)
    # Drive setup for --drive-lang
    if args.drive_lang:
        import cupy as cp_local
        from sim.text_embeddings import orthogonal_drive_pattern
        verb_drive = orthogonal_drive_pattern(cue_idx=8, n_cues=16,
                                                 n_neurons=2048,
                                                 drive_max_pA=200.0, sparsity=0.05)
        motor_drive = orthogonal_drive_pattern(cue_idx=0, n_cues=16,
                                                  n_neurons=2048,
                                                  drive_max_pA=200.0, sparsity=0.05)
        both_gpu = cp_local.asarray(verb_drive + motor_drive, dtype=cp_local.float32)
        lang_arr_gpu = cp_local.asarray(
            list(bridge.region_manager.indices("language_input")),
            dtype=cp_local.int64)
        ext = cp_local.zeros(n_total, dtype=cp_local.float32)
    for i in range(args.n_steps):
        if args.drive_lang:
            ext.fill(0)
            ext[lang_arr_gpu] = both_gpu
            bridge.cp_external_input_current[:] = ext
        # Optional: engram-record/commit cycle every N steps
        if args.engram_cycle > 0 and i % args.engram_cycle == 0:
            tag_name = f"debug_tag_{engram_count}"
            bridge.start_engram_recording(tag_name)
        bridge._run_one_simulation_step()
        if args.pattern_accum:
            pattern_accum += bridge.cp_firing_states
        if args.engram_cycle > 0 and (i + 1) % args.engram_cycle == 0:
            bridge.commit_engram_tag(tag_name, top_k=100)
            engram_count += 1
            if args.pattern_accum:
                # Save pattern to host (like compose_engram_retrieval does)
                import cupy as cp_local
                saved_patterns.append(cp_local.asnumpy(pattern_accum))
                pattern_accum.fill(0)
        if args.free_pool_every > 0 and (i + 1) % args.free_pool_every == 0 and pool:
            pool.free_all_blocks()
        if args.gc_every > 0 and (i + 1) % args.gc_every == 0:
            gc.collect()
        if (i + 1) % args.check_every == 0:
            elapsed = time.time() - t0
            rss = get_rss_mb()
            pool_used = (pool.used_bytes() / 1024 / 1024) if pool else 0
            print(f"  step {i+1}/{args.n_steps} RAM={rss:.1f} MB (+{rss-base_rss:.1f}) "
                  f"pool_used={pool_used:.1f} MB engrams={engram_count} "
                  f"elapsed={elapsed:.1f}s", flush=True)
    print(f"Done. Total RAM growth: {get_rss_mb()-base_rss:.1f} MB",
          flush=True)


if __name__ == "__main__":
    main()
