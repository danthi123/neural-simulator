"""Cheap-first de-risk: a BLOCK-DIAGONAL batched on-bridge spiking forward.

WHY: the deep-credit-on-spikes de-risks (`_semantic_inheritance_onbridge_spiking_derisk`) run the spiking forward
ONE example at a time (`_forward_spiking` -> settle_steps x `_run_one_simulation_step`), so a single trained-scale
config is ~1-2 hr (on numpy the per-call Python overhead of the step pipeline dominates a ~200-neuron bridge stepped
~768k times). That per-example wall-clock is the demonstrated binding limit on iterating the credit training. The fix
is to evaluate M examples AT ONCE as M disjoint block-diagonal copies of the net on ONE bridge: M examples advance in
ONE `_run_one_simulation_step` call (the overhead is amortized M x). The RF composer already proved disjoint neuron
slices on one bridge don't cross-talk (byte-isolated), so M copies with NO cross-copy synapses advance independently.

THIS PROVES THE CORE CLAIM cheap-first (forward/eval only, fixed weights -> no weight-tying needed):
  (1) CORRECTNESS: the batched read cp_bdsp_E for copy m == the serial per-example read for example m (to tolerance).
  (2) SPEEDUP: the batched forward of M examples is meaningfully faster than M serial forwards.
If GO -> integrate block-diagonal batched EVAL into the runner's `_forward_batch` (accuracy on train/held-out each
epoch is fixed-weight), then extend to minibatch training (weight-tying: average the per-copy weight moves).

Run (numpy, tiny): E:/.../python.exe -m research.runners._batched_onbridge_forward_derisk [--M 8] [--seed 42]
NO `sim/` edit -- pure reuse of SimulationBridge + inject_explicit_wiring + cp_bdsp_E.
"""
import argparse, time
import numpy as np


def _sizes(n_in, hidden, k):
    return [n_in, hidden, hidden, k]


def build_net(n_in, hidden, k, W, seed, settle_steps, n_copies=1):
    """Build a bridge holding `n_copies` DISJOINT block-diagonal copies of a depth-2 [in|H1|H2|out] net, each wired
    with the SAME feedforward weights `W` (a list of dense layer matrices). Returns (bridge, per_copy_slices)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
    sizes = _sizes(n_in, hidden, k)
    per = int(np.sum(sizes))                       # neurons per copy
    n_total = per * n_copies
    cfg = CoreSimConfig()
    cfg.num_neurons = n_total
    cfg.dt_ms = 1.0
    cfg.enable_bdsp = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_ou_process = False   # deterministic forward: batched==serial EXACTLY iff the block-diagonal mechanism
                                    # has no cross-copy coupling (OU-on differs only by the per-copy noise realization)
    cfg.enable_homeostasis = False  # rule out threshold-homeostasis (firing-rate EMA) as the cross-copy coupler
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_inhibitory_neurons = False
    cfg.num_traits = 1   # <-- THE ROOT: cp_traits = cp.random.randint(0,num_traits) (bridge-internal RNG, NOT seeded by
                         # actual_seed_used) assigns neuron TYPE -> vr -> v_init. num_traits=1 => all trait 0 => identical
                         # deterministic vr => reproducible init => the block-diagonal batched forward is EXACT.
    cfg.seed = int(seed)                      # cfg.seed is what ACTUALLY seeds the substrate
    cfg.actual_seed_used = int(seed)
    br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                          viz_config=VisualizationConfig(), runtime_state=RuntimeState())
    br._initialize_simulation_data()

    # per-copy layer slices (absolute neuron index ranges)
    copy_slices = []
    plan = {}
    for c in range(n_copies):
        base = c * per
        starts = base + np.cumsum([0] + sizes)
        slices = [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(sizes))]
        copy_slices.append(slices)
        for li in range(len(sizes) - 1):
            pre = np.arange(slices[li].start, slices[li].stop)
            post = np.arange(slices[li + 1].start, slices[li + 1].stop)
            Wl = W[li]
            P, Q, Wv = [], [], []
            for ai, a in enumerate(pre):
                for bi, b in enumerate(post):
                    P.append(int(a)); Q.append(int(b)); Wv.append(float(Wl[ai, bi]))
            plan[f"c{c}_ff{li}"] = dict(pre_indices=P, post_indices=Q, initial_weights=Wv, plastic=True, conn_type="ff")
    br.inject_explicit_wiring(plan)
    # CLONE the 1-copy REFERENCE net's per-neuron state into every copy, so each copy == the canonical single-copy net
    # (the bridge randomizes vr/v_init/params over ALL n_total, so the M-copy's own init differs from a lone 1-copy net
    # -> the divergence). Cloning the REFERENCE (not the M-copy's own block0) is what aligns batched with serial: after
    # this + OU off, the batched forward == the serial per-example forward EXACTLY.
    if n_copies > 1:
        from sim.backend import to_host, get_backend
        xp, _ = get_backend()
        ref, _, _ = build_net(n_in, hidden, k, W, seed, settle_steps, n_copies=1)   # the canonical 1-copy net
        for a in dir(br):
            if not a.startswith("cp_"):
                continue
            arr = getattr(br, a, None); refarr = getattr(ref, a, None)
            try:
                if (arr is not None and refarr is not None and hasattr(arr, "shape") and arr.ndim == 1
                        and int(arr.shape[0]) == n_total and int(np.asarray(to_host(refarr)).shape[0]) == per):
                    host = np.asarray(to_host(arr)).copy()
                    refblock = np.asarray(to_host(refarr)).copy()
                    for c in range(n_copies):
                        host[c * per:(c + 1) * per] = refblock
                    setattr(br, a, xp.asarray(host).astype(arr.dtype))
            except Exception:
                pass
    return br, copy_slices, sizes


def _drive_and_settle(br, copy_slices, sizes, feats, settle_steps, tonic_h=450.0, tonic_o=500.0,
                      in_cur=520.0, in_bias=260.0, solo=False):
    """Drive each copy c with feats[c] on its input slice (+ tonic on hidden/out), settle, return per-copy layer E.
    solo=True: drive ONLY copy 0 (all other copies get ZERO drive -> silent) -> presence-vs-activity coupling test."""
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    n_total = int(br.cp_membrane_potential_v.shape[0])
    drive = np.zeros(n_total, dtype=np.float32)
    # reset BDSP rate state so E reflects THIS drive
    if br.cp_bdsp_E is not None:
        br.cp_bdsp_E[...] = 0.0; br.cp_bdsp_B[...] = 0.0
        br.cp_bdsp_last_spike_step = xp.full(n_total, -1000000, dtype=xp.int64)
    for c, slices in enumerate(copy_slices):
        if solo and c != 0:
            continue                                    # leave this copy fully silent (zero drive)
        for li in range(1, len(sizes) - 1):
            drive[slices[li]] = tonic_h
        drive[slices[-1]] = tonic_o
        f = np.asarray(feats[c], dtype=np.float32)
        drive[slices[0]] = np.clip(in_bias + in_cur * f, 0.0, 1600.0).astype(np.float32)
    br.cp_external_input_current = xp.asarray(drive)
    if br.cp_bdsp_apical_drive is not None:
        br.cp_bdsp_apical_drive[...] = 0.0
    for _ in range(settle_steps):
        br._run_one_simulation_step()
    E = np.asarray(to_host(br.cp_bdsp_E)).copy()
    return [[E[slices[li]] for li in range(len(sizes))] for slices in copy_slices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8, help="batch size (copies)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-in", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--settle-steps", type=int, default=40)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sizes = _sizes(args.n_in, args.hidden, args.k)
    # shared FF weights (same across all copies) — Xavier
    W = []
    for li in range(len(sizes) - 1):
        lim = np.sqrt(6.0 / (sizes[li] + sizes[li + 1]))
        W.append(rng.uniform(-lim, lim, (sizes[li], sizes[li + 1])) * 4.0)
    feats = [rng.standard_normal(args.n_in).astype(np.float32) for _ in range(args.M)]

    # ---- SERIAL reference: M single-copy bridges, one example each ----
    t0 = time.time()
    serial_reads = []
    for m in range(args.M):
        br1, cs1, sz1 = build_net(args.n_in, args.hidden, args.k, W, args.seed, args.settle_steps, n_copies=1)
        r = _drive_and_settle(br1, cs1, sz1, [feats[m]], args.settle_steps)
        serial_reads.append(r[0])
    t_serial = time.time() - t0

    # ---- BATCHED: ONE M-copy block-diagonal bridge, M examples at once ----
    t1 = time.time()
    brM, csM, szM = build_net(args.n_in, args.hidden, args.k, W, args.seed, args.settle_steps, n_copies=args.M)
    batched_reads = _drive_and_settle(brM, csM, szM, feats, args.settle_steps)
    t_batched = time.time() - t1

    # ---- CORRECTNESS: batched copy m == serial example m, per layer ----
    max_abs = 0.0
    per_copy = []
    for m in range(args.M):
        cm = 0.0
        for li in range(len(sizes)):
            d = float(np.max(np.abs(np.asarray(batched_reads[m][li]) - np.asarray(serial_reads[m][li]))))
            cm = max(cm, d)
        per_copy.append(round(cm, 4))
        max_abs = max(max_abs, cm)
    ok = max_abs < 1e-6
    print(f"  per-copy max|batched-serial| = {per_copy}  (copy 0 == serial[0] iff init is the coupler)")

    # SPEEDUP note: serial builds M bridges (build cost too); the fair forward-only compare is the settle loop, but the
    # end-to-end wall-clock is what matters for the runner. Report both the end-to-end and a per-example amortization.
    print(f"=== batched on-bridge forward de-risk (M={args.M}, sizes={sizes}, settle={args.settle_steps}) ===")
    print(f"  CORRECTNESS max|batched - serial| = {max_abs:.2e}  -> {'MATCH (GO)' if ok else 'MISMATCH (NO-GO)'}")
    print(f"  serial  (M single-copy forwards): {t_serial:.2f}s  ({t_serial/args.M*1000:.0f} ms/example)")
    print(f"  batched (1 M-copy forward)       : {t_batched:.2f}s  ({t_batched/args.M*1000:.0f} ms/example)")
    sp = t_serial / max(t_batched, 1e-9)
    print(f"  end-to-end speedup: {sp:.1f}x")

    # --- PRESENCE vs ACTIVITY diagnostic: copy 0 in an M-copy bridge with ALL OTHERS SILENT vs a lone 1-copy bridge ---
    br1, cs1, sz1 = build_net(args.n_in, args.hidden, args.k, W, args.seed, args.settle_steps, n_copies=1)
    solo_ref = _drive_and_settle(br1, cs1, sz1, [feats[0]], args.settle_steps)[0]
    brS, csS, szS = build_net(args.n_in, args.hidden, args.k, W, args.seed, args.settle_steps, n_copies=args.M)
    solo_read = _drive_and_settle(brS, csS, szS, feats, args.settle_steps, solo=True)[0]
    solo_d = max(float(np.max(np.abs(np.asarray(solo_read[li]) - np.asarray(solo_ref[li])))) for li in range(len(sizes)))
    print(f"  PRESENCE test: copy0 (others SILENT) vs lone-1copy max|Δ| = {solo_d:.4f}  "
          f"-> {'PRESENCE-coupled (a count/global op over all neurons)' if solo_d > 1e-6 else 'ACTIVITY-coupled (silent others do not perturb)'}")
    print(f"  VERDICT: {'GO — block-diagonal batched forward is correct AND faster' if (ok and sp > 1.5) else ('CORRECT but not faster at this M/size' if ok else 'NO-GO (mismatch)')}")


if __name__ == "__main__":
    main()
