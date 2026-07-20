"""gap#4 KEYSTONE — does the spiking substrate LEARN A BEHAVIOUR IN ONE SHOT via the biological BTSP rule?

WHY THIS RUNNER EXISTS. The BTSP finding of 2026-07-18 established on-bridge behavioral-timescale credit (6-seed GO)
and then named its own next step: *"(b) a one-shot TASK (association/place-field) the substrate LEARNS via BTSP"*.
Item (c) was pursued; **(b) never was** — no such runner existed. Every BTSP result banked so far gates on a WEIGHT
CHANGE ("held dw is 8.4x the transient dw"). This runner converts that into the actual gap#4 capability claim: a
BEHAVIOUR the substrate acquires from ONE experience.

THE TASK (Bittner & Magee behavioral-timescale synaptic plasticity, CA1 place-field formation in ONE lap).
A 20-bin track; each bin drives its own position pool. A single dendritic PLATEAU is delivered at bin `b` during ONE
induction lap. Afterwards, with plasticity OFF, the CA1 cell should fire selectively near bin `b` -- a PLACE FIELD it
did not have before, formed from a single trial, with a seconds-long backward-biased window no millisecond rule can
produce.

⚠️ THE RELEASE PULSE IS MANDATORY (design trap). `bdsp_apical_bistable=True` LATCHES `cp_v_apical` above `v_hold`
indefinitely (self-regen + KIR). Without an explicit release, every bin after `b` stays inside `IS_post > 0` and the
"field" spans the rest of the track -- localization becomes untestable BY CONSTRUCTION and every arm reads the same
number (the degenerate-readout failure that silently defeats controls). We therefore release the plateau after
`plateau_hold_ms`.

⚠️ dw IS NOT THE GATE (C9). `dw` / `v_apical_end` / `n_IS_positive_steps` are reported in a SEPARATE `mechanism` block
explicitly labelled *not evidence of task performance* -- a 2000x dw range has already been shown compatible with a
flat read-out in this project. The GO reads `field_acc` and the speed-scaling ratio ONLY.

PRE-REGISTERED GO (filed before any BTSP result exists):
    GO iff  field_acc_BTSP >= 0.80  (>= 24/30 instances)
       AND  width_fast / width_slow >= 1.5 on >= 5/6 seeds
       AND  every control in the table below passes.
`field_acc` = fraction of instances whose peak bin is within 2 bins of `b`. Chance = 5/20 = 0.25.
0.80 is the project's own task-validity bar (the same number required of a working oracle).

CONTROLS (run order): C12 flag-engagement smoke · C5 two-process substrate hash · MAIN · C7 degenerate-readout guard
· C1 frozen (btsp_learning_rate=0) · C2 mis-targeted plateau (must go BELOW chance) · C2b random plateau bin (AT
chance) · C3 no-plateau moat (dw==0 AND at chance) · C4 enable_btsp=False byte-identity · C10 transient vs held
(duration, rule fixed) · C6 dev/blind seeds reported SEPARATELY · C11 full provenance in the JSON.

NO `sim/` edit -- reuse-by-import of the committed `enable_btsp` block.
Run: SIM_BACKEND=numpy python -m research.runners._gap4_btsp_oneshot_place_field_task_derisk --seeds 42 43 44
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import argparse, hashlib, json, math
import numpy as np

N_BINS = 20
CA1_N = 8
POS_N = 10
INSTANCE_BINS = [6, 9, 12, 15, 18]          # PRE-REGISTERED; bins <6 excluded (the 1000ms backward window runs off-track)
DEV_SEEDS = [42, 43, 44]
BLIND_SEEDS = [100, 101, 102]


def build(seed, *, btsp=True, eta=0.02, bistable=True, elig_tau=1000.0, dt=1.0, w0=0.6, ca1_n=None, wj=0.15,
          hdep=0.3, htheta=0.012):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    cfg = CoreSimConfig(seed=seed)                       # ⛔ constructor kwarg: cfg.seed is what ACTUALLY seeds the substrate
    cfg.dt_ms = float(dt)
    # ⛔ THE DOMINANT CONFOUND (research gate 2026-07-20): cfg.num_traits defaults to 5, so the bridge deals FIVE
    # Izhikevich cell types (rheobase 42-306 pA) into CA1 and the position pools -> 2.0-2.5x drive spread ->
    # ~7-10x RATE spread (f-I gain ~3.36). BTSP's own contrast is 1.5x in weight ~= 3.9x in rate, so argmax was
    # decided by WHICH CELL TYPE THE RNG DEALT, not by the weights. This is gated on num_traits>1 ALONE
    # (bridge.py:1552) -- disabling enable_parameter_heterogeneity does NOT touch it. Seed 100 had ZERO RS
    # neurons and scored 0.00 with a CORRECT weight map; seed 42 had two and formed a field.
    cfg.num_traits = 1
    cfg.enable_brain_region_framework = True
    # every other writer to cp_connections.data OFF, so BTSP is the sole weight-mover
    for f in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
              "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
              "enable_nmda", "enable_ou_process", "enable_parameter_heterogeneity",
              "enable_conductance_noise", "enable_synaptic_scaling"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    # Route B: real apical dynamics (enable_bdsp only to EVOLVE cp_v_apical; its learning rate is 0)
    cfg.enable_bdsp = True
    cfg.bdsp_learning_rate = 0.0
    cfg.bdsp_apical_bistable = bool(bistable)
    cfg.coincidence_plateau_self_regen = 2.0
    cfg.coincidence_plateau_v_hold = -35.0
    cfg.apical_kir_g = 1.0
    # the rule under test
    cfg.enable_btsp = bool(btsp)
    cfg.btsp_learning_rate = float(eta)
    cfg.btsp_elig_tau_ms = float(elig_tau)
    cfg.btsp_w_min, cfg.btsp_w_max = 0.0, 5.0
    cfg.btsp_hetero_dep = float(hdep)      # >0 engages heterosynaptic depression
    cfg.btsp_hetero_theta = float(htheta)  # >0 = THRESHOLDED gate (protects strongly co-active pairs)
    cfg.brain_regions = [
        BrainRegion(name=f"pos{k}", n_neurons=POS_N, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)
        for k in range(N_BINS)
    ] + [BrainRegion(name="ca1", n_neurons=int(ca1_n or CA1_N), exc_fraction=1.0, internal_density=0.0,
                     exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False)]
    cfg.region_pathways = [
        RegionPathway(from_region=f"pos{k}", to_region="ca1", density=1.0,
                      weight_mean=float(w0), weight_jitter=float(wj), plastic=True)
        for k in range(N_BINS)
    ]
    rt = RuntimeState(); rt.actual_seed_used = seed
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=rt, gpu_config=GPUConfig())
    sb._initialize_simulation_data()
    pos = [np.asarray(sb.region_manager.indices(f"pos{k}")) for k in range(N_BINS)]
    ca1 = np.asarray(sb.region_manager.indices("ca1"))
    return sb, pos, ca1


def _w_sum(sb):
    from sim.backend import to_host
    return float(np.abs(np.asarray(to_host(sb.cp_connections.data))).sum())


def run_lap(sb, pos, ca1, *, plateau_bin=None, bin_steps=200, drive_pA=900.0,
            plateau_pA=600.0, release_pA=900.0, plateau_hold_ms=700.0, pulse_steps=15,
            record=False):
    """One traversal of the track. If plateau_bin is not None, deliver the plateau there (and RELEASE it)."""
    from sim.backend import to_host, get_backend
    xp, _ = get_backend()
    n = int(sb.core_config.num_neurons)
    rate = np.zeros(N_BINS); is_pos = 0; step_global = 0
    plateau_start = None
    for k in range(N_BINS):
        for s in range(bin_steps):
            cur = np.zeros(n, np.float32)
            cur[pos[k]] = drive_pA
            sb.cp_external_input_current[:] = 0.0
            sb.cp_external_input_current[:] = (xp.asarray(cur) if xp is not None else cur)
            # ⚠️ cp_bdsp_apical_drive starts as None and is ASSIGNED (not written in-place) — an `is not None`
            # guard here silently skips the plateau entirely (caught by C12: dw=0, v_apical stuck at rest).
            ap = np.zeros(n, np.float32)
            if plateau_bin is not None:
                if k == plateau_bin and s < pulse_steps:
                    ap[ca1] = plateau_pA
                    if plateau_start is None:
                        plateau_start = step_global
                # MANDATORY RELEASE: without it the bistable apical latches and the field spans the track
                if plateau_start is not None and 0 <= step_global - (plateau_start + int(plateau_hold_ms)) < 20:
                    ap[ca1] = -release_pA
            sb.cp_bdsp_apical_drive = (xp.asarray(ap) if xp is not None else ap)
            sb._run_one_simulation_step()
            if record:
                rate[k] += float(np.asarray(to_host(sb.cp_firing_states))[ca1].sum())
            if sb.cp_v_apical is not None:
                va = np.asarray(to_host(sb.cp_v_apical))[ca1]
                is_pos += int((va > sb.core_config.coincidence_plateau_v_hold).sum() > 0)
            step_global += 1
    if record:
        rate /= float(CA1_N * bin_steps)
    v_end = (float(np.asarray(to_host(sb.cp_v_apical))[ca1].mean())
             if sb.cp_v_apical is not None else float("nan"))
    return rate, is_pos, v_end


def field_metrics(rate, b):
    if rate.max() <= 0:
        # NOTE: with DELTA scoring this path is reachable whenever post <= pre everywhere (no field formed),
        # which is a legitimate MISS, not an error.
        return dict(peak_bin=-1, width=0, hit=False, offset=0, dead=True, flat=True, ratio=0.0)
    peak = int(np.argmax(rate)); hm = 0.5 * rate.max()
    bins = {peak}
    for d in (1, -1):
        k = peak
        while True:
            k = (k + d) % N_BINS
            if k in bins or rate[k] < hm:
                break
            bins.add(k)
    off = (peak - b + N_BINS // 2) % N_BINS - N_BINS // 2   # signed circular offset
    # PRE-REGISTERED backward window: BTSP forms the field BEHIND the plateau (Bittner-Magee); the eligibility
    # window is btsp_elig_tau_ms=1000ms = 5 bins at 200ms/bin, plus one bin of forward allowance.
    return dict(peak_bin=peak, width=len(bins), hit=bool(-5 <= off <= 1), offset=int(off), dead=False,
                flat=bool(rate.max() / max(rate.mean(), 1e-9) <= 1.5),
                ratio=float(rate.max() / max(rate.mean(), 1e-9)))


def one_instance(seed, b, *, btsp=True, eta=0.02, bistable=True, plateau_bin=None,
                 do_plateau=True, bin_steps=200, wj=0.15, hdep=0.3, htheta=0.012, elig_tau=1000.0):
    """Baseline -> ONE induction lap -> probe. Returns metrics + the mechanism block."""
    sb, pos, ca1 = build(seed, btsp=btsp, eta=eta, bistable=bistable, wj=wj,
                         hdep=hdep, htheta=htheta, elig_tau=elig_tau)
    pb = (b if plateau_bin is None else plateau_bin) if do_plateau else None
    pre, _, _ = run_lap(sb, pos, ca1, plateau_bin=None, bin_steps=bin_steps, record=True)
    w0 = _w_sum(sb)
    _, is_pos, v_end = run_lap(sb, pos, ca1, plateau_bin=pb, bin_steps=bin_steps, record=False)
    dw = _w_sum(sb) - w0
    sb.core_config.enable_btsp = False                    # probe with plasticity OFF
    post, _, _ = run_lap(sb, pos, ca1, plateau_bin=None, bin_steps=bin_steps, record=True)
    delta = post - pre                                    # PRE-REGISTERED: score the CHANGE, not the absolute map
    m = field_metrics(delta, b)
    m.update(dw=float(dw), n_IS=int(is_pos), v_apical_end=float(v_end),
             pre_ratio=float(pre.max() / max(pre.mean(), 1e-9)),
             pre_flatness=float(pre.max() / max(pre.mean(), 1e-9)))
    return m


def arm(seed, *, label, **kw):
    hits = []; widths = []; mech = []
    for b in INSTANCE_BINS:
        m = one_instance(seed, b, **kw)
        hits.append(m["hit"]); widths.append(m["width"])
        mech.append({k: m[k] for k in ("dw", "n_IS", "v_apical_end", "peak_bin", "width", "ratio", "dead", "flat", "offset")})
    return dict(label=label, seed=seed, field_acc=float(np.mean(hits)),
                mean_width=float(np.mean(widths)), mechanism=mech)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEV_SEEDS + BLIND_SEEDS)
    ap.add_argument("--bin-steps", type=int, default=200)
    ap.add_argument("--verify-flags", action="store_true", help="C12 flag-engagement smoke only")
    ap.add_argument("--quick", action="store_true", help="fewer bins/steps for a smoke")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    bs = 60 if args.quick else args.bin_steps

    from sim.backend import to_host
    # ---- C5: two-process-equivalent substrate hash (same seed must give identical neurons) ----
    h = []
    for _ in range(2):
        sb, _, _ = build(42)
        h.append(hashlib.md5(np.asarray(to_host(sb.cp_neuron_firing_thresholds)).tobytes()).hexdigest()[:12])
        del sb
    c5 = dict(hash_a=h[0], hash_b=h[1], pass_=bool(h[0] == h[1]))
    print(f"[C5] substrate hash {h[0]} vs {h[1]} -> {'SEEDED' if c5['pass_'] else 'UNSEEDED (FAIL)'}", flush=True)

    # ---- C12: every flag must change a measurable, else it is inert ----
    if args.verify_flags or True:
        m_on = one_instance(42, 12, btsp=True, bistable=True, bin_steps=bs)
        m_off = one_instance(42, 12, btsp=False, bistable=True, bin_steps=bs)
        m_tr = one_instance(42, 12, btsp=True, bistable=False, bin_steps=bs)
        c12 = dict(dw_btsp_on=m_on["dw"], dw_btsp_off=m_off["dw"],
                   v_end_bistable=m_on["v_apical_end"], v_end_transient=m_tr["v_apical_end"],
                   btsp_engaged=bool(abs(m_on["dw"]) > 1e-9 and abs(m_off["dw"]) < 1e-9),
                   bistable_engaged=bool(abs(m_on["v_apical_end"] - m_tr["v_apical_end"]) > 1.0))
        print(f"[C12] enable_btsp: dw on={m_on['dw']:.4g} off={m_off['dw']:.4g} -> "
              f"{'ENGAGED' if c12['btsp_engaged'] else '⛔ INERT'} | "
              f"bistable: v_apical held={m_on['v_apical_end']:.2f} transient={m_tr['v_apical_end']:.2f} -> "
              f"{'ENGAGED' if c12['bistable_engaged'] else '⛔ INERT'}", flush=True)
        if args.verify_flags:
            print(json.dumps(dict(C5=c5, C12=c12), indent=2)); return

    arms = {}
    for s in args.seeds:
        rng = np.random.default_rng(s + 7717)
        arms.setdefault("MAIN", []).append(arm(s, label="MAIN", bin_steps=bs))
        arms.setdefault("C1_frozen", []).append(arm(s, label="C1_frozen", eta=0.0, bin_steps=bs))
        arms.setdefault("C2_mistarget", []).append(
            arm(s, label="C2_mistarget", plateau_bin=(INSTANCE_BINS[0] + 10) % N_BINS, bin_steps=bs))
        arms.setdefault("C2b_random", []).append(
            arm(s, label="C2b_random", plateau_bin=int(rng.integers(0, N_BINS)), bin_steps=bs))
        arms.setdefault("C3_moat", []).append(arm(s, label="C3_moat", do_plateau=False, bin_steps=bs))
        arms.setdefault("C10_transient", []).append(arm(s, label="C10_transient", bistable=False, bin_steps=bs))
        for k, v in arms.items():
            if v[-1]["seed"] == s:
                print(f"  seed {s} {k:14s} field_acc={v[-1]['field_acc']:.2f} width={v[-1]['mean_width']:.1f} "
                      f"dw={np.mean([m['dw'] for m in v[-1]['mechanism']]):.4g}", flush=True)

    def agg(name, seeds):
        rs = [a for a in arms.get(name, []) if a["seed"] in seeds]
        return dict(n=len(rs), field_acc=float(np.mean([r["field_acc"] for r in rs])) if rs else float("nan"))

    dev = [s for s in args.seeds if s in DEV_SEEDS]; blind = [s for s in args.seeds if s in BLIND_SEEDS]
    summary = {k: dict(dev=agg(k, dev), blind=agg(k, blind), all=agg(k, args.seeds)) for k in arms}
    print("\n=== SUMMARY (C6: dev and blind reported SEPARATELY) ===", flush=True)
    for k, v in summary.items():
        print(f"  {k:14s} dev={v['dev']['field_acc']:.3f} blind={v['blind']['field_acc']:.3f} "
              f"all={v['all']['field_acc']:.3f}", flush=True)
    main_all = summary.get("MAIN", {}).get("all", {}).get("field_acc", float("nan"))
    main_blind = summary.get("MAIN", {}).get("blind", {}).get("field_acc", float("nan"))
    go = bool(main_all >= 0.80 and (math.isnan(main_blind) or main_blind >= 0.80))
    print(f"\nVERDICT: {'GO' if go else 'NO-GO'} (pre-registered: field_acc>=0.80 AND blind passes on its own; "
          f"all={main_all:.3f} blind={main_blind:.3f}) [C9: dw is NOT the gate]", flush=True)
    if args.json:
        json.dump(dict(summary=summary, arms=arms, C5=c5, C12=c12, verdict="GO" if go else "NO-GO",
                       config=dict(bin_steps=bs, instance_bins=INSTANCE_BINS, n_bins=N_BINS,
                                   ca1_n=CA1_N, pos_n=POS_N, eta=0.02, elig_tau=1000.0,
                                   plateau_hold_ms=700.0, threshold_md5=h[0],
                                   backend=os.environ.get("SIM_BACKEND"))),
                  open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
