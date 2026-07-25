"""gap#5 imaginative-replay READOUT — 6-SEED GO (2026-07-25). A from-scratch Ecker-2022-style CA3 model: a GAUSSIAN
near-diagonal recurrent band (W[i,j]=w_scale·exp(-((i-j)/sigma)^2)) over a 2000-neuron place-field track of AdEx neurons
(ECKER_CA3_PC preset) + a PVBC inhibitory pool. A brief cue ignites a LOCALIZED activity packet that SELF-SUSTAINS and
TRAVELS along the track, and decodes (Davidson-2009 Bayesian population decode, weighted-corr) as a clean DIRECTIONAL
trajectory (DECODE_r=1.000, 6/6 seeds). MECHANISM (verify-go, lesion-attributed): the travel is driven by the recurrent
BAND (no-band -> DECODE_r 0.000) + AdEx spike-reset REFRACTORINESS -- NOT the neg-a/large-b adaptation, which is INERT in
this sparse single-fire regime (lesioning a AND b leaves the travel identical). Directionality is artifact-free ONLY with
a FORWARD-BIASED band (back_frac<1): a symmetric band + interior cue spreads both ways (width grows, does not decode);
the forward-biased band = the learned asymmetric place-field connectivity that makes real hippocampal replay directional.
Modes: default = verify-go mechanism-attribution controls; `directional` = the edge-artifact check; `sixseed` = the
6-seed GO (REAL fwd-band+interior-cue vs NO-BAND + SYMMETRIC-middle controls). dt=0.1 (dt=0.5 blows up the stiff AdEx).
NO sim/ edit beyond the committed additive ECKER_CA3_PC preset. GPU (SIM_BACKEND=cupy). Finding:
2026-07-25-gap5-ecker-nS-recurrent-model-SCAFFOLD-built-dt-fixed-recurrent-transmission-blocker.md."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
sys.path.insert(0, "/home/dant123/Projects/sim")
import numpy as np
from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
from sim.bridge import SimulationBridge
from sim.regions import BrainRegion
from sim.enums import NeuronModel, NeuronType
from sim.backend import to_host, get_backend

N_PC = 2000
N_PVBC = 150
SEED = 42


def build(w_scale, sigma, pc_w_pvbc, pvbc_w_pc, seed=SEED, b_override=None, a_override=None, back_frac=1.0):
    cp, _ = get_backend()
    regions = [
        BrainRegion(name="pc", n_neurons=N_PC, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="pvbc", n_neurons=N_PVBC, exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0,
                    inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    cfg = CoreSimConfig()
    cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed); cfg.dt_ms = 0.1; cfg.num_traits = 1  # Ecker dt (0.5 too coarse for DeltaT=4.23 -> V blows up)
    cfg.neuron_model_type = NeuronModel.ADEX.name
    cfg.default_neuron_type_adex = NeuronType.ADEX_ECKER_CA3_PC.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True; cfg.brain_regions = list(regions); cfg.region_pathways = []
    for f in ("enable_homeostasis", "enable_stdp", "enable_hebbian_learning", "enable_structural_plasticity",
              "enable_parameter_heterogeneity"):
        setattr(cfg, f, False)
    cfg.enable_ou_process = True; cfg.ou_noise_sigma_pa = 40.0   # small background noise (Ecker uses noisy drive)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    if b_override is not None:   # override spike-triggered adaptation post-init (kernel reads core_config.adex_b each step)
        b.core_config.adex_b = float(b_override)
    if a_override is not None:   # override subthreshold adaptation (neg-a = the Ecker traveling-bump crux)
        b.core_config.adex_a = float(a_override)
    pc = np.asarray(b.region_manager.indices("pc"), int)
    pv = np.asarray(b.region_manager.indices("pvbc"), int)
    # --- Gaussian near-diagonal recurrent band PC->PC (vectorized) ---
    cutoff = int(3 * sigma)
    offs = np.arange(-cutoff, cutoff + 1)
    offs = offs[offs != 0]
    wof = w_scale * np.exp(-(offs / sigma) ** 2)
    keep = wof > 0.02
    offs, wof = offs[keep], wof[keep]
    pre, post, w = [], [], []
    ii = np.arange(N_PC)
    for o, wv in zip(offs, wof):
        wv_dir = wv if o > 0 else wv * back_frac   # o>0 = forward (post ahead); back_frac<1 -> directional band
        if wv_dir <= 0:
            continue
        j = ii + o
        m = (j >= 0) & (j < N_PC)
        pre.append(pc[ii[m]]); post.append(pc[j[m]]); w.append(np.full(m.sum(), wv_dir, np.float64))
    if pre:   # w_scale=0 (NO-BAND control) -> empty band; keep only the PC<->PVBC edges
        pre = np.concatenate(pre); post = np.concatenate(post); w = np.concatenate(w)
    else:
        pre = np.zeros(0, int); post = np.zeros(0, int); w = np.zeros(0, float)
    # --- PC<->PVBC feedback (local): each PVBC covers a track segment ---
    pcen = (np.arange(N_PVBC) * N_PC / N_PVBC).astype(int)
    pre2, post2, w2 = [], [], []
    for k, c in enumerate(pcen):
        lo, hi = max(0, c - 120), min(N_PC, c + 120)
        seg = np.arange(lo, hi)
        pre2.append(pc[seg]); post2.append(np.full(len(seg), pv[k])); w2.append(np.full(len(seg), pc_w_pvbc))   # PC->PVBC
        lo2, hi2 = max(0, c - 180), min(N_PC, c + 180)
        seg2 = np.arange(lo2, hi2)
        pre2.append(np.full(len(seg2), pv[k])); post2.append(pc[seg2]); w2.append(np.full(len(seg2), pvbc_w_pc))  # PVBC->PC (inhib via inh presyn)
    pre = np.concatenate([pre] + pre2); post = np.concatenate([post] + post2); w = np.concatenate([w] + w2)
    b.inject_explicit_wiring({"rec": {"pre_indices": pre.astype(int).tolist(), "post_indices": post.astype(int).tolist(),
                                      "initial_weights": w.astype(float).tolist(), "plastic": False, "conn_type": "ff"}})
    return b, pc, pv, cp


def _wcorr(x, y, w):
    w = w / w.sum()
    mx = (w * x).sum(); my = (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    vx = (w * (x - mx) ** 2).sum(); vy = (w * (y - my) ** 2).sum()
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0


def decode_and_width(F, n_pos=100, tau_bin=25, place_w=1.5):
    """Bayesian population decode (Davidson 2009) + bump-WIDTH — the discriminator between a LOCALIZED traveling
    packet (Ecker replay: constant small width, high weighted-corr) and a SPREADING front (growing width)."""
    T, N = F.shape
    neuron_pos = np.arange(N) / N * n_pos
    centers = np.arange(n_pos) + 0.5
    fmat = np.exp(-((neuron_pos[:, None] - centers[None, :]) / place_w) ** 2)   # (N, n_pos) place fields
    logf = np.log(fmat + 1e-9); fsum = fmat.sum(0)
    dec, peaks, widths, tbins = [], [], [], []
    for i, t0 in enumerate(range(0, T, tau_bin)):
        counts = F[t0:t0 + tau_bin].sum(0).astype(float)
        if counts.sum() < 3:
            continue
        logp = counts @ logf - fsum
        logp -= logp.max(); p = np.exp(logp); p /= p.sum()
        dec.append(float((p * centers).sum())); peaks.append(float(p.max())); tbins.append(i)
        act = np.where(counts > 0)[0]
        widths.append(float(np.std(neuron_pos[act])) if len(act) > 1 else 0.0)
    if len(dec) < 4:
        return 0.0, 0.0, 0.0, 0.0
    dec = np.array(dec); peaks = np.array(peaks); widths = np.array(widths); tbins = np.array(tbins, float)
    r = _wcorr(tbins, dec, peaks)
    h = len(widths) // 2
    width_growth = float(np.mean(widths[h:]) - np.mean(widths[:h]))   # >0 = spreading front; ~0 = localized
    return r, float(np.mean(widths)), width_growth, float(dec.max() - dec.min())


def run_one(w_scale=3.0, sigma=25.0, pc_w_pvbc=2.0, pvbc_w_pc=3.0, cue_pa=6000.0, cue_steps=20, T=400, cue_width=80,
            b_override=None, a_override=None, seed=SEED, tag="", back_frac=1.0, cue_start=0):
    t0 = time.time()
    b, pc, pv, cp = build(w_scale, sigma, pc_w_pvbc, pvbc_w_pc, seed=seed, b_override=b_override, a_override=a_override,
                          back_frac=back_frac)
    cue = cp.asarray(pc[cue_start:cue_start + cue_width], dtype=cp.int64)   # ignite at cue_start (0=edge, N/2=middle)
    F = np.zeros((T, N_PC), dtype=bool)
    pv_fire = 0
    for t in range(T):
        b.cp_external_input_current[:] = 0.0
        if t < cue_steps:
            b.cp_external_input_current[cue] += float(cue_pa)
        b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states))
        F[t] = fs[pc].astype(bool)
        pv_fire += int(fs[pv].sum())
    # center-of-mass position per time bin (over active PC)
    pos = np.arange(N_PC)
    com = []
    for t in range(0, T, 5):
        fr = F[t:t + 5].sum(0)
        s = fr.sum()
        com.append(float((fr * pos).sum() / s) if s > 3 else np.nan)
    com = np.array(com)
    valid = ~np.isnan(com)
    travel = (np.nanmax(com) - np.nanmin(com)) if valid.sum() > 2 else 0.0
    monotonic = float(np.mean(np.diff(com[valid]) > 0)) if valid.sum() > 2 else 0.0
    # DECODE proxy: correlation of the decoded position (COM) vs time -> a traveling replay is a high |r| monotonic sweep
    bins_t = np.arange(len(com))[valid]
    travel_r = float(np.corrcoef(bins_t, com[valid])[0, 1]) if valid.sum() > 3 else 0.0
    dec_r, mean_w, w_growth, dec_range = decode_and_width(F)
    # time-SHUFFLE null: permute the per-step population vectors across time -> destroys the trajectory, keeps rate/place
    rng = np.random.default_rng(seed * 7 + 1)
    Fsh = F[rng.permutation(F.shape[0])]
    sh_r, _, _, _ = decode_and_width(Fsh)
    print(f"  [{tag}] w={w_scale} pv_pc={pvbc_w_pc} b={b_override} a={a_override}: F_active={F.mean():.4f} "
          f"range={travel:.0f}/{N_PC} COMr={travel_r:+.3f} | DECODE_r={dec_r:+.3f} shuffle_r={sh_r:+.3f} "
          f"bump_width={mean_w:.1f} width_growth={w_growth:+.1f} dec_range={dec_range:.0f}/100 PVBC={pv_fire} ({time.time()-t0:.0f}s)", flush=True)
    return {"tag": tag, "seed": seed, "com_r": travel_r, "dec_r": dec_r, "shuffle_r": sh_r,
            "bump_width": mean_w, "width_growth": w_growth, "dec_range": dec_range, "F_active": float(F.mean()), "pvbc": pv_fire}


# the confirmed real config (w=600 Gaussian band + ECKER neg-a + spike-adapt + PVBC, edge cue, 250 ms)
RC = dict(w_scale=600.0, sigma=25.0, pc_w_pvbc=150.0, pvbc_w_pc=3.0, cue_pa=10000.0, cue_steps=40, T=2500, cue_width=100)


def verify_go():
    print(f"gap#5 ECKER VERIFY-GO mechanism attribution (dt=0.1). REAL=w600+neg-a+PVBC. GO iff REAL decodes localized-traveling "
          f"(DECODE_r>0.6, width<8, |growth|<4, shuffle_r~0) AND no-band collapses; then attribute the trailing-edge suppressor.", flush=True)
    run_one(**RC, b_override=120.0, tag="REAL band+adapt+PVBC")
    run_one(**{**RC, "w_scale": 0.0}, b_override=120.0, tag="C1 NO-BAND (must collapse)")
    run_one(**RC, b_override=0.0, tag="C2 b=0 (spike-adapt off; neg-a still on)")
    run_one(**RC, b_override=0.0, a_override=0.0, tag="C3 FULL adapt-lesion a=0,b=0")
    run_one(**{**RC, "pvbc_w_pc": 0.0}, b_override=0.0, a_override=0.0, tag="C4 no-adapt + NO-PVBC (refractoriness only)")
    print("  READ: C1 must collapse (band required). Among C2/C3/C4 the FIRST arm whose travel COLLAPSES names the "
          "trailing-edge suppressor. shuffle_r~0 in every traveling arm. A band-mediated localized decodable traveling "
          "replay on-substrate = the gap#5 capability.", flush=True)
    print("GAP5-ECKER-RECURRENT DONE", flush=True)


def six_seed():
    seeds = [42, 43, 44, 100, 101, 102]
    mid = N_PC // 2 - 50
    # REAL = the artifact-free DIRECTIONAL replay: FORWARD-biased band (back_frac=0) + MIDDLE cue -> directional travel
    # from an interior location (not an edge-boundary spread). Controls: NO-BAND (must collapse) + SYMMETRIC-from-middle
    # (spreads both ways, must NOT decode -> the asymmetry is load-bearing for DIRECTIONAL replay).
    print(f"gap#5 ECKER 6-SEED directional-replay validation (dt=0.1), seeds {seeds}. GO iff REAL (fwd-band + middle cue) "
          f"decodes a localized directional traveling packet (dec_r>0.6, width<8, |growth|<4, |shuffle|<0.4) in 6/6 "
          f"AND NO-BAND collapses (dec_r<0.3) 6/6 AND SYMMETRIC-middle does NOT decode directionally (dec_r<0.5) 6/6.", flush=True)
    real, noband, symm = [], [], []
    for s in seeds:
        real.append(run_one(**RC, b_override=120.0, seed=s, back_frac=0.0, cue_start=mid, tag=f"REAL fwd+mid s{s}"))
        noband.append(run_one(**{**RC, "w_scale": 0.0}, b_override=120.0, seed=s, back_frac=0.0, cue_start=mid, tag=f"NOBAND s{s}"))
        symm.append(run_one(**RC, b_override=120.0, seed=s, back_frac=1.0, cue_start=mid, tag=f"SYM+mid s{s}"))
    rr = np.array([r["dec_r"] for r in real]); rw = np.array([r["bump_width"] for r in real])
    rg = np.array([r["width_growth"] for r in real]); rsh = np.array([abs(r["shuffle_r"]) for r in real])
    rdr = np.array([r["dec_range"] for r in real]); nb = np.array([r["dec_r"] for r in noband])
    sm = np.array([r["dec_r"] for r in symm]); smw = np.array([r["bump_width"] for r in symm])
    real_go = int(((rr > 0.6) & (rw < 8) & (np.abs(rg) < 4) & (rsh < 0.4)).sum())
    nb_collapse = int((nb < 0.3).sum())
    sym_fail = int((sm < 0.5).sum())
    print(f"\n=== 6-SEED SUMMARY ===", flush=True)
    print(f"REAL dec_r: {np.round(rr,3).tolist()} mean={rr.mean():.3f} min={rr.min():.3f}", flush=True)
    print(f"REAL width={rw.mean():.1f} growth={rg.mean():+.1f} |shuffle|={rsh.mean():.3f} dec_range={rdr.mean():.0f}/100", flush=True)
    print(f"NO-BAND dec_r: {np.round(nb,3).tolist()} max={nb.max():.3f}", flush=True)
    print(f"SYMMETRIC-middle dec_r: {np.round(sm,3).tolist()} width={smw.mean():.1f} (spreads, should not decode directionally)", flush=True)
    verdict = "GO" if (real_go == 6 and nb_collapse == 6 and sym_fail == 6) else "NO-GO"
    print(f"REAL directional-traveling {real_go}/6 | NO-BAND collapses {nb_collapse}/6 | SYM-mid fails-to-decode {sym_fail}/6 -> {verdict}", flush=True)
    print("GAP5-ECKER-6SEED DONE", flush=True)


def directional():
    # Is the forward travel genuine DIRECTIONAL replay, or an EDGE ARTIFACT (symmetric band can only spread forward
    # from the edge)? Cue the MIDDLE: symmetric band -> spreads BOTH ways (no net trajectory, low dec_r/dec_range);
    # a forward-biased (asymmetric) band -> directional travel from ANY location (real directional replay).
    print(f"gap#5 ECKER DIRECTIONALITY — is the travel real directional replay or an edge artifact? (dt=0.1)", flush=True)
    mid = N_PC // 2 - 50
    run_one(**{**RC, "cue_pa": 10000.0}, b_override=120.0, cue_start=0, back_frac=1.0, tag="symmetric band + EDGE cue (baseline)")
    run_one(**RC, b_override=120.0, cue_start=mid, back_frac=1.0, tag="symmetric band + MIDDLE cue (should SPREAD both ways -> low r)")
    run_one(**RC, b_override=120.0, cue_start=mid, back_frac=0.0, tag="FORWARD-only band + MIDDLE cue (should TRAVEL directionally)")
    run_one(**RC, b_override=120.0, cue_start=0, back_frac=0.0, tag="FORWARD-only band + EDGE cue (directional replay)")
    print("  READ: symmetric+middle should give LOW dec_r + LOW dec_range (spreads both ways, no net trajectory) = the "
          "edge-artifact caveat; forward-only+middle should give HIGH dec_r + HIGH dec_range (directional replay from any "
          "cue) = genuine directional replay via asymmetric learned connectivity (Ecker forward-replay).", flush=True)
    print("GAP5-ECKER-DIRECTIONAL DONE", flush=True)


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else ""
    ({"sixseed": six_seed, "directional": directional}.get(_mode, verify_go))()
