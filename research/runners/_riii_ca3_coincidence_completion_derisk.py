"""R-iii surpass, ON-SUBSTRATE rung: does the project's EXISTING spiking dendritic-COINCIDENCE plateau
(`fused_coincidence_plateau` / `enable_coincidence_detection`, byte-inert when off) COMPLETE a partial CA3 cue
on the REAL SimulationBridge where the LINEAR point-neuron CA3 could not (the CYCLE-1064 boundary)? The minimal
numpy de-risk (CYCLE 1065) proved a supra-linear plateau + synaptic clustering is SUFFICIENT in principle; this
realizes it on the spiking substrate with NO new `sim/` edit — it flips `coincidence_detector=True` on the
`ca3->ca3` recurrent RegionPathway (so those recurrent synapses route through the supralinear NMDA-spike plateau)
and reads the plateau in WEIGHTED-DRIVE mode (`coincidence_weighted_drive`: c_drive = sum of effective weight over
coincident recurrent inputs) so a held-out ensemble MEMBER's LTP-strengthened partners cross the all-or-none switch
where a non-member's weak inputs do not (Poirazi-Brannon-Mel 2003 weighted subunit; Major-Larkum-Schiller NMDA
spike). Reuse-by-import of the CYCLE-1064 CA3 harness + the validated trisynaptic helpers.

The ONLY variable vs CYCLE 1064: coincidence-plateau ON vs OFF (OFF is byte-identical to the CYCLE-1064 LINEAR CA3).
GO = coincidence-ON COMPLETES the held-out (non-cued) stored neurons from a partial cue, FAR above coincidence-OFF
(the mechanism is load-bearing), SPECIFICALLY (non-stored neurons are not completed), and the NO-TRAIN control
collapses (the weighted plateau reads the LEARNED attractor, not merely "any coincidence fires everything").

Anti-cheats: (A) coincidence-OFF fails = the CYCLE-1064 boundary (the plateau is load-bearing, not more drive);
(B) SPECIFICITY -- non-stored neurons stay silent (the weighted plateau reads within-ensemble LTP, not raw fan-in);
(C) NO-TRAIN -- coincidence ON but no recurrent LTP -> completion collapses (structure is load-bearing). numpy-smoke
runs on CPU; the real run is GPU (SIM_BACKEND=cupy). NO `sim/` edit.
"""
from __future__ import annotations
import argparse, time
import numpy as np
from research.runners.validate_trisynaptic_loop import measure_region_response, build_drive_pattern


def _build(seed, n_lang=384, n_ec=200, n_dg=300, n_ca3=150, n_ca1=120, ca3w=6.0, ca3_density=0.5,
           coincidence=True, k_thresh=18.0, plateau_strength=120.0, weighted=True, two_comp=False, train=True,
           hebb_max=None, mg=None, apical_R=None, apical_gc=None, hebb_lr=None, hebb_decay=None, hebb_sym=False,
           hebb_rate=False, coact_decay=None, coact_thresh=None, ca3_fb_inhib=None, ca3_fb_n=None, mossy_weight=None,
           ca3_to_ca1_density=0.30, ca1_fb_inhib=None, ca1_fb_n=None):
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import build_biological_brain_regions
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang, n_motor_per_action=16, n_motor_fs_per_action=4, enable_motor_fs=True,
        enable_language_output=True, n_lang_output=n_lang, enable_hippocampus_consolidation=True,
        n_ec=n_ec, n_dg=n_dg, n_ca3=n_ca3, n_ca1=n_ca1, ca3_recurrent_density=ca3_density,
        ca3_recurrent_weight=(ca3w if train else 1.5), ca3_to_ca1_density=ca3_to_ca1_density)
    if coincidence:
        # Route the ca3->ca3 recurrent pathway through the dendritic-coincidence plateau (runner-side flip of the
        # returned dataclass -- NO sim/ edit). The recurrent synapses now trigger a supralinear NMDA-spike plateau
        # when their coincident WEIGHTED drive crosses k_thresh, instead of summing linearly at the point soma.
        for p in pathways:
            if getattr(p, "from_region", None) == "ca3" and getattr(p, "to_region", None) == "ca3":
                p.coincidence_detector = True
    if ca3_fb_inhib is not None:
        # CYCLE-1072 FIX (research gate 2026-07-09): CA3 has NO feedback inhibition wired (internal_density=0.0
        # leaves its 15% inhibitory cells unconnected; every X->ca3 pathway is excitatory) -> uncapped recurrent
        # excitation spreads activity to 35-47% of CA3 (the distributed code that blocks attractor formation). Add a
        # ca3_pv_basket FS FEEDBACK-inhibition pool (E->I->E loop) -- a copy of the ALREADY-VALIDATED dg_pv_basket
        # sparsifier wiring (text_minimal_isolation.py:699-706,1100-1109), but FEEDBACK (ca3->basket->ca3) not
        # feedforward. Caps active-cell count (sparsity) AND paces survivors into synchronous gamma volleys (PING),
        # fixing BOTH the sparsity + the asynchrony with one mechanism. Runner-side append; NO sim/ edit.
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronType
        _nb = int(ca3_fb_n) if ca3_fb_n is not None else max(8, int(0.25 * n_ca3))
        regions.append(BrainRegion(
            name="ca3_pv_basket", n_neurons=_nb, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(RegionPathway(from_region="ca3", to_region="ca3_pv_basket",
                                      density=0.40, weight_mean=5.0, weight_jitter=0.2, plastic=False))
        pathways.append(RegionPathway(from_region="ca3_pv_basket", to_region="ca3",
                                      density=1.0, weight_mean=float(ca3_fb_inhib), weight_jitter=0.2, plastic=False))
    if ca1_fb_inhib is not None:
        # CYCLE-1086 lever: CA1 feedback inhibition -> SPARSE ca1 firing during SWR replay. The consolidation is
        # assembly-specific ONLY when ca1 fires sparsely (broad ca1 firing potentiates the Schaffer non-specifically);
        # biology ensures ca1 sparsity via CA1 basket-cell feedback inhibition. A copy of the ca3_pv_basket wiring
        # for ca1 (E->I->E feedback loop). Runner-side append; NO sim/ edit.
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronType
        _nb1 = int(ca1_fb_n) if ca1_fb_n is not None else max(8, int(0.25 * n_ca1))
        regions.append(BrainRegion(
            name="ca1_pv_basket", n_neurons=_nb1, exc_fraction=0.0, internal_density=0.0,
            exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name))
        pathways.append(RegionPathway(from_region="ca1", to_region="ca1_pv_basket",
                                      density=0.40, weight_mean=5.0, weight_jitter=0.2, plastic=False))
        pathways.append(RegionPathway(from_region="ca1_pv_basket", to_region="ca1",
                                      density=1.0, weight_mean=float(ca1_fb_inhib), weight_jitter=0.2, plastic=False))
    if mossy_weight is not None:
        # Rung 2 (mossy DETONATOR, Kandel Ch 54): strengthen the sparse dg->ca3 mossy synapses so a few DG-selected
        # CA3 cells fire HARD (detonate) from their DG input, while the feedback inhibition suppresses the rest ->
        # a SPARSE + STRONGLY-FIRING SELECTIVE ensemble the co-activity rule can bind. Runner-side weight bump of the
        # returned dg->ca3 pathway (like the coincidence flip); NO sim/ edit.
        for p in pathways:
            if getattr(p, "from_region", None) == "dg" and getattr(p, "to_region", None) == "ca3":
                p.weight_mean = float(mossy_weight)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions); cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0; cfg.seed = seed; cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False; cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = True; cfg.stdp_w_max = max(10.0, 2.5 * ca3w); cfg.fast_spike_reset = True
    if hebb_lr is not None:
        cfg.hebbian_learning_rate = float(hebb_lr)   # CYCLE-1068 formation arc: does a stronger rate-Hebbian write within-ensemble specificity?
    if hebb_decay is not None:
        cfg.hebbian_weight_decay = float(hebb_decay)  # decay-off test: distinguishes "offset never satisfied" from "decay eats the potentiation"
    if hebb_sym:
        cfg.hebbian_symmetric = True   # offset-free co-activity: potentiate synchronously-co-firing ensemble members (the CA3 attractor fix)
    if hebb_rate:
        cfg.hebbian_rate_window = True  # windowed co-activity (BCM/rate-Hebbian): the robust attractor-formation rule
        if coact_decay is not None:
            cfg.hebbian_coactivity_decay = float(coact_decay)
        if coact_thresh is not None:
            cfg.hebbian_coactivity_thresh = float(coact_thresh)
    # ROOT-CAUSE FIX (CYCLE 1066): hebbian_max_weight defaults to 1.0, so the active rate-Hebbian rule DRIVES the
    # ca3->ca3 recurrents (init ca3w=6) DOWN toward 1.0 -> it FLATTENS the attractor instead of potentiating within-
    # ensemble co-active pairs. Raise it above the design weight so rate-Hebbian can WRITE the specific attractor.
    cfg.hebbian_max_weight = float(hebb_max) if hebb_max is not None else max(30.0, 5.0 * ca3w)
    if coincidence:
        cfg.enable_coincidence_detection = True
        cfg.coincidence_weighted_drive = bool(weighted)
        cfg.coincidence_k_threshold = float(k_thresh)
        cfg.coincidence_plateau_strength = float(plateau_strength)
        cfg.enable_two_compartment_dap = bool(two_comp)
        if mg is not None:
            cfg.nmda_mg_concentration = float(mg)   # lower -> Mg2+ block opens -> plateau flows at rest (bootstrap test)
        if two_comp:
            # apical coupling regime (CYCLE 1067 next mechanism): raise apical_R so the clustered plateau current
            # depolarizes the high-local-resistance apical (large dV for small I) -> Mg-regenerative dAP -> soma.
            if apical_R is not None:
                cfg.apical_R = float(apical_R)
            if apical_gc is not None:
                cfg.apical_g_couple = float(apical_gc)
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


_GATES = ["ca3_swr_burst", "dg_to_ca3", "ec_to_dg", "lang_to_ec"]


def _set_gates(bridge, v):
    for g in _GATES:
        try:
            bridge.set_plasticity_gate(g, v)
        except Exception:
            pass


def run_seed(seed, n_mem=2, train_events=100, drive_pA=200.0, do_train=True, coincidence=True,
             n_lang=384, n_ca3=150, n_dg=300, ca3_density=0.5, ca3_weight=6.0, k_thresh=18.0,
             plateau_strength=120.0, weighted=True, two_comp=False, hebb_max=None,
             apical_R=None, apical_gc=None, hebb_lr=None, hebb_decay=None, hebb_sym=False,
             hebb_rate=False, coact_decay=None, coact_thresh=None,
             reset_steps=15, drive_steps=55, recall_steps=60):
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    bridge = _build(seed, n_lang=n_lang, n_ca3=n_ca3, n_dg=n_dg, ca3_density=ca3_density, ca3w=ca3_weight,
                    coincidence=coincidence, k_thresh=k_thresh, plateau_strength=plateau_strength,
                    weighted=weighted, two_comp=two_comp, train=do_train, hebb_max=hebb_max,
                    apical_R=apical_R, apical_gc=apical_gc, hebb_lr=hebb_lr, hebb_decay=hebb_decay, hebb_sym=hebb_sym,
                    hebb_rate=hebb_rate, coact_decay=coact_decay, coact_thresh=coact_thresh)
    rm = bridge.region_manager
    lang = list(rm.indices("language_input"))
    ca3_idx = list(rm.indices("ca3"))
    ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    n_lang = len(lang)
    patterns = [build_drive_pattern(n_neurons=n_lang, sparsity=0.1, seed=seed * 100 + m) for m in range(n_mem)]
    stored = {}

    if do_train:
        _set_gates(bridge, 1.0)
    rec_last = min(10, max(1, train_events // 3))
    lang_arr = np.asarray(lang, dtype=np.int64)
    for m, pat in enumerate(patterns):
        drv = cp.asarray(lang_arr[pat], dtype=cp.int64)
        spikes = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for ev in range(train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[drv] = float(drive_pA)
            recording = ev >= train_events - rec_last
            for _ in range(drive_steps):
                bridge._run_one_simulation_step()
                if recording:
                    spikes += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        bridge.cp_external_input_current[:] = 0.0
        sp = to_host(spikes)
        n_stored = max(4, int(0.10 * len(ca3_idx)))
        top = np.argsort(-sp)[:n_stored]
        top = top[sp[top] > 0]
        stored[m] = np.array([ca3_idx[i] for i in top], dtype=np.int64)
    if do_train:
        _set_gates(bridge, 0.0)

    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    stored_all = set(int(x) for m in range(n_mem) for x in stored[m])
    non_stored = np.array([g for g in ca3_idx if int(g) not in stored_all], dtype=np.int64)

    def _cdrive_for_cue(cue_global):
        """DIRECT mechanism diagnostic: the WEIGHTED coincident drive c_drive[post] = sum_j w_eff[post,j]*mask*x[j]
        that fused_coincidence_plateau reads, for a cue indicator x (the partial-cue CA3 neurons all firing). Uses
        the SAME masked-weighted transposed matvec the bridge uses (cp_connections.data masked by the coincidence
        synapse mask), so it tells us the plateau's ACTUAL input scale -> we calibrate k_thresh to the real gap."""
        if getattr(bridge, "cp_coincidence_synapse_mask", None) is None or bridge.cp_connections is None:
            return None
        from sim.backend import get_sparse_module
        csp, _ = get_sparse_module(), None
        nnz = int(bridge.cp_connections.nnz)
        mask = bridge.cp_coincidence_synapse_mask[:nnz].astype(cp.float32)
        data = bridge.cp_connections.data[:nnz] * mask
        mat = csp.csr_matrix((data, bridge.cp_connections.indices, bridge.cp_connections.indptr),
                             shape=bridge.cp_connections.shape)
        x = cp.zeros(bridge.cp_connections.shape[0], dtype=cp.float32)
        x[cp.asarray(cue_global, dtype=cp.int64)] = 1.0
        return to_host((mat.T @ x))

    held_list, cue_list, nonstored_list = [], [], []
    diag = {"held_cdrive": [], "nonstored_cdrive": []}
    for m in range(n_mem):
        se = stored[m]
        if len(se) < 4:
            return None
        np.random.default_rng(seed + m).shuffle(se)
        n_part = max(2, int(0.5 * len(se)))
        cue, held = se[:n_part], se[n_part:]
        part_resp = measure_region_response(bridge, "ca3", cue, drive_pA=drive_pA, drive_region="ca3", n_steps=recall_steps)
        held_pos = [ca3_pos[int(g)] for g in held if int(g) in ca3_pos]
        cue_pos = [ca3_pos[int(g)] for g in cue if int(g) in ca3_pos]
        ns_pos = [ca3_pos[int(g)] for g in non_stored[:40] if int(g) in ca3_pos]
        held_act = float(np.mean(part_resp[held_pos])) if held_pos else 0.0
        cue_act = float(np.mean(part_resp[cue_pos])) if cue_pos else 1.0
        ns_act = float(np.mean(part_resp[ns_pos])) if ns_pos else 0.0
        held_list.append(held_act / (cue_act + 1e-9))
        nonstored_list.append(ns_act / (cue_act + 1e-9))
        cd = _cdrive_for_cue(cue) if coincidence else None
        if cd is not None:
            # cd is GLOBAL-indexed (length n_neurons) -> index by GLOBAL neuron id, not CA3-local position.
            diag["held_cdrive"].append(float(np.mean([cd[int(g)] for g in held])) if len(held) else 0.0)
            diag["nonstored_cdrive"].append(float(np.mean([cd[int(g)] for g in non_stored[:40]])) if len(non_stored) else 0.0)
    return {"heldout_completion": float(np.mean(held_list)),
            "nonstored_completion": float(np.mean(nonstored_list)),
            "held_cdrive": float(np.mean(diag["held_cdrive"])) if diag["held_cdrive"] else None,
            "nonstored_cdrive": float(np.mean(diag["nonstored_cdrive"])) if diag["nonstored_cdrive"] else None,
            "n_stored": int(np.mean([len(stored[m]) for m in range(n_mem)]))}


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--train-events", type=int, default=100)
    ap.add_argument("--ca3-density", type=float, default=0.5)
    ap.add_argument("--ca3-weight", type=float, default=6.0)
    ap.add_argument("--k-thresh", type=float, default=18.0, help="coincidence plateau threshold (WEIGHT units, weighted_drive)")
    ap.add_argument("--plateau-strength", type=float, default=120.0)
    ap.add_argument("--two-comp", action="store_true", help="regenerate the plateau on the apical dAP compartment (anti-runaway)")
    ap.add_argument("--diag-only", action="store_true", help="run ONLY coincidence-ON + report c_drive (fast mechanism read)")
    ap.add_argument("--hebb-max", type=float, default=None, help="hebbian_max_weight (default max(30,5*ca3w); the attractor-forming fix)")
    ap.add_argument("--apical-R", type=float, default=None, help="apical input resistance (thin-high-R dendrite; the CYCLE-1068 completion regime)")
    ap.add_argument("--apical-gc", type=float, default=None, help="apical->soma coupling")
    ap.add_argument("--hebb-lr", type=float, default=None, help="hebbian_learning_rate")
    ap.add_argument("--hebb-decay", type=float, default=None, help="hebbian_weight_decay (0 = off)")
    ap.add_argument("--hebb-sym", action="store_true", help="SYMMETRIC (offset-free) co-activity Hebbian -- forms the CA3 attractor from synchronous co-firing")
    ap.add_argument("--hebb-rate", action="store_true", help="RATE-WINDOW (BCM) co-activity Hebbian")
    ap.add_argument("--coact-decay", type=float, default=None, help="co-activity trace decay")
    ap.add_argument("--coact-thresh", type=float, default=None, help="co-activity potentiation threshold")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii CA3 coincidence completion] k_thresh={a.k_thresh} density={a.ca3_density} weight={a.ca3_weight} "
          f"strength={a.plateau_strength} two_comp={a.two_comp} hebb_sym={a.hebb_sym} | coincidence-ON vs OFF (=CYCLE-1064 linear) held-out completion", flush=True)
    rows = []
    kw = dict(train_events=a.train_events, ca3_density=a.ca3_density, ca3_weight=a.ca3_weight,
              k_thresh=a.k_thresh, plateau_strength=a.plateau_strength, two_comp=a.two_comp, hebb_max=a.hebb_max,
              apical_R=a.apical_R, apical_gc=a.apical_gc, hebb_lr=a.hebb_lr, hebb_decay=a.hebb_decay, hebb_sym=a.hebb_sym,
              hebb_rate=a.hebb_rate, coact_decay=a.coact_decay, coact_thresh=a.coact_thresh)
    for s in seeds:
        t0 = time.time()
        if a.diag_only:
            on = run_seed(s, do_train=True, coincidence=True, **kw)
            if on is None:
                print(f"  [seed {s}] NOT-EVALUABLE"); continue
            print(f"  [seed {s} DIAG] c_drive: held={on.get('held_cdrive')} nonstored={on.get('nonstored_cdrive')} "
                  f"| held-out completion (ON)={on['heldout_completion']:.3f} non-stored={on['nonstored_completion']:.3f} "
                  f"n_stored={on['n_stored']} ({time.time()-t0:.0f}s)", flush=True)
            continue
        on = run_seed(s, do_train=True, coincidence=True, **kw)               # plateau ON, trained
        off = run_seed(s, do_train=True, coincidence=False, **kw)             # LINEAR (CYCLE-1064 baseline), trained
        notr = run_seed(s, do_train=False, coincidence=True, **kw)            # plateau ON, NO recurrent LTP
        if on is None or off is None or notr is None:
            print(f"  [seed {s}] NOT-EVALUABLE (stored ensemble too small)"); continue
        row = {"seed": s, "on_held": on["heldout_completion"], "off_held": off["heldout_completion"],
               "notrain_held": notr["heldout_completion"], "on_nonstored": on["nonstored_completion"],
               "held_cdrive": on.get("held_cdrive"), "nonstored_cdrive": on.get("nonstored_cdrive"),
               "gain_vs_linear": on["heldout_completion"] - off["heldout_completion"],
               "gain_vs_notrain": on["heldout_completion"] - notr["heldout_completion"], "n_stored": on["n_stored"]}
        rows.append(row)
        _cd = f"c_drive[held={on.get('held_cdrive'):.1f} nonstored={on.get('nonstored_cdrive'):.1f}]" if on.get("held_cdrive") is not None else ""
        print(f"  [seed {s}] held-out completion: COINC-ON={on['heldout_completion']:.3f} LINEAR-OFF={off['heldout_completion']:.3f} "
              f"NO-TRAIN={notr['heldout_completion']:.3f} | non-stored={on['nonstored_completion']:.3f} {_cd} "
              f"(vs-linear={row['gain_vs_linear']:+.3f} vs-notrain={row['gain_vs_notrain']:+.3f}) ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        on_h = [r["on_held"] for r in rows]
        gl = [r["gain_vs_linear"] for r in rows]
        gn = [r["gain_vs_notrain"] for r in rows]
        ns = [r["on_nonstored"] for r in rows]
        go = (all(h > 0.30 for h in on_h) and all(g > 0.15 for g in gl)
              and all(g > 0.15 for g in gn) and all(n < 0.20 for n in ns))
        print(f"\n  AGGREGATE: COINC-ON held-out={np.mean(on_h):.3f} | gain vs LINEAR={np.mean(gl):+.3f} "
              f"vs NO-TRAIN={np.mean(gn):+.3f} | non-stored={np.mean(ns):.3f}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the spiking dendritic-COINCIDENCE plateau COMPLETES the held-out CA3 neurons from a partial cue where the LINEAR point-neuron CA3 (CYCLE 1064) could not, specifically (non-stored silent) and dependent on the learned attractor (no-train collapses) -> the R-iii CA3 completion boundary is SURPASSED on the real spiking substrate, reuse-only' if go else 'the plateau does not yet cleanly complete + separate at these params; sweep k_thresh / density / plateau_strength / two_comp'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
