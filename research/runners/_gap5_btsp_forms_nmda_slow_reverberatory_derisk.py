"""Gap #5 OPEN RESIDUAL — does an EMERGENTLY-FORMED (BTSP one-shot, NOT hand-installed) within-assembly ca3->ca3
attractor reach the SOMATIC slow-NMDA reverberatory completion operating point (the 2026-08-10 GO)?

CONTEXT (VERIFIED). `_gap5_ca3_nmda_slow_reverberatory_derisk.py` reached a 6/6 bistable + cue-specific CA3 completion
on the POINT soma with a HAND-INSTALLED perfect within-assembly potentiation W (idealized outcome of a perfect
recurrent LTP; frozen). Its own "Honest scope" names the OPEN residual: EMERGENT FORMATION into this reverberatory
operating point -- the substrate's rate-Hebbian COLLAPSES ca3->ca3 to a uniform fixed point (2026-07-17), so the
attractor never forms via rate-LTP. The 2026-07-18 BTSP<->bistable UNIFICATION formed a BTSP attractor but read it via
the DENDRITIC two-compartment coincidence readout (cue ~0.18), NEVER via this somatic slow-NMDA reverberatory gate. So
"BTSP formation -> somatic slow-NMDA completion" is the genuinely-untested combination this runner closes-or-bounds.

MECHANISM (brain-based; ONE spiking substrate; runner-side only, NO sim/ edit):
  ENCODE  BTSP plateau-gated ONE-SHOT storing (Bittner-Magee 2017; Milstein-Magee 2021). The bistable APICAL compartment
          (cfg.enable_bdsp + bdsp_apical_bistable) is enabled ONLY to supply the plateau IS_post gate
          (_is_post = max(cp_v_apical - v_hold, 0)); BDSP *learning* is OFF (bdsp_learning_rate=0.0) -- this is NOT the
          dendritic deep-credit/BDSP hidden-credit rule, which is tested-NEGATIVE
          (2026-05-17-dendritic-credit-assignment-NEGATIVE). During a co-fire drive the assembly cells get BOTH
          pre-eligibility (co-firing) AND a plateau (IS_post) -> fused_btsp_update potentiates the WITHIN-assembly
          recurrent one-shot, SPECIFICALLY (member->non-member post has no plateau). Weights come from the PLASTICITY
          RULE (fused_btsp_update), never a hand-set constant -- ASSERTED (w_within grows from baseline; between stays;
          no C.data[within]=W write in the BTSP arm).
  KOPSICK optional homeostatic DIVISIVE downscaling of each POST cell's TOTAL incoming recurrent weight to a set-point T
          (Kopsick-Ascoli 2024) -- applied to ALL ca3->ca3 incoming (post=COLUMN; sim applies connections.T@drive), so
          it PRESERVES the within>>between ratio BTSP created (BTSP stays load-bearing) while giving every seed/neuron
          the SAME effective recurrent gain (seed-robustness).
  READ    the SAME slow-NMDA reverberatory gate as the committed runner: hard-silence (clears the tau=100ms
          g_nmda_recurrent so a latched high state cannot leak), drive a condition, warm (slow NMDA builds), accumulate
          held-out firing over a long window; frozen plasticity. The readout machinery below is COPIED VERBATIM from
          _gap5_ca3_nmda_slow_reverberatory_derisk.run_seed to guarantee an IDENTICAL measurement; the ONLY change is
          hand-install -> BTSP formation. A HAND-INSTALL cross-check arm on THIS bridge empirically proves the readout
          is faithful to the committed 6/6 GO (same W -> same cue/perm/nocue).

ANTI-CHEATS (the established instrument; omitting these produced 3 retractions in this arc):
  - plasticity FROZEN at recall (enable_hebbian_learning/stdp/btsp/bdsp all False during the read).
  - OU controllable: report OU-off AND OU-on.
  - PERMUTED cue (random non-assembly cells) -> held-out must NOT reactivate (GO needs cue >= 3x perm).
  - NO-ENCODING control: skip the BTSP encode -> weights stay at baseline -> completion collapses (BTSP load-bearing).
  - NO-PLATEAU control: run the co-fire WITHOUT the apical plateau -> no IS_post -> no potentiation -> collapse
    (proves plateau-GATED one-shot, not mere co-fire).
  - RECURRENCE-ZERO: zero ca3->ca3 -> completion collapses (it is the reverberation, not cue re-drive).
  - silent-rest nocue <= 0.10 (no always-on limit cycle / self-ignition artifact).
  - GENUINE-FORMATION: assert the formed within-assembly weight AROSE from fused_btsp_update (w_within_after > baseline
    AND within_dw > 3x between_dw) and was NEVER hand-set in the BTSP arm.
  - cfg.seed=seed set explicitly; build-twice threshold-hash determinism check.

GO bar (6-seed 42/43/44/100/101/102): held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND
held_nocue<=0.10, on all/most seeds. A PARTIAL (n/6) or a quantified NEGATIVE (how far the formed weight is from the
W~5000 operating point) is a first-class honest result. SIM_BACKEND=cupy.
  Run: SIM_BACKEND=cupy python -m research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk \
         --seeds 42 43 44 100 101 102 --both-ou --json research/findings/raw/_gap5_btsp_nmda/btsp_forms_6seed.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
import numpy as np

from research.runners._riii_ca3_coincidence_completion_derisk import _build


def _threshold_hash(bridge):
    from sim.backend import to_host
    arr = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if arr is None:
        return "none"
    return hashlib.sha1(np.asarray(to_host(arr)).tobytes()).hexdigest()[:12]


def _csr_row_col(cp, C):
    n = C.shape[0]
    rows = cp.repeat(cp.arange(n, dtype=cp.int64), cp.diff(C.indptr))
    cols = C.indices.astype(cp.int64)
    return rows, cols


class Readout:
    """Namespace holding the slow-NMDA readout closures + masks (see make_readout)."""
    pass


def make_readout(bridge, seed, *, n_assembly=3, assembly_frac=0.18, cue_frac=0.5,
                 drive_pA=300.0, warm_steps=200, read_steps=200, silence_steps=50, assemblies_ext=None):
    """Build the slow-NMDA reverberatory readout on `bridge`. COPIED VERBATIM (variable-for-variable) from
    _gap5_ca3_nmda_slow_reverberatory_derisk.run_seed lines ~83-203 so the MEASUREMENT is provably identical; the
    hand-install cross-check arm empirically confirms it reproduces the committed 6/6 GO on this bridge.

    assemblies_ext (ADDITIVE, default None => byte-identical): when provided, a list of CA3 GLOBAL-index arrays to use
    as the assembly membership INSTEAD of the internal random permutation. This is the seam for the end-to-end
    composition (`_gap5_emergent_end_to_end_episodic_loop_derisk.py`): the membership is DG-SELECTED (mossy detonator),
    NOT a hand-set/random-permutation mask. All downstream masks/closures are built from whatever `assemblies` ends up
    being, so the entire instrument (formation, cue/perm/nocue, cross_dw) is identical, only the membership emergent."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    ca3_arr = cp.asarray(ca3_idx, dtype=cp.int64)
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}

    C = bridge.cp_connections
    if not hasattr(C, "indptr"):
        C = C.tocsr(); bridge.cp_connections = C
    rows, cols = _csr_row_col(cp, C)
    baseline_data = C.data.copy()

    n = C.shape[0]
    is_ca3 = cp.zeros(n, dtype=cp.bool_); is_ca3[ca3_arr] = True
    rec_mask = is_ca3[rows] & is_ca3[cols]                 # all ca3->ca3 (excitatory recurrent) synapses
    n_rec = int(to_host(cp.sum(rec_mask)))

    if assemblies_ext is not None:
        # EMERGENT membership (end-to-end composition): DG-selected CA3 global indices, not a random permutation.
        assemblies = [np.asarray(A, dtype=np.int64) for A in assemblies_ext]
        a_size = int(np.mean([len(A) for A in assemblies])) if assemblies else 0
    else:
        rng = np.random.default_rng(seed)
        perm_idx = rng.permutation(len(ca3_idx))
        a_size = max(6, int(assembly_frac * len(ca3_idx)))
        assemblies = [ca3_idx[perm_idx[a * a_size:(a + 1) * a_size]] for a in range(n_assembly)]
    withinA_masks = []
    within_union = cp.zeros(len(rows), dtype=cp.bool_)
    for A in assemblies:
        is_A = cp.zeros(n, dtype=cp.bool_); is_A[cp.asarray(A)] = True
        m = is_A[rows] & is_A[cols]
        withinA_masks.append(m); within_union |= m
    between_mask = rec_mask & ~within_union

    def restore_baseline():
        C.data[:] = baseline_data

    def install_selective(W):
        restore_baseline()
        for m in withinA_masks:
            C.data[m] = cp.float32(W)

    def zero_recurrents():
        restore_baseline()
        C.data[rec_mask] = cp.float32(0.0)

    def hard_silence():
        if getattr(bridge, "cp_izh_c_reset", None) is not None:
            bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
        else:
            bridge.cp_membrane_potential_v[:] = -65.0
        bridge.cp_recovery_variable_u[:] = 0.0
        if getattr(bridge, "cp_firing_states", None) is not None:
            bridge.cp_firing_states[:] = False
        for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise",
                   "cp_conductance_g_e", "cp_conductance_g_i",
                   "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise"):
            _arr = getattr(bridge, _a, None)
            if _arr is not None:
                _arr[:] = 0.0
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(silence_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

    def drive_read(drive_indices, sustain=False):
        hard_silence()
        if drive_indices is not None and len(drive_indices) > 0:
            darr = cp.asarray(np.asarray(drive_indices, dtype=np.int64), dtype=cp.int64)
            bridge.cp_external_input_current[darr] = cp.float32(drive_pA)
        else:
            darr = None
        for _ in range(warm_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        cnt = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for _ in range(read_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            cnt += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        v_on = to_host(cnt) / float(read_steps)
        if not sustain:
            if darr is not None:
                bridge.cp_external_input_current[darr] = 0.0
            return v_on
        if darr is not None:
            bridge.cp_external_input_current[darr] = 0.0
        cnt2 = cp.zeros(len(ca3_idx), dtype=cp.float32)
        for _ in range(read_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
            cnt2 += bridge.cp_firing_states[ca3_arr].astype(cp.float32)
        return v_on, to_host(cnt2) / float(read_steps)

    def eval_assembly(A):
        se = np.asarray(A, dtype=np.int64)
        r = np.random.default_rng(seed * 131 + int(se[0]))
        se = se[r.permutation(len(se))]
        n_cue = max(2, int(cue_frac * len(se)))
        cue, held = se[:n_cue], se[n_cue:]
        held_pos = [ca3_pos[int(g)] for g in held]
        member = set(int(g) for g in se)
        nonmember_pos = [i for g, i in ca3_pos.items() if g not in member]

        v_cue, v_sustain = drive_read(cue, sustain=True)
        nonA = np.asarray([g for g in ca3_idx if int(g) not in member], dtype=np.int64)
        perm_cue = r.choice(nonA, size=len(cue), replace=False)
        v_perm = drive_read(perm_cue)

        held_cue = float(np.mean(v_cue[held_pos])) if held_pos else 0.0
        held_sustain = float(np.mean(v_sustain[held_pos])) if held_pos else 0.0
        held_perm = float(np.mean(v_perm[held_pos])) if held_pos else 0.0
        nonmember_act = float(np.mean(v_cue[nonmember_pos])) if nonmember_pos else 0.0
        return {"held_cue": held_cue, "held_sustain": held_sustain, "held_perm": held_perm,
                "nonmember_act": nonmember_act, "n_held": len(held_pos), "n_cue": n_cue}

    def no_cue_rest():
        v = drive_read(None)
        se0 = np.asarray(assemblies[0], dtype=np.int64)
        r = np.random.default_rng(seed * 131 + int(se0[0]))
        se0 = se0[r.permutation(len(se0))]
        held0 = se0[max(2, int(cue_frac * len(se0))):]
        held0_pos = [ca3_pos[int(g)] for g in held0]
        return float(np.mean(v[held0_pos])) if held0_pos else float(np.mean(v))

    # --- weight stats (genuine-formation instrument) ---
    def w_within():
        return float(to_host(cp.mean(C.data[within_union]))) if int(to_host(cp.sum(within_union))) else 0.0

    def w_between():
        return float(to_host(cp.mean(C.data[between_mask]))) if int(to_host(cp.sum(between_mask))) else 0.0

    def within_sum_per_post():
        """Mean over post cells of the SUM of incoming within-assembly recurrent weight (the reverberatory drive
        scale). post=COLUMN (sim applies connections.T@drive). Returns the mean per-post within-assembly weight sum."""
        wu = within_union
        cols_w = cols[wu]
        data_w = C.data[wu]
        sums = cp.bincount(cols_w, weights=data_w, minlength=n).astype(cp.float32)
        # average over post cells that actually receive within-assembly input
        recv = cp.bincount(cols_w, minlength=n) > 0
        return float(to_host(cp.mean(sums[recv]))) if int(to_host(cp.sum(recv))) else 0.0

    def kopsick_downscale(target_T):
        """Homeostatic divisive downscaling (Kopsick-Ascoli 2024): scale each POST cell's TOTAL incoming ca3->ca3
        recurrent weight to the set-point target_T. Applied to ALL recurrent incoming (rec_mask), so the within>>between
        ratio BTSP created is PRESERVED. post=COLUMN."""
        rm_mask = rec_mask
        cols_r = cols[rm_mask]
        data_r = C.data[rm_mask]
        sums = cp.bincount(cols_r, weights=data_r, minlength=n).astype(cp.float32)
        scale = cp.ones(n, dtype=cp.float32)
        nz = sums > 1e-6
        scale[nz] = cp.float32(target_T) / sums[nz]
        # apply per-synapse: new data = data * scale[post=col]
        new_data = C.data.copy()
        new_data[rm_mask] = data_r * scale[cols_r]
        C.data[:] = new_data

    R = Readout()
    R.cp = cp; R.to_host = to_host
    R._assembly_frac = assembly_frac; R._cue_frac = cue_frac; R._drive_pA = drive_pA
    R._warm = warm_steps; R._read = read_steps; R._silence = silence_steps
    R.ca3_idx = ca3_idx; R.ca3_arr = ca3_arr; R.ca3_pos = ca3_pos
    R.C = C; R.rows = rows; R.cols = cols; R.n = n
    R.baseline_data = baseline_data
    R.assemblies = assemblies; R.withinA_masks = withinA_masks
    R.within_union = within_union; R.between_mask = between_mask; R.rec_mask = rec_mask
    R.a_size = a_size; R.n_rec = n_rec
    R.restore_baseline = restore_baseline
    R.install_selective = install_selective
    R.zero_recurrents = zero_recurrents
    R.hard_silence = hard_silence
    R.drive_read = drive_read
    R.eval_assembly = eval_assembly
    R.no_cue_rest = no_cue_rest
    R.w_within = w_within; R.w_between = w_between
    R.within_sum_per_post = within_sum_per_post
    R.kopsick_downscale = kopsick_downscale
    return R


def _form_one_assembly(bridge, R, ai, *, btsp_w_max, btsp_lr, encode_drive, encode_plateau_pA,
                       train_events, drive_steps, reset_steps, plateau=True):
    """Form ONE assembly (index ai) on `bridge` via BTSP plateau-gated one-shot. Enables the bistable apical (IS_post
    gate; BDSP LEARNING OFF -- NOT the tested-NEGATIVE 2026-05-17-dendritic-credit-assignment-NEGATIVE hidden-credit
    rule) + the BTSP block; drives ONLY assembly ai to co-fire (pre-eligibility) with a plateau (IS_post) so
    fused_btsp_update potentiates its WITHIN-assembly recurrent. No C.data hand-set -> the formed weight is the RULE's
    output. Because ONLY assembly ai is ever driven on this (fresh, isolated) bridge, NO cross-assembly synapse can
    potentiate -- the pattern-separation the DG provides in vivo (temporally-distinct memory encoding episodes)."""
    cp = R.cp
    cfg = bridge.core_config
    cfg.enable_hebbian_learning = False; cfg.enable_stdp = False; cfg.enable_structural_plasticity = False
    cfg.enable_bdsp = True; cfg.bdsp_apical_bistable = True; cfg.bdsp_learning_rate = 0.0
    cfg.coincidence_plateau_self_regen = 2.0; cfg.coincidence_plateau_v_hold = -35.0; cfg.apical_kir_g = 1.0
    cfg.enable_btsp = True; cfg.btsp_learning_rate = float(btsp_lr); cfg.btsp_elig_tau_ms = 1000.0
    cfg.btsp_w_max = float(btsp_w_max); cfg.btsp_hetero_dep = 0.0
    bridge.cp_bdsp_apical_drive = cp.zeros(cfg.num_neurons, dtype=cp.float32)
    assy = R.assemblies[ai]
    assy_arr = cp.asarray(np.asarray(assy, dtype=np.int64), dtype=cp.int64)
    pv = cp.full(len(assy), float(encode_plateau_pA), dtype=cp.float32)
    for _ev in range(train_events):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_bdsp_apical_drive[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        for _st in range(drive_steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[assy_arr] = float(encode_drive)   # co-fire (pre-eligibility)
            bridge.cp_bdsp_apical_drive[:] = 0.0
            if plateau:
                bridge.cp_bdsp_apical_drive[assy_arr] = pv                     # plateau (IS_post) -- gate
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[:] = 0.0
    cfg.enable_bdsp = False; cfg.enable_btsp = False; bridge.cp_bdsp_apical_drive = None


def form_btsp_multi(seed, build_kwargs, R_target, *, btsp_w_max, btsp_lr, encode_drive, encode_plateau_pA,
                    train_events, drive_steps, reset_steps, plateau=True, assemblies_ext=None):
    """EMERGENT multi-assembly formation. Each assembly is formed in its OWN isolated encoding episode (a fresh bridge,
    same seed/connectivity, only that assembly driven), then its BTSP-formed WITHIN-assembly weights are written onto
    the shared readout bridge R_target. This guarantees cross-assembly dW == 0 (specificity by construction) while every
    within-assembly weight is the output of fused_btsp_update (never a hand-set constant). Returns diagnostics.

    assemblies_ext (ADDITIVE, default None): forwarded to each isolated readout so the SAME emergent (DG-selected)
    membership is formed on every episode bridge as on R_target (end-to-end composition)."""
    cp = R_target.cp; to_host = R_target.to_host
    for ai in range(len(R_target.assemblies)):
        bi = _build_bridge(seed, **build_kwargs)      # fresh, isolated (enable_ou False in build_kwargs)
        Ri = make_readout(bi, seed, assembly_frac=R_target._assembly_frac, cue_frac=R_target._cue_frac,
                          drive_pA=R_target._drive_pA, warm_steps=R_target._warm, read_steps=R_target._read,
                          silence_steps=R_target._silence, assemblies_ext=assemblies_ext)
        _form_one_assembly(bi, Ri, ai, btsp_w_max=btsp_w_max, btsp_lr=btsp_lr, encode_drive=encode_drive,
                           encode_plateau_pA=encode_plateau_pA, train_events=train_events,
                           drive_steps=drive_steps, reset_steps=reset_steps, plateau=plateau)
        m = Ri.withinA_masks[ai]
        R_target.C.data[m] = bi.cp_connections.data[m]   # copy ONLY the within-ai BTSP-formed weights
        del bi, Ri
    # diagnostics on the combined matrix
    rows, cols = R_target.rows, R_target.cols
    is_asm = cp.zeros(R_target.n, dtype=cp.bool_)
    for A in R_target.assemblies:
        is_asm[cp.asarray(A)] = True
    cross = R_target.rec_mask & is_asm[rows] & is_asm[cols] & (~R_target.within_union)
    w_within = float(to_host(cp.mean(R_target.C.data[R_target.within_union])))
    cross_dw = (float(to_host(cp.mean(R_target.C.data[cross]))) - 1.5) if int(to_host(cp.sum(cross))) else 0.0
    nonmem_mask = R_target.between_mask & ~cross  # member->non + non->member + non->non
    nonmem_dw = (float(to_host(cp.mean(R_target.C.data[nonmem_mask]))) - 1.5) if int(to_host(cp.sum(nonmem_mask))) else 0.0
    return dict(w_within=w_within, cross_dw=cross_dw, nonmem_dw=nonmem_dw,
                within_sum_per_post=R_target.within_sum_per_post())


def _build_bridge(seed, *, n_ca3, ca3_density, ca3_fb_inhib, nmda_tau, nmda_ratio, enable_ou, element):
    _nmda_rec = (element == "nmda_slow")
    bridge = _build(seed, n_ca3=n_ca3, ca3_density=ca3_density, coincidence=False, two_comp=False,
                    nmda_recurrent=_nmda_rec, nmda_tau=nmda_tau, nmda_ratio=nmda_ratio,
                    ca3_fb_inhib=ca3_fb_inhib, train=False, enable_ou=enable_ou)
    cfg = bridge.core_config
    cfg.enable_hebbian_learning = False; cfg.enable_stdp = False; cfg.enable_structural_plasticity = False
    return bridge


def run_seed(seed, *, n_ca3=400, ca3_density=0.12, assembly_frac=0.18, cue_frac=0.5, ca3_fb_inhib=60.0,
             nmda_tau=100.0, nmda_ratio=1.0, drive_pA=300.0, warm_steps=200, read_steps=200, silence_steps=50,
             enable_ou=False, element="nmda_slow",
             btsp_w_max_grid=(2500.0, 5000.0, 9000.0), btsp_lr=0.05, encode_drive=700.0, encode_plateau_pA=250.0,
             train_events=40, drive_steps=48, reset_steps=15,
             handinstall_W=(2500.0, 5000.0, 9000.0), kopsick_T=None, verbose=True, assemblies_ext=None):
    from sim.backend import get_backend, to_host
    from tools.lab import attributable_to
    cp, _ = get_backend()
    rows_out = []
    build_kwargs = dict(n_ca3=n_ca3, ca3_density=ca3_density, ca3_fb_inhib=ca3_fb_inhib, nmda_tau=nmda_tau,
                        nmda_ratio=nmda_ratio, enable_ou=False, element=element)   # ENCODE episodes are deterministic
    read_kwargs = dict(assembly_frac=assembly_frac, cue_frac=cue_frac, drive_pA=drive_pA,
                       warm_steps=warm_steps, read_steps=read_steps, silence_steps=silence_steps,
                       assemblies_ext=assemblies_ext)

    # ---------- (1) HAND-INSTALL CROSS-CHECK: reproduce the committed 6/6 GO on THIS bridge (readout fidelity) ----------
    bridge = _build_bridge(seed, **{**build_kwargs, "enable_ou": enable_ou})
    R = make_readout(bridge, seed, **read_kwargs)
    R.restore_baseline()
    unt = [R.eval_assembly(A) for A in R.assemblies]           # no-encoding baseline (weak recurrent)
    unt_held_cue = float(np.mean([e["held_cue"] for e in unt]))
    baseline_w_within = R.w_within()
    for W in handinstall_W:
        R.install_selective(W)
        rest = R.no_cue_rest()
        evals = [R.eval_assembly(A) for A in R.assemblies]
        hc = float(np.mean([e["held_cue"] for e in evals]))
        hs = float(np.mean([e["held_sustain"] for e in evals]))
        hp = float(np.mean([e["held_perm"] for e in evals]))
        rows_out.append(dict(seed=seed, arm="handinstall", enable_ou=bool(enable_ou), element=element,
                             W=float(W), w_within=float(W), w_between=baseline_w_within,
                             held_cue=hc, held_sustain=hs, held_perm=hp, held_nocue=rest,
                             no_encoding_held_cue=unt_held_cue,
                             GO=bool(hc >= 0.20 and hc >= 3 * (hp + 1e-6) and hc >= 3 * (rest + 1e-6) and rest <= 0.10)))
        if verbose:
            print(f"  [s{seed} ou{int(enable_ou)} HANDINSTALL W{W:>6.0f}] cue={hc:.3f} sustain={hs:.3f} perm={hp:.3f} "
                  f"nocue={rest:.3f} (noenc={unt_held_cue:.3f}) {'GO' if rows_out[-1]['GO'] else '--'}", flush=True)
    del bridge, R

    # ---------- (2) BTSP EMERGENT FORMATION (primary): isolated per-assembly episodes, sweep the saturation ceiling ---
    for wmax in btsp_w_max_grid:
        bridge = _build_bridge(seed, **{**build_kwargs, "enable_ou": enable_ou})   # readout bridge (OU=condition)
        R = make_readout(bridge, seed, **read_kwargs)
        diag = form_btsp_multi(seed, build_kwargs, R, btsp_w_max=wmax, btsp_lr=btsp_lr, encode_drive=encode_drive,
                               encode_plateau_pA=encode_plateau_pA, train_events=train_events,
                               drive_steps=drive_steps, reset_steps=reset_steps, plateau=True,
                               assemblies_ext=assemblies_ext)
        w1_within, cross_dw, nonmem_dw = diag["w_within"], diag["cross_dw"], diag["nonmem_dw"]
        within_sum = diag["within_sum_per_post"]
        # GENUINE-FORMATION teeth: within grew from the RULE, cross/non-member did NOT (specificity by construction)
        genuine = bool(w1_within > 100.0 and w1_within <= wmax * 1.01
                       and abs(cross_dw) < 0.02 * w1_within and abs(nonmem_dw) < 0.02 * w1_within)

        kT = None
        if kopsick_T is not None:                    # optional homeostatic downscaling (preserves within>>between)
            R.kopsick_downscale(float(kopsick_T)); kT = float(kopsick_T)
            w1_within = R.w_within(); within_sum = R.within_sum_per_post()

        rest = R.no_cue_rest()
        evals = [R.eval_assembly(A) for A in R.assemblies]
        hc = float(np.mean([e["held_cue"] for e in evals]))
        hs = float(np.mean([e["held_sustain"] for e in evals]))
        hp = float(np.mean([e["held_perm"] for e in evals]))
        _saved = R.C.data.copy(); R.C.data[R.rec_mask] = cp.float32(0.0)      # RECURRENCE-ZERO teeth on FORMED matrix
        zr = float(np.mean([R.eval_assembly(A)["held_cue"] for A in R.assemblies])); R.C.data[:] = _saved
        attributable_to(f"[s{seed} ou{int(enable_ou)} wmax{wmax:.0f}] BTSP-formed completion vs NO-ENCODING baseline",
                        hc, unt_held_cue)
        attributable_to(f"[s{seed} ou{int(enable_ou)} wmax{wmax:.0f}] SPECIFICITY: correct-cue vs PERMUTED-cue",
                        hc, hp)
        go = bool(hc >= 0.20 and hc >= 3 * (hp + 1e-6) and hc >= 3 * (rest + 1e-6) and rest <= 0.10)
        rows_out.append(dict(seed=seed, arm="btsp", enable_ou=bool(enable_ou), element=element,
                             btsp_w_max=float(wmax), btsp_lr=btsp_lr, w_within=w1_within, cross_dw=cross_dw,
                             nonmem_dw=nonmem_dw, within_sum_per_post=within_sum, genuine_formation=genuine,
                             kopsick_T=kT, held_cue=hc, held_sustain=hs, held_perm=hp, held_nocue=rest,
                             recurrence_zero_held_cue=zr, no_encoding_held_cue=unt_held_cue, GO=go))
        if verbose:
            print(f"  [s{seed} ou{int(enable_ou)} BTSP wmax{wmax:>6.0f}] cue={hc:.3f} sustain={hs:.3f} perm={hp:.3f} "
                  f"nocue={rest:.3f} | w_in={w1_within:.0f} crossdW={cross_dw:.2f} nonmemdW={nonmem_dw:.2f} "
                  f"sum={within_sum:.0f} genuine={genuine} reczero={zr:.3f} {'GO' if go else '--'}", flush=True)
        del bridge, R

    # ---------- (3) NO-PLATEAU control at the strongest ceiling (plateau-gating has teeth) ----------
    wmax = max(btsp_w_max_grid)
    bridge = _build_bridge(seed, **{**build_kwargs, "enable_ou": enable_ou})
    R = make_readout(bridge, seed, **read_kwargs)
    diag = form_btsp_multi(seed, build_kwargs, R, btsp_w_max=wmax, btsp_lr=btsp_lr, encode_drive=encode_drive,
                           encode_plateau_pA=encode_plateau_pA, train_events=train_events,
                           drive_steps=drive_steps, reset_steps=reset_steps, plateau=False,  # NO plateau -> no IS_post
                           assemblies_ext=assemblies_ext)
    evals = [R.eval_assembly(A) for A in R.assemblies]
    hc = float(np.mean([e["held_cue"] for e in evals]))
    rows_out.append(dict(seed=seed, arm="btsp_noplateau", enable_ou=bool(enable_ou), element=element,
                         btsp_w_max=float(wmax), w_within=diag["w_within"], held_cue=hc, GO=False))
    if verbose:
        print(f"  [s{seed} ou{int(enable_ou)} BTSP-NOPLATEAU wmax{wmax:.0f}] cue={hc:.3f} w_in={diag['w_within']:.1f} "
              f"(should stay ~baseline -> collapse)", flush=True)
    del bridge, R
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--n-ca3", type=int, default=400)
    ap.add_argument("--density", type=float, default=0.12)
    ap.add_argument("--assembly-frac", type=float, default=0.18)
    ap.add_argument("--fb-inhib", type=float, default=60.0)
    ap.add_argument("--nmda-tau", type=float, default=100.0)
    ap.add_argument("--drive-pa", type=float, default=300.0)
    ap.add_argument("--warm-steps", type=int, default=200)
    ap.add_argument("--read-steps", type=int, default=200)
    ap.add_argument("--btsp-wmax", default="2500,5000,9000")
    ap.add_argument("--btsp-lr", type=float, default=0.05)
    ap.add_argument("--encode-drive", type=float, default=700.0)
    ap.add_argument("--encode-plateau", type=float, default=250.0)
    ap.add_argument("--train-events", type=int, default=40)
    ap.add_argument("--handinstall-w", default="2500,5000,9000")
    ap.add_argument("--kopsick-t", type=float, default=None, help="Kopsick per-post total-recurrent set-point (None=off)")
    ap.add_argument("--element", default="nmda_slow", choices=["nmda_slow", "ampa"])
    ap.add_argument("--ou", action="store_true")
    ap.add_argument("--both-ou", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.replace(",", " ").split()]
    wmax_grid = [float(x) for x in a.btsp_wmax.replace(",", " ").split()]
    hi_grid = [float(x) for x in a.handinstall_w.replace(",", " ").split()]
    ou_modes = [False, True] if a.both_ou else [bool(a.ou)]

    _nr = (a.element == "nmda_slow")
    b1 = _build(seeds[0], n_ca3=a.n_ca3, ca3_density=a.density, coincidence=False, two_comp=False,
                nmda_recurrent=_nr, nmda_tau=a.nmda_tau, ca3_fb_inhib=a.fb_inhib, train=False, enable_ou=False)
    b2 = _build(seeds[0], n_ca3=a.n_ca3, ca3_density=a.density, coincidence=False, two_comp=False,
                nmda_recurrent=_nr, nmda_tau=a.nmda_tau, ca3_fb_inhib=a.fb_inhib, train=False, enable_ou=False)
    h1, h2 = _threshold_hash(b1), _threshold_hash(b2)
    print(f"[determinism] threshold-hash build1={h1} build2={h2} -> {'SEEDED' if h1 == h2 else 'UNSEEDED-BUG'}", flush=True)
    del b1, b2

    print(f"[gap5 BTSP-forms-nmda_slow] seeds={seeds} n_ca3={a.n_ca3} density={a.density} fb={a.fb_inhib} "
          f"btsp_wmax={wmax_grid} btsp_lr={a.btsp_lr} enc_drive={a.encode_drive} plateau={a.encode_plateau} "
          f"train_events={a.train_events} kopsick_T={a.kopsick_t} handinstall={hi_grid} ou={ou_modes}", flush=True)
    print("  GO gate (per seed): held_cue>=0.20 & cue>=3x perm & cue>=3x nocue & nocue<=0.10", flush=True)
    all_rows = []
    for ou in ou_modes:
        ngo_btsp = 0
        for s in seeds:
            t0 = time.time()
            rr = run_seed(s, n_ca3=a.n_ca3, ca3_density=a.density, assembly_frac=a.assembly_frac,
                          ca3_fb_inhib=a.fb_inhib, nmda_tau=a.nmda_tau, drive_pA=a.drive_pa,
                          warm_steps=a.warm_steps, read_steps=a.read_steps, enable_ou=ou, element=a.element,
                          btsp_w_max_grid=wmax_grid, btsp_lr=a.btsp_lr, encode_drive=a.encode_drive,
                          encode_plateau_pA=a.encode_plateau, train_events=a.train_events,
                          handinstall_W=hi_grid, kopsick_T=a.kopsick_t)
            all_rows.extend(rr)
            btsp_go = any(r["arm"] == "btsp" and r["GO"] for r in rr)
            ngo_btsp += int(btsp_go)
            best = max([r for r in rr if r["arm"] == "btsp"], key=lambda r: (r["GO"], r["held_cue"]))
            print(f"    (seed {s} ou{int(ou)}: {time.time()-t0:.0f}s) BTSP best wmax={best['btsp_w_max']:.0f} "
                  f"cue={best['held_cue']:.3f} perm={best['held_perm']:.3f} nocue={best['held_nocue']:.3f} "
                  f"w_in={best['w_within']:.1f} genuine={best['genuine_formation']} seedGO={btsp_go}", flush=True)
        print(f"  RESULT ou{int(ou)}: BTSP-formed {ngo_btsp}/{len(seeds)} seeds have >=1 GO working point", flush=True)
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump(all_rows, open(a.json, "w"), indent=1)
        print(f"  wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
