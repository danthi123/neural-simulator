"""gap#5 RANK 2 — SEQUENCE REPLAY (forward, and reverse if present) of a stored ORDER of CA3 assemblies.

2026-07-22 (research gate 2026-07-21-gap5-SWR-generative-replay-research-gate.md, RANK 2). RANK 1 de-risked the
FIRST piece of the SWR generative-replay loop: a SINGLE stored bistable CA3 assembly spontaneously + basin-selectively
reactivates under weak non-specific background, discretely, with none of the 3 retracted confounds (1-seed CLEAN GO).
RANK 2 is the next rung of the Ecker et al. eLife 2022 (71850) template: a SEQUENCE of >=3 assemblies A->B->C stored
as a bistable-completable CHAIN with DIRECTIONAL inter-assembly links (each assembly's cells potentiate onto the NEXT
assembly's cells -- theta-compressed order via BTSP, asymmetric forward + a symmetric component). Then, during the
REST phase (freeze plasticity + weak non-specific background, NO cue), does a spontaneous reactivation trigger an
ORDERED replay A->B->C (forward)? Is there any reverse (C->B->A)?

THE MECHANISM (reuse-by-import of the RANK 1 CLOSED bistable store; NO `sim/` edit):
  - WITHIN phase: BTSP-encode each assembly individually (co-fire + own plateau) -> each is a bistable-completable
    attractor (== RANK 1's single-assembly store, done 3x).
  - CHAIN phase: theta-compressed SWEEPS. Each forward sweep drives A then B then C in short successive windows, each
    with co-fire (pre-eligibility) + a plateau (IS_post). Because the BTSP pre-eligibility (tau 1000ms) from an EARLIER
    assembly persists into a LATER assembly's plateau window, the rule potentiates A->B, B->C (forward). Making the
    plateau TRANSIENT during the sweep (self_regen=0, restored to 0.15 for recall) is LOAD-BEARING: with the bistable
    latch ON, an earlier assembly's plateau stays latched into the later window and reverse links (B->A) form too ->
    the chain becomes symmetric. Transient plateau => clean asymmetric forward. A small number of REVERSE sweeps
    (C->B->A) adds the symmetric component (for reverse replay).

GO GATE (verify, don't assert): forward ordered-replay score (fraction of multi-assembly events whose per-assembly
ONSET order is strictly forward A->B->C) >> the shuffled-order floor (1/n! chance) and >> reverse_frac, on the CPU
smoke; the net RESTS silent between events (discrete). Anti-cheats (each retires a named risk):
  - SCRAMBLE inter-assembly weights (destroy the directional chain, keep within-assembly) -> ORDER collapses to the
    shuffled floor (forward_frac ~ reverse_frac ~ chance). [the chain is load-bearing for the ORDER]
  - NO-NOISE (OU/poisson off) -> SILENT, 0 events.        [THE ACID TEST -- retires the self-sustaining artifact]
  - NO-ENCODE -> no ordered events.                        [retires the noise artifact]
  - FROZEN plasticity during rest (byte-verify).           [retires the plasticity+noise (Wang) confound]
  - DENDRITIC-RESET verified.                              [retires the `_hard_silence` bug]

HONEST NOTE: the OLD v16 sequence-storage was a BOUNDARY + reverse-replay was NULL -- BOTH on a NON-bistable
substrate, predating the bistable-completion keystone. This RE-de-risks on the bistable substrate. A partial/negative
is a real, valuable result (the traveling-wave hand-off A->B->C is the genuine open question). A "replay" with NO
noise is the retracted artifact, NOT a GO.

CPU-smoke: SIM_BACKEND=numpy python -m research.runners._gap5_sequence_replay_derisk --seeds 42 --n-ca3 2000 \
    --within-events 24 --chain-fwd 24 --chain-rev 8 --rest-steps 1400
Full run (GPU): SIM_BACKEND=cupy python -m research.runners._gap5_sequence_replay_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

# reuse-by-import of the RANK 1 CLOSED bistable store machinery (NO sim/ edit anywhere)
from research.runners._riii_ca3_coincidence_completion_derisk import _build, _set_gates  # noqa: E402
from research.runners._gap5_spontaneous_reactivation_derisk import (  # noqa: E402
    _extract_ca3ca3_vec, _rest_and_detect, _noise_label,
)

OUT = _REPO / "research" / "findings" / "raw" / "_gap5_sequence_replay_derisk.json"

# The CHAIN store == RANK 1's CLOSED bistable store (BTSP encode; the completing store), n_mem=3 DISJOINT assemblies.
SEQ_CFG = dict(
    n_ca3=2000, ca3_density=0.05, assembly_frac=0.12, encode_drive=3000.0, no_sync=True,
    bistable=True, nmda_recurrent=False, enable_ou=False, selective_inhib=True, structural_sep=1,
    plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=1.0, apical_gc_read=5.0,
    # BTSP encode (== RANK 1 / _gap4 BTSP_CFG):
    encode_btsp=True, encode_ca3w=0.5, encode_plateau_pA=250.0, btsp_lr=0.02, hebb_max=300.0,
    recall_k_thresh=40.0,
    # run()/_build defaults made explicit so _prepare_sequence reproduces the substrate faithfully:
    drive_steps=48, reset_steps=15, k_thresh=18.0, plateau_strength=120.0, coact_thresh=0.02,
    ca3_fb_inhib=20.0, apical_R=50.0, plateau_v_hold=-35.0, sel_inhib_spare=0.0, encode_btsp_hetero=0.0,
    # SEQUENCE encode schedule:
    n_mem=3, within_events=24, chain_fwd=24, chain_rev=8, seq_win_steps=16, chain_reset_steps=12,
)
GO_CFG = SEQ_CFG  # _rest_and_detect (imported) reads its module-global GO_CFG for the plateau_v_hold dendrite check;
                  # this module's GO_CFG has the same plateau_v_hold, so the imported function is unaffected.


# ----------------------------------------------------------------------------------------------------------------------
def _zero_elig(bridge):
    """Zero the BTSP presynaptic eligibility trace (cp_btsp_pre_elig, tau=1000ms) so eligibility from a PRIOR event's
    last assembly cannot leak into THIS event's first assembly (which would form a spurious wrap-around/reverse link).
    None until the btsp block has run at least one step; a no-op then (it starts at zero anyway)."""
    e = getattr(bridge, "cp_btsp_pre_elig", None)
    if e is not None:
        e[:] = 0.0
    es = getattr(bridge, "cp_btsp_pre_elig_slow", None)
    if es is not None:
        es[:] = 0.0


def _silence_soma_apical(bridge, settle=2):
    """Clear the SOMATIC + APICAL state (v/u/firing/conductances/v_apical) so the PREVIOUSLY-driven assembly is OFF
    before the next sweep window -- the biological inhibitory THETA reset that separates phase-precessed windows. Does
    NOT touch cp_btsp_pre_elig, so an earlier assembly keeps its pre-eligibility (carrying the FORWARD link) while its
    plateau is gone (so no B->A reverse link forms from a self-sustaining latch). Load-bearing for the asymmetry."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    if getattr(bridge, "cp_izh_c_reset", None) is not None:
        bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
    else:
        bridge.cp_membrane_potential_v[:] = -65.0
    bridge.cp_recovery_variable_u[:] = 0.0
    if getattr(bridge, "cp_firing_states", None) is not None:
        bridge.cp_firing_states[:] = False
    for _a in ("cp_conductance_g_nmda_recurrent", "cp_conductance_g_e", "cp_conductance_g_i",
               "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
               "cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
        _arr = getattr(bridge, _a, None)
        if _arr is not None:
            _arr[:] = 0.0
    if getattr(bridge, "cp_v_apical", None) is not None:
        bridge.cp_v_apical[:] = cp.float32(getattr(bridge.core_config, "apical_E_rest", -65.0))
    bridge.cp_external_input_current[:] = 0.0
    if getattr(bridge, "cp_bdsp_apical_drive", None) is not None:
        bridge.cp_bdsp_apical_drive[:] = 0.0
    for _ in range(settle):
        bridge._run_one_simulation_step()


def _prepare_sequence(seed, cfg, do_encode=True):
    """Build the CLOSED bistable store and BTSP-encode a SEQUENCE of n_mem DISJOINT assemblies as a directional chain
    (WITHIN each assembly + FORWARD A->B->C links via theta-compressed sweeps + a small REVERSE component). Returns a
    prep dict compatible with the RANK 1 _rest_and_detect (bridge, ca3_arr_host, assemblies_local, assembly_local,
    ca3_exc_local, within_flat) plus the sequence diagnostics (between-edge weights, per-assembly local positions)."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    n_ca3 = int(cfg["n_ca3"])
    _init_ca3w = float(cfg["encode_ca3w"]) if cfg.get("encode_btsp") else 6.0
    bridge = _build(seed, n_ca3=n_ca3, ca3w=_init_ca3w, ca3_density=cfg["ca3_density"],
                    coincidence=True, two_comp=True, nmda_recurrent=False, nmda_tau=100.0, nmda_ratio=1.0,
                    apical_R=cfg["apical_R"], apical_gc=cfg["apical_gc"], k_thresh=cfg["k_thresh"],
                    plateau_strength=cfg["plateau_strength"], train=True, hebb_max=cfg["hebb_max"], hebb_rate=True,
                    ca3_fb_inhib=cfg["ca3_fb_inhib"], coact_thresh=cfg["coact_thresh"], hebb_lr=None, enable_ou=False,
                    plateau_self_regen=cfg["plateau_self_regen"], plateau_v_hold=cfg["plateau_v_hold"],
                    apical_kir_g=cfg["apical_kir_g"], apical_gc_read=cfg["apical_gc_read"], ca1_ff_inhib=None,
                    enable_stp=cfg.get("enable_stp", False), mossy_stp_disabled=cfg.get("enable_stp", False))
    rm = bridge.region_manager
    ca3_idx = list(rm.indices("ca3"))
    ca3_pos = {int(g): i for i, g in enumerate(ca3_idx)}
    n_mem = int(cfg["n_mem"])
    n_assy = max(6, int(cfg["assembly_frac"] * n_ca3))
    # DISJOINT assemblies (share NO cells) -> unambiguous order (a shared cell would belong to two positions in the
    # sequence). Same seed draw family as RANK 1 (seed*17+3), drawn from ONE without-replacement pool.
    # --overlap-draw (diagnostic, default off): RANK 1's INDEPENDENT per-assembly draw (each drawn from the full pool ->
    # ~12% expected overlap at n_assy=240/n_ca3=2000). Isolates whether RANK 1's reactivation at n_mem>=2 rides on the
    # inter-assembly OVERLAP that the disjoint (order-preserving) draw removes.
    rng = np.random.default_rng(seed * 17 + 3)
    if cfg.get("overlap_draw"):
        assemblies = [np.asarray(sorted(rng.choice(ca3_idx, n_assy, replace=False)), dtype=np.int64) for _ in range(n_mem)]
    else:
        pool = rng.choice(ca3_idx, n_assy * n_mem, replace=False)
        assemblies = [np.asarray(sorted(pool[i * n_assy:(i + 1) * n_assy]), dtype=np.int64) for i in range(n_mem)]

    flat_h, pre_l_h, post_l_h = _extract_ca3ca3_vec(bridge, ca3_idx, to_host)
    conn = bridge.cp_connections

    # per-assembly local membership + a local-position -> assembly-index map (-1 = non-member)
    asm_of_local = np.full(len(ca3_idx), -1, dtype=np.int64)
    for m, a in enumerate(assemblies):
        asm_of_local[np.asarray([ca3_pos[int(g)] for g in a], dtype=np.int64)] = m

    # gap#5 forward-asymmetry (research gate 2026-07-23): flat CSR indices of the BETWEEN-assembly ca3->ca3 synapses.
    # The WITHIN-REFRESH's bistable/completing plateau spreads through the chain to the NEIGHBOR assembly and writes
    # ~137 SYMMETRIC between-links that swamp the ~6 p(asymmetric) BTSP forward bias. Freezing these during the refresh
    # (cp_plasticity_rate_gain[between_flat]=0) leaves the pure forward BTSP chain as the sole between-write. Computed
    # once here (indices into cp_connections.data, parallel to cp_plasticity_rate_gain); used only if freeze_between_refresh.
    _a_pre_all = asm_of_local[pre_l_h]; _a_post_all = asm_of_local[post_l_h]
    _between_mask = (_a_pre_all >= 0) & (_a_post_all >= 0) & (_a_pre_all != _a_post_all)
    between_flat = flat_h[_between_mask].astype(np.int64)

    # ------------------------------------------------------------------ ENCODE ------------------------------------
    _set_gates(bridge, 1.0)
    if do_encode and cfg.get("encode_btsp"):
        cfg_b = bridge.core_config
        cfg_b.enable_hebbian_learning = False
        cfg_b.enable_bdsp = True; cfg_b.bdsp_apical_bistable = True; cfg_b.bdsp_learning_rate = 0.0
        # BISTABLE plateau during the WITHIN encode (self_regen=plateau_self_regen, == RANK 1's proven _prepare): the
        # latching plateau accumulates IS_post across events -> a strong within-attractor that spontaneously reactivates.
        # (Switched to TRANSIENT=0 before the CHAIN phase below for clean asymmetric forward links; the earlier
        # self_regen=0-for-the-whole-encode starved the within-attractor -> 0 reactivation, the RANK 2 blocker.)
        # Restored to the recall value after encode.
        cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
        cfg_b.coincidence_plateau_v_hold = float(cfg["plateau_v_hold"]); cfg_b.apical_kir_g = float(cfg["apical_kir_g"])
        cfg_b.enable_btsp = True; cfg_b.btsp_learning_rate = float(cfg["btsp_lr"])
        cfg_b.btsp_elig_tau_ms = 1000.0; cfg_b.btsp_w_max = float(cfg["hebb_max"])
        cfg_b.btsp_hetero_dep = float(cfg["encode_btsp_hetero"])
        bridge.cp_bdsp_apical_drive = cp.zeros(int(cfg_b.num_neurons), dtype=cp.float32)
        encode_drive = float(cfg["encode_drive"]); plateau_pA = float(cfg["encode_plateau_pA"])
        reset_steps = int(cfg["reset_steps"]); drive_steps = int(cfg["drive_steps"])
        win = int(cfg["seq_win_steps"]); chain_reset = int(cfg["chain_reset_steps"])
        assy_dev = [cp.asarray(a, dtype=cp.int64) for a in assemblies]

        def _reset(nsteps):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_bdsp_apical_drive[:] = 0.0
            for _ in range(nsteps):
                bridge._run_one_simulation_step()

        def _drive_window(m, nsteps):
            """Co-fire assembly m (pre-eligibility) + plateau on m (IS_post) for nsteps -> BTSP potentiates whatever has
            pre-eligibility NOW (incl. earlier assemblies in the sweep, tau 1000ms) ONTO m."""
            ad = assy_dev[m]
            for _ in range(nsteps):
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[ad] = encode_drive
                bridge.cp_bdsp_apical_drive[:] = 0.0
                bridge.cp_bdsp_apical_drive[ad] = plateau_pA
                bridge._run_one_simulation_step()

        # WITHIN phase: each assembly a bistable-completable attractor (== RANK 1 single-assembly store, done n_mem x).
        # Eligibility is KEPT across an assembly's own events (== RANK 1 -> the within weights accumulate to the ~27
        # completion scale). Per-assembly the event loop (_reset + _drive_window) is BYTE-IDENTICAL to RANK 1's proven
        # _prepare loop (no silence between events). At the assembly BOUNDARY we state-reset the PREVIOUS assembly (clear
        # its latched plateau + eligibility) so it cannot leak into m+1's within-encode as a spurious cross-link -- but
        # with settle=0 (NO sim steps).
        # *** THE RANK-2 within-reactivation FIX (2026-07-22): the boundary silence originally ran settle=3 SIM STEPS.
        # Those steps -- not the state reset -- STARVE the bistable within-latch of the assembly that FOLLOWS (measured:
        # settle=3 -> w_within 5.0; settle=0 -> 27-30; == RANK 1 27.4). The plateau VALUE is cleared by the state reset
        # alone; the settling steps are unnecessary here and were the sole divergence from RANK 1's _prepare that gave
        # 0 rest-phase reactivation. The CHAIN phase keeps its own settle=2 theta reset (untouched, asym forward stands).
        rank1_encode = bool(cfg.get("rank1_encode"))   # diagnostic: EXACT RANK 1 loop (no boundary silence/zero_elig)
        for m in range(n_mem):
            if not rank1_encode:
                _silence_soma_apical(bridge, settle=0); _zero_elig(bridge)   # fresh start for assembly m (settle=0: no starving steps)
            else:
                bridge.cp_external_input_current[:] = 0.0; bridge.cp_bdsp_apical_drive[:] = 0.0   # == RANK 1's per-event zeroing only
            for _ev in range(int(cfg["within_events"])):
                _reset(reset_steps)
                _drive_window(m, drive_steps)
            bridge.cp_external_input_current[:] = 0.0
        if not rank1_encode:
            _silence_soma_apical(bridge, settle=0); _zero_elig(bridge)       # last assembly OFF before the chain phase (settle=0)
        # TRANSIENT plateau for the CHAIN phase (self_regen=0): the plateau tracks the drive so ONLY the currently-driven
        # assembly has IS_post -> clean asymmetric forward (a latched plateau would keep an earlier assembly's IS_post ON
        # into the later window -> reverse links). The within-attractors are already stored above (bistable plateau).
        cfg_b.coincidence_plateau_self_regen = 0.0

        # gap#5 causal-STDP chain (2026-07-23, default OFF = byte-identical BTSP path). The BTSP chain produces a
        # NEAR-SYMMETRIC store (R0/R1 proof: adj_fwd~=adj_rev; the numpy GO rode start=A). Swap the chain's operative
        # rule to ASYMMETRIC (Bi-Poo) STDP on the forward-swept A->B->C drive: A spikes, theta-reset, B spikes ~win ms
        # later -> the A->B synapse sees pre-before-post (Dt>0) -> LTP; the B->A synapse sees post-before-pre -> LTD.
        # => adj_fwd >> adj_rev by construction (Skaggs-McNaughton phase precession + causal STDP, Sato-Yamaguchi).
        # BTSP OFF so it doesn't overwrite with the symmetric store; STDP is the sole chain writer.
        _chain_rule = str(cfg.get("chain_rule", "btsp"))
        _stdp_saved = None
        if _chain_rule == "stdp":
            _stdp_saved = (cfg_b.enable_btsp, cfg_b.enable_stdp, cfg_b.stdp_a_plus, cfg_b.stdp_a_minus,
                           cfg_b.stdp_w_max, cfg_b.stdp_tau_plus_ms, cfg_b.stdp_tau_minus_ms)
            cfg_b.enable_btsp = False
            cfg_b.enable_stdp = True
            cfg_b.stdp_a_plus = float(cfg.get("stdp_a_plus", 0.08))    # boosted: few chain events must write meaningful fwd links
            cfg_b.stdp_a_minus = float(cfg.get("stdp_a_minus", 0.10))  # a_minus > a_plus -> net LTD on the reverse link (sharpen asym)
            cfg_b.stdp_w_max = float(cfg["hebb_max"])
            cfg_b.stdp_tau_plus_ms = float(cfg.get("stdp_tau", 20.0))
            cfg_b.stdp_tau_minus_ms = float(cfg.get("stdp_tau", 20.0))

        # CHAIN phase FORWARD sweeps: A -> B -> ... each a short window separated by a THETA RESET (silence soma+apical,
        # KEEP eligibility) so the earlier assembly's plateau is OFF when the later fires -> A->B forms (elig_A x plateau_B)
        # but B->A does NOT (A has no plateau). eligibility persists -> forward links.
        chain_edges = cfg.get("chain_edges")   # None = LINEAR order (default, byte-identical); else an explicit directed
                                               # edge list [(pre,post),...] for a BRANCHING topology (RANK 3 recombination).
        if chain_edges is None:
            order_fwd = list(range(n_mem))
            for _ev in range(int(cfg["chain_fwd"])):
                _zero_elig(bridge)              # start each sweep with NO carried eligibility (no wrap-around C->A)
                for m in order_fwd:
                    _silence_soma_apical(bridge, settle=2)   # theta reset: previous assembly OFF, eligibility kept
                    _drive_window(m, win)
                bridge.cp_external_input_current[:] = 0.0

            # CHAIN phase REVERSE sweeps (the symmetric component; fewer -> forward stays dominant): ... -> B -> A.
            order_rev = list(range(n_mem - 1, -1, -1))
            for _ev in range(int(cfg["chain_rev"])):
                _zero_elig(bridge)
                for m in order_rev:
                    _silence_soma_apical(bridge, settle=2)
                    _drive_window(m, win)
                bridge.cp_external_input_current[:] = 0.0
        else:
            # EXPLICIT-EDGE topology (RANK 3): form each directed edge (pre->post) independently. Each edge is its own
            # 2-node theta-separated sweep with eligibility zeroed BEFORE it -> ONLY pre->post potentiates (pre has elig
            # when post's plateau is up; post has no earlier elig for post->pre). A shared node B with edges (A,B),(B,C),
            # (X,B),(B,Y) becomes a branch point whose replay can RECOMBINE A->B->Y / X->B->C (never stored as a whole).
            for _ev in range(int(cfg["chain_fwd"])):
                for (pre, post) in chain_edges:
                    _zero_elig(bridge)
                    _silence_soma_apical(bridge, settle=2); _drive_window(pre, win)
                    _silence_soma_apical(bridge, settle=2); _drive_window(post, win)
                bridge.cp_external_input_current[:] = 0.0

        # gap#5 causal-STDP: restore BTSP (and the STDP knobs) after the chain phase so the WITHIN-REFRESH (which relies
        # on BTSP eligibility x plateau) runs its normal rule. Byte-identical when chain_rule != "stdp".
        if _stdp_saved is not None:
            (cfg_b.enable_btsp, cfg_b.enable_stdp, cfg_b.stdp_a_plus, cfg_b.stdp_a_minus,
             cfg_b.stdp_w_max, cfg_b.stdp_tau_plus_ms, cfg_b.stdp_tau_minus_ms) = _stdp_saved

        # WITHIN-REFRESH (2026-07-22 fix, default off): the CHAIN phase's transient-plateau + per-window theta silencing
        # ERODES the within-attractors the within-encode built (measured: w_within 15.2 -> 6.3 at n_mem=2 -> below the
        # reactivation threshold = the RANK 2 blocker). A post-chain bistable within-refresh RESTORES each within-attractor
        # WITHOUT touching the asymmetric cross-links: driving assembly m ALONE (bistable plateau) potentiates within-m
        # (pre=m eligible x plateau_m) but a cross-link m->k has post=k which is SILENT (no plateau) -> not potentiated,
        # not depressed (btsp_hetero_dep=0). So the forward chain survives while the within-basin is rebuilt.
        refresh = int(cfg.get("within_refresh", 0))
        if refresh > 0:
            cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])   # BISTABLE (latching) for the refresh
            _refresh_freeze_saved = None; _between_dev = None
            if cfg.get("freeze_between_refresh") and between_flat.size > 0:
                # gap#5 forward-asymmetry fix (research gate 2026-07-23, NO sim/ edit): FREEZE the between-assembly
                # synapses during the refresh so the bistable plateau's spread does NOT write the ~137 SYMMETRIC
                # between-link contaminant. cp_plasticity_rate_gain gates the BTSP write per-synapse; allocate ones if
                # absent, zero the between subset, restore after the refresh. Default OFF = byte-identical.
                if getattr(bridge, "cp_plasticity_rate_gain", None) is None:
                    bridge.cp_plasticity_rate_gain = cp.ones(int(bridge.cp_connections.nnz), dtype=cp.float32)
                _between_dev = cp.asarray(between_flat)
                _refresh_freeze_saved = bridge.cp_plasticity_rate_gain[_between_dev].copy()
                bridge.cp_plasticity_rate_gain[_between_dev] = 0.0
            # The refresh must build a REACTIVATABLE within-attractor -> use the SAME rank1_encode structure that Test C/D
            # proved reactivates (NO per-assembly boundary silence/zero_elig -- those are the blocker). Clear the CHAIN's
            # eligibility ONCE at the start so chain-elig does not corrupt the refresh potentiation, then a plain per-
            # assembly loop keeping eligibility (== RANK 1). Cross-links added here are SYMMETRIC; the chain's forward
            # asymmetry (asym +2.66) survives as long as it exceeds the symmetric refresh contribution.
            _zero_elig(bridge)
            for m in range(n_mem):
                if not rank1_encode:
                    _silence_soma_apical(bridge, settle=0); _zero_elig(bridge)
                else:
                    bridge.cp_external_input_current[:] = 0.0; bridge.cp_bdsp_apical_drive[:] = 0.0
                for _ev in range(refresh):
                    _reset(reset_steps)
                    _drive_window(m, drive_steps)
                bridge.cp_external_input_current[:] = 0.0
            if not rank1_encode:
                _silence_soma_apical(bridge, settle=0); _zero_elig(bridge)
            if _refresh_freeze_saved is not None:
                bridge.cp_plasticity_rate_gain[_between_dev] = _refresh_freeze_saved   # restore the between-synapse gate

        cfg_b.enable_bdsp = False; cfg_b.enable_btsp = False; bridge.cp_bdsp_apical_drive = None
        # restore the bistable plateau for recall/rest (completion latches)
        cfg_b.coincidence_plateau_self_regen = float(cfg["plateau_self_regen"])
    _set_gates(bridge, 0.0)

    # ------------------------------------------------------------ edge classification + diagnostics ---------------
    d = np.asarray(to_host(conn.data))
    a_pre = asm_of_local[pre_l_h]; a_post = asm_of_local[post_l_h]
    mem_pre = a_pre >= 0; mem_post = a_post >= 0
    within = mem_pre & mem_post & (a_pre == a_post)
    between = mem_pre & mem_post & (a_pre != a_post)
    within_flat = flat_h[within].astype(np.int64)
    between_flat = flat_h[between].astype(np.int64)
    w_within = float(np.mean(d[within_flat])) if within_flat.size else 0.0
    # forward = post-assembly index > pre-assembly index (A->B, A->C, B->C); reverse = <.
    fwd_mask = between & (a_post > a_pre)
    rev_mask = between & (a_post < a_pre)
    adj_fwd = between & (a_post == a_pre + 1)
    adj_rev = between & (a_post == a_pre - 1)
    w_forward = float(np.mean(d[flat_h[fwd_mask]])) if fwd_mask.any() else 0.0
    w_reverse = float(np.mean(d[flat_h[rev_mask]])) if rev_mask.any() else 0.0
    w_adj_fwd = float(np.mean(d[flat_h[adj_fwd]])) if adj_fwd.any() else 0.0
    w_adj_rev = float(np.mean(d[flat_h[adj_rev]])) if adj_rev.any() else 0.0

    # STRUCTURAL SEPARATION (structural_sep=1: zero true-outsider->member; PRESERVES inter-assembly member->member so
    # the directional chain survives -- verified: `within`/`between` are both member&member, only outsider->member zeroed)
    if int(cfg["structural_sep"]) >= 1:
        zsel = mem_post & (~mem_pre)
        if zsel.any():
            conn.data[cp.asarray(flat_h[zsel], dtype=cp.int64)] = cp.zeros(int(zsel.sum()), dtype=conn.data.dtype)

    if cfg.get("recall_k_thresh") is not None:
        bridge.core_config.coincidence_k_threshold = float(cfg["recall_k_thresh"])

    # ASSEMBLY-SELECTIVE INHIBITION (spare all members from the shared basket; RANK 1 approach on the member union)
    if cfg["selective_inhib"]:
        n_all = int(bridge.core_config.num_neurons)
        bask_bool = np.zeros(n_all, dtype=bool); bask_bool[np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)] = True
        assy_bool = np.zeros(n_all, dtype=bool); assy_bool[np.asarray(sorted(int(g) for a in assemblies for g in a), dtype=np.int64)] = True
        conn2 = bridge.cp_connections; nnz = int(conn2.nnz)
        indptr = np.asarray(to_host(conn2.indptr)); indices = np.asarray(to_host(conn2.indices))
        pre_of = np.searchsorted(indptr, np.arange(nnz), side="right") - 1
        spare = bask_bool[pre_of] & assy_bool[indices[:nnz]]
        if spare.any():
            conn2.data[cp.asarray(np.nonzero(spare)[0], dtype=cp.int64)] = cp.full(int(spare.sum()), float(cfg["sel_inhib_spare"]), dtype=conn2.data.dtype)

    assemblies_local = [np.asarray(sorted(ca3_pos[int(g)] for g in a), dtype=np.int64) for a in assemblies]
    assembly_local = np.asarray(sorted(ca3_pos[int(g)] for a in assemblies for g in a), dtype=np.int64)
    ca3_arr_host = np.asarray(ca3_idx, dtype=np.int64)
    try:
        ca3_inh = set(int(g) for g in rm.inhibitory_indices("ca3"))
    except Exception:
        ca3_inh = set()
    ca3_exc_local = np.asarray([i for i, g in enumerate(ca3_idx) if int(g) not in ca3_inh], dtype=np.int64)
    return dict(bridge=bridge, ca3_idx=ca3_idx, ca3_arr_host=ca3_arr_host, assemblies=assemblies,
                assemblies_local=assemblies_local, assembly_local=assembly_local, ca3_exc_local=ca3_exc_local,
                within_flat=within_flat, between_flat=between_flat, w_within=w_within,
                w_forward=w_forward, w_reverse=w_reverse, w_adj_fwd=w_adj_fwd, w_adj_rev=w_adj_rev,
                n_between_fwd=int(fwd_mask.sum()), n_between_rev=int(rev_mask.sum()), n_assy=n_assy)


def _scramble_between_weights(prep, seed):
    """SCRAMBLE inter-assembly weights: permute the learned BETWEEN-assembly recurrent edge weights AMONG themselves
    (same multiset, destroyed forward/reverse pairing) -> the DIRECTIONAL chain is gone, the within-assembly attractors
    + the between weight-budget preserved. => events still occur, but in NO particular order (the load-bearing control)."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    conn = prep["bridge"].cp_connections
    bf = prep["between_flat"]
    if len(bf) < 2:
        return 0
    d = np.asarray(to_host(conn.data))
    vals = d[bf].copy()
    np.random.default_rng(seed * 29 + 7).shuffle(vals)
    conn.data[cp.asarray(bf, dtype=cp.int64)] = cp.asarray(vals, dtype=conn.data.dtype)
    return int(len(bf))


# ----------------------------------------------------------------------------------------------------------------------
# SEQUENCE ORDER DETECTION: on the rest-phase firing F [T, n_ca3], detect discrete events (smoothed total CA3 co-firing
# crossing a robust threshold, == RANK 1 windowing), then within each event score the per-assembly ONSET order.
# ----------------------------------------------------------------------------------------------------------------------
def _smooth(x, W):
    return np.convolve(x.astype(float), np.ones(W), mode="same") if W > 1 else x.astype(float)


def _event_windows(F, W=5, ev_floor=0.4, ev_k=4.0, asize_ref=1):
    """Detect event windows from the smoothed TOTAL CA3 co-firing (unbiased: on ALL CA3, then classified per-assembly)."""
    T, _ = F.shape
    pop = F.sum(1).astype(float)
    S = _smooth(pop, W)
    med = float(np.median(S)); mad = float(np.median(np.abs(S - med))) * 1.4826
    thr = max(med + ev_k * mad, ev_floor * asize_ref)
    inev = S > thr
    events, t = [], 0
    while t < T:
        if inev[t]:
            s = t
            while t < T and inev[t]:
                t += 1
            events.append((s, min(t, T)))
        t += 1
    return events, float(inev.mean()), float(pop.mean() / max(1, F.shape[1]))


def _order_stat(onsets):
    """onsets: list of (asm_index, onset_time) for the ACTIVE assemblies (>=2). Returns dict(n, tau, fwd, rev) where
    fwd/rev = strictly monotonic forward/reverse in ASSEMBLY INDEX order, tau = Kendall rank correlation (index vs time)."""
    items = sorted(onsets, key=lambda kv: kv[0])   # by assembly index
    ts = [t for _, t in items]
    n = len(ts)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ts[i] < ts[j]:
                conc += 1
            elif ts[i] > ts[j]:
                disc += 1
    tot = conc + disc
    tau = (conc - disc) / tot if tot > 0 else 0.0
    fwd = all(ts[i] < ts[i + 1] for i in range(n - 1))
    rev = all(ts[i] > ts[i + 1] for i in range(n - 1))
    return dict(n=n, tau=tau, fwd=bool(fwd), rev=bool(rev))


def _detect_sequence_events(F, assemblies_local, W=5, ev_floor=0.4, ev_k=4.0, active_frac=0.12, onset_frac=0.08,
                            min_ev_len=4):
    """Detect ordered replay. For each event, per-assembly smoothed firing fraction over time; an assembly is ACTIVE if
    its peak fraction >= active_frac; its ONSET = first step its smoothed fraction crosses onset_frac (tie-break by peak
    time then center-of-mass). Score forward/reverse strict order + Kendall tau over multi-assembly (>=2 active) events."""
    T, nca3 = F.shape
    asizes = [max(1, len(a)) for a in assemblies_local]
    asize_ref = float(np.mean(asizes))
    events, duty, pop_rate = _event_windows(F, W=W, ev_floor=ev_floor, ev_k=ev_k, asize_ref=asize_ref)

    per_asm_active = [0] * len(assemblies_local)
    per_asm_peak = [[] for _ in assemblies_local]
    n_multi = n_full = 0
    fwd_multi = rev_multi = 0
    fwd_full = rev_full = 0
    taus = []
    chance_terms = []          # 1/n! expected forward-by-chance per multi event
    ev_records = []
    import math
    for (s, e) in events:
        if e - s < min_ev_len:
            continue
        active = []
        for k, A in enumerate(assemblies_local):
            a_t = _smooth(F[s:e][:, A].sum(1), W) / asizes[k]      # per-step active fraction of assembly k
            peak = float(a_t.max()) if a_t.size else 0.0
            per_asm_peak[k].append(peak)
            if peak >= active_frac:
                per_asm_active[k] += 1
                cross = np.nonzero(a_t >= onset_frac)[0]
                onset = float(cross[0]) if cross.size else float(np.argmax(a_t))
                ptime = float(np.argmax(a_t))
                com = float((np.arange(a_t.size) * a_t).sum() / max(a_t.sum(), 1e-9))
                active.append((k, onset, ptime, com, peak))
        if len(active) < 2:
            continue
        # order statistic on ONSET; break exact-onset ties with peak time, then center-of-mass
        onsets = [(k, onset + 1e-3 * ptime + 1e-6 * com) for (k, onset, ptime, com, _p) in active]
        st = _order_stat(onsets)
        if st is None:
            continue
        n_multi += 1
        taus.append(st["tau"])
        chance_terms.append(1.0 / math.factorial(st["n"]))
        if st["fwd"]:
            fwd_multi += 1
        if st["rev"]:
            rev_multi += 1
        if st["n"] == len(assemblies_local):
            n_full += 1
            if st["fwd"]:
                fwd_full += 1
            if st["rev"]:
                rev_full += 1
        ev_records.append(dict(span=[int(s), int(e)], n_active=st["n"], tau=round(st["tau"], 3),
                               fwd=st["fwd"], rev=st["rev"],
                               order=[int(k) for k, _ in sorted(onsets, key=lambda kv: kv[1])]))
    return dict(
        n_events=len(events), duty_cycle=duty, pop_rate=pop_rate,
        n_multi=n_multi, n_full=n_full,
        forward_frac=(fwd_multi / n_multi) if n_multi else 0.0,
        reverse_frac=(rev_multi / n_multi) if n_multi else 0.0,
        forward_frac_full=(fwd_full / n_full) if n_full else 0.0,
        reverse_frac_full=(rev_full / n_full) if n_full else 0.0,
        mean_tau=float(np.mean(taus)) if taus else 0.0,
        chance_forward=float(np.mean(chance_terms)) if chance_terms else 0.0,
        per_asm_active=per_asm_active,
        per_asm_peak=[float(np.mean(p)) if p else 0.0 for p in per_asm_peak],
        events=ev_records[:40],
    )


# ----------------------------------------------------------------------------------------------------------------------
def one_seed(seed, cfg, noise, rest_steps, W, ev_floor, ev_k, active_frac, onset_frac):
    t0 = time.time()
    out = {"seed": seed}

    # -- GO: encode the chain, run the rest phase, detect ordered replay --
    prep = _prepare_sequence(seed, cfg, do_encode=True)
    out["encode"] = dict(w_within=prep["w_within"], w_forward=prep["w_forward"], w_reverse=prep["w_reverse"],
                         w_adj_fwd=prep["w_adj_fwd"], w_adj_rev=prep["w_adj_rev"],
                         n_between_fwd=prep["n_between_fwd"], n_between_rev=prep["n_between_rev"],
                         assembly_sizes=[int(len(a)) for a in prep["assemblies"]])
    print(f"  [seed {seed}] ENCODE w_within={prep['w_within']:.1f} w_fwd={prep['w_forward']:.2f} "
          f"w_rev={prep['w_reverse']:.2f} (adj fwd={prep['w_adj_fwd']:.2f} rev={prep['w_adj_rev']:.2f}) "
          f"asym={prep['w_forward'] - prep['w_reverse']:+.2f} ({time.time()-t0:.0f}s)", flush=True)

    ev1, F = _rest_and_detect(prep, noise, rest_steps, seed, W=W, ev_floor=0.5, ev_k=ev_k, min_frac=0.30)
    seq = _detect_sequence_events(F, prep["assemblies_local"], W=W, ev_floor=ev_floor, ev_k=ev_k,
                                  active_frac=active_frac, onset_frac=onset_frac)
    out["go"] = {**seq, "duty_from_rank1": ev1["duty_cycle"], "weights_frozen": ev1["weights_frozen"],
                 "apical_rest_max": ev1["apical_rest_max"], "apical_n_latched": ev1["apical_n_latched"],
                 "noise": _noise_label(noise)}
    print(f"  [seed {seed}] GO {_noise_label(noise):>20}: events={seq['n_events']} multi={seq['n_multi']} "
          f"full3={seq['n_full']} | FWD={seq['forward_frac']:.3f} REV={seq['reverse_frac']:.3f} "
          f"(full3 fwd={seq['forward_frac_full']:.3f} rev={seq['reverse_frac_full']:.3f}) tau={seq['mean_tau']:+.3f} "
          f"chance={seq['chance_forward']:.3f} | duty={seq['duty_cycle']:.3f} pop={seq['pop_rate']:.4f} "
          f"asm_active={seq['per_asm_active']} frozen={ev1['weights_frozen']} "
          f"apic_latched={ev1['apical_n_latched']} ({time.time()-t0:.0f}s)", flush=True)

    # -- NO-NOISE (acid): must be SILENT (retires the self-sustaining artifact) --
    nn1, Fnn = _rest_and_detect(prep, ("none",), rest_steps, seed, W=W, ev_floor=0.5, ev_k=ev_k, min_frac=0.30)
    seq_nn = _detect_sequence_events(Fnn, prep["assemblies_local"], W=W, ev_floor=ev_floor, ev_k=ev_k,
                                     active_frac=active_frac, onset_frac=onset_frac)
    out["nonoise"] = {**seq_nn, "assembly_rest_frac": nn1["assembly_rest_frac"], "pop_rate": nn1["pop_rate"]}
    print(f"  [seed {seed}] NO-NOISE (acid): events={seq_nn['n_events']} multi={seq_nn['n_multi']} "
          f"pop={nn1['pop_rate']:.5f} assembly_rest={nn1['assembly_rest_frac']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # -- NO-ENCODE (fresh bridge, store skipped, same noise) -> no ordered events --
    prep_ne = _prepare_sequence(seed, cfg, do_encode=False)
    ne1, Fne = _rest_and_detect(prep_ne, noise, rest_steps, seed, W=W, ev_floor=0.5, ev_k=ev_k, min_frac=0.30)
    seq_ne = _detect_sequence_events(Fne, prep_ne["assemblies_local"], W=W, ev_floor=ev_floor, ev_k=ev_k,
                                     active_frac=active_frac, onset_frac=onset_frac)
    out["noencode"] = {**seq_ne, "w_within": prep_ne["w_within"], "pop_rate": ne1["pop_rate"]}
    print(f"  [seed {seed}] NO-ENCODE: events={seq_ne['n_events']} multi={seq_ne['n_multi']} "
          f"FWD={seq_ne['forward_frac']:.3f} REV={seq_ne['reverse_frac']:.3f} w_within={prep_ne['w_within']:.2f} "
          f"pop={ne1['pop_rate']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # -- SCRAMBLE inter-assembly weights (fresh encoded bridge, shuffle between, same noise) -> order collapses --
    prep_sc = _prepare_sequence(seed, cfg, do_encode=True)
    n_sc = _scramble_between_weights(prep_sc, seed)
    sc1, Fsc = _rest_and_detect(prep_sc, noise, rest_steps, seed, W=W, ev_floor=0.5, ev_k=ev_k, min_frac=0.30)
    seq_sc = _detect_sequence_events(Fsc, prep_sc["assemblies_local"], W=W, ev_floor=ev_floor, ev_k=ev_k,
                                     active_frac=active_frac, onset_frac=onset_frac)
    out["scramble"] = {**seq_sc, "n_between_shuffled": n_sc}
    print(f"  [seed {seed}] SCRAMBLE-BETWEEN ({n_sc} edges): events={seq_sc['n_events']} multi={seq_sc['n_multi']} "
          f"FWD={seq_sc['forward_frac']:.3f} REV={seq_sc['reverse_frac']:.3f} tau={seq_sc['mean_tau']:+.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -- PER-SEED VERDICT --
    go = out["go"]
    fwd = go["forward_frac"]; rev = go["reverse_frac"]; chance = max(go["chance_forward"], 1e-6)
    scr_fwd = out["scramble"]["forward_frac"]
    acid_silent = (nn1["assembly_rest_frac"] < 0.05 and seq_nn["n_multi"] == 0)
    frozen_ok = bool(ev1["weights_frozen"] and nn1["weights_frozen"] and sc1["weights_frozen"])
    dendrite_reset_ok = (ev1["apical_rest_max"] is None or ev1["apical_rest_max"] <= float(GO_CFG["plateau_v_hold"]) + 1e-3)
    enough_multi = (go["n_multi"] >= 4)
    forward_ordered = (fwd >= 2.0 * chance and fwd > rev and go["n_multi"] >= 4)
    scramble_collapses = (scr_fwd <= max(0.6 * fwd, 1.5 * chance) or out["scramble"]["n_multi"] == 0)
    noencode_retired = (seq_ne["n_multi"] == 0 or seq_ne["forward_frac"] <= 1.5 * chance)
    discrete = (go["duty_cycle"] <= 0.45)
    reverse_present = (rev >= 2.0 * chance)      # DIAGNOSTIC (symmetric component -> reverse replay), not required
    seed_go = bool(forward_ordered and discrete and acid_silent and frozen_ok and dendrite_reset_ok
                   and scramble_collapses and noencode_retired and enough_multi)
    out["checks"] = dict(forward_ordered=forward_ordered, discrete=discrete, acid_silent=acid_silent,
                         frozen_ok=frozen_ok, dendrite_reset_ok=dendrite_reset_ok,
                         scramble_collapses=scramble_collapses, noencode_retired=noencode_retired,
                         enough_multi=enough_multi, reverse_present=reverse_present)
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-ca3", type=int, default=2000)
    ap.add_argument("--n-mem", type=int, default=3, help="assemblies in the sequence (>=3)")
    ap.add_argument("--assembly-frac", type=float, default=0.12)
    ap.add_argument("--within-events", type=int, default=24)
    ap.add_argument("--chain-fwd", type=int, default=24)
    ap.add_argument("--chain-rev", type=int, default=8, help="reverse sweeps (symmetric component; 0 = forward-only)")
    ap.add_argument("--seq-win-steps", type=int, default=16)
    ap.add_argument("--noise", choices=["poisson", "ou"], default="poisson")
    ap.add_argument("--poisson-rate", type=float, default=0.015)
    ap.add_argument("--poisson-pa", type=float, default=1500.0)
    ap.add_argument("--poisson-dur", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=200.0)
    ap.add_argument("--rest-steps", type=int, default=1400)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ev-floor", type=float, default=0.4)
    ap.add_argument("--ev-k", type=float, default=4.0)
    ap.add_argument("--active-frac", type=float, default=0.12, help="per-assembly peak fraction to count as ACTIVE in an event")
    ap.add_argument("--onset-frac", type=float, default=0.08, help="per-assembly fraction for the ONSET crossing")
    ap.add_argument("--ca3-density", type=float, default=None, help="within-CA3 recurrent density (RANK 1 completion needs 0.35 for spontaneous reactivation; SEQ_CFG default 0.05 is too weak)")
    ap.add_argument("--structural-sep", type=int, default=None, help="basin isolation (RANK 1 completion uses 2; SEQ_CFG default 1)")
    ap.add_argument("--within-refresh", type=int, default=0, help="post-chain bistable within-refresh events per assembly (fix: chain erodes within 15.2->6.3; refresh rebuilds the basin, preserves the forward chain)")
    ap.add_argument("--overlap-draw", action="store_true", help="RANK 1-style INDEPENDENT (overlapping) assembly draw instead of DISJOINT (diagnostic: isolates whether reactivation rides on inter-assembly overlap)")
    ap.add_argument("--rank1-encode", action="store_true", help="diagnostic: EXACT RANK 1 within-encode loop (no per-assembly boundary silence/zero_elig; eligibility persists across assemblies)")
    ap.add_argument("--encode-only", action="store_true", help="FAST tuning: build+encode + report w_within/w_fwd/w_rev only (no rest phase)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cfg = dict(SEQ_CFG)
    cfg["n_ca3"] = int(a.n_ca3); cfg["n_mem"] = int(a.n_mem); cfg["assembly_frac"] = float(a.assembly_frac)
    cfg["within_events"] = int(a.within_events); cfg["chain_fwd"] = int(a.chain_fwd); cfg["chain_rev"] = int(a.chain_rev)
    cfg["seq_win_steps"] = int(a.seq_win_steps)
    if a.ca3_density is not None:
        cfg["ca3_density"] = float(a.ca3_density)
    if a.structural_sep is not None:
        cfg["structural_sep"] = int(a.structural_sep)
    cfg["within_refresh"] = int(a.within_refresh)
    cfg["overlap_draw"] = bool(a.overlap_draw)
    cfg["rank1_encode"] = bool(a.rank1_encode)
    noise = ("poisson", a.poisson_rate, a.poisson_pa, a.poisson_dur) if a.noise == "poisson" else ("ou", a.sigma)

    if a.encode_only:
        t0 = time.time()
        print(f"[gap5-seq ENCODE-ONLY] n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} within={cfg['within_events']} "
              f"chain_fwd={cfg['chain_fwd']} chain_rev={cfg['chain_rev']} win={cfg['seq_win_steps']} seeds={a.seeds}", flush=True)
        for s in a.seeds:
            prep = _prepare_sequence(s, cfg, do_encode=True)
            print(f"  [seed {s}] w_within={prep['w_within']:.1f} w_fwd={prep['w_forward']:.2f} w_rev={prep['w_reverse']:.2f} "
                  f"asym={prep['w_forward']-prep['w_reverse']:+.2f} | adj_fwd={prep['w_adj_fwd']:.2f} adj_rev={prep['w_adj_rev']:.2f} "
                  f"(n_fwd={prep['n_between_fwd']} n_rev={prep['n_between_rev']}) sizes={[len(x) for x in prep['assemblies']]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return 0

    print(f"[gap5-seq] n_ca3={cfg['n_ca3']} n_mem={cfg['n_mem']} assy~{max(6,int(cfg['assembly_frac']*cfg['n_ca3']))} "
          f"within={cfg['within_events']} chain_fwd={cfg['chain_fwd']} chain_rev={cfg['chain_rev']} "
          f"noise={_noise_label(noise)} rest_steps={a.rest_steps} seeds={a.seeds} "
          f"backend={os.environ.get('SIM_BACKEND','auto')}", flush=True)
    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, cfg, noise, a.rest_steps, a.window, a.ev_floor, a.ev_k, a.active_frac, a.onset_frac))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_go = sum(1 for p in per if p["seed_go"])
        go = n_go >= max(1, (len(per) + 1) // 2)
        mf = float(np.mean([p["go"]["forward_frac"] for p in per]))
        mr = float(np.mean([p["go"]["reverse_frac"] for p in per]))
        msc = float(np.mean([p["scramble"]["forward_frac"] for p in per]))
        mch = float(np.mean([p["go"]["chance_forward"] for p in per]))
        masym = float(np.mean([p["encode"]["w_forward"] - p["encode"]["w_reverse"] for p in per]))
        mnn = float(np.mean([p["nonoise"]["assembly_rest_frac"] for p in per]))
        verdict = (
            f"{'GO' if go else 'PARTIAL/NEGATIVE'} {n_go}/{len(per)} -- a stored 3-assembly chain "
            f"{'REPLAYS IN FORWARD ORDER' if go else 'did NOT cleanly replay in forward order'} under weak non-specific "
            f"background: forward_frac {mf:.3f} vs reverse {mr:.3f} vs SCRAMBLE {msc:.3f} vs chance {mch:.3f}; "
            f"encode asym w_fwd-w_rev {masym:+.2f}; NO-NOISE assembly rest {mnn:.4f} (acid: ~0). "
            + ("=> the SEQUENCE-REPLAY rung of the SWR loop de-risks on the bistable substrate; run the 6-seed GPU confirm."
               if go else "Per THE LAW: a stored-sequence forward-order rung is the genuine open question -- tune the "
               "chain vs within strength (chain_fwd/within_events), noise (poisson pa/rate), seq_win_steps, "
               "recall_k_thresh; a partial/negative on the traveling-wave hand-off is a real, honestly-reported result.")
        )
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"

    summary = {"probe": "gap5_sequence_replay", "GO": go, "n_go": n_go, "seeds": a.seeds,
               "n_ca3": cfg["n_ca3"], "n_mem": cfg["n_mem"], "noise": _noise_label(noise),
               "rest_steps": a.rest_steps, "within_events": cfg["within_events"], "chain_fwd": cfg["chain_fwd"],
               "chain_rev": cfg["chain_rev"], "elapsed_seconds": round(time.time() - t0, 1),
               "verdict": verdict, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 116 + f"\n[gap5-seq] VERDICT: {verdict}\n[gap5-seq] wrote {a.out}\n" + "=" * 116, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
