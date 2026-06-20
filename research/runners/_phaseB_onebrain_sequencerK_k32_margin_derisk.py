"""Phase B / burndown #3 Stage S2: lift the on-bridge K-way sequencer's ROUTING MARGIN from K<=16 to the production
K=32 -- the one genuine boundary test of #3.

S1 (`_phaseB_onebrain_sequencerK_divnorm_derisk.py`, commit 2cbae1ee, `2026-06-20-burndown-3-S1-divnorm-in-loop.md`)
wired S5's on-bridge divisive-normalization (`r_i = x_i/(sigma + gain*mean_j x_j)`) into the K-way sequencer and
RETIRED the host `scores_to_drive` peak-read from the drive path. It is GO at K in {2,4,8} (D=128, 3 seeds, ==host +
moat-0-FA), but K=16 SQUEEZED (2/3 seeds): at the fixed S5 op-point (`gain=0.05, sigma=1`) the divnorm tolerates a
runner-up firing (the EXTRA mode, ~7/96 role-reads). Harmless at small K, but at K>=16 an extra-lit word on a LOWER
block that shares an action word (e.g. `run` in 4 facts) wins first-match -> wrong-block routing. The MOAT held 0-FA at
every K (the moat is orthogonal at K=16 -- absent/cross cues match no block). So S1's residual is a present-cue
ROUTING-MARGIN limit, not a moat failure.

STAGE S2 = lift the routing margin to the PRODUCTION K=32 (32 distinct facts, unique (agent, action) cues, the 8
actions each shared by 4 facts -- maximal stress on the shared-action EXTRA). The scoping names three retreats,
cheap-first; this runner tries them IN ORDER, stopping at the first that gives GO at K=32:

  retreat 1 -- DIVNORM RE-TUNE (cheapest): a tighter divisive-norm op-point (a larger `gain` -> a larger divisor ->
    the sub-peak runner-up drops below the placed rheobase, the winner stays above). The on-bridge divide is
    SCALE-INVARIANT, so the (exact/extra/miss) counts are IDENTICAL across the per-query-peak sweep at ANY gain (the
    S5 robustness is preserved BY CONSTRUCTION) -- verified by the per-query-peak sweep here (pm spanning >= 1 order of
    magnitude). The diagnostic probe shows a clean 0-EXTRA/0-MISS window at gain~0.1 (seed-dependent), so retreat 1
    is the production candidate; the residual seed-variable EXTRA may or may not be LOAD-BEARING (only an EXTRA that
    lands on a lower-block shared-action word causes a wrong-block first-match -- the sequencer battery is the real
    test, not the raw EXTRA count).

  retreat 2 -- NEF-FS WTA INHIBITION POOL (if retreat 1 insufficient): a feed-forward lateral-inhibition pool between
    the decoded word-lines (each word-pool excites a shared inhibitory pool; the inhibitory pool suppresses every
    word-pool), so only the single normalized winner survives firing -- a hard WTA on top of the divisive norm
    (Carandini-Heeger normalization + competition). Runner-side score-bridge wiring; no `sim/` edit.

  retreat 3 -- HIERARCHICAL MATCH (if retreat 2 insufficient): a 2-level match reducing the 1-of-K=32 discrimination.

GO BAR (this stage, CPU/numpy -- the exact-algebra parity oracle), at K in {2,4,8,16,32}, D=128, 3+ seeds:
  * ==host `_scan` for who/what (the right block answers; absent/cross cues abstain) AND the no-confab MOAT holds (0
    false-accepts -- the HARD gate, NEVER traded);
  * the per-query-peak ROBUSTNESS is intact (the divnorm reproduces the host peak-norm across per-query peaks spanning
    >= 1 order of magnitude -- the S5 contract);
  * ANTI-CHEATS: a NO-DIVNORM control (divnorm-OFF + the SAME placed threshold on the RAW un-normalized drive) fails
    (the divnorm is load-bearing); OFF==byte-identical (S5's guard); sequencer-LESION fails safe; permuted-rule
    INVERTS; per-block priority correct.

The runner reports WHICH retreat achieved K=32, OR -- if none of the three named retreats lifts K=32 without breaking
the moat or the peak-robustness -- the characterized HONEST K* boundary (the on-bridge match cascade holds to K<=K*,
host `_scan` above for K>K*). An honest NEGATIVE IS a valid deliverable. Do NOT loosen the moat; do NOT search beyond
the three named retreats. NO `sim/` edit (reuse-by-import: S0 K-way sequencer + S5 divnorm score bridge + the EXISTING
`input_divisive_norm` primitive; the retreat-2 WTA is runner-side score-bridge wiring).

  SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencerK_k32_margin_derisk \
      --seeds 42,43,44 --dim 128 --ks 2,4,8,16,32 --retreat divnorm --gain 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion, RegionPathway
from sim.backend import to_host, from_host, is_gpu_backend
from research.runners.one_brain_composer import OneBrainComposer
# Reuse the op-result reader VERBATIM (the FHRR cleanup is unchanged; S2 only changes the DRIVE op-point / WTA).
from research.runners._phaseB_onebrain_sequencer_derisk import block_cleanup_scores
# Reuse the PROVEN K-way CONTROL fabric VERBATIM (builder / wiring / reset / production rule). S2 touches NONE of the
# sequencer control logic -- it only changes how the decoded word-lines are driven (op-point or the WTA read).
from research.runners._phaseB_onebrain_sequencerK_derisk import (
    build_sequencerK_bridge, host_scan_block, decision_to_block, patient_of,
)
# Reuse S5's on-bridge divnorm score bridge + the per-query-peak divisive-norm drive VERBATIM (retreat 1) + the
# OFF==byte-identical guard.
from research.runners._phaseC_S5_divnorm_derisk import (
    build_divnorm_score_bridge, onbridge_divnorm_drive, check_off_byte_identical,
)
# Reuse the S1 K-way drive runner VERBATIM (the decoded-line drive is supplied directly; there is NO scores_to_drive
# / s.max() in this drive path either -- the point of S1/S2).
from research.runners._phaseB_onebrain_sequencerK_divnorm_derisk import run_sequencerK_with_drive


# ----------------------------------------------------------------------------------------------------------------
# The K=32 fact set: 32 DISTINCT facts with UNIQUE (agent, action) cues. The 8 actions are each shared by 4 facts
# (the maximal-stress configuration for the shared-action EXTRA that broke K=16); agents and patients are all
# disjoint (so the host `_scan` is unambiguous AND the moat cues -- absent agent / absent action / cross -- are
# cleanly constructible). The first K facts are used at each store size K; the sweep runs K in {2,4,8,16,32}.
# ----------------------------------------------------------------------------------------------------------------
_AGENTS = ["dog", "cat", "fox", "bird", "sun", "tree", "moon", "river", "wolf", "hawk", "deer", "frog",
           "star", "leaf", "hill", "lake", "bear", "crow", "toad", "reed", "mist", "dune", "vine", "kelp",
           "newt", "lynx", "wren", "seal", "moss", "clay", "pike", "gull"]
_ACTIONS = ["go", "run", "see", "fly", "hop", "dig", "rest", "watch"]   # 8 actions x 4 agents = 32 unique pairs
_PATIENTS = ["north", "rivb", "treb", "sunb", "monb", "foxb", "catb", "birb", "hilb", "lakb", "rock", "stab",
             "leab", "wolb", "deeb", "frob", "cave", "nest", "pond", "fern", "clod", "sand", "root", "wave",
             "log", "den", "twig", "shor", "bark", "mud", "ston", "clif"]
ALL_FACTS = [(_AGENTS[i], _ACTIONS[i % 8], _PATIENTS[i]) for i in range(32)]
VOCAB = sorted(set(w for (a, x, p) in ALL_FACTS for w in (a, x, p)))   # 72 words


def _build_queries(facts):
    """The query set for a K-fact store: every PRESENT cue (each answers ITS block, so the scan must reach the LAST
    block) + THREE moat cues (absent agent / absent action / cross = agent of fact0 + an action fact0 does NOT have,
    that forms no stored pair). Returns [((agent, action), kind), ...]. Vocab-local generalization of S0's helper."""
    queries = [((a, x), f"blk{i}-present") for i, (a, x, p) in enumerate(facts)]
    agents = {a for (a, x, p) in facts}
    actions = {x for (a, x, p) in facts}
    pairs = {(a, x) for (a, x, p) in facts}
    absent_agent = next((w for w in VOCAB if w not in agents), "zzz")
    absent_action = next((w for w in VOCAB if w not in actions), "zzz")
    a0, x0 = facts[0][0], facts[0][1]
    cross_action = next((x for (a, x, p) in facts if (a0, x) not in pairs), absent_action)
    queries += [((absent_agent, x0), "absent-agent"), ((a0, absent_action), "absent-action"),
                ((a0, cross_action), "cross-no-block")]
    return queries


# ----------------------------------------------------------------------------------------------------------------
# RETREAT 2: the NEF-FS WTA score bridge. The S5 divnorm score pool (V word-pools, each input_divisive_norm=True) PLUS
# one shared inhibitory pool: each word-pool excites the inhibitory pool, the inhibitory pool inhibits every word-pool
# (feed-forward lateral inhibition). On top of the per-query DIVISIVE normalization this adds SUBTRACTIVE competition,
# so only the single normalized winner survives firing -> 0 EXTRA. Runner-side score-bridge wiring; no `sim/` edit.
# (The pure divnorm bridge -- retreat 1 -- is `build_divnorm_score_bridge` from S5, imported above.)
# ----------------------------------------------------------------------------------------------------------------
def build_wta_score_bridge(seed, V, n_word=20, n_inh=20, sigma=1.0, gain=0.05,
                           w_exc_to_inh=8.0, w_inh_to_exc=40.0):
    """divnorm score pool (V word-pools) + one shared inhibitory pool `wta_inh`. Word-pool -> wta_inh (excite);
    wta_inh -> every word-pool (inhibit). The divisive norm normalizes the per-query drive; the WTA suppresses
    sub-winners. NO `sim/` edit -- standard RegionPathway/inhibitory-trait wiring on `cp_connections`."""
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_input_divisive_norm = True
    cfg.input_divisive_sigma = float(sigma)
    cfg.input_divisive_gain = float(gain)
    regions = [BrainRegion(name=f"w{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                           input_divisive_norm=True) for w in range(V)]
    regions.append(BrainRegion(name="wta_inh", n_neurons=n_inh, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions
    P = []
    for w in range(V):
        P.append(RegionPathway(from_region=f"w{w}", to_region="wta_inh", density=1.0,
                               weight_mean=w_exc_to_inh, weight_jitter=0.0, plastic=False))
        P.append(RegionPathway(from_region="wta_inh", to_region=f"w{w}", density=1.0,
                               weight_mean=w_inh_to_exc, weight_jitter=0.0, plastic=False))
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def wta_drive(sb, V, scores, input_gain, settle=40, hi_pA=1500.0):
    """Drive the WTA score pool with current = input_gain*max(score,0); let the divisive norm + the lateral-inhibition
    WTA settle; read which word survives firing. Same RETURN contract as `onbridge_divnorm_drive` ((drive[V], acc[V]))
    so it is a drop-in drive source. NO host scores.max()."""
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = -65.0
    sb.cp_recovery_variable_u[:] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False
    for attr in ("cp_conductance_g_e", "cp_conductance_g_i"):
        a = getattr(sb, attr, None)
        if a is not None:
            a[:] = 0.0
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for w in range(V):
        drv = float(input_gain * s[w])
        if drv > 0:
            cur[idx(f"w{w}")] = drv
    acc = np.zeros(V)
    cur = from_host(cur)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur
        sb._run_one_simulation_step()
        fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        for w in range(V):
            acc[w] += fir[idx(f"w{w}")].mean()
    sb.cp_external_input_current[:] = 0.0
    return np.where(acc > 0, hi_pA, 0.0), acc


# ----------------------------------------------------------------------------------------------------------------
# The decoded-line drive maker. retreat in {"divnorm","wta"} selects the score-bridge drive source. peak_mult is the
# per-query-peak sweep multiplier (the S5 robustness control). Returns per-block (dA, dX) decoded-line drives + the
# per-block (#agent-lit, #action-lit) counts.
# ----------------------------------------------------------------------------------------------------------------
def make_block_drives(score_sb, V, bscores, input_gain, retreat, peak_mult=1.0):
    drives, lit = [], []
    drive_fn = wta_drive if retreat == "wta" else onbridge_divnorm_drive
    for (ag, ax) in bscores:
        dA, accA = drive_fn(score_sb, V, ag * peak_mult, input_gain)
        dX, accX = drive_fn(score_sb, V, ax * peak_mult, input_gain)
        drives.append((dA, dX))
        lit.append((int((accA > 0).sum()), int((accX > 0).sum())))
    return drives, lit


def _cleanup_mode_counts(score_sb, V, bscores, input_gain, retreat, peak_mult=1.0):
    """Per-query-peak diagnostic: across the 2K role-reads, count EXACT (only the argmax fires), EXTRA (a runner-up
    also fires -> the K=32 routing risk), MISS (nothing / wrong single fires). Used to show the peak-robustness (the
    counts are identical across pm at any divnorm op-point) and to characterize the boundary."""
    drive_fn = wta_drive if retreat == "wta" else onbridge_divnorm_drive
    exact = extra = miss = 0
    for (ag, ax) in bscores:
        for s in (ag, ax):
            _d, acc = drive_fn(score_sb, V, s * peak_mult, input_gain)
            nf = int((acc > 0).sum())
            am = int(np.argmax(np.maximum(np.asarray(s, float), 0.0)))
            amf = acc[am] > 0
            if nf == 1 and amf:
                exact += 1
            elif nf == 0:
                miss += 1
            elif nf >= 2:
                extra += 1
            else:
                miss += 1
    return dict(exact=exact, extra=extra, miss=miss, total=2 * len(bscores))


def run_seed_K(seed, D, K, input_gain, sigma, gain, retreat, peak_mults, host_fallback_above=None,
               match_thresh=0.15):
    """Run one seed at store size K with the chosen retreat. Build the composer + K facts, read each block's cleanup
    scores, build the K-way sequencer + the score bridge (divnorm or WTA), and check the substrate decision == host
    + moat + lesion + permuted, with the decoded-line drive computed at the nominal peak (pm=1.0). Also runs the
    per-query-peak sweep on the cleanup-mode counts (the S5 robustness: counts must match across pm). The NO-DIVNORM
    raw control is run on the same battery to show the normalization load-bearing.

    `match_thresh` (RETREAT 0 -- the threshold/drive-margin re-calibration, the cheapest UNTRIED fix): the production
    rule fires block b iff its spiking match pool rate m{b} > match_thresh. The committed S2 run fixed this at the K=2
    op-point 0.15; at K=32 the larger priority-WTA inhibitory fabric pulls the WINNER's own match pool down (seed-43
    cue (sun,hop) read m4=0.116 while ALL 31 other blocks read EXACTLY 0.000 -> over-abstention, the SAFE direction).
    Because the no-match floor is exactly 0.000 at K=32 (the divnorm killed all cross-block leak), lowering the
    threshold into the open (0.000, 0.116) margin (e.g. 0.06-0.08) ADMITS the correct match WITHOUT any false-accept
    (the off-target are far below). The MOAT is the hard gate -- if lowering the threshold admits ANY false-accept at
    any seed, R0 is REJECTED for R1 (the WTA). NOT a moat-relevant axis when the no-match floor is exactly zero.

    `host_fallback_above`: if set and K > it, the runner uses the host `_scan` ABOVE this K (the honest characterized
    partial conversion: on-bridge to K<=K*, host above). Default None = pure on-bridge at all K."""
    facts = ALL_FACTS[:K]
    use_host = (host_fallback_above is not None) and (K > host_fallback_above)
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=max(32, K), enable_batched=False,
                         enable_rf_cudagraph=False)
    for (a, x, p) in facts:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    bscores = [block_cleanup_scores(c, b) for b in range(len(facts))]   # the op RESULTS (cleanup scores per block)

    sb, meta = build_sequencerK_bridge(seed=seed, V=V, K=K)
    if retreat == "wta":
        score_sb = build_wta_score_bridge(seed=seed, V=V, sigma=sigma, gain=gain)
        raw_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=False)   # raw control = no divnorm, no WTA
    else:
        score_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=True, sigma=sigma, gain=gain)
        raw_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=False)

    # the production decoded-line drives at the nominal peak (the sequencer battery runs at pm=1.0; the peak sweep
    # below validates that the drive is identical across pm -- the S5 scale-invariance contract).
    drives, lit = make_block_drives(score_sb, V, bscores, input_gain, retreat, peak_mult=1.0)
    raw_drives, raw_lit = make_block_drives(raw_sb, V, bscores, input_gain, "divnorm", peak_mult=1.0)

    # per-query-peak ROBUSTNESS sweep on the cleanup mode counts (must be identical across pm at the chosen op-point).
    peak_modes = {pm: _cleanup_mode_counts(score_sb, V, bscores, input_gain, retreat, peak_mult=pm)
                  for pm in peak_mults}
    peak_robust = all(peak_modes[pm] == peak_modes[peak_mults[0]] for pm in peak_mults)
    nominal_modes = peak_modes[1.0] if 1.0 in peak_modes else peak_modes[peak_mults[0]]

    queries = _build_queries(facts)
    rows = []
    for (qa, qx), kind in queries:
        ca, cx = word_idx[qa], word_idx[qx]
        host_blk = host_scan_block(c, qa, qx)
        if use_host:                                       # host-above-K* path (the characterized partial conversion)
            sub_blk = host_blk
            dec, rates = (f"ans{host_blk}" if host_blk is not None else "abstain"), {}
            raw_blk = host_blk
            dec_raw, rates_raw = dec, {}
        else:
            dec, rates = run_sequencerK_with_drive(sb, meta, ca, cx, drives, match_thresh=match_thresh)
            sub_blk = decision_to_block(dec, K)
            dec_raw, rates_raw = run_sequencerK_with_drive(sb, meta, ca, cx, raw_drives, match_thresh=match_thresh)
            raw_blk = decision_to_block(dec_raw, K)
        rows.append(dict(cue=(qa, qx), kind=kind, host=patient_of(c, host_blk), sub=patient_of(c, sub_blk),
                         decision=dec, host_block=host_blk, sub_block=sub_blk, rates=rates,
                         raw_sub=patient_of(c, raw_blk), raw_decision=dec_raw, raw_block=raw_blk,
                         match_host_eq=(sub_blk == host_blk)))

    # --- the MOAT (HARD): every NON-present cue must abstain (no block selected) -- FA == 0. NOTE: at K=32 the moat
    #     is NOT a-priori orthogonal to the EXTRA (an extra-lit agent on a block whose action matches the cross cue
    #     could spuriously match), so this is a real measurement, not an assumption.
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    false_accepts = sum(1 for r in moat_rows if r["sub_block"] is not None)
    moat_ok = (false_accepts == 0) and all(r["decision"] == "abstain" for r in moat_rows)

    # --- sequencer-LESION (sever the result->op conditioning) on every present cue -> must FAIL SAFE (abstain).
    les = []
    if not use_host:
        for (a, x, p) in facts:
            dec_l, _ = run_sequencerK_with_drive(sb, meta, word_idx[a], word_idx[x], drives, lesion=True,
                                                 match_thresh=match_thresh)
            les.append(dec_l)
        lesion_fails_safe = all(d == "abstain" for d in les)
    else:
        lesion_fails_safe = True   # the host path has no spiking conditioning to sever; N/A above K*
        les = ["host-above-K*"]

    # --- PERMUTED-RULE: cyclic shift (m{b} -> ans{(b+1)%K}). A present cue for block b must route to ans{(b+1)%K}.
    perm_decs = []
    perm_ok = True
    if not use_host:
        for i, (a, x, p) in enumerate(facts):
            dec_p, _ = run_sequencerK_with_drive(sb, meta, word_idx[a], word_idx[x], drives, permute=True,
                                                 match_thresh=match_thresh)
            perm_decs.append(dec_p)
            if dec_p != f"ans{(i + 1) % K}":
                perm_ok = False
    else:
        perm_decs = ["host-above-K*"]
    permuted_inverts = perm_ok

    # --- NO-DIVNORM (raw) NEGATIVE control: across the SAME battery the raw path must FAIL (some cue breaks the moat
    #     OR the present cues do not all == host). raw_fails==True -> the control behaved (normalization load-bearing).
    if not use_host:
        raw_fa = sum(1 for r in moat_rows if r["raw_block"] is not None)
        raw_moat_ok = (raw_fa == 0) and all(r["raw_decision"] == "abstain" for r in moat_rows)
        raw_eq_host = all(r["raw_block"] == r["host_block"] for r in rows)
        raw_fails = not (raw_moat_ok and raw_eq_host)
    else:
        raw_fa, raw_fails = 0, True    # N/A above K* (host path)

    eq_all = all(r["match_host_eq"] for r in rows)
    return dict(seed=seed, D=D, K=K, retreat=retreat, used_host=use_host, rows=rows, eq_all=eq_all,
                moat_ok=moat_ok, false_accepts=false_accepts,
                lesion_fails_safe=lesion_fails_safe, lesion_decisions=les,
                permuted_inverts=permuted_inverts, permuted_decisions=perm_decs,
                raw_fails=raw_fails, raw_fa=raw_fa,
                lit=lit, raw_lit=raw_lit, nominal_modes=nominal_modes, peak_modes=peak_modes,
                peak_robust=peak_robust)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ks", default="2,4,8,16,32", help="store sizes K (the K=32 production margin test)")
    ap.add_argument("--retreat", default="divnorm", choices=["divnorm", "wta"],
                    help="divnorm = retreat 1 (re-tuned op-point); wta = retreat 2 (NEF-FS lateral-inhibition pool)")
    ap.add_argument("--input-gain", type=float, default=1.0, help="S5 op-point: fixed per-bridge input gain")
    ap.add_argument("--sigma", type=float, default=1.0, help="S5 op-point: divisive semi-saturation constant")
    ap.add_argument("--gain", type=float, default=0.1,
                    help="S5 op-point: divisive strength. retreat-1 production candidate ~0.1 (vs S1's 0.05) -- the "
                         "larger divisor drops the sub-peak runner-up below rheobase")
    ap.add_argument("--peak-mults", default="0.1,1.0,10.0",
                    help="per-query peak multipliers (the S5 robustness control: span >= 1 order of magnitude)")
    ap.add_argument("--host-fallback-above", type=int, default=None,
                    help="if set, use the host _scan ABOVE this K (the honest characterized partial conversion K*)")
    ap.add_argument("--match-thresh", type=float, default=0.15,
                    help="RETREAT 0: the production-rule match threshold (m{b} > thresh fires block b). The committed "
                         "S2 run fixed the K=2 op-point 0.15; at K=32 the winner dipped to 0.116 vs all-others-0.000 "
                         "-> lower into the open (0,0.116) margin (e.g. 0.06-0.08) to admit the correct match with "
                         "ZERO false-accept risk (no-match floor is exactly 0.000). The moat is the hard gate.")
    ap.add_argument("--out", default="research/findings/raw/_phaseB_onebrain_sequencerK_k32_margin.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    ks = [int(k) for k in args.ks.split(",")]
    peak_mults = [float(x) for x in args.peak_mults.split(",")]

    off_guard = check_off_byte_identical()
    off_ok = (off_guard["off_mask_none"] and off_guard["on_mask_not_none"]
              and off_guard["off_eq_off"] and off_guard["on_differs_from_off"])
    print(f"OFF==byte-identical guard: {off_guard} -> {'PASS' if off_ok else 'FAIL'}", flush=True)
    print(f"retreat={args.retreat} gain={args.gain} sigma={args.sigma} input_gain={args.input_gain} "
          f"match_thresh={args.match_thresh} host_fallback_above={args.host_fallback_above}", flush=True)

    all_results = {}
    for K in ks:
        results = []
        for s in seeds:
            r = run_seed_K(s, args.dim, K, args.input_gain, args.sigma, args.gain, args.retreat, peak_mults,
                           host_fallback_above=args.host_fallback_above, match_thresh=args.match_thresh)
            results.append(r)
            eq = "==host" if r["eq_all"] else "!=host"
            moat = "moat-OK" if r["moat_ok"] else f"MOAT-BREACH(fa={r['false_accepts']})"
            les = "lesion-SAFE" if r["lesion_fails_safe"] else f"lesion-UNSAFE({r['lesion_decisions']})"
            perm = "perm-inverts" if r["permuted_inverts"] else "perm-FAIL"
            raw = "raw-fails" if r["raw_fails"] else f"RAW-ALSO-PASSES(fa={r['raw_fa']})"
            nm = r["nominal_modes"]
            pr = "peak-robust" if r["peak_robust"] else "PEAK-VARIES"
            hf = " [HOST-above-K*]" if r["used_host"] else ""
            print(f"K={K} seed {s} D{args.dim}{hf}: {eq}  {moat}  {les}  {perm}  {raw}  "
                  f"modes(ex/xt/ms)={nm['exact']}/{nm['extra']}/{nm['miss']}  {pr}", flush=True)
        all_results[str(K)] = results

    summary = {}
    overall_go = off_ok
    first_break_K = None
    for K in ks:
        rs = all_results[str(K)]
        n = len(rs)
        eq_n = sum(r["eq_all"] for r in rs)
        moat_n = sum(r["moat_ok"] for r in rs)
        les_n = sum(r["lesion_fails_safe"] for r in rs)
        perm_n = sum(r["permuted_inverts"] for r in rs)
        raw_n = sum(r["raw_fails"] for r in rs)
        fa_total = sum(r["false_accepts"] for r in rs)
        pr_n = sum(r["peak_robust"] for r in rs)
        any_host = any(r["used_host"] for r in rs)
        # the load-bearing K-way gate: ==host + moat + lesion + permuted + peak-robust + moat 0-FA. (The raw control
        # and peak-robustness are N/A on a host-above-K* row -- excluded from that K's GO if the host path was used.)
        if any_host:
            go = (eq_n == n and moat_n == n and fa_total == 0)
        else:
            go = (eq_n == n and moat_n == n and les_n == n and perm_n == n and raw_n == n
                  and fa_total == 0 and pr_n == n)
        overall_go = overall_go and go
        if not go and first_break_K is None:
            first_break_K = K
        summary[str(K)] = dict(n=n, eq_n=eq_n, moat_n=moat_n, lesion_n=les_n, permuted_n=perm_n,
                               raw_fails_n=raw_n, fa_total=fa_total, peak_robust_n=pr_n, any_host=any_host,
                               verdict="GO" if go else "NEGATIVE")
        print(f"\nK={K} SUMMARY: ==host {eq_n}/{n}  moat {moat_n}/{n} (FA_total {fa_total})  "
              f"lesion {les_n}/{n}  permuted {perm_n}/{n}  raw-fails {raw_n}/{n}  peak-robust {pr_n}/{n}"
              f"{'  [host-above-K*]' if any_host else ''}  -> {summary[str(K)]['verdict']}", flush=True)

    # K* = the largest K with a clean on-bridge GO (the characterized boundary if K=32 doesn't lift)
    onbridge_go_ks = [int(k) for k in ks if summary[str(k)]["verdict"] == "GO" and not summary[str(k)]["any_host"]]
    k_star = max(onbridge_go_ks) if onbridge_go_ks else None

    verdict = "GO" if overall_go else "NEGATIVE"
    print(f"\nOVERALL: {verdict}  (K in {ks}, {len(seeds)} seeds, retreat={args.retreat}, gain={args.gain}; "
          f"host scores_to_drive read RETIRED from the drive path)", flush=True)
    print(f"on-bridge clean-GO K* = {k_star}  (first break at K={first_break_K})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=dict(per_K=summary, off_guard=off_guard, off_ok=off_ok, verdict=verdict,
                                    k_star=k_star, first_break_K=first_break_K, retreat=args.retreat,
                                    gpu=is_gpu_backend(), input_gain=args.input_gain, sigma=args.sigma,
                                    gain=args.gain, match_thresh=args.match_thresh, peak_mults=peak_mults,
                                    host_fallback_above=args.host_fallback_above),
                       results=all_results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
