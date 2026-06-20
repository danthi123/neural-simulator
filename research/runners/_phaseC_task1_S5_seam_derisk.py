"""Phase C — TASK 1 (THE NOVEL SEAM, S5): the result->sequencer coupling on-substrate (option a), in ISOLATION.

Where the arc's novel claim lives or dies (`2026-06-19-tier2-phaseC-integrated-loop-design.md` §2.3, §4.3, §5
Task 1, §6). In Phase B the cleanup RESULT was read to host (`block_cleanup_scores` -> `scores_to_drive`) and the
decoded-word-line DRIVE was an external current on a SEPARATE Izhikevich sequencer bridge. That is a host DATA
round-trip in the middle of the loop. Option (a) asks: can a FIXED ON-BRIDGE projection convert the cleanup's
graded membrane result into the SAME decoded-word-line drive -- WITHOUT a host read of the scores -- cleanly
enough that the gated-match cascade stays decisive (true match >= ~0.22, no-match <= ~0.10) and the moat holds?

THE STRUCTURAL QUESTION (the §6 risk made concrete): the cleanup result is RF state `Re(c)` on
`cp_membrane_potential_v` -- a GRADED matched-filter score, winner ~peak, runner-up ~0.4*peak (so the
discrimination is RELATIVE). To drive an Izhikevich decoded line through `cp_connections` (the only host-read-free
route) the cleanup neuron must FIRE; but on the composer's Izhikevich-model bridge a membrane at the cleanup
score (~1e5-1e6) is FAR above the Izhikevich threshold (~+30mV), so the winner AND the runner-up AND every
off-target with positive Re all fire identically -- a binary spike that DESTROYS the relative magnitude the
selection needs. The peak-normalization `scores_to_drive`/`_spiking_cleanup` perform is a HOST op; a fixed
on-bridge `cp_connections` projection cannot express it.

This runner DEMONSTRATES that, not just argues it. For each block it:
  - reads the REAL cleanup membrane (the validated `_read_block` op on the production OneBrainComposer);
  - OPTION (b) baseline: Phase B's exact host coupling (`block_cleanup_scores` -> `scores_to_drive`) -> the
    decoded drive -> Phase B's sequencer -> the match cascade (m0/m1). The known-GO control.
  - OPTION (a) ATTEMPT: a fixed ON-BRIDGE projection -- the cleanup neurons (seeded with the real cleanup v) are
    stepped as Izhikevich and project through a fixed `cp_connections` route to the decoded lines (NO host read of
    the scores). Measure the decoded-line firing it produces, feed THAT into the SAME sequencer, read the match
    cascade. If the firing is the whole role's row (winner + runner-up + off-target all fire) the match washes out
    (true ~ no-match, OR the moat breaks) -- the honest negative.

GREEN (option a): the on-bridge decoded drive reproduces the Phase-B host-driven pattern closely enough that the
match stays decisive (true >= ~0.22, no-match <= ~0.10) AND the moat holds, 3 seeds; lesion (sever the
projection) -> decoded lines silent -> abstain (fails safe).
HONEST NEGATIVE (option a walls): the fixed projection washes the match out -> record the seam boundary; the loop
proceeds with the host result-read (option b), the CONTROL still on-substrate (Phase B's GO). NOT a softer moat.

NO `sim/` edit (reuse-by-import: OneBrainComposer + Phase B's build_sequencer_bridge/run_sequencer + a fixed
cp_connections projection on a plain Izhikevich bridge). 3 seeds (the match-cascade parity, the §6 rule).

  SIM_BACKEND=cupy python -u -m research.runners._phaseC_task1_S5_seam_derisk --seeds 42,43,44 --dim 64
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
from research.runners._phaseB_onebrain_sequencer_derisk import (
    block_cleanup_scores, scores_to_drive, build_sequencer_bridge, run_sequencer,
    decision_to_patient, host_scan, FACTS, VOCAB,
)


# ----------------------------------------------------------------------------------------------------------------
# OPTION (a) on-bridge realization: a FIXED projection cleanup-neuron -> decoded-word-line, NO host read of scores.
# The cleanup neurons hold Re(c) on cp_membrane_potential_v. We build a plain Izhikevich bridge with V "cleanup"
# source neurons (per role) + V "decoded" target neurons (per role), a fixed 1:1 identity projection cleanup_w ->
# decoded_w on cp_connections, SEED the cleanup neurons' membrane with the REAL cleanup v (the only host touch is
# copying the bridge's own membrane state from the composer to the co-resident slice -- this is the co-residence
# the full Task 2 loop makes a single bridge; here it isolates the COUPLING), step the Izhikevich dynamics, and
# read which decoded lines FIRE. This is exactly "drive the decoded line through a fixed cp_connections projection
# from the cleanup membrane" -- the option-(a) mechanism. If the membrane is suprathreshold for winner AND
# runner-up, the projection passes BOTH -> the decoded drive is non-decisive.
# ----------------------------------------------------------------------------------------------------------------
def build_onbridge_projection(seed, V, n_word=20, w_proj=300.0, n_role=2):
    """A plain Izhikevich bridge: for each role r in [0,n_role) -- cleanup source pool `srcA/srcX` (V word-pools) +
    decoded target pool `decA/decX` (V word-pools), with a FIXED 1:1 `src_{r}{w} -> dec_{r}{w}` projection on
    cp_connections. The cleanup membrane is copied into the src pools; one+ Izhikevich step propagates a spike to
    the matching dec pool IFF the src crossed threshold. Reads the dec firing = the on-bridge decoded-line drive."""
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
    roles = ["A", "X"][:n_role]
    regions = []
    for r in roles:
        regions += [BrainRegion(name=f"src{r}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
        regions += [BrainRegion(name=f"dec{r}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
    cfg.brain_regions = regions
    P = []
    for r in roles:
        for w in range(V):
            P.append(RegionPathway(from_region=f"src{r}_{w}", to_region=f"dec{r}_{w}", density=1.0,
                                   weight_mean=w_proj, weight_jitter=0.0, plastic=False))
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb, roles


def onbridge_decoded_drive(proj_sb, roles, V, agent_scores, action_scores, settle=8, hi_pA=1500.0):
    """OPTION (a): seed the src word-pools with the REAL cleanup membrane (Re(c)) for agent+action, step the
    Izhikevich projection, and read which DECODED pools fire. Returns (dA_drive, dX_drive) as per-word hi_pA/0
    vectors (the on-bridge analogue of `scores_to_drive`, but produced by the SUBSTRATE projection -- NO host
    threshold-vs-peak of the scores). A decoded word is 'driven' iff its dec pool fired during the settle."""
    idx = lambda nm: np.asarray(proj_sb.region_manager.indices(nm))
    # reset to resting, then SEED the src membrane with the cleanup score (the cleanup neuron's own v). This copy is
    # the co-residence the full loop dissolves; the COMPARISON/threshold is NOT host-computed -- the substrate is.
    if getattr(proj_sb, "cp_izh_c_reset", None) is not None:
        proj_sb.cp_membrane_potential_v[:] = proj_sb.cp_izh_c_reset
    else:
        proj_sb.cp_membrane_potential_v[:] = -65.0
    proj_sb.cp_recovery_variable_u[:] = 0.0
    if getattr(proj_sb, "cp_firing_states", None) is not None:
        proj_sb.cp_firing_states[:] = False
    import sim.backend as _b
    xp, _ = _b.get_backend()
    for (r, sc) in (("A", agent_scores), ("X", action_scores)):
        if r not in roles:
            continue
        for w in range(V):
            proj_sb.cp_membrane_potential_v[idx(f"src{r}_{w}")] = float(sc[w])   # the REAL cleanup v on the src pool
    accA = np.zeros(V); accX = np.zeros(V)
    for _ in range(settle):
        proj_sb._run_one_simulation_step()
        fir = np.asarray(to_host(proj_sb.cp_firing_states)).astype(float)
        for w in range(V):
            if "A" in roles:
                accA[w] += fir[idx(f"decA_{w}")].mean()
            if "X" in roles:
                accX[w] += fir[idx(f"decX_{w}")].mean()
    dA = np.where(accA > 0, hi_pA, 0.0)
    dX = np.where(accX > 0, hi_pA, 0.0)
    return dA, dX, accA, accX


# ----------------------------------------------------------------------------------------------------------------
# OPTION (a2) -- the MOST CHARITABLE host-read-free fixed projection: the cleanup score drives the src pool as a
# proportional CURRENT (a fixed per-bridge gain, NOT per-query), the src pools laterally inhibit each other (a WTA
# driver pool -- "a small driver pool whose firing gates the decoded line", §2.3), and the winner's firing
# propagates to its decoded line. This is the fairest fixed-projection realization (graded rate -> WTA), and tests
# whether even WTA over a fixed-gain current can select the winner when BOTH winner + runner-up are above rheobase
# WITHOUT a per-query peak-normalization (which would be a host op). The fixed gain is calibrated ONCE so a typical
# peak lands at a strong drive; the crux it exposes: the cleanup peak VARIES per block/query, so one fixed gain
# can't normalize, and a 2.5x current ratio over rheobase doesn't cleanly separate via lateral inhibition.
# ----------------------------------------------------------------------------------------------------------------
def build_onbridge_wta_projection(seed, V, n_word=20, w_proj=300.0, w_wta_inhib=180.0, n_role=2):
    """Like build_onbridge_projection but the src word-pools (per role) form a WTA via a shared inhibitory pool
    `winh{r}` (each src excites winh, winh inhibits all src) -- so the highest-current src wins the firing race;
    src{r}{w} -> dec{r}{w} forwards the winner's spikes. The cleanup score drives src as CURRENT (not seeded v)."""
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
    roles = ["A", "X"][:n_role]
    regions = []
    for r in roles:
        regions += [BrainRegion(name=f"src{r}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
        regions += [BrainRegion(name=f"dec{r}_{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0)
                    for w in range(V)]
        regions.append(BrainRegion(name=f"winh{r}", n_neurons=n_word, exc_fraction=0.0, internal_density=0.0))
    cfg.brain_regions = regions
    P = []
    for r in roles:
        for w in range(V):
            P.append(RegionPathway(from_region=f"src{r}_{w}", to_region=f"dec{r}_{w}", density=1.0,
                                   weight_mean=w_proj, weight_jitter=0.0, plastic=False))
            P.append(RegionPathway(from_region=f"src{r}_{w}", to_region=f"winh{r}", density=1.0,
                                   weight_mean=w_wta_inhib, weight_jitter=0.0, plastic=False))   # src -> shared inh
            P.append(RegionPathway(from_region=f"winh{r}", to_region=f"src{r}_{w}", density=1.0,
                                   weight_mean=w_wta_inhib, weight_jitter=0.0, plastic=False))   # inh -| all src (WTA)
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb, roles


def onbridge_wta_decoded_drive(proj_sb, roles, V, agent_scores, action_scores, gain, drive_cap_pA=1200.0,
                               settle=30, hi_pA=1500.0):
    """OPTION (a2): drive each src pool with current = clip(gain * score, 0, cap) (the FIXED-gain projection, no
    per-query peak-normalization), settle the WTA, read which DECODED pools fired. The winner SHOULD dominate IF a
    fixed gain + WTA can separate; the negative is if the runner-up also fires (a 2.5x ratio over rheobase) OR an
    off-target block's wrong-scaled drive lights spuriously (the moat leak)."""
    idx = lambda nm: np.asarray(proj_sb.region_manager.indices(nm))
    if getattr(proj_sb, "cp_izh_c_reset", None) is not None:
        proj_sb.cp_membrane_potential_v[:] = proj_sb.cp_izh_c_reset
    else:
        proj_sb.cp_membrane_potential_v[:] = -65.0
    proj_sb.cp_recovery_variable_u[:] = 0.0
    if getattr(proj_sb, "cp_firing_states", None) is not None:
        proj_sb.cp_firing_states[:] = False
    cur = np.zeros(proj_sb.core_config.num_neurons, dtype=np.float64)
    for (r, sc) in (("A", agent_scores), ("X", action_scores)):
        if r not in roles:
            continue
        for w in range(V):
            drv = float(np.clip(gain * max(float(sc[w]), 0.0), 0.0, drive_cap_pA))
            if drv > 0:
                cur[idx(f"src{r}_{w}")] = drv
    accA = np.zeros(V); accX = np.zeros(V)
    cur = from_host(cur)                                 # match bridge backend (numpy build -> cupy under SIM_BACKEND=cupy)
    for _ in range(settle):
        proj_sb.cp_external_input_current[:] = cur
        proj_sb._run_one_simulation_step()
        fir = np.asarray(to_host(proj_sb.cp_firing_states)).astype(float)
        for w in range(V):
            if "A" in roles:
                accA[w] += fir[idx(f"decA_{w}")].mean()
            if "X" in roles:
                accX[w] += fir[idx(f"decX_{w}")].mean()
    proj_sb.cp_external_input_current[:] = 0.0
    dA = np.where(accA > 0, hi_pA, 0.0)
    dX = np.where(accX > 0, hi_pA, 0.0)
    return dA, dX, accA, accX


def run_sequencer_with_drive(sb, meta, cue_a, cue_x, drives, settle=60, match_thresh=0.15):
    """Run Phase B's sequencer but with the decoded-line drive supplied DIRECTLY (the option-(a) on-bridge drive),
    instead of computed from `scores_to_drive` inside run_sequencer. `drives` = [(dA,dX), ...] per block."""
    V = meta["V"]
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
    from research.runners._phaseB_onebrain_sequencer_derisk import reset_sequencer_state
    reset_sequencer_state(sb)
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur[idx(f"cueA_{cue_a}")] = 1500.0
    cur[idx(f"cueX_{cue_x}")] = 1500.0
    for bi, (dA, dX) in enumerate(drives[:2]):
        for w in range(V):
            if dA[w] > 0:
                cur[idx(f"d{bi}A_{w}")] = dA[w]
            if dX[w] > 0:
                cur[idx(f"d{bi}X_{w}")] = dX[w]
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur = from_host(cur)                                 # match bridge backend (numpy build -> cupy under SIM_BACKEND=cupy)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur
        sb._run_one_simulation_step()
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    sb.cp_external_input_current[:] = 0.0
    m0 = acc[idx("m0")].mean() / settle
    m1 = acc[idx("m1")].mean() / settle
    f0, f1 = (m0 > match_thresh), (m1 > match_thresh)
    rule = {(True, True): "ans0", (True, False): "ans0", (False, True): "ans1", (False, False): "abstain"}
    return rule[(f0, f1)], {"m0": round(m0, 3), "m1": round(m1, 3), "f0": f0, "f1": f1}


def _gates(rows, sub_key, decision_key, rates_key):
    """==host (all rows) + moat (every absent/cross cue abstains) + decisive (true-present m >= 0.20)."""
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    moat_ok = all(r[sub_key] is None for r in moat_rows)
    eq_host = all(r[sub_key] == r["host"] for r in rows)
    present = [r for r in rows if r["kind"].endswith("present")]
    true_m = [max(r[rates_key]["m0"], r[rates_key]["m1"]) for r in present]
    decisive = all(m >= 0.20 for m in true_m)
    return eq_host, moat_ok, decisive, true_m


def run_seed(seed, D):
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in FACTS:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    blocks = list(range(len(FACTS)))
    bscores = [block_cleanup_scores(c, b) for b in blocks]            # the REAL cleanup op results per block

    seq_sb, meta = build_sequencer_bridge(seed=seed, V=V)            # Phase B's sequencer (the CONTROL, GO)
    proj_sb, roles = build_onbridge_projection(seed=seed, V=V)       # option (a1) raw-membrane projection
    wta_sb, wta_roles = build_onbridge_wta_projection(seed=seed, V=V)   # option (a2) rescaled-current + WTA driver
    # calibrate the FIXED gain ONCE (not per-query): target block-0's agent peak -> ~900 pA drive (a strong rheobase)
    peak0 = float(np.maximum(bscores[0][0], 0.0).max())
    gain = (900.0 / peak0) if peak0 > 0 else 1.0

    queries = [(("dog", "go"), "blk0-present"), (("cat", "run"), "blk1-present"),
               (("fox", "go"), "absent-agent"), (("dog", "see"), "absent-action"),
               (("dog", "run"), "cross-no-block")]
    rows = []
    for (qa, qx), kind in queries:
        ca, cx = word_idx[qa], word_idx[qx]
        host = host_scan(c, qa, qx)
        # OPTION (b) baseline (Phase B host coupling): scores_to_drive inside run_sequencer (the known GO control)
        dec_b, rates_b = run_sequencer(seq_sb, meta, ca, cx, bscores)
        sub_b = decision_to_patient(c, dec_b, blocks)
        # OPTION (a1) raw-membrane projection: the on-bridge projection produces the decoded drive (no host scores->drive)
        drives_a1, lit_a1 = [], []
        for (ag, ax) in bscores:
            dA, dX, accA, accX = onbridge_decoded_drive(proj_sb, roles, V, ag, ax)
            drives_a1.append((dA, dX)); lit_a1.append((int((accA > 0).sum()), int((accX > 0).sum())))
        dec_a1, rates_a1 = run_sequencer_with_drive(seq_sb, meta, ca, cx, drives_a1)
        sub_a1 = decision_to_patient(c, dec_a1, blocks)
        # OPTION (a2) rescaled-current + WTA driver (the most charitable fixed projection)
        drives_a2, lit_a2 = [], []
        for (ag, ax) in bscores:
            dA, dX, accA, accX = onbridge_wta_decoded_drive(wta_sb, wta_roles, V, ag, ax, gain)
            drives_a2.append((dA, dX)); lit_a2.append((int((accA > 0).sum()), int((accX > 0).sum())))
        dec_a2, rates_a2 = run_sequencer_with_drive(seq_sb, meta, ca, cx, drives_a2)
        sub_a2 = decision_to_patient(c, dec_a2, blocks)
        rows.append(dict(cue=(qa, qx), kind=kind, host=host,
                         optB_sub=sub_b, optB_decision=dec_b, optB_rates=rates_b,
                         optA1_sub=sub_a1, optA1_decision=dec_a1, optA1_rates=rates_a1, optA1_decoded_lit=lit_a1,
                         optA2_sub=sub_a2, optA2_decision=dec_a2, optA2_rates=rates_a2, optA2_decoded_lit=lit_a2))
    optB_eq_host, optB_moat_ok, optB_decisive, optB_true_m = _gates(rows, "optB_sub", "optB_decision", "optB_rates")
    optA1_eq_host, optA1_moat_ok, optA1_decisive, optA1_true_m = _gates(rows, "optA1_sub", "optA1_decision", "optA1_rates")
    optA2_eq_host, optA2_moat_ok, optA2_decisive, optA2_true_m = _gates(rows, "optA2_sub", "optA2_decision", "optA2_rates")

    # LESION (sever the projection): on a present cue, the decoded lines get ZERO drive -> must abstain (fail safe)
    les = []
    for (qa, qx) in (("dog", "go"), ("cat", "run")):
        zero = [(np.zeros(V), np.zeros(V)) for _ in bscores]
        dec_l, _ = run_sequencer_with_drive(seq_sb, meta, word_idx[qa], word_idx[qx], zero)
        les.append(dec_l)
    lesion_fails_safe = all(d == "abstain" for d in les)

    return dict(seed=seed, D=D, rows=rows, gain=gain, peak0=peak0,
                optB_eq_host=optB_eq_host, optB_moat_ok=optB_moat_ok, optB_decisive=optB_decisive, optB_true_m=optB_true_m,
                optA1_eq_host=optA1_eq_host, optA1_moat_ok=optA1_moat_ok, optA1_decisive=optA1_decisive, optA1_true_m=optA1_true_m,
                optA2_eq_host=optA2_eq_host, optA2_moat_ok=optA2_moat_ok, optA2_decisive=optA2_decisive, optA2_true_m=optA2_true_m,
                lesion_fails_safe=lesion_fails_safe, lesion_decisions=les)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out", default="research/findings/raw/_phaseC_task1_S5_seam_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    results = []
    for s in seeds:
        r = run_seed(s, args.dim)
        results.append(r)
        print(f"seed {s} D{args.dim} (gain={r['gain']:.2e}, peak0={r['peak0']:.1f}):", flush=True)
        print(f"  OPT-B  (Phase B host coupling, the CONTROL): ==host={r['optB_eq_host']}  moat={r['optB_moat_ok']}  "
              f"decisive={r['optB_decisive']} true_m={[round(x,3) for x in r['optB_true_m']]}", flush=True)
        print(f"  OPT-A1 (raw-membrane on-bridge projection):  ==host={r['optA1_eq_host']}  moat={r['optA1_moat_ok']}  "
              f"decisive={r['optA1_decisive']} true_m={[round(x,3) for x in r['optA1_true_m']]}", flush=True)
        print(f"  OPT-A2 (rescaled-current + WTA projection):  ==host={r['optA2_eq_host']}  moat={r['optA2_moat_ok']}  "
              f"decisive={r['optA2_decisive']} true_m={[round(x,3) for x in r['optA2_true_m']]}", flush=True)
        print(f"  lesion-fails-safe={r['lesion_fails_safe']}", flush=True)
        for row in r["rows"]:
            print(f"    {row['kind']:16s} host={str(row['host']):6s} | B={str(row['optB_sub']):6s} {row['optB_rates']} "
                  f"| A1={str(row['optA1_sub']):6s} {row['optA1_rates']} lit={row['optA1_decoded_lit']} "
                  f"| A2={str(row['optA2_sub']):6s} {row['optA2_rates']} lit={row['optA2_decoded_lit']}", flush=True)

    n = len(results)
    optB_go = sum(r["optB_eq_host"] and r["optB_moat_ok"] for r in results)
    optA1_go = sum(r["optA1_eq_host"] and r["optA1_moat_ok"] and r["optA1_decisive"] and r["lesion_fails_safe"]
                   for r in results)
    optA2_go = sum(r["optA2_eq_host"] and r["optA2_moat_ok"] and r["optA2_decisive"] and r["lesion_fails_safe"]
                   for r in results)
    optA_go = max(optA1_go, optA2_go)
    optA1_moat = sum(r["optA1_moat_ok"] for r in results)
    optA2_moat = sum(r["optA2_moat_ok"] for r in results)
    if optA_go == n:
        which = "a1 (raw projection)" if optA1_go == n else "a2 (rescaled+WTA)"
        verdict = f"OPTION-A-GO via {which} -- the novel result->sequencer seam is ON-SUBSTRATE"
    elif optB_go == n:
        verdict = ("OPTION-B (HONEST NEGATIVE): the fixed on-bridge projection (a1 + a2) walls -> the result->sequencer "
                   "DATA hand-off stays a host read; the CONTROL (match + answer/abstain) is on-substrate (Phase B GO). "
                   "Moat NEVER weakened (a fixed projection that washes the match out either abstains or breaks the "
                   "moat; we report the boundary, not a softer moat).")
    else:
        verdict = "INCONCLUSIVE (option B control did not reproduce Phase B GO)"
    summary = dict(n=n, optB_go=optB_go, optA1_go=optA1_go, optA2_go=optA2_go, optA_go=optA_go,
                   optA1_moat=optA1_moat, optA2_moat=optA2_moat, verdict=verdict, gpu=is_gpu_backend())
    print(f"\nSUMMARY: optB(control) GO {optB_go}/{n}  optA1 GO {optA1_go}/{n} (moat {optA1_moat}/{n})  "
          f"optA2 GO {optA2_go}/{n} (moat {optA2_moat}/{n})\n  -> {verdict}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
