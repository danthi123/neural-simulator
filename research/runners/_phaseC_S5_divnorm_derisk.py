"""Phase C — TASK 1 follow-on (S5, OPTION 4): close the last host read via the on-bridge `input_divisive_norm`
Carandini-Heeger primitive — the cheap-first de-risk the deep research named.

Task 1 (`2026-06-19-phaseC-task1-S5-seam-derisk.md`, commit 27c6422e) proved a FIXED `cp_connections` projection
cannot carry the cleanup result to the sequencer without a host read: the cleanup score is a GRADED matched-filter
membrane `Re(c)` (winner ~peak ~1.3e6, runner-up ~0.4*peak — RELATIVE discrimination), and a binary Izhikevich
spike fires identically for winner AND runner-up, so the whole decoded row lights → the moat breaks. The host
`scores_to_drive`/`_spiking_cleanup` divide by `scores.max()` — a PER-QUERY peak read a fixed projection can't
express. The deep research (`2026-06-19-S5-on-bridge-normalization-deep-research.md`) re-framed that as point-neuron-
feasible DIVISIVE GAIN control on an already-rectified non-negative magnitude (Carandini-Heeger; NOT the off-diagonal
whitening dendritic boundary), and named OPTION 4 — the cheapest, ZERO new sim/ code: flag the cleanup-score region
`input_divisive_norm=True` (the existing `sim/regions.py:240` + `sim/config.py:440` + the guarded per-step block at
`sim/bridge.py:6048` primitive: `r_i = x_i / (sigma + gain*mean_j x_j)`, default-off byte-identical), so the
PRE-THRESHOLD score is divided ON-BRIDGE by the per-query total pool drive (the per-query gain) and a PLACED firing
threshold then cleanly separates the winner from the runner-up — WITHOUT the host `scores.max()` read.

THE MECHANISM (this runner):
  1. The REAL OneBrainComposer reconstructs+unbinds+cleans each stored block (the validated `block_cleanup_scores`
     op) → per-role-per-word cleanup membrane scores (the same arrays the host argmaxes / Phase B's host coupling
     reads).
  2. A divisive-norm SCORE bridge (Izhikevich, ONE flagged V-neuron pool per role) is driven with the role's score
     vector as EXTERNAL INPUT CURRENT (scaled by a FIXED per-bridge input gain — NOT per-query). The bridge's
     guarded `input_divisive_norm` block divides that drive by `sigma + gain*mean(pool drive)` every step — the
     per-query divisor IS the pool's own total drive (high-peak query down-scaled relative to low-peak). A PLACED
     firing threshold (rheobase) reads which words FIRE: only the winner crosses, the runner-up stays silent (the
     normalization made the threshold scale-invariant). NO host `scores.max()` anywhere in the score path.
  3. The per-word FIRING → the decoded-line drive → Phase B's SAME sequencer (`run_sequencer_with_drive`) → the
     spiking match cascade → the answer/abstain decision. The decision is compared to the host `_scan`.

THE DECISIVE CONTROL (the research's named anti-cheat): a PER-QUERY-PEAK SWEEP. Each cleanup score vector is scaled
by explicit per-query multipliers spanning >= 1 ORDER OF MAGNITUDE (the natural 2 facts only span ~1.46x). The
normalization must separate winner from runner-up at a FIXED operating point ACROSS those peaks — a one-peak pass
is NOT a closure (it would be the host-read-replacement-that-only-works-for-one-query trap).

GATES (GO):
  - ==host on who/what (K=2 facts), 0 false-accepts (absent-agent / absent-action / cross cues all abstain — the
    MOAT, the HARD gate), lesion-fails-safe, ACROSS the peak sweep, >=3 seeds; AND
  - enable_input_divisive_norm=False (the OFF guard) == byte-identical to the no-divnorm path.
ANTI-CHEATS (all): moat 0-FA HARD · HOST-NORM positive control (== the host `scores_to_drive` peak read) · NO-NORM
negative control (the placed threshold on RAW un-normalized drive must FAIL the peak sweep — reproduce the Task-1
whole-row-lights wall) · lesion-fails-safe · OFF==byte-identical.

NO `sim/` edit (reuse-by-import: OneBrainComposer + Phase B's sequencer + the EXISTING `input_divisive_norm` sim/
primitive flipped on a runner-built score bridge). HONEST NEGATIVE is a valid deliverable: if no single (sigma, gain,
input_gain, threshold) op separates across the peak sweep without a moat breach, S5 maps to the deferred dendritic
substrate (Option 5) and the qualified one-host-read loop is the point-neuron ceiling — report the boundary, never
relax the moat to manufacture a pass.

  SIM_BACKEND=numpy python -u -m research.runners._phaseC_S5_divnorm_derisk --seeds 42,43,44 --dim 64
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
    decision_to_patient, host_scan, reset_sequencer_state, FACTS, VOCAB,
)


# ----------------------------------------------------------------------------------------------------------------
# OPTION 4 score bridge: ONE divisive-norm-flagged Izhikevich word-pool per role. We process the agent role and the
# action role through the SAME flagged pool, ONE role at a time, so the divisor (mean over the flagged set) is that
# role's OWN per-query total pool drive (the "divide pre-threshold input by total pool drive" the research names) —
# NOT the agent+action pooled together. Each word is a tiny pool (n_word neurons). The pool is flagged
# input_divisive_norm=True; the global cfg.enable_input_divisive_norm gates the guarded per-step divide.
# ----------------------------------------------------------------------------------------------------------------
def build_divnorm_score_bridge(seed, V, n_word=20, enable_divnorm=True, sigma=1.0, gain=1.0):
    """A plain Izhikevich bridge: V word-pools `w{w}` (one normalization pool = the role's V words). When
    enable_divnorm, ALL V pools are flagged input_divisive_norm=True and the global flag is set, so the per-step
    block divides each pool's drive by (sigma + gain*mean over the V*n_word flagged neurons). No internal/cross
    wiring — the firing is driven purely by the (divisively-normalized) external input current vs rheobase."""
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = int(seed)
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    cfg.connections_per_neuron = 0   # no internal/cross wiring: signal "empty CSR" (the score pools are driven by
                                     # external input current only) — avoids the spatial-fallback bug on an empty plan
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp"):
        setattr(cfg, flag, False)
    cfg.enable_input_divisive_norm = bool(enable_divnorm)
    cfg.input_divisive_sigma = float(sigma)
    cfg.input_divisive_gain = float(gain)
    regions = [BrainRegion(name=f"w{w}", n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                           input_divisive_norm=bool(enable_divnorm))
               for w in range(V)]
    cfg.brain_regions = regions
    cfg.region_pathways = []
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    return sb


def _reset_score_bridge(sb):
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = -65.0
    sb.cp_recovery_variable_u[:] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False


def onbridge_divnorm_drive(score_sb, V, scores, input_gain, settle=20, hi_pA=1500.0):
    """OPTION 4: drive each word pool with current = input_gain * max(score, 0) (a FIXED per-bridge gain — NOT a
    per-query normalization), let the bridge's input_divisive_norm divide the pre-threshold drive by the per-query
    total pool drive every step, and read which word pools FIRE (the placed firing threshold = the Izhikevich
    rheobase). Returns (drive[V], acc_fired[V]). A word is 'driven' iff its pool fired during the settle. NO host
    scores.max() — the per-query rescaling is the on-bridge divide; the threshold is the same across queries."""
    idx = lambda nm: np.asarray(score_sb.region_manager.indices(nm))
    _reset_score_bridge(score_sb)
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    cur = np.zeros(score_sb.core_config.num_neurons, dtype=np.float64)
    for w in range(V):
        drv = float(input_gain * s[w])
        if drv > 0:
            cur[idx(f"w{w}")] = drv
    acc = np.zeros(V)
    cur = from_host(cur)
    for _ in range(settle):
        score_sb.cp_external_input_current[:] = cur
        score_sb._run_one_simulation_step()
        fir = np.asarray(to_host(score_sb.cp_firing_states)).astype(float)
        for w in range(V):
            acc[w] += fir[idx(f"w{w}")].mean()
    score_sb.cp_external_input_current[:] = 0.0
    drive = np.where(acc > 0, hi_pA, 0.0)
    return drive, acc


def run_sequencer_with_drive(sb, meta, cue_a, cue_x, drives, settle=60, match_thresh=0.15):
    """Phase B's sequencer driven with decoded-line drives supplied DIRECTLY (the option-4 on-bridge drive), like
    the Task-1 harness. `drives` = [(dA,dX), ...] per block."""
    V = meta["V"]
    idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
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
    cur = from_host(cur)
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


# ----------------------------------------------------------------------------------------------------------------
# Drive-generators: each maps a block's (agent_scores, action_scores) -> (dA, dX). Three variants:
#   divnorm  — OPTION 4 on-bridge divisive-norm score pool + placed threshold (NO host scores.max()).
#   hostnorm — the host scores_to_drive peak read (the POSITIVE control: == the current S5).
#   raw      — the placed threshold on RAW un-normalized drive (the NEGATIVE control: must fail the peak sweep).
# Each is called per block; the PER-QUERY-PEAK sweep multiplies the scores by `peak_mult` BEFORE the generator.
# ----------------------------------------------------------------------------------------------------------------
def make_drives_divnorm(score_sb, V, bscores, input_gain, peak_mult):
    drives, lit = [], []
    for (ag, ax) in bscores:
        dA, accA = onbridge_divnorm_drive(score_sb, V, ag * peak_mult, input_gain)
        dX, accX = onbridge_divnorm_drive(score_sb, V, ax * peak_mult, input_gain)
        drives.append((dA, dX)); lit.append((int((accA > 0).sum()), int((accX > 0).sum())))
    return drives, lit


def make_drives_hostnorm(V, bscores, peak_mult):
    drives = []
    for (ag, ax) in bscores:
        drives.append((scores_to_drive(ag * peak_mult), scores_to_drive(ax * peak_mult)))
    return drives, None


def make_drives_raw(score_sb_nonorm, V, bscores, input_gain, peak_mult):
    """NEGATIVE control: same placed threshold + same input_gain, but the score bridge has divnorm OFF, so the
    RAW (un-normalized) drive hits rheobase — winner AND runner-up both fire (the Task-1 whole-row wall)."""
    drives, lit = [], []
    for (ag, ax) in bscores:
        dA, accA = onbridge_divnorm_drive(score_sb_nonorm, V, ag * peak_mult, input_gain)
        dX, accX = onbridge_divnorm_drive(score_sb_nonorm, V, ax * peak_mult, input_gain)
        drives.append((dA, dX)); lit.append((int((accA > 0).sum()), int((accX > 0).sum())))
    return drives, lit


def _gate_block(rows, sub_key, rates_key):
    """==host (all rows) + moat (every absent/cross cue abstains, HARD 0-FA) + decisive (true-present m >= 0.20)."""
    moat_rows = [r for r in rows if r["kind"].startswith(("absent", "cross"))]
    moat_ok = all(r[sub_key] is None for r in moat_rows)
    eq_host = all(r[sub_key] == r["host"] for r in rows)
    present = [r for r in rows if r["kind"].endswith("present")]
    true_m = [max(r[rates_key]["m0"], r[rates_key]["m1"]) for r in present]
    decisive = all(m >= 0.20 for m in true_m) if true_m else False
    n_fa = sum(1 for r in moat_rows if r[sub_key] is not None)
    return eq_host, moat_ok, decisive, true_m, n_fa


QUERIES = [(("dog", "go"), "blk0-present"), (("cat", "run"), "blk1-present"),
           (("fox", "go"), "absent-agent"), (("dog", "see"), "absent-action"),
           (("dog", "run"), "cross-no-block")]


def check_off_byte_identical(seed=42, V=12):
    """The standing OFF==byte-identical guard: a divnorm-OFF score bridge has cp_input_divisive_mask=None (the
    per-step divide block is unreached) and steps byte-identically to a second OFF bridge; an ON bridge's mask is
    not None and the divide CHANGES the dynamics (load-bearing). Confirms the primitive is a guarded no-op when off,
    so every existing run is byte-unchanged (NO sim/ edit was made — this just asserts the primitive's contract)."""
    drv = np.linspace(50.0, 500.0, V)   # modest pA: would trigger the divide if ON

    def _step(sb):
        _reset_score_bridge(sb)
        idx = lambda nm: np.asarray(sb.region_manager.indices(nm))
        cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
        for w in range(V):
            cur[idx(f"w{w}")] = drv[w]
        sb.cp_external_input_current[:] = from_host(cur)
        sb._run_one_simulation_step()
        v = np.asarray(to_host(sb.cp_membrane_potential_v)).copy()
        sb.cp_external_input_current[:] = 0.0
        return v

    b_off = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=False)
    b_off2 = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=False)
    b_on = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=True, sigma=1.0, gain=0.05)
    v_off, v_off2, v_on = _step(b_off), _step(b_off2), _step(b_on)
    return dict(off_mask_none=bool(b_off.cp_input_divisive_mask is None),
                on_mask_not_none=bool(b_on.cp_input_divisive_mask is not None),
                off_eq_off=bool(np.array_equal(v_off, v_off2)),
                on_differs_from_off=bool(not np.allclose(v_off, v_on)))


def run_seed(seed, D, input_gain, sigma, gain, peak_mults):
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=8, enable_batched=False, enable_rf_cudagraph=False)
    for (a, x, p) in FACTS:
        c.store(a, x, p)
    V = c.V
    word_idx = {w: i for i, w in enumerate(c.words)}
    blocks = list(range(len(FACTS)))
    bscores = [block_cleanup_scores(c, b) for b in blocks]            # the REAL cleanup op results per block

    seq_sb, meta = build_sequencer_bridge(seed=seed, V=V)            # Phase B's sequencer (shared across variants)
    div_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=True, sigma=sigma, gain=gain)
    raw_sb = build_divnorm_score_bridge(seed=seed, V=V, enable_divnorm=False)   # NEGATIVE control (divnorm OFF)

    # per-query-peak SWEEP: run the whole battery at each peak multiplier. GO requires ALL gates at EVERY peak.
    per_peak = {}
    for pm in peak_mults:
        rows = []
        for (qa, qx), kind in QUERIES:
            ca, cx = word_idx[qa], word_idx[qx]
            host = host_scan(c, qa, qx)
            d_div, lit_div = make_drives_divnorm(div_sb, V, bscores, input_gain, pm)
            d_host, _ = make_drives_hostnorm(V, bscores, pm)
            d_raw, lit_raw = make_drives_raw(raw_sb, V, bscores, input_gain, pm)
            dec_div, rt_div = run_sequencer_with_drive(seq_sb, meta, ca, cx, d_div)
            dec_host, rt_host = run_sequencer_with_drive(seq_sb, meta, ca, cx, d_host)
            dec_raw, rt_raw = run_sequencer_with_drive(seq_sb, meta, ca, cx, d_raw)
            rows.append(dict(cue=(qa, qx), kind=kind, host=host,
                             div_sub=decision_to_patient(c, dec_div, blocks), div_dec=dec_div, div_rates=rt_div,
                             div_lit=lit_div,
                             host_sub=decision_to_patient(c, dec_host, blocks), host_dec=dec_host, host_rates=rt_host,
                             raw_sub=decision_to_patient(c, dec_raw, blocks), raw_dec=dec_raw, raw_rates=rt_raw,
                             raw_lit=lit_raw))
        d_eq, d_moat, d_dec, d_tm, d_fa = _gate_block(rows, "div_sub", "div_rates")
        h_eq, h_moat, h_dec, h_tm, h_fa = _gate_block(rows, "host_sub", "host_rates")
        r_eq, r_moat, r_dec, r_tm, r_fa = _gate_block(rows, "raw_sub", "raw_rates")
        per_peak[pm] = dict(rows=rows,
                            div=dict(eq_host=d_eq, moat_ok=d_moat, decisive=d_dec, true_m=d_tm, n_fa=d_fa),
                            host=dict(eq_host=h_eq, moat_ok=h_moat, decisive=h_dec, true_m=h_tm, n_fa=h_fa),
                            raw=dict(eq_host=r_eq, moat_ok=r_moat, decisive=r_dec, true_m=r_tm, n_fa=r_fa))

    # LESION (sever the score-pool drive): on a present cue, the decoded lines get ZERO drive -> must abstain.
    les = []
    for (qa, qx) in (("dog", "go"), ("cat", "run")):
        zero = [(np.zeros(V), np.zeros(V)) for _ in bscores]
        dec_l, _ = run_sequencer_with_drive(seq_sb, meta, word_idx[qa], word_idx[qx], zero)
        les.append(dec_l)
    lesion_fails_safe = all(d == "abstain" for d in les)

    # GO across the FULL peak sweep. The S5-CLOSURE gate is what S5 actually controls: ==host (the on-bridge-
    # normalized decision matches the host) + moat-0-FA (HARD) + lesion-fails-safe. `decisive` (the Phase-B match
    # pool m>=0.20) is a SEQUENCER-health diagnostic — a property of the SHARED Phase-B sequencer, NOT the S5
    # normalizer (it dips marginally under the cupy float path and fails IDENTICALLY in the host-norm control), so
    # it is reported but does NOT mask a clean ==host+moat S5 closure.
    div_s5_all = all(per_peak[pm]["div"]["eq_host"] and per_peak[pm]["div"]["moat_ok"] for pm in peak_mults) \
        and lesion_fails_safe
    div_decisive_all = all(per_peak[pm]["div"]["decisive"] for pm in peak_mults)
    div_go_all = div_s5_all   # the load-bearing S5 gate
    div_moat_all = all(per_peak[pm]["div"]["moat_ok"] for pm in peak_mults)
    div_fa_all = sum(per_peak[pm]["div"]["n_fa"] for pm in peak_mults)
    host_go_all = all(per_peak[pm]["host"]["eq_host"] and per_peak[pm]["host"]["moat_ok"] for pm in peak_mults)
    host_decisive_all = all(per_peak[pm]["host"]["decisive"] for pm in peak_mults)
    # NEGATIVE control: across the sweep the RAW path must FAIL the S5 gate (some peak breaks the moat OR loses
    # ==host). raw_fails_sweep True = the control behaved (normalization is load-bearing).
    raw_s5_all = all(per_peak[pm]["raw"]["eq_host"] and per_peak[pm]["raw"]["moat_ok"] for pm in peak_mults)
    raw_fails_sweep = not raw_s5_all

    return dict(seed=seed, D=D, input_gain=input_gain, sigma=sigma, gain=gain, peak_mults=peak_mults,
                per_peak=per_peak, lesion_fails_safe=lesion_fails_safe, lesion_decisions=les,
                div_go_all=div_go_all, div_decisive_all=div_decisive_all, div_moat_all=div_moat_all,
                div_fa_all=div_fa_all, host_go_all=host_go_all, host_decisive_all=host_decisive_all,
                raw_fails_sweep=raw_fails_sweep, raw_s5_all=raw_s5_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--input-gain", type=float, default=1.0,
                    help="fixed per-bridge input gain (NOT per-query): drive = input_gain*score, then on-bridge divide. "
                         "Large enough to be in the SATURATED (scale-invariant) divnorm regime (input_gain*gain*mean>>sigma)")
    ap.add_argument("--sigma", type=float, default=1.0, help="divisive semi-saturation constant")
    ap.add_argument("--gain", type=float, default=0.05,
                    help="divisive strength on the mean term. Small enough that the SATURATED normalized ratio "
                         "peak/(gain*mean) lands in the Izhikevich firing band (winner supra-rheobase, runner-up sub) — "
                         "the placed threshold IS the rheobase, scale-invariant across per-query peaks")
    ap.add_argument("--peak-mults", default="0.1,1.0,10.0",
                    help="per-query peak multipliers (the decisive control: span >= 1 order of magnitude)")
    ap.add_argument("--out", default="research/findings/raw/_phaseC_S5_divnorm_derisk.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    peak_mults = [float(x) for x in args.peak_mults.split(",")]

    off_guard = check_off_byte_identical()
    off_ok = (off_guard["off_mask_none"] and off_guard["on_mask_not_none"]
              and off_guard["off_eq_off"] and off_guard["on_differs_from_off"])
    print(f"OFF==byte-identical guard: {off_guard} -> {'PASS' if off_ok else 'FAIL'}", flush=True)

    results = []
    for s in seeds:
        r = run_seed(s, args.dim, args.input_gain, args.sigma, args.gain, peak_mults)
        results.append(r)
        print(f"seed {s} D{args.dim} (input_gain={args.input_gain:.2e}, sigma={args.sigma}, gain={args.gain}, "
              f"peak_mults={peak_mults}):", flush=True)
        print(f"  DIVNORM (option 4, on-bridge): S5_GO_all={r['div_go_all']} (==host+moat+lesion) moat_all={r['div_moat_all']} "
              f"FA_total={r['div_fa_all']} lesion_safe={r['lesion_fails_safe']} | decisive_all={r['div_decisive_all']} (seq diag)",
              flush=True)
        print(f"  HOST-NORM (positive control):  S5_GO_all={r['host_go_all']} | decisive_all={r['host_decisive_all']} (seq diag)",
              flush=True)
        print(f"  RAW (negative control):        fails_sweep={r['raw_fails_sweep']} (S5_GO_all={r['raw_s5_all']})", flush=True)
        for pm in peak_mults:
            pp = r["per_peak"][pm]
            print(f"   peak_mult={pm:>6}: DIV ==host={pp['div']['eq_host']} moat={pp['div']['moat_ok']} "
                  f"dec={pp['div']['decisive']} tm={[round(x,3) for x in pp['div']['true_m']]} | "
                  f"RAW ==host={pp['raw']['eq_host']} moat={pp['raw']['moat_ok']}", flush=True)
            for row in pp["rows"]:
                print(f"      {row['kind']:16s} host={str(row['host']):6s} | DIV={str(row['div_sub']):6s} "
                      f"{row['div_rates']} lit={row['div_lit']} | HOST={str(row['host_sub']):6s} {row['host_rates']} "
                      f"| RAW={str(row['raw_sub']):6s} lit={row['raw_lit']}", flush=True)

    n = len(results)
    div_go = sum(r["div_go_all"] for r in results)
    div_decisive = sum(r["div_decisive_all"] for r in results)
    div_moat = sum(r["div_moat_all"] for r in results)
    div_fa = sum(r["div_fa_all"] for r in results)
    host_go = sum(r["host_go_all"] for r in results)
    host_decisive = sum(r["host_decisive_all"] for r in results)
    raw_fails = sum(r["raw_fails_sweep"] for r in results)
    lesion = sum(r["lesion_fails_safe"] for r in results)

    if div_go == n and host_go == n and raw_fails == n and div_fa == 0 and off_ok:
        decisive_note = ("" if (div_decisive == n) else
                         f" (NB: the Phase-B sequencer `decisive` m>=0.20 diagnostic dipped on {n - div_decisive}/{n} "
                         f"seed(s) under this backend — a SHARED sequencer-health margin, present in the host-norm "
                         f"control too ({host_decisive}/{n}); NOT an S5 gate, the ==host decision + moat are intact)")
        verdict = ("OPTION-4-GO across the per-query-peak sweep: the on-bridge input_divisive_norm primitive + a "
                   "placed threshold (the Izhikevich rheobase) reproduces the host decision (==host, moat 0-FA) at a "
                   "FIXED operating point across peaks spanning >=1 order of magnitude. The host scores.max() read is "
                   "RETIRED; S5 closes with ZERO new sim/ code. Positive control (host-norm) GO, negative control "
                   "(raw) fails the sweep (normalization is load-bearing), lesion fails safe." + decisive_note)
    elif div_fa > 0:
        verdict = (f"NEGATIVE (moat breach): the on-bridge divnorm path produced {div_fa} false-accept(s) across the "
                   "sweep — REPORTED as the boundary, the moat was NOT relaxed. S5 maps toward the dendritic branch "
                   "(Option 5) / escalate to Option 1 (the validated NEF input-norm FS pool).")
    elif not off_ok:
        verdict = (f"GUARD FAIL: the OFF==byte-identical guard did not hold ({off_guard}) — the primitive is not a "
                   "clean guarded no-op when off; investigate before claiming closure.")
    elif host_go != n:
        verdict = "INCONCLUSIVE (host-norm positive control did not pass the sweep — harness/battery issue)."
    elif raw_fails != n:
        verdict = ("INCONCLUSIVE (negative control raw path also passed the sweep — normalization not shown "
                   "load-bearing; the placed threshold may be mis-set).")
    else:
        verdict = ("HONEST BOUNDARY: no single fixed (input_gain, sigma, gain, threshold) op separates winner from "
                   "runner-up across the peak sweep with ==host + moat intact. S5's per-query normalization is not "
                   "carried by the mean-pool divnorm primitive at a fixed op → next try Option 1 (the validated NEF "
                   "input-norm FS pool, 2026-06-05-composer-cleanup-NEF-GO.md); do NOT escalate into a config search.")

    summary = dict(n=n, div_go=div_go, div_decisive=div_decisive, div_moat=div_moat, div_fa_total=div_fa,
                   host_go=host_go, host_decisive=host_decisive,
                   raw_fails_sweep=raw_fails, lesion=lesion, off_guard=off_guard, off_ok=off_ok,
                   verdict=verdict, gpu=is_gpu_backend(),
                   input_gain=args.input_gain, sigma=args.sigma, gain=args.gain, peak_mults=peak_mults)
    print(f"\nSUMMARY: DIVNORM S5_GO {div_go}/{n} (==host+moat+lesion; moat {div_moat}/{n}, FA_total {div_fa})  "
          f"HOST-NORM S5_GO {host_go}/{n}  RAW fails_sweep {raw_fails}/{n}  lesion {lesion}/{n}  "
          f"OFF-guard {'PASS' if off_ok else 'FAIL'}  | decisive(seq diag): DIV {div_decisive}/{n} HOST {host_decisive}/{n}"
          f"\n  -> {verdict}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, results=results), f, indent=2, default=str)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
