"""GNW Rung-2b de-risk: SPIKE-FREQUENCY-ADAPTATION eviction on an ASYNC competitive workspace.

Rung-2 (`_gnw_rung2_competitive_access_derisk.py`, 6-seed GO on MUTUAL EXCLUSION) established the shared-inhibition
WTA (only one content ignites) but left the salience-SELECTED winner + the causal-swap membership test SCOPED-PENDING:
the ignited state is a synchronous period-3 limit cycle, a challenger pulse lands on an arbitrary phase, and — with no
FATIGUE — an established attractor is either un-evictable (locks in) or annihilates. The Rung-2 finding named the exact
next mechanism (Dehaene & Changeux 2011, Neuron 70(2):200-227, metastability: an ignited workspace state must be able to
be "destabilized" and "spontaneously replaced by another"): an ASYNC rate attractor (heterogeneity + low OU noise) +
ADAPTATION-BASED EVICTION.

THIS RUNG adds the one limb Rung-2 never tried: SPIKE-FREQUENCY ADAPTATION (SFA) on the workspace assemblies. The
workspace neuron model is Izhikevich-2007, so SFA is the intrinsic spike-triggered recovery increment `d` (after every
spike `u += d`, which hyperpolarizes and adapts firing; `du/dt = a*(b*(v-vr) - u)`, so `a` sets the adaptation decay
timescale). A sustained incumbent fatigues over the incumbency-settle window and a continuously-present, more-salient
challenger DISPLACES it via biased competition — leaving EXACTLY ONE content ignited (the challenger), never NONE.

CONFIG-ONLY, NO `sim/` edit: after `build_competitive_bridge` we WRITE `cp_izh_d_increment` and `cp_izh_a` on the
workspace-region neurons (identical `d`/`a` on BOTH assemblies A and B — one frozen, init-invariant fatigue rule) and
re-settle+re-snapshot. `enable_homeostasis` / `enable_short_term_plasticity` stay OFF (banked foot-guns; the fatigue is
ONLY the intrinsic Izhikevich recovery current — GABA_B and STP were both banked as KILLED/annihilating for eviction).

SCAFFOLD REMOVED: the takeover in the headline result occurs in a CONTINUOUS free-run with NO `_restore_state` between
incumbent settling and challenger arrival (anti-cheat 3) — distinguishing real fatigue-eviction from a per-hop
snapshot-restore wash-out (the reset scaffold P1.2, the coincidence-integrator, and Rung-2 all currently lean on). The
sweep points DO reset between INDEPENDENT challenger drives (legitimate trial isolation, not a mid-competition wash-out).

GO GATE — 6 seeds 42/43/44/100/101/102 at ONE FROZEN (izh_d, izh_a, fs_to_ws, ou_noise, incumbent_settle), ALL of:
  - mutual exclusion preserved (no co-ignition anywhere in the incumbency sweep);
  - a_holds_weak: incumbent HOLDS a sub-crossover weak challenger;
  - monotone salience-graded takeover: a SINGLE crossover challenger-drive (B above / A below), no reversal;
  - causal-swap membership: swapping which assembly carries the higher salience FLIPS the ignited content,
    attributable >= ~0.9 of swaps;
  - post-takeover n_ignited == 1 (challenger), never 0 (anti-annihilation);
  - continuous run (anti-cheat 3): the takeover holds with NO mid-competition `_restore_state`;
  - controls: SFA-OFF (RS d/a) reproduces the phase-erratic negative; fs_to_ws=0 co-ignites (WTA load-bearing).

Usage:
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2b_sfa_workspace_eviction_derisk --seed 42 --smoke \
      --json research/findings/raw/_gnw_rung2b_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_rung2b_sfa_workspace_eviction_derisk --seed 42 \
      --izh-d 400 --izh-a 0.02 --fs-to-ws 16 --ou-noise 40 --incumbent-settle 120 \
      --json research/findings/raw/_gnw_rung2b_seed42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from sim.backend import get_backend, to_host
from tools.verdict import Verdict

from research.runners._gnw_rung1_ignition_curve_derisk import (
    _snapshot_state, _restore_state, DEFAULT_ATTRACTOR_WEIGHT, DRIVE_STEPS, FREE_STEPS, SETTLE_STEPS,
)
from research.runners._gnw_rung2_competitive_access_derisk import (
    build_competitive_bridge, _ignited, IGNITE_FRAC, SOLO_PLATEAU,
    WORKSPACE_N, ASSEMBLY_SIZE, FS_TO_WS_WEIGHT,
)

# Izhikevich-2007 RS defaults (SFA-OFF baseline == cfg default == "no added adaptation"): d=100, a=0.03.
RS_IZH_D = 100.0
RS_IZH_A = 0.03


# ── SFA injection: config-frozen adaptation on the workspace region (NO sim/ edit) ─────────────────────────────
def _apply_sfa(bridge, xp, izh_d: float, izh_a: float):
    """Write the SAME spike-triggered recovery increment `izh_d` and adaptation rate `izh_a` onto EVERY
    workspace-region neuron (both assemblies A and B identically — one init-invariant fatigue rule; anti-cheat 5).
    The `workspace_fs` inhibitory pool KEEPS its FS params (fast inhibition, no SFA). Returns (ws, d_applied,
    a_applied) where the applied values are READ BACK from the arrays (knob-verification, anti-cheat 8)."""
    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    ws_dev = xp.asarray(ws)
    bridge.cp_izh_d_increment[ws_dev] = xp.float32(izh_d)
    bridge.cp_izh_a[ws_dev] = xp.float32(izh_a)
    d_applied = float(to_host(bridge.cp_izh_d_increment[ws_dev].astype(xp.float64).mean()))
    a_applied = float(to_host(bridge.cp_izh_a[ws_dev].astype(xp.float64).mean()))
    return ws, d_applied, a_applied


def build_sfa_bridge(seed: int = 42, izh_d: float = RS_IZH_D, izh_a: float = RS_IZH_A,
                     fs_to_ws: float = FS_TO_WS_WEIGHT, ou_noise_pA: float = 40.0, heterogeneity: bool = True,
                     fs_lesion: bool = False, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT):
    """Rung-2 competitive bridge (async: heterogeneity + OU) + SFA on the workspace region, re-settled+re-snapshotted
    so the quiescent snapshot reflects the adapted substrate. Returns (bridge, xp, A_dev, B_dev, snap, handles)."""
    bridge, xp, A_dev, B_dev, _snap0, handles = build_competitive_bridge(
        seed=seed, attractor_weight=attractor_weight, fs_lesion=fs_lesion, fs_to_ws=fs_to_ws,
        heterogeneity=heterogeneity, ou_noise_pA=ou_noise_pA)
    ws, d_applied, a_applied = _apply_sfa(bridge, xp, izh_d, izh_a)
    # re-settle to quiescence with the adaptation in place, then snapshot the rest state.
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)
    handles = dict(handles)
    handles.update({"izh_d": float(izh_d), "izh_a": float(izh_a), "izh_d_applied": d_applied,
                    "izh_a_applied": a_applied, "ou_noise_pA": float(ou_noise_pA),
                    "heterogeneity": bool(heterogeneity), "ws_n": int(ws.size)})
    return bridge, xp, A_dev, B_dev, snap, handles


# restore-call accounting: the continuous headline MUST make ZERO restore calls (anti-cheat 3).
_RESTORE_CALLS = {"n": 0}


def _restore_counted(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _restore_state(bridge, snap)


def _late_rate(bridge, xp, A_dev, B_dev, n_steps: int, late_start: int):
    """Free-run `n_steps` (zero external current) and return the late-window per-neuron rate of A and B."""
    a_late = 0
    b_late = 0
    for t in range(n_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            a_late += int(to_host(bridge.cp_firing_states[A_dev].astype(xp.float64).sum()))
            b_late += int(to_host(bridge.cp_firing_states[B_dev].astype(xp.float64).sum()))
    denom = float((n_steps - late_start) * A_dev.shape[0])
    return a_late / denom, b_late / denom


def run_incumbency_sfa(bridge, xp, A_dev, B_dev, snap, drive_inc: float, drive_chal: float,
                       challenger_is_B: bool = True, incumbent_settle: int = 120, isolate: bool = True):
    """ONE incumbency competition trial, CONTINUOUS within the competition (anti-cheat 3): NO `_restore_state`
    between the incumbent settling and the challenger arriving — the incumbent fatigues via intrinsic SFA and a
    more-salient challenger displaces it.
      isolate=True  -> `_restore_state(snap)` ONCE at the very start (independent-trial isolation for the sweep);
      isolate=False -> NO restore at all (the fully-continuous headline; the bridge is assumed at rest).
    Returns (late_rate_A, late_rate_B)."""
    inc_dev = A_dev if challenger_is_B else B_dev
    chal_dev = B_dev if challenger_is_B else A_dev
    bridge.cp_external_input_current[:] = 0.0
    if isolate:
        _restore_counted(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    for _ in range(DRIVE_STEPS):                          # (1) ignite the incumbent alone
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[inc_dev] = xp.float32(drive_inc)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(incumbent_settle):                     # (2) the incumbent holds AND fatigues (SFA)
        bridge._run_one_simulation_step()
    for _ in range(DRIVE_STEPS):                          # (3) the challenger arrives (NO restore before this)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[chal_dev] = xp.float32(drive_chal)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0             # (4) free -> the settled winner
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    return _late_rate(bridge, xp, A_dev, B_dev, FREE_STEPS, late_start)


def _winner(ra, rb):
    ia, ib = _ignited(ra), _ignited(rb)
    return ("A" if (ia and not ib) else ("B" if (ib and not ia) else ("both" if (ia and ib) else "none"))), ia, ib


def _is_monotone(winners):
    """A single crossover A...A -> B...B with no reversal (ignoring 'none'/'both' points, which are handled by
    their own gates). Returns (monotone, crossover_present)."""
    seq = [w for w in winners if w in ("A", "B")]
    crossover = ("A" in seq) and ("B" in seq)
    # monotone == at most ONE A->B transition and NO B->A transition.
    transitions = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1) if seq[i] != seq[i + 1]]
    monotone = (len([t for t in transitions if t == ("B", "A")]) == 0) and (len(transitions) <= 1)
    return bool(monotone), bool(crossover)


def _threshold_hash(bridge, xp):
    """Hash the seed-derived per-neuron Izhikevich params (heterogeneity is seeded from cfg.seed). Build twice at
    one seed -> identical -> the substrate is actually seeded by cfg.seed (NOT actual_seed_used)."""
    parts = []
    for name in ("cp_izh_C", "cp_izh_k", "cp_izh_vt", "cp_izh_vr", "cp_izh_vpeak"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            parts.append(np.asarray(to_host(arr), dtype=np.float64))
    if not parts:
        return ""
    return hashlib.sha256(np.concatenate(parts).tobytes()).hexdigest()


# ── the four properties at a fixed operating point ─────────────────────────────────────────────────────────────
def evaluate_operating_point(seed, izh_d, izh_a, fs_to_ws, ou_noise, incumbent_settle, heterogeneity,
                             drive_inc, chal_max, n_chal, attractor_weight=DEFAULT_ATTRACTOR_WEIGHT, verbose=True):
    """Build the SFA bridge at (izh_d, izh_a, fs, ou) and measure: the challenger sweep (mutual exclusion,
    a_holds_weak, monotone crossover, anti-annihilation), the causal-swap membership test, and the CONTINUOUS
    (no-restore) headline takeover. Returns a result dict."""
    drive_chals = list(np.linspace(0.0, float(chal_max), int(n_chal)))

    bridge, xp, A_dev, B_dev, snap, handles = build_sfa_bridge(
        seed=seed, izh_d=izh_d, izh_a=izh_a, fs_to_ws=fs_to_ws, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=False, attractor_weight=attractor_weight)

    # ── challenger-drive sweep (independent trials; isolate=True) ──────────────────────────────────────────────
    winners, a_rates, b_rates, both, none = [], [], [], [], []
    for dC in drive_chals:
        ra, rb = run_incumbency_sfa(bridge, xp, A_dev, B_dev, snap, drive_inc, float(dC),
                                    challenger_is_B=True, incumbent_settle=incumbent_settle, isolate=True)
        w, ia, ib = _winner(ra, rb)
        winners.append(w); a_rates.append(ra); b_rates.append(rb)
        both.append(bool(ia and ib)); none.append(bool((not ia) and (not ib)))
        if verbose:
            print(f"  [sweep] chal={dC:8.1f}  A={ra:.4f}{'*' if ia else ' '}  B={rb:.4f}{'*' if ib else ' '}  -> {w}",
                  flush=True)

    co_ignition_any = any(both)
    # a_holds_weak: at the SMALLEST non-zero challenger the incumbent A holds (winner A, not none/both).
    weak_idx = 1 if len(winners) > 1 else 0
    a_holds_weak = bool(winners[weak_idx] == "A")
    b_takes_strong = bool(winners[-1] == "B")
    monotone, crossover = _is_monotone(winners)
    # anti-annihilation on the sweep: no 'none' point once we are past the weakest challenger (a weak challenger
    # knocking the incumbent to NONE is the mutual-annihilation failure Rung-2 flagged).
    annihilation_any = any(none[weak_idx:])

    # ── causal-swap membership (salience is the ONLY swapped variable) ────────────────────────────────────────
    strong = float(chal_max)
    swap_trials = []
    # B challenges strong -> expect B; A challenges strong -> expect A. Repeat over a couple of strong drives to
    # get a fraction (attributable_to >= ~0.9). geometry/seed/params all held fixed; ONLY the salient role swaps.
    for strong_drive in (strong, 0.85 * strong):
        raB, rbB = run_incumbency_sfa(bridge, xp, A_dev, B_dev, snap, drive_inc, strong_drive,
                                      challenger_is_B=True, incumbent_settle=incumbent_settle, isolate=True)
        wB, _, _ = _winner(raB, rbB)
        raA, rbA = run_incumbency_sfa(bridge, xp, A_dev, B_dev, snap, drive_inc, strong_drive,
                                      challenger_is_B=False, incumbent_settle=incumbent_settle, isolate=True)
        wA, _, _ = _winner(raA, rbA)
        swap_trials.append({"strong_drive": float(strong_drive),
                            "B_challenges": {"A": raB, "B": rbB, "winner": wB, "follows_salience": wB == "B"},
                            "A_challenges": {"A": raA, "B": rbA, "winner": wA, "follows_salience": wA == "A"}})
    follows = [t["B_challenges"]["follows_salience"] for t in swap_trials] + \
              [t["A_challenges"]["follows_salience"] for t in swap_trials]
    swap_attribution = float(np.mean(follows)) if follows else 0.0
    causal_swap = bool(swap_attribution >= 0.9)

    # ── CONTINUOUS headline (anti-cheat 3): NO restore anywhere in the competition ────────────────────────────
    # Fresh bridge at rest -> ONE uninterrupted incumbent->evict run at the strong challenger; assert 0 restores.
    bridge_c, xp_c, A_c, B_c, snap_c, _ = build_sfa_bridge(
        seed=seed, izh_d=izh_d, izh_a=izh_a, fs_to_ws=fs_to_ws, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=False, attractor_weight=attractor_weight)
    restore_before = _RESTORE_CALLS["n"]
    # sample the incumbent BEFORE the challenger (continuous, isolate=False): ignite A, settle, read A/B.
    bridge_c.cp_external_input_current[:] = 0.0
    for _ in range(DRIVE_STEPS):
        bridge_c.cp_external_input_current[:] = 0.0
        bridge_c.cp_external_input_current[A_c] = xp_c.float32(drive_inc)
        bridge_c._run_one_simulation_step()
    bridge_c.cp_external_input_current[:] = 0.0
    for _ in range(incumbent_settle):
        bridge_c._run_one_simulation_step()
    ls = FREE_STEPS - max(1, FREE_STEPS // 3)
    a_pre, b_pre = _late_rate(bridge_c, xp_c, A_c, B_c, FREE_STEPS, FREE_STEPS - 1)  # instantaneous-ish pre read
    incumbent_ignited_pre = bool(_ignited(a_pre) and not _ignited(b_pre))
    # now the strong challenger arrives (still NO restore) and we read the settled winner.
    for _ in range(DRIVE_STEPS):
        bridge_c.cp_external_input_current[:] = 0.0
        bridge_c.cp_external_input_current[B_c] = xp_c.float32(strong)
        bridge_c._run_one_simulation_step()
    bridge_c.cp_external_input_current[:] = 0.0
    a_post, b_post = _late_rate(bridge_c, xp_c, A_c, B_c, FREE_STEPS, ls)
    w_post, ia_post, ib_post = _winner(a_post, b_post)
    n_ignited_post = int(ia_post) + int(ib_post)
    restore_after = _RESTORE_CALLS["n"]
    continuous_no_restore = bool(restore_after == restore_before)  # ZERO restore calls in the headline
    continuous_takeover = bool(w_post == "B")
    anti_annihilation = bool(n_ignited_post == 1)

    # ── determinism: build twice at this seed, hash the seed-derived params (cfg.seed, NOT actual_seed_used) ────
    h1 = _threshold_hash(bridge, xp)
    bridge2, xp2, _, _, _, _ = build_sfa_bridge(
        seed=seed, izh_d=izh_d, izh_a=izh_a, fs_to_ws=fs_to_ws, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=False, attractor_weight=attractor_weight)
    h2 = _threshold_hash(bridge2, xp2)
    seed_deterministic = bool(h1 == h2 and h1 != "")

    op_go = bool(
        (not co_ignition_any) and a_holds_weak and b_takes_strong and monotone and crossover
        and causal_swap and (not annihilation_any)
        and continuous_no_restore and continuous_takeover and anti_annihilation and seed_deterministic)

    result = {
        "seed": int(seed),
        "izh_d_applied": handles.get("izh_d_applied"), "izh_a_applied": handles.get("izh_a_applied"),
        "operating_point": {"izh_d": float(izh_d), "izh_a": float(izh_a), "fs_to_ws": float(fs_to_ws),
                            "ou_noise_pA": float(ou_noise), "incumbent_settle": int(incumbent_settle),
                            "heterogeneity": bool(heterogeneity), "drive_inc": float(drive_inc),
                            "chal_max": float(chal_max), "n_chal": int(n_chal)},
        "drive_challengers": [float(x) for x in drive_chals],
        "a_rates": [float(x) for x in a_rates], "b_rates": [float(x) for x in b_rates],
        "winner_per_challenger": winners,
        "mutual_exclusion": bool(not co_ignition_any),
        "a_holds_weak": a_holds_weak, "b_takes_strong": b_takes_strong,
        "monotone": monotone, "crossover": crossover, "annihilation_on_sweep": annihilation_any,
        "causal_swap": {"attribution": swap_attribution, "pass": causal_swap, "trials": swap_trials},
        "continuous_headline": {
            "no_restore_calls": continuous_no_restore, "takeover": continuous_takeover,
            "incumbent_ignited_pre_challenge": incumbent_ignited_pre,
            "n_ignited_post": n_ignited_post, "anti_annihilation": anti_annihilation,
            "A_pre": a_pre, "B_pre": b_pre, "A_post": a_post, "B_post": b_post, "winner_post": w_post},
        "seed_deterministic": seed_deterministic, "threshold_hash": h1,
        "op_go": op_go,
    }
    if verbose:
        print(f"  [op seed={seed} d={izh_d} a={izh_a} fs={fs_to_ws} ou={ou_noise} settle={incumbent_settle}] "
              f"go={op_go} | mutual_excl={not co_ignition_any} holds_weak={a_holds_weak} "
              f"takes_strong={b_takes_strong} monotone={monotone} crossover={crossover} "
              f"causal_swap={swap_attribution:.2f} contin_no_restore={continuous_no_restore} "
              f"contin_takeover={continuous_takeover} n_ign_post={n_ignited_post} det={seed_deterministic}",
              flush=True)
    return result


# ── anti-cheat controls (SFA-OFF reproduces the negative; WTA lesion co-ignites) ───────────────────────────────
def control_sfa_off(seed, fs_to_ws, ou_noise, incumbent_settle, heterogeneity, drive_inc, chal_max, n_chal):
    """SFA-OFF (RS d/a): the monotone takeover + causal-swap must FAIL (reproduce the Rung-2 phase-erratic 1/6) ->
    the eviction is CAUSED by the added SFA, not by the async knobs alone."""
    r = evaluate_operating_point(seed, RS_IZH_D, RS_IZH_A, fs_to_ws, ou_noise, incumbent_settle, heterogeneity,
                                 drive_inc, chal_max, n_chal, verbose=False)
    reproduces_negative = bool(not (r["monotone"] and r["crossover"] and r["causal_swap"]["pass"]))
    return {"reproduces_negative": reproduces_negative, "monotone": r["monotone"], "crossover": r["crossover"],
            "causal_swap_pass": r["causal_swap"]["pass"], "winner_per_challenger": r["winner_per_challenger"]}


def control_wta_lesion(seed, izh_d, izh_a, fs_to_ws, ou_noise, incumbent_settle, heterogeneity, drive_inc, chal_max):
    """WTA lesion (fs_to_ws=0) WITH SFA on: BOTH assemblies co-ignite -> mutual exclusion comes from the shared
    inhibition, not from SFA silencing one assembly."""
    bridge, xp, A_dev, B_dev, snap, _ = build_sfa_bridge(
        seed=seed, izh_d=izh_d, izh_a=izh_a, fs_to_ws=fs_to_ws, ou_noise_pA=ou_noise,
        heterogeneity=heterogeneity, fs_lesion=True)
    ra, rb = run_incumbency_sfa(bridge, xp, A_dev, B_dev, snap, drive_inc, float(chal_max),
                                challenger_is_B=True, incumbent_settle=incumbent_settle, isolate=True)
    both = bool(_ignited(ra) and _ignited(rb))
    return {"both_ignite": both, "A": ra, "B": rb}


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-2b SFA workspace-eviction de-risk (async competitive).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung2b_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    # Defaults = the tested async operating regime (heterogeneity raises the ignition knee to ~4000-5000, so
    # drive_inc=5000 robustly ignites the incumbent; chal_max=8000 spans up to ~1.6x the incumbent's salience).
    ap.add_argument("--izh-d", type=float, default=400.0, help="spike-triggered recovery increment (SFA strength)")
    ap.add_argument("--izh-a", type=float, default=0.03, help="adaptation decay rate (smaller = slower/accumulates)")
    ap.add_argument("--fs-to-ws", type=float, default=28.0, help="shared mutual-inhibition strength")
    ap.add_argument("--ou-noise", type=float, default=40.0, help="OU noise std (pA) for async desync")
    ap.add_argument("--incumbent-settle", type=int, default=150, help="free steps the incumbent holds + fatigues")
    ap.add_argument("--no-heterogeneity", action="store_true", help="disable parameter heterogeneity")
    ap.add_argument("--drive-inc", type=float, default=5000.0)
    ap.add_argument("--chal-max", type=float, default=8000.0)
    ap.add_argument("--n-chal", type=int, default=9)
    ap.add_argument("--smoke", action="store_true", help="grid-scan (izh_d,izh_a) on ONE seed to find the window")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)
    het = not args.no_heterogeneity

    if args.smoke:
        print(f"[rung2b-smoke] seed={args.seed} fs={args.fs_to_ws} ou={args.ou_noise} settle={args.incumbent_settle} "
              f"het={het} — scanning (izh_d, izh_a) for the ignite-hold-AND-evictable window", flush=True)
        grid = []
        for izh_d in (200.0, 400.0, 600.0):   # spans the dichotomy: hold (un-evictable) -> self-extinction
            for izh_a in (0.01, 0.02, 0.03):
                r = evaluate_operating_point(args.seed, izh_d, izh_a, args.fs_to_ws, args.ou_noise,
                                             args.incumbent_settle, het, args.drive_inc, args.chal_max,
                                             args.n_chal, verbose=True)
                grid.append(r)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_rung2b_sfa_workspace_eviction_derisk", "mode": "smoke", "grid": grid}, f, indent=2)
        any_go = any(g["op_go"] for g in grid)
        print(f"\n[rung2b-smoke] wrote {args.json}  any_op_go={any_go}", flush=True)
        return 0 if any_go else 1

    # single-seed evaluation at the frozen operating point + the two anti-cheat controls.
    print(f"[rung2b] seed={args.seed} d={args.izh_d} a={args.izh_a} fs={args.fs_to_ws} ou={args.ou_noise} "
          f"settle={args.incumbent_settle} het={het} (frozen operating point)", flush=True)
    r = evaluate_operating_point(args.seed, args.izh_d, args.izh_a, args.fs_to_ws, args.ou_noise,
                                 args.incumbent_settle, het, args.drive_inc, args.chal_max, args.n_chal, verbose=True)
    off = control_sfa_off(args.seed, args.fs_to_ws, args.ou_noise, args.incumbent_settle, het,
                          args.drive_inc, args.chal_max, args.n_chal)
    lesion = control_wta_lesion(args.seed, args.izh_d, args.izh_a, args.fs_to_ws, args.ou_noise,
                                args.incumbent_settle, het, args.drive_inc, args.chal_max)
    print(f"[rung2b] SFA-OFF reproduces_negative={off['reproduces_negative']} (monotone={off['monotone']} "
          f"causal_swap={off['causal_swap_pass']}) | WTA-lesion both_ignite={lesion['both_ignite']}", flush=True)

    go = bool(r["op_go"] and off["reproduces_negative"] and lesion["both_ignite"])

    # The verdict must travel with what earned it (tools.verdict.Verdict -> a `preconditions` block). These are
    # the VALIDITY conditions of the experiment (all must hold); the negative is the ABSENCE of a clean takeover
    # (go=False), NOT a failed precondition. If any of these did NOT hold, the result would be UNDEFINED, not a
    # negative.
    v = Verdict("rung2b SFA salience-eviction @ frozen operating point (seed %d)" % args.seed)
    v.require("incumbent ignites & holds a weak challenger", r["a_holds_weak"], expect=True)
    v.reaches("SFA injected (izh_d differs from RS default)", before=RS_IZH_D, after=r["operating_point"]["izh_d"])
    v.knob("izh_d applied to workspace neurons", requested=r["operating_point"]["izh_d"], applied=r["izh_d_applied"])
    v.require("WTA inhibition load-bearing (fs=0 lesion co-ignites)", lesion["both_ignite"], expect=True)
    v.require("SFA-OFF reproduces the phase-erratic negative", off["reproduces_negative"], expect=True)
    v.require("continuous headline: zero _restore_state calls", r["continuous_headline"]["no_restore_calls"], expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", r["seed_deterministic"], expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("short_term_plasticity", why="STP banked as annihilating for eviction (2026-08-01)")
    vd = v.decide(go=go)

    result = {"runner": "_gnw_rung2b_sfa_workspace_eviction_derisk", "mode": "single", "go": go,
              "verdict": vd["status"], "preconditions": vd["preconditions"],
              "disabled_processes": vd["disabled_processes"], "undefined_reasons": vd["undefined_reasons"],
              "backend": args.backend, "operating_point": r["operating_point"], "eval": r,
              "control_sfa_off": off, "control_wta_lesion": lesion}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[rung2b] seed={args.seed} GO={go}  wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
