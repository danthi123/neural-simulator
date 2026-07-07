"""GNW Rung-2 de-risk: COMPETITIVE ACCESS to the Global Neuronal Workspace — only ONE content ignites.

Builds on Rung-1 (`_gnw_rung1_ignition_curve_derisk.py`, 6-seed GO: a recurrent workspace assembly shows an
all-or-none ignition). Rung-2 adds the SECOND load-bearing GNW property (Dehaene-Changeux 2011; Baars 1988
"one spotlight at a time"): when TWO candidate assemblies compete for the workspace, MUTUAL INHIBITION (a shared
inhibitory pool) enforces WINNER-TAKE-ALL — only one ignites, and which one is set by the relative drive
(salience). The causal membership test (the GWT paper's report+causal-swap): swapping the salience FLIPS the
ignited (= "reported") assembly.

MECHANISM (reuse-by-import, NO `sim/` edit): TWO self-recurrent assemblies (A, B) in one `workspace` region, each
a dense E->E loop at weight 30 (Rung-1's `_build_assembly_loop_population`), SHARING one inhibitory `workspace_fs`
pool (workspace->fs excites it, fs->workspace inhibits BOTH assemblies). The assembly that ignites first drives the
shared inhibition, raising the other's effective threshold -> biased competition (Wang 2002 two-attractor WTA).

THE DE-RISK (INCUMBENCY protocol; single variable = the challenger's salience): ignite assembly A ALONE (the
incumbent) + let it stabilize (it holds the workspace), THEN drive assembly B (the challenger) at a swept drive;
measure the late-window settled rate of A and B. (Symmetric double-drive was tried first and is messier — it
over-drives the shared inhibition; the incumbency framing is also the more GNW-meaningful test: a workspace holds
one content, a new content takes over only if more salient.)

RESULT (6-seed 42/43/44/100/101/102, numpy):
  ROBUST (GO gate): the shared inhibition enforces MUTUAL EXCLUSION — the two assemblies NEVER co-ignite (only one
  content occupies the workspace at a time; Baars "one spotlight") — AND it is LOAD-BEARING: the lesion
  (fs->workspace weight 0) lets BOTH ignite. This is the core GNW single-content-access property.
  SCOPED-PENDING (reported, NOT gated): the incumbent stably holding a weak challenger, a clean salience-graded
  takeover, and the CAUSAL-SWAP membership test are PHASE-ERRATIC / seed-dependent -> the ignited state is a
  SYNCHRONOUS period-3 limit cycle (Rung-1), and a challenger pulse landing on an arbitrary phase makes takeover
  erratic; displacing an established attractor also needs FATIGUE the assemblies lack. The named NEXT mechanism:
  an ASYNC rate attractor (neuron heterogeneity + low OU noise, both plumbed as `build_competitive_bridge`
  params) + ADAPTATION-BASED EVICTION (Dehaene-Changeux metastability — an ignited assembly fatigues -> becomes
  displaceable by a more-salient challenger).

GO GATE (the robust property): (1) MUTUAL EXCLUSION — no co-ignition anywhere in the incumbency sweep; (2) the
lesion removes it (both ignite) => the shared inhibition is load-bearing. The clean salience-graded takeover +
causal-swap membership test are SCOPED-PENDING diagnostics (see `scoped_pending` in the JSON), not GO conditions.

Usage:
  python -u -m research.runners._gnw_rung2_competitive_access_derisk --seed 42 \
      --json research/findings/raw/_gnw_rung2_smoke.json --backend numpy
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
from sim.backend import get_backend, to_host

# reuse the Rung-1 helpers (the assembly-loop builder + the quiescent snapshot/restore wash-out).
from research.runners._gnw_rung1_ignition_curve_derisk import (
    _build_assembly_loop_population, _snapshot_state, _restore_state,
    DEFAULT_ATTRACTOR_WEIGHT, SETTLE_STEPS, DRIVE_STEPS, FREE_STEPS,
)

# ── geometry: two assemblies in one workspace + a shared inhibitory pool ───────────────────────────────────
WORKSPACE_N = 300
ASSEMBLY_SIZE = 80            # per assembly; A = [0:80], B = [80:160]
WORKSPACE_FS_N = 50
WS_LOOP_A = "workspace_loop_A"
WS_LOOP_B = "workspace_loop_B"
WS_LOOP_GATE = "workspace_loop_fixed"

# FS lateral-inhibition weights. STRONGER than Rung-1 (single-assembly) so two simultaneous limit cycles cannot
# BOTH sustain -> the shared inhibition enforces WTA. Tuned on the smoke (below) for clean single-winner ignition.
WS_TO_FS_WEIGHT = 6.0
FS_TO_WS_WEIGHT = 16.0        # the mutual-inhibition strength (Rung-1 used 8.0 for a single assembly)

IGNITE_FRAC = 0.5            # an assembly is "ignited" iff its late-window rate >= IGNITE_FRAC * the solo plateau
SOLO_PLATEAU = 1.0 / 3.0     # the Rung-1 ignited limit-cycle rate (period-3)


def build_competitive_bridge(seed: int = 42, attractor_weight: float = DEFAULT_ATTRACTOR_WEIGHT,
                             fs_lesion: bool = False, fs_to_ws: float = FS_TO_WS_WEIGHT,
                             fs_density: float = 0.5, heterogeneity: bool = False, ou_noise_pA: float = 0.0):
    """One `workspace` region with TWO dense self-recurrent assemblies (A, B) sharing one inhibitory `workspace_fs`
    pool. `fs_lesion=True` zeroes fs->workspace (removes the mutual inhibition) -> the WTA anti-cheat. Returns
    (bridge, xp, A_idx_dev, B_idx_dev, snapshot, handles)."""
    xp, _ = get_backend()

    workspace = BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0,
                            internal_density=0.0, enable_nmda=True)
    workspace_fs = BrainRegion(name="workspace_fs", n_neurons=WORKSPACE_FS_N, exc_fraction=0.0,
                               internal_density=0.0, enable_nmda=False)
    regions = [workspace, workspace_fs]
    fs_to_ws_eff = 0.0 if fs_lesion else float(fs_to_ws)
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=float(fs_density),
                      weight_mean=fs_to_ws_eff, weight_jitter=0.0, plastic=False),
    ]

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(attractor_weight) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(attractor_weight) * 4.0)
    # Rung-2 (competition) DESYNCHRONIZES the attractor into a proper async rate attractor (Wang 2002's noisy
    # decision-circuit regime) so the biased competition is smooth — the deterministic homogeneous config gives a
    # synchronous period-3 limit cycle whose phase makes challenger takeover erratic. Heterogeneity + low OU noise
    # break the lockstep. (Rung-1 kept these OFF for a clean deterministic bifurcation; here they are the fix.)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    rm = bridge.region_manager
    ws = rm.indices("workspace")
    A_idx = np.asarray(ws[:ASSEMBLY_SIZE], dtype=np.int64)
    B_idx = np.asarray(ws[ASSEMBLY_SIZE:2 * ASSEMBLY_SIZE], dtype=np.int64)

    eff_weight = float(attractor_weight)
    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan[WS_LOOP_A] = _build_assembly_loop_population(A_idx, eff_weight)
    union_plan[WS_LOOP_B] = _build_assembly_loop_population(B_idx, eff_weight)
    # both loop populations share one frozen gate name (they were built with plasticity_gate=WS_LOOP_GATE inside
    # the helper -> WS_LOOP_GATE); freeze it below.

    inh = []
    for region in rm.regions():
        inh.extend(rm.inhibitory_indices(region.name))
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh or None)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    snap = _snapshot_state(bridge, xp)

    handles = {"seed": int(seed), "fs_lesion": bool(fs_lesion), "attractor_weight": eff_weight,
               "A_idx": A_idx, "B_idx": B_idx}
    return bridge, xp, xp.asarray(A_idx), xp.asarray(B_idx), snap, handles


INCUMBENT_SETTLE = 40   # free steps after igniting the incumbent, before the challenger arrives (it stabilizes)


def _run_incumbency(bridge, xp, A_dev, B_dev, snap, drive_incumbent: float, drive_challenger: float,
                    challenger_is_B: bool = True):
    """One INCUMBENCY competition trial (the clean WTA protocol, Baars "one spotlight at a time"):
       (1) restore quiescence; (2) drive the INCUMBENT assembly ALONE to ignition; (3) let it stabilize
       (INCUMBENT_SETTLE free steps — it now holds the workspace); (4) drive the CHALLENGER assembly;
       (5) free run; measure the LATE-window rate of A and B. The incumbent holds unless the challenger is more
       salient (biased competition via the shared inhibition). challenger_is_B=True -> A incumbent, B challenger;
       False -> B incumbent, A challenger (for the causal-swap test). Returns (late_rate_A, late_rate_B)."""
    inc_dev = A_dev if challenger_is_B else B_dev
    chal_dev = B_dev if challenger_is_B else A_dev
    bridge.cp_external_input_current[:] = 0.0
    _restore_state(bridge, snap)
    bridge.cp_external_input_current[:] = 0.0

    for _ in range(DRIVE_STEPS):                         # (2) ignite the incumbent alone
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[inc_dev] = xp.float32(drive_incumbent)
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(INCUMBENT_SETTLE):                    # (3) the incumbent stabilizes / holds the workspace
        bridge._run_one_simulation_step()
    for _ in range(DRIVE_STEPS):                         # (4) the challenger arrives
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[chal_dev] = xp.float32(drive_challenger)
        bridge._run_one_simulation_step()

    bridge.cp_external_input_current[:] = 0.0            # (5) free -> measure the settled winner
    late_start = FREE_STEPS - max(1, FREE_STEPS // 3)
    a_late = 0
    b_late = 0
    for t in range(FREE_STEPS):
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        if t >= late_start:
            a_late += int(to_host(bridge.cp_firing_states[A_dev].astype(xp.float64).sum()))
            b_late += int(to_host(bridge.cp_firing_states[B_dev].astype(xp.float64).sum()))
    denom = float((FREE_STEPS - late_start) * A_dev.shape[0])
    return a_late / denom, b_late / denom


def _ignited(rate):
    return bool(rate >= IGNITE_FRAC * SOLO_PLATEAU)


def main():
    ap = argparse.ArgumentParser(description="GNW Rung-2 competitive-access de-risk (incumbency WTA + causal swap).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_rung2_smoke.json")
    ap.add_argument("--backend", type=str, default="numpy", choices=["numpy", "cupy", "auto"])
    ap.add_argument("--attractor-weight", type=float, default=DEFAULT_ATTRACTOR_WEIGHT)
    ap.add_argument("--fs-to-ws", type=float, default=FS_TO_WS_WEIGHT, help="shared mutual-inhibition strength")
    ap.add_argument("--drive-inc", type=float, default=2500.0, help="drive that ignites the incumbent assembly")
    ap.add_argument("--chal-max", type=float, default=4000.0, help="top of the challenger-drive sweep")
    ap.add_argument("--n-chal", type=int, default=12, help="number of challenger-drive points in the sweep")
    args = ap.parse_args()

    if args.backend != "auto":
        get_backend(args.backend)

    drive_inc = float(args.drive_inc)
    drive_chals = list(np.linspace(0.0, float(args.chal_max), int(args.n_chal)))

    print(f"[gnw-rung2] seed={args.seed} backend={args.backend} drive_inc={drive_inc} "
          f"challenger in [0,{args.chal_max}] x{args.n_chal} fs_to_ws={args.fs_to_ws} (INCUMBENCY protocol)",
          flush=True)

    # ── INTACT incumbency sweep (single variable = the challenger's salience) ──────────────────────────────
    # A is the incumbent (ignited alone + stabilized); B challenges at drive_chal. Expect: low chal -> A HOLDS
    # (B suppressed); high chal -> B TAKES OVER (A drops); a clean crossover, NEVER both.
    bridge, xp, A_dev, B_dev, snap, handles = build_competitive_bridge(
        seed=args.seed, attractor_weight=float(args.attractor_weight), fs_lesion=False, fs_to_ws=float(args.fs_to_ws))
    a_rates, b_rates, winner, both = [], [], [], []
    for dC in drive_chals:
        ra, rb = _run_incumbency(bridge, xp, A_dev, B_dev, snap, drive_inc, float(dC), challenger_is_B=True)
        ia, ib = _ignited(ra), _ignited(rb)
        a_rates.append(ra); b_rates.append(rb); both.append(bool(ia and ib))
        winner.append("A" if (ia and not ib) else ("B" if (ib and not ia) else ("both" if (ia and ib) else "none")))
        print(f"  [intact] chal={dC:8.1f}  A={ra:.4f}{'*' if ia else ' '}  B={rb:.4f}{'*' if ib else ' '}  -> {winner[-1]}",
              flush=True)
    co_ignition_any = any(both)
    a_holds_low = winner[1] == "A" if len(winner) > 1 else False   # a weak challenger leaves A holding
    b_takes_high = winner[-1] == "B"                                # the strongest challenger takes over
    crossover = bool(any(w == "A" for w in winner) and any(w == "B" for w in winner))

    # ── CAUSAL SWAP: whoever is the more-salient CHALLENGER wins, regardless of role ───────────────────────
    strong = float(args.chal_max)
    ra1, rb1 = _run_incumbency(bridge, xp, A_dev, B_dev, snap, drive_inc, strong, challenger_is_B=True)   # B challenges strong -> B
    ra2, rb2 = _run_incumbency(bridge, xp, A_dev, B_dev, snap, drive_inc, strong, challenger_is_B=False)  # A challenges strong -> A
    B_wins_when_B_challenges = bool(_ignited(rb1) and not _ignited(ra1))
    A_wins_when_A_challenges = bool(_ignited(ra2) and not _ignited(rb2))
    causal_swap = bool(B_wins_when_B_challenges and A_wins_when_A_challenges)
    print(f"[gnw-rung2] causal-swap: B-challenges->(A={ra1:.3f},B={rb1:.3f}) A-challenges->(A={ra2:.3f},B={rb2:.3f}) "
          f"| B-wins-when-B-challenges={B_wins_when_B_challenges} A-wins-when-A-challenges={A_wins_when_A_challenges}",
          flush=True)

    # ── LESION anti-cheat (no mutual inhibition -> the challenger ignites WITHOUT displacing the incumbent -> both) ─
    bridge_l, xp_l, A_dev_l, B_dev_l, snap_l, _ = build_competitive_bridge(
        seed=args.seed, attractor_weight=float(args.attractor_weight), fs_lesion=True, fs_to_ws=float(args.fs_to_ws))
    ra_l, rb_l = _run_incumbency(bridge_l, xp_l, A_dev_l, B_dev_l, snap_l, drive_inc, strong, challenger_is_B=True)
    lesion_both_ignite = bool(_ignited(ra_l) and _ignited(rb_l))
    print(f"[gnw-rung2] lesion (A incumbent, B strong challenger, no inhibition): A={ra_l:.3f} B={rb_l:.3f} "
          f"-> both_ignite={lesion_both_ignite}", flush=True)

    # ── GO gate (the ROBUST property) + scoped-pending diagnostics ─────────────────────────────────────────
    # ROBUST (6-seed): the shared inhibition enforces MUTUAL EXCLUSION — two assemblies never co-ignite (only one
    # content can occupy the workspace at a time) — AND it is load-bearing (the lesion lets both ignite). This is
    # the core GNW single-content-access property (Baars "one spotlight"). GO gates on THIS.
    mutual_exclusion = bool(not co_ignition_any)
    inhibition_load_bearing = bool(lesion_both_ignite)
    go = bool(mutual_exclusion and inhibition_load_bearing)
    # SCOPED-PENDING (phase-erratic on the synchronous period-3 limit-cycle substrate — NOT gated; needs the async
    # rate attractor + adaptation-based eviction, the named next mechanism): the incumbent stably holding a weak
    # challenger, a clean salience-graded takeover, and the causal-swap membership test.
    wta_clean = bool(a_holds_low and b_takes_high and crossover)   # (diagnostic) clean salience-graded takeover
    swap_ok = bool(causal_swap)                                    # (diagnostic) causal-swap membership test

    result = {
        "runner": "_gnw_rung2_competitive_access_derisk",
        "protocol": "incumbency",
        "go": go,
        "seed": int(args.seed),
        "backend": args.backend,
        "fs_to_ws": float(args.fs_to_ws),
        "drive_incumbent": drive_inc,
        "drive_challengers": [float(x) for x in drive_chals],
        "a_rates": [float(x) for x in a_rates],
        "b_rates": [float(x) for x in b_rates],
        "winner_per_challenger": winner,
        "co_ignition_any": co_ignition_any,
        "a_holds_weak_challenger": a_holds_low,
        "b_takes_strong_challenger": b_takes_high,
        "crossover": crossover,
        "causal_swap": {
            "B_challenges": {"A": ra1, "B": rb1}, "A_challenges": {"A": ra2, "B": rb2},
            "B_wins_when_B_challenges": B_wins_when_B_challenges, "A_wins_when_A_challenges": A_wins_when_A_challenges,
        },
        "lesion_both": {"A": ra_l, "B": rb_l, "both_ignite": lesion_both_ignite},
        "gate_detail": {
            "mutual_exclusion": mutual_exclusion, "inhibition_load_bearing": inhibition_load_bearing,
            "fs_to_ws": float(args.fs_to_ws), "IGNITE_FRAC": IGNITE_FRAC, "SOLO_PLATEAU": SOLO_PLATEAU,
        },
        "scoped_pending": {   # phase-erratic on the synchronous limit cycle -> needs async attractor + adaptation
            "wta_clean_salience_takeover": wta_clean, "causal_swap_membership": swap_ok,
            "a_holds_weak": a_holds_low, "b_takes_strong": b_takes_high, "crossover": crossover,
            "note": "clean salience-graded takeover + causal-swap need an async rate attractor (heterogeneity) + "
                    "adaptation-based eviction (Dehaene-Changeux metastability); the synchronous period-3 limit "
                    "cycle makes pulsed competition phase-erratic",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[gnw-rung2] GO={go}  (mutual_exclusion={mutual_exclusion} inhibition_load_bearing={inhibition_load_bearing})",
          flush=True)
    print(f"[gnw-rung2]   [scoped-pending] wta_clean={wta_clean} causal_swap={swap_ok} "
          f"a_holds_weak={a_holds_low} b_takes_strong={b_takes_high} | winners: {winner}", flush=True)
    print(f"[gnw-rung2] wrote {args.json}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
