"""CYCLE 104 — sentence-generation de-templating, cheap-first PHASE B (the SUBSTRATE test).

Phase A (pure numpy) showed the competitive-queuing serial-order mechanism + the fact-as-teacher work (held-out
true 1.000 vs permuted-order control 0.333, 6/6; no-learning control fails). PHASE B asks the substrate question
the G1 arc left open: does the order survive on REAL SPIKES? Mechanism: the CQ planning-layer primacy gradient is
realized as GRADED EXTERNAL CURRENT into the (driven, non-attractor) concept pools of one fact -- the
highest-primacy role (agent) gets the most current, patient the least -- and the spiking RATE tracks the drive, so
the per-pool rate RANKING = the emission order (rate-coded competitive queuing; robust, not delicate first-spike
latency). Read each pool's rate, order the fillers by rate, score the emitted order vs the permuted-ORDER control.

This is the spiking realization of the read-out the host f-string did. It reuses the validated driven-pool bridge
pattern (the on-bridge bind de-risk's inert-anchor trick: a non-attractor pool whose rate tracks current) + the
pre-registered anti-cheat harness (`song_g1_core` score_order / permuted_order_controls / g1_verdict, FIXED bars).
The NO-LEARNING control = EQUAL current to all three pools -> rates ~equal -> order ~random -> must FAIL (proves
the order comes from the primacy GRADIENT, not pool bias).

GATE (>=6 seeds, FIXED g1_verdict bars 0.10/0.5): GO if the emitted order clears floor 0.5 AND beats the permuted
control by >=10% AND the equal-drive control fails -> the SPIKING substrate produces the serial order (rate-coded
CQ) -> the on-bridge Option-1 sentence-generation build is viable. PARTIAL/NEGATIVE -> localize (the rate gradient
doesn't separate on the substrate). GPU (tiny bridge).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_serial_order_spiking_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners.song_g1_core import score_order, permuted_order_controls, g1_verdict  # noqa: E402

N_ROLES, VOCAB, N_FACTS, N_PERM = 3, 16, 24, 5
N_PER = 30               # neurons per concept pool
RUN_STEPS = 40           # drive+read window (rate = spikes / RUN_STEPS)
PRIMACY_pA = (2400.0, 1700.0, 1000.0)   # role primacy gradient as graded current (agent > action > patient)
EQUAL_pA = 1700.0        # the no-learning control: equal current to all 3 pools


def build_pool_bridge(seed):
    """One 'pools' region (VOCAB pools x N_PER, driven, non-attractor: rate tracks current) + an inert anchor
    region so the wiring plan is non-empty (avoids the empty-plan init bug). No pathways = clean f-I rate read."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="pools", n_neurons=VOCAB * N_PER, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="_anchor", n_neurons=4, exc_fraction=1.0, internal_density=1.0),  # inert: never driven
    ]
    cfg.region_pathways = []
    cfg.dt = 1.0
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    return b, np.asarray(b.region_manager.indices("pools"))


def pool_rates(bridge, pool_idx, drive_by_pool):
    """Drive each pool's N_PER neurons with its current; read per-pool spike rate over RUN_STEPS."""
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    cur = np.zeros(VOCAB * N_PER, np.float32)
    for c, pA in drive_by_pool.items():
        cur[c * N_PER:(c + 1) * N_PER] = pA
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[pool_idx] = xp.asarray(cur) if xp is not None else cur
    counts = np.zeros(int(bridge.core_config.num_neurons), np.float64)
    for _ in range(RUN_STEPS):
        bridge._run_one_simulation_step()
        counts += np.asarray(to_host(bridge.cp_firing_states)).astype(np.float64)
    bridge.cp_external_input_current[:] = 0.0
    rate = counts[pool_idx].reshape(VOCAB, N_PER).mean(1) / RUN_STEPS
    return rate


def build_facts(seed):
    rng = np.random.default_rng(seed * 101 + 7)
    facts, seen = [], set()
    while len(facts) < N_FACTS:
        trip = tuple(int(x) for x in rng.choice(VOCAB, N_ROLES, replace=False))
        if trip not in seen:
            seen.add(trip); facts.append(trip)
    return facts[N_FACTS // 2:]            # held-out half (no training needed: the primacy gradient is the frame)


def emit_order(bridge, pool_idx, trip, drive):
    """Drive the 3 filler pools with `drive` (role-graded or equal), read rates, emit fillers by rate DESC."""
    drive_by_pool = {trip[r]: drive[r] for r in range(N_ROLES)}
    rate = pool_rates(bridge, pool_idx, drive_by_pool)
    return [trip[r] for r in sorted(range(N_ROLES), key=lambda r: -rate[trip[r]])]


def run_seed(seed):
    bridge, pool_idx = build_pool_bridge(seed)
    held = build_facts(seed)
    rng = np.random.default_rng(seed * 71 + 3)
    trues, perms, c_trues, c_perms = [], [], [], []
    for trip in held:
        intended = list(trip)                                  # true SVO order = (agent, action, patient)
        emitted = emit_order(bridge, pool_idx, trip, PRIMACY_pA)            # primacy-graded drive
        controls = permuted_order_controls(intended, rng, N_PERM)
        trues.append(score_order(emitted, intended))
        perms.append(max((score_order(emitted, c) for c in controls), default=0.0))
        c_emit = emit_order(bridge, pool_idx, trip, (EQUAL_pA,) * N_ROLES)  # no-learning control: equal drive
        c_trues.append(score_order(c_emit, intended))
        c_perms.append(max((score_order(c_emit, c) for c in controls), default=0.0))
    t_true, t_perm = float(np.mean(trues)), float(np.mean(perms))
    c_true, c_perm = float(np.mean(c_trues)), float(np.mean(c_perms))
    v = g1_verdict(t_true, t_perm, gate_cleared=True)
    print(f"  [seed {seed}] SPIKING primacy true {t_true:.3f} vs perm {t_perm:.3f} -> {v['GATE']} | "
          f"equal-drive control true {c_true:.3f} vs perm {c_perm:.3f}", flush=True)
    return {"seed": seed, "true": t_true, "perm": t_perm, "gate": v["gate"], "ctrl_true": c_true, "ctrl_perm": c_perm}


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    print(f"[serial-order SPIKING de-risk PHASE B] does the spiking substrate (primacy gradient = graded current "
          f"-> rate ranking = emission order) produce the SVO order, beating the permuted-order control? "
          f"(host-template baseline = 1.000 by construction)", flush=True)
    rows = [run_seed(s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    t_true, t_perm, c_true, c_perm = m("true"), m("perm"), m("ctrl_true"), m("ctrl_perm")
    n_pass = sum(1 for r in rows if r["gate"])
    agg = g1_verdict(t_true, t_perm, gate_cleared=True)
    print(f"\n{'='*98}\n  MEAN (6 seeds): SPIKING primacy true {t_true:.3f} vs perm {t_perm:.3f} ({n_pass}/6 PASS) | "
          f"equal-drive control true {c_true:.3f} vs perm {c_perm:.3f} | aggregate {agg['GATE']} "
          f"({agg['pct_over_permuted']:.0f}% over perm)", flush=True)
    print(f"{'='*98}", flush=True)
    ctrl_ok = c_true < c_perm * 1.10 + 1e-9
    if agg["gate"] and n_pass >= 5 and ctrl_ok:
        print(f"  GO: the SPIKING substrate produces the serial order -- primacy-graded current -> rate ranking = "
              f"SVO order, true {t_true:.3f} >> permuted {t_perm:.3f} ({agg['pct_over_permuted']:.0f}% over, floor "
              f"{agg['abs_floor']}), {n_pass}/6 seeds, while the equal-drive control fails ({c_true:.3f} ~ "
              f"{c_perm:.3f}). ==> the rate-coded competitive-queuing serial-order read-out works ON SPIKES -> the "
              f"on-bridge Option-1 sentence-generation build (replace the f-string) is VIABLE.", flush=True)
    elif agg["gate"] and not ctrl_ok:
        print(f"  SUSPECT: passes but equal-drive control ALSO clears ({c_true:.3f} vs {c_perm:.3f}) -- pool bias, "
              f"not the gradient. Inspect.", flush=True)
    elif t_true >= t_perm * 1.10:
        print(f"  PARTIAL: real order signal on spikes (true {t_true:.3f} > perm {t_perm:.3f}) but below floor "
              f"{agg['abs_floor']} or <5/6 -- widen the primacy current gap / RUN_STEPS.", flush=True)
    else:
        print(f"  NEGATIVE: the rate gradient doesn't separate on the substrate ({t_true:.3f} vs {t_perm:.3f}) -- "
              f"the spiking serial-order read-out needs a different mechanism (latency / WTA-with-IoR). Localize.",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"true": t_true, "perm": t_perm, "n_pass": n_pass, "ctrl_true": c_true, "ctrl_perm": c_perm,
           "aggregate_gate": agg, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_serial_order_spiking.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
