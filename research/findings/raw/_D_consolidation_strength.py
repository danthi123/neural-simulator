"""D cue-recall arc, consolidation-strength de-risk (the PRIMARY hypothesis the coincidence three-factor CAN test):
does REPEATED SWR co-replay of an associated pair STRENGTHEN the A->B pathway enough to lift cue->associate recall
(drive A alone -> B fires), SPECIFICALLY (consolidating A->B must NOT lift the un-consolidated A->C)? Reuses the
VALIDATED three-factor rule from bio_three_factor.update_eligibility_and_weights (coincidence eligibility x dopamine),
re-implemented on a clean 3-pool substrate (A,B,C + plastic A->B, A->C). McClelland CLS: sleep replay consolidates
the association into a directed cortical pathway. GATE: post-consolidation A->B cue-recall >> baseline AND >> A->C
(specificity = the anti-cheat). Design: docs/plans/2026-06-05-D-cue-recall-SWR-consolidation-design.md.

RESULT 2026-06-05: the consolidation MECHANISM works -- the re-implemented coincidence three-factor grows A->B |w|
to 25 (the cap) via repeated co-replay (3/3 seeds). BUT cue-recall (B firing on A-drive) stays 0.000 even at |w|=25:
the synaptic PROPAGATION (A->B -> B firing) does not happen in this from-scratch region-framework substrate (A fires
only ~8% @ 900 pA; the summed A->B synaptic current never reaches B's threshold). So the WEIGHT-LEARNING layer is
SOLVED; the PROPAGATION/readout layer is the remaining blocker -- a synaptic-scaling/config difference vs the g*
runners, which DO propagate pool->pool (cortex->D1 etc.). The minimal-substrate path has now isolated every layer
(build / connections-orientation / weight-learning / propagation) but the propagation needs the g* synaptic config.
NEXT (stop the from-scratch path): build the SWR-consolidation de-risk ON a g* runner's bridge OR the v16 concept-pool
architecture (both have WORKING pool->pool propagation + the real 27.5% cue-recall baseline), and add the co-replay
consolidation phase there -- reuse working propagation + working three-factor, don't rebuild either from scratch.
"""
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from sim.backend import to_host


def build(seed, pool=80):
    regions = [BrainRegion(n, pool, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0)
               for n in ("A", "B", "C")]
    pathways = [RegionPathway("A", "B", density=1.0, weight_mean=0.0, weight_jitter=0.0, plastic=True),
                RegionPathway("A", "C", density=1.0, weight_mean=0.0, weight_jitter=0.0, plastic=True)]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.neuron_model_type = "IZHIKEVICH"
    cfg.seed = int(seed); cfg.dt_ms = 1.0; cfg.num_traits = 1
    cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.ou_std_current_pA = 0.0
    for f in ("enable_short_term_plasticity", "enable_structural_plasticity", "enable_homeostasis",
              "enable_reward_modulation", "enable_neuromodulator_subsystem"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def _drive(b, idx, pA=900.0):
    import sim.backend as B
    xp, _ = B.get_backend()
    b.cp_external_input_current[:] = 0.0
    if len(idx):
        b.cp_external_input_current[xp.asarray(idx)] = pA


def _collect(b, drive_idx, pools, steps):
    _drive(b, drive_idx)
    acc = [np.zeros(len(p)) for p in pools]
    for _ in range(steps):
        b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states)).astype(float)
        for k, p in enumerate(pools):
            acc[k] += fs[p]
    return acc


def cue_recall(b, A, T, window=60):
    """Drive A alone -> mean firing of target pool T (the cue->associate readout)."""
    (ft,) = _collect(b, A, [T], window)
    b.cp_external_input_current[:] = 0.0
    for _ in range(30):
        b._run_one_simulation_step()
    return float(ft.mean()) / window


def consolidate(b, A, B, cycles=60, lr=0.3, da=1.0, decay=0.9, wmax=25.0):
    """Re-implemented coincidence three-factor (bio_three_factor rule) on the A->B block: co-replay A+B, accumulate
    pre x post coincidence eligibility, w += lr*elig*da -> strengthen the A->B pathway."""
    nA, nB = len(A), len(B)
    elig = np.zeros((nA, nB))
    M = to_host(b.cp_connections)
    w = np.asarray(M[A][:, B].todense(), dtype=float)
    pre = np.repeat(A, nB); post = np.tile(B, nA)
    for _ in range(cycles):
        fa, fb = _collect(b, np.concatenate([A, B]), [A, B], 12)
        b.cp_external_input_current[:] = 0.0
        for _ in range(8):
            b._run_one_simulation_step()
        elig = elig * decay + np.outer((fa > 0).astype(float), (fb > 0).astype(float))
        w = np.clip(w + lr * elig * da, 0.0, wmax)
        b.set_pathway_weights("ab", pre, post, w.flatten().astype(np.float32))
    return float(w.mean())


def run(seed):
    b = build(seed)
    rm = b.region_manager
    A = np.asarray(rm.indices("A")); B = np.asarray(rm.indices("B")); C = np.asarray(rm.indices("C"))
    pre_ab = cue_recall(b, A, B); pre_ac = cue_recall(b, A, C)
    wmean = consolidate(b, A, B)                       # consolidate A->B ONLY
    post_ab = cue_recall(b, A, B); post_ac = cue_recall(b, A, C)
    return pre_ab, post_ab, pre_ac, post_ac, wmean


if __name__ == "__main__":
    for seed in (42, 43, 44):
        pab, qab, pac, qac, wm = run(seed)
        print(f"seed={seed}: A->B cue {pab:.3f}->{qab:.3f}  (control A->C {pac:.3f}->{qac:.3f})  "
              f"A->B |w| {wm:.2f}  {'LIFT+SPECIFIC' if qab > 0.02 and qab > 3*max(qac,1e-3) else 'no'}", flush=True)
