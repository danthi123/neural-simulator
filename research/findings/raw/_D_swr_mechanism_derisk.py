"""D cue-recall arc, cheap-first de-risk: does TEMPORAL-ORDERED SWR co-replay (A-before-B + STDP) build a DIRECTED
A->B association that propagates cue->associate, where SYMMETRIC co-firing (one-shot encoding) does not? This is the
crux of the heteroassociative-asymmetry hypothesis (SWR replays sequences in temporal order -> directed cortical
pathways; Buzsaki 2015 + McClelland CLS). Minimal substrate: 2 concept pools A,B + a plastic A->B pathway (zero-init)
+ STDP. Measure B's firing when A alone is driven (the cue-recall proxy), under: (baseline) no consolidation;
(symmetric) simultaneous A+B co-fire; (SWR) temporal-ordered A-then-B co-replay. GATE: SWR lifts cue->associate
firing well above baseline AND above symmetric. Anti-cheat: a B->A-only consolidation must NOT lift the A->B cue.
Design: docs/plans/2026-06-05-D-cue-recall-SWR-consolidation-design.md.

WIP 2026-06-05 (DIAGNOSED -- substrate works; mechanism understood):
- cp_connections is PRE x POST (the A->B pathway is the [A,B]=M[pre=A][:,post=B] block; my first read used [B,A]=0,
  a read bug now fixed). Connections exist (nnz 6400 = 80x80, init |w|~0.01).
- The substrate CAN learn: with enable_hebbian_learning=True, co-firing grows |A->B| 0.01->0.05. BUT Hebbian is
  SYMMETRIC -- symmetric/swr/swr_rev all grow ~equally (0.05/0.052/0.051), so it CANNOT test the temporal-order
  asymmetry the SWR hypothesis needs.
- enable_stdp ALONE does NOT change the weight (stays 0.01): the bridge's STDP forms an eligibility trace
  (reward-gated THREE-FACTOR), not a direct weight update. The timing-based (directed) STDP needs a consolidation
  REWARD signal to convert eligibility->weight.
- B still does not fire (cue-recall 0): the grown weight (~0.05) + synaptic propagation is too weak vs the 900 pA
  external drive.
NEXT (the precise continuation): (1) enable_stdp + inject a consolidation REWARD during co-replay (the three-factor
mechanism: timing-based eligibility x reward -> DIRECTED weight) -- find the reward setter (grep bridge for
current_reward_signal / reward application); (2) tune the weight ceiling + synaptic scaling so the grown A->B fires B
(the cue-recall readout); (3) THEN compare swr (A-before-B) vs swr_rev (B-before-A) -- the directed-association
asymmetry IS the hypothesis test. GPU for the real run.
"""
import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from sim.backend import to_host


def build(seed, pool=80):
    regions = [BrainRegion("A", pool, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0),
               BrainRegion("B", pool, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0)]
    pathways = [RegionPathway("A", "B", density=1.0, weight_mean=0.0, weight_jitter=0.0, plastic=True)]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.neuron_model_type = "IZHIKEVICH"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.enable_stdp = True
    cfg.stdp_w_max = 30.0
    cfg.enable_hebbian_learning = False
    cfg.ou_std_current_pA = 0.0
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


def _run(b, n):
    for _ in range(n):
        b._run_one_simulation_step()


def cue_recall_B(b, A, B_idx, drive_pA=900.0, window=60):
    """Drive A alone; measure B's mean firing (the cue->associate propagation via the A->B pathway) + A's firing
    (diagnostic: confirms the drive works)."""
    _drive(b, A, drive_pA)
    fb = fa = 0.0
    for _ in range(window):
        b._run_one_simulation_step()
        fs = np.asarray(to_host(b.cp_firing_states)).astype(float)
        fb += float(fs[B_idx].mean()); fa += float(fs[A].mean())
    b.cp_external_input_current[:] = 0.0
    _run(b, 40)
    return fb / window, fa / window


def consolidate(b, A, B_idx, mode, cycles=90, drive_pA=900.0):
    """mode: 'symmetric' = A+B together; 'swr' = A then B (temporal order, A leads -> A->B LTP); 'swr_rev' = B then A."""
    for _ in range(cycles):
        if mode == "symmetric":
            _drive(b, np.concatenate([A, B_idx])); _run(b, 12)
        elif mode == "swr":
            _drive(b, A); _run(b, 6)
            _drive(b, np.concatenate([A, B_idx])); _run(b, 8)        # A already spiking, B joins -> A precedes B
        elif mode == "swr_rev":
            _drive(b, B_idx); _run(b, 6)
            _drive(b, np.concatenate([A, B_idx])); _run(b, 8)        # B precedes A (anti-cheat: should NOT build A->B)
        b.cp_external_input_current[:] = 0.0
        _run(b, 10)


def _ab_weight(b, A, B_idx):
    """Mean/max |A->B| connection weight. cp_connections is PRE x POST (verified: the A->B pathway is the [A,B]
    block = M[pre=A][:, post=B]), NOT post x pre."""
    M = to_host(b.cp_connections)
    sub = M[np.asarray(A)][:, np.asarray(B_idx)]
    d = np.asarray(sub.todense() if hasattr(sub, "todense") else sub, dtype=float)
    return float(np.abs(d).mean()), float(np.abs(d).max())


def run(seed):
    out = {}
    for mode in ("baseline", "symmetric", "swr", "swr_rev"):
        b = build(seed)
        rm = b.region_manager
        A = np.asarray(rm.indices("A")); B_idx = np.asarray(rm.indices("B"))
        wpre = _ab_weight(b, A, B_idx)
        preb, prea = cue_recall_B(b, A, B_idx)
        if mode != "baseline":
            consolidate(b, A, B_idx, mode)
        wpost = _ab_weight(b, A, B_idx)
        postb, posta = cue_recall_B(b, A, B_idx)
        out[mode] = (preb, postb, posta, wpre, wpost)
    return out


if __name__ == "__main__":
    for seed in (42,):
        o = run(seed)
        for m, (pre, post, fa, wpre, wpost) in o.items():
            print(f"seed={seed} {m}: B {pre:.3f}->{post:.3f} (A={fa:.2f})  "
                  f"|A->B|w mean {wpre[0]:.3f}->{wpost[0]:.3f} max {wpre[1]:.2f}->{wpost[1]:.2f}", flush=True)
