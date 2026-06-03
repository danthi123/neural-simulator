"""Content-selection Milestone 2 (Approach 3), cheap-first load-bearing test: can a SPIKING dlPFC
working-memory region hold a fading multi-concept conversational context the way the structured
ContextBuffer (Milestone 1) does?

Reuses the project's dlPFC working-memory configuration (recurrent self-excitation + NMDA bistability
for persistent activity; see g11_bg_runner.py dlpfc_wm) via a minimal one-region bridge. Drives a
sequence of concept patterns into the dlPFC region, reads the sustained firing after each turn, and
decodes which concepts are active (cosine of the firing pattern to each concept pattern). The
load-bearing question: is the read-out context a FADING SUPERPOSITION -- the most recently driven
concept strongest, earlier ones present but faded -- which is exactly what the structured context
buffer provides. If yes, the spiking dlPFC can replace the structured buffer (Milestone 2 proceeds);
if no, characterize the limit honestly.

Reuse-by-import only; no protected-module edits. GPU (CuPy) when available, else NumPy.

  python -m research.runners.content_selection_spiking --seed 42
"""
from __future__ import annotations
import argparse
import numpy as np


def generate_concept_patterns(n_concepts, n_pfc, pattern_size, seed=42):
    """Each concept = a distinct random subset of `pattern_size` dlPFC neuron indices (sparse,
    near-orthogonal codes -- distinct concepts overlap little)."""
    rng = np.random.default_rng(seed)
    return {i: rng.choice(n_pfc, size=pattern_size, replace=False) for i in range(n_concepts)}


def build_dlpfc_context_bridge(n_pfc=500, pfc_density=0.2, seed=42, plastic_recurrence=False,
                               hebbian=False, exc_weight=2.0, verbose=True):
    """Minimal region-framework bridge with a single recurrent NMDA-enabled dlPFC working-memory region,
    using the project's validated dlpfc_wm configuration. plastic_recurrence + hebbian enable attractor
    formation (shaping the recurrence so concept patterns self-sustain)."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion
    from sim.enums import NeuronType
    region = BrainRegion(
        name="dlpfc_wm", n_neurons=n_pfc, exc_fraction=0.8, internal_density=pfc_density,
        exc_weight_mean=exc_weight, inh_weight_mean=4.0, weight_jitter=0.2,
        plastic_internal=plastic_recurrence,
        izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [region]
    cfg.region_pathways = []
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True            # global NMDA on; only dlPFC (enable_nmda=True) gets bistability
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = bool(hebbian)
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 10.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if verbose:
        print(f"[dlpfc context bridge] {n_pfc} dlPFC neurons, recurrent density {pfc_density}, NMDA on",
              flush=True)
    return bridge


def build_loop_wm_bridge(n=400, density=0.1, loop_weight=4.0, loop_density=0.15, seed=42, verbose=True):
    """Two mutually-exciting regions forming a cortico-PFC LOOP (cortex_ctx <-> dlpfc_wm), both NMDA-
    enabled. The hypothesis (from the Milestone-2 standalone-region negative): persistent activity is
    sustained by reverberation around the loop, which a single recurrent region cannot do."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    def reg(name):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=0.8, internal_density=density,
                           exc_weight_mean=2.0, inh_weight_mean=4.0, weight_jitter=0.2,
                           plastic_internal=False,
                           izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name, enable_nmda=True)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [reg("cortex_ctx"), reg("dlpfc_wm")]
    cfg.region_pathways = [
        RegionPathway(from_region="cortex_ctx", to_region="dlpfc_wm", density=loop_density,
                      weight_mean=loop_weight, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="dlpfc_wm", to_region="cortex_ctx", density=loop_density,
                      weight_mean=loop_weight, weight_jitter=0.2, plastic=False),
    ]
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    if verbose:
        print(f"[loop WM bridge] cortex_ctx<->dlpfc_wm loop, {n} neurons each, loop weight {loop_weight}, "
              f"NMDA on", flush=True)
    return bridge


class SpikingContextBuffer:
    """Spiking analogue of the Milestone-1 ContextBuffer: the dlPFC region's sustained firing IS the
    discourse context. drive() injects a concept pattern and lets NMDA recurrence sustain it; read()
    samples the sustained firing; decode() maps it back to active concepts via cosine to the patterns."""

    def __init__(self, bridge, patterns, drive_pA=2500.0, stim_steps=50, settle_steps=20):
        import sim.backend as B
        self.B = B
        self.xp, _ = B.get_backend()
        self.bridge = bridge
        self.patterns = patterns
        self.drive_pA = drive_pA
        self.stim_steps = stim_steps
        self.settle_steps = settle_steps
        idx = bridge.region_manager.indices("dlpfc_wm")
        self.idx = self.xp.asarray(idx)
        self.n = len(idx)

    def drive(self, concept_id):
        xp = self.xp
        full = self.bridge.cp_external_input_current
        pat = self.idx[xp.asarray(self.patterns[concept_id])]
        for _ in range(self.stim_steps):
            full[:] = 0.0
            full[pat] = self.drive_pA
            self.bridge._run_one_simulation_step()
        full[:] = 0.0
        for _ in range(self.settle_steps):   # let the drive stop and NMDA recurrence take over
            self.bridge._run_one_simulation_step()

    def read(self, window=20):
        xp = self.xp
        acc = xp.zeros(self.n, dtype=xp.float32)
        for _ in range(window):
            self.bridge.cp_external_input_current[:] = 0.0
            self.bridge._run_one_simulation_step()
            acc += self.bridge.cp_firing_states[self.idx].astype(xp.float32)
        return self.B.to_host(acc)

    def decode(self, activity):
        a = activity / (np.linalg.norm(activity) + 1e-9)
        out = {}
        for cid, pat in self.patterns.items():
            v = np.zeros(self.n, dtype=np.float32)
            v[pat] = 1.0
            v /= (np.linalg.norm(v) + 1e-9)
            out[cid] = float(a @ v)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pfc", type=int, default=500)
    ap.add_argument("--pattern-size", type=int, default=50)
    a = ap.parse_args()

    bridge = build_dlpfc_context_bridge(n_pfc=a.n_pfc, seed=a.seed)
    patterns = generate_concept_patterns(3, a.n_pfc, a.pattern_size, seed=a.seed)
    scb = SpikingContextBuffer(bridge, patterns)

    print("Driving concepts c0 -> c1 -> c2 in sequence; reading the dlPFC context after each turn.")
    print("Expect a FADING SUPERPOSITION: the just-driven concept strongest, earlier ones present but")
    print("decaying.")
    history = []
    for c in [0, 1, 2]:
        scb.drive(c)
        act = scb.read()
        ctx = scb.decode(act)
        history.append(ctx)
        print(f"  after driving c{c}:  raw_firing_sum={act.sum():.0f}  " +
              "  ".join(f"c{k}={ctx[k]:+.2f}" for k in sorted(ctx)))

    # Load-bearing checks on the final state (after c0,c1,c2):
    final = history[-1]
    recent_strongest = final[2] >= final[1] >= final[0] - 1e-6   # c2 >= c1 >= c0 (fading order)
    recent_active = final[2] > 0.15                              # the just-driven concept is clearly present
    print("\nload-bearing checks (final context after c0,c1,c2):")
    print(f"  fading order c2>=c1>=c0 : {recent_strongest}   (c2={final[2]:.2f} c1={final[1]:.2f} c0={final[0]:.2f})")
    print(f"  most-recent clearly present (c2>0.15): {recent_active}")
    if recent_strongest and recent_active:
        print("\nVERDICT: RESOLVES -- the spiking dlPFC holds a fading multi-concept context like the "
              "structured buffer -> wire it into the controller (replace ContextBuffer) and re-run the "
              "Milestone-1 coherence eval.")
    else:
        print("\nVERDICT: does-not-cleanly-hold -- characterize: the spiking dlPFC does not reproduce the "
              "fading-superposition behavior at these params; tune drive/density/decay or report the limit.")


if __name__ == "__main__":
    main()
