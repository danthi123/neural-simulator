"""LEARNED basal-ganglia gate selection (cheat-removal item #3): the gate that opens is selected by a TRAINED
cortico-striatal pathway, not commanded by hand.

Item #2 (`gated_compose_bg_genuine_demo`) made the gate opened by a genuine D1 -| GPi -| thal disinhibition
cascade -- but WHICH D1 pool was driven (= which gate opens) was set by direct current (commanded). This removes
that: a plastic verb-cue -> striatal D1 pathway is TRAINED so the cue learns to select its own D1 pool.

  Training (supervised / embodied): co-drive the verb cue `verb_V` AND a teacher current on the CORRECT D1 pool
  `d1_V_{TRUE_MAP[V]}`. The two fire together; STDP (LTP biased) binds verb_V -> the correct D1.
  Inference: drive the verb cue ALONE (no teacher, no command). The learned verb -> D1 weight drives the correct
  D1 -> it silences its GPi -> disinhibits the thalamic relay -> opens the cortical route gate -> the verb routes
  to its motor. Selection is now LEARNED end-to-end; nothing is commanded at inference.

ANTI-CHEAT (the honest control): retrain from scratch with a PERMUTED teacher mapping. If the selection is
genuinely learned from the teacher (not a structural prior), the permuted run must bind the PERMUTED mapping, so
the TRUE mapping decodes at chance. A true-mapping accuracy that stays high under a permuted teacher would mean
the result is structural, not learned -- that would be the finding.

STATUS 2026-06-04 — PARTIAL (the LEARNING half is validated; end-to-end routing pending a synaptic-drive fix).
  - VALIDATED: cortico-striatal STDP selectively LEARNS the map. After 20 epochs of teacher-paired training the
    CORRECT verb->D1 synapse grows 0.5 -> ~15-18 while WRONG targets stay at the 0.5 init (verb_GO->d1_GO_N=18.2,
    verb_GO->d1_GO_S=0.5; verb_COME->d1_COME_S=15.6, verb_COME->d1_COME_N=0.5). Selection is genuinely LEARNED
    from the teacher, not commanded -- the scientific core of #3.
  - LOAD-BEARING BUG FOUND: `_run_one_simulation_step()` does NOT advance `current_time_ms` (the batch-run loop
    does, bridge.py:3179). Calling the step directly froze the clock, so every spike got timestamp 0, delta_t=0,
    and STDP was a SILENT no-op (weights frozen at exactly the init). `_step()` advances the clock; with it, STDP
    learns. (The #2 demo also called the step directly -- harmless there, no plasticity, the cascade still works.)
  - REMAINING GAP (engineering, not science): the learned weight does NOT drive the high-rheobase striatal MSN-D1
    to fire SYNAPTICALLY at inference (during training d1 fired from the 1500 pA teacher). This is the SAME wall #2
    sidestepped with direct current -- sel->d1 at weight 40 was also too weak to fire d1 there. CONTINUATION:
    scale the presynaptic drive the way the validated Tier-1 word->action recipe does (500-1000 neuron pools +
    motor FS), or insert a more-excitable cortico-striatal relay upstream of the MSN; then the learned cue fires
    its D1 -> the genuine #2 cascade -> the gate. The selection-LEARNING mechanism (the hard part) is done.

  SIM_BACKEND=numpy python -m research.runners.gated_compose_bg_learned_demo
"""
import numpy as np

from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronType
from research.runners.gated_compose_demo import VERBS, MOTORS, TRUE_MAP, decode  # noqa: F401
from research.runners.gated_compose_bg_genuine_demo import THAL_TONIC_PA, GPI_TONIC_PA

D1_W = 15.0       # D1 -| GPi (g11_bg scale; large weights explode g_i -> see cheat-#2 finding)
GPI_W = 8.0       # GPi -| thal
ROUTE_W = 40.0    # verb -> motor cortical route (gated)
VERB_TO_D1_INIT = 0.5   # plastic cortico-striatal init (STDP grows the correct one from here)


def build_learned_bg_gated_bridge(seed=42, n=30, n_verb=None):
    # n_verb scales the presynaptic verb (cue) pool independently of the cascade pools. The #3 close (2026-06-04)
    # found the learned weight (~16) fires the high-rheobase MSN-D1 SYNAPTICALLY only with a Tier-1-scale cue pool
    # (>=300; silent at 30/100) -- _msn_synaptic_drive_probe. Default keeps the 30-neuron toy; pass n_verb>=300 to
    # close the end-to-end.
    n_verb = n if n_verb is None else int(n_verb)
    from sim import SimulationBridge, CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    cfg = CoreSimConfig()
    cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_brain_region_framework = True
    cfg.ou_std_current_pA = 0.0
    # STDP ON (learns the cortico-striatal map); the others OFF for a clean reduced model.
    cfg.enable_stdp = True
    cfg.stdp_w_max = 50.0        # CLAUDE.md gotcha: soft-bound must sit ABOVE the design weight the cue must grow to
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation"):
        setattr(cfg, flag, False)

    pairs = [(v, m) for v in VERBS for m in MOTORS]
    cfg.brain_regions = (
        [BrainRegion(name=f"verb_{v}", n_neurons=n_verb, exc_fraction=1.0, internal_density=0.0) for v in VERBS]
        + [BrainRegion(name=f"motor_{m}", n_neurons=n, exc_fraction=1.0, internal_density=0.0) for m in MOTORS]
        + [BrainRegion(name=f"thal_{v}_{m}", n_neurons=n, exc_fraction=1.0, internal_density=0.0,
                       izh_neuron_type=NeuronType.IZH2007_THALAMIC_RELAY.name) for v, m in pairs]
        + [BrainRegion(name=f"d1_{v}_{m}", n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                       izh_neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1.name) for v, m in pairs]
        + [BrainRegion(name=f"gpi_{v}_{m}", n_neurons=n, exc_fraction=0.0, internal_density=0.0,
                       izh_neuron_type=NeuronType.IZH2007_GPI_OUTPUT.name) for v, m in pairs]
    )
    cfg.region_pathways = (
        # the LEARNED selection: verb cue -> ALL of its bindings' D1 pools, plastic (STDP), low init weight.
        [RegionPathway(from_region=f"verb_{v}", to_region=f"d1_{v}_{m}", density=1.0, weight_mean=VERB_TO_D1_INIT,
                       weight_jitter=0.3, plastic=True) for v, m in pairs]
        # the genuine #2 disinhibition cascade (g11_bg-scale weights):
        + [RegionPathway(from_region=f"d1_{v}_{m}", to_region=f"gpi_{v}_{m}", density=1.0, weight_mean=D1_W,
                         weight_jitter=0.0, plastic=False) for v, m in pairs]
        + [RegionPathway(from_region=f"gpi_{v}_{m}", to_region=f"thal_{v}_{m}", density=1.0, weight_mean=GPI_W,
                         weight_jitter=0.0, plastic=False) for v, m in pairs]
        # the cortical route, gated by the thalamic relay:
        + [RegionPathway(from_region=f"verb_{v}", to_region=f"motor_{m}", density=1.0, weight_mean=ROUTE_W,
                         weight_jitter=0.0, plastic=False, transmission_gate=f"g_{v}_{m}") for v, m in pairs]
    )
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                          runtime_state=RuntimeState(), gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    for v in VERBS:
        for m in MOTORS:
            sb.couple_gate_to_pool(f"g_{v}_{m}", f"thal_{v}_{m}", threshold=0.03)
    return sb


def _step(sb):
    """Step the sim AND advance the clock. `_run_one_simulation_step` does NOT advance current_time_ms (the
    batch-run loop does, bridge.py:3179) -- without this, every spike gets timestamp 0, delta_t=0, and STDP is a
    silent no-op (weights frozen). Load-bearing for any plasticity driven by calling the step directly."""
    sb._run_one_simulation_step()
    sb.runtime_state.current_time_ms += sb.core_config.dt_ms
    sb.runtime_state.current_time_step += 1


def _set_tonics(sb):
    sb.cp_external_input_current[:] = 0.0
    for v in VERBS:
        for m in MOTORS:
            sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"thal_{v}_{m}"))] = THAL_TONIC_PA
            sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"gpi_{v}_{m}"))] = GPI_TONIC_PA


def train(sb, mapping, epochs=20, steps_per_trial=30, verb_pA=1500.0, teacher_pA=1500.0):
    """Supervised: co-drive verb_V + teacher current on d1_{V, mapping[V]}; STDP binds verb -> correct D1."""
    rng = np.random.default_rng(sb.core_config.seed + 7)
    order = [(v, mapping[v]) for v in VERBS]
    for _ in range(epochs):
        rng.shuffle(order)
        for v, m in order:
            _set_tonics(sb)
            sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"verb_{v}"))] = verb_pA
            sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"d1_{v}_{m}"))] = teacher_pA
            for _ in range(steps_per_trial):
                _step(sb)


def decode_learned(sb, verb, settle=40, n_steps=60, verb_pA=1500.0):
    """Drive the verb cue ALONE (no teacher, no D1 command); the LEARNED verb->D1 selects the gate."""
    from sim.backend import to_host
    _set_tonics(sb)
    sb.cp_external_input_current[np.asarray(sb.region_manager.indices(f"verb_{verb}"))] = verb_pA
    for _ in range(settle):
        _step(sb)
    acc = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    for _ in range(n_steps):
        _step(sb)
        acc += to_host(sb.cp_firing_states).astype(np.float64)
    return max(MOTORS, key=lambda m: acc[np.asarray(sb.region_manager.indices(f"motor_{m}"))].mean())


def _eval(mapping, seeds=(42, 43, 44), epochs=20, n_verb=30):
    total_ok = 0
    lines = []
    for seed in seeds:
        sb = build_learned_bg_gated_bridge(seed=seed, n_verb=n_verb)
        train(sb, mapping, epochs=epochs)
        ok = 0
        per = []
        for v in VERBS:
            best = decode_learned(sb, v)
            correct = best == mapping[v]
            ok += int(correct)
            per.append(f"{v}->{best}{'(ok)' if correct else '(X)'}")
        total_ok += ok
        lines.append(f"  seed {seed}: {ok}/4   [{'  '.join(per)}]")
    return total_ok, lines


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cheat-removal #3: learned BG gate selection.")
    ap.add_argument("--n-verb", type=int, default=30,
                    help="presynaptic verb (cue) pool size; >=300 fires the MSN-D1 synaptically (Tier-1 scale)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    print(f"=== LEARNED BG gate selection (cheat-removal #3): cue learns to select its gate "
          f"| n_verb={args.n_verb} epochs={args.epochs} seeds={list(seeds)} ===\n", flush=True)

    print("TRUE teacher mapping {GO:N, COME:S, STOP:W, LOOK:E} -- train verb->D1, test verb ALONE:", flush=True)
    true_ok, true_lines = _eval(TRUE_MAP, seeds=seeds, epochs=args.epochs, n_verb=args.n_verb)
    for ln in true_lines:
        print(ln, flush=True)
    n_tot = 4 * len(seeds)
    print(f"  -> TRUE mapping: {true_ok}/{n_tot} across {len(seeds)} seeds\n", flush=True)

    # Anti-cheat: permuted teacher. A learned result must FOLLOW the teacher, so the TRUE mapping now decodes
    # at chance (the permuted mapping is what gets bound).
    permuted = {"GO": "S", "COME": "W", "STOP": "E", "LOOK": "N"}
    print(f"PERMUTED teacher mapping {permuted} -- score against the SAME true labels (anti-cheat):", flush=True)
    perm_true_ok = 0
    perm_perm_ok = 0
    for seed in seeds:
        sb = build_learned_bg_gated_bridge(seed=seed, n_verb=args.n_verb)
        train(sb, permuted, epochs=args.epochs)
        for v in VERBS:
            best = decode_learned(sb, v)
            perm_true_ok += int(best == TRUE_MAP[v])     # how often it still hits the TRUE label (should be ~chance)
            perm_perm_ok += int(best == permuted[v])     # how often it hits the PERMUTED (taught) label
    print(f"  -> under permuted teacher: TRUE-label hits {perm_true_ok}/{n_tot} (want ~chance={n_tot//4}/{n_tot}), "
          f"PERMUTED-label hits {perm_perm_ok}/{n_tot} (want high)\n", flush=True)

    learned = true_ok >= 0.75 * n_tot and perm_perm_ok >= 0.75 * n_tot and perm_true_ok <= 0.42 * n_tot
    print(f"  => {'LEARNED SELECTION (true high, follows permuted teacher, true-under-permuted ~chance)' if learned else 'NEEDS TUNING / NOT YET LEARNED'}",
          flush=True)


if __name__ == "__main__":
    main()
