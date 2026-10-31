"""C2 patient-pool inhibition: the shared builder + drive (step 0d validated knobs), plus the composer's OWN opt-in wiring.

Moved verbatim out of research/runners/c2_hops_step0.py (which re-imports `build_inhib_score_bridge` and
`onbridge_inhib_drive` from here, so its behaviour is byte-identical) so that `one_brain_composer` can use them without
importing a step-0 runner.

COMPOSER WIRING (owner ruling c2-s0d-score-1029), research-only, default OFF behind ONE env flag
`BRAIN_C2_COMPOSER_PATIENT_INHIB` in {1,true,yes,on}. Unset: nothing here is called, no extra bridge is built, and the
composer's score bridge, sequencer and drives are byte-identical. Set: the composer owns a SECOND score bridge
(`OneBrainComposer.patient_inhib_bridge()`, lazily built from the composer's own word-pool bridge: same layout, seed, sigma,
gain, per-neuron word-pool parameters copied from `_seq_score`) that has the external-only divisive stage
(`input_divisive_norm_external_only`), stable integration, and the shared FS `pinh` population with FF drive. The
sequencer's agent / action drives (`make_block_drives`, `_seq_drives`) and `_seq_score` are NOT touched; since
2026-10-31 `_ensure_sequencer` (flag ON only) also relays each block's PATIENT slice through this bridge into
`_seq_patient_drives` (`composer_patient_relay`, at the end of this file). No host argmax / top-k / threshold:
which pool fires is decided only by the spiking synapses (E->pinh, pinh->E, the FF drive into pinh).
"""
from __future__ import annotations

import os

COMPOSER_PATIENT_INHIB_ENV = "BRAIN_C2_COMPOSER_PATIENT_INHIB"
# the step-0d operating point (finding 2026-10-29-c2-step0b-patient-inhibition.md: ffx, I->E 2.0, FF 3e-4)
COMPOSER_KNOBS = {"n_inh": 30, "w_e2i": 60.0, "w_i2e": 2.0, "ff_gain": 3e-4, "settle": 20,
                  "izh_type": "IZH2007_FS_CORTICAL_INTERNEURON"}


def composer_patient_inhib_enabled() -> bool:
    """`BRAIN_C2_COMPOSER_PATIENT_INHIB` in {1,true,yes,on} (default OFF, byte-identical when unset)."""
    v = os.environ.get(COMPOSER_PATIENT_INHIB_ENV)
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


def build_inhib_score_bridge(seed, V, sigma, gain, n_word=20, n_inh=30, w_e2i=60.0, w_i2e=320.0,
                             izh_type="IZH2007_FS_CORTICAL_INTERNEURON", ref_sb=None, stable=False, w_relay=None, external_only=False):
    """`build_divnorm_score_bridge(seed, V, n_word, True, sigma, gain)` plus ONE shared FS inhibitory region `pinh`:
    w{w} -> pinh (excitatory, w_e2i) and pinh -> w{w} (inhibitory by the presynaptic region's exc_fraction=0, w_i2e).
    w_i2e=0 builds the same population and the same synapses with zero weight (the lesion arm)."""
    from sim import CoreSimConfig, SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
    from sim.enums import NeuronModel
    from sim.regions import BrainRegion, RegionPathway
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
    if external_only:   # sim/ flag (default OFF): divisive stage divides only the external drive, synaptic current undivided
        cfg.input_divisive_norm_external_only = True
    if stable:      # the SIM_STABLE_INTEGRATION set (SimulationBridge._apply_stable_integration), set explicitly
        cfg.enable_bounded_conductance_integration = True
        cfg.izh_membrane_substeps = 4
        cfg.izh_substep_refractory_rule = "model_reset"
    regions = [BrainRegion(name="w%d" % w, n_neurons=n_word, exc_fraction=1.0, internal_density=0.0,
                           input_divisive_norm=True) for w in range(V)]
    regions.append(BrainRegion(name="pinh", n_neurons=int(n_inh), exc_fraction=0.0, internal_density=0.0,
                               izh_neuron_type=izh_type))
    if w_relay:     # step 0c variant: an UNFLAGGED relay layer r{w} (the divnorm block divides every synaptic current onto the
        # flagged word pools by the pool-mean divisor, so inhibition onto them is ~1/D; the relay is read instead)
        regions += [BrainRegion(name="r%d" % w, n_neurons=n_word, exc_fraction=1.0, internal_density=0.0) for w in range(V)]
    cfg.brain_regions = regions
    P = []
    if w_relay:
        for w in range(V):
            P.append(RegionPathway(from_region="w%d" % w, to_region="r%d" % w, density=1.0, weight_mean=float(w_relay),
                                   weight_jitter=0.0, plastic=False))
            P.append(RegionPathway(from_region="pinh", to_region="r%d" % w, density=1.0, weight_mean=float(w_i2e),
                                   weight_jitter=0.0, plastic=False))
    for w in range(V):
        P.append(RegionPathway(from_region="w%d" % w, to_region="pinh", density=1.0, weight_mean=float(w_e2i),
                               weight_jitter=0.0, plastic=False))
        P.append(RegionPathway(from_region="pinh", to_region="w%d" % w, density=1.0, weight_mean=float(w_i2e),
                               weight_jitter=0.0, plastic=False))
    cfg.region_pathways = P
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                          gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    # The global heterogeneity overwrites a region's preset with RS-centred jitter (finding 2026-10-20 RS-centred
    # heterogeneity): re-apply the FS preset (sim/enums.py IZH2007_FS_CORTICAL_INTERNEURON, +-10% jitter, seeded) to the
    # `pinh` slice ONLY, and copy the word pools' per-neuron parameters from the composer's own score bridge (`ref_sb`), so the pools are the composer's.
    import numpy as np
    from sim.backend import to_host, from_host
    from research.runners._phaseC_S5_divnorm_derisk import region_idx
    fs = dict(cp_izh_C=20.0, cp_izh_k=1.0, cp_izh_vr=-55.0, cp_izh_vt=-40.0, cp_izh_vpeak=25.0, cp_izh_a=0.2,
              cp_izh_b=-2.0, cp_izh_c_reset=-45.0, cp_izh_d_increment=25.0)
    rng = np.random.RandomState(int(seed) + 7919)
    ii = region_idx(sb, "pinh")
    for attr, val in fs.items():
        arr = np.array(to_host(getattr(sb, attr)), dtype=np.float32)
        arr[ii] = (val * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=len(ii)))).astype(np.float32) if val > 0 else \
            (val * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, size=len(ii)))).astype(np.float32)
        if ref_sb is not None:                     # word pools: the composer's own per-neuron parameters, exactly
            nw = int(V) * int(n_word)
            arr[:nw] = np.array(to_host(getattr(ref_sb, attr)), dtype=np.float32)[:nw]
        setattr(sb, attr, from_host(arr))
    return sb


def onbridge_inhib_drive(score_sb, V, scores, input_gain, settle=20, ff_gain=0.0, trace=None, read="w"):
    """`onbridge_divnorm_drive` on the inhibition bridge (same drive, same settle, same read), plus the inhibitory
    population's spike count (ENGAGEMENT). Returns (acc_fired[V], inh_spikes). `ff_gain` > 0 (step 0c): the SAME external
    drive that goes into the word pools also goes into `pinh`, as ff_gain x the summed patient-slice drive (a fixed
    per-bridge gain; no normalization, no argmax). `trace` (dict) is filled with the first-spike step of every fired pool
    and of pinh, and the max |v| and max inhibitory conductance seen (physical validity)."""
    import numpy as np
    from sim.backend import to_host, from_host
    from research.runners._phaseC_S5_divnorm_derisk import _reset_score_bridge, region_idx
    _reset_score_bridge(score_sb)
    for _k in [k for k in vars(score_sb) if k.startswith("cp_conductance_")] + ["cp_prev_firing_states"]:
        _a = getattr(score_sb, _k, None)         # per-query reset of the synaptic state: `_reset_score_bridge` only resets
        if _a is not None:                       # v/u/firing, so inhibitory conductance leaked from block to block
            _a[:] = 0
    _pi = region_idx(score_sb, "pinh")             # the FS cells start at REST (vr), not at the spike-reset potential:
    score_sb.cp_membrane_potential_v[_pi] = score_sb.cp_izh_vr[_pi]      # c_reset > vt would fire them with no input
    s = np.maximum(np.asarray(scores, dtype=float), 0.0)
    cur = np.zeros(score_sb.core_config.num_neurons, dtype=np.float64)
    for w in range(V):
        drv = float(input_gain * s[w])
        if drv > 0:
            cur[region_idx(score_sb, "w%d" % w)] = drv
    acc = np.zeros(V)
    inh_idx = region_idx(score_sb, "pinh")
    inh_spikes = 0.0
    if ff_gain:
        cur[inh_idx] = float(ff_gain) * float(sum(input_gain * s[w] for w in range(V)))
    first = np.full(V, -1, dtype=int)
    first_inh, max_v, max_gi = -1, 0.0, 0.0
    cur = from_host(cur)
    for t in range(settle):
        score_sb.cp_external_input_current[:] = cur
        score_sb._run_one_simulation_step()
        fir = np.asarray(to_host(score_sb.cp_firing_states)).astype(float)
        for w in range(V):
            f = fir[region_idx(score_sb, "%s%d" % (read, w))].mean()
            acc[w] += f
            if f > 0 and first[w] < 0:
                first[w] = t
        ni = float(fir[inh_idx].sum())
        inh_spikes += ni
        if ni > 0 and first_inh < 0:
            first_inh = t
        if trace is not None:
            max_v = max(max_v, float(np.abs(np.asarray(to_host(score_sb.cp_membrane_potential_v))).max()))
            max_gi = max(max_gi, float(np.asarray(to_host(score_sb.cp_conductance_g_i)).max()))
    score_sb.cp_external_input_current[:] = 0.0
    if trace is not None:
        trace.update(first_spike_step={int(w): int(first[w]) for w in range(V) if first[w] >= 0},
                     first_spike_step_pinh=first_inh, max_abs_v=max_v, max_g_i=max_gi,
                     ff_current_pA=float(ff_gain) * float(sum(input_gain * s[w] for w in range(V))) if ff_gain else 0.0)
    return acc, inh_spikes


def composer_patient_inhib_bridge(buf, w_i2e=None):
    """The composer's patient-read inhibition bridge for `buf` (a OneBrainComposer), built lazily and cached on `buf`
    (`_seq_score_inhib`, keyed by (seed, V, sigma, gain, w_i2e)). `w_i2e=0` is the lesion bridge (same population, same
    synapses, zero I->E). Requires `buf._seq_score` (the composer's own word-pool bridge) for the per-neuron parameters."""
    k = COMPOSER_KNOBS
    w = float(k["w_i2e"] if w_i2e is None else w_i2e)
    key = (int(buf.seed), int(buf.V), float(buf.sequencer_sigma), float(buf.sequencer_gain), w)
    cache = getattr(buf, "_seq_score_inhib", None)
    if cache is None:
        cache = {}
        buf._seq_score_inhib = cache
    if key not in cache:
        cache[key] = build_inhib_score_bridge(
            buf.seed, int(buf.V), buf.sequencer_sigma, buf.sequencer_gain, n_inh=k["n_inh"], w_e2i=k["w_e2i"], w_i2e=w,
            izh_type=k["izh_type"], ref_sb=buf._seq_score, stable=True, external_only=True)
    return cache[key]


def composer_patient_inhib_read(buf, scores, w_i2e=None, trace=None):
    """The patient read on the composer's own inhibition bridge: returns (acc_fired[V], inh_spikes)."""
    k = COMPOSER_KNOBS
    return onbridge_inhib_drive(composer_patient_inhib_bridge(buf, w_i2e), int(buf.V), scores, buf.sequencer_input_gain,
                                settle=k["settle"], ff_gain=k["ff_gain"], trace=trace)


# PRODUCTION patient relay (2026-10-31; the flag must reach production, not only a side bridge). With the flag ON,
# `OneBrainComposer._ensure_sequencer` relays each block's PATIENT cleanup slice through the composer's patient-inhibition
# bridge and keeps the decoded-line patient drive per block in `_seq_patient_drives` (the agent / action drives
# `_seq_drives` are untouched). With the flag OFF nothing here runs inside production. The step-0 read (c) reads the
# production drives when the flag is ON and calls `composer_patient_relay` (the packet 2.3 relay) when it is OFF, so the
# instrument follows the flag.
COMPOSER_PATIENT_INHIB_W_I2E_ENV = "BRAIN_C2_COMPOSER_PATIENT_INHIB_W_I2E"
PATIENT_HI_PA = 1500.0          # the same fixed decoded-line drive `onbridge_divnorm_drive` emits for a fired pool


def composer_patient_inhib_w_i2e():
    """The production relay's I->E weight override (`BRAIN_C2_COMPOSER_PATIENT_INHIB_W_I2E`; 0 = the LESION arm: same
    population, same synapses, zero I->E). Unset = None = COMPOSER_KNOBS['w_i2e']."""
    v = os.environ.get(COMPOSER_PATIENT_INHIB_W_I2E_ENV)
    return None if v in (None, "") else float(v)


def composer_patient_relay(buf, pslice, trace=None):
    """The relay of ONE block's clipped patient cleanup slice into its decoded-line patient drive: (drive[V], acc_fired[V],
    info). Which pools are lit is decided only by which word pools SPIKE on the bridge (no host argmax / top-k / threshold
    on the scores); the host maps a fired pool to the fixed PATIENT_HI_PA line drive, as the agent / action relay does.
    Flag ON: the composer's inhibition bridge (`w_i2e` from `composer_patient_inhib_w_i2e`). Flag OFF: `onbridge_divnorm_drive`
    on the composer's own score bridge `_seq_score` at `sequencer_input_gain` (the packet 2.3 relay, byte-identical to the
    step-0 read (c) it replaces)."""
    import numpy as np
    V = int(buf.V)
    if composer_patient_inhib_enabled():
        w = composer_patient_inhib_w_i2e()
        acc, inh = composer_patient_inhib_read(buf, pslice, w_i2e=w, trace=trace)
        info = {"relay": "composer_patient_inhib", "w_i2e": float(COMPOSER_KNOBS["w_i2e"] if w is None else w),
                "inh_spikes": float(inh)}
    else:
        from research.runners._phaseC_S5_divnorm_derisk import onbridge_divnorm_drive
        _d, acc = onbridge_divnorm_drive(buf._seq_score, V, pslice, buf.sequencer_input_gain)
        info = {"relay": "onbridge_divnorm"}
    acc = np.asarray(acc, dtype=float)
    return np.where(acc > 0, PATIENT_HI_PA, 0.0), acc, info
