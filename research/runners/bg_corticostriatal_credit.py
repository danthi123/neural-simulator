"""Corticostriatal three-factor credit on the PRODUCTION basal-ganglia organ (flag `BRAIN_BG_CREDIT`, default OFF).

The production SPEAK/STAY-SILENT organ (`bg_action_selection_production_organ.py`) runs the Gate-A v2 selector with a
host-supplied per-candidate salience current. This module adds, only when asked, what would let that selector LEARN
which channel a context favours: cortical context pools projecting onto the striatal D1/D2 MSNs through PLASTIC
corticostriatal synapses, credited by the bridge's own three-factor rule (coactivity eligibility tag x dopamine) with
the dopamine produced by a spiking SNc (reward_us -> SNc <- striosomal critic via GABA_B, read by the
`from_region_firing_signed` production rule). Optional: a working-memory pair that can hold a cue and bias the MSNs
through Mg2+-gated slow NMDA synapses.

Everything is added through `build_selector_bridge`'s own `extra_regions` / `extra_pathways` / `core_config_updates`
arguments; the selector's regions, pathways and constants are untouched and NO `sim/` file is edited. With the flag
unset nothing here is imported by the organ's default path (the byte-identical-off test pins that).

Host boundary (CLAUDE.md non-negotiable 1): the WORLD drives the context pools, the outcome screen and `reward_us`;
the BODY reads the committed motor pool. `cfg.current_reward_signal` stays 0: the third factor is the SNc's firing.
The SNc tonic set-point is measured from the network's own pacemaking (`calibrate_da`).

Preregistration: research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

from research.runners._vocal_action_selector_gate import (
    CHANNELS,
    DIRECT_PATH_GATE,
    _indices,
    _region,
    _set_equal_tonic_current,
    _step,
    build_selector_bridge,
    selector_config,
)
from sim.backend import get_backend, to_host
from sim.enums import NeuronType
from sim.regions import RegionPathway

FLAG = "BRAIN_BG_CREDIT"
CONTEXTS = ("ctx_a", "ctx_b", "ctx_x", "ctx_y", "ctx_n", "ctx_t")   # cues, distractors, neutral decoy, terminal
MEMORY = ("mem_a", "mem_b")
MSN = tuple("str_%s_%d" % (kind, ch) for kind in ("d1", "d2") for ch in CHANNELS)


def bg_credit_enabled() -> bool:
    """DEFAULT OFF. Only an explicit truthy `BRAIN_BG_CREDIT` turns the corticostriatal credit extension on."""
    v = os.environ.get(FLAG)
    return v is not None and v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class CreditParams:
    # cortical context pools (world-driven sensory representations)
    n_ctx: int = 24
    ctx_pA: float = 600.0
    w_ctx_msn: float = 20.0          # ctx -> str_d1/str_d2 (plastic); the selector's proposal -> MSN weight is 400
    ctx_msn_density: float = 0.5
    ctx_receptor: str = "nmda_slow"  # Mg2+-gated: biases MSNs the arousal barrage depolarized, cannot drive a race
    # working memory (D3 / Rubicon recipe)
    with_memory: bool = True
    n_mem: int = 30
    n_mem_fs: int = 20
    n_clear: int = 20
    clear_pA: float = 600.0
    w_cue_mem: float = 15.0
    w_mem_recur: float = 12.0
    mem_recur_density: float = 0.9
    w_mem_fs: float = 1.4
    w_fs_mem: float = 25.0
    w_clear_fs: float = 30.0
    w_mem_msn: float = 10.0          # mem -> str_d1/str_d2 (plastic, nmda_slow)
    mem_msn_density: float = 0.5
    # limbic core: outcome screen -> striosomal critic; reward_us -> SNc <- critic (GABA_B)
    n_outc: int = 24
    n_strio: int = 30
    n_rus: int = 20
    n_snc: int = 20
    w_outc_strio: float = 30.0
    w_rus_snc: float = 25.0
    w_strio_snc: float = 4.0
    rus_pA: float = 600.0
    snc_tonic_pA: float = 180.0
    gabab_prop: float = 0.22
    gabab_tau_ms: float = 150.0
    # three-factor rule
    lr: float = 5.0
    coact_scale: float = 0.05
    coact_pre_tau_ms: float = 20.0
    elig_tau_ms: float = 1000.0
    w_max: float = 90.0              # plastic-only clip (see `restrict_clip_to_plastic`)
    # A2 soft bounds (RULING a02-approved): flows into cfg.reward_soft_bounds via
    # credit_extension. 0 = OFF = byte-identical; exponents frozen at 1.0.
    soft_bounds: int = 0
    soft_bound_n_up: float = 1.0
    soft_bound_n_down: float = 1.0
    # dopamine production from SNc firing
    da_sensitivity: float = 80.0
    da_window_ms: float = 30.0
    da_decay_ms: float = 100.0
    da_baseline: float = 0.5


def _pw(a, b, w, dens, plastic, jit=0.1, **kw):
    return RegionPathway(from_region=a, to_region=b, density=float(dens), weight_mean=float(w),
                         weight_jitter=float(jit), plastic=bool(plastic), **kw)


def credit_extension(p: CreditParams, *, elig_tau_ms=None, tag_gain=1.0):
    """(extra_regions, extra_pathways, core_config_updates) for `build_selector_bridge`. `elig_tau_ms` / `tag_gain`
    implement the trace manipulations (equal-integral: tau x gain held fixed; equal-peak: gain 1)."""
    from sim.neuromodulators import NeuromodulatorConfig, ProductionRule
    rs = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL
    fs = NeuronType.IZH2007_FS_CORTICAL_INTERNEURON
    regions = [_region(name, p.n_ctx, exc_fraction=1.0, neuron_type=rs) for name in CONTEXTS]
    regions.append(_region("ctx_o", p.n_outc, exc_fraction=1.0, neuron_type=rs))
    paths = []
    for src in CONTEXTS:
        for dst in MSN:
            paths.append(_pw(src, dst, p.w_ctx_msn, p.ctx_msn_density, True, jit=0.2, exc_receptor=p.ctx_receptor))
    if p.with_memory:
        regions += [_region("mem_a", p.n_mem, exc_fraction=1.0, neuron_type=rs, enable_nmda=True),
                    _region("mem_b", p.n_mem, exc_fraction=1.0, neuron_type=rs, enable_nmda=True),
                    _region("mem_fs", p.n_mem_fs, exc_fraction=0.0, neuron_type=fs),
                    _region("mem_clear", p.n_clear, exc_fraction=1.0, neuron_type=rs)]
        paths += [_pw("ctx_a", "mem_a", p.w_cue_mem, 0.5, False), _pw("ctx_b", "mem_b", p.w_cue_mem, 0.5, False),
                  _pw("mem_a", "mem_a", p.w_mem_recur, p.mem_recur_density, False, jit=0.05, exc_receptor="nmda_slow"),
                  _pw("mem_b", "mem_b", p.w_mem_recur, p.mem_recur_density, False, jit=0.05, exc_receptor="nmda_slow"),
                  _pw("mem_a", "mem_fs", p.w_mem_fs, 0.6, False), _pw("mem_b", "mem_fs", p.w_mem_fs, 0.6, False),
                  _pw("mem_fs", "mem_a", p.w_fs_mem, 0.6, False), _pw("mem_fs", "mem_b", p.w_fs_mem, 0.6, False),
                  _pw("mem_clear", "mem_fs", p.w_clear_fs, 0.6, False)]
        for src in MEMORY:
            for dst in MSN:
                paths.append(_pw(src, dst, p.w_mem_msn, p.mem_msn_density, True, jit=0.2, exc_receptor="nmda_slow"))
    regions += [_region("striosome_value", p.n_strio, exc_fraction=0.0,
                        neuron_type=NeuronType.IZH2007_STRIATAL_MSN_D1),
                _region("reward_us", p.n_rus, exc_fraction=1.0, neuron_type=rs),
                _region("snc", p.n_snc, exc_fraction=1.0, neuron_type=NeuronType.IZH2007_DOPAMINE)]
    paths += [_pw("ctx_o", "striosome_value", p.w_outc_strio, 0.6, True, jit=0.5),
              _pw("reward_us", "snc", p.w_rus_snc, 0.6, False, jit=0.2),
              _pw("striosome_value", "snc", p.w_strio_snc, 0.5, False, jit=0.2, receptor="gaba_b")]
    tau = float(p.elig_tau_ms if elig_tau_ms is None else elig_tau_ms)
    updates = {
        "enable_reward_modulation": True,
        "reward_eligibility_from_coactivity": True,
        "reward_coactivity_trace_tau_ms": float(p.coact_pre_tau_ms),
        "reward_coactivity_scale": float(p.coact_scale) * float(tag_gain),
        "reward_eligibility_tau_ms": tau,
        "reward_learning_rate": float(p.lr),
        "reward_soft_bounds": int(getattr(p, "soft_bounds", 0)),
        "reward_soft_bound_n_up": float(getattr(p, "soft_bound_n_up", 1.0)),
        "reward_soft_bound_n_down": float(getattr(p, "soft_bound_n_down", 1.0)),
        "reward_baseline": 0.0,
        "current_reward_signal": 0.0,
        "hebbian_min_weight": 0.0,
        "hebbian_max_weight": float(p.w_max),
        "enable_d1_d2_asymmetry": True,
        "enable_nmda_recurrent": True,
        "nmda_recurrent_tau_decay_ms": 100.0,
        "nmda_recurrent_propagation_strength": 0.05,
        "enable_gabab": True,
        "gabab_reversal_potential": -90.0,
        "gabab_tau_decay": float(p.gabab_tau_ms),
        "gabab_propagation_strength": float(p.gabab_prop),
        "gabab_conductance_max": 0.0,
        "enable_neuromodulator_subsystem": True,
        # The DA deviation itself is the signed third factor: no plasticity_rate target (a second DA-level
        # multiplier adds an LTP-favouring asymmetry; standalone dev history, SPIKING-WORKLOG.md).
        "neuromodulators": [NeuromodulatorConfig(
            name="dopamine", baseline=float(p.da_baseline), decay_tau_ms=float(p.da_decay_ms),
            concentration_min=0.0, concentration_max=2.0, targets=[],
            production_rules=[ProductionRule(rule_type="from_region_firing_signed",
                                             sensitivity=float(p.da_sensitivity), threshold=0.0,
                                             window_ms=float(p.da_window_ms), source_regions=["snc"])])],
    }
    return regions, paths, updates


def restrict_clip_to_plastic(bridge):
    """With `enable_stdp=False` the reward block clips EVERY synapse to [hebbian_min, hebbian_max] unless a per-synapse
    plasticity gain exists, which would cut the selector's fixed 400-weight proposal -> MSN synapses. A gain equal to the
    plastic mask confines the clip (and every reward update) to the plastic pathways."""
    xp, _ = get_backend()
    pm = np.asarray(to_host(bridge.cp_synapse_plastic_mask)).astype(np.float32)
    cap = int(getattr(bridge, "_synapse_capacity", pm.size) or pm.size)
    gain = np.zeros(max(cap, pm.size), dtype=np.float32)
    gain[:pm.size] = pm
    bridge.cp_plasticity_rate_gain = xp.asarray(gain)


def build_credit_selector(seed, params: CreditParams = CreditParams(), *, elig_tau_ms=None, tag_gain=1.0,
                          version="v2"):
    """The production Gate-A v2 selector plus the credit extension. Returns (bridge, selector_config)."""
    config = selector_config(version)
    regions, paths, updates = credit_extension(params, elig_tau_ms=elig_tau_ms, tag_gain=tag_gain)
    bridge = build_selector_bridge(int(seed), config, commit_enable_nmda=True, extra_regions=regions,
                                   extra_pathways=paths, core_config_updates=updates)
    bridge.set_transmission_gate(DIRECT_PATH_GATE, 1.0)
    restrict_clip_to_plastic(bridge)
    return bridge, config


class Synapses:
    """Pathway masks over the CSR data order (rows = presynaptic, indices = postsynaptic)."""

    def __init__(self, bridge):
        self.b = bridge
        rm = bridge.region_manager
        indptr = np.asarray(to_host(bridge.cp_connections.indptr))
        indices = np.asarray(to_host(bridge.cp_connections.indices))
        nnz = int(bridge.cp_connections.nnz)
        n = len(indptr) - 1
        lab = np.empty(n, dtype=object)
        for reg in rm.regions():
            lab[np.asarray(rm.indices(reg.name), dtype=np.int64)] = reg.name
        rows = np.repeat(np.arange(n), np.diff(indptr))[:nnz]
        self.pre, self.post = lab[rows], lab[indices[:nnz]]
        self.nnz = nnz
        self._m = {}
        pm = np.asarray(to_host(bridge.cp_synapse_plastic_mask))[:nnz].astype(bool)
        self.plastic_idx = np.flatnonzero(pm)

    def mask(self, a, b):
        key = (a, b)
        if key not in self._m:
            self._m[key] = np.flatnonzero((self.pre == a) & (self.post == b))
        return self._m[key]

    def weights(self, a, b):
        return np.asarray(to_host(self.b.cp_connections.data))[:self.nnz][self.mask(a, b)]

    def zero(self, a, b):
        xp, _ = get_backend()
        m = self.mask(a, b)
        if m.size == 0:
            raise RuntimeError("no synapses %s -> %s" % (a, b))
        self.b.cp_connections.data[xp.asarray(m)] = 0.0
        return int(m.size)

    def eligibility(self):
        """The eligibility of every synapse (host copy, nnz-long)."""
        return np.asarray(to_host(self.b.cp_eligibility_trace))[:self.nnz].astype(np.float64)


def baseline_currents(bridge, config, p: CreditParams):
    """Equal tonic currents of the selector + the SNc pacemaker bias (intrinsic pacemaking of DA neurons)."""
    xp, _ = get_backend()
    _set_equal_tonic_current(bridge, config)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "snc"))] = xp.float32(p.snc_tonic_pA)


def drive(bridge, region, pA):
    xp, _ = get_backend()
    idx = xp.asarray(_indices(bridge, region))
    bridge.cp_external_input_current[idx] = bridge.cp_external_input_current[idx] + xp.float32(pA)


def calibrate_da(bridge, config, p: CreditParams, n_steps=20000):
    """Set the signed DA rule's threshold to the SNc's own tonic firing fraction, measured over 20 s of pacemaking
    with plasticity off, then return the concentration to its baseline."""
    cfg = bridge.core_config
    saved = cfg.reward_learning_rate
    cfg.reward_learning_rate = 0.0
    snc = _indices(bridge, "snc")
    baseline_currents(bridge, config, p)
    total = 0
    for k in range(int(n_steps)):
        _step(bridge)
        if k >= 1000:
            total += int(np.asarray(to_host(bridge.cp_firing_states))[snc].sum())
    frac = total / float(len(snc) * (int(n_steps) - 1000))
    cfg.neuromodulators[0].production_rules[0].threshold = frac
    nm = bridge.neuromodulator_manager
    nm._concentrations["dopamine"] = float(p.da_baseline)
    _step(bridge, 200)
    cfg.reward_learning_rate = saved
    return frac
