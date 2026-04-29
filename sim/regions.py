"""Brain-region framework (Session E.2).

A first-class framework for declaring multiple cortical / subcortical
populations that share a single SimulationBridge. Each `BrainRegion`
owns a contiguous slice of the neuron-index space; each `RegionPathway`
declares cross-region projections with optional neuromodulator gating.

Default OFF: when CoreSimConfig.brain_regions is empty (which is the
default), the bridge runs as a single population — today's behavior
unchanged.

Composes with sim/neuromodulators.py from Session E.1: pathways can
declare `neuromodulator_gates=["dopamine"]` to make their plasticity
rate depend on a specific neuromodulator's concentration. Regions
register themselves as neuron groups with the NeuromodulatorManager
so target scope `group:NAME` resolves naturally.

See:
- docs/plans/2026-04-24-brain-region-framework.md
- sim/neuromodulators.py (E.1, composes with this)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class BrainRegion:
    """One brain region: a population of neurons with local connectivity.

    name:
        Unique identifier. Also registered as a neuron-group name with
        the experiment engine and neuromodulator manager so target
        scopes like `group:PFC` resolve here.

    n_neurons:
        Number of neurons. Allocated as a contiguous slice of the
        global neuron-index space; concatenation order matches the
        order in core_config.brain_regions.

    exc_fraction:
        Fraction excitatory (rest inhibitory). 0.8 matches cortical
        layer 2/3 (Markram et al. 2015).

    internal_density:
        Fraction of all-pairs internal connections that exist
        (sparse Erdős-Rényi within the region).

    exc_weight_mean, inh_weight_mean:
        Mean weight of internal excitatory / inhibitory connections.

    weight_jitter:
        Relative std of normal noise around the means (0.2 = 20%).

    plastic_internal:
        Whether internal synapses are plastic (subject to STDP and
        reward modulation). False (reservoir style) for sensorimotor
        regions; True for cortical learning regions like PFC working
        memory.

    nm_outputs:
        List[str] of neuromodulator names this region produces. Used
        by future `from_region_activity` production rules. Currently
        informational; integrates with neuromodulator subsystem
        production rules in a later task.
    """

    name: str
    n_neurons: int
    exc_fraction: float = 0.8
    internal_density: float = 0.1
    exc_weight_mean: float = 0.3
    inh_weight_mean: float = 0.8
    weight_jitter: float = 0.2
    plastic_internal: bool = False
    nm_outputs: List[str] = field(default_factory=list)
    # Per-region neuron type override. If set, the bridge uses this
    # NeuronType enum name when initializing neurons in this region's
    # index slice. Allows e.g. striatum_D1 region to use IZH2007_STRIATAL_MSN_D1
    # while motor region uses IZH2007_RS_CORTICAL_PYRAMIDAL.
    # Falls back to cfg.default_neuron_type_izh / _hh / _adex if None.
    # 2026-04-25: required for Phase B (BG action selection module).
    izh_neuron_type: str = None
    hh_neuron_type: str = None
    adex_neuron_type: str = None

    # Per-region GABA_A reversal potential override in mV. None = use global
    # cfg.syn_reversal_potential_i. Used to model regions with different
    # chloride homeostasis (e.g., striatal MSNs ~−60 mV per PBR-160 ch 6;
    # SNc DA ~−55 mV per ch 11). MSNs lack the deep negative ECl seen in
    # cortical pyramidals: gramicidin perforated patch measurements give
    # ~-60 mV, producing shunting (depolarizing-near-rest, hyperpolarizing-
    # near-threshold) inhibition. SNc DA neurons lack KCC2 entirely.
    syn_reversal_potential_i_override: Optional[float] = None

    # Cluster C v2 (2026-04-29): per-action DA compartmentalization.
    # When a region is action-specific (cortex_X, str_D1_X, str_D2_X,
    # gpi_X, thal_X, motor_X, etc), this is the action index in [0, N-1]
    # corresponding to the action channel. None for global / non-action-
    # specific regions (sensory, place_cells, stn, dopamine, hippocampus,
    # PFC, etc.).
    #
    # Used by inject_explicit_wiring() to populate cp_synapse_action_tag
    # so per-action DA modulators can target only synapses with their
    # action_index. See docs/plans/2026-04-29-cluster-c-v2-compartmentalized-
    # da-design.md.
    action_index: Optional[int] = None


@dataclass
class RegionPathway:
    """Directed projection from one region to another.

    from_region, to_region:
        BrainRegion.name strings. Both must exist in
        core_config.brain_regions.

    density:
        Fraction of pre-post pairs that have a synapse.

    weight_mean, weight_jitter:
        Mean weight + relative std (default 0.2 = 20%) of pathway
        synapses.

    plastic:
        Whether pathway synapses are plastic (subject to STDP +
        reward modulation). Cross-region projections default True
        so learning rules can shape them.

    neuromodulator_gates:
        List[str] of neuromodulator names that gate this pathway's
        plasticity rate. Each named modulator's
        `compute_plasticity_rate_multiplier()` contribution is
        multiplied with the global rate. Empty = no gating.

        Biological analogue: D1 corticostriatal LTP is gated by
        phasic dopamine; cortical LTP can be gated by acetylcholine
        attention signals. This field implements that as a config
        knob.

    plasticity_gate:
        Optional name for a runtime-controllable plasticity gate. When
        set, all synapses in this pathway share a per-synapse plasticity
        gain that defaults to 1.0 (full plasticity) and can be modified
        at runtime via `bridge.set_plasticity_gate(name, value)`. Setting
        the gain to 0.0 freezes the pathway (no STDP, no eligibility
        accumulation, no reward-driven updates). Setting it back to 1.0
        thaws.

        Biological analogue: developmental staging (sensory cortex
        matures before association cortex), critical periods (visual
        cortex ocular dominance plasticity closes via PV interneuron
        maturation), and neuromodulator-gated plasticity windows. The
        gate is the abstraction; what controls it (a fixed schedule, a
        neuromodulator concentration, a developmental clock) is up to
        the runner / experiment configuration.

        None = always-on (current behavior, not added to any gate).
    """

    from_region: str
    to_region: str
    density: float = 0.5
    weight_mean: float = 1.0
    weight_jitter: float = 0.2
    plastic: bool = True
    neuromodulator_gates: List[str] = field(default_factory=list)
    plasticity_gate: str = None


class RegionManager:
    """Owns per-region neuron-index allocation, inhibitory cell selection,
    and (later) wiring-plan generation.

    Lifecycle:
        mgr = RegionManager(regions, pathways)
        mgr.initialize(seed=42)              # allocate index ranges + inh
        plan = mgr.build_wiring_plan(rng=...)  # used by bridge.inject_explicit_wiring (Task 3+)
        mgr.region_indices_dict()             # for nm_mgr.set_group_indices

    Backward compat: an empty regions list yields total_neurons() == 0
    and an empty wiring plan, so the bridge falls through to the legacy
    single-population path.
    """

    def __init__(self,
                 regions: Sequence[BrainRegion],
                 pathways: Sequence[RegionPathway]):
        self._regions: List[BrainRegion] = list(regions)
        self._pathways: List[RegionPathway] = list(pathways)
        self._indices: Dict[str, List[int]] = {}
        self._inhibitory: Dict[str, List[int]] = {}
        self._total_neurons: int = 0

    def initialize(self, seed: int = 0) -> None:
        """Allocate contiguous index ranges for each region and pick
        inhibitory cells deterministically from `seed`."""
        rng = random.Random(seed)
        cursor = 0
        self._indices = {}
        self._inhibitory = {}
        for region in self._regions:
            start = cursor
            end = cursor + int(region.n_neurons)
            idx_list = list(range(start, end))
            self._indices[region.name] = idx_list

            # Pick inhibitory subset deterministically
            n_inh = int(round((1.0 - region.exc_fraction) * region.n_neurons))
            n_inh = max(0, min(region.n_neurons, n_inh))
            inh_chosen = sorted(rng.sample(idx_list, n_inh)) if n_inh > 0 else []
            self._inhibitory[region.name] = inh_chosen

            cursor = end
        self._total_neurons = cursor

    def total_neurons(self) -> int:
        return self._total_neurons

    def indices(self, region_name: str) -> List[int]:
        if region_name not in self._indices:
            raise KeyError(region_name)
        return list(self._indices[region_name])

    def inhibitory_indices(self, region_name: str) -> List[int]:
        if region_name not in self._inhibitory:
            raise KeyError(region_name)
        return list(self._inhibitory[region_name])

    def region_indices_dict(self) -> Dict[str, List[int]]:
        """Returns {name: indices} suitable for
        sim.neuromodulators.NeuromodulatorManager.set_group_indices().
        """
        return {name: list(idx) for name, idx in self._indices.items()}

    def regions(self) -> List[BrainRegion]:
        return list(self._regions)

    def pathways(self) -> List[RegionPathway]:
        return list(self._pathways)

    def build_wiring_plan(self, seed: int = 0) -> Dict[str, dict]:
        """Build a `wiring_plan` dict in the format consumed by
        bridge.inject_explicit_wiring().

        Each entry is one population of synapses with shape:
            {
                "pre_indices": [int, ...],
                "post_indices": [int, ...],
                "initial_weights": [float, ...],
                "plastic": bool,
                "conn_type": str,
                "count": int,
            }

        Population names:
            "{region}_internal"           — sparse internal connectivity
            "pathway_{from}_to_{to}"      — cross-region projection

        Determinism: rng seeded from `seed`. Independent of initialize()'s
        seed so the same RegionManager can re-build with different seeds.
        """
        if self._total_neurons == 0:
            return {}

        rng = random.Random(seed)
        plan: Dict[str, dict] = {}

        # ----- Internal connectivity per region -----
        for region in self._regions:
            entry = self._build_region_internal(region, rng)
            if entry is None:
                continue
            plan[f"{region.name}_internal"] = entry

        # ----- Cross-region pathways -----
        for pw in self._pathways:
            if pw.from_region not in self._indices:
                raise KeyError(pw.from_region)
            if pw.to_region not in self._indices:
                raise KeyError(pw.to_region)
            entry = self._build_pathway(pw, rng)
            if entry is None:
                continue
            plan[f"pathway_{pw.from_region}_to_{pw.to_region}"] = entry

        return plan

    def _build_region_internal(self, region: BrainRegion,
                                rng: random.Random) -> dict:
        """Sparse Erdős-Rényi internal connectivity for a region.

        Each ordered (pre, post) pair (pre != post) within the region is
        included with probability `region.internal_density`.
        """
        if region.n_neurons <= 1 or region.internal_density <= 0:
            return None

        idx = self._indices[region.name]
        inh = set(self._inhibitory[region.name])
        density = region.internal_density

        pre_list: List[int] = []
        post_list: List[int] = []
        weights: List[float] = []
        for pre in idx:
            base_w = region.inh_weight_mean if pre in inh else region.exc_weight_mean
            jitter = region.weight_jitter
            for post in idx:
                if pre == post:
                    continue
                if rng.random() < density:
                    pre_list.append(int(pre))
                    post_list.append(int(post))
                    if jitter > 0:
                        w = base_w * (1.0 + rng.gauss(0.0, jitter))
                    else:
                        w = base_w
                    # Clamp to a reasonable positive minimum
                    weights.append(max(0.01, float(w)))

        if not pre_list:
            return None

        return {
            "pre_indices": pre_list,
            "post_indices": post_list,
            "initial_weights": weights,
            "plastic": bool(region.plastic_internal),
            "conn_type": "MIXED",
            "count": len(pre_list),
        }

    def _build_pathway(self, pw: RegionPathway, rng: random.Random) -> dict:
        """Sparse Erdős-Rényi connectivity for a directed cross-region pathway."""
        pre_idx = self._indices[pw.from_region]
        post_idx = self._indices[pw.to_region]
        if pw.density <= 0 or not pre_idx or not post_idx:
            return None

        pre_list: List[int] = []
        post_list: List[int] = []
        weights: List[float] = []
        for pre in pre_idx:
            for post in post_idx:
                if rng.random() < pw.density:
                    pre_list.append(int(pre))
                    post_list.append(int(post))
                    if pw.weight_jitter > 0:
                        w = pw.weight_mean * (1.0 + rng.gauss(0.0, pw.weight_jitter))
                    else:
                        w = pw.weight_mean
                    weights.append(max(0.01, float(w)))

        if not pre_list:
            return None

        return {
            "pre_indices": pre_list,
            "post_indices": post_list,
            "initial_weights": weights,
            "plastic": bool(pw.plastic),
            "conn_type": "E_TO_MIX",
            "count": len(pre_list),
            # Pathway-specific metadata used in Task 8 for plasticity gating
            "neuromodulator_gates": list(pw.neuromodulator_gates),
            # Per-pathway plasticity gate name (runtime-controllable). None = always-on.
            "plasticity_gate": pw.plasticity_gate,
        }
