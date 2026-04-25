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
from typing import Dict, List, Sequence


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
    """

    from_region: str
    to_region: str
    density: float = 0.5
    weight_mean: float = 1.0
    weight_jitter: float = 0.2
    plastic: bool = True
    neuromodulator_gates: List[str] = field(default_factory=list)


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
