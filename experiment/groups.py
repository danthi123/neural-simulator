"""Neuron group management for experiments.

Manages designated neuron populations (input/output/hidden) with methods
to resolve group names to indices, populate groups from trait arrays,
and track group statistics.
"""

from sim.config import NeuronGroup
from sim.enums import NeuronGroupRole


class NeuronGroupManager:
    """Manages designated neuron populations (input/output/hidden).

    Provides methods to resolve group names to indices, populate groups
    from trait arrays, and track group statistics.
    """

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.groups = {}  # name -> NeuronGroup

    def initialize(self, group_defs, cp_traits=None, cp_module=None):
        """Set up neuron groups from definitions.

        Args:
            group_defs: List[NeuronGroup] definitions
            cp_traits: GPU array of neuron trait indices (for trait-based population)
            cp_module: CuPy module reference
        """
        for gdef in group_defs:
            group = NeuronGroup(
                name=gdef.name,
                role=gdef.role,
                neuron_indices=list(gdef.neuron_indices),
                trait_index=gdef.trait_index,
                index_start=gdef.index_start,
                index_end=gdef.index_end,
                fraction_of_trait=gdef.fraction_of_trait,
                highlight_color=list(gdef.highlight_color),
            )

            # Auto-populate from trait if indices not specified
            if not group.neuron_indices:
                if group.trait_index >= 0 and cp_traits is not None:
                    trait_np = cp_traits.get() if hasattr(cp_traits, 'get') else cp_traits
                    all_trait_indices = [int(i) for i in range(len(trait_np)) if int(trait_np[i]) == group.trait_index]

                    if group.fraction_of_trait < 1.0 and len(all_trait_indices) > 0:
                        import random as py_random
                        n_select = max(1, int(len(all_trait_indices) * group.fraction_of_trait))
                        group.neuron_indices = sorted(py_random.sample(all_trait_indices, n_select))
                    else:
                        group.neuron_indices = all_trait_indices

                elif group.index_end > group.index_start:
                    group.neuron_indices = list(range(
                        max(0, group.index_start),
                        min(self.n_neurons, group.index_end)
                    ))

            self.groups[group.name] = group

    def get_group(self, name):
        """Get a neuron group by name."""
        return self.groups.get(name)

    def get_groups_by_role(self, role):
        """Get all groups with a specific role."""
        return [g for g in self.groups.values() if g.role == role]

    def get_group_mask(self, name, cp_module):
        """Get a boolean GPU mask for a neuron group."""
        group = self.groups.get(name)
        if group is None or not group.neuron_indices:
            return cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
        mask = cp_module.zeros(self.n_neurons, dtype=cp_module.bool_)
        mask[cp_module.array(group.neuron_indices, dtype=cp_module.int32)] = True
        return mask

    def get_summary(self):
        """Get a summary of all groups for logging."""
        summary = {}
        for name, group in self.groups.items():
            summary[name] = {
                "role": group.role,
                "n_neurons": len(group.neuron_indices),
                "trait_index": group.trait_index,
            }
        return summary
