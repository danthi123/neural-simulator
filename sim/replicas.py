"""Batched-replica framework (Session E.3).

Runs B independent replica simulations in a single bridge process via
**block-diagonal embedding**: a B×N super-network where each block is an
independent reservoir. Per-step compute amortizes across all B replicas
(one sparse matmul, one neuron-dynamics kernel, etc.) instead of B
separate processes each with its own Python+CuPy dispatch overhead.

Default OFF: when no replicas are configured, the bridge runs as a single
population unchanged.

Composes with E.1 (NeuromodulatorManager) and E.2 (BrainRegion / RegionManager).
Replicas can override per-replica neuromodulator parameters via
`scope="replica:N"` (a future extension of E.1's scope system).

See:
- docs/plans/2026-04-24-batched-replica-framework.md
- docs/plans/2026-04-24-cuda-cpp-step-kernel-feasibility.md (rejected alternative)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


@dataclass
class ReplicaConfig:
    """One replica = one independent simulation block.

    replica_id:
        Unique identifier (typically just the index 0, 1, 2, ...).
    seed_offset:
        Added to the bridge's main `seed` to produce this replica's RNG
        state. Different offsets -> different initial weights / neuron
        parameter heterogeneity within the same block-diagonal structure.
    neuromodulator_overrides:
        Dict[str, float] of per-replica overrides for neuromodulator
        parameters. Examples:
            {"ne_sensitivity": 60.0, "ne_threshold": 0.6}
        These are interpreted by the runner / NeuromodulatorManager
        when applying scope="replica:N" effects. Schema is intentionally
        loose at MVP scope; specific keys depend on the runner.
    """

    replica_id: int
    seed_offset: int = 0
    neuromodulator_overrides: Dict[str, float] = field(default_factory=dict)


class ReplicaManager:
    """Owns the per-replica index ranges in the block-diagonal super-network.

    Lifecycle:
        mgr = ReplicaManager(replicas, neurons_per_replica=N)
        mgr.initialize()
        mgr.indices(replica_id)             # neuron indices belonging to that block
        mgr.replica_for_neuron(idx)         # reverse lookup
        mgr.replica_indices_dict()          # for nm_mgr scope=replica:N

    The super-network is structured so block i occupies neurons
    [i*N, (i+1)*N). Each block's connectivity is identical in structure
    (same template); seed-jitter on weights makes them numerically
    independent. Cross-block synapses are forbidden (zero entries in
    the block-diagonal sparse matrix).
    """

    def __init__(self, replicas: Sequence[ReplicaConfig], neurons_per_replica: int):
        self._replicas: List[ReplicaConfig] = list(replicas)
        self._neurons_per_replica: int = int(neurons_per_replica)
        self._indices: Dict[int, List[int]] = {}
        self._total_neurons: int = 0

    def initialize(self) -> None:
        cursor = 0
        self._indices = {}
        N = self._neurons_per_replica
        for r in self._replicas:
            start = cursor
            end = cursor + N
            self._indices[r.replica_id] = list(range(start, end))
            cursor = end
        self._total_neurons = cursor

    def n_replicas(self) -> int:
        return len(self._replicas)

    def total_neurons(self) -> int:
        return self._total_neurons

    def neurons_per_replica(self) -> int:
        return self._neurons_per_replica

    def indices(self, replica_id: int) -> List[int]:
        if replica_id not in self._indices:
            raise KeyError(replica_id)
        return list(self._indices[replica_id])

    def replica_for_neuron(self, neuron_index: int) -> int:
        """Reverse lookup: which replica id contains the global neuron index?

        O(1) since blocks are uniform-size and contiguous.
        """
        if neuron_index < 0 or neuron_index >= self._total_neurons:
            raise IndexError(neuron_index)
        # Find which replica the index falls in. Replicas are stored in
        # the order they were declared; their indices are contiguous in
        # the order of declaration.
        replica_position = neuron_index // self._neurons_per_replica
        return self._replicas[replica_position].replica_id

    def replica_indices_dict(self) -> Dict[int, List[int]]:
        """{replica_id: [neuron indices]}. For NeuromodulatorManager set_group_indices
        when scope='replica:N' is desired (Task 4)."""
        return {r.replica_id: list(self._indices[r.replica_id]) for r in self._replicas}

    def replicas(self) -> List[ReplicaConfig]:
        return list(self._replicas)


# ---------- Wiring-plan replication ----------

def replicate_wiring_plan(
    template: Dict[str, dict],
    n_replicas: int,
    neurons_per_replica: int,
) -> Dict[str, dict]:
    """Tile a single-replica wiring plan into a B×N block-diagonal version.

    `template` is a wiring_plan dict in the format consumed by
    bridge.inject_explicit_wiring (the same format produced by g9_runner's
    _build_g9_plan and by RegionManager.build_wiring_plan). All neuron
    indices in the template must lie in [0, neurons_per_replica).

    For each group in the template, this function tiles the synapses
    `n_replicas` times, shifting indices by `replica_id * neurons_per_replica`.
    Weights are tiled exactly (no jitter — use replicate_wiring_plan_with_seeds
    for that).

    Returns a new wiring_plan dict with the same group keys, each with
    `count = original_count * n_replicas`.
    """
    out: Dict[str, dict] = {}
    for group_name, group in template.items():
        if not isinstance(group, dict) or "pre_indices" not in group:
            # Pass non-synapse metadata through (e.g. "layout")
            out[group_name] = group
            continue

        pre_orig = list(group["pre_indices"])
        post_orig = list(group["post_indices"])
        weights_orig = list(group["initial_weights"])
        n_orig = len(pre_orig)

        new_pre: List[int] = []
        new_post: List[int] = []
        new_w: List[float] = []
        for r in range(n_replicas):
            shift = r * neurons_per_replica
            new_pre.extend(int(p) + shift for p in pre_orig)
            new_post.extend(int(p) + shift for p in post_orig)
            new_w.extend(float(w) for w in weights_orig)

        out[group_name] = {
            "pre_indices": new_pre,
            "post_indices": new_post,
            "initial_weights": new_w,
            "plastic": bool(group.get("plastic", True)),
            "conn_type": group.get("conn_type", "MIXED"),
            "count": n_orig * n_replicas,
        }
        # Forward any pathway metadata (E.2 RegionPathway emits this)
        if "neuromodulator_gates" in group:
            out[group_name]["neuromodulator_gates"] = list(group["neuromodulator_gates"])
        # Forward plasticity_gate (per-pathway runtime-controllable gate).
        # Bug-fix 2026-04-30: this used to be silently dropped during
        # replication, causing replicated F v2 evals to lose the
        # `cerebellum_pf_pc` gate the runner depends on for CF-gated LTD.
        # See research/findings/2026-04-30-fv2-correction-replicated-runner-bug.md
        if "plasticity_gate" in group and group["plasticity_gate"]:
            out[group_name]["plasticity_gate"] = group["plasticity_gate"]

    return out


def replicate_wiring_plan_with_seeds(
    template: Dict[str, dict],
    replicas: Sequence[ReplicaConfig],
    neurons_per_replica: int,
    weight_jitter: float = 0.2,
) -> Dict[str, dict]:
    """Like replicate_wiring_plan but each replica gets its own seeded
    weight realization.

    For each replica, weights are drawn as
        w_replicated = w_template * (1 + jitter * normal(0, 1))
    using a Random(replica.seed_offset) per replica. Weights are clamped
    to >= 0.01 to avoid zero or negative values.

    With weight_jitter == 0, the result equals replicate_wiring_plan
    output (template tiled exactly).
    """
    out: Dict[str, dict] = {}
    for group_name, group in template.items():
        if not isinstance(group, dict) or "pre_indices" not in group:
            out[group_name] = group
            continue

        pre_orig = list(group["pre_indices"])
        post_orig = list(group["post_indices"])
        weights_orig = list(group["initial_weights"])
        n_orig = len(pre_orig)

        new_pre: List[int] = []
        new_post: List[int] = []
        new_w: List[float] = []
        for r_cfg in replicas:
            shift = r_cfg.replica_id * neurons_per_replica
            new_pre.extend(int(p) + shift for p in pre_orig)
            new_post.extend(int(p) + shift for p in post_orig)
            if weight_jitter <= 0:
                new_w.extend(float(w) for w in weights_orig)
            else:
                rng = random.Random(r_cfg.seed_offset)
                for w in weights_orig:
                    jittered = float(w) * (1.0 + rng.gauss(0.0, weight_jitter))
                    new_w.append(max(0.01, jittered))

        out[group_name] = {
            "pre_indices": new_pre,
            "post_indices": new_post,
            "initial_weights": new_w,
            "plastic": bool(group.get("plastic", True)),
            "conn_type": group.get("conn_type", "MIXED"),
            "count": n_orig * len(replicas),
        }
        if "neuromodulator_gates" in group:
            out[group_name]["neuromodulator_gates"] = list(group["neuromodulator_gates"])
        if "plasticity_gate" in group and group["plasticity_gate"]:
            out[group_name]["plasticity_gate"] = group["plasticity_gate"]

    return out
