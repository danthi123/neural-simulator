"""Pure helper for the biologized spiking mode-unification arc.

Returns N_GAMMA_SLOTS deterministic per-seed spike-phase position
symbols, generated via the same FHRR primitive (random uniform phases
quantised to spike times via `phases_to_spikes`) the validated
SpikingPhasorFHRR and ResonateFireFHRR use for vocabulary symbols.
The positions represent the gamma slots within one theta cycle
(Lisman-Idiart 1995; 7 slots is the biologically grounded value).

Reuse-by-import only; no protected/frozen module modified; no
automatic differentiation. Plain ASCII.
"""
from __future__ import annotations

from typing import List

import numpy as np

from research.runners.spiking_phasor_fhrr import phases_to_spikes


def gamma_slot_positions(seed: int, n_slots: int,
                          n_dim: int) -> List[np.ndarray]:
    """Return n_slots deterministic per-seed spike-phase position
    symbols, each of dimension n_dim. The positions are independently-
    seeded random uniform phases quantised to spike times via
    `phases_to_spikes` (the same mechanism the FHRR pipeline uses to
    construct any symbol). Deterministic in (seed, n_slots, n_dim);
    pairwise near-orthogonal at the FHRR capacity-curve regime."""
    rng = np.random.default_rng(int(seed))
    return [phases_to_spikes(rng.uniform(0.0, 1.0, size=int(n_dim)))
            for _ in range(int(n_slots))]
