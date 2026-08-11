"""INTEGRATION #7 BURN-DOWN #2 -- the SPIKING conjunctive familiarity/source-monitor gate.

#7's no-confab moat for the PLASTICITY-LEARNED facts is `ConjunctiveFamiliarityGate`
(`_teacher_loop_contrastive_familiarity_moat_derisk`), a LEARNED anti-Hebbian projector that is CORRECT in
mechanism but HOST-implemented: it renders the (referent-percept x action) conjunction as a real vector and reads
novelty via `RealAntiHebbianFamiliarity` (a numpy Gram-Schmidt projector on the raw real cue -- see that class's
own docstring: "only the input adaptor differs ... no phasor cos/sin render").

This module supplies the SPIKING realization that closes burn-down #2. It is a drop-in for `ConjunctiveFamiliarity
Gate` -- SAME interface (`imprint`/`novelty`/`novelty_settled`/`familiar`/`lesion` over `(env, referent, action)`),
SAME NOV_GATE=0.5 -- but the abstain is now decided on SPIKES:

  (1) CUE ON SPIKE-PHASORS. The referent percept is mapped to phasor PHASES by a fixed complex random projection
      (the `_phase_map` bridge from `cortex_learned_cleanup_derisk`: a real sensory code -> a valid FHRR phase code
      spread across the [0,1) circle, structure-preserving so a noisy draw lands near its clean proto). Each action
      is a fixed random phase code. The conjunction is bound by the project's GENUINE resonate-and-fire spike-phasor
      neuron (`phase_sum_neuron`, Orchard Algorithm 1 -- time-stepped p/q integrators over spike trains), so any
      mismatch (untaught referent OR untaught action) shifts the whole spike-phasor cue -> novel.
  (2) SPIKING FAMILIARITY READOUT. Novelty is read by `AntiHebbianFamiliarity` (catalog D.04 perirhinal repetition
      suppression, the LEARNED spiking anti-Hebbian pool validated at V=320 in
      `2026-06-11-familiarity-gate-v320-GO.md`): the phase cue is rendered to the population's in-phase/quadrature
      drive `[cos(2*pi*phi), sin(2*pi*phi)]` (the standard resonate-and-fire real read of a phasor) and scored
      N(x)=||x||^2 - x^T W x. Familiar (taught) cue -> N~0 -> ACCEPT; novel cue -> N~1 -> ABSTAIN. LEARNED,
      lesionable (the abstain rides the projector W).

HONEST SEAM (declared, identical to the numpy gate + the v320 finding): the codebook -- the percept->phase complex
projection and the per-action phase codes -- is the composer-as-idealization host seam (what a learned cortex would
encode). The LOAD-BEARING learned, spiking, lesionable part is the anti-Hebbian projector W read via the I/Q render.

LESION SEMANTICS. `lesion()` FULLY clears the learned pool (re-creates the empty `AntiHebbianFamiliarity`), matching
`RealAntiHebbianFamiliarity.lesion()` (`_basis=[]`) exactly -- so #7's `_lesion_margin` (lesion -> margin collapses;
re-imprint -> restore) behaves identically to the numpy gate. (A pure zero-W lesion would leave `_basis` populated,
so a subsequent re-imprint of an already-in-span cue would NOT rebuild W -- see `AntiHebbianFamiliarity.imprint`'s
`nrm>1e-6` guard -- which would leave the gate lesioned for the post-hoc teeth that run after `_lesion_margin`.)

DISCIPLINE: reuse-by-import (the v320 spiking anti-Hebbian pool + the spike-phasor primitives). NO sim/ edit.
SIM_BACKEND=numpy. Deterministic per seed (`default_rng(seed+707)`, matching the numpy gate's stream offset).
`spike_bind=True` routes the cue through the genuine `phase_sum_neuron`; `spike_bind=False` uses the algebraically
identical modular phase-add (`phase_sum_neuron` provably computes `(phi_a+phi_b) mod T`; verified diff ~2e-3, the
spike-train integer-step rounding) as a fast oracle.
"""
from __future__ import annotations

import numpy as np

# reuse-by-import: the v320 SPIKING anti-Hebbian familiarity pool + the real-code->phasor-phase bridge.
from research.runners.cortex_learned_cleanup_derisk import AntiHebbianFamiliarity
# reuse-by-import: the project's validated spike-phasor primitives (genuine resonate-and-fire bind).
from research.runners.spiking_phasor_fhrr import phases_to_spikes, spikes_to_phases, phase_sum_neuron
# the same action vocabulary + a-priori novelty threshold the numpy gate uses (parity).
from research.runners._teacher_loop_contrastive_familiarity_moat_derisk import ACTIONS, NOV_GATE


class SpikingConjunctiveFamiliarityGate:
    """Drop-in SPIKING replacement for `ConjunctiveFamiliarityGate`. The learned, load-bearing part is the spiking
    anti-Hebbian projector `AntiHebbianFamiliarity`; the cue is a spike-phasor conjunction of the referent percept
    and the action. Interface (matched to the numpy gate, as #7 calls it):
        imprint(env, referent, action) / novelty(env, referent, action) / novelty_settled(...) /
        familiar(env, referent, action) -> bool / lesion().
    """

    def __init__(self, seed, d_p=12, D=256, spike_bind=True):
        rng = np.random.default_rng(seed + 707)          # SAME stream offset as ConjunctiveFamiliarityGate
        self.D = int(D)
        self.d_p = int(d_p)
        self.spike_bind = bool(spike_bind)
        # percept -> phasor PHASE via a fixed complex random projection (the _phase_map real-code->phase bridge):
        # angle(Wc @ centered_percept) spreads phases across the full circle while preserving cross-percept
        # structure, so a noisy draw of a taught referent maps near its clean proto's phase code.
        self._Wc = ((rng.standard_normal((self.D, self.d_p))
                     + 1j * rng.standard_normal((self.D, self.d_p))) / np.sqrt(2.0 * self.d_p))
        self.act_phase = {a: rng.uniform(0.0, 1.0, self.D) for a in ACTIONS}   # fixed per-action phase codes
        self.gate = AntiHebbianFamiliarity(self.D)       # the LEARNED spiking anti-Hebbian pool (2D I/Q read)

    # ---- cue rendering: the (referent-percept x action) conjunction as a spike-phasor phase code ----
    def _percept_phase(self, percept):
        z = self._Wc @ (np.asarray(percept, dtype=np.float64) - 0.5)          # center-surround DC removal (retina/LGN)
        return np.mod(np.angle(z) / (2.0 * np.pi), 1.0)

    def _cue(self, percept, action):
        """The conjunctive phase cue = bind(percept_phase, action_phase). Default routes through the genuine
        spike-phasor neuron `phase_sum_neuron`; the fast path is the algebraically identical modular phase-add."""
        pp = self._percept_phase(percept)
        ap = self.act_phase[action]
        if self.spike_bind:
            bound = phase_sum_neuron(phases_to_spikes(pp), phases_to_spikes(ap))   # resonate-and-fire bind (spikes)
            return spikes_to_phases(bound)
        return np.mod(pp + ap, 1.0)                                                # FHRR bind = modular phase-add

    def imprint(self, env, referent, action):
        # imprint the CLEAN prototype cue for the taught pair (one basis vector per taught conjunction)
        self.gate.imprint(self._cue(env.proto(referent), action))

    def novelty(self, env, referent, action):
        return self.gate.novelty(self._cue(env.draw(referent), action))

    def novelty_settled(self, env, referent, action, n=15):
        # a settled read: the source-monitor integrates a brief viewing (n glances), not one instantaneous sample.
        return float(np.mean([self.novelty(env, referent, action) for _ in range(n)]))

    def familiar(self, env, referent, action, n=15):
        return self.novelty_settled(env, referent, action, n) < NOV_GATE

    def lesion(self):
        # FULLY clear the learned pool (matches RealAntiHebbianFamiliarity.lesion's `_basis=[]`) so a subsequent
        # re-imprint rebuilds W -- #7's _lesion_margin lesions then restores.
        self.gate = AntiHebbianFamiliarity(self.D)
