"""Spiking two-compartment pyramidal (Larkum BAC; Guerguiev-Lillicrap-
Richards 2017 segregated dendrites). Per-neuron compartments: basal
(bottom-up forward drive), apical (top-down feedback through a FIXED
RANDOM projection -- feedback alignment, set once from seed, NEVER
learned, NO weight transport from forward weights), soma (BAC
integration: basal alone needs high threshold; apical depolarization
LOWERS the effective threshold). Pure numpy; biologically-local by
construction; learning is local Hebbian-style only (this module does
NO automatic differentiation, NO reverse-mode, NO computational
graph). ASCII only. Mirrors the SHAPE of sim/bptt_snn.LIFLayer but
does NOT import or modify it."""
from __future__ import annotations
import numpy as np


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class DendriticLayer:
    def __init__(self, n_pre, n_post, n_teacher, seed=0,
                 theta_high=1.0, apical_gain=0.5, leak=0.9):
        rng = np.random.default_rng(seed)
        self.W_basal = rng.normal(0.0, 1.0, (n_pre, n_post))
        # FIXED RANDOM apical feedback -- feedback alignment. Never
        # learned, never read from W_basal (no weight transport).
        self.B_apical = rng.normal(0.0, 1.0, (n_teacher, n_post))
        self.theta_high = float(theta_high)
        self.apical_gain = float(apical_gain)
        self.leak = float(leak)
        self.v_basal = np.zeros(n_post)
        self.v_apical = np.zeros(n_post)

    def _apical_drive(self, teacher):
        return np.asarray(teacher, float) @ self.B_apical

    def _apical_depol(self, teacher):
        # Larkum BAC Ca2+ plateau magnitude tracks the strength of the
        # apical input drive, not its signed net (the fixed-random
        # feedback sign is arbitrary; an excitatory teacher must
        # depolarize the apical compartment regardless of seed).
        return np.abs(self._apical_drive(teacher))

    def effective_threshold(self, teacher):
        # Larkum BAC: apical depolarization lowers the threshold.
        return self.theta_high - self.apical_gain * self._apical_depol(
            teacher)

    def step(self, x_basal, teacher):
        self.v_basal = (self.leak * self.v_basal
                        + np.asarray(x_basal, float) @ self.W_basal)
        self.v_apical = self._apical_drive(teacher)
        theta_eff = self.theta_high - self.apical_gain * np.abs(
            self.v_apical)
        soma_rate = _sig(self.v_basal - theta_eff)
        return {"soma_rate": soma_rate, "v_basal": self.v_basal.copy(),
                "v_apical": self.v_apical.copy(),
                "theta_eff": theta_eff}
