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

    # ---------------------------------------------------------------
    # ADDITIVE / DEFAULT-OFF extension (2026-08-09): a MULTIPLICATIVE
    # apical-basal COINCIDENCE (Larkum BAC / sigma-pi), distinct from
    # step()'s ADDITIVE threshold-shift. step() is UNCHANGED above, so
    # a caller that never invokes the two methods below observes the
    # byte-identical layer (guarded: reached only on explicit call).
    #
    # WHY step() is not a product: soma_rate = sig(v_basal - theta_high
    # + gain*|v_apical|) is ADDITIVE inside the sigmoid -- it has no
    # product term, so v_basal alone (with a lowered threshold) fires;
    # it cannot form the AND-only conjunction a bind needs (verified in
    # research/runners/_phaseB_dendritic_bind_derisk.py header, 2026-06-19).
    #
    # THE COINCIDENCE (Larkum 2013 BAC firing; Mel/Poirazi sigma-pi;
    # catalog G.02 active dendrites + J.08 NMDA coincidence): a somatic
    # burst requires a basal spike AND a coincident apical Ca2+ plateau.
    # Each compartment drive passes a NON-NEGATIVE saturating plateau
    # phi (Michaelis-Menten / finite NMDA-Ca conductance: 0 at rest,
    # ~linear for |z|<<z0, saturating for |z|>>z0). The soma output is
    # their PRODUCT phi(basal)*phi(apical) -> NON-ZERO ONLY when BOTH
    # compartments are engaged (a genuine coincidence AND; phi(0)=0).
    # The multiplication is the DENDRITIC UNIT's intrinsic operation
    # (a point neuron, summing, cannot form it), NOT a host product of
    # two precomputed answers.
    @staticmethod
    def dendritic_plateau(z, z0=1.0):
        """Non-negative saturating dendritic plateau (Michaelis-Menten /
        finite-conductance NMDA-Ca form). A plateau is a depolarization
        (>=0); signed inputs are carried by separate ON/OFF drive
        channels at the caller (biological push-pull), so z is expected
        >=0 and abs() is a safety rectifier only. phi(0)=0 (the AND
        anchor: no drive -> no plateau)."""
        a = np.abs(np.asarray(z, float))
        return a / (1.0 + a / max(float(z0), 1e-9))

    def apical_basal_coincidence(self, x_basal, x_apical,
                                 z0_basal=1.0, z0_apical=1.0):
        """MULTIPLICATIVE BAC coincidence (default-off). Basal drive =
        x_basal @ W_basal; apical drive = x_apical @ B_apical; each
        through the plateau; soma = phi(basal)*phi(apical). Returns the
        soma coincidence + the compartment traces (for anti-cheat
        witnesses). This is the sigma-pi conjunction a bind needs."""
        vb = np.asarray(x_basal, float) @ self.W_basal
        va = np.asarray(x_apical, float) @ self.B_apical
        gb = self.dendritic_plateau(vb, z0_basal)
        ga = self.dendritic_plateau(va, z0_apical)
        return {"soma": gb * ga, "g_basal": gb, "g_apical": ga,
                "v_basal": vb, "v_apical": va}

    # ---------------------------------------------------------------
    # ADDITIVE / DEFAULT-OFF extension (2026-08-09): the FAITHFUL
    # SPIKING BAC coincidence. apical_basal_coincidence above returns a
    # STATIC rate product phi(basal)*phi(apical) -- one host multiply,
    # no time, no spikes (the 2026-08-09 adversarial verify flagged that
    # as a rate sigma-pi, not a spiking-membrane coincidence). This
    # method runs a REAL TEMPORAL spiking process and returns per-channel
    # SPIKE COUNTS: the conjunction emerges from membrane dynamics over
    # time -- there is NO host product anywhere in it.
    #
    # THE MECHANISM (Larkum 2013 BAC firing; Larkum/Zhu/Sakmann 1999 Ca2+
    # plateau; catalog G.02 active dendrites + J.08 NMDA coincidence):
    #   * basal drive (a saturating dendritic plateau phi, so it is
    #     bounded) leaky-integrates the SOMA membrane v (tau_m);
    #   * a supra-threshold APICAL drive IGNITES a regenerative Ca2+
    #     plateau (graded, self-sustaining, decays with plateau_tau) that
    #     injects a SUSTAINED depolarizing current (apical_gain) into the
    #     soma across a temporal WINDOW;
    #   * a somatic SPIKE (HARD threshold theta + reset + refractory)
    #     fires ONLY when the basal depolarization coincides IN TIME with
    #     an active apical plateau.
    # The AND is the HARD SPIKE THRESHOLD acting on two individually
    # sub-threshold inputs -- the conjunction a SOFT (sigmoid) soma
    # cannot form (step() sums inside a sigmoid, so basal alone leaks
    # through; a hard threshold with both inputs sub-threshold does not).
    # theta is set homeostatically by the caller from taught-cell drive
    # statistics (between the single-input peak and the coincident sum).
    # Signed factors are carried by the caller as non-negative ON/OFF
    # channels (a spiking population rate is >=0). basal_onset DELAYS the
    # basal drive: the TEMPORAL-COINCIDENCE witness -- basal arriving
    # after the plateau has decayed collapses the output to the AND floor
    # (impossible for a static product, which has no time).
    def bac_spiking_coincidence(self, x_basal, x_apical, theta,
                                z0_basal=1.0, z0_apical=1.0, T=40,
                                tau_m=3.0, plateau_tau=18.0,
                                plateau_thresh=0.3, apical_gain=0.6,
                                basal_onset=0, plateau_onset=6,
                                refractory=2, v_reset=0.0,
                                return_traces=False):
        xb = self.dendritic_plateau(
            np.maximum(np.atleast_1d(np.asarray(x_basal, float)), 0.0),
            z0_basal)
        xa = (np.maximum(np.atleast_1d(np.asarray(x_apical, float)), 0.0)
              / max(float(z0_apical), 1e-9))
        n = int(xb.shape[-1])
        v = np.zeros(n)
        plateau = np.zeros(n)
        refr = np.zeros(n)
        counts = np.zeros(n)
        v_peak = np.zeros(n)
        plateau_peak = np.zeros(n)
        zero = np.zeros(n)
        m_leak = float(np.exp(-1.0 / max(float(tau_m), 1e-9)))
        a_leak = float(np.exp(-1.0 / max(float(plateau_tau), 1e-9)))
        for t in range(int(T)):
            if t < int(plateau_onset):
                ig = self.dendritic_plateau(
                    np.maximum(xa - float(plateau_thresh), 0.0), 1.0)
                plateau = np.maximum(plateau, ig)   # graded regen ignition
            plateau = plateau * a_leak              # Ca plateau decay
            plateau_peak = np.maximum(plateau_peak, plateau)
            basal = xb if t >= int(basal_onset) else zero
            v = m_leak * v + basal + float(apical_gain) * plateau
            v_peak = np.maximum(v_peak, v)
            spike = (v >= float(theta)) & (refr <= 0.0)
            counts = counts + spike.astype(float)
            v = np.where(spike, float(v_reset), v)  # somatic reset
            refr = np.where(spike, float(refractory),
                            np.maximum(refr - 1.0, 0.0))
        if return_traces:
            return {"counts": counts, "v_peak": v_peak,
                    "plateau_peak": plateau_peak}
        return counts
