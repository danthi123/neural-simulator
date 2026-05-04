"""Provenance for biological parameters — adapted from OpenWorm's c302.

Every tunable parameter in this codebase has provenance: where it came
from (literature citation or empirical tuning), how confident we are
in the value, and what units it's expressed in. This module formalizes
that.

Adapted from OpenWorm's c302 `bioparameters.py` — see
research/findings/2026-05-04-openworm-research.md for the analysis.

Use cases:
  - Audit: which parameters in our 375-entry feature catalog are
    BlindGuess vs literature-cited?
  - Sweep design: don't sweep BlindGuess parameters before the
    high-certainty ones (those are likely noise; tune the priors first).
  - Reproducibility: when reporting results, the parameter's source
    is part of the configuration record.

Example:
    >>> from sim.bioparameter import BioParameter, Certainty, Source
    >>> stdp_w_max = BioParameter(
    ...     name="stdp_w_max",
    ...     value=5.0,
    ...     unit="dimensionless",
    ...     certainty=Certainty.MEDIUM,
    ...     source="empirical (matches design weight 3.0 + headroom)",
    ... )
    >>> stdp_w_max.value
    5.0
    >>> stdp_w_max.certainty
    <Certainty.MEDIUM: 'medium'>

Currently this is a registry, not a runtime enforcement layer. See
PARAMETER_REGISTRY for currently-tracked params. Extend the registry
as biological parameters are tuned with deliberate provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Certainty(str, Enum):
    """Confidence in a parameter's value.

    HIGH: literature-cited, multiple papers agree, value matches biology.
    MEDIUM: literature-cited but may be context-dependent, OR empirically
        validated against multiple test conditions.
    LOW: literature-cited but limited evidence, OR empirically tuned for
        one specific use case.
    BLINDGUESS: no biological grounding; tuned to make the sim work.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BLINDGUESS = "blindguess"


class Source(str, Enum):
    """Common citation sources for parameters in this project."""
    KANDEL = "Kandel et al. Principles of Neural Science (6th ed)"
    BI_POO = "Bi & Poo 1998"
    SCHULTZ = "Schultz 1998"
    HEBB = "Hebb 1949"
    IZHIKEVICH = "Izhikevich 2007"
    BRETTE_GERSTNER = "Brette & Gerstner 2005"
    HODGKIN_HUXLEY = "Hodgkin & Huxley 1952"
    PULVERMULLER = "Pulvermuller 2001-2003"
    VOGELS = "Vogels et al 2011"
    HOFER = "Hofer et al 2011"
    WANG = "Wang 2002"
    HUBEL_WIESEL = "Hubel & Wiesel 1962"
    ALBUS = "Albus 1971"
    EMPIRICAL = "empirical (this project's measurements)"
    BLINDGUESS = "BlindGuess (no biology source)"


@dataclass(frozen=True)
class BioParameter:
    """A biological parameter with full provenance.

    Frozen so it can be used as a dict key or hashed for registry lookup.

    Attributes:
        name: Short identifier matching the CoreSimConfig field name where
            possible (e.g. "stdp_w_max", "ou_std_current_pA").
        value: The actual value used.
        unit: SI/biology unit string. "dimensionless" if unitless.
        certainty: Confidence level (Certainty enum).
        source: Citation. Use Source enum members or a free-form string.
        notes: Optional additional context.
    """
    name: str
    value: Any
    unit: str
    certainty: Certainty
    source: str
    notes: Optional[str] = None

    def __post_init__(self):
        # Allow Source enum or string
        if isinstance(self.source, Source):
            object.__setattr__(self, "source", self.source.value)


def _bp(name, value, unit, certainty, source, notes=None) -> BioParameter:
    """Helper for short registry entries."""
    return BioParameter(
        name=name, value=value, unit=unit,
        certainty=certainty, source=source, notes=notes,
    )


# Registry of currently-tracked parameters with provenance.
# This is a SEED — extend as parameters are tuned with deliberate
# attribution. To audit which CoreSimConfig fields are NOT in this
# registry, see audit_unregistered_params().
PARAMETER_REGISTRY: Dict[str, BioParameter] = {
    # Numerical integration
    "dt_ms": _bp("dt_ms", 0.5, "ms", Certainty.MEDIUM,
                 "Izhikevich 2007 + empirical (1.0 ms validated stable for this network)",
                 "0.05 ms required for HH; 1.0 ms safe for Izhikevich at this scale"),
    "fast_spike_reset": _bp("fast_spike_reset", False, "bool",
                             Certainty.HIGH, Source.EMPIRICAL,
                             "TDD numerical equivalence verified; opt-in for backward compat"),

    # STDP plasticity (Bi & Poo 1998)
    "stdp_a_plus": _bp("stdp_a_plus", 0.012, "dimensionless",
                       Certainty.MEDIUM, Source.BI_POO,
                       "LTP magnitude; tuned at single-process for 1800-step navigation"),
    "stdp_a_minus": _bp("stdp_a_minus", 0.012, "dimensionless",
                        Certainty.MEDIUM, Source.BI_POO,
                        "LTD magnitude; symmetric with stdp_a_plus by default"),
    "stdp_tau_pre_ms": _bp("stdp_tau_pre_ms", 20.0, "ms",
                            Certainty.HIGH, Source.BI_POO,
                            "Bi-Poo Fig 6: ~20 ms decay constant"),
    "stdp_tau_post_ms": _bp("stdp_tau_post_ms", 20.0, "ms",
                             Certainty.HIGH, Source.BI_POO),
    "stdp_w_max": _bp("stdp_w_max", 2.0, "dimensionless",
                      Certainty.LOW, Source.EMPIRICAL,
                      "soft-bound saturation; default 2.0 is too low for some "
                      "configs (text I/O needs 5.0). Override per runner."),

    # Reward modulation (Schultz 1998)
    "reward_learning_rate": _bp("reward_learning_rate", 0.01, "dimensionless",
                                 Certainty.MEDIUM, Source.SCHULTZ,
                                 "DA gain on eligibility; tuned empirically"),
    "eligibility_decay_tau_ms": _bp("eligibility_decay_tau_ms", 1000.0, "ms",
                                     Certainty.MEDIUM, Source.SCHULTZ),

    # OU noise (background drive)
    "ou_std_current_pA": _bp("ou_std_current_pA", 80.0, "pA",
                              Certainty.LOW, Source.EMPIRICAL,
                              "noise floor; high enough for spontaneous spiking, "
                              "low enough that signal-to-noise is reasonable"),
    "ou_tau_ms": _bp("ou_tau_ms", 5.0, "ms",
                      Certainty.LOW, Source.EMPIRICAL),

    # Inhibitory reversal (corrected 2026-04-25)
    "E_inh_mV": _bp("E_inh_mV", -75.0, "mV",
                     Certainty.HIGH, Source.KANDEL,
                     "GABAergic reversal potential; -75 mV per Cl- equilibrium"),

    # Hodgkin-Huxley per-gate Q10 (corrected 2026-04-25)
    "hh_q10_m": _bp("hh_q10_m", 3.0, "dimensionless",
                     Certainty.HIGH, Source.HODGKIN_HUXLEY,
                     "Sodium activation Q10; experimentally measured"),
    "hh_q10_h": _bp("hh_q10_h", 1.5, "dimensionless",
                     Certainty.HIGH, Source.HODGKIN_HUXLEY,
                     "Sodium inactivation Q10"),
    "hh_q10_n": _bp("hh_q10_n", 1.5, "dimensionless",
                     Certainty.HIGH, Source.HODGKIN_HUXLEY,
                     "Potassium activation Q10"),

    # Biology-grounded fixes (2026-05-03 sweep)
    "topographic_bias_factor": _bp("topographic_bias_factor", 1.5,
                                     "dimensionless",
                                     Certainty.MEDIUM, Source.PULVERMULLER,
                                     "Wernicke->motor topographic prior. "
                                     "1.5/0.7 ratio matches Pulvermuller 2001-2003 "
                                     "reported 2-3x somatotopic asymmetry."),
    "off_target_bias_factor": _bp("off_target_bias_factor", 0.7,
                                    "dimensionless",
                                    Certainty.MEDIUM, Source.PULVERMULLER),
    "n_motor_fs_per_action": _bp("n_motor_fs_per_action", 3, "count",
                                   Certainty.MEDIUM, Source.VOGELS,
                                   "PV-FSI count ~12% of motor pool of 25; "
                                   "Vogels 2011 / Hofer 2011 cortical PV-FSI 10-15%."),
}


def audit_unregistered_params() -> List[str]:
    """List CoreSimConfig fields that are NOT in PARAMETER_REGISTRY.

    These are unaudited parameters — either trivial (booleans for
    enabling features), or parameters we've tuned without recording
    provenance. The latter should be added to the registry over time.

    Returns a list of field names not yet registered.
    """
    try:
        from dataclasses import fields
        from sim.config import CoreSimConfig
        all_fields = {f.name for f in fields(CoreSimConfig)}
    except Exception:
        return []
    return sorted(all_fields - set(PARAMETER_REGISTRY.keys()))


def registry_summary() -> Dict[str, int]:
    """Counts by certainty level. Useful for "how many BlindGuess
    parameters do we still have?" reports."""
    counts: Dict[str, int] = {c.value: 0 for c in Certainty}
    for bp in PARAMETER_REGISTRY.values():
        counts[bp.certainty.value] += 1
    return counts


def get(name: str) -> Optional[BioParameter]:
    """Look up a registered parameter by name. None if unregistered."""
    return PARAMETER_REGISTRY.get(name)


__all__ = [
    "BioParameter", "Certainty", "Source",
    "PARAMETER_REGISTRY",
    "audit_unregistered_params",
    "registry_summary",
    "get",
]
