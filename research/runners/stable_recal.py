"""SIM_STABLE_RECAL -- DEFAULT-OFF switch: the three GNW workspaces whose ignition was the old integrator's numerical
runaway (the ignition bus, the thought-swap workspace and the global-stop distributed-overwrite workspace),
re-calibrated by their OWN rules on the stable integrator (research/stable-recal, 2026-10-27).

Why. On the old integrator the GNW ignition-bus workspace, the thought-swap workspace and the global-stop workspace
ignite at the drive where their cells leave the valid membrane range, and the ignited rate 1/3 is that runaway
(findings 2026-10-26-stable-integration-dev7.md and 2026-10-27-stable-recal-dev78.md). Under SIM_STABLE_INTEGRATION=1
the membranes stay valid and the three workspaces stop igniting at their production constants. This switch applies the
re-calibrations measured on the isolated organs, chosen by each organ's documented rules and never from a reply.

Values.
  * Unset, "" or "0": nothing changes. Every accessor below returns the production value it was handed, and the
    production code paths are byte-identical (tests/test_stable_recal_switch.py).
  * "1": requires SIM_STABLE_INTEGRATION=1 (raises RuntimeError without it, because the re-calibration is for the
    stable membranes only). Any other value raises ValueError.

The accessors are read at call time, so the switch is a process-level environment setting like
SIM_STABLE_INTEGRATION. Organs not listed here (the ACC-BG stop trigger, ...) keep their production constants.

REMAPPING RULE, and its limit. The production organ builders take their constants as default arguments, so under the
switch a value-conditional accessor (`bus_attractor_weight`, `bus_d_sub`, `swap_value`) cannot tell a default from an
explicit argument: ANY value equal to the production value is remapped to the re-calibrated one, including an explicit
argument a caller passed on purpose (an explicit `weight=30.0` on the bus, `w_rec=34.0`, `establish_pa=8000.0` or
`thal_tonic_pa=700.0` on the swap organ, a bus drive of 1000.0 or 1400.0 pA, the global stop's `w_rec=34.0`,
`norm_ws_w=16.0`, `thal_tonic_pA=700.0`, `margin_ref=MARGIN_REF` or `boost_scale=BOOST_SCALE`, which
webapp/gnw_global_stop.py passes explicitly). Only an explicit value that DIFFERS from the production value is kept (a
lesion's weight 0, a research weight such as 12). A research script that wants the production number itself under
the switch has to pass a number that differs from it, or leave the switch off. The threshold accessors
(`bus_solo_plateau`, `bus_thr`, `swap_ignite_thresh`, `stop_ignite_thresh`) are called with a module constant, not a
caller argument, and return the re-calibrated threshold unconditionally while the switch is on.
tests/test_stable_recal_switch.py (test_explicit_production_value_is_remapped_under_the_switch) pins this.

Provenance of the swap values: research/findings/raw/_stable_recal/swap_calibration_record.json, written by
research/runners/stable_recal_calibration_record.py from the cited artifacts (it also checks that the record's
production and re-calibrated values equal the `SWAP` table below). `w_pat_pred` (32) is formula-only: no cited
artifact varies it. Provenance of the global-stop values:
research/findings/raw/_stable_recal/stop_calibration_record.json, written by
research/runners/stable_recal_stop_calibration_record.py (it checks the record against the `STOP` table).
The global-stop organ's own GO gate passes on 5 of 15 dev seeds at these values (the conflict-gated stop itself on 11
of 15; see the record), so this re-calibration is a measured operating point on valid membranes, not a GO.
"""
from __future__ import annotations

import os

STABLE_RECAL_ENV = "SIM_STABLE_RECAL"


def stable_recal_requested() -> bool:
    """True when SIM_STABLE_RECAL is "1" (and SIM_STABLE_INTEGRATION is "1"); False when unset, "" or "0"."""
    raw = os.environ.get(STABLE_RECAL_ENV)
    if raw is None:
        return False
    s = raw.strip()
    if s in ("", "0"):
        return False
    if s != "1":
        raise ValueError(f"{STABLE_RECAL_ENV} must be unset, '0' or '1', got {raw!r}")
    from sim.bridge import stable_integration_requested, STABLE_INTEGRATION_ENV
    if not stable_integration_requested():
        raise RuntimeError(f"{STABLE_RECAL_ENV}=1 re-calibrates the GNW workspaces for the stable integrator and is "
                           f"meaningless without {STABLE_INTEGRATION_ENV}=1; set both or neither")
    return True


# ── GNW ignition bus (research/runners/_p1_2_workspace_deliberation_loop_derisk.build_workspace_bridge and every hop
#    that reads it). Source: research/findings/raw/_stable_integration/bus_ignition_curve_s7.json and
#    bus_quorum_w5_s7.json (finding 2026-10-26-stable-integration-dev7.md, "A calibration both integrators agree on").
BUS_PRODUCTION_WEIGHT = 30.0   # DEFAULT_ATTRACTOR_WEIGHT (rung-1 self-attractor recipe)
BUS_WEIGHT = 5.0               # both integrators agree at 5 (no runaway on either): knee 1000-1500 pA
BUS_PLATEAU = 0.089            # the measured ignited plateau at weight 5 (membranes near -25 mV)
BUS_THR = 0.5 * BUS_PLATEAU    # IGNITE_FRAC x plateau, the bus's own ignition criterion
BUS_D_SUB = {3: 500.0,         # unanimity quorum rule 2*d_sub < knee <= 3*d_sub with the knee in (1000, 1500]
             2: 750.0}         # coincidence rule d_sub < knee <= 2*d_sub
BUS_PRODUCTION_D_SUB = {3: 1000.0, 2: 1400.0}


def bus_attractor_weight(weight: float) -> float:
    """The workspace assembly's recurrent weight: the production 30 becomes 5 under the switch (an explicit 30.0 is
    remapped too: see the module docstring); any other value (a lesion's 0, an explicit research weight) is kept."""
    w = float(weight)
    if w == BUS_PRODUCTION_WEIGHT and stable_recal_requested():
        return BUS_WEIGHT
    return w


def bus_solo_plateau(default: float) -> float:
    return BUS_PLATEAU if stable_recal_requested() else default


def bus_thr(default: float) -> float:
    return BUS_THR if stable_recal_requested() else default


def bus_d_sub(n_organs: int, default: float) -> float:
    """The per-organ subthreshold drive for an `n_organs` quorum (3 = unanimity bus, 2 = coincidence bus). An explicit
    drive equal to the production value (1000.0 for 3, 1400.0 for 2) is remapped like the default."""
    if stable_recal_requested() and float(default) == BUS_PRODUCTION_D_SUB.get(int(n_organs)):
        return BUS_D_SUB[int(n_organs)]
    return default


# ── thought-swap workspace (research/runners/_gnw_neural_swap_intention_derisk: `build` + `run_intention_swap`, used
#    by webapp/gnw_thought_swap.py, the ACC-BG stop trigger and the congruence reader). Source and the rule behind
#    each value: research/findings/raw/_stable_recal/swap_calibration_record.json.
SWAP = {
    "w_rec": (34.0, 6.5),            # lowest weight with a steady (non-oscillating) sustained up-state, plateau 0.084
    "norm_ws_w": (16.0, 4.0),        # largest normalization without the ~60-step relaxation oscillation
    "thal_tonic_pa": (700.0, 25.0),  # largest thalamic support with no spontaneous ignition in 1000 steps
    "solo_plateau": (1.0 / 3.0, 0.084),   # the measured ignited plateau; threshold IGNITE_FRAC x plateau = 0.042
    "establish_pa": (8000.0, 1280.0),     # production ratio establish/knee 8000/2500 = 3.2, x the stable knee 400
    "w_ws_occ": (8.0, 32.0),         # occupancy read input conserved: 8 x (1/3) / 0.084
    "w_pat_pred": (8.0, 32.0),       # FORMULA-ONLY (not measured): the same conservation as w_ws_occ, no run varies it
    "w_occ_gate": (20.0, 80.0),      # the vacancy veto knee (occupied gate 0.215 -> 0.044)
    "w_pred_mm": (20.0, 80.0),       # the organ's match bar: match mismatch rate < 0.02
    "w_gate_ws": (100.0, 4.0),       # middle of the band 2-8 where the gate primitive holds on both seeds
    "timing_window": (9, 36),        # the organ's 3-ignition-period rule: 3 spikes of an ignited neuron
}


def swap_value(name: str, value):
    """The swap constant `name`: the re-calibrated value under the switch when `value` equals the production value
    (a default and an explicit argument with the production value are indistinguishable, so both are remapped: see the
    module docstring); otherwise `value` unchanged (a lesion or an explicit research setting that differs from the
    production value is kept)."""
    prod, recal = SWAP[name]
    if stable_recal_requested() and float(value) == float(prod):
        return type(value)(recal) if isinstance(value, int) and not isinstance(value, bool) else float(recal)
    return value


def swap_ignite_thresh(ignite_frac: float, default: float) -> float:
    """The swap organ's ignition criterion, scoped to the swap organ (Rung-2's SOLO_PLATEAU, which the global-stop
    workspace shares, is not touched)."""
    if stable_recal_requested():
        return float(ignite_frac) * SWAP["solo_plateau"][1]
    return default


# ── global-stop distributed-overwrite workspace (research/runners/_gnw_distributed_overwrite_workspace_derisk:
#    `build_overwrite_bridge`, `run_confident`, `run_conflict_stop`, `evaluate_seed`, `_margin`; used by
#    webapp/gnw_global_stop.py). Source and the rule behind each value:
#    research/findings/raw/_stable_recal/stop_calibration_record.json.
STOP = {
    "w_rec": (34.0, 6.5),            # ~1.44x the sustain knee (4.0-4.5) at nws 4, below neighbour recruitment (10)
    "norm_ws_w": (16.0, 4.0),        # largest normalization whose sustained state reads reliably in the 15-step window
    "thal_tonic_pa": (700.0, 50.0),  # largest thalamic support with no spontaneous ignition in 1000 steps
    "solo_plateau": (1.0 / 3.0, 0.084),   # the measured ignited plateau; criterion IGNITE_FRAC x plateau = 0.042
    "strong_pa": (5000.0, 800.0),    # STRONG_PA = 2 x the knee; stable knee by the organ's 45-step read = 400 pA
    "margin_ref": ((1.0 / 3.0) * 0.5, 0.057),  # halfway between the measured confident and conflict margins
    "boost_scale": (0.85, 31.0),     # full-conflict boost 0.8, the middle of the robust clearing band 0.6-1.0
}


def stop_value(name: str, value):
    """The global-stop constant `name`: the re-calibrated value under the switch when `value` equals the production
    value (default and explicit production arguments are both remapped: see the module docstring); otherwise `value`
    unchanged (a lesion's 0 or an explicit research setting that differs from the production value is kept)."""
    prod, recal = STOP[name]
    if stable_recal_requested() and float(value) == float(prod):
        return float(recal)
    return value


def stop_ignite_thresh(ignite_frac: float, default: float) -> float:
    """The global-stop organ's ignition criterion, scoped to that organ (Rung-2's SOLO_PLATEAU, shared with other
    organs, is not touched)."""
    if stable_recal_requested():
        return float(ignite_frac) * STOP["solo_plateau"][1]
    return default
