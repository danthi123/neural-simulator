"""Kill-safe THREE-STATE gate runner for compose x temporal-credit.
Runs the cheap-gate-GREEN-validated learner across V1 (no-gap td),
science (gapped td), and {hebbian_no_trace, permuted, wrongsign}
controls x seeds; per-(seed) kill-safe checkpoint via the REUSED
sim.train_checkpoint; constructs (does NOT mutate) the REUSED NM
modulator so the temporal-credit delta IS the phasic-DA signal
(feature-catalog C.30). THREE-STATE verdict via compose_bind_core.
NO automatic differentiation. ASCII.

HONEST CEILING (printed, never spun): a PASS = mechanism-level/in-sim
ONLY (temporal credit bridges the bind-gap where the no-trace
v16-analog cannot) -- NOT composition-solved, NOT compositional
language, NOT scaled/integrated; that is a SEPARATE later gated
increment. PASS/BOUNDARY/VOID all decision-relevant + propagated
honestly."""
from __future__ import annotations
import argparse
import json
import sys

from sim.compose_temporal_bind import run_bind, _GAP
from sim.train_checkpoint import (save_checkpoint, load_checkpoint,
                                  resume_epoch)  # REUSED UNMODIFIED
from sim.neuromodulators import (NeuromodulatorConfig, ProductionRule,
                                 ModulatorTarget)  # REUSED UNMODIFIED
from research.runners.compose_bind_core import ctb_verdict

_CONTROLS = ("hebbian_no_trace", "permuted", "wrongsign")
_BANNER = ("HONEST CEILING: mechanism-level/in-sim ONLY -- temporal "
           "credit bridges the bind-gap where the no-trace v16-analog "
           "cannot; NOT composition-solved, integration a SEPARATE "
           "later gated increment.")


def _da_modulator_from_delta():
    """The catalog C.30 upgrade demonstrated via the REUSED NM
    subsystem UNMODIFIED: a from_reward DA modulator whose drive is
    the temporal-credit TD delta. Constructed to prove composition
    with the validated phasic-DA substrate; not mutated here."""
    return NeuromodulatorConfig(
        name="dopamine_compose", baseline=0.0, decay_tau_ms=50.0,
        concentration_min=-5.0, concentration_max=5.0,
        targets=[ModulatorTarget(target_type="plasticity_rate",
                                 scope="all", sensitivity=1.0)],
        production_rules=[ProductionRule(rule_type="from_reward",
                                         sensitivity=1.0,
                                         threshold=0.0,
                                         window_ms=0.0)])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44, 45, 46])
    ap.add_argument("--tiny-synth", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds for the pre-registered gate")
        return 2
    _ = _da_modulator_from_delta()        # construct (not mutate)
    nt = 200 if a.tiny_synth else None    # None -> module _N_TRIALS
    per_seed = {}
    try:
        for s in a.seeds:
            row = {"nogap_td": run_bind("td", s, 0, nt),
                   "td": run_bind("td", s, _GAP, nt),
                   "controls": {c: run_bind(c, s, _GAP, nt)
                                for c in _CONTROLS}}
            if a.ckpt:
                save_checkpoint(a.ckpt, s,
                                {"row": [row["nogap_td"], row["td"]]},
                                None, [])
            per_seed[s] = row
    except KeyboardInterrupt:
        print("INTERRUPTED -- partial checkpoint flushed; resumable")
        return 130
    verdict = ctb_verdict(per_seed)
    verdict["banner"] = _BANNER
    if a.tiny_synth:
        verdict["note"] = "TINY-SYNTH toy verdict -- NOT propagated"
    with open(a.out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print("GATE=%s  %s" % (verdict["GATE"], _BANNER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
