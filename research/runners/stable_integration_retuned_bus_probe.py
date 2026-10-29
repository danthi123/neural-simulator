"""RE-CALIBRATED BUS ON THE STABLE INTEGRATOR: does it restore recall verification? (research/stable-integration)

The attribution probe shows the GNW ignition-bus workspace decides recall verification and the inner C5 result, and
the isolated-organ probe (`stable_integration_bus_ignition_probe.py`, dev seed 7) shows why: on the old integrator
the bus's ignition knee (between 2000 and 2500 pA at the production recurrent weight 30) is the drive at which all 80
cells of the slot leave the valid membrane range, and its 1/3 ignited rate is that runaway. On the stable integrator
weight 30 puts the assembly in depolarization block. At weight 5 both integrators agree (no runaway): solo knee
between 1000 and 1500 pA, plateau about 0.089.

This arm runs the whole brain on the stable set (SIM_STABLE_INTEGRATION=1) with the bus workspace re-calibrated by the
bus's OWN documented rules applied to that measured curve, chosen from the isolated organ and never from replies:
  * recurrent weight 5 for every workspace built by `build_workspace_bridge` (production 30; a lesioned build keeps 0);
  * ignition threshold = IGNITE_FRAC (0.5) x the measured plateau (0.089) = 0.0445 (production 0.5 x 1/3), set on
    every module that holds its own copy (`THR` in the N-organ, coincidence and reentrant modules; `SOLO_PLATEAU` in
    the P1.2 module, whose `_deliberate_hop` computes it at call time);
  * per-organ subthreshold drive by the quorum rule (N-1)*d_sub < knee <= N*d_sub with the knee in (1000, 1500]:
    three-organ unanimity d_sub = 500 (production 1000), two-organ coincidence d_sub = 750 (production 1400).
`calibrate_theta` (the reentrant organ's conflict threshold) re-calibrates itself on the built organ, unchanged.
No sim/ or webapp/ edit: module attributes are set in the worker before the brain is built. The census JSON carries
the per-bridge read-back plus `retuned_bus` (every constant set, before -> after, and how many workspace builds took
the new weight).

  .venv/bin/python -m research.runners.stable_integration_retuned_bus_probe --shard --seed 7 --group inner \
      --out research/findings/raw/_stable_integration/census/on_retuned_bus/s7/inner__intact_a.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

ENV = "SIM_STABLE_INTEGRATION"
PRODUCTION_WEIGHT = 30.0
RETUNED_WEIGHT = 5.0
STABLE_PLATEAU = 0.089            # measured: bus_ignition_curve_s7.json, arm on, weight 5, drives 1500-4000 pA
RETUNED_THR = 0.5 * STABLE_PLATEAU
RETUNED_D_SUB_3 = 500.0
RETUNED_D_SUB_2 = 750.0
_LOG = {"weight_substituted": 0, "weight_kept": [], "set": {}}


def _set(mod, name, value):
    before = getattr(mod, name)
    setattr(mod, name, value)
    _LOG["set"]["%s.%s" % (mod.__name__, name)] = {"before": before, "after": value}


def install():
    import importlib
    # import every holder FIRST: the threshold copies are computed from P.SOLO_PLATEAU at import, so an import after
    # the patch would already carry the new value and the before -> after record would hide the change
    names = ("research.runners._gnw_norgan_bus_derisk", "research.runners._gnw_coincidence_integrator_derisk",
             "research.runners._gnw_reentrant_metacog_gated_deliberation_derisk")
    mods = [importlib.import_module(nm) for nm in names]
    tob = importlib.import_module("webapp.gnw_three_organ_bus")
    two = importlib.import_module("webapp.gnw_two_organ_bus")
    P = importlib.import_module("research.runners._p1_2_workspace_deliberation_loop_derisk")
    orig_loop = P._build_assembly_loop_population

    def _loop(assembly_idx, attractor_weight):
        if float(attractor_weight) == PRODUCTION_WEIGHT:
            _LOG["weight_substituted"] += 1
            return orig_loop(assembly_idx, RETUNED_WEIGHT)
        _LOG["weight_kept"].append(float(attractor_weight))
        return orig_loop(assembly_idx, attractor_weight)

    P._build_assembly_loop_population = _loop
    _LOG["set"]["%s._build_assembly_loop_population" % P.__name__] = {
        "before": "attractor_weight", "after": "weight %g -> %g" % (PRODUCTION_WEIGHT, RETUNED_WEIGHT)}
    _set(P, "SOLO_PLATEAU", STABLE_PLATEAU)
    for m in mods:
        _set(m, "THR", RETUNED_THR)
        if hasattr(m, "SOLO_PLATEAU"):
            _set(m, "SOLO_PLATEAU", STABLE_PLATEAU)
    _set(tob, "_D_SUB_3", RETUNED_D_SUB_3)
    _set(two, "D_SUB_DEFAULT", RETUNED_D_SUB_2)
    nb = mods[0]
    before = dict(nb.D_SUB_UNANIMITY)
    nb.D_SUB_UNANIMITY[3] = RETUNED_D_SUB_3          # the shared dict object (gnw_bus_shadow imported it by name)
    _LOG["set"]["%s.D_SUB_UNANIMITY[3]" % nb.__name__] = {"before": before[3], "after": RETUNED_D_SUB_3}


def run_worker(seed, group, out_path):
    from research.runners import membrane_census as MC
    if os.environ.get(ENV) != "1":
        raise RuntimeError("the re-calibrated bus arm runs on the stable integrator: set %s=1" % ENV)
    install()
    rc = MC.run_worker(seed, group, "intact_a", out_path)
    with open(out_path) as f:
        d = json.load(f)
    d["retuned_bus"] = {"constants": _LOG["set"], "workspace_loop_builds_retuned": _LOG["weight_substituted"],
                        "workspace_loop_builds_kept": _LOG["weight_kept"],
                        "source_curve": "research/findings/raw/_stable_integration/bus_ignition_curve_s7.json",
                        "runner": "research/runners/stable_integration_retuned_bus_probe.py"}
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    print("[retuned bus] %d workspace loop builds retuned; kept %s" % (_LOG["weight_substituted"],
                                                                     _LOG["weight_kept"]), flush=True)
    return rc


def run_shard(seed, group, out_path):
    from research.runners import stage2_exit_gate as SG
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    env["SIM_BACKEND"] = "numpy"
    env[ENV] = "1"
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.stable_integration_retuned_bus_probe", "--worker",
           "--seed", str(int(seed)), "--group", group, "--out", part]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[retuned bus] worker FAILED rc=%s" % p.returncode, flush=True)
        return 1
    with open(part) as f:
        d = json.load(f)
    d["backend"] = "numpy"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    for suffix in ("", ".prov.json"):
        try:
            os.remove(part + suffix)
        except OSError:
            pass
    return 0


def main():
    from research.runners import stage2_exit_gate as SG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--group", required=True, choices=sorted(SG.GROUPS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed in SG.GATE_SEEDS:
        sys.exit("GATE seed %d refused: dev-seed diagnostic only" % a.seed)
    if a.worker:
        sys.exit(run_worker(a.seed, a.group, a.out))
    if a.shard:
        sys.exit(run_shard(a.seed, a.group, a.out))
    ap.error("choose --worker or --shard")


if __name__ == "__main__":
    main()
