"""PER-ORGAN ATTRIBUTION of the stable-integration reply changes, by intervention (research/stable-integration).

The whole-brain switch SIM_STABLE_INTEGRATION=1 changes replies (research/findings/2026-10-26-stable-integration-dev7.md);
the response read-outs point at the GNW thought-swap workspace (webapp/gnw_thought_swap.py), whose period-3 ignition
stops under ON. This probe tests that by flipping the integrator for that ONE organ's bridge against the rest:

  --mode on_except_<organ>   SIM_STABLE_INTEGRATION=1 everywhere, but the organ's bridges are constructed with the
                             switch unset, so they keep their organ's own integrator (bounded False, 1 sub-step).
  --mode off_except_<organ>  the switch unset everywhere, but the organ's bridges are constructed with it set, so they
                             alone run the stable set (bounded True, 4 sub-steps, model_reset), re-asserted per step.

Mechanics: the worker wraps `SimulationBridge.__init__`; a construction whose call stack holds the organ's builder
frame (ORGANS: swap = `_gnw_neural_swap_intention_derisk.build`, the substrate of webapp/gnw_thought_swap.py; bus =
`_p1_2_workspace_deliberation_loop_derisk.build_workspace_bridge`, the workspace of the GNW ignition buses that wrap
the recall gate) reads SIM_STABLE_INTEGRATION the opposite way (env toggled for that construction only), so the
override holds from the bridge's first step (the builders warm their bridges up). The membrane census workload
(`research.runners.membrane_census.run_worker`, the live brain_chat handler under the stage-2 gate's frozen env) then
runs in a fresh subprocess, exactly as the census shard does. No sim/ or webapp/ edit. The census JSON carries, per
bridge, the read-back that proves which bridges ran which integrator, plus the per-turn replies; the record also
carries `exempt_probe` (mode, organ, every patched construction with its read-back).

  .venv/bin/python -m research.runners.stable_integration_exempt_probe --shard --mode on_except_swap --seed 7 \
      --group inner --out research/findings/raw/_stable_integration/census/on_except_swap/s7/inner__intact_a.json
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

MODES = ("on_except_swap", "off_except_swap", "on_except_bus", "off_except_bus")
ENV = "SIM_STABLE_INTEGRATION"
_PATCHED = []


def _effective(cfg):
    return {"enable_bounded_conductance_integration": bool(cfg.enable_bounded_conductance_integration),
            "izh_membrane_substeps": cfg.izh_membrane_substeps,
            "izh_substep_refractory_rule": cfg.izh_substep_refractory_rule}


# organ -> the (module suffix, function) whose frame on the call stack marks a bridge construction as that organ's
ORGANS = {
    "swap": ("_gnw_neural_swap_intention_derisk", "build"),                  # webapp/gnw_thought_swap.py substrate
    "bus": ("_p1_2_workspace_deliberation_loop_derisk", "build_workspace_bridge"),  # the GNW ignition-bus workspace
}


def _organ_of_mode(mode):
    return mode.split("_except_", 1)[1]


def install(mode):
    """Wrap SimulationBridge.__init__ so a bridge constructed under the organ's builder frame reads the switch the
    opposite way to the rest of the brain (the switch is read in __init__; the env is toggled for that construction
    only and restored). A builder that warms its bridge up steps it after construction, so the override holds from
    the first step. Every patched construction is recorded with its read-back."""
    import sim.bridge as SBM
    organ = _organ_of_mode(mode)
    mod_suffix, fn_name = ORGANS[organ]
    on_except = mode.startswith("on_except_")
    SB = SBM.SimulationBridge
    orig = SB.__dict__["__init__"]

    def _init(self, *a, **k):
        f = sys._getframe(1)
        hit = None
        while f is not None:
            if f.f_code.co_name == fn_name and str(f.f_globals.get("__name__", "")).endswith(mod_suffix):
                hit = "%s:%s" % (f.f_globals.get("__name__"), fn_name)
                break
            f = f.f_back
        if hit is None:
            return orig(self, *a, **k)
        saved = os.environ.get(ENV)
        if on_except:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = "1"
        try:
            orig(self, *a, **k)
        finally:
            if saved is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = saved
        if bool(self.stable_integration_requested) == on_except:
            raise RuntimeError("patched bridge read the wrong switch value at construction")
        _PATCHED.append({"mode": mode, "builder": hit, "requested": bool(self.stable_integration_requested),
                         "effective_at_construction": _effective(self.core_config)})

    SB.__init__ = _init
    return orig


def run_worker(seed, group, mode, out_path):
    from research.runners import membrane_census as MC
    install(mode)
    rc = MC.run_worker(seed, group, "intact_a", out_path)
    with open(out_path) as f:
        d = json.load(f)
    d["exempt_probe"] = {"mode": mode, "organ": _organ_of_mode(mode), "builder": ORGANS[_organ_of_mode(mode)],
                         "n_bridges_patched": len(_PATCHED), "patched": _PATCHED,
                         "runner": "research/runners/stable_integration_exempt_probe.py"}
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    print("[exempt probe] %s: %d bridge(s) patched %s" % (mode, len(_PATCHED), _PATCHED[:4]), flush=True)
    return rc


def run_shard(seed, group, mode, out_path):
    from research.runners import stage2_exit_gate as SG
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    env["SIM_BACKEND"] = "numpy"
    if mode.startswith("on_except_"):
        env[ENV] = "1"
    else:
        env.pop(ENV, None)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.stable_integration_exempt_probe", "--worker", "--mode", mode,
           "--seed", str(int(seed)), "--group", group, "--out", part]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[exempt probe] worker FAILED rc=%s" % p.returncode, flush=True)
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
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--group", required=True, choices=sorted(SG.GROUPS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed in SG.GATE_SEEDS:
        sys.exit("GATE seed %d refused: dev-seed diagnostic only" % a.seed)
    if a.worker:
        sys.exit(run_worker(a.seed, a.group, a.mode, a.out))
    if a.shard:
        sys.exit(run_shard(a.seed, a.group, a.mode, a.out))
    ap.error("choose --worker or --shard")


if __name__ == "__main__":
    main()
