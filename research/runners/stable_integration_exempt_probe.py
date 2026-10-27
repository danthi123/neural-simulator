"""PER-ORGAN ATTRIBUTION of the stable-integration reply changes, by intervention (research/stable-integration).

The whole-brain switch SIM_STABLE_INTEGRATION=1 changes replies (research/findings/2026-10-26-stable-integration-dev7.md);
the response read-outs point at the GNW thought-swap workspace (webapp/gnw_thought_swap.py), whose period-3 ignition
stops under ON. This probe tests that by flipping the integrator for that ONE organ's bridge against the rest:

  --mode on_except_swap   SIM_STABLE_INTEGRATION=1 everywhere, but the thought-swap workspace bridge is constructed
                          with the switch unset, so it keeps its organ's own integrator (bounded False, 1 sub-step).
  --mode off_except_swap  the switch unset everywhere, but the thought-swap workspace bridge is constructed with it
                          set, so it alone runs the stable set (bounded True, 4 sub-steps, model_reset), re-asserted
                          every step like the switch does.

Mechanics: the swap workspace builds its substrate through `webapp.gnw_thought_swap._build_swap_substrate` (lazily, on
the first topic turn); the probe wraps that name so the returned bridge gets the override, then runs the membrane census
workload (`research.runners.membrane_census.run_worker`, the live brain_chat handler under the stage-2 gate's frozen
env) in a fresh subprocess, exactly as the census shard does. No sim/ or webapp/ edit; the wrapper is installed only in
the worker process. The census JSON then carries, per bridge, the read-back that proves which bridges ran which
integrator, plus the per-turn replies; the record also carries `exempt_probe` (mode, how many swap builds were
patched and each patched bridge's effective values).

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

MODES = ("on_except_swap", "off_except_swap")
ENV = "SIM_STABLE_INTEGRATION"
_PATCHED = []


def _effective(cfg):
    return {"enable_bounded_conductance_integration": bool(cfg.enable_bounded_conductance_integration),
            "izh_membrane_substeps": cfg.izh_membrane_substeps,
            "izh_substep_refractory_rule": cfg.izh_substep_refractory_rule}


def install(mode):
    """Wrap webapp.gnw_thought_swap._build_swap_substrate so its bridge runs the integrator opposite to the rest."""
    import webapp.gnw_thought_swap as GTS
    orig = GTS._build_swap_substrate

    def _wrapped(*a, **k):
        # The substrate build constructs AND warms up its bridge (it steps it), so the override must hold from the
        # bridge's construction: the switch is read in SimulationBridge.__init__, so the env is set the opposite way
        # for the duration of this build only (the build is synchronous in the handler thread), then restored.
        saved = os.environ.get(ENV)
        if mode == "on_except_swap":
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = "1"
        try:
            S = orig(*a, **k)
        finally:
            if saved is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = saved
        b = S["bridge"]
        cfg = b.core_config
        want_requested = (mode == "off_except_swap")
        if bool(b.stable_integration_requested) != want_requested:
            raise RuntimeError("swap bridge read the wrong switch value at construction")
        _PATCHED.append({"mode": mode, "effective": _effective(cfg), "requested": bool(b.stable_integration_requested),
                         "applied": bool(b.stable_integration_applied), "substep_calls_at_build": int(b.substep_calls),
                         "n_neurons": int(cfg.num_neurons)})
        return S

    GTS._build_swap_substrate = _wrapped
    return orig


def run_worker(seed, group, mode, out_path):
    from research.runners import membrane_census as MC
    install(mode)
    rc = MC.run_worker(seed, group, "intact_a", out_path)
    with open(out_path) as f:
        d = json.load(f)
    d["exempt_probe"] = {"mode": mode, "n_swap_builds_patched": len(_PATCHED), "patched": _PATCHED,
                         "runner": "research/runners/stable_integration_exempt_probe.py"}
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    print("[exempt probe] %s: %d swap build(s) patched %s" % (mode, len(_PATCHED), _PATCHED), flush=True)
    return rc


def run_shard(seed, group, mode, out_path):
    from research.runners import stage2_exit_gate as SG
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    env["SIM_BACKEND"] = "numpy"
    if mode == "on_except_swap":
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
