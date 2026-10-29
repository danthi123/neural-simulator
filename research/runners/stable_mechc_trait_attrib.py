"""Which trait-typed organ changes the replies? One organ at a time (research/stable-mechc2; dev seeds only, numpy CPU).

With SIM_TRAIT_CENTRED_HET=1 the five Stage-2 groups change the cells of every trait-typed bridge in the brain (the census
read-back lists them: LearnedAssocGraph, the novelty-habituation bank, the Broca frame-slot bridge, ...). This runs ONE
group under the same membrane census as research.runners.membrane_census, with the switch ON for the bridges whose
construction call chain contains `--only` (a substring of a module or function name, e.g. `build_slot_bridge`) and OFF for
every other bridge. Nothing in sim/ changes: the restriction patches `sim.bridge.trait_centred_het_requested` in this
process, so the bridge's own read-back (requested / applied / readback) shows exactly which bridges were centred.

  SIM_STABLE_INTEGRATION=1 SIM_TRAIT_CENTRED_HET=1 .venv/bin/python -m research.runners.stable_mechc_trait_attrib \\
      --shard --only build_slot_bridge --seed 7 --group inner --arm intact_a --out <dir>/inner__intact_a.json
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


def chain_names(depth_skip=2, limit=40):
    """'module.function' of every frame above the caller (innermost first), for matching `--only`."""
    out = []
    try:
        f = sys._getframe(depth_skip)
    except ValueError:
        return out
    while f is not None and len(out) < limit:
        out.append("%s.%s" % (f.f_globals.get("__name__", "?"), f.f_code.co_name))
        f = f.f_back
    return out


def restrict(only):
    """Patch sim.bridge.trait_centred_het_requested so it is True only while a frame matching `only` is on the stack
    (and the real switch is on). Returns the original for restoring."""
    import sim.bridge as SB
    real = SB.trait_centred_het_requested

    def restricted():
        if not real():
            return False
        return any(only in n for n in chain_names(depth_skip=2))

    SB.trait_centred_het_requested = restricted
    return real


def run_shard(only, seed, group, arm, out_path):
    """A fresh subprocess with the Stage-2 gate's frozen env, as membrane_census.run_shard does for the plain census."""
    from research.runners import stage2_exit_gate as SG
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.stable_mechc_trait_attrib", "--worker", "--only", only,
           "--seed", str(int(seed)), "--group", group, "--arm", arm, "--out", part]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[trait attrib] worker FAILED rc=%s" % p.returncode, flush=True)
        return 1
    with open(part) as f:
        d = json.load(f)
    d["backend"] = "numpy"
    d["trait_attribution_only"] = only
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
    from research.runners import membrane_census as MC
    from research.runners import stage2_exit_gate as SG
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", action="store_true")
    ap.add_argument("--only", required=True, help="substring of a module.function name in the bridge's construction chain")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--group", required=True, choices=sorted(SG.GROUPS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.seed in SG.GATE_SEEDS:
        sys.stderr.write("GATE seed %d refused: dev seeds only.\n" % a.seed)
        sys.exit(2)
    if a.worker:
        restrict(a.only)
        sys.exit(MC.run_worker(a.seed, a.group, a.arm, a.out))
    if a.shard:
        sys.exit(run_shard(a.only, a.seed, a.group, a.arm, a.out))
    ap.error("choose --worker or --shard")


if __name__ == "__main__":
    main()
