"""Pool battery lines for the stable-recal dev battery (research/stable-recal): the membrane census (replies + per-organ
membrane validity) on dev seeds 1-12 x the five Stage-2 groups x three arms, written NOT queued to
research/coordination/stable-recal-pool-lines.txt in the format of membrane-census-pool-lines.txt (command TAB reason).

Arms (environment of the census subprocess; everything else is the Stage-2 gate's frozen env):
  off           -- SIM_STABLE_INTEGRATION and SIM_STABLE_RECAL empty (= off: production);
  stable        -- SIM_STABLE_INTEGRATION=1 only (valid membranes, production workspace constants);
  stable_recal  -- SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL=1 (valid membranes, bus + thought-swap re-calibrated).
Same runner, memory and caps as the census lines (membrane_census.POOL_MEM_GB / POOL_MEMCAP_GB / POOL_MAX_AGE_H).
Dev seeds only; the gate seeds are refused by the census itself.

  .venv/bin/python -m research.runners.stable_recal_pool_lines --write research/coordination/stable-recal-pool-lines.txt
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SEEDS = tuple(range(1, 13))
ARMS = {"off": "SIM_STABLE_INTEGRATION= SIM_STABLE_RECAL= ", "stable": "SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL= ",
        "stable_recal": "SIM_STABLE_INTEGRATION=1 SIM_STABLE_RECAL=1 "}   # each arm pinned explicitly ("" = off)
OUT_DIR = "research/findings/raw/_stable_recal/dev"
FINDING = "research/findings/2026-10-27-stable-recal-dev78.md"


def pool_job_lines(sha="<sha>", seeds=SEEDS, out_dir=OUT_DIR):
    from research.runners import membrane_census as MC
    from research.runners import stage2_exit_gate as SG
    out = []
    for seed in seeds:
        if int(seed) in SG.GATE_SEEDS:
            raise ValueError("dev seeds only; %d is a gate seed" % seed)
        for group in SG.PRIMARY_GROUPS:
            for arm, env in ARMS.items():
                d = "%s/%s/s%d" % (out_dir, arm, seed)
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && bash tools/memcap.sh "
                       "%d -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                       "NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= %s.venv/bin/python -u -m "
                       "research.runners.membrane_census --shard --seed %d --group %s --arm intact_a "
                       "--out %s/%s__intact_a.json" % (sha, MC.POOL_MEM_GB, d, MC.POOL_MEMCAP_GB, env, seed, group, d,
                                                       group))
                why = ("stable-recal dev battery, dev seed %d %s arm %s (intact_a); record: %s holds dev seeds 7 and 8 "
                       "only, and no other seed has the off / stable / stable+recal replies and census; "
                       "mem_gb=%d max_age_h=%d" % (seed, group, arm, FINDING, MC.POOL_MEM_GB, MC.POOL_MAX_AGE_H))
                out.append((cmd, why))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", required=True, help="output path (command<TAB>reason per line; not queued)")
    ap.add_argument("--sha", default="<sha>")
    a = ap.parse_args()
    lines = pool_job_lines(a.sha)
    with open(a.write, "w") as f:
        for cmd, why in lines:
            f.write("%s\t%s\n" % (cmd, why))
    print("[stable-recal pool lines] %d lines -> %s (NOT queued)" % (len(lines), a.write))
    return 0


if __name__ == "__main__":
    sys.exit(main())
