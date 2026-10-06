"""Review evidence for docs/drafts/seam-map/CLAUDE-REVIEW-2026-09-29.md section 3 (the owner's question: does the draft
probe's DEGRADED panel confound the host-share reading?).

Re-runs the local model's degraded panel EXACTLY as written on branch `research/draft-seam-map`
(`research/runners/_draft_host_share_probe.py`: same composer build, same `default_rng(seed)` noise sequence, same
restore-after-each-level), loaded read-only from that branch with `git show`, and records two things the draft does
not: (1) whether each "answered" row returned the TAUGHT patient, (2) whether the store's cached CSR was rebuilt during
a noise level (i.e. whether the noise reaches the answer path or only the confidence trace).

DEV seeds only (1-7 by default). Not a verdict on any capability.

  bash tools/memcap.sh 6 -- .venv/bin/python -m research.runners._host_share_draft_confound_check --seeds 1,2,3,4,5,6,7 \
      --out research/findings/raw/_host_share/draft_confound_dev_s1-7.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# pinned to the draft-branch head that was reviewed (2026-09-29), not the moving branch name
DRAFT_REF = "34017865436defdac42ff2e348b8baf6dded434a:research/runners/_draft_host_share_probe.py"


def load_draft_probe(ref=DRAFT_REF):
    src = subprocess.run(["git", "-C", _REPO, "show", ref], capture_output=True, text=True, check=True).stdout
    fd, path = tempfile.mkstemp(suffix="_draft_host_share_probe.py")
    with os.fdopen(fd, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("_draft_host_share_probe_readonly", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.remove(path)
    return mod, subprocess.run(["git", "-C", _REPO, "rev-parse", ref.split(":")[0]], capture_output=True,
                               text=True).stdout.strip()


def run_seed(dp, seed):
    import numpy as np
    from research.runners.metacog_production_organ import MetacogProductionOrgan
    comp = dp.build_composer(seed)
    organ = MetacogProductionOrgan(seed=seed)
    organ.ensure_built()
    base = list(comp.store_conns)
    rng = np.random.default_rng(seed)
    per = []
    for sigma in dp.SIGMA_GRID:
        comp.store_conns = dp._noise(base, sigma, rng)
        csr_before = id(comp._store_csr)
        rows = []
        for a, v, p in dp.FACTS:
            ans, tr = dp.ask_and_trace(comp, a, v)
            if ans is None or not tr:
                rows.append({"want": p, "ans": ans, "abstained": True})
                continue
            h, n = dp._host_mrc(tr), dp._neural_mrc(tr)
            ch, cn = dp._confident(organ, h)["confident"], dp._confident(organ, n)["confident"]
            rows.append({"want": p, "ans": ans, "abstained": False, "correct": ans == p, "host_mrc": h,
                         "neural_mrc": n, "host_conf": ch, "neural_conf": cn, "straddle": ch != cn})
        comp.store_conns = list(base)
        per.append({"sigma": sigma, "csr_rebuilt_during_level": id(comp._store_csr) != csr_before, "rows": rows})
    ans_rows = [r for s in per for r in s["rows"] if not r["abstained"]]
    strad = [r for r in ans_rows if r["straddle"]]
    return {"per_sigma": per, "n_answered": len(ans_rows),
            "n_answered_wrong": sum(not r["correct"] for r in ans_rows),
            "n_straddle": len(strad), "n_straddle_wrong": sum(not r["correct"] for r in strad),
            "n_straddle_host_hedges_neural_confident": sum(1 for r in strad if r["neural_conf"] and not r["host_conf"]),
            "wrong_answered_host_confident": sum(1 for r in ans_rows if not r["correct"] and r["host_conf"]),
            "wrong_answered_neural_confident": sum(1 for r in ans_rows if not r["correct"] and r["neural_conf"]),
            "csr_rebuilt_levels": [s["sigma"] for s in per if s["csr_rebuilt_during_level"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    if set(seeds) & {42, 43, 44, 100, 101, 102}:
        ap.error("dev seeds only")
    dp, draft_sha = load_draft_probe()
    res = {"label": "DEV, not a verdict", "draft_ref": DRAFT_REF, "draft_sha": draft_sha, "per_seed": {}}
    for seed in seeds:
        res["per_seed"][seed] = run_seed(dp, seed)
        print(seed, {k: v for k, v in res["per_seed"][seed].items() if k != "per_sigma"}, flush=True)
    tot = {}
    for k in ("n_answered", "n_answered_wrong", "n_straddle", "n_straddle_wrong",
              "n_straddle_host_hedges_neural_confident", "wrong_answered_host_confident",
              "wrong_answered_neural_confident"):
        tot[k] = sum(r[k] for r in res["per_seed"].values())
    res["pooled"] = tot
    print("POOLED", tot, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
