# Capability: Lifelong learning
# Record: brain/research/2026-11-23-kp-identity-prereg.md (section 2 Step-0 gate, amendment A2); _scratch/kpid/gate.py
# Expected-fail: yes, as the prereg names it (masking may give back the language win; one night may not show the cost)
"""AWS job wrapper for the K-P identity gate (prereg A2): one seed, all four arms B, KP, KPU, KP0 on one GPU, CUDA fp32
(TF32 off), each arm's night, drift and probes in the order of _scratch/kpid/pool/run_pool_gate.sh. Host only: it calls
the committed gate.py steps and adds nothing to them. Run by tools/brain_batch_run.sh as
  <name> brain.experiments.kpid_aws --seed S --sha SHA --model ../models/<file>.pth   (EXTRA adds --device cuda)
with AWS_BRAIN_UPLOAD having placed ~/brainv2/kpid_repo/_scratch/{kpid/gate.py,kpid/drift_read.py,stage2/read/lifeS1F_s<S>.json}
and ~/brainv2/kpid_src/src_s<S>. Outputs: ~/brainv2/results/kpid_s<S>/ (homes, drift/probe JSONs, logs) and --out."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ARMS = ("B", "KP", "KPU", "KP0")


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sha", required=True, help="expected sha256 of src_s<S>/cortex_adapters.pt")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda", choices=("cuda",))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = os.path.expanduser("~/brainv2")
    repo = os.path.join(base, "kpid_repo")
    gate = os.path.join(repo, "_scratch", "kpid", "gate.py")
    src = os.path.join(base, "kpid_src", f"src_s{a.seed}")
    out = os.path.join(base, "results", f"kpid_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    link = os.path.join(repo, "brain")
    if not os.path.exists(link):
        os.symlink(os.path.join(base, "code", "brain"), link)
    model = os.path.abspath(a.model)
    got = _sha(os.path.join(src, "cortex_adapters.pt"))
    rec = {"seed": a.seed, "device": "cuda fp32", "src_sha": got, "want_sha": a.sha, "steps": []}
    prog = os.path.join(out, "progress.json")   # --out is written only on success, so a rerun is never skipped

    def save(path):
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)

    if got != a.sha:
        rec["error"] = "source sha mismatch"
        save(prog)
        return 1
    env = dict(os.environ, PYTHONPATH=repo)
    dev = ["--device", "cuda", "--cuda-precision", "fp32", "--model", model]
    for arm in ARMS:
        home = os.path.join(out, f"home_{arm}_s{a.seed}")
        bf = ["--bank-from", os.path.join(out, f"home_KP_s{a.seed}")] if arm == "KPU" else []
        steps = [("night", home + ".result.json", ["night", arm, src, home, str(a.seed)] + bf),
                 ("drift", os.path.join(out, f"drift_{arm}_s{a.seed}.json"),
                  ["drift", arm, src, home, str(a.seed), os.path.join(out, f"drift_{arm}_s{a.seed}.json")]),
                 ("probe", os.path.join(out, f"probe_{arm}_s{a.seed}.json"),
                  ["probe", arm, home, str(a.seed), os.path.join(out, f"probe_{arm}_s{a.seed}.json")])]
        for name, done, args in steps:
            if os.path.exists(done):
                continue
            t0 = time.time()
            with open(os.path.join(out, f"{name}_{arm}_s{a.seed}.log"), "w") as log:
                rc = subprocess.run([sys.executable, "-u", gate] + args + dev, env=env, cwd=repo,
                                    stdout=log, stderr=subprocess.STDOUT).returncode
            rec["steps"].append({"arm": arm, "step": name, "rc": rc, "seconds": round(time.time() - t0, 1)})
            save(prog)
            if rc != 0:
                rec["error"] = f"{name} {arm} rc {rc}"
                save(prog)
                return 1
    rec["done"] = True
    save(prog)
    save(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
