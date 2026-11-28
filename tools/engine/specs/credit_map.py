"""Verdict of brain/research/2026-11-28-credit-map-T0-prereg.md, by DELEGATION: the bundled units of the one seed are
written to a local directory and the harness's own `brain.experiments.credit_map score` reads them, with its refusals
(no units, mixed model / commit / L / steps, no commit). No rule is restated here. No pick: the spec has no dev grid."""
import json
import shutil
from pathlib import Path

SEED = 7
TAG = "t0"


def _dump(bundles, d):
    from brain.experiments import credit_map as C
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for b in bundles:
        for f, u in b["units"].items():
            C.atomic_write(str(d / f), u)
    return d


def pick(dev, ctx):
    """Unused (no dev grid): kept so a spec with `pick: pick` would still run the one seed."""
    return {"params": f"--arms all --tag {TAG}"}


def verdict(six, pick, ctx):
    from brain.experiments import credit_map as C
    if set(six) != {SEED}:
        return {"verdict": "ERROR", "summary": f"expected exactly seed {SEED}, got {sorted(six)}"}
    d = _dump(six.values(), Path(ctx["res"]) / "t0_units")
    out = Path(ctx["res"]) / "harness_verdict.json"
    out.unlink(missing_ok=True)
    try:
        C.main(["score", "--dir", str(d), "--tag", TAG, "--seed", str(SEED), "--out", str(out)])
    except SystemExit as e:
        if e.code not in (None, 0):
            return {"verdict": "ERROR", "summary": f"harness score refused: {e.code}"}
    if not out.exists():
        return {"verdict": "ERROR", "summary": "harness score wrote no verdict"}
    v = json.loads(out.read_text())
    s = f"{v['verdict']}; carriers {v.get('carriers')}; cos over R {v.get('cos_mean_R')}"
    if v.get("why"):
        s += f"; why {v['why']}"
    return {**v, "summary": s}
