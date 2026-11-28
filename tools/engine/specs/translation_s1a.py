"""Verdict of brain/research/2026-11-28-translation-s1a-prereg.md, by DELEGATION: the bundled units of the three KR lives
are written to a local directory and the harness's own `brain.experiments.translation_s1a score` reads them (its refusals:
missing seeds or units, mixed commit / model / layer / tag, failed reproduction or engagement). No rule is restated here.
The gain is chosen inside the verdict from each life's DEV_FACTS36 units; the pick only fixes the six-seed memcap."""
import json
import shutil
from pathlib import Path

SEEDS = (319, 320, 321)


def pick(dev, ctx):
    """No dev grid. The six-seed jobs run at the spec's memory_gb (13 GB; engine mem() takes min(memcap, memory_gb)), not
    at ceil(1.3 x smoke peak + 1): the smoke is limited to 3 questions and one gain, so its peak may sit below the full job's."""
    return {"params": "", "memcap": {s: int(ctx["memory_gb"]) for s in SEEDS}}


def _dump(bundles, d):
    from brain.experiments import voice_ceiling_oracle as VC
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    for b in bundles:
        for f, u in b["units"].items():
            VC._save_json(str(d / f), u)
    return d


def verdict(six, pick, ctx):
    from brain.experiments import translation_s1a as T
    if set(six) != set(SEEDS):
        return {"verdict": "ERROR", "summary": f"expected seeds {list(SEEDS)}, got {sorted(six)}"}
    d = _dump(six.values(), Path(ctx["res"]) / "s1a_units")
    out = Path(ctx["res"]) / "harness_verdict.json"
    out.unlink(missing_ok=True)
    T.main(["score", "--dir", str(d), "--seeds", *map(str, SEEDS), "--out", str(out)])
    if not out.exists():
        return {"verdict": "ERROR", "summary": "harness score wrote no verdict"}
    v = json.loads(out.read_text())
    s = f"{v['verdict']}; primary DENSE-OFF {(v.get('primary') or {}).get('diff')} (p {(v.get('primary') or {}).get('p')})"
    if v.get("why"):
        s += f"; why {v['why']}"
    if v.get("failed"):
        s += f"; failed {v['failed']}"
    return {**v, "summary": s}
