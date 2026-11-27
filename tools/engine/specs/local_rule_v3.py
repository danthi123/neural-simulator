"""Pick and verdict of brain/research/2026-11-27-local-rule-v3-prereg.md, by DELEGATION: the bundled units are written to a
local directory and the harness's own CLI (`brain.experiments.local_rule_v3 pick` / `score --pick`) reads them, with all
its refusals (missing G0 / ref / cells, mixed model / commit / prompts). No rule is restated here."""
import json
import shutil
from pathlib import Path


def _dump(bundles, d):
    from brain.experiments import local_rule_v3 as V
    shutil.rmtree(d, ignore_errors=True)          # only this run's units: no same-commit leftovers
    d.mkdir(parents=True)
    for b in bundles:
        for f, u in b["units"].items():
            V.atomic_write(str(d / f), u)
    return d


def _call(argv):
    """Run the harness CLI; SystemExit (a refusal) comes back as its message."""
    from brain.experiments import local_rule_v3 as V
    try:
        V.main(argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            return str(e.code)
    return None


def pick(dev, ctx):
    from brain.experiments import local_rule_v3 as V
    d = _dump(dev.values(), Path(ctx["res"]) / "dev_units")
    out = Path(ctx["res"]) / "pick.json"
    out.unlink(missing_ok=True)
    err = _call(["pick", "--dir", str(d), "--out", str(out)])
    if err or not out.exists():
        return {"stop": f"harness pick refused: {err}"}
    p = json.loads(out.read_text())
    if p.get("verdict") == "NEGATIVE-DEV":
        return None                                   # this model's six seeds are not run
    if p.get("verdict") != "PICKED":                  # STOP-G0, UNDEFINED-DEV
        return {"stop": f"harness pick verdict {p.get('verdict')}: {p.get('why')}", "harness_pick": p}
    lr, keep = p["pick"]["lr"], p["pick"]["keep"]
    return {"harness_pick": p, "lr": lr, "keep": keep,
            "params": f"--arms {','.join(V.ARMS)} --lr {lr} --keep {keep} --tag six"}


def verdict(six, pick, ctx):
    d = _dump(six.values(), Path(ctx["res"]) / "six_units")
    pf, out = Path(ctx["res"]) / "pick.json", Path(ctx["res"]) / "harness_verdict.json"
    pf.write_text(json.dumps(pick["harness_pick"]))
    out.unlink(missing_ok=True)
    err = _call(["score", "--dir", str(d), "--tag", "six", "--out", str(out), "--pick", str(pf)])
    if err or not out.exists():
        return {"verdict": "ERROR", "summary": f"harness score refused: {err}"}
    v = json.loads(out.read_text())
    failed = [k for k, ok in (v.get("clauses") or {}).items() if not ok]
    return {**v, "summary": f"{v['verdict']}; failed clauses {failed}" + (f"; {v['reading']}" if v.get("reading") else "")}
