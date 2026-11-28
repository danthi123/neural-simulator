"""Pick and verdict of brain/research/2026-11-28-three-factor-sleep-prereg.md, by DELEGATION: the bundled units are
written to a local directory and the harness's own CLI (`brain.experiments.three_factor_sleep pick` / `score --pick`)
reads them, with all its refusals (missing refs / cells, mixed model / commit / prompts / config). No rule is restated
here."""
import json
import shutil
from pathlib import Path


def _dump(bundles, d):
    from brain.experiments import three_factor_sleep as T
    shutil.rmtree(d, ignore_errors=True)          # only this run's units: no same-commit leftovers
    d.mkdir(parents=True)
    for b in bundles:
        for f, u in b["units"].items():
            T.atomic_write(str(d / f), u)
    return d


def _call(argv):
    """Run the harness CLI; SystemExit (a refusal) comes back as its message."""
    from brain.experiments import three_factor_sleep as T
    try:
        T.main(argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            return str(e.code)
    return None


def pick(dev, ctx):
    from brain.experiments import three_factor_sleep as T
    d = _dump(dev.values(), Path(ctx["res"]) / "dev_units")
    out = Path(ctx["res"]) / "pick.json"
    out.unlink(missing_ok=True)
    err = _call(["pick", "--dir", str(d), "--out", str(out)])
    if err or not out.exists():
        return {"stop": f"harness pick refused: {err}"}
    p = json.loads(out.read_text())
    if p.get("verdict") in ("NEGATIVE-DEV", "KILL-DEV"):
        # the engine then writes its own pick.json ("NEGATIVE-DEV") over this one, so the harness's word (KILL-DEV
        # retires the method at this budget, NEGATIVE-DEV does not) is kept in harness_pick_dev.json
        shutil.copyfile(out, Path(ctx["res"]) / "harness_pick_dev.json")
        return None                                   # this model's six seeds are not run
    if p.get("verdict") != "PICKED":                  # UNDEFINED-DEV
        return {"stop": f"harness pick verdict {p.get('verdict')}: {p.get('why')}", "harness_pick": p}
    lr, sg = p["pick"]["lr"], p["pick"]["sigma"]
    return {"harness_pick": p, "lr": lr, "sigma": sg,
            "params": f"--arms {','.join(T.ARMS)} --lr {lr} --sigma {sg} --tag six"}


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
