"""Pick and verdict of brain/research/2026-11-28-local-rule-v3q-prereg-DRAFT.md, by DELEGATION: the bundled units are
written to a local directory and the harness's own CLI (`brain.experiments.local_rule_v3q pick` / `score --pick`) reads
them, with all its refusals (missing G0Q / ref / cells, mixed model / commit / prompts / cue set). No rule is restated."""
import json
import shutil
from pathlib import Path


def _dump(bundles, d):
    from brain.experiments import local_rule_v3q as Q
    shutil.rmtree(d, ignore_errors=True)          # only this run's units: no same-commit leftovers
    d.mkdir(parents=True)
    for b in bundles:
        for f, u in b["units"].items():
            Q.atomic_write(str(d / f), u)
    return d


def _call(argv):
    """Run the harness CLI; SystemExit (a refusal) comes back as its message."""
    from brain.experiments import local_rule_v3q as Q
    try:
        Q.main(argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            return str(e.code)
    return None


def pick(dev, ctx):
    from brain.experiments import local_rule_v3q as Q
    d = _dump(dev.values(), Path(ctx["res"]) / "dev_units")
    out = Path(ctx["res"]) / "pick.json"
    out.unlink(missing_ok=True)
    err = _call(["pick", "--dir", str(d), "--out", str(out)])
    if err or not out.exists():
        return {"stop": f"harness pick refused: {err}"}
    p = json.loads(out.read_text())
    if p.get("verdict") == "NEGATIVE-DEV":
        return None                                   # this model's six seeds are not run
    if p.get("verdict") != "PICKED":                  # STOP-G0Q, UNDEFINED-DEV
        return {"stop": f"harness pick verdict {p.get('verdict')}: {p.get('why')}", "harness_pick": p}
    lr, keep = p["pick"]["lr"], p["pick"]["keep"]
    return {"harness_pick": p, "lr": lr, "keep": keep,
            "params": f"--arms {','.join(Q.ARMS)} --lr {lr} --keep {keep} --tag six"}


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
    rep = v.get("report") or {}
    g = (rep.get("graded") or {}).get("v3q") or {}
    pt = rep.get("paraphrase_transfer") or {}
    return {**v, "summary": f"{v['verdict']}; failed clauses {failed}" + (f"; {v['reading']}" if v.get("reading") else "")
            + f"; graded v3q logprob - no_sleep {g.get('mean_logprob_minus_no_sleep')}"
            + f"; eval-cue share {rep.get('eval_cue_gain_share')}" + ("; CAVEAT eval-cue carried"
                                                                       if rep.get("eval_cue_carried") else "")
            + f"; paraphrase v3q - no_sleep {pt.get('v3q_minus_no_sleep')}"}
