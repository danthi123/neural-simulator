"""Verdict of the engine smoke: every seed's job imported every module, ran exactly once, under this commit."""


def verdict(six, pick, ctx):
    per, bad = {}, []
    for s, r in sorted(six.items()):
        c = {"imports": bool(r.get("imports")) and all(m.get("ok") for m in r["imports"].values()),
             "once": r.get("count") == 1,
             "commit": r.get("commit") == ctx.get("commit")}
        per[s] = {**c, "host": r.get("host"), "imports_detail": r.get("imports"), "torch": r.get("torch"),
                  "cgroup": r.get("cgroup"), "python": r.get("python")}
        bad += [f"s{s}:{k}" for k, ok in c.items() if not ok]
    return {"verdict": "SMOKE-FAIL" if bad or not six else "SMOKE-OK", "per_seed": per,
            "summary": f"{len(six)} node jobs" + (f"; failed {bad}" if bad else "; all imported, once, at this commit")}
