"""Pick and verdict rules of brain/research/2026-11-27-local-sleep-shy-prereg-DRAFT.md, transcribed as code (no new thresholds)."""
import json
import math
from pathlib import Path

LRS, GS = (0.1, 0.3), (0.8, 0.6, 0.4, 0.25)
REF = "brain/experiments/results/2026-11-27-local-sleep/dev_s7_04b/ref.json"   # pinned gradient KL and no_sleep recall


def pick(dev, ctx):
    """Seed-7 dev: g=1 excluded; local drift <= 1.25 x pinned gradient KL and recall >= no_sleep + 2; max recall, then larger g, then lower drift."""
    ref = json.loads((Path(ctx["repo"]) / REF).read_text())["arms"]
    cap, floor = 1.25 * ref["gradient"]["generic_kl"], ref["no_sleep"]["recall"] + 2
    cands = []
    for res in dev.values():
        if round(res.get("lr", -1), 6) not in LRS or res.get("feedback") != "transport":
            continue
        for g in GS:
            r = res["arms"].get(f"local@g{g:g}")
            if r and r["generic_kl"] <= cap and r["recall"] >= floor:
                cands.append((r["recall"], g, -r["generic_kl"], res["lr"], r))
    if not cands:
        return None
    rec, g, nkl, lr, _ = max(cands, key=lambda c: c[:3])
    return {"lr": lr, "g": g, "recall": rec, "kl": -nkl, "cap": cap,
            "params": f"--arms no_sleep,gradient,local,local_shuffled --feedback transport --lr {lr} --decay 1e-3 "
                      f"--downscale-grid 1,{g:g} --downscale-factors B"}


def _arm(res, name, g=None):
    a = res["arms"]
    for k in ([f"{name}@g{g:g}"] if g is not None else []) + [name] + sorted(x for x in a if x.startswith(name + "@")):
        if k in a:
            return a[k]
    raise KeyError(name)


def verdict(six, pick, ctx):
    """Six-seed clauses (a)-(f) and the GO-SHY / GO-PLAIN-LOCAL / UNDEFINED / NEGATIVE split of the prereg."""
    g = pick["g"]
    mean = lambda arm, key, gg=None: sum(_arm(r, arm, gg)[key] for r in six.values()) / len(six)
    G1, Gg = mean("gradient", "recall", 1), mean("gradient", "recall", g)
    ns, L, S = mean("no_sleep", "recall"), mean("local", "recall", g), mean("local_shuffled", "recall", g)
    kl = lambda arm, gg: mean(arm, "generic_kl", gg)
    norms = [_arm(r, "local", gg)["adapter_norm"] for r in six.values() for gg in (1, g)]
    num = {"G1": G1, "Gg": Gg, "no_sleep": ns, "local_g": L, "shuffled_g": S, "kl_local_g": kl("local", g),
           "kl_local_1": kl("local", 1), "kl_grad_1": kl("gradient", 1), "kl_grad_g": kl("gradient", g)}
    if G1 < 2 or any(not math.isfinite(n) or n == 0 for n in norms) or S > 1.0:
        return {"verdict": "UNDEFINED", "summary": f"reference or control invalid (G1={G1:.2f}, shuffled={S:.2f})", "numbers": num}
    beats = sum(_arm(r, "local", g)["recall"] > _arm(r, "local_shuffled", g)["recall"] for r in six.values())
    c = {"a": L >= max(0.5 * max(G1, Gg), ns + 2),
         "b": beats >= 5 and S <= 1.0,
         "c": num["kl_local_g"] <= 1.25 * num["kl_grad_1"],
         "d": mean("local", "false_recall", g) <= mean("gradient", "false_recall", 1) + 0.5,
         "e": L >= 0.5 * Gg and num["kl_local_g"] <= 1.25 * num["kl_grad_g"],
         "f": num["kl_local_1"] > 1.25 * num["kl_grad_1"]}
    abcde = all(c[k] for k in "abcde")
    v = ("GO-SHY" if c["f"] else "GO-PLAIN-LOCAL") if abcde else "NEGATIVE"
    failed = [k for k in "abcdef" if not c[k]]
    return {"verdict": v, "clauses": c, "numbers": num,
            "summary": f"local@g{g:g} recall {L:.2f} vs gradient {G1:.2f}, drift {num['kl_local_g']:.3f} vs {num['kl_grad_1']:.3f}"
                       + (f"; failed clauses {failed}" if failed else "")}
