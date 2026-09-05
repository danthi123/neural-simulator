"""ONE-BRAIN SINGLE-POOL FLIP — brain-chat REGRESSION harness (the `BRAIN_ONEBRAIN_SINGLE_POOL` soak).

Asserts that flipping the single-pool merge ON (all 4 core organs on ONE `merge_organs` pool) is
ANSWER-PRESERVING vs the current TWO-pool production default (surprise+world-model on `MergedSubstrate` #1,
metacog+pragmatic on `MergedSubstrate2` #2), across 6 seeds, through the SAME LIVE organ read paths the
`webapp/server.py` chat handler uses (each organ's `get_organ()` singleton + its own read-isolation) — NOT the
organ-read verify's harness-driven full-snapshot-restore. This is the regression the organ-read GO named as the
gate for the DEFAULT-ON flip; metacog + pragmatic are default-ON in live chat, so they are the load-bearing
targets.

WHY SUBPROCESS-ISOLATED. The 4 organs + the pools are PROCESS singletons (`_ORGAN`, `_POOL`, the two
`MergedSubstrate*` caches). The two configs (flag ON vs unset) cannot coexist cleanly in one process. Each
(seed, config) therefore runs in its OWN fresh subprocess (clean singletons + RNG), building the 4 organs via
their REAL `get_organ()` singletons — so this exercises the ACTUAL production wiring
(`onebrain_single_pool_production.single_pool_enabled` -> `get_single_pool`), not a bespoke pool. The worker
dumps a fixed live-answer battery; the controller compares ON-vs-OFF per organ per seed.

THE ANSWER BATTERY (the LIVE chat reads, not internal Hz):
  surprise    -> the `surprised` bool over a 4-item (stored, asserted) battery (confirm / contradict / novel).
  world-model -> the (pred_sign(+ctx), pred_sign(-ctx)) prediction signs + the (expected, violated) `surprised` bits.
  metacog     -> the `confident` bool over evidence {0.1, 0.5, 0.9} (the production `nmda_norm` read).
  pragmatic   -> the enriched scalar-implicature interpretation over {some, all, none}.

GATE (per seed): every organ's ON answer == its OFF (two-pool) answer. ALL-GO == 6/6 seeds, all 4 organs.

Reproduce (numpy smoke, 1 seed — proves the harness RUNS + an early answer-preservation read):
    SIM_BACKEND=numpy python -m research.runners._onebrain_single_pool_flip_regression \
        --seeds 42 --out research/findings/raw/_onebrain_single_pool_flip_smoke.json

The decisive 6-seed run is a cupy brain-chat soak (ONE GPU brain proc at a time) — QUEUE it on gpu_queue.sh:
    SIM_BACKEND=cupy python -m research.runners._onebrain_single_pool_flip_regression \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_onebrain_single_pool_flip_6seed.json

NO `sim/` edit. The pools are the tiny (N=2034) numpy/cupy nets the organ-read GO validated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── the fixed live-answer battery, evaluated on the 4 organs' REAL production read APIs ──
_SURPRISE_BATTERY = [("alpha", "alpha"), ("beta", "gamma"), ("delta", "omega"), ("kappa", "kappa")]
_METACOG_EVIDENCE = (0.1, 0.5, 0.9)
_PRAGMATIC_UTTS = ("some", "all", "none")


def _worker_answers(seed: int) -> dict:
    """Build the 4 core organs via their LIVE `get_organ()` singletons (whatever `BRAIN_ONEBRAIN_SINGLE_POOL`
    resolves to in THIS process's env) and read the fixed live-answer battery. The organ read APIs are exactly
    the ones `webapp/server.py`'s chat handler calls."""
    from research.runners.onebrain_single_pool_production import single_pool_enabled
    import research.runners.surprise_production_organ as SO
    import research.runners.worldmodel_production_organ as WM
    import research.runners.metacog_production_organ as MC
    import research.runners.pragmatic_production_organ as PR

    surprise = SO.get_organ(seed=seed)
    worldmodel = WM.get_organ(seed=seed)
    metacog = MC.get_organ(seed=seed)
    pragmatic = PR.get_organ(seed=seed)

    surprise.ensure_built(); worldmodel.ensure_built(); metacog.ensure_built(); pragmatic.ensure_built()

    surprise_ans = [bool(surprise.judge("agent", "acts", ps, pa)["surprised"]) for ps, pa in _SURPRISE_BATTERY]

    wm_pred = [int(worldmodel.expectation(+1)["pred_sign"]), int(worldmodel.expectation(-1)["pred_sign"])]
    # (expected turn, violated turn) surprised bits for each context sign
    wm_surp = [
        bool(worldmodel.read_surprise(+1, +1)["surprised"]), bool(worldmodel.read_surprise(+1, -1)["surprised"]),
        bool(worldmodel.read_surprise(-1, -1)["surprised"]), bool(worldmodel.read_surprise(-1, +1)["surprised"]),
    ]

    metacog_ans = [bool(metacog.judge(e)["confident"]) for e in _METACOG_EVIDENCE]

    pragmatic_ans = [str(pragmatic.interpret(u)["enriched_interpretation"]) for u in _PRAGMATIC_UTTS]

    return {
        "seed": int(seed),
        "single_pool_enabled": bool(single_pool_enabled()),
        "surprise": surprise_ans,
        "worldmodel_pred": wm_pred,
        "worldmodel_surprised": wm_surp,
        "metacog": metacog_ans,
        "pragmatic": pragmatic_ans,
    }


def _run_worker(seed: int, single_pool: bool) -> dict:
    """Spawn a FRESH subprocess (clean singletons + RNG) with `BRAIN_ONEBRAIN_SINGLE_POOL` set/unset, collect its
    JSON answer battery. Inherits SIM_BACKEND from the parent (numpy smoke / cupy soak)."""
    env = dict(os.environ)
    if single_pool:
        env["BRAIN_ONEBRAIN_SINGLE_POOL"] = "1"
    else:
        env["BRAIN_ONEBRAIN_SINGLE_POOL"] = "0"   # FORCE two-pool. The single-pool flag flipped to default-ON
        # (2026-09-05, c1343b238), so popping the env no longer disables it — the off arm MUST set 0 explicitly or
        # the A/B is confounded (both arms single-pool => trivially all_same, meaningless). The `not off_flag`
        # guard in verify_seed() correctly reports GO=False when this is wrong, which is how the confound was caught.
    cmd = [sys.executable, "-m", "research.runners._onebrain_single_pool_flip_regression",
           "--worker", "--seed", str(seed)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        raise RuntimeError(f"worker(seed={seed}, single_pool={single_pool}) failed rc={proc.returncode}\n"
                           f"STDERR:\n{proc.stderr[-4000:]}")
    # the worker prints exactly one JSON line prefixed by the sentinel (other prints are ignored)
    line = next(l for l in reversed(proc.stdout.splitlines()) if l.startswith("__ANSWERS__ "))
    return json.loads(line[len("__ANSWERS__ "):])


_ORGAN_KEYS = ("surprise", "worldmodel_pred", "worldmodel_surprised", "metacog", "pragmatic")


def verify_seed(seed: int) -> dict:
    on = _run_worker(seed, single_pool=True)
    off = _run_worker(seed, single_pool=False)
    per_key = {}
    for k in _ORGAN_KEYS:
        per_key[k] = {"on": on[k], "off": off[k], "same": bool(on[k] == off[k])}
    # roll the two world-model reads into one organ verdict
    organ_same = {
        "surprise": per_key["surprise"]["same"],
        "worldmodel": bool(per_key["worldmodel_pred"]["same"] and per_key["worldmodel_surprised"]["same"]),
        "metacog": per_key["metacog"]["same"],
        "pragmatic": per_key["pragmatic"]["same"],
    }
    go = bool(all(organ_same.values()) and on["single_pool_enabled"] and not off["single_pool_enabled"])
    return {"seed": int(seed), "per_key": per_key, "organ_answer_preserved": organ_same,
            "on_flag": on["single_pool_enabled"], "off_flag": off["single_pool_enabled"], "GO": go}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--worker", action="store_true", help="internal: build 4 organs + print the answer battery")
    args = ap.parse_args()

    if args.worker:
        ans = _worker_answers(args.seed)
        print("__ANSWERS__ " + json.dumps(ans), flush=True)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=== ONE-BRAIN SINGLE-POOL FLIP — brain-chat answer-preservation regression ===")
    print("    BRAIN_ONEBRAIN_SINGLE_POOL=1 (all 4 organs on ONE merge_organs pool) vs the two-pool default")
    per_seed = [verify_seed(s) for s in seeds]
    for p in per_seed:
        flags_ok = p["on_flag"] and not p["off_flag"]
        print(f"  [seed {p['seed']}] flags_ok={flags_ok} answer_preserved="
              f"{ {k: v for k, v in p['organ_answer_preserved'].items()} } -> GO={p['GO']}", flush=True)
        for k in _ORGAN_KEYS:
            if not p["per_key"][k]["same"]:
                print(f"      DIVERGE {k}: on={p['per_key'][k]['on']} off={p['per_key'][k]['off']}", flush=True)

    n = len(seeds)
    n_go = sum(p["GO"] for p in per_seed)
    per_organ = {k: sum(p["organ_answer_preserved"][k] for p in per_seed)
                 for k in ("surprise", "worldmodel", "metacog", "pragmatic")}
    all_go = bool(n_go == n and n > 0)
    print("\n=== VERDICT (single-pool flip answer-preservation) ===")
    for k in ("surprise", "worldmodel", "metacog", "pragmatic"):
        print(f"  {k:11s} answer_preserved {per_organ[k]}/{n}")
    print(f"  ALL-ORGAN ALL-SEED answer-preservation: {n_go}/{n}  ->  ALL-GO={all_go}")

    payload = {"mode": "onebrain_single_pool_flip_regression", "seeds": seeds,
               "per_seed": per_seed, "per_organ": per_organ, "n_go": n_go, "n_seeds": n, "all_go": all_go,
               "backend": os.environ.get("SIM_BACKEND", "(default)")}

    try:
        from tools.verdict import Verdict
        v = Verdict("one-brain SINGLE-POOL flip — brain-chat answer-preservation (4 organs on ONE pool vs two-pool)")
        v.require("every organ's live chat answer preserved under the single-pool flip, every seed", n_go, expect=n)
        v.disabled("cross-region interaction (the one-brain INTEGRATION goal)",
                   why="MIGRATION flip: the single pool has zero cross-organ synapses by construction")
        decided = v.decide(go=all_go)
        payload.update(decided)
    except Exception as _ve:
        payload["verdict_note"] = f"Verdict unavailable ({type(_ve).__name__}: {_ve})"

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
