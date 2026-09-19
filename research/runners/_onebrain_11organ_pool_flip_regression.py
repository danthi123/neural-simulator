"""ONE-BRAIN 11-ORGAN POOL FLIP — brain-chat REGRESSION harness (the `BRAIN_ONEBRAIN_WAVE3_POOL` soak).

The gate for the DEFAULT-ON flip that puts every WIRED cortical organ on the ONE shared 11-organ `merge_organs`
pool (`onebrain_wave3_pool_production.get_merged_cortical_pool` -> `get_wave3_pool`). Extends the shipped
`_onebrain_single_pool_flip_regression` (4 core organs) to the 8 organs the flip actually routes:
surprise + world-model + metacog + pragmatic (min_wave=1) + comprehension (min_wave=1) + self_schema + curiosity +
causal_whatif (min_wave=2). THREE organs are documented EXCLUSIONS not covered here (source_provenance's shipped
incremental wrapper, prospective_memory scope-reduced, d6_multiref_wm per-session in webapp) — see
`onebrain_wave3_pool_production.py`'s docstring.

WHAT IT ASSERTS, per seed, ON (`BRAIN_ONEBRAIN_WAVE3_POOL=1`) vs OFF (`=0`, the byte-identical escape), each arm a
FRESH subprocess (clean singletons + RNG) so both arms share an identical background-noise trajectory:
  (A) ANSWER-PRESERVATION — every organ's LIVE categorical read (the decision the webapp chat handler exposes, not
      a continuous Hz/margin) is identical ON vs OFF. This is the "flipping the pool ON does not change any
      answer" claim, through the ACTUAL production wiring (each organ's real `get_organ()` singleton).
  (B) ONE BRAIN, NO DOUBLE-BUILD (the coherence claim the flip lives or dies on) — in the ON arm all 8 organs'
      `_shared` is the SAME pool object (n_distinct_pools==1) AND it is the 11-organ wave3 pool. If the flip were
      incoherent (organs split across single/wave1/wave2/wave3 pools), n_distinct_pools would be >1 here.
  (C) ESCAPE REVERTS — in the OFF arm no organ is on the wave3 pool; the base 4 fall back co-resident on the
      single_pool (n_base4_distinct==1), the rest to their standalone bridges. This is the byte-identical revert.

GATE (per seed): (A) all 8 organs answer-preserved AND (B) ON one-brain AND (C) OFF escape AND the flags read
correctly (ON arm wave3 on, OFF arm wave3 off). ALL-GO == every seed.

CAUSAL note: causal_whatif's deep read (`what_if`/`why`) needs a live composer (heavy); here it is covered by
the WIRING + pool-identity check (B/C) only — its answer-preservation is carried by the wave2/wave3 organread GO
(`2026-09-17-onebrain-wave3-...-GO.md`, co-residence byte-identical 6/6) + the integrated battery.

Reproduce (numpy smoke, 1 seed — proves the harness RUNS + an early answer-preservation + one-brain read):
    SIM_BACKEND=numpy python -m research.runners._onebrain_11organ_pool_flip_regression \
        --seeds 42 --out research/findings/raw/_onebrain_11organ_pool_flip_smoke.json

The decisive 6-seed run is a cupy brain soak (ONE GPU brain proc at a time) — QUEUE it on gpu_queue.sh:
    SIM_BACKEND=cupy python -m research.runners._onebrain_11organ_pool_flip_regression \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_onebrain_11organ_pool_flip_6seed.json

NO `sim/` edit. The pool is the tiny (N~2034+) numpy/cupy net the 11-organ organread GO validated.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# ── the fixed live-answer batteries, evaluated on each organ's REAL production read API ──
_SURPRISE_BATTERY = [("alpha", "alpha"), ("beta", "gamma"), ("delta", "omega"), ("kappa", "kappa")]
_METACOG_EVIDENCE = (0.1, 0.5, 0.9)
_PRAGMATIC_UTTS = ("some", "all", "none")
_COMP_BATTERY = [("dog", "chase", "cat"), ("cat", "eat", "fish"), ("fox", "bite", "bird")]
_CURIOSITY_NOVELTY = (0.95, 0.0)


def _worker_answers(seed: int) -> dict:
    """Build the 8 WIRED organs via their LIVE `get_organ()` singletons (whatever `BRAIN_ONEBRAIN_WAVE3_POOL`
    resolves to in THIS process's env) and read the fixed categorical answer battery + the pool-identity signature.
    The organ read APIs are exactly the ones `webapp/server.py`'s chat handler calls (`_get_*_organ`)."""
    from research.runners.onebrain_wave3_pool_production import wave3_pool_enabled, get_merged_cortical_pool
    import research.runners.surprise_production_organ as SO
    import research.runners.worldmodel_production_organ as WM
    import research.runners.metacog_production_organ as MC
    import research.runners.pragmatic_production_organ as PR
    import research.runners.comprehension_production_organ as CO
    import research.runners.self_schema_production_organ as SS
    import research.runners.curiosity_production_organ as CU
    import research.runners.causal_whatif_production_organ as CA

    surprise = SO.get_organ(seed=seed)
    worldmodel = WM.get_organ(seed=seed)
    metacog = MC.get_organ(seed=seed)
    pragmatic = PR.get_organ(seed=seed)
    comprehension = CO.get_organ(seed=seed)
    self_schema = SS.get_organ(seed=seed)
    curiosity = CU.get_organ(seed=seed)
    causal = CA.get_organ(None, seed=seed)   # construction routes shared=; deep read needs a composer (see docstring)

    for o in (surprise, worldmodel, metacog, pragmatic, comprehension, self_schema, curiosity):
        o.ensure_built()

    # ── (A) categorical answer battery ──
    surprise_ans = [bool(surprise.judge("agent", "acts", ps, pa)["surprised"]) for ps, pa in _SURPRISE_BATTERY]
    wm_pred = [int(worldmodel.expectation(+1)["pred_sign"]), int(worldmodel.expectation(-1)["pred_sign"])]
    wm_surp = [
        bool(worldmodel.read_surprise(+1, +1)["surprised"]), bool(worldmodel.read_surprise(+1, -1)["surprised"]),
        bool(worldmodel.read_surprise(-1, -1)["surprised"]), bool(worldmodel.read_surprise(-1, +1)["surprised"]),
    ]
    metacog_ans = [bool(metacog.judge(e)["confident"]) for e in _METACOG_EVIDENCE]
    pragmatic_ans = [str(pragmatic.interpret(u)["enriched_interpretation"]) for u in _PRAGMATIC_UTTS]
    # comprehension: competent (in-lexicon) + comprehended (spiking margin >= threshold) — both categorical decisions
    comp_ans = []
    for n0, v, n1 in _COMP_BATTERY:
        comp_ok = bool(comprehension.competent(n0, v, n1))
        margin = comprehension.read_margin(n0, v, n1)
        comp_ans.append([comp_ok, bool(margin >= comprehension.threshold)])
    self_schema_ans = [str(self_schema.read_author(True)["label"]), str(self_schema.read_author(False)["label"])]
    curiosity_ans = [bool(curiosity.judge(nv)["curious"]) for nv in _CURIOSITY_NOVELTY]

    # ── (B/C) pool-identity signature: is every wired organ on ONE object, and is it the wave3 pool? ──
    organs = {
        "surprise": surprise, "worldmodel": worldmodel, "metacog": metacog, "pragmatic": pragmatic,
        "comprehension": comprehension, "self_schema": self_schema, "curiosity": curiosity, "causal": causal,
    }
    shared_ids = {k: (id(o._shared) if getattr(o, "_shared", None) is not None else None) for k, o in organs.items()}
    non_null = [i for i in shared_ids.values() if i is not None]
    n_distinct_pools = len(set(non_null))
    n_standalone = sum(1 for i in shared_ids.values() if i is None)
    base4 = ("surprise", "worldmodel", "metacog", "pragmatic")
    base4_ids = [shared_ids[k] for k in base4 if shared_ids[k] is not None]
    n_base4_distinct = len(set(base4_ids))
    wave3_on = bool(wave3_pool_enabled())
    wave3_pool_id = id(get_merged_cortical_pool(seed, min_wave=1)) if wave3_on else None
    n_on_wave3 = sum(1 for i in shared_ids.values() if wave3_pool_id is not None and i == wave3_pool_id)

    return {
        "seed": int(seed),
        "wave3_enabled": wave3_on,
        "surprise": surprise_ans, "worldmodel_pred": wm_pred, "worldmodel_surprised": wm_surp,
        "metacog": metacog_ans, "pragmatic": pragmatic_ans, "comprehension": comp_ans,
        "self_schema": self_schema_ans, "curiosity": curiosity_ans,
        "n_distinct_pools": int(n_distinct_pools), "n_standalone": int(n_standalone),
        "n_base4_distinct": int(n_base4_distinct), "n_on_wave3": int(n_on_wave3),
        "n_organs": len(organs),
    }


def _run_worker(seed: int, wave3: bool) -> dict:
    """Spawn a FRESH subprocess (clean singletons + RNG) with `BRAIN_ONEBRAIN_WAVE3_POOL` explicitly set, collect
    its JSON answer battery. The OFF arm sets "0" EXPLICITLY (never pop) — the flag is default-ON now, so popping
    would leave it ON and confound the A/B (the flip_offarm_staleness discipline)."""
    env = dict(os.environ)
    env["BRAIN_ONEBRAIN_WAVE3_POOL"] = "1" if wave3 else "0"
    cmd = [sys.executable, "-m", "research.runners._onebrain_11organ_pool_flip_regression",
           "--worker", "--seed", str(seed)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        raise RuntimeError(f"worker(seed={seed}, wave3={wave3}) failed rc={proc.returncode}\n"
                           f"STDERR:\n{proc.stderr[-4000:]}")
    line = next(l for l in reversed(proc.stdout.splitlines()) if l.startswith("__ANSWERS__ "))
    return json.loads(line[len("__ANSWERS__ "):])


# the categorical answer keys compared ON vs OFF (organ-answer-preservation, arm A)
_ANSWER_KEYS = ("surprise", "worldmodel_pred", "worldmodel_surprised", "metacog", "pragmatic",
                "comprehension", "self_schema", "curiosity")
_ORGAN_ROLLUP = {
    "surprise": ("surprise",), "worldmodel": ("worldmodel_pred", "worldmodel_surprised"),
    "metacog": ("metacog",), "pragmatic": ("pragmatic",), "comprehension": ("comprehension",),
    "self_schema": ("self_schema",), "curiosity": ("curiosity",),
}


def verify_seed(seed: int) -> dict:
    on = _run_worker(seed, wave3=True)
    off = _run_worker(seed, wave3=False)
    per_key = {}
    for k in _ANSWER_KEYS:
        per_key[k] = {"on": on[k], "off": off[k], "same": bool(on[k] == off[k])}
    organ_same = {org: bool(all(per_key[k]["same"] for k in keys)) for org, keys in _ORGAN_ROLLUP.items()}
    # (A) answer-preservation across the 7 read-covered organs
    a_ok = bool(all(organ_same.values()))
    # (B) ON arm: one brain, no double-build — all 8 on ONE object, and it is the wave3 pool
    b_ok = bool(on["wave3_enabled"] and on["n_distinct_pools"] == 1 and on["n_on_wave3"] == on["n_organs"])
    # (C) OFF arm: escape reverts — no organ on the wave3 pool, base-4 still co-resident (on single_pool)
    c_ok = bool((not off["wave3_enabled"]) and off["n_on_wave3"] == 0 and off["n_base4_distinct"] == 1)
    go = bool(a_ok and b_ok and c_ok)
    return {"seed": int(seed), "per_key": per_key, "organ_answer_preserved": organ_same,
            "A_answer_preserved": a_ok, "B_one_brain_on": b_ok, "C_escape_off": c_ok,
            "on_pool": {"n_distinct_pools": on["n_distinct_pools"], "n_on_wave3": on["n_on_wave3"],
                        "n_standalone": on["n_standalone"], "wave3_enabled": on["wave3_enabled"]},
            "off_pool": {"n_on_wave3": off["n_on_wave3"], "n_base4_distinct": off["n_base4_distinct"],
                         "n_standalone": off["n_standalone"], "wave3_enabled": off["wave3_enabled"]},
            "GO": go}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--worker", action="store_true", help="internal: build the 8 organs + print the answer battery")
    args = ap.parse_args()

    if args.worker:
        ans = _worker_answers(args.seed)
        print("__ANSWERS__ " + json.dumps(ans), flush=True)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=== ONE-BRAIN 11-ORGAN POOL FLIP — brain-chat answer-preservation + one-brain coherence regression ===")
    print("    BRAIN_ONEBRAIN_WAVE3_POOL=1 (8 wired organs on ONE 11-organ merge_organs pool) vs =0 (escape)")
    per_seed = [verify_seed(s) for s in seeds]
    for p in per_seed:
        print(f"  [seed {p['seed']}] A_answer={p['A_answer_preserved']} B_one_brain={p['B_one_brain_on']} "
              f"C_escape={p['C_escape_off']} on={p['on_pool']} off={p['off_pool']} -> GO={p['GO']}", flush=True)
        for k in _ANSWER_KEYS:
            if not p["per_key"][k]["same"]:
                print(f"      DIVERGE {k}: on={p['per_key'][k]['on']} off={p['per_key'][k]['off']}", flush=True)

    n = len(seeds)
    n_go = sum(p["GO"] for p in per_seed)
    organs = ("surprise", "worldmodel", "metacog", "pragmatic", "comprehension", "self_schema", "curiosity")
    per_organ = {k: sum(p["organ_answer_preserved"][k] for p in per_seed) for k in organs}
    n_a = sum(p["A_answer_preserved"] for p in per_seed)
    n_b = sum(p["B_one_brain_on"] for p in per_seed)
    n_c = sum(p["C_escape_off"] for p in per_seed)
    all_go = bool(n_go == n and n > 0)
    print("\n=== VERDICT (11-organ pool flip: answer-preservation + one-brain coherence) ===")
    for k in organs:
        print(f"  {k:14s} answer_preserved {per_organ[k]}/{n}")
    print(f"  A answer-preserved (all 7 read organs) : {n_a}/{n}")
    print(f"  B one-brain ON (8 organs, 1 pool, wave3): {n_b}/{n}")
    print(f"  C escape OFF (no organ on wave3 pool)   : {n_c}/{n}")
    print(f"  ALL-SEED GO: {n_go}/{n}  ->  ALL-GO={all_go}")

    payload = {"mode": "onebrain_11organ_pool_flip_regression", "seeds": seeds,
               "per_seed": per_seed, "per_organ": per_organ,
               "n_A_answer": n_a, "n_B_one_brain": n_b, "n_C_escape": n_c,
               "n_go": n_go, "n_seeds": n, "all_go": all_go,
               "excluded_organs": ["source_provenance", "prospective_memory", "d6_multiref_wm"],
               "backend": os.environ.get("SIM_BACKEND", "(default)")}

    try:
        from tools.verdict import Verdict
        v = Verdict("one-brain 11-ORGAN POOL flip — brain-chat answer-preservation + one-brain coherence")
        v.require("every wired organ's live chat answer preserved under the pool flip, every seed", n_a, expect=n)
        v.require("ON arm: all 8 wired organs on ONE pool object (the 11-organ wave3 pool), every seed", n_b, expect=n)
        v.require("OFF arm: escape reverts (no organ on the wave3 pool), every seed", n_c, expect=n)
        v.disabled("source_provenance / prospective_memory / d6_multiref_wm pool wiring",
                   why="documented exclusions (incremental wrapper API / scope-reduced / per-session in webapp)")
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
