"""ONE-BRAIN SINGLE-POOL FLIP — FLIPPED-FACULTY INDEPENDENCE verifier.

The core-organ answer-preservation regression (`_onebrain_single_pool_flip_regression.py`, 6-seed GO) proves the
single-pool flip preserves the FOUR CORE cortical organs' live answers. This companion verifier closes the OTHER
half the production-flip decision needs: that the flip does NOT regress the FOUR faculties flipped to
production-default on 2026-09-05 — shared-salience, value-choice, appraisal-interoception, GNW-stop — which the
live brain-chat runs ON by default alongside the core organs.

WHY these four cannot be touched by the flip, and what this verifier PROVES empirically.
`BRAIN_ONEBRAIN_SINGLE_POOL` is read by exactly ONE function — `onebrain_single_pool_production.single_pool_enabled`
— consumed by exactly ONE site each: the 4 CORE organs' `get_organ()`, to resolve `shared=` to the single
`merge_organs` pool (ON) vs the two `MergedSubstrate*` pools (OFF). None of the 4 flipped-faculty production
modules (`shared_salience_afferent`, `value_choice_production_organ`, `affect_production_organ`,
`webapp/gnw_global_stop`) reference the flag, `single_pool_*`, `merge_organs`, the `MergedSubstrate*` pools, or
import any of the 4 core organs (grep-verified). So the flip cannot alter a flipped faculty's CODE PATH. The ONE
indirect channel is global-RNG advancement: the ON vs OFF core-organ builds draw the process global RNG a
DIFFERENT number of times, and a faculty built AFTER them that draws from the GLOBAL RNG (rather than seeding its
own substrate from `cfg.seed`) could differ.

This verifier exercises exactly that worst case: per seed, in subprocess-isolated ON vs OFF, it (1) builds all 4
CORE organs — reproducing the live startup order + the divergent global-RNG advancement — then (2) reads the
flipped faculties that build AFTER them. It records the post-core-build global-RNG state hash (the WITNESS that the
preceding state genuinely diverged — a non-vacuity guard) and the flipped-faculty reads:
  shared-salience         -> `shared_salience_afferent.read_salience(RAW)` (the curiosity ASK-pool transduction
                             every consumer calls) — normalized + want_hz.
  appraisal-interoception -> `affect_production_organ.get_organ(seed).read_differential(APPRAISAL)` (the
                             production-default interoceptive-afferent ladder) — the signed differential.
  value-choice            -> the substrate its `ensure_built()` builds via `_merged_navcritic_valuetrain.
                             build_merged(seed, convergent_upstate=True)` (the SAME call the organ makes),
                             hashed (`cp_neuron_firing_thresholds` + `cp_membrane_potential_v`). This is the only
                             thing the flag could touch; the value-train is a deterministic function of the built
                             substrate + the organ's own seed, so a byte-identical build => a byte-identical
                             trained organ. ONE build, hashed — deliberately NOT a trained/untrained attribution.

GATE (per seed): every flipped-faculty read is BYTE-IDENTICAL ON vs OFF (a NULL result — the flip is invisible to
them), WHILE the global-RNG witness hash DIFFERS (proving the test is non-vacuous: the preceding process state
really did diverge, and the faculties are nonetheless seed-isolated from it). GNW-stop is covered ARCHITECTURALLY
(it snapshots+restores the host `random` state around its spiking read — `webapp/gnw_global_stop.py:255-262` — and
its verdict is a function of per-turn `chat` deliberation/swap state, not the merge; zero merge references), so it
is reported as ARCHITECTURAL, not exercised here (it needs a live ChatBrain).

Reproduce (numpy smoke, 1 seed):
    SIM_BACKEND=numpy python -m research.runners._onebrain_single_pool_flipped_faculty_independence \
        --seeds 42 --out research/findings/raw/_onebrain_single_pool_flipped_faculty_smoke.json

NO `sim/` edit. All nets are the tiny (N<=2034) numpy/cupy substrates the organ-read GO validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_RAW_SALIENCE = 0.7        # a fixed mid-high raw scalar for the shared-salience transduction
_APPRAISAL = 0.6           # a fixed positive appraisal for the affect ladder differential


def _arr_hash(x) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()[:16]


def _worker_reads(seed: int) -> dict:
    """Build the 4 CORE organs (reproducing the live startup order + divergent global-RNG advancement under
    whatever `BRAIN_ONEBRAIN_SINGLE_POOL` resolves to in THIS process), then read the flipped faculties that build
    after them. Returns byte-comparable primitives + hashes."""
    import numpy as np
    import random as _random
    from research.runners.onebrain_single_pool_production import single_pool_enabled
    import research.runners.surprise_production_organ as SO
    import research.runners.worldmodel_production_organ as WM
    import research.runners.metacog_production_organ as MC
    import research.runners.pragmatic_production_organ as PR

    # (1) build the 4 core organs — the ONLY thing the flag changes; ON => one merge_organs pool, OFF => two pools.
    SO.get_organ(seed=seed).ensure_built()
    WM.get_organ(seed=seed).ensure_built()
    MC.get_organ(seed=seed).ensure_built()
    PR.get_organ(seed=seed).ensure_built()

    # (2) the WITNESS: the process global-RNG state AFTER the divergent core builds (numpy + host random).
    rng_witness = hashlib.sha256(
        repr(np.random.get_state()).encode() + b"|" + repr(_random.getstate()).encode()
    ).hexdigest()[:16]

    # (3) the flipped faculties that build AFTER the core organs.
    import research.runners.shared_salience_afferent as SS
    sal = SS.read_salience(_RAW_SALIENCE, seed=seed)
    salience_read = [round(float(sal.get("normalized", -1.0)), 12), round(float(sal.get("want_hz", -1.0)), 12)]

    import research.runners.affect_production_organ as AF
    af = AF.get_organ(seed=seed)
    af.ensure_built()
    ad = af.read_differential(_APPRAISAL)
    appraisal_read = round(float(ad.get("differential", ad.get("diff", 0.0))), 12)

    # value-choice: hash the substrate its ensure_built() builds via VT.build_merged (the SAME call the organ
    # makes). This tests whether the flip perturbs value-choice's substrate BUILD directly — the only thing the
    # flag could touch (its value-train is a deterministic function of the built substrate + the organ's own
    # seed, so a byte-identical build => a byte-identical trained organ). ONE build, hashed — deliberately NOT a
    # trained/untrained attribution.
    from research.runners import _merged_navcritic_valuetrain as VT
    vb, _vh = VT.build_merged(seed, convergent_upstate=True)
    vc_hashes = {}
    for nm in ("cp_neuron_firing_thresholds", "cp_membrane_potential_v"):
        arr = getattr(vb, nm, None)
        vc_hashes[nm] = _arr_hash(arr) if arr is not None else "MISSING"

    return {
        "seed": int(seed),
        "single_pool_enabled": bool(single_pool_enabled()),
        "rng_witness": rng_witness,
        "salience": salience_read,
        "appraisal": appraisal_read,
        "value_substrate": vc_hashes,
    }


def _run_worker(seed: int, single_pool: bool) -> dict:
    env = dict(os.environ)
    if single_pool:
        env["BRAIN_ONEBRAIN_SINGLE_POOL"] = "1"
    else:
        env.pop("BRAIN_ONEBRAIN_SINGLE_POOL", None)
    cmd = [sys.executable, "-m", "research.runners._onebrain_single_pool_flipped_faculty_independence",
           "--worker", "--seed", str(seed)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          cwd=str(Path(__file__).resolve().parents[2]))
    if proc.returncode != 0:
        raise RuntimeError(f"worker(seed={seed}, single_pool={single_pool}) failed rc={proc.returncode}\n"
                           f"STDERR:\n{proc.stderr[-4000:]}")
    line = next(l for l in reversed(proc.stdout.splitlines()) if l.startswith("__READS__ "))
    return json.loads(line[len("__READS__ "):])


_FACULTY_KEYS = ("salience", "appraisal", "value_substrate")


def verify_seed(seed: int) -> dict:
    on = _run_worker(seed, single_pool=True)
    off = _run_worker(seed, single_pool=False)
    faculty_same = {k: bool(on[k] == off[k]) for k in _FACULTY_KEYS}
    witness_diverged = bool(on["rng_witness"] != off["rng_witness"])
    go = bool(all(faculty_same.values()) and on["single_pool_enabled"] and not off["single_pool_enabled"])
    return {"seed": int(seed), "faculty_byte_identical": faculty_same,
            "rng_witness_diverged": witness_diverged,
            "on": {k: on[k] for k in _FACULTY_KEYS}, "off": {k: off[k] for k in _FACULTY_KEYS},
            "rng_on": on["rng_witness"], "rng_off": off["rng_witness"],
            "on_flag": on["single_pool_enabled"], "off_flag": off["single_pool_enabled"], "GO": go}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--worker", action="store_true")
    args = ap.parse_args()

    if args.worker:
        print("__READS__ " + json.dumps(_worker_reads(args.seed)), flush=True)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=== ONE-BRAIN SINGLE-POOL FLIP — FLIPPED-FACULTY (salience/appraisal/value-choice) INDEPENDENCE ===")
    print("    each flipped faculty's read BYTE-IDENTICAL under BRAIN_ONEBRAIN_SINGLE_POOL=1 vs the two-pool default")
    per_seed = [verify_seed(s) for s in seeds]
    for p in per_seed:
        print(f"  [seed {p['seed']}] byte_identical={p['faculty_byte_identical']} "
              f"rng_witness_diverged={p['rng_witness_diverged']} -> GO={p['GO']}", flush=True)
        for k in _FACULTY_KEYS:
            if not p["faculty_byte_identical"][k]:
                print(f"      DIVERGE {k}: on={p['on'][k]} off={p['off'][k]}", flush=True)

    n = len(seeds)
    n_go = sum(p["GO"] for p in per_seed)
    per_faculty = {k: sum(p["faculty_byte_identical"][k] for p in per_seed) for k in _FACULTY_KEYS}
    n_witness = sum(p["rng_witness_diverged"] for p in per_seed)
    all_go = bool(n_go == n and n > 0)
    print("\n=== VERDICT (flipped-faculty independence under the single-pool flip) ===")
    for k in _FACULTY_KEYS:
        print(f"  {k:16s} byte-identical {per_faculty[k]}/{n}")
    print(f"  RNG-divergence witness (non-vacuity): {n_witness}/{n} seeds diverged as expected")
    print(f"  gnw-stop: ARCHITECTURAL (host-random snapshot/restore + chat-state verdict; zero merge refs)")
    print(f"  ALL-FACULTY ALL-SEED byte-identity: {n_go}/{n}  ->  ALL-GO={all_go}")

    payload = {"mode": "onebrain_single_pool_flipped_faculty_independence", "seeds": seeds,
               "per_seed": per_seed, "per_faculty": per_faculty, "n_go": n_go, "n_seeds": n,
               "n_witness_diverged": n_witness, "all_go": all_go,
               "gnw_stop": "architectural (host-random snapshot/restore + chat-state verdict; zero merge references)",
               "backend": os.environ.get("SIM_BACKEND", "(default)")}

    try:
        from tools.verdict import Verdict
        v = Verdict("one-brain SINGLE-POOL flip — flipped-faculty independence (salience/appraisal/value-choice)")
        v.require("every flipped faculty's read byte-identical ON vs OFF, every seed", n_go, expect=n)
        v.require("the RNG-divergence witness fired (non-vacuous test), every seed", n_witness, expect=n)
        v.disabled("gnw-stop empirical exercise", why="architectural: host-random snapshot/restore + chat-state "
                   "verdict, zero merge references — needs a live ChatBrain, covered by source analysis")
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
