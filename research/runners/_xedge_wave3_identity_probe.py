"""S02 (2026-09-24 midnight plan) — LIVE xedge-vs-wave3 identity probe. Read-only: builds, never edits.

QUESTION. `comprehension_production_organ.get_organ()` resolves the ONE-BRAIN WAVE-3 pool FIRST (production default
`BRAIN_ONEBRAIN_WAVE3_POOL`), and only falls through to the ONE-BRAIN CROSS-EDGE pool (`BRAIN_ONEBRAIN_XEDGE`, also
default-ON) when wave3 returns None. Meanwhile `webapp.server._get_multiref_organ` gives every PER-SESSION d6 organ
`shared=get_xedge_pool(seed).pool`. If both reads hold at shipped defaults, the d6 organ that holds the referent
lives on the xedge pool while the comprehension organ that reads the role lives on the wave3 pool: the frozen/learned
w{k}->sel cross-edge exists, but on a pool the live comprehension read never touches (SEVERED).

This probe reads the object identities LIVE (it does not trust the code read at comprehension_production_organ.py
:836-843), at shipped defaults, `cfg.seed` via seed=42 on every builder, SIM_BACKEND=numpy:
  (a) `get_organ(42) is get_xedge_pool(42).comp_organ`
  (b) `webapp.server._get_multiref_organ(<test session>)._shared is get_merged_cortical_pool(42, min_wave=1)`
  (c) `xedge_enabled()`, `wave3_pool_enabled()` and `_WAVE3_POOL_DEFAULT_ON`.
Verdict: (a) and (b) both False -> CONFIRMED-SEVERED; either True -> RECONCILED.

Supplementary identity facts (recorded, not part of the verdict): which pool the d6 organ's `_shared` IS, whether
the comprehension organ's pool carries the xedge read handles (`xedge_amb_read` / `xedge_codrive_params`), and
whether the wave3 pool has regions named like the xedge candidate pools (w0/w1/w2/w3) -- that last one decides
whether the comprehension read's `_xedge_codrive(wm_focus="w0")` is a no-op or drives a d6 slice on the wave3 pool.

Run:  SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -u -m research.runners._xedge_wave3_identity_probe \
          --out research/findings/raw/_xedge_wave3_probe/s42.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return os.environ.get("POOL_REVISION", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="research/findings/raw/_xedge_wave3_probe/s42.json")
    args = ap.parse_args()
    seed = int(args.seed)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()

    # every BRAIN_* in the environment, so "shipped defaults" is checkable, not asserted.
    brain_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("BRAIN_")}

    import research.runners.onebrain_wave3_pool_production as W3
    import research.runners.onebrain_xedge_production as XE
    import research.runners.comprehension_production_organ as CO

    out = {"probe": "xedge_wave3_identity_probe", "plan_step": "S02 (2026-09-24 midnight plan)", "seed": seed,
           "backend": os.environ.get("SIM_BACKEND"), "git_sha": _git_sha(), "host": platform.node(),
           "pool_job_id": os.environ.get("POOL_JOB_ID"), "pool_job_mem_gb": os.environ.get("POOL_JOB_MEM_GB"),
           "brain_env": brain_env}

    # (c) the flags, read first (no build)
    out["c_xedge_enabled"] = bool(XE.xedge_enabled())
    out["c_xedge_learn_enabled"] = bool(XE.xedge_learn_enabled())
    out["c_wave3_pool_enabled"] = bool(W3.wave3_pool_enabled())
    out["c_wave3_default_on"] = bool(W3._WAVE3_POOL_DEFAULT_ON)

    # the production comprehension organ (this BUILDS the wave3 pool when wave3 is on)
    t = time.time()
    comp_organ = CO.get_organ(seed)
    out["t_comprehension_get_organ_s"] = round(time.time() - t, 2)
    t = time.time()
    merged = W3.get_merged_cortical_pool(seed, min_wave=1)
    out["t_get_merged_cortical_pool_s"] = round(time.time() - t, 2)
    t = time.time()
    xp = XE.get_xedge_pool(seed)
    out["t_get_xedge_pool_s"] = round(time.time() - t, 2)
    out["xedge_pool_ok"] = bool(xp is not None)

    # (a)
    out["a_comprehension_organ_is_xedge_comp_organ"] = bool(xp is not None and comp_organ is xp.comp_organ)
    out["a_detail"] = {
        "comp_organ_shared_is_merged_cortical_pool": bool(merged is not None and comp_organ._shared is merged),
        "comp_organ_shared_is_xedge_pool": bool(xp is not None and comp_organ._shared is xp.pool),
        "comp_organ_shared_type": type(comp_organ._shared).__name__ if comp_organ._shared is not None else None,
        "comp_pool_has_xedge_amb_read": bool(getattr(comp_organ._shared, "xedge_amb_read", None) is not None),
        "comp_pool_has_xedge_codrive_params": bool(
            getattr(comp_organ._shared, "xedge_codrive_params", None) is not None),
    }

    # (b) the PER-SESSION d6 organ exactly as the live handler builds it
    import webapp.server as S
    t = time.time()
    d6 = S._get_multiref_organ("__s02_identity_probe_session__")
    out["t_get_multiref_organ_s"] = round(time.time() - t, 2)
    out["b_multiref_shared_is_merged_cortical_pool"] = bool(merged is not None and d6._shared is merged)
    out["b_detail"] = {
        "multiref_shared_is_xedge_pool": bool(xp is not None and d6._shared is xp.pool),
        "multiref_shared_type": type(d6._shared).__name__ if d6._shared is not None else None,
        "multiref_pool_has_xedge_codrive_params": bool(
            getattr(d6._shared, "xedge_codrive_params", None) is not None),
        "brain_chat_seed": int(S._brain_chat_seed()),
    }

    # supplementary: does the wave3 pool carry regions named like the xedge candidate pools?
    try:
        from research.runners._onebrain_integration_r2_threefactor_selforganized import CAND_POOLS, BASE_POOL
        names = list(CAND_POOLS) + [BASE_POOL, "sel_agent", "sel_patient"]
    except Exception:
        names = ["w0", "w1", "w2", "w3", "sel_agent", "sel_patient"]
    have = {}
    if merged is not None:
        merged.ensure_built()
        rm = merged.bridge.region_manager
        for nm in names:
            try:
                have[nm] = int(len(rm.indices(nm)))
            except Exception:
                have[nm] = None
        out["merged_pool_n_neurons"] = int(merged.bridge.core_config.num_neurons)
        out["merged_pool_organ_regions"] = {k: list(v) for k, v in (merged.organ_regions or {}).items()}
    out["merged_pool_region_sizes_for_xedge_names"] = have
    if xp is not None:
        out["xedge_pool_n_neurons"] = int(xp.bridge.core_config.num_neurons)
        out["xedge_pool_cross_weights"] = {k: float(v) for k, v in (xp.cross_weights or {}).items()}

    a, b = out["a_comprehension_organ_is_xedge_comp_organ"], out["b_multiref_shared_is_merged_cortical_pool"]
    if (not a) and (not b):
        verdict = "CONFIRMED-SEVERED"
    else:
        verdict = "RECONCILED"
    out["verdict"] = verdict
    out["verdict_rule"] = "a and b both False -> CONFIRMED-SEVERED; either True -> RECONCILED (plan S02)"
    out["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3)
    out["wall_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("verdict", "a_comprehension_organ_is_xedge_comp_organ",
                                          "b_multiref_shared_is_merged_cortical_pool", "c_xedge_enabled",
                                          "c_wave3_pool_enabled", "c_wave3_default_on", "peak_rss_gb", "wall_s")}),
          flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
