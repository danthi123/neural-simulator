"""R2 (board #108 cluster, 2026-09-02): direct, targeted check that `webapp/server.py::_load_or_build_ltm_store`'s
NEW default (`enable_codebook_cache=True, enable_decode_escalation=True` via `_ltm_codebook_cache_on()`/
`_ltm_decode_escalation_on()`) is BYTE-IDENTICAL to the pre-change behavior (both flags OFF) on the shipped
`wikidata_core_15k` bundle -- the exact function `_build_chat_brain`'s tiny-demo branch calls, which the broader
`_knowledge_scale_100k_production_verify.py --bundle <15k core>` battery (a separate, slower, load_developed_brain-
level check) does not exercise directly.

Method: call `_load_or_build_ltm_store` TWICE on the same bundle -- once with flags explicitly False (the exact
pre-2026-09-02 behavior), once with the new defaults (None -> resolves True/True) -- and compare `query_patient`/
`ask_yes_no` answers on a large random sample of real (agent, action) cues plus a moat battery of unknown cues.
0 diffs required for the byte-identical claim.

Run (numpy-CPU, light -- the 15k core, not the 100k bundle):
  SIM_BACKEND=numpy .venv/bin/python \\
      research/findings/raw/_flip108_r2_wiring/verify_load_or_build_ltm_store_15k_byte_identical.py
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k"


def main():
    import numpy as np
    import webapp.server as S

    t0 = time.time()
    with open(os.path.join(BUNDLE, "facts.json"), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    facts = [r["fact"] if isinstance(r, dict) and "fact" in r else r for r in raw]
    fm = {}
    for f in facts:
        a, act, p = f.get("agent"), f.get("action"), f.get("patient")
        if not (isinstance(a, str) and isinstance(act, str) and isinstance(p, str)):
            continue
        if f.get("polarity", "AFFIRM") != "AFFIRM":
            continue
        fm.setdefault((a, act), p)
    keys = list(fm.keys())
    rng = np.random.default_rng(42)
    idx = rng.choice(len(keys), size=min(300, len(keys)), replace=False)
    probes = [(keys[i][0], keys[i][1], fm[keys[i]]) for i in idx]
    real_actions = sorted({f.get("action") for f in facts if isinstance(f.get("action"), str)})
    unknown_agents = [f"zzz_unknown_entity_{j}_xq" for j in range(20)]
    moat_cues = [(ua, real_actions[int(rng.integers(0, len(real_actions)))]) for ua in unknown_agents]

    print(f"[{time.time()-t0:.1f}s] loaded {len(facts)} facts, {len(probes)} probes, {len(moat_cues)} moat cues",
          flush=True)

    print(f"[{time.time()-t0:.1f}s] building OFF store (enable_codebook_cache=False, enable_decode_escalation=False) ...",
          flush=True)
    ltm_off = S._load_or_build_ltm_store(BUNDLE, seed=42, enable_codebook_cache=False, enable_decode_escalation=False)
    print(f"[{time.time()-t0:.1f}s] OFF store built ({type(ltm_off).__name__})", flush=True)

    print(f"[{time.time()-t0:.1f}s] building DEFAULT store (no explicit kwargs -> resolves via "
          f"_ltm_codebook_cache_on()={S._ltm_codebook_cache_on()} / "
          f"_ltm_decode_escalation_on()={S._ltm_decode_escalation_on()}) ...", flush=True)
    ltm_default = S._load_or_build_ltm_store(BUNDLE, seed=42)
    print(f"[{time.time()-t0:.1f}s] DEFAULT store built ({type(ltm_default).__name__})", flush=True)

    diffs = []
    for (a, v, gt) in probes:
        off_ans = ltm_off.query_patient(a, v)
        def_ans = ltm_default.query_patient(a, v)
        if off_ans != def_ans:
            diffs.append({"cue": [a, v], "off": repr(off_ans), "default": repr(def_ans), "gt": repr(gt)})
        off_yn = ltm_off.ask_yes_no(a, v, gt)
        def_yn = ltm_default.ask_yes_no(a, v, gt)
        if off_yn != def_yn:
            diffs.append({"cue": [a, v, gt], "kind": "yesno", "off": repr(off_yn), "default": repr(def_yn)})

    moat_diffs = []
    moat_off_confab = moat_def_confab = 0
    for (a, v) in moat_cues:
        off_ans = ltm_off.query_patient(a, v)
        def_ans = ltm_default.query_patient(a, v)
        if off_ans is not None:
            moat_off_confab += 1
        if def_ans is not None:
            moat_def_confab += 1
        if off_ans != def_ans:
            moat_diffs.append({"cue": [a, v], "off": repr(off_ans), "default": repr(def_ans)})

    n_probe_checks = 2 * len(probes)
    out = {
        "bundle": BUNDLE,
        "n_facts": len(facts),
        "n_probe_cues": len(probes),
        "n_probe_checks": n_probe_checks,
        "n_probe_diffs": len(diffs),
        "probe_diffs": diffs,
        "n_moat_cues": len(moat_cues),
        "n_moat_diffs": len(moat_diffs),
        "moat_off_confab": moat_off_confab,
        "moat_default_confab": moat_def_confab,
        "default_resolves_to": {
            "enable_codebook_cache": S._ltm_codebook_cache_on(),
            "enable_decode_escalation": S._ltm_decode_escalation_on(),
        },
        "byte_identical_off_vs_default": bool(len(diffs) == 0 and len(moat_diffs) == 0),
        "moat_safe": bool(moat_off_confab == 0 and moat_def_confab == 0),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = os.path.join(_HERE, "verify_load_or_build_ltm_store_15k_byte_identical.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 100)
    print(f"  byte_identical_off_vs_default: {out['byte_identical_off_vs_default']} "
          f"({out['n_probe_diffs']} probe diffs / {n_probe_checks} checks, {out['n_moat_diffs']} moat diffs)")
    print(f"  moat_safe: {out['moat_safe']} (off_confab={moat_off_confab}, default_confab={moat_def_confab})")
    print(f"  default resolves to: {out['default_resolves_to']}")
    print(f"  [saved] {os.path.relpath(out_path, _REPO)}  ({out['elapsed_s']}s)")
    print("=" * 100)
    return 0 if (out["byte_identical_off_vs_default"] and out["moat_safe"]) else 1


if __name__ == "__main__":
    sys.exit(main())
