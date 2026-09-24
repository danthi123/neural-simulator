"""Dev-seed calibration of the BRAIN_MULTIREF_FOCUS_BIND retrieval (the focus WTA) on the PRIVATE D6 organ (no full
brain build; ~seconds per seed on numpy). REFUSES the evaluation seeds {42, 43, 44, 100, 101, 102}.

Per seed and per (gain, cue, run) setting it measures, with the organ's own code path (`load` -> a whole-bridge reset
standing in for another session/organ between turns -> `resolve_anaphor` -> `read_held`):
  * two_live   : a 2-referent load in BOTH mention orders -- which register wins, the WTA margin, and whether both
                 referents are still held after the retrieval (the retrieval reads the buffer, it must not clear it);
  * empty      : a never-loaded buffer -- must resolve nothing (the cue alone cannot win);
  * lesion     : the recur=0 organ-scope lesion buffer -- the hold dies, must resolve nothing.
Pre-registration: research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md.

  SIM_BACKEND=numpy OMP_NUM_THREADS=1 tools/memcap.sh 1 -- .venv/bin/python -m research.runners._wm_focus_bind_calib \\
      --seeds 7,11,13,17,19,23,29,31 --settings 12000:150:60,12000:150:150,20000:150:60,20000:150:150,8000:150:150 \\
      --out research/findings/raw/_wm_focus_bind/calib_dev_seeds.json
"""
from __future__ import annotations

import argparse
import json
import os

EVAL_SEEDS = {42, 43, 44, 100, 101, 102}


def _case(D6, seed, order, lesion=False):
    from research.runners._d3_persistent_slot_derisk import _reset
    org = D6.MultiReferentWMOrgan(seed=seed, shared=None)
    ld = org.load(order, lesion=lesion)
    _reset((org._lesion_buf() if lesion else org.buf).sb)
    r = org.resolve_anaphor("it", lesion=lesion)
    h = org.read_held(lesion=lesion)
    return {"order": order, "load_hold_alive_min": ld["hold_alive_min"], "resolved": r["resolved"],
            "register": r["resolved_register"], "pool": r["resolved_pool"], "margin": r["margin"],
            "register_rates": r["register_rates"], "wta_rates": r["wta_rates"],
            "held_after": h["recovered"] if h else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,11,13,17,19,23,29,31")
    ap.add_argument("--settings", default="20000:150:150", help="comma list of gain:cue:run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if set(seeds) & EVAL_SEEDS:
        raise SystemExit("refusing evaluation seeds %s -- calibration is dev-seed only" % sorted(set(seeds) & EVAL_SEEDS))
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ["BRAIN_MULTIREF_FOCUS_BIND"] = "1"
    import research.runners.d6_multiref_wm_production_organ as D6
    shipped = {"gain": D6.FOCUS_WTA_GAIN_PA, "cue": D6.FOCUS_WTA_CUE_PA, "run": D6.FOCUS_WTA_RUN}
    out = {"runner": "research.runners._wm_focus_bind_calib", "kind": "dev-seed calibration", "seeds": seeds,
           "shipped_operating_point": shipped, "settings": []}
    for spec in a.settings.split(","):
        gain, cue, run = (float(x) for x in spec.split(":"))
        D6.FOCUS_WTA_GAIN_PA, D6.FOCUS_WTA_CUE_PA, D6.FOCUS_WTA_RUN = gain, cue, int(run)
        rows = []
        for seed in seeds:
            os.environ.pop("BRAIN_MULTIREF_LESION", None)
            row = {"seed": seed, "two_live": [_case(D6, seed, ["dog", "cat"]), _case(D6, seed, ["cat", "dog"])]}
            emp = D6.MultiReferentWMOrgan(seed=seed, shared=None)
            emp.ensure_built()
            emp.buf.reset()
            emp._stash(emp.buf)
            e = emp.resolve_anaphor("it")
            row["empty"] = {"resolved": e["resolved"], "wta_rates": e["wta_rates"]}
            os.environ["BRAIN_MULTIREF_LESION"] = "1"
            row["lesion"] = _case(D6, seed, ["dog", "cat"], lesion=True)
            os.environ.pop("BRAIN_MULTIREF_LESION", None)
            rows.append(row)
        s = {"gain": gain, "cue": cue, "run": int(run), "rows": rows,
             "n_two_live_resolved": sum(all(c["resolved"] is not None for c in r["two_live"]) for r in rows),
             "n_same_register_both_orders": sum(r["two_live"][0]["register"] == r["two_live"][1]["register"]
                                                for r in rows),
             "n_both_held_after": sum(all(len(c["held_after"] or {}) == 2 for c in r["two_live"]) for r in rows),
             "n_empty_resolved": sum(r["empty"]["resolved"] is not None for r in rows),
             "n_lesion_resolved": sum(r["lesion"]["resolved"] is not None for r in rows),
             "min_two_live_margin": min(c["margin"] for r in rows for c in r["two_live"])}
        out["settings"].append(s)
        print(json.dumps({k: v for k, v in s.items() if k != "rows"}), flush=True)
    D6.FOCUS_WTA_GAIN_PA, D6.FOCUS_WTA_CUE_PA, D6.FOCUS_WTA_RUN = shipped["gain"], shipped["cue"], shipped["run"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=str)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
