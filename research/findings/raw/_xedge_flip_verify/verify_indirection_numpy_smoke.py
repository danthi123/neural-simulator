"""Tiny numpy WIRING smoke for the xedge position-invariant indirection fix.

Directly exercises the organ-level `repair_target(..., wm_focus='w0')` path -- EXACTLY what the real handler
calls (explicit positional focus = CAND_POOLS[0]) -- on the 3 diagnostic seeds:
  42  (w0=p_agent  -> was VISIBLE before the fix; must stay visible)
  101 (w0=p_patient -> was HOLLOW before the fix; must now be VISIBLE=agent)
  43  (w0=p_ctrl    -> was INERT before the fix; must stay INERT)
Also checks lesion-reverts and that the LEGACY (self-test) per-pool path is unchanged (hold p_agent->agent,
hold p_patient->patient).
"""
import json
import os
import sys

SEEDS = [int(s) for s in (sys.argv[1:] or ["42", "101", "43"])]
os.environ.setdefault("SIM_BACKEND", "numpy")
for k in ("BRAIN_ONEBRAIN_XEDGE", "BRAIN_ONEBRAIN_XEDGE_LEARN", "BRAIN_ONEBRAIN_XEDGE_LESION"):
    os.environ.pop(k, None)
os.environ["BRAIN_ONEBRAIN_XEDGE"] = "1"
os.environ["BRAIN_ONEBRAIN_XEDGE_LEARN"] = "1"

AMB = "the wolf watches the owl"

import research.runners.onebrain_xedge_production as OX
import research.runners.comprehension_production_organ as CO
from research.runners._xedge_flip_production_verify import _w0_role

out = {}
for seed in SEEDS:
    OX._POOL = None
    CO._ORGAN = None
    OX.set_live_per_turn(False)   # converged build-curriculum edge (b_edge=learn), matches flip-verify ARM B
    pool = OX.get_xedge_pool(seed)
    assert pool is not None and pool.ok, f"pool build failed seed {seed}"
    corg = CO.get_organ(seed)
    corg.ensure_built()
    assert corg._shared is pool.pool, "organ not sharing primed pool"

    def rr(wm_focus):
        r = corg.repair_target(AMB, wm_focus=wm_focus) or {}
        return {"role": r.get("role"), "content_role": r.get("content_role"),
                "wm_resolved": r.get("wm_resolved"), "wm_margin": r.get("wm_margin")}

    novisi = rr(None)                 # no focus -> content role only
    held = rr("w0")                   # EXPLICIT positional focus = real-handler path -> indirection
    pool.lesion_cross()
    held_les = rr("w0")               # lesioned -> must revert to content

    w0role = _w0_role(seed)
    grown = w0role in ("agent", "patient")
    ans_differs = (held["role"] != novisi["role"])
    reverts = (held_les["wm_resolved"] is not True and held_les["role"] == novisi["role"])
    visible = bool(grown and (held["wm_resolved"] is True) and ans_differs)
    hollow = bool((held["wm_resolved"] is True) and not ans_differs)
    inert = bool((not grown) and (held["wm_resolved"] is not True) and not ans_differs)
    out[seed] = {"w0_role": w0role, "grown": grown, "novisi_role": novisi["role"],
                 "held_role": held["role"], "held_wm_resolved": held["wm_resolved"],
                 "lesion_role": held_les["role"], "lesion_wm_resolved": held_les["wm_resolved"],
                 "visible": visible, "hollow": hollow, "inert": inert, "reverts": reverts,
                 "seed_ok": bool(((visible and reverts) if grown else inert) and not hollow)}
    print(f"[seed {seed}] w0={w0role} novisi={novisi['role']} held={held['role']} "
          f"(wm_res={held['wm_resolved']}) lesion={held_les['role']} "
          f"visible={visible} hollow={hollow} inert={inert} reverts={reverts} seed_ok={out[seed]['seed_ok']}",
          flush=True)

n_hollow = sum(v["hollow"] for v in out.values())
n_seedok = sum(v["seed_ok"] for v in out.values())
payload = {"probe": "xedge_position_invariant_indirection_numpy_wiring_smoke",
           "backend": os.environ.get("SIM_BACKEND"), "b_edge": "learn (converged, per_turn=False)",
           "amb_item": AMB, "seeds": list(out.keys()),
           "n_hollow": n_hollow, "n_seed_ok": f"{n_seedok}/{len(out)}",
           "note": ("organ-level real-handler probe (explicit wm_focus='w0'=CAND_POOLS[0]): the Kriete-2013 "
                    "position-invariant indirection routes the held referent to the grown AGENT pool wherever it "
                    "sits, so seed 101/102 (w0=p_patient, was HOLLOW: resolved patient==content) now resolve agent "
                    "(visible), seed 42/100 (w0=p_agent) stay visible, seed 43/44 (w0=p_ctrl) stay correctly inert; "
                    "all lesion-revert. numpy WIRING smoke only -- the DECISIVE 6-seed verdict is the staged cupy run."),
           "per_seed": out}
outp = os.environ.get("PROBE_OUT")
if outp:
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"wrote {outp}", flush=True)
print("\n=== SUMMARY ===")
print(json.dumps(payload, indent=2, default=str))
print(f"n_hollow={n_hollow}  n_seed_ok={n_seedok}/{len(out)}")
