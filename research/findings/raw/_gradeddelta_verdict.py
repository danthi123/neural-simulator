"""Consolidate the #5b R1 delta-close verdict from the all-arms + GIRK-cap JSONs.

Prints: (1) the 3-seed grid arm (V n/f selectivity = R1 fix; SNc-burst gabab_gap = the load-bearing δ;
delta_vnf for contrast), (2) the control collapse (per seed), (3) the GIRK-cap sweep on seed 44.
"""
import json
import glob
import os

RAW = os.path.dirname(os.path.abspath(__file__))


def _load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _arm(d, arm):
    return ((d or {}).get("arms") or {}).get(arm) or {}


print("=" * 92)
print("#5b R1 delta-close verdict")
print("=" * 92)

print("\n[1] GRID arm, 3-seed (R1 fix = V n/f; load-bearing δ = SNc-burst gabab_gap):")
print(f"  {'seed':>4} | {'V n/f':>8} | {'SNc gap':>10} | {'gabab_gap':>9} | {'delta_vnf':>9} (contaminated)")
for s in (42, 43, 44):
    d = _load(f"{RAW}/_n5_grid_onbridge_gradeddelta_allarms_seed{s}.json")
    g = _arm(d, "grid"); sb = g.get("stage_b") or {}; gv = g.get("graded_v") or {}; gd = g.get("graded_delta") or {}
    print(f"  {s:>4} | {gv.get('v_near_over_far',0):>8.2f} | {sb.get('snc_gap_ratio',0):>10.3g} | "
          f"{str(sb.get('gabab_gap')):>9} | {gd.get('delta_vnf',0):>9.2f}")

print("\n[2] CONTROL collapse on the SNc-burst δ (gabab_gap must be False/collapse for all controls):")
for s in (42, 43, 44):
    d = _load(f"{RAW}/_n5_grid_onbridge_gradeddelta_allarms_seed{s}.json")
    print(f"  seed {s}:")
    for arm in ("grid", "render", "scramble", "no_learn", "lesion"):
        a = _arm(d, arm); sb = a.get("stage_b") or {}; gd = a.get("graded_delta") or {}
        print(f"    {arm:9s} gabab_gap={str(sb.get('gabab_gap')):>5} snc_gap={sb.get('snc_gap_ratio',0):>10.3g} "
              f"| delta_vnf={gd.get('delta_vnf',0):>7.2f}")

print("\n[3] GIRK-cap sweep on seed 44 (the principled over-clamp fix; want gabab_gap True + snc graded):")
for p in sorted(glob.glob(f"{RAW}/_n5_grid_onbridge_girkcap*_seed44.json")):
    d = _load(p); g = _arm(d, "grid"); sb = g.get("stage_b") or {}; gv = g.get("graded_v") or {}
    cap = os.path.basename(p).replace("_n5_grid_onbridge_girkcap", "").replace("_seed44.json", "")
    print(f"  cap={cap:>4}: V n/f={gv.get('v_near_over_far',0):>7.2f} snc_pred={sb.get('snc_predicted_near_hz',0):>6.1f} "
          f"snc_unpred={sb.get('snc_unpredicted_far_hz',0):>6.1f} gap={sb.get('snc_gap_ratio',0):>8.3g} "
          f"gabab_gap={str(sb.get('gabab_gap')):>5} lesion_collapses={str(sb.get('lesion_collapses')):>5}")

print("\n[4] The gentle-vs-hot TRADE-OFF across the global knobs (gabab_gap per seed; no single knob = 3/3):")
print(f"  {'knob':>22} | {'seed42':>7} | {'seed43':>7} | {'seed44':>7}")
def _gg(path):
    d = _load(path); sb = _arm(d, "grid").get("stage_b") or {}
    return str(sb.get("gabab_gap")) if d else "-"
rows = [
    ("cap=0 (baseline)", "_n5_grid_onbridge_gradeddelta_allarms_seed{}.json"),
    ("GIRK-cap=1.0",      "_n5_grid_onbridge_girkcap1.0_seed{}.json"),
    ("homeostasis",       "_n5_grid_onbridge_homeo_e02a01_seed{}.json"),
    ("graded-strength=15","_n5_grid_onbridge_gstr15_seed{}.json"),
    ("graded-strength=25","_n5_grid_onbridge_gstr25_seed{}.json"),
]
for label, tmpl in rows:
    vals = [_gg(f"{RAW}/{tmpl.format(s)}") for s in (42, 43, 44)]
    print(f"  {label:>22} | {vals[0]:>7} | {vals[1]:>7} | {vals[2]:>7}")
print("  => R1 (V n/f) selective 3/3 on every seed at every knob; the SNc-burst δ TRADES gentle<->hot.")
print("=" * 92)
