"""Aggregate the gap#1 RF-phase-encode gate: MAIN vs anti-cheats (memoryless, scramble) vs the M1 reference.
Prints the comparison table + a clear GO/collapse verdict. Reads the per-arm JSONs written by the on-bridge runner."""
import json, sys, glob

RAW = "research/findings/raw"
ARMS = {
    "M1 host-inject (ref)": "_gap1_M1_reconfirm.json",
    "RF phase encode (MAIN)": "_gap1_rf_encode_MAIN_s42.json",
    "memoryless anti-cheat": "_gap1_rf_encode_MEMORYLESS_s42.json",
    "scramble anti-cheat": "_gap1_rf_encode_SCRAMBLE_s42.json",
}

def load(f):
    try:
        return json.load(open(f"{RAW}/{f}"))
    except Exception as e:
        return {"_err": str(e)}

rows = []
for name, f in ARMS.items():
    d = load(f)
    if "_err" in d:
        rows.append((name, None, None, None, "MISSING"))
        continue
    bd = d.get("by_depth", {})
    deep = bd.get("10-99", {})
    mid = bd.get("6-9", {})
    rows.append((name, d.get("map_corr"), deep.get("vs_trigram"), mid.get("vs_trigram"), "GO" if d.get("go") else "no-go"))

print(f"{'arm':<26} {'map_corr':>9} {'deep(10-99)':>12} {'mid(6-9)':>10} {'verdict':>8}")
print("-" * 70)
for name, mc, deep, mid, v in rows:
    mcs = f"{mc:.3f}" if isinstance(mc, (int, float)) else "-"
    ds = f"{deep:+.3f}" if isinstance(deep, (int, float)) else "-"
    ms = f"{mid:+.3f}" if isinstance(mid, (int, float)) else "-"
    print(f"{name:<26} {mcs:>9} {ds:>12} {ms:>10} {v:>8}")

# the GO logic for THIS surpass: MAIN GO + BOTH anti-cheats collapse (deep <= 0 or map_corr low)
r = {name: (mc, deep) for name, mc, deep, mid, v in rows}
main = r.get("RF phase encode (MAIN)", (None, None))
mem = r.get("memoryless anti-cheat", (None, None))
scr = r.get("scramble anti-cheat", (None, None))
print("\n--- SURPASS LOGIC ---")
if all(isinstance(x[1], (int, float)) for x in (main, mem, scr)):
    main_go = main[1] > 0.02
    mem_collapse = mem[1] < 0.1          # memoryless: no deep-context advantage
    scr_collapse = (scr[0] is not None and scr[0] < 0.3) or scr[1] < 0.1  # scramble: state destroyed
    verdict = main_go and mem_collapse and scr_collapse
    print(f"  MAIN deep {main[1]:+.3f} > 0.02 : {main_go}")
    print(f"  memoryless deep {mem[1]:+.3f} < 0.10 (collapse) : {mem_collapse}")
    print(f"  scramble map_corr {scr[0]:.3f}/deep {scr[1]:+.3f} (collapse) : {scr_collapse}")
    print(f"\n  => RF-PHASE-ENCODE SURPASS: {'*** GO ***' if verdict else 'NOT CLEAN — inspect'}")
else:
    print("  (not all arms complete yet)")
