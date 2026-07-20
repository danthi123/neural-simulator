"""Aggregate the gap#1 RF-phase-encode 6-seed generalization gate. Per seed: M1 reference vs RF-phase-encode MAIN
(deep-10-99 vs-trigram, map_corr). Reports dev (42/43/44) and blind (100/101/102) SEPARATELY. Anti-cheat spot-checks
on seed 100. GO = RF-MAIN GO (deep>0.02, map_corr>0.9) on all 6, tracking M1; anti-cheats collapse."""
import json, os
RAW = os.path.dirname(os.path.abspath(__file__))
def L(f):
    try: return json.load(open(os.path.join(RAW, f)))
    except Exception: return None
def deep(d):
    if not d: return None, None
    b = d.get("by_depth", {}).get("10-99", {})
    return d.get("map_corr"), b.get("vs_trigram")

DEV = [42, 43, 44]; BLIND = [100, 101, 102]
print(f"{'seed':>5} {'grp':>5} | {'M1 corr':>8} {'M1 deep':>8} | {'RF corr':>8} {'RF deep':>8} | {'RF vs M1':>9} {'GO':>4}")
print("-" * 74)
rf_go = []
for grp, seeds in [("dev", DEV), ("blind", BLIND)]:
    for s in seeds:
        m1c, m1d = deep(L(f"_gap1_ms_M1_seed{s}.json"))
        rfc, rfd = deep(L(f"_gap1_ms_RF_seed{s}.json"))
        go = (rfd is not None and rfd > 0.02 and rfc is not None and rfc > 0.9)
        rf_go.append(go)
        def fmt(x, p="+.3f"): return format(x, p) if isinstance(x, (int, float)) else "-"
        dvm = fmt(rfd - m1d) if isinstance(rfd, (int, float)) and isinstance(m1d, (int, float)) else "-"
        print(f"{s:>5} {grp:>5} | {fmt(m1c,'.3f'):>8} {fmt(m1d):>8} | {fmt(rfc,'.3f'):>8} {fmt(rfd):>8} | {dvm:>9} {'GO' if go else 'NO':>4}")

print("\n--- anti-cheat spot-checks (seed 100, RF phase encode) ---")
for tag, f in [("RF+memoryless", "_gap1_ms_RFmemless_seed100.json"), ("RF+scramble", "_gap1_ms_RFscramble_seed100.json")]:
    c, dd = deep(L(f))
    coll = (dd is not None and dd < 0.1)
    print(f"  {tag:<16} map_corr {c if c is None else round(c,3)!s:>6}  deep {dd if dd is None else round(dd,3):>7}  "
          f"{'COLLAPSE ✓' if coll else 'NOT collapsed — inspect'}")

n = len([g for g in rf_go if g is not None])
print(f"\n=> RF-MAIN GO on {sum(bool(g) for g in rf_go)}/{len(rf_go)} seeds "
      f"(dev {sum(bool(g) for g in rf_go[:3])}/3, blind {sum(bool(g) for g in rf_go[3:])}/3)")
print(f"=> MULTI-SEED SURPASS: {'*** GO (6/6) ***' if sum(bool(g) for g in rf_go)==6 else 'PARTIAL/inspect'}")
