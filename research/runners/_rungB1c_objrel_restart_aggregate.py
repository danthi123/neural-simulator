"""Aggregate the parallel per-seed objrel basin-escape restart runs into the K-sweep verdict.

Reads research/findings/raw/_restart_seed<N>.json (one per seed, written by the fanned-out
_rungB1c_objrel_restart_basin_escape_derisk runs) and computes the pre-registered K-sweep:
per-K recovery over the GENUINELY-EMERGENT counted seeds (excluding init-lucky pre>=0.85),
blind recovery, mean objrel, mean test-oracle, and the CRITIC~ORACLE selection rate (the
honest guardrail: does the reward critic pick the objrel-recovering restart, vs a good restart
merely EXISTING among the K).

Pre-registered split: DEV {42,43,44,45,46}, BLIND {100,101,102,103,104}.
Usage: SIM_BACKEND=numpy python -m research.runners._rungB1c_objrel_restart_aggregate
"""
import json, glob, os

DEV = [42, 43, 44, 45, 46]
BLIND = [100, 101, 102, 103, 104]
KS = [1, 3, 5, 8]
INIT_LUCKY_THRESH = 0.85   # pre-learning objrel >= this => init-lucky, EXCLUDED from the counted GO tally
RECOV_THRESH = 0.85        # objrel-slot0 >= this => recovered

def load_rows():
    rows = {}
    for f in sorted(glob.glob("research/findings/raw/_restart_seed*.json")):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  [skip {os.path.basename(f)}: {e}]"); continue
        for r in d.get("rows", []):
            rows[r["seed"]] = r
    return rows

def g(r, k, field):
    pk = r.get("per_k", {})
    e = pk.get(str(k), pk.get(k, {}))
    return e.get(field)

def main():
    rows = load_rows()
    print(f"loaded {len(rows)} seeds: {sorted(rows)}")
    missing = [s for s in DEV + BLIND if s not in rows]
    if missing:
        print(f"  MISSING seeds (not done yet / crashed): {missing}")

    # genuinely-emergent = pre-learning objrel < INIT_LUCKY_THRESH
    def pre(r):
        pl = r.get("pre_learning_k0", {})
        return pl.get("objrel_slot0_THEME", pl.get("objrel_slot0", 0.0))
    counted = [s for s in DEV + BLIND if s in rows and pre(rows[s]) < INIT_LUCKY_THRESH]
    init_lucky = [s for s in DEV + BLIND if s in rows and pre(rows[s]) >= INIT_LUCKY_THRESH]
    counted_blind = [s for s in counted if s in BLIND]
    print(f"genuinely-emergent counted={counted}  init-lucky-EXCLUDED={init_lucky}  counted-blind={counted_blind}")

    print(f"\n{'K':>3} | {'recov(counted)':>16} | {'recov(blind)':>13} | {'mean-objr':>9} | {'mean-oracle':>11} | {'critic~oracle':>13} | anti-cheats")
    curve = []
    for k in KS:
        rec = [s for s in counted if (g(rows[s], k, "objrel_slot0_THEME") or 0.0) >= RECOV_THRESH]
        recb = [s for s in counted_blind if (g(rows[s], k, "objrel_slot0_THEME") or 0.0) >= RECOV_THRESH]
        objs = [g(rows[s], k, "objrel_slot0_THEME") or 0.0 for s in counted]
        oras = [g(rows[s], k, "test_oracle_objrel_slot0") or 0.0 for s in counted]
        # critic~oracle: per counted seed, does the critic-selected objrel == the oracle objrel (within eps)?
        cmatch = [s for s in counted if abs((g(rows[s], k, "objrel_slot0_THEME") or 0.0) - (g(rows[s], k, "test_oracle_objrel_slot0") or 0.0)) < 1e-6]
        ac = all(
            (g(rows[s], k, "dale_legal") is True) and
            ((g(rows[s], k, "no_reward_objrel_slot0") or 0.0) < RECOV_THRESH) and
            ((g(rows[s], k, "shuffled_reward_objrel_slot0") or 0.0) < RECOV_THRESH) and
            ((g(rows[s], k, "scramble_objrel_slot0") or 0.0) < RECOV_THRESH) and
            ((g(rows[s], k, "no_spike_objrel_slot0") or 0.0) < RECOV_THRESH)
            for s in counted
        )
        mo = sum(objs)/len(objs) if objs else 0.0
        mor = sum(oras)/len(oras) if oras else 0.0
        allblind = len(recb) == len(counted_blind) and len(counted_blind) > 0
        print(f"{k:>3} | {len(rec)}/{len(counted)} {'(ALL-blind)' if allblind else '':>4} | {len(recb)}/{len(counted_blind):<11} | {mo:>9.3f} | {mor:>11.3f} | {len(cmatch)}/{len(counted):<11} | {'clean' if ac else 'FAIL'}")
        curve.append((k, len(rec), len(counted), len(recb), len(counted_blind), mo, mor, len(cmatch), ac))

    # verdict logic
    print("\n=== READ ===")
    best = max(curve, key=lambda c: (c[1], c[7]))
    kb, rc, nc, rb, nb, mo, mor, cm, ac = best
    monotone = all(curve[i][1] <= curve[i+1][1] for i in range(len(curve)-1))
    print(f"best K={kb}: recovered {rc}/{nc} counted ({rb}/{nb} blind), critic~oracle {cm}/{nc}, monotone-in-K={monotone}, anti-cheats={'clean' if ac else 'FAIL'}")
    # the CRUX: oracle vs critic gap at best K
    oracle_recov = [s for s in counted if (g(rows[s], kb, "test_oracle_objrel_slot0") or 0.0) >= RECOV_THRESH]
    print(f"  at K={kb}: ORACLE recovers {len(oracle_recov)}/{nc} (a good restart EXISTS); CRITIC recovers {rc}/{nc}. Gap = {len(oracle_recov)-rc} seeds where a good basin exists but the reward critic can't select it.")
    if rc == nc and nb > 0 and rb == nb and ac:
        print("  => GO candidate: K-restart+critic recovers ALL counted incl. all-blind, anti-cheats clean. (adversarial-verify before claiming.)")
    elif len(oracle_recov) > rc:
        print("  => HONEST RESIDUAL: restarts make a good basin EXIST (oracle) but the reward critic can't reliably SELECT it on some seeds -> the next lever is a better critic (e.g. salience-weighted training reward on held-out; multi-restart consensus), NOT more restarts.")
    else:
        print("  => characterize honestly per the numbers.")

if __name__ == "__main__":
    main()
