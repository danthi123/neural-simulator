"""NO-REGRESSION verify for the DG sparse-index fast path wired into OneBrainComposer (#150 knowledge-scale).

TWO gates, both must pass:

  (A) API-LEVEL PARITY + MOAT on the REAL composer. Build a OneBrainComposer, store a battery of facts, and run the
      full recall API (query_patient / query_agent / ask_yes_no) with the sparse index OFF (the byte-identical full
      bridge cleanup) and ON (the DG-routed shard cleanup). GATE: (1) every stored block decodes to the SAME
      {role: word} row on both paths; (2) every in-store query returns the IDENTICAL answer; (3) every out-of-store /
      cross cue ABSTAINS under BOTH (query_patient/query_agent -> None, ask_yes_no -> 'unknown') with ZERO new
      confabulation (ON never answers where OFF abstained). This is the no-confab moat preserved through sharding.

  (B) SCALE SPEEDUP on a large synthetic codebook in the COMPOSER's phase convention. Build the same DGSparseIndex the
      wiring uses over V in {5k, 50k, 200k} fractional-cycle FHRR concepts, and measure the full-codebook matched
      filter vs the DG-routed shard: top-1 parity, rows touched, wall time, and out-of-store abstain. GATE: parity
      >= 0.98, shard ~constant (rows-speedup widens with V, >= 10x at Vmax), out-of-store abstains under both, 0 new
      confab. (This reproduces the de-risk's 6-seed GO with the exact index the composer imports.)

Run:  .venv/bin/python -m research.runners._wire_sparse_index_verify --json research/findings/raw/_wire_sparse_index_verify.json
NO sim/ edit. Host-rate de-risk of the wiring; the spiking-DG burn-down is named in research/biology/dg-ca3-sparse-index.md.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from research.runners._sparse_indexed_retrieval_derisk import DGSparseIndex


# ------------------------------- (A) API-level parity + moat on the real composer -------------------------------
def build_battery(seed):
    """A deterministic vocab + a battery of unique-(agent, action) facts + in-store and out-of-store queries."""
    agents = ["dog", "cat", "boy", "girl", "man", "woman", "bird", "fish", "horse", "cow", "wolf", "fox"]
    actions = ["chased", "saw", "ate", "found", "wanted", "held", "pushed", "carried"]
    patients = ["ball", "bone", "apple", "fish", "leaf", "rock", "cup", "book", "flower", "worm", "star", "key"]
    rng = np.random.default_rng(seed)
    vocab = sorted(set(agents + actions + patients + ["north", "south", "unknownX", "unknownY", "phantom"]))
    facts = []
    used = set()
    for _ in range(16):
        while True:
            a = agents[rng.integers(len(agents))]; x = actions[rng.integers(len(actions))]
            if (a, x) not in used:
                used.add((a, x)); break
        p = patients[rng.integers(len(patients))]
        pol = "AFFIRM" if rng.random() < 0.75 else "NEGATE"
        facts.append((a, x, p, pol))
    return vocab, facts


def run_composer_battery(seed, enable_sparse_index):
    """Store the battery, then decode every block + run the full recall API; return the decoded rows + answers."""
    vocab, facts = build_battery(seed)
    comp = OneBrainComposer(seed=seed, D=128, vocab=vocab, k_max=32,
                            enable_spiking_cleanup=False,          # the numpy-CPU test-oracle argmax path
                            enable_sparse_index=enable_sparse_index)
    for (a, x, p, pol) in facts:
        comp.store(a, x, p, polarity=pol)
    rows = comp._read_blocks()                                     # per-block {role: word} decode
    ans_patient = [comp.query_patient(a, x) for (a, x, _p, _pol) in facts]
    ans_agent = [comp.query_agent(x, p) for (_a, x, p, _pol) in facts]
    ans_yesno = [comp.ask_yes_no(a, x, p) for (a, x, p, _pol) in facts]          # asserted == stored patient
    # cross / out-of-store cues (the moat): unknown agent, unknown action, wrong patient, phantom SVO
    moat = {
        "unknown_agent_patient": comp.query_patient("phantom", facts[0][1]),
        "unknown_action_patient": comp.query_patient(facts[0][0], "north"),
        "unknown_agent_agent": comp.query_agent("south", "phantom"),
        "phantom_yesno": comp.ask_yes_no("phantom", facts[0][1], facts[0][2]),
        "wrongpatient_yesno": comp.ask_yes_no(facts[0][0], facts[0][1], "phantom"),
    }
    return {"facts": facts, "rows": rows, "ans_patient": ans_patient, "ans_agent": ans_agent,
            "ans_yesno": ans_yesno, "moat": moat}


def check_api_parity(seed):
    off = run_composer_battery(seed, enable_sparse_index=False)
    on = run_composer_battery(seed, enable_sparse_index=True)
    # (1) per-block-role decode identity
    rows_match = (off["rows"] == on["rows"])
    n_role_cells = sum(len(r) for r in off["rows"])
    n_role_mismatch = sum(1 for ro, rn in zip(off["rows"], on["rows"])
                          for k in set(ro) | set(rn) if ro.get(k) != rn.get(k))
    # (2) answer identity across the recall API
    pat_match = off["ans_patient"] == on["ans_patient"]
    agn_match = off["ans_agent"] == on["ans_agent"]
    yn_match = off["ans_yesno"] == on["ans_yesno"]
    # (3) moat: every cue that abstains OFF must abstain ON (0 new confab); the abstain sentinels are None / 'unknown'
    def is_abstain(v):
        return v is None or v == "unknown"
    moat_off_abstains = {k: is_abstain(v) for k, v in off["moat"].items()}
    moat_new_confab = sum(1 for k in off["moat"]
                          if moat_off_abstains[k] and not is_abstain(on["moat"][k]))
    moat_match = off["moat"] == on["moat"]
    all_moat_abstain = all(moat_off_abstains.values())            # the battery's moat cues genuinely abstain OFF
    ok = (rows_match and pat_match and agn_match and yn_match and moat_match
          and moat_new_confab == 0 and all_moat_abstain)
    return {
        "seed": seed, "ok": bool(ok),
        "rows_match": bool(rows_match), "n_role_cells": int(n_role_cells), "n_role_mismatch": int(n_role_mismatch),
        "ans_patient_match": bool(pat_match), "ans_agent_match": bool(agn_match), "ans_yesno_match": bool(yn_match),
        "moat_match": bool(moat_match), "moat_new_confab": int(moat_new_confab),
        "moat_all_abstain_off": bool(all_moat_abstain),
        "n_in_store_queries": len(off["ans_patient"]) + len(off["ans_agent"]) + len(off["ans_yesno"]),
    }


# ------------------------------- (B) scale speedup on the composer's synthetic codebook -------------------------
def gen_composer_codebook(V, D, rng):
    """V FHRR concept codes as FRACTIONAL-CYCLE phases in [0,1) -- exactly OneBrainComposer/RFPhasorComposer's
    self.comp.concepts convention (the phasor is exp(2pi i phase))."""
    return rng.uniform(0.0, 1.0, size=(V, D)).astype(np.float64)


def host_matched_filter(codebook, rec):
    """score_w = sum_k cos(2pi(rec - code_w)) over the given codebook rows -- the composer's on-substrate cleanup
    (rf_phasor_composer.py:662), computed here over host to measure full vs shard cost."""
    return np.cos(2.0 * np.pi * (rec[None, :] - codebook)).sum(axis=1)


def eval_scale(V, D, seed, g, G, c, sigma, tau, n_query, oos_query):
    rng = np.random.default_rng(seed * 1000003 + V)
    cb = gen_composer_codebook(V, D, rng)                          # fractional-cycle
    m = max(2, int(np.ceil(V ** (1.0 / g))))
    idx = DGSparseIndex(D=D, m=m, g=g, G=G, c=c, seed=seed)
    idx.build(cb * (2.0 * np.pi))                                 # index in radians (the wiring's convention)

    q_ids = rng.integers(0, V, size=n_query)
    q = np.stack([cb[i] + rng.normal(0.0, sigma / (2 * np.pi), size=D) for i in q_ids])   # noisy cue (frac-cycle)

    full_ans, dg_ans, full_wall, dg_wall, full_rows, dg_rows = [], [], [], [], [], []
    for j in range(n_query):
        t = time.perf_counter()
        sc = host_matched_filter(cb, q[j]); fa = int(np.argmax(sc)); fpk = float(sc[fa])
        fa = fa if fpk >= tau * D else None
        full_wall.append(time.perf_counter() - t); full_rows.append(V); full_ans.append(fa)
        t = time.perf_counter()
        shard = idx.query(q[j] * (2.0 * np.pi))
        if shard.size:
            ssc = host_matched_filter(cb[shard], q[j]); k = int(np.argmax(ssc)); pk = float(ssc[k])
            da = int(shard[k]) if pk >= tau * D else None
        else:
            da = None
        dg_wall.append(time.perf_counter() - t); dg_rows.append(int(shard.size)); dg_ans.append(da)

    valid = [j for j in range(n_query) if full_ans[j] is not None]
    parity = float(np.mean([dg_ans[j] == full_ans[j] for j in valid])) if valid else 0.0

    # moat: out-of-store cues (codes NOT in cb) must abstain under BOTH
    oos = gen_composer_codebook(oos_query, D, np.random.default_rng(seed * 31337 + V))
    full_abstain = dg_abstain = dg_confab = 0
    for j in range(oos_query):
        sc = host_matched_filter(cb, oos[j]); fa = None if float(sc.max()) < tau * D else int(np.argmax(sc))
        shard = idx.query(oos[j] * (2.0 * np.pi))
        if shard.size:
            ssc = host_matched_filter(cb[shard], oos[j]); da = None if float(ssc.max()) < tau * D else int(shard[int(np.argmax(ssc))])
        else:
            da = None
        full_abstain += (fa is None); dg_abstain += (da is None); dg_confab += (fa is None and da is not None)

    return {
        "V": V, "m": m, "parity": parity,
        "rows_full": float(np.mean(full_rows)), "rows_dg": float(np.mean(dg_rows)),
        "rows_speedup": float(np.mean(full_rows) / max(1e-9, np.mean(dg_rows))),
        "wall_full_ms": float(np.mean(full_wall) * 1e3), "wall_dg_ms": float(np.mean(dg_wall) * 1e3),
        "wall_speedup": float(np.mean(full_wall) / max(1e-12, np.mean(dg_wall))),
        "oos_full_abstain": float(full_abstain / max(1, oos_query)),
        "oos_dg_abstain": float(dg_abstain / max(1, oos_query)), "dg_new_confab": int(dg_confab),
    }


def build_verdict(api, by_V, scales, api_ok, scale_ok, parity_ok, sublinear, speedup_ok, moat_ok, su_large):
    """Earn the verdict via tools.verdict.Verdict so the artifact carries the preconditions that earned it
    (verdict-preconditions gate). UNDEFINED unless every precondition is measured and holds."""
    from tools.verdict import Verdict
    min_scale_parity = float(min(np.mean([r["parity"] for r in by_V[V]]) for V in scales))
    v = Verdict("DG sparse-index wired into OneBrainComposer (#150 knowledge-scale) — no-regression")
    v.require("(A) API decode + answers IDENTICAL flag OFF vs ON (all seeds)",
              all(r["rows_match"] and r["ans_patient_match"] and r["ans_agent_match"] and r["ans_yesno_match"]
                  for r in api), expect=True)
    v.require("(A) API moat preserved: 0 new confab + all moat cues abstain OFF (all seeds)",
              (sum(r["moat_new_confab"] for r in api) == 0) and all(r["moat_all_abstain_off"] for r in api),
              expect=True)
    v.require("(B) scale top-1 parity vs full scan >= 0.98 at every V", min_scale_parity,
              expect=lambda x: x >= 0.98)
    v.floor("(B) scale rows-speedup @ Vmax vs 10x floor", float(su_large), floor=10.0)
    v.require("(B) scale shard SUBLINEAR (grows slower than V)", bool(sublinear), expect=True)
    v.require("(B) scale moat: out-of-store abstains under both + 0 new confab", bool(moat_ok), expect=True)
    v.disabled("spiking DG granule-cell WTA",
               why="the host-rate DG sparse projection is a DECLARED stand-in; the in-shard matched-filter cleanup "
                   "IS the composer's on-substrate op. Burn-down: _riii_ca3_completion_specificity_derisk.py / "
                   "cortex_dg_ca3_cleanup_probe.py / _gap5_emergent_dg_selection_derisk.py")
    v.disabled("enable_spiking_cleanup",
               why="the verify runs the numpy-CPU test-oracle argmax path (enable_spiking_cleanup=False); the "
                   "spiking-cleanup WTA winner == argmax (validated), so the index composes with it unchanged")
    return v.decide(go=bool(api_ok and scale_ok))


def main():
    ap = argparse.ArgumentParser(description="no-regression verify for the wired DG sparse index (#150)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--scales", default="5000,50000,200000")
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--g", type=int, default=3); ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.30); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-query", type=int, default=60); ap.add_argument("--oos-query", type=int, default=60)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    scales = [int(x) for x in a.scales.split(",")]

    print("=== (A) API-level parity + moat on the REAL OneBrainComposer (flag OFF vs ON) ===", flush=True)
    api = [check_api_parity(s) for s in seeds]
    for r in api:
        print(f"  seed {r['seed']}: ok={r['ok']} | decode rows_match={r['rows_match']} "
              f"(role-cell mismatches {r['n_role_mismatch']}/{r['n_role_cells']}) | "
              f"ans patient={r['ans_patient_match']} agent={r['ans_agent_match']} yesno={r['ans_yesno_match']} | "
              f"moat match={r['moat_match']} new-confab={r['moat_new_confab']} "
              f"(all-abstain-off={r['moat_all_abstain_off']})", flush=True)
    api_ok = all(r["ok"] for r in api)

    print("\n=== (B) scale speedup on the composer's synthetic codebook ===", flush=True)
    all_rows = []
    for s in seeds:
        for V in scales:
            r = eval_scale(V, a.D, s, a.g, a.G, a.c, a.sigma, a.tau, a.n_query, a.oos_query)
            all_rows.append({"seed": s, **r})
            print(f"  seed {s} V={V:>7d} m={r['m']:>3d} | parity={r['parity']:.3f} | rows FULL={r['rows_full']:.0f} "
                  f"DG={r['rows_dg']:.1f} ({r['rows_speedup']:.0f}x) | wall FULL={r['wall_full_ms']:.3f}ms "
                  f"DG={r['wall_dg_ms']:.3f}ms ({r['wall_speedup']:.1f}x) | moat oos-abstain FULL="
                  f"{r['oos_full_abstain']:.3f} DG={r['oos_dg_abstain']:.3f} new-confab={r['dg_new_confab']}", flush=True)

    # (B) verdict
    Vmax = max(scales); Vmin = min(scales)
    by_V = {V: [r for r in all_rows if r["V"] == V] for V in scales}
    parity_ok = all(np.mean([r["parity"] for r in by_V[V]]) >= 0.98 for V in scales)
    su_small = np.mean([r["rows_speedup"] for r in by_V[Vmin]]); su_large = np.mean([r["rows_speedup"] for r in by_V[Vmax]])
    shard_small = np.mean([r["rows_dg"] for r in by_V[Vmin]]); shard_large = np.mean([r["rows_dg"] for r in by_V[Vmax]])
    sublinear = (su_large > su_small) and (shard_large / max(1e-9, shard_small) < Vmax / Vmin)
    speedup_ok = su_large >= 10.0
    moat_ok = all(np.mean([r["oos_dg_abstain"] for r in by_V[V]]) >= 0.999
                  and sum(r["dg_new_confab"] for r in by_V[V]) == 0 for V in scales)
    scale_ok = parity_ok and sublinear and speedup_ok and moat_ok

    verdict = build_verdict(api, by_V, scales, api_ok, scale_ok, parity_ok, sublinear, speedup_ok,
                            moat_ok, su_large)
    print("\n  ===== VERDICT =====", flush=True)
    print(f"  (A) API parity+moat: {'PASS' if api_ok else 'FAIL'} "
          f"(identical decode+answers + moat preserved, 0 new confab, all seeds)", flush=True)
    print(f"  (B) scale: parity>=0.98={parity_ok} sublinear={sublinear} rows-speedup@Vmax="
          f"{su_large:.0f}x(>=10x={speedup_ok}) moat(oos-abstain,0 new-confab)={moat_ok}", flush=True)
    print(f"\n  {verdict['status']} -- the wired DG sparse index preserves the composer's recall answers + the "
          f"no-confab moat BYTE-FOR-BYTE at the API level while making the cleanup O(shard) not O(V) at scale.",
          flush=True)

    if a.json:
        out = {"seeds": seeds, "scales": scales, "api": api, "scale": all_rows,
               "api_ok": api_ok, "scale_ok": scale_ok, **verdict}
        json.dump(out, open(a.json, "w"), indent=1)
    return 0 if verdict["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
