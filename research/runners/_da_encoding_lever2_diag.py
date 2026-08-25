"""LEVER-2 DIAGNOSTIC + HOMEOSTATIC-CANDIDATE SWEEP for the DA-gated-encoding flip gate (2026-08-22).

STEP 1 (DIAGNOSE-FIRST / verify the instrument at the leak). The lean soak returned UNDEFINED on moat_fail_total=2
(seed 44: sigma 1.0 moat_on=false; sigma 2.0 moat_off=false). This reproduces BOTH points with per-block decode
instrumentation and DETERMINES, for each, whether the leak is (a) a REAL encoding-manufactures-a-fact leak (the DA write
gain flips an unstored cue from abstain->answer at a FIXED damage draw) or (b) a read-floor/instrument artifact (the
baseline read machinery spuriously completes the unstored cue EVEN with encoding OFF, g=1). It finds WHICH stored block
the damaged read mis-decodes to match the moat cue, and that block's gain.

STEP 2 (candidate homeostatic gain maps). Sweeps OFF vs a family of ON gain vectors over all 6 seeds x the full sigma
grid, reporting recall_off/recall_on per (seed,sigma), the stress-net violations, and the moat DECOMPOSED into
encoding-INTRODUCED leaks (moat_off True & moat_on False -- the genuine residual) vs baseline read-floor artifacts
(moat_off False -- an instrument property of the control arm, NOT the coupling). Candidates:
  RAW       : the current clamped map  g = clip(0.5, 3.0, 1 + k(DA-base))                 (reproduces the UNDEFINED)
  H2FLOOR   : recall-safe floor raised to 1.0  g = clip(1.0, 3.0, 1 + k(DA-base))          (low/tonic == OFF)
  HOMEO     : Turrigiano multiplicative scaling  g = clip(1.0, 3.0, s * r),  r = 1+k(DA-base) UNCLAMPED,
              s = A*/mean(r) with set-point A*=1.0  (COMMON population factor; preserves the high-side relative order)
  HOMEO_GM  : same but s = A*/geomean(r_pos)                                                (log-centred variant)
  MINPROT   : min-protect multiplicative  s = 1.0/min(raw_clamped)  g = clip(1.0,3.0,s*raw_clamped)

Reuses the soak's EXACT machinery (build the OFF composer once/seed, derive each ON arm by block-scaling store_conns,
identical read-damage draws) so every arm differs from OFF in NOTHING but per-fact write magnitude.

Run (cupy, through gpu_queue -- one brain at a time):
  cd <worktree> && SIM_BACKEND=cupy python -u -m research.runners._da_encoding_lever2_diag --out <path>
"""
from __future__ import annotations
import argparse, json, os, sys
os.environ.setdefault("SIM_BACKEND", "cupy")
import logging
logging.getLogger().setLevel(logging.ERROR)
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._da_encoding_leansoak import (
    SEEDS, D as D_DEFAULT, K_DA, DA_BASELINE, SIGMAS, VOCAB, FACTS, DA_CLASS, _DA, UNSTORED,
    _battery, _build_off, _scale_blocks, _recall_set, _moat_holds, _block_mean_mag,
)
from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import (
    da_to_encoding_gain, _damage_store_conns,
)


# ---------------------------------------------------------------------------
# candidate gain maps (all are functions of the per-fact DA schedule `das`)
# ---------------------------------------------------------------------------
def gains_raw(das):
    return [da_to_encoding_gain(da, DA_BASELINE, K_DA, 0.5, 3.0) for da in das]

def gains_h2floor(das):
    return [da_to_encoding_gain(da, DA_BASELINE, K_DA, 1.0, 3.0) for da in das]

def _r_unclamped(das):
    return [1.0 + K_DA * (da - DA_BASELINE) for da in das]

def gains_homeo(das, a_star=1.0, g_floor=1.0, g_max=3.0):
    r = _r_unclamped(das)
    mu = float(np.mean(r))
    s = a_star / mu if mu > 0 else 1.0
    return [float(min(g_max, max(g_floor, s * ri))) for ri in r]

def gains_homeo_gm(das, a_star=1.0, g_floor=1.0, g_max=3.0):
    r = _r_unclamped(das)
    pos = [max(ri, 1e-6) for ri in r]
    gm = float(np.exp(np.mean(np.log(pos))))
    s = a_star / gm if gm > 0 else 1.0
    return [float(min(g_max, max(g_floor, s * ri))) for ri in r]

def gains_minprot(das, g_floor=1.0, g_max=3.0):
    raw = gains_raw(das)
    s = 1.0 / max(min(raw), 1e-6)
    return [float(min(g_max, max(g_floor, s * ri))) for ri in raw]

CANDIDATES = {
    "RAW": gains_raw,
    "H2FLOOR": gains_h2floor,
    "HOMEO": gains_homeo,
    "HOMEO_GM": gains_homeo_gm,
    "MINPROT": gains_minprot,
}


# ---------------------------------------------------------------------------
# STEP 1 -- per-block decode instrumentation at a fixed damage draw
# ---------------------------------------------------------------------------
def _set_arm(comp, store_conns):
    comp.store_conns = store_conns
    comp._store_dirty = True
    comp._store_csr = None
    if getattr(comp, "_csr_cache", None) is not None:
        comp._csr_cache = {}


def _instrument_moat(comp, arm_conns, unstored, sigma, dmg_seed, gains, facts):
    """Set the arm, apply the SAME moat damage draw the soak uses, and read out: the returned patient for the unstored
    cue, WHICH block (if any) the damaged read selects for that cue, and the decoded (agent,action,patient) + gain of
    every block. Restores clean store_conns after."""
    _set_arm(comp, arm_conns)
    clean = comp.store_conns
    rng = np.random.default_rng(dmg_seed)
    damaged = _damage_store_conns(clean, sigma, rng)
    try:
        comp.store_conns = damaged
        comp._store_dirty = True; comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
        a, act = unstored
        idx = comp._seq_block(a, act)                    # the block the (damaged) cue-match selects, or None (abstain)
        patient = comp.query_patient(a, act)             # the moat result (None == abstain == moat holds)
        rows = comp._read_blocks()                       # every block's decoded roles under the SAME damage
        decoded = []
        for i, got in enumerate(rows):
            decoded.append({"block": i, "stored_fact": list(facts[i]), "gain": round(float(gains[i]), 4),
                            "decoded_agent": got.get("agent"), "decoded_action": got.get("action"),
                            "decoded_patient": got.get("patient")})
    finally:
        comp.store_conns = clean
        comp._store_dirty = True; comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
    return {"moat_holds": patient is None, "returned_patient": patient,
            "selected_block": (int(idx) if idx is not None else None),
            "decoded_blocks": decoded}


def diagnose_leaks(dim):
    """Reproduce the two soak leak points (seed 44, sigma 1.0 & 2.0) on BOTH arms with per-block decode readout."""
    facts, das, unstored = _battery()
    g_raw = gains_raw(das)
    g_off = [1.0] * len(facts)
    seed = 44
    out = {"seed": seed, "unstored_cue": list(unstored), "points": []}
    c = _build_off(seed, facts, dim, VOCAB, len(facts) + 4)
    off_conns = list(c.store_conns)
    on_conns = _scale_blocks(off_conns, g_raw, dim)
    dmg_seed = seed * 100003 + 99999                      # == the soak's moat damage draw
    for sigma in (1.0, 2.0):
        off = _instrument_moat(c, off_conns, unstored, sigma, dmg_seed, g_off, facts)
        on = _instrument_moat(c, on_conns, unstored, sigma, dmg_seed, g_raw, facts)
        # attribution: encoding-INTRODUCED (off holds, on leaks) vs baseline artifact (off leaks)
        if not off["moat_holds"]:
            verdict = "BASELINE READ-FLOOR ARTIFACT (OFF arm leaks with encoding DISABLED, g=1 -> NOT the coupling)"
        elif off["moat_holds"] and not on["moat_holds"]:
            verdict = "ENCODING-INTRODUCED (OFF abstains, ON leaks at the SAME damage draw -> gain-attributable)"
        else:
            verdict = "NO LEAK (both arms abstain)"
        out["points"].append({"sigma": sigma, "off": off, "on": on, "attribution": verdict})
    del c
    return out


# ---------------------------------------------------------------------------
# STEP 2 -- candidate sweep over all seeds x sigmas
# ---------------------------------------------------------------------------
def sweep_candidate(seed, gains, dim, facts, sigmas, off_conns_cache):
    """recall_off/recall_on + per-arm moat for one candidate gain vector on one seed (reuses a cached OFF composer)."""
    c, off_conns = off_conns_cache
    on_conns = _scale_blocks(off_conns, gains, dim)
    rows = []
    for arm_name, conns in (("off", off_conns), ("on", on_conns)):
        _set_arm(c, conns)
        for sigma in sigmas:
            rec = _recall_set(c, facts, sigma, seed)
            moat = _moat_holds(c, unstored=UNSTORED, sigma=sigma, seed=seed)
            rows.append((arm_name, sigma, int(sum(rec)), bool(moat)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--D", type=int, default=D_DEFAULT)
    ap.add_argument("--candidates", type=str, nargs="+", default=list(CANDIDATES.keys()))
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw", "_da_encoding_lever2",
                                         "diag_sweep.json"))
    args = ap.parse_args()
    seeds = args.seeds
    sigmas = sorted(set(args.sigmas))
    dim = args.D
    facts, das, unstored = _battery()
    k_max = len(facts) + 4

    # STEP 1
    print("[STEP 1] diagnosing the two moat-leak points (seed 44, sigma 1.0 & 2.0)...", flush=True)
    diag = diagnose_leaks(dim)
    for p in diag["points"]:
        print(f"  sigma={p['sigma']}: {p['attribution']}", flush=True)
        print(f"    OFF: moat_holds={p['off']['moat_holds']} returned={p['off']['returned_patient']!r} "
              f"selected_block={p['off']['selected_block']}", flush=True)
        print(f"     ON: moat_holds={p['on']['moat_holds']} returned={p['on']['returned_patient']!r} "
              f"selected_block={p['on']['selected_block']}", flush=True)
        for arm_key in ("off", "on"):
            sb = p[arm_key]["selected_block"]
            if sb is not None:
                blk = p[arm_key]["decoded_blocks"][sb]
                print(f"       [{arm_key}] selected block {sb}: stored={blk['stored_fact']} gain={blk['gain']} "
                      f"decoded=({blk['decoded_agent']},{blk['decoded_action']},{blk['decoded_patient']})", flush=True)

    # STEP 2
    print(f"\n[STEP 2] candidate homeostatic sweep: {args.candidates}", flush=True)
    cand_gains = {name: CANDIDATES[name](das) for name in args.candidates}
    print("  candidate gain vectors (per fact):", flush=True)
    for name in args.candidates:
        print(f"    {name:9s}: {[round(g,3) for g in cand_gains[name]]}", flush=True)

    per_candidate = {}
    for name in args.candidates:
        per_candidate[name] = {"gains": cand_gains[name], "per_seed": []}

    total_possible = len(facts) * len(seeds)
    for seed in seeds:
        c = _build_off(seed, facts, dim, VOCAB, k_max)
        off_conns = list(c.store_conns)
        # OFF recall + moat are candidate-independent; compute once, reuse.
        _set_arm(c, off_conns)
        off_recall = {}; off_moat = {}
        for sigma in sigmas:
            off_recall[sigma] = int(sum(_recall_set(c, facts, sigma, seed)))
            off_moat[sigma] = bool(_moat_holds(c, unstored=UNSTORED, sigma=sigma, seed=seed))
        for name in args.candidates:
            on_conns = _scale_blocks(off_conns, cand_gains[name], dim)
            _set_arm(c, on_conns)
            sweep = []
            for sigma in sigmas:
                on_r = int(sum(_recall_set(c, facts, sigma, seed)))
                on_m = bool(_moat_holds(c, unstored=UNSTORED, sigma=sigma, seed=seed))
                sweep.append({"sigma": sigma, "recall_off": off_recall[sigma], "recall_on": on_r,
                              "moat_off": off_moat[sigma], "moat_on": on_m})
            per_candidate[name]["per_seed"].append({"seed": seed, "sweep": sweep})
        del c

    # aggregate + verdict per candidate
    summary = {}
    for name in args.candidates:
        agg = {s: {"off": 0, "on": 0} for s in sigmas}
        stress_violations = 0          # (seed,sigma>0) with recall_on < recall_off
        clean_regress = 0              # sigma==0 regressions
        moat_introduced = 0            # moat_off True & moat_on False  (the genuine encoding residual)
        moat_baseline = 0              # moat_off False (control-arm read-floor artifact)
        for ps in per_candidate[name]["per_seed"]:
            for r in ps["sweep"]:
                agg[r["sigma"]]["off"] += r["recall_off"]
                agg[r["sigma"]]["on"] += r["recall_on"]
                if r["sigma"] <= 0.0:
                    if r["recall_on"] < r["recall_off"]:
                        clean_regress += 1
                else:
                    if r["recall_on"] < r["recall_off"]:
                        stress_violations += 1
                if not r["moat_off"]:
                    moat_baseline += 1
                elif r["moat_off"] and not r["moat_on"]:
                    moat_introduced += 1
        curve = {("%.4g" % s): {"off": agg[s]["off"], "on": agg[s]["on"]} for s in sigmas}
        go = (stress_violations == 0 and clean_regress == 0 and moat_introduced == 0)
        summary[name] = {"gains": [round(g, 4) for g in cand_gains[name]],
                         "stress_net_violations": stress_violations, "clean_regressions": clean_regress,
                         "moat_introduced_leaks": moat_introduced, "moat_baseline_artifacts": moat_baseline,
                         "GO_clean": bool(go), "curve": curve}

    out = {"runner": "research/runners/_da_encoding_lever2_diag.py",
           "config": {"seeds": seeds, "sigmas": sigmas, "D": dim, "k_da": K_DA, "da_baseline": DA_BASELINE,
                      "facts": FACTS, "da_class": DA_CLASS, "da_values": _DA, "unstored_moat_cue": list(UNSTORED),
                      "total_possible_per_sigma": total_possible},
           "step1_diagnosis": diag,
           "step2_candidates": per_candidate,
           "summary": summary}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar, flush=True)
    print("  CANDIDATE SUMMARY (GO_clean = 0 stress-net violations AND 0 clean regressions AND 0 encoding-introduced "
          "moat leaks)", flush=True)
    print(bar, flush=True)
    for name in args.candidates:
        s = summary[name]
        print(f"  {name:9s} gains={s['gains']}", flush=True)
        print(f"            stress_net_violations={s['stress_net_violations']:3d}  clean_regress={s['clean_regressions']}"
              f"  moat_introduced={s['moat_introduced_leaks']}  moat_baseline_artifact={s['moat_baseline_artifacts']}"
              f"   -> GO_clean={s['GO_clean']}", flush=True)
        row = "            per-sigma on/off: " + "  ".join(
            f"{sig}:{s['curve']['%.4g' % sig]['on']}/{s['curve']['%.4g' % sig]['off']}" for sig in sigmas)
        print(row, flush=True)
    print(f"\n  [saved] {args.out}\n" + bar, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
