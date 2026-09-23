"""D3 ONE-BRAIN — verify the AFFECT organ's migration onto the shared cortical pool (12 organs, ONE bridge) AND the
first cross-region synapse it brings (the ladder's own held arousal -> the D2 surprise pool).

Module under test: `research/runners/onebrain_affect_pool.py` (read its docstring for the readiness inventory and
the declared residuals). Everything is DEFAULT-OFF (`BRAIN_ONEBRAIN_AFFECT_POOL`, `BRAIN_ONEBRAIN_AFFECT_XEDGE`).

PRE-REGISTERED GATE (written 2026-09-23 BEFORE any 12-organ / cross-edge result existed; thresholds are constants
below and are NOT to be edited after a result is seen — a changed threshold is a new, separately-labelled gate).

ARM M — MIGRATION (answer-preservation; the 12-organ pool WITHOUT the cross-edge):
  M1 co-residence byte-identity: the affect read battery on the 12-organ pool == the SAME battery with affect alone
     on the 12-organ SUPERSET config (max |delta| == 0.0 exactly) — the merge adds no non-synaptic coupling.
  M2 carried-organ answer preservation (STRICT): each of the 11 Wave-3 organs' read battery on the 12-organ pool
     is BYTE-IDENTICAL to the SHIPPED `get_wave3_pool(seed)` (the production-default pool), and its rendered answer
     is identical — adding affect perturbs no production organ.
  M3 affect answer preservation vs TODAY'S production path: the graded tone LEVEL (the categorical value
     `content_plan`/`manner_for` consume) at every production appraisal (-1,-0.5,0,+0.5,+1) equals the standalone
     production ladder's (`AppraisalInteroceptiveLadder`, shared=None). Continuous differential reported, not
     gated (different per-neuron het/OU realisation by construction — the same honest residual Waves 1-3 declared
     for their new organs).
  M4 faculty alive on the pool: sign correct at |a|>=0.5, |neutral| < LADDER_NEUTRAL_TOL, affect_out lesion -> 0.0,
     interoceptive-synapse lesion collapses |diff| at +-1 to <= 0.25x intact.
  M5 weights frozen: every ladder synapse weight (both endpoints in affect regions) byte-identical before vs after
     the full 12-organ train+read lifecycle (the pool's global Hebbian cannot move plastic=False edges).
  M6 legacy discriminator: with the name-keyed seams OFF, affect's slice init DIVERGES merged-vs-alone (M1 is not
     vacuous).
  M7 determinism: the affect battery read twice on the same pool is identical.

ARM X — INTEGRATION (the arousal->surprise synapse on the one pool; mechanism level):
  Protocol, in ONE continuous sequence on the pool: appraisal ramp into the ladder via the interoceptive relays ->
  drive-off hold (the ladder LATCHES arousal) -> relays silent -> the surprise organ's own prediction+assertion
  drive on every trained block (CONTRADICT: cue i, assert (i+1)%n; CONFIRM: cue i, assert i). Mean surprise Hz.
  X1 SHIFT: C(+1) - C(0) >= SHIFT_FLOOR_HZ AND C(-1) - C(0) >= SHIFT_FLOOR_HZ (arousal is valence-independent).
  X2 LESION: gate `affect_arousal_to_surprise`=0 -> C(+1) - C(0) <= LESION_RATIO * X1's +1 shift.
  X3 NULL (no-edge pool, the same 12 organs without the synapse): C(+1) - C(0) == 0.0 exactly.
  X4 INTERO-NULL (relay->ladder synapse cut: the relays still fire, the ladder never latches): shift <= LESION_RATIO
     * intact — the effect is carried by the ladder's held state, not by the host drive on the relays.
  X5 BYTE-OFF: C(0) on the edge pool == C(0) on the no-edge pool exactly, AND every one of the 12 organs' standard
     read batteries is byte-identical edge-pool vs no-edge pool (the synapse is inert when arousal is silent).
  X6 HELD AROUSAL: arousal-rung rate in the surprise window > 0 at a=+-1 and == 0 at a=0; no external current on
     any ladder rung or relay during the surprise window (asserted every step).
  X7 DETERMINISM: C(+1) read twice -> identical.
  Reported (not gated): CONFIRM shift and the contradict/confirm shift ratio (gain-like vs DC), the fraction of
  contradict trials over the organ's own calibrated threshold, and a FAINT-assertion (325 pA) verdict flip count.

GO = all of M1..M7 and X1..X7 on every seed. The cross-edge weight `XEDGE_W` is FIXED before the 6-seed run by
`--calibrate` on seed 42 only (declared calibration seed): the smallest weight in CAL_WEIGHTS whose X1 shift clears
the floor with the mean CONFIRM rate still below the organ's threshold. Seeds: 42 43 44 100 101 102.

COMPUTE: numpy CPU, one ~7.7k-neuron pool (a few GB). Pool nodes: one seed per queue line.
  SIM_BACKEND=numpy python -u -m research.runners._onebrain_affect_pool_verify --seeds 42 \
      --json research/findings/raw/_onebrain_affect_pool/verify_seed42.json
  SIM_BACKEND=numpy python -u -m research.runners._onebrain_affect_pool_verify --calibrate --seeds 42 \
      --json research/findings/raw/_onebrain_affect_pool/calibrate_seed42.json
Aggregate (the literal GO-gate command):
  python -m research.runners._onebrain_affect_pool_verify --aggregate research/findings/raw/_onebrain_affect_pool/verify_seed*.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import lever, void_if, undefined_if_empty, attributable_to  # noqa: E402

SEEDS = (42, 43, 44, 100, 101, 102)
SHIFT_FLOOR_HZ = 0.10        # X1: minimum contradict-surprise shift high-vs-neutral arousal, both valences
LESION_RATIO = 0.34          # X2/X4: lesioned/null shift must be below this fraction of the intact shift
INTERO_COLLAPSE = 0.25       # M4: interoceptive-synapse lesion |diff| <= this x intact at +-1
CAL_WEIGHTS = (0.02, 0.05, 0.1, 0.2, 0.4)   # the pre-registered calibration sweep (effective per-synapse weight)
CAL_BUILD_W = 0.4            # calibration builds at the max weight; lower weights = transmission gate fraction
ASSERT_PA = 600.0
ASSERT_PA_WEAK = 325.0
PRE_STEPS = 60
HOLD = 60


# ─────────────────────────────────────────────────────────────────────────────────────────────
def _maxdelta(a, b):
    from research.runners._onebrain_wave1_organread_verify import _maxdelta as md
    return md(a, b)


def _ladder_weights(pool):
    """Every synapse weight with BOTH endpoints in an affect region (the M5 frozen-weights witness)."""
    from research.runners.onebrain_affect_pool import AFFECT_KEY
    b = pool.bridge
    idx = np.concatenate([np.asarray(v) for v in pool.idx(AFFECT_KEY).values()])
    coo = b.cp_connections.tocoo()
    row = np.asarray(coo.row); col = np.asarray(coo.col)
    m = np.isin(row, idx) & np.isin(col, idx)
    return np.asarray(coo.data)[m].astype(np.float64)


def _reads_all(pool, descs, seed):
    from research.runners._onebrain_wave1_organread_verify import _isolated_reads
    return _isolated_reads(pool, descs, seed)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  ARM X instrument — the ladder holds arousal, then the surprise organ reads, in ONE sequence.
# ─────────────────────────────────────────────────────────────────────────────────────────────
def arousal_surprise_battery(pool, ladder, sorg, appraisal, *, xedge_lesion=False, intero_lesion=False,
                             xedge_gain=1.0, assert_pa=ASSERT_PA, include_confirm=True):
    from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state, _snapshot_state
    from research.runners._spiking_expectation_rpe_derisk import _set_drives
    from research.runners.onebrain_affect_pool import XEDGE_GATE
    b, xp = pool.bridge, pool.xp
    ladder.ensure_built(); sorg.ensure_built()
    idx_map, meta = sorg.idx_map, sorg.meta
    nt = int(meta["n_trained"])
    sur_idx = np.asarray(idx_map["surprise"])
    n_sur = max(1, sur_idx.size)
    guard_idx = np.concatenate([ladder._ladder_flat] + [np.asarray(v) for v in ladder.relay_idx.values()])
    contra, conf, ar_rate = [], [], []
    with pool.sequence_isolation():
        _restore_state(b, pool.snap)
        b.cp_external_input_current[:] = 0.0
        with ladder.local_ou():
            ladder.set_gates(intero_lesion=intero_lesion, xedge_lesion=xedge_lesion)
            if ladder._xedge and not xedge_lesion:
                b.set_transmission_gate(XEDGE_GATE, float(xedge_gain))
            try:
                ladder.run_appraisal_phase(appraisal)
                held = _snapshot_state(b, xp)
                trials = [((i + 1) % nt, contra) for i in range(nt)]
                if include_confirm:
                    trials += [(i, conf) for i in range(nt)]
                for t_i, (j, bucket) in enumerate(trials):
                    i = t_i % nt
                    _restore_state(b, held)
                    b._blk = meta["blk"]
                    _set_drives(b, idx_map, {"cue": (i, 600.0)}, xp)
                    for _ in range(PRE_STEPS):
                        b._run_one_simulation_step()
                    _set_drives(b, idx_map, {"cue": (i, 600.0), "patient_asserted": (j, float(assert_pa))}, xp)
                    assert float(np.abs(np.asarray(b.cp_external_input_current)[guard_idx]).max()) == 0.0, \
                        "a ladder rung / relay got external current in the surprise window"
                    s_cnt = a_cnt = 0
                    for _ in range(HOLD):
                        b._run_one_simulation_step()
                        fs = np.asarray(b.cp_firing_states)
                        s_cnt += int(fs[sur_idx].sum())
                        a_cnt += int(fs[ladder.arousal_flat].sum())
                    bucket.append(s_cnt / n_sur / (HOLD * 1e-3))
                    ar_rate.append(a_cnt / max(1, ladder.arousal_flat.size) / (HOLD * 1e-3))
            finally:
                ladder.restore_gates()
        b.cp_external_input_current[:] = 0.0
    thr = float(sorg.threshold)
    return {"appraisal": float(appraisal), "contradict_hz": float(np.mean(contra)),
            "confirm_hz": float(np.mean(conf)) if conf else None,
            "contradict_per_block": [float(x) for x in contra],
            "frac_contradict_over_threshold": float(np.mean([c >= thr for c in contra])),
            "arousal_rung_hz": float(np.mean(ar_rate)), "threshold": thr}


def _surprise_and_ladder(pool, seed, organs=None):
    from research.runners.onebrain_affect_pool import PoolAffectLadder
    from research.runners.surprise_production_organ import SurpriseProductionOrgan
    sorg = organs["surprise"] if organs and "surprise" in organs else SurpriseProductionOrgan(seed=seed, shared=pool)
    lad = PoolAffectLadder(seed, shared=pool)
    return sorg, lad


# ─────────────────────────────────────────────────────────────────────────────────────────────
def calibrate(seed):
    from research.runners.onebrain_affect_pool import build_affect_pool
    t0 = time.time()
    pool = build_affect_pool(seed, xedge=True, xedge_w=CAL_BUILD_W)
    sorg, lad = _surprise_and_ladder(pool, seed)
    sorg.ensure_built()
    print(f"[calibrate seed {seed}] pool N={int(pool.bridge.cp_membrane_potential_v.shape[0])} surprise thr="
          f"{sorg.threshold:.3f} built in {time.time() - t0:.0f}s", flush=True)
    base = arousal_surprise_battery(pool, lad, sorg, 0.0)
    rows, chosen = [], None
    for w in CAL_WEIGHTS:
        g = float(w) / CAL_BUILD_W
        hi = arousal_surprise_battery(pool, lad, sorg, 1.0, xedge_gain=g)
        lo = arousal_surprise_battery(pool, lad, sorg, -1.0, xedge_gain=g)
        sp, sn = hi["contradict_hz"] - base["contradict_hz"], lo["contradict_hz"] - base["contradict_hz"]
        ok = bool(sp >= SHIFT_FLOOR_HZ and sn >= SHIFT_FLOOR_HZ and hi["confirm_hz"] < hi["threshold"])
        rows.append({"w": w, "shift_pos": sp, "shift_neg": sn, "confirm_hi": hi["confirm_hz"],
                     "confirm_base": base["confirm_hz"], "arousal_hz_hi": hi["arousal_rung_hz"], "ok": ok})
        print(f"  w={w:<5} shift+={sp:+.3f} shift-={sn:+.3f} confirm(+1)={hi['confirm_hz']:.3f} "
              f"(base {base['confirm_hz']:.3f}, thr {hi['threshold']:.3f}) arousal={hi['arousal_rung_hz']:.1f}Hz "
              f"{'OK' if ok else '-'}", flush=True)
        if ok and chosen is None:
            chosen = w
    print(f"  chosen XEDGE_W = {chosen} (smallest passing; None = no weight in the sweep passed)", flush=True)
    return {"seed": seed, "base": base, "sweep": rows, "chosen_w": chosen, "elapsed_s": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────────────────────
def verify_seed(seed, arms=("M", "X")):
    from research.runners.onebrain_affect_pool import (
        affect_descriptors, AFFECT_DESCRIPTOR, AFFECT_KEY, PoolAffectLadder, PROD_SWEEP, build_affect_pool,
        XEDGE_W)
    from research.runners.onebrain_merge_framework import merge_organs, substrate_byte_identity
    from research.runners._onebrain_wave1_organread_verify import _isolated_read_one
    from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
    from research.runners.onebrain_wave3_pool_production import get_wave3_pool
    from research.runners.affect_production_organ import tone_level
    from research.runners._stageA_full_integration_derisk import LADDER_NEUTRAL_TOL
    t0 = time.time()
    descs = affect_descriptors()
    keys = [d.key for d in descs]
    carried = [k for k in keys if k != AFFECT_KEY]
    res = {"seed": int(seed), "xedge_w": float(XEDGE_W)}

    # ── the 12-organ pool WITHOUT the synapse (used by both arms) ──
    merged = build_affect_pool(seed, xedge=False)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    w_before = _ladder_weights(merged)
    R_m, A_m, organs_m = _reads_all(merged, descs, seed)
    w_after = _ladder_weights(merged)
    res["n_all_neurons"] = n_all
    print(f"[seed {seed}] 12-organ pool N={n_all} read in {time.time() - t0:.0f}s", flush=True)

    checks = {}
    if "M" in arms:
        # M1 co-residence
        core = merge_organs([AFFECT_DESCRIPTOR], seed, config_descriptors=descs, wire=True)
        c_reads, c_ans = _isolated_read_one(core, AFFECT_DESCRIPTOR, seed)
        d1, wk1, miss1 = _maxdelta(R_m[AFFECT_KEY], c_reads)
        checks["M1_affect_coresidence_byte_identical"] = bool(d1 == 0.0 and not miss1 and c_ans == A_m[AFFECT_KEY])
        res["M1"] = {"maxdelta": d1, "worst_key": wk1, "missing": miss1}
        # M2 carried organs vs the shipped wave-3 pool
        ship = get_wave3_pool(seed)
        R_s, A_s, _ = _reads_all(ship, _wave3_descriptors(), seed)
        m2 = {}
        for k in carried:
            dd, wk, miss = _maxdelta(R_m[k], R_s[k])
            m2[k] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                     "read_byte_identical": bool(dd == 0.0 and not miss), "answer_same": bool(A_m[k] == A_s[k])}
        checks["M2_carried_11_read_byte_identical_and_answer_same"] = all(
            v["read_byte_identical"] and v["answer_same"] for v in m2.values())
        res["M2"] = m2
        # M3 affect answer vs today's standalone production ladder
        std = PoolAffectLadder(seed, shared=None)
        std_diffs = [std.read_differential(a)["differential"] for a in PROD_SWEEP]
        std_levels = tuple(int(tone_level(d)) for d in std_diffs)
        pool_diffs = [R_m[AFFECT_KEY][f"diff[{a:+.1f}]"] for a in PROD_SWEEP]
        checks["M3_affect_tone_levels_equal_standalone"] = bool(std_levels == tuple(A_m[AFFECT_KEY]))
        res["M3"] = {"sweep": list(PROD_SWEEP), "pool_diffs": pool_diffs, "standalone_diffs": std_diffs,
                     "pool_levels": list(A_m[AFFECT_KEY]), "standalone_levels": list(std_levels)}
        # M4 alive
        r = R_m[AFFECT_KEY]
        signs = all((r[f"diff[{a:+.1f}]"] > 0) == (a > 0) and r[f"diff[{a:+.1f}]"] != 0.0
                    for a in PROD_SWEEP if abs(a) >= 0.5)
        neutral = abs(r["diff[+0.0]"]) < LADDER_NEUTRAL_TOL
        les = r["diff_lesion[+0.7]"] == 0.0
        il = (abs(r["diff_intero_lesion[+1.0]"]) <= INTERO_COLLAPSE * abs(r["diff[+1.0]"])
              and abs(r["diff_intero_lesion[-1.0]"]) <= INTERO_COLLAPSE * abs(r["diff[-1.0]"]))
        checks["M4_affect_alive_on_pool"] = bool(signs and neutral and les and il)
        res["M4"] = {"signs": signs, "neutral": neutral, "readout_lesion_zero": les, "intero_lesion_collapses": il}
        # M5 frozen
        checks["M5_ladder_weights_frozen"] = bool(w_before.shape == w_after.shape and w_before.size > 0
                                                  and float(np.max(np.abs(w_before - w_after))) == 0.0)
        res["M5"] = {"n_ladder_synapses": int(w_before.size)}
        # M6 legacy discriminator
        leg_m = merge_organs(descs, seed, legacy=True)
        leg_c = merge_organs([AFFECT_DESCRIPTOR], seed, config_descriptors=descs, legacy=True)
        lbi = substrate_byte_identity(leg_m, leg_c, list(AFFECT_DESCRIPTOR.regions))
        checks["M6_legacy_discriminator_diverges"] = bool(lbi["maxerr"] > 0.0)
        res["M6"] = {"legacy_maxerr": lbi["maxerr"]}
        # M7 determinism
        lad2 = PoolAffectLadder(seed, shared=merged)
        again = [lad2.read_differential(a)["differential"] for a in PROD_SWEEP]
        checks["M7_affect_read_deterministic"] = bool(again == pool_diffs)
        print(f"[seed {seed}] ARM M: " + " ".join(f"{k.split('_')[0]}={v}" for k, v in checks.items()), flush=True)

    if "X" in arms:
        # X-arm pool: the SAME 12 organs + the arousal->surprise synapse
        xpool = build_affect_pool(seed, xedge=True)
        R_x, A_x, organs_x = _reads_all(xpool, descs, seed)
        x5_reads = {}
        for k in keys:
            dd, wk, miss = _maxdelta(R_x[k], R_m[k])
            x5_reads[k] = {"maxdelta": dd, "worst_key": wk, "same_answer": bool(A_x[k] == A_m[k]),
                           "byte_identical": bool(dd == 0.0 and not miss)}
        sx, lx = _surprise_and_ladder(xpool, seed, organs_x)
        sm, lm = _surprise_and_ladder(merged, seed, organs_m)
        base = arousal_surprise_battery(xpool, lx, sx, 0.0)
        pos = arousal_surprise_battery(xpool, lx, sx, 1.0)
        neg = arousal_surprise_battery(xpool, lx, sx, -1.0)
        pos_again = arousal_surprise_battery(xpool, lx, sx, 1.0)
        les = arousal_surprise_battery(xpool, lx, sx, 1.0, xedge_lesion=True)
        inull = arousal_surprise_battery(xpool, lx, sx, 1.0, intero_lesion=True)
        nb = arousal_surprise_battery(merged, lm, sm, 0.0)
        npos = arousal_surprise_battery(merged, lm, sm, 1.0)
        weak0 = arousal_surprise_battery(xpool, lx, sx, 0.0, assert_pa=ASSERT_PA_WEAK, include_confirm=False)
        weak1 = arousal_surprise_battery(xpool, lx, sx, 1.0, assert_pa=ASSERT_PA_WEAK, include_confirm=False)
        sp = pos["contradict_hz"] - base["contradict_hz"]
        sn = neg["contradict_hz"] - base["contradict_hz"]
        sl = les["contradict_hz"] - base["contradict_hz"]
        si = inull["contradict_hz"] - base["contradict_hz"]
        s_null = npos["contradict_hz"] - nb["contradict_hz"]
        lever("arousal->surprise synapse gate (intact vs lesion, C(+1))", pos["contradict_hz"], les["contradict_hz"],
              required=False)
        void_x = void_if(pos["arousal_rung_hz"] == 0.0, "the arousal ladder never latched at a=+1 — the "
                                                         "cross-edge had no presynaptic activity to carry")
        checks["X1_shift_both_valences"] = bool(sp >= SHIFT_FLOOR_HZ and sn >= SHIFT_FLOOR_HZ) and not void_x
        checks["X2_lesion_collapses"] = bool(sp > 0 and sl <= LESION_RATIO * sp)
        checks["X3_no_edge_null_exact_zero"] = bool(s_null == 0.0)
        checks["X4_intero_null_collapses"] = bool(sp > 0 and si <= LESION_RATIO * sp)
        checks["X5_byte_off_inert_at_rest"] = bool(base["contradict_hz"] == nb["contradict_hz"]
                                                   and base["confirm_hz"] == nb["confirm_hz"]
                                                   and all(v["byte_identical"] and v["same_answer"]
                                                           for v in x5_reads.values()))
        checks["X6_held_arousal_carries"] = bool(pos["arousal_rung_hz"] > 0 and neg["arousal_rung_hz"] > 0
                                                 and base["arousal_rung_hz"] == 0.0)
        checks["X7_deterministic"] = bool(pos_again["contradict_hz"] == pos["contradict_hz"]
                                          and pos_again["contradict_per_block"] == pos["contradict_per_block"])
        conf_shift = pos["confirm_hz"] - base["confirm_hz"]
        res["X"] = {"base": base, "pos": pos, "neg": neg, "lesion": les, "intero_null": inull,
                    "noedge_base": nb, "noedge_pos": npos, "shift_pos": sp, "shift_neg": sn,
                    "shift_lesion": sl, "shift_intero_null": si, "shift_noedge": s_null,
                    "confirm_shift_pos": conf_shift,
                    "gain_ratio_contradict_over_confirm": (sp / conf_shift) if conf_shift > 0 else None,
                    "attributable_frac": attributable_to("arousal->surprise synapse owns the shift", sp, sl),
                    "weak_assert_frac_over_thr": {"a0": weak0["frac_contradict_over_threshold"],
                                                  "a1": weak1["frac_contradict_over_threshold"]},
                    "x5_reads": x5_reads}
        print(f"[seed {seed}] ARM X: shift+={sp:+.3f} shift-={sn:+.3f} lesion={sl:+.3f} intero-null={si:+.3f} "
              f"no-edge={s_null:+.3f} arousal+={pos['arousal_rung_hz']:.1f}Hz base_arousal="
              f"{base['arousal_rung_hz']:.1f}Hz | " + " ".join(f"{k.split('_')[0]}={v}" for k, v in checks.items()
                                                                if k.startswith("X")), flush=True)

    res["checks"] = checks
    res["GO"] = bool(checks) and all(checks.values())
    res["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[seed {seed}] GO={res['GO']} ({res['elapsed_s']}s)", flush=True)
    return res


def aggregate(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        rows.extend(d.get("per_seed", []))
    seeds = sorted({r["seed"] for r in rows})
    n_go = sum(1 for r in rows if r.get("GO"))
    missing = sorted(set(SEEDS) - set(seeds))
    print(f"affect->one-brain-pool verify: {n_go}/{len(rows)} seeds GO; seeds present {seeds}; missing {missing}")
    for r in rows:
        bad = [k for k, v in r.get("checks", {}).items() if not v]
        print(f"  seed {r['seed']}: GO={r.get('GO')} failed={bad}")
    undefined_if_empty("affect->pool 6-seed GO", len(rows), n_go, len(SEEDS))
    all_go = bool(len(rows) == len(SEEDS) and not missing and n_go == len(SEEDS))
    print(f"ALL-GO (6/6, every M1-M7 + X1-X7): {all_go}")
    return all_go


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--arms", default="MX", help="M, X or MX")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.aggregate:
        paths = []
        for p in a.aggregate:
            paths.extend(sorted(glob.glob(p)) or [p])
        ok = aggregate(paths)
        sys.exit(0 if ok else 1)
    if a.calibrate:
        out = {"mode": "calibrate", "per_seed": [calibrate(s) for s in a.seeds]}
    else:
        out = {"mode": "verify", "arms": a.arms, "per_seed": [verify_seed(s, arms=tuple(a.arms)) for s in a.seeds],
               "preregistered": {"SHIFT_FLOOR_HZ": SHIFT_FLOOR_HZ, "LESION_RATIO": LESION_RATIO,
                                 "INTERO_COLLAPSE": INTERO_COLLAPSE}}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
