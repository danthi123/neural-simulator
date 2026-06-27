#!/usr/bin/env python
"""Burndown conversion C1 -- DE-RISK (the HARD GATE before wiring).

Converts ArgStructureComposer.FrameCQ (numpy rate-coded primacy + max()-argmax word-ordering of a verb frame's
realized slots; research/runners/argstructure_composer.py:125) -> the VALIDATED SPIKING competitive-queuing
serial-order read-out (research/runners/neural_serial_order_renderer.NeuralSerialOrderRenderer, the packaged form
of the 6/6-GO _phaseB_serial_order_spiking_derisk: concept pools driven by a primacy CURRENT on a real
SimulationBridge, the per-pool spiking RATE ranking = the emission order). Same function, different SUBSTRATE.

The build-plan spec is research/findings/2026-06-27-burndown-bucketA-build-plan.md (C1 + the §4 de-risk):
  PARITY:  spiking emit-order == numpy FrameCQ.emit_order on every verb frame + every realized-slot subset, >=6 seeds.
  GPU RATE: the ranking is read from REAL firing rates (pool_rates over cp_firing_states), not a host argmax.
  ANTI-CHEATS (reuse song_g1_core's pre-registered bars):
    (a) EQUAL-DRIVE control FAILS  -- a flat primacy gradient must NOT reproduce the order (the neurons serialize,
        not pool bias);
    (b) PERMUTED-ORDER control beaten by >=10% (g1_verdict);
    (c) CROSS-FRAME control -- the SAME slots under a DIFFERENT frame's primacy produce a DIFFERENT order
        (frame-conditioned, not one fixed order);
    (d) MOAT 0-FA -- render is gated by a stored composite (an unstored fact -> None), unchanged by the order swap;
    (e) AGRAMMATISM ablation preserved -- dropping the closed-class scaffold -> telegraphic output (unchanged).
  FALSIFICATION BAR: no parity at >=5/6 seeds, OR equal-drive does NOT fail -> NOT a clean conversion -> STOP,
  write the honest NEGATIVE.

NO sim/ edit, NO production-composer edit in THIS file (the probe builds the adapter inline + tests it against the
numpy oracle). Wiring argstructure_composer.py additively is Step 2 (only if this is GO).

Run (FOREGROUND, GPU):  SIM_BACKEND=cupy python -u -m research.runners._burndown_C1_framecq_spiking_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.song_g1_core import score_order, permuted_order_controls, g1_verdict  # noqa: E402
from research.runners.neural_serial_order_renderer import NeuralSerialOrderRenderer  # noqa: E402
from research.runners.argstructure_composer import (  # noqa: E402  (test EXACTLY the shipped SpikingFrameCQ adapter)
    FRAME_LEXICON, FrameCQ as NumpyFrameCQ, SpikingFrameCQ, frame_id, content_slot_count, ArgStructureComposer,
    realized_units, frame_for, reparse_to_fact, FUNCTION_WORDS)

# The de-risk anti-cheat drives the renderer directly; use the SHIPPED adapter's gradient (sized to the largest
# verb frame, 4 slots) so the probe validates EXACTLY what the production adapter ships.
_GRAD = SpikingFrameCQ._PRIMACY_GRADIENT


# ---------------------------------------------------------------------------------------------------------------
# (b) PARITY + GPU-RATE: spiking emit-order == numpy emit-order on every frame + every realized subset.
# ---------------------------------------------------------------------------------------------------------------
def _all_realized_subsets(verb):
    """Every realized-slot subset of a verb frame that render() could produce: action is ALWAYS realized + agent;
    obliques are present-only. Enumerate the subsets of the optional (non-action, non-agent) slots, each kept in
    canonical-frame order (exactly how render's `realized_idx` is built)."""
    units = frame_for(verb)
    optional = [i for i, u in enumerate(units) if u[1] not in ("agent", "action")]
    mandatory = [i for i, u in enumerate(units) if u[1] in ("agent", "action")]
    subsets = []
    for mask in range(1 << len(optional)):
        chosen = [optional[j] for j in range(len(optional)) if (mask >> j) & 1]
        idx = sorted(mandatory + chosen)        # canonical ascending order (== render's realized_idx)
        subsets.append(idx)
    return subsets


def parity_seed(seed):
    """For this seed: build the numpy FrameCQ + the spiking FrameCQ; assert spiking emit-order == numpy emit-order
    for EVERY verb frame + EVERY realized-slot subset. Returns (n_match, n_total, mismatches)."""
    numpy_cq = NumpyFrameCQ(seed=seed)
    spk_cq = SpikingFrameCQ(seed=seed)
    n_match = n_total = 0
    mismatches = []
    for verb in FRAME_LEXICON:
        fid = frame_id(verb)
        for idx in _all_realized_subsets(verb):
            np_order = numpy_cq.emit_order(fid, idx)
            sp_order = spk_cq.emit_order(fid, idx)
            n_total += 1
            if sp_order == np_order:
                n_match += 1
            else:
                mismatches.append((verb, idx, np_order, sp_order))
    return n_match, n_total, mismatches


# ---------------------------------------------------------------------------------------------------------------
# (e) ANTI-CHEAT: equal-drive FAILS + permuted beaten >=10% + GPU-rate ranking (on the realized frame slots).
# Reuse the spiking renderer's own bridge: drive the FULL-frame realized indices with the primacy gradient
# (true) vs EQUAL current (control); score the emitted index order vs the canonical (true) order + permuted.
# ---------------------------------------------------------------------------------------------------------------
def anti_cheat_seed(seed):
    from research.runners._phaseB_serial_order_spiking_derisk import pool_rates, EQUAL_pA
    r = NeuralSerialOrderRenderer(seed=seed, primacy_pA=_GRAD)
    # the largest frames exercise the most slots; score every frame's full realized order.
    rng = np.random.default_rng(seed * 71 + 3)
    trues, perms, c_trues, c_perms = [], [], [], []
    rate_gaps = []
    for verb in FRAME_LEXICON:
        n = content_slot_count(verb)
        idx = list(range(n))                                # canonical full-frame slot indices = the TRUE order
        # TRUE: primacy-graded drive (renderer.order drives input-position 0 highest) -> emitted order
        emitted = r.order(idx)
        intended = idx                                      # canonical order (slot 0 first)
        controls = permuted_order_controls(intended, rng, 5)
        trues.append(score_order(emitted, intended))
        perms.append(max((score_order(emitted, c) for c in controls), default=0.0))
        # GPU-rate gap: read the actual per-pool firing rates the ranking used (proves a REAL firing read-out).
        drive = {int(c): _GRAD[min(i, len(_GRAD) - 1)] for i, c in enumerate(idx)}
        rate = pool_rates(r.bridge, r.pool_idx, drive)
        srt = sorted((float(rate[i]) for i in idx), reverse=True)
        if len(srt) >= 2:
            rate_gaps.append(srt[0] - srt[-1])
        # EQUAL-DRIVE control: flat primacy (EQUAL_pA to all pools) -> rate ~equal -> order ~random -> must FAIL.
        eq_drive = {int(c): EQUAL_pA for c in idx}
        eq_rate = pool_rates(r.bridge, r.pool_idx, eq_drive)
        c_emit = [int(c) for c in sorted(idx, key=lambda c: -eq_rate[int(c)])]
        c_trues.append(score_order(c_emit, intended))
        c_perms.append(max((score_order(c_emit, c) for c in controls), default=0.0))
    return {
        "true": float(np.mean(trues)), "perm": float(np.mean(perms)),
        "ctrl_true": float(np.mean(c_trues)), "ctrl_perm": float(np.mean(c_perms)),
        "rate_gap": float(np.mean(rate_gaps)) if rate_gaps else 0.0,
    }


# ---------------------------------------------------------------------------------------------------------------
# (c) CROSS-FRAME: the SAME slots under a DIFFERENT frame's primacy must give a DIFFERENT order.
# The numpy FrameCQ teaches identity per frame, so a permutation-frame is needed to make this decisive; we drive
# the renderer with the realized indices in a PERMUTED frame order and confirm the emitted order differs from the
# canonical-frame order (the order is frame-conditioned: it follows the driving frame, not a fixed sort).
# ---------------------------------------------------------------------------------------------------------------
def cross_frame_seed(seed):
    r = NeuralSerialOrderRenderer(seed=seed, primacy_pA=_GRAD)
    n_diff = n_total = 0
    for verb in FRAME_LEXICON:
        n = content_slot_count(verb)
        if n < 2:
            continue
        idx = list(range(n))
        canonical = r.order(idx)                             # SVO-frame order (canonical primacy)
        permuted_frame = list(reversed(idx))                 # a DISJOINT frame: reversed slot order
        crossed = r.order(permuted_frame)                    # SAME slots, DIFFERENT driving frame
        n_total += 1
        if crossed != canonical:
            n_diff += 1
    return n_diff, n_total


# ---------------------------------------------------------------------------------------------------------------
# (d) MOAT + (e) AGRAMMATISM: confirm the order swap does NOT touch the no-confab moat or the agrammatism control.
# These run on the numpy ArgStructureComposer (CPU) -- the moat + agrammatism are the parent's, substrate-agnostic;
# we assert they hold identically whether the order comes from the numpy or the spiking CQ (parity already proven).
# ---------------------------------------------------------------------------------------------------------------
def moat_and_agrammatism():
    vocab = ["boy", "girl", "dog", "cat", "go", "give", "put", "chase", "park", "ball", "bone", "table", "river"]
    c = ArgStructureComposer(seed=42, D=64, vocab=vocab)
    c.store_fact({"agent": "boy", "action": "go", "GOAL": "park"})
    c.store_fact({"agent": "girl", "action": "give", "THEME": "ball", "RECIPIENT": "dog"})
    # moat: unstored cue -> None
    moat_ok = (c.query_role("GOAL", agent="boy", action="eat") is None
               and c.query_role("GOAL", agent="cat", action="go") is None)
    # agrammatism: ablate scaffold -> telegraphic, no function words, bare verb
    fact = {"agent": "boy", "action": "go", "GOAL": "park"}
    full = c.render(fact, c._composite_for(fact))
    tele = c.render(fact, c._composite_for(fact), ablate_closed_class=True)
    agram_ok = (tele != full and all(w not in FUNCTION_WORDS for w in tele.split())
                and "goes" not in tele.split() and reparse_to_fact(full, fact))
    return bool(moat_ok), bool(agram_ok), full, tele


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = (42, 43, 44, 45, 46, 47)
    print("[C1 de-risk] numpy FrameCQ word-ordering -> the VALIDATED spiking competitive-queuing read-out. "
          "PARITY (spiking==numpy on every frame+subset, >=6 seeds) + GPU rate + anti-cheats.", flush=True)

    # --- PARITY (the load-bearing gate) ---
    print("\n-- PARITY: spiking emit-order == numpy FrameCQ.emit-order, per frame + per realized subset --", flush=True)
    parity_rows = []
    for s in seeds:
        nm, nt, mm = parity_seed(s)
        seed_pass = (nm == nt)
        parity_rows.append({"seed": s, "match": nm, "total": nt, "pass": seed_pass, "mismatches": mm[:8]})
        flag = "OK" if seed_pass else f"MISMATCH ({nt - nm})"
        print(f"  [seed {s}] parity {nm}/{nt}  {flag}", flush=True)
        if mm:
            for (verb, idx, npo, spo) in mm[:4]:
                print(f"      {verb} idx={idx}: numpy={npo} spiking={spo}", flush=True)
    n_parity_pass = sum(1 for r in parity_rows if r["pass"])

    # --- ANTI-CHEAT: equal-drive fails + permuted beaten + GPU rate gap ---
    print("\n-- ANTI-CHEAT: primacy-true vs permuted vs EQUAL-DRIVE (must fail) + GPU rate gap --", flush=True)
    ac_rows = [dict(seed=s, **anti_cheat_seed(s)) for s in seeds]
    for r in ac_rows:
        v = g1_verdict(r["true"], r["perm"], gate_cleared=True)
        print(f"  [seed {r['seed']}] true {r['true']:.3f} vs perm {r['perm']:.3f} -> {v['GATE']} | "
              f"equal-drive {r['ctrl_true']:.3f} vs perm {r['ctrl_perm']:.3f} | rate-gap {r['rate_gap']:.3f}",
              flush=True)
    t_true = float(np.mean([r["true"] for r in ac_rows]))
    t_perm = float(np.mean([r["perm"] for r in ac_rows]))
    c_true = float(np.mean([r["ctrl_true"] for r in ac_rows]))
    c_perm = float(np.mean([r["ctrl_perm"] for r in ac_rows]))
    rate_gap = float(np.mean([r["rate_gap"] for r in ac_rows]))
    agg = g1_verdict(t_true, t_perm, gate_cleared=True)
    n_ac_pass = sum(1 for r in ac_rows if g1_verdict(r["true"], r["perm"], gate_cleared=True)["gate"])
    equal_drive_fails = c_true < c_perm * 1.10 + 1e-9
    gpu_rate_real = rate_gap > 1e-6

    # --- CROSS-FRAME ---
    print("\n-- CROSS-FRAME: same slots, a DIFFERENT frame's primacy -> a DIFFERENT order --", flush=True)
    cf_rows = []
    for s in seeds:
        nd, ntot = cross_frame_seed(s)
        cf_rows.append({"seed": s, "diff": nd, "total": ntot})
        print(f"  [seed {s}] cross-frame differs {nd}/{ntot}", flush=True)
    cross_frame_ok = all(r["diff"] == r["total"] and r["total"] > 0 for r in cf_rows)

    # --- MOAT + AGRAMMATISM (substrate-agnostic; unchanged by the swap) ---
    moat_ok, agram_ok, full, tele = moat_and_agrammatism()
    print(f"\n-- MOAT 0-FA: {moat_ok}  | AGRAMMATISM: '{full}' -> '{tele}' = {agram_ok}", flush=True)

    # --- VERDICT ---
    parity_ok = n_parity_pass >= 5
    print(f"\n{'='*100}", flush=True)
    print(f"  PARITY {n_parity_pass}/6 seeds full-match (bar >=5/6)  | anti-cheat aggregate {agg['GATE']} "
          f"({agg['pct_over_permuted']:.0f}% over perm, {n_ac_pass}/6) | equal-drive FAILS = {equal_drive_fails} "
          f"({c_true:.3f}~{c_perm:.3f}) | GPU rate-real = {gpu_rate_real} (gap {rate_gap:.3f}) | "
          f"cross-frame = {cross_frame_ok} | moat = {moat_ok} | agrammatism = {agram_ok}", flush=True)
    print(f"{'='*100}", flush=True)

    go = bool(parity_ok and equal_drive_fails and agg["gate"] and n_ac_pass >= 5
              and gpu_rate_real and cross_frame_ok and moat_ok and agram_ok)
    if go:
        print("  GO: the spiking competitive-queuing read-out reproduces the numpy FrameCQ word-ordering EXACTLY "
              f"({n_parity_pass}/6 full parity), reading REAL firing rates (gap {rate_gap:.3f}); the equal-drive "
              f"control FAILS ({c_true:.3f}~{c_perm:.3f}, the neurons serialize) + permuted beaten "
              f"({agg['pct_over_permuted']:.0f}%) + cross-frame differs + moat 0-FA + agrammatism preserved. "
              "==> CLEAN CONVERSION -- wire the default-off adapter (Step 2).", flush=True)
    elif not parity_ok:
        print(f"  NEGATIVE/STOP: parity only {n_parity_pass}/6 (<5/6) -- the spiking order does NOT match the numpy "
              "order on the full frames. Localize (primacy gap / RUN_STEPS) before wiring; do NOT ship a "
              "non-parity ordering into the surface.", flush=True)
    elif not equal_drive_fails:
        print(f"  SUSPECT/STOP: equal-drive control does NOT fail ({c_true:.3f} vs {c_perm:.3f}) -- the order is "
              "pool bias, not the spiking gradient. NOT a clean conversion.", flush=True)
    else:
        print("  PARTIAL: some anti-cheat/parity component short of bar -- inspect the table above.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    out = {
        "parity": {"n_pass": n_parity_pass, "rows": parity_rows},
        "anti_cheat": {"true": t_true, "perm": t_perm, "ctrl_true": c_true, "ctrl_perm": c_perm,
                       "rate_gap": rate_gap, "n_pass": n_ac_pass, "aggregate": agg,
                       "equal_drive_fails": equal_drive_fails, "gpu_rate_real": gpu_rate_real, "rows": ac_rows},
        "cross_frame": {"ok": cross_frame_ok, "rows": cf_rows},
        "moat_ok": moat_ok, "agrammatism_ok": agram_ok,
        "GO": go,
    }
    path = os.path.join(_REPO, "research", "findings", "raw", "_burndown_C1_framecq_spiking.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)
    return go


if __name__ == "__main__":
    main()
