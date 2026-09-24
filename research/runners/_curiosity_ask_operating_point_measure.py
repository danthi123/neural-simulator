"""READ THE SUBSTRATE before choosing v5's companion process (curiosity x metacog, the rung after v4's NO-GO 3/6).

v4 (`_curiosity_lcne_phasic_gain_derisk.py`, finding
`research/findings/2026-09-24-curiosity-metacog-lcne-phasic-gain-v4-6seed-NOGO-3of6-gain-holds-circuit-residual.md`)
failed seeds 44 (G1) and 100 (G7) on gates the gain does not control. The finding's own hypothesis was an
OPERATING-POINT problem (each seed's ASK pool sits at a different place on its firing curve), with a homeostatic
set-point as the companion process. This runner MEASURES that hypothesis on DEV seeds only (7-12), before any
mechanism is chosen:

  1. THE ASK POOL'S OWN OPERATING POINT. ASK's response to the point-edge's drive at the most uncertain evidence
     level, with lc_ne's gain pathway closed (the reference arm v4's G11 uses), over a fine edge-drive grid:
     baseline rate (edge closed), threshold drive (first drive with ASK >= ASK_ON_HZ), slope over the rising limb,
     and the drive where the reference peaks. Also the same curve with ASK's slow feedback loop open (FB loop closed).
  2. THE COMPARATOR'S OWN OUTPUT PER EVIDENCE LEVEL. metacog's margin comparator (`meta_schema`) is two halves,
     meta_0 and meta_1 (the per-class comparators). The point-edge sums BOTH onto ASK. Per level: meta_0 / meta_1 /
     total rates, the relay halves, the two workspace assemblies, and ASK -- in the intact arm, the class-swap arm
     and the gain-lesioned arm.
  3. THE DECISIVE QUESTION, in numbers: is the edge's INPUT to ASK (the comparator total) already non-monotone in
     evidence? If input(level j) >= input(level i) for a more-confident j, no operating point of ASK -- no threshold
     shift, no slope change, any monotone transfer f -- can put ASK(j) below ASK(i). `best_threshold_rho` is the
     best Spearman rho reachable by ANY threshold applied to the measured input (the operating-point route's
     ceiling), with the number of levels it silences.

Measurement only: no gate, no verdict, no mechanism. DEV seeds only (refuses 42/43/44/100/101/102). Every substrate
is built through v4's `build_v4_pool` with `seed=` (cfg.seed set by merge_organs).

Run (one dev seed per process):
  SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -u -m research.runners._curiosity_ask_operating_point_measure \\
      --seeds 7 --out research/findings/raw/_curiosity_ask_op_measure_dev_s7.json
  python -m research.runners._curiosity_ask_operating_point_measure --summarize <files> --out <summary.json>

FUNCTIONAL READ-OUTS ONLY -- rates of named spiking populations; no felt state is asserted.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import resource
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from research.runners import _curiosity_lcne_phasic_gain_derisk as V4  # noqa: E402
from research.runners._curiosity_metacog_conflict_xedge_derisk import (  # noqa: E402
    EVIDENCE_GRID, spearman, RELAY_N_PER_CLASS,
)

MEASURE_DEV_SEEDS = frozenset({7, 8, 9, 10, 11, 12})
EVAL_SEEDS = frozenset({42, 43, 44, 100, 101, 102})
FI_GRID = tuple(float(x) for x in np.round(np.arange(0.50, 1.60 + 1e-9, 0.05), 2))   # edge_drive, fine
ASK_ON_HZ = 0.5          # "ASK responds" for the threshold-drive read-out (reported; not a gate)


class MeasureRecorder(V4.Recorder):
    """v4's Recorder plus the comparator halves, the relay halves and the two workspace assemblies. v4's
    `read_level` turns every key of `idx` into a `<key>_hz` level mean, so adding keys is all it needs."""

    def __init__(self, pool):
        super().__init__(pool)
        rm = pool.bridge.region_manager
        meta = np.asarray(rm.indices("meta_schema"), np.int64)
        rel = np.asarray(rm.indices("meta_margin_fs"), np.int64)
        ws = np.asarray(rm.indices("workspace"), np.int64)
        hm, hw = meta.size // 2, ws.size // 2
        self.idx.update({"meta_0": meta[:hm], "meta_1": meta[hm:], "relay_0": rel[:RELAY_N_PER_CLASS],
                         "relay_1": rel[RELAY_N_PER_CLASS:], "asm_0": ws[:hw], "asm_1": ws[hw:]})
        self.reset()


def _sweep(pool, org, rec, gates, swap=False):
    b = pool.bridge
    V4._set(b, **V4.MECH_GATES)
    V4._set(b, **gates)
    V4.prime(pool, org, rec)
    sw = V4.sweep(pool, org, rec, swap=swap)
    V4._set(b, **V4.MECH_GATES)
    keys = ("ask", "lc_ne", "ask_fb", "lc_add", "meta_0", "meta_1", "relay_0", "relay_1", "asm_0", "asm_1")
    return {"levels": [{"evidence": l["evidence"], "balance": l["balance"], "confident": l["confident"],
                        **{f"{k}_hz": l[f"{k}_hz"] for k in keys}} for l in sw["levels"]]}


def _fi_curve(pool, org, rec, gates):
    """ASK Hz at the most uncertain level (evidence 0) vs the point-edge's drive scale."""
    b = pool.bridge
    out = {}
    for g in (0.0,) + FI_GRID:
        V4._set(b, **V4.MECH_GATES)
        V4._set(b, **gates)
        V4._set(b, **{V4.EDGE_GATE: g})
        V4.prime(pool, org, rec)
        with pool.sequence_isolation():
            lv = V4.read_level(pool, org, rec, 0.0)
        out[g] = {"ask_hz": lv["ask_hz"], "ask_fb_hz": lv["ask_fb_hz"],
                  "meta_total_hz": 0.5 * (lv["meta_0_hz"] + lv["meta_1_hz"])}
    V4._set(b, **V4.MECH_GATES)
    return out


def operating_point_readout(fi: dict) -> dict:
    """baseline / threshold drive / rising-limb slope / peak of one ASK drive-response curve (fi: drive -> Hz)."""
    grid = sorted(g for g in fi if g > 0.0)
    y = [fi[g] for g in grid]
    peak_i = int(np.argmax(y))
    on = [g for g in grid if fi[g] >= ASK_ON_HZ]
    d_th = on[0] if on else None
    limb = [g for g in grid[:peak_i + 1] if fi[g] >= 0.25]
    slope = None
    if len(limb) >= 2:
        slope = float(np.polyfit(limb, [fi[g] for g in limb], 1)[0])
    return {"baseline_hz_edge_closed": float(fi[0.0]), "threshold_drive": d_th, "peak_drive": grid[peak_i],
            "peak_hz": float(y[peak_i]), "rising_limb": limb, "n_rising_limb": len(limb),
            "rising_limb_slope_hz_per_unit_drive": slope,
            "falls_after_peak": bool(peak_i < len(grid) - 1 and min(y[peak_i:]) < 0.8 * y[peak_i])}


def best_threshold_rho(inp) -> dict:
    """The operating-point route's CEILING: apply every threshold t to the measured input (levels with input < t ->
    0, i.e. silenced ASK; the rest keep their rank) and take the most negative Spearman rho(evidence, output).
    Any monotone non-decreasing transfer of the input gives outputs whose ranks are the input's ranks with some
    adjacent ranks merged; a threshold is the merge that helps a U-shaped input most."""
    x = list(EVIDENCE_GRID)
    v = np.asarray(inp, np.float64)
    best = {"rho": spearman(x, list(v)), "threshold": None, "n_silenced": 0, "n_active": int(v.size)}
    for t in sorted(set(v.tolist())):
        out = np.where(v < t, 0.0, v)
        r = spearman(x, list(out))
        if r is not None and (best["rho"] is None or r < best["rho"] - 1e-12):
            best = {"rho": r, "threshold": float(t), "n_silenced": int((v < t).sum()), "n_active": int((v >= t).sum())}
    return best


def comparator_readout(sw) -> dict:
    lv = sw["levels"]
    m0 = np.asarray([l["meta_0_hz"] for l in lv])
    m1 = np.asarray([l["meta_1_hz"] for l in lv])
    tot = m0 + m1
    fav, riv = m0, m1              # evidence drives assembly 0 in the intact arm (member_dev[0]); swapped in swap
    if np.mean(m1[-3:]) > np.mean(m0[-3:]):
        fav, riv = m1, m0
    mn = np.minimum(m0, m1)
    ask = [l["ask_hz"] for l in lv]
    x = list(EVIDENCE_GRID)
    i_min = int(np.argmin(tot))
    return {"meta_0_hz": m0.tolist(), "meta_1_hz": m1.tolist(), "meta_total_hz": tot.tolist(),
            "meta_min_hz": mn.tolist(), "ask_hz": ask,
            "rho_total": spearman(x, list(tot)), "rho_favored": spearman(x, list(fav)),
            "rho_rival": spearman(x, list(riv)), "rho_min": spearman(x, list(mn)), "rho_ask": spearman(x, ask),
            "total_argmin_evidence": x[i_min], "total_tail_over_min": float(tot[-1] / tot[i_min]) if tot[i_min] > 0 else None,
            "favored_end_over_start": float(fav[-1] / fav[0]) if fav[0] > 0 else None,
            "best_threshold_rho_on_total": best_threshold_rho(tot)}


def measure_seed(seed: int) -> dict:
    t0 = time.time()
    pool = V4.build_v4_pool(seed)
    assert int(pool.bridge.core_config.seed) == int(seed), "cfg.seed must be the substrate seed"
    org = V4.MetacogProductionOrgan(seed=seed, shared=pool)
    org.ensure_built()
    rec = MeasureRecorder(pool)
    V4._set(pool.bridge, **V4.MECH_GATES)
    with pool.sequence_isolation():
        for ev in EVIDENCE_GRID:
            V4.read_level(pool, org, rec, ev)
    arms = {
        "intact": _sweep(pool, org, rec, {}),
        "class_swap": _sweep(pool, org, rec, {}, swap=True),
        "gain_lesion": _sweep(pool, org, rec, {V4.LC_GAIN_GATE: 0.0}),
        "class_swap_gain_lesion": _sweep(pool, org, rec, {V4.LC_GAIN_GATE: 0.0}, swap=True),
    }
    fi = {"off_lc_gain_closed": _fi_curve(pool, org, rec, {V4.LC_GAIN_GATE: 0.0}),
          "open_loop_fb_and_lc_closed": _fi_curve(pool, org, rec, {V4.LC_GAIN_GATE: 0.0, V4.FB_LOOP_GATE: 0.0})}
    thr = np.asarray(V4.to_host(pool.bridge.cp_izh_vpeak))[np.asarray(pool.bridge.region_manager.indices("ask"))]
    # whose ASK range is it? (the lc gain pathway's share, the same subtraction v4's G3 makes; reported only)
    from tools.lab import attributable_to

    def _range(sw):
        v = [l["ask_hz"] for l in sw["levels"]]
        return float(max(v) - min(v))
    gain_share = {
        "intact": attributable_to(f"seed{seed} ASK range = lc gain (intact arm)",
                                  _range(arms["intact"]), _range(arms["gain_lesion"])),
        "class_swap": attributable_to(f"seed{seed} ASK range = lc gain (class-swap arm)",
                                      _range(arms["class_swap"]), _range(arms["class_swap_gain_lesion"])),
    }
    res = {
        "seed": seed,
        "lc_gain_share_of_ask_range": gain_share,
        "operating_point": {k: operating_point_readout({g: v["ask_hz"] for g, v in c.items()})
                            for k, c in fi.items()},
        "fi_curves": {k: {str(g): v for g, v in c.items()} for k, c in fi.items()},
        "comparator": {k: comparator_readout(v) for k, v in arms.items()},
        "arms": arms,
        "ask_vpeak_mean": float(thr.mean()),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    for k, c in res["comparator"].items():
        print(f"[seed {seed}] {k:24s} rho_total={c['rho_total']} rho_fav={c['rho_favored']} rho_min={c['rho_min']} "
              f"rho_ask={c['rho_ask']} argmin={c['total_argmin_evidence']} tail/min={c['total_tail_over_min']} "
              f"best_thr_rho={c['best_threshold_rho_on_total']}", flush=True)
        print(f"[seed {seed}]   total={[round(x, 2) for x in c['meta_total_hz']]} "
              f"ask={[round(x, 2) for x in c['ask_hz']]}", flush=True)
    for k, o in res["operating_point"].items():
        print(f"[seed {seed}] OP {k}: {o}", flush=True)
    print(f"[seed {seed}] elapsed {res['elapsed_s']}s peak_rss {res['peak_rss_mb']:.0f} MB", flush=True)
    return res


def summarize(paths, out):
    rows = []
    for fp in paths:
        d = json.loads(Path(fp).read_text())
        rows.extend(d["per_seed"])
    rows.sort(key=lambda r: r["seed"])
    table = []
    for r in rows:
        op = r["operating_point"]["off_lc_gain_closed"]
        row = {"seed": r["seed"], "threshold_drive": op["threshold_drive"], "peak_drive": op["peak_drive"],
               "peak_hz": op["peak_hz"], "slope": op["rising_limb_slope_hz_per_unit_drive"],
               "n_rising_limb_on_0.05_grid": op["n_rising_limb"]}
        for arm in ("intact", "class_swap"):
            c = r["comparator"][arm]
            row.update({f"{arm}_rho_ask": c["rho_ask"], f"{arm}_rho_total": c["rho_total"],
                        f"{arm}_rho_min": c["rho_min"], f"{arm}_argmin": c["total_argmin_evidence"],
                        f"{arm}_tail_over_min": c["total_tail_over_min"],
                        f"{arm}_best_threshold_rho": c["best_threshold_rho_on_total"]["rho"],
                        f"{arm}_best_threshold_silences": c["best_threshold_rho_on_total"]["n_silenced"]})
        table.append(row)
    Path(out).write_text(json.dumps({"kind": "DEV measurement (not evidence)", "rows": table,
                                     "inputs": list(paths)}, indent=1, default=str))
    for row in table:
        print(row, flush=True)
    return 0


def selftest():
    """The derived read-outs on synthetic data (no simulation), so a read-out bug cannot eat a measurement run."""
    fi = {0.0: 0.0}
    fi.update({g: max(0.0, 4.0 * (g - 0.85)) if g <= 1.2 else 1.0 for g in FI_GRID})
    op = operating_point_readout(fi)
    assert op["threshold_drive"] == 1.0 and op["peak_drive"] == 1.2 and op["falls_after_peak"], op
    assert abs(op["rising_limb_slope_hz_per_unit_drive"] - 4.0) < 1e-6, op
    u = [181, 164, 149, 134, 111, 90, 82, 76, 78, 88, 111]          # a U-shaped input (seed-44-like)
    b = best_threshold_rho(u)
    assert b["rho"] < spearman(list(EVIDENCE_GRID), u) and b["n_silenced"] >= 7, b
    mono = [200, 180, 160, 140, 120, 100, 80, 60, 40, 20, 0]
    assert best_threshold_rho(mono)["rho"] == -1.0
    lv = [{"meta_0_hz": 2.0 + 3.0 * e, "meta_1_hz": 2.0 * (1 - e), "ask_hz": 3.0 * (1 - e)} for e in EVIDENCE_GRID]
    c = comparator_readout({"levels": lv})
    assert c["rho_favored"] == 1.0 and c["rho_rival"] == -1.0 and c["rho_min"] == -1.0, c
    assert c["rho_total"] == 1.0 and c["total_argmin_evidence"] == 0.0, c
    print("[ask op measure selftest] read-outs OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--summarize", nargs="+", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out:
        ap.error("--out is required")
    if a.summarize:
        return summarize(a.summarize, a.out)
    bad = [s for s in a.seeds if s not in MEASURE_DEV_SEEDS]
    if bad:
        print(f"[ask op measure] REFUSED: {bad} -- measurement is for dev seeds {sorted(MEASURE_DEV_SEEDS)} only "
              f"(evaluation seeds {sorted(EVAL_SEEDS)} are held out)", flush=True)
        return 2
    rows = [measure_seed(s) for s in a.seeds]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"kind": "DEV measurement (not evidence)", "per_seed": rows,
                                       "fi_grid": list(FI_GRID), "ask_on_hz": ASK_ON_HZ}, indent=1, default=str))
    print(f"[ask op measure] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
