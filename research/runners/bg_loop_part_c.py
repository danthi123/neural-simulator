#!/usr/bin/env python3
"""Step-0c Part C: isolated-cell f-I curves, kernel-faithful (PACKET.md, ruling
parallel-builds-1021).

For each of the 7 organ presets, with jitter centred on the preset: rate,
ISI-CV and burst fraction against injected current (0..1200 pA), plus the
rebound response to a 200 ms hyperpolarising step. Numpy only, minutes (the
packet's Part-C budget), no GPU.

KERNEL FAITHFUL — mirrors sim/kernels.py's fused Izhikevich-2007 megastep
(strict path at sim/kernels.py:1225-1235, float32):
  - simultaneous v/u update computed from the SAME old (v, u):
      dv = (k (v - vr)(v - vt) - u + I) / C_safe ;  v_dyn = v + dv dt
      du = a (b (v - vr) - u) ;                     u_dyn = u + du dt
  - fired = (v_dyn >= vpeak) & (refractory <= 0)
  - reset: v <- c_reset, u <- u_dyn + d_increment, refractory <- 1
    (refractory_period_steps = 2 at sim/config.py:391; the megastep uses
     refractory_reset = period - 1); otherwise refractory <- max(r-1, 0)
  - dt = 1.0 ms (packet line 70). Isolated cells: I_syn = 0, no OU, no
    conductances, so bounded-integration and lesion flags are inert here.

JITTER CENTRED ON THE PRESET — per-parameter draw specs from
sim/bridge.py:4094-4100 (defaults keyed to the centre): a lognormal
(sigma_log 0.3) when a > 0 else gaussian (std 0.3|a|); b gaussian (std
0.25|b|); d gaussian (std 0.25|d|, or 10.0 when d == 0); C gaussian (std
0.15 C). As in preset-centred heterogeneity (ratio x preset, ratio =
sample / centre kept in [0.1, 3.0] — docs/drafts/preset-centred-het/CENSUS.md
line 166 and tests/test_preset_centred_heterogeneity.py:209), value =
preset * clamp(sample / preset, 0.1, 3.0). The single preset IS the centre
for an isolated population. Seed via --seed (cfg.seed discipline).

Reference ranges: published value + citation where the local corpus
(feature-catalog.md, PBR-160 = basal-ganglia reviews in sim-catalog) gives
one; otherwise "no source found" as the packet directs (PACKET.md line 75).

Usage:
  python -m research.runners.bg_loop_part_c --seed 9 --out <dir>
  python -m research.runners.bg_loop_part_c --presets IZH2007_STN_BURST \
      --n-reps 4 --currents 0 600 1200   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_WT_ROOT = Path(__file__).resolve().parent.parent.parent
for _ in range(3):
    if (_WT_ROOT / "sim" / "__init__.py").exists():
        break
    _WT_ROOT = _WT_ROOT.parent
sys.path.insert(0, str(_WT_ROOT))

from sim.enums import NeuronType, DefaultIzhikevichParamsManager  # noqa: E402

PRESETS = [
    "IZH2007_STN_BURST",
    "IZH2007_GPE_PACEMAKER",
    "IZH2007_GPI_OUTPUT",
    "IZH2007_THALAMIC_RELAY",
    "IZH2007_STRIATAL_MSN_D1",
    "IZH2007_STRIATAL_MSN_D2",
    "IZH2007_FS_CORTICAL_INTERNEURON",
]

# Published range it should match + citation, or "no source found" (packet line 75).
# Sources: /home/dant123/Projects/sim-catalog/references/feature-catalog.md lines
# 237/250/273/132 (which cite PBR-160 chapters; PBR-160 = Tepper/Abercrombie/Bolam
# basal-ganglia review volume in sim-catalog).
REFERENCES = {
    "IZH2007_STN_BURST": {
        "range_hz": "5-15 (autonomous, slice)",
        "citation": "PBR-160 ch 10, Bevan et al. pp 175-176 (feature-catalog.md line 273)",
    },
    "IZH2007_GPE_PACEMAKER": {
        "range_hz": "30-80 (PV+ prototypic, in vivo tonic)",
        "citation": "PBR-160 ch 7 p 112 (feature-catalog.md line 237); ~50 Hz spontaneous PV+ (line 366)",
    },
    "IZH2007_GPI_OUTPUT": {
        "range_hz": "40-80 (tonic in vivo)",
        "citation": "feature-catalog.md lines 132/250 (Part V Ch 30-39; A.04 supplemental)",
    },
    "IZH2007_THALAMIC_RELAY": {"range_hz": "no source found", "citation": ""},
    "IZH2007_STRIATAL_MSN_D1": {"range_hz": "no source found", "citation": ""},
    "IZH2007_STRIATAL_MSN_D2": {"range_hz": "no source found", "citation": ""},
    "IZH2007_FS_CORTICAL_INTERNEURON": {"range_hz": "no source found", "citation": ""},
}

DT_MS = 1.0
REFRACTORY_PERIOD_STEPS = 2  # sim/config.py:391
RATIO_LO, RATIO_HI = 0.1, 3.0  # preset-centred het bounds
BURST_ISI_MS = 50.0  # overridden by --burst-isi-ms in main()


def jittered_params(preset: dict, rng: np.random.Generator) -> dict:
    """Preset-centred jitter of {a, b, d, C}; returns float32 params."""
    out = dict(preset)
    specs = [
        ("a", "lognormal" if preset["a"] > 0 else "gaussian"),
        ("b", "gaussian"),
        ("d_increment", "gaussian"),
        ("C", "gaussian"),
    ]
    for key, kind in specs:
        centre = float(preset[key])
        if kind == "lognormal":
            sample = float(rng.lognormal(mean=np.log(centre), sigma=0.3))
        else:
            if key == "b":
                std = abs(centre) * 0.25
            elif key == "d_increment":
                std = abs(centre) * 0.25 if centre != 0.0 else 10.0
            else:  # C
                std = abs(centre) * 0.15
            sample = float(rng.normal(loc=centre, scale=std))
        if centre == 0.0:
            out[key] = np.float32(sample)  # no ratio exists: additive draw
            continue
        ratio = sample / centre
        ratio = min(max(ratio, RATIO_LO), RATIO_HI)
        out[key] = np.float32(centre * ratio)
    return out


def _metrics(all_spikes, window_ms):
    """Per-rep rate Hz, ISI-CV, burst fraction (ISI <= burst_isi_ms)."""
    rates, cvs, bursts = [], [], []
    for times in all_spikes:
        n = len(times)
        rates.append(n / (window_ms / 1000.0))
        isis = np.diff(np.asarray(times, dtype=np.float64))
        if len(isis) >= 2 and float(isis.mean()) > 0:
            cvs.append(float(isis.std() / isis.mean()))
        elif len(isis) == 1:
            cvs.append(float("nan"))
        bursts.append(
            float((isis <= BURST_ISI_MS).mean()) if len(isis) else float("nan")
        )
    return rates, cvs, bursts


def _mean_sd(xs):
    arr = np.asarray(xs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return None, None
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def run_preset(name, seed, n_reps, currents, hyper_pA,
               stim_ms, settle_ms, rebound_read_ms, v0_offset_mV):
    preset = DefaultIzhikevichParamsManager.get_params(
        NeuronType[name], use_2007_formulation=True)
    rng = np.random.default_rng(seed)
    reps = [jittered_params(preset, rng) for _ in range(n_reps)]
    # Per-rep jittered params as float32 arrays; vectorised over reps.
    C = np.array([r["C"] for r in reps], dtype=np.float32)
    kk = np.array([r["k"] for r in reps], dtype=np.float32)
    vr = np.array([r["vr"] for r in reps], dtype=np.float32)
    vt = np.array([r["vt"] for r in reps], dtype=np.float32)
    aa = np.array([r["a"] for r in reps], dtype=np.float32)
    bb = np.array([r["b"] for r in reps], dtype=np.float32)
    vp = np.array([r["vpeak"] for r in reps], dtype=np.float32)
    cr = np.array([r["c_reset"] for r in reps], dtype=np.float32)
    dd = np.array([r["d_increment"] for r in reps], dtype=np.float32)
    v = vr.copy()
    u = np.zeros(n_reps, dtype=np.float32)
    refrac = np.zeros(n_reps, dtype=np.int32)

    def step_vec(I, refrac):
        C_safe = np.where(C == 0.0, np.float32(1.0), C)
        dv = (kk * (v - vr) * (v - vt) - u + I) / C_safe
        du = aa * (bb * (v - vr) - u)
        v_dyn = v + np.float32(dv * DT_MS)
        u_dyn = u + np.float32(du * DT_MS)
        fired = (v_dyn >= vp) & (refrac <= 0)
        v[...] = np.where(fired, cr, v_dyn)
        u[...] = np.where(fired, u_dyn + dd, u_dyn)
        refrac[...] = np.where(fired, np.int32(1),
                               np.maximum(refrac - 1, 0))
        return fired

    def run_phase(I, n_steps, track_min=False):
        spk = [[] for _ in range(n_reps)]
        v_min = np.full(n_reps, np.float32(1e9), dtype=np.float32)
        for s in range(n_steps):
            fired = step_vec(I, refrac)
            if track_min:
                v_min = np.minimum(v_min, v)
            for i in np.nonzero(fired)[0]:
                spk[int(i)].append(s * DT_MS)
        return spk, v_min

    def reset_state():
        # v starts a small perturbation OFF the exact (vr, u=0) fixed point:
        # starting exactly there freezes even a pacemaker (dv = du = 0
        # float32-exactly), which would wrongly report 0 Hz at I = 0. A +1 mV
        # nudge lets an unstable equilibrium escape to its limit cycle and a
        # stable one decay back, so the 0 pA row reports autonomous behavior.
        v[...] = vr + np.float32(v0_offset_mV)
        u[...] = np.float32(0.0)
        refrac[...] = 0

    n_settle = int(settle_ms / DT_MS)
    n_stim = int(stim_ms / DT_MS)
    fi = []
    for I in currents:
        reset_state()
        run_phase(np.float32(0.0), n_settle)
        spk, _ = run_phase(np.float32(I), n_stim)
        rates, cvs, bursts = _metrics(spk, stim_ms)
        rm, rs = _mean_sd(rates)
        cm, cs = _mean_sd(cvs)
        bm, bs = _mean_sd(bursts)
        fi.append({
            "current_pA": float(I),
            "rate_hz_mean": rm, "rate_hz_sd": rs,
            "isi_cv_mean": cm, "isi_cv_sd": cs,
            "burst_frac_mean": bm, "burst_frac_sd": bs,
            "spikes_per_rep": [len(t) for t in spk],
        })

    # Rebound: settle at 0, 200 ms hyperpolarising step, readout window.
    n_hyper = int(200.0 / DT_MS)
    n_read = int(rebound_read_ms / DT_MS)
    reset_state()
    run_phase(np.float32(0.0), n_settle)
    spk_hyper, v_min_hyper = run_phase(
        np.float32(hyper_pA), n_hyper, track_min=True)
    spk_read, _ = run_phase(np.float32(0.0), n_read)
    rebound = {
        "hyper_current_pA": float(hyper_pA),
        "hyper_ms": 200.0,
        "readout_ms": rebound_read_ms,
        "hyper_min_vm_per_rep": [float(x) for x in v_min_hyper],
        "spikes_during_hyper_per_rep": [len(t) for t in spk_hyper],
        "rebound_spikes_per_rep": [len(t) for t in spk_read],
        "rebound_first_spike_ms_per_rep": [
            (t[0] if t else None) for t in spk_read
        ],
    }
    return {
        "preset": name,
        "preset_params_unjittered": {k: float(vv) for k, vv in preset.items()},
        "reference": REFERENCES.get(
            name, {"range_hz": "no source found", "citation": ""}),
        "fi_curve": fi,
        "rebound": rebound,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--n-reps", type=int, default=20)
    ap.add_argument("--presets", nargs="+", default=list(PRESETS))
    ap.add_argument("--currents", nargs="+", type=float,
                    default=list(range(0, 1201, 100)))
    ap.add_argument("--settle-ms", type=float, default=1000.0)
    ap.add_argument("--stim-ms", type=float, default=2000.0)
    ap.add_argument("--rebound-read-ms", type=float, default=1000.0)
    ap.add_argument("--hyper-pA", type=float, default=-200.0)
    ap.add_argument("--burst-isi-ms", type=float, default=50.0)
    ap.add_argument("--v0-offset-mV", type=float, default=1.0)
    ap.add_argument("--out", default="research/findings/raw/_bg_live_seam/part_c")
    args = ap.parse_args()

    global BURST_ISI_MS
    BURST_ISI_MS = args.burst_isi_ms

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for name in args.presets:
        res = run_preset(
            name, seed=args.seed, n_reps=args.n_reps,
            currents=args.currents, hyper_pA=args.hyper_pA,
            stim_ms=args.stim_ms, settle_ms=args.settle_ms,
            rebound_read_ms=args.rebound_read_ms,
            v0_offset_mV=args.v0_offset_mV,
        )
        path = out_root / f"{name}.json"
        path.write_text(json.dumps(res, indent=2))
        results.append((name, path))
        ref = res["reference"]["range_hz"]
        print(f"{name}: written {path} (reference {ref})", flush=True)

    print(f"Part C done: {len(results)} presets, "
          f"{time.time() - t0:.1f}s wall", flush=True)


if __name__ == "__main__":
    main()
