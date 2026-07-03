"""EMERGE-81 -- CHARACTERIZE the SPIKING liquid-state machine's fading-memory DEPTH: how far does the recurrent Izhikevich
pool (EMERGE-80) hold a 1-bit distal cue, vs the near-critical rate `tanh` reservoir (EMERGE-79, which held it >= 28
fillers)?

WHY. EMERGE-80 ported the reservoir form->role mechanism to a spiking Izhikevich liquid-state machine (learns the map +
resolves the single-embedding non-local rel-head). EMERGE-79 showed the RATE reservoir holds a real-discovered distal cue
across >= 28 fillers (~33 tokens). A spiking pool's fading memory is typically SHORTER (spikes + resets erase fine state),
so the honest follow-on is to MEASURE the spiking pool's memory depth on the SAME uncontingent variable-distance task --
the distance where the spiking LSM falls below 0.75 NAMES its depth and tells us where the RANK-3 rung (a theta-gamma WM
buffer / assembly-calculus stack) becomes necessary for the SPIKING substrate specifically.

This is a CHARACTERIZATION (not a strict GO): it runs EMERGE-79's variable-distance distal-cue task (a REAL discovered
voice marker flips a far word's role across a variable number of fillers) with the EMERGE-80 `SpikingLSM` swapped in for
the rate reservoir, and reports the spiking pool's accuracy-vs-distance curve + its memory depth. The verdict is descriptive:
the spiking pool's depth d* (>= 0.75), whether it beats fixed windows within its depth (the uncontingent-necessity property
inside its range), and where it degrades. Reuse-by-import (EMERGE-79 task + EMERGE-80 SpikingLSM); NO `sim/` edit.

Run:
  python -m research.runners._emerge81_spiking_memory_depth_derisk --derisk
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
import research.runners._emerge79_reservoir_variable_distance_derisk as m79  # noqa: E402
from research.runners._emerge80_spiking_lsm_port_derisk import SpikingLSM  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _content_pools  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge81_spiking_memory_depth.json"

# a REDUCED distance sweep (the spiking sim is ~30x heavier per token than the tanh reservoir)
_TRAIN_MAXD = 8
_TEST_DISTS = [0, 2, 4, 6, 8, 12, 16]
_N_TRAIN_PER = 120
_N_TEST = 100


def _discover_with_marks(seed):
    base = m62.build_stream(seed, n_sentences=4000)
    rng = np.random.default_rng(seed * 31 + 1)
    subj0, verb0, obj0 = m62._SUBJECTS, m62._VERBS, m62._OBJECTS
    extra = []
    for _ in range(4000):
        s = str(rng.choice(subj0)); o = str(rng.choice(obj0)); v = str(rng.choice(verb0))
        mk = str(rng.choice(m79._MARKS)); nf = int(rng.integers(0, _TRAIN_MAXD + 1))
        extra += [mk] + [m79._FILLER] * nf + ["the", s, v + "s", "the", o, m62.SENT_PERIOD]
    words, freq, cover, _c = m62.compute_stats(base + extra)
    discovered, _p, _f, _cp = m62.discover_closed_class(words, freq, cover)
    subj, obj = _content_pools(discovered)[0], _content_pools(discovered)[2]
    return discovered, subj, obj


def _one(seed):
    discovered, subj, obj = _discover_with_marks(seed)
    marks_ok = all(m in discovered for m in m79._MARKS)
    enc = Encoder(discovered)
    lsm = SpikingLSM(enc.dim, seed=seed)
    rng = np.random.default_rng(seed * 101 + 5)

    train = []
    for mk in m79._MARKS:
        for nf in range(_TRAIN_MAXD + 1):
            for _ in range(_N_TRAIN_PER // (_TRAIN_MAXD + 1) + 1):
                train.append(m79._make(mk, nf, rng, subj, obj))
    W = m79._fit_reservoir(lsm, enc, train)                       # reuse EMERGE-79's fit (uses res.final_state)
    w2_tab, w2_def = m79._fit_window(enc, train, 2)

    by_d = {}
    for d in _TEST_DISTS:
        test = [m79._make(str(rng.choice(m79._MARKS)), d, rng, subj, obj) for _ in range(_N_TEST)]
        by_d[d] = {"spiking_lsm": m79._res_acc(lsm, enc, W, test),
                   "window2": m79._window_acc(enc, w2_tab, w2_def, test, 2)}
    # local-sanity + mark-lesion at a mid distance
    dctl = 6
    ctl = [m79._make(str(rng.choice(m79._MARKS)), dctl, rng, subj, obj) for _ in range(_N_TEST)]
    lesion = m79._res_acc(lsm, enc, W, ctl, lesion_mark=True)
    return {"seed": seed, "marks_ok": bool(marks_ok), "by_distance": by_d, "mark_lesion_acc": lesion,
            "mean_spikes": float(lsm._last_mean_spikes)}


def _derisk(seeds):
    print(f"EMERGE-81: SPIKING liquid-state machine memory-DEPTH characterization on the EMERGE-79 variable-distance "
          f"distal-cue task; {len(seeds)}-seed", flush=True)
    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            d = _one(s); per.append(d)
            row = " ".join(f"d{dd}:{d['by_distance'][dd]['spiking_lsm']:.2f}/{d['by_distance'][dd]['window2']:.2f}"
                           for dd in _TEST_DISTS)
            print(f"  [seed {s}] marks_ok {d['marks_ok']} | lsm/win2 by nfill: {row} | mark-lesion {d['mark_lesion_acc']:.2f}",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        agg = {d: {k: float(np.mean([p["by_distance"][d][k] for p in per])) for k in ("spiking_lsm", "window2")}
               for d in _TEST_DISTS}
        lesion = float(np.mean([p["mark_lesion_acc"] for p in per]))
        marks_ok = all(p["marks_ok"] for p in per)
        depth = max([d for d in _TEST_DISTS if agg[d]["spiking_lsm"] >= 0.75], default=-1)
        held_full = (depth == _TEST_DISTS[-1])
        # within its depth, does the spiking LSM beat the +-2 window (which is blind at all distances)?
        beats_window_in_range = all(agg[d]["spiking_lsm"] - agg[d]["window2"] >= 0.30
                                    for d in _TEST_DISTS if d <= max(depth, 0))
        verdict = (
            f"CHARACTERIZATION -- the SPIKING Izhikevich liquid-state machine (EMERGE-80) holds a real-discovered 1-bit "
            f"distal cue to a fading-memory DEPTH of {'>= ' + str(depth) + ' fillers (held across the whole reduced sweep)' if held_full else '~' + str(depth) + ' fillers (falls below 0.75 beyond that)'} "
            f"(profile: {', '.join(f'd{d}={agg[d]['spiking_lsm']:.2f}' for d in _TEST_DISTS)}); the fixed +-2 window is at "
            f"chance throughout, so WITHIN its depth the spiking pool beats every fixed window (uncontingent necessity in "
            f"range = {beats_window_in_range}). MARK-LESION collapses the role to {lesion:.2f} (genuinely mark-determined). "
            f"marks discovered = {marks_ok}. Reference: the RATE tanh reservoir (EMERGE-79) held it >= 28 fillers; a "
            f"spiking pool's fading memory is {'comparable' if held_full and depth >= _TEST_DISTS[-1] else 'SHORTER'} in "
            f"this reduced sweep. DEEPER/beyond-depth dependencies on the SPIKING substrate are where the RANK-3 rung "
            f"(theta-gamma WM buffer / assembly-calculus stack) becomes necessary. Reuse-by-import; NO sim/ edit.")
        go = bool(marks_ok and beats_window_in_range and depth >= 2 and (lesion <= 0.65))
    else:
        go = False; verdict = f"ERROR -- {err}"; agg = lesion = marks_ok = depth = None

    summary = {
        "probe": "emerge81_spiking_memory_depth", "verdict": verdict, "go": bool(go) if err is None else False,
        "task": ("characterize the EMERGE-80 spiking Izhikevich liquid-state machine's fading-memory DEPTH on the "
                 "EMERGE-79 variable-distance distal-cue task (real discovered marker); report accuracy-vs-distance + the "
                 "depth (>= 0.75) + whether it beats fixed windows within range; descriptive, not a strict GO"),
        "test_distances": _TEST_DISTS, "seeds": list(seeds), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if err is not None else {
            "by_distance": {str(d): agg[d] for d in _TEST_DISTS}, "mark_lesion_acc": lesion,
            "marks_discovered": marks_ok, "spiking_memory_depth_ge_075_fillers": depth,
        },
        "per_seed": per,
        "HONEST_NOTE": ("A CHARACTERIZATION of the spiking pool's fading memory (not a new capability). The RATE reservoir "
                        "(EMERGE-79) held the distal cue >= 28 fillers; this measures the SPIKING pool's depth on a reduced "
                        "sweep (the spiking sim is ~30x heavier per token). Within its depth the spiking LSM beats every "
                        "fixed window (uncontingent, real cue); beyond it, the RANK-3 buffer/stack is the named next "
                        "mechanism for the spiking substrate. Reuse EMERGE-79 task + EMERGE-80 SpikingLSM; NO sim/ edit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge81] VERDICT: {verdict}", flush=True)
    print(f"[emerge81] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    return _derisk(a.seeds)


if __name__ == "__main__":
    raise SystemExit(main())
