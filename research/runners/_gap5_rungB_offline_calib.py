"""gap#5 RUNG-B OFFLINE detector calibration (2026-08-25). Loads the F-dumps written by
`_gap5_dg_detonator_ignition_derisk --dump-traces` (the GO / no-detonator / reverse-detonation rest-phase firing
matrices) and RE-SCORES the event detector OFFLINE across settings -- NO GPU rebuild. This is the RUNG-B instrument:
it finds the detector configuration that COUNTS the transient discrete single-assembly bursts of the fixed forward
store while keeping the no-detonator control SILENT (the anti-cheat), and re-scores the reverse-detonation control.

Usage:
  .venv/bin/python -m research.runners._gap5_rungB_offline_calib research/findings/raw/gap5_r4/rungA_traces
It prints, per seed and per detector setting (SUM=shipped default, MEAN=transient RUNG-B) x an ev_floor sweep:
  - GO readout: n_events, per_asm_active (does a1 activate?), member_frac vs random_frac, forward/reverse frac
  - NO-DETONATOR: must stay SILENT (n_events small, assembly_rest low) under the SAME setting
  - REVERSE-DET: upstream reactivation (must be ~0 for a store-driven forward order)
"""
from __future__ import annotations
import sys, os, glob, json
import numpy as np
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._gap5_spontaneous_reactivation_derisk import _detect_events
from research.runners._gap5_sequence_replay_derisk import _detect_sequence_events


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    T, ncols = int(d["F_shape"][0]), int(d["F_shape"][1])
    def _unpack(key):
        return np.unpackbits(d[key], axis=1)[:, :ncols].astype(bool)
    n_mem = int(d["n_mem"][0])
    al = [d[f"asm{i}"].astype(np.int64) for i in range(n_mem)]
    return dict(F_go=_unpack("F_go"), F_nd=_unpack("F_nd"),
                F_rev=(_unpack("F_rev") if "F_rev" in d.files else None), al=al, n_mem=n_mem)


def _score_one(F, al, seed, aidx, ev_floor, mean_smooth):
    a_other = al[(aidx + 1) % len(al)] if len(al) > 1 else None
    ev = _detect_events(F, al[aidx], seed, other_local=a_other, W=5, ev_floor=ev_floor, ev_k=4.0,
                        min_frac=0.30, ev_mean_smooth=mean_smooth)
    seq = _detect_sequence_events(F, al, W=5, ev_floor=ev_floor, ev_k=4.0, active_frac=0.12, onset_frac=0.08,
                                  ev_mean_smooth=mean_smooth)
    return ev, seq


def main():
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "research/findings/raw/gap5_r4/rungA_traces"
    floors = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5, 0.3, 0.2, 0.15, 0.1]
    files = sorted(glob.glob(os.path.join(trace_dir, "seed*_traces.npz")))
    if not files:
        print(f"no trace dumps in {trace_dir}"); return 1
    print(f"[rungB-calib] {len(files)} seed dumps in {trace_dir}\n")
    for f in files:
        seed = int(os.path.basename(f).split("seed")[1].split("_")[0])
        D = _load(f); al = D["al"]; last = D["n_mem"] - 1
        print(f"==== seed {seed}  (n_mem={D['n_mem']}, asize={len(al[0])}, T={D['F_go'].shape[0]}) ====")
        for mean_smooth in (False, True):
            tag = "MEAN(transient)" if mean_smooth else "SUM(default) "
            for fl in floors:
                gev, gseq = _score_one(D["F_go"], al, seed, 0, fl, mean_smooth)
                nev, _ = _score_one(D["F_nd"], al, seed, 0, fl, mean_smooth)
                rev_up = "-"
                if D["F_rev"] is not None:
                    _re, rseq = _score_one(D["F_rev"], al, seed, last, fl, mean_smooth)
                    rev_up = int(sum(rseq["per_asm_active"][:last]))
                print(f"  {tag} ev_floor={fl:<4}: GO n_ev={gev['n_events']:<2} per_asm={gseq['per_asm_active']} "
                      f"memb={gev['member_frac']:.3f} rand={gev['random_frac']:.3f} "
                      f"FWD={gseq['forward_frac']:.2f} REV={gseq['reverse_frac']:.2f} nmulti={gseq['n_multi']} "
                      f"| NO-DET n_ev={nev['n_events']:<2} arest={nev['assembly_rest_frac']:.3f} "
                      f"| REVdet upstream={rev_up}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
