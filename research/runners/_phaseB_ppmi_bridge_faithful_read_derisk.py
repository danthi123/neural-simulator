"""CYCLE 91 — on-bridge faithful-read de-risk: does the SPIKING substrate PRESERVE the PPMI structure when the
cortex code is read FAITHFULLY (the hub layer's own firing), removing the CYCLE-88 random-readout confound?

CYCLE 88 found PPMI (local normalization) reaches host (+0.50) + generalizes in numpy. The on-bridge --ppmi-drive
smoke was CONFOUNDED: the cortex-forward runner reads a RANDOM 0.1-density readout projection, which destroys the
real-corpus structure regardless of drive (the same confound the interneuron probe showed). The faithful read is
the HUB LAYER's OWN firing -- the hubs are driven directly by PPMI, so their firing rates ARE the spiking
realization of the PPMI code. This probe: drive the hubs with PPMI(concept) and read the hub firing-rate vector
as the code; cos vs host. If the spiking hub firing preserves the PPMI cosine structure, the on-bridge cortex
path is viable (the readout was the confound) and the remaining work is the neural per-concept normalization.

ARMS (real corpus, 3 seeds via single-seed smoke first):
  numpy PPMI (reference)      cos of the PPMI vectors                 ~+0.50 (the target the bridge should preserve)
  BRIDGE faithful hub-read    spiking hub firing-rate code, cos       <- the test
  BRIDGE log-drive (control)  the prior +0.155 regime (log input)     should trail PPMI-drive
GATE: bridge PPMI-drive faithful-read Pearson >= 0.70 x numpy PPMI (the spiking realization preserves it). A
PASS confirms the readout was the CYCLE-88 confound + the spiking substrate preserves PPMI -> build the neural
per-concept normalization next. A clear shortfall = the genuine spiking-realization loss (characterized).

Reuse-by-import (_build_cortex_bridge, build_real_corpus, ppmi_matrix); GPU (cupy) for the bridge. NO sim/ edits.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_ppmi_bridge_faithful_read_derisk --seeds 42
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402
from research.runners.dendritic_d1_learn_graded_structure_derisk import _cos_sim, _pearson_vs_Strue  # noqa: E402
from research.runners.dendritic_cortex_forward_codes_derisk import _build_cortex_bridge  # noqa: E402
from research.runners.learned_graded_cortex_fair_test import build_real_corpus, ppmi_matrix  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def present_read_hubs(bridge, hub_idx, drive_vec, drive_scale, window, settle):
    """Drive the hubs with drive_vec * scale; read the HUB LAYER's own firing-rate vector (the faithful code)."""
    hub_idx = np.asarray(hub_idx)
    N = int(bridge.cp_membrane_potential_v.shape[0])
    xp = bridge._cp if hasattr(bridge, "_cp") else None
    d = (np.asarray(drive_vec, dtype=np.float64) * drive_scale).astype(np.float32)
    bridge.cp_external_input_current[:] = 0.0
    if xp is not None:
        bridge.cp_external_input_current[hub_idx] = xp.asarray(d)
    else:
        bridge.cp_external_input_current[hub_idx] = d
    acc = np.zeros(hub_idx.size, dtype=np.float64)
    n = 0
    for t in range(settle + window):
        bridge._run_one_simulation_step()
        if t >= settle:
            acc += np.asarray(to_host(bridge.cp_firing_states))[hub_idx].astype(np.float64)
            n += 1
    bridge.cp_external_input_current[:] = 0.0
    return acc / max(1, n)


def read_codes(bridge, hub_idx, C_drive, drive_scale, window, settle, warmup):
    Nc = C_drive.shape[0]
    for _ in range(warmup):                                  # warm the per-hub adaptation EMA over the stream
        for i in range(Nc):
            present_read_hubs(bridge, hub_idx, C_drive[i], drive_scale, window, settle)
    codes = np.zeros((Nc, np.asarray(hub_idx).size))
    for i in range(Nc):
        codes[i] = present_read_hubs(bridge, hub_idx, C_drive[i], drive_scale, window, settle)
    return codes


def run_seed(seed, args):
    C, labels, S_true = build_real_corpus(seed, args.n_hub)
    labels = np.asarray(labels)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1), alpha=0.75), labels)
    ppmi = ppmi_matrix(C, 0.75)
    numpy_ppmi = _pearson_vs_Strue(_cos_sim(ppmi), S_true)
    log_drive = np.log1p(np.maximum(C, 0.0))
    n_hub = C.shape[1]
    print(f"\n[faithful-read seed {seed}] {C.shape[0]}c x {n_hub}h | host {host_p:+.3f} | numpy PPMI {numpy_ppmi:+.3f}",
          flush=True)

    def bridge_codes(drive):
        bridge, hub_idx, _ = _build_cortex_bridge(n_hub, 50, seed, False, 1.0, 0.002, 0.1)
        return read_codes(bridge, hub_idx, drive, args.drive_scale, args.window, args.settle, args.warmup)

    ppmi_codes = bridge_codes(ppmi)
    ppmi_p = _pearson_vs_Strue(_cos_sim(ppmi_codes), S_true)
    silent = float((ppmi_codes.sum(1) == 0).mean())
    log_codes = bridge_codes(log_drive)
    log_p = _pearson_vs_Strue(_cos_sim(log_codes), S_true)
    print(f"  BRIDGE faithful hub-read: PPMI-drive {ppmi_p:+.3f} ({ppmi_p/max(numpy_ppmi,1e-9):.0%} of numpy PPMI; "
          f"silent {silent:.2f}) | log-drive control {log_p:+.3f}", flush=True)
    return {"seed": seed, "host": host_p, "numpy_ppmi": numpy_ppmi, "bridge_ppmi": ppmi_p, "bridge_log": log_p,
            "silent": silent}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42")
    p.add_argument("--n-hub", type=int, default=500)
    p.add_argument("--drive-scale", type=float, default=120.0)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--settle", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[on-bridge faithful-read de-risk] seeds={seeds} drive_scale={args.drive_scale} -- does the spiking "
          f"hub firing PRESERVE PPMI structure (faithful read, no readout confound)?", flush=True)
    rows = [run_seed(s, args) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    host, npp, bp, bl = m("host"), m("numpy_ppmi"), m("bridge_ppmi"), m("bridge_log")
    frac = bp / npp if npp > 1e-9 else 0.0
    print(f"\n{'='*96}\n  MEAN ({len(seeds)} seeds): host {host:+.3f} | numpy PPMI {npp:+.3f} | BRIDGE faithful "
          f"PPMI-drive {bp:+.3f} ({frac:.0%} of numpy) | bridge log-drive {bl:+.3f}", flush=True)
    print(f"{'='*96}", flush=True)
    if bp >= 0.70 * npp and bp >= bl + 0.05:
        print(f"  GO: the spiking hub firing PRESERVES the PPMI structure (faithful read {bp:+.3f} = {frac:.0%} of "
              f"numpy PPMI {npp:+.3f}, beats log-drive {bl:+.3f}). ==> the CYCLE-88 null was the RANDOM-READOUT "
              f"confound, NOT a spiking-realization wall. The on-bridge cortex path is viable; build the neural "
              f"per-concept normalization next (the one missing local op). Edits approved.", flush=True)
    elif bp >= bl + 0.05:
        print(f"  PARTIAL: PPMI-drive ({bp:+.3f}) beats log-drive ({bl:+.3f}) but reaches only {frac:.0%} of numpy "
              f"PPMI -- the spiking nonlinearity (threshold/saturation of the firing-rate code) loses some "
              f"structure. Tune drive_scale/window or read a graded quantity (g_e) instead of spike rate.", flush=True)
    else:
        print(f"  NEGATIVE: faithful PPMI-drive ({bp:+.3f}) does not clearly beat log-drive ({bl:+.3f}) -- the "
              f"spiking realization loses the PPMI structure even read faithfully; characterize the loss.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"host": host, "numpy_ppmi": npp, "bridge_ppmi": bp, "bridge_log": bl, "frac_of_numpy": frac,
           "per_seed": rows, "config": vars(args)}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_ppmi_bridge_faithful_read.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
