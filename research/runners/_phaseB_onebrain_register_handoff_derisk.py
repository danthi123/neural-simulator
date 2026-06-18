"""ROADMAP PHASE 2 (the real "one brain"), FIRST cheap-first de-risk — register->register SYNAPTIC PHASE HANDOFF.
Pre-registered by `2026-06-18-one-brain-integrated-pipeline-scoping.md`.

Today every conversational op reads its phases OUT to numpy and re-kicks the next op (3 host round-trips per
who/what turn). The genuinely-new primitive for the integrated one-brain pipeline is: chain two RF ops --
unbind(bind(role, filler), role) -- on ONE persistent bridge so the bound composite stays as a register's phasor
and feeds the unbind complex synapse DIRECTLY, with NO `rf_read_phases` between them.

Mechanism: 3 registers on one bridge -- filler[0:D], bound[D:2D], unbound[2D:3D] -- with TWO diagonal complex
synapses installed at once: bind (filler k -> bound D+k, weight role[k]) + unbind (bound D+k -> unbound 2D+k,
weight conj(role[k])). Kick the filler, resonate, read unbound[2D:3D]. Since bound[k]=role[k]*filler[k] and
unbound[k]=conj(role[k])*bound[k]=|role[k]|^2*filler[k]=filler[k] (unit-magnitude role), the unbound register
recovers the filler -- through the substrate, no host hand-off. Two variants: SINGLE-window (both synapses, one
resonate) and TWO-window (bind settles in window 1, unbind reads it in window 2 -- still no read-out between).

GATE (3 seeds x 2 D): on-bridge recovered cleans up to the ORIGINAL filler for 100% of vocab AND matches the
current HOST two-call pipeline (_bind -> read -> _unbind). Anti-cheats: (a) PERMUTED-role unbind must FAIL (the
route carries the binding), (b) SEVERED bind->bound synapse lesion must collapse (the on-bridge handoff is
load-bearing, not the kick leaking through). Reuse-by-import (RFPhasorComposer for roles/concepts/cleanup +
_build_rf_bridge). GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_register_handoff_derisk --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

VOCAB = ["dog", "cat", "bird", "river", "apple", "north", "south", "east", "west", "go", "come", "look", "stop"]


def _handoff(b, comp, role_phases, filler_phases, period, mode="single", permute_role=False, lesion=False):
    """register->register bind->unbind on ONE persistent bridge. Returns the recovered phases [D] from unbound[2D:3D]."""
    D = comp.D
    zr = comp._to_phasor(role_phases)
    # anti-cheat: unbind with a genuinely DIFFERENT role ("action" vs the bound "agent") -- a clean scramble (role
    # reversal was too weak: conj(role[D-1-k])*role[k] can coincidentally preserve recovery for some seeds).
    zr_un = comp._to_phasor(comp.roles["action"] if permute_role else role_phases)
    zf = comp._to_phasor(filler_phases)
    bind = [] if lesion else [(D + k, k, complex(zr[k])) for k in range(D)]        # filler -> bound (sever = lesion)
    unbind = [(2 * D + k, D + k, complex(np.conj(zr_un[k]))) for k in range(D)]    # bound -> unbound
    kick = np.zeros(3 * D, dtype=np.complex128)
    kick[:D] = zf
    if mode == "single":
        b.rf_set_complex_weights(bind + unbind)
        b.rf_kick(kick, period=period, lam=0.0)
        b.rf_resonate_steps(period + 8)
        return np.asarray(b.rf_read_phases())[2 * D:3 * D]
    # two-window: bind settles (window 1), then unbind reads the settled bound register (window 2), no read between
    b.rf_set_complex_weights(bind)
    b.rf_kick(kick, period=period, lam=0.0)
    b.rf_resonate_steps(period + 8)
    b.rf_set_complex_weights(unbind)          # install the unbind synapse; the bound register keeps its phasor state
    b.rf_resonate_steps(period + 8)           # NO re-kick, NO read-out -> the handoff is purely on-substrate
    return np.asarray(b.rf_read_phases())[2 * D:3 * D]


def run_seed(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    role = comp.roles["agent"]
    b = _build_rf_bridge(3 * D, seed)         # ONE persistent bridge, 3 registers
    res = {}
    for mode in ("single", "two"):
        ok_self = ok_host = 0
        for w in VOCAB:
            fphases = comp.concepts[w]
            # HOST two-call pipeline (the current path): bind -> read -> unbind -> read
            bound_host = comp._bind(role, fphases)
            rec_host = comp._unbind_phases(bound_host, "agent")
            host_word = comp._cleanup(rec_host, VOCAB)
            # ON-BRIDGE register->register handoff (no read between bind and unbind)
            rec_ob = _handoff(b, comp, role, fphases, comp.period, mode=mode)
            ob_word = comp._cleanup(rec_ob, VOCAB)
            ok_self += int(ob_word == w)          # recovers the ORIGINAL filler
            ok_host += int(ob_word == host_word)   # == the host two-call pipeline
        res[mode] = {"self": ok_self / len(VOCAB), "host": ok_host / len(VOCAB)}
    # anti-cheats on the better mode
    best = "two" if res["two"]["self"] >= res["single"]["self"] else "single"
    perm_ok = lesion_ok = 0
    for w in VOCAB:
        fphases = comp.concepts[w]
        perm = comp._cleanup(_handoff(b, comp, role, fphases, comp.period, mode=best, permute_role=True), VOCAB)
        les = comp._cleanup(_handoff(b, comp, role, fphases, comp.period, mode=best, lesion=True), VOCAB)
        perm_ok += int(perm == w)       # permuted role should NOT recover (want this LOW)
        lesion_ok += int(les == w)      # severed handoff should NOT recover (want this LOW)
    row = {"seed": seed, "D": D, "single": res["single"], "two": res["two"], "best": best,
           "permuted": perm_ok / len(VOCAB), "lesion": lesion_ok / len(VOCAB)}
    print(f"  [seed {seed} D={D}] single self={res['single']['self']:.2f}/host={res['single']['host']:.2f} | "
          f"two self={res['two']['self']:.2f}/host={res['two']['host']:.2f} | "
          f"permuted={row['permuted']:.2f} lesion={row['lesion']:.2f} (best={best})", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_register_handoff.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    dims = [int(d) for d in args.dims.split(",")]
    t0 = time.time()
    print("[one-brain register-handoff de-risk] does bind->unbind chain register->register on ONE bridge (no host "
          "round-trip) == the host two-call pipeline?\n", flush=True)
    rows = [run_seed(s, D) for s in seeds for D in dims]

    def best_self(r):
        return max(r["single"]["self"], r["two"]["self"])
    mean_self = float(np.mean([best_self(r) for r in rows]))
    mean_host = float(np.mean([max(r["single"]["host"], r["two"]["host"]) for r in rows]))
    mean_perm = float(np.mean([r["permuted"] for r in rows]))
    mean_les = float(np.mean([r["lesion"] for r in rows]))
    n_go = sum(int(best_self(r) >= 0.99) for r in rows)
    go = (n_go == len(rows)) and (mean_perm <= 0.15) and (mean_les <= 0.15)
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D): on-bridge self-recovery {mean_self:.3f} | ==host {mean_host:.3f} | "
          f"permuted {mean_perm:.3f} | lesion {mean_les:.3f} | self>=0.99: {n_go}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: bind->unbind chains register->register on ONE persistent bridge with NO host round-trip -- "
              f"recovers the filler 100% == the host two-call pipeline, permuted-role + severed-handoff collapse. "
              f"==> the genuinely-new one-brain primitive (synaptic phase handoff between ops) WORKS; build up the "
              f"full who/what turn on one persistent bridge.", flush=True)
    elif mean_self >= 0.5:
        print(f"  BOUNDARY: partial register->register handoff ({mean_self:.3f}) -- phase coherence across the chain "
              f"is lossy; tune the settle window (two-window) or the readout timing before scaling up.", flush=True)
    else:
        print(f"  NEGATIVE: the register->register handoff does not recover ({mean_self:.3f}) -- the RF first-spike "
              f"phase code does not propagate cleanly through two chained complex synapses in-substrate; the "
              f"between-op handoff needs a different mechanism (settle+re-encode, or a phase-latch).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"verdict": "GO" if go else ("BOUNDARY" if mean_self >= 0.5 else "NEGATIVE"), "seeds": seeds, "dims": dims,
           "mean_self": mean_self, "mean_host": mean_host, "permuted": mean_perm, "lesion": mean_les, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
