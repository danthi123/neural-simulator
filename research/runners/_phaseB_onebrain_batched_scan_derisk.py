"""ROADMAP PHASE 2, A5 LEVER 1 -- the BATCHED SCAN over the persistent store (the cheapest speed lever; composer-layer,
NO sim/ edit). The OneBrainComposer query today is reconstruct-PER-BLOCK: for each of K stored blocks, one resonate to
reconstruct + one to unbind/cleanup = O(K) x the 208-step resonate (the 5.6x gap vs the rf path, latency probe). The
batched scan reads ALL K blocks in 3 resonate windows TOTAL: fire ALL K triggers at once -> the K readout blocks
reconstruct IN PARALLEL (the validated per-block isolation -- an unused/other block stays 0, zero cross-talk) -> a
BLOCK-DIAGONAL unbind (each block's 3 roles, tiled) -> a block-diagonal cleanup -> read all K x 3 role words. O(K) ->
O(1) resonate windows = ~K x fewer launch-bound steps (the resonate is ~98% of an op's cost).

This de-risk stores K facts (host-encoded composites, to isolate the SCAN) and reads them BOTH ways: the per-block loop
(the validated `_read_block` mechanism) and the batched scan. GATE (3 seeds x 2 D): the batched read == the per-block
read == ground truth for every fact's every role (answer-identical), AND the batched scan is faster (report the
speedup). If GO, integrate into the OneBrainComposer behind a flag (the per-block path stays the correctness oracle).
Reuse-by-import (RFPhasorComposer + _build_rf_bridge); NO sim/ edit. GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_batched_scan_derisk --seeds 42,43,44 --k 8
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

from sim.backend import to_host  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, _build_rf_bridge  # noqa: E402

AG = ["dog", "cat", "bird", "river", "apple", "tree", "sun", "moon"]
AC = ["go", "come", "look", "stop", "swim", "walk", "run", "jump"]
PA = ["north", "east", "south", "west", "home", "hill", "lake", "sky"]
VOCAB = AG + AC + PA
ROLES3 = ["agent", "action", "patient"]


class BatchedStore:
    """K 3-role facts tiled into ONE bridge's complex weights; read per-block OR all-at-once (batched)."""

    def __init__(self, comp, k):
        self.comp = comp; self.D = comp.D; self.V = len(VOCAB); self.k = k
        D = self.D
        self.block = 1 + D
        self.store_base = 0
        self.q_base = self.store_base + k * self.block       # K*3 Q registers (block i, role r) at q_base + (i*3+r)*D
        self.c_base = self.q_base + k * 3 * D                 # K*3 V-concept blocks
        self.n_total = self.c_base + k * 3 * self.V
        self.store_conns = []

    def build(self, facts, seed):
        self.b = _build_rf_bridge(self.n_total, seed)
        for i, fact in enumerate(facts):
            comp_phases = self.comp._encode({ROLES3[r]: fact[r] for r in range(3)})   # host composite (isolates the SCAN)
            zc = self.comp._to_phasor(comp_phases)
            trig = self.store_base + i * self.block
            self.store_conns += [(trig + 1 + d, trig, complex(zc[d])) for d in range(self.D)]
        return self

    def _cleanup_codebook_conns(self, q_reg, c_block):
        """conj-codebook synapses from one Q register (q_reg block) to one V-concept block (c_block)."""
        D, V = self.D, self.V
        out = []
        for j in range(V):
            cc = np.conj(self.comp._to_phasor(self.comp.concepts[VOCAB[j]]))
            out += [(self.c_base + c_block * V + j, self.q_base + q_reg * D + d, complex(cc[d])) for d in range(D)]
        return out

    def read_per_block(self):
        """The validated per-block loop: for each block, reconstruct + unbind 3 roles (parallel) + cleanup. Uses only
        the first 3 Q registers + first 3 concept blocks (reused per block). Returns list of (a,v,p)."""
        comp, b, D, Pd, V = self.comp, self.b, self.D, self.comp.period, self.V
        out = []
        for i in range(self.k):
            b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
            trig = self.store_base + i * self.block
            kick = np.zeros(self.n_total, dtype=np.complex128); kick[trig] = 1.0
            b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0); b.rf_resonate_steps(Pd + 8)
            unbind = []
            for r in range(3):
                zc = np.conj(comp._to_phasor(comp.roles[ROLES3[r]]))
                unbind += [(self.q_base + r * D + d, trig + 1 + d, complex(zc[d])) for d in range(D)]
            b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
            clean = []
            for r in range(3):
                clean += self._cleanup_codebook_conns(r, r)
            b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
            mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
            row = [VOCAB[int(np.argmax(np.maximum(mem[self.c_base + r * V:self.c_base + (r + 1) * V], 0.0)))]
                   for r in range(3)]
            out.append(tuple(row))
        return out

    def read_batched(self):
        """All K blocks in 3 resonate windows: fire ALL triggers -> reconstruct all; block-diagonal unbind (K*3) ->
        block-diagonal cleanup (K*3) -> read all K*3. Returns list of (a,v,p)."""
        comp, b, D, Pd, V = self.comp, self.b, self.D, self.comp.period, self.V
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        kick = np.zeros(self.n_total, dtype=np.complex128)
        for i in range(self.k):
            kick[self.store_base + i * self.block] = 1.0                         # fire EVERY trigger
        b.rf_set_complex_weights(self.store_conns); b.rf_kick(kick, period=Pd, lam=0.0); b.rf_resonate_steps(Pd + 8)
        unbind = []
        for i in range(self.k):
            trig = self.store_base + i * self.block
            for r in range(3):
                zc = np.conj(comp._to_phasor(comp.roles[ROLES3[r]]))
                qreg = i * 3 + r
                unbind += [(self.q_base + qreg * D + d, trig + 1 + d, complex(zc[d])) for d in range(D)]
        b.rf_set_complex_weights(unbind); b.rf_resonate_steps(Pd + 8)
        clean = []
        for i in range(self.k):
            for r in range(3):
                qreg = i * 3 + r
                clean += self._cleanup_codebook_conns(qreg, qreg)
        b.rf_set_complex_weights(clean); b.rf_resonate_steps(1)
        mem = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)
        out = []
        for i in range(self.k):
            row = [VOCAB[int(np.argmax(np.maximum(
                mem[self.c_base + (i * 3 + r) * V:self.c_base + (i * 3 + r + 1) * V], 0.0)))] for r in range(3)]
            out.append(tuple(row))
        return out


def run_seed(seed, D, k):
    comp = RFPhasorComposer(seed=seed, D=D, vocab=VOCAB, period=200)
    facts = [(AG[i], AC[i], PA[i]) for i in range(k)]
    store = BatchedStore(comp, k).build(facts, seed)
    t0 = time.time(); per = store.read_per_block(); t_per = time.time() - t0
    t0 = time.time(); bat = store.read_batched(); t_bat = time.time() - t0
    truth = facts
    per_ok = sum(int(per[i] == truth[i]) for i in range(k)) / k
    bat_ok = sum(int(bat[i] == truth[i]) for i in range(k)) / k
    ident = sum(int(per[i] == bat[i]) for i in range(k)) / k
    row = {"seed": seed, "D": D, "k": k, "per_truth": per_ok, "bat_truth": bat_ok, "identical": ident,
           "ms_per": 1000 * t_per / k, "ms_bat": 1000 * t_bat / k, "speedup": t_per / max(t_bat, 1e-6)}
    print(f"  [seed {seed} D={D} K={k}] per-block=={per_ok:.2f} batched=={bat_ok:.2f} truth | identical={ident:.2f} | "
          f"{1000*t_per/k:.0f} -> {1000*t_bat/k:.0f} ms/fact ({t_per/max(t_bat,1e-6):.1f}x faster)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44"); ap.add_argument("--dims", type=str, default="64,128")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_onebrain_batched_scan.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]; dims = [int(d) for d in args.dims.split(",")]; k = min(args.k, len(AG))
    t0 = time.time()
    print(f"[A5 lever 1: batched scan] does reading ALL K={k} blocks in 3 resonate windows == the per-block loop "
          f"(answer-identical) + faster?\n", flush=True)
    rows = [run_seed(s, D, k) for s in seeds for D in dims]

    def m(key):
        return float(np.mean([r[key] for r in rows]))
    pt, bt, idn, sp = m("per_truth"), m("bat_truth"), m("identical"), m("speedup")
    n_ok = sum(int(r["bat_truth"] >= 0.99 and r["identical"] >= 0.99) for r in rows)
    go = (n_ok == len(rows)) and (sp > 1.5)
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN ({len(rows)} seed*D, K={k}): per-block {pt:.3f} / batched {bt:.3f} truth | identical {idn:.3f} | "
          f"speedup {sp:.1f}x | per-row full {n_ok}/{len(rows)}", flush=True)
    if go:
        print(f"  GO: the batched scan reads all K blocks in 3 resonate windows == the per-block loop == ground truth "
              f"(answer-identical), {sp:.1f}x faster. ==> A5 lever 1 works; integrate into OneBrainComposer behind a "
              f"flag (per-block stays the correctness oracle), re-measure the onebrain-vs-rf gap.", flush=True)
    elif idn >= 0.99:
        print(f"  PARTIAL: answer-identical ({idn:.3f}) but speedup {sp:.1f}x < 1.5x -- the bigger batched CSR build "
              f"offsets the fewer resonate windows; the win needs the masked-megakernel (lever 3) too. Reportable.", flush=True)
    else:
        print(f"  NEGATIVE: batched != per-block (identical {idn:.3f}) -- firing all K triggers cross-talks the parallel "
              f"reconstruct; the batched scan needs a per-block settle micro-schedule. The per-block scan stays. Reportable.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}", flush=True)
    out = {"verdict": "GO" if go else ("PARTIAL" if idn >= 0.99 else "NEGATIVE"), "seeds": seeds, "dims": dims, "k": k,
           "per_truth": pt, "bat_truth": bt, "identical": idn, "speedup": sp, "per_row": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
