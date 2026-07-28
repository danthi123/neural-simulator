"""SPIKING-STP realization of the novel-referent fast-weight binder — the cheap-first MECHANISM de-risk (numpy STP
dynamics, before the full SimulationBridge port). RUNG 6c (`2026-07-13-RUNG6c-...GO.md`) validated the binder with a
generic Hebbian outer-product fast weight; the fully-spiking rung realizes it in short-term plasticity (Mongillo 2008
synaptic-WM: facilitation `u` decaying with `tau_f` = the `sim` STP `cp_stp_u`). THE MECHANISM QUESTION this resolves:
Tsodyks-Markram STP facilitation is PRESYNAPTIC (`u` rises wherever the PREsynaptic barcode fired, regardless of which
slot won) — can that do SELECTIVE content-addressable binding (bind barcode_e→ITS slot, not all slots)? Or does the bind
need HEBBIAN (pre×post) short-term potentiation (`u` rises only on the WTA-winner slot)?

MODEL: per-synapse facilitation `u[slot, barcode_bit]`; drive(slot) = Σ_bit base_w·u[slot,bit]·barcode[bit]; WTA winner =
argmax_slot drive (a fresh/unfacilitated slot wins for a NOVEL barcode via a small free-slot prior). Two update rules:
  - `presynaptic` (TM-faithful): u[:, active_bits] += U·(1-u)  -> rises on ALL slots (predicted NON-selective).
  - `hebbian` (short-term potentiation): u[winner, active_bits] += U·(1-u) -> rises only on the winner (predicted select).
Between clauses u decays exp(-Δ/tau_f) (facilitation fade). Then re-express the possession narrative in slot space + the
validated discrete-attractor tracks the holder (reuse RUNG6c metric: held-out NOVEL entities, entity-level deref,
merge/no-bind lesions).

GO (for whichever rule binds): novel entity-track == the attractor slot-ceiling (binding-penalty ~0), collisions ~0,
merge/no-bind collapse, >=3 seeds. The comparison IS the finding — which STP variant the spiking bridge must use.
numpy-CPU; reuse-by-import (`_novel_referent_hebbian_fastweight_derisk` task + metric); NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._stp_facilitation_binder_derisk --seed 42 --rule presynaptic
     SIM_BACKEND=numpy python -m research.runners._stp_facilitation_binder_derisk --seeds 42 43 44 --rule hebbian
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time

import numpy as np

import research.runners._novel_referent_hebbian_fastweight_derisk as NR
from research.runners._d3_group_composition_derisk import discrete_attractor_rnn

_K = NR._K; _DIM = NR._DIM


class STPBinder:
    """Content-addressable binder realized in short-term-plasticity facilitation `u[slot, bit]` (Mongillo `cp_stp_u`)."""
    def __init__(self, rule="hebbian", U=0.6, tau_f=1.5, clause_dt=0.15, base_w=1.0, theta=0.5):
        self.u = np.zeros((_K, _DIM), np.float32)             # facilitation per (slot, barcode-input-synapse)
        self.rule = rule; self.U = U; self.decay = float(np.exp(-clause_dt / tau_f))
        self.base_w = base_w; self.theta = theta; self.free = 0

    def slot(self, code, no_bind_rng=None):
        if no_bind_rng is not None:
            return int(no_bind_rng.integers(_K))
        active = code > 0
        # NORMALIZED (cosine) facilitation match: content-addressable memory reads by SIMILARITY, not raw magnitude. This
        # is scale-INVARIANT -> robust to BOTH facilitation DECAY (scales u[s] uniformly -> cosine unchanged) AND
        # overlapping-barcode cross-facilitation (a novel code sharing m of k bits gives cosine ~m/k << 1 real re-mention).
        unorm = np.sqrt((self.u ** 2).sum(1)) + 1e-9
        drive = (self.u[:, active].sum(1)) / (unorm * np.sqrt(active.sum()))
        if self.free > 0 and float(drive.max()) > self.theta:
            s = int(np.argmax(drive))
        else:
            s = min(self.free, _K - 1); self.free = min(self.free + 1, _K)
        if self.rule == "presynaptic":                        # TM: u rises wherever the PRE (barcode) fired -> all slots
            self.u[:, active] += self.U * (1.0 - self.u[:, active])
        else:                                                 # hebbian: u rises only on the WTA-winner slot (pre×post)
            self.u[s, active] += self.U * (1.0 - self.u[s, active])
        return s

    def step(self):
        self.u *= self.decay                                  # facilitation fade between clauses


def _to_slot_task_stp(items, codes, rule, no_bind_rng=None, theta=3.0):
    N = len(items); Lmax = max(L for _, _, L in items)
    X = np.zeros((N, Lmax, 2 * _K), np.float32); STATE = np.zeros((N, Lmax), np.int64)
    SEQ = np.full((N, Lmax), -1, np.int64); L = np.zeros(N, np.int64); Y = np.zeros(N, np.int64)
    maps = []
    for n, (pairs, hseq, L_) in enumerate(items):
        b = STPBinder(rule=rule, theta=theta); e2s = {}
        for t, (a, bb) in enumerate(pairs):
            sa = b.slot(codes[a], no_bind_rng); sb = b.slot(codes[bb], no_bind_rng)
            e2s[a] = sa; e2s[bb] = sb
            X[n, t, sa] = 1.0; X[n, t, _K + sb] = 1.0
            STATE[n, t] = e2s.get(hseq[t], 0); SEQ[n, t] = sa * _K + sb
            b.step()
        L[n] = L_; maps.append(e2s)
    return {"train": (X, Y, L, SEQ, STATE), "test_same": (X, Y, L, SEQ, STATE),
            "test_deeper": (X, Y, L, SEQ, STATE), "K": _K, "ident": 0, "n_pool": 2 * _K,
            "color": np.zeros(_K, np.int64), "p_transfer": 0.6}, maps


def _track(tr_items, te_items, codes, rule, seed, n_hid=160, epochs=70, no_bind_seed=None, theta=3.0):
    nb_tr = np.random.default_rng(no_bind_seed) if no_bind_seed is not None else None
    nb_te = np.random.default_rng(no_bind_seed + 1) if no_bind_seed is not None else None
    tr, _ = _to_slot_task_stp(tr_items, codes, rule, nb_tr, theta=theta)
    te, te_maps = _to_slot_task_stp(te_items, codes, rule, nb_te, theta=theta)
    task = {**tr, "test_deeper": te["test_deeper"], "test_same": te["test_deeper"]}
    r = discrete_attractor_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs)
    Xe, Ye, Le, SEQe, STe = te["test_deeper"]
    ent = NR._entity_acc(NR._final_slots(r["weights"], Xe, Le), te_items, te_maps)
    coll = float(np.mean([len(set(mp.values())) < len(mp) for mp in te_maps]))
    return ent, float(r["state_deeper"]), coll


def run(seed, rule, n_per_len=1200, theta=3.0):
    rng = np.random.default_rng(seed)
    codes = NR._mint_codes(rng, 12)
    tr = NR._narratives(rng, list(range(6)), NR._LENS_TR, n_per_len)
    te = NR._narratives(rng, list(range(6, 12)), NR._LENS_TE, max(300, n_per_len // 4))
    novel, slot, coll = _track(tr, te, codes, rule, seed, theta=theta)
    codesm = NR._mint_codes(np.random.default_rng(seed + 1), 12, merge=True)
    merge, _, _ = _track(tr, te, codesm, rule, seed, theta=theta)
    nobind, _, _ = _track(tr, te, codes, rule, seed, no_bind_seed=seed + 30, theta=theta)
    te_task, te_maps = _to_slot_task_stp(te, codes, rule, theta=theta)
    Le = te_task["test_deeper"][2]
    rt = NR._entity_acc(np.zeros(len(Le), np.int64), te, te_maps)
    pen = slot - novel
    go = (pen < 0.05) and (novel > rt + 0.15) and (merge < 0.35) and (nobind < 0.35) and (coll < 0.05)
    print(f"[stp-binder seed={seed} rule={rule}] novel={novel:.3f} (slot-ceil={slot:.3f} pen={pen:+.3f} "
          f"coll={coll:.3f}) merge={merge:.3f} no-bind={nobind:.3f} retention={rt:.3f} -> {'GO' if go else 'no'}")
    return dict(seed=seed, rule=rule, novel=round(novel, 3), slot_ceiling=round(slot, 3),
                binding_penalty=round(pen, 3), collisions=round(coll, 3), merge=round(merge, 3),
                nobind=round(nobind, 3), retention=round(rt, 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--rule", choices=["presynaptic", "hebbian"], default="hebbian")
    ap.add_argument("--theta", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    t0 = time.time()
    results = [run(s, a.rule, theta=a.theta) for s in seeds]
    if len(results) > 1:
        print(f"[stp-binder rule={a.rule}] {sum(1 for r in results if r['go'])}/{len(results)} seeds GO")
    if a.out:
        json.dump(dict(results=results, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
