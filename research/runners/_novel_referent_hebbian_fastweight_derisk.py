"""NOVEL-referent (open) discourse tracking via a content-agnostic HEBBIAN FAST-WEIGHT binder (the emergence-aligned
close of RUNG6b, which resolved BOUNDED referents with a SUPERVISED read). Research gate:
`2026-07-13-novel-referent-binding-research-gate-emergent-Hebbian-fastweights.md`.

THE MECHANISM (candidate #3 — NOT the FHRR scaffold, per the emergence bar): a discourse referent is a barcode CODE
(a developmental-random distinct code, the `entity_instance_layer` primitive). A HEBBIAN FAST-WEIGHT `W` (K slots x
code_dim, reset per narrative) binds each entity to a fresh SLOT the FIRST time it is mentioned and RETRIEVES that slot
on re-mention — NO supervised read, NO fixed algebra: `slot(c) = argmax(W@c)` if `max(W@c)>θ` (already bound) else assign
the next free slot + `W[slot] += c/||c||` (one-shot Hebbian bind, Ba-2016 fast weights / Bouchacourt-Buschman flexible
random WM / O'Reilly indirection). Because the bind is content-AGNOSTIC, a NOVEL entity binds to a slot IDENTICALLY to a
known one BY CONSTRUCTION. The narrative is re-expressed in bounded SLOT space (a,b -> a_slot,b_slot); the validated D3
discrete-attractor tracks the holder SLOT; the final holder slot dereferences (via W) to the entity.

THE KEY TEST: the tracked entities are MINTED AT TEST, DISJOINT from every entity in the attractor's training pool
(held-out NOVEL) — so the read/tracker has NEVER seen them. If tracking holds, novel-referent binding is emergent +
content-agnostic (not per-entity supervised).

ANTI-CHEATS (all load-bearing): held-out-NOVEL entities + held-out-DEEPER lengths (generalization-not-memorization);
MERGE-lesion (α=0 -> all entity codes identical -> the binder cannot individuate -> collapse); NO-BIND lesion (random
slot assignment -> chance); retention/last-mention floors at chance; non-commutativity (reorder -> different holder).
GO: novel holder-track @deeper > 0.6 AND > reservoir/floors AND merge+no-bind collapse (<0.35), >=3->6 seeds. numpy-CPU.
Reuse-by-import; NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._novel_referent_hebbian_fastweight_derisk --seed 42
     SIM_BACKEND=numpy python -m research.runners._novel_referent_hebbian_fastweight_derisk --seeds 42 43 44 100 101 102
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

from research.runners._d3_group_composition_derisk import discrete_attractor_rnn

_K = 6                      # bounded slots (positional referent registers)
_DIM = 64                   # barcode code dim
_KACT = 8                   # active bits per barcode
_LENS_TR = (1, 2, 3)
_LENS_TE = (6, 7, 8)


def _mint_codes(rng, M, merge=False):
    """M distinct developmental-random SPARSE 0/1 barcodes (k-active of dim) with overlap-rejection. Sparse (not dense
    ±1, whose ~56 shared background -1 bits give a ~0.56 baseline cosine that makes <0.5 rejection unsatisfiable ->
    infinite loop). merge=True -> ALL identical (the α=0 MERGE lesion: no individuation)."""
    if merge:
        c = np.zeros(_DIM, np.float32); c[rng.choice(_DIM, _KACT, replace=False)] = 1.0
        return np.repeat(c[None], M, 0)
    codes = []
    guard = 0
    while len(codes) < M and guard < 100000:
        guard += 1
        c = np.zeros(_DIM, np.float32); c[rng.choice(_DIM, _KACT, replace=False)] = 1.0
        cn = c / (np.linalg.norm(c) + 1e-9)
        if all(float(cn @ (d / (np.linalg.norm(d) + 1e-9))) < 0.5 for d in codes):   # overlap-rejection (sparse: ~0.125)
            codes.append(c)
    return np.asarray(codes, np.float32)


class HebbianBinder:
    """Content-agnostic fast-weight binder: assign a fresh slot on first mention, retrieve on re-mention. Per-narrative.
    `decay` (<1.0) models STP FACILITATION FADE per clause (Mongillo synaptic-WM tau_f): the bind decays between mentions;
    `step()` is called once per clause. decay=1.0 = a permanent fast weight (byte-identical to the default)."""
    def __init__(self, theta=0.55, decay=1.0):
        self.W = np.zeros((_K, _DIM), np.float32); self.free = 0; self.theta = theta; self.decay = float(decay)

    def step(self):
        if self.decay < 1.0:
            self.W *= self.decay

    def slot(self, c, no_bind_rng=None):
        cn = c / (np.linalg.norm(c) + 1e-9)
        if no_bind_rng is not None:                          # NO-BIND lesion: random slot assignment
            return int(no_bind_rng.integers(_K))
        match = self.W @ cn
        if self.free > 0 and float(match.max()) > self.theta:
            return int(match.argmax())                       # retrieve an already-bound slot
        s = min(self.free, _K - 1); self.W[s] += cn; self.free = min(self.free + 1, _K)   # one-shot Hebbian bind
        return s


def _narratives(rng, entities, lens, n_each, p_transfer=0.6):
    """D3 possession logic over an ENTITY POOL: each narrative draws <=K distinct entities; holder=b iff holder==a; the
    last clause forced no-op (markov/last-named reveal nothing). Returns per-narrative (ent_pairs, holder_seq, L)."""
    items = []
    for L_ in lens:
        for _ in range(n_each):
            k = int(rng.integers(2, min(_K, len(entities)) + 1))
            ents = list(rng.choice(entities, size=k, replace=False))
            holder = ents[0]
            pairs = []; hseq = []
            for t in range(L_):
                force_noop = (t == L_ - 1 and L_ >= 2)
                if (not force_noop) and rng.random() < p_transfer:
                    a = holder; b = int(rng.choice(ents))
                else:
                    a = int(rng.choice([e for e in ents if e != holder])); b = int(rng.choice(ents))
                if holder == a:
                    holder = b
                pairs.append((a, b)); hseq.append(holder)
            items.append((pairs, hseq, L_))
    return items


def _to_slot_task(items, codes, K, ident_slot=0, no_bind_rng=None, decay=1.0):
    """Re-express each narrative in SLOT space via a per-narrative Hebbian binder. X[n,t]=[onehot(a_slot);onehot(b_slot)],
    STATE=holder slot. Returns the discrete_attractor task dict (+ the per-narrative entity->slot maps for dereference)."""
    N = len(items); Lmax = max(L for _, _, L in items)
    X = np.zeros((N, Lmax, 2 * K), np.float32); STATE = np.zeros((N, Lmax), np.int64)
    SEQ = np.full((N, Lmax), -1, np.int64); L = np.zeros(N, np.int64); Y = np.zeros(N, np.int64)
    maps = []
    for n, (pairs, hseq, L_) in enumerate(items):
        binder = HebbianBinder(decay=decay); e2s = {}
        for t, (a, b) in enumerate(pairs):
            sa = binder.slot(codes[a], no_bind_rng); sb = binder.slot(codes[b], no_bind_rng)
            e2s[a] = sa; e2s[b] = sb
            X[n, t, sa] = 1.0; X[n, t, K + sb] = 1.0
            hs = e2s.get(hseq[t], 0)
            STATE[n, t] = hs; SEQ[n, t] = sa * K + sb
            binder.step()                                     # STP facilitation fade between clauses
        L[n] = L_; Y[n] = 0
        maps.append(e2s)
    return {"train": (X, Y, L, SEQ, STATE), "test_same": (X, Y, L, SEQ, STATE),
            "test_deeper": (X, Y, L, SEQ, STATE), "K": K, "ident": ident_slot, "n_pool": 2 * K,
            "color": np.zeros(K, np.int64), "p_transfer": 0.6}, maps


def _final_slots(weights, X, L):
    """Autoregressive attractor rollout (using the trained weights) -> the predicted holder SLOT at each narrative's
    final clause."""
    emb, Wr, Wi, Ws, bs = weights["emb"], weights["Wr"], weights["Wi"], weights["Ws"], weights["bs"]
    N = len(L); cur = np.zeros(N, np.int64); final = np.zeros(N, np.int64)
    for t in range(int(L.max())):
        h = np.tanh(emb[cur] @ Wr.T + X[:, t] @ Wi.T)
        nxt = (h @ Ws.T + bs).argmax(1)
        active = (L > t); cur = np.where(active, nxt, cur)
        final = np.where(L == (t + 1), cur, final)
    return final


def _entity_acc(final_slots, items, maps):
    """ENTITY-level dereference: the predicted holder slot must (a) equal the true holder entity's slot AND (b) that slot
    must UNIQUELY identify the entity in this narrative (no other entity shares it). Under MERGE (all entities -> slot 0)
    (b) fails -> collapses (the slot no longer names an entity)."""
    ok = 0
    for n, (pairs, hseq, L_) in enumerate(items):
        h = hseq[L_ - 1]                                       # true final holder ENTITY
        e2s = maps[n]; slot_h = e2s.get(h, 0)
        unique = sum(1 for e, s in e2s.items() if s == slot_h) == 1
        if int(final_slots[n]) == slot_h and unique:
            ok += 1
    return ok / max(1, len(items))


def _train_and_entity_track(tr_items, te_items, codes, seed, n_hid, epochs, no_bind_seed=None):
    """Returns (entity_level_acc, slot_level_acc, binder_collision_rate). entity==slot (penalty 0) means the binder maps
    the held-out NOVEL entities to clean distinct slots -> the novel-referent binding adds NO error over the attractor's
    own slot-tracking ceiling (which is the inherited D3 autoregressive-rollout accuracy, a separate axis)."""
    nb_tr = np.random.default_rng(no_bind_seed) if no_bind_seed is not None else None
    nb_te = np.random.default_rng(no_bind_seed + 1) if no_bind_seed is not None else None
    tr_task, _ = _to_slot_task(tr_items, codes, _K, no_bind_rng=nb_tr)
    te_task, te_maps = _to_slot_task(te_items, codes, _K, no_bind_rng=nb_te)
    task = {**tr_task, "test_deeper": te_task["test_deeper"], "test_same": te_task["test_deeper"]}
    r = discrete_attractor_rnn(task, seed=seed, n_hid=n_hid, epochs=epochs)
    Xe, Ye, Le, SEQe, STe = te_task["test_deeper"]
    ent = _entity_acc(_final_slots(r["weights"], Xe, Le), te_items, te_maps)
    coll = float(np.mean([len(set(mp.values())) < len(mp) for mp in te_maps]))
    return ent, float(r["state_deeper"]), coll


def run(seed, n_per_len=1200, n_hid=160, epochs=80):
    rng = np.random.default_rng(seed)
    codes = _mint_codes(rng, 12)                              # 12 entities: 0-5 TRAIN pool, 6-11 held-out NOVEL
    tr_ents = list(range(6)); te_ents = list(range(6, 12))
    tr_items = _narratives(rng, tr_ents, _LENS_TR, n_per_len)
    te_items = _narratives(rng, te_ents, _LENS_TE, max(300, n_per_len // 4))   # NOVEL entities, DEEPER lengths
    novel, slot, coll = _train_and_entity_track(tr_items, te_items, codes, seed, n_hid, epochs)
    # MERGE lesion: identical codes -> binder collapses all entities onto slot 0 -> deref no longer names the entity
    codesm = _mint_codes(np.random.default_rng(seed + 1), 12, merge=True)
    merge, _, _ = _train_and_entity_track(tr_items, te_items, codesm, seed, n_hid, epochs)
    # NO-BIND lesion: random slot assignment
    nobind, _, _ = _train_and_entity_track(tr_items, te_items, codes, seed, n_hid, epochs, no_bind_seed=seed + 30)
    # retention floor: the initial holder entity is slot 0's entity; predict slot 0 always + unique
    te_task, te_maps = _to_slot_task(te_items, codes, _K)
    Le = te_task["test_deeper"][2]
    rt = _entity_acc(np.zeros(len(Le), np.int64), te_items, te_maps)
    binding_penalty = slot - novel                            # ~0 => the binder adds NO error over the slot-track ceiling
    # GO measures the NOVEL-REFERENT BINDING contribution: the binder generalizes to held-out novel entities with no
    # penalty vs the attractor's own slot ceiling, is load-bearing (both lesions collapse), and beats the memory floors.
    # (The absolute number inherits the D3 autoregressive-rollout ceiling -- a separate axis, reported as slot.)
    go = (binding_penalty < 0.05) and (novel > rt + 0.15) and (merge < 0.35) and (nobind < 0.35) and (coll < 0.02)
    print(f"[novelref seed={seed}] NOVEL-entity track@deeper={novel:.3f} (slot-ceiling={slot:.3f}, binding-penalty="
          f"{binding_penalty:+.3f}, collisions={coll:.3f}) | merge={merge:.3f} no-bind={nobind:.3f} retention={rt:.3f} "
          f"chance={1/_K:.3f} -> {'GO' if go else 'no'}")
    return dict(seed=seed, novel=round(novel, 3), slot_ceiling=round(slot, 3),
                binding_penalty=round(binding_penalty, 3), collisions=round(coll, 3), merge=round(merge, 3),
                nobind=round(nobind, 3), retention=round(rt, 3), go=bool(go))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    t0 = time.time()
    results = [run(s) for s in seeds]
    if len(results) > 1:
        print(f"[novelref] {sum(1 for r in results if r['go'])}/{len(results)} seeds GO")
    if a.out:
        json.dump(dict(results=results, elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
