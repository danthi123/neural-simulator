"""SPIKING realization of the variable-binding fast-weight BIND, on a real `SimulationBridge` (brain-based-only close of
the WM-GO residual (c): "the fast-weight BIND is host numpy; its spiking-STP realisation is a banked next rung").

WHAT THIS BUILDS (per RUNG 6d, which resolved the MECHANISM question): the RUNG6c content-agnostic Hebbian fast weight
`A += eta * post * pre^T` is realized as HEBBIAN SHORT-TERM POTENTIATION on the substrate -- a barcode input region +
K slot pools + a shared FS (winner-take-all), with barcode->slot synapses PLASTIC via `sim`'s rate-window Hebbian
coactivity rule (`cp_hebb_coactivity_trace`, the pre x post coincidence window). The bind lives in REAL SYNAPSE WEIGHTS
(`cp_connections.data`), written by the substrate's own spiking plasticity -- NOT a host numpy matrix. BIND: drive a
barcode + its (host-allocated, as RUNG6c's `free` counter) slot -> coincident firing potentiates barcode->slot.
RETRIEVE: re-present the barcode ALONE -> the potentiated synapses drive its bound slot to fire (content-addressable),
read from spikes (`cp_firing_states`).

WHY NOT THE LITERAL Mongillo STP FACILITATION (`cp_stp_u`/`cp_stp_x`): RUNG 6d found presynaptic Tsodyks-Markram
facilitation is NON-SELECTIVE (u rises on ALL barcode->slot synapses the barcode drives, regardless of which slot won
=> 0.999 collisions). AND its window is too short: Mongillo 2008's facilitation tau_F = 1500 ms (verified at source,
Frontiers fnint 2022.972055 quoting Mongillo 2008), while `sim`'s `stp_tau_f` defaults to 50 ms -- ~30x too short to
bridge a WM span. So `cp_stp_u`/`cp_stp_x` ALONE cannot hold a selective bind; the HEBBIAN synaptic potentiation
surpasses BOTH failure modes (selective + a durable weight that persists natively). This runner MEASURES both on the
substrate: the `stponly` arm (STP enabled, Hebbian OFF) is the honest cp_stp_u/x negative; the `hebbian` arm is the
surpass.

THE QUESTION: does the spiking-STP (Hebbian) bind reproduce the RUNG6c GO -- 0 collisions / clean recovery on HELD-OUT
NOVEL entities minted at test -- as a genuine synaptic mechanism (not host numpy)?

ANTI-CHEATS (tools.lab + tools.verdict.Verdict):
  - held-out NOVEL fillers: entities minted at test with a disjoint RNG; content-agnostic => works identically (assert
    novel recovery == a same-run "known"-pool control, no memorisation gap).
  - LOAD-BEARING: freeze plasticity (hebbian_lr=0, `stponly`) -> no weight change -> retrieval collapses to chance.
  - NO host fast-weight array in the readout: the binder object holds NO numpy `W`; retrieval reads only spikes; the
    bind is asserted to live in `cp_connections.data` (weights MOVED: before vs after).
  - permuted-binding: score barcode_e against a shuffled slot map -> chance.
  - MERGE lesion: identical barcodes -> cannot individuate -> collisions high, recovery -> chance.
  - backend/device emitted (assert_backend numpy); cfg.seed set (substrate actually seeded); plasticity-bound clamp
    (hebbian_max_weight > every design weight) verified.

Reuse-by-import (brain-region framework + rate-window Hebbian + FS-WTA are all `sim` public config); NO `sim/` edit.

Run: SIM_BACKEND=numpy python -m research.runners._spiking_stp_bind_derisk --smoke --seed 42
     SIM_BACKEND=numpy python -m research.runners._spiking_stp_bind_derisk --derisk --seeds 42 43 44 100 101 102 \
         --out research/findings/raw/_spiking_stp_bind/stp_bind_6seed.json
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

from tools.lab import assert_backend, bound_check
from tools.verdict import Verdict

# --- geometry (small, fast; the mechanism, not scale, is under test) ---
_K = 4                 # bounded slots (host-allocated, as RUNG6c's `free` counter)
_N_BARCODE = 48
_KACT = 6              # active bits per sparse barcode (overlap-rejection < 3 shared)
_N_SLOT = 36
_N_FS = 20
_T_BIND = 50           # steps per bind presentation
_T_READ = 50           # steps per retrieve presentation
_INIT_W = 30.0         # barcode->slot init weight: SUBTHRESHOLD so an unbound slot is silent
_MAX_W = 4000.0        # Hebbian ceiling (>> every design weight: the plasticity-bound clamp)
_HEBB_LR = 0.25
_DRIVE = 1300.0        # barcode drive (pA)
_CLAMP = 1700.0        # teaching clamp on the host-allocated target slot during BIND (the "post" of pre x post)


def _mint(rng, M, merge=False):
    """M sparse 0/1 barcodes (k-active of dim), overlap-rejected (< 3 shared bits). merge=True -> ALL identical (the
    no-individuation lesion)."""
    if merge:
        c = np.zeros(_N_BARCODE, np.float32); c[rng.choice(_N_BARCODE, _KACT, replace=False)] = 1.0
        return np.repeat(c[None], M, 0)
    codes = []
    while len(codes) < M:
        c = np.zeros(_N_BARCODE, np.float32); c[rng.choice(_N_BARCODE, _KACT, replace=False)] = 1.0
        if all(float((c > 0) @ (d > 0)) < 3 for d in codes):
            codes.append(c)
    return np.asarray(codes, np.float32)


class SpikingSTPBinder:
    """A real `SimulationBridge`: barcode -> K slot pools -> shared FS. barcode->slot PLASTIC (rate-window Hebbian).
    The bind is written by the substrate's spiking coactivity into `cp_connections.data`; retrieval reads spikes.
    `stp_only=True` freezes the Hebbian rule (relies on `cp_stp_u`/`cp_stp_x` presynaptic facilitation) = the honest
    cp_stp_u/x negative arm."""

    def __init__(self, seed, hebb_lr=_HEBB_LR, init_w=_INIT_W, max_w=_MAX_W, stp_only=False):
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.regions import BrainRegion, RegionPathway
        self.plastic = (hebb_lr > 0.0) and (not stp_only)
        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        regions = [BrainRegion(name="barcode", n_neurons=_N_BARCODE, exc_fraction=1.0, internal_density=0.0)]
        for s in range(_K):
            regions.append(BrainRegion(name=f"slot{s}", n_neurons=_N_SLOT, exc_fraction=1.0, internal_density=0.0))
        regions.append(BrainRegion(name="slot_fs", n_neurons=_N_FS, exc_fraction=0.0, internal_density=0.0))
        cfg.brain_regions = regions
        paths = []
        for s in range(_K):
            paths.append(RegionPathway(from_region="barcode", to_region=f"slot{s}", density=1.0,
                                       weight_mean=init_w, weight_jitter=0.1, plastic=self.plastic))
            paths.append(RegionPathway(from_region=f"slot{s}", to_region="slot_fs",
                                       density=0.6, weight_mean=2.0, weight_jitter=0.1, plastic=False))
            paths.append(RegionPathway(from_region="slot_fs", to_region=f"slot{s}",
                                       density=0.6, weight_mean=3.0, weight_jitter=0.1, plastic=False))
        cfg.region_pathways = paths
        cfg.dt = 1.0
        cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed     # substrate ACTUALLY seeded (CLAUDE.md cfg.seed trap)
        cfg.enable_ou_process = False
        cfg.enable_stdp = False
        cfg.enable_short_term_plasticity = True                    # STP (cp_stp_u/x) ON in BOTH arms
        cfg.enable_hebbian_learning = self.plastic
        cfg.hebbian_rate_window = True
        cfg.hebbian_coactivity_decay = 0.9
        cfg.hebbian_coactivity_thresh = 0.04
        cfg.hebbian_learning_rate = hebb_lr if self.plastic else 0.0
        cfg.hebbian_max_weight = max_w
        cfg.hebbian_min_weight = 0.05
        cfg.enable_homeostasis = False
        self.max_w = max_w; self.init_w = init_w
        rt = RuntimeState(); rt.actual_seed_used = seed
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=rt, gpu_config=GPUConfig())
        b._initialize_simulation_data()
        self.b = b
        self.idx = {"barcode": np.asarray(b.region_manager.indices("barcode"))}
        for s in range(_K):
            self.idx[f"slot{s}"] = np.asarray(b.region_manager.indices(f"slot{s}"))
        # snapshot the fresh init state so a narrative can be RESET without a rebuild (== RUNG6c per-narrative W reset)
        from sim.backend import to_host
        self._snap = dict(
            w=np.asarray(to_host(b.cp_connections.data)).copy(),
            v=np.asarray(to_host(b.cp_membrane_potential_v)).copy(),
            u=np.asarray(to_host(b.cp_recovery_variable_u)).copy(),
            trace=(np.asarray(to_host(b.cp_hebb_coactivity_trace)).copy()
                   if b.cp_hebb_coactivity_trace is not None else None),
        )
        # bind synapse index map: which entries of cp_connections.data are barcode->slot (for the "weights MOVED" read)
        self._bind_syn = self._barcode_to_slot_syn_indices()

    def _barcode_to_slot_syn_indices(self):
        """Positions in cp_connections.data (CSR, row=pre, col=post; .data aligns with .tocoo() row-major) that are
        barcode(pre)->slot(post): the synapses the bind writes. Sparsity pattern is fixed (no structural plasticity),
        so these positions are stable across learning."""
        from sim.backend import to_host
        coo = self.b.cp_connections.tocoo()
        rows = np.asarray(to_host(coo.row)); cols = np.asarray(to_host(coo.col))
        bc = set(self.idx["barcode"].tolist())
        out = {}
        for s in range(_K):
            sl = set(self.idx[f"slot{s}"].tolist())
            out[s] = np.where(np.array([(r in bc) and (c in sl) for r, c in zip(rows, cols)]))[0]
        return out

    def reset(self):
        """Restore the fresh init state (weights + membrane + recovery + Hebbian trace) -> a fresh narrative."""
        xp = self.b.xp
        self.b.cp_connections.data[:] = xp.asarray(self._snap["w"])
        self.b.cp_membrane_potential_v[:] = xp.asarray(self._snap["v"])
        self.b.cp_recovery_variable_u[:] = xp.asarray(self._snap["u"])
        if self.b.cp_hebb_coactivity_trace is not None and self._snap["trace"] is not None:
            self.b.cp_hebb_coactivity_trace[:] = xp.asarray(self._snap["trace"])

    def _run(self, code, n_steps, clamp_slot=None, learn=False):
        from sim.backend import to_host
        self.b.core_config.enable_hebbian_learning = bool(learn) and self.plastic
        bc = self.idx["barcode"]; active = bc[code > 0]
        counts = np.zeros(_K, np.float64)
        for _ in range(n_steps):
            self.b.cp_external_input_current[:] = 0.0
            self.b.cp_external_input_current[active] = _DRIVE
            if clamp_slot is not None:
                self.b.cp_external_input_current[self.idx[f"slot{clamp_slot}"]] = _CLAMP
            self.b._run_one_simulation_step()
            fs = np.asarray(to_host(self.b.cp_firing_states)).astype(np.float64)
            for s in range(_K):
                counts[s] += fs[self.idx[f"slot{s}"]].sum()
        self.b.cp_external_input_current[:] = 0.0
        return counts / (n_steps * _N_SLOT)

    def bind(self, code, slot, reps=3):
        """Write barcode->slot: drive the barcode + CLAMP the host-allocated target slot; coincident firing potentiates
        (the substrate's Hebbian coactivity). Host chooses the slot (as RUNG6c); the substrate does the potentiation."""
        for _ in range(reps):
            self._run(code, _T_BIND, clamp_slot=slot, learn=True)

    def retrieve(self, code):
        """Re-present the barcode ALONE (no clamp, learn off) -> the potentiated synapses drive its bound slot. Read
        the winner from SPIKES only."""
        rates = self._run(code, _T_READ, clamp_slot=None, learn=False)
        return int(np.argmax(rates)), rates

    def bind_weight(self, slot):
        """Mean barcode->slot synapse weight, read from cp_connections.data (the substrate's own state)."""
        from sim.backend import to_host
        w = np.asarray(to_host(self.b.cp_connections.data))
        idx = self._bind_syn[slot]
        return float(w[idx].mean()) if len(idx) else 0.0


def _bind_retrieve_trial(binder, codes, slots, permute_rng=None):
    """Bind each (code -> host-allocated slot), then retrieve each code alone. Returns (recovery, collisions,
    winners). collisions = fraction of codes whose retrieved winner is shared by another code (retrieval not
    individuating). permute_rng -> score against a SHUFFLED slot map (the permuted-binding control)."""
    binder.reset()
    for c, s in zip(codes, slots):
        binder.bind(c, s)
    winners = []
    for c in codes:
        w, _ = binder.retrieve(c)
        winners.append(w)
    target = list(slots)
    if permute_rng is not None:
        target = list(np.asarray(slots)[permute_rng.permutation(len(slots))])
    recovery = float(np.mean([w == t for w, t in zip(winners, target)]))
    from collections import Counter
    cnt = Counter(winners)
    collisions = float(np.mean([cnt[w] > 1 for w in winners]))
    return recovery, collisions, winners


def run(seed, n_trials=24, k_entities=_K, verbose=False, hebb_lr=_HEBB_LR, init_w=_INIT_W, max_w=_MAX_W):
    """One seed: HELD-OUT novel pool (main) + KNOWN-pool control + lesions/controls. Each trial mints a fresh entity
    set and host-allocates them to distinct slots; the mechanism is content-agnostic so novel == known by construction."""
    rng = np.random.default_rng(seed)
    binder = SpikingSTPBinder(seed, hebb_lr=hebb_lr, init_w=init_w, max_w=max_w)
    stp_only = SpikingSTPBinder(seed, init_w=init_w, max_w=max_w, stp_only=True)   # cp_stp_u/x arm (Hebbian frozen)

    # plasticity-bound clamp: the Hebbian ceiling must exceed every design weight (else the soft-bound collapses it)
    bound_check("hebbian_max_weight", binder.max_w, weight=max(_INIT_W, _CLAMP / 100.0, 3.0), strict=True)

    def _sweep(bd, permute=False, merge=False, weights_probe=False):
        recs, colls = [], []
        w_before = w_after = None
        for t in range(n_trials):
            codes = _mint(np.random.default_rng(seed * 1000 + t + (7 if merge else 0)), k_entities, merge=merge)
            slots = list(range(k_entities))                # host allocation (distinct), as RUNG6c's free-counter
            if weights_probe and t == 0:
                bd.reset(); w_before = bd.bind_weight(slots[0]); bd.bind(codes[0], slots[0]); w_after = bd.bind_weight(slots[0])
            pr = np.random.default_rng(seed * 31 + t) if permute else None
            r, c, _ = _bind_retrieve_trial(bd, codes, slots, permute_rng=pr)
            recs.append(r); colls.append(c)
        return float(np.mean(recs)), float(np.mean(colls)), w_before, w_after

    novel_rec, novel_coll, w_before, w_after = _sweep(binder, weights_probe=True)
    # KNOWN-pool control: reuse ONE fixed entity set across trials (no minting) -> same recovery => no memorisation gap
    fixed = _mint(np.random.default_rng(seed * 99), k_entities)
    known_recs, known_colls = [], []
    for t in range(n_trials):
        r, c, _ = _bind_retrieve_trial(binder, fixed, list(range(k_entities)))
        known_recs.append(r); known_colls.append(c)
    known_rec = float(np.mean(known_recs)); known_coll = float(np.mean(known_colls))

    lesion_rec, lesion_coll, _, _ = _sweep(stp_only)                       # freeze Hebbian (cp_stp_u/x only)
    perm_rec, _, _, _ = _sweep(binder, permute=True)                       # permuted-binding
    merge_rec, merge_coll, _, _ = _sweep(binder, merge=True)               # identical codes

    chance = 1.0 / k_entities
    weights_moved = (w_after is not None) and (w_after > w_before + 1.0)

    v = Verdict(f"spiking-STP(Hebbian) bind seed={seed}", chance=chance)
    v.disabled("OU noise", "symmetry break not needed with a host-allocated write target")
    v.disabled("homeostasis", "per-narrative binder; homeostatic scaling would erode the fast weight")
    v.floor("novel recovery vs chance", novel_rec, chance)
    v.require("novel recovery clean (>=0.90)", novel_rec, expect=lambda x: x >= 0.90)
    v.require("novel collisions ~0 (<=0.05)", novel_coll, expect=lambda x: x <= 0.05)
    v.control("held-out novel == known (no memorisation gap)", novel_rec, known_rec, min_separation=-1.0)  # informational
    v.require("no memorisation gap (|novel-known|<=0.10)", abs(novel_rec - known_rec), expect=lambda x: x <= 0.10)
    v.reaches("bind weight MOVED (cp_connections.data)", before=w_before, after=w_after)
    v.require("bind is in synapses not host numpy (no .W attr)", hasattr(binder, "W"), expect=False)
    v.require("weights actually moved", weights_moved, expect=True)
    v.control("LOAD-BEARING: freeze-Hebbian collapses recovery", novel_rec, lesion_rec, min_separation=0.30)
    v.require("frozen-Hebbian (cp_stp_u/x-only) at/near chance (<=0.45)", lesion_rec, expect=lambda x: x <= 0.45)
    v.control("permuted-binding -> chance", novel_rec, perm_rec, min_separation=0.30)
    v.require("permuted recovery near chance (<=0.45)", perm_rec, expect=lambda x: x <= 0.45)
    v.control("MERGE (identical codes) collapses recovery", novel_rec, merge_rec, min_separation=0.30)
    v.require("merge collisions high (>=0.50)", merge_coll, expect=lambda x: x >= 0.50)

    go = (novel_rec >= 0.90 and novel_coll <= 0.05 and lesion_rec <= 0.45 and perm_rec <= 0.45
          and merge_rec <= (chance + 0.20) and weights_moved and abs(novel_rec - known_rec) <= 0.10)
    decided = v.decide(go=go, verbose=verbose)

    print(f"[stp-bind seed={seed}] NOVEL rec={novel_rec:.3f} coll={novel_coll:.3f} | known rec={known_rec:.3f} | "
          f"cp_stp_u/x-only(frozen-Hebb) rec={lesion_rec:.3f} | permuted={perm_rec:.3f} | merge rec={merge_rec:.3f} "
          f"coll={merge_coll:.3f} | chance={chance:.3f} | W {w_before:.1f}->{w_after:.1f} moved={weights_moved} "
          f"-> {'GO' if go else 'no'} [{decided['status']}]")
    return dict(seed=seed, novel_rec=round(novel_rec, 3), novel_coll=round(novel_coll, 3),
                known_rec=round(known_rec, 3), stponly_rec=round(lesion_rec, 3), permuted_rec=round(perm_rec, 3),
                merge_rec=round(merge_rec, 3), merge_coll=round(merge_coll, 3), chance=round(chance, 3),
                w_before=round(w_before, 2), w_after=round(w_after, 2), weights_moved=bool(weights_moved),
                status=decided["status"], go=bool(go and decided["status"] == "GO"))


def run_smoke(seed=42, hebb_lr=_HEBB_LR, init_w=_INIT_W, max_w=_MAX_W):
    """Fast one-narrative diagnostic: bind 2 codes, retrieve, print rates + weight movement (tune drive/clamp/lr)."""
    assert_backend("numpy")
    binder = SpikingSTPBinder(seed, hebb_lr=hebb_lr, init_w=init_w, max_w=max_w)
    codes = _mint(np.random.default_rng(seed), 4)
    binder.reset()
    w0 = binder.bind_weight(0)
    binder.bind(codes[0], 0); binder.bind(codes[1], 1)
    w0b = binder.bind_weight(0); w1b = binder.bind_weight(1); w2b = binder.bind_weight(2)
    r0, rates0 = binder.retrieve(codes[0]); r1, rates1 = binder.retrieve(codes[1])
    rN, ratesN = binder.retrieve(codes[3])   # novel unbound -> should not fire a bound slot strongly
    print(f"[smoke seed={seed}] W: slot0 {w0:.1f}->{w0b:.1f} slot1->{w1b:.1f} slot2(unbound)->{w2b:.1f}")
    print(f"  retrieve code0 -> winner {r0} rates={np.round(rates0,3)} (want 0)")
    print(f"  retrieve code1 -> winner {r1} rates={np.round(rates1,3)} (want 1)")
    print(f"  retrieve NOVEL -> winner {rN} rates={np.round(ratesN,3)} (unbound, low)")
    print(f"  -> {'BIND/RETRIEVE OK' if (r0 == 0 and r1 == 1) else 'iterate (drive/clamp/lr/init_w)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--n-trials", type=int, default=24)
    ap.add_argument("--hebb-lr", type=float, default=_HEBB_LR, help="cfg.hebbian_learning_rate (the fast-weight write rate)")
    ap.add_argument("--init-w", type=float, default=_INIT_W, help="barcode->slot init weight (subthreshold so unbound slots are silent)")
    ap.add_argument("--max-w", type=float, default=_MAX_W, help="cfg.hebbian_max_weight (the potentiation ceiling / plasticity-bound clamp)")
    a = ap.parse_args()
    t0 = time.time()
    if a.smoke:
        run_smoke(a.seed, hebb_lr=a.hebb_lr, init_w=a.init_w, max_w=a.max_w)
        print(f"  ({time.time()-t0:.1f}s)")
        return
    assert_backend("numpy")
    seeds = a.seeds if a.seeds else [a.seed]
    results = [run(s, n_trials=a.n_trials, verbose=(len(seeds) == 1),
                   hebb_lr=a.hebb_lr, init_w=a.init_w, max_w=a.max_w) for s in seeds]
    n_go = sum(1 for r in results if r["go"])
    print(f"[stp-bind] {n_go}/{len(results)} seeds GO  ({time.time()-t0:.1f}s)")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(results=results, n_go=n_go, n_seeds=len(results),
                       elapsed_s=round(time.time() - t0, 1)), open(a.out, "w"))


if __name__ == "__main__":
    main()
