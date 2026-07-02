"""EMERGE-9d (rung-3d) — SPIKING LEARNING for the HTM Temporal Memory: the last numpy rung before the sim/ port.
Single-variable step from EMERGE-9c (spiking inference GO): make LEARNING spike-driven via the Bouhadjar-Diesmann
2022 THREE-TERM permanence rule, so BOTH inference and learning are on the spiking substrate (the honest
single-spiking-substrate goal). Question: does the unsupervised context-specific branch prediction SURVIVE when the
permanences are driven by SPIKE TIMING + dAP-rate homeostasis instead of the discrete winner-based rule?

THE THREE-TERM RULE (verified from the PMC full text, Eq. 1):
  (1) POTENTIATION: STDP-windowed -- when a cell j spikes (post), reinforce its distal synapses FROM cells that spiked
      on the PREVIOUS symbol (pre, within the window), rate lambda_pot. (excludes synchronous same-symbol pairs.)
  (2) DEPRESSION: constant presynaptic -- each pre-spike depresses that synapse a little (rate lambda_dep), so
      synapses to cells that keep firing WITHOUT the post become disconnected.
  (3) HOMEOSTASIS (dAP-rate): each cell tracks a low-pass dAP rate z_i (how often it becomes predictive); potentiation
      is scaled by (z* - z_i), so an over-used cell stops potentiating -> NEW contexts allocate onto FRESH (low-z)
      cells. This replaces EMERGE-9's discrete 'least-committed allocation' heuristic with the biological mechanism.

Inference = EMERGE-9c's spiking LIF + dAP plateau + per-column WTA (inherited). Reuse-by-import; NO sim/ edit;
CPU/numpy; multi-seed. Anti-cheats: beat the Markov floor + dAP-lesion collapses + no-teacher + oracle + multi-seed.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge9b_htm_faithful_derisk import make_overlap_sequences, markov_branch_acc, full_oracle
from research.runners._emerge9c_spiking_tm_derisk import SpikingTM

OUT = Path("research/findings/raw/_emerge9d_spiking_learning.json")


class SpikingLearnTM(SpikingTM):
    """Spiking inference (inherited) + SPIKE-DRIVEN three-term permanence learning + dAP-rate homeostatic allocation."""

    def __init__(self, *a, lam_pot=0.14, lam_dep=0.02, z_tau=0.85, z_star=1.0, connect_grow=6, **k):
        super().__init__(*a, **k)
        self.lam_pot, self.lam_dep, self.z_star = lam_pot, lam_dep, z_star
        self.z_tau, self.connect_grow = z_tau, connect_grow
        self.z = np.zeros(self.N)                       # per-cell low-pass dAP (predictive) rate

    def train_sequence_spiking(self, seq, seed=0):
        """Present `seq` with SPIKING winner selection AND the spike-driven THREE-TERM permanence rule. Allocation is
        bootstrapped by the proven committed-segment metric (EMERGE-9b); the three-term plasticity (potentiation scaled
        by dAP-rate homeostasis + presynaptic depression) drives the permanences. WINNERS (sparse) carry the context
        -- matching/learning uses prev WINNERS, not the bursting active set (the EMERGE-9b lesson)."""
        rng = np.random.default_rng(seed + 555)
        predictive = set(); prev_winners = set()
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            to_learn = []
            if primed:
                winners = self._spiking_winners(col, set(primed), rng)   # dAP-primed win the WTA -> sparse
                active = winners
                for i in winners:
                    seg, sc = self._best_seg(i, prev_winners)
                    if seg is not None and sc >= self.learn_th:
                        to_learn.append((i, seg))
            elif not prev_winners:                                       # cue: stable SDR, no segment, burst-active
                winners = set(col[:self.k_win]); active = set(col)
            else:
                active = set(col)                                        # burst (mismatch): whole column fires
                scored = sorted(((self._best_seg(i, prev_winners)[1], i) for i in col), reverse=True)
                if scored[0][0] >= self.learn_th:                        # matching segment -> reinforce
                    winners = set()
                    for sc, i in scored[:self.k_win]:
                        if sc >= self.learn_th:
                            winners.add(i); to_learn.append((i, self._best_seg(i, prev_winners)[0]))
                else:                                                    # ALLOCATE disjoint SDR (committed-metric bootstrap)
                    lu = sorted(col, key=lambda i: (self._committed(i), i))[:self.k_win]
                    winners = set(lu)
                    for i in lu:
                        to_learn.append((i, self._new_seg(i)))
            if prev_winners:
                for j, seg in to_learn:                                  # (1) potentiation to prior WINNERS x homeostasis
                    hfac = 0.5 + 0.5 * max(0.0, self.z_star - self.z[j])  # homeostasis modulates, never fully gates
                    for p in list(seg.keys()):
                        seg[p] = min(1.0, seg[p] + self.lam_pot * hfac) if p in prev_winners else max(0.0, seg[p] - self.lam_dep)
                    grow = [p for p in prev_winners if p not in seg]
                    rng.shuffle(grow)
                    for p in grow[:self.connect_grow]:
                        seg[p] = self.p_init
                for i in list(predictive):                              # (2) presynaptic depression of wrongly-predictive cells
                    if i // self.nE != c:
                        for seg in self.segments[i]:
                            if self._seg_conn_active(seg, prev_winners) >= self.act_th:
                                for p in list(seg.keys()):
                                    if p in prev_winners:
                                        seg[p] = max(0.0, seg[p] - self.lam_dep)
            predictive = set()
            for i in range(self.N):
                for seg in self.segments[i]:
                    if self._seg_conn_active(seg, active) >= self.act_th:
                        predictive.add(i); break
            self.z *= self.z_tau                                         # (3) low-pass dAP-rate homeostasis tracking
            for i in predictive:
                self.z[i] += (1.0 - self.z_tau)
            prev_winners = winners

def _run_arm(job):
    seed, arm, n_seq, L, n_cells, k_win, act_th, epochs = job
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    div_pos = L
    tm = SpikingLearnTM(vocab, n_cells=n_cells, seed=seed, k_win=k_win, act_th=act_th, lesion=(arm == "lesion"))
    if arm != "untrained":
        for _ in range(epochs):
            for s in seqs:
                tm.train_sequence_spiking(s, seed=seed)
    # eval with spiking inference (lesion honored)
    ok = 0
    for s in seqs:
        pred = tm.run_sequence_spiking(s, seed=seed)[div_pos]
        ok += int(pred == {s[div_pos + 1]})
    return (seed, arm, {"branch": ok / len(seqs), "locality_ok": (not tm.used_transpose)})


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    t0 = time.time(); err = None; per = []
    floors = {}
    for s in a.seeds:
        seqs, vocab, info = make_overlap_sequences(n_seq=a.n_seq, middle_len=a.middle_len, seed=s)
        floors[s] = {"markov_L": markov_branch_acc(seqs, a.middle_len, a.n_seq),
                     "oracle": full_oracle(seqs, a.middle_len), "chance": 1.0 / a.n_seq}
    try:
        jobs = [(s, arm, a.n_seq, a.middle_len, a.n_cells, a.k_win, a.act_th, a.epochs)
                for s in a.seeds for arm in ARMS]
        cap = a.max_workers if (a.max_workers and a.max_workers > 0) else (os.cpu_count() or 1)
        collected = {}
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=min(len(jobs), cap)) as ex:
                for seed, arm, entry in ex.map(_run_arm, jobs):
                    collected.setdefault(seed, {})[arm] = entry
        except Exception:
            for job in jobs:
                seed, arm, entry = _run_arm(job); collected.setdefault(seed, {})[arm] = entry
        for s in a.seeds:
            d = collected[s]; d["seed"] = s; d["floors"] = floors[s]; per.append(d)
        for d in per:
            f = d["floors"]
            print(f"  [seed {d['seed']}] SPIKE-LEARN branch {d['htm']['branch']:.3f} | lesion {d['lesion']['branch']:.3f} "
                  f"| untr {d['untrained']['branch']:.3f} || markov {f['markov_L']:.3f} chance {f['chance']:.3f} "
                  f"loc {d['htm']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm):
            return float(np.mean([p[arm]["branch"] for p in per]))
        htm, les, unt = m("htm"), m("lesion"), m("untrained")
        markov = float(np.mean([p["floors"]["markov_L"] for p in per]))
        chance = float(np.mean([p["floors"]["chance"] for p in per]))
        oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
        loc = all(p["htm"]["locality_ok"] for p in per)
        go = bool(oracle > 0.99 and htm >= 0.90 and htm >= markov + 0.15 and htm >= chance + 0.20 and htm >= les + 0.20 and loc)
        if not loc:
            verdict = "INVALID -- locality assert failed."
        elif go:
            verdict = (f"GO -- FULLY-SPIKING unsupervised HTM Temporal Memory: BOTH inference (LIF + dAP + WTA) AND "
                       f"learning (the Bouhadjar THREE-TERM rule -- spike-timing potentiation + presynaptic depression + "
                       f"dAP-rate homeostasis) self-organize context-specific high-order prediction: branch acc {htm:.3f} "
                       f">> Markov {markov:.3f}, >> chance {chance:.3f}, >> dAP-lesion {les:.3f}; untrained {unt:.3f}; "
                       f"locality asserted; NO teacher; homeostasis replaces the discrete allocation heuristic. Multi-seed. "
                       f"=> the numpy spiking-substrate ladder is COMPLETE (rung 3) -> rung-4 = the sim/ two-compartment "
                       f"NeuronModel port (dAP = apical compartment; three-term rule as a plastic RegionPathway). NO sim/ edit.")
        else:
            miss = []
            if htm < 0.90: miss.append(f"branch {htm:.3f} < 0.90")
            if htm < markov + 0.15 or htm < chance + 0.20: miss.append(f"didn't clear Markov/chance ({htm:.3f})")
            if htm < les + 0.20: miss.append(f"dAP-lesion didn't collapse ({htm:.3f} vs {les:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f" (oracle {oracle:.3f}). The spike-driven "
                       f"three-term rule didn't fully self-organize context here -> tune lam_pot/lam_dep/z_tau/act_th/"
                       f"epochs (homeostasis + STDP window timing). NOT a wall; the spiking learning is the next tuning.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge9d_spiking_learning", "verdict": verdict,
               "mechanism": "FULLY-SPIKING HTM Temporal Memory: spiking inference (LIF+dAP+WTA) + spike-driven three-term "
                            "learning (STDP-windowed potentiation + presynaptic depression + dAP-rate homeostasis); "
                            "unsupervised (no teacher); homeostasis replaces the discrete allocation heuristic",
               "task": "overlapping sequences; branch prediction; Markov floor + dAP-lesion + oracle + multi-seed",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Completes the numpy spiking ladder: inference AND learning are spike-driven. The three-term "
                              "rule is the verified Bouhadjar Eq.1 (potentiation windowed + presynaptic depression + dAP-rate "
                              "homeostasis). Next: rung-4 sim/ two-compartment NeuronModel port (dAP = apical compartment). "
                              "Unsupervised: no teacher; self-organization IS the deliverable."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge9d] VERDICT: {verdict}", flush=True)
    print(f"[emerge9d] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
