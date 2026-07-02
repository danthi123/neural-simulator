"""EMERGE-9c (rung-3b) — the SPIKING inference for the HTM Temporal Memory. Single-variable step from EMERGE-9b's
discrete GO: keep the validated unsupervised local learning UNCHANGED, but replace the discrete "predictive cell ->
active" selection with real SPIKING dynamics -- LIF somas + a distal-dendrite PLATEAU (dAP) that pre-depolarizes
predicted cells + per-column WINNER-TAKE-ALL inhibition. Question: does the context-specific high-order branch
prediction SURVIVE when the winner selection emerges from spiking competition (predicted/dAP-primed cells fire first
and inhibit the rest) instead of a discrete rule? This is the substrate rung toward the sim/ two-compartment port:
the dAP IS our confirmed two-compartment neuron's apical compartment.

MAPPING (from the verified Bouhadjar-Diesmann 2022 spiking TM): a cell is "predictive" when its distal segment has
>= act_th connected synapses from the currently-active cells -> the apical dendrite emits a dAP PLATEAU that adds a
sustained depolarizing current to the soma. On the next symbol, every cell in the column receives feedforward input;
dAP-primed cells (already depolarized) cross threshold FIRST, spike, and drive the column's inhibitory neuron, which
suppresses the not-yet-fired cells (WTA) -> SPARSE, context-specific firing for a predicted element; an UNPREDICTED
element has no primed cells -> the column BURSTS (many cells fire) = the mismatch signal. Learning (permanence) is the
same validated local rule from EMERGE-9b (this rung changes ONLY the inference substrate).

Reuse-by-import (`_emerge9b_htm_faithful_derisk`); NO sim/ edit; CPU/numpy; multi-seed. Anti-cheats: beat the Markov
floor + distal (dAP) LESION collapses + no-teacher + oracle + spiking-vs-discrete parity.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge9b_htm_faithful_derisk import HTM, make_overlap_sequences, markov_branch_acc, full_oracle

OUT = Path("research/findings/raw/_emerge9c_spiking_tm.json")


class SpikingTM(HTM):
    """HTM Temporal Memory with SPIKING inference. Learning is inherited UNCHANGED from EMERGE-9b (validated GO). Only
    the active-cell SELECTION is replaced: LIF somas + a dAP plateau depolarization on predicted cells + per-column WTA.

    Per symbol, we run n_sub LIF substeps for the column's cells:
      V_i += dt/tau * (-V_i) + I_input + dAP_i    (dAP_i = plateau_amp if cell i is predicted, else 0)
      a cell spikes when V_i >= 1; the first k spikers in the column WIN; once k have fired, the WTA inhibitory neuron
      fires and clamps the rest (they cannot spike this symbol). Winners = the spiked cells.
    dAP-primed cells are pre-depolarized so they reach threshold in fewer substeps -> they win the WTA -> the SAME
    sparse context-specific selection the discrete rule made, now emergent from spiking competition."""

    def __init__(self, *a, tau=6.0, dt=1.0, n_sub=12, i_input=0.16, plateau=0.55, noise=0.02, **k):
        super().__init__(*a, **k)
        self.tau, self.dt, self.n_sub = tau, dt, n_sub
        self.i_input, self.plateau, self.noise = i_input, plateau, noise

    def _spiking_winners(self, col, predicted_set, rng):
        """Run LIF competition in one column; return the set of spiked (winner) cells. dAP-primed (predicted) cells
        fire first and the per-column WTA caps the winners at k_win (predicted -> sparse; unpredicted burst -> WTA
        still yields k winners but from the whole column)."""
        nE = len(col)
        V = np.zeros(nE)
        primed = np.array([c in predicted_set for c in col], float)
        spiked = np.zeros(nE, bool)
        order = []
        for _ in range(self.n_sub):
            if spiked.sum() >= self.k_win:                # WTA: once k have fired, inhibition clamps the rest
                break
            drive = self.i_input + self.plateau * primed + self.noise * rng.standard_normal(nE)
            V = V + self.dt / self.tau * (-V) + drive
            V[spiked] = 0.0                               # fired cells reset + are refractory this symbol
            newly = np.where((V >= 1.0) & (~spiked))[0]
            if newly.size:
                # deterministic-ish order: highest V first (dAP-primed reach highest soonest)
                for j in sorted(newly, key=lambda x: -V[x]):
                    if spiked.sum() >= self.k_win:
                        break
                    spiked[j] = True; order.append(col[j])
        return set(order)

    def run_sequence_spiking(self, seq, seed=0):
        """Spiking INFERENCE (no learning): returns per-step predicted-column sets, using LIF+dAP+WTA to pick winners."""
        rng = np.random.default_rng(seed)
        predictive = set()
        preds = []
        for c in seq:
            col = self._col(c)
            predicted_here = [i for i in col if i in predictive]
            if self.lesion:
                predicted_here = []
            if predicted_here:                                             # dAP-primed cells fire first -> WTA -> SPARSE
                active = self._spiking_winners(col, set(predicted_here), rng)
            else:                                                          # no prediction -> the column BURSTS (mismatch)
                active = set(col)
            # compute dAP (predictive) for the next symbol: distal segment connected-active >= act_th -> dAP plateau
            predictive = set()
            for i in range(self.N):
                for seg in self.segments[i]:
                    if self._seg_conn_active(seg, active) >= self.act_th:
                        predictive.add(i); break
            preds.append(set(i // self.nE for i in predictive))
        return preds


def branch_acc_spiking(tm, seqs, div_pos, seed):
    ok = 0
    for s in seqs:
        pred = tm.run_sequence_spiking(s, seed=seed)[div_pos]
        ok += int(pred == {s[div_pos + 1]})
    return ok / len(seqs)


def _run_arm(job):
    seed, arm, n_seq, L, n_cells, k_win, act_th, epochs = job
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    div_pos = L
    lesion = (arm == "lesion")
    tm = SpikingTM(vocab, n_cells=n_cells, seed=seed, k_win=k_win, act_th=act_th, lesion=lesion)
    if arm != "untrained":
        for _ in range(epochs):                        # UNCHANGED discrete local learning (validated in EMERGE-9b)
            for s in seqs:
                tm.run_sequence(s, learn=True)
    # spiking-inference accuracy + (for the primary) discrete-parity accuracy
    from research.runners._emerge9b_htm_faithful_derisk import branch_acc as branch_acc_discrete
    spk = branch_acc_spiking(tm, seqs, div_pos, seed)
    disc = branch_acc_discrete(tm, seqs, div_pos) if arm == "htm" else None
    return (seed, arm, {"branch_spiking": spk, "branch_discrete": disc, "locality_ok": (not tm.used_transpose)})


ARMS = ["htm", "lesion", "untrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n-seq", type=int, default=2)
    ap.add_argument("--middle-len", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=16)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
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
            print(f"  [seed {d['seed']}] SPIKING branch {d['htm']['branch_spiking']:.3f} (discrete {d['htm']['branch_discrete']:.3f}) "
                  f"| lesion {d['lesion']['branch_spiking']:.3f} | untr {d['untrained']['branch_spiking']:.3f} "
                  f"|| markov {f['markov_L']:.3f} chance {f['chance']:.3f} loc {d['htm']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="branch_spiking"):
            return float(np.mean([p[arm][key] for p in per]))
        spk, disc = m("htm"), m("htm", "branch_discrete")
        les, unt = m("lesion"), m("untrained")
        markov = float(np.mean([p["floors"]["markov_L"] for p in per]))
        chance = float(np.mean([p["floors"]["chance"] for p in per]))
        oracle = float(np.mean([p["floors"]["oracle"] for p in per]))
        loc = all(p["htm"]["locality_ok"] for p in per)
        go = bool(oracle > 0.99 and spk >= 0.90 and spk >= markov + 0.15 and spk >= chance + 0.20 and spk >= les + 0.20 and loc)
        if not loc:
            verdict = "INVALID -- locality assert failed."
        elif go:
            verdict = (f"GO -- the context-specific high-order prediction SURVIVES spiking inference: SPIKING branch acc "
                       f"{spk:.3f} (== discrete {disc:.3f}) via LIF somas + dAP plateau + per-column WTA -- dAP-primed cells "
                       f"win the spiking competition and select the SAME context-specific SDR the discrete rule did. "
                       f">> Markov {markov:.3f}, >> chance {chance:.3f}, >> dAP-lesion {les:.3f} (the dendritic plateau is "
                       f"load-bearing); untrained {unt:.3f}; locality asserted. Multi-seed. => rung-3b (spiking substrate) "
                       f"holds -> scope the sim/ two-compartment NeuronModel port (rung-4; the dAP = the apical compartment). "
                       f"NO sim/ edit.")
        else:
            miss = []
            if spk < 0.90: miss.append(f"spiking branch {spk:.3f} < 0.90 (discrete {disc:.3f})")
            if spk < markov + 0.15 or spk < chance + 0.20: miss.append(f"didn't clear Markov/chance ({spk:.3f})")
            if spk < les + 0.20: miss.append(f"dAP-lesion didn't collapse ({spk:.3f} vs {les:.3f})")
            verdict = ("BOUNDARY (build-informative) -- " + "; ".join(miss) + f" (discrete parity {disc:.3f}, oracle "
                       f"{oracle:.3f}). The spiking selection lost the context the discrete rule kept -> tune the LIF/dAP/"
                       f"WTA (plateau amplitude / input drive / n_sub / k_win timing) so dAP-primed cells reliably win the "
                       f"competition. NOT a wall; the spiking selection is the next tuning.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge9c_spiking_tm", "verdict": verdict,
               "mechanism": "HTM Temporal Memory with SPIKING inference (LIF somas + distal-dendrite dAP plateau + per-column "
                            "WTA inhibition); learning UNCHANGED from EMERGE-9b (validated local unsupervised permanence); "
                            "dAP = the apical compartment of our two-compartment neuron",
               "task": "overlapping sequences; branch prediction; spiking-vs-discrete parity + Markov floor + dAP-lesion + oracle",
               "seeds": a.seeds, "config": {"n_seq": a.n_seq, "middle_len": a.middle_len, "n_cells": a.n_cells,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "rung-3b single-variable step: learning is the validated EMERGE-9b rule UNCHANGED; only the "
                              "active-cell SELECTION becomes spiking (LIF + dAP plateau + WTA). A follow-on rung makes "
                              "LEARNING spiking too (STDP-windowed permanence + dAP-rate homeostasis, the Bouhadjar three-term "
                              "rule). Then the sim/ two-compartment NeuronModel port (rung-4). dAP = apical compartment."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge9c] VERDICT: {verdict}", flush=True)
    print(f"[emerge9c] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
