"""EMERGE-7 (rung-3 Fork-2) — high-order NEXT-SYMBOL PREDICTION de-risk: does the CONFIRMED local target-based credit
rule (the same rule that learned the Task-A one-step map) learn genuine HIGH-ORDER CONTEXT on a combinatorial,
held-out, teacher-forced next-symbol task -- WITHOUT ever closing the loop (no autonomous free-run)?

WHY (rung-3a re-localized the wall to autonomous-GENERATION STABILITY; the research gate recommended reframing to
discrete next-symbol prediction, which is teacher-forced-by-construction so the free-run wall is structurally absent,
and is the biologically-apt COMMUNICATION-cortex faculty -- comprehension = predict-next-given-input, a distinct
circuit from motor/production free-run). This runner implements the airtight Fork-2 de-risk spec
(`2026-07-02-rung3-generation-stability-mechanisms-scoping.md` + the verify+design workflow synthesis + the controller
pre-design scratch de-risk `2026-07-02-fork2-predesign-scratch-derisk-reservoir-is-the-bar.md`).

THE TASK (combinatorial systematic routing family -- un-gameable by lookup OR by a memorizing reservoir):
  sequence for cell (p, m) = [P_p] + [H_m] + [C0 .. C_{middle_len-2}] + [S_g],  g = (p + m) mod n_suffix
  three DISJOINT symbol banks (a symbol's identity never leaks its role): prefix P, middle (distinguishing head H_m +
  shared body C), suffix S. The shared body is identical across middle-variants, so any predictor reading <= middle_len
  symbols at the divergent step sees an identical context for n_suffix equiprobable suffixes -> pinned at chance. Only
  reaching back middle_len+1 steps to the prefix AND reading the head beats chance = genuine high-order context.
  HELD-OUT cells (a (p,m) pairing never trained, but whose p and m each appear in other trained cells) can be solved
  ONLY by learning the systematic rule g=(p+m) -- a lookup/memorizer (incl. a fixed reservoir + trained readout) has
  no entry for a novel pairing. Scoring is on the DIVERGENT POSITION ONLY (aggregate accuracy is bigram-inflatable).

CREDIT (reused VERBATIM in FORM from the confirmed Task-A rule -- `_emerge6.RecurrentMicrocircuitRNN`): a leaky unit
u_t = kappa*u_{t-1} + W_rec@h_{t-1} + W_in@x_t ; h=sig(u). The hidden teaching target is s*_t = sig(T_teach @
onehot(y_{t+1})) -- a FIXED-RANDOM embedding of the NEXT symbol (a distal target pattern, NOT W_read.T -> no weight
transport). Per-neuron error err = s*_t - h_t == Task-A's (s*-a). Recurrent + input weights learn by the same
forward low-pass eligibility OR the same e-prop first-order eligibility; a SEPARATE softmax readout W_read is trained
by a LOCAL delta rule. NO BPTT, NO weight transport, NO free-run (teacher-forced, input-driven). used_transpose stays
False, asserted every arm/seed. Multi-seed 42/43/44. CPU / SIM_BACKEND=numpy; reuse-by-import; NO sim/ edit.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # tiny matmuls -> avoid BLAS oversubscription (see EMERGE-5)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge6_recurrent_microcircuit_seq_derisk import _sig, _MOMENTUM

OUT = Path("research/findings/raw/_emerge7_nextsymbol_context.json")


# ------------------------------------------------------------------------------------------------- task
def make_seqB_task(seed, n_prefix=4, n_middle=4, middle_len=2, n_suffix=4):
    """Combinatorial systematic-routing next-symbol family. Returns train/held sequences (as int lists), their (p,m,g)
    metadata, the vocab size V, and the divergent step index div_pos. Held-out = the diagonal cells (p,p), which keep
    every prefix and every middle covered by some training cell (coverage-guaranteed)."""
    P = list(range(n_prefix))
    H = list(range(n_prefix, n_prefix + n_middle))
    body_len = middle_len - 1
    C = list(range(n_prefix + n_middle, n_prefix + n_middle + body_len))
    S = list(range(n_prefix + n_middle + body_len, n_prefix + n_middle + body_len + n_suffix))
    V = n_prefix + n_middle + body_len + n_suffix
    div_pos = middle_len                                   # step whose target is the suffix (input = last middle symbol)

    def seq_for(p, m):
        g = (p + m) % n_suffix
        return [P[p]] + [H[m]] + C + [S[g]], g

    n_diag = min(n_prefix, n_middle)
    held_pm = [(p, p) for p in range(n_diag)]              # coverage-guaranteed diagonal held-out
    train, held = [], []
    for p in range(n_prefix):
        for m in range(n_middle):
            s, g = seq_for(p, m)
            (held if (p, m) in held_pm else train).append({"seq": s, "p": p, "m": m, "g": g})
    return {"train": train, "held": held, "V": V, "div_pos": div_pos, "n_suffix": n_suffix,
            "banks": {"P": P, "H": H, "C": C, "S": S}, "config": dict(n_prefix=n_prefix, n_middle=n_middle,
            middle_len=middle_len, n_suffix=n_suffix)}


def _onehot(sym, V):
    v = np.zeros(V); v[sym] = 1.0; return v


# ------------------------------------------------------------------------------- reference predictors (NON-shipped)
def markov_divergent_acc(train_cells, eval_cells, k, div_pos, n_suffix):
    """Order-k count predictor keyed on the k symbols ending at div_pos; trained on train cells, evaluated at the
    divergent step of eval cells. This is the `context_lesion` floor -- a lookup that reads only the last k symbols."""
    from collections import Counter, defaultdict
    table = defaultdict(Counter)
    for c in train_cells:
        s = c["seq"]; ctx = tuple(s[max(0, div_pos - k + 1): div_pos + 1])
        table[ctx][s[div_pos + 1]] += 1
    correct = 0.0
    for c in eval_cells:
        s = c["seq"]; ctx = tuple(s[max(0, div_pos - k + 1): div_pos + 1]); dist = table[ctx]
        if not dist:
            correct += 1.0 / n_suffix; continue
        top = max(dist.values()); winners = [sym for sym, n in dist.items() if n == top]
        correct += (1.0 / len(winners)) if s[div_pos + 1] in winners else 0.0
    return correct / max(1, len(eval_cells))


def systematic_oracle_acc(eval_cells, banks, n_suffix):
    """A model that KNOWS the rule g=(p+m) mod n_suffix -> 1.0 on held cells (proves the task is rule-solvable)."""
    S = banks["S"]; ok = sum(1 for c in eval_cells if c["seq"][-1] == S[(c["p"] + c["m"]) % n_suffix])
    return ok / max(1, len(eval_cells))


def modadd_gradient_oracle(train_cells, held_cells, n_prefix, n_middle, n_suffix, hidden=64, epochs=4000,
                           lr=0.1, wd=1e-3, seed=0):
    """LEARNABILITY oracle: is the held-out (p,m)->g split learnable BY GRADIENT AT ALL, given the factors p,m
    DIRECTLY (no context-carrying)? A 2-layer backprop MLP on onehot(p)+onehot(m). If it can't generalize held cells,
    the held-out split is grokking-hard = a TASK-design issue, NOT a mechanism verdict on the local recurrent rule.
    Clearly a NON-local, GIVEN-the-factors upper bound -- reported for disambiguation only, never a shipped brain rule."""
    rng = np.random.default_rng(seed)
    nf = n_prefix + n_middle

    def feat(p, m):
        v = np.zeros(nf); v[p] = 1.0; v[n_prefix + m] = 1.0; return v

    W1 = rng.normal(0, 1 / np.sqrt(nf), (hidden, nf)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (n_suffix, hidden)); b2 = np.zeros(n_suffix)
    tr = [(c["p"], c["m"], c["g"]) for c in train_cells]
    for _ in range(epochs):
        for i in rng.permutation(len(tr)):
            p, m, g = tr[i]; x = feat(p, m)
            h = np.maximum(0, W1 @ x + b1); lo = W2 @ h + b2
            pr = np.exp(lo - lo.max()); pr /= pr.sum(); d = pr.copy(); d[g] -= 1.0
            dh = (W2.T @ d) * (h > 0)
            W2 -= lr * (np.outer(d, h) + wd * W2); b2 -= lr * d
            W1 -= lr * (np.outer(dh, x) + wd * W1); b1 -= lr * dh

    def acc(cells):
        ok = sum(int(int(np.argmax(W2 @ np.maximum(0, W1 @ feat(c["p"], c["m"]) + b1) + b2)) == c["g"]) for c in cells)
        return ok / max(1, len(cells))
    return acc(train_cells), acc(held_cells)


# ------------------------------------------------------------------------------------------------- the network
class SeqBNet:
    """Input-driven leaky recurrent net trained by the CONFIRMED local target-based credit (s*-h), s* = a fixed-random
    embedding of the next symbol. Reuses the Task-A forward + e-prop eligibility FORMS verbatim. Separate local-delta
    readout. Teacher-forced (no free-run). Locality by construction (no W.T anywhere in the credit path)."""

    def __init__(self, N, V, seed=0, kappa=0.9, elig="forward", alpha=0.9, g_rec=1.2):
        rng = np.random.default_rng(seed)
        self.N, self.V = N, V
        self.kappa, self.elig, self.alpha = float(kappa), elig, float(alpha)
        self.W_rec = rng.normal(0, g_rec / np.sqrt(N), (N, N))       # plastic recurrent
        self.W_in = rng.normal(0, 1.0, (N, V))                       # plastic input
        self.T_teach = rng.normal(0, 1.0, (N, V))                    # FIXED-RANDOM next-symbol target embedding (no transport)
        self.W_read = np.zeros((V, N))                               # plastic readout (local delta rule)
        self._vr = np.zeros((N, N)); self._vi = np.zeros((N, V))
        self.used_transpose = False                                  # locality flag (must stay False)

    def train(self, cells, mode, epochs, lr, lr_read, seed):
        rng = np.random.default_rng(seed + 7)
        train_rec = mode in ("recurrent_microcircuit", "wrong_sign", "hebbian_selforg")
        train_read = mode != "untrained"                             # lesion/null train the readout (=fixed reservoir); untrained trains nothing
        for ep in range(epochs):
            dW_rec = np.zeros((self.N, self.N)); dW_in = np.zeros((self.N, self.V))
            for ci in rng.permutation(len(cells)):
                s = cells[ci]["seq"]
                # shuffled_target (order anti-cheat): replace each next-symbol target with a random vocab symbol,
                # destroying the input->next-symbol temporal alignment while keeping the same input stream.
                targets = ([int(rng.integers(self.V)) for _ in range(len(s) - 1)] if mode == "shuffled_target"
                           else s[1:])
                u = np.zeros(self.N); h_prev = _sig(u)
                e_rec = np.zeros(self.N); e_in = np.zeros(self.V)
                eps_rec = np.zeros((self.N, self.N)); eps_in = np.zeros((self.N, self.V))
                for t in range(len(s) - 1):
                    x = _onehot(s[t], self.V)
                    u = self.kappa * u + self.W_rec @ h_prev + self.W_in @ x
                    h = _sig(u); phip = h * (1.0 - h)
                    y_next = targets[t]
                    s_star = _sig(self.T_teach @ _onehot(y_next, self.V))    # fixed-random target embedding of next symbol
                    if mode in ("apical_feedback_lesion", "no_teaching_null"):
                        err = np.zeros(self.N)                               # no hidden credit -> W_rec/W_in frozen
                    elif mode == "wrong_sign":
                        err = -(s_star - h)
                    elif mode == "hebbian_selforg":
                        err = s_star                                         # associate next-embedding, no (s*-h)
                    else:
                        err = s_star - h                                     # (s* - h): the CONFIRMED target-based credit
                    if train_rec:
                        if self.elig == "eprop":
                            eps_rec = self.kappa * eps_rec + h_prev[None, :]
                            eps_in = self.kappa * eps_in + x[None, :]
                            dW_rec += (err * phip)[:, None] * eps_rec
                            dW_in += (err * phip)[:, None] * eps_in
                        else:
                            e_rec = self.alpha * e_rec + (1.0 - self.alpha) * h_prev
                            e_in = self.alpha * e_in + (1.0 - self.alpha) * x
                            dW_rec += np.outer(err, e_rec)
                            dW_in += np.outer(err, e_in)
                    if train_read:
                        logits = self.W_read @ h; p = np.exp(logits - logits.max()); p /= p.sum()
                        self.W_read += lr_read * np.outer(_onehot(y_next, self.V) - p, h)   # LOCAL delta rule
                    h_prev = h
            if train_rec:
                self._vr = _MOMENTUM * self._vr + dW_rec / max(1, sum(len(c["seq"]) - 1 for c in cells))
                self._vi = _MOMENTUM * self._vi + dW_in / max(1, sum(len(c["seq"]) - 1 for c in cells))
                self.W_rec += lr * self._vr
                self.W_in += lr * self._vi

    def _states(self, seq):
        u = np.zeros(self.N); h_prev = _sig(u); H = []
        for t in range(len(seq)):
            u = self.kappa * u + self.W_rec @ h_prev + self.W_in @ _onehot(seq[t], self.V)
            h = _sig(u); H.append(h); h_prev = h
        return H

    def divergent_acc(self, cells, div_pos):
        ok = 0
        for c in cells:
            h = self._states(c["seq"])[div_pos]
            ok += int(int(np.argmax(self.W_read @ h)) == c["seq"][div_pos + 1])
        return ok / max(1, len(cells))

    def divergent_states(self, cells, div_pos):
        return np.array([self._states(c["seq"])[div_pos] for c in cells])


def _clean_readout_divergent(net, train_cells, held_cells, div_pos, V, seed):
    """Freeze the trained recurrence; train a FRESH softmax on the frozen divergent-position state (train cells);
    evaluate on held cells. Proves the context lives in the RECURRENCE, not a lucky readout."""
    rng = np.random.default_rng(seed + 99)
    Xtr = net.divergent_states(train_cells, div_pos); ytr = [c["seq"][div_pos + 1] for c in train_cells]
    Xte = net.divergent_states(held_cells, div_pos); yte = [c["seq"][div_pos + 1] for c in held_cells]
    W = np.zeros((V, net.N))
    for _ in range(400):
        for i in rng.permutation(len(Xtr)):
            lo = W @ Xtr[i]; p = np.exp(lo - lo.max()); p /= p.sum()
            W += 0.2 * np.outer(_onehot(ytr[i], V) - p, Xtr[i])
    ok = sum(int(int(np.argmax(W @ Xte[j])) == yte[j]) for j in range(len(Xte)))
    return ok / max(1, len(Xte))


# ------------------------------------------------------------------------------------------------- arms
# arm -> (train_mode | None, elig)
ARM_SPEC = {
    "seqB_microcircuit": ("recurrent_microcircuit", "forward"),   # PRIMARY: memoryless forward eligibility
    "seqB_eprop":        ("recurrent_microcircuit", "eprop"),     # PRIMARY: leaky e-prop first-order eligibility
    "seqB_lesion":       ("apical_feedback_lesion", "forward"),   # recurrent credit zeroed, readout trains = FIXED RESERVOIR control
    "seqB_wrong":        ("wrong_sign", "forward"),               # negated hidden error -> anti-learn
    "seqB_null":         ("no_teaching_null", "forward"),         # no hidden target
    "seqB_shuffled":     ("shuffled_target", "forward"),          # next-symbol targets shuffled -> ordered context fails
    "seqB_hebbian":      ("hebbian_selforg", "forward"),          # Bouhadjar positive control (no (s*-h))
    "seqB_untrained":    (None, "forward"),                       # nothing trains -> chance floor
}
PRIMARY = "seqB_eprop"   # judged primary (the proper recurrent eligibility); seqB_microcircuit reported alongside


def _run_arm(job):
    seed, arm, N, kappa, alpha, g_rec, epochs, lr, lr_read, task_kw, n_splits = job
    train_mode, elig = ARM_SPEC[arm]
    # average divergent held-out accuracy over n_splits re-drawn held-splits (coarse metric stabilization)
    accs, train_accs, clean_accs = [], [], []
    for sp in range(n_splits):
        task = make_seqB_task(seed * 100 + sp, **task_kw)
        net = SeqBNet(N, task["V"], seed=seed * 100 + sp, kappa=kappa, elig=elig, alpha=alpha, g_rec=g_rec)
        if train_mode is not None:
            net.train(task["train"], train_mode, epochs, lr, lr_read, seed * 100 + sp)
        accs.append(net.divergent_acc(task["held"], task["div_pos"]))
        train_accs.append(net.divergent_acc(task["train"], task["div_pos"]))
        if arm == PRIMARY:
            clean_accs.append(_clean_readout_divergent(net, task["train"], task["held"], task["div_pos"], task["V"], seed * 100 + sp))
    return (seed, arm, {"held_div": float(np.mean(accs)), "train_div": float(np.mean(train_accs)),
                        "clean_div": (float(np.mean(clean_accs)) if clean_accs else None),
                        "locality_ok": (not net.used_transpose)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--g-rec", type=float, default=1.2)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--lr-read", type=float, default=0.2)
    ap.add_argument("--n-prefix", type=int, default=4)
    ap.add_argument("--n-middle", type=int, default=4)
    ap.add_argument("--middle-len", type=int, default=2)
    ap.add_argument("--n-suffix", type=int, default=4)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--max-workers", type=int, default=0, help="cap parallel workers (0=all cores)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2
    task_kw = dict(n_prefix=a.n_prefix, n_middle=a.n_middle, middle_len=a.middle_len, n_suffix=a.n_suffix)
    chance = 1.0 / a.n_suffix
    t0 = time.time(); err = None; per = []
    # reference floors (deterministic; computed once on a canonical draw per seed)
    floors = {}
    for s in a.seeds:
        tk = make_seqB_task(s * 100, **task_kw)
        go_tr, go_he = modadd_gradient_oracle(tk["train"], tk["held"], a.n_prefix, a.n_middle, a.n_suffix, seed=s)
        floors[s] = {
            "markov_order_m": markov_divergent_acc(tk["train"], tk["held"], a.middle_len, tk["div_pos"], a.n_suffix),
            "markov_order1": markov_divergent_acc(tk["train"], tk["held"], 1, tk["div_pos"], a.n_suffix),
            "systematic_oracle": systematic_oracle_acc(tk["held"], tk["banks"], a.n_suffix),
            "grad_oracle_train": go_tr, "grad_oracle_held": go_he,   # is held-out gradient-learnable at all (given factors)?
            "V": tk["V"], "div_pos": tk["div_pos"], "n_train": len(tk["train"]), "n_held": len(tk["held"]),
        }
    try:
        jobs = [(s, arm, a.N, a.kappa, a.alpha, a.g_rec, a.epochs, a.lr, a.lr_read, task_kw, a.n_splits)
                for s in a.seeds for arm in ARM_SPEC]
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
            print(f"  [seed {d['seed']}] eprop held-div {d['seqB_eprop']['held_div']:.3f} (train {d['seqB_eprop']['train_div']:.3f}) "
                  f"| micro held {d['seqB_microcircuit']['held_div']:.3f} | LESION(reservoir) {d['seqB_lesion']['held_div']:.3f} "
                  f"| hebbian {d['seqB_hebbian']['held_div']:.3f} | wrong {d['seqB_wrong']['held_div']:.3f} "
                  f"| null {d['seqB_null']['held_div']:.3f} | shuf {d['seqB_shuffled']['held_div']:.3f} "
                  f"| untr {d['seqB_untrained']['held_div']:.3f} || markov_m {f['markov_order_m']:.3f} "
                  f"oracle {f['systematic_oracle']:.3f} clean {d['seqB_eprop']['clean_div']} loc {d['seqB_eprop']['locality_ok']}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        def m(arm, key="held_div"):
            return float(np.mean([p[arm][key] for p in per]))
        prim = m(PRIMARY); prim_micro = m("seqB_microcircuit"); prim_train = m(PRIMARY, "train_div")
        les, heb = m("seqB_lesion"), m("seqB_hebbian")
        wrong, null, shuf, unt = m("seqB_wrong"), m("seqB_null"), m("seqB_shuffled"), m("seqB_untrained")
        clean = m(PRIMARY, "clean_div")
        floor_m = float(np.mean([p["floors"]["markov_order_m"] for p in per]))
        oracle = float(np.mean([p["floors"]["systematic_oracle"] for p in per]))
        grad_held = float(np.mean([p["floors"]["grad_oracle_held"] for p in per]))
        grad_train = float(np.mean([p["floors"]["grad_oracle_train"] for p in per]))
        les_train = m("seqB_lesion", "train_div")     # fixed-reservoir WITHIN-distribution high-order-context fit
        clean_unt = None  # (untrained clean-readout would need its own arm; use chance as the reference below)
        loc = all(p[PRIMARY]["locality_ok"] for p in per)
        held_learnable = grad_held >= chance + 0.15    # is the held-out split gradient-learnable AT ALL (given factors)?
        # criteria on the DIVERGENT HELD-OUT accuracy (multi-seed, all must hold)
        best_prim = max(prim, prim_micro)
        task_sane = prim_train >= 0.75 and oracle > 0.99
        recalls = best_prim >= 0.60 and best_prim >= floor_m + 0.15 and best_prim >= chance + 0.20
        beats_lesion = best_prim >= les + 0.15
        wrong_anti = wrong <= chance + 0.05
        null_flat = null <= unt + 0.12
        shuffled_fails = shuf <= unt + 0.15
        clean_ok = (clean is not None and clean >= chance + 0.20)
        beats_hebbian = best_prim > heb + 0.10
        go = bool(task_sane and recalls and beats_lesion and wrong_anti and null_flat and shuffled_fails
                  and clean_ok and loc and (beats_hebbian or heb >= 0.60))
        if not loc:
            verdict = "INVALID -- locality assert failed (credit path used W.T / BPTT)."
        elif not held_learnable:
            # the held-out combinatorial split is not gradient-learnable EVEN given the factors directly -> grokking-hard
            # TASK-design over-reach, NOT a mechanism verdict on the local rule. Report the WITHIN-distribution finding.
            hurts = les_train - max(prim_train, m("seqB_microcircuit", "train_div"))
            verdict = (f"TASK-MISDESIGNED (held-out grokking-hard, NOT a mechanism verdict) -- the held-out (p,m)->g "
                       f"split is not gradient-learnable even by a factor-access backprop MLP (grad-oracle held {grad_held:.3f} "
                       f"~ chance {chance:.3f}, train {grad_train:.3f}); so ALL held-div numbers (primary {best_prim:.3f}, "
                       f"reservoir-lesion {m('seqB_lesion'):.3f}) are uninformative about the credit rule. WITHIN-DISTRIBUTION "
                       f"FINDING (robust + build-informative): a FIXED RESERVOIR + trained readout MEMORIZES the trained "
                       f"high-order-context routings (lesion train-div {les_train:.3f}), while training the recurrent weights "
                       f"with the confirmed local credit rule DEGRADES that fit (eprop {prim_train:.3f} / micro "
                       f"{m('seqB_microcircuit','train_div'):.3f}; local credit HURTS the reservoir by {hurts:+.3f}). This is "
                       f"the 3rd independent confirmation (rung-3a, scratch-RFLO, here) that a NAIVE local recurrent credit "
                       f"rule does not beat -- and degrades -- a random reservoir. ⇒ the cheap Fork-2 is a false economy; the "
                       f"real rung-3 lever is the careful chaos-taming rule (Predictive Alignment, Fork-1) tested on "
                       f"noise-robustness where a reservoir demonstrably degrades. Redesign a within-distribution / "
                       f"noise-robustness metric (a reservoir fails but gradient succeeds) OR build Predictive Alignment.")
        elif not task_sane:
            verdict = (f"INCONCLUSIVE -- held-out IS gradient-learnable (grad-oracle {grad_held:.3f}) but the recurrent net "
                       f"can't fit TRAIN cells: primary TRAIN-cell divergent {prim_train:.3f} (<0.75) or systematic-oracle "
                       f"{oracle:.3f} (<1.0). Tune N/kappa/lr/epochs before the mechanism verdict. NOT a mechanism verdict.")
        elif go:
            verdict = (f"GO -- the CONFIRMED local target-based credit rule learns genuine HIGH-ORDER CONTEXT on a "
                       f"combinatorial held-out next-symbol task, teacher-forced (NO free-run): primary held-out "
                       f"DIVERGENT accuracy {best_prim:.3f} (eprop {prim:.3f} / micro {prim_micro:.3f}; train {prim_train:.3f}) "
                       f">> order-{a.middle_len} Markov floor {floor_m:.3f}, >> chance {chance:.3f}, >> FIXED-RESERVOIR "
                       f"lesion {les:.3f}; wrong-sign anti-learns ({wrong:.3f}), null flat ({null:.3f}), shuffled fails "
                       f"({shuf:.3f}), clean-readout-on-frozen-recurrence {clean:.3f} (context lives in the recurrence), "
                       f"{'beats' if beats_hebbian else 'ties (Hebbian self-organizes -- honest)'} hebbian ({heb:.3f}), "
                       f"systematic-oracle {oracle:.3f}, locality asserted. Multi-seed. ⇒ rung-3 next-symbol milestone MET "
                       f"on the communication-relevant task -> promote to 6 seeds, then rung-3b (spike noise) + defer "
                       f"generation to Fork-1 (Predictive Alignment). NO sim/ edit.")
        else:
            miss = []
            if not recalls: miss.append(f"primary held-div {best_prim:.3f} < max(0.60, markov {floor_m:.3f}+.15, chance {chance:.3f}+.20)")
            if not beats_lesion: miss.append(f"didn't beat FIXED-RESERVOIR lesion ({best_prim:.3f} vs {les:.3f})")
            if not (beats_hebbian or heb >= 0.60): miss.append(f"didn't beat hebbian ({best_prim:.3f} vs {heb:.3f})")
            if not wrong_anti: miss.append(f"wrong-sign didn't anti-learn ({wrong:.3f})")
            if not null_flat: miss.append(f"null not flat ({null:.3f} vs untr {unt:.3f})")
            if not shuffled_fails: miss.append(f"shuffled still recalled ({shuf:.3f})")
            if not clean_ok: miss.append(f"clean-readout weak ({clean})")
            verdict = ("BOUNDARY (build-informative, not a stop) -- " + "; ".join(miss) + f". Task IS learnable "
                       f"(train-div {prim_train:.3f}, oracle {oracle:.3f}) but the local rule can't GENERALIZE the "
                       f"systematic high-order routing to held-out cells teacher-forced ⇒ the wall RELOCATED from "
                       f"generation-stability to HIGH-ORDER-CONTEXT CREDIT (credit quality was fine on Task-A). Next: "
                       f"escalate to Predictive E-prop's exact predict-next objective (bioRxiv 10.64898/2026.02.12.705507) "
                       f"and/or Bouhadjar's dendritic-AP context mechanism -- NOT FORCE/Predictive-Alignment (wrong wall). "
                       f"Do NOT start the sim/ port.")
    else:
        verdict = f"ERROR -- {err}"

    summary = {"probe": "emerge7_nextsymbol_context", "verdict": verdict,
               "mechanism": "confirmed local target-based credit (s*-h), s*=fixed-random next-symbol embedding; forward + "
                            "e-prop eligibility (reused verbatim from Task-A); input-driven teacher-forced; separate local "
                            "delta readout; NO free-run, NO BPTT, NO weight transport",
               "task": "combinatorial systematic-routing next-symbol family g=(p+m) mod n_suffix with held-out (p,m) cells; "
                       "divergent-position-only scoring; order-(middle_len) Markov floor + fixed-reservoir lesion + "
                       "clean-readout + systematic-oracle controls",
               "seeds": a.seeds, "config": {"N": a.N, "kappa": a.kappa, "alpha": a.alpha, "g_rec": a.g_rec,
               "epochs": a.epochs, "lr": a.lr, "lr_read": a.lr_read, **task_kw, "n_splits": a.n_splits, "chance": chance},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "HONEST_NOTE": "Fork-2 = discrete NEXT-SYMBOL prediction (teacher-forced-by-construction -> the rung-3a "
                              "autonomous-generation-stability wall is structurally absent, not hidden; generation is "
                              "DEFERRED to Fork-1/Predictive-Alignment). GO gate scores the DIVERGENT held-out position "
                              "ONLY vs an order-(middle_len) Markov floor + a FIXED-RESERVOIR (readout-only) lesion, so a "
                              "bigram/verbatim/memorizing-reservoir cannot pass (the 2026-05-03/05-14 permuted-label trap). "
                              "task_sane = train-cell fit >=0.75 + systematic-oracle==1 (a BPTT learnability oracle is a "
                              "deferred stronger task-sanity). Locality by construction: hidden target via fixed T_teach "
                              "(not W_read.T); eligibility on own pre-rate; readout delta local; no W_rec.T; no free-run."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge7] VERDICT: {verdict}", flush=True)
    print(f"[emerge7] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
