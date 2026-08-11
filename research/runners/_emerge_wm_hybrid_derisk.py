"""EMERGENCE-ENGINE + WORKING-MEMORY HYBRID DE-RISK — does WIRING the variable-binding WM slot INTO the on-bridge HTM
Temporal-Memory emergence engine give the COMBINED spiking system a faculty NEITHER piece has alone?

TWO BANKED, VERIFIED FACTS DEFINE THIS DE-RISK (read the findings; do NOT re-derive):
  (1) The emergence engine (on-bridge HTM-TM) MEMORISES but does NOT ABSTRACT. On an agreement stream
      [subject]+[L random fillers]+[verb], held-out branch(verb) collapses to 0.000 the moment intervening fillers
      vary — it cannot latch a long-range latent variable across novel fillers.
      (2026-08-11-emergence-engine-stream-language-htm-memorises-not-generalises-...-SMOKE.md)
  (2) The BG-gated slow-NMDA bistable HOLD slot + content-agnostic Hebbian bind LATCHES exactly that variable and
      carries it across NOVEL fillers (held-out 1.000). Spiking + load-bearing (lesion-the-hold collapses it).
      (2026-08-11-variable-binding-working-memory-gated-slot-surpasses-HTM-...-6seed-GO.md)

THE OPEN QUESTION: build a genuinely COMPOSITIONAL stream where the predicted token depends on BOTH
  * a LONG-RANGE latent variable (the SUBJECT, L+1 tokens back) — only the WM can carry it across varying fillers; AND
  * a LOCAL sequence property computable on NOVEL fillers (the CLASS/half of the LAST filler token) — the HTM can learn
    it locally and it generalises to filler tuples unseen in training (the class is a fixed function of the token).
  verb = combine(subject, local_class(last filler)) = one of n_subj*n_cls verb tokens.
  So WM-alone knows the subject but not the local class -> caps at ~1/n_cls; HTM-alone recovers the local class from
  the last filler but LOSES the subject across novel fillers -> caps at ~1/n_subj; only a system with BOTH can reach
  ceiling. This is the "the WM faculty is load-bearing on conversation" test.

THE HYBRID — THE COMBINATION IS NEURAL (brain-based-only; the crux). NOT a host argmax/ensemble of two answers.
  The WM slot's HELD subject enters the emergence engine as an EXTRA CORTICAL AFFERENT: the substrate has n_subj
  dedicated WM-MEMORY columns (neurons in the SAME bridge). At the verb-decision step the slot is READ from its SPIKES
  (argmax pool firing over cp_firing_states = a WTA over the persistent-activity attractor), and the winning pool's
  labelled-line projection co-activates its WM-memory column IN THE HTM's coincidence context. The engine's OWN branch
  prediction is then the read-out: the verb cell fires only when it receives COINCIDENT distal drive from BOTH the
  last-filler afferent AND the WM-memory afferent — a genuine dendritic coincidence-AND (measured: the bridge's apical
  plateau is binary above threshold, so a coincidence threshold set BETWEEN single-pathway drive (~k_win) and
  double-pathway drive (~2*k_win) fires ONLY on the conjunction). No host code compares a "WM answer" to an "HTM answer";
  the WM's subject enters purely as which cortical column is co-active, and the verb prediction EMERGES from the
  substrate's coincidence of the two co-active populations. Evidence it reads SPIKES: (a) the afferent's subject comes
  from the slot's cp_firing_states argmax; (b) lesion-the-hold (recur=0) kills the spiking sustain -> the afferent is
  wrong -> the hybrid collapses; (c) cp_external_input_current is asserted identically ZERO across the hold+read span.

ARMS / ANTI-CHEATS (all REQUIRED; honest-negative is a first-class deliverable):
  * htm_alone       : the emergence engine on the compositional stream (no WM afferent). Expect: recovers the LOCAL
                      class (~1.0) but loses the SUBJECT across novel fillers -> exact ~ 1/n_subj.
  * wm_alone        : the spiking slot's held subject read out directly (verb = subject's most-likely verb). Expect:
                      recovers the SUBJECT (~1.0) but cannot see the local class -> exact ~ 1/n_cls.
  * hybrid          : WM afferent -> HTM engine; the engine's own branch prediction. The test.
  * lesion_wm_aff   : hybrid with the WM afferent ablated at read (filler afferent only) -> MUST collapse (>= to htm_alone).
  * lesion_htm_aff  : hybrid with the filler afferent ablated at read (WM afferent only) -> MUST collapse (>= to wm_alone).
  * lesion_hold     : hybrid with the slot's recur=0 (the hold dies) -> the spiking afferent is garbage -> MUST collapse.
                      (proves the afferent READS the slot's spikes, not a host store.)
  * subj_shuffle    : the slot->subject deref is permuted -> the WM afferent points to the WRONG subject column ->
                      subject part collapses to chance (no positional/topic leakage; the carried VALUE is load-bearing).
  * held-out NOVEL fillers (disjoint tuples), n-gram HELD-OUT floor, chance, and a task-solvable oracle (=1 by build).

GO GATE: hybrid held-out exact >= max(htm_alone, wm_alone) + 0.20 AND >= chance + 0.30, generalising to NOVEL fillers,
  with BOTH afferent lesions load-bearing (each collapses the hybrid to/below the corresponding single-system baseline)
  AND lesion-the-hold collapsing (the fusion reads spikes). HONEST NEGATIVE (first-class) if the neural fusion does not
  beat both: report the precise reason (is the coincidence-AND lossy? does the HTM ignore the afferent? is the local
  property unlearnable?).

Reuse-by-import: the EMERGE-14 on-bridge learner (build_pool_bridge / OnBridgeLearner / apply_kernel_update /
coincidence_predict), the compositional stream is a local extension of _emerge_stream_language_derisk's layout, the
spiking WM slot + marker gate + persistent subject binder from _var_bind_gated_slot_derisk. NO sim/ edit.
SIM_BACKEND=numpy (sub-1k-neuron coincidence + attractor loops are launch-bound; CPU is correct + faster).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_derisk --seeds 42 --debug
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_derisk --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_emerge_wm_hybrid/hybrid_6seed.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import time
import traceback
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

# --- the EMERGE-14 on-bridge HTM Temporal-Memory engine (the emergence engine) ---
from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, OnBridgeLearner, apply_kernel_update, coincidence_predict, _host)
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
# --- the banked spiking WM: slow-NMDA bistable HOLD slot + persistent subject binder + marker gate ---
from research.runners._var_bind_gated_slot_derisk import (
    SpikingSlot, build_codebook, persistent_subject_binder)
from research.runners._novel_referent_hebbian_fastweight_derisk import _K

try:
    from tools.lab import lever, attributable_to, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond: print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_emerge_wm_hybrid/hybrid.json")


# ====================================================================================================================
# The COMPOSITIONAL stream. Columns: subjects | fillers | verbs (n_subj*n_cls) | WM-memory (n_subj).
#   sentence = [subject_i] + [L i.i.d. fillers] + [verb(i, class(last_filler))]
#   class(f) = which contiguous block of the filler pool the token f is in (a fixed function of f -> generalises).
# ====================================================================================================================
def hybrid_layout(n_subj, n_fill, n_cls):
    subj = list(range(n_subj))
    fill = list(range(n_subj, n_subj + n_fill))
    nv = n_subj * n_cls
    verb = list(range(n_subj + n_fill, n_subj + n_fill + nv))
    wm = list(range(n_subj + n_fill + nv, n_subj + n_fill + nv + n_subj))
    V = n_subj + n_fill + nv + n_subj
    return subj, fill, verb, wm, V


def cls_of(fill_tok, n_subj, n_fill, n_cls):
    off = int(fill_tok) - n_subj
    return min(n_cls - 1, off * n_cls // n_fill)


def verb_of(verb_list, subj_idx, cls, n_cls):
    return verb_list[subj_idx * n_cls + cls]


def decode_verb(verb_list, verb_tok, n_cls):
    """verb token -> (subject_idx, class). -1 -> (-1,-1)."""
    if verb_tok < 0 or verb_tok not in verb_list:
        return -1, -1
    k = verb_list.index(verb_tok)
    return k // n_cls, k % n_cls


def make_comp_stream(n_subj, n_fill, n_cls, L, n_sent, rng):
    subj, fill, verb, wm, V = hybrid_layout(n_subj, n_fill, n_cls)
    seqs = []
    for _ in range(n_sent):
        i = int(rng.integers(0, n_subj))
        ftuple = [int(fill[rng.integers(0, n_fill)]) for _ in range(L)]
        cls = cls_of(ftuple[-1], n_subj, n_fill, n_cls)
        seqs.append([subj[i]] + ftuple + [verb_of(verb, i, cls, n_cls)])
    return seqs


def make_comp_heldout(n_subj, n_fill, n_cls, L, n_sent, rng, train_ftuples, max_tries_mult=80):
    """Held-out sentences whose filler tuples are DISJOINT from training (true novel-filler generalisation)."""
    subj, fill, verb, wm, V = hybrid_layout(n_subj, n_fill, n_cls)
    seqs, tries = [], 0
    while len(seqs) < n_sent and tries < n_sent * max_tries_mult:
        tries += 1
        i = int(rng.integers(0, n_subj))
        ftuple = tuple(int(fill[rng.integers(0, n_fill)]) for _ in range(L))
        if ftuple in train_ftuples:
            continue
        cls = cls_of(ftuple[-1], n_subj, n_fill, n_cls)
        seqs.append([subj[i]] + list(ftuple) + [verb_of(verb, i, cls, n_cls)])
    gen_defined = len(seqs) >= max(20, 4 * n_subj)
    return seqs, gen_defined


def ngram_floor_comp(train, test, L, n_verb_chance_subj):
    """BEST fixed-order n-gram HELD-OUT floor on the compositional stream. Order-1 (last filler) recovers the local
    CLASS (a fixed fn of the last token) but not the subject -> pinned near 1/n_subj; higher orders back off to chance
    on novel tuples. Returns (best_exact, best_order). (chance for the FULL verb is 1/(n_subj*n_cls).)"""
    best, best_k = 0.0, 0
    for k in range(1, L + 2):
        counts = defaultdict(Counter)
        for s in train:
            t = len(s) - 2
            ctx = tuple(s[max(0, t - k + 1): t + 1])
            counts[ctx][s[t + 1]] += 1
        ok = 0.0
        for s in test:
            t = len(s) - 2
            ctx = tuple(s[max(0, t - k + 1): t + 1])
            dist = counts.get(ctx)
            if not dist:
                ok += 1.0 / n_verb_chance_subj
                continue
            top = max(dist.values()); win = [x for x, n in dist.items() if n == top]
            ok += (1.0 / len(win)) if s[t + 1] in win else 0.0
        acc = ok / max(1, len(test))
        if acc > best:
            best, best_k = acc, k
    return best, best_k


# ====================================================================================================================
# On-bridge emergence engine — hybrid train / predict. Reuses the OnBridgeLearner primitives; injects the WM afferent
# as an EXTRA pre-synaptic population at the verb-prediction step (a dendritic coincidence-AND, not a host ensemble).
# ====================================================================================================================
def apical_drive(b, cells_idx, active_cells):
    """Run the bridge's WEIGHTED coincidence recurrence from `active_cells` and return per-cell apical plateau voltage
    (the SUBSTRATE's own prediction signal; None under the dAP lesion). Same primitive coincidence_predict uses."""
    if getattr(b, "cp_v_apical", None) is None and not b.core_config.enable_coincidence_detection:
        return None
    ab = np.zeros(len(cells_idx), bool)
    for i in active_cells:
        ab[i] = True
    _prime_from_winners(b, cells_idx, ab)
    _vap = getattr(b, "cp_v_apical", None)
    if _vap is None or np.asarray(_host(_vap)).ndim == 0:
        return None
    return np.asarray(_host(_vap))[cells_idx]


def hybrid_train_sequence(lr, seq, L, wm_cells, use_wm):
    """Train one compositional sentence. Identical to OnBridgeLearner.train_sequence EXCEPT: at the verb step the
    pre-synaptic set for the kernel update (and the winner-match scoring) is AUGMENTED with the WM-memory column's
    cells, so the fused kernel potentiates BOTH last-filler->verb and WM-memory->verb onto the SAME verb winners."""
    predictive, prev_winners = set(), set()
    for pos, c in enumerate(seq):
        is_verb = (pos == L + 1)
        pre = set(prev_winners)
        if is_verb and use_wm and wm_cells:
            pre |= set(wm_cells)
        col = lr._col(c)
        primed = [i for i in col if i in predictive] if not lr.lesion else []
        if primed:
            winners = set(primed[:lr.k_win])
        elif not prev_winners:
            winners = set(col[:lr.k_win])
        else:
            scored = sorted(((lr._match_count(i, pre), i) for i in col), reverse=True)
            if scored[0][0] >= lr.learn_th:
                winners = set(i for sc, i in scored[:lr.k_win] if sc >= lr.learn_th)
            else:
                wc = lr._committed_count()
                winners = set(sorted(col, key=lambda i: (wc[i], i))[:lr.k_win])
        if pre:
            apply_kernel_update(lr.b, lr.row, lr.col, lr.cells_idx, pre, winners,
                                lr.z, lr.lam_pot, lr.lam_dep, lr.z_star)
        active = winners if primed else (set(col) if prev_winners or not primed else winners)
        predictive = coincidence_predict(lr.b, lr.cells_idx, active, lr.N, lr.nE)
        lr.z *= lr.z_tau
        for i in predictive:
            lr.z[i] += (1.0 - lr.z_tau)
        prev_winners = winners


def hybrid_predict_verb(lr, seq, L, wm_cells, verb_cols, use_htm=True, use_wm=True, thr_off=2.0, return_drive=False):
    """Forward the priming chain through subject+fillers, then at the verb-decision step fuse the requested afferents
    (filler burst and/or WM-memory) and read the engine's coincidence prediction: the verb column with the highest
    supra-threshold apical drive (or -1). This IS the emergence engine's own branch prediction, conditioned on the
    afferents. use_htm=False -> WM afferent only; use_wm=False -> filler afferent only (the two afferent lesions)."""
    E_rest = float(getattr(lr.b.core_config, "apical_E_rest", -65.0))
    thr = E_rest + thr_off
    predictive = set()
    for pos in range(L + 1):                                  # positions 0..L : subject + L fillers
        c = seq[pos]
        col = lr._col(c)
        primed = [i for i in col if i in predictive] if not lr.lesion else []
        active = set(primed[:lr.k_win]) if primed else set(col)
        if pos == L:                                          # the verb-decision step: fuse afferents, read prediction
            fused = set()
            if use_htm:
                fused |= active
            if use_wm and wm_cells:
                fused |= set(wm_cells)
            drive = apical_drive(lr.b, lr.cells_idx, fused)
            if drive is None:
                return (-1, {}) if return_drive else -1
            col_drive = {vt: float(np.max(drive[lr._col(vt)])) for vt in verb_cols}
            best_tok, best_d = -1, thr
            for vt, d in col_drive.items():
                if d > best_d:
                    best_d, best_tok = d, vt
            return (best_tok, col_drive) if return_drive else best_tok
        predictive = coincidence_predict(lr.b, lr.cells_idx, active, lr.N, lr.nE)
    return (-1, {}) if return_drive else -1


def build_engine(seed, V, n_cells, act_th, k_win, learn_th):
    b, cells_idx, row, col = build_pool_bridge(V, n_cells, seed, act_th=act_th, coincidence=True)
    lr = OnBridgeLearner(b, row, col, cells_idx, V, n_cells, k_win=k_win, act_th=act_th, learn_th=learn_th)
    return lr


def train_engine_hybrid(lr, train_seqs, L, wm_col_of_subj_idx, subj_list, use_wm, epochs):
    """Train the engine. The WM afferent during training is driven by the TRUE subject (justified: the banked spiking
    slot latches the subject with ~1.0 accuracy — this is the WM working during development). The decisive spiking-slot
    read is used at TEST (carrying the subject across NOVEL fillers). use_wm=False -> the standalone HTM (no afferent)."""
    for _ in range(epochs):
        for s in train_seqs:
            subj_idx = subj_list.index(s[0])
            wm_cells = lr._col(wm_col_of_subj_idx[subj_idx])[:lr.k_win] if use_wm else None
            hybrid_train_sequence(lr, s, L, wm_cells, use_wm)


# ====================================================================================================================
# The spiking WM slot: marker gate latches the subject at t=0, holds (zero input) across the filler span, reads (spikes).
# ====================================================================================================================
def slot_carry_subject(slot: SpikingSlot, seq, slot_of_subj, subj_of_slot):
    """Marker-gated spiking carry: WRITE the subject's attractor pool at t=0, HOLD (external input asserted zero) across
    every filler, then READ the latched pool from cp_firing_states (argmax pool rate = a WTA over the attractor).
    Returns (subject_token_hat, hold_alive, zero_input_ok)."""
    toks = seq[:-1]                                            # subject + fillers (exclude the verb target)
    slot.reset()
    for t, tok in enumerate(toks):
        if t == 0:
            slot.write(slot_of_subj[int(tok)])                # LOAD the subject (gate OPEN)
        else:
            slot.hold()                                       # HOLD (gate CLOSED, zero external drive)
    shat, alive = slot.read()                                 # READ from spikes
    subj_tok = subj_of_slot.get(shat, -1)
    return subj_tok, float(alive), bool(slot._zero_input_span)


# ====================================================================================================================
# Metrics
# ====================================================================================================================
def score_preds(pred_verbs, test_seqs, verb_list, n_cls):
    """pred_verbs: list of predicted verb TOKENS (aligned to test_seqs). Returns exact/subject/class accuracy."""
    ex = sub = cl = 0
    n = max(1, len(test_seqs))
    for pv, s in zip(pred_verbs, test_seqs):
        ts, tc = decode_verb(verb_list, s[-1], n_cls)
        ps, pc = decode_verb(verb_list, pv, n_cls)
        ex += int(pv == s[-1])
        sub += int(ps == ts and ps >= 0)
        cl += int(pc == tc and pc >= 0)
    return ex / n, sub / n, cl / n


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, n_subj, n_fill, n_cls, L, n_cells, k_win, act_th, and_th, learn_th, epochs, n_train, n_test,
              recur, hold_steps, load_steps, clear_steps, debug=False):
    rng = np.random.default_rng(seed)
    subj, fill, verb, wm, V = hybrid_layout(n_subj, n_fill, n_cls)
    n_verb = n_subj * n_cls
    chance = 1.0 / n_verb
    wm_col_of_subj_idx = {i: wm[i] for i in range(n_subj)}

    # --- the compositional stream + disjoint NOVEL-filler held-out test ---
    train_seqs = make_comp_stream(n_subj, n_fill, n_cls, L, n_train, rng)
    train_ftuples = set(tuple(s[1:-1]) for s in train_seqs)
    test_seqs, gen_defined = make_comp_heldout(n_subj, n_fill, n_cls, L, n_test, rng, train_ftuples)

    # --- the spiking WM: codebook + persistent subject binder (reuse-by-import of the banked pieces) ---
    code_of, _ = build_codebook(n_subj, n_fill, np.random.default_rng(seed + 7))
    binder, slot_of_subj, subj_of_slot = persistent_subject_binder([int(s) for s in subj], code_of)

    def _slot(rc=recur, off=0):
        return SpikingSlot(seed + off, _K, recur=rc, hold_steps=hold_steps, load_steps=load_steps,
                           clear_steps=clear_steps)

    # --- engines: standalone HTM (normal threshold) + hybrid (coincidence-AND threshold) ---
    lr_htm = build_engine(seed, V, n_cells, act_th, k_win, learn_th)
    train_engine_hybrid(lr_htm, train_seqs, L, wm_col_of_subj_idx, subj, use_wm=False, epochs=epochs)

    lr_hyb = build_engine(seed, V, n_cells, and_th, k_win, learn_th)
    train_engine_hybrid(lr_hyb, train_seqs, L, wm_col_of_subj_idx, subj, use_wm=True, epochs=epochs)

    # --- the spiking slot's held-subject read for every test sentence (intact + hold-lesion) ---
    slot = _slot()
    slot_les = _slot(rc=0.0, off=100)
    subj_hat, hold_alive, zero_ok = [], [], True
    subj_hat_les = []
    for s in test_seqs:
        sh, al, zo = slot_carry_subject(slot, s, slot_of_subj, subj_of_slot)
        subj_hat.append(sh); hold_alive.append(al); zero_ok = zero_ok and zo
        shl, _, _ = slot_carry_subject(slot_les, s, slot_of_subj, subj_of_slot)
        subj_hat_les.append(shl)
    slot_decode_acc = float(np.mean([int(sh == s[0]) for sh, s in zip(subj_hat, test_seqs)]))

    def wm_cells_from(subj_tok):
        if subj_tok in subj:
            return lr_hyb._col(wm[subj.index(int(subj_tok))])[:k_win]
        return None

    # ---- ARM predictions on held-out ----
    # htm_alone (standalone engine; no WM afferent)
    p_htm = [hybrid_predict_verb(lr_htm, s, L, None, verb, use_htm=True, use_wm=False) for s in test_seqs]

    # hybrid (WM afferent from the SPIKING slot read + filler afferent)
    p_hyb = [hybrid_predict_verb(lr_hyb, s, L, wm_cells_from(sh), verb, use_htm=True, use_wm=True)
             for s, sh in zip(test_seqs, subj_hat)]

    # lesion the WM afferent (filler only) — must collapse >= to htm_alone
    p_les_wm = [hybrid_predict_verb(lr_hyb, s, L, None, verb, use_htm=True, use_wm=False) for s in test_seqs]

    # lesion the HTM/filler afferent (WM only) — must collapse >= to wm_alone
    p_les_htm = [hybrid_predict_verb(lr_hyb, s, L, wm_cells_from(sh), verb, use_htm=False, use_wm=True)
                 for s, sh in zip(test_seqs, subj_hat)]

    # lesion-the-hold (recur=0 slot) — the spiking afferent is garbage -> must collapse (proves it reads spikes)
    p_les_hold = [hybrid_predict_verb(lr_hyb, s, L, wm_cells_from(sh), verb, use_htm=True, use_wm=True)
                  for s, sh in zip(test_seqs, subj_hat_les)]

    # subject-shuffle: deref the slot to a WRONG subject -> the WM afferent carries the wrong subject value
    dr = np.random.default_rng(seed + 13); perm = list(range(n_subj))
    for _ in range(64):
        dr.shuffle(perm)
        if all(perm[i] != i for i in range(n_subj)):
            break
    def wm_cells_shuf(subj_tok):
        if subj_tok in subj:
            return lr_hyb._col(wm[perm[subj.index(int(subj_tok))]])[:k_win]
        return None
    p_shuf = [hybrid_predict_verb(lr_hyb, s, L, wm_cells_shuf(sh), verb, use_htm=True, use_wm=True)
              for s, sh in zip(test_seqs, subj_hat)]

    # wm_alone: the slot's held subject -> the subject's most-likely verb (no access to the local class)
    cls_counts = Counter(cls_of(s[-2], n_subj, n_fill, n_cls) for s in train_seqs)
    guess_cls = cls_counts.most_common(1)[0][0] if cls_counts else 0
    p_wm = []
    for sh in subj_hat:
        if sh in subj:
            p_wm.append(verb_of(verb, subj.index(int(sh)), guess_cls, n_cls))
        else:
            p_wm.append(-1)

    ex_htm, sub_htm, cl_htm = score_preds(p_htm, test_seqs, verb, n_cls)
    ex_hyb, sub_hyb, cl_hyb = score_preds(p_hyb, test_seqs, verb, n_cls)
    ex_wm, sub_wm, cl_wm = score_preds(p_wm, test_seqs, verb, n_cls)
    ex_lwm, _, _ = score_preds(p_les_wm, test_seqs, verb, n_cls)
    ex_lhtm, sub_lhtm, cl_lhtm = score_preds(p_les_htm, test_seqs, verb, n_cls)
    ex_lhold, _, _ = score_preds(p_les_hold, test_seqs, verb, n_cls)
    ex_shuf, sub_shuf, cl_shuf = score_preds(p_shuf, test_seqs, verb, n_cls)

    ngram, ngram_k = ngram_floor_comp(train_seqs, test_seqs, L, n_subj)

    if debug:
        # inspect the coincidence-AND margin on a few held-out sentences
        print(f"    [debug seed={seed} L={L}] slot_decode_acc={slot_decode_acc:.3f} hold_alive={np.mean(hold_alive):.4f}",
              flush=True)
        for s, sh in list(zip(test_seqs, subj_hat))[:3]:
            ts, tc = decode_verb(verb, s[-1], n_cls)
            _, cd = hybrid_predict_verb(lr_hyb, s, L, wm_cells_from(sh), verb, use_htm=True, use_wm=True, return_drive=True)
            _, cdh = hybrid_predict_verb(lr_htm, s, L, None, verb, use_htm=True, use_wm=False, return_drive=True)
            top = sorted(cd.items(), key=lambda kv: -kv[1])[:4]
            print(f"      true verb {s[-1]} (subj{ts},cls{tc}) | HYBRID drive top: "
                  + ", ".join(f"{decode_verb(verb,vt,n_cls)}={dv:.1f}" for vt, dv in top), flush=True)

    return {"seed": seed, "L": L, "distance": L + 1, "n_fill": n_fill, "n_cls": n_cls, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(n_fill) ** L,
            "slot_decode_acc": slot_decode_acc, "hold_alive": float(np.mean(hold_alive)), "zero_input_ok": bool(zero_ok),
            "htm_exact": ex_htm, "htm_subj": sub_htm, "htm_cls": cl_htm,
            "wm_exact": ex_wm, "wm_subj": sub_wm, "wm_cls": cl_wm,
            "hybrid_exact": ex_hyb, "hybrid_subj": sub_hyb, "hybrid_cls": cl_hyb,
            "lesion_wm_aff_exact": ex_lwm, "lesion_htm_aff_exact": ex_lhtm,
            "lesion_htm_aff_subj": sub_lhtm, "lesion_htm_aff_cls": cl_lhtm,
            "lesion_hold_exact": ex_lhold, "subj_shuffle_exact": ex_shuf, "subj_shuffle_subj": sub_shuf,
            "subj_shuffle_cls": cl_shuf, "ngram_floor_exact": ngram, "ngram_order": ngram_k, "oracle": 1.0}


def agg(per):
    keys = ["slot_decode_acc", "hold_alive", "htm_exact", "htm_subj", "htm_cls", "wm_exact", "wm_subj", "wm_cls",
            "hybrid_exact", "hybrid_subj", "hybrid_cls", "lesion_wm_aff_exact", "lesion_htm_aff_exact",
            "lesion_htm_aff_subj", "lesion_htm_aff_cls", "lesion_hold_exact", "subj_shuffle_exact", "subj_shuffle_subj",
            "subj_shuffle_cls", "ngram_floor_exact", "oracle"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    a.update({"L": per[0]["L"], "distance": per[0]["distance"], "n_fill": per[0]["n_fill"], "n_cls": per[0]["n_cls"],
              "chance": per[0]["chance"], "path_space": per[0]["path_space"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "ngram_order": int(np.round(np.mean([p["ngram_order"] for p in per]))), "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-subj", type=int, default=4)
    ap.add_argument("--n-fill", type=int, default=8)
    ap.add_argument("--n-cls", type=int, default=2, help="# local classes (a fn of the LAST filler); WM-alone caps ~1/n_cls")
    ap.add_argument("--distances", type=int, nargs="+", default=[3], help="filler-span L (dependency distance = L+1)")
    ap.add_argument("--n-cells", type=int, default=32, help="cells/column")
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3, help="coincidence threshold for the STANDALONE HTM (filler-alone fires)")
    ap.add_argument("--and-th", type=int, default=6, help="coincidence-AND threshold for the HYBRID (needs BOTH afferents)")
    ap.add_argument("--learn-th", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-test", type=int, default=64, help="held-out NOVEL-filler test sentences")
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--go-distance", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    n_verb = a.n_subj * a.n_cls
    chance = 1.0 / n_verb
    dists = sorted(set(a.distances))
    subj, fill, verb, wm, V = hybrid_layout(a.n_subj, a.n_fill, a.n_cls)
    print(f"backend={backend} device={device} | n_subj={a.n_subj} n_fill={a.n_fill} n_cls={a.n_cls} n_verb={n_verb} "
          f"chance={chance:.3f} | L={dists} | vocab={V} n_cells={a.n_cells} act_th={a.act_th} AND_th={a.and_th} "
          f"epochs={a.epochs} n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            per = [run_point(s, a.n_subj, a.n_fill, a.n_cls, L, a.n_cells, a.k_win, a.act_th, a.and_th, a.learn_th,
                             a.epochs, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps, a.clear_steps,
                             debug=a.debug) for s in a.seeds]
            p = agg(per); points.append(p)
            print(f"  [L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']}] "
                  f"HYBRID exact {p['hybrid_exact']:.3f} (subj {p['hybrid_subj']:.3f} cls {p['hybrid_cls']:.3f}) || "
                  f"HTM-alone {p['htm_exact']:.3f} (subj {p['htm_subj']:.3f} cls {p['htm_cls']:.3f}) | "
                  f"WM-alone {p['wm_exact']:.3f} (subj {p['wm_subj']:.3f} cls {p['wm_cls']:.3f}) || "
                  f"LES-wm-aff {p['lesion_wm_aff_exact']:.3f} | LES-htm-aff {p['lesion_htm_aff_exact']:.3f} | "
                  f"LES-hold {p['lesion_hold_exact']:.3f} | subj-shuf {p['subj_shuffle_exact']:.3f} || "
                  f"n-gram {p['ngram_floor_exact']:.3f}@k{p['ngram_order']} chance {chance:.3f} | "
                  f"slot-decode {p['slot_decode_acc']:.3f} alive {p['hold_alive']:.4f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = None
    if err is None and far is not None:
        base = max(far["htm_exact"], far["wm_exact"])
        print(f"\n-- GO gate + lesion teeth at L={far['L']} (dist {far['distance']}, held-out NOVEL fillers) --", flush=True)
        void_if(not far["gen_defined"], "path space too small to hold out novel fillers -> generalisation UNDEFINED")
        lever("HYBRID vs LESION-the-WM-afferent (WM afferent load-bearing)", round(far["lesion_wm_aff_exact"], 3),
              round(far["hybrid_exact"], 3), required=False)
        lever("HYBRID vs LESION-the-HTM-afferent (HTM afferent load-bearing)", round(far["lesion_htm_aff_exact"], 3),
              round(far["hybrid_exact"], 3), required=False)
        lever("HYBRID vs LESION-the-HOLD (spiking sustain load-bearing; the afferent reads spikes)",
              round(far["lesion_hold_exact"], 3), round(far["hybrid_exact"], 3), required=False)
        attributable_to("HYBRID held-out over the best single system", far["hybrid_exact"], base)
        attributable_to("HYBRID held-out over the n-gram floor", far["hybrid_exact"], far["ngram_floor_exact"])

        gen = far["gen_defined"]
        beats_both = far["hybrid_exact"] >= base + 0.20
        above_chance = far["hybrid_exact"] >= chance + 0.30
        wm_aff_lb = far["hybrid_exact"] >= far["lesion_wm_aff_exact"] + 0.20 and far["lesion_wm_aff_exact"] <= far["htm_exact"] + 0.10
        htm_aff_lb = far["hybrid_exact"] >= far["lesion_htm_aff_exact"] + 0.20 and far["lesion_htm_aff_exact"] <= far["wm_exact"] + 0.10
        hold_lb = far["hybrid_exact"] >= far["lesion_hold_exact"] + 0.20
        no_leak = far["subj_shuffle_exact"] <= base + 0.05
        spikes_ok = far["zero_input_ok"] and far["hold_alive"] > 1e-3
        core = bool(gen and beats_both and above_chance and wm_aff_lb and htm_aff_lb and hold_lb and no_leak and spikes_ok)
        go = bool(core and not smoke)

        if not gen:
            verdict = (f"INCONCLUSIVE — L={far['L']} path space {far['path_space']:.0f} too small to hold out novel "
                       f"fillers; generalisation UNDEFINED. Increase n_fill/L.")
        elif core:
            tag = "GO" if go else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — WIRING the spiking WM slot INTO the emergence engine gives the COMBINED system a faculty "
                       f"NEITHER has alone. On a compositional stream verb=combine(subject, local-class(last filler)), "
                       f"held-out (NOVEL fillers) exact branch(verb): HYBRID {far['hybrid_exact']:.3f} >> HTM-alone "
                       f"{far['htm_exact']:.3f} (recovers the local class {far['htm_cls']:.3f} but loses the subject "
                       f"{far['htm_subj']:.3f}) and >> WM-alone {far['wm_exact']:.3f} (recovers the subject "
                       f"{far['wm_subj']:.3f} but not the class {far['wm_cls']:.3f}); >> n-gram floor "
                       f"{far['ngram_floor_exact']:.3f}, >> chance {chance:.3f}. The fusion is a NEURAL dendritic "
                       f"coincidence-AND (the WM's latched subject enters as an extra cortical afferent read from the "
                       f"slot's SPIKES; the verb cell fires only on the conjunction of the filler and WM afferents): "
                       f"lesion-the-WM-afferent -> {far['lesion_wm_aff_exact']:.3f} (= HTM-alone), lesion-the-HTM-afferent "
                       f"-> {far['lesion_htm_aff_exact']:.3f} (= WM-alone), BOTH load-bearing; lesion-the-hold "
                       f"(recur=0) -> {far['lesion_hold_exact']:.3f} (the afferent reads the slot's spiking sustain, "
                       f"external input asserted zero); subject-shuffle -> {far['subj_shuffle_exact']:.3f} (no leakage). "
                       f"The WM faculty is load-bearing on this structure. Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not beats_both: miss.append(f"hybrid {far['hybrid_exact']:.3f} not >= max(HTM {far['htm_exact']:.3f}, WM {far['wm_exact']:.3f})+0.20")
            if not above_chance: miss.append(f"hybrid {far['hybrid_exact']:.3f} not >= chance+0.30 ({chance:.3f})")
            if not wm_aff_lb: miss.append(f"WM afferent not cleanly load-bearing (lesion-wm-aff {far['lesion_wm_aff_exact']:.3f} vs HTM-alone {far['htm_exact']:.3f})")
            if not htm_aff_lb: miss.append(f"HTM afferent not cleanly load-bearing (lesion-htm-aff {far['lesion_htm_aff_exact']:.3f} vs WM-alone {far['wm_exact']:.3f})")
            if not hold_lb: miss.append(f"hold not load-bearing (lesion-hold {far['lesion_hold_exact']:.3f})")
            if not no_leak: miss.append(f"leakage (subject-shuffle {far['subj_shuffle_exact']:.3f} > base+0.05)")
            if not spikes_ok: miss.append(f"hold not alive/zero-input (alive {far['hold_alive']:.4f}, zero {far['zero_input_ok']})")
            verdict = ("HONEST NEGATIVE / PARTIAL — the neural WM->HTM fusion did not clear the GO bar at L={}: ".format(far["L"])
                       + "; ".join(miss) + ". Read the per-arm numbers (exact/subject/class) to see which faculty the "
                       "fusion failed to combine, and whether the coincidence-AND is lossy or an afferent is ignored.")
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = "ERROR — no points computed"

    # --- earned verdict preconditions (VALIDITY travels with the verdict) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("emerge_wm_hybrid", chance=chance)
        if far is not None:
            Vd.require("oracle_task_solvable", 1.0, expect=lambda x: x > 0.99,
                       note="verb is determined by (subject, local-class) by construction -> the task IS solvable")
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out filler tuples disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("htm_alone_loses_subject", round(far["htm_subj"], 4), expect=lambda x: x <= 0.5,
                       note="HTM-alone must lose the long-range subject on novel fillers (its banked failure)")
            Vd.require("wm_alone_loses_class", round(far["wm_cls"], 4), expect=lambda x: x <= 1.0 / a.n_cls + 0.15,
                       note="WM-alone cannot see the local class (it only carries the subject)")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across the hold+read span")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(far is not None and far["gen_defined"]
                   and far["hybrid_exact"] >= max(far["htm_exact"], far["wm_exact"]) + 0.20
                   and far["hybrid_exact"] >= chance + 0.30
                   and far["hybrid_exact"] >= far["lesion_wm_aff_exact"] + 0.20
                   and far["hybrid_exact"] >= far["lesion_htm_aff_exact"] + 0.20
                   and far["hybrid_exact"] >= far["lesion_hold_exact"] + 0.20)
        dec = Vd.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_wm_hybrid", "verdict": verdict, "backend": backend, "sim_backend": backend,
               "device": device, "cost_acknowledged": True, "smoke": smoke, "preconditions": preconditions,
               "mechanism": "the WM slot's LATCHED subject (read from the slow-NMDA attractor's spikes: argmax pool "
                            "firing over cp_firing_states) enters the on-bridge HTM Temporal-Memory emergence engine as "
                            "an EXTRA cortical afferent (a dedicated WM-memory column per subject, neurons in the SAME "
                            "SimulationBridge); the engine's own coincidence-recurrence prediction is the read-out and "
                            "the verb cell fires only on the dendritic coincidence-AND of the last-filler afferent AND "
                            "the WM-memory afferent (the apical plateau is binary above threshold, so a threshold set "
                            "between single- and double-pathway drive realises the AND). NOT a host ensemble: no host "
                            "code compares a WM answer to an HTM answer.",
               "task": "compositional agreement stream [subject]+[L i.i.d. fillers]+[verb], verb=combine(subject, "
                       "local-class(last filler)); held-out TEST = disjoint NOVEL filler tuples; the local class is a "
                       "fixed fn of the last token (HTM-learnable, generalises) and the subject is L+1-back (only the WM "
                       "carries it); arms: htm_alone/wm_alone/hybrid + lesion-wm-afferent + lesion-htm-afferent + "
                       "lesion-the-hold + subject-shuffle + n-gram floor + chance + oracle",
               "seeds": a.seeds, "config": {"n_subj": a.n_subj, "n_fill": a.n_fill, "n_cls": a.n_cls, "distances": dists,
               "n_cells": a.n_cells, "k_win": a.k_win, "act_th": a.act_th, "and_th": a.and_th, "learn_th": a.learn_th,
               "epochs": a.epochs, "n_train": a.n_train, "n_test": a.n_test, "recur": a.recur,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps, "chance": chance,
               "go_distance": go_L}, "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the EMERGE-14 on-bridge HTM engine + the banked spiking WM slot "
                              "(slow-NMDA hold) + persistent subject binder; NO sim/ edit. The WM afferent during "
                              "TRAINING is driven by the TRUE subject (the WM working during development; banked slot "
                              "latch acc ~1.0); the decisive spiking-slot READ carries the subject across NOVEL fillers "
                              "at TEST. The subject->slot bind + the WM-memory labelled-line projection are fixed host "
                              "wiring (topographic); the LATCH + READ are spiking. 1-seed is a SMOKE indicator; the "
                              "6-seed sweep is decisive."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[emerge_wm_hybrid] VERDICT: {verdict}", flush=True)
    print(f"[emerge_wm_hybrid] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
