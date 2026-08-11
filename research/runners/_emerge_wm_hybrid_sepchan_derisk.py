"""EMERGENCE-ENGINE + WORKING-MEMORY HYBRID, RUNG 3b — SEPARATE-CHANNEL fusion (close the PARTIAL).

BANKED PARTIAL (do NOT re-derive; read 2026-08-11-emergence-engine-plus-WM-afferent-hybrid-PARTIAL-...):
  The naive coincidence-AND fusion (`_emerge_wm_hybrid_derisk`) GENUINELY COMBINES the WM + HTM faculties (hybrid
  0.641 [min 0.562] BEATS HTM-alone 0.224 and WM-alone 0.516; both afferent lesions load-bearing; lesion-the-hold
  collapses -> reads spikes) but MISSES the strict GO bar (0.716) because the fusion is LOSSY on the SUBJECT: the WM
  alone latches subj 1.000, but under fusion the HTM's local-CLASS afferent CORRUPTS it (hybrid subj -> 0.667) while
  the class combines cleanly (hybrid cls 0.974). The two afferents interfere where they should be INDEPENDENT because
  BOTH write the SAME verb columns and the last-filler afferent carries a residual SUBJECT bias that overpowers the
  WM's subject signal on ~1/3 of cases.

THE FIX (rung 3b) — route the WM's held-SUBJECT and the HTM's local-CLASS on DISTINCT NEURAL CHANNELS so the HTM's
class prediction CANNOT overwrite the WM's subject latch, then combine the two clean channels by a LEARNED dendritic
CONJUNCTION (not a host argmax/ensemble over two predictions):

  * SUBJECT channel (WM owns it) : the spiking WM slot's LATCHED subject (read from its slow-NMDA attractor spikes,
      argmax over cp_firing_states) selects its dedicated WM-MEMORY column wm[s]. The HTM engine does NOT write wm[].
  * CLASS channel (HTM owns it)  : a SEPARATE, SUBJECT-AGNOSTIC class-readout layer `clsrd` (n_cls columns, NO subject
      dimension). The on-bridge HTM Temporal-Memory engine runs its normal priming chain over [subject]+[fillers] and
      at the last filler its branch prediction potentiates last-filler-winners -> clsrd[class]. Because clsrd has NO
      subject dimension it CANNOT carry a subject bias into the verb layer -- this is the whole fix. The WM does NOT
      write clsrd[].
  * VERB read-out = a LEARNED, NEURAL, dendritic CONJUNCTION. A conjunction engine (its own bridge) is trained so
      each verb(s,c) column potentiates incoming coincidence synapses from BOTH wm[s] AND clsrd[c]. At decision time
      the substrate is PRIMED from the union {wm[subject_hat] cells} U {clsrd[class_hat] cells} and the verb column
      with the max apical drive wins: verb(s*,c*) is the UNIQUE double-driven cell -> its apical plateau charges
      (v_apical ~ +17 mV, measured) while every single-channel verb stays sub-threshold (~ -62 mV) -> it dominates by
      a huge margin. THIS IS NOT A HOST ENSEMBLE: the runner NEVER computes verb = table_lookup(subject_hat, class_hat);
      it primes the two spiking-decoded populations into the substrate and the verb IDENTITY EMERGES from the learned
      coincidence synapses (an UNTRAINED conjunction bridge collapses to chance -- the `conj_untrained` control proves
      the binding lives in learned synapses, not in wiring). The subject reaching the verb layer is ONLY wm[subject_hat]
      (clsrd is subject-agnostic) -> the subject is PRESERVED (hybrid subj -> ~1.0, up from the old fusion's 0.667).

  Evidence it READS SPIKES: (a) subject_hat = the slot's cp_firing_states argmax; (b) class_hat = the HTM's apical
  (cp_v_apical) branch prediction; (c) lesion-the-hold (recur=0 slot) -> the spiking sustain dies -> wrong wm[] column
  -> the conjunction fires verb(WRONG subject, c*) -> collapses; (d) the slot's external input is ASSERTED zero across
  the hold+read span.

ARMS (all REQUIRED; the PARTIAL's set kept so the comparison is direct + honest-negative is first-class):
  * htm_alone        : the standalone emergence engine predicting the verb directly (no WM). BANKED baseline.
  * wm_alone         : the slot's held subject -> the subject's most-likely verb (no local class). BANKED baseline.
  * old_fusion       : the RUNG-3 coincidence-AND fusion, RE-RUN here (reproduces 0.641 / subj 0.667) so the
                       improvement is attributable to the CHANNEL SEPARATION, not to any environment difference.
  * hybrid_sep       : the NEW separate-channel conjunctive fusion. THE TEST. Report exact / subject / class.
  * lesion_wm_chan   : hybrid_sep with the SUBJECT channel ablated (prime clsrd[c*] only) -> class known, subject
                       unresolved -> MUST drop to ~HTM-alone.
  * lesion_htm_chan  : hybrid_sep with the CLASS channel ablated (prime wm[subject_hat] only) -> subject known, class
                       unresolved -> MUST drop to ~WM-alone.
  * lesion_hold      : hybrid_sep with the slot recur=0 (the hold dies) -> subject_hat garbage -> MUST collapse
                       (proves the subject channel READS the slot's spikes).
  * subj_shuffle     : the slot->subject deref permuted -> wm[] points to the WRONG subject -> subject collapses to
                       chance (no positional/topic leakage; the carried VALUE is load-bearing).
  * conj_untrained   : the SEPARATE-CHANNEL readout on an UNTRAINED conjunction bridge -> ~chance (the neural bind is
                       LEARNED, not host wiring -> rebuts "it's a host lookup").
  * held-out NOVEL fillers (disjoint tuples), n-gram HELD-OUT floor, chance, task-solvable oracle (=1 by build).

GO GATE: hybrid_sep held-out exact >= max(htm_alone, wm_alone) + 0.20 AND >= chance + 0.30, generalising to NOVEL
  fillers, with the SUBJECT PRESERVED (hybrid subj >= 0.90, up from the old fusion's 0.667) AND the CLASS clean
  (>= 0.90) AND both channel lesions load-bearing (each drops the hybrid >= 0.20 to <= the corresponding single-system
  baseline + 0.10) AND lesion-the-hold collapsing. HONEST NEGATIVE (first-class) otherwise: map whether separate
  channels helped -- did subj recover? did the conjunction cost class? -- precisely.

Reuse-by-import: the compositional stream + layout + spiking WM slot carry + the OLD-fusion arms from
`_emerge_wm_hybrid_derisk`; the EMERGE-14 on-bridge HTM engine primitives (build_pool_bridge / apply_kernel_update /
coincidence_predict) + the banked spiking WM slot. Change ONLY the fusion (coincidence-AND -> separate-channel
conjunctive) + the read-out. NO sim/ edit. SIM_BACKEND=numpy (sub-1k-neuron loops are launch-bound -> CPU is correct
+ faster). Verified: build_pool_bridge sets cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = seed, so the substrate
IS seeded (the `actual_seed_used` trap does not apply).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_sepchan_derisk --seeds 42 --debug
6-seed decisive (fan one seed per process; see the 6-seed command in the finding):
  SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_sepchan_derisk --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_emerge_wm_hybrid_sepchan/sepchan_6seed.json
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
from collections import Counter
from pathlib import Path

import numpy as np

# --- the EMERGE-14 on-bridge HTM Temporal-Memory engine primitives ---
from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    apply_kernel_update, coincidence_predict)
# --- REUSE the compositional stream, layout, spiking WM slot carry, and the OLD-fusion arms (rung 3) ---
from research.runners._emerge_wm_hybrid_derisk import (
    hybrid_layout, cls_of, verb_of, decode_verb, make_comp_stream, make_comp_heldout, ngram_floor_comp,
    score_preds, apical_drive, build_engine, train_engine_hybrid, hybrid_predict_verb, slot_carry_subject)
# --- the banked spiking WM: slow-NMDA bistable HOLD slot + persistent subject binder + codebook ---
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

OUT = Path("research/findings/raw/_emerge_wm_hybrid_sepchan/sepchan.json")


# ====================================================================================================================
# Layout: the base compositional layout (subj | fill | verb | wm) + a SEPARATE, SUBJECT-AGNOSTIC class-readout layer.
#   clsrd = n_cls columns AFTER the base layout. Stream tokens (subj/fill/verb) keep their base indices; clsrd columns
#   are INTERNAL readout units that never appear in a sentence. The verb layer (n_subj*n_cls) is the CONJUNCTION.
# ====================================================================================================================
def sepchan_layout(n_subj, n_fill, n_cls):
    subj, fill, verb, wm, V_base = hybrid_layout(n_subj, n_fill, n_cls)
    clsrd = list(range(V_base, V_base + n_cls))
    V = V_base + n_cls
    return subj, fill, verb, wm, clsrd, V_base, V


# ====================================================================================================================
# CLASS CHANNEL (HTM owns it) — the on-bridge engine's OWN branch prediction, read out on a SUBJECT-AGNOSTIC channel.
# Train the temporal chain over [subject]+[fillers] (normal HTM allocation) AND potentiate the last-filler winners ->
# clsrd[class(last filler)]. clsrd has NO subject dimension -> it cannot carry a subject bias (the fix).
# ====================================================================================================================
def train_class_channel(lr, seq, L, clsrd, n_subj, n_fill, n_cls):
    predictive, prev_winners = set(), set()
    for pos in range(L + 1):                                       # subject + L fillers (no verb token needed)
        c = seq[pos]
        col = lr._col(c)
        primed = [i for i in col if i in predictive] if not lr.lesion else []
        if primed:
            winners = set(primed[:lr.k_win])
        elif not prev_winners:
            winners = set(col[:lr.k_win])
        else:
            scored = sorted(((lr._match_count(i, prev_winners), i) for i in col), reverse=True)
            if scored[0][0] >= lr.learn_th:
                winners = set(i for sc, i in scored[:lr.k_win] if sc >= lr.learn_th)
            else:
                wc = lr._committed_count()
                winners = set(sorted(col, key=lambda i: (wc[i], i))[:lr.k_win])
        if prev_winners:                                           # normal temporal potentiation prev -> cur
            apply_kernel_update(lr.b, lr.row, lr.col, lr.cells_idx, prev_winners, winners,
                                lr.z, lr.lam_pot, lr.lam_dep, lr.z_star)
        if pos == L:                                               # branch prediction: last-filler winners -> clsrd[class]
            cls = cls_of(seq[pos], n_subj, n_fill, n_cls)
            target = set(lr._col(clsrd[cls])[:lr.k_win])
            apply_kernel_update(lr.b, lr.row, lr.col, lr.cells_idx, winners, target,
                                lr.z, lr.lam_pot, lr.lam_dep, lr.z_star)
        active = winners if primed else set(col)
        predictive = coincidence_predict(lr.b, lr.cells_idx, active, lr.N, lr.nE)
        lr.z *= lr.z_tau
        for i in predictive:
            lr.z[i] += (1.0 - lr.z_tau)
        prev_winners = winners


def class_read(lr, seq, L, clsrd, thr_off=2.0):
    """The HTM's local-class branch prediction on the subject-agnostic channel: forward the priming chain to the last
    filler, then the argmax clsrd column over the substrate's apical drive. Returns the class INDEX (or -1)."""
    tok = hybrid_predict_verb(lr, seq, L, None, clsrd, use_htm=True, use_wm=False, thr_off=thr_off)
    return clsrd.index(tok) if tok in clsrd else -1


# ====================================================================================================================
# THE LEARNED, NEURAL CONJUNCTION (verb layer). Train each verb(s,c) to potentiate incoming coincidence synapses from
# BOTH wm[s] AND clsrd[c]; at decision time PRIME from {wm[s*] U clsrd[c*]} and read the verb with max apical drive.
# verb(s*,c*) is the UNIQUE double-driven cell -> plateau (~+17mV) >> single-channel (~-62mV) -> it wins by a huge
# margin (subject preserved: the ONLY subject signal in the verb layer is wm[s*], clsrd being subject-agnostic).
# ====================================================================================================================
def train_conjunction(lr, seqs, L, wm, clsrd, verb, subj_list, n_subj, n_fill, n_cls, epochs):
    for _ in range(epochs):
        for s in seqs:
            si = subj_list.index(s[0])
            c = cls_of(s[L], n_subj, n_fill, n_cls)
            pre = set(lr._col(wm[si])[:lr.k_win]) | set(lr._col(clsrd[c])[:lr.k_win])
            tgt = set(lr._col(verb_of(verb, si, c, n_cls))[:lr.k_win])
            apply_kernel_update(lr.b, lr.row, lr.col, lr.cells_idx, pre, tgt,
                                lr.z, lr.lam_pot, lr.lam_dep, lr.z_star)


def conj_read(lr, pre_cells, verb_cols, thr_off):
    """Prime the conjunction bridge from the union of the decoded channel populations and read the verb column with the
    max apical drive above the floor (a GRADED dendritic coincidence read). The coincidence verb dominates single-channel
    verbs by a huge margin; if only ONE channel is present (a lesion arm) the single-channel verbs of the surviving
    dimension are the max -> the ablated dimension collapses to ~1/n (the single-system baseline). Returns a verb token."""
    E_rest = float(getattr(lr.b.core_config, "apical_E_rest", -65.0))
    thr = E_rest + thr_off
    pre = set(pre_cells)
    if not pre:
        return -1
    drive = apical_drive(lr.b, lr.cells_idx, pre)
    if drive is None:
        return -1
    best_tok, best_d = -1, thr
    for vt in verb_cols:
        d = float(np.max(drive[lr._col(vt)]))
        if d > best_d:
            best_d, best_tok = d, vt
    return best_tok


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, n_subj, n_fill, n_cls, L, n_cells, k_win, act_th, and_th, learn_th, epochs, n_train, n_test,
              recur, hold_steps, load_steps, clear_steps, thr_off_conj, debug=False):
    rng = np.random.default_rng(seed)
    subj, fill, verb, wm, clsrd, V_base, V = sepchan_layout(n_subj, n_fill, n_cls)
    n_verb = n_subj * n_cls
    chance = 1.0 / n_verb
    wm_col_of_subj_idx = {i: wm[i] for i in range(n_subj)}

    # --- the compositional stream + disjoint NOVEL-filler held-out test (identical to the PARTIAL) ---
    train_seqs = make_comp_stream(n_subj, n_fill, n_cls, L, n_train, rng)
    train_ftuples = set(tuple(s[1:-1]) for s in train_seqs)
    test_seqs, gen_defined = make_comp_heldout(n_subj, n_fill, n_cls, L, n_test, rng, train_ftuples)

    # --- the spiking WM: codebook + persistent subject binder (reuse-by-import) ---
    code_of, _ = build_codebook(n_subj, n_fill, np.random.default_rng(seed + 7))
    binder, slot_of_subj, subj_of_slot = persistent_subject_binder([int(s) for s in subj], code_of)

    def _slot(rc=recur, off=0):
        return SpikingSlot(seed + off, _K, recur=rc, hold_steps=hold_steps, load_steps=load_steps,
                           clear_steps=clear_steps)

    # --- the spiking slot's held-subject read for every test sentence (intact + hold-lesion) ---
    slot = _slot()
    slot_les = _slot(rc=0.0, off=100)
    subj_hat, hold_alive, zero_ok, subj_hat_les = [], [], True, []
    for s in test_seqs:
        sh, al, zo = slot_carry_subject(slot, s, slot_of_subj, subj_of_slot)
        subj_hat.append(sh); hold_alive.append(al); zero_ok = zero_ok and zo
        shl, _, _ = slot_carry_subject(slot_les, s, slot_of_subj, subj_of_slot)
        subj_hat_les.append(shl)
    slot_decode_acc = float(np.mean([int(sh == s[0]) for sh, s in zip(subj_hat, test_seqs)]))

    # ================= REFERENCE ARMS (build on V_base -> byte-identical to the banked PARTIAL) =================
    lr_htm = build_engine(seed, V_base, n_cells, act_th, k_win, learn_th)
    train_engine_hybrid(lr_htm, train_seqs, L, wm_col_of_subj_idx, subj, use_wm=False, epochs=epochs)
    lr_old = build_engine(seed, V_base, n_cells, and_th, k_win, learn_th)
    train_engine_hybrid(lr_old, train_seqs, L, wm_col_of_subj_idx, subj, use_wm=True, epochs=epochs)

    def wm_cells_old(sh):
        return lr_old._col(wm[subj.index(int(sh))])[:k_win] if sh in subj else None

    p_htm = [hybrid_predict_verb(lr_htm, s, L, None, verb, use_htm=True, use_wm=False) for s in test_seqs]
    p_old = [hybrid_predict_verb(lr_old, s, L, wm_cells_old(sh), verb, use_htm=True, use_wm=True)
             for s, sh in zip(test_seqs, subj_hat)]

    # wm_alone: the slot's held subject -> the subject's most-likely verb (banked logic; no access to the local class)
    cls_counts = Counter(cls_of(s[-2], n_subj, n_fill, n_cls) for s in train_seqs)
    guess_cls = cls_counts.most_common(1)[0][0] if cls_counts else 0
    p_wm = [verb_of(verb, subj.index(int(sh)), guess_cls, n_cls) if sh in subj else -1 for sh in subj_hat]

    # ================= NEW SEPARATE-CHANNEL ARMS (build on V_sep) =================
    lr_cls = build_engine(seed, V, n_cells, act_th, k_win, learn_th)              # class channel (normal threshold)
    for _ in range(epochs):
        for s in train_seqs:
            train_class_channel(lr_cls, s, L, clsrd, n_subj, n_fill, n_cls)
    lr_conj = build_engine(seed, V, n_cells, and_th, k_win, learn_th)             # conjunction (AND threshold)
    train_conjunction(lr_conj, train_seqs, L, wm, clsrd, verb, subj, n_subj, n_fill, n_cls, epochs)
    lr_unt = build_engine(seed, V, n_cells, and_th, k_win, learn_th)              # UNTRAINED conjunction (control)

    # the HTM's local-class branch prediction (subject-agnostic channel) for every held-out sentence
    c_hat = [class_read(lr_cls, s, L, clsrd) for s in test_seqs]
    cls_chan_acc = float(np.mean([int(ch == cls_of(s[-2], n_subj, n_fill, n_cls)) for ch, s in zip(c_hat, test_seqs)]))

    def wmc(lr, sh):
        return set(lr._col(wm[subj.index(int(sh))])[:k_win]) if sh in subj else set()

    def clc(lr, ch):
        return set(lr._col(clsrd[ch])[:k_win]) if ch is not None and ch >= 0 else set()

    # hybrid_sep: prime {wm[subject_hat] U clsrd[class_hat]} -> the learned conjunction fires verb(s*,c*)
    p_sep = [conj_read(lr_conj, wmc(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
             for sh, ch in zip(subj_hat, c_hat)]
    # lesion the WM (subject) channel -> class only -> MUST drop to ~HTM-alone
    p_les_wm = [conj_read(lr_conj, clc(lr_conj, ch), verb, thr_off_conj) for ch in c_hat]
    # lesion the HTM (class) channel -> subject only -> MUST drop to ~WM-alone
    p_les_htm = [conj_read(lr_conj, wmc(lr_conj, sh), verb, thr_off_conj) for sh in subj_hat]
    # lesion-the-hold: subject_hat from the recur=0 slot is garbage -> wrong wm[] -> collapses (reads spikes)
    p_les_hold = [conj_read(lr_conj, wmc(lr_conj, shl) | clc(lr_conj, ch), verb, thr_off_conj)
                  for shl, ch in zip(subj_hat_les, c_hat)]
    # subject-shuffle: deref the slot to a WRONG subject column
    dr = np.random.default_rng(seed + 13); perm = list(range(n_subj))
    for _ in range(64):
        dr.shuffle(perm)
        if all(perm[i] != i for i in range(n_subj)):
            break

    def wmc_shuf(lr, sh):
        return set(lr._col(wm[perm[subj.index(int(sh))]])[:k_win]) if sh in subj else set()

    p_shuf = [conj_read(lr_conj, wmc_shuf(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
              for sh, ch in zip(subj_hat, c_hat)]
    # conj_untrained control: the SAME separate-channel readout on an UNTRAINED conjunction bridge -> ~chance
    p_unt = [conj_read(lr_unt, wmc(lr_unt, sh) | clc(lr_unt, ch), verb, thr_off_conj)
             for sh, ch in zip(subj_hat, c_hat)]

    # ---- score ----
    ex_htm, sub_htm, cl_htm = score_preds(p_htm, test_seqs, verb, n_cls)
    ex_old, sub_old, cl_old = score_preds(p_old, test_seqs, verb, n_cls)
    ex_wm, sub_wm, cl_wm = score_preds(p_wm, test_seqs, verb, n_cls)
    ex_sep, sub_sep, cl_sep = score_preds(p_sep, test_seqs, verb, n_cls)
    ex_lwm, sub_lwm, cl_lwm = score_preds(p_les_wm, test_seqs, verb, n_cls)
    ex_lhtm, sub_lhtm, cl_lhtm = score_preds(p_les_htm, test_seqs, verb, n_cls)
    ex_lhold, _, _ = score_preds(p_les_hold, test_seqs, verb, n_cls)
    ex_shuf, sub_shuf, cl_shuf = score_preds(p_shuf, test_seqs, verb, n_cls)
    ex_unt, _, _ = score_preds(p_unt, test_seqs, verb, n_cls)

    ngram, ngram_k = ngram_floor_comp(train_seqs, test_seqs, L, n_subj)

    if debug:
        print(f"    [debug seed={seed} L={L}] slot_decode_acc={slot_decode_acc:.3f} hold_alive={np.mean(hold_alive):.4f} "
              f"cls_chan_acc={cls_chan_acc:.3f}", flush=True)
        for s, sh, ch in list(zip(test_seqs, subj_hat, c_hat))[:3]:
            ts, tc = decode_verb(verb, s[-1], n_cls)
            drive = apical_drive(lr_conj.b, lr_conj.cells_idx, wmc(lr_conj, sh) | clc(lr_conj, ch))
            cd = {decode_verb(verb, vt, n_cls): round(float(np.max(drive[lr_conj._col(vt)])), 1) for vt in verb}
            top = sorted(cd.items(), key=lambda kv: -kv[1])[:4]
            print(f"      true (subj{ts},cls{tc}) | slot->subj{subj.index(sh) if sh in subj else -1} htm->cls{ch} | "
                  f"conj drive top: {top}", flush=True)

    return {"seed": seed, "L": L, "distance": L + 1, "n_fill": n_fill, "n_cls": n_cls, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(n_fill) ** L,
            "slot_decode_acc": slot_decode_acc, "cls_chan_acc": cls_chan_acc,
            "hold_alive": float(np.mean(hold_alive)), "zero_input_ok": bool(zero_ok),
            "htm_exact": ex_htm, "htm_subj": sub_htm, "htm_cls": cl_htm,
            "wm_exact": ex_wm, "wm_subj": sub_wm, "wm_cls": cl_wm,
            "old_fusion_exact": ex_old, "old_fusion_subj": sub_old, "old_fusion_cls": cl_old,
            "hybrid_sep_exact": ex_sep, "hybrid_sep_subj": sub_sep, "hybrid_sep_cls": cl_sep,
            "lesion_wm_chan_exact": ex_lwm, "lesion_wm_chan_subj": sub_lwm, "lesion_wm_chan_cls": cl_lwm,
            "lesion_htm_chan_exact": ex_lhtm, "lesion_htm_chan_subj": sub_lhtm, "lesion_htm_chan_cls": cl_lhtm,
            "lesion_hold_exact": ex_lhold, "subj_shuffle_exact": ex_shuf, "subj_shuffle_subj": sub_shuf,
            "conj_untrained_exact": ex_unt, "ngram_floor_exact": ngram, "ngram_order": ngram_k, "oracle": 1.0}


def agg(per):
    keys = ["slot_decode_acc", "cls_chan_acc", "hold_alive",
            "htm_exact", "htm_subj", "htm_cls", "wm_exact", "wm_subj", "wm_cls",
            "old_fusion_exact", "old_fusion_subj", "old_fusion_cls",
            "hybrid_sep_exact", "hybrid_sep_subj", "hybrid_sep_cls",
            "lesion_wm_chan_exact", "lesion_wm_chan_subj", "lesion_wm_chan_cls",
            "lesion_htm_chan_exact", "lesion_htm_chan_subj", "lesion_htm_chan_cls",
            "lesion_hold_exact", "subj_shuffle_exact", "subj_shuffle_subj", "conj_untrained_exact",
            "ngram_floor_exact", "oracle"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    a.update({k + "_min": float(np.min([p[k] for p in per]))
              for k in ["hybrid_sep_exact", "hybrid_sep_subj", "hybrid_sep_cls"]})
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
    ap.add_argument("--act-th", type=int, default=3, help="coincidence threshold for the standalone HTM / class channel")
    ap.add_argument("--and-th", type=int, default=6, help="coincidence-AND threshold for the conjunction (needs BOTH channels)")
    ap.add_argument("--learn-th", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-test", type=int, default=64, help="held-out NOVEL-filler test sentences")
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--thr-off-conj", type=float, default=0.5,
                    help="apical read floor (E_rest+off) for the conjunction; low enough to admit a single-channel "
                         "sub-threshold bump (~-63mV) in a lesion arm but exclude rest (-65mV)")
    ap.add_argument("--go-distance", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    n_verb = a.n_subj * a.n_cls
    chance = 1.0 / n_verb
    dists = sorted(set(a.distances))
    _, _, _, _, _, V_base, V = sepchan_layout(a.n_subj, a.n_fill, a.n_cls)
    print(f"backend={backend} device={device} | n_subj={a.n_subj} n_fill={a.n_fill} n_cls={a.n_cls} n_verb={n_verb} "
          f"chance={chance:.3f} | L={dists} | vocab_base={V_base} vocab_sep={V} n_cells={a.n_cells} act_th={a.act_th} "
          f"AND_th={a.and_th} thr_off_conj={a.thr_off_conj} epochs={a.epochs} n_train={a.n_train} n_test={a.n_test} | "
          f"seeds={a.seeds}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            per = [run_point(s, a.n_subj, a.n_fill, a.n_cls, L, a.n_cells, a.k_win, a.act_th, a.and_th, a.learn_th,
                             a.epochs, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps, a.clear_steps,
                             a.thr_off_conj, debug=a.debug) for s in a.seeds]
            p = agg(per); points.append(p)
            print(f"  [L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']}] "
                  f"HYBRID-SEP exact {p['hybrid_sep_exact']:.3f}[min {p['hybrid_sep_exact_min']:.3f}] "
                  f"(subj {p['hybrid_sep_subj']:.3f} cls {p['hybrid_sep_cls']:.3f}) || "
                  f"old-fusion {p['old_fusion_exact']:.3f} (subj {p['old_fusion_subj']:.3f} cls {p['old_fusion_cls']:.3f}) | "
                  f"HTM {p['htm_exact']:.3f} (subj {p['htm_subj']:.3f} cls {p['htm_cls']:.3f}) | "
                  f"WM {p['wm_exact']:.3f} (subj {p['wm_subj']:.3f} cls {p['wm_cls']:.3f}) || "
                  f"LES-wm-chan {p['lesion_wm_chan_exact']:.3f}(subj {p['lesion_wm_chan_subj']:.3f} cls {p['lesion_wm_chan_cls']:.3f}) | "
                  f"LES-htm-chan {p['lesion_htm_chan_exact']:.3f}(subj {p['lesion_htm_chan_subj']:.3f} cls {p['lesion_htm_chan_cls']:.3f}) | "
                  f"LES-hold {p['lesion_hold_exact']:.3f} | subj-shuf {p['subj_shuffle_exact']:.3f} | "
                  f"conj-untrained {p['conj_untrained_exact']:.3f} || n-gram {p['ngram_floor_exact']:.3f}@k{p['ngram_order']} "
                  f"chance {chance:.3f} | slot {p['slot_decode_acc']:.3f} cls-chan {p['cls_chan_acc']:.3f} "
                  f"alive {p['hold_alive']:.4f}", flush=True)
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
        lever("HYBRID-SEP vs OLD coincidence-AND fusion (channel separation)", round(far["old_fusion_exact"], 3),
              round(far["hybrid_sep_exact"], 3), required=False)
        lever("HYBRID-SEP subject vs OLD-fusion subject (the fix: preserve the latch)", round(far["old_fusion_subj"], 3),
              round(far["hybrid_sep_subj"], 3), required=False)
        lever("HYBRID-SEP vs LESION-the-WM-channel (subject channel load-bearing)", round(far["lesion_wm_chan_exact"], 3),
              round(far["hybrid_sep_exact"], 3), required=False)
        lever("HYBRID-SEP vs LESION-the-HTM-channel (class channel load-bearing)", round(far["lesion_htm_chan_exact"], 3),
              round(far["hybrid_sep_exact"], 3), required=False)
        lever("HYBRID-SEP vs LESION-the-HOLD (spiking sustain load-bearing; reads spikes)",
              round(far["lesion_hold_exact"], 3), round(far["hybrid_sep_exact"], 3), required=False)
        lever("HYBRID-SEP vs UNTRAINED conjunction (the bind is learned, not host wiring)",
              round(far["conj_untrained_exact"], 3), round(far["hybrid_sep_exact"], 3), required=False)
        attributable_to("HYBRID-SEP held-out over the best single system", far["hybrid_sep_exact"], base)
        attributable_to("HYBRID-SEP held-out over the n-gram floor", far["hybrid_sep_exact"], far["ngram_floor_exact"])

        gen = far["gen_defined"]
        beats_both = far["hybrid_sep_exact"] >= base + 0.20
        above_chance = far["hybrid_sep_exact"] >= chance + 0.30
        subj_preserved = far["hybrid_sep_subj"] >= 0.90
        cls_clean = far["hybrid_sep_cls"] >= 0.90
        wm_chan_lb = far["hybrid_sep_exact"] >= far["lesion_wm_chan_exact"] + 0.20 and far["lesion_wm_chan_exact"] <= far["htm_exact"] + 0.10
        htm_chan_lb = far["hybrid_sep_exact"] >= far["lesion_htm_chan_exact"] + 0.20 and far["lesion_htm_chan_exact"] <= far["wm_exact"] + 0.10
        hold_lb = far["hybrid_sep_exact"] >= far["lesion_hold_exact"] + 0.20
        no_leak = far["subj_shuffle_exact"] <= base + 0.05
        bind_learned = far["conj_untrained_exact"] <= chance + 0.10
        spikes_ok = far["zero_input_ok"] and far["hold_alive"] > 1e-3
        core = bool(gen and beats_both and above_chance and subj_preserved and cls_clean and wm_chan_lb and htm_chan_lb
                    and hold_lb and no_leak and bind_learned and spikes_ok)
        go = bool(core and not smoke)

        if not gen:
            verdict = (f"INCONCLUSIVE — L={far['L']} path space {far['path_space']:.0f} too small to hold out novel "
                       f"fillers; generalisation UNDEFINED. Increase n_fill/L.")
        elif core:
            tag = "GO" if go else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — SEPARATE-CHANNEL fusion CLOSES the PARTIAL: routing the WM's held-SUBJECT (wm[] column) "
                       f"and the HTM's local-CLASS (subject-agnostic clsrd channel) on DISTINCT channels, then binding "
                       f"them by a LEARNED dendritic conjunction, PRESERVES the subject (hybrid-sep subj "
                       f"{far['hybrid_sep_subj']:.3f} vs the old fusion's {far['old_fusion_subj']:.3f}) while keeping the "
                       f"class clean ({far['hybrid_sep_cls']:.3f}). Held-out (NOVEL fillers) exact branch(verb): "
                       f"HYBRID-SEP {far['hybrid_sep_exact']:.3f} >> old-fusion {far['old_fusion_exact']:.3f}, "
                       f">> HTM-alone {far['htm_exact']:.3f}, >> WM-alone {far['wm_exact']:.3f}, >> n-gram "
                       f"{far['ngram_floor_exact']:.3f}, >> chance {chance:.3f}. The conjunction is NEURAL (verb(s*,c*) "
                       f"is the unique double-driven cell -> apical plateau dominates single-channel verbs): "
                       f"lesion-the-WM-channel -> {far['lesion_wm_chan_exact']:.3f} (~HTM-alone), lesion-the-HTM-channel "
                       f"-> {far['lesion_htm_chan_exact']:.3f} (~WM-alone), BOTH load-bearing; lesion-the-hold -> "
                       f"{far['lesion_hold_exact']:.3f} (the subject channel reads the slot's spiking sustain, external "
                       f"input asserted zero); UNTRAINED conjunction -> {far['conj_untrained_exact']:.3f} (~chance, so "
                       f"the bind is LEARNED not host wiring); subject-shuffle -> {far['subj_shuffle_exact']:.3f} (no "
                       f"leakage). A WORKING WM+HTM hybrid. Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not beats_both: miss.append(f"hybrid-sep {far['hybrid_sep_exact']:.3f} not >= max(HTM {far['htm_exact']:.3f}, WM {far['wm_exact']:.3f})+0.20")
            if not above_chance: miss.append(f"hybrid-sep {far['hybrid_sep_exact']:.3f} not >= chance+0.30 ({chance:.3f})")
            if not subj_preserved: miss.append(f"subject NOT preserved (hybrid-sep subj {far['hybrid_sep_subj']:.3f} < 0.90)")
            if not cls_clean: miss.append(f"class not clean (hybrid-sep cls {far['hybrid_sep_cls']:.3f} < 0.90)")
            if not wm_chan_lb: miss.append(f"WM channel not cleanly load-bearing (lesion {far['lesion_wm_chan_exact']:.3f} vs HTM-alone {far['htm_exact']:.3f})")
            if not htm_chan_lb: miss.append(f"HTM channel not cleanly load-bearing (lesion {far['lesion_htm_chan_exact']:.3f} vs WM-alone {far['wm_exact']:.3f})")
            if not hold_lb: miss.append(f"hold not load-bearing (lesion-hold {far['lesion_hold_exact']:.3f})")
            if not no_leak: miss.append(f"leakage (subject-shuffle {far['subj_shuffle_exact']:.3f} > base+0.05)")
            if not bind_learned: miss.append(f"conjunction not learned (untrained {far['conj_untrained_exact']:.3f} > chance+0.10)")
            if not spikes_ok: miss.append(f"hold not alive/zero-input (alive {far['hold_alive']:.4f}, zero {far['zero_input_ok']})")
            verdict = ("HONEST NEGATIVE / PARTIAL — separate-channel fusion did not clear the GO bar at L={}: ".format(far["L"])
                       + "; ".join(miss) + ". Read the per-arm numbers (exact/subject/class) vs the old fusion to see "
                       "whether separating the channels recovered the subject and at what cost to the class.")
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = "ERROR — no points computed"

    # --- earned verdict preconditions (VALIDITY travels with the verdict) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("emerge_wm_hybrid_sepchan", chance=chance)
        if far is not None:
            Vd.require("oracle_task_solvable", 1.0, expect=lambda x: x > 0.99,
                       note="verb is determined by (subject, local-class) by construction -> the task IS solvable")
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out filler tuples disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("htm_alone_loses_subject", round(far["htm_subj"], 4), expect=lambda x: x <= 0.5,
                       note="HTM-alone must lose the long-range subject on novel fillers (its banked failure)")
            Vd.require("wm_alone_loses_class", round(far["wm_cls"], 4), expect=lambda x: x <= 1.0 / a.n_cls + 0.15,
                       note="WM-alone cannot see the local class (it only carries the subject)")
            Vd.require("old_fusion_corrupts_subject", round(far["old_fusion_subj"], 4), expect=lambda x: x <= 0.85,
                       note="the banked coincidence-AND fusion is LOSSY on the subject (the PARTIAL being closed)")
            Vd.require("conjunction_is_learned", round(far["conj_untrained_exact"], 4), expect=lambda x: x <= chance + 0.10,
                       note="an UNTRAINED conjunction bridge collapses to ~chance -> the bind lives in learned synapses")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across the hold+read span")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(far is not None and far["gen_defined"]
                   and far["hybrid_sep_exact"] >= max(far["htm_exact"], far["wm_exact"]) + 0.20
                   and far["hybrid_sep_exact"] >= chance + 0.30
                   and far["hybrid_sep_subj"] >= 0.90 and far["hybrid_sep_cls"] >= 0.90
                   and far["hybrid_sep_exact"] >= far["lesion_wm_chan_exact"] + 0.20
                   and far["hybrid_sep_exact"] >= far["lesion_htm_chan_exact"] + 0.20
                   and far["hybrid_sep_exact"] >= far["lesion_hold_exact"] + 0.20)
        dec = Vd.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_wm_hybrid_sepchan", "verdict": verdict, "backend": backend, "sim_backend": backend,
               "device": device, "cost_acknowledged": True, "smoke": smoke, "preconditions": preconditions,
               "mechanism": "SEPARATE-CHANNEL fusion. The WM slot's LATCHED subject (read from the slow-NMDA attractor's "
                            "spikes) selects its dedicated WM-MEMORY column wm[s] (the SUBJECT channel; the HTM does not "
                            "write it). The on-bridge HTM Temporal-Memory engine predicts the LOCAL CLASS on a SEPARATE, "
                            "SUBJECT-AGNOSTIC class-readout layer clsrd (n_cls columns, no subject dimension; the WM does "
                            "not write it) -- clsrd cannot carry a subject bias into the verb layer, which is the fix. A "
                            "conjunction bridge is TRAINED so verb(s,c) potentiates coincidence synapses from BOTH wm[s] "
                            "AND clsrd[c]; at decision time the substrate is PRIMED from {wm[subject_hat] U clsrd[class_hat]} "
                            "and the verb with max apical drive wins -- verb(s*,c*) is the unique double-driven cell "
                            "(apical plateau ~+17mV >> single-channel ~-62mV). NOT a host ensemble: the runner never does "
                            "verb=lookup(subject_hat,class_hat); the verb emerges from the LEARNED coincidence synapses "
                            "(an untrained conjunction -> chance). The only subject signal in the verb layer is "
                            "wm[subject_hat] -> the subject is PRESERVED.",
               "task": "compositional agreement stream [subject]+[L i.i.d. fillers]+[verb], verb=combine(subject, "
                       "local-class(last filler)); held-out TEST = disjoint NOVEL filler tuples; arms: htm_alone / "
                       "wm_alone / old_fusion (rung-3 reference) / hybrid_sep + lesion-wm-channel + lesion-htm-channel + "
                       "lesion-the-hold + subject-shuffle + conj_untrained + n-gram floor + chance + oracle",
               "seeds": a.seeds, "config": {"n_subj": a.n_subj, "n_fill": a.n_fill, "n_cls": a.n_cls, "distances": dists,
               "n_cells": a.n_cells, "k_win": a.k_win, "act_th": a.act_th, "and_th": a.and_th, "learn_th": a.learn_th,
               "epochs": a.epochs, "n_train": a.n_train, "n_test": a.n_test, "recur": a.recur, "thr_off_conj": a.thr_off_conj,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps, "chance": chance,
               "go_distance": go_L}, "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the rung-3 compositional stream + spiking WM slot + OLD-fusion arms; "
                              "the EMERGE-14 on-bridge HTM engine + banked WM slot. Change from rung 3 = ONLY the fusion "
                              "(coincidence-AND over shared verb columns -> separate WM/clsrd channels + a learned "
                              "conjunction) + the read-out. The class channel during TRAINING is the HTM's own branch "
                              "prediction; the subject channel during TRAINING is the TRUE subject (the WM working in "
                              "development, banked slot latch ~1.0); the decisive spiking-slot READ carries the subject "
                              "across NOVEL fillers at TEST. wm[] and clsrd[] labelled-line projections are fixed "
                              "topographic host wiring; the LATCH + READ + the coincidence bind are spiking/synaptic. "
                              "1-seed is a SMOKE indicator; the 6-seed sweep is decisive. NO sim/ edit."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[emerge_wm_hybrid_sepchan] VERDICT: {verdict}", flush=True)
    print(f"[emerge_wm_hybrid_sepchan] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
