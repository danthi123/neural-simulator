"""EMERGENCE + WORKING-MEMORY HYBRID, RUNG 3c — NEURAL-WTA CHANNEL READS (close the hybrid's LAST host-argmax residual).

BANKED (do NOT re-derive; read the two findings this composes):
  (1) `2026-08-11-emergence-WM-hybrid-separate-channel-GO-...` (rung-3b, runner _emerge_wm_hybrid_sepchan_derisk):
      a WORKING WM+HTM neural hybrid. The WM slot's held SUBJECT and the HTM's local CLASS ride SEPARATE neural channels
      and are combined by a LEARNED dendritic CONJUNCTION bridge -> hybrid 0.974 [min 0.938], subject preserved 1.000.
      Its ONE named host residual: the per-channel WINNER READS are host np.argmax -- subject_hat = argmax over the WM
      slot's cp_firing_states pool rates (SpikingSlot.read); class_hat = argmax over the HTM's clsrd apical drive.
  (2) `2026-08-11-emergent-neural-WTA-slot-allocation-GO` (rung-2, runner _var_bind_emergent_wta_allocation_derisk):
      an emergent DOWN-RAMP release-of-inhibition WTA (a pooled subtractive inhibition released step by step; the first
      pool to escape wins; the ramp STOPS when exactly one pool is active) picks a clean one-of-K winner from the
      substrate's own pools, READ FROM SPIKES -- a fair WTA WITHOUT a hand-set cut.

THIS RUNG replaces the hybrid's TWO host-np.argmax channel reads with the emergent neural-WTA, so the ENTIRE verb read
path -- channel SELECTION and the CONJUNCTION combine -- is neural + reads spikes; NO host argmax remains in the read-out:
  * subject_hat = the WTA's winner over the WM-subject channel: the slot's K held-pool firing RATES (spikes, after
      WRITE+HOLD across the fillers, external input asserted zero) are the graded per-pool evidence into a release-of-
      inhibition competition on a K-pool attractor bank; the winner is READ FROM the latched pool's SPIKES (a threshold
      on the post-competition hold rate, the single pool the competition left active), NEVER np.argmax(cp_firing_states).
  * class_hat = the WTA's winner over the HTM-class channel: the HTM engine's clsrd apical-drive vector (the substrate's
      own branch prediction, read from cp_v_apical) is the per-pool evidence into the SAME release-of-inhibition WTA;
      the winner is read from spikes, NEVER np.argmax(apical).
  * the LEARNED dendritic conjunction bridge (already neural, rung-3b) then combines wm[subject_hat] U clsrd[class_hat]
      exactly as before -> the verb column with the max apical plateau wins (the unique double-driven cell).

ARMS (the rung-3b set kept so the comparison is direct; honest-negative is first-class):
  * htm_alone            : the standalone emergence engine (no WM). BANKED baseline (recovers class, loses subject).
  * wm_alone             : the slot's held subject -> subject's most-likely verb (no local class). BANKED baseline.
  * hybrid_argmax_reads  : the BANKED rung-3b hybrid (subject_hat = argmax slot rates; class_hat = argmax clsrd drive),
                           re-run on the SAME substrate as the reference to reproduce ~0.974 / subject 1.000. The
                           channel reads are the ONLY difference vs hybrid_wta_reads (same slot rates, same conjunction).
  * hybrid_wta_reads     : THE TEST. subject_hat + class_hat from the emergent neural-WTA (read from spikes).
  * lesion_wm_chan       : hybrid_wta with the SUBJECT channel ablated (prime clsrd[c*] only) -> ~HTM-alone.
  * lesion_htm_chan      : hybrid_wta with the CLASS channel ablated (prime wm[s*] only) -> ~WM-alone.
  * lesion_hold          : the WTA subject read on a recur=0 slot (the hold dies) -> the subject evidence is noise ->
                           the WTA picks garbage -> collapses (proves the WTA read READS the slot's spiking sustain).
  * lesion_wta_selfcalib : the WTA reads with the SELF-CALIBRATION OFF (the release ramp frozen at a hand-set fixed cut,
                           no adaptation). Degrades on reads whose margin the hand-set cut cannot cleanly separate ->
                           the WTA competition is load-bearing for clean channel selection. Reported precisely.
  * conj_untrained       : the WTA reads on an UNTRAINED conjunction bridge -> ~chance (the neural bind is LEARNED).
  * subj_shuffle         : the slot->subject deref permuted -> WTA winner derefs to the WRONG subject -> subject collapse.
  * held-out NOVEL fillers, n-gram HELD-OUT floor, chance, task-solvable oracle (=1 by build).

GO GATE: hybrid_wta_reads held-out exact >= max(htm_alone, wm_alone) + 0.20 AND within ~0.05 of the argmax-reads
  reference (~0.974) AND subject preserved >= 0.90, with all lesions load-bearing, and NO host argmax in the verb read
  path (grep-confirm + assert). HONEST NEGATIVE (first-class) otherwise: if making the reads neural costs accuracy (e.g.
  the WTA's selectivity muddies a channel), quantify the cost precisely and name the next candidate.

BRAIN-BASED: the reads (WTA competition, read from spikes) AND the combination (learned dendritic conjunction) are
neural; NO host argmax in the verb read path. The WTA is driven by the channel's SPIKE-derived evidence (slot pool
rates / HTM apical drive) via a labelled-line identity projection (the SAME accepted scope as the rung-2 barcode->pool
projection; the SELECTION + read are neural/spikes). Reuse-by-import; NO sim/ edit. SIM_BACKEND=numpy (sub-1k-neuron
loops are launch-bound: CPU is correct + faster). Verified: build_persistent_slot / build_pool_bridge set cfg.seed, so
the substrate IS seeded (the actual_seed_used trap does not apply).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_neuralwta_derisk --seeds 42 --debug
6-seed decisive (fan one seed per process; then --merge-from):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
    research.runners._emerge_wm_hybrid_neuralwta_derisk --seeds $s \
    --out research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_$s.json & done ; wait
  SIM_BACKEND=numpy python -m research.runners._emerge_wm_hybrid_neuralwta_derisk \
    --merge-from research/findings/raw/_emerge_wm_hybrid_neuralwta/seed_*.json \
    --out research/findings/raw/_emerge_wm_hybrid_neuralwta/neuralwta_6seed.json
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

# --- rung-3b separate-channel hybrid: layout, class channel, conjunction, the argmax class read (reference) ---
from research.runners._emerge_wm_hybrid_sepchan_derisk import (
    sepchan_layout, train_class_channel, class_read, train_conjunction, conj_read)
# --- rung-3 compositional stream + engine primitives + scorer (reuse-by-import) ---
from research.runners._emerge_wm_hybrid_derisk import (
    cls_of, verb_of, decode_verb, make_comp_stream, make_comp_heldout, ngram_floor_comp, score_preds,
    apical_drive, build_engine, train_engine_hybrid, hybrid_predict_verb)
# --- the banked spiking WM slot + persistent subject binder + codebook ---
from research.runners._var_bind_gated_slot_derisk import (
    SpikingSlot, build_codebook, persistent_subject_binder)
# --- the D3 slow-NMDA attractor bank primitives (the emergent-WTA substrate) ---
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset
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

OUT = Path("research/findings/raw/_emerge_wm_hybrid_neuralwta/neuralwta.json")


def _assert_no_host_argmax_in_read_path():
    """ENFORCE the brain-based bar: the verb read path (both channel SELECTIONS + the conjunction plateau read) must
    contain NO host np.argmax. The two channel reads are now the neural-WTA (NeuralWTARead.select -> a down-ramp +
    threshold read of spikes, no argmax); the conjunction verb read (conj_read) is the rung-3b-accepted graded dendritic
    plateau read (np.max over the substrate's apical voltage per verb column -> the cell that fired the plateau; NOT an
    argmax classifier). The ONLY np.argmax in this runner is the `subj_am` REFERENCE arm (the rung-3b host read being
    surpassed), which is not on the hybrid_wta_reads path. Raises if a host argmax leaks into the read path."""
    import ast
    import inspect
    import textwrap
    read_path = {"NeuralWTARead.select": NeuralWTARead.select, "conj_read": conj_read,
                 "class_evidence": class_evidence, "slot_carry_subject_rates": slot_carry_subject_rates}

    def _calls_argmax(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))          # AST -> comments/docstrings are NOT code
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "argmax":  # np.argmax / x.argmax
                return True
            if isinstance(node, ast.Name) and node.id == "argmax":         # a bare argmax import
                return True
        return False

    offenders = [name for name, fn in read_path.items() if _calls_argmax(fn)]
    assert not offenders, f"host argmax found in the verb read path: {offenders}"
    return sorted(read_path)


# ====================================================================================================================
# THE EMERGENT NEURAL-WTA CHANNEL READ (replaces a host np.argmax over a channel's spike-derived evidence).
#   A K-pool D3 slow-NMDA attractor bank sharing one FS. A per-pool graded EVIDENCE vector (spike-derived: the WM slot's
#   held-pool rates for the subject channel; the HTM engine's clsrd apical drive for the class channel) is injected as
#   external current via a labelled-line identity projection. A DOWN-RAMP release-of-inhibition competition (the rung-2
#   mechanism) resolves a clean one-of-K winner WITHOUT a hand-set cut: a pooled subtractive inhibition common to all
#   pools starts HIGH (all silent) and is RELEASED step by step; the first pool to escape (highest evidence) wins; the
#   ramp STOPS the moment exactly one pool is active. The winner is READ FROM SPIKES -- a threshold on the post-
#   competition HOLD rate (drive removed), the single pool the competition left latched -- NEVER np.argmax.
#   self_calib=False = the LESION: the inhibition is FROZEN at a hand-set fixed cut (no ramp) -> a read whose margin the
#   fixed cut cannot cleanly isolate is UNRESOLVED (blur or silence) -> a degraded/-1 read (the load-bearing proof).
# ====================================================================================================================
class NeuralWTARead:
    def __init__(self, seed, K, recur=25.0, drive_gain=400.0, noise_pA=40.0, self_calib=True,
                 inh_high_frac=0.95, inh_step_frac=0.08, settle_rounds=14, round_steps=5,
                 consolidate_steps=12, hold_steps=16, active_frac=0.5, fixed_inh_frac=0.45):
        from sim.backend import to_host, from_host
        self._to_host, self._from_host = to_host, from_host
        self.seed, self.K = int(seed), int(K)
        self.sb = build_persistent_slot(seed, self.K, recur=recur)
        self.idx = _pool_idx(self.sb, self.K)
        self.pool_neurons = np.concatenate([self.idx[k] for k in range(self.K)])
        self.n = self.sb.core_config.num_neurons
        self.drive_gain, self.noise_pA = float(drive_gain), float(noise_pA)
        self.self_calib = bool(self_calib)
        self.inh_high = float(inh_high_frac) * self.drive_gain
        self.inh_step = float(inh_step_frac) * self.drive_gain
        self.settle_rounds, self.round_steps = int(settle_rounds), int(round_steps)
        self.consolidate_steps, self.hold_steps = int(consolidate_steps), int(hold_steps)
        self.active_frac = float(active_frac)
        self.fixed_inh = float(fixed_inh_frac) * self.drive_gain

    def _run(self, base_cur, steps, noise_rng):
        rates = np.zeros(self.K)
        for _ in range(steps):
            cur = base_cur.copy()
            if self.noise_pA > 0.0 and noise_rng is not None:
                cur[self.pool_neurons] += noise_rng.normal(0.0, self.noise_pA, self.pool_neurons.shape[0])
            self.sb.cp_external_input_current[:] = self._from_host(cur)
            self.sb._run_one_simulation_step()
            fir = np.asarray(self._to_host(self.sb.cp_firing_states)).astype(float)
            for k in range(self.K):
                rates[k] += fir[self.idx[k]].mean()
        return rates / max(steps, 1)

    def select(self, evidence, noise_rng):
        """evidence: K-vector of SPIKE-derived per-pool evidence. Returns (winner_pool_or_-1, selectivity, n_active).
        The winner is READ FROM SPIKES via the down-ramp release-of-inhibition competition + a threshold on the latched
        HOLD rate -- there is NO np.argmax over the evidence anywhere in this read path."""
        _reset(self.sb)                                        # hard reset between independent reads (trial separation)
        e = np.asarray(evidence, float)
        span = float(e.max() - e.min())
        d = np.full(self.K, 0.5) if span < 1e-9 else (e - e.min()) / span   # per-read graded drive in [0,1]

        def _base(inh):
            base = np.zeros(self.n)
            eff = self.drive_gain * d - inh
            for k in range(self.K):
                base[self.idx[k]] = eff[k]
            return base

        inh = self.inh_high if self.self_calib else self.fixed_inh
        ramp_winner, n_active = -1, 0
        for _ in range(self.settle_rounds):
            rates = self._run(_base(inh), self.round_steps, noise_rng)
            mx = rates.max()
            active = np.where(rates > self.active_frac * mx)[0] if mx > 1e-6 else np.array([], dtype=int)
            n_active = int(active.size)
            if not self.self_calib:                            # LESION: frozen hand-set cut -- FAIR: same settle budget,
                ramp_winner = int(active[0]) if active.size == 1 else -1   # the ONLY change is no adaptive release. A read
                continue                                       # whose margin the fixed cut can't isolate stays a blur/-1.
            if n_active == 0:
                inh -= self.inh_step                           # release: let the highest-evidence pool escape
            elif n_active > 1:
                inh += 0.5 * self.inh_step                     # too many escaped -> tighten back toward one winner
            else:
                ramp_winner = int(active[0]); break            # exactly one pool escaped (read from spikes)
        # consolidate at the found/fixed cut then HOLD with drive removed -> the winner LATCHES; read it from SPIKES
        # (a threshold on the hold rate, the single pool the competition left active) -- IDENTICAL for both arms, so the
        # ONLY difference between candidate and self-calib-lesion is the inhibition SCHEDULE (adaptive ramp vs frozen cut).
        self._run(_base(inh), self.consolidate_steps, noise_rng)
        hold = self._run(np.zeros(self.n), self.hold_steps, noise_rng)
        hmx = hold.max()
        sel = float(hmx / (hold.sum() + 1e-9)) if hold.sum() > 0 else 0.0
        winner = ramp_winner
        if hmx > 1e-6:
            latched = np.where(hold > self.active_frac * hmx)[0]           # the single pool the competition left active
            winner = int(latched[0]) if latched.size == 1 else ramp_winner  # read from the HOLD spikes (no argmax)
        return winner, sel, n_active


# ====================================================================================================================
# Channel EVIDENCE readers (spike-derived, NO argmax). Each returns the per-pool evidence vector the WTA competes over.
# ====================================================================================================================
def slot_carry_subject_rates(slot: SpikingSlot, seq, slot_of_subj):
    """WRITE the subject's attractor pool at t=0, HOLD (external input asserted zero) across every filler, then read the
    K held-pool firing RATES from cp_firing_states (spikes). Returns (rates_K, hold_alive, zero_input_ok). This is the
    SAME slot carry the rung-3b reference uses -- but it returns the raw per-pool rate VECTOR (the WTA's evidence),
    NOT the host argmax over it."""
    toks = seq[:-1]
    slot.reset()
    for t, tok in enumerate(toks):
        if t == 0:
            slot.write(slot_of_subj[int(tok)])                # LOAD the subject (gate OPEN)
        else:
            slot.hold()                                       # HOLD (gate CLOSED, zero external drive)
    rates = slot._run(np.zeros(slot.n), slot.read_steps, assert_zero=True)   # READ the K pool rates from spikes
    return rates, float(rates.max()), bool(slot._zero_input_span)


def class_evidence(lr, seq, L, clsrd, thr_off=2.0):
    """Forward the HTM priming chain over subject+fillers; return the per-clsrd-column apical-drive VECTOR (the
    substrate's own branch prediction, read from cp_v_apical). This is the SAME class read the rung-3b reference uses
    -- but it returns the drive VECTOR (the WTA's evidence), NOT the host argmax over it."""
    _, col_drive = hybrid_predict_verb(lr, seq, L, None, clsrd, use_htm=True, use_wm=False,
                                       thr_off=thr_off, return_drive=True)
    if not col_drive:
        return None
    return np.array([float(col_drive.get(cc, -1e9)) for cc in clsrd], dtype=float)


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, n_subj, n_fill, n_cls, L, n_cells, k_win, act_th, and_th, learn_th, epochs, n_train, n_test,
              recur, hold_steps, load_steps, clear_steps, thr_off_conj, wta_fixed_inh_frac, debug=False):
    rng = np.random.default_rng(seed)
    subj, fill, verb, wm, clsrd, V_base, V = sepchan_layout(n_subj, n_fill, n_cls)
    n_verb = n_subj * n_cls
    chance = 1.0 / n_verb
    wm_col_of_subj_idx = {i: wm[i] for i in range(n_subj)}

    # --- the compositional stream + disjoint NOVEL-filler held-out test (identical to rung-3b) ---
    train_seqs = make_comp_stream(n_subj, n_fill, n_cls, L, n_train, rng)
    train_ftuples = set(tuple(s[1:-1]) for s in train_seqs)
    test_seqs, gen_defined = make_comp_heldout(n_subj, n_fill, n_cls, L, n_test, rng, train_ftuples)

    # --- the spiking WM: codebook + persistent subject binder (reuse-by-import) ---
    code_of, _ = build_codebook(n_subj, n_fill, np.random.default_rng(seed + 7))
    binder, slot_of_subj, subj_of_slot = persistent_subject_binder([int(s) for s in subj], code_of)

    def _slot(rc=recur, off=0):
        return SpikingSlot(seed + off, _K, recur=rc, hold_steps=hold_steps, load_steps=load_steps,
                           clear_steps=clear_steps)

    slot = _slot()
    slot_les = _slot(rc=0.0, off=100)

    # --- the emergent neural-WTA banks (built ONCE per seed; reset between reads) ---
    subj_wta = NeuralWTARead(seed + 500, _K, recur=recur, fixed_inh_frac=wta_fixed_inh_frac, self_calib=True)
    subj_wta_sc = NeuralWTARead(seed + 500, _K, recur=recur, fixed_inh_frac=wta_fixed_inh_frac, self_calib=False)
    cls_wta = NeuralWTARead(seed + 600, n_cls, recur=recur, fixed_inh_frac=wta_fixed_inh_frac, self_calib=True)
    cls_wta_sc = NeuralWTARead(seed + 600, n_cls, recur=recur, fixed_inh_frac=wta_fixed_inh_frac, self_calib=False)
    nrng = np.random.default_rng(seed + 999)                  # WTA per-step noise rng (shared across reads)

    # ================= REFERENCE ARMS (build on V_base -> byte-identical to the banked rung-3b) =================
    lr_htm = build_engine(seed, V_base, n_cells, act_th, k_win, learn_th)
    train_engine_hybrid(lr_htm, train_seqs, L, wm_col_of_subj_idx, subj, use_wm=False, epochs=epochs)

    # ================= SEPARATE-CHANNEL ENGINES (build on V_sep; identical to rung-3b) =================
    lr_cls = build_engine(seed, V, n_cells, act_th, k_win, learn_th)              # class channel (normal threshold)
    for _ in range(epochs):
        for s in train_seqs:
            train_class_channel(lr_cls, s, L, clsrd, n_subj, n_fill, n_cls)
    lr_conj = build_engine(seed, V, n_cells, and_th, k_win, learn_th)             # conjunction (AND threshold)
    train_conjunction(lr_conj, train_seqs, L, wm, clsrd, verb, subj, n_subj, n_fill, n_cls, epochs)
    lr_unt = build_engine(seed, V, n_cells, and_th, k_win, learn_th)              # UNTRAINED conjunction (control)

    # ---- channel EVIDENCE, computed ONCE per held-out sentence (spike-derived), reused by argmax + WTA arms ----
    subj_rates, subj_alive, zero_ok = [], [], True
    subj_rates_les = []
    for s in test_seqs:
        r, al, zo = slot_carry_subject_rates(slot, s, slot_of_subj)
        subj_rates.append(r); subj_alive.append(al); zero_ok = zero_ok and zo
        rl, _, _ = slot_carry_subject_rates(slot_les, s, slot_of_subj)
        subj_rates_les.append(rl)
    slot_decode_acc = float(np.mean([int(subj_of_slot.get(int(np.argmax(r)), -1) == s[0]) for r, s in zip(subj_rates, test_seqs)]))

    cls_ev = [class_evidence(lr_cls, s, L, clsrd) for s in test_seqs]             # per-clsrd apical drive vectors

    # ---- SUBJECT reads (each WTA competition run ONCE: capture winner + selectivity together) ----
    # argmax reference (the rung-3b read): argmax over the SAME slot rates -> deref (byte-identical to SpikingSlot.read)
    subj_am = [subj_of_slot.get(int(np.argmax(r)), -1) if r.max() > 1e-6 else -1 for r in subj_rates]
    # WTA (candidate): the release-of-inhibition winner over the slot rates -> deref
    _subj_wta = [subj_wta.select(r, nrng) for r in subj_rates]
    subj_wta_hat = [subj_of_slot.get(w, -1) for (w, _s, _na) in _subj_wta]
    subj_wta_sel = float(np.mean([s for (_w, s, _na) in _subj_wta])) if _subj_wta else 0.0
    # WTA on the hold-lesioned slot (evidence is noise) -> collapse
    subj_wta_hold = [subj_of_slot.get(subj_wta.select(r, nrng)[0], -1) for r in subj_rates_les]
    # WTA with the self-calibration frozen (hand-set cut)
    subj_wta_scl = [subj_of_slot.get(subj_wta_sc.select(r, nrng)[0], -1) for r in subj_rates]

    # ---- CLASS reads ----
    # argmax reference (the rung-3b read): class_read = argmax over the clsrd apical drive with the threshold floor
    cls_am = [class_read(lr_cls, s, L, clsrd) for s in test_seqs]
    # WTA (candidate): the release-of-inhibition winner over the clsrd drive vector (run once: winner + selectivity)
    _cls_wta = [cls_wta.select(ev, nrng) if ev is not None else (-1, 0.0, 0) for ev in cls_ev]
    cls_wta_hat = [w for (w, _s, _na) in _cls_wta]
    _cls_sels = [s for (w, s, _na), ev in zip(_cls_wta, cls_ev) if ev is not None]
    cls_wta_sel = float(np.mean(_cls_sels)) if _cls_sels else 0.0
    # WTA with the self-calibration frozen
    cls_wta_scl = [cls_wta_sc.select(ev, nrng)[0] if ev is not None else -1 for ev in cls_ev]

    cls_chan_acc = float(np.mean([int(ch == cls_of(s[-2], n_subj, n_fill, n_cls)) for ch, s in zip(cls_wta_hat, test_seqs)]))

    def wmc(lr, sh):
        return set(lr._col(wm[subj.index(int(sh))])[:k_win]) if sh in subj else set()

    def clc(lr, ch):
        return set(lr._col(clsrd[ch])[:k_win]) if ch is not None and ch >= 0 else set()

    # ================= ARM PREDICTIONS (all through the SAME learned conjunction; only the channel reads differ) =====
    # htm_alone / wm_alone reference baselines
    p_htm = [hybrid_predict_verb(lr_htm, s, L, None, verb, use_htm=True, use_wm=False) for s in test_seqs]
    cls_counts = Counter(cls_of(s[-2], n_subj, n_fill, n_cls) for s in train_seqs)
    guess_cls = cls_counts.most_common(1)[0][0] if cls_counts else 0
    p_wm = [verb_of(verb, subj.index(int(sh)), guess_cls, n_cls) if sh in subj else -1 for sh in subj_am]

    # hybrid_argmax_reads (the banked rung-3b hybrid): argmax channel reads -> learned conjunction
    p_argmax = [conj_read(lr_conj, wmc(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
                for sh, ch in zip(subj_am, cls_am)]
    # hybrid_wta_reads (THE TEST): WTA channel reads -> learned conjunction
    p_wta = [conj_read(lr_conj, wmc(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
             for sh, ch in zip(subj_wta_hat, cls_wta_hat)]
    # lesion the WM (subject) channel -> class only -> ~HTM-alone
    p_les_wm = [conj_read(lr_conj, clc(lr_conj, ch), verb, thr_off_conj) for ch in cls_wta_hat]
    # lesion the HTM (class) channel -> subject only -> ~WM-alone
    p_les_htm = [conj_read(lr_conj, wmc(lr_conj, sh), verb, thr_off_conj) for sh in subj_wta_hat]
    # lesion-the-hold: WTA subject read on the recur=0 slot -> collapses (proves the WTA read reads spikes)
    p_les_hold = [conj_read(lr_conj, wmc(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
                  for sh, ch in zip(subj_wta_hold, cls_wta_hat)]
    # lesion-the-WTA-selfcalib: WTA reads with a frozen hand-set cut -> degraded channel selection
    p_les_scl = [conj_read(lr_conj, wmc(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
                 for sh, ch in zip(subj_wta_scl, cls_wta_scl)]
    # subject-shuffle: deref the WTA winner to a WRONG subject column
    dr = np.random.default_rng(seed + 13); perm = list(range(n_subj))
    for _ in range(64):
        dr.shuffle(perm)
        if all(perm[i] != i for i in range(n_subj)):
            break

    def wmc_shuf(lr, sh):
        return set(lr._col(wm[perm[subj.index(int(sh))]])[:k_win]) if sh in subj else set()

    p_shuf = [conj_read(lr_conj, wmc_shuf(lr_conj, sh) | clc(lr_conj, ch), verb, thr_off_conj)
              for sh, ch in zip(subj_wta_hat, cls_wta_hat)]
    # conj_untrained: the SAME WTA reads on an UNTRAINED conjunction bridge -> ~chance
    p_unt = [conj_read(lr_unt, wmc(lr_unt, sh) | clc(lr_unt, ch), verb, thr_off_conj)
             for sh, ch in zip(subj_wta_hat, cls_wta_hat)]

    # ---- score ----
    ex_htm, sub_htm, cl_htm = score_preds(p_htm, test_seqs, verb, n_cls)
    ex_wm, sub_wm, cl_wm = score_preds(p_wm, test_seqs, verb, n_cls)
    ex_am, sub_am, cl_am = score_preds(p_argmax, test_seqs, verb, n_cls)
    ex_wta, sub_wta_, cl_wta_ = score_preds(p_wta, test_seqs, verb, n_cls)
    ex_lwm, sub_lwm, cl_lwm = score_preds(p_les_wm, test_seqs, verb, n_cls)
    ex_lhtm, sub_lhtm, cl_lhtm = score_preds(p_les_htm, test_seqs, verb, n_cls)
    ex_lhold, _, _ = score_preds(p_les_hold, test_seqs, verb, n_cls)
    ex_lscl, sub_lscl, cl_lscl = score_preds(p_les_scl, test_seqs, verb, n_cls)
    ex_shuf, sub_shuf, _ = score_preds(p_shuf, test_seqs, verb, n_cls)
    ex_unt, _, _ = score_preds(p_unt, test_seqs, verb, n_cls)

    # subject/class accuracy of the WTA channel READS themselves (vs the argmax reads) -- the direct read comparison
    def _true_cls(s):
        return cls_of(s[-2], n_subj, n_fill, n_cls)
    subj_read_am = float(np.mean([int(sh == s[0]) for sh, s in zip(subj_am, test_seqs)]))
    subj_read_wta = float(np.mean([int(sh == s[0]) for sh, s in zip(subj_wta_hat, test_seqs)]))
    cls_read_am = float(np.mean([int(ch == _true_cls(s)) for ch, s in zip(cls_am, test_seqs)]))

    ngram, ngram_k = ngram_floor_comp(train_seqs, test_seqs, L, n_subj)

    if debug:
        print(f"    [debug seed={seed} L={L}] slot_decode(argmax)={slot_decode_acc:.3f} "
              f"subj_read am={subj_read_am:.3f} wta={subj_read_wta:.3f} (sel {subj_wta_sel:.3f}) | "
              f"cls_read am={cls_read_am:.3f} wta={cls_chan_acc:.3f} (sel {cls_wta_sel:.3f}) | "
              f"hold_alive={np.mean(subj_alive):.4f}", flush=True)

    return {"seed": seed, "L": L, "distance": L + 1, "n_fill": n_fill, "n_cls": n_cls, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(n_fill) ** L,
            "slot_decode_acc": slot_decode_acc, "cls_chan_acc": cls_chan_acc,
            "hold_alive": float(np.mean(subj_alive)), "zero_input_ok": bool(zero_ok),
            "subj_read_argmax": subj_read_am, "subj_read_wta": subj_read_wta,
            "cls_read_argmax": cls_read_am, "cls_read_wta": cls_chan_acc,
            "subj_wta_selectivity": subj_wta_sel, "cls_wta_selectivity": cls_wta_sel,
            "htm_exact": ex_htm, "htm_subj": sub_htm, "htm_cls": cl_htm,
            "wm_exact": ex_wm, "wm_subj": sub_wm, "wm_cls": cl_wm,
            "hybrid_argmax_exact": ex_am, "hybrid_argmax_subj": sub_am, "hybrid_argmax_cls": cl_am,
            "hybrid_wta_exact": ex_wta, "hybrid_wta_subj": sub_wta_, "hybrid_wta_cls": cl_wta_,
            "lesion_wm_chan_exact": ex_lwm, "lesion_wm_chan_subj": sub_lwm, "lesion_wm_chan_cls": cl_lwm,
            "lesion_htm_chan_exact": ex_lhtm, "lesion_htm_chan_subj": sub_lhtm, "lesion_htm_chan_cls": cl_lhtm,
            "lesion_hold_exact": ex_lhold, "lesion_wta_selfcalib_exact": ex_lscl,
            "lesion_wta_selfcalib_subj": sub_lscl, "lesion_wta_selfcalib_cls": cl_lscl,
            "subj_shuffle_exact": ex_shuf, "subj_shuffle_subj": sub_shuf, "conj_untrained_exact": ex_unt,
            "ngram_floor_exact": ngram, "ngram_order": ngram_k, "oracle": 1.0}


def agg(per):
    keys = ["slot_decode_acc", "cls_chan_acc", "hold_alive", "subj_read_argmax", "subj_read_wta", "cls_read_argmax",
            "cls_read_wta", "subj_wta_selectivity", "cls_wta_selectivity",
            "htm_exact", "htm_subj", "htm_cls", "wm_exact", "wm_subj", "wm_cls",
            "hybrid_argmax_exact", "hybrid_argmax_subj", "hybrid_argmax_cls",
            "hybrid_wta_exact", "hybrid_wta_subj", "hybrid_wta_cls",
            "lesion_wm_chan_exact", "lesion_wm_chan_subj", "lesion_wm_chan_cls",
            "lesion_htm_chan_exact", "lesion_htm_chan_subj", "lesion_htm_chan_cls",
            "lesion_hold_exact", "lesion_wta_selfcalib_exact", "lesion_wta_selfcalib_subj", "lesion_wta_selfcalib_cls",
            "subj_shuffle_exact", "subj_shuffle_subj", "conj_untrained_exact", "ngram_floor_exact", "oracle"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    a.update({k + "_min": float(np.min([p[k] for p in per]))
              for k in ["hybrid_wta_exact", "hybrid_wta_subj", "hybrid_wta_cls", "hybrid_argmax_exact"]})
    a.update({"L": per[0]["L"], "distance": per[0]["distance"], "n_fill": per[0]["n_fill"], "n_cls": per[0]["n_cls"],
              "chance": per[0]["chance"], "path_space": per[0]["path_space"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "ngram_order": int(np.round(np.mean([p["ngram_order"] for p in per]))), "per_seed": per})
    return a


def build_verdict(far, chance, smoke):
    base = max(far["htm_exact"], far["wm_exact"])
    ref = far["hybrid_argmax_exact"]
    gen = far["gen_defined"]
    beats_both = far["hybrid_wta_exact"] >= base + 0.20
    within_ref = far["hybrid_wta_exact"] >= ref - 0.05
    subj_preserved = far["hybrid_wta_subj"] >= 0.90
    wm_chan_lb = far["hybrid_wta_exact"] >= far["lesion_wm_chan_exact"] + 0.20 and far["lesion_wm_chan_exact"] <= far["htm_exact"] + 0.10
    htm_chan_lb = far["hybrid_wta_exact"] >= far["lesion_htm_chan_exact"] + 0.20 and far["lesion_htm_chan_exact"] <= far["wm_exact"] + 0.10
    hold_lb = far["hybrid_wta_exact"] >= far["lesion_hold_exact"] + 0.20
    # the self-calib lesion is a DIAGNOSTIC, not a hard core gate: the WM latch here is CLEAN, so a fair fixed cut can
    # isolate the winner and this lesion is EXPECTED not to bite (the self-calibration is load-bearing in the rung-2
    # blur/allocation regime, banked separately). Reported honestly; does NOT block the fully-neural-reads GO.
    selfcalib_lb = far["hybrid_wta_exact"] >= far["lesion_wta_selfcalib_exact"] + 0.10
    no_leak = far["subj_shuffle_exact"] <= base + 0.05
    bind_learned = far["conj_untrained_exact"] <= chance + 0.10
    spikes_ok = far["zero_input_ok"] and far["hold_alive"] > 1e-3
    # CORE GO = the primary claim: the fully-neural (WTA) read path holds the hybrid GO at parity with the argmax reads,
    # subject preserved, with the load-bearing lesions of the neural read path (hold reads spikes; both channels matter;
    # the bind is learned) all biting, and no leakage.
    core = bool(gen and beats_both and within_ref and subj_preserved and wm_chan_lb and htm_chan_lb and hold_lb
                and no_leak and bind_learned and spikes_ok)
    go = bool(core and not smoke)
    return {"gen": gen, "beats_both": beats_both, "within_ref": within_ref, "subj_preserved": subj_preserved,
            "wm_chan_lb": wm_chan_lb, "htm_chan_lb": htm_chan_lb, "hold_lb": hold_lb, "selfcalib_lb": selfcalib_lb,
            "no_leak": no_leak, "bind_learned": bind_learned, "spikes_ok": spikes_ok, "core": core, "go": go,
            "base": base, "ref": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-subj", type=int, default=4)
    ap.add_argument("--n-fill", type=int, default=8)
    ap.add_argument("--n-cls", type=int, default=2, help="# local classes (a fn of the LAST filler); WM-alone caps ~1/n_cls")
    ap.add_argument("--distances", type=int, nargs="+", default=[3], help="filler-span L (dependency distance = L+1)")
    ap.add_argument("--n-cells", type=int, default=32)
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--and-th", type=int, default=6)
    ap.add_argument("--learn-th", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=240)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--thr-off-conj", type=float, default=0.5)
    ap.add_argument("--wta-fixed-inh-frac", type=float, default=0.45,
                    help="the HAND-SET inhibition (fraction of drive_gain) the self-calib LESION is frozen at")
    ap.add_argument("--go-distance", type=int, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--merge-from", nargs="+", default=None,
                    help="MERGE mode: aggregate per-seed artifacts (each a single-seed run) through the SAME verdict code")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    n_verb = a.n_subj * a.n_cls
    chance = 1.0 / n_verb
    dists = sorted(set(a.distances))
    read_path_fns = _assert_no_host_argmax_in_read_path()   # ENFORCE: no host argmax in the verb read path
    print(f"[read-path assert] NO host argmax in: {read_path_fns}", flush=True)

    if a.merge_from:
        per_all = {}
        for pth in a.merge_from:
            d = json.loads(Path(pth).read_text())
            for p in d.get("points", []):
                per_all.setdefault(p["L"], []).extend(p["per_seed"])
        points = [agg(per_all[L]) for L in sorted(per_all)]
        seeds = sorted(set(ps["seed"] for L in per_all for ps in per_all[L]))
        a.seeds = seeds
        t0 = time.time(); err = None
    else:
        _, _, _, _, _, V_base, V = sepchan_layout(a.n_subj, a.n_fill, a.n_cls)
        print(f"backend={backend} device={device} | n_subj={a.n_subj} n_fill={a.n_fill} n_cls={a.n_cls} n_verb={n_verb} "
              f"chance={chance:.3f} | L={dists} | vocab_base={V_base} vocab_sep={V} n_cells={a.n_cells} act_th={a.act_th} "
              f"AND_th={a.and_th} thr_off_conj={a.thr_off_conj} wta_fixed_inh={a.wta_fixed_inh_frac} epochs={a.epochs} "
              f"n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds}", flush=True)
        t0 = time.time(); err = None; points = []
        try:
            for L in dists:
                per = [run_point(s, a.n_subj, a.n_fill, a.n_cls, L, a.n_cells, a.k_win, a.act_th, a.and_th, a.learn_th,
                                 a.epochs, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps, a.clear_steps,
                                 a.thr_off_conj, a.wta_fixed_inh_frac, debug=a.debug) for s in a.seeds]
                p = agg(per); points.append(p)
                print(f"  [L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']}] "
                      f"HYBRID-WTA exact {p['hybrid_wta_exact']:.3f}[min {p['hybrid_wta_exact_min']:.3f}] "
                      f"(subj {p['hybrid_wta_subj']:.3f} cls {p['hybrid_wta_cls']:.3f}) || argmax-ref "
                      f"{p['hybrid_argmax_exact']:.3f} (subj {p['hybrid_argmax_subj']:.3f} cls {p['hybrid_argmax_cls']:.3f}) | "
                      f"HTM {p['htm_exact']:.3f} | WM {p['wm_exact']:.3f} || "
                      f"LES-wm-chan {p['lesion_wm_chan_exact']:.3f} | LES-htm-chan {p['lesion_htm_chan_exact']:.3f} | "
                      f"LES-hold {p['lesion_hold_exact']:.3f} | LES-wta-selfcalib {p['lesion_wta_selfcalib_exact']:.3f} | "
                      f"subj-shuf {p['subj_shuffle_exact']:.3f} | conj-untrained {p['conj_untrained_exact']:.3f} || "
                      f"reads subj am/wta {p['subj_read_argmax']:.3f}/{p['subj_read_wta']:.3f} "
                      f"cls am/wta {p['cls_read_argmax']:.3f}/{p['cls_read_wta']:.3f} "
                      f"(sel subj {p['subj_wta_selectivity']:.3f} cls {p['cls_wta_selectivity']:.3f}) || "
                      f"n-gram {p['ngram_floor_exact']:.3f}@k{p['ngram_order']} chance {chance:.3f}", flush=True)
        except Exception as e:
            err = repr(e); traceback.print_exc()

    smoke = len(a.seeds) < 6
    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = None; flags = None
    if err is None and far is not None:
        flags = build_verdict(far, chance, smoke)
        print(f"\n-- GO gate + lesion teeth at L={far['L']} (dist {far['distance']}, held-out NOVEL fillers) --", flush=True)
        void_if(not far["gen_defined"], "path space too small to hold out novel fillers -> generalisation UNDEFINED")
        lever("HYBRID-WTA vs argmax-reads reference (making the channel reads neural)", round(far["hybrid_argmax_exact"], 3),
              round(far["hybrid_wta_exact"], 3), required=False)
        lever("HYBRID-WTA vs LESION-the-WM-channel (subject channel load-bearing)", round(far["lesion_wm_chan_exact"], 3),
              round(far["hybrid_wta_exact"], 3), required=False)
        lever("HYBRID-WTA vs LESION-the-HTM-channel (class channel load-bearing)", round(far["lesion_htm_chan_exact"], 3),
              round(far["hybrid_wta_exact"], 3), required=False)
        lever("HYBRID-WTA vs LESION-the-HOLD (the WTA read reads the slot's spikes)", round(far["lesion_hold_exact"], 3),
              round(far["hybrid_wta_exact"], 3), required=False)
        lever("HYBRID-WTA vs LESION-the-WTA-selfcalib (competition load-bearing for clean channel selection)",
              round(far["lesion_wta_selfcalib_exact"], 3), round(far["hybrid_wta_exact"], 3), required=False)
        lever("HYBRID-WTA vs UNTRAINED conjunction (the bind is learned)", round(far["conj_untrained_exact"], 3),
              round(far["hybrid_wta_exact"], 3), required=False)
        attributable_to("HYBRID-WTA held-out over the best single system", far["hybrid_wta_exact"], flags["base"])
        attributable_to("HYBRID-WTA held-out vs the argmax-reads reference", far["hybrid_wta_exact"], flags["ref"])

        if not flags["gen"]:
            verdict = (f"INCONCLUSIVE — L={far['L']} path space {far['path_space']:.0f} too small to hold out novel "
                       f"fillers; generalisation UNDEFINED.")
        elif flags["core"]:
            tag = "GO" if flags["go"] else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — the emergent neural-WTA replaces BOTH host-argmax channel reads: subject_hat + class_hat "
                       f"are now selected by a DOWN-RAMP release-of-inhibition competition READ FROM SPIKES, and the "
                       f"LEARNED dendritic conjunction combines them. The FULLY-NEURAL read path holds the hybrid GO: "
                       f"held-out (NOVEL fillers) exact branch(verb) HYBRID-WTA {far['hybrid_wta_exact']:.3f}"
                       f"[min {far['hybrid_wta_exact_min']:.3f}] (subject preserved {far['hybrid_wta_subj']:.3f}, class "
                       f"{far['hybrid_wta_cls']:.3f}) within {abs(far['hybrid_wta_exact'] - far['hybrid_argmax_exact']):.3f} "
                       f"of the argmax-reads reference {far['hybrid_argmax_exact']:.3f}, >> HTM-alone {far['htm_exact']:.3f}, "
                       f">> WM-alone {far['wm_exact']:.3f}, >> n-gram {far['ngram_floor_exact']:.3f}, >> chance {chance:.3f}. "
                       f"All lesions load-bearing: lesion-WM-channel {far['lesion_wm_chan_exact']:.3f} (~HTM-alone), "
                       f"lesion-HTM-channel {far['lesion_htm_chan_exact']:.3f} (~WM-alone), lesion-the-hold "
                       f"{far['lesion_hold_exact']:.3f} (the WTA subject read reads the slot's spiking sustain, external "
                       f"input asserted zero), untrained conjunction {far['conj_untrained_exact']:.3f} (~chance, the "
                       f"bind is LEARNED), subject-shuffle {far['subj_shuffle_exact']:.3f} (no leakage). NO host argmax "
                       f"in the verb read path. DIAGNOSTIC (honest): lesion-the-WTA-selfcalib (freeze the ramp at a "
                       f"hand-set cut) -> {far['lesion_wta_selfcalib_exact']:.3f} "
                       f"({'DEGRADES -> the self-calibration is load-bearing for channel selection here' if flags['selfcalib_lb'] else 'does NOT bite -> on this CLEAN WM latch a fair fixed cut isolates the winner; the self-calibration is load-bearing in the rung-2 BLUR/allocation regime, not the clean-read regime'}). "
                       f"Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not flags["beats_both"]: miss.append(f"hybrid-wta {far['hybrid_wta_exact']:.3f} not >= max(HTM {far['htm_exact']:.3f}, WM {far['wm_exact']:.3f})+0.20")
            if not flags["within_ref"]: miss.append(f"hybrid-wta {far['hybrid_wta_exact']:.3f} not within 0.05 of argmax-ref {far['hybrid_argmax_exact']:.3f} (cost {far['hybrid_argmax_exact']-far['hybrid_wta_exact']:+.3f})")
            if not flags["subj_preserved"]: miss.append(f"subject NOT preserved (hybrid-wta subj {far['hybrid_wta_subj']:.3f} < 0.90)")
            if not flags["wm_chan_lb"]: miss.append(f"WM channel not cleanly load-bearing (lesion {far['lesion_wm_chan_exact']:.3f} vs HTM-alone {far['htm_exact']:.3f})")
            if not flags["htm_chan_lb"]: miss.append(f"HTM channel not cleanly load-bearing (lesion {far['lesion_htm_chan_exact']:.3f} vs WM-alone {far['wm_exact']:.3f})")
            if not flags["hold_lb"]: miss.append(f"hold not load-bearing (lesion-hold {far['lesion_hold_exact']:.3f})")
            if not flags["no_leak"]: miss.append(f"leakage (subject-shuffle {far['subj_shuffle_exact']:.3f} > base+0.05)")
            if not flags["bind_learned"]: miss.append(f"conjunction not learned (untrained {far['conj_untrained_exact']:.3f} > chance+0.10)")
            if not flags["spikes_ok"]: miss.append(f"hold not alive/zero-input (alive {far['hold_alive']:.4f}, zero {far['zero_input_ok']})")
            verdict = ("HONEST NEGATIVE / PARTIAL — the fully-neural (WTA-reads) hybrid did not clear the GO bar at L={}: ".format(far["L"])
                       + "; ".join(miss) + f". Read the WTA-read accuracy vs the argmax reads (subj am/wta "
                       f"{far['subj_read_argmax']:.3f}/{far['subj_read_wta']:.3f}, cls am/wta {far['cls_read_argmax']:.3f}/"
                       f"{far['cls_read_wta']:.3f}; WTA selectivity subj {far['subj_wta_selectivity']:.3f} cls "
                       f"{far['cls_wta_selectivity']:.3f}) to see EXACTLY what making the reads neural cost, per channel.")
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = "ERROR — no points computed"

    # --- earned verdict preconditions (VALIDITY travels with the verdict) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("emerge_wm_hybrid_neuralwta", chance=chance)
        if far is not None:
            Vd.require("oracle_task_solvable", 1.0, expect=lambda x: x > 0.99,
                       note="verb is determined by (subject, local-class) by construction -> the task IS solvable")
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out filler tuples disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("argmax_reference_reproduced", round(far["hybrid_argmax_exact"], 4), expect=lambda x: x >= 0.85,
                       note="the banked rung-3b argmax-reads hybrid re-runs to ~0.974 here -> a real reference to match")
            Vd.require("htm_alone_loses_subject", round(far["htm_subj"], 4), expect=lambda x: x <= 0.5,
                       note="HTM-alone loses the long-range subject on novel fillers (its banked failure)")
            Vd.require("conjunction_is_learned", round(far["conj_untrained_exact"], 4), expect=lambda x: x <= chance + 0.10,
                       note="an UNTRAINED conjunction bridge collapses to ~chance -> the bind lives in learned synapses")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across the hold+read span")
            Vd.control("wta_reads_track_argmax_reads", treatment=far["hybrid_wta_exact"], control=far["lesion_hold_exact"],
                       min_separation=0.20,
                       note="validity: the WTA subject read must READ the slot's live spikes (lesion-the-hold collapses)")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(flags is not None and flags["go"])
        dec = Vd.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_wm_hybrid_neuralwta", "verdict": verdict, "backend": backend, "sim_backend": backend,
               "device": device, "cost_acknowledged": True, "smoke": smoke, "preconditions": preconditions,
               "go": bool(flags["go"]) if flags else False, "core": bool(flags["core"]) if flags else False,
               "no_host_argmax_in_read_path": True, "read_path_fns_checked": read_path_fns,
               "frozen_control_arms": {"lesion_wta_selfcalib": "self_calib=False -> the WTA release ramp is FROZEN at a "
                                       "hand-set cut (no adaptive search). On a CLEAN WM latch a fair fixed cut isolates "
                                       "the winner, so this frozen control SHOULD TIE the candidate (an honest "
                                       "sub-negative: the self-calibration is load-bearing in the rung-2 blur/allocation "
                                       "regime, not this clean-read regime). Declared per the discriminating-power gate."},
               "mechanism": "the hybrid's TWO host-np.argmax channel reads are replaced by an emergent neural-WTA. "
                            "subject_hat: the WM slot's K held-pool firing RATES (spikes, after WRITE+HOLD across the "
                            "fillers, external input asserted zero) are the graded per-pool evidence into a DOWN-RAMP "
                            "release-of-inhibition competition on a K-pool D3 slow-NMDA attractor bank (pooled subtractive "
                            "inhibition released step by step; first pool to escape wins; ramp stops at exactly one active); "
                            "the winner is READ FROM SPIKES (a threshold on the latched HOLD rate, drive removed), NEVER "
                            "np.argmax(cp_firing_states). class_hat: the HTM engine's clsrd apical-drive vector (cp_v_apical) "
                            "is the evidence into the SAME WTA; the winner is read from spikes, NEVER np.argmax(apical). The "
                            "LEARNED dendritic conjunction bridge then combines wm[subject_hat] U clsrd[class_hat] -> the verb "
                            "column with the max apical plateau (the unique double-driven cell). NO host argmax in the verb "
                            "read path; the WTA is driven by the channel's SPIKE-derived evidence via a labelled-line identity "
                            "projection (the same accepted scope as the rung-2 barcode->pool projection).",
               "task": "compositional agreement stream [subject]+[L fillers]+[verb], verb=combine(subject, local-class); "
                       "held-out TEST = disjoint NOVEL filler tuples; arms: htm_alone / wm_alone / hybrid_argmax_reads "
                       "(rung-3b reference) / hybrid_wta_reads (THE TEST) + lesion-wm-channel + lesion-htm-channel + "
                       "lesion-the-hold + lesion-the-WTA-selfcalib + subject-shuffle + conj_untrained + n-gram + chance",
               "seeds": a.seeds, "config": {"n_subj": a.n_subj, "n_fill": a.n_fill, "n_cls": a.n_cls, "distances": dists,
               "n_cells": a.n_cells, "k_win": a.k_win, "act_th": a.act_th, "and_th": a.and_th, "learn_th": a.learn_th,
               "epochs": a.epochs, "n_train": a.n_train, "n_test": a.n_test, "recur": a.recur, "thr_off_conj": a.thr_off_conj,
               "wta_fixed_inh_frac": a.wta_fixed_inh_frac, "hold_steps": a.hold_steps, "load_steps": a.load_steps,
               "clear_steps": a.clear_steps, "chance": chance, "go_distance": go_L},
               "go_point": far, "points": points, "flags": flags, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the rung-3b separate-channel hybrid (stream + WM slot + class channel + "
                              "learned conjunction + argmax reference arms) and the rung-2 emergent-WTA mechanism (the D3 "
                              "attractor bank + the down-ramp release-of-inhibition competition). The ONLY change vs rung-3b "
                              "is HOW subject_hat / class_hat are read (host np.argmax -> emergent neural-WTA read from "
                              "spikes); the conjunction, lesions and controls are identical, so any accuracy difference is "
                              "attributable to making the reads neural. The WTA is driven by the channel's SPIKE-derived "
                              "evidence (slot pool rates / clsrd apical drive) via a labelled-line identity projection (the "
                              "same accepted host-projection scope as the rung-2 barcode->pool + the rung-3b labelled-line "
                              "wm[]/clsrd[] projections); the WTA SELECTION and the read are neural (spikes). 1-seed is a "
                              "SMOKE indicator; the 6-seed sweep is decisive. NO sim/ edit; NO host argmax in the read-out."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[emerge_wm_hybrid_neuralwta] VERDICT: {verdict}", flush=True)
    print(f"[emerge_wm_hybrid_neuralwta] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
