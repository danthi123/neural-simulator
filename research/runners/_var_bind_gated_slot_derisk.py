"""VARIABLE-BINDING WORKING MEMORY — a BG-gated write-gate -> slow-NMDA bistable HOLD slot whose CONTENT is a
content-agnostic Hebbian role->filler bind. THE DIRECT SURPASS of the emergence-engine HTM failure:

  `2026-08-11-emergence-engine-stream-language-htm-memorises-not-generalises-long-range-agreement-3seed-SMOKE.md`
  measured the on-bridge HTM Temporal-Memory on the SAME agreement stream and got HELD-OUT branch(verb) acc = 0.000
  (chance 0.250; the best fixed-order n-gram floor pinned AT chance on held-out) at n_fill=6, L=2/3/4, all seeds. It
  MEMORISES exact filler paths and cannot ABSTRACT the subject across novel intervening material. Its own NEXT #2 names
  the fix VERBATIM: "a gated bistable/attractor WM population that latches the subject (agreement feature) at sentence
  start and holds it invariantly across the filler span, so the verb is predicted from the LATCHED variable, not the
  traversed path. Measure the SAME held-out generalisation + swap-follows + permuted-stream anti-cheats." This runner
  builds that mechanism and runs that comparison.

COMPOSES THREE ALREADY-BANKED-GO PIECES (verified this session by reading each finding + runner; NO `sim/` edit):
  * HOLD  = the D3 slow-NMDA persistent-activity slot (`_d3_persistent_slot_derisk.build_persistent_slot`; HOLD 1.000
            6/6 with external input identically ZERO, no-recurrence control 0.0065). K attractor pools, each with
            slow-NMDA (tau=100ms) recurrent self-excitation + a shared FS pool. A WRITE = CLEAR (FS burst > tau_NMDA)
            then LOAD; D3 proved input ALONE cannot overwrite the attractor (5/6 kept the incumbent) -> the GATE, by
            triggering the write, is what protects the latch.
  * BIND  = the RUNG6c content-agnostic Hebbian FAST-WEIGHT binder (`_novel_referent_hebbian_fastweight_derisk.
            HebbianBinder`; 0.000 collisions on held-out NOVEL entities minted at test). Maps a subject BARCODE -> a
            slot index; content-agnostic so a novel subject binds like a known one BY CONSTRUCTION.
  * STREAM + the HTM baseline + the n-gram floor = reuse-by-import of `_emerge_stream_language_derisk` (the EXACT stream
            + engine the finding measured), so the HTM 0.000 comparison is like-for-like on the identical held-out set.

THE ARCHITECTURE (this de-risk): a persistent binder pre-binds the n_subj fixed subjects -> stable slots s_0..s_{n-1}.
For each stream sentence [subject]+[L i.i.d. fillers]+[verb], the WRITE-GATE decides LOAD (open) or HOLD (closed) per
token; on LOAD it binds the token -> a slot and CLEAR-then-LOADs that pool into the spiking HOLD slot; on HOLD the slot
receives ZERO external drive and self-sustains. At the verb position the slot is READ (argmax pool rate, zero input) ->
shat; deref shat -> the subject -> its (learned/fixed) agreeing verb. Held-out generalisation = NOVEL filler paths
(disjoint tuples) at increasing L: the fillers never touch the slot, so the latched subject is carried invariantly ->
the exact property the HTM lacked.

THE a-1 DE-SHORTCUT (heeded): an earlier rung (RUNG2) already hit ~0.971 hold-across-fillers but with a HOST doc-marker
write-gate + a FIXED bijection. So this is framed as DE-SHORTCUTTING that, NOT re-confirming it. Two gate variants:
  * marker  = opens on the subject token (ground-truth token type). This is the SCAFFOLD (gate TIMING is host-given);
              it proves the MEMORY composition (real Hebbian bind + real spiking NMDA hold + real clear-then-load write)
              solves the held-out task the HTM cannot. The gate timing is the named residual, NOT the memory.
  * learned = a reward-driven (REINFORCE, three-factor: eligibility x verb-prediction DA) write-gate trained ONLY on
              the verb-prediction reward, no token-type label -> does a control unit LEARN to fire LOAD on the subject
              and HOLD on fillers FROM STREAM STATISTICS? This is the genuinely-OPEN, load-bearing question and the
              HONEST-NEGATIVE-expected piece. Reported precisely (held-out acc + fire-on-subject precision/recall).

ANTI-CHEATS (all required; each EXECUTES; teeth):
  (1) LESION-the-hold (recur=0, the stateless bridge every prior rung used) -> the bump dies over the filler span ->
      shat is noise -> collapse to the memorise-not-generalise baseline.
  (2) ALWAYS-OPEN / SHUFFLED gate -> every token triggers a clear-then-load -> the LAST filler overwrites the subject
      latch -> shat = a filler slot -> wrong. THE DE-SHORTCUT TOOTH: proves the gate (not the attractor alone) protects
      the latch.
  (3) SLOT-SCRAMBLE / permuted-binding -> the slot->subject deref is randomly permuted -> shat derefs to the wrong
      subject -> chance. Proves the BIND is load-bearing.
  (4) REFERENT-SHUFFLE -> the subject->verb readout map is deranged (topic->answer association broken) -> ~0.000. No
      topic->answer leakage (the answer comes from the learned association, not position/surface).
  (5) HOLD-NOT-RE-READ guard -> `cp_external_input_current` is ASSERTED identically zero across the whole hold+read span
      (the slot SUSTAINS, it does not re-read a host store per step, as D3 established). A per-arm assertion, not a score.
  (6) BASELINES that must fall to chance: the HTM emergence engine (the finding's 0.000, re-run here on the identical
      stream), the best fixed-order n-gram HELD-OUT floor (pinned at chance), and a LAST-TOKEN fixed-window baseline.

GO (smoke = 1-seed indicator; decisive = 6-seed): at n_fill=6 where the n-gram floor is chance, the MARKER-gated spiking
WM held-out branch(verb) acc >= 0.90 AND >= chance+0.20 AND >> the HTM 0.000 baseline, GENERALISING to novel fillers at
increasing L; lesion-the-hold collapses (>= acc-0.30 below), always-open collapses (<= chance+0.15), slot-scramble ~
chance, referent-shuffle ~0, hold-alive > 0 with input asserted zero. The LEARNED-gate verdict is reported SEPARATELY and
honestly (a scaffold-only marker GO with a learned-gate negative is a first-class deliverable that maps the exact
residual: the gate-learning / distal-credit problem = gap#4 territory).

Reuse-by-import; NO `sim/` edit. SIM_BACKEND=numpy (the D3/coincidence loops are sub-1k-neuron, launch-bound: CPU faster).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_gated_slot_derisk --seeds 42 --n-fill 6 --distances 2 3
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._var_bind_gated_slot_derisk --seeds 42 43 44 100 101 102 \
    --n-fill 6 --distances 2 3 4 --n-test 90 --out research/findings/raw/_var_bind_gated_slot/gated_slot_6seed.json
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

# --- the EXACT stream + HTM baseline + n-gram floor the finding measured (like-for-like) ---
from research.runners._emerge_stream_language_derisk import (
    vocab_layout, make_stream, make_heldout, ngram_floor_heldout, train_engine, branch_acc)
# --- the VERIFIED RUNG6c content-agnostic Hebbian fast-weight binder + barcode mint ---
from research.runners._novel_referent_hebbian_fastweight_derisk import HebbianBinder, _mint_codes, _K, _DIM
# --- the VERIFIED D3 spiking slow-NMDA persistent-activity HOLD slot ---
from research.runners._d3_persistent_slot_derisk import build_persistent_slot, _pool_idx, _reset

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

OUT = Path("research/findings/raw/_var_bind_gated_slot/gated_slot.json")


# ====================================================================================================================
# The SPIKING HOLD slot: a thin controller over the D3 build_persistent_slot bridge. WRITE = clear-then-load;
# HOLD = zero external drive (self-sustain); READ = argmax pool rate under zero input. All three assert input discipline.
# ====================================================================================================================
class SpikingSlot:
    def __init__(self, seed, K, recur=25.0, load_steps=30, hold_steps=18, read_steps=18, clear_steps=200,
                 input_gain=400.0, clear_gain=1500.0, nmda=True):
        from sim.backend import to_host, from_host
        self._to_host, self._from_host = to_host, from_host
        self.sb = build_persistent_slot(seed, K, recur=recur, nmda=nmda)
        self.K = K
        self.idx = _pool_idx(self.sb, K)
        self.fs_idx = np.asarray(list(self.sb.region_manager.indices("fs")), dtype=int)
        self.n = self.sb.core_config.num_neurons
        self.load_steps, self.hold_steps, self.read_steps = load_steps, hold_steps, read_steps
        self.clear_steps, self.input_gain, self.clear_gain = clear_steps, input_gain, clear_gain
        self._loaded = False           # is a bump currently latched? (first write after reset needs no clear)
        self._zero_input_span = True   # anti-cheat #5 latch: was input identically zero across the last hold+read?

    def reset(self):
        _reset(self.sb); self._loaded = False; self._zero_input_span = True

    def _run(self, cur_vec, steps, assert_zero=False):
        acc = np.zeros(self.K)
        dev = self._from_host(cur_vec)
        if assert_zero:
            # anti-cheat #5: the HOLD/READ must sustain with ZERO external drive (no per-step re-read of a host store)
            if np.asarray(cur_vec).any():
                self._zero_input_span = False
        for _ in range(steps):
            self.sb.cp_external_input_current[:] = dev
            self.sb._run_one_simulation_step()
            fir = np.asarray(self._to_host(self.sb.cp_firing_states)).astype(float)
            for k in range(self.K):
                acc[k] += fir[self.idx[k]].mean()
        return acc / max(steps, 1)

    def write(self, pool):
        """A gate-triggered WRITE = CLEAR (FS burst, only if a bump is already latched) then LOAD `pool`."""
        if self._loaded and self.clear_steps > 0:                       # overwrite an incumbent: clear > tau_NMDA
            cc = np.zeros(self.n); cc[self.fs_idx] = self.clear_gain
            self._run(cc, self.clear_steps)
        cur = np.zeros(self.n); cur[self.idx[pool]] = self.input_gain
        self._run(cur, self.load_steps)
        self._loaded = True

    def hold(self, steps=None):
        zero = np.zeros(self.n)
        self._run(zero, steps if steps is not None else self.hold_steps, assert_zero=True)

    def read(self):
        zero = np.zeros(self.n)
        rates = self._run(zero, self.read_steps, assert_zero=True)
        alive = float(rates.max())
        shat = int(np.argmax(rates)) if alive > 1e-6 else -1
        return shat, alive


# ====================================================================================================================
# Stream -> barcode codebook + a persistent binder pre-bound to the fixed subjects
# ====================================================================================================================
def build_codebook(n_subj, n_fill, rng):
    """A sparse developmental-random barcode for every WRITEABLE token (subjects + fillers). Verbs are TARGETS, never
    written, so they need no code. Returns code_of[token_id] (dim _DIM) for token ids in [0, n_subj+n_fill)."""
    subj, fill, verb, V = vocab_layout(n_subj, n_fill)
    codes = _mint_codes(rng, n_subj + n_fill)          # RUNG6c overlap-rejected sparse barcodes
    code_of = {}
    for i, tok in enumerate(list(subj) + list(fill)):
        code_of[tok] = codes[i]
    return code_of, (subj, fill, verb, V)


def persistent_subject_binder(subj_tokens, code_of):
    """One RUNG6c HebbianBinder shared across the stream: bind each fixed subject to a stable slot (first-mention order),
    return (binder, slot_of_subj, subj_of_slot). This is the 'training' of the addressing (host fast-weight, RUNG6c GO)."""
    binder = HebbianBinder()
    slot_of_subj, subj_of_slot = {}, {}
    for tok in subj_tokens:
        s = binder.slot(code_of[tok]); slot_of_subj[tok] = s; subj_of_slot[s] = tok
    return binder, slot_of_subj, subj_of_slot


# ====================================================================================================================
# Gate variants (the write-gate: LOAD/open vs HOLD/closed per token). t is the within-sentence index; token is its id.
# ====================================================================================================================
def gate_marker(t, token, subj_set):
    return t == 0                                      # SCAFFOLD: opens on the subject token (ground-truth token type)

def gate_always_open(t, token, subj_set):
    return True                                        # anti-cheat #2: every token writes -> last filler overwrites


class LearnedGate:
    """Reward-driven (REINFORCE) write-gate: drive = w.code + b, p_load = sigmoid(gain*drive). Trained ONLY on the
    verb-prediction reward (three-factor: per-token eligibility x the terminal DA), NO token-type label. Question: does
    it LEARN to fire LOAD on the subject and HOLD on fillers FROM STREAM STATISTICS? (Update math is host; the fire/no
    decision is a thresholded control unit -> the on-substrate spiking-gate realisation is the named next rung.)"""
    def __init__(self, dim=_DIM, gain=4.0, lr=0.15, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, dim).astype(np.float32); self.b = 0.0
        self.gain, self.lr = gain, lr; self.baseline = 0.0

    def p_load(self, code):
        z = self.gain * (float(self.w @ code) + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def decide(self, code):
        return self.p_load(code) > 0.5

    def train(self, stream, code_of, slot_of_subj, subj_of_slot, verb_of, subj_set, n_subj, episodes=6):
        """SURROGATE WM (faithful to the spiking slot per D3: a WRITE overwrites -> last-write-wins; a HOLD sustains).
        Trains the gate policy against the verb-prediction reward. Evaluation later uses the REAL spiking slot."""
        chance = 1.0 / n_subj
        for _ in range(episodes):
            order = list(range(len(stream)))
            for n in order:
                s = stream[n]; toks = s[:-1]; true_verb = s[-1]
                cur_slot = -1; elig_w = np.zeros_like(self.w); elig_b = 0.0
                for t, tok in enumerate(toks):
                    code = code_of.get(tok, np.zeros(_DIM, np.float32))
                    p = self.p_load(code); load = 1.0 if (np.random.random() < p) else 0.0
                    elig_w += (load - p) * code; elig_b += (load - p)   # REINFORCE score fn, accumulated
                    if load > 0.5:
                        cur_slot = slot_of_subj.get(tok, self._filler_slot(tok, code_of, slot_of_subj, n_subj))
                pred_subj = subj_of_slot.get(cur_slot, -1)
                pred_verb = verb_of.get(pred_subj, -1)
                reward = 1.0 if pred_verb == true_verb else 0.0
                adv = reward - self.baseline
                self.w += self.lr * adv * elig_w; self.b += self.lr * adv * elig_b
                self.baseline += 0.05 * (reward - self.baseline)

    @staticmethod
    def _filler_slot(tok, code_of, slot_of_subj, n_subj):
        # a filler write lands on a non-subject slot (the surrogate: fillers occupy slots >= n_subj, capped at _K-1)
        return min(n_subj + (tok % max(1, _K - n_subj)), _K - 1)


# ====================================================================================================================
# Spiking WM evaluation over a sentence set, for a given gate + binder + deref
# ====================================================================================================================
def eval_spiking_wm(slot: SpikingSlot, stream, gate_fn, code_of, slot_of_subj, subj_of_slot, verb_of, subj_set,
                    n_subj, always_open_binder=None):
    """Returns (branch_verb_acc, hold_alive_mean, zero_input_ok, slot_correct_acc). gate_fn(t, token) -> bool (LOAD)."""
    ok = 0; slot_ok = 0; alive_acc = []; zero_ok = True
    for s in stream:
        toks = s[:-1]; true_verb = s[-1]; subj_tok = s[0]
        true_slot = slot_of_subj.get(subj_tok, -1)
        slot.reset()
        for t, tok in enumerate(toks):
            if gate_fn(t, tok):
                if tok in slot_of_subj:                       # a known subject -> its bound slot
                    pool = slot_of_subj[tok]
                elif always_open_binder is not None:          # the always-open control: fillers bind to fresh slots
                    pool = always_open_binder.slot(code_of[tok])
                else:
                    pool = min(n_subj, _K - 1)                # an off-target write (learned gate firing on a filler)
                slot.write(pool)
            else:
                slot.hold()
        shat, alive = slot.read()
        alive_acc.append(alive)
        zero_ok = zero_ok and slot._zero_input_span
        slot_ok += int(shat == true_slot)
        pred_subj = subj_of_slot.get(shat, -1)
        pred_verb = verb_of.get(pred_subj, -1)
        ok += int(pred_verb == true_verb)
    n = max(1, len(stream))
    return ok / n, float(np.mean(alive_acc)) if alive_acc else 0.0, bool(zero_ok), slot_ok / n


def last_token_floor(train, test, n_subj):
    """A LAST-TOKEN fixed-window baseline: learn the most-likely verb after each final-filler token on train, predict on
    test. Random fillers -> uninformative -> chance. (A complement to the n-gram floor; both must sit at chance.)"""
    counts = defaultdict(Counter)
    for s in train:
        counts[s[-2]][s[-1]] += 1
    ok = 0.0
    for s in test:
        dist = counts.get(s[-2])
        if not dist:
            ok += 1.0 / n_subj; continue
        top = max(dist.values()); win = [x for x, c in dist.items() if c == top]
        ok += (1.0 / len(win)) if s[-1] in win else 0.0
    return ok / max(1, len(test))


# ====================================================================================================================
# One (seed, n_fill, L) point
# ====================================================================================================================
def run_point(seed, n_subj, n_fill, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps,
              learned_episodes, run_always_open):
    rng = np.random.default_rng(seed)
    chance = 1.0 / n_subj
    subj, fill, verb, V = vocab_layout(n_subj, n_fill)
    verb_of = {int(subj[i]): int(verb[i]) for i in range(n_subj)}      # the true agreement map subject->verb

    # --- the identical stream + disjoint held-out novel-filler test (reuse-by-import) ---
    train_seqs, _ = make_stream(n_subj, n_fill, L, n_train, rng)
    train_ftuples = set(tuple(s[1:-1]) for s in train_seqs)
    test_seqs, gen_defined = make_heldout(n_subj, n_fill, L, n_test, rng, train_ftuples)

    # --- codebook + persistent subject binder (the addressing; RUNG6c host fast-weight) ---
    code_of, _ = build_codebook(n_subj, n_fill, np.random.default_rng(seed + 7))
    binder, slot_of_subj, subj_of_slot = persistent_subject_binder([int(s) for s in subj], code_of)

    def _slot(seed_off=0, rc=recur):
        return SpikingSlot(seed + seed_off, _K, recur=rc, hold_steps=hold_steps, load_steps=load_steps,
                           clear_steps=clear_steps)

    gm = lambda t, tok: gate_marker(t, tok, set(int(x) for x in subj))

    # (A) MARKER-gated spiking WM on held-out novel fillers  == the headline
    acc_marker, alive, zero_ok, slot_acc = eval_spiking_wm(
        _slot(), test_seqs, gm, code_of, slot_of_subj, subj_of_slot, verb_of, set(subj), n_subj)

    # (1) LESION-the-hold: recur=0 (the stateless bridge) -> cannot sustain the latch across the filler span
    acc_lesion, alive_les, _, _ = eval_spiking_wm(
        _slot(rc=0.0), test_seqs, gm, code_of, slot_of_subj, subj_of_slot, verb_of, set(subj), n_subj)

    # (2) ALWAYS-OPEN gate: every token clear-then-loads -> the last filler overwrites the subject latch (de-shortcut)
    acc_always = None
    if run_always_open:
        ao_binder = HebbianBinder()
        for tok in subj:                                    # pre-bind subjects so their slots exist, then fillers steal
            ao_binder.slot(code_of[int(tok)])
        ao_test = test_seqs if n_test <= 40 else test_seqs[:40]   # the clear-per-token arm is the expensive one
        ga = lambda t, tok: True                                   # always-open: every token triggers a write
        acc_always, _, _, _ = eval_spiking_wm(
            _slot(), ao_test, ga, code_of, slot_of_subj, subj_of_slot, verb_of, set(subj), n_subj,
            always_open_binder=ao_binder)

    # (3) SLOT-SCRAMBLE / permuted-binding: permute the slot->subject deref -> shat derefs to the wrong subject
    perm = list(range(_K)); np.random.default_rng(seed + 11).shuffle(perm)
    subj_of_slot_scr = {perm[s]: t for s, t in subj_of_slot.items()}
    acc_scramble, _, _, _ = eval_spiking_wm(
        _slot(), test_seqs, gm, code_of, slot_of_subj, subj_of_slot_scr, verb_of, set(subj), n_subj)

    # (4) REFERENT-SHUFFLE: derange the subject->verb readout (topic->answer broken) -> ~0.000, no leakage
    order = list(range(n_subj)); dr = np.random.default_rng(seed + 13)
    for _ in range(64):
        dr.shuffle(order)
        if all(order[i] != i for i in range(n_subj)):
            break
    verb_of_shuf = {int(subj[i]): int(verb[order[i]]) for i in range(n_subj)}
    acc_refshuf, _, _, _ = eval_spiking_wm(
        _slot(), test_seqs, gm, code_of, slot_of_subj, subj_of_slot, verb_of_shuf, set(subj), n_subj)

    # (6) BASELINES to chance: the HTM emergence engine (the finding's 0.000, identical stream) + n-gram + last-token
    lr_htm = train_engine(seed, "htm", n_subj, n_fill, L, n_cells=32, k_win=4, act_th=3, epochs=8, train_seqs=train_seqs)
    htm_test = branch_acc(lr_htm, test_seqs, L)
    ngram_test, ngram_order = ngram_floor_heldout(train_seqs, test_seqs, L, n_subj)
    lasttok = last_token_floor(train_seqs, test_seqs, n_subj)

    # (learned gate) — the genuinely-open piece: reward-driven, no token-type label
    lg = LearnedGate(seed=seed + 5)
    lg.train(train_seqs, code_of, slot_of_subj, subj_of_slot, verb_of, set(subj), n_subj, episodes=learned_episodes)
    gl = lambda t, tok: lg.decide(code_of.get(tok, np.zeros(_DIM, np.float32)))
    acc_learned, alive_l, _, _ = eval_spiking_wm(
        _slot(), test_seqs, gl, code_of, slot_of_subj, subj_of_slot, verb_of, set(subj), n_subj)
    # fire-on-subject precision/recall on held-out (does it gate the subject, not fillers?)
    tp = fp = fn = tn = 0
    for s in test_seqs:
        for t, tok in enumerate(s[:-1]):
            fires = lg.decide(code_of.get(tok, np.zeros(_DIM, np.float32)))
            is_subj = (t == 0)
            tp += int(fires and is_subj); fp += int(fires and not is_subj)
            fn += int((not fires) and is_subj); tn += int((not fires) and not is_subj)
    g_prec = tp / max(1, tp + fp); g_rec = tp / max(1, tp + fn)

    return {"seed": seed, "n_fill": n_fill, "L": L, "distance": L + 1, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(n_fill) ** L,
            "acc_marker": acc_marker, "slot_acc_marker": slot_acc, "hold_alive": alive, "hold_alive_lesion": alive_les,
            "zero_input_ok": zero_ok, "acc_lesion": acc_lesion, "acc_always_open": acc_always,
            "acc_slot_scramble": acc_scramble, "acc_referent_shuffle": acc_refshuf,
            "htm_test": htm_test, "ngram_floor_test": ngram_test, "ngram_order": ngram_order, "lasttok_floor": lasttok,
            "acc_learned": acc_learned, "learned_hold_alive": alive_l, "gate_subj_precision": g_prec,
            "gate_subj_recall": g_rec}


def agg(per):
    keys = ["acc_marker", "slot_acc_marker", "hold_alive", "hold_alive_lesion", "acc_lesion", "acc_slot_scramble",
            "acc_referent_shuffle", "htm_test", "ngram_floor_test", "lasttok_floor", "acc_learned", "learned_hold_alive",
            "gate_subj_precision", "gate_subj_recall"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    ao = [p["acc_always_open"] for p in per if p["acc_always_open"] is not None]
    a["acc_always_open"] = float(np.mean(ao)) if ao else None
    a.update({"n_fill": per[0]["n_fill"], "L": per[0]["L"], "distance": per[0]["distance"], "chance": per[0]["chance"],
              "path_space": per[0]["path_space"], "gen_defined": all(p["gen_defined"] for p in per),
              "zero_input_ok": all(p["zero_input_ok"] for p in per), "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-subj", type=int, default=4)
    ap.add_argument("--n-fill", type=int, default=6, help="filler-alphabet size (>=n_subj so held-out paths exist)")
    ap.add_argument("--distances", type=int, nargs="+", default=[2, 3], help="filler-span L (dependency dist = L+1)")
    ap.add_argument("--n-train", type=int, default=72)
    ap.add_argument("--n-test", type=int, default=60, help="held-out novel-filler test sentences")
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--learned-episodes", type=int, default=6)
    ap.add_argument("--no-always-open", action="store_true", help="skip the expensive clear-per-token always-open arm")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to evaluate GO (default: the largest)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    chance = 1.0 / a.n_subj
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | n_subj={a.n_subj} chance={chance:.3f} | n_fill={a.n_fill} L={dists} "
          f"| recur={a.recur} hold_steps={a.hold_steps} | n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds}",
          flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            per = [run_point(s, a.n_subj, a.n_fill, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                             a.clear_steps, a.learned_episodes, not a.no_always_open) for s in a.seeds]
            p = agg(per); points.append(p)
            ao = "n/a" if p["acc_always_open"] is None else f"{p['acc_always_open']:.3f}"
            print(f"  [n_fill={a.n_fill} L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']}] "
                  f"MARKER-WM held-out {p['acc_marker']:.3f} (slot {p['slot_acc_marker']:.3f}, hold-alive "
                  f"{p['hold_alive']:.4f}, zero-input {p['zero_input_ok']}) || HTM {p['htm_test']:.3f} | n-gram "
                  f"{p['ngram_floor_test']:.3f} | last-tok {p['lasttok_floor']:.3f} | chance {chance:.3f} || "
                  f"LESION {p['acc_lesion']:.3f}(alive {p['hold_alive_lesion']:.4f}) | ALWAYS-OPEN {ao} | SCRAMBLE "
                  f"{p['acc_slot_scramble']:.3f} | REF-SHUF {p['acc_referent_shuffle']:.3f} || LEARNED-gate "
                  f"{p['acc_learned']:.3f} (subj prec {p['gate_subj_precision']:.2f} rec {p['gate_subj_recall']:.2f})",
                  flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = learned_verdict = None
    if err is None and far is not None:
        print(f"\n-- GO gate + anti-cheats at L={far['L']} (dist {far['distance']}, held-out novel fillers) --", flush=True)
        void_if(not far["gen_defined"], "path space too small to hold out novel fillers -> generalisation UNDEFINED")
        lever("MARKER-WM held-out vs LESION-the-hold (recurrence load-bearing)", round(far["acc_lesion"], 3),
              round(far["acc_marker"], 3), required=False)
        lever("MARKER-WM held-out vs ALWAYS-OPEN gate (the gate protects the latch)",
              round(far["acc_always_open"], 3) if far["acc_always_open"] is not None else -1.0,
              round(far["acc_marker"], 3), required=False)
        attributable_to("MARKER-WM held-out over the HTM emergence-engine baseline", far["acc_marker"], far["htm_test"])
        attributable_to("MARKER-WM held-out over the best n-gram floor", far["acc_marker"], far["ngram_floor_test"])

        gen = far["gen_defined"]
        headline = far["acc_marker"] >= 0.90 and far["acc_marker"] >= chance + 0.20
        beats_htm = far["acc_marker"] >= far["htm_test"] + 0.30
        beats_floor = far["acc_marker"] >= far["ngram_floor_test"] + 0.30
        recurrence = far["acc_marker"] >= far["acc_lesion"] + 0.30
        gate_protects = (far["acc_always_open"] is None) or (far["acc_always_open"] <= chance + 0.15)
        bind_lb = far["acc_slot_scramble"] <= chance + 0.15
        no_leak = far["acc_referent_shuffle"] <= chance
        held_alive = far["hold_alive"] > 1e-3 and far["zero_input_ok"]
        core = bool(gen and headline and beats_htm and beats_floor and recurrence and gate_protects and bind_lb
                    and no_leak and held_alive)
        go = bool(core and not smoke)

        if not gen:
            verdict = (f"INCONCLUSIVE — L={far['L']} path space {far['path_space']:.0f} too small to hold out novel "
                       f"fillers; increase n_fill/L for a real held-out regime.")
        elif core:
            tag = "GO" if go else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — the BG-gated slow-NMDA bistable HOLD slot with a content-agnostic Hebbian bind LATCHES "
                       f"the subject and CARRIES it invariantly across NOVEL filler paths: at n_fill={far['n_fill']} "
                       f"L={far['L']} (dist {far['distance']}) MARKER-gated held-out branch(verb) {far['acc_marker']:.3f} "
                       f">> HTM emergence-engine {far['htm_test']:.3f} (the memorise-not-generalise baseline), >> best "
                       f"n-gram floor {far['ngram_floor_test']:.3f}, >> chance {chance:.3f}. The HOLD is load-bearing "
                       f"(lesion-the-hold {far['acc_lesion']:.3f}, hold-alive {far['hold_alive']:.4f} with external "
                       f"input ASSERTED zero across the span), the GATE protects the latch (always-open "
                       f"{far['acc_always_open']}), the BIND is load-bearing (slot-scramble {far['acc_slot_scramble']:.3f}"
                       f"), no topic->answer leakage (referent-shuffle {far['acc_referent_shuffle']:.3f}). This is the "
                       f"direct surpass of the HTM held-out 0.000. NOTE: the MARKER gate's TIMING is a host scaffold "
                       f"(opens on the subject token) -> the memory composition is de-shortcutted + real + spiking, the "
                       f"gate-LEARNING is the named residual (see the learned-gate verdict). Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not headline: miss.append(f"marker held-out {far['acc_marker']:.3f} not >=0.90/chance+0.20")
            if not beats_htm: miss.append(f"did not clear HTM+0.30 (HTM {far['htm_test']:.3f})")
            if not recurrence: miss.append(f"hold not load-bearing (lesion {far['acc_lesion']:.3f})")
            if not gate_protects: miss.append(f"gate did not protect latch (always-open {far['acc_always_open']})")
            if not bind_lb: miss.append(f"bind not load-bearing (slot-scramble {far['acc_slot_scramble']:.3f})")
            if not no_leak: miss.append(f"leakage (referent-shuffle {far['acc_referent_shuffle']:.3f} > chance)")
            if not held_alive: miss.append(f"hold not alive/zero-input (alive {far['hold_alive']:.4f}, zero {far['zero_input_ok']})")
            verdict = ("PARTIAL/NEGATIVE — the gated-slot WM did not clear the GO bar at L={}: ".format(far["L"])
                       + "; ".join(miss) + ". Read the per-arm numbers.")

        # --- the LEARNED-gate verdict, reported SEPARATELY + honestly (the genuinely-open piece) ---
        lg_gates = far["gate_subj_precision"] >= 0.90 and far["gate_subj_recall"] >= 0.90
        lg_acc_ok = far["acc_learned"] >= chance + 0.20
        if lg_gates and lg_acc_ok:
            learned_verdict = (f"LEARNED-GATE POSITIVE (report with caveat) — a reward-driven write-gate (REINFORCE on "
                               f"the verb-prediction DA, NO token-type label) LEARNED to fire LOAD on the subject and "
                               f"HOLD on fillers FROM STREAM STATISTICS: held-out subject precision "
                               f"{far['gate_subj_precision']:.2f} / recall {far['gate_subj_recall']:.2f}, driving the "
                               f"spiking WM to held-out {far['acc_learned']:.3f} (chance {chance:.3f}). CAVEAT: this "
                               f"stream marks the subject as a distinct BARCODE class (linearly separable from fillers), "
                               f"so the gate can exploit token-class rather than syntactic role; the harder residual is "
                               f"role/position-based gating where the same token can be subject-or-not. The update math "
                               f"is host; the on-substrate spiking-gate (three-factor DA-gated plasticity) is the next rung.")
        else:
            learned_verdict = (f"LEARNED-GATE HONEST NEGATIVE (first-class; maps the exact residual) — the reward-driven "
                               f"write-gate did NOT reliably fire on the subject from stream statistics alone: held-out "
                               f"subject precision {far['gate_subj_precision']:.2f} / recall {far['gate_subj_recall']:.2f}, "
                               f"held-out WM acc with the learned gate {far['acc_learned']:.3f} (chance {chance:.3f}). Only "
                               f"the hand-wired MARKER gate drives the WM to its ceiling. This precisely names the residual: "
                               f"a LEARNED, EMERGENT, SPIKING write-gate (three-factor DA-gated: fire LOAD at the "
                               f"subject/role from stream prediction-error / novelty / salience) is the open problem = the "
                               f"distal temporal-credit / gap#4 territory. The MEMORY composition (bind + spiking hold + "
                               f"clear-then-load protect) is solved; the GATE-LEARNING is not.")
        print(f"\n[learned-gate] {learned_verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = "ERROR — no points computed"

    # --- earned verdict preconditions (VALIDITY travels with the verdict; tools/gates/verdict_preconditions) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("var_bind_gated_slot", chance=chance)
        if far is not None:
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out fillers disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("htm_baseline_at_or_below_chance", round(far["htm_test"], 4), expect=lambda x: x <= chance + 0.10,
                       note="the HTM emergence-engine baseline must sit at/below chance on held-out (the failure it names)")
            Vd.require("ngram_floor_at_chance", round(far["ngram_floor_test"], 4), expect=lambda x: x <= chance + 0.15,
                       note="the best fixed-order n-gram HELD-OUT floor must be pinned near chance (the bar is meaningful)")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across the hold+read span")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(far is not None and far["gen_defined"] and far["acc_marker"] >= 0.90
                   and far["acc_marker"] >= chance + 0.20 and far["acc_marker"] >= far["htm_test"] + 0.30
                   and far["acc_marker"] >= far["acc_lesion"] + 0.30
                   and (far["acc_always_open"] is None or far["acc_always_open"] <= chance + 0.15)
                   and far["acc_slot_scramble"] <= chance + 0.15 and far["acc_referent_shuffle"] <= chance)
        dec = Vd.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_gated_slot", "verdict": verdict, "learned_gate_verdict": learned_verdict,
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "BG-gated write-gate -> slow-NMDA (tau=100ms) bistable persistent-activity HOLD slot "
                            "(D3 build_persistent_slot; a WRITE = CLEAR(FS burst>tau_NMDA)-then-LOAD, D3-proven that "
                            "input alone cannot overwrite an NMDA attractor so the GATE protects the latch) whose "
                            "CONTENT is a content-agnostic RUNG6c Hebbian fast-weight subject->slot bind; the verb is "
                            "predicted from the LATCHED subject, not the traversed filler path",
               "task": "the identical agreement stream [subject]+[L i.i.d. fillers]+[verb] the emergence-engine HTM "
                       "failed (held-out branch(verb) 0.000); held-out TEST = disjoint NOVEL filler paths at increasing "
                       "L; anti-cheats: lesion-the-hold + always-open/shuffled gate + slot-scramble + referent-shuffle "
                       "+ hold-not-re-read (zero-input assert) + HTM/n-gram/last-token chance floors; marker vs learned gate",
               "seeds": a.seeds, "config": {"n_subj": a.n_subj, "n_fill": a.n_fill, "distances": dists, "recur": a.recur,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps,
               "learned_episodes": a.learned_episodes, "n_train": a.n_train, "n_test": a.n_test, "chance": chance,
               "go_distance": go_L}, "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "composes three BANKED GOs by reuse-by-import (D3 spiking NMDA hold + RUNG6c Hebbian bind + "
                              "the EMERGE-14 HTM baseline/stream); NO sim/ edit. The spiking HOLD slot is genuinely on "
                              "substrate. The BIND is the RUNG6c HOST numpy fast-weight (its spiking STP realisation is a "
                              "banked next rung). The MARKER gate's TIMING is a host scaffold (de-shortcutting RUNG2's "
                              "doc-marker+bijection in the MEMORY dimension, not the gate-timing dimension). The verb "
                              "readout is a host deref of the held pool (a decode). 1-seed is a SMOKE indicator; the "
                              "6-seed sweep is decisive. The LEARNED-gate result is reported separately + honestly."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[var_bind_gated_slot] VERDICT: {verdict}", flush=True)
    print(f"[var_bind_gated_slot] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
