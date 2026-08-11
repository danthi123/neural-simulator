"""ROLE-BASED (syntactic, not token-class) WRITE-GATE DE-RISK — the precisely-named residual of the variable-binding
working-memory GO (`2026-08-11-variable-binding-working-memory-gated-slot-surpasses-HTM-heldout-1.000-vs-0.000-6seed-GO`).

THE RESIDUAL (verbatim from that finding): the reward-driven write-gate LEARNED to fire LOAD on the subject with held-out
precision/recall 1.00 — BUT on a stream where the subject is a BARCODE-SEPARABLE CLASS (linearly separable from fillers),
so it learned a token-CLASS boundary, NOT syntactic ROLE. In real language the SAME token is subject-or-not by
POSITION/SYNTAX. That role-based gate is the genuinely-open next problem.

THE HARDER STREAM (this de-risk): the subject is NOT a distinct token class. ONE shared noun pool of N nouns; EVERY noun
appears as BOTH the subject (position 0, the agreement-controlling slot) AND as an intervening DISTRACTOR (positions
1..L), drawn i.i.d. from the SAME pool. Each noun carries an agreement FEATURE feat(noun) in {0..F-1} (its number/gender
class — the agreement lexicon). The verb agrees with the SUBJECT's feature: verb = verb_tok[feat(subject)]. So which token
is the subject is fixed by POSITION, not identity, and the distractors' features are LURES (uninformative about the verb).
  * chance = 1/F (predicting the verb == predicting the subject's feature).
  * The best fixed-order n-gram HELD-OUT floor is pinned at chance (the last-k tokens are random distractors; order>=L+1
    contexts are unique per sentence -> unseen on held-out -> back-off to chance), EXACTLY as the barcode stream.
  * Held-out TEST = disjoint NOVEL distractor tuples -> the fillers never touch the slot, the latched subject is carried
    invariantly, tests that the gate IGNORES novel distractors.

THE QUESTION (the load-bearing, honest-negative-expected piece): can a reward-driven write-gate learn to LOAD the
ROLE-defined subject (from the verb-prediction reward ALONE, NO role/position label) using POSITION/STRUCTURE — when
token IDENTITY no longer reveals the role? We test three gate drivers against the same spiking WM (the banked D3 slow-NMDA
bistable HOLD slot; write = clear-then-load; the feature is the write content = the agreement lexicon, a host scaffold
exactly as verb_of was — the GATE is the object of study, not the bind):
  * marker-ROLE (SCAFFOLD): fires t==0 (the true role position). Proves the MEMORY composition solves the ROLE-defined
    task when the timing is given -> the memory is not the residual (parallel to the finding's marker gate).
  * identity-LEARNED (the EXISTING mechanism, code-only): the finding's LearnedGate, p_load(code) — sees ONLY the token
    code, NO position. On the shared pool it CANNOT gate the same token differently by position BY CONSTRUCTION -> expected
    to FAIL the role task AND the token-identity control. This reproduces the residual precisely.
  * role-LEARNED-recurrent (the CANDIDATE, biologically-motivated + fair): a reward-driven gate with a RECURRENT
    onset-latch state g ("have I loaded the controller yet"), drive = w.code + w_g*g + b; after a LOAD g->1
    (self-sustaining within the sentence), suppressing further firing if w_g learns negative. Sees the token code AND its
    OWN recurrent state (NOT a position label). Trained ONLY on the verb-prediction reward (REINFORCE, no role label).
    Question: does it learn "fire the first content token, then latch off" -> role-gate the position-0 subject and HOLD
    the same-pool distractors, generalising to novel fillers? (The latch is a host recurrent state; the REINFORCE math is
    host — their spiking DA-gated realisations are the named next rungs, per brain-based-only.)
  * role-LEARNED-posoracle (DIAGNOSTIC CEILING, a HOST positional oracle): same policy but the extra input is the raw
    normalised position t/(nC-1) (the sign is LEARNED, not handed). Isolates "can the REWARD drive position-gating when an
    explicit position signal is available" from "can the recurrent latch PROVIDE that signal" -> maps which half is the
    residual. Labelled a host oracle (NOT a spiking mechanism).

THE CRUX TOOTH — the TOKEN-IDENTITY control: over held-out nouns that appear at BOTH position 0 (subject) and position >0
(distractor), the gate must fire DIFFERENTLY by POSITION (LOAD at 0, IGNORE at >0). token_identity_gap = mean over those
nouns of (fire-rate@pos0 - fire-rate@pos>0). A gate firing on token IDENTITY has gap == 0 (its decision depends only on
the code -> identical at both positions) and FAILS; a ROLE gate has gap ~ 1. This is the exact cheat to catch.

ANTI-CHEATS (all EXECUTE; teeth):
  (1) LESION-the-hold (recur=0, the stateless bridge) -> the bump dies over the distractor span -> collapse.
  (2) ALWAYS-OPEN gate -> every token clear-then-loads -> the LAST distractor's feature overwrites (a RECENCY lure) ->
      chance. The gate (not the attractor alone) protects the latch.
  (3) FEATURE-SCRAMBLE (permuted pool->feature deref) -> chance. The bind/deref is load-bearing.
  (4) REFERENT-SHUFFLE (derange feature->verb) -> ~0. No topic->answer leakage.
  (5) HOLD-NOT-RE-READ -> external input ASSERTED zero across hold+read (the slot SUSTAINS, per D3).
  (6) PERMUTED-POSITION control -> at TEST, swap the subject out of position 0 (a distractor now sits at 0) while keeping
      the TRUE verb -> a position-0 gate loads a distractor -> ~chance. Proves the task (and the gate) are POSITIONAL:
      destroy the position->role mapping and the marker collapses.
  (7) RECENCY gate (fires the LAST content token) -> loads feat(last distractor) -> chance. A positional-but-WRONG gate
      fails -> the role gate must fire the RIGHT position, not just a fixed one.
  Plus baselines to chance: the HTM emergence engine (memorises paths), the best fixed-order n-gram HELD-OUT floor, a
  last-token floor.

GO (memory composition on the ROLE task; marker scaffold) = at a stream point where the n-gram floor is chance, the
marker-ROLE spiking WM held-out branch(verb) acc >= 0.90 AND >= chance+0.20 AND >> HTM/n-gram, lesion collapses,
always-open <= chance+0.15, feature-scramble ~ chance, referent-shuffle ~0, hold-alive>0 zero-input, permuted-position <=
chance+0.15. The ROLE-GATE verdict is reported SEPARATELY + honestly: a LEARNED role gate is a role-GO iff held-out >=
chance+0.20 AND position-selectivity high AND the token-identity control PASSES (gap large) AND identity-LEARNED FAILS it
(gap ~0). HONEST NEGATIVE (first-class, MORE valuable than a scaffolded GO): if NO reward-driven gate learns role from
stream statistics alone (only marker works), that maps the exact residual -> role induction needs a positional/structural
substrate signal, supervised syntax, or the emergence-engine's own sequence code. NOT fabricated: subjects & distractors
share the pool, so the subject is NOT separable by identity.

Reuse-by-import (the banked SpikingSlot + D3 hold + RUNG6c barcodes + the EMERGE-14 HTM/stream floors); NO sim/ edit.
SIM_BACKEND=numpy (sub-1k-neuron loops are launch-bound: CPU faster).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_role_gate_derisk --seeds 42 --distances 2 3 --n-test 30
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._var_bind_role_gate_derisk --seeds 42 43 44 100 101 102 \
    --distances 2 3 4 --n-test 90 --out research/findings/raw/_var_bind_role_gate/role_gate_6seed.json
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

# --- the banked spiking WM slot (D3 slow-NMDA HOLD + clear-then-load write) + the RUNG6c sparse barcodes ---
from research.runners._var_bind_gated_slot_derisk import SpikingSlot
from research.runners._novel_referent_hebbian_fastweight_derisk import _mint_codes, _DIM
# --- the EMERGE-14 on-bridge HTM engine + the generic n-gram HELD-OUT floor (like-for-like baselines) ---
from research.runners._emerge14_stageC_onbridge_learning_derisk import build_pool_bridge, OnBridgeLearner
from research.runners._emerge_stream_language_derisk import branch_acc, ngram_floor_heldout

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

OUT = Path("research/findings/raw/_var_bind_role_gate/role_gate.json")


# ====================================================================================================================
# The HARDER stream: ONE shared noun pool; subject = position 0; distractors = SAME pool at positions 1..L; the verb
# agrees with the SUBJECT's feature. Token ids: nouns [0, N) | verbs [N, N+F).  chance = 1/F.
# ====================================================================================================================
def role_layout(N, F):
    nouns = list(range(N))
    verbs = list(range(N, N + F))
    feat_of = {i: i % F for i in range(N)}       # agreement lexicon: balanced features (subject feature ~ uniform)
    verb_tok_of_feat = {f: N + f for f in range(F)}
    V = N + F
    return nouns, verbs, feat_of, verb_tok_of_feat, V


def make_role_stream(N, F, L, n_sent, rng, feat_of, verb_tok_of_feat, exclude=None):
    """Sentences [subj_noun] + [L i.i.d. distractor nouns from the SAME pool] + [verb=verb_tok(feat(subj))]. exclude: a
    set of distractor-tuple keys to AVOID (for a disjoint held-out set). Returns (seqs, n_novel)."""
    exclude = exclude or set()
    seqs, novel = [], 0
    for _ in range(n_sent):
        subj = int(rng.integers(0, N))
        distr = tuple(int(rng.integers(0, N)) for _ in range(L))
        seqs.append([subj] + list(distr) + [verb_tok_of_feat[feat_of[subj]]])
        novel += int(distr not in exclude)
    return seqs, novel


def make_role_heldout(N, F, L, n_sent, rng, train_dtuples, feat_of, verb_tok_of_feat, max_tries_mult=60):
    """Held-out test: distractor tuples DISJOINT from train (true held-out generalisation over the intervening span)."""
    seqs = []
    tries = 0
    while len(seqs) < n_sent and tries < n_sent * max_tries_mult:
        tries += 1
        subj = int(rng.integers(0, N))
        distr = tuple(int(rng.integers(0, N)) for _ in range(L))
        if distr in train_dtuples:
            continue
        seqs.append([subj] + list(distr) + [verb_tok_of_feat[feat_of[subj]]])
    gen_defined = len(seqs) >= max(20, 4 * F)
    if not gen_defined and len(seqs) < n_sent:
        extra, _ = make_role_stream(N, F, L, n_sent - len(seqs), rng, feat_of, verb_tok_of_feat)
        seqs += extra
    return seqs, gen_defined


def permute_subject_out(s, rng):
    """PERMUTED-POSITION control: swap the subject (pos 0) with a random distractor position so a DISTRACTOR now sits at
    position 0, keeping the TRUE verb. A position-0 gate now loads a distractor -> ~chance. (If L==0 returns s unchanged.)"""
    toks = s[:-1]; verb = s[-1]
    if len(toks) <= 1:
        return list(s)
    j = int(rng.integers(1, len(toks)))
    swapped = list(toks); swapped[0], swapped[j] = swapped[j], swapped[0]
    return swapped + [verb]


# ====================================================================================================================
# Gate drivers.  Each exposes reset() + decide(t, tok, code, nC) -> bool (LOAD).  t = within-sentence index; nC = #
# content tokens.  The learned gates share ONE policy class with a mode; the extra input is the mode's structural signal.
# ====================================================================================================================
class MarkerRoleGate:
    """SCAFFOLD: fires at the true role position (subject = position 0). Gate timing is host-given -> proves the memory."""
    def reset(self): pass
    def decide(self, t, tok, code, nC): return t == 0


class RecencyGate:
    """A positional-but-WRONG control: fires the LAST content token -> loads feat(last distractor) -> chance."""
    def reset(self): pass
    def decide(self, t, tok, code, nC): return t == nC - 1


class PolicyGate:
    """A reward-driven (REINFORCE) write-gate. p_load = sigmoid(gain*(w.code + w_e*extra + b)). Trained ONLY on the
    verb-prediction reward (three-factor: per-token eligibility x terminal DA), NO role/position label. `mode` selects the
    STRUCTURAL signal `extra` fed alongside the token code:
      * 'identity'  : extra == 0 always (the EXISTING mechanism; code-only -> cannot condition on position).
      * 'recurrent' : extra == g, a self-generated onset-latch state (0 at sentence start; -> 1 after a LOAD, sustained
                      within the sentence). A biologically-motivated recurrent 'have-I-loaded-the-controller-yet' state.
      * 'posoracle' : extra == t/(nC-1), the raw normalised position (a HOST positional ORACLE; the sign is LEARNED)."""
    def __init__(self, mode, dim=_DIM, gain=4.0, lr=0.15, seed=0):
        rng = np.random.default_rng(seed)
        self.mode = mode
        self.w = rng.normal(0, 0.01, dim).astype(np.float32)
        self.w_e = 0.0; self.b = 0.0
        self.gain, self.lr, self.baseline = gain, lr, 0.0
        self.g = 0.0

    def reset(self): self.g = 0.0

    def p_load(self, code, extra):
        z = self.gain * (float(self.w @ code) + self.w_e * float(extra) + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def _extra(self, t, nC):
        if self.mode == "recurrent": return self.g
        if self.mode == "posoracle": return t / max(1, nC - 1)
        return 0.0

    def decide(self, t, tok, code, nC):
        extra = self._extra(t, nC)
        load = self.p_load(code, extra) > 0.5
        if self.mode == "recurrent" and load:
            self.g = 1.0
        return load

    def train(self, stream, code_of, feat_of, verb_tok_of_feat, F, episodes=8):
        """SURROGATE WM faithful to the spiking slot (a WRITE overwrites -> last-LOAD's feature wins; no LOAD -> wrong).
        Trains the gate policy against the verb-prediction reward. Evaluation later uses the REAL spiking slot."""
        for _ in range(episodes):
            order = list(range(len(stream))); np.random.shuffle(order)
            for n in order:
                s = stream[n]; toks = s[:-1]; true_verb = s[-1]; nC = len(toks)
                g = 0.0; cur_feat = -1
                elig_w = np.zeros_like(self.w); elig_we = 0.0; elig_b = 0.0
                for t, tok in enumerate(toks):
                    code = code_of[tok]
                    extra = g if self.mode == "recurrent" else (t / max(1, nC - 1) if self.mode == "posoracle" else 0.0)
                    p = self.p_load(code, extra)
                    load = 1.0 if (np.random.random() < p) else 0.0
                    elig_w += (load - p) * code; elig_we += (load - p) * extra; elig_b += (load - p)
                    if load > 0.5:
                        cur_feat = feat_of[tok]
                        if self.mode == "recurrent":
                            g = 1.0
                pred_verb = verb_tok_of_feat.get(cur_feat, -1)
                reward = 1.0 if pred_verb == true_verb else 0.0
                adv = reward - self.baseline
                self.w += self.lr * adv * elig_w; self.w_e += self.lr * adv * elig_we; self.b += self.lr * adv * elig_b
                self.baseline += 0.05 * (reward - self.baseline)


# ====================================================================================================================
# Spiking-WM evaluation over a sentence set, for a given gate.  The write CONTENT is feat(token) (the agreement lexicon,
# a host scaffold); read_perm = pool->feature deref (identity default; permuted = feature-scramble); verb_map =
# feature->verb (identity default; deranged = referent-shuffle).
# ====================================================================================================================
def eval_role_wm(slot, seqs, gate, code_of, feat_of, verb_tok_of_feat, F, read_perm=None, verb_map=None,
                 write_every=False):
    rp = read_perm if read_perm is not None else list(range(F))
    vf = verb_map if verb_map is not None else verb_tok_of_feat
    ok = 0; feat_ok = 0; alive = []; zero_ok = True
    for s in seqs:
        toks = s[:-1]; true_verb = s[-1]; subj = s[0]; true_feat = feat_of[subj]; nC = len(toks)
        slot.reset(); gate.reset()
        for t, tok in enumerate(toks):
            load = True if write_every else gate.decide(t, tok, code_of[tok], nC)
            if load:
                slot.write(feat_of[tok])              # write content = the gated token's agreement feature (the lexicon)
            else:
                slot.hold()
        shat, a = slot.read(); alive.append(a); zero_ok = zero_ok and slot._zero_input_span
        pred_feat = rp[shat] if 0 <= shat < F else -1
        feat_ok += int(pred_feat == true_feat)
        pred_verb = vf.get(pred_feat, -1)
        ok += int(pred_verb == true_verb)
    n = max(1, len(seqs))
    return ok / n, float(np.mean(alive)) if alive else 0.0, bool(zero_ok), feat_ok / n


def positional_fire(gate, seqs, code_of):
    """The TOKEN-IDENTITY / position-selectivity instrument. Returns (pos0_rate, posgt0_rate, token_identity_gap,
    n_matched_nouns): the gap is the mean over nouns appearing at BOTH pos 0 and pos >0 of (fire-rate@0 - fire-rate@>0).
    A ROLE gate -> gap ~ 1; an IDENTITY gate -> gap == 0 (code-deterministic, identical at both positions)."""
    per0 = defaultdict(list); pergt0 = defaultdict(list); f0 = []; fgt0 = []
    for s in seqs:
        toks = s[:-1]; nC = len(toks); gate.reset()
        for t, tok in enumerate(toks):
            fired = 1.0 if gate.decide(t, tok, code_of[tok], nC) else 0.0
            if t == 0:
                f0.append(fired); per0[tok].append(fired)
            else:
                fgt0.append(fired); pergt0[tok].append(fired)
    both = [k for k in per0 if k in pergt0]
    gaps = [float(np.mean(per0[k]) - np.mean(pergt0[k])) for k in both]
    return (float(np.mean(f0)) if f0 else 0.0, float(np.mean(fgt0)) if fgt0 else 0.0,
            float(np.mean(gaps)) if gaps else 0.0, len(both))


def role_htm(seed, V, train_seqs, test_seqs, L, n_cells=32, k_win=4, act_th=3, epochs=8):
    """The EMERGE-14 on-bridge HTM engine trained on the harder stream -> held-out branch(verb) acc (memorise-not-
    generalise baseline on the identical stream, like-for-like with the barcode-stream finding's 0.004)."""
    b, cells_idx, row, col = build_pool_bridge(V, n_cells, seed, act_th=act_th, coincidence=True)
    lr = OnBridgeLearner(b, row, col, cells_idx, V, n_cells, k_win=k_win, act_th=act_th, lesion=False)
    for _ in range(epochs):
        for s in train_seqs:
            lr.train_sequence(s)
    return branch_acc(lr, test_seqs, L)


def last_token_floor(train, test, F):
    counts = defaultdict(Counter)
    for s in train:
        counts[s[-2]][s[-1]] += 1
    ok = 0.0
    for s in test:
        dist = counts.get(s[-2])
        if not dist:
            ok += 1.0 / F; continue
        top = max(dist.values()); win = [x for x, c in dist.items() if c == top]
        ok += (1.0 / len(win)) if s[-1] in win else 0.0
    return ok / max(1, len(test))


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, N, F, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps, learned_episodes,
              run_always_open):
    rng = np.random.default_rng(seed)
    chance = 1.0 / F
    nouns, verbs, feat_of, verb_tok_of_feat, V = role_layout(N, F)
    code_of = _mint_codes(np.random.default_rng(seed + 7), N)      # a sparse barcode PER NOUN (shared across roles)
    code_of = {i: code_of[i] for i in range(N)}

    # --- the harder stream + a DISJOINT held-out novel-distractor test set ---
    train_seqs, _ = make_role_stream(N, F, L, n_train, rng, feat_of, verb_tok_of_feat)
    train_dtuples = set(tuple(s[1:-1]) for s in train_seqs)
    test_seqs, gen_defined = make_role_heldout(N, F, L, n_test, rng, train_dtuples, feat_of, verb_tok_of_feat)

    def _slot(rc=recur):
        return SpikingSlot(seed, F, recur=rc, hold_steps=hold_steps, load_steps=load_steps, clear_steps=clear_steps)

    # (A) MARKER-ROLE spiking WM on held-out novel distractors == the memory-composition headline
    acc_marker, alive, zero_ok, feat_acc = eval_role_wm(
        _slot(), test_seqs, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F)

    # (1) LESION-the-hold: recur=0 -> cannot sustain across the distractor span
    acc_lesion, alive_les, _, _ = eval_role_wm(
        _slot(rc=0.0), test_seqs, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F)

    # (2) ALWAYS-OPEN: every token clear-then-loads -> the LAST distractor's feature overwrites (recency lure)
    acc_always = None
    if run_always_open:
        ao_test = test_seqs if len(test_seqs) <= 40 else test_seqs[:40]
        acc_always, _, _, _ = eval_role_wm(
            _slot(), ao_test, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F, write_every=True)

    # (3) FEATURE-SCRAMBLE: DERANGE the pool->feature deref (no fixed points -> the read derefs to the wrong feature for
    # EVERY pool) -> collapse. A plain shuffle over only F=4 features leaves ~1 expected fixed point (high-variance,
    # instrument fails to reliably break), so a derangement is required for the tooth to bite cleanly.
    perm = list(range(F)); fs = np.random.default_rng(seed + 11)
    for _ in range(64):
        fs.shuffle(perm)
        if all(perm[i] != i for i in range(F)):
            break
    acc_scramble, _, _, _ = eval_role_wm(
        _slot(), test_seqs, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F, read_perm=perm)

    # (4) REFERENT-SHUFFLE: derange feature->verb (topic->answer broken) -> ~0, no leakage
    order = list(range(F)); dr = np.random.default_rng(seed + 13)
    for _ in range(64):
        dr.shuffle(order)
        if all(order[i] != i for i in range(F)):
            break
    verb_map_shuf = {f: verb_tok_of_feat[order[f]] for f in range(F)}
    acc_refshuf, _, _, _ = eval_role_wm(
        _slot(), test_seqs, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F, verb_map=verb_map_shuf)

    # (6) PERMUTED-POSITION: swap the subject out of position 0 (a distractor sits at 0), TRUE verb kept -> ~chance
    pp_rng = np.random.default_rng(seed + 17)
    pp_test = [permute_subject_out(s, pp_rng) for s in test_seqs]
    acc_permpos, _, _, _ = eval_role_wm(
        _slot(), pp_test, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F)

    # (7) RECENCY gate (positional-but-WRONG): fires the last content token -> feat(last distractor) -> chance
    acc_recency, _, _, _ = eval_role_wm(
        _slot(), test_seqs, RecencyGate(), code_of, feat_of, verb_tok_of_feat, F)

    # baselines: HTM (memorise-not-generalise) + n-gram HELD-OUT floor + last-token floor
    htm_test = role_htm(seed, V, train_seqs, test_seqs, L)
    ngram_test, ngram_order = ngram_floor_heldout(train_seqs, test_seqs, L, F)
    lasttok = last_token_floor(train_seqs, test_seqs, F)

    # --- the LEARNED gates (the genuinely-open piece): identity (existing), recurrent (candidate), posoracle (ceiling) ---
    def _learned(mode, off):
        g = PolicyGate(mode, seed=seed + off)
        g.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=learned_episodes)
        acc, alv, _, _ = eval_role_wm(_slot(), test_seqs, g, code_of, feat_of, verb_tok_of_feat, F)
        p0, pgt0, gap, nmatch = positional_fire(g, test_seqs, code_of)
        return {"acc": acc, "hold_alive": alv, "fire_pos0": p0, "fire_posgt0": pgt0,
                "token_identity_gap": gap, "n_matched": nmatch}

    l_ident = _learned("identity", 5)
    l_recur = _learned("recurrent", 21)
    l_posor = _learned("posoracle", 33)

    return {"seed": seed, "N": N, "F": F, "L": L, "distance": L + 1, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(N) ** L,
            "acc_marker": acc_marker, "feat_acc_marker": feat_acc, "hold_alive": alive, "hold_alive_lesion": alive_les,
            "zero_input_ok": zero_ok, "acc_lesion": acc_lesion, "acc_always_open": acc_always,
            "acc_feature_scramble": acc_scramble, "acc_referent_shuffle": acc_refshuf, "acc_permuted_position": acc_permpos,
            "acc_recency_gate": acc_recency, "htm_test": htm_test, "ngram_floor_test": ngram_test,
            "ngram_order": ngram_order, "lasttok_floor": lasttok,
            "ident_gate": l_ident, "recur_gate": l_recur, "posoracle_gate": l_posor}


def _agg_gate(per, key):
    sub = [p[key] for p in per]
    return {k: float(np.mean([d[k] for d in sub])) for k in ("acc", "hold_alive", "fire_pos0", "fire_posgt0",
            "token_identity_gap", "n_matched")}


def agg(per):
    keys = ["acc_marker", "feat_acc_marker", "hold_alive", "hold_alive_lesion", "acc_lesion", "acc_feature_scramble",
            "acc_referent_shuffle", "acc_permuted_position", "acc_recency_gate", "htm_test", "ngram_floor_test",
            "lasttok_floor"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    ao = [p["acc_always_open"] for p in per if p["acc_always_open"] is not None]
    a["acc_always_open"] = float(np.mean(ao)) if ao else None
    a["ident_gate"] = _agg_gate(per, "ident_gate")
    a["recur_gate"] = _agg_gate(per, "recur_gate")
    a["posoracle_gate"] = _agg_gate(per, "posoracle_gate")
    a.update({"N": per[0]["N"], "F": per[0]["F"], "L": per[0]["L"], "distance": per[0]["distance"],
              "chance": per[0]["chance"], "path_space": per[0]["path_space"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-noun", type=int, default=12, help="shared noun-pool size (subjects AND distractors); >=F")
    ap.add_argument("--n-feat", type=int, default=4, help="# agreement features = # verbs (chance = 1/F)")
    ap.add_argument("--distances", type=int, nargs="+", default=[2, 3], help="distractor-span L (dependency dist = L+1)")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=60, help="held-out novel-distractor test sentences")
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--learned-episodes", type=int, default=8)
    ap.add_argument("--no-always-open", action="store_true", help="skip the expensive clear-per-token always-open arm")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to evaluate GO (default: the largest)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    F = a.n_feat; chance = 1.0 / F
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | N_noun={a.n_noun} F_feat={F} chance={chance:.3f} | L={dists} "
          f"| recur={a.recur} hold_steps={a.hold_steps} | n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds}",
          flush=True)
    np.random.seed(a.seeds[0])                                     # the surrogate REINFORCE sampler uses np.random

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            per = [run_point(s, a.n_noun, F, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                             a.clear_steps, a.learned_episodes, not a.no_always_open) for s in a.seeds]
            p = agg(per); points.append(p)
            ao = "n/a" if p["acc_always_open"] is None else f"{p['acc_always_open']:.3f}"
            ig, rg, pg = p["ident_gate"], p["recur_gate"], p["posoracle_gate"]
            print(f"  [N={a.n_noun} F={F} L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']}] "
                  f"MARKER-ROLE held-out {p['acc_marker']:.3f} (feat {p['feat_acc_marker']:.3f}, hold-alive "
                  f"{p['hold_alive']:.4f}, zero-input {p['zero_input_ok']}) || HTM {p['htm_test']:.3f} | n-gram "
                  f"{p['ngram_floor_test']:.3f} | last-tok {p['lasttok_floor']:.3f} | chance {chance:.3f} || LESION "
                  f"{p['acc_lesion']:.3f} | ALWAYS-OPEN {ao} | FEAT-SCRAMBLE {p['acc_feature_scramble']:.3f} | REF-SHUF "
                  f"{p['acc_referent_shuffle']:.3f} | PERM-POS {p['acc_permuted_position']:.3f} | RECENCY "
                  f"{p['acc_recency_gate']:.3f}", flush=True)
            print(f"     LEARNED gates (acc | fire pos0/pos>0 | token-identity-gap):  "
                  f"IDENTITY(code-only) {ig['acc']:.3f} | {ig['fire_pos0']:.2f}/{ig['fire_posgt0']:.2f} | "
                  f"gap {ig['token_identity_gap']:+.2f}  ||  RECURRENT-latch {rg['acc']:.3f} | "
                  f"{rg['fire_pos0']:.2f}/{rg['fire_posgt0']:.2f} | gap {rg['token_identity_gap']:+.2f}  ||  "
                  f"POS-ORACLE {pg['acc']:.3f} | {pg['fire_pos0']:.2f}/{pg['fire_posgt0']:.2f} | "
                  f"gap {pg['token_identity_gap']:+.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = role_verdict = None
    if err is None and far is not None:
        ig, rg, pg = far["ident_gate"], far["recur_gate"], far["posoracle_gate"]
        print(f"\n-- memory-composition GO (MARKER scaffold) + anti-cheats at L={far['L']} (dist {far['distance']}, "
              f"held-out novel distractors) --", flush=True)
        void_if(not far["gen_defined"], "path space too small to hold out novel distractors -> generalisation UNDEFINED")
        lever("MARKER-ROLE held-out vs LESION-the-hold (recurrence load-bearing)", round(far["acc_lesion"], 3),
              round(far["acc_marker"], 3), required=False)
        lever("MARKER-ROLE held-out vs PERMUTED-POSITION (the task/gate are POSITIONAL)",
              round(far["acc_permuted_position"], 3), round(far["acc_marker"], 3), required=False)
        attributable_to("MARKER-ROLE held-out over the HTM emergence-engine baseline", far["acc_marker"], far["htm_test"])
        attributable_to("MARKER-ROLE held-out over the best n-gram floor", far["acc_marker"], far["ngram_floor_test"])

        gen = far["gen_defined"]
        headline = far["acc_marker"] >= 0.90 and far["acc_marker"] >= chance + 0.20
        beats_htm = far["acc_marker"] >= far["htm_test"] + 0.30
        beats_floor = far["acc_marker"] >= far["ngram_floor_test"] + 0.30
        recurrence = far["acc_marker"] >= far["acc_lesion"] + 0.30
        gate_protects = (far["acc_always_open"] is None) or (far["acc_always_open"] <= chance + 0.15)
        bind_lb = far["acc_feature_scramble"] <= chance + 0.15
        no_leak = far["acc_referent_shuffle"] <= chance + 0.05
        held_alive = far["hold_alive"] > 1e-3 and far["zero_input_ok"]
        positional = far["acc_permuted_position"] <= chance + 0.15
        core = bool(gen and headline and beats_htm and beats_floor and recurrence and gate_protects and bind_lb
                    and no_leak and held_alive and positional)
        go = bool(core and not smoke)

        if not gen:
            verdict = (f"INCONCLUSIVE — L={far['L']} path space {far['path_space']:.0f} too small to hold out novel "
                       f"distractors; increase N_noun/L for a real held-out regime.")
        elif core:
            tag = "GO" if go else "SMOKE-GO (1-seed indicator; run the 6-seed sweep)"
            verdict = (f"{tag} — on the HARDER same-pool positional stream (subject NOT a token class), the memory "
                       f"composition still carries the ROLE-defined subject across NOVEL distractors when the gate TIMING "
                       f"is given: MARKER-ROLE held-out branch(verb) {far['acc_marker']:.3f} >> HTM {far['htm_test']:.3f} "
                       f">> n-gram {far['ngram_floor_test']:.3f} >> chance {chance:.3f}. HOLD load-bearing (lesion "
                       f"{far['acc_lesion']:.3f}, hold-alive {far['hold_alive']:.4f} zero-input), gate protects the latch "
                       f"(always-open {far['acc_always_open']}), bind load-bearing (feat-scramble "
                       f"{far['acc_feature_scramble']:.3f}), no leakage (referent-shuffle {far['acc_referent_shuffle']:.3f}"
                       f"), POSITIONAL (permuted-position {far['acc_permuted_position']:.3f} ~ chance, recency-gate "
                       f"{far['acc_recency_gate']:.3f} ~ chance). This confirms the MEMORY is not the residual; the "
                       f"LEARNED ROLE gate is (see the role-gate verdict). Reuse-by-import; NO sim/ edit.")
        else:
            miss = []
            if not headline: miss.append(f"marker held-out {far['acc_marker']:.3f} not >=0.90/chance+0.20")
            if not beats_htm: miss.append(f"did not clear HTM+0.30 (HTM {far['htm_test']:.3f})")
            if not recurrence: miss.append(f"hold not load-bearing (lesion {far['acc_lesion']:.3f})")
            if not gate_protects: miss.append(f"gate did not protect latch (always-open {far['acc_always_open']})")
            if not bind_lb: miss.append(f"bind not load-bearing (feat-scramble {far['acc_feature_scramble']:.3f})")
            if not no_leak: miss.append(f"leakage (referent-shuffle {far['acc_referent_shuffle']:.3f} > chance)")
            if not held_alive: miss.append(f"hold not alive/zero-input (alive {far['hold_alive']:.4f})")
            if not positional: miss.append(f"not positional (permuted-position {far['acc_permuted_position']:.3f})")
            verdict = ("PARTIAL/NEGATIVE — the memory composition did not clear the GO bar at L={}: ".format(far["L"])
                       + "; ".join(miss) + ". Read the per-arm numbers.")

        # --- the ROLE-GATE verdict: did a LEARNED gate induce ROLE (position) rather than token-class/identity? ---
        print(f"\n-- ROLE-GATE verdict at L={far['L']} (reward-driven, NO role label; the genuinely-open piece) --",
              flush=True)
        lever("token-identity control: IDENTITY-gate gap (must be ~0 -> gates by identity, FAILS)", 0.0,
              round(ig["token_identity_gap"], 3), required=False)
        lever("token-identity control: RECURRENT-gate gap (a ROLE gate -> large -> gates by POSITION)", 0.0,
              round(rg["token_identity_gap"], 3), required=False)

        # a LEARNED gate counts as ROLE iff: above-chance held-out AND high position-selectivity AND the token-identity
        # control PASSES (same token gated differently by position: gap large) AND its acc clears the identity gate.
        def _is_role(gate_stats):
            return (gate_stats["acc"] >= chance + 0.20 and gate_stats["fire_pos0"] >= 0.70
                    and gate_stats["fire_posgt0"] <= 0.30 and gate_stats["token_identity_gap"] >= 0.50)
        recur_role = _is_role(rg)
        posor_role = _is_role(pg)
        identity_fails = ig["token_identity_gap"] <= 0.20         # the existing mechanism cannot gate by position
        role_go = bool(recur_role and identity_fails and not smoke)

        if recur_role and identity_fails:
            tag = "ROLE-GATE POSITIVE" if not smoke else "ROLE-GATE POSITIVE (1-seed indicator; run the 6-seed sweep)"
            role_verdict = (f"{tag} (report with caveat) — a reward-driven write-gate with a RECURRENT onset-latch "
                            f"(REINFORCE on the verb-prediction DA, NO role/position label) LEARNED to gate by POSITION, "
                            f"not identity: held-out branch(verb) {rg['acc']:.3f} (chance {chance:.3f}), fires pos0 "
                            f"{rg['fire_pos0']:.2f} / pos>0 {rg['fire_posgt0']:.2f}, and PASSES the token-identity control "
                            f"— the SAME nouns are LOADed at position 0 and IGNORED as distractors (gap "
                            f"{rg['token_identity_gap']:+.2f} over {rg['n_matched']:.0f} matched nouns). The EXISTING "
                            f"code-only IDENTITY gate FAILS it (acc {ig['acc']:.3f}, gap {ig['token_identity_gap']:+.2f} — "
                            f"gates on token class, cannot condition on position), reproducing the finding's residual on "
                            f"the harder stream. The pos-oracle ceiling reads acc {pg['acc']:.3f} (gap "
                            f"{pg['token_identity_gap']:+.2f}). CAVEAT: the recurrent latch + REINFORCE math are HOST; the "
                            f"on-substrate spiking realisations (a recurrent 'controller-seen' population + three-factor "
                            f"DA-gated plasticity) are the named next rungs.")
        else:
            posor_note = (f" A HOST pos-oracle CEILING (raw position fed in, sign LEARNED) reaches acc {pg['acc']:.3f} / "
                          f"gap {pg['token_identity_gap']:+.2f}" + (" -> so the REWARD can drive position-gating GIVEN an "
                          "explicit position signal; the residual is providing that signal from a fair recurrent state."
                          if _is_role(pg) else " -> even with an explicit position signal the reward-driven gate did not "
                          "cleanly induce role, so the residual is deeper than the signal (the credit assignment).") )
            role_verdict = (f"ROLE-GATE HONEST NEGATIVE (first-class; maps the exact residual, MORE valuable than a "
                            f"scaffolded GO) — the reward-driven RECURRENT-latch gate did NOT cleanly induce syntactic "
                            f"role from stream statistics alone: held-out {rg['acc']:.3f} (chance {chance:.3f}), fires "
                            f"pos0 {rg['fire_pos0']:.2f} / pos>0 {rg['fire_posgt0']:.2f}, token-identity gap "
                            f"{rg['token_identity_gap']:+.2f} over {rg['n_matched']:.0f} matched nouns. The EXISTING "
                            f"code-only IDENTITY gate cannot even represent position (acc {ig['acc']:.3f}, gap "
                            f"{ig['token_identity_gap']:+.2f} ~ 0 -> gates by token class, FAILS the token-identity "
                            f"control), confirming the finding's residual: identity-gating does NOT transfer to a "
                            f"same-pool positional grammar.{posor_note}. Only the hand-wired MARKER (position given) drives "
                            f"the WM to its ceiling ({far['acc_marker']:.3f}). This precisely names the next mechanism: "
                            f"role induction needs a POSITIONAL/STRUCTURAL substrate signal (a spiking ordinal/phase code "
                            f"or a recurrent controller-seen population), supervised syntax, or the emergence-engine's own "
                            f"sequence code — NOT reward over token statistics.")
        print(f"[role-gate] {role_verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR — {err}"
    else:
        verdict = "ERROR — no points computed"

    # --- earned verdict preconditions (VALIDITY travels with the verdict; tools/gates/verdict_preconditions) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("var_bind_role_gate", chance=chance)
        if far is not None:
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out distractor tuples disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("htm_baseline_at_or_below_chance", round(far["htm_test"], 4), expect=lambda x: x <= chance + 0.10,
                       note="the HTM emergence-engine baseline must sit at/below chance on held-out (memorise-not-generalise)")
            Vd.require("ngram_floor_at_chance", round(far["ngram_floor_test"], 4), expect=lambda x: x <= chance + 0.15,
                       note="the best fixed-order n-gram HELD-OUT floor must be pinned near chance (the bar is meaningful)")
            Vd.require("positional_task_marker_beats_permpos", round(far["acc_marker"] - far["acc_permuted_position"], 4),
                       expect=lambda x: x >= 0.30,
                       note="the ROLE is POSITIONAL: the position-0 marker must beat the permuted-position control by a "
                            "margin, else the task is not genuinely positional (VALIDITY of the whole role premise)")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across the hold+read span")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(far is not None and far["gen_defined"] and far["acc_marker"] >= 0.90
                   and far["acc_marker"] >= chance + 0.20 and far["acc_marker"] >= far["htm_test"] + 0.30
                   and far["acc_marker"] >= far["acc_lesion"] + 0.30
                   and (far["acc_always_open"] is None or far["acc_always_open"] <= chance + 0.15)
                   and far["acc_feature_scramble"] <= chance + 0.15 and far["acc_referent_shuffle"] <= chance + 0.05
                   and far["acc_permuted_position"] <= chance + 0.15)
        dec = Vd.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_role_gate", "verdict": verdict, "role_gate_verdict": role_verdict,
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "the banked variable-binding WM (BG-gated slow-NMDA bistable HOLD slot; write = clear-then-load; "
                            "content = the gated token's agreement feature = the lexicon) driven by a ROLE-based write-gate; "
                            "the gate is the object of study — marker (position given) vs code-only identity (the existing "
                            "mechanism) vs a reward-driven RECURRENT onset-latch gate (the candidate) vs a host pos-oracle "
                            "ceiling — all trained ONLY on the verb-prediction reward, NO role/position label",
               "task": "a HARDER same-pool positional agreement stream: ONE shared noun pool, subject = position 0, "
                       "distractors = the SAME pool at positions 1..L (every noun is subject-or-distractor by POSITION, not "
                       "identity); verb agrees with the subject's feature; held-out TEST = disjoint NOVEL distractor tuples; "
                       "anti-cheats: lesion-the-hold + always-open (recency lure) + feature-scramble + referent-shuffle + "
                       "hold-not-re-read + permuted-position + recency-gate + HTM/n-gram/last-token chance floors; CRUX = the "
                       "token-identity control (same noun gated differently by position)",
               "seeds": a.seeds, "config": {"N_noun": a.n_noun, "F_feat": F, "distances": dists, "recur": a.recur,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps,
               "learned_episodes": a.learned_episodes, "n_train": a.n_train, "n_test": a.n_test, "chance": chance,
               "go_distance": go_L}, "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "de-risks the finding's named residual: the barcode-CLASS write-gate on a SAME-POOL POSITIONAL "
                              "grammar where the subject is NOT separable by identity. Reuse-by-import of the banked "
                              "SpikingSlot (D3 slow-NMDA hold) + RUNG6c barcodes + the EMERGE-14 HTM/n-gram floors; NO sim/ "
                              "edit. The write CONTENT (the agreement feature) is a host lexicon scaffold (exactly as verb_of "
                              "was) — the GATE is the object of study. The recurrent latch + REINFORCE math are HOST (their "
                              "spiking DA-gated realisations are the named next rungs). 1-seed is a SMOKE indicator; the "
                              "6-seed sweep is decisive. HONEST NEGATIVE (role NOT learned from reward alone) is the "
                              "expected, first-class outcome and MORE valuable than a scaffolded role-GO."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 114, flush=True)
    print(f"[var_bind_role_gate] VERDICT: {verdict}", flush=True)
    print(f"[var_bind_role_gate] ROLE-GATE: {role_verdict}", flush=True)
    print(f"[var_bind_role_gate] wrote {a.out}\n" + "=" * 114, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
