"""ROLE-GATE x VARIABLE-SUBJECT-POSITION (LEVER 4) -- make "does transport-free CREDIT induce role" TESTABLE by
removing the ONSET confound that LEVER 3 exposed.

WHY (the methodological confound LEVER 3 exposed -- do NOT re-derive; read the banked finding):
  * LEVER 3 (`_var_bind_rolegate_competitive_stabilizer_derisk`, 6-seed HONEST NEGATIVE) added a COMPETITIVE FORWARD
    STABILIZER (output feedback inhibition + hidden divisive normalisation) that CLOSED the fire-everything basin --
    KP+stabilizer 1.000[min 1.000]. But two killer anti-cheats exposed it as STRUCTURAL, not credit-driven: an UNTRAINED
    stabilized gate (random weights, ZERO learning) scores 1.000 on all 6 seeds, and KP+stabilizer with PERMUTED reward
    also scores 1.000. The stabilizer's fire-ONCE budget is an ONSET GATE that loads the FIRST token -- which in a
    SUBJECT-FIRST stream trivially IS the subject. So the whole arc's task was ONSET-TRIVIAL, and "does transport-free
    credit induce role" was NEVER actually testable.

THE FIX (LEVER 4): a VARIABLE-SUBJECT-POSITION agreement stream where ONSET != THE ANSWER. The design crux -- "what
makes a token the subject if not its position?" -- is answered by a LEARNABLE CUE: a CASE MARKER (the Bates-MacWhinney
COMPETITION MODEL of cue-based role assignment; MacWhinney-Bates-Kliegl 1984; the Japanese case-vs-order studies,
Sasaki & MacWhinney / Kilborn & Ito). In a case language (Japanese-style, FREE word order) the subject is the noun
carrying the NOMINATIVE marker, at ANY ordinal; word order is uninformative. WHICH tag is nominative is ARBITRARY and
LANGUAGE-SPECIFIC -- the comprehender LEARNS it (cue validity = availability x reliability, error-driven). So:
  * The subject noun carries the NOM case tag at a RANDOM ordinal j in [0, L]; the L distractors carry NON-NOM tags.
  * The verb agrees with the SUBJECT NOUN's feature (verb = verb_tok[feat(subj_noun)]).
  * Each token's gate input is a COMPOSITE barcode = OR(noun_barcode, tag_barcode) in the SAME 64-dim code space (so the
    2-layer gate's W1 is unchanged; the noun carries the CONTENT, the tag carries the ROLE cue, both linearly readable).
  * Because the NOM tag->subject mapping is ARBITRARY (one of C random tag barcodes, defined only by the reward), an
    ONSET/positional prior loads a DISTRACTOR, an UNTRAINED gate loads a RANDOM token, and PERMUTED-reward cannot learn
    which tag is NOM -> all three FAIL. Only a LEARNED, content/context-conditioned credit signal can fire the NOM token.

WHY THIS REMOVES THE ONSET CONFOUND (the three validity guarantees, each an executed control):
  (a) subject at a RANDOM ordinal -> the ONSET gate (fires t==0) loads pos0 = subject only 1/(L+1) of the time. At the GO
      distance L=4 that is 1/5 = 0.20 < chance (1/F = 0.25) -> the onset gate FAILS.
  (b) the cue is LEARNABLE-only (arbitrary NOM-tag->role) -> the UNTRAINED stabilized gate and the PERMUTED-reward gate
      cannot know which tag is NOM -> both collapse to the structural onset prior (~1/(L+1)) -> FAIL.
  (c) held-out = NOVEL distractor (noun,tag,position) configurations disjoint from train (generalisation, not memorisation).
  (d) the MARKER/ORACLE ceiling (a gate handed the rule "fire the NOM-tagged noun" = detects the NOM barcode) still
      reaches ~1.000 -> a target exists on the new task.

THE ARMS (all the SAME 2-layer CompetitiveHiddenChainedGate + stabilizer + the REAL spiking D3 SpikingSlot at eval;
reuse-by-import of the LEVER-3 machinery -- ONLY the stream + the subject cue + the eval's subject identification change):
  * case_marker_oracle  the CEILING: fires iff the NOM barcode is present in the code (knows the rule) -> MUST reach ~1.000.
  * onset               fires t==0 -> MUST FAIL (~1/(L+1), <= chance): the proof onset != the answer.
  * aligned_stab        aligned (weight-transport) credit + stabilizer -- the credit-fidelity CEILING (does EXACT feedback
                        learn the cue->role mapping? the upper bound on learning).
  * fa_stab / kp_stab   chained-FA+sigma' / canonical-KP + stabilizer -- the TRANSPORT-FREE candidates -- THE test.
  * fa_nostab/kp_nostab the LEVER-3 no-stabilizer lesion (does the stabilizer still matter on varpos?).
  * untrained_stab      random weights, ZERO training, stabilizer ON -- MUST now FAIL (<= chance+0.15): the proof the onset
                        confound is REMOVED (this scored 1.000 on the subject-first task).
  * permrew_fa/kp_stab  the transport-free candidates with the verb target SHUFFLED per sentence -- MUST now collapse
                        (<= chance+0.15): the proof the LEARNING SIGNAL (credit) is required.
  * identity_control    a fa+stab gate trained + deployed on NOUN-ONLY codes (the tag STRIPPED) -- MUST FAIL: the token/
                        noun identity cannot solve it (every noun is NOM in some sentences, non-NOM in others) -> proves
                        the TAG cue is load-bearing and the gate is not cheating on noun statistics.
  + lesion-the-hold (recur=0), permuted-cue validity (move the NOM tag off the subject -> the oracle fails), the n-gram
    HELD-OUT floor + chance. The CRUX tooth is the TAG-IDENTITY gap (below), not a positional gap.

THE CRUX TOOTH (the varpos analog of token-identity): tag_identity_gap = mean over nouns appearing with BOTH the NOM tag
AND a non-NOM tag of (fire@NOM - fire@non-NOM). A gate keying on NOUN IDENTITY has gap==0 (same noun -> same decision
regardless of tag); a ROLE gate has gap ~1 (loads the noun WHEN NOM-tagged, ignores it when not). Position-fire
(pos0 vs pos>0) is ALSO reported: a role gate on this uniform-subject-position stream fires ~equally by ordinal (it keys
on the TAG, not the ordinal) -- the opposite of an onset gate.

PRIMARY VALIDITY CHECK (must PASS or the task is STILL confounded): UNTRAINED stabilized gate <= chance+0.15 AND
permuted-reward <= chance+0.15. THEN the GO question: does a TRANSPORT-FREE arm (fa_stab OR kp_stab) LEARN role on the
genuinely-hard task -- 6-seed mean acc >= chance+0.30 AND min >= 0.60 AND tag-gap min >= 0.30? Three honest outcomes:
  * GO: transport-free credit induces role on the genuinely-hard task -> the reliability crux is SOLVED.
  * NEGATIVE (credit can't, the ceiling can): aligned (or the oracle) reaches role but transport-free does not -> the
    residual is finally CLEAN (transport-free credit genuinely insufficient on the real task; name the next mechanism).
  * NEGATIVE (nothing learns it): even the aligned credit ceiling struggles -> the TASK/architecture needs work.

Reuse-by-import: `CompetitiveHiddenChainedGate` (LEVER 3: the 2-layer gate + chained-FA/KP transport-free credit + the
competitive stabilizer), the REAL spiking `SpikingSlot`, `role_layout`, `MarkerRoleGate`, `_mint_codes`, `_DIM`, the
n-gram floor. The 2-layer net + chained credit + the competition are HOST math (their on-substrate spiking DA-gated /
lateral-inhibitory realisation is the named next rung). NO sim/ edit. SIM_BACKEND=numpy (sub-1k-neuron LIF loops are
launch-bound: CPU faster). SpikingSlot sets cfg.seed -> the substrate IS seeded per (seed) (verified; banked LEVER-2 note).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_varpos_derisk --seeds 42 --smoke --distances 3 4
6-seed decisive (fan the seeds across processes, then --merge-from):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
    research.runners._var_bind_rolegate_varpos_derisk --seeds $s --distances 2 3 4 --n-test 90 \
    --out research/findings/raw/_rolegate_varpos/seed_$s.json & done ; wait
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_varpos_derisk \
    --merge-from research/findings/raw/_rolegate_varpos/seed_*.json \
    --out research/findings/raw/_rolegate_varpos/varpos_6seed.json
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
from collections import defaultdict
from pathlib import Path

import numpy as np

# reuse-by-import: the LEVER-3 2-layer competitive-stabilizer gate (chained-FA/KP transport-free credit + the stabilizer),
# the feature/verb layout, the onset marker, the REAL spiking slot, the sparse barcodes, the n-gram HELD-OUT floor.
from research.runners._var_bind_rolegate_competitive_stabilizer_derisk import CompetitiveHiddenChainedGate, _cos
from research.runners._var_bind_role_gate_derisk import (
    role_layout, MarkerRoleGate, SpikingSlot, _mint_codes, _DIM)
from research.runners._emerge_stream_language_derisk import ngram_floor_heldout

try:
    from tools.lab import lever, attributable_to, void_if, LeverError
except Exception:  # tools.lab optional at import time; the runner still runs
    class LeverError(Exception):
        pass
    def lever(name, before, after, required=True, continuous=None):
        moved = before != after
        print(f"  LEVER {name}: {before} -> {after} [{'MOVED' if moved else 'UNCHANGED'}]")
        if required and not moved:
            raise LeverError(name)
        return moved
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond:
            print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_rolegate_varpos/varpos.json")


# ====================================================================================================================
# THE VARIABLE-SUBJECT-POSITION STREAM. ONE shared noun pool (N nouns, feature feat(noun) in {0..F-1}); C case tags,
# tag 0 = NOM (the subject cue). Each sentence = L+1 (noun, tag) tokens: the SUBJECT (a random noun, NOM tag) sits at a
# RANDOM ordinal; the L DISTRACTORS are random nouns with random NON-NOM tags. The verb agrees with the SUBJECT's
# feature. So which token is the subject is fixed by the CASE TAG (a learnable arbitrary cue), NOT by position. A
# sentence is a dict: tokens=[(noun,tag),...], subj_pos, subj_noun, verb.  chance = 1/F.
# ====================================================================================================================
def _cfg_key(sent):
    """Held-out key = the DISTRACTOR configuration (position, noun, tag) at the non-subject positions -> a novel filler
    arrangement is a genuine generalisation item. The subject noun is EXCLUDED so held-out is about the fillers/order."""
    return tuple((t, nt[0], nt[1]) for t, nt in enumerate(sent["tokens"]) if t != sent["subj_pos"])


def make_varpos_stream(N, F, C, L, n_sent, rng, feat_of, verb_tok_of_feat, exclude=None, subj_lo=1):
    """n_sent variable-subject-position case-marked sentences. exclude = a set of distractor-config keys to AVOID.
    subj_lo = the LOWEST ordinal the subject may occupy (default 1 -> position 0 is RESERVED as a distractor slot, so the
    ONSET prior the stabilizer's fire-once budget creates lands on a DISTRACTOR and onset acc == chance DECISIVELY -- the
    direct demonstration that onset != the answer). The subject is still at a RANDOM ordinal in [subj_lo, L] (no fixed
    position solves it), and the crux tag-gap rejects any residual positional shortcut (it needs TAG selectivity)."""
    exclude = exclude or set()
    seqs, novel = [], 0
    for _ in range(n_sent):
        subj = int(rng.integers(0, N))
        subj_pos = int(rng.integers(subj_lo, L + 1))           # the subject sits at a RANDOM ordinal in [subj_lo, L]
        toks = [None] * (L + 1)
        toks[subj_pos] = (subj, 0)                             # tag 0 = NOM (the subject cue)
        for t in range(L + 1):
            if t == subj_pos:
                continue
            dn = int(rng.integers(0, N))                       # a distractor: same pool, a NON-NOM tag
            dtag = int(rng.integers(1, C)) if C > 1 else 0
            toks[t] = (dn, dtag)
        sent = {"tokens": toks, "subj_pos": subj_pos, "subj_noun": subj,
                "verb": verb_tok_of_feat[feat_of[subj]]}
        seqs.append(sent)
        novel += int(_cfg_key(sent) not in exclude)
    return seqs, novel


def make_varpos_heldout(N, F, C, L, n_sent, rng, train_keys, feat_of, verb_tok_of_feat, max_tries_mult=80, subj_lo=1):
    """Held-out test: distractor configurations DISJOINT from train (true held-out generalisation over novel fillers)."""
    seqs = []
    tries = 0
    while len(seqs) < n_sent and tries < n_sent * max_tries_mult:
        tries += 1
        cand, _ = make_varpos_stream(N, F, C, L, 1, rng, feat_of, verb_tok_of_feat, subj_lo=subj_lo)
        if _cfg_key(cand[0]) in train_keys:
            continue
        seqs.append(cand[0])
    gen_defined = len(seqs) >= max(20, 4 * F)
    if not gen_defined and len(seqs) < n_sent:
        extra, _ = make_varpos_stream(N, F, C, L, n_sent - len(seqs), rng, feat_of, verb_tok_of_feat, subj_lo=subj_lo)
        seqs += extra
    return seqs, gen_defined


def permute_cue_out(sent, rng):
    """PERMUTED-CUE validity control (the varpos analog of permuted-position): move the NOM tag OFF the subject onto a
    random distractor position (that distractor becomes NOM-tagged; the true subject gets a NON-NOM tag), KEEP the true
    verb. A gate/oracle that loads the NOM-tagged token now loads a DISTRACTOR -> ~chance. Proves the task is TAG-driven."""
    toks = [list(nt) for nt in sent["tokens"]]
    L1 = len(toks)
    if L1 <= 1:
        return dict(sent)
    others = [t for t in range(L1) if t != sent["subj_pos"]]
    j = int(others[int(rng.integers(0, len(others)))])
    # give the true subject a non-NOM tag; give the chosen distractor the NOM tag
    subj_new_tag = 1 if toks[sent["subj_pos"]][1] == 0 else toks[sent["subj_pos"]][1]
    toks[sent["subj_pos"]][1] = subj_new_tag if subj_new_tag != 0 else 1
    toks[j][1] = 0
    return {"tokens": [tuple(x) for x in toks], "subj_pos": sent["subj_pos"], "subj_noun": sent["subj_noun"],
            "verb": sent["verb"]}


# ====================================================================================================================
# CODES: a sparse barcode per NOUN and per TAG (independent developmental-random 8-of-64 barcodes). The gate input for a
# (noun, tag) token is their binary OR (superposition) in the SAME 64-dim space -> the noun carries the CONTENT, the tag
# carries the ROLE cue, both linearly readable; W1 (H x 64) is unchanged. use_tag=False -> the NOUN-ONLY code (the tag
# stripped) for the identity control.
# ====================================================================================================================
def build_codes(seed, N, C):
    noun_codes = _mint_codes(np.random.default_rng(seed + 7), N).astype(np.float64)
    tag_codes = _mint_codes(np.random.default_rng(seed + 700), C).astype(np.float64)
    return noun_codes, tag_codes


def compose_code(noun_codes, tag_codes, noun, tag, use_tag=True):
    if not use_tag:
        return noun_codes[noun].copy()
    return np.minimum(noun_codes[noun] + tag_codes[tag], 1.0)   # binary OR (superposition)


# ====================================================================================================================
# THE ROLE-GATE (varpos). Subclass the LEVER-3 CompetitiveHiddenChainedGate so the 2-layer forward + the competitive
# stabilizer + decide/_forward (the DEPLOYED policy, used verbatim on the REAL spiking slot) are INHERITED UNCHANGED.
# The ONLY override is train_varpos: a FAITHFUL copy of the parent train_chained credit loop with EXACTLY TWO changes --
# (1) the per-token code is the precomputed COMPOSITE (noun OR tag), (2) the softmax TARGET is the SUBJECT's feature
# where the subject is the NOM-tagged token at a RANDOM ordinal (NOT position 0). The credit rule (chained multi-hop FA +
# sigma' / canonical KP), the stabilizer forward (output feedback inhibition + hidden divisive normalisation), the e-prop
# leak-weighted eligibility, and the homeostasis are byte-for-byte the parent's -- so any result is attributable to the
# STREAM + the CUE, not a credit-rule change.
# ====================================================================================================================
class VarposCompetitiveGate(CompetitiveHiddenChainedGate):
    def train_varpos(self, train_input, F, episodes=80, perm_reward=False):
        """train_input: list of (codes[T x D float64], feats[T int], subj_feat int). A faithful copy of the parent
        CompetitiveHiddenChainedGate.train_chained loop; ONLY the code source + the subject-feature target differ."""
        H, D = self.H, self.W1.shape[1]
        rs = self.readout_scale
        rB = np.random.default_rng(self.seed + 202)            # SAME RNG stream/order as the parent (clean lesion)
        if self.feedback == "aligned":
            B_out = rs * np.eye(F, dtype=np.float64)
        else:
            B_out = rB.normal(0.0, 1.0 / np.sqrt(F), (F, F))
        B1 = rB.normal(0.0, 1.0 / np.sqrt(H), H)
        R = (rs * np.eye(F, dtype=np.float64)) if self.feedback == "chained_kp" else None
        RT0 = (R.T if R is not None else rs * np.eye(F))
        self.cos_hopA_init = _cos(B_out, RT0)
        self.cos_hopB_init = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)

        for _ in range(episodes):
            order = np.random.permutation(len(train_input))
            for n in order:
                codes, feats, subj_feat = train_input[n]
                T = len(codes)
                true_feat = int(subj_feat)                     # the SUBJECT's feature (subject = the NOM-tagged token)
                if perm_reward:
                    true_feat = int(np.random.randint(0, F))   # shuffled target -> the learning signal carries no signal
                # -------- forward pass (WITH the competitive stabilizer; store per-token quantities) --------
                g = 0.0; m = np.zeros(F); p_sum = 0.0; s_inh = 0.0
                rec = []
                for t in range(T):
                    code = codes[t]
                    v = np.zeros(F); v[feats[t]] = 1.0
                    a1 = self.W1 @ code + self.w_g1 * g + self.b1
                    r1 = 1.0 / (1.0 + np.exp(-self.hidden_gain * a1))
                    denom = 1.0 + self.div_k * float(np.mean(r1))    # hidden divisive normalisation
                    h = r1 / denom
                    sp1 = self.hidden_gain * r1 * (1.0 - r1) / denom  # hidden sigma' (diagonal local slope)
                    z2 = float(self.W2 @ h + self.b2) - self.out_lambda * s_inh   # output feedback inhibition
                    p = 1.0 / (1.0 + np.exp(-self.gain * z2))
                    sp2 = self.gain * p * (1.0 - p)                  # output sigma'
                    dm = v - m
                    rec.append((code, h, sp1, sp2, dm, g))
                    m = (1.0 - p) * m + p * v
                    p_sum += p
                    s_inh = self.inh_leak * s_inh + p               # the pooled inhibitory integrator
                    if p > 0.5:
                        g = 1.0
                # -------- distal verb-prediction learning signal (transport-free) --------
                logits = (R @ m) if R is not None else (rs * m)
                ez = np.exp(logits - logits.max()); probs = ez / ez.sum()
                delta = probs.copy(); delta[true_feat] -= 1.0
                ell = B_out @ delta
                # -------- backward accumulation (chained-FA + sigma', leak-weighted e-prop trace) --------
                B1_use = self.W2 if self.feedback == "aligned" else B1
                gW2 = np.zeros(H); gb2 = 0.0
                gW1 = np.zeros((H, D)); gwg1 = np.zeros(H); gb1 = np.zeros(H)
                for t in range(T):
                    code, h, sp1, sp2, dm, g_in = rec[t]
                    w_t = self.elig_leak ** (T - 1 - t)
                    g_p = float(ell @ dm)
                    e2 = w_t * g_p * sp2
                    gW2 += e2 * h; gb2 += e2
                    sp1_use = sp1 if self.sigma_prime else 1.0     # the sigma' LEVER
                    e1 = (B1_use * e2) * sp1_use                    # HOP B: chained-FA + sigma' (transport-free)
                    gW1 += np.outer(e1, code); gwg1 += e1 * g_in; gb1 += e1
                # -------- gradient descent --------
                self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
                self.W1 -= self.lr * gW1; self.w_g1 -= self.lr * gwg1; self.b1 -= self.lr * gb1
                # intrinsic firing-rate homeostasis (identical to the parent; a slow scalar average -- the STRUCTURAL
                # competition, not this, is what forbids fire-everything)
                if self.homeo > 0.0:
                    tgt = self.target_rate if self.target_rate is not None else (1.0 / max(1, T))
                    self.b2 -= self.homeo * (p_sum / max(1, T) - tgt)
                # -------- canonical Kolen-Pollack co-adaptation (transport-free; both hops) --------
                if self.feedback == "chained_kp":
                    gR = np.outer(delta, m)
                    R -= self.lr * self.kp_ro_lr_scale * (gR + self.kp_wd * R)
                    B_out -= self.kp_lr * self.lr * (gR.T + self.kp_wd * B_out)
                    self.W2 -= self.lr * self.kp_wd * self.W2
                    B1 -= self.kp_lr * self.lr * (gW2 + self.kp_wd * B1)
        RTf = (R.T if R is not None else rs * np.eye(F))
        self.cos_hopA_final = _cos(B_out, RTf)
        self.cos_hopB_final = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)


# ====================================================================================================================
# The CEILING oracle + the onset floor gate. decide(t, tok, code, nC) -> bool (LOAD).
# ====================================================================================================================
class CaseMarkerOracle:
    """THE CEILING (a gate handed the rule): fires iff the NOM barcode is present in the code (template match on the NOM
    tag). Exactly one NOM token per sentence -> fires exactly once -> loads the subject -> ~1.000. This is the varpos
    analog of the position MARKER, and it KNOWS the arbitrary NOM-tag->role rule an untrained/onset gate cannot."""
    def __init__(self, nom_code, thresh=None):
        self.nom = np.asarray(nom_code, dtype=np.float64)
        self.thresh = float(thresh) if thresh is not None else float(self.nom.sum()) * 0.6
    def reset(self): pass
    def decide(self, t, tok, code, nC):
        return float(np.asarray(code, dtype=np.float64) @ self.nom) >= self.thresh


# ====================================================================================================================
# Deployment + instruments over the REAL spiking slot. The gate's decide gets the COMPOSITE code per token; a LOAD writes
# the token's own feature (clear-then-load); the read -> predicted feature -> verb. Subject id = feat(subj_noun).
# ====================================================================================================================
def eval_varpos_wm(slot, sentences, gate, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F, use_tag=True):
    ok = 0; alive = []; zero_ok = True
    for sent in sentences:
        toks = sent["tokens"]; nC = len(toks); true_verb = sent["verb"]
        slot.reset(); gate.reset()
        for t, (noun, tag) in enumerate(toks):
            code = compose_code(noun_codes, tag_codes, noun, tag, use_tag=use_tag)
            if gate.decide(t, noun, code, nC):
                slot.write(feat_of[noun])
            else:
                slot.hold()
        shat, a = slot.read(); alive.append(a); zero_ok = zero_ok and slot._zero_input_span
        pred_verb = verb_tok_of_feat.get(shat, -1) if 0 <= shat < F else -1
        ok += int(pred_verb == true_verb)
    n = max(1, len(sentences))
    return ok / n, float(np.mean(alive)) if alive else 0.0, bool(zero_ok)


def tag_position_fire(gate, sentences, noun_codes, tag_codes, use_tag=True):
    """The crux instrument. tag_identity_gap = mean over nouns appearing with BOTH the NOM tag AND a non-NOM tag of
    (fire@NOM - fire@non-NOM). Also returns fire on NOM vs non-NOM tokens, and fire by ordinal (pos0 vs pos>0) to show
    the gate keys on the TAG, not the onset."""
    nom_fire = []; obl_fire = []; f0 = []; fgt0 = []
    per_nom = defaultdict(list); per_obl = defaultdict(list)
    for sent in sentences:
        toks = sent["tokens"]; nC = len(toks); gate.reset()
        for t, (noun, tag) in enumerate(toks):
            code = compose_code(noun_codes, tag_codes, noun, tag, use_tag=use_tag)
            fired = 1.0 if gate.decide(t, noun, code, nC) else 0.0
            if tag == 0:
                nom_fire.append(fired); per_nom[noun].append(fired)
            else:
                obl_fire.append(fired); per_obl[noun].append(fired)
            if t == 0:
                f0.append(fired)
            else:
                fgt0.append(fired)
    both = [k for k in per_nom if k in per_obl]
    gaps = [float(np.mean(per_nom[k]) - np.mean(per_obl[k])) for k in both]
    return {"fire_nom": float(np.mean(nom_fire)) if nom_fire else 0.0,
            "fire_obl": float(np.mean(obl_fire)) if obl_fire else 0.0,
            "tag_identity_gap": float(np.mean(gaps)) if gaps else 0.0, "n_matched": len(both),
            "fire_pos0": float(np.mean(f0)) if f0 else 0.0, "fire_posgt0": float(np.mean(fgt0)) if fgt0 else 0.0}


def _deploy_stats(gate, slot_factory, sentences, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F, use_tag=True):
    acc, alive, zok = eval_varpos_wm(slot_factory(), sentences, gate, noun_codes, tag_codes, feat_of,
                                     verb_tok_of_feat, F, use_tag=use_tag)
    tf = tag_position_fire(gate, sentences, noun_codes, tag_codes, use_tag=use_tag)
    st = {"acc": float(acc), "hold_alive": float(alive)}
    st.update(tf)
    for ck in ("cos_hopA_init", "cos_hopA_final", "cos_hopB_init", "cos_hopB_final"):
        st[ck] = getattr(gate, ck, None)
    return st


def _flat_seqs(sentences, C, N):
    """Flat token-id sequences for the n-gram HELD-OUT floor: (noun,tag) -> noun*C+tag; verb -> N*C + feat. The last-k
    tokens are random distractors -> held-out contexts are novel -> back-off to chance (the bar is meaningful)."""
    out = []
    for sent in sentences:
        seq = [nt[0] * C + nt[1] for nt in sent["tokens"]]
        seq.append(N * C + (sent["verb"]))                     # verb token kept distinct from the noun-tag ids
        out.append(seq)
    return out


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, N, F, C, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps, episodes,
              hidden, lr, kp_lr, kp_wd, readout_scale, homeo, b2_init, div_k, out_lambda, inh_leak, subj_lo=1):
    np.random.seed(seed)                                       # episode order uses np.random
    rng = np.random.default_rng(seed)
    chance = 1.0 / F
    nouns, verbs, feat_of, verb_tok_of_feat, V = role_layout(N, F)
    noun_codes, tag_codes = build_codes(seed, N, C)

    train_seqs, _ = make_varpos_stream(N, F, C, L, n_train, rng, feat_of, verb_tok_of_feat, subj_lo=subj_lo)
    train_keys = set(_cfg_key(s) for s in train_seqs)
    test_seqs, gen_defined = make_varpos_heldout(N, F, C, L, n_test, rng, train_keys, feat_of, verb_tok_of_feat,
                                                 subj_lo=subj_lo)

    def _slot(rc=recur):
        return SpikingSlot(seed, F, recur=rc, hold_steps=hold_steps, load_steps=load_steps, clear_steps=clear_steps)

    # ---- the CEILING oracle + validity teeth (lesion-the-hold + permuted-cue) ----
    oracle = CaseMarkerOracle(tag_codes[0])
    acc_oracle, alive, zero_ok = eval_varpos_wm(_slot(), test_seqs, oracle, noun_codes, tag_codes, feat_of,
                                                verb_tok_of_feat, F)
    acc_oracle_lesion, _, _ = eval_varpos_wm(_slot(rc=0.0), test_seqs, CaseMarkerOracle(tag_codes[0]), noun_codes,
                                             tag_codes, feat_of, verb_tok_of_feat, F)
    pc_rng = np.random.default_rng(seed + 17)
    pc_test = [permute_cue_out(s, pc_rng) for s in test_seqs]
    acc_permcue, _, _ = eval_varpos_wm(_slot(), pc_test, CaseMarkerOracle(tag_codes[0]), noun_codes, tag_codes,
                                       feat_of, verb_tok_of_feat, F)

    # ---- the ONSET floor (fires t==0) -- MUST FAIL now that the subject is at a random ordinal ----
    onset = _deploy_stats(MarkerRoleGate(), _slot, test_seqs, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F)

    # ---- the n-gram HELD-OUT floor (the bar is meaningful) ----
    try:
        ngram_test, ngram_order = ngram_floor_heldout(_flat_seqs(train_seqs, C, N), _flat_seqs(test_seqs, C, N), L, F)
    except Exception:
        ngram_test, ngram_order = float("nan"), -1

    # ---- precompute per-arm training inputs (composite codes; the identity control gets NOUN-ONLY codes) ----
    def _train_input(use_tag):
        data = []
        for sent in train_seqs:
            codes = [compose_code(noun_codes, tag_codes, nn, tg, use_tag=use_tag) for (nn, tg) in sent["tokens"]]
            feats = [feat_of[nn] for (nn, tg) in sent["tokens"]]
            data.append((codes, feats, feat_of[sent["subj_noun"]]))
        return data
    train_in = _train_input(True)
    train_in_noun = _train_input(False)

    def _gate(feedback, off, stabilize=True, sigma_prime=True, perm_reward=False, train=True, use_tag=True):
        g = VarposCompetitiveGate(hidden=hidden, gain=4.0, lr=lr, seed=seed + off, feedback=feedback,
                                  kp_lr=kp_lr, kp_wd=kp_wd, elig_leak=1.0, readout_scale=readout_scale,
                                  sigma_prime=sigma_prime, homeo=homeo, b2_init=b2_init,
                                  stabilize=stabilize, div_k=div_k, out_lambda=out_lambda, inh_leak=inh_leak)
        if train:
            g.train_varpos(train_in_noun if not use_tag else train_in, F, episodes=episodes, perm_reward=perm_reward)
        return _deploy_stats(g, _slot, test_seqs, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F, use_tag=use_tag)

    # arms (offsets distinct per arm so each gate is independently seeded)
    aligned_stab = _gate("aligned", 131, stabilize=True)                 # credit-fidelity ceiling
    fa_stab = _gate("chained_fa", 137, stabilize=True)                   # transport-free candidate 1
    kp_stab = _gate("chained_kp", 143, stabilize=True)                   # transport-free candidate 2
    fa_nostab = _gate("chained_fa", 137, stabilize=False)                # LEVER-3 no-stabilizer lesion
    kp_nostab = _gate("chained_kp", 143, stabilize=False)                # LEVER-3 no-stabilizer lesion
    untrained_stab = _gate("chained_fa", 161, stabilize=True, train=False)   # VALIDITY: must FAIL (onset confound gone)
    permrew_fa_stab = _gate("chained_fa", 137, stabilize=True, perm_reward=True)   # VALIDITY: must collapse
    permrew_kp_stab = _gate("chained_kp", 143, stabilize=True, perm_reward=True)   # VALIDITY: must collapse
    identity_control = _gate("chained_fa", 173, stabilize=True, use_tag=False)     # noun-only -> must FAIL (crux)

    # the ONSET gate's EXPECTED accuracy = P(pos0 is the subject)*1 + P(pos0 is a distractor)*(1/F). With subj_lo>=1 the
    # subject is never at pos0 -> p_hit=0 -> onset == chance DECISIVELY. (The single-load partial-credit 1/F is why the
    # floor is 1/F, not 0, and why the validity bar must be read against this, not a naive 0.)
    p_hit0 = (1.0 / (L + 1 - subj_lo)) if subj_lo == 0 else 0.0
    onset_expected = p_hit0 + (1.0 - p_hit0) * chance
    return {"seed": seed, "N": N, "F": F, "C": C, "L": L, "distance": L + 1, "chance": chance, "subj_lo": subj_lo,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs),
            "onset_expected": onset_expected, "hidden": hidden,
            "acc_oracle": acc_oracle, "acc_oracle_lesion": acc_oracle_lesion, "acc_permuted_cue": acc_permcue,
            "hold_alive": alive, "zero_input_ok": zero_ok, "ngram_floor_test": ngram_test, "ngram_order": ngram_order,
            "onset_gate": onset, "aligned_stab_gate": aligned_stab, "fa_stab_gate": fa_stab, "kp_stab_gate": kp_stab,
            "fa_nostab_gate": fa_nostab, "kp_nostab_gate": kp_nostab, "untrained_stab_gate": untrained_stab,
            "permrew_fa_stab_gate": permrew_fa_stab, "permrew_kp_stab_gate": permrew_kp_stab,
            "identity_control_gate": identity_control}


_GATE_KEYS = ["onset_gate", "aligned_stab_gate", "fa_stab_gate", "kp_stab_gate", "fa_nostab_gate", "kp_nostab_gate",
              "untrained_stab_gate", "permrew_fa_stab_gate", "permrew_kp_stab_gate", "identity_control_gate"]


def _agg_gate(per, key):
    sub = [p[key] for p in per]
    base = {k: float(np.mean([d[k] for d in sub])) for k in ("acc", "hold_alive", "fire_nom", "fire_obl",
            "fire_pos0", "fire_posgt0", "tag_identity_gap", "n_matched")}
    for ck in ("cos_hopA_init", "cos_hopA_final", "cos_hopB_init", "cos_hopB_final"):
        if all(ck in d and d[ck] is not None for d in sub):
            base[ck] = float(np.mean([d[ck] for d in sub]))
    base["acc_min"] = float(np.min([d["acc"] for d in sub]))
    base["acc_max"] = float(np.max([d["acc"] for d in sub]))
    base["acc_std"] = float(np.std([d["acc"] for d in sub]))
    base["gap_min"] = float(np.min([d["tag_identity_gap"] for d in sub]))
    base["fire_nom_min"] = float(np.min([d["fire_nom"] for d in sub]))
    base["fire_obl_max"] = float(np.max([d["fire_obl"] for d in sub]))
    return base


def agg(per):
    keys = ["acc_oracle", "acc_oracle_lesion", "acc_permuted_cue", "hold_alive", "ngram_floor_test", "onset_expected"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    a["acc_oracle_min"] = float(np.min([p["acc_oracle"] for p in per]))
    for gk in _GATE_KEYS:
        a[gk] = _agg_gate(per, gk)
    a.update({"N": per[0]["N"], "F": per[0]["F"], "C": per[0]["C"], "L": per[0]["L"], "distance": per[0]["distance"],
              "chance": per[0]["chance"], "hidden": per[0]["hidden"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-noun", type=int, default=12, help="shared noun-pool size (subjects AND distractors); >=F")
    ap.add_argument("--n-feat", type=int, default=4, help="# agreement features = # verbs (chance = 1/F)")
    ap.add_argument("--n-tags", type=int, default=4, help="# case tags; tag 0 = NOM (subject cue), 1..C-1 = distractors")
    ap.add_argument("--distances", type=int, nargs="+", default=[4, 5, 6],
                    help="distractor-span L; subject at a RANDOM ordinal in [subj_lo,L] (default excludes pos0)")
    ap.add_argument("--subj-lo", type=int, default=1, help="lowest ordinal the subject may occupy (1 -> pos0 reserved as "
                    "a distractor so the ONSET prior lands on a distractor and onset==chance decisively; 0 -> fully uniform)")
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=80, help="matched training-episode budget across ALL learned gates")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--kp-lr", type=float, default=0.3)
    ap.add_argument("--kp-wd", type=float, default=0.01)
    ap.add_argument("--readout-scale", type=float, default=3.0)
    ap.add_argument("--homeo", type=float, default=0.10)
    ap.add_argument("--b2-init", type=float, default=0.3)
    ap.add_argument("--div-k", type=float, default=2.0)
    ap.add_argument("--out-lambda", type=float, default=4.0)
    ap.add_argument("--inh-leak", type=float, default=1.0)
    ap.add_argument("--smoke", action="store_true", help="fast 1-seed indicator (reduced n-test)")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to gate GO (default: the largest)")
    ap.add_argument("--merge-from", nargs="+", default=None, help="MERGE mode: build the multi-seed aggregate + verdict "
                    "from per-seed artifacts (each a single-seed run of THIS runner) -> lets the 6 seeds run in PARALLEL "
                    "then aggregate through the SAME verdict code (run_point reseeds per seed; agg() is deterministic).")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.merge_from:
        merged_seeds = []
        by_L = {}
        for pth in a.merge_from:
            d = json.loads(Path(pth).read_text())
            for pt in d.get("points", []):
                by_L.setdefault(pt["L"], []).extend(pt.get("per_seed", []))
            for pt in d.get("points", [])[:1]:
                merged_seeds.extend([ps["seed"] for ps in pt.get("per_seed", [])])
        a.seeds = sorted(set(merged_seeds))
        a.distances = sorted(by_L.keys())
    smoke = a.smoke or len(a.seeds) < 6
    if a.smoke:
        a.n_test = min(a.n_test, 30)
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    F = a.n_feat; C = a.n_tags; chance = 1.0 / F
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | N_noun={a.n_noun} F_feat={F} C_tags={C} chance={chance:.3f} | L={dists} "
          f"| recur={a.recur} | episodes={a.episodes} hidden={a.hidden} lr={a.lr} | STABILIZER div_k={a.div_k} "
          f"out_lambda={a.out_lambda} inh_leak={a.inh_leak} | homeo={a.homeo} b2_init={a.b2_init} | "
          f"n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds} smoke={smoke}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            if a.merge_from:
                per = [ps for ps in by_L.get(L, [])]
            else:
                per = [run_point(s, a.n_noun, F, C, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                                 a.clear_steps, a.episodes, a.hidden, a.lr, a.kp_lr, a.kp_wd, a.readout_scale, a.homeo,
                                 a.b2_init, a.div_k, a.out_lambda, a.inh_leak, a.subj_lo) for s in a.seeds]
            p = agg(per); points.append(p)
            print(f"  [N={a.n_noun} F={F} C={C} L={L} dist={L+1} gen={p['gen_defined']} H={a.hidden}] ORACLE "
                  f"{p['acc_oracle']:.3f}[min {p['acc_oracle_min']:.3f}] | onset-prob {p['onset_expected']:.3f} | "
                  f"n-gram {p['ngram_floor_test']:.3f} | chance {chance:.3f} || ORACLE-lesion {p['acc_oracle_lesion']:.3f}"
                  f" PERM-CUE {p['acc_permuted_cue']:.3f}", flush=True)
            print(f"     gates (acc[min,std] | fire NOM/obl [nom_min/obl_max] | pos0/p>0 | tag-gap[min]):", flush=True)
            order = [("onset(FLOOR)", "onset_gate"), ("aligned_STAB(ceiling)", "aligned_stab_gate"),
                     ("chained_FA_STAB", "fa_stab_gate"), ("chained_KP_STAB", "kp_stab_gate"),
                     ("FA_NOSTAB", "fa_nostab_gate"), ("KP_NOSTAB", "kp_nostab_gate"),
                     ("UNTRAINED_STAB", "untrained_stab_gate"), ("FA_PERMREW", "permrew_fa_stab_gate"),
                     ("KP_PERMREW", "permrew_kp_stab_gate"), ("identity_control(noun-only)", "identity_control_gate")]
            for tag, gk in order:
                gs = p[gk]
                print(f"       {tag:28s} {gs['acc']:.3f}[{gs['acc_min']:.3f},{gs['acc_std']:.3f}] | "
                      f"{gs['fire_nom']:.2f}/{gs['fire_obl']:.2f} [{gs['fire_nom_min']:.2f}/{gs['fire_obl_max']:.2f}] | "
                      f"{gs['fire_pos0']:.2f}/{gs['fire_posgt0']:.2f} | gap {gs['tag_identity_gap']:+.2f}"
                      f"[{gs['gap_min']:+.2f}]", flush=True)
            # per-seed for the two candidates + the two validity controls (so the reviewer SEES per-seed behaviour)
            print(f"     per-seed acc | fire(NOM/obl) [FA-stab || KP-stab || UNTRAINED || FA-permrew]:", flush=True)
            for ps in per:
                gf = ps["fa_stab_gate"]; gk = ps["kp_stab_gate"]; gu = ps["untrained_stab_gate"]
                gp = ps["permrew_fa_stab_gate"]
                print(f"       seed {ps['seed']}: FA {gf['acc']:.3f} {gf['fire_nom']:.2f}/{gf['fire_obl']:.2f} || "
                      f"KP {gk['acc']:.3f} {gk['fire_nom']:.2f}/{gk['fire_obl']:.2f} || UNTR {gu['acc']:.3f} "
                      f"{gu['fire_nom']:.2f}/{gu['fire_obl']:.2f} || PERMREW {gp['acc']:.3f} "
                      f"{gp['fire_nom']:.2f}/{gp['fire_obl']:.2f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = None; role_go = False; valid = False
    if err is None and far is not None:
        chance = far["chance"]
        orc = far["acc_oracle"]; onset = far["onset_gate"]
        al = far["aligned_stab_gate"]; fa = far["fa_stab_gate"]; kp = far["kp_stab_gate"]
        unt = far["untrained_stab_gate"]; prf = far["permrew_fa_stab_gate"]; prk = far["permrew_kp_stab_gate"]
        idc = far["identity_control_gate"]

        print(f"\n-- ROLE-GATE x VARIABLE-SUBJECT-POSITION verdict at L={far['L']} (dist {far['distance']}, held-out "
              f"novel fillers; subject at a RANDOM ordinal) --", flush=True)
        tol = chance + 0.15
        # THE PRIMARY VALIDITY CHECK: the task is no longer onset-trivial iff the UNTRAINED gate AND permuted-reward both
        # fail. (On the subject-first LEVER-3 task both scored 1.000 -- that is the confound this lever removes.)
        untrained_fails = unt["acc"] <= tol
        permrew_fa_fails = prf["acc"] <= tol
        permrew_kp_fails = prk["acc"] <= tol
        onset_fails = onset["acc"] <= tol
        valid = bool(untrained_fails and permrew_fa_fails and permrew_kp_fails)
        try:
            lever("VALIDITY: UNTRAINED stabilized gate (was 1.000 on the onset-trivial task) -- must now FAIL",
                  round(orc, 3), round(unt["acc"], 3), required=False,
                  continuous=f"untrained {unt['acc']:.3f} vs chance+0.15 {tol:.3f} (fails={untrained_fails})")
            lever("VALIDITY: permuted-reward FA (learning signal removed) -- must now collapse",
                  round(fa["acc"], 3), round(prf["acc"], 3), required=False,
                  continuous=f"permrew-FA {prf['acc']:.3f} vs chance+0.15 {tol:.3f} (fails={permrew_fa_fails})")
            lever("ONSET floor (subject now at random ordinal) -- must FAIL (~1/(L+1))",
                  round(orc, 3), round(onset["acc"], 3), required=False,
                  continuous=f"onset {onset['acc']:.3f} vs onset-prob {far['onset_expected']:.3f} (fails={onset_fails})")
            lever("identity control (noun-only, tag stripped) -- must FAIL (the tag cue is load-bearing)",
                  round(fa["acc"], 3), round(idc["acc"], 3), required=False,
                  continuous=f"identity-control {idc['acc']:.3f} (gap {idc['tag_identity_gap']:+.2f})")
        except LeverError:
            pass
        attributable_to("transport-free role attributable to CREDIT (FA-stab acc vs permuted-reward FA acc)",
                        fa["acc"], prf["acc"])

        def _is_role(g):
            return (g["acc"] >= chance + 0.20 and g["fire_nom"] >= 0.60 and g["fire_obl"] <= 0.40
                    and g["tag_identity_gap"] >= 0.30)

        def _reliable(g):   # the 6-seed reliability bar (min over seeds) -- the whole point
            return (not smoke) and g["acc"] >= chance + 0.30 and g["acc_min"] >= 0.60 and g["gap_min"] >= 0.30

        oracle_ok = orc >= chance + 0.30                       # a target exists
        aligned_reaches = _is_role(al) and (smoke or al["acc_min"] >= 0.60)
        idc_fails = idc["acc"] <= tol or idc["tag_identity_gap"] <= 0.20
        fa_role = _is_role(fa); kp_role = _is_role(kp)
        fa_go = bool(valid and fa_role and _reliable(fa) and oracle_ok and idc_fails and not smoke)
        kp_go = bool(valid and kp_role and _reliable(kp) and oracle_ok and idc_fails and not smoke)
        role_go = bool(fa_go or kp_go)
        which = "chained-KP" if kp_go else ("chained-FA" if fa_go else None)

        common = (f"VALIDITY[untrained {unt['acc']:.3f}(fails={untrained_fails}), permrew-FA {prf['acc']:.3f}"
                  f"(fails={permrew_fa_fails}), permrew-KP {prk['acc']:.3f}(fails={permrew_kp_fails}), onset "
                  f"{onset['acc']:.3f}(fails={onset_fails}, prob {far['onset_expected']:.3f})] -> task-valid={valid}. "
                  f"ORACLE(ceiling) {orc:.3f}[min {far['acc_oracle_min']:.3f}] (lesion {far['acc_oracle_lesion']:.3f}, "
                  f"perm-cue {far['acc_permuted_cue']:.3f}); n-gram {far['ngram_floor_test']:.3f}; chance {chance:.3f}. "
                  f"aligned_STAB(credit ceiling) {al['acc']:.3f}[min {al['acc_min']:.3f}] fire {al['fire_nom']:.2f}/"
                  f"{al['fire_obl']:.2f} gap {al['tag_identity_gap']:+.2f}[min {al['gap_min']:+.2f}]; chained_FA_STAB "
                  f"{fa['acc']:.3f}[min {fa['acc_min']:.3f}] fire {fa['fire_nom']:.2f}/{fa['fire_obl']:.2f} gap "
                  f"{fa['tag_identity_gap']:+.2f}[min {fa['gap_min']:+.2f}]; chained_KP_STAB {kp['acc']:.3f}[min "
                  f"{kp['acc_min']:.3f}] fire {kp['fire_nom']:.2f}/{kp['fire_obl']:.2f} gap {kp['tag_identity_gap']:+.2f}"
                  f"[min {kp['gap_min']:+.2f}]; identity-control(noun-only) {idc['acc']:.3f} gap "
                  f"{idc['tag_identity_gap']:+.2f} (fails={idc_fails}).")
        smoketag = "" if not smoke else " (1-seed indicator; run the 6-seed sweep)"

        if not valid:
            verdict = (f"ROLE-GATE VARPOS INVALID{smoketag} -- the task is STILL confounded: the primary validity check "
                       f"did NOT pass (untrained and/or permuted-reward exceed chance+0.15). {common} A validity failure "
                       f"means an onset/structural prior can still solve the stream -- FIX the stream (raise the "
                       f"distractor span L so onset-prob << chance, or check the untrained gate's firing) before reading "
                       f"any credit result. NO sim/ edit.")
        elif role_go or (smoke and valid and (fa_role or kp_role) and oracle_ok and idc_fails):
            wtag = "chained-KP" if (kp_role and (smoke or _reliable(kp))) else "chained-FA"
            top = kp if wtag == "chained-KP" else fa
            verdict = (f"ROLE-GATE VARPOS POSITIVE (brain-based, transport-free){smoketag} -- on a GENUINELY-HARD "
                       f"variable-subject-position case-marked stream (onset != the answer; task-valid: untrained "
                       f"{unt['acc']:.3f} and permuted-reward {prf['acc']:.3f}/{prk['acc']:.3f} FAIL <= chance+0.15, onset "
                       f"{onset['acc']:.3f} FAILS), TRANSPORT-FREE credit ({wtag} + the competitive stabilizer) LEARNS to "
                       f"fire the NOM-tagged subject at ANY ordinal: {wtag} {top['acc']:.3f}[min {top['acc_min']:.3f}] "
                       f"fire NOM {top['fire_nom']:.2f}/obl {top['fire_obl']:.2f}, tag-gap {top['tag_identity_gap']:+.2f}"
                       f"[min {top['gap_min']:+.2f}]. {common} The credit is REQUIRED (permuted-reward collapses), the TAG "
                       f"cue is load-bearing (the noun-only identity control fails), the oracle ceiling confirms a target "
                       f"exists. So the reliability crux is SOLVED where the LEVER-3 result was onset-structural. CAVEAT: "
                       f"the 2-layer net + chained credit + the competition are HOST math; their on-substrate spiking "
                       f"DA-gated / lateral-inhibitory realisation is the named next rung. Reuse-by-import; NO sim/ edit.")
        elif aligned_reaches:
            verdict = (f"ROLE-GATE VARPOS HONEST NEGATIVE (first-class; the residual is finally CLEAN){smoketag} -- the "
                       f"task is VALID (untrained {unt['acc']:.3f} + permuted-reward {prf['acc']:.3f}/{prk['acc']:.3f} + "
                       f"onset {onset['acc']:.3f} all FAIL <= chance+0.15) and the credit-fidelity CEILING (aligned = "
                       f"weight-transport) DOES reach role (aligned_STAB {al['acc']:.3f}[min {al['acc_min']:.3f}], "
                       f"gap {al['tag_identity_gap']:+.2f}) -- but the TRANSPORT-FREE arms do NOT clear the reliability "
                       f"bar (FA {fa['acc']:.3f}[min {fa['acc_min']:.3f}], KP {kp['acc']:.3f}[min {kp['acc_min']:.3f}]). "
                       f"{common} This is the CLEAN residual the whole arc chased: on a genuinely-hard task where onset "
                       f"cannot cheat, EXACT feedback learns the cue->role mapping but transport-free chained-FA/KP does "
                       f"not (yet) reach it reliably -- so transport-free credit is genuinely insufficient HERE. NEXT "
                       f"MECHANISM (the precisely-isolated residual): the feedback-alignment quality of the transport-free "
                       f"hop under the harder cue-detection objective (co-adapt harder / longer, or a deeper hidden code "
                       f"that makes the NOM template linearly cleaner), trained WITH this credit rule. NO sim/ edit.")
        else:
            verdict = (f"ROLE-GATE VARPOS HONEST NEGATIVE (first-class; the TASK/architecture needs work){smoketag} -- the "
                       f"task is VALID (untrained/permuted-reward/onset all FAIL) and the oracle ceiling reaches "
                       f"{orc:.3f}, but EVEN the aligned (weight-transport) credit CEILING does not cleanly reach role "
                       f"(aligned_STAB {al['acc']:.3f}[min {al['acc_min']:.3f}], fire NOM {al['fire_nom']:.2f}/obl "
                       f"{al['fire_obl']:.2f}, gap {al['tag_identity_gap']:+.2f}). {common} So the residual is NOT the "
                       f"transport-free credit alone -- the 2-layer gate + this credit rule struggle to learn the "
                       f"arbitrary NOM-tag->role cue detection from the distal verb reward even with perfect feedback. "
                       f"Read the per-arm fire distributions: if fire_nom stays low the gate is not detecting the tag "
                       f"(the cue is not linearly separable enough in the composite code / the objective is too distal); "
                       f"if fire_nom is high but fire_obl is also high the selectivity is the residual. NEXT: strengthen "
                       f"the cue's separability or the credit density before re-testing transport-free. NO sim/ edit.")
        print(f"[rolegate-varpos] {verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR -- {err}"
    else:
        verdict = "ERROR -- no points computed"

    # ---- earned verdict preconditions (validity travels with the verdict) ----
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("var_bind_rolegate_varpos", chance=chance)
        if far is not None:
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out distractor configs disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("oracle_ceiling_exists", round(far["acc_oracle"], 4), expect=lambda x: x >= chance + 0.30,
                       note="the case-marker oracle (knows the NOM->subject rule) must clear chance -> a target exists")
            Vd.require("ngram_floor_at_chance", round(far["ngram_floor_test"], 4), expect=lambda x: x <= chance + 0.15,
                       note="the best fixed-order n-gram HELD-OUT floor is pinned near chance (the bar is meaningful)")
            Vd.require("onset_gate_fails", round(far["onset_gate"]["acc"], 4), expect=lambda x: x <= chance + 0.15,
                       note="ONSET GATE FAILS -> onset != the answer (the confound LEVER 3 exposed is REMOVED)")
            Vd.require("untrained_gate_fails", round(far["untrained_stab_gate"]["acc"], 4), expect=lambda x: x <= chance + 0.15,
                       note="the UNTRAINED stabilized gate FAILS (it scored 1.000 on the onset-trivial task) -> valid")
            Vd.require("permuted_reward_fails", round(far["permrew_fa_stab_gate"]["acc"], 4), expect=lambda x: x <= chance + 0.15,
                       note="PERMUTED-REWARD FAILS -> the learning signal (credit) is required -> valid")
            Vd.require("oracle_beats_permuted_cue", round(far["acc_oracle"] - far["acc_permuted_cue"], 4),
                       expect=lambda x: x >= 0.30,
                       note="the task is TAG-driven: the NOM oracle beats the permuted-cue control (moving the tag off the subject)")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across hold+read")
            Vd.control("credit_differs_from_permuted_reward",
                       treatment=far["fa_stab_gate"]["acc"], control=far["permrew_fa_stab_gate"]["acc"],
                       min_separation=1e-6, note="the trained transport-free arm must differ from its permuted-reward control")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        dec = Vd.decide(role_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_rolegate_varpos", "verdict": verdict, "role_go": bool(role_go),
               "task_valid": bool(valid), "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke,
               "cost_acknowledged": True, "preconditions": preconditions,
               "mechanism": "the banked variable-binding WM (D3 slow-NMDA bistable HOLD slot; write = clear-then-load; "
                            "content = the gated token's agreement feature = the lexicon) driven by the LEVER-3 2-LAYER "
                            "role-gate (barcode + recurrent latch -> hidden sigmoid population -> scalar load-logit) WITH "
                            "the competitive forward stabilizer (output feedback inhibition + hidden divisive "
                            "normalisation), trained TRANSPORT-FREE by chained multi-hop FA + sigma' / canonical KP. The "
                            "ONLY change vs LEVER 3 is the STREAM + the SUBJECT CUE: the subject is the NOM-case-tagged "
                            "noun at a RANDOM ordinal (a learnable arbitrary cue, the Bates-MacWhinney competition-model "
                            "case marker), so onset != the answer. Arms: case-marker oracle (ceiling) / onset (floor) / "
                            "aligned-stab (credit-fidelity ceiling) / FA-stab + KP-stab (transport-free candidates) + "
                            "no-stab lesions + UNTRAINED-stab + PERMUTED-reward (the two validity killers) + a noun-only "
                            "identity control",
               "task": "a VARIABLE-SUBJECT-POSITION case-marked agreement stream: ONE shared noun pool; the subject "
                       "carries the NOM case tag at a RANDOM ordinal in [0,L]; L distractors carry NON-NOM tags; the verb "
                       "agrees with the subject noun's feature; each token's gate input = OR(noun barcode, tag barcode); "
                       "held-out = disjoint NOVEL distractor configs. CRUX = the TAG-IDENTITY gap (same noun loaded when "
                       "NOM-tagged, ignored when not). This REMOVES the onset confound LEVER 3 exposed (subject-first).",
               "seeds": a.seeds, "config": {"N_noun": a.n_noun, "F_feat": F, "C_tags": C, "distances": dists,
               "subj_lo": a.subj_lo,
               "recur": a.recur, "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps,
               "episodes": a.episodes, "hidden": a.hidden, "lr": a.lr, "kp_lr": a.kp_lr, "kp_wd": a.kp_wd,
               "readout_scale": a.readout_scale, "homeo": a.homeo, "b2_init": a.b2_init, "div_k": a.div_k,
               "out_lambda": a.out_lambda, "inh_leak": a.inh_leak, "n_train": a.n_train, "n_test": a.n_test,
               "chance": chance, "go_distance": go_L},
               "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "LEVER 4 removes the onset confound LEVER 3 exposed by making the subject's ordinal RANDOM "
                              "and cueing it with an ARBITRARY LEARNABLE case tag (Bates-MacWhinney competition model), so "
                              "an onset/untrained/permuted-reward prior loads a DISTRACTOR and only a learned "
                              "content/context-conditioned credit signal can solve it. PRIMARY VALIDITY: untrained AND "
                              "permuted-reward must both FAIL (<= chance+0.15); on the LEVER-3 subject-first task both "
                              "scored 1.000. The 2-layer net + chained credit + the competition are HOST math (their "
                              "spiking realisation is the named next rung). 1-seed is a SMOKE indicator; the 6-seed sweep "
                              "is decisive. Three honest outcomes are all first-class (GO / clean-negative-credit-can't / "
                              "task-needs-work), NOT a fabricated GO."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 114, flush=True)
    print(f"[rolegate-varpos] VERDICT: {verdict}", flush=True)
    print(f"[rolegate-varpos] role_go={role_go} task_valid={valid}  wrote {a.out}\n" + "=" * 114, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
