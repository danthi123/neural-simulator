"""ROLE-GATE x HIDDEN-LAYER + CHAINED-FA + sigma' -- the structural lever (LEVER 2) on the role-gate's
transport-free RELIABILITY residual.

WHY (the precisely-located residual this composes, do NOT re-derive):
  * ROLE-GATE x gap#4 DEEP CREDIT (banked, `_var_bind_rolegate_gap4_credit_derisk`): on a SAME-POOL positional
    agreement stream (subject = position 0; distractors = the SAME noun pool at positions 1..L; verb agrees with the
    subject's feature; held-out = NOVEL distractor tuples), the SINGLE-LAYER role-gate trained by the gap#4 deep-credit
    rule (e-prop forward eligibility + a distal verb-prediction learning signal) reaches role RELIABLY 6/6 (1.000
    [min 1.000]) at the ALIGNED=R^T TRANSPORT CEILING, where plain REINFORCE is high-variance (min 0.233) and a host
    position-oracle fails (0.265). THE CREDIT ASSIGNMENT was the residual -- confirmed + adversarially verified.
  * The TRANSPORT-FREE (brain-based) realisation is the OPEN residual. Canonical co-adapting Kolen-Pollack (Akrout
    2019) RECOVERS feedback alignment (cos(B,R^T) +0.96 -> alignment is STRUCTURAL not dimensional) AND partially
    induces role transport-free (6-seed 0.637, gap +0.52) but is HIGH-VARIANCE (min 0.144), NOT the aligned ceiling.
  * LEVER 1 (readout-regularization) is a BANKED HONEST-NEGATIVE (commit 874f543b): the variance is INTRINSIC to the
    SINGLE-LAYER joint R+B+gate dynamics (some seeds find the role solution, others collapse to fire-everything).

THE HYPOTHESIS (LEVER 2, the structural one): the single-layer gate cannot express the gap#4 lane's PROVEN
transport-free mechanism -- CHAINED multi-hop Feedback Alignment + the sigma' (surrogate-derivative) factor, banked in
`research/findings/2026-08-01-gap4-transport-free-ceiling-FALSIFIED-chained-FA-sigmaprime-clears-it-plus-MNIST-depth4-KP-rescue-6seed.md`
(a transport-free LOCAL rule clearing a depth-2 ceiling 6-seed 0.935, surviving net-depth 4). Its measured
LOAD-BEARING ingredient: sigma' is STRICTLY NECESSARY (cube main effect +0.230, largest + tightest; off collapses the
headline 0.951->0.465) and chained multi-hop feedback x sigma' (interaction +0.301) is what clears the ceiling -- the
wall was direct-DFA-WITHOUT-sigma', NOT the transport-free credit class. So: ADD A HIDDEN LAYER (barcode + the
recurrent latch -> a HIDDEN sigmoid population -> the scalar load-logit) and propagate credit TRANSPORT-FREE by chained
FA + sigma'. Does a hidden layer + chained-FA + sigma' achieve RELIABLE transport-free role induction where the
single-layer canonical KP could not?

THE 2-LAYER GATE (this runner; the ONLY new machinery -- everything else is reuse-by-import).
  Forward per token t (within a sentence):
    a1 = W1 @ code + w_g1 * g + b1                 # HIDDEN pre-activation (H,): code AND the recurrent latch g feed it
    h  = sigmoid(a1)                               # hidden population; sp1 = h*(1-h) is THE hidden sigma'
    z2 = W2 @ h + b2                               # scalar output pre-activation
    p  = sigmoid(gain * z2)                        # LOAD probability; sp2 = gain*p*(1-p) is the output sigma'
    m  = (1-p) m + p v                             # gated memory (v = onehot(feat(token)); hard limit = slot overwrite)
    if p > 0.5: g = 1                              # the recurrent onset-latch ('controller-seen')
  The latch feeds the HIDDEN layer (not a direct output shortcut) DELIBERATELY: it makes the hidden population the
  ROLE-relevant (code x latch) conjunction the chained-FA credit must shape, so the hidden layer is genuinely
  load-bearing for role -- not a deeper path on the role-irrelevant code alone.
  Credit (TWO-PASS per sentence, forward-eligibility approximated exactly as the single-layer EpropCreditGate; NO BPTT,
  NO weight transport):
    delta = softmax(R m_T) - onehot(feat(subj))    # the DISTAL verb-prediction error (verb IS in the stream; feat(subj)
                                                   #   is read ONLY via the verb, never as a label)
    ell   = B_out @ delta                          # HOP A feedback: memory-error dLoss/dm (B_out replaces R^T)
    per token t (leak-weighted w_t = elig_leak^(T-1-t), the e-prop slow-hold trace):
      g_p = ell . (v_t - m_t)                       # memory error projected onto the (scalar) gate output
      e2  = w_t * g_p * sp2_t                        # OUTPUT-unit error
      grad W2 += e2 h_t ;  grad b2 += e2
      e1  = (B1 * e2) * sp1_t                        # HOP B feedback: CHAINED-FA + sigma' (B1 replaces W2^T; sp1 = the
                                                    #   hidden sigma', the 2026-08-01 load-bearing factor)
      grad W1 += outer(e1, code_t) ; grad w_g1 += e1 g_t ; grad b1 += e1
  This is chained multi-hop FA + sigma' -- the backward path reads ONLY the feedback matrices (B_out, B1), NEVER a
  forward weight's transpose (R^T / W2^T / W1^T).

THE FEEDBACK ARMS (Bi at every hop; the credit-fidelity axis):
  * aligned      B_out = R^T (= readout_scale*I) AND B1 = W2 at every hop -- weight TRANSPORT, the credit-fidelity
                 CEILING (a labelled shortcut, mirroring gap#4's BPTT ceiling). Expected reliable; the upper bound.
  * chained_fa   B_out, B1 = FIXED RANDOM -- transport-free FA + sigma'. THE BRAIN-BASED CANDIDATE (matches the
                 2026-08-01 toy's winning 'dfa' arm). This is the arm the GO gate is about.
  * chained_kp   B_out co-adapts with a learnable readout R, B1 co-adapts with W2 -- canonical Kolen-Pollack
                 (Akrout 2019: co-adapt forward + feedback + weight decay) at BOTH hops; transport-free.
  * aligned_nosp aligned feedback but sigma' DROPPED (hidden sp1 forced to 1) -- the sigma' lever; MUST hurt vs aligned
                 (sigma' load-bearing per the 2026-08-01 cube). Isolated on the ceiling so the collapse is attributable
                 to sigma' alone, not a feedback confound.
  * single_layer_ref  the banked EpropCreditGate('kp_canon') at the MATCHED budget -- the 0.637[min 0.144] to BEAT on
                 reliability (min over seeds).
  * permreward   chained_fa with the verb target SHUFFLED per sentence -- the learning signal carries no signal ->
                 ANTI-CHEAT (must collapse to chance / no selectivity).
  + reinforce (matched-budget), marker ceiling, chance, HTM, n-gram floor, lesion-the-hold, the code-only identity gate
    (the token-identity crux control -- must FAIL: gap ~0).

THE CRUX TOOTH (reused verbatim): token_identity_gap = mean over held-out nouns appearing at BOTH position 0 and
position >0 of (fire@0 - fire@>0). An identity gate has gap==0; a ROLE gate has gap ~1.

GO (the brain-based reliability question): TRANSPORT-FREE chained_fa reaches role RELIABLY iff -- across the 6 seeds --
held-out acc mean >= chance+0.30 AND min over seeds >= 0.60 (beating single-layer kp_canon's min 0.144) AND
token-identity gap min >= 0.30, with sigma' load-bearing (aligned_nosp collapses) and permuted-reward collapsing. If
transport-free does NOT clear it but the aligned ceiling does -> a first-class HONEST-NEGATIVE that further isolates the
residual (is it alignment, sigma', depth, or reliability?).

Reuse-by-import of `_var_bind_rolegate_gap4_credit_derisk` (stream, SpikingSlot eval, the crux, EpropCreditGate +
baselines + teeth). The 2-layer gate + chained-FA credit are RUNNER-side host math (their on-substrate spiking DA-gated
realisation is the named next rung). NO sim/ edit. SIM_BACKEND=numpy (sub-1k-neuron LIF loops are launch-bound: CPU
faster). Verified: SpikingSlot's build_persistent_slot sets cfg.seed -> the substrate IS seeded per (seed).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_hidden_chainedFA_derisk --seeds 42 --smoke \
    --distances 2 3
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_hidden_chainedFA_derisk \
    --seeds 42 43 44 100 101 102 --distances 2 3 4 --n-test 90 \
    --out research/findings/raw/_rolegate_hidden_chainedFA/rolegate_hidden_chainedFA_6seed.json
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
from pathlib import Path

import numpy as np

# reuse-by-import: the SAME-POOL positional stream, the REAL spiking SpikingSlot eval, the crux instrument, the
# single-layer EpropCreditGate reference + the baselines/teeth (all re-exported by the gap#4 credit runner).
from research.runners._var_bind_rolegate_gap4_credit_derisk import (
    role_layout, make_role_stream, make_role_heldout, permute_subject_out,
    SpikingSlot, MarkerRoleGate, PolicyGate, EpropCreditGate, _gate_stats,
    eval_role_wm, positional_fire, role_htm, ngram_floor_heldout, _mint_codes, _DIM)

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

OUT = Path("research/findings/raw/_rolegate_hidden_chainedFA/rolegate_hidden_chainedFA.json")


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0


# ====================================================================================================================
# THE 2-LAYER ROLE-GATE (barcode + recurrent latch -> hidden sigmoid population -> scalar load-logit), trained
# TRANSPORT-FREE by chained multi-hop FA + sigma' (two-pass forward-eligibility credit). Subclasses PolicyGate so
# reset() (self.g = 0 latch) + the recurrent-latch decide semantics carry over; ONLY the forward (2-layer) + the
# training rule differ. Deployment (decide) is used verbatim by eval_role_wm on the REAL spiking slot.
# ====================================================================================================================
class HiddenChainedGate(PolicyGate):
    def __init__(self, dim=_DIM, hidden=32, gain=4.0, lr=0.05, seed=0, feedback="chained_fa",
                 kp_lr=0.3, kp_wd=0.01, kp_ro_lr_scale=1.0, elig_leak=1.0, readout_scale=3.0,
                 sigma_prime=True, hidden_gain=1.0, w_init=0.10, b2_init=0.3, homeo=0.10, target_rate=None):
        super().__init__("recurrent", dim=dim, gain=gain, lr=lr, seed=seed)
        self.H = int(hidden)
        self.feedback = feedback           # "aligned" | "chained_fa" | "chained_kp"
        self.sigma_prime = bool(sigma_prime)
        self.hidden_gain = float(hidden_gain)
        self.kp_lr = float(kp_lr); self.kp_wd = float(kp_wd); self.kp_ro_lr_scale = float(kp_ro_lr_scale)
        self.elig_leak = float(elig_leak)
        self.readout_scale = float(readout_scale)
        self.homeo = float(homeo); self.target_rate = target_rate
        self.seed = int(seed)
        rp = np.random.default_rng(seed + 1)
        # layer 1: code (D) + latch g (scalar) -> hidden (H)
        self.W1 = rp.normal(0.0, w_init, (self.H, dim)).astype(np.float64)
        self.w_g1 = np.zeros(self.H, dtype=np.float64)        # latch -> hidden (role-critical; learned by chained-FA)
        self.b1 = rp.normal(0.0, 0.01, self.H).astype(np.float64)
        # layer 2: hidden (H) -> scalar load-logit
        self.W2 = rp.normal(0.0, w_init, self.H).astype(np.float64)
        self.b2 = float(b2_init)
        # alignment reads (transport-free health, gap#4 style): cos(B_hop, forward^T) init -> final, per hop
        self.cos_hopA_init = self.cos_hopA_final = None       # cos(B_out, R^T)  -- memory/readout feedback
        self.cos_hopB_init = self.cos_hopB_final = None       # cos(B1, W2)      -- the NEW chained hop
        # the neutered inherited single-layer params are unused by this gate's decide/train
        self.w = None; self.b = None

    # ---- forward (deployment): p_load from the 2-layer net; used verbatim by eval_role_wm / positional_fire ----
    def _forward(self, code, g):
        a1 = self.W1 @ code + self.w_g1 * g + self.b1
        h = 1.0 / (1.0 + np.exp(-self.hidden_gain * a1))
        z2 = float(self.W2 @ h + self.b2)
        p = 1.0 / (1.0 + np.exp(-self.gain * z2))
        return a1, h, z2, p

    def decide(self, t, tok, code, nC):
        _a1, _h, _z2, p = self._forward(code, self.g)
        load = p > 0.5
        if load:
            self.g = 1.0
        return load

    # ---- training: two-pass forward-eligibility + transport-free chained-FA + sigma' ----
    def train_chained(self, stream, code_of, feat_of, F, episodes=80, perm_reward=False):
        H, D = self.H, self.W1.shape[1]
        rs = self.readout_scale
        rB = np.random.default_rng(self.seed + 202)
        # HOP A feedback (memory <- verb-error). aligned: R^T = rs*I (transport). fixed/kp: separate random stream.
        if self.feedback == "aligned":
            B_out = rs * np.eye(F, dtype=np.float64)
        else:
            B_out = rB.normal(0.0, 1.0 / np.sqrt(F), (F, F))
        # HOP B feedback (hidden <- output). aligned uses B1 = W2 at use-time (transport); fixed/kp: separate random.
        B1 = rB.normal(0.0, 1.0 / np.sqrt(H), H)
        # canonical-KP co-adapts a learnable readout R at hop A (init = the frozen lexicon rs*I).
        R = (rs * np.eye(F, dtype=np.float64)) if self.feedback == "chained_kp" else None
        RT0 = (R.T if R is not None else rs * np.eye(F))
        self.cos_hopA_init = _cos(B_out, RT0)
        self.cos_hopB_init = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)

        for _ in range(episodes):
            order = np.random.permutation(len(stream))
            for n in order:
                s = stream[n]; toks = s[:-1]; T = len(toks)
                true_feat = feat_of[toks[0]]                  # subject feature -- read ONLY via the verb target
                if perm_reward:
                    true_feat = int(np.random.randint(0, F))  # shuffled target -> the learning signal carries no signal
                # -------- forward pass (store per-token quantities for the second pass) --------
                g = 0.0; m = np.zeros(F); p_sum = 0.0
                rec = []
                for t, tok in enumerate(toks):
                    code = code_of[tok]
                    v = np.zeros(F); v[feat_of[tok]] = 1.0
                    a1 = self.W1 @ code + self.w_g1 * g + self.b1
                    h = 1.0 / (1.0 + np.exp(-self.hidden_gain * a1))
                    sp1 = self.hidden_gain * h * (1.0 - h)     # hidden sigma' (THE load-bearing factor)
                    z2 = float(self.W2 @ h + self.b2)
                    p = 1.0 / (1.0 + np.exp(-self.gain * z2))
                    sp2 = self.gain * p * (1.0 - p)            # output sigma'
                    dm = v - m
                    rec.append((code, h, sp1, sp2, dm, g))
                    m = (1.0 - p) * m + p * v
                    p_sum += p
                    if p > 0.5:
                        g = 1.0
                # -------- distal verb-prediction learning signal (transport-free) --------
                logits = (R @ m) if R is not None else (rs * m)
                ez = np.exp(logits - logits.max()); probs = ez / ez.sum()
                delta = probs.copy(); delta[true_feat] -= 1.0
                ell = B_out @ delta                            # dLoss/dm (F,), transport-free
                # -------- backward accumulation (chained-FA + sigma', leak-weighted e-prop trace) --------
                B1_use = self.W2 if self.feedback == "aligned" else B1
                gW2 = np.zeros(H); gb2 = 0.0
                gW1 = np.zeros((H, D)); gwg1 = np.zeros(H); gb1 = np.zeros(H)
                for t in range(T):
                    code, h, sp1, sp2, dm, g_in = rec[t]
                    w_t = self.elig_leak ** (T - 1 - t)        # slow-hold trace weighting (e-prop)
                    g_p = float(ell @ dm)                      # memory error projected onto the (scalar) gate output
                    e2 = w_t * g_p * sp2                        # OUTPUT-unit error
                    gW2 += e2 * h; gb2 += e2
                    sp1_use = sp1 if self.sigma_prime else 1.0  # the sigma' LEVER
                    e1 = (B1_use * e2) * sp1_use                # HOP B: chained-FA + sigma' (transport-free)
                    gW1 += np.outer(e1, code); gwg1 += e1 * g_in; gb1 += e1
                # -------- gradient descent --------
                self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
                self.W1 -= self.lr * gW1; self.w_g1 -= self.lr * gwg1; self.b1 -= self.lr * gb1
                # intrinsic firing-rate homeostasis (Turrigiano companion; keeps the 1-vs-L class imbalance from
                # collapsing the gate silent) -- nudges the output bias toward ~1 LOAD per sentence.
                if self.homeo > 0.0:
                    tgt = self.target_rate if self.target_rate is not None else (1.0 / max(1, T))
                    self.b2 -= self.homeo * (p_sum / max(1, T) - tgt)
                # -------- canonical Kolen-Pollack co-adaptation (transport-free; both hops) --------
                if self.feedback == "chained_kp":
                    gR = np.outer(delta, m)                     # dLoss/dR for logits = R m
                    R -= self.lr * self.kp_ro_lr_scale * (gR + self.kp_wd * R)
                    B_out -= self.kp_lr * self.lr * (gR.T + self.kp_wd * B_out)   # tracks R^T
                    self.W2 -= self.lr * self.kp_wd * self.W2                     # forward decay (KP attractor)
                    B1 -= self.kp_lr * self.lr * (gW2 + self.kp_wd * B1)          # tracks W2 (single output)
        RTf = (R.T if R is not None else rs * np.eye(F))
        self.cos_hopA_final = _cos(B_out, RTf)
        self.cos_hopB_final = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)


def _hidden_stats(gate, slot_factory, test_seqs, code_of, feat_of, verb_tok_of_feat, F):
    st = _gate_stats(gate, slot_factory, test_seqs, code_of, feat_of, verb_tok_of_feat, F)
    st["cos_hopA_init"] = gate.cos_hopA_init; st["cos_hopA_final"] = gate.cos_hopA_final
    st["cos_hopB_init"] = gate.cos_hopB_init; st["cos_hopB_final"] = gate.cos_hopB_final
    return st


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, N, F, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps, episodes,
              hidden, lr, kp_lr, kp_wd, readout_scale, homeo, b2_init, reinf_episodes):
    np.random.seed(seed)                                       # REINFORCE sampler + episode order use np.random
    rng = np.random.default_rng(seed)
    chance = 1.0 / F
    nouns, verbs, feat_of, verb_tok_of_feat, V = role_layout(N, F)
    code_of = _mint_codes(np.random.default_rng(seed + 7), N)
    code_of = {i: code_of[i] for i in range(N)}

    train_seqs, _ = make_role_stream(N, F, L, n_train, rng, feat_of, verb_tok_of_feat)
    train_dtuples = set(tuple(s[1:-1]) for s in train_seqs)
    test_seqs, gen_defined = make_role_heldout(N, F, L, n_test, rng, train_dtuples, feat_of, verb_tok_of_feat)

    def _slot(rc=recur):
        return SpikingSlot(seed, F, recur=rc, hold_steps=hold_steps, load_steps=load_steps, clear_steps=clear_steps)

    # ---- memory ceiling (marker scaffold) + lesion-the-hold teeth ----
    acc_marker, alive, zero_ok, feat_acc = eval_role_wm(
        _slot(), test_seqs, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F)
    acc_lesion, _, _, _ = eval_role_wm(
        _slot(rc=0.0), test_seqs, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F)
    pp_rng = np.random.default_rng(seed + 17)
    pp_test = [permute_subject_out(s, pp_rng) for s in test_seqs]
    acc_permpos, _, _, _ = eval_role_wm(
        _slot(), pp_test, MarkerRoleGate(), code_of, feat_of, verb_tok_of_feat, F)

    htm_test = role_htm(seed, V, train_seqs, test_seqs, L)
    ngram_test, ngram_order = ngram_floor_heldout(train_seqs, test_seqs, L, F)

    # ---- BASELINES: plain-REINFORCE recurrent gate (matched budget) + the code-only identity crux control ----
    g_reinf = PolicyGate("recurrent", gain=4.0, lr=0.15, seed=seed + 21)
    g_reinf.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=episodes)
    reinf = _gate_stats(g_reinf, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)
    g_ident = PolicyGate("identity", gain=4.0, lr=0.15, seed=seed + 5)
    g_ident.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=episodes)
    ident = _gate_stats(g_ident, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)

    # ---- SINGLE-LAYER REFERENCE: the banked EpropCreditGate('kp_canon') at the MATCHED budget (0.637[min 0.144]) ----
    g_sl = EpropCreditGate(gain=4.0, lr=0.08, seed=seed + 61, feedback="kp_canon", kp_lr=kp_lr,
                           elig_leak=1.0, readout_scale=readout_scale, homeo=homeo, b_init=0.5, kp_wd=kp_wd)
    g_sl.train_eprop(train_seqs, code_of, feat_of, F, N, episodes=episodes)
    sl_ref = _gate_stats(g_sl, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)
    sl_ref["bw_cos_init"] = g_sl.bw_cos_init; sl_ref["bw_cos_final"] = g_sl.bw_cos_final

    # ---- THE 2-LAYER HIDDEN + CHAINED-FA + sigma' gates (same deployment; ONLY credit differs) ----
    def _hidden(feedback, off, sigma_prime=True, perm_reward=False):
        g = HiddenChainedGate(hidden=hidden, gain=4.0, lr=lr, seed=seed + off, feedback=feedback, kp_lr=kp_lr,
                              kp_wd=kp_wd, elig_leak=1.0, readout_scale=readout_scale, sigma_prime=sigma_prime,
                              homeo=homeo, b2_init=b2_init)
        g.train_chained(train_seqs, code_of, feat_of, F, episodes=episodes, perm_reward=perm_reward)
        return _hidden_stats(g, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)

    hid_aligned = _hidden("aligned", 131)                     # transport CEILING
    hid_chained_fa = _hidden("chained_fa", 137)               # THE brain-based candidate (transport-free FA + sigma')
    hid_chained_kp = _hidden("chained_kp", 143)               # transport-free co-adapting KP + sigma'
    hid_chained_fa_nosp = _hidden("chained_fa", 149, sigma_prime=False)  # LEVER: sigma' dropped ON THE CANDIDATE (must
                                                              # hurt -- sigma' is the 2026-08-01 load-bearing factor for
                                                              # the TRANSPORT-FREE chained path, not for exact transport)
    hid_permrew = _hidden("chained_fa", 151, perm_reward=True)      # ANTI-CHEAT: permuted reward -> role selectivity collapses

    return {"seed": seed, "N": N, "F": F, "L": L, "distance": L + 1, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(N) ** L, "hidden": hidden,
            "acc_marker": acc_marker, "feat_acc_marker": feat_acc, "hold_alive": alive, "zero_input_ok": zero_ok,
            "acc_lesion": acc_lesion, "acc_permuted_position": acc_permpos, "htm_test": htm_test,
            "ngram_floor_test": ngram_test, "ngram_order": ngram_order,
            "reinforce_gate": reinf, "identity_gate": ident, "single_layer_ref_gate": sl_ref,
            "hidden_aligned_gate": hid_aligned, "hidden_chained_fa_gate": hid_chained_fa,
            "hidden_chained_kp_gate": hid_chained_kp, "hidden_chained_fa_nosp_gate": hid_chained_fa_nosp,
            "hidden_permreward_gate": hid_permrew}


_GATE_KEYS = ["reinforce_gate", "identity_gate", "single_layer_ref_gate", "hidden_aligned_gate",
              "hidden_chained_fa_gate", "hidden_chained_kp_gate", "hidden_chained_fa_nosp_gate",
              "hidden_permreward_gate"]


def _agg_gate(per, key):
    sub = [p[key] for p in per]
    base = {k: float(np.mean([d[k] for d in sub])) for k in ("acc", "hold_alive", "fire_pos0", "fire_posgt0",
            "token_identity_gap", "n_matched")}
    for ck in ("bw_cos_init", "bw_cos_final", "cos_hopA_init", "cos_hopA_final", "cos_hopB_init", "cos_hopB_final"):
        if all(ck in d and d[ck] is not None for d in sub):
            base[ck] = float(np.mean([d[ck] for d in sub]))
    base["acc_min"] = float(np.min([d["acc"] for d in sub]))
    base["acc_std"] = float(np.std([d["acc"] for d in sub]))
    base["gap_min"] = float(np.min([d["token_identity_gap"] for d in sub]))
    return base


def agg(per):
    keys = ["acc_marker", "feat_acc_marker", "hold_alive", "acc_lesion", "acc_permuted_position", "htm_test",
            "ngram_floor_test"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    for gk in _GATE_KEYS:
        a[gk] = _agg_gate(per, gk)
    a.update({"N": per[0]["N"], "F": per[0]["F"], "L": per[0]["L"], "distance": per[0]["distance"],
              "chance": per[0]["chance"], "path_space": per[0]["path_space"], "hidden": per[0]["hidden"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-noun", type=int, default=12, help="shared noun-pool size (subjects AND distractors); >=F")
    ap.add_argument("--n-feat", type=int, default=4, help="# agreement features = # verbs (chance = 1/F)")
    ap.add_argument("--distances", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=80, help="matched training-episode budget across ALL gates")
    ap.add_argument("--hidden", type=int, default=32, help="hidden population size H (the new layer)")
    ap.add_argument("--lr", type=float, default=0.05, help="2-layer gate learning rate")
    ap.add_argument("--kp-lr", type=float, default=0.3)
    ap.add_argument("--kp-wd", type=float, default=0.01)
    ap.add_argument("--readout-scale", type=float, default=3.0)
    ap.add_argument("--homeo", type=float, default=0.10, help="intrinsic firing-rate homeostasis strength (companion)")
    ap.add_argument("--b2-init", type=float, default=0.3, help="output bias init (keeps it firing early)")
    ap.add_argument("--reinforce-episodes", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="fast 1-seed indicator (reduced n-test)")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to gate GO (default: the largest)")
    ap.add_argument("--merge-from", nargs="+", default=None, help="MERGE mode: build the multi-seed aggregate + verdict "
                    "from per-seed artifacts (each a single-seed run of THIS runner) instead of recomputing -- lets the "
                    "6 seeds run in PARALLEL then aggregate through the SAME verdict code. Verified byte-equivalent to a "
                    "native multi-seed run (run_point reseeds per seed; agg() is deterministic).")
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
    F = a.n_feat; chance = 1.0 / F
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | N_noun={a.n_noun} F_feat={F} chance={chance:.3f} | L={dists} "
          f"| recur={a.recur} | episodes={a.episodes} hidden={a.hidden} lr={a.lr} kp_lr={a.kp_lr} kp_wd={a.kp_wd} "
          f"readout_scale={a.readout_scale} homeo={a.homeo} b2_init={a.b2_init} | n_train={a.n_train} "
          f"n_test={a.n_test} | seeds={a.seeds} smoke={smoke}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            if a.merge_from:
                per = [ps for ps in by_L.get(L, [])]          # per-seed points loaded from the parallel artifacts
            else:
                per = [run_point(s, a.n_noun, F, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                                 a.clear_steps, a.episodes, a.hidden, a.lr, a.kp_lr, a.kp_wd, a.readout_scale, a.homeo,
                                 a.b2_init, a.reinforce_episodes) for s in a.seeds]
            p = agg(per); points.append(p)
            re, iy, sl = p["reinforce_gate"], p["identity_gate"], p["single_layer_ref_gate"]
            al, fa, kp = p["hidden_aligned_gate"], p["hidden_chained_fa_gate"], p["hidden_chained_kp_gate"]
            ns, pr = p["hidden_chained_fa_nosp_gate"], p["hidden_permreward_gate"]
            print(f"  [N={a.n_noun} F={F} L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']} "
                  f"H={a.hidden}] MARKER {p['acc_marker']:.3f} | HTM {p['htm_test']:.3f} | n-gram "
                  f"{p['ngram_floor_test']:.3f} | chance {chance:.3f} || LESION {p['acc_lesion']:.3f} PERM-POS "
                  f"{p['acc_permuted_position']:.3f}", flush=True)
            print(f"     gates (acc[min,std] | fire p0/p>0 | id-gap[min] | cos hopA/hopB):", flush=True)
            for tag, gs in (("REINFORCE@%d" % a.episodes, re), ("identity", iy), ("SINGLE-LAYER(kp_canon)", sl),
                            ("hidden_ALIGNED", al), ("hidden_chained_FA", fa), ("hidden_chained_KP", kp),
                            ("hidden_chained_FA_noSP", ns), ("hidden_PERMREW", pr)):
                cos = ""
                if "cos_hopA_final" in gs:
                    cos = (f"  A {gs['cos_hopA_init']:+.2f}->{gs['cos_hopA_final']:+.2f} "
                           f"B {gs['cos_hopB_init']:+.2f}->{gs['cos_hopB_final']:+.2f}")
                elif "bw_cos_final" in gs:
                    cos = f"  cos(B,R^T) {gs['bw_cos_init']:+.2f}->{gs['bw_cos_final']:+.2f}"
                print(f"       {tag:22s} {gs['acc']:.3f}[{gs['acc_min']:.3f},{gs['acc_std']:.3f}] | "
                      f"{gs['fire_pos0']:.2f}/{gs['fire_posgt0']:.2f} | gap {gs['token_identity_gap']:+.2f}"
                      f"[{gs['gap_min']:+.2f}]{cos}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = None; role_go = False
    if err is None and far is not None:
        chance = far["chance"]
        re, iy, sl = far["reinforce_gate"], far["identity_gate"], far["single_layer_ref_gate"]
        al, fa, kp = far["hidden_aligned_gate"], far["hidden_chained_fa_gate"], far["hidden_chained_kp_gate"]
        ns, pr = far["hidden_chained_fa_nosp_gate"], far["hidden_permreward_gate"]

        print(f"\n-- ROLE-GATE x HIDDEN + CHAINED-FA + sigma' verdict at L={far['L']} (dist {far['distance']}, "
              f"held-out novel distractors) --", flush=True)
        try:
            lever("hidden chained-FA vs single-layer kp_canon -- reliability (min over seeds)",
                  round(sl["acc_min"], 3), round(fa["acc_min"], 3), required=False,
                  continuous=f"mean: single-layer {sl['acc']:.3f} vs chained-FA {fa['acc']:.3f}")
            lever("sigma' on the CANDIDATE (chained-FA no-sigma' vs chained-FA) -- the load-bearing chained-FA factor",
                  round(ns["acc"], 3), round(fa["acc"], 3), required=False,
                  continuous=f"hopB cos: no-sigma' {ns.get('cos_hopB_final', float('nan')):+.2f} vs "
                             f"sigma' {fa.get('cos_hopB_final', float('nan')):+.2f}")
            lever("permuted-reward (chained-FA) -- learning signal carries no signal (role selectivity gap collapses)",
                  round(fa["token_identity_gap"], 3), round(pr["token_identity_gap"], 3), required=False,
                  continuous=f"acc: chained-FA {fa['acc']:.3f} vs permuted-reward {pr['acc']:.3f}")
        except LeverError:
            pass
        attributable_to("transport-free reliability attributable to the HIDDEN LAYER (chained-FA min vs single-layer min)",
                        fa["acc_min"], sl["acc_min"])

        def _is_role(g):
            return (g["acc"] >= chance + 0.20 and g["fire_pos0"] >= 0.60 and g["fire_posgt0"] <= 0.40
                    and g["token_identity_gap"] >= 0.30)

        def _reliable(g):   # the 6-seed reliability bar (min over seeds), the whole point of this de-risk
            return (not smoke) and g["acc"] >= chance + 0.30 and g["acc_min"] >= 0.60 and g["gap_min"] >= 0.30
        fa_role = _is_role(fa); aligned_role = _is_role(al)
        identity_fails = iy["token_identity_gap"] <= 0.20
        # sigma' load-bearing FOR THE TRANSPORT-FREE CANDIDATE (mean drop OR the worst-seed reliability drop; on a
        # shallow 1-hidden-layer role solution sigma' may bite mainly on reliability, per the 2026-08-01 depth story)
        sigmaprime_bites = (fa["acc"] >= ns["acc"] + 0.10) or (fa["acc_min"] >= ns["acc_min"] + 0.10)
        # permuted-reward -> the learning signal carries no signal -> ROLE SELECTIVITY collapses (the crux gap is the
        # principled teeth; acc is confounded upward by the early-firing homeostasis prior shared with the real arm)
        perm_collapses = pr["token_identity_gap"] <= 0.20 and pr["acc"] <= fa["acc"] - 0.20
        beats_single = fa["acc_min"] >= sl["acc_min"] + 0.10        # the hidden layer improves worst-seed reliability
        # the strict BRAIN-BASED GO: transport-free chained-FA reaches role RELIABLY
        role_go = bool(fa_role and _reliable(fa) and identity_fails and sigmaprime_bites and perm_collapses
                       and not smoke)

        common = (f"single-layer kp_canon(ref) {sl['acc']:.3f}[min {sl['acc_min']:.3f}] (gap "
                  f"{sl['token_identity_gap']:+.2f}); REINFORCE {re['acc']:.3f}[min {re['acc_min']:.3f}]; "
                  f"hidden_ALIGNED(transport ceiling) {al['acc']:.3f}[min {al['acc_min']:.3f}] "
                  f"(gap {al['token_identity_gap']:+.2f}[min {al['gap_min']:+.2f}], p0/p>0 {al['fire_pos0']:.2f}/"
                  f"{al['fire_posgt0']:.2f}); hidden_chained_FA(transport-free) {fa['acc']:.3f}[min {fa['acc_min']:.3f}] "
                  f"(gap {fa['token_identity_gap']:+.2f}[min {fa['gap_min']:+.2f}], cos A "
                  f"{fa.get('cos_hopA_final', float('nan')):+.2f} B {fa.get('cos_hopB_final', float('nan')):+.2f}); "
                  f"hidden_chained_KP {kp['acc']:.3f}[min {kp['acc_min']:.3f}] (gap {kp['token_identity_gap']:+.2f}, "
                  f"cos A {kp.get('cos_hopA_final', float('nan')):+.2f} B {kp.get('cos_hopB_final', float('nan')):+.2f}); "
                  f"marker {far['acc_marker']:.3f}; chance {chance:.3f}. identity-crux gap {iy['token_identity_gap']:+.2f} "
                  f"(fails={identity_fails}). LEVERS: chained-FA-no-sigma' {ns['acc']:.3f}[min {ns['acc_min']:.3f}] "
                  f"(hopB cos {ns.get('cos_hopB_final', float('nan')):+.2f}; sigma'-bites={sigmaprime_bites}); "
                  f"permuted-reward acc {pr['acc']:.3f} gap {pr['token_identity_gap']:+.2f} (collapses={perm_collapses}). "
                  f"beats-single-layer-min={beats_single}.")
        smoketag = "" if not smoke else " (1-seed indicator; run the 6-seed sweep)"
        if fa_role and (smoke or _reliable(fa)) and identity_fails and sigmaprime_bites and perm_collapses:
            verdict = (f"ROLE-GATE POSITIVE (brain-based, transport-free){smoketag} -- a HIDDEN LAYER + chained "
                       f"multi-hop FA + sigma' INDUCES syntactic role RELIABLY WITHOUT weight transport, where the "
                       f"single-layer canonical KP could not (min {sl['acc_min']:.3f} -> {fa['acc_min']:.3f}). {common} "
                       f"sigma' is the load-bearing chained-FA factor; permuted-reward collapses; the identity gate fails "
                       f"the token-identity crux. CAVEAT: the 2-layer net + chained-FA credit are HOST math; their "
                       f"on-substrate spiking DA-gated realisation is the named next rung. Reuse-by-import; NO sim/ edit.")
        elif aligned_role and (smoke or (not smoke and al["acc_min"] >= 0.60)) and identity_fails:
            verdict = (f"ROLE-GATE HONEST-NEGATIVE at the transport-free bar; the CEILING clears{smoketag} -- adding a "
                       f"hidden layer does NOT (yet) make the TRANSPORT-FREE chained-FA reach role reliably, but the "
                       f"aligned (transport) CEILING with a hidden layer does reach role "
                       f"(aligned {al['acc']:.3f}[min {al['acc_min']:.3f}], gap {al['token_identity_gap']:+.2f}). {common} "
                       f"THE RESIDUAL, isolated: chained-FA reaches acc {fa['acc']:.3f}[min {fa['acc_min']:.3f}] "
                       f"(cos hopA {fa.get('cos_hopA_final', float('nan')):+.2f}, hopB "
                       f"{fa.get('cos_hopB_final', float('nan')):+.2f}); chained-KP {kp['acc']:.3f}[min "
                       f"{kp['acc_min']:.3f}] (cos hopA {kp.get('cos_hopA_final', float('nan')):+.2f}, hopB "
                       f"{kp.get('cos_hopB_final', float('nan')):+.2f}). Read: if the FA/KP cos stays ~0 the residual is "
                       f"ALIGNMENT (co-adapt harder / longer); if cos aligns but acc_min stays < 0.60 the residual is "
                       f"RELIABILITY (seed-dependent collapse), not depth or sigma' (sigma'-bites={sigmaprime_bites}). "
                       f"Next candidate accordingly. Reuse-by-import; NO sim/ edit.")
        else:
            verdict = (f"ROLE-GATE HONEST NEGATIVE (first-class){smoketag} -- a hidden layer + chained-FA + sigma' did "
                       f"NOT reach role even at the transport CEILING. {common} The residual is deeper than the "
                       f"single-layer transport-free credit: it is NOT expressivity/depth alone. Read the per-arm numbers "
                       f"and the cos alignment to isolate whether it is the credit path, the operating point, or the "
                       f"positional signal the gate conditions on. Reuse-by-import; NO sim/ edit.")
        print(f"[rolegate-hidden-chainedFA] {verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR -- {err}"
    else:
        verdict = "ERROR -- no points computed"

    # ---- earned verdict preconditions (validity travels with the verdict) ----
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("var_bind_rolegate_hidden_chainedFA", chance=chance)
        if far is not None:
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out distractor tuples disjoint from train (a real generalisation regime), else UNDEFINED")
            Vd.require("marker_ceiling_exists", round(far["acc_marker"], 4), expect=lambda x: x >= chance + 0.30,
                       note="the memory ceiling (marker scaffold) must clear chance -> a role-induction target exists")
            Vd.require("htm_baseline_at_or_below_chance", round(far["htm_test"], 4), expect=lambda x: x <= chance + 0.10,
                       note="the HTM emergence-engine baseline sits at/below chance on held-out (memorise-not-generalise)")
            Vd.require("ngram_floor_at_chance", round(far["ngram_floor_test"], 4), expect=lambda x: x <= chance + 0.15,
                       note="the best fixed-order n-gram HELD-OUT floor is pinned near chance (the bar is meaningful)")
            Vd.require("positional_task_marker_beats_permpos",
                       round(far["acc_marker"] - far["acc_permuted_position"], 4), expect=lambda x: x >= 0.30,
                       note="the ROLE is POSITIONAL: the position-0 marker beats the permuted-position control (validity)")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across hold+read")
            Vd.control("hidden_differs_from_single_layer",
                       treatment=far["hidden_aligned_gate"]["acc"], control=far["single_layer_ref_gate"]["acc"],
                       min_separation=1e-6, note="the hidden-layer arm must differ from the single-layer reference")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        dec = Vd.decide(role_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_rolegate_hidden_chainedFA", "verdict": verdict, "role_go": bool(role_go),
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "the banked variable-binding WM (D3 slow-NMDA bistable HOLD slot; write = clear-then-load; "
                            "content = the gated token's agreement feature = the lexicon) driven by a 2-LAYER role-gate "
                            "(barcode + recurrent latch -> a HIDDEN sigmoid population -> the scalar load-logit), trained "
                            "TRANSPORT-FREE by chained multi-hop FEEDBACK ALIGNMENT + the sigma' surrogate-derivative "
                            "(the 2026-08-01 load-bearing ingredients) via a two-pass forward-eligibility (e-prop) credit "
                            "rule + intrinsic firing-rate homeostasis. Feedback arms: aligned (weight-transport ceiling) "
                            "/ chained_fa (fixed-random, the brain-based candidate) / chained_kp (co-adapting Kolen-"
                            "Pollack at both hops). Compared to the banked SINGLE-LAYER EpropCreditGate('kp_canon') "
                            "reference (0.637[min 0.144]) + plain REINFORCE + the code-only identity crux control + a "
                            "no-sigma' lever + a permuted-reward anti-cheat",
               "task": "the HARDER same-pool positional agreement stream (subject = position 0; distractors = the SAME "
                       "noun pool at positions 1..L; verb agrees with the subject's feature; held-out = disjoint novel "
                       "distractor tuples); CRUX = the token-identity control (same noun gated differently by position)",
               "seeds": a.seeds, "config": {"N_noun": a.n_noun, "F_feat": F, "distances": dists, "recur": a.recur,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps,
               "episodes": a.episodes, "hidden": a.hidden, "lr": a.lr, "kp_lr": a.kp_lr, "kp_wd": a.kp_wd,
               "readout_scale": a.readout_scale, "homeo": a.homeo, "b2_init": a.b2_init, "n_train": a.n_train,
               "n_test": a.n_test, "chance": chance, "go_distance": go_L},
               "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "LEVER 2 (structural) on the role-gate transport-free reliability residual: adds a HIDDEN "
                              "LAYER so the gate can express the gap#4 lane's PROVEN transport-free mechanism (chained "
                              "multi-hop FA + sigma'). The 2-layer net + chained-FA credit are HOST math (their spiking "
                              "DA-gated realisation is the named next rung). 1-seed is a SMOKE indicator; the 6-seed sweep "
                              "is decisive. If transport-free does NOT clear the reliability bar but the aligned ceiling "
                              "does, that is a first-class HONEST NEGATIVE that further isolates the residual (alignment "
                              "vs sigma' vs depth vs reliability), NOT a fabricated GO."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 114, flush=True)
    print(f"[rolegate-hidden-chainedFA] VERDICT: {verdict}", flush=True)
    print(f"[rolegate-hidden-chainedFA] role_go={role_go}  wrote {a.out}\n" + "=" * 114, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
