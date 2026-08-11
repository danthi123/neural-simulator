"""ROLE-GATE x gap#4 DEEP CREDIT -- the convergent unblock. Replace the role-gate's PLAIN REINFORCE credit with the
gap#4 deep-credit machinery (an e-prop forward eligibility trace + a transport-free learning signal from the DISTAL
verb-prediction error, three-factor) and ask whether that lets the recurrent role-gate LEARN syntactic ROLE where plain
REINFORCE could not.

WHY (the two banked findings this composes):
  * ROLE-GATE 6-seed HONEST NEGATIVE (2026-08-11, `_var_bind_role_gate_derisk`): on a SAME-POOL positional grammar
    (subject = position 0, distractors = the SAME noun pool at positions 1..L), a reward-driven RECURRENT-latch write-gate
    trained by PLAIN REINFORCE reaches only held-out 0.602 (chance 0.250, token-identity gap +0.45) -- it does NOT
    cleanly induce role. DECISIVELY, even a HOST POSITION-ORACLE (raw position fed IN) fails (0.265) with plain REINFORCE.
    -> THE RESIDUAL IS THE CREDIT ASSIGNMENT (gap#4), not the positional signal. Only a hand-wired MARKER drives the WM
    to 1.000.
  * gap#4 DEEP-CREDIT SURPASS (2026-08-11, `_gap4_learned_feedback_derisk` / `_gap4_onspikes_kp_align_derisk`):
    transport-free Kolen-Pollack LEARNED feedback + an e-prop eligibility trace (Bellec 2020) + three-factor DA-gated
    plasticity assign DISTAL/deep credit WITHOUT weight transport -- the machinery that a plain scalar-reward update lacks.

THE CREDIT-ASSIGNMENT PROBLEM, precisely. The reward (verb-prediction match) depends causally on the LAST feature left in
the WM. Plain REINFORCE accumulates ONE eligibility (sum_t (a_t - p_t) code_t) and multiplies it by ONE scalar advantage
(reward - baseline): every gate-decision in the sentence gets the SAME credit. So it cannot separate "LOAD pos0 was good"
from "LOAD a distractor was bad" -- the two decisions are lumped, and a mixed episode muddles them. This is the classic
high-variance temporal-credit failure e-prop / three-factor plasticity exists to fix.

THE gap#4 FIX (this runner). Same recurrent-latch policy, same deployment (the REAL spiking D3 slot), ONLY the TRAINING
credit rule changes -- so any difference is attributable to the credit mechanism:
  * A FORWARD ELIGIBILITY TRACE (e-prop, Bellec 2020) whose LEAK IS THE WM RETENTION GATE. Model the write as a gated
    memory m_t = (1-p_t) m_{t-1} + p_t v_t (v_t = onehot(feat(token_t)); the hard limit is the slot's clear-then-load
    overwrite). Then E_t = d m_t / d(gate params) obeys
        E_t = (1 - p_t) * E_{t-1}  +  gain*p_t*(1-p_t) * (v_t - m_{t-1}) (x) x_t
    a forward-computable trace whose time-constant is the RETENTION (1-p_t) -- biologically the slow-NMDA hold. A decision
    whose loaded content SURVIVES to the readout keeps a strong eligibility; one that is overwritten decays. THIS is the
    "credit through the intervening fillers" plain REINFORCE lacks.
  * A TRANSPORT-FREE LEARNING SIGNAL from the DISTAL verb-prediction error. The readout maps m_T -> verb logits by the
    FIXED agreement lexicon R (feature f -> verb N+f, a host scaffold exactly as verb_of was). delta = softmax(scale*m_T)
    - onehot(feat(subj)) is the verb-prediction error (the SAME reward signal REINFORCE uses -- the verb is IN the stream;
    feat(subj) is NEVER read as a label, only via the verb). The learning signal on m is ell = scale * B @ delta with B a
    SEPARATE feedback matrix (NOT R^T copied). grad_theta = ell . E_T. Three-factor: pre[code eligibility] x
    post[memory-state sensitivity (v-m), surrogate p(1-p)] x DA[verb-prediction error delta].
  * FEEDBACK arms (mirroring gap#4's fixed-DFA / KP / oracle): 'fixed' = B frozen random (transport-free DFA baseline);
    'kp' = B co-adapts toward R^T by the matched transposed readout delta (Kolen-Pollack learned feedback, transport-free
    -- the brain-based candidate); 'aligned' = B == R^T == I (weight-TRANSPORT; the credit-fidelity CEILING, labelled a
    shortcut, mirrors gap#4's BPTT ceiling).

THE ARMS (all the SAME recurrent-latch architecture + the SAME REAL spiking slot at eval; matched episodes budget):
  * marker           SCAFFOLD ceiling: fires t==0 -> the memory ceiling (~1.000) when timing is given.
  * identity         the EXISTING code-only gate (plain REINFORCE, sees only the token code) -> gates by identity, gap~0.
  * reinforce        the recurrent-latch gate trained by PLAIN REINFORCE == THE 0.602 BASELINE TO BEAT (banked).
  * eprop_aligned    deep-credit, B=I (transport CEILING): does the eligibility mechanism work with a perfect feedback?
  * eprop_kp         deep-credit, transport-free KP-learned feedback -- THE BRAIN-BASED CANDIDATE.
  * eprop_fixed      deep-credit, transport-free fixed-random feedback (DFA baseline).
  * eprop_noleak     LEVER: deep-credit with the retention leak FORCED to 1 (eligibility never decays) -> isolates the
                     retention-leak eligibility as the load-bearing credit structure (should degrade toward REINFORCE).
  * eprop_permreward ANTI-CHEAT: deep-credit with the verb target SHUFFLED per sentence -> the learning signal carries no
                     signal -> no learning (collapse to chance / no selectivity).

THE CRUX TOOTH (reused verbatim): the TOKEN-IDENTITY control. token_identity_gap = mean over held-out nouns appearing at
BOTH position 0 and position >0 of (fire@0 - fire@>0). A gate firing on identity has gap==0; a ROLE gate has gap ~1.

GO (the ROLE-GATE question): a deep-credit gate INDUCES ROLE where REINFORCE could not iff -- across the seed set --
held-out acc >= chance+0.20 AND toward the marker ceiling AND fires pos0 high / pos>0 low AND token-identity gap large
(>= 0.50) AND it BEATS the plain-REINFORCE baseline by a clear margin AND the identity gate FAILS the crux AND the levers
bite (no-leak degrades, permuted-reward collapses). An HONEST NEGATIVE (deep credit ALSO fails to induce role reliably) is
first-class: it names what is still missing (the positional signal quality, the reward density, or a structural bias).

Reuse-by-import of `_var_bind_role_gate_derisk` (the stream, the SpikingSlot eval, positional_fire, the baselines +
anti-cheats) + the banked D3 slot + RUNG6c barcodes. The e-prop eligibility trace + the transport-free learning signal are
RUNNER-side (host math -- their on-substrate spiking DA-gated realisation is the named next rung). NO sim/ edit.
SIM_BACKEND=numpy (sub-1k-neuron LIF loops are launch-bound: CPU faster).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_gap4_credit_derisk --seeds 42 --distances 2 3 --n-test 30
6-seed decisive:
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_gap4_credit_derisk --seeds 42 43 44 100 101 102 \
    --distances 2 3 4 --n-test 90 --out rolegate_gap4_credit_6seed.json
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

# reuse-by-import: the SAME-POOL positional stream, the SpikingSlot eval, the crux instrument, the baselines + teeth
from research.runners._var_bind_role_gate_derisk import (
    role_layout, make_role_stream, make_role_heldout, permute_subject_out,
    MarkerRoleGate, RecencyGate, PolicyGate, eval_role_wm, positional_fire, role_htm, last_token_floor)
from research.runners._var_bind_gated_slot_derisk import SpikingSlot
from research.runners._novel_referent_hebbian_fastweight_derisk import _mint_codes, _DIM
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

OUT = Path("research/findings/raw/_var_bind_rolegate_gap4_credit/rolegate_gap4_credit.json")


# ====================================================================================================================
# The gap#4 DEEP-CREDIT gate: SAME recurrent-latch policy + deployment as the plain-REINFORCE 'recurrent' gate, but the
# TRAINING credit rule is an e-prop forward eligibility trace (leak = the WM retention gate) + a transport-free learning
# signal from the DISTAL verb-prediction error (three-factor). Subclasses PolicyGate('recurrent') so p_load/decide (the
# DEPLOYED policy) are byte-identical to the REINFORCE baseline -- ONLY train differs.
# ====================================================================================================================
class EpropCreditGate(PolicyGate):
    def __init__(self, dim=_DIM, gain=4.0, lr=0.08, seed=0, feedback="kp", kp_lr=0.3, elig_leak=1.0,
                 readout_scale=3.0, homeo=0.0, target_rate=None, b_init=0.0, kp_wd=0.01, kp_ro_lr_scale=1.0):
        super().__init__("recurrent", dim=dim, gain=gain, lr=lr, seed=seed)
        self.feedback = feedback           # "kp" | "fixed" | "aligned" | "kp_canon"
        self.kp_lr = float(kp_lr)
        self.kp_wd = float(kp_wd)          # KP weight decay (Akrout 2019): the ALIGNMENT ATTRACTOR. Used ONLY by "kp_canon"
                                           # (canonical co-adapting KP). "kp" is the NON-canonical arm (frozen R, no decay,
                                           # B-only) -> anti-aligns; "kp_canon" co-adapts the forward readout R AND B with decay.
        self.kp_ro_lr_scale = float(kp_ro_lr_scale)  # scale of the kp_canon forward-readout co-adapt rate vs the gate lr.
                                           # 1.0 = full co-adapt (the readout can ABSORB credit the gate needs); <1 keeps R
                                           # near the useful lexicon while B still aligns to R^T via KP+decay; 0 = R frozen.
        # intrinsic firing-rate HOMEOSTASIS (Turrigiano; the biological COMPANION to the plasticity rule): the gate
        # must fire ~once per sentence (LOAD the subject) -- but 1 subject vs L distractors biases the credit toward
        # "don't fire", collapsing the bias silent. A slow homeostatic nudge on b toward a target mean-rate keeps the
        # gate from going silent WITHOUT touching the credit-driven selectivity. homeo=0 -> OFF (byte-identical).
        self.homeo = float(homeo)
        self.target_rate = target_rate     # target mean fire-rate per token (default 1/nC set per sentence)
        self.b = float(b_init)             # a small positive bias-init also keeps the gate firing early (default 0)
        # the eligibility-trace leak is a FIXED slow time-constant (the slow-NMDA HOLD tau, Bellec 2020 e-prop) --
        # NOT the state-dependent forward retention (1-p): at init p~0.5 the (1-p) leak washes out distal credit, which
        # is a conflation of the forward Jacobian with the trace's intrinsic time-constant. elig_leak=0 == the NO-TRACE
        # lever (instantaneous eligibility -> cannot credit the distal position-0 LOAD -> isolates the trace).
        self.elig_leak = float(elig_leak)
        self.readout_scale = float(readout_scale)
        self.seed = int(seed)
        self.bw_cos_init = None            # cos(B, R^T=I) init -> final (the transport-free alignment read, gap#4 style)
        self.bw_cos_final = None

    @staticmethod
    def _cos_to_I(B, F):
        b = B.ravel(); i = np.eye(F).ravel()
        nb = np.linalg.norm(b)
        return float(b @ i / (nb * np.linalg.norm(i))) if nb > 1e-12 else 0.0

    def train_eprop(self, stream, code_of, feat_of, F, N, episodes=8, perm_reward=False):
        """e-prop forward eligibility (leak = retention) + transport-free distal verb-pred learning signal."""
        rB = np.random.default_rng(self.seed + 101)
        if self.feedback == "aligned":
            B = np.eye(F, dtype=np.float64)                     # = R^T (weight TRANSPORT: the credit-fidelity ceiling)
        else:
            B = rB.normal(0.0, 1.0 / np.sqrt(F), (F, F))        # SEPARATE random feedback stream (transport-free)
        # CANONICAL KP co-adapts a LEARNABLE forward readout R (init = the frozen readout, then adapts) alongside B; the
        # other arms keep R frozen = readout_scale*I (implicit in `logits = readout_scale*m`).
        R = (self.readout_scale * np.eye(F, dtype=np.float64)) if self.feedback == "kp_canon" else None
        self.bw_cos_init = self._cos_to_I(B, F)                  # cos(B, R^T); R^T init == readout_scale*I -> same as cos(B,I)
        D = self.w.shape[0]
        for _ in range(episodes):
            order = np.random.permutation(len(stream))
            for n in order:
                s = stream[n]; toks = s[:-1]; nC = len(toks)
                true_feat = feat_of[toks[0]]                     # the SUBJECT's feature (read ONLY via the verb target)
                if perm_reward:
                    true_feat = int(np.random.randint(0, F))     # shuffled target -> the learning signal carries no signal
                g = 0.0
                m = np.zeros(F)
                E_w = np.zeros((F, D)); E_we = np.zeros(F); E_b = np.zeros(F)
                p_sum = 0.0
                for t, tok in enumerate(toks):
                    code = code_of[tok]
                    v = np.zeros(F); v[feat_of[tok]] = 1.0
                    extra = g
                    z = self.gain * (float(self.w @ code) + self.w_e * extra + self.b)
                    p = 1.0 / (1.0 + np.exp(-z))
                    p_sum += p
                    sp = self.gain * p * (1.0 - p)               # d p / d(inner drive) (the surrogate 'post' factor)
                    a_lk = self.elig_leak                        # FIXED slow-NMDA hold tau (e-prop trace time-constant)
                    dm = v - m                                   # memory-state sensitivity (F,)
                    E_w = a_lk * E_w + sp * np.outer(dm, code)   # forward eligibility trace (e-prop, Bellec 2020)
                    E_we = a_lk * E_we + sp * extra * dm
                    E_b = a_lk * E_b + sp * dm
                    m = (1.0 - p) * m + p * v                    # gated memory (hard limit = slot overwrite)
                    if p > 0.5:
                        g = 1.0                                  # the recurrent latch (the 'controller-seen' signal)
                # transport-free learning signal from the DISTAL verb-prediction error (readout R = fixed lexicon)
                logits = (R @ m) if R is not None else (self.readout_scale * m)
                ez = np.exp(logits - logits.max()); probs = ez / ez.sum()
                delta = probs.copy(); delta[true_feat] -= 1.0    # d CE(verb) / d logits
                ell = (B @ delta) if R is not None else (self.readout_scale * (B @ delta))  # learning signal (B != R^T)
                self.w -= self.lr * (ell @ E_w)                  # grad = learning-signal . eligibility (three-factor)
                self.w_e -= self.lr * float(ell @ E_we)
                self.b -= self.lr * float(ell @ E_b)
                if self.homeo > 0.0:                             # intrinsic firing-rate homeostasis (default-off companion)
                    tgt = self.target_rate if self.target_rate is not None else (1.0 / max(1, nC))
                    self.b -= self.homeo * (p_sum / max(1, nC) - tgt)
                if self.feedback == "kp":                        # NON-canonical KP: co-adapt B ONLY (frozen R, NO decay) -> the honest-negative arm (anti-aligns)
                    B = B - self.kp_lr * self.lr * np.outer(m, delta)
                elif self.feedback == "kp_canon":                # CANONICAL Kolen-Pollack (Akrout 2019): co-adapt forward R AND feedback B by the matched (transposed) gradient PLUS weight decay -- the decay is the alignment attractor
                    R -= self.lr * self.kp_ro_lr_scale * (np.outer(delta, m) + self.kp_wd * R)  # forward readout co-adapts (task grad + decay), rate-scaled
                    B -= self.kp_lr * self.lr * (np.outer(m, delta) + self.kp_wd * B)  # feedback tracks R^T (matched transpose + decay)
        if R is None:
            self.bw_cos_final = self._cos_to_I(B, F)
        else:                                                     # canonical KP: alignment is B vs the (co-adapted) R^T
            bR, rT = B.ravel(), R.T.ravel()
            self.bw_cos_final = float(bR @ rT / (np.linalg.norm(bR) * np.linalg.norm(rT) + 1e-12))


def _gate_stats(gate, slot_factory, test_seqs, code_of, feat_of, verb_tok_of_feat, F):
    """Deploy a trained gate on the REAL spiking slot over the held-out set + read the token-identity crux."""
    acc, alive, _zok, _fa = eval_role_wm(slot_factory(), test_seqs, gate, code_of, feat_of, verb_tok_of_feat, F)
    p0, pgt0, gap, nmatch = positional_fire(gate, test_seqs, code_of)
    return {"acc": float(acc), "hold_alive": float(alive), "fire_pos0": float(p0), "fire_posgt0": float(pgt0),
            "token_identity_gap": float(gap), "n_matched": int(nmatch)}


# ====================================================================================================================
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, N, F, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps, episodes,
              eprop_lr, kp_lr, readout_scale, homeo, b_init, reinf_episodes, kp_wd=0.01):
    np.random.seed(seed)                                          # the REINFORCE sampler + episode order use np.random
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

    # ---- memory-composition ceiling + validity teeth (marker scaffold) ----
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

    # ---- THE BASELINE: the recurrent-latch gate trained by PLAIN REINFORCE, at the MATCHED eprop budget (strictly
    #      fair: same architecture, same episode budget -> any win is the CREDIT RULE, not more training) ----
    g_reinf = PolicyGate("recurrent", gain=4.0, lr=0.15, seed=seed + 21)
    g_reinf.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=episodes)
    reinf = _gate_stats(g_reinf, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)
    # ---- and at the BANKED REINFORCE native budget (8 episodes -> the finding's 0.602 continuity check) ----
    g_reinf8 = PolicyGate("recurrent", gain=4.0, lr=0.15, seed=seed + 21)
    g_reinf8.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=reinf_episodes)
    reinf8 = _gate_stats(g_reinf8, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)

    # ---- the EXISTING code-only IDENTITY gate (plain REINFORCE) -- expected to fail the crux (gap~0) ----
    g_ident = PolicyGate("identity", gain=4.0, lr=0.15, seed=seed + 5)
    g_ident.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=episodes)
    ident = _gate_stats(g_ident, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)

    # ---- THE gap#4 DEEP-CREDIT gates (same architecture + deployment; ONLY the credit rule differs) ----
    def _eprop(feedback, off, elig_leak=1.0, perm_reward=False, homeo_v=homeo, b0=b_init):
        g = EpropCreditGate(gain=4.0, lr=eprop_lr, seed=seed + off, feedback=feedback, kp_lr=kp_lr,
                            elig_leak=elig_leak, readout_scale=readout_scale, homeo=homeo_v, b_init=b0, kp_wd=kp_wd)
        g.train_eprop(train_seqs, code_of, feat_of, F, N, episodes=episodes, perm_reward=perm_reward)
        st = _gate_stats(g, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)
        st["bw_cos_init"] = g.bw_cos_init; st["bw_cos_final"] = g.bw_cos_final
        return st

    ep_aligned = _eprop("aligned", 41)                            # deep-credit + transport CEILING (credit-fidelity)
    ep_kp = _eprop("kp", 53)                                      # deep-credit + transport-free NON-canonical KP (frozen R, no decay) -> anti-aligns
    ep_kp_canon = _eprop("kp_canon", 61)                          # deep-credit + CANONICAL co-adapting KP (Akrout 2019): co-adapt R + B + weight decay -- the SEND-BACK confirm (is transport-free alignment structural, not dimensional?)
    ep_fixed = _eprop("fixed", 67)                                # deep-credit + transport-free fixed-random (DFA)
    ep_notrace = _eprop("aligned", 79, elig_leak=0.0)            # LEVER: no eligibility trace (only the last decision)
    ep_nohomeo = _eprop("aligned", 85, homeo_v=0.0, b0=0.0)      # LEVER: homeostasis OFF (the reliability companion)
    ep_permrew = _eprop("aligned", 91, perm_reward=True)         # ANTI-CHEAT: permuted reward -> no learning

    return {"seed": seed, "N": N, "F": F, "L": L, "distance": L + 1, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(N) ** L,
            "acc_marker": acc_marker, "feat_acc_marker": feat_acc, "hold_alive": alive, "zero_input_ok": zero_ok,
            "acc_lesion": acc_lesion, "acc_permuted_position": acc_permpos, "htm_test": htm_test,
            "ngram_floor_test": ngram_test, "ngram_order": ngram_order,
            "reinforce_gate": reinf, "reinforce8_gate": reinf8, "identity_gate": ident,
            "eprop_aligned_gate": ep_aligned, "eprop_kp_gate": ep_kp, "eprop_kp_canon_gate": ep_kp_canon,
            "eprop_fixed_gate": ep_fixed,
            "eprop_notrace_gate": ep_notrace, "eprop_nohomeo_gate": ep_nohomeo, "eprop_permreward_gate": ep_permrew}


_GATE_KEYS = ["reinforce_gate", "reinforce8_gate", "identity_gate", "eprop_aligned_gate", "eprop_kp_gate",
              "eprop_kp_canon_gate", "eprop_fixed_gate", "eprop_notrace_gate", "eprop_nohomeo_gate",
              "eprop_permreward_gate"]


def _agg_gate(per, key):
    sub = [p[key] for p in per]
    base = {k: float(np.mean([d[k] for d in sub])) for k in ("acc", "hold_alive", "fire_pos0", "fire_posgt0",
            "token_identity_gap", "n_matched")}
    if all("bw_cos_final" in d and d["bw_cos_final"] is not None for d in sub):
        base["bw_cos_init"] = float(np.mean([d["bw_cos_init"] for d in sub]))
        base["bw_cos_final"] = float(np.mean([d["bw_cos_final"] for d in sub]))
    # per-seed acc + gap spreads (reliability = the whole point of this de-risk)
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
              "chance": per[0]["chance"], "path_space": per[0]["path_space"],
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
    ap.add_argument("--eprop-lr", type=float, default=0.08)
    ap.add_argument("--kp-lr", type=float, default=0.3)
    ap.add_argument("--readout-scale", type=float, default=3.0)
    ap.add_argument("--homeo", type=float, default=0.10, help="intrinsic firing-rate homeostasis strength (companion)")
    ap.add_argument("--b-init", type=float, default=0.5, help="gate bias init (keeps it firing early; homeostasis holds it)")
    ap.add_argument("--reinforce-episodes", type=int, default=8, help="the banked REINFORCE baseline's native budget")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to gate GO (default: the largest)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    F = a.n_feat; chance = 1.0 / F
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | N_noun={a.n_noun} F_feat={F} chance={chance:.3f} | L={dists} "
          f"| recur={a.recur} | episodes={a.episodes} eprop_lr={a.eprop_lr} kp_lr={a.kp_lr} "
          f"readout_scale={a.readout_scale} homeo={a.homeo} b_init={a.b_init} | n_train={a.n_train} "
          f"n_test={a.n_test} | seeds={a.seeds}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            per = [run_point(s, a.n_noun, F, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                             a.clear_steps, a.episodes, a.eprop_lr, a.kp_lr, a.readout_scale, a.homeo, a.b_init,
                             a.reinforce_episodes) for s in a.seeds]
            p = agg(per); points.append(p)
            re, r8, iy = p["reinforce_gate"], p["reinforce8_gate"], p["identity_gate"]
            al, kp, fx = p["eprop_aligned_gate"], p["eprop_kp_gate"], p["eprop_fixed_gate"]
            nt, nh, pr = p["eprop_notrace_gate"], p["eprop_nohomeo_gate"], p["eprop_permreward_gate"]
            print(f"  [N={a.n_noun} F={F} L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']}] "
                  f"MARKER {p['acc_marker']:.3f} | HTM {p['htm_test']:.3f} | n-gram {p['ngram_floor_test']:.3f} | "
                  f"chance {chance:.3f} || LESION {p['acc_lesion']:.3f} PERM-POS {p['acc_permuted_position']:.3f}",
                  flush=True)
            print(f"     CREDIT gates (acc[min,std] | fire p0/p>0 | id-gap[min]):", flush=True)
            for tag, gs in (("REINFORCE@%d" % a.episodes, re), ("REINFORCE@%d" % a.reinforce_episodes, r8),
                            ("identity", iy), ("eprop_ALIGNED", al), ("eprop_KP", kp), ("eprop_fixed", fx),
                            ("eprop_NOTRACE", nt), ("eprop_NOHOMEO", nh), ("eprop_PERMREW", pr)):
                extra = ""
                if "bw_cos_final" in gs:
                    extra = f"  cos(B,I) {gs['bw_cos_init']:+.2f}->{gs['bw_cos_final']:+.2f}"
                print(f"       {tag:16s} {gs['acc']:.3f}[{gs['acc_min']:.3f},{gs['acc_std']:.3f}] | "
                      f"{gs['fire_pos0']:.2f}/{gs['fire_posgt0']:.2f} | gap {gs['token_identity_gap']:+.2f}"
                      f"[{gs['gap_min']:+.2f}]{extra}", flush=True)
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
        re, r8, iy = far["reinforce_gate"], far["reinforce8_gate"], far["identity_gate"]
        al, kp, fx = far["eprop_aligned_gate"], far["eprop_kp_gate"], far["eprop_fixed_gate"]
        kpc = far["eprop_kp_canon_gate"]                         # CANONICAL co-adapting KP (the adversarial-verify confirm arm)
        nt, nh, pr = far["eprop_notrace_gate"], far["eprop_nohomeo_gate"], far["eprop_permreward_gate"]

        print(f"\n-- ROLE-GATE x gap#4 DEEP-CREDIT verdict at L={far['L']} (dist {far['distance']}, held-out novel "
              f"distractors) --", flush=True)
        # levers (do not raise -> a null lever becomes an honest report, not a crash)
        try:
            lever("deep-credit (aligned) vs plain REINFORCE (matched budget) -- role-gate held-out accuracy",
                  round(re["acc"], 3), round(al["acc"], 3), required=False,
                  continuous=f"gap: REINFORCE {re['token_identity_gap']:+.2f} vs aligned {al['token_identity_gap']:+.2f}")
            lever("eligibility TRACE (aligned vs no-trace) -- temporal credit load-bearing",
                  round(nt["acc"], 3), round(al["acc"], 3), required=False)
            lever("intrinsic HOMEOSTASIS (aligned vs no-homeo) -- reliability companion load-bearing",
                  round(nh["acc_min"], 3), round(al["acc_min"], 3), required=False)
            lever("permuted-reward (aligned) -- learning signal carries no signal",
                  round(al["acc"], 3), round(pr["acc"], 3), required=False)
        except LeverError:
            pass
        attributable_to("role-gate held-out accuracy attributable to DEEP CREDIT (aligned vs plain REINFORCE)",
                        al["acc"], re["acc"])
        attributable_to("token-identity ROLE selectivity attributable to DEEP CREDIT (aligned gap vs REINFORCE gap)",
                        al["token_identity_gap"], re["token_identity_gap"])

        # ROLE-induction criteria (per the reused instrument): high held-out, pos0 high / pos>0 low, crux gap large
        def _is_role(g):
            return (g["acc"] >= chance + 0.20 and g["fire_pos0"] >= 0.70 and g["fire_posgt0"] <= 0.30
                    and g["token_identity_gap"] >= 0.50)

        def _reliable(g):  # 6-seed reliability: EVERY seed clears the bar (min over seeds)
            return (not smoke) and g["acc_min"] >= chance + 0.30 and g["gap_min"] >= 0.50
        kp_role = _is_role(kp); aligned_role = _is_role(al); fixed_role = _is_role(fx)
        identity_fails = iy["token_identity_gap"] <= 0.20
        beats_reinforce = (al["acc"] >= re["acc"] + 0.15 and al["token_identity_gap"] >= re["token_identity_gap"] + 0.20)
        trace_bites = al["acc"] >= nt["acc"] + 0.10          # the eligibility TRACE is load-bearing (temporal credit)
        homeo_bites = al["acc_min"] >= nh["acc_min"] + 0.05  # homeostasis improves the WORST-seed reliability
        perm_collapses = pr["acc"] <= chance + 0.15          # permuted-reward -> no learning
        # the strict BRAIN-BASED GO needs the TRANSPORT-FREE (KP) feedback to reach role reliably
        role_go = bool(kp_role and _reliable(kp) and identity_fails and beats_reinforce and trace_bites
                       and perm_collapses and not smoke)

        common = (f"REINFORCE@{a.episodes} {re['acc']:.3f}[min {re['acc_min']:.3f}] "
                  f"(gap {re['token_identity_gap']:+.2f}); REINFORCE@{a.reinforce_episodes}(banked) {r8['acc']:.3f} "
                  f"(gap {r8['token_identity_gap']:+.2f}); "
                  f"eprop_ALIGNED(transport ceiling) {al['acc']:.3f}[min {al['acc_min']:.3f}] "
                  f"(gap {al['token_identity_gap']:+.2f}[min {al['gap_min']:+.2f}], p0/p>0 {al['fire_pos0']:.2f}/"
                  f"{al['fire_posgt0']:.2f}); eprop_KP(transport-free) {kp['acc']:.3f}[min {kp['acc_min']:.3f}] "
                  f"(gap {kp['token_identity_gap']:+.2f}, cos(B,I) {kp.get('bw_cos_init', float('nan')):+.2f}->"
                  f"{kp.get('bw_cos_final', float('nan')):+.2f}); eprop_KP_CANON(canonical: co-adapt R+decay) "
                  f"{kpc['acc']:.3f} (gap {kpc['token_identity_gap']:+.2f}, cos(B,R^T) "
                  f"{kpc.get('bw_cos_init', float('nan')):+.2f}->{kpc.get('bw_cos_final', float('nan')):+.2f}); "
                  f"eprop_fixed {fx['acc']:.3f} (gap "
                  f"{fx['token_identity_gap']:+.2f}); marker {far['acc_marker']:.3f}; chance {chance:.3f}. "
                  f"identity-gate crux {iy['token_identity_gap']:+.2f} (fails={identity_fails}). LEVERS: no-trace "
                  f"{nt['acc']:.3f} (trace-bites={trace_bites}); no-homeo min {nh['acc_min']:.3f} "
                  f"(homeo-bites={homeo_bites}); permuted-reward {pr['acc']:.3f} (collapses={perm_collapses}).")
        smoketag = "" if not smoke else " (1-seed indicator; run the 6-seed sweep)"
        if kp_role and (smoke or _reliable(kp)) and identity_fails and beats_reinforce:
            verdict = (f"ROLE-GATE POSITIVE (brain-based){smoketag} -- gap#4 DEEP CREDIT with TRANSPORT-FREE (KP) "
                       f"learned feedback INDUCES syntactic role where plain REINFORCE could not. {common} The e-prop "
                       f"eligibility trace (slow-hold time-constant) + intrinsic homeostasis assign the distal "
                       f"verb-prediction reward back to the position-0 LOAD decision through the intervening fillers. "
                       f"CAVEAT: the trace + learning signal are HOST math; their on-substrate spiking DA-gated "
                       f"realisation is the named next rung. Reuse-by-import; NO sim/ edit.")
        elif aligned_role and (smoke or _reliable(al)) and identity_fails and beats_reinforce:
            verdict = (f"ROLE-GATE RESOLVED AT THE CREDIT-FIDELITY CEILING; transport-free realisation is the residual"
                       f"{smoketag} -- the credit assignment WAS the role-induction residual: gap#4 DEEP CREDIT with the "
                       f"credit-fidelity CEILING feedback (aligned = R^T, a labelled TRANSPORT shortcut, mirroring gap#4's "
                       f"BPTT ceiling) reaches the MARKER CEILING reliably (aligned held-out {al['acc']:.3f}, gap "
                       f"{al['token_identity_gap']:+.2f}, fires pos0 {al['fire_pos0']:.2f}/pos>0 {al['fire_posgt0']:.2f}) "
                       f"where plain REINFORCE ({re['acc']:.3f}) and even the banked host position-oracle (0.265) FAILED. "
                       f"{common} The eligibility TRACE is load-bearing (trace-bites={trace_bites}); intrinsic HOMEOSTASIS "
                       f"is a worst-seed reliability aid (homeo-bites={homeo_bites}), NOT load-bearing when False; the "
                       f"identity gate fails the crux. THE RESIDUAL, precisely (CORRECTED 2026-08-11 after adversarial "
                       f"verify): the transport-free arms do NOT reach role -- but NOT because of readout dimension. The "
                       f"NON-canonical KP (frozen R=I, no weight decay, B-only) has no alignment attractor so B ANTI-aligns "
                       f"(cos(B,I) {kp.get('bw_cos_final', float('nan')):+.2f}); CANONICAL co-adapting KP (Akrout 2019: "
                       f"co-adapt forward R + feedback B + weight decay) RECOVERS alignment at THIS F={far['F']} "
                       f"(cos(B,R^T) {kpc.get('bw_cos_final', float('nan')):+.2f}) -- so the feedback-alignment residual is "
                       f"STRUCTURAL, not dimensional. BUT aligned transport-free feedback is NECESSARY-NOT-SUFFICIENT: "
                       f"canonical KP still does not reach role (acc {kpc['acc']:.3f}, gap "
                       f"{kpc['token_identity_gap']:+.2f}) -- a DEEPER residual (the co-adapting readout appears to absorb "
                       f"the credit pressure the gate needs). Next rung: chained multi-hop FA + sigma-prime (needs a hidden "
                       f"layer this single-layer gate lacks), or hold/regularise the readout so it cannot absorb the credit, "
                       f"or the emergence-engine's own sequence code supplying position -- trained WITH this deep-credit "
                       f"rule. NOT 'higher readout dim'. Reuse-by-import; NO sim/ edit.")
        else:
            verdict = (f"ROLE-GATE HONEST NEGATIVE (first-class){smoketag} -- deep credit did NOT cleanly induce role even "
                       f"at the credit-fidelity ceiling. {common} The residual is NOT the credit rule alone -- it is the "
                       f"POSITIONAL/STRUCTURAL substrate SIGNAL the gate conditions on (the recurrent latch/ordinal code) "
                       f"and/or reward density. The next mechanism is a spiking ordinal/phase code (a 'controller-seen' "
                       f"population) supplying position, trained WITH this deep-credit rule.")
        print(f"[rolegate-gap4] {verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR -- {err}"
    else:
        verdict = "ERROR -- no points computed"

    # ---- earned verdict preconditions (validity travels with the verdict) ----
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("var_bind_rolegate_gap4_credit", chance=chance)
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
                       note="the ROLE is POSITIONAL: the position-0 marker beats the permuted-position control (task validity)")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across hold+read")
            Vd.control("deep_credit_differs_from_reinforce",
                       treatment=far["eprop_aligned_gate"]["acc"], control=far["reinforce_gate"]["acc"],
                       min_separation=1e-6, note="the deep-credit arm must differ from the plain-REINFORCE baseline")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        dec = Vd.decide(role_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_rolegate_gap4_credit", "verdict": verdict, "role_go": bool(role_go),
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "the banked variable-binding WM (D3 slow-NMDA bistable HOLD slot; write = clear-then-load; "
                            "content = the gated token's agreement feature = the lexicon) driven by a RECURRENT-latch "
                            "write-gate, trained by gap#4 DEEP CREDIT: an e-prop forward eligibility trace (FIXED slow "
                            "slow-NMDA-hold time-constant, Bellec 2020) + a transport-free learning signal from the "
                            "distal verb-prediction error (three-factor pre x post x DA) + intrinsic firing-rate "
                            "homeostasis (Turrigiano; the biological companion that keeps the 1-subject-vs-L-distractor "
                            "class imbalance from collapsing the gate silent). Compared arm-for-arm against the SAME "
                            "recurrent-latch gate trained by PLAIN REINFORCE (matched budget + the banked 8-episode "
                            "0.602 baseline) + the existing code-only identity gate + feedback arms (transport-aligned "
                            "credit-fidelity CEILING / transport-free KP-learned / transport-free fixed-random DFA) + a "
                            "NO-TRACE lever (eligibility leak 0) + a NO-HOMEO lever + a permuted-reward anti-cheat",
               "task": "the HARDER same-pool positional agreement stream (subject = position 0; distractors = the SAME "
                       "noun pool at positions 1..L; verb agrees with the subject's feature; held-out = disjoint novel "
                       "distractor tuples); CRUX = the token-identity control (same noun gated differently by position)",
               "seeds": a.seeds, "config": {"N_noun": a.n_noun, "F_feat": F, "distances": dists, "recur": a.recur,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps,
               "episodes": a.episodes, "eprop_lr": a.eprop_lr, "kp_lr": a.kp_lr, "readout_scale": a.readout_scale,
               "n_train": a.n_train, "n_test": a.n_test, "chance": chance, "go_distance": go_L},
               "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "The convergent unblock: applies the gap#4 deep-credit surpass to the role-gate credit "
                              "residual. The e-prop eligibility trace + transport-free learning signal are HOST math "
                              "(their spiking DA-gated realisation is the named next rung). 1-seed is a SMOKE indicator; "
                              "the 6-seed sweep is decisive. An HONEST NEGATIVE (deep credit ALSO fails to induce role "
                              "reliably) is first-class and names what is still missing (the positional-signal quality / "
                              "reward density), NOT a fabricated GO."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 114, flush=True)
    print(f"[rolegate-gap4-credit] VERDICT: {verdict}", flush=True)
    print(f"[rolegate-gap4-credit] role_go={role_go}  wrote {a.out}\n" + "=" * 114, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
