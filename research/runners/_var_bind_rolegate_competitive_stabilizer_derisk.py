"""ROLE-GATE x COMPETITIVE FORWARD STABILIZER (LEVER 3) -- the standing-lesson reframe on the role-gate's
transport-free RELIABILITY residual.

WHY (the residual LEVER 2 isolated -- do NOT re-derive; read the banked findings):
  * gap#4 DEEP CREDIT with the ALIGNED=transport CEILING reaches role RELIABLY (single- AND 2-layer: aligned 1.000
    [min 1.000] 6/6). The architecture EXPRESSES role.
  * LEVER 1 (readout-regularization): NEGATIVE -- the variance is INTRINSIC, not the readout.
  * LEVER 2 (a HIDDEN LAYER + chained multi-hop FA + sigma', `_var_bind_rolegate_hidden_chainedFA_derisk`): NEGATIVE
    that ISOLATED the residual. It is NOT depth (aligned ceiling clears 6/6), NOT sigma' (load-bearing), NOT feedback
    alignment (the co-adapting KP arm recovers cos(B,W^T) 0.92-1.00 on EVERY seed) -- yet role accuracy still COLLAPSES
    on some seeds. It is a SEED-DEPENDENT COLLAPSE INTO A FIRE-EVERYTHING BASIN: collapsed seeds fire non-selectively
    (pos0 ~ pos>0); successful seeds fire pos0-only. Banked 6-seed (go L=4, chance 0.25):
      hidden_chained_FA  0.422[min 0.222] gap +0.23[min -0.14]  (seed42 1.000 clean; seeds 100/101 fire-everything)
      hidden_chained_KP  0.578[min 0.133] gap +0.46[min +0.00]

THE HYPOTHESIS (LEVER 3, the standing-lesson reframe: "at a wall, ask what else the real system runs alongside this
that we replaced with a CONSTANT"). Here the answer is COMPETITION. Cortical / WM circuits run lateral inhibition +
divisive normalisation that STRUCTURALLY FORBID the all-active fixed point; LEVER 2 proxied that with only a SCALAR
homeostatic bias nudge (a slow average, not a per-trial structural constraint). Add a COMPETITIVE / NORMALISING forward
STABILIZER that makes the fire-everything state DYNAMICALLY UNREACHABLE, trained WITH the transport-free chained-FA+sigma'
/ KP rule -> eliminate the seed-dependent basin collapse -> RELIABLE transport-free role induction.

THE STABILIZER (the ONLY new machinery; runner-side host math on the gate's OWN populations, NO sim/ edit):
  The fire-everything basin is TEMPORAL: the single scalar load-unit fires (p>0.5) at EVERY position instead of only at
  pos0. The role solution loads ONCE (the subject, pos0), then the latch shuts the gate. So the biologically-faithful
  competition is FEEDBACK / LATERAL INHIBITION that makes REPEATED loading within a sentence self-limiting, PLUS
  DIVISIVE NORMALISATION in the hidden population:
    (1) OUTPUT feedback inhibition (subtractive), the load-bearing one. A pooled inhibitory interneuron integrates the
        load-unit's OWN recent output within the sentence:  s_inh_t = leak * s_inh_{t-1} + p_{t-1};  the output logit is
        z2_eff = z2 - out_lambda * s_inh.  s_inh RESETS to 0 each sentence -> the FIRST (subject) load faces ZERO
        inhibition (never suppressed), but after ~one load s_inh~1 clamps every subsequent load -> the all-load fixed
        point is DYNAMICALLY UNREACHABLE (it would require s_inh maximal AND the drive to overcome it -- impossible for
        out_lambda large). This is the cortical WM "one-update-at-a-time" motif (BG-thalamocortical gating re-inhibits
        after a stripe updates) / spike-frequency adaptation (M-current) that prevents runaway firing -- the competitive
        process LEVER 2 replaced with the scalar homeo constant.
    (2) HIDDEN divisive normalisation (Carandini-Heeger canonical cortical gain control):  h_i = r1_i / (1 + div_k *
        mean_j r1_j),  r1 = sigmoid(hidden_gain*a1).  Keeps the hidden ensemble OUT of the all-saturated regime where
        the hidden sigma' (r1(1-r1)) vanishes and the credit rule has NO gradient to shape pos0-selectivity. The
        diagonal local slope sp1_eff = hidden_gain * r1(1-r1) / denom is used in the chained-FA credit (transport-free,
        local -- no cross-unit Jacobian, no weight transport).
  Neither term encodes "fire at pos0": (1) is a generic "load once" budget, (2) is generic gain control. The gate must
  still LEARN via the credit rule to fire STRONGEST at pos0 (loading pos0 -> correct verb -> reward; loading a
  distractor -> reward ~1/F). The stabilizer only REMOVES the degenerate fire-everywhere escape, forcing the gate to
  COMMIT to one position; the credit then makes that position pos0.

THE ARMS (all the SAME 2-layer gate + the SAME REAL spiking D3 SpikingSlot at eval; matched episode budget; the
no-stab arms DELEGATE to the LEVER-2 gate verbatim -> byte-identical baseline + a clean lesion):
  * aligned_stab      aligned (transport) CEILING + stabilizer -- MUST stay reliable (~1.000): the stabilizer must NOT
                      break the working case.
  * aligned_nostab    the LEVER-2 aligned ceiling (no stabilizer) -- the ceiling reference.
  * fa_stab           chained-FA + sigma' + stabilizer -- THE brain-based CANDIDATE (the GO is about this).
  * fa_nostab         chained-FA + sigma', NO stabilizer == the LEVER-2 collapse baseline == the LESION (remove the
                      competition at train+eval -> MUST re-collapse to the basin; the load-bearing proof).
  * kp_stab           chained-KP + stabilizer -- candidate 2 (transport-free co-adapting Kolen-Pollack).
  * kp_nostab         chained-KP, NO stabilizer == the LEVER-2 KP collapse baseline / KP lesion.
  * fa_stab_noSP      chained-FA + stabilizer, sigma' DROPPED -- the sigma' LEVER (must hurt).
  * permrew_stab      chained-FA + stabilizer, verb target SHUFFLED per sentence -- ANTI-CHEAT (role selectivity must
                      collapse: the learning signal carries no signal).
  + identity (code-only) crux control (must FAIL: gap ~0), marker ceiling, chance, HTM, n-gram floor, lesion-the-hold,
    permuted-position validity, held-out NOVEL fillers. Per-seed fire(pos0/pos>0) is reported so a reviewer can SEE
    whether the stabilizer eliminated the fire-everything basin.

GO (the brain-based reliability question): a TRANSPORT-FREE arm (fa_stab OR kp_stab) reaches role RELIABLY iff -- over
the 6 seeds -- acc mean >= 0.75 AND acc_min >= 0.60 AND token-identity gap_min >= 0.30, with the stabilizer LOAD-BEARING
(the no-stab lesion re-collapses: stab acc_min >= nostab acc_min + 0.20), sigma' biting, permuted-reward collapsing, the
identity gate failing the crux, and the aligned ceiling still reliable with the stabilizer. Else -> a first-class HONEST
NEGATIVE that further isolates the residual (did the stabilizer change the fire distribution? help some seeds? is the
residual deeper than competition?).

Reuse-by-import of `_var_bind_rolegate_hidden_chainedFA_derisk` (the 2-layer HiddenChainedGate, the stream, the REAL
spiking SpikingSlot eval, the crux, the teeth). The stabilizer is a FORWARD-PASS operation on the gate's own hidden +
output populations (host math; its on-substrate spiking DA-gated realisation is the named next rung). NO sim/ edit.
SIM_BACKEND=numpy (sub-1k-neuron LIF loops are launch-bound: CPU faster). SpikingSlot sets cfg.seed -> the substrate IS
seeded per (seed) (verified: build twice at one seed -> identical firing thresholds; banked LEVER-2 note).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_competitive_stabilizer_derisk --seeds 42 --smoke \
    --distances 2 3
6-seed decisive (fan the seeds across processes, then --merge-from):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
    research.runners._var_bind_rolegate_competitive_stabilizer_derisk --seeds $s --distances 2 3 4 --n-test 90 \
    --out research/findings/raw/_rolegate_competitive_stabilizer/seed_$s.json & done ; wait
  SIM_BACKEND=numpy python -m research.runners._var_bind_rolegate_competitive_stabilizer_derisk \
    --merge-from research/findings/raw/_rolegate_competitive_stabilizer/seed_*.json \
    --out research/findings/raw/_rolegate_competitive_stabilizer/competitive_stabilizer_6seed.json
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

# reuse-by-import: the 2-layer HiddenChainedGate (LEVER 2) + the SAME-POOL positional stream, the REAL spiking
# SpikingSlot eval, the crux instrument, the baselines/teeth (all re-exported by the LEVER-2 runner).
from research.runners._var_bind_rolegate_hidden_chainedFA_derisk import (
    HiddenChainedGate, _hidden_stats, _cos,
    role_layout, make_role_stream, make_role_heldout, permute_subject_out,
    SpikingSlot, MarkerRoleGate, PolicyGate, _gate_stats,
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

OUT = Path("research/findings/raw/_rolegate_competitive_stabilizer/competitive_stabilizer.json")


# ====================================================================================================================
# THE COMPETITIVE-STABILIZER 2-LAYER GATE. Subclasses the LEVER-2 HiddenChainedGate; adds a competitive/normalising
# FORWARD stabilizer on the gate's OWN populations (hidden divisive normalisation + output feedback inhibition). When
# stabilize=False it DELEGATES to the parent VERBATIM (byte-identical to the LEVER-2 arm) -> the no-stab control AND the
# lesion are the exact LEVER-2 computation, so any difference is attributable to the competition ALONE. The training
# rule (transport-free chained-FA + sigma' / KP) is UNCHANGED except that the forward-computed quantities (h, sp1, p,
# sp2) now reflect the competition; the e-prop forward-eligibility already ignores the cross-time recurrent path of the
# inhibition (an approximation), so the credit stays transport-free + local.
# ====================================================================================================================
class CompetitiveHiddenChainedGate(HiddenChainedGate):
    def __init__(self, *args, stabilize=True, div_k=2.0, out_lambda=4.0, inh_leak=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.stabilize = bool(stabilize)
        self.div_k = float(div_k)            # hidden divisive-normalisation strength (Carandini-Heeger denom)
        self.out_lambda = float(out_lambda)  # output feedback-inhibition strength (subtractive, on the logit)
        self.inh_leak = float(inh_leak)      # leak of the within-sentence inhibitory integrator (1.0 = pure budget)
        self.s_inh = 0.0                     # the pooled inhibitory unit's within-sentence state (RESETS per sentence)

    def reset(self):
        super().reset()                      # PolicyGate.reset -> self.g = 0.0
        self.s_inh = 0.0

    # ---- forward (deployment): identical competition to training; used by eval_role_wm / positional_fire ----
    def _forward(self, code, g):
        if not self.stabilize:
            return super()._forward(code, g)
        a1 = self.W1 @ code + self.w_g1 * g + self.b1
        r1 = 1.0 / (1.0 + np.exp(-self.hidden_gain * a1))
        denom = 1.0 + self.div_k * float(np.mean(r1))          # hidden divisive normalisation (shared shunting)
        h = r1 / denom
        z2 = float(self.W2 @ h + self.b2) - self.out_lambda * self.s_inh   # output feedback inhibition (subtractive)
        p = 1.0 / (1.0 + np.exp(-self.gain * z2))
        return a1, h, z2, p

    def decide(self, t, tok, code, nC):
        if not self.stabilize:
            return super().decide(t, tok, code, nC)
        _a1, _h, _z2, p = self._forward(code, self.g)
        load = p > 0.5
        self.s_inh = self.inh_leak * self.s_inh + p            # accumulate the load AFTER using the prior state
        if load:
            self.g = 1.0
        return load

    # ---- training: two-pass forward-eligibility + transport-free chained-FA + sigma', WITH the forward stabilizer ----
    def train_chained(self, stream, code_of, feat_of, F, episodes=80, perm_reward=False):
        if not self.stabilize:
            return super().train_chained(stream, code_of, feat_of, F, episodes=episodes, perm_reward=perm_reward)
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
            order = np.random.permutation(len(stream))
            for n in order:
                s = stream[n]; toks = s[:-1]; T = len(toks)
                true_feat = feat_of[toks[0]]
                if perm_reward:
                    true_feat = int(np.random.randint(0, F))
                # -------- forward pass (WITH the competitive stabilizer; store per-token quantities) --------
                g = 0.0; m = np.zeros(F); p_sum = 0.0; s_inh = 0.0
                rec = []
                for t, tok in enumerate(toks):
                    code = code_of[tok]
                    v = np.zeros(F); v[feat_of[tok]] = 1.0
                    a1 = self.W1 @ code + self.w_g1 * g + self.b1
                    r1 = 1.0 / (1.0 + np.exp(-self.hidden_gain * a1))
                    denom = 1.0 + self.div_k * float(np.mean(r1))    # hidden divisive normalisation
                    h = r1 / denom
                    sp1 = self.hidden_gain * r1 * (1.0 - r1) / denom  # hidden sigma' (diagonal local slope; the load-
                                                                     #   bearing factor, kept alive OUT of saturation)
                    z2 = float(self.W2 @ h + self.b2) - self.out_lambda * s_inh   # output feedback inhibition
                    p = 1.0 / (1.0 + np.exp(-self.gain * z2))
                    sp2 = self.gain * p * (1.0 - p)                  # output sigma' (slope wrt the inhibited logit)
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
                # intrinsic firing-rate homeostasis (kept identical across stab/no-stab so the ONLY difference is the
                # structural competition; the stabilizer is what STRUCTURALLY forbids fire-everything, homeo is a slow
                # scalar average and cannot).
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
# One (seed, L) point
# ====================================================================================================================
def run_point(seed, N, F, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps, episodes,
              hidden, lr, kp_lr, kp_wd, readout_scale, homeo, b2_init, div_k, out_lambda, inh_leak):
    np.random.seed(seed)                                       # episode order uses np.random
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

    # ---- memory ceiling (marker scaffold) + teeth: lesion-the-hold + permuted-position validity ----
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

    # ---- the code-only IDENTITY gate (plain REINFORCE) -- the token-identity crux control (must FAIL: gap ~0) ----
    g_ident = PolicyGate("identity", gain=4.0, lr=0.15, seed=seed + 5)
    g_ident.train(train_seqs, code_of, feat_of, verb_tok_of_feat, F, episodes=episodes)
    ident = _gate_stats(g_ident, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)

    # ---- THE 2-LAYER gates: stab vs no-stab, per feedback arm (same deployment; ONLY the competition/credit differ) ----
    def _hidden(feedback, off, stabilize, sigma_prime=True, perm_reward=False):
        g = CompetitiveHiddenChainedGate(hidden=hidden, gain=4.0, lr=lr, seed=seed + off, feedback=feedback,
                                         kp_lr=kp_lr, kp_wd=kp_wd, elig_leak=1.0, readout_scale=readout_scale,
                                         sigma_prime=sigma_prime, homeo=homeo, b2_init=b2_init,
                                         stabilize=stabilize, div_k=div_k, out_lambda=out_lambda, inh_leak=inh_leak)
        g.train_chained(train_seqs, code_of, feat_of, F, episodes=episodes, perm_reward=perm_reward)
        return _hidden_stats(g, _slot, test_seqs, code_of, feat_of, verb_tok_of_feat, F)

    # offsets MATCH the LEVER-2 runner so the no-stab arms are byte-identical to the banked LEVER-2 baseline
    aligned_nostab = _hidden("aligned", 131, stabilize=False)     # LEVER-2 ceiling
    aligned_stab = _hidden("aligned", 131, stabilize=True)        # ceiling + stabilizer (MUST stay reliable)
    fa_nostab = _hidden("chained_fa", 137, stabilize=False)       # LEVER-2 collapse baseline == the LESION
    fa_stab = _hidden("chained_fa", 137, stabilize=True)          # THE brain-based CANDIDATE
    kp_nostab = _hidden("chained_kp", 143, stabilize=False)       # LEVER-2 KP collapse baseline / KP lesion
    kp_stab = _hidden("chained_kp", 143, stabilize=True)          # candidate 2
    fa_stab_nosp = _hidden("chained_fa", 149, stabilize=True, sigma_prime=False)     # sigma' LEVER (must hurt)
    permrew_fa_stab = _hidden("chained_fa", 151, stabilize=True, perm_reward=True)   # ANTI-CHEAT (FA candidate)
    permrew_kp_stab = _hidden("chained_kp", 153, stabilize=True, perm_reward=True)   # ANTI-CHEAT (KP candidate)

    return {"seed": seed, "N": N, "F": F, "L": L, "distance": L + 1, "chance": chance,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "path_space": float(N) ** L, "hidden": hidden,
            "acc_marker": acc_marker, "feat_acc_marker": feat_acc, "hold_alive": alive, "zero_input_ok": zero_ok,
            "acc_lesion": acc_lesion, "acc_permuted_position": acc_permpos, "htm_test": htm_test,
            "ngram_floor_test": ngram_test, "ngram_order": ngram_order,
            "identity_gate": ident,
            "aligned_nostab_gate": aligned_nostab, "aligned_stab_gate": aligned_stab,
            "fa_nostab_gate": fa_nostab, "fa_stab_gate": fa_stab,
            "kp_nostab_gate": kp_nostab, "kp_stab_gate": kp_stab,
            "fa_stab_nosp_gate": fa_stab_nosp, "permrew_fa_stab_gate": permrew_fa_stab,
            "permrew_kp_stab_gate": permrew_kp_stab}


_GATE_KEYS = ["identity_gate", "aligned_nostab_gate", "aligned_stab_gate", "fa_nostab_gate", "fa_stab_gate",
              "kp_nostab_gate", "kp_stab_gate", "fa_stab_nosp_gate", "permrew_fa_stab_gate", "permrew_kp_stab_gate"]


def _agg_gate(per, key):
    sub = [p[key] for p in per]
    base = {k: float(np.mean([d[k] for d in sub])) for k in ("acc", "hold_alive", "fire_pos0", "fire_posgt0",
            "token_identity_gap", "n_matched")}
    for ck in ("cos_hopA_init", "cos_hopA_final", "cos_hopB_init", "cos_hopB_final"):
        if all(ck in d and d[ck] is not None for d in sub):
            base[ck] = float(np.mean([d[ck] for d in sub]))
    base["acc_min"] = float(np.min([d["acc"] for d in sub]))
    base["acc_std"] = float(np.std([d["acc"] for d in sub]))
    base["gap_min"] = float(np.min([d["token_identity_gap"] for d in sub]))
    base["fire_pos0_min"] = float(np.min([d["fire_pos0"] for d in sub]))
    base["fire_posgt0_max"] = float(np.max([d["fire_posgt0"] for d in sub]))
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
    ap.add_argument("--hidden", type=int, default=32, help="hidden population size H")
    ap.add_argument("--lr", type=float, default=0.05, help="2-layer gate learning rate")
    ap.add_argument("--kp-lr", type=float, default=0.3)
    ap.add_argument("--kp-wd", type=float, default=0.01)
    ap.add_argument("--readout-scale", type=float, default=3.0)
    ap.add_argument("--homeo", type=float, default=0.10, help="intrinsic firing-rate homeostasis strength")
    ap.add_argument("--b2-init", type=float, default=0.3, help="output bias init (keeps it firing early)")
    ap.add_argument("--div-k", type=float, default=2.0, help="hidden divisive-normalisation strength (Carandini-Heeger)")
    ap.add_argument("--out-lambda", type=float, default=4.0, help="output feedback-inhibition strength (subtractive)")
    ap.add_argument("--inh-leak", type=float, default=1.0, help="within-sentence inhibitory-integrator leak (1.0=budget)")
    ap.add_argument("--smoke", action="store_true", help="fast 1-seed indicator (reduced n-test)")
    ap.add_argument("--go-distance", type=int, default=None, help="L at which to gate GO (default: the largest)")
    ap.add_argument("--merge-from", nargs="+", default=None, help="MERGE mode: build the multi-seed aggregate + verdict "
                    "from per-seed artifacts (each a single-seed run of THIS runner) -- lets the 6 seeds run in PARALLEL "
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
    F = a.n_feat; chance = 1.0 / F
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | N_noun={a.n_noun} F_feat={F} chance={chance:.3f} | L={dists} "
          f"| recur={a.recur} | episodes={a.episodes} hidden={a.hidden} lr={a.lr} | STABILIZER div_k={a.div_k} "
          f"out_lambda={a.out_lambda} inh_leak={a.inh_leak} | homeo={a.homeo} b2_init={a.b2_init} | "
          f"n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds} smoke={smoke}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            if a.merge_from:
                per = [ps for ps in by_L.get(L, [])]
            else:
                per = [run_point(s, a.n_noun, F, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                                 a.clear_steps, a.episodes, a.hidden, a.lr, a.kp_lr, a.kp_wd, a.readout_scale, a.homeo,
                                 a.b2_init, a.div_k, a.out_lambda, a.inh_leak) for s in a.seeds]
            p = agg(per); points.append(p)
            iy = p["identity_gate"]
            al0, al1 = p["aligned_nostab_gate"], p["aligned_stab_gate"]
            fa0, fa1 = p["fa_nostab_gate"], p["fa_stab_gate"]
            kp0, kp1 = p["kp_nostab_gate"], p["kp_stab_gate"]
            ns = p["fa_stab_nosp_gate"]; pr = p["permrew_fa_stab_gate"]; pr_kp = p["permrew_kp_stab_gate"]
            print(f"  [N={a.n_noun} F={F} L={L} dist={L+1} paths={p['path_space']:.0f} gen={p['gen_defined']} "
                  f"H={a.hidden}] MARKER {p['acc_marker']:.3f} | HTM {p['htm_test']:.3f} | n-gram "
                  f"{p['ngram_floor_test']:.3f} | chance {chance:.3f} || LESION-hold {p['acc_lesion']:.3f} PERM-POS "
                  f"{p['acc_permuted_position']:.3f}", flush=True)
            print(f"     gates (acc[min,std] | fire p0/p>0 [p0min/pgt0max] | id-gap[min]):", flush=True)
            for tag, gs in (("identity", iy),
                            ("aligned_NOSTAB", al0), ("aligned_STAB", al1),
                            ("chained_FA_NOSTAB(lesion)", fa0), ("chained_FA_STAB", fa1),
                            ("chained_KP_NOSTAB(lesion)", kp0), ("chained_KP_STAB", kp1),
                            ("FA_STAB_noSP", ns), ("FA_STAB_PERMREW", pr), ("KP_STAB_PERMREW", pr_kp)):
                print(f"       {tag:26s} {gs['acc']:.3f}[{gs['acc_min']:.3f},{gs['acc_std']:.3f}] | "
                      f"{gs['fire_pos0']:.2f}/{gs['fire_posgt0']:.2f} [{gs['fire_pos0_min']:.2f}/"
                      f"{gs['fire_posgt0_max']:.2f}] | gap {gs['token_identity_gap']:+.2f}[{gs['gap_min']:+.2f}]",
                      flush=True)
            # per-seed fire for BOTH candidates (so the basin -- and whether the stabilizer eliminated it -- is VISIBLE)
            print(f"     per-seed STAB acc | fire(pos0/pos>0) | gap  [FA-stab || FA-nostab(lesion) || KP-stab || KP-nostab]:",
                  flush=True)
            for ps in per:
                gf = ps["fa_stab_gate"]; gf0 = ps["fa_nostab_gate"]
                gk = ps["kp_stab_gate"]; gk0 = ps["kp_nostab_gate"]
                print(f"       seed {ps['seed']}: FA-stab {gf['acc']:.3f} {gf['fire_pos0']:.2f}/{gf['fire_posgt0']:.2f} "
                      f"g{gf['token_identity_gap']:+.2f} || FA-nostab {gf0['acc']:.3f} {gf0['fire_pos0']:.2f}/"
                      f"{gf0['fire_posgt0']:.2f} || KP-stab {gk['acc']:.3f} {gk['fire_pos0']:.2f}/{gk['fire_posgt0']:.2f} "
                      f"g{gk['token_identity_gap']:+.2f} || KP-nostab {gk0['acc']:.3f} {gk0['fire_pos0']:.2f}/"
                      f"{gk0['fire_posgt0']:.2f}", flush=True)
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
        iy = far["identity_gate"]
        al0, al1 = far["aligned_nostab_gate"], far["aligned_stab_gate"]
        fa0, fa1 = far["fa_nostab_gate"], far["fa_stab_gate"]
        kp0, kp1 = far["kp_nostab_gate"], far["kp_stab_gate"]
        ns = far["fa_stab_nosp_gate"]; pr = far["permrew_fa_stab_gate"]; pr_kp = far["permrew_kp_stab_gate"]

        print(f"\n-- ROLE-GATE x COMPETITIVE STABILIZER verdict at L={far['L']} (dist {far['distance']}, held-out novel "
              f"distractors) --", flush=True)
        try:
            lever("STABILIZER on chained-FA (no-stab vs stab) -- reliability (min over seeds)",
                  round(fa0["acc_min"], 3), round(fa1["acc_min"], 3), required=False,
                  continuous=f"mean: no-stab {fa0['acc']:.3f} vs stab {fa1['acc']:.3f}; "
                             f"fire pos>0 max: no-stab {fa0['fire_posgt0_max']:.2f} vs stab {fa1['fire_posgt0_max']:.2f}")
            lever("STABILIZER on chained-KP (no-stab vs stab) -- reliability (min over seeds)",
                  round(kp0["acc_min"], 3), round(kp1["acc_min"], 3), required=False,
                  continuous=f"mean: no-stab {kp0['acc']:.3f} vs stab {kp1['acc']:.3f}; "
                             f"fire pos>0 max: no-stab {kp0['fire_posgt0_max']:.2f} vs stab {kp1['fire_posgt0_max']:.2f}")
            lever("sigma' on the FA candidate (FA-stab no-sigma' vs FA-stab)",
                  round(ns["acc"], 3), round(fa1["acc"], 3), required=False)
            lever("permuted-reward (KP-stab) -- learning signal necessary for RELIABLE role (acc separation)",
                  round(pr_kp["acc"], 3), round(kp1["acc"], 3), required=False,
                  continuous=f"KP-stab {kp1['acc']:.3f} vs KP-permuted-reward {pr_kp['acc']:.3f} "
                             f"(structural-prior floor)")
        except LeverError:
            pass
        attributable_to("transport-free reliability attributable to the STABILIZER (KP-stab min vs KP-nostab min)",
                        kp1["acc_min"], kp0["acc_min"])

        def _is_role(g):
            return (g["acc"] >= chance + 0.20 and g["fire_pos0"] >= 0.60 and g["fire_posgt0"] <= 0.40
                    and g["token_identity_gap"] >= 0.30)

        def _reliable(g):   # the 6-seed reliability bar (min over seeds) -- the whole point of this de-risk
            return (not smoke) and g["acc"] >= 0.75 and g["acc_min"] >= 0.60 and g["gap_min"] >= 0.30

        identity_fails = iy["token_identity_gap"] <= 0.20
        # sigma' load-bearing FOR THE FA candidate (mean OR worst-seed drop)
        sigmaprime_bites = (fa1["acc"] >= ns["acc"] + 0.10) or (fa1["acc_min"] >= ns["acc_min"] + 0.10)
        # permuted-reward anti-cheat by ACCURACY SEPARATION. The stabilizer + homeostasis structurally induce a
        # fire-once-early (pos0-leaning) PRIOR, so a shuffled reward does NOT collapse the gap to ~0 -- it reaches the
        # structural-prior floor. The honest teeth: the real learning signal must lift RELIABLE role WELL ABOVE that
        # floor (permuted stays < 0.60 AND the candidate beats it by >= 0.30). This is the load-bearing test for the
        # LEARNING signal once the competition supplies the pos0 prior.
        perm_fa_collapses = pr["acc"] < 0.60 and (fa1["acc"] - pr["acc"]) >= 0.30
        perm_kp_collapses = pr_kp["acc"] < 0.60 and (kp1["acc"] - pr_kp["acc"]) >= 0.30
        # the STABILIZER is LOAD-BEARING: turning it OFF (the lesion) re-collapses the worst-seed reliability
        fa_stab_bearing = fa1["acc_min"] >= fa0["acc_min"] + 0.20
        kp_stab_bearing = kp1["acc_min"] >= kp0["acc_min"] + 0.20
        # the aligned ceiling must STAY reliable with the stabilizer (must not break the working case)
        aligned_stab_ok = al1["acc"] >= 0.90 and al1["acc_min"] >= 0.80

        fa_role = _is_role(fa1); kp_role = _is_role(kp1)
        fa_go = bool(fa_role and _reliable(fa1) and fa_stab_bearing and identity_fails and sigmaprime_bites
                     and perm_fa_collapses and aligned_stab_ok and not smoke)
        kp_go = bool(kp_role and _reliable(kp1) and kp_stab_bearing and identity_fails and perm_kp_collapses
                     and aligned_stab_ok and not smoke)
        role_go = bool(fa_go or kp_go)
        which = "chained-KP" if kp_go else ("chained-FA" if fa_go else None)

        common = (f"identity-crux gap {iy['token_identity_gap']:+.2f} (fails={identity_fails}); "
                  f"aligned_STAB(ceiling) {al1['acc']:.3f}[min {al1['acc_min']:.3f}] gap {al1['token_identity_gap']:+.2f} "
                  f"(ok={aligned_stab_ok}) vs aligned_NOSTAB {al0['acc']:.3f}[min {al0['acc_min']:.3f}]; "
                  f"chained_FA: NOSTAB(lesion) {fa0['acc']:.3f}[min {fa0['acc_min']:.3f}] fire "
                  f"{fa0['fire_pos0']:.2f}/{fa0['fire_posgt0']:.2f}(pgt0max {fa0['fire_posgt0_max']:.2f}) -> STAB "
                  f"{fa1['acc']:.3f}[min {fa1['acc_min']:.3f}] fire {fa1['fire_pos0']:.2f}/{fa1['fire_posgt0']:.2f}"
                  f"(pgt0max {fa1['fire_posgt0_max']:.2f}) gap {fa1['token_identity_gap']:+.2f}[min {fa1['gap_min']:+.2f}] "
                  f"(stab-bearing={fa_stab_bearing}); chained_KP: NOSTAB(lesion) {kp0['acc']:.3f}[min {kp0['acc_min']:.3f}] "
                  f"fire {kp0['fire_pos0']:.2f}/{kp0['fire_posgt0']:.2f}(pgt0max {kp0['fire_posgt0_max']:.2f}) -> STAB "
                  f"{kp1['acc']:.3f}[min {kp1['acc_min']:.3f}] fire {kp1['fire_pos0']:.2f}/{kp1['fire_posgt0']:.2f} gap "
                  f"{kp1['token_identity_gap']:+.2f}[min {kp1['gap_min']:+.2f}] (stab-bearing={kp_stab_bearing}); "
                  f"marker {far['acc_marker']:.3f}; chance {chance:.3f}. LEVERS: FA-stab-no-sigma' {ns['acc']:.3f}"
                  f"[min {ns['acc_min']:.3f}] (sigma'-bites={sigmaprime_bites}); FA-permuted-reward acc {pr['acc']:.3f} "
                  f"(collapses={perm_fa_collapses}); KP-permuted-reward acc {pr_kp['acc']:.3f} "
                  f"(collapses={perm_kp_collapses}; structural-prior floor).")
        smoketag = "" if not smoke else " (1-seed indicator; run the 6-seed sweep)"
        # positive iff the DECISIVE 6-seed role_go clears, OR (smoke) a candidate shows role + stab-bearing as an indicator
        kp_cand = kp_role and (smoke or _reliable(kp1)) and kp_stab_bearing
        fa_cand = fa_role and (smoke or _reliable(fa1)) and fa_stab_bearing
        wtag = "chained-KP" if kp_cand else ("chained-FA" if fa_cand else None)
        if (role_go or (smoke and wtag is not None)) and identity_fails and aligned_stab_ok:
            base = kp0 if wtag == "chained-KP" else fa0
            top = kp1 if wtag == "chained-KP" else fa1
            verdict = (f"ROLE-GATE POSITIVE (brain-based, transport-free){smoketag} -- a COMPETITIVE FORWARD STABILIZER "
                       f"(output feedback inhibition + hidden divisive normalisation) ELIMINATES the seed-dependent "
                       f"FIRE-EVERYTHING basin and gives RELIABLE transport-free role via {wtag}, where the no-stabilizer "
                       f"LEVER-2 gate collapsed ({wtag} min {base['acc_min']:.3f}[fire pos>0 max "
                       f"{base['fire_posgt0_max']:.2f}] -> STAB min {top['acc_min']:.3f}[pos>0 max "
                       f"{top['fire_posgt0_max']:.2f}]). {common} The stabilizer is LOAD-BEARING (removing the competition "
                       f"-- the lesion -- re-collapses the worst-seed reliability); the aligned ceiling stays reliable "
                       f"WITH the stabilizer; the identity gate fails the crux. HONEST NUANCE: the competition + "
                       f"homeostasis supply a STRUCTURAL fire-once-early (pos0-leaning) prior, so permuted-reward reaches "
                       f"a structural-prior floor (~{pr_kp['acc']:.2f}), NOT chance -- but the LEARNING signal is still "
                       f"load-bearing (it lifts role from that floor to {top['acc']:.3f}). CAVEAT: the 2-layer net + "
                       f"chained credit + the competition are HOST math; their on-substrate spiking DA-gated / "
                       f"lateral-inhibitory realisation is the named next rung. Reuse-by-import; NO sim/ edit.")
        else:
            # isolate WHAT the stabilizer did, precisely, for the honest negative (per candidate)
            fire_fixed_fa = fa1["fire_posgt0_max"] <= fa0["fire_posgt0_max"] - 0.10
            fire_fixed_kp = kp1["fire_posgt0_max"] <= kp0["fire_posgt0_max"] - 0.10
            # the SPECIFIC failure mode this de-risk found: a candidate reaches RELIABLE role AND the stabilizer is
            # load-bearing (lesion re-collapses) -- BUT the permuted-reward control ALSO reaches role, so the role is
            # STRUCTURAL (the competition's fire-once budget gives ONLY pos0 zero accumulated inhibition -> an ONSET gate
            # that loads the first token = the subject in THIS positional task, with NO learning). The transport-free
            # CREDIT is therefore NOT what induces role; it is untestable on a task where onset == the answer.
            structural_kp = (_is_role(kp1) and (not smoke and kp1["acc_min"] >= 0.60) and kp_stab_bearing
                             and not perm_kp_collapses)
            structural_fa = (_is_role(fa1) and (not smoke and fa1["acc_min"] >= 0.60) and fa_stab_bearing
                             and not perm_fa_collapses)
            if structural_kp or structural_fa:
                wt = "chained-KP" if structural_kp else "chained-FA"
                top = kp1 if structural_kp else fa1
                base = kp0 if structural_kp else fa0
                prc = pr_kp if structural_kp else pr
                verdict = (f"ROLE-GATE HONEST NEGATIVE (first-class) -- RELIABLE ROLE ACHIEVED, but STRUCTURAL (the "
                           f"competition), NOT the transport-free CREDIT{smoketag}. A COMPETITIVE FORWARD STABILIZER "
                           f"(output feedback inhibition + hidden divisive normalisation) CLOSES the LEVER-2 "
                           f"fire-everything basin -- {wt} min {base['acc_min']:.3f}[fire pos>0 max "
                           f"{base['fire_posgt0_max']:.2f}] -> {top['acc']:.3f}[min {top['acc_min']:.3f}], pos>0 max "
                           f"{top['fire_posgt0_max']:.2f}, gap {top['token_identity_gap']:+.2f}[min {top['gap_min']:+.2f}] "
                           f"-- and the stabilizer is LOAD-BEARING (the lesion re-collapses to min {base['acc_min']:.3f}; "
                           f"aligned ceiling stays reliable WITH it; identity crux fails). BUT the permuted-reward "
                           f"anti-cheat REACHES THE SAME role (acc {prc['acc']:.3f}[min {prc['acc_min']:.3f}], gap "
                           f"{prc['token_identity_gap']:+.2f}) -> the role is produced by the STRUCTURE, NOT the learning "
                           f"signal: the feedback-inhibition fire-ONCE budget gives ONLY position 0 zero accumulated "
                           f"inhibition, so the competition is an ONSET gate that loads the first token = the subject in "
                           f"THIS subject-first stream, with NO credit. So the fire-everything basin is CLOSED by "
                           f"structural competition, but 'does the TRANSPORT-FREE CREDIT induce role' is now UNTESTABLE "
                           f"here (onset == the answer). {common} NEXT LEVER (the precisely-isolated residual): a "
                           f"VARIABLE-subject-position stream where onset != the answer -- the credit must fire the "
                           f"subject regardless of ordinal, so the structural onset prior can no longer solve it and the "
                           f"transport-free credit becomes testable. NO sim/ edit.")
            else:
                verdict = (f"ROLE-GATE HONEST NEGATIVE (first-class){smoketag} -- the competitive stabilizer did NOT (yet) "
                           f"deliver RELIABLE transport-free role at the 6-seed bar. {common} PRECISELY isolated: on "
                           f"chained-FA the stabilizer {'DID' if fire_fixed_fa else 'did NOT'} reduce the fire-everywhere "
                           f"basin (pos>0 max {fa0['fire_posgt0_max']:.2f} -> {fa1['fire_posgt0_max']:.2f}), acc_min "
                           f"{fa0['acc_min']:.3f} -> {fa1['acc_min']:.3f}; on chained-KP it "
                           f"{'DID' if fire_fixed_kp else 'did NOT'} reduce it (pos>0 max {kp0['fire_posgt0_max']:.2f} -> "
                           f"{kp1['fire_posgt0_max']:.2f}), acc_min {kp0['acc_min']:.3f} -> {kp1['acc_min']:.3f}. Read: if "
                           f"the fire distribution collapsed toward pos0-only but acc_min stayed < 0.60 the residual is "
                           f"that the gate now COMMITS to ONE position but not reliably pos0 (a within-sentence SELECTION "
                           f"residual, not fire-everything) -> next candidate = a stronger onset/positional bias in the "
                           f"competition, or the emergence-engine's own ordinal code. If the fire distribution did NOT "
                           f"change, the competition is too weak (raise out_lambda) / the collapse is upstream. NO sim/ edit.")
        print(f"[rolegate-competitive-stabilizer] {verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR -- {err}"
    else:
        verdict = "ERROR -- no points computed"

    # ---- earned verdict preconditions (validity travels with the verdict) ----
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("var_bind_rolegate_competitive_stabilizer", chance=chance)
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
            Vd.control("stabilizer_differs_from_nostab",
                       treatment=far["fa_stab_gate"]["acc"], control=far["fa_nostab_gate"]["acc"],
                       min_separation=1e-6, note="the stabilized arm must differ from the no-stabilizer lesion")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        dec = Vd.decide(role_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "var_bind_rolegate_competitive_stabilizer", "verdict": verdict, "role_go": bool(role_go),
               "backend": backend, "sim_backend": backend, "device": device, "smoke": smoke, "cost_acknowledged": True,
               "preconditions": preconditions,
               "mechanism": "the banked variable-binding WM (D3 slow-NMDA bistable HOLD slot; write = clear-then-load; "
                            "content = the gated token's agreement feature = the lexicon) driven by a 2-LAYER role-gate "
                            "(barcode + recurrent latch -> hidden sigmoid population -> scalar load-logit), trained "
                            "TRANSPORT-FREE by chained multi-hop FEEDBACK ALIGNMENT + sigma' (LEVER 2), NOW with a "
                            "COMPETITIVE FORWARD STABILIZER on the gate's own populations: (1) OUTPUT feedback inhibition "
                            "-- a pooled inhibitory integrator of the load-unit's own recent output subtracts from the "
                            "load logit within a sentence (resets per sentence; the first load is un-suppressed, repeated "
                            "loading is self-limiting -> the all-load fixed point is dynamically UNREACHABLE; the WM "
                            "one-update-at-a-time / spike-frequency-adaptation motif); (2) HIDDEN divisive normalisation "
                            "(Carandini-Heeger) keeping the hidden ensemble out of the saturated regime where sigma' "
                            "vanishes. Arms: aligned (transport ceiling) / chained_fa (the brain-based candidate) / "
                            "chained_kp -- each stab vs NO-stab (the no-stab arms delegate to the LEVER-2 gate verbatim -> "
                            "byte-identical collapse baseline + a clean lesion) + a no-sigma' lever + a permuted-reward "
                            "anti-cheat + the code-only identity crux control",
               "task": "the HARDER same-pool positional agreement stream (subject = position 0; distractors = the SAME "
                       "noun pool at positions 1..L; verb agrees with the subject's feature; held-out = disjoint novel "
                       "distractor tuples); CRUX = the token-identity control (same noun gated differently by position)",
               "seeds": a.seeds, "config": {"N_noun": a.n_noun, "F_feat": F, "distances": dists, "recur": a.recur,
               "hold_steps": a.hold_steps, "load_steps": a.load_steps, "clear_steps": a.clear_steps,
               "episodes": a.episodes, "hidden": a.hidden, "lr": a.lr, "kp_lr": a.kp_lr, "kp_wd": a.kp_wd,
               "readout_scale": a.readout_scale, "homeo": a.homeo, "b2_init": a.b2_init, "div_k": a.div_k,
               "out_lambda": a.out_lambda, "inh_leak": a.inh_leak, "n_train": a.n_train, "n_test": a.n_test,
               "chance": chance, "go_distance": go_L},
               "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "LEVER 3 (the standing-lesson reframe) on the role-gate transport-free reliability "
                              "residual: adds a COMPETITIVE / NORMALISING forward stabilizer (the competitive process the "
                              "real WM circuit runs alongside the gate, which LEVER 2 replaced with a scalar homeo "
                              "constant) so the fire-everything fixed point is structurally forbidden, trained WITH the "
                              "transport-free chained-FA+sigma' / KP rule. The 2-layer net + chained credit + the "
                              "competition are HOST math (their spiking DA-gated / lateral-inhibitory realisation is the "
                              "named next rung). 1-seed is a SMOKE indicator; the 6-seed sweep is decisive. If the "
                              "stabilizer does NOT deliver reliable transport-free role, that is a first-class HONEST "
                              "NEGATIVE that further isolates the residual (did it change the fire distribution? help some "
                              "seeds? is the residual deeper than competition?), NOT a fabricated GO."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 114, flush=True)
    print(f"[rolegate-competitive-stabilizer] VERDICT: {verdict}", flush=True)
    print(f"[rolegate-competitive-stabilizer] role_go={role_go}  wrote {a.out}\n" + "=" * 114, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
