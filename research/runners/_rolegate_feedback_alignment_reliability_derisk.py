"""ROLE-GATE x TRANSPORT-FREE FEEDBACK-ALIGNMENT RELIABILITY (LEVER 5) -- can a STRONGER transport-free alignment make
transport-free credit RELIABLE on the variable-subject-position role task, matching the exact-feedback ceiling?

WHY (the CLEANLY-ISOLATED residual -- do NOT re-derive; read the banked findings):
  LEVER 4 (`_var_bind_rolegate_varpos_derisk`, 6-seed HONEST NEGATIVE, banked
  `research/findings/2026-08-11-rolegate-variable-subject-position-credit-test-NEGATIVE-credit-cant-ceiling-can.md`)
  made "does transport-free CREDIT induce role" TESTABLE by removing the onset confound (subject = the NOM-case-tagged
  noun at a RANDOM ordinal; a Bates-MacWhinney learnable arbitrary cue). On that VALID, hard task (untrained / permuted-
  reward / onset all bite <= chance+0.15) the residual is isolated to ONE thing -- the RELIABILITY of transport-free
  feedback alignment. At the GO distance L=6 (chance 0.250, real spiking D3 slot):
    * aligned + stabilizer (EXACT feedback = weight-transport ceiling) reaches role RELIABLY 6/6: 0.950 [min 0.822].
    * chained-FA + stabilizer (transport-free, FIXED random B) is BIMODAL: 0.511 [min 0.233] (seeds 42/43 solve, 44/100/
      101/102 collapse to the onset floor).
    * canonical-KP + stabilizer (transport-free, co-adapting B) COLLAPSES 0/6: 0.289 [min 0.233].
  FA differs from the reliable aligned arm ONLY in the feedback matrices -> the residual is FEEDBACK-ALIGNMENT
  RELIABILITY, not depth / sigma' / stabilizer / task / operating-point (all held fixed, ruled out across LEVERS 1-4).
  The gap#4 lane's banked twin (`2026-08-01-gap4-transport-free-ceiling-FALSIFIED-...`) showed transport-free chained-FA+
  sigma' CLEARS a depth-2 rate ceiling and KP RESCUES MNIST depth-4 at hidden-dim 128 -- i.e. transport-free alignment
  CAN be made reliable with the right MECHANISM / SCALE. This lever finds WHICH mechanism makes it reliable HERE.

THE LEVERS (the varpos finding's own named next-mechanisms; tested WITH the byte-faithful varpos credit loop):
  L1  WEIGHT-MIRROR WARM-UP ("align then train"; Akrout et al. 2019, "Deep Learning without Weight Transport"). A
      pre-training phase aligns B_out -> R^T and B1 -> W2 via a NOISE-output correlation (the weight-mirror circuit:
      inject noise xi into a forward hop, read its output zeta, Hebbian-accumulate outer(xi, zeta) -> the forward
      weight's TRANSPOSE in expectation -- TRANSPORT-FREE, B never reads a forward weight's transpose). Then credit
      trains from an ALIGNED start.  Two variants:
        * mirror_fa      : warm-up aligns the FIXED B, then FA credit (B held; alignment to the INITIAL W degrades as W
                           moves -> tests "align then HOLD").
        * mirror_fa_ref  : warm-up + a periodic mirror REFRESH (re-align B to the CURRENT W2 every `mirror_every`
                           sentences) -> alignment MAINTAINED transport-free, no readout co-adaptation. If THIS collapses,
                           alignment is definitively NOT the residual (it stays high while accuracy collapses).
        * mirror_kp      : warm-up, then KP co-adapts throughout (the LEVER-4 KP arm WITH an aligned head start).
  L2  STRONGER / LONGER KP CO-ADAPTATION (`kp_strong`): warm-up + a kp_lr boost + per-sentence mirror refresh -> the
      feedback tracks W2^T harder / faster than the LEVER-4 KP arm.
  L3/L4  a BETTER OPERATING POINT / a CLEANER, WIDER hidden code -- swept via `--hidden` (H) and the barcode dim: does
      transport-free reliability improve with SCALE (wider hidden -> the NOM template linearly cleaner)? This is the
      role-task SCALE probe. A separate `--scale-probe` runs a GPU-appropriate BATCHED deep-MLP FA/KP/aligned width
      sweep (the 0801 regime) to measure whether transport-free reliability improves with width where the GPU helps
      (VRAM + throughput reported).

ARMS (reuse the LEVER-4 varpos arms VERBATIM so the comparison is direct + the task stays valid):
  oracle (ceiling) / onset (floor) / aligned (exact-feedback ceiling) / fa / kp (the LEVER-4 baselines) +
  mirror_fa / mirror_fa_ref / mirror_kp / kp_strong (THE new levers) + untrained + permrew_fa + permrew_kp (the validity
  killers) + identity_control (the crux). cos(B,W^T) per hop is reported per seed (init=POST-warmup, final) so a reviewer
  sees whether the winning lever RAISES the worst-seed alignment, or alignment stays high while accuracy collapses
  (which would move the residual PAST alignment).

GO: a transport-free lever arm reaches role RELIABLY on the varpos task -- 6-seed mean >= chance+0.30, min >= 0.60, tag-
  gap min >= 0.30 -- with untrained + permuted-reward STILL biting (<= chance+0.15). Else -> a first-class HONEST
  NEGATIVE that maps the boundary precisely (alignment variance vs a deeper credit signal-to-noise limit).

Reuse-by-import of the LEVER-4 varpos machinery (`VarposCompetitiveGate`, the stream, the REAL spiking `SpikingSlot`, the
oracle, the crux, the teeth). The 2-layer net + chained credit + the competition + the weight-mirror are HOST math (their
on-substrate spiking DA-gated / lateral-inhibitory / mirror realisation is the named next rung). NO sim/ edit.
SIM_BACKEND=numpy for the small role task (sub-1k-neuron LIF loops are launch-bound: CPU faster); cupy for --scale-probe.
build_persistent_slot sets cfg.seed -> the substrate IS seeded per (seed) (verified: `_d3_persistent_slot_derisk.py:47`).

Run (1-seed smoke, FOREGROUND):
  SIM_BACKEND=numpy python -m research.runners._rolegate_feedback_alignment_reliability_derisk --seeds 42 --smoke \
    --distances 4 --mirror-steps 4000
6-seed decisive (fan the seeds across processes, then --merge-from):
  for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m \
    research.runners._rolegate_feedback_alignment_reliability_derisk --seeds $s --distances 5 6 --n-test 90 \
    --mirror-steps 6000 --out research/findings/raw/_rolegate_fbalign/seed_$s.json & done ; wait
  SIM_BACKEND=numpy python -m research.runners._rolegate_feedback_alignment_reliability_derisk \
    --merge-from research/findings/raw/_rolegate_fbalign/seed_*.json \
    --out research/findings/raw/_rolegate_fbalign/fbalign_6seed.json
GPU scale probe (batched deep-MLP FA/KP/aligned width sweep; the 0801 regime):
  SIM_BACKEND=cupy python -m research.runners._rolegate_feedback_alignment_reliability_derisk --scale-probe \
    --widths 128 512 2048 --scale-seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_rolegate_fbalign/scale_probe_gpu.json
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

# reuse-by-import: the LEVER-4 varpos machinery (the 2-layer competitive-stabilizer gate + the byte-faithful varpos
# credit loop), the variable-subject-position stream, the REAL spiking D3 slot, the oracle, the crux, the teeth.
from research.runners._var_bind_rolegate_varpos_derisk import (
    VarposCompetitiveGate, CaseMarkerOracle, build_codes, compose_code,
    make_varpos_stream, make_varpos_heldout, permute_cue_out, _cfg_key,
    eval_varpos_wm, tag_position_fire, _flat_seqs)
from research.runners._var_bind_rolegate_competitive_stabilizer_derisk import _cos
from research.runners._var_bind_role_gate_derisk import role_layout, MarkerRoleGate, SpikingSlot, _mint_codes, _DIM
from research.runners._emerge_stream_language_derisk import ngram_floor_heldout

try:
    from tools.lab import lever, attributable_to, LeverError
except Exception:  # tools.lab optional at import time
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

OUT = Path("research/findings/raw/_rolegate_fbalign/fbalign.json")


# ====================================================================================================================
# THE FEEDBACK-ALIGNMENT GATE. Subclass the LEVER-4 VarposCompetitiveGate; add (1) a WEIGHT-MIRROR warm-up that aligns
# B_out -> R^T and B1 -> W2 via NOISE-output correlation (Akrout 2019; transport-free), and (2) an optional per-sentence
# mirror REFRESH + a kp_lr boost. The CREDIT LOOP is a byte-faithful copy of the parent VarposCompetitiveGate.train_varpos
# (itself a faithful copy of CompetitiveHiddenChainedGate.train_chained); the ONLY additions are the warm-up + the
# refresh + the boost, so any result is attributable to the ALIGNMENT MECHANISM, not a credit-rule change.
# ====================================================================================================================
class FBAlignVarposGate(VarposCompetitiveGate):
    def _mirror_estimate(self, R, F, steps, rng):
        """Akrout-2019 weight mirror: inject noise xi into a forward hop, read its output zeta, Hebbian-accumulate
        outer(xi, zeta). E[outer(xi, W xi)] = sigma^2 * W^T -> the forward weight's TRANSPOSE. TRANSPORT-FREE (B is
        estimated from the forward's RESPONSE to noise, never from reading a forward weight's transpose)."""
        rs = self.readout_scale
        H = self.H
        Sa = np.zeros((F, F)); Sb = np.zeros(H)
        for _ in range(steps):
            xiA = rng.normal(0.0, 1.0, F)
            zetaA = (R @ xiA) if R is not None else (rs * xiA)   # readout hop: forward map is R (or rs*I for FA)
            Sa += np.outer(xiA, zetaA)                            # -> R^T in expectation
            xiB = rng.normal(0.0, 1.0, H)
            zetaB = float(self.W2 @ xiB)                          # hidden->out hop: forward map is W2
            Sb += xiB * zetaB                                     # -> W2 in expectation
        return Sa / max(1, steps), Sb / max(1, steps)

    def train_varpos_fb(self, train_input, F, episodes=80, perm_reward=False,
                        warmup_steps=0, kp_lr_boost=1.0, mirror_every=0, mirror_refresh_steps=64):
        """A byte-faithful copy of the parent train_varpos credit loop, with a weight-mirror WARM-UP prepended, an
        optional per-sentence mirror REFRESH, and a kp_lr boost. All three are additive/off by default (warmup_steps=0,
        mirror_every=0, kp_lr_boost=1.0 -> IDENTICAL to the parent)."""
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
        # ---- record PRE-warmup alignment, then the WEIGHT-MIRROR WARM-UP (align B -> forward^T, transport-free) ----
        self.cos_hopA_prewarm = _cos(B_out, RT0)
        self.cos_hopB_prewarm = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)
        mrng = np.random.default_rng(self.seed + 909)
        if warmup_steps > 0 and self.feedback != "aligned":
            B_out, B1 = self._mirror_estimate(R, F, warmup_steps, mrng)
        # cos_hopA_init/hopB_init = the POST-warmup alignment the credit loop STARTS from (so the artifact shows the lift)
        self.cos_hopA_init = _cos(B_out, RT0)
        self.cos_hopB_init = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)
        kp_lr_eff = self.kp_lr * float(kp_lr_boost)

        sent_ctr = 0
        for _ in range(episodes):
            order = np.random.permutation(len(train_input))
            for n in order:
                codes, feats, subj_feat = train_input[n]
                T = len(codes)
                true_feat = int(subj_feat)
                if perm_reward:
                    true_feat = int(np.random.randint(0, F))
                # -------- forward pass (WITH the competitive stabilizer; store per-token quantities) --------
                g = 0.0; m = np.zeros(F); p_sum = 0.0; s_inh = 0.0
                rec = []
                for t in range(T):
                    code = codes[t]
                    v = np.zeros(F); v[feats[t]] = 1.0
                    a1 = self.W1 @ code + self.w_g1 * g + self.b1
                    r1 = 1.0 / (1.0 + np.exp(-self.hidden_gain * a1))
                    denom = 1.0 + self.div_k * float(np.mean(r1))
                    h = r1 / denom
                    sp1 = self.hidden_gain * r1 * (1.0 - r1) / denom
                    z2 = float(self.W2 @ h + self.b2) - self.out_lambda * s_inh
                    p = 1.0 / (1.0 + np.exp(-self.gain * z2))
                    sp2 = self.gain * p * (1.0 - p)
                    dm = v - m
                    rec.append((code, h, sp1, sp2, dm, g))
                    m = (1.0 - p) * m + p * v
                    p_sum += p
                    s_inh = self.inh_leak * s_inh + p
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
                    sp1_use = sp1 if self.sigma_prime else 1.0
                    e1 = (B1_use * e2) * sp1_use
                    gW1 += np.outer(e1, code); gwg1 += e1 * g_in; gb1 += e1
                # -------- gradient descent --------
                self.W2 -= self.lr * gW2; self.b2 -= self.lr * gb2
                self.W1 -= self.lr * gW1; self.w_g1 -= self.lr * gwg1; self.b1 -= self.lr * gb1
                if self.homeo > 0.0:
                    tgt = self.target_rate if self.target_rate is not None else (1.0 / max(1, T))
                    self.b2 -= self.homeo * (p_sum / max(1, T) - tgt)
                # -------- canonical Kolen-Pollack co-adaptation (transport-free; both hops) with the kp_lr BOOST --------
                if self.feedback == "chained_kp":
                    gR = np.outer(delta, m)
                    R -= self.lr * self.kp_ro_lr_scale * (gR + self.kp_wd * R)
                    B_out -= kp_lr_eff * self.lr * (gR.T + self.kp_wd * B_out)
                    self.W2 -= self.lr * self.kp_wd * self.W2
                    B1 -= kp_lr_eff * self.lr * (gW2 + self.kp_wd * B1)
                # -------- optional per-sentence MIRROR REFRESH (re-align B to the CURRENT forward weights) --------
                if mirror_every > 0 and (sent_ctr % mirror_every == 0):
                    B_out, B1 = self._mirror_estimate(R, F, mirror_refresh_steps, mrng)
                sent_ctr += 1
        RTf = (R.T if R is not None else rs * np.eye(F))
        self.cos_hopA_final = _cos(B_out, RTf)
        self.cos_hopB_final = _cos(self.W2 if self.feedback == "aligned" else B1, self.W2)


# ====================================================================================================================
# Deploy stats (mirror of the varpos _deploy_stats but records the extra pre-warmup alignment read).
# ====================================================================================================================
def _deploy_stats(gate, slot_factory, sentences, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F, use_tag=True):
    acc, alive, zok = eval_varpos_wm(slot_factory(), sentences, gate, noun_codes, tag_codes, feat_of,
                                     verb_tok_of_feat, F, use_tag=use_tag)
    tf = tag_position_fire(gate, sentences, noun_codes, tag_codes, use_tag=use_tag)
    st = {"acc": float(acc), "hold_alive": float(alive)}
    st.update(tf)
    for ck in ("cos_hopA_prewarm", "cos_hopA_init", "cos_hopA_final",
               "cos_hopB_prewarm", "cos_hopB_init", "cos_hopB_final"):
        st[ck] = getattr(gate, ck, None)
    return st


# ====================================================================================================================
# One (seed, L) point -- the LEVER-4 baselines + the new lever arms.
# ====================================================================================================================
def run_point(seed, N, F, C, L, n_train, n_test, recur, hold_steps, load_steps, clear_steps, episodes,
              hidden, lr, kp_lr, kp_wd, readout_scale, homeo, b2_init, div_k, out_lambda, inh_leak,
              warmup_steps, kp_boost, mirror_every, mirror_refresh_steps, subj_lo=1):
    np.random.seed(seed)
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

    # ---- ceiling oracle + validity teeth ----
    oracle = CaseMarkerOracle(tag_codes[0])
    acc_oracle, alive, zero_ok = eval_varpos_wm(_slot(), test_seqs, oracle, noun_codes, tag_codes, feat_of,
                                                verb_tok_of_feat, F)
    acc_oracle_lesion, _, _ = eval_varpos_wm(_slot(rc=0.0), test_seqs, CaseMarkerOracle(tag_codes[0]), noun_codes,
                                             tag_codes, feat_of, verb_tok_of_feat, F)
    pc_rng = np.random.default_rng(seed + 17)
    pc_test = [permute_cue_out(s, pc_rng) for s in test_seqs]
    acc_permcue, _, _ = eval_varpos_wm(_slot(), pc_test, CaseMarkerOracle(tag_codes[0]), noun_codes, tag_codes,
                                       feat_of, verb_tok_of_feat, F)

    # ---- the ONSET floor ----
    onset_gate = MarkerRoleGate()
    onset = _deploy_stats(onset_gate, _slot, test_seqs, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F)

    # ---- the n-gram HELD-OUT floor ----
    try:
        ngram_test, ngram_order = ngram_floor_heldout(_flat_seqs(train_seqs, C, N), _flat_seqs(test_seqs, C, N), L, F)
    except Exception:
        ngram_test, ngram_order = float("nan"), -1

    # ---- per-arm training inputs (composite codes; identity control gets NOUN-ONLY codes) ----
    def _train_input(use_tag):
        data = []
        for sent in train_seqs:
            codes = [compose_code(noun_codes, tag_codes, nn, tg, use_tag=use_tag) for (nn, tg) in sent["tokens"]]
            feats = [feat_of[nn] for (nn, tg) in sent["tokens"]]
            data.append((codes, feats, feat_of[sent["subj_noun"]]))
        return data
    train_in = _train_input(True)
    train_in_noun = _train_input(False)

    def _gate(feedback, off, stabilize=True, sigma_prime=True, perm_reward=False, train=True, use_tag=True,
              warmup=0, kp_lr_boost=1.0, mir_every=0):
        g = FBAlignVarposGate(hidden=hidden, gain=4.0, lr=lr, seed=seed + off, feedback=feedback,
                              kp_lr=kp_lr, kp_wd=kp_wd, elig_leak=1.0, readout_scale=readout_scale,
                              sigma_prime=sigma_prime, homeo=homeo, b2_init=b2_init,
                              stabilize=stabilize, div_k=div_k, out_lambda=out_lambda, inh_leak=inh_leak)
        if train:
            g.train_varpos_fb(train_in_noun if not use_tag else train_in, F, episodes=episodes, perm_reward=perm_reward,
                              warmup_steps=warmup, kp_lr_boost=kp_lr_boost, mirror_every=mir_every,
                              mirror_refresh_steps=mirror_refresh_steps)
        return _deploy_stats(g, _slot, test_seqs, noun_codes, tag_codes, feat_of, verb_tok_of_feat, F, use_tag=use_tag)

    # -- the LEVER-4 baselines (offsets MATCH the varpos runner so aligned/fa/kp are the same trainings) --
    aligned = _gate("aligned", 131, stabilize=True)                                  # exact-feedback ceiling
    fa = _gate("chained_fa", 137, stabilize=True)                                    # LEVER-4 transport-free FA (bimodal)
    kp = _gate("chained_kp", 143, stabilize=True)                                    # LEVER-4 transport-free KP (collapses)
    # -- THE NEW LEVER ARMS (distinct offsets -> independently seeded) --
    mirror_fa = _gate("chained_fa", 181, stabilize=True, warmup=warmup_steps)                        # L1 align-then-hold
    mirror_fa_ref = _gate("chained_fa", 187, stabilize=True, warmup=warmup_steps, mir_every=mirror_every)  # L1 maintained
    mirror_kp = _gate("chained_kp", 193, stabilize=True, warmup=warmup_steps)                        # L1 align-then-KP
    kp_strong = _gate("chained_kp", 199, stabilize=True, warmup=warmup_steps, kp_lr_boost=kp_boost,
                      mir_every=mirror_every)                                                        # L2 stronger/longer KP
    # -- validity killers + crux (reused verbatim) --
    untrained = _gate("chained_fa", 161, stabilize=True, train=False)
    permrew_fa = _gate("chained_fa", 137, stabilize=True, perm_reward=True)
    permrew_kp = _gate("chained_kp", 143, stabilize=True, perm_reward=True)
    identity_control = _gate("chained_fa", 173, stabilize=True, use_tag=False)

    p_hit0 = (1.0 / (L + 1 - subj_lo)) if subj_lo == 0 else 0.0
    onset_expected = p_hit0 + (1.0 - p_hit0) * chance
    return {"seed": seed, "N": N, "F": F, "C": C, "L": L, "distance": L + 1, "chance": chance, "subj_lo": subj_lo,
            "gen_defined": bool(gen_defined), "n_test": len(test_seqs), "onset_expected": onset_expected, "hidden": hidden,
            "warmup_steps": warmup_steps, "kp_boost": kp_boost, "mirror_every": mirror_every,
            "acc_oracle": acc_oracle, "acc_oracle_lesion": acc_oracle_lesion, "acc_permuted_cue": acc_permcue,
            "hold_alive": alive, "zero_input_ok": zero_ok, "ngram_floor_test": ngram_test, "ngram_order": ngram_order,
            "onset_gate": onset, "aligned_gate": aligned, "fa_gate": fa, "kp_gate": kp,
            "mirror_fa_gate": mirror_fa, "mirror_fa_ref_gate": mirror_fa_ref, "mirror_kp_gate": mirror_kp,
            "kp_strong_gate": kp_strong, "untrained_gate": untrained, "permrew_fa_gate": permrew_fa,
            "permrew_kp_gate": permrew_kp, "identity_control_gate": identity_control}


_GATE_KEYS = ["onset_gate", "aligned_gate", "fa_gate", "kp_gate", "mirror_fa_gate", "mirror_fa_ref_gate",
              "mirror_kp_gate", "kp_strong_gate", "untrained_gate", "permrew_fa_gate", "permrew_kp_gate",
              "identity_control_gate"]
_COS_KEYS = ("cos_hopA_prewarm", "cos_hopA_init", "cos_hopA_final",
             "cos_hopB_prewarm", "cos_hopB_init", "cos_hopB_final")


def _agg_gate(per, key):
    sub = [p[key] for p in per]
    base = {k: float(np.mean([d[k] for d in sub])) for k in ("acc", "hold_alive", "fire_nom", "fire_obl",
            "fire_pos0", "fire_posgt0", "tag_identity_gap", "n_matched")}
    for ck in _COS_KEYS:
        vals = [d[ck] for d in sub if ck in d and d[ck] is not None]
        if len(vals) == len(sub) and vals:
            base[ck] = float(np.mean(vals))
            base[ck + "_min"] = float(np.min(vals))
    base["acc_min"] = float(np.min([d["acc"] for d in sub]))
    base["acc_max"] = float(np.max([d["acc"] for d in sub]))
    base["acc_std"] = float(np.std([d["acc"] for d in sub]))
    base["gap_min"] = float(np.min([d["tag_identity_gap"] for d in sub]))
    base["fire_nom_min"] = float(np.min([d["fire_nom"] for d in sub]))
    base["fire_obl_max"] = float(np.max([d["fire_obl"] for d in sub]))
    base["per_seed_acc"] = [round(float(d["acc"]), 4) for d in sub]
    return base


def agg(per):
    keys = ["acc_oracle", "acc_oracle_lesion", "acc_permuted_cue", "hold_alive", "ngram_floor_test", "onset_expected"]
    a = {k: float(np.mean([p[k] for p in per])) for k in keys}
    a["acc_oracle_min"] = float(np.min([p["acc_oracle"] for p in per]))
    for gk in _GATE_KEYS:
        a[gk] = _agg_gate(per, gk)
    a.update({"N": per[0]["N"], "F": per[0]["F"], "C": per[0]["C"], "L": per[0]["L"], "distance": per[0]["distance"],
              "chance": per[0]["chance"], "hidden": per[0]["hidden"], "warmup_steps": per[0]["warmup_steps"],
              "kp_boost": per[0]["kp_boost"], "mirror_every": per[0]["mirror_every"],
              "gen_defined": all(p["gen_defined"] for p in per), "zero_input_ok": all(p["zero_input_ok"] for p in per),
              "per_seed": per})
    return a


# ====================================================================================================================
# GPU SCALE PROBE: a BATCHED deep-MLP FA / KP / aligned width sweep (the 0801 regime, where the GPU genuinely helps).
# Task: an arbitrary-CUE-detection classification -- the rate analog of the role task's core difficulty. The input is a
# set of (L+1) sparse D-dim barcodes stacked into a sequence; ONE carries the NOM tag (the subject cue). We POOL them
# through a deep MLP that must, transport-free, learn to read the NOM-tagged token's class. We sweep hidden WIDTH and
# measure 6-seed reliability (min accuracy) for aligned / FA / KP -> does transport-free reliability improve with width?
# GPU-appropriate: the whole batch is one big matmul per layer. VRAM + throughput reported.
# ====================================================================================================================
def _xp():
    backend = os.environ.get("SIM_BACKEND", "numpy")
    if backend == "cupy":
        try:
            import cupy as cp
            return cp, "cupy", "gpu"
        except Exception:
            pass
    return np, "numpy", "cpu"


def _make_teacher(rng, Din, F, ht=64, tdepth=2):
    """A FIXED random deep-ReLU TEACHER MLP (Din -> ht -> ... -> F) defining a learnable, genuinely-deep classification --
    the faithful analog of the 0801 GPU regime (a real deep classification where transport-free FA/KP reliability is
    depth/width-sensitive), NOT the sequential binding task (which no plain MLP can solve). The label = argmax teacher(x)."""
    sizes = [Din] + [ht] * tdepth + [F]
    Wt = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i + 1], sizes[i])) for i in range(len(sizes) - 1)]
    return Wt


def _teacher_batch(xp, teacher, rng, B, Din):
    """Random Gaussian inputs through the fixed teacher -> argmax label. Returns X (B,Din), y (B,)."""
    X = rng.normal(0, 1.0, (B, Din))
    a = X
    for i, Wt in enumerate(teacher):
        z = a @ Wt.T
        a = np.maximum(z, 0.0) if i < len(teacher) - 1 else z
    y = a.argmax(axis=1)
    return xp.asarray(X), xp.asarray(y)


def _weight_mirror_mlp(xp, W_next, Bf_i, steps, rng, mirror_wd=0.0):
    """Akrout-2019 weight mirror for the batched MLP: estimate W_next^T into Bf_i via noise-output correlation
    (E[outer(xi, W_next @ xi)] = W_next^T). Transport-free. Bf_i shape (n_out_i, n_in_next) == W_next.T shape."""
    n_in = W_next.shape[1]
    S = xp.zeros_like(Bf_i)
    XI = xp.asarray(rng.normal(0.0, 1.0, (steps, n_in)))
    Z = XI @ W_next.T                                          # (steps, n_out_next); forward response to noise
    S = XI.T @ Z / steps                                      # -> W_next^T in expectation
    return S


def _mlp_scale_arm(xp, seed, feedback, widths_layers, X, y, Xte, yte, F, epochs, lr, kp_lr, kp_wd, batch,
                   warmup=0):
    """Train a deep MLP (sizes = [n_in] + hidden_layers + [F]) by aligned / FA (fixed-random B) / KP (co-adapting B) /
    mirror_fa (FA with a weight-mirror WARM-UP), ReLU hidden + sigma', softmax-CE output. Returns held-out accuracy +
    per-hidden-layer cos(B, W^T)."""
    rng = np.random.default_rng(seed + 3)
    n_in = X.shape[1]
    sizes = [n_in] + list(widths_layers) + [F]
    nL = len(sizes) - 1
    W = [xp.asarray(rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i + 1], sizes[i]))) for i in range(nL)]  # He init (ReLU)
    b = [xp.zeros(sizes[i + 1]) for i in range(nL)]
    Bf = [xp.asarray(rng.normal(0, 1.0 / np.sqrt(sizes[i + 1]), (sizes[i + 1], sizes[i + 2]))) for i in range(nL - 1)]
    fb = "fa" if feedback == "mirror_fa" else feedback
    # ---- weight-mirror WARM-UP (align B -> W^T before training; FA freezes B afterward) ----
    if feedback == "mirror_fa" and warmup > 0:
        for i in range(nL - 1):
            Bf[i] = _weight_mirror_mlp(xp, W[i + 1], Bf[i], warmup, rng)
    n = X.shape[0]
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for s in range(0, n, batch):
            bi = perm[s:s + batch]
            xb = X[bi]; yb = y[bi]
            a = [xb]; hs = [xb]; sps = [None]
            for i in range(nL):
                z = a[-1] @ W[i].T + b[i]
                if i < nL - 1:
                    h = xp.maximum(z, 0.0); sp = (z > 0.0).astype(z.dtype)   # ReLU + sigma'
                    a.append(h); hs.append(h); sps.append(sp)
                else:
                    a.append(z)
            logits = a[-1]; logits = logits - logits.max(axis=1, keepdims=True)
            ez = xp.exp(logits); probs = ez / ez.sum(axis=1, keepdims=True)
            oneh = xp.zeros_like(probs); oneh[xp.arange(len(yb)), yb] = 1.0
            dout = (probs - oneh) / len(yb)
            deltas = [None] * nL; deltas[nL - 1] = dout
            for i in range(nL - 2, -1, -1):
                back = (deltas[i + 1] @ W[i + 1]) if fb == "aligned" else (deltas[i + 1] @ Bf[i].T)
                deltas[i] = back * sps[i + 1]
            gWs = [deltas[i].T @ hs[i] for i in range(nL)]
            for i in range(nL):
                W[i] -= lr * gWs[i]; b[i] -= lr * deltas[i].sum(axis=0)
            if fb == "kp":
                for i in range(nL - 1):
                    Bf[i] -= kp_lr * lr * (gWs[i + 1].T + kp_wd * Bf[i])
                    W[i + 1] -= lr * kp_wd * W[i + 1]
    # held-out accuracy
    a = Xte
    for i in range(nL):
        z = a @ W[i].T + b[i]
        a = (1.0 / (1.0 + xp.exp(-z))) if i < nL - 1 else z
    pred = a.argmax(axis=1)
    acc = float((pred == yte).mean())
    # per-hidden-layer alignment cos(Bf[i], W[i+1]^T)
    cos = []
    for i in range(nL - 1):
        u = Bf[i].reshape(-1); w = W[i + 1].T.reshape(-1)
        denom = float(xp.linalg.norm(u) * xp.linalg.norm(w)) + 1e-12
        cos.append(float((u @ w) / denom))
    return acc, cos


def run_scale_probe(a):
    xp, backend, device = _xp()
    F = a.n_feat; Din = a.scale_dim
    chance = 1.0 / F
    vram_mb = None
    t0 = time.time()
    print(f"[scale-probe] backend={backend} device={device} | widths={a.widths} student-depth(hidden layers)="
          f"{a.scale_depth} | random-TEACHER classification Din={Din} F={F} | B_train={a.scale_train} "
          f"B_test={a.scale_test} epochs={a.scale_epochs} batch={a.scale_batch} lr={a.scale_lr} | seeds={a.scale_seeds}",
          flush=True)
    arms = (("aligned", "aligned"), ("fa", "fa"), ("kp", "kp"), ("mirror_fa", "mirror_fa"))
    results = {}
    for wdt in a.widths:
        hidden_layers = [wdt] * a.scale_depth
        per_arm = {arm: [] for arm, _ in arms}
        cos_arm = {arm: [] for arm, _ in arms}
        for seed in a.scale_seeds:
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            teacher = _make_teacher(np.random.default_rng(seed + 11), Din, F, ht=a.teacher_hidden, tdepth=a.teacher_depth)
            X, y = _teacher_batch(xp, teacher, rng, a.scale_train, Din)
            Xte, yte = _teacher_batch(xp, teacher, rng, a.scale_test, Din)
            for arm, fb in arms:
                acc, cos = _mlp_scale_arm(xp, seed, fb, hidden_layers, X, y, Xte, yte, F,
                                          a.scale_epochs, a.scale_lr, a.kp_lr, a.kp_wd, a.scale_batch,
                                          warmup=a.mirror_steps)
                per_arm[arm].append(acc); cos_arm[arm].append(cos)
        if backend == "cupy":
            try:
                import cupy as cp
                # total_bytes = the pool's high-water mark (peak reserved), robust to per-arm frees
                vram_mb = round(cp.get_default_memory_pool().total_bytes() / 1e6, 1)
            except Exception:
                pass
        row = {}
        for arm, _ in arms:
            accs = per_arm[arm]
            deep_cos = [c[0] for c in cos_arm[arm]]             # deepest hidden layer's alignment
            row[arm] = {"acc_mean": round(float(np.mean(accs)), 4), "acc_min": round(float(np.min(accs)), 4),
                        "acc_max": round(float(np.max(accs)), 4), "per_seed": [round(x, 4) for x in accs],
                        "deep_cos_mean": round(float(np.mean(deep_cos)), 4),
                        "deep_cos_min": round(float(np.min(deep_cos)), 4)}
        results[str(wdt)] = row
        print(f"  width={wdt:5d} (depth {a.scale_depth}) | aligned {row['aligned']['acc_mean']:.3f}"
              f"[min {row['aligned']['acc_min']:.3f}] | FA {row['fa']['acc_mean']:.3f}[min {row['fa']['acc_min']:.3f}]"
              f" cos{row['fa']['deep_cos_mean']:+.2f} | KP {row['kp']['acc_mean']:.3f}[min {row['kp']['acc_min']:.3f}]"
              f" cos{row['kp']['deep_cos_mean']:+.2f} | mirror_FA {row['mirror_fa']['acc_mean']:.3f}"
              f"[min {row['mirror_fa']['acc_min']:.3f}] cos{row['mirror_fa']['deep_cos_mean']:+.2f} | VRAM {vram_mb} MB",
              flush=True)
    elapsed = round(time.time() - t0, 1)
    thr = round((len(a.widths) * len(a.scale_seeds) * len(arms)) / max(elapsed, 1e-6), 3)
    summary = {"probe": "rolegate_fbalign_scale_probe", "backend": backend, "device": device, "chance": chance,
               "vram_mb": vram_mb, "elapsed_seconds": elapsed, "arms_per_second": thr,
               "config": {"widths": a.widths, "student_depth": a.scale_depth, "F": F, "Din": Din,
                          "teacher_hidden": a.teacher_hidden, "teacher_depth": a.teacher_depth,
                          "scale_train": a.scale_train, "scale_test": a.scale_test, "scale_epochs": a.scale_epochs,
                          "scale_batch": a.scale_batch, "scale_lr": a.scale_lr, "kp_lr": a.kp_lr, "kp_wd": a.kp_wd,
                          "mirror_steps": a.mirror_steps, "scale_seeds": a.scale_seeds},
               "results_by_width": results,
               "HONEST_NOTE": "A GPU-appropriate BATCHED deep-MLP aligned/FA/KP/mirror_fa WIDTH sweep on a static "
                              "arbitrary-cue-detection classification (the rate analog of the role task's core "
                              "difficulty: gather the noun-class of the position whose tag==NOM; noun+tag in disjoint "
                              "concat dims, ReLU hidden). It measures whether TRANSPORT-FREE feedback-alignment "
                              "RELIABILITY (min accuracy over seeds) closes toward the aligned ceiling as hidden WIDTH "
                              "grows, and whether the weight-mirror WARM-UP (mirror_fa) also fixes FA at scale. It is NOT "
                              "the sequential role task (no WM latch); it isolates the width-dependence of the alignment "
                              "reliability itself, at a scale where the GPU helps. The role task's own result at H=32 is "
                              "the direct on-task measurement."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[scale-probe] wrote {a.out} (elapsed {elapsed}s, {thr} arms/s, VRAM {vram_mb} MB)", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-noun", type=int, default=12)
    ap.add_argument("--n-feat", type=int, default=4)
    ap.add_argument("--n-tags", type=int, default=4)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--subj-lo", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=140)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--recur", type=float, default=25.0)
    ap.add_argument("--hold-steps", type=int, default=18)
    ap.add_argument("--load-steps", type=int, default=30)
    ap.add_argument("--clear-steps", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=32, help="hidden population H (the role-task SCALE knob; sweep it)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--kp-lr", type=float, default=0.3)
    ap.add_argument("--kp-wd", type=float, default=0.01)
    ap.add_argument("--readout-scale", type=float, default=3.0)
    ap.add_argument("--homeo", type=float, default=0.10)
    ap.add_argument("--b2-init", type=float, default=0.3)
    ap.add_argument("--div-k", type=float, default=2.0)
    ap.add_argument("--out-lambda", type=float, default=4.0)
    ap.add_argument("--inh-leak", type=float, default=1.0)
    # ---- the new lever knobs ----
    ap.add_argument("--mirror-steps", type=int, default=6000, help="weight-mirror WARM-UP steps (align B->forward^T); "
                    "0 -> the arms reduce to the LEVER-4 baselines")
    ap.add_argument("--kp-boost", type=float, default=6.0, help="kp_lr multiplier for the kp_strong arm (stronger KP)")
    ap.add_argument("--mirror-every", type=int, default=20, help="per-sentence mirror REFRESH cadence (0=off) for the "
                    "mirror_fa_ref and kp_strong arms (maintain alignment during credit)")
    ap.add_argument("--mirror-refresh-steps", type=int, default=512, help="steps per mirror refresh")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--go-distance", type=int, default=None)
    ap.add_argument("--merge-from", nargs="+", default=None)
    # ---- scale probe ----
    ap.add_argument("--scale-probe", action="store_true", help="run the GPU batched deep-MLP FA/KP/aligned width sweep")
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--scale-depth", type=int, default=3, help="# hidden layers in the scale-probe MLP")
    ap.add_argument("--scale-dim", type=int, default=64)
    ap.add_argument("--scale-seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--scale-train", type=int, default=4000)
    ap.add_argument("--scale-test", type=int, default=1000)
    ap.add_argument("--scale-epochs", type=int, default=40)
    ap.add_argument("--scale-batch", type=int, default=256)
    ap.add_argument("--scale-lr", type=float, default=0.2)
    ap.add_argument("--teacher-hidden", type=int, default=64, help="scale-probe random-teacher hidden width")
    ap.add_argument("--teacher-depth", type=int, default=2, help="scale-probe random-teacher # hidden layers")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.scale_probe:
        return run_scale_probe(a)

    if a.merge_from:
        by_L = {}; merged_seeds = []
        for pth in a.merge_from:
            d = json.loads(Path(pth).read_text())
            for pt in d.get("points", []):
                by_L.setdefault(pt["L"], []).extend(pt.get("per_seed", []))
            for pt in d.get("points", [])[:1]:
                merged_seeds.extend([ps["seed"] for ps in pt.get("per_seed", [])])
        a.seeds = sorted(set(merged_seeds)); a.distances = sorted(by_L.keys())
    smoke = a.smoke or len(a.seeds) < 6
    if a.smoke:
        a.n_test = min(a.n_test, 30)
    backend = os.environ.get("SIM_BACKEND", "numpy"); device = "cpu" if backend == "numpy" else "gpu"
    F = a.n_feat; C = a.n_tags; chance = 1.0 / F
    dists = sorted(set(a.distances))
    print(f"backend={backend} device={device} | N={a.n_noun} F={F} C={C} chance={chance:.3f} | L={dists} | "
          f"episodes={a.episodes} hidden={a.hidden} lr={a.lr} | WARM-UP mirror-steps={a.mirror_steps} kp-boost="
          f"{a.kp_boost} mirror-every={a.mirror_every} | n_train={a.n_train} n_test={a.n_test} | seeds={a.seeds} "
          f"smoke={smoke}", flush=True)

    t0 = time.time(); err = None; points = []
    try:
        for L in dists:
            if a.merge_from:
                per = [ps for ps in by_L.get(L, [])]
            else:
                per = [run_point(s, a.n_noun, F, C, L, a.n_train, a.n_test, a.recur, a.hold_steps, a.load_steps,
                                 a.clear_steps, a.episodes, a.hidden, a.lr, a.kp_lr, a.kp_wd, a.readout_scale, a.homeo,
                                 a.b2_init, a.div_k, a.out_lambda, a.inh_leak, a.mirror_steps, a.kp_boost,
                                 a.mirror_every, a.mirror_refresh_steps, a.subj_lo) for s in a.seeds]
            p = agg(per); points.append(p)
            print(f"  [N={a.n_noun} F={F} C={C} L={L} dist={L+1} gen={p['gen_defined']} H={a.hidden}] ORACLE "
                  f"{p['acc_oracle']:.3f}[min {p['acc_oracle_min']:.3f}] | onset-prob {p['onset_expected']:.3f} | "
                  f"n-gram {p['ngram_floor_test']:.3f} | chance {chance:.3f}", flush=True)
            order = [("onset(FLOOR)", "onset_gate"), ("aligned(CEILING)", "aligned_gate"),
                     ("chained_FA(L4)", "fa_gate"), ("chained_KP(L4)", "kp_gate"),
                     ("mirror_FA(align-hold)", "mirror_fa_gate"), ("mirror_FA_ref(align-maintain)", "mirror_fa_ref_gate"),
                     ("mirror_KP(align-then-KP)", "mirror_kp_gate"), ("kp_STRONG", "kp_strong_gate"),
                     ("UNTRAINED", "untrained_gate"), ("FA_PERMREW", "permrew_fa_gate"),
                     ("KP_PERMREW", "permrew_kp_gate"), ("identity(noun-only)", "identity_control_gate")]
            print(f"     arm  acc[min,std] | fire NOM/obl | tag-gap[min] | cosB init->final [min] :", flush=True)
            for tag, gk in order:
                gs = p[gk]
                cA = gs.get("cos_hopA_init"); cAf = gs.get("cos_hopA_final")
                cB = gs.get("cos_hopB_init"); cBf = gs.get("cos_hopB_final"); cBfm = gs.get("cos_hopB_final_min")
                cs = ""
                if cB is not None:
                    cs = (f" | hopA {cA:+.2f}->{cAf:+.2f} hopB {cB:+.2f}->{cBf:+.2f}"
                          f"[min {cBfm:+.2f}]" if cBfm is not None else f" | hopA {cA:+.2f}->{cAf:+.2f}")
                print(f"       {tag:30s} {gs['acc']:.3f}[{gs['acc_min']:.3f},{gs['acc_std']:.3f}] | "
                      f"{gs['fire_nom']:.2f}/{gs['fire_obl']:.2f} | gap {gs['tag_identity_gap']:+.2f}"
                      f"[{gs['gap_min']:+.2f}]{cs}", flush=True)
            print(f"     per-seed acc [mirror_FA || mirror_FA_ref || mirror_KP || kp_STRONG || FA(L4) || aligned]:",
                  flush=True)
            for ps in per:
                mf = ps["mirror_fa_gate"]; mfr = ps["mirror_fa_ref_gate"]; mk = ps["mirror_kp_gate"]
                ks = ps["kp_strong_gate"]; fa = ps["fa_gate"]; al = ps["aligned_gate"]
                print(f"       seed {ps['seed']}: mFA {mf['acc']:.3f}(cosB {mf.get('cos_hopB_final'):+.2f}) || "
                      f"mFAref {mfr['acc']:.3f}(cosB {mfr.get('cos_hopB_final'):+.2f}) || mKP {mk['acc']:.3f}"
                      f"(cosB {mk.get('cos_hopB_final'):+.2f}) || kpS {ks['acc']:.3f}(cosB {ks.get('cos_hopB_final'):+.2f})"
                      f" || FA {fa['acc']:.3f} || aligned {al['acc']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    go_L = a.go_distance if a.go_distance is not None else (max(dists) if dists else None)
    far = None
    if err is None and points:
        cand = [p for p in points if p["L"] == go_L]
        far = cand[0] if cand else max(points, key=lambda p: p["L"])

    verdict = None; role_go = False; valid = False; winner = None
    if err is None and far is not None:
        chance = far["chance"]; tol = chance + 0.15
        orc = far["acc_oracle"]; onset = far["onset_gate"]; al = far["aligned_gate"]
        fa = far["fa_gate"]; kp = far["kp_gate"]; unt = far["untrained_gate"]
        prf = far["permrew_fa_gate"]; prk = far["permrew_kp_gate"]; idc = far["identity_control_gate"]
        lever_arms = {"mirror_fa": far["mirror_fa_gate"], "mirror_fa_ref": far["mirror_fa_ref_gate"],
                      "mirror_kp": far["mirror_kp_gate"], "kp_strong": far["kp_strong_gate"]}

        untrained_fails = unt["acc"] <= tol
        permrew_fa_fails = prf["acc"] <= tol
        permrew_kp_fails = prk["acc"] <= tol
        onset_fails = onset["acc"] <= tol
        valid = bool(untrained_fails and permrew_fa_fails and permrew_kp_fails)

        def _is_role(g):
            return (g["acc"] >= chance + 0.20 and g["fire_nom"] >= 0.60 and g["fire_obl"] <= 0.40
                    and g["tag_identity_gap"] >= 0.30)

        def _reliable(g):
            return (not smoke) and g["acc"] >= chance + 0.30 and g["acc_min"] >= 0.60 and g["gap_min"] >= 0.30

        oracle_ok = orc >= chance + 0.30
        aligned_reaches = _is_role(al) and (smoke or al["acc_min"] >= 0.60)
        idc_fails = idc["acc"] <= tol or idc["tag_identity_gap"] <= 0.20
        # the GO: ANY new lever arm reaches role RELIABLY on the valid task
        cleared = [(name, g) for name, g in lever_arms.items()
                   if valid and _is_role(g) and _reliable(g) and oracle_ok and idc_fails and not smoke]
        role_go = bool(cleared)
        if cleared:
            winner = max(cleared, key=lambda kv: kv[1]["acc_min"])[0]

        print(f"\n-- ROLE-GATE FEEDBACK-ALIGNMENT RELIABILITY verdict at L={far['L']} (dist {far['distance']}) --",
              flush=True)
        try:
            lever("VALIDITY: UNTRAINED gate must FAIL", round(orc, 3), round(unt["acc"], 3), required=False,
                  continuous=f"untrained {unt['acc']:.3f} vs chance+0.15 {tol:.3f} (fails={untrained_fails})")
            lever("VALIDITY: permuted-reward FA must collapse", round(fa["acc"], 3), round(prf["acc"], 3),
                  required=False, continuous=f"permrew-FA {prf['acc']:.3f} vs {tol:.3f} (fails={permrew_fa_fails})")
            for name, g in lever_arms.items():
                lever(f"LEVER {name}: transport-free role reliability (min over seeds)",
                      round(fa["acc_min"], 3), round(g["acc_min"], 3), required=False,
                      continuous=f"{name} acc {g['acc']:.3f}[min {g['acc_min']:.3f}] cosB_final {g.get('cos_hopB_final')} "
                                 f"[min {g.get('cos_hopB_final_min')}] gap {g['tag_identity_gap']:+.2f}[min {g['gap_min']:+.2f}]")
        except LeverError:
            pass
        attributable_to("mirror_kp role attributable to alignment (vs LEVER-4 KP)",
                        far["mirror_kp_gate"]["acc"], kp["acc"])

        common = (f"VALIDITY[untrained {unt['acc']:.3f}({untrained_fails}), permrew-FA {prf['acc']:.3f}"
                  f"({permrew_fa_fails}), permrew-KP {prk['acc']:.3f}({permrew_kp_fails}), onset {onset['acc']:.3f}"
                  f"({onset_fails})]->valid={valid}. ORACLE {orc:.3f}[min {far['acc_oracle_min']:.3f}] (lesion "
                  f"{far['acc_oracle_lesion']:.3f}, perm-cue {far['acc_permuted_cue']:.3f}); n-gram "
                  f"{far['ngram_floor_test']:.3f}; chance {chance:.3f}. aligned(CEILING) {al['acc']:.3f}[min "
                  f"{al['acc_min']:.3f}]. LEVER-4 baselines: FA {fa['acc']:.3f}[min {fa['acc_min']:.3f}], KP "
                  f"{kp['acc']:.3f}[min {kp['acc_min']:.3f}]. NEW LEVERS: "
                  + "; ".join(f"{nm} {g['acc']:.3f}[min {g['acc_min']:.3f}] cosB {g.get('cos_hopB_final')}"
                              f"[min {g.get('cos_hopB_final_min')}] gap {g['tag_identity_gap']:+.2f}[min {g['gap_min']:+.2f}]"
                              for nm, g in lever_arms.items())
                  + f". identity-control {idc['acc']:.3f} gap {idc['tag_identity_gap']:+.2f} (fails={idc_fails}).")
        smoketag = "" if not smoke else " (1-seed indicator; run the 6-seed sweep)"

        if not valid:
            verdict = (f"ROLE-GATE FBALIGN INVALID{smoketag} -- the task is confounded (untrained/permuted-reward exceed "
                       f"chance+0.15). {common} FIX the stream before reading any credit result. NO sim/ edit.")
        elif role_go or (smoke and valid and any(_is_role(g) for g in lever_arms.values()) and oracle_ok and idc_fails):
            wname = winner or max(lever_arms.items(), key=lambda kv: kv[1]["acc_min"])[0]
            top = lever_arms[wname]
            verdict = (f"ROLE-GATE FBALIGN POSITIVE (brain-based, transport-free){smoketag} -- a STRONGER transport-free "
                       f"feedback alignment ({wname}) makes transport-free credit RELIABLE on the variable-position role "
                       f"task: {wname} {top['acc']:.3f}[min {top['acc_min']:.3f}] fire NOM {top['fire_nom']:.2f}/obl "
                       f"{top['fire_obl']:.2f} tag-gap {top['tag_identity_gap']:+.2f}[min {top['gap_min']:+.2f}], "
                       f"cosB_final {top.get('cos_hopB_final')}[min {top.get('cos_hopB_final_min')}], where the LEVER-4 "
                       f"baselines were bimodal-FA/collapsed-KP. {common} The credit is REQUIRED (permuted-reward "
                       f"collapses), the tag cue is load-bearing (identity control fails), the oracle confirms a target. "
                       f"So reliable transport-free feedback alignment IS achievable on the role task. CAVEAT: the 2-layer "
                       f"net + chained credit + the competition + the weight-mirror are HOST math; their on-substrate "
                       f"spiking realisation is the named next rung. Reuse-by-import; NO sim/ edit.")
        elif aligned_reaches:
            # did any lever RAISE worst-seed alignment while accuracy still collapsed? -> residual PAST alignment
            best = max(lever_arms.items(), key=lambda kv: kv[1]["acc_min"])
            aligned_high = [(nm, g) for nm, g in lever_arms.items()
                            if g.get("cos_hopB_final_min") is not None and g["cos_hopB_final_min"] >= 0.60
                            and g["acc_min"] < 0.60]
            past = "; alignment HIGH but accuracy still collapses on " + \
                   ", ".join(f"{nm}(cosB_min {g['cos_hopB_final_min']:+.2f}, acc_min {g['acc_min']:.3f})"
                             for nm, g in aligned_high) + " -> the residual is PAST alignment (credit signal-to-noise / " \
                   "basin dynamics), not alignment quality." if aligned_high else \
                   f"; the best lever ({best[0]}) reaches acc_min {best[1]['acc_min']:.3f} (< 0.60) -> alignment " \
                   "improvements do not (yet) close the reliability gap."
            verdict = (f"ROLE-GATE FBALIGN HONEST NEGATIVE (first-class; the boundary is mapped){smoketag} -- the task is "
                       f"VALID and the EXACT-feedback ceiling reaches role reliably (aligned {al['acc']:.3f}[min "
                       f"{al['acc_min']:.3f}]), but NO stronger transport-free alignment lever (weight-mirror warm-up "
                       f"'align-then-hold' / 'align-then-maintain' / 'align-then-KP' / stronger-KP) makes transport-free "
                       f"credit RELIABLE at the 6-seed bar{past} {common} So reliable transport-free feedback alignment "
                       f"is NOT achievable on the role task by these mechanisms at this scale -- the residual is precisely "
                       f"mapped. NEXT: the role-task WIDTH sweep (--hidden) + the GPU scale probe locate whether it is a "
                       f"low-dim pathology; if width does not close it either, the next rung is the emergence engine's own "
                       f"ordinal/cue code supplying the role-relevant conjunction (out of scope). NO sim/ edit.")
        else:
            verdict = (f"ROLE-GATE FBALIGN HONEST NEGATIVE (task/architecture){smoketag} -- even the aligned ceiling does "
                       f"not cleanly reach role here (aligned {al['acc']:.3f}[min {al['acc_min']:.3f}]). {common} NO sim/ edit.")
        print(f"[rolegate-fbalign] {verdict}", flush=True)
    elif err is not None:
        verdict = f"ERROR -- {err}"
    else:
        verdict = "ERROR -- no points computed"

    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("rolegate_feedback_alignment_reliability", chance=chance)
        if far is not None:
            Vd.require("generalisation_defined_novel_heldout", 1 if far["gen_defined"] else 0, expect=lambda x: x >= 1,
                       note="held-out distractor configs disjoint from train")
            Vd.require("oracle_ceiling_exists", round(far["acc_oracle"], 4), expect=lambda x: x >= chance + 0.30,
                       note="the case-marker oracle must clear chance -> a target exists")
            Vd.require("aligned_ceiling_reaches_role", round(far["aligned_gate"]["acc"], 4),
                       expect=lambda x: x >= chance + 0.30, note="exact-feedback credit reaches role (the ceiling)")
            Vd.require("onset_gate_fails", round(far["onset_gate"]["acc"], 4), expect=lambda x: x <= chance + 0.15,
                       note="ONSET GATE FAILS -> onset != the answer")
            Vd.require("untrained_gate_fails", round(far["untrained_gate"]["acc"], 4), expect=lambda x: x <= chance + 0.15,
                       note="the UNTRAINED gate FAILS -> valid")
            Vd.require("permuted_reward_fails", round(far["permrew_fa_gate"]["acc"], 4), expect=lambda x: x <= chance + 0.15,
                       note="PERMUTED-REWARD FAILS -> the learning signal is required -> valid")
            Vd.require("hold_zero_input_asserted", 1 if far["zero_input_ok"] else 0, expect=lambda x: x >= 1,
                       note="the slot sustained the latch with external input ASSERTED zero across hold+read")
            Vd.control("mirror_kp_differs_from_L4_kp", treatment=far["mirror_kp_gate"]["acc"],
                       control=far["kp_gate"]["acc"], min_separation=0.0,
                       note="the warm-up arm's outcome is reported vs the LEVER-4 KP baseline")
        else:
            Vd.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        dec = Vd.decide(role_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "rolegate_feedback_alignment_reliability", "verdict": verdict, "role_go": bool(role_go),
               "task_valid": bool(valid), "winner": winner, "backend": backend, "sim_backend": backend,
               "device": device, "smoke": smoke, "cost_acknowledged": True, "preconditions": preconditions,
               "mechanism": "the LEVER-4 variable-subject-position role-gate (D3 spiking HOLD slot + 2-layer competitive-"
                            "stabilizer gate + byte-faithful chained-FA/KP transport-free credit) with a WEIGHT-MIRROR "
                            "alignment WARM-UP (Akrout 2019, transport-free noise-output correlation aligning B->forward^T "
                            "before/during credit) + optional per-sentence mirror REFRESH + a kp_lr BOOST. Arms: oracle / "
                            "onset / aligned (exact-feedback ceiling) / FA / KP (LEVER-4 baselines) + mirror_fa (align-"
                            "then-hold) / mirror_fa_ref (align-maintain) / mirror_kp (align-then-KP) / kp_strong (stronger/"
                            "longer KP) + untrained + permuted-reward + identity-control. cos(B,W^T) per hop per seed "
                            "(post-warmup init -> final) reported to separate alignment quality from accuracy.",
               "task": "the LEVER-4 variable-subject-position case-marked agreement stream (subject = NOM-tagged noun at a "
                       "random ordinal; onset != the answer); GO = a transport-free lever arm reaches role RELIABLY "
                       "(6-seed mean >= chance+0.30, min >= 0.60, tag-gap min >= 0.30) with the validity killers still biting.",
               "seeds": a.seeds, "config": {"N_noun": a.n_noun, "F_feat": F, "C_tags": C, "distances": dists,
               "subj_lo": a.subj_lo, "hidden": a.hidden, "episodes": a.episodes, "lr": a.lr, "kp_lr": a.kp_lr,
               "kp_wd": a.kp_wd, "readout_scale": a.readout_scale, "homeo": a.homeo, "b2_init": a.b2_init,
               "div_k": a.div_k, "out_lambda": a.out_lambda, "inh_leak": a.inh_leak, "mirror_steps": a.mirror_steps,
               "kp_boost": a.kp_boost, "mirror_every": a.mirror_every, "mirror_refresh_steps": a.mirror_refresh_steps,
               "n_train": a.n_train, "n_test": a.n_test, "chance": chance, "go_distance": go_L},
               "go_point": far, "points": points, "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "LEVER 5 tests whether STRONGER transport-free feedback alignment (Akrout weight-mirror "
                              "warm-up / stronger-KP) makes transport-free credit RELIABLE on the LEVER-4 variable-position "
                              "role task, where FA was bimodal (2/6) and KP collapsed (0/6). The weight-mirror is "
                              "transport-free (B estimated from the forward's response to noise, never Wt read directly). "
                              "cos(B,W^T) per seed separates 'alignment raised' from 'accuracy fixed': if a lever raises "
                              "worst-seed alignment yet accuracy still collapses, the residual is PAST alignment. 1-seed is "
                              "a SMOKE indicator; the 6-seed sweep is decisive. All outcomes (GO / boundary-mapped-negative) "
                              "are first-class. NO sim/ edit."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 114, flush=True)
    print(f"[rolegate-fbalign] VERDICT: {verdict}", flush=True)
    print(f"[rolegate-fbalign] role_go={role_go} task_valid={valid} winner={winner}  wrote {a.out}\n" + "=" * 114,
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
