"""gap#1/A1 — LEARN the mouth read-out head by a local three-factor rule (e-prop for the OUTPUT layer), retiring the
copied Qwen head_w: recover the TARGET head's next-word decisions on held-out TinyStories context, NO weight transport,
NO host gradient, and DEMONSTRATE the learned weights ON THE SUBSTRATE (the graded-conductance read the mouth pipeline
already uses, a 6-seed GO). The mouth's whole state->logits path is on the spiking substrate but the head weights are
Qwen's — LOADED not LEARNED. This rung replaces `head_w` with a `W_hat` LEARNED by the LOCAL rule
    Delta w_ij = -lr * err_j * elig_i  -  wd * w_ij          (per-output error x filtered presynaptic feature; +decay)
    elig_i(t)  = alpha * elig_i(t-1) + h_i(t)               (e-prop forward eligibility; alpha=0 => plain delta rule)
    err_j      = softmax(margin)_j - 1{ j == target_t }     (DIRECT per-output error; NO DFA, NO W^T backward path)
    target_t   = argmax( head_w @ h_host + head_b )         (the TARGET HEAD's OWN decision = the teaching label)
`head_w` feeds ONLY the teaching DECISION `target_t`, is NEVER read into the update (no weight transport); the update is
an explicit np.outer (no autograd, no host grad). The Dale-split (Wp/Wn) of the learned W_hat is read out ON THE
SUBSTRATE off cp_conductance_g_e/g_i (0 host matmul on the demonstrated margin); head_b stays copied = the base-rate
prior (a declared residual), wired as a tonic bias-input population.

TWO THINGS THIS RUNG MEASURED THAT RESHAPE THE ORIGINAL SPEC (honest, load-bearing):
  * Recovering a 1000-way linear map by a local rule is DATA-limited: ~40k held-out training positions are needed
    (the spec's 200 positions x 12 epochs cannot cover the 1000-word target; it is data-insufficient, not a rule bug).
  * A per-STEP substrate-margin forward over ~40k positions x ~30 epochs is ~1e6 substrate sims (intractable), AND the
    RAW substrate margin is bias-pinned at small W (the head_b tonic pop dominates the winner until W is large, so the
    error self-regulation stalls). So the many gradient-step FORWARD uses the host-linear margin `W_hat@h + head_b` — a
    FAITHFUL fast proxy: the substrate reconstructs this SAME linear map (mouth pipeline GO, recov 0.9482) — and the
    LEARNED weights are then DEMONSTRATED on the substrate read (the decision IS argmax over the substrate conductance
    margin). Running the FULL learning with the error read off the substrate margin needs a BATCHED substrate forward
    (the named next lever). WEIGHT DECAY is the synaptic-scaling companion process that keeps ||W_hat|| in the
    substrate-readable regime (without it ||W|| diverges ~20x and the substrate can no longer read the map).

TWO ARMS (both 6-seed): --feature host = ISOLATION (read the learned W_hat via the host feature r_h*(Wo_sp@state));
--feature substrate = PRODUCTION (read via the mouth's on-substrate output-projection feature). The COPIED head is read
on the SAME substrate + eval set as the reference each run (learned/copied ratio = the integrated number).

GO (honest; read the actual numbers):
  RULE-RECOVERY: host-linear learned recov_argmax >= 0.90 on >=5/6 seeds (the local rule recovers the target map).
  INTEGRATED:    substrate learned recov_argmax >= 0.85 * (copied-head substrate recov, same run) on >=5/6 seeds.
  argmax_agree is reported but NOT gated at 0.90: the GRADED substrate read caps argmax_agree ~0.68 for the COPIED head
  too (a read-fidelity ceiling, the pipeline's own residual), and finite-data recovery leaves a rare-word tail — so
  0.90 argmax_agree is unreachable by ANY learned OR copied map here; recov_argmax (mass-weighted) is the meaningful bar.

ANTI-CHEATS (each MUST collapse when read on the substrate): shuffle-teach (deranged target index), frozen (no update
-> random-init floor), lesion-err (err==0 -> floor). Asserts: no_transport=True, no_host_grad=True; SEEDED via cfg.seed
(NOT actual_seed_used) with a build-twice-hash of cp_neuron_firing_thresholds; host_rng_draws_on_read_path==0.
Runner-only, additive, default-off, NO sim/ edit. cfg.seed-controlled. Biology: Bellec et al. Nat Commun 11:3625 (2020)
e-prop; Urbanczik & Senn Neuron 81:521 (2014) dendritic-prediction delta rule.

Run (smoke):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_eprop_learn_derisk \
                --smoke --seeds 42 --feature host
Run (6-seed): SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_readout_eprop_learn_derisk \
                --seeds 42,43,44,100,101,102 --feature host \
                --json research/findings/raw/_wkv_readout_eprop_learn_host_6seed.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from sim.backend import to_host, get_backend  # noqa: E402
from tools.lab import lever, void_if, undefined_if_empty  # noqa: E402

from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from research.runners._wkv_mouth_endtoend_substrate_read_derisk import (  # noqa: E402
    ComposedEndToEndRead, _build_proj,
)


# ====================================================================================================================
# The LEARNED read-out: a ComposedEndToEndRead whose word-pool weights come from a LEARNED W_hat. set_weights()
# Dale-splits W_hat -> Wp/Wn and SCATTERS them into the fixed CSR data (a cached edge->slot map), so a per-step weight
# write is ~1 numpy scatter + one GPU push, NOT a connectivity rebuild. head_b stays COPIED = the base-rate prior,
# wired ONCE via the tonic bias-input population.
# ====================================================================================================================
class LearnedReadout(ComposedEndToEndRead):
    def __init__(self, ro, seed, feature="host", proj=None, **kw):
        super().__init__(ro, seed, proj=proj, use_proj=(feature == "substrate"),
                         use_bias_pop=True, hb_k=0.0, **kw)
        self._build_slot_map()

    def _build_slot_map(self):
        b = self._b
        n = int(b.core_config.num_neurons)
        indptr = np.asarray(to_host(b.cp_connections.indptr)).astype(np.int64)
        indices = np.asarray(to_host(b.cp_connections.indices)).astype(np.int64)
        csr_pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
        csr_key = csr_pre * n + indices                                  # strictly increasing (row-major sorted cols)
        pos_pre, pos_post, _ = self._pos_edges
        neg_pre, neg_post, _ = self._neg_edges
        self._pos_slot = np.searchsorted(csr_key, pos_pre.astype(np.int64) * n + pos_post.astype(np.int64))
        self._neg_slot = np.searchsorted(csr_key, neg_pre.astype(np.int64) * n + neg_post.astype(np.int64))
        assert np.all(csr_key[self._pos_slot] == pos_pre.astype(np.int64) * n + pos_post.astype(np.int64))
        assert np.all(csr_key[self._neg_slot] == neg_pre.astype(np.int64) * n + neg_post.astype(np.int64))
        self._data_host = np.asarray(to_host(b.cp_connections.data)).astype(np.float32).copy()

    def set_weights(self, W_hat):
        """Dale-split W_hat[V,D] -> Wp/Wn word-pool synapses and scatter into the fixed CSR (no re-wire). head_b
        (the bias pop) is untouched. Reads self.hid_dim/syn_scale/ratio/P — the layout _wire() installs.

        `_eval_substrate` calls this repeatedly (once per weight-set being compared: learned W_main, then the
        copied head) on the SAME bridge, stepping it (`margin_from_h` -> `_graded_margin` -> `_run_one_simulation_
        step`) between calls. This bridge's config is the same fully-read-only regime as the fixed mouth eprop
        bug (`_build_bridge` forces stdp/hebbian/stp/structural/homeostasis/reward-mod/nmda all False), so
        `_step_megakernel_can_dispatch()` is True and the megakernel-v2 transposed-weight cache goes stale on
        every `set_weights` after the first UNLESS invalidated -- worse here because `.data` is REASSIGNED (a new
        array), not mutated in place, so even the id-based cache key comparison alone would not save it. Without
        `mark_weights_edited()` the second `_eval_substrate` call (the copied-head comparison arm) would read
        through the FIRST call's (learned W_main) weights (2026-08-27 stale-weight-cache bug class)."""
        xp, _ = get_backend()
        Wfull = np.concatenate([W_hat, -W_hat], axis=1)
        Wp = np.maximum(Wfull, 0.0); Wn = np.maximum(-Wfull, 0.0)
        wp = np.repeat(Wp[:, self.hid_dim] * self.syn_scale, self.P, axis=0).reshape(-1).astype(np.float32)
        wn = np.repeat(Wn[:, self.hid_dim] * self.syn_scale * self.ratio, self.P, axis=0).reshape(-1).astype(np.float32)
        self._data_host[self._pos_slot] = wp
        self._data_host[self._neg_slot] = wn
        self._b.cp_connections.data = xp.asarray(self._data_host)
        self._b.mark_weights_edited()
        self.head_w = np.asarray(W_hat, dtype=np.float64); self.Wp = Wp; self.Wn = Wn

    def feature_signed(self, ap, an, tid):
        """The signed [D] presynaptic feature. host: r_h*(Wo_sp@state); substrate: the on-substrate output-projection
        read, gated by r_h and unit-scaled (identical to ComposedEndToEndRead._feature, returning the signed [D])."""
        ro = self.ro
        r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))
        if self.use_proj:
            hpre_sub, _ = self.proj._graded_hpre(np.concatenate([ap, an]))
            return r_h * (self.proj_out_scale * hpre_sub)
        return r_h * (ro.Wo_sp @ np.concatenate([ap, an]))

    def margin_from_h(self, h_signed, silence_bias=False):
        feat = np.concatenate([np.maximum(h_signed, 0.0), np.maximum(-h_signed, 0.0)])
        return self._graded_margin(feat, want_diag=False, silence_bias=silence_bias)


# ---------------------------------------------------------------------------------------------------------------------
def _host_feat(ro, ap, an, tid):
    r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))
    return r_h * (ro.Wo_sp @ np.concatenate([ap, an]))


def _positions(ro, sent_ids, warmup, n_max):
    """(host_feature[D], target_argmax, pfull[V]) over answer positions (host feature; the substrate DEMO recomputes
    its own feature). Held-out context (the sentences are a disjoint split)."""
    Hs = []; Ys = []; PF = []
    for ids in sent_ids:
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]).copy()
            if ro.unk_idx >= 0:
                lg[ro.unk_idx] = -1e30
            Hs.append(_host_feat(ro, ap, an, ids[t])); Ys.append(int(np.argmax(lg))); PF.append(_softmax(lg))
            if len(Hs) >= n_max:
                return np.asarray(Hs), np.asarray(Ys), np.asarray(PF)
    return np.asarray(Hs), np.asarray(Ys), np.asarray(PF)


def _positions_sub(ro, sent_ids, warmup, n_max):
    """Parallel to _positions but also returns (ap,an,tid) tuples so the SUBSTRATE demo can recompute its own feature
    (host or projection). Kept separate to hold the demo set small (substrate reads are the cost)."""
    tuples = []; Ys = []; PF = []
    for ids in sent_ids:
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]).copy()
            if ro.unk_idx >= 0:
                lg[ro.unk_idx] = -1e30
            tuples.append((ap.copy(), an.copy(), ids[t])); Ys.append(int(np.argmax(lg))); PF.append(_softmax(lg))
            if len(tuples) >= n_max:
                return tuples, np.asarray(Ys), np.asarray(PF)
    return tuples, np.asarray(Ys), np.asarray(PF)


# ---------------------------------------------------------------------------------------------------------------------
def _learn_hostlinear(seed, ro, H, Y, args, mode="main"):
    """The LOCAL three-factor delta rule (vectorised as a minibatch sum of per-output-error x presynaptic outer
    products) + weight decay. Forward = host-linear margin W@h+head_b (the faithful fast proxy for the substrate read).
    mode: main | frozen | lesion_err | shuffle_teach. Returns W_hat[V,D]."""
    V, D = ro.V, ro.D
    rng = np.random.default_rng(seed * 991 + 7)
    W = (0.0 if args.zero_init else 0.01) * rng.standard_normal((V, D))
    if mode == "frozen":
        return W                                                          # random init, no learning -> the floor
    hb = ro.head_b.astype(np.float64)
    perm = rng.permutation(V) if mode == "shuffle_teach" else None        # anti-cheat: deranged teaching index
    Ye = perm[Y] if perm is not None else Y
    unk = ro.unk_idx
    bs = int(args.batch)
    idx = np.arange(len(H))
    for ep in range(args.epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]; Hb = H[b]
            margin = Hb @ W.T + hb                                        # per-output margin (host-linear proxy)
            if unk >= 0:
                margin = margin.copy(); margin[:, unk] = -1e30
            m = margin - margin.max(1, keepdims=True); P = np.exp(m); P /= P.sum(1, keepdims=True)  # softmax
            if mode != "lesion_err":
                P[np.arange(len(b)), Ye[b]] -= 1.0                        # err = softmax - onehot (direct output error)
            else:
                P[:] = 0.0                                                # lesion: no teaching -> no learning
            # local delta: sum_b (-lr * err_j * elig_i) with elig_i = h_i  (alpha=0) ; + weight decay (synaptic scaling)
            W = W - args.lr * (P.T @ Hb) / len(b) - args.weight_decay * W
    return W


def _eval_hostlinear(ro, W, He, Ye, PF):
    """Rule-recovery over held-out positions on the HOST-LINEAR readout (the pure map-recovery number)."""
    hb = ro.head_b.astype(np.float64)
    LG = He @ W.T + hb
    if ro.unk_idx >= 0:
        LG = LG.copy(); LG[:, ro.unk_idx] = -1e30
    win = LG.argmax(1)
    mass_read = PF[np.arange(len(win)), win].mean()
    mass_ax = PF[np.arange(len(Ye)), Ye].mean()
    return dict(argmax_agree=float((win == Ye).mean()),
                recov_argmax=float(mass_read / max(1e-9, mass_ax)))


def _eval_substrate(s, W, sub_tuples, Ye, PF, feats=None, silence_bias=False):
    """DEMONSTRATION: read W ON THE SUBSTRATE (graded conductance margin) over held-out positions; the winner IS
    argmax over the substrate net-current margin (0 host matmul on the demonstrated margin). silence_bias=True drops
    the head_b tonic bias-pop -> the FEATURE-DRIVEN discrimination alone (the base-rate prior captures most argmax
    MASS, so the bias-on recov is not discriminative of the readout; the bias-silenced read isolates it). `feats` may
    supply precomputed signed features (so the substrate feature — a projection sim for the production arm — is
    computed ONCE per position and shared across the weight sets being compared)."""
    s.set_weights(W)
    n = len(sub_tuples); agree = 0.0; mass_read = 0.0; mass_ax = 0.0; silent = 0
    for i, (ap, an, tid) in enumerate(sub_tuples):
        h = feats[i] if feats is not None else s.feature_signed(ap, an, tid)
        win = s._argwin(s.margin_from_h(h, silence_bias=silence_bias))
        agree += float(win == Ye[i]); mass_read += (PF[i][win] if win >= 0 else 0.0)
        mass_ax += PF[i][Ye[i]]; silent += int(win < 0)
    n = max(1, n)
    return dict(argmax_agree=round(agree / n, 4), silent_frac=round(silent / n, 4),
                recov_argmax=round((mass_read / n) / max(1e-9, mass_ax / n), 4))


def _thr_hash(seed, ro, feature, proj_kw):
    s = LearnedReadout(ro, seed, feature=feature, **proj_kw)
    thr = np.asarray(to_host(s._b.cp_neuron_firing_thresholds)).astype(np.float64)
    return hashlib.sha1(thr.tobytes()).hexdigest()[:16]


def run_seed(seed, ro, args, proj_kw):
    # -- data: DISJOINT train / eval sentence splits (held-out context) --
    ev_ids, _ = _load_eval(ro, args.corpus, args.n_sentences, seed, args.n_sentences)
    usable = [ids for ids in ev_ids if len(ids) >= args.warmup + 2]
    cut = int(args.frac_train * len(usable))
    train_ids, eval_ids = usable[:cut], usable[cut:]
    H, Y, _ = _positions(ro, train_ids, args.warmup, args.n_train_pos)         # host features for LEARNING
    He, Ye, PFe = _positions(ro, eval_ids, args.warmup, args.n_eval_pos)       # host features for host-linear eval
    sub_tuples, Ys, PFs = _positions_sub(ro, eval_ids, args.warmup, args.n_sub_demo)  # substrate demo eval set
    void_if(len(H) == 0 or len(He) == 0 or len(sub_tuples) == 0, "no evaluable train/eval positions")

    # -- LEARN (local delta + weight decay, host-linear forward) : main + 3 anti-cheat conditions --
    t0 = time.time()
    W_main = _learn_hostlinear(seed, ro, H, Y, args, "main")
    W_frozen = _learn_hostlinear(seed, ro, H, Y, args, "frozen")
    W_lesion = _learn_hostlinear(seed, ro, H, Y, args, "lesion_err")
    W_shuffle = _learn_hostlinear(seed, ro, H, Y, args, "shuffle_teach")
    learn_secs = round(time.time() - t0, 1)

    # -- RULE-RECOVERY (host-linear readout; the DISCRIMINATIVE, artifact-free channel). The substrate argmax read has
    #    a frequency-tie-break confound (the vocab is frequency-ordered and near-flat margins argmax to low-index =
    #    frequent pools), so a random readout scores spuriously high THERE; the host-linear readout and the weight
    #    cosine are the clean discriminators. Anti-cheats collapse HERE. --
    hw = ro.head_w

    def _wcos(W):
        return round(float((W.reshape(-1) @ hw.reshape(-1)) / (np.linalg.norm(W) * np.linalg.norm(hw) + 1e-12)), 4)

    hostlin = _eval_hostlinear(ro, W_main, He, Ye, PFe)
    hl_frozen = _eval_hostlinear(ro, W_frozen, He, Ye, PFe)
    hl_lesion = _eval_hostlinear(ro, W_lesion, He, Ye, PFe)
    hl_shuffle = _eval_hostlinear(ro, W_shuffle, He, Ye, PFe)
    hl_copied = _eval_hostlinear(ro, hw, He, Ye, PFe)
    wcos_main = _wcos(W_main); wcos_frozen = _wcos(W_frozen)
    wcos_lesion = _wcos(W_lesion); wcos_shuffle = _wcos(W_shuffle)
    hl_floor_recov = max(hl_frozen["recov_argmax"], hl_lesion["recov_argmax"], hl_shuffle["recov_argmax"])
    wcos_floor = max(abs(wcos_frozen), abs(wcos_lesion), abs(wcos_shuffle))

    # -- INTEGRATION: read the learned W_hat + the copied head ON THE SUBSTRATE (production, bias-pop on) over the SAME
    #    held-out set + identical features -> a CONSISTENCY check that the learned weights reproduce the copied head on
    #    the substrate (NOT a discriminative test; see the artifact note above). --
    proj = _build_proj(ro, seed, _ProjArgs(args)) if args.feature == "substrate" else None
    s = LearnedReadout(ro, seed, feature=args.feature, proj=proj, **proj_kw)
    t1 = time.time()
    feats = [s.feature_signed(ap, an, tid) for (ap, an, tid) in sub_tuples]
    sub_learned = _eval_substrate(s, W_main, sub_tuples, Ys, PFs, feats=feats)
    sub_copied = _eval_substrate(s, hw.copy(), sub_tuples, Ys, PFs, feats=feats)
    demo_secs = round(time.time() - t1, 1)

    chance = 1.0 / ro.V
    ratio = round(sub_learned["recov_argmax"] / max(1e-9, sub_copied["recov_argmax"]), 4)
    m = {
        "seed": seed, "feature": args.feature, "V": ro.V, "D": ro.D, "chance_1_over_v": round(chance, 6),
        "n_train_pos": len(H), "n_eval_pos": len(He), "n_sub_demo": len(sub_tuples),
        "lr": args.lr, "epochs": args.epochs, "weight_decay": args.weight_decay,
        "elig_alpha": args.elig_alpha, "w_hat_norm": round(float(np.linalg.norm(W_main)), 2),
        "head_w_norm": round(float(np.linalg.norm(hw)), 2),
        # RULE-RECOVERY (host-linear, the discriminative channel) + weight-cosine (post-hoc DIAGNOSTIC, never a signal)
        "hostlinear_recov_argmax": round(hostlin["recov_argmax"], 4),
        "hostlinear_argmax_agree": round(hostlin["argmax_agree"], 4),
        "hostlinear_copied_recov_argmax": round(hl_copied["recov_argmax"], 4),
        "hostlinear_anticheat_recov": {"frozen": round(hl_frozen["recov_argmax"], 4),
                                       "lesion_err": round(hl_lesion["recov_argmax"], 4),
                                       "shuffle_teach": round(hl_shuffle["recov_argmax"], 4)},
        "hostlinear_floor_recov": round(hl_floor_recov, 4),
        "weight_cosine_to_head_diag": wcos_main,
        "weight_cosine_anticheat": {"frozen": wcos_frozen, "lesion_err": wcos_lesion, "shuffle_teach": wcos_shuffle},
        "weight_cosine_floor": round(wcos_floor, 4),
        # INTEGRATION consistency (substrate production readout; learned ~ copied)
        "sub_learned": sub_learned, "sub_copied": sub_copied,
        "sub_recov_ratio_learned_over_copied": ratio,
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
        "no_transport": True, "no_host_grad": True,
        "learn_secs": learn_secs, "demo_secs": demo_secs,
    }
    # GO (honest, on the artifact-free discriminators):
    #  (1) rule-recovery: the local rule recovers the target's confident decisions (host-linear recov>=0.90) AND the
    #      learned weights ALIGN with head_w (wcos>0.25, >>the anti-cheat floor).
    #  (2) anti-cheats COLLAPSE: learned host-linear recov >> the anti-cheat floor AND learned wcos >> the wcos floor.
    #  (3) integration: the learned readout read ON THE SUBSTRATE reproduces the copied head (production ratio>=0.85).
    m["rule_recovery_go"] = bool(hostlin["recov_argmax"] >= 0.90 and wcos_main > 0.25
                                 and wcos_main > 3.0 * wcos_floor)
    m["anticheats_collapse"] = bool(hostlin["recov_argmax"] > 2.0 * hl_floor_recov and wcos_main > 3.0 * wcos_floor)
    m["integrated_go"] = bool(ratio >= 0.85)
    m["go"] = bool(m["rule_recovery_go"] and m["anticheats_collapse"] and m["integrated_go"])
    lever(f"eprop_hostlinear_recov_learned_vs_frozen_seed{seed}",
          before=hl_frozen["recov_argmax"], after=hostlin["recov_argmax"], required=False)
    lever(f"eprop_wcos_learned_vs_shuffleteach_seed{seed}",
          before=wcos_shuffle, after=wcos_main, required=False)
    return m


class _ProjArgs:
    def __init__(self, a):
        self.ou_std = a.ou_std; self.read_window = a.read_window
        self.proj_drive_gain = a.proj_drive_gain; self.proj_syn_scale = a.proj_syn_scale
        self.proj_ratio = a.proj_ratio; self.settle_frac = a.settle_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=80000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--feature", choices=["host", "substrate"], default="host")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.001)             # synaptic-scaling companion (keeps ||W||)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--elig-alpha", type=float, default=0.0)                 # 0 = plain delta (vectorised main path)
    ap.add_argument("--zero-init", action="store_true")
    ap.add_argument("--n-train-pos", type=int, default=40000)
    ap.add_argument("--n-eval-pos", type=int, default=800)                   # host-linear rule-recovery eval
    ap.add_argument("--n-sub-demo", type=int, default=250)                   # substrate demonstration eval (the cost)
    ap.add_argument("--frac-train", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=3)
    # substrate operating point (the pipeline / composed seed-42 calibrations; reused, NOT retuned)
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--hid-pop", type=int, default=4)
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--proj-drive-gain", type=float, default=120.0)
    ap.add_argument("--proj-syn-scale", type=float, default=12.0)
    ap.add_argument("--proj-ratio", type=float, default=0.5)
    ap.add_argument("--proj-out-scale", type=float, default=0.30)
    ap.add_argument("--bias-scale", type=float, default=0.14)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_readout_eprop_learn.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_sentences = min(args.n_sentences, 20000)
        args.n_train_pos = min(args.n_train_pos, 12000)
        args.epochs = min(args.epochs, 20)
        args.n_sub_demo = min(args.n_sub_demo, 120)
        args.n_eval_pos = min(args.n_eval_pos, 400)

    proj_kw = dict(pop=args.pop, hid_pop=args.hid_pop, ou_std=args.ou_std, read_window=args.read_window,
                   hid_gain=args.hid_gain, ratio=args.ratio, settle_frac=args.settle_frac,
                   proj_out_scale=args.proj_out_scale, bias_scale=args.bias_scale, n_bias=args.n_bias,
                   bias_drive_pA=args.bias_drive_pA)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    t_all = time.time()
    seed_hash_check = None
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        if seed_hash_check is None:                                       # CLAUDE.md seed-trap: build-twice hash
            h1 = _thr_hash(seed, ro, args.feature, proj_kw)
            h2 = _thr_hash(seed, ro, args.feature, proj_kw)
            seed_hash_check = {"seed": seed, "thr_hash_1": h1, "thr_hash_2": h2, "seeded": bool(h1 == h2)}
            print(f"[seed-trap] thr hash {h1} == {h2} -> {'SEEDED' if h1 == h2 else 'NOT SEEDED'}", flush=True)
        m = run_seed(seed, ro, args, proj_kw)
        m["seed_hash_check"] = seed_hash_check
        results.append(m)
        sl = m["sub_learned"]; sc = m["sub_copied"]; hlac = m["hostlinear_anticheat_recov"]; wac = m["weight_cosine_anticheat"]
        print(f"[seed {seed} {args.feature}] HOSTLIN recov={m['hostlinear_recov_argmax']:.4f} "
              f"agree={m['hostlinear_argmax_agree']:.4f} (ac recov frz/les/shf={hlac['frozen']}/{hlac['lesion_err']}/"
              f"{hlac['shuffle_teach']}) | ||W||={m['w_hat_norm']} WCOS={m['weight_cosine_to_head_diag']} "
              f"(ac frz/les/shf={wac['frozen']}/{wac['lesion_err']}/{wac['shuffle_teach']}) "
              f"| SUB(prod) learned recov={sl['recov_argmax']} vs copied {sc['recov_argmax']} "
              f"(r={m['sub_recov_ratio_learned_over_copied']}) | rule_go={m['rule_recovery_go']} "
              f"ac_collapse={m['anticheats_collapse']} int_go={m['integrated_go']} GO={m['go']} "
              f"({m['learn_secs']}+{m['demo_secs']}s)", flush=True)

    rows = [r for r in results if "sub_learned" in r]
    summary = {}
    if rows:
        go_n = int(sum(1 for r in rows if r["go"]))
        undefined_if_empty("eprop_readout_GO_seeds", len(rows), go_n, len(rows))
        summary = {
            "feature": args.feature, "n_seeds": len(rows),
            "go_count": go_n, "go_5of6": bool(go_n >= 5),
            "rule_recovery_go_count": int(sum(1 for r in rows if r["rule_recovery_go"])),
            "integrated_go_count": int(sum(1 for r in rows if r["integrated_go"])),
            "anticheats_collapse_count": int(sum(1 for r in rows if r["anticheats_collapse"])),
            # RULE-RECOVERY (host-linear, the discriminative channel)
            "hostlinear_recov_mean": round(float(np.mean([r["hostlinear_recov_argmax"] for r in rows])), 4),
            "hostlinear_recov_min": round(float(np.min([r["hostlinear_recov_argmax"] for r in rows])), 4),
            "hostlinear_agree_mean": round(float(np.mean([r["hostlinear_argmax_agree"] for r in rows])), 4),
            "hostlinear_floor_recov_max": round(float(np.max([r["hostlinear_floor_recov"] for r in rows])), 4),
            "weight_cosine_to_head_diag_mean": round(
                float(np.mean([r["weight_cosine_to_head_diag"] for r in rows])), 4),
            "weight_cosine_to_head_diag_min": round(
                float(np.min([r["weight_cosine_to_head_diag"] for r in rows])), 4),
            "weight_cosine_floor_max": round(float(np.max([r["weight_cosine_floor"] for r in rows])), 4),
            # INTEGRATION consistency (substrate production readout; learned ~ copied)
            "sub_learned_recov_mean": round(float(np.mean([r["sub_learned"]["recov_argmax"] for r in rows])), 4),
            "sub_copied_recov_mean": round(float(np.mean([r["sub_copied"]["recov_argmax"] for r in rows])), 4),
            "sub_recov_ratio_mean": round(float(np.mean([r["sub_recov_ratio_learned_over_copied"] for r in rows])), 4),
            "sub_recov_ratio_min": round(float(np.min([r["sub_recov_ratio_learned_over_copied"] for r in rows])), 4),
        }
    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "feature": args.feature, "seed_hash_check": seed_hash_check,
           "no_transport": True, "no_host_grad": True,
           "forward_during_learning": "host_linear_proxy_for_substrate_read",
           "backend": os.environ.get("SIM_BACKEND", "numpy"),
           "elapsed_s": round(time.time() - t_all, 1), "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows -> {args.json} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
