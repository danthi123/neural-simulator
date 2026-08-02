"""gap#4 Lane C -- CREDIT ON TOP OF A REPRESENTABLE FORWARD: put the 2026-07-25 coincidence-plateau
expander UNDERNEATH the ON-BRIDGE e-prop credit rule and ask whether training the HIDDEN feedforward
pathways then MATTERS (deep_credit_share > 0) where on the plain spiking forward it did not (~0).

WHY THIS EXPERIMENT (named as the highest-value never-run test, 2026-08-01). gap#4 has two halves that were
each solved in isolation and NEVER combined:

  * FORWARD representability -- SURPASSED on-bridge 2026-07-25. `PlateauExpander` (a fixed decorrelated
    coincidence dendritic-PLATEAU expansion on a real `SimulationBridge`) lifts held-out LINEAR decodability
    of the compositional-inheritance task from the 0.34 boundary to 0.611 (reproducibility 1.000, 6 seeds).
    Its own title: "so the CPU-rate-GO credit has features to shape".
  * CREDIT on-bridge -- the ported e-prop rule (`OnBridgeEpropNet`, Izhikevich two-compartment substrate)
    TRAINS the forward map, but its OWN frozen-hidden reservoir control (`reservoir_control`, added
    2026-07-16) shows it is ~a FIXED RANDOM SPIKING RESERVOIR: `deep_credit_share` = 0.005 at pool_k=16
    (`2026-08-01-gap4-6seed-bar-...shuffleDFA-leaks-...`). Training the hidden pathways adds ~nothing.

`PlateauExpander` is imported by exactly ONE file -- its own probe. The on-bridge e-prop credit runner has
NEVER seen it: every deep_credit_share on record was measured against the PLAIN spiking forward, the one the
07-25 finding diagnosed as frozen. This runner is the missing combination.

THE ONE VARIABLE: the forward representation fed as input current to `OnBridgeEpropNet`.
  mode `raw`      -- the task features straight in (the plain spiking forward; the ~0 deep_credit_share cond.)
  mode `expander` -- the SAME features through `PlateauExpander` first (the 2026-07-25 surpassed forward),
                     the plateau CODON fed as the input-current features. EVERYTHING downstream is identical:
                     same OnBridgeEpropNet, same e-prop DFA credit rule, same hidden depth, same seeds, same
                     reservoir_control / permuted / shuffle-DFA anti-cheats.

Both the expander AND the credit are neurons/synapses/dendrites on a spiking `SimulationBridge` (the expander
is a coincidence-plateau bridge; the credit is the Izhikevich bridge). NOTE / DOCUMENTED SHORTCUT: this is
TWO bridges composed (codon precomputed on bridge A, fed as input current to bridge B), not one integrated
brain -- the same two-stage composition the 07-25 finding and `_gap4_*_on_expanded_forward` runners use. The
one-brain integration (a single substrate whose early layers ARE the coincidence expansion) is the follow-on;
this de-risk isolates whether a representable forward makes hidden-layer credit matter at all.

JUDGING CRITERION (owner steer 2026-08-01): judge deep credit by the CAPABILITY that depends on it -- the
substrate LEARNING hierarchical structure a fixed random projection can't (held-out compositional
generalization). GO: with the expander, `deep_credit_share` clearly > 0 (say > 0.15) AT AN n_prop WHERE THE
FROZEN-HIDDEN RESERVOIR FAILS (froz_inh near chance) -- vs ~0 without the expander. If the expander instead
makes the codon so linearly separable that the frozen reservoir ALSO solves it, deep_credit_share -> ~0
HONESTLY (the expansion solved the task rather than enabling deep credit) -- that is a valid negative, and the
frozen_hidden_inherit column is what distinguishes the two. Task = `make_task_semantic_inheritance` with the
DEPTH knob n_prop in {3,4} (n_super=24), where a big random reservoir should FAIL (n_prop=2 is reservoir-solvable).

ANTI-CHEATS (reused from the e-prop port, UNCHANGED):
  * frozen-hidden reservoir_control -> deep_credit_share (the WHOLE point).
  * permuted-label -> ~chance (no leakage / teacher-contingency).
  * shuffle-DFA -> ~chance (the DFA credit route is load-bearing, not the forward alone).
  * oracle (depth-2 rate DendriticMLP fit on the SAME representation) -> the ceiling / task-still-learnable check.

Reuse-by-import; NO `sim/` edit. CPU: SIM_BACKEND=numpy + OPENBLAS_NUM_THREADS=1.

NOTE ON THE IMPORT: `_gap4_plateau_expander_probe` runs its full 6-seed experiment at MODULE level, so
importing `PlateauExpander` executes that experiment (~15 s CPU) and prints its results; its stdout is captured
here. That file is not ours to edit.

Run (SMOKE -- one (n_prop, mode) per process, parallelize across them):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._gap4_representable_forward_plus_credit_derisk \
        --seeds 42 --n-prop 3 --mode expander --epochs 60 --train-subsample 160 \
        --out research/findings/raw/gap4/rep_fwd_credit_np3_exp_s42.json

THE 2026-08-02 XOR LEVER (ADDITIVE, default OFF). The 2026-08-02 finding established that on the RAW Izhikevich
forward, e-prop's deep_credit_share is ~0 on XOR because the on-bridge forward cannot train XOR at all (the
oracle solves it, e-prop sits at chance) -- the residual is the FORWARD, not the credit rule. The named
highest-value lever is credit ON TOP OF the REPRESENTABLE (plateau-expanded) forward, on XOR. `--task-xor`
swaps the task; the PlateauExpander + e-prop + reservoir_control + deep_credit_share + permuted + shuffle-DFA
machinery is otherwise UNCHANGED. The +/-1 XOR bits become the expander's active-feature SET via a bit-faithful
reader (`--xor-encoding literal`, the fair signed-literal basis; see _xor_active_sets), since topk_active is
meaningless on equal-valued +/-1 bits. Decisive gate (deliverable d): deep_credit_share > 0.3 AND frozen <
eprop AND shuffle-DFA <= chance+0.10 AND permuted ~chance AND trains_the_task.
    # 1-seed SMOKE (expander only -- first deep_credit_share read on the representable XOR forward):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._gap4_representable_forward_plus_credit_derisk \
        --task-xor --seeds 42 --mode expander --epochs 60 --train-subsample 160 \
        --out research/findings/raw/gap4/rep_fwd_credit_xor_smoke_s42.json

THE 2026-08-02 LEARNED-FEEDBACK (KP) LEVER (ADDITIVE, default OFF). The fixed-DFA XOR run above sat at chance on the
sparse representable codon: the 2026-08-02 finding located the wall at the on-bridge e-prop CREDIT RULE -- the LOCAL
biological rule cannot find the weights on Izhikevich, and its DFA feedback is FIXED-random (misaligned). The roadmap's
named fix: LEARN the DFA feedback B_direct via Kolen-Pollack (`--learned-feedback`) so it tracks W^T in DIRECTION,
transport-free. Decisive gate (deliverable d): does e-prop now TRAIN XOR (eprop > chance AND > frozen) on the sparse
representable codon where FIXED DFA gave chance? YES => learned feedback is the fix (deep_credit_share then > 0); NO =>
the residual is deeper (surrogate/eligibility on Izhikevich, phi'-vanishing, operating-point) -- name it.
    # 1-seed SMOKE (KP learned feedback, sparse codon --act-th 3 -- run FIRST):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._gap4_representable_forward_plus_credit_derisk \
        --task-xor --act-th 3 --learned-feedback --mode expander --seeds 42 --epochs 60 --train-subsample 160 \
        --out research/findings/raw/gap4/rep_fwd_credit_xor_kp_smoke_s42.json
    # 6-seed (only if the smoke shows B_direct moving + eprop lifting off chance):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._gap4_representable_forward_plus_credit_derisk \
        --task-xor --act-th 3 --learned-feedback --mode expander --seeds 42 43 44 100 101 102 --epochs 60 \
        --train-subsample 160 --out research/findings/raw/gap4/rep_fwd_credit_xor_kp_6seed.json
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # BEFORE importing the probe (it setdefaults cupy)
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import logging  # noqa: E402
logging.disable(logging.INFO)

# reuse-by-import: the ported on-bridge e-prop net + its trainer + softmax (the credit machinery, VERBATIM).
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet, _train_eprop, _softmax  # noqa: E402
# reuse-by-import: the task + the rate oracle for the ceiling.
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, _train_oracle, _acc_on)
# reuse-by-import: the NON-reservoir-decodable depth-2 XOR->threshold task (the 2026-08-02 lever). Same 4-tuple
# interface as make_task_semantic_inheritance ((Xtr,ytr,Ltr),(Xte,yte,Lte),meta,idx; idx["inh_idx"]=the whole
# held-out set). Additive: only consulted under --task-xor. XOR is provably NOT decodable from a fixed random
# projection, so a frozen-hidden reservoir MUST underperform a trained hidden IF the credit is real -- the task
# on which deep_credit_share can rise off the ~0 it reads on the linearly-reservoir-decodable inheritance task.
from research.runners._gap4_bptt_snn_chained_fa_transport_free_derisk import make_task_xor  # noqa: E402
from sim.dendritic_mlp import DendriticMLP  # noqa: E402

# the plateau expander RUNS its 6-seed experiment on import -> capture the banner (that file is not ours to edit).
_probe_banner = io.StringIO()
with contextlib.redirect_stdout(_probe_banner):
    from research.runners._gap4_plateau_expander_probe import (  # noqa: E402
        PlateauExpander, topk_active, TOPK as PROBE_TOPK)

OUT = _REPO / "research" / "findings" / "raw" / "gap4" / "rep_fwd_credit.json"


def _expand(exp, X, topk):
    """Feature vector -> plateau CODON, reading active features exactly as the validated probe does
    (`topk_active`: a fixed active-count, so the codon is not also encoding how many features cleared a
    threshold -- the third defect the earlier expanded-forward runner had to fix)."""
    return np.asarray([exp.codon(a) for a in topk_active(X, topk)], dtype=np.float64)


def _codon_reproducibility(exp, X, topk, n=8):
    """Read the same rows TWICE; fraction byte-identical. The expander's whole claim is reliability; a verdict
    off an unreliable codon is a verdict about noise."""
    acts = topk_active(X[:n], topk)
    a = np.asarray([exp.codon(s) for s in acts]); b = np.asarray([exp.codon(s) for s in acts])
    return float(np.mean([float((a[i] == b[i]).all()) for i in range(len(acts))]))


# ---- XOR-task active-feature READER (the one integration subtlety, only under --task-xor) -----------------------
# PlateauExpander.codon takes a SET of active feature indices, not a real-valued vector. `topk_active` picks the
# top-k features by MAGNITUDE -- correct for the inheritance task's real-valued X, but MEANINGLESS on the XOR
# task's +/-1 bits (every ON bit has the identical value +1, so "top-4" is an arbitrary tie-break that discards
# which bits are on and never sees the OFF bits). So the +/-1 bit vectors get a bit-faithful reader instead.
def _xor_active_sets(X, encoding):
    """+/-1 XOR bit rows -> list of active-feature index SETS + the feature count the expander must be built with.
      'literal' (DEFAULT, the FAIR representation): each bit i -> TWO literal slots, 2i='bit i ON', 2i+1='bit i
        OFF'; the active set holds EXACTLY ONE literal per bit (n_bits active of 2*n_bits). This gives the
        coincidence expander the SIGNED-LITERAL monomial basis in which pair-XOR is a linear function of the raw
        literals + their pairwise conjunctions (the codon's AND-of-2-active columns) -- i.e. the expander gets a
        genuine shot at making XOR representable. n_feat = 2*n_bits.
      'onbits' (the cheap fallback): active set = the ON bits only ({i : X[i] > 0}); n_feat = n_bits. Strictly
        weaker -- the expander never sees which bits are OFF, so it cannot form OFF-literal conjunctions."""
    Xb = (np.asarray(X) > 0)                                   # (n, n_bits) boolean on-bits
    n_bits = int(Xb.shape[1])
    if encoding == "onbits":
        return [set(np.where(row)[0].tolist()) for row in Xb], n_bits
    sets = []
    for row in Xb:
        on = np.where(row)[0]; off = np.where(~row)[0]
        sets.append(set((2 * on).tolist()) | set((2 * off + 1).tolist()))
    return sets, 2 * n_bits


def _expand_sets(exp, active_sets):
    """Codon matrix from precomputed active-feature SETS (the XOR path; the inheritance path uses `_expand`)."""
    return np.asarray([exp.codon(a) for a in active_sets], dtype=np.float64)


def _codon_reproducibility_sets(exp, active_sets, n=8):
    m = min(n, len(active_sets))
    a = np.asarray([exp.codon(s) for s in active_sets[:m]]); b = np.asarray([exp.codon(s) for s in active_sets[:m]])
    return float(np.mean([float((a[i] == b[i]).all()) for i in range(m)])) if m else float("nan")


# ---- LEARNED-FEEDBACK (Kolen-Pollack) on-bridge e-prop credit -- the roadmap's named gap#4 crux fix ------------------
# The on-bridge e-prop rule uses a FIXED-random DFA feedback B_direct (the 2026-08-02 finding located the wall THERE:
# the local rule cannot find the weights on Izhikevich, and the DFA feedback is fixed/misaligned). This subclass makes
# B_direct LEARNED via Kolen-Pollack (Payeur/Akrout KP): each batch, every HIDDEN pathway's feedback matrix B_direct[li]
# gets ONE increment  B_direct[li] += kp_lr*<outer> - kp_decay*B_direct[li],  <outer> = mean over the batch of
# (output error delta_k) (x) (that hidden layer's summed-spike activity). This is the SAME increment FORM as the LIF
# chained_fa_kp KP branch in _gap4_bptt_snn_chained_fa_transport_free_derisk (`_chained_fa_grads`:
# `Y += lr*(kp_lr*outer - kp_decay*Y)`, outer = post-error (x) pre-activity), ported to the DIRECT-feedback on-bridge
# rule -- so B_direct tracks the transpose of the hidden->output map IN DIRECTION. TRANSPORT-FREE: the update reads ONLY
# the output error and the RECORDED post-layer spikes -- NEVER a forward weight (B_direct is a plain runner-side numpy
# list; NO sim/ edit). Everything else -- the spiking forward, eligibility, membrane surrogate, readout, pool_k,
# reservoir_control / permuted / shuffle-DFA anti-cheats -- is the parent's, UNCHANGED. Additive: only used under
# --learned-feedback; with it OFF the runner uses the plain OnBridgeEpropNet + fixed DFA = byte-identical to the banked
# runs. train_batch is REIMPLEMENTED (the parent exposes no hook) as the parent's method VERBATIM plus the KP
# accumulation+update; frozen-hidden pathways (train_layers) keep their fixed B_direct so the frozen-reservoir control
# is byte-identical to the fixed-DFA frozen arm.
class KPFeedbackEpropNet(OnBridgeEpropNet):
    def __init__(self, *args, kp_lr=0.1, kp_decay=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.kp_lr = float(kp_lr); self.kp_decay = float(kp_decay)

    def train_batch(self, Xb, yb, shuffle_dfa=False, rng=None):
        # ---- parent OnBridgeEpropNet.train_batch VERBATIM (spiking forward + softmax deltas + e-prop grads) ----
        recs = []
        for i in range(len(Xb)):
            sp, vv, acts = self._forward_record(Xb[i])
            recs.append((sp, vv, self._logits_from(sp, vv, acts)))
        deltas = []
        for (sp, vv, logits), y in zip(recs, np.asarray(yb)):
            p = _softmax(logits / self.logit_temp)
            d = p.copy(); d[int(y)] -= 1.0
            deltas.append(d)
        if shuffle_dfa and rng is not None and len(deltas) > 1:
            deltas = [deltas[j] for j in rng.permutation(len(deltas))]   # credit mismatched to the example
        L = len(self.sizes) - 1
        grads = [np.zeros((self.sizes_phys[li], self.sizes_phys[li + 1]), dtype=np.float64) for li in range(L)]
        leaky = (self.logit_source == "leaky_readout")
        # ---- KP accumulation over the batch (mirrors `_chained_fa_grads` kp_accum: post-error (x) pre-activity). The
        #      pre-activity for B_direct[li] is the SUMMED spikes over the settle window of the POST layer (slice li+1),
        #      the layer B_direct[li] feeds credit to -- the direct analog of the LIF's `e_above.T @ spikes[li][t]`. ----
        kp_accum = [np.zeros_like(B) for B in self.B_direct]
        for (sp, vv, _lg), d in zip(recs, deltas):
            self._accum_grad(grads, sp, vv, d, skip_output=leaky)
            if leaky:
                r = self._readout_feature(sp)                              # (n_Hlast_phys,)
                dphys = self._broadcast(np.asarray(d, dtype=np.float64), L) / self.pool_k
                grads[L - 1] += np.outer(r, dphys)
            for li in range(len(self.B_direct)):
                if self.train_layers is not None and li not in self.train_layers:
                    continue                     # frozen hidden pathway -> keep its B_direct fixed (control byte-identical)
                post_summed = sp[:, self.slices[li + 1]].sum(axis=0).astype(np.float64)   # (n_post_phys,)
                kp_accum[li] += np.outer(np.asarray(d, dtype=np.float64), post_summed)     # (k, n_post_phys)
        self._apply_grads(grads, len(Xb))
        # ---- ONE KP feedback update per batch: B += kp_lr*outer - kp_decay*B (transport-free; reads only error+spikes,
        #      never a forward W). Per-(example,step) normalization (denom = B*T) mirrors the LIF KP's denom = Bn*T. ----
        denom = max(1, len(Xb) * int(self.settle_steps))
        for li in range(len(self.B_direct)):
            if self.train_layers is not None and li not in self.train_layers:
                continue
            outer = kp_accum[li] / denom
            self.B_direct[li] = self.B_direct[li] + self.kp_lr * outer - self.kp_decay * self.B_direct[li]


def run_one(seed, n_prop, use_expander, a):
    """One (seed, n_prop, representation): oracle ceiling + full e-prop + frozen-hidden reservoir + permuted +
    shuffle-DFA on the chosen forward. Returns the metrics dict with deep_credit_share."""
    t0 = time.time()
    # TASK SELECTOR (ADDITIVE, default OFF => byte-identical to the banked inheritance runs). --task-xor swaps in
    # the NON-reservoir-decodable depth-2 XOR->threshold task; everything downstream (oracle, e-prop arm, frozen
    # reservoir_control -> deep_credit_share, permuted, shuffle-DFA) is UNCHANGED. The --n-prop/--n-super/... knobs
    # are ignored when --task-xor is set.
    if a.task_xor:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    else:
        task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                           n_prop=int(n_prop), member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = int(meta["k_classes"]); inh_idx = idx["inh_idx"]
    yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))

    repro = float("nan"); codon_sparsity = float("nan"); n_feat_raw = int(Xtr.shape[1])
    if use_expander:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):     # bridge init prints to stdout
            # codon SPARSITY lever (additive, default-preserving): ACT_TH is the coincidence threshold PlateauExpander
            # reads at __init__ (a column fires if >= ACT_TH of its SAMP sampled features are active). Default 2 =>
            # dense codon (~50% active); raise to 3 (all SAMP active) => sparse codon. Set the probe module global so
            # PlateauExpander picks it up, WITHOUT editing the shared probe.
            import research.runners._gap4_plateau_expander_probe as _pep
            if a.act_th is not None:
                _pep.ACT_TH = int(a.act_th)
            if a.task_xor:
                # bit-faithful active-set reader (topk_active is meaningless on +/-1 bits; see _xor_active_sets)
                sets_tr, n_feat_exp = _xor_active_sets(Xtr, a.xor_encoding)
                sets_te, _ = _xor_active_sets(Xte, a.xor_encoding)
                exp = PlateauExpander(n_feat_exp, a.n_col, seed)
                Rtr = _expand_sets(exp, sets_tr); Rte = _expand_sets(exp, sets_te)
                repro = _codon_reproducibility_sets(exp, sets_tr)
            else:
                exp = PlateauExpander(Xtr.shape[1], a.n_col, seed)
                Rtr = _expand(exp, Xtr, a.topk); Rte = _expand(exp, Xte, a.topk)
                repro = _codon_reproducibility(exp, Xtr, a.topk)
        codon_sparsity = float(Rtr.mean())
    else:
        # RAW arm: features straight in. For XOR that is the +/-1 X the 2026-08-02 finding fed into the plain
        # forward (deep_share ~0); for inheritance the real-valued X. Reproduces the "without expander" baseline.
        Rtr, Rte = Xtr.astype(np.float64), Xte.astype(np.float64)
    n_in = int(Rtr.shape[1])

    # --- oracle ceiling: a depth-2 rate DendriticMLP fit on the SAME representation e-prop sees (apples-to-apples) ---
    onet = DendriticMLP([n_in] + [96] * a.n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Rtr, ytr, 250, 0.3, 128, seed)
    oracle_inh = _acc_on(onet, Rte, yte, inh_idx)

    # subsample the TRAIN set for the on-bridge spiking arms (held-out NEVER subsampled).
    if a.train_subsample and len(Rtr) > a.train_subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Rtr))[:a.train_subsample]
        Rtr_b, ytr_b = Rtr[keep], ytr[keep]
    else:
        Rtr_b, ytr_b = Rtr, ytr

    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha,
              in_current_pA=a.in_current_pA, in_bias_pA=a.in_bias_pA, hidden_lr_scale=a.hidden_lr_scale)

    def _mk():
        # --learned-feedback: swap the fixed-DFA net for the KP LEARNED-feedback net (all arms, so the ONLY change
        # vs the fixed-DFA condition is learned-vs-fixed feedback). Default OFF => plain OnBridgeEpropNet, byte-identical.
        if a.learned_feedback:
            return KPFeedbackEpropNet(n_in, a.hidden, k, seed=seed, n_hidden_layers=a.n_hidden_layers,
                                      settle_steps=a.settle_steps, eprop_lr=a.eprop_lr, eps_leak=a.eps_leak,
                                      surrogate=a.surrogate, alpha_surr=a.alpha_surr, beta_surr=a.beta_surr,
                                      logit_source=a.logit_source, w_clip=a.w_clip, hp=hp, pool_k=a.pool_k,
                                      kp_lr=a.kp_lr, kp_decay=a.kp_decay)
        return OnBridgeEpropNet(n_in, a.hidden, k, seed=seed, n_hidden_layers=a.n_hidden_layers,
                                settle_steps=a.settle_steps, eprop_lr=a.eprop_lr, eps_leak=a.eps_leak,
                                surrogate=a.surrogate, alpha_surr=a.alpha_surr, beta_surr=a.beta_surr,
                                logit_source=a.logit_source, w_clip=a.w_clip, hp=hp, pool_k=a.pool_k)

    # --- main e-prop arm (ALL feedforward pathways trained) ---
    net = _mk(); w0 = net.ff_weight_norm()
    _train_eprop(net, Rtr_b, ytr_b, a.epochs, a.batch, seed)
    train_acc = net.accuracy(Rtr_b, ytr_b)
    inh_acc = net.acc_on(Rte, yte, inh_idx)
    ff_moved = float(abs(net.ff_weight_norm() - w0))

    # --- permuted-label control -> ~chance (teacher-contingency) ---
    prng = np.random.default_rng(seed + 555)
    yperm = ytr_b[prng.permutation(len(ytr_b))]
    pnet = _mk(); _train_eprop(pnet, Rtr_b, yperm, a.epochs, a.batch, seed)
    perm_inh = pnet.acc_on(Rte, yte, inh_idx)

    # --- shuffle-DFA control -> ~chance (the DFA credit route is load-bearing) ---
    snet = _mk(); _train_eprop(snet, Rtr_b, ytr_b, a.epochs, a.batch, seed, shuffle_dfa=True)
    shuf_inh = snet.acc_on(Rte, yte, inh_idx)

    # --- FROZEN-HIDDEN RESERVOIR control -> deep_credit_share (the whole point) ---
    # train ONLY the last FF pathway (the leaky readout); freeze the hidden FF pathways at random init.
    fnet = _mk(); fnet.train_layers = {fnet.n_hidden_layers}
    _train_eprop(fnet, Rtr_b, ytr_b, a.epochs, a.batch, seed)
    froz_inh = fnet.acc_on(Rte, yte, inh_idx)
    deep_share = float("nan")
    if not (np.isnan(inh_acc) or np.isnan(froz_inh)) and (inh_acc - chance) > 1e-9:
        # deep_credit_share IS an attribution: what fraction of e-prop's above-chance skill is NOT in the
        # frozen-hidden reservoir control (i.e. is owned by TRAINING the hidden feedforward pathways). Make it
        # explicit via tools.lab so the treatment/control subtraction is checked, not just co-measured (the gap#5
        # lesson: both arms measured != the difference attributed). Equals (inh_acc - froz_inh)/(inh_acc - chance).
        from tools.lab import attributable_to
        deep_share = float(attributable_to("deep-credit: eprop-hidden-training vs frozen-reservoir",
                                           inh_acc - chance, froz_inh - chance, warn_below=-1.0))

    # trains_the_task: the SAME definition the e-prop port uses -- the e-prop arm clears chance AND every control
    # (permuted, shuffle-DFA, frozen-hidden reservoir) by a margin. Load-bearing for the XOR GO gate (deliverable d).
    trains = bool((not np.isnan(inh_acc)) and inh_acc > chance + 0.05 and inh_acc > perm_inh + 0.05
                  and inh_acc > shuf_inh + 0.05 and (not np.isnan(froz_inh)) and inh_acc > froz_inh + 0.05)

    return {"seed": seed, "n_prop": int(n_prop), "mode": ("expander" if use_expander else "raw"),
            "task": ("xor" if a.task_xor else "inheritance"),
            "credit": ("kp_learned_feedback" if a.learned_feedback else "fixed_dfa"),
            "kp_lr": (a.kp_lr if a.learned_feedback else None),
            "kp_decay": (a.kp_decay if a.learned_feedback else None),
            "xor_encoding": (a.xor_encoding if a.task_xor else None),
            "k_classes": k, "chance": chance, "n_features_in": n_in, "n_features_raw": n_feat_raw,
            "n_train_smoke": int(len(ytr_b)), "n_inherit_heldout": int(len(inh_idx)),
            "oracle_inherit": oracle_inh, "eprop_train_acc": train_acc, "eprop_inherit_heldout": inh_acc,
            "eprop_ff_weight_moved": ff_moved, "permuted_inherit": perm_inh, "shuffle_dfa_inherit": shuf_inh,
            "frozen_hidden_inherit": froz_inh, "deep_credit_share": deep_share, "trains_the_task": trains,
            "codon_reproducibility": repro, "codon_sparsity": codon_sparsity,
            "elapsed_seconds": round(time.time() - t0, 1)}


def _fmt(r):
    return (f"[seed {r['seed']} n_prop {r['n_prop']} {r['mode']:8s}] k={r['k_classes']} chance {r['chance']:.3f} "
            f"| oracle {r['oracle_inherit']:.3f} | eprop-train {r['eprop_train_acc']:.3f} eprop-inherit "
            f"{r['eprop_inherit_heldout']:.3f} | FROZEN {r['frozen_hidden_inherit']:.3f} => DEEP_SHARE "
            f"{r['deep_credit_share']:+.3f} | perm {r['permuted_inherit']:.3f} shufDFA {r['shuffle_dfa_inherit']:.3f} "
            f"| ff-moved {r['eprop_ff_weight_moved']:.1f} ({r['elapsed_seconds']:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description="gap#4: does a representable (plateau-expanded) forward make hidden-layer "
                                             "credit matter (deep_credit_share>0) on the on-bridge e-prop rule?")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-prop", type=int, default=3, help="depth knob: 2^n_prop classes; test 3 and 4 (reservoir should FAIL)")
    ap.add_argument("--mode", choices=["raw", "expander", "both"], default="both")
    # THE 2026-08-02 LEVER (ADDITIVE, default OFF => byte-identical to the banked inheritance runs). Swap in the
    # NON-reservoir-decodable depth-2 XOR->threshold task; the deep_credit_share test on a task a frozen reservoir
    # CANNOT shortcut, where the raw Izhikevich forward gave deep_share ~0 because it could not train XOR at all.
    ap.add_argument("--task-xor", action="store_true",
                    help="use the non-reservoir-decodable depth-2 XOR->threshold task (make_task_xor) instead of "
                         "semantic-inheritance; --n-prop/--n-super/... are ignored. Additive, default off.")
    ap.add_argument("--xor-encoding", choices=["literal", "onbits"], default="literal",
                    help="how +/-1 XOR bits become the expander's active-feature SET (only under --task-xor): "
                         "'literal' (default) = 2 slots/bit (on-literal + off-literal), the FAIR signed-literal "
                         "monomial basis; 'onbits' = ON bits only (weaker fallback). See _xor_active_sets.")
    # THE LEARNED-FEEDBACK (Kolen-Pollack) LEVER (ADDITIVE, default OFF => byte-identical to the banked fixed-DFA runs).
    # The 2026-08-02 finding located the wall at the on-bridge e-prop CREDIT RULE (fixed-random DFA feedback cannot find
    # the weights on Izhikevich). The roadmap's named fix: LEARN the DFA feedback B_direct via KP so it tracks W^T in
    # direction (transport-free). --learned-feedback swaps in KPFeedbackEpropNet for ALL e-prop arms.
    ap.add_argument("--learned-feedback", action="store_true",
                    help="LEARN the DFA feedback B_direct via Kolen-Pollack (B += kp_lr*outer - kp_decay*B, "
                         "outer = output-error (x) hidden-spike-activity) instead of the fixed-random B_direct. "
                         "Transport-free (never reads a forward W). Additive, default off = fixed DFA (byte-identical).")
    ap.add_argument("--kp-lr", type=float, default=0.1,
                    help="KP feedback learning rate (only under --learned-feedback); mirrors the LIF chained_fa_kp "
                         "kp_lr form. Start point to sweep if the smoke shows no B_direct movement / blow-up.")
    ap.add_argument("--kp-decay", type=float, default=1e-4,
                    help="KP feedback weight decay (only under --learned-feedback); mirrors the LIF chained_fa_kp kp_decay.")
    # expander
    ap.add_argument("--n-col", type=int, default=200, help="PlateauExpander columns (the probe's N_COL)")
    ap.add_argument("--act-th", type=int, default=None,
                    help="codon SPARSITY lever: coincidence threshold (probe default 2 => dense ~50%%; 3 => sparse). "
                         "None keeps the probe's ACT_TH.")
    ap.add_argument("--topk", type=int, default=PROBE_TOPK, help="active features per row before expansion (probe's TOPK)")
    # e-prop net (the port's regime; smoke-scaled)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--eps-leak", type=float, default=0.9)
    ap.add_argument("--surrogate", choices=["atan_vt", "std"], default="atan_vt")
    ap.add_argument("--alpha-surr", type=float, default=0.15)
    ap.add_argument("--beta-surr", type=float, default=1.0)
    ap.add_argument("--logit-source", choices=["spike_sum", "event_rate", "membrane", "leaky_readout"],
                    default="leaky_readout")
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--train-subsample", type=int, default=160)
    ap.add_argument("--pool-k", type=int, default=1)
    ap.add_argument("--hidden-lr-scale", type=float, default=5.0)
    # drive hp (the port's working regime)
    ap.add_argument("--tonic-h-pA", type=float, default=100.0)
    ap.add_argument("--tonic-o-pA", type=float, default=150.0)
    ap.add_argument("--ff-w-init", type=float, default=2000.0)
    ap.add_argument("--in-current-pA", type=float, default=700.0)
    ap.add_argument("--in-bias-pA", type=float, default=300.0)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
    # task knobs (n_super=24 so 2^n_prop classes are all represented at n_prop<=4)
    ap.add_argument("--n-super", type=int, default=24)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    modes = ["raw", "expander"] if a.mode == "both" else [a.mode]
    t0 = time.time(); err = None; rows = []
    try:
        for s in a.seeds:
            for m in modes:
                r = run_one(s, a.n_prop, m == "expander", a)
                rows.append(r)
                print(_fmt(r), flush=True)
                # checkpoint after every arm (an interrupted smoke should not lose finished arms)
                try:
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps({"probe": "gap4_representable_forward_plus_credit",
                                                       "partial": True, "config": vars(a), "rows": rows},
                                                      indent=2, default=str))
                except Exception as _ck:
                    print(f"[warn] checkpoint failed ({_ck})", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    # ---- WITH-vs-WITHOUT verdict per n_prop (means over seeds) ----
    def _mean(mode, key):
        vals = [r[key] for r in rows if r["mode"] == mode]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {"probe": "gap4_representable_forward_plus_credit", "n_prop": a.n_prop, "seeds": a.seeds,
               "config": vars(a), "elapsed_seconds": round(time.time() - t0, 1), "rows": rows,
               "error": err}
    if err is None and rows:
        raw_share = _mean("raw", "deep_credit_share"); exp_share = _mean("expander", "deep_credit_share")
        raw_froz = _mean("raw", "frozen_hidden_inherit"); exp_froz = _mean("expander", "frozen_hidden_inherit")
        raw_eprop = _mean("raw", "eprop_inherit_heldout"); exp_eprop = _mean("expander", "eprop_inherit_heldout")
        ch = _mean("raw", "chance"); ch = ch if ch == ch else _mean("expander", "chance")
        exp_perm = _mean("expander", "permuted_inherit"); exp_shuf = _mean("expander", "shuffle_dfa_inherit")
        exp_oracle = _mean("expander", "oracle_inherit"); exp_repro = _mean("expander", "codon_reproducibility")
        exp_rows = [r for r in rows if r["mode"] == "expander"]
        exp_trains_all = bool(exp_rows and all(r.get("trains_the_task", False) for r in exp_rows))
        controls_clean = bool((exp_perm != exp_perm or exp_perm <= ch + 0.10)
                              and (exp_shuf != exp_shuf or exp_shuf <= ch + 0.10))
        if a.task_xor:
            # THE DECISIVE XOR GATE (deliverable d): deep_credit_share > 0.3 AND frozen < eprop AND
            # shuffle-DFA <= chance+0.10 AND permuted ~chance AND trains_the_task=True. If it holds, credit on a
            # REPRESENTABLE forward is load-bearing and the production-bridge residual was the FORWARD; if
            # deep_share stays ~0 the next residual is named by the columns (frozen HIGH => codon made XOR
            # reservoir-solvable; eprop ~chance / oracle ~chance => the codon forward is still too weak/noisy).
            froz_lt_eprop = bool(exp_froz == exp_froz and exp_eprop == exp_eprop and exp_froz + 1e-9 < exp_eprop)
            go = bool(exp_share == exp_share and exp_share > 0.3 and froz_lt_eprop
                      and controls_clean and exp_trains_all)
            summary["aggregate"] = {"chance": ch, "raw_deep_share": raw_share, "expander_deep_share": exp_share,
                                    "raw_frozen_inherit": raw_froz, "expander_frozen_inherit": exp_froz,
                                    "raw_eprop_inherit": raw_eprop, "expander_eprop_inherit": exp_eprop,
                                    "expander_oracle_inherit": exp_oracle, "expander_permuted": exp_perm,
                                    "expander_shuffle_dfa": exp_shuf, "expander_codon_reproducibility": exp_repro,
                                    "expander_trains_the_task_all_seeds": exp_trains_all,
                                    "frozen_lt_eprop": froz_lt_eprop, "controls_clean": controls_clean}
            summary["SIGNAL"] = go
            summary["verdict"] = (
                f"XOR (chance {ch:.3f}): deep_credit_share raw {raw_share:+.3f} -> expander {exp_share:+.3f} "
                f"(frozen-reservoir inherit raw {raw_froz:.3f} / expander {exp_froz:.3f}; eprop inherit raw "
                f"{raw_eprop:.3f} / expander {exp_eprop:.3f}; expander oracle {exp_oracle:.3f}; controls perm "
                f"{exp_perm:.3f} shufDFA {exp_shuf:.3f}; codon repro {exp_repro:.3f}; trains-all "
                f"{exp_trains_all}). "
                + ("GO -- credit on a REPRESENTABLE (plateau-expanded) forward IS load-bearing on XOR "
                   "(deep_credit_share>0.3, frozen<eprop, controls clean, trains). The production-bridge residual "
                   "was the FORWARD, and the PlateauExpander forward closes it." if go else
                   "NOT-GO -- name the next residual from the columns: frozen ~= eprop (HIGH) => the codon made "
                   "XOR reservoir-solvable (deep credit not needed on this forward); eprop ~chance while oracle "
                   ">> chance => the codon forward is still too weak/noisy for the on-bridge credit rule; oracle "
                   "~chance => the expansion did not make XOR representable (try --xor-encoding, more --n-col)."))
        else:
            # inheritance GO (unchanged): deep_credit_share>0.15 where the frozen reservoir fails, controls clean,
            # clearly above the raw share.
            froz_fails = bool(exp_froz == exp_froz and exp_froz <= ch + 0.15)
            go = bool(exp_share == exp_share and exp_share > 0.15 and froz_fails and controls_clean
                      and (raw_share != raw_share or exp_share > raw_share + 0.10))
            summary["aggregate"] = {"chance": ch, "raw_deep_share": raw_share, "expander_deep_share": exp_share,
                                    "raw_frozen_inherit": raw_froz, "expander_frozen_inherit": exp_froz,
                                    "raw_eprop_inherit": raw_eprop, "expander_eprop_inherit": exp_eprop,
                                    "expander_permuted": exp_perm, "expander_shuffle_dfa": exp_shuf,
                                    "frozen_reservoir_fails": froz_fails, "controls_clean": controls_clean}
            summary["SIGNAL"] = go
            summary["verdict"] = (
                f"n_prop={a.n_prop} (chance {ch:.3f}): deep_credit_share raw {raw_share:+.3f} -> expander {exp_share:+.3f} "
                f"(frozen-reservoir inherit raw {raw_froz:.3f} / expander {exp_froz:.3f}; eprop inherit raw {raw_eprop:.3f} "
                f"/ expander {exp_eprop:.3f}; expander controls perm {exp_perm:.3f} shufDFA {exp_shuf:.3f}). "
                + ("GO -- the representable forward MAKES hidden-layer credit matter (deep_credit_share>0.15 where the "
                   "frozen reservoir fails)." if go else
                   "NOT-GO (smoke) -- see the frozen_hidden_inherit column: if it is HIGH the expansion made the task "
                   "reservoir-solvable (deep credit not needed); if the eprop arm itself is near chance the smoke is "
                   "under-trained. Honest negative if it holds at more seeds/epochs."))
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no rows"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[rep-fwd+credit] {summary['verdict']}", flush=True)
    print(f"[rep-fwd+credit] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
