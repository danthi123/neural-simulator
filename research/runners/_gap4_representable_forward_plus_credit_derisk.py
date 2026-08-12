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

THE 2026-08-02 LEARNED SELF-PREDICTING MICROCIRCUIT (Sacramento Eq.9) LEVER -- the roadmap's §2.8 crux (ADDITIVE, default
OFF). The 2026-08-02 finding ELIMINATIVELY located the on-bridge deep-credit wall at the e-prop's LOCAL credit FACTOR: on
the sparse representable codon (--task-xor --act-th 3 --mode expander) fixed-DFA AND learned-KP-feedback BOTH gave chance,
so it is NOT feedback direction, NOT representability, NOT codon density. The roadmap's named fix (LIF-rate 6-seed GO,
`2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO.md`, commit 56c90d67): a plastic interneuron that LEARNS to
predict/cancel the top-down so the apical dendrite computes a LOCAL error approximating backprop's (apical-silent when
correct, EARNED). `--microcircuit` (see MicrocircuitEpropNet) replaces the hidden credit factor delta_k @ B_direct with the
interneuron-CANCELLED apical src_pred @ W_PI - onehot @ B_direct, W_PI learned by the transport-free Eq.9 rule. NO sim/ edit.
Decisive gate (deliverable d): does the microcircuit let on-bridge e-prop TRAIN XOR (eprop > chance AND > frozen,
deep_credit_share > 0) on the sparse representable codon where the plain surrogate/eligibility gave chance? YES => the
local-credit-factor was the wall and the microcircuit is the fix; NO => the residual is the surrogate/eligibility
weight-finding on Izhikevich itself (operating-point / phi'-vanishing) -- name it.
    # 1-seed SMOKE (microcircuit, sparse codon --act-th 3, expander only -- run FIRST):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._gap4_representable_forward_plus_credit_derisk \
        --task-xor --act-th 3 --microcircuit --mode expander --seeds 42 --epochs 60 --train-subsample 160 \
        --out research/findings/raw/gap4/rep_fwd_credit_xor_micro_smoke_s42.json
    # 6-seed (only if the smoke shows W_PI learning [selfpred_cos rising] + eprop lifting off chance):
    SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 python -m research.runners._gap4_representable_forward_plus_credit_derisk \
        --task-xor --act-th 3 --microcircuit --mode expander --seeds 42 43 44 100 101 102 --epochs 60 \
        --train-subsample 160 --out research/findings/raw/gap4/rep_fwd_credit_xor_micro_6seed.json
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


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


# ---- LEARNED SELF-PREDICTING MICROCIRCUIT (Sacramento-Senn 2018 Eq.9) on-bridge e-prop credit -- the roadmap's §2.8 crux ----
# The 2026-08-02 finding ELIMINATIVELY located the on-bridge deep-credit wall at the e-prop's LOCAL credit FACTOR: on the
# sparse representable codon (--task-xor --act-th 3 --mode expander) the oracle solves XOR (0.94) but on-bridge e-prop sits
# at chance whether the DFA feedback is FIXED-random OR LEARNED (Kolen-Pollack) -- so it is NOT feedback DIRECTION, and NOT
# forward representability/density. The roadmap's named fix (the LIF-rate 6-seed GO, `2026-07-24-gap4-learned-selfpredicting-
# microcircuit-CPUrate-GO.md`, commit 56c90d67, MicroNet): a plastic interneuron that LEARNS to predict/cancel the top-down
# so the apical dendrite computes a LOCAL error that approximates backprop's -- Sacramento-Senn 2018 "apical silent when
# correct" EARNED, not initialized. This subclass ports that LOCAL-CREDIT FACTOR onto the on-bridge e-prop.
#
# WHAT CHANGES (only the HIDDEN learning signal). Parent OnBridgeEpropNet hidden credit (in _accum_grad):
#     Lsig[li] = ( delta_k @ B_direct[li] ) / T ,  delta_k = softmax(logits) - onehot     (raw fixed-random DFA)
# Microcircuit hidden credit = the interneuron-CANCELLED apical (teacher via the fixed feedback B_direct, the network's OWN
# prediction via a LEARNED interneuron W_PI):
#     Lsig_micro[li] = ( src_pred @ W_PI[li] - onehot @ B_direct[li] ) / T ,  src_pred = softmax(logits)
# At the self-predicting fixed point W_PI == B_direct this reduces to delta_k @ B_direct EXACTLY (byte-identical to fixed-DFA)
# -> the microcircuit is a strict GENERALIZATION whose distinct behaviour is the LEARNED, apical-silent-when-correct
# trajectory of the credit (the multiplicative suppression of plasticity when the net is already correct + the interneuron's
# lower-variance cancellation of the noisy spiking top-down -- the thing that could differ ON SPIKES where fixed/learned-KP
# both failed). W_PI is learned from a NOISY init by the LOCAL, transport-free Eq.9 self-prediction rule:
#     dW_PI[li] = wpi_lr * ( src_pred^T @ ( src_pred @ (B_direct[li] - W_PI[li]) ) ) / m   -> drives W_PI -> B_direct.
# TRANSPORT-FREE: reads ONLY src_pred (a local activity), B_direct and W_PI -- NEVER a forward weight (cp_connections). W_PI
# is a plain runner-side numpy list; NO `sim/` edit. Everything else -- the spiking forward, eligibility eps, membrane
# surrogate psi, leaky readout, pool_k, reservoir_control / permuted / shuffle-DFA anti-cheats, the oracle -- is the
# parent's, UNCHANGED. Additive: only used under --microcircuit; with it OFF the runner is byte-identical to the banked runs.
# TOPOLOGY NOTE: the LIF-rate GO put W_PI at the TOP hidden layer only (sequential FA chains e_upper down); this e-prop uses
# DIRECT feedback alignment (every hidden layer li gets its OWN projection delta_k @ B_direct[li]), so each hidden layer gets
# its own interneuron W_PI[li] -- the faithful adaptation of "the interneuron cancels the top-down each layer receives" to
# the DFA topology. The self-prediction rule + the reservoir-control freeze both respect self.train_layers so the frozen-
# reservoir arm (hidden FF + its interneurons frozen at init) stays a clean control.
class MicrocircuitEpropNet(OnBridgeEpropNet):
    def __init__(self, *args, wpi_lr=0.2, wpi_init="noisy", wpi_noise=1.0, wpi_plastic=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.wpi_lr = float(wpi_lr); self.wpi_init = str(wpi_init); self.wpi_plastic = bool(wpi_plastic)
        _seed = int(kwargs.get("seed", 0))
        wrng = np.random.default_rng(_seed + 4242)     # SEPARATE stream (no transport): the interneuron init
        self.W_PI = []
        for B in self.B_direct:                        # one interneuron per hidden pathway; shape == B_direct[li] == (k, n_post_phys)
            if wpi_init == "fixedpoint":
                self.W_PI.append(B.copy())             # start AT the self-predicting fixed point (positive control)
            else:
                self.W_PI.append(wrng.normal(0.0, float(wpi_noise), B.shape))   # 'noisy' (default): silence must be EARNED
        self._selfpred_cos_traj = []

    def _wpi_selfpredict_update(self, src_pred_batch):
        """LOCAL, transport-free Sacramento-Senn Eq.9 self-prediction for every hidden interneuron W_PI[li]:
        dW_PI[li] = wpi_lr * ( r_int^T @ v_free ) / m,  r_int = src_pred (interneuron rate = the net's OWN softmax),
        v_free = src_pred @ (B_direct[li] - W_PI[li]) (the free-phase residual apical). Drives W_PI -> B_direct. Reads
        ONLY src_pred (an activity), self.B_direct, self.W_PI -- NEVER a forward weight (cp_connections)."""
        sp = np.atleast_2d(np.asarray(src_pred_batch, dtype=np.float64))       # (m, k)
        m = max(1, sp.shape[0])
        for li in range(len(self.W_PI)):
            if self.train_layers is not None and li not in self.train_layers:
                continue                               # frozen hidden pathway -> keep W_PI fixed (reservoir control byte-clean)
            v_free = sp @ (self.B_direct[li] - self.W_PI[li])                  # (m, n_post_phys)
            dWpi = (sp.T @ v_free) / m                                         # (k, n_post_phys) == W_PI[li].shape
            self.W_PI[li] = self.W_PI[li] + self.wpi_lr * dWpi

    def _accum_grad_micro(self, grads, sp, vv, src_pred, onehot, skip_output=False):
        """Parent OnBridgeEpropNet._accum_grad VERBATIM EXCEPT the HIDDEN learning signal, which is the interneuron-
        CANCELLED apical (src_pred @ W_PI[li] - onehot @ B_direct[li]) / T instead of (delta_k @ B_direct[li]) / T.
        The eligibility eps, the membrane surrogate psi, and the leaky-readout output path are byte-identical to parent."""
        L = len(self.sizes) - 1
        T = sp.shape[0]
        src_pred = np.asarray(src_pred, dtype=np.float64); onehot = np.asarray(onehot, dtype=np.float64)
        delta_k = src_pred - onehot                                            # output-layer delta (unused when skip_output)
        delta_out_phys = self._broadcast(delta_k, L) / self.pool_k
        std_stats = None
        if self.surrogate == "std":
            std_stats = {}
            for li in range(L):
                post_sl = self.slices[li + 1]
                vp = vv[:, post_sl]
                std_stats[li] = (vp.mean(), vp.std() + 1e-6)
        eps = [np.zeros(self.sizes_phys[li], dtype=np.float64) for li in range(L)]
        last = L - 1
        for t in range(T):
            for li in range(L):
                if skip_output and li == last:
                    continue
                z_pre = sp[t, self.slices[li]].astype(np.float64)
                eps[li] = self.eps_leak * eps[li] + z_pre
                post_sl = self.slices[li + 1]
                v_post = vv[t, post_sl].astype(np.float64)
                sp_post = sp[t, post_sl]
                if self.surrogate == "std":
                    m, s = std_stats[li]
                    z = (v_post - m) / s
                    sub = 1.0 / (1.0 + (self.beta_surr * z) ** 2)
                    psi = np.where(sp_post > 0.5, 1.0, sub)
                else:
                    psi = self._surrogate(v_post, sp_post, post_sl)
                if li == last:
                    Lsig = delta_out_phys / T
                    if self.output_psi_one:
                        psi = np.ones_like(psi)
                else:
                    # THE MICROCIRCUIT CHANGE: interneuron-cancelled apical (teacher via fixed B_direct, prediction via
                    # the LEARNED interneuron W_PI). == delta_k @ B_direct at the fixed point W_PI == B_direct.
                    Lsig = (src_pred @ self.W_PI[li] - onehot @ self.B_direct[li]) / T
                g = Lsig * psi
                grads[li] += np.outer(eps[li], g)

    def train_batch(self, Xb, yb, shuffle_dfa=False, rng=None):
        # ---- forward passes + per-example src_pred / onehot (the parent's softmax-delta setup, split into p and onehot) ----
        recs = []
        for i in range(len(Xb)):
            sp, vv, acts = self._forward_record(Xb[i])
            recs.append((sp, vv, self._logits_from(sp, vv, acts)))
        yb = np.asarray(yb)
        ps = []; onehots = []
        for (sp, vv, logits), y in zip(recs, yb):
            p = _softmax(logits / self.logit_temp)
            oh = np.zeros_like(p); oh[int(y)] = 1.0
            ps.append(p); onehots.append(oh)
        # ---- W_PI self-prediction update (Eq.9) from the batch's TRUE src_pred (free phase; teacher-independent) ----
        if self.wpi_plastic:
            self._wpi_selfpredict_update(np.stack(ps))
            for li in range(len(self.W_PI)):
                self._selfpred_cos_traj.append(_cos(self.W_PI[li], self.B_direct[li]))
        # ---- shuffle-DFA: permute the (src_pred, onehot) APICAL credit across the batch (eligibility stays with the
        #      example; credit mismatched) -- the direct analog of the parent's `deltas = deltas[perm]` scramble. ----
        cred_idx = list(range(len(recs)))
        if shuffle_dfa and rng is not None and len(cred_idx) > 1:
            cred_idx = list(rng.permutation(len(cred_idx)))
        L = len(self.sizes) - 1
        grads = [np.zeros((self.sizes_phys[li], self.sizes_phys[li + 1]), dtype=np.float64) for li in range(L)]
        leaky = (self.logit_source == "leaky_readout")
        for ex_i, cj in enumerate(cred_idx):
            sp, vv, _lg = recs[ex_i]
            p_cred = ps[cj]; oh_cred = onehots[cj]                             # credit from example cj (== ex_i unless shuffled)
            self._accum_grad_micro(grads, sp, vv, p_cred, oh_cred, skip_output=leaky)
            if leaky:
                r = self._readout_feature(sp)
                d = p_cred - oh_cred                                           # readout delta (shuffled under shuffle_dfa, as parent)
                dphys = self._broadcast(d, L) / self.pool_k
                grads[L - 1] += np.outer(r, dphys)
        self._apply_grads(grads, len(Xb))

    def selfpred_cos(self):
        """cos(W_PI[li], B_direct[li]) per hidden layer -- the EARNED self-prediction: ~0 at the noisy init, -> ~1 as
        W_PI learns to predict the top-down (the load-bearing 'the interneuron LEARNED to cancel' observable)."""
        return [float(_cos(self.W_PI[li], self.B_direct[li])) for li in range(len(self.W_PI))]

    def apical_silent_stats(self, X, y, idx, max_examples=120):
        """The on-bridge RATE-observable analogue: mean|apical| (the interneuron-cancelled residual
        src_pred @ W_PI[top] - onehot @ B_direct[top]) on CORRECT vs INCORRECT held-out predictions. EARNED-silent =>
        correct << incorrect (silent_ratio small). Reuses the forward pass; capped for cost."""
        if idx is None or len(idx) == 0 or len(self.W_PI) == 0:
            return None
        top = len(self.W_PI) - 1
        y = np.asarray(y)
        ii = list(idx)[:max_examples]
        mags = []; corr = []
        for i in ii:
            sp, vv, acts = self._forward_record(X[i])
            logits = self._logits_from(sp, vv, acts)
            p = _softmax(logits / self.logit_temp)
            oh = np.zeros_like(p); oh[int(y[i])] = 1.0
            apical = p @ self.W_PI[top] - oh @ self.B_direct[top]             # (n_post_phys,)
            mags.append(float(np.abs(apical).mean()))
            corr.append(bool(int(np.argmax(logits)) == int(y[i])))
        mags = np.asarray(mags); corr = np.asarray(corr)
        mc = float(mags[corr].mean()) if corr.any() else float("nan")
        mi = float(mags[~corr].mean()) if (~corr).any() else float("nan")
        ratio = float(mc / (mi + 1e-12)) if (corr.any() and (~corr).any()) else float("nan")
        return {"apical_correct": mc, "apical_incorrect": mi, "silent_ratio": ratio,
                "frac_correct": float(corr.mean()), "selfpred_cos_top": float(_cos(self.W_PI[top], self.B_direct[top]))}


# ==================================================================================================================
# THE 2026-08-02 CREDIT-FACTOR DIAGNOSTIC (`--measure-credit-factor`, ADDITIVE, default OFF). A MEASUREMENT, not a
# training run. Update 3 of the finding ELIMINATIVELY isolated the on-bridge deep-credit wall to a SINGLE mechanism:
# the on-bridge e-prop's LOCAL CREDIT FACTOR itself -- the atan_vt membrane surrogate sigma'(v_soma - theta) TIMES the
# forward eligibility, on the post-reset Izhikevich membrane (the phi'-vanishing / operating-point residual). Every
# error-ROUTING fix (fixed-DFA, learned-KP, self-predicting microcircuit) was proven inert; what remains is whether
# that local factor carries CREDIT-USABLE SELECTIVITY. This probe answers it directly, on the sparse representable
# codon (--task-xor --act-th 3 --mode expander), with NO sim/ edit -- the surrogate + eligibility + DFA learning-signal
# are read from the SAME recorded cp_membrane_potential_v / cp_firing_states the e-prop rule uses (net._surrogate,
# net.B_direct), and the reference "credit these neurons SHOULD receive" is the EMPIRICAL backprop-oracle per hidden
# neuron measured by finite-difference on the exact substrate: oracle_j = dLoss/dI_j (nudge neuron j's input current,
# re-run the spiking forward, read the loss change) -- the gold-standard total-derivative credit e-prop's per-neuron
# factor (Lsig_j * psi_j) is trying to estimate. All arms share the SAME representation and the SAME (readout-fit)
# operating point, so the comparison is apples-to-apples.
#
# THE THREE DECISIVE READS (owner-named):
#   (i)   surrogate ~0 / tiny dynamic range  => phi'-VANISHING (atan surrogate collapses on the post-reset membrane) =>
#         the fix is a surrogate / operating-point that keeps sigma' informative.
#   (ii)  surrogate HAS range but the credit factor has ZERO alignment with the FD oracle => no credit-usable
#         selectivity (a substrate-level read limit) => honest substrate limit or a different local factor.
#   (iii) alignment decent-but-weak => the eligibility / optimization is the residual (more epochs / lr / batch).
# The Lsig-only vs Lsig*psi alignment split attributes any misalignment to the FEEDBACK (already eliminated) vs the
# SURROGATE: if multiplying by psi DEGRADES alignment, the surrogate is vanishing on exactly the high-credit neurons
# (the phi'-vanishing mechanism biting the credit) -- read (i)/(ii); if psi is ~constant, alignment is unchanged
# (surrogate inert, not the differentiator).


def _make_codon(seed, use_expander, a):
    """Build the task + (optionally plateau-expanded) representation EXACTLY as run_one does -- the sparse representable
    codon on --task-xor --act-th 3 --mode expander. Returns (Rtr, ytr, Rte, yte, inh_idx, k, chance, meta,
    codon_sparsity, repro). Byte-identical construction to run_one (the diagnostic measures the SAME forward)."""
    if a.task_xor:
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_xor(seed)
    else:
        task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                           n_prop=int(a.n_prop), member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise)
        (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = int(meta["k_classes"]); inh_idx = idx["inh_idx"]
    yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    repro = float("nan"); codon_sparsity = float("nan")
    if use_expander:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            import research.runners._gap4_plateau_expander_probe as _pep
            if a.act_th is not None:
                _pep.ACT_TH = int(a.act_th)
            if a.task_xor:
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
        Rtr, Rte = Xtr.astype(np.float64), Xte.astype(np.float64)
    return Rtr, ytr, Rte, yte, inh_idx, k, chance, meta, codon_sparsity, repro


def _build_net_for_measure(n_in, k, seed, a, hp):
    """Build the SAME e-prop net run_one's _mk() builds (default plain fixed-DFA = the residual's target; --microcircuit
    / --learned-feedback honored so the diagnostic can also read those local factors)."""
    if a.microcircuit:
        return MicrocircuitEpropNet(n_in, a.hidden, k, seed=seed, n_hidden_layers=a.n_hidden_layers,
                                    settle_steps=a.settle_steps, eprop_lr=a.eprop_lr, eps_leak=a.eps_leak,
                                    surrogate=a.surrogate, alpha_surr=a.alpha_surr, beta_surr=a.beta_surr,
                                    logit_source=a.logit_source, w_clip=a.w_clip, hp=hp, pool_k=a.pool_k,
                                    wpi_lr=a.wpi_lr, wpi_init=a.wpi_init, wpi_noise=a.wpi_noise,
                                    wpi_plastic=(not a.wpi_frozen))
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


def _spearman(a, b):
    """Rank correlation without a scipy dep: Pearson on the ranks. NaN if <2 points or a degenerate (constant) input."""
    a = np.asarray(a, dtype=np.float64).ravel(); b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _psi_over_time(net, sp, vv, post_sl):
    """Per-post-neuron surrogate psi_j AGGREGATED (mean) and its raw per-(t,j) samples, using the net's OWN _surrogate
    (byte-identical to the psi the e-prop rule uses in _accum_grad). Returns (psi_mean[n_post], psi_samples[T,n_post])."""
    T = sp.shape[0]
    samples = np.zeros((T, int(post_sl.stop - post_sl.start)), dtype=np.float64)
    for t in range(T):
        samples[t] = net._surrogate(vv[t, post_sl].astype(np.float64), sp[t, post_sl], post_sl)
    return samples.mean(axis=0), samples


def _fd_oracle_layer(net, feat_row, y, hidden_slices, delta_pA, logit_temp):
    """EMPIRICAL backprop-oracle per hidden neuron on the EXACT substrate: oracle_j = dLoss/dI_j via forward-difference.
    Nudge hidden neuron j's constant input current by +delta_pA (through the monkeypatched _base_drive), re-run the
    spiking forward, read the cross-entropy change. Returns a list (per hidden layer) of per-neuron credit arrays.
    Reads ONLY the loss and the substrate response -- the gold-standard total-derivative credit e-prop estimates."""
    def _loss_of_logits(logits):
        p = _softmax(np.asarray(logits, dtype=np.float64) / logit_temp)
        return float(-np.log(max(1e-12, p[int(y)])))
    sp, vv, acts = net._forward_record(feat_row)
    base_loss = _loss_of_logits(net._logits_from(sp, vv, acts))
    out = []
    for sl in hidden_slices:
        creds = np.zeros(int(sl.stop - sl.start), dtype=np.float64)
        for jj, g in enumerate(range(sl.start, sl.stop)):
            net._probe_delta[g] = delta_pA
            spg, vvg, actg = net._forward_record(feat_row)
            net._probe_delta[g] = 0.0
            creds[jj] = (_loss_of_logits(net._logits_from(spg, vvg, actg)) - base_loss) / delta_pA
        out.append(creds)
    return out, base_loss


def measure_credit_factor(seed, a):
    """THE DIAGNOSTIC (deliverable): does the on-bridge local credit factor (surrogate x eligibility DFA signal) carry
    credit-usable selectivity on the sparse representable codon? Per hidden layer: alignment of the e-prop per-neuron
    credit factor (Lsig_j * psi_j) with the finite-difference backprop oracle (dLoss/dI_j), the Lsig-only baseline, and
    the surrogate's dynamic range / CV. Returns a dict; no training-arm side effects (readout is fit = reservoir op-pt)."""
    t0 = time.time()
    use_expander = (a.mode != "raw")            # the residual lives on the plateau-expanded (representable) codon
    Rtr, ytr, Rte, yte, inh_idx, k, chance, meta, codon_sparsity, repro = _make_codon(seed, use_expander, a)
    # subsample the train set exactly as run_one (readout fit + measurement batch drawn from it)
    if a.train_subsample and len(Rtr) > a.train_subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Rtr))[:a.train_subsample]
        Rtr_b, ytr_b = Rtr[keep], ytr[keep]
    else:
        Rtr_b, ytr_b = Rtr, ytr
    n_in = int(Rtr_b.shape[1])
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha,
              in_current_pA=a.in_current_pA, in_bias_pA=a.in_bias_pA, hidden_lr_scale=a.hidden_lr_scale)
    net = _build_net_for_measure(n_in, k, seed, a, hp)

    # OPERATING POINT: fit the readout (readout-only training = the frozen-hidden RESERVOIR arm) so W_readout is
    # meaningful (delta_k, and hence Lsig + the FD-oracle loss, are non-degenerate) while the HIDDEN weights stay at
    # the reservoir init -- exactly where e-prop must take its FIRST hidden step. This is where "can credit even get
    # started" is decided. (--credit-measure-on trained: additionally train ALL layers first, to read the converged
    # operating point; default 'init' = the fair "start of hidden learning" read.)
    net.train_layers = {net.n_hidden_layers}
    _train_eprop(net, Rtr_b, ytr_b, a.epochs, a.batch, seed)
    if a.credit_measure_on == "trained":
        net.train_layers = None
        _train_eprop(net, Rtr_b, ytr_b, a.epochs, a.batch, seed)
    net.train_layers = None

    # monkeypatch _base_drive to add a per-neuron current perturbation vector (default zeros => byte-identical forward).
    # _forward_record calls _base_drive() once/forward then overwrites ONLY the input slice, so a perturbation on the
    # HIDDEN slices survives for every settle step. No sim/ edit (pure host-side drive arithmetic).
    net._probe_delta = np.zeros(net.n_total, dtype=np.float32)
    _orig_base_drive = net._base_drive
    def _patched_base_drive():
        return _orig_base_drive() + net._probe_delta
    net._base_drive = _patched_base_drive

    # the HIDDEN layers that receive a DFA learning signal (B_direct exists for li in range(len(sizes)-2)); their post
    # slices are the hidden-neuron index ranges. pool_k=1 => physical == logical.
    n_hidden_pathways = len(net.B_direct)
    hidden_slices = [net.slices[li + 1] for li in range(n_hidden_pathways)]

    # measurement batch (a fixed subsample of the train set)
    mb = min(int(a.fd_batch), len(Rtr_b))
    mrng = np.random.default_rng(seed + 909)
    m_idx = mrng.permutation(len(Rtr_b))[:mb]

    peak = float(net._psi_peak)
    # per-layer accumulators (concatenate per-neuron vectors across the batch for a pooled alignment)
    acc = {li: {"bridge": [], "lsig": [], "psi": [], "oracle": [], "cos_bridge": [], "cos_lsig": [],
                "sp_bridge": [], "sp_lsig": []} for li in range(n_hidden_pathways)}
    psi_all = {li: [] for li in range(n_hidden_pathways)}     # every per-(t,j) surrogate sample (read i)
    last_readout_align = []                                    # exact last-hidden readout-grad cross-check (cos)

    # exact last-hidden readout-gradient oracle (free, exact for the top hidden layer feeding the leaky readout):
    #   oracle_last_j = dLoss/d r_j = (delta_k @ W_readout^T)_j -- validates the FD oracle on the layer where it is exact.
    from sim.backend import to_host
    W_ro = None
    if net.logit_source == "leaky_readout":
        W_ro = np.asarray(to_host(net.br.cp_connections.data[net._data_idx_flat[-1]]), dtype=np.float64).reshape(
            net.sizes_phys[-2], net.sizes_phys[-1])          # (n_Hlast_phys, k*K)

    for i in m_idx:
        feat = Rtr_b[i]; y = int(ytr_b[i])
        sp, vv, acts = net._forward_record(feat)
        logits = net._logits_from(sp, vv, acts)
        p = _softmax(np.asarray(logits, dtype=np.float64) / net.logit_temp)
        onehot = np.zeros_like(p); onehot[y] = 1.0
        delta_k = p - onehot
        # FD empirical oracle per hidden layer (dLoss/dI_j)
        fd_creds, _bl = _fd_oracle_layer(net, feat, y, hidden_slices, a.fd_delta_pA, net.logit_temp)
        for li in range(n_hidden_pathways):
            post_sl = net.slices[li + 1]
            psi_mean, psi_samples = _psi_over_time(net, sp, vv, post_sl)
            if hasattr(net, "W_PI"):
                # microcircuit local factor: interneuron-cancelled apical (src_pred @ W_PI - onehot @ B_direct)
                lsig = (p @ net.W_PI[li] - onehot @ net.B_direct[li])
            else:
                lsig = (delta_k @ net.B_direct[li])          # fixed-DFA / KP learned feedback (same delta_k @ B form)
            bridge = lsig * psi_mean
            oracle = fd_creds[li]
            acc[li]["bridge"].append(bridge); acc[li]["lsig"].append(lsig)
            acc[li]["psi"].append(psi_mean); acc[li]["oracle"].append(oracle)
            acc[li]["cos_bridge"].append(_cos(bridge, oracle)); acc[li]["cos_lsig"].append(_cos(lsig, oracle))
            acc[li]["sp_bridge"].append(_spearman(bridge, oracle)); acc[li]["sp_lsig"].append(_spearman(lsig, oracle))
            psi_all[li].append(psi_samples.ravel())
        if W_ro is not None:
            r = net._readout_feature(sp)                      # (n_Hlast_phys,)
            oracle_last = (delta_k @ W_ro.T)                  # dLoss/d r_j at the top hidden layer (exact)
            last_readout_align.append(_cos(fd_creds[-1], oracle_last))

    def _stat(x):
        x = np.asarray([v for v in x if v == v], dtype=np.float64)
        return (float(np.mean(x)) if x.size else float("nan"), float(np.std(x)) if x.size else float("nan"))

    layers = []
    for li in range(n_hidden_pathways):
        psi_flat = np.concatenate(psi_all[li]) if psi_all[li] else np.array([np.nan])
        psi_norm = psi_flat / peak
        bridge_pool = np.concatenate(acc[li]["bridge"]); oracle_pool = np.concatenate(acc[li]["oracle"])
        lsig_pool = np.concatenate(acc[li]["lsig"])
        cb_m, cb_s = _stat(acc[li]["cos_bridge"]); cl_m, cl_s = _stat(acc[li]["cos_lsig"])
        sb_m, _ = _stat(acc[li]["sp_bridge"]); sl_m, _ = _stat(acc[li]["sp_lsig"])
        psi_mean = float(np.nanmean(psi_flat)); psi_std = float(np.nanstd(psi_flat))
        layers.append({
            "layer": li, "n_neurons": int(net.sizes_phys[li + 1]),
            "psi_peak": peak,
            "psi_mean": psi_mean, "psi_mean_frac_of_peak": float(psi_mean / peak) if peak > 0 else float("nan"),
            "psi_std": psi_std, "psi_cv": float(psi_std / psi_mean) if psi_mean > 1e-12 else float("nan"),
            "psi_p05_frac_peak": float(np.nanpercentile(psi_norm, 5)),
            "psi_p50_frac_peak": float(np.nanpercentile(psi_norm, 50)),
            "psi_p95_frac_peak": float(np.nanpercentile(psi_norm, 95)),
            "psi_dynamic_range_frac_peak": float(np.nanpercentile(psi_norm, 95) - np.nanpercentile(psi_norm, 5)),
            "psi_frac_below_0p05peak": float(np.mean(psi_norm < 0.05)),
            "psi_frac_at_peak": float(np.mean(psi_norm > 0.95)),
            "bridge_abs_mean": float(np.mean(np.abs(bridge_pool))),
            "lsig_abs_mean": float(np.mean(np.abs(lsig_pool))),
            "oracle_abs_mean": float(np.mean(np.abs(oracle_pool))),
            # ALIGNMENT of the e-prop per-neuron credit factor (Lsig*psi) with the FD backprop oracle (dLoss/dI):
            "cos_bridge_vs_oracle_mean": cb_m, "cos_bridge_vs_oracle_std": cb_s,
            "cos_lsig_vs_oracle_mean": cl_m, "cos_lsig_vs_oracle_std": cl_s,
            "spearman_bridge_vs_oracle_mean": sb_m, "spearman_lsig_vs_oracle_mean": sl_m,
            "cos_bridge_pooled": _cos(bridge_pool, oracle_pool), "cos_lsig_pooled": _cos(lsig_pool, oracle_pool),
            "surrogate_helps_alignment": bool((cb_m == cb_m) and (cl_m == cl_m) and cb_m > cl_m + 1e-6),
        })

    # DECISIVE CLASSIFICATION (per the owner's (i)/(ii)/(iii)). Read the TOP hidden layer (the one whose FD oracle is
    # corroborated by the exact readout-gradient) as the headline; report all layers.
    top = layers[-1]
    vanish = bool(top["psi_mean_frac_of_peak"] < 0.10 or top["psi_dynamic_range_frac_peak"] < 0.02
                  or top["bridge_abs_mean"] < 1e-9)
    aligned = bool(abs(top["cos_bridge_vs_oracle_mean"]) > 0.30)
    weakly_aligned = bool(0.10 < abs(top["cos_bridge_vs_oracle_mean"]) <= 0.30)
    if vanish:
        read = "(i) phi'-VANISHING -- the atan_vt surrogate collapses on the post-reset Izhikevich membrane (tiny " \
               "mean/dynamic-range), so the local credit factor is ~0 / non-selective. FIX: a surrogate or " \
               "operating-point that keeps sigma' informative (widen alpha, shift the membrane operating point " \
               "toward theta, or a voltage-independent eligibility)."
    elif aligned:
        read = "(iii) OPTIMIZATION RESIDUAL -- the surrogate has range AND the credit factor is aligned with the " \
               "backprop oracle (|cos|>0.30): the credit IS usable; the residual is the eligibility/optimization " \
               "(more epochs / lr / batch / hidden_lr_scale)."
    elif weakly_aligned:
        read = "(iii-weak) the credit factor is WEAKLY aligned (0.10<|cos|<=0.30): usable-but-weak -- try more " \
               "epochs / lr, but on the boundary of (ii)."
    else:
        read = "(ii) NO CREDIT-USABLE SELECTIVITY -- the surrogate has range but the credit factor has ~ZERO " \
               "alignment with the backprop oracle (|cos|<=0.10): a substrate-level read limit (cf the reservoir " \
               "finding). Honest substrate limit, or a fundamentally different local credit factor."
    surrogate_is_culprit = bool(top["cos_lsig_vs_oracle_mean"] == top["cos_lsig_vs_oracle_mean"]
                                and top["cos_bridge_vs_oracle_mean"] < top["cos_lsig_vs_oracle_mean"] - 1e-6)

    return {"seed": seed, "task": ("xor" if a.task_xor else "inheritance"), "mode": ("expander" if use_expander else "raw"),
            "act_th": a.act_th, "credit": ("microcircuit" if a.microcircuit else "kp" if a.learned_feedback else "fixed_dfa"),
            "credit_measure_on": a.credit_measure_on, "k_classes": k, "chance": chance,
            "codon_sparsity": codon_sparsity, "codon_reproducibility": repro, "n_features_in": n_in,
            "fd_batch": mb, "fd_delta_pA": a.fd_delta_pA, "settle_steps": a.settle_steps,
            "layers": layers,
            "last_hidden_fd_vs_readoutgrad_cos": (float(np.mean(last_readout_align)) if last_readout_align else float("nan")),
            "READ": read, "top_layer_psi_mean_frac_peak": top["psi_mean_frac_of_peak"],
            "top_layer_cos_bridge_vs_oracle": top["cos_bridge_vs_oracle_mean"],
            "top_layer_cos_lsig_vs_oracle": top["cos_lsig_vs_oracle_mean"],
            "surrogate_degrades_alignment": surrogate_is_culprit,
            "elapsed_seconds": round(time.time() - t0, 1)}


def _fmt_credit(r):
    tl = r["layers"][-1]
    return (f"[seed {r['seed']} {r['credit']} {r['mode']}] codon_sparsity {r['codon_sparsity']:.3f} "
            f"| TOP-HIDDEN psi_mean/peak {tl['psi_mean_frac_of_peak']:.4f} dyn-range {tl['psi_dynamic_range_frac_peak']:.3f} "
            f"frac<0.05peak {tl['psi_frac_below_0p05peak']:.2f} | cos(bridge,oracle) "
            f"{tl['cos_bridge_vs_oracle_mean']:+.3f} cos(lsig,oracle) {tl['cos_lsig_vs_oracle_mean']:+.3f} "
            f"| FD-vs-readoutgrad {r['last_hidden_fd_vs_readoutgrad_cos']:+.3f} ({r['elapsed_seconds']:.0f}s)\n"
            f"    => {r['READ']}")


# ============================================================================================================
# THE 2026-08-02 FA-CONVERGENCE MEASUREMENT (`--measure-fa-convergence`, ADDITIVE, default OFF). Update 4 of the
# finding REFUTED phi'-vanishing (the atan surrogate is HEALTHY, psi 0.31-0.32, dynamic-range 0.94) and the
# FD-oracle credit-factor read DEGRADED at the trained state (FD-vs-readoutgrad cos +0.916 init -> +0.235 trained),
# so the trained-state alignment was NOT clean. This probe AVOIDS the FD oracle entirely and measures the classic
# Lillicrap-2016 feedback-alignment CONVERGENCE signature DIRECTLY: during e-prop training, do the FORWARD weights
# ALIGN to the FIXED feedback so the transport-free credit approaches the true-gradient direction? Measured as
# cos(W_forward, B_direct^T) per epoch -- NO oracle, NO finite-difference perturbation. Logs (per epoch, for the
# on-bridge net trained by REAL e-prop, ALL layers):
#   (1) fa_cos[li] = cos( downstream-forward-chain(li) , B_direct[li]^T ) per hidden pathway. For the LAST hidden
#       layer the chain IS the H_last->out readout weight (the classic Lillicrap readout FA signature). RISING over
#       epochs => FA CONVERGES; flat/~0 => it does NOT. Weight-only read (no forward) -> logged EVERY epoch.
#   (2) credit_align[li] = mean_batch cos( delta_k @ B_direct[li] , delta_k @ chain(li)^T ) -- the DELIVERED DFA
#       credit direction vs the TRANSPORT gradient direction (weight transport is legitimate in a MEASUREMENT, never
#       in the learning rule). An FD-FREE cross-check of (1) at the credit-signal level. RISING => the delivered
#       credit is becoming the true gradient.
#   (3) eprop_inherit per epoch: does held-out accuracy track the alignment.
# Compare to the LIF reference (_snn_bptt_forward_vs_learning_isolation_derisk --measure-fa-convergence), where the
# SAME e-prop DFA rule TRAINS (inherit ~0.895) -- the "FA converges" fingerprint. LIF cos RISES + Izhikevich cos
# FLAT => FA-convergence fails on the point-neuron Izhikevich substrate specifically (the precisely-named residual,
# pointing to a different local rule / the dendritic substrate). Izhikevich cos ALSO rises but accuracy stays at
# chance => alignment converges but the credit magnitude/eligibility is the issue (a distinct next mechanism). NO
# sim/ edit -- W (cp_connections.data, read via to_host + reshape) and B_direct are host-side; the weight read is a
# measurement, not a transport used by the rule.


def _ff_weight(net, p):
    """FF pathway p forward weight matrix (n_pre_phys, n_post_phys), read from the substrate cp_connections (the SAME
    .data slots e-prop moves). A MEASUREMENT read: weight transport is legitimate in a diagnostic, never in the rule."""
    from sim.backend import to_host
    return np.asarray(to_host(net.br.cp_connections.data[net._data_idx_flat[p]]), dtype=np.float64).reshape(
        net.sizes_phys[p], net.sizes_phys[p + 1])


def _fa_forward_chain(net, li):
    """Downstream forward map from hidden pathway li's POST layer to the output = product of FF pathway weight
    matrices (li+1 .. readout). For the LAST hidden pathway this is exactly the H_last->out readout weight (the
    classic Lillicrap readout FA signature). Shape (sizes_phys[li+1], sizes_phys[-1])."""
    L = len(net.sizes) - 1                       # number of FF pathways; readout index = L-1
    M = _ff_weight(net, li + 1)
    for p in range(li + 2, L):
        M = M @ _ff_weight(net, p)
    return M


def measure_fa_convergence(seed, a):
    """FA-CONVERGENCE measurement (deliverable): per-epoch cos(forward-weight, B_direct^T) as the on-bridge e-prop
    trains ALL layers on the (sparse representable) codon. Avoids the FD oracle. Returns a dict with the per-epoch
    trajectory + a decisive RISES/FLAT read on the top (readout) hidden pathway."""
    t0 = time.time()
    use_expander = (a.mode != "raw")
    Rtr, ytr, Rte, yte, inh_idx, k, chance, meta, codon_sparsity, repro = _make_codon(seed, use_expander, a)
    if a.train_subsample and len(Rtr) > a.train_subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Rtr))[:a.train_subsample]
        Rtr_b, ytr_b = Rtr[keep], ytr[keep]
    else:
        Rtr_b, ytr_b = Rtr, ytr
    n_in = int(Rtr_b.shape[1])
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha,
              in_current_pA=a.in_current_pA, in_bias_pA=a.in_bias_pA, hidden_lr_scale=a.hidden_lr_scale)
    net = _build_net_for_measure(n_in, k, seed, a, hp)     # train_layers=None => ALL layers train (FA can develop)
    if net.logit_source == "leaky_readout":
        net.fit_readout_norm(Rtr_b)
    n_hp = len(net.B_direct)

    # fixed measurement batch for the credit-align cross-check (item 2)
    mb = min(int(a.fd_batch), len(Rtr_b))
    mrng = np.random.default_rng(seed + 909)
    m_idx = mrng.permutation(len(Rtr_b))[:mb]

    def _fa_cos_now():
        return [_cos(_fa_forward_chain(net, li), net.B_direct[li].T) for li in range(n_hp)]

    def _credit_align_now():
        chains = [_fa_forward_chain(net, li) for li in range(n_hp)]
        cols = [[] for _ in range(n_hp)]
        for i in m_idx:
            sp, vv, acts = net._forward_record(Rtr_b[i])
            logits = net._logits_from(sp, vv, acts)
            p = _softmax(np.asarray(logits, dtype=np.float64) / net.logit_temp)
            oh = np.zeros_like(p); oh[int(ytr_b[i])] = 1.0
            dk = p - oh                                     # (k,) softmax error
            for li in range(n_hp):
                cols[li].append(_cos(dk @ net.B_direct[li], dk @ chains[li].T))   # DFA credit vs transport gradient
        return [float(np.nanmean(c)) if c else float("nan") for c in cols]

    def _rec(ep, heavy):
        r = {"epoch": ep, "fa_cos": _fa_cos_now()}
        if heavy:
            r["credit_align"] = _credit_align_now()
            r["inherit"] = net.acc_on(Rte, yte, inh_idx)
        else:
            r["credit_align"] = None; r["inherit"] = None
        return r

    traj = [_rec(0, True)]                        # init (pre-training) read
    rng = np.random.default_rng(seed + 777)       # SAME stream _train_eprop uses (faithful training dynamics)
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(Rtr_b))
        for i in range(0, len(Rtr_b), a.batch):
            b = perm[i:i + a.batch]
            net.train_batch(Rtr_b[b], ytr_b[b])
        heavy = ((ep % max(1, a.fa_eval_every) == 0) or ep == a.epochs)
        traj.append(_rec(ep, heavy))

    fa_top = [rec["fa_cos"][-1] for rec in traj]              # the LAST hidden pathway = the readout FA signature
    inh = [rec["inherit"] for rec in traj if rec["inherit"] is not None]
    init_c = fa_top[0]; final_c = fa_top[-1]
    peak_c = max(fa_top, key=lambda v: abs(v))
    rise = final_c - init_c
    converges = bool(abs(peak_c) - abs(init_c) > 0.05 and final_c > init_c)
    read = (f"FA {'CONVERGES' if converges else 'does NOT converge (flat/near-0)'}: top-hidden cos(W,B^T) "
            f"init {init_c:+.3f} -> final {final_c:+.3f} (peak {peak_c:+.3f}, rise {rise:+.3f}); "
            f"eprop_inherit {(inh[0] if inh else float('nan')):.3f} -> {(inh[-1] if inh else float('nan')):.3f} "
            f"(chance {chance:.3f}).")
    return {"seed": seed, "substrate": "izhikevich", "task": ("xor" if a.task_xor else "inheritance"),
            "mode": ("expander" if use_expander else "raw"), "act_th": a.act_th,
            "credit": ("microcircuit" if a.microcircuit else "kp" if a.learned_feedback else "fixed_dfa"),
            "k_classes": k, "chance": chance, "codon_sparsity": codon_sparsity, "codon_reproducibility": repro,
            "n_features_in": n_in, "n_hidden_pathways": n_hp, "epochs": a.epochs, "fa_eval_every": a.fa_eval_every,
            "fa_cos_top_init": init_c, "fa_cos_top_final": final_c, "fa_cos_top_peak": peak_c,
            "fa_cos_top_rise": rise, "fa_converges": converges,
            "inherit_init": (inh[0] if inh else float("nan")), "inherit_final": (inh[-1] if inh else float("nan")),
            "trajectory": traj, "READ": read, "elapsed_seconds": round(time.time() - t0, 1)}


def _fmt_fa(r):
    return (f"[seed {r['seed']} {r['substrate']} {r['credit']} {r['mode']}] codon_sparsity "
            f"{r.get('codon_sparsity', float('nan')):.3f} | top-hidden cos(W,B^T) {r['fa_cos_top_init']:+.3f} -> "
            f"{r['fa_cos_top_final']:+.3f} (peak {r['fa_cos_top_peak']:+.3f}) | inherit {r['inherit_init']:.3f} -> "
            f"{r['inherit_final']:.3f} (chance {r['chance']:.3f}) ({r['elapsed_seconds']:.0f}s)\n    => {r['READ']}")


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
        # --microcircuit: swap in the LEARNED SELF-PREDICTING MICROCIRCUIT net for ALL arms, so the ONLY change vs the
        # fixed-DFA condition is the interneuron-cancelled apical LOCAL CREDIT FACTOR (Sacramento Eq.9). Default OFF.
        if a.microcircuit:
            return MicrocircuitEpropNet(n_in, a.hidden, k, seed=seed, n_hidden_layers=a.n_hidden_layers,
                                        settle_steps=a.settle_steps, eprop_lr=a.eprop_lr, eps_leak=a.eps_leak,
                                        surrogate=a.surrogate, alpha_surr=a.alpha_surr, beta_surr=a.beta_surr,
                                        logit_source=a.logit_source, w_clip=a.w_clip, hp=hp, pool_k=a.pool_k,
                                        wpi_lr=a.wpi_lr, wpi_init=a.wpi_init, wpi_noise=a.wpi_noise,
                                        wpi_plastic=(not a.wpi_frozen))
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
    # microcircuit diagnostics (only under --microcircuit): the EARNED self-prediction cos(W_PI,B_direct) per hidden
    # layer + the apical-silent-when-correct read. selfpred_cos -> ~1 proves the interneuron LEARNED to cancel; a
    # silent_ratio << 1 proves apical-silent-on-correct EARNED. Diagnostic-only; the GO gate is the same XOR gate below.
    micro_selfpred = net.selfpred_cos() if a.microcircuit else None
    micro_apical = net.apical_silent_stats(Rte, yte, inh_idx) if a.microcircuit else None

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
            "credit": ("microcircuit_selfpredict" if a.microcircuit
                       else "kp_learned_feedback" if a.learned_feedback else "fixed_dfa"),
            "kp_lr": (a.kp_lr if a.learned_feedback else None),
            "kp_decay": (a.kp_decay if a.learned_feedback else None),
            "wpi_lr": (a.wpi_lr if a.microcircuit else None),
            "wpi_init": (a.wpi_init if a.microcircuit else None),
            "wpi_plastic": ((not a.wpi_frozen) if a.microcircuit else None),
            "micro_selfpred_cos": micro_selfpred,
            "micro_apical_silent": micro_apical,
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
    # THE LEARNED SELF-PREDICTING MICROCIRCUIT LEVER (ADDITIVE, default OFF => byte-identical to the banked runs). The
    # roadmap's §2.8 crux fix: replace the on-bridge e-prop's HIDDEN local credit factor (delta_k @ B_direct) with the
    # Sacramento Eq.9 interneuron-CANCELLED apical (src_pred @ W_PI - onehot @ B_direct), W_PI a plastic interneuron
    # LEARNED (from a noisy init) to predict/cancel the top-down (transport-free; NO sim/ edit). Decisive gate (d): does
    # this let on-bridge e-prop TRAIN XOR on the sparse representable codon where fixed-DFA AND learned-KP both gave chance?
    ap.add_argument("--microcircuit", action="store_true",
                    help="use the LEARNED self-predicting microcircuit (Sacramento Eq.9) interneuron-cancelled apical as "
                         "the HIDDEN local credit factor instead of raw fixed-DFA. Additive, default off. Mutually "
                         "exclusive with --learned-feedback.")
    ap.add_argument("--wpi-lr", type=float, default=0.2,
                    help="interneuron self-prediction learning rate (only under --microcircuit); mirrors the LIF MicroNet wpi_lr.")
    ap.add_argument("--wpi-init", choices=["noisy", "fixedpoint"], default="noisy",
                    help="interneuron W_PI init: 'noisy' (default; silence must be EARNED) or 'fixedpoint' (W_PI:=B_direct, "
                         "silent from step 0 -- the positive control that the cancellation IS the fixed-DFA credit).")
    ap.add_argument("--wpi-noise", type=float, default=1.0, help="std of the noisy W_PI init (only under --microcircuit).")
    ap.add_argument("--wpi-frozen", action="store_true",
                    help="freeze W_PI at its noisy init (the Sacramento anti-cheat: with the Eq.9 plasticity OFF, apical "
                         "must NOT go silent on correct -- selfpred_cos stays ~0, silent_ratio ~1).")
    # THE 2026-08-02 CREDIT-FACTOR DIAGNOSTIC (ADDITIVE, default OFF => byte-identical to the banked training runs). A
    # MEASUREMENT: does the on-bridge local credit factor (surrogate x eligibility DFA signal) carry credit-usable
    # SELECTIVITY on the sparse representable codon? Reads (i)/(ii)/(iii) -- phi'-vanishing / no-selectivity / optimization.
    ap.add_argument("--measure-credit-factor", action="store_true",
                    help="DIAGNOSTIC (no training verdict): measure the on-bridge per-hidden-neuron local credit factor "
                         "(Lsig_j*psi_j) vs the finite-difference backprop oracle (dLoss/dI_j) + the surrogate's "
                         "dynamic range / CV, per hidden layer. Additive, default off. Run on --task-xor --act-th 3.")
    ap.add_argument("--fd-batch", type=int, default=16,
                    help="examples used for the finite-difference oracle + alignment (only under --measure-credit-factor).")
    ap.add_argument("--fd-delta-pA", type=float, default=20.0,
                    help="input-current perturbation for the FD oracle dLoss/dI_j (only under --measure-credit-factor).")
    ap.add_argument("--credit-measure-on", choices=["init", "trained"], default="init",
                    help="operating point for the credit-factor read: 'init' (default; readout-fit + hidden at the "
                         "reservoir init = the start of hidden learning) or 'trained' (all layers trained first).")
    # FA-CONVERGENCE probe (additive, default off): the FD-oracle-FREE trained-state alignment read.
    ap.add_argument("--measure-fa-convergence", action="store_true",
                    help="FA-CONVERGENCE probe (no training verdict): per-epoch cos(forward-weight, B_direct^T) as "
                         "on-bridge e-prop trains ALL layers -- the FD-oracle-free Lillicrap-2016 FA signature "
                         "(rising => FA converges; flat/~0 => it does not). Additive, default off. Run on "
                         "--task-xor --act-th 3 --mode expander; compare to the LIF-reference runner's same flag. "
                         "Reuses --fd-batch for the item-(2) credit-align cross-check batch.")
    ap.add_argument("--fa-eval-every", type=int, default=1,
                    help="epoch cadence for the HEAVY per-epoch reads (credit_align + eprop_inherit) under "
                         "--measure-fa-convergence; the cheap cos(W,B^T) read is logged EVERY epoch regardless.")
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

    if a.microcircuit and a.learned_feedback:
        ap.error("--microcircuit and --learned-feedback are mutually exclusive (both replace the hidden credit factor).")

    # ---- THE CREDIT-FACTOR DIAGNOSTIC BRANCH (additive; skips the 4 heavy training arms + the DendriticMLP oracle) ----
    if a.measure_credit_factor:
        t0 = time.time(); err = None; rows = []
        try:
            for s in a.seeds:
                r = measure_credit_factor(s, a)
                rows.append(r)
                print(_fmt_credit(r), flush=True)
                try:
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps({"probe": "gap4_credit_factor_diagnostic", "partial": True,
                                                       "config": vars(a), "rows": rows}, indent=2, default=str))
                except Exception as _ck:
                    print(f"[warn] checkpoint failed ({_ck})", flush=True)
        except Exception as e:
            err = repr(e); traceback.print_exc()
        # aggregate the headline read across seeds (top hidden layer)
        def _amean(key):
            vals = [r[key] for r in rows if r.get(key) == r.get(key)]
            return float(np.mean(vals)) if vals else float("nan")
        reads = [r["READ"].split(" -- ")[0] for r in rows] if rows else []
        summary = {"probe": "gap4_credit_factor_diagnostic", "seeds": a.seeds, "config": vars(a),
                   "elapsed_seconds": round(time.time() - t0, 1), "rows": rows, "error": err,
                   "aggregate": {"top_layer_psi_mean_frac_peak": _amean("top_layer_psi_mean_frac_peak"),
                                 "top_layer_cos_bridge_vs_oracle": _amean("top_layer_cos_bridge_vs_oracle"),
                                 "top_layer_cos_lsig_vs_oracle": _amean("top_layer_cos_lsig_vs_oracle"),
                                 "last_hidden_fd_vs_readoutgrad_cos": _amean("last_hidden_fd_vs_readoutgrad_cos"),
                                 "reads": reads}}
        summary["verdict"] = (f"CREDIT-FACTOR DIAGNOSTIC (top-hidden, {len(a.seeds)} seed(s)): "
                              f"psi_mean/peak {_amean('top_layer_psi_mean_frac_peak'):.4f}, "
                              f"cos(bridge,oracle) {_amean('top_layer_cos_bridge_vs_oracle'):+.3f}, "
                              f"cos(lsig,oracle) {_amean('top_layer_cos_lsig_vs_oracle'):+.3f}, "
                              f"FD-vs-readoutgrad {_amean('last_hidden_fd_vs_readoutgrad_cos'):+.3f} | reads: {reads}"
                              + (f" | ERROR {err}" if err else ""))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
        print("\n" + "=" * 100, flush=True)
        print(f"[credit-factor] {summary['verdict']}", flush=True)
        print(f"[credit-factor] wrote {a.out}\n" + "=" * 100, flush=True)
        return 0 if (err is None and rows) else 1

    # ---- THE FA-CONVERGENCE BRANCH (additive; per-epoch cos(W,B^T) trajectory, no training GO verdict) ----
    if a.measure_fa_convergence:
        t0 = time.time(); err = None; rows = []
        try:
            for s in a.seeds:
                r = measure_fa_convergence(s, a)
                rows.append(r)
                print(_fmt_fa(r), flush=True)
                try:
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(a.out).write_text(json.dumps({"probe": "gap4_fa_convergence", "partial": True,
                                                       "config": vars(a), "rows": rows}, indent=2, default=str))
                except Exception as _ck:
                    print(f"[warn] checkpoint failed ({_ck})", flush=True)
        except Exception as e:
            err = repr(e); traceback.print_exc()
        def _amean(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r.get(key) == r.get(key)]
            return float(np.mean(vals)) if vals else float("nan")
        n_conv = int(sum(1 for r in rows if r.get("fa_converges")))
        summary = {"probe": "gap4_fa_convergence", "seeds": a.seeds, "config": vars(a),
                   "elapsed_seconds": round(time.time() - t0, 1), "rows": rows, "error": err,
                   "aggregate": {"fa_cos_top_init": _amean("fa_cos_top_init"),
                                 "fa_cos_top_final": _amean("fa_cos_top_final"),
                                 "fa_cos_top_peak": _amean("fa_cos_top_peak"),
                                 "fa_cos_top_rise": _amean("fa_cos_top_rise"),
                                 "inherit_final": _amean("inherit_final"),
                                 "n_converges": n_conv, "n_seeds": len(rows)}}
        summary["verdict"] = (
            f"FA-CONVERGENCE (izhikevich, {len(rows)}/{len(a.seeds)} seed(s)): top-hidden cos(W,B^T) "
            f"{_amean('fa_cos_top_init'):+.3f} -> {_amean('fa_cos_top_final'):+.3f} "
            f"(peak {_amean('fa_cos_top_peak'):+.3f}, rise {_amean('fa_cos_top_rise'):+.3f}); "
            f"inherit_final {_amean('inherit_final'):.3f}; converges {n_conv}/{len(rows)}"
            + (f" | ERROR {err}" if err else ""))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
        print("\n" + "=" * 100, flush=True)
        print(f"[fa-convergence] {summary['verdict']}", flush=True)
        print(f"[fa-convergence] wrote {a.out}\n" + "=" * 100, flush=True)
        return 0 if (err is None and rows) else 1

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
            if a.microcircuit:
                # microcircuit diagnostics (means over expander seeds): EARNED self-prediction cos(W_PI,B_direct) -> ~1,
                # apical-silent-on-correct ratio -> small. Diagnostic-only; the GO gate above is unchanged.
                _mc_rows = [r for r in rows if r["mode"] == "expander" and r.get("micro_selfpred_cos")]
                _sp = [float(np.mean(r["micro_selfpred_cos"])) for r in _mc_rows if r["micro_selfpred_cos"]]
                _sr = [r["micro_apical_silent"]["silent_ratio"] for r in _mc_rows
                       if r.get("micro_apical_silent")
                       and r["micro_apical_silent"].get("silent_ratio") == r["micro_apical_silent"].get("silent_ratio")]
                summary["aggregate"]["microcircuit_selfpred_cos"] = float(np.mean(_sp)) if _sp else float("nan")
                summary["aggregate"]["microcircuit_apical_silent_ratio"] = float(np.mean(_sr)) if _sr else float("nan")
            summary["SIGNAL"] = go
            _credit = ("microcircuit" if a.microcircuit else "kp" if a.learned_feedback else "fixed-dfa")
            _mc_diag = ("" if not a.microcircuit else
                        f" [micro selfpred_cos {summary['aggregate'].get('microcircuit_selfpred_cos', float('nan')):.3f} "
                        f"apical_silent_ratio {summary['aggregate'].get('microcircuit_apical_silent_ratio', float('nan')):.3f}]")
            summary["verdict"] = (
                f"XOR [credit={_credit}]{_mc_diag} (chance {ch:.3f}): deep_credit_share raw {raw_share:+.3f} -> expander {exp_share:+.3f} "
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
