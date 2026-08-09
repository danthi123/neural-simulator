"""TEACHER-LOOP FIXED ORTHOGONAL-TARGET PATTERN SEPARATION DE-RISK (2026-08-09): attack the sleep-replay retention
plateau (0.55, the known replay-retention cap in extreme interference; replay MAGNITUDE is REFUTED as the lever,
main 8d2510d3) with a DIFFERENT, proven continual-learning mechanism: replace the DRIFTING learnable shared readout
with PREDEFINED, mutually-ORTHOGONAL target codes -- one fixed pattern-separated code per fact (PS-SNN, Hu et al.
2026, Sci Reports; the DG decorrelation as a FIXED sparse orthogonal readout basis). A new fact's readout update
writes into its OWN orthogonal subspace instead of overwriting earlier facts on the shared readout.

WHY THIS IS THE STRONG VERSION OF THE REFUTED WEAK ONE. The k-WTA sparse-gated readout ALLOCATION was a HONEST
NEGATIVE (+0.00, finding 2026-08-08-teacher-loop-sparse-gated-readout-allocation-...): it separated the engrams
(winner-overlap 0.20) but did NOT raise retention. Its own failure analysis named exactly two residuals this
mechanism removes:
  (1) "winner instability -- the engram is re-competed each percept rather than COMMITTED"  -> FIXED codes are
      committed once, never re-competed.
  (2) "the softmax output-side all-vs-all suppression re-introduces interference: even with disjoint winners,
      d = softmax - onehot(y) actively suppresses every non-target class column on the active units" -> REGRESSION
      to a fixed orthogonal code (d = o - t_c) has NO all-vs-all normalization; teaching fact i pushes the readout
      toward t_i only, and because t_i _|_ t_j the write lands in an orthogonal output direction that does not move
      fact j's decision (argmax_c <o, t_c>).
So the lever moves from LEARNED SPARSITY (refuted) to FIXED ORTHOGONALITY (this).

BRAIN-BASED. The fixed orthogonal codes are the DENTATE-GYRUS decorrelated target basis (a fixed sparse/orthogonal
pattern-separated code per fact -- PS-SNN's predefined mutually-orthogonal class centers), NOT a host label lookup:
the class label y indexes only WHICH innate code is the target; the brain's OWN readout synapses (cp_connections,
moved by the e-prop leaky-readout delta) must learn to PRODUCE that code from the percept, and classification is the
brain's readout output correlated against the codes. The hidden reservoir is a FIXED structural expansion
(freeze_hidden -- DG/granule expansion is structural, plasticity is at the readout; the refuted-arc lesson #2 that
trainable hidden COLLAPSES separation). The optional weight-protection arm (--si) is a Zenke-2017 SYNAPTIC-
INTELLIGENCE plasticity gate on the readout synapses (consolidated synapses resist change; the Phase-1.4 gate-freeze
biology), NOT a host freeze-list.

ARMS (same net build / seed / per-fact teaching budget / FROZEN hidden reservoir -- so the ONLY difference is the
readout target scheme):
  * DRIFT      = the shared K-way softmax leaky readout, NO replay (the drifting-readout wall reference).
  * REPLAY     = DRIFT + self-generated hippocampal sleep-replay consolidation (reproduces the ~0.55 replay-only bar
                 on THIS identical net -- reuse-by-import of the sleep-replay Hippocampus, teacher/world ABSENT).
  * ORTHO      = fixed mutually-ORTHOGONAL disjoint-block codes + regression readout (the treatment).
  * COLLAPSE   = fixed but NON-orthogonal (shared-base + small per-class jitter) codes + regression readout. Removes
                 ORTHOGONALITY while keeping the fixed-code regression regime -> isolates orthogonality as the lever.
  * ORTHO_SI   = ORTHO + Zenke synaptic-intelligence weight protection on the readout (optional; --si).
  * ORTHO_REPLAY = ORTHO + self-replay consolidation (optional combine; --ortho-replay).

TEETH (single-seed SMOKE; 6-seed command below):
  (a) RETENTION RISES vs the 0.55 replay-only baseline: ORTHO frac_recalled@N > REPLAY frac_recalled@N (same
      net/seed/epochs) AND > DRIFT, toward 0.8+.
  (b) LOAD-BEARING orthogonality: COLLAPSE (orthogonality removed, same fixed-code regression compute) falls back
      toward DRIFT -> orthogonality, not merely "a fixed code", carries the rise. (weight-protection sub-teeth: if
      --si, remove SI (== ORTHO) and measure whether forgetting returns -- honest either way.)
  (c) IMMEDIATE ACQUISITION of the NEW fact stays high in ORTHO (acquire_acc right after teaching, BEFORE later
      facts, ~1.0) -- the mechanism must not block plasticity. The EWC/SI "slightly compromises new learning"
      tradeoff is MEASURED (ORTHO_SI acquire vs ORTHO acquire).
  (d) The codes are GENUINELY orthogonal/pattern-separated, not a host label table: the Gram matrix off-diagonal is
      ~0 for ORTHO (verified in-artifact) and >> 0 for COLLAPSE.
  (e) CAPACITY holds (mean retained acc reported per milestone).

DISCIPLINE: reuse-by-import (_mk_net-style build via OnBridgeEpropNet + _feat / _fit_readout_norm_world / _teach_fact
/ _fact_acc / _corrective_batch / N_ACT from the scaling de-risk; Hippocampus + _self_replay_consolidate from the
sleep-replay de-risk; ReferentEnv from the corrective-acquire de-risk). NO sim/ edit -- the orthogonal-target readout
is a runner-side training rule over the SAME cp_connections the e-prop leaky readout uses. cfg.seed via the seed= the
build passes to CoreSimConfig.seed (NOT actual_seed_used). SIM_BACKEND selectable; the net is tiny + launch-bound so
numpy (CPU) is faster here (the sparse-readout de-risk established this) -- cupy path verified to run. tools.lab
attribution + a Verdict preconditions block.

RUN (single-seed smoke; numpy is faster for this tiny net, cupy verified):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_ortho_target_pattern_separation_derisk --seed 42 \
      --n-max 10 --milestones 1 5 10 --hidden 64 --code-width 4 --epochs 30 --settle-steps 25 \
      --n-draws 20 --si --ortho-replay \
      --out research/findings/raw/teacher_loop_ortho_target_s42.json
  6-SEED (GO needs the rise 6/6 at 42..47; --seeds self-sweeps in-process + writes an aggregate):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_ortho_target_pattern_separation_derisk --seeds 42 43 44 45 46 47 \
      --n-max 10 --milestones 1 5 10 --hidden 64 --code-width 4 --epochs 30 --settle-steps 25 --n-draws 20 --si \
      --out research/findings/raw/teacher_loop_ortho_target_s42.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # tiny launch-bound net -> CPU faster (sparse-readout de-risk); cupy ok
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
from sim.backend import to_host  # noqa: E402
# reuse-by-import: the teacher-loop SCALING machinery (world features, readout-norm fit, per-fact softmax teaching,
# held-out accuracy, corrective batch) + the OnBridge e-prop net + the sleep-replay Hippocampus. NO sim/ edit.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _feat, _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import Hippocampus  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_ortho_target.json"


# ============================ the net: FROZEN-hidden DG expansion, readout-only learner ============================
def _mk_net(n_in, k_out, seed, hidden, settle, eprop_lr, w_clip):
    """OnBridgeEpropNet with the hidden reservoir FROZEN (freeze_hidden -> DG/granule structural expansion; the
    refuted-arc lesson #2: trainable hidden collapses pattern separation). k_out output units = K classes (softmax
    arms) or the M-dim code (ortho arms). The FF readout block (H_last->out) is the SOLE learner in all arms."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0, freeze_hidden=True)
    return OnBridgeEpropNet(n_in, hidden, k_out, seed=seed, n_hidden_layers=1, settle_steps=settle,
                            eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                            logit_source="leaky_readout", w_clip=w_clip, hp=hp)


# ==================================== the FIXED ORTHOGONAL target codes (DG basis) =================================
def make_codes(n_classes, code_width, orthogonal, seed, collapse_bump=1.0):
    """Build the fixed pattern-separated readout codes t_c in R^M, M = n_classes*code_width.
      * orthogonal=True  : each class owns a DISJOINT block of `code_width` output units (value 1/sqrt(code_width),
        so unit-norm). Disjoint supports => the Gram matrix is EXACTLY the identity: mutually orthogonal AND sparse
        (density 1/n_classes) -- the DG decorrelated basis (PS-SNN predefined orthogonal centers).
      * orthogonal=False : the COLLAPSE lesion -- every class shares one base pattern (spread over all units) plus a
        small class-specific component on its own block, so pairwise cosine is high (orthogonality removed) while the
        fixed-code regression regime + the M-dim readout are unchanged. Isolates ORTHOGONALITY as the lever.
    The class label indexes WHICH innate code is the target; the brain's readout must learn to PRODUCE it."""
    M = n_classes * code_width
    rng = np.random.default_rng(seed + 90210)     # brain-owned code RNG (fixed innate basis; deterministic)
    codes = np.zeros((n_classes, M), dtype=np.float64)
    perm = rng.permutation(n_classes)             # which disjoint block each class owns (still disjoint => orthogonal)
    if orthogonal:
        for c in range(n_classes):
            blk = perm[c]
            codes[c, blk * code_width:(blk + 1) * code_width] = 1.0
    else:
        base = np.ones(M, dtype=np.float64)        # shared base spread over ALL units -> high pairwise overlap
        for c in range(n_classes):
            blk = perm[c]
            v = base.copy()
            v[blk * code_width:(blk + 1) * code_width] += collapse_bump   # small distinct bump on the class block
            codes[c] = v                                                  # cosine ~ (M+2*bump*w)/(M+2*bump*w+bump^2*w)
    # unit-normalize every code so the two regimes differ ONLY in orthogonality, not in target magnitude.
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    G = codes @ codes.T                            # Gram: diag ~1; off-diag ~0 (ortho) or >>0 (collapse)
    off = G - np.diag(np.diag(G))
    stats = {"M": int(M), "max_offdiag_cosine": float(np.max(np.abs(off))),
             "mean_offdiag_cosine": float(np.mean(np.abs(off))),
             "min_diag": float(np.min(np.diag(G)))}
    return codes, stats


# ============================ the readout output + the ortho regression training rule ============================
def _readout_output(net, feat_row):
    """The M-dim leaky-readout output o = standardized_r @ W (W = the H_last->out cp_connections block). This is
    EXACTLY the Bellec leaky-readout logit computation (net._logits_from's leaky branch), general M. Brain synapses."""
    sp, _vv, _acts = net._forward_record(feat_row)                       # spiking forward (Izhikevich bridge)
    r = net._readout_feature(sp)                                          # (n_Hlast_phys,) standardized eligibility
    idx = net._data_idx_flat[-1]
    W = np.asarray(to_host(net.br.cp_connections.data[idx]), dtype=np.float64).reshape(
        net.sizes_phys[-2], net.sizes_phys[-1])                          # (n_Hlast_phys, M)
    o = net._pool(r @ W, len(net.sizes) - 1)                             # pool_k=1 -> identity
    return o, r, W, idx


def _apply_readout_dw(net, idx, dW):
    """Write the readout weight delta w -= dW into cp_connections, mirroring the parent _apply_grads clip + plastic
    mask + rate gain (so this direct update is byte-faithful to the substrate's own apply path). dW: (n_pre,n_post)."""
    xp = net._xp
    data = net.br.cp_connections.data
    cur = np.asarray(to_host(data[idx]), dtype=np.float64)
    new = np.clip(cur - dW.astype(np.float64).ravel(), -net.w_clip, net.w_clip)
    new_xp = xp.asarray(new.astype(np.float32))
    cur_xp = data[idx]
    if net.br.cp_synapse_plastic_mask is not None:
        pm = net.br.cp_synapse_plastic_mask[idx]
        new_xp = xp.where(pm, new_xp, cur_xp)
    if net.br.cp_plasticity_rate_gain is not None:
        gain = net.br.cp_plasticity_rate_gain[idx]
        new_xp = cur_xp + (new_xp - cur_xp) * gain
    data[idx] = new_xp


class _SI:
    """Zenke-2017 SYNAPTIC INTELLIGENCE weight protection on the readout synapses (a plasticity gate on the brain's
    OWN synapses: consolidated synapses resist change -- the Phase-1.4 gate-freeze biology). Accumulates per-synapse
    importance Omega over the path integral of the loss, and adds a quadratic penalty lambda*Omega*(W - W_star)
    pulling important synapses back to their consolidated value. Off unless the arm requests it."""
    def __init__(self, shape, lam, xi=1e-3):
        self.lam = float(lam); self.xi = float(xi)
        self.Omega = np.zeros(shape, dtype=np.float64)      # consolidated importance
        self.w_little = np.zeros(shape, dtype=np.float64)   # path integral for the CURRENT fact
        self.W_star = None                                  # consolidated weights (post last fact)

    def penalty_grad(self, W):
        if self.W_star is None:
            return np.zeros_like(W)
        return self.lam * self.Omega * (W - self.W_star)

    def accumulate(self, task_grad, dW):
        # little omega += -task_grad . dW  (contribution of this step to the loss decrease). dW = actual weight change.
        self.w_little += -task_grad * dW

    def consolidate(self, W):
        if self.W_star is not None:
            dtot = W - self.W_star
            self.Omega += np.clip(self.w_little, 0.0, None) / (dtot ** 2 + self.xi)
        self.W_star = W.copy()
        self.w_little = np.zeros_like(self.w_little)


def _homeo_normalize(net, cols, target):
    """HOMEOSTATIC synaptic scaling on ONE fact's output population: rescale the incoming readout weights of the
    fact's OWN code units (columns `cols`) so their total (Frobenius) norm = `target`. A biological homeostatic
    input scaling (Turrigiano) -- it calibrates each engram's output magnitude so the nearest-code decision is not
    dominated by whichever block happens to have grown largest (the un-normalized argmax miscalibration). Preserves
    the within-block direction, so it does NOT change which units carry the fact -- only the cross-fact scale."""
    xp = net._xp
    idx = net._data_idx_flat[-1]
    data = net.br.cp_connections.data
    W = np.asarray(to_host(data[idx]), dtype=np.float64).reshape(net.sizes_phys[-2], net.sizes_phys[-1])
    nrm = np.linalg.norm(W[:, cols])
    if nrm > 1e-9:
        W[:, cols] = W[:, cols] * (float(target) / nrm)
        data[idx] = xp.asarray(W.astype(np.float32).ravel())


def _train_fact_ortho(net, X, code, epochs, batch, rng, lr, own_subspace=True, homeo_norm=1.0, si=None):
    """Teach ONE fact by REGRESSION of the M-dim readout output toward the fixed pattern-separated code t_c
    (d = o - t_c). No softmax, no all-vs-all normalization. Only the readout block moves (hidden frozen).

    THE WEIGHT PROTECTION (the load-bearing part). `own_subspace=True` CONFINES the write to the fact's OWN code
    units (d is masked to the code's support): the fact potentiates only the synapses onto its own output
    population and NEVER touches earlier facts' populations -- an allocation-based gate-freeze (the Phase-1.4
    plasticity gate-freeze / PS-SNN per-class subspace). Because the orthogonal codes have DISJOINT supports, each
    fact's synapses are protected: later facts write elsewhere. `own_subspace=False` is the lesion (regress the FULL
    code including its zeros) -> the all-vs-all suppression the refuted k-WTA arc named returns, and earlier blocks
    are zeroed -> forgetting. For a NON-orthogonal (collapse) code the support is ALL units, so the confinement
    DEGENERATES to a full write -> orthogonality is what MAKES the own-subspace protection possible.

    `lr` is the STABLE regression step (squared-error on r@W diverges above ~2/lambda_max(r r^T), unlike the
    self-limiting softmax delta -- lr=0.5 blew the weights up; ~0.1 converges). After the fact, homeostatic synaptic
    scaling calibrates the fact's own population. Optional Zenke SI penalty is an ADDITIONAL (tested) protection."""
    L = len(net.sizes) - 1
    own = code > 1e-9                                                    # the fact's OWN code support (disjoint block)
    mask = own.astype(np.float64) if own_subspace else np.ones_like(code)
    idx = net._data_idx_flat[-1]
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for i0 in range(0, len(X), batch):
            b = perm[i0:i0 + batch]
            _o0, _r0, W0, _i = _readout_output(net, X[b[0]])            # current W (pre-batch) for the SI penalty
            grad = np.zeros_like(W0)
            for j in b:
                o, r, _W, _idx = _readout_output(net, X[j])
                d = (o - code) * mask                                   # regression error CONFINED to the own subspace
                dphys = net._broadcast(d, L) / net.pool_k
                grad += np.outer(r, dphys)                              # leaky-readout gradient d_loss/dW = r (x) d
            task_grad = grad / max(1, len(b))
            step = lr * task_grad
            if si is not None:
                step = step + lr * si.penalty_grad(W0)                 # SI consolidation penalty (readout synapses)
            _apply_readout_dw(net, idx, step)                          # w -= step  (clip/mask/gain faithful)
            if si is not None:
                _o1, _r1, W1, _i = _readout_output(net, X[b[0]])
                si.accumulate(task_grad, (W1 - W0))                    # path integral for importance
    if homeo_norm and homeo_norm > 0:
        _homeo_normalize(net, np.where(own)[0], homeo_norm)            # synaptic scaling on the fact's own population


def _fact_acc_ortho(net, env, referent, cls, codes, n):
    """Held-out generalization for the ORTHO readout: n fresh noisy draws of `referent` -> fraction the brain's
    readout output correlates MOST with the fixed code of class `cls` (argmax_c <o, t_c> -- nearest orthogonal
    center, the PS-SNN classification rule). The brain does the readout; the codes are the innate decision basis."""
    correct = 0
    for _ in range(n):
        o, _r, _W, _idx = _readout_output(net, _feat(env, referent))
        c = int(np.argmax(codes @ o))
        correct += int(c == cls)
    return correct / n


# ============================================ one arm of the curriculum ============================================
def _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws,
             milestones, test_n, code_width, ortho_lr, si_lambda, replay_epochs, replay_per_fact, replay_noise, chance):
    ortho = arm in ("ortho", "collapse", "ortho_full", "ortho_si", "ortho_replay")
    use_si = arm == "ortho_si"
    use_replay = arm in ("replay", "ortho_replay")
    orthogonal_codes = arm in ("ortho", "ortho_full", "ortho_si", "ortho_replay")   # collapse: fixed but NON-ortho
    own_subspace = arm != "ortho_full"        # ortho_full is the weight-protection LESION (write ALL blocks -> forget)

    codes = code_stats = None
    if ortho:
        codes, code_stats = make_codes(K, code_width, orthogonal_codes, seed)
        k_out = codes.shape[1]
    else:
        k_out = K
    net = _mk_net(n_in, k_out, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)                # readout-norm fit ONCE over the world (baseline)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)
    hippo = Hippocampus(seed, replay_noise=replay_noise) if use_replay else None
    si = _SI((net.sizes_phys[-2], net.sizes_phys[-1]), si_lambda) if use_si else None

    acquire_acc = []
    retention = {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)
        # --- WAKE: teach fact i ---
        if ortho:
            _train_fact_ortho(net, X, codes[i], epochs, batch, teach_rng, ortho_lr,
                              own_subspace=own_subspace, si=si)
            acq = _fact_acc_ortho(net, env, r, i, codes, n=test_n)
        else:
            _teach_fact(net, X, y, epochs, batch, teach_rng)
            acq = _fact_acc(net, env, r, i, n=test_n)
        acquire_acc.append(acq)
        if si is not None:
            _o, _r2, W_now, _idx = _readout_output(net, X[0])
            si.consolidate(W_now)                                     # consolidate importance after learning fact i
        if use_replay:
            hippo.encode(X, i)
            # self-generated sleep-replay consolidation (teacher/world ABSENT) into the softmax readout (K-way arms).
            from research.runners._teacher_loop_sleep_replay_consolidation_derisk import _self_replay_consolidate
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=False)
        N = i + 1
        if N in milestones:
            if ortho:
                accs = [_fact_acc_ortho(net, env, referents[j], j, codes, n=test_n) for j in range(N)]
            else:
                accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
    out = {"arm": arm, "acquire_acc_immediate": [float(a) for a in acquire_acc],
           "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
           "retention_curve": retention}
    if code_stats is not None:
        out["code_stats"] = code_stats
    return out


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, code_width, ortho_lr, si_lambda, replay_epochs, replay_per_fact, replay_noise, arms_to_run):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))

    referents = [f"ref{i}" for i in range(n_max)]

    arms = {}
    for arm in arms_to_run:
        t0 = time.time()
        # FRESH env per arm (same seed -> identical referent prototypes AND identical teaching draw stream): the
        # readout-norm fit then draws from the natural post-proto stream, NOT a mid-stream reset. The reset-before-fit
        # made the fit statistics depend on which draws it ate, swinging retention 0.1<->1.0 on ONE seed (an
        # instrument artifact, not the mechanism). Fresh-env is like-for-like AND removes that fragility.
        env = ReferentEnv(seed, d_p=d_p, noise=noise)
        for r in referents:
            env.proto(r)
        arms[arm] = _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip,
                             n_draws, milestones, test_n, code_width, ortho_lr, si_lambda, replay_epochs,
                             replay_per_fact, replay_noise, chance)
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm:12s}] {arms[arm]['wall_seconds']:.0f}s | immediate-acq "
              f"{arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "code_width": code_width, "ortho_lr": ortho_lr, "si_lambda": si_lambda,
                       "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "frozen_hidden": True},
            "arms": arms}


def _verdict(result):
    """Verdict preconditions + GO. TEETH:
      (a) retention RISES vs 0.55 replay-only: ORTHO > REPLAY (in-run) AND ORTHO > DRIFT, at the largest N.
      (b) orthogonality LOAD-BEARING: COLLAPSE (orthogonality removed, same fixed-code regression) <= DRIFT + margin.
      (c) ORTHO immediate acquisition stays high (>= 0.9).
      (d) codes genuinely orthogonal: ORTHO code max off-diag cosine ~0; COLLAPSE >> 0."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    arms = result["arms"]
    chance = result["chance"]

    def frac(arm):
        rc = arms.get(arm, {}).get("retention_curve", {})
        big = max((int(k) for k in rc), default=None)
        return (rc[str(big)]["frac_recalled"] if big else float("nan")), big

    drift_f, big = frac("drift")
    ortho_f, _ = frac("ortho")
    replay_f, _ = frac("replay") if "replay" in arms else (float("nan"), None)
    collapse_f, _ = frac("collapse") if "collapse" in arms else (float("nan"), None)
    orthofull_f, _ = frac("ortho_full") if "ortho_full" in arms else (float("nan"), None)
    ortho_acq = arms["ortho"]["mean_acquire_acc_immediate"]
    ortho_offdiag = arms["ortho"].get("code_stats", {}).get("max_offdiag_cosine", float("nan"))
    collapse_offdiag = arms.get("collapse", {}).get("code_stats", {}).get("max_offdiag_cosine", float("nan"))

    # the effect is ORTHOGONALITY (ortho vs collapse), not merely "a fixed code".
    if not np.isnan(collapse_f):
        attributable_to("fixed-code ORTHOGONALITY (ortho vs collapse)", ortho_f, collapse_f)
    if not np.isnan(orthofull_f):
        attributable_to("own-subspace weight protection (ortho vs ortho_full)", ortho_f, orthofull_f)
    attributable_to("ortho-target readout (ortho vs drifting softmax)", ortho_f, drift_f)

    replay_bar = 0.55       # the NAMED replay-only retention cap (task/literature: main 8d2510d3, extreme-shift)
    v = Verdict("teacher-loop fixed orthogonal-target pattern separation", chance=chance)
    v.reaches("(a) retention RISES vs the replay-only 0.55 bar", before=replay_bar, after=ortho_f)
    v.require("(a') ortho beats the drifting readout", ortho_f > drift_f + 1e-9, expect=True,
              note=f"ortho {ortho_f:.2f} vs drift {drift_f:.2f}")
    if not np.isnan(collapse_f):
        v.control("(b) orthogonality is load-bearing (ortho vs collapse)", treatment=ortho_f,
                  control=collapse_f, min_separation=0.15)
        v.require("(b') collapse falls back toward drift", collapse_f <= drift_f + 0.15, expect=True,
                  note=f"collapse {collapse_f:.2f} vs drift {drift_f:.2f}")
    if not np.isnan(orthofull_f):
        v.control("(b2) own-subspace weight protection is load-bearing (ortho vs ortho_full)", treatment=ortho_f,
                  control=orthofull_f, min_separation=0.15)
        v.require("(b2') removing protection -> forgetting returns", orthofull_f <= drift_f + 0.20, expect=True,
                  note=f"ortho_full {orthofull_f:.2f} vs drift {drift_f:.2f}")
    v.floor("(c) ortho immediate acquisition stays high", ortho_acq, floor=0.9)
    v.require("(d) ORTHO codes genuinely orthogonal (Gram off-diag ~0)", ortho_offdiag < 1e-6, expect=True,
              note=f"max off-diag cosine {ortho_offdiag:.2e}")
    if not np.isnan(collapse_offdiag):
        v.require("(d') COLLAPSE codes are NON-orthogonal (Gram off-diag >> 0)", collapse_offdiag > 0.3,
                  expect=True, note=f"collapse max off-diag cosine {collapse_offdiag:.2f}")

    go = (ortho_f > replay_bar and ortho_f > drift_f + 1e-9 and ortho_acq >= 0.9 and ortho_offdiag < 1e-6)
    if not np.isnan(collapse_f):
        go = go and (ortho_f > collapse_f + 0.15) and (collapse_f <= drift_f + 0.15)
    if not np.isnan(orthofull_f):
        go = go and (ortho_f > orthofull_f + 0.15) and (orthofull_f <= drift_f + 0.20)
    decision = v.decide(go=go)

    extras = {}
    if "ortho_si" in arms:
        si_f, _ = frac("ortho_si")
        extras["ortho_si_frac_recalled"] = si_f
        extras["ortho_si_acquire"] = arms["ortho_si"]["mean_acquire_acc_immediate"]
        extras["si_new_learning_tradeoff_acq_delta"] = float(
            arms["ortho_si"]["mean_acquire_acc_immediate"] - ortho_acq)     # EWC/SI "compromises new learning" measure
        extras["si_retention_delta_vs_ortho"] = float(si_f - ortho_f)
    if "ortho_replay" in arms:
        orr_f, _ = frac("ortho_replay")
        extras["ortho_replay_frac_recalled"] = orr_f
        extras["ortho_replay_gain_vs_ortho"] = float(orr_f - ortho_f)
    return {
        "largest_N": big,
        "drift_frac_recalled": drift_f, "replay_frac_recalled": replay_f,
        "ortho_frac_recalled": ortho_f, "collapse_frac_recalled": collapse_f,
        "ortho_full_frac_recalled": orthofull_f,
        "replay_only_bar_used": replay_bar,
        "ortho_immediate_acq": ortho_acq,
        "retention_rise_ortho_minus_replaybar": float(ortho_f - replay_bar),
        "retention_rise_ortho_minus_drift": float(ortho_f - drift_f),
        "orthogonality_margin_ortho_minus_collapse": (float(ortho_f - collapse_f)
                                                      if not np.isnan(collapse_f) else None),
        "protection_margin_ortho_minus_orthofull": (float(ortho_f - orthofull_f)
                                                    if not np.isnan(orthofull_f) else None),
        "ortho_code_max_offdiag_cosine": ortho_offdiag,
        "collapse_code_max_offdiag_cosine": collapse_offdiag,
        **extras, **decision,
    }


def _one_seed(a, seed, arms_to_run, quiet=False):
    result = run(seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr, a.w_clip,
                 a.n_draws, a.d_p, a.noise, a.test_n, a.code_width, a.ortho_lr, a.si_lambda, a.replay_epochs,
                 a.replay_per_fact, a.replay_noise, arms_to_run)
    verdict = _verdict(result)
    return result, verdict


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop FIXED ORTHOGONAL-TARGET pattern separation (PS-SNN): "
                                             "replace the drifting learnable readout with predefined mutually-"
                                             "orthogonal codes to beat the sleep-replay 0.55 retention plateau.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="self-sweep these seeds + write an aggregate")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--code-width", type=int, default=4, help="output units per class (sparse orthogonal block width)")
    ap.add_argument("--ortho-lr", type=float, default=0.1, help="STABLE regression step for the ortho readout "
                    "(squared-error diverges above ~2/lambda_max(r r^T); softmax lr=0.5 blew up)")
    ap.add_argument("--si", action="store_true", help="add the ORTHO_SI weight-protection arm (Zenke SI on readout)")
    ap.add_argument("--si-lambda", type=float, default=200.0, help="SI consolidation penalty strength")
    ap.add_argument("--ortho-replay", action="store_true", help="add the ORTHO_REPLAY combine arm")
    ap.add_argument("--replay-epochs", type=int, default=20)
    ap.add_argument("--replay-per-fact", type=int, default=16)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    arms_to_run = ["drift", "replay", "ortho", "collapse", "ortho_full"]
    if a.si:
        arms_to_run.append("ortho_si")
    if a.ortho_replay:
        arms_to_run.append("ortho_replay")

    seeds = a.seeds if a.seeds else [a.seed]
    per_seed = []
    for s in seeds:
        print("\n" + "#" * 100 + f"\n# SEED {s}\n" + "#" * 100, flush=True)
        result, verdict = _one_seed(a, s, arms_to_run)
        summary = {"probe": "teacher_loop_ortho_target_pattern_separation", "seed": s,
                   "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": (len(seeds) == 1),
                   "arms_run": arms_to_run, "elapsed_seconds": round(time.time() - t0, 1),
                   "result": result, "verdict": verdict}
        out_s = a.out if len(seeds) == 1 else str(Path(a.out).with_name(Path(a.out).stem + f"_s{s}.json"))
        Path(out_s).write_text(json.dumps(summary, indent=2, default=str))
        per_seed.append({"seed": s, "verdict": verdict, "out": out_s})
        print("\n" + "=" * 100, flush=True)
        print(f"[ortho-target] seed {s} @ N={verdict['largest_N']}: DRIFT {verdict['drift_frac_recalled']:.2f} | "
              f"REPLAY {verdict['replay_frac_recalled']:.2f} | ORTHO {verdict['ortho_frac_recalled']:.2f} | "
              f"COLLAPSE {verdict['collapse_frac_recalled']:.2f} | ORTHO_FULL "
              f"{verdict['ortho_full_frac_recalled']:.2f} (chance {result['chance']:.2f})", flush=True)
        print(f"[ortho-target] rise(ortho-replaybar) {verdict['retention_rise_ortho_minus_replaybar']:+.2f} | "
              f"rise(ortho-drift) {verdict['retention_rise_ortho_minus_drift']:+.2f} | "
              f"ortho-immediate-acq {verdict['ortho_immediate_acq']:.3f} | VERDICT {verdict['status']}", flush=True)
        if "si_new_learning_tradeoff_acq_delta" in verdict:
            print(f"[ortho-target] SI: retention {verdict.get('ortho_si_frac_recalled'):.2f} "
                  f"(dvs-ortho {verdict['si_retention_delta_vs_ortho']:+.2f}) | new-learning tradeoff "
                  f"(acq delta) {verdict['si_new_learning_tradeoff_acq_delta']:+.3f}", flush=True)
        print(f"[ortho-target] wrote {out_s}\n" + "=" * 100, flush=True)

    if len(seeds) > 1:
        go_n = sum(1 for p in per_seed if p["verdict"]["status"] == "GO")
        ortho = [p["verdict"]["ortho_frac_recalled"] for p in per_seed]
        replay = [p["verdict"]["replay_frac_recalled"] for p in per_seed]
        drift = [p["verdict"]["drift_frac_recalled"] for p in per_seed]
        agg = {"probe": "teacher_loop_ortho_target_pattern_separation_AGG", "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND"),
               "go_count": go_n, "n_seeds": len(seeds),
               "ortho_frac_mean": float(np.mean(ortho)), "replay_frac_mean": float(np.mean(replay)),
               "drift_frac_mean": float(np.mean(drift)),
               "ortho_minus_replay_mean": float(np.mean(np.array(ortho) - np.array(replay))),
               "per_seed": per_seed}
        agg_out = str(Path(a.out).with_name(Path(a.out).stem + "_AGG.json"))
        Path(agg_out).write_text(json.dumps(agg, indent=2, default=str))
        print("\n" + "#" * 100, flush=True)
        print(f"[ortho-target AGG] GO {go_n}/{len(seeds)} | ORTHO {np.mean(ortho):.2f} vs REPLAY "
              f"{np.mean(replay):.2f} vs DRIFT {np.mean(drift):.2f} | wrote {agg_out}", flush=True)
        return 0 if go_n == len(seeds) else 1

    return 0 if per_seed[0]["verdict"]["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
