"""gap#4 ON-BRIDGE SPIKING self-predicting microcircuit -- the DECISIVE test of the gap#4 hypothesis (BUILD-AHEAD
SKELETON; the full multi-seed GPU run is the controller's, launched once the 3090 frees).

WHY THIS RUN (the thesis): the CPU numpy-RATE phase is a 6/6-seed GO
(`_gap4_learned_microcircuit_selfpredict_derisk` + `2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO.md`)
BUT its load-bearing FLAG G2 is: at RATE the LEARNED feedback arms (kp / plastic-Eq.9 micro) are ACCURACY-BYTE-IDENTICAL
to plain fixed-random feedback-alignment -- the learned/interneuron machinery is INERT on the feedforward weights at
rate. The learned-vs-fixed SEPARATION only appears ON SPIKES, where the burst-coded finite-sample saturating credit
degrades a fixed-random projection's alignment in a way a learned/aligned projection resists. ⇒ THIS on-bridge SPIKING
run is the one that actually tests gap#4. Full design: `2026-07-24-gap4-onbridge-spiking-port-DESIGN.md`.

REUSE-BY-IMPORT (NO `sim/` edit anywhere -- all BDSP machinery is the committed additive/default-off flags in
`sim/config.py` + `sim/bridge.py`):
  - `_semantic_inheritance_onbridge_spiking_derisk.OnBridgeBDSPNet` -- the depth-2 two-compartment spiking net on ONE
    `SimulationBridge` (input(features) -> H1 -> H2 -> out), population coding, fixed-random Y credit descent,
    cp_bdsp_apical_drive / cp_bdsp_int_drive wiring, graded-credit flag. THIS runner SUBCLASSES it and adds the
    gap#4-specific feedback modes + the genuinely-new plastic-Eq.9 W^PI.
  - `_semantic_inheritance_deep_credit_derisk.{make_task_semantic_inheritance}` -- the XOR-over-pool
    compositional-inheritance task + the held-out-inheritance metric (the only proven depth-required AND
    transport-free-learnable instrument).
  - `sim.dendritic_mlp.DendriticMLP` -- the fenced backprop ORACLE ceiling (task validity).

ARMS (held-out INHERITANCE accuracy on spikes, the GO metric):
  reservoir         : hidden apical = 0 (H1,H2 FROZEN at random init) ; only the H2->out readout learns  = the credit-INDEPENDENT baseline.
  fixed_fa          : fixed-random Y feedback, sequential-FA descent, graded BDSP FF plasticity on spikes  = the fixed-feedback credit (the G2 control to BEAT).
  kp                : Kolen-Pollack LEARNED feedback dY = kp_lr*outer(e_above,E) - kp_decay*Y (Y^T -> W, transport-free)  = learned DIRECTION.
  micro             : plastic-Eq.9 self-predicting interneuron W^PI at the TOP hidden layer (apical-silent EARNED)        = the genuinely-new build.
  transport_ceiling : Y := (pooled forward W)^T each step (weight transport ~ backprop)                                    = the labeled CHEAT (its no-transport guard MUST fail).

GO-GATE (the decisive spiking claim): best(kp, micro) held-out > fixed_fa on spikes by a real margin, >= 5/6 seeds, in a
regime where fixed_fa <= reservoir + margin (the spiking FA-wall the learned feedback closes), task valid (oracle
>= 0.80), ALL anti-cheats pass. An honest negative (learned == fixed at every reachable op-point) IS the deliverable.

ANTI-CHEATS: fixed-FA control (the baseline) ; transport-lesion/ceiling (guard MUST fail on the ceiling; structural
AST + separate-RNG guard on kp/micro) ; shuffle (shuffled-target -> chance; shufE directed-credit scramble -> collapse)
; no-plasticity frozen (bdsp_learning_rate 0 / apical-lesion) ; apical-silent EARNED (plastic W^PI silent_ratio <<
frozen-noisy, selfpred_cos -> 1) ; memorization control (untaught super stays ~chance).

SEED DISCIPLINE (a real bug): the reused base sets `cfg.seed` (the field the bridge reads); `actual_seed_used` SEEDS
NOTHING. The construct-smoke asserts two builds at one seed produce byte-identical `cp_neuron_firing_thresholds`.

HONEST SCOPE: the FEEDFORWARD weight changes (the deep credit) are carried by the spiking substrate (fused_bdsp_update
over cp_connections, driven by each layer's apical-modulated graded burst deviation). The credit PROJECTION (Y @ error,
the W^PI cancellation VALUE, the KP/W^PI updates) is host-computed (as in every D1/EMERGE reference); it is
transport-free + LOCAL (no host backprop / no W^T chain). A fully-on-substrate spiking interneuron projection is the
deeper follow-on. See the DESIGN doc S7.

RUN -- construct-smoke ONLY (this is a BUILD-AHEAD skeleton; the full GPU run is deferred):
    SIM_BACKEND=numpy python -m research.runners._gap4_onbridge_spiking_selfpredict_derisk --construct-smoke

The full 6-seed GPU de-risk (the CONTROLLER's, once the 3090 frees -- one process per seed):
    for s in 42 43 44 100 101 102; do
      SIM_BACKEND=cupy python -u -m research.runners._gap4_onbridge_spiking_selfpredict_derisk \
          --seeds $s --arms reservoir fixed_fa kp micro transport_ceiling \
          --hidden 64 --pool-k 16 --n-hidden-layers 2 --epochs 40 --graded-credit --wpi-plastic --wpi-init noisy \
          --assert-no-transport --out research/findings/raw/gap4/onbridge_spiking_seed$s.json &
    done; wait
"""
from __future__ import annotations
import argparse, ast, inspect, json, os, sys, textwrap, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
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
from research.runners._semantic_inheritance_onbridge_spiking_derisk import (  # noqa: E402
    OnBridgeBDSPNet, _softmax)
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance)

OUT = _REPO / "research" / "findings" / "raw" / "gap4" / "onbridge_spiking_selfpredict.json"

# feedback modes -> the parent `rule` used at construct time (only 'micro' needs enable_bdsp_microcircuit on-bridge).
_ARM_RULE = {"reservoir": "plain_fa", "fixed_fa": "plain_fa", "kp": "plain_fa",
             "micro": "microcircuit", "transport_ceiling": "plain_fa"}


def _cos(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-12 else 0.0


# ============================================================================================================
# The gap#4 on-bridge net: OnBridgeBDSPNet + the gap#4 feedback modes + the genuinely-new plastic-Eq.9 W^PI.
# ============================================================================================================
class Gap4OnBridgeNet(OnBridgeBDSPNet):
    """OnBridgeBDSPNet with a gap#4 `feedback` mode dispatch. `feedback` in {reservoir, fixed, learned (KP), micro
    (plastic-Eq.9 W^PI), transport_ceiling}. The FEEDFORWARD plasticity is the committed spiking `enable_bdsp` rule
    (unchanged from the parent); ONLY how the descending apical credit (+ the W^PI interneuron cancellation for micro)
    is set differs. NO `sim/` edit."""

    def __init__(self, n_in, hidden, k, seed=0, feedback="fixed", wpi_plastic=False, wpi_init="noisy",
                 wpi_lr=0.2, wpi_noise=1.0, kp_lr=0.2, kp_decay=1e-4, **kw):
        super().__init__(n_in, hidden, k, seed=seed, rule=_ARM_RULE.get(feedback, "plain_fa"), **kw)
        self.feedback = str(feedback)
        self.kp_lr = float(kp_lr); self.kp_decay = float(kp_decay)
        self.wpi_plastic = bool(wpi_plastic); self.wpi_init = str(wpi_init); self.wpi_lr = float(wpi_lr)
        # plastic-Eq.9 interneuron cancellation W^PI at the TOP hidden layer (shape == Y[top] == (k_classes, H_top)).
        # 'noisy' (default): OFF the self-predicting fixed point -> silence must be EARNED. 'fixedpoint': W^PI := Y (silent from step 0, the positive control).
        top = self.n_hidden_layers - 1
        Ytop = self.Y[top]
        wrng = np.random.default_rng(seed + 4242)            # SEPARATE stream (no transport)
        self.W_PI = Ytop.copy() if wpi_init == "fixedpoint" else wrng.normal(0.0, wpi_noise, Ytop.shape)
        self._selfpred_cos_traj = []

    # ---- LOCAL, transport-free feedback/interneuron updates (their OWN methods so the AST guard can inspect them) ----
    def _kp_update_Y(self, k, pre_rate, post_err):
        """Kolen-Pollack learned apical feedback for Y[k] (feedback='learned'). dY = kp_lr*(post^T @ pre) - kp_decay*Y
        drives Y^T -> W by a LOCAL anti-Hebbian rule (Akrout weight-mirror). TRANSPORT-FREE: reads ONLY the layer event
        rate pre_rate, the descending error post_err, and Y -- NEVER a forward weight (self.br.cp_connections)."""
        pre = np.atleast_2d(np.asarray(pre_rate)); post = np.atleast_2d(np.asarray(post_err))
        m = max(1, pre.shape[0])
        outer = (post.T @ pre) / m                            # (sizes[k+2], sizes[k+1]) == Y[k].shape ; LOCAL only
        self.Y[k] = self.Y[k] + self.lr * (self.kp_lr * outer - self.kp_decay * self.Y[k])

    def _wpi_selfpredict_update(self, src_pred):
        """The local, transport-free Sacramento-Senn Eq.9 self-prediction update for the top-layer W^PI:
        dW^PI = +wpi_lr*lr*( r_int^T @ v_free ), r_int = src_pred (interneuron rate = the net's own softmax), v_free =
        src_pred @ (Y_top - W^PI) (the free-phase residual apical). Drives W^PI -> Y_top (self-prediction). LOCAL +
        TRANSPORT-FREE: reads ONLY src_pred (an activity), self.Y[top], self.W_PI -- NEVER self.br.cp_connections."""
        top = self.n_hidden_layers - 1
        sp = np.atleast_2d(np.asarray(src_pred))
        m = max(1, sp.shape[0])
        v_free = sp @ (self.Y[top] - self.W_PI)               # (1, H_top)
        dWpi = (sp.T @ v_free) / m                            # (k_classes, H_top) == W^PI.shape
        self.W_PI = self.W_PI + self.wpi_lr * self.lr * dWpi

    # ---- transport ceiling: sync Y := (pooled forward W)^T (the CHEAT; the no-transport guard MUST fail) ----
    def _read_ff_logical(self, li, _coo_cache=None):
        """Reconstruct the LOGICAL-unit (sizes[li] x sizes[li+1]) forward weight of pathway ff_{li} from cp_connections
        (block-mean over the K-pools). Used ONLY by transport_ceiling to copy Y := W^T (the labeled cheat).

        VECTORIZED (2026-07-24 gap#4 surpass): the pre/post index ranges are CONTIGUOUS (arange from inject), so a
        boolean mask on the COO + offset-subtracted fancy-index reconstructs the dense block WITHOUT the per-edge Python
        loop (the old loop over ~all edges x per-example x per-Y was the transport_ceiling launch-cost). Byte-identical
        output. `_coo_cache` = (row, col, data) host arrays transferred once by _sync_transport (avoids re-transfer)."""
        from sim.backend import to_host
        if _coo_cache is not None:
            row, col, data = _coo_cache
        else:
            coo = self.br._get_cached_coo()
            row = np.asarray(to_host(coo.row)).astype(np.int64)
            col = np.asarray(to_host(coo.col)).astype(np.int64)
            data = np.asarray(to_host(self.br.cp_connections.data)).astype(np.float64)
        pre, post = self._ff_edges[li]
        r0, r1 = int(pre[0]), int(pre[-1]) + 1
        c0, c1 = int(post[0]), int(post[-1]) + 1
        mask = (row >= r0) & (row < r1) & (col >= c0) & (col < c1)
        M = np.zeros((len(pre), len(post)))
        M[row[mask] - r0, col[mask] - c0] = data[mask]        # each (r,c) appears once (dense pathway)
        K = self.pool_k
        if K > 1:
            M = M.reshape(self.sizes[li], K, self.sizes[li + 1], K).mean(axis=(1, 3))
        return M                                              # (sizes[li], sizes[li+1]) logical

    def _sync_transport(self):
        # Y[k] descends from layer k+2 to k+1; the matching forward pathway is ff_{k+1} (layer k+1 -> k+2), logical
        # shape (sizes[k+1], sizes[k+2]) -> its transpose is (sizes[k+2], sizes[k+1]) == Y[k].shape. Copy it => transport.
        # Transfer the COO to host ONCE (not once per Y matrix) -> the vectorized _read_ff_logical reconstructs each block.
        from sim.backend import to_host
        coo = self.br._get_cached_coo()
        cache = (np.asarray(to_host(coo.row)).astype(np.int64),
                 np.asarray(to_host(coo.col)).astype(np.int64),
                 np.asarray(to_host(self.br.cp_connections.data)).astype(np.float64))
        for k in range(len(self.Y)):
            self.Y[k] = self._read_ff_logical(k + 1, _coo_cache=cache).T.copy()

    # ---- the gap#4 per-example online credit pass (reuses the parent helpers; NO parent _train_one call) ----
    def _train_one(self, feat_row, y, mode):
        from sim.backend import to_host  # noqa: F401 (parity with the parent; reads are via helpers below)
        xp = self._xp; n = self.n_total
        if self.feedback == "transport_ceiling":
            self._sync_transport()                            # descend through Y == W^T (transport used)
        acts = self._forward_spiking(feat_row)                # per-slice pooled event rates (logical)
        logits = acts[-1][None, :]
        src_pred = _softmax(logits)                           # (1,k) softmax = interneuron rate / the net's prediction
        onehot = np.zeros((1, self.k)); onehot[0, int(y)] = 1.0
        delta_out = src_pred.copy(); delta_out[0, int(y)] -= 1.0    # (1,k) +gradient at the output
        if mode == "wrong_sign":
            delta_out = -delta_out; onehot = src_pred + delta_out    # keep the teacher consistent with the flip
        e_upper = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        if mode == "shufE" and getattr(self, "_shuf_perm", None) is not None:
            e_upper = e_upper[:, self._shuf_perm]             # scramble the descending error's class assignment (hidden path)

        apical = np.zeros(n, dtype=np.float64)
        int_drive = np.zeros(n, dtype=np.float64)
        nhid = self.n_hidden_layers

        # OUTPUT-layer credit (all arms incl. reservoir: the output "has target access" -> the readout learns).
        E_out = acts[-1][None, :]
        out_err = (E_out * (1.0 - E_out)) * e_upper
        apical[self.slices[-1]] = self.apical_gain_pA * self._broadcast(out_err[0], len(self.sizes) - 1)

        # HIDDEN descent (top -> bottom). reservoir: leave hidden apical == 0 (dev == 0 -> H1,H2 frozen at random init).
        if self.feedback != "reservoir":
            for k in range(nhid - 1, -1, -1):
                li = k + 1
                E = acts[li][None, :]
                phi = E * (1.0 - E)
                if self.feedback == "learned" and mode == "bdsp":
                    self._kp_update_Y(k, E, e_upper)          # KP learned feedback (transport-free), before using Y[k]
                Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
                if self.feedback == "micro" and k == nhid - 1:
                    # MICROCIRCUIT top layer: deliver drive (raw top-down) and int_drive (interneuron cancellation)
                    # SEPARATELY -> the committed enable_bdsp_microcircuit block integrates (drive - int_drive) into
                    # cp_v_apical. At W^PI == Y this == phi*((onehot - src_pred)@Y) == the clean FA credit, SILENT when
                    # correct. The plastic W^PI (noisy init) EARNS that silence.
                    raw = phi * (onehot @ Yk)                 # (1,size) raw top-down teaching
                    canc = phi * (src_pred @ self.W_PI)       # (1,size) interneuron prediction (cancellation)
                    apical[self.slices[li]] = self.apical_gain_pA * self._broadcast(raw[0], li)
                    int_drive[self.slices[li]] = self.apical_gain_pA * self._broadcast(canc[0], li)
                    if self.wpi_plastic and mode == "bdsp":
                        self._wpi_selfpredict_update(src_pred)
                        self._selfpred_cos_traj.append(_cos(self.W_PI, self.Y[k]))
                    soma_err = phi * ((onehot - src_pred) @ Yk)   # descend the CANCELLED clean error
                else:
                    v_api = e_upper @ Yk                      # (1,size) clean apical error (weighted sum, low-noise)
                    soma_err = phi * v_api
                    apical[self.slices[li]] = self.apical_gain_pA * self._broadcast(soma_err[0], li)
                e_upper = soma_err

        # inject the apical (+ int_drive for micro) and run the credit steps WHILE this example's input still drives.
        ap = np.zeros(n, dtype=np.float32); ap[:] = apical
        self.br.cp_bdsp_apical_drive = xp.asarray(ap)
        if self.feedback == "micro":
            it = np.zeros(n, dtype=np.float32); it[:] = int_drive
            self.br.cp_bdsp_int_drive = xp.asarray(it)
        drive = self._base_drive()
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * np.asarray(feat_row, np.float32), 0.0, 1600.0)
        drive[self.slices[0]] = self._broadcast(in_cur, 0).astype(np.float32)
        self.br.cp_external_input_current = xp.asarray(drive)
        for _ in range(self.credit_steps):
            self.br._run_one_simulation_step()
        if self.br.cp_bdsp_apical_drive is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        if self.br.cp_bdsp_int_drive is not None:
            self.br.cp_bdsp_int_drive[...] = 0.0

    # ---- the genuinely-new RATE-observable analogue read on-bridge: apical-silent-when-correct ----
    def apical_silent_stats(self, X, y):
        """mean |effective apical| (drive - int = src_target@Y - src_pred@W^PI at the top hidden layer) on CORRECT vs
        INCORRECT outputs. EARNED-silent => correct << incorrect (silent_ratio small) with selfpred_cos(W^PI,Y_top)->1.
        Host read of the injected-credit magnitude (matches the CPU MicroNet observable; a physical cp_v_apical read is
        the follow-on)."""
        X = np.asarray(X); y = np.asarray(y)
        top = self.n_hidden_layers - 1
        mags = []; corr = []
        acts_b = self._forward_batch(X)
        for i in range(len(X)):
            lg = acts_b[-1][i]
            sp = _softmax(lg[None, :])[0]
            st = np.zeros(self.k); st[int(y[i])] = 1.0
            v_apical = st @ self.Y[top] - sp @ self.W_PI      # (H_top,)
            mags.append(float(np.abs(v_apical).mean()))
            corr.append(bool(int(np.argmax(lg)) == int(y[i])))
        mags = np.asarray(mags); corr = np.asarray(corr)
        mc = float(mags[corr].mean()) if corr.any() else float("nan")
        mi = float(mags[~corr].mean()) if (~corr).any() else float("nan")
        ratio = float(mc / (mi + 1e-12)) if (corr.any() and (~corr).any()) else float("nan")
        return {"apical_correct": mc, "apical_incorrect": mi, "silent_ratio": ratio,
                "frac_correct": float(corr.mean()), "selfpred_cos": _cos(self.W_PI, self.Y[top])}

    # ---- anti-cheat: no weight transport (structural) ----
    def no_weight_transport(self):
        if self.feedback == "transport_ceiling":
            return False                                      # Y := W^T by construction -> the guard MUST report a violation
        if self.feedback in ("learned", "micro"):
            # kp/micro LEARN their feedback (Y / W^PI) -> the byte 'not-transpose' check is inapplicable; the guarantee
            # is STRUCTURAL: the update methods read ONLY activities + Y/W^PI, never a forward-W array. AST-verify.
            return bool(_ast_no_forward_W(type(self)))
        return True                                           # fixed/reservoir: Y from a separate stream, never written


# ============================================================================================================
# Anti-cheat AST guard: the feedback/interneuron update methods never read a forward-weight array.
# ============================================================================================================
def _ast_no_forward_W(cls, method_names=("_kp_update_Y", "_wpi_selfpredict_update")):
    """The KP / W^PI update methods must NEVER read the forward weights (self.br.cp_connections / self._ff_edges /
    self._read_ff_logical). AST-walk each method for those attribute/name reads; assert none (a forward-W read in a
    feedback update = backprop-in-disguise / weight transport)."""
    forbidden_attrs = {"cp_connections", "_read_ff_logical", "_ff_edges", "_sync_transport"}
    for name in method_names:
        meth = getattr(cls, name, None)
        if meth is None:
            continue
        try:
            src = textwrap.dedent(inspect.getsource(meth))
        except (OSError, TypeError):
            continue
        # drop the docstring (its prose legitimately mentions cp_connections).
        tree = ast.parse(src)
        fn = tree.body[0]
        if (getattr(fn, "body", None) and isinstance(fn.body[0], ast.Expr)
                and isinstance(getattr(fn.body[0], "value", None), ast.Constant)
                and isinstance(fn.body[0].value.value, str)):
            src = src.replace(fn.body[0].value.value, "")
            tree = ast.parse(textwrap.dedent(src))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                return False
    return True


# ============================================================================================================
# Training / evaluation (the STRUCTURE for the full run; only the construct-smoke RUNS locally in this skeleton).
# ============================================================================================================
def _train_arm(net, Xtr, ytr, mode, epochs, batch, seed):
    rng = np.random.default_rng(seed + 777)
    if mode == "shufE":
        net._shuf_perm = np.random.default_rng(seed * 4099 + 11).permutation(net.k)
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            net.train_step(Xtr[b], ytr[b], mode=mode)


def _build_net(feedback, n_in, k, args, seed):
    net = Gap4OnBridgeNet(
        n_in, args.hidden, k, seed=seed, feedback=feedback,
        n_hidden_layers=args.n_hidden_layers, pool_k=args.pool_k,
        settle_steps=args.settle_steps, credit_steps=args.credit_steps, lr=args.lr,
        in_current_pA=args.in_current_pA, in_bias_pA=args.in_bias_pA, apical_gain_pA=args.apical_gain_pA,
        tonic_h_pA=args.tonic_h_pA, tonic_o_pA=args.tonic_o_pA, beta=args.beta, p0=args.p0,
        graded_credit=args.graded_credit,
        wpi_plastic=args.wpi_plastic, wpi_init=args.wpi_init, wpi_lr=args.wpi_lr,
        kp_lr=args.kp_lr, kp_decay=args.kp_decay)
    # WIDEN THE BDSP FF WEIGHT CLAMP (gap#4 surpass, 2026-07-24): the parent hardcodes cfg.bdsp_w_min/max = +-6; the
    # kernel reads getattr(cfg, "bdsp_w_max", ...) DYNAMICALLY each step (bridge.py:8009) and net.cfg IS the bridge's
    # core_config, so setting it here takes effect. gap#5 found the +-5/6 clamp caps even at lr=0 -> it may cap the
    # learnable weights on a multi-way task. --bdsp-w-max defaults to 6.0 (== parent behavior; byte-identical when
    # unspecified). Additive/runner-only, NO sim/ edit.
    wmax = float(getattr(args, "bdsp_w_max", 6.0))
    net.cfg.bdsp_w_max = wmax
    net.cfg.bdsp_w_min = -wmax
    return net


def run_seed(seed, args):
    """The per-seed FULL run (arms + anti-cheats + GO components). NOTE: this is the CONTROLLER's GPU run; the skeleton
    ships it for launch-readiness but the DEFAULT entrypoint is the construct-smoke (main --construct-smoke)."""
    from sim.dendritic_mlp import DendriticMLP
    from sim.backend import to_host          # DendriticMLP routes through the pluggable backend -> its logits are cupy on GPU
    t_seed = time.time()
    tk = dict(n_super=args.n_super, n_members=args.n_members, held_per_super=args.held_per_super,
              n_prop=args.n_prop, member_id_dim=args.member_id_dim, n_obs=args.n_obs, noise=args.noise)
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte), meta, idx = make_task_semantic_inheritance(seed, **tk)
    n_in = Xtr.shape[1]; k = int(meta["k_classes"]); inh = idx["inh_idx"]
    chance = float(max(np.mean(yte[inh] == c) for c in np.unique(yte[inh]))) if len(inh) else float("nan")

    # ORACLE ceiling (fenced backprop) -- task validity. Always the FULL train set (the 0.80 ceiling must be honest).
    onet = DendriticMLP([n_in, args.hidden, args.hidden, k], seed=seed)
    r = np.random.default_rng(seed + 777)
    for _ in range(args.oracle_epochs):
        p = r.permutation(len(ytr))
        for i in range(0, len(ytr), args.oracle_batch):
            b = p[i:i + args.oracle_batch]
            onet.train_step(Xtr[b], ytr[b], mode="oracle", lr=args.oracle_lr)
    _, olg = onet._forward(np.asarray(Xte[inh], float))
    olg = np.asarray(to_host(olg))            # cupy -> host so the argmax/compare is a plain numpy op on GPU too
    oracle = float(np.mean(np.argmax(olg, 1) == yte[inh])) if len(inh) else float("nan")
    print(f"[gap4-onbridge][seed {seed}] task n_in={n_in} k={k} n_train={len(ytr)} n_inh={len(inh)} chance={chance:.3f} "
          f"| ORACLE {oracle:.3f} ({time.time()-t_seed:.0f}s)", flush=True)

    # RUNTIME lever: subsample the TRAIN set the ON-BRIDGE spiking arms see (per-example spiking is the bottleneck;
    # the reference on-bridge runner subsamples identically). 0 => FULL train. Oracle above always uses FULL.
    Xtr_b, ytr_b = Xtr, ytr
    if args.train_subsample and args.train_subsample > 0 and len(Xtr) > args.train_subsample:
        srng = np.random.default_rng(seed * 13 + 1)
        keep = srng.permutation(len(Xtr))[:args.train_subsample]
        Xtr_b, ytr_b = Xtr[keep], ytr[keep]
        print(f"[gap4-onbridge][seed {seed}] on-bridge arms train on subsample {len(ytr_b)}/{len(ytr)}", flush=True)

    arms = {}; nets = {}
    for arm in args.arms:
        t_arm = time.time()
        net = _build_net(arm, n_in, k, args, seed)
        w0 = net.ff_weight_norm()                        # FF |w| sum before training (learning diagnostic)
        _train_arm(net, Xtr_b, ytr_b, "bdsp", args.epochs, args.batch, seed)
        w1 = net.ff_weight_norm()
        arms[arm] = {"inherit_heldout": float(net.acc_on(Xte, yte, inh)),
                     "memctrl_heldout": float(net.acc_on(Xte, yte, idx["memctrl_idx"])),
                     "train_acc": float(net.accuracy(Xtr_b, ytr_b)),
                     "ff_weight_moved": float(abs(w1 - w0)),
                     "no_weight_transport": bool(net.no_weight_transport())}
        nets[arm] = net
        print(f"[gap4-onbridge][seed {seed}]   arm {arm:<17} held-out {arms[arm]['inherit_heldout']:.3f} "
              f"train {arms[arm]['train_acc']:.3f} memctrl {arms[arm]['memctrl_heldout']:.3f} "
              f"ff-moved {arms[arm]['ff_weight_moved']:.2f} nwt {arms[arm]['no_weight_transport']} "
              f"({time.time()-t_arm:.0f}s)", flush=True)

    # anti-cheat arms on the base credit net (micro if present, else fixed_fa). SELECTIVE SCALE-UP lever
    # (--core-arms-only): skip the 4 anti-cheat nets (lesion/shuf_tgt/shufE + micro-frozen) so ALL the compute goes to
    # reservoir/fixed_fa/kp/micro at higher epochs/data -- put the power where the signal is; re-add anti-cheats on the
    # winning config (per the power-limited-pass interpretation rule).
    base = "micro" if "micro" in args.arms else ("fixed_fa" if "fixed_fa" in args.arms else args.arms[0])
    les_acc = shuf_acc = shufE_acc = float("nan")
    apical = {}
    if not getattr(args, "core_arms_only", False):
        lesion = _build_net(base, n_in, k, args, seed); _train_arm(lesion, Xtr_b, ytr_b, "apical_lesion", args.epochs, args.batch, seed)
        les_acc = float(lesion.acc_on(Xte, yte, inh))
        prng = np.random.default_rng(seed + 555); yperm = ytr_b[prng.permutation(len(ytr_b))]
        shuftgt = _build_net(base, n_in, k, args, seed); _train_arm(shuftgt, Xtr_b, yperm, "bdsp", args.epochs, args.batch, seed)
        shuf_acc = float(shuftgt.acc_on(Xte, yte, inh))
        shufE = _build_net(base, n_in, k, args, seed); _train_arm(shufE, Xtr_b, ytr_b, "shufE", args.epochs, args.batch, seed)
        shufE_acc = float(shufE.acc_on(Xte, yte, inh))
        print(f"[gap4-onbridge][seed {seed}]   anti-cheats: lesion {les_acc:.3f} shuf_tgt {shuf_acc:.3f} "
              f"shufE {shufE_acc:.3f}", flush=True)
        if "micro" in nets:
            apical["micro_plastic"] = nets["micro"].apical_silent_stats(Xte, yte)
            frozen = _build_net("micro", n_in, k, args, seed); frozen.wpi_plastic = False
            _train_arm(frozen, Xtr_b, ytr_b, "bdsp", args.epochs, args.batch, seed)
            apical["micro_frozen_wpi"] = frozen.apical_silent_stats(Xte, yte)
    elif "micro" in nets:
        apical["micro_plastic"] = nets["micro"].apical_silent_stats(Xte, yte)   # cheap host read; keep even in core-only
        print(f"[gap4-onbridge][seed {seed}]   (core-arms-only: anti-cheat nets skipped)", flush=True)

    # ---- the LOAD-BEARING FA-wall precondition + the GO read (per seed) ----
    res_acc = arms.get("reservoir", {}).get("inherit_heldout", float("nan"))
    ff_acc = arms.get("fixed_fa", {}).get("inherit_heldout", float("nan"))
    learned = [arms[a]["inherit_heldout"] for a in ("kp", "micro") if a in arms]
    best_learned = max(learned) if learned else float("nan")
    fa_wall = bool(ff_acc <= res_acc + 0.02)      # fixed_fa degraded to <= reservoir => a gap for learned to close
    seed_go = bool(fa_wall and best_learned > ff_acc + 0.05)
    print(f"[gap4-onbridge][seed {seed}] FA-WALL: fixed_fa {ff_acc:.3f} vs reservoir {res_acc:.3f} "
          f"(fa_wall={fa_wall}) | best_learned(kp,micro) {best_learned:.3f} vs fixed_fa {ff_acc:.3f} "
          f"-> seed_go={seed_go} ({time.time()-t_seed:.0f}s total)", flush=True)

    return {"seed": seed, "meta": meta, "chance": chance, "oracle_heldout": oracle, "n_in": n_in, "k": k,
            "arms": arms,
            "lesion": {"inherit_heldout": les_acc},
            "shuffled_target": {"inherit_heldout": shuf_acc},
            "shufE": {"inherit_heldout": shufE_acc},
            "apical": apical,
            "fa_wall": {"reservoir": res_acc, "fixed_fa": ff_acc, "best_learned": best_learned,
                        "fa_wall_holds": fa_wall, "seed_go": seed_go},
            "elapsed_seconds": round(time.time() - t_seed, 1),
            "guards": {"ast_no_forward_W": bool(_ast_no_forward_W(Gap4OnBridgeNet))}}


# ============================================================================================================
# CONSTRUCT-SMOKE (the ONLY thing this build-ahead skeleton RUNS locally): build EACH arm at TINY sizes + step it a
# few times + read held-out + check the no-transport guard + (micro) the W^PI update + apical-silent read. Proves the
# pipeline CONSTRUCTS + STEPS without crashing. NOT an accuracy run.
# ============================================================================================================
def construct_smoke(args):
    print("=" * 108, flush=True)
    print("[gap4-onbridge-smoke] CONSTRUCT-SMOKE (build + step only; NOT the accuracy science run).", flush=True)
    seed = args.seeds[0]
    # tiny task: n_super=8 (>= 2^n_prop), n_prop=2 -> k=5 classes, few members/obs.
    tk = dict(n_super=8, n_members=4, held_per_super=1, n_prop=2, member_id_dim=3, n_obs=4, noise=0.02)
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte), meta, idx = make_task_semantic_inheritance(seed, **tk)
    n_in = Xtr.shape[1]; k = int(meta["k_classes"]); inh = idx["inh_idx"]
    print(f"[gap4-onbridge-smoke] task: n_in={n_in} k={k} n_train={len(ytr)} n_inh_heldout={len(inh)}", flush=True)

    # tiny net knobs (override args for the smoke): small hidden, K=1, tiny settle/credit windows, 2 samples, 1 epoch.
    class _A:
        pass
    a = _A()
    for attr in vars(args):
        setattr(a, attr, getattr(args, attr))
    a.hidden = 4; a.pool_k = 1; a.n_hidden_layers = 2; a.settle_steps = 4; a.credit_steps = 4; a.lr = 0.05
    a.wpi_plastic = True; a.wpi_init = "noisy"

    # ---- (1) two-build seed identity (the substrate-actually-seeded check) ----
    n1 = _build_net("fixed_fa", n_in, k, a, seed)
    n2 = _build_net("fixed_fa", n_in, k, a, seed)
    from sim.backend import to_host
    import hashlib
    def _thr_hash(net):
        thr = getattr(net.br, "cp_neuron_firing_thresholds", None)
        if thr is None:
            return None
        return hashlib.md5(np.asarray(to_host(thr)).tobytes()).hexdigest()[:16]
    h1, h2 = _thr_hash(n1), _thr_hash(n2)
    seed_ok = (h1 is not None and h1 == h2)
    print(f"[gap4-onbridge-smoke] seed check: cp_neuron_firing_thresholds md5 {h1} vs {h2} -> "
          f"{'IDENTICAL (cfg.seed controls the substrate)' if seed_ok else 'DIFFER / absent'}", flush=True)

    results = {}
    Xs, ys = Xtr[:2], ytr[:2]                                 # 2 train samples for the step-smoke
    for arm in args.arms:
        rec = {"built": False, "stepped": False, "heldout_read": None, "no_weight_transport": None, "error": None}
        try:
            net = _build_net(arm, n_in, k, a, seed)
            rec["built"] = True
            # step: one full per-example credit pass per sample (build + step without crashing = the smoke bar).
            for xi in range(len(Xs)):
                net._train_one(Xs[xi], int(ys[xi]), "bdsp")
            rec["stepped"] = True
            rec["heldout_read"] = float(net.acc_on(Xte, yte, inh)) if len(inh) else None
            rec["no_weight_transport"] = bool(net.no_weight_transport())
            if arm == "micro":
                # exercise the plastic-Eq.9 W^PI update + the apical-silent read explicitly.
                cos0 = _cos(net.W_PI, net.Y[net.n_hidden_layers - 1])
                net._wpi_selfpredict_update(_softmax(np.eye(k)[:1]))
                cos1 = _cos(net.W_PI, net.Y[net.n_hidden_layers - 1])
                sil = net.apical_silent_stats(Xte, yte)
                rec["wpi_selfpred_cos_before_after"] = [round(cos0, 4), round(cos1, 4)]
                rec["apical_silent_read"] = {kk: (round(vv, 4) if isinstance(vv, float) and not np.isnan(vv) else vv)
                                             for kk, vv in sil.items()}
        except Exception as e:
            rec["error"] = repr(e)
            traceback.print_exc()
        results[arm] = rec
        status = ("OK" if (rec["built"] and rec["stepped"] and rec["error"] is None) else "FAIL")
        extra = ""
        if arm == "transport_ceiling":
            extra = f" (no_weight_transport={rec['no_weight_transport']} -- MUST be False = guard correctly flags the cheat)"
        if arm == "micro" and rec.get("apical_silent_read"):
            extra = f" (wpi_cos {rec.get('wpi_selfpred_cos_before_after')}, silent_ratio {rec['apical_silent_read'].get('silent_ratio')})"
        print(f"[gap4-onbridge-smoke]   arm {arm:<17} built={rec['built']} stepped={rec['stepped']} "
              f"heldout_read={rec['heldout_read']} -> {status}{extra}", flush=True)

    ast_ok = bool(_ast_no_forward_W(Gap4OnBridgeNet))
    ceiling_ok = (results.get("transport_ceiling", {}).get("no_weight_transport") is False
                  if "transport_ceiling" in args.arms else None)
    all_ok = all(r["built"] and r["stepped"] and r["error"] is None for r in results.values()) and seed_ok and ast_ok
    print(f"[gap4-onbridge-smoke] AST no-forward-W guard on the KP/W^PI updates: {ast_ok}", flush=True)
    out = {"probe": "gap4_onbridge_spiking_selfpredict_CONSTRUCT_SMOKE", "seed": seed, "task_meta": meta,
           "seed_identity_ok": bool(seed_ok), "ast_no_forward_W": ast_ok,
           "transport_ceiling_guard_correctly_fails": ceiling_ok, "arms": results,
           "CONSTRUCT_SMOKE_PASS": bool(all_ok),
           "NOTE": ("BUILD-AHEAD skeleton -- this proves the on-bridge microcircuit CONSTRUCTS + STEPS on CPU (numpy) "
                    "without crashing. The multi-seed GPU accuracy run (the decisive gap#4 test) is the CONTROLLER's, "
                    "deferred until the 3090 frees. See 2026-07-24-gap4-onbridge-spiking-port-DESIGN.md.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print("=" * 108, flush=True)
    print(f"[gap4-onbridge-smoke] CONSTRUCT_SMOKE_PASS={all_ok}  (seed_ok={seed_ok}, ast_ok={ast_ok}, "
          f"ceiling_guard_fails={ceiling_ok})", flush=True)
    print(f"[gap4-onbridge-smoke] wrote {args.out}", flush=True)
    print("=" * 108, flush=True)
    return 0 if all_ok else 1


def main():
    ap = argparse.ArgumentParser(description="gap#4 on-bridge spiking self-predicting microcircuit (BUILD-AHEAD skeleton).")
    ap.add_argument("--construct-smoke", action="store_true",
                    help="the ONLY locally-run mode: build each arm at tiny sizes + step it (build+step smoke). Default action.")
    ap.add_argument("--full", action="store_true",
                    help="run the per-seed FULL science (arms + anti-cheats). This is the CONTROLLER's GPU run -- NOT run in build-ahead.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arms", nargs="+", default=["reservoir", "fixed_fa", "kp", "micro", "transport_ceiling"])
    # net scale
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=16)
    ap.add_argument("--n-hidden-layers", dest="n_hidden_layers", type=int, default=2)
    ap.add_argument("--settle-steps", dest="settle_steps", type=int, default=40)
    ap.add_argument("--credit-steps", dest="credit_steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--train-subsample", dest="train_subsample", type=int, default=0,
                    help="subsample the TRAIN set the on-bridge spiking arms see (0=full; runtime lever, oracle uses full).")
    ap.add_argument("--core-arms-only", dest="core_arms_only", action="store_true",
                    help="skip the 4 anti-cheat nets (lesion/shuf_tgt/shufE/micro-frozen) -> all compute on the core arms "
                         "at higher epochs/data (selective scale-up; re-add anti-cheats on the winning config).")
    # BDSP + drive knobs (the committed sim/ knobs + the drive levers)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bdsp-w-max", dest="bdsp_w_max", type=float, default=6.0,
                    help="BDSP FF weight clamp magnitude (cfg.bdsp_w_min/max = -/+ this). Default 6.0 == parent behavior; "
                         "widen (e.g. 20-50) to test whether the +-5/6 clamp caps learning on a multi-way task (gap#5 clip).")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--in-current-pA", dest="in_current_pA", type=float, default=520.0)
    ap.add_argument("--in-bias-pA", dest="in_bias_pA", type=float, default=260.0)
    ap.add_argument("--apical-gain-pA", dest="apical_gain_pA", type=float, default=900.0)
    ap.add_argument("--tonic-h-pA", dest="tonic_h_pA", type=float, default=450.0)
    ap.add_argument("--tonic-o-pA", dest="tonic_o_pA", type=float, default=500.0)
    ap.add_argument("--graded-credit", dest="graded_credit", action="store_true", default=True,
                    help="enable_bdsp_graded_credit (clean-error credit E*(P-Pbar); default ON for gap#4).")
    ap.add_argument("--no-graded-credit", dest="graded_credit", action="store_false")
    # learned-feedback knobs
    ap.add_argument("--wpi-plastic", dest="wpi_plastic", action="store_true", default=False,
                    help="plastic-Eq.9 W^PI (apical-silent EARNED) for the micro arm; default off => W^PI held at init.")
    ap.add_argument("--wpi-init", dest="wpi_init", default="noisy", choices=["noisy", "fixedpoint"])
    ap.add_argument("--wpi-lr", dest="wpi_lr", type=float, default=0.2)
    ap.add_argument("--kp-lr", dest="kp_lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", dest="kp_decay", type=float, default=1e-4)
    ap.add_argument("--assert-no-transport", dest="assert_no_transport", action="store_true")
    # task knobs
    ap.add_argument("--n-super", dest="n_super", type=int, default=24)
    ap.add_argument("--n-members", dest="n_members", type=int, default=8)
    ap.add_argument("--held-per-super", dest="held_per_super", type=int, default=3)
    ap.add_argument("--n-prop", dest="n_prop", type=int, default=3)
    ap.add_argument("--member-id-dim", dest="member_id_dim", type=int, default=3)
    ap.add_argument("--n-obs", dest="n_obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    # oracle
    ap.add_argument("--oracle-epochs", dest="oracle_epochs", type=int, default=250)
    ap.add_argument("--oracle-lr", dest="oracle_lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", dest="oracle_batch", type=int, default=128)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.assert_no_transport:
        assert _ast_no_forward_W(Gap4OnBridgeNet), "AST guard FAILED: a KP/W^PI update reads a forward-weight array"

    if a.full:
        # THE CONTROLLER's GPU run -- guarded so build-ahead cannot accidentally spend the science.
        t0 = time.time(); per = []; err = None
        try:
            for s in a.seeds:
                per.append(run_seed(s, a))
        except Exception as e:
            err = repr(e); traceback.print_exc()

        agg = {}
        if per:
            def _m(keys):
                vals = []
                for p in per:
                    v = p
                    ok = True
                    for kk in keys:
                        if isinstance(v, dict) and kk in v:
                            v = v[kk]
                        else:
                            ok = False; break
                    if ok and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(float(v))
                return float(np.mean(vals)) if vals else float("nan")
            n_seed = len(per)
            n_fa_wall = sum(bool(p["fa_wall"]["fa_wall_holds"]) for p in per)
            n_seed_go = sum(bool(p["fa_wall"]["seed_go"]) for p in per)
            oracle = _m(["oracle_heldout"]); chance = _m(["chance"])
            res = _m(["arms", "reservoir", "inherit_heldout"]); ff = _m(["arms", "fixed_fa", "inherit_heldout"])
            kp = _m(["arms", "kp", "inherit_heldout"]); mic = _m(["arms", "micro", "inherit_heldout"])
            ceil = _m(["arms", "transport_ceiling", "inherit_heldout"])
            les = _m(["lesion", "inherit_heldout"]); shuf = _m(["shuffled_target", "inherit_heldout"])
            shufE = _m(["shufE", "inherit_heldout"]); memctrl = _m(["arms", "fixed_fa", "memctrl_heldout"])
            best_learned = np.nanmax([kp, mic])
            mp_ratio = _m(["apical", "micro_plastic", "silent_ratio"]); mp_cos = _m(["apical", "micro_plastic", "selfpred_cos"])
            mf_ratio = _m(["apical", "micro_frozen_wpi", "silent_ratio"]); mf_cos = _m(["apical", "micro_frozen_wpi", "selfpred_cos"])
            # anti-cheat / guard checks
            ceiling_guard_fails = all(p["arms"].get("transport_ceiling", {}).get("no_weight_transport") is False
                                      for p in per if "transport_ceiling" in p["arms"])
            ast_ok = all(p["guards"]["ast_no_forward_W"] for p in per)
            lesion_collapse = bool(les <= chance + 0.10); shuf_collapse = bool(shuf <= chance + 0.10)
            shufE_collapse = bool(shufE <= chance + 0.10); memctrl_ok = bool(memctrl <= chance + 0.10)
            task_ok = bool(oracle >= 0.80)
            apical_earned = (bool(mp_ratio < 0.5 and mp_cos > 0.6 and mf_ratio > 0.8)
                             if not np.isnan(mp_ratio) else None)
            GO = bool(task_ok and n_fa_wall >= 5 and n_seed_go >= 5 and ceiling_guard_fails and ast_ok
                      and lesion_collapse and shuf_collapse and memctrl_ok)
            agg = {"n_seeds": n_seed, "n_fa_wall": n_fa_wall, "n_seed_go": n_seed_go,
                   "oracle": oracle, "chance": chance, "reservoir": res, "fixed_fa": ff, "kp": kp, "micro": mic,
                   "best_learned": float(best_learned), "transport_ceiling": ceil,
                   "lesion": les, "shuffled_target": shuf, "shufE": shufE, "memctrl": memctrl,
                   "micro_plastic_silent_ratio": mp_ratio, "micro_plastic_selfpred_cos": mp_cos,
                   "micro_frozen_silent_ratio": mf_ratio, "micro_frozen_selfpred_cos": mf_cos,
                   "fa_wall_holds_all": bool(n_fa_wall >= 5), "task_ok": task_ok,
                   "ceiling_guard_correctly_fails": bool(ceiling_guard_fails), "ast_no_forward_W": bool(ast_ok),
                   "lesion_collapse": lesion_collapse, "shuffled_collapse": shuf_collapse,
                   "shufE_collapse": shufE_collapse, "memctrl_ok": memctrl_ok, "apical_silent_earned": apical_earned,
                   "GO": GO,
                   "verdict": (f"{'GO' if GO else 'NO-GO'} ({n_seed_go}/{n_seed} seed_go; FA-wall {n_fa_wall}/{n_seed}). "
                               f"reservoir {res:.3f} | fixed_fa {ff:.3f} | kp {kp:.3f} | micro {mic:.3f} | ceiling "
                               f"{ceil:.3f} | oracle {oracle:.3f} (chance {chance:.3f}). apical-silent EARNED: plastic "
                               f"ratio {mp_ratio:.3f} (cos {mp_cos:.2f}) vs frozen {mf_ratio:.3f} (cos {mf_cos:.2f}). "
                               f"anti-cheats: lesion {les:.3f} shuf {shuf:.3f} shufE {shufE:.3f} memctrl {memctrl:.3f}; "
                               f"ceiling-guard-fails {ceiling_guard_fails}; ast {ast_ok}.")}
        summary = {"probe": "gap4_onbridge_spiking_selfpredict_FULL", "seeds": a.seeds, "config": vars(a),
                   "elapsed_seconds": round(time.time() - t0, 1), "error": err, "per_seed": per, "aggregate": agg,
                   "GO": bool(agg.get("GO", False)),
                   "NOTE": ("GO = best(kp,micro) > fixed_fa on >=5/6 seeds IN the FA-wall regime (fixed_fa <= reservoir) "
                            "+ all anti-cheats; see 2026-07-24-gap4-onbridge-spiking-port-DESIGN.md S5.")}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
        print("=" * 108, flush=True)
        print(f"[gap4-onbridge] {agg.get('verdict', 'ERROR: ' + str(err))}", flush=True)
        print(f"[gap4-onbridge] FULL run wrote {a.out}", flush=True)
        print("=" * 108, flush=True)
        return 0 if agg.get("GO") else 1

    # default: the construct-smoke.
    return construct_smoke(a)


if __name__ == "__main__":
    sys.exit(main())
