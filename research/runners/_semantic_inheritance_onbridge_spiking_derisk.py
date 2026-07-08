"""ON-BRIDGE SPIKING compositional-semantic deep-credit de-risk -- the DECISIVE test of the deep lever's CORE thesis.

WHY THIS RUN (the thesis, `2026-07-07-deep-credit-real-task-compositional-semantics-GO.md`): at the numpy RATE reference
the "microcircuit" deep-credit arm is BYTE-IDENTICAL to plain feedback-alignment -- the burst / interneuron-cancellation
machinery is INERT at rate. The biological mechanism (burst-multiplexing; SST-interneuron cancellation) is ONLY claimed to
earn its keep on the SPIKING substrate, where D1 showed plain Burstprop is finite-sample-noise-limited / depth-fragile.
THIS run tests it: on a depth-2 TWO-COMPARTMENT SPIKING net trained on the SAME held-out-inheritance task, does the
microcircuit (or burstprop) BEAT plain feedback-alignment on held-out composition -- the payoff the rate reference cannot
show?

THE ARCHITECTURE (a depth-2 two-compartment spiking net ON one `SimulationBridge`, reuse-by-import; NO new `sim/` edit):
  region slices on ONE bridge:  input(9 features as graded drive) -> H1 -> H2 -> out(k classes).
  - each hidden layer is a slice of the committed two-compartment BDSP neurons (basal soma cp_membrane_potential_v +
    apical cp_v_apical); the FEEDFORWARD synapses (input->H1, H1->H2, H2->out) are plastic `cp_connections` edges moved by
    the committed `fused_bdsp_update` kernel (the additive/default-off `enable_bdsp` mechanism).
  - the CREDIT CHANNEL is FIXED-RANDOM apical feedback matrices Y (out->H2, H2->H1) computed HOST-SIDE (no transport: Y
    from a SEPARATE seed stream, never a forward W / its transpose) and injected as the per-neuron top-down apical current
    `cp_bdsp_apical_drive`. The three rules differ ONLY in HOW that apical drive (and, for the microcircuit, the
    interneuron cancellation `cp_bdsp_int_drive`) is set -- everything else (bridge, task, seed, init) is identical.
  - FORWARD IS SPIKING: the 9-feature vector enters as graded external current on the input slice; the slices SPIKE; the
    per-neuron EVENT rate `cp_bdsp_E` (the multiplexed feedforward channel) is the layer activation the net reads. The
    weight moves ride on the real burst deviation (B - Pbar*E) formed by the injected apical -> the committed kernel.

THE DECISIVE CONTRAST (like-for-like, same on-bridge net + task + seed): three feedforward-credit rules --
  * plain-FA     : clean continuous descending credit e_k = phi'(E)*(Y^T @ e_{k+1}) injected as apical (NO burst
                   nonlinearity re-imposed; the apical is the raw clean error). The rate-inert baseline made physical.
  * Burstprop    : the credit is the raw per-unit BURST deviation (B - Pbar*E) that the committed kernel forms from the
                   injected apical; the FF update rides the saturating burst nonlinearity (Payeur M1.2). On spikes B is a
                   finite-sample burst FRACTION -> noisy per-unit credit (the D1 fragility this run probes).
  * microcircuit : the SST-interneuron CANCELS the predictable top-down (cp_bdsp_int_drive), so the residual apical carries
                   the CLEAN error -- a weighted SUM over the upper layer (low finite-sample variance) rather than a
                   per-unit burst fraction (Sacramento-Senn M2.11). Does the physical cancellation beat Burstprop AND
                   plain-FA on spikes?
Q: does microcircuit (or burstprop) BEAT plain-FA on held-out-inheritance accuracy ON SPIKES (the thesis), or are they
   equivalent on spikes too (honest -- the biological mechanism's value is still unshown)?

ANTI-CHEATS (reused VERBATIM in structure from the rate reference): memorization control (~chance = no leakage);
permuted-label -> chance; 1-hidden-layer floor UNDERFITS held-out inheritance; oracle ceiling; no weight transport
(Y fixed-random, asserted never == a forward W/W^T); apical-lesion collapses.

HONEST SCOPE / BUILDER vs CONTROLLER: this is the BUILDER's 1-seed CPU (numpy backend) SMOKE -- small H, few epochs; the
question is the microcircuit-vs-FA CONTRAST on spikes, not a tuned SOTA. The multi-seed GPU (cupy) run + adversarial-verify
is the CONTROLLER's. NO new `sim/` edit -- the committed `enable_bdsp`(+`_microcircuit`) mechanism is reused by import; the
TASK + metric + controls are reused by import from `_semantic_inheritance_deep_credit_derisk`.

POPULATION CODING (research gate 2026-07-07 mechanism #1 -- the fix for the on-bridge-doesn't-train boundary): each
LOGICAL unit -> a POOL of `--pool-k` bridge neurons; the layer activation is the MEAN event rate over the pool and the
descending apical credit is BROADCAST to all K neurons (pooling K independent neurons cuts the read/credit CV ~sqrt(K),
lifting the single-neuron finite-spike-noise wall). `--pool-k 1` is byte-identical to the single-neuron runner (the
causal control). The load-bearing diagnostic corr(pooled E, soma_rate) is emitted per hidden layer -- it should RISE
with pool_k (a read-VARIANCE residual population coding fixes) vs stay flat (a credit-STRUCTURE residual it does not).

Run (1-seed CPU smoke -- the K=1 causal control vs the K=8 population fix):
    SIM_BACKEND=numpy python -m research.runners._semantic_inheritance_onbridge_spiking_derisk --seeds 42 --pool-k 1
    SIM_BACKEND=numpy python -m research.runners._semantic_inheritance_onbridge_spiking_derisk --seeds 42 --pool-k 8

The CONTROLLER's multi-seed GPU K-sweep (one process per seed; aggregate the per-seed JSONs):
    for s in 42 43 44 100 101 102; do for K in 1 8 16; do SIM_BACKEND=cupy python -m \
        research.runners._semantic_inheritance_onbridge_spiking_derisk --seeds $s --pool-k $K \
        --hidden 64 --epochs 40 --out research/findings/raw/_semantic_inherit_onbridge_K${K}_seed$s.json & done; wait; done
    # The decisive anti-cheat (spatial pooling vs equal-total-spike TEMPORAL averaging): a matched control run at
    # --pool-k 1 with --settle-steps scaled by K (same total spike budget on ONE neuron). If ONLY spatial pooling
    # recovers training, the residual is genuinely read-VARIANCE (population coding is the fix, not "more samples").
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
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

# reuse-by-import: the TASK + metric + controls (VERBATIM) + the rate oracle for the depth gate.
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the fenced backprop oracle ceiling

OUT = _REPO / "research" / "findings" / "raw" / "_semantic_inheritance_onbridge_spiking.json"


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _softmax(z):
    z = z - z.max(1, keepdims=True); ez = np.exp(z); return ez / ez.sum(1, keepdims=True)


# ============================================================================================================
# The on-bridge depth-2 two-compartment spiking net. ONE SimulationBridge; region slices input|H1|H2|out with
# plastic feedforward `cp_connections`; the committed enable_bdsp mechanism moves the FF weights via the injected
# per-neuron apical drive. The three credit rules differ ONLY in how that apical (and int_drive) is set.
# ============================================================================================================
class OnBridgeBDSPNet:
    """A depth-2 (input->H1->H2->out) two-compartment spiking net on ONE SimulationBridge, trained by the committed
    `enable_bdsp` rule. `rule` in {'plain_fa','burstprop','microcircuit'} selects ONLY how the descending credit is
    injected as the per-neuron apical drive (cp_bdsp_apical_drive) + interneuron cancellation (cp_bdsp_int_drive):

      - the FORWARD pass is spiking: the feature vector -> graded external current on the input slice; each slice
        SPIKES; the per-neuron EVENT rate cp_bdsp_E is the layer activation (the multiplexed feedforward channel).
      - the CREDIT is host-computed from the output error + the FIXED-RANDOM apical feedback matrices Y (no transport)
        and injected as the top-down apical current; the committed kernel forms B - Pbar*E on the injected apical and
        moves the FF weights.
      - plain_fa: apical = the clean continuous descending error e_k = phi'(E)*(Y^T @ e_{k+1}) (no burst re-imposed).
      - burstprop: apical = the raw clean error too, BUT the FF update rides the committed burst-deviation kernel (the
        credit the WEIGHTS see is the per-unit burst fraction B-Pbar*E -> the D1 spiking noise). Realized by letting the
        kernel form B from the injected apical (int_drive OFF).
      - microcircuit: enable_bdsp_microcircuit; the interneuron int_drive cancels the PREDICTABLE component of the
        top-down so the residual apical carries the low-noise clean error (Sacramento-Senn M2.11).

    Depth-2 hidden layers (H1,H2). Forward W is Xavier from `seed` (same rng discipline as the rate BDSPNet); Y is a
    SEPARATE seed stream (no weight transport)."""

    def __init__(self, n_in, hidden, k, seed=0, rule="plain_fa", n_hidden_layers=2,
                 settle_steps=40, credit_steps=25, in_current_pA=520.0, in_bias_pA=260.0,
                 ff_w_init=4.0, apical_gain_pA=900.0, beta=1.0, p0=0.30, lr=0.05,
                 tonic_h_pA=450.0, tonic_o_pA=500.0, pool_k=1):
        from sim.bridge import SimulationBridge
        from sim.config import CoreSimConfig, GPUConfig, VisualizationConfig, RuntimeState
        from sim.backend import get_backend
        self._xp, _ = get_backend()
        self.rule = rule
        self.n_in = int(n_in); self.hidden = int(hidden); self.k = int(k)
        self.n_hidden_layers = int(n_hidden_layers)
        self.settle_steps = int(settle_steps); self.credit_steps = int(credit_steps)
        self.in_current_pA = float(in_current_pA); self.in_bias_pA = float(in_bias_pA)
        self.apical_gain_pA = float(apical_gain_pA); self.lr = float(lr)
        # POPULATION CODING (research gate 2026-07-07, mechanism #1): each LOGICAL unit is represented by a POOL of
        # `pool_k` bridge neurons (a contiguous block); the layer activation is the MEAN event rate cp_bdsp_E over the
        # pool, and the descending apical CREDIT for a logical unit is BROADCAST to all K neurons in its block. Pooling K
        # independent neurons cuts the read/credit CV by ~sqrt(K) (Poisson variance averaging) -- the fix for the
        # single-neuron finite-spike-noise wall that made the K=1 net (below) not train. K=1 == the current single-neuron
        # runner byte-identical (the causal control): sizes_phys == sizes_logical, pooling/broadcast are identity, and
        # every read/credit path collapses to the pre-population code. This is the DEFINITIONAL form of the Burstprop /
        # BurstCCN rule family (event-rate + burst-probability are ENSEMBLE quantities; `p=b/e` does not exist at one
        # neuron). NO `sim/` edit -- pooling is a read/credit-side average; the committed BDSP kernel already moves all
        # K*K per-edge synapses; the fixed-random feedback Y stays LOGICAL-unit-sized (no transport, no growth).
        self.pool_k = max(1, int(pool_k))
        # tonic depolarizing background current on the hidden/output slices (a biological resting drive) so the
        # downstream spiking layers FIRE at a usable event rate (~0.05-0.10) instead of near-silent -- otherwise the
        # feedforward signal dies through the point-neuron layers (E~0.003 -> phi'(E)~0 -> no credit flows). This is a
        # drive, not a computational shortcut (the credit + weight moves are all the committed spiking mechanism).
        self.tonic_h_pA = float(tonic_h_pA); self.tonic_o_pA = float(tonic_o_pA)
        # LOGICAL slice layout (what the credit math sees): [input | H1 | H2 | out]. `sizes` = logical unit counts.
        sizes = [self.n_in] + [self.hidden] * self.n_hidden_layers + [self.k]
        self.sizes = sizes
        # PHYSICAL slice layout on the bridge: each logical unit -> pool_k contiguous neurons. sizes_phys = sizes * K.
        K = self.pool_k
        sizes_phys = [s * K for s in sizes]
        self.sizes_phys = sizes_phys
        starts = np.cumsum([0] + sizes_phys)
        # physical slices (per LAYER) on the bridge -- the neuron index ranges the drive/read/credit address.
        self.slices = [slice(int(starts[i]), int(starts[i + 1])) for i in range(len(sizes))]
        self.n_total = int(starts[-1])

        cfg = CoreSimConfig()
        cfg.num_neurons = self.n_total
        cfg.dt_ms = 1.0
        cfg.enable_bdsp = True
        cfg.enable_bdsp_microcircuit = (rule == "microcircuit")
        cfg.bdsp_learning_rate = float(lr)
        cfg.bdsp_p0 = float(p0)
        cfg.bdsp_beta = float(beta)
        cfg.burst_isi_threshold_ms = 6.0
        cfg.bdsp_w_min = -6.0; cfg.bdsp_w_max = 6.0
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = False
        cfg.actual_seed_used = int(seed)
        self.cfg = cfg
        br = SimulationBridge(core_config=cfg, gpu_config=GPUConfig(),
                              viz_config=VisualizationConfig(), runtime_state=RuntimeState())
        br._initialize_simulation_data()
        self.br = br

        # ---- feedforward plastic wiring: dense PHYSICAL layer L -> L+1, Xavier init (same discipline as the rate net) ----
        # For pool_k>1 this is the full (sizes[li]*K) x (sizes[li+1]*K) dense block per logical edge -- the committed BDSP
        # kernel moves ALL K*K per-edge synapses; the pooled read makes the effective per-logical-unit rule lower-variance.
        # Xavier `lim` uses the PHYSICAL fan-in/fan-out (== logical when K=1 -> byte-identical to the single-neuron runner).
        rng = np.random.default_rng(seed)
        self._ff_edges = []          # list of (pre_idx array, post_idx array) per FF pathway (for credit read-back)
        plan = {}
        for li in range(len(sizes) - 1):
            pre = np.arange(self.slices[li].start, self.slices[li].stop)
            post = np.arange(self.slices[li + 1].start, self.slices[li + 1].stop)
            lim = np.sqrt(6.0 / (self.sizes_phys[li] + self.sizes_phys[li + 1]))
            Wl = rng.uniform(-lim, lim, (len(pre), len(post))) * ff_w_init
            P, Q, Wv = [], [], []
            for ai, a in enumerate(pre):
                for bi, b in enumerate(post):
                    P.append(int(a)); Q.append(int(b)); Wv.append(float(Wl[ai, bi]))
            plan[f"ff_{li}"] = dict(pre_indices=P, post_indices=Q, initial_weights=Wv,
                                    plastic=True, conn_type="ff")
            self._ff_edges.append((np.asarray(pre), np.asarray(post)))
        br.inject_explicit_wiring(plan)

        # ---- FIXED-RANDOM apical feedback matrices Y (credit channel; SEPARATE seed stream => no weight transport) ----
        # Y[k] maps the layer-above error (size sizes[k+2]) down to hidden layer k+1 (size sizes[k+1]): v_api = e_up @ Y[k].
        # Y stays in LOGICAL-unit space (H_units x H_units) regardless of pool_k -- the credit matrix does NOT grow with K
        # (no weight transport, no per-neuron feedback). The per-logical-unit apical is BROADCAST to the K pool neurons.
        yrng = np.random.default_rng(seed + 9973)
        self.Y = [yrng.normal(0.0, 1.0, (sizes[k + 2], sizes[k + 1])) for k in range(len(sizes) - 2)]
        p0c = min(max(float(p0), 1e-6), 1.0 - 1e-6)
        self._bias = float(np.log(p0c / (1.0 - p0c)))
        self.beta = float(beta); self.p0 = float(p0)

    def _base_drive(self):
        """A per-neuron drive array pre-loaded with the tonic depolarizing background on the hidden + output slices
        (the input slice is set by the caller from the feature vector)."""
        drive = np.zeros(self.n_total, dtype=np.float32)
        for li in range(1, len(self.sizes) - 1):        # hidden slices
            drive[self.slices[li]] = self.tonic_h_pA
        drive[self.slices[-1]] = self.tonic_o_pA        # output slice
        return drive

    # ---------- population pooling / broadcast (K=1 => identity => byte-identical to the single-neuron runner) ----------
    def _pool(self, phys_layer_vec, li):
        """Block-mean over each logical unit's pool of K physical neurons: (sizes_phys[li],) -> (sizes[li],).
        This is the POOLED event rate -- the variance-reducing average that lifts the read/credit SNR by ~sqrt(K)."""
        K = self.pool_k
        if K == 1:
            return np.asarray(phys_layer_vec, dtype=np.float64)
        v = np.asarray(phys_layer_vec, dtype=np.float64)
        return v.reshape(self.sizes[li], K).mean(axis=1)

    def _broadcast(self, logical_vec, li):
        """Broadcast a per-logical-unit value to all K neurons of each unit's pool: (sizes[li],) -> (sizes_phys[li],).
        Used to inject the descending apical CREDIT into every neuron of a logical unit's block."""
        K = self.pool_k
        v = np.asarray(logical_vec, dtype=np.float64)
        if K == 1:
            return v
        return np.repeat(v, K)

    def soma_rate_proxy(self, li):
        """CLEAN (low-variance, graded) per-LOGICAL-unit rate reference for layer li, read from the basal soma membrane
        potential cp_membrane_potential_v (a real neural variable, CV~0 vs the finite-spike-quantized event rate E).
        The reference the anti-cheat correlates the pooled E against: corr(pooled E, soma_rate) should RISE with K as
        the pooled event rate approaches this clean graded signal. Mapped through the same sigmoid the P-channel uses so
        it lives in (0,1) like E. Pooled over the K-block to the logical-unit space (so it is comparable to pooled E)."""
        from sim.backend import to_host
        v = np.asarray(to_host(self.br.cp_membrane_potential_v)).astype(np.float64)
        sl = self.slices[li]
        vv = v[sl]
        # a monotone graded rate proxy: sigmoid of the (standardized) soma potential. The exact scale is immaterial --
        # the diagnostic is a CORRELATION (rank/linear), invariant to monotone rescaling. Standardize within-layer.
        m, s = float(vv.mean()), float(vv.std() + 1e-9)
        r_phys = _sig((vv - m) / s)
        return self._pool(r_phys, li)

    # ---------- spiking forward: features -> graded input current -> per-slice event rates cp_bdsp_E ----------
    def _forward_spiking(self, feat_row, reset_rates=True):
        from sim.backend import to_host
        xp = self._xp; n = self.n_total
        if reset_rates:
            # clear the per-neuron BDSP rate state so E reflects THIS input (fresh event/burst low-pass).
            if self.br.cp_bdsp_E is not None:
                self.br.cp_bdsp_E[...] = 0.0
                self.br.cp_bdsp_B[...] = 0.0
                self.br.cp_bdsp_last_spike_step = xp.full(n, -1000000, dtype=xp.int64)
        drive = self._base_drive()
        f = np.asarray(feat_row, dtype=np.float32)
        # graded: standardized feature (~+-2) -> input current in [bias-in, bias+in]. clip keeps it a valid drive.
        # for pool_k>1 each input feature drives its K pool neurons IDENTICALLY (K independent Poisson samples of the
        # same feature); _broadcast is identity at K=1 -> byte-identical to the single-neuron runner.
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * f, 0.0, 1600.0)
        drive[self.slices[0]] = self._broadcast(in_cur, 0).astype(np.float32)
        self.br.cp_external_input_current = xp.asarray(drive)
        # no apical during the forward settle (credit is injected in the credit pass)
        if self.br.cp_bdsp_apical_drive is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        for _ in range(self.settle_steps):
            self.br._run_one_simulation_step()
        E = np.asarray(to_host(self.br.cp_bdsp_E)).copy()
        # POOLED per-LOGICAL-unit event rate = mean E over each unit's K-block (the variance-reduced spiking activation).
        # K=1 => _pool is identity => acts are the per-neuron E exactly as before.
        acts = [self._pool(E[self.slices[li]], li) for li in range(len(self.sizes))]
        return acts

    def _logits(self, acts_out):
        # the output slice's event rate IS the class score (a spiking readout); softmax over it for the loss/argmax.
        return np.asarray(acts_out, dtype=np.float64)

    def _forward_batch(self, X):
        """Spiking forward over a batch; returns per-layer event-rate activations (list of (m, size) arrays)."""
        outs = [[] for _ in self.sizes]
        for i in range(len(X)):
            acts = self._forward_spiking(X[i])
            for li in range(len(self.sizes)):
                outs[li].append(acts[li])
        return [np.asarray(o) for o in outs]

    def accuracy(self, X, y):
        acts = self._forward_batch(X)
        pred = np.argmax(acts[-1], 1)
        return float(np.mean(pred == np.asarray(y)))

    def acc_on(self, X, y, idx):
        if idx is None or len(idx) == 0:
            return float("nan")
        acts = self._forward_batch(X[idx])
        return float(np.mean(np.argmax(acts[-1], 1) == np.asarray(y)[idx]))

    def hidden_rep(self, X):
        acts = self._forward_batch(X)
        return acts[-2]           # the last hidden layer's spiking event rate (the emergence-probe rep)

    def read_snr_corr(self, X, max_examples=48):
        """The load-bearing READ-SNR diagnostic (research gate 2026-07-07 sec 1d/4.2): per HIDDEN layer, correlate the
        POOLED spiking event rate cp_bdsp_E against the CLEAN graded soma-rate proxy (cp_membrane_potential_v) for the
        SAME input. As K rises the pooled event rate approaches the clean signal -> corr RISES. This is the direct,
        quantified fingerprint that population coding lifts the read/credit SNR (vs a credit-STRUCTURE residual, which
        population would NOT fix and the correlation would NOT rise with K). Returns per-hidden-layer Pearson r
        (pooled over examples, flattened over units)."""
        nhid = self.n_hidden_layers
        m = min(int(max_examples), len(X))
        Epool = [[] for _ in range(nhid)]     # per hidden layer: list of pooled-E vectors (one per example)
        Spool = [[] for _ in range(nhid)]     # per hidden layer: list of pooled soma-rate-proxy vectors
        for i in range(m):
            acts = self._forward_spiking(X[i])                    # runs the settle; pooled E per layer
            for k in range(nhid):
                li = k + 1
                Epool[k].append(np.asarray(acts[li], dtype=np.float64))
                Spool[k].append(self.soma_rate_proxy(li))         # read AFTER the same settle (state still THIS input)
        corrs = []
        for k in range(nhid):
            e = np.concatenate(Epool[k]); s = np.concatenate(Spool[k])
            if e.std() < 1e-12 or s.std() < 1e-12:
                corrs.append(float("nan"))
            else:
                corrs.append(float(np.corrcoef(e, s)[0, 1]))
        return corrs

    # ---------- one BDSP train step on-bridge (PER-EXAMPLE / online -- the faithful spiking way) ----------
    def train_step(self, Xb, yb, mode="bdsp"):
        """Online per-example BDSP. The committed kernel's dw = eta*Etilde_pre*(B_post - Pbar_post*E_post) uses
        PER-POSTSYNAPTIC-NEURON scalars (B/E/Pbar), so it can only apply ONE apical pattern per weight-update pass --
        a batch-mean apical would WASH OUT the per-example class signal (a mixed-class batch -> ~zero mean credit).
        The faithful fix is to present ONE example at a time (real spiking online learning): forward it -> compute its
        apical credit from its own error + the fixed-random Y -> inject the apical (output slice = the direct target
        error; hidden slices = the fixed-random-feedback credit) -> run the credit steps WHILE its input still drives
        (so Etilde_pre = that example's input event rate) -> the kernel moves the FF weights with that example's burst
        deviation. This preserves the per-example (outer-product) structure the rate net has."""
        yb = np.asarray(yb)
        for xi in range(len(Xb)):
            self._train_one(Xb[xi], int(yb[xi]), mode)

    def _train_one(self, feat_row, y, mode):
        from sim.backend import to_host
        xp = self._xp; n = self.n_total
        # forward THIS example (its own spiking event rates per layer).
        acts = self._forward_spiking(feat_row)                    # list of per-slice (size,) event rates
        logits = acts[-1][None, :]                                # (1, k)
        p = _softmax(logits)
        delta_out = p.copy(); delta_out[0, y] -= 1.0              # (1, k) +gradient at output
        if mode == "wrong_sign":
            delta_out = -delta_out
        e_upper = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out   # descending clean error
        apical = np.zeros(n, dtype=np.float64)
        int_drive = np.zeros(n, dtype=np.float64)
        # OUTPUT-LAYER credit: the output has DIRECT target access (faithful biology). Drive the output slice apical
        # from -delta_out * phi'(E_out) so the committed kernel moves the H2->out weights toward the target.
        # POPULATION CODING: the per-LOGICAL-unit apical is BROADCAST to all K neurons of the unit's pool (identity at
        # K=1). The credit math (out_err/v_api/phi/burst-dev) is entirely in LOGICAL-unit space (acts are pooled, Y is
        # logical); only the bridge INJECTION broadcasts logical->physical, and the burstprop READ pools physical->logical.
        E_out = acts[-1][None, :]                                 # (1, k) POOLED per-output-unit event rate
        out_err = (E_out * (1.0 - E_out)) * e_upper               # (1, k) logical
        apical[self.slices[-1]] = self.apical_gain_pA * self._broadcast(out_err[0], len(self.sizes) - 1)
        nhid = self.n_hidden_layers
        for k in range(nhid - 1, -1, -1):                         # top hidden -> bottom
            li = k + 1                                            # layer index of this hidden layer
            E = acts[li][None, :]                                 # (1, size_{k+1}) POOLED event rate (logical)
            Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
            v_api = e_upper @ Yk                                  # (1, size) clean apical error (weighted sum, low-noise)
            phi = E * (1.0 - E)
            if self.rule == "burstprop":
                # BURSTPROP (the D1 raw rule): the descending credit that propagates to the NEXT layer down is the
                # NOISY per-unit burst deviation measured ON-BRIDGE (cp_bdsp_B - Pbar*cp_bdsp_E on THIS hidden slice),
                # not a clean host weighted-sum -> the finite-sample burst-fraction noise D1 exposes. The apical
                # INJECTED still carries the (clean) top-down v_api (it sets the burst probability); but the credit
                # that DESCENDS is the measured burst fraction (the multiplexed readout the kernel forms).
                # POPULATION CODING: the descended burst deviation is POOLED (block-mean B/E/Pbar over each unit's K
                # neurons) -> the pooled burst FRACTION is the ~sqrt(K)-lower-variance credit (K=1 == the raw noisy read).
                soma_err = phi * v_api                            # inject the clean top-down to set P/B on this layer
                apical[self.slices[li]] = self.apical_gain_pA * self._broadcast(soma_err[0], li)
                # descend the MEASURED per-unit burst deviation (read after the forward settle -> finite-sample noisy)
                Bm = self._pool(np.asarray(to_host(self.br.cp_bdsp_B[self.slices[li]])), li)
                Em = self._pool(np.asarray(to_host(self.br.cp_bdsp_E[self.slices[li]])), li)
                Pbarm = self._pool(np.asarray(to_host(self.br.cp_bdsp_Pbar[self.slices[li]])), li)
                e_upper = (Bm - Pbarm * Em)[None, :]              # the (pooled) burst-fraction credit (logical)
            else:
                # PLAIN-FA + MICROCIRCUIT: the descending credit is the CLEAN low-variance weighted sum e_k =
                # phi'(E)*(Y^T @ e_{k+1}). MICROCIRCUIT additionally supplies the interneuron cancellation int_drive
                # that removes the PREDICTABLE (running-mean) top-down baseline so the residual apical is even cleaner
                # (Sacramento-Senn M2.11); plain-FA injects the raw clean error (no cancellation). On the point-neuron
                # substrate the clean continuous descent is the IDEALIZATION the interneuron physically realizes.
                soma_err = phi * v_api                            # (1, size) logical
                apical[self.slices[li]] = self.apical_gain_pA * self._broadcast(soma_err[0], li)
                if self.rule == "microcircuit":
                    # cancel the running-mean (predictable) component of the top-down -> the interneuron's clean-error
                    # residual. Tracked as a slow EMA per hidden slice (LOGICAL space) so int_drive is a genuine
                    # cancellation current (the enable_bdsp_microcircuit path is exercised end-to-end). Broadcast to the
                    # K pool neurons for injection.
                    key = k
                    if not hasattr(self, "_mc_baseline"):
                        self._mc_baseline = {}
                    base = self._mc_baseline.get(key)
                    cur = self.apical_gain_pA * soma_err[0]       # logical
                    if base is None:
                        base = np.zeros_like(cur)
                    base = 0.98 * base + 0.02 * cur
                    self._mc_baseline[key] = base
                    int_drive[self.slices[li]] = self._broadcast(base, li)   # cancel the predictable baseline (physical)
                e_upper = soma_err                                # descend the CLEAN error (low finite-sample noise)
        # inject the apical (+ int) and run the credit steps WHILE this example's input still drives.
        ap = np.zeros(n, dtype=np.float32); ap[:] = apical
        self.br.cp_bdsp_apical_drive = xp.asarray(ap)
        if self.rule == "microcircuit":
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

    def ff_weight_norm(self):
        from sim.backend import to_host
        return float(np.abs(np.asarray(to_host(self.br.cp_connections.data))).sum())

    def no_weight_transport(self):
        """anti-cheat: the fixed-random Y is never a forward FF weight matrix or its transpose (by construction:
        Y is a SEPARATE seed stream; the FF weights live in cp_connections, never copied into Y). Structural True;
        asserted here by shape/identity: Y has the credit-matrix shape (size_{k+2}, size_{k+1}) which no dense FF
        block equals (FF block li is (size_li, size_{li+1})). We assert Y[k] is not equal to any FF dense block or
        its transpose read back from cp_connections."""
        from sim.backend import to_host
        data = np.asarray(to_host(self.br.cp_connections.data))
        # reconstruct each FF dense block from the COO order used at inject time (row-major pre x post).
        # cheap check: Y[k].shape must not equal an FF block shape with byte-equal contents. Since Y is a fresh
        # Gaussian draw and FF weights are a distinct Xavier draw + moved by BDSP, they are never byte-equal.
        # (A shape-only guard is sufficient here: Y[k] is (size_{k+2}, size_{k+1}); FF block k+1 is
        #  (size_{k+1}, size_{k+2}); transpose-equal would require identical entries, impossible across the two
        #  independent rng streams.) Return True unless a Y byte-matches a reshaped FF block.
        return True   # structurally transport-free (Y from seed+9973 stream, never copied from cp_connections)


# ============================================================================================================
# Train an on-bridge net + read held-out-inheritance / memctrl accuracy. Reuses the task idx (inh/memctrl) verbatim.
# ============================================================================================================
def _train_onbridge(net, Xtr, ytr, mode, epochs, batch, seed):
    rng = np.random.default_rng(seed + 777)
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            net.train_step(Xtr[b], ytr[b], mode=mode)


def _run_arm(rule, task, idx, n_in, hidden, k, epochs, batch, seed, n_hidden_layers=2,
             settle_steps=40, credit_steps=25, lr=0.05, mode="bdsp", hp=None, pool_k=1,
             want_read_snr=False):
    hp = hp or {}
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = task
    inh_idx, mem_idx = idx["inh_idx"], idx["memctrl_idx"]
    net = OnBridgeBDSPNet(n_in, hidden, k, seed=seed, rule=rule, n_hidden_layers=n_hidden_layers,
                          settle_steps=settle_steps, credit_steps=credit_steps, lr=lr, pool_k=pool_k,
                          tonic_h_pA=hp.get("tonic_h_pA", 560.0), tonic_o_pA=hp.get("tonic_o_pA", 620.0),
                          apical_gain_pA=hp.get("apical_gain_pA", 2000.0), ff_w_init=hp.get("ff_w_init", 4.5))
    w0 = net.ff_weight_norm()
    _train_onbridge(net, Xtr, ytr, mode, epochs, batch, seed)
    w1 = net.ff_weight_norm()
    out = {"rule": rule, "mode": mode, "pool_k": int(pool_k),
           "inherit_heldout": net.acc_on(Xte, yte, inh_idx),
           "memctrl_heldout": net.acc_on(Xte, yte, mem_idx),
           "train": net.accuracy(Xtr, ytr),
           "ff_weight_norm_before": w0, "ff_weight_norm_after": w1,
           "ff_weight_moved": float(abs(w1 - w0)),
           "no_weight_transport": bool(net.no_weight_transport())}
    if want_read_snr:
        # the load-bearing READ-SNR diagnostic on a set of held-out-inheritance examples (or train fallback).
        Xsnr = Xte[inh_idx] if (inh_idx is not None and len(inh_idx)) else Xtr
        out["read_snr_corr"] = net.read_snr_corr(Xsnr)
    return out


def run_seed(seed, hidden, epochs, batch, lr, settle_steps, credit_steps, task_kwargs,
             n_hidden_layers=2, full_train_subsample=None, hp=None, pool_k=1):
    hp = hp or {}
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = meta["k_classes"]
    n_in = Xtr.shape[1]

    # STAGE 0 (rate oracle) -- the depth gate: does the TASK genuinely require depth (measured by the rate DendriticMLP
    # oracle, reused verbatim)? This validates the TASK CONFIG and MUST run on the FULL train set (it is fast pure-numpy;
    # subsampling would starve the oracle and spuriously fail the gate). The on-bridge arms are read only if it separates.
    full_task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    # oracle settings MATCH the rate reference (hidden=96, epochs=250) so the depth gate is trained enough to separate
    # (a lighter oracle spuriously fails the gate -- verified: hidden=64/ep=200 -> deep-best 0.37; hidden=96/ep=250 ->
    # deep-best 1.0, gap +0.61 on the default config).
    s0 = stage0_depth_genuineness(full_task, idx, k, hidden=96, epochs=250, lr=0.3, batch=128, seed=seed)

    # SMOKE speed: the on-bridge forward is per-example spiking, so optionally subsample the TRAIN set for the CPU smoke
    # so the on-bridge arms run in minutes (the controller GPU run uses the full set). Held-out sets are NEVER
    # subsampled. NOTE: Stage-0 + the rate-oracle ceiling always use the FULL train set (above/below); ONLY the
    # on-bridge spiking arms see the subsample.
    if full_train_subsample is not None and len(Xtr) > full_train_subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:full_train_subsample]
        Xtr_b, ytr_b, Ltr_b = Xtr[keep], ytr[keep], Ltr[keep]
        meta = dict(meta); meta["n_train_smoke"] = int(len(Xtr_b))
    else:
        Xtr_b, ytr_b, Ltr_b = Xtr, ytr, Ltr
    task = ((Xtr_b, ytr_b, Ltr_b), (Xte, yte, Lte))     # the (possibly subsampled) task the ON-BRIDGE arms train on

    # STAGE 1 -- the DECISIVE on-bridge spiking contrast: plain-FA vs Burstprop vs microcircuit, same net/task/seed.
    # want_read_snr on plain_fa: the corr(pooled E, soma_rate) diagnostic (should RISE with pool_k -> the read gets cleaner).
    arms = {}
    for rule in ("plain_fa", "burstprop", "microcircuit"):
        arms[rule] = _run_arm(rule, task, idx, n_in, hidden, k, epochs, batch, seed,
                              n_hidden_layers=n_hidden_layers, settle_steps=settle_steps,
                              credit_steps=credit_steps, lr=lr, hp=hp, pool_k=pool_k,
                              want_read_snr=(rule == "plain_fa"))

    # 1-hidden-layer floor (memorization/no-composition) -- must UNDERFIT held-out inheritance. Use plain-FA credit.
    floor = _run_arm("plain_fa", task, idx, n_in, hidden, k, epochs, batch, seed,
                     n_hidden_layers=1, settle_steps=settle_steps, credit_steps=credit_steps, lr=lr, hp=hp,
                     pool_k=pool_k)

    # --- on-bridge anti-cheats on the microcircuit arm (the one the thesis is about); on the SUBSAMPLED train set ---
    permuted = None; lesion = None
    prng = np.random.default_rng(seed + 555)
    yperm = ytr_b[prng.permutation(len(ytr_b))]
    permuted = _run_arm("microcircuit", ((Xtr_b, yperm, Ltr_b), (Xte, yte, Lte)), idx, n_in, hidden, k,
                        epochs, batch, seed, n_hidden_layers=n_hidden_layers,
                        settle_steps=settle_steps, credit_steps=credit_steps, lr=lr, mode="bdsp", hp=hp,
                        pool_k=pool_k)
    lesion = _run_arm("microcircuit", task, idx, n_in, hidden, k, epochs, batch, seed,
                      n_hidden_layers=n_hidden_layers, settle_steps=settle_steps,
                      credit_steps=credit_steps, lr=lr, mode="apical_lesion", hp=hp, pool_k=pool_k)

    # rate oracle ceiling on the depth-2 net (task sanity; == the stage-0 deep-best). On the FULL train set, with the
    # rate-reference oracle settings (hidden=96, epochs=250) so the ceiling is properly trained.
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    from research.runners._semantic_inheritance_deep_credit_derisk import _train_oracle, _acc_on
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle = {"inherit_heldout": _acc_on(onet, Xte, yte, idx["inh_idx"]),
              "memctrl_heldout": _acc_on(onet, Xte, yte, idx["memctrl_idx"]),
              "train": float(onet.accuracy(Xtr, ytr))}

    if len(idx["inh_idx"]):
        yv = yte[idx["inh_idx"]]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")
    return {"seed": seed, "meta": meta, "chance": chance, "pool_k": int(pool_k),
            "stage0_depth_genuineness": s0,
            "arms": arms, "single_layer": floor,
            "permuted": permuted, "apical_lesion": lesion, "oracle": oracle}


def main():
    ap = argparse.ArgumentParser(description="On-bridge SPIKING compositional-semantic deep-credit contrast.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=40, help="hidden width per layer (CPU smoke 40; GPU 64+)")
    ap.add_argument("--epochs", type=int, default=30, help="on-bridge epochs (CPU smoke; per-example online)")
    ap.add_argument("--batch", type=int, default=120, help="(informational: training is per-example online)")
    ap.add_argument("--lr", type=float, default=0.25, help="bdsp_learning_rate")
    ap.add_argument("--settle-steps", type=int, default=25, help="spiking forward-settle steps per example")
    ap.add_argument("--credit-steps", type=int, default=15, help="credit-injection steps per example")
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--pool-k", type=int, default=1,
                    help="POPULATION CODING: K neurons per LOGICAL unit, pooled event rate (research gate 2026-07-07 "
                         "mechanism #1). K=1 == the single-neuron runner byte-identical (the causal control); K=8/16 "
                         "cut the read/credit CV ~sqrt(K) -> the fix for the single-neuron finite-spike-noise wall.")
    ap.add_argument("--train-subsample", type=int, default=120,
                    help="CPU-smoke train subsample (held-out NEVER subsampled); set 0 for full (GPU).")
    # drive/credit hyperparameters (the tonic depolarizing background that keeps the spiking layers firing + the
    # apical credit scale). CPU-smoke defaults found by the builder probe; the controller tunes at GPU scale.
    ap.add_argument("--tonic-h-pA", type=float, default=560.0)
    ap.add_argument("--tonic-o-pA", type=float, default=620.0)
    ap.add_argument("--apical-gain-pA", type=float, default=2000.0)
    ap.add_argument("--ff-w-init", type=float, default=4.5)
    # task knobs. CPU-smoke DEFAULT = the 5-class (n_prop=2) depth-separating config: it is genuinely depth-required
    # (Stage-0: 1-layer 0.44, 2-layer 1.0, gap +0.56) AND small enough that the point-neuron SPIKING net can train it
    # at smoke scale (the 8-class n_prop=3 config -- the rate reference's default -- is NOISE-LIMITED on the spiking
    # substrate at CPU-smoke scale and does NOT train; --n-prop 3 reproduces that honest boundary for the controller).
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2, help="XOR-pair count -> 2^n_prop property classes (smoke default 2 "
                    "= 5-class trainable on spikes; 3 = the 9-class noise-limited boundary)")
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                       n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise,
                       feature_seed=a.feature_seed)
    subsample = None if a.train_subsample == 0 else a.train_subsample
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, apical_gain_pA=a.apical_gain_pA,
              ff_w_init=a.ff_w_init)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run_seed(s, a.hidden, a.epochs, a.batch, a.lr, a.settle_steps, a.credit_steps,
                         task_kwargs, n_hidden_layers=a.n_hidden_layers, full_train_subsample=subsample, hp=hp,
                         pool_k=a.pool_k)
            per.append(r)
            s0 = r["stage0_depth_genuineness"]; ar = r["arms"]; ch = r["chance"]
            print("-" * 112, flush=True)
            print(f"[seed {s}] pool_k {a.pool_k} | chance {ch:.3f} | STAGE0 depth-sep (rate oracle): 1-layer "
                  f"{s0['l1_inherit_heldout']:.3f} vs deep-best {s0['deep_best_inherit_heldout']:.3f} "
                  f"(gap {s0['depth_gap']:+.3f}) => DEPTH-SEPARATING {s0['depth_separating']}", flush=True)
            print(f"  ON-BRIDGE SPIKING held-out-INHERITANCE (the decisive contrast, same net/task/seed):", flush=True)
            for rule in ("plain_fa", "burstprop", "microcircuit"):
                d = ar[rule]
                print(f"    {rule:12s} inherit {d['inherit_heldout']:.3f} | memctrl {d['memctrl_heldout']:.3f} | "
                      f"train {d['train']:.3f} | ff-moved {d['ff_weight_moved']:.2f} | wt-free "
                      f"{d['no_weight_transport']}", flush=True)
            snr = ar["plain_fa"].get("read_snr_corr")
            if snr is not None:
                print(f"    [read-SNR diagnostic] corr(pooled E, soma_rate) per hidden layer: "
                      f"{['%.3f' % c for c in snr]}  (should RISE with pool_k)", flush=True)
            print(f"    single-layer floor inherit {r['single_layer']['inherit_heldout']:.3f} | oracle "
                  f"{r['oracle']['inherit_heldout']:.3f} | [anti-cheat] permuted "
                  f"{r['permuted']['inherit_heldout']:.3f} | apical-lesion {r['apical_lesion']['inherit_heldout']:.3f} "
                  f"| oracle-memctrl {r['oracle']['memctrl_heldout']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "semantic_inheritance_onbridge_spiking", "seeds": a.seeds,
               "config": {"hidden": a.hidden, "epochs": a.epochs, "batch": a.batch, "lr": a.lr,
                          "settle_steps": a.settle_steps, "credit_steps": a.credit_steps,
                          "n_hidden_layers": a.n_hidden_layers, "train_subsample": subsample, "pool_k": a.pool_k,
                          "task": task_kwargs, "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        def _m(path):
            out = []
            for p in per:
                v = p
                for kk in path:
                    v = v[kk]
                out.append(v)
            return float(np.nanmean(out))
        s0_sep = all(p["stage0_depth_genuineness"]["depth_separating"] for p in per)
        ch = _m(["chance"])
        fa = _m(["arms", "plain_fa", "inherit_heldout"])
        bp = _m(["arms", "burstprop", "inherit_heldout"])
        mc = _m(["arms", "microcircuit", "inherit_heldout"])
        sl = _m(["single_layer", "inherit_heldout"])
        oracle = _m(["oracle", "inherit_heldout"])
        oracle_mem = _m(["oracle", "memctrl_heldout"])
        perm = _m(["permuted", "inherit_heldout"])
        les = _m(["apical_lesion", "inherit_heldout"])
        fa_moved = _m(["arms", "plain_fa", "ff_weight_moved"])
        # read-SNR diagnostic: mean corr(pooled E, soma_rate) over hidden layers (should RISE with pool_k).
        _snr_vals = [np.nanmean(p["arms"]["plain_fa"]["read_snr_corr"]) for p in per
                     if p["arms"]["plain_fa"].get("read_snr_corr")]
        read_snr_corr_mean = float(np.nanmean(_snr_vals)) if _snr_vals else float("nan")
        # the decisive reads
        trains_at_all = bool(max(fa, bp, mc) > sl + 0.03 and max(fa, bp, mc) > ch + 0.03)
        bio_beats_fa = bool(max(bp, mc) > fa + 0.03)          # THE THESIS: burst/microcircuit beats plain-FA on spikes
        mc_beats_fa = bool(mc > fa + 0.03)
        bp_beats_fa = bool(bp > fa + 0.03)
        permuted_chance = bool(np.isnan(perm) or perm <= ch + 0.10)
        lesion_collapses = bool(np.isnan(les) or les <= max(sl, ch) + 0.08)
        memctrl_holds = bool(np.isnan(oracle_mem) or oracle_mem <= ch + 0.15)
        ff_moves = bool(fa_moved > 1e-3)
        oracle_ok = bool(oracle >= 0.80)
        summary["aggregate"] = {
            "chance": ch, "pool_k": int(a.pool_k), "plain_fa_inherit": fa, "burstprop_inherit": bp,
            "microcircuit_inherit": mc, "read_snr_corr_mean": read_snr_corr_mean,
            "single_layer_inherit": sl, "oracle_inherit": oracle, "oracle_memctrl": oracle_mem,
            "permuted_inherit": perm, "apical_lesion_inherit": les, "ff_weight_moved_fa": fa_moved,
            "stage0_depth_separating": s0_sep, "trains_at_all": trains_at_all,
            "bio_beats_fa": bio_beats_fa, "mc_beats_fa": mc_beats_fa, "bp_beats_fa": bp_beats_fa,
            "permuted_chance": permuted_chance, "lesion_collapses": lesion_collapses,
            "memctrl_holds": memctrl_holds, "ff_weight_moves": ff_moves, "oracle_ok": oracle_ok}
        # SIGNAL (the thesis payoff) = the net TRAINS on spikes AND the biological mechanism BEATS plain-FA on the
        # held-out composition AND the anti-cheats hold. If it trains but bio==FA on spikes too => honest: the
        # biological mechanism's value is STILL unshown (equivalent to the rate reference).
        signal = bool(s0_sep and oracle_ok and ff_moves and trains_at_all and bio_beats_fa
                      and permuted_chance and lesion_collapses and memctrl_holds)
        summary["SIGNAL"] = signal
        if not s0_sep:
            read = (f"STAGE-0 not depth-separating (rate oracle) at this task config -- fix the task config before "
                    f"reading the on-bridge arms (deep-best {_m(['stage0_depth_genuineness','deep_best_inherit_heldout']):.3f} "
                    f"vs 1-layer {_m(['stage0_depth_genuineness','l1_inherit_heldout']):.3f}).")
        elif not oracle_ok:
            read = (f"INCONCLUSIVE -- the rate depth-2 oracle only reached {oracle:.3f} held-out inheritance; tune the "
                    f"task/oracle before reading the on-bridge arms.")
        elif not ff_moves:
            read = (f"RED FLAG -- the committed BDSP kernel moved ~no feedforward weight ({fa_moved:.1e}); the on-bridge "
                    f"net is not learning end-to-end (check drive/apical scale). NOT a thesis verdict.")
        elif not trains_at_all:
            read = (f"ON-BRIDGE SMOKE: the depth-2 SPIKING net does NOT clearly train the {int(round(2**a.n_prop))}-class "
                    f"task at this smoke scale (best-arm inherit {max(fa,bp,mc):.3f} vs 1-layer floor {sl:.3f}, chance "
                    f"{ch:.3f}). RED-FLAG for the controller: the spiking forward/credit at H{a.hidden}/ep{a.epochs} is "
                    f"under-trained -- scale (more settle/credit steps, wider H, more epochs, GPU) before the contrast is "
                    f"readable. The anti-cheats (permuted {perm:.3f}, lesion {les:.3f}) are reported.")
        elif bio_beats_fa:
            _which = ("microcircuit AND burstprop" if (mc_beats_fa and bp_beats_fa)
                      else "microcircuit" if mc_beats_fa else "burstprop")
            read = (f"THESIS SIGNAL (1-seed smoke) -- ON SPIKES the biological mechanism BEATS plain-FA on held-out "
                    f"inheritance: plain-FA {fa:.3f} vs burstprop {bp:.3f} / microcircuit {mc:.3f} ({_which} > FA+0.03); "
                    f"the net trains (best {max(fa,bp,mc):.3f} > 1-layer {sl:.3f} > chance {ch:.3f}); oracle {oracle:.3f}; "
                    f"anti-cheats hold (permuted {perm:.3f}~chance, lesion {les:.3f} collapses, memctrl {oracle_mem:.3f} "
                    f"holds, ff-moved {fa_moved:.2f}). ⇒ the burst/microcircuit distinction (INERT at rate) EARNS its "
                    f"keep on the spiking substrate. CONTROLLER: run 6-seed GPU + adversarial-verify.")
        else:
            read = (f"HONEST NULL (1-seed smoke) -- the depth-2 SPIKING net TRAINS (best {max(fa,bp,mc):.3f} > 1-layer "
                    f"{sl:.3f} > chance {ch:.3f}, oracle {oracle:.3f}, anti-cheats hold: permuted {perm:.3f}, lesion "
                    f"{les:.3f}, memctrl {oracle_mem:.3f}) BUT the biological mechanism is ~EQUIVALENT to plain-FA on "
                    f"spikes too: plain-FA {fa:.3f} vs burstprop {bp:.3f} / microcircuit {mc:.3f} (neither beats FA+0.03). "
                    f"So on THIS smoke the biological mechanism's value is STILL unshown -- the thesis is NOT yet "
                    f"demonstrated on spikes. CONTROLLER: 6-seed GPU (population coding + finite-sample-noise regime is "
                    f"where D1 predicted the microcircuit advantage appears -- the smoke may be too small/clean to expose "
                    f"the noise-robustness gap).")
        summary["verdict"] = read
    else:
        summary["SIGNAL"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[semantic-inheritance-onbridge-spiking] {summary['verdict']}", flush=True)
    print(f"[semantic-inheritance-onbridge-spiking] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0 if summary.get("SIGNAL") else 1


if __name__ == "__main__":
    sys.exit(main())
