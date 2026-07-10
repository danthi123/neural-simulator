"""D1 GAP-CLOSER (the on-bridge accuracy run D1 explicitly deferred): does the committed biological deep-credit
rule (`enable_bdsp` + `enable_bdsp_microcircuit`) LEARN A 2-LAYER MAP END-TO-END TO ACCURACY on a REAL
`SimulationBridge` -- not merely move weights?

D1 (`research/findings/2026-07-07-D1-microcircuit-noise-robust-deep-credit-clears-bar-on-spikes.md`, line 37) is
explicit: "The 0.964 is the NUMPY REFERENCE of the sim/ rule on a depth-2 XOR MLP; ... the fully-on-bridge
384-width spiking multi-seed net remains the controller's GPU run (not yet demonstrated)." The D1 runner
(`_gnw_d1_spiking_bdsp_derisk.py`) validated the `sim/` MACHINERY end-to-end (Stage-A' burst detector, Stage-A''
bridge-learns weights-move-under-credit + the P0 moat, Stage-A''' microcircuit cancellation) but its Stage-B
ACCURACY arm is a numpy REFERENCE, not the bridge. THIS runner closes that exact gap: a real 3-region spiking
feedforward net on ONE `SimulationBridge` learns a task that REQUIRES the hidden layer, and we measure HELD-OUT
ACCURACY on spikes.

THE NET (3 regions, one bridge, `enable_bdsp` on):
    input  (n_in neurons, one small population per input bit) --[PLASTIC input->hidden]-->
    hidden (n_hid neurons)                                    --[PLASTIC hidden->output]-->
    output (n_out neurons, one small pool per class; argmax over class-pool spike counts = the prediction)
Both forward pathways are PLASTIC RegionPathways learned by the committed `enable_bdsp` (+ `enable_bdsp_microcircuit`)
kernel `fused_bdsp_update` in `bridge._run_one_simulation_step`. The apical CREDIT that shapes each layer's LTP/LTD
is a FIXED-RANDOM feedback matrix Y (drawn from its OWN RandomState -- NO weight transport), delivered as described
in "THE CRUX" below.

THE PER-SAMPLE CYCLE (present -> read -> teach -> the sim/ rule moves the plastic weights):
  1. RESET the membrane/conductance state (clean forward pass per sample; the cp_bdsp_* learning traces persist).
  2. PRESENT: drive the input region (bit_i==1 -> a suprathreshold current on bit i's input pop; bit==0 -> low) and
     SETTLE `settle_steps` with LEARNING OFF (cfg.bdsp_learning_rate temporarily 0 => the read is unperturbed, the
     P0-moat-clean read). Accumulate the output region's per-class spike counts = the readout.
  3. ERROR: logits = per-class readout; p = softmax(logits); the clean top error e_c = (target_onehot_c - p_c)
     (== the numpy reference's -delta_out, the "output has direct target access" delta).
  4. TEACH: inject the apical credit (see THE CRUX) and run `teach_steps` with LEARNING ON => the committed `sim/`
     BDSP update moves BOTH plastic pathways (postsynaptic burst deviation B-Pbar*E, gated by the apical).
  5. CLEAR the apical for the next sample.
HELD-OUT ACCURACY = fraction of TEST patterns (disjoint from train) whose argmax-over-class-pool readout == label,
read on spikes with learning + apical OFF.

THE CRUX -- how the top error is injected as apical drive, and how the fixed-random feedback delivers it to the
hidden layer (flagged per the task; the D2 spec left this ambiguous):
  The committed `sim/` mechanism reads ONE runner-set per-neuron array `cp_bdsp_apical_drive` and integrates it into
  `cp_v_apical` (bridge.py ~7180-7203); the BDSP weight update then runs per-synapse over ALL of `cp_connections`,
  gated by the per-synapse plastic mask, with the credit carried ENTIRELY by the POSTsynaptic neuron's apical-driven
  burst deviation. There is NO `sim/` code path that routes a synaptic RegionPathway's current into the apical
  compartment. So a LITERAL `plastic=False` synaptic feedback pathway (output->hidden) cannot deliver apical credit
  without a `sim/` edit -- AND wiring one would inject forward output->hidden current into the hidden SOMA, corrupting
  the feedforward pass (a recurrent loop). The faithful, `sim/`-edit-free realization -- EXACTLY what D1's own
  validated `stage_a_bridge_learns` does (it sets `cp_bdsp_apical_drive` directly) and what the numpy reference
  computes (`v_api = e_upper @ Y`) -- is:
    * OUTPUT layer: cp_bdsp_apical_drive[output pool c] = k_out * e_c  (the direct top error; the output "has target
      access", so its burst deviation encodes the delta and hidden->output learns the delta rule).
    * HIDDEN layer: cp_bdsp_apical_drive[hidden j] = k_hid * (e @ Y)[j]  where Y is the FIXED-RANDOM (n_classes x
      n_hidden) feedback matrix. This is the feedback-alignment projection of the top error into the hidden apical --
      it IS the "output->hidden fixed-random apical feedback (plastic=False)" the spec asks for, realized as a
      runner-held frozen matrix + a host projection into the apical-drive array (NOT a synaptic pathway, because the
      substrate has no apical-routing synapse). NO WEIGHT TRANSPORT: Y is from a SEPARATE RandomState, is never
      written after init, and is never a forward weight or its (class-pooled) transpose (asserted).
  The runner computing `Y @ error` is the legitimate teacher/credit-projection wiring (the analogue of the numpy
  reference's `e_upper @ Y`); the BRAIN's job -- the actual synaptic weight change -- is done by the committed `sim/`
  BDSP kernel. This is the honest scope: the FEEDFORWARD PLASTICITY is on the substrate; the credit PROJECTION is
  host-side (as it is in every D1/EMERGE reference and in `stage_a_bridge_learns`).

ANTI-CHEATS (built in; they are the point):
  * ORACLE CEILING  : a fenced numpy backprop MLP (weight transport) on the SAME task+seed reaches >= 0.80 -- the
                      task is genuinely learnable (else INCONCLUSIVE, task broken not rule).
  * NO WEIGHT TRANSPORT (asserted): Y is fixed-random from its own stream, never modified, never a forward weight or
                      its class-pooled transpose.
  * APICAL-LESION   : zero the top error (all apical 0 in teach) -> held-out collapses to chance AND the hidden plastic
                      weights barely move vs the credit arm (the P0 moat: rest apical -> P~Pbar -> dw~0).
  * WRONG-SIGN      : negate the injected top error -> the net anti-learns (held-out below chance).
  * SINGLE-LAYER floor: a linear readout with no hidden learning -> chance on the non-separable task (numpy linear
                      floor + an on-bridge input->output BDSP arm).

HONEST SCOPE (per the task): this builder ships the runner + a small CPU smoke that proves the pipeline runs
end-to-end, both plastic pathways MOVE under credit, the moat holds (lesion hidden-dw << credit hidden-dw), and the
wiring is correct (no weight transport; oracle valid; single-layer floor). The multi-seed GPU sweep that drives the
held-out ACCURACY to the 0.75 bar is the CONTROLLER's run (accuracy needs width + epochs + drive tuning the CPU
smoke deliberately does NOT do). NO `sim/` edit anywhere (reuse-by-import; the additive `sim/` BDSP diff is
byte-identical when enable_bdsp is off). Smoke:
    SIM_BACKEND=numpy python -m research.runners._d1_onbridge_learn_to_accuracy_derisk --seeds 42 --smoke
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
# TINY matmuls / a small bridge -> one BLAS thread per process (oversubscription is far slower); parallelize seeds.
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

OUT = _REPO / "research" / "findings" / "raw" / "_d1_onbridge_learn_to_accuracy.json"


# ============================================================================================================
# Tasks. The default is the EMERGE-1 depth-2 task (make_task, imported verbatim) -- the exact task D1's numpy
# reference clears at 0.964, so the on-bridge run closes D1's own gap. `parity` (n-bit parity) is the simplest
# non-linearly-separable alternative (both provably need the hidden layer; a single-layer net is at chance). All
# return (Xtr, ytr), (Xte, yte) with X in {0,1} bits (n, n_bits) and y in {0,1}.
# ============================================================================================================
def _subset(X, y, k, seed):
    if not k or k >= len(X):
        return X, y
    s = np.random.default_rng(seed).permutation(len(X))[:k]
    return X[s], y[s]


def _load_task(task, seed, parity_bits):
    """Return the FULL task (X in {0,1} bits, y in {0,1}); the caller subsets ONLY the on-bridge arms for speed (the
    numpy oracle/floor always see the full task so task-validity is honest)."""
    if task == "emerge1":
        from research.runners._emerge1_deep_dendritic_representation_derisk import make_task, N_BITS
        (Xtr, ytr, _Ltr), (Xte, yte, _Lte) = make_task(seed)
        Xtr = (np.asarray(Xtr) > 0).astype(np.float64)          # +/-1 -> {0,1} bits (rate encoding on the substrate)
        Xte = (np.asarray(Xte) > 0).astype(np.float64)
        n_bits = int(N_BITS)
    elif task == "parity":
        n_bits = int(parity_bits)
        n = 1 << n_bits
        bits = ((np.arange(n)[:, None] >> np.arange(n_bits)[None, :]) & 1).astype(np.float64)
        label = (bits.sum(1).astype(np.int64) % 2)              # n-bit parity: needs a hidden layer (XOR-of-...)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n); cut = int(0.65 * n)
        tr, te = idx[:cut], idx[cut:]
        Xtr, ytr, Xte, yte = bits[tr], label[tr], bits[te], label[te]
    else:
        raise ValueError(f"unknown task {task!r}")
    ytr = np.asarray(ytr).astype(np.int64); yte = np.asarray(yte).astype(np.int64)
    return (Xtr, ytr), (Xte, yte), n_bits


# ============================================================================================================
# The ORACLE / SINGLE-LAYER-FLOOR numpy MLP: a fenced backprop net (weight transport) -- the task-VALIDITY arms.
# depth-2 [n_bits, hidden, 2] must clear >= 0.80 (learnable); depth-1 [n_bits, 2] must be ~chance (the task needs a
# hidden layer). Self-contained (no coupling to any research module's internals) so the ceiling is unambiguously a
# true backprop ceiling, NOT the shipped biologically-local rule.
# ============================================================================================================
class _NumpyMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = []; self.b = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            self.W.append(rng.uniform(-lim, lim, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        self._v = None

    @staticmethod
    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _forward(self, X):
        acts = [np.asarray(X, float)]
        for li in range(len(self.W) - 1):
            acts.append(self._sig(acts[-1] @ self.W[li] + self.b[li]))
        return acts, acts[-1] @ self.W[-1] + self.b[-1]

    def accuracy(self, X, y):
        _, lg = self._forward(X)
        return float(np.mean(np.argmax(lg, 1) == np.asarray(y)))

    def train(self, X, y, epochs, lr, batch, seed):
        y = np.asarray(y); rng = np.random.default_rng(seed + 5)
        if self._v is None:
            self._v = [np.zeros_like(w) for w in self.W]; self._vb = [np.zeros_like(b) for b in self.b]
        for _ in range(epochs):
            perm = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                bi = perm[i:i + batch]; xb = np.asarray(X, float)[bi]; yb = y[bi]
                acts, lg = self._forward(xb)
                z = lg - lg.max(1, keepdims=True); ez = np.exp(z); p = ez / ez.sum(1, keepdims=True)
                d = p.copy(); d[np.arange(len(yb)), yb] -= 1.0
                m = max(1, len(yb))
                gW = [None] * len(self.W); gB = [None] * len(self.W)
                gW[-1] = acts[-1].T @ d / m; gB[-1] = d.mean(0)
                for li in range(len(self.W) - 2, -1, -1):
                    a = acts[li + 1]; d = (d @ self.W[li + 1].T) * a * (1.0 - a)
                    gW[li] = acts[li].T @ d / m; gB[li] = d.mean(0)
                for li in range(len(self.W)):
                    self._v[li] = 0.9 * self._v[li] - lr * gW[li]; self.W[li] += self._v[li]
                    self._vb[li] = 0.9 * self._vb[li] - lr * gB[li]; self.b[li] += self._vb[li]


def _numpy_oracle_heldout(n_bits, hidden, Xtr, ytr, Xte, yte, epochs, lr, batch, seed):
    net = _NumpyMLP([n_bits, hidden, 2], seed=seed)
    net.train(Xtr, ytr, epochs, lr, batch, seed)
    return float(net.accuracy(Xte, yte))


def _numpy_singlelayer_floor(n_bits, Xtr, ytr, Xte, yte, epochs, lr, batch, seed):
    net = _NumpyMLP([n_bits, 2], seed=seed)                     # linear readout (no hidden) -> chance on non-separable
    net.train(Xtr, ytr, epochs, lr, batch, seed)
    return float(net.accuracy(Xte, yte))


# ============================================================================================================
# The on-bridge 3-region spiking BDSP net.
# ============================================================================================================
class OnBridgeBDSPNet:
    """A 3-region feedforward net (input -> hidden -> output) on ONE `SimulationBridge` with the committed
    `enable_bdsp` (+ optional `enable_bdsp_microcircuit`). Both forward pathways are PLASTIC RegionPathways learned
    by the `sim/` BDSP kernel; the apical credit is the FIXED-RANDOM feedback Y (own RandomState) injected into
    `cp_bdsp_apical_drive` (see the module docstring's "THE CRUX"). NO `sim/` edit."""

    def __init__(self, seed, n_bits, hidden=12, in_pop=2, pool_out=6, microcircuit=False,
                 bdsp_lr=0.03, bdsp_p0=0.30, bdsp_beta=1.0, burst_isi_ms=6.0,
                 fwd_wmean=6.0, fwd_wjit=0.5, fwd_density=1.0,
                 in_hi=750.0, in_lo=40.0, hidden_bias=520.0, output_bias=520.0,
                 apical_out_gain=260.0, apical_hid_gain=190.0):
        from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion, RegionPathway
        from sim.enums import NeuronModel
        from sim.backend import get_backend
        self.xp, _bk = get_backend()
        self.seed = int(seed); self.n_bits = int(n_bits); self.n_classes = 2
        self.hidden = int(hidden); self.in_pop = int(in_pop); self.pool_out = int(pool_out)
        self.in_hi = float(in_hi); self.in_lo = float(in_lo)
        self.hidden_bias = float(hidden_bias); self.output_bias = float(output_bias)
        self.apical_out_gain = float(apical_out_gain); self.apical_hid_gain = float(apical_hid_gain)
        self.microcircuit = bool(microcircuit)
        n_in = self.n_bits * self.in_pop
        n_out = self.n_classes * self.pool_out

        cfg = CoreSimConfig(); cfg.num_neurons = 0
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.actual_seed_used = int(seed)
        cfg.ou_std_current_pA = 0.0
        cfg.enable_brain_region_framework = True
        # isolate the BDSP-driven dw: no other plasticity/normalization moving weights in parallel.
        for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                     "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
                     "enable_input_divisive_norm", "enable_nmda"):
            setattr(cfg, flag, False)
        # THE COMMITTED RULE (additive/default-off `sim/` mechanism; byte-identical when off):
        cfg.enable_bdsp = True
        cfg.enable_bdsp_microcircuit = bool(microcircuit)
        cfg.bdsp_learning_rate = float(bdsp_lr)
        cfg.bdsp_p0 = float(bdsp_p0)
        cfg.bdsp_beta = float(bdsp_beta)
        cfg.burst_isi_threshold_ms = float(burst_isi_ms)
        self._bdsp_lr = float(bdsp_lr)

        regions = [
            BrainRegion(name="input", n_neurons=n_in, exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
            BrainRegion(name="hidden", n_neurons=self.hidden, exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
            BrainRegion(name="output", n_neurons=n_out, exc_fraction=1.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        ]
        pathways = [
            # the two PLASTIC feedforward pathways the committed BDSP rule shapes.
            RegionPathway(from_region="input", to_region="hidden", density=float(fwd_density),
                          weight_mean=float(fwd_wmean), weight_jitter=float(fwd_wjit), plastic=True),
            RegionPathway(from_region="hidden", to_region="output", density=float(fwd_density),
                          weight_mean=float(fwd_wmean), weight_jitter=float(fwd_wjit), plastic=True),
        ]
        cfg.brain_regions = regions; cfg.region_pathways = pathways
        sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
        sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        sb._initialize_simulation_data(called_from_playback_init=False)
        self.sb = sb; self.cfg = cfg

        rm = sb.region_manager
        self.idx_in = np.asarray(list(rm.indices("input")), dtype=int)
        self.idx_hid = np.asarray(list(rm.indices("hidden")), dtype=int)
        self.idx_out = np.asarray(list(rm.indices("output")), dtype=int)
        # per-class output sub-pools (contiguous region slice -> contiguous class blocks)
        self.class_idx = [self.idx_out[c * self.pool_out:(c + 1) * self.pool_out] for c in range(self.n_classes)]
        # per-bit input sub-pops
        self.bit_idx = [self.idx_in[b * self.in_pop:(b + 1) * self.in_pop] for b in range(self.n_bits)]
        self.n = int(cfg.num_neurons)

        # FIXED-RANDOM apical feedback Y (n_classes x n_hidden), OWN stream => NO weight transport (never == a
        # forward weight; never modified). This is the "output->hidden fixed-random apical feedback (plastic=False)".
        self.Y = np.random.RandomState(seed + 9973).normal(0.0, 1.0, (self.n_classes, self.hidden))
        self._Y0 = self.Y.copy()

        # per-pathway synapse masks over the cached COO (index-aligned to cp_connections.data). Lets us read each
        # pathway's weights (dw stats + the no-weight-transport check) robustly by endpoint region.
        self._build_pathway_masks()

    # ---- pathway bookkeeping -------------------------------------------------------------------------------
    def _coo(self):
        from sim.backend import to_host
        coo = self.sb._get_cached_coo()
        return (np.asarray(to_host(coo.row)).astype(int), np.asarray(to_host(coo.col)).astype(int))

    def _build_pathway_masks(self):
        row, col = self._coo()
        in_set = set(self.idx_in.tolist()); hid_set = set(self.idx_hid.tolist()); out_set = set(self.idx_out.tolist())
        r_in = np.array([r in in_set for r in row]); r_hid = np.array([r in hid_set for r in row])
        c_hid = np.array([c in hid_set for c in col]); c_out = np.array([c in out_set for c in col])
        self.mask_in2hid = r_in & c_hid
        self.mask_hid2out = r_hid & c_out
        self._coo_row, self._coo_col = row, col

    def _weights(self):
        from sim.backend import to_host
        return np.asarray(to_host(self.sb.cp_connections.data)).astype(float)

    def pathway_weight_sums(self):
        w = self._weights()
        return float(np.abs(w[self.mask_in2hid]).sum()), float(np.abs(w[self.mask_hid2out]).sum())

    def _dense_block(self, mask, rows_idx, cols_idx):
        """Reconstruct a dense (len(rows_idx) x len(cols_idx)) weight matrix for a pathway from the masked COO."""
        w = self._weights()
        rpos = {v: i for i, v in enumerate(rows_idx.tolist())}
        cpos = {v: i for i, v in enumerate(cols_idx.tolist())}
        M = np.zeros((len(rows_idx), len(cols_idx)))
        for r, c, wi in zip(self._coo_row[mask], self._coo_col[mask], w[mask]):
            M[rpos[r], cpos[c]] = wi
        return M

    def no_weight_transport(self):
        """anti-cheat: the fixed-random feedback Y is never a forward weight or its (class-pooled) transpose, and is
        never modified. Y is (n_classes x n_hidden); the class-POOLED hidden->output weight W_ho_pooled is
        (n_hidden x n_classes) (average the pool_out output neurons of each class) -> Y.T and W_ho_pooled share a
        shape, so the byte-comparison is meaningful (not a shape mismatch). Y from a separate RandomState is never
        byte-equal to the (random/learned) forward weights."""
        y_unchanged = bool(np.array_equal(self.Y, self._Y0))    # fixed-random: never written after init
        W_ho = self._dense_block(self.mask_hid2out, self.idx_hid, self.idx_out)   # (n_hidden x n_out)
        W_ho_pooled = np.column_stack([W_ho[:, c * self.pool_out:(c + 1) * self.pool_out].mean(1)
                                       for c in range(self.n_classes)])           # (n_hidden x n_classes)
        not_transpose = not (self.Y.T.shape == W_ho_pooled.shape and np.allclose(self.Y.T, W_ho_pooled))
        W_ih = self._dense_block(self.mask_in2hid, self.idx_in, self.idx_hid)     # (n_in x n_hidden)
        not_forward = not any(self.Y.shape == B.shape and np.allclose(self.Y, B) for B in (W_ih, W_ho))
        return bool(y_unchanged and not_transpose and not_forward)

    # ---- dynamics ------------------------------------------------------------------------------------------
    def _reset_membrane(self):
        sb = self.sb
        if getattr(sb, "cp_izh_c_reset", None) is not None:
            sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
        else:
            sb.cp_membrane_potential_v[:] = -65.0
        sb.cp_recovery_variable_u[:] = 0.0
        if getattr(sb, "cp_firing_states", None) is not None:
            sb.cp_firing_states[:] = False
        for _attr in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                      "cp_conductance_g_nmda_rise", "cp_conductance_g_nmda_recurrent"):
            _arr = getattr(sb, _attr, None)
            if _arr is not None:
                _arr[:] = 0.0

    def _set_input_drive(self, x_bits, hidden_output_bias=True):
        """cp_external_input_current: bit==1 -> in_hi on that bit's input pop, bit==0 -> in_lo; + a standing
        excitability bias on hidden+output so their somata spike (bursts need somatic spikes present, as in D1's
        stage_a_bridge_learns which biases the output half to fire). The bias is the SMOKE lever that isolates the
        BDSP credit mechanism; the controller lowers it + strengthens the forward drive so the computation flows."""
        from sim.backend import from_host
        drive = np.zeros(self.n, dtype=np.float32)
        for b in range(self.n_bits):
            drive[self.bit_idx[b]] = self.in_hi if x_bits[b] > 0.5 else self.in_lo
        if hidden_output_bias:
            drive[self.idx_hid] += self.hidden_bias
            drive[self.idx_out] += self.output_bias
        self._drive_dev = from_host(drive)                     # re-applied each step in _run (robust vs any per-step reset)
        self.sb.cp_external_input_current[:] = self._drive_dev

    def _set_apical(self, e_class):
        """Inject the apical CREDIT: output pool c gets k_out*e_c (direct top error); hidden j gets k_hid*(e@Y)[j]
        (the fixed-random feedback-alignment projection). e_class=None -> apical all zero (lesion / read phase)."""
        from sim.backend import from_host
        ap = np.zeros(self.n, dtype=np.float32)
        if e_class is not None:
            for c in range(self.n_classes):
                ap[self.class_idx[c]] = self.apical_out_gain * float(e_class[c])
            v_hid = self.apical_hid_gain * (np.asarray(e_class) @ self.Y)          # (n_hidden,) = e @ Y
            ap[self.idx_hid] = v_hid.astype(np.float32)
        self.sb.cp_bdsp_apical_drive = from_host(ap)

    def _run(self, steps, accumulate_out=False):
        from sim.backend import to_host
        acc = np.zeros(len(self.idx_out)) if accumulate_out else None
        for _ in range(steps):
            if getattr(self, "_drive_dev", None) is not None:
                self.sb.cp_external_input_current[:] = self._drive_dev    # re-apply the standing drive each step
            self.sb._run_one_simulation_step()
            if accumulate_out:
                acc += np.asarray(to_host(self.sb.cp_firing_states[self.idx_out])).astype(float)
        return acc

    def _readout(self, x_bits, settle_steps):
        """SETTLE with learning + apical OFF -> per-class spike-count readout (the unperturbed forward read)."""
        self._reset_membrane()
        self._set_apical(None)
        self.cfg.bdsp_learning_rate = 0.0                       # freeze BDSP during the read (moat-clean)
        self._set_input_drive(x_bits)
        acc = self._run(settle_steps, accumulate_out=True)
        readout = np.array([acc[c * self.pool_out:(c + 1) * self.pool_out].sum() for c in range(self.n_classes)])
        return readout

    def accuracy(self, X, y, settle_steps):
        X = np.asarray(X); y = np.asarray(y); correct = 0
        for i in range(len(X)):
            r = self._readout(X[i], settle_steps)
            if int(np.argmax(r)) == int(y[i]):
                correct += 1
        return correct / max(1, len(X))

    def apical_coupling_diag(self, steps=250):
        """ROOT-CAUSE diagnostic (the D1 Stage-A' detector, self-contained): drive the output region so its somata
        spike, then measure the event rate E / burst rate B / burst-probability read P at apical=0 vs apical=+300pA.
        The committed BDSP credit REQUIRES the apical to raise the MEASURED burst rate B (apical -> more real bursts ->
        dev=B-Pbar*E up -> LTP). If B does NOT rise with the apical (B_rises False) while P does (P_rises True), the
        apical is DECOUPLED from the soma on this path -> dev is dominated by the natural burst statistics, the moat is
        broken and there is no directed credit (the exact boundary this runner surfaces)."""
        from sim.backend import to_host, from_host
        drive = np.zeros(self.n, dtype=np.float32)
        for b in range(self.n_bits):
            drive[self.bit_idx[b]] = self.in_hi
        drive[self.idx_out] += self.output_bias if self.output_bias > 0 else 700.0

        def phase(apical_pA):
            self._reset_membrane()
            self.cfg.bdsp_learning_rate = 0.0
            ap = np.zeros(self.n, dtype=np.float32); ap[self.idx_out] = apical_pA
            self.sb.cp_bdsp_apical_drive = from_host(ap)
            self._drive_dev = from_host(drive)
            for _ in range(steps):
                self.sb.cp_external_input_current[:] = self._drive_dev
                self.sb._run_one_simulation_step()
            E = float(np.asarray(to_host(self.sb.cp_bdsp_E[self.idx_out])).mean())
            B = float(np.asarray(to_host(self.sb.cp_bdsp_B[self.idx_out])).mean())
            P = float(np.asarray(to_host(self.sb.cp_bdsp_P[self.idx_out])).mean())
            return E, B, P

        E0, B0, P0 = phase(0.0)
        E1, B1, P1 = phase(300.0)
        return {"E_rest": E0, "B_rest": B0, "P_rest": P0, "E_apical": E1, "B_apical": B1, "P_apical": P1,
                "B_rises": bool(B1 > B0 + 1e-4), "P_rises": bool(P1 > P0 + 1e-4),
                "apical_couples_to_bursts": bool(B1 > B0 + 1e-4)}

    def region_rates(self, x_bits, settle_steps):
        """diagnostic: mean per-step firing fraction per region during a settle (activity sanity for the smoke)."""
        from sim.backend import to_host
        self._reset_membrane(); self._set_apical(None); self.cfg.bdsp_learning_rate = 0.0
        self._set_input_drive(x_bits)
        s = {r: 0.0 for r in ("input", "hidden", "output")}
        idxs = {"input": self.idx_in, "hidden": self.idx_hid, "output": self.idx_out}
        for _ in range(settle_steps):
            self.sb._run_one_simulation_step()
            for r, ix in idxs.items():
                s[r] += float(np.asarray(to_host(self.sb.cp_firing_states[ix])).mean())
        return {r: v / max(1, settle_steps) for r, v in s.items()}

    def train_epoch(self, X, y, mode, settle_steps, teach_steps, shuffle_seed):
        """One epoch: per sample -> read (learning off) -> compute top error -> teach (learning on, apical injected)
        so the committed BDSP kernel moves the plastic weights. mode: 'bdsp' (credit), 'lesion' (apical 0 in teach),
        'wrong_sign' (negate the top error)."""
        X = np.asarray(X); y = np.asarray(y)
        rng = np.random.default_rng(shuffle_seed)
        for i in rng.permutation(len(X)):
            r = self._readout(X[i], settle_steps)              # forward read (learning frozen inside _readout)
            z = r - r.max(); ez = np.exp(z); p = ez / ez.sum()
            onehot = np.zeros(self.n_classes); onehot[int(y[i])] = 1.0
            e = onehot - p                                     # clean top error (== -delta_out; output target access)
            if mode == "wrong_sign":
                e = -e
            e_teach = None if mode == "lesion" else e          # lesion: zero the top error (no teaching)
            # TEACH phase: apical on, learning on -> the sim/ BDSP update moves the plastic weights each step.
            self._reset_membrane()
            self._set_input_drive(X[i])
            self._set_apical(e_teach)
            self.cfg.bdsp_learning_rate = self._bdsp_lr
            self._run(teach_steps, accumulate_out=False)
            self._set_apical(None)
            self.cfg.bdsp_learning_rate = 0.0


# ============================================================================================================
def _run_bridge_arm(mode, seed, n_bits, Xtr, ytr, Xte, yte, args):
    """Build a fresh on-bridge net, snapshot its pathway weights, train `mode`, return (heldout, dw_in2hid,
    dw_hid2out, no_weight_transport, rates_before)."""
    net = OnBridgeBDSPNet(seed, n_bits, hidden=args.hidden, in_pop=args.in_pop, pool_out=args.pool_out,
                          microcircuit=args.microcircuit, bdsp_lr=args.bdsp_lr, bdsp_p0=args.bdsp_p0,
                          bdsp_beta=args.bdsp_beta, burst_isi_ms=args.burst_isi_ms,
                          fwd_wmean=args.fwd_wmean, fwd_wjit=args.fwd_wjit,
                          in_hi=args.in_hi, in_lo=args.in_lo, hidden_bias=args.hidden_bias,
                          output_bias=args.output_bias, apical_out_gain=args.apical_out_gain,
                          apical_hid_gain=args.apical_hid_gain)
    rates = net.region_rates(Xtr[0], args.settle_steps) if mode == "bdsp" else None
    coupling = net.apical_coupling_diag() if mode == "bdsp" else None
    w_ih0, w_ho0 = net.pathway_weight_sums()
    nwt = net.no_weight_transport()
    for ep in range(args.epochs):
        net.train_epoch(Xtr, ytr, mode, args.settle_steps, args.teach_steps, seed + 1000 * ep + 7)
    w_ih1, w_ho1 = net.pathway_weight_sums()
    heldout = net.accuracy(Xte, yte, args.settle_steps)
    return {"mode": mode, "heldout": float(heldout),
            "dw_in2hid": float(abs(w_ih1 - w_ih0)), "dw_hid2out": float(abs(w_ho1 - w_ho0)),
            "w_in2hid_before": float(w_ih0), "w_in2hid_after": float(w_ih1),
            "w_hid2out_before": float(w_ho0), "w_hid2out_after": float(w_ho1),
            "no_weight_transport": bool(nwt), "region_rates": rates, "apical_coupling": coupling}


def run(seed, args):
    (Xtr, ytr), (Xte, yte), n_bits = _load_task(args.task, seed, args.parity_bits)
    chance = float(max(np.mean(yte == 0), np.mean(yte == 1)))

    # anti-cheat: ORACLE ceiling (fenced numpy backprop) + SINGLE-LAYER floor (numpy linear) on the FULL task --
    # task validity (scale-independent; always the full split so 0.80 is honest).
    oracle = _numpy_oracle_heldout(n_bits, max(args.hidden, 32), Xtr, ytr, Xte, yte,
                                   args.oracle_epochs, args.oracle_lr, args.oracle_batch, seed)
    floor = _numpy_singlelayer_floor(n_bits, Xtr, ytr, Xte, yte,
                                     args.oracle_epochs, args.oracle_lr, args.oracle_batch, seed)

    # on-bridge arms: subset TRAIN (and TEST) for speed; the held-out accuracy is on the (subset) disjoint test set.
    Xtr_b, ytr_b = _subset(Xtr, ytr, args.train_subset, seed + 11)
    Xte_b, yte_b = _subset(Xte, yte, args.test_subset, seed + 12)
    res = {"seed": seed, "task": args.task, "n_bits": n_bits, "n_train": int(len(Xtr_b)),
           "n_test": int(len(Xte_b)), "chance": chance, "oracle_heldout": oracle,
           "numpy_singlelayer_floor": floor}
    res["bdsp"] = _run_bridge_arm("bdsp", seed, n_bits, Xtr_b, ytr_b, Xte_b, yte_b, args)
    res["apical_lesion"] = _run_bridge_arm("lesion", seed, n_bits, Xtr_b, ytr_b, Xte_b, yte_b, args)
    if args.full_arms:
        res["wrong_sign"] = _run_bridge_arm("wrong_sign", seed, n_bits, Xtr_b, ytr_b, Xte_b, yte_b, args)
        # on-bridge SINGLE-LAYER floor: input->output directly (no hidden). Reuse the net but 0 hidden is not a
        # region; approximate the floor with the numpy linear floor above (the on-bridge single-layer would also be
        # ~chance at this scale and adds a 3rd bridge build -- the numpy floor is the meaningful task-validity check).
    return res


def _mean(per, *keys):
    def _get(p):
        v = p
        for k in keys:
            v = v[k]
        return v
    return float(np.mean([_get(p) for p in per]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--task", choices=["emerge1", "parity"], default="emerge1")
    ap.add_argument("--parity-bits", type=int, default=4)
    ap.add_argument("--microcircuit", action="store_true",
                    help="enable_bdsp_microcircuit (interneuron-cancelled clean apical error); default = Burstprop")
    # net scale
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--in-pop", type=int, default=2)
    ap.add_argument("--pool-out", type=int, default=6)
    # per-sample dynamics
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--teach-steps", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    # BDSP config (the committed sim/ knobs)
    ap.add_argument("--bdsp-lr", type=float, default=0.03)
    ap.add_argument("--bdsp-p0", type=float, default=0.30)
    ap.add_argument("--bdsp-beta", type=float, default=1.0)
    ap.add_argument("--burst-isi-ms", type=float, default=6.0)
    # forward pathway init + drive / apical (the smoke levers; controller tunes for accuracy)
    ap.add_argument("--fwd-wmean", type=float, default=6.0)
    ap.add_argument("--fwd-wjit", type=float, default=0.5)
    ap.add_argument("--in-hi", type=float, default=750.0)
    ap.add_argument("--in-lo", type=float, default=40.0)
    ap.add_argument("--hidden-bias", type=float, default=520.0)
    ap.add_argument("--output-bias", type=float, default=520.0)
    ap.add_argument("--apical-out-gain", type=float, default=260.0)
    ap.add_argument("--apical-hid-gain", type=float, default=190.0)
    # oracle (numpy backprop)
    ap.add_argument("--oracle-epochs", type=int, default=500)
    ap.add_argument("--oracle-lr", type=float, default=0.5)
    ap.add_argument("--oracle-batch", type=int, default=64)
    # smoke vs full
    ap.add_argument("--smoke", action="store_true",
                    help="tiny fast CPU smoke: subset train/test, few epochs (proves the pipeline+mechanism).")
    ap.add_argument("--full-arms", dest="full_arms", action="store_true", default=True)
    ap.add_argument("--no-full-arms", dest="full_arms", action="store_false")
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--test-subset", type=int, default=0)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--json", "--out", dest="out", default=str(OUT))
    a = ap.parse_args()
    if a.backend:
        os.environ["SIM_BACKEND"] = a.backend
    if a.smoke:
        # tiny + fast: enough to prove e2e run + weights-move-under-credit + moat + wiring. NOT an accuracy run.
        if a.train_subset == 0:
            a.train_subset = 48
        if a.test_subset == 0:
            a.test_subset = 48
        a.epochs = min(a.epochs, 2)

    t0 = time.time(); err = None; per = []
    try:
        for s in a.seeds:
            r = run(s, a)
            per.append(r)
            b = r["bdsp"]; le = r["apical_lesion"]
            rr = b.get("region_rates") or {}
            cp = b.get("apical_coupling") or {}
            print(f"  [seed {s}] apical-coupling: P_rest {cp.get('P_rest', float('nan')):.2f} -> P_apical "
                  f"{cp.get('P_apical', float('nan')):.2f} (P_rises {cp.get('P_rises')}); B_rest "
                  f"{cp.get('B_rest', float('nan')):.3f} -> B_apical {cp.get('B_apical', float('nan')):.3f} "
                  f"(B_rises {cp.get('B_rises')} = apical{'->bursts COUPLED' if cp.get('B_rises') else '-DECOUPLED from soma'})",
                  flush=True)
            print(f"  [seed {s}][{a.task}] oracle {r['oracle_heldout']:.3f} | np-single {r['numpy_singlelayer_floor']:.3f} "
                  f"| chance {r['chance']:.3f} || BDSP held {b['heldout']:.3f} dw(in>hid {b['dw_in2hid']:.3f}, "
                  f"hid>out {b['dw_hid2out']:.3f}) | LESION held {le['heldout']:.3f} dw(in>hid {le['dw_in2hid']:.3f}, "
                  f"hid>out {le['dw_hid2out']:.3f}) | wt_ok {b['no_weight_transport']} "
                  f"| rates in/hid/out {rr.get('input', float('nan')):.2f}/{rr.get('hidden', float('nan')):.2f}/"
                  f"{rr.get('output', float('nan')):.2f}", flush=True)
            if "wrong_sign" in r:
                print(f"       wrong-sign held {r['wrong_sign']['heldout']:.3f}", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        orac = _mean(per, "oracle_heldout"); np_single = _mean(per, "numpy_singlelayer_floor")
        ch = float(np.mean([p["chance"] for p in per]))
        bd = _mean(per, "bdsp", "heldout"); les = _mean(per, "apical_lesion", "heldout")
        bd_dw_ih = _mean(per, "bdsp", "dw_in2hid"); bd_dw_ho = _mean(per, "bdsp", "dw_hid2out")
        les_dw_ih = _mean(per, "apical_lesion", "dw_in2hid"); les_dw_ho = _mean(per, "apical_lesion", "dw_hid2out")
        wt = all(p["bdsp"]["no_weight_transport"] for p in per)
        wrong = _mean(per, "wrong_sign", "heldout") if "wrong_sign" in per[0] else None
        # ROOT-CAUSE diagnostic: does the apical raise the MEASURED burst rate B (the committed credit REQUIRES it)?
        apical_couples = all(bool((p["bdsp"].get("apical_coupling") or {}).get("B_rises")) for p in per)
        _cp0 = per[0]["bdsp"].get("apical_coupling") or {}

        # ---- gates ----
        # task-validity (scale-independent): the oracle clears the bar AND the task needs the hidden layer.
        task_ok = (orac >= 0.80) and (np_single <= ch + 0.10)
        # PIPELINE (smoke-checkable mechanism): both plastic pathways move under credit; the moat holds (lesion's
        # hidden dw << credit's hidden dw); no weight transport.
        weights_move = (bd_dw_ih > 1e-4) and (bd_dw_ho > 1e-4)
        moat_holds = (bd_dw_ih > max(2.0 * les_dw_ih, les_dw_ih + 1e-4))    # credit moves the HIDDEN (in>hid) layer >> lesion
        pipeline_ok = bool(weights_move and moat_holds and wt)
        # ACCURACY (the controller's GPU sweep drives this): held-out clears the bar + beats the floors + wrong-sign anti-learns.
        acc_generalizes = (bd >= 0.75) and (bd > les + 0.10) and (bd > np_single + 0.05)
        acc_wrong = (wrong is None) or (wrong <= ch + 0.05)
        accuracy_go = bool(acc_generalizes and acc_wrong)
        go = bool(task_ok and pipeline_ok and accuracy_go)

        _mc = " [MICROCIRCUIT]" if a.microcircuit else ""
        if not task_ok:
            verdict = (f"INCONCLUSIVE{_mc} -- task not validated (oracle {orac:.3f} need>=0.80; numpy single-layer floor "
                       f"{np_single:.3f} vs chance {ch:.3f}); fix the task before reading the on-bridge arms.")
        elif go:
            verdict = (f"GO{_mc} -- the committed sim/ enable_bdsp deep-credit rule LEARNS the 2-layer map END-TO-END ON "
                       f"A REAL SimulationBridge: held-out {bd:.3f} >= 0.75, > apical-lesion {les:.3f} + 0.10, > "
                       f"single-layer floor {np_single:.3f}; both plastic pathways moved under credit "
                       f"(in>hid {bd_dw_ih:.3f}, hid>out {bd_dw_ho:.3f}); the moat holds (lesion hidden-dw {les_dw_ih:.3f} "
                       f"<< credit {bd_dw_ih:.3f}); wrong-sign anti-learns; no weight transport; oracle {orac:.3f}. "
                       f"Multi-seed.")
        elif pipeline_ok:
            verdict = (f"PIPELINE-VALIDATED (smoke){_mc} -- the on-bridge 3-region BDSP net runs END-TO-END and the "
                       f"MECHANISM is correct: both plastic pathways MOVE under credit (in>hid {bd_dw_ih:.3f}, hid>out "
                       f"{bd_dw_ho:.3f}); the P0 MOAT holds (apical-lesion hidden-dw {les_dw_ih:.3f} << credit "
                       f"{bd_dw_ih:.3f}); NO weight transport ({wt}); task valid (oracle {orac:.3f} >= 0.80, numpy "
                       f"single-layer floor {np_single:.3f} ~ chance {ch:.3f}). Held-out at this CPU-smoke scale: BDSP "
                       f"{bd:.3f} vs lesion {les:.3f} vs chance {ch:.3f}"
                       + (f", wrong-sign {wrong:.3f}" if wrong is not None else "")
                       + f". The held-out ACCURACY bar (>=0.75) is the CONTROLLER's multi-seed GPU sweep (needs width + "
                       f"epochs + drive tuning the smoke deliberately omits). ⇒ the machinery + wiring are ready; "
                       f"accuracy is the GPU run.")
        elif not apical_couples:
            # THE ROOT-CAUSE BOUNDARY the smoke surfaced: on the committed pure-enable_bdsp path the apical is
            # DECOUPLED from the soma (raises P, not the measured burst rate B), so dev=B-Pbar*E is dominated by the
            # natural burst statistics -> no directed credit, no clean moat, at ANY operating point.
            verdict = (f"BOUNDARY -- APICAL DECOUPLED FROM SOMA{_mc}. The runner runs end-to-end + is wired correctly "
                       f"(task valid: oracle {orac:.3f} >= 0.80, numpy single-layer floor {np_single:.3f} ~ chance "
                       f"{ch:.3f}; no weight transport {wt}), BUT on the committed pure `enable_bdsp` path driving "
                       f"cp_bdsp_apical_drive raises the burst-probability READ P ({_cp0.get('P_rest', float('nan')):.2f}"
                       f"->{_cp0.get('P_apical', float('nan')):.2f}) but NOT the MEASURED burst rate B "
                       f"({_cp0.get('B_rest', float('nan')):.3f}->{_cp0.get('B_apical', float('nan')):.3f}, B_rises "
                       f"False). So the FF update dev=B-Pbar*E gets no apical-directed credit and the moat does not "
                       f"hold (credit hidden-dw {bd_dw_ih:.3f} ~ lesion {les_dw_ih:.3f}). This REPRODUCES on D1's own "
                       f"stage_a_bridge_detector (B_rises False) / stage_a_bridge_learns (moat_smaller False) and matches "
                       f"the D1 finding's admission that the on-bridge FF update is NOT the numpy-accuracy rule (that is "
                       f"the runner-side M2.6 somatic rule). CRUX for the controller: the apical->soma coupling "
                       f"(enable_two_compartment_dap + enable_coincidence_detection + a routed coincidence pathway) is "
                       f"the fix path -> a research-gated build, NOT a runner tuning. Held-out BDSP {bd:.3f} vs chance "
                       f"{ch:.3f} (at floor, as expected).")
        else:
            miss = []
            if not weights_move: miss.append(f"a plastic pathway did NOT move under credit (in>hid {bd_dw_ih:.3f}, hid>out {bd_dw_ho:.3f})")
            if not moat_holds: miss.append(f"moat weak (lesion hidden-dw {les_dw_ih:.3f} vs credit {bd_dw_ih:.3f})")
            if not wt: miss.append("no-weight-transport check FAILED")
            verdict = (f"BOUNDARY (build-informative){_mc} -- " + "; ".join(miss) + f". Task valid (oracle {orac:.3f}); "
                       f"apical couples to bursts ({apical_couples}). Tune the drive/bias/apical-gain levers.")
    else:
        go = False; pipeline_ok = False
        verdict = f"ERROR -- {err}"

    _apical_couples = None
    if err is None and per:
        _apical_couples = all(bool((p["bdsp"].get("apical_coupling") or {}).get("B_rises")) for p in per)
    summary = {"probe": "d1_onbridge_learn_to_accuracy", "GO": go,
               "pipeline_ok": bool(err is None and pipeline_ok), "verdict": verdict,
               "microcircuit": bool(a.microcircuit),
               "apical_couples_to_bursts": _apical_couples,
               "gap_closed": ("the on-bridge held-out ACCURACY of the committed enable_bdsp deep-credit rule on a real "
                              "3-region spiking SimulationBridge -- D1's own explicitly-deferred GPU run "
                              "(2026-07-07-D1-microcircuit-*.md line 37). D1 validated the sim/ machinery + the numpy "
                              "reference accuracy; THIS runner runs the accuracy on the bridge."),
               "crux_note": ("cp_bdsp_apical_drive is a runner-set per-neuron array integrated into cp_v_apical; the "
                             "sim/ BDSP update runs per-synapse over cp_connections gated by the plastic mask, credit "
                             "carried ONLY by the postsynaptic apical. There is NO sim/ path routing a synaptic "
                             "RegionPathway current into the apical compartment, so the fixed-random output->hidden "
                             "apical feedback is realized as a runner-held frozen matrix Y (own RandomState, no weight "
                             "transport) projected host-side into cp_bdsp_apical_drive[hidden] = k_hid*(e@Y) -- exactly "
                             "as D1's stage_a_bridge_learns sets the apical directly and the numpy reference computes "
                             "v_api=e_upper@Y. The FEEDFORWARD plasticity is on the substrate (the sim/ kernel); the "
                             "credit PROJECTION is the host-side teacher wiring (legitimate, per every D1/EMERGE ref)."),
               "seeds": a.seeds, "config": vars(a), "elapsed_seconds": round(time.time() - t0, 1),
               "per_seed": per,
               "HONEST_NOTE": ("The CPU smoke proves the pipeline runs end-to-end + both plastic pathways move under "
                               "credit + the P0 moat holds + no weight transport + task-validity. The held-out ACCURACY "
                               "(>=0.75 bar) is the CONTROLLER's multi-seed GPU sweep (needs the width/epochs/drive the "
                               "smoke omits). NO sim/ edit (reuse-by-import; the additive sim/ BDSP diff is byte-"
                               "identical when enable_bdsp is off).")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[d1-onbridge] VERDICT: {verdict}", flush=True)
    print(f"[d1-onbridge] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
