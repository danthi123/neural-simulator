"""gap#4 transport-ceiling READOUT lever -- make the copied-weight ceiling interpretable, then compare
micro_inengine vs fixed_fa vs a frozen control at ONE cfg.seed (plan step S19; GPU step G5).

PRE-REGISTRATION: research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md (commit
1b6c15690, before any run of this file). Evaluation seeds need the committed AMENDMENT path (--prereg-amendment).

WHY. The 2026-09-15 in-engine self-predicting-interneuron run was UNDEFINED: the transport_ceiling arm (Y := pooled
forward W^T) never cleared chance, and gpu_queue.log shows TRAIN accuracy at chance in every arm, the frozen readout
included. The instrument fails upstream of deep credit: the output read / readout learning cannot carry the class.
Levers (all default to the legacy value, so with no flag this net IS Gap4InEngineNet -- shown in data by
--identity-selftest):
  --settle-steps S --read-window W  LONGER read: acts = pooled event rate time-averaged over the last W settle steps
                                    (legacy W=0: the ~10 ms low-pass snapshot at the last step).
  --read-gain g                     STRONGER read: the output error uses softmax(g * rates) (divisive-normalization
                                    gain of the output population; legacy g=1, near-uniform over rates in [0,0.3]).
  --isi-steps I                     inter-stimulus relaxation after each credit phase: no stimulus, no teaching, so
                                    the negative P-Pbar lobe does not land on the NEXT example's input.
  --eval-frozen                     instrument fix: bdsp_learning_rate = 0 during evaluation reads (restored after).
  --spi-silence-outside-credit      instrument fix: the in-engine interneuron rate is zeroed after the credit phase
                                    (the engine otherwise keeps projecting an uncancelled -int_drive into the next
                                    forward pass and into evaluation reads; the runner-supplied micro arm never did).

ARMS (ONE cfg.seed per seed => the same neurons in every arm):
  frozen            hidden apical 0 (Gap4 'reservoir'): only the output layer learns   = the credit-independent floor
  fixed_fa          fixed random Y descent                                           = the frozen-signal baseline
  micro_inengine    fixed Y top-down minus the in-engine LEARNED interneuron cancellation = the mechanism under test
  transport_ceiling Y := pooled forward W^T each example                             = the interpretability ceiling
  micro_inengine_lesion / micro_inengine_freeze_spi                                  = anti-cheats (run on demand)
R task REPLICATES per seed (FA-wall coverage): replicate r uses task seed `seed` (r=0; the 2026-09-15 task) or
`seed + 10007*r`. cfg.seed = seed in every replicate and arm.

CHECKPOINT / RESUME: every (seed, replicate, arm) shard is written atomically to <ckpt-dir>/s{seed}_r{r}_{arm}.json
with a config fingerprint; a restarted run skips completed shards whose fingerprint matches. Shards can run as
separate processes (one --arms value each) against one ckpt dir; --aggregate-only merges them into --out.

DECLARED HOST RESIDUALS: the credit projection (softmax of output rates, error, Y @ error, the transport copy) and the
argmax read-out are host code, as in every gap#4 runner; g is a parameter of that residual. decode_h2 (ridge on the
pooled top-hidden read) is an INSTRUMENT, never a pathway. Forward pass, BDSP weight changes and the in-engine
interneuron learning are on the spiking substrate. Functional read-outs only.

RUN (dev calibration, numpy, seed 7):
  SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -u -m research.runners._gap4_transport_ceiling_readout_derisk \
     --seeds 7 --replicates 0 --arms frozen transport_ceiling --hidden 32 --pool-k 4 --epochs 10 \
     --train-subsample 400 --settle-steps 100 --read-window 80 --read-gain 20 --isi-steps 60 --eval-frozen \
     --spi-silence-outside-credit --out research/findings/raw/gap4/transport_ceiling_readout/calib_C4_s7.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import sys
import time
import traceback
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

import research.runners._gap4_onbridge_spiking_selfpredict_derisk as _ob_mod  # noqa: E402
import research.runners._gap4_selfpredict_interneuron_inengine_derisk as _ie_mod  # noqa: E402
from research.runners._gap4_onbridge_spiking_selfpredict_derisk import _ast_no_forward_W  # noqa: E402
from research.runners._gap4_selfpredict_interneuron_inengine_derisk import Gap4InEngineNet  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

PREREG = "research/findings/2026-09-24-gap4-transport-ceiling-readout-lever-PREREGISTRATION.md"
RAW = _REPO / "research" / "findings" / "raw" / "gap4" / "transport_ceiling_readout"
EVAL_SEEDS = {42, 43, 44, 100, 101, 102}
# AMENDMENT 5 (review fix): the apical-lesion anti-cheat is a MATCHED cut. In micro_inengine the engine forms the
# interneuron cancellation from cp_spi_int_rate every step; the parent's apical_lesion zeroes only the raw teacher, so the
# untrained cancellation (random W^PI) kept driving the top hidden apical (measured 17.35 mV max |v_apical - rest|).
# The lesion now also zeroes the interneuron rate, and every lesion shard records whether the hidden apical stayed at rest.
LESION_KIND = "matched: hidden Y zeroed AND interneuron rate zeroed (engine int_drive 0)"
LESION_SILENT_TOL_MV = 1e-6
# AMENDMENT 5 (review fix): an evaluation seed runs only under a COMMITTED amendment whose heading names an
# EVALUATION CONFIG and whose body registers this run's config fingerprint and seeds.
_EVAL_HEADING = "EVALUATION CONFIG"
_EVAL_FP_KEY = "evaluation-config-fingerprint:"
_EVAL_SEEDS_KEY = "evaluation-seeds:"
CORE_ARMS = ["frozen", "fixed_fa", "micro_inengine", "transport_ceiling"]
ALL_ARMS = CORE_ARMS + ["micro_inengine_lesion", "micro_inengine_freeze_spi"]
# arm -> (Gap4 feedback mode, training mode, freeze the in-engine interneuron)
_ARM = {"frozen": ("reservoir", "bdsp", False),
        "fixed_fa": ("fixed_fa", "bdsp", False),
        "micro_inengine": ("micro_inengine", "bdsp", False),
        "transport_ceiling": ("transport_ceiling", "bdsp", False),
        "micro_inengine_lesion": ("micro_inengine", "apical_lesion", False),
        "micro_inengine_freeze_spi": ("micro_inengine", "bdsp", True)}
# args that change a shard's result (the checkpoint fingerprint); run-layout args (seeds/arms/out) are excluded.
_FP_KEYS = ("hidden", "pool_k", "n_hidden_layers", "settle_steps", "credit_steps", "epochs", "batch",
            "train_subsample", "lr", "bdsp_w_max", "beta", "p0", "in_current_pA", "in_bias_pA", "apical_gain_pA",
            "tonic_h_pA", "tonic_o_pA", "graded_credit", "wpi_init", "wpi_lr", "kp_lr", "kp_decay",
            "read_window", "read_gain", "isi_steps", "eval_frozen", "spi_silence", "n_super", "n_members",
            "held_per_super", "n_prop", "member_id_dim", "n_obs", "noise", "oracle_epochs", "oracle_lr",
            "oracle_batch", "decode_ridge", "read_quantity", "no_structural", "ff_w_init", "propagation_strength",
            "no_ff_stp", "pbar_alpha", "hidden_lr_gain")


# ============================================================================================================
class Gap4ReadoutNet(Gap4InEngineNet):
    """Gap4InEngineNet + the read-regime levers. Every lever at its legacy value => the parent's code path."""

    def __init__(self, n_in, hidden, k, seed=0, feedback="fixed", read_window=0, read_gain=1.0, isi_steps=0,
                 eval_frozen=False, spi_silence=False, read_quantity="event", no_structural=False,
                 propagation_strength=None, no_ff_stp=False, hidden_lr_gain=None, **kw):
        super().__init__(n_in, hidden, k, seed=seed, feedback=feedback, **kw)
        # AMENDMENT 2 (operating point). The dev transmission scan (diag_transmit_scan_*_s7.json) shows that with the
        # default Tsodyks-Markram short-term depression ON, NO feedforward gain (ff_w_init 4->40, propagation
        # 0.05->0.5) and no tonic level changes hidden-layer rates: the explicit feedforward pathway transmits
        # ~nothing at these presynaptic rates, so no arm's learning can reach the output read. no_ff_stp bypasses the
        # STP factor on the explicit FEEDFORWARD synapses only (cp_stp_disabled_mask; the recurrent background keeps
        # it). Requires no_structural (elimination compaction would misalign the per-synapse mask). Defaults = legacy.
        if propagation_strength is not None:
            self.cfg.propagation_strength = float(propagation_strength)
        self.no_ff_stp = bool(no_ff_stp)
        if self.no_ff_stp:
            if not no_structural:
                raise ValueError("--no-ff-stp requires --no-structural-plasticity (mask alignment)")
            from sim.backend import to_host
            coo = self.br._get_cached_coo()
            row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
            ff = np.zeros(row.shape[0], dtype=bool)
            for pre, post in self._ff_edges:
                ff |= ((row >= pre[0]) & (row <= pre[-1]) & (col >= post[0]) & (col <= post[-1]))
            self.br.cp_stp_disabled_mask = self._xp.asarray(ff)
            self.n_ff_stp_disabled = int(ff.sum())
        # AMENDMENT 4 (per-layer step size, arc meta-lesson #1): the committed per-synapse plasticity gain
        # cp_plasticity_rate_gain in [0,1] scales the BDSP step on every synapse whose POSTSYNAPTIC neuron is in a
        # hidden layer; the output pathway keeps the full lr. None = legacy (no gain array).
        self.hidden_lr_gain = None if hidden_lr_gain is None else float(hidden_lr_gain)
        if self.hidden_lr_gain is not None:
            if not no_structural:
                raise ValueError("--hidden-lr-gain requires --no-structural-plasticity (mask alignment)")
            from sim.backend import to_host
            coo = self.br._get_cached_coo()
            col = np.asarray(to_host(coo.col))
            g = np.ones(col.shape[0], dtype=np.float32)
            for li in range(1, len(self.sizes) - 1):
                sl = self.slices[li]
                g[(col >= sl.start) & (col < sl.stop)] = self.hidden_lr_gain
            self.br.cp_plasticity_rate_gain = self._xp.asarray(g)
        # read_quantity: "event" = cp_bdsp_E (isolated / first-of-burst spikes; legacy). "spikes" = EVERY somatic
        # spike (events + burst spikes), per step. AMENDMENT 1: the dev diagnostic diag_eread_monotonic_s7.json shows
        # E is NON-MONOTONIC in drive and the output layer sits at its peak, so LTP LOWERS the event read.
        self.read_quantity = str(read_quantity)
        if self.read_quantity not in ("event", "spikes"):
            raise ValueError("read_quantity must be event|spikes")
        if bool(no_structural):
            # AMENDMENT 1: the default-on synapse ELIMINATION treats every weight < 0.05 as weak, i.e. every
            # negative signed feedforward weight, and zeroes it at 5e-7/step (a companion process not built for
            # signed BDSP weights). Off => those weights persist. Default False => legacy.
            self.cfg.enable_structural_plasticity = False
        self.no_structural = bool(no_structural)
        self.read_window = int(read_window)
        self.read_gain = float(read_gain)
        self.isi_steps = int(isi_steps)
        self.eval_frozen = bool(eval_frozen)
        self.spi_silence = bool(spi_silence)
        self.n_steps = 0
        self._lesion_int_silenced = False   # set only inside a micro_inengine apical_lesion credit phase (AMENDMENT 5)
        self._track_apical = False          # instrument: running max |v_apical - rest| over hidden neurons
        self.apical_max_dev_mV = 0.0
        _step = self.br._run_one_simulation_step

        def _counted():
            self.n_steps += 1
            if self._lesion_int_silenced:
                # MATCHED LESION: the engine forms int_drive from cp_spi_int_rate on every step; a zero rate gives a
                # zero cancellation, so the lesioned top hidden apical receives nothing at all.
                if self.br.cp_spi_int_rate is not None:
                    self.br.cp_spi_int_rate[...] = 0.0
                if self.br.cp_bdsp_int_drive is not None:
                    self.br.cp_bdsp_int_drive[...] = 0.0
            out = _step()
            if self._track_apical and self.br.cp_v_apical is not None:
                self._apical_track_step()
            return out
        self.br._run_one_simulation_step = _counted     # instance-level wrapper: counts steps, same call

    # ---- instrument: is the hidden apical at rest? (the TERMS.md 'lesion' condition, checked while it runs) ----
    def _apical_track_step(self):
        xp = self._xp
        er = float(getattr(self.cfg, "apical_E_rest", -65.0))
        hid = slice(self.slices[1].start, self.slices[len(self.sizes) - 2].stop)
        m = xp.max(xp.abs(self.br.cp_v_apical[hid] - er))
        self._apical_max_dev_dev = m if getattr(self, "_apical_max_dev_dev", None) is None \
            else xp.maximum(self._apical_max_dev_dev, m)

    def apical_max_deviation_mV(self):
        from sim.backend import to_host
        v = getattr(self, "_apical_max_dev_dev", None)
        return 0.0 if v is None else float(np.asarray(to_host(v)))

    # ---- instrument: how many feedforward synapses sit at the hard BDSP bound (review issue: the clamp) ----
    def bound_census(self):
        """Per feedforward pathway: the fraction of synapses at (or, before the kernel first clamps them, beyond)
        +w_max / -w_max, mean |w| and max |w|. Reads only; no RNG, no state change (verified: shard results equal the
        pre-census runner's on a tiny run)."""
        from sim.backend import to_host
        coo = self.br._get_cached_coo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        w = np.asarray(to_host(self.br.cp_connections.data)).astype(np.float64)
        wmax, wmin = float(self.cfg.bdsp_w_max), float(self.cfg.bdsp_w_min)
        tol = 1e-4 * max(abs(wmax), abs(wmin), 1.0)
        out = {"w_max": wmax, "w_min": wmin, "pathways": []}
        tot_n = tot_hi = tot_lo = 0
        for li, (pre, post) in enumerate(self._ff_edges):
            m = (row >= pre[0]) & (row <= pre[-1]) & (col >= post[0]) & (col <= post[-1])
            wl = w[m]
            hi = int(np.sum(wl >= wmax - tol)); lo = int(np.sum(wl <= wmin + tol)); n = int(wl.size)
            tot_n += n; tot_hi += hi; tot_lo += lo
            out["pathways"].append({"pathway": f"ff_{li}", "post_is_hidden": bool(li + 1 < len(self.sizes) - 1),
                                    "n": n, "frac_at_or_above_w_max": hi / max(1, n), "frac_at_or_below_w_min": lo / max(1, n),
                                    "mean_abs_w": float(np.abs(wl).mean()) if n else None,
                                    "max_abs_w": float(np.abs(wl).max()) if n else None})
        out["frac_at_bound_all_ff"] = (tot_hi + tot_lo) / max(1, tot_n)
        return out

    # ---- LONGER read: time-average of the pooled event rate over the last W settle steps ----
    def _forward_spiking(self, feat_row, reset_rates=True):
        if self.read_window <= 0 and self.read_quantity == "event":
            return super()._forward_spiking(feat_row, reset_rates)
        from sim.backend import to_host
        xp = self._xp; n = self.n_total
        if reset_rates and self.br.cp_bdsp_E is not None:        # identical reset to the parent
            self.br.cp_bdsp_E[...] = 0.0
            self.br.cp_bdsp_B[...] = 0.0
            self.br.cp_bdsp_last_spike_step = xp.full(n, -1000000, dtype=xp.int64)
        drive = self._base_drive()
        f = np.asarray(feat_row, dtype=np.float32)
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * f, 0.0, 1600.0)
        drive[self.slices[0]] = self._broadcast(in_cur, 0).astype(np.float32)
        self.br.cp_external_input_current = xp.asarray(drive)
        if self.br.cp_bdsp_apical_drive is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        W = max(1, min(self.read_window if self.read_window > 0 else 1, self.settle_steps))
        acc = None
        for s in range(self.settle_steps):
            self.br._run_one_simulation_step()
            if s >= self.settle_steps - W:
                if self.read_quantity == "spikes":
                    # fired THIS step <=> the BDSP block stamped this step as the neuron's last spike
                    cur = (self.br.cp_bdsp_last_spike_step == self.br._bdsp_step_counter).astype(xp.float32)
                else:
                    cur = self.br.cp_bdsp_E
                acc = cur.copy() if acc is None else acc + cur
        E = np.asarray(to_host(acc)).astype(np.float64) / float(W)
        return [self._pool(E[self.slices[li]], li) for li in range(len(self.sizes))]

    @contextlib.contextmanager
    def _gain(self):
        """STRONGER read: softmax(g * rates) inside the parents' credit pass (both modules' _softmax)."""
        if self.read_gain == 1.0:
            yield
            return
        o_ob, o_ie, g = _ob_mod._softmax, _ie_mod._softmax, self.read_gain
        _ob_mod._softmax = lambda z: o_ob(g * np.asarray(z))
        _ie_mod._softmax = lambda z: o_ie(g * np.asarray(z))
        try:
            yield
        finally:
            _ob_mod._softmax, _ie_mod._softmax = o_ob, o_ie

    def _train_one(self, feat_row, y, mode):
        # AMENDMENT 5: a matched apical lesion in the in-engine arm silences the interneuron too (see LESION_KIND).
        self._lesion_int_silenced = (mode == "apical_lesion" and self.feedback == "micro_inengine")
        try:
            with self._gain():
                super()._train_one(feat_row, y, mode)
        finally:
            self._lesion_int_silenced = False
        xp = self._xp
        if self.spi_silence and self.br.cp_spi_int_rate is not None:
            self.br.cp_spi_int_rate = xp.zeros_like(self.br.cp_spi_int_rate)   # interneuron silent outside credit
            if self.br.cp_bdsp_int_drive is not None:
                self.br.cp_bdsp_int_drive[...] = 0.0
        if self.isi_steps > 0:
            drive = self._base_drive()                   # tonic background only; input slice gets NO stimulus
            drive[self.slices[0]] = 0.0
            self.br.cp_external_input_current = xp.asarray(drive)
            if self.br.cp_bdsp_apical_drive is not None:
                self.br.cp_bdsp_apical_drive[...] = 0.0
            for _ in range(self.isi_steps):
                self.br._run_one_simulation_step()

    @contextlib.contextmanager
    def frozen_reads(self):
        """--eval-frozen: no BDSP weight change while the evaluation reads run (restored afterwards)."""
        if not self.eval_frozen:
            yield
            return
        lr0 = self.cfg.bdsp_learning_rate
        sp0 = self.cfg.enable_structural_plasticity     # AMENDMENT 1: elimination also edits weights during reads
        self.cfg.bdsp_learning_rate = 0.0
        self.cfg.enable_structural_plasticity = False
        try:
            yield
        finally:
            self.cfg.bdsp_learning_rate = lr0
            self.cfg.enable_structural_plasticity = sp0

    def no_weight_transport(self):
        return super().no_weight_transport()


# ============================================================================================================
def _fingerprint(args):
    d = {k: getattr(args, k) for k in _FP_KEYS}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _task_seed(seed, r):
    return int(seed) if r == 0 else int(seed) + 10007 * int(r)


def _build(arm, n_in, k, args, seed):
    fb, _mode, freeze = _ARM[arm]
    net = Gap4ReadoutNet(
        n_in, args.hidden, k, seed=seed, feedback=fb,
        n_hidden_layers=args.n_hidden_layers, pool_k=args.pool_k,
        settle_steps=args.settle_steps, credit_steps=args.credit_steps, lr=args.lr,
        in_current_pA=args.in_current_pA, in_bias_pA=args.in_bias_pA, apical_gain_pA=args.apical_gain_pA,
        tonic_h_pA=args.tonic_h_pA, tonic_o_pA=args.tonic_o_pA, beta=args.beta, p0=args.p0,
        graded_credit=args.graded_credit, wpi_plastic=True, wpi_init=args.wpi_init, wpi_lr=args.wpi_lr,
        kp_lr=args.kp_lr, kp_decay=args.kp_decay,
        read_window=args.read_window, read_gain=args.read_gain, isi_steps=args.isi_steps,
        eval_frozen=args.eval_frozen, spi_silence=args.spi_silence,
        read_quantity=args.read_quantity, no_structural=args.no_structural,
        ff_w_init=args.ff_w_init, propagation_strength=args.propagation_strength, no_ff_stp=args.no_ff_stp,
        pbar_alpha=args.pbar_alpha, hidden_lr_gain=args.hidden_lr_gain)
    net.cfg.bdsp_w_max = float(args.bdsp_w_max)
    net.cfg.bdsp_w_min = -float(args.bdsp_w_max)
    net._spi_frozen = bool(freeze)
    assert int(net.cfg.seed) == int(seed), "cfg.seed must be the substrate seed"
    return net


def _thr_hash(net):
    from sim.backend import to_host
    thr = getattr(net.br, "cp_neuron_firing_thresholds", None)
    return None if thr is None else hashlib.md5(np.asarray(to_host(thr)).tobytes()).hexdigest()[:16]


def _w_hash(net):
    from sim.backend import to_host
    return hashlib.md5(np.asarray(to_host(net.br.cp_connections.data)).tobytes()).hexdigest()[:16]


def _ridge_decode(Htr, ytr, Hte, yte, k, lam):
    """INSTRUMENT: is the class linearly present in the pooled top-hidden read? (never a pathway)"""
    Htr = np.asarray(Htr, float); Hte = np.asarray(Hte, float)
    mu, sd = Htr.mean(0), Htr.std(0) + 1e-9
    A = np.c_[(Htr - mu) / sd, np.ones(len(Htr))]
    B = np.c_[(Hte - mu) / sd, np.ones(len(Hte))]
    T = np.eye(k)[np.asarray(ytr, int)]
    Wr = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ T)
    return (float(np.mean(np.argmax(A @ Wr, 1) == np.asarray(ytr))),
            float(np.mean(np.argmax(B @ Wr, 1) == np.asarray(yte))))


def _binom_p(acc, n, chance):
    from scipy.stats import binomtest
    kk = int(round(float(acc) * n))
    return float(binomtest(kk, int(n), float(chance), alternative="greater").pvalue)


def _task(seed, r, args):
    from sim.dendritic_mlp import DendriticMLP
    from sim.backend import to_host
    ts = _task_seed(seed, r)
    tk = dict(n_super=args.n_super, n_members=args.n_members, held_per_super=args.held_per_super,
              n_prop=args.n_prop, member_id_dim=args.member_id_dim, n_obs=args.n_obs, noise=args.noise)
    (Xtr, ytr, _l1), (Xte, yte, _l2), meta, idx = make_task_semantic_inheritance(ts, **tk)
    k = int(meta["k_classes"]); inh = np.asarray(idx["inh_idx"])
    chance = float(max(np.mean(yte[inh] == c) for c in np.unique(yte[inh])))
    onet = DendriticMLP([Xtr.shape[1], args.hidden, args.hidden, k], seed=ts)
    rr = np.random.default_rng(ts + 777)
    for _ in range(args.oracle_epochs):
        p = rr.permutation(len(ytr))
        for i in range(0, len(ytr), args.oracle_batch):
            b = p[i:i + args.oracle_batch]
            onet.train_step(Xtr[b], ytr[b], mode="oracle", lr=args.oracle_lr)
    _, olg = onet._forward(np.asarray(Xte[inh], float))
    oracle = float(np.mean(np.argmax(np.asarray(to_host(olg)), 1) == yte[inh]))
    Xb, yb = Xtr, ytr
    if args.train_subsample and len(Xtr) > args.train_subsample:
        keep = np.random.default_rng(ts * 13 + 1).permutation(len(Xtr))[:args.train_subsample]
        Xb, yb = Xtr[keep], ytr[keep]
    # AMENDMENT 5 (review fix): TRAINING chance is the majority-class rate of the training subsample, the prereg's own
    # chance definition applied to the training items. It is NOT 1/k: class 8 has no training items on this task, so
    # guessing "among nine classes" understates it (seed 7: r0 0.1825, r1 0.170, r2 0.1625).
    train_chance = float(np.bincount(np.asarray(yb, int), minlength=k).max() / len(yb))
    return dict(task_seed=ts, Xtr=Xb, ytr=yb, Xte=Xte, yte=yte, inh=inh, k=k, n_in=int(Xtr.shape[1]),
                chance=chance, oracle=oracle, n_inh=int(len(inh)), train_chance=train_chance)


def run_shard(seed, r, arm, args, T):
    t0 = time.time()
    net = _build(arm, T["n_in"], T["k"], args, seed)
    thr = _thr_hash(net)
    fb, mode, _fz = _ARM[arm]
    w0 = net.ff_weight_norm()
    census0 = net.bound_census()                              # reads only (review issue: the hard weight bound)
    from tools.lab import bound_check
    bound_ok = bound_check("bdsp_w_max", net.cfg.bdsp_w_max,
                           max(p["max_abs_w"] or 0.0 for p in census0["pathways"]), strict=False)
    net._track_apical = (mode == "apical_lesion")             # the TERMS.md 'lesion' check, measured while it runs
    rng = np.random.default_rng(T["task_seed"] + 777)
    if mode == "shufE":
        net._shuf_perm = np.random.default_rng(seed * 4099 + 11).permutation(net.k)
    ep_times = []
    for ep in range(args.epochs):
        te = time.time()
        perm = rng.permutation(len(T["Xtr"]))
        for i in range(0, len(T["Xtr"]), args.batch):
            b = perm[i:i + args.batch]
            net.train_step(T["Xtr"][b], T["ytr"][b], mode=mode)
        ep_times.append(time.time() - te)
        if ep == 0 or (ep + 1) % max(1, args.epochs // 5) == 0 or ep + 1 == args.epochs:
            eta = np.mean(ep_times) * (args.epochs - ep - 1)
            print(f"[gap4-tc][s{seed} r{r} {arm}] epoch {ep + 1}/{args.epochs} {ep_times[-1]:.0f}s "
                  f"(steps {net.n_steps}, {1e3 * (time.time() - t0) / max(1, net.n_steps):.2f} ms/step, "
                  f"ETA train {eta / 60:.1f} min)", flush=True)
    w1 = net.ff_weight_norm()
    census1 = net.bound_census()
    apical_dev = net.apical_max_deviation_mV() if net._track_apical else None
    net._track_apical = False
    t_train = time.time() - t0
    steps_train = net.n_steps
    with net.frozen_reads():
        wr0 = _w_hash(net)
        acts_te = net._forward_batch(T["Xte"][T["inh"]])      # held-out first (legacy order), then train
        acts_tr = net._forward_batch(T["Xtr"])
        wr1 = _w_hash(net)
    yte_inh = np.asarray(T["yte"])[T["inh"]]
    held = float(np.mean(np.argmax(acts_te[-1], 1) == yte_inh))
    train = float(np.mean(np.argmax(acts_tr[-1], 1) == np.asarray(T["ytr"])))
    dec_tr, dec_te = _ridge_decode(acts_tr[-2], T["ytr"], acts_te[-2], yte_inh, T["k"], args.decode_ridge)
    out_rate = float(np.mean(acts_te[-1]))
    pred_tr = np.argmax(acts_tr[-1], 1); pred_te = np.argmax(acts_te[-1], 1)
    hist = lambda v: np.bincount(np.asarray(v, int), minlength=T["k"]).tolist()
    try:
        from research.runners import _git_head
        git_sha, git_dirty = _git_head(full=True)
    except Exception:
        git_sha, git_dirty = "unknown", None
    n_tr = len(T["ytr"])
    res = {"seed": seed, "replicate": r, "arm": arm, "feedback": fb, "mode": mode,
           "git_sha": git_sha, "git_dirty": git_dirty,
           "task_seed": T["task_seed"], "chance": T["chance"], "oracle_heldout": T["oracle"], "n_inh": T["n_inh"],
           "inherit_heldout": held, "train_acc": train,
           "train_chance": T["train_chance"], "train_binom_p": _binom_p(train, n_tr, T["train_chance"]),
           "ff_l1_norm_start": float(w0), "ff_l1_norm_end": float(w1),
           "bound_census_start": census0, "bound_census_end": census1, "bound_check_at_build": bound_ok,
           "decode_h2_train": dec_tr, "decode_h2_heldout": dec_te,
           "mean_output_rate_heldout": out_rate,
           "pred_hist_train": hist(pred_tr), "true_hist_train": hist(T["ytr"]),
           "pred_hist_heldout": hist(pred_te), "true_hist_heldout": hist(yte_inh),
           "mean_output_read_by_class_train": [float(acts_tr[-1][np.asarray(T["ytr"]) == c, c].mean())
                                               if np.any(np.asarray(T["ytr"]) == c) else None for c in range(T["k"])],
           "mean_output_read_other_class_train": [float(acts_tr[-1][np.asarray(T["ytr"]) != c, c].mean())
                                                  for c in range(T["k"])],
           "read_quantity": args.read_quantity,
           "ff_weight_moved": float(abs(w1 - w0)),
           "ff_weight_moved_definition": "|L1 norm at end - L1 norm at start| over the feedforward weights "
                                         "(a change in L1 norm, not a total per-synapse change)",
           "eval_reads_left_weights_unchanged": bool(wr0 == wr1),
           "no_weight_transport": bool(net.no_weight_transport()),
           "ast_no_forward_W": bool(_ast_no_forward_W(Gap4ReadoutNet)),
           "thr_hash": thr, "n_steps_train": int(steps_train), "n_steps_total": int(net.n_steps),
           "train_seconds": round(t_train, 1), "elapsed_seconds": round(time.time() - t0, 1),
           "ms_per_step": round(1e3 * (time.time() - t0) / max(1, net.n_steps), 3),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "fingerprint": _fingerprint(args)}
    if arm.startswith("micro_inengine"):
        st = net.inengine_apical_silent_stats(T["Xte"][T["inh"]], yte_inh) if args.silent_stats else {}
        res["inengine_apical"] = {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv) for kk, vv in st.items()}
    if mode == "apical_lesion":
        res["lesion_kind"] = LESION_KIND
        res["lesion_hidden_apical_max_abs_dev_mV"] = apical_dev
        res["lesion_holds"] = bool(apical_dev is not None and apical_dev <= LESION_SILENT_TOL_MV)
    print(f"[gap4-tc][s{seed} r{r} {arm}] held-out {held:.3f} train {train:.3f} decode_h2 {dec_te:.3f} "
          f"(chance {T['chance']:.3f}, oracle {T['oracle']:.3f}) ff-moved {res['ff_weight_moved']:.1f} "
          f"nwt {res['no_weight_transport']} eval-reads-unchanged {res['eval_reads_left_weights_unchanged']} "
          f"| {res['elapsed_seconds']:.0f}s {res['ms_per_step']:.2f} ms/step rss {res['peak_rss_mb']:.0f} MB",
          flush=True)
    return res


def _ckpt_path(args, seed, r, arm):
    return Path(args.ckpt_dir) / f"s{seed}_r{r}_{arm}.json"


def _atomic_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".%d.tmp" % os.getpid())   # per-process: shards share one --out
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _load_shard(args, seed, r, arm):
    p = _ckpt_path(args, seed, r, arm)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    if d.get("fingerprint") != _fingerprint(args):
        return None
    if _ARM[arm][1] == "apical_lesion" and d.get("lesion_kind") != LESION_KIND:
        return None          # a pre-AMENDMENT-5 (unmatched) lesion shard is never resumed or aggregated
    return d


# ============================================================================================================
def aggregate(args):
    fp = _fingerprint(args)
    per_seed = {}
    for s in args.seeds:
        reps = []
        for r in args.replicates:
            sh = {a: _load_shard(args, s, r, a) for a in ALL_ARMS}
            sh = {a: v for a, v in sh.items() if v is not None}
            if not sh:
                continue
            any_sh = next(iter(sh.values()))
            chance, n = any_sh["chance"], any_sh["n_inh"]
            acc = {a: v["inherit_heldout"] for a, v in sh.items()}
            row = {"replicate": r, "task_seed": any_sh["task_seed"], "chance": chance, "n_inh": n,
                   "train_chance": any_sh.get("train_chance"),
                   "oracle": any_sh["oracle_heldout"], "arms_done": sorted(sh), "inherit_heldout": acc,
                   "train_acc": {a: v["train_acc"] for a, v in sh.items()},
                   "decode_h2_heldout": {a: v["decode_h2_heldout"] for a, v in sh.items()}}
            ce, fr = acc.get("transport_ceiling"), acc.get("frozen")
            ff, mi = acc.get("fixed_fa"), acc.get("micro_inengine")
            if ce is not None:
                row["ceiling_binom_p"] = _binom_p(ce, n, chance)
                row["ceiling_clears_chance"] = bool(row["ceiling_binom_p"] < 0.05)
            if ce is not None and fr is not None:
                row["headroom"] = float(ce - fr)
                ok = row["headroom"] >= 0.05
                row["deep_credit_share"] = {a: (float((acc[a] - fr) / row["headroom"]) if ok else None)
                                            for a in ("fixed_fa", "micro_inengine") if a in acc}
                if not ok:
                    row["deep_credit_share_note"] = "UNDEFINED: headroom < 0.05 (not a score of 0)"
                # AMENDMENT 5 rule 1: a replicate is INTERPRETABLE only if the ceiling clears chance AND has
                # headroom >= 0.05 over frozen (else deep_credit_share is UNDEFINED and no negative can be read).
                row["interpretable"] = bool(row.get("ceiling_clears_chance") and ok)
            if ff is not None and fr is not None:
                row["fa_wall"] = bool(ff <= fr + 0.02)
            if ff is not None and mi is not None:
                row["surpass"] = bool(mi > ff + 0.05)
            # anti-cheat ATTRIBUTION (prereg rule 4): how much of micro_inengine is NOT in each control.
            for ctl in ("micro_inengine_lesion", "micro_inengine_freeze_spi"):
                if mi is not None and ctl in acc:
                    row.setdefault("attribution", {})[ctl] = attributable_to(
                        f"s{s} r{r} micro_inengine vs {ctl}", float(mi), float(acc[ctl]))
            if "micro_inengine_lesion" in sh:
                # TERMS.md 'lesion': the cut must be verified to hold while it ran, or it is only an attempted lesion
                row["lesion_holds"] = bool(sh["micro_inengine_lesion"].get("lesion_holds"))
            reps.append(row)
        if not reps:
            continue
        n_ceil = sum(1 for x in reps if x.get("ceiling_clears_chance"))
        n_int = sum(1 for x in reps if x.get("interpretable"))
        n_fa = sum(1 for x in reps if x.get("fa_wall"))
        n_sur = sum(1 for x in reps if x.get("surpass"))
        complete = all(set(CORE_ARMS) <= set(x["arms_done"]) for x in reps) and len(reps) == len(args.replicates)
        mean = lambda a: (float(np.mean([x["inherit_heldout"][a] for x in reps if a in x["inherit_heldout"]]))
                          if any(a in x["inherit_heldout"] for x in reps) else None)
        m = {a: mean(a) for a in ALL_ARMS}
        seed_status = ("INCOMPLETE" if not complete else
                       ("UNDEFINED (interpretable on %d/%d replicates < 2; ceiling clears chance on %d, AMENDMENT 5 "
                        "also needs headroom >= 0.05)" % (n_int, len(reps), n_ceil)
                        if n_int < 2 else "DEFINED"))
        per_seed[str(s)] = {"replicates": reps, "n_replicates": len(reps), "complete": complete,
                            "n_ceiling_clears_chance": n_ceil, "n_interpretable": n_int,
                            "n_fa_wall": n_fa, "n_surpass": n_sur,
                            "mean_inherit_heldout": m,
                            "micro_minus_fixed": (None if m["micro_inengine"] is None or m["fixed_fa"] is None
                                                  else m["micro_inengine"] - m["fixed_fa"]),
                            "fa_wall_coverage": ("complete" if n_fa >= 3 else "incomplete (%d/3)" % n_fa),
                            "status": seed_status}
    six = sorted(EVAL_SEEDS)
    six_done = all(str(s) in per_seed and per_seed[str(s)]["complete"] for s in six)
    verdict = {"scope": "de-risk (%d seed%s)" % (len(per_seed), "" if len(per_seed) == 1 else "s"),
               "six_seed_rule_evaluable": six_done}
    if six_done:
        ps = [per_seed[str(s)] for s in six]
        # AMENDMENT 5 rule 1: the gate counts INTERPRETABLE replicates (ceiling clears chance AND headroom >= 0.05),
        # so a NO-GO can never be read off an instrument with no headroom.
        c1 = sum(1 for p in ps if p["n_interpretable"] >= 2) >= 5
        c2 = sum(1 for p in ps if (p["micro_minus_fixed"] or -1) > 0.05) >= 5
        verdict["interpretability_gate"] = c1
        verdict["surpass_gate"] = c2
        verdict["status"] = ("UNDEFINED (ceiling gate fails)" if not c1 else
                             ("NO-GO" if not c2 else "PENDING anti-cheats + deep_credit_share rule 3"))
    else:
        verdict["status"] = "no 6-seed verdict (seeds incomplete); per-seed status only"
    out = {"probe": "gap4_transport_ceiling_readout", "prereg": PREREG, "fingerprint": fp, "config": vars(args),
           "per_seed": per_seed, "verdict": verdict,
           "NOTE": "UNDEFINED is not a negative. decode_h2 is an instrument (host ridge), never a pathway."}
    _atomic_write(Path(args.out), out)
    for s, p in per_seed.items():
        print(f"[gap4-tc][agg s{s}] {p['status']} | ceiling clears {p['n_ceiling_clears_chance']}/{p['n_replicates']} "
              f"| interpretable {p['n_interpretable']}/{p['n_replicates']} "
              f"| n_fa_wall {p['n_fa_wall']} | surpass {p['n_surpass']} | means "
              + " ".join(f"{a}={v:.3f}" for a, v in p["mean_inherit_heldout"].items() if v is not None), flush=True)
    print(f"[gap4-tc] verdict: {verdict['status']} -> wrote {args.out}", flush=True)
    return out


def _committed_text(rel):
    """The COMMITTED content of `rel`: `git show HEAD:<rel>` in a git work tree (the file must be tracked; working-tree
    edits are ignored), or the file itself in a manifest-verified git-archive export (the pool's revision dirs, whose
    tree IS a committed revision). Returns (text, how) or (None, reason)."""
    import subprocess
    try:
        inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=_REPO, capture_output=True,
                                text=True, timeout=10).stdout.strip() == "true"
    except Exception:
        inside = False
    if inside:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=_REPO,
                                 capture_output=True, text=True, timeout=10).returncode == 0
        if not tracked:
            return None, "not tracked by git (an uncommitted file cannot register a config)"
        shown = subprocess.run(["git", "show", "HEAD:%s" % rel], cwd=_REPO, capture_output=True, text=True,
                               timeout=10)
        if shown.returncode != 0:
            return None, "not in HEAD (staged but not committed)"
        return shown.stdout, "git HEAD"
    try:
        from research.runners import _source_snapshot, verify_immutable_source_manifest
        snap = _source_snapshot()
        if snap.get("source_kind") == "git_archive" and verify_immutable_source_manifest(snap).get(
                "source_manifest_verified"):
            p = _REPO / rel
            return (p.read_text(encoding="utf-8"), "git archive %s" % snap.get("git_sha")) if p.exists() \
                else (None, "missing from the archive")
    except Exception as exc:
        return None, "no git tree and no verified archive (%s)" % exc
    return None, "no git tree and no verified archive"


def _eval_sections(text):
    """(heading, fingerprints, seeds) for every markdown section whose HEADING names an EVALUATION CONFIG amendment.
    Prose that merely mentions the phrase does not count; the registration keys must sit inside such a section."""
    out = []
    cur = None
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("#"):
            if cur is not None:
                out.append(cur)
            h = st.lstrip("#").strip()
            cur = (h, [], []) if (h.upper().startswith("AMENDMENT") and _EVAL_HEADING in h.upper()) else None
            continue
        if cur is None:
            continue
        low = st.strip("`*- ").lower()
        if low.startswith(_EVAL_FP_KEY):
            cur[1].append(low[len(_EVAL_FP_KEY):].strip().strip("`"))
        elif low.startswith(_EVAL_SEEDS_KEY):
            cur[2].extend(int(t) for t in low[len(_EVAL_SEEDS_KEY):].replace(",", " ").split() if t.isdigit())
    if cur is not None:
        out.append(cur)
    return out


def check_eval_amendment(seeds, fingerprint, amendment, read_committed=None):
    """The evaluation-seed guard. Returns (ok, reason). Fails unless `amendment` is COMMITTED and holds an
    'AMENDMENT ... EVALUATION CONFIG' section registering exactly this run's config fingerprint and every evaluation
    seed the run asks for. `read_committed` is injectable for the selftest."""
    ev = sorted(set(int(s) for s in seeds) & EVAL_SEEDS)
    if not ev:
        return True, "no evaluation seed requested"
    if not amendment:
        return False, "evaluation seeds %s need --prereg-amendment <committed file>" % ev
    text, how = (read_committed or _committed_text)(amendment)
    if text is None:
        return False, "amendment %s: %s" % (amendment, how)
    secs = _eval_sections(text)
    if not secs:
        return False, ("amendment %s has no '## AMENDMENT <n> ... %s' section (a prereg without one, or an amendment "
                       "that does not fix an evaluation config, cannot unlock evaluation seeds)" % (amendment,
                                                                                                  _EVAL_HEADING))
    for h, fps, sds in secs:
        if fingerprint in fps and set(ev) <= set(sds):
            return True, "registered in '%s' (%s)" % (h, how)
    return False, ("no %s section registers fingerprint %s for seeds %s (found: %s)"
                   % (_EVAL_HEADING, fingerprint, ev, [(h, fps, sds) for h, fps, sds in secs]))


def guard_selftest(args):
    """The evaluation-seed guard must FAIL in its failing directions and PASS only on a registered, committed config."""
    fp = _fingerprint(args)
    good = ("## AMENDMENT 9 -- EVALUATION CONFIG (test)\n\n%s `%s`\n%s 42 43 44 100 101 102\n"
            % (_EVAL_FP_KEY, fp, _EVAL_SEEDS_KEY))
    cases = [
        ("the real prereg at HEAD (no evaluation-config amendment exists)", [42], PREREG, None, False),
        ("an untracked file", [42], "research/findings/_no_such_amendment_%d.md" % os.getpid(), None, False),
        ("no amendment path", [42], None, None, False),
        ("evaluation section, other fingerprint", [42], "x.md",
         lambda _p: (good.replace(fp, "0" * 16), "fake"), False),
        ("evaluation section, seed not registered", [42, 7, 999], "x.md",
         lambda _p: (good.replace(" 42 ", " "), "fake"), False),
        ("phrase only in prose, keys outside a heading", [42], "x.md",
         lambda _p: ("EVALUATION CONFIG\n%s %s\n%s 42\n" % (_EVAL_FP_KEY, fp, _EVAL_SEEDS_KEY), "fake"), False),
        ("uncommitted content (reader says not in HEAD)", [42], "x.md",
         lambda _p: (None, "not in HEAD (staged but not committed)"), False),
        ("registered, committed, matching", [42, 43], "x.md", lambda _p: (good, "fake"), True),
        ("dev seed only needs nothing", [7], None, None, True),
    ]
    ok_all = True
    rows = []
    for name, seeds, am, reader, want in cases:
        got, why = check_eval_amendment(seeds, fp, am, read_committed=reader)
        ok = (got == want)
        ok_all = ok_all and ok
        rows.append({"case": name, "want": want, "got": got, "reason": why, "ok": ok})
        print(f"[gap4-tc-guard] {'ok ' if ok else 'BAD'} {name}: got {got} ({why})", flush=True)
    print(f"[gap4-tc-guard] PASS={ok_all}", flush=True)
    return (0 if ok_all else 1), rows


def run(args):
    if set(args.seeds) & EVAL_SEEDS and not args.aggregate_only:
        ok, why = check_eval_amendment(args.seeds, _fingerprint(args), args.prereg_amendment)
        if not ok:
            raise SystemExit("REFUSED: %s. The prereg fixes the evaluation config only by an amendment committed "
                             "first (AMENDMENT 5); get this run's fingerprint with --print-fingerprint." % why)
        print(f"[gap4-tc] evaluation seeds allowed: {why}", flush=True)
    if args.aggregate_only:
        return aggregate(args)
    tasks = {}
    for s in args.seeds:
        for r in args.replicates:
            for arm in args.arms:
                if _load_shard(args, s, r, arm) is not None:
                    print(f"[gap4-tc][s{s} r{r} {arm}] RESUME: shard already complete, skipped", flush=True)
                    continue
                key = (s, r)
                if key not in tasks:
                    tasks[key] = _task(s, r, args)
                    T = tasks[key]
                    print(f"[gap4-tc][s{s} r{r}] task_seed {T['task_seed']} n_in {T['n_in']} k {T['k']} "
                          f"n_train {len(T['ytr'])} n_inh {T['n_inh']} chance {T['chance']:.3f} "
                          f"oracle {T['oracle']:.3f}", flush=True)
                try:
                    res = run_shard(s, r, arm, args, tasks[key])
                except Exception:
                    traceback.print_exc()
                    raise
                _atomic_write(_ckpt_path(args, s, r, arm), res)
    return aggregate(args)


# ============================================================================================================
def identity_selftest(args):
    """With every lever at its legacy value, Gap4ReadoutNet must reproduce Gap4InEngineNet's weights and reads
    byte-for-byte (the data proof); with the levers on, they must differ (the levers are not inert)."""
    from sim.backend import to_host
    seed = 7
    T = make_task_semantic_inheritance(seed, n_super=8, n_members=4, held_per_super=1, n_prop=2,
                                       member_id_dim=3, n_obs=4, noise=0.02)
    (Xtr, ytr, _a), (Xte, yte, _b), meta, idx = T
    k = int(meta["k_classes"]); n_in = Xtr.shape[1]
    kw = dict(n_hidden_layers=2, pool_k=2, settle_steps=12, credit_steps=6, graded_credit=True,
              wpi_plastic=True, wpi_init="noisy")
    out = {"probe": "gap4_transport_ceiling_readout_IDENTITY_SELFTEST", "seed": seed, "arms": {}}
    ok_all = True
    for fb in ("reservoir", "fixed_fa", "micro_inengine", "transport_ceiling"):
        hashes = {}
        for tag, cls, extra in (("parent", Gap4InEngineNet, {}), ("legacy", Gap4ReadoutNet, {}),
                                ("levers_on", Gap4ReadoutNet, dict(read_window=8, read_gain=20.0, isi_steps=5,
                                                                   eval_frozen=True, spi_silence=True,
                                                                   read_quantity="spikes", no_structural=True))):
            net = cls(n_in, 6, k, seed=seed, feedback=fb, **kw, **extra)
            thr_build = _thr_hash(net)          # AT BUILD: thresholds adapt with activity, so compare before training
            for i in range(4):
                net._train_one(Xtr[i], int(ytr[i]), "bdsp")
            w = hashlib.md5(np.asarray(to_host(net.br.cp_connections.data)).tobytes()).hexdigest()
            acts = net._forward_batch(Xte[:3])
            a = hashlib.md5(np.concatenate([np.asarray(x, float).ravel() for x in acts]).tobytes()).hexdigest()
            hashes[tag] = {"weights_md5": w, "reads_md5": a, "thr": thr_build, "thr_after_training": _thr_hash(net)}
        same = (hashes["parent"]["weights_md5"] == hashes["legacy"]["weights_md5"]
                and hashes["parent"]["reads_md5"] == hashes["legacy"]["reads_md5"])
        moved = hashes["levers_on"]["weights_md5"] != hashes["legacy"]["weights_md5"]
        seed_ok = hashes["parent"]["thr"] == hashes["legacy"]["thr"] == hashes["levers_on"]["thr"]
        out["arms"][fb] = {"hashes": hashes, "legacy_identical_to_parent": same, "levers_on_differ": moved,
                           "same_thresholds_all_builds": seed_ok}
        ok_all = ok_all and same and moved and seed_ok
        print(f"[gap4-tc-identity] {fb:<18} legacy==parent {same} | levers-on differ {moved} | "
              f"same cfg.seed thresholds {seed_ok}", flush=True)
    out["IDENTITY_SELFTEST_PASS"] = bool(ok_all)
    _atomic_write(Path(args.out), out)
    print(f"[gap4-tc-identity] PASS={ok_all} -> {args.out}", flush=True)
    return 0 if ok_all else 1


def lesion_selftest(args):
    """AMENDMENT 5: the matched apical lesion must keep every hidden apical at rest during training, and the check must
    be able to fail: the parent's unmatched lesion (Gap4InEngineNet) and the intact micro_inengine arm must read > 0."""
    seed = 7
    (Xtr, ytr, _a), _te, meta, _idx = make_task_semantic_inheritance(seed, n_super=8, n_members=4, held_per_super=1,
                                                                      n_prop=2, member_id_dim=3, n_obs=4, noise=0.02)
    k = int(meta["k_classes"]); n_in = Xtr.shape[1]
    kw = dict(n_hidden_layers=2, pool_k=2, settle_steps=12, credit_steps=6, graded_credit=True,
              wpi_plastic=True, wpi_init="noisy")

    def measure(cls, fb, mode):
        net = cls(n_in, 6, k, seed=seed, feedback=fb, **kw)
        er = float(getattr(net.cfg, "apical_E_rest", -65.0))
        hid = slice(net.slices[1].start, net.slices[len(net.sizes) - 2].stop)
        box = {"m": 0.0}
        step = net.br._run_one_simulation_step

        def tracked():
            out = step()
            if net.br.cp_v_apical is not None:
                from sim.backend import to_host
                box["m"] = max(box["m"], float(np.max(np.abs(np.asarray(to_host(net.br.cp_v_apical))[hid] - er))))
            return out
        net.br._run_one_simulation_step = tracked
        for i in range(6):
            net._train_one(Xtr[i], int(ytr[i]), mode)
        return box["m"]
    cases = [("matched lesion (Gap4ReadoutNet micro_inengine apical_lesion)", Gap4ReadoutNet, "micro_inengine",
              "apical_lesion", "zero"),
             ("fixed_fa apical_lesion (reference, no interneuron)", Gap4ReadoutNet, "fixed_fa", "apical_lesion", "zero"),
             ("UNMATCHED parent lesion (Gap4InEngineNet micro_inengine apical_lesion)", Gap4InEngineNet,
              "micro_inengine", "apical_lesion", "positive"),
             ("intact micro_inengine (bdsp)", Gap4ReadoutNet, "micro_inengine", "bdsp", "positive")]
    rows = []; ok_all = True
    for name, cls, fb, mode, want in cases:
        m = measure(cls, fb, mode)
        ok = (m <= LESION_SILENT_TOL_MV) if want == "zero" else (m > 1.0)
        ok_all = ok_all and ok
        rows.append({"case": name, "hidden_apical_max_abs_dev_mV": m, "want": want, "ok": ok})
        print(f"[gap4-tc-lesion] {'ok ' if ok else 'BAD'} {name}: max |v_apical - rest| = {m:.4f} mV (want {want})",
              flush=True)
    out = {"probe": "gap4_tc_matched_lesion_SELFTEST", "seed": seed, "lesion_kind": LESION_KIND, "cases": rows,
           "LESION_SELFTEST_PASS": bool(ok_all)}
    _atomic_write(Path(args.out), out)
    print(f"[gap4-tc-lesion] PASS={ok_all} -> {args.out}", flush=True)
    return 0 if ok_all else 1


def select_calibration(paths, out):
    """Apply the PRE-REGISTERED selection rule mechanically to the seed-7 calibration artifacts."""
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        cfg = d["config"]; s = d["per_seed"].get("7")
        if not s:
            continue
        rep = s["replicates"][0]
        acc = rep["inherit_heldout"]
        ce, fr = acc.get("transport_ceiling"), acc.get("frozen")
        steps = cfg["settle_steps"] + cfg["credit_steps"] + cfg["isi_steps"]
        # AMENDMENTS 3-4 registered the cost as epochs x steps per example (review fix: the code used steps only).
        cost = int(cfg["epochs"]) * steps
        rows.append({"artifact": p, "label": cfg.get("label"), "settle": cfg["settle_steps"],
                     "read_window": cfg["read_window"], "read_gain": cfg["read_gain"], "isi": cfg["isi_steps"],
                     "epochs": int(cfg["epochs"]), "steps_per_example": steps, "cost_epochs_x_steps": cost,
                     "ceiling": ce, "frozen": fr,
                     "ceiling_train": rep["train_acc"].get("transport_ceiling"),
                     "frozen_train": rep["train_acc"].get("frozen"),
                     "decode_h2_frozen": rep["decode_h2_heldout"].get("frozen"),
                     "ceiling_binom_p": rep.get("ceiling_binom_p"), "headroom": rep.get("headroom"),
                     "chance": rep["chance"], "oracle": rep["oracle"]})
    cand = [r for r in rows if r["label"] != "C0" and r["ceiling_binom_p"] is not None
            and r["ceiling_binom_p"] < 0.05 and (r["headroom"] or -1) >= 0.05]
    chosen = None
    if cand:
        best = max(r["headroom"] for r in cand)
        near = [r for r in cand if r["headroom"] >= best - 0.02]
        chosen = min(near, key=lambda r: (r["cost_epochs_x_steps"], -r["headroom"]))
    res = {"rule": "prereg: ceiling binom p<0.05 AND headroom>=0.05; max headroom; within 0.02 -> fewest total "
                   "training steps (epochs x steps per example, AMENDMENTS 3-4)",
           "rows": rows, "qualifying": [r["label"] for r in cand], "chosen": chosen,
           "status": ("SELECTED " + chosen["label"]) if chosen else "NONE QUALIFIES -> instrument UNDEFINED at dev"}
    _atomic_write(Path(out), res)
    for r in rows:
        print(f"[gap4-tc-calib] {r['label']}: ceiling {r['ceiling']} (p={r['ceiling_binom_p']}) frozen {r['frozen']} "
              f"headroom {r['headroom']} | train ceil {r['ceiling_train']} frozen {r['frozen_train']} | "
              f"decode_h2(frozen) {r['decode_h2_frozen']} | steps/ex {r['steps_per_example']} "
              f"cost {r['cost_epochs_x_steps']}", flush=True)
    print(f"[gap4-tc-calib] {res['status']} -> {out}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description="gap#4 transport-ceiling readout lever (prereg: %s)" % PREREG)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=list(CORE_ARMS), choices=ALL_ARMS)
    ap.add_argument("--label", default="")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=16)
    ap.add_argument("--n-hidden-layers", dest="n_hidden_layers", type=int, default=2)
    ap.add_argument("--settle-steps", dest="settle_steps", type=int, default=40)
    ap.add_argument("--credit-steps", dest="credit_steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--train-subsample", dest="train_subsample", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bdsp-w-max", dest="bdsp_w_max", type=float, default=6.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--in-current-pA", dest="in_current_pA", type=float, default=520.0)
    ap.add_argument("--in-bias-pA", dest="in_bias_pA", type=float, default=260.0)
    ap.add_argument("--apical-gain-pA", dest="apical_gain_pA", type=float, default=900.0)
    ap.add_argument("--tonic-h-pA", dest="tonic_h_pA", type=float, default=450.0)
    ap.add_argument("--tonic-o-pA", dest="tonic_o_pA", type=float, default=500.0)
    ap.add_argument("--graded-credit", dest="graded_credit", action="store_true", default=True)
    ap.add_argument("--no-graded-credit", dest="graded_credit", action="store_false")
    ap.add_argument("--wpi-init", dest="wpi_init", default="noisy", choices=["noisy", "fixedpoint"])
    ap.add_argument("--wpi-lr", dest="wpi_lr", type=float, default=0.2)
    ap.add_argument("--kp-lr", dest="kp_lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", dest="kp_decay", type=float, default=1e-4)
    # --- the read-regime levers (legacy defaults) ---
    ap.add_argument("--read-window", dest="read_window", type=int, default=0)
    ap.add_argument("--read-gain", dest="read_gain", type=float, default=1.0)
    ap.add_argument("--isi-steps", dest="isi_steps", type=int, default=0)
    ap.add_argument("--eval-frozen", dest="eval_frozen", action="store_true")
    ap.add_argument("--spi-silence-outside-credit", dest="spi_silence", action="store_true")
    ap.add_argument("--read-quantity", dest="read_quantity", default="event", choices=["event", "spikes"])
    ap.add_argument("--no-structural-plasticity", dest="no_structural", action="store_true")
    ap.add_argument("--ff-w-init", dest="ff_w_init", type=float, default=4.0)
    ap.add_argument("--propagation-strength", dest="propagation_strength", type=float, default=None)
    ap.add_argument("--no-ff-stp", dest="no_ff_stp", action="store_true")
    # AMENDMENT 3: the burst-probability baseline Pbar is an EMA (alpha 0.05/step) that returns to p0 after every
    # teaching transient, so the time-integral of (P - Pbar) is ~0 per presentation; alpha 0 = a PRESET baseline at
    # p0 (the BurstCCN preset-baseline form, already a parameter of OnBridgeBDSPNet). Default 0.05 = legacy.
    ap.add_argument("--pbar-alpha", dest="pbar_alpha", type=float, default=0.05)
    ap.add_argument("--hidden-lr-gain", dest="hidden_lr_gain", type=float, default=None)
    ap.add_argument("--silent-stats", dest="silent_stats", action="store_true")
    ap.add_argument("--decode-ridge", dest="decode_ridge", type=float, default=1.0)
    # --- task (the 2026-09-15 task) ---
    ap.add_argument("--n-super", dest="n_super", type=int, default=24)
    ap.add_argument("--n-members", dest="n_members", type=int, default=8)
    ap.add_argument("--held-per-super", dest="held_per_super", type=int, default=3)
    ap.add_argument("--n-prop", dest="n_prop", type=int, default=3)
    ap.add_argument("--member-id-dim", dest="member_id_dim", type=int, default=3)
    ap.add_argument("--n-obs", dest="n_obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--oracle-epochs", dest="oracle_epochs", type=int, default=250)
    ap.add_argument("--oracle-lr", dest="oracle_lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", dest="oracle_batch", type=int, default=128)
    # --- layout ---
    ap.add_argument("--out", default=str(RAW / "run.json"))
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default=None)
    ap.add_argument("--aggregate-only", dest="aggregate_only", action="store_true")
    ap.add_argument("--prereg-amendment", dest="prereg_amendment", default=None)
    ap.add_argument("--identity-selftest", dest="identity_selftest", action="store_true")
    ap.add_argument("--guard-selftest", dest="guard_selftest", action="store_true",
                    help="show the evaluation-seed guard fails in its failing directions (AMENDMENT 5)")
    ap.add_argument("--lesion-selftest", dest="lesion_selftest", action="store_true",
                    help="show the matched apical lesion keeps the hidden apical at rest (AMENDMENT 5)")
    ap.add_argument("--print-fingerprint", dest="print_fingerprint", action="store_true",
                    help="print this flag set's config fingerprint (to register in an EVALUATION CONFIG amendment)")
    ap.add_argument("--select-calibration", dest="select_calibration", nargs="+", default=None)
    a = ap.parse_args()
    if a.ckpt_dir is None:
        a.ckpt_dir = str(Path(a.out).with_suffix("")) + "_ckpt"
    if a.print_fingerprint:
        print(_fingerprint(a))
        return 0
    if a.guard_selftest:
        rc, rows = guard_selftest(a)
        _atomic_write(Path(a.out), {"probe": "gap4_tc_eval_seed_guard_SELFTEST", "fingerprint": _fingerprint(a),
                                    "cases": rows, "GUARD_SELFTEST_PASS": rc == 0})
        return rc
    if a.lesion_selftest:
        return lesion_selftest(a)
    if a.identity_selftest:
        return identity_selftest(a)
    if a.select_calibration:
        select_calibration(a.select_calibration, a.out)
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
