"""ON-BRIDGE deep-credit DECORRELATION forward-only probe -- the DIRECT test of the a0 read-SNR-wall root cause + fix.

WHY THIS RUN (the a0 root cause, already diagnosed -- do NOT re-diagnose): the on-bridge spiking deep-credit net
(`OnBridgeBDSPNet` in `_semantic_inheritance_onbridge_spiking_derisk`) has a FLAT read-SNR across pool size K (K=1/8/16
all ~0.28) -- pooling K neurons per logical unit gives NO ~sqrt(K) variance reduction. ROOT CAUSE (from reading the
substrate): each neuron is driven by a DETERMINISTIC constant tonic current (tonic_h/tonic_o, identical per slice via
`_base_drive`) so the K neurons of a unit fire in near-LOCKSTEP (highly CORRELATED) -> almost no INDEPENDENT noise to
average -> pooling can't help. The biology-grounded FIX (Destexhe-Rudolph high-conductance state; cortical neurons get
INDEPENDENT background bombardment -> decorrelated firing -> population averaging works): give each neuron INDEPENDENT
background noise so the pool DECORRELATES and the ~sqrt(K) pooling gain is restored.

WHAT THIS RUNNER IS (reuse-by-import; FORWARD-ONLY; NO training; NO `sim/` edit): a THIN forward-only driver around the
committed `OnBridgeBDSPNet`. The read-SNR + pairwise correlation + pooling gain are FORWARD properties -- a fresh/untrained
net is sufficient (no weight learning). The ONE new knob is `--noise-pA`: INDEPENDENT per-neuron background noise, realized
by the bridge's OWN Ornstein-Uhlenbeck process (`cfg.enable_ou_process` + `cfg.ou_std_current_pA`, a config flag on the
net's own bridge, NOT a `sim/` edit). `cp.random.randn(n_neurons)` per step (bridge.py) => each neuron gets an INDEPENDENT
OU trajectory (Destexhe-Rudolph). `--noise-pA 0` FORCES OU OFF (`enable_ou_process=False`) = the deterministic constant
drive = the correlated / lockstep baseline the a0 diagnosis describes; `--noise-pA >0` = independent OU noise at that
sigma.

  SUBSTRATE NOTE (read, not assumed): `CoreSimConfig()` DEFAULTS `enable_ou_process=True, ou_std_current_pA=100.0`, and
  `OnBridgeBDSPNet.__init__` builds its bridge from a bare `CoreSimConfig()` WITHOUT disabling OU -> the committed net as
  built ALREADY carries OU sigma=100. So `--noise-pA 100` reproduces the ACTUAL committed default; `--noise-pA 0` (OU off)
  is the deterministic baseline the a0 diagnosis assumed the committed net had. This probe reports BOTH -- an honest read
  of what the substrate actually does.

THE FORWARD-ONLY MEASUREMENTS (for ~30-50 distinct inputs, per pool size K in {1,8,16} x per --noise-pA level):
  (A) PAIRWISE WITHIN-UNIT CORRELATION: for each logical unit, the mean pairwise correlation of its K neurons' `cp_bdsp_E`
      across the input set (the ACROSS-INPUT corr) + across REPEATS of a FIXED input (the pure-NOISE corr). Expected:
      HIGH near 1 at noise=0 (lockstep); DROPPING as noise rises (decorrelation). The DIRECT decorrelation check.
  (B) READ-SNR / POOLING GAIN: (B1) corr(pooled E, clean soma-rate reference) per hidden layer as a function of K -- RISES
      with K only when the pool is decorrelated. (B2) the noise CV of the POOLED event rate vs the single-neuron noise CV
      over repeats of a FIXED input -> the empirical pooling gain single_CV/pooled_CV, which -> ~sqrt(K) only when the
      pool's noise is INDEPENDENT (~1 when fully correlated). The direct ~sqrt(K) test.

GATE / verdict: (a) at noise=0, pairwise-corr HIGH + read-SNR flat across K + pooling gain ~1 (reproduces the correlated-
pool root cause); (b) at noise>0, pairwise-corr DROPS + read-SNR rises with K + pooling gain -> ~sqrt(K) (the decorrelation
fix WORKS -> population coding becomes viable => GO-candidate for the controller to test end-to-end training). If noise
does NOT decorrelate (pool stays correlated even with noise, or CV/gain does not move) => an honest negative that reframes
the boundary deeper (the residual is not read-variance).

HONEST SCOPE: BUILDER 1-seed CPU (numpy) FORWARD-ONLY probe -- a few minutes. NO training loop, NO multi-seed sweep (the
controller runs any follow-up end-to-end-training GO test). Held-out is irrelevant here (forward properties). NO `sim/`
edit (all reuse-by-import; the `--noise-pA` knob is a config flag on the net's own bridge).

Run (1-seed CPU forward-only probe):
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -m research.runners._onbridge_deep_credit_decorrelation_derisk \
        --seed 42 --k-list 1 8 16 --noise-pA-list 0 20 50 100
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

# reuse-by-import: the committed on-bridge spiking deep-credit net (built forward-only; NO training) + the task builder.
from research.runners._semantic_inheritance_onbridge_spiking_derisk import OnBridgeBDSPNet  # noqa: E402
from research.runners._semantic_inheritance_deep_credit_derisk import make_task_semantic_inheritance  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_onbridge_deep_credit_decorrelation.json"

CORR_RISE_MARGIN = 0.05   # read-SNR corr(Kmax) must exceed corr(K1) by this to count as "rises with K"
CORR_DROP_MARGIN = 0.10   # within-unit corr must fall by this from noise=0 to count as "decorrelates"


# ---------------------------------------------------------------------------------------------------------------
# INDEPENDENT per-neuron background noise via the bridge's OWN OU process (config flag on the net's bridge; NO sim/ edit).
# noise_pA<=0 => FORCE OU OFF (deterministic constant drive = the correlated/lockstep baseline the a0 diagnosis describes).
# noise_pA>0  => OU on at that sigma; cp.random.randn(n) per step => INDEPENDENT OU trajectory per neuron (Destexhe-Rudolph).
# ---------------------------------------------------------------------------------------------------------------
def _set_bg_noise(net, noise_pA, ou_seed):
    cfg = net.cfg
    if noise_pA <= 0.0:
        cfg.enable_ou_process = False
        cur = getattr(net.br, "cp_ou_current", None)
        if cur is not None:
            cur[...] = 0.0
    else:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(noise_pA)
        # recompute the exact-OU step coefficients (ou_noise_std/decay) + reset cp_ou_current to the mean.
        net.br._initialize_ou_process_state(cfg, net.n_total)
    # seed the backend GLOBAL RNG (bridge OU uses cp.random.randn == np.random on the numpy backend) for reproducibility.
    np.random.seed(int(ou_seed))


def _offdiag_mean_corr(mat_NxM):
    """Mean of the off-diagonal entries of the MxM Pearson-corr matrix of the columns of an (N, M) array (N samples,
    M neurons). NaN-safe: neurons with ~zero variance -> their corr row/col is NaN and is dropped by nanmean.
    Returns NaN if M<2 or every pair is degenerate."""
    a = np.asarray(mat_NxM, dtype=np.float64)
    if a.shape[1] < 2 or a.shape[0] < 3:
        return float("nan")
    # zero-variance columns -> corrcoef yields NaN there; keep them, nanmean drops the NaNs.
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(a, rowvar=False)
    M = c.shape[0]
    iu = np.triu_indices(M, k=1)
    vals = c[iu]
    if not np.any(np.isfinite(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def _units_pairwise_corr(Ephys_NxP, size_logical, K):
    """Mean over logical units of the within-unit pairwise corr of its K contiguous physical neurons.
    Physical layout is UNIT-MAJOR (unit u owns phys idx [u*K, u*K+K); matches OnBridgeBDSPNet._pool reshape(size,K))."""
    if K < 2:
        return float("nan")
    a = np.asarray(Ephys_NxP, dtype=np.float64)
    per_unit = []
    for u in range(size_logical):
        sub = a[:, u * K:(u + 1) * K]
        per_unit.append(_offdiag_mean_corr(sub))
    per_unit = [v for v in per_unit if np.isfinite(v)]
    return float(np.mean(per_unit)) if per_unit else float("nan")


def _read_host_E(net):
    from sim.backend import to_host
    return np.asarray(to_host(net.br.cp_bdsp_E)).astype(np.float64).copy()


def measure_net(net, X, fixed_inputs, n_repeats):
    """FORWARD-ONLY measurements on a built+noise-configured net.
      main loop over X: per hidden layer collect physical E, pooled E, clean soma-rate proxy ->
        (A) across-input within-unit pairwise corr; (B1) read-SNR = corr(pooled E, soma proxy).
      fixed-input repeat loop: per fixed input, n_repeats forward passes with fresh OU noise ->
        (A) pure-noise within-unit pairwise corr; (B2) single-neuron noise-CV vs pooled noise-CV -> pooling gain."""
    K = net.pool_k
    sizes = net.sizes
    nhid = net.n_hidden_layers
    hid_li = [k + 1 for k in range(nhid)]        # layer indices of the hidden layers

    # ---- main loop: across the input set ----
    Ephys = {li: [] for li in hid_li}            # per layer: list of (sizes_phys[li],) physical E vectors
    pooledE = {li: [] for li in hid_li}          # per layer: list of (sizes[li],) pooled E
    somaE = {li: [] for li in hid_li}            # per layer: list of (sizes[li],) clean soma-rate proxy
    E_rate = {li: [] for li in hid_li}           # mean physical event rate per layer per input (activity check)
    for x in X:
        net._forward_spiking(x)                  # runs the spiking settle; leaves cp_bdsp_E set for THIS input
        Efull = _read_host_E(net)
        for li in hid_li:
            ep = Efull[net.slices[li]]
            Ephys[li].append(ep)
            pooledE[li].append(ep.reshape(sizes[li], K).mean(axis=1))
            somaE[li].append(net.soma_rate_proxy(li))
            E_rate[li].append(float(ep.mean()))

    across_corr = []          # per hidden layer: across-input within-unit pairwise corr
    read_snr = []             # per hidden layer: corr(pooled E, soma proxy)
    act_rate = []             # per hidden layer: mean physical event rate
    for li in hid_li:
        Eph = np.asarray(Ephys[li])              # (N, sizes_phys[li])
        across_corr.append(_units_pairwise_corr(Eph, sizes[li], K))
        pe = np.asarray(pooledE[li]).reshape(-1)
        se = np.asarray(somaE[li]).reshape(-1)
        if pe.std() < 1e-12 or se.std() < 1e-12:
            read_snr.append(float("nan"))
        else:
            with np.errstate(invalid="ignore"):
                read_snr.append(float(np.corrcoef(pe, se)[0, 1]))
        act_rate.append(float(np.mean(E_rate[li])))

    # ---- fixed-input repeat loop: pure-noise correlation + pooling gain (~sqrt(K) test) ----
    noise_corr_layers = []    # per layer: pure-noise within-unit pairwise corr (mean over fixed inputs)
    single_cv_layers = []     # per layer: single-neuron noise CV
    pooled_cv_layers = []     # per layer: pooled noise CV
    gain_layers = []          # per layer: single_CV / pooled_CV (empirical pooling gain)
    for li in hid_li:
        ncorr_f, scv_f, pcv_f, gain_f = [], [], [], []
        for x in fixed_inputs:
            reps = []
            for _ in range(n_repeats):
                net._forward_spiking(x)          # fresh OU realization each repeat (OU state evolves)
                reps.append(_read_host_E(net)[net.slices[li]])
            R = np.asarray(reps)                 # (n_repeats, sizes_phys[li])
            # pure-noise within-unit pairwise corr (variation across repeats = noise only, for a FIXED input)
            ncorr_f.append(_units_pairwise_corr(R, sizes[li], K))
            # single-neuron noise CV: std over repeats / mean over repeats, per neuron; drop ~silent neurons.
            mu = R.mean(axis=0); sd = R.std(axis=0)
            live = mu > 1e-6
            if np.any(live):
                scv_f.append(float(np.mean(sd[live] / mu[live])))
            else:
                scv_f.append(float("nan"))
            # pooled noise CV: pool each unit's K neurons per repeat -> (n_repeats, sizes[li]); CV over repeats.
            P = R.reshape(n_repeats, sizes[li], K).mean(axis=2)   # (n_repeats, sizes[li])
            pmu = P.mean(axis=0); psd = P.std(axis=0); plive = pmu > 1e-6
            if np.any(plive):
                pcv = float(np.mean(psd[plive] / pmu[plive]))
                pcv_f.append(pcv)
                s = scv_f[-1]
                gain_f.append(float(s / pcv) if (np.isfinite(s) and pcv > 1e-9) else float("nan"))
            else:
                pcv_f.append(float("nan")); gain_f.append(float("nan"))
        noise_corr_layers.append(float(np.nanmean(ncorr_f)) if np.any(np.isfinite(ncorr_f)) else float("nan"))
        single_cv_layers.append(float(np.nanmean(scv_f)) if np.any(np.isfinite(scv_f)) else float("nan"))
        pooled_cv_layers.append(float(np.nanmean(pcv_f)) if np.any(np.isfinite(pcv_f)) else float("nan"))
        gain_layers.append(float(np.nanmean(gain_f)) if np.any(np.isfinite(gain_f)) else float("nan"))

    def _m(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    return {
        "pool_k": int(K),
        "across_input_pairwise_corr": _m(across_corr), "across_input_pairwise_corr_by_layer": across_corr,
        "read_snr_corr": _m(read_snr), "read_snr_corr_by_layer": read_snr,
        "noise_pairwise_corr": _m(noise_corr_layers), "noise_pairwise_corr_by_layer": noise_corr_layers,
        "single_neuron_noise_cv": _m(single_cv_layers), "pooled_noise_cv": _m(pooled_cv_layers),
        "pooling_gain": _m(gain_layers), "pooling_gain_by_layer": gain_layers,
        "sqrt_k": float(np.sqrt(K)), "mean_event_rate": _m(act_rate)}


def run_seed(seed, k_list, noise_list, hidden, settle_steps, n_hidden_layers, n_inputs, n_fixed, n_repeats,
             hp, task_kwargs, ou_seed):
    # build the task ONCE (for realistic input vectors); forward-only -> labels/held-out irrelevant.
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte), meta, idx = make_task_semantic_inheritance(seed, **task_kwargs)
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    X_all = np.concatenate([Xtr, Xte], axis=0)
    rng = np.random.default_rng(seed + 7)
    sel = rng.permutation(len(X_all))[:min(n_inputs, len(X_all))]
    X = X_all[sel]
    fixed = X[:min(n_fixed, len(X))]

    grid = {}   # (K, noise) -> measurement
    for K in k_list:
        for noise in noise_list:
            net = OnBridgeBDSPNet(n_in, hidden, k, seed=seed, rule="plain_fa",
                                  n_hidden_layers=n_hidden_layers, settle_steps=settle_steps, credit_steps=1,
                                  lr=0.0, pool_k=K,
                                  tonic_h_pA=hp["tonic_h_pA"], tonic_o_pA=hp["tonic_o_pA"],
                                  apical_gain_pA=hp["apical_gain_pA"], ff_w_init=hp["ff_w_init"],
                                  pbar_alpha=hp["pbar_alpha"])
            _set_bg_noise(net, noise, ou_seed)
            m = measure_net(net, X, fixed, n_repeats)
            grid[(K, noise)] = m
            del net
    return {"seed": seed, "meta": meta, "n_in": int(n_in), "k_classes": int(k), "n_inputs": int(len(X)),
            "n_fixed": int(len(fixed)), "n_repeats": int(n_repeats), "k_list": [int(x) for x in k_list],
            "noise_list": [float(x) for x in noise_list], "grid": grid}


def _fmt_grid(res):
    """Print the (K x noise) tables + a verdict from one seed's grid."""
    k_list = res["k_list"]; noise_list = res["noise_list"]; grid = res["grid"]
    lines = []

    def cell(K, noise, key):
        return grid[(K, noise)][key]

    # ---- (A) pairwise within-unit correlation tables ----
    lines.append("(A) PAIRWISE WITHIN-UNIT CORRELATION of cp_bdsp_E  [HIGH=lockstep/correlated; LOW=decorrelated]")
    lines.append("    across-input corr (signal+noise):")
    hdr = "        K \\ noise  " + "".join(f"{n:>10.0f}" for n in noise_list)
    lines.append(hdr)
    for K in k_list:
        row = "".join((f"{cell(K, n, 'across_input_pairwise_corr'):>10.3f}"
                       if np.isfinite(cell(K, n, 'across_input_pairwise_corr')) else f"{'n/a':>10}") for n in noise_list)
        lines.append(f"        K={K:<8d}{row}")
    lines.append("    pure-NOISE corr (fixed input, across repeats):")
    lines.append(hdr)
    for K in k_list:
        row = "".join((f"{cell(K, n, 'noise_pairwise_corr'):>10.3f}"
                       if np.isfinite(cell(K, n, 'noise_pairwise_corr')) else f"{'n/a':>10}") for n in noise_list)
        lines.append(f"        K={K:<8d}{row}")

    # ---- (B1) read-SNR corr(pooled E, soma) vs K ----
    lines.append("(B1) READ-SNR  corr(pooled E, clean soma-rate)  [should RISE with K only when decorrelated]")
    lines.append(hdr)
    for K in k_list:
        row = "".join((f"{cell(K, n, 'read_snr_corr'):>10.3f}"
                       if np.isfinite(cell(K, n, 'read_snr_corr')) else f"{'n/a':>10}") for n in noise_list)
        lines.append(f"        K={K:<8d}{row}")

    # ---- (B2) pooling gain single_CV/pooled_CV vs K (target ~sqrt(K)) ----
    lines.append("(B2) POOLING GAIN  single_CV / pooled_CV   [target ~sqrt(K): K=8->2.83, K=16->4.0; ~1=correlated]")
    lines.append(hdr)
    for K in k_list:
        row = "".join((f"{cell(K, n, 'pooling_gain'):>10.3f}"
                       if np.isfinite(cell(K, n, 'pooling_gain')) else f"{'n/a':>10}") for n in noise_list)
        lines.append(f"        K={K:<8d}{row}  (sqrt(K)={np.sqrt(K):.2f})")

    lines.append("     mean event rate cp_bdsp_E per hidden layer (activity sanity):")
    lines.append(hdr)
    for K in k_list:
        row = "".join(f"{cell(K, n, 'mean_event_rate'):>10.3f}" for n in noise_list)
        lines.append(f"        K={K:<8d}{row}")
    return "\n".join(lines)


def _verdict(res):
    k_list = res["k_list"]; noise_list = res["noise_list"]; grid = res["grid"]
    Kmax = max(k_list); K1 = min(k_list)
    noise0 = min(noise_list)
    pos_noise = [n for n in noise_list if n > 0]
    if not pos_noise:
        return "NO noise>0 level tested -- add --noise-pA-list with positive levels to test the decorrelation fix."
    noise_hi = max(pos_noise)

    def g(K, n, key):
        return grid[(K, n)][key]

    # baseline (noise=0): correlated pool + flat read-SNR + no pooling gain.
    base_corr = g(Kmax, noise0, "across_input_pairwise_corr")
    base_readsnr_k1 = g(K1, noise0, "read_snr_corr")
    base_readsnr_kmax = g(Kmax, noise0, "read_snr_corr")
    base_readsnr_flat = bool((not np.isfinite(base_readsnr_k1)) or (not np.isfinite(base_readsnr_kmax))
                             or abs(base_readsnr_kmax - base_readsnr_k1) < CORR_RISE_MARGIN)
    base_gain_kmax = g(Kmax, noise0, "pooling_gain")

    # at the highest noise: decorrelation + read-SNR rises with K + pooling gain -> sqrt(K).
    hi_corr = g(Kmax, noise_hi, "across_input_pairwise_corr")
    hi_noisecorr = g(Kmax, noise_hi, "noise_pairwise_corr")
    hi_readsnr_k1 = g(K1, noise_hi, "read_snr_corr")
    hi_readsnr_kmax = g(Kmax, noise_hi, "read_snr_corr")
    hi_gain_kmax = g(Kmax, noise_hi, "pooling_gain")
    sqrtKmax = float(np.sqrt(Kmax))

    decorrelates = bool(np.isfinite(base_corr) and np.isfinite(hi_corr) and (base_corr - hi_corr) > CORR_DROP_MARGIN)
    readsnr_rises = bool(np.isfinite(hi_readsnr_k1) and np.isfinite(hi_readsnr_kmax)
                         and (hi_readsnr_kmax - hi_readsnr_k1) > CORR_RISE_MARGIN)
    # pooling gain "restored" = at high noise the gain at Kmax is clearly >1 and clearly above the noise=0 baseline
    # AND reaches at least ~60% of the ideal sqrt(K) (independent-noise averaging is realized, if imperfectly).
    gain_restored = bool(np.isfinite(hi_gain_kmax) and hi_gain_kmax > 1.3
                         and hi_gain_kmax > (base_gain_kmax if np.isfinite(base_gain_kmax) else 1.0) + 0.3
                         and hi_gain_kmax >= 0.60 * sqrtKmax)

    baseline_reproduced = bool((np.isfinite(base_corr) and base_corr > 0.5) and base_readsnr_flat)

    # --- the DIRECT-MEASUREMENT branch the a0 diagnosis did NOT anticipate: is the pool ALREADY decorrelated at noise=0
    # (within-unit corr near zero) AND is the ~sqrt(K) pooling gain ALREADY present at noise=0? If so, the a0
    # correlated-pool / lockstep premise is REFUTED by measurement, and independent noise is NOT the missing lever. ---
    base_already_decorrelated = bool(np.isfinite(base_corr) and base_corr < 0.15)
    base_pool_works = bool(np.isfinite(base_gain_kmax) and base_gain_kmax >= 0.60 * sqrtKmax and base_gain_kmax > 1.3)
    base_readsnr_rises = bool(np.isfinite(base_readsnr_k1) and np.isfinite(base_readsnr_kmax)
                              and (base_readsnr_kmax - base_readsnr_k1) > CORR_RISE_MARGIN)
    # does adding noise HELP or HURT the across-input read-SNR at Kmax?
    noise_helps_readsnr = bool(np.isfinite(base_readsnr_kmax) and np.isfinite(hi_readsnr_kmax)
                               and (hi_readsnr_kmax - base_readsnr_kmax) > CORR_RISE_MARGIN)

    tag = (f"[baseline noise=0: within-unit corr {base_corr:.3f}, read-SNR K{K1}->{Kmax} "
           f"{base_readsnr_k1:.3f}->{base_readsnr_kmax:.3f}, gain@K{Kmax} {base_gain_kmax:.2f}/sqrt {sqrtKmax:.2f}] "
           f"[noise={noise_hi:.0f}: within-unit corr {hi_corr:.3f} (noise-corr {hi_noisecorr:.3f}), read-SNR "
           f"K{K1}->{Kmax} {hi_readsnr_k1:.3f}->{hi_readsnr_kmax:.3f}, gain@K{Kmax} {hi_gain_kmax:.2f}]")

    if base_already_decorrelated and base_pool_works and not base_readsnr_rises:
        return (f"A0-PREMISE REFUTED / HONEST NEGATIVE (1-seed forward-only) -- DIRECT MEASUREMENT overturns the a0 "
                f"correlated-pool root cause: at noise=0 the pool is ALREADY DECORRELATED (within-unit corr {base_corr:.3f}"
                f", not lockstep) and the ~sqrt(K) pooling gain is ALREADY present (gain@K{Kmax} {base_gain_kmax:.2f} vs "
                f"sqrt(K) {sqrtKmax:.2f}) -- pooling ALREADY averages independent variance (each hidden neuron has its own "
                f"random FF weights + membrane-state carryover between passes decorrelate the K-block). YET the across-input "
                f"read-SNR corr(pooled E, soma) does NOT rise with K ({base_readsnr_k1:.3f}->{base_readsnr_kmax:.3f}) and "
                f"independent OU noise {'HELPS' if noise_helps_readsnr else 'does NOT help (it HURTS)'} it "
                f"({base_readsnr_kmax:.3f}->{hi_readsnr_kmax:.3f} at K{Kmax}; OU also contaminates the soma reference). => "
                f"the flat-read-SNR-across-K residual is NOT read-variance / not a correlated pool; independent noise is "
                f"NOT the fix. REFRAME DEEPER: the residual is representational / credit-STRUCTURE (pooling cuts trial "
                f"noise but not across-input signal fidelity). CONTROLLER: do NOT pursue independent-noise; the lever is "
                f"the microcircuit clean-error channel / representation, not population read-variance. {tag}")

    if decorrelates and readsnr_rises and gain_restored:
        return (f"GO-CANDIDATE (1-seed forward-only) -- INDEPENDENT per-neuron OU noise DECORRELATES the pool AND "
                f"restores the ~sqrt(K) pooling gain: within-unit corr FALLS ({base_corr:.3f}->{hi_corr:.3f}), read-SNR "
                f"RISES with K (K{K1}->{Kmax}: {hi_readsnr_k1:.3f}->{hi_readsnr_kmax:.3f}), pooling gain @K{Kmax} "
                f"{hi_gain_kmax:.2f} (~{100*hi_gain_kmax/sqrtKmax:.0f}% of sqrt(K)={sqrtKmax:.2f}); the noise=0 baseline "
                f"reproduces the correlated-pool root cause ({'yes' if baseline_reproduced else 'PARTIAL'}). => population "
                f"coding becomes viable; CONTROLLER: test end-to-end training (K-sweep x noise, 6-seed GPU). {tag}")
    if decorrelates and (readsnr_rises or gain_restored):
        return (f"PARTIAL / PROMISING (1-seed forward-only) -- independent OU noise DECORRELATES the pool (within-unit "
                f"corr {base_corr:.3f}->{hi_corr:.3f}) and lifts one of the two pooling reads (read-SNR rises: "
                f"{readsnr_rises}; ~sqrt(K) gain restored: {gain_restored}) but not both cleanly at this forward-only "
                f"scale. The read-variance lever is REAL; CONTROLLER: sweep noise x K wider (higher sigma / more repeats / "
                f"GPU) then test end-to-end training. {tag}")
    if decorrelates:
        return (f"HONEST MIXED (1-seed forward-only) -- independent OU noise DOES decorrelate the pool (within-unit corr "
                f"{base_corr:.3f}->{hi_corr:.3f}) but neither read-SNR-rises-with-K nor the ~sqrt(K) pooling gain clearly "
                f"materialized at these noise levels. The decorrelation is necessary-not-sufficient here; the read may be "
                f"limited by another factor (event-rate floor, settle window, non-Poisson spiking). CONTROLLER: sweep "
                f"sigma higher + more repeats; if still flat, the residual is not pure read-variance. {tag}")
    return (f"HONEST NEGATIVE (1-seed forward-only) -- independent OU noise up to {noise_hi:.0f} pA did NOT decorrelate "
            f"the pool (within-unit corr {base_corr:.3f}->{hi_corr:.3f}) -> the tonic drive dominates the spike timing and "
            f"pooling still cannot average independent variance. This REFRAMES the boundary DEEPER: the correlated-pool is "
            f"not fixed by this noise regime; the lever is either MUCH higher conductance-noise (a stronger high-conductance "
            f"state) or a credit-STRUCTURE change (not read-variance). {tag}")


def main():
    ap = argparse.ArgumentParser(description="On-bridge deep-credit DECORRELATION forward-only probe (read-SNR-wall fix).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-list", type=int, nargs="+", default=[1, 8, 16],
                    help="pool sizes K (neurons per logical unit). K=1 = the single-neuron baseline; K>1 = population.")
    ap.add_argument("--noise-pA-list", type=float, nargs="+", default=[0, 20, 50, 100],
                    help="independent per-neuron OU background-noise sigma (pA). 0 = FORCE OU OFF (deterministic "
                         "correlated baseline); >0 = independent OU noise (100 == the committed CoreSimConfig default).")
    ap.add_argument("--hidden", type=int, default=24, help="hidden LOGICAL units per layer (forward-only; small=fast)")
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--settle-steps", type=int, default=25, help="spiking forward-settle steps per example")
    ap.add_argument("--n-inputs", type=int, default=40, help="distinct inputs for the across-input reads (A)/(B1)")
    ap.add_argument("--n-fixed", type=int, default=6, help="fixed inputs for the pure-noise repeat reads (A-noise)/(B2)")
    ap.add_argument("--n-repeats", type=int, default=10, help="repeats per fixed input (fresh OU noise each) for (B2)")
    ap.add_argument("--ou-seed", type=int, default=12345, help="reproducibility seed for the OU noise draws")
    # drive hyperparameters -- MATCH the population runner's smoke defaults so the noise=0 arm reproduces the committed
    # correlated-pool baseline (the a0 flat ~0.28 read-SNR).
    ap.add_argument("--tonic-h-pA", type=float, default=560.0)
    ap.add_argument("--tonic-o-pA", type=float, default=620.0)
    ap.add_argument("--apical-gain-pA", type=float, default=2000.0)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
    ap.add_argument("--ff-w-init", type=float, default=4.5)
    # task knobs -- MATCH the on-bridge runner's CPU-smoke defaults (n_prop=2 = the 5-class config) so the net's regime
    # matches the committed arm. (labels unused here -- forward only.)
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                       n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise,
                       feature_seed=a.feature_seed)
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, apical_gain_pA=a.apical_gain_pA,
              ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha)

    t0 = time.time(); err = None; res = None
    try:
        res = run_seed(a.seed, a.k_list, a.noise_pA_list, a.hidden, a.settle_steps, a.n_hidden_layers,
                       a.n_inputs, a.n_fixed, a.n_repeats, hp, task_kwargs, a.ou_seed)
        print("-" * 112, flush=True)
        print(f"[seed {a.seed}] n_in {res['n_in']} | k_classes {res['k_classes']} | inputs {res['n_inputs']} | "
              f"fixed {res['n_fixed']} x repeats {res['n_repeats']} | K {a.k_list} | noise-pA {a.noise_pA_list}",
              flush=True)
        print(_fmt_grid(res), flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    # JSON-safe: stringify the (K,noise) tuple keys.
    grid_json = None
    if res is not None:
        grid_json = {f"K{K}_noise{int(n)}": v for (K, n), v in res["grid"].items()}

    summary = {"probe": "onbridge_deep_credit_decorrelation_forward_only", "seed": a.seed,
               "k_list": a.k_list, "noise_pA_list": a.noise_pA_list,
               "config": {"hidden": a.hidden, "n_hidden_layers": a.n_hidden_layers, "settle_steps": a.settle_steps,
                          "n_inputs": a.n_inputs, "n_fixed": a.n_fixed, "n_repeats": a.n_repeats,
                          "ou_seed": a.ou_seed, "hp": hp, "task": task_kwargs,
                          "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1),
               "result": (dict(seed=res["seed"], meta=res["meta"], n_in=res["n_in"], k_classes=res["k_classes"],
                               n_inputs=res["n_inputs"], n_fixed=res["n_fixed"], n_repeats=res["n_repeats"],
                               k_list=res["k_list"], noise_list=res["noise_list"], grid=grid_json)
                          if res is not None else None)}
    if err is None and res is not None:
        summary["verdict"] = _verdict(res)
    else:
        summary["verdict"] = f"ERROR -- {err}" if err else "no result"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[onbridge-deep-credit-decorrelation] {summary['verdict']}", flush=True)
    print(f"[onbridge-deep-credit-decorrelation] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
