"""Tier-2 ON-BRIDGE realization of the linattn mouth's num/den divisive normalization (2026-09-04).

DESIGN: research/findings/2026-09-03-linattn-spike-native-normalization-DESIGN.md (read in full before this
file was written). This is the "real spiking test" the design's Sec 4 names: instantiate the shunting-
normalization-interneuron circuit on the bridge (a D-neuron READ pool + a small NORM-neuron pool, a GABA_A-
like shunting divisor, the fluctuation-driven/high-conductance regime), and compare the read pool's settled
firing to the exact `num_t / (den_t + eps)` the linattn mouth's read performs today
(`research/runners/_emerge_wkv_lm_derisk.py`, `LinAttnLayer.forward`, line ~689).

THE ON-BRIDGE PRIMITIVE this runner drives is `enable_shunt_norm_pool` (sim/config.py, sim/regions.py,
sim/bridge.py, added alongside this runner): a THIRD, independent divisive-gain pool with the SAME
`r_i = x_i / (sigma + gain*pool)` machinery as `enable_input_divisive_norm`, but whose divisor is an
EXTERNAL scalar `den_ema` -- the firing-rate EMA of a separate, designated norm-neuron region -- rather
than the flagged (read) set's own current mean (the wrong axis; the already-refuted --dual-nonneg-divnorm
channel-pool NO-GO, DESIGN doc Sec 1/4). See `tests/test_shunt_norm_pool.py` for the primitive's own
byte-identical-when-off + mechanism + sigma-domination proofs at the sim/ level (CPU, seconds). THIS file
is one level up: it drives that primitive with REAL trained linattn weights (checkpoint
`bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz`, D=192) and REAL token statistics, and runs
the Tier-2-SPECIFIC anti-cheats (divisive-vs-subtractive, sigma-domination) as richer, checkpoint-grounded
sweeps.

THE CIRCUIT (DESIGN doc Sec 3a, the ON/OFF signed-value split): `num_t` is signed (`v` is signed), so the
READ side is TWO non-negative sub-pools -- `read_on` (D neurons, driven by `relu(num_i)`) and `read_off`
(D neurons, driven by `relu(-num_i)`) -- both flagged `shunt_norm_read=True`, sharing ONE `shunt_norm_source`
norm-neuron pool whose firing-rate EMA carries `den`. The final on-bridge read is `rate(read_on) -
rate(read_off)`, directly comparable (mod an overall current<->rate calibration, exposed as
`--num-to-pA-gain`/`--den-to-pA-gain`) to the exact `num_i / (sigma + den)`. This ON/OFF convention is not
invented here -- it is the SAME one this exact checkpoint's OTHER layers already use on-bridge/on-disk
(`extra_ssm.0.Wo_sp.weight` is (192, 384) = D and 2*D, i.e. concatenated [ON;OFF]).

TWO ANTI-CHEATS, BOTH MEASURED HERE, CHECKPOINT-INDEPENDENT (pure circuit properties -- also exercised, at
tiny scale, by `--smoke`):
  1. DIVISIVE-NOT-SUBTRACTIVE (the Holt & Koch 1997 crux, DESIGN doc Sec 2/4). Probe the read pool's f-I
     curve (settled rate vs. injected `num`-current) at several `den` levels and fit a local slope
     (gain) at each. DIVISIVE means the slope SCALES as ~1/(sigma + gain*den) (so 1/slope is LINEAR in
     den -- `linear_fit_r2` below); SUBTRACTIVE means the OFFSET/rheobase shifts while the slope stays
     roughly flat. Both are fit and reported; the design's crux is which one dominates on OUR substrate.
  2. SIGMA-DOMINATION (the "97% of the effect was the clamp" trap, CLAUDE.md / DESIGN doc Sec 4). Sweep
     `shunt_norm_sigma`; at the CHOSEN operating sigma, the silent-vs-driven-source contrast must still be
     large (den matters) -- if sigma is so large the contrast collapses to ~1.0, den has stopped
     mattering and the "division" is actually a fixed, den-independent gain.

MARGIN PRESERVATION (DESIGN doc Sec 4's THIRD Tier-2 GO-gate criterion, `margin_vs_trigram >= +0.03` 6/6
seeds vs the exact-division linattn) is **NOT measured by this runner**, and this is an HONEST, NAMED
RESIDUAL, not an oversight: `margin_vs_trigram` requires substituting the on-bridge shunt into the FULL
trained stack's generation pipeline (`emb` -> `extra`/`extra_ssm`/`linattn_layers.{0,1}` in whatever order
produced this checkpoint -> `head`) and re-running the SAME held-out per-depth-bucket eval
`_emerge_wkv_lm_derisk.eval_perdepth` uses -- which in turn requires an EVAL-ONLY checkpoint-loading path
into that script (today it only trains-then-evals; `main()` has no `--load` flag). Building that path is
squarely what Tier-1's SEPARATE, already-assigned rung needs too (the rate-model swap on "the ALREADY-
TRAINED checkpoint", DESIGN doc Sec 4 Tier 1 -- see branch `research/linattn-shunt-gain-tier1-redo`), so it
is Tier-1's dependency to land first; THIS runner is written to plug into the SAME `--linattn-div` slot the
design names (Sec 3e: `LinAttnLayer.forward`'s `read = num/(den+eps)`) via `--linattn-div-mode
shunt_onbridge` once that lands, rather than duplicating/racing a second checkpoint-eval-only path. Instead,
this runner measures MECHANISM FIDELITY directly and honestly: for real layer-0 trained weights
(Wq/Wk/Wv/w/LN) replayed (host-side, numpy -- the SAME graded-fast-weight-state concession the shipped
mouth already carries, DESIGN doc Sec 3a) over real embedded token sequences, how close is the on-bridge
settled read to the exact `num/(den+eps)` -- per-token relative error + cosine similarity, bucketed by
context depth (mirroring the d10-99 framing). This is the PRIMARY quantity Tier 2's own Sec 4 wording
targets ("compare the read pool's settled output rate to the exact num/(den+eps)"); margin_vs_trigram is
the downstream, full-stack consequence, queued as the explicit next integration step below.

COST-ROUTING: GPU queue (trivial VRAM, D=192*2+n_norm neurons per trial -- well within the single-3090
consumer reference), sequenced AFTER the live verification frees the GPU (not an agent; mechanical replay +
measurement). CPU/numpy is fine for a quick check of the mechanism-only anti-cheats (no checkpoint needed).

USAGE:
    # tiny smoke (CPU, seconds; confirms imports/parses/starts a step; NOT a scientific verdict):
    SIM_BACKEND=numpy python -m research.runners._linattn_shunt_gain_tier2_onbridge_derisk --smoke

    # full de-risk (GPU queue; NOT run by this file's own authors -- see the residual above):
    python -m research.runners._linattn_shunt_gain_tier2_onbridge_derisk \\
        --seeds 42,43,44,100,101,102 --n-settle-steps 40 --n-trials 64 \\
        --json research/findings/raw/_linattn_shunt_gain_tier2_onbridge.json

    # re-check the GO gate against a previously produced result, no recompute:
    python -m research.runners._linattn_shunt_gain_tier2_onbridge_derisk --check-go <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from sim.backend import to_host as _to_host   # backend-safe cupy/numpy -> host numpy (bridge arrays may be cupy)

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]
DEFAULT_CKPT_TMPL = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
DEFAULT_BPE_PATH = "bridges/wkv_ckpt/wkv_bpe8k.json"
DEFAULT_CORPUS = "data/corpus/tinystories_train.txt"
EPS = 1e-6

# ─────────────────────────────────────────────────────────────────── GO-gate thresholds (DESIGN doc Sec 4) ──
MARGIN_MIN = 0.03                 # margin_vs_trigram >= +0.03, 6/6 seeds (when measured -- see residual above)
DIVISIVE_R2_MIN = 0.80            # 1/slope(den) linear-in-den fit quality required to call it "divisive"
SIGMA_DOMINATION_MAX_RATIO = 0.5  # at the operating sigma, silent/driven contrast ratio must stay BELOW this
SIGMA_DOMINATION_COLLAPSE_MIN = 0.85  # sanity: a deliberately-huge sigma must show the ratio COLLAPSE near 1.0


# ═══════════════════════════════════════════════════════════════════════════ host-side (graded) recurrence ══
def phi_elu(x: np.ndarray) -> np.ndarray:
    """elu(x)+1 (the class default phi, LinAttnLayer._phi 'elu' branch): non-negative feature map,
    Katharopoulos et al. 2020 Eq.7."""
    return np.where(x > 0.0, x, np.expm1(x)) + 1.0


def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def load_layer0_weights(ckpt_path: str, layer: int = 0) -> dict:
    """Load ONE LinAttnLayer's trained weights + the shared embedding table from the deployed checkpoint.
    Numpy-only (no torch needed for pure inference) -- keys verified against the actual .npz this session:
    `linattn_layers.{i}.{Wq,Wk,Wv,Wr,Wo}.weight` (192,192 each), `linattn_layers.{i}.w` (1,) [uniform_decay],
    `linattn_layers.{i}.ln.{weight,bias}` (192,), plus `emb.weight` (V,192) and `words` (V,)."""
    d = np.load(ckpt_path, allow_pickle=True)
    pfx = f"linattn_layers.{layer}."
    missing = [k for k in ("Wq.weight", "Wk.weight", "Wv.weight", "Wr.weight", "Wo.weight",
                            "w", "ln.weight", "ln.bias") if (pfx + k) not in d.files]
    if missing:
        raise KeyError(f"{ckpt_path}: missing {[pfx + m for m in missing]} -- not a linattn checkpoint "
                        f"with a layer {layer}, or the save format changed. Keys present: {list(d.files)[:20]}...")
    return {
        "Wq": d[pfx + "Wq.weight"], "Wk": d[pfx + "Wk.weight"], "Wv": d[pfx + "Wv.weight"],
        "Wr": d[pfx + "Wr.weight"], "Wo": d[pfx + "Wo.weight"],
        "w": d[pfx + "w"], "ln_w": d[pfx + "ln.weight"], "ln_b": d[pfx + "ln.bias"],
        "emb": d["emb.weight"], "words": d["words"] if "words" in d.files else None,
        "D": int(d[pfx + "Wq.weight"].shape[0]), "V": int(d["emb.weight"].shape[0]),
    }


def host_layer0_num_den(weights: dict, token_ids: list[int], phi=phi_elu) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Faithful, self-contained reimplementation of LinAttnLayer.forward's recurrence for LAYER 0 ONLY
    (mirrors research/runners/_emerge_wkv_lm_derisk.py LinAttnLayer.forward line-for-line, read in full
    2026-09-03/04), driven by the REAL trained embedding table as this layer's input `h` -- a clearly-
    scoped "layer-0-isolated" replay (the true input to linattn.0 in the full multi-layer stack also
    includes whatever `extra`/`extra_ssm` layers precede it; using raw embeddings still gives REALISTIC-
    MAGNITUDE q/k/v/M/zden/num/den statistics from the ACTUAL trained Wq/Wk/Wv/w, which is what a
    division-mechanism fidelity test needs -- see the module docstring's honest-residual note on why the
    FULL multi-layer margin_vs_trigram is not attempted here).

    Returns (num_seq [T,D], den_seq [T], exact_read_seq [T,D]) -- exact_read = num/(den+eps), the ground
    truth this runner's on-bridge settle is compared against.
    """
    D = weights["D"]
    h = weights["emb"][np.asarray(token_ids, dtype=np.int64)]           # [T,D] real trained embeddings
    z = layernorm(h, weights["ln_w"], weights["ln_b"])
    q = phi(z @ weights["Wq"].T)
    k = phi(z @ weights["Wk"].T)
    v = z @ weights["Wv"].T
    lam = float(np.exp(-np.logaddexp(0.0, -weights["w"][0])))            # exp(-softplus(w)), uniform_decay
    T = len(token_ids)
    M = np.zeros((D, D), dtype=np.float64)
    zden = np.zeros(D, dtype=np.float64)
    num_seq = np.zeros((T, D), dtype=np.float64)
    den_seq = np.zeros(T, dtype=np.float64)
    for t in range(T):
        M = lam * M + np.outer(k[t], v[t])
        zden = lam * zden + k[t]
        num_seq[t] = q[t] @ M
        den_seq[t] = float(q[t] @ zden)
    exact_read_seq = num_seq / (den_seq[:, None] + EPS)
    return num_seq, den_seq, exact_read_seq


def sample_synthetic_token_ids(V: int, T: int, rng: np.random.Generator, zipf_a: float = 1.1) -> list[int]:
    """Fallback when no real corpus is available (this worktree; portable to wherever this eventually
    runs): Zipfian-shaped token ids over the real vocab size -- a standard, honest proxy for natural-
    language token frequency (NOT claimed as real text; labeled `corpus_source=synthetic_zipf` in output)."""
    ranks = rng.zipf(zipf_a, size=T * 3)
    ranks = ranks[ranks <= V]
    if len(ranks) < T:
        ranks = np.concatenate([ranks, rng.integers(1, V + 1, size=T)])
    return (ranks[:T] - 1).astype(np.int64).tolist()


def load_real_token_ids(corpus_path: str, bpe_path: str, T: int) -> list[int] | None:
    if not (os.path.exists(corpus_path) and os.path.exists(bpe_path)):
        return None
    from sim.bpe_tokenizer import BPETokenizer
    tok = BPETokenizer.load(bpe_path)
    txt = open(corpus_path, encoding="utf-8", errors="ignore").read(T * 12)   # bounded read, ~chars/token headroom
    ids = tok.encode(txt)
    return ids[:T] if len(ids) >= T else None


# ═══════════════════════════════════════════════════════════════════════════════ on-bridge circuit + settle ══
def build_onbridge_circuit(D: int, n_norm: int = 8, seed: int = 42, sigma: float = EPS, gain: float = 1.0,
                            tau_ms: float = 8.0):
    """Instantiate the DESIGN doc Sec 3 circuit: ONE norm-neuron pool (shunt_norm_source) shunting TWO
    D-neuron read sub-pools, read_on/read_off (shunt_norm_read), realizing the signed num/den division via
    `enable_shunt_norm_pool` (sim/config.py + sim/regions.py + sim/bridge.py). Returns
    (bridge, norm_idx, on_idx, off_idx). A zero-weight norm->read_on/off pathway is declared to sidestep a
    PRE-EXISTING, unrelated sim/bridge.py connectivity-fallback bug that fires when a brain-region config
    declares ZERO region_pathways (`UnboundLocalError: profile_name_for_conn`; flagged as a residual, not
    fixed here) -- real synapse objects exist so the fallback is never hit, but weight_mean=0.0 means their
    conductance contribution is exactly zero, so read_on/off's dynamics are driven ONLY by (a) their own
    injected `num` current and (b) the shared shunt-norm divisor, never a literal synapse from norm."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False            # a settle burst is a read-out, not a learning episode (DESIGN Sec 3b)
    cfg.enable_homeostasis = False     # keep the divisive-gain measurement free of a confounding threshold drift
    cfg.brain_regions = [
        BrainRegion(name="norm", n_neurons=n_norm, exc_fraction=1.0, internal_density=0.0,
                    shunt_norm_source=True),
        BrainRegion(name="read_on", n_neurons=D, exc_fraction=1.0, internal_density=0.0,
                    shunt_norm_read=True),
        BrainRegion(name="read_off", n_neurons=D, exc_fraction=1.0, internal_density=0.0,
                    shunt_norm_read=True),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="norm", to_region="read_on", density=0.1, weight_mean=0.0, weight_jitter=0.0),
        RegionPathway(from_region="norm", to_region="read_off", density=0.1, weight_mean=0.0, weight_jitter=0.0),
    ]
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.ou_seed = seed
    cfg.heterogeneity_seed = seed
    cfg.enable_shunt_norm_pool = True
    cfg.shunt_norm_sigma = sigma
    cfg.shunt_norm_gain = gain
    cfg.shunt_norm_rate_tau_ms = tau_ms
    rt = RuntimeState()
    rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=rt, gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    norm_idx = np.asarray(bridge.region_manager.indices("norm"))
    on_idx = np.asarray(bridge.region_manager.indices("read_on"))
    off_idx = np.asarray(bridge.region_manager.indices("read_off"))
    return bridge, norm_idx, on_idx, off_idx


def settle_read(D: int, num_vec: np.ndarray, den_scalar: float, *, n_norm: int = 8, seed: int = 42,
                 sigma: float = EPS, gain: float = 1.0, tau_ms: float = 8.0,
                 n_settle_steps: int = 40, warmup_frac: float = 0.5,
                 num_to_pA_gain: float = 60.0, den_to_pA_gain: float = 400.0) -> tuple[np.ndarray, float]:
    """Inject ONE (num_vec, den_scalar) pair, settle, and read back the on-bridge signed rate. Builds a
    fresh circuit per call (simple + obviously independent across trials -- not the performance-optimal
    choice for a large sweep; see the honest residual in main()'s docstring about batching many trials
    into one bridge via parallel region-copies, named but not built here). Returns (on_bridge_read [D],
    settled_den_ema [scalar])."""
    bridge, norm_idx, on_idx, off_idx = build_onbridge_circuit(
        D, n_norm=n_norm, seed=seed, sigma=sigma, gain=gain, tau_ms=tau_ms)
    on_drive = np.maximum(num_vec, 0.0) * num_to_pA_gain
    off_drive = np.maximum(-num_vec, 0.0) * num_to_pA_gain
    bridge.cp_external_input_current[on_idx] = on_drive
    bridge.cp_external_input_current[off_idx] = off_drive
    bridge.cp_external_input_current[norm_idx] = float(den_scalar) * den_to_pA_gain
    measure_start = int(n_settle_steps * warmup_frac)
    on_count = np.zeros(D, dtype=np.int64)
    off_count = np.zeros(D, dtype=np.int64)
    n_measured = 0
    for t in range(n_settle_steps):
        bridge._run_one_simulation_step()
        if t >= measure_start:
            fired = _to_host(bridge.cp_firing_states)
            on_count += fired[on_idx].astype(np.int64)
            off_count += fired[off_idx].astype(np.int64)
            n_measured += 1
    n_measured = max(n_measured, 1)
    on_rate = on_count / n_measured
    off_rate = off_count / n_measured
    den_ema_final = float(_to_host(bridge.cp_shunt_norm_den_ema)[0])
    return on_rate - off_rate, den_ema_final


# ═══════════════════════════════════════════════════════════════════════════════════════ anti-cheat sweeps ══
def divisive_vs_subtractive_check(D: int = 8, n_norm: int = 8, seed: int = 42, sigma: float = 0.05,
                                   gain: float = 20.0, n_settle_steps: int = 40,
                                   den_probe_scalars=(0.0, 0.3, 0.6, 1.0),
                                   num_probe_scale=(0.5, 1.0, 1.5, 2.0, 2.5), **settle_kw) -> dict:
    """THE LOAD-BEARING Tier-2 anti-cheat (DESIGN doc Sec 4/2, Holt & Koch 1997). At each `den` probe
    value, sweep `num`'s magnitude and fit the read pool's local f-I SLOPE (gain) and OFFSET (intercept).
    DIVISIVE: slope(den) ~ 1/(sigma+gain*den) -- equivalently 1/slope(den) is LINEAR in den (high R^2).
    SUBTRACTIVE (Holt & Koch): the offset shifts with den while slope stays ~flat. Both are fit; verdict
    picks whichever the data actually supports (an honest measurement, not a foregone conclusion)."""
    probe_dir = np.zeros(D); probe_dir[0] = 1.0     # a single probe channel is enough to trace the f-I curve
    slopes, intercepts, dens_measured = [], [], []
    for den in den_probe_scalars:
        rates, den_ema = [], []
        for s in num_probe_scale:
            r, de = settle_read(D, probe_dir * s, den, n_norm=n_norm, seed=seed, sigma=sigma, gain=gain,
                                 n_settle_steps=n_settle_steps, **settle_kw)
            rates.append(r[0])
            den_ema.append(de)
        A = np.vstack([num_probe_scale, np.ones(len(num_probe_scale))]).T
        slope, intercept = np.linalg.lstsq(A, np.asarray(rates), rcond=None)[0]
        slopes.append(float(slope)); intercepts.append(float(intercept)); dens_measured.append(float(np.mean(den_ema)))
    slopes = np.asarray(slopes); dens_measured = np.asarray(dens_measured)
    safe_slopes = np.where(np.abs(slopes) < 1e-9, np.sign(slopes) * 1e-9 + 1e-9, slopes)
    inv_slope = 1.0 / safe_slopes
    if len(dens_measured) >= 3 and np.ptp(dens_measured) > 1e-9:
        r_lin = float(np.corrcoef(dens_measured, inv_slope)[0, 1]) ** 2
        r_flat_slope = 1.0 - (np.var(slopes) / (np.var(slopes) + 1e-12)) if np.var(slopes) > 0 else 1.0
        offset_range = float(np.ptp(intercepts))
    else:
        r_lin, r_flat_slope, offset_range = float("nan"), float("nan"), float("nan")
    verdict = "UNDEFINED"
    if not np.isnan(r_lin):
        verdict = "DIVISIVE" if r_lin >= DIVISIVE_R2_MIN else "SUBTRACTIVE_OR_MIXED"
    return {
        "den_probe_scalars": list(den_probe_scalars), "den_measured": dens_measured.tolist(),
        "num_probe_scale": list(num_probe_scale), "slopes": slopes.tolist(), "intercepts": intercepts,
        "linear_fit_r2_1_over_slope_vs_den": r_lin, "offset_range_across_den": offset_range,
        "verdict": verdict,
    }


def sigma_domination_check(D: int = 8, n_norm: int = 8, seed: int = 42, gain: float = 20.0,
                            sigma_sweep=(1e-6, 0.05, 1.0, 1e6), n_settle_steps: int = 40,
                            num_probe_scale: float = 1.5, **settle_kw) -> dict:
    """The other named Tier-2 anti-cheat (DESIGN doc Sec 4, "the clamp owned 97% of the effect" trap): at
    each sigma, compare a SILENT norm-source (den~noise-floor) against a DRIVEN one (den well above it),
    same num-probe. A den-sensitive sigma shows a LOW ratio (driven << silent); a sigma-DOMINATED regime
    shows the ratio COLLAPSE toward 1.0 (den stopped mattering) -- verified analytically + empirically
    (see tests/test_shunt_norm_pool.py) for this primitive's r=x/(sigma+gain*den) shape."""
    probe = np.zeros(D); probe[0] = num_probe_scale
    ratios = []
    for sigma in sigma_sweep:
        r_silent, _ = settle_read(D, probe, 0.0, n_norm=n_norm, seed=seed, sigma=sigma, gain=gain,
                                   n_settle_steps=n_settle_steps, **settle_kw)
        r_driven, _ = settle_read(D, probe, 1.0, n_norm=n_norm, seed=seed, sigma=sigma, gain=gain,
                                   n_settle_steps=n_settle_steps, **settle_kw)
        ratio = float(r_driven[0] / r_silent[0]) if abs(r_silent[0]) > 1e-9 else float("nan")
        ratios.append(ratio)
    return {"sigma_sweep": list(sigma_sweep), "driven_over_silent_ratio": ratios}


# ═══════════════════════════════════════════════════════════════════════════════════ mechanism fidelity (full) ══
def mechanism_fidelity_eval(seed: int, ckpt_tmpl: str, corpus_path: str, bpe_path: str, n_tokens: int,
                             n_trials: int, n_settle_steps: int, sigma: float, gain: float, tau_ms: float,
                             num_to_pA_gain: float, den_to_pA_gain: float, n_norm: int) -> dict:
    """FULL MODE (routed to the GPU queue; NOT run by this file's authors -- see the module docstring's
    honest residual on margin_vs_trigram). Loads real layer-0 weights, replays real (or, absent a local
    corpus, Zipfian-synthetic -- clearly labeled) token statistics, and for a SAMPLE of `n_trials` token
    positions (capped for tractability; see the batching residual in settle_read's docstring) compares the
    on-bridge settled read to the exact num/(den+eps). Depth-bucketed (shallow = position < n_tokens/2,
    deep = position >= n_tokens/2) mirroring the d10-99 framing the rest of the linattn arc uses."""
    ckpt_path = ckpt_tmpl.format(seed=seed)
    weights = load_layer0_weights(ckpt_path)
    D = weights["D"]
    real_ids = load_real_token_ids(corpus_path, bpe_path, n_tokens)
    corpus_source = "real_tinystories" if real_ids is not None else "synthetic_zipf"
    token_ids = real_ids if real_ids is not None else sample_synthetic_token_ids(
        weights["V"], n_tokens, np.random.default_rng(seed))
    num_seq, den_seq, exact_read_seq = host_layer0_num_den(weights, token_ids)

    # CALIBRATION NOTE (found empirically this session, reading the real seed42 checkpoint): den_seq/
    # num_seq from a REAL trained layer are O(1e2-1e4) (median den ~300, range ~27-5500 in a 256-token
    # sample) -- NOT the [0,1]-bounded firing-RATE-FRACTION scale `den_ema` naturally lives on. Feeding
    # raw real-scale values through num_to_pA_gain/den_to_pA_gain (calibrated against the anti-cheat
    # sweeps' own small synthetic probes, ~0.5-3.0) would silently saturate every trial. Fix: rescale
    # BOTH num and den by the SAME per-replay constant (den's own median) before injection -- this leaves
    # their RATIO (and hence exact_read_seq, already computed above from the UNSCALED values) unchanged
    # to extremely high precision, since eps=1e-6 is négligible next to den at both scales (den's min in
    # the sample above is ~27; eps/scale stays ~1e-9-scale after rescaling). So exact_read_seq is reused
    # as-is as the ground truth; only the ON-BRIDGE INJECTION is rescaled.
    den_scale = max(float(np.median(den_seq)), 1e-6)
    num_seq_scaled = num_seq / den_scale
    den_seq_scaled = den_seq / den_scale

    rng = np.random.default_rng(seed)
    trial_positions = sorted(rng.choice(len(token_ids), size=min(n_trials, len(token_ids)), replace=False).tolist())
    rel_err, cos_sim, depth_bucket = [], [], []
    for pos in trial_positions:
        onb_read, _den_ema = settle_read(
            D, num_seq_scaled[pos], den_seq_scaled[pos], n_norm=n_norm, seed=seed, sigma=sigma, gain=gain,
            tau_ms=tau_ms, n_settle_steps=n_settle_steps, num_to_pA_gain=num_to_pA_gain,
            den_to_pA_gain=den_to_pA_gain)
        exact = exact_read_seq[pos]
        denom = np.linalg.norm(exact) + 1e-9
        rel_err.append(float(np.linalg.norm(onb_read - exact) / denom))
        cos = float(np.dot(onb_read, exact) / ((np.linalg.norm(onb_read) + 1e-9) * denom))
        cos_sim.append(cos)
        depth_bucket.append("deep" if pos >= len(token_ids) // 2 else "shallow")
    depth_bucket = np.asarray(depth_bucket)
    rel_err = np.asarray(rel_err); cos_sim = np.asarray(cos_sim)

    def _bucket_stats(mask):
        if not mask.any():
            return None
        return {"n": int(mask.sum()), "mean_relative_error": float(rel_err[mask].mean()),
                "mean_cosine_sim": float(cos_sim[mask].mean())}

    return {
        "seed": seed, "ckpt_path": ckpt_path, "D": D, "corpus_source": corpus_source,
        "n_tokens_replayed": len(token_ids), "n_trials_settled": len(trial_positions), "den_scale": den_scale,
        "overall": {"mean_relative_error": float(rel_err.mean()) if len(rel_err) else None,
                    "mean_cosine_sim": float(cos_sim.mean()) if len(cos_sim) else None},
        "shallow": _bucket_stats(depth_bucket == "shallow"),
        "deep": _bucket_stats(depth_bucket == "deep"),
    }


# ═══════════════════════════════════════════════════════════════════════════════════════════════ GO gate ══
def go_gate(result: dict) -> tuple[bool | None, list[str]]:
    """Structural GO-gate checker for the DESIGN doc Sec 4 Tier-2 criteria. Returns (verdict, messages).
    verdict is True/False when fully decidable, None ("PARTIAL") when margin_vs_trigram has not yet been
    measured (the named, honest residual -- see module docstring) but the mechanism-level anti-cheats
    (which THIS runner does measure end to end) both pass."""
    msgs = []
    ok = True

    margin = result.get("margin_vs_trigram")
    margin_ok = None
    if margin is not None:
        per_seed = margin if isinstance(margin, list) else [margin]
        margin_ok = all(m >= MARGIN_MIN for m in per_seed)
        msgs.append(f"[{'PASS' if margin_ok else 'FAIL'}] margin_vs_trigram >= {MARGIN_MIN}: {per_seed}")
        ok = ok and margin_ok
    else:
        msgs.append("[PENDING] margin_vs_trigram NOT measured (needs the full-stack --linattn-div "
                    "integration named in this runner's module docstring, after Tier-1's checkpoint-eval "
                    "capability lands) -- mechanism-level checks below stand on their own.")

    dvs = result.get("divisive_vs_subtractive", {})
    r2 = dvs.get("linear_fit_r2_1_over_slope_vs_den")
    dvs_ok = (r2 is not None) and not (isinstance(r2, float) and np.isnan(r2)) and r2 >= DIVISIVE_R2_MIN
    msgs.append(f"[{'PASS' if dvs_ok else 'FAIL'}] divisive-not-subtractive: 1/slope~den R^2={r2} "
                f"(need >= {DIVISIVE_R2_MIN}), verdict={dvs.get('verdict')}")
    ok = ok and dvs_ok

    sd = result.get("sigma_domination", {})
    ratios = sd.get("driven_over_silent_ratio", [])
    sweep = sd.get("sigma_sweep", [])
    sd_ok = False
    if ratios and sweep:
        op_ratio = ratios[len(ratios) // 2] if len(ratios) > 2 else ratios[0]  # a mid-sweep "operating" sigma
        collapse_ratio = ratios[-1]  # the sweep's largest sigma, expected to collapse toward 1.0
        sd_ok = (op_ratio < SIGMA_DOMINATION_MAX_RATIO) and (
            abs(collapse_ratio - 1.0) < (1.0 - SIGMA_DOMINATION_COLLAPSE_MIN) or collapse_ratio >= SIGMA_DOMINATION_COLLAPSE_MIN)
        msgs.append(f"[{'PASS' if sd_ok else 'FAIL'}] sigma-domination: operating ratio={op_ratio:.3f} "
                    f"(need < {SIGMA_DOMINATION_MAX_RATIO}), huge-sigma collapse ratio={collapse_ratio:.3f}")
    else:
        msgs.append("[FAIL] sigma-domination: no sweep data")
    ok = ok and sd_ok

    if margin is None:
        return (None, msgs) if (dvs_ok and sd_ok) else (False, msgs)
    return ok, msgs


# ═══════════════════════════════════════════════════════════════════════════════════════════ smoke / full ══
def run_smoke() -> dict:
    """TINY, checkpoint-independent plumbing check: build the circuit, settle a couple of synthetic
    (num,den) pairs, run both anti-cheat sweeps at reduced scale/steps. Confirms imports/parses/starts a
    step -- NOT a scientific verdict (too few steps/neurons for a clean signal; see the printed caveat)."""
    t0 = time.time()
    D = 8
    rng = np.random.default_rng(0)
    num_probe = rng.normal(size=D)
    on_read, den_ema = settle_read(D, num_probe, 0.5, n_norm=4, seed=0, sigma=0.05, gain=20.0,
                                    n_settle_steps=20, num_to_pA_gain=60.0, den_to_pA_gain=400.0)
    exact = num_probe / (0.5 + EPS)
    dvs = divisive_vs_subtractive_check(D=D, n_norm=4, seed=0, sigma=0.05, gain=20.0, n_settle_steps=20,
                                         den_probe_scalars=(0.0, 0.5, 1.0), num_probe_scale=(1.0, 2.0, 3.0))
    sd = sigma_domination_check(D=D, n_norm=4, seed=0, gain=20.0, sigma_sweep=(0.05, 1e6), n_settle_steps=20)
    result = {
        "mode": "smoke", "wall_s": round(time.time() - t0, 2),
        "one_pair_probe": {"on_bridge_read_sample": on_read[:3].tolist(),
                            "exact_read_sample": exact[:3].tolist(), "den_ema_settled": den_ema},
        "divisive_vs_subtractive": dvs, "sigma_domination": sd,
        "note": "SMOKE SCALE (D=8, ~20 settle steps) -- plumbing check only, not the full de-risk's GO gate.",
    }
    return result


def run_full(seeds, ckpt_tmpl, corpus_path, bpe_path, n_tokens, n_trials, n_settle_steps,
             sigma, gain, tau_ms, num_to_pA_gain, den_to_pA_gain, n_norm,
             dvs_D, sigma_sweep) -> dict:
    """The real Tier-2 de-risk. NOT run by this task (see module docstring); routed to the GPU queue after
    the live verification frees it. Per-seed mechanism fidelity (needs the checkpoint) + ONE shared
    checkpoint-independent anti-cheat pair (divisive-vs-subtractive, sigma-domination -- these do not
    depend on the seed's checkpoint, so they are computed once, not per seed)."""
    per_seed = []
    missing_ckpts = []
    for seed in seeds:
        ckpt_path = ckpt_tmpl.format(seed=seed)
        if not os.path.exists(ckpt_path):
            missing_ckpts.append(ckpt_path)
            continue
        per_seed.append(mechanism_fidelity_eval(
            seed, ckpt_tmpl, corpus_path, bpe_path, n_tokens, n_trials, n_settle_steps,
            sigma, gain, tau_ms, num_to_pA_gain, den_to_pA_gain, n_norm))
    dvs = divisive_vs_subtractive_check(D=dvs_D, n_norm=n_norm, seed=seeds[0], sigma=sigma, gain=gain,
                                         n_settle_steps=n_settle_steps)
    sd = sigma_domination_check(D=dvs_D, n_norm=n_norm, seed=seeds[0], gain=gain, sigma_sweep=sigma_sweep,
                                 n_settle_steps=n_settle_steps)
    return {
        "mode": "full", "mechanism_fidelity_per_seed": per_seed, "missing_checkpoints": missing_ckpts,
        "divisive_vs_subtractive": dvs, "sigma_domination": sd,
        "margin_vs_trigram": None,  # HONEST RESIDUAL -- see module docstring
        "config": {"sigma": sigma, "gain": gain, "tau_ms": tau_ms, "num_to_pA_gain": num_to_pA_gain,
                   "den_to_pA_gain": den_to_pA_gain, "n_norm": n_norm, "n_settle_steps": n_settle_steps,
                   "n_tokens": n_tokens, "n_trials": n_trials},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="tiny, checkpoint-independent plumbing check only")
    ap.add_argument("--check-go", type=str, default=None, help="re-check the GO gate against an existing result JSON")
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--ckpt-tmpl", type=str, default=DEFAULT_CKPT_TMPL)
    ap.add_argument("--corpus", type=str, default=DEFAULT_CORPUS)
    ap.add_argument("--bpe", type=str, default=DEFAULT_BPE_PATH)
    ap.add_argument("--n-tokens", type=int, default=512, help="tokens replayed host-side per seed")
    ap.add_argument("--n-trials", type=int, default=32, help="token positions actually settled on-bridge per seed")
    ap.add_argument("--n-settle-steps", type=int, default=40)
    ap.add_argument("--n-norm", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--gain", type=float, default=20.0)
    ap.add_argument("--tau-ms", type=float, default=8.0)
    ap.add_argument("--num-to-pA-gain", dest="num_to_pA_gain", type=float, default=60.0)
    ap.add_argument("--den-to-pA-gain", dest="den_to_pA_gain", type=float, default=400.0)
    ap.add_argument("--dvs-D", dest="dvs_D", type=int, default=8, help="probe-channel width for the anti-cheat sweeps")
    ap.add_argument("--sigma-sweep", type=str, default="1e-6,0.05,1.0,1e6")
    ap.add_argument("--json", "--out", dest="out", type=str,
                     default="research/findings/raw/_linattn_shunt_gain_tier2_onbridge.json")
    args = ap.parse_args()

    if args.check_go:
        with open(args.check_go) as f:
            result = json.load(f)
        verdict, msgs = go_gate(result)
        for m in msgs:
            print(m)
        print(f"GO-GATE VERDICT: {'GO' if verdict else ('PARTIAL' if verdict is None else 'NO-GO')}")
        sys.exit(0 if verdict else (0 if verdict is None else 1))

    if args.smoke:
        result = run_smoke()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        smoke_out = args.out.replace(".json", "_SMOKE.json")
        with open(smoke_out, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        print(f"\nSMOKE OK -- wrote {smoke_out}. This is a plumbing check, not a GO verdict.")
        return

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sigma_sweep = [float(s) for s in args.sigma_sweep.split(",") if s.strip()]
    result = run_full(seeds, args.ckpt_tmpl, args.corpus, args.bpe, args.n_tokens, args.n_trials,
                       args.n_settle_steps, args.sigma, args.gain, args.tau_ms,
                       args.num_to_pA_gain, args.den_to_pA_gain, args.n_norm, args.dvs_D, sigma_sweep)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    verdict, msgs = go_gate(result)
    for m in msgs:
        print(m)
    print(f"GO-GATE VERDICT: {'GO' if verdict else ('PARTIAL' if verdict is None else 'NO-GO')}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
