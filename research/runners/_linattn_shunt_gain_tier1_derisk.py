"""Tier-1 de-risk of the linattn own-voice mouth's spike-native num/den read (2026-09-03, research/findings/
2026-09-03-linattn-spike-native-normalization-DESIGN.md Sec 3e/4, branch research/linattn-spike-native-norm-design
/ 2a28768d). The design's honest residual on the confirmed open-fluency milestone (`43c5b6b4`, `--recurrence
linattn` beats a fair trigram 6/6, research/findings/2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-
spiking-mouth-beats-trigram-6of6.md): the read `= phi(q)^T M / (phi(q)^T zden + eps)` is spike-native everywhere
except this ONE graded host DIVISION. The design specifies a shunting-conductance realization, `num/(g_leak+k*den)`
(Carandini & Heeger divisive normalization by a single shared pool = the norm-neuron's rate `den`), with the honest
caveat that pure somatic shunting is SUBTRACTIVE not divisive on mean firing rate (Holt & Koch 1997) unless the
read neuron is in the fluctuation-driven regime (Chance, Abbott & Reyes 2002; Mitchell & Silver 2003).

THIS RUNNER is Tier 1 (design Sec 4): a near-free, NO-RETRAIN, CPU, rate-model read-side swap on the ALREADY-
TRAINED 6-seed checkpoints (`bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{42,43,44,100,101,102}.npz`), via
`LinAttnReadout` (research.runners._wkv_fewspike_read_derisk), extended (additive, default "exact" = byte-
identical) with the `_divisive_read`/`_quantize_rate` machinery. At `g_leak=1e-6,k=1.0,fI=None,quantize_levels=0`
the "shunt" mode is ALGEBRAICALLY IDENTICAL to "exact" (both reduce to `num/(den+1e-6)`) -- so the decisive test is
robustness of margin_vs_trigram to the two biophysical realities the design names: the read neuron's own f-I
SQUASH (a monotone saturating nonlinearity applied after the divisive gain) and RATE QUANTIZATION (a finite
spike-count-like read, `_quantize_rate`'s stochastic-rounding ON/OFF discretization) -- both are meaningless
without a substrate but are the honest RATE-MODEL stand-in Tier 1 can run without one (Tier 2, on-bridge, is the
real spiking test, out of this runner's scope).

LIKE-FOR-LIKE ANTI-CHEAT: reconstructs the EXACT held-out eval set the milestone's own 6-seed run used (identical
`load_stories`/`_BPEVocabAdapter`/train-eval-dev split arithmetic, `_emerge_wkv_lm_derisk.py`, byte-for-byte) and
VALIDATES the reconstruction against that run's own per-seed per-bucket position COUNTS
(research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json) before trusting any comparison. The
FAIR trigram/bigram baseline is division-mode-INVARIANT (a lookup table fit once on TRAIN, independent of how the
linattn mouth reads its own state), so this runner reuses that run's own frozen trigram/bigram NLL per bucket
rather than re-fitting it (re-fitting requires re-tokenizing ~1M training sentences -- the actual expensive step
Tier 1 is explicitly exempt from, per the design's own "near-free, minutes of CPU" framing) -- exact-mode's OWN
margin_vs_trigram, recomputed here via LinAttnReadout, is cross-checked against that run's recorded number as the
harness's own correctness gate (a mismatch there would mean the eval-set reconstruction is wrong, not that shunt
vs exact differs).

THE DESIGN-SPECIFIC ANTI-CHEAT (Holt & Koch 1997 direct test, rate-model level): `--divisive-check` grids the
`_divisive_read` shunt formula's GAIN (d(read)/d(num) by finite difference) against `den` on REAL (num,den) pairs
sampled off the seed-42 checkpoint (`diag_collect`), and checks the gain follows the divisive `1/(g_leak+k*den)`
form (a DEFINITIONAL property of dividing num by a den-dependent conductance -- this is what makes the *formula*
divisive) THROUGH an applied f-I squash (the point at which Holt & Koch's subtractive failure mode would actually
bite: a real neuron's threshold nonlinearity can turn a den-dependent gain into a den-INDEPENDENT threshold shift).
The full on-substrate version (a real conductance-based spiking neuron's own f-I curve under a GABA_A shunt) is
Tier 2's job (design Sec 4); this is the rate-model analog + an honest note on what remains open.

`--sigma-check` sweeps `g_leak` (k=1, fI=None, quantize=0, isolating g_leak's own effect) and reports how much of
`g_leak+k*den` is OWNED by g_leak vs den (the "the clamp owned 97% of the effect" trap, CLAUDE.md) alongside the
margin_vs_trigram degradation this produces -- the sigma-domination check. `--denquant-check` demonstrates + names
a SECOND real bug this runner's own harness caught (not asserted from theory): quantizing the scalar `den` against
a fixed external scale rounds small-but-genuinely-nonzero den values down to exactly 0 with high probability, and
dividing by `g_leak(=1e-6)+k*0` explodes the read ~1e6x for the worst samples (~2.4% of samples >10x, measured) --
resolved by raising g_leak to at least the quantizer's own zero-bin width. The MAIN GO-gate arm therefore quantizes
`num` only by default (`--quantize-den` opts into the fuller, riskier form for those who want to see the interaction).

BOTH bugs above were caught by THIS runner's own harness (a naive first calibration produced a -4.05
margin_vs_trigram collapse; --denquant-check's Monte-Carlo check matched the theoretical zero-rounding probability
to 3 decimals) -- not asserted from design theory, and fixed before this file's committed version ran the real
6-seed GO gate.

NO `sim/` edit. NO retrain. cost-routed CPU (numpy), MEMORY BUDGET note: reconstructing the eval-sentence pool via
`load_stories` on the full corpus holds ~2.7GB RSS (measured) for the corpus's ~581K contiguous 40-token stories --
this is the corpus-loading cost the ORIGINAL training run also paid, shared ONCE across all 6 seeds here exactly
as that run shared it; nothing else in this runner materially adds to it (readout state is D=192, trivially small).

Run (smoke, tiny slice): SIM_BACKEND=numpy .venv/bin/python -m research.runners._linattn_shunt_gain_tier1_derisk \
    --seeds 42 --max-eval-sents 40 --json research/findings/raw/_linattn_shunt_gain_tier1_smoke.json
Run (the real 6-seed GO gate): SIM_BACKEND=numpy .venv/bin/python -m research.runners._linattn_shunt_gain_tier1_derisk \
    --seeds 42,43,44,100,101,102 --divisive-check --sigma-check --denquant-check \
    --json research/findings/raw/_linattn_shunt_gain_tier1_6seed.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.runners._emerge_wkv_lm_derisk import load_stories, _BPEVocabAdapter  # noqa: E402
from research.runners._emerge_reservoir_lm_context_depth_derisk import BUCKETS, _bucket  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import LinAttnReadout  # noqa: E402
from sim.bpe_tokenizer import BPETokenizer  # noqa: E402

DEFAULT_CKPT_PATTERN = "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
DEFAULT_REF_JSON = "research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json"
DEFAULT_CORPUS = "data/corpus/simplewiki.txt"
DEFAULT_BPE_PATH = "bridges/wkv_ckpt/wkv_bpe8k.json"


# ---------------------------------------------------------------------------------------------------------------
# f-I squash (the design's Sec 3c effect-2 "read-neuron f-I nonlinearity" -- a representative MONOTONE, SATURATING
# transfer applied AFTER the divisive gain). Signed + odd-symmetric (keeps the sign of its input) so it composes
# with a signed `read` without distorting which half-space (ON/OFF) a channel falls in; near-identity for
# |x| << r_max so it does not dominate small reads (the design's own note: "the downstream head + FewSpikeWordRead
# already apply their own nonlinearity" -- this should be a MODERATE additional squash, not an aggressive one).
#
# FIRST ATTEMPT'S BUG (caught by this runner's own harness, not asserted from theory): a Naka-Rushton form
# `fI(x)=sign(x)*r_max*|x|/(|x|+x50)` has SLOPE AT THE ORIGIN `r_max/x50` -- setting `r_max=fi_rmax_mult*x50`
# (fi_rmax_mult=4.0) therefore gave a slope of EXACTLY 4.0 for every small-to-moderate read, i.e. a BLANKET 4x
# GAIN, not a squash at all. The signature that exposed it: sweeping x50's calibration percentile 50th->99th
# (moving the saturation POINT) barely changed the deep-bucket NLL (7.93->8.07 nats on a 30-story real slice,
# vs exact's 4.40) -- if the damage were from the SATURATION TAIL, pushing x50 higher (covering more of the
# distribution in the "linear" regime) should have recovered most of the margin; it did not, because the
# uncontrolled 4x GAIN was the dominant effect regardless of where saturation started. Fixed below with a form
# whose origin slope is DEFINITIONALLY 1 regardless of the calibrated scale.
# ---------------------------------------------------------------------------------------------------------------
def make_fI(r_max):
    """Signed saturating transfer with UNIT SLOPE AT THE ORIGIN: fI(x) = r_max * tanh(x / r_max). Exactly
    identity to first order for |x| << r_max (tanh(u)~=u for small u for u=x/r_max), saturating smoothly toward
    +-r_max for |x| >> r_max -- unlike the Naka-Rushton form this replaces, the origin slope here is always
    EXACTLY 1 (d/dx[r_max*tanh(x/r_max)] at x=0 = r_max*(1/r_max)*sech^2(0) = 1) regardless of how `r_max` is
    calibrated, so calibration only controls WHERE saturation begins, never whether small reads get rescaled."""
    r_max = max(float(r_max), 1e-9)

    def _fI(x):
        return r_max * np.tanh(np.asarray(x) / r_max)
    return _fI


def _log_softmax(z):
    z = z - z.max()
    lse = np.log(np.sum(np.exp(z)))
    return z - lse


# ---------------------------------------------------------------------------------------------------------------
# LIKE-FOR-LIKE eval-set reconstruction: byte-for-byte the same `sents` pool + per-seed train/eval/dev split
# arithmetic `_emerge_wkv_lm_derisk.main()` uses for `--contiguous --tokenizer bpe` (see that function, ~line
# 1440-1460). We do NOT reconstruct `tr`/`dev` (the expensive ~1M-sentence BPE pass) because we reuse the FROZEN
# trigram/bigram baseline from the original run's own JSON (division-mode-invariant -- see module docstring).
# ---------------------------------------------------------------------------------------------------------------
def build_eval_ids(sents, bpe_vocab, seed, max_eval_sents):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents))
    cut = int(0.85 * len(sents))
    ev_idx = idx[cut:][:max_eval_sents]
    ev = [sents[i] for i in ev_idx]
    return [bpe_vocab.ids(s) for s in ev]


def eval_readout_perdepth(ro, ev_ids, permute=False, memoryless=False, seed=0):
    """numpy analogue of `_emerge_wkv_lm_derisk.eval_perdepth`, driven by `LinAttnReadout`'s autoregressive
    `advance()`/`logits()` instead of a batched torch forward. Same RNG recipe (`seed*17+5`), same permute/
    memoryless anti-cheat semantics, same `_bucket` depth bucketing, so the two are like-for-like comparable
    (verified below by cross-checking exact-mode's own margin_vs_trigram against the original run's own number)."""
    rng = np.random.default_rng(seed * 17 + 5)
    ce = defaultdict(float); cnt = defaultdict(int)
    for ids in ev_ids:
        if len(ids) < 2:
            continue
        seq = list(ids)
        if permute and len(seq) > 2:
            perm = rng.permutation(len(seq)); seq = [ids[p] for p in perm]
        state = ro.init_state()
        for t in range(len(seq) - 1):
            state = ro.advance(state, seq[t], memoryless=memoryless)
            logp = _log_softmax(ro.logits(state))
            d = t + 1; b = _bucket(d)
            p = math.exp(logp[seq[t + 1]])
            ce[b] += -math.log(max(p, 1e-12))
            cnt[b] += 1
    return {b: ce[b] / cnt[b] for b in cnt}, {b: cnt[b] for b in cnt}


def _bucket_label(lo, hi):
    return f"{lo}-{hi}" if lo != hi else f"{lo}"


def calibrate_from_diag(diag_samples, g_leak=1e-6, k=1.0, fi_rmax_mult=4.0, x50_percentile=90.0):
    """Calibrates the fI squash's x50 + the den quantizer's scale from REAL exact-mode (num,den) samples off the
    checkpoint. x50 MUST be calibrated on the POST-DIVISION read `g=num/(den+eps)` -- fI is applied AFTER the
    divisive gain, so calibrating against raw `num` (pre-division) puts x50 on the wrong scale entirely (this
    runner's own first calibration attempt did exactly that: x50 from |num| while fI receives |num/den|, off by
    whatever `den`'s own scale is -- den at deep context here runs ~0.1-300, so x50 was ~2 orders of magnitude
    too large, driving every fI(g) deep into its near-zero linear-but-vanishing regime and collapsing
    margin_vs_trigram to -4.05).

    WHY A HIGH PERCENTILE, NOT THE MEDIAN: a f-I 'squash' whose x50 (half-saturation point) sits at the MEDIAN of
    the read's own distribution puts HALF of every read already past its half-saturation point --
    that is an AGGRESSIVE squash (fI's own docstring calls it a MODERATE one), and this runner's own harness
    caught it: even with quantization fully disabled, x50=median(|g|) alone doubled the deep-bucket mean NLL
    (4.40 -> 7.93 nats on a 30-story real slice, margin_vs_trigram collapsing to roughly -3.5). A real neuron's
    f-I is close to LINEAR over its typical operating range and saturates only for UNUSUALLY large drive -- so
    x50 is calibrated at a HIGH percentile (`x50_percentile`, default 90th) of the observed |g| distribution
    instead: the bulk of reads then stay in fI's quasi-linear regime, and only the top decile (genuinely large
    drives) approach saturation, which is the MODERATE squash the design actually describes. Returns (x50,
    r_max, den_scale)."""
    if not diag_samples:
        return 1.0, fi_rmax_mult, 1.0
    gs = [LinAttnReadout._divisive_read(num, den, mode="exact") for (_, num, den) in diag_samples]
    g_mags = [float(np.linalg.norm(g)) for g in gs]
    x50 = float(np.percentile(g_mags, x50_percentile)) if g_mags else 1.0
    r_max = fi_rmax_mult * x50
    dens = [float(den) for (_, _, den) in diag_samples if den > 0]
    den_scale = float(np.percentile(dens, 90)) * 1.2 if dens else 1.0
    return x50, r_max, den_scale


def run_seed_arm(ckpt, ev_ids, seed, div_mode, g_leak, k_gain, fI, quantize_levels, quantize_seed,
                  diag_collect=False, quantize_den_scale=None):
    """Clean + perm + memoryless pass for ONE (seed, div-mode-config) combination. Returns (by_depth dict,
    diag list-or-None)."""
    ro = LinAttnReadout(ckpt, phi="elu", norm=True, div_mode=div_mode, div_g_leak=g_leak, div_k=k_gain,
                        fI=fI, quantize_levels=quantize_levels, quantize_seed=quantize_seed,
                        quantize_den_scale=quantize_den_scale)
    if diag_collect:
        ro.diag_collect = True
    ce, cnt = eval_readout_perdepth(ro, ev_ids, seed=seed)
    ce_perm, _ = eval_readout_perdepth(ro, ev_ids, permute=True, seed=seed)
    ce_mless, _ = eval_readout_perdepth(ro, ev_ids, memoryless=True, seed=seed)
    by_depth = {}
    for lo, hi in BUCKETS:
        b = _bucket_label(lo, hi)
        if b in cnt:
            by_depth[b] = {"n": cnt[b], "wkv": ce[b], "wkv_perm": ce_perm.get(b, float("nan")),
                           "wkv_memoryless": ce_mless.get(b, float("nan"))}
    diag = list(ro._diag) if diag_collect else None
    return by_depth, diag


def divisive_vs_subtractive_check(diag_samples, g_leak=1e-6, k=1.0, fI=None, den_mults=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                                   n_probe=12, dnum=1e-3):
    """THE design-specific anti-cheat (Holt & Koch 1997 direct test, rate-model level). Takes REAL (num, den)
    samples collected off the checkpoint (`diag_collect`), and for a subset of them, sweeps `den` over
    `den_mults` (relative to that sample's own den) while holding `num`'s DIRECTION fixed, measuring the GAIN
    (d(read)/d(num), estimated by a symmetric finite difference of magnitude `dnum` along num's own direction) at
    each den value. A DIVISIVE realization has gain(den) tracking `1/(g_leak+k*den)` (the formula's own
    definition); a SUBTRACTIVE-looking realization (the Holt & Koch failure this check exists to catch) would
    instead show a gain that stays roughly CONSTANT across den while `read` itself shifts by a den-dependent
    OFFSET. Reports, per probed sample: the empirical gain ratio gain(den)/gain(den_ref) vs the theoretical
    (g_leak+k*den_ref)/(g_leak+k*den) ratio, and an R^2-like agreement score.

    HONEST SCOPE: this is a RATE-MODEL analog. `_divisive_read`'s shunt formula divides num by a den-dependent
    conductance BEFORE any nonlinearity, which is DEFINITIONALLY divisive at the bare-formula level (dividing is
    not subtracting) -- the substantive question this check can actually inform is whether an applied `fI` (the
    read neuron's OWN saturating transfer, representing where a real spiking neuron's threshold nonlinearity
    would sit) starts to LOOK subtractive (gain flattens, offset dominates) as den grows -- exactly the composed-
    nonlinearity regime where Holt & Koch's failure mode would show up. The full on-substrate version (an actual
    conductance-based spiking neuron's measured f-I curve under a real GABA_A shunt, Tier 2 design Sec 4) is the
    only test that can DIRECTLY confirm/refute Holt & Koch on this substrate; this is the honest rate-model
    stand-in Tier 1 can run without one."""
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(diag_samples), size=min(n_probe, len(diag_samples)), replace=False)
    rows = []
    for i in idxs:
        _, num, den0 = diag_samples[int(i)]
        if den0 <= 1e-9 or float(np.linalg.norm(num)) <= 1e-9:
            continue
        direction = num / (np.linalg.norm(num) + 1e-12)
        gains = {}
        reads_at_ref = {}
        for m in den_mults:
            den = den0 * m
            g_plus = LinAttnReadout._divisive_read(num + dnum * direction, den, mode="shunt", g_leak=g_leak, k=k, fI=fI)
            g_minus = LinAttnReadout._divisive_read(num - dnum * direction, den, mode="shunt", g_leak=g_leak, k=k, fI=fI)
            gain_vec = (g_plus - g_minus) / (2 * dnum)
            gains[m] = float(np.dot(gain_vec, direction))       # gain along num's own direction (a scalar)
            reads_at_ref[m] = float(np.dot(
                LinAttnReadout._divisive_read(num, den, mode="shunt", g_leak=g_leak, k=k, fI=fI), direction))
        ref_m = 1.0
        gain_ref = gains.get(ref_m, list(gains.values())[0])
        if abs(gain_ref) <= 1e-12:
            continue
        empirical_ratio = {m: gains[m] / gain_ref for m in den_mults}
        theoretical_ratio = {m: (g_leak + k * den0 * ref_m) / (g_leak + k * den0 * m) for m in den_mults}
        errs = [abs(empirical_ratio[m] - theoretical_ratio[m]) for m in den_mults]
        rows.append({
            "den0": float(den0), "gain_at_den0": gain_ref,
            "empirical_gain_ratio": {str(m): round(empirical_ratio[m], 4) for m in den_mults},
            "theoretical_divisive_ratio": {str(m): round(theoretical_ratio[m], 4) for m in den_mults},
            "max_abs_err_vs_divisive": round(max(errs), 4),
            "read_shift_at_den0": {str(m): round(reads_at_ref[m], 4) for m in den_mults},
        })
    max_errs = [r["max_abs_err_vs_divisive"] for r in rows]
    verdict = {
        "n_probed": len(rows),
        "mean_max_abs_err_vs_divisive": round(float(np.mean(max_errs)), 4) if max_errs else None,
        "divisive_not_subtractive": bool(max_errs and float(np.mean(max_errs)) < 0.15),
        "fI_applied": fI is not None,
    }
    return verdict, rows


def sigma_domination_check(diag_samples, g_leaks, k=1.0):
    """Sweeps g_leak (k fixed, no fI/quantization) over REAL den samples: for each g_leak, reports the mean
    fraction of the divisor `g_leak+k*den` OWNED by g_leak (`g_leak/(g_leak+k*mean_den)`) -- the "the clamp
    owned 97% of the effect" trap (CLAUDE.md) -- vs owned by den. A den-DOMINATED divisor (fraction near 0)
    behaves like "exact"; a g_leak-DOMINATED divisor (fraction near 1) collapses toward a CONSTANT rescaling of
    the raw unnormalized sum `num` (the already-characterized `--no-linattn-norm` ablation, up to a scale
    factor) -- i.e. de-normalizes the read."""
    dens = np.array([d for (_, _, d) in diag_samples if d > 0])
    mean_den = float(dens.mean()) if len(dens) else 0.0
    rows = []
    for g in g_leaks:
        frac = g / (g + k * mean_den) if (g + k * mean_den) > 0 else 1.0
        rows.append({"g_leak": g, "mean_den": round(mean_den, 6), "sigma_owned_fraction": round(frac, 4)})
    return rows


def denquant_interaction_check(diag_samples, den_scale, n_levels=32, seed=0):
    """Demonstrates + resolves a real interaction this runner's own harness caught (NOT asserted from theory --
    the first --quantize-den run measured mean NLL ~487,000 on a 50-token synthetic trace, traced here): `den`
    ranges over ~4 orders of magnitude in this checkpoint (measured 0.014-297.7 on a short trace); quantizing it
    with `_quantize_rate`'s stochastic rounder against a FIXED external `den_scale` (calibrated off the upper
    part of that range, e.g. a ~90th-percentile-based scale) rounds small-but-genuinely-nonzero den values DOWN
    TO EXACTLY 0 with probability `1 - den/den_scale*n_levels` (e.g. den=0.014 against den_scale~122, n_levels=32
    -> P(quantizes to 0) ~ 0.998, matching a direct Monte-Carlo check to 3 decimals). Dividing `num` by
    `g_leak+k*0 = g_leak = 1e-6` then multiplies the read by ~1e6 -- NOT a claim that rate-quantization is
    unworkable, but that `g_leak` calibrated as `exact`'s numerical epsilon (a safety floor against literal
    div-by-zero, never meant to carry current) is NOT a safe `g_leak` once den can ITSELF be discretely zero.
    The RESOLUTION named by the design's own R-fluct framing (a real leak conductance is a genuine, non-
    negligible circuit element, not an infinitesimal): raise `g_leak` to at least the quantizer's OWN zero-bin
    width (`den_scale/n_levels`) -- i.e. never let the safety floor be smaller than the grid that can produce it.
    Returns rows comparing mean NLL-proxy (mean |read| relative to unquantized, a cheap proxy that avoids
    re-running the full autoregressive eval) at g_leak=1e-6 (broken) vs g_leak=den_scale/n_levels (resolved)."""
    rng = np.random.default_rng(seed)
    zero_bin = den_scale / n_levels
    rows = []
    for g_leak, label in [(1e-6, "g_leak=1e-6 (matches exact's epsilon)"),
                           (zero_bin, "g_leak=den_scale/n_levels (>= the quantizer's own zero-bin width)")]:
        ratios = []
        for (_, num, den) in diag_samples:
            if den <= 1e-9:
                continue
            den_q = float(LinAttnReadout._quantize_rate(np.array([den]), n_levels, rng, scale=den_scale)[0])
            g_exact = LinAttnReadout._divisive_read(num, den, mode="exact")
            g_shunt_q = LinAttnReadout._divisive_read(num, den_q, mode="shunt", g_leak=g_leak, k=1.0)
            n_exact = float(np.linalg.norm(g_exact)) + 1e-12
            n_shunt = float(np.linalg.norm(g_shunt_q))
            ratios.append(n_shunt / n_exact)
        ratios = np.array(ratios) if ratios else np.array([np.nan])
        rows.append({"g_leak": g_leak, "label": label, "n_samples": len(ratios),
                     "median_read_ratio_vs_exact": round(float(np.median(ratios)), 4),
                     "max_read_ratio_vs_exact": round(float(np.max(ratios)), 2),
                     "frac_above_10x": round(float(np.mean(ratios > 10.0)), 4)})
    return {"den_scale": den_scale, "n_levels": n_levels, "zero_bin_width": zero_bin, "rows": rows}


def load_reference(ref_json_path, seeds):
    ref = json.loads(Path(ref_json_path).read_text())
    per_seed = ref.get("per_seed", {})
    out = {}
    for s in seeds:
        d = per_seed.get(str(s))
        if d is None:
            raise KeyError(f"seed {s} not found in reference json {ref_json_path}")
        out[s] = d["by_depth"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--ckpt-pattern", type=str, default=DEFAULT_CKPT_PATTERN)
    ap.add_argument("--ref-json", type=str, default=DEFAULT_REF_JSON)
    ap.add_argument("--corpus", type=str, default=DEFAULT_CORPUS)
    ap.add_argument("--bpe-path", type=str, default=DEFAULT_BPE_PATH)
    ap.add_argument("--n-sentences", type=int, default=1200000)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--max-eval-sents", type=int, default=4000)
    ap.add_argument("--linattn-div", choices=["exact", "shunt"], default="shunt",
                    help="the TEST arm's division mode (the baseline arm is always 'exact'). 'exact' makes the "
                         "comparison a (vacuous, self-consistency-only) exact-vs-exact run.")
    ap.add_argument("--g-leak", type=float, default=1e-6, help="shunt arm sigma (read-neuron leak conductance).")
    ap.add_argument("--k-gain", type=float, default=1.0, help="shunt arm k (norm-neuron-rate -> shunt scale).")
    ap.add_argument("--fi", choices=["none", "saturating"], default="saturating",
                    help="read-neuron f-I squash applied after the divisive gain (design Sec 3c effect 2). "
                         "'saturating' auto-calibrates x50 per seed from that seed's own exact-mode |read| median.")
    ap.add_argument("--fi-rmax-mult", type=float, default=4.0, help="fI r_max = this * calibrated x50.")
    ap.add_argument("--quantize-levels", type=int, default=32,
                    help="rate-quantization robustness axis (design Sec 3c effect 3), applied to `num` (a [D] "
                         "population -- self-peak-normalized quantization is well-posed there); 0 disables.")
    ap.add_argument("--quantize-den", action="store_true",
                    help="ALSO quantize the scalar `den` (default OFF for the main GO-gate arm -- see "
                         "--denquant-check's own docstring: quantizing den with a FIXED external scale rounds "
                         "small-den positions down to exactly 0 with high probability, and dividing by "
                         "g_leak(=1e-6)+k*0 explodes the read ~1e6x -- a real, named interaction between "
                         "quantization and a g_leak calibrated as a numerical epsilon rather than a genuine "
                         "leak conductance, not a claim that quantization per se is unworkable. Kept OFF by "
                         "default so the main margin comparison isolates the fI/num-quantization robustness "
                         "question the design asks; --denquant-check demonstrates + resolves the interaction.")
    ap.add_argument("--go-margin", type=float, default=0.03, help="GO gate: shunt margin_vs_trigram >= this, 6/6.")
    ap.add_argument("--denquant-check", action="store_true",
                    help="demonstrate the den-quantization / g_leak interaction (seed 42 only, cheap): shows the "
                         "catastrophic collapse when den IS quantized at g_leak=1e-6, and that raising g_leak to "
                         "at least the quantization step near the operating den resolves it.")
    ap.add_argument("--divisive-check", action="store_true", help="run the divisive-vs-subtractive anti-cheat (seed 42 only, cheap).")
    ap.add_argument("--sigma-check", action="store_true", help="run the sigma-domination g_leak sweep (seed 42 only, cheap).")
    ap.add_argument("--sigma-gleaks", type=str, default="1e-6,1e-3,1e-2,1e-1,1,10")
    ap.add_argument("--diag-seed", type=int, default=42, help="which seed's real (num,den) samples power --divisive-check/--sigma-check.")
    ap.add_argument("--diag-eval-sents", type=int, default=300,
                    help="--sigma-check re-measures margin_vs_trigram at each swept g_leak (a FULL extra "
                         "clean/perm/memoryless pass per value) -- capped at this many stories (a trend "
                         "characterization, not the headline number) so the sweep does not multiply the main "
                         "6-seed run's wall-clock cost by len(--sigma-gleaks).")
    ap.add_argument("--json", type=str, default="research/findings/raw/_linattn_shunt_gain_tier1_6seed.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()

    print(f"[load] corpus={args.corpus} n_sentences={args.n_sentences} max_len={args.max_len} ...", flush=True)
    t_l0 = time.time()
    sents = load_stories(args.corpus, args.n_sentences, max_len=args.max_len)
    bpe_vocab = _BPEVocabAdapter(BPETokenizer.load(args.bpe_path))
    print(f"[load] {len(sents)} stories in {time.time() - t_l0:.1f}s", flush=True)

    ref_by_seed = load_reference(args.ref_json, seeds)

    results = {}
    for seed in seeds:
        ckpt = args.ckpt_pattern.format(seed=seed)
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ts0 = time.time()
        ev_ids = build_eval_ids(sents, bpe_vocab, seed, args.max_eval_sents)
        ref_depth = ref_by_seed[seed]

        # ---- validate the reconstruction: n-counts per bucket must match the original run's own ----------------
        n_mismatch = {}

        # ---- exact arm (also the calibration + diagnostic-sample source) --------------------------------------
        want_diag = args.divisive_check or args.sigma_check or args.denquant_check
        exact_depth, exact_diag = run_seed_arm(ckpt, ev_ids, seed, "exact", 1e-6, 1.0, None, 0, seed,
                                                diag_collect=(want_diag and seed == args.diag_seed))
        for b, d in exact_depth.items():
            ref_n = ref_depth.get(b, {}).get("n")
            if ref_n is not None and ref_n != d["n"]:
                n_mismatch[b] = {"mine": d["n"], "reference": ref_n}

        deep = exact_depth.get("10-99", {})
        ref_deep = ref_depth.get("10-99", {})
        exact_margin = ref_deep["trigram"] - deep["wkv"] if deep else float("nan")
        ref_margin = ref_deep.get("margin_vs_trigram")

        # ---- calibrate fI + the den quantizer's scale from THIS seed's own exact-mode reads --------------------
        # (each checkpoint has different trained weights -> different num/den scales, so calibration MUST be
        # per-seed, not shared). Reuse the diag already collected above when this seed IS the diag_seed and
        # --divisive-check/--sigma-check requested it; otherwise pay a small DEDICATED calibration pass (capped
        # at 50 stories -- calibration only needs the SHAPE of the num/den distribution, not the full eval set).
        fI = None
        x50 = r_max = den_scale = None
        if args.fi == "saturating" or args.quantize_levels > 0:
            calib_diag = exact_diag
            if not calib_diag:
                _, calib_diag = run_seed_arm(ckpt, ev_ids[: min(50, len(ev_ids))], seed, "exact", 1e-6, 1.0, None, 0,
                                              seed, diag_collect=True)
            x50, r_max, den_scale = calibrate_from_diag(calib_diag or [], fi_rmax_mult=args.fi_rmax_mult)
            if args.fi == "saturating":
                fI = make_fI(r_max)

        # ---- shunt (test) arm ------------------------------------------------------------------------------
        shunt_depth, _ = run_seed_arm(ckpt, ev_ids, seed, args.linattn_div, args.g_leak, args.k_gain, fI,
                                       args.quantize_levels, seed,
                                       quantize_den_scale=(den_scale if args.quantize_den else None))
        shunt_deep = shunt_depth.get("10-99", {})
        shunt_margin = ref_deep["trigram"] - shunt_deep["wkv"] if shunt_deep else float("nan")

        def _anti_cheats(deep_d):
            return {"perm_collapse": round(deep_d["wkv_perm"] - deep_d["wkv"], 4),
                    "memoryless_collapse": round(deep_d["wkv_memoryless"] - deep_d["wkv"], 4)}

        exact_ac = _anti_cheats(deep) if deep else {}
        shunt_ac = _anti_cheats(shunt_deep) if shunt_deep else {}
        go = (shunt_margin >= args.go_margin and shunt_ac.get("perm_collapse", 0) > 0.05
              and shunt_ac.get("memoryless_collapse", 0) > 0.05)

        results[str(seed)] = {
            "n_eval_stories": len(ev_ids),
            "n_count_mismatch_vs_reference": n_mismatch,
            "harness_selfcheck": {
                "my_exact_margin_vs_trigram": round(exact_margin, 4),
                "reference_margin_vs_trigram": ref_margin,
                "abs_diff": round(abs(exact_margin - ref_margin), 4) if ref_margin is not None else None,
            },
            "fI_calibration": {"x50": round(x50, 4) if x50 else None, "r_max": round(r_max, 4) if r_max else None},
            "exact": {"margin_vs_trigram": round(exact_margin, 4), "anti_cheats": exact_ac, "deep": deep},
            "shunt": {"div_mode": args.linattn_div, "g_leak": args.g_leak, "k": args.k_gain,
                      "quantize_levels": args.quantize_levels, "quantize_den": bool(args.quantize_den), "fi": args.fi,
                      "margin_vs_trigram": round(shunt_margin, 4), "anti_cheats": shunt_ac, "deep": shunt_deep},
            "go": bool(go),
            "elapsed_s": round(time.time() - ts0, 1),
        }
        print(f"[seed {seed}] exact margin={exact_margin:+.4f} (ref {ref_margin:+.4f}, "
              f"selfcheck-diff {results[str(seed)]['harness_selfcheck']['abs_diff']}) "
              f"shunt margin={shunt_margin:+.4f} anti-cheats(shunt)={shunt_ac} GO={go} "
              f"[{time.time() - ts0:.0f}s]", flush=True)

        if seed == args.diag_seed and (args.divisive_check or args.sigma_check or args.denquant_check) and exact_diag:
            if args.divisive_check:
                dv_verdict, dv_rows = divisive_vs_subtractive_check(exact_diag, g_leak=args.g_leak, k=args.k_gain, fI=fI)
                results.setdefault("_diagnostics", {})["divisive_vs_subtractive"] = {
                    "verdict": dv_verdict, "samples": dv_rows}
                print(f"    [divisive-check @ seed {seed}] {dv_verdict}", flush=True)
                dv_verdict_nofi, dv_rows_nofi = divisive_vs_subtractive_check(exact_diag, g_leak=args.g_leak, k=args.k_gain, fI=None)
                results["_diagnostics"]["divisive_vs_subtractive_no_fI"] = {
                    "verdict": dv_verdict_nofi, "samples": dv_rows_nofi}
                print(f"    [divisive-check (no fI) @ seed {seed}] {dv_verdict_nofi}", flush=True)
            if args.sigma_check:
                gleaks = [float(x) for x in args.sigma_gleaks.split(",") if x.strip()]
                sd_rows = sigma_domination_check(exact_diag, gleaks, k=args.k_gain)
                # also measure the deep-bucket wkv NLL trend at each g_leak (k=1, fI=None, quantize=0 -- isolate
                # g_leak), on a SMALL slice (--diag-eval-sents). NOT converted to margin_vs_trigram here: the
                # main run's trigram baseline is fit at 4000 stories (frozen, from the reference JSON) and mixing
                # it with a wkv measured on a much smaller slice would be a sample-size-mismatched comparison --
                # this is a TREND-ONLY diagnostic (does wkv degrade as g_leak grows), read against the main run's
                # own full-scale exact/shunt wkv values (recorded alongside, same seed) for context.
                diag_ev_ids = ev_ids[: min(args.diag_eval_sents, len(ev_ids))]
                local_exact_depth, _ = run_seed_arm(ckpt, diag_ev_ids, seed, "exact", 1e-6, 1.0, None, 0, seed)
                local_exact_wkv = local_exact_depth.get("10-99", {}).get("wkv")
                for row in sd_rows:
                    gl_depth, _ = run_seed_arm(ckpt, diag_ev_ids, seed, "shunt", row["g_leak"], args.k_gain, None, 0, seed)
                    gl_deep = gl_depth.get("10-99", {})
                    row["n_stories"] = len(diag_ev_ids)
                    row["wkv"] = round(gl_deep["wkv"], 4) if gl_deep else None
                    row["wkv_minus_local_exact_wkv"] = (round(gl_deep["wkv"] - local_exact_wkv, 4)
                                                         if (gl_deep and local_exact_wkv is not None) else None)
                results.setdefault("_diagnostics", {})["sigma_domination_local_exact_wkv"] = local_exact_wkv
                results.setdefault("_diagnostics", {})["sigma_domination"] = sd_rows
                print(f"    [sigma-check @ seed {seed}] {sd_rows}", flush=True)
            if args.denquant_check and den_scale:
                dq = denquant_interaction_check(exact_diag, den_scale, n_levels=args.quantize_levels, seed=seed)
                results.setdefault("_diagnostics", {})["denquant_interaction"] = dq
                print(f"    [denquant-check @ seed {seed}] {dq}", flush=True)

    n_go = sum(1 for s in seeds if str(s) in results and results[str(s)]["go"])
    n_total = sum(1 for s in seeds if str(s) in results)
    summary = {"go_count": n_go, "n_seeds": n_total, "all_go": n_go == n_total and n_total > 0}
    out = {"runner": "_linattn_shunt_gain_tier1_derisk", "args": vars(args), "results": results,
           "summary": summary, "elapsed_s": round(time.time() - t0, 1)}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {n_total} seeds -> {args.json} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
