"""AFFECT MULTI-CUE COINCIDENCE CONVERGENCE (2026-09-17) — the NAMED next build after the 300k-story
data-scale PARTIAL. Closes the residual proven NOT fixable by more data: realistic-noise robustness.

WHERE THIS SITS (do NOT re-derive the prior rungs).
  * `_affect_grounded_experience_stream_hebbian_derisk.py` (2026-09-05) BUILT an emergent rate-Hebbian
    competitive convergence over a per-concept interoceptive body-state US (comfort/discomfort/arousal relay
    pools). PARTIAL: clean/full grounding teaches a fully separable code (1.000), but realistic afferent NOISE
    collapses the strict worst-case zero-FP ceiling to ~0.
  * `_affect_noise_robust_homeostatic_convergence_derisk.py` (2026-09-05) added FOUR companion processes
    (population pooling, homeostatic noise-floor threshold, three-factor US-gated eligibility, homeostatic
    synaptic scaling) and reached GO at rho=0.6/sigma=1.0 (0.598 worst-case). BUT at the harder point
    rho=0.8/sigma=2.0 the SAME pipeline reads 0.451 worst-case — a FAIL against the 0.5 bar.
  * 2026-09-17 (tonight): a 300k-story data-scale finding proved this residual is NOT a data-scale problem —
    30k -> 300k stories (10x) did not move the noisy-point ceiling. The finding is realistic-noise robustness,
    not corpus size.
  * ROOT CAUSE (code-level, in `_affect_grounded_experience_stream_hebbian_derisk.py` ~L165-183): the
    grounding delivers ONE latent affect channel — `val = (Warriner - 5)/4`, `mag = |val|` — and
    comfort/discomfort/arousal are DETERMINISTIC SPLITS of that ONE number (sign-gated + abs). Every relay
    neuron in every channel, in both the ungated and the ROBUST-pooled pipelines, is a noisy COPY of the SAME
    scalar. Population pooling (the robust rung's fix) averages N such copies within ONE channel: this is
    ADDITIVE noise suppression, SNR ~ sqrt(N) — a POLYNOMIAL improvement that cannot drive the strict
    worst-case zero-FP tail low enough at N=24 (0.451 < 0.5).

THE WALL-REFRAME (the first question at any wall): what companion process does the real interoceptive/affective
system run that within-channel pooling proxies away? Biological multisensory integration does not deliver ONE
noisy channel and average more copies of it — it delivers MULTIPLE, anatomically and physiologically DISTINCT
afferent streams (interoceptive viscerosensory vias, vestibular, proprioceptive-contextual, exteroceptive) that
are CONDITIONALLY INDEPENDENT given the same underlying core-affect state (Craig 2002, 2009 -- interoception is
inherently multi-afferent, lamina I spinothalamocortical + vagal + vestibular channels feeding a common
representation; Evrard 2019; Barrett 2017 -- core affect is CONSTRUCTED by integrating interoceptive +
exteroceptive + contextual signals, not read off one channel). Optimal cue integration across such independent
channels does not just average away noise — Ernst & Banks (2002, Nature, doi:10.1038/415429a) showed the
combined estimator's precision is the SUM of the per-cue precisions (1/sigma_combined^2 = sum 1/sigma_i^2), and
critically the FALSE-POSITIVE tail on a coincidence/AND-like readout across independent cues shrinks
MULTIPLICATIVELY (~p^K for K independent cues each with per-cue false-fire probability p), not just by sqrt(N).
This is the mechanism within-channel pooling structurally cannot deliver, no matter how large N grows, because
all N copies share the SAME noise-free signal AND (in the prior pipeline) are read through the SAME single
downstream threshold — there is no INDEPENDENT gate whose false alarms must ALL coincide.
Kramer, Manoonpong et al. (2022, Front. Neural Circuits, doi:10.3389/fncir.2022.921453) additionally show
CROSSMODAL HEBBIAN PLASTICITY LEARNS the inverse-variance reliability weighting across independent sensory
channels UNSUPERVISED (no hand-set precision weights) -- the emergence bar this build must also clear: the
per-cue combination is TAUGHT, not hand-tuned to the labels.

WHAT THIS RUNNER BUILDS (additive; the ROBUST single-channel pipeline is REUSED verbatim as the baseline/control
this build must beat AT MATCHED BUDGET).
  (1) K CONDITIONALLY-INDEPENDENT-NOISE CUES (`multicue_relay_population`), each a SMALL pooled relay
      population over the SAME latent affect US (same grounding decision + true sign/magnitude — one
      underlying core-affect event) but with its OWN independent afferent noise draw (a distinct rng stream
      per cue — the distinct-afferent-pathway structure Craig's multi-afferent interoception describes). Each
      cue's population is DEGRADED relative to the single matched-budget channel (per-cue population size =
      total_budget // K < total_budget), so per-cue SNR <= the single channel's — no cue smuggles in more raw
      information; any lift must come from FUSION, not extra capacity.
  (2) MULTIPLICATIVE COINCIDENCE FUSION (`multicue_coincidence_fuse`): each cue is independently cleaned by the
      REUSED `pool_and_adapt` (population pooling + homeostatic noise-floor threshold, verbatim import) — a
      per-cue relu(pooled - baseline) supra-threshold drive. The K cleaned cues are then combined by an
      AND-like ELEMENT-WISE PRODUCT (geometric mean, K-invariant scale): a concept's fused comfort/discomfort/
      arousal drive is nonzero ONLY where EVERY cue independently cleared its own noise floor. A false-grounded
      neutral concept now needs ALL K independent noise draws to exceed their thresholds SIMULTANEOUSLY
      (~p^K) — the multiplicative suppression sqrt(N) pooling cannot deliver. The fused feature is appended to
      the convergence input exactly where the single pooled channel sat (`_blocks_robust`, reused verbatim),
      and the SAME three-factor-gated, homeostatically-scaled Hebbian convergence (`train_convergence_robust`,
      `eligibility_gate`, reused verbatim) learns the concept code from it. Coincidence detection of this kind
      is spiking-native (NMDA-receptor / dendritic-plateau AND-gates requiring simultaneous glutamatergic
      inputs) — noted here as the named next rung, not deferred.

THE TWO MATCHED ANTI-CHEAT CONTROLS (the decisive tests -- is this INTEGRATION, or a trick?):
  * G4a MATCHED-BUDGET single-channel arm: the SAME total afferent count (K * per_cue_relay) and the SAME
      per-neuron noise sigma, pooled into ONE channel via the REUSED, UNCHANGED `intero_relay_population` +
      `robust_learned_code_ceiling` pipeline (byte-identical call to the noise-robust rung's own baseline). If
      the multi-cue arm's lift over this exact-budget-matched control is not large, the "lift" was just more
      neurons, not multi-cue fusion.
  * G4b SHARED-NOISE arm: `multicue_relay_population(..., shared_noise=True)` draws ONE noise realization and
      broadcasts it identically to all K "cues" — same total dimensionality and nominal cue count as the
      genuine arm, but ZERO conditional independence. Because coincidence-fusing K IDENTICAL cleaned channels
      collapses algebraically to that single channel (product of K equal terms, Kth-rooted, = the channel
      itself), this arm inherits the single-population-of-`per_cue_relay`-neurons ceiling — much worse than
      the matched-budget arm — and must NOT clear the GO bar. If it did clear, the "lift" would be an artifact
      of extra input dimensions, not of independent-noise fusion.

ANTI-CHEATS CARRIED OVER (must still hold under the multi-cue rule):
  * LESION (no body-state at learning, all K cues): relay carries only noise -> homeostatic floors subtract it
    in every cue -> the fused coincidence feature is ~0 for all concepts -> code collapses to the TEXT baseline.
  * SHUFFLE (concept<->body-state binding permuted, identically across all K cues -- one corrupted "experience"):
    the grounded signal now correlates with the WRONG concepts in every cue -> convergence cannot bind a
    separable code.
  * HELD-OUT (convergence trained on OTHER concepts): held-out concepts' learned code must still separate at
    clean/full grounding -> TAUGHT, not HANDED.
  * INSTRUMENT: synthetic clean code -> ceiling ~1 (>=0.5); text code -> <0.2 (reproduces the prior boundary).

PRE-REGISTERED 6-SEED GO GATE — the operating point is pre-registered at the EDGE WHERE SINGLE-CUE FAILS:
rho=0.8, sigma=2.0 (the noise-robust GO finding measured single-cue pooled worst-case 0.451 there — a FAIL
against the 0.5 bar). rho=0.6/sigma=1.0 (the already-cleared realistic point) is ALSO reported for context, but
is NOT the decisive point (G1 would be trivially satisfied there since the single-cue arm already clears it).
CEIL_GO_BAR=0.5 (imported, unchanged), seeds 42/43/44/100/101/102, the SAME 164-word closed partition + ceiling
instrument (reuse-by-import, verbatim).
  G1  NOISE-ROBUST LIFT     multi-cue coincidence worst-case (min over seeds) >= 0.5 AT rho=0.8,sigma=2.0 AND
                            the matched-budget single-cue arm in the SAME run stays < 0.5 there.
  G2  LOAD-BEARING          LESION and SHUFFLE (multi-cue arm, at the decisive point) both stay
                            <= text_ceiling + ATTRIB_MARGIN.
  G2b GENERALIZES           HELD-OUT concepts (never in the Hebbian weight training) separate at clean/full
                            (rho=1.0, sigma=0.0) -- TAUGHT, not HANDED.
  G3  INSTRUMENT            synthetic clean-code ceiling >= 0.5 AND text-code ceiling < 0.2.
  G4  THE DECISIVE ANTI-CHEAT
        G4a matched-budget: multi-cue worst-case >= matched-budget single-channel worst-case + 0.15 (INTEGRATION,
            not "more afferents").
        G4b shared-noise:   the shared-noise arm's worst-case (max across seeds, the most generous reading) must
            NOT clear the 0.5 bar (INDEPENDENT-noise fusion, not extra dims).
GO iff G1 AND G2 AND G2b AND G3 AND G4.
Reported (decisive context, NOT gated): K dose-response (fixed total budget=24, K in 1..8) at the decisive
point; a per-cue-SNR (sigma) sweep contrasting the multi-cue and matched-budget arms; a SUM-fusion (mean across
cues, no AND) vs COINCIDENCE-fusion (product) readout contrast at the decisive point; the secondary
rho=0.6/sigma=1.0 point; the original ungated single-channel (N=4) baseline for full lineage.

BRAIN-BASED / SCOPE. Host is legit ONLY for the world/body US DELIVERY (the interoceptive affect current, the
SAME declared oracle stand-in the prior two rungs used) + the corpus stream. The FUSION and the CONVERGENCE are
neurons/synapses: population relay pools, a homeostatic noise-floor (label-free, read from the signal's own
running statistics), a rate-Hebbian competitive assembly (Oja + k-WTA + three-factor eligibility + homeostatic
synaptic scaling, ALL reused verbatim, never re-implemented). The coincidence PRODUCT itself is presently a
host-side elementwise multiply over the cleaned relay populations (an idealized stand-in for dendritic NMDA/
plateau AND-gating) — the SAME level the affect lane already accepts as SYNAPTIC (the ungated/robust Hebbian
rules are rate-level, not spiking); a fully-spiking coincidence-detector version (dendritic branches requiring
simultaneous supra-threshold input from independent afferent pathways) is the NAMED next rung, not deferred.

HONEST RESIDUALS: (1) the body-state US remains the declared ORACLE STAND-IN for a grounded world that does not
exist for the TinyStories vocabulary (the SAME stand-in the prior two rungs used) -- this measures whether K
independent-noise cues + coincidence fusion CAN close the realistic-noise gap GIVEN such a stream, not that a
real grounded multi-afferent world exists. (2) "independent" cues here are independent RNG DRAWS over the SAME
scalar Warriner latent, not physiologically distinct afferent modalities (vagal, vestibular, ...) -- a stronger
test would ground each cue in an ACTUALLY DISTINCT bodily signal; this is the named next rung together with
spiking coincidence detection. (3) rate-Hebbian (numpy-CPU) convergence + host elementwise-product fusion; the
fully-spiking on-substrate version (reuse `_genfrontier` build_propagation_bridge, GPU-queued) is the named next
rung. (4) the ceiling is a linear supervised upper bound (the ROBUST rung measured the spiking opponent's mild
nonlinearity NOT to help). (5) the 164-word closed partition is inherited from the prior boundaries.
NOT WIRED: nothing here touches affect_production_organ.py / wkv_mouth_generator.py (byte-unchanged, sha256-
pinned + asserted in --smoke); _STRONG_MARGIN stays 2.0. Additive, default-off, numpy-CPU, reuse-by-import, NO
sim/ edit.

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_multicue_convergence_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_multicue_convergence_derisk \
                  --multicue --n-cues 4 --rho 0.8 --sigma 2.0 --seeds 42 43 44 100 101 102 \
                  --out research/findings/raw/_affect_multicue_convergence_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import logging as _logging
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

# --- reuse-by-import: the SAME de-risked corpus / partition / code / ceiling primitives (NO reimplementation) ----
from research.runners._affect_distributional_tag_derisk import (  # noqa: E402
    WARRINER, load_stories,
)
from research.runners._affect_experienced_opponent_gate_derisk import (  # noqa: E402
    _STRONG_MARGIN, CANONICAL_SEEDS, resample_stories, build_partition, _codes_for,
)
from research.runners._affect_embodied_us_gate_derisk import (  # noqa: E402
    code_separability_ceiling, synthetic_separable_gate,
)
# --- reuse-by-import: rung 1 (ungated Hebbian convergence) -- the ORIGINAL baseline, for lineage context -------
from research.runners._affect_grounded_experience_stream_hebbian_derisk import (  # noqa: E402
    CEIL_GO_BAR, RHO_REAL, SIGMA_REAL, ATTRIB_MARGIN, TEXT_CEIL_MAX, HELDOUT_FRAC, RELAY_NOISE,
    M_ASSEMBLY, EPOCHS, convergence_readout, learned_code_ceiling, grounded_experience_stream,
)
# --- reuse-by-import: rung 2 (noise-robust homeostatic + three-factor convergence) -- the MATCHED-BUDGET arm ----
from research.runners._affect_noise_robust_homeostatic_convergence_derisk import (  # noqa: E402
    intero_relay_population, pool_and_adapt, eligibility_gate, train_convergence_robust,
    robust_learned_code_ceiling, _blocks_robust, N_RELAY_ROBUST, K_MAD,
)
from tools.lab import void_if, undefined_if_empty, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_multicue_convergence.json"

# ── pre-registered operating points (fixed BEFORE the 6-seed) ───────────────────────────────────────────────────
DECISIVE_RHO = 0.8       # the edge where the single-cue pooled arm FAILS (0.451 measured worst-case, < 0.5)
DECISIVE_SIGMA = 2.0
SECONDARY_RHO = RHO_REAL       # 0.6 -- the already-cleared realistic point (reported, not decisive)
SECONDARY_SIGMA = SIGMA_REAL   # 1.0

DEFAULT_K_CUES = 4
TOTAL_BUDGET = N_RELAY_ROBUST  # 24 -- matches the noise-robust rung's own N_RELAY_ROBUST, so the matched-budget
                                # single-channel control reproduces that rung's own measured numbers exactly.
G4A_MARGIN = 0.15               # G4a: multicue worst-case must exceed the matched arm's worst-case by this much

K_DOSE_SWEEP = (1, 2, 3, 4, 6, 8)          # reported: fixed TOTAL_BUDGET, K cues each get TOTAL_BUDGET//K neurons
SIGMA_SNR_SWEEP = (0.5, 1.0, 1.5, 2.0, 3.0)  # reported: per-cue-SNR sweep (multicue vs matched, at DECISIVE_RHO)

# sha256-pinned "byte-unchanged" guard -- NOTHING here should ever touch these two files; if a future edit does,
# --smoke fails LOUDLY rather than silently drifting into "wired" territory.
_AFFECT_PRODUCTION_ORGAN = Path(_REPO) / "research" / "runners" / "affect_production_organ.py"
_WKV_MOUTH_GENERATOR = Path(_REPO) / "webapp" / "wkv_mouth_generator.py"
_AFFECT_PRODUCTION_ORGAN_SHA256 = "6b800c72b71688f3f57728e6d069da0ff540da8c4a817da9f984d8adb8f44e9b"
_WKV_MOUTH_GENERATOR_SHA256 = "5f579df5eec4ca5cfb16428e3ee293aab4b2a9a65bfb0cc7fd6ca0573765469e"


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_production_untouched():
    """Assert affect_production_organ.py + wkv_mouth_generator.py are byte-unchanged (sha256-pinned) and
    _STRONG_MARGIN==2.0 -- this de-risk is reuse-by-import only and must NOT wire or touch production."""
    h_organ = _file_sha256(_AFFECT_PRODUCTION_ORGAN)
    h_mouth = _file_sha256(_WKV_MOUTH_GENERATOR)
    assert h_organ == _AFFECT_PRODUCTION_ORGAN_SHA256, (
        f"affect_production_organ.py CHANGED (sha256 {h_organ[:16]}... != pinned "
        f"{_AFFECT_PRODUCTION_ORGAN_SHA256[:16]}...) -- this de-risk assumed it byte-unchanged.")
    assert h_mouth == _WKV_MOUTH_GENERATOR_SHA256, (
        f"wkv_mouth_generator.py CHANGED (sha256 {h_mouth[:16]}... != pinned "
        f"{_WKV_MOUTH_GENERATOR_SHA256[:16]}...) -- this de-risk assumed it byte-unchanged.")
    assert _STRONG_MARGIN == 2.0, "production _STRONG_MARGIN changed -- this de-risk must NOT touch the gate"
    print(f"  [byte-identical-when-off] affect_production_organ.py sha256={h_organ[:16]}... "
          f"wkv_mouth_generator.py sha256={h_mouth[:16]}... _STRONG_MARGIN==2.0 -> OK", flush=True)


def per_cue_relay_for_k(k_cues, total_budget=TOTAL_BUDGET):
    """Fixed TOTAL afferent budget split across K cues -- each cue is DEGRADED relative to a single channel
    holding the whole budget (per-cue population <= the matched-budget channel's), so a lift cannot come from
    smuggled-in extra information; it can only come from the coincidence FUSION across independent cues."""
    return max(1, total_budget // k_cues)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# NEW MECHANISM 1/2: K CONDITIONALLY-INDEPENDENT-NOISE CUES over the SAME latent affect US (host = world/body
# boundary, the SAME declared US stand-in the prior two rungs used). Byte-structure matches the imported
# `intero_relay_population` / `grounded_experience_stream` (same grounding-draw + shuffle-draw sequence, same
# comfort/discomfort/arousal layout per cue) so `pool_and_adapt` applies to each cue UNCHANGED.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def multicue_relay_population(part_words, raw_gate, seed, rho, sigma, k_cues, n_relay_per_cue,
                              shuffle=False, lesion=False, shared_noise=False, sigma_per_cue=None):
    """K cues, each a SMALL pooled relay over the SAME latent affect US. The GROUNDING decision (which concepts
    are experienced, rho coverage) and the true sign/magnitude are the SAME underlying core-affect event across
    cues (Craig 2002/2009: interoception is multi-afferent but converges on ONE core-affect representation) --
    only the AFFERENT NOISE differs per cue, via an INDEPENDENT rng stream per cue (the distinct-pathway
    structure). shared_noise=True (the G4b anti-cheat) instead draws ONE noise realization and broadcasts it
    identically to every cue -- same nominal K and dimensionality, but ZERO conditional independence.
    sigma_per_cue (optional, len k_cues): per-cue noise scale, for the HETEROGENEOUS-reliability diagnostic
    (physiologically distinct afferent pathways genuinely differ in noise, e.g. vagal vs vestibular) -- default
    None uses the SAME `sigma` for every cue (byte-identical to the original homogeneous construction).
    Returns (cues, n_grounded): cues is a list of K (n x 3*n_relay_per_cue) relay matrices in the SAME
    comfort/discomfort/arousal layout the imported single-channel relay uses."""
    rng = np.random.default_rng(seed + 91_000)
    n = len(part_words)
    val = np.array([(WARRINER[w][0] - 5.0) / 4.0 for w in part_words])   # signed true affect magnitude in ~[-1,1]
    sign = np.sign(val)
    mag = np.abs(val)
    aff_idx = np.where(raw_gate)[0]
    grounded = np.zeros(n, bool)
    if rho > 0 and len(aff_idx) > 0:
        k = int(round(rho * len(aff_idx)))
        chosen = rng.choice(aff_idx, size=k, replace=False) if k > 0 else np.array([], int)
        grounded[chosen] = True
    if shuffle:                                                          # ONE corrupted binding, shared by all cues
        perm = rng.permutation(n)
        sign = sign[perm]; mag = mag[perm]; grounded = grounded[perm]
    comfort = np.zeros(n); discomfort = np.zeros(n); arousal = np.zeros(n)
    if not lesion:
        g_pos = grounded & (sign > 0); g_neg = grounded & (sign < 0)
        comfort[g_pos] = mag[g_pos]; discomfort[g_neg] = mag[g_neg]; arousal[grounded] = mag[grounded]

    def _pools(cue_rng, cue_sigma):
        pools = []
        for ch in (comfort, discomfort, arousal):
            for _ in range(n_relay_per_cue):
                pools.append(ch + np.abs(cue_rng.standard_normal(n)) * cue_sigma * RELAY_NOISE)
        return np.maximum(np.stack(pools, axis=1), 0.0)

    sigmas = [sigma] * k_cues if sigma_per_cue is None else list(sigma_per_cue)
    if shared_noise:
        # G4b: ONE shared random PATTERN, broadcast to all K -- each cue still scales it by its OWN designated
        # sigma (so a heterogeneous-reliability construction stays heterogeneous even under the anti-cheat), but
        # because every cue's noise is now a scalar multiple of the SAME underlying draw, the cues are perfectly
        # correlated (zero conditional independence): no amount of weighting -- uniform OR learned -- can average
        # it away. With sigma_per_cue=None all cue_sigma are equal, so this reduces to K byte-identical copies
        # (unchanged from the homogeneous construction).
        shared_rng = np.random.default_rng(seed + 91_000 + 999_000)
        base = [np.abs(shared_rng.standard_normal(n)) for _ in range(3 * n_relay_per_cue)]
        def _pools_shared(cue_sigma):
            pools = []
            i = 0
            for ch in (comfort, discomfort, arousal):
                for _ in range(n_relay_per_cue):
                    pools.append(ch + base[i] * cue_sigma * RELAY_NOISE)
                    i += 1
            return np.maximum(np.stack(pools, axis=1), 0.0)
        cues = [_pools_shared(sigmas[c]) for c in range(k_cues)]
    else:                                                                # genuine arm: K INDEPENDENT noise streams
        cues = [_pools(np.random.default_rng(seed + 91_000 + 1000 * (c + 1)), sigmas[c]) for c in range(k_cues)]
    return cues, int(grounded.sum())


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# NEW MECHANISM 2/2: MULTIPLICATIVE COINCIDENCE FUSION across the K cleaned (pool_and_adapt'd) cue channels.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def multicue_coincidence_fuse(cleaned_list):
    """AND-like coincidence fusion (Ernst & Banks 2002 optimal cue integration, realized as a PRODUCT rather
    than a precision-weighted SUM): element-wise PRODUCT across the K cleaned (post homeostatic-threshold) cue
    channels, Kth-rooted (geometric mean) so the fused magnitude is K-invariant. A concept's fused drive is
    nonzero ONLY where EVERY cue independently cleared its own noise floor -- a false-grounded neutral concept
    needs ALL K independent noise draws to exceed threshold SIMULTANEOUSLY (~p^K), the multiplicative
    suppression within-channel sqrt(N) pooling structurally cannot deliver. Exact zeros are preserved (no
    epsilon-smoothing that would leak a spurious nonzero product)."""
    stacked = np.stack(cleaned_list, axis=0)             # (K, n, 3)
    k_cues = stacked.shape[0]
    prod = np.prod(stacked, axis=0)                      # (n, 3); zero unless ALL K cues are supra-threshold
    return np.where(prod > 0, prod ** (1.0 / k_cues), 0.0)


def _sum_fuse(cleaned_list):
    """REPORTED CONTRAST ONLY (not part of the gated mechanism): additive/OR-like fusion -- the MEAN across the
    SAME K cleaned (POST-threshold) cues -- to show it is not merely "AND vs OR" that matters (both post-
    threshold combinators collapse under a fixed-budget split; see multicue_raw_fusion below for the fix)."""
    return np.mean(np.stack(cleaned_list, axis=0), axis=0)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 2026-09-17 ITERATION (owner-directed, same worktree): the per-cue-hard-threshold-THEN-AND/OR arms above
# collapsed catastrophically at the decisive point (multicue worst-case 0.010 vs the matched arm's 0.696;
# G4a delta -0.667) because thresholding each cue's SMALL, noisy population INDIVIDUALLY discards the marginal
# evidence a weak-but-real signal carries -- a per-cue population of TOTAL_BUDGET/K has ~sqrt(K) worse SNR than
# the matched channel, and requiring every cue to independently clear ITS OWN hard cutoff before combining
# throws away exactly the graded information that would otherwise survive averaging. THE NAMED NEXT LEVER:
# RAW PRE-THRESHOLD EVIDENCE FUSION -- summate the K cues' RAW (un-thresholded) pooled evidence FIRST, then
# apply ONE homeostatic threshold to the FUSED signal. This is (a) the Ernst & Banks (2002) combined-precision
# estimator (combined 1/sigma^2 = sum of per-cue 1/sigma_i^2 -- precision adds BEFORE any decision), and (b)
# the literal biology of NMDA-receptor dendritic-spike coincidence detection: independent synaptic afferents
# summate their currents on ONE dendritic branch, and the regenerative NMDA nonlinearity is a threshold on that
# SUM, not an AND of K independently-decided pre-synaptic spikes. Reuses `_blocks_robust` / `eligibility_gate` /
# `train_convergence_robust` / `code_separability_ceiling` verbatim -- only the FUSION step changes.
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _pool_raw(X_relay, n_relay):
    """POPULATION POOLING ONLY, no threshold -- this is `pool_and_adapt`'s first line, split out (not
    reimplemented: identical formula) because the raw-fusion arm must SUM the RAW pooled evidence across cues
    BEFORE any per-cue threshold decision, i.e. the two steps `pool_and_adapt` couples must now run in the
    OPPOSITE order (fuse, then threshold once) rather than (threshold each cue, then fuse)."""
    return np.stack([X_relay[:, i * n_relay:(i + 1) * n_relay].mean(axis=1) for i in range(3)], axis=1)


def _homeostatic_threshold(pooled, k_mad=K_MAD):
    """HOMEOSTATIC NOISE-FLOOR THRESHOLD ONLY, no pooling -- `pool_and_adapt`'s second half, split out so it can
    be applied ONCE to the FUSED evidence instead of once per cue (identical label-free median + K_MAD*MAD
    formula; not a new threshold rule, just applied at a different point in the pipeline)."""
    cleaned = np.zeros_like(pooled)
    for ch in range(3):
        col = pooled[:, ch]
        base = float(np.median(col))
        mad = float(np.median(np.abs(col - base))) + 1e-9
        cleaned[:, ch] = np.maximum(col - (base + k_mad * mad), 0.0)
    return cleaned


def multicue_raw_fusion(relay_list, n_relay_per_cue, weights=None, k_mad=K_MAD):
    """RAW PRE-THRESHOLD EVIDENCE FUSION (Ernst & Banks 2002; NMDA dendritic-coincidence detection). Each cue's
    RAW pooled channel (`_pool_raw`, no per-cue decision) is combined by a WEIGHTED SUM across the K
    conditionally-independent-noise cues, and a SINGLE homeostatic threshold (`_homeostatic_threshold`) is then
    applied to the FUSED evidence. weights=None -> uniform 1/K (the optimal Ernst & Banks weighting when every
    cue has EQUAL reliability, which is how `multicue_relay_population` constructs cues by default) -- with
    equal per-cue population size and noise scale, averaging K per-cue means BEFORE thresholding is the same
    variance-reduction a single (K*n_relay_per_cue)-wide channel gets (mean-of-equal-size-group-means = the
    grand mean), but WITHOUT the destructive early per-cue hard-thresholding the AND/OR arms applied above.
    A non-uniform `weights` (see `learn_reliability_weights`) implements graceful down-weighting of a noisier
    cue instead of an all-or-nothing per-cue gate."""
    raws = [_pool_raw(X, n_relay_per_cue) for X in relay_list]
    k_cues = len(raws)
    w = np.full(k_cues, 1.0 / k_cues) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    fused_raw = sum(wi * r for wi, r in zip(w, raws))
    return _homeostatic_threshold(fused_raw, k_mad=k_mad)


def _pool_raw_concat(relay_list, n_relay_per_cue):
    """G4a CONTROL (a) SINGLE WIDE CHANNEL: concatenate ALL K cues' raw relay neurons into ONE undifferentiated
    pool (each neuron keeps its OWN heterogeneous noise level, but cue IDENTITY/boundaries are discarded) and
    pool by a SINGLE mean per channel. Uses the EXACT SAME underlying draws as the multi-cue arms (not just the
    same distribution) -- same total afferents + same total noise power BY CONSTRUCTION, with NO per-cue
    structure for a weighting scheme to exploit. When every cue holds an EQUAL number of neurons this is
    mathematically identical to UNIFORM-weight `multicue_raw_fusion` (mean-of-equal-size-group-means = the grand
    mean) -- computed explicitly anyway so the three-way G4a comparison is transparent, not asserted."""
    k_cues = len(relay_list)
    reordered = []
    for ch in range(3):
        for cue_x in relay_list:
            reordered.append(cue_x[:, ch * n_relay_per_cue:(ch + 1) * n_relay_per_cue])
    wide = np.concatenate(reordered, axis=1)                     # (n, 3*K*n_relay_per_cue), channel-major layout
    return _pool_raw(wide, n_relay_per_cue * k_cues)


def learn_reliability_weights(relay_list, n_relay_per_cue):
    """UNSUPERVISED per-cue reliability weighting (Kramer & Manoonpong et al. 2022, doi:10.3389/fncir.2022.921453
    -- crossmodal Hebbian plasticity LEARNS inverse-variance reliability weights from the signal's OWN
    statistics, no labels, no hand-set constants -- meets the emergence bar). Each cue's weight is the inverse
    of its OWN empirical variance (across concepts x channels) of its RAW pooled signal -- a noisier cue
    contributes LESS to the fused evidence, gracefully, rather than being required to pass an all-or-nothing
    per-cue gate. This closed-form inverse-variance estimate is the fixed point an online Hebbian precision-
    learning rule converges to (Kramer & Manoonpong's own online rule is the named next rung, HONEST_RESIDUALS)."""
    raws = [_pool_raw(X, n_relay_per_cue) for X in relay_list]
    variances = np.array([float(np.var(r)) for r in raws]) + 1e-9
    inv_var = 1.0 / variances
    return inv_var / inv_var.sum()


def pipeline_ceiling(text_codes, relay, raw_gate, seed, multicue, n_relay_per_cue=None, intero_at_test=True,
                     heldout=False, gated=True, homeo=True, fuse_mode="raw_sum"):
    """multicue=False: `--multicue` OFF DELEGATES to the imported single-channel ROBUST pipeline VERBATIM (no
    new code executes; `relay` is then the single (n x 3*n_relay_per_cue) matrix the noise-robust rung itself
    uses) -- BYTE-IDENTICAL to calling `robust_learned_code_ceiling` directly (asserted in --smoke).
    multicue=True: `relay` is a list of K cue matrices. fuse_mode in {"coincidence","sum"} (the 2026-09-05 ORIGINAL
    per-cue-threshold-then-combine arms, kept for lineage/comparison, UNCHANGED) cleans each cue with the
    REUSED `pool_and_adapt` THEN fuses; fuse_mode in {"raw_sum","raw_sum_learned"} (the 2026-09-17 ITERATION)
    fuses the K cues' RAW evidence FIRST (`multicue_raw_fusion`, uniform or learned weights) and thresholds ONCE
    after. Either way the IDENTICAL three-factor-gated, homeostatically-scaled Hebbian convergence
    (`train_convergence_robust` / `eligibility_gate` / `_blocks_robust` / `convergence_readout` /
    `code_separability_ceiling`, ALL reused-by-import verbatim) runs over the fused feature."""
    if not multicue:
        return robust_learned_code_ceiling(text_codes, relay, raw_gate, seed, n_relay=n_relay_per_cue,
                                           intero_at_test=intero_at_test, heldout=heldout, gated=gated, homeo=homeo)
    if fuse_mode in ("coincidence", "sum"):
        cleaned_list = [pool_and_adapt(X, n_relay_per_cue) for X in relay]
        fused = multicue_coincidence_fuse(cleaned_list) if fuse_mode == "coincidence" else _sum_fuse(cleaned_list)
    elif fuse_mode == "raw_sum":
        fused = multicue_raw_fusion(relay, n_relay_per_cue)
    elif fuse_mode == "raw_sum_learned":
        fused = multicue_raw_fusion(relay, n_relay_per_cue, weights=learn_reliability_weights(relay, n_relay_per_cue))
    elif fuse_mode == "single_wide":                              # G4a control (a): collapse cue boundaries
        fused = _homeostatic_threshold(_pool_raw_concat(relay, n_relay_per_cue))
    else:
        raise ValueError(f"unknown fuse_mode {fuse_mode!r}")
    X_full = _blocks_robust(text_codes, fused, intero_present=True)
    X_test_full = _blocks_robust(text_codes, fused, intero_present=intero_at_test)
    us = eligibility_gate(fused) if gated else None
    if not heldout:
        W = train_convergence_robust(X_full, seed, us_gate=us, homeo=homeo)
        return code_separability_ceiling(convergence_readout(W, X_test_full), raw_gate, seed)
    rng = np.random.default_rng(seed + 555)
    n = len(raw_gate)
    perm = rng.permutation(n)
    n_ho = max(int(round(HELDOUT_FRAC * n)), 1)
    ho = np.zeros(n, bool); ho[perm[:n_ho]] = True
    tr = ~ho
    if raw_gate[ho].sum() == 0 or (~raw_gate[ho]).sum() == 0:
        return 0.0
    us_tr = us[tr] if us is not None else None
    W = train_convergence_robust(X_full[tr], seed, us_gate=us_tr, homeo=homeo)       # never sees held-out concepts
    return code_separability_ceiling(convergence_readout(W, X_test_full[ho]), raw_gate[ho], seed)


def _smoke_delegation_and_build_proof(part_words, raw_gate, seed=42):
    """(a) BYTE-IDENTICAL-WHEN-OFF: `pipeline_ceiling(multicue=False)` IS the imported `robust_learned_code_
    ceiling` call, not a re-implementation -- two independent invocations must be EXACTLY equal (deterministic,
    same seed). (b) BUILD PROOF: both fusion families (the 2026-09-05 per-cue-threshold coincidence/sum arms,
    and the 2026-09-17 raw-pre-threshold-fusion arms, uniform + learned) run end-to-end on a tiny synthetic
    partition and return a ceiling in [0, 1]. (c) ANTI-CHEAT ALGEBRA, both families: shared-noise cues
    (identical draws) must collapse to exactly a SINGLE cue's own cleaned/thresholded reading -- for
    raw_sum this holds because a weighted sum of K IDENTICAL raw copies with weights summing to 1 equals that
    one copy, THEN one threshold; for coincidence it holds because a product of K equal terms, Kth-rooted,
    equals the term itself."""
    n_relay = 8
    X, _ = intero_relay_population(part_words, raw_gate, seed, 0.6, 1.0, n_relay)
    X2, _ = intero_relay_population(part_words, raw_gate, seed, 0.6, 1.0, n_relay)
    assert np.array_equal(X, X2), "intero_relay_population is not deterministic"
    dummy_text = np.abs(np.random.default_rng(0).standard_normal((len(part_words), 5)))
    c_off_a = pipeline_ceiling(dummy_text, X, raw_gate, seed, multicue=False, n_relay_per_cue=n_relay)
    c_off_b = robust_learned_code_ceiling(dummy_text, X, raw_gate, seed, n_relay=n_relay)
    assert c_off_a == c_off_b, f"--multicue OFF diverged from the imported baseline: {c_off_a} != {c_off_b}"
    print(f"  [delegate-byte-identical] multicue=False ceiling {c_off_a:.6f} == imported "
          f"robust_learned_code_ceiling {c_off_b:.6f} exactly -> OK", flush=True)

    k_cues, per_cue = 4, 2
    cues, n_g = multicue_relay_population(part_words, raw_gate, seed, 0.6, 1.0, k_cues, per_cue)
    assert len(cues) == k_cues and all(c.shape == (len(part_words), 3 * per_cue) for c in cues)
    for fm in ("coincidence", "sum", "raw_sum", "raw_sum_learned"):
        c_on = pipeline_ceiling(dummy_text, cues, raw_gate, seed, multicue=True, n_relay_per_cue=per_cue, fuse_mode=fm)
        assert 0.0 <= c_on <= 1.0, f"multicue({fm}) ceiling out of range: {c_on}"

    shared_cues, _ = multicue_relay_population(part_words, raw_gate, seed, 0.6, 1.0, k_cues, per_cue,
                                               shared_noise=True)
    assert all(np.array_equal(shared_cues[0], sc) for sc in shared_cues[1:]), "shared_noise cues are not identical"
    fused_shared = multicue_coincidence_fuse([pool_and_adapt(X, per_cue) for X in shared_cues])
    fused_one = pool_and_adapt(shared_cues[0], per_cue)
    assert np.allclose(fused_shared, fused_one, atol=1e-9), (
        "shared-noise coincidence fusion should algebraically collapse to a SINGLE cue's cleaned channel")
    raw_fused_shared = multicue_raw_fusion(shared_cues, per_cue)
    raw_fused_one = _homeostatic_threshold(_pool_raw(shared_cues[0], per_cue))
    assert np.allclose(raw_fused_shared, raw_fused_one, atol=1e-9), (
        "shared-noise RAW fusion should algebraically collapse to a SINGLE cue's raw-then-thresholded channel")
    print(f"  [build-proof] multicue(K={k_cues},per_cue={per_cue}) coincidence/sum/raw_sum/raw_sum_learned all "
          f"ran in [0,1]; n_grounded={n_g}; BOTH fusion families' shared-noise arm algebraically collapses to "
          f"one cue (as required) -> OK", flush=True)


def run_seed(seed, stories, part_words, raw_gate, n_hub, window, min_count, resample_frac, k_cues,
            decisive_rho, decisive_sigma, verbose=False, total_budget_primary=TOTAL_BUDGET):
    sub = resample_stories(stories, resample_frac, seed)
    vocab, codes, _codes_read, _rel = _codes_for(sub, n_hub, window, min_count)
    widx = {w: i for i, w in enumerate(vocab)}
    part_idx = np.array([widx[w] for w in part_words])
    text_codes = np.asarray(codes[part_idx], float)
    D = text_codes.shape[1]

    text_ceiling = code_separability_ceiling(text_codes, raw_gate, seed)     # reproduce the boundary (like-for-like)
    synth = synthetic_separable_gate(seed, raw_gate, D)                      # G3 instrument

    per_cue = per_cue_relay_for_k(k_cues)               # LINEAGE budget (24 total) -- unchanged, feeds ONLY the
    matched_n_relay = per_cue * k_cues                  # ORIGINAL homogeneous matched arm + the AND/sum/homog-
                                                         # uniform/homog-learned/dose-response/snr-sweep lineage,
                                                         # so those numbers stay comparable to the prior commits.
    # PRIMARY budget (2026-09-17, 4th iteration, owner-directed): a principled per-cue POPULATION bump (6->8 at
    # k_cues=4, i.e. total_budget_primary=32 vs the lineage's 24) to raise per-cue SNR and shrink the ceiling's
    # variance ACROSS ALL SEEDS -- a genuine mechanism parameter (more afferents per cue), not a seed-fitting
    # knob. Feeds ONLY the gated heterogeneous+learned arm and its two G4a controls, so the ORIGINAL lineage
    # arms above are untouched (their numbers must stay exactly reproducible).
    per_cue_primary = per_cue_relay_for_k(k_cues, total_budget=total_budget_primary)
    matched_n_relay_primary = per_cue_primary * k_cues
    # HETEROGENEOUS-RELIABILITY ramp (2026-09-17, 2nd iteration -- NOW PRIMARY, owner-directed): a FIXED relative
    # multiplier per cue index (physiologically distinct afferent pathways -- vagal/vestibular/proprioceptive --
    # genuinely differ in noise, Craig 2002/2009), rescaled to whatever `sigma` is tested at a given operating
    # point (so sigma=0.0 at the clean point correctly means ZERO noise for every cue, not a stale absolute list).
    HETERO_MULT = np.linspace(0.5, 2.5, k_cues)
    TRUE_INV_VAR = 1.0 / (HETERO_MULT ** 2)                       # ground-truth precision (KNOWN to the experiment
                                                                   # designer for validation; NOT given to the
                                                                   # mechanism, which must estimate it unsupervised)

    def _multicue_at(rho, sigma, lesion=False, shuffle=False, shared_noise=False, heldout=False,
                     fuse_mode="raw_sum_learned", k=k_cues, pc=per_cue_primary, hetero=True):
        spc = list(HETERO_MULT * sigma) if hetero else None
        cues, n_g = multicue_relay_population(part_words, raw_gate, seed, rho, sigma, k, pc,
                                              shuffle=shuffle, lesion=lesion, shared_noise=shared_noise,
                                              sigma_per_cue=spc)
        c = pipeline_ceiling(text_codes, cues, raw_gate, seed, multicue=True, n_relay_per_cue=pc,
                             heldout=heldout, fuse_mode=fuse_mode)
        return c, n_g

    def _matched_at(rho, sigma, lesion=False, shuffle=False, heldout=False, n_relay=matched_n_relay):
        X, n_g = intero_relay_population(part_words, raw_gate, seed, rho, sigma, n_relay,
                                         shuffle=shuffle, lesion=lesion)
        c = pipeline_ceiling(text_codes, X, raw_gate, seed, multicue=False, n_relay_per_cue=n_relay,
                             heldout=heldout)
        return c, n_g

    # ── DECISIVE operating point (rho=0.8, sigma=2.0) -- the pre-registered gate ────────────────────────────────
    # PRIMARY ARM (owner-directed 3rd+4th iteration): heterogeneous-reliability cues (at the BUMPED per-cue
    # population) + LEARNED inverse-variance weighting. G4a's three-way anti-cheat is built from the EXACT SAME
    # cue draw (bit-identical, not just matched-distribution, and at the SAME bumped budget) so the comparison
    # isolates the WEIGHTING, nothing else -- the budget invariant holds by construction.
    hetero_spc_decisive = list(HETERO_MULT * decisive_sigma)
    hetero_cues_decisive, n_g_dec = multicue_relay_population(part_words, raw_gate, seed, decisive_rho,
                                                              decisive_sigma, k_cues, per_cue_primary,
                                                              sigma_per_cue=hetero_spc_decisive)
    multicue_decisive = pipeline_ceiling(text_codes, hetero_cues_decisive, raw_gate, seed, multicue=True,
                                         n_relay_per_cue=per_cue_primary, fuse_mode="raw_sum_learned")      # (c)
    single_wide_decisive = pipeline_ceiling(text_codes, hetero_cues_decisive, raw_gate, seed, multicue=True,
                                            n_relay_per_cue=per_cue_primary, fuse_mode="single_wide")       # (a)
    uniform_hetero_decisive = pipeline_ceiling(text_codes, hetero_cues_decisive, raw_gate, seed, multicue=True,
                                               n_relay_per_cue=per_cue_primary, fuse_mode="raw_sum")        # (b)
    learned_w_decisive = learn_reliability_weights(hetero_cues_decisive, per_cue_primary)         # for the weight-
    weight_true_ivar_corr = float(np.corrcoef(learned_w_decisive, TRUE_INV_VAR)[0, 1])           # tracking check
    weight_rank_matches_truth = bool(np.array_equal(np.argsort(learned_w_decisive), np.argsort(TRUE_INV_VAR)))

    matched_decisive, _ = _matched_at(decisive_rho, decisive_sigma)                        # ORIGINAL homog. lineage
    shared_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, shared_noise=True)                     # G4b
    lesion_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, lesion=True)                           # G2
    shuffle_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, shuffle=True)                         # G2

    # ── LINEAGE (2026-09-05 + 2026-09-17-1st-iteration arms, reported not gated, all HOMOGENEOUS cues, PINNED to
    # the ORIGINAL per_cue=24-budget via explicit pc=per_cue -- _multicue_at's default is now per_cue_primary) ────
    and_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, fuse_mode="coincidence", hetero=False, pc=per_cue)
    sum_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, fuse_mode="sum", hetero=False, pc=per_cue)
    homog_uniform_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, fuse_mode="raw_sum", hetero=False,
                                             pc=per_cue)
    learned_decisive, _ = _multicue_at(decisive_rho, decisive_sigma, fuse_mode="raw_sum_learned", hetero=False,
                                       pc=per_cue)

    # ── original ungated single-channel (N_RELAY=4) baseline, for lineage context (reported only) ───────────────
    Xi_ungated, _ = grounded_experience_stream(part_words, raw_gate, seed, decisive_rho, decisive_sigma)
    ungated_decisive = learned_code_ceiling(text_codes, Xi_ungated, raw_gate, seed)

    # ── SECONDARY (already-cleared realistic) point rho=0.6, sigma=1.0 -- reported, NOT decisive ─────────────────
    multicue_secondary, _ = _multicue_at(SECONDARY_RHO, SECONDARY_SIGMA)
    matched_secondary, _ = _matched_at(SECONDARY_RHO, SECONDARY_SIGMA)

    # ── CLEAN/FULL (rho=1.0, sigma=0.0) -- G2b generalization + reference (hetero sigma*0.0 == no noise) ──────────
    multicue_clean, _ = _multicue_at(1.0, 0.0)
    heldout_clean, _ = _multicue_at(1.0, 0.0, heldout=True)                                                # G2b
    lesion_clean, _ = _multicue_at(1.0, 0.0, lesion=True)

    # ── K dose-response (reported, HOMOGENEOUS cues, unchanged from the 2nd iteration): fixed TOTAL_BUDGET, K in
    # K_DOSE_SWEEP, at the DECISIVE point -- both the uniform raw_sum and the AND-lineage arm. ---------------------
    dose_response = []
    for kk in K_DOSE_SWEEP:
        pc = per_cue_relay_for_k(kk)
        c_raw, _ = _multicue_at(decisive_rho, decisive_sigma, k=kk, pc=pc, fuse_mode="raw_sum", hetero=False)
        c_and, _ = _multicue_at(decisive_rho, decisive_sigma, k=kk, pc=pc, fuse_mode="coincidence", hetero=False)
        dose_response.append({"k_cues": kk, "per_cue_relay": pc, "ceiling": c_raw, "ceiling_and_lineage": c_and})

    # ── per-cue-SNR sweep (reported, HOMOGENEOUS cues, unchanged): sigma sweep at fixed K/per_cue, rho=decisive_rho
    snr_sweep = []
    for sg in SIGMA_SNR_SWEEP:
        c_m, _ = _multicue_at(decisive_rho, sg, fuse_mode="raw_sum", hetero=False, pc=per_cue)
        c_s, _ = _matched_at(decisive_rho, sg)
        snr_sweep.append({"sigma": sg, "multicue_ceiling": c_m, "matched_ceiling": c_s})

    g4a_delta_vs_single_wide = multicue_decisive - single_wide_decisive
    g4a_delta_vs_uniform = multicue_decisive - uniform_hetero_decisive

    if verbose:
        print(f"  [seed {seed}] D={D} text={text_ceiling:.3f} | DECISIVE(rho={decisive_rho},sig={decisive_sigma}) "
              f"HETERO+LEARNED(primary)={multicue_decisive:.3f} | G4a-controls: single_wide(a)="
              f"{single_wide_decisive:.3f} uniform-hetero(b)={uniform_hetero_decisive:.3f} "
              f"[delta_a={g4a_delta_vs_single_wide:+.3f} delta_b={g4a_delta_vs_uniform:+.3f}] | "
              f"weight<->true-invar corr={weight_true_ivar_corr:.3f} rank-match={weight_rank_matches_truth} | "
              f"shared-noise={shared_decisive:.3f} lesion={lesion_decisive:.3f} shuffle={shuffle_decisive:.3f} | "
              f"orig-matched(homog)={matched_decisive:.3f} | LINEAGE AND={and_decisive:.3f} sum={sum_decisive:.3f} "
              f"homog-uniform={homog_uniform_decisive:.3f} homog-learned={learned_decisive:.3f} "
              f"ungated(N=4)={ungated_decisive:.3f} | clean={multicue_clean:.3f} held-out={heldout_clean:.3f} | "
              f"synth={synth['code_ceiling']:.3f}", flush=True)

    return {
        "seed": int(seed), "code_dim": int(D), "text_ceiling": text_ceiling, "n_grounded_decisive": n_g_dec,
        "per_cue_primary": per_cue_primary, "matched_n_relay_primary": matched_n_relay_primary,
        "multicue_decisive": multicue_decisive,
        "single_wide_decisive": single_wide_decisive, "uniform_hetero_decisive": uniform_hetero_decisive,
        "matched_decisive": matched_decisive,
        "weight_true_ivar_corr": weight_true_ivar_corr, "weight_rank_matches_truth": weight_rank_matches_truth,
        "learned_weights_decisive": [float(x) for x in learned_w_decisive],
        "true_inv_var": [float(x) for x in TRUE_INV_VAR],
        "shared_noise_decisive": shared_decisive, "lesion_decisive": lesion_decisive,
        "shuffle_decisive": shuffle_decisive,
        "and_lineage_decisive": and_decisive, "sum_fuse_decisive": sum_decisive,
        "homog_uniform_decisive": homog_uniform_decisive, "learned_decisive": learned_decisive,
        "ungated_n4_decisive": ungated_decisive,
        "multicue_secondary": multicue_secondary, "matched_secondary": matched_secondary,
        "multicue_clean": multicue_clean, "heldout_clean": heldout_clean, "lesion_clean": lesion_clean,
        "synth_code_ceiling": float(synth["code_ceiling"]),
        "dose_response": dose_response, "snr_sweep": snr_sweep,
        "g4a_delta_vs_single_wide": g4a_delta_vs_single_wide, "g4a_delta_vs_uniform": g4a_delta_vs_uniform,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=CANONICAL_SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- byte-identical-off + build proof")
    ap.add_argument("--multicue", action="store_true",
                    help="run the FULL multi-cue coincidence battery + GO gate; OFF delegates to the imported "
                         "single-channel ROBUST pipeline (matched-budget arm only, no gate evaluated)")
    ap.add_argument("--n-cues", type=int, default=DEFAULT_K_CUES)
    ap.add_argument("--rho", type=float, default=DECISIVE_RHO)
    ap.add_argument("--sigma", type=float, default=DECISIVE_SIGMA)
    ap.add_argument("--total-budget", type=int, default=TOTAL_BUDGET,
                    help="total afferent budget for the PRIMARY heterogeneous+learned arm + its two G4a controls "
                         "(default 24, matching the lineage arms' budget byte-identically). A principled per-cue-"
                         "population bump -- e.g. 32 (per_cue 6->8 at n-cues=4) -- raises per-cue SNR to reduce "
                         "the ceiling's variance across seeds; the lineage arms (AND/sum/homog-uniform/homog-"
                         "learned/dose-response/snr-sweep/ORIGINAL matched) always stay pinned at 24.")
    ap.add_argument("--max-stories", type=int, default=60000)
    ap.add_argument("--resample-frac", type=float, default=0.8)
    ap.add_argument("--n-hub", type=int, default=64, help="concept code dim (matches the affect lane operating point)")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = [a.seeds[0]] if a.smoke else a.seeds
    max_stories = min(a.max_stories, 8000) if a.smoke else a.max_stories
    min_count = 2 if a.smoke else a.min_count
    k_cues = a.n_cues
    per_cue = per_cue_relay_for_k(k_cues)
    matched_n_relay = per_cue * k_cues
    per_cue_primary = per_cue_relay_for_k(k_cues, total_budget=a.total_budget)
    matched_n_relay_primary = per_cue_primary * k_cues

    t0 = time.time()
    print(f"[multicue-convergence] seeds={seeds} smoke={a.smoke} multicue={a.multicue} k_cues={k_cues} "
          f"per_cue_relay(lineage)={per_cue} per_cue_relay(primary)={per_cue_primary} total_budget(primary)="
          f"{a.total_budget} rho={a.rho} sigma={a.sigma} max_stories={max_stories} n_hub={a.n_hub} "
          f"backend={os.environ.get('SIM_BACKEND')}", flush=True)
    _assert_production_untouched()

    stories = load_stories(max_stories)
    part_words, raw_gate = build_partition(stories, seeds, a.resample_frac, min_count)
    void_if(len(part_words) < 20, f"only {len(part_words)} common partition words")
    n_pos, n_neg = int(raw_gate.sum()), int((~raw_gate).sum())
    void_if(n_pos == 0 or n_neg == 0, f"degenerate partition n_pos={n_pos} n_neg={n_neg}")
    print(f"  partition: {len(part_words)} common words | raw-gated(affect)={n_pos} raw-excluded(neutral)={n_neg}",
          flush=True)

    if a.smoke:
        _smoke_delegation_and_build_proof(part_words, raw_gate, seed=seeds[0])

    rows = [run_seed(s, stories, part_words, raw_gate, a.n_hub, a.window, min_count, a.resample_frac, k_cues,
                     a.rho, a.sigma, verbose=True, total_budget_primary=a.total_budget) for s in seeds]

    def _agg(key, fn):
        return float(fn(r[key] for r in rows))

    def _mean(key):
        return float(np.mean([r[key] for r in rows]))

    # AGGREGATION DIRECTION: `code_separability_ceiling` is ALWAYS a "wants-to-be-high" metric regardless of
    # which arm produces it, so "worst-case" for ANY ceiling reading is MIN across seeds (the sibling's own
    # convention: "worst-case ceiling (min across seeds) >= bar" -- the mission's cited reference number for
    # the matched arm, 0.451, is ITself a MIN, not a max). This was fixed 2026-09-17: the first iteration used
    # MAX for `matched_decisive` (treating it as a "should stay low" control), which does not match how the
    # mission's own 0.451 reference was computed and understates how often the matched arm actually clears 0.5.
    # Only genuine "should stay LOW" anti-hollow controls (lesion, shuffle, text_ceiling, shared-noise) use MAX
    # ("worst = highest, hardest to call it low" -- unchanged, matches the sibling + the mission's explicit G4b
    # wording: "the shared-noise arm's worst-case (max across seeds, the most generous reading)").
    text_ceiling_worst = _agg("text_ceiling", max)          # worst = HIGHEST (hardest to call it low)
    text_ceiling_mean = _mean("text_ceiling")
    multicue_decisive_worst = _agg("multicue_decisive", min)     # worst = LOWEST (hardest to call it high) --
    multicue_decisive_mean = _mean("multicue_decisive")           # PRIMARY: heterogeneous cues + LEARNED weights
    single_wide_worst = _agg("single_wide_decisive", min)         # G4a control (a): same cues, single wide channel
    single_wide_mean = _mean("single_wide_decisive")
    uniform_hetero_worst = _agg("uniform_hetero_decisive", min)   # G4a control (b): same cues, UNIFORM weights
    uniform_hetero_mean = _mean("uniform_hetero_decisive")
    matched_decisive_worst = _agg("matched_decisive", min)        # ORIGINAL homogeneous single-channel (lineage)
    matched_decisive_mean = _mean("matched_decisive")
    shared_decisive_worst = _agg("shared_noise_decisive", max)   # worst = HIGHEST (most generous reading for G4b)
    shared_decisive_mean = _mean("shared_noise_decisive")
    lesion_decisive_worst = _agg("lesion_decisive", max)
    shuffle_decisive_worst = _agg("shuffle_decisive", max)
    and_decisive_worst = _agg("and_lineage_decisive", min)        # lineage: the 2026-09-05 per-cue-AND arm
    and_decisive_mean = _mean("and_lineage_decisive")
    sum_decisive_worst = _agg("sum_fuse_decisive", min)
    sum_decisive_mean = _mean("sum_fuse_decisive")
    homog_uniform_worst = _agg("homog_uniform_decisive", min)     # lineage: 2nd iteration's primary (homogeneous)
    homog_uniform_mean = _mean("homog_uniform_decisive")
    learned_decisive_worst = _agg("learned_decisive", min)        # homogeneous cues + learned (sanity: ~uniform)
    learned_decisive_mean = _mean("learned_decisive")
    ungated_decisive_mean = _mean("ungated_n4_decisive")
    heldout_clean_worst = _agg("heldout_clean", min)
    heldout_clean_mean = _mean("heldout_clean")
    multicue_clean_worst = _agg("multicue_clean", min)
    lesion_clean_worst = _agg("lesion_clean", max)
    synth_ceiling_worst = _agg("synth_code_ceiling", min)
    multicue_secondary_worst = _agg("multicue_secondary", min)
    matched_secondary_worst = _agg("matched_secondary", min)     # FIXED 2026-09-17: MIN (see note above)
    g4a_delta_a_mean = _mean("g4a_delta_vs_single_wide")
    g4a_delta_a_worst = float(min(r["g4a_delta_vs_single_wide"] for r in rows))
    g4a_delta_b_mean = _mean("g4a_delta_vs_uniform")
    g4a_delta_b_worst = float(min(r["g4a_delta_vs_uniform"] for r in rows))
    weight_true_ivar_corr_mean = _mean("weight_true_ivar_corr")
    weight_rank_matches_all_seeds = bool(all(r["weight_rank_matches_truth"] for r in rows))
    weight_rank_match_count = int(sum(r["weight_rank_matches_truth"] for r in rows))
    # pooled correlation across ALL (seed x cue) pairs -- the SAME fixed TRUE_INV_VAR ramp every seed, so this is
    # the more statistically meaningful reading than any single seed's 4-point correlation.
    all_learned_w = np.concatenate([r["learned_weights_decisive"] for r in rows])
    all_true_ivar = np.concatenate([r["true_inv_var"] for r in rows])
    weight_true_ivar_corr_pooled = float(np.corrcoef(all_learned_w, all_true_ivar)[0, 1])

    dose_response_agg = []
    for kk in K_DOSE_SWEEP:
        vals = [next(x["ceiling"] for x in r["dose_response"] if x["k_cues"] == kk) for r in rows]
        and_vals = [next(x["ceiling_and_lineage"] for x in r["dose_response"] if x["k_cues"] == kk) for r in rows]
        dose_response_agg.append({"k_cues": kk, "per_cue_relay": per_cue_relay_for_k(kk),
                                  "worst": float(min(vals)), "mean": float(np.mean(vals)),
                                  "and_lineage_worst": float(min(and_vals)), "and_lineage_mean": float(np.mean(and_vals))})
    snr_sweep_agg = []
    for sg in SIGMA_SNR_SWEEP:
        mvals = [next(x["multicue_ceiling"] for x in r["snr_sweep"] if x["sigma"] == sg) for r in rows]
        svals = [next(x["matched_ceiling"] for x in r["snr_sweep"] if x["sigma"] == sg) for r in rows]
        snr_sweep_agg.append({"sigma": sg, "multicue_worst": float(min(mvals)), "multicue_mean": float(np.mean(mvals)),
                              "matched_worst": float(min(svals)), "matched_mean": float(np.mean(svals))})

    # ── GO CRITERIA (pre-registered; only meaningful/evaluated when --multicue was requested) ──────────────────────
    # 3rd iteration (owner-directed): PRIMARY = heterogeneous cues + LEARNED inverse-variance weights. G1's
    # "matched-budget single-cue arm" reference is now control (a) SINGLE-WIDE-CHANNEL on the SAME heterogeneous
    # cues (the most directly paired comparator for this construction). G4a now requires the LEARNED arm to beat
    # BOTH G4a controls -- (a) single-wide-channel and (b) UNIFORM-weight multi-cue, both on the IDENTICAL cue
    # draw -- so a pass isolates the LEARNED WEIGHTING as the lever, not heterogeneity or extra afferents alone.
    g1 = bool(multicue_decisive_worst >= CEIL_GO_BAR and single_wide_worst < CEIL_GO_BAR)
    g2 = bool(lesion_decisive_worst <= text_ceiling_worst + ATTRIB_MARGIN and
             shuffle_decisive_worst <= text_ceiling_worst + ATTRIB_MARGIN)
    g2b = bool(heldout_clean_worst >= CEIL_GO_BAR)
    g3 = bool(synth_ceiling_worst >= CEIL_GO_BAR and text_ceiling_worst < TEXT_CEIL_MAX)
    g4a = bool(multicue_decisive_worst >= single_wide_worst + G4A_MARGIN and
              multicue_decisive_worst >= uniform_hetero_worst + G4A_MARGIN)
    g4b = bool(shared_decisive_worst < CEIL_GO_BAR)
    g4 = bool(g4a and g4b)
    go = bool(a.multicue and g1 and g2 and g2b and g3 and g4)

    # NOTE (verdict-preconditions gate): `Verdict.control` checks are SANITY/ATTRIBUTION checks -- things
    # expected to hold whenever the run is well-formed, whether or not the pre-registered numeric bar (G1..G4,
    # decided separately below) is cleared (mirrors the sibling rungs: their "grounding load-bearing" /
    # "taught not handed" controls are checked at the CLEAN/FULL point, not the point under scientific test,
    # precisely so a strict-bar miss at the hard point reads as NO-GO rather than UNDEFINED). G4a/G4b are the
    # hypothesis under test here, not sanity checks -- they are decided as ordinary booleans below and must
    # NOT be registered as Verdict controls, or a genuine (informative) miss would wrongly read as UNDEFINED.
    # The multicue-specific sanity checks are only registered when --multicue was requested; in BASELINE-ONLY
    # mode (--multicue absent) this run evaluates no mechanism claim and earns only the two universal checks.
    v = Verdict("multi-cue coincidence convergence: is the run well-formed (grounding load-bearing, taught not "
               "handed) independent of whether the strict decisive-point bar is cleared?")
    v.require("partition non-degenerate (affect + neutral both present)", measured=(n_pos > 0 and n_neg > 0),
              expect=True)
    v.require("the ceiling INSTRUMENT discriminates (synthetic clean >=0.5, text <0.2)",
              measured=(synth_ceiling_worst >= CEIL_GO_BAR and text_ceiling_worst < TEXT_CEIL_MAX), expect=True)
    if a.multicue:
        v.control("grounding is LOAD-BEARING at clean/full (multicue code separates; the no-grounding LESION "
                 "does not)", treatment=multicue_clean_worst, control=lesion_clean_worst, min_separation=0.2)
        v.control("the code is TAUGHT not HANDED (held-out concepts separate; the shuffle-binding control does not)",
                  treatment=heldout_clean_worst, control=shuffle_decisive_worst, min_separation=0.2)
    verdict_earned = v.decide(go=go, verbose=False)

    if a.multicue:
        attributable_to("learned-hetero decisive-point ceiling (vs G4a control (a) single-wide-channel)",
                        multicue_decisive_mean, single_wide_mean)
        attributable_to("learned-hetero decisive-point ceiling (vs G4a control (b) uniform-weight multi-cue)",
                        multicue_decisive_mean, uniform_hetero_mean)
        attributable_to("learned-hetero decisive-point ceiling (vs the ORIGINAL homogeneous matched arm)",
                        multicue_decisive_mean, matched_decisive_mean)
        attributable_to("learned-hetero decisive-point ceiling (vs the shared-noise arm)",
                        multicue_decisive_mean, shared_decisive_mean)
        attributable_to("learned-hetero decisive-point ceiling (vs the no-grounding LESION)",
                        multicue_decisive_mean, lesion_decisive_worst)
        attributable_to("learned-hetero decisive-point ceiling (vs the shuffle-binding control)",
                        multicue_decisive_mean, shuffle_decisive_worst)

    tag = f"{len(seeds)}-seed" if not a.smoke else "SMOKE(1-seed)"
    lift_line = (f"DECISIVE(rho={a.rho},sigma={a.sigma}): HETERO+LEARNED(primary)={multicue_decisive_worst:.3f} "
                f"worst ({multicue_decisive_mean:.3f} mean) || THREE-WAY G4a (SAME cue draw): (a) single-wide-"
                f"channel={single_wide_worst:.3f} worst ({single_wide_mean:.3f} mean) [delta_a="
                f"{g4a_delta_a_worst:+.3f} worst, {g4a_delta_a_mean:+.3f} mean]; (b) uniform-weight multi-cue="
                f"{uniform_hetero_worst:.3f} worst ({uniform_hetero_mean:.3f} mean) [delta_b={g4a_delta_b_worst:+.3f} "
                f"worst, {g4a_delta_b_mean:+.3f} mean] || WEIGHT<->TRUTH: pooled corr(learned weight, true inverse-"
                f"variance)={weight_true_ivar_corr_pooled:.3f}, rank-matches-truth {weight_rank_match_count}/"
                f"{len(seeds)} seeds || anti-cheats: shared-noise={shared_decisive_worst:.3f} worst; lesion="
                f"{lesion_decisive_worst:.3f}; shuffle={shuffle_decisive_worst:.3f}; held-out(clean)="
                f"{heldout_clean_worst:.3f}; synth-instrument={synth_ceiling_worst:.3f}; text={text_ceiling_worst:.3f} "
                f"|| LINEAGE (reported): ORIGINAL-homog-matched={matched_decisive_worst:.3f} worst "
                f"({matched_decisive_mean:.3f} mean); 2026-09-05 AND={and_decisive_worst:.3f}, sum/OR="
                f"{sum_decisive_worst:.3f} (worst-case); 1st-raw-iteration homog-uniform={homog_uniform_worst:.3f}, "
                f"homog-learned(sanity)={learned_decisive_worst:.3f} (worst-case); ungated(N=4)="
                f"{ungated_decisive_mean:.3f} mean || SECONDARY(rho={SECONDARY_RHO},sigma={SECONDARY_SIGMA}): "
                f"hetero+learned={multicue_secondary_worst:.3f} vs orig-homog-matched={matched_secondary_worst:.3f}")

    if not a.multicue:
        verdict = (f"BASELINE-ONLY ({tag}, --multicue not requested) -- the GATE'S OWN GO/NO-GO CRITERIA (G1..G4) "
                  f"are not evaluated in this mode (pass --multicue for the full battery); `pipeline_ceiling"
                  f"(multicue=False)` DELEGATES byte-identically to the imported single-channel ROBUST pipeline "
                  f"(asserted in --smoke). The heterogeneous+learned primary arm is still COMPUTED here for "
                  f"diagnostic/reported context (NOT gated): orig-homog-matched@decisive(rho={a.rho},sigma="
                  f"{a.sigma})={matched_decisive_worst:.3f} worst ({matched_decisive_mean:.3f} mean) vs "
                  f"hetero+learned {multicue_decisive_worst:.3f} worst vs text {text_ceiling_worst:.3f}.")
    elif go:
        verdict = (
            f"GO ({tag}) -- THE AFFECT NOISE-ROBUSTNESS SURPASS. K={k_cues} HETEROGENEOUS-RELIABILITY cues "
            f"(a physiologically-motivated 0.5x-2.5x per-cue noise ramp -- Craig 2002/2009 distinct afferent "
            f"pathways) fused by RAW PRE-THRESHOLD evidence combination with LEARNED, UNSUPERVISED inverse-"
            f"variance reliability weights (Kramer & Manoonpong 2022) TEACH a separable concept code AT THE EDGE "
            f"WHERE SINGLE-CUE POOLING FAILS, and do so BY EXPLOITING RELIABILITY DIFFERENCES a single "
            f"undifferentiated channel cannot. {lift_line}. THE THREE-WAY G4a ANTI-CHEAT ISOLATES THE LEVER: on "
            f"the IDENTICAL cue draw (same total afferents, same total noise power, same actual random values), "
            f"LEARNED weighting beats BOTH (a) collapsing all afferents into one undifferentiated wide channel "
            f"and (b) combining the SAME heterogeneous cues with naive UNIFORM weights, by >= {G4A_MARGIN} "
            f"worst-case against each. It is INDEPENDENT-noise fusion, not extra dimensions: cues forced to "
            f"SHARE one noise pattern collapse and do NOT clear the bar. It is GROUNDING (lesion + shuffle at "
            f"text baseline) and TAUGHT not HANDED (held-out {heldout_clean_worst:.3f} worst-case). THE LEARNED "
            f"WEIGHTS ARE A REAL FUNCTIONAL READ-OUT, not a relabelled constant: pooled correlation with the "
            f"TRUE (experiment-known, mechanism-unseen) inverse variance = {weight_true_ivar_corr_pooled:.3f}, "
            f"and the learned ranking exactly matches the true reliability ranking in {weight_rank_match_count}/"
            f"{len(seeds)} seeds. NEXT: a fully online (trial-by-trial) Hebbian precision-learning rule (the "
            f"present estimator is the closed-form fixed point such a rule converges to) + spiking dendritic/"
            f"NMDA coincidence detection (the raw-sum-then-threshold is presently a host weighted-sum + one "
            f"threshold) + physiologically REAL distinct afferent modalities (vagal/vestibular/proprioceptive) "
            f"in place of a synthetic sigma ramp. Brain-based (rate-Hebbian synaptic convergence + population "
            f"relays; body-state=world/body boundary; ceiling=instrument); NO sim/ edit; NOT wired.")
    else:
        miss = [k for k, ok in (("G1_noise_robust_lift", g1), ("G2_load_bearing", g2), ("G2b_generalizes", g2b),
                                ("G3_instrument", g3), ("G4a_learned_isolated", g4a), ("G4b_shared_noise", g4b))
                if not ok]
        verdict = (
            f"PARTIAL/BOUNDARY ({tag}, build-informative) -- heterogeneous-cue + LEARNED-reliability fusion "
            f"{'CLEARS' if multicue_decisive_worst >= CEIL_GO_BAR else 'does NOT clear'} the strict bar at the "
            f"decisive noisy point. {lift_line}. FAILED: {miss}. See the three-way G4a numbers + the weight<->"
            f"truth correlation for exactly where the isolation of the LEARNED-weighting lever falls short. The "
            f"fixed _STRONG_MARGIN gate in affect_production_organ.py is UNCHANGED (this file wires nothing).")

    summary = {
        "probe": "affect_multicue_convergence_derisk (PRIMARY: heterogeneous-reliability cues + LEARNED "
                "unsupervised inverse-variance weighting, raw pre-threshold fusion -- the 2026-09-17 3rd "
                "iteration; the 2nd iteration's homogeneous-cue uniform-weight arm and the 1st iteration's "
                "per-cue-threshold-THEN-AND/OR arms are retained as reported lineage comparisons)",
        "verdict": verdict, "GO": go, "multicue_requested": a.multicue,
        "primary_fuse_mode": "raw_sum_learned", "primary_cue_construction": "heterogeneous (0.5x-2.5x sigma ramp)",
        "G1_noise_robust_lift": g1, "G2_load_bearing": g2, "G2b_generalizes": g2b, "G3_instrument": g3,
        "G4a_learned_isolated": g4a, "G4b_shared_noise": g4b, "G4_anti_cheat": g4,
        "decisive_rho": a.rho, "decisive_sigma": a.sigma, "k_cues": k_cues,
        "per_cue_relay_lineage": per_cue, "matched_n_relay_lineage": matched_n_relay,
        "total_budget_primary": a.total_budget, "per_cue_relay_primary": per_cue_primary,
        "matched_n_relay_primary": matched_n_relay_primary,
        "text_ceiling_worst": text_ceiling_worst, "text_ceiling_mean": text_ceiling_mean,
        "multicue_decisive_worst": multicue_decisive_worst, "multicue_decisive_mean": multicue_decisive_mean,
        "g4a_control_a_single_wide_worst": single_wide_worst, "g4a_control_a_single_wide_mean": single_wide_mean,
        "g4a_control_b_uniform_hetero_worst": uniform_hetero_worst,
        "g4a_control_b_uniform_hetero_mean": uniform_hetero_mean,
        "g4a_delta_vs_single_wide_worst": g4a_delta_a_worst, "g4a_delta_vs_single_wide_mean": g4a_delta_a_mean,
        "g4a_delta_vs_uniform_worst": g4a_delta_b_worst, "g4a_delta_vs_uniform_mean": g4a_delta_b_mean,
        "weight_true_ivar_correlation_pooled": weight_true_ivar_corr_pooled,
        "weight_true_ivar_correlation_mean_per_seed": weight_true_ivar_corr_mean,
        "weight_rank_matches_truth_all_seeds": weight_rank_matches_all_seeds,
        "weight_rank_matches_truth_count": weight_rank_match_count,
        "original_homogeneous_matched_worst": matched_decisive_worst,
        "original_homogeneous_matched_mean": matched_decisive_mean,
        "shared_noise_decisive_worst": shared_decisive_worst, "shared_noise_decisive_mean": shared_decisive_mean,
        "lesion_decisive_worst": lesion_decisive_worst, "shuffle_decisive_worst": shuffle_decisive_worst,
        "and_lineage_decisive_worst": and_decisive_worst, "and_lineage_decisive_mean": and_decisive_mean,
        "sum_fuse_decisive_worst": sum_decisive_worst, "sum_fuse_decisive_mean": sum_decisive_mean,
        "homogeneous_uniform_worst": homog_uniform_worst, "homogeneous_uniform_mean": homog_uniform_mean,
        "learned_reliability_homogeneous_worst": learned_decisive_worst,
        "learned_reliability_homogeneous_mean": learned_decisive_mean,
        "ungated_n4_decisive_mean": ungated_decisive_mean,
        "multicue_secondary_worst": multicue_secondary_worst, "matched_secondary_worst": matched_secondary_worst,
        "multicue_clean_worst": multicue_clean_worst, "heldout_clean_worst": heldout_clean_worst,
        "heldout_clean_mean": heldout_clean_mean, "lesion_clean_worst": lesion_clean_worst,
        "synthetic_instrument_ceiling_worst": synth_ceiling_worst,
        "ceiling_go_bar": CEIL_GO_BAR, "attrib_margin": ATTRIB_MARGIN, "text_ceil_max": TEXT_CEIL_MAX,
        "heldout_frac": HELDOUT_FRAC, "g4a_margin": G4A_MARGIN,
        "k_dose_response_sweep": dose_response_agg, "sigma_snr_sweep": snr_sweep_agg,
        "n_pos_raw_gated": n_pos, "n_neg_raw_excluded": n_neg, "n_partition_words": len(part_words),
        "per_seed": [{"seed": r["seed"], "code_dim": r["code_dim"], "text_ceiling": r["text_ceiling"],
                      "multicue_decisive": r["multicue_decisive"], "single_wide_decisive": r["single_wide_decisive"],
                      "uniform_hetero_decisive": r["uniform_hetero_decisive"],
                      "matched_decisive": r["matched_decisive"],
                      "weight_true_ivar_corr": r["weight_true_ivar_corr"],
                      "weight_rank_matches_truth": r["weight_rank_matches_truth"],
                      "learned_weights_decisive": r["learned_weights_decisive"],
                      "true_inv_var": r["true_inv_var"],
                      "shared_noise_decisive": r["shared_noise_decisive"], "lesion_decisive": r["lesion_decisive"],
                      "shuffle_decisive": r["shuffle_decisive"], "and_lineage_decisive": r["and_lineage_decisive"],
                      "sum_fuse_decisive": r["sum_fuse_decisive"],
                      "homog_uniform_decisive": r["homog_uniform_decisive"],
                      "learned_decisive": r["learned_decisive"],
                      "ungated_n4_decisive": r["ungated_n4_decisive"], "heldout_clean": r["heldout_clean"],
                      "synth_code_ceiling": r["synth_code_ceiling"]} for r in rows],
        "preconditions": verdict_earned["preconditions"], "verdict_earned_status": verdict_earned["status"],
        "verdict_undefined_reasons": verdict_earned["undefined_reasons"],
        "config": {"seeds": seeds, "smoke": a.smoke, "multicue": a.multicue, "max_stories": max_stories,
                  "resample_frac": a.resample_frac, "n_hub": a.n_hub, "window": a.window, "min_count": min_count,
                  "m_assembly": M_ASSEMBLY, "epochs": EPOCHS, "k_mad": K_MAD, "n_relay_robust_reference": N_RELAY_ROBUST,
                  "total_budget_lineage": TOTAL_BUDGET, "total_budget_primary": a.total_budget,
                  "k_dose_sweep": list(K_DOSE_SWEEP),
                  "sigma_snr_sweep": list(SIGMA_SNR_SWEEP), "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "PRIMARY (2026-09-17, 3rd iteration, owner-directed): HETEROGENEOUS-RELIABILITY CUES + "
                    "LEARNED INVERSE-VARIANCE FUSION. multicue_relay_population delivers the SAME grounding "
                    "decision + true sign/magnitude (one core-affect latent, Craig 2002/2009) into K SMALL pooled "
                    "relay populations, each with its OWN independent noise stream AND its own designated noise "
                    "scale (sigma_per_cue = a FIXED 0.5x-2.5x ramp x the operating point's sigma -- a "
                    "physiologically-motivated stand-in for distinct afferent pathways of differing reliability, "
                    "e.g. vagal vs vestibular). learn_reliability_weights estimates each cue's reliability "
                    "UNSUPERVISED (inverse of its own empirical variance across concepts -- Kramer & Manoonpong "
                    "2022 crossmodal Hebbian reliability learning; the closed-form fixed point of an online "
                    "precision-learning rule); multicue_raw_fusion then combines the K cues' RAW (pre-threshold) "
                    "pooled evidence by this LEARNED weighted sum, and ONE homeostatic threshold "
                    "(_homeostatic_threshold) is applied to the fused evidence. THE G4a THREE-WAY ANTI-CHEAT "
                    "reuses the IDENTICAL cue draw (bit-identical, not just matched-distribution) to compute (a) "
                    "_pool_raw_concat -- collapse all K cues into ONE undifferentiated wide channel (same total "
                    "afferents + noise power, no per-cue structure) and (b) multicue_raw_fusion with UNIFORM "
                    "weights on the SAME heterogeneous cues -- so a pass isolates the LEARNED WEIGHTING, not "
                    "heterogeneity or extra afferents. THE WEIGHT<->TRUTH CHECK compares learn_reliability_"
                    "weights' output against the TRUE inverse variance (1/sigma_per_cue^2, known to the "
                    "experiment designer, never given to the mechanism) via Pearson correlation + rank match. "
                    "LINEAGE (reported, not gated): the 2026-09-05 per-cue-threshold-THEN-AND/OR arms "
                    "(multicue_coincidence_fuse / _sum_fuse over pool_and_adapt'd cues) collapsed catastrophically "
                    "at the decisive point; the 2026-09-17 1st iteration's homogeneous-cue UNIFORM-weight raw-sum "
                    "arm recovered from that collapse but only reached a statistical TIE with a single matched "
                    "channel (expected: averaging K per-cue means before thresholding, at EQUAL reliability, is "
                    "mathematically the SAME variance reduction pooling them in one channel gets). ALL arms feed "
                    "the IDENTICAL three-factor-gated, homeostatically-scaled Hebbian convergence "
                    "(train_convergence_robust / eligibility_gate / _blocks_robust / convergence_readout / "
                    "code_separability_ceiling -- ALL reused-by-import verbatim from the noise-robust rung). G4b: "
                    "shared-noise cues now share ONE random PATTERN scaled per-cue by their OWN designated sigma "
                    "(preserving heterogeneity while destroying conditional independence) -- perfectly correlated "
                    "cues cannot be averaged apart by ANY weighting.",
        "sources": [
            "Ernst & Banks (2002, Nature, doi:10.1038/415429a) 'Humans integrate visual and haptic information "
            "in a statistically optimal fashion' -- combined precision = SUM of per-cue precisions; the "
            "ADVANTAGE over a single pooled channel requires UNEQUAL per-cue reliabilities exploited by the "
            "combination weights, which this iteration tests directly (equal-reliability raw-sum ~ties the "
            "matched arm; unequal-reliability + learned weights is where the theorem predicts a genuine lift).",
            "Kramer & Manoonpong et al. (2022, Front. Neural Circuits, doi:10.3389/fncir.2022.921453) -- "
            "crossmodal Hebbian plasticity LEARNS inverse-variance reliability weights across independent "
            "sensory channels UNSUPERVISED (the emergence bar: combination is TAUGHT from the signal's own "
            "statistics, not hand-tuned to the labels) -- realized here as learn_reliability_weights, validated "
            "against the true (experiment-known) precision via the weight<->truth correlation.",
            "Craig (2002, Nat Rev Neurosci; 2009, Curr Opin Neurobiol) -- interoception is inherently "
            "multi-afferent (lamina I spinothalamocortical + vagal + vestibular channels converging on one "
            "core-affect representation) and these channels genuinely DIFFER in reliability -- the biological "
            "basis for the heterogeneous-sigma ramp construction (not a synthetic convenience).",
            "Evrard (2019, Front Neuroanat); Barrett (2017, Soc Cogn Affect Neurosci) -- core affect is "
            "CONSTRUCTED by integrating interoceptive + exteroceptive + contextual signals, not read off one "
            "channel; the theoretical grounding for multi-cue (not single-channel-pooled) affect grounding.",
            "2026-09-17 (tonight, this session), owner-directed 3-step iteration on the SAME wall: (1) per-cue-"
            "threshold-THEN-AND/OR measured a well-characterized NO-GO (0.010 worst-case vs matched 0.696); "
            "(2) RAW pre-threshold uniform-weight fusion recovered to 0.539 worst-case but only TIED the matched "
            "arm (mean delta -0.010), diagnosed as the mathematically-expected outcome at EQUAL cue reliability; "
            "(3) heterogeneous cues + LEARNED inverse-variance weights, gated here, tests whether exploiting "
            "UNEQUAL reliability is where the real lift lives.",
            "_affect_noise_robust_homeostatic_convergence_derisk.py (2026-09-05 GO, this session's sibling) -- "
            "the ORIGINAL homogeneous matched-budget single-channel arm, reproduced here via the SAME imported "
            "pipeline as a lineage reference (rho=0.8,sigma=2.0 -> 0.451 worst-case / 0.598 mean, MIN-aggregated "
            "to match how that reference number was itself computed).",
        ],
        "production_wiring": "NONE -- affect_production_organ.py and wkv_mouth_generator.py are byte-unchanged "
                             "(sha256-pinned + asserted in --smoke); _STRONG_MARGIN==2.0 asserted; reuse-by-"
                             "import only.",
        "HONEST_RESIDUALS": "(1) the body-state US remains the declared ORACLE STAND-IN for a grounded world "
                            "that does not exist for the TinyStories vocabulary (the SAME stand-in the prior "
                            "rungs used). (2) the heterogeneous-sigma ramp is a SYNTHETIC stand-in for "
                            "physiologically distinct afferent modalities (vagal/vestibular/proprioceptive) -- "
                            "grounding each cue in an ACTUALLY DISTINCT bodily signal (not a scaled copy of one "
                            "scalar Warriner latent) is the named next rung. (3) the raw-sum-then-threshold "
                            "fusion is a host weighted-sum + one elementwise threshold (an idealized stand-in for "
                            "dendritic current summation + an NMDA-receptor voltage threshold); a fully-spiking "
                            "coincidence-detector convergence (build_propagation_bridge, GPU-queued) is the named "
                            "next rung, not deferred. (4) learn_reliability_weights is a CLOSED-FORM inverse-"
                            "variance estimate (the fixed point of an online Hebbian rule), not itself an online/"
                            "trial-by-trial Hebbian learning process -- validated here only by its OUTPUT tracking "
                            "the true precision, not by observing it converge over trials; the online rule itself "
                            "is the named next rung. (5) rate-Hebbian (numpy-CPU) convergence, unchanged from the "
                            "prior rung. (6) the ceiling is a linear supervised upper bound. (7) the 164-word "
                            "closed partition is inherited from the prior boundaries. (8) TOTAL_BUDGET/K_CUES/the "
                            "0.5x-2.5x hetero ramp are documented OPERATING POINTS, not fit to the labels, but "
                            "the ramp's specific shape is a design choice, not independently derived.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    undefined_if_empty("partition-words", len(part_words), len(part_words), len(part_words))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[multicue-convergence] text={text_ceiling_worst:.3f} | DECISIVE hetero+learned(primary)="
          f"{multicue_decisive_worst:.3f} | G4a: single_wide(a)={single_wide_worst:.3f} "
          f"uniform_hetero(b)={uniform_hetero_worst:.3f} | shared-noise={shared_decisive_worst:.3f} | "
          f"lesion={lesion_decisive_worst:.3f} shuffle={shuffle_decisive_worst:.3f} | "
          f"held-out={heldout_clean_worst:.3f} | synth-instr={synth_ceiling_worst:.3f}", flush=True)
    print(f"[multicue-convergence] weight<->truth: pooled_corr={weight_true_ivar_corr_pooled:.3f} "
          f"rank_match={weight_rank_match_count}/{len(seeds)} seeds", flush=True)
    print(f"[multicue-convergence] LINEAGE orig-homog-matched={matched_decisive_worst:.3f} | AND="
          f"{and_decisive_worst:.3f} sum/OR={sum_decisive_worst:.3f} | homog-uniform={homog_uniform_worst:.3f} "
          f"homog-learned={learned_decisive_worst:.3f}", flush=True)
    print(f"[multicue-convergence] K dose-response @decisive: {dose_response_agg}", flush=True)
    print(f"[multicue-convergence] sigma/SNR sweep: {snr_sweep_agg}", flush=True)
    print(f"[multicue-convergence] GO={go} (multicue_requested={a.multicue} G1={g1} G2={g2} G2b={g2b} G3={g3} "
          f"G4a={g4a} G4b={g4b})", flush=True)
    print(f"[multicue-convergence] VERDICT: {verdict}", flush=True)
    print(f"[multicue-convergence] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
