"""GROUNDED-EXPERIENCE STREAM + HEBBIAN CONVERGENCE (2026-09-05) — the NAMED next build after the grounded-code PARTIAL.

WHERE THIS SITS (do NOT re-derive the prior two rungs).
  * `2026-09-05-affect-gate-embodied-US-necessary-not-sufficient-concept-code-must-be-grounded-BOUNDARY.md` proved
    (6-seed) that the TEXT-derived affect concept code does not separate affect from register-neutral words even
    under a PERFECT embodied US: a supervised label-given noise-free ridge CEILING reads ~0.000 worst-case
    recall@FP0. The register confound lives in the CONCEPT CODE (perception), so the code must be GROUNDED.
  * `2026-09-05-affect-gate-grounded-concept-code-lifts-ceiling-requirements-derisk-PARTIAL.md` de-risked the
    REQUIREMENTS with a RAW ORACLE body-state block FUSED (concatenated) onto the text code and read the SAME
    ceiling: a grounded code IS separable where text is not (clean/full 1.000 vs text ~0; lesion/shuffle at
    baseline). GO-bar not met only because ZERO-FP was the binding constraint (5%-FP -> 0.43 at a realistic point).
    NAMED next mechanism: a grounded-experience STREAM -- an embodied/interoceptive per-concept body-state delivered
    via the board #49/#84 interoceptive relay + a HEBBIAN CONVERGENCE (the vision->concept `_genfrontier_capstone`
    template) that TEACHES a concept code, rather than a hand-set block fused on.

WHAT THIS RUNNER BUILDS (the missing piece: an EMERGENT, brain-based concept code, not an oracle-handed one).
  The PARTIAL's grounded code was HAND-SET (the oracle body-state axes were concatenated verbatim, then read). This
  runner replaces that with a genuine SYNAPTIC CONVERGENCE that LEARNS the concept code:
    (1) GROUNDED-EXPERIENCE STREAM (world/body -- legit stand-in for US delivery, the SAME discipline the embodied-US
        runner + board #49/#84 use). When a concept's referent is "experienced", the world/body delivers a signed
        affect CURRENT of its true affect magnitude (Warriner, the grounded-world physics stand-in) into
        INTEROCEPTIVE RELAY POOLS: +affect -> a COMFORT pool, -affect -> a DISCOMFORT pool, |affect| -> an AROUSAL
        pool (the board #49/#84 relay structure -- concepts sharing a bodily consequence drive the SAME relay pool).
        Coverage rho (only a fraction of affect concepts are ever experienced) + afferent NOISE sigma degrade it, as
        a real teacher would. This is a per-concept body-state STREAM, NOT a code axis: it is the relay-pool
        ACTIVATION, and the CODE is what a convergence LEARNS from it.
    (2) HEBBIAN CONVERGENCE (BRAIN-BASED -- the load-bearing new piece). A concept ASSEMBLY layer of M rate neurons
        receives BOTH the (register-confounded) TEXT perception code AND the interoceptive relay activation, through
        feedforward synapses W learned by a COMPETITIVE, Oja-stabilized HEBBIAN rule (emergent -- W starts
        small-random, is never hand-set). Because the COMFORT / DISCOMFORT relay axes are the RELIABLE, SHARED
        structure across grounded affect concepts (the exact structure the text code lacks), the competitive
        convergence self-organizes assembly neurons SELECTIVE to those body-state axes (the EMERGE-34 / vision->
        concept pattern: shared features -> assembly selectivity), while the idiosyncratic register-confounded text
        dims do not yield consistent assembly selectivity. The LEARNED CONCEPT CODE = the assembly's response, read
        by the SAME validated separability CEILING instrument (reuse-by-import, verbatim).

WHY THIS IS A GENUINE RUNG, NOT THE PARTIAL WITH EXTRA STEPS (the anti-cheats ARE the deliverable):
  * LESION (load-bearing anti-hollow): deliver NO body-state during learning (US:=0) -> the assembly can only learn
    from the text code -> the learned code must collapse to the TEXT baseline (~0). If it did not, the convergence
    would be reading text, not grounding. THIS is the decisive control.
  * SHUFFLE: permute which concept receives which body-state (destroy the concept<->body-state binding) -> the
    convergence has no consistent shared axis to bind -> the learned code must collapse. A positive result must
    depend on the RIGHT concept carrying the RIGHT body-state.
  * HELD-OUT CONVERGENCE GENERALIZATION (the "TAUGHT not HANDED" criterion -- the piece the oracle FUSION could not
    have): hold a fraction of concepts OUT of the Hebbian weight training entirely; read their learned code through
    the assembly trained ONLY on the OTHER concepts. If the convergence merely memorized per-concept assignments,
    held-out concepts would not separate. If it learned a REUSABLE text+body-state -> code map, they do. This is
    what makes the code TAUGHT (a generalizing synaptic map) rather than a look-up of the oracle labels.
  * TEXT-ONLY TRANSFER (the deepest honest read, reported): grounding present ONLY during learning, ABSENT at test
    (intero:=0 at readout) -> does grounding-during-learning REORGANIZE the text->concept map so the WORD ALONE
    evokes a more separable code than the raw text code? (Likely partial -- the text code is confounded -- but it is
    the strongest test of "grounding TEACHES perception", so it is measured, not deferred.)
  * INSTRUMENT (reused verbatim): synthetic clean code -> ceiling ~1.0; text code -> < 0.2 -> the probe discriminates.
  * TEXT-ONLY baseline: reproduce the BOUNDARY's ~0 text ceiling on the SAME partition + seeds (like-for-like).

PRE-REGISTERED GO GATE (fixed BEFORE the 6-seed; a grounded-TEACHER verdict, NOT a gate retirement):
  G1 LIFT         the grounded-TAUGHT (Hebbian-learned, grounded-recall readout) concept code's worst-case ceiling
                  (min across seeds) >= CEIL_GO_BAR (0.5) at joint-FP=0 at a REALISTIC operating point
                  (rho >= RHO_REAL, sigma <= SIGMA_REAL) -- i.e. it CLEARS where the text code is ~0.
  G2 LOAD-BEARING the LESION (no body-state at learning) AND the SHUFFLE control both stay <= text_ceiling +
                  ATTRIB_MARGIN (grounding is what lifts it, not the convergence machinery or extra dims).
  G2b GENERALIZES the HELD-OUT-concept learned code (convergence trained on OTHER concepts) clears the bar at the
                  demonstrable clean/full operating point (rho=1.0, sigma=0.0) -- the code is TAUGHT (a generalizing
                  map), not HANDED (a per-concept lookup).
  G3 INSTRUMENT   synthetic clean-code ceiling >= 0.5 AND text-code ceiling < 0.2 on the SAME partition + seeds.
GO iff G1 AND G2 AND G2b AND G3 ==> "an emergent Hebbian convergence over a grounded-experience stream TEACHES a
     concept code that lifts the separability ceiling the text code cannot, and it generalizes." (NOT "gate retired".)
Reported (decisive, not all gated): the (rho x sigma) frontier of the LEARNED code; the relaxed-FP sensitivity (the
     PARTIAL's binding-constraint check); text-only transfer; clean/full learned vs the PARTIAL's raw-fusion 1.000.

BRAIN-BASED / SCOPE. Host is legit ONLY for the world/body US DELIVERY (the interoceptive affect current) + the
corpus stream. The CONVERGENCE is neurons/synapses: rate assembly neurons + a competitive Oja-Hebbian FF map
(emergent, never hand-set) -- the SAME level the affect lane accepts as SYNAPTIC (selforg_opponent_weights is a
rate-level three-factor Hebbian outer-product), with the SAME declared residual: a fully-spiking on-substrate
convergence (reuse `_genfrontier` build_propagation_bridge, GPU-queued) is the NEXT rung, named not deferred.
HONEST RESIDUALS: (1) the body-state US is a declared ORACLE STAND-IN for a grounded world that does not exist for
the TinyStories vocabulary (the SAME stand-in the embodied-US + grounded-code runners used); this measures whether a
convergence CAN teach a separable code GIVEN such a stream, and at what coverage/noise -- it does NOT deliver a real
grounded world. (2) rate-Hebbian (numpy-CPU) convergence; fully-spiking write = the named next rung. (3) the ceiling
is a linear supervised UPPER BOUND (the spiking opponent's mild nonlinearity was measured NOT to help by the prior
boundaries). (4) the 164-word closed partition is inherited from the prior boundaries.
NOT WIRED: nothing here touches affect_production_organ.py / wkv_mouth_generator.py (byte-unchanged; _STRONG_MARGIN
stays). Additive, default-off, numpy-CPU, reuse-by-import, NO sim/ edit.

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._affect_grounded_experience_stream_hebbian_derisk --smoke
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._affect_grounded_experience_stream_hebbian_derisk \
                  --seeds 42 43 44 100 101 102 \
                  --out research/findings/raw/_affect_grounded_experience_stream_hebbian_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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
from tools.lab import void_if, undefined_if_empty, attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affect_grounded_experience_stream_hebbian.json"

CEIL_GO_BAR = 0.5        # pre-registered: the ceiling the grounded-TAUGHT code must clear (== the lane's bar)
RHO_REAL = 0.6           # "realistic" coverage a grounded teacher could plausibly deliver
SIGMA_REAL = 1.0         # "realistic" interoceptive afferent noise at the realistic operating point
ATTRIB_MARGIN = 0.15     # G2: lesion + shuffle ceilings must stay within this of the text ceiling
TEXT_CEIL_MAX = 0.20     # G3: the text code ceiling must read low (reproduce the boundary) for the instrument test
HELDOUT_FRAC = 0.30      # G2b: fraction of concepts held OUT of the Hebbian weight training (generalization test)

RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SIGMA_GRID = (0.0, 0.5, 1.0, 2.0)

# Relaxed-FP sensitivity (reported, NOT gated): mirrors the PARTIAL so the two rungs are directly comparable.
FP_TOLS = (0.0, 0.05, 0.10)
FP_POINTS = ((0.6, 1.0), (0.8, 0.5), (1.0, 0.5), (1.0, 1.0))

# convergence hyperparameters (fixed; documented -- an operating point, not a fit to the labels)
M_ASSEMBLY = 48          # concept-assembly rate neurons
N_RELAY = 4              # interoceptive relay neurons per body-state channel (comfort/discomfort/arousal)
EPOCHS = 40
ETA = 0.05               # Hebbian learning rate
K_WTA = 12               # soft k-WTA competition (active assembly neurons per concept)
RELAY_NOISE = 0.5        # afferent relay noise scale (multiplies sigma)


def grounded_experience_stream(part_words, raw_gate, seed, rho, sigma, shuffle=False, lesion=False):
    """The GROUNDED-EXPERIENCE STREAM (host = world/body boundary, a declared stand-in for a grounded-perception
    teacher's US delivery). Returns the INTEROCEPTIVE RELAY activation matrix (n x 3*N_RELAY): comfort / discomfort
    / arousal pools (board #49/#84 structure). +affect concepts drive the COMFORT pool, -affect the DISCOMFORT pool,
    |affect| the AROUSAL pool -- concepts sharing a bodily consequence drive the SAME relay pool (the shared axis a
    text co-occurrence code lacks). Only a rho fraction of AFFECT concepts are grounded (COVERAGE); Gaussian afferent
    NOISE sigma on every relay neuron (incl. neutral -> false grounding).

    lesion=True -> NO body-state is ever delivered (pure relay noise) = the anti-hollow NO-GROUNDING control.
    shuffle=True -> the concept<->body-state binding is permuted = the SHUFFLE control.
    Neither must let the convergence teach a separable code if the lift is genuine grounding signal."""
    rng = np.random.default_rng(seed + 91_000)
    n = len(part_words)
    val = np.array([(WARRINER[w][0] - 5.0) / 4.0 for w in part_words])    # signed true affect magnitude in ~[-1,1]
    sign = np.sign(val)
    mag = np.abs(val)
    aff_idx = np.where(raw_gate)[0]
    grounded = np.zeros(n, bool)
    if rho > 0 and len(aff_idx) > 0:
        k = int(round(rho * len(aff_idx)))
        chosen = rng.choice(aff_idx, size=k, replace=False) if k > 0 else np.array([], int)
        grounded[chosen] = True
    if shuffle:                                                           # destroy the concept<->body-state binding
        perm = rng.permutation(n)
        sign = sign[perm]; mag = mag[perm]; grounded = grounded[perm]
    comfort = np.zeros(n); discomfort = np.zeros(n); arousal = np.zeros(n)
    if not lesion:
        g_pos = grounded & (sign > 0)
        g_neg = grounded & (sign < 0)
        comfort[g_pos] = mag[g_pos]                                       # +affect -> shared COMFORT relay pool
        discomfort[g_neg] = mag[g_neg]                                    # -affect -> shared DISCOMFORT relay pool
        arousal[grounded] = mag[grounded]                                 # |affect| -> AROUSAL relay pool
    pools = []
    for ch in (comfort, discomfort, arousal):
        for _ in range(N_RELAY):                                          # a small POPULATION relay per channel
            pools.append(ch + np.abs(rng.standard_normal(n)) * sigma * RELAY_NOISE)
    X_intero = np.maximum(np.stack(pools, axis=1), 0.0)                   # non-negative relay rates (PPMI-like)
    return X_intero, int(grounded.sum())


def _blocks_scaled(text_codes, X_intero, intero_present=True):
    """Concatenate the TEXT perception code and the interoceptive relay activation into ONE convergence input, each
    modality block L2-normalized per concept so neither is scale-privileged (the interoceptive current then competes
    on equal footing for the assembly's Hebbian selectivity). intero_present=False zeros the relay block (the
    TEXT-ONLY TRANSFER readout: grounding absent at test)."""
    T = text_codes / (np.linalg.norm(text_codes, axis=1, keepdims=True) + 1e-12)
    if intero_present:
        I = X_intero / (np.linalg.norm(X_intero, axis=1, keepdims=True) + 1e-12)
    else:
        I = np.zeros_like(X_intero)
    return np.concatenate([T, I], axis=1)


def arousal_gate(X_intero):
    """LABEL-FREE neuromodulatory gate = the AROUSAL relay-pool magnitude (the interoceptive US SALIENCE the
    world/body actually delivered), median-normalized. It uses the afferent DRIVE, never the affect label -- a real
    US raises arousal, a neutral concept carries only relay noise. Feeds the three-factor plasticity gate below."""
    ar = X_intero[:, 2 * N_RELAY:3 * N_RELAY].mean(axis=1)                # the AROUSAL channel's relay pool
    med = float(np.median(ar[ar > 0])) if np.any(ar > 0) else 0.0
    return np.clip(ar / (med + 1e-9), 0.0, 3.0) if med > 0 else np.zeros_like(ar)


def train_convergence(X_train, seed, m=M_ASSEMBLY, epochs=EPOCHS, eta=ETA, k_wta=K_WTA, us_gate=None):
    """The HEBBIAN CONVERGENCE (brain-based): a concept-assembly FF weight matrix W (m x Din) learned by a
    competitive, Oja-stabilized Hebbian rule over the training concepts' convergence inputs. W is EMERGENT -- it
    starts small-random (never hand-set) and self-organizes assembly neurons selective to the reliable shared input
    directions (the interoceptive comfort/discomfort axes), exactly the EMERGE-34 / vision->concept convergence
    pattern. Competition = soft k-WTA per concept (only the top-k assembly neurons stay active, so assemblies
    specialize). Oja's rule (dW = eta * a (x - a w)) keeps the weights bounded without a hand-set clamp.

    us_gate (optional, len n): the THREE-FACTOR neuromodulatory gate (pre x post x US) -- the plasticity rate is
    scaled by the US salience actually delivered (arousal_gate), so noise-only concepts (no real US) barely update.
    us_gate=None -> ungated (the pre-registered primary; byte-identical to the original 2-factor rule)."""
    rng = np.random.default_rng(seed + 123)
    n, din = X_train.shape
    g = np.ones(n) if us_gate is None else np.asarray(us_gate, float)
    W = np.abs(rng.standard_normal((m, din))) * 0.01                      # small non-negative excitatory FF init
    W /= (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    idx = np.arange(n)
    for _ep in range(epochs):
        rng.shuffle(idx)
        for c in idx:
            x = X_train[c]
            a = np.maximum(W @ x, 0.0)                                    # rectified assembly rate
            if k_wta < m:                                                 # soft k-WTA competition
                thr = np.partition(a, -k_wta)[-k_wta]
                a = np.where(a >= thr, a, 0.0)
            W += eta * g[c] * (np.outer(a, x) - (a ** 2)[:, None] * W)   # three-factor-gated Oja-Hebbian update
    return W


def convergence_readout(W, X):
    """The LEARNED CONCEPT CODE = the assembly's population response to each concept's convergence input, divisively
    normalized (population gain control). This n x m code is what the SAME validated ceiling instrument reads."""
    A = np.maximum(X @ W.T, 0.0)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def learned_code_ceiling(text_codes, X_intero, raw_gate, seed, intero_at_test=True, heldout=False):
    """Train the Hebbian convergence, read the learned concept code, and return its separability ceiling. If
    heldout=True, W is trained on a random (1-HELDOUT_FRAC) subset of concepts and the ceiling is read ONLY on the
    held-out concepts (the generalization / TAUGHT-not-HANDED test); the ceiling's own k-fold CV then holds out
    again within that set. Otherwise W trains on all concepts and the ceiling reads all (grounded-recall)."""
    X_full = _blocks_scaled(text_codes, X_intero, intero_present=True)
    if not heldout:
        W = train_convergence(X_full, seed)
        X_test = _blocks_scaled(text_codes, X_intero, intero_present=intero_at_test)
        code = convergence_readout(W, X_test)
        return code_separability_ceiling(code, raw_gate, seed)
    rng = np.random.default_rng(seed + 555)
    n = len(raw_gate)
    perm = rng.permutation(n)
    n_ho = max(int(round(HELDOUT_FRAC * n)), 1)
    ho = np.zeros(n, bool); ho[perm[:n_ho]] = True
    tr = ~ho
    if raw_gate[ho].sum() == 0 or (~raw_gate[ho]).sum() == 0:             # need both classes in the held-out set
        return 0.0
    W = train_convergence(X_full[tr], seed)                               # convergence NEVER sees held-out concepts
    X_test = _blocks_scaled(text_codes, X_intero, intero_present=intero_at_test)
    code_ho = convergence_readout(W, X_test[ho])
    return code_separability_ceiling(code_ho, raw_gate[ho], seed)


def run_seed(seed, stories, part_words, raw_gate, n_hub, window, min_count, resample_frac, verbose=False):
    """One seed: build the text code; measure the text-only ceiling, then the grounded-TAUGHT learned-code ceiling
    across the (rho x sigma) grid (grounded-recall); + lesion / shuffle / held-out / text-only-transfer controls at
    the realistic AND clean/full operating points; + the reused synthetic instrument validation."""
    sub = resample_stories(stories, resample_frac, seed)
    vocab, codes, _codes_read, _rel = _codes_for(sub, n_hub, window, min_count)
    widx = {w: i for i, w in enumerate(vocab)}
    part_idx = np.array([widx[w] for w in part_words])
    text_codes = np.asarray(codes[part_idx], float)
    D = text_codes.shape[1]

    text_ceiling = code_separability_ceiling(text_codes, raw_gate, seed)  # reproduce the BOUNDARY's ~0 (like-for-like)

    # (rho x sigma) grid: the grounded-TAUGHT learned code (grounded-recall readout)
    grid = []
    for rho in RHO_GRID:
        for sigma in SIGMA_GRID:
            Xi, n_grounded = grounded_experience_stream(part_words, raw_gate, seed, rho, sigma)
            c_learn = learned_code_ceiling(text_codes, Xi, raw_gate, seed)
            grid.append({"rho": rho, "sigma": sigma, "n_grounded": n_grounded, "learned_ceiling": c_learn})

    def _cell(rho, sigma):
        return next(g["learned_ceiling"] for g in grid if g["rho"] == rho and g["sigma"] == sigma)

    # controls at the REALISTIC operating point (rho=0.6, sigma=1.0)
    Xi_real, _ = grounded_experience_stream(part_words, raw_gate, seed, RHO_REAL, SIGMA_REAL)
    real_learned = _cell(RHO_REAL, SIGMA_REAL)
    Xi_les, _ = grounded_experience_stream(part_words, raw_gate, seed, RHO_REAL, SIGMA_REAL, lesion=True)
    lesion_ceiling = learned_code_ceiling(text_codes, Xi_les, raw_gate, seed)                       # G2 anti-hollow
    Xi_shuf, _ = grounded_experience_stream(part_words, raw_gate, seed, RHO_REAL, SIGMA_REAL, shuffle=True)
    shuffle_ceiling = learned_code_ceiling(text_codes, Xi_shuf, raw_gate, seed)                     # G2 anti-hollow
    textonly_transfer_real = learned_code_ceiling(text_codes, Xi_real, raw_gate, seed, intero_at_test=False)

    # clean/full operating point (rho=1.0, sigma=0.0) -- the DEMONSTRABLE grounding point
    Xi_clean, _ = grounded_experience_stream(part_words, raw_gate, seed, 1.0, 0.0)
    clean_learned = _cell(1.0, 0.0)
    heldout_clean = learned_code_ceiling(text_codes, Xi_clean, raw_gate, seed, heldout=True)        # G2b generalizes
    heldout_real = learned_code_ceiling(text_codes, Xi_real, raw_gate, seed, heldout=True)
    textonly_transfer_clean = learned_code_ceiling(text_codes, Xi_clean, raw_gate, seed, intero_at_test=False)
    lesion_clean = learned_code_ceiling(text_codes,
                                        grounded_experience_stream(part_words, raw_gate, seed, 1.0, 0.0,
                                                                   lesion=True)[0], raw_gate, seed)

    # relaxed-FP sensitivity (grounded-recall learned code) at the FP_POINTS -- the PARTIAL's binding-constraint check
    fp_sens = []
    for (rho, sigma) in FP_POINTS:
        Xi, _ = grounded_experience_stream(part_words, raw_gate, seed, rho, sigma)
        Xf = _blocks_scaled(text_codes, Xi, intero_present=True)
        W = train_convergence(Xf, seed)
        code = convergence_readout(W, Xf)
        row = {"arm": "grounded", "rho": rho, "sigma": sigma}
        for tol in FP_TOLS:
            row[f"fp{tol}"] = code_separability_ceiling(code, raw_gate, seed, max_fp_frac=tol)
        fp_sens.append(row)
    for label, Xi in (("lesion", Xi_les), ("shuffle", Xi_shuf)):                                    # controls at same FP
        Xf = _blocks_scaled(text_codes, Xi, intero_present=True)
        W = train_convergence(Xf, seed)
        code = convergence_readout(W, Xf)
        rowc = {"arm": label, "rho": RHO_REAL, "sigma": SIGMA_REAL}
        for tol in FP_TOLS:
            rowc[f"fp{tol}"] = code_separability_ceiling(code, raw_gate, seed, max_fp_frac=tol)
        fp_sens.append(rowc)

    # US-GATED THREE-FACTOR convergence (REPORTED, not in the GO gate): the biologically-correct companion process
    # the ungated rule proxies away -- plasticity gated by the delivered US salience (arousal_gate, label-free), so
    # noise-only concepts do not bind. Does neuromodulatory gating recover the noisy realistic point? -----------------
    def _gated_ceiling(Xi, fp=0.0):
        Xf = _blocks_scaled(text_codes, Xi, intero_present=True)
        W = train_convergence(Xf, seed, us_gate=arousal_gate(Xi))
        return code_separability_ceiling(convergence_readout(W, Xf), raw_gate, seed, max_fp_frac=fp)
    gated_real = _gated_ceiling(Xi_real)
    gated_real_fp5 = _gated_ceiling(Xi_real, 0.05)
    gated_clean = _gated_ceiling(Xi_clean)
    gated_lesion = _gated_ceiling(Xi_les)                                                           # gated anti-hollow

    synth = synthetic_separable_gate(seed, raw_gate, D)                                             # G3 instrument

    if verbose:
        print(f"  [seed {seed}] D={D} text={text_ceiling:.3f} | learned@real(rho={RHO_REAL},sig={SIGMA_REAL})="
              f"{real_learned:.3f} clean/full={clean_learned:.3f} | lesion={lesion_ceiling:.3f} "
              f"shuffle={shuffle_ceiling:.3f} | held-out(clean)={heldout_clean:.3f} | text-only-xfer(clean)="
              f"{textonly_transfer_clean:.3f} | GATED@real={gated_real:.3f}(fp5%={gated_real_fp5:.3f}) "
              f"GATED_clean={gated_clean:.3f} | synth_instr={synth['code_ceiling']:.3f}", flush=True)
    return {"seed": int(seed), "code_dim": int(D), "text_ceiling": text_ceiling, "grid": grid,
            "real_learned_ceiling": real_learned, "clean_learned_ceiling": clean_learned,
            "lesion_ceiling": lesion_ceiling, "lesion_clean_ceiling": lesion_clean,
            "shuffle_ceiling": shuffle_ceiling, "heldout_clean_ceiling": heldout_clean,
            "heldout_real_ceiling": heldout_real, "textonly_transfer_real_ceiling": textonly_transfer_real,
            "textonly_transfer_clean_ceiling": textonly_transfer_clean,
            "gated_real_ceiling": gated_real, "gated_real_fp5_ceiling": gated_real_fp5,
            "gated_clean_ceiling": gated_clean, "gated_lesion_ceiling": gated_lesion,
            "synth_code_ceiling": float(synth["code_ceiling"]), "fp_sensitivity": fp_sens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=CANONICAL_SEEDS)
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny corpus -- proves it RUNS + controls live")
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

    t0 = time.time()
    print(f"[grounded-exp-stream-hebbian] seeds={seeds} smoke={a.smoke} max_stories={max_stories} n_hub={a.n_hub} "
          f"M={M_ASSEMBLY} epochs={EPOCHS} backend={os.environ.get('SIM_BACKEND')}", flush=True)
    stories = load_stories(max_stories)
    part_words, raw_gate = build_partition(stories, seeds, a.resample_frac, min_count)
    void_if(len(part_words) < 20, f"only {len(part_words)} common partition words")
    n_pos, n_neg = int(raw_gate.sum()), int((~raw_gate).sum())
    void_if(n_pos == 0 or n_neg == 0, f"degenerate partition n_pos={n_pos} n_neg={n_neg}")
    print(f"  partition: {len(part_words)} common words | raw-gated(affect)={n_pos} raw-excluded(neutral)={n_neg}",
          flush=True)

    rows = [run_seed(s, stories, part_words, raw_gate, a.n_hub, a.window, min_count, a.resample_frac, verbose=True)
            for s in seeds]

    # ── aggregate the (rho x sigma) frontier (worst-case + mean across seeds) ─────────────────────────────────────
    frontier = []
    for rho in RHO_GRID:
        for sigma in SIGMA_GRID:
            vals = [next(g["learned_ceiling"] for g in r["grid"] if g["rho"] == rho and g["sigma"] == sigma)
                    for r in rows]
            frontier.append({"rho": rho, "sigma": sigma, "learned_worst": float(min(vals)),
                             "learned_mean": float(np.mean(vals)), "clears_go_bar": bool(min(vals) >= CEIL_GO_BAR)})

    text_ceiling_worst = float(max(r["text_ceiling"] for r in rows))       # worst = HIGHEST (hardest to call it low)
    text_ceiling_mean = float(np.mean([r["text_ceiling"] for r in rows]))
    real_learned_worst = float(min(r["real_learned_ceiling"] for r in rows))
    real_learned_mean = float(np.mean([r["real_learned_ceiling"] for r in rows]))
    clean_learned_worst = float(min(r["clean_learned_ceiling"] for r in rows))
    clean_learned_mean = float(np.mean([r["clean_learned_ceiling"] for r in rows]))
    lesion_worst = float(max(r["lesion_ceiling"] for r in rows))           # worst = HIGHEST (must stay low)
    lesion_clean_worst = float(max(r["lesion_clean_ceiling"] for r in rows))
    shuffle_worst = float(max(r["shuffle_ceiling"] for r in rows))
    heldout_clean_worst = float(min(r["heldout_clean_ceiling"] for r in rows))
    heldout_clean_mean = float(np.mean([r["heldout_clean_ceiling"] for r in rows]))
    heldout_real_worst = float(min(r["heldout_real_ceiling"] for r in rows))
    textonly_clean_worst = float(min(r["textonly_transfer_clean_ceiling"] for r in rows))
    textonly_clean_mean = float(np.mean([r["textonly_transfer_clean_ceiling"] for r in rows]))
    textonly_real_mean = float(np.mean([r["textonly_transfer_real_ceiling"] for r in rows]))
    synth_ceiling_worst = float(min(r["synth_code_ceiling"] for r in rows))
    gated_real_worst = float(min(r["gated_real_ceiling"] for r in rows))
    gated_real_mean = float(np.mean([r["gated_real_ceiling"] for r in rows]))
    gated_real_fp5_worst = float(min(r["gated_real_fp5_ceiling"] for r in rows))
    gated_real_fp5_mean = float(np.mean([r["gated_real_fp5_ceiling"] for r in rows]))
    gated_clean_worst = float(min(r["gated_clean_ceiling"] for r in rows))
    gated_lesion_worst = float(max(r["gated_lesion_ceiling"] for r in rows))

    # relaxed-FP sensitivity aggregate (worst-case + mean across seeds)
    fp_sensitivity = []
    arm_points = [("grounded", rho, sigma) for (rho, sigma) in FP_POINTS] + \
                 [("lesion", RHO_REAL, SIGMA_REAL), ("shuffle", RHO_REAL, SIGMA_REAL)]
    for (arm, rho, sigma) in arm_points:
        row = {"arm": arm, "rho": rho, "sigma": sigma}
        for tol in FP_TOLS:
            vals = [next(x[f"fp{tol}"] for x in r["fp_sensitivity"]
                         if x["arm"] == arm and x["rho"] == rho and x["sigma"] == sigma) for r in rows]
            row[f"fp{tol}_worst"] = float(min(vals))
            row[f"fp{tol}_mean"] = float(np.mean(vals))
        fp_sensitivity.append(row)

    # smallest rho that clears the GO bar (worst-case) at each sigma -> the teacher's required coverage spec
    coverage_spec = {}
    for sigma in SIGMA_GRID:
        clearing = [f["rho"] for f in frontier if f["sigma"] == sigma and f["clears_go_bar"]]
        coverage_spec[str(sigma)] = (min(clearing) if clearing else None)

    # ── GO CRITERIA (pre-registered) ──────────────────────────────────────────────────────────────────────────────
    g1 = bool(real_learned_worst >= CEIL_GO_BAR)
    g2 = bool(lesion_worst <= text_ceiling_worst + ATTRIB_MARGIN and shuffle_worst <= text_ceiling_worst + ATTRIB_MARGIN)
    g2b = bool(heldout_clean_worst >= CEIL_GO_BAR)
    g3 = bool(synth_ceiling_worst >= CEIL_GO_BAR and text_ceiling_worst < TEXT_CEIL_MAX)
    go = bool(g1 and g2 and g2b and g3)

    v = Verdict("grounded-experience-stream Hebbian convergence: is the learned-code lift interpretable + attributable?")
    v.require("partition non-degenerate (affect + neutral both present)", measured=(n_pos > 0 and n_neg > 0),
              expect=True)
    v.require("the ceiling INSTRUMENT discriminates (synthetic clean code >=0.5, text code <0.2)",
              measured=(synth_ceiling_worst >= CEIL_GO_BAR and text_ceiling_worst < TEXT_CEIL_MAX), expect=True)
    v.control("grounding is LOAD-BEARING (clean/full grounded-taught code separates; the no-grounding LESION does not)",
              treatment=clean_learned_worst, control=lesion_clean_worst, min_separation=0.2)
    v.control("the code is TAUGHT not HANDED (held-out concepts separate; the shuffle-binding control does not)",
              treatment=heldout_clean_worst, control=shuffle_worst, min_separation=0.2)
    verdict_earned = v.decide(go=go, verbose=False)

    attributable_to("grounded-taught learned ceiling (vs the text-only ceiling)", real_learned_mean, text_ceiling_mean)
    attributable_to("grounded-taught learned ceiling (vs the no-grounding LESION)", real_learned_mean, lesion_worst)
    attributable_to("grounded-taught learned ceiling (vs the shuffle-binding control)", real_learned_mean, shuffle_worst)
    attributable_to("held-out generalization ceiling (vs the shuffle-binding control)", heldout_clean_mean, shuffle_worst)

    tag = f"{len(seeds)}-seed" if not a.smoke else "SMOKE(1-seed)"
    lift_line = (f"clean/full learned={clean_learned_worst:.3f} worst ({clean_learned_mean:.3f} mean) vs text "
                 f"{text_ceiling_worst:.3f}; lesion={lesion_worst:.3f}; shuffle={shuffle_worst:.3f}; held-out(clean)="
                 f"{heldout_clean_worst:.3f}; text-only-transfer(clean)={textonly_clean_worst:.3f} worst "
                 f"({textonly_clean_mean:.3f} mean); synth-instrument={synth_ceiling_worst:.3f}. THREE-FACTOR "
                 f"US-gated (neuromodulatory-gated plasticity) @realistic={gated_real_worst:.3f} worst "
                 f"(fp5%={gated_real_fp5_worst:.3f}), clean={gated_clean_worst:.3f}, lesion={gated_lesion_worst:.3f}")
    if go:
        verdict = (
            f"GO ({tag}) -- an EMERGENT Hebbian convergence over a grounded-experience STREAM TEACHES a concept code "
            f"that lifts the separability ceiling the TEXT code cannot. Grounded-recall learned ceiling reaches "
            f"{real_learned_worst:.3f} worst-case at a REALISTIC operating point (rho>={RHO_REAL}, sigma<={SIGMA_REAL}) "
            f"vs text {text_ceiling_worst:.3f}, CLEARING the {CEIL_GO_BAR} bar. It is GROUNDING, not the convergence "
            f"machinery: the no-body-state LESION reads {lesion_worst:.3f} and the shuffle-binding control "
            f"{shuffle_worst:.3f} (both at the text baseline). It is TAUGHT not HANDED: HELD-OUT concepts (never in the "
            f"Hebbian weight training) separate at {heldout_clean_worst:.3f} worst-case, so the convergence learned a "
            f"REUSABLE text+body-state -> code map, not a per-concept lookup. {lift_line}. NEXT: a fully-spiking "
            f"on-substrate convergence (reuse `_genfrontier` build_propagation_bridge, GPU-queued) + a real grounded "
            f"world delivering the per-concept US. Brain-based (rate-Hebbian synaptic convergence; the body-state is "
            f"the world/body boundary; the ceiling is the instrument); NO sim/ edit; NOT wired.")
    else:
        miss = [k for k, ok in (("G1_lift_clears_bar", g1), ("G2_load_bearing", g2),
                                ("G2b_generalizes", g2b), ("G3_instrument", g3)) if not ok]
        verdict = (
            f"PARTIAL/BOUNDARY ({tag}, build-informative) -- the EMERGENT Hebbian convergence over a grounded-"
            f"experience stream {'DOES' if clean_learned_worst >= CEIL_GO_BAR else 'does NOT clearly'} teach a "
            f"separable code at clean/full grounding ({clean_learned_worst:.3f} worst-case vs text "
            f"{text_ceiling_worst:.3f}), and grounding is load-bearing (lesion {lesion_worst:.3f}, shuffle "
            f"{shuffle_worst:.3f}). But the pre-registered strict bar failed: {lift_line}. FAILED: {miss}. See the "
            f"(rho,sigma) frontier + relaxed-FP sensitivity for what coverage/noise/FP-tolerance WOULD clear it. The "
            f"fixed _STRONG_MARGIN gate in affect_production_organ.py is UNCHANGED (this file wires nothing).")

    summary = {
        "probe": "affect_grounded_experience_stream_hebbian_derisk (an EMERGENT Hebbian convergence teaches the code)",
        "verdict": verdict, "GO": go,
        "G1_lift_clears_bar": g1, "G2_load_bearing": g2, "G2b_generalizes": g2b, "G3_instrument": g3,
        "text_ceiling_worst": text_ceiling_worst, "text_ceiling_mean": text_ceiling_mean,
        "grounded_taught_realistic_worst": real_learned_worst, "grounded_taught_realistic_mean": real_learned_mean,
        "grounded_taught_clean_worst": clean_learned_worst, "grounded_taught_clean_mean": clean_learned_mean,
        "lesion_control_worst": lesion_worst, "lesion_clean_control_worst": lesion_clean_worst,
        "shuffle_control_worst": shuffle_worst,
        "heldout_generalization_clean_worst": heldout_clean_worst, "heldout_generalization_clean_mean": heldout_clean_mean,
        "heldout_generalization_real_worst": heldout_real_worst,
        "textonly_transfer_clean_worst": textonly_clean_worst, "textonly_transfer_clean_mean": textonly_clean_mean,
        "textonly_transfer_real_mean": textonly_real_mean,
        "synthetic_instrument_ceiling_worst": synth_ceiling_worst,
        "gated_three_factor_realistic_worst": gated_real_worst, "gated_three_factor_realistic_mean": gated_real_mean,
        "gated_three_factor_realistic_fp5_worst": gated_real_fp5_worst,
        "gated_three_factor_realistic_fp5_mean": gated_real_fp5_mean,
        "gated_three_factor_clean_worst": gated_clean_worst, "gated_three_factor_lesion_worst": gated_lesion_worst,
        "ceiling_go_bar": CEIL_GO_BAR, "rho_realistic": RHO_REAL, "sigma_realistic": SIGMA_REAL,
        "attrib_margin": ATTRIB_MARGIN, "text_ceil_max": TEXT_CEIL_MAX, "heldout_frac": HELDOUT_FRAC,
        "rho_sigma_frontier": frontier,
        "relaxed_fp_sensitivity": fp_sensitivity,
        "required_coverage_spec_by_sigma": coverage_spec,
        "n_pos_raw_gated": n_pos, "n_neg_raw_excluded": n_neg, "n_partition_words": len(part_words),
        "per_seed": [{"seed": r["seed"], "code_dim": r["code_dim"], "text_ceiling": r["text_ceiling"],
                      "real_learned_ceiling": r["real_learned_ceiling"],
                      "clean_learned_ceiling": r["clean_learned_ceiling"], "lesion_ceiling": r["lesion_ceiling"],
                      "shuffle_ceiling": r["shuffle_ceiling"], "heldout_clean_ceiling": r["heldout_clean_ceiling"],
                      "textonly_transfer_clean_ceiling": r["textonly_transfer_clean_ceiling"],
                      "synth_code_ceiling": r["synth_code_ceiling"]} for r in rows],
        "preconditions": verdict_earned["preconditions"], "verdict_earned_status": verdict_earned["status"],
        "verdict_undefined_reasons": verdict_earned["undefined_reasons"],
        "config": {"seeds": seeds, "smoke": a.smoke, "max_stories": max_stories, "resample_frac": a.resample_frac,
                   "n_hub": a.n_hub, "window": a.window, "min_count": min_count, "m_assembly": M_ASSEMBLY,
                   "n_relay": N_RELAY, "epochs": EPOCHS, "eta": ETA, "k_wta": K_WTA,
                   "rho_grid": list(RHO_GRID), "sigma_grid": list(SIGMA_GRID), "backend": os.environ.get("SIM_BACKEND")},
        "mechanism": "TEXT concept code = build_cooccurrence -> codes_from_cooccurrence (register-confounded PPMI "
                     "stream code, reused). GROUNDED-EXPERIENCE STREAM = a per-concept body-state US (Warriner, the "
                     "world/body ORACLE STAND-IN) delivered into COMFORT/DISCOMFORT/AROUSAL interoceptive relay pools "
                     "(board #49/#84 structure) at coverage rho + afferent noise sigma. HEBBIAN CONVERGENCE = a "
                     "concept-assembly FF map (M rate neurons) learned by a competitive Oja-stabilized Hebbian rule "
                     "over [L2(text) | L2(intero)] (EMERGENT -- small-random init, k-WTA competition, never hand-set). "
                     "LEARNED CONCEPT CODE = the divisively-normalized assembly response, read by the VALIDATED "
                     "supervised ridge k-fold CEILING (reused verbatim from the embodied-US runner = the upper bound "
                     "on any readout). Controls: LESION (no body-state at learning), SHUFFLE (binding destroyed), "
                     "HELD-OUT (convergence trained on OTHER concepts -> generalization), TEXT-ONLY TRANSFER "
                     "(grounding absent at test), synthetic clean code (instrument).",
        "sources": [
            "2026-09-05-affect-gate-grounded-concept-code-lifts-ceiling-requirements-derisk-PARTIAL.md -- named this "
            "exact build (a grounded-experience stream via the interoceptive relay + a Hebbian convergence teaching "
            "the code); this runner replaces its RAW ORACLE FUSION with an EMERGENT synaptic convergence + a "
            "generalization test.",
            "2026-09-05-affect-gate-embodied-US-necessary-not-sufficient-concept-code-must-be-grounded-BOUNDARY.md -- "
            "proved the text code ceiling is ~0 (reproduced here as the like-for-like baseline).",
            "2026-08-19-embodied-affect-interoception-GO.md -- the board #49/#84 interoceptive relay (comfort/"
            "discomfort/arousal pools driven by a body-state current) whose structure the grounded stream reuses.",
            "2026-07-02-emerge34-perception-grounded-emergence-GO.md + _genfrontier_capstone_vision_to_concept -- the "
            "perception->concept convergence template (shared features -> Hebbian assembly selectivity -> held-out "
            "generalization) this convergence follows.",
            "Namburi, Tye et al. (2015, Nature) -- opponent valence populations bound to a real US, not lexical company.",
        ],
        "production_wiring": "NONE -- affect_production_organ.py and wkv_mouth_generator.py are byte-unchanged; this "
                             "is a standalone de-risk (reuse-by-import only). _STRONG_MARGIN unchanged.",
        "HONEST_RESIDUALS": "(1) the body-state US is a declared ORACLE STAND-IN for a grounded world that does not "
                            "exist for the TinyStories vocabulary (the SAME stand-in the embodied-US + grounded-code "
                            "runners used); this measures whether a convergence CAN teach a separable code GIVEN such "
                            "a stream + at what coverage/noise -- it does NOT deliver a real grounded world (the named "
                            "next build). (2) GO here means the grounded-TEACHER convergence arc is de-risked, NOT "
                            "that the gate is retired. (3) rate-Hebbian (numpy-CPU) convergence; a fully-spiking "
                            "on-substrate convergence (build_propagation_bridge, GPU-queued) is the named next rung. "
                            "(4) the ceiling is a linear supervised upper bound (the spiking opponent's mild "
                            "nonlinearity was measured NOT to help by the prior boundaries). (5) the 164-word closed "
                            "partition is inherited from the prior boundaries.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    undefined_if_empty("partition-words", len(part_words), len(part_words), len(part_words))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[grounded-exp-stream-hebbian] text={text_ceiling_worst:.3f} | learned@real={real_learned_worst:.3f} "
          f"clean/full={clean_learned_worst:.3f} | lesion={lesion_worst:.3f} shuffle={shuffle_worst:.3f} | "
          f"held-out={heldout_clean_worst:.3f} | text-only-xfer={textonly_clean_worst:.3f} | "
          f"GATED@real={gated_real_worst:.3f}(fp5%={gated_real_fp5_worst:.3f}) | "
          f"synth-instr={synth_ceiling_worst:.3f}", flush=True)
    print(f"[grounded-exp-stream-hebbian] required coverage spec by sigma: {coverage_spec}", flush=True)
    print(f"[grounded-exp-stream-hebbian] GO={go} (G1={g1} G2={g2} G2b={g2b} G3={g3})", flush=True)
    print(f"[grounded-exp-stream-hebbian] VERDICT: {verdict}", flush=True)
    print(f"[grounded-exp-stream-hebbian] wrote {a.out}  ({summary['elapsed_seconds']}s)\n" + "=" * 112, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
