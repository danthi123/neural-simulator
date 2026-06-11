"""STEP 3 (true cortex) -- Storkey local covariance-corrected Hopfield probe on the brain's correlated codes.

Per the explicit recommendation in research/findings/2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md:
  "direct-CA3 storage with Storkey/pseudo-inverse Hopfield weights on the raw correlated codes, which
  targets the correlated-pattern capacity directly and bypasses the DG-irreproducibility blocker."

SCIENTIFIC QUESTION:
  On the brain's RAW correlated denoise64 codes (between-code cosine ~0.81 -- highly correlated), does
  a COVARIANCE-CORRECTED LOCAL Hopfield weight rule (Storkey 1997) recover argmax-parity on clean/noised
  cues WITHOUT a god's-eye codebook lookup, and show pattern-completion value-add on partial cues?

FOUR MECHANISMS (labelled explicitly):
  - argmax:           god's-eye matched filter (the IDEALIZATION being replaced; REFERENCE only).
  - hopfield_vanilla: Hebbian outer product W = (1/N) sum xi xi^T; the documented collapse (REFERENCE).
  - hopfield_storkey: Storkey 1997 LOCAL covariance rule (THE brain-based candidate; the deliverable).
  - hopfield_pinv:    pseudo-inverse W = C(C^T C)^-1 C^T; linear-attractor CEILING (REFERENCE; host op).

STORKEY-CORRECTNESS SANITY CHECK (REQUIRED):
  On random near-orthogonal patterns, Storkey capacity (~0.39N) must EXCEED vanilla Hebbian (~0.14N).
  If Storkey doesn't beat vanilla on random patterns, the implementation is wrong. Checked in main().

TESTS (multi-seed 42/43/44):
  TEST 1 -- PARITY: noised cues (flip fraction p in {0.0, 0.1, 0.2, 0.3}), score recovery accuracy.
  TEST 2 -- COMPLETION: partial cues (keep fraction k in {0.5, 0.35, 0.25, 0.15}), score completion.
  CAPACITY SWEEP: V in {8, 16, 32, 64, 128, 256} stored patterns, run TEST 1 parity per mechanism.
  ANTI-CHEAT 1 (lesion): zero recurrent weights -> attractor must collapse to chance.
  ANTI-CHEAT 2 (shuffle): Storkey weights built from a shuffled/permuted codebook -> must drop to chance.

BRAIN-BASED BAR:
  The attractor SETTLE is the brain-based cleanup mechanism. Reading "which concept did it land on"
  by cosine to the codebook is a SCORING step (legitimate grading), not the cleanup computation.
  The KEY win: Storkey MATCHES argmax's cleanup WITHOUT needing the god's-eye codebook comparison
  during the settle -- patterns live distributively in the recurrent weights.

CPU-only; no sim/ edits; SIM_BACKEND=numpy; minutes per seed.

Run: python -m research.runners.cortex_storkey_ca3_cleanup_probe --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

CACHE = os.path.join(_REPO, "research", "findings", "raw",
                     "activity_level_integration_cache", "denoise64_seed%d.npz")


# ---------------------------------------------------------------------------
# Code loading (EXACT harness from cortex_dg_ca3_cleanup_probe.py and
# cortex_learned_cleanup_derisk.py -- same codes the prior probes used).
# ---------------------------------------------------------------------------
def load_real_codes(seed, proj_dim, rng):
    """Load the brain's REAL denoise64 concept codes -> signed real codes [V, D].
    Treatment matches the two prior probes: mean over obs samples per word, random-Gaussian
    project to proj_dim (preserves cosines), mean-center + unit-normalize.
    NO decorrelation -- these are the RAW correlated codes (the whole point).
    Returns (words, codes, raw_between_cos)."""
    d = np.load(CACHE % seed)
    ws = sorted(k[5:] for k in d.files if k.startswith("obs__"))
    raw = np.stack([d["obs__" + w].mean(axis=0) for w in ws]).astype(np.float64)   # [V, 3200]
    if proj_dim and proj_dim > 0:
        P = rng.standard_normal((raw.shape[1], proj_dim)) / np.sqrt(raw.shape[1])
        raw = raw @ P
    codes = raw - raw.mean(axis=1, keepdims=True)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    V = codes.shape[0]
    cs = [float(codes[i] @ codes[k]) for i in range(V) for k in range(i + 1, V)]
    between_cos = float(np.mean(cs)) if cs else 0.0
    return ws, codes, between_cos


# ---------------------------------------------------------------------------
# Hopfield mechanisms (all in bipolar {-1,+1} space; codes are real-valued,
# binarised before storing).
# ---------------------------------------------------------------------------

def _to_bipolar(codes):
    """Convert real-valued codes to bipolar {-1,+1} by thresholding at median per pattern."""
    out = np.zeros_like(codes)
    for i in range(codes.shape[0]):
        med = np.median(codes[i])
        out[i] = np.where(codes[i] >= med, 1.0, -1.0)
    return out


def build_vanilla(patterns):
    """Vanilla Hebbian: W = (1/N) sum_p xi_p xi_p^T, diag zeroed."""
    N = patterns.shape[1]
    W = (patterns.T @ patterns) / float(N)
    np.fill_diagonal(W, 0.0)
    return W


def build_storkey(patterns):
    """Storkey 1997 LOCAL covariance-corrected rule.

    For each pattern xi (in order), compute the local field h = W @ xi (using the
    CURRENT weight matrix before this pattern's update), then apply:
        DeltaW_ij = (1/N) * [xi_i * xi_j - xi_i * h_j - h_i * xi_j]

    This is local: each delta W_ij uses only xi_i, xi_j (the pre/post activities) and
    h_i, h_j (the local field contributions already in the weights), not the global
    correlation matrix. W is kept symmetric and zero-diagonal throughout.

    Reference: Storkey 1997 'Increasing the capacity of a Hopfield network without
    sacrificing functionality', ICANN 1997."""
    N = patterns.shape[1]
    W = np.zeros((N, N), dtype=np.float64)
    for xi in patterns:
        h = W @ xi          # local field [N], using CURRENT W
        xi_col = xi.reshape(-1, 1)
        xi_row = xi.reshape(1, -1)
        h_col = h.reshape(-1, 1)
        h_row = h.reshape(1, -1)
        delta = (xi_col * xi_row - xi_col * h_row - h_col * xi_row) / float(N)
        W += delta
        np.fill_diagonal(W, 0.0)
    return W


def build_pinv(patterns):
    """Pseudo-inverse rule: W = C (C^T C)^-1 C^T, diag zeroed.
    This is the linear-attractor CAPACITY CEILING. The matrix inverse is a HOST computation --
    NOT brain-based. Reference / ceiling only."""
    C = patterns.T   # [N, P]
    W = C @ np.linalg.pinv(C)   # [N, N]
    np.fill_diagonal(W, 0.0)
    return W


def hopfield_settle(W, cue, iters=50):
    """Synchronous Hopfield settle. Binarise cue, iterate sign(W @ s) until fixed point or max iters.
    Returns the settled bipolar state."""
    s = np.sign(cue.astype(np.float64))
    s[s == 0] = 1.0
    for _ in range(iters):
        s_new = np.sign(W @ s)
        # handle ties (s_new == 0) by keeping previous state
        ties = s_new == 0
        s_new[ties] = s[ties]
        if np.all(s_new == s):
            break
        s = s_new
    return s


def argmax_cleanup(cue, codes):
    """God's-eye matched filter: nearest stored code by cosine over the codebook.
    REFERENCE / idealization being replaced. Operates on the REAL (non-binarised) codes and cue."""
    sims = codes @ cue / (np.linalg.norm(codes, axis=1) * (np.linalg.norm(cue) + 1e-12) + 1e-12)
    return int(np.argmax(sims))


def hopfield_cleanup(W, cue, patterns_bipolar):
    """Attractor-based cleanup: settle the binarised cue under W, then score by cosine
    to each stored bipolar pattern to identify which attractor the settle landed in.
    The SETTLE is the brain-based step; the cosine-scoring is the legitimate grading step."""
    cue_bip = np.sign(cue.astype(np.float64))
    cue_bip[cue_bip == 0] = 1.0
    settled = hopfield_settle(W, cue_bip)
    # score settled state against each stored bipolar pattern
    sims = patterns_bipolar @ settled / (
        np.linalg.norm(patterns_bipolar, axis=1) * (np.linalg.norm(settled) + 1e-12) + 1e-12)
    return int(np.argmax(sims))


# ---------------------------------------------------------------------------
# Cue generators.
# ---------------------------------------------------------------------------

def noisy_cue(code, rng, flip_frac):
    """Noised cue: randomly flip flip_frac fraction of binarised bits.
    Operates on the REAL code; binarise first then flip."""
    D = code.shape[0]
    bip = np.sign(code.copy())
    bip[bip == 0] = 1.0
    n_flip = max(0, int(round(flip_frac * D)))
    if n_flip > 0:
        idx = rng.choice(D, size=n_flip, replace=False)
        bip[idx] *= -1
    return bip


def partial_cue(code, rng, keep_frac):
    """Partial cue: keep keep_frac fraction of bits, set rest to 0 (unknown).
    Argmax-on-partial sees only the kept dims; an attractor can complete the rest."""
    D = code.shape[0]
    bip = np.sign(code.copy())
    bip[bip == 0] = 1.0
    n_keep = max(1, int(round(keep_frac * D)))
    out = np.zeros_like(bip)
    keep = rng.choice(D, size=n_keep, replace=False)
    out[keep] = bip[keep]
    return out


# ---------------------------------------------------------------------------
# Storkey sanity check (validates the implementation on random near-orthogonal patterns).
# ---------------------------------------------------------------------------

def storkey_sanity_check():
    """Verify: on random near-orthogonal patterns, Storkey capacity (~0.39N) > vanilla (~0.14N).
    Returns (passed: bool, details: dict)."""
    rng = np.random.default_rng(12345)
    N = 200
    n_trials = 60
    flip_frac = 0.05   # 5% bit flip for recall test

    results = {}
    for P_frac, label in [(0.14, "at_0.14N"), (0.25, "at_0.25N"), (0.35, "at_0.35N")]:
        P = max(1, int(P_frac * N))
        patterns = np.array([np.sign(rng.standard_normal(N)) for _ in range(P)])
        W_van = build_vanilla(patterns)
        W_sto = build_storkey(patterns)
        van_ok = sto_ok = 0
        rng2 = np.random.default_rng(999)
        for _ in range(n_trials):
            i = int(rng2.integers(P))
            cue = noisy_cue(patterns[i], rng2, flip_frac)
            settled_van = hopfield_settle(W_van, cue)
            settled_sto = hopfield_settle(W_sto, cue)
            van_ok += int(np.mean(settled_van == patterns[i]) > 0.95)
            sto_ok += int(np.mean(settled_sto == patterns[i]) > 0.95)
        results[label] = {"P": P, "vanilla_acc": van_ok / n_trials, "storkey_acc": sto_ok / n_trials}

    # At 0.25N: Storkey should work, vanilla should not
    passed = (results["at_0.25N"]["storkey_acc"] > results["at_0.25N"]["vanilla_acc"] + 0.3
              and results["at_0.25N"]["storkey_acc"] > 0.7)
    return passed, results


# ---------------------------------------------------------------------------
# TEST 1 -- parity on noised cues.
# ---------------------------------------------------------------------------

def run_test1_parity(codes, words, rng, n_trials, flip_fracs):
    """For each stored concept, present noised cues, settle each mechanism, score recovery.
    Recovery = cosine-argmax over codebook of the settled state == true concept."""
    V = codes.shape[0]
    bip = _to_bipolar(codes)

    W_van = build_vanilla(bip)
    W_sto = build_storkey(bip)
    W_pinv = build_pinv(bip)

    chance = 1.0 / V
    results = {}
    for flip_frac in flip_fracs:
        n_arg = n_van = n_sto = n_pinv = 0
        for _ in range(n_trials):
            i = int(rng.integers(V))
            cue = noisy_cue(codes[i], rng, flip_frac)
            n_arg += int(argmax_cleanup(cue.astype(float), codes) == i)
            n_van += int(hopfield_cleanup(W_van, cue, bip) == i)
            n_sto += int(hopfield_cleanup(W_sto, cue, bip) == i)
            n_pinv += int(hopfield_cleanup(W_pinv, cue, bip) == i)
        results[float(flip_frac)] = {
            "argmax": n_arg / n_trials,
            "vanilla": n_van / n_trials,
            "storkey": n_sto / n_trials,
            "pinv": n_pinv / n_trials,
            "chance": chance,
            "n": n_trials,
        }
    return results


# ---------------------------------------------------------------------------
# TEST 2 -- pattern completion on partial cues.
# ---------------------------------------------------------------------------

def run_test2_completion(codes, words, rng, n_trials, keep_fracs):
    """Partial cue (unknown bits zeroed), settle, score completion accuracy.
    Compare hopfield_storkey vs argmax-on-partial."""
    V = codes.shape[0]
    bip = _to_bipolar(codes)

    W_sto = build_storkey(bip)
    W_pinv = build_pinv(bip)

    chance = 1.0 / V
    results = {}
    for keep_frac in keep_fracs:
        n_arg = n_sto = n_pinv = n_van = 0
        W_van = build_vanilla(bip)
        for _ in range(n_trials):
            i = int(rng.integers(V))
            cue = partial_cue(codes[i], rng, keep_frac)
            # argmax on partial: use the partial real cue directly
            cue_real = cue.astype(float)
            n_arg += int(argmax_cleanup(cue_real, codes) == i)
            n_van += int(hopfield_cleanup(W_van, cue, bip) == i)
            n_sto += int(hopfield_cleanup(W_sto, cue, bip) == i)
            n_pinv += int(hopfield_cleanup(W_pinv, cue, bip) == i)
        results[float(keep_frac)] = {
            "argmax_on_partial": n_arg / n_trials,
            "vanilla": n_van / n_trials,
            "storkey": n_sto / n_trials,
            "pinv": n_pinv / n_trials,
            "chance": chance,
            "n": n_trials,
            "storkey_edge": (n_sto - n_arg) / n_trials,  # + = Storkey beats argmax
        }
    return results


# ---------------------------------------------------------------------------
# CAPACITY SWEEP -- accuracy vs V (number of stored patterns).
# ---------------------------------------------------------------------------

def run_capacity_sweep(all_codes, all_words, rng, n_trials, capacities, flip_frac=0.1):
    """Store V in `capacities` of the all_codes codes (first V), run TEST 1 parity at flip_frac."""
    V_total = all_codes.shape[0]
    results = {}
    for V in capacities:
        if V > V_total:
            continue
        codes = all_codes[:V]
        bip = _to_bipolar(codes)
        W_van = build_vanilla(bip)
        W_sto = build_storkey(bip)
        W_pinv = build_pinv(bip)
        chance = 1.0 / V
        n_arg = n_van = n_sto = n_pinv = 0
        for _ in range(n_trials):
            i = int(rng.integers(V))
            cue = noisy_cue(codes[i], rng, flip_frac)
            n_arg += int(argmax_cleanup(cue.astype(float), codes) == i)
            n_van += int(hopfield_cleanup(W_van, cue, bip) == i)
            n_sto += int(hopfield_cleanup(W_sto, cue, bip) == i)
            n_pinv += int(hopfield_cleanup(W_pinv, cue, bip) == i)
        results[int(V)] = {
            "argmax": n_arg / n_trials,
            "vanilla": n_van / n_trials,
            "storkey": n_sto / n_trials,
            "pinv": n_pinv / n_trials,
            "chance": chance,
            "n": n_trials,
            "flip_frac": flip_frac,
        }
    return results


# ---------------------------------------------------------------------------
# ANTI-CHEAT 1 (lesion) -- zero the recurrent matrix -> cleanup must collapse.
# ---------------------------------------------------------------------------

def run_anticheat_lesion(codes, words, rng, n_trials, flip_frac=0.1):
    """Build Storkey weights, zero them (lesion), settle, score -> must collapse to chance."""
    V = codes.shape[0]
    bip = _to_bipolar(codes)
    W_sto = build_storkey(bip)
    W_lesion = np.zeros_like(W_sto)  # zeroed weights
    n_intact = n_lesion = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue = noisy_cue(codes[i], rng, flip_frac)
        n_intact += int(hopfield_cleanup(W_sto, cue, bip) == i)
        n_lesion += int(hopfield_cleanup(W_lesion, cue, bip) == i)
    chance = 1.0 / V
    return {
        "intact_acc": n_intact / n_trials,
        "lesion_acc": n_lesion / n_trials,
        "chance": chance,
        "collapses": bool(n_lesion / n_trials < n_intact / n_trials - 0.2),
        "n": n_trials,
    }


# ---------------------------------------------------------------------------
# ANTI-CHEAT 2 (shuffle) -- Storkey weights from SHUFFLED codebook -> must drop to chance.
# ---------------------------------------------------------------------------

def run_anticheat_shuffle(codes, words, rng, n_trials, flip_frac=0.1):
    """Build Storkey weights from a PERMUTED codebook (different codes), test recall
    against the TRUE codes -> must drop to chance."""
    V = codes.shape[0]
    bip = _to_bipolar(codes)
    # Shuffled codebook: permute the rows
    perm = rng.permutation(V)
    bip_shuffled = bip[perm]
    W_shuffled = build_storkey(bip_shuffled)
    W_true = build_storkey(bip)
    n_true = n_shuf = 0
    for _ in range(n_trials):
        i = int(rng.integers(V))
        cue = noisy_cue(codes[i], rng, flip_frac)
        n_true += int(hopfield_cleanup(W_true, cue, bip) == i)
        n_shuf += int(hopfield_cleanup(W_shuffled, cue, bip) == i)
    chance = 1.0 / V
    return {
        "true_codebook_acc": n_true / n_trials,
        "shuffled_codebook_acc": n_shuf / n_trials,
        "chance": chance,
        "drops_to_chance": bool(n_shuf / n_trials < chance + 0.15),
        "n": n_trials,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run_seed(seed, args, sanity_passed):
    print("\n" + "=" * 72, flush=True)
    print("=== Storkey CA3 cleanup probe (seed %d) ===" % seed, flush=True)
    print("=" * 72, flush=True)

    if not os.path.exists(CACHE % seed):
        print("[probe] MISSING denoise64 cache %s" % (CACHE % seed), flush=True)
        return None

    rng = np.random.default_rng(seed)
    words, codes, between_cos = load_real_codes(seed, args.proj_dim, rng)
    V = len(words)
    D = codes.shape[1]
    print("[codes] seed=%d  V=%d  D=%d  between-code cosine (RAW correlated) = %.3f"
          % (seed, V, D, between_cos), flush=True)

    # Build all four weight matrices once (on all V stored patterns)
    bip = _to_bipolar(codes)
    W_van = build_vanilla(bip)
    W_sto = build_storkey(bip)
    W_pinv = build_pinv(bip)

    # TEST 1 -- parity
    print("\n--- TEST 1: parity on noised cues ---", flush=True)
    flip_fracs = [0.0, 0.1, 0.2, 0.3]
    t1 = run_test1_parity(codes, words, np.random.default_rng(seed + 11), args.n_trials, flip_fracs)
    for p, r in sorted(t1.items()):
        print("  flip=%.1f:  argmax=%.3f  vanilla=%.3f  storkey=%.3f  pinv=%.3f  (chance=%.3f)"
              % (p, r["argmax"], r["vanilla"], r["storkey"], r["pinv"], r["chance"]), flush=True)

    # TEST 2 -- completion
    print("\n--- TEST 2: completion on partial cues ---", flush=True)
    keep_fracs = [0.5, 0.35, 0.25, 0.15]
    t2 = run_test2_completion(codes, words, np.random.default_rng(seed + 22), args.n_trials, keep_fracs)
    for k, r in sorted(t2.items(), reverse=True):
        edge_sym = "+" if r["storkey_edge"] > 0 else ""
        print("  keep=%.2f:  argmax_partial=%.3f  vanilla=%.3f  storkey=%.3f  pinv=%.3f  edge=%s%.3f  (chance=%.3f)"
              % (k, r["argmax_on_partial"], r["vanilla"], r["storkey"], r["pinv"],
                 edge_sym, r["storkey_edge"], r["chance"]), flush=True)

    # CAPACITY SWEEP
    print("\n--- CAPACITY SWEEP (accuracy vs V at flip=0.1) ---", flush=True)
    capacities = [8, 16, 32, 64, 128, 256]
    # We need more codes for larger V -- use the same codes repeatedly if V > len(words)
    # Actually denoise64 only has 16 codes; for larger V we must use the same 16 (or fewer)
    # For the sweep we can only test up to V=len(words)=16
    cap_results = run_capacity_sweep(codes, words, np.random.default_rng(seed + 33),
                                     args.n_trials, [c for c in capacities if c <= V],
                                     flip_frac=0.1)
    for v_cap, r in sorted(cap_results.items()):
        print("  V=%3d:  argmax=%.3f  vanilla=%.3f  storkey=%.3f  pinv=%.3f  (chance=%.3f)"
              % (v_cap, r["argmax"], r["vanilla"], r["storkey"], r["pinv"], r["chance"]), flush=True)
    print("  [note: denoise64 has only 16 unique codes; V > 16 not tested on this seed]", flush=True)

    # ANTI-CHEAT 1
    print("\n--- ANTI-CHEAT 1: lesion (zero weights) -> cleanup collapses ---", flush=True)
    ac1 = run_anticheat_lesion(codes, words, np.random.default_rng(seed + 44),
                               args.n_trials, flip_frac=0.1)
    print("  intact_acc=%.3f  lesion_acc=%.3f  (chance=%.3f)  collapses=%s"
          % (ac1["intact_acc"], ac1["lesion_acc"], ac1["chance"], ac1["collapses"]), flush=True)

    # ANTI-CHEAT 2
    print("\n--- ANTI-CHEAT 2: shuffled codebook -> drops to chance ---", flush=True)
    ac2 = run_anticheat_shuffle(codes, words, np.random.default_rng(seed + 55),
                                args.n_trials, flip_frac=0.1)
    print("  true_codebook_acc=%.3f  shuffled_acc=%.3f  (chance=%.3f)  drops_to_chance=%s"
          % (ac2["true_codebook_acc"], ac2["shuffled_codebook_acc"], ac2["chance"], ac2["drops_to_chance"]),
          flush=True)

    # GATES
    # Parity gate: Storkey ~= argmax at flip=0.1 (within tol=0.10)
    p01 = t1.get(0.1, t1.get(0.0, {}))
    storkey_parity_ok = p01.get("storkey", 0.0) >= p01.get("argmax", 1.0) - args.tol
    # Parity gate at flip=0.2
    p02 = t1.get(0.2, {})
    storkey_parity_02 = p02.get("storkey", 0.0) >= p02.get("argmax", 1.0) - args.tol

    # Completion edge gate: Storkey > argmax_partial at any keep_frac
    completion_edge = any(r["storkey_edge"] > 0.05 for r in t2.values())

    # Lesion gate
    lesion_collapses = ac1["collapses"]

    # Shuffle gate
    shuffle_collapses = ac2["drops_to_chance"]

    # Capacity wall analysis: is it a CAPACITY wall (pinv fails too) or LOCALITY wall (pinv succeeds)?
    pinv_parity = p01.get("pinv", 0.0) >= p01.get("argmax", 1.0) - args.tol
    storkey_collapses = p01.get("storkey", 0.0) < 0.5
    vanilla_collapses = p01.get("vanilla", 0.0) < 0.3

    if storkey_parity_ok and lesion_collapses and shuffle_collapses:
        verdict = "GO"
        wall_diagnosis = "none -- Storkey SUCCEEDS on correlated codes"
    elif storkey_parity_ok and (lesion_collapses or shuffle_collapses):
        verdict = "PARTIAL"
        wall_diagnosis = "anti-cheats partially passed"
    elif not storkey_parity_ok and not pinv_parity:
        verdict = "NEGATIVE"
        wall_diagnosis = "CAPACITY WALL -- even pinv (the linear ceiling) fails; correlated codes exceed linear capacity"
    elif not storkey_parity_ok and pinv_parity:
        verdict = "NEGATIVE"
        wall_diagnosis = "LOCALITY WALL -- pinv (host ceiling) succeeds but Storkey (local rule) fails"
    elif storkey_collapses and not vanilla_collapses:
        verdict = "NEGATIVE"
        wall_diagnosis = "STORKEY-WORSE-THAN-VANILLA -- unusual; implementation check needed"
    else:
        verdict = "BOUNDARY"
        wall_diagnosis = "partial signal -- see per-flip results"

    print("\n  GATES:", flush=True)
    print("    storkey_parity(flip=0.1)=%s  storkey_parity(flip=0.2)=%s"
          % (storkey_parity_ok, storkey_parity_02), flush=True)
    print("    completion_edge_anywhere=%s  lesion_collapses=%s  shuffle_collapses=%s"
          % (completion_edge, lesion_collapses, shuffle_collapses), flush=True)
    print("    pinv_parity=%s  -> wall_diagnosis: %s" % (pinv_parity, wall_diagnosis), flush=True)
    print("  === SEED %d VERDICT: %s ===" % (seed, verdict), flush=True)

    return {
        "seed": seed, "V": V, "D": D, "between_code_cos_raw": between_cos,
        "test1_parity": {str(k): v for k, v in t1.items()},
        "test2_completion": {str(k): v for k, v in t2.items()},
        "capacity_sweep": {str(k): v for k, v in cap_results.items()},
        "anticheat_lesion": ac1,
        "anticheat_shuffle": ac2,
        "gates": {
            "storkey_parity_flip01": bool(storkey_parity_ok),
            "storkey_parity_flip02": bool(storkey_parity_02),
            "completion_edge": bool(completion_edge),
            "lesion_collapses": bool(lesion_collapses),
            "shuffle_collapses": bool(shuffle_collapses),
            "pinv_parity": bool(pinv_parity),
        },
        "wall_diagnosis": wall_diagnosis,
        "verdict": verdict,
        "args": {"proj_dim": args.proj_dim, "n_trials": args.n_trials, "tol": args.tol},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated list, e.g. 42,43,44 (overrides --seed)")
    ap.add_argument("--proj-dim", type=int, default=512,
                    help="random-Gaussian projection dim for real codes (preserves cosines)")
    ap.add_argument("--n-trials", type=int, default=200,
                    help="number of trials per test (higher = more reliable)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="parity tolerance: Storkey acc >= argmax acc - tol counts as parity")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "findings", "raw",
                                         "_cortex_storkey_ca3_cleanup_seed%(seed)s.json"))
    ap.add_argument("--out-multiseed", type=str,
                    default=os.path.join(_HERE, "..", "findings", "raw",
                                         "_cortex_storkey_ca3_cleanup_multiseed.json"))
    args = ap.parse_args()

    # --- STORKEY SANITY CHECK (required; run before any seed) ---
    print("=" * 72, flush=True)
    print("=== STORKEY SANITY CHECK (random near-orthogonal patterns) ===", flush=True)
    sanity_passed, sanity_details = storkey_sanity_check()
    for label, r in sorted(sanity_details.items()):
        print("  %s (P=%d): vanilla=%.3f  storkey=%.3f"
              % (label, r["P"], r["vanilla_acc"], r["storkey_acc"]), flush=True)
    if sanity_passed:
        print("  SANITY CHECK PASSED -- Storkey beats vanilla at 0.25N (%.3f vs %.3f)"
              % (sanity_details["at_0.25N"]["storkey_acc"],
                 sanity_details["at_0.25N"]["vanilla_acc"]), flush=True)
    else:
        print("  SANITY CHECK FAILED -- implementation error; halting", flush=True)
        return 1
    print("=" * 72, flush=True)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    results = []
    for s in seeds:
        t0 = time.time()
        r = run_seed(s, args, sanity_passed)
        if r is not None:
            r["wall_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            # save per-seed JSON
            per_seed_out = args.out % {"seed": str(s)}
            os.makedirs(os.path.dirname(os.path.normpath(per_seed_out)), exist_ok=True)
            json.dump(r, open(os.path.normpath(per_seed_out), "w", encoding="utf-8"), indent=2)
            print("  wrote %s" % os.path.normpath(per_seed_out), flush=True)

    # Multi-seed roll-up
    if results:
        def mean(key_path):
            vals = []
            for r in results:
                d = r
                for k in key_path:
                    d = d[k]
                vals.append(float(d))
            return float(np.mean(vals))

        verdicts = [r["verdict"] for r in results]
        n_go = sum(v == "GO" for v in verdicts)
        n_partial = sum(v == "PARTIAL" for v in verdicts)
        n_neg = sum(v == "NEGATIVE" for v in verdicts)
        overall = ("GO" if n_go == len(results) else
                   "PARTIAL" if (n_go + n_partial) >= 1 else "NEGATIVE")

        print("\n" + "#" * 72, flush=True)
        print("MULTI-SEED ROLL-UP (%d seeds: %s)" % (len(results), ",".join(str(r["seed"]) for r in results)),
              flush=True)
        print("PARITY (flip=0.1): argmax=%.3f  vanilla=%.3f  storkey=%.3f  pinv=%.3f"
              % (mean(["test1_parity", "0.1", "argmax"]),
                 mean(["test1_parity", "0.1", "vanilla"]),
                 mean(["test1_parity", "0.1", "storkey"]),
                 mean(["test1_parity", "0.1", "pinv"])), flush=True)
        print("PARITY (flip=0.2): argmax=%.3f  vanilla=%.3f  storkey=%.3f  pinv=%.3f"
              % (mean(["test1_parity", "0.2", "argmax"]),
                 mean(["test1_parity", "0.2", "vanilla"]),
                 mean(["test1_parity", "0.2", "storkey"]),
                 mean(["test1_parity", "0.2", "pinv"])), flush=True)
        print("PARITY (flip=0.3): argmax=%.3f  vanilla=%.3f  storkey=%.3f  pinv=%.3f"
              % (mean(["test1_parity", "0.3", "argmax"]),
                 mean(["test1_parity", "0.3", "vanilla"]),
                 mean(["test1_parity", "0.3", "storkey"]),
                 mean(["test1_parity", "0.3", "pinv"])), flush=True)
        print("ANTICHEAT lesion (intact vs lesioned): %.3f vs %.3f  collapses=%s"
              % (mean(["anticheat_lesion", "intact_acc"]),
                 mean(["anticheat_lesion", "lesion_acc"]),
                 all(r["anticheat_lesion"]["collapses"] for r in results)), flush=True)
        print("ANTICHEAT shuffle (true vs shuffled): %.3f vs %.3f  drops=%s"
              % (mean(["anticheat_shuffle", "true_codebook_acc"]),
                 mean(["anticheat_shuffle", "shuffled_codebook_acc"]),
                 all(r["anticheat_shuffle"]["drops_to_chance"] for r in results)), flush=True)
        print("Per-seed verdicts: %s  =>  OVERALL: %s" % (verdicts, overall), flush=True)
        print("Wall diagnoses: %s" % [r["wall_diagnosis"] for r in results], flush=True)

        multiseed_out = {
            "probe": "cortex_storkey_ca3_cleanup_probe",
            "seeds": [r["seed"] for r in results],
            "overall_verdict": overall,
            "sanity_check_passed": sanity_passed,
            "sanity_details": sanity_details,
            "per_seed": results,
            "rollup": {
                "parity_flip01_argmax": mean(["test1_parity", "0.1", "argmax"]),
                "parity_flip01_vanilla": mean(["test1_parity", "0.1", "vanilla"]),
                "parity_flip01_storkey": mean(["test1_parity", "0.1", "storkey"]),
                "parity_flip01_pinv": mean(["test1_parity", "0.1", "pinv"]),
                "parity_flip02_argmax": mean(["test1_parity", "0.2", "argmax"]),
                "parity_flip02_vanilla": mean(["test1_parity", "0.2", "vanilla"]),
                "parity_flip02_storkey": mean(["test1_parity", "0.2", "storkey"]),
                "parity_flip02_pinv": mean(["test1_parity", "0.2", "pinv"]),
                "parity_flip03_argmax": mean(["test1_parity", "0.3", "argmax"]),
                "parity_flip03_vanilla": mean(["test1_parity", "0.3", "vanilla"]),
                "parity_flip03_storkey": mean(["test1_parity", "0.3", "storkey"]),
                "parity_flip03_pinv": mean(["test1_parity", "0.3", "pinv"]),
                "anticheat_lesion_intact": mean(["anticheat_lesion", "intact_acc"]),
                "anticheat_lesion_lesioned": mean(["anticheat_lesion", "lesion_acc"]),
                "anticheat_shuffle_true": mean(["anticheat_shuffle", "true_codebook_acc"]),
                "anticheat_shuffle_shuffled": mean(["anticheat_shuffle", "shuffled_codebook_acc"]),
            },
        }
        ms_path = os.path.normpath(args.out_multiseed)
        os.makedirs(os.path.dirname(ms_path), exist_ok=True)
        json.dump(multiseed_out, open(ms_path, "w", encoding="utf-8"), indent=2)
        print("wrote %s" % ms_path, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
