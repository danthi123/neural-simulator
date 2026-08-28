"""DG-shard ESCALATION diagnostic (board #66/#192, real-time knowledge-recall wall).

WHY. The 2026-08-28 sparse-index PORT finding (`2026-08-28-shard-composer-dg-sparse-index-port-modest-latency-
reduction.md`) measured a real but MODEST ~25% warm-query latency reduction (33.21s -> 24.92s median) at
347,695-word vocab, and named -- without measuring -- two live candidate explanations for why the reduction is
modest, not dramatic: (1) "the DG shard match is frequently non-decisive on real Wikidata entity codes ... so many
role-cleanups still ESCALATE to the full-codebook scan", or (2) this session's memory pressure inflating every
number including the untouched resonate step. This runner measures (1) DIRECTLY: the escalation FRACTION on the
real bundle's own codebook, and separates it from a third, previously-unconsidered candidate (3): the shard
composer's PRODUCTION operating point is D=128 (`wikidata_500k_fast/manifest.json`), not D=256 -- the value the
original de-risk (`_sparse_indexed_retrieval_derisk.py`) validated GO under and its own docstring labels "the
production console op-point". If escalation is high for BOTH real Wikidata codes AND synthetic uniform codes at
the matched D=128/V=347,695 operating point, the wall is (3) an operating-point issue, not (1) something peculiar
to real-entity geometry -- material because peculiar-to-real-geometry needs a different fix (a hash matched to
real code statistics) than an operating-point wall (which the existing mechanism's own knobs, G/conf_floor, may
fix directly).

METHOD (efficient -- does NOT resonate/store the full 748,956-fact bundle; ~O(minutes) not ~O(hours)).

  Step 1 (REAL, small, PRODUCTION path -- calibrates the noise the synthetic sweep must use). Build ONE
  `RFPhasorComposer` over the REAL, FULL bundle codebook (same seed as the bundle manifest -> byte-identical
  codes to production, `sharded_phasor_store.py`'s own documented "the codebook regenerates byte-identically
  from seed+vocab" property). Store `--n-store` REAL facts sampled from the bundle's `facts.json` through the
  ACTUAL production `RFPhasorComposer.store()` (a real RF resonate bind of agent+action+patient[+polarity], with
  the genuine intra-fact crosstalk multi-role superposition introduces). For each stored fact, `_unbind_phases`
  the cue role and compare the RECOVERED phase to the TRUE stored concept phase -- this calibrates a REAL
  resonate-recovery noise sigma (measured, not assumed; the original de-risk's sigma=0.30 was never measured
  against this composer's actual RF dynamics).

  Step 2 (escalation sweep, REAL codes). Using the calibrated sigma, sample `--n-query` REAL words from the
  bundle vocab, build a noisy query phase (true concept phase + calibrated per-component Gaussian noise), and
  call the PRODUCTION `RFPhasorComposer._dg_shard_select` (imported by using the composer instance directly --
  NOT reimplemented) against the DG index already built (Step 1's composer, same codebook). Tally escalate vs.
  decisive, and the peak-score margin (peak/D - conf_floor) distribution. A subsample (`--n-parity`) ALSO gets
  the ground-truth full-codebook argmax (one batched BLAS matmul, reusing `_full_host_select`'s own op, called
  directly) to verify the shard's decisive answer, when it returns one, actually AGREES with the full scan
  (parity) -- not merely that escalation is rare.

  Step 3 (comparison, matched D/V, SYNTHETIC codes). The identical sweep over a SYNTHETIC uniform-random FHRR
  codebook at the IDENTICAL V and D (the original de-risk's own `gen_fhrr_phases`/`DGSparseIndex`, reused by
  import, not reimplemented), same calibrated sigma. There is no RFPhasorComposer for the synthetic path (it is
  a bare codebook, not a fact store), so the scoring/decision arithmetic is reproduced as a literal one-line
  copy of `RFPhasorComposer._dg_shard_select`'s own formula (`score = cos(2pi(rec-code)).sum()`, escalate if
  peak < conf_floor*D) -- documented here, not hidden, as the one place this script duplicates rather than calls
  production code, because the synthetic path has no composer instance to call it on.

Interpreting the two escalation numbers: REAL >> SYNTHETIC at matched D/V => explanation (1) (real-geometry-
specific). REAL ~= SYNTHETIC, both high => explanation (3) (operating-point: the de-risk's own GO was never
established past V=200,000 nor below D=256, and this project runs the shard composer at D=128/V=347,695).

LEVER SUPPORT (--lever {none, more_probes, lower_conf_floor}). `more_probes`: doubles `--G` (multi-probe OR-
amplification, the mechanism's own documented noise-robustness knob) -- requires a full index rebuild.
`lower_conf_floor`: lowers `--conf-floor` at DECIDE time only (no rebuild) -- cheaper, but must be checked for a
parity regression (a shard peak that clears a lowered floor but is NOT the true global argmax), which this
script does via the Step-2 parity subsample.

Memory: builds ONE (V,D) codebook (~V*D*8 bytes, ~340 MB at V=347,695/D=128) + ONE DGSparseIndex bucket set at a
time (real and synthetic phases are run SEQUENTIALLY with an explicit `del`+`gc.collect()` between them, so peak
RSS is bounded to one index's worth, not both). Prints RSS at each checkpoint (`resource.getrusage`).

Run:
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._dg_shard_escalation_diagnostic \\
      --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_500k_fast --seed 42 \\
      --n-store 300 --n-query 3000 --n-parity 500 \\
      --json research/findings/raw/dg_shard_escalation/diag_seed42.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time

import numpy as np


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _log(msg: str, t0: float):
    print(f"[{time.time()-t0:7.1f}s | RSS {_rss_mb():7.1f}MB] {msg}", flush=True)


def load_bundle(bundle_dir: str):
    with open(os.path.join(bundle_dir, "manifest.json")) as f:
        mani = json.load(f)
    with open(os.path.join(bundle_dir, "facts.json")) as f:
        raw = json.load(f)
    facts = [r["fact"] if isinstance(r, dict) and "fact" in r else r for r in raw]
    return mani, facts


def calibrate_noise(composer, facts, n_store: int, seed: int, cue_role: str, t0: float):
    """Store n_store REAL facts through the production store() path; unbind cue_role from each and compare to
    the TRUE stored concept phase. Returns (sigma_fractional_cycle, n_measured, raw_abs_errors)."""
    rng = np.random.default_rng(seed * 7919 + 1)
    # only facts whose roles are all plain strings (skip Clause/attributed patients -- keeps this a clean,
    # single-role-per-slot calibration; the composer's crosstalk from MULTIPLE bound roles is still exercised
    # since agent+action+patient[+polarity] are all bound into one composite either way).
    usable = [f for f in facts
              if isinstance(f.get("agent"), str) and isinstance(f.get("action"), str)
              and isinstance(f.get("patient"), str)]
    idx = rng.choice(len(usable), size=min(n_store, len(usable)), replace=False)
    sample = [usable[i] for i in idx]
    errs = []
    stored = []
    for f in sample:
        composer.store(f["agent"], f["action"], f.get("polarity") and f["patient"] or f["patient"],
                        polarity=f.get("polarity") or "AFFIRM")
        stored.append(f)
    _log(f"calibration store: {len(stored)} real facts stored via production RFPhasorComposer.store()", t0)
    # unbind the cue role from each just-stored composite; compare to the TRUE concept phase for that role's
    # filler word. Composite index = len(composer.kb) - len(stored) + i (facts appended in order, nothing else
    # stored yet on this composer).
    base = len(composer.kb) - len(stored)
    for i, f in enumerate(stored):
        _fact, comp = composer.kb[base + i]
        true_word = f[cue_role]
        rec = composer._unbind_phases(comp, cue_role)
        true_phase = composer.concepts[true_word]
        d = (np.asarray(rec) - np.asarray(true_phase) + 0.5) % 1.0 - 0.5   # wrap to [-0.5, 0.5) fractional cycle
        errs.append(d)
    errs = np.concatenate(errs) if errs else np.zeros(0)
    sigma = float(np.std(errs)) if errs.size else 0.0
    return sigma, len(stored), errs


def raw_shard_table(composer, vocab, sigma_frac: float, n_query: int, seed: int, t0: float):
    """Step 2 (data collection): sample n_query REAL words, add the CALIBRATED noise, and call the PRODUCTION
    `_dg_shard_select` with conf_floor forced to -999 (i.e. it NEVER escalates a non-empty shard) so it always
    returns the shard's raw local top-1 (word, peak) -- the SAME production routing + scoring, just with the
    escalate GATE disabled so the raw table can be built ONCE and swept over MANY candidate conf_floor values as
    a cheap post-process (no rebuild, no re-routing, per candidate). ALSO records `true_in_shard`: whether the
    TRUE stored word's own index is even a MEMBER of the DG-routed candidate set -- distinguishes a MISROUTE
    (the true code never reached the shard at all, so no threshold could ever recover it) from a threshold-only
    miss (the true code is IN the shard but its score didn't clear the gate). Returns a list of row dicts."""
    rng = np.random.default_rng(seed * 104729 + 2)
    words = rng.choice(vocab, size=n_query, replace=True)
    word_to_idx = {w: i for i, w in enumerate(composer.words)}
    saved_floor = composer._dg_conf_floor
    composer._dg_conf_floor = -999.0
    rows = []
    for w in words:
        true_phase = np.asarray(composer.concepts[w], dtype=float)
        noisy = (true_phase + rng.normal(0.0, sigma_frac, size=true_phase.shape)) % 1.0
        decided, peak = composer._dg_shard_select(noisy)   # never escalates here (floor=-999) unless shard empty
        shard = composer._dg_index.query(noisy * (2.0 * np.pi))
        true_in_shard = bool(word_to_idx[w] in set(shard.tolist()))
        rows.append({"true": w, "shard_top1": decided, "peak": peak, "shard_size": int(shard.size),
                     "true_in_shard": true_in_shard})
    composer._dg_conf_floor = saved_floor
    n_true_in = sum(1 for r in rows if r["true_in_shard"])
    _log(f"MISROUTE check: true word's own code is a member of its DG-routed shard in "
         f"{n_true_in}/{len(rows)} ({n_true_in/len(rows):.1%}) queries (the rest can NEVER be answered correctly "
         f"by the shard at ANY conf_floor -- misrouted, not merely under-confident)", t0)
    return rows


def floor_sweep_report(rows, D: int, floors, label: str, t0: float):
    """Post-process the raw table over several candidate conf_floor values: at each floor, a row ESCALATES iff
    shard_size==0 OR peak < floor*D (the exact `_dg_shard_select` gate, replayed here without re-routing). Reports
    escalation_frac and PARITY (fraction of the newly-decisive rows whose shard_top1 == the true word) per floor
    -- the number that must stay ~1.0 for a lowered floor to be safe."""
    out = []
    for floor in floors:
        n_esc = 0
        decisive_true = decisive_correct = 0
        for r in rows:
            if r["shard_size"] == 0 or r["peak"] < floor * D:
                n_esc += 1
            else:
                decisive_true += 1
                if r["shard_top1"] == r["true"]:
                    decisive_correct += 1
        n = len(rows)
        parity = decisive_correct / decisive_true if decisive_true else None
        out.append({"conf_floor": floor, "escalation_frac": n_esc / n if n else 0.0,
                     "n_decisive": decisive_true, "parity": parity})
        _log(f"{label} floor={floor:.3f}: escalation_frac={n_esc/n:.1%} n_decisive={decisive_true} "
             f"parity(decisive_top1==true)={parity}", t0)
    return out


def real_escalation_sweep(composer, vocab, sigma_frac: float, n_query: int, seed: int, conf_floor: float, t0: float):
    """Convenience wrapper: build the raw table once, report at the single requested conf_floor (used by main()'s
    baseline/lever run), and also return the raw table for a downstream multi-floor sweep."""
    rows = raw_shard_table(composer, vocab, sigma_frac, n_query, seed, t0)
    rep = floor_sweep_report(rows, composer.D, [conf_floor], "real", t0)[0]
    escalated_empty_shard = sum(1 for r in rows if r["shard_size"] == 0)
    escalated_low_peak = sum(1 for r in rows if r["shard_size"] > 0 and r["peak"] < conf_floor * composer.D)
    decisive_matches = [(r["true"], r["shard_top1"]) for r in rows
                         if r["shard_size"] > 0 and r["peak"] >= conf_floor * composer.D]
    n = len(rows)
    _log(f"real sweep: n={n} escalated={n-rep['n_decisive']} ({rep['escalation_frac']:.1%}) "
         f"[empty_shard={escalated_empty_shard} low_peak={escalated_low_peak}] "
         f"decisive_top1_self_agree={rep['parity']}", t0)
    return {
        "n_query": n, "escalated": n - rep["n_decisive"], "escalation_frac": rep["escalation_frac"],
        "escalated_empty_shard": escalated_empty_shard, "escalated_low_peak": escalated_low_peak,
        "mean_shard_size": float(np.mean([r["shard_size"] for r in rows])),
        "median_shard_size": float(np.median([r["shard_size"] for r in rows])),
        "peak_scores_D_frac": [float(r["peak"] / composer.D) for r in rows if r["shard_size"] > 0],
        "decisive_self_agree": rep["parity"],
        "decisive_words": decisive_matches,
        "raw_rows": rows,   # kept for an optional downstream multi-floor sweep; stripped before JSON dump
    }


def parity_check(composer, decisive_matches, n_parity: int, seed: int, t0: float):
    """Ground-truth full-codebook argmax for a subsample of the DECISIVE rows (the shard-answered ones) -- verify
    the shard's answer, when it returns one, agrees with the full scan (recall/parity, not just escalation rate).
    One batched BLAS matmul (reuses the same op `_full_host_select` performs, called on a batch of rows)."""
    if not decisive_matches:
        return {"n_checked": 0, "parity": None}
    rng = np.random.default_rng(seed * 15485863 + 3)
    idx = rng.choice(len(decisive_matches), size=min(n_parity, len(decisive_matches)), replace=False)
    sub = [decisive_matches[i] for i in idx]
    true_words = [w for w, _d in sub]
    # rebuild the exact noisy query used isn't retained per-row (memory); re-derive is unnecessary -- parity is
    # about whether the SHARD-decided word equals the TRUE word (both a self-consistency + a same-noise-regime
    # top-1 accuracy check). We ALSO run a real full-codebook argmax over the composer's OWN true concept phase
    # for each true_word (i.e. would the full scan, given the identical noiseless truth code, single out the
    # right concept) -- a sanity floor, not the noisy-query parity itself (that is the decisive_self_agree stat
    # already reported by the sweep, which compares against the SAME noisy query both paths would receive).
    cb = composer._dg_codebook
    true_z = np.exp(2j * np.pi * np.stack([composer.concepts[w] for w in true_words]))
    cb_z = np.exp(2j * np.pi * cb)
    sims = (true_z @ np.conj(cb_z).T).real
    full_argmax_words = [composer.words[int(j)] for j in np.argmax(sims, axis=1)]
    agree = float(np.mean([fw == tw for fw, tw in zip(full_argmax_words, true_words)]))
    _log(f"parity floor check (full-scan self-consistency over {len(sub)} true concept codes): {agree:.4f}", t0)
    return {"n_checked": len(sub), "full_scan_self_consistency": agree}


def synthetic_sweep(V: int, D: int, g: int, G: int, c: int, conf_floor: float, sigma_rad: float,
                     n_query: int, seed: int, t0: float):
    """Step 3: the SAME sweep over a synthetic uniform-random FHRR codebook at matched V/D, reusing the original
    de-risk's gen_fhrr_phases/DGSparseIndex by import (not reimplemented). Scoring/decide is a literal one-line
    reproduction of _dg_shard_select's formula (documented in the module docstring) since there is no composer
    instance for a bare synthetic codebook."""
    from research.runners._sparse_indexed_retrieval_derisk import gen_fhrr_phases, DGSparseIndex
    rng = np.random.default_rng(seed * 962927 + 4)
    phases = gen_fhrr_phases(V, D, rng)     # radians, (V, D)
    m = max(2, int(np.ceil(V ** (1.0 / g))))
    idx = DGSparseIndex(D=D, m=m, g=g, G=G, c=c, seed=seed)
    tb = time.perf_counter()
    idx.build(phases)
    build_s = time.perf_counter() - tb
    _log(f"synthetic index build: V={V} D={D} m={m} g={g} G={G} c={c} ({build_s:.1f}s)", t0)

    qrng = np.random.default_rng(seed * 1299709 + 5)
    q_ids = qrng.integers(0, V, size=n_query)
    escalated = 0
    escalated_empty_shard = 0
    escalated_low_peak = 0
    shard_sizes = []
    peaks = []
    for i in q_ids:
        noisy = phases[i] + qrng.normal(0.0, sigma_rad, size=phases[i].shape)
        shard = idx.query(noisy)
        shard_sizes.append(int(shard.size))
        if shard.size == 0:
            escalated += 1
            escalated_empty_shard += 1
            continue
        cb = phases[shard]
        sc = np.cos(noisy[None, :] - cb).sum(axis=1)         # SAME formula as _dg_shard_select (radians here)
        t = int(np.argmax(sc)); peak = float(sc[t])
        if peak < conf_floor * D:
            escalated += 1
            escalated_low_peak += 1
        else:
            peaks.append(peak)
    n = len(q_ids)
    esc_frac = escalated / n if n else 0.0
    _log(f"synthetic sweep: n={n} escalated={escalated} ({esc_frac:.1%}) [empty_shard={escalated_empty_shard} "
         f"low_peak={escalated_low_peak}] mean_shard_size={np.mean(shard_sizes):.1f}", t0)
    del idx, phases
    gc.collect()
    return {
        "V": V, "D": D, "m": m, "g": g, "G": G, "c": c, "build_s": build_s,
        "n_query": n, "escalated": escalated, "escalation_frac": esc_frac,
        "escalated_empty_shard": escalated_empty_shard, "escalated_low_peak": escalated_low_peak,
        "mean_shard_size": float(np.mean(shard_sizes)), "median_shard_size": float(np.median(shard_sizes)),
        "peak_scores_D_frac": [float(p / D) for p in peaks],
    }


def main():
    ap = argparse.ArgumentParser(description="DG-shard escalation diagnostic (board #66/#192)")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-store", type=int, default=300, help="real facts stored to calibrate resonate noise")
    ap.add_argument("--n-query", type=int, default=3000, help="escalation-sweep samples (real AND synthetic)")
    ap.add_argument("--n-parity", type=int, default=500, help="subsample for the full-scan parity floor check")
    ap.add_argument("--cue-role", default="patient", help="role unbound for the noise calibration")
    ap.add_argument("--g", type=int, default=3)
    ap.add_argument("--G", type=int, default=16)
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--conf-floor", type=float, default=0.5)
    ap.add_argument("--lever", choices=["none", "more_probes", "lower_conf_floor"], default="none")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--floor-sweep", default=None,
                     help="comma-separated conf_floor values to ALSO report from the SAME raw table (no rebuild, "
                          "no re-routing) -- e.g. '0.20,0.25,0.30,0.35,0.40,0.45,0.50'")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    t0 = time.time()
    G = a.G * 2 if a.lever == "more_probes" else a.G
    conf_floor = a.conf_floor * 0.7 if a.lever == "lower_conf_floor" else a.conf_floor
    _log(f"start: bundle={a.bundle} seed={a.seed} lever={a.lever} G={G} conf_floor={conf_floor:.3f}", t0)

    mani, facts = load_bundle(a.bundle)
    D = int(mani["D"]); vocab = mani["vocab"]; V = len(vocab)
    _log(f"bundle loaded: n_facts={mani['n_facts']} vocab={V} n_shards={mani['n_shards']} D={D}", t0)

    from research.runners.rf_phasor_composer import RFPhasorComposer
    composer = RFPhasorComposer(seed=mani["seed"], D=D, vocab=vocab, enable_sparse_index=True,
                                 sparse_index_g=a.g, sparse_index_G=G, sparse_index_c=a.c,
                                 sparse_index_conf_floor=conf_floor)
    _log(f"composer built (codebook only, byte-identical to production seed+vocab); RSS checkpoint", t0)

    sigma_frac, n_calibrated, errs = calibrate_noise(composer, facts, a.n_store, a.seed, a.cue_role, t0)
    _log(f"calibrated resonate-recovery noise: sigma_fractional_cycle={sigma_frac:.5f} "
         f"(n={n_calibrated} facts, role={a.cue_role}) | de-risk's own assumed sigma=0.30 RADIANS = "
         f"{0.30/(2*np.pi):.5f} fractional-cycle for comparison", t0)

    tb = time.perf_counter()
    composer._ensure_dg_index()
    build_s = time.perf_counter() - tb
    _log(f"REAL DG index build: V={V} D={D} m={composer._dg_index.m} g={a.g} G={G} c={a.c} ({build_s:.1f}s)", t0)

    real = real_escalation_sweep(composer, vocab, sigma_frac, a.n_query, a.seed, conf_floor, t0)
    parity = parity_check(composer, real["decisive_words"], a.n_parity, a.seed, t0)
    real["parity"] = parity
    real["build_s"] = build_s

    floor_sweep_results = None
    if a.floor_sweep:
        floors = [float(x) for x in a.floor_sweep.split(",")]
        floor_sweep_results = floor_sweep_report(real["raw_rows"], D, floors, "real-floor-sweep", t0)

    real.pop("decisive_words")   # large; not needed in the JSON artifact
    real.pop("raw_rows")         # large; not needed in the JSON artifact (floor_sweep_results already derived)

    # free the real index before building the synthetic one (peak-RSS bound)
    del composer._dg_index, composer._dg_codebook
    gc.collect()
    _log("real DG index freed (gc.collect) before synthetic build", t0)

    synth = None
    if not a.skip_synthetic:
        sigma_rad = sigma_frac * 2.0 * np.pi
        synth = synthetic_sweep(V, D, a.g, G, a.c, conf_floor, sigma_rad, a.n_query, a.seed, t0)

    out = {
        "bundle": a.bundle, "seed": a.seed, "lever": a.lever, "V": V, "D": D,
        "g": a.g, "G": G, "c": a.c, "conf_floor": conf_floor,
        "sigma_fractional_cycle_calibrated": sigma_frac, "n_calibration_facts": n_calibrated,
        "cue_role": a.cue_role,
        "real": real, "synthetic": synth, "floor_sweep": floor_sweep_results,
        "rss_mb_final": _rss_mb(),
        "elapsed_s": time.time() - t0,
    }
    if a.json:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        _log(f"wrote {a.json}", t0)

    print(f"\n===== VERDICT INPUTS (lever={a.lever}) =====")
    print(f"  REAL escalation_frac      = {real['escalation_frac']:.1%}  (n={real['n_query']})")
    if synth:
        print(f"  SYNTHETIC escalation_frac = {synth['escalation_frac']:.1%}  (n={synth['n_query']}, matched V/D)")
        gap = real['escalation_frac'] - synth['escalation_frac']
        print(f"  REAL - SYNTHETIC gap      = {gap:+.1%}  "
              f"({'real-geometry-specific' if gap > 0.10 else 'operating-point (D/V), not real-geometry-specific'})")
    print(f"  full-scan self-consistency floor = {parity.get('full_scan_self_consistency')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
