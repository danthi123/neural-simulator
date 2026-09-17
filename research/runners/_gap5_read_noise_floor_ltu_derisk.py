"""gap#5 learn-through-use, READ-FIDELITY INSTRUMENT (2026-09-17): a DIAGNOSTIC, NOT a mechanism lever.

THE QUESTION: several gap#5 learn-through-use runners on the Ecker AdEx CA3 store (graded-recall,
reverse-edge-hetero-depression, forward-band-homeostatic-scaling, ...) report SOME seeds as "insensitive" -- the
weak-cue depth_frac gain (after-consolidation minus before) does not clear the `--depth-gain-min` bar (0.05) on
those seeds, while others clear it easily. Every one of those runners reads recall with EXACTLY ONE replay trial
per seed per weight-state (one `rest_and_replay` call -> one `_score_periods_graded` scalar). That conflates two
possible explanations for a seed reading "no gain": (a) the substrate genuinely did not deepen recall for that
seed's specific encoded band/noise draw (real variance), or (b) the single read trial's own noise floor (which
memory the non-specific prefix seed happens to pick each SWR period, which subsample of cue cells the read
happens to draw) is comparable to or larger than 0.05 in depth_frac units, so "no gain" on one trial is
UNDEFINED, not a negative -- the WALL REFRAME's "the instrument is part of the emulation" applied literally to
this lane's own READ.

THIS RUNNER measures that read-trial noise directly, independent of any weight-update mechanism: it takes ONE
already-encoded store (built with `build_store`+`encode`, exactly as every gap#5 Ecker-store runner does) and ONE
already-consolidated weight snapshot (via the ESTABLISHED directional write, `consolidate_by_btsp_replay_delayed`
-- UNCHANGED, called only to produce a realistic post-consolidation weight state to read from, never as the thing
being evaluated), then re-reads the SAME weak-cue weak-cue operating point `--n-read-trials` times with the
SUBSTRATE, ENCODED BAND and OU/membrane-noise seed all held BIT-FIXED (`seed` unchanged) and ONLY the read-trial
cue-cell subsample + which-memory-replays choice varied (`read_trial_seed=seed*1000003+k`, STEP 1 below). The
spread across those repeated reads at a FIXED weight state IS the population-read noise floor; the spread of the
mean read ACROSS SEEDS is the substrate/encoding variance the mechanism runners are actually trying to detect.
Comparing the two (read_SNR) tells you, PER SEED and in aggregate, whether "insensitive" means read-noise-
dominated (need more read trials, not a new weight mechanism) or genuine substrate variance (a real biological
question).

STEP 1 (additive, default-off, in `_gap5_ecker_adex_ca3_stdp_band_derisk.rest_and_replay`): a new trailing kwarg
`read_trial_seed=None` keys ONLY the cue-cell subsample RNG (`cell_rng`) and the which-memory-replays RNG
(`choice_rng`) off `rt = int(seed) if read_trial_seed is None else int(read_trial_seed)` instead of `seed`
directly. Every other seed in the pipeline (`build_store`'s heterogeneity/OU seed, `encode`'s own cue-cell RNG,
`consolidate_by_btsp_replay_delayed`'s own cue-cell/choice RNGs) is UNTOUCHED -- this is a READ-side-only
instrument. `read_trial_seed=None` (the default, what every existing caller in the repo still passes implicitly)
-> `rt == seed` -> IDENTICAL RNG streams to before the edit -> BYTE-IDENTICAL output, verified below by an exact
SHA-256 hash match between the old call shape (no kwarg) and the new call shape (`read_trial_seed=int(seed)`
passed explicitly) on the same encoded store.

Reuse-by-import (NO sim/ edit, NO new mechanism -- only a new READ-TRIAL RNG axis + a repeated-read wrapper on top
of instruments/writes that already exist):
  build_store / encode / rest_and_replay / measure_band / _load_weights / _smooth <- _gap5_ecker_adex_ca3_stdp_band_derisk
  consolidate_by_btsp_replay_delayed / measure_band_from <- _gap5_ecker_replay_learn_through_use_derisk (the
    ESTABLISHED directional write -- called unmodified, only to produce a weight state to read from)
  _score_periods_graded / verify_instrument <- _gap5_graded_recall_learn_through_use_derisk (the PROVEN-graded
    depth+tau instrument, unmodified -- re-verified per seed here since a per-seed number is the deliverable)

METRICS (per seed): mu_read_before / sigma_read_before (N reads on w_learned, pre-consolidation), mu_read /
sigma_read (== mu_read_after / sigma_read_after -- N reads on w_consol, the weak-cue RECALL OPERATING POINT the
mechanism runners actually score gain at). AGGREGATE: sigma_read_bar = mean(sigma_read) across seeds (the pooled
single-trial read-noise floor); sigma_substrate = std across seeds of mu_read (the across-seed spread the
mechanism runners are trying to attribute to biology); read_SNR = sigma_substrate / sigma_read_bar (>>1 =>
genuine substrate variance; ~1 or below => the seed-to-seed spread reported elsewhere in this lane is
READ-NOISE-DOMINATED, UNDEFINED as a substrate claim on a single read trial); MDG = 2*sqrt(2)*sigma_read_bar (the
minimum depth_frac gain a SINGLE read-trial comparison could resolve from noise alone); K_reads_needed =
ceil((MDG / resolve_bar)^2) (independent read trials per arm needed to bring the minimum-detectable-gain down to
the 0.05 `--depth-gain-min` bar this lane's GO thresholds already use).

INSTRUMENT-VALIDITY gate (the NUMBER is the deliverable regardless of magnitude -- this is NOT a mechanism GO):
  * verify_instrument reads GRADED on each seed's own known-good encoded store (reused unmodified).
  * weights stay FROZEN through every one of the 2*N read trials per seed (`rest_and_replay`'s own
    weights_frozen check, injects ONLY external current -- the NUMPY-REFERENCE guard).
  * sigma_read > 0 on >= 5/6 seeds -- THE ANTI-CHEAT: distinct read_trial_seed values must actually change the
    read (a wiring bug that left `rt` a no-op, or a read regime so long/oversampled that N=12 trials always land
    identically, would show sigma_read == 0 and must not be reported as a valid noise-floor measurement).
  * byte-identical-off: `read_trial_seed=None` (omitted, i.e. every existing caller) reproduces the EXACT SHA-256
    hash of `read_trial_seed=int(seed)` passed explicitly, on the same encoded store (`--byte-identical-check`).

  Byte-identical-off: SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_read_noise_floor_ltu_derisk --byte-identical-check --seeds 42
  Smoke (1 seed):     SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_read_noise_floor_ltu_derisk --seeds 42 --n-read-trials 12
  6-seed:             SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_read_noise_floor_ltu_derisk --seeds 42 43 44 100 101 102 --n-read-trials 12 \\
          --out research/findings/raw/gap5_ecker_adex/read_noise_floor_6seed.json
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import to_host, get_backend  # noqa: E402
from research.runners._gap5_ecker_adex_ca3_stdp_band_derisk import (  # noqa: E402
    build_store, encode, rest_and_replay, measure_band, _load_weights, _smooth,
)
from research.runners._gap5_ecker_replay_learn_through_use_derisk import (  # noqa: E402
    consolidate_by_btsp_replay_delayed, measure_band_from,
)
from research.runners._gap5_graded_recall_learn_through_use_derisk import (  # noqa: E402
    _score_periods_graded, verify_instrument,
)
from tools.verdict import Verdict  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "read_noise_floor_ltu.json"


# ----------------------------------------------------------------------------------------------------------------------
# ONE read trial: fresh store, load a given weight snapshot, replay READ with a given read_trial_seed (STEP 1's new
# kwarg), score with the graded instrument. Mirrors `_read_graded` (graded-recall runner) exactly except for the
# read_trial_seed passthrough -- everything else (store build seed, cue timing, detection thresholds) is identical.
# ----------------------------------------------------------------------------------------------------------------------
def _read_at(bkw, seed, w_host, a, *, cue_pa, cue_frac, swr_period, rest_steps, read_trial_seed, tag):
    s = build_store(seed, **bkw)
    _load_weights(s, w_host)
    r = rest_and_replay(s, rest_steps, seed, swr_period=swr_period, cue_pa=cue_pa,
                        cue_steps=a.cue_steps, cue_frac=cue_frac, seed_on=True,
                        read_trial_seed=read_trial_seed)
    sc = _score_periods_graded(r["F"], s["asm_local"], r["env_seed_log"], swr_period,
                               W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    return dict(depth_frac=sc["depth_frac_mean"], depth_mean=sc["depth_mean"], tau=sc["tau_mean"],
                n_multi=sc["n_multi"], frozen=r["weights_frozen"], tag=tag)


# ----------------------------------------------------------------------------------------------------------------------
# BYTE-IDENTICAL-OFF CHECK (asserted IN THE DATA, per docs/TERMS.md -- a hash/exact compare, not read-the-code).
# Same encoded store/weights; one read call OMITS read_trial_seed (the old call shape every existing caller uses),
# the other passes read_trial_seed=int(seed) EXPLICITLY. Both must land on rt==seed -> bit-identical firing.
# ----------------------------------------------------------------------------------------------------------------------
def byte_identical_check(seed, a):
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult
    read_period = a.read_swr_period if a.read_swr_period > 0 else a.swr_period

    st0 = build_store(seed, **bkw)
    encode(st0, seed, **enc_kw)
    w_learned = np.asarray(to_host(st0["bridge"].cp_connections.data)).copy()

    st_old = build_store(seed, **bkw)
    _load_weights(st_old, w_learned)
    r_old = rest_and_replay(st_old, a.rest_steps, seed, swr_period=read_period, cue_pa=weak_pa,
                            cue_steps=a.cue_steps, cue_frac=a.weak_cue_frac, seed_on=True)  # NO read_trial_seed kwarg

    st_new = build_store(seed, **bkw)
    _load_weights(st_new, w_learned)
    r_new = rest_and_replay(st_new, a.rest_steps, seed, swr_period=read_period, cue_pa=weak_pa,
                            cue_steps=a.cue_steps, cue_frac=a.weak_cue_frac, seed_on=True,
                            read_trial_seed=int(seed))  # EXPLICIT read_trial_seed == seed

    h_old = hashlib.sha256(np.ascontiguousarray(r_old["F"]).tobytes()).hexdigest()
    h_new = hashlib.sha256(np.ascontiguousarray(r_new["F"]).tobytes()).hexdigest()
    env_match = bool(r_old["env_seed_log"] == r_new["env_seed_log"])
    exact = bool(h_old == h_new) and env_match
    print(f"[byte-identical-check] seed={seed} sha256_old={h_old[:16]} sha256_new={h_new[:16]} "
          f"EXACT_HASH_MATCH={exact} env_seed_log_match={env_match} "
          f"frozen_old={r_old['weights_frozen']} frozen_new={r_new['weights_frozen']}", flush=True)
    return dict(seed=seed, sha256_old=h_old, sha256_new=h_new, exact_hash_match=exact, env_seed_log_match=env_match)


# ----------------------------------------------------------------------------------------------------------------------
# PER-SEED: build+encode -> w_learned; consolidate (established write, UNCHANGED) -> w_consol; then N independent
# weak-cue graded reads on EACH weight state, varying ONLY read_trial_seed. The spread over those N reads at a
# FIXED weight state is the read-noise floor this instrument exists to quantify.
# ----------------------------------------------------------------------------------------------------------------------
def one_seed(seed, a):
    t0 = time.time()
    out = {"seed": seed}
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult
    read_period = a.read_swr_period if a.read_swr_period > 0 else a.swr_period

    # 0. INSTRUMENT VALIDITY: verify_instrument (reused unmodified) must read GRADED on THIS seed's own known-good
    #    encoded store -- run per seed since a per-seed number is the deliverable, not a single pre-flight check.
    verify = verify_instrument(seed, a)
    print(f"  [seed {seed}] verify_instrument: graded={verify['graded']} depth_frac_range="
          f"{verify['depth_frac_range']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # 1. BUILD + ENCODE (identical scaffold to every gap#5 Ecker-store runner in this lane).
    st = build_store(seed, **bkw)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band_before = measure_band(st)
    out["band_before"] = band_before

    # 2. CONSOLIDATE via the ESTABLISHED directional write (unmodified) -- produces a realistic post-consolidation
    #    weight snapshot to read from. This runner does not evaluate or tune this write; it is a fixed input.
    st_c = build_store(seed, **bkw)
    _load_weights(st_c, w_learned)
    cons = consolidate_by_btsp_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                              elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                              eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                              delay_steps=a.fwd_delay_steps, **cons_kw)
    w_consol = cons["w_after"]
    band_after = measure_band_from(w_consol, st_c)
    out["band_after"] = band_after
    out["consolidate"] = dict(dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"])
    print(f"  [seed {seed}] CONSOLIDATE(established write): dw_fwd={cons['dw_fwd']:.2f} dw_rev={cons['dw_rev']:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 3. THE INSTRUMENT: N independent weak-cue graded reads on w_learned (before) and w_consol (after -- the
    #    weak-cue RECALL OPERATING POINT), read_trial_seed = seed*1000003 + k for k in 0..N-1. Store build + OU/
    #    membrane noise stay pinned to `seed` throughout (FIXED substrate + intrinsic noise); ONLY the cue-cell
    #    subsample + which-memory-replays choice vary -- that isolates population-READ noise from substrate
    #    variance (the task this instrument exists to answer).
    N = int(a.n_read_trials)
    depths_before, depths_after, frozen_all = [], [], []
    for k in range(N):
        rt = int(seed) * 1000003 + k
        rb = _read_at(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, swr_period=read_period,
                      rest_steps=a.rest_steps, read_trial_seed=rt, tag=f"before_k{k}")
        ra = _read_at(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac, swr_period=read_period,
                      rest_steps=a.rest_steps, read_trial_seed=rt, tag=f"after_k{k}")
        depths_before.append(rb["depth_frac"]); depths_after.append(ra["depth_frac"])
        frozen_all.append(bool(rb["frozen"])); frozen_all.append(bool(ra["frozen"]))
    mu_before = float(np.mean(depths_before))
    sigma_before = float(np.std(depths_before, ddof=1)) if N > 1 else 0.0
    mu_after = float(np.mean(depths_after))
    sigma_after = float(np.std(depths_after, ddof=1)) if N > 1 else 0.0
    frozen_ok = bool(all(frozen_all))
    print(f"  [seed {seed}] READ-NOISE ({N} trials/arm): before depth_frac mu={mu_before:.3f} sigma={sigma_before:.4f} "
          f"| after(weak-cue recall op-point) mu={mu_after:.3f} sigma={sigma_after:.4f} frozen_ok={frozen_ok} "
          f"({time.time()-t0:.0f}s)", flush=True)

    out["reads_before"] = depths_before
    out["reads_after"] = depths_after
    out["mu_read_before"] = mu_before
    out["sigma_read_before"] = sigma_before
    out["mu_read"] = mu_after            # == mu_read_after: the weak-cue RECALL OPERATING POINT
    out["sigma_read"] = sigma_after      # == sigma_read_after: the read-noise floor at that operating point
    out["verify_instrument"] = dict(graded=bool(verify["graded"]), depth_frac_range=verify["depth_frac_range"],
                                    tau_range=verify["tau_range"])
    out["frozen_ok"] = frozen_ok

    # ============ PER-SEED INSTRUMENT-VALIDITY (this is a DIAGNOSTIC -- the NUMBER is the deliverable regardless
    # of magnitude; the gate is whether the instrument itself is trustworthy, never a mechanism performance bar). ==
    sigma_read_positive = bool(sigma_after > 0.0)
    seed_valid = bool(verify["graded"] and frozen_ok and sigma_read_positive)
    out["checks"] = dict(instrument_graded=bool(verify["graded"]), frozen_ok=frozen_ok,
                         sigma_read_positive=sigma_read_positive)
    out["seed_valid"] = seed_valid
    print(f"  [seed {seed}] => {'VALID' if seed_valid else 'INVALID'}  checks={out['checks']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-mem", type=int, default=6)
    ap.add_argument("--asm-size", type=int, default=80)
    ap.add_argument("--within-density", type=float, default=0.5)
    ap.add_argument("--rest-steps", type=int, default=9000)
    ap.add_argument("--consol-steps", type=int, default=6500)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--w-within", type=float, default=60.0)
    ap.add_argument("--between-init", type=float, default=15.0)
    ap.add_argument("--b-override", type=float, default=120.0)
    ap.add_argument("--stdp-w-max", type=float, default=900.0)
    ap.add_argument("--stdp-a-plus", type=float, default=0.05)
    ap.add_argument("--stdp-a-minus", type=float, default=0.06)
    ap.add_argument("--stdp-tau", type=float, default=20.0)
    # the ESTABLISHED directional write (IDENTICAL to the graded-recall/hetero-depression runners' decisive cfg --
    # this runner does not tune it; it only reads from the weight states it produces).
    ap.add_argument("--btsp-elig-tau", type=float, default=80.0)
    ap.add_argument("--btsp-plat-tau", type=float, default=1.0)
    ap.add_argument("--btsp-eta", type=float, default=0.001)
    ap.add_argument("--btsp-w-max", type=float, default=900.0)
    ap.add_argument("--fwd-delay-steps", type=int, default=90)
    # ENCODE
    ap.add_argument("--n-laps", type=int, default=14)
    ap.add_argument("--enc-step", type=int, default=80)
    ap.add_argument("--enc-dwell", type=int, default=40)
    ap.add_argument("--enc-gap", type=int, default=600)
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0)
    ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    # SWR replay / prefix seed (consolidation write side)
    ap.add_argument("--swr-period", type=int, default=650)
    ap.add_argument("--cue-pa", type=float, default=9000.0)
    ap.add_argument("--cue-steps", type=int, default=40)
    ap.add_argument("--cue-frac", type=float, default=0.6)
    ap.add_argument("--weak-cue-mult", type=float, default=0.5)
    ap.add_argument("--weak-cue-frac", type=float, default=0.35)
    ap.add_argument("--ou-sigma", type=float, default=40.0)
    ap.add_argument("--read-swr-period", type=int, default=0)
    # detection
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--active-frac", type=float, default=0.10)
    ap.add_argument("--onset-frac", type=float, default=0.06)
    # THE INSTRUMENT's own knobs
    ap.add_argument("--n-read-trials", type=int, default=12, help="independent read trials per weight-state per "
                    "seed (read_trial_seed = seed*1000003 + k, k in 0..N-1)")
    ap.add_argument("--resolve-bar", type=float, default=0.05, help="the depth_frac gain bar (matches the "
                    "graded-recall / reverse-edge-hetero-depression runners' --depth-gain-min) that K_reads_needed "
                    "resolves against")
    # instrument verification (reused unmodified from the graded-recall runner, run PER SEED here)
    ap.add_argument("--verify-cue-mults", type=float, nargs="+", default=[1.0, 0.85, 0.7, 0.5, 0.35, 0.2])
    ap.add_argument("--verify-min-range", type=float, default=0.15)
    # modes
    ap.add_argument("--byte-identical-check", action="store_true", help="run ONLY the byte-identical-off hash "
                    "comparison (rest_and_replay default call-shape vs explicit read_trial_seed=seed), skip "
                    "everything else")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    _, backend = get_backend()
    print(f"[read-noise-floor] Ecker AdEx CA3 READ-FIDELITY INSTRUMENT (diagnostic, not a weight lever) | "
          f"n_read_trials={a.n_read_trials} weak_cue={a.cue_pa*a.weak_cue_mult:.0f}pA@{a.weak_cue_frac} "
          f"swr={a.swr_period} n_mem={a.n_mem} asm={a.asm_size} resolve_bar={a.resolve_bar} seeds={a.seeds} "
          f"backend={backend}", flush=True)

    if a.byte_identical_check:
        rows = [byte_identical_check(s, a) for s in a.seeds]
        all_exact = all(r["exact_hash_match"] for r in rows)
        print(f"[read-noise-floor] BYTE-IDENTICAL-OFF: "
              f"{'CONFIRMED (exact hash match)' if all_exact else 'NOT exact -- see per-seed rows'} on "
              f"{sum(r['exact_hash_match'] for r in rows)}/{len(rows)} seeds", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(str(a.out) + ".byte_identical_check.json").write_text(json.dumps(rows, indent=2, default=str))
        return 0 if all_exact else 1

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None and per:
        n_seeds = len(per)
        bar = max(1, (n_seeds + 1) // 2) if n_seeds < 6 else 5
        n_valid = sum(1 for p in per if p.get("seed_valid"))
        go = n_valid >= bar

        sigma_read_bar = float(np.mean([p["sigma_read"] for p in per]))
        mu_reads_after = [p["mu_read"] for p in per]
        sigma_substrate = float(np.std(mu_reads_after, ddof=1)) if n_seeds > 1 else 0.0
        read_snr = sigma_substrate / sigma_read_bar if sigma_read_bar > 0 else float("inf")
        mdg = 2.0 * math.sqrt(2.0) * sigma_read_bar
        k_reads_needed = int(math.ceil((mdg / max(a.resolve_bar, 1e-9)) ** 2)) if sigma_read_bar > 0 else 0

        n_sigma_pos = sum(1 for p in per if p["checks"]["sigma_read_positive"])
        n_graded = sum(1 for p in per if p["checks"]["instrument_graded"])
        n_frozen = sum(1 for p in per if p["checks"]["frozen_ok"])
        classification = "READ-NOISE-DOMINATED" if read_snr < 1.0 else "GENUINE-SUBSTRATE-VARIANCE"

        verdict = (f"READ-FIDELITY INSTRUMENT: {n_valid}/{n_seeds} seeds instrument-VALID (graded {n_graded}/"
                   f"{n_seeds}, weights frozen {n_frozen}/{n_seeds}, sigma_read>0 {n_sigma_pos}/{n_seeds} "
                   f"[anti-cheat]). Pooled sigma_read_bar={sigma_read_bar:.4f} (single weak-cue read-trial noise, "
                   f"depth_frac units, at the post-consolidation recall operating point) vs "
                   f"sigma_substrate={sigma_substrate:.4f} (across-{n_seeds}-seed spread of that same operating "
                   f"point's mean read) => read_SNR={read_snr:.2f} => this lane's seed-to-seed spread reads as "
                   f"{classification}. Minimum-detectable-gain from a SINGLE read trial: MDG={mdg:.4f}; to resolve "
                   f"the {a.resolve_bar:.2f} depth_frac gain bar this lane's GO thresholds already use would need "
                   f"K_reads_needed={k_reads_needed} independent read trials per arm (this run used "
                   f"N={a.n_read_trials}/arm). This is a DIAGNOSTIC NUMBER, not a mechanism verdict -- it does not "
                   f"itself confirm or refute any weight lever; it says whether a single-read 'no gain' on those "
                   f"levers is a real substrate result or an under-sampled instrument.")

        v = Verdict("Ecker AdEx CA3: population-read SNR instrument at the weak-cue recall operating point "
                    "(diagnostic -- reports a number, is NOT itself a mechanism lever)")
        v.require("the underlying GRADED depth/tau instrument (reused unmodified) itself reads graded on each "
                  "seed's own known-good encoded store", n_graded, expect=lambda x, b=bar: x >= b)
        v.require("weights stay FROZEN through every read trial (the read_trial_seed axis changes the cue draw "
                  "only, never the substrate/consolidated weights)", n_frozen, expect=lambda x, b=bar: x >= b)
        v.require("distinct read-trial seeds actually VARY the read (sigma_read > 0) -- the anti-cheat that the "
                  "new RNG wiring is load-bearing, not a silent no-op", n_sigma_pos, expect=lambda x, b=bar: x >= b)
        decided = v.decide(go=go, verbose=False)
        summary_extra = dict(GO=go, n_valid=n_valid, status=decided.get("status"),
                             sigma_read_bar=sigma_read_bar, sigma_substrate=sigma_substrate, read_SNR=read_snr,
                             MDG=mdg, K_reads_needed=k_reads_needed, resolve_bar=a.resolve_bar,
                             classification=classification, n_graded=n_graded, n_frozen=n_frozen,
                             n_sigma_pos=n_sigma_pos, preconditions=decided.get("preconditions", []),
                             decided=decided)
    else:
        go = False; n_valid = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_valid=0)

    summary = {"probe": "gap5_read_noise_floor_ltu",
               "mechanism": "DIAGNOSTIC (not a weight lever): population-read SNR at the weak-cue recall "
                            "operating point via a read_trial_seed-keyed cue-subsample + replay-choice RNG axis "
                            "in rest_and_replay, substrate build + OU/membrane-noise seed held fixed",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size, "n_read_trials": a.n_read_trials,
               "cfg": vars(a),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[read-noise-floor] VERDICT: {verdict}\n[read-noise-floor] wrote {a.out}\n"
          + "=" * 120, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
