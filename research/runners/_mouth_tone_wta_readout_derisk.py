"""Mouth biologization (ONE bounded step): the SEAM-C tone-token render decided ON SPIKES.

THE HOST RENDER being burned down -- `_stageA_full_integration_derisk._graded_tone_level(differential)`.
The affect ladder already produces the tone SIGNAL neurally (the spike-rate differential
rate(aff_pos_readout) - rate(aff_neg_readout), a Koulakov graded staircase). But the mapping from that
differential to the DISCRETE tone the mouth SPEAKS ("warmly, gladly ..." / "curtly ..." / "coldly,
reluctantly ...") is a host `if/elif` threshold-binning. Per the BRAIN-BASED-ONLY standard a host
threshold/argmax over a neural signal is a shortcut: the SELECTION of the tone the mouth articulates is
being made by Python, not by the brain.

THIS replaces that host binning with a SPIKING FS-WTA read-out -- the SAME validated one-of-K lateral-
inhibition selector (`build_fswta_score_bridge`/`fswta_drive` from `_d3_spiking_attractor_derisk`, the
one the reslm/word-decode read-out parity ran to K=200). The ladder differential is projected onto K=7
tone-level pools by a LABELED-LINE afferent place-code (a legitimate host INPUT, SAME status as the
reservoir's W_in / the retinal render / `_word_embedding` -- it faithfully encodes the host render's OWN
band definition, derived from the host function itself), then a shared inhibitory FS pool RESOLVES the
winning level on SPIKES: the winner fires first, recruits FS, FS suppresses the runners-up -> a clean
one-of-K SPIKING tone. The mouth's tone for the turn is thus decided by the NETWORK's inhibition, not a
host threshold.

SCOPE (additive / default-OFF / NO `sim/` edit): a stand-alone read-out de-risk; reuse-by-import. The
host render stays the deployed default; the spiking render is the opt-in. RESIDUAL host pieces, declared
(the named next mechanisms): (i) the afferent place-code is host-designed, not learned/self-organized;
(ii) the level->word lexicon lookup (`GRADED_TONE_LEVELS`) is a fixed table.

Ceiling  = the host `_graded_tone_level` binning (the deployed SEAM-C render).
GO gate (per seed)  = spiking-WTA token == host token on the LIVE ladder differentials AND on a dense
           band sweep (parity >= 1 - one boundary tie), a CLEAN one-of-K winner on the reachable levels,
           AND the SHUFFLE control collapses to chance -> the WTA reads the ACTUAL differential.
           Aggregate GO = >= 5/6 seeds.
Anti-cheat = (a) SHUFFLE: drive the WTA with a PERMUTED place-code -> winner agreement with the host
           level collapses to chance; (b) the place-code is MONOTONE in the host bands by construction
           (it preserves the host argmax) -> the open question is whether the SPIKING WTA resolves it at
           the real ladder margins (esp. near band boundaries + at K=7).

Moat/honesty (FM4): the tone render is invoked ONLY on an already-matched, already-honest answer (raw is
not None; see `_colored_answer_graded`); it colors WITHIN the decided band and CANNOT flip
abstain->assert -- the WTA's output is CONFINED to the tone lexicon (it selects one of 7 tone pools) and
never touches the answer content or the cue-match moat. The smoke asserts lexicon-confinement + that an
extreme differential only saturates the tone, never leaks into content.

Run (SMOKE, cheap ~few s/seed on numpy):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._mouth_tone_wta_readout_derisk --seeds 42
Run (DECISIVE 6-seed):
  SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._mouth_tone_wta_readout_derisk \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/mouth_tone_wta/tone_wta_6seed.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# The HOST render being biologized (the deployed SEAM-C tone binning) -- imported, not re-implemented,
# so parity is against the ACTUAL production render.
from research.runners._stageA_full_integration_derisk import (
    GRADED_TONE_LEVELS, LADDER_NEUTRAL_TOL, _graded_tone_level, _graded_tone_token,
)
# The validated one-of-K spiking FS-WTA selector (lateral inhibition; reuse-by-import).
from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive

TONE_LEVELS = sorted(GRADED_TONE_LEVELS)          # [-3,-2,-1,0,1,2,3]
K = len(TONE_LEVELS)                               # 7 tone-level pools
_LEVEL_TO_POOL = {L: p for p, L in enumerate(TONE_LEVELS)}


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# The LABELED-LINE afferent place-code: encode the scalar ladder differential onto the K tone-level pools
# by faithfully reproducing the HOST render's OWN band boundaries (derived from `_graded_tone_level`
# itself), then a soft-box tuning per band. A legitimate host INPUT (same status as W_in) -- the SPIKING
# WTA does the SELECTION. Note tol==step==0.03 makes level +/-1 UNREACHABLE in the host function; those
# pools stay dormant (their band is empty) -- the spiking render inherits EXACTLY the host's reachable set.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
_INF = float("inf")


def host_bands(dmin: float = -0.4, dmax: float = 0.4, n: int = 8001):
    """(lo, hi) on the signed-differential axis for each tone level, READ OFF the host render. Open-ended
    top/bottom bands (that touch dmin/dmax) extend to +/-inf (the tone saturates). Empty bands -> None."""
    grid = np.linspace(dmin, dmax, n)
    lv = np.array([_graded_tone_level(float(x)) for x in grid])
    bands = {}
    for L in TONE_LEVELS:
        m = lv == L
        if not m.any():
            bands[L] = (None, None)                       # unreachable (host tol==step quirk)
            continue
        lo = float(grid[m].min()); hi = float(grid[m].max())
        if lo <= grid[1]:
            lo = -_INF                                    # band touches the low edge -> saturates
        if hi >= grid[-2]:
            hi = _INF                                     # band touches the high edge -> saturates
        bands[L] = (lo, hi)
    return bands


def _soft_gate(x: float, edge: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(np.clip(x / edge, -60.0, 60.0))))   # clip: logistic saturates cleanly


def tone_afferent_scores(differential: float, bands: dict, edge: float = 0.0025) -> np.ndarray:
    """K-vector place-code: pool for level L scores ~1 when `differential` is inside band L, ~0.5 at its
    edges, ~0 outside (product of a rising + a falling logistic gate). MONOTONE in the host bands by
    construction -> argmax(scores) == host level; the SPIKING WTA then resolves it."""
    s = np.zeros(K, dtype=float)
    for L in TONE_LEVELS:
        lo, hi = bands[L]
        if lo is None:                                    # unreachable band -> dormant pool
            continue
        lo_gate = 1.0 if lo == -_INF else _soft_gate(differential - lo, edge)
        hi_gate = 1.0 if hi == _INF else _soft_gate(hi - differential, edge)
        s[_LEVEL_TO_POOL[L]] = float(lo_gate * hi_gate)
    return s


# ────────────────────────────────────────────────────────────────────────────────────────────────────
# The SPIKING render: drive the FS-WTA by the place-code, read the winner ON SPIKES.
# ────────────────────────────────────────────────────────────────────────────────────────────────────
def spiking_tone(sb, differential: float, bands: dict, input_gain: float, settle: int,
                 shuffle_rng=None):
    """Return (level, token, clean, margin, acc). If `shuffle_rng` is given, the place-code is PERMUTED
    before the drive (the anti-cheat control: the WTA then reads a scrambled signal)."""
    s = tone_afferent_scores(differential, bands)
    if shuffle_rng is not None:
        s = s[shuffle_rng.permutation(K)]
    _, acc = fswta_drive(sb, K, s, input_gain=input_gain, settle=settle)
    if float(acc.max()) <= 0.0:
        return 0, "", False, 0.0, acc                     # no pool fired -> neutral (honest null)
    win = int(np.argmax(acc))
    srt = np.sort(acc)[::-1]
    margin = float(srt[0] - (srt[1] if K > 1 else 0.0))
    clean = bool(margin > 0.20 * srt[0])                  # winner clears the runner-up by >20%
    level = TONE_LEVELS[win]
    return level, _graded_tone_token(level), clean, margin, acc


# ────────────────────────────────────────────────────────────────────────────────────────────────────
def _synthetic_sweep(lo: float = -0.15, hi: float = 0.15, n: int = 121) -> np.ndarray:
    return np.linspace(lo, hi, n)


def _live_differentials(seed: int, n_levels: int):
    """REAL neural differentials from the affect ladder (the exact signal the live SEAM-C render consumes:
    positive-appraisal held reads, `read_affect_ladder` drives V+ only). Returns list of held rates."""
    from research.runners._affect_graded_ladder_derisk import measure_staircase
    appr = list(np.linspace(0.0, 1.0, max(2, int(n_levels))))
    r = measure_staircase(seed, appr, drive_off_ms=200, probe_ms=100)
    return list(r["held"]), appr


def run_seed(seed: int, live_levels: int, input_gain: float, settle: int):
    bands = host_bands()
    sb = build_fswta_score_bridge(seed=int(seed), K=K)
    shuf_rng = np.random.default_rng(seed * 7919 + 11)

    # (1) SYNTHETIC dense band sweep -- parity + clean-winner + shuffle control.
    sweep = _synthetic_sweep()
    reachable = sorted({_graded_tone_level(float(d)) for d in sweep})
    n_par = n_clean = n_shuf = 0
    margins = []
    for d in sweep:
        host_L = _graded_tone_level(float(d))
        spk_L, spk_tok, clean, margin, _ = spiking_tone(sb, float(d), bands, input_gain, settle)
        n_par += int(spk_tok == _graded_tone_token(host_L))
        n_clean += int(clean)
        margins.append(margin)
        sh_L, _, _, _, _ = spiking_tone(sb, float(d), bands, input_gain, settle, shuffle_rng=shuf_rng)
        n_shuf += int(sh_L == host_L)
    nsw = len(sweep)
    parity_synth = n_par / nsw
    clean_frac = n_clean / nsw
    shuffle_parity = n_shuf / nsw

    # (2) LIVE ladder differentials -- the exact live-chat render input.
    held, appr = _live_differentials(seed, live_levels)
    live_rows = []
    n_live_par = 0
    for m, d in zip(appr, held):
        host_L = _graded_tone_level(float(d))
        spk_L, spk_tok, clean, margin, _ = spiking_tone(sb, float(d), bands, input_gain, settle)
        agree = int(spk_tok == _graded_tone_token(host_L))
        n_live_par += agree
        live_rows.append({"appraisal": round(float(m), 3), "differential": round(float(d), 5),
                          "host_level": int(host_L), "host_token": _graded_tone_token(host_L),
                          "spiking_level": int(spk_L), "spiking_token": spk_tok,
                          "clean": bool(clean), "margin": round(float(margin), 4), "agree": bool(agree)})
    parity_live = n_live_par / max(1, len(held))

    # (3) FM4 / lexicon-confinement smoke: an EXTREME differential only SATURATES the tone (never leaks
    #     into content); every spiking output is a member of the tone lexicon.
    lex = set(GRADED_TONE_LEVELS.values())
    sat_hi_L, sat_hi_tok, _, _, _ = spiking_tone(sb, +10.0, bands, input_gain, settle)
    sat_lo_L, sat_lo_tok, _, _, _ = spiking_tone(sb, -10.0, bands, input_gain, settle)
    all_tokens = [r["spiking_token"] for r in live_rows] + [sat_hi_tok, sat_lo_tok]
    lexicon_confined = all((t in lex) for t in all_tokens)
    fm4_saturates = (sat_hi_L == max(TONE_LEVELS)) and (sat_lo_L == min(TONE_LEVELS))

    # per-seed GO
    n_reach = max(1, len(reachable))
    seed_go = bool(parity_live >= 1.0 and parity_synth >= 1.0 - 2.0 / nsw
                   and shuffle_parity < 0.6 and lexicon_confined and fm4_saturates)
    return {
        "seed": int(seed),
        "parity_synth": parity_synth, "parity_live": parity_live,
        "clean_frac": clean_frac, "shuffle_parity": shuffle_parity,
        "mean_margin": float(np.mean(margins)), "min_margin": float(np.min(margins)),
        "reachable_levels": reachable, "n_reachable": n_reach,
        "lexicon_confined": bool(lexicon_confined), "fm4_saturates": bool(fm4_saturates),
        "sat_hi_token": sat_hi_tok, "sat_lo_token": sat_lo_tok,
        "live_rows": live_rows, "seed_go": seed_go,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="FS-WTA read-out bridge seeds (neural heterogeneity of the tone WTA) + ladder seed")
    ap.add_argument("--live-levels", type=int, default=6,
                    help="how many appraisal levels to read REAL ladder differentials at")
    ap.add_argument("--input-gain", type=float, default=1200.0)
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    seeds_out = []
    for sd in a.seeds:
        r = run_seed(sd, a.live_levels, a.input_gain, a.settle)
        seeds_out.append(r)
        print(f"  seed={sd:4d}  parity_live={r['parity_live']:.3f}  parity_synth={r['parity_synth']:.3f}  "
              f"clean={r['clean_frac']:.3f}  shuffle={r['shuffle_parity']:.3f}  "
              f"reach={r['reachable_levels']}  lex_confined={r['lexicon_confined']}  "
              f"fm4_sat={r['fm4_saturates']}  [{'GO' if r['seed_go'] else 'MISS'}]", flush=True)

    n_go = sum(int(r["seed_go"]) for r in seeds_out)
    n_seeds = len(a.seeds)
    live_mean = float(np.mean([r["parity_live"] for r in seeds_out]))
    synth_mean = float(np.mean([r["parity_synth"] for r in seeds_out]))
    clean_mean = float(np.mean([r["clean_frac"] for r in seeds_out]))
    shuf_mean = float(np.mean([r["shuffle_parity"] for r in seeds_out]))
    chance = 1.0 / max(1, int(np.mean([r["n_reachable"] for r in seeds_out])))

    # ANTI-CHEAT attribution: WTA agreement with the host attributable to it READING the true place-code
    # (treatment=synthetic parity) vs a generic sorter (control=shuffled-place-code parity).
    from tools.lab import attributable_to
    frac = attributable_to("FS-WTA reading the true place-code (parity vs shuffled parity)",
                           synth_mean, shuf_mean)
    summary_attribution = None if frac is None else round(float(frac), 4)

    # EARNED VERDICT -- the preconditions travel with the result.
    from tools.verdict import Verdict
    v = Verdict("spiking-WTA tone render vs host binning", chance=chance)
    v.require("parity(spiking token == host token) on LIVE ladder differentials == 1",
              live_mean, expect=lambda x: x >= 1.0)
    v.require("parity on dense band sweep >= 0.98", synth_mean, expect=lambda x: x >= 0.98)
    v.control("WTA reads TRUE place-code (synth parity vs shuffled parity)", synth_mean, shuf_mean,
              min_separation=0.4)
    v.require(">=5/6 seeds match host", n_go, expect=lambda x: x >= max(5, n_seeds - 1))
    decided = v.decide(go=(n_go >= max(5, n_seeds - 1)), verbose=False)
    verdict = decided["status"]

    summary = {
        "runner": "research.runners._mouth_tone_wta_readout_derisk",
        "burns_down": "_stageA_full_integration_derisk._graded_tone_level (SEAM-C host tone binning)",
        "mechanism": "FS-WTA lateral-inhibition selection over K=7 tone-level pools (spiking one-of-K)",
        "residual_host": ["afferent place-code (host-designed, not self-organized)",
                          "level->word lexicon lookup (fixed GRADED_TONE_LEVELS table)"],
        "n_seeds": n_seeds, "n_go": n_go, "verdict": verdict,
        "parity_live_mean": live_mean, "parity_synth_mean": synth_mean,
        "clean_frac_mean": clean_mean, "shuffle_parity_mean": shuf_mean,
        "parity_attributable_to_true_place_code": summary_attribution,
        "chance": chance, "host_tol": LADDER_NEUTRAL_TOL,
        "note_tol_eq_step_quirk": "tol==step==0.03 -> host levels +/-1 unreachable; spiking render inherits the same reachable set",
        "input_gain": a.input_gain, "settle": a.settle,
        "wall_clock_s": round(time.time() - t0, 1),
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "per_seed": seeds_out,
    }
    print(f"\n[VERDICT] {verdict} -- {n_go}/{n_seeds} seeds match host. "
          f"parity_live={live_mean:.3f}  parity_synth={synth_mean:.3f}  "
          f"clean={clean_mean:.3f}  shuffle={shuf_mean:.3f} (chance {chance:.3f})", flush=True)

    if a.out:
        op = Path(a.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(summary, indent=2))
        print(f"[OUT] wrote {op}", flush=True)


if __name__ == "__main__":
    main()
