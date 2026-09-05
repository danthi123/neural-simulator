"""gap#5 learn-through-use, FORWARD-BAND HOMEOSTATIC-SCALING variant (build-ahead, scoped 2026-09-04, NOT YET RUN
past a tiny structural smoke -- see the companion prep doc for why this is queued, not executed, by this pass).

LINEAGE (read this before touching the config -- every hyperparameter below is held IDENTICAL to the established
write on purpose, so this runner isolates exactly ONE new variable):
  1. [[2026-08-27-swr-envelope-learn-through-use-NOGO]] -- the OLD bistable-completion CA3 store never segments
     into discrete replay (co-fire 0.966-0.983); NO op-point on THAT store unblocks learn-through-use. CLOSED.
  2. [[2026-08-27-ecker-adex-store-learn-through-use-NOGO]] -- the Ecker AdEx store DOES segment (forward-from-seed
     0.914), but plain replay-driven STDP consolidation SYMMETRIZES the band (reverse potentiates ~6x forward).
  3. [[2026-08-27-conduction-delay-directional-replay-learn-through-use-PARTIAL]] +
     [[2026-08-27-btsp-directional-write-learn-through-use-PARTIAL]] -- a forward-edge conduction delay
     (Izhikevich 2006 polychronization) + BTSP-eligibility gating makes the write net-DIRECTIONAL 6/6 (was 0/6).
  4. [[2026-08-27-graded-recall-instrument-learn-through-use-NOGO]] -- but a PROVEN-graded recall-depth instrument
     (not the saturating legacy forward_frac) shows weak-cue recall DECREASING after this write on 5/6 seeds. Root
     cause PINNED: the directional write is PURE POTENTIATION -- reverse edges still deepen at ~84% of the forward
     rate (dw_rev/dw_fwd), and that absolute reverse growth intrudes on completion.
  5. [[2026-08-27-reverse-edge-heterosynaptic-depression-learn-through-use-NOGO]] -- suppressing the reverse band
     (Chistiakova-Volgushev heterosynaptic LTD; Kim & Kim 2025 PMID 40623085) ELIMINATES the reverse deepening
     (ratio 0.84->0.074, 6/6) and flips the MEAN recall change from decreasing to increasing (0.586->0.612) -- but
     only 2/6 seeds individually clear the +0.05 gain bar, and driving the depression 20-30x further (fully
     negative reverse growth) does NOT move the other 4 seeds AT ALL. Verdict: reverse-edge growth was A cause for
     2/6 seeds, not THE cause for the rest. That finding names TWO next candidates, verbatim: "characterize the
     READ-side noise floor directly ... and/or investigate whether forward-band ABSOLUTE magnitude at encode time
     (not just its ratio to reverse) is the actual limiting factor for weak-cue completion depth -- a homeostatic
     forward-band STRENGTHENING process (rather than further reverse suppression) is the next candidate lever."
  6. [[2026-08-27-decorrelation-read-shared-fidelity-wall-PARTIAL]] tested a READ-side fix (retina/LGN-style
     cross-assembly common-mode removal) for candidate 1 -- decisive 6-seed NO-GO (subtractive inert-to-harmful,
     divisive worsens recall). "The LTU residual is a separate genuine effect" (not the mouth's stale-COO
     measurement artifact). Candidate 1 (read-side) is NOT closed either, but THIS specific read-side lever failed;
     it does not touch candidate 2 (write-side absolute-magnitude), which has never been attempted for LTU.

THIS RUNNER tests candidate 2, UNTOUCHED BY ANY PRIOR LTU FINDING: does the recall-depth residual trace to the
forward band's own ABSOLUTE magnitude at encode time (observed to vary substantially seed-to-seed, per finding 5's
own "band fwd 145-320" range), rather than only its ratio to reverse? Turrigiano-style multiplicative synaptic
SCALING -- rescale every forward-band synapse by the SAME factor, toward a target the band's own post-encode
activity sets, preserving each synapse's relative weight (the defining property that distinguishes homeostatic
scaling from picking one dominant synapse to boost) -- is the biologically standard mechanism for exactly this
class of correction (Turrigiano GG, Nelson SB. "Homeostatic plasticity in the developing nervous system." Nat Rev
Neurosci. 2004;5(2):97-107, PMID 14735113). The IDENTICAL mechanism CLASS (multiplicative rescale of a population's
own synapses toward a set point computed from the population's own measured state) already ships and is validated
elsewhere in this repo: `webapp/da_encoding_drives_chat.py::apply_substrate_homeostasis` ->
`OneBrainComposer.apply_homeostatic_scaling` (6/6 GO, `_da_encoding_leansoak.py --substrate-scaling`, byte-equal to
a real production build) -- that one targets a per-engram UNIT set-point on the D5 store; this applies the SAME
multiplicative-rescale-toward-a-set-point FORM to the Ecker CA3 forward band's ABSOLUTE magnitude on a DIFFERENT
substrate (the standalone research bridge these _gap5_ derisks build, not OneBrainComposer) -- not a re-derivation
of that GO, a new application of the same accepted mechanism class. NOT a host-invented shortcut: the scale factor
is computed FROM the substrate's OWN measured post-encode weight state (adj_fwd, itself grown by real STDP over
real spikes during `encode`), never from an environment-side oracle.

WHY THIS IS A CHEAP, SURGICAL ADDITION (no new step-loop, unlike findings 3/5 above): the mechanism is a ONE-TIME
discrete rescale of the ALREADY-ENCODED forward-band weights, applied ONCE between `encode` and `consolidate` --
not a per-step update-rule change. So it needs NO new cupy step loop: `apply_forward_band_homeostasis` is pure
host-array arithmetic on the weight vector `encode()` already produced, and the established
`consolidate_by_btsp_replay_delayed` (reuse-by-import, UNCHANGED) runs exactly as before, just starting from a
different initial condition. This keeps the isolation clean: if a gain appears, it is attributable to the STARTING
MAGNITUDE, not to any change in the write rule itself (which is untouched, so directionality is inherited for
free -- this runner does not re-litigate finding 3).

Reuse-by-import (NO `sim/` edit; only ONE new function -- the rescale -- built on top of existing pieces):
  build_store / encode / rest_and_replay / measure_band / measure_band_from / _load_weights
      <- research.runners._gap5_ecker_adex_ca3_stdp_band_derisk / _gap5_ecker_replay_learn_through_use_derisk
  consolidate_by_btsp_replay_delayed <- _gap5_ecker_replay_learn_through_use_derisk (the established DIRECTIONAL
      write, called UNMODIFIED -- this runner never edits the per-step update rule)
  _score_periods_graded / _read_graded / verify_instrument <- _gap5_graded_recall_learn_through_use_derisk (the
      PROVEN-graded depth+tau instrument, unmodified)

THE NEW LEVER: `--fwd-scale-mult` (float, default 1.0 = OFF). RELATIVE, not absolute -- the target for seed s is
`mult * band_before_this_seed`, so one CLI value auto-scales to each seed's own baseline rather than imposing one
hard-coded cross-seed number (baselines already vary ~2x seed-to-seed per finding 5). mult=1.0 is IDENTITY: the
rescale function returns the input array UNCHANGED (no arithmetic at all, not even a `*1.0` -- see
`apply_forward_band_homeostasis`), so the whole pipeline is BYTE-IDENTICAL to the established write's output at the
default -- verified below (`--byte-identical-check`), not asserted from reading the code.

THE ANTI-CHEAT THIS MECHANISM NEEDS THAT NO PRIOR LTU RUNNER NEEDED: because the rescale is a ONE-TIME step BEFORE
consolidation (not gated by replay ignition the way the write-rule changes in findings 3/5 were), the standard
NO-SEED lesion control (seed_on=False during consolidate) does NOT merely null the mechanism the way it did for
the BTSP/hetero-depression writes -- the rescale still happened. That makes NO-SEED here do DOUBLE DUTY as the
decisive control this mechanism specifically needs: "does a statically stronger band read better regardless of
replay" (a bigness effect with NOTHING to do with learn-through-USE) vs "does replay-driven consolidation ON TOP
of a homeostatically-corrected band durably strengthen recall" (genuine use-dependence). GO requires the SEEDED
arm's recall gain to EXCEED the NO-SEED arm's (a `Verdict.control` separation) -- mirroring the two-legs pattern
the D5 organ's own learn-through-use GO used (clamp isolates plateau-vs-cue-current; disjoint-held completion
makes it retrieval-driven) -- so a positive result here cannot be "scaling alone was already enough," only
"scaling made the SAME replay-driven consolidation durably effective."

NOT ADDRESSED BY THIS RUNNER (named, not built here -- see the companion prep doc): finding 5's OTHER named
candidate, "characterize the READ-side noise floor directly (repeated graded-instrument reads of the SAME frozen
weights)". `_read_graded`/`rest_and_replay` derive their cue-subsample and env-choice RNGs FROM the passed
substrate `seed` (see `_gap5_ecker_adex_ca3_stdp_band_derisk.rest_and_replay`), so repeating a read at the SAME
seed on frozen weights is deterministic by this repo's own seeding discipline (`tests/test_determinism.py::
TestSubstrateActuallySeeded`) -- it would trivially read zero noise, not a meaningful diagnostic. A genuine
read-noise-floor probe needs a READ-TRIAL seed independent of the substrate-build seed threaded through
`rest_and_replay`'s cue/choice RNGs -- a small, additive signature change to a function five other findings already
depend on being unchanged. Scoped, not attempted here, to avoid destabilizing that shared surface under a
"don't fully build" mandate; see the prep doc for the exact parameter to add.

Usage:
  Byte-identical-off check: SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk --byte-identical-check --seeds 42
  Cheap single-seed mult scan: SIM_BACKEND=numpy .venv/bin/python -m
      research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk --scan-mult --seeds 42
  TINY structural smoke (this pass only -- confirms import/parse/step, NOT a decisive result):
      SIM_BACKEND=numpy .venv/bin/python -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk \\
          --smoke --seeds 42
  6-seed decisive (QUEUED, not run by this pass -- needs cupy for a real n_mem/asm_size at the established scale):
      SIM_BACKEND=cupy .venv/bin/python -m research.runners._gap5_forward_band_homeostatic_scaling_ltu_derisk \\
          --fwd-scale-mult <chosen> --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
import json
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
    build_store, encode, rest_and_replay, measure_band, _load_weights,
)
from research.runners._gap5_ecker_replay_learn_through_use_derisk import (  # noqa: E402
    measure_band_from, consolidate_by_btsp_replay_delayed,
)
from research.runners._gap5_graded_recall_learn_through_use_derisk import (  # noqa: E402
    _score_periods_graded, _read_graded, verify_instrument,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "forward_band_homeostatic_scaling_ltu.json"
SCAN_OUT = _REPO / "research" / "findings" / "raw" / "gap5_ecker_adex" / "forward_band_homeostatic_scaling_scan.json"


# ----------------------------------------------------------------------------------------------------------------------
# THE NEW LEVER. Pure host-array arithmetic -- no cupy, no step loop, no `sim/` edit. Multiplicatively rescales
# EVERY forward-band synapse by the SAME factor (the defining property of Turrigiano-style scaling: relative
# weights within the band are preserved, unlike hand-picking one synapse to boost). mult<=1.0 is IDENTITY -- for
# mult==1.0 EXACTLY the function returns the input array itself, untouched (no arithmetic performed at all), so
# the byte-identical-off property does not depend on floating-point `x*1.0==x` (true for IEEE754 but this makes
# it true by construction, not by a numerical coincidence -- see byte_identical_check below).
# ----------------------------------------------------------------------------------------------------------------------
def apply_forward_band_homeostasis(w_host, fwd_pos, *, mult=1.0, w_min=0.0, w_max=900.0):
    """Turrigiano-style multiplicative synaptic scaling (Turrigiano & Nelson 2004, Nat Rev Neurosci 5:97, PMID
    14735113) applied ONLY to the CA3 forward between-assembly band, toward `mult * this_seed's_own_adj_fwd` -- a
    RELATIVE target (not one hard-coded absolute number across seeds whose baselines differ ~2x, per
    [[2026-08-27-reverse-edge-heterosynaptic-depression-learn-through-use-NOGO]]). Reads the band's OWN post-encode
    mean weight (grown by real STDP over real spikes during `encode` -- not a host-invented target). Strengthening-
    only by construction (mult is a user-supplied lower-bounded-at-1.0 multiplier, not fit to any observed recall
    outcome). Returns (w_scaled_host, info) with info={'adj_fwd_before','adj_fwd_after','mult_applied'}.
    """
    mult = float(mult)
    band_before = float(np.asarray(w_host)[fwd_pos].mean()) if fwd_pos.size else 0.0
    if mult <= 1.0:
        # IDENTITY: no arithmetic, not even a same-shape copy-and-multiply-by-1.0 -- guarantees byte-identical-off
        # by construction rather than by an IEEE754 property that a future refactor could quietly break.
        return np.asarray(w_host), dict(adj_fwd_before=band_before, adj_fwd_after=band_before, mult_applied=1.0)
    w = np.asarray(w_host, dtype=np.float32).copy()
    w[fwd_pos] = np.clip(w[fwd_pos] * np.float32(mult), np.float32(w_min), np.float32(w_max))
    band_after = float(w[fwd_pos].mean()) if fwd_pos.size else 0.0
    return w, dict(adj_fwd_before=band_before, adj_fwd_after=band_after, mult_applied=mult)


# ----------------------------------------------------------------------------------------------------------------------
# BYTE-IDENTICAL-OFF CHECK (asserted IN THE DATA, per docs/TERMS.md -- a hash/exact compare, not read-the-code).
# mult=1.0 (default) through the FULL pipeline (encode -> rescale(identity) -> consolidate) must reproduce the
# established write's w_after exactly.
# ----------------------------------------------------------------------------------------------------------------------
def byte_identical_check(seed, a):
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)

    st_a = build_store(seed, **bkw); encode(st_a, seed, **enc_kw)
    w_learned = np.asarray(to_host(st_a["bridge"].cp_connections.data)).copy()

    st_old = build_store(seed, **bkw); _load_weights(st_old, w_learned)
    old = consolidate_by_btsp_replay_delayed(st_old, a.consol_steps, seed, seed_on=True,
                                             elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                             eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                             delay_steps=a.fwd_delay_steps, **cons_kw)

    w_scaled, info = apply_forward_band_homeostasis(w_learned, st_a["fwd_pos"], mult=1.0, w_min=0.0,
                                                     w_max=a.btsp_w_max)
    st_new = build_store(seed, **bkw); _load_weights(st_new, w_scaled)
    new = consolidate_by_btsp_replay_delayed(st_new, a.consol_steps, seed, seed_on=True,
                                             elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                             eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                             delay_steps=a.fwd_delay_steps, **cons_kw)
    w_old = np.asarray(old["w_after"]); w_new = np.asarray(new["w_after"])
    h_old = hashlib.sha256(w_old.tobytes()).hexdigest(); h_new = hashlib.sha256(w_new.tobytes()).hexdigest()
    maxdiff = float(np.max(np.abs(w_old - w_new)))
    exact = bool(h_old == h_new)
    print(f"[byte-identical-check] seed={seed} sha256_old={h_old[:16]} sha256_new={h_new[:16]} "
          f"EXACT_HASH_MATCH={exact} max_abs_diff={maxdiff:.3e} mult_applied={info['mult_applied']}", flush=True)
    return dict(seed=seed, sha256_old=h_old, sha256_new=h_new, exact_hash_match=exact, max_abs_diff=maxdiff,
               mult_applied=info["mult_applied"])


# ----------------------------------------------------------------------------------------------------------------------
# CHEAP MULT SCAN (single seed): sweep --fwd-scale-mult, holding everything else at the established write's
# hyperparameters, and report dw_fwd/dw_rev/ratio/adj_fwd-before-after/weak-cue depth_frac gain per point.
# ----------------------------------------------------------------------------------------------------------------------
def scan_mult(seed, a, mults):
    bkw = dict(m_asm=a.n_mem, asm_size=a.asm_size, w_within=a.w_within, between_init=a.between_init,
               within_density=a.within_density, b_override=a.b_override, a_override=None, ou_sigma=a.ou_sigma,
               dt=a.dt, stdp_w_max=a.stdp_w_max, stdp_a_plus=a.stdp_a_plus, stdp_a_minus=a.stdp_a_minus,
               stdp_tau=a.stdp_tau)
    enc_kw = dict(n_laps=a.n_laps, enc_step=a.enc_step, enc_dwell=a.enc_dwell, enc_gap=a.enc_gap,
                  cue_pa=a.enc_cue_pa, cue_frac=a.enc_cue_frac, dt=a.dt)
    cons_kw = dict(swr_period=a.swr_period, cue_pa=a.cue_pa, cue_steps=a.cue_steps, cue_frac=a.cue_frac, dt=a.dt)
    weak_pa = a.cue_pa * a.weak_cue_mult
    read_period = a.read_swr_period if a.read_swr_period > 0 else a.swr_period

    st = build_store(seed, **bkw); encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    rd_weak_before = _read_graded(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_before")
    rows = []
    for mult in mults:
        w_scaled, info = apply_forward_band_homeostasis(w_learned, st["fwd_pos"], mult=mult, w_min=0.0,
                                                         w_max=a.btsp_w_max)
        st_c = build_store(seed, **bkw); _load_weights(st_c, w_scaled)
        cons = consolidate_by_btsp_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                                   elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                   eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                   delay_steps=a.fwd_delay_steps, **cons_kw)
        rd_weak_after = _read_graded(bkw, seed, cons["w_after"], a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                     swr_period=read_period, rest_steps=a.rest_steps, tag=f"weak_after_mult{mult}")
        ratio = cons["dw_rev"] / max(cons["dw_fwd"], 1e-6)
        gain = rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"]
        rows.append(dict(mult=mult, adj_fwd_before=info["adj_fwd_before"], adj_fwd_after=info["adj_fwd_after"],
                         dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"], rev_fwd_ratio=ratio,
                         weak_depth_frac_before=rd_weak_before["depth_frac"],
                         weak_depth_frac_after=rd_weak_after["depth_frac"], depth_gain=gain,
                         weak_tau_before=rd_weak_before["tau"], weak_tau_after=rd_weak_after["tau"]))
        print(f"  [scan] mult={mult:.3f}: adj_fwd {info['adj_fwd_before']:.1f}->{info['adj_fwd_after']:.1f} "
              f"dw_fwd={cons['dw_fwd']:.2f} dw_rev={cons['dw_rev']:.2f} ratio={ratio:.3f} weak depth_frac "
              f"{rd_weak_before['depth_frac']:.3f}->{rd_weak_after['depth_frac']:.3f} (gain {gain:+.3f})", flush=True)
    return dict(seed=seed, weak_depth_frac_before=rd_weak_before["depth_frac"], rows=rows)


# ----------------------------------------------------------------------------------------------------------------------
# DECISIVE PER-SEED TEST: same BUILD/ENCODE/READ scaffold as the graded-recall NO-GO's one_seed(), plus the rescale
# step between ENCODE and CONSOLIDATE. Both the SEEDED and NO-SEED arms consolidate from the SAME rescaled weights
# -- NO-SEED therefore isolates "does the rescale alone (no replay-driven consolidation) already move recall",
# which is the decisive anti-cheat this mechanism (unlike findings 3/5) specifically needs.
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

    st = build_store(seed, **bkw)
    encode(st, seed, **enc_kw)
    w_learned = np.asarray(to_host(st["bridge"].cp_connections.data)).copy()
    band_before_encode = measure_band(st)
    out["band_before_encode"] = band_before_encode
    print(f"  [seed {seed}] ENCODE: band fwd={band_before_encode['adj_fwd']:.1f} "
          f"rev={band_before_encode['adj_rev']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    rd_full_before = _read_graded(bkw, seed, w_learned, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="full_before")
    rd_weak_before = _read_graded(bkw, seed, w_learned, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_before")
    print(f"  [seed {seed}] BEFORE (pre-rescale): full depth_frac={rd_full_before['depth_frac']:.3f} | weak "
          f"depth_frac={rd_weak_before['depth_frac']:.3f} tau={rd_weak_before['tau']:.3f} "
          f"n_multi={rd_weak_before['n_multi']} ({time.time()-t0:.0f}s)", flush=True)

    # THE NEW STEP: one-time homeostatic rescale of the forward band, BEFORE any consolidation.
    w_scaled, scale_info = apply_forward_band_homeostasis(w_learned, st["fwd_pos"], mult=a.fwd_scale_mult,
                                                           w_min=0.0, w_max=a.btsp_w_max)
    out["scale"] = scale_info
    print(f"  [seed {seed}] RESCALE(mult={a.fwd_scale_mult}): adj_fwd {scale_info['adj_fwd_before']:.1f}->"
          f"{scale_info['adj_fwd_after']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    overlap_kw = dict(W=a.window, active_frac=a.active_frac, onset_frac=a.onset_frac)
    st_c = build_store(seed, **bkw)
    _load_weights(st_c, w_scaled)
    cons = consolidate_by_btsp_replay_delayed(st_c, a.consol_steps, seed, seed_on=True,
                                              elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                              eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                              delay_steps=a.fwd_delay_steps, overlap_kw=overlap_kw, **cons_kw)
    w_consol = cons["w_after"]
    rev_fwd_ratio = cons["dw_rev"] / max(cons["dw_fwd"], 1e-6)
    out["consolidate"] = dict(dw_fwd=cons["dw_fwd"], dw_rev=cons["dw_rev"], rev_fwd_ratio=rev_fwd_ratio,
                              volley_overlap=cons.get("volley_overlap"), changed=cons["changed"])
    print(f"  [seed {seed}] CONSOLIDATE(seeded, on rescaled band): dw_fwd={cons['dw_fwd']:.2f} "
          f"dw_rev={cons['dw_rev']:.2f} rev/fwd={rev_fwd_ratio:.3f} volley_overlap={cons.get('volley_overlap')} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rd_full_after = _read_graded(bkw, seed, w_consol, a, cue_pa=a.cue_pa, cue_frac=a.cue_frac,
                                 swr_period=read_period, rest_steps=a.rest_steps, tag="full_after")
    rd_weak_after = _read_graded(bkw, seed, w_consol, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                 swr_period=read_period, rest_steps=a.rest_steps, tag="weak_after")
    band_after = measure_band_from(w_consol, st_c)
    out["band_after"] = band_after
    print(f"  [seed {seed}] AFTER: full depth_frac={rd_full_after['depth_frac']:.3f} | weak depth_frac="
          f"{rd_weak_after['depth_frac']:.3f} tau={rd_weak_after['tau']:.3f} band fwd={band_after['adj_fwd']:.1f} "
          f"rev={band_after['adj_rev']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    # NO-SEED ARM -- double duty (see module docstring): the SAME rescaled weights, but replay never ignites
    # (seed_on=False). This is BOTH the standard lesion-the-replay null AND the decisive "is this a static
    # bigness effect, or does it need genuine replay-driven USE" control this mechanism specifically needs.
    st_n = build_store(seed, **bkw)
    _load_weights(st_n, w_scaled)
    cons_ns = consolidate_by_btsp_replay_delayed(st_n, a.consol_steps, seed, seed_on=False,
                                                 elig_tau_ms=a.btsp_elig_tau, plat_tau_ms=a.btsp_plat_tau,
                                                 eta=a.btsp_eta, w_min=0.0, w_max=a.btsp_w_max,
                                                 delay_steps=a.fwd_delay_steps, **cons_kw)
    w_noseed = cons_ns["w_after"]
    rd_weak_noseed = _read_graded(bkw, seed, w_noseed, a, cue_pa=weak_pa, cue_frac=a.weak_cue_frac,
                                  swr_period=read_period, rest_steps=a.rest_steps, tag="weak_noseed")
    out["no_seed"] = dict(dw_fwd=cons_ns["dw_fwd"], dw_rev=cons_ns["dw_rev"],
                          weak_depth_frac=rd_weak_noseed["depth_frac"], weak_tau=rd_weak_noseed["tau"])
    print(f"  [seed {seed}] NO-SEED (rescaled band, replay never ignites): dw_fwd={cons_ns['dw_fwd']:.3f} weak "
          f"depth_frac={rd_weak_noseed['depth_frac']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["reads"] = dict(full_before=rd_full_before, weak_before=rd_weak_before,
                        full_after=rd_full_after, weak_after=rd_weak_after)

    # ============ PER-SEED VERDICT ============
    dw_fwd = cons["dw_fwd"]; dw_rev = cons["dw_rev"]; dw_ns = cons_ns["dw_fwd"]
    directional = ((dw_fwd - dw_rev) >= a.dw_min)
    headroom = (rd_weak_before["depth_frac"] <= a.headroom_max)
    fwd_raised = ((scale_info["adj_fwd_after"] - scale_info["adj_fwd_before"]) >= a.fwd_raise_min) \
        if a.fwd_scale_mult > 1.0 else True   # a no-op rescale (mult<=1) trivially satisfies "did the knob apply"
    depth_gain = ((rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"]) >= a.depth_gain_min)
    tau_gain = ((rd_weak_after["tau"] - rd_weak_before["tau"]) >= a.tau_gain_min)
    recall_gain = bool(depth_gain or tau_gain)
    noseed_gain = (rd_weak_noseed["depth_frac"] - rd_weak_before["depth_frac"])
    seeded_gain = (rd_weak_after["depth_frac"] - rd_weak_before["depth_frac"])
    use_dependent = bool(seeded_gain > noseed_gain + a.use_dependence_margin)   # the decisive new anti-cheat
    lesion_controlled = (abs(dw_ns) <= a.noseed_max_frac * max(abs(dw_fwd), 1e-6))
    seed_go = bool(directional and headroom and fwd_raised and recall_gain and use_dependent and lesion_controlled)
    out["checks"] = dict(directional=directional, headroom=headroom, fwd_raised=fwd_raised, depth_gain=depth_gain,
                         tau_gain=tau_gain, recall_gain=recall_gain, use_dependent=use_dependent,
                         lesion_controlled=lesion_controlled, dw_fwd=round(dw_fwd, 3), dw_rev=round(dw_rev, 3),
                         rev_fwd_ratio=round(rev_fwd_ratio, 3), dw_noseed=round(dw_ns, 3),
                         weak_depth_frac_before=round(rd_weak_before["depth_frac"], 3),
                         weak_depth_frac_after=round(rd_weak_after["depth_frac"], 3),
                         weak_depth_frac_noseed=round(rd_weak_noseed["depth_frac"], 3),
                         seeded_gain=round(seeded_gain, 3), noseed_gain=round(noseed_gain, 3))
    out["seed_go"] = seed_go
    print(f"  [seed {seed}] => {'GO' if seed_go else 'no'}  checks={out['checks']} ({time.time()-t0:.0f}s)", flush=True)
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
    # the ESTABLISHED directional write (IDENTICAL to the graded-recall NO-GO's decisive 6-seed cfg -- unchanged;
    # this runner does not touch the write rule, only the weight state it starts from)
    ap.add_argument("--btsp-elig-tau", type=float, default=80.0)
    ap.add_argument("--btsp-plat-tau", type=float, default=1.0)
    ap.add_argument("--btsp-eta", type=float, default=0.001)
    ap.add_argument("--btsp-w-max", type=float, default=900.0)
    ap.add_argument("--fwd-delay-steps", type=int, default=90)
    # NEW: the ONLY new lever. 1.0 = OFF = the established write, byte-identical (see apply_forward_band_homeostasis).
    ap.add_argument("--fwd-scale-mult", type=float, default=1.0, help="relative Turrigiano-style rescale target "
                    "for the forward band = mult * this seed's OWN post-encode adj_fwd; <=1.0 is identity (OFF)")
    # ENCODE
    ap.add_argument("--n-laps", type=int, default=14)
    ap.add_argument("--enc-step", type=int, default=80)
    ap.add_argument("--enc-dwell", type=int, default=40)
    ap.add_argument("--enc-gap", type=int, default=600)
    ap.add_argument("--enc-cue-pa", type=float, default=9000.0)
    ap.add_argument("--enc-cue-frac", type=float, default=0.6)
    # SWR replay / prefix seed (write side)
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
    # GO thresholds
    ap.add_argument("--dw-min", type=float, default=5.0)
    ap.add_argument("--headroom-max", type=float, default=0.90)
    ap.add_argument("--fwd-raise-min", type=float, default=10.0, help="minimum adj_fwd rise (absolute weight "
                    "units) required from the rescale when mult > 1.0, for the 'knob actually reached its own "
                    "target variable' check (tools.verdict-style 'reaches')")
    ap.add_argument("--depth-gain-min", type=float, default=0.05)
    ap.add_argument("--tau-gain-min", type=float, default=0.05)
    ap.add_argument("--use-dependence-margin", type=float, default=0.02, help="the SEEDED arm's weak-cue depth "
                    "gain must exceed the NO-SEED arm's by more than this, or the effect is a static bigness "
                    "artifact rather than genuine learn-through-USE")
    ap.add_argument("--noseed-max-frac", type=float, default=0.20)
    # instrument verification (reused unmodified from the graded-recall runner)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-cue-mults", type=float, nargs="+", default=[1.0, 0.85, 0.7, 0.5, 0.35, 0.2])
    ap.add_argument("--verify-min-range", type=float, default=0.15)
    # modes
    ap.add_argument("--byte-identical-check", action="store_true", help="run ONLY the byte-identical-off hash "
                    "comparison (mult=1.0 new pipeline vs the established write), skip everything else")
    ap.add_argument("--scan-mult", action="store_true", help="run ONLY a single-seed fwd-scale-mult scan, skip "
                    "the decisive multi-seed test")
    ap.add_argument("--scan-mults", type=float, nargs="+", default=[1.0, 1.25, 1.5, 2.0, 3.0])
    ap.add_argument("--smoke", action="store_true", help="TINY structural smoke: shrink n_mem/asm_size/steps to "
                    "confirm import/parse/one-step-executes; NOT a decisive result. Use before ever queuing the "
                    "full run.")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--scan-out", default=str(SCAN_OUT))
    a = ap.parse_args()

    if a.smoke:
        # Shrink everything by ~10-20x so a "does this run at all" check completes in seconds on numpy/CPU,
        # per this pass's mandate: confirm imports/parses/starts a step, do NOT run the decisive de-risk.
        a.n_mem = 3; a.asm_size = 12; a.n_laps = 2; a.enc_step = 20; a.enc_dwell = 10; a.enc_gap = 40
        a.rest_steps = 300; a.consol_steps = 300; a.swr_period = 80; a.cue_steps = 10
        a.verify_cue_mults = [1.0, 0.5]; a.skip_verify = True
        if a.fwd_scale_mult <= 1.0:
            a.fwd_scale_mult = 1.5   # exercise the NEW code path (not just the byte-identical-off default)

    _, backend = get_backend()
    print(f"[fwdband-ltu] Ecker AdEx CA3 FORWARD-BAND HOMEOSTATIC-SCALING learn-through-use | "
          f"write=btsp+delay(UNCHANGED) elig_tau={a.btsp_elig_tau} plat_tau={a.btsp_plat_tau} eta={a.btsp_eta} "
          f"fwd_delay={a.fwd_delay_steps}steps fwd_scale_mult={a.fwd_scale_mult} | n_mem={a.n_mem} "
          f"asm={a.asm_size} | swr={a.swr_period} seeds={a.seeds} backend={backend} smoke={a.smoke}", flush=True)

    if a.byte_identical_check:
        rows = [byte_identical_check(s, a) for s in a.seeds]
        all_exact = all(r["exact_hash_match"] for r in rows)
        print(f"[fwdband-ltu] BYTE-IDENTICAL-OFF: {'CONFIRMED (exact hash match)' if all_exact else 'NOT exact -- see max_abs_diff'} "
              f"on {sum(r['exact_hash_match'] for r in rows)}/{len(rows)} seeds", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(str(a.out) + ".byte_identical_check.json").write_text(json.dumps(rows, indent=2, default=str))
        return 0 if all_exact else 1

    if a.scan_mult:
        result = scan_mult(a.seeds[0], a, a.scan_mults)
        Path(a.scan_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.scan_out).write_text(json.dumps(dict(seeds=a.seeds, cfg=vars(a), **result), indent=2, default=str))
        print(f"[fwdband-ltu] wrote {a.scan_out}", flush=True)
        return 0

    verify = None
    if not a.skip_verify:
        verify = verify_instrument(a.seeds[0], a)
        if not verify["graded"]:
            print("[fwdband-ltu] instrument validation failed (unexpected -- reused unmodified from the graded-"
                  "recall runner); aborting.", flush=True)
            return 1

    t0 = time.time(); per = []; err = None
    try:
        for s in a.seeds:
            per.append(one_seed(s, a))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if a.smoke:
        # SMOKE MODE: report structural success (ran to completion, produced the expected keys) and STOP -- this
        # pass's mandate is "confirm it imports/parses/starts a step", not a verdict. No GO/NO-GO is computed or
        # written as a decisive result at smoke scale (tiny n_mem/asm_size would make any verdict meaningless).
        ok = (err is None and len(per) == len(a.seeds)
              and all("seed_go" in p and "checks" in p and "scale" in p for p in per))
        summary = {"probe": "gap5_forward_band_homeostatic_scaling_ltu", "mode": "SMOKE_STRUCTURAL_ONLY",
                  "smoke_ok": ok, "error": err, "seeds": a.seeds, "cfg": vars(a), "per_seed": per,
                  "elapsed_seconds": round(time.time() - t0, 1),
                  "note": "structural smoke only -- tiny n_mem/asm_size/steps; NOT a decisive de-risk result. "
                          "Run without --smoke (cupy, established scale, 6 seeds) for the real verdict."}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        smoke_out = str(a.out).replace(".json", "_smoke.json")
        Path(smoke_out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"\n[fwdband-ltu] SMOKE {'OK' if ok else 'FAILED'} -- wrote {smoke_out}", flush=True)
        return 0 if ok else 1

    if err is None and per:
        n_go = sum(1 for p in per if p.get("seed_go"))
        bar = max(1, (len(per) + 1) // 2) if len(per) < 6 else 5
        go = n_go >= bar
        mdwf = float(np.mean([p["consolidate"]["dw_fwd"] for p in per]))
        mdwr = float(np.mean([p["consolidate"]["dw_rev"] for p in per]))
        mratio = float(np.mean([p["consolidate"]["rev_fwd_ratio"] for p in per]))
        mdwns = float(np.mean([p["no_seed"]["dw_fwd"] for p in per]))
        mwdf_b = float(np.mean([p["reads"]["weak_before"]["depth_frac"] for p in per]))
        mwdf_a = float(np.mean([p["reads"]["weak_after"]["depth_frac"] for p in per]))
        mwdf_ns = float(np.mean([p["no_seed"]["weak_depth_frac"] for p in per]))
        mwtau_b = float(np.mean([p["reads"]["weak_before"]["tau"] for p in per]))
        mwtau_a = float(np.mean([p["reads"]["weak_after"]["tau"] for p in per]))
        mfwd_b = float(np.mean([p["scale"]["adj_fwd_before"] for p in per]))
        mfwd_a = float(np.mean([p["scale"]["adj_fwd_after"] for p in per]))
        n_headroom = sum(1 for p in per if p["checks"]["headroom"])
        n_directional = sum(1 for p in per if p["checks"]["directional"])
        n_fwd_raised = sum(1 for p in per if p["checks"]["fwd_raised"])
        n_use_dependent = sum(1 for p in per if p["checks"]["use_dependent"])
        n_lesion_ok = sum(1 for p in per if p["checks"]["lesion_controlled"])
        if go:
            verdict = (f"FORWARD-BAND HOMEOSTATIC-SCALING GO {n_go}/{len(per)} (mult={a.fwd_scale_mult}) -- the "
                       f"rescale raises adj_fwd {mfwd_b:.1f}->{mfwd_a:.1f} ({n_fwd_raised}/{len(per)}), the write "
                       f"stays directional (dw_fwd {mdwf:.1f} vs dw_rev {mdwr:.1f}, {n_directional}/{len(per)}), "
                       f"weak-cue recall GAINS: depth_frac {mwdf_b:.3f}->{mwdf_a:.3f} (tau {mwtau_b:.3f}->"
                       f"{mwtau_a:.3f}), AND the gain is USE-DEPENDENT not a static bigness artifact (seeded > "
                       f"no-seed on {n_use_dependent}/{len(per)}; no-seed weak depth_frac {mwdf_ns:.3f} vs before "
                       f"{mwdf_b:.3f}), headroom held {n_headroom}/{len(per)}, lesion-null {n_lesion_ok}/{len(per)} "
                       f"(dw_fwd_noseed {mdwns:.2f}~0). => converts the graded-recall NO-GO via forward-band "
                       f"absolute-magnitude homeostasis, the named next mechanism after reverse-edge depression.")
        else:
            verdict = (f"FORWARD-BAND HOMEOSTATIC-SCALING NO-GO {n_go}/{len(per)} (mult={a.fwd_scale_mult}) -- "
                       f"adj_fwd raised {mfwd_b:.1f}->{mfwd_a:.1f} ({n_fwd_raised}/{len(per)}), directional "
                       f"{n_directional}/{len(per)}, headroom {n_headroom}/{len(per)} (weak depth_frac_before "
                       f"{mwdf_b:.3f}), but recall depth_frac {mwdf_b:.3f}->{mwdf_a:.3f} (tau {mwtau_b:.3f}->"
                       f"{mwtau_a:.3f}) does not clear the gain bar with use-dependence on enough seeds "
                       f"(use_dependent {n_use_dependent}/{len(per)}, lesion-null {n_lesion_ok}/{len(per)}). => a "
                       f"GENUINE negative for absolute-magnitude forward-band homeostasis at this mult on this "
                       f"substrate; the residual is NOT (solely) forward-band magnitude either -- redirect to the "
                       f"read-side noise-floor probe (finding 5's other named candidate, not built by this runner).")
        v = Verdict("Ecker AdEx CA3: does Turrigiano-style forward-band homeostatic scaling (absolute magnitude, "
                    "not reverse suppression) restore a weak-cue GRADED recall GAIN that pure-potentiation BTSP "
                    "and reverse-edge heterosynaptic depression could not fully restore?")
        v.require("the GRADED instrument reads graded on a known-good store (pre-flight verify_instrument, reused)",
                  bool(verify is None or verify["graded"]), expect=True)
        v.require("weak-cue depth_frac BEFORE has headroom (not at ceiling) on >= bar seeds", n_headroom,
                  expect=lambda x, b=bar: x >= b)
        v.require("the write is DIRECTIONAL (dw_fwd > dw_rev + dw_min) on >= bar seeds -- inherited from the "
                  "UNCHANGED established write, not re-tested by this mechanism", n_directional,
                  expect=lambda x, b=bar: x >= b)
        v.reaches("the rescale knob actually reaches its own target variable (adj_fwd rises when mult > 1.0)",
                  before=mfwd_b, after=mfwd_a)
        v.require("adj_fwd rise clears --fwd-raise-min on >= bar seeds", n_fwd_raised,
                  expect=lambda x, b=bar: x >= b)
        v.control("USE-DEPENDENCE: SEEDED (rescale + replay-driven consolidation) weak-cue depth gain vs NO-SEED "
                  "(rescale alone, replay never ignites) -- must show the SEEDED arm ahead, or this is a static "
                  "bigness effect having nothing to do with learn-through-USE",
                  treatment=(mwdf_a - mwdf_b), control=(mwdf_ns - mwdf_b), min_separation=a.use_dependence_margin)
        v.disabled("within-assembly recurrence + assembly identity + the per-step write rule itself; ONLY the "
                  "PRE-CONSOLIDATION weight magnitude changes (same scope as the established write's own scaffold)",
                  why="scope: isolates ONE variable (starting forward-band magnitude) by construction -- the "
                      "consolidate step reuses consolidate_by_btsp_replay_delayed UNMODIFIED")
        decided = v.decide(go=go, verbose=False)
        attributable_to("weak-cue depth_frac gain (seeded rescaled-replay vs NO-SEED rescaled-no-replay)",
                        mwdf_a - mwdf_b, mwdf_ns - mwdf_b)
        summary_extra = dict(GO=go, n_go=n_go, status=decided.get("status"), fwd_scale_mult=a.fwd_scale_mult,
                             dw_fwd=mdwf, dw_rev=mdwr, rev_fwd_ratio=mratio, dw_fwd_noseed=mdwns,
                             adj_fwd_before=mfwd_b, adj_fwd_after=mfwd_a,
                             weak_depth_frac_before=mwdf_b, weak_depth_frac_after=mwdf_a,
                             weak_depth_frac_noseed=mwdf_ns, weak_tau_before=mwtau_b, weak_tau_after=mwtau_a,
                             n_headroom=n_headroom, n_directional=n_directional, n_fwd_raised=n_fwd_raised,
                             n_use_dependent=n_use_dependent, n_lesion_ok=n_lesion_ok, instrument_verify=verify,
                             preconditions=decided.get("preconditions", []), decided=decided)
    else:
        go = False; n_go = 0
        verdict = f"ERROR -- {err}" if err else "NO RESULTS"
        summary_extra = dict(GO=False, n_go=0, instrument_verify=verify)

    summary = {"probe": "gap5_forward_band_homeostatic_scaling_ltu",
               "mechanism": "Turrigiano-style multiplicative synaptic scaling of the CA3 FORWARD band toward an "
                            "ABSOLUTE (per-seed-relative) magnitude target, applied ONCE between encode and "
                            "consolidation, on top of the established BTSP+forward-conduction-delay directional "
                            "write (unmodified)",
               "seeds": a.seeds, "n_mem": a.n_mem, "asm_size": a.asm_size,
               "cfg": vars(a),
               "elapsed_seconds": round(time.time() - t0, 1), "verdict": verdict, "per_seed": per, **summary_extra}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 120 + f"\n[fwdband-ltu] VERDICT: {verdict}\n[fwdband-ltu] wrote {a.out}\n" + "=" * 120,
          flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
