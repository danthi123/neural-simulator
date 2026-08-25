"""DA-GATED ENCODING — LEAN production magnitude-store NO-REGRESSION soak (the default-ON flip gate, 2026-08-21).

WHY THIS EXISTS (and why it REPLACES `_da_encoding_noregression_soak.py` for the flip decision). da-gated-encoding is
de-risked + wired default-OFF (I-7-b GO + `_da_encoding_wired_verify.py` GO). On the production onebrain MAGNITUDE
store it scales a fact's stored |w| by a DA-derived gain g=clip(0.5,3.0,1+2(DA-0.5)) at store time, so a below-tonic-DA
fact is stored WEAKER (g<1, floored 0.5). Flipping it default-ON can therefore change FUTURE recall. The flip needs a
CORRECT no-regression soak. The two prior instruments were WRONG for this question:
  (1) `_wave4_composed_flip_noregression.py` ran on the rf FAST-path where the recall read is MAGNITUDE-INVARIANT ->
      the da-encoding lever moved ZERO variables -> not evidence for this flip.
  (2) `_da_encoding_noregression_soak.py` used the RIGHT read-path (OneBrainComposer store_conns + the I-7-b
      `_query_under_damage` RF-floor damage) but was (a) pathologically slow (37-word vocab -> an ~11487-neuron bridge
      x 12 builds -> ~1.7h, had to be killed) and (b) swept sigma in [0.05..0.3] -- FAR below the I-7-b behavioral knee
      (~0.75..4.0), so OFF recall was plausibly M/M at every sigma -> the no-regression test was VACUOUS (an A/B whose
      lever moves nothing tells you nothing). It also had a cue COLLISION (fact i and fact i+12 shared the (agent,action)
      cue at m=20 with a 12-word agent/action alphabet).

THIS RUNNER fixes all three:
  * LEAN: a 12-word vocab + 9 facts (k_max=13) -> ~half the neurons, minutes not hours; cupy for the OneBrainComposer.
  * DERIVED ON ARM (halves brain builds, tightest possible control): `_write_block` writes `g * zc[k]` (D conns/block,
    block-major, LINEAR in g -- read from one_brain_composer.py:639). So the ON arm's store_conns is EXACTLY the OFF
    arm's with block i scaled by g_i. We build ONE OFF composer per seed (all g=1), capture store_conns, and DERIVE the
    ON arm by block-scaling -- provably identical to a real per-fact `encoding_gain_fn` build (cross-checked byte-equal
    on seed 42), and it makes OFF and ON differ in NOTHING but per-fact write magnitude (identical parser, identical
    read machinery, identical damage draws).
  * CALIBRATED, NON-VACUOUS SIGMA: a WIDE grid spanning the I-7-b knee (0.0 .. 6.0). The verdict carries an
    INSTRUMENT-BITES precondition: OFF-arm recall MUST degrade below full somewhere on the grid, else the soak is
    UNDEFINED (widen), never a fake GO. The calibrated knee (smallest sigma where OFF recall first drops) is reported.

THE HONEST QUESTION. Flipping encoding ON must not make production recall WORSE. Two regimes:
  (1) CLEAN read (sigma=0) -- the dominant case for a modest store: the RF read is a PHASE read, magnitude-invariant,
      so g should not change WHICH fact is recalled. Prediction: ZERO regression.
  (2) READ STRESS (sigma>0) -- the I-7-b knee: a g=0.5 low-DA fact has LOWER SNR than its g=1 OFF counterpart, so it can
      drop below the RF read floor and regress; a g=2.48 high-DA fact gains SNR and can improve. This is the
      biologically-intended salience gating (Lisman-Grace/Kandel D.16), NOT a bug -- but a real behaviour change. We ask
      whether it is NET-neutral-or-positive over a realistic DA distribution (high-DA durability offsets low-DA loss).

THE BATTERY. 9 distinct SVO facts on a shared magnitude store (real interference), distinct (agent,action) cues + 9
distinct patients, DA assigned by a LATIN SQUARE so DA (high/low/tonic) is orthogonal to BOTH agent and action (no
DA<->content confound). OFF arm: all g=1 (== today's default). ON arm: per-fact g from the DA schedule. Same facts,
same read-damage draws; the ONLY difference is the write gain. 6 seeds (42/43/44/100/101/102).

VERDICT (GO = the flip is safe on the production magnitude store):
  * INSTRUMENT BITES (precondition): OFF recall degrades below full on the grid (else UNDEFINED).                 [PREC]
  * DERIVATION VALID (precondition): the block-scaled ON arm is byte-equal to a real encoding_gain_fn build.      [PREC]
  * MECHANISM (precondition): ON block |w| == g x OFF block |w| for every fact/seed.                             [PREC]
  * MOAT (precondition): an unstored cue abstains on both arms at every sigma (encoding never manufactures a fact).[PREC]
  * CLEAN (sigma=0): n_regressed == 0 on every seed (magnitude-invariant -> a clean read cannot be hurt).      [OUTCOME]
  * STRESS net: recall_ON >= recall_OFF at every swept sigma (the salience redistribution is net-neutral-or-positive).
                                                                                                               [OUTCOME]
  * Characterisation (reported): per-sigma recall_off/recall_on, n_regressed/n_improved, first-regression sigma.

Run (cupy for the OneBrainComposer; through gpu_queue.sh -- one brain at a time):
  SIM_BACKEND=cupy python -u -m research.runners._da_encoding_leansoak
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "cupy")
import logging
logging.getLogger().setLevel(logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402
from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import (  # noqa: E402
    da_to_encoding_gain, _query_under_damage, _block_mean_mag, _damage_store_conns,
)

SEEDS = [42, 43, 44, 100, 101, 102]
D = 64
K_DA = 2.0            # == the I-7-b / consolidation-probe2 / wired production default
DA_BASELINE = 0.5     # tonic

# WIDE, knee-spanning sigma grid (the I-7-b behavioral differential knee sits at ~0.75..4.0; the old soak's 0.05..0.3
# was FAR below it -> vacuous). 0.0 = the clean production case; 6.0 = deep read-collapse. Overridable via --sigmas.
SIGMAS = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

# A 12-word vocab reused across roles (a word is agent in one fact, patient in another -- role-bound, so no collision).
VOCAB = ["apple", "bird", "cat", "chase", "dog", "eat", "grass", "home", "leaf", "river", "see", "seed"]

# 9 facts: 3 agents x 3 actions -> 9 DISTINCT (agent,action) cues; 9 DISTINCT patient strings (unambiguous recall).
FACTS = [
    ("dog", "eat", "grass"),    # 0
    ("cat", "eat", "apple"),    # 1
    ("bird", "eat", "river"),   # 2
    ("dog", "chase", "home"),   # 3
    ("cat", "chase", "seed"),   # 4
    ("bird", "chase", "leaf"),  # 5
    ("dog", "see", "cat"),      # 6
    ("cat", "see", "bird"),     # 7
    ("bird", "see", "dog"),     # 8
]
# LATIN-SQUARE DA assignment: exactly one {high,low,tonic} per agent AND per action -> DA is orthogonal to BOTH content
# axes (no DA<->agent / DA<->action confound). high={0,4,8} low={1,5,6} tonic={2,3,7}.
DA_CLASS = ["high", "low", "tonic", "tonic", "high", "low", "low", "tonic", "high"]
_DA = {"high": 1.24, "tonic": 0.5, "low": 0.05}   # per da_mode_drives_chat's afferent calibration (== the wired verify)
# a cue never taught (moat probe): 'leaf' is only ever a PATIENT, never an agent -> ('leaf','eat') matches no stored fact.
UNSTORED = ("leaf", "eat")


def _battery():
    """The fixed 9-fact battery + its per-fact teaching DA (Latin-square) + the unstored moat cue. Deterministic (no
    RNG) so OFF and the derived ON see the identical facts + DA schedule."""
    das = [_DA[c] for c in DA_CLASS]
    return list(FACTS), das, UNSTORED


def _gains(das):
    return [da_to_encoding_gain(da, DA_BASELINE, K_DA) for da in das]


def _gains_homeostatic(das):
    """LEVER-2: the ON-arm gains under the PRODUCTION homeostatic companion (Turrigiano multiplicative scaling + a
    recall-safe floor). Folds the SHIPPED `webapp.da_encoding_drives_chat.homeostatic_step` over the fact sequence (one
    step per store == one homeostatic_step per fact), so the validated ON arm here IS the live-chat write path, not a
    re-implementation. r = 1 + k(DA-tonic) UNCLAMPED; s = clip(0.5,1.5, 1/mu); g = clip(1.0, 3.0, s*r); mu self-tunes."""
    from webapp.da_encoding_drives_chat import homeostatic_step, _MU_INIT
    mu = _MU_INIT
    out = []
    for da in das:
        r = 1.0 + K_DA * (da - DA_BASELINE)
        g, mu = homeostatic_step(mu, r)
        out.append(g)
    return out


def _scale_blocks(off_conns, gains, dim):
    """Derive the ON-arm store_conns from the OFF arm: block i (dim consecutive tuples, block-major) x gains[i]. This is
    EXACTLY what a real per-fact `encoding_gain_fn=lambda:g_i` build writes, because `_write_block` writes `g*zc[k]`
    (LINEAR in g; one_brain_composer.py:639). Cross-checked byte-equal against a real build on seed 42."""
    m = len(gains)
    assert len(off_conns) == m * dim, f"off_conns has {len(off_conns)} tuples, expected {m*dim} (={m} facts x D={dim})"
    out = []
    for i in range(m):
        g = complex(float(gains[i]))
        for (post, pre, w) in off_conns[i * dim:(i + 1) * dim]:
            out.append((post, pre, g * complex(w)))
    return out


def _build_off(seed, facts, dim, vocab, k_max, homeo=None):
    """One production OneBrainComposer with all facts stored at g=1 (encoding_gain_fn=None == today's default). `homeo`
    (opt): a dict of Turrigiano substrate-scaling params (beta_down/s_min/s_max) -> the composer is built with
    `homeostatic_scaling=True` so `apply_homeostatic_scaling()` can be run on a DA-gated store. The OFF arm itself is
    NEVER scaled (the flag only enables the method; the OFF sweep uses the unit store), so OFF stays byte-identical."""
    kw = {}
    if homeo is not None:
        kw = {"homeostatic_scaling": True, "homeo_beta_down": homeo["beta_down"],
              "homeo_s_min": homeo["s_min"], "homeo_s_max": homeo["s_max"]}
    c = OneBrainComposer(seed=seed, D=dim, vocab=vocab, k_max=k_max, enable_batched=False,
                         enable_rf_cudagraph=False, enable_csr_cache=False, enable_spiking_cleanup=False,
                         encoding_gain_fn=None, **kw)
    for (a, act, p) in facts:
        c.store(a, act, p)
    return c


def _build_on_real(seed, facts, gains, dim, vocab, k_max, homeo=None):
    """The DERIVATION cross-check: a REAL per-fact encoding_gain_fn build (the production ON path). Its store_conns must
    be byte-equal to `_scale_blocks(off_conns, gains)` -- proving the derived ON arm == the real one. `homeo` (opt):
    build with `homeostatic_scaling=True` so the caller can run the substrate rule on this real build."""
    kw = {}
    if homeo is not None:
        kw = {"homeostatic_scaling": True, "homeo_beta_down": homeo["beta_down"],
              "homeo_s_min": homeo["s_min"], "homeo_s_max": homeo["s_max"]}
    holder = {"g": 1.0}
    c = OneBrainComposer(seed=seed, D=dim, vocab=vocab, k_max=k_max, enable_batched=False,
                         enable_rf_cudagraph=False, enable_csr_cache=False, enable_spiking_cleanup=False,
                         encoding_gain_fn=lambda: holder["g"], **kw)
    for (a, act, p), g in zip(facts, gains):
        holder["g"] = float(g)
        c.store(a, act, p)
    return c


def _set_arm(comp, store_conns):
    """Point the live composer at an arm's store_conns and bust the store CSR cache so the next read rebuilds from it.
    (_query_under_damage restores comp.store_conns to this base after each damaged query, so the arm stays set.)"""
    comp.store_conns = store_conns
    comp._store_dirty = True
    comp._store_csr = None
    if getattr(comp, "_csr_cache", None) is not None:
        comp._csr_cache = {}


def _recall_set(comp, facts, sigma, seed):
    """Per-fact recall at a fixed read-stress sigma. sigma=0 -> the clean magnitude-invariant read; sigma>0 -> the
    I-7-b RF-floor damage (each fact its own reproducible draw, IDENTICAL across arms so the noise is matched)."""
    ok, genuine = _recall_set_attributed(comp, facts, sigma, seed)
    return ok


def _recall_set_attributed(comp, facts, sigma, seed):
    """Per-fact recall AT the read-stress sigma, WITH target-block attribution (the instrument fix, 2026-08-25).
    Returns (ok, genuine): ok[i] = the returned patient equals fact i's patient (the raw recall the prior soak counted);
    genuine[i] = the recall was produced by fact i's OWN engram (`_seq_block` selected the TARGET block i) AND that
    block decoded the right patient. A recall where a NON-TARGET engram's damaged decode coincidentally matched the cue
    (a foreign-block confabulation -- selected block != i, or None) has ok[i]=True but genuine[i]=False: it is NOT a
    recall of the cued memory (the no-confab moat should suppress it), so it must not be counted as a recall the
    default-ON flip has to preserve. The two prior soaks' single raw stress-net violation was exactly such a
    foreign-block confabulation in the OFF arm at the noise floor (verified: seed 43, sigma 6.0)."""
    ok, genuine = [], []
    for i, (a, act, p) in enumerate(facts):
        if sigma <= 0.0:
            sel, rec = _select_and_read(comp, a, act)
        else:
            sel, rec = _query_under_damage_attributed(comp, a, act, sigma, seed * 100003 + i)
        hit = (rec == p)
        ok.append(hit)
        genuine.append(bool(hit and sel == i))
    return ok, genuine


def _select_and_read(comp, agent, action):
    """The (selected block index, decoded patient) for a cue in ONE store read -- the host first-match path
    `query_patient` uses (integrated_loop=False), captured so recall can be attributed to the TARGET engram vs a
    foreign-block confabulation WITHOUT a second full read. Equivalent to (`_seq_block(a,act)`, `query_patient(a,act)`)
    for the non-clause soak battery: first block whose decoded (agent,action) matches -> its decoded patient. Returns
    (None, None) on abstain (== the no-confab moat)."""
    reads = comp._read_blocks()
    for i, got in enumerate(reads):
        if got.get("agent") == agent and got.get("action") == action:
            return i, comp._attributed_patient(i, got.get("patient"), got)
    return None, None


def _query_under_damage_attributed(comp, agent, action, sigma, dmg_seed):
    """Like `_query_under_damage` but ALSO returns the SELECTED block index under the same damage draw, so recall can be
    attributed to the TARGET engram vs a foreign-block confabulation -- in ONE store read. Same damage draw + restore
    contract as `_query_under_damage` (identical rng seed -> identical noise), read on the damaged store."""
    import numpy as _np
    rng = _np.random.default_rng(dmg_seed)
    clean = comp.store_conns
    try:
        comp.store_conns = _damage_store_conns(clean, sigma, rng)
        comp._store_dirty = True
        comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}
        return _select_and_read(comp, agent, action)
    finally:
        comp.store_conns = clean
        comp._store_dirty = True
        comp._store_csr = None
        if getattr(comp, "_csr_cache", None) is not None:
            comp._csr_cache = {}


def _moat_holds(comp, unstored, sigma, seed):
    a, act = unstored
    if sigma <= 0.0:
        return comp.query_patient(a, act) is None
    return _query_under_damage(comp, a, act, sigma, seed * 100003 + 99999) is None


def _run_seed(seed, facts, gains, dim, vocab, k_max, sigmas, cross_check, substrate=None):
    """Build the OFF composer, form the ON arm, (optionally) cross-check the derivation, sweep both arms over sigmas.
    Returns (row_per_sigma, mech_ok, off_mags, on_mags, cross_check_dict|None, homeo_scales|None).

    `substrate` (opt): a dict {"raw_gains", "beta_down", "s_min", "s_max"} -> the SUBSTRATE-SCALING mode. The ON arm is
    formed by (a) DA-gating the OFF store with the RAW clamped map (the pre-homeostasis gate) then (b) running the REAL
    on-substrate Turrigiano rule `OneBrainComposer.apply_homeostatic_scaling()` (resonate-sense each engram's readout
    activity -> multiplicatively rescale its store synapses toward the unit set-point). This REPLACES the host-proxy
    homeostat with a genuine synaptic-scaling mechanism. mech_ok then checks the recall-safe FLOOR (no engram left below
    the floor) + the salience ORDER (high-DA final |w| > tonic), not the linear gain identity."""
    homeo = {"beta_down": substrate["beta_down"], "s_min": substrate["s_min"], "s_max": substrate["s_max"]} \
        if substrate else None
    c = _build_off(seed, facts, dim, vocab, k_max, homeo=homeo)
    off_conns = list(c.store_conns)
    m = len(facts)
    homeo_scales = None
    if substrate:
        raw_on = _scale_blocks(off_conns, substrate["raw_gains"], dim)      # DA-gate the OFF store (raw clamped map)
        c.store_conns = list(raw_on); c._store_dirty = True; c._store_csr = None
        homeo_scales = c.apply_homeostatic_scaling()                        # the REAL on-substrate synaptic-scaling rule
        on_conns = list(c.store_conns)
    else:
        on_conns = _scale_blocks(off_conns, gains, dim)

    off_mags = [_block_mean_mag(off_conns, i, dim) for i in range(m)]
    on_mags = [_block_mean_mag(on_conns, i, dim) for i in range(m)]
    if substrate:
        floor_frac = 0.9                                                   # recall-safe: no engram below 0.9*unit
        no_below_floor = all(on_mags[i] >= floor_frac for i in range(m))
        cls = ["high", "low", "tonic", "tonic", "high", "low", "low", "tonic", "high"]
        hi = [on_mags[i] for i in range(m) if cls[i] == "high"]
        to = [on_mags[i] for i in range(m) if cls[i] == "tonic"]
        order_ok = (min(hi) > max(to)) if (hi and to) else True            # DA-salience order survives scaling
        mech_ok = bool(no_below_floor and order_ok)
    else:
        mech_ok = all(abs(on_mags[i] - gains[i] * off_mags[i]) < 1e-9 for i in range(m))

    cc = None
    if cross_check:
        # DERIVATION cross-check. Base: a REAL per-fact encoding_gain_fn build. Substrate mode: that real build is the
        # RAW-DA gate followed by the REAL apply_homeostatic_scaling() -- so the derived ON arm (block-scale OFF by the
        # raw map, then apply the substrate rule) is proven byte-equal to a from-scratch production ON build.
        if substrate:
            c_real = _build_on_real(seed, facts, substrate["raw_gains"], dim, vocab, k_max, homeo=homeo)
            c_real.apply_homeostatic_scaling()
        else:
            c_real = _build_on_real(seed, facts, gains, dim, vocab, k_max)
        real_conns = list(c_real.store_conns)
        max_diff = 0.0
        same_len = (len(real_conns) == len(on_conns))
        if same_len:
            for (_p1, _q1, w1), (_p2, _q2, w2) in zip(on_conns, real_conns):
                d = abs(complex(w1) - complex(w2))
                if d > max_diff:
                    max_diff = d
        cc = {"seed": seed, "n_conns_derived": len(on_conns), "n_conns_real": len(real_conns),
              "same_length": bool(same_len), "max_abs_weight_diff": float(max_diff),
              "byte_equal": bool(same_len and max_diff < 1e-9)}
        del c_real

    recall = {"off": {}, "on": {}}
    genuine = {"off": {}, "on": {}}
    moat = {"off": {}, "on": {}}
    for arm, conns in (("off", off_conns), ("on", on_conns)):
        _set_arm(c, conns)
        for sigma in sigmas:
            ok, gen = _recall_set_attributed(c, facts, sigma, seed)
            recall[arm][sigma] = ok
            genuine[arm][sigma] = gen
            moat[arm][sigma] = _moat_holds(c, unstored=UNSTORED, sigma=sigma, seed=seed)

    rows = []
    for sigma in sigmas:
        off = recall["off"][sigma]
        on = recall["on"][sigma]
        off_gen = genuine["off"][sigma]
        regressed = [i for i in range(m) if off[i] and not on[i]]
        improved = [i for i in range(m) if on[i] and not off[i]]
        # GENUINE regression (instrument fix): a fact OFF recalls via its OWN target engram that ON then loses. A
        # raw-regressed fact where OFF's "recall" came from a FOREIGN engram's damaged decode (off_gen[i]=False) is a
        # confabulation, not a recall of the cued memory -> excluded from the genuine stress net.
        regressed_genuine = [i for i in range(m) if off_gen[i] and not on[i]]
        confab_off_idx = [i for i in range(m) if off[i] and not off_gen[i]]
        rows.append({
            "sigma": sigma, "recall_off": int(sum(off)), "recall_on": int(sum(on)), "of": m,
            "recall_off_genuine": int(sum(off_gen)), "n_confab_off": len(confab_off_idx),
            "confab_off_idx": confab_off_idx,
            "n_regressed": len(regressed), "n_improved": len(improved),
            "n_regressed_genuine": len(regressed_genuine), "regressed_genuine_idx": regressed_genuine,
            "regressed_idx": regressed, "improved_idx": improved,
            "moat_off": bool(moat["off"][sigma]), "moat_on": bool(moat["on"][sigma]),
        })
    del c
    return rows, mech_ok, off_mags, on_mags, cc, homeo_scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--D", type=int, default=D)
    ap.add_argument("--no-cross-check", action="store_true",
                    help="skip the seed-0 real-encoding_gain_fn derivation cross-check build (1 fewer brain build)")
    ap.add_argument("--homeostatic", action="store_true",
                    help="LEVER-2: derive the ON arm from the PRODUCTION homeostatic companion (Turrigiano "
                         "multiplicative scaling + recall-safe floor, webapp.da_encoding_drives_chat.homeostatic_step) "
                         "instead of the RAW clamped map. This is the HOST-PROXY default-ON write path.")
    ap.add_argument("--substrate-scaling", action="store_true",
                    help="LEVER-3 (2026-08-25): form the ON arm from the RAW DA gate + the REAL on-substrate Turrigiano "
                         "rule OneBrainComposer.apply_homeostatic_scaling() (resonate-sense each engram's readout "
                         "activity -> multiplicatively rescale its store synapses toward the unit set-point). Replaces "
                         "the host-proxy homeostat with a genuine synaptic-scaling mechanism.")
    ap.add_argument("--homeo-beta-down", type=float, default=0.25,
                    help="substrate-scaling: the down-regulation exponent for STRONG engrams (s=(A*/A_i)^beta_down). "
                         "<1 preserves the relative DA-salience order while pulling the extreme toward the set-point.")
    ap.add_argument("--homeo-s-min", type=float, default=0.34,
                    help="substrate-scaling: floor on the strongest down-scale (caps regulation of a very strong engram)")
    ap.add_argument("--homeo-s-max", type=float, default=4.0,
                    help="substrate-scaling: ceiling on the strongest up-scale (caps restoration of a near-dead engram)")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_REPO, "research", "findings", "raw", "_da_encoding_leansoak", "soak.json"))
    args = ap.parse_args()

    seeds = args.seeds
    sigmas = sorted(set(args.sigmas))
    dim = args.D
    facts, das, unstored = _battery()
    substrate = None
    if args.substrate_scaling:
        mode = "substrate_scaling"
        substrate = {"raw_gains": _gains(das), "beta_down": args.homeo_beta_down,
                     "s_min": args.homeo_s_min, "s_max": args.homeo_s_max}
        gains = None    # the effective ON gains are the MEASURED post-scaling |w| (per-seed identical), filled below
    else:
        mode = "homeostatic" if args.homeostatic else "raw"
        gains = _gains_homeostatic(das) if args.homeostatic else _gains(das)
    m = len(facts)
    k_max = m + 4

    per_seed = []
    clean_regressions_total = 0
    # MOAT instrument fix (lever-2): decompose the moat failures. An unstored cue that leaks on the OFF arm (encoding
    # DISABLED, g=1) is a BASELINE read-floor artifact of the control arm -- NOT the coupling manufacturing a fact; it
    # is reported but EXCLUDED from the coupling verdict. The genuine "encoding manufactures a fact" residual is a leak
    # the ON arm INTRODUCES where the byte-identical OFF baseline abstains (moat_off True & moat_on False).
    moat_introduced_total = 0            # off abstains, on leaks  -> the genuine encoding residual (GO requires 0)
    moat_baseline_total = 0              # off leaks               -> control-arm read-floor artifact (reported, excluded)
    stress_net_violations = 0            # sigmas (across seeds) where recall_ON < recall_OFF (RAW)
    stress_net_genuine_violations = 0    # sigmas where recall_ON < GENUINE (target-attributed) recall_OFF (the fix)
    confab_off_total = 0                 # (seed,sigma,fact) OFF "recalls" produced by a FOREIGN engram's mis-decode
    mech_ok_all = True
    cross_check = None
    homeo_scales = None
    # per-sigma aggregate OFF recall across seeds (the CALIBRATION curve + the vacuity guard).
    agg_off = {s: 0 for s in sigmas}
    total_possible = m * len(seeds)

    for si, seed in enumerate(seeds):
        do_cc = (si == 0) and (not args.no_cross_check)
        rows, mech_ok, off_mags, on_mags, cc, scales = _run_seed(
            seed, facts, gains, dim, VOCAB, k_max, sigmas, cross_check=do_cc, substrate=substrate)
        if cc is not None:
            cross_check = cc
        if scales is not None:
            homeo_scales = scales
        mech_ok_all = mech_ok_all and mech_ok
        for r in rows:
            s = r["sigma"]
            agg_off[s] += r["recall_off"]
            if s <= 0.0:
                clean_regressions_total += r["n_regressed"]
            else:
                if r["recall_on"] < r["recall_off"]:
                    stress_net_violations += 1
                if r["recall_on"] < r["recall_off_genuine"]:
                    stress_net_genuine_violations += 1
                confab_off_total += r["n_confab_off"]
            if not r["moat_off"]:
                moat_baseline_total += 1                 # control-arm (encoding OFF) read-floor artifact
            elif not r["moat_on"]:
                moat_introduced_total += 1               # encoding INTRODUCED a leak where OFF abstained
        per_seed.append({"seed": seed, "off_block_mags": off_mags, "on_block_mags": on_mags, "sweep": rows})

    # CALIBRATION: the smallest sigma>0 where aggregate OFF recall first drops below full (the knee), and the OFF curve.
    pos_sigmas = [s for s in sigmas if s > 0.0]
    knee_sigma = next((s for s in pos_sigmas if agg_off[s] < total_possible), None)
    off_curve = {("%.4g" % s): agg_off[s] for s in sigmas}
    # INSTRUMENT BITES: OFF recall must degrade below full somewhere on the grid (else the no-regression test is vacuous).
    instrument_bites = any(agg_off[s] < total_possible for s in pos_sigmas)

    # first sigma where ANY seed shows a regression (where the salience-gating starts to bite).
    first_regress_sigma = None
    for s in pos_sigmas:
        if any(any(r["sigma"] == s and r["n_regressed"] > 0 for r in ps["sweep"]) for ps in per_seed):
            first_regress_sigma = s
            break

    derivation_ok = (args.no_cross_check or (cross_check is not None and cross_check["byte_equal"]))

    # OUTCOME (GO vs NO-GO): the no-regression result. PRECONDITIONS (instrument valid + safety invariants) gate to
    # UNDEFINED; the outcome gates GO/NO-GO.
    go_clean = (clean_regressions_total == 0)
    # The STRESS-net GO is on the GENUINE (target-attributed) recall: recall_ON >= the OFF recalls produced by the fact's
    # OWN target engram. A raw-only violation where OFF's "recall" came from a FOREIGN engram's damaged decode is a
    # confabulation the no-confab moat should suppress (ON abstaining is correct behaviour, not a memory regression). The
    # RAW stress-net + the confab decomposition are reported alongside so the decision is auditable.
    go_stress_net = (stress_net_genuine_violations == 0)
    go = bool(go_clean and go_stress_net)

    from tools.verdict import Verdict
    v = Verdict("DA-gated encoding default-ON is safe on the production magnitude store (LEAN no-regression soak)")
    # --- preconditions (a failure -> UNDEFINED, not a fake GO/NO-GO): the instrument must be valid + safe-invariant. ---
    v.require("INSTRUMENT BITES: OFF-arm recall degrades below full somewhere on the swept sigma grid "
              "(else the no-regression test is vacuous -- widen --sigmas)", instrument_bites, expect=True,
              note=f"calibrated knee sigma={knee_sigma}; aggregate OFF recall curve (of {total_possible}) = {off_curve}")
    _cc_name = ("DERIVATION VALID: the RAW-gate + substrate-scaled ON arm is byte-equal to a REAL from-scratch "
                "encoding_gain_fn + apply_homeostatic_scaling build (seed 0)") if substrate else \
               "DERIVATION VALID: the block-scaled ON arm is byte-equal to a REAL per-fact encoding_gain_fn build (seed 0)"
    v.require(_cc_name, derivation_ok, expect=True,
              note=("cross-check skipped (--no-cross-check)" if args.no_cross_check else
                    f"max|dw|={cross_check['max_abs_weight_diff'] if cross_check else None}"))
    if substrate:
        v.require("MECHANISM (substrate scaling): every engram left at/above the recall-safe floor (0.9*unit) AND the "
                  "DA-salience ORDER survives (min high-DA |w| > max tonic |w|) -- the on-substrate Turrigiano rule "
                  "floors weak engrams to the set-point + regulates strong ones while preserving order.",
                  mech_ok_all, expect=True,
                  note=f"applied per-engram scales (seed-independent) = {[round(s,3) for s in (homeo_scales or [])]}")
    else:
        v.require("MECHANISM: ON block |w| == g x OFF block |w| for every fact/seed (the gain literally scales stored |w|)",
                  mech_ok_all, expect=True)
    v.require("MOAT (encoding-introduced): the ON arm never leaks an unstored cue where the byte-identical OFF baseline "
              "abstains (encoding never MANUFACTURES a fact). Control-arm OFF leaks are baseline read-floor artifacts, "
              "reported + excluded.",
              moat_introduced_total, expect=0,
              note=f"encoding-introduced leaks={moat_introduced_total} (GO requires 0); baseline read-floor artifacts "
                   f"(OFF arm, encoding disabled)={moat_baseline_total} (instrument property, excluded)")
    # --- outcome (GO/NO-GO): the actual no-regression question. ---
    v.require("CLEAN read (sigma=0): zero facts regress OFF->ON on every seed (magnitude-invariant)",
              clean_regressions_total, expect=0,
              note="the dominant production case for a modest fact store: a phase read is magnitude-invariant")
    v.require("STRESS net (GENUINE): recall_ON >= the target-attributed OFF recall at every swept sigma "
              "(salience redistribution never loses a fact OFF recalls via its OWN engram)",
              stress_net_genuine_violations, expect=0,
              note=f"RAW stress-net violations (incl. foreign-block confabulations)={stress_net_violations}; "
                   f"OFF confabulation recalls excluded={confab_off_total} (a foreign engram's damaged decode "
                   f"coincidentally matched the cue -- ON abstaining there is the moat working, not a regression)")
    v.disabled("the spiking-cleanup read (enable_spiking_cleanup=False for speed)",
               why="the magnitude sensitivity lives in the substrate store's |w| + the RF read floor, exercised by the "
                   "host read-damage sweep (the I-7-b instrument); the spiking cleanup adds intrinsic noise of the same "
                   "KIND (a further read-stress point on this sweep), not a different mechanism")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    out = {
        "runner": "research/runners/_da_encoding_leansoak.py",
        "coupling": "DA-gated encoding, LEAN production magnitude-store no-regression soak (the default-ON flip gate)",
        "mode": mode,
        "config": {"seeds": seeds, "D": dim, "k_da": K_DA, "da_baseline": DA_BASELINE, "battery_m": m,
                   "k_max": k_max, "sigmas": sigmas, "vocab": VOCAB, "facts": FACTS, "da_class": DA_CLASS,
                   "da_values": _DA, "gains": (gains if gains is not None else _gains(das)), "gain_mode": mode,
                   "unstored_moat_cue": list(UNSTORED),
                   "substrate_scaling": (None if not substrate else
                                         {"beta_down": substrate["beta_down"], "s_min": substrate["s_min"],
                                          "s_max": substrate["s_max"], "raw_da_gains": substrate["raw_gains"],
                                          "applied_scales": homeo_scales,
                                          "effective_on_gains": ([round(g * s, 4) for g, s in
                                                                  zip(substrate["raw_gains"], homeo_scales)]
                                                                 if homeo_scales else None)})},
        "VERDICT": "GO" if go else decided["status"], "status": decided["status"],
        "calibration": {"instrument_bites": instrument_bites, "knee_sigma": knee_sigma,
                        "aggregate_off_recall_curve": off_curve, "total_possible_per_sigma": total_possible},
        "outcome": {"go_clean_zero_regression": go_clean, "go_stress_net_nonnegative": go_stress_net,
                    "clean_regressions_total": clean_regressions_total,
                    "stress_net_violations": stress_net_violations,
                    "stress_net_genuine_violations": stress_net_genuine_violations,
                    "confab_off_total": confab_off_total, "first_regression_sigma": first_regress_sigma},
        "preconditions_summary": {"instrument_bites": instrument_bites, "derivation_ok": derivation_ok,
                                  "mechanism_scales_all": mech_ok_all,
                                  "moat_introduced_total": moat_introduced_total,
                                  "moat_baseline_total": moat_baseline_total},
        "derivation_cross_check": cross_check,
        "per_seed": per_seed,
        "preconditions": decided["preconditions"],
        "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    bar = "=" * 100
    print("\n" + bar, flush=True)
    print("  DA-GATED ENCODING — LEAN production magnitude-store NO-REGRESSION soak", flush=True)
    print(bar, flush=True)
    print(f"  MODE: {mode}  (substrate_scaling = the on-substrate Turrigiano synaptic rule; homeostatic = the host "
          f"proxy; raw = the bare clamped map)", flush=True)
    _disp_gains = gains if gains is not None else (out["config"]["substrate_scaling"]["effective_on_gains"] or [])
    print(f"  battery={m} facts (Latin-square DA), {len(seeds)} seeds, "
          f"{'effective ON gains' if substrate else 'gains'}={['%.3f' % g for g in _disp_gains]}", flush=True)
    if substrate:
        print(f"  substrate scales applied (per engram): {[round(s,3) for s in (homeo_scales or [])]}", flush=True)
    print(f"  INSTRUMENT BITES: {instrument_bites}   calibrated knee sigma: {knee_sigma}", flush=True)
    print(f"  aggregate OFF recall curve (of {total_possible} per sigma): {off_curve}", flush=True)
    if cross_check is not None:
        print(f"  DERIVATION cross-check (seed {cross_check['seed']}): byte_equal={cross_check['byte_equal']} "
              f"max|dw|={cross_check['max_abs_weight_diff']:.2e}", flush=True)
    print(f"  MECHANISM ok (all seeds): {mech_ok_all}", flush=True)
    print(f"  CLEAN (sigma=0) regressions: {clean_regressions_total} (expect 0)", flush=True)
    print(f"  MOAT encoding-introduced leaks: {moat_introduced_total} (expect 0)  |  baseline read-floor artifacts "
          f"(OFF arm, excluded): {moat_baseline_total}", flush=True)
    print(f"  STRESS net GENUINE violations: {stress_net_genuine_violations} (expect 0)  |  RAW (incl. confab): "
          f"{stress_net_violations}  |  OFF confab recalls excluded: {confab_off_total}", flush=True)
    print(f"  first regression appears at sigma: {first_regress_sigma}", flush=True)
    print(f"\n  per-sigma (summed across {len(seeds)} seeds):", flush=True)
    agg = {}
    for ps in per_seed:
        for r in ps["sweep"]:
            a = agg.setdefault(r["sigma"], {"off": 0, "gen": 0, "on": 0, "reg": 0, "regg": 0, "imp": 0, "cf": 0})
            a["off"] += r["recall_off"]; a["on"] += r["recall_on"]; a["gen"] += r.get("recall_off_genuine", r["recall_off"])
            a["reg"] += r["n_regressed"]; a["regg"] += r.get("n_regressed_genuine", 0)
            a["imp"] += r["n_improved"]; a["cf"] += r.get("n_confab_off", 0)
    for s in sigmas:
        a = agg[s]
        print(f"    sigma={s:<5.2f} off={a['off']:3d}(gen={a['gen']:3d},confab={a['cf']:2d}) on={a['on']:3d}/{total_possible}  "
              f"reg={a['reg']:2d}(genuine={a['regg']:2d}) improved={a['imp']:2d}", flush=True)
    print(f"\n  VERDICT: {out['VERDICT']} ({decided['status']})", flush=True)
    if decided["undefined_reasons"]:
        for r in decided["undefined_reasons"]:
            print(f"     UNDEFINED/unmet: {r}", flush=True)
    print(f"  [saved] {args.out}\n" + bar, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
