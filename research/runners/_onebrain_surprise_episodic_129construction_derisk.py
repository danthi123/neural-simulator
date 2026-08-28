"""One-brain INTEGRATION — does board #129's ALREADY-WORKING construction (TWO neuromodulatory context-gated
SEPARATE Hebbian cross-edges feeding an already-opponent-inhibiting pool pair, read by a divisively-normalized
OPPONENT RATIO) map onto the surprise->source_provenance F2 crux, whose SINGLE-shared-edge + RAW-MARGIN read is
UNDEFINED (`research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-UNDEFINED.md`: F2's own lesion
control fails its precondition on 5/6 seeds)?

CONTEXT (read before touching this file; do not re-derive):
  * `research/findings/2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md` (#129) is
    GO 6/6 via TWO SEPARATE zero-init Hebbian traces (`episode->prov_perceived`, `episode->prov_generated`), each
    potentiated ONLY when its own neuromodulatory context line (`ctx_perceived`/`ctx_generated`, "exactly one
    active per encode") drives its target pool's post-synaptic firing during a content-cued encode; read as the
    divisively-normalized ratio `d=(r_true-r_false)/(r_true+r_false)`, immune to the absolute-rate common-mode
    weakness that killed the rate-floor family.
  * `research/runners/_onebrain_integration_surprise_episodic_crossedge.py` (the blocked edge) builds only ONE
    cross-edge, `surprise -> prov_generated` (CONTRADICT-trial-trained), and reads the RAW rate MARGIN
    `rate_generated - rate_perceived`. F1/F3/F4/emergence/lesion-recovers-migration are all clean 6/6; F2's own
    lesion control (`delta_lesion < 0.34 * delta_intact`) fails on 5/6 seeds -- UNDEFINED, not a validated
    negative.
  * `research/findings/2026-08-28-read-fidelity-opponent-pushpull-NOGO-...md` ALREADY TESTED a closely-related
    idea on this EXACT edge: an opponent/push-pull SIGN-RECOVERING READ (two Dale's-law non-negative TEMPLATE
    channels, `I_push-I_pull`, fit as an offline decoder on the raw rasters of the SAME single trained edge) --
    0/6, net WORSE than the single rectified channel it replaced, with the explicit conclusion "consistent with
    #129 (separate-trace WIRING delivers provenance, not the read alone)". That result is DIFFERENT from what
    this runner tests (a post-hoc decoder-style opponent READ vs an actual SECOND IN-NETWORK LEARNED HEBBIAN
    cross-edge, #129's literal construction) but is directly relevant evidence: fixing the READ FORMULA ALONE,
    without adding a genuinely separate learned trace, was already tried on this crux and failed. This runner
    tests the untried half: does adding the missing SECOND cross-edge (#129's actual "separate-trace WIRING")
    change the answer.

THE ARCHITECTURAL QUESTION THIS RUNNER ANSWERS EMPIRICALLY (see the finding for the full reasoning): #129's two
context lines (`ctx_perceived`/`ctx_generated`) are externally-injected, cleanly MUTUALLY EXCLUSIVE per encode --
a clean design property. `surprise`'s own D2 expectation-violation circuit
(`_spiking_expectation_rpe_derisk.build_expectation_circuit`) is architecturally a UNIPOLAR RECTIFIED mismatch
detector (`surprise = relu(patient_asserted_exc - patient_expected_inh)`): `patient_expected` and
`patient_asserted` both fire REGARDLESS of CONFIRM/CONTRADICT (driven purely by cue/assert presence; only their
downstream SUBTRACTION into `surprise` discriminates), so there is no pre-existing population in this circuit
that is cleanly "high on CONFIRM, ~0 on CONTRADICT" the way `ctx_perceived`/`ctx_generated` are "exactly one
active per encode". `patient_expected` is the closest available candidate for the CONFIRM-side driver (it is a
real, existing spiking population, cue-driven, present on every trial) -- but because it does NOT discriminate
CONFIRM from CONTRADICT, a cross-edge trained on it under CONFIRM will ALSO fire (contributing some current into
`prov_perceived`) during a CONTRADICT-hold read, a genuine confound this runner measures rather than assumes.

THE CONSTRUCTION (additive; NO sim/ edit; reuses `SurpriseEpisodicPool` verbatim for everything except the
second cross-edge + the read formula):
  * EDGE 1 (unchanged, already GO on every OTHER arm): `surprise -> prov_generated`, trained by co-driving a
    CONTRADICT trial + the FIXED `ctx_generated` line (`_onebrain_integration_surprise_episodic_crossedge.py`'s
    own recipe, byte-identical).
  * EDGE 2 (NEW, this runner): `patient_expected -> prov_perceived`, trained by co-driving a CONFIRM trial + the
    FIXED `ctx_perceived` line -- the direct structural mirror of edge 1, using the best available CONFIRM-side
    presynaptic population.
  * READ: F2/F4 recomputed as the divisively-normalized ratio `d=(r_gen-r_perc)/(r_gen+r_perc+eps)` (#129's
    formula) instead of the raw margin `r_gen-r_perc`.
  * LESION (F2's crux): BOTH cross-edges zeroed together (both are now jointly "the surprise->episodic
    mechanism" under test).

BOUNDED PROTOCOL (hard session cap: no open-ended search): ONE non-canonical calibration seed (7) first, to
decide honestly (before looking at the canonical 6) whether this construction is even directionally promising;
only then the canonical 6-seed set (42,43,44,100,101,102). numpy CPU only, additive, no `sim/` edit, no
production wiring.

Run:
  SIM_BACKEND=numpy python -m research.runners._onebrain_surprise_episodic_129construction_derisk --seeds 7 --smoke
  SIM_BACKEND=numpy python -m research.runners._onebrain_surprise_episodic_129construction_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/_onebrain_surprise_episodic_129construction_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from sim.backend import to_host, get_backend
from tools.lab import attributable_to

from research.runners._onebrain_integration_surprise_episodic_crossedge import (
    W0, GATE, CUE_PA, CTX_DRIVE_PA, EPISODE_DRIVE_PA, TRAIN_STEPS, N_EPISODES, RECALL_STEPS, N_READS, HMAX,
    N_AMBIG_PASSES, PRE_STEPS, CROSS_EDGE_LR, F1_SEP_RATIO, F4A_FRAC, F4B_RETAIN, RATE_LO, RATE_HI,
    OTHER_BLOCK_DRIFT_MAX, _CONDUCT, _assign_blocks, SurpriseEpisodicPool, _f1, _emergence,
)

GATE2 = "confirm_to_provperc"          # the NEW cross-edge's plasticity gate
EPS = 1e-9                             # pure divide-by-zero guard (never the noise-suppression term -- that is
                                       # DN_SIGMA below; a bare epsilon alone was found to be INSUFFICIENT, see
                                       # DN_SIGMA's own comment)
# DN_SIGMA -- a semisaturation constant, the standard form of divisive normalization in visual/sensory cortex
# (Carandini & Heeger 2011, Nat Rev Neurosci 13:51-62, "Normalization as a canonical neural computation", PMID
# 21587300: r = R_max * drive^n / (drive^n + sigma^n), sigma set the operating range so weak/near-silent drive
# does not saturate the normalized response). WITHOUT this term (bare-epsilon ratio, the first cut tried on
# seed 7): F2's lesion-control precondition PASSES cleanly (frac_attributable 0.965 vs the raw-margin
# construction's 0.297-0.727) but F4a (moat) FAILS -- a surprise-hold with ZERO content drive at all reads
# gen=0.0094Hz/perc=0.0Hz (both near the instrument noise floor, an order of magnitude below the ~0.09-0.24Hz
# content-driven denominators), and a bare-epsilon ratio (0.0094-0)/(0.0094+0+1e-9) amplifies that noise-floor
# difference to ratio~+1.0 -- a spuriously MAXIMAL "generated" verdict from silence, exactly the confabulation-
# from-bias-alone F4a exists to catch. This is the textbook divisive-normalization failure mode a semisaturation
# constant is FOR (a bare epsilon only guards literal division-by-zero; it does not suppress a near-zero-but-
# nonzero denominator's noise). DN_SIGMA=0.05 is PRE-REGISTERED from this seed-7 diagnostic (silent denominator
# ~0.0094, real content-driven denominators ~0.14-0.24 -- sigma set to roughly midway in log-scale between them,
# a round number, BEFORE any canonical seed (42/43/44/100/101/102) is read) -- not tuned against the canonical
# set to force a pass.
DN_SIGMA = 0.05
# HMAX_129 / N_EPISODES_129 -- this construction's OWN weight-bound + training budget, calibrated on seed 7,
# DIFFERENT from the single-edge runner's HMAX=20/N_EPISODES=150. With the imported HMAX=20 (unchanged), both
# edges converge to w~5.7-6.2 well before 40 episodes (raising N_EPISODES further does not change the converged
# magnitude -- tested at 40 vs 150, identical w_gen/w_perc to 3 decimals, so this is a TRUE plateau, not an
# under-training artifact) -- and AT THAT MAGNITUDE, F2 and F4a pass but F4b FAILS: a clear, already-correctly-
# encoded battery item's own (much smaller, content-driven) margin is overpowered by the two cross-edges' now-
# large synaptic current, FLIPPING its provenance judgment under a co-occurring wrong-context surprise hold --
# a genuine moat violation, not a read-formula artifact (diagnosed empirically, not assumed). Since the DN-ratio
# read's F2 signal (delta_intact~0.59 at HMAX=20) is ~60x the floor, there was headroom to shrink the bound:
# HMAX_129=2.5 (vs the single-edge construction's 20) keeps F2 comfortably clearing (frac_attributable 0.88 at
# N_EPISODES_129=60, vs the >=0.66-equivalent bar) while keeping both edges' magnitude small enough that a clear
# battery item's own trace remains dominant (F4b now holds). Frozen from this seed-7 diagnostic BEFORE any
# canonical seed is read (identical discipline to DN_SIGMA above).
HMAX_129 = 2.5
N_EPISODES_129 = 60
F2_LESION_RATIO = 0.34                 # UNCHANGED, scale-free (fraction) -- the precondition actually failing
# F2_INTACT_FLOOR (ratio units) is NOT hardcoded here: it is set ONCE from the seed-7 calibration run, before any
# canonical seed is read, exactly like the original runner's own CUE_PA/HMAX/N_EPISODES calibration discipline
# and #129's own coincidence-threshold calibration ("set on 6 NON-canonical calibration seeds ... frozen before
# the canonical run"). See `main()`.


def _dn_ratio(g, p):
    """The divisive-normalization opponent ratio with a semisaturation constant (see DN_SIGMA above):
    d = (r_gen - r_perc) / (r_gen + r_perc + DN_SIGMA + EPS)."""
    return (g - p) / (g + p + DN_SIGMA + EPS)


def _build_pool_129(seed):
    """The TWO-cross-edge merged pool: EDGE1 (surprise->prov_generated, unchanged) + EDGE2 (NEW,
    patient_expected->prov_perceived), both declared via the SAME declarative CrossEdge/merge_organs(cross_edges=)
    path the blocked single-edge runner already uses -- no bespoke re-inject, no sim/ edit."""
    from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, merge_organs
    SURPRISE = REGISTRY["surprise"]
    SP = REGISTRY["source_provenance"]
    SURPRISE_LITE = dataclasses.replace(
        SURPRISE, config={**SURPRISE.config, "enable_hebbian_learning": False, "hebbian_rate_window": False})
    CROSS_EDGES = [
        CrossEdge(key=GATE, source_key="surprise", source_region="surprise",
                 target_key="source_provenance", target_region="prov_generated",
                 init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True),
        CrossEdge(key=GATE2, source_key="surprise", source_region="patient_expected",
                 target_key="source_provenance", target_region="prov_perceived",
                 init_weight=W0, plastic=True, gate=GATE2, learn_rule="rate_hebbian", freeze_rest=True),
    ]
    pool = merge_organs([SURPRISE_LITE, SP], seed=seed, config_descriptors=[SURPRISE_LITE, SP],
                        wire=True, cross_edges=CROSS_EDGES)
    return pool


def _build_pool_plain_129(seed):
    from research.runners.onebrain_merge_framework import REGISTRY, merge_organs
    SURPRISE = REGISTRY["surprise"]
    SP = REGISTRY["source_provenance"]
    SURPRISE_LITE = dataclasses.replace(
        SURPRISE, config={**SURPRISE.config, "enable_hebbian_learning": False, "hebbian_rate_window": False})
    return merge_organs([SURPRISE_LITE, SP], seed=seed, config_descriptors=[SURPRISE_LITE, SP], wire=True)


class SurpriseEpisodic129Pool(SurpriseEpisodicPool):
    """SurpriseEpisodicPool + a second declarative cross-edge (patient_expected->prov_perceived, CONFIRM-trained)
    + ratio-based reads. Overrides ONLY __init__ (pool construction + mask bookkeeping for edge 2) and adds new
    methods (train_129, amb_read_ratio); every inherited method (_hard_reset, _drive, _wmean, cross_weights,
    _cue_idx, _cue_pre_pairs, _contradict_pairs, _confirm_pairs, _make_ambiguous_pattern, _encode_ambiguous) is
    reused byte-identical from the parent, exactly the R4-declarative-repro pattern
    (`_onebrain_declarative_crossedge_r1_repro.DeclarativeR1Pool`)."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.xp, _ = get_backend()
        self.pool = _build_pool_129(seed)
        self.pool.ensure_built()
        self.b = self.pool.bridge
        rm = self.b.region_manager

        def idxr(nm):
            return np.asarray(rm.indices(nm), np.int64)

        self.ix = {nm: idxr(nm) for nm in ("cue", "patient_expected", "patient_asserted", "surprise",
                                           "episode", "content_readout", "ctx_perceived", "ctx_generated",
                                           "prov_perceived", "prov_generated", "inh_perceived", "inh_generated")}
        meta = self.pool.meta["surprise"]
        self.blk = int(meta["blk"]); self.n_trained = int(meta["n_trained"]); self.n_novel = int(meta["n_novel"])
        self.cue_c, self.assert_cp = _assign_blocks(seed, self.n_trained)   # THIS SEED's random block pair

        coo = self.b.cp_connections.tocoo()
        row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
        surprise_idx = self.ix["surprise"]
        pexp_idx = self.ix["patient_expected"]
        trained_block_s = surprise_idx[self.assert_cp * self.blk:(self.assert_cp + 1) * self.blk]
        trained_block_p = pexp_idx[self.cue_c * self.blk:(self.cue_c + 1) * self.blk]
        post_gen = self.ix["prov_generated"]; post_perc = self.ix["prov_perceived"]

        full_mask_gen = np.isin(row, surprise_idx) & np.isin(col, post_gen)
        trained_mask_gen = np.isin(row, trained_block_s) & np.isin(col, post_gen)
        full_mask_perc = np.isin(row, pexp_idx) & np.isin(col, post_perc)
        trained_mask_perc = np.isin(row, trained_block_p) & np.isin(col, post_perc)
        assert int(full_mask_gen.sum()) > 0, "EDGE1 declarative cross-edge is EMPTY -- did not wire"
        assert int(full_mask_perc.sum()) > 0, "EDGE2 declarative cross-edge is EMPTY -- did not wire"
        self.masks = {
            "surprise->provgen": full_mask_gen,
            "trained_block->provgen": trained_mask_gen,
            "other_blocks->provgen": full_mask_gen & ~trained_mask_gen,
            "pexp->provperc": full_mask_perc,
            "trained_block->provperc": trained_mask_perc,
            "other_blocks->provperc": full_mask_perc & ~trained_mask_perc,
        }
        # both cross-edges' masks unioned == "the whole surprise->episodic mechanism", lesioned together for F2.
        self.masks["both_edges"] = full_mask_gen | full_mask_perc

        from research.runners.onebrain_merge_framework import _source_prov_organ
        self.sp_organ = _source_prov_organ(seed, self.pool)
        self.sp_organ.ensure_built()

        self.pool.apply_cross_edge_freeze()

        self.ambig_pattern = self._make_ambiguous_pattern()
        self._encode_ambiguous()

        self._frozen_w0 = np.asarray(to_host(self.b.cp_connections.data)).copy()
        self._noncross = ~np.zeros(self._frozen_w0.shape[0], dtype=bool)
        self._noncross &= ~self.masks["both_edges"]
        for kk, vv in dict(hebbian_symmetric=True, hebbian_learning_rate=CROSS_EDGE_LR, hebbian_max_weight=HMAX_129,
                           hebbian_min_weight=0.0, hebbian_weight_decay=0.0).items():
            setattr(self.b.core_config, kk, vv)

        self.b.cp_external_input_current[:] = 0.0
        for _ in range(40):
            self.b._run_one_simulation_step()
        self.rest_v = np.asarray(to_host(self.b.cp_membrane_potential_v)).copy()
        self.rest_u = np.asarray(to_host(self.b.cp_recovery_variable_u)).copy()

    # ---- train BOTH edges: block 1 (existing recipe, unchanged) then block 2 (NEW, symmetric) ----
    def train_129(self, n_episodes=N_EPISODES_129):
        traj = [dict(ep=0, w_gen=self._wmean("trained_block->provgen"), w_gen_o=self._wmean("other_blocks->provgen"),
                     w_perc=self._wmean("trained_block->provperc"), w_perc_o=self._wmean("other_blocks->provperc"))]
        for ep in range(n_episodes):
            self._hard_reset()
            drives = self._contradict_pairs() + [(self.ix["ctx_generated"], CTX_DRIVE_PA)]
            self._drive(drives, TRAIN_STEPS, learn=True, pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
        for ep in range(n_episodes):
            self._hard_reset()
            drives = self._confirm_pairs() + [(self.ix["ctx_perceived"], CTX_DRIVE_PA)]
            self._drive(drives, TRAIN_STEPS, learn=True, pre_pairs=self._cue_pre_pairs(), pre_steps=PRE_STEPS)
        traj.append(dict(ep=2 * n_episodes, w_gen=self._wmean("trained_block->provgen"),
                         w_gen_o=self._wmean("other_blocks->provgen"),
                         w_perc=self._wmean("trained_block->provperc"),
                         w_perc_o=self._wmean("other_blocks->provperc")))
        self.b.core_config.enable_hebbian_learning = False
        now = np.asarray(to_host(self.b.cp_connections.data))
        self.frozen_maxdrift = float(np.max(np.abs(now[self._noncross] - self._frozen_w0[self._noncross])))
        return traj

    # ---- ratio read: same protocol as amb_read, but returns the #129 divisive-normalization ratio too ----
    def amb_read_ratio(self, hold_surprise, band=None):
        ix = self.ix
        ep_idx = ix["episode"][self.ambig_pattern]
        margins, ratios, rates = [], [], {"prov_generated": 0.0, "prov_perceived": 0.0, "surprise": 0.0,
                                          "patient_expected": 0.0}
        for _ in range(N_READS):
            self._hard_reset()
            pairs = [(ep_idx, EPISODE_DRIVE_PA)]
            pre, pre_steps = None, 0
            if hold_surprise:
                pairs = pairs + self._contradict_pairs()
                pre, pre_steps = self._cue_pre_pairs(), PRE_STEPS
            read = {"gen": ix["prov_generated"], "perc": ix["prov_perceived"]}
            if band is not None:
                for r in rates:
                    read[r] = ix[r]
            acc = self._drive(pairs, RECALL_STEPS, read=read, pre_pairs=pre, pre_steps=pre_steps)
            g, p = acc["gen"], acc["perc"]
            margins.append(g - p)
            ratios.append(_dn_ratio(g, p))
            if band is not None:
                for r in rates:
                    rates[r] += acc.get(r, 0.0)
        out = {"margin": float(np.mean(margins)), "ratio": float(np.mean(ratios))}
        if band is not None:
            out["rates"] = {r: rates[r] / N_READS for r in rates}
        return out


def _f2_ratio(sep, floor):
    """F2 via the #129 divisive-normalization ratio, lesioning BOTH cross-edges together."""
    base_i = sep.amb_read_ratio(False, band=True)
    held_i = sep.amb_read_ratio(True)
    d_i = held_i["ratio"] - base_i["ratio"]
    data = np.asarray(to_host(sep.b.cp_connections.data)).copy()
    data[sep.masks["both_edges"]] = 0.0
    sep.b.cp_connections.data = sep.xp.asarray(data, dtype=sep.b.cp_connections.data.dtype)
    base_l = sep.amb_read_ratio(False)
    held_l = sep.amb_read_ratio(True)
    d_l = held_l["ratio"] - base_l["ratio"]
    shift_ok = (d_i > floor) and (abs(d_l) < F2_LESION_RATIO * abs(d_i))
    frac = attributable_to("F2(ratio) surprise-hold shift toward GENERATED = both cross-edges", d_i, d_l)
    return {"frac_attributable": (None if frac is None else float(frac)),
            "ratio_base_intact": base_i["ratio"], "ratio_held_intact": held_i["ratio"],
            "ratio_base_lesion": base_l["ratio"], "ratio_held_lesion": held_l["ratio"],
            "margin_base_intact": base_i["margin"], "margin_held_intact": held_i["margin"],
            "delta_intact": float(d_i), "delta_lesion": float(d_l),
            "rates_base_intact": base_i.get("rates", {}),
            "floor_used": float(floor),
            "shift_toward_generated": bool(shift_ok), "PASS": bool(shift_ok)}


def _f4_ratio(sep):
    """F4 MOAT/HONESTY via the ratio read (mirrors the original _f4 formula-for-formula)."""
    ix = sep.ix
    sep._hard_reset()
    silent = sep._drive(sep._contradict_pairs(), RECALL_STEPS,
                        read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]},
                        pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)
    silence_ratio = _dn_ratio(silent["gen"], silent["perc"])
    amb_base = sep.amb_read_ratio(False)["ratio"]
    clear_pat = sep.sp_organ.patterns["perceived"][0]
    ep_idx = ix["episode"][np.asarray(clear_pat, np.int64)]
    sep._hard_reset()
    clear_nohold = sep._drive([(ep_idx, EPISODE_DRIVE_PA)], RECALL_STEPS,
                              read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]})
    m_nohold = _dn_ratio(clear_nohold["gen"], clear_nohold["perc"])
    decision = max(abs(amb_base), abs(m_nohold), 1e-9)
    f4a_ok = bool(abs(silence_ratio) < F4A_FRAC * decision)
    sep._hard_reset()
    clear_held = sep._drive([(ep_idx, EPISODE_DRIVE_PA)] + sep._contradict_pairs(), RECALL_STEPS,
                            read={"gen": ix["prov_generated"], "perc": ix["prov_perceived"]},
                            pre_pairs=sep._cue_pre_pairs(), pre_steps=PRE_STEPS)
    m_wrong = _dn_ratio(clear_held["gen"], clear_held["perc"])
    same_sign = (m_wrong < 0) == (m_nohold < 0)
    retained = abs(m_wrong) >= F4B_RETAIN * abs(m_nohold)
    f4b_ok = bool(m_nohold < 0 and same_sign and retained)
    return {"silence_ratio": float(silence_ratio), "decision_scale": float(decision),
            "f4a_no_winner_from_silence": f4a_ok,
            "clear_nohold": float(m_nohold), "clear_wrong_hold": float(m_wrong),
            "f4b_clear_not_flipped": f4b_ok, "PASS": bool(f4a_ok and f4b_ok)}


def _f3_129(sep, traj, f2):
    rates = f2.get("rates_base_intact", {})
    band_pools = ("prov_generated", "prov_perceived")
    in_band = all(RATE_LO < rates.get(p, 0.0) < RATE_HI for p in band_pools) if rates else False
    grown_gen = traj[-1]["w_gen"]; grown_perc = traj[-1]["w_perc"]
    bounded = (grown_gen <= HMAX_129) and (grown_perc <= HMAX_129)
    alive = rates.get("prov_generated", 0.0) > RATE_LO or rates.get("prov_perceived", 0.0) > RATE_LO
    return {"rates": rates, "in_band": bool(in_band), "grown_gen": float(grown_gen), "grown_perc": float(grown_perc),
            "bounded_by_hmax": bool(bounded), "pool_alive": bool(alive),
            "PASS": bool(in_band and bounded and alive)}


def _emergence_129(traj, frozen_maxdrift):
    g_gen = traj[-1]["w_gen"]; g_gen_o = traj[-1]["w_gen_o"]
    g_perc = traj[-1]["w_perc"]; g_perc_o = traj[-1]["w_perc_o"]
    grew_gen = g_gen > 5 * W0
    grew_perc = g_perc > 5 * W0
    specific_gen = bool(abs(g_gen_o - W0) < OTHER_BLOCK_DRIFT_MAX)
    specific_perc = bool(abs(g_perc_o - W0) < OTHER_BLOCK_DRIFT_MAX)
    return {"trajectory": traj,
            "final_weight_gen_trained": float(g_gen), "final_weight_gen_other": float(g_gen_o),
            "final_weight_perc_trained": float(g_perc), "final_weight_perc_other": float(g_perc_o),
            "grew_from_near_zero": bool(grew_gen and grew_perc),
            "other_blocks_stayed_near_seed": bool(specific_gen and specific_perc),
            "frozen_weight_maxdrift": float(frozen_maxdrift), "no_corruption": bool(frozen_maxdrift < 1e-6),
            "PASS": bool(grew_gen and grew_perc and specific_gen and specific_perc and frozen_maxdrift < 1e-6)}


def _migration_invariant_129(seed, sep, sp_battery_lesioned):
    from research.runners.onebrain_merge_framework import _source_prov_organ
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
    pool0 = _build_pool_plain_129(seed)
    pool0.ensure_built()
    sp0 = _source_prov_organ(seed, pool0); sp0.ensure_built()

    def edge_map(pool):
        coo = pool.bridge.cp_connections.tocoo()
        r = to_host(coo.row); c = to_host(coo.col); d = to_host(coo.data)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, d)}
    k0 = edge_map(pool0)
    k1 = edge_map(sep.pool)
    xmask = sep.masks["both_edges"]
    coo1 = sep.b.cp_connections.tocoo()
    r1 = to_host(coo1.row); c1 = to_host(coo1.col)
    xpairs = set(zip((int(x) for x in r1[xmask]), (int(x) for x in c1[xmask])))
    k1_base = {kk: vv for kk, vv in k1.items() if kk not in xpairs}
    struct_identical = bool(set(k1_base.keys()) == set(k0.keys()))
    base = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = sp0.brain.recall(sp0.patterns[prov][i])
            base.append(rec["rate_perceived"] - rec["rate_generated"])
    maxerr = float(np.max(np.abs(np.asarray(base) - np.asarray(sp_battery_lesioned)))) if base else 0.0
    return {"base_connectivity_structurally_identical": struct_identical,
            "sp_battery_maxerr": maxerr,
            "PASS": bool(struct_identical and maxerr < 0.05)}


def run_seed(seed, floor):
    t0 = time.time()
    sep = SurpriseEpisodic129Pool(seed)
    traj = sep.train_129()
    emg = _emergence_129(traj, sep.frozen_maxdrift)
    f1 = _f1(sep)
    f4 = _f4_ratio(sep)                              # F4 BEFORE F2 (F2 lesions both edges in place)
    f2 = _f2_ratio(sep, floor)                        # F2 lesions BOTH cross-edges at its end
    sep._hard_reset()
    from research.runners._laneC_source_provenance_opponent_derisk import PROVENANCES, N_PAIRS
    sp_les = []
    for prov in PROVENANCES:
        for i in range(N_PAIRS):
            rec = sep.sp_organ.brain.recall(sep.sp_organ.patterns[prov][i])
            sp_les.append(rec["rate_perceived"] - rec["rate_generated"])
    f3 = _f3_129(sep, traj, f2)
    mig = _migration_invariant_129(seed, sep, sp_les)
    go = bool(f1["PASS"] and f2["PASS"] and f3["PASS"] and f4["PASS"] and emg["PASS"] and mig["PASS"])
    return {"seed": int(seed), "PASS": go, "elapsed_s": round(time.time() - t0, 1),
            "cue_concept": sep.cue_c, "assert_concept": sep.assert_cp,
            "emergence": emg, "F1": f1, "F2": f2, "F3": f3, "F4": f4, "lesion_recovers_migration": mig}


def _agg(runs):
    def frac(key):
        return sum(1 for r in runs if r[key.split(".")[0]][key.split(".")[1]]) if "." in key else 0
    keys = ["F1.PASS", "F2.PASS", "F3.PASS", "F4.PASS", "emergence.PASS", "lesion_recovers_migration.PASS"]
    return {k: f"{frac(k)}/{len(runs)}" for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1 seed indicator")
    ap.add_argument("--floor", type=float, default=None,
                    help="F2 intact-ratio floor. If omitted, uses a calibration run on the non-canonical seed 7"
                         " (frozen BEFORE reading the requested --seeds), mirroring the original runner's own"
                         " CUE_PA/HMAX/N_EPISODES calibration discipline.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    floor = args.floor
    calib = None
    if floor is None:
        print("[calibration] no --floor given -- running NON-CANONICAL seed 7 to set F2_INTACT_FLOOR (ratio "
              "units) BEFORE reading any of the requested seeds.", flush=True)
        calib_run = run_seed(7, floor=0.0)   # floor=0.0 placeholder; only delta_intact is read from this run
        d_i_calib = calib_run["F2"]["delta_intact"]
        # Pre-registered rule (frozen before looking at the canonical seeds): the floor is 25% of the
        # calibration seed's own intact shift, rounded to 3 significant figures -- a modest fraction, not a
        # value chosen to force a pass on the canonical set (which has not been read yet at this point).
        floor = round(abs(d_i_calib) * 0.25, 4) if d_i_calib else 0.02
        calib = {"seed": 7, "delta_intact_ratio": d_i_calib, "floor_rule": "0.25 * |calibration delta_intact|",
                "floor_frozen": floor}
        print(f"[calibration] seed 7: delta_intact(ratio)={d_i_calib:+.4f} delta_lesion(ratio)="
              f"{calib_run['F2']['delta_lesion']:+.4f} frac_attrib={calib_run['F2']['frac_attributable']} "
              f"-> F2_INTACT_FLOOR(ratio) FROZEN at {floor}", flush=True)

    runs = []
    for s in seeds:
        r = run_seed(s, floor)
        runs.append(r)
        f2 = r["F2"]
        print(f"[seed {s}] {'GO' if r['PASS'] else 'no'} ({r['elapsed_s']}s) block(c={r['cue_concept']},"
              f"c'={r['assert_concept']}) | "
              f"emerge w_gen={r['emergence']['final_weight_gen_trained']:.2f} "
              f"w_perc={r['emergence']['final_weight_perc_trained']:.2f} | "
              f"F1={r['F1']['PASS']} | "
              f"F2(ratio) delta={f2['delta_intact']:+.4f}(les {f2['delta_lesion']:+.4f}) "
              f"frac={f2['frac_attributable']} ={f2['PASS']} | F3={r['F3']['PASS']} F4={r['F4']['PASS']} "
              f"mig={r['lesion_recovers_migration']['PASS']}", flush=True)

    n_go = sum(r["PASS"] for r in runs)
    agg = _agg(runs)
    all_go_raw = (n_go == len(runs)) and not args.smoke

    dec, preconditions = None, []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("onebrain_surprise_episodic_129construction")
        Vd.require("f2_lesion_removes_shift", 1 if all(
            abs(r["F2"]["delta_lesion"]) < F2_LESION_RATIO * max(abs(r["F2"]["delta_intact"]), 1e-9)
            for r in runs) else 0, expect=lambda x: x >= 1,
            note="the F2 ratio-shift must VANISH under lesion of BOTH edges, or it is a confound (the crux control -- this is the EXACT precondition that failed 5/6 on the raw-margin, single-edge construction)")
        Vd.require("migration_byte_identity", 1 if all(r["lesion_recovers_migration"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1)
        Vd.require("emergence_grew_from_near_zero", 1 if all(r["emergence"]["PASS"] for r in runs) else 0,
                   expect=lambda x: x >= 1)
        Vd.require("moat_no_winner_from_silence", 1 if all(r["F4"]["f4a_no_winner_from_silence"] for r in runs) else 0,
                   expect=lambda x: x >= 1)
        Vd.require("anti_cheat_random_assignment", 1 if len(set((r["cue_concept"], r["assert_concept"])
                   for r in runs)) > 1 else 0, expect=lambda x: x >= 1)
        dec = Vd.decide(all_go_raw, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    verdict_status = dec.get("status") if dec else None
    all_go = all_go_raw if dec is None else bool(dec.get("go"))
    if verdict_status == "UNDEFINED":
        tag = "UNDEFINED"
    elif args.smoke:
        tag = "SMOKE-GO (1-seed indicator)" if n_go == len(runs) else "NO-GO/PARTIAL"
    else:
        tag = "GO" if all_go_raw else "NO-GO/PARTIAL"
    verdict = (f"{tag} -- #129's TWO-cross-edge/opponent-ratio construction applied to surprise->source_"
               f"provenance: {n_go}/{len(runs)} seeds pass ALL of F1+F2(ratio, both-edge lesion)+F3+F4+emergence+"
               f"lesion-recovers-migration. Per-arm: {agg}. "
               + (f" UNDEFINED, NOT a validated negative: {len(dec.get('undefined_reasons', []))} precondition(s) "
                  f"unmet -- {'; '.join(dec.get('undefined_reasons', []))}."
                  if verdict_status == "UNDEFINED" else ""))

    payload = {"probe": "onebrain_surprise_episodic_129construction", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(runs), "per_arm": agg, "seeds": seeds,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "preconditions": preconditions, "calibration": calib,
               "config": {"W0": W0, "cross_edge_hebbian_lr": CROSS_EDGE_LR, "hebbian_max_weight": HMAX_129,
                          "n_episodes_per_edge": N_EPISODES_129, "dn_sigma": DN_SIGMA,
                          "f2_intact_floor_ratio": floor,
                          "f2_lesion_ratio": F2_LESION_RATIO},
               "mechanism": ("TWO declarative CrossEdge rows on the SAME merged [surprise, source_provenance] "
                             "pool: (1) surprise->prov_generated (unchanged, CONTRADICT-trained, the edge that "
                             "already passes F1/F3/F4/emergence/migration 6/6); (2) NEW patient_expected->"
                             "prov_perceived (CONFIRM-trained, the direct structural mirror). Read as the #129 "
                             "divisively-normalized opponent ratio d=(r_gen-r_perc)/(r_gen+r_perc); F2 lesions "
                             "BOTH edges together."),
               "honest_architecture_note": ("patient_expected (edge 2's presynaptic pool) does NOT cleanly "
                                            "discriminate CONFIRM from CONTRADICT the way #129's ctx_perceived/"
                                            "ctx_generated lines do -- it fires whenever cue is driven, in BOTH "
                                            "conditions, unlike #129's cleanly mutually-exclusive context lines. "
                                            "This is a genuine, declared architecture mismatch with #129's "
                                            "construction, not a hidden one; the numeric result below is the "
                                            "honest empirical answer to whether it matters."),
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[SURP->EPISODIC 129-CONSTRUCTION] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
