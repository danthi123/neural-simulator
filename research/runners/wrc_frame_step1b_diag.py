#!/usr/bin/env python3
"""Step 1b engagement diagnosis (PACKET docs/drafts/word-role-frame/PACKET.md;
RULING wrc-frame-1-1024 item (2): DIAGNOSE-ONLY, no code change).

Diagnoses why T->role never learned (0/2736, frozen at init) and why the
discharge pool never fired, on dev seeds, cheaply, with file:line answers:

  (a) Is T->role in the plastic mask, and does the learning rule's update
      touch it inside _train? update-step count + sum |dw| on T->role vs
      word->role.
  (b) Is wrp_trole_plastic open at the steps where T->role co-fires (gate
      read-back per presentation + every set_plasticity_gate call)? does
      enforce_plastic_mask_in_hebbian or a learning-rate field exclude the
      new block?
  (c) What drives the discharge pool, and is that drive below its cells'
      rheobase? (arriving pA vs a measured I_rh titration).

ANSWER RULES WRITTEN BEFORE THE RUN (also printed into the output JSON):

  (a) REACHED            : sum|dw| on the T->role range over training > 0.
     NOT-REACHED-GAIN    : gain over the range reads 0.0 at training
                           steps (rule multiplies by gain ->
                           sim/bridge.py:10341-10342, zero before mask).
     NOT-REACHED-ACTIVESET: gain > 0 but the causal active set
                           (prev[t-1] & post[t], bridge.py:10333-10336)
                           selects 0 T->role edges during training.
     MASK-EXCLUDED        : cp_synapse_plastic_mask False on the range AND
                           enforcement actually active at run time
                           (bridge.py:1259-1271, config.py:472 default
                           False; read back env+cfg, do not assume).
  (b) OPEN  : registered synapses > 0 AND gain over the range > 0 at every
              read-back, with the call log showing the open before training
              and the only close at word_role_parser_organ.py:489 after it.
              CLOSED otherwise -> name the closer / missing registration.
  (c) SUB-RHOBASE    : max arriving drive (gate spikes at t-1 x W_GD,
                       organ:404) < measured I_rh of the discharge pool.
     NOT-SUB-RHOBASE : drive >= I_rh with 0 spikes (report other inputs)
                       or spikes > 0 (then first_discharge_step=-1 was an
                       instrument error: say so).

Fix-site tags (for the later ruling; NO fix is executed here):
  FIX-SITE research/  -> driver-fixable next round on research/wrc-frame.
  FIX-SITE sim/       -> needs a Claude ruling to lift the no-sim rule.

Measurement-only: wraps methods from the research side (no sim/, no organ
edits, no constant changed). Dev seeds only. Backend numpy.
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):
    _HERE = os.path.dirname(_HERE)
    if os.path.isdir(os.path.join(_HERE, "research")):
        break
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

from research.runners import word_role_parser_organ as W  # noqa: E402
from research.runners import wrc_frame_step1 as S1  # noqa: E402
from research.runners import declare_output  # noqa: E402
from tools.lab import assert_backend  # noqa: E402
from tools.lab import attributable_to
from tools.verdict import Verdict  # noqa: E402

RULES = {
    "a": "REACHED iff sum|dw| on T->role over training > 0; else "
         "NOT-REACHED-GAIN (gain 0.0 at training steps, bridge.py:10341) | "
         "NOT-REACHED-ACTIVESET (gain>0 but causal set empty, "
         "bridge.py:10333-10336) | MASK-EXCLUDED (mask False on range AND "
         "enforcement active at run, bridge.py:1259-1271).",
    "b": "OPEN iff registered trole synapses>0 and gain over the range>0 at "
         "every read-back, call log shows open before training and only the "
         "close at word_role_parser_organ.py:489 after it; else CLOSED with "
         "the closer / missing registration named.",
    "c": "SUB-RHOBASE iff max arriving drive (gate spikes t-1 x W_GD, "
         "organ:404) < measured I_rh; NOT-SUB if drive>=I_rh with 0 spikes "
         "(report other input) or spikes>0 (instrument error, say so).",
}
FILELINE = {
    "gate_open_site": "word_role_parser_organ.py:421 (_ensure), :481 (_train)",
    "gate_close_site": "word_role_parser_organ.py:489",
    "gate_write_site": "sim/bridge.py:5467-5491 (bound check :5490)",
    "rule_site": "sim/bridge.py:10333-10336 causal; :10341 gain; :10343 mask",
    "mask_enforce": "sim/bridge.py:1259-1271; sim/config.py:472 default False",
    "trole_plan": "word_role_parser_organ.py:401-403 (plastic=True, "
                  "plasticity_gate=wrp_trole_plastic)",
    "disch_plan": "word_role_parser_organ.py:404-408 (W_GD, W_DT)",
    "coo_site": "sim/bridge.py:10236-10238 (cached full COO)",
}


class Recorder:
    """Accumulates the (a)/(b)/(c) evidence. Measurement-only."""

    def __init__(self):
        self.phase = "SETUP"
        self.gate_calls = []           # (name, value, trole_gain_min, gain_nn)
        self.present = []              # per-presentation rows (train)
        self.per_step = []             # tier-1 only: per-step rows (train)
        self.drive = {"TRAIN": {"max_pA": 0.0, "n_steps": 0, "spikes": 0},
                      "PROBE": {"max_pA": 0.0, "n_steps": 0, "spikes": 0}}
        self.pres_i = 0
        self.epoch_i = 0
        self.ranges = {}               # name -> {"idx": ndarray, "size": int}
        self.weight0 = {}
        self.weight1 = {}
        self.registration = {}         # gate name -> {size, max, gain_nn}
        self.enforce = None
        self.rh = None  # measured I_rh of the discharge pool (pA)
        self.rh_sweep = []
        self.notes = []
        self.mask_trole_false = False
        self.drive_geo = {}
        self.per_step_cap = 40000      # tier-1 safety cap on logged rows


REC = Recorder()


def _np(a):
    return np.asarray(a)


def wrap_bridge(bridge, coo_row, coo_col, organ):
    """Attach measurement wraps to the freshly built bridge (post-inject)."""
    # ---- registration / ranges (static read, pre-train) -------------------
    reg = getattr(bridge, "_plasticity_gate_to_synapses", {})
    for gname in (W.GATE_TROLE_PLASTIC, W.GATE_WORD_PLASTIC):
        idx = np.asarray(reg.get(gname, []), dtype=np.int64)
        REC.ranges[gname] = {"idx": idx, "size": int(idx.size),
                             "max": int(idx.max()) if idx.size else -1}
    gain = getattr(bridge, "cp_plasticity_rate_gain", None)
    gn = int(gain.shape[0]) if gain is not None else -1
    for gname, r in REC.ranges.items():
        REC.registration[gname] = {
            "registered": r["size"],
            "gain_nn": gn,
            "indices_fit_gain": bool(r["size"] == 0 or r["max"] < gn),
        }
    # expected trole block size from the organ itself (plan:401-403)
    try:
        n_trace = int(len(organ.all_trace))
        n_role = int(sum(len(organ.role_idx[r]) for r in W.ROLES))
        REC.registration["trole_expected_block"] = n_trace * n_role
    except Exception as exc:  # pragma: no cover - evidence only
        REC.notes.append("trole_expected_block failed: %r" % (exc,))
    # COO row/col slices per range (CSR order == cached COO order)
    idx_t = REC.ranges[W.GATE_TROLE_PLASTIC]["idx"]
    idx_w = REC.ranges[W.GATE_WORD_PLASTIC]["idx"]
    # discharge drive geometry: edges gate->discharge come from the plan;
    # recover via (row, col) pairs whose col is in discharge_idx
    dset = set(int(x) for x in organ.discharge_idx)
    dis_mask = np.array([int(c) in dset for c in coo_col], dtype=bool)
    # NOTE: mask over ALL nnz; w taken from current weights (fixed block)
    drive_rows = coo_row[dis_mask]
    drive_cols = coo_col[dis_mask]
    drive_w = np.asarray(bridge.cp_connections.data)[dis_mask].astype(np.float64)
    colpos = {int(c): i for i, c in enumerate(sorted(dset))}
    drive_pos = np.array([colpos[int(c)] for c in drive_cols], dtype=np.int64)
    n_d = len(dset)
    d_dev = np.asarray(sorted(dset), dtype=np.int64)
    REC.drive_geo = {"n_edges": int(dis_mask.sum()), "n_cells": n_d,
                     "w_min": float(drive_w.min()) if drive_w.size else 0.0,
                     "w_max": float(drive_w.max()) if drive_w.size else 0.0}

    o_idx_t = idx_t.copy()
    o_idx_w = idx_w.copy()
    # ---- gate call log ----------------------------------------------------
    orig_set_gate = type(bridge).set_plasticity_gate

    def set_gate(_s, name, value):
        out = orig_set_gate(_s, name, value)
        try:
            gmin = None
            g = getattr(bridge, "cp_plasticity_rate_gain", None)
            if g is not None and name in getattr(
                    bridge, "_plasticity_gate_to_synapses", {}):
                gi = np.asarray(
                    bridge._plasticity_gate_to_synapses[name], dtype=np.int64)
                if gi.size:
                    gmin = float(np.asarray(g[gi]).min())
            REC.gate_calls.append({"name": name, "value": float(value),
                                   "range_gain_min": gmin,
                                   "phase": REC.phase,
                                   "pres": REC.pres_i})
        except Exception as exc:  # pragma: no cover
            REC.notes.append("gate log failed: %r" % (exc,))
        return out

    type(bridge).set_plasticity_gate = set_gate

    # ---- per-step instrument (tier 1) ------------------------------------
    orig_step = type(bridge)._run_one_simulation_step

    # explicit row/col for the two ranges (CSR order)
    rt_row, rt_col = coo_row[idx_t], coo_col[idx_t]
    rw_row, rw_col = coo_row[idx_w], coo_col[idx_w]

    def step_wrap2(_s):
        prev = np.asarray(bridge.cp_prev_firing_states).astype(bool)
        out = orig_step(_s)
        if REC.phase == "TRAIN" and len(REC.per_step) < REC.per_step_cap:
            cur = np.asarray(bridge.cp_firing_states).astype(bool)
            act_t = int(np.count_nonzero(prev[rt_row] & cur[rt_col]))
            act_w = int(np.count_nonzero(prev[rw_row] & cur[rw_col]))
            g = bridge.cp_plasticity_rate_gain
            g_t = float(np.asarray(g[o_idx_t]).min()) if g is not None else -1.0
            # arriving drive into discharge cells from gate spikes at t-1
            dsum = prev[drive_rows]
            if dsum.any():
                contrib = dsum.astype(np.float64) * drive_w
                cells = np.bincount(drive_pos, weights=contrib, minlength=n_d)
                mx = float(cells.max())
            else:
                mx = 0.0
            slot = REC.drive[REC.phase]
            slot["max_pA"] = max(slot["max_pA"], mx)
            slot["n_steps"] += 1
            slot["spikes"] += int(np.count_nonzero(cur[d_dev]))
            REC.per_step.append({"k": len(REC.per_step), "pres": REC.pres_i,
                                 "act_trole": act_t, "act_word": act_w,
                                 "gain_trole_min": g_t, "drive_pA": mx})
        elif REC.phase == "PROBE":
            dsum = prev[drive_rows]
            if dsum.any():
                contrib = dsum.astype(np.float64) * drive_w
                cells = np.bincount(drive_pos, weights=contrib, minlength=n_d)
                mx = float(cells.max())
            else:
                mx = 0.0
            slot = REC.drive[REC.phase]
            slot["max_pA"] = max(slot["max_pA"], mx)
            slot["n_steps"] += 1
            slot["spikes"] += int(np.count_nonzero(
                np.asarray(bridge.cp_firing_states)[d_dev]))
        return out

    type(bridge)._run_one_simulation_step = step_wrap2

    # ---- per-presentation instrument (train) -----------------------------
    orig_present = type(organ)._present

    def present(_s, word, pos, steps, teach=None, count=True):
        if REC.phase != "TRAIN":
            return orig_present(_s, word, pos, steps, teach=teach, count=count)
        w0_t = np.asarray(bridge.cp_connections.data[o_idx_t]).copy()
        w0_w = np.asarray(bridge.cp_connections.data[o_idx_w]).copy()
        out = orig_present(_s, word, pos, steps, teach=teach, count=count)
        w1_t = np.asarray(bridge.cp_connections.data[o_idx_t]).copy()
        w1_w = np.asarray(bridge.cp_connections.data[o_idx_w]).copy()
        g = bridge.cp_plasticity_rate_gain
        g_t = float(np.asarray(g[o_idx_t]).min()) if g is not None else -1.0
        g_w = float(np.asarray(g[o_idx_w]).min()) if g is not None else -1.0
        REC.present.append({
            "pres": REC.pres_i, "word": word, "teach": teach,
            "epoch": REC.epoch_i,
            "sum_abs_dw_trole": float(np.abs(w1_t - w0_t).sum()),
            "sum_abs_dw_word": float(np.abs(w1_w - w0_w).sum()),
            "n_moved_trole": int(np.count_nonzero(w1_t != w0_t)),
            "n_moved_word": int(np.count_nonzero(w1_w != w0_w)),
            "gain_trole_min": g_t, "gain_word_min": g_w,
        })
        if not REC.weight0:
            REC.weight0 = {"trole": w0_t, "word": w0_w}
        REC.weight1 = {"trole": w1_t, "word": w1_w}
        REC.pres_i += 1
        return out

    type(organ)._present = present
    return {"orig": (orig_set_gate, orig_step, orig_present),
            "bridge": bridge}


def record_enforcement(bridge, organ):
    """(a) sub-question: is the plastic mask enforced at run time, and does
    it exclude the T->role range?  Read env + cfg + the mask itself."""
    REC.enforce = {
        "env": os.environ.get("BRAIN_ENFORCE_PLASTIC_MASK", ""),
        "cfg": bool(getattr(bridge.core_config,
                            "enforce_plastic_mask_in_hebbian", False)),
        "mask_built": getattr(bridge, "cp_synapse_plastic_mask",
                              None) is not None,
    }
    try:
        m = getattr(bridge, "cp_synapse_plastic_mask", None)
        if m is not None:
            gi = np.asarray(
                bridge._plasticity_gate_to_synapses.get(
                    W.GATE_TROLE_PLASTIC, []), dtype=np.int64)
            if gi.size:
                mv = np.asarray(m[gi])
                REC.mask_trole_false = bool(not mv.all())
    except Exception as exc:  # pragma: no cover - evidence only
        REC.notes.append("mask read-back failed: %r" % (exc,))


def rheobase_titration(organ, lo=0.0, hi=800.0, step=50.0, t_steps=150):
    """I_rh of the discharge pool: smallest step current giving >=1 spike
    in t_steps of an otherwise quiet network (cs-race lesson: never trust a
    drive constant)."""
    b, xp = organ._bridge, organ._xp
    ddev = xp.asarray(organ.discharge_idx)
    from sim.backend import to_host
    saved_phase = REC.phase
    REC.phase = "TITRATION"
    try:
        for I in np.arange(lo, hi + 1.0, step):
            W._reset_quiescent(b, xp, organ._snap)
            cur = xp.zeros(organ._n, dtype=xp.float32)
            cur[ddev] = xp.float32(I)
            b.cp_external_input_current[:] = cur
            spikes = 0
            for _ in range(t_steps):
                b._run_one_simulation_step()
                spikes += int(to_host(
                    b.cp_firing_states[ddev].astype(xp.float64).sum()))
            b.cp_external_input_current[:] = 0.0
            REC.rh_sweep.append({"I_pA": float(I), "spikes": int(spikes)})
            if spikes >= 1 and REC.rh is None:
                REC.rh = float(I)
        W._reset_quiescent(b, xp, organ._snap)
    finally:
        REC.phase = saved_phase
    return REC.rh


def diagnose(rec):
    """Apply the pre-written answer rules to the recorded evidence."""
    train = rec["train"]
    gate = rec["gate"]
    out = {"rules": RULES, "fileline": FILELINE}
    # ---- (b) gate ------------------------------------------------------
    trole_reg = rec["registration"].get(W.GATE_TROLE_PLASTIC, {})
    gmin_t = train.get("min_gain_trole", -1.0)
    gmin_present = [gmin_t] if gmin_t is not None else [-1.0]
    expected = float(rec.get("plast_gain_expected", 1.0))
    opens = [g for g in gate if g["name"] == W.GATE_TROLE_PLASTIC]
    open_before = any(g["value"] > 0 for g in opens)
    closes_after = [g for g in opens if g["value"] == 0.0]
    gain_ok = min(gmin_present) >= expected * 0.999
    registered_ok = bool(trole_reg.get("registered", 0) > 0)
    fits = bool(trole_reg.get("indices_fit_gain", False))
    expected_blk = rec["registration"].get("trole_expected_block", -1)
    block_ok = (trole_reg.get("registered", -1) == expected_blk)
    out["b"] = {
        "verdict": "OPEN" if (registered_ok and fits and block_ok
                              and open_before and gain_ok) else "CLOSED",
        "registered": trole_reg.get("registered"),
        "expected_block": expected_blk,
        "gain_nn": trole_reg.get("gain_nn"),
        "indices_fit_gain": fits,
        "open_before_training": open_before,
        "n_gate_calls": len(opens),
        "closes": closes_after,
        "train_gain_min": min(gmin_present),
        "expected_gain": expected,
        "enforcement": rec.get("enforce"),
    }
    # ---- (a) reach -----------------------------------------------------
    sum_t = train.get("sum_abs_dw_trole", 0.0)
    sum_w = train.get("sum_abs_dw_word", 0.0)
    steps_t = train.get("update_steps_trole", 0)
    steps_w = train.get("update_steps_word", 0)
    enfo = rec.get("enforce") or {}
    enforcement_active = bool(enfo.get("env")) or bool(enfo.get("cfg"))
    mask_excluded = bool(enforcement_active
                         and rec.get("mask_trole_false", False))
    if sum_t > 0:
        a_verdict = "REACHED"
    elif mask_excluded:
        a_verdict = "MASK-EXCLUDED"
    elif gmin_present and min(gmin_present) < expected * 0.5:
        a_verdict = "NOT-REACHED-GAIN"
    else:
        a_verdict = "NOT-REACHED-ACTIVESET"
    out["a"] = {
        "verdict": a_verdict,
        "sum_abs_dw_trole": sum_t,
        "sum_abs_dw_word": sum_w,
        "update_steps_trole": steps_t,
        "update_steps_word": steps_w,
        "n_presentations": train.get("n_presentations", 0),
        "n_moved_trole_final": train.get("n_moved_trole_final", 0),
        "n_moved_word_final": train.get("n_moved_word_final", 1),
        "active_set": train.get("active_set", {}),
        "enforcement_active": enforcement_active,
        "mask_excluded": mask_excluded,
        "mask_trole_all_plastic": bool(not rec.get("mask_trole_false", True)),
        "rule": "sim/bridge.py:10333-10336 causal, :10341 gain",
    }
    # ---- (c) discharge -------------------------------------------------
    rh = rec.get("rh")
    dmax = rec.get("drive_max", 0.0)
    spikes_seen = rec.get("drive_train", {}).get("spikes", 0)
    probe = rec.get("probe", {})
    parse_disch = probe.get("disch_spikes", 0)
    if rh is None:
        c_verdict = "NO-RHEOBASE-MEASURED"
    elif dmax < rh:
        c_verdict = "SUB-RHOBASE"
    elif (dmax >= rh) and (spikes_seen + parse_disch) == 0:
        c_verdict = "NOT-SUB-RHOBASE-NO-SPIKES"
    else:
        c_verdict = "FIRING"
    wgd = rec.get("W_GD", 200.0)
    out["c"] = {
        "verdict": c_verdict,
        "max_arriving_drive_pA": dmax,
        "I_rh_pA": rh,
        "W_GD": wgd,
        "drive_arith": ("max_gate_spikes_at_t1 * %s per edge "
                        "(organ:404); geo=%s" % (wgd, rec.get("drive_geo"))),
        "disch_spikes_train": rec.get("drive_train", {}).get("spikes", 0),
        "disch_spikes_probe": parse_disch,
        "rh_sweep": rec.get("rh_sweep"),
        "drive_train": rec.get("drive_train"),
        "drive_probe": rec.get("drive_probe"),
    }
    return out


def seed_probe(seed, tier, out_dir):
    t0 = time.time()
    # build + train + measure in one go, with wraps attached right after
    # construction. WordRoleParserOrgan.__init__ builds the bridge; .train()
    # is called separately by S1.seed_probe, so we replicate its construction
    # to interleave instrumentation.
    params, commit = W.commit_organ_params(0)

    origs_holder = {}

    # Wrap the CLASS-level _ensure so we attach the recorder the moment the
    # bridge exists (it is built lazily on the first _ensure inside _train).
    orig_ensure = W.WordRoleParserOrgan._ensure

    def ensure_wrap(self, *a, **kw):
        out = orig_ensure(self, *a, **kw)
        if not origs_holder and self._bridge is not None:
            coo = self._bridge._get_cached_coo()
            origs_holder.update(wrap_bridge(
                self._bridge, np.asarray(coo.row, dtype=np.int64),
                np.asarray(coo.col, dtype=np.int64), self))
        return out

    W.WordRoleParserOrgan._ensure = ensure_wrap
    try:
        organ = W.WordRoleParserOrgan(sorted(S1.FACTS), seed, params, commit,
                                      frame=W.frame_organ_params())
        # tier gating: enable per-step rows only for the full tier
        if tier != "full":
            REC.per_step_cap = 0
        REC.phase = "TRAIN"
        organ.train()
        REC.phase = "POST-TRAIN"
        record_enforcement(organ._bridge, organ)
        # ---- parse-phase probe: the two step-1 measurements re-run --------
        REC.phase = "PROBE"
        items_a = S1.run_frame(organ, [(w, None) for w in S1.SINGLE_ROLE])
        items_b = S1.run_frame(organ,
                               [(w, None) for w in S1.SINGLE_ROLE],
                               frame_lesion=True)
        probe = {"cofire": {}, "disch_spikes": 0, "disch_spikes_lesion": 0,
                 "gain_own": {}}
        for wo, wl in zip(items_a, items_b):
            probe["cofire"][wo["word"]] = int(wo["cofire_steps"])
            probe["disch_spikes"] += 0 if wo["first_discharge_step"] < 0 else 1
            probe["disch_spikes_lesion"] += (0 if wl["first_discharge_step"] < 0
                                             else 1)
            g = S1.own_gain(wo["role_counts"], wl["role_counts"], wo["word"])
            if g is not None:
                probe["gain_own"][wo["word"]] = g
        gains = list(probe["gain_own"].values())
        probe["gain_mean"] = float(np.mean(gains)) if gains else None
        probe["cofire_any"] = any(v >= S1.COFIRE_MIN
                                  for v in probe["cofire"].values())
        # Genuine treatment/control attribution: whose discharge-firing at
        # parse is the FRAME's (intact vs frame lesion)?
        probe["frame_attribution"] = float(attributable_to(
            "discharge-firing words: frame intact vs frame lesion",
            probe["disch_spikes"], probe["disch_spikes_lesion"]))
        REC.phase = "POST-PROBE"
        # ---- rheobase ----------------------------------------------------
        rh = rheobase_titration(organ)
        # ---- restore class methods ---------------------------------------
        (ob_set, ob_step, ob_pres) = origs_holder["orig"]
        type(organ._bridge).set_plasticity_gate = ob_set
        type(organ._bridge)._run_one_simulation_step = ob_step
        type(organ)._present = ob_pres
        W.WordRoleParserOrgan._ensure = orig_ensure
    finally:
        W.WordRoleParserOrgan._ensure = orig_ensure

    # ---- train-phase aggregates -----------------------------------------
    present = REC.present
    per_step = REC.per_step
    steps_t = sum(1 for r in per_step if r["act_trole"] > 0)
    steps_w = sum(1 for r in per_step if r["act_word"] > 0)
    active = {
        "per_step_logged": len(per_step),
        "steps_with_trole_active": steps_t,
        "steps_with_word_active": steps_w,
        "max_trole_active_edges": max((r["act_trole"] for r in per_step),
                                      default=0),
        "max_word_active_edges": max((r["act_word"] for r in per_step),
                                     default=0),
        "mean_trole_active": (float(np.mean([r["act_trole"] for r in per_step]))
                              if per_step else None),
        "mean_word_active": (float(np.mean([r["act_word"] for r in per_step]))
                             if per_step else None),
    }
    train_agg = {
        "n_presentations": len(present),
        "disch_spikes_train": REC.drive.get("TRAIN", {}).get("spikes", 0),
        "sum_abs_dw_trole": float(sum(p["sum_abs_dw_trole"] for p in present)),
        "sum_abs_dw_word": float(sum(p["sum_abs_dw_word"] for p in present)),
        "n_moved_trole_final": (int(np.count_nonzero(
            REC.weight1.get("trole", []) != REC.weight0.get("trole", [])))
            if REC.weight0 else 0),
        "n_moved_word_final": (int(np.count_nonzero(
            REC.weight1.get("word", []) != REC.weight0.get("word", [])))
            if REC.weight0 else 0),
        "update_steps_trole": steps_t,
        "update_steps_word": steps_w,
        "active_set": active,
        "min_gain_trole": min((p["gain_trole_min"] for p in present),
                              default=-1.0),
        "min_gain_word": min((p["gain_word_min"] for p in present),
                             default=-1.0),
        "sum_drive_max_train": REC.drive.get("TRAIN", {}).get("max_pA", 0.0),
    }
    rec = {
        "seed": int(seed), "tier": tier,
        "frame_built": bool(organ.f is not None),
        "plast_gain_expected": float(organ.f.get("PLAST_GAIN", 1.0))
        if organ.f else 1.0,
        "W_GD": float(organ.f.get("W_GD", 200.0)) if organ.f else 200.0,
        "nt": int(organ.f.get("NT", 4)) if organ.f else 4,
        "registration": REC.registration,
        "gate": REC.gate_calls,
        "train": train_agg,
        "train_present_head": present[:3],
        "probe": probe,
        "rh": rh,
        "rh_sweep": REC.rh_sweep,
        "drive_geo": REC.drive_geo,
        "drive_max": float(max(REC.drive.get("TRAIN", {}).get("max_pA", 0.0),
                               REC.drive.get("PROBE", {}).get("max_pA", 0.0))),
        "drive_train": REC.drive.get("TRAIN"),
        "drive_probe": REC.drive.get("PROBE"),
        "enforce": REC.enforce,
        "mask_trole_false": REC.mask_trole_false,
        "per_step_head": per_step[:5],
        "notes": REC.notes,
        "wall_s": round(time.time() - t0, 1),
    }
    ans = diagnose(rec)
    rec["answers"] = ans
    path = os.path.join(out_dir, "seed%d_%s.json" % (seed, tier))
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True, default=float)
    declare_output(path)
    print("seed %d [%s]: (a)=%s dw_t=%.4g dw_w=%.4g steps_t=%d/%d | "
          "(b)=%s gain_t=%s reg=%s | (c)=%s drive=%.1fpA I_rh=%s | %s s"
          % (seed, tier, ans["a"]["verdict"], ans["a"]["sum_abs_dw_trole"],
             ans["a"]["sum_abs_dw_word"], ans["a"]["update_steps_trole"],
             train_agg["active_set"]["per_step_logged"], ans["b"]["verdict"],
             train_agg["min_gain_trole"],
             rec["registration"].get(W.GATE_TROLE_PLASTIC, {}).get("registered"),
             ans["c"]["verdict"], ans["c"]["max_arriving_drive_pA"], rh,
             rec["wall_s"]))
    return rec


def reset_recorder():
    global REC
    REC = Recorder()
    return REC


def main(argv=None):
    ap = argparse.ArgumentParser(description="wrc-frame step 1b diagnosis")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--tier", choices=("full", "light"), default="full",
                    help="full = per-step instrument; light = aggregates only")
    ap.add_argument("--out",
                    default="research/findings/raw/_wrc_frame/step1b")
    args = ap.parse_args(argv)
    backend = assert_backend("numpy", note="wrc-frame step 1b diagnosis")
    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
    out = args.out
    os.makedirs(out, exist_ok=True)
    recs = []
    for seed in seeds:
        reset_recorder()
        recs.append(seed_probe(seed, args.tier, out))
    summary = {
        "tier": args.tier, "seeds": seeds, "backend": backend,
        "device": "cpu",
        "rules": RULES, "fileline": FILELINE,
        "answers_per_seed": {r["seed"]: r["answers"] for r in recs},
        "wall_s_total": round(sum(r["wall_s"] for r in recs), 1),
        "provisional": ("agent-filed diagnosis per RULING wrc-frame-1-1024 "
                        "(2): DIAGNOSE-ONLY, no code changed"),
    }
    v = Verdict("wrc-frame step 1b engagement diagnosis (evidence pre-"
                "written; no bar)")
    v.require("seeds ran", len(recs) == len(seeds))
    v.require("backend numpy", backend == "numpy")
    v.require("frame stage built", all(r["frame_built"] for r in recs))
    ans_ok = all(set(r.get("answers", {})) == {"a", "b", "c"}
                 and all("verdict" in r["answers"][k] for k in ("a", "b", "c"))
                 for r in recs)
    rh_ok = all(r.get("rh") is not None for r in recs)
    dec = v.decide(go=bool(ans_ok and rh_ok), verbose=False)
    summary.update(dec)
    sp = os.path.join(out, "summary_%s.json" % args.tier)
    with open(sp, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=float)
    declare_output(sp)
    for r in recs:
        a = r["answers"]
        print("ANSWER seed %d: (a) %s | (b) %s | (c) %s"
              % (r["seed"], a["a"]["verdict"], a["b"]["verdict"],
                 a["c"]["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())