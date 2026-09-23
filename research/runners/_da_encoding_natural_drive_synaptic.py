"""DA-GATED ENCODING under a NATURAL drive, v3: the DA effect acts THROUGH SYNAPSES, each arm in a FRESH PROCESS.

WHY v3 (the 2026-09-23 adversarial review of v2, three required fixes):
  1. v2's 24 h outcome was decided by a HOST RULE: `TagCaptureLedger.observe_da` compared the DA scalar to 0.62 and a
     per-block exp() factor erased every uncaptured block, so G3/G4/G5 followed arithmetically from G1. v3 uses
     `webapp.da_tag_capture.SynapticTagCaptureLedger`: the brain's SNc DA level is broadcast onto the spiking D1
     (`write_gain`) population, whose measured firing-rate excess drives a cell-wide PRP pool; per-synapse tags are
     set by each synapse's own early-LTP amplitude; each synapse's bistable late-phase variable z_k decides its own
     fate. No DA-vs-number compare exists anywhere in the capture path. The ODE constants are pre-registered; the one
     calibrated constant (gamma) is fixed from a 5-min Go-boundary protocol with NO brain data (see the module).
  2. v2's robustness band was near-empty (tau_e points cannot fail at a 24 h read; beta=2.0 was UNDEFINED and silently
     excluded). v3's capture band perturbs only constants that MOVE the capture boundary, each with a recorded
     fail-ability precondition (its own critical D1 activation differs from the primary's by >= 10 % and lies inside
     the population's range), and an UNDEFINED band point FAILS G8 (never excluded). beta points are reported
     separately: beta does not move the capture boundary, so it cannot test the DA claim.
  3. v2's arms depended on run order (the lazily-built spiking gain reader drew OU noise from the global RNG, so the
     first arm in a process got different write gains). v3 runs EVERY arm in its own fresh subprocess AND resets the
     process state at arm start (`_reset_arm_state`), logs the gain the store ACTUALLY used (v2 logged a separate peek
     read), and re-runs one arm in a second fresh process as a determinism replicate (G0).

Run one seed (numpy CPU; always under memcap):
  bash tools/memcap.sh 6 -- .venv/bin/python -u -m research.runners._da_encoding_natural_drive_synaptic \
      --seed 42 --workers 3 --out research/findings/raw/_da_encoding_natural_drive_v3/seed42.json
Selftest (instrument checks):   ... --selftest
Aggregate:                      ... --aggregate research/findings/raw/_da_encoding_natural_drive_v3
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from research.runners._da_encoding_natural_drive_persistence import (  # noqa: E402
    BASE_ENV, DELAY_H, FACTS, IMMEDIATE_H, SEEDS, TURN_H, VOCAB, _counts, _env, _recall, _ShimChat, conversation,
    da_trace)

COMPOSER_D = 128                    # the production composer size (v2 addendum); unchanged
D1_READER_SEED = 42                 # the production write-gain reader seed (`_leaf_gain` calls with the default 42)

# ── kernel points. PRIMARY = the module's pre-registered constants. gamma is calibrated per process (deterministic).
PRIMARY = {"name": "primary", "beta": 1.0, "gamma_mult": 1.0, "tau_p": 1.0, "tau_tag": 1.5, "tau_z": 0.5,
           "tau_e": 1.5}


def _pt(name, **kw):
    d = dict(PRIMARY)
    d.update(kw)
    d["name"] = name
    return d


BAND_CAPTURE = [_pt("gamma_x0.5", gamma_mult=0.5), _pt("gamma_x2", gamma_mult=2.0),
                _pt("tau_p_x0.5", tau_p=0.5), _pt("tau_p_x2", tau_p=2.0),
                _pt("tau_tag_x0.5", tau_tag=0.75), _pt("tau_tag_x2", tau_tag=3.0),
                _pt("tau_z_x0.5", tau_z=0.25), _pt("tau_z_x2", tau_z=1.0)]
BAND_BETA = [_pt("beta_0.67", beta=0.67), _pt("beta_2.0", beta=2.0)]
ARMS_PRIMARY = ["intact", "lesion_da_encoding", "lesion_capture", "companion_off", "lesion_novelty"]
ARMS_BAND = ["intact", "lesion_da_encoding"]
ARM_ENV = {"intact": {}, "lesion_da_encoding": {"BRAIN_DA_ENCODING_LESION": "1"},
           "lesion_capture": {"BRAIN_DA_CAPTURE_LESION": "1"}, "companion_off": {}, "lesion_novelty": {}}
FAILABILITY_MIN_REL_SHIFT = 0.10    # a band point must move the critical D1 activation by >= 10 %
MIN_INFORMATIVE_BAND_POINTS = 4


def _build_composer(seed):
    from research.runners.one_brain_composer import OneBrainComposer
    return OneBrainComposer(seed=seed, D=COMPOSER_D, vocab=list(VOCAB), k_max=16)


def _reset_arm_state(seed):
    """Make an arm independent of anything that ran before it in this process: drop the cached spiking readers (their
    post-build RNG stream is what made v2's first arm different) and reseed every global RNG."""
    from research.runners import _da_write_gain_spiking_derisk as W
    W._READERS.clear()
    np.random.seed(int(seed) % (2 ** 32))
    random.seed(int(seed))


def _gamma_for(d1, point):
    from webapp.da_tag_capture import calibrate_gamma
    g0 = calibrate_gamma(d1.a_go)           # at the MODULE's primary tau constants (a priori operating point)
    return g0, g0 * float(point["gamma_mult"])


def run_arm(seed, condition, arm, trace, point):
    """One fresh brain: tell the conversation, idle-tick homeostasis, recall at +1 min and at +24 h."""
    from webapp import da_encoding_drives_chat as DAE
    from webapp.da_tag_capture import SpikingD1Activation, SynapticTagCaptureLedger
    _reset_arm_state(seed)
    env = dict(BASE_ENV)
    env.update(ARM_ENV[arm])
    with _env(env):
        d1 = SpikingD1Activation(reader_seed=D1_READER_SEED)       # built FIRST in every arm -> identical calibration
        gamma0, gamma = _gamma_for(d1, point)
        comp = _build_composer(seed)
        chat = _ShimChat(comp)
        DAE.install_encoding_gain(chat)
        used_gains = []
        fn = comp.encoding_gain_fn
        if fn is not None:
            def _logged(*a, **k):
                g = fn(*a, **k)
                used_gains.append(float(g))
                return g
            comp.encoding_gain_fn = _logged
        ledger = None if arm == "companion_off" else SynapticTagCaptureLedger(
            seed, gamma=gamma, d1=d1, beta=point["beta"], tau_early_h=point["tau_e"], tau_tag_h=point["tau_tag"],
            tau_p_h=point["tau_p"], tau_z_h=point["tau_z"])
        conv = conversation(condition)
        for ti, ((text, fi), tr) in enumerate(zip(conv, trace)):
            t = ti * TURN_H
            chat._last_da_drives = {"da_level": tr["da"]}
            if ledger is not None:
                ledger.observe_turn(t, TURN_H, tr["da"])
            if fi is not None:
                comp.store(*FACTS[fi]["lem"])
                if ledger is not None:
                    ledger.on_store(comp, t)
        t_end = len(conv) * TURN_H
        scales = DAE.apply_substrate_homeostasis(chat)
        if ledger is not None and scales:
            ledger.apply_homeostasis_scales(scales)
        if ledger is not None:
            ledger.advance(comp, t_end + IMMEDIATE_H)
        imm = _recall(comp)
        if ledger is not None:
            ledger.advance(comp, t_end + DELAY_H)
        late = _recall(comp)
        return {"arm": arm, "condition": condition, "point": point["name"], "point_params": point,
                "gamma_primary": gamma0, "gamma": gamma, "d1_a_go": d1.a_go, "d1_rate_tonic": d1.rate_tonic,
                "d1_rate_go": d1.rate_go, "d1_rate_hi": d1.rate_hi, "write_gains_used": used_gains,
                "homeostasis_scales": (None if scales is None else [float(s) for s in scales]),
                "turn_log": (None if ledger is None else ledger.turn_log),
                "p_max": (None if ledger is None else ledger.p_max),
                "blocks": (None if ledger is None else ledger.summary()),
                "immediate": imm, "immediate_counts": _counts(imm), "delayed_24h": late,
                "delayed_counts": _counts(late), "pid": os.getpid()}


# ── fresh-process arm execution ────────────────────────────────────────────────────────────────────────────────
def _arm_subprocess(spec):
    with tempfile.TemporaryDirectory() as td:
        sp, op = os.path.join(td, "spec.json"), os.path.join(td, "out.json")
        json.dump(spec, open(sp, "w"))
        cmd = [sys.executable, "-u", "-m", "research.runners._da_encoding_natural_drive_synaptic",
               "--arm-worker", sp, op, "--D", str(COMPOSER_D)]
        r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(op):
            raise RuntimeError("arm worker failed (%s/%s/%s): %s" % (spec["condition"], spec["arm"],
                                                                     spec["point"]["name"], r.stderr[-2000:]))
        return json.load(open(op))


def _arm_specs(seed, traces):
    specs = []
    for cond in ("neutral", "salient"):
        for arm in ARMS_PRIMARY:
            tr = traces[cond + ("_novelty_lesion" if arm == "lesion_novelty" else "")]
            specs.append({"seed": seed, "condition": cond, "arm": arm, "trace": tr, "point": PRIMARY})
        for pt in BAND_CAPTURE + BAND_BETA:
            for arm in ARMS_BAND:
                specs.append({"seed": seed, "condition": cond, "arm": arm, "trace": traces[cond], "point": pt})
    # G0 determinism replicate: the salient/intact primary arm again, in ANOTHER fresh process
    specs.append({"seed": seed, "condition": "salient", "arm": "intact", "trace": traces["salient"],
                  "point": PRIMARY, "replicate": True})
    return specs


def failability(d1_a_go):
    """Kernel-only (no brain data) fail-ability precondition for each capture-band point."""
    from webapp.da_tag_capture import calibrate_gamma, critical_activation
    g0 = calibrate_gamma(d1_a_go)
    a0 = critical_activation(g0, PRIMARY["tau_p"], PRIMARY["tau_tag"], PRIMARY["tau_z"])
    out = {"primary_a_crit": a0, "gamma_primary": g0, "points": {}}
    for pt in BAND_CAPTURE:
        a = critical_activation(g0 * pt["gamma_mult"], pt["tau_p"], pt["tau_tag"], pt["tau_z"])
        rel = None if (a is None or not a0) else abs(a / a0 - 1.0)
        out["points"][pt["name"]] = {"a_crit": a, "rel_shift": rel,
                                     "informative": bool(a is not None and 0.0 < a < 1.0 and rel is not None
                                                         and rel >= FAILABILITY_MIN_REL_SHIFT)}
    return out


def run_seed(seed, out, workers=3):
    t0 = time.time()
    res = {"seed": seed, "version": "v3-synaptic", "composer_D": COMPOSER_D, "turn_h": TURN_H, "delay_h": DELAY_H,
           "vocab": VOCAB, "primary": PRIMARY, "band_capture": BAND_CAPTURE, "band_beta": BAND_BETA,
           "traces": {}, "arms": []}
    for cond in ("neutral", "salient"):
        res["traces"][cond] = da_trace(seed, cond)
        res["traces"][cond + "_novelty_lesion"] = da_trace(seed, cond, novelty_lesion=True)
        print("[seed %d] trace %s done %.0fs" % (seed, cond, time.time() - t0), flush=True)
    rep = da_trace(seed, "salient")
    res["da_trace_repeat_identical"] = [r["da"] for r in rep] == [r["da"] for r in res["traces"]["salient"]]
    specs = _arm_specs(seed, res["traces"])
    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_arm_subprocess, s): s for s in specs}
        for f in cf.as_completed(futs):
            s = futs[f]
            a = f.result()
            a["replicate"] = bool(s.get("replicate", False))
            res["arms"].append(a)
            print("[seed %d] arm %s/%s/%s%s done %.0fs" % (seed, a["condition"], a["arm"], a["point"],
                                                          " (replicate)" if a["replicate"] else "",
                                                          time.time() - t0), flush=True)
    res["arms"].sort(key=lambda a: (a["condition"], a["point"], a["arm"], a["replicate"]))
    res["failability"] = failability(res["arms"][0]["d1_a_go"])
    res["gates"] = grade_seed(res)
    res["elapsed_s"] = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    print(json.dumps(res["gates"], indent=2, default=str), flush=True)
    return res


def _arm(res, cond, arm, point="primary", replicate=False):
    for a in res["arms"]:
        if a["condition"] == cond and a["arm"] == arm and a["point"] == point and a["replicate"] == replicate:
            return a
    return None


def _holds(res, pname, n):
    def dc(c, a):
        x = _arm(res, c, a, pname)
        return None if x is None else x["delayed_counts"]["correct"]
    imm = all(_arm(res, c, a, pname) is not None and _arm(res, c, a, pname)["immediate_counts"]["correct"] == n
              for c in ("neutral", "salient") for a in ARMS_BAND)
    bi, bl, bn = dc("salient", "intact"), dc("salient", "lesion_da_encoding"), dc("neutral", "intact")
    defined = bool(imm and None not in (bi, bl, bn))
    return {"immediate_ok": imm, "salient_intact": bi, "salient_lesion": bl, "neutral_intact": bn,
            "defined": defined,
            "holds": bool(defined and bi >= n - 1 and bl <= 1 and bn <= 1)}


def grade_seed(res):
    """The v3 pre-registered per-seed gates (see the v3 PREREG). UNDEFINED is never a pass."""
    th = 0.62
    n = len(FACTS)
    g = {}
    sal_fact_da = [r["da"] for r in res["traces"]["salient"] if r["fact"] is not None]
    neu_all_da = [r["da"] for r in res["traces"]["neutral"]]
    g["G1_salient_fact_da_min"] = min(sal_fact_da)
    g["G1_neutral_all_da_max"] = max(neu_all_da)
    g["G1_natural_drive"] = bool(min(sal_fact_da) >= th + 0.10 and max(neu_all_da) < th)
    g["G1b_da_trace_deterministic"] = bool(res.get("da_trace_repeat_identical"))
    # G0: order independence / fresh-process determinism
    a1, a2 = _arm(res, "salient", "intact"), _arm(res, "salient", "intact", replicate=True)
    g["G0_replicate_identical"] = bool(a1 is not None and a2 is not None and a1["pid"] != a2["pid"]
                                       and a1["write_gains_used"] == a2["write_gains_used"]
                                       and [t["a"] for t in a1["turn_log"]] == [t["a"] for t in a2["turn_log"]]
                                       and a1["delayed_24h"] == a2["delayed_24h"] and a1["blocks"] == a2["blocks"])
    gammas = {round(a["gamma_primary"], 12) for a in res["arms"]}
    g["G0_gamma_identical_all_arms"] = len(gammas) == 1

    def dc(cond, arm, p="primary"):
        a = _arm(res, cond, arm, p)
        return None if a is None else a["delayed_counts"]["correct"]

    g["G2_immediate_recall_intact_all_primary_arms"] = bool(all(
        _arm(res, c, a)["immediate_counts"]["correct"] == n for c in ("neutral", "salient") for a in ARMS_PRIMARY))
    si, sl = dc("salient", "intact"), dc("salient", "lesion_da_encoding")
    g["salient_24h_intact"], g["salient_24h_lesion_da_encoding"] = si, sl
    g["G3_salient_intact_vs_lesion"] = bool(si >= n - 1 and sl <= 1)
    ni = dc("neutral", "intact")
    g["neutral_24h_intact"] = ni
    g["G4_neutral_not_captured"] = bool(ni <= 1)
    sc = dc("salient", "lesion_capture")
    g["salient_24h_lesion_capture"] = sc
    g["G5_capture_subedge"] = bool(sc <= 1)
    so, no = dc("salient", "companion_off"), dc("neutral", "companion_off")
    g["salient_24h_companion_off"], g["neutral_24h_companion_off"] = so, no
    g["G6_production_default_null"] = bool(so == n and no == n)
    # attribution: how much of the salient intact-vs-lesion 24 h change is absent from the same comparison in the
    # neutral conversation? (None == UNDEFINED when the salient effect itself is ~0.)
    from tools.lab import attributable_to
    g["neutral_intact_minus_lesion"] = ni - dc("neutral", "lesion_da_encoding")
    g["attributable_to_surprise_context"] = attributable_to(
        "v3 DA-through-synapses 24h effect: salient vs neutral context", si - sl, g["neutral_intact_minus_lesion"])
    g["confab_total_primary_and_capture_band"] = sum(
        a["delayed_counts"]["confab"] + a["immediate_counts"]["confab"] for a in res["arms"]
        if a["point"] == "primary" or a["point"] in [p["name"] for p in BAND_CAPTURE])
    fa = res["failability"]["points"]
    band = {}
    for pt in BAND_CAPTURE:
        band[pt["name"]] = dict(_holds(res, pt["name"], n), **fa[pt["name"]])
    g["G8_band_capture"] = band
    n_inf = sum(1 for v in band.values() if v["informative"])
    g["G8_n_informative"] = n_inf
    g["G8_robust"] = bool(n_inf >= MIN_INFORMATIVE_BAND_POINTS
                          and all(v["holds"] for v in band.values() if v["informative"]))
    g["G8_all_points_defined"] = all(v["defined"] for v in band.values())
    g["readability_band_beta_REPORTED_NOT_GATED"] = {pt["name"]: _holds(res, pt["name"], n) for pt in BAND_BETA}
    g["exploratory_lesion_novelty"] = {c: dc(c, "lesion_novelty") for c in ("neutral", "salient")}
    si_arm, sl_arm = _arm(res, "salient", "intact"), _arm(res, "salient", "lesion_da_encoding")
    sc_arm = _arm(res, "salient", "lesion_capture")
    g["salient_write_gain_used_intact_mean"] = float(np.mean(si_arm["write_gains_used"]))
    g["salient_write_gain_used_lesion_mean"] = float(np.mean(sl_arm["write_gains_used"]))
    g["salient_p_max_intact"] = si_arm["p_max"]
    g["salient_p_max_lesion_da_encoding"] = sl_arm["p_max"]
    g["salient_p_max_lesion_capture"] = sc_arm["p_max"]
    g["salient_z_mean_by_block_intact"] = [b["z_mean"] for b in si_arm["blocks"]]
    g["neutral_z_mean_by_block_intact"] = [b["z_mean"] for b in _arm(res, "neutral", "intact")["blocks"]]
    from tools.verdict import Verdict
    v = Verdict("v3 synaptic DA tag-and-capture, natural drive, 24 h recall (seed %s)" % res["seed"])
    v.require("G0 fresh-process replicate identical (gains, D1 reads, 24 h replies, synapse state)",
              g["G0_replicate_identical"])
    v.require("G0 calibrated gamma identical in every arm process", g["G0_gamma_identical_all_arms"])
    v.require("G1 natural DA contrast (salient facts >= 0.72, every neutral turn < 0.62)", g["G1_natural_drive"])
    v.require("G1b DA trace deterministic at this seed", g["G1b_da_trace_deterministic"])
    v.require("G2 immediate recall intact in every primary arm", g["G2_immediate_recall_intact_all_primary_arms"])
    v.require("G8 every capture-band point DEFINED (an UNDEFINED point is not excluded)", g["G8_all_points_defined"])
    v.reaches("DA->encoding lesion reaches the write gain", before=g["salient_write_gain_used_intact_mean"],
              after=g["salient_write_gain_used_lesion_mean"])
    v.reaches("DA->encoding lesion reaches the PRP pool", before=g["salient_p_max_intact"],
              after=g["salient_p_max_lesion_da_encoding"])
    v.reaches("capture lesion reaches the PRP pool", before=g["salient_p_max_intact"],
              after=g["salient_p_max_lesion_capture"])
    v.disabled("recall-turn DA -> D1 drive", why="recall questions' DA is not fed (testing-effect confound at +1 min)")
    go = all(g[k] for k in ("G3_salient_intact_vs_lesion", "G4_neutral_not_captured", "G5_capture_subedge",
                            "G6_production_default_null", "G8_robust"))
    d = v.decide(go, verbose=False)
    g["verdict"] = d["status"]
    res["status"] = d["status"]
    res["preconditions"] = d["preconditions"]
    res["undefined_reasons"] = d["undefined_reasons"]
    res["disabled_processes"] = d["disabled_processes"]
    return g


def seed_signflip_p(diffs):
    """One-sided exact sign-flip permutation p with the SEED as the unit (facts within a session share one PRP pool
    and are not exchangeable). p = fraction of the 2^k sign assignments whose sum >= the observed sum."""
    diffs = [float(d) for d in diffs]
    obs = sum(diffs)
    hits = tot = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        tot += 1
        hits += sum(s * d for s, d in zip(signs, diffs)) >= obs - 1e-12
    return hits / tot


def aggregate(d):
    rows = []
    for p in sorted(x for x in glob.glob(os.path.join(d, "seed*.json")) if not x.endswith(".prov.json")):
        r = json.load(open(p))
        g = r["gates"]
        rows.append({"seed": r["seed"], **{k: g[k] for k in (
            "verdict", "G1_salient_fact_da_min", "G1_neutral_all_da_max", "salient_24h_intact",
            "salient_24h_lesion_da_encoding", "neutral_24h_intact", "salient_24h_lesion_capture",
            "salient_24h_companion_off", "neutral_24h_companion_off", "confab_total_primary_and_capture_band",
            "G8_robust", "G8_n_informative", "G0_replicate_identical")},
            "band_holds": {k: v["holds"] for k, v in g["G8_band_capture"].items()},
            "readability_beta": g["readability_band_beta_REPORTED_NOT_GATED"]})
    seeds = sorted(r["seed"] for r in rows)
    out = {"version": "v3-synaptic", "seeds": seeds, "rows": rows, "all_six_seeds_present": seeds == sorted(SEEDS),
           "n_go": sum(r["verdict"] == "GO" for r in rows)}
    diffs = [r["salient_24h_intact"] - r["salient_24h_lesion_da_encoding"] for r in rows]
    out["seed_diffs_intact_minus_lesion"] = diffs
    out["seed_signflip_p"] = seed_signflip_p(diffs) if rows else None
    diffs_sn = [r["salient_24h_intact"] - r["neutral_24h_intact"] for r in rows]
    out["seed_diffs_salient_minus_neutral"] = diffs_sn
    out["seed_signflip_p_salient_vs_neutral"] = seed_signflip_p(diffs_sn) if rows else None
    from tools.verdict import Verdict
    v = Verdict("v3 synaptic DA tag-and-capture, 6-seed aggregate")
    v.require("all six seeds present", out["all_six_seeds_present"])
    v.require("no seed UNDEFINED", all(r["verdict"] != "UNDEFINED" for r in rows) if rows else None)
    v.require("seed-level sign-flip p <= 0.05 (intact vs DA->encoding lesion, unit = seed)",
              (out["seed_signflip_p"] is not None and out["seed_signflip_p"] <= 0.05) if rows else None)
    d_ = v.decide(out["n_go"] == len(SEEDS), verbose=False)
    out["status"] = d_["status"]
    out["preconditions"] = d_["preconditions"]
    out["undefined_reasons"] = d_["undefined_reasons"]
    return out


class _TinyComp:
    """One synapse per block (D=1): enough to replay the ledger's kernel on a recorded drive."""

    def __init__(self):
        self.D = 1
        self.store_conns = []


def _replay_blocks(arm, gamma):
    """Replay the v3 kernel on an arm's RECORDED D1 drive (turn_log a_eff) and recorded tags (tag0_mean), at a
    different gamma. Returns the z of each block at +24 h. No brain run: the brain's contribution is the recorded
    drive and the write gains; only gamma changes."""
    from webapp.da_tag_capture import SynapticTagCaptureLedger
    pt = arm["point_params"]
    L = SynapticTagCaptureLedger(0, gamma=gamma, beta=0.0, tau_early_h=pt["tau_e"], tau_tag_h=pt["tau_tag"],
                                 tau_p_h=pt["tau_p"], tau_z_h=pt["tau_z"])
    comp = _TinyComp()
    conv = conversation(arm["condition"])
    fact_turns = [ti for ti, (_t, fi) in enumerate(conv) if fi is not None]
    tags = [b["tag0_mean"] for b in arm["blocks"]]
    with _env({"BRAIN_DA_CAPTURE_LESION": "0", "BRAIN_DA_ENCODING_LESION": "0"}):
        for ti, rec in enumerate(arm["turn_log"]):
            L.observe_turn(rec["t_h"], TURN_H, rec["da"], a_override=rec["a_eff"])
            if ti in fact_turns:
                k = fact_turns.index(ti)
                comp.store_conns.append((k + 1, 0, complex(tags[k])))
                L.on_store(comp, rec["t_h"])
    L.advance(comp, len(conv) * TURN_H + DELAY_H)
    return [float(b["z"][0]) for b in L.blocks]


def gamma_flip_margins(arm):
    """EXPLORATORY (not pre-registered): the gamma multiplier at which this arm's 24 h capture pattern would flip,
    given its recorded drive. For a captured arm: the largest m < 1 at which some block is lost. For an uncaptured
    arm: the smallest m > 1 at which some block is captured (None if even m = 1e4 cannot: zero drive)."""
    g = arm["gamma"]
    z1 = _replay_blocks(arm, g)
    captured = [z > 0.5 for z in z1]

    def flipped(m):
        return [z > 0.5 for z in _replay_blocks(arm, g * m)] != captured

    if any(captured):
        lo, hi = 1e-4, 1.0
        if not flipped(lo):
            return {"replay_captured": captured, "m_flip": None}
        for _ in range(30):
            mid = (lo * hi) ** 0.5
            lo, hi = (mid, hi) if flipped(mid) else (lo, mid)    # keep lo flipped, hi unflipped
        return {"replay_captured": captured, "m_flip": lo}
    lo, hi = 1.0, 1e4
    if not flipped(hi):
        return {"replay_captured": captured, "m_flip": None}
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if not flipped(mid) else (lo, mid)
    return {"replay_captured": captured, "m_flip": hi}


def margins(d):
    rows = []
    for p in sorted(x for x in glob.glob(os.path.join(d, "seed*.json")) if not x.endswith(".prov.json")):
        r = json.load(open(p))
        row = {"seed": r["seed"]}
        for cond in ("salient", "neutral"):
            a = _arm(r, cond, "intact")
            m = gamma_flip_margins(a)
            row[cond] = dict(m, recorded_24h_correct=a["delayed_counts"]["correct"],
                             replay_matches_run=[z > 0.5 for z in _replay_blocks(a, a["gamma"])] ==
                             [b["z_mean"] > 0.5 for b in a["blocks"]])
        rows.append(row)
    return {"note": "EXPLORATORY, not pre-registered: gamma multiplier at which each seed's primary intact capture "
                    "pattern would flip, replaying the kernel on the recorded D1 drive and tags", "rows": rows}


def selftest(seed=7):
    """Instrument checks (seed 7, not a gate seed):
      S1 exact pass-through: beta=0, read at the write time -> the composer's own write BIT-EXACT.
      S2 flag gate: maybe_ledger() is None with BRAIN_DA_TAG_CAPTURE unset (production byte-identical path).
      S3 FORGOTTEN is reportable: zero D1 drive -> the block abstains at +24 h.
      S4 REMEMBERED is reportable: sustained strong D1 drive -> the block is recalled at +24 h.
      S5 lesions act on the edge: encoding lesion pins the D1 pool's DA to tonic; capture lesion zeroes a_eff.
      S6 no host threshold: a SUB-Go-boundary D1 drive sustained long enough DOES capture (the compare is gone)."""
    from webapp import da_tag_capture as TC
    out = {}
    comp_a, comp_b = _build_composer(seed), _build_composer(seed)
    for f in FACTS:
        comp_a.store(*f["lem"])
    L0 = TC.SynapticTagCaptureLedger(seed, gamma=10.0, beta=0.0)
    for f in FACTS:
        comp_b.store(*f["lem"])
        L0.on_store(comp_b, 0.0)
    out["S1_bit_exact_passthrough"] = [(p, q, complex(w)) for (p, q, w) in comp_a.store_conns] == \
        [(p, q, complex(w)) for (p, q, w) in comp_b.store_conns]
    with _env({"BRAIN_DA_TAG_CAPTURE": "0"}):
        out["S2_flag_off_no_ledger"] = TC.maybe_ledger(seed) is None
    g0 = TC.calibrate_gamma(0.224)
    comp_c = _build_composer(seed)
    L1 = TC.SynapticTagCaptureLedger(seed, gamma=g0, beta=1.0)
    for i, f in enumerate(FACTS):
        t = 0.0 if i < 2 else 5.0
        if i < 2:
            L1.observe_turn(t, 10.0 / 60.0, 0.9, a_override=0.6)
        comp_c.store(*f["lem"])
        L1.on_store(comp_c, t)
    L1.advance(comp_c, 5.0 + IMMEDIATE_H)
    out["S0_all_recalled_fresh"] = [r["outcome"] for r in _recall(comp_c)]
    L1.advance(comp_c, 5.0 + DELAY_H)
    rec = _recall(comp_c)
    out["S3_S4_outcomes_24h"] = [r["outcome"] for r in rec]
    out["S3_forgotten_detectable"] = all(r["outcome"] != "correct" for r in rec[2:])
    out["S4_remembered_detectable"] = all(r["outcome"] == "correct" for r in rec[:2])
    with _env({"BRAIN_DA_ENCODING_LESION": "1"}):
        a = TC.prp_da(0.9)
    with _env({"BRAIN_DA_ENCODING_LESION": "0", "BRAIN_DA_CAPTURE_LESION": "1"}):
        L2 = TC.SynapticTagCaptureLedger(seed, gamma=g0)
        b = L2.observe_turn(0.0, 1.0, 0.9, a_override=0.8)
    out["S5_lesions_act_on_edge"] = bool(a == 0.5 and b == 0.0)
    out["S6_subthreshold_sustained_captures"] = TC.kernel_capture_single(0.224 * 0.5, g0, dur_h=0.5) > 0.5
    out["pass"] = all(out[k] for k in ("S1_bit_exact_passthrough", "S2_flag_off_no_ledger", "S3_forgotten_detectable",
                                       "S4_remembered_detectable", "S5_lesions_act_on_edge",
                                       "S6_subthreshold_sustained_captures")) and \
        all(o == "correct" for o in out["S0_all_recalled_fresh"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--aggregate", default=None)
    ap.add_argument("--arm-worker", nargs=2, metavar=("SPEC", "OUT"), default=None)
    ap.add_argument("--margins", default=None, help="exploratory gamma-flip margins over a result directory")
    a = ap.parse_args()
    global COMPOSER_D
    COMPOSER_D = int(a.D)
    if a.margins:
        r = margins(a.margins)
        json.dump(r, open(os.path.join(a.margins, "margins_exploratory.json"), "w"), indent=2, default=str)
        print(json.dumps(r, indent=2, default=str))
        return
    if a.arm_worker:
        spec = json.load(open(a.arm_worker[0]))
        r = run_arm(spec["seed"], spec["condition"], spec["arm"], spec["trace"], spec["point"])
        json.dump(r, open(a.arm_worker[1], "w"), default=str)
        return
    if a.selftest:
        r = selftest()
        print(json.dumps(r, indent=2, default=str))
        sys.exit(0 if r["pass"] else 1)
    if a.aggregate:
        r = aggregate(a.aggregate)
        json.dump(r, open(os.path.join(a.aggregate, "aggregate.json"), "w"), indent=2, default=str)
        print(json.dumps(r, indent=2, default=str))
        return
    out = a.out or os.path.join("research/findings/raw/_da_encoding_natural_drive_v3", "seed%d.json" % a.seed)
    run_seed(a.seed, out, workers=a.workers)


if __name__ == "__main__":
    main()
