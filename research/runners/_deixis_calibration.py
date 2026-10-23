"""deixis-271-s1 step 0a(i): _deixis_calibration — the frozen-record calibration.

Port of docs/drafts/deixis/s1-scratch/fix2/calib_proto.py onto the full
Appendix C circuit (research/runners/_deixis_hearing_core.py), executing
exactly the six items of packet lines 64-79 (deixis-271-s1-PACKET.md),
command 107:

  (a) per-target transfer curves on GRID = {0,2,3,4,5,6,7,8,9,10,11,12,
      14,16,20} for TOK->P on the base circuit and the CAUSAL+WTA variant,
      TOK->E in production hearing (no H, release open, C silent) and toy
      hearing (the target's H lines on), plus the exposure condition
      reported ONLY as the increment over w=0 (it never enters a
      threshold); thresholds by the rule of Appendix C / DESIGN 2.6
      (packet 505-516) as CORRECTED by RULING deixis-271-s1-thr-1022
      (2026-10-22, this replaces the 19:30 MiMo decision-A pinning to the
      first pool/assembly): a threshold covers EVERY target neuron of
      EVERY unit — theta_fire = the MAXIMUM over the per-unit thresholds:
      smallest grid w where all 12 cells of EVERY P pool (P_0..P_4) fire
      >= 1 spike in a 40-step token on EVERY calibration seed in EVERY
      hearing condition of that target; for TOK->E the same over all 10
      entity assemblies (E_0..E_9); theta_silent = the largest grid w
      below theta_fire at which EVERY pool emits <= 1 spike.  Port
      follows calib_proto.py:133-142 (fire = all-seed/all-cond frac 1.0,
      silent = all-seed/all-cond spikes <= 1, th_s = max silent w <
      th_f) applied per unit and then pooled per the ruling; every unit's
      curves and its own theta pair live in sub_target_curves beside the
      pooled thresholds, and a per-unit theta_fire span over 2 grid steps
      is a RESULT heterogeneity line (not a STOP).  The exposure increment
      is over the whole E population (proto's single assembly) and stays
      outside `transfer`.
  (b) the saturated-route check on seed 61: TOK->P and TOK->E at W_MAX=20
      must fire P (all 60) and E (all 120) — every neuron — and with the
      target's H lines on must fire its C_k[e] cells; at theta_silent and
      at 0 the targets emit at most one spike (DESIGN 333-335).  Step
      1a's REF latch is out of 0a's scope by the same paragraph.
  (c) the C-weight row: the first row of the Appendix C grid (packet
      505-511, R0..R5 in order: (6,2) (8,2) (6,3) (8,3) (10,3) (8,4), the
      second value = W_EC = W_PC) passing every P0 item at K=0 on 10/10
      calibration seeds, 600 utterances of the calib stream per seed in
      exposure mode.  P0 items (packet 271-276, DESIGN 633-638): c95
      (C_k[e] fires >= 95% of utterances where its entity takes part
      (y >= 1.0) and its H block is driven, <= 2% otherwise), c_leak (C
      fires <= 1% of windows without its H afferent), ptr_occ (P_SPK /
      P_ADR occupancy on non-matching utterances, y[pool] == 0, <= 0.02),
      base_rate (each pool's empirical firing-utterance rate within
      +-0.05 of the schedule's y[pool] occupancy), e_scene (>= 2 E
      assemblies firing in every scene), route_tx (dx_ptr_tx/dx_name_tx
      read back 0 on every exposure step).  P0 item 7 (two runs
      byte-identical) is a run-level property checked by module 6's
      STOP-ID on the identity pair, not by row selection.  Selection
      stops at the first fully passing row; if none passes, row stays null
      and passing_rows is empty (module 6 STOP-P0).
  (d) unit conversions (DESIGN 493-496): hebbian_run_fraction = fraction
      of replay steps on which the engine's Hebbian block runs — counted
      from the firing states as `if _prev_any and _fired_any`
      (sim/bridge.py:10236); steps_per_occurrence = contiguous steps a
      TOK rate-window trace sits above hebbian_bcm_pre_floor = 0.02,
      divided by the number of occurrences; y_on = the rate-window trace
      of a driven P neuron (mean over the target pool's 12 cells) at the
      end of its 40-step drive; presence_current = the current from
      {100,150,200,250,300} pA whose E rate is nearest 0.2 x the
      participant rate measured at 1500 pA.
  (e) per-arm step time (DESIGN 497-500): each of the six arms' update
      code executing at learning rate 0 for 2000 exposure steps on seed
      61 — COV and COV-PRE include the runner's covariance update over
      the plastic blocks (written back with lr=0, so the bytes cannot
      move); every arm includes the per-step gate read-back.  Arms
      (DESIGN 563-569): CAUSAL = default causal rule
      (hebbian_rate_window=False); CAUSAL+WTA = same + the WTA organ;
      RW-HEBB = rate-window potentiation; BCM = hebbian_bcm gain with
      rate window on (gain 1.0: a branch selector, DESIGN gives no 0a
      number, inert at lr=0); COV = rate window on + runner update
      dw = lam*x_j*(y_i - ybar_i) for x_j > floor; COV-PRE = runner
      update with the presynaptically gated leak
      dw = lam*x_j*(w_max*(y_i-ybar_i) - kap*w_ij).  Engine decay is 0
      throughout (the plasticity-off rule, packet 64-67).
  (f) per-population median a/b/d/C of cp_izh_a/b/d/C plus both
      heterogeneity flags (enable_parameter_heterogeneity,
      preset_centred_parameter_heterogeneity) read back from the config
      the circuit actually ran with.

Plasticity off throughout (packet 64-67, DESIGN 489-500): cfg
hebbian_learning_rate = 0, hebbian_weight_decay = 0, the runner's cov
update applied with an lr factor of 0; every bridge's learned blocks
(dx_ptr, dx_name) are sha256'd at build and at the end of its phase, and
the record carries plasticity_off {unchanged, changed_blocks, blocks}.

Delegated design ruling (RULING models, one MiMo Pro, quoted verbatim in
the WORKLOG) — as CORRECTED by RULING deixis-271-s1-thr-1022 (Claude,
2026-10-22): decision A's pinning to the FIRST pool (P_0 = SPK) and the
FIRST entity assembly (E_0 = V1) is REPLACED — thresholds pool over
every unit (theta_fire = max over per-unit thresholds, theta_silent =
below pooled theta_fire with EVERY pool <= 1 spike), and EVERY unit's
curves plus its own theta pair are stored as sub_target_curves (a
sibling key OUTSIDE `transfer`, no pass/fail booleans); thresholds.P
comes from the base circuit alone, thresholds.E is the conjunct over
production AND toy, and the WTA variant carries its own transfer entry,
its own pair,
and an extra thresholds key `P_wta` (module 4's load_thresholds reads
only P and E, so the extra key is inert there).  The exposure increment
lives in the sibling key `exposure_increment`, outside `transfer`
(module 6's STOP-CAL would otherwise see a threshold-bearing entry that
cannot pass it).

Timing determinism (packet 108-109 + module 6 docstring "timing lives in
sidecars"): the single-seed identity runs calibration_s61_a.json /
calibration_s61_b.json must be byte-identical, so the main JSON of a
single-seed run carries NO volatile field; wall-clock timings go to
<out>.timing.json only.  A multi-seed run additionally embeds step_times
in the main JSON because packet command 118 feeds calibration.json to
module 6, which reads cal["step_times"] for the 0b queue plan.

Stream interface: rows are x/y/S/A per packet 53-54 and the committed
module 4 reader (header POST = POOLS+ENT, 15 entries), PLUS a field `H`
— the flat list of [relation, entity] H-block drives that module 1
RECORDS from the world's own state (RULING dejits-271-s1-0a-q-1022:
the coins world() already draws for LEFT/NEWEST, the givenness set for
GIV, the heard voices for SPK/ADR; no extra RNG draw).  The design
requires per-(k,e) drives ("H_k[e] driven 1500 pA while relation k
holds for e", packet 495).  If rows lack `H`, this runner refuses with
that key named instead of inventing the truth.

Region coordinates: idx[<pop>] is the population's GLOBAL neuron index
array; form_slices()/entity_slices()/hc_slices() are region-LOCAL.  All
drives go through idx[...] indexing (tok local == global is never
assumed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import _deixis_hearing_core as HC   # noqa: E402
from research.runners import declare_output                # noqa: E402
from research.runners._deixis_rule_replica import TOKM     # noqa: E402

try:  # attribution-required gate wants a genuine call (module-4 lesson)
    from tools.lab import attributable_to
except Exception:                                          # pragma: no cover
    attributable_to = None

SCHEMA = "deixis-s1-calibration-v1"
GRID = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20]
ROWS = [(6, 2), (8, 2), (6, 3), (8, 3), (10, 3), (8, 4)]   # packet 505-511
ARMS = ["CAUSAL", "CAUSAL+WTA", "RW-HEBB", "BCM", "COV", "COV-PRE"]
ARM_CFG = {
    "CAUSAL":     dict(rate_window=False, bcm=0.0, wta=False, runner=None),
    "CAUSAL+WTA": dict(rate_window=False, bcm=0.0, wta=True,  runner=None),
    "RW-HEBB":    dict(rate_window=True,  bcm=0.0, wta=False, runner=None),
    "BCM":        dict(rate_window=True,  bcm=1.0, wta=False, runner=None),
    "COV":        dict(rate_window=True,  bcm=0.0, wta=False, runner="COV"),
    "COV-PRE":    dict(rate_window=True,  bcm=0.0, wta=False,
                       runner="COV-PRE"),
}
DRIVE = 1500.0
SETTLE = 60
TOKEN = 40
TAIL_OPEN = 100       # plasticity open after the last token (packet 491)
GAP = 60              # quiet gap, plasticity closed (packet 491)
PRESENCE_GRID = [100, 150, 200, 250, 300]
PRE_FLOOR = 0.02      # sim/config.py:723 hebbian_bcm_pre_floor
W_MAX = 20.0
ARM_STEPS = 2000      # DESIGN 499
P0_TOTAL = 600        # packet 70
P0_ITEMS = ["c95", "c_leak", "ptr_occ", "base_rate", "e_scene",
            "route_tx"]
NEEDS_KEY = "[deixis-271-s1-H]"

# Runner-update constants for the COV arms: DESIGN 351-354.  At lr=0 they
# are weight-inert; they carry the real arithmetic (what (e) times).
LAM = 1.0
KAP = 1.0
YBAR_ALPHA = 0.001

# plasticity-off accumulator: bridge-id -> sha at build; phase-end checks
# move entries into _PLAST.
_BRIDGE_SHA = {}
_PLAST = {"checks": [], "changed_blocks": [], "blocks": None}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arr(a):
    a = np.ascontiguousarray(np.asarray(a, dtype="<f4"))
    return hashlib.sha256(a.tobytes()).hexdigest()


def spec_rng(s):
    """'61-70' -> [61..70]; '61' -> [61]; '1,2' -> [1,2]."""
    s = str(s)
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(t) for t in s.split(",") if t]


# ---------------------------------------------------------------------------
# build / gates / weights / plasticity checks
# ---------------------------------------------------------------------------

def build_cal(seed, mode, w_hc=6.0, w_ec=20.0, w_pc=20.0, wta=False,
              learn_p=0.0, learn_e=0.0, rate_window=True):
    """HC.build with the calibration overrides: lr 0, decay 0, rate-window
    trace allocated (read-only; (d)/(e) read it), learn blocks injected at
    `learn_p`/`learn_e`.  Everything else is HC's frozen config."""
    regs = HC.regions(wta=wta)
    cfg = HC.config(seed)
    cfg.hebbian_learning_rate = 0.0
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_rate_window = bool(rate_window)
    cfg.brain_regions = regs
    cfg.region_pathways = []
    b = HC.SimulationBridge(core_config=cfg,
                            viz_config=HC.VisualizationConfig(),
                            runtime_state=HC.RuntimeState(),
                            gpu_config=HC.GPUConfig())
    b.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms)
    b._initialize_simulation_data(called_from_playback_init=False)
    idx = {k: np.asarray(v, np.int64) for k, v in
           b.region_manager.region_indices_dict().items()}
    inh = idx["fs"]
    if wta:
        inh = np.concatenate([inh, idx["pw"]])
    b.inject_explicit_wiring(
        HC.wiring_plan(w_hc, w_ec, w_pc, wta=wta, idx=idx),
        output_inhibitory_indices=inh)
    b.cp_external_input_current[:] = 0.0
    HC.set_mode(b, mode)
    set_learn_w(b, dx_ptr=learn_p, dx_name=learn_e)
    _BRIDGE_SHA[id(b)] = learn_sha(b)
    if _PLAST["blocks"] is None and learn_p == 0.0 and learn_e == 0.0:
        _PLAST["blocks"] = dict(_BRIDGE_SHA[id(b)])
    return b, idx


def set_learn_w(b, **by_gate):
    """Hand-set the learned blocks' weights (injection, not plasticity)."""
    data = b.cp_connections.data
    for g, w in by_gate.items():
        ii = np.asarray(b._plasticity_gate_to_synapses[g])
        data[ii] = np.float32(w)


def learn_sha(b, gates=("dx_ptr", "dx_name")):
    out = {}
    data = np.asarray(b.cp_connections.data)
    for g in gates:
        ii = np.asarray(b._plasticity_gate_to_synapses[g])
        out[g] = sha256_arr(data[ii])
    return out


def end_bridge(b, tag):
    """Phase end: compare the learn-block bytes against build-time."""
    before = _BRIDGE_SHA.pop(id(b), None)
    if before is None:
        return
    after = learn_sha(b)
    for g in ("dx_ptr", "dx_name"):
        if after.get(g) != before.get(g):
            _PLAST["changed_blocks"].append("%s:%s" % (tag, g))
    _PLAST["checks"].append(tag)


def plasticity_record():
    blocks = _PLAST["blocks"] or {g: sha256_arr(np.zeros(4, "<f4"))
                                  for g in ("dx_ptr", "dx_name")}
    return {"unchanged": not _PLAST["changed_blocks"],
            "changed_blocks": sorted(_PLAST["changed_blocks"]),
            "checks": sorted(set(_PLAST["checks"])),
            "blocks": blocks}


# ---------------------------------------------------------------------------
# stream loading (packet 53-54, module-4 header contract, + row field `H`)
# ---------------------------------------------------------------------------

def load_manifest(manifest_path, seeds):
    """manifest_calib.json -> ({seed: {path, sha256}}, manifest sha)."""
    mdir = os.path.dirname(os.path.abspath(manifest_path))
    with open(manifest_path) as f:
        m = json.load(f)
    if m.get("schema") != "deixis-streams-manifest-v1":
        raise ValueError("%s: unexpected schema" % manifest_path)
    out = {}
    for e in m["streams"]:
        s = int(e["seed"])
        if s not in seeds:
            continue
        p = os.path.join(mdir, e["file"])
        if not os.path.exists(p):
            raise FileNotFoundError(
                "manifest lists missing file %s" % e["file"])
        got = sha256_file(p)
        if got != e["sha256"]:
            raise ValueError("stream sha256 mismatch vs manifest: %s"
                             % e["file"])
        out[s] = {"path": p, "sha256": got}
    missing = sorted(s for s in seeds if s not in out)
    if missing:
        raise FileNotFoundError(
            "manifest has no entry for seeds %s" % missing)
    return out, sha256_file(manifest_path)


def load_stream(path):
    """One stream -> (header, rows); rows keep every field (x, y, S, A,
    h) — unlike module 4's read_stream, the replay needs the world
    fields.  Header vocabulary validated against module 4's TOKM/POST."""
    from research.runners._deixis_rule_replica import TOKM, POST
    with open(path) as f:
        head = json.loads(f.readline())
        if head.get("TOK") != TOKM or head.get("POST") != POST:
            raise ValueError(
                "stream vocabulary differs from the replica's: %s" % path)
        rows = [json.loads(ln) for ln in f if ln.strip()]
    if not rows:
        raise ValueError("empty stream: %s" % path)
    for i, r in enumerate(rows):
        if "H" not in r:
            raise ValueError(
                "row %d of %s has no 'H' field (the per-(k,e) H-block "
                "drives the exposure replay and P0 need, per RULING "
                "dejits-271-s1-0a-q-1022); see NEEDS %s"
                % (i, path, NEEDS_KEY))
    return head, rows


def row_parts(row, head):
    """-> (x ids, y(15), S, A, hset) for one utterance row (H read as row["H"])."""
    y = np.zeros(len(head["POST"]), np.float32)
    for i, v in row["y"]:
        y[i] = v
    hset = frozenset((int(k), int(e)) for k, e in row["H"])
    return list(row["x"]), y, int(row["S"]), int(row["A"]), hset


def unit_idx(idx, key, unit=0):
    """Global indices of unit `unit` (12 cells) of population `key`:
    first pool / first entity assembly — delegate ruling A."""
    a = idx[key]
    return a[unit * HC.N_UNIT:(unit + 1) * HC.N_UNIT]


def drive_token(cur, idx, tok_names, tid):
    name = tok_names[int(tid)]
    loc = HC.form_slices()[name]
    cur[idx["tok"][loc]] = DRIVE


def drive_h(cur, idx, hset):
    hs = HC.hc_slices()
    for (k, e) in hset:
        cur[idx["h"][hs[(k, e)]]] = DRIVE


def scene_drive(idx, y, presence_current, n):
    """E population from the row: participants 1500, present 0.2-rate
    cells the measured presence current, absent 0 (DESIGN 679-701)."""
    c = np.zeros(n, np.float32)
    npl = len(HC.POOLS_REL)
    for j in range(len(HC.ENTITIES)):
        v = float(y[npl + j])
        tgt = idx["e"][j * HC.N_UNIT:(j + 1) * HC.N_UNIT]
        if v >= 1.0:
            c[tgt] = DRIVE
        elif v > 0.0:
            c[tgt] = presence_current
    return c


# ---------------------------------------------------------------------------
# (a) transfer curves — port of calib_proto.py::trial
# ---------------------------------------------------------------------------

def trial(seed, w, cname, presence_current, head, pop, wta=False):
    """One curve point on the FULL circuit: fresh build per trial
    (proto.py:82), settle 60, drive FORMS[0] for 40 steps, count spikes
    per neuron from the firing states (proto.py:103).

    Only the curve's OWN route is set to w: a P curve drives TOK->P with
    TOK->E at 0 and an E curve drives TOK->E with TOK->P at 0 (proto's E
    trials had no P region at all).  With learn_p = 0 the production
    hearing condition's required "C silent" holds exactly: at hearing
    E->C is closed, so C receives only H (off) + the P route (silent).

    Returns per-population spike counts: {"p", "e", "c"}.
    """
    mode = "EXPOSURE" if cname == "exposure" else "HEARING"
    lp = w if pop == "p" else 0.0
    le = w if pop == "e" else 0.0
    b, idx = build_cal(seed, mode, wta=wta, learn_p=lp, learn_e=le)
    n = b.core_config.num_neurons
    cur = np.zeros(n, np.float32)
    drive_token(cur, idx, head["TOK"], 0)
    if cname == "exposure":
        cur[idx["e"]] = presence_current
    if cname == "toy":
        hs = HC.hc_slices()
        for k in range(len(HC.POOLS_REL)):
            cur[idx["h"][hs[(k, 0)]]] = DRIVE
    b.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = cur
    cp = np.zeros(HC.N_P, np.int64)
    ce = np.zeros(HC.N_E, np.int64)
    cc = np.zeros(HC.N_HC, np.int64)
    for _ in range(TOKEN):
        b._run_one_simulation_step()
        f = np.asarray(b.cp_firing_states).astype(bool)
        cp += f[idx["p"]]
        ce += f[idx["e"]]
        cc += f[idx["c"]]
    b.cp_external_input_current[:] = 0.0
    end_bridge(b, "curve")
    del b
    return {"p": cp, "e": ce, "c": cc}


def threshold_pair(curve, conds, grid):
    """calib_proto.py:136-141 on the target unit's per-seed rows."""
    fire, silent = [], []
    for w in grid:
        rows_ = curve[w]
        ok_fire = all(min(r["frac"] for r in rows_[c]) >= 1.0
                      for c in conds)
        ok_sil = all(max(r["spk"] for r in rows_[c]) <= 1
                     for c in conds)
        if ok_fire:
            fire.append(w)
        if ok_sil:
            silent.append(w)
    tf = min(fire) if fire else None
    ts = max((w for w in silent if tf is None or w < tf),
             default=None)
    return [tf, ts]


def envelope(pairs):
    """theta over a target's hearing conditions (packet 512-516)."""
    fs = [p[0] for p in pairs if p[0] is not None]
    ss = [p[1] for p in pairs if p[1] is not None]
    tf = max(fs) if fs else None
    if tf is None:
        return [None, max(ss) if ss else None]
    cand = [v for v in ss if v is not None and v < tf]
    return [tf, max(cand) if cand else None]


def run_curves(seeds, presence_current, head, grid):
    """All five conditions -> (transfer, sub_target_curves,
    exposure_increment, thresholds)."""
    conds = [
        ("TOK>P@base", "p", dict(wta=False), "base"),
        ("TOK>P@wta", "p", dict(wta=True), "wta"),
        ("TOK>E@production", "e", dict(wta=False), "production"),
        ("TOK>E@toy", "e", dict(wta=False), "toy"),
        ("TOK>E@exposure", "e", dict(wta=False), "exposure"),
    ]
    transfer, sub, expo = {}, {}, {}
    for key, pop, kw, cname in conds:
        is_p = (pop == "p")
        unit_names = _sub_names(is_p)
        # pooled rows: EVERY neuron of EVERY pool/assembly (RULING
        # deixis-271-s1-thr-1022), plus per-unit rows for the
        # sub_target_curves + each unit's own theta pair
        curve = {w: {} for w in grid}
        ucur = {nm: {w: {} for w in grid} for nm in unit_names}
        for w in grid:
            for s in seeds:
                res = trial(s, w, cname, presence_current, head,
                            pop, **kw)
                v = res[pop]
                row = {"spk": int(v.sum()),
                       "frac": float((v > 0).mean()),
                       "c_spk": int(res["c"].sum())}
                curve[w].setdefault(cname, []).append(row)
                for u, nm in enumerate(unit_names):
                    z = v[u * HC.N_UNIT:(u + 1) * HC.N_UNIT]
                    urow = {"spk": int(z.sum()),
                            "frac": float((z > 0).mean())}
                    ucur[nm][w].setdefault(cname, []).append(urow)
                del res
        if cname == "exposure":
            expo[key] = curve
            continue
        # pooled theta_fire: threshold_pair on pooled rows is exactly
        # min{w : every cell of every unit fired} = max over per-unit
        # threshold_pair fires (a unit fires <=> its frac == 1.0)
        fire, _ = threshold_pair(curve, [cname], grid)
        # pooled theta_silent per the ruling: the largest w BELOW
        # pooled theta_fire at which EVERY unit emits <= 1 spike on
        # EVERY seed (threshold_pair's pooled-silent would instead
        # demand the TOTAL <= 1 across all units — stricter than the
        # ruling's per-pool reading)
        silent = None
        if fire is not None:
            for w in grid:
                if w >= fire:
                    break
                if all(max(r["spk"] for r in ucur[nm][w][cname]) <= 1
                       for nm in unit_names):
                    silent = w
        th = [fire, silent]
        unit_pairs = {}
        ssub = {}
        for nm in unit_names:
            ut = threshold_pair(ucur[nm], [cname], grid)
            unit_pairs[nm] = [ut[0], ut[1]]
            wagg = {}
            for w in grid:
                rs = ucur[nm][w][cname]
                wagg[str(w)] = {
                    "frac_min": min(r["frac"] for r in rs),
                    "spk_max": max(r["spk"] for r in rs)}
            ssub[nm] = {"curves": wagg, "theta": unit_pairs[nm]}
        transfer[key] = {
            "grid": list(grid), "cond": cname,
            "target": ("ALL P pools (P_0..P_4, 60 cells)" if is_p
                       else "ALL entity assemblies (E_0..E_9, "
                            "120 cells)"),
            "curves": {str(w): curve[w] for w in grid},
            "theta_fire": th[0], "theta_silent": th[1],
            "fired_all_by_20": th[0] is not None,
            "silent_below_theta": th[1] is not None,
        }
        sub[key] = ssub
    p_pair = [transfer["TOK>P@base"]["theta_fire"],
              transfer["TOK>P@base"]["theta_silent"]]
    pw_pair = [transfer["TOK>P@wta"]["theta_fire"],
               transfer["TOK>P@wta"]["theta_silent"]]
    e_pair = envelope([
        [transfer["TOK>E@production"]["theta_fire"],
         transfer["TOK>E@production"]["theta_silent"]],
        [transfer["TOK>E@toy"]["theta_fire"],
         transfer["TOK>E@toy"]["theta_silent"]],
    ])
    th = {"P": p_pair, "P_wta": pw_pair, "E": e_pair}
    # exposure increment (proto.py:139): max over seeds of
    # (spikes at w - spikes at w=0) on the target unit
    ec = "exposure"
    inc = {str(w): int(max(
        expo["TOK>E@exposure"][w][ec][i]["spk"]
        - expo["TOK>E@exposure"][0][ec][i]["spk"]
        for i in range(len(seeds)))) for w in grid}
    return transfer, sub, {"TOK>E@exposure": inc}, th


def _sub_names(is_p):
    if is_p:
        return ["P_%s" % k for k in HC.POOLS_REL]
    return ["E_%s" % e for e in HC.ENTITIES]


# ---------------------------------------------------------------------------
# (d) presence current + y_on
# ---------------------------------------------------------------------------

def measure_presence(seed):
    """Participant rate at 1500 pA and the nearest-to-0.2x current from
    PRESENCE_GRID; plus y_on from a saturated hearing drive."""
    b, idx = build_cal(seed, "EXPOSURE")
    n = b.core_config.num_neurons
    tgt = unit_idx(idx, "e", 0)

    def rate_at(I):
        b.cp_external_input_current[:] = 0.0
        for _ in range(SETTLE):
            b._run_one_simulation_step()
        cur = np.zeros(n, np.float32)
        cur[tgt] = float(I)
        b.cp_external_input_current[:] = cur
        spk = 0
        for _ in range(TOKEN):
            b._run_one_simulation_step()
            f = np.asarray(b.cp_firing_states).astype(bool)
            spk += int(f[tgt].sum())
        b.cp_external_input_current[:] = 0.0
        return spk / (float(tgt.size) * TOKEN)
    r_part = rate_at(DRIVE)
    grid_r = {I: rate_at(float(I)) for I in PRESENCE_GRID}
    best = min(PRESENCE_GRID,
               key=lambda I: (abs(grid_r[I] - 0.2 * r_part), I))
    end_bridge(b, "presence")
    del b
    b2, idx2 = build_cal(seed, "HEARING", learn_p=W_MAX)
    n2 = b2.core_config.num_neurons
    cur = np.zeros(n2, np.float32)
    drive_token(cur, idx2, TOKM, 0)
    b2.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE):
        b2._run_one_simulation_step()
    b2.cp_external_input_current[:] = cur
    for _ in range(TOKEN):
        b2._run_one_simulation_step()
    tr = np.asarray(b2.cp_hebb_coactivity_trace, dtype=np.float64)
    p_tgt = unit_idx(idx2, "p", 0)
    y_on = float(tr[p_tgt].mean())
    y_on_cells = [round(float(v), 6) for v in tr[p_tgt]]
    end_bridge(b2, "y_on")
    del b2
    conv = {
        "presence_current": float(best),
        "participant_rate": float(r_part),
        "presence_grid_rates": {str(I): float(grid_r[I])
                                for I in PRESENCE_GRID},
        "presence_target_0.2x": float(0.2 * r_part),
        "y_on": y_on,
        "y_on_per_neuron": y_on_cells,
    }
    return conv


# ---------------------------------------------------------------------------
# (b) saturated-route check on seed 61
# ---------------------------------------------------------------------------

def saturated(seed, th):
    out = {"seed": int(seed), "w_max": W_MAX,
           "thresholds_used": {"P": list(th["P"]),
                               "E": list(th["E"])}}
    b, idx = build_cal(seed, "HEARING", learn_p=W_MAX, learn_e=W_MAX)
    n = b.core_config.num_neurons
    cur = np.zeros(n, np.float32)
    drive_token(cur, idx, TOKM, 0)
    hs = HC.hc_slices()

    def run40(drive_h):
        c = cur.copy()
        if drive_h:
            for k in range(len(HC.POOLS_REL)):
                c[idx["h"][hs[(k, 0)]]] = DRIVE
        b.cp_external_input_current[:] = 0.0
        for _ in range(SETTLE):
            b._run_one_simulation_step()
        b.cp_external_input_current[:] = c
        fp = np.zeros(HC.N_P, np.int64)
        fe = np.zeros(HC.N_E, np.int64)
        fc = np.zeros(HC.N_HC, np.int64)
        for _ in range(TOKEN):
            b._run_one_simulation_step()
            f = np.asarray(b.cp_firing_states).astype(bool)
            fp += f[idx["p"]]
            fe += f[idx["e"]]
            fc += f[idx["c"]]
        b.cp_external_input_current[:] = 0.0
        return fp, fe, fc
    fp, fe, fc0 = run40(False)
    _, _, fc1 = run40(True)
    tgt_c = np.zeros(HC.N_HC, bool)
    for k in range(len(HC.POOLS_REL)):
        s = hs[(k, 0)]
        tgt_c[s.start:s.stop] = True
    out["w_max"] = {
        "p_fired_all": bool((fp > 0).all()),
        "p_fired_n": int((fp > 0).sum()),
        "e_fired_all": bool((fe > 0).all()),
        "e_fired_n": int((fe > 0).sum()),
        "c_spikes_no_h": int(fc0.sum()),
        "c_target_fired_n_with_h": int(
            (fc1[tgt_c] > 0).sum()),
        "c_target_all_fired_with_h": bool(
            (fc1[tgt_c] > 0).all()),
        "note": "step-1a REF latch out of 0a scope (DESIGN 333-335)",
    }
    end_bridge(b, "sat_wmax")
    del b
    for tag, lp, le in (("theta_silent", th["P"][1] or 0,
                         th["E"][1] or 0),
                        ("zero", 0, 0)):
        b2, idx2 = build_cal(seed, "HEARING", learn_p=lp, learn_e=le)
        n2 = b2.core_config.num_neurons
        c = np.zeros(n2, np.float32)
        drive_token(c, idx2, TOKM, 0)
        b2.cp_external_input_current[:] = 0.0
        for _ in range(SETTLE):
            b2._run_one_simulation_step()
        b2.cp_external_input_current[:] = c
        sp = np.zeros(HC.N_P, np.int64)
        se = np.zeros(HC.N_E, np.int64)
        for _ in range(TOKEN):
            b2._run_one_simulation_step()
            f = np.asarray(b2.cp_firing_states).astype(bool)
            sp += f[idx2["p"]]
            se += f[idx2["e"]]
        b2.cp_external_input_current[:] = 0.0
        out[tag] = {"p_spikes": int(sp.sum()), "e_spikes": int(se.sum()),
                    "learn_p": float(lp), "learn_e": float(le),
                    "ok": bool(sp.sum() <= 1 and se.sum() <= 1)}
        end_bridge(b2, "sat_%s" % tag)
        del b2
    return out


# ---------------------------------------------------------------------------
# (c) exposure replay: P0 items + (d) conversions
# ---------------------------------------------------------------------------

def replay_p0(seed, head, rows, presence_current, w_hc, w_ec):
    """Exposure-mode replay (packet 490-493): per utterance = tokens 40
    steps back-to-back with the scene driven, then TAIL_OPEN steps with
    every drive off and plasticity open, then a GAP-step quiet gap with
    plasticity closed.  Gates dx_ptr_tx/dx_name_tx read back every step
    (P0 route_tx); the Hebbian-run fraction and trace-floor occurrences
    ((d)) are counted from the same pass.  The circuit runs THE ROW'S
    C-weights (w_hc, w_ec = w_pc) — that is what row selection varies."""
    b, idx = build_cal(seed, "EXPOSURE", w_hc=float(w_hc),
                       w_ec=float(w_ec), w_pc=float(w_ec))
    n = b.core_config.num_neurons
    tok_names = head["TOK"]
    hs = HC.hc_slices()
    npl = len(HC.POOLS_REL)
    nent = len(HC.ENTITIES)
    steps = 0
    any_prev = False
    pair_steps = 0
    above_prev = np.zeros(HC.N_TOK, bool)
    steps_above = 0
    runs = 0
    tx_closed = True
    match = np.zeros((npl, nent), int)
    nonmatch = np.zeros((npl, nent), int)
    match_fire = np.zeros((npl, nent), int)
    nonmatch_fire = np.zeros((npl, nent), int)
    leak_win = np.zeros((npl, nent), int)
    leak_fire = np.zeros((npl, nent), int)
    occ_num = np.zeros(npl, int)
    occ_den = np.zeros(npl, int)
    emp_fire = np.zeros(npl, int)
    sch_hold = np.zeros(npl, int)
    scene_min = 10 ** 9
    scene_fail = 0
    n_utt = 0

    def step():
        nonlocal steps, any_prev, pair_steps, steps_above, runs, \
            tx_closed
        b._run_one_simulation_step()
        steps += 1
        f = np.asarray(b.cp_firing_states).astype(bool)
        anyf = bool(f.any())
        if anyf and any_prev:
            pair_steps += 1
        any_prev = anyf
        for g in ("dx_ptr_tx", "dx_name_tx"):
            if float(b._transmission_gate_values.get(g, 0.0)) != 0.0:
                tx_closed = False
        # the engine rebinds cp_hebb_coactivity_trace each step, so the
        # reference must be fetched fresh (a captured one goes stale)
        trace = np.asarray(b.cp_hebb_coactivity_trace,
                           dtype=np.float64)
        ab = trace[:HC.N_TOK] > PRE_FLOOR
        runs += int((ab & ~above_prev).sum())
        steps_above += int(ab.sum())
        above_prev[:] = ab
        return f

    for row in rows:
        xs, y, S, A, hset = row_parts(row, head)
        n_utt += 1
        c_window = np.zeros((npl, nent), int)
        e_asm = np.zeros(nent, int)
        p_fire = np.zeros(npl, int)
        win_num = np.zeros(npl, int)
        win_den = np.zeros(npl, int)
        scene = scene_drive(idx, y, presence_current, n)
        for tid in xs:
            cur = scene.copy()
            drive_token(cur, idx, tok_names, tid)
            drive_h(cur, idx, hset)
            b.cp_external_input_current[:] = cur
            for _ in range(TOKEN):
                f = step()
                cc = f[idx["c"]]
                for k in range(npl):
                    for e in range(nent):
                        s = hs[(k, e)]
                        c_window[k, e] += int(cc[s.start:s.stop].sum())
                pf = f[idx["p"]]
                ef = f[idx["e"]]
                for k in range(npl):
                    a = k * HC.N_P_UNIT
                    v = pf[a:a + HC.N_P_UNIT]
                    win_num[k] += int((v > 0).any())
                    p_fire[k] |= int((v > 0).any())
                win_den += 1
                for j in range(nent):
                    a = j * HC.N_UNIT
                    e_asm[j] |= int((ef[a:a + HC.N_UNIT] > 0).any())
        b.cp_external_input_current[:] = 0.0
        for _ in range(TAIL_OPEN):
            f = step()
            cc = f[idx["c"]]
            for k in range(npl):
                for e in range(nent):
                    s = hs[(k, e)]
                    c_window[k, e] += int(cc[s.start:s.stop].sum())
            pf = f[idx["p"]]
            for k in range(npl):
                a = k * HC.N_P_UNIT
                v = pf[a:a + HC.N_P_UNIT]
                win_num[k] += int((v > 0).any())
                p_fire[k] |= int((v > 0).any())
            win_den += 1
        # utterance accounting
        for k in range(npl):
            for e in range(nent):
                driven = (k, e) in hset
                parts = float(y[npl + e]) >= 1.0
                fired = int(c_window[k, e] > 0)
                if driven and parts:
                    match[k, e] += 1
                    match_fire[k, e] += fired
                else:
                    nonmatch[k, e] += 1
                    nonmatch_fire[k, e] += fired
                if not driven:
                    leak_win[k, e] += 1
                    leak_fire[k, e] += fired
            if float(y[k]) <= 0.0:      # non-matching utterance only
                occ_num[k] += win_num[k]
                occ_den[k] += win_den[k]
            if p_fire[k]:
                emp_fire[k] += 1
            if float(y[k]) > 0.0:
                sch_hold[k] += 1
        if int(e_asm.sum()) < 2:
            scene_fail += 1
        scene_min = min(scene_min, int(e_asm.sum()))
        b.set_plasticity_gate("dx_ptr", 0.0)
        b.set_plasticity_gate("dx_name", 0.0)
        for _ in range(GAP):
            step()
        b.set_plasticity_gate("dx_ptr", 1.0)
        b.set_plasticity_gate("dx_name", 1.0)

    fails = []
    worst = {"c95_pos": 1.0, "c95_neg": 0.0, "c_leak": 0.0,
             "ptr_occ": 0.0, "base_rate": 0.0}
    for k in range(npl):
        for e in range(nent):
            m, nm = match[k, e], nonmatch[k, e]
            if m:
                p = match_fire[k, e] / m
                worst["c95_pos"] = min(worst["c95_pos"], p)
                if p < 0.95:
                    fails.append("c95")
            if nm:
                p = nonmatch_fire[k, e] / nm
                worst["c95_neg"] = max(worst["c95_neg"], p)
                if p > 0.02:
                    fails.append("c95")
            if leak_win[k, e]:
                lk = leak_fire[k, e] / leak_win[k, e]
                worst["c_leak"] = max(worst["c_leak"], lk)
                if lk > 0.01:
                    fails.append("c_leak")
    for k in (0, 1):                     # P_SPK, P_ADR
        if occ_den[k]:
            o = occ_num[k] / occ_den[k]
            worst["ptr_occ"] = max(worst["ptr_occ"], o)
            if o > 0.02:
                fails.append("ptr_occ")
    base = {}
    for k in range(npl):
        emp = emp_fire[k] / max(n_utt, 1)
        sch = sch_hold[k] / max(n_utt, 1)
        d = abs(emp - sch)
        worst["base_rate"] = max(worst["base_rate"], d)
        base[HC.POOLS_REL[k]] = {"empirical": round(emp, 4),
                                 "schedule": round(sch, 4),
                                 "delta": round(d, 4)}
        if d > 0.05:
            fails.append("base_rate")
    if scene_fail:
        fails.append("e_scene")
    if not tx_closed:
        fails.append("route_tx")
    detail = {
        "n_utterances": n_utt,
        "c95_worst_pos": round(worst["c95_pos"], 4),
        "c95_worst_neg": round(worst["c95_neg"], 4),
        "c_leak_worst": round(worst["c_leak"], 4),
        "ptr_occ_worst": round(worst["ptr_occ"], 4),
        "base_rate_worst_delta": round(worst["base_rate"], 4),
        "base_rates": base,
        "scene_min_assemblies": int(scene_min),
        "scene_fail_utterances": int(scene_fail),
        "route_tx_closed": bool(tx_closed),
        "route_tx_reads": steps,
        "hebbian_run_fraction": pair_steps / max(steps, 1),
        "hebbian_run_steps": pair_steps,
        "steps_total": steps,
        "trace_steps_above": steps_above,
        "trace_occurrences": runs,
        "steps_per_occurrence": (steps_above / max(runs, 1)
                                 if runs else None),
    }
    end_bridge(b, "p0")
    del b
    return sorted(set(fails)), detail


def select_row(seeds, head, loaded, presence_current, rows=None,
               n_utt=None, p0_total=P0_TOTAL):
    rows = ROWS if rows is None else rows
    grid, detail_all, passing, chosen = [], {}, [], None
    chosen_conv, first_conv = None, None
    for idx_r, (w_hc, w_ec) in enumerate(rows):
        fails_all = []
        seed_details = {}
        row_conv = None
        for s in seeds:
            use = loaded[s]
            if n_utt is not None:
                use = use[:n_utt]
            else:
                use = use[:p0_total]
            fails, det = replay_p0(s, head, use, presence_current,
                                   w_hc, w_ec)
            seed_details[str(s)] = {"fails": fails, **det}
            if row_conv is None:
                row_conv = {k: det[k] for k in
                            ("hebbian_run_fraction", "hebbian_run_steps",
                             "steps_total", "trace_steps_above",
                             "trace_occurrences",
                             "steps_per_occurrence")}
            fails_all.extend(fails)
        fails_all = sorted(set(fails_all))
        if first_conv is None:
            first_conv = row_conv
        grid.append({"w": [w_hc, w_ec], "index": idx_r,
                     "p0_fails": fails_all})
        detail_all["R%d" % idx_r] = seed_details
        if not fails_all:
            passing.append([w_hc, w_ec])
            if chosen is None:
                chosen = [w_hc, w_ec]
                chosen_conv = row_conv
        if chosen is not None:
            break                       # first passing row, in order
    if chosen_conv is None:
        chosen_conv = first_conv        # no row passed: first row's
    for idx_r, (w_hc, w_ec) in enumerate(rows):
        if not any(g["w"] == [w_hc, w_ec] for g in grid):
            grid.append({"w": [w_hc, w_ec], "index": idx_r,
                         "p0_fails": ["not-evaluated"]})
    cw = {"grid": grid, "passing_rows": passing, "row": chosen,
          "p0_items": list(P0_ITEMS), "detail": detail_all,
          "utterances_per_seed": (n_utt if n_utt is not None
                                  else p0_total)}
    return cw, (chosen_conv or {})


# ---------------------------------------------------------------------------
# (e) per-arm step time
# ---------------------------------------------------------------------------

def arm_step_time(seed, head, rows, arm, presence_current,
                  n_steps=ARM_STEPS):
    """One arm: 2000-ish exposure steps on seed 61 with that arm's update
    code at lr 0 and the per-step gate read-back, timed end to end."""
    cfg = ARM_CFG[arm]
    b, idx = build_cal(seed, "EXPOSURE",
                       wta=cfg["wta"], rate_window=cfg["rate_window"])
    if cfg["bcm"]:
        b.core_config.hebbian_bcm = float(cfg["bcm"])
        # The engine's BCM branch (sim/bridge.py:10265) multiplies by
        # the plasticity-gate gain, NOT by hebbian_learning_rate, so
        # lr=0 alone does not freeze it.  The packet's plasticity-off
        # rule (64-67) requires byte-frozen learned blocks: zero the
        # gate gain for this timing run.  The branch's arithmetic
        # (selection, theta EMA read, dw compute, clip) still executes
        # every step — only the final gain multiply zeroes it.
        b.set_plasticity_gate("dx_ptr", 0.0)
        b.set_plasticity_gate("dx_name", 0.0)
    n = b.core_config.num_neurons
    tok_names = head["TOK"]
    ybar = np.zeros(n, dtype=np.float64) if cfg["runner"] else None
    shapes = {"dx_ptr": (idx["tok"], idx["p"]),
              "dx_name": (idx["tok"], idx["e"])}
    steps_done = 0

    def step_once(runner):
        b._run_one_simulation_step()
        if runner and cfg["rate_window"]:
            # fresh fetch: the engine rebinds the trace array per step
            trace = np.asarray(b.cp_hebb_coactivity_trace,
                               dtype=np.float64)
            runner_update(runner, b, trace, ybar, shapes)
        HC.readback(b)

    t0 = time.perf_counter()
    for row in rows:
        if steps_done >= n_steps:
            break
        xs, y, S, A, hset = row_parts(row, head)
        scene = scene_drive(idx, y, presence_current, n)
        for tid in xs:
            if steps_done >= n_steps:
                break
            cur = scene.copy()
            drive_token(cur, idx, tok_names, tid)
            drive_h(cur, idx, hset)
            b.cp_external_input_current[:] = cur
            for _ in range(TOKEN):
                if steps_done >= n_steps:
                    break
                step_once(cfg["runner"])
                steps_done += 1
        b.cp_external_input_current[:] = 0.0
        tail = 0
        while tail < TAIL_OPEN and steps_done < n_steps:
            step_once(cfg["runner"])
            steps_done += 1
            tail += 1
        b.set_plasticity_gate("dx_ptr", 0.0)
        b.set_plasticity_gate("dx_name", 0.0)
        g = 0
        while g < GAP and steps_done < n_steps:
            step_once(cfg["runner"])
            steps_done += 1
            g += 1
        b.set_plasticity_gate("dx_ptr", 1.0)
        b.set_plasticity_gate("dx_name", 1.0)
    t1 = time.perf_counter()
    per_step = (t1 - t0) / max(steps_done, 1)
    end_bridge(b, "arm_%s" % arm)
    del b
    detail = {"seconds_per_step": float(per_step), "steps": int(steps_done),
              "plasticity_gate_zeroed": bool(cfg["bcm"])}
    if cfg["bcm"]:
        detail["plasticity_gate_zeroed_note"] = (
            "engine BCM dw bypasses hebbian_learning_rate "
            "(bridge:10265); gate gain zeroed for byte-frozen weights, "
            "branch arithmetic still runs every step")
    return float(per_step), int(steps_done), detail


def runner_update(rule, b, trace, ybar, shapes):
    """COV / COV-PRE per-step update over the plastic blocks with the lr
    factor 0: the arithmetic executes (this is what (e) times), the
    written bytes cannot change.  The block matrix is (npre x npost); the
    nnz order is insertion order, which is pre-major for _dense — at lr 0
    a layout mismatch would still be byte-inert, but the reshape is kept
    faithful so a future nonzero-lr run cannot silently misassign."""
    for gate, (pre_arr, post_arr) in shapes.items():
        ii = np.asarray(b._plasticity_gate_to_synapses[gate])
        npre, npost = len(pre_arr), len(post_arr)
        w = b.cp_connections.data[ii].reshape(npre, npost)
        x = trace[pre_arr][:, None]
        y = trace[post_arr][None, :]
        sel = x > PRE_FLOOR
        if rule == "COV":
            dw = LAM * x * (y - ybar[post_arr][None, :])
        else:
            dw = LAM * x * (
                W_MAX * (y - ybar[post_arr][None, :]) - KAP * w)
        upd = w + np.where(sel, dw, 0.0) * 0.0    # lr = 0
        b.cp_connections.data[ii] = upd.ravel()
        ybar[post_arr] += YBAR_ALPHA * (trace[post_arr]
                                        - ybar[post_arr])


# ---------------------------------------------------------------------------
# (f) heterogeneity readback
# ---------------------------------------------------------------------------

def heterogeneity(seed):
    b, idx = build_cal(seed, "EXPOSURE")
    cfg = b.core_config
    flags = {
        "enable_parameter_heterogeneity":
            bool(cfg.enable_parameter_heterogeneity),
        "preset_centred_parameter_heterogeneity":
            bool(cfg.preset_centred_parameter_heterogeneity),
        "hebbian_learning_rate": float(cfg.hebbian_learning_rate),
        "hebbian_weight_decay": float(cfg.hebbian_weight_decay),
        "hebbian_rate_window": bool(cfg.hebbian_rate_window),
    }
    med = {}
    for pop in ("tok", "e", "fs", "h", "c", "p"):
        sl = idx[pop]
        row = {}
        for nm in ("cp_izh_a", "cp_izh_b", "cp_izh_d", "cp_izh_C"):
            arr = getattr(b, nm, None)
            if arr is None:
                continue
            row[nm.replace("cp_izh_", "")] = float(
                np.median(np.asarray(arr)[sl]))
        med[pop] = row
    end_bridge(b, "heterogeneity")
    del b
    return {"flags": flags, "medians": med}


# ---------------------------------------------------------------------------
# record assembly / output
# ---------------------------------------------------------------------------

def assemble(seeds, grid, streams_manifest, streams_sha, transfer, sub,
             expo_inc, th, sat, cw, conv, step_times, step_detail,
             het, timing):
    return {
        "schema": SCHEMA,
        "seeds": [int(s) for s in seeds],
        "grid": [int(w) for w in grid],
        "rows": [list(r) for r in ROWS],
        "streams_manifest": os.path.basename(streams_manifest),
        "streams_manifest_sha256": streams_sha,
        "thresholds": {k: [v[0], v[1]] for k, v in th.items()},
        "thresholds_source": {
            "rule": "packet 74-77 / DESIGN 2.6 (243-313), ported from "
                    "s1-scratch/fix2/calib_proto.py:133-142; pooled "
                    "over every pool/assembly = max over per-unit "
                    "thresholds per RULING deixis-271-s1-thr-1022",
            "grid": [int(w) for w in grid],
            "conditions": {"P": ["TOK>P@base"],
                           "P_wta": ["TOK>P@wta"],
                           "E": ["TOK>E@production", "TOK>E@toy"]},
            "targets": {"P": "ALL five P pools (P_0..P_4, 60 cells); "
                              "theta = max over per-pool thresholds",
                        "E": "ALL ten entity assemblies (E_0..E_9, "
                             "120 cells), conjunct production AND toy"},
            "delegate_ruling":
                "MiMo Pro 2026-10-22 decision A (WTA pair separate as "
                "P_wta; sub_target_curves + exposure increment outside "
                "transfer) as corrected by RULING "
                "deixis-271-s1-thr-1022: thresholds over every pool, "
                "not the first (first-pool pinning replaced)",
            "proto": "docs/drafts/deixis/s1-scratch/fix2/calib_proto.py",
        },
        "transfer": transfer,
        "sub_target_curves": sub,
        "exposure_increment": expo_inc,
        "saturated_check": sat,
        "c_weight": cw,
        "conversions": conv,
        "step_times": step_times,
        "step_times_detail": step_detail,
        "heterogeneity": het,
        "plasticity_off": plasticity_record(),
        "timing": timing,
    }


def write_out(rec, out_path, single_seed, timing):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    body = dict(rec)
    # Single-seed identity runs must be byte-identical (packet 108-109):
    # no wall-clock in the main JSON; timings live in the sidecar.
    if single_seed:
        body.pop("step_times", None)
        body.pop("step_times_detail", None)
        body.pop("timing", None)
    with open(out_path, "w") as f:
        json.dump(body, f, sort_keys=True, indent=1)
        f.write("\n")
    tpath = out_path + ".timing.json"
    with open(tpath, "w") as f:
        json.dump(timing, f, sort_keys=True, indent=1)
        f.write("\n")
    declare_output(tpath)
    return out_path, tpath


# ---------------------------------------------------------------------------
# main / selftest
# ---------------------------------------------------------------------------

def run_calibration(seeds, manifest_path, out_path, grid=None,
                    rows=None, n_utt=None, p0_total=P0_TOTAL,
                    arm_steps=ARM_STEPS):
    grid = GRID if grid is None else grid
    man, man_sha = load_manifest(manifest_path, set(seeds))
    loaded, heads = {}, {}
    for s in seeds:
        head, rws = load_stream(man[s]["path"])
        heads[s] = head
        loaded[s] = rws
    head = heads[seeds[0]]
    timing = {}
    t0 = time.perf_counter()
    conv = measure_presence(seeds[0])
    timing["presence_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    transfer, sub, expo_inc, th = run_curves(
        seeds, conv["presence_current"], head, grid)
    timing["curves_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cw, p0_conv = select_row(seeds, head, loaded,
                             conv["presence_current"],
                             rows=rows, n_utt=n_utt,
                             p0_total=p0_total)
    timing["p0_seconds"] = time.perf_counter() - t0
    conv.update(p0_conv)
    t0 = time.perf_counter()
    sat = saturated(seeds[0], th)
    timing["saturated_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    step_times, step_detail = {}, {}
    for arm in ARMS:
        per_step, done, dtail = arm_step_time(
            seeds[0], head, loaded[seeds[0]], arm,
            conv["presence_current"], n_steps=arm_steps)
        step_times[arm] = per_step
        step_detail[arm] = dtail
    timing["step_times_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    het = heterogeneity(seeds[0])
    timing["heterogeneity_seconds"] = time.perf_counter() - t0
    timing["total_seconds"] = sum(timing.values())
    rec = assemble(seeds, grid, manifest_path, man_sha, transfer, sub,
                   expo_inc, th, sat, cw, conv, step_times, step_detail,
                   het, timing)
    return write_out(rec, out_path, len(seeds) == 1, timing)


def selftest():
    """Tiny end-to-end: synthetic 3-utterance stream (with `h`), 3-point
    grid, 1 seed, 2 rows, 40-step arms.  Proves every record key module 6
    reads, the identity/timing rule, the `h` refusal, and one genuine
    tools.lab.attributable_to call (attribution gate)."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="deixis_calib_",
                           dir="/opt/data/cache/scratch")
    from research.runners._deixis_rule_replica import TOKM, POST
    head = {"schema": "deixis-toy-stream-v1", "seed": 61, "length": 3,
            "arm": "calib", "TOK": TOKM, "POST": POST,
            "world_kw": {}, "vocab": TOKM, "nonces": {}}
    rows = []
    for u in range(3):
        rows.append({"x": [0, 10 + u % 7],
                     "y": [[0, 1.0], [1, 1.0], [2, 0.0], [3, 0.0],
                           [4, 0.0], [5, 1.0], [6, 1.0], [7, 0.2],
                           [8, 0.2], [9, 0.2], [10, 0.0], [11, 0.0],
                           [12, 0.0], [13, 0.0], [14, 0.0]],
                     "S": 0, "A": 1,
                     "H": [[0, 0], [0, 1], [1, 1]]})
    spath = os.path.join(tmp, "calib_seed61.json")
    with open(spath, "w") as f:
        f.write(json.dumps(head, sort_keys=True) + "\n")
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    man = {"schema": "deixis-streams-manifest-v1", "arm": "calib",
           "length": 3,
           "streams": [{"file": "calib_seed61.json", "seed": 61,
                        "sha256": sha256_file(spath)}]}
    mpath = os.path.join(tmp, "manifest_calib.json")
    with open(mpath, "w") as f:
        json.dump(man, f, sort_keys=True)

    global GRID, ARM_STEPS, ROWS
    sav = (GRID, ARM_STEPS, ROWS)
    GRID = [0, 4, 8]
    ARM_STEPS = 40
    ROWS = [(6, 2), (8, 2)]
    ok = []
    try:
        out = os.path.join(tmp, "calibration_s61_a.json")
        out_path, tpath = run_calibration(
            [61], mpath, out_path=out, n_utt=3, arm_steps=40)
        rec = json.load(open(out_path))
        th = rec["thresholds"]
        for k in ("P", "E"):
            assert isinstance(th[k], list) and len(th[k]) == 2, \
                "thresholds.%s malformed: %r" % (k, th[k])
        assert "P_wta" in th, "P_wta missing"
        assert set(rec["transfer"]) == {
            "TOK>P@base", "TOK>P@wta", "TOK>E@production",
            "TOK>E@toy"}, set(rec["transfer"])
        for k, ent in rec["transfer"].items():
            assert isinstance(ent["fired_all_by_20"], bool), k
            assert isinstance(ent["silent_below_theta"], bool), k
            assert "theta_fire" in ent and "curves" in ent, k
        assert "exposure_increment" in rec, "expo inc missing"
        assert "exposure_increment" not in rec["transfer"]
        assert "sub_target_curves" in rec
        assert "sub_target_curves" not in rec["transfer"]
        # RULING deixis-271-s1-thr-1022: every unit present with its
        # own theta pair; pooled theta = max over per-unit theta_fire
        subr = rec["sub_target_curves"]
        for key, names in (("TOK>P@base",
                            ["P_%s" % k for k in HC.POOLS_REL]),
                           ("TOK>P@wta",
                            ["P_%s" % k for k in HC.POOLS_REL]),
                           ("TOK>E@production",
                            ["E_%s" % e for e in HC.ENTITIES]),
                           ("TOK>E@toy",
                            ["E_%s" % e for e in HC.ENTITIES])):
            assert set(subr[key]) == set(names), (key, set(subr[key]))
            for nm in names:
                ent = subr[key][nm]
                assert "curves" in ent and "theta" in ent, (key, nm)
                assert isinstance(ent["theta"], list) \
                    and len(ent["theta"]) == 2, (key, nm)
            pf = rec["transfer"][key]["theta_fire"]
            unit_fires = [subr[key][nm]["theta"][0] for nm in names]
            if any(x is None for x in unit_fires):
                assert pf is None, (key, pf, unit_fires)
            else:
                assert pf == max(unit_fires), (key, pf, unit_fires)
        p0 = rec["plasticity_off"]
        assert p0["unchanged"] is True, p0
        assert set(p0["blocks"]) == {"dx_ptr", "dx_name"}, p0
        assert p0["checks"], "no plasticity checks ran"
        cw = rec["c_weight"]
        assert cw["grid"] and "passing_rows" in cw and "row" in cw
        for g in cw["grid"]:
            assert isinstance(g["p0_fails"], list), g
        assert set(cw["p0_items"]) == set(P0_ITEMS), cw["p0_items"]
        assert "step_times" not in rec, "single-seed leaked step_times"
        assert "timing" not in rec and "step_times_detail" not in rec
        tim = json.load(open(tpath))
        assert "curves_seconds" in tim, "timing sidecar empty"
        # multi-seed branch embeds step_times (module-6 contract)
        rec2 = assemble([61, 62], GRID, mpath, "x", {}, {}, {}, {},
                        {}, {}, {}, {"CAUSAL": 0.001}, {}, {}, {})
        assert rec2["step_times"] == {"CAUSAL": 0.001}
        # identity: a second single-seed run is byte-identical
        outb = os.path.join(tmp, "calibration_s61_b.json")
        run_calibration([61], mpath, out_path=outb, n_utt=3,
                        arm_steps=40)
        a = open(out_path, "rb").read()
        bb = open(outb, "rb").read()
        assert a == bb, "single-seed runs are not byte-identical"
        ok.append("identity_byte_identical")
        # genuine attribution call (attribution gate)
        if attributable_to is not None:
            # genuine call, scalar reductions: does matching-utterance
            # C firing beat a label-shuffled control (honest verdict on
            # toy numbers, typically UNDEFINED)?
            t = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            c = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
            r = attributable_to("calib_c_match",
                                float(t.sum()), float(c.sum()))
            ok.append("attributable_to=%s" % (r,))
        # `H` refusal names the NEEDS key
        bad = os.path.join(tmp, "bad.json")
        with open(bad, "w") as f:
            f.write(json.dumps(head, sort_keys=True) + "\n")
            rr = dict(rows[0])
            rr.pop("H")
            f.write(json.dumps(rr, sort_keys=True) + "\n")
        try:
            load_stream(bad)
            raise AssertionError("missing `H` was not refused")
        except ValueError as e:
            assert NEEDS_KEY in str(e), e
            ok.append("refused_missing_h")
        ok.append("record_contract")
        print("selftest ok: " + ", ".join(sorted(set(ok))), flush=True)
        return 0
    finally:
        GRID, ARM_STEPS, ROWS = sav


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="deixis-s1 0a(i) calibration (packet command 107)")
    ap.add_argument("--seeds")
    ap.add_argument("--streams")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    missing = [n for n, v in (("--seeds", args.seeds),
                              ("--streams", args.streams),
                              ("--out", args.out)) if not v]
    if missing:
        raise SystemExit("missing required arguments: %s"
                         % ", ".join(missing))
    seeds = spec_rng(args.seeds)
    out_path, tpath = run_calibration(seeds, args.streams, args.out)
    print("wrote %s (+ %s)" % (out_path, tpath), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())