"""Scorer for cs-race f1j STEP 0: identity mode (after wave 1) and the full step-0 read (after wave 2, plus wave 3).

Packet: research/coordination/cs-race-f1j-PACKET.md (Step 0: P0', grid, arms, identity (i)-(iv), items C0-C8 with C7(c)
rewritten, per-NMDA-cell reads, rescue attribution, selection, selected-cell NMDA attribution, STOP rules, falsifiers
F1-F4, decision tree; K1; guards S1-S14). It extends the f1i scorer (research/runners/cs_race_f1i_score.py), whose
item readings C0-C6, C7 (0)/(a)/(b)/(d) and C8, artifact validity, schedule and consistency checks it reuses unchanged.
It adds no bar and loosens none. It reads data only and runs no brain.

    python -m research.runners.cs_race_f1j_score --mode identity --pin <sha>     # after wave 1 (90 lines)
    python -m research.runners.cs_race_f1j_score --mode full --pin <sha> [--out research/findings/raw/_cs_race/f1j/score/x.json]

It must run on a checkout that holds research/findings/raw/_cs_race/f1i/step0/ (landed c8c9a2595): the 45 early, 6 late
and 21 passing f1i C6 records and the 23 defined corner pairs are RE-DERIVED from those artifacts (a count other than
45 / 6 / 21 / 23 is STOP-ID), and identity (i) / (iii) compare against them.

HOW EACH RULE IS READ (the bar text is the packet's):
  ARTIFACT VALIDITY as f1i (JSON + .prov.json parse; clean run of --pin, git_archive, manifest verified; sidecar argv ==
      the pool line naming this output; schedule c1 [150, 800), c3 X [0, 150), Y [300, 450), window [600, 950)), plus
      the record's flags naming this arm, cell, seed and f1j flags (flags.f1j). An invalid artifact makes every item it
      feeds UNDEFINED, never a pass or a fail. In the full read a count (rescue, late, kept, C6, eraser, IPSC-12) would
      take an invalid or missing record as a shortfall, so any invalid or missing wave 1-2 artifact makes the whole
      read UNDEFINED ("re-run the N invalid or missing artifact(s), then re-score; not a pass, not a falsifier"), as
      identity mode's IDENTITY UNDEFINED; F1-F4, the selection and branches 2-9 are read on complete data only.
      STOP-ENGINE / STOP-ID (branch 1) still come first. NOMEM (wave 3, read after the decision) is UNDEFINED alone.
  IDENTITY (i): the no-flag record minus wall_s hashes equal to the landed f1i cal_sa0p125_ms215_s<41|45>.json.
  IDENTITY (ii): every LESION record: g_ap 0 at every ms and the cs_som gate 0.0 at measurement (f1i read-backs).
  IDENTITY (iii): every APV record, with its f1j fields removed (flags.f1j, config_readback.f1j, c5.f1j_nmda, every
      trial's "nmda"; these are the f1j echo and read-backs), hashes equal minus wall_s to the landed f1i TRACE record
      of the same cell and seed, and its f1j fields read g_N 0.0 at every ms (S11). A mismatch names the host pair
      (tools/where_ran.py) and, when the two host classes differ, is HOST-CLASS PENDING (re-run on the reference's
      class) rather than STOP-ID; only a mismatch on the same host class is STOP-ID.
  IDENTITY (iv): S1, S2, S3, S7 (the f1i read-backs) and S9-S14 hold on every f1j --som record (S14 on --nmda-g0 > 0).
      The identity (i) records carry no f1j field by construction (their hash equality to f1i is their check).
  S9  every trial's nmda.reset_readback == {"s_max": 0.0} (c1, c2b, the three c3, both C5 races).
  S10 every trial: g_N over the mention cells that received no drive is 0.0 at every ms (sparse list empty, run max
      0.0), and over every undriven neuron the run max is 0.0.
  S11 every --nmda-g0 0 record: g_N 0.0 at every recorded ms (pulsed words, undriven cells, the races), AND each pulsed
      word's per-ms s max reaches s_full (rel 1e-5) by its last pulse ms (c1 vole 149; c3 X 149, Y 449). A missing s
      field is UNDEFINED and counts as STOP-ID.
  S12 covered by the runner test at the pin (tests/test_cs_race_f1j_flags.py) plus the S9 read-backs.
  S13 config.syn_tau_g_i == 12 on the IPSC-12 arms and 10 on every other arm; decay_i == exp(-1 / syn_tau_g_i).
  S14 every --nmda-g0 G > 0 record: (a) enable_apical_nmda True, g0 == G, tau == T (from the stem), drive_ref == 120;
      (b) the c1 vole g_N at ms 149 == G (rel 1e-5); (c) for every ms t >= 149 with g_N(t) > 1e-30,
      g_N(t+1)/g_N(t) == exp(-1/T) (rel 1e-5). A missing field is UNDEFINED and counts as STOP-ID.
  ITEMS at each TRACE cell (9 NMDA cells x W_MS 215, 304; W_SA 0.125), 8/8 = eight defined passes: C0-C6 and C8 as
      f1i; C7 (0)/(a)/(b)/(d) as f1i with the LESION record of the same NMDA cell, W_MS and seed; C7(c) REWRITTEN
      (RULING f1i-0-1028): every SOM cell >= 1 spike in [300, 600) AND the older slot's first g_ap increase after Y_on
      (the f1i scorer's g_ap_onset_ms) precedes its termination ms; with no termination ms, the spikes and the older
      g_ap peak over [300, 950) > 0. The peak and its ms are reported, never barred.
  PER NMDA CELL (TRACE + GRID at the 10 ms IPSC against the APV line of the same f1i cell and seed):
      rescue /45: f1i early records that FAIL C6 in APV and PASS C6 in TRACE / GRID, AND (attribution) whose TRACE /
          GRID max g_ap on the c1 pulsed slot over [150, the APV twin's fall ms] is >= 0.32 g_L AND >= 0.8x the APV
          twin's max over the same window. Other rescued records are "rescued by IPSP removal", listed apart, not
          counted. Second reading reported beside it (never the bar): the 0.3189 g_L floor the packet's parenthesis
          names. An APV twin that fails C6 with no fall ms has no window: UNDEFINED attribution, listed apart.
      late /6 (the same rule on the 6 late records), kept /21 (f1i C6 passes still passing), C6 /72, grid C7(a) over
          defined pairs and C7(b) over 216, eraser /23 (the 23 corner pairs defined in f1i, each a defined C7(a) pass
          in the corner TRACE record; seed 50 frog>vole reported apart). working = rescue >= 40 AND eraser 23/23.
  SELECTION: among working NMDA cells with kept 21/21, the TRACE cell with C0-C4 and C6-C8 at 8/8; W_MS 215 before
      304, then the smallest gN0, then the shortest tauN.
  SELECTED-CELL NMDA ATTRIBUTION: the selected cell's APV twin (APV at W_SA 0.125 and its W_MS; C7(d) from the landed
      f1i LESION record of that cell and seed, the APV twin being identical to f1i by (iii)) is scored on the same items;
      seeds / items / pairs where APV fails and TRACE passes are listed; if that is only the seed-50 frog>vole pair, the
      label "NMDA-attributed on 1/24 pairs; C6 rescue evidence is GRID-only" is set.
  STOP: STOP-ENGINE (P0' files at the pin), STOP-ID ((i)-(iv), invalid f1i references, a re-derived count != the
      packet's), STOP-CAL (no selection), STOP-SMOKE (median C5 < 2 pA in either clause at the selected cell OR over
      the 144 TRACE records; either fires), STOP-ATTRIB (every C3-passing TRACE cell UNATTRIBUTED: C3 8/8 while C7(c)
      or C7(d) is not 24/24), STOP-FALS.
  F1: every NMDA cell with eraser 23/23 has rescue < 40 (including no such cell). F2: >= 1 working cell and every
      working cell has tauN 40. F3: >= 1 working cell and none passes, at --tau-gi 12, C6 8/8 at W_MS 215 AND at 304
      AND (W_MS 215) C7(a) on every defined corner pair with >= 23 corner pairs defined and passing. F4: >= 1 working
      cell and every working cell has gN0 16. K1 (H-K1 falsifier): (a) a late C6 fail in the APV or GRID arms with 0
      SOM spikes in [200, fall ms), or (b) K1-NOFAC late C6 fails >= APV late C6 fails on the same 16 records. Each
      reads True / False / UNDEFINED (None): (a) is True on one hit, else UNDEFINED while any APV / GRID record is
      invalid or missing (its count is reported); (b) is UNDEFINED unless all 16 K1-NOFAC and all 16 APV records are
      valid with a defined C6.
  DECISION TREE: the packet's branches 1-9 in order; the first that applies is named, every fired item listed (branch 1
      alone: "Nothing else is read"). The f1i host summary covers every landed f1i step-0 artifact (the packet's quote).
  NOMEM (wave 3, informational): rescue counted the same way per working gN0 over the 45 early records.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import math
import os
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import cs_race_f1i_score as F1I  # noqa: E402
from tools.lab import attributable_to  # noqa: E402  (A-NMDA's share of the C6 passes vs APV: gates/attribution_required)
from research.runners import cs_race_f1j_lines as LN  # noqa: E402

PACKET = "research/coordination/cs-race-f1j-PACKET.md"
STEP0_DIR = LN.OUT_DIR
F1I_DIR = LN.F1I_DIR
LINES = LN.LINES
LINES_WAVE3 = LN.LINES_WAVE3

SEEDS = LN.SEEDS
IDENTITY_SEEDS = LN.IDENTITY_SEEDS
W_SA, W_MS = LN.W_SA, LN.W_MS
TAU_N, G_N0 = LN.TAU_N, LN.G_N0
NMDA_CELLS = LN.NMDA_CELLS
PRIMARY_MS = LN.PRIMARY_MS
F1I_CELLS = LN.F1I_CELLS
K1_CELLS = LN.K1_CELLS
CORNER = (0.125, 215)
CORNER_UNDEF_PAIR = (50, "frog", "vole")
PAIRS = F1I.PAIRS
C1_WORD = F1I.C1_WORD
V_HOLD = F1I.V_HOLD
PASS, FAIL, UNDEF = F1I.PASS, F1I.FAIL, F1I.UNDEF
ITEMS_SEL = F1I.ITEMS_SEL
SUBS = F1I.SUBS

# frozen numbers of the packet
N_EARLY, N_LATE, N_PASS, N_CORNER_DEF = 45, 6, 21, 23
RESCUE_BAR = 40
RESCUE_GAP_FLOOR = 0.32           # g_L: the bar text
RESCUE_GAP_FLOOR_ALT = 0.3189     # the parenthesis' exact minimum (second reading, never the bar)
RESCUE_APV_FRAC = 0.8
APICAL_PULSE = 120.0
APV_TAU = 50.0
NOMEM_TAU = float(LN.NOMEM_TAU)
TAU_GI_BASE, TAU_GI_IPSC = 10.0, 12.0
REL = 1e-5
G_TINY = 1e-30
SOM_TAIL = (126, 201)             # SOM spikes in (125, 200]
LATE_SOM_FROM = 200
C1_LAST = F1I.C1_PULSE[1] - 1     # 149
X_LAST = F1I.X_PULSE[1] - 1       # 149
Y_LAST = F1I.Y_PULSE[1] - 1       # 449

# The unrecorded open-loop replay's predictions at IPSC 10 (packet section R, PREDICTED, never measured):
# (tauN, gN0) -> rescue /45, late /6, kept /21, C6 /72, corner (a) /24, grid (a) passes / defined, grid (b) /216.
PREDICTED = {
    None: (0, 0, 21, 21, 23, (82, 82), 62),
    (40, 4): (36, 1, 21, 58, 24, (189, 189), 170), (40, 8): (42, 2, 21, 65, 24, (212, 212), 197),
    (40, 16): (45, 2, 21, 68, 24, (216, 216), 208), (50, 4): (38, 2, 21, 61, 24, (192, 192), 172),
    (50, 8): (43, 2, 21, 66, 24, (213, 213), 203), (50, 16): (45, 3, 21, 69, 24, (216, 216), 212),
    (75, 4): (39, 3, 21, 63, 24, (195, 195), 176), (75, 8): (44, 5, 21, 70, 24, (214, 214), 212),
    (75, 16): (45, 5, 21, 71, 20, (212, 216), 215),
}


# ── names ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def art_name(arm, seed, sa=None, ms=None, tau=None, g0=None):
    c = LN.cell_tag(sa, ms) if sa is not None else None
    if arm == "identity":
        return "identity_f1i_%s_s%d.json" % (LN.cell_tag(*CORNER), seed)
    if arm in ("apv", "k1nofac", "ipsc12_apv"):
        return "%s_%s_s%d.json" % (arm, c, seed)
    if arm == "nomem":
        return "nomem_g%d_%s_s%d.json" % (int(g0), c, seed)
    return "%s_%s_%s_s%d.json" % (arm, LN.nmda_tag(tau, g0), c, seed)


def arm_spec(arm, tau=None, g0=None):
    """(the f1j flag echo the record must carry, som_lesion, som_nofac, the realized syn_tau_g_i, g0, tauN)."""
    if arm == "identity":
        return None, False, False, TAU_GI_BASE, None, None
    if arm in ("apv", "k1nofac"):
        return ({"nmda_g0": 0.0, "nmda_tau": APV_TAU}, False, arm == "k1nofac", TAU_GI_BASE, 0.0, APV_TAU)
    if arm == "ipsc12_apv":
        return ({"nmda_g0": 0.0, "nmda_tau": APV_TAU, "tau_gi": TAU_GI_IPSC}, False, False, TAU_GI_IPSC, 0.0,
                APV_TAU)
    if arm == "nomem":
        return {"nmda_g0": float(g0), "nmda_tau": NOMEM_TAU}, False, False, TAU_GI_BASE, float(g0), NOMEM_TAU
    echo = {"nmda_g0": float(g0), "nmda_tau": float(tau)}
    if arm == "ipsc12":
        echo["tau_gi"] = TAU_GI_IPSC
    return (echo, arm == "lesion", False, TAU_GI_IPSC if arm == "ipsc12" else TAU_GI_BASE, float(g0), float(tau))


# ── the landed f1i reference lists ─────────────────────────────────────────────────────────────────────────────────

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def f1i_c6_records(f1i_root):
    """The f1i TRACE records' C6 classes, re-derived from the landed artifacts: early (FAIL, the c1 plateau first below
    v_hold in [150, 200)), late (FAIL at 200 or after), pass. Lists of (W_SA, W_MS, seed); drop ms per record."""
    out = {"early": [], "late": [], "pass": [], "undefined": [], "drop_ms": {}, "problems": []}
    for sa, ms in F1I_CELLS:
        for s in SEEDS:
            name = F1I.art_name("trace", s, sa, ms)
            try:
                rec = _load_json(os.path.join(f1i_root, name))
                it = F1I.seed_items(rec["per_seed"][0], None)
            except Exception as e:  # noqa: BLE001 - a missing reference is reported, never guessed
                out["problems"].append("f1i %s unreadable: %s" % (name, e))
                continue
            key = (sa, ms, s)
            out["drop_ms"][key] = it.get("c6_drop_ms")
            if it["C6"] == FAIL:
                d = it.get("c6_drop_ms")
                (out["early"] if d is not None and d < LATE_SOM_FROM else out["late"]).append(key)
            elif it["C6"] == PASS:
                out["pass"].append(key)
            else:
                out["undefined"].append(key)
    return out


def f1i_corner_pairs(f1i_root):
    """The corner cell's (W_SA 0.125, W_MS 215) C3 pairs in f1i: defined (C7(0) true) and undefined, (seed, X, Y)."""
    out = {"defined": [], "undefined": [], "problems": []}
    for s in SEEDS:
        name = F1I.art_name("trace", s, *CORNER)
        try:
            ps = _load_json(os.path.join(f1i_root, name))["per_seed"][0]
            for (x, y), c in zip(PAIRS, ps["c3"]):
                p = F1I.c7_pair(c, None, x, y)
                (out["defined"] if p["0"] else out["undefined"]).append((s, x, y))
        except Exception as e:  # noqa: BLE001
            out["problems"].append("f1i %s unreadable: %s" % (name, e))
    return out


# ── stripping the f1j fields (identity (iii)) ──────────────────────────────────────────────────────────────────────

def strip_f1j(rec):
    """A deep copy of RECORD without the f1j echo and read-backs: flags.f1j, per_seed[*].config_readback.f1j,
    per_seed[*].c5.f1j_nmda and every trial's "nmda"."""
    r = json.loads(json.dumps(rec))
    (r.get("flags") or {}).pop("f1j", None)
    for ps in r.get("per_seed") or []:
        (ps.get("config_readback") or {}).pop("f1j", None)
        (ps.get("c5") or {}).pop("f1j_nmda", None)
        for key in ("c1", "c2b"):
            if isinstance(ps.get(key), dict):
                ps[key].pop("nmda", None)
        for c in ps.get("c3") or []:
            if isinstance(c, dict):
                c.pop("nmda", None)
    return r


def _first_diffs(a, b, path="", out=None, limit=8):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k == "wall_s" and not path:
                continue
            if k not in a or k not in b:
                out.append("%s/%s present in one record only" % (path, k))
            else:
                _first_diffs(a[k], b[k], "%s/%s" % (path, k), out, limit)
            if len(out) >= limit:
                break
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s length %d != %d" % (path, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                _first_diffs(x, y, "%s[%d]" % (path, i), out, limit)
                if len(out) >= limit:
                    break
    elif a != b:
        out.append("%s: %r != %r" % (path, a, b))
    return out


# ── flags and guards ───────────────────────────────────────────────────────────────────────────────────────────────

def flag_problems(rec, arm, seed, sa, ms, tau=None, g0=None):
    """The record's own flags name this arm, cell, seed and f1j flags (base cell f1h-0-1027 + the f1i SOM path)."""
    p = []
    fl = rec.get("flags") or {}
    want = {"W_IM": 40.0, "W_MI": 20.0, "W_MM": 0.0, "NIM": 8, "trace": "intrinsic", "soma_g": 40.0,
            "apical_pa": APICAL_PULSE, "encode_words": True}
    for k, v in want.items():
        if fl.get(k) != v:
            p.append("flags.%s=%r (want %r)" % (k, fl.get(k), v))
    if rec.get("grid_row") != 3:
        p.append("grid_row %r" % rec.get("grid_row"))
    if (rec.get("record_label") or {}).get("latch_exclusion_w_gis") != 30.0:
        p.append("w_gis not 30")
    if rec.get("ment_pa") != 150.0:
        p.append("ment_pa %r" % rec.get("ment_pa"))
    if rec.get("seeds") != [seed] or len(rec.get("per_seed") or []) != 1 or rec["per_seed"][0].get("seed") != seed:
        p.append("seed mismatch (want %d)" % seed)
    if fl.get("som_w_ms") != float(ms) or fl.get("som_w_sa") != float(sa):
        p.append("cell mismatch: som_w_ms %r som_w_sa %r" % (fl.get("som_w_ms"), fl.get("som_w_sa")))
    echo, lesion, nofac, _tgi, _g, _t = arm_spec(arm, tau, g0)
    if fl.get("som_lesion") is not lesion:
        p.append("som_lesion %r for arm %s" % (fl.get("som_lesion"), arm))
    if fl.get("som_nofac") is not nofac:
        p.append("som_nofac %r for arm %s" % (fl.get("som_nofac"), arm))
    if bool(fl.get("stp_reset", False)):
        p.append("stp_reset on")
    if echo is None:
        if "f1j" in fl:
            p.append("identity record carries f1j flags %r" % (fl["f1j"],))
    elif fl.get("f1j") != echo:
        p.append("flags.f1j %r (want %r)" % (fl.get("f1j"), echo))
    return p


def _nm_trials(ps):
    return [(name, (tr or {}).get("nmda")) for name, tr in F1I._trials(ps)]


def _rel_ok(a, b, rel=REL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= rel * abs(float(b))


def guard_problems(rec, arm, tau=None, g0=None):
    """S9, S10, S11, S13, S14 on one f1j record (identity (iv)); the f1i S1/S2/S3/S7 read-backs are added by the caller
    (F1I.stop_id_problems). Every entry is STOP-ID; a missing field is UNDEFINED and counts as STOP-ID."""
    p = []
    echo, _l, _n, tau_gi, g_want, t_want = arm_spec(arm, tau, g0)
    if echo is None:
        return p
    ps = rec["per_seed"][0]
    cr = (ps.get("config_readback") or {}).get("f1j")
    if not cr:
        p.append("config_readback.f1j missing")
    for tname, nm in _nm_trials(ps):
        if not nm:
            p.append("%s: no nmda record" % tname)
            continue
        cfg = nm.get("config") or {}
        if cfg.get("syn_tau_g_i") != tau_gi or not _rel_ok(cfg.get("decay_i"), math.exp(-1.0 / tau_gi), 1e-9):
            p.append("%s: S13 syn_tau_g_i %r decay_i %r (want %r)" % (tname, cfg.get("syn_tau_g_i"),
                                                                       cfg.get("decay_i"), tau_gi))
        if (cfg.get("enable_apical_nmda") is not True or cfg.get("apical_nmda_g0") != g_want
                or cfg.get("apical_nmda_tau_ms") != t_want or cfg.get("apical_nmda_drive_ref") != APICAL_PULSE):
            p.append("%s: S14(a) realized config %r (want g0 %r tau %r drive_ref %r)" % (
                tname, {k: cfg.get(k) for k in ("enable_apical_nmda", "apical_nmda_g0", "apical_nmda_tau_ms",
                                                 "apical_nmda_drive_ref")}, g_want, t_want, APICAL_PULSE))
        if (nm.get("build_readback") or {}).get("problems"):
            p.append("%s: build read-back problems %r" % (tname, nm["build_readback"]["problems"]))
        if nm.get("reset_readback") != {"s_max": 0.0}:
            p.append("%s: S9 reset read-back %r" % (tname, nm.get("reset_readback")))
        if (nm.get("g_n_undriven_mention_max_run") != 0.0 or nm.get("g_n_undriven_mention_nonzero_ms") != []
                or nm.get("g_n_undriven_all_cells_max_run") not in (0.0,)):
            p.append("%s: S10 undriven g_N %r / %r / all cells %r" % (
                tname, nm.get("g_n_undriven_mention_max_run"), (nm.get("g_n_undriven_mention_nonzero_ms") or [])[:3],
                nm.get("g_n_undriven_all_cells_max_run")))
        pw = nm.get("per_word_ms")
        if pw is None:
            p.append("%s: per_word_ms missing" % tname)
            continue
        s_full = cfg.get("s_full")
        if g_want == 0.0:                                     # S11: a receptor-only lesion
            last = {"c1": {C1_WORD: C1_LAST}}.get(tname)
            if tname.startswith("c3"):
                ent = ps["c3"][int(tname[3])]
                x, y = ent["v_apical"]["pulsed_words"]
                last = {x: X_LAST, y: Y_LAST}
            for w, d in pw.items():
                if not d.get("g_n_max"):
                    p.append("%s: S11 g_N field missing on %s (UNDEFINED)" % (tname, w))
                elif any(v != 0.0 for v in d["g_n_max"]):
                    p.append("%s: S11 APV g_N != 0 on %s" % (tname, w))
                s = d.get("s_max")
                if s is None:
                    p.append("%s: S11 s field missing on %s (UNDEFINED)" % (tname, w))
                elif last is not None and w in last:
                    if not s[: last[w] + 1] or not _rel_ok(max(s[: last[w] + 1]), s_full):
                        p.append("%s: S11 s max %r on driven %s did not reach s_full %r by ms %d" % (
                            tname, max(s[: last[w] + 1]) if s[: last[w] + 1] else None, w, s_full, last[w]))
            if last is not None and set(pw) != set(last):
                p.append("%s: S11 pulsed words %r != %r" % (tname, sorted(pw), sorted(last)))
    c1 = (ps.get("c1") or {}).get("nmda") or {}
    g = ((c1.get("per_word_ms") or {}).get(C1_WORD) or {}).get("g_n_max")
    if g_want and g_want > 0.0:                               # S14 (b) and (c)
        if not g or len(g) <= C1_LAST + 1:
            p.append("c1: S14 g_N series missing (UNDEFINED)")
        else:
            if not _rel_ok(g[C1_LAST], g_want):
                p.append("c1: S14(b) g_N at ms %d = %r != g0 %r" % (C1_LAST, g[C1_LAST], g_want))
            d = math.exp(-1.0 / t_want)
            bad = [t for t in range(C1_LAST, len(g) - 1) if g[t] > G_TINY and not _rel_ok(g[t + 1] / g[t], d)]
            if bad:
                p.append("c1: S14(c) decay ratio off exp(-1/%g) at %d ms (first %d)" % (t_want, len(bad), bad[0]))
    c5 = (ps.get("c5") or {}).get("f1j_nmda") or {}
    for cl in ("clause_a", "clause_b"):
        r = c5.get(cl)
        if not r:
            p.append("c5 %s: no f1j_nmda record" % cl)
            continue
        if r.get("reset_readback") != {"s_max": 0.0}:
            p.append("c5 %s: S9 reset read-back %r" % (cl, r.get("reset_readback")))
        if r.get("g_n_undriven_all_cells_max_run") not in (0.0,):
            p.append("c5 %s: S10 undriven g_N %r" % (cl, r.get("g_n_undriven_all_cells_max_run")))
        if g_want == 0.0 and any(v.get("g_n") != 0.0 for v in (r.get("per_word_run_max") or {}).values()):
            p.append("c5 %s: S11 APV race g_N != 0" % cl)
    return p


# ── loading ────────────────────────────────────────────────────────────────────────────────────────────────────────

class Loader:
    """Loads and validates f1j artifacts once each; keeps invalid / STOP-ID problems by name."""

    def __init__(self, root, lines, pin):
        self.root, self.lines, self.pin = root, lines, pin
        self.cache = {}
        self.invalid, self.stop_id, self.consistency = {}, {}, {}

    def get(self, arm, seed, sa, ms, tau=None, g0=None):
        name = art_name(arm, seed, sa, ms, tau, g0)
        if name in self.cache:
            return self.cache[name]
        rec, probs = F1I.load_artifact(self.root, name, self.lines, rel_dir=STEP0_DIR, pin=self.pin)
        if rec is not None:
            probs += flag_problems(rec, arm, seed, sa, ms, tau, g0)
            if not probs:
                probs += F1I.schedule_problems(rec["per_seed"][0])
            if not probs:
                sid = F1I.stop_id_problems(rec, "lesion" if arm == "lesion" else "trace")
                sid += guard_problems(rec, arm, tau, g0)
                if sid:
                    self.stop_id[name] = sid
                bad = F1I.consistency(rec["per_seed"][0])
                if bad:
                    self.consistency[name] = bad
        if probs:
            self.invalid[name] = probs
            rec = None
        self.cache[name] = rec
        return rec


def _ps(rec):
    return rec["per_seed"][0] if rec is not None else None


def _c6(rec):
    """(C6 status, the c1 fall ms, the record's items) or (UNDEF, None, None) for an invalid record."""
    if rec is None:
        return UNDEF, None, None
    it = F1I.seed_items(_ps(rec), None)
    return it["C6"], it.get("c6_drop_ms"), it


def _gap_max(rec, lo, hi):
    g = _ps(rec)["c1"]["som"]["per_word_ms"][C1_WORD]["g_ap_max"]
    seg = [v for v in g[lo:hi + 1] if v is not None]
    return max(seg) if seg else None


def _som_tail(rec):
    return int(sum(_ps(rec)["c1"]["som"]["som_spikes_per_ms"][SOM_TAIL[0]:SOM_TAIL[1]]))


# ── C7 with the rewritten (c) ──────────────────────────────────────────────────────────────────────────────────────

def c7_pair_f1j(tc, lc, older, newer, v_hold=V_HOLD):
    """f1i's C7 pair with (c) rewritten before any run (RULING f1i-0-1028). The f1i peak reading is kept as c_f1i."""
    p = F1I.c7_pair(tc, lc, older, newer, v_hold)
    p["c_f1i_peak_reading"] = p["c"]
    spikes_cell = all(c >= 1 for c in p["som_spikes_per_cell"])
    term, onset = p["termination_ms"], p["g_ap_onset_ms"]
    if term is None:
        p["c"] = bool(spikes_cell and p["g_ap_peak"] is not None and p["g_ap_peak"] > 0.0)
    else:
        p["c"] = bool(spikes_cell and onset is not None and onset < term)
    if p["c"]:
        p["c_fail_reason"] = None
    elif not spikes_cell:
        p["c_fail_reason"] = "a SOM cell silent in [300, 600)"
    elif term is None:
        p["c_fail_reason"] = "older g_ap never > 0 in [300, 950)"
    else:
        p["c_fail_reason"] = "no g_ap increase after Y_on before the termination ms"
    if not p["0"]:
        p["status"] = UNDEF
    elif any(p[k] is False for k in SUBS):
        p["status"] = FAIL
    elif any(p[k] is None for k in SUBS):
        p["status"] = UNDEF
    else:
        p["status"] = PASS
    return p


def seed_items_f1j(ps, lesion_ps=None):
    it = F1I.seed_items(ps, lesion_ps)
    if "som" not in ps["c1"]:
        return it
    lc3 = (lesion_ps or {}).get("c3") if lesion_ps else None
    it["c7_pairs"] = [c7_pair_f1j(c, (lc3[i] if lc3 else None), x, y)
                      for i, ((x, y), c) in enumerate(zip(PAIRS, ps["c3"]))]
    sts = [p["status"] for p in it["c7_pairs"]]
    it["C7"] = FAIL if FAIL in sts else (UNDEF if UNDEF in sts else PASS)
    nm3 = [(c.get("nmda") or {}).get("older_g_n_at_newer_offset") for c in ps["c3"]]
    it["older_g_n_at_newer_offset"] = nm3
    return it


def summarize_trace_cell(rows, n_seeds=len(SEEDS)):
    """f1i's cell summary over the f1j seed rows (C7 with the rewritten (c)), plus latencies and the eraser-boundary
    conductance."""
    cs = F1I.summarize_cell(rows, n_seeds)
    defined = [p for r in rows.values() if r for p in r.get("c7_pairs") or [] if p["0"]]
    cs["c7_timing"] = [{"seed": s, "older": p["older"], "newer": p["newer"], "latency_ms": p["latency_ms"],
                        "first_rise_ms": p["g_ap_onset_ms"], "peak": p["g_ap_peak"], "peak_ms": p["g_ap_peak_ms"]}
                       for s, r in rows.items() if r for p in r["c7_pairs"] if p["0"]]
    cs["c7_sub"]["c_f1i_peak_reading"] = sum(1 for p in defined if p.get("c_f1i_peak_reading") is True)
    gn = [v for r in rows.values() if r for v in (r.get("older_g_n_at_newer_offset") or []) if v is not None]
    cs["older_g_n_at_newer_offset"] = ({"min": min(gn), "median": statistics.median(gn), "max": max(gn), "n": len(gn)}
                                       if gn else None)
    return cs


# ── per NMDA cell ──────────────────────────────────────────────────────────────────────────────────────────────────

def _tg_arm(sa, ms):
    return "cal" if (sa, ms) in ((0.125, 215), (0.125, 304)) else "grid"


def rescue_reads(L, records, tau, g0, *, arm_of=None, tau_gi_arm=None):
    """Rescue with attribution over RECORDS ((sa, ms, seed) list) for one NMDA setting against APV (same pin)."""
    rows = []
    for sa, ms, s in records:
        arm = arm_of(sa, ms) if arm_of else _tg_arm(sa, ms)
        tr = L.get(arm, s, sa, ms, tau, g0)
        ap = L.get("apv", s, sa, ms)
        t_st, _t_drop, _ = _c6(tr)
        a_st, a_drop, _ = _c6(ap)
        row = {"cell": LN.cell_tag(sa, ms), "seed": s, "trace_c6": t_st, "apv_c6": a_st, "apv_fall_ms": a_drop}
        if tr is None or ap is None:
            row["status"] = "UNDEFINED (invalid artifact)"
        elif a_st == FAIL and t_st == PASS:
            row["som_tail_trace"], row["som_tail_apv"] = _som_tail(tr), _som_tail(ap)
            if a_drop is None:
                row["status"] = "rescued, attribution UNDEFINED (APV fall ms absent)"
            else:
                gt, ga = _gap_max(tr, 150, a_drop), _gap_max(ap, 150, a_drop)
                row["g_ap_max_trace"], row["g_ap_max_apv"] = gt, ga
                ok = gt is not None and ga is not None and gt >= RESCUE_GAP_FLOOR and gt >= RESCUE_APV_FRAC * ga
                ok_alt = (gt is not None and ga is not None and gt >= RESCUE_GAP_FLOOR_ALT
                          and gt >= RESCUE_APV_FRAC * ga)
                row["status"] = "rescue" if ok else "rescued by IPSP removal"
                row["alt_floor_0p3189"] = "rescue" if ok_alt else "rescued by IPSP removal"
        elif t_st == PASS and a_st == PASS:
            row["status"] = "passes in APV too (not a rescue)"
        else:
            row["status"] = "not rescued"
        rows.append(row)
    n = sum(1 for r in rows if r["status"] == "rescue")
    n_alt = sum(1 for r in rows if r.get("alt_floor_0p3189") == "rescue")
    return {"count": n, "count_alt_floor_0p3189": n_alt, "rows": rows,
            "invalid": sum(1 for r in rows if r["status"] == "UNDEFINED (invalid artifact)"),
            "removal": [r for r in rows if r["status"] == "rescued by IPSP removal"],
            "undefined": [r for r in rows if r["status"].startswith(("rescued, attribution", "UNDEFINED"))]}


def nmda_cell_reads(L, ref, tau, g0):
    """rescue / late / kept / C6 / grid C7 / eraser for one NMDA cell (TRACE + GRID at the 10 ms IPSC)."""
    out = {"tauN": tau, "gN0": g0}
    out["rescue"] = rescue_reads(L, ref["c6"]["early"], tau, g0)
    out["late"] = rescue_reads(L, ref["c6"]["late"], tau, g0)
    kept = 0
    c6_pass = 0
    grid_def = grid_a = grid_b = 0
    invalid = 0
    for sa, ms in F1I_CELLS:
        for s in SEEDS:
            rec = L.get(_tg_arm(sa, ms), s, sa, ms, tau, g0)
            st, _d, _ = _c6(rec)
            c6_pass += st == PASS
            if (sa, ms, s) in ref["c6"]["pass"] and st == PASS:
                kept += 1
            if rec is None:
                invalid += 1
                continue
            for (x, y), c in zip(PAIRS, _ps(rec)["c3"]):
                p = F1I.c7_pair(c, None, x, y)
                if p["0"]:
                    grid_def += 1
                    grid_a += bool(p["a"])
                    grid_b += bool(p["b"])
    out.update(kept=kept, c6=c6_pass, grid_c7_defined=grid_def, grid_c7a=grid_a, grid_c7b=grid_b,
               invalid_trace_grid=invalid)
    # The A-NMDA's share of the C6 passes over the 72 records (reported, not a bar: the bars are rescue and eraser):
    # APV is the same pin with the receptor blocked and the gating running, so it holds everything but g_N fixed.
    apv_pass = sum(_c6(L.get("apv", s, sa, ms))[0] == PASS for sa, ms in F1I_CELLS for s in SEEDS)
    with contextlib.redirect_stdout(io.StringIO()):
        frac = attributable_to("A-NMDA %s: C6 passes /72, TRACE+GRID vs APV" % LN.nmda_tag(tau, g0), c6_pass,
                               apv_pass)
    out["c6_apv"] = apv_pass
    out["c6_fraction_attributable_to_nmda"] = frac
    er, er_undef, seed50 = 0, 0, None
    for s, x, y in ref["corner"]["defined"]:
        rec = L.get("cal", s, *CORNER, tau, g0)
        if rec is None:
            er_undef += 1
            continue
        c = _ps(rec)["c3"][PAIRS.index((x, y))]
        p = F1I.c7_pair(c, None, x, y)
        er += bool(p["0"] and p["a"])
    rec50 = L.get("cal", CORNER_UNDEF_PAIR[0], *CORNER, tau, g0)
    if rec50 is not None:
        c = _ps(rec50)["c3"][PAIRS.index(CORNER_UNDEF_PAIR[1:])]
        p = F1I.c7_pair(c, None, *CORNER_UNDEF_PAIR[1:])
        seed50 = {"defined": p["0"], "a": p["a"], "latency_ms": p["latency_ms"]}
    out["eraser"] = er
    out["eraser_undefined_pairs"] = er_undef                 # corner pairs whose record is invalid or missing
    out["eraser_seed50_frog_vole"] = seed50
    out["working"] = bool(out["rescue"]["count"] >= RESCUE_BAR and er == N_CORNER_DEF)
    out["predicted"] = PREDICTED.get((tau, g0))
    return out


def ipsc12_reads(L, tau, g0):
    """F3's reading at --tau-gi 12 for one NMDA cell: C6 at W_MS 215 and 304 (8 seeds each) and the W_MS 215 corner
    pairs' C7(a) (defined count, defined passes)."""
    out = {}
    invalid = 0
    for ms in PRIMARY_MS:
        recs = [L.get("ipsc12", s, 0.125, ms, tau, g0) for s in SEEDS]
        invalid += sum(r is None for r in recs)
        out["c6_ms%d" % ms] = [_c6(r)[0] for r in recs].count(PASS)
    out["invalid"] = invalid                                  # of the 16 IPSC-12 records
    d = a = 0
    for s in SEEDS:
        rec = L.get("ipsc12", s, 0.125, 215, tau, g0)
        if rec is None:
            continue
        for (x, y), c in zip(PAIRS, _ps(rec)["c3"]):
            p = F1I.c7_pair(c, None, x, y)
            if p["0"]:
                d += 1
                a += bool(p["a"])
    out["corner_defined"], out["corner_a"] = d, a
    out["pass"] = bool(out["c6_ms215"] == len(SEEDS) and out["c6_ms304"] == len(SEEDS) and d >= N_CORNER_DEF
                       and a == d)
    return out


def f1i_artifacts(f1i_root):
    """Every landed f1i step-0 artifact (not the sidecars): the set the packet's quoted f1i host summary covers
    ("host summary: 155 file(s): pool=138, workstation=17")."""
    return sorted(f for f in glob.glob(os.path.join(f1i_root, "*.json")) if not f.endswith(".prov.json"))


# ── identity ───────────────────────────────────────────────────────────────────────────────────────────────────────

def _host(path):
    try:
        from tools import where_ran as WR
        row = WR.where(path, WR.RunsIndex(WR.RUNS))
        return row.get("host"), row.get("class")
    except Exception:  # noqa: BLE001
        return None, "unknown"


def host_summary(paths):
    """tools/where_ran.py's host summary line for PATHS (quoted verbatim in the RESULT)."""
    try:
        from tools import where_ran as WR
        runs = WR.RunsIndex(WR.RUNS)
        rows = [WR.where(p, runs) for p in paths if os.path.exists(p)]
        return WR.render(rows, WR.summarize(rows)).split("\n")[len(rows) + 1]
    except Exception as e:  # noqa: BLE001
        return "host summary unavailable: %s" % (e,)


def identity_i(L, f1i_root):
    out = {}
    for s in IDENTITY_SEEDS:
        name = art_name("identity", s)
        rec = L.get("identity", s, *CORNER)
        ref_path = os.path.join(f1i_root, F1I.art_name("trace", s, *CORNER))
        try:
            ref = _load_json(ref_path)
        except Exception as e:  # noqa: BLE001
            out[s] = {"equal": None, "problems": ["f1i reference unreadable: %s" % e]}
            continue
        eq = rec is not None and F1I.canon_hash(rec) == F1I.canon_hash(ref)
        row = {"equal": eq if rec is not None else None, "problems": L.invalid.get(name, [])}
        if rec is not None and not eq:
            row.update(_host_rule(os.path.join(L.root, name), ref_path, rec, ref))
        out[s] = row
    return out


def _host_rule(path, ref_path, rec, ref):
    h1, c1 = _host(path)
    h2, c2 = _host(ref_path)
    same = (c1 == c2) and c1 not in ("unknown",)
    return {"host_pair": [h1, c1, h2, c2], "diffs": _first_diffs(rec, ref),
            "verdict": ("STOP-ID (same host class %s)" % c1) if same else
                       ("HOST-CLASS PENDING: re-run on the reference's host class (%s) before STOP-ID" % c2)}


def identity_iii(L, f1i_root, cells=None, seeds=None):
    out = {}
    for sa, ms in (cells or F1I_CELLS):
        for s in (seeds or SEEDS):
            name = art_name("apv", s, sa, ms)
            rec = L.get("apv", s, sa, ms)
            ref_path = os.path.join(f1i_root, F1I.art_name("trace", s, sa, ms))
            if rec is None:
                out[name] = {"equal": None, "problems": L.invalid.get(name, ["invalid"])}
                continue
            try:
                ref = _load_json(ref_path)
            except Exception as e:  # noqa: BLE001
                out[name] = {"equal": None, "problems": ["f1i reference unreadable: %s" % e]}
                continue
            eq = F1I.canon_hash(strip_f1j(rec)) == F1I.canon_hash(ref)
            row = {"equal": eq, "problems": []}
            if not eq:
                row.update(_host_rule(os.path.join(L.root, name), ref_path, strip_f1j(rec), ref))
            out[name] = row
    return out


def _identity_stop(ident_i, ident_iii, undefined_is_stop=False):
    """(STOP-ID, pending host-class re-runs) from the identity rows. A row with no comparison (invalid or missing
    artifact) is UNDEFINED: STOP-ID only when undefined_is_stop (the full read, as f1i), never a pass."""
    stop, pending = False, []
    for key, row in list(ident_i.items()) + list(ident_iii.items()):
        if row["equal"] is True and not row["problems"]:
            continue
        if row["equal"] is False and str(row.get("verdict", "")).startswith("HOST-CLASS PENDING"):
            pending.append(key)
            continue
        if row["equal"] is None and not undefined_is_stop:
            continue
        stop = True
    return stop, pending


# ── P0' ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _git_show(rev, path):
    try:
        r = subprocess.run(["git", "-C", _REPO, "show", "%s:%s" % (rev, path)], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def p0_engine(pin):
    """P0' read-only part: the E build and its tests at the pin, with file:line (the test run is recorded by hand)."""
    want = {"sim/config.py": "enable_apical_nmda: bool = False",
            "sim/bridge.py": "def apical_nmda_gating_step(",
            "research/runners/cs_race_f1.py": "def _reset_nmda(self):",
            "tests/test_apical_nmda.py": "def test_replay_match_single_slot(",
            "tests/test_cs_race_f1j_flags.py": "def test_runner_writes_the_gating_only_in_the_reset(",
            "research/biology/apical-nmda-conductance.md": "id: apical-nmda-conductance"}
    out = {}
    for path, needle in want.items():
        text = _git_show(pin, path) if pin else None
        if text is None:
            out[path] = None
            continue
        hit = [i + 1 for i, ln in enumerate(text.split("\n")) if needle in ln]
        out[path] = ("%s:%d" % (path, hit[0])) if hit else False
    return out


# ── the reads ──────────────────────────────────────────────────────────────────────────────────────────────────────

def f1i_reference(f1i_root):
    ref = {"c6": f1i_c6_records(f1i_root), "corner": f1i_corner_pairs(f1i_root)}
    counts = (len(ref["c6"]["early"]), len(ref["c6"]["late"]), len(ref["c6"]["pass"]), len(ref["corner"]["defined"]))
    ref["counts"] = dict(zip(("early", "late", "pass", "corner_defined"), counts))
    ref["ok"] = (counts == (N_EARLY, N_LATE, N_PASS, N_CORNER_DEF) and not ref["c6"]["problems"]
                 and not ref["corner"]["problems"] and CORNER_UNDEF_PAIR in ref["corner"]["undefined"])
    return ref


def read_lines(lines_path, wave3_path=None):
    lines = F1I.parse_lines(lines_path) if lines_path else None
    if lines is not None and wave3_path and os.path.exists(wave3_path):
        w3 = F1I.parse_lines(wave3_path)
        dup = set(w3) & set(lines)
        if dup:
            raise ValueError("wave-3 --out paths repeat wave 1-2 paths: %s" % sorted(dup)[:3])
        lines.update(w3)
    return lines


def score_identity(root, f1i_root, lines_path, pin, wave3_path=None):
    """Identity mode (after wave 1): identity (i) and (iii), S9-S13 (S11 with the gating field) on the APV and
    K1-NOFAC records. PASS queues wave 2 with no ruling; a fail is STOP-ID after the host-class rule."""
    L = Loader(root, read_lines(lines_path, wave3_path), pin)
    ref = f1i_reference(f1i_root)
    ii = identity_i(L, f1i_root)
    iii = identity_iii(L, f1i_root)
    for sa, ms in K1_CELLS:
        for s in SEEDS:
            L.get("k1nofac", s, sa, ms)
    stop, pending = _identity_stop(ii, iii)
    undefined = sorted(set(L.invalid) | {str(k) for k, v in list(ii.items()) + list(iii.items())
                                          if v["equal"] is None})
    stop = bool(stop or L.stop_id or not ref["ok"])
    paths = [os.path.join(root, n) for n in L.cache]
    res = {"mode": "identity", "packet": PACKET, "pin": pin, "f1i_reference_counts": ref["counts"],
           "f1i_reference_ok": ref["ok"], "identity_i": ii, "identity_iii": iii, "invalid": L.invalid,
           "stop_id_problems": L.stop_id, "consistency": L.consistency, "host_class_pending": pending,
           "host_summary_f1j": host_summary(paths),
           "host_summary_f1i_refs": host_summary(f1i_artifacts(f1i_root))}
    res["STOP_ID"] = stop
    res["undefined"] = undefined
    if stop:
        res["verdict"] = "STOP-ID: post and STOP"
    elif undefined:
        res["verdict"] = ("IDENTITY UNDEFINED: %d invalid or missing artifact(s); fix or re-run them, then re-score "
                          "(not a pass)" % len(undefined))
    elif pending:
        res["verdict"] = ("HOST-CLASS PENDING: re-run %d record(s) on the reference host class, then re-score"
                          % len(pending))
    else:
        res["verdict"] = "IDENTITY PASS: queue wave 2 (952 lines), no ruling needed"
    return res


def k1_reads(L):
    """H-K1's falsifier, read from the step-0 records. Each reading is True (falsified), False (not falsified) or None
    (UNDEFINED: an invalid or missing artifact, never read as "not falsified").
    (a) reads every APV and GRID record; one hit (a late C6 fail with 0 SOM spikes in [200, fall ms)) falsifies H-K1 on
        the records that load, so a hit is True whatever else is missing; with no hit, any invalid or missing record
        makes (a) UNDEFINED (that record could hold the hit). A valid record whose C6 is UNDEFINED (the plateau never
        rose) cannot be a late fail; it is counted and reported, never a hit.
    (b) compares late C6 fails on the same 16 K1 (cell, seed) records: it is read only when all 16 K1-NOFAC and all 16
        APV records are valid with a defined C6; otherwise UNDEFINED."""
    late_apv_grid, zero_som = [], []
    a_reads = {"records": 0, "invalid": 0, "c6_undefined": 0}
    for sa, ms in F1I_CELLS:
        for s in SEEDS:
            recs = [("apv", L.get("apv", s, sa, ms))]
            if (sa, ms) not in ((0.125, 215), (0.125, 304)):
                recs += [("grid_%s" % LN.nmda_tag(t, g), L.get("grid", s, sa, ms, t, g)) for t, g in NMDA_CELLS]
            for arm, rec in recs:
                a_reads["records"] += 1
                st, drop, _ = _c6(rec)
                if rec is None:
                    a_reads["invalid"] += 1
                    continue
                if st == UNDEF:
                    a_reads["c6_undefined"] += 1
                if st == FAIL and drop is not None and drop >= LATE_SOM_FROM:
                    n = int(sum(_ps(rec)["c1"]["som"]["som_spikes_per_ms"][LATE_SOM_FROM:drop]))
                    late_apv_grid.append((arm, LN.cell_tag(sa, ms), s, drop, n))
                    if n == 0:
                        zero_som.append((arm, LN.cell_tag(sa, ms), s, drop))
    nofac = {"late": 0, "early": 0, "n": 0}
    apv = {"late": 0, "early": 0, "n": 0}
    b_undef = {"k1nofac": {"invalid": 0, "c6_undefined": 0}, "apv": {"invalid": 0, "c6_undefined": 0}}
    for sa, ms in K1_CELLS:
        for s in SEEDS:
            for key, arm in ((nofac, "k1nofac"), (apv, "apv")):
                rec = L.get(arm, s, sa, ms)
                st, drop, _ = _c6(rec)
                if rec is None:
                    b_undef[arm]["invalid"] += 1
                    continue
                if st == UNDEF:
                    b_undef[arm]["c6_undefined"] += 1
                    continue
                key["n"] += 1
                if st == FAIL and drop is not None:
                    key["late" if drop >= LATE_SOM_FROM else "early"] += 1
    fals_a = True if zero_som else (None if a_reads["invalid"] else False)
    n_k1 = len(K1_CELLS) * len(SEEDS)
    fals_b = (nofac["late"] >= apv["late"]) if (nofac["n"] == apv["n"] == n_k1) else None
    h = True if (fals_a is True or fals_b is True) else (None if None in (fals_a, fals_b) else False)
    return {"late_fails_apv_grid": late_apv_grid, "late_fails_with_0_som_after_200": zero_som, "nofac": nofac,
            "apv": apv, "a_records": a_reads, "b_undefined": b_undef, "falsified_a": fals_a, "falsified_b": fals_b,
            "H_K1_falsified": h}


def select_cell(trace_cells, ncell):
    """(working NMDA cells, selectable TRACE cells as (W_MS, gN0, tauN) in the selection order, the selected one)."""
    working = [k for k in NMDA_CELLS if ncell[k]["working"]]
    cands = sorted((ms, g, t) for (t, g) in working if ncell[(t, g)]["kept"] == N_PASS
                   for ms in PRIMARY_MS if trace_cells[(t, g, ms)]["selectable"])
    return working, cands, (cands[0] if cands else None)


def falsifiers(ncell, ipsc, working):
    """F1-F4 as the packet reads them, from the per-NMDA-cell reads alone. They are read on complete data only: an
    invalid or missing record would read as a fail in a count (rescue, eraser, IPSC-12 C6), so score_full reports the
    read UNDEFINED before this runs, and a call on incomplete reads raises."""
    short = [LN.nmda_tag(*k) for k in NMDA_CELLS
             if ncell[k].get("invalid_trace_grid") or ncell[k].get("eraser_undefined_pairs")
             or ncell[k]["rescue"].get("invalid") or (ncell[k].get("late") or {}).get("invalid")
             or (ipsc.get(k) or {}).get("invalid")]
    if short:
        raise ValueError("F1-F4 read on incomplete data (invalid or missing records at %s): UNDEFINED, re-run first"
                         % short)
    er_ok = [k for k in NMDA_CELLS if ncell[k]["eraser"] == N_CORNER_DEF]
    return {"F1": all(ncell[k]["rescue"]["count"] < RESCUE_BAR for k in er_ok),
            "F2": bool(working) and all(k[0] == 40 for k in working),
            "F3": bool(working) and not any(ipsc[k]["pass"] for k in working),
            "F4": bool(working) and all(k[1] == 16 for k in working),
            "eraser_23_cells": [LN.nmda_tag(*k) for k in er_ok]}


def branches(*, stop_engine, stop_id, fz, selected, stop_attrib, stop_smoke):
    """The decision tree's branches that apply, in the packet's order (the first is the one named). Branch 1 reads
    nothing else, so it is returned alone (FZ may then be None: F1-F4 are not read on incomplete data)."""
    if stop_engine or stop_id:
        return ["1: STOP-ENGINE / STOP-ID: instrument; nothing else is read"]
    fals = [n for n in ("F1", "F2", "F3", "F4") if fz[n]]
    fired = []
    if fz["F1"]:
        fired.append("2: F1 fires: the A-NMDA METHOD verdict is banked as not closing the self-termination defect "
                     "once the loop closes; next: the research gate on the closed-loop residual (what amplifies the "
                     "self-recruited SOM tail), never a gN0 / tauN retune")
    if fz["F2"]:
        fired.append("3: F2 fires (only the 40 ms edge of the adult range works): recorded as a finding; the "
                     "protocol's 150 ms word and 150 ms gap are the next suspect, recorded and not retuned")
    if fz["F3"]:
        fired.append("4: F3 fires: the rescue is fragile to a 20% slower IPSC; recorded as a finding; the ruling "
                     "chooses")
    if fz["F4"]:
        fired.append("5: F4 fires: the rescue needs the top of the memo's set, 4x its smallest value (gN0 16); "
                     "recorded as a finding, never retuned; next suspect: the current-source drive")
    if not selected and not fals:
        fired.append("6: STOP-CAL with no falsifier: post which item fails at every working cell")
    if stop_attrib:
        fired.append("7: STOP-ATTRIB: post and STOP")
    if selected and stop_smoke:
        fired.append("8: a cell is selected but STOP-SMOKE fires (f1i's B4): the trace holds and overwrites but does "
                     "not reach the engram, an f1h coupling (soma-g) verdict")
    if selected and not (stop_engine or stop_id or fals or stop_attrib or stop_smoke):
        fired.append("9: a cell is selected, no STOP and no falsifier: post; step 1 runs only after a Claude RULING")
    if not fired:
        fired.append("STOP-CAL (no selection) with a falsifier already named above")
    return fired


def score_full(root, f1i_root, lines_path, pin, wave3_path=None, nomem_g0=None):
    L = Loader(root, read_lines(lines_path, wave3_path), pin)
    ref = f1i_reference(f1i_root)
    res = {"mode": "full", "packet": PACKET, "pin": pin, "f1i_reference_counts": ref["counts"],
           "f1i_reference_ok": ref["ok"]}
    ii = identity_i(L, f1i_root)
    iii = identity_iii(L, f1i_root)
    # TRACE cells: items
    trace_cells, trace_rows, all_c5a, all_c5b = {}, {}, [], []
    lesion_ii = []
    for t, g in NMDA_CELLS:
        for ms in PRIMARY_MS:
            rows = {}
            for s in SEEDS:
                tr = L.get("cal", s, 0.125, ms, t, g)
                lr = L.get("lesion", s, 0.125, ms, t, g)
                lesion_ii.append(lr is not None and art_name("lesion", s, 0.125, ms, t, g) not in L.stop_id)
                if tr is None:
                    rows[s] = None
                    continue
                rows[s] = seed_items_f1j(_ps(tr), _ps(lr))
                if rows[s]["C5a"] is not None:
                    all_c5a.append(rows[s]["C5a"])
                if rows[s]["C5b"] is not None:
                    all_c5b.append(rows[s]["C5b"])
            trace_cells[(t, g, ms)] = summarize_trace_cell(rows)
            trace_rows[(t, g, ms)] = rows
    # NMDA cells: rescue, late, kept, eraser, working; IPSC-12
    ncell = {(t, g): nmda_cell_reads(L, ref, t, g) for t, g in NMDA_CELLS}
    ipsc = {(t, g): ipsc12_reads(L, t, g) for t, g in NMDA_CELLS}
    ipsc_apv = {ms: sum(_c6(L.get("ipsc12_apv", s, 0.125, ms))[0] == PASS for s in SEEDS) for ms in PRIMARY_MS}
    working, cands, sel = select_cell(trace_cells, ncell)
    # selected-cell NMDA attribution (APV twin scored on the same items)
    sel_attr = None
    if sel is not None:
        ms, g, t = sel
        diffs_items, diffs_pairs = [], []
        apv_rows = {}
        for s in SEEDS:
            ap = L.get("apv", s, 0.125, ms)
            try:
                f1i_l = _ps(_load_json(os.path.join(f1i_root, F1I.art_name("lesion", s, 0.125, ms))))
            except Exception:  # noqa: BLE001
                f1i_l = None
            apv_rows[s] = seed_items_f1j(_ps(ap), f1i_l) if ap is not None else None
            tr_r = trace_rows[(t, g, ms)].get(s)
            if tr_r is None:
                continue
            for item in ITEMS_SEL:
                a_st = apv_rows[s][item] if apv_rows[s] else UNDEF
                if tr_r[item] == PASS and a_st != PASS:
                    diffs_items.append((s, item, a_st))
            for pa, pt in zip((apv_rows[s] or {}).get("c7_pairs") or [], tr_r.get("c7_pairs") or []):
                if pt["status"] == PASS and pa["status"] != PASS:
                    diffs_pairs.append((s, pt["older"], pt["newer"], pa["status"]))
        only50 = (len(diffs_pairs) == 1 and tuple(diffs_pairs[0][:3]) == CORNER_UNDEF_PAIR
                  and all(x[0] == CORNER_UNDEF_PAIR[0] and x[1] == "C7" for x in diffs_items))
        sel_attr = {"apv_items": {str(s): ({k: r[k] for k in ITEMS_SEL} if r else None) for s, r in apv_rows.items()},
                    "apv_fails_trace_passes_items": diffs_items, "apv_fails_trace_passes_pairs": diffs_pairs,
                    "label": ("NMDA-attributed on 1/24 pairs; C6 rescue evidence is GRID-only" if only50 else None),
                    "step1_apv_arm_also_at_ms304": bool(only50)}
    # STOP rules
    p0 = p0_engine(pin)
    stop_engine = any(v in (None, False) for v in p0.values())
    stop_iii, pending = _identity_stop(ii, iii, undefined_is_stop=True)
    stop_id = bool(stop_iii or L.stop_id or not ref["ok"])
    gm = {"a": round(statistics.median(all_c5a), 4) if all_c5a else None,
          "b": round(statistics.median(all_c5b), 4) if all_c5b else None}
    smoke_global = None if None in gm.values() else bool(gm["a"] < F1I.SMOKE_PA or gm["b"] < F1I.SMOKE_PA)
    smoke_sel = None
    if sel is not None:
        cm = trace_cells[(sel[2], sel[1], sel[0])]["c5_median"]
        smoke_sel = None if None in cm.values() else bool(cm["a"] < F1I.SMOKE_PA or cm["b"] < F1I.SMOKE_PA)
    c3_cells = [k for k in trace_cells if trace_cells[k]["C3"]["full"]]
    stop_attrib = bool(c3_cells) and all(trace_cells[k]["unattributed"] for k in c3_cells)
    k1 = k1_reads(L)
    # Every wave 1-2 record the decision reads is loaded by now. An invalid or missing one feeds its counts (rescue,
    # eraser, kept, C6, IPSC-12) as a shortfall, so the read is UNDEFINED until it is re-run: not a pass, not a
    # falsifier (as score_identity). STOP-ENGINE / STOP-ID (branch 1) still come first: nothing else is read there.
    invalid_read = sorted(L.invalid)
    undefined = bool(invalid_read) and not (stop_engine or stop_id)
    fz = falsifiers(ncell, ipsc, working) if not invalid_read else None
    fals = [n for n in ("F1", "F2", "F3", "F4") if fz[n]] if fz else []
    if undefined:
        msg = ("UNDEFINED: re-run the %d invalid or missing artifact(s), then re-score; not a pass, not a falsifier"
               % len(invalid_read))
        d = {"UNDEFINED": True, "invalid_or_missing": len(invalid_read), "working": None, "selected": None,
             "selectable": None, "STOP_ENGINE": False, "STOP_ID": False, "host_class_pending": pending,
             "STOP_CAL": None, "STOP_SMOKE_selected": None, "STOP_SMOKE_global": None, "STOP_SMOKE": None,
             "STOP_ATTRIB": None, "falsifiers": [], "falsifier_reads": None, "STOP_FALS": None,
             "K1": {"H_K1_falsified": k1["H_K1_falsified"]}, "selected_cell_attribution": None,
             "stop_cal_reading": None, "branch": msg, "branches_fired": [msg], "stops": [], "verdict": msg}
    # STOP-CAL readings (branch 6)
    cal_reading = None
    if sel is None and not fals and not undefined:
        fail_items = {}
        for (t, g) in working:
            for ms in PRIMARY_MS:
                c = trace_cells[(t, g, ms)]
                fail_items["%s_ms%d" % (LN.nmda_tag(t, g), ms)] = [i for i in ITEMS_SEL if not c[i]["full"]]
        common = (set.intersection(*[set(v) for v in fail_items.values()]) if fail_items else set())
        b1 = bool(working) and all(not trace_cells[(t, g, ms)]["c7_sub"]["b_full"] for (t, g) in working
                                   for ms in PRIMARY_MS)
        # K1's pre-decided reading: at every working TRACE cell a C8 fail or a late C6 fail, each on a record with SOM
        # spikes after 200 ms in c1.
        k1_cells = {}
        for (t, g) in working:
            for ms in PRIMARY_MS:
                hits = []
                for s in SEEDS:
                    rec = L.get("cal", s, 0.125, ms, t, g)
                    r = trace_rows[(t, g, ms)].get(s)
                    if rec is None or r is None:
                        continue
                    som_after = int(sum(_ps(rec)["c1"]["som"]["som_spikes_per_ms"][LATE_SOM_FROM:]))
                    late_c6 = r["C6"] == FAIL and (r.get("c6_drop_ms") or 0) >= LATE_SOM_FROM
                    if (r["C8"] == FAIL or late_c6) and som_after > 0:
                        hits.append((s, "C8" if r["C8"] == FAIL else "late C6", som_after))
                k1_cells["%s_ms%d" % (LN.nmda_tag(t, g), ms)] = hits
        k1_every = bool(k1_cells) and all(k1_cells.values())
        cal_reading = {"failing_items_per_working_trace_cell": fail_items, "failing_at_every_working_cell":
                       sorted(common), "C7b_fails_at_every_working_cell (f1i B1 text)": b1,
                       "K1_reading_C8_or_late_C6_with_SOM_after_200_per_cell": k1_cells,
                       "K1_at_every_working_cell (K1 goes first)": k1_every}
    if not undefined:
        d = {"UNDEFINED": False, "invalid_or_missing": len(invalid_read), "working": [LN.nmda_tag(*k) for k in working],
             "selected": (None if sel is None else
                          "%s_%s" % (LN.nmda_tag(sel[2], sel[1]), LN.cell_tag(0.125, sel[0]))),
             "selectable": ["%s_ms%d" % (LN.nmda_tag(t, g), ms) for (ms, g, t) in cands],
             "STOP_ENGINE": stop_engine, "STOP_ID": stop_id, "host_class_pending": pending, "STOP_CAL": sel is None,
             "STOP_SMOKE_selected": smoke_sel, "STOP_SMOKE_global": smoke_global,
             "STOP_SMOKE": bool(sel is not None and (smoke_sel or smoke_global)), "STOP_ATTRIB": stop_attrib,
             "falsifiers": fals, "falsifier_reads": fz, "STOP_FALS": bool(fals),
             "K1": {"H_K1_falsified": k1["H_K1_falsified"]},
             "selected_cell_attribution": sel_attr, "stop_cal_reading": cal_reading}
        fired = branches(stop_engine=stop_engine, stop_id=stop_id, fz=fz, selected=sel is not None,
                         stop_attrib=stop_attrib, stop_smoke=d["STOP_SMOKE"])
        d["branch"] = fired[0]
        d["branches_fired"] = fired
        stops = [n for n in ("STOP_ENGINE", "STOP_ID", "STOP_CAL", "STOP_SMOKE", "STOP_ATTRIB", "STOP_FALS") if d[n]]
        d["stops"] = stops
        d["verdict"] = ("STOP: " + ", ".join(stops + fals)) if stops else ("CELL SELECTED: %s" % d["selected"])
    # NOMEM (wave 3)
    nomem = None
    if nomem_g0:
        nomem = {}
        for g in sorted(set(int(x) for x in nomem_g0)):
            rr = rescue_reads(L, ref["c6"]["early"], None, g, arm_of=lambda sa, ms: "nomem")
            # wave 3 is informational and read after the decision; an invalid or missing NOMEM record makes this
            # rescue count UNDEFINED (None), its count over the valid records reported beside it
            nomem["g%d" % g] = {"rescue": None if rr["invalid"] else rr["count"], "invalid": rr["invalid"],
                                "rescue_on_valid_records": rr["count"], "removal": len(rr["removal"]),
                                "rows": rr["rows"],
                                "beside_working_cells": None if undefined else {
                                    LN.nmda_tag(*k): ncell[k]["rescue"]["count"] for k in working if k[1] == g}}
    res.update({"identity_i": ii, "identity_iii": iii,
                "identity_ii": {"lesion_records": len(lesion_ii), "clean": sum(lesion_ii)},
                "invalid": L.invalid, "stop_id_problems": L.stop_id, "consistency": L.consistency,
                "p0_engine_at_pin": p0, "global_c5_median": gm,
                "trace_cells": {"%s_ms%d" % (LN.nmda_tag(t, g), ms): v for (t, g, ms), v in trace_cells.items()},
                "seed_rows": {"%s_ms%d" % (LN.nmda_tag(t, g), ms): {str(s): r for s, r in v.items()}
                              for (t, g, ms), v in trace_rows.items()},
                "nmda_cells": {LN.nmda_tag(*k): v for k, v in ncell.items()},
                "ipsc12": {LN.nmda_tag(*k): v for k, v in ipsc.items()}, "ipsc12_apv_c6": ipsc_apv,
                "k1": k1, "nomem": nomem, "decision": d,
                "flip_scale_pa": list(F1I.FLIP_PA),
                "host_summary_f1j": host_summary([os.path.join(root, n) for n in L.cache]),
                "host_summary_f1i_refs": host_summary(f1i_artifacts(f1i_root))})
    return res


# ── rendering ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _pfu(c):
    return "%d/%d/%d" % (c[PASS], c[FAIL], c[UNDEF])


def render(res):
    L = []
    if res["mode"] == "identity":
        L.append("cs-race f1j step-0 IDENTITY mode, pin %s" % str(res["pin"])[:9])
        L.append("f1i reference counts (re-derived): %s ok=%s" % (json.dumps(res["f1i_reference_counts"]),
                                                                  res["f1i_reference_ok"]))
        L.append("Identity (i): %s" % json.dumps({str(k): v for k, v in res["identity_i"].items()}))
        bad = {k: v for k, v in res["identity_iii"].items() if v["equal"] is not True}
        L.append("Identity (iii): %d/%d APV records equal to f1i; not equal: %s" % (
            len(res["identity_iii"]) - len(bad), len(res["identity_iii"]), json.dumps(bad)))
        L.append("Invalid artifacts: %d; STOP-ID read-back problems (S1-S3, S7, S9-S13): %d; consistency: %d" % (
            len(res["invalid"]), len(res["stop_id_problems"]), len(res["consistency"])))
        for k, v in list(res["stop_id_problems"].items())[:10]:
            L.append("  %s: %s" % (k, v[:3]))
        L.append("f1j records: %s" % res["host_summary_f1j"])
        L.append("f1i references: %s" % res["host_summary_f1i_refs"])
        L.append("VERDICT: %s" % res["verdict"])
        return "\n".join(L)
    d = res["decision"]
    L.append("cs-race f1j step-0 FULL read, pin %s; seeds %s" % (str(res["pin"])[:9], ",".join(str(s) for s in SEEDS)))
    L.append("")
    L.append("| TRACE cell | C0 | C1 | C2 | C3 | C4 | C6 | C7 | C8 | med C5a | med C5b | selectable |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, c in res["trace_cells"].items():
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            tag, _pfu(c["C0"]), _pfu(c["C1"]), _pfu(c["C2"]), _pfu(c["C3"]), _pfu(c["C4"]), _pfu(c["C6"]),
            _pfu(c["C7"]), _pfu(c["C8"]), c["c5_median"]["a"], c["c5_median"]["b"],
            "yes" if c["selectable"] else "no"))
    L.append("")
    L.append("| NMDA cell | rescue /45 (alt floor) | removal | late /6 | kept /21 | C6 /72 | grid C7(a) / def | "
             "grid C7(b) /216 | eraser /23 | seed-50 pair | working | predicted (rescue, late, kept, C6, corner, "
             "grid a/def, grid b) | IPSC-12 C6 215/304, corner a/def |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, c in res["nmda_cells"].items():
        ip = res["ipsc12"][tag]
        L.append("| %s | %d (%d) | %d | %d | %d | %d | %d/%d | %d | %d | %s | %s | %s | %d/%d, %d/%d |" % (
            tag, c["rescue"]["count"], c["rescue"]["count_alt_floor_0p3189"], len(c["rescue"]["removal"]),
            c["late"]["count"], c["kept"], c["c6"], c["grid_c7a"], c["grid_c7_defined"], c["grid_c7b"], c["eraser"],
            json.dumps(c["eraser_seed50_frog_vole"]), "yes" if c["working"] else "no", json.dumps(c["predicted"]),
            ip["c6_ms215"], ip["c6_ms304"], ip["corner_a"], ip["corner_defined"]))
    L.append("A-NMDA share of the C6 passes (tools.lab.attributable_to, TRACE+GRID vs APV /72, reported, not a bar): %s"
             % json.dumps({t: [c["c6"], c["c6_apv"], c["c6_fraction_attributable_to_nmda"]]
                           for t, c in res["nmda_cells"].items()}))
    L.append("Replay prediction without NMDA (IPSC 10): %s; IPSC-12 APV C6: %s" % (
        json.dumps(PREDICTED[None]), json.dumps(res["ipsc12_apv_c6"])))
    L.append("")
    for tag, c in res["nmda_cells"].items():
        rem = c["rescue"]["removal"] + c["rescue"]["undefined"]
        if rem:
            L.append("- %s rescued apart (IPSP removal / attribution UNDEFINED): %s" % (tag, json.dumps(rem)))
    L.append("")
    for tag, c in res["trace_cells"].items():
        L.append("- %s: C7 latencies %s; older g_N at Y_on+150 %s; C7(c) f1i peak reading %d vs rewritten %d" % (
            tag, F1I._lat_summary(c["c7_sub"]["latencies_ms"]), json.dumps(c["older_g_n_at_newer_offset"]),
            c["c7_sub"]["c_f1i_peak_reading"], c["c7_sub"]["c"]))
    L.append("")
    L.append("Identity (i): %s" % json.dumps({str(k): v for k, v in res["identity_i"].items()}))
    bad = {k: v for k, v in res["identity_iii"].items() if v["equal"] is not True}
    L.append("Identity (iii): %d/%d equal; not equal: %s" % (len(res["identity_iii"]) - len(bad),
                                                            len(res["identity_iii"]), json.dumps(bad)))
    L.append("Identity (ii): %d/%d LESION records clean" % (res["identity_ii"]["clean"],
                                                            res["identity_ii"]["lesion_records"]))
    L.append("Invalid artifacts: %d; STOP-ID read-back problems: %d; consistency mismatches: %d" % (
        len(res["invalid"]), len(res["stop_id_problems"]), len(res["consistency"])))
    L.append("P0' at pin: %s" % json.dumps(res["p0_engine_at_pin"]))
    L.append("Global C5 median over the 144 TRACE records: %s (flip scale %s pA)" % (
        json.dumps(res["global_c5_median"]), res["flip_scale_pa"]))
    L.append("K1: %s" % json.dumps({k: v for k, v in res["k1"].items() if k != "late_fails_apv_grid"}))
    if res.get("nomem"):
        L.append("NOMEM (wave 3): %s" % json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                                    for k, v in res["nomem"].items()}))
    L.append("f1j records: %s" % res["host_summary_f1j"])
    L.append("f1i references: %s" % res["host_summary_f1i_refs"])
    if res["invalid"]:
        L.append("Invalid or missing records per NMDA cell (TRACE+GRID /72, rescue rows, eraser pairs, IPSC-12 /16): %s"
                 % json.dumps({t: [c.get("invalid_trace_grid"), c["rescue"].get("invalid"),
                                   c.get("eraser_undefined_pairs"), res["ipsc12"][t].get("invalid")]
                               for t, c in res["nmda_cells"].items()}))
        L.append("Invalid or missing artifacts (first 10): %s" % json.dumps(sorted(res["invalid"])[:10]))
    L.append("")
    L.append("VERDICT: %s" % d["verdict"])
    L.append("DECISION: %s" % json.dumps(d))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("identity", "full"), required=True)
    ap.add_argument("--pin", required=True, help="the revision every f1j step-0 line ran at (the lines' <sha>)")
    ap.add_argument("--root", default=os.path.join(_REPO, STEP0_DIR))
    ap.add_argument("--f1i-root", default=os.path.join(_REPO, F1I_DIR))
    ap.add_argument("--lines", default=os.path.join(_REPO, LINES))
    ap.add_argument("--wave3-lines", default=os.path.join(_REPO, LINES_WAVE3))
    ap.add_argument("--nomem-g0", default=None, help="full mode: the working gN0 set wave 3 ran (comma-separated)")
    ap.add_argument("--out", default=None, help="JSON result (under research/findings/raw for a provenance sidecar)")
    a = ap.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):
        if a.mode == "identity":
            res = score_identity(a.root, a.f1i_root, a.lines, a.pin, a.wave3_lines)
        else:
            res = score_full(a.root, a.f1i_root, a.lines, a.pin, a.wave3_lines,
                             [int(x) for x in a.nomem_g0.split(",")] if a.nomem_g0 else None)
    print(render(res))
    if a.out:
        out = a.out if os.path.isabs(a.out) else os.path.join(_REPO, a.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=1, sort_keys=True, default=str)
        try:
            from research.runners import declare_output
            declare_output(out)
        except Exception:  # noqa: BLE001 - provenance registration is never fatal
            pass
        print("-> %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
