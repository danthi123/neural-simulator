"""Scorer for cs-race f1i STEP 0: the whole step-0 read in one module (C0-C8, identity, STOP rules, selection, branch).

Packet: research/coordination/cs-race-f1i-PACKET.md (Step 0 items C0-C8, Selection, STOP rules, Branches; ruling R1).
C0-C6 are f1h's items (research/coordination/cs-race-f1h-PACKET.md), read as the runner computes them
(research/runners/cs_race_f1.py run_calibration) and re-derived from the record's own counts where it keeps them. C7 and
C8 are new in f1i: the runner keeps their raw series (rec["som"]: per_word_ms v_apical min / max and g_ap max per ms for
the pulsed words, som_spikes_per_ms, som_cell_events) and this module is the scorer the packet names ("the scorer
re-derives every item"). It adds no bar and loosens none. It reads data only and runs no brain.

    python -m research.runners.cs_race_f1i_score [--root DIR] [--out research/findings/raw/_cs_race/f1i/score/x.json]

ARTIFACT VALIDITY (an artifact that fails any rule makes every item it feeds UNDEFINED, never a pass or a fail): the JSON
and its .prov.json exist and parse; the sidecar is a clean run of the pin (git_sha == PIN, git_dirty False, source_kind
git_archive, manifest verified at start and at exit); the sidecar argv equals, token for token, the pool line that names
this output (research/coordination/cs-race-f1i-0-pool-lines.txt); the record's flags name this cell, arm and seed; the
schedule read from the record is the packet's (c1 window [150, 800), c3 X [0, 150), Y [300, 450), window [600, 950)).

STOP-ID checks on every --som record (packet S2, S7 and identity (ii)): build_readback.problems empty; realized STP msom
(U 0.049, tau_f 1797, tau_d 399, conn_type 1) and ee (the grid row's U and tau_f, conn_type 0); reset_readback msom
u 0 and x 1, g_ap 0, the gate at its arm's value; under --som-lesion, g_ap_max_run == 0 and
gate_cs_som_at_measurement == 0.0 on every trial. Identity (i): the no-flag record minus top-level wall_s hashes equal to
the landed f1h record at the base cell (seeds 41 and 45, R1).

ITEM READINGS (the bar text is the packet's; this is how each is read from the record):
  C0  every pulse event of the c1 trial and of the three c3 trials has min_cell_spikes >= 1 (runner c0, RULING
      cs-race-f1g-0-1022 C).
  C1  c1 pulsed rate in [2, 20] Hz per cell over [150, 800), re-derived from ments_spikes.
  C2  c1 unpulsed max rate < 1 Hz AND c2b (2 s, no input) max rate < 1 Hz.
  C3  rate form on all three pairs (vole>frog, frog>vole, lynx>grass): newer 2-20 Hz AND older < 1 Hz over [600, 950).
  C4  c1 burst_ms is None AND latch_ms is None (the runner's first-burst rule; the engram raster is not kept).
  C5  clause_a / clause_b delta pA per seed; the cell median is read against STOP-SMOKE (2 pA) and the +4/+6 pA flip
      scale. Not a selection item.
  C6  c1: UNDEFINED if the pulsed slot's run max never exceeded v_hold; else PASS iff its window min > v_hold AND every
      unpulsed slot's window max < v_hold. The pulsed slot's window min / max are re-derived from per_word_ms.
  C7  per C3 pair (X older, Y newer), from the TRACE c3 trial and the LESION c3 trial of the same cell, seed and pair:
      (0) min over [150, 300) of the older slot's per-ms min v_apical > v_hold; otherwise the pair is UNDEFINED.
      (a) termination ms = the first ms t >= Y_on (300) such that the older slot's per-ms max v_apical < v_hold at every
          ms of [t, 950); latency = t - 300; PASS iff it exists and latency <= 300.
      (b) min over [450, 950) of the newer slot's per-ms min v_apical > v_hold.
      (c) every SOM cell fires >= 1 spike in [300, 600) (som_cell_events), AND the older slot's g_ap peak over [300, 950)
          is > 0 AND, when a termination ms exists, the peak's first ms is before it. With no termination ms (the
          plateau held) there is nothing to precede, so (c) is the spikes and g_ap > 0: the packet's B3 text ("SOM fires,
          g_ap > 0, the plateau holds") reads (c) on exactly that case.
      (d) LESION trial: the older slot's min v_apical over [600, 950) > v_hold AND the rate form of C3 FAILS on the pair.
      Pair: UNDEFINED if (0) fails, else FAIL if any of (a)-(d) fails, UNDEFINED if any is unreadable, else PASS. Seed:
      PASS iff all three pairs PASS; FAIL if any pair FAILs; else UNDEFINED.
  C8  c1 (one word, then 650 ms): SOM spikes over [200, 800) / (n_som x 0.6 s) < 1 Hz (Hz per cell, the unit of every
      rate in the runner and the packets).
  Second readings, each reported beside the primary, with a flag wherever a verdict would differ: C7(c) spikes pooled
  (sum >= n_som) against per cell (primary, the stricter); the g_ap peak over the whole trial against [300, 950)
  (primary); C8 as the pool's summed rate against the per-cell rate (primary).

CELL: an item is 8/8 when it is PASS on all eight seeds (41, 45-51; R1). Selection: smallest W_SA, then smallest W_MS,
among cells with C0-C4 and C6-C8 at 8/8. A C7 sub-item passes at a cell when it is a defined pass on all 24 pairs.
UNATTRIBUTED: C3 8/8 while C7(c) or C7(d) is not 24/24. STOP-ATTRIB: every C3-passing cell is UNATTRIBUTED (with no
C3-passing cell it does not fire; that is reported). STOP-SMOKE: a cell median of C5 below 2 pA in either clause, read at
the selected cell; the median over all 72 TRACE records is reported beside it. STOP-CAL: no cell selected. Branches
B1-B4 by the packet's text on the cell-level sub-items. R1 Q5: every item at seed 41, STP-reset arm against TRACE.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shlex
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import attributable_to  # noqa: E402  (the lesion's share of the overwrite: gates/attribution_required)

PACKET = "research/coordination/cs-race-f1i-PACKET.md"
STEP0_DIR = "research/findings/raw/_cs_race/f1i/step0"
F1H_DIR = "research/findings/raw/_cs_race/f1h/step0"
LINES = "research/coordination/cs-race-f1i-0-pool-lines.txt"
PIN = "ee349f8b11e19a69456b2addc06218c4c0f0bbbf"
RUNNER_MODULE = "research.runners.cs_race_f1"

SEEDS = (41, 45, 46, 47, 48, 49, 50, 51)       # R1: seeds 42-44 are gate seeds
IDENTITY_SEEDS = (41, 45)                        # pool lines: 45 stands in for the packet's 42
STPRESET_SEED = 41
W_SA = (0.125, 0.25, 0.5)
W_MS = (215, 304, 429)
PAIRS = (("vole", "frog"), ("frog", "vole"), ("lynx", "grass"))
MENTION_WORDS = ("lynx", "chase", "vole", "frog", "yak", "eat", "grass", "moose", "graze")  # cs_race_f1.py:72
C1_WORD = "vole"

# frozen thresholds (f1g / f1h / f1i packets; the runner's thresholds_frozen_in_packet)
C1_LO, C1_HI = 2.0, 20.0
QUIET_HZ = 1.0
SMOKE_PA = 2.0
FLIP_PA = (4.0, 6.0)
V_HOLD = -35.0

# schedules (packet C7 / C8 text; checked against each record)
C1_PULSE = (0, 150)
C1_WIN = (150, 800)
C8_WIN = (C1_PULSE[1] + 50, C1_PULSE[1] + 650)  # [pulse end + 50, pulse end + 650)
X_PULSE = (0, 150)
Y_ON = 300
Y_PULSE = (Y_ON, Y_ON + 150)
C3_WIN = (Y_ON + 300, Y_ON + 650)                # [600, 950)
PRE_LATCH = (X_PULSE[1], Y_ON)                   # C7(0): [150, 300)
LAT_BAR = 300                                    # C7(a)
NEWER_HOLD = (Y_ON + 150, Y_ON + 650)            # C7(b): [450, 950)
SOM_WIN = (Y_ON, Y_ON + 300)                     # C7(c): [300, 600)

STP_MSOM = {"U": 0.049, "tau_f": 1797.0, "tau_d": 399.0}
ITEMS_SEL = ("C0", "C1", "C2", "C3", "C4", "C6", "C7", "C8")
SUBS = ("a", "b", "c", "d")

PASS, FAIL, UNDEF = "PASS", "FAIL", "UNDEFINED"


def _st(ok):
    return UNDEF if ok is None else (PASS if ok else FAIL)


def cell_tag(sa, ms):
    return "sa%s_ms%d" % (("%g" % sa).replace(".", "p"), int(ms))


def art_name(arm, seed, sa=None, ms=None):
    if arm == "identity":
        return "identity_f1h_sg40_wim40_s%d.json" % seed
    return "%s_%s_s%d.json" % ({"trace": "cal", "lesion": "lesion", "stpreset": "stpreset"}[arm],
                               cell_tag(sa, ms), seed)


def canon_hash(rec):
    """sha256 of the record minus top-level wall_s, keys sorted (packet identity (i))."""
    r = {k: v for k, v in rec.items() if k != "wall_s"}
    return hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ── pool lines ─────────────────────────────────────────────────────────────────────────────────────────────────────

def parse_lines(path):
    """{out path: argv tokens after `-m research.runners.cs_race_f1`} from the pool lines (cmd<TAB>reason)."""
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            cmd = ln.split("\t", 1)[0]
            toks = shlex.split(cmd)
            i = toks.index("-m")
            if toks[i + 1] != RUNNER_MODULE:
                raise ValueError("pool line runs %r, not %s" % (toks[i + 1], RUNNER_MODULE))
            args = toks[i + 2:]
            dst = args[args.index("--out") + 1]
            if dst in out:
                raise ValueError("duplicate --out in pool lines: %s" % dst)
            out[dst] = args
    return out


# ── loading and validity ───────────────────────────────────────────────────────────────────────────────────────────

def load_artifact(root, name, lines, rel_dir=STEP0_DIR, pin=PIN):
    """(record or None, problems). Sidecar must be a clean run of PIN whose argv is the pool line for this output."""
    path = os.path.join(root, name)
    probs = []
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception as e:  # noqa: BLE001 - a missing or broken file is UNDEFINED, reported
        return None, ["artifact unreadable: %s" % (e,)]
    try:
        with open(path + ".prov.json") as f:
            prov = json.load(f)
    except Exception as e:  # noqa: BLE001
        return rec, ["sidecar unreadable: %s" % (e,)]
    if prov.get("git_sha") != pin:
        probs.append("sidecar git_sha %s != pin %s" % (prov.get("git_sha"), pin))
    if prov.get("git_dirty") is not False:
        probs.append("sidecar git_dirty %r" % (prov.get("git_dirty"),))
    if prov.get("source_kind") != "git_archive":
        probs.append("sidecar source_kind %r" % (prov.get("source_kind"),))
    if not (prov.get("source_manifest_verified_at_start") and prov.get("source_manifest_verified_at_exit")):
        probs.append("source manifest not verified at start and exit")
    rel = rel_dir.rstrip("/") + "/" + name
    if lines is not None:
        want = lines.get(rel)
        argv = list(prov.get("argv") or [])
        if want is None:
            probs.append("no pool line names %s" % rel)
        elif argv[1:] != want:
            probs.append("sidecar argv differs from the pool line")
        if not str(argv[0] if argv else "").endswith("research/runners/cs_race_f1.py"):
            probs.append("sidecar runner is not cs_race_f1.py")
    return rec, probs


def flag_problems(rec, arm, seed, sa=None, ms=None):
    """The record's own flags name this arm, cell and seed (base cell f1h-0-1027; the f1h step-0 flags)."""
    p = []
    fl = rec.get("flags") or {}
    want = {"W_IM": 40.0, "W_MI": 20.0, "W_MM": 0.0, "NIM": 8, "trace": "intrinsic", "soma_g": 40.0,
            "apical_pa": 120.0, "encode_words": True}
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
    if arm == "identity":
        if "som_w_ms" in fl:
            p.append("identity record carries f1i flags")
        return p
    if fl.get("som_w_ms") != float(ms) or fl.get("som_w_sa") != float(sa):
        p.append("cell mismatch: som_w_ms %r som_w_sa %r" % (fl.get("som_w_ms"), fl.get("som_w_sa")))
    if fl.get("som_lesion") is not (arm == "lesion"):
        p.append("som_lesion %r for arm %s" % (fl.get("som_lesion"), arm))
    if fl.get("som_nofac") is not False:
        p.append("som_nofac on")
    if bool(fl.get("stp_reset", False)) is not (arm == "stpreset"):
        p.append("stp_reset %r for arm %s" % (fl.get("stp_reset"), arm))
    return p


def _trials(ps):
    return [("c1", ps.get("c1")), ("c2b", ps.get("c2b"))] + [
        ("c3[%d]" % i, c) for i, c in enumerate(ps.get("c3") or [])]


def stop_id_problems(rec, arm):
    """S2 / S7 / identity (ii) read-backs on every trial of a --som record. Any entry is STOP-ID."""
    p = []
    ps = rec["per_seed"][0]
    grid = rec.get("grid") or {}
    lesion = arm == "lesion"
    gate_want = 0.0 if lesion else 1.0
    for tname, tr in _trials(ps):
        som = (tr or {}).get("som")
        if not som:
            p.append("%s: no som record" % tname)
            continue
        br = som.get("build_readback") or {}
        if br.get("problems"):
            p.append("%s: build_readback problems %r" % (tname, br["problems"]))
        if br.get("apical_targets_are_mention_cells") is not True or not br.get("g_ap_len"):
            p.append("%s: g_ap not allocated on the mention cells" % tname)
        rs = br.get("realized_stp") or {}
        m, e = rs.get("msom") or {}, rs.get("ee") or {}
        if (m.get("conn_type") != [1] or m.get("U") != [STP_MSOM["U"]] or m.get("tau_f") != [STP_MSOM["tau_f"]]
                or m.get("tau_d") != [STP_MSOM["tau_d"]]):
            p.append("%s: realized msom STP %r" % (tname, m))
        if e.get("conn_type") != [0] or e.get("U") != [grid.get("U")] or e.get("tau_f") != [grid.get("tau_f")]:
            p.append("%s: realized ee STP %r" % (tname, e))
        rb = som.get("reset_readback") or {}
        if (rb.get("msom_u_max") != 0.0 or rb.get("msom_x_min") != 1.0 or rb.get("g_ap_max") != 0.0
                or rb.get("gate_cs_som") != gate_want):
            p.append("%s: reset_readback %r" % (tname, rb))
        if som.get("gate_cs_som_at_measurement") != gate_want:
            p.append("%s: gate_cs_som at measurement %r (want %r)" % (tname, som.get("gate_cs_som_at_measurement"),
                                                                     gate_want))
        if som.get("lesion") is not lesion:
            p.append("%s: som.lesion %r" % (tname, som.get("lesion")))
        if lesion and som.get("g_ap_max_run") != 0.0:
            p.append("%s: lesion g_ap_max_run %r != 0" % (tname, som.get("g_ap_max_run")))
        if lesion:
            for w, d in (som.get("per_word_ms") or {}).items():
                if any(v != 0.0 for v in d.get("g_ap_max") or []):
                    p.append("%s: lesion g_ap > 0 on %s" % (tname, w))
        sts = (som.get("stp_trial_start") or {}).get("stp_reset")
        if sts is not (arm == "stpreset"):
            p.append("%s: stp_trial_start.stp_reset %r" % (tname, sts))
    return p


def schedule_problems(ps):
    p = []
    c1 = ps.get("c1") or {}
    if c1.get("window_ms") != list(C1_WIN):
        p.append("c1 window %r" % c1.get("window_ms"))
    ev = c1.get("pulse_events") or []
    if [(e["word"], e["t0"], e["t1"] + 1) for e in ev] != [(C1_WORD, C1_PULSE[0], C1_PULSE[1])]:
        p.append("c1 pulse schedule %r" % [(e["word"], e["t0"], e["t1"]) for e in ev])
    c3s = ps.get("c3") or []
    if len(c3s) != 3:
        p.append("c3 trials %d" % len(c3s))
    for (x, y), c in zip(PAIRS, c3s):
        if c.get("window_ms") != list(C3_WIN):
            p.append("c3 %s>%s window %r" % (x, y, c.get("window_ms")))
        sch = [(e["word"], e["t0"], e["t1"] + 1) for e in c.get("pulse_events") or []]
        if sch != [(x, X_PULSE[0], X_PULSE[1]), (y, Y_PULSE[0], Y_PULSE[1])]:
            p.append("c3 %s>%s pulse schedule %r" % (x, y, sch))
        if c.get("word1_rate_hz") is None or c.get("word2_rate_hz") is None:
            p.append("c3 rates missing")
        som = c.get("som")
        if som is not None:
            pw = som.get("per_word_ms") or {}
            if set(pw) != {x, y}:
                p.append("c3 %s>%s per_word_ms words %r" % (x, y, sorted(pw)))
            if len(som.get("som_spikes_per_ms") or []) < C3_WIN[1]:
                p.append("c3 som series too short")
    if c1.get("som") is not None and len(c1["som"].get("som_spikes_per_ms") or []) < C8_WIN[1]:
        p.append("c1 som series too short")
    return p


# ── re-derivations ─────────────────────────────────────────────────────────────────────────────────────────────────

def _rate(count, n_cells, ms_len):
    return round(float(count) / (float(n_cells) * float(ms_len) / 1000.0), 4)


def termination_ms(vmax, v_hold, t0=Y_ON, t_end=C3_WIN[1]):
    """First ms t >= t0 such that vmax < v_hold at every ms of [t, t_end); None if the series is not below at t_end-1."""
    seg = list(vmax[t0:t_end])
    if len(seg) != t_end - t0:
        return None
    above = [i for i, v in enumerate(seg) if not (v is not None and v < v_hold)]
    if not above:
        return t0
    if above[-1] == len(seg) - 1:
        return None
    return t0 + above[-1] + 1


def _peak(series, lo, hi):
    seg = [(-1.0 if v is None else v) for v in series[lo:hi]]
    if not seg:
        return None, None
    m = max(seg)
    return m, lo + seg.index(m)


def rate_c3_pass(c):
    return bool(C1_LO <= float(c["word2_rate_hz"]) <= C1_HI and float(c["word1_rate_hz"]) < QUIET_HZ)


def c7_pair(tc, lc, older, newer, v_hold=V_HOLD):
    """C7 on one C3 pair: tc = the TRACE c3 trial, lc = the LESION c3 trial of the same cell, seed and pair (or None)."""
    som = tc["som"]
    pw = som["per_word_ms"]
    o, n = pw[older], pw[newer]
    out = {"older": older, "newer": newer}
    pre_min = min(o["v_apical_min"][PRE_LATCH[0]:PRE_LATCH[1]])
    out["pre_latch_min"] = pre_min
    out["0"] = bool(pre_min > v_hold)
    term = termination_ms(o["v_apical_max"], v_hold)
    out["termination_ms"] = term
    out["latency_ms"] = None if term is None else term - Y_ON
    out["a"] = bool(term is not None and term - Y_ON <= LAT_BAR)
    nmin = min(n["v_apical_min"][NEWER_HOLD[0]:NEWER_HOLD[1]])
    out["newer_hold_min"] = nmin
    out["b"] = bool(nmin > v_hold)
    n_som = int(som["n_som"])
    per_cell = [0] * n_som
    for ms, cell in som["som_cell_events"]:
        if SOM_WIN[0] <= ms < SOM_WIN[1]:
            per_cell[int(cell)] += 1
    pooled = int(sum(som["som_spikes_per_ms"][SOM_WIN[0]:SOM_WIN[1]]))
    out["som_spikes_per_cell"] = per_cell
    out["som_spikes_pooled"] = pooled
    spikes_cell = all(c >= 1 for c in per_cell)
    spikes_pool = pooled >= n_som
    gpk, gms = _peak(o["g_ap_max"], Y_ON, C3_WIN[1])
    gpk_all, gms_all = _peak(o["g_ap_max"], 0, len(o["g_ap_max"]))
    out["g_ap_peak"], out["g_ap_peak_ms"] = gpk, gms
    out["g_ap_peak_run"], out["g_ap_peak_run_ms"] = gpk_all, gms_all

    def _gap_ok(pk, pms):
        return bool(pk is not None and pk > 0.0 and (term is None or pms < term))

    out["c"] = bool(spikes_cell and _gap_ok(gpk, gms))
    out["c_alt_pooled"] = bool(spikes_pool and _gap_ok(gpk, gms))
    out["c_alt_runpeak"] = bool(spikes_cell and _gap_ok(gpk_all, gms_all))
    # DIAGNOSTICS, never a bar: why (c) failed, and whether the g_ap RISE preceded the termination (the self-skeptic
    # record's wording; the bar text says PEAK). The rise is the first ms after Y_on at which g_ap INCREASES (a routed
    # SOM spike arriving); "first ms > 0" would read the X volley's decayed residue (1e-6 at 300 ms) as a rise.
    g = o["g_ap_max"]
    onset = next((t for t in range(Y_ON, min(len(g), C3_WIN[1]))
                  if g[t] is not None and g[t - 1] is not None and g[t] > g[t - 1] + 1e-6), None)
    out["g_ap_onset_ms"] = onset
    out["g_ap_at_term_minus_1"] = (g[term - 1] if term is not None and term > Y_ON else None)
    out["diag_onset_precedes_term"] = bool(spikes_cell and onset is not None and (term is None or onset < term))
    if out["c"]:
        out["c_fail_reason"] = None
    elif not spikes_cell:
        out["c_fail_reason"] = "a SOM cell silent in [300, 600)"
    elif gpk is None or gpk <= 0.0:
        out["c_fail_reason"] = "older g_ap never > 0 in [300, 950)"
    else:
        out["c_fail_reason"] = "older g_ap peak at or after the termination ms"
    out["trace_rate_c3"] = rate_c3_pass(tc)
    if lc is None:
        out["d"] = None
        out["lesion_older_min"] = out["lesion_older_ended"] = None
        out["lesion_rate_c3"] = None
    else:
        lw = lc["som"]["per_word_ms"][older]
        lmin = min(lw["v_apical_min"][C3_WIN[0]:C3_WIN[1]])
        out["lesion_older_min"] = lmin
        out["lesion_older_ended"] = bool(max(lw["v_apical_max"][C3_WIN[0]:C3_WIN[1]]) < v_hold)
        out["lesion_rate_c3"] = rate_c3_pass(lc)
        out["d"] = bool(lmin > v_hold and not out["lesion_rate_c3"])
    if not out["0"]:
        out["status"] = UNDEF
    elif any(out[k] is False for k in SUBS):
        out["status"] = FAIL
    elif any(out[k] is None for k in SUBS):
        out["status"] = UNDEF
    else:
        out["status"] = PASS
    return out


def consistency(ps, v_hold=V_HOLD):
    """Re-derive the runner's summary fields from the raw the record keeps; every mismatch is listed (instrument)."""
    bad = []
    c1 = ps["c1"]
    win = c1["window_len_ms"]
    n_cells = c1["pulse_events"][0]["n_cells"]
    for wi, w in enumerate(MENTION_WORDS):
        r = _rate(c1["ments_spikes"][wi], n_cells, win)
        if abs(r - c1["word_rates_hz"][w]) > 1e-3:
            bad.append("c1 rate %s %s != %s" % (w, r, c1["word_rates_hz"][w]))
    for (x, y), c in zip(PAIRS, ps["c3"]):
        for wi, w in enumerate(MENTION_WORDS):
            r = _rate(c["ments_spikes"][wi], n_cells, c["window_len_ms"])
            if abs(r - c["word_rates_hz"][w]) > 1e-3:
                bad.append("c3 %s>%s rate %s" % (x, y, w))
    if "som" in c1:
        pw = c1["som"]["per_word_ms"][C1_WORD]
        mn = round(min(pw["v_apical_min"][C1_WIN[0]:C1_WIN[1]]), 4)
        mx = round(max(pw["v_apical_max"][C1_WIN[0]:C1_WIN[1]]), 4)
        wv = c1["v_apical"]["window"][C1_WORD]
        if abs(mn - wv["min"]) > 1e-3 or abs(mx - wv["max"]) > 1e-3:
            bad.append("c1 v_apical window %s/%s != %s/%s" % (mn, mx, wv["min"], wv["max"]))
        for tname, tr in _trials(ps):
            som = tr["som"]
            ev = [0] * len(som["som_spikes_per_ms"])
            for ms, _cell in som["som_cell_events"]:
                ev[ms] += 1
            if ev != list(som["som_spikes_per_ms"]):
                bad.append("%s: som_cell_events != som_spikes_per_ms" % tname)
        for (x, y), c, ca in zip(PAIRS, ps["c3"], ps.get("c3_apical") or [None] * 3):
            pw3 = c["som"]["per_word_ms"]
            omax = round(max(pw3[x]["v_apical_max"][C3_WIN[0]:C3_WIN[1]]), 4)
            nmin = round(min(pw3[y]["v_apical_min"][C3_WIN[0]:C3_WIN[1]]), 4)
            if abs(omax - c["v_apical"]["window"][x]["max"]) > 1e-3 or abs(
                    nmin - c["v_apical"]["window"][y]["min"]) > 1e-3:
                bad.append("c3 %s>%s v_apical window re-derivation" % (x, y))
            if ca is not None:
                term = termination_ms(pw3[x]["v_apical_max"], v_hold)
                a = bool(term is not None and term - Y_ON <= LAT_BAR)
                if a != bool(ca["older_plateau_ended"]):
                    bad.append("c3 %s>%s C7(a) %s != runner older_plateau_ended %s" % (x, y, a,
                                                                                      ca["older_plateau_ended"]))
    return bad


# ── per-seed items ─────────────────────────────────────────────────────────────────────────────────────────────────

def seed_items(ps, lesion_ps=None, v_hold=V_HOLD):
    """Every step-0 item on one seed's TRACE record (and its LESION record for C7(d))."""
    c1, c2b, c3s = ps["c1"], ps["c2b"], ps["c3"]
    it = {}
    evs = list(c1.get("pulse_events") or []) + [e for c in c3s for e in (c.get("pulse_events") or [])]
    it["C0"] = _st(bool(evs) and all(int(e["min_cell_spikes"]) >= 1 for e in evs)) if evs else UNDEF
    win = c1["window_len_ms"]
    n_cells = c1["pulse_events"][0]["n_cells"]
    r1 = _rate(c1["ments_spikes"][MENTION_WORDS.index(C1_WORD)], n_cells, win)
    it["c1_rate_hz"] = r1
    it["C1"] = _st(C1_LO <= r1 <= C1_HI)
    it["C2"] = _st(float(c1["unpulsed_max_hz"]) < QUIET_HZ and float(c2b["max_word_hz"]) < QUIET_HZ)
    it["c3_rate_pairs"] = [rate_c3_pass(c) for c in c3s]
    it["C3"] = _st(all(it["c3_rate_pairs"]))
    it["C4"] = _st(c1.get("burst_ms") is None and c1.get("latch_ms") is None)
    c5 = ps.get("c5") or {}
    it["C5a"], it["C5b"] = c5.get("clause_a_delta_pa"), c5.get("clause_b_delta_pa")
    va = c1["v_apical"]
    reached = va["run_max"][C1_WORD] > v_hold
    pmin = va["window"][C1_WORD]["min"]
    un_max = max(va["window"][w]["max"] for w in MENTION_WORDS if w != C1_WORD)
    it["c6_pulsed_window_min"], it["c6_unpulsed_window_max"] = pmin, un_max
    it["C6"] = UNDEF if not reached else _st(pmin > v_hold and un_max < v_hold)
    if "som" not in c1:
        it["C7"] = it["C8"] = UNDEF
        it["c7_pairs"] = []
        return it
    lc3 = (lesion_ps or {}).get("c3") if lesion_ps else None
    pairs = [c7_pair(c, (lc3[i] if lc3 else None), x, y, v_hold) for i, ((x, y), c) in enumerate(zip(PAIRS, c3s))]
    it["c7_pairs"] = pairs
    sts = [p["status"] for p in pairs]
    it["C7"] = FAIL if FAIL in sts else (UNDEF if UNDEF in sts else PASS)
    som = c1["som"]
    n_som = int(som["n_som"])
    sp = int(sum(som["som_spikes_per_ms"][C8_WIN[0]:C8_WIN[1]]))
    secs = (C8_WIN[1] - C8_WIN[0]) / 1000.0
    it["c8_spikes"] = sp
    it["c8_rate_hz"] = round(sp / (n_som * secs), 4)
    it["c8_pool_sum_hz"] = round(sp / secs, 4)
    it["C8"] = _st(it["c8_rate_hz"] < QUIET_HZ)
    it["C8_alt_poolsum"] = _st(it["c8_pool_sum_hz"] < QUIET_HZ)
    # DIAGNOSTICS, never a bar: when the c1 word's own plateau fell (first ms of [150, 800) with its per-ms min below
    # v_hold), its own SOM tail (spikes in [150, 200)) and the g_ap it carried just after its drive ended.
    pw = som["per_word_ms"][C1_WORD]
    it["c6_drop_ms"] = next((t for t in range(C1_WIN[0], C1_WIN[1]) if pw["v_apical_min"][t] < v_hold), None)
    it["c1_som_tail_spikes"] = int(sum(som["som_spikes_per_ms"][C1_PULSE[1]:C1_PULSE[1] + 50]))
    it["c1_g_ap_max_150_250"] = max(pw["g_ap_max"][C1_PULSE[1]:C1_PULSE[1] + 100])
    return it


# ── cell level and decision ────────────────────────────────────────────────────────────────────────────────────────

def _counts(vals):
    return {PASS: vals.count(PASS), FAIL: vals.count(FAIL), UNDEF: vals.count(UNDEF), "n": len(vals)}


def summarize_cell(seed_rows, n_seeds=len(SEEDS)):
    """seed_rows: {seed: items or None (invalid artifact)}."""
    cs = {}
    for item in ITEMS_SEL:
        vals = [(r[item] if r else UNDEF) for r in seed_rows.values()]
        cs[item] = _counts(vals)
        cs[item]["full"] = bool(vals.count(PASS) == n_seeds == len(vals))
    pairs = [p for r in seed_rows.values() if r for p in r.get("c7_pairs") or []]
    n_pairs = 3 * n_seeds
    defined = [p for p in pairs if p["0"]]
    sub = {"pairs": len(pairs), "defined": len(defined)}
    for k in SUBS + ("c_alt_pooled", "c_alt_runpeak", "diag_onset_precedes_term"):
        sub[k] = sum(1 for p in defined if p[k] is True)
        sub[k + "_full"] = bool(sub[k] == n_pairs)
        sub[k + "_defined_fails"] = sum(1 for p in defined if p[k] is False)
    sub["latencies_ms"] = [p["latency_ms"] for p in defined]   # defined pairs only: an unlatched plateau has none
    reasons = {}
    for p in defined:
        if p.get("c_fail_reason"):
            reasons[p["c_fail_reason"]] = reasons.get(p["c_fail_reason"], 0) + 1
    sub["c_fail_reasons"] = reasons
    sub["undefined_pairs"] = [(s, p["older"], p["newer"], p["pre_latch_min"]) for s, r in seed_rows.items() if r
                              for p in r.get("c7_pairs") or [] if not p["0"]]
    # The lesion's share of the overwrite on the defined pairs (reported, not a bar: the bars are C7(c) and (d)).
    with_l = [p for p in defined if p["lesion_older_ended"] is not None]
    att = {}
    for key, t_key, c_key in (("older_plateau_ended", "a", "lesion_older_ended"),
                              ("rate_c3_pair_pass", "trace_rate_c3", "lesion_rate_c3")):
        t = sum(1 for p in with_l if p[t_key])
        c = sum(1 for p in with_l if p[c_key])
        with contextlib.redirect_stdout(io.StringIO()):
            f = attributable_to("SOM->apical gate: %s" % key, t, c)
        att[key] = {"trace": t, "lesion": c, "n_pairs": len(with_l), "fraction_attributable": f}
    sub["attribution"] = att
    cs["c7_sub"] = sub
    c8alt = [(r.get("C8_alt_poolsum", UNDEF) if r else UNDEF) for r in seed_rows.values()]
    cs["C8_alt_poolsum"] = _counts(c8alt)
    cs["C8_alt_poolsum"]["full"] = bool(c8alt.count(PASS) == n_seeds == len(c8alt))
    a = [r["C5a"] for r in seed_rows.values() if r and r.get("C5a") is not None]
    b = [r["C5b"] for r in seed_rows.values() if r and r.get("C5b") is not None]
    cs["c5_median"] = {"a": round(statistics.median(a), 4) if len(a) == n_seeds else None,
                       "b": round(statistics.median(b), 4) if len(b) == n_seeds else None}
    cs["c6_fail_drop_ms"] = sorted(r.get("c6_drop_ms") for r in seed_rows.values()
                                   if r and r["C6"] == FAIL and r.get("c6_drop_ms") is not None)
    cs["c1_rates_hz"] = [r["c1_rate_hz"] for r in seed_rows.values() if r]
    cs["c8_rates_hz"] = [r.get("c8_rate_hz") for r in seed_rows.values() if r]
    cs["selectable"] = all(cs[i]["full"] for i in ITEMS_SEL)
    cs["unattributed"] = bool(cs["C3"]["full"] and not (sub["c_full"] and sub["d_full"]))
    cs["smoke"] = (None if None in cs["c5_median"].values() else
                   bool(cs["c5_median"]["a"] < SMOKE_PA or cs["c5_median"]["b"] < SMOKE_PA))
    # what the primary vs the second readings would change at this cell
    alt = {}
    if sub["c_full"] != sub["c_alt_pooled_full"]:
        alt["C7(c) pooled spikes"] = sub["c_alt_pooled_full"]
    if sub["c_full"] != sub["c_alt_runpeak_full"]:
        alt["C7(c) whole-trial g_ap peak"] = sub["c_alt_runpeak_full"]
    if cs["C8"]["full"] != cs["C8_alt_poolsum"]["full"]:
        alt["C8 pool-sum rate"] = cs["C8_alt_poolsum"]["full"]
    cs["second_readings_differ"] = alt
    return cs


def decide(cells, stop_engine=False, stop_id=False, global_c5=None):
    """cells: {(sa, ms): summary}. The packet's selection, STOP rules and branches, in its order."""
    order = sorted(cells, key=lambda k: (k[0], k[1]))
    selectable = [k for k in order if cells[k]["selectable"]]
    sel = selectable[0] if selectable else None
    c3_cells = [k for k in order if cells[k]["C3"]["full"]]
    d = {"selected": None if sel is None else cell_tag(*sel), "selectable": [cell_tag(*k) for k in selectable],
         "c3_passing": [cell_tag(*k) for k in c3_cells],
         "unattributed": [cell_tag(*k) for k in order if cells[k]["unattributed"]]}
    d["STOP_ENGINE"] = bool(stop_engine)
    d["STOP_ID"] = bool(stop_id)
    d["STOP_CAL"] = sel is None
    d["STOP_SMOKE_selected"] = None if sel is None else cells[sel]["smoke"]
    d["STOP_SMOKE_global"] = (None if not global_c5 or None in global_c5.values() else
                              bool(global_c5["a"] < SMOKE_PA or global_c5["b"] < SMOKE_PA))
    d["STOP_ATTRIB"] = bool(c3_cells) and all(cells[k]["unattributed"] for k in c3_cells)
    sub = {k: cells[k]["c7_sub"] for k in order}
    d["B1"] = bool(order) and all(sub[k]["a_full"] and not sub[k]["b_full"] for k in order)
    d["B2"] = bool(order) and all(not cells[k]["C8"]["full"] for k in order)
    d["B3"] = bool(order) and all(sub[k]["c_full"] and not sub[k]["a_full"] for k in order)
    c3c7 = [k for k in order if cells[k]["C3"]["full"] and cells[k]["C7"]["full"]]
    d["B4"] = bool(c3c7) and all(cells[k]["smoke"] for k in c3c7)
    if sel is not None:
        s = cells[sel]["c5_median"]
        d["selected_c5_vs_flip"] = {"a": s["a"], "b": s["b"], "flip_scale_pa": list(FLIP_PA),
                                    "a_at_or_above_4": s["a"] >= FLIP_PA[0], "b_at_or_above_6": s["b"] >= FLIP_PA[1]}
    d["c3_cells_c5_median"] = {cell_tag(*k): cells[k]["c5_median"] for k in c3_cells}
    d["c3_cells_attribution"] = {cell_tag(*k): {"c": cells[k]["c7_sub"]["c"],
                                                "c_defined_fails": cells[k]["c7_sub"]["c_defined_fails"],
                                                "d": cells[k]["c7_sub"]["d"],
                                                "d_defined_fails": cells[k]["c7_sub"]["d_defined_fails"],
                                                "defined": cells[k]["c7_sub"]["defined"]} for k in c3_cells}
    stops = [n for n in ("STOP_ENGINE", "STOP_ID", "STOP_CAL", "STOP_ATTRIB") if d[n]]
    if d["STOP_SMOKE_selected"]:
        stops.append("STOP_SMOKE")
    d["stops"] = stops
    d["branches"] = [b for b in ("B1", "B2", "B3", "B4") if d[b]]
    if stops or d["branches"]:
        d["verdict"] = "STOP: " + ", ".join(stops + d["branches"])
    else:
        d["verdict"] = "CELL SELECTED: %s (no STOP rule fires; step 1 waits for the ruling)" % d["selected"]
    return d


# ── the whole read ─────────────────────────────────────────────────────────────────────────────────────────────────

def _git_has(rev, path):
    try:
        r = subprocess.run(["git", "-C", _REPO, "cat-file", "-e", "%s:%s" % (rev, path)], capture_output=True)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return None


def p0_engine(pin=PIN):
    """P0' read-only part: the E build's files exist at the pin (the test run itself is recorded by hand)."""
    files = ("tests/test_apical_gaba_routing.py", "research/biology/dendrite-targeting-som-inhibition.md",
             "tests/test_cs_race_f1i_flags.py")
    return {f: _git_has(pin, f) for f in files}


def score(root, f1h_root, lines_path, pin=PIN, seeds=SEEDS):
    lines = parse_lines(lines_path) if lines_path else None
    res = {"packet": PACKET, "pin": pin, "root": root, "seeds": list(seeds), "invalid": {}, "stop_id_problems": {},
           "consistency": {}}
    # identity (i)
    ident = {}
    for s in IDENTITY_SEEDS:
        name = art_name("identity", s)
        rec, probs = load_artifact(root, name, lines, pin=pin)
        if rec is not None:
            probs += flag_problems(rec, "identity", s)
        f1h = os.path.join(f1h_root, "cal_sg40_wim40_s%d.json" % s)
        try:
            with open(f1h) as f:
                ref = json.load(f)
            eq = rec is not None and canon_hash(rec) == canon_hash(ref)
            byte_eq = rec is not None and file_sha(os.path.join(root, name)) == file_sha(f1h)
        except Exception as e:  # noqa: BLE001
            probs.append("f1h reference unreadable: %s" % (e,))
            eq = byte_eq = None
        ident[s] = {"hash_equal_minus_wall_s": eq, "file_sha256_equal": byte_eq, "problems": probs}
    res["identity_i"] = ident
    # per cell
    cells, cell_rows, all_c5a, all_c5b = {}, {}, [], []
    lesion_ii = []
    for sa in W_SA:
        for ms in W_MS:
            rows, lrows = {}, {}
            for s in seeds:
                names = {arm: art_name(arm, s, sa, ms) for arm in ("trace", "lesion")}
                recs = {}
                for arm, name in names.items():
                    rec, probs = load_artifact(root, name, lines, pin=pin)
                    if rec is not None:
                        probs += flag_problems(rec, arm, s, sa, ms)
                        if not probs:
                            probs += schedule_problems(rec["per_seed"][0])
                        sid = stop_id_problems(rec, arm) if not probs else []
                        if sid:
                            res["stop_id_problems"][name] = sid
                        if arm == "lesion":
                            lesion_ii.append(not sid and not probs)
                    if probs:
                        res["invalid"][name] = probs
                        rec = None
                    recs[arm] = rec
                tr, lr = recs["trace"], recs["lesion"]
                if tr is None:
                    rows[s] = None
                else:
                    ps = tr["per_seed"][0]
                    bad = consistency(ps)
                    if bad:
                        res["consistency"][names["trace"]] = bad
                    lps = lr["per_seed"][0] if lr is not None else None
                    rows[s] = seed_items(ps, lps)
                    if rows[s]["C5a"] is not None:
                        all_c5a.append(rows[s]["C5a"])
                    if rows[s]["C5b"] is not None:
                        all_c5b.append(rows[s]["C5b"])
                if lr is not None:
                    lps = lr["per_seed"][0]
                    lrows[s] = {"rate_c3_pairs": [rate_c3_pass(c) for c in lps["c3"]],
                                "older_held": [min(c["som"]["per_word_ms"][x]["v_apical_min"][C3_WIN[0]:C3_WIN[1]])
                                               > V_HOLD for (x, _y), c in zip(PAIRS, lps["c3"])],
                                "c6": seed_items(lps, None)["C6"], "som_spikes_c3": [
                                    int(sum(c["som"]["som_spikes_per_ms"])) for c in lps["c3"]]}
                else:
                    lrows[s] = None
            cells[(sa, ms)] = summarize_cell(rows, len(seeds))
            cells[(sa, ms)]["lesion_arm"] = {
                "rate_c3_pairs_pass": sum(sum(r["rate_c3_pairs"]) for r in lrows.values() if r),
                "older_held_pairs": sum(sum(r["older_held"]) for r in lrows.values() if r),
                "c6_pass": sum(1 for r in lrows.values() if r and r["c6"] == PASS),
                "n": sum(1 for r in lrows.values() if r)}
            cell_rows[(sa, ms)] = rows
    res["identity_ii"] = {"lesion_records": len(lesion_ii), "clean": sum(lesion_ii)}
    gm = {"a": round(statistics.median(all_c5a), 4) if all_c5a else None,
          "b": round(statistics.median(all_c5b), 4) if all_c5b else None}
    res["global_c5_median"] = gm
    p0 = p0_engine(pin)
    res["p0_engine_files_at_pin"] = p0
    stop_engine = any(v is False for v in p0.values())
    stop_id = (any(not (v["hash_equal_minus_wall_s"] and not v["problems"]) for v in ident.values())
               or bool(res["stop_id_problems"]))
    res["cells"] = {cell_tag(*k): v for k, v in cells.items()}
    res["seed_rows"] = {cell_tag(*k): {str(s): r for s, r in v.items()} for k, v in cell_rows.items()}
    res["decision"] = decide(cells, stop_engine=stop_engine, stop_id=stop_id, global_c5=gm)
    res["stpreset"] = stpreset_compare(root, lines, pin, cell_rows)
    return res


def stpreset_compare(root, lines, pin, cell_rows):
    """R1 Q5: every item verdict at seed 41, STP-reset arm against the TRACE line of the same cell."""
    out = {}
    s = STPRESET_SEED
    for sa in W_SA:
        for ms in W_MS:
            tag = cell_tag(sa, ms)
            name = art_name("stpreset", s, sa, ms)
            rec, probs = load_artifact(root, name, lines, pin=pin)
            if rec is not None:
                probs += flag_problems(rec, "stpreset", s, sa, ms)
                if not probs:
                    probs += schedule_problems(rec["per_seed"][0])
                probs += stop_id_problems(rec, "stpreset") if not probs else []
            base = cell_rows[(sa, ms)].get(s)
            if probs or base is None:
                out[tag] = {"problems": probs or ["TRACE seed-41 record invalid"], "changed": None}
                continue
            lesion_name = art_name("lesion", s, sa, ms)
            lrec, lprobs = load_artifact(root, lesion_name, lines, pin=pin)
            lps = lrec["per_seed"][0] if (lrec is not None and not lprobs) else None
            it = seed_items(rec["per_seed"][0], lps)
            changed = {}
            for k in ITEMS_SEL:
                if it[k] != base[k]:
                    changed[k] = {"trace": base[k], "stpreset": it[k]}
            for i, (pb, pr) in enumerate(zip(base.get("c7_pairs") or [], it.get("c7_pairs") or [])):
                for k in ("0",) + SUBS:
                    if pb[k] != pr[k]:
                        changed["C7(%s) pair %d" % (k, i)] = {"trace": pb[k], "stpreset": pr[k]}
            ee0 = rec["per_seed"][0]["c1"]["som"]["stp_trial_start"]["ee"]
            out[tag] = {"problems": [], "changed": changed, "items": {k: it[k] for k in ITEMS_SEL},
                        "stpreset_ee_u_max_at_c1_start": ee0["u_max"] if ee0 else None}
    return out


# ── rendering ──────────────────────────────────────────────────────────────────────────────────────────────────────

def _pfu(c):
    return "%d/%d/%d" % (c[PASS], c[FAIL], c[UNDEF])


def render(res):
    L = []
    d = res["decision"]
    L.append("Pin %s; seeds %s; items as P/F/U counts over the eight seeds (8/8 = 8/0/0)." % (
        res["pin"][:9], ",".join(str(s) for s in res["seeds"])))
    L.append("")
    L.append("| cell | C0 | C1 | C2 | C3 | C4 | C6 | C7 | C8 | med C5a | med C5b | selectable |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, c in res["cells"].items():
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            tag, _pfu(c["C0"]), _pfu(c["C1"]), _pfu(c["C2"]), _pfu(c["C3"]), _pfu(c["C4"]), _pfu(c["C6"]),
            _pfu(c["C7"]), _pfu(c["C8"]), c["c5_median"]["a"], c["c5_median"]["b"],
            "yes" if c["selectable"] else "no"))
    L.append("")
    L.append("C7 sub-items, defined passes over 24 pairs ((0) = pairs whose older plateau latched before Y), "
             "latencies in ms after Y_on (None = the older plateau never ended), lesion arm per cell:")
    L.append("")
    L.append("| cell | (0) | (a) | (b) | (c) | (d) | latencies ms | lesion rate-C3 pass | lesion older held | "
             "lesion C6 | C1 Hz | C8 Hz |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, c in res["cells"].items():
        s = c["c7_sub"]
        la = c["lesion_arm"]
        lat = s["latencies_ms"]
        L.append("| %s | %d | %d | %d | %d | %d | %s | %d/24 | %d/24 | %d/%d | %s-%s | %s |" % (
            tag, s["defined"], s["a"], s["b"], s["c"], s["d"],
            _lat_summary(lat), la["rate_c3_pairs_pass"], la["older_held_pairs"], la["c6_pass"], la["n"],
            min(c["c1_rates_hz"]) if c["c1_rates_hz"] else None, max(c["c1_rates_hz"]) if c["c1_rates_hz"] else None,
            max(c["c8_rates_hz"]) if c["c8_rates_hz"] else None))
    L.append("")
    for tag, c in res["cells"].items():
        s = c["c7_sub"]
        if s["c_fail_reasons"] or s["undefined_pairs"]:
            L.append("- %s: C7(c) defined fails by reason %s; diagnostic (not a bar) first g_ap increase after Y_on before termination "
                     "%d/%d defined; UNDEFINED pairs (seed, older, newer, older min v_apical in [150, 300)) %s" % (
                         tag, json.dumps(s["c_fail_reasons"]), s["diag_onset_precedes_term"], s["defined"],
                         json.dumps(s["undefined_pairs"])))
    L.append("")
    L.append("Lesion attribution on the defined pairs (tools.lab.attributable_to, reported, not a bar): %s" % json.dumps(
        {t: {k: "%d vs %d of %d -> %s" % (v["trace"], v["lesion"], v["n_pairs"], v["fraction_attributable"])
             for k, v in c["c7_sub"]["attribution"].items()} for t, c in res["cells"].items()}))
    L.append("")
    L.append("C6 FAIL diagnostic (not a bar): ms at which the c1 word's own plateau fell, per failing seed: %s" % (
        json.dumps({t: c["c6_fail_drop_ms"] for t, c in res["cells"].items() if c["c6_fail_drop_ms"]}),))
    L.append("")
    alts = {t: c["second_readings_differ"] for t, c in res["cells"].items() if c["second_readings_differ"]}
    L.append("Second readings that would change a cell-level verdict: %s" % (json.dumps(alts) if alts else "none"))
    L.append("Identity (i): %s" % json.dumps({s: {k: v for k, v in r.items()} for s, r in res["identity_i"].items()}))
    L.append("Identity (ii): %d/%d lesion records clean (gate 0.0 at measurement and g_ap 0 at every ms, every trial)"
             % (res["identity_ii"]["clean"], res["identity_ii"]["lesion_records"]))
    L.append("Invalid artifacts: %d; STOP-ID read-back problems: %d; consistency mismatches: %d" % (
        len(res["invalid"]), len(res["stop_id_problems"]), len(res["consistency"])))
    L.append("P0' files at pin: %s" % json.dumps(res["p0_engine_files_at_pin"]))
    L.append("Global C5 median over all TRACE records: %s" % json.dumps(res["global_c5_median"]))
    L.append("R1 Q5 (STP reset at seed 41, item changes against TRACE): %s" % json.dumps(
        {t: v["changed"] for t, v in res["stpreset"].items()}))
    L.append("")
    L.append("DECISION: %s" % json.dumps(d))
    return "\n".join(L)


def _lat_summary(lat):
    nums = sorted(x for x in lat if x is not None)
    none = sum(1 for x in lat if x is None)
    if not nums:
        return "none ended (%d)" % none
    return "%d-%d (med %g)%s" % (nums[0], nums[-1], statistics.median(nums), ("; %d None" % none) if none else "")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join(_REPO, STEP0_DIR))
    ap.add_argument("--f1h-root", default=os.path.join(_REPO, F1H_DIR))
    ap.add_argument("--lines", default=os.path.join(_REPO, LINES))
    ap.add_argument("--pin", default=PIN)
    ap.add_argument("--out", default=None, help="JSON result (under research/findings/raw for a provenance sidecar)")
    a = ap.parse_args(argv)
    res = score(a.root, a.f1h_root, a.lines, pin=a.pin)
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
