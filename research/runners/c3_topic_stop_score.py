"""Scorer for the C3 curiosity per-topic-stop battery (L+P): the dev read (bars D1-D8) and the gate-seed read (G1-G10).

Packet: research/coordination/c3-topic-stop-PACKET.md (section 7 arms and bars D1-D8, decision rule). Gate prereg:
research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md (G1-G8 = D1-D8 at the six gate seeds, G9 the Stage-2
C3 clause, G10 the answer-rate bar, the UNDEFINED rules, the DEC line). This module adds no bar and loosens none; it is
the reader both documents name. It reads data only and runs no brain, so it may read gate-seed shards (the gate-seed
guard is for the runs that PRODUCE them).

    python -m research.runners.c3_topic_stop_score --mode dev                                 # the 192 dev shards
    python -m research.runners.c3_topic_stop_score --mode gate --pin <F, full sha> \\
        [--root research/findings/raw/_c3_topic_stop/gate] [--seeds 42,43,44,100,101,102] [--out <json>]
    python -m research.runners.c3_topic_stop_score --emit-lines [<sha>]                        # the gate pool lines (TSV)
    python -m research.runners.c3_topic_stop_score --closure [--rev <sha>]                     # the arms' import closure
    python -m research.runners.c3_topic_stop_score --closure-check <P> <F>                     # IDENTICAL or the diffs

Layout: <root>/s<seed>/ask__<tag>.json plus its `.prov.json` sidecar, one fresh process per (seed, tag); tags and env
are TAGS below (packet section 7, sixteen arms per seed).

CELL VALIDITY (a cell that fails any rule is UNDEFINED, never a pass or a fail): the shard and its sidecar exist and
parse; the sidecar is a clean measurement of the pin (git_sha == pin, git_dirty False; gate mode also needs
source_kind git_archive with the manifest verified at start and exit, and `--allow-gate-seed --prereg <PREREG>` in its
argv); the sidecar argv names this arm and seed; the sidecar's BRAIN_CURIOSITY* env equals the tag's; the shard's own
arm / seed / group; `stage2_exit_gate.arm_validity` (source, PYTHONHASHSEED, corpus, errored turns, organ errors, the
registered turn table); ENGAGEMENT: with L on, every probe turn that abstained carries a `curiosity_topic_stop` trace
with `applied` True, no `error` and `held.ou_enabled` False, and (except under lesion_L) `held.stp_tau_d_ms` 12000;
with P on, every applied trace carries `phasic_da` with `phasic_da_flag` and `phasic_da_applied` True and (except under
lesion_P) `reward_us_spikes` > 0 wherever `salience` > 0; p_const_salience arms carry `const_salience` True on every
applied trace; L-pinned arms carry no phasic read; flag-off arms carry no `curiosity_topic_stop` key on any turn.

BAR READINGS (the bar text is the packet's / the prereg's; this is how each is read):
  1  (D1/G1) C intact_a and C reversed: `score_ask(shard)["all"]`; a seed is FAIL if either arm is False, UNDEFINED
     if either reads `moat_undefined_decides` (and none False), else PASS.
  2  (D2/G2) C intact_b's asked vector over the twelve probe labels equals C intact_a's.
  3  (D3/G3) `score_ask(C_permuted, permuted=True)["discriminates"]` is True (the shard's own teach acquisition).
  4  (D4/G4) both orders: lesion_L asks u_toad_4 and C does not; and the pooled attribution
     `tools.lab.attributable_to(treatment, control)` >= 0.5, treatment = toad probes (u_toad_1..4) asked under
     lesionL_intact_a and not under C_intact_a, summed over the counted seeds; control = toad probes where C_intact_a
     and C_intact_b differ in asked state, summed over the same seeds. The reversed-order treatment is reported.
  5  (D5/G5) lesionP_intact_a: lp_learnable False and lp_unlearnable False.
  6  (D6/G6) C_lesion_curiosity: `asked` False on every turn.
  7  (D7/G7) `abstained` and `recalled_svo` (stage2_exit_gate._primary) identical on every turn to the same-order
     flag-off arm, for the eight pairs of D7_PAIRS.
  8  (D8/G8) every applied trace of each lesion arm reads its lesion back (lesion_L stp_tau_d_ms 200; lesion_P
     reward_us_spikes 0; lesion_curiosity topic_ask_weight_sum 0), at least one applied trace, and no abstained probe
     without a trace.
  S2C3 (G9) the Stage-2 C3 clause: `score_ask(shard)["all"]` True on C intact_a and on C reversed (the same call as 1).
  AR (G10) for each AR_PAIRS pair, per turn: no turn where the flag-off reply is `answered` and the flag-on reply is not
     (`tools.answer_rate.classify`), and `tools.answer_rate.rate(on)["answered"] >= rate(off)["answered"]` over all turns.
Tallies: gate mode needs every given seed (6 of 6 at the gate); dev mode needs 11 of the 11 counted seeds (9 for D5),
seed 9 (the design seed) reported and never counted. A bar is PASS when its PASS seeds reach the need, FAIL when its
FAIL seeds make the need unreachable, else UNDEFINED. DEC: every bar PASS -> PASS; any FAIL -> FAIL; else UNDEFINED.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import subprocess
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage2_exit_gate as G  # noqa: E402
from tools.lab import attributable_to  # noqa: E402  (the lesion's share of the toad asks: gates/attribution_required)
from tools.answer_rate import classify, rate  # noqa: E402

PREREG = "research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md"
PACKET = "research/coordination/c3-topic-stop-PACKET.md"
DEV_DIR = "research/findings/raw/_c3_topic_stop/dev"
GATE_DIR = "research/findings/raw/_c3_topic_stop/gate"
PREFLIGHT_DIR = "research/findings/raw/_c3_topic_stop/preflight"
DEV_PIN = "79c6f5128916003d240f1d1cbb3f1abd715f163d"
GATE_SEEDS = tuple(G.GATE_SEEDS)
DEV_SEEDS = tuple(range(1, 13))
DESIGN_SEED = 9
DEV_COUNTED = tuple(s for s in DEV_SEEDS if s != DESIGN_SEED)
MEM_GB, MEMCAP_GB, MAX_AGE_H = 9, 10, 240

C_ENV = (("BRAIN_CURIOSITY_TOPIC_STOP", "1"), ("BRAIN_CURIOSITY_PHASIC_DA", "1"))
_L_LES = (("BRAIN_CURIOSITY_TOPIC_STOP_LESION", "1"),)
_P_LES = (("BRAIN_CURIOSITY_PHASIC_DA_LESION", "1"),)
_PIN = (("BRAIN_CURIOSITY_TOPIC_STOP", "1"), ("BRAIN_CURIOSITY_DA_LESION", "1"))
_CONST = (("BRAIN_CURIOSITY_PHASIC_DA_CONST_SALIENCE", "1"),)
# tag -> (runner arm, line env in the dev lines' order, kind). kind: counted | baseline | informational
TAGS = {
    "C_intact_a": ("intact_a", C_ENV, "counted"),
    "C_intact_b": ("intact_b", C_ENV, "counted"),
    "C_reversed": ("reversed", C_ENV, "counted"),
    "C_permuted": ("permuted", C_ENV, "counted"),
    "C_lesion_curiosity": ("lesion_curiosity", C_ENV, "counted"),
    "lesionL_intact_a": ("intact_a", C_ENV + _L_LES, "counted"),
    "lesionL_reversed": ("reversed", C_ENV + _L_LES, "counted"),
    "lesionP_intact_a": ("intact_a", C_ENV + _P_LES, "counted"),
    "lesionP_reversed": ("reversed", C_ENV + _P_LES, "counted"),
    "off_intact_a": ("intact_a", (), "baseline"),
    "off_reversed": ("reversed", (), "baseline"),
    "off_permuted": ("permuted", (), "baseline"),
    "Lpinned_intact_a": ("intact_a", _PIN, "informational"),
    "Lpinned_reversed": ("reversed", _PIN, "informational"),
    "pconst_intact_a": ("intact_a", C_ENV + _CONST, "informational"),
    "pconst_reversed": ("reversed", C_ENV + _CONST, "informational"),
}
TOADS = ("u_toad_1", "u_toad_2", "u_toad_3", "u_toad_4")
LAB12 = ("g_eel", "m_yak", "g_crab", "m_moose", "g_newt", "m_goat", "lp_ibex_1", "lp_ibex_2") + TOADS
D7_PAIRS = (("C_intact_a", "off_intact_a"), ("C_intact_b", "off_intact_a"), ("lesionL_intact_a", "off_intact_a"),
            ("lesionP_intact_a", "off_intact_a"), ("C_reversed", "off_reversed"), ("lesionL_reversed", "off_reversed"),
            ("lesionP_reversed", "off_reversed"), ("C_permuted", "off_permuted"))
AR_PAIRS = (("C_intact_a", "off_intact_a"), ("C_intact_b", "off_intact_a"), ("C_reversed", "off_reversed"),
            ("C_permuted", "off_permuted"))
READBACKS = (("lesionL_intact_a", "tau", 200.0), ("lesionL_reversed", "tau", 200.0),
             ("lesionP_intact_a", "us", 0), ("lesionP_reversed", "us", 0), ("C_lesion_curiosity", "w", 0.0))
BARS = ("1", "2", "3", "4", "5", "6", "7", "8", "S2C3", "AR")
PACKET_BARS = ("1", "2", "3", "4", "5", "6", "7", "8")
# the processes the arms run: the runner (it imports webapp.server inside the worker), the handler and the two organs
CLOSURE_ROOTS = ("research/runners/stage2_exit_gate.py", "webapp/server.py", "webapp/curiosity_topic_stop_chat.py",
                 "research/runners/curiosity_topic_stop_organ.py")
_SYM = {True: "1", False: "0", None: "-"}


def env_of(tag):
    return dict(TAGS[tag][1])


def _l_on(tag):
    return env_of(tag).get("BRAIN_CURIOSITY_TOPIC_STOP") == "1"


def _p_on(tag):
    return env_of(tag).get("BRAIN_CURIOSITY_PHASIC_DA") == "1"


def ts_of(resp):
    c = resp.get("curiosity") if isinstance(resp, dict) else None
    return c.get("curiosity_topic_stop") if isinstance(c, dict) else None


# ══════════════════════════════════════════════ cells: load + validity ════════════════════════════════════════════
def cell_path(root, seed, tag):
    return os.path.join(root, "s%d" % int(seed), "ask__%s.json" % tag)


def load_cell(root, seed, tag):
    """{"shard", "prov", "missing"/"load_error"}: a cell as it is on disk (no judgement)."""
    p = cell_path(root, seed, tag)
    cell = {"shard": None, "prov": None, "missing": None, "load_error": None}
    if not os.path.exists(p):
        cell["missing"] = "shard missing (%s)" % os.path.relpath(p, _REPO) if p.startswith(_REPO) else "shard missing"
        return cell
    try:
        with open(p) as f:
            cell["shard"] = json.load(f)
    except Exception as e:  # noqa: BLE001
        cell["load_error"] = "shard unreadable: %r" % (e,)
        return cell
    pp = p + ".prov.json"
    if os.path.exists(pp):
        try:
            with open(pp) as f:
                cell["prov"] = json.load(f)
        except Exception as e:  # noqa: BLE001
            cell["load_error"] = "sidecar unreadable: %r" % (e,)
    return cell


def load_cells(root, seeds):
    return {(int(s), tag): load_cell(root, s, tag) for s in seeds for tag in TAGS}


def engagement_problems(shard, tag):
    """The ENGAGEMENT rules of the module docstring: a cell that breaks one is not a measurement of its arm."""
    why = []
    resp = shard.get("responses") or {}
    turns = [t for t in (shard.get("turns") or []) if isinstance(t, dict)]
    if not _l_on(tag):
        keyed = [t["label"] for t in turns if ts_of(resp.get(t["label"])) is not None]
        if keyed:
            why.append("engagement: flag-off arm carries a curiosity_topic_stop trace on %s" % ",".join(keyed[:6]))
        return why
    lesion_l = env_of(tag).get("BRAIN_CURIOSITY_TOPIC_STOP_LESION") == "1"
    lesion_p = env_of(tag).get("BRAIN_CURIOSITY_PHASIC_DA_LESION") == "1"
    const = env_of(tag).get("BRAIN_CURIOSITY_PHASIC_DA_CONST_SALIENCE") == "1"
    no_trace, bad = [], []
    for t in turns:
        r = resp.get(t["label"])
        ts = ts_of(r)
        if t.get("role") == "probe" and G._abstained(r) is True and not (isinstance(ts, dict) and ts.get("applied")):
            no_trace.append(t["label"])
        if not (isinstance(ts, dict) and ts.get("applied")):
            continue
        h = ts.get("held") or {}
        if ts.get("error"):
            bad.append("%s error" % t["label"])
        if h.get("ou_enabled") is not False:
            bad.append("%s ou_enabled %s" % (t["label"], h.get("ou_enabled")))
        if not lesion_l and h.get("stp_tau_d_ms") != 12000.0:
            bad.append("%s stp_tau_d_ms %s" % (t["label"], h.get("stp_tau_d_ms")))
        p = ts.get("phasic_da")
        if _p_on(tag):
            if not (isinstance(p, dict) and ts.get("phasic_da_flag") is True and ts.get("phasic_da_applied") is True):
                bad.append("%s phasic read not applied" % t["label"])
            elif not lesion_p and (p.get("salience") or 0) > 0 and not (p.get("reward_us_spikes") or 0) > 0:
                bad.append("%s salience %s with reward_us_spikes %s" % (t["label"], p.get("salience"),
                                                                       p.get("reward_us_spikes")))
            if const and not (isinstance(p, dict) and p.get("const_salience") is True):
                bad.append("%s const_salience not set" % t["label"])
        elif isinstance(p, dict) or ts.get("phasic_da_flag"):
            bad.append("%s phasic read present with P off" % t["label"])
    if no_trace:
        why.append("engagement: abstained probe(s) without an applied topic-stop trace %s" % ",".join(no_trace[:6]))
    if bad:
        why.append("engagement: %s" % "; ".join(bad[:6]))
    return why


def cell_problems(cell, seed, tag, pin, mode):
    """Why one cell cannot be scored (empty list = DEFINED). The first entry of a coverage gap says `missing`."""
    if cell.get("missing"):
        return [cell["missing"]]
    if cell.get("load_error"):
        return [cell["load_error"]]
    arm = TAGS[tag][0]
    sh, pv = cell["shard"], cell["prov"]
    why = []
    if not isinstance(pv, dict):
        why.append("instrument: no provenance sidecar")
    else:
        if pin and pv.get("git_sha") != pin:
            why.append("instrument: sidecar git_sha %s is not the pin %s" % (pv.get("git_sha"), pin))
        if pv.get("git_dirty") is not False:
            why.append("instrument: sidecar git_dirty %s" % pv.get("git_dirty"))
        av = [str(a) for a in (pv.get("argv") or [])]

        def _after(flag):
            return av[av.index(flag) + 1] if flag in av and av.index(flag) + 1 < len(av) else None
        if _after("--arm") != arm or _after("--seed") != str(seed) or _after("--group") != "ask":
            why.append("instrument: sidecar argv arm/seed/group %s/%s/%s" % (_after("--arm"), _after("--seed"),
                                                                            _after("--group")))
        if mode == "gate":
            if pv.get("source_kind") != "git_archive" or pv.get("source_manifest_verified_at_start") is not True \
                    or pv.get("source_manifest_verified_at_exit") is not True:
                why.append("instrument: sidecar is not a verified git_archive measurement (%s, start %s, exit %s)"
                           % (pv.get("source_kind"), pv.get("source_manifest_verified_at_start"),
                              pv.get("source_manifest_verified_at_exit")))
            if "--allow-gate-seed" not in av or _after("--prereg") != PREREG:
                why.append("instrument: sidecar argv lacks --allow-gate-seed --prereg %s" % PREREG)
        got = {k: v for k, v in (pv.get("env") or {}).items() if k.startswith("BRAIN_CURIOSITY")}
        if got != env_of(tag):
            why.append("instrument: sidecar env %s is not the arm's %s" % (got, env_of(tag)))
    if not isinstance(sh, dict):
        return why + ["instrument: shard is not an object"]
    if sh.get("arm") != arm or sh.get("seed") != seed or sh.get("group") != "ask":
        why.append("instrument: shard arm/seed/group %s/%s/%s" % (sh.get("arm"), sh.get("seed"), sh.get("group")))
    why += ["instrument: %s" % w for w in G.arm_validity("ask", arm, sh)]
    if "responses" in sh:
        why += ["instrument: %s" % w for w in engagement_problems(sh, tag)]
    return why


def judge_cells(cells, seeds, pin, mode):
    """{(seed, tag): problems list} for every cell."""
    return {(s, tag): cell_problems(cells.get((s, tag)) or {"missing": "shard missing"}, s, tag, pin, mode)
            for s in seeds for tag in TAGS}


# ══════════════════════════════════════════════ per-seed reads ═════════════════════════════════════════════════════
def _vec(sh):
    return {l: G.asked((sh.get("responses") or {}).get(l)) for l in LAB12}


def _vecstr(sh):
    v = _vec(sh)
    return "".join(_SYM[v[l]] for l in LAB12)


def _ab_svo(sh, label):
    p = G._primary((sh.get("responses") or {}).get(label))
    return None if p is None else (p["abstained"], p["recalled_svo"])


def _undef(tags, probs):
    """A short reason naming the undefined cells a read depends on (coverage reads say `missing`)."""
    bad = [t for t in tags if probs.get(t)]
    return "; ".join("%s %s" % (t, probs[t][0]) for t in bad) if bad else None


def score_seed(cells, probs, s):
    """Every bar's read on one seed: {bar: {"ok": True/False/None, ...}} plus the informational reads."""
    pr = {tag: probs[(s, tag)] for tag in TAGS}

    def get(tag):
        return None if pr[tag] else cells[(s, tag)]["shard"]
    ia, ib, rv, pm = get("C_intact_a"), get("C_intact_b"), get("C_reversed"), get("C_permuted")
    row = {"seed": s}
    # 1 / S2C3
    d1, s2 = {}, {}
    for tag, sh in (("C_intact_a", ia), ("C_reversed", rv)):
        if sh is None:
            d1[tag] = {"ok": None, "why": _undef([tag], pr)}
            s2[tag] = None
            continue
        sc = G.score_ask(sh)
        d1[tag] = {"ok": True if sc["all"] else (None if sc["moat_undefined_decides"] else False),
                   "asked": "".join(_SYM[sc["asked_by_label"][l]] for l in LAB12),
                   **{k: sc[k] for k in ("ask_rate_gap", "ask_rate_moat", "discriminates", "lp_learnable",
                                         "lp_unlearnable", "all_gaps_abstain", "builtin_dog_cat", "moats_undefined")}}
        if not sc["all"] and sc["moat_undefined_decides"]:
            d1[tag]["why"] = "moat rate undefined decides (no taught moat acquired)"
        s2[tag] = sc["all"]
    oks = [d1[t]["ok"] for t in d1]
    row["1"] = {"ok": False if False in oks else (None if None in oks else True), "arms": d1}
    # S2C3: the Stage-2 C3 clause is the same `score_ask(...)["all"]` call on the same two arms, so it reads as 1 does
    row["S2C3"] = {"ok": row["1"]["ok"], "all": s2}
    # 2
    if ia is None or ib is None:
        row["2"] = {"ok": None, "why": _undef(["C_intact_a", "C_intact_b"], pr)}
    else:
        va, vb = _vec(ia), _vec(ib)
        if any(v is None for v in list(va.values()) + list(vb.values())):
            row["2"] = {"ok": None, "why": "instrument: asked() undefined on a label"}
        else:
            row["2"] = {"ok": va == vb, "diff": [l for l in LAB12 if va[l] != vb[l]],
                        "responses_identical": (json.dumps(ia.get("responses"), sort_keys=True)
                                                == json.dumps(ib.get("responses"), sort_keys=True))}
    # 3
    if pm is None:
        row["3"] = {"ok": None, "why": _undef(["C_permuted"], pr)}
    else:
        sp = G.score_ask(pm, permuted=True)
        row["3"] = {"ok": sp["discriminates"], "ask_rate_gap": sp["ask_rate_gap"], "ask_rate_moat": sp["ask_rate_moat"],
                    "moats_undefined": sp["moats_undefined"],
                    "asked": "".join(_SYM[sp["asked_by_label"][l]] for l in ("m_yak", "m_moose", "m_goat", "g_eel",
                                                                            "g_crab", "g_newt"))}
        if sp["discriminates"] is None:
            row["3"]["why"] = "moat rate undefined decides (no taught moat acquired)"
    # 4
    d4 = {}
    for order in ("intact_a", "reversed"):
        L, Cc = get("lesionL_" + order), get("C_" + order)
        if L is None or Cc is None:
            d4[order] = {"ok": None, "why": _undef(["lesionL_" + order, "C_" + order], pr)}
            continue
        al, ac = G.asked(L["responses"].get("u_toad_4")), G.asked(Cc["responses"].get("u_toad_4"))
        tl = [G.asked(L["responses"].get(t)) for t in TOADS]
        tc = [G.asked(Cc["responses"].get(t)) for t in TOADS]
        d4[order] = {"ok": None if al is None or ac is None else (al is True and al != ac),
                     "lesion_toads": "".join(_SYM[x] for x in tl), "C_toads": "".join(_SYM[x] for x in tc),
                     "extra_toad_asks": sum(1 for a, b in zip(tl, tc) if a is True and b is False)}
    if ia is not None and ib is not None:
        d4["null_toad_diffs"] = sum(1 for t in TOADS if G.asked(ia["responses"].get(t)) != G.asked(ib["responses"].get(t)))
    else:
        d4["null_toad_diffs"] = None
    o4 = [d4[o]["ok"] for o in ("intact_a", "reversed")]
    row["4"] = {"ok": False if False in o4 else (None if None in o4 else True), **d4}
    # 5
    LP = get("lesionP_intact_a")
    if LP is None:
        row["5"] = {"ok": None, "why": _undef(["lesionP_intact_a"], pr)}
    else:
        sl = G.score_ask(LP, moat_acq=(G.acquired_teaches(ia) if ia is not None else None))
        row["5"] = {"ok": sl["lp_learnable"] is False and sl["lp_unlearnable"] is False,
                    "lp_learnable": sl["lp_learnable"], "lp_unlearnable": sl["lp_unlearnable"],
                    "ask_rate_gap": sl["ask_rate_gap"], "asked": "".join(_SYM[sl["asked_by_label"][l]] for l in LAB12)}
    # 6
    LC = get("C_lesion_curiosity")
    if LC is None:
        row["6"] = {"ok": None, "why": _undef(["C_lesion_curiosity"], pr)}
    else:
        a = {t["label"]: G.asked(LC["responses"].get(t["label"])) for t in LC["turns"]}
        row["6"] = {"ok": None if any(v is None for v in a.values()) else not any(a.values()),
                    "asked_on": [l for l, v in a.items() if v]}
    # 7
    d7 = {}
    for on, off in D7_PAIRS:
        A, B = get(on), get(off)
        if A is None or B is None:
            d7[on] = {"ok": None, "why": _undef([on, off], pr)}
            continue
        diffs, undef = [], False
        for t in A["turns"]:
            x, y = _ab_svo(A, t["label"]), _ab_svo(B, t["label"])
            if x is None or y is None:
                undef = True
            elif x != y:
                diffs.append(t["label"])
        d7[on] = {"ok": None if undef else not diffs, "n_turns": len(A["turns"]), "differs_on": diffs}
    o7 = [d7[k]["ok"] for k in d7]
    row["7"] = {"ok": False if False in o7 else (None if None in o7 else True), "pairs": d7}
    # 8
    d8 = {}
    for tag, key, want in READBACKS:
        sh = get(tag)
        if sh is None:
            d8[tag] = {"ok": None, "why": _undef([tag], pr)}
            continue
        n, bad, no_ts = 0, [], []
        for t in sh["turns"]:
            rr = sh["responses"].get(t["label"])
            ts = ts_of(rr)
            if t["role"] == "probe" and G._abstained(rr) and not isinstance(ts, dict):
                no_ts.append(t["label"])
            if not isinstance(ts, dict) or not ts.get("applied"):
                continue
            n += 1
            if key == "tau":
                v = (ts.get("held") or {}).get("stp_tau_d_ms")
            elif key == "us":
                v = (ts.get("phasic_da") or {}).get("reward_us_spikes")
            else:
                v = (ts.get("held") or {}).get("topic_ask_weight_sum")
            if v is None or float(v) != float(want):
                bad.append((t["label"], v))
        d8[tag] = {"ok": n > 0 and not bad and not no_ts, "n_applied": n, "not_held": bad[:6],
                   "abstained_probes_without_trace": no_ts}
    o8 = [d8[k]["ok"] for k in d8]
    row["8"] = {"ok": False if False in o8 else (None if None in o8 else True), "arms": d8}
    # AR (G10)
    dar = {}
    for on, off in AR_PAIRS:
        A, B = get(on), get(off)
        if A is None or B is None:
            dar[on] = {"ok": None, "why": _undef([on, off], pr)}
            continue
        labels = [t["label"] for t in A["turns"]]
        lost = [l for l in labels if classify(B["responses"].get(l)) == "answered"
                and classify(A["responses"].get(l)) != "answered"]
        ra = rate([A["responses"].get(l) for l in labels])
        rb = rate([B["responses"].get(l) for l in labels])
        dar[on] = {"ok": (not lost) and ra["answered"] >= rb["answered"], "on_answered": ra["answered"],
                   "off_answered": rb["answered"], "n": ra["n"], "answers_lost_on": lost}
    oar = [dar[k]["ok"] for k in dar]
    row["AR"] = {"ok": False if False in oar else (None if None in oar else True), "pairs": dar}
    row["info"] = info_seed(get, ia)
    return row


def info_seed(get, ia):
    """Reported, never barred: the informational arms, null_count, toads, the flag-off failure, the host threshold."""
    info = {}
    for tag in ("C_intact_a", "C_reversed", "Lpinned_intact_a", "Lpinned_reversed", "pconst_intact_a",
                "pconst_reversed", "off_intact_a", "off_reversed", "lesionP_intact_a", "lesionP_reversed",
                "lesionL_intact_a", "lesionL_reversed"):
        sh = get(tag)
        if sh is None:
            info[tag] = None
            continue
        sc = G.score_ask(sh)
        rec = {"all": sc["all"], "asked": "".join(_SYM[sc["asked_by_label"][l]] for l in LAB12),
               "discriminates": sc["discriminates"], "lp_learnable": sc["lp_learnable"],
               "lp_unlearnable": sc["lp_unlearnable"], "gap": sc["ask_rate_gap"], "moat": sc["ask_rate_moat"],
               "toads": "".join(_SYM[G.asked(sh["responses"].get(t))] for t in TOADS)}
        order = tag.split("_", 1)[1]
        ref = get("C_" + order) if order in ("intact_a", "reversed") else None
        if ref is not None and not tag.startswith("C_"):
            rec["equals_C_vector"] = _vecstr(sh) == _vecstr(ref)
        if tag.startswith("Lpinned"):
            rec["da_lesion_held"] = G.da_lesion_held(sh)["status"]
        if tag.startswith("pconst"):
            rec["const_salience_probes"] = sum(1 for t in sh["turns"] if ((ts_of(sh["responses"].get(t["label"])) or {})
                                                                         .get("phasic_da") or {}).get("const_salience") is True)
        if tag.startswith(("C_", "lesionP")):
            hits, n, das = 0, 0, []
            for t in sh["turns"]:
                ts = ts_of(sh["responses"].get(t["label"]))
                if not (isinstance(ts, dict) and ts.get("applied")):
                    continue
                n += 1
                want, thr = ts.get("want_decision_hz"), ts.get("threshold")
                if want is not None and thr is not None:
                    hits += (float(want) >= float(thr)) == bool(G.asked(sh["responses"].get(t["label"])))
                das.append((ts.get("tonic_da") or {}).get("da_level"))
            rec["host_threshold_agrees"] = [hits, n]       # asked == (want_decision_hz >= threshold), per applied probe
            rec["tonic_da_first_last"] = [das[0], das[-1]] if das else None
        info[tag] = rec
    if ia is not None:                                   # null_count, the packet's host count rule (k = 1, 2)
        seen, nc = {}, {1: {}, 2: {}}
        for t in ia["turns"]:
            rr = ia["responses"].get(t["label"]) or {}
            c = rr.get("curiosity") or {}
            if not c or not rr.get("abstained"):
                continue
            tp = c.get("topic")
            cur = (c.get("want_hz") or 0) >= (c.get("threshold") or 1e9)
            for k in (1, 2):
                nc[k][t["label"]] = bool(cur and seen.get(tp, 0) < k)
            seen[tp] = seen.get(tp, 0) + 1
        av = {t["label"]: G.asked(ia["responses"].get(t["label"])) for t in ia["turns"]}
        info["null_count"] = {str(k): [sum(1 for l in nc[k] if nc[k][l] == av[l]), len(nc[k])] for k in (1, 2)}
    return info


# ══════════════════════════════════════════════ tallies + verdict ══════════════════════════════════════════════════
def tally(rows, seeds, bar, need):
    vals = {s: rows[s][bar]["ok"] for s in seeds}
    n_true = sum(1 for v in vals.values() if v is True)
    n_false = sum(1 for v in vals.values() if v is False)
    n_undef = sum(1 for v in vals.values() if v is None)
    verdict = "PASS" if n_true >= need else ("FAIL" if n_true + n_undef < need else "UNDEFINED")
    return {"verdict": verdict, "pass": n_true, "fail": n_false, "undefined": n_undef, "need": need, "of": len(seeds),
            "fail_seeds": [s for s in seeds if vals[s] is False], "undefined_seeds": [s for s in seeds if vals[s] is None]}


def attribution(rows, seeds):
    tr = [rows[s]["4"]["intact_a"].get("extra_toad_asks") for s in seeds]
    ct = [rows[s]["4"].get("null_toad_diffs") for s in seeds]
    trr = [rows[s]["4"]["reversed"].get("extra_toad_asks") for s in seeds]
    out = {"treatment_intact_a": sum(x for x in tr if x is not None), "control_null_toad_diffs":
           sum(x for x in ct if x is not None), "treatment_reversed_reported": sum(x for x in trr if x is not None)}
    if any(x is None for x in tr + ct):
        out.update(fraction=None, verdict="UNDEFINED", why="a seed's lesionL_intact_a / C_intact_a / C_intact_b cell missing or undefined")
        return out
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        f = attributable_to("lesion_L extra toad asks vs the intact_b null", float(sum(tr)), float(sum(ct)))
    out.update(fraction=f, log=buf.getvalue().strip(),
               verdict="UNDEFINED" if f is None else ("PASS" if f >= 0.5 else "FAIL"))
    return out


def score(cells, seeds, mode, pin, counted=None, needs=None):
    """The whole read: cell validity, per-seed rows, bar tallies, the D4/G4 attribution, DEC."""
    seeds = tuple(int(s) for s in seeds)
    counted = tuple(counted) if counted is not None else seeds
    probs = judge_cells(cells, seeds, pin, mode)
    rows = {s: score_seed(cells, probs, s) for s in seeds}
    needs = dict(needs or {})
    bars = {b: tally(rows, counted, b, needs.get(b, len(counted))) for b in BARS}
    att = attribution(rows, counted)
    v4 = [bars["4"]["verdict"], att["verdict"]]
    bars["4"]["toad_verdict"] = bars["4"]["verdict"]
    bars["4"]["attribution"] = att
    bars["4"]["verdict"] = "FAIL" if "FAIL" in v4 else ("UNDEFINED" if "UNDEFINED" in v4 else "PASS")
    decided = PACKET_BARS if mode == "dev" else BARS
    vs = [bars[b]["verdict"] for b in decided]
    dec = "PASS" if all(v == "PASS" for v in vs) else ("FAIL" if "FAIL" in vs else "UNDEFINED")
    if mode == "dev":
        rule = {"PASS": "every dev bar holds: the method goes to a gate-seed preregistration (packet section 7)",
                "FAIL": "a dev bar fails: no gate-seed preregistration", "UNDEFINED": "a dev bar is undefined"}[dec]
    else:
        rule = {"PASS": "L+P qualifies for the default-ON flip (a separate, later flip prereg/commit) before the "
                        "11-21 Stage-2 re-pin",
                "FAIL": "STOP with the data: no flip, no retuning on gate seeds; the verdict is on the method",
                "UNDEFINED": "re-run each undefined cell once at the same revision (logged before reading); a cell still "
                             "UNDEFINED never counts, and the gate then reads STOP with the data"}[dec]
    return {"mode": mode, "pin": pin, "seeds": list(seeds), "counted": list(counted), "decided_bars": list(decided),
            "cells": {"%d/%s" % k: (v or "DEFINED") for k, v in sorted(probs.items())},
            "n_cells": len(probs), "n_defined": sum(1 for v in probs.values() if not v),
            "bars": bars, "DEC": {"verdict": dec, "reads": rule}, "rows": {str(s): rows[s] for s in seeds}}


def render(res):
    pre = "D" if res["mode"] == "dev" else "G"
    name = {b: (pre + b if b in PACKET_BARS else {"S2C3": "G9 (S2-C3)", "AR": "G10 (AR)"}[b]) for b in BARS}
    out = ["C3 topic-stop %s read, pin %s, seeds %s (counted %s)" % (res["mode"], res["pin"], res["seeds"], res["counted"]),
           "cells %d, DEFINED %d" % (res["n_cells"], res["n_defined"])]
    for k, v in res["cells"].items():
        out.append("  cell %s %s" % (k, "DEFINED" if v == "DEFINED" else "UNDEFINED (%s)" % "; ".join(v)[:300]))
    for b in BARS:
        t = res["bars"][b]
        tag = "" if b in res["decided_bars"] else " [reported]"
        out.append("%s %s %d/%d need %d%s%s%s" % (name[b], t["verdict"], t["pass"], t["of"], t["need"], tag,
                                                 " fail %s" % t["fail_seeds"] if t["fail_seeds"] else "",
                                                 " undefined %s" % t["undefined_seeds"] if t["undefined_seeds"] else ""))
        if b == "4":
            a = t["attribution"]
            out.append("   attribution %s fraction %s (treatment %s, control %s; reversed treatment %s, reported)"
                       % (a["verdict"], a.get("fraction"), a["treatment_intact_a"], a["control_null_toad_diffs"],
                          a["treatment_reversed_reported"]))
    for s, r in res["rows"].items():
        bits = []
        for b in BARS:
            ok = r[b]["ok"]
            bits.append("%s=%s" % (name[b].split()[0], {True: "PASS", False: "FAIL", None: "UNDEFINED"}[ok]))
        out.append("  seed %s: %s" % (s, " ".join(bits)))
        for b in BARS:
            if r[b]["ok"] is None:
                whys = [r[b].get("why")] + [x.get("why") for x in (r[b].get("arms") or r[b].get("pairs") or {}).values()
                                            if isinstance(x, dict)]
                whys += [r[b][o].get("why") for o in ("intact_a", "reversed") if isinstance(r[b].get(o), dict)]
                whys = [w for w in whys if w]
                out.append("    %s UNDEFINED (%s)" % (name[b].split()[0], "; ".join(whys)[:300] or "no reason recorded"))
        i = r["info"]
        out.append("    vectors C_a %s C_r %s | Lpinned %s/%s | pconst %s/%s | off %s/%s | null_count %s" % (
            (i.get("C_intact_a") or {}).get("asked"), (i.get("C_reversed") or {}).get("asked"),
            (i.get("Lpinned_intact_a") or {}).get("all"), (i.get("Lpinned_reversed") or {}).get("all"),
            (i.get("pconst_intact_a") or {}).get("all"), (i.get("pconst_reversed") or {}).get("all"),
            (i.get("off_intact_a") or {}).get("all"), (i.get("off_reversed") or {}).get("all"), i.get("null_count")))
    out.append("DEC %s: %s" % (res["DEC"]["verdict"], res["DEC"]["reads"]))
    return "\n".join(out)


# ══════════════════════════════════════════════ pool lines ═════════════════════════════════════════════════════════
def emit_lines(sha="<sha>", seeds=GATE_SEEDS, out_dir=GATE_DIR, prereg=PREREG):
    """[(command, reason)]: the dev battery's lines byte-for-byte (the 2026-10-27 c3-battery queue, prefix and runner
    invocation) except the seed, the --out directory and the gate-seed pair `--allow-gate-seed --prereg <prereg>` the
    runner requires on a gate seed (stage2_exit_gate.main refuses a gate seed without it)."""
    lines = []
    for s in seeds:
        d = "%s/s%d" % (out_dir, s)
        for tag, (arm, env, kind) in TAGS.items():
            envs = " ".join("%s=%s" % kv for kv in env)
            cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && bash tools/memcap.sh %d "
                   "-- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                   "NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= %s .venv/bin/python -u -m research.runners.stage2_exit_gate "
                   "--shard --seed %d --group ask --arm %s --allow-gate-seed --prereg %s --out %s/ask__%s.json"
                   % (sha, MEM_GB, d, MEMCAP_GB, envs, s, arm, prereg, d, tag))
            why = ("prereg %s (F %s); C3 topic-stop gate (L+P = BRAIN_CURIOSITY_TOPIC_STOP=1 BRAIN_CURIOSITY_PHASIC_DA=1) "
                   "gate seed %d arm %s (%s), the arms of packet %s section 7; record: the dev battery passed D1-D8 on "
                   "11/11 counted dev seeds at 79c6f5128 (research/findings/raw/_c3_topic_stop/dev/, 192 shards, "
                   "skeptic re-score agreed); no gate seed has been run for this method; mem_gb=%d max_age_h=%d"
                   % (prereg, sha, s, tag, kind, PACKET, MEM_GB, MAX_AGE_H))
            lines.append((cmd, why))
    return lines


# ══════════════════════════════════════════════ import closure ═════════════════════════════════════════════════════
def _git(*args):
    return subprocess.run(["git", "-C", _REPO] + list(args), capture_output=True, text=True, check=True).stdout


def _source(path, rev):
    if rev is None:
        p = os.path.join(_REPO, path)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return f.read()
    try:
        return _git("show", "%s:%s" % (rev, path))
    except subprocess.CalledProcessError:
        return None


def _module_file(mod, exists):
    base = mod.replace(".", "/")
    for cand in (base + ".py", base + "/__init__.py"):
        if exists(cand):
            return cand
    return None


def import_closure(rev=None, roots=CLOSURE_ROOTS):
    """Repo files reachable from `roots` through ANY import statement (module level or inside a function: the server
    imports most organs lazily) and `importlib.import_module("<literal>")`, statically; package __init__ files of every
    imported module included. A superset of what one arm process loads; the gate's identity check reads it."""
    if rev is None:
        def exists(p):
            return os.path.isfile(os.path.join(_REPO, p))
    else:
        tree = set(_git("ls-tree", "-r", "--name-only", rev).split("\n"))

        def exists(p):
            return p in tree
    seen, todo = set(), [r for r in roots if exists(r)]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        src = _source(path, rev)
        if src is None:
            continue
        try:
            tree_ = ast.parse(src)
        except SyntaxError:
            continue
        pkg = os.path.dirname(path).replace("/", ".")
        mods = set()
        for node in ast.walk(tree_):
            if isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = pkg.split(".") if pkg else []
                    parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = ".".join(parts + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                mods.add(base)
                mods.update("%s.%s" % (base, a.name) for a in node.names if a.name != "*")
            elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "import_module" and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                mods.add(node.args[0].value)
        for m in mods:
            parts = m.split(".")
            for i in range(1, len(parts) + 1):
                f = _module_file(".".join(parts[:i]), exists)
                if f and f not in seen:
                    todo.append(f)
    return sorted(seen)


def closure_check(p_rev, f_rev):
    """Files of the union closure (at P and at F) whose blob differs between P and F (absent counts as a difference)."""
    files = sorted(set(import_closure(p_rev)) | set(import_closure(f_rev)))
    diffs = []
    for path in files:
        def blob(rev):
            try:
                return _git("rev-parse", "%s:%s" % (rev, path)).strip()
            except subprocess.CalledProcessError:
                return None
        if blob(p_rev) != blob(f_rev):
            diffs.append(path)
    return files, diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("dev", "gate"), default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated (dev default 1-12, gate default the six gate seeds)")
    ap.add_argument("--pin", default=None, help="the full sha every sidecar must name (gate: required; dev: 79c6f5128...)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-lines", nargs="?", const="<sha>", default=None, metavar="SHA")
    ap.add_argument("--line-seeds", default=None, help="--emit-lines: seeds (default the six gate seeds; the pre-flight "
                                                       "read uses dev seed 7)")
    ap.add_argument("--line-dir", default=GATE_DIR, help="--emit-lines: output root (pre-flight: %s)" % PREFLIGHT_DIR)
    ap.add_argument("--closure", action="store_true")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--closure-check", nargs=2, default=None, metavar=("P", "F"))
    a = ap.parse_args(argv)
    if a.emit_lines is not None:
        ls = tuple(int(x) for x in a.line_seeds.split(",")) if a.line_seeds else GATE_SEEDS
        for cmd, why in emit_lines(a.emit_lines, seeds=ls, out_dir=a.line_dir):
            print("%s\t%s" % (cmd, why))
        return 0
    if a.closure:
        for f in import_closure(a.rev):
            print(f)
        return 0
    if a.closure_check:
        files, diffs = closure_check(*a.closure_check)
        print("closure %d files; %s" % (len(files), "IDENTICAL" if not diffs else "DIFFERS on %d: %s" % (len(diffs),
                                                                                                    " ".join(diffs))))
        return 0 if not diffs else 1
    if a.mode is None:
        ap.error("choose --mode dev|gate, --emit-lines, --closure or --closure-check")
    if a.mode == "gate" and not (a.pin and len(a.pin) == 40):
        ap.error("--mode gate needs --pin <F, the full 40-hex sha the lines ran at>")
    root = a.root or os.path.join(_REPO, DEV_DIR if a.mode == "dev" else GATE_DIR)
    if a.mode == "dev":
        seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else DEV_SEEDS
        counted = tuple(s for s in seeds if s != DESIGN_SEED)
        needs = {"5": 9} if counted == DEV_COUNTED else {}
        res = score(load_cells(root, seeds), seeds, "dev", a.pin or DEV_PIN, counted=counted, needs=needs)
    else:
        seeds = tuple(int(x) for x in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        res = score(load_cells(root, seeds), seeds, "gate", a.pin)
    print(render(res))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        print("-> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
