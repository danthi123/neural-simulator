"""C2 match-unit smoke (RULING c2-drain-wta-1101): the arm table, the pool lines, and the scorer with bars FIXED before any
arm ran.

ARMS (each = the 37 step-0 lines of research/coordination/c2-step0-pool-lines.txt, dev seeds 1-7, with these env flags):
  A0  BRAIN_C2_GATE_RESET=1 BRAIN_C2_COMPOSER_PATIENT_INHIB=1 (the gate-reset wave's flags; every new flag unset)
  A1  A0 + BRAIN_C2_QUIET_DRAIN=60                     (the physical inter-query quiet period)
  A2w A1 + BRAIN_C2_ROLE_WTA=1                         (agent/action rows through the NEF-FS lateral-inhibition WTA)
  A2r A1 + BRAIN_C2_ROLE_RELAY=1                       (agent/action rows through the patient-relay inhibition bridge)
  L1  A2r + BRAIN_C2_ROLE_RELAY_I2E=0                  (lesion: the relay's I->E zeroed)
  L1w A2w + BRAIN_C2_ROLE_WTA_I2E=0                    (lesion: the WTA's I->E zeroed)
  L3  A2r + BRAIN_C2_ROW_SHUFFLE=1                     (lesion: block b fed another block's agent/action row)

FROZEN DENOMINATORS. Every (shard, cue, block) pair is classified ONCE, on the reference wave
research/findings/raw/_c2_step0_gatereset (classification of _scratch/night1101/_c2_match_s0b.py: true = the block is a
true match of the cue; else both-lit / one-lit / neither by whether the cue's agent is lit in the block's agent row and the
cue's action in its action row, read from the REFERENCE drives). The counts must be true 98, both-lit 86, one-lit 501,
neither 575, role reads 630, cues 140, else every arm is UNDEFINED. A pair "fires" iff the ARM's m{b} > match_thresh.

BARS (fixed now):
  A0  m_rates identical to _c2_step0_gatereset in 140/140 cues.
  A1  one-lit fires <= 2/501; both-lit fires == 86/86.
  A2w, A2r  B1 true fires >= 96/98; B2 both-lit fires <= 5/86; B3 one-lit fires <= 2/501 and neither fires <= 2/575;
      B4 the role drive lights exactly the decoded agent (agent row) / action (action row) in >= 600/630 role reads;
      B5 read-(c) patient lit sets identical to A0 in every block; B6 every true block of every AMBIGUOUS_CUES cue fires.
  L1  both-lit fires >= 60/86.   L1w  report only (no bar fixed by the ruling).   L3  true fires <= 20/98.
  Report only: inhibitory spikes vs ceiling; accA/accX separability AUC (decoded word vs other lit words).
An arm with a missing shard, an unreadable or non-engaged record is UNDEFINED (never PASS or FAIL).

  .venv/bin/python -m research.runners.c2_drain_wta_score --score research/findings/raw/_c2_drain_wta [--out X.json]
  .venv/bin/python -m research.runners.c2_drain_wta_score --emit-pool-lines <40-hex sha> > research/coordination/c2-drain-wta-pool-lines.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RULING = "c2-drain-wta-1101"
OUT_ROOT = "research/findings/raw/_c2_drain_wta"
REF_ROOT = "research/findings/raw/_c2_step0_gatereset"
BASE_LINES = "research/coordination/c2-step0-pool-lines.txt"
BASE_OUT_ROOT = "research/findings/raw/_c2_step0"
SEEDS = (1, 2, 3, 4, 5, 6, 7)
MEM_GB, MAX_AGE_H = 9, 240                         # the base step-0 lines' sizing (c2_hops_step0 MEM_GB / MAX_AGE_H)
CUE_ORDER = ("heron,hunt", "heron,eat", "frog,eat", "frog,chase")
FROZEN = {"true": 98, "both": 86, "one": 501, "neither": 575, "role_reads": 630, "cues": 140}
BASE_ENV = (("BRAIN_C2_GATE_RESET", "1"), ("BRAIN_C2_COMPOSER_PATIENT_INHIB", "1"))
ARMS = {
    "A0": BASE_ENV,
    "A1": BASE_ENV + (("BRAIN_C2_QUIET_DRAIN", "60"),),
    "A2w": BASE_ENV + (("BRAIN_C2_QUIET_DRAIN", "60"), ("BRAIN_C2_ROLE_WTA", "1")),
    "A2r": BASE_ENV + (("BRAIN_C2_QUIET_DRAIN", "60"), ("BRAIN_C2_ROLE_RELAY", "1")),
    "L1": BASE_ENV + (("BRAIN_C2_QUIET_DRAIN", "60"), ("BRAIN_C2_ROLE_RELAY", "1"), ("BRAIN_C2_ROLE_RELAY_I2E", "0")),
    "L1w": BASE_ENV + (("BRAIN_C2_QUIET_DRAIN", "60"), ("BRAIN_C2_ROLE_WTA", "1"), ("BRAIN_C2_ROLE_WTA_I2E", "0")),
    "L3": BASE_ENV + (("BRAIN_C2_QUIET_DRAIN", "60"), ("BRAIN_C2_ROLE_RELAY", "1"), ("BRAIN_C2_ROW_SHUFFLE", "1")),
}
# the role-drive engagement each arm must show (mode, I->E weight); None = the unchanged divnorm relay
ARM_ROLE = {"A0": ("divnorm", None), "A1": ("divnorm", None), "A2w": ("wta", 40.0), "A2r": ("relay", 2.0),
            "L1": ("relay", 0.0), "L1w": ("wta", 0.0), "L3": ("relay", 2.0)}
ALL_FLAGS = ("BRAIN_C2_QUIET_DRAIN", "BRAIN_C2_ROLE_WTA", "BRAIN_C2_ROLE_WTA_I2E", "BRAIN_C2_ROLE_RELAY",
             "BRAIN_C2_ROLE_RELAY_I2E", "BRAIN_C2_ROW_SHUFFLE", "BRAIN_C2_GATE_RESET", "BRAIN_C2_COMPOSER_PATIENT_INHIB")


def _tables():
    from research.runners import c2_hops_step0 as S0
    return dict(S0.TABLE_ARMS)


# ═════════════════════════════════════════════════ POOL LINES ═══════════════════════════════════════════════════════
def pool_lines(sha, arms=tuple(ARMS), base_path=None):
    """(cmd, reason) per (arm, base line): the 37 step-0 lines, revision-pinned to `sha`, the arm's env flags inserted
    right after the line's `env `, the out root moved to OUT_ROOT/<ARM>/."""
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", sha or ""):
        raise ValueError("pin a full 40-hex sha, got %r" % sha)
    with open(base_path or os.path.join(_REPO, BASE_LINES)) as f:
        base = [ln.rstrip("\n").split("\t")[0] for ln in f if ln.strip()]
    if len(base) != 37:
        raise ValueError("expected the 37 step-0 lines, read %d" % len(base))
    out = []
    for arm in arms:
        env = " ".join("%s=%s" % kv for kv in ARMS[arm])
        for cmd in base:
            if cmd.count("~/derisk-pool/revisions/<sha>") != 1 or cmd.count(" env ") != 1 \
                    or cmd.count(BASE_OUT_ROOT + "/") != 2:
                raise ValueError("unexpected base line shape: %s" % cmd[:200])
            c = cmd.replace("~/derisk-pool/revisions/<sha>", "~/derisk-pool/revisions/%s" % sha)
            c = c.replace(" env ", " env %s " % env)
            c = c.replace(BASE_OUT_ROOT + "/", "%s/%s/" % (OUT_ROOT, arm))
            # the pool reads the job size from `mem_gb=N` anywhere in the line (tools/pool_autodispatch.sh), as the base lines
            out.append((c, "RULING %s: C2 match-unit smoke arm %s (dev seeds 1-7); mem_gb=%d max_age_h=%d"
                        % (RULING, arm, MEM_GB, MAX_AGE_H)))
    return out


# ═══════════════════════════════════════════════════ LOADING ════════════════════════════════════════════════════════
def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": repr(e)}


def _reads(shard):
    return ((shard or {}).get("c2_hops") or {}).get("reads") or {}


def load_wave(root, seeds=SEEDS):
    """{(seed, table): shard or None} for the five step-0 tables of one wave root."""
    from research.runners import c2_hops_step0 as S0
    return {(s, t): _load(S0.shard_path(root, s, a)) for s in seeds for t, a in _tables().items()}


def classify_reference(ref):
    """The frozen classification: {(seed, table, cue, block): cls} plus the reference m_rates, decoded rows and K, from
    the reference wave. Returns (pairs, info, why) with `why` non-empty when the reference cannot be read."""
    pairs, info, why = {}, {}, []
    for (s, t), sh in sorted(ref.items()):
        r = _reads(sh)
        b = r.get("b") or {}
        if sh is None or not b.get("cues") or not b.get("drives_lit_words"):
            why.append("reference shard s%d %s missing or without read (b)" % (s, t))
            continue
        dlw = b["drives_lit_words"]
        info[(s, t)] = {"K": int(r["K"]), "rows": r.get("decoded_rows") or [], "thr": float(b["match_thresh"]),
                        "m": {ck: b["cues"][ck].get("m_rates") for ck in CUE_ORDER},
                        "c": [blk.get("fired_pools") for blk in ((r.get("c") or {}).get("blocks") or [])]}
        for ck in CUE_ORDER:
            c = b["cues"][ck]
            a, x = c["cue"]
            tm = set(c.get("true_matches") or [])
            for blk in range(int(r["K"])):
                if blk in tm:
                    cls = "true"
                else:
                    n = (a in set(dlw[blk][0])) + (x in set(dlw[blk][1]))
                    cls = {2: "both", 1: "one", 0: "neither"}[n]
                pairs[(s, t, ck, blk)] = cls
    return pairs, info, why


def frozen_counts(pairs, info):
    from collections import Counter
    c = Counter(pairs.values())
    return {"true": c["true"], "both": c["both"], "one": c["one"], "neither": c["neither"],
            "role_reads": sum(2 * v["K"] for v in info.values()), "cues": len(CUE_ORDER) * len(info)}


# ═════════════════════════════════════════════════ ENGAGEMENT ══════════════════════════════════════════════════════
def engagement(arm, shard, seed, table):
    """Why this arm's shard cannot be scored (empty = engaged and readable)."""
    from research.runners import c2_hops_step0 as S0
    why = list(S0.shard_validity(shard, seed, _tables()[table]))
    if why:
        return why
    r = _reads(shard)
    b = r.get("b") or {}
    if r.get("error") or b.get("error") or not b.get("cues"):
        return ["read (b) missing or errored"]
    rec = b.get("c2dw")
    if not isinstance(rec, dict):
        return ["no c2dw engagement record (pre-branch code?)"]
    want = dict(ARMS[arm])
    flags = rec.get("flags") or {}
    for n in ALL_FLAGS:
        if flags.get(n) != want.get(n):
            why.append("flag %s=%r, arm %s wants %r" % (n, flags.get(n), arm, want.get(n)))
    if not rec.get("accs_recorded"):
        why.append("accA/accX not recorded")
    ri = rec.get("role_info") or {}
    mode, w = ARM_ROLE[arm]
    if ri.get("mode") != mode:
        why.append("role drive mode %r, expected %r" % (ri.get("mode"), mode))
    if w is not None:
        if ri.get("w_i2e") != w:
            why.append("role I->E %r, expected %r" % (ri.get("w_i2e"), w))
        if not (ri.get("reads") or 0) > 0:
            why.append("no role reads recorded")
        if w > 0 and not (ri.get("inh_spikes") or 0) > 0:
            why.append("role inhibition never spiked (non-engaged)")
    perm = ri.get("row_shuffle_perm")
    if arm == "L3":
        K = int(r.get("K") or 0)
        if not perm or sorted(perm) != list(range(K)) or any(perm[i] == i for i in range(K)):
            why.append("row shuffle not engaged (perm %r)" % (perm,))
    elif perm is not None:
        why.append("row shuffle engaged in arm %s" % arm)
    qd = int(want["BRAIN_C2_QUIET_DRAIN"]) if "BRAIN_C2_QUIET_DRAIN" in want else 20
    for ck in CUE_ORDER:
        c = b["cues"].get(ck) or {}
        if c.get("m_rates") is None:
            why.append("cue %s has no m_rates" % ck)
        elif (c.get("drain") or {}).get("drain_steps") != qd:
            why.append("cue %s drain %r, expected %d steps" % (ck, (c.get("drain") or {}).get("drain_steps"), qd))
    return why


# ═══════════════════════════════════════════════════ COUNTS ═════════════════════════════════════════════════════════
def _auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return round(s / (len(pos) * len(neg)), 4)


def arm_counts(arm_wave, pairs, info):
    """Fire counts per frozen class, role-read exactness, read-(c) patient sets, ambiguous-cue fires, inhibition."""
    from research.runners import c2_hops_step0 as S0
    fires = {"true": 0, "both": 0, "one": 0, "neither": 0}
    exact = 0
    pos, neg = {"agent": [], "action": []}, {"agent": [], "action": []}
    inh, ceil = 0.0, 0.0
    m_same = 0
    pending = 0
    c_sets = {}
    amb = []
    for (s, t), sh in sorted(arm_wave.items()):
        r = _reads(sh)
        b = r["b"]
        thr = float(b["match_thresh"])
        for ck in CUE_ORDER:
            m = b["cues"][ck]["m_rates"]
            m_same += int(m == info[(s, t)]["m"][ck])
            pending += int((b["cues"][ck].get("drain") or {}).get("prev_firing_at_clear") or 0)
            for blk in range(len(m)):
                cls = pairs.get((s, t, ck, blk))
                if cls is not None and m[blk] > thr:
                    fires[cls] += 1
        for ck_t in S0.AMBIGUOUS_CUES.get(t, ()):
            ck = "%s,%s" % ck_t
            m = b["cues"][ck]["m_rates"]
            tm = [blk for blk in range(len(m)) if pairs.get((s, t, ck, blk)) == "true"]
            amb.append({"seed": s, "table": t, "cue": ck, "true_blocks": tm, "fired": [blk for blk in tm if m[blk] > thr]})
        rec = b["c2dw"]
        rows = info[(s, t)]["rows"]
        for blk, (aA, aX) in enumerate(rec["accs"]):
            row = rows[blk] if blk < len(rows) else {}
            for role, acc in (("agent", aA), ("action", aX)):
                want = row.get(role)
                exact += int(sorted(acc) == [want])
                for w, v in acc.items():
                    (pos if w == want else neg)[role].append(float(v))
        ri = rec.get("role_info") or {}
        inh += float(ri.get("inh_spikes") or 0.0)
        ceil += float(ri.get("inh_ceiling") or 0.0)
        c_sets[(s, t)] = [blk.get("fired_pools") for blk in ((r.get("c") or {}).get("blocks") or [])]
    return {"fires": fires, "role_reads_exact": exact, "m_rates_identical_to_ref": m_same,
            "ambiguous": amb, "ambiguous_all_fire": all(len(a["fired"]) == len(a["true_blocks"]) >= 2 for a in amb)
            and len(amb) > 0,
            "report_only": {"inh_spikes": inh, "inh_ceiling": ceil, "prev_firing_at_clear_sum": pending,
                            "inh_fraction_of_ceiling": round(inh / ceil, 4) if ceil else None,
                            "acc_auc_decoded_vs_other_lit": {k: _auc(pos[k], neg[k]) for k in pos},
                            "n_other_lit": {k: len(neg[k]) for k in neg}},
            "_c_sets": c_sets}


def _bar(name, ok, got, rule):
    return {"bar": name, "pass": bool(ok), "got": got, "rule": rule}


def bars(arm, cnt, a0_sets):
    f = cnt["fires"]
    if arm == "A0":
        return [_bar("A0_m_rates_identical", cnt["m_rates_identical_to_ref"] == 140, cnt["m_rates_identical_to_ref"],
                     "== 140/140 cues")]
    if arm == "A1":
        return [_bar("A1_one_lit", f["one"] <= 2, f["one"], "<= 2/501"),
                _bar("A1_both_lit", f["both"] == 86, f["both"], "== 86/86")]
    if arm in ("A2w", "A2r"):
        out = [_bar("B1_true", f["true"] >= 96, f["true"], ">= 96/98"),
               _bar("B2_both_lit", f["both"] <= 5, f["both"], "<= 5/86"),
               _bar("B3_one_lit_and_neither", f["one"] <= 2 and f["neither"] <= 2, [f["one"], f["neither"]],
                    "one <= 2/501 and neither <= 2/575"),
               _bar("B4_role_reads_exact", cnt["role_reads_exact"] >= 600, cnt["role_reads_exact"], ">= 600/630")]
        if a0_sets is None:
            out.append({"bar": "B5_patient_sets_equal_A0", "pass": None, "got": "UNDEFINED: A0 not scorable",
                        "rule": "identical in every block"})
        else:
            diff = sorted("%s/%s" % k for k, v in cnt["_c_sets"].items() if v != a0_sets.get(k))
            out.append(_bar("B5_patient_sets_equal_A0", not diff, diff, "identical in every block"))
        out.append(_bar("B6_ambiguous_both_fire", cnt["ambiguous_all_fire"],
                        [a for a in cnt["ambiguous"] if len(a["fired"]) != len(a["true_blocks"])],
                        "every true block of every AMBIGUOUS_CUES cue fires"))
        return out
    if arm == "L1":
        return [_bar("L1_both_lit_return", f["both"] >= 60, f["both"], ">= 60/86")]
    if arm == "L3":
        return [_bar("L3_true_fires", f["true"] <= 20, f["true"], "<= 20/98")]
    return []                                      # L1w: report only


def score(root=OUT_ROOT, ref_root=REF_ROOT, seeds=SEEDS, arms=tuple(ARMS)):
    ref = load_wave(ref_root, seeds)
    pairs, info, why_ref = classify_reference(ref)
    fc = frozen_counts(pairs, info)
    res = {"ruling": RULING, "root": root, "ref_root": ref_root, "frozen_expected": FROZEN, "frozen_got": fc, "arms": {}}
    if why_ref or fc != FROZEN:
        res["verdict"] = "UNDEFINED"
        res["why"] = why_ref + ([] if fc == FROZEN else ["frozen denominators differ from the ruling's"])
        return res
    cnts = {}
    for arm in arms:
        wave = load_wave(os.path.join(root, arm), seeds)
        why = []
        for (s, t), sh in sorted(wave.items()):
            for w in engagement(arm, sh, s, t):
                why.append("s%d %s: %s" % (s, t, w))
            if not why and len(_reads(sh)["b"]["cues"][CUE_ORDER[0]]["m_rates"]) != info[(s, t)]["K"]:
                why.append("s%d %s: K differs from the reference" % (s, t))
            if not why and [[(r or {}).get(k) for k in ("agent", "action")] for r in _reads(sh).get("decoded_rows") or []] \
                    != [[(r or {}).get(k) for k in ("agent", "action")] for r in info[(s, t)]["rows"]]:
                why.append("s%d %s: decoded agent/action rows differ from the reference" % (s, t))
        if why:
            res["arms"][arm] = {"verdict": "UNDEFINED", "why": why[:40], "n_why": len(why)}
            continue
        cnts[arm] = arm_counts(wave, pairs, info)
    a0_sets = cnts["A0"]["_c_sets"] if "A0" in cnts else None
    for arm, cnt in cnts.items():
        bl = bars(arm, cnt, a0_sets)
        if any(x["pass"] is None for x in bl):
            v = "UNDEFINED"
        elif not bl:
            v = "REPORT_ONLY"
        else:
            v = "PASS" if all(x["pass"] for x in bl) else "FAIL"
        res["arms"][arm] = {"verdict": v, "bars": bl, **{k: x for k, x in cnt.items() if not k.startswith("_")}}
    res["attribution"] = attributions(cnts)
    return res


# (treatment, control, class, label): whose is the difference? Each lesion/fix pair, report only, never a bar input.
# Treatment/control are FIRE COUNTS of one frozen class; the fraction is (treatment - control) / treatment.
ATTRIBUTION_PAIRS = (("A0", "A1", "one", "one-lit fires owned by the 20-step drain (A0) vs the 60-step quiet period (A1)"),
                     ("A1", "A2r", "both", "both-lit fires the role relay removes (A1 treatment, A2r control)"),
                     ("A1", "A2w", "both", "both-lit fires the role WTA removes (A1 treatment, A2w control)"),
                     ("L1", "A2r", "both", "both-lit fires that return when the relay I->E is zeroed (L1 vs A2r)"),
                     ("L1w", "A2w", "both", "both-lit fires that return when the WTA I->E is zeroed (L1w vs A2w)"),
                     ("A2r", "L3", "true", "true fires owned by each block's OWN row (A2r vs row-shuffled L3)"))


def attributions(cnts):
    """`tools.lab.attributable_to` on every scorable (treatment, control) pair; None when either arm is UNDEFINED."""
    import contextlib
    from tools.lab import attributable_to
    out = {}
    for t, c, cls, label in ATTRIBUTION_PAIRS:
        key = "%s_vs_%s_%s" % (t, c, cls)
        if t not in cnts or c not in cnts:
            out[key] = {"label": label, "fraction": None, "why": "an arm is UNDEFINED or not scored"}
            continue
        tv, cv = cnts[t]["fires"][cls], cnts[c]["fires"][cls]
        with contextlib.redirect_stdout(sys.stderr):
            fr = attributable_to(label, tv, cv)
        out[key] = {"label": label, "treatment": tv, "control": cv, "fraction": fr}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--score", default=None, metavar="ROOT")
    ap.add_argument("--ref", default=REF_ROOT)
    ap.add_argument("--arms", default=None, help="comma list (default all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-pool-lines", default=None, metavar="SHA")
    a = ap.parse_args()
    arms = tuple(a.arms.split(",")) if a.arms else tuple(ARMS)
    if a.emit_pool_lines:
        for cmd, why in pool_lines(a.emit_pool_lines, arms=arms):
            print("%s\t%s" % (cmd, why))
        return
    if a.score:
        res = score(a.score, a.ref, arms=arms)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=1, default=str)
        print(json.dumps({"frozen_got": res["frozen_got"], "verdict": res.get("verdict"),
                          "arms": {k: {"verdict": v["verdict"], "bars": v.get("bars"), "why": v.get("why", [])[:5],
                                       "report_only": v.get("report_only")} for k, v in res["arms"].items()}},
                         indent=1, default=str))
        return
    ap.error("choose --score or --emit-pool-lines")


if __name__ == "__main__":
    main()
