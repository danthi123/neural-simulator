# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md (section 5, "Scorers": G7, G8, G10, G11; DEV validation)
# Expected-fail: none (host-only post-hoc read of life JSONs; K-P is expected to FAIL G7/G8/G10 at DEV; G11 is a setup check (both-off rows equal), it cannot fail an arm)
"""Stage-2 TEST v2 cost guards G7 (self-name loss), G8 (refusal: (a) rate, (b) collapse night), G10 (experience
claims in lived turns), G11 (both-off confabulation), read from life results JSONs, plus the leak on non-refusal rows
(G4 / D2-PR2 second reading) and the per-guard bar verdicts of an arm against B paired by seed.

Rows (prereg section 5): battery probes (res['probes']) of the intact and hippocampus-off arms, line "trunk", no tag
(as life_leak._rows, probe source only), nights 1-15. Both-off probes (same line/tag filter) only for G11. Text is
life_leak._norm(reply). Regexes are copied verbatim from the prereg and must never be edited after a v2 row exists.

Amendment T1 rules (shared with life_compare, which imports the helpers below, so both scorers read seeds identically):
- Setup check (seed UNDEFINED for every bar, listed with the reason; neither pass nor fail, the read waits for a rerun):
  arm and B differ in split, or in device / cuda_precision / torch / torch_cuda / gpu_name where recorded on BOTH lives
  (a field absent on either side is "provenance unrecorded": UNDEFINED in registered mode, a note under --dev), or their both-off rows
  (night, question `you`, reply) differ on the nights both completed (G11: both-off is the same pretrained cortex for
  every arm, so a difference is a setup mismatch).
- Missing / killed lives: completed nights k = the longest run 1..k of complete nights (trunk night record, intact and
  hippocampus-off trunk probes, interview probes on nights 10/15). An ARM life with k < 15 (a section 10 kill) is scored
  on nights 1..k, and B is restricted to the same nights for that seed; a bar that needs a later night counts that seed
  as not passing. A missing arm life is not passing. A missing B life, or a B life with k < 15, makes the seed UNDEFINED.
  A bar with fewer than need (5 of 6) scorable seeds is UNDEFINED; otherwise it passes on >= need passing seeds.
- G11 is a setup check, not a cost guard: per seed PASS iff the both-off rows are equal, else UNDEFINED; the confabulation
  rate (and its old rate <= B + 0.15 comparison) is reported only.
- G8b: the any-seed exception decides; the same-seed variant (excused only by B >= 0.6 at the same seed, arm and night)
  is reported beside it, and every excused cell is listed.

CLI: python -m brain.eval.life_costs --arm A_s1.json ... --b B_s1.json ... [--dev] [--no-setup-check] [--out costs.json]
(--arm and --b are paired by the life's seed; default expects the six registered seeds, --dev scores the seeds given)."""
from __future__ import annotations

import argparse
import json
import re

from brain.eval import life_bars, life_leak

ARMS = ("intact", "hippocampus-off")
BOTH_OFF = "both-off"
NIGHTS = range(1, 16)
LAST_NIGHT = 15
LEAK_NIGHTS = (10, 15)
REGISTERED_SEEDS = (42, 43, 44, 100, 101, 102)
UNDEF = "UNDEFINED"
PROV_CHECKED = ("device", "cuda_precision", "torch", "torch_cuda", "gpu_name")
PROV_REPORTED = ("tf32_matmul", "tf32_cudnn", "code_commit", "aws_ami_id")
IDENTITY = ("dan", "nova")

# ---- verbatim from the prereg, section 5 (do not edit)
REFUSAL_RX = re.compile(r"^\W*(?:(?:i'?m|i am) sorry,?\s*)?(?:but\s+)?(?:i\s+(?:don'?t|do not)\s+(?:know|understand)\b|i'?m not sure\b|i am not sure\b)")
REFUSAL_SUBSTR = "don't know what you mean"
SENT_SPLIT = r"[.!?]"
EXPERIENCE_RX = re.compile(r"\bi(?:'ve| have)\s+(?:seen|been to|been there|visited|tasted|watched|heard|met|had it|never had|never been)\b|\bi saw\b|\bi went\b|\bi slept\b")
FRIEND_RX_TMPL = r"\b(?:my name is|i'?m called|i am called|call me|i'?m|i am)\s+{friend}\b"

# bar margins (prereg section 5 table)
M_G7, M_G8A, M_G10, M_G11, G8B_CUT = 0.25, 0.03, 0.03, 0.15, 0.6
EPS = 1e-12


def norm(s: str) -> str:
    return life_leak._norm(s)


def is_refusal(reply: str) -> bool:
    """G8 refusal: the first sentence (split [.!?]) matches the refusal regex, or the reply contains the stock phrase.
    Text is already lower-case (life_leak._norm), so the regex needs no IGNORECASE.
    Literal split: a reply that opens with sentence punctuation ("...I don't know.") has an EMPTY first sentence and is
    not a refusal unless the stock phrase is present. Kept as written (and identical to life_compare.is_refusal);
    0 DEV rows are affected. Record this reading in the freeze note."""
    t = norm(reply)
    first = re.split(SENT_SPLIT, t, maxsplit=1)[0]
    return bool(REFUSAL_RX.match(first)) or REFUSAL_SUBSTR in t


def is_experience_claim(reply: str) -> bool:
    return bool(EXPERIENCE_RX.search(norm(reply)))


def friend_rx(friend: str = "Dan"):
    # friend name lower-cased because the text is lower-cased by _norm
    return re.compile(FRIEND_RX_TMPL.format(friend=re.escape(norm(friend))))


def friend_of(res: dict) -> str:
    # `or`, not a .get default: friend None or "" would make friend_rx match any "i am ..." reply (life_leak does the same).
    # One-token deviation from the prereg's `brain_kw.get("friend","Dan")`; identical whenever friend is a non-empty name.
    return (res.get("brain_kw") or {}).get("friend") or "Dan"


def probe_rows(res: dict, arms=ARMS):
    """(arm, night, row): battery probe rows, given arms, line trunk, no tag, nights 1-15 (life_leak._rows filter)."""
    for src, arm, night, r in _all_rows(res, arms):
        if src == "probe" and night in NIGHTS:
            yield arm, night, r


def _all_rows(res: dict, arms):
    if set(arms) <= set(life_leak.ARMS):
        yield from life_leak._rows(res)  # the stock filter, intact + hippocampus-off
        return
    for p in res.get("probes", []):  # same filter for other arms (both-off)
        if p["arm"] in arms and p.get("line", "trunk") == "trunk" and not p.get("tag"):
            for r in p["rows"]:
                yield "probe", p["arm"], p["night"], r


def _answerable(r: dict) -> bool:
    k = r.get("kind")
    return ((k == "H" and r.get("item") not in IDENTITY) or k in ("rotating", "neardup")) and r.get("told") is not False


def _non_identity(r: dict) -> bool:
    k = r.get("kind")
    return (k == "H" and r.get("item") not in IDENTITY) or k in ("rotating", "neardup", "never")


def _rate(k, n):
    return k / n if n else None


def g7(res: dict) -> dict:
    """Self-name loss: kind H, item nova, share with ok False, intact + hippocampus-off pooled; friend-name rate beside."""
    rx = friend_rx(friend_of(res))
    n = w = fr = 0
    by_arm = {a: [0, 0] for a in ARMS}
    for arm, _, r in probe_rows(res):
        if r.get("kind") == "H" and r.get("item") == "nova":
            bad = r.get("ok") is False
            n += 1; w += bad; fr += bool(rx.search(norm(r.get("brain", ""))))
            by_arm[arm][0] += bad; by_arm[arm][1] += 1
    return {"n": n, "wrong": w, "rate": _rate(w, n), "friend_name": fr, "friend_rate": _rate(fr, n),
            "by_arm": {a: {"wrong": v[0], "n": v[1]} for a, v in by_arm.items()}}


def g8a(res: dict) -> dict:
    """Refusal rate on answerable rows (H not dan/nova, rotating, neardup; told False out), per probe arm."""
    out = {a: {"refusals": 0, "n": 0} for a in ARMS}
    for arm, _, r in probe_rows(res):
        if _answerable(r):
            out[arm]["n"] += 1; out[arm]["refusals"] += is_refusal(r.get("brain", ""))
    for c in out.values():
        c["rate"] = _rate(c["refusals"], c["n"])
    return out


def g8b(res: dict) -> dict:
    """Refusal share per (arm, night) over every non-identity row (H not dan/nova, rotating, neardup, never)."""
    out = {a: {} for a in ARMS}
    for arm, night, r in probe_rows(res):
        if _non_identity(r):
            c = out[arm].setdefault(night, [0, 0]); c[0] += is_refusal(r.get("brain", "")); c[1] += 1
    return {a: {n: {"refusals": c[0], "n": c[1], "share": _rate(c[0], c[1])} for n, c in sorted(d.items())}
            for a, d in out.items()}


def g10(res: dict, lines=("trunk",)) -> dict:
    """Experience claims in lived trunk turns (res['lived'], brain replies). `lines` is the prereg's trunk; any other
    value is a diagnostic only (the prereg's DEV counts "of 166" were read over trunk + prunedA, see the tests)."""
    n = k = 0
    for d in res.get("lived", []):
        if lines is not None and d.get("line", "trunk") not in lines:
            continue
        for x in d["rows"]:
            n += 1; k += is_experience_claim(x.get("brain", ""))
    return {"hits": k, "n": n, "rate": _rate(k, n)}


def g11(res: dict) -> dict:
    """Both-off confabulation: both-off rows of the G8b kinds, share neither a G8 refusal nor ok."""
    n = k = 0
    for _, _, r in probe_rows(res, (BOTH_OFF,)):
        if _non_identity(r):
            n += 1; k += (not is_refusal(r.get("brain", ""))) and not r.get("ok")
    return {"confab": k, "n": n, "rate": _rate(k, n)}


def leak_nonrefusal(res: dict, nights=(10, 15)) -> dict:
    """Stock leak (life_leak) on nights 10+15, both arms, all rows and with G8-refusal replies dropped."""
    kw = res.get("brain_kw") or {}
    needles = life_leak.permanent_replies(kw.get("name") or life_leak.NAME, kw.get("friend") or life_leak.FRIEND)
    ident = life_leak.identity_items(res)
    a = {"n": 0, "leaks": 0}; nr = {"n": 0, "leaks": 0}
    for _, arm, night, r in life_leak._rows(res):
        if night not in nights or r.get("item") in ident:
            continue
        leak = life_leak.is_leak(r.get("brain", ""), needles)
        a["n"] += 1; a["leaks"] += leak
        if not is_refusal(r.get("brain", "")):
            nr["n"] += 1; nr["leaks"] += leak
    return {"all": a, "non_refusal": nr}




# ---------------------------------------------------------------- Amendment T1: seeds, setup check, restriction
def completed_nights(res: dict, top: int = LAST_NIGHT) -> int:
    """Longest run of complete nights 1..k: trunk night record (when the life has a nights list), intact and
    hippocampus-off trunk untagged probes, and on nights 10/15 their trunk interview probes."""
    tn = {n["night"] for n in res.get("nights", []) if n.get("line", "trunk") == "trunk"} if "nights" in res else None
    pn = {a: {p["night"] for p in res.get("probes", []) if p["arm"] == a and p.get("line", "trunk") == "trunk"
              and not p.get("tag")} for a in ARMS}
    iv = {a: {p["night"] for p in res.get("interview_probes", []) if p["arm"] == a and (p.get("line") or "trunk") == "trunk"
              and not p.get("tag")} for a in ARMS}
    k = 0
    for n in range(1, top + 1):
        ok = (tn is None or n in tn) and all(n in pn[a] for a in ARMS) and \
            (n not in LEAK_NIGHTS or all(n in iv[a] for a in ARMS))
        if not ok:
            break
        k = n
    return k


def restrict(res: dict, k: int) -> dict:
    """Shallow copy of a life with nights / probes / lived days > k dropped; records k as _completed_nights."""
    out = dict(res)
    for key, f in (("nights", "night"), ("probes", "night"), ("interview_probes", "night"), ("cue_probes", "night"),
                   ("lived", "day")):
        if key in res:
            out[key] = [x for x in res[key] if x.get(f) is None or x[f] <= k]
    out["_completed_nights"] = k
    return out


def provenance(res: dict) -> dict:
    """The recorded stack fields of a life (only the keys the life recorded: an absent key is 'unrecorded')."""
    p = {}
    if "device" in res:
        p["device"] = res["device"]
    rec = res.get("provenance")
    if isinstance(rec, dict):
        for k in PROV_CHECKED[1:] + PROV_REPORTED:
            if k in rec:
                p[k] = rec[k]
    return p


def both_off_rows(res: dict) -> dict:
    """{night: [(question, reply), ...]} of the trunk untagged both-off probes, nights 1-15."""
    out = {}
    for p in res.get("probes", []):
        if p["arm"] == BOTH_OFF and p.get("line", "trunk") == "trunk" and not p.get("tag") and p["night"] in NIGHTS:
            out.setdefault(p["night"], []).extend((r.get("you"), r.get("brain")) for r in p["rows"])
    return out


def setup_info(res: dict) -> dict:
    return {"split": res.get("split"), "prov": provenance(res), "both_off": both_off_rows(res),
            "k": res.get("_completed_nights", completed_nights(res))}


def both_off_equal(ai: dict, bi: dict) -> list:
    """Nights (<= both lives' completed nights) on which the both-off rows differ (empty = equal)."""
    m = min(ai["k"], bi["k"])
    return [n for n in sorted(set(ai["both_off"]) | set(bi["both_off"])) if n <= m
            and ai["both_off"].get(n) != bi["both_off"].get(n)]


def setup_reasons(ai: dict, bi: dict, registered: bool = False) -> tuple:
    """(reasons -> seed UNDEFINED, notes -> reported only) for an arm / B pair of setup_info dicts. registered (not --dev):
    a checked provenance field unrecorded on either side is a reason (seed UNDEFINED); --dev: a note only."""
    why, notes = [], []
    if ai["split"] != bi["split"]:
        why.append(f"split differs (arm {ai['split']}, B {bi['split']})")
    for f in PROV_CHECKED:
        if f in ai["prov"] and f in bi["prov"]:
            if ai["prov"][f] != bi["prov"][f]:
                why.append(f"{f} differs (arm {ai['prov'][f]!r}, B {bi['prov'][f]!r})")
        else:
            side = "+".join(x for x, d in (("arm", ai), ("B", bi)) if f not in d["prov"])
            (why if registered else notes).append(f"provenance unrecorded: {f} ({side})")
    bad = both_off_equal(ai, bi)
    if bad:
        why.append(f"both-off rows differ on nights {bad} (G11 setup mismatch)")
    if not ai["both_off"] or not bi["both_off"]:
        notes.append("both-off rows absent on " + "+".join(x for x, d in (("arm", ai), ("B", bi)) if not d["both_off"]))
    return why, notes


def seed_status(seeds, A: dict, B: dict, setup_check: bool = True, registered: bool = False) -> dict:
    """Per seed: undefined reasons, notes, arm/B completed nights and whether the arm life is missing (the unified rule,
    used verbatim by life_compare). registered = not --dev (unrecorded checked provenance -> UNDEFINED)."""
    out = {}
    for s in seeds:
        a, b = A.get(s), B.get(s)
        st = {"undefined": [], "notes": [], "arm_k": None, "b_k": None, "arm_missing": a is None}
        if b is None:
            st["undefined"].append("B life missing")
        else:
            st["b_k"] = completed_nights(b)
            if st["b_k"] < LAST_NIGHT:
                st["undefined"].append(f"B life ended after night {st['b_k']}")
        if a is None:
            st["notes"].append("arm life missing (not passing)")
        else:
            st["arm_k"] = completed_nights(a)
            if st["arm_k"] < LAST_NIGHT:
                st["notes"].append(f"arm life ended after night {st['arm_k']}: scored on nights 1-{st['arm_k']}, "
                                   f"B restricted to the same nights")
        if a is not None and b is not None:
            why, notes = setup_reasons(setup_info(restrict(a, st["arm_k"])), setup_info(restrict(b, st["b_k"])),
                                       registered)
            st["notes"] += notes
            if setup_check:
                st["undefined"] += why
            elif why:
                st["notes"] += [f"(setup check off) {w}" for w in why]
        out[s] = st
    return out


def score(res: dict) -> dict:
    return {"seed": res.get("seed"), "g7": g7(res), "g8a": g8a(res), "g8b": g8b(res), "g10": g10(res),
            "g11": g11(res), "leak": leak_nonrefusal(res), "completed_nights": res.get("_completed_nights"),
            "setup": setup_info(res)}


def _le(a, b, m):
    """Per-seed bar a <= b + m. None (MISSING, not passing) when either rate is undefined (no rows)."""
    if a is None or b is None:
        return None
    return a <= b + m + EPS


def _all_le(pairs):
    v = [_le(a, b, m) for a, b, m in pairs]
    return None if None in v else all(v)


def _g8b_cells(a):
    return [(x, n, c["share"]) for x in ARMS for n, c in a["g8b"][x].items()
            if c["share"] is not None and c["share"] >= G8B_CUT]


def bars(arm: list, base: list, status: dict | None = None, setup_check: bool = True, seeds=None) -> dict:
    """Per-guard verdicts of an arm against B, lives paired by seed (lists of score() dicts or None, same seed order).
    status: seed_status() output (compare() passes it); without it only the setup check runs here (no completeness).
    Per-seed flag: True / False / None (MISSING: arm life missing or no rows; not passing) / "UNDEFINED".
    Guard: UNDEFINED if fewer than need seeds are scorable (need 5 of 6; ceil(5n/6) off six seeds, labelled DEV);
    else PASS iff >= need seeds pass. G8b: every scorable seed passes (no 5/6 allowance); a cell >= 0.6 is excused if
    B >= 0.6 at that arm and night in ANY scorable B seed. G11: both-off rows equal (setup check), rate reported."""
    seeds = list(seeds) if seeds is not None else [(a or b)["seed"] for a, b in zip(arm, base)]
    assert all(a is None or b is None or a["seed"] == b["seed"] for a, b in zip(arm, base)), "arm and B must be paired by seed"
    if status is None:
        status = {}
        for s, a, b in zip(seeds, arm, base):
            st = {"undefined": [], "notes": [], "arm_missing": a is None}
            if b is None:
                st["undefined"].append("B life missing")
            if a is not None and b is not None:
                why, notes = setup_reasons(a["setup"], b["setup"])
                st["notes"] += notes
                st["undefined" if setup_check else "notes"] += why
            status[s] = st
    undef = {s for s in seeds if status[s]["undefined"]}
    good_b = [b for s, b in zip(seeds, base) if b is not None and s not in undef]

    def b_collapsed(x, night):
        return [bs["seed"] for bs in good_b if (bs["g8b"][x].get(night) or {}).get("share") is not None
                and bs["g8b"][x][night]["share"] >= G8B_CUT]
    per, g8b_bad, g8b_same, excused = [], [], [], []
    for s, a, b in zip(seeds, arm, base):
        row = {"seed": s, "undefined": status[s]["undefined"], "notes": status[s]["notes"]}
        if a is None or b is None:
            raw = {g: None for g in ("G7", "G8a", "G10", "G11_rate_cmp")}
            bo = None
        else:
            raw = {"G7": _le(a["g7"]["rate"], b["g7"]["rate"], M_G7),
                   "G8a": _all_le((a["g8a"][x]["rate"], b["g8a"][x]["rate"], M_G8A) for x in ARMS),
                   "G10": _le(a["g10"]["rate"], b["g10"]["rate"], M_G10),
                   "G11_rate_cmp": _le(a["g11"]["rate"], b["g11"]["rate"], M_G11)}
            bo = not both_off_equal(a["setup"], b["setup"])
        if s in undef:
            row.update({g: UNDEF for g in ("G7", "G8a", "G8b", "G10", "G11")})
        else:
            row.update({g: raw[g] for g in ("G7", "G8a", "G10")})
            row["G11"] = None if a is None else (True if bo else UNDEF)
        row["G11_rate_cmp (reported)"] = raw["G11_rate_cmp"]
        row["raw"] = raw
        if a is not None and b is not None and s not in undef:
            bad, same = [], []
            for x, n, sh in _g8b_cells(a):
                bc = b["g8b"][x].get(n)
                cell = {"arm": x, "night": n, "share": sh, "b_share": bc["share"] if bc else None}
                by = b_collapsed(x, n)
                if by:
                    excused.append(dict(seed=s, **cell, excused_by_b_seeds=by))
                else:
                    bad.append(cell)
                if not (bc and bc["share"] is not None and bc["share"] >= G8B_CUT):
                    same.append(cell)
            row["G8b"] = not bad
            row["G8b same-seed (reported)"] = not same
            g8b_bad += [dict(seed=s, **d) for d in bad]
            g8b_same += [dict(seed=s, **d) for d in same]
        elif s not in undef:
            row["G8b"] = None
            row["G8b same-seed (reported)"] = None
        else:
            row["G8b same-seed (reported)"] = UNDEF
        per.append(row)
    n = len(per)
    need = life_bars._need(n)
    scorable = n - len(undef)
    dev = n != 6

    def verdict(k, all_needed=False):
        if scorable < need:
            core = UNDEF
        elif all_needed:
            core = "PASS" if k == scorable else "FAIL"
        else:
            core = "PASS" if k >= need else "FAIL"
        return core, (f"DEV ({n} seeds) {core}" if dev else core)
    out = {"per_seed": per, "n_seeds": n, "need": need, "n_scorable": scorable, "mode": "DEV" if dev else "registered",
           "undefined_seeds": {s: status[s]["undefined"] for s in sorted(undef)},
           "g8b_violations": g8b_bad, "g8b_excused_cells": excused,
           "g8b_same_seed (reported)": {"violations": g8b_same, "pass": None if scorable < need else not g8b_same}}
    for g in ("G7", "G8a", "G10", "G11"):
        k = sum(r[g] is True for r in per)
        core, lab = verdict(k, all_needed=(g == "G11"))
        out[g] = {"n_pass": k, "n_missing": sum(r[g] is None for r in per), "n_undefined": len(undef) if g != "G11" else
                  sum(r[g] == UNDEF for r in per), "verdict_core": core, "verdict": lab,
                  "pass": None if dev or core == UNDEF else core == "PASS", "pass_all": k == n}
    k = sum(r["G8b"] is True for r in per)
    core, lab = verdict(k, all_needed=True)
    out["G8b"] = {"n_pass": k, "n_undefined": len(undef), "verdict_core": core, "verdict": lab,
                  "pass": None if core == UNDEF else core == "PASS"}
    out["G11"]["role"] = "setup check (not a cost guard): both-off rows equal per seed; rate reported only"
    if out["G11"]["verdict_core"] == "FAIL":   # G11 is a setup check: a mismatch or missing life is UNDEFINED, never a FAIL
        out["G11"]["verdict_core"] = UNDEF
        out["G11"]["verdict"] = f"DEV ({n} seeds) {UNDEF}" if dev else UNDEF
        out["G11"]["pass"] = None
    return out


def compare(arm_lives: list, b_lives: list, expect_seeds=REGISTERED_SEEDS, setup_check: bool = True) -> dict:
    """Pair raw lives by seed and apply the T1 seed rule (seed_status), then bars(). expect_seeds None = DEV (the seeds
    given, union of arm and B)."""
    A, B = {r["seed"]: r for r in arm_lives}, {r["seed"]: r for r in b_lives}
    if len(A) != len(arm_lives) or len(B) != len(b_lives):
        raise ValueError("duplicate seed among the arm or B lives")
    seeds = sorted(set(A) | set(B)) if expect_seeds is None else sorted(expect_seeds)
    ignored = sorted((set(A) | set(B)) - set(seeds))
    st = seed_status(seeds, A, B, setup_check, registered=expect_seeds is not None)
    arm, base = [], []
    for s in seeds:
        k = st[s]["arm_k"] if s in A else LAST_NIGHT
        arm.append(score(restrict(A[s], k)) if s in A else None)
        base.append(score(restrict(B[s], k)) if s in B else None)
    v = bars(arm, base, status=st, seeds=seeds)
    v["ignored_seeds"] = ignored
    v["status"] = st
    return {"arm": arm, "b": base, "bars": v}


def _f(x):
    return "-" if x is None else f"{x:.3f}"


def _flag(x):
    return {True: "PASS", False: "FAIL", None: "MISSING", UNDEF: UNDEF}[x]


def render(arm: list, base: list | None = None, verdict: dict | None = None) -> str:
    L = ["| life | seed | G7 wrong/n (rate) | friend-name | G8a intact | G8a hoff | G8b max night | G10 hits/n | "
         "G11 confab/n (reported) | leak n10+15 all | leak non-refusal |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, lives in (("arm", arm), ("B", base or [])):
        for s in lives:
            if s is None:
                continue
            a, ga = s["g7"], s["g8a"]
            mx = max((c["share"] for x in ARMS for c in s["g8b"][x].values() if c["share"] is not None), default=None)
            L.append(f"| {tag} | {s['seed']} | {a['wrong']}/{a['n']} ({_f(a['rate'])}) | {a['friend_name']}/{a['n']} | "
                     f"{ga['intact']['refusals']}/{ga['intact']['n']} ({_f(ga['intact']['rate'])}) | "
                     f"{ga['hippocampus-off']['refusals']}/{ga['hippocampus-off']['n']} ({_f(ga['hippocampus-off']['rate'])}) | "
                     f"{_f(mx)} | {s['g10']['hits']}/{s['g10']['n']} | {s['g11']['confab']}/{s['g11']['n']} ({_f(s['g11']['rate'])}) | "
                     f"{s['leak']['all']['leaks']}/{s['leak']['all']['n']} | {s['leak']['non_refusal']['leaks']}/{s['leak']['non_refusal']['n']} |")
    if verdict:
        G = ("G7", "G8a", "G8b", "G10", "G11")
        L += ["", "| seed | G7 | G8a | G8b | G8b same-seed (reported) | G10 | G11 (setup check) | undefined / notes |",
              "|---|---|---|---|---|---|---|---|"]
        for r in verdict["per_seed"]:
            L.append(f"| {r['seed']} | " + " | ".join(_flag(r[g]) for g in G[:3]) + f" | {_flag(r['G8b same-seed (reported)'])} | "
                     f"{_flag(r['G10'])} | {_flag(r['G11'])} | {'; '.join(r['undefined'] + r['notes']) or '-'} |")
        L.append("| guard | " + " | ".join(f"{verdict[g]['n_pass']}/{verdict['n_seeds']} {verdict[g]['verdict']}" for g in G[:3])
                 + f" | {'-' if verdict['g8b_same_seed (reported)']['pass'] is None else ('PASS' if verdict['g8b_same_seed (reported)']['pass'] else 'FAIL')} | "
                 + " | ".join(f"{verdict[g]['n_pass']}/{verdict['n_seeds']} {verdict[g]['verdict']}" for g in G[3:]) + " | |")
        L.append(f"\nScorable seeds {verdict['n_scorable']}/{verdict['n_seeds']} (need {verdict['need']}); "
                 f"UNDEFINED seeds: {verdict['undefined_seeds'] or 'none'}.")
        L.append("G8b violations (any-seed rule): " + (", ".join(f"s{d['seed']} {d['arm']} n{d['night']} {d['share']:.2f}"
                                                                 for d in verdict["g8b_violations"]) or "none"))
        L.append("G8b excused cells: " + (", ".join(f"s{d['seed']} {d['arm']} n{d['night']} {d['share']:.2f} (B >= 0.6 on seeds "
                                                    f"{d['excused_by_b_seeds']})" for d in verdict["g8b_excused_cells"]) or "none"))
        L.append("G8b same-seed variant violations (reported): " + (", ".join(
            f"s{d['seed']} {d['arm']} n{d['night']} {d['share']:.2f}" for d in verdict["g8b_same_seed (reported)"]["violations"]) or "none"))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", nargs="+", required=True)
    ap.add_argument("--b", nargs="*", default=[])
    ap.add_argument("--dev", action="store_true", help="score the seeds given (need ceil(5n/6)), verdicts labelled DEV")
    ap.add_argument("--no-setup-check", action="store_true", help="diagnostic: report setup mismatches, do not apply them")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    load = lambda p: json.load(open(p))
    if a.b:
        r = compare([load(p) for p in a.arm], [load(p) for p in a.b], None if a.dev else REGISTERED_SEEDS,
                    setup_check=not a.no_setup_check)
        arm, base, verdict = r["arm"], r["b"], r["bars"]
    else:
        arm = sorted((score(load(p)) for p in a.arm), key=lambda s: s["seed"])
        base, verdict = [], None
    print(render(arm, base, verdict))
    if a.out:
        strip = lambda L: [None if s is None else {k: v for k, v in s.items() if k != "setup"} for s in L]
        with open(a.out, "w") as f:
            json.dump({"arm_files": a.arm, "b_files": a.b, "arm": strip(arm), "b": strip(base), "bars": verdict}, f,
                      indent=1, default=str)


if __name__ == "__main__":
    main()
