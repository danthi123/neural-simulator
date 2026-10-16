"""Dev battery for the emotion-category organ (packet s2-c4-build; docs/drafts/stage2-c4-build/PLAN.md).

Organ level, no handler, numpy, ~5-10 s per (seed, row). DEV SEEDS 1-8 ONLY: gate seeds 42/43/44/100/101/102 are
refused (the Stage-2 gate runs them through the production handler in its window, 2026-11-03..11-10).

Each case replays the Stage-2 C4 protocol at the organ (each situation twice, then a read), with the gate's frozen
situation texts and its permuted swap. A seed PASSES iff every case below is right:

  intact        reads joy, fear, anger, sadness after the four situations, every read threshold-invariant
  permuted      the affect words swapped (joy<->sadness, fear<->anger) -> reads sadness, anger, fear, joy
  affect_lesion BRAIN_AFFECT_LESION semantics (`affect_out` closed) -> no category on all four
  word_lesion   word -> appraisal cut -> no category on all four
  latch_lesion  rung self-recurrence cut -> no category on all four (the state must be HELD to be named)
  commit_lesion readout -> commit gate cut -> no category on all four
  shuffle       readout k wired to commit gate k+1 (a build) -> reads fear, anger, sadness, joy (follows the wiring)
  teach_swap    fear/anger triggers exchanged in development (a build) -> the fear situation reads anger and vice versa
  ladder_only   a fresh session, appraisal ensemble k driven directly (no word) -> reads k, for each k
  baseline      a fresh session reads no category; a neutral message after joy leaves joy

The last six are what separate a brain whose category is read off its held ladder from a host word lookup (a lookup
names under the affect lesion, ignores the shuffle and the teaching, and names nothing on a ladder-only drive). The
bar-sanity spec (docs/drafts/stage2-c4-build/bar_sanity_spec_dev.py) calls `seed_pass` / `battery_verdict` on synthetic
records in this format.

    python -m research.runners.emotion_category_dev --seeds 1-8 --op 0 --out research/findings/raw/_s2_c4_build/dev_op0.json
    python -m research.runners.emotion_category_dev --select --out research/findings/raw/_s2_c4_build/dev_select.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
CATS = ("joy", "fear", "anger", "sadness")
# copied from research/runners/stage2_exit_gate.py (frozen by the Stage-2 prereg; equality pinned by the tests when the
# runner is importable)
SITUATIONS = {
    "joy": "you win a gift and you are happy and glad and delighted",
    "fear": "a wolf is attacking you and you are scared and afraid",
    "anger": "someone cheats you and you are angry and mad and furious",
    "sadness": "you are alone and lonely and sad and you cry",
}
SITUATION_AFFECT_WORDS = {"joy": ("happy", "glad", "delighted"), "fear": ("scared", "afraid"),
                          "anger": ("angry", "mad", "furious"), "sadness": ("lonely", "sad", "cry")}
SITUATION_SWAP = {"joy": "sadness", "sadness": "joy", "fear": "anger", "anger": "fear"}
ROTATE = {c: CATS[(i + 1) % len(CATS)] for i, c in enumerate(CATS)}
TEACH_SWAP = {"joy": "joy", "fear": "anger", "anger": "fear", "sadness": "sadness"}
NEUTRAL = "what does the cat eat"
# the ladder-only probe's direct current into one appraisal ensemble (pA). Calibrated before the battery so the
# ensemble fires at the rate the situation WORDS give it (word-driven intact: 20-57 spikes per 60-step exposure on dev
# seeds 1-4; direct 200 pA -> 23-29, 400 pA -> 42-55 on dev seeds 1/4/7; 1500 pA, the word assemblies' own drive,
# overdrives the ensemble and latches every channel). Fixed, not tuned on the battery's outcome.
LADDER_ONLY_PA = 300.0

# the expected read per case (situation -> category; None = no category)
EXPECT = {
    "intact": {c: c for c in CATS},
    "permuted": {c: SITUATION_SWAP[c] for c in CATS},
    "affect_lesion": {c: None for c in CATS},
    "word_lesion": {c: None for c in CATS},
    "latch_lesion": {c: None for c in CATS},
    "commit_lesion": {c: None for c in CATS},
    "shuffle": dict(ROTATE),
    "teach_swap": dict(TEACH_SWAP),
    "ladder_only": {c: c for c in CATS},
    "baseline": {"fresh": None, "neutral_after_joy": "joy"},
}


def situation_text(cat: str, swap: bool = False) -> str:
    msg = SITUATIONS[cat]
    if swap:
        src, dst = SITUATION_AFFECT_WORDS[cat], SITUATION_AFFECT_WORDS[SITUATION_SWAP[cat]]
        for i, w in enumerate(src):
            msg = re.sub(r"\b%s\b" % re.escape(w), dst[i % len(dst)], msg)
    return msg


def _read_rec(r: dict) -> dict:
    return {"category": r["category"], "read_invariant": bool(r["read_invariant"]), "co_commit": bool(r["co_commit"]),
            "slot_counts": r["slot_counts"], "readout_counts": r["readout_counts"], "gate_counts": r["gate_counts"]}


def run_seed(seed: int, op: int) -> dict:
    """Every case for one dev seed at operating-point row `op`. Returns {case: {situation: read record}}."""
    if int(seed) in GATE_SEEDS:
        raise SystemExit("refused: %d is a gate seed (dev seeds 1-8 only)" % seed)
    from research.runners import emotion_category_organ as E
    p = E.GRID[int(op)]
    base = E.EmotionCategoryOrgan(seed=seed, params=p).train()
    shuf = E.EmotionCategoryOrgan(seed=seed, params=p, shuffle=True).train()
    swap = E.EmotionCategoryOrgan(seed=seed, params=p, teach_swap=True).train()
    out: Dict[str, dict] = {}

    def sequence(org, key, swap_words=False, **kw):
        lesion = kw.get("lesion", False)
        recs = {}
        for c in CATS:
            msg = situation_text(c, swap_words)
            for _ in range(2):
                org.observe(key, msg, word_lesion=kw.get("word_lesion", False),
                            latch_lesion=kw.get("latch_lesion", False), lesion=lesion)
            recs[c] = _read_rec(org.read(key, lesion=lesion, commit_lesion=kw.get("commit_lesion", False),
                                         latch_lesion=kw.get("latch_lesion", False)))
        return recs

    out["intact"] = sequence(base, ("intact", seed))
    out["permuted"] = sequence(base, ("permuted", seed), swap_words=True)
    out["affect_lesion"] = sequence(base, ("affect_lesion", seed), lesion=True)
    out["word_lesion"] = sequence(base, ("word_lesion", seed), word_lesion=True)
    out["latch_lesion"] = sequence(base, ("latch_lesion", seed), latch_lesion=True)
    out["commit_lesion"] = sequence(base, ("commit_lesion", seed), commit_lesion=True)
    out["shuffle"] = sequence(shuf, ("shuffle", seed))
    out["teach_swap"] = sequence(swap, ("teach_swap", seed))
    lo = {}
    for c in CATS:
        key = ("ladder_only", seed, c)
        for _ in range(2):
            base.observe(key, "", drive_channels={c: LADDER_ONLY_PA})
        lo[c] = _read_rec(base.read(key))
    out["ladder_only"] = lo
    key = ("baseline", seed)
    bl = {"fresh": _read_rec(base.read(key))}
    for _ in range(2):
        base.observe(key, situation_text("joy"))
    base.observe(key, NEUTRAL)
    bl["neutral_after_joy"] = _read_rec(base.read(key))
    out["baseline"] = bl
    for org in (base, shuf, swap):
        org._sessions.clear()
    # attribution: did each cut / build move the NAMED categories relative to intact (the coarse read), and the latched
    # slot spikes (the continuous read beside it)? `lever` prints both; required=False (a control may legitimately not
    # move, and the verdict is the case table, not this record).
    from tools.lab import lever

    def names(case):
        return tuple(out[case][c]["category"] for c in CATS)

    def spikes(case):
        return int(sum(sum(out[case][c]["slot_counts"].values()) for c in CATS))
    attribution = {}
    for case in ("affect_lesion", "word_lesion", "latch_lesion", "commit_lesion", "shuffle", "teach_swap"):
        moved = lever("%s named categories (seed %d)" % (case, seed), names("intact"), names(case), required=False,
                      continuous=(spikes("intact"), spikes(case)))
        attribution[case] = {"moved": bool(moved), "intact": list(names("intact")), "arm": list(names(case)),
                             "slot_spikes_intact": spikes("intact"), "slot_spikes_arm": spikes(case)}
    return {"seed": int(seed), "op": int(op), "cases": out, "weights": base.weight_summary, "attribution": attribution}


def case_pass(case: str, recs: dict) -> bool:
    """A case is right iff every read names the expected category and, where a category is named, the read is
    threshold-invariant."""
    want = EXPECT[case]
    for k, w in want.items():
        r = recs.get(k)
        if r is None or r.get("category") != w:
            return False
        if w is not None and not r.get("read_invariant", False):
            return False
    return True


def seed_pass(rec: dict) -> dict:
    per = {case: case_pass(case, rec["cases"].get(case, {})) for case in EXPECT}
    return {"pass": all(per.values()), "cases": per}


def battery_verdict(records: List[dict], seeds=DEV_SEEDS) -> dict:
    """PASS iff every seed in `seeds` is present and passes every case."""
    by = {int(r["seed"]): r for r in records}
    per = {s: (seed_pass(by[s]) if s in by else {"pass": False, "cases": {}, "missing": True}) for s in seeds}
    return {"pass": all(v["pass"] for v in per.values()), "n_pass": sum(v["pass"] for v in per.values()),
            "n_seeds": len(seeds), "per_seed": per}


# ── handler-level dev lines (the Stage-2 runner's own worker, flag ON; for the pool via the runbook's routes) ─────────
HANDLER_OUT = "research/findings/raw/_s2_c4_build/handler"
FLAG_ENV = "BRAIN_EMOTION_CATEGORY=1"
FLAG_OFF_ENV = "BRAIN_EMOTION_CATEGORY=0"   # explicit: the server default is ON since 2026-10-15


def handler_lines(sha: str, seeds=DEV_SEEDS, groups=("emo",), arms=None, flag_on: bool = True,
                  out_dir: str = HANDLER_OUT, max_age_h: int = 168):
    """Pool lines for DEV seeds through `research.runners.stage2_exit_gate --shard` at revision `sha` (which must
    contain this module AND the Stage-2 runner). Each line is the Stage-2 runner's own line with BRAIN_EMOTION_CATEGORY=1
    (flag_on) or =0 (flag_off; explicit since the default flipped ON 2026-10-15) added to its `env`, written under
    `out_dir/{on,off}/`. Refuses gate seeds."""
    import research.runners.stage2_exit_gate as G
    bad = [s for s in seeds if int(s) in GATE_SEEDS]
    if bad:
        raise SystemExit("refused: gate seeds %s (dev seeds 1-8 only)" % bad)
    sub = os.path.join(out_dir, "on" if flag_on else "off")
    out = []
    for cmd, why in G.pool_job_lines(sha, seeds=[int(s) for s in seeds], out_dir=sub, informational=False,
                                     max_age_h=max_age_h):
        m = re.search(r"--group (\S+) --arm (\S+)", cmd)
        if not m or m.group(1) not in groups or (arms is not None and m.group(2) not in arms):
            continue
        cmd = cmd.replace(" --allow-gate-seed", "").replace(" --prereg %s" % G.PREREG, "")
        flag = FLAG_ENV if flag_on else FLAG_OFF_ENV
        cmd = cmd.replace("-- env SIM_BACKEND=numpy", "-- env %s SIM_BACKEND=numpy" % flag)
        assert flag in cmd
        why = "s2-c4-build handler dev (%s, flag %s); %s" % (sha[:9], "ON" if flag_on else "OFF",
                                                            why.split("; ", 1)[1].split("; record:")[0])
        out.append((cmd, why + "; mem_gb=%d max_age_h=%d" % (G.MEM_GB, min(int(max_age_h), 240))))
    return out


def flip_check(on_dir: str, off_dir: str, seeds) -> dict:
    """FLIP SPECIFICITY + answer rate on handler shards (flag on vs off, same seed/group/arm): on every turn that is not
    a feel turn the reply fields (answer, abstained, recalled_svo) are identical; on every feel turn the ON answer is the
    OFF answer plus exactly one feel clause; the ON answer rate (tools/answer_rate.classify over every turn) is not
    below OFF. `FLIP-HURTS-ANSWERS` / `FLIP-CHANGES-NONFEEL` flags are reported; UNMEASURED pairs are listed."""
    import glob
    # Import through the repo root (tools/ is a namespace package). Putting tools/ first on sys.path let
    # tools/experiment.py shadow the root `experiment` package (ImportError: ExperimentEngine), 2026-10-15.
    from tools.answer_rate import classify
    from research.runners import emotion_category_organ as E
    clauses = {E.feel_clause(c) for c in list(CATS) + [None]}
    per, flags, unmeasured = {}, set(), []
    for s in seeds:
        for on_path in sorted(glob.glob(os.path.join(on_dir, "s%d" % int(s), "*.json"))):
            if on_path.endswith(".prov.json"):
                continue
            name = os.path.basename(on_path)
            off_path = os.path.join(off_dir, "s%d" % int(s), name)
            if not os.path.exists(off_path):
                unmeasured.append("s%d/%s" % (int(s), name))
                continue
            on, off = json.load(open(on_path)), json.load(open(off_path))
            ro, rf = on.get("responses") or {}, off.get("responses") or {}
            rec = {"nonfeel_differ": [], "feel_bad": [], "answered_on": 0, "answered_off": 0, "n": 0}
            for lab in rf:
                a, b = ro.get(lab) or {}, rf.get(lab) or {}
                rec["n"] += 1
                rec["answered_on"] += classify(a) == "answered"
                rec["answered_off"] += classify(b) == "answered"
                if b.get("inner_state_readout"):
                    ans_on, ans_off = str(a.get("answer") or ""), str(b.get("answer") or "")
                    if not (ans_on.startswith(ans_off) and ans_on[len(ans_off):] in clauses):
                        rec["feel_bad"].append(lab)
                elif any(a.get(k) != b.get(k) for k in ("answer", "abstained", "recalled_svo")):
                    rec["nonfeel_differ"].append(lab)
            if rec["answered_on"] < rec["answered_off"]:
                flags.add("FLIP-HURTS-ANSWERS")
            if rec["nonfeel_differ"]:
                flags.add("FLIP-CHANGES-NONFEEL")
            if rec["feel_bad"]:
                flags.add("FEEL-NOT-APPEND-ONLY")
            per["s%d/%s" % (int(s), name)] = rec
    return {"pairs": per, "flags": sorted(flags), "unmeasured": unmeasured, "n_pairs": len(per)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-handler-lines", default=None, metavar="SHA",
                    help="print pool lines (TSV) for the handler dev check at SHA; see --groups/--arms/--flag-off")
    ap.add_argument("--groups", default="emo")
    ap.add_argument("--arms", default=None, help="comma-separated arms (default: every registered arm of the group)")
    ap.add_argument("--flag-off", action="store_true")
    ap.add_argument("--flip-check", nargs=2, default=None, metavar=("ON_DIR", "OFF_DIR"))
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("--op", type=int, default=0)
    ap.add_argument("--select", action="store_true", help="rows in GRID order; stop at the first that passes 8/8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    if "-" in a.seeds:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
    if a.emit_handler_lines:
        for cmd, why in handler_lines(a.emit_handler_lines, seeds=seeds, groups=tuple(a.groups.split(",")),
                                      arms=(a.arms.split(",") if a.arms else None), flag_on=not a.flag_off):
            print("%s\t%s" % (cmd, why))
        return 0
    if a.flip_check:
        rep = flip_check(a.flip_check[0], a.flip_check[1], seeds)
        print(json.dumps({k: rep[k] for k in ("flags", "unmeasured", "n_pairs")}))
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump(rep, f, indent=1)
        return 0
    if not a.out:
        ap.error("--out is required for the battery")
    from research.runners import emotion_category_organ as E
    rows = list(range(len(E.GRID))) if a.select else [a.op]
    report = {"runner": "research/runners/emotion_category_dev.py", "seeds": seeds, "rows": {}, "selected": None}
    t0 = time.time()
    for op in rows:
        recs = []
        for s in seeds:
            r = run_seed(s, op)
            recs.append(r)
            sp = seed_pass(r)
            print("[emo dev] op=%d seed=%d %s %s" % (op, s, "PASS" if sp["pass"] else "FAIL",
                                                     {k: v for k, v in sp["cases"].items() if not v}), flush=True)
        v = battery_verdict(recs, tuple(seeds))
        report["rows"][str(op)] = {"verdict": v, "records": recs, "params": E.GRID[op]}
        if a.select and v["pass"]:
            report["selected"] = op
            break
    report["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("[emo dev] -> %s (selected=%s, %.0fs)" % (a.out, report["selected"], report["wall_s"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
