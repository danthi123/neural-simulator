"""Organ-level DEV sweep of the BG commit stage (flag BRAIN_BG_COMMIT; organ region `bg-commit`): the operating point.

Not a verdict: dev seeds 1-8 only (gate seeds are refused). Worklog: docs/drafts/bg-commit/WORKLOG.md. Amendment:
research/findings/2026-10-14-bg-live-seam-commit-AMENDMENT.md (the selection rule below is registered there).

Per BG_COMMIT_GRID row and dev seed, on fresh organs (`NeuralSeamSelector(seed, commit=row)`):
  intact   the live session's four turns in order ('...', 'what does the dog chase', '?', 'hi'), then 12 trials per
           class (empty / content / one_word, four messages cycled), on ONE warm organ;
  lesions  commit (motor -> gate closed), latch (latch recurrence closed), excl (exclusion closed): one fresh organ
           each, 12 trials (four per class).
PASS (all must hold on the seed): C1 every intact and lesion trial's read is invariant at READ_FRAC x0.5 and x2;
C2 empty trials (the two live empties + 12) commit STAY-SILENT on >= 0.75; C3 no content trial (the live question +
12) commits STAY-SILENT; C4 the commit and latch lesions commit nothing on any trial; C5 every trial keeps the
membrane in [-150, +100] mV and starts with no latch carried over; C6 the live sequence holds both empties and does
not hold the question. Selection (`--select`), written before the sweep ran: the first row in grid order that passes on
all eight dev seeds. It selected NONE (2026-10-14): C2 / C3 / C6 test the race, not the read (the old latch read on
the same trials fails them on 5 of 8 seeds), so `select` also reports the DECLARED POST-DATA rule (dev data only,
before any gate seed; worklog and amendment): among the rows whose STAGE criteria (C1, C4, C5) hold on all eight
seeds, the one whose committed action agrees most often with the old latch read on the same intact trials (it
retires the read without changing what the seam decides); a tie goes to the first in grid order.
REPORTED (never in pass): co-commits, the exclusion lesion's outcomes, agreement with the old latch read (same
trials), the old latch read's own x0.5 / x2 changes, the production first-to-12 read, latched minimum / unlatched
maximum latch counts, the one-word hold rate.

  SIM_BACKEND=numpy python -m research.runners._bg_commit_derisk --sweep --seeds 1 2 3 4 5 6 7 8 \
      --out-dir research/findings/raw/_bg_commit/dev_organ
  python -m research.runners._bg_commit_derisk --select --out-dir research/findings/raw/_bg_commit/dev_organ
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
os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from research.runners import bg_action_selection_production_organ as BG  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
LIVE = [("hsb_empty1", "...", "empty"), ("hsb_q", "what does the dog chase", "content"),
        ("hsb_empty2", "?", "empty"), ("hsb_hi", "hi", "one_word")]
MESSAGES = {"empty": ["...", "?", "..", "!"], "content": ["what does the dog chase", "tell me about cats",
                                                          "the wolf bites the apple", "what does the cat eat"],
            "one_word": ["hi", "ok", "yes", "hmm"]}
N_PER_CLASS, N_LESION = 12, 12
LESIONS = ("commit", "latch", "excl")
STAGE_CHECKS = ("C1_read_invariant_every_trial", "C4_commit_and_latch_lesions_commit_nothing",
                "C5_membrane_and_no_carry")
BARS = {"empty_held_min": 0.75, "content_held_max": 0.0, "v_lo": -150.0, "v_hi": 100.0}


def trial_record(cls, msg, r, label=None) -> dict:
    """One trial's record (pure on the trial dict)."""
    c = r["commit"]
    return {"label": label, "class": cls, "msg": msg, "winner": c["winner"], "co_commit": bool(c["co_commit"]),
            "read_invariant": bool(c["read_invariant"]), "read_variants": c["read_variants"],
            "slot_counts": c["slot_counts"], "slot_carry": c["slot_carry"], "threshold": c["threshold"],
            "gate_counts_drive": c["gate_counts_drive"], "gate_counts_hold": c["gate_counts_hold"],
            "motor_counts_hold": c["motor_counts_hold"], "old_latch_winner": c["old_latch_winner"],
            "old_latch_variants": c["old_latch_variants"], "old_latch_rates": c["old_latch_rates"],
            "first_to_12": r["readouts"]["first_to_12_motor (production rule)"],
            "v_range": [float(r["v_min"]), float(r["v_max"])]}


def run_seed(seed: int, op: int) -> dict:
    row = BG.bg_commit_row(op)
    org = BG.NeuralSeamSelector(seed=seed, commit=row)
    intact = [trial_record(cls, m, org.select_once(m), label) for label, m, cls in LIVE]
    for cls, msgs in MESSAGES.items():
        intact += [trial_record(cls, msgs[i % len(msgs)], org.select_once(msgs[i % len(msgs)]))
                   for i in range(N_PER_CLASS)]
    lesions = {}
    for les in LESIONS:
        o = BG.NeuralSeamSelector(seed=seed, commit=row, commit_lesion=les)
        cls_cycle = list(MESSAGES)
        lesions[les] = [trial_record(cls_cycle[i % 3], MESSAGES[cls_cycle[i % 3]][(i // 3) % 4],
                                     o.select_once(MESSAGES[cls_cycle[i % 3]][(i // 3) % 4]))
                        for i in range(N_LESION)]
    return judge({"probe": "bg_commit_organ_derisk", "label": "DEV, not a verdict", "seed": int(seed), "op": int(op),
                  "row": row, "intact": intact, "lesions": lesions})


def judge(res: dict, bars=BARS) -> dict:
    """The pass criteria and the reported numbers from the trial records (pure)."""
    it, les = res["intact"], res["lesions"]
    every = it + [t for v in les.values() for t in v]

    def frac(rows, pred):
        return round(sum(1 for t in rows if pred(t)) / len(rows), 4) if rows else None

    by = {c: [t for t in it if t["class"] == c] for c in MESSAGES}
    live = {t["label"]: t for t in it if t["label"]}
    held = lambda t: t["winner"] == BG.STAY_SILENT  # noqa: E731
    committed = [t for t in it if t["winner"] is not None]
    on = [x for t in every for x in t["slot_counts"] if x >= t["threshold"]]
    off = [x for t in every for x in t["slot_counts"] if x < t["threshold"]]
    checks = {
        "C1_read_invariant_every_trial": all(t["read_invariant"] for t in every),
        "C2_empty_held": frac(by["empty"], held) >= bars["empty_held_min"],
        "C3_content_never_held": frac(by["content"], held) <= bars["content_held_max"],
        "C4_commit_and_latch_lesions_commit_nothing": all(t["winner"] is None for k in ("commit", "latch")
                                                          for t in les[k]),
        "C5_membrane_and_no_carry": all(bars["v_lo"] <= t["v_range"][0] and t["v_range"][1] <= bars["v_hi"]
                                        and sum(t["slot_carry"]) == 0 for t in every),
        "C6_live_sequence": (held(live["hsb_empty1"]) and held(live["hsb_empty2"])
                             and not held(live["hsb_q"])),
    }
    res["checks"] = checks
    res["all_pass"] = all(checks.values())
    res["reported"] = {
        "held": {c: frac(v, held) for c, v in by.items()},
        "commit_rate": {c: frac(v, lambda t: t["winner"] is not None) for c, v in by.items()},
        "co_commit": sum(t["co_commit"] for t in it),
        "agree_old_latch_read": frac(it, lambda t: t["winner"] == t["old_latch_winner"]),
        "agree_old_latch_read_when_both_commit": frac([t for t in committed if t["old_latch_winner"] is not None],
                                                      lambda t: t["winner"] == t["old_latch_winner"]),
        "old_latch_read_threshold_change": frac(it, lambda t: any(v != t["old_latch_winner"]
                                                                  for v in t["old_latch_variants"].values())),
        "first_to_12_change": frac(it, lambda t: t["first_to_12"] != t["winner"]),
        "latched_min": min(on) if on else None, "unlatched_max": max(off) if off else None,
        "lesion_commit_rate": {k: frac(v, lambda t: t["winner"] is not None) for k, v in les.items()},
        "excl_lesion_co_commit": sum(t["co_commit"] for t in les["excl"]),
        "v_range": [min(t["v_range"][0] for t in every), max(t["v_range"][1] for t in every)],
        "live": {k: v["winner"] for k, v in live.items()},
    }
    # ATTRIBUTION: how much of the intact commit rate is the stage's (commit lesion = motor -> gate closed, the
    # competition untouched). Reported only.
    rate = lambda rows: sum(1 for t in rows if t["winner"] is not None) / len(rows)  # noqa: E731
    res["reported"]["commit_rate_attributable_to_stage"] = attributable_to(
        "commit rate (intact vs commit lesion)", rate(it), rate(les["commit"]))
    return res


def run_value(seed: int, op: int) -> dict:
    """REPORTED only (the G3 preview): the value lesion of the neural seam (heard/hold -> D1 closed) with the commit
    stage on, 12 trials (four per class) on one fresh organ: held and commit rates per class, read invariance."""
    row = BG.bg_commit_row(op)
    o = BG.NeuralSeamSelector(seed=seed, lesion="value", commit=row)
    cls_cycle = list(MESSAGES)
    tr = [trial_record(cls_cycle[i % 3], MESSAGES[cls_cycle[i % 3]][(i // 3) % 4],
                       o.select_once(MESSAGES[cls_cycle[i % 3]][(i // 3) % 4])) for i in range(N_LESION)]
    by = {c: [t for t in tr if t["class"] == c] for c in MESSAGES}
    frac = lambda rows, pred: round(sum(1 for t in rows if pred(t)) / len(rows), 4)  # noqa: E731
    return {"probe": "bg_commit_value_lesion", "label": "DEV, reported only", "seed": int(seed), "op": int(op),
            "trials": tr, "held": {c: frac(v, lambda t: t["winner"] == BG.STAY_SILENT) for c, v in by.items()},
            "commit_rate": {c: frac(v, lambda t: t["winner"] is not None) for c, v in by.items()},
            "read_invariant_all": all(t["read_invariant"] for t in tr)}


def select(out_dir: str, seeds=DEV_SEEDS) -> dict:
    """The registered selection: the first BG_COMMIT_GRID row in grid order with all_pass on every dev seed."""
    table = {}
    for op in range(len(BG.BG_COMMIT_GRID)):
        rows = {}
        for s in seeds:
            p = os.path.join(out_dir, "op%d_s%d.json" % (op, s))
            rows[s] = json.load(open(p))["all_pass"] if os.path.exists(p) else None
        table[op] = rows
    sel = next((op for op, r in table.items() if all(v is True for v in r.values())), None)
    stage, agree = {}, {}
    for op in table:
        recs = [json.load(open(os.path.join(out_dir, "op%d_s%d.json" % (op, s)))) for s in seeds
                if os.path.exists(os.path.join(out_dir, "op%d_s%d.json" % (op, s)))]
        if len(recs) != len(seeds):
            continue
        stage[op] = all(r["checks"][k] for r in recs for k in STAGE_CHECKS)
        it = [t for r in recs for t in r["intact"]]
        agree[op] = round(sum(1 for t in it if t["winner"] == t["old_latch_winner"]) / len(it), 4)
    ok = [op for op in sorted(agree) if stage[op]]
    post = max(ok, key=lambda op: (agree[op], -op)) if ok else None
    return {"table": table, "selected": sel, "rule": "first row in grid order with all_pass on all dev seeds",
            "declared_post_data": {"rule": "max agreement with the old latch read among rows with C1, C4, C5 on all "
                                           "dev seeds; tie -> first in grid order",
                                   "stage_ok": stage, "agree_old_latch_read": agree, "selected": post}}


def summary(out_dir: str, seeds=DEV_SEEDS) -> dict:
    """Per-row aggregates over the dev sweep files (pure; the amendment's table cites this file)."""
    out = {}
    for op in range(len(BG.BG_COMMIT_GRID)):
        recs = [json.load(open(os.path.join(out_dir, "op%d_s%d.json" % (op, s)))) for s in seeds
                if os.path.exists(os.path.join(out_dir, "op%d_s%d.json" % (op, s)))]
        if len(recs) != len(seeds):
            continue
        it = [t for r in recs for t in r["intact"]]
        every = it + [t for r in recs for v in r["lesions"].values() for t in v]
        by = {c: [t for t in it if t["class"] == c] for c in MESSAGES}
        n_held = lambda ts, k="winner": sum(1 for t in ts if t[k] == BG.STAY_SILENT)  # noqa: E731
        on = [x for t in every for x in t["slot_counts"] if x >= t["threshold"]]
        off = [x for t in every for x in t["slot_counts"] if x < t["threshold"]]
        out[op] = {
            "trials_intact": len(it), "trials_all": len(every),
            "read_invariant": sum(1 for t in every if t["read_invariant"]),
            "latched_min": min(on), "latched_max": max(on), "unlatched_max": max(off),
            "empty_held": n_held(by["empty"]), "n_empty": len(by["empty"]),
            "empty_held_frac": round(n_held(by["empty"]) / len(by["empty"]), 4),
            "content_held": n_held(by["content"]), "n_content": len(by["content"]),
            "content_held_frac": round(n_held(by["content"]) / len(by["content"]), 4),
            "one_word_held": n_held(by["one_word"]), "co_commit": sum(1 for t in it if t["co_commit"]),
            "agree_old_latch_read": round(sum(1 for t in it if t["winner"] == t["old_latch_winner"]) / len(it), 4),
            "old_read_empty_held": n_held(by["empty"], "old_latch_winner"),
            "old_read_content_held": n_held(by["content"], "old_latch_winner"),
            "lesion_commits": {k: sum(1 for r in recs for t in r["lesions"][k] if t["winner"] is not None)
                               for k in LESIONS},
            "lesion_trials": {k: sum(len(r["lesions"][k]) for r in recs) for k in LESIONS},
            "seeds_all_pass": sum(1 for r in recs if r["all_pass"]),
            "live_q_speak": sum(1 for r in recs for t in r["intact"] if t["label"] == "hsb_q" and t["winner"] == BG.SPEAK),
        }
    return {"runner": "research.runners._bg_commit_derisk", "argv": list(sys.argv), "seeds": list(seeds),
            "backend": os.environ.get("SIM_BACKEND"), "rows": out,
            "note": "aggregates of the dev sweep files (no brain built); the sweep ran on numpy, CPU"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--summary", action="store_true", help="write <out-dir>/summary.json (per-row aggregates)")
    ap.add_argument("--value-report", action="store_true", help="reported value-lesion arm (op<I>_s<S>_value.json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--ops", type=int, nargs="+", default=None)
    ap.add_argument("--out-dir", type=str, default="research/findings/raw/_bg_commit/dev_organ")
    a = ap.parse_args(argv)
    if set(a.seeds) & set(GATE_SEEDS):
        raise SystemExit("dev sweep only: gate seeds are refused")
    if a.summary:
        r = summary(a.out_dir, a.seeds)
        json.dump(r, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=1)
        print(json.dumps(r["rows"]))
        return 0
    if a.select:
        r = select(a.out_dir, a.seeds)
        r.update({"runner": "research.runners._bg_commit_derisk", "argv": list(sys.argv),
                  "backend": os.environ.get("SIM_BACKEND"), "seeds": list(a.seeds),
                  "note": "reads the sweep files only (no brain built); the sweep ran on numpy, CPU"})
        print(json.dumps(r))
        json.dump(r, open(os.path.join(a.out_dir, "selection.json"), "w"), indent=1)
        return 0
    if a.value_report:
        os.makedirs(a.out_dir, exist_ok=True)
        for op in (a.ops if a.ops is not None else [BG.BG_COMMIT_OP]):
            for s in a.seeds:
                res = run_value(s, op)
                json.dump(res, open(os.path.join(a.out_dir, "op%d_s%d_value.json" % (op, s)), "w"), indent=1)
                print("value op%d s%d held=%s commit=%s inv=%s" % (op, s, res["held"], res["commit_rate"],
                                                                   res["read_invariant_all"]), flush=True)
        return 0
    if a.sweep:
        os.makedirs(a.out_dir, exist_ok=True)
        for op in (a.ops if a.ops is not None else range(len(BG.BG_COMMIT_GRID))):
            for s in a.seeds:
                p = os.path.join(a.out_dir, "op%d_s%d.json" % (op, s))
                if os.path.exists(p):
                    continue
                res = run_seed(s, op)
                json.dump(res, open(p, "w"), indent=1)
                print("op%d s%d all_pass=%s %s %s" % (op, s, res["all_pass"],
                                                      {k: v for k, v in res["checks"].items() if not v},
                                                      json.dumps(res["reported"]["held"])), flush=True)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
