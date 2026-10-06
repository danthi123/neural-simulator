"""Word-role COMMIT organ battery: the spiking commit stage decides each word's role (packet word-role-commit).

Plan: docs/drafts/spiking-commit/PLAN.md (application A1). Preregistration:
research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md. The organ is `WordRoleParserOrgan` built with
`commit=` params (research/runners/word_role_parser_organ.py, COMMIT_GRID): per role a commit gate (self-recurrent,
mutually exclusive across roles) and a role-slot latch (bistable, carried across the words of one parse). A word's
role is the slot latch that turned on while it was presented; the first word to fill a role keeps it. The host
ROLE_MIN / ROLE_MARGIN rule and the per-role argmax over words are not called on this path.

BATTERY (all must hold for `all_pass`), on a fresh organ trained on the count organ's BATTERY_FACTS:
  svo, self, adverb, swapped, novel, lesion_self   the six cases of the word-role-parse battery, same wants;
  commit_lesion_self   role -> commit-gate transmission closed: parse(brain, use) declines (the role ensembles still
                       fire; only the commit stage is cut);
  latch_lesion_self    slot-latch recurrence closed: parse(brain, use) declines (a slot that cannot sustain itself
                       after the word ends is not read as committed: persistence carries the decision);
  read_invariant       on every case, re-reading the same spikes at READ_FRAC x0.5 and x2 gives the same parse.
DIAGNOSTICS (never in all_pass): dual-role words cat/fish at positions 0 and 2; the host count rule re-applied to the
same run's role-ensemble counts (paired baseline); the exclusion lesion's co-commit count on the dual-role reads;
latched-minimum and unlatched-maximum slot counts; co-commit and re-latch totals.

CLI mirrors word_role_parser_organ.py: `--sweep --seeds S --out-dir D` writes op<I>_s<S>.json for every COMMIT_GRID
row; `--seed S --op I --out F [--allow-gate-seed --prereg P]` writes one battery. Gate seeds are refused without
`--allow-gate-seed --prereg`. Backend: process backend (numpy on the pool). No `sim/` edit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import word_role_parser_organ as W  # noqa: E402
from tools.lab import lever  # noqa: E402

GATE_SEEDS = W.GATE_SEEDS
DEV_SEEDS = W.DEV_SEEDS


def host_rule_parse(organ, words: Sequence[str], counts: Dict[str, Dict[str, int]]) -> List[Optional[str]]:
    """The count organ's host read (ROLE_MIN / ROLE_MARGIN, then the per-role argmax) applied to the role-ensemble
    counts this run recorded: the paired baseline. Diagnostics only; the commit path never calls it."""
    roles = {w: organ._rule(counts[w]) for w in words}
    best = {}
    for w in words:
        r = roles[w]
        if r is not None and (r not in best or counts[w][r] > counts[best[r]][r]):
            best[r] = w
    return [best.get("agent"), best.get("action")]


# The battery's cases: (name, content words, wanted (agent, action) or None for a decline, parse lesions).
CASES = (
    ("svo", ("dog", "chase"), ("dog", "chase"), {}),
    ("self", ("brain", "use"), ("brain", "use"), {}),
    ("adverb", ("brain", "usually", "use"), ("brain", "use"), {}),
    ("swapped", ("chase", "dog"), ("dog", "chase"), {}),
    ("novel", ("zorp", "glim"), None, {}),
    ("lesion_self", ("brain", "use"), None, {"lesion": True}),
    ("commit_lesion_self", ("brain", "use"), None, {"commit_lesion": True}),
    ("latch_lesion_self", ("brain", "use"), None, {"latch_lesion": True}),
)


def case_record(organ, content, want, r, lesioned: bool) -> dict:
    """One battery case from a commit parse result `r` (pure). `pass` = the parse equals the want (or declines)."""
    cm = r["commit"]
    got = (r["agent"], r["action"])
    return {"content": list(content), "want": list(want) if want is not None else None,
            "got": list(got) if got != (None, None) else None, "roles": r["roles"], "counts": r["counts"],
            "slot_counts": cm["slot_counts"], "gate_counts": cm["gate_counts"], "per_item": cm["per_item"],
            "read_variants": cm["read_variants"], "read_invariant": bool(cm["read_invariant"]),
            "threshold": cm["threshold"], "co_commit": int(cm["co_commit"]), "relatch": int(cm["relatch"]),
            "host_rule": None if lesioned else host_rule_parse(organ, content, r["counts"]),
            "pass": bool((got == tuple(want)) if want is not None else got == (None, None))}


def battery_verdict(cases: Dict[str, dict]) -> dict:
    """The registered organ bar for one seed (pure): every case passes AND every case's read is threshold-invariant
    (re-read at READ_FRAC x0.5 and x2 gives the same parse). Returns {read_invariant, all_pass}."""
    inv = all(c["read_invariant"] for c in cases.values())
    return {"read_invariant": bool(inv), "all_pass": bool(inv and all(c["pass"] for c in cases.values()))}


def commit_battery(seed: int, op_index: int) -> dict:
    params, commit = W.commit_organ_params(op_index)
    W.set_commit_operating_point(op_index)
    organ = W.WordRoleParserOrgan(W.BATTERY_FACTS, seed=seed, params=params, commit=commit).train()
    cases = {name: case_record(organ, content, want, organ.parse(list(content), **kw), bool(kw))
             for name, content, want, kw in CASES}
    lat_min, off_max = None, 0
    for c in cases.values():
        for sc in c["slot_counts"]:
            for v in sc.values():
                if v >= c["threshold"]:
                    lat_min = v if lat_min is None else min(lat_min, v)
                else:
                    off_max = max(off_max, v)
    co_total = sum(c["co_commit"] for c in cases.values())
    relatch_total = sum(c["relatch"] for c in cases.values())
    bv = battery_verdict(cases)
    lesion_moved = bool(lever("role -> commit gate (brain, use)", cases["self"]["got"],
                              cases["commit_lesion_self"]["got"], required=False))
    diag = {"dual_role": {}, "dual_role_excl_lesion": {}}
    try:                                                   # diagnostics only: never part of all_pass, never raise
        for w, k, want in (("cat", 0, "agent"), ("cat", 2, "patient"), ("fish", 0, "agent"), ("fish", 2, "patient")):
            role, d = organ.commit_read_word(w, k)
            diag["dual_role"]["%s@%d" % (w, k)] = {"want": want, "got": role, "ok": role == want,
                                                   "slot_counts": d["items"][0]["slot_counts"]}
            items = W._isolated(lambda w=w, k=k: organ._commit_run([(w, k)], excl_lesion=True))
            rd = organ._commit_read(items)
            diag["dual_role_excl_lesion"]["%s@%d" % (w, k)] = {"new": rd["per_item"][0]["new"],
                                                               "co_commit": rd["co_commit"]}
        diag["dual_role_ok"] = int(sum(v["ok"] for v in diag["dual_role"].values()))
        diag["excl_lesion_co_commit"] = int(sum(v["co_commit"] for v in diag["dual_role_excl_lesion"].values()))
        intact = ("svo", "self", "adverb", "swapped", "novel")
        diag["host_rule_agrees"] = {k: (cases[k]["host_rule"] == [cases[k]["got"][0], cases[k]["got"][1]]
                                        if cases[k]["got"] else cases[k]["host_rule"] == [None, None])
                                    for k in intact}
    except Exception as e:                                 # noqa: BLE001
        diag["error"] = repr(e)
    diag["latched_min"] = lat_min
    diag["unlatched_max"] = off_max
    diag["co_commit_total"] = co_total
    diag["relatch_total"] = relatch_total
    try:
        backend = W.get_backend()[0].__name__
    except Exception:
        backend = "unknown"
    return {"seed": int(seed), "op_index": int(op_index), "mode": "commit", "params": params, "commit": commit,
            "cases": cases, "read_invariant": bv["read_invariant"], "all_pass": bv["all_pass"], "backend": backend,
            "weights": organ.weight_summary, "diagnostics": diag, "commit_lesion_moved_parse": lesion_moved}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEV_SEEDS))
    ap.add_argument("--op", type=int, default=0, help="COMMIT_GRID index")
    ap.add_argument("--ops", type=int, nargs="*", default=None, help="COMMIT_GRID indices for --sweep (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--allow-gate-seed", action="store_true")
    ap.add_argument("--prereg", default=None)
    a = ap.parse_args(argv)
    seeds = a.seeds if a.sweep else [a.seed]
    for s in seeds:
        W._check_seed(s, a.allow_gate_seed, a.prereg)
    if a.sweep:
        out_dir = a.out_dir or os.path.join(_REPO, "research", "findings", "raw", "word_role_commit")
        for op in (a.ops if a.ops is not None else range(len(W.COMMIT_GRID))):
            for s in seeds:
                res = commit_battery(s, op)
                W._write(os.path.join(out_dir, "op%d_s%d.json" % (op, s)), res)
                print("op%d seed%d all_pass=%s %s" % (op, s, res["all_pass"],
                                                       {k: v["pass"] for k, v in res["cases"].items()}))
        return 0
    res = commit_battery(a.seed, a.op)
    if a.prereg:
        res["prereg"] = a.prereg      # tools/s2p_score.py step O requires the registered prereg path here
    if a.out:
        W._write(a.out, res)
    print(json.dumps({k: res[k] for k in ("seed", "op_index", "all_pass")}),
          {k: v["pass"] for k, v in res["cases"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
