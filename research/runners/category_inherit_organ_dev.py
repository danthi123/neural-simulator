"""Organ-level DEV read of the category-inheritance organ (research/runners/category_inherit_organ.py) on the Stage-1
C1 fact sequence, fed directly (no chat, no brain): the numbers the prereg's section 11 quotes for dev seeds.
DEV SEEDS ONLY (1-12); a gate seed is refused. Not a verdict: an integrity and design read before the gate.

Per seed: intact (the canary streamed with the bird motion frame), permuted (with the fish motion frame), the read
lesion `all`, the read lesion `lateral`, the encoding ablation `lateral_learn`, a same-seed rebuild (determinism), a
bridging member (`bat` eats the bug and flies in the sky), and the C2/C3 turn shapes (never-taught (agent, verb)
pairs must abstain). Each read records the admitted frame rates and the organ's decision.

  .venv/bin/python -m research.runners.category_inherit_organ_dev --seeds 1,2,3,4,5,6,7,8 \
      --out research/findings/raw/_stage1_c1_category/dev/organ_dev_seeds1-8.json

HOP ATTENUATION DEV BAR (`--hop`: every organ built with BRAIN_CATEGORY_INHERIT_HOP=1; 2026-10-06). The battery adds
the bridge with the canary heard BEFORE the bat (`bridge_bat_canary_first`), and `hop_dev_verdict` scores it: a dev
seed passes iff the C1 canary still guesses worm, the permuted canary guesses bug, the read lesion abstains with the
cut held, the C2/C3 shapes abstain, and in BOTH bridge orders the canary's `swim` abstains and its `eat` guesses worm.
PASS iff every seed of HOP_DEV_SEEDS (dev 1-16) passes (bar-sanity:
docs/drafts/stage1-c1-inheritance/bar_sanity_record_hop_dev.json).

  .venv/bin/python -m research.runners.category_inherit_organ_dev --hop --seeds 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
      --out research/findings/raw/_stage1_c1_category/dev/organ_dev_hop_seeds1-16.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

C1_FACTS = [("robin", "eat", "worm"), ("trout", "eat", "bug"), ("robin", "fly", "sky"), ("trout", "swim", "lake"),
            ("sparrow", "eat", "worm"), ("salmon", "eat", "bug"), ("sparrow", "fly", "sky"),
            ("salmon", "swim", "lake"), ("finch", "eat", "worm"), ("finch", "fly", "sky")]
GATE_SEEDS = (42, 43, 44, 100, 101, 102)


def _organ(seed, facts, lesion=None):
    from research.runners.category_inherit_organ import CategoryInheritOrgan
    o = CategoryInheritOrgan(seed=seed, lesion=lesion)
    for f in facts:
        o.observe(*f)
    return o


def _q(o, pairs):
    out = {}
    for a, v in pairs:
        r = o.infer(a, v)
        out["%s/%s" % (a, v)] = {"patient": r["patient"], "reason": r["reason"], "admitted": r["admitted"],
                                 "lesion": r.get("lesion"), "lesion_held": r.get("lesion_held"),
                                 "attribution": r.get("attribution")}
        if "hop" in r:                                  # recorded only with the option on (off rows: the 09-30 format)
            out["%s/%s" % (a, v)].update(hop=r["hop"], query_steps=r.get("query_steps"))
    return out


def run_seed(seed):
    t0 = time.time()
    canary = [("canary", "fly", "sky")]
    row = {"seed": int(seed)}
    o = _organ(seed, C1_FACTS + canary)
    row["hop"] = bool(o.hop)
    row["intact"] = _q(o, [("canary", "eat"), ("canary", "swim"), ("robin", "eat"), ("trout", "fly")])
    audit = o.weight_audit()
    row["weight_audit"] = {"canary": audit["member_frame"]["canary"], "robin": audit["member_frame"]["robin"],
                           "lateral_fly_sky": audit["lateral"]["fly sky"],
                           "lateral_eat_worm": audit["lateral"]["eat worm"]}
    for mode in ("all", "lateral"):
        o.lesion = mode
        row["read_lesion_" + mode] = _q(o, [("canary", "eat")])
    o.lesion = None
    o2 = _organ(seed, C1_FACTS + canary)
    row["determinism_same_seed_read_equal"] = (o.read("canary") == o2.read("canary"))
    row["permuted"] = _q(_organ(seed, C1_FACTS + [("canary", "swim", "lake")]), [("canary", "eat")])
    row["lateral_learn"] = _q(_organ(seed, C1_FACTS + canary, lesion="lateral_learn"), [("canary", "eat")])
    row["bridge_bat"] = _q(_organ(seed, C1_FACTS + [("bat", "eat", "bug"), ("bat", "fly", "sky")] + canary),
                           [("canary", "eat"), ("canary", "swim"), ("trout", "fly")])
    row["bridge_bat_canary_first"] = _q(_organ(seed, C1_FACTS + canary + [("bat", "eat", "bug"), ("bat", "fly", "sky")]),
                                        [("canary", "eat"), ("canary", "swim"), ("trout", "fly")])
    row["stage1_c2_c3_shapes"] = _q(_organ(seed, [("otter", "eat", "clam"), ("bear", "chase", "otter"),
                                                  ("wolf", "bite", "apple")]),
                                    [("otter", "chase"), ("clam", "eat"), ("wolf", "chase"), ("bear", "eat")])
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def summarize(rows):
    def rate(r, key, obj):
        return dict(r[key]["canary/eat"]["admitted"]).get(obj)
    s = {"n_seeds": len(rows), "seeds": [r["seed"] for r in rows]}
    s["intact_worm_rate"] = [rate(r, "intact", "worm") for r in rows]
    s["intact_bug_rate"] = [rate(r, "intact", "bug") for r in rows]
    s["intact_patient"] = [r["intact"]["canary/eat"]["patient"] for r in rows]
    s["permuted_bug_rate"] = [rate(r, "permuted", "bug") for r in rows]
    s["permuted_worm_rate"] = [rate(r, "permuted", "worm") for r in rows]
    s["permuted_patient"] = [r["permuted"]["canary/eat"]["patient"] for r in rows]
    s["read_lesion_all_patient"] = [r["read_lesion_all"]["canary/eat"]["patient"] for r in rows]
    s["read_lesion_lateral_worm_rate"] = [rate(r, "read_lesion_lateral", "worm") for r in rows]
    s["lateral_learn_patient"] = [r["lateral_learn"]["canary/eat"]["patient"] for r in rows]
    s["determinism"] = [r["determinism_same_seed_read_equal"] for r in rows]
    s["bridge_canary_eat"] = [r["bridge_bat"]["canary/eat"]["reason"] + ":" + str(r["bridge_bat"]["canary/eat"]["patient"])
                              for r in rows]
    s["bridge_canary_swim"] = [r["bridge_bat"]["canary/swim"]["patient"] for r in rows]
    s["bridge_trout_fly"] = [r["bridge_bat"]["trout/fly"]["patient"] for r in rows]
    if all("bridge_bat_canary_first" in r for r in rows):
        b2 = [r["bridge_bat_canary_first"] for r in rows]
        s["bridge_canary_first_canary_eat"] = [b["canary/eat"]["reason"] + ":" + str(b["canary/eat"]["patient"]) for b in b2]
        s["bridge_canary_first_canary_swim"] = [b["canary/swim"]["patient"] for b in b2]
        s["bridge_canary_first_trout_fly"] = [b["trout/fly"]["patient"] for b in b2]
    s["c2_c3_shapes_all_abstain"] = [all(v["patient"] is None for v in r["stage1_c2_c3_shapes"].values()) for r in rows]
    s["canary_member_frame_eat_worm_w"] = [r["weight_audit"]["canary"]["eat worm"] for r in rows]
    s["canary_member_frame_fly_sky_w"] = [r["weight_audit"]["canary"]["fly sky"] for r in rows]
    s["lateral_fly_sky_eat_worm_w"] = [r["weight_audit"]["lateral_fly_sky"]["eat worm"] for r in rows]
    s["attribution_lateral_at_read"] = [(r["intact"]["canary/eat"]["attribution"] or {}).get("lateral_at_read")
                                        for r in rows]
    # WHOSE is the canary's `eat worm` rate? tools.lab.attributable_to against each control arm (treatment = the
    # intact rate; control = the same frame's rate in the arm): 1.0 = none of it survives the manipulation.
    from tools.lab import attributable_to
    for key, arm in (("read_lesion_all", "read_lesion_all"), ("lateral_learn", "lateral_learn"),
                     ("permuted_exposure", "permuted")):
        s["attributable_worm_rate_vs_" + key] = [
            attributable_to("seed %d canary eat-worm rate, intact vs %s" % (r["seed"], key),
                            rate(r, "intact", "worm") or 0.0, rate(r, arm, "worm") or 0.0) for r in rows]
    return s


def hop_seed_problems(r):
    """The hop dev bar for one seed row (run_seed's format). Returns the list of failed clauses ([] = pass)."""
    bad = []
    def p(arm, q):
        return ((r.get(arm) or {}).get(q) or {}).get("patient")
    if p("intact", "canary/eat") != "worm":
        bad.append("C1 canary eat is %r, not worm" % p("intact", "canary/eat"))
    if p("permuted", "canary/eat") != "bug":
        bad.append("permuted canary eat is %r, not bug" % p("permuted", "canary/eat"))
    les = (r.get("read_lesion_all") or {}).get("canary/eat") or {}
    if les.get("patient") is not None or les.get("lesion") != "all" or les.get("lesion_held") is not True:
        bad.append("read lesion not an engaged abstain: %r" % ({k: les.get(k) for k in ("patient", "lesion", "lesion_held")},))
    if any(v.get("patient") is not None for v in (r.get("stage1_c2_c3_shapes") or {"missing": {"patient": "?"}}).values()):
        bad.append("a C2/C3 moat shape answered")
    for arm in ("bridge_bat", "bridge_bat_canary_first"):
        if arm not in r:
            bad.append("%s missing" % arm)
            continue
        if p(arm, "canary/swim") is not None:
            bad.append("%s: canary swim guesses %r" % (arm, p(arm, "canary/swim")))
        if p(arm, "canary/eat") != "worm":
            bad.append("%s: canary eat is %r, not worm" % (arm, p(arm, "canary/eat")))
    return bad


HOP_DEV_SEEDS = tuple(range(1, 17))      # the hop dev bar's seeds (dev only; 1-8 first, 9-16 added by verify-go)


def hop_dev_verdict(rows, seeds=HOP_DEV_SEEDS):
    """The hop dev bar: PASS iff every dev seed passes `hop_seed_problems`; UNDEFINED if a seed is missing or any
    row did not run with hop attenuation on (`hop` true in the row)."""
    by = {int(r.get("seed", -1)): r for r in rows}
    missing = [s for s in seeds if s not in by]
    not_hop = [s for s in seeds if s in by and by[s].get("hop") is not True]
    if missing or not_hop:
        return {"verdict": "UNDEFINED", "missing": missing, "not_hop": not_hop}
    per = {s: hop_seed_problems(by[s]) for s in seeds}
    n_pass = sum(1 for v in per.values() if not v)
    return {"verdict": "PASS" if n_pass == len(seeds) else "FAIL", "n_pass": n_pass, "n": len(seeds),
            "why": {s: v for s, v in per.items() if v}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8", help="comma-separated DEV seeds (never a gate seed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hop", action="store_true", help="build every organ with BRAIN_CATEGORY_INHERIT_HOP=1 and "
                    "score the hop dev bar (hop_dev_verdict)")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    bad = [s for s in seeds if s in GATE_SEEDS]
    if bad:
        sys.exit("gate seed(s) %s refused: this is a dev read" % bad)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    if a.hop:
        os.environ["BRAIN_CATEGORY_INHERIT_HOP"] = "1"
    rows = []
    for s in seeds:
        rows.append(run_seed(s))
        print("[organ-dev] seed %d: canary/eat -> %s (%.0fs)" % (s, rows[-1]["intact"]["canary/eat"]["patient"],
                                                                 rows[-1]["wall_s"]), flush=True)
    out = {"runner": "research/runners/category_inherit_organ_dev.py", "backend": os.environ.get("SIM_BACKEND"),
           "note": "dev seeds only; organ fed the Stage-1 C1 facts directly; not a verdict",
           "hop": bool(a.hop), "summary": summarize(rows), "rows": rows}
    if a.hop:
        out["hop_dev_verdict"] = hop_dev_verdict(rows, seeds=tuple(seeds))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out["summary"], indent=1, default=str))
    if a.hop:
        print("[organ-dev] hop dev bar: %s" % json.dumps(out["hop_dev_verdict"], default=str))


if __name__ == "__main__":
    main()
