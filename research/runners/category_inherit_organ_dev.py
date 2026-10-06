"""Organ-level DEV read of the category-inheritance organ (research/runners/category_inherit_organ.py) on the Stage-1
C1 fact sequence, fed directly (no chat, no brain): the numbers the prereg's section 11 quotes for dev seeds.
DEV SEEDS ONLY (1-12); a gate seed is refused. Not a verdict: an integrity and design read before the gate.

Per seed: intact (the canary streamed with the bird motion frame), permuted (with the fish motion frame), the read
lesion `all`, the read lesion `lateral`, the encoding ablation `lateral_learn`, a same-seed rebuild (determinism), a
bridging member (`bat` eats the bug and flies in the sky), and the C2/C3 turn shapes (never-taught (agent, verb)
pairs must abstain). Each read records the admitted frame rates and the organ's decision.

  .venv/bin/python -m research.runners.category_inherit_organ_dev --seeds 1,2,3,4,5,6,7,8 \
      --out research/findings/raw/_stage1_c1_category/dev/organ_dev_seeds1-8.json
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
    return out


def run_seed(seed):
    t0 = time.time()
    canary = [("canary", "fly", "sky")]
    row = {"seed": int(seed)}
    o = _organ(seed, C1_FACTS + canary)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8", help="comma-separated DEV seeds (1-12)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    bad = [s for s in seeds if s in GATE_SEEDS]
    if bad:
        sys.exit("gate seed(s) %s refused: this is a dev read" % bad)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    rows = []
    for s in seeds:
        rows.append(run_seed(s))
        print("[organ-dev] seed %d: canary/eat -> %s (%.0fs)" % (s, rows[-1]["intact"]["canary/eat"]["patient"],
                                                                 rows[-1]["wall_s"]), flush=True)
    out = {"runner": "research/runners/category_inherit_organ_dev.py", "backend": os.environ.get("SIM_BACKEND"),
           "note": "dev seeds only; organ fed the Stage-1 C1 facts directly; not a verdict",
           "summary": summarize(rows), "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out["summary"], indent=1, default=str))


if __name__ == "__main__":
    main()
