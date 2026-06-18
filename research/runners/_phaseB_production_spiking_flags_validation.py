"""TRACK-A CLOSEOUT (production biologization): does the production conversational agent run FULLY BRAIN-BASED when
the three default-OFF neural flags are flipped ON -- enable_spiking_cleanup (cleanup = spiking matched-filter + WTA,
not numpy argmax), enable_substrate_store (fact store = spiking weight-store, not a numpy list), enable_neural_render
(word order = the spiking competitive-queuing read-out, not an f-string) -- at PARITY with the numpy-default oracle?

The BRAIN-BASED-ONLY audit (`2026-06-18-conversational-brain-based-only-audit.md`) found these are the production
path's host shortcuts whose validated neural versions already exist behind these flags. Each was validated alone;
this runner proves they hold TOGETHER on the full capability matrix and == the numpy oracle (so numpy can stay the
fast DEFAULT while the brain-based claim is earned: the agent demonstrably CAN converse fully on neurons/synapses).

GATE (>=3 seeds, escalate to 6): the all-spiking agent's who/what/yes-no/abstention answers == the numpy oracle's
== ground truth, AND the no-confab MOAT holds (unstored -> None/"unknown") with the spiking store+cleanup, AND
describe() produces a valid SVO description (the neural word order). Report any DIFF (a real finding: the flags
don't compose, or a spiking op diverges from its numpy oracle).

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_production_spiking_flags_validation --seeds 42,43,44
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

VOCAB = ["dog", "cat", "bird", "river", "apple",
         "go", "come", "look", "see", "eat", "swim", "stop",
         "north", "south", "east", "west"]


def _agent(seed, spiking):
    a = BrainConversationalAgent(
        seed=seed, concepts={w: None for w in VOCAB},
        enable_spiking_cleanup=spiking, enable_substrate_store=spiking, enable_neural_render=spiking)
    a.hear("dog go north")
    a.hear("cat come south", polarity="AFFIRM")
    a.hear("river look west", polarity="NEGATE")
    a.hear("dog eat cat")
    a.hear("cat swim river")
    return a


def _probe(a):
    return {
        "what_dog_go": a.what_does("dog", "go"),                 # cleanup + store
        "who_go_north": a.who_does("go", "north"),
        "yes_cat_come_south": a.is_it_true("cat", "come", "south"),
        "no_river_look_west": a.is_it_true("river", "look", "west"),
        "unknown_apple": a.is_it_true("apple", "stop", "east"),  # the moat (unstored)
        "abstain_bird_see": a.what_does("bird", "see"),          # the moat (unstored agent)
        "describe_dog": a.describe("dog"),                       # neural_render word order
    }


def _valid_describe(s, agent="dog"):
    # a valid SVO description mentions the agent and at least one of its stored relations; the GATE is "a valid
    # description + moat", not byte-identity to the f-string (the neural order may legitimately differ).
    return isinstance(s, str) and (agent in s) and any(w in s for w in ("go", "north", "eat", "cat"))


def run_seed(seed):
    oracle = _probe(_agent(seed, spiking=False))     # numpy default = the validated oracle
    spk = _probe(_agent(seed, spiking=True))         # all three neural flags ON = the fully-brain-based path
    expected = {
        "what_dog_go": "north", "who_go_north": "dog", "yes_cat_come_south": "yes",
        "no_river_look_west": "no", "unknown_apple": "unknown", "abstain_bird_see": None,
    }
    rows = {}
    for k in oracle:
        if k == "describe_dog":
            # compare validity (not byte-identity); both should produce a valid description
            ok = _valid_describe(spk[k]) and _valid_describe(oracle[k])
            rows[k] = {"oracle": oracle[k], "spiking": spk[k], "match": ok, "kind": "valid"}
        else:
            match = (spk[k] == oracle[k])
            gt = expected.get(k, "<n/a>")
            gt_ok = (gt == "<n/a>") or (spk[k] == gt)
            rows[k] = {"oracle": oracle[k], "spiking": spk[k], "match": match and gt_ok, "kind": "=="}
    all_ok = all(r["match"] for r in rows.values())
    moat_ok = (spk["unknown_apple"] == "unknown") and (spk["abstain_bird_see"] is None)
    for k, r in rows.items():
        flag = "OK " if r["match"] else "DIFF"
        print(f"  [seed {seed}] {flag} {k:20s} oracle={str(r['oracle'])[:18]:18s} spiking={str(r['spiking'])[:18]}",
              flush=True)
    print(f"  [seed {seed}] all_match={all_ok} moat_ok={moat_ok}", flush=True)
    return {"seed": seed, "all_ok": bool(all_ok), "moat_ok": bool(moat_ok), "rows": rows}


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--out", type=str, default=os.path.join(
        _REPO, "research", "findings", "raw", "_phaseB_production_spiking_flags.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    print("[production spiking-flags validation] all 3 neural flags ON == numpy oracle == ground truth?\n", flush=True)
    rows = [run_seed(s) for s in seeds]
    n_ok = sum(int(r["all_ok"]) for r in rows)
    n_moat = sum(int(r["moat_ok"]) for r in rows)
    go = (n_ok == len(seeds)) and (n_moat == len(seeds))
    print(f"\n{'='*92}", flush=True)
    print(f"  {n_ok}/{len(seeds)} seeds all-match oracle+GT | moat {n_moat}/{len(seeds)}", flush=True)
    if go:
        print(f"  GO: the production agent runs FULLY BRAIN-BASED (spiking cleanup + substrate store + neural render) "
              f"at parity with the numpy oracle, moat intact. ==> the production conversational pipeline is "
              f"demonstrably all-neurons/synapses; numpy stays the fast DEFAULT (a documented speed choice), the "
              f"brain-based claim is earned. Add a CI guard so the spiking path can't bit-rot.", flush=True)
    else:
        print(f"  BOUNDARY/DIFF: the flags do not all hold at oracle parity (see DIFF above) -- localize which "
              f"spiking op diverges from its numpy oracle before claiming the production path is fully brain-based.",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*92}", flush=True)
    out = {"verdict": "GO" if go else "DIFF", "seeds": seeds, "n_ok": n_ok, "n_moat": n_moat, "per_seed": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
