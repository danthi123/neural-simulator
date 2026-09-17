"""VERIFY: MultiTurnAgentV2's host anaphor word-list retirement is byte-identical (scaffold-retirement follow-on).

`multi_turn_agent_v2.MultiTurnAgentV2` used to gate its `_resolve()` pronoun check on a bare host Python set:
`_ANAPHORS = {"it","that","them","they","this"}; ... if word.lower() in _ANAPHORS`. This mirrors the IDENTICAL
shortcut `multi_turn_agent.MultiTurnAgent._resolve` had until its host fallback was DELETED 2026-09-16
(`2026-09-16-host-removal-novelty-anaphor-qroute-RETIRED-byte-identical-differential-GO.md`, which named V2's
own list as the un-retired follow-on). This runner is the byte-identical differential proving V2's retirement is
safe: the SAME spiking CA3 pattern-completion organ (`spiking_anaphor_detection_organ.SpikingAnaphorDetectorOrgan`)
now used in `MultiTurnAgentV2._anaphor_is` (verbatim mirror of `MultiTurnAgent._anaphor_is`) recognises EXACTLY the
tokens the retired host set did, on clean typed text, across all 6 canonical seeds.

METHOD. A frozen `_OldHostAnaphorV2` subclass reinstalls the EXACT retired host-list `_resolve` decision (the ONLY
method overridden) so the differential isolates the one changed thing -- everything else (the composer, the
order-encoded WM, hear/describe/narrate) is the SAME code both sides run. For each of the 6 seeds:
  1. TOKEN-LEVEL agreement: every probe token (the 5 retired anaphors + case/punctuation variants + non-anaphor
     content words + other function words) gets the identical is-anaphor verdict from the frozen host set and the
     live spiking organ.
  2. SCENARIO-LEVEL agreement: a fixed multi-turn script (hear -> pronoun what_does/who_does/is_it_true ->
     pronoun-cued reason_chain -> describe -> narrate) run through BOTH agents produces byte-identical results.

GO iff both hold on all 6 seeds (42, 43, 44, 100, 101, 102). NO sim/ edit; CPU/numpy backend (each op is a small
SimulationBridge, matching the production organ's own test posture).

Usage:
    SIM_BACKEND=numpy .venv/bin/python -m research.runners._v2_anaphor_retire_verify \\
        --out research/findings/raw/_v2_anaphor_retire_verify/differential_result.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent_v2 import MultiTurnAgentV2

SEEDS = [42, 43, 44, 100, 101, 102]
NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat", "see"]

# The EXACT retired host set (frozen here verbatim for the differential -- do NOT "improve" this copy; it must stay
# a faithful snapshot of what `multi_turn_agent_v2._ANAPHORS` used to be).
_RETIRED_HOST_ANAPHORS = {"it", "that", "them", "they", "this"}

# Probe tokens: the 5 retired anaphors, case/whitespace variants, non-anaphor content words (nouns + verbs), and
# other closed-class words the old set never covered (articles/other pronouns) -- so a false positive would show.
PROBE_TOKENS = [
    "it", "that", "them", "they", "this",              # the retired set, verbatim
    "It", "THAT", "They", "This", "Them",              # case variants (both paths lowercase before compare)
    "dog", "cat", "fish", "bird", "worm", "ball",       # referent nouns -- must NOT be anaphors
    "chase", "eat", "see",                              # action words -- must NOT be anaphors
    "the", "a", "he", "she", "there", "then",          # other function words -- must NOT be anaphors
]


class _OldHostAnaphorV2(MultiTurnAgentV2):
    """MultiTurnAgentV2 with ONLY the detection decision reverted to the frozen retired host-list test. Every other
    method (hear/what_does/who_does/is_it_true/reason_chain/describe/narrate, the order-encoded WM) is the
    UNCHANGED production code both sides run -- the differential isolates exactly the one retired decision."""

    def _anaphor_is(self, word):
        return isinstance(word, str) and word.lower() in _RETIRED_HOST_ANAPHORS


def _build(cls, seed):
    a = cls(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=seed)
    c = a.agent.composer
    c.store("cat", "eat", "fish")
    c.store("dog", "eat", "worm")
    c.store("fish", "eat", "worm")
    c.store("bird", "eat", "ball")
    return a


def _run_scenario(cls, seed):
    """A fixed multi-turn script exercising every pronoun-resolution call site. Returns a JSON-serializable dict."""
    a = _build(cls, seed)
    out = {}
    a.hear("dog see cat")                                    # cat most-recent
    out["most_recent_after_hear"] = a.most_recent_referent()
    out["what_does_it_eat"] = a.what_does("it", "eat")        # 'it' -> cat -> fish
    out["who_eats_it"] = a.who_does("eat", "it")              # 'it' -> cat -> who eats cat? (none stored -> None)
    out["is_it_true_it_eat_fish"] = a.is_it_true("it", "eat", "fish")
    out["reason_chain_it_eat_eat"] = a.reason_chain("it", ["eat", "eat"])   # cat -eat-> fish -eat-> worm
    out["describe_it"] = a.describe("it")
    out["narrate_surface"] = a.narrate(["cat", "dog", "fish"])
    # A second independent agent: empty discourse -> the no-confab moat must hold identically both ways.
    b = _build(cls, seed)
    out["moat_empty_most_recent"] = b.most_recent_referent()
    out["moat_empty_what_does"] = b.what_does("it", "eat")
    out["moat_empty_is_it_true"] = b.is_it_true("it", "eat", "fish")
    # Non-anaphor pronoun-shaped probe (a token neither path treats as an anaphor) -- should pass through literally.
    c = _build(cls, seed)
    c.hear("dog see cat")
    out["passthrough_content_word"] = c.what_does("dog", "eat")   # 'dog' is not a pronoun -> resolves to itself
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="research/findings/raw/_v2_anaphor_retire_verify/differential_result.json")
    args = p.parse_args()

    per_seed = {}
    all_token_clean = True
    all_scenario_clean = True
    for seed in SEEDS:
        new_agent = _build(MultiTurnAgentV2, seed)
        token_mismatches = []
        for tok in PROBE_TOKENS:
            old_verdict = isinstance(tok, str) and tok.lower() in _RETIRED_HOST_ANAPHORS
            new_verdict = new_agent._anaphor_is(tok)
            if old_verdict != new_verdict:
                token_mismatches.append({"token": tok, "old": old_verdict, "new": new_verdict})
        if token_mismatches:
            all_token_clean = False

        old_scn = _run_scenario(_OldHostAnaphorV2, seed)
        new_scn = _run_scenario(MultiTurnAgentV2, seed)
        scenario_identical = (old_scn == new_scn)
        if not scenario_identical:
            all_scenario_clean = False

        per_seed[str(seed)] = {
            "token_mismatches": token_mismatches,
            "token_level_identical": (not token_mismatches),
            "scenario_identical": scenario_identical,
            "old_scenario": old_scn,
            "new_scenario": new_scn,
        }
        print(f"seed {seed}: token_identical={not token_mismatches} scenario_identical={scenario_identical}")

    go = bool(all_token_clean and all_scenario_clean)
    result = {
        "mechanism": "multi_turn_agent_v2-host-anaphor-list-retirement",
        "seeds": SEEDS,
        "probe_tokens": PROBE_TOKENS,
        "byte_identical": go,
        "all_token_level_identical": all_token_clean,
        "all_scenario_identical": all_scenario_clean,
        "per_seed": per_seed,
        "verdict": "GO" if go else "NO-GO",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nVERDICT: {result['verdict']} (byte_identical={go}) -- wrote {args.out}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
