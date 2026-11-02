"""bar-sanity spec (amendment A6, DRAFT 2026-11-01): Stage-2 C2's FAIL verdict, read as a capability claim.

decide = the real aggregate (`research.runners.stage2_exit_gate.aggregate` at level `bar_sanity_s2_common.LEVEL`) scores
group `infer` FAIL over six synthetic gate seeds. A FAIL says "the capability is absent"; this spec asks whether that
claim can be made of a brain that has the capability, and whether it is still made of brains that lack it.
  'perfect' kind  = a brain that LACKS a C2 capability (a FAIL is the right verdict; it must be FAIL >= 0.95):
                    the exposure-blind brain that takes the permuted teach and still answers the registered conclusion,
                    the one-hop and the unmarked-derivation brains, an inert chain-route lesion.
  'null' kind     = a brain the verdict FAIL would MISDESCRIBE (it must be FAIL <= 0.05): the capable brain whose
                    permuted hop teach was REFUSED (the 10-26 gate's shape: the D4 comprehension gate's clarification on
                    `t_heron_hunt`, nothing stored), and the fully capable brain.
Under the pre-A6 scorer (`LEVEL = "A5"`) the refused-teach brain is scored FAIL on every draw (TOOTHLESS: that is the 10-26
reason the RESULT doc's Claude amendment corrected); at A6 it is UNDEFINED, never FAIL and never PASS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402

NAME = "stage2-C2 infer FAIL-claim (A6)"
DESCRIPTION = ("Stage-2 C2 (inference on held-out), amendment A6: the real aggregate's `infer` verdict is FAIL. 'perfect' = a brain "
               "that lacks a C2 capability (FAIL is right); 'null' = a brain FAIL would misdescribe: the capable brain whose "
               "permuted hop teach was refused (the 10-26 gate), and the fully capable brain.")
SCORER_FILES = ["research/runners/stage2_exit_gate.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]


def decide(data):
    return C.decide_group_fail(data, "infer")


def _mk(**kw):
    return lambda rng: C.make_group(rng, "infer", **kw)


def _intact(v):
    return {"intact_a": v, "intact_b": v}


perfect = _mk(over={"permuted": "permuted+blind"})                 # exposure-blind, the teach taken
perfect_blind_exact = _mk(over={"permuted": "permuted+blind"}, realistic=False)
perfect_one_hop = _mk(over=_intact("one_hop"))
perfect_unmarked_derivation = _mk(over=_intact("no_flag"))
perfect_chain_route_inert = _mk(over={"lesion_chain_route": "intact"})
null = _mk(over={"permuted": "permuted+refused"})                  # capable; the permuted hop teach refused
null_refused_exact = _mk(over={"permuted": "permuted+refused"}, realistic=False)
null_capable = _mk()                                              # capable; the permuted teach received
