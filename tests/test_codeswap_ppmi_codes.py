"""CI GUARD for shortcut #12 codes-half (option 2): swapping the production conversational composer from CURATED
(composer-self-generated random) codes to LEARNED codes (the PPMI stream cortex's grounded_codes path) must keep the
who/what/yes-no answers IDENTICAL and the no-confab moat at 0 false-accepts.

The owner's decision (2026-06-20-fhrr-frontier-decision-scoping.md, path B2): retire the "curated codes" label — the
production conversation runs on codes it LEARNED FROM CONVERSATION — without changing answers, and NEVER weakening the
moat. The full-scale 320-concept A/B lives in research/runners/_codeswap_12codes_ppmi_ab.py (needs the cached PPMI
.npy + GPU for the onebrain path); THIS test pins the same invariant at a small vocab on CPU (the rf composer + the
existing grounded_codes plumbing) so it runs in CI and the swap can't silently bit-rot.

CPU-only (rf composer, D=64 RF ops are tiny). No GPU, no concept-code cache needed (the codes are generated here)."""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402

VOCAB = ["dog", "cat", "bird", "fish", "apple", "ball", "tree", "river",
         "eat", "play", "sleep", "swim", "go", "look", "north", "south"]

# affirmative SVO facts (all words in VOCAB) + one negated fact for the yes/no "no" path.
FACTS = [
    ("dog", "eat", "apple"),
    ("cat", "play", "ball"),
    ("bird", "sleep", "tree"),
    ("fish", "swim", "river"),
]
NEG_FACT = ("dog", "look", "north")
# unstored cues that MUST abstain (real words, never stored together).
ABSENT_WHAT = [("dog", "play"), ("cat", "eat"), ("bird", "swim"), ("fish", "sleep")]
ABSENT_WHO = [("eat", "ball"), ("play", "apple"), ("swim", "tree")]


def _learned_codes(seed):
    """Synthetic LEARNED codes for the vocab: per-word phase codes from a distinct RNG (standing in for the PPMI
    stream cortex's learned grounded codes). They differ from the composer's own seed-derived random codes, so the
    A/B genuinely compares 'curated' vs 'learned' rather than the same codes twice. Length 128 = the production
    BrainConversationalAgent's rf composer D (it builds RFPhasorComposer with D=128)."""
    rng = np.random.default_rng(seed * 104729 + 7)
    return {w: rng.uniform(0.0, 1.0, 128) for w in VOCAB}


def _drive(agent):
    """Hear the facts (affirmative + one negated) and return all who/what/yes-no answers + the moat false-accept count.
    Identical calls for both composers so the A/B is like-for-like."""
    for a, v, o in FACTS:
        agent.hear(f"{a} {v} {o}", polarity="AFFIRM")
    agent.hear(f"{NEG_FACT[0]} {NEG_FACT[1]} {NEG_FACT[2]}", polarity="NEGATE")

    ans = {}
    for a, v, o in FACTS:
        ans[("what", a, v)] = agent.what_does(a, v)
        ans[("who", v, o)] = agent.who_does(v, o)
    ans[("yn", *FACTS[0])] = agent.is_it_true(*FACTS[0])           # expect "yes"
    ans[("yn", *NEG_FACT)] = agent.is_it_true(*NEG_FACT)           # expect "no"
    ans[("yn", "cat", "go", "south")] = agent.is_it_true("cat", "go", "south")  # never stored -> not "yes"

    false_accept = 0
    for a, v in ABSENT_WHAT:
        r = agent.what_does(a, v)
        ans[("absent_what", a, v)] = r
        false_accept += int(r is not None)
    for v, o in ABSENT_WHO:
        r = agent.who_does(v, o)
        ans[("absent_who", v, o)] = r
        false_accept += int(r is not None)
    return ans, false_accept


def _build(seed, grounded):
    concepts = {w: None for w in VOCAB}
    return BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=grounded, composer_kind="rf")


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_ppmi_codeswap_answers_identical_and_moat_holds(seed):
    """The production agent on LEARNED codes answers == on CURATED codes (who/what/yes-no), and the moat holds (0
    false-accepts) on BOTH. This is the GREEN gate the #12 codes-swap needs: same answers, moat never weakened."""
    try:
        agent_cur = _build(seed, grounded=None)              # curated (composer-self-generated) codes
        agent_learned = _build(seed, grounded=_learned_codes(seed))   # the swap: learned codes
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")

    ans_cur, fa_cur = _drive(agent_cur)
    ans_learned, fa_learned = _drive(agent_learned)

    # == curated who/what: every answer identical between curated and learned codes
    mismatches = {k: (ans_cur[k], ans_learned.get(k)) for k in ans_cur if ans_cur[k] != ans_learned.get(k)}
    assert not mismatches, f"the codes-swap changed answers (seed {seed}): {mismatches}"

    # moat 0-FA (HARD) on BOTH paths -- the learned codes must still abstain on every unstored cue
    assert fa_cur == 0, f"curated path leaked the moat (seed {seed}): {fa_cur} false-accepts"
    assert fa_learned == 0, f"MOAT BREACH on the learned codes (seed {seed}): {fa_learned} false-accepts"

    # sanity: the stored facts actually recall (the matrix is non-trivial), and yes/no is right
    assert ans_learned[("what", "dog", "eat")] == "apple"
    assert ans_learned[("who", "eat", "apple")] == "dog"
    assert ans_learned[("yn", *FACTS[0])] == "yes"
    assert ans_learned[("yn", *NEG_FACT)] == "no"
    assert ans_learned[("yn", "cat", "go", "south")] != "yes"


def test_curated_codes_escape_still_available():
    """The escape: with no grounded_codes the composer self-generates its own (curated) codes -- the test-oracle /
    numpy-CPU default path stays intact (the swap is a default that can be opted out of)."""
    try:
        agent = _build(42, grounded=None)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    ans, fa = _drive(agent)
    assert fa == 0, "the curated escape path must hold the moat"
    assert ans[("what", "dog", "eat")] == "apple", "the curated escape path must still recall"
