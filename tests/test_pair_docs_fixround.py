"""Regression tests for the pair-docs-final fix round (independent review, 2026-09-25).

Two defects were found by an independent review of branch research/pair-docs-final and fixed on top of it:

  HIGH   research/findings/2026-09-25-pair-production-path-seed7-dev-smoke.md miscounted the sn family's
         long-delay recall split as "8 of 10 recalled / exactly the two lesion arms abstained". Reading each
         arm's own final `*_recall` key in the raw artifacts shows a THIRD arm, `lneu_rc`, also abstained --
         the real split is 7/10 recalled, 3/10 abstained -- and `lneu_rc`'s abstain is the pre-registered
         EXPECTED outcome for the neutral condition (Amendment 7's own SN1 GO condition), categorically
         different from the two lesion arms' abstains.

  MEDIUM research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md's Amendment 7 "Predictions"
         section added three numbers (0.086, 0.103, 0.209) whose own `<!--derived-->` marker sat on a LATER
         line of the same bullet, so tools/claim_check.py's per-line marker scoping did not cover them and
         the file no longer passed claim_check clean, contrary to the commit message that landed it.

Both tests here FAIL on the pre-fix content and PASS once the fix is applied -- verified by running each test
against `git show <pre-fix-sha>:<path>` written to a temp file before the fix landed.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

_CLAIM_CHECK_PATH = os.path.join(ROOT, "tools", "claim_check.py")
_spec = importlib.util.spec_from_file_location("claim_check", _CLAIM_CHECK_PATH)
claim_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(claim_check)  # type: ignore[union-attr]

PREREG = os.path.join(ROOT, "research", "findings", "2026-09-24-sleep-replay-capture-PREREGISTRATION.md")
SMOKE = os.path.join(ROOT, "research", "findings", "2026-09-25-pair-production-path-seed7-dev-smoke.md")
SN_SEED7_DIR = os.path.join(ROOT, "research", "findings", "raw", "_pair_production_path_smoke", "sn_seed7")

# The pre-registered neutral-condition control (its abstain at long delay is EXPECTED, not a lesion effect).
NEUTRAL_ARM = "lneu_rc"
# The two arms whose own label names a lesion.
LESION_ARMS = {"lsal_rc_dalesion", "lsal_rc_wakelesion"}


def _sn_seed7_recall_split():
    """Reread every sn_seed7 arm's own final `*_recall` key, independently of any doc's prose."""
    recalled, abstained = [], []
    for path in sorted(glob.glob(os.path.join(SN_SEED7_DIR, "*.json"))):
        if path.endswith(".prov.json"):
            continue
        arm = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            data = json.load(f)
        recall_keys = [k for k in data.keys() if k.endswith("_recall")]
        assert len(recall_keys) == 1, "expected exactly one *_recall key in %s, found %r" % (path, recall_keys)
        rec = data[recall_keys[0]]
        (abstained if rec.get("abstained") else recalled).append(arm)
    return recalled, abstained


def test_sn_seed7_recall_split_is_7_of_10_not_8_of_10():
    """HIGH fix: the real split (from the artifacts themselves) is 7 recalled / 3 abstained, and the three
    abstains are the neutral control plus the two lesion arms -- NOT just the two lesion arms."""
    recalled, abstained = _sn_seed7_recall_split()
    assert len(recalled) == 7, "expected 7 arms to recall the told fact, got %d: %r" % (len(recalled), recalled)
    assert len(abstained) == 3, "expected 3 arms to abstain, got %d: %r" % (len(abstained), abstained)
    assert set(abstained) == LESION_ARMS | {NEUTRAL_ARM}, (
        "the abstaining arms must be exactly the neutral control plus the two lesion arms, got %r" % sorted(abstained)
    )


def test_seed7_dev_smoke_finding_states_the_correct_split():
    """HIGH fix, doc side: the finding's own prose must say 7/10 (and 3/10), not the old wrong 8/10, and must
    name lneu_rc's abstain as the pre-registered expected neutral outcome rather than folding it into the two
    lesion arms' abstains."""
    text = open(SMOKE).read()
    assert "8 of 10 arms" not in text, "the finding still states the refuted 8-of-10 recall count"
    assert "7 of 10 arms" in text, "the finding must state the corrected 7-of-10 recall count"
    assert "3 of 10 abstained" in text, "the finding must state the corrected 3-of-10 abstain count"
    # lneu_rc's abstain must be distinguished from the two lesion arms', not conflated with them.
    assert "pre-registered" in text and "lneu_rc" in text
    assert "SN1" in text, "the finding should tie lneu_rc's abstain to Amendment 7's own SN1 GO condition"


def test_prereg_amendment7_predictions_claim_check_clean():
    """MEDIUM fix: tools/claim_check.py must pass clean on the preregistration doc -- it did NOT on the
    branch tip after Amendment 7's own edits (3 unsupported numeric claims: 0.086, 0.103, 0.209), even
    though the landing commit's message claimed it did."""
    rc = claim_check.check(PREREG, verbose=False)
    assert rc == 0, "tools/claim_check.py must pass clean on the Amendment 7 preregistration doc"


def test_claim_check_would_have_caught_the_medium_regression():
    """Negative control: the checker must actually be able to FAIL on the exact defect that shipped, so the
    previous test is not vacuously green. Pulls the REAL pre-fix content of the preregistration doc, as it
    stood at the reviewed branch tip (b1e41656e786b74e21b839ebfb42c5288296959e, before this fix round), via
    `git show`, and confirms claim_check flags it -- proving the checker distinguishes the two states."""
    import subprocess
    import tempfile

    base_sha = "b1e41656e786b74e21b839ebfb42c5288296959e"
    rel_path = "research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md"
    pre_fix_text = subprocess.run(
        ["git", "show", "%s:%s" % (base_sha, rel_path)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    assert "0.086" in pre_fix_text and "0.103" in pre_fix_text and "0.209" in pre_fix_text
    findings_dir = os.path.join(ROOT, "research", "findings")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, dir=findings_dir,
                                      prefix="_test_prefix_claimcheck_") as f:
        f.write(pre_fix_text)
        tmp_path = f.name
    try:
        rc = claim_check.check(tmp_path, verbose=False)
        assert rc == 1, "claim_check must FAIL on the pre-fix preregistration content (b1e41656e)"
    finally:
        os.unlink(tmp_path)
