"""bar-sanity spec (amendment A6, the BEFORE read): the same brains and decision as bar_sanity_spec_s2_c2_infer_A6_fail.py,
scored at the PRE-A6 level ("A5", the runner's default, byte-identical to the scorer the 10-26 gate used). Its record is
cited as `bar-sanity-before:` only: it shows the FAIL claim the pre-A6 scorer makes of a capable brain whose permuted hop
teach was refused (expected TOOTHLESS)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_s2_common as C  # noqa: E402
import bar_sanity_spec_s2_c2_infer_A6_fail as A6  # noqa: E402

C.LEVEL = "A5"
NAME = "stage2-C2 infer FAIL-claim at the pre-A6 scorer (A5)"
DESCRIPTION = A6.DESCRIPTION + " Scored at level A5 (the pre-A6 scorer)."
SCORER_FILES = A6.SCORER_FILES + ["docs/drafts/stage2-exit/bar_sanity_spec_s2_c2_infer_A6_fail.py"]
decide = A6.decide
for _k in [k for k in dir(A6) if k == "null" or k == "perfect" or k.startswith(("null_", "perfect_"))]:
    globals()[_k] = getattr(A6, _k)
