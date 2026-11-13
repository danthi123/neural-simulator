# Checkpoint: tautology-1113

kind: merge_review
branch: research/pi-lane-1105
sha: 5a2b1d1bd

## Question

Merge verdict on research/pi-lane-1105 head 5a2b1d1bd for the self-predicated-tautology lane item (work commits 018e8d2c7 code+tests+rerun artifacts, 62d83b752 MiMo-review hardening). Built exactly to spec: is_tautology keeps its two rules and adds the clause rule (clauses split exactly as is_echo splits, now the shared _clauses helper with the identical regex): a clause holding the key at least twice with a linking verb (is, are, was, 's) standing between the first and a later occurrence. Tests in brain/tests/test_strict_scorer.py: the three named strings are tautologies (and correct_answer False), the four named are not, and correct_answer("Marguerite! Your sister is Marguerite.", ...) is True; every pre-existing test in test_strict_scorer.py, test_weekly_scorer.py, test_seeded_keys.py passes unchanged (72 brain/tests). live_brain_rescore.py rerun on results/2026-11-03-live-brain-12facts/ (tool-regenerated rescored.json + table_correct.txt, nothing hand-edited): 8 rows changed, all True->False, all self-predicated ("Biscuit is a dog breed named after the English name Biscuit..." s100 b rows 4+16; "Pixel is a cat called Pixel." s101 b row0; "Biscuit is a nickname for Biscuit." s101/s102/s43 b row4; "Lyon is called Parc Lyon." s44 a row2; "The cello is a very good instrument for the cello." s44 b row20). Aggregates: a recall 1.83->1.67, b recall 5.00->4.17, b paraphrase 3.33->3.00; sleep adds (a-b) recall -3.17->-2.5, paraphrase -1.17->-0.83, false 1 unchanged; P-checks unchanged. Questions: (1) accept and close the item? (2) please RULE on one spec ambiguity I did not decide unilaterally: "holds the key twice" is implemented with the module's documented key-as-word-start semantics (has_key docstring "peanut matches peanuts"; rule 1's \w*), so "Pixel is pixelated." holds the key twice and is flagged as self-predicated; whole-word-only matching would instead break plural keys ("Pixels are called Pixels"). MiMo's dissent is in research/findings/raw/_tautology_review.diff. (3) confirm the 8-row True->False list above is the changed-rows list the item's Done line requires.

## Bar

Item Done line verbatim: "Done = commit on research/pi-lane-1105 + tests passing + the changed-rows list + a merge_review checkpoint (Claude), not a MiMo self-review." Plus the item's test bar: the three named strings are tautologies; the four named strings are not ("Marguerite! Your sister is Marguerite." gives correct_answer True); every existing test in test_strict_scorer.py, test_weekly_scorer.py and test_seeded_keys.py passes unchanged.

## Evidence

- brain/experiments/strict_scorer.py
- brain/tests/test_strict_scorer.py
- brain/experiments/results/2026-11-03-live-brain-12facts/table_correct.txt
- research/coordination/PI_LANE.md
- research/findings/raw/_tautology_review.diff
