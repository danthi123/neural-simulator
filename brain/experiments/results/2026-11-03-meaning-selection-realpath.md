# 2026-11-03 Meaning-based selection through the real hippocampus path (S_all vs S0), refit bar, no duplicate memory

Follow-up to `2026-11-03-meaning-selection.md` (skeptic: not fit to merge). That probe recomputed similarities from
Gram matrices and never called `KeyedHippocampus.recall` (no strength weighting, no bar, no margin, no stored fp16
delta); its bar/margin were S0's; `S_all` stored a second fp32 copy of the whole memory per episode. Capability:
memory selection (`brain/PREREQUISITES.md`); record: `brain/research/2026-11-02-record-brief-consolidation-and-selection.md`
part B; `tools/before_you_build.sh` ran (11-03) and surfaced only the 11-03 writeup itself.

## Changes

- `Hippocampus.recall` split into `similarities()` -> `scores()` (strength-weighted) -> `_compete()` (bar, margin, use
  boost). Pure refactor: class constants (FAMILIARITY 0.12, MARGIN 0.02, GAIN 4) unchanged, S0 test still identical.
- `KeyedHippocampus`: competition is the parent's `_compete`. `S_all` (and `S_mid`, `S<l>`) no longer stores a key:
  the similarity is read from the episode's stored fp16 delta at recall, one layer at a time (exactly the centred
  cosine of the concatenated per-layer-normalised key; test). `familiarity=` / `margin=` set a per-instance bar;
  the class defaults are untouched. Pooled activity is kept only for the learned key.
- `DEV_FACTS36` + `DEV_NEVER_TOLD` (`facts_bank.py`): 36 new teacher-written facts (same format, own confusables,
  12 never-told questions), no shared answer word or cue text with the 48-fact test bank (asserted). Used only to fit.
- `brain/experiments/selection_realpath.py`: everything through `KeyedHippocampus.scores/recall/reinstate`.

## Pre-registration (written before the run; instrument fixed)

Setup. Models 0.1B and 0.4B (`/home/dant123/models/rwkv7/`), CPU. Seeds 42/43/44/100/101/102. TEST: the 48-fact bank,
split per seed as the 11-03 probe (`Random(seed)` shuffle; 24 stored, 12 untold); cues: same-words (orig), light
(para_a), heavy (para_b) rewording of each stored fact (72 told cues); untold cues = the 12 untold facts' questions +
the 12 NEVER_TOLD questions. DEV: the same on DEV_FACTS36 (24 stored, 12 untold) + DEV_NEVER_TOLD. Every episode is
fresh at each cue (age 0, strength 1, use boost undone after each cue).

Fit (DEV only, pooled over the six seeds, per key and model): grid FAMILIARITY -0.10..0.80 step 0.005 x MARGIN
{0, .005, .01, .02, .03, .05, .08, .12}; maximise J = P(fire and the right episode | told cue, all three cue types) -
P(fire | untold cue); ties to the larger bar, then larger margin. The fitted values are then frozen for TEST.

Arms on TEST, each through `recall()`: `forced` (bar and margin off: top-1), `default` (0.12 / 0.02), `refit`
(dev-fit), for S0 and S_all; plus `oracle` (right episode reinstated) and `none` (no memory). Recall = reinstated
episode (or none if recall did not fire), greedy 16-token answer, hit if the answer word is in it.

Criteria (each must hold at BOTH sizes):
- P1 selection: forced S_all > S0 on selection for para_a and for para_b, each on >= 5/6 seeds.
- P2 recall: refit S_all > refit S0 on para_b recall on >= 5/6 seeds, and the mean gap >= 0.10.
- P3 told-vs-untold under refit: J_test(S_all refit) > J_test(S0 refit) on >= 5/6 seeds. Reported, not a criterion:
  untold fire rate under refit; above 0.25 means the familiarity bar does not control false recall for that key.
- P4 replication: forced S_all para_b selection mean within 0.10 of the Gram probe (0.69 at 0.1B, 0.76 at 0.4B);
  otherwise the 11-03 numbers do not stand and this writeup replaces them.
- P5 memory: S_all bytes per episode == its fp16 delta bytes (before: measured from the 11-03 code).
- P6 consistency: every recall() decision equals the bar/margin rule applied to scores() (asserted in-run; a failure
  aborts the run).
Default recommendation rule: S_all is recommended as the Hippocampus default only if P1-P3 hold at both sizes, and then
only after a live-brain test (below); a lead is not a GO.

## Results

(to be filled from the result JSONs)
