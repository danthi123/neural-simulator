---
type: finding
status: live
date: 2026-10-04
lane: integration-first (stage exits on the calendar)
mechanism: Stage-1 (INFANT) exit gate on the integrated production brain, six seeds, scored by the registered scorer
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/stage1_exit_gate.py
artifacts:
  - research/findings/raw/_stage1_exit/gate/aggregate.json
---

# Stage-1 (INFANT) exit gate: result (2026-10-04)

Prereg: research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md (gate revision c238ff325a38e40a68b1f9751822630ff729377b). Scored only by `python -m research.runners.stage1_exit_gate --aggregate ... --seeds 42,43,44,100,101,102`. Every number below is read from research/findings/raw/_stage1_exit/gate/aggregate.json.

**Stage-1 exit verdict: FAIL** (primary criteria: C1 FAIL, C2 PASS, C3 FAIL, C4 FAIL)

| group | criterion | role | verdict | correct | lesion effect | reproduced | permuted tracks | why |
|---|---|---|---|---|---|---|---|---|
| cat_disc | C1 | primary | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| cat_told | C1 | diagnostic | FAIL | 0/6 | lesion_draw 0/6 | lesion_draw 6/6 | 0/6 | capability absent on 6 of 6 seeds |
| word | C2 | primary | PASS | 6/6 | lesion_recruit 6/6 | lesion_recruit 6/6 | 6/6 | - |
| lived | C3 | primary | FAIL | 6/6 | lesion_recall 0/6, lesion_episodic 6/6 | lesion_recall 6/6, lesion_episodic 6/6 | 6/6 | lesion_recall: reply-level effect on 0 of 6 seeds (not load-bearing) |
| val_heldout | C4 | primary | FAIL | 6/6 | lesion_affect 6/6 | lesion_affect 6/6 | 0/6 | permuted exposure tracked on 0 of 6 seeds (not exposure-driven) |
| val_novel | C4 | diagnostic | FAIL | 0/6 | lesion_affect 0/6 | lesion_affect 6/6 | 0/6 | capability absent on 6 of 6 seeds |

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = 0.938 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.938; defined
- seed 43: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; defined
- seed 44: r = 0.951 (n = 24, sign agreement 0.88); lesion r = None; permuted-map r = 0.951; defined
- seed 100: r = 0.947 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.947; defined
- seed 101: r = 0.946 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.946; defined
- seed 102: r = 0.944 (n = 24, sign agreement 0.75); lesion r = None; permuted-map r = 0.944; defined


## Per-criterion reading (fixed sentences from HANDOFF section 8)

C1 (`cat_disc`): FAIL, capability absent on 6 of 6 seeds. The integrated brain does not show this Stage-1 ability at production defaults; the gap named in prereg section 8 applies.

C2 (`word`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced; the permuted exposure was tracked on 6 of 6.

C3 (`lived`): FAIL, not load-bearing: the reply was correct but the lesion_recall arm changed it on only 0 of 6 seeds.

C4 (`val_heldout`): FAIL, not exposure-driven: the permuted exposure was tracked on only 0 of 6 seeds.

`cat_told` (diagnostic, not counted): FAIL, correct 0/6.

`val_novel` (diagnostic, not counted): FAIL, correct 0/6.

## Claude review (2026-10-04, verify-go): the C2 PASS stands, narrowed

Two independent skeptics were told to refute the C2 PASS: one on gate-cheat and instrument trust, one on control
integrity and reproducibility. Neither refuted it. Both found caveats, which are recorded here and travel with every
board or ledger use of the word PASS.

- **Re-score.** The registered scorer, re-run from the committed raw shards, gives a `groups.word` block equal to
  `aggregate.json`.
- **Instrument.** "Correct" is read from the structured `recalled_svo` / `abstained` fields, not by substring
  matching on reply text. The counted replies are genuine: `p_bear_chase` "... the bear chases the otter",
  `p_otter_eat` "... the otter eats the clam". Both moat probes and the pre-exposure probe say "I don't know about
  that", and dog->cat survives.
- **Determinism and null.** `intact_a` and `intact_b` are whole-JSON identical on every turn of every seed, and
  `lesion_recruit` equals its `.rep0`. The lesion effect is therefore not run-to-run noise.
- **Scorer weaker than the prereg prose (caveat 1).** The aggregate's lesion `effect` is `any(differs)` over the two
  counted probes, and it never scores dog->cat survival in the lesion arm. The prereg asks that BOTH answers are
  removed and the built-in fact survives. Checked by hand on all six seeds: the only labels that differ from intact
  are `p_bear_chase` and `p_otter_eat`, both become abstains, and `c_dog_chase` stays `[dog, chase, cat]`. The claim
  holds in the data, but the registered scorer would not have caught a one-answer or a dog-killing lesion. Tighten
  it before the next stage-exit attempt; this result stands as scored.
- **Lesion scope (caveat 2).** `lesion_recruit` replaces `OneBrainComposer._recruit_word` with `return False`
  (21 calls on every seed). The host `kb` still holds the taught fact and the reply still abstains, so the answer
  depends on the cleanup decode and not on the host table. But the lesion removes all three unseen words at once
  through the shared codebook. "Necessary" names the cleanup-slot pool, not one assembly.
- **Seeds (caveat 3).** The six seeds are distinct substrates: the otter-agent margins are 0.643 / 0.578 / 0.716 /
  0.678 / 0.621 / 0.676. The reply-level outcome is the same on every seed, so "6/6" is one deterministic outcome
  replicated six times, not six independent draws.
- **Ownership (caveat 4, from research/findings/2026-10-04-stage1-host-share-RESULT.md).** At the same C2 turns,
  every decision seam is HOST-OWNED: k-select, bg-select, organ-c, metacog and ignition. The panel host decision
  share is 1.000 and the neural fraction 0.000 (n 48). The content itself is learned (learned_content_fraction
  1.000). C3's seams read host 0.967 / neural 0.033 (n 30).

**What C2 PASS shows:** in the integrated production handler, a word heard for the first time this conversation is
recalled in both roles, with the moat intact, and the recruit cleanup-slot pool is necessary for that reply. **What
it does not show:** that neurons made the decision. The SVO acquisition, the first-match selection and the
commitment are host scaffolds, as prereg section 5 predicted. In-session words also never become discourse referents
(anaphora 0/6, reported, not counted).

Ledger wording: `Stage 1 exit attempt 1 (2026-10-04): FAIL (C1, C3, C4); C2 word PASS 6/6, reply-level, recruit pool
load-bearing, decision host-owned (host share 1.000)`.
