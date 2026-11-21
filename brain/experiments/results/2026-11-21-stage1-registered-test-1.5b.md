# Stage-1 registered multi-week life at 1.5B (TEST, six seeds): three capabilities are measured negatives, savings is undefined, and readers prefer the untrained cortex

Registered run, design `brain/research/2026-11-19-multiweek-life-design.md` §5 bars as amended in §13 (written
11-20/11-21, before any TEST life was read). Seeds 42/43/44/100/101/102, `--split test`, defaults (B), the §13 command
with every flag (`--weekend --full-probe-nights 1,5,10,15 --frozen-probe --span --transfer --pruned-a --cue-ladder
--interview --lang-guard`), 15 nights plus the 3 L-prunedA branch nights. Code 6d5726850 per the batch launch (no commit
is recorded inside the life JSONs; §13 names the Stage-1 pieces at b83a6c036, which 6d5726850 contains). The §13 hashes
for cues (2055dc92), interview spec (4b70347f) and held-out set (adc7d902) are recorded in every result; the TEST bank
is hash-guarded at run time but its hash is not recorded per life. Compute: AWS g5.xlarge (A10G), batches 20261121-0059
(s42, s43, host ip-172-31-34-100), -0124 (s44, s100, ip-172-31-9-210), -0149 (s101, s102, ip-172-31-86-241); every life
rc 0 (batch `run.log`s), 4.1-4.2 h of recorded compute each (model load excluded). `tools/where_ran.py` does not parse
the JSON `host` field and prints "unknown"; the hosts above are each result's `host`.
Raw lives: `_scratch/stage1/reg/life_s*.json` (not committed, 1.2 MB each). Read-outs, committed in
`2026-11-21-stage1-registered-test-1.5b/`:
- `bars.md/json`: `brain/eval/life_bars.py` at 6e1f4932e. It was built and adversarially reviewed on DEV s316-318 only
  (the builder and the reviewer did not open the TEST results), and committed at 10:44:14, 9 s before its TEST run but
  about 14 min after the TEST life table, reader units and interview prompts had been generated on TEST (10:30-10:44);
- `life_table.md` (`life_table.py`), `leak.txt` (`life_leak.py`, one block per seed in order 42, 43, 44, 100, 101, 102);
- `interview_read.md/json` (blind sonnet claims extractor over 1098 units, then `life_interview_read.py`);
- `reader_score.md/json`: the §7 reader (spec 6cf3690a, validated on DEV s316-318 at 211a88835). TEST read: 2
  independent sonnet readers, 47 units, 15 pairs in both A/B orders, 300 answer prompts.
Skeptics: 5 fresh-context refuters (number fidelity, wrong negatives, wrong positives and scope, protocol integrity,
mechanism), merged by opus. Every registered verdict label survived; three causal claims of the first draft were
refuted or narrowed and are corrected below.

## Verdicts (the registered bars, read literally)

| Capability | Bar | Result | Verdict |
|---|---|---|---|
| P0 forgetting precondition | LOW recall at age >= 10 nights 0.15 below age 1, log-odds lower, and 0.10 above the never-told false rate, on 5/6 | 0/6 seeds. The never-told false rate ("the reply names any told key", 3 fixed never-told questions per probe, 45 rows per seed) is 0.40-0.76; LOW recall is 0.29 at age 1 (pooled) and 0.0 at age >= 10 on s44/s100/s101/s102, so the floor clause fails even against a zero false rate there; the age >= 10 bins hold 2-4 items per seed | **NOT VALID** (thinly decided) |
| Savings (L-prunedA) | G_null_sub > 0 on 5/6 + pooled p < 0.05; dR_null_sub >= +1.5 of 8 | G_null_sub +0.67 pooled, 4/6 seeds positive, p 0.053; dR_null_sub +0.17 (2/6 >= 1.5). Branch night 2 (not primary): +1.58, p 0.004 | **UNDEFINED** (P0 not valid); it would fail the threshold anyway |
| Cued recall | R_cue > 0 at +1 (intact) and +3 (hippocampus-off), pooled p < 0.05; direct beats uncued by >= 0.4 at age <= 3; H 0 wrong on every probe on 6/6; HIGH >= LOW on 5/6 | R_cue +0.046 (p 0.057) and +0.031 (p 0.114), 3/6 seeds each; C0 - C4 = 0.233; H not-right on every seed (311 intact not-right H answers); HIGH >= LOW 6/6, but only through the brain's own facts (dan/nova, answered from permanent self-episodes): without them 3/6, and both-off passes trivially (H = LOW = 0) | **measured negative** (no clause passes on a discriminating read) |
| Transfer (sysadmin) | T > 0 at k = 4 on 5/6 + p < 0.05, increasing with k, both-off and L-nosleep about 0, consolidated-off reduces T | T(k=4) +0.24, 4/6 seeds, p 0.044; pooled T 0.2423 / 0.6258 / 0.2419 at k = 1 / 2 / 4 (non-monotone, k4 ties k1); consolidated-off reduces T on 6/6 at night 15 only (mean 0.59; s100 +0.35 and s102 +0.16 lie inside the k = 0 noise spread -0.37 to +0.13); both-off T equals T0 = +/-1.046 exactly on every seed, so that clause does not discriminate; L-nosleep not run (UNDEFINED) | **measured negative** (expected to fail, §5) |
| Honest self-report | AUROC >= 0.65; invented memory below both-off; over-abstention < 0.3; lesion sensitivity on 4/6; 0 asserted experience claims on 6/6 | AUROC 0.53 (null 0.51, p 0.49; 16 positives of 156, None on s101); invented memory 0.85 intact vs 0.86 both-off (literally below, p 0.50: no claim); over-abstention 0.00 (does not discriminate: intact answers "Of course, you're Dan" on 127 of 156 KNOW prompts and matches the claim-all decoy); lesion sensitivity UNDEFINED (defined on s43/s100/s102 only, 0/3; 4/6 unreachable); asserted experience claims on every seed by the extractor (3-16 per seed), but on s42/s43/s101 they are verbatim echoes of user lines ("I felt rather restless"); self-generated claims on s44 (1), s100 (2), s102 (3); both-off also asserts claims on 5/6 seeds | **measured negative** |
| Language guard | night-15 KL <= 2x night 1; collapse on <= 3/6 seeds per arm | KL night 15 is 12.8-17.2x night 1 on 6/6 seeds (0.14-0.17 -> 2.10-2.41); held-out ordinary-text NLL with adapters on, read through the speaking path with no hippocampus, rises from 1.99-2.02 to 3.81-4.25 (pretrained 1.931 every night); collapse on exactly 3/6 intact seeds (s43 night 3, s44 night 2, s101 night 9: at the limit, passes); s44 meets the §10 per-seed kill at night 15 (the last night: no life was cut) | **FAIL** (KL) |
| Whole-transcript read (§7) | intact preferred to both-off on week 3 on 4/6 beyond duplicate noise; week-3 defects not above week 1 on 4/6 | Instrument valid on TEST (canaries 6/6 and 5/6 strict, 6/6 loose; all 3 decoys flagged and rated at or below both-off; kappa any-defect 1.0, pairwise 0.93). Both-off preferred on 6/6 seeds (reader 1, every pair consistent in both orders; reader 2 5/6 plus 1 tie). Intact week 1 preferred to week 3 on 6/6 (both readers). Week-3 defects not above week 1: 3/6. The intact vs both-off pair differs in battery answers only (lived turns identical by construction), and replies are length-confounded (31-40 vs 83-183 characters) with no long-but-wrong decoy | **FAIL** |

**Milestone (2026-12-14) as §5 defines MET.** A six-seed multi-week life ran with nightly sleep. P0 was decided by rule
(thinly: 2-4 items per old bin, 3 repeated never-told questions). Three capabilities have measured negative verdicts and
savings is UNDEFINED with its reason. Decoy validity is shown for the §7 reader and the interview AUROC; the
strict-scorer bars have no decoy or label-shuffle check. So the measurement milestone is met on the §5 letter, with
those caveats. The capability milestone is not: no capability passes.

**§10 programme kill (ruling, 11-21; the owner may override).** §10 says: if at least 3 of the 6 seeds "collapse
before night 10", the milestone is a measured partial at 1.5B and further method work stops. Intact reaches a
single-night collapse share >= 0.4 before night 10 on exactly 3 seeds (s43 night 3, s44 night 2, s101 night 9), each
recovering the next probe night. The programme kill sits in the kill-rules section after the per-seed kill, so I read
"collapse" there as the per-seed kill event ("ends that seed's life"), which §13 defines as collapse on 2 consecutive
probe nights. Only s44 meets it, at nights 14-15. So the programme kill is NOT met and Stage-2 method work continues.
The literal alternative (any single-night collapse counts) would meet it; it is recorded here and on the board.

## What the slept brain is doing (the biggest conversational defects)

- **Sleep wears the cortex's ordinary language down.** Held-out ordinary-text NLL roughly doubles over 15 nights
  (1.93 pretrained, about 2.0 after night 1, 3.8-4.25 after night 15). KL against pretrained rises near-linearly with
  the adapter norm (correlation 0.98-0.996), with night-to-night dips on 6/6 seeds (s42 2.39 -> 2.10 on nights 14-15;
  s101 2.29 -> 1.85 on nights 11-12) and no plateau through night 15. The adapter norm itself is monotonic, about +42
  per night, 9.4-11.9x night 1 by night 15. The sleep keep term holds its own 3 ORDINARY texts at a logged KL of
  0.012-0.067 every night while held-out KL reaches 2.1-2.4: the 3-text keep set does not protect language in general.
- **Readers prefer the untrained cortex.** On week 3 the intact brain's battery answers carry 106 reader-1 defects
  against 45 for both-off (hippocampus-off also 106; the gap is already 96 vs 56 in week 1). Across all units intact
  has more off-topic (78 vs 58) and experience-claim (36 vs 24) rows, both-off more invented (50 vs 36); 6 units per
  arm-week. Coherence falls in both arms (intact 2.0 -> 1.17, both-off 2.0 -> 1.33, reader 1).
- **First-person echo.** The brain restates the user's line in the user's own voice ("My cat is called Pixel"). 150 of
  the 311 not-right intact H answers contain the correct key this way and are rejected by the strict scorer.
  Sensitivity only (no verdict change on TEST, §9): with a key-hit scorer CUE-D is 0.45 and R_cue is 0.138 (p 0.0009,
  6/6) at +1 and 0.094 (p 0.0024, 4/6) at +3; CUE-H still fails on every seed. Whether an echo is a "correct slot"
  (§5 R_cue wording) is a ruling for a fresh split, not this read.
- **The stock self-reply leaks.** Name substrings ("you're Dan", "my name is Nova", "I'm Nova") appear in 24.4% (night
  10) and 24.6% (night 15) of probe and interview replies to non-identity questions, pooled over seeds and both arms
  (9.8-35.4% per seed per night; interviews 33-47%, battery probes 1-14%; DEV s318 B 0.26). It is NOT what fails P0:
  only 30 of the 158 hippocampus-off never-told false rows contain the names; the rest quote other told keys
  ("administrator", "bandstand"), and P0 fails on every seed without the name rows.
- **Recall decays and wrong fetches stay high.** H with the hippocampus off falls from 28/36 on night 1 to 10-13/36 from
  night 10. Intact-arm wrong fetches (hippocampus-off logs none) are 23/36 on night 1, 42/49 on night 10 and 40/50 on
  night 15.

## Caveats (instrument; none changes a verdict)

- §5 asks every bar also be computed on label-shuffled DEV data for its false-positive rate. That run is not built.
  Without it the clauses marked as passing (HIGH >= LOW, consolidated-off reduces T, over-abstention, collapse, both-off
  T about 0) are not shown to discriminate; HIGH >= LOW, both-off T and collapse pass under the both-off null.
- Raw T carries a seed-parity offset (T0 = +/-1.046 by seed parity; 4 even and 2 odd registered seeds). Intact dT at
  k = 4 is about +0.99 on the even seeds and about -0.2 on the odd seeds (mean +0.59, 5/6 positive). The registered
  quantity is raw T (a HANDOFF note of 11-20, dab53c019, said the both-off bar "must read dT", but it never reached
  §13). Reading dT does not change the verdict: mean dT is 0.591 at k = 1 and 0.590 at k = 4, so "increasing with k"
  fails either way.
- The bars use the strict scorer. The §7 answer prompts give the reader-verdict RIGHT beside it (intact 32/98 RIGHT
  vs 33 strict ok, 1 veto), so no recall verdict moves.
- Builder rules, not registered: the both-off "about 0" rule (|mean| <= 2 SE), and the reader's "scaled to 4/6" and
  "within MDE" counting.
- P0: four of the six seeds (42, 43, 101, 102; 43 and 101 on both) also fail the drop or log-odds clause; 44 and 100
  fail only the floor. The TEST verdict stands as registered (no change after a TEST read, §9).

## Next (method work by the 12-08 Stage-2 deadline; any re-test on a FRESH TEST split, §9)

1. **Language guard first: the sleep write's keep set.** The keep term holds its 3 ORDINARY texts while held-out
   language drifts, and the norm grows about 42 per night from a fresh Adam at lr 5e-4. The leading method is a
   broader ordinary-text keep set (interleaving general text with episode replay, the complementary-learning-systems
   prescription). Separating reads first: `life_langguard.measure()` on ORDINARY vs HELDOUT on saved homes, and a
   keep-only control night. Scale is not ruled out (only 1.5B ran).
2. **D2-PR (loss-prioritised replay; the DEV 319-321 screen is running) stays the stock-reply lever.** Its prereg
   guards the language-guard KL; the read also reports whether PR slows the NLL rise.
3. **First-person echo** is the largest single cause of the strict recall negatives and a reader-visible defect
   (perspective rows). It needs its own record check before a method.
4. **P0 needs never-told questions not to surface told keys, and real old-age bins.** The false rate comes mainly
   from other told keys, and age >= 10 recall is 0.0 on four seeds with 2-4 items per bin. Fixing the leak alone will
   not make P0 valid; a fresh split needs more old-age items per seed.
