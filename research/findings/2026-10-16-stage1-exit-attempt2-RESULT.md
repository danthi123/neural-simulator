---
type: finding
status: live
lane: integration
date: 2026-10-16
artifacts:
  - research/findings/raw/_stage1_exit/gate_a2/aggregate.json
  - research/findings/raw/_stage1_exit/gate_a2/aggregate.json.prov.json
  - research/queue/s1_attempt2_c02180f2c.tsv
verdict: PASS (C1-C4 primary each 6/6; diagnostics cat_told + val_novel FAIL 0/6, not counted)
---

# Stage-1 exit gate — attempt 2 RESULT

Filed by Hermes per RULING `s1-score-now-1016` / `s1-attempt2-score`; **verified by Claude 2026-10-16**
(verify-go, section "Claude verification" below): the PASS stands as registered, with four caveats. Bars untouched.

<!--derived--> **Stage-1 exit verdict: PASS** (primary criteria: C1 PASS, C2 PASS, C3 PASS, C4 PASS)

Source of truth: `research/findings/raw/_stage1_exit/gate_a2/aggregate.json` (verdict `PASS`, GO `true`,
missing_shards `[]`, undefined_reasons `[]`), reproduced with
`.venv/bin/python -m research.runners.stage1_exit_gate --report research/findings/raw/_stage1_exit/gate_a2/aggregate.json`.

## Per-criterion verdicts (HANDOFF §8 sentences, numbers from the aggregate table)

- C1 (`cat_disc`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.
- C2 (`word`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.
- C3 (`lived`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.
- C4 (`val_heldout`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = 0.817 (n = 24, sign agreement 0.58); lesion r = None; permuted-map r = -0.144; defined
- seed 43: r = 0.805 (n = 24, sign agreement 0.46); lesion r = None; permuted-map r = 0.046; defined
- seed 44: r = 0.824 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.081; defined
- seed 100: r = 0.809 (n = 24, sign agreement 0.62); lesion r = None; permuted-map r = -0.073; defined
- seed 101: r = 0.836 (n = 24, sign agreement 0.54); lesion r = None; permuted-map r = -0.001; defined
- seed 102: r = 0.820 (n = 24, sign agreement 0.54); lesion r = None; permuted-map r = -0.104; defined

`cat_told` (diagnostic, not counted): FAIL, correct 0/6.
`val_novel` (diagnostic, not counted): FAIL, correct 0/6.

Both diagnostics read `capability absent on 6 of 6 seeds`; the prereg reports them beside C1/C4 and freezes them
as "not a pass condition" (HANDOFF §8: diagnostic, not counted).

## preconditions

All `require` entries in `aggregate.json.preconditions` are `ok: true`:
no primary shard missing; null clean (intact_a == intact_b) on every primary (group, seed); every response from
the expected brain (`tiny-demo +LTM`); `PYTHONHASHSEED=0` in every primary arm; corpus
(`data/corpus/tinystories.txt`) present in every primary arm; every in-process lesion applied (patch called at
least once); and the remaining require-list carried by the runner (reasons verbatim in the artifact).

## Reproducibility / where the numbers came from

216/216 A2 shards at SHA2 `c02180f2c7e1464cf0f2192a46cf2f399bd1c215`, seeds 42,43,44,100,101,102 (36 arms each),
committed with prov sidecars as `1d462264b`. Host summary (`tools/where_ran.py`): aws `ip-172-31-34-243` 90 +
`ip-172-31-47-61` 90 (the two r7i.4xlarge of RULING `s1-awsrun-answer-1016`, spend ~$9.26), pool nodes 19
(seed-42 lane), workstation 17 (seed-42 backfill), server 1 = the local seconds-long aggregate itself. The
corpus and brain inputs the provisioner ships are asserted per-arm by the preconditions above, so validity does
not depend on host class here. Delegate verdict (MiMo Pro, 2/2 budget): **PASS-FILE**.

## artifacts

- `research/findings/raw/_stage1_exit/gate_a2/aggregate.json` (+ `.prov.json`)
- `research/findings/raw/_stage1_exit/gate_a2/s{42,43,44,100,101,102}/*.json` (+ sidecars) — 216 shards
- emit TSV: `research/queue/s1_attempt2_c02180f2c.tsv`

## Claude verification (verify-go, 2026-10-16)

**Verdict: PASS confirmed as registered (§7), with caveats.** Lenses run: an independent re-score, two adversarial
skeptics (gate-cheat + prereg compliance + control integrity; data integrity + seeding + instrument trust), and the
three required lenses (engagement, answer-rate, where-ran).

What survived:
- **Re-score.** `stage1_exit_gate --aggregate` re-run on W over the committed shards reproduces `aggregate.json`
  exactly (verdict, criteria, every group block).
- **Data integrity.** All 216 sidecar argvs match exactly one line of `research/queue/s1_attempt2_c02180f2c.tsv`,
  with no duplicates. Every shard has git_sha c02180f2c, git_dirty false and one verified source manifest. Host-independent
  fields (sources, backend, pythonhashseed_ok, corpus_present) are identical across the AWS, pool and workstation
  hosts.
- **Prereg compliance.** The runner and the prereg are unchanged between the A2 registration (21e5d5c67, 10-13) and
  c02180f2c. `aggregate.bars` equals the prereg's bars. The prereg itself excludes `cat_told` and `val_novel`
  ("not a pass condition", §1; "reported and never counted", §7).
- **Engagement.** Every counted in-process lesion recorded calls: inherit_pathway 2, recruit 21, recall 4. The
  episodic and affect lesions are traced as lesioned. The C4 permuted arm has n_words 24 and a scrambled |r| of at
  most 0.035. C1's permuted arm flips the answer to `bug` on 6/6, which rules out a frequency shortcut.
- **Margins** (derived by the data skeptic from the per-seed shards `research/findings/raw/_stage1_exit/gate_a2/s*/*.json`):
  <!--derived--> C4 r is 0.805-0.836, at least 0.255 above the 0.55 bar. C1's worm rate is 0.047-0.088 against a 0.02
  <!--derived--> floor, with the competitor at 0. C3's apical cue is at least 0.556 against 0.20. No single seed carries a
  criterion.
- **Where-ran.** aws 180 / pool 19 / workstation 17. The learned valence map is git-tracked and loaded on every
  host, and the cross-host nulls match, so validity does not depend on host. Seed 42 is the only non-AWS seed, so
  seed and host class are confounded, but C1 and C4 values reproduce the earlier pool runs exactly.

Caveats (what the PASS does and does not show):
1. **C2 and C3: the recall lesions change how the answer is framed, not what it is.** `lesion_recruit` (C2) and
   `lesion_recall` (C3) turn a recalled answer into a guess with the SAME content ("perhaps the bear chases the
   otter [a guess]"). The default-on category organ (C1) still supplies the answer. §7 is met, since a primary
   reply field differs, but the §1/§6 wording that the lesion "removes" the answer is not. The C3
   `lesion_episodic` arm does remove it (disclosure becomes "I don't recall", 6/6). Load-bearing claims for the
   recruit and recall pathways must cite this.
2. **C1-C3 replies are identical across the six seeds.** Seeds change internal traces and C4, but not these
   discrete answers. For C2, seeds 43-102 give byte-identical text. Six seeds here show robustness, not six
   independent varied samples.
3. **C4 measures how faithfully the brain follows a precomputed valence map.** The map is an offline numpy PPMI +
   label-propagation table seeded from Warriner norms. Intact r (about 0.82) sits below the raw-norm reference
   (0.94). 8-13 of the 24 reads per seed are neutral, which is why sign agreement is only 0.46-0.67, though no
   non-neutral read has the wrong sign. Valence inheritance during a session (`val_novel`) is absent, 0/6.
4. **Minor.** The generative probe ("what might...") abstains on 6/6 because the host topic extractor reads
   "might"; it is uncounted. Prereg §4.1 calls the worm/bug exposure equal, but the teach turns hold worm x3 and
   bug x2; the permuted flip shows no frequency shortcut. A2.2's constrained scramble draw makes the C4 control
   pass for a perfect follower about 99% of the time, so that clause is near-determined by construction.

Decisions-owned note: this stage bar does not measure host-decision share. The k-select step is still host code,
so this PASS is not evidence that the brain owns these decisions (phase plan
`docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md`).

