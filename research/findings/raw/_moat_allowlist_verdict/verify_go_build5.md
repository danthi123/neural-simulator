# verify-go on the build-5 GO (workflow `wf_2fc14ca3-a56`, 2026-10-21)

Four independent skeptics, one lens each, each told to refute and to default to refuted. Each prompt ended with the
verbatim refuter-lens block. Ran before the build-5 composer replays (bar 5's reply-level half) existed; those are
checked separately when they land. **None refuted the GO.** What each found:

## 1. Reproducibility and instrument (sonnet)

- The round-7 hunt re-run at the build-5 code in a fresh worktree reproduces `r7/results_build5.json` exactly: all 270
  rows, every arm, both entries, seven seeds (only argv and elapsed time differ).
- `run_hunt.py` cannot count an error as "not kept": a non-boolean record makes the case UNDEFINED and the arm not
  VALID. All 3,780 allowlist records are booleans from the claim gate.
- Recomputed from raw rows: U 0 of 191, B 0 of 25, F 30 of 54 through either entry; the bound is 3/191.
- Bar 1 from the strata files: 0 U/N and 0 B kept at all seven seeds; re-running `measure_strata.py` reproduces both
  files. Bar 4 recomputes to 31 of 38.
- `verdict_numbers.json` matches the raw files for builds 1-5.
- Caveats: the seven parser seeds are one deterministic replicate (no verdict varies by seed), so the bound is over 191
  cases, not seeds; the stub population ran at parser seed 1 only; S3 strata and the stub population were stamped
  from a dirty tree at `5ccf24761` (matcher bytes equal HEAD, S3 reproduced exactly).

## 2. Engagement, controls, answer cost (sonnet)

- The flag engaged on every case: every reject carries an `allowlist:` reason; no fallback, error or undefined record.
- The hunt could see leaks: on the same 191 U the B2h arm keeps 5 and stock keeps 5 at the composer entry; the
  allowlist rejects those for the specific build-5 reasons.
- Bar 3 has less power than its size suggests: 174 of the 191 U are rejected as "no shape" and only 17 reach the
  later checks; 21 of the 22 cases in the class build 5 was built to close were relabelled B, so the strict reading
  (0 of 216) is the figure that carries that fix.
- The 24 faithful twins rejected are all wordings the shape set does not cover ("is located in time zone", "is
  twinned with", ...), plus three "Is" names. On real held-out prose the allowlist keeps 8 of 92 faithful sentences on
  S1 and 27 of 86 on S3 (B2h 13 and 25; stock 1 and 0).
- HURTS 0 on the stub population is a preregistered classification: 5,444 one-fact facts (6.9% of the store) cannot be
  voiced under the flag in any wording. A plain-English read of 36 of them found the father, author, conflict and
  instrument sentences clearly wrong; child, parent_taxon, platform and spouse (about 465) are ambiguous, and the
  table itself calls three of them conservative; sport (4,116) is a judgement call. State it as "classified as stock
  leaks by a reviewed table", never as "all are leaks" or "no answer loss".
- Flag-off identity holds: the golden fixture has one commit, from unmodified code, and its tests pass in the junit
  record.

## 3. Pre-registration, blindness, adjudication (opus)

- Commit order is clean: last matcher change `ef51d6243`, then Amendment 5, the skeptic, the cases, the judges, the
  adjudication, and one run at a clean tree. No matcher or table change came after the cases.
- The relabel rule recomputes exactly in rounds 4-7 (0 mismatches) and was fixed before the cases existed; the judge
  labels match the judge workflow's raw output case by case; the judges saw no expect label or class.
- The skeptic's 44 tool calls stayed inside the blindness rule (spec-level knowledge only, as allowed).
- No amendment loosened bar 3. The wording "at all seven seeds through both entries" can be read as an intersection;
  the verdict uses the strict union and passes it.
- Round 7 has no case in the Amendment-5 known-B-residue classes, so the strict 0 of 216 says nothing about them.

## 4. White-box leak hunt (opus; about 55 verified attempts, full code access)

No clear U leak under the protocol's own adjudication rule, so the GO stands; the findings narrow it:
- **Definite-description names read as anaphora.** "Rick Wakeman is a member of Yes band. The band award received Grammy
  lifetime achievement award." (`the_band` is a store entity, The Band) is kept at all seven seeds through both
  entries; a plain reader takes it as Yes receiving the award. Round 6 relabelled all 57 cases of this mechanism B, so
  the protocol would likely call it B. Not adjudicated; "no plain-reader U leak" is not claimed.
- **Names that read as clauses.** The clause-word rule covers finite auxiliaries and negation only; names holding a
  lexical verb, a modal or a first person are kept ("I need a beat genre hip hop music.", "Louis the alligator composer
  I heart tuesdays."). B by round-6 precedent; the first-person ones bear on the honesty boundary.
- **The composer checks sentences one at a time.** Under the flag each fact's sentence is checked alone and the
  sentences are then joined; the checker never sees the joined reply, so the cross-sentence rule applies only inside
  one candidate. That is how the anaphora case would reach a user.
- **Paths that bypass the flag.** With the flag on, the spiking recall surface (default on), Touchpoint A,
  `ChatBrain.render` and the hypothesis path still use the stock single-fact check, and a mood marker can be appended
  after verification. Only 5 of 78,857 store facts fit the spiking frame, so the measured impact is tiny, but the
  spec's "the flag owns the whole path" is not literally true.
- Build 5 still keeps 81 of round 6's 134 B cases (39 anaphoric).
- Tried and rejected: zero-width joiner, fullwidth letters, no-break space in a relation, line separators, soft hyphen,
  capitalised underscore slugs, meaning-drift relations, odd inflections.

## What follows

All four lenses: GO holds as registered, with the caveats above carried into the verdict. Before any flip: check the
joined reply (not only each sentence), route every render path through the flag, adjudicate the anaphora and
clause-name cases blindly, and rule on the answer cost.
