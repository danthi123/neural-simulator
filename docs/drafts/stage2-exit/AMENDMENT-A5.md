# Stage-2 exit gate, Amendment A5: two informational host-ownership arms in C2 `infer` (Claude, 2026-10-26)

bar-sanity: N/A informational arms only; no bar, counted arm, turn, seed or scored field is registered or changed
host-decision: none the two arms only intervene on and read the existing host list to measure who owns the answer; they add no host code that decides a reply

Amends `research/findings/2026-09-30-stage2-exit-gate-PREREGISTRATION.md` (kept byte-identical, so `PREREG_SHA` is
unchanged) together with Amendments A3 and A4. Filed on 2026-10-26, after the 2026-10-26 gate run and before the next
gate run is emitted. The only data read are dev seeds 7 and 1 (`research/findings/2026-10-26-c2-kb-scramble-host-ownership.md`),
which govern nothing (prereg section 11). Every bar, counted arm, turn, seed, prediction and scored field of every
criterion is unchanged.

## A5.1 Two informational arms in group `infer` (reported, never counted)

- `kb_scramble`: after the four teach turns, the patient strings of the told facts in `OneBrainComposer.kb` (the list the
  possessive-chain route scans) are deranged, and the bound spiking vectors are left untouched. The question: do
  `derived_svo` and `derived_from` of `p_prey_eat` follow the scrambled list?
- `kb_scramble_spiking_control`: the same derangement, with the route's own fallback condition forced (`_distinct_patients`
  reports nothing scannable), so both hops become spiking `query_patient` reads. The question: do they then follow the
  bound vectors?

Both are informational. Neither is ever a pass condition, a precondition or part of the group or stage verdict, and neither
is read by `--aggregate` for any verdict. They report engagement (the list read back deranged, the conclusion-relevant fact
changed; a manipulation that did not hold is reported as not engaged, never as "no effect"), what the conclusion follows,
the route's list reads and spiking reads, and which turns differ from `intact_a`. Every pre-existing arm keeps its turn
table, env, patch and kind, pinned by a golden hash in `tests/test_stage2_kb_scramble.py`.

## A5.2 Lane

Pool lines: 198 -> 210 (6 seeds x 2 arms = 12 more jobs, all `infer`). Each line declares `mem_gb=9` under `memcap.sh 10` and
`max_age_h`, like every line. The `infer` group peaked at about 6.4 GB per arm at dev seed 7 and took 1246 s on the
workstation (prereg A2). The 12 jobs come to about 4.2 h at workstation speed and about 5.8 h on the pool (~1.4x slower).
`--no-informational` drops them (180 lines remain, all counted).

## What this does NOT establish

- Two dev seeds, one probe turn and one question shape. The `infer` group is one deterministic measurement across seeds
  (2026-10-26 diagnosis), so more seeds replicate the read and do not add independent evidence.
- The amendment attributes nothing about the C2 criterion, which FAILED 0/6 at the 2026-10-26 gate because the permuted
  teach was refused by the D4 comprehension gate (`research/findings/2026-10-26-stage2-fail-diagnoses.md`).
- The control is a forced fallback inside the arm's own worker process, not a production path.

## Consequences for the packet

- `PREREG_SHA` is unchanged (`24dac499e`). The prereg file is not touched.
- `GATE_SHA` must be pinned at a revision that contains this file and the runner and tests of commit `8491e0df1`
  (`git cat-file -e $SHA:docs/drafts/stage2-exit/AMENDMENT-A5.md`). The next emitted pool lines carry the two arms.
- No dev line is re-run.
