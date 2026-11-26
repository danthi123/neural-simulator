# Voice ceiling read (2026-11-26): UNDEFINED by the registered rule (and by a design flaw); report-only, trained arms answer fewer untrained told facts

Prereg: `brain/research/2026-11-26-voice-ceiling-prereg.md` (Amendment R1 governs). Harness `brain/experiments/voice_ceil.py`
at d7132cd0f. Host: pool40 (node-mxl1462drl), all three seeds (the JSONs record host node-mxl1462drl but no device field; `where_ran.py` reports unknown) (s319 11:46-14:00, s320 14:00-16:12,
s321 16:12-18:23 UTC, rc 0 each). Result files: `2026-11-26-voice-ceiling/vc_full_s3{19,20,21}.json`, `verdict.json`.

## Registered verdict: UNDEFINED
`voice_ceil table` prints UNDEFINED, "a seed has no live, engaged night".
- **Engagement.** 7 of 9 nights fail the engagement rule for T or S. Live nights: s319 night 6 and s320 night 14;
  s321 has none.
  - The failures are on the loss-ratio leg: last-10/first-10 guard KL is 0.35-0.52 against the 0.3 bar.
  - Or they are on the agreement leg: 0.866-0.898 against 0.9.
  - The guard term is one target per step, so a 10-step mean is a noisy estimate. The rule was too tight for that
    design. This is a weakness of the rule, recorded, not repaired after the data.
- **Design flaw, found at read time.** R1's UNDEFINED bar "knows on the test items under 20 for A0" could never be met:
  - 2 test items × 3 nights × 3 seeds gives at most 18 told rows;
  - A0 knows all 18.
  So R1 returned UNDEFINED under any outcome. The pre-data review did not catch it, and neither did I.
- G1 (teacher) passed: 32 distinct targets, 21 name their key, 0 of the 21 in the user's voice, 11 declines.

## Report only (all 9 nights, engagement ignored; told rows, `VS.knows`, `CS.own2`, `CS.refusal`)
| items | arm | told | knows | user voice given knows | refusals |
|---|---|---|---|---|---|
| test (pixel, marguerite; never trained) | A0 | 18 | 18 | 10 | 0 |
| | T | 18 | 6 | 3 | 9 |
| | S | 18 | 5 | 2 | 5 |
| trained (rotating, own cues) | A0 | 36 | 19 | 3 | 6 |
| | T | 36 | 15 | 1 | 13 |
| | S | 36 | 1 | 0 | 12 |
| report (volvo, admin) | A0 | 18 | 8 | 0 | 5 |
| | T | 18 | 3 | 0 | 8 |
| | S | 18 | 4 | 0 | 8 |

- **Report-only signal: fewer answers, more refusals, on items never trained.**
  - Over all 9 nights, T knows 6 of 18 test rows against A0's 18, and has 9 refusals against 0. S: 5 of 18.
  - 7 of those 9 nights did not engage, so this is not a clean reading of converged training.
  - On the 2 engaged nights the test-item knows are A0 2/2 and 2/2, T 1/2 and 1/2, S 0/2 and 1/2.
  - At this n, T and S cannot be told apart. No keep-off arm was run, so the keep term's role is not measured.
- On the trained items T kept most of its knowledge (15 vs 19), while the sham lost almost all of it (1). The user
  voice there was already rare in A0 (3/19).
- **Blind hand-read** (fresh sonnet, 30 T and 30 S rows on the test and report items):
  - correct fact in second person: T 5, S 4;
  - correct fact in the user's voice: T 2, S 1;
  - refusal: T 14, S 11;
  - wrong fact: T 7, S 8;
  - other: T 2, S 6. S's "other" is mostly deflection, so S is a weak sham.
- This resembles the 11-18 context-distillation kill (commit 20fcf3497, distilled refusals). That is a comparison, not
  a measurement here.
- The smoke log printed one T number before R1 (n6 uv|knows 0/4). Those rows are not used here.
- **Skeptic** (fresh sonnet, refuter-lens block): the table and UNDEFINED SURVIVE, and so does the unreachable bar.
  "Erases" was PARTIALLY REFUTED as overstated, and "class closed at 1.5B" was REFUTED as an overclaim (UNDEFINED, 3
  seeds, one method). Both are reworded here.

## Decision
- No registered verdict. The method (context distillation of the in-context answer into the slept adapters, R1
  design) is **not rescued and not rerun**. The rule's flaws are recorded as flaws:
  - the unreachable A0 knows bar;
  - the noisy one-target engagement test.
  They are not read as a negative.
- Report only, the trained arms answered fewer untrained told facts and refused more. That is reason enough not to
  pursue this method. It is not a verdict on the capability.
- Timebox 11-28. This is the 9th lever on the voice defect without a pass.
- Next method, from the record and the architecture: act on what is stored and replayed. Either sleep replays the
  cortex's comprehended form of the telling, or a speaker tag travels with the episode (self/other, PREREQUISITES
  missing). Not a second training pass over the slept adapters. It needs a before-build first.
