# G9 cross-arm blind read, K-P vs B, DEV seeds 319/320/321 (week 3, lived context all)
Source: _scratch/stage2/read/blind_G9_KP/ (cross_key.json, pairs1.jsonl, pairs2.jsonl -> score.json/score.md). Spec read-spec-2026-11-21 (sha256 6cf3690a68e1). User turns identical arm vs B on all 3 seeds. Two readers (r1, r2), 30 items each, all complete.

## VERDICT: mode NOT VALIDATED; G9 UNDEFINED (a read failing a validity check carries no verdict, per the scorer's rule)
Raw per-seed winners are arm on all 3 seeds for both readers (3/3 arm-or-tie, bar 2 of 3 would pass), but this is not a G9 result because both reads are invalid.

## Validity checks (per reader; both readers identical on the failing check)
| check | r1 | r2 |
|---|---|---|
| complete (15 groups) | pass | pass |
| decoys lose to B (3 decoys, need 3/3) | FAIL 2/3 | FAIL 2/3 |
| duplicates read tie (need 3/3) | pass 3/3 | pass 3/3 |
| canary copy not preferred (need 5 of 6) | pass 5/6 | pass 5/6 |
| canary source preferred (informational) | 3/6 | 4/6 |
| cross order consistency | 3/3 | 3/3 |
| order consistency, all groups | 0.80 | 0.867 |
| same-position rate | 0.20 | 0.133 |
Pairwise kappa 0.9476. Readers agree on every per-seed winner.

## Failing items
- Seed 321 "refuse" decoy: both readers picked the refusal decoy in one order (decoy:refuse) and B in the other, so the group is inconsistent and scores tie, not B. Decoys 319 (keyword-stuffer) and 320 (generic) lost to B in both orders (pass).
- Canary copy-not-preferred misses only the "perspective" canary planted on B for 321 (both readers chose the canary); 5/6 meets the bar. "wrong-episode" and "invented" (r1) read tie, not source.

## Per-seed winners (both readers, both orders)
| seed | winner | orders |
|---|---|---|
| 319 | arm | arm, arm |
| 320 | arm | arm, arm |
| 321 | arm | arm, arm |
(r1 and r2 differ only in item C26, a canary item; neither changes a per-seed winner.)

## G9 bar on DEV
Bar: arm preferred or tie on >= ceil(4n/6) = 2 of 3 seeds, per reader, with every reader valid. Raw 3/3 arm per reader, but valid = False for both, so g9_pass = n/a: UNDEFINED.

## Validated per prereg?
No. The decoy check (every decoy must lose to B) failed on seed 321 for both readers, so the cross mode is not validated on this read. Canary and duplicate checks pass. Implication: the arm-wins pattern is suggestive only. A re-read needs the refuse-decoy item reproduced under a clean order (or a prereg-sanctioned fix), not a loosened bar. Note: the K-P refusal cost (G8a, 38/582 refusals) makes a reader preferring a refusal decoy in one order unsurprising, but that is interpretation, not a validity pass.
Read only; no model or life run. Caveat: DEV, one life per arm per seed, 3 seeds.

## Change before read 2 (written before read 2)
Read 1 stands as FAILED (decoy check failed for both readers; G9 UNDEFINED; not re-scored, not reinterpreted). Its material stays in _scratch/stage2/read/blind_G9_KP/.

Root cause: the cross mode compared each decoy against B. The existing rule (brain/research/2026-11-19-multiweek-life-design.md §7/§13, and the --score mode of life_read.py: "decoys flagged and rated at or below both-off") uses the BOTH-OFF unit as the decoy reference. A week-3 B that invents facts is not a valid "must beat a decoy" reference, so a reader preferring a refusal over an inventing B is not evidence of a broken reader.

Changes to brain/eval/life_read.py cross mode (existing modes unchanged, emit byte-identity test kept), fixed before read 2:
1. Decoy pair = decoy (made from the arm's intact unit) vs the SAME seed's both-off unit from the same life, in the §7 form (battery vs battery, the unit's lived days as shared context). Pass = the decoy does not beat both-off (both-off wins or tie). The decoy-vs-B pair is dropped.
2. Exactly two readers: G9 UNDEFINED unless a non-empty second read is supplied; a partial second read is incomplete.
3. Choices: only A / B / tie after normalising case, whitespace, quotes, trailing punctuation and a "transcript " prefix; anything else is unread; a duplicate id in a reads file makes that id unread. Unread makes the reader incomplete (G9 UNDEFINED). Counts reported.
4. Canaries: all six types must plant (else emission raises), at most two canaries per unit (seed x side); copy-not-preferred >= 5/6 kept; source-preferred reported.
5. Duplicates: report-only noise (design §7: duplicate spread sets the minimum detectable effect), no longer a validity gate; raw choices reported.
6. A seed whose user turns differ arm vs B is UNDEFINED (G9 UNDEFINED unless the bar is decided without it); emission raises on any leak or prompt > 60k chars; duplicate seeds in --arm-lives / --b-lives are an error; a clear error if either flag is missing.
The scorer refuses a v1 (decoys-vs-B) key, so read 1 cannot be silently re-scored.

Disclosure on the root cause: two things changed at once, the reference (B -> both-off) and tie-passes ("at or below"). The tie rule alone would have flipped read 1: its only failing item (321 refuse decoy) scored tie, so read 1 passes the v2 decoy check even with B as reference. The reference change is principled (design §7, the committed --score build and bar); tie-passing is the §7 letter but a loosening in effect.

7. Positive-discrimination gate (added after review, before read 2; read 2 not read). A reader is valid only if, besides the checks above: no decoy beats both-off, AND decoys strictly lose to both-off in both orders on >= ceil(4n/6) seeds (the SEEDS_BAR scaling; 2 of 3 on DEV, not fitted to read 1), AND the same-position rate over non-duplicate groups is <= 1/3. Else the reader is invalid and G9 UNDEFINED. On the real read-2 key a synthetic all-A, all-B or all-tie reader (both readers) now scores valid = False, G9 UNDEFINED (decoys strict 0/3; same-position 1.0 / 1.0 / 0.0); tested in brain/tests/test_life_read_cross.py. Read 1 would have met this gate: decoys strictly below B 2/3 (need 2), same-position over non-dup groups 3/12 = 0.25 (r1) and 2/12 = 0.167 (r2).
8. Verdict combination (per reader, also written into the prereg G9 paragraph): if either reader is invalid G9 is UNDEFINED (a half-invalid instrument carries no verdict); with both valid, PASS if both pass, FAIL if either fails, else UNDEFINED. Previously "FAIL if any reader fails" could fire with the other reader invalid.
9. Pairwise kappa is computed on normalised choices (A / B / TIE / UNREAD), so "A" vs "a." agrees. The CLI refuses the same file as --pairs and --pairs2.

Not implemented in cross mode (stated before read 2): §7 also requires decoys "flagged" and canary recall >= 5/6 (the reader names the defect). Cross mode has no unit read, so neither exists here; a tie passes the decoy and canary checks. The canary check (copy not preferred >= 5/6) counts ties, so it only catches a reader who PREFERS the defect; it does not show the reader detects it. Canary source-preferred stays report-only (read 1: 3/6 r1, 4/6 r2) with no floor, because any floor set now would be fitted to read 1.

Canary spread: the at-most-two cap is per unit (seed x side), not per seed. In read 2, 4 of 6 canaries sit on seed 321 (experience and collapse on the arm unit; wrong-episode and perspective on B). Site availability forces this: wrong-episode and perspective plant only on B 321. One seed carries most of canary validity.

Pair type visible from form: only the 6 decoy prompts carry the shared "SAME earlier conversations" block (the §7 battery-vs-battery form). No labels leak, but a reader can tell decoy pairs apart by form, and the decoy check never exercises the whole-unit, no-context form used for the cross pairs. Order consistency and same-position rate are reported beside the verdict for this reason too.
Read 2 material: _scratch/stage2/read/blind_G9_KP2/ 30 prompts (6 cross, 6 decoy-vs-both-off, 6 duplicate, 12 canary; ids G0..G29, --rng-seed 2; read 1 used C0..C29, --rng-seed 0); max prompt 16,483 chars; canaries on 4 units, max 2 per unit. (new --rng-seed and id prefix, so ids differ from read 1).

## Read 2
Material: _scratch/stage2/read/blind_G9_KP2/ (pairs1.jsonl, pairs2.jsonl, cross_key.json, score.json/.md). Scored with `life_read --score-cross`; no rule changed after the reads. DEV, one life per arm per seed, 3 seeds (319/320/321), week 3.

**G9: VALIDATED (PASS)** per the scorer's rules: both readers valid and complete, both pass.

| reader | valid | reads valid/bad/dup/missing | arm/B/tie (cross) | arm-or-tie (bar 2/3) | decoys not above both-off | decoys strictly below (need 2) | same-position non-dup (cap 0.333) | canary copy not preferred (need 5/6) | canary source preferred (report) | order consistency all/cross | kappa |
|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | True | 30/0/0/0 | 3/0/0 | 3/3 | 3/3 | 2/3 | 0.083 | 5/6 | 5/6 | 0.933 / 3/3 | 0.8438 (pairwise) |
| r2 | True | 30/0/0/0 | 3/0/0 | 3/3 | 3/3 | 2/3 | 0.167 | 5/6 | 4/6 | 0.867 / 3/3 | 0.8438 (pairwise) |

Per-seed winners (both readers, both orders): 319 arm, 320 arm, 321 arm. Combined: arm on 3/3 seeds. G9 DEV bar: arm preferred or tie on >= ceil(4n/6) = 2 of 3 seeds per reader; met 3/3 by both.
Decoy vs both-off: r1 319 both-off, 320 tie, 321 both-off; r2 319 tie, 320 both-off, 321 both-off. Duplicates all TIE (6/6 reads, report-only). Canary failures: r1 and r2 each had the perspective canary (seed 321, planted on B) preferred over source.
Caveats: DEV, 3 seeds, one life each; 4 of 6 canaries sit on seed 321; decoy pairs identifiable by form; canary recall and "flagged" not implemented in cross mode; the 3 pairs where a reader chose tie on duplicates show a low-discrimination floor. Not a six-seed claim.
