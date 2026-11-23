# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-21-stage1-registered-test-1.5b.md; brain/experiments/results/2026-11-22-kp-dev-read-1.5b.md
# Expected-fail: K-P FAILS G8 and G10 (3/3 at DEV), G7 and G2a (2/3), and G3; G11 cannot fail (same-device B lifeBg, Amendment T1), unless the §0 K-P-f screen fixes the bank; language "fixed" bar (n15 KL <= 2x n1; K-P DEV 3.5-5.3x); P0-v2 if forgetting is total; cued recall, transfer, self-report (Stage-1 negatives)
# Stage-2 registered TEST on split v2: B vs K-P (or K-P-f) (vs D2-PR2 if it passes its DEV prereg), 1.5B, six seeds (prereg, 2026-11-23, revised after review)

Written before any v2 TEST life. Rules: design `2026-11-19-multiweek-life-design.md` §5 (bars), §9 (rotation), §13
(command); split v2 `2026-11-22-fresh-test-split-design.md` (age ladder, P0-v2, FM-v2, §6 decision rules).

## 0. Before the freeze: the bank-cost decision (decided now)
- DEV K-P fails G8 and G10 on 3/3 seeds, G7 and G2a on 2/3, and G3, against the same-device B `lifeBg` (Amendment T1). A TEST of K-P unchanged is a probable FAIL (about $40 of
  its lives), and a fix after it forces TEST v3 (§9). Possible mechanism: the bank is the prior's own chat text. Rough
  audit of the two CPU proxy banks (s319/s320, 512 texts each, "User:/Bot:" frames): "don't know / not sure" in 16 and
  10 texts, "my name is / I'm an assistant" in 4 and 10. Not shown to cause the cost.
- **Decision:** a cheap DEV screen of **K-P-f** (K-P with bank texts whose non-User turns match the G8 refusal pattern or
  a self-name pattern dropped and resampled) gets its own prereg by 11-25: the CPU one-night gate on s316 plus 3 DEV
  lives (s319-321, AWS g5 fp16, about $20), read against `lifeBg_s319-321` (the same-device B) on LG-P1/P2, G7, G8,
  G10, G2a and G3 by 12-01.
  K-P-f passes (primary plus G7, G8, G10 and G2a on 3/3, and G3): the TEST arm is K-P-f, K-P is not run. Otherwise K-P runs unchanged,
  the Expected-fail line stands, and the TEST is aimed at those exact costs. The freeze commit fixes the arm list.
- Every bank (TEST and DEV) records the share of its texts with the G8 refusal pattern and with "my name is" in a
  non-User turn (G0, report only).

## 1. Record brief
- Stage 1 (TEST v1, B): language guard FAIL (n15 KL 12.8-17.2x n1, 6/6); P0 NOT VALID 0/6; cue, transfer,
  self-report negatives; §7 FAIL (both-off preferred 6/6).
- K-P DEV (s319-321, `keep_bank="prior"`, AWS fp16; B on CPU fp32): KEEP at DEV, gate-overridden. KL n15 0.323 vs B
  2.724 (0.118x), 3/3. NLL n15 2.18-2.26 vs B 4.11-4.75. Leak 17/460 vs 135/460; on non-refusal replies 17/436 vs 135/460.
- K-P DEV costs (recounted on the JSONs, trunk untagged, intact + hippocampus-off, 30 nova rows per seed):
  - self-name wrong (`ok` False) 26/26/17 of 30 vs B 12/3/7; friend name ("I'm Dan") 13/21/7 vs B 6/0/2. The other
    wrong K-P names: Philippa, Alex, Assistant, Aiko and others. The reader note's "B 0/18" was wrong: B is 8/90.
  - refusal on answerable rows (G8 pattern): worst arm per seed 0.087/0.130/0.137 vs B 0; collapse nights s319 hoff n15
    (8/11), s320 hoff n11-12 (11/11), s321 intact n15 (11/11). Reader: lived "I've seen it" claims, generic assistant
    prose, both-off confabulation where B refused (invented 55 vs 22-27; a device artefact: both-off is identical
    between arms on one device, Amendment T1).
  - d12 nights 1-4: s320 -1.24 of 6 (21/29 vs 27/29), against the CPU B. Against the same-device B (`lifeBg`), G2a is
    -1.03 on s320 and s321 (Amendment T1).
- D2-PR2 (`replay_priority="loss_a"`) DEV read committed 11-23 (`results/2026-11-23-d2pr2-pra-dev-read-1.5b.md`): PASSES 3.1-3.3
  by the letter of its §4, so the D2-PR2 arm ENTERS. Registered risk: on DEV s321 PR-a collapsed on probe nights 1-2 (top reply
  share 0.50, 0.45: the §10 per-seed kill would have ended it), B 0/3. In TEST the §10 kill is live for every arm; a PR-a seed it
  ends counts as missing for later-night bars and as a collapse in G-collapse, and if it ends >= 2 of 6 PR-a seeds while B loses
  fewer, D2-PR2 is NARROWED at best whatever its primary shows. Leak DEV: 48/460 vs B 135/460 (-64%).

## 2. Arms, seeds, command
| Arm | Switch | Enters the TEST |
|---|---|---|
| B | none (defaults) | always |
| K-P or K-P-f | `--keep-bank prior` (K-P-f: its flag, §0) | always, exactly one of the two (§0) |
| D2-PR2 | `--replay-priority loss_a` | only if its DEV read passes its own prereg §4 |

- **D2-PR2 entry (exact):** a DEV read committed before the freeze states PR-a passes 3.1-3.3 of its prereg; else no
  D2-PR2 arm, no add-on. K-P already cuts the leak (0.037 vs 0.294); no marginal value over K-P is claimed.
- Seeds 42/43/44/100/101/102, every arm, 15 days. Command (the §13 command plus v2):
  `python -m brain.eval.life --model 1.5b --seed S --days 15 --weekend --split test2 --age-ladder
  --full-probe-nights 1,5,10,15 --frozen-probe --span --transfer --pruned-a --cue-ladder --interview --lang-guard
  --device cuda [<arm switch>] --homes <home>_<arm>_sS --out <out>/life<arm>_sS`
- Defaults kept: `--pruned-a-day 11`, interviews 5,10,15, `--cuda-precision fp16`. `--inquiry` is unset, so it follows
  the Brain default at the frozen commit. `--nosleep-arm` is not run, so the transfer clause "L-nosleep about 0" is
  UNDEFINED. No other switch, no `--days` cut.

## 3. Compute, precision, budget
- **All arms on AWS g5 (A10G), CUDA fp16**, the device of the DEV K-P lives and of the DEV B-fp16 lives
  (`lifeBg_s319-321`); same-device arms close the confound. W fp32 is dropped (no parity run covering the K-P bank
  sampler, no timing; 12-18 serial lives would pass 12-09).
- TF32: `rwkv/model.py` sets `allow_tf32 = True` (matmul, cudnn) at import, so GPU fp32 matmuls (PlasticCortex training
  included) are TF32 on every arm: uniform, as DEV. `--cuda-precision fp32` (68a05818b) gives CPU numerics only once
  `Cortex.__init__` sets both False; out of scope here.
- Every life JSON records (code change before the freeze) `device`, `host` and a `provenance` block: `cuda_precision`,
  torch and torch CUDA versions, GPU name, both TF32 flags, `code_commit`, AWS AMI id; also sha256 of each §7 file; `life.py` refuses a registered run from a dirty tree.
- `--resume` only on g5 fp16; a seed never moves device. Launch checklist: each instance copies `keep_bank.json` and the home off before termination.
- **Lives budget (real timings).** B about 4.4 h (night 649-662 s plus the 13 min ladder), K-P about 6.5 h
  (1141-1175 s nights, 1802-1877 s night 1), D2-PR2 about 4.5 h (estimated as B; no GPU timing). 18 lives about
  92 g5-hours, about $90 (v2 note rate). The cap is $50/day; the 11-22 $150 raise is gone.
  - At most 7 instances: 12-04 6 K-P + 1 B (about $44); 12-05 5 B + 2 D2-PR2, then 4 D2-PR2 as B frees (about $49).
    Without D2-PR2, 12-05 is 5 B (about $22). The g5 vCPU quota for 7 is checked first; the W GPU return is irrelevant.
- **Claude-side budget (sonnet only), estimate:** extractor about 1,100 units per arm batched 20 per call (about 2M
  input tokens per arm), §7 readers about 1.2M per arm, G9 pairs about 0.6M, 5 skeptics about 1M: about 10M for three
  arms, 7M for two. B is read once and shared; a reader reruns only on a canary failure.

## 4. Primary bars (six seeds, each arm vs B, computed by `life_compare.py`, §7)
**Common rule:** direction on >= 5/6 seeds; "beats B" = paired sign-flip test over the six per-seed differences
(64 flips, one-sided p < 0.05), or the pooled item-level permutation (items nested in seeds) where §5 asks for it.
**K-P primary: language drift (K-P DEV prereg §3.2, scaled to 5/6).**
- LG-P1: night-15 `lang_guard.kl`: mean over the six seeds <= 0.5x B's mean over seeds (as DEV pooled), AND below B on
  >= 5/6.
- LG-P2: `lang_guard.nll_on` rise at night 15, per seed: NLL_arm(n15) - NLL_B(n1) <= 0.5 x (NLL_B(n15) - NLL_B(n1)) on
  >= 5/6. The arm's own night 1 is reported beside it. Night 10 is reported, not a bar.
- Caveat stated now: K-P's keep term is KL to the pretrained model on prior samples from a "User:" frame, so LG-P1/P2
  are close to its own objective. They cannot carry a KEEP alone; G5, G9 and G10 test conversation quality.
**D2-PR2 primary (if it enters):** the stock leak (`life_leak.py`, nights 10+15, both arms) below B on >= 5/6 and >= 30%
lower pooled. It must hold both on all rows and on **non-refusal rows** (G8-pattern replies dropped from numerator and
denominator). LG-P1/P2 are reported; its language guard is G6.
**"Language-drift defect fixed"** = the §5 bar (n15 KL <= 2x n1 on >= 5/6, the arm's own lives), expected to FAIL for
K-P (3.5-5.3x at DEV). A KEEP here is "drift reduced vs B" and is never reported as the Stage-1 language guard fixed.
**Capability bars (§5 as amended in §13, per arm, `life_bars.py` on v2):**
- P0-v2 precondition (VALID >= 5/6; else savings UNDEFINED). Savings, cue, transfer, self-report, FM-v2, §7: as registered.
- A capability pass is claimed only if B fails that bar or the arm beats B on it. The label-shuffle check is not built:
  a clause that also passes on both-off is reported non-discriminating. Capability bars never change the verdict.

## 5. Guards (any fail = that arm FAILS)
| ID | Guard | Bar (per seed, >= 5/6 unless stated) |
|---|---|---|
| G1 | H hippocampus-off, nights 5-15 | arm >= B - 1.0 of 6 |
| G2a / G2b | hippocampus-off day-1-2 recall (d12: H or rotating rows, item min(told_days) <= 2, `ok`, rate x 6), **nights 1-4** and **nights 5-15**, two bars | arm >= B - 1.0 of 6, each. Nights 1-15 reported |
| G3 | collapse (`life_bars.language`, share >= 0.4) | seeds-ever per probe arm <= B's and <= 3/6 |
| G4 | stock leak (n10+15), all rows AND non-refusal rows | not higher than B, pooled and on >= 5/6, both ways |
| G5 | §7 blind read (`life_read.py`, spec 6cf3690a, 2 readers, week 3, both orders) | no defect class absent in B |
| G6 | language-guard KL n15 (D2-PR2 only) | <= 1.25x B |
| G7 | self-name loss (below) | wrong rate <= B + 0.25 |
| G8 | refusal (below), (a) rate and (b) collapse night | (a) <= B + 0.03, each probe arm; (b) no seed-arm-night >= 0.6 unless B >= 0.6 there (any seed) |
| G9 | cross-arm blind pairwise read (below) | arm preferred or tie on >= 4/6 |
| G10 | experience claims in lived turns (below) | rate <= B + 0.03 |
| G11 | both-off setup check (Amendment T1; not a cost guard) | arm both-off rows equal B's, else the seed is UNDEFINED. Rate reported |
| G0 | engagement. K-P: bank 512, >= 90% distinct, mean >= 40 tokens, < 10% told key, record every night, bank file present. D2-PR2: its 3.1 on >= 5/6 seeds, nights 5 and 10 | fail or missing bank = UNDEFINED |

**Scorers.** New host-only `brain/eval/life_costs.py` plus `brain/tests/test_life_costs.py`, written and hash-frozen
before the first run. Rows: probes with arm intact or hippocampus-off, `line == "trunk"`, no `tag` (as `life_leak._rows`),
nights 1-15. Text is `life_leak._norm(reply)`. Both-off is used only for G11.
- **G7 self-name.** Rows: kind "H", item "nova". Rate = share with `ok` False, intact and hippocampus-off pooled, about
  30 rows per seed. Because the 15 nights are near-duplicates, the effective n is about 5 per seed. The friend-name rate
  (`\b(?:my name is|i'?m called|i am called|call me|i'?m|i am)\s+{friend}\b`, `friend = brain_kw.get("friend","Dan")`)
  is reported beside it, not a bar. "I'm Dan's assistant" matches that regex; it is listed, not excluded.
- **G8 refusal.** Refusal = the first sentence (split `[.!?]`) matches
  `^\W*(?:(?:i'?m|i am) sorry,?\s*)?(?:but\s+)?(?:i\s+(?:don'?t|do not)\s+(?:know|understand)\b|i'?m not sure\b|i am not sure\b)`,
  or the reply contains "don't know what you mean".
  - (a) Denominator: answerable rows, kind H (not dan/nova), rotating, or neardup, excluding `told` False, regardless of
    `ok`. Rate per seed and per probe arm.
  - (b) Night share over every non-identity row (H not dan/nova, rotating, neardup, never; about 11 a night).
- **G9 cross-arm pairwise.** New `life_read.py` mode: the arm's intact unit vs B's intact unit, same seed, week 3,
  `--lived-context all`, 2 readers. A winner only if both A/B orders agree; else tie. Validated on DEV K-P vs B before
  the freeze (canary and decoy checks as §13). A reader is valid only if complete, no decoy beats both-off, decoys
  strictly lose to both-off in both orders on >= ceil(4n/6) seeds, same-position rate over non-duplicate groups
  <= 1/3, and canary copy not preferred >= 5/6. Combination: either reader invalid = UNDEFINED; both valid = PASS if
  both pass, FAIL if either fails, else UNDEFINED.
- **G10 experience claims.** Lived trunk turns (`res["lived"]`, brain replies). Regex
  `\bi(?:'ve| have)\s+(?:seen|been to|been there|visited|tasted|watched|heard|met|had it|never had|never been)\b|\bi saw\b|\bi went\b|\bi slept\b`.
  Rate per life. The reader's experience-claim count is reported beside it.
- **G11 both-off setup check (Amendment T1).** Rate = share neither a G8 refusal nor ok, reported only. Both-off is
  identical between arms on one device; the DEV difference was a device artefact. FM-v2's both-off reference is
  arm-independent.
- **DEV validation (computed 11-23 on `_scratch/stage2/read/life{KP,S1F}_s319-321.json`). K-P fails each bar on 3/3 (against the CPU B; superseded by T1):**
  G7 wrong 0.87/0.87/0.57 vs B 0.40/0.10/0.23; G8a worst arm 0.087/0.130/0.137 vs 0; G8b nights in §1, B max 0;
  G10 10/16/38 vs 4/3/0 of 166; G11 1.00/1.00/0.00 vs 0.24/0.74/0.74 (fails 2/3; a device artefact, Amendment T1).
  These are against the CPU B (`lifeS1F`); the same-device numbers are in Amendment T1. Leak on non-refusal rows: K-P 0/148,
  3/150, 14/138 vs B 39/156, 54/152, 42/152 (G4 holds). `life_costs.py` must reproduce these; regexes never edited
  after a v2 row exists.
- **Margins and noise, honestly.**
  - Every G7-G11 margin was chosen after the DEV read, to sit below the DEV effect.
  - The B-vs-B margin-widening rule first written here is replaced by Amendment T1: no margin is widened.
  - G1 widened 0.5 -> 1.0 (B seed spread 1.82-2.55; DEV K-P +1.45 to +2.0). G7's s319 B is already 0.40.
  - G3 sees the DEV refusal nights on the same device (K-P hoff seeds-ever 2 vs B 1, intact 1 vs 0: G3 FAILS); G8
    also carries them. G2's -1.0 margin: same-device K-P is -1.03 on s320 and s321, so G2a fails there.
  - G2's nights 5-15 window was first chosen with the DEV values in view, after the K-P read, and the D2-PR2 "precedent"
    is from the same day. Both windows are therefore bars now.

## 6. Decision (per arm, against B)
- **KEEP + flip-candidate:** G0 holds, the primary passes in full, G1-G10 all pass. G11 is a setup check, not a guard.
- **UNDEFINED seeds and bars** (Amendment T1): an UNDEFINED seed is neither pass nor fail; a bar with fewer than 5
  scorable seeds is UNDEFINED, and the read waits for a rerun. No KEEP while any bar is UNDEFINED.
- **G9 UNDEFINED -> no KEEP**, and no re-read with changed rules.
- **NARROWED:** G0 and every guard pass, the primary passes in part (LG-P1 or LG-P2 alone, or 4/6). The arm stays OFF.
- **FAILED:** any guard fails, or the primary fails outright. **UNDEFINED:** G0 fails or a bank is missing.
- B is shared by two arm verdicts, so they are correlated and uncorrected. Each verdict is conjunctive (about 12 bars),
  so KEEP is not inflated, but false-FAIL is likely on single lives. That risk is accepted and stated.
- **Default flip (land-result):** a KEEP arm only, with the TEST's B, numbers read from the JSONs, a fresh-context
  skeptic, and a lesion that changes the reply. Any refusal rate above B's (pooled, either arm, even within G8) also
  needs a written Claude ruling (`answer_rate.py` rule). Two KEEPs flip one at a time; the combination is a new method.
- **Rotation (§9):** any method change after this read writes TEST v3. v2 is never re-tuned on. **§10 kills:** collapse
  on 2 consecutive probe nights only (§13).

## 7. Frozen before the first run, gates, timebox
One freeze commit holds this note, the code commit, and the sha256 of each file below. Every life records them (§3):
- `life_bank_test2.py`, `life_cues_test2.py` (their sha256 files); `life_interview_spec.py` 4b70347f...,
  `life_heldout.py` adc7d902..., `life_read_spec.py` 6cf3690a...;
- `life.py`, `life_age_ladder.py`, `life_bars.py`, `life_leak.py`, `life_costs.py`, `life_compare.py` (new: all
  vs-B bars, LG-P1/P2, G1-G4, G6, d12 copied from `kp_bars.py`, sign-flip test), `life_read.py` (with the G9 mode),
  `life_interview_read.py`, `life_table.py`, `brain/brain.py`, `brain/sleep.py`, `brain/cortex.py`;
- reader and extractor model IDs and prompt templates (by hash), the K-P-f decision (§0), the arm list.
- **Gates for the freeze:** the v2 DEV row check `lifeAL_s322` is back and passes v2 §6. `life_costs.py` reproduces the
  §5 DEV counts. DEV tests run with `LIFE_COSTS_REQUIRE_DEV=1`, 0 skipped. `life.py` writes the §3 provenance fields,
  and the shared same-device check (`life_costs.seed_status`, used by `life_compare.py`) is in the frozen scorers: a seed
  whose arm and B differ in device, `cuda_precision`, torch, torch CUDA or GPU name is UNDEFINED. The G9 mode is validated. The D2-PR2 and K-P-f DEV reads are committed, or treated as absent.
- **Timebox:** K-P-f prereg 11-25, its read by 12-01. Freeze by 12-02, launch 12-04-05 (§3), read by 12-10 (bars,
  costs, §7 and G9 reads, 5 skeptics), land 12-14. If the launch slips past 12-04, D2-PR2 is dropped first; nights and
  seeds are never cut.

## Review (11-23, 17 defects; all verified on the DEV JSONs or the code, none rejected)
1. G8 vacuous (`ok` denominator), 80% at the ceiling. **Fixed:** `ok`-free answerable denominator, `told` False out,
   night share over all non-identity rows at 0.6, "not sure / don't understand" added, DEV counts stated (3/3 fire).
2. G7 missed other names; "B 0/18" false. **Fixed:** `ok`-failure rate (69/90 vs 22/90), friend regex beside, n_eff ~5.
3. Abstention passes the leak. **Fixed:** G4 and D2-PR2 primary also on non-refusal rows (DEV numbers given).
4. No conversation-quality bar. **Fixed:** G9 pairwise, G10 experience claims, G11 both-off confabulation, LG caveat.
5. Expected FAIL unstated. **Fixed:** header; §0 bank audit and K-P-f decision before the TEST.
6. G2 window post hoc. **Fixed:** two bars (1-4, 5-15), margin vs -1.24 and disclosure stated.
7. TF32 (verified `rwkv/model.py` 11-12, 179-180). **Fixed:** W fp32 dropped, flags recorded, `Cortex` fix named.
8. Parity weak, W timing unrealistic. **Fixed:** all arms AWS fp16; resume device class; precision recorded.
9. Comparison code unfrozen. **Fixed:** `life_compare.py` hashed; pooled, `nll_on`, sign-flip test defined.
10. Lost banks. **Fixed:** copy-off checklist; missing bank = UNDEFINED.
11. D2-PR2. **Fixed:** G0 6-seed form, marginal value disclaimed, shared-B correlation and false-FAIL stated,
    composition fixed by the freeze, drop-first kept.
12. G3 blind to refusal nights (false on the same device, see T1). **Fixed:** stated; G8 carries it.
13. §13 gaps. **Fixed:** `code_commit`, hashes, dirty-tree refusal, reader/extractor IDs and prompts, nosleep UNDEFINED,
    label-shuffle note, s322 gate, `--inquiry` default.
14-16. Claude spend (token estimate), AWS plan (<= 7 instances, two capped days), noise (B-vs-B rule, G1 1.0, G7 0.25).
17. Minor. **Fixed:** "fixed" expected FAIL kept; no Stage-1-guard-fixed wording; G8 one rule in table and text.

## Amendment T1 (2026-11-23, before the freeze; no v2 row exists). Scorers built, DEV-validated, review findings ruled
Scorers: `brain/eval/life_costs.py` (G7, G8a/b, G10, G11, leak on non-refusal rows, the shared seed rule) and
`brain/eval/life_compare.py` (LG-P1/P2, LG-FIX, G1-G4, G6, d12, D2-PR2 primary, sign-flip), both Fable-reviewed and fixed.
DEV reproduction tests run with `LIFE_COSTS_REQUIRE_DEV=1` (0 skipped is part of the freeze gate). Reproduced exactly
against `lifeS1F` (with the setup check off): G7, G8a, G8b, leak (all and non-refusal rows), every `kp_bars.py` and
D2-PR2 PR-a number. Rulings:
- **No widening.** The §5 B-vs-B pair (lifeBg AWS fp16 vs lifeS1F CPU) differs by DEVICE. Two AWS fp16 B lives of s321
  on different instances (20261122-0922 cut, 20261122-1027) are byte-identical on every probe reply of nights 1-4 (90/90,
  60/60, 60/60, 60/60) and every night scalar except wall time; both-off rows are byte-identical between arms on one device.
  Rerun noise on one device and software stack is zero (s321, nights 1-4, B only; nights 5-15 and K-P unverified).
  Sensitivity of a life to a small perturbation is not zero; the device read is its only estimate. Guards stay unwidened,
  raising false-FAIL (accepted, §6). A guard FAIL inside the device-read band is labelled so in the report; the verdict
  does not change. The device-read values (G2a B-2.28, G1 -1.18, G2b -1.40, G4 +0.051, G6 1.40x, G11 +0.26;
  `results/2026-11-23-test2-b-vs-b-margins.md`) are that band; they are never applied as margins.
- **Same device, recorded (code).** `life.py` writes a `provenance` block at a fresh life start: `cuda_precision`, torch,
  torch CUDA, GPU name (cuda only), both TF32 flags, `code_commit`, AWS AMI id. It draws no RNG and changes no state;
  replies are identical with or without it (tested). A seed is UNDEFINED if its arm and B differ in device,
  `cuda_precision`, torch, torch CUDA or GPU name where both lives record them; a field missing on either side is noted
  "provenance unrecorded", never a fail (under `--dev`; registered mode: UNDEFINED, below). The DEV lives record only device. K-P vs `lifeS1F` is UNDEFINED on every bar.
- **G11 is a setup check, not a cost guard (code).** Both-off is the same pretrained cortex for every arm. Per seed, the
  arm's both-off rows (night, question, reply) must equal B's, else that seed is UNDEFINED (setup mismatch). An UNDEFINED
  seed is neither pass nor fail, and the read waits for a rerun. The rate, and the old rate <= B + 0.15 comparison, are
  reported only. FM-v2's both-off reference is therefore arm-independent. `life_costs.py` (the only scorer with a G11 bar): G11 is never FAIL (any core
  FAIL, e.g. a missing arm life, is UNDEFINED with pass None).
- **Dirty-tree guard (code).** `life.py` records the sha256 of each §7 file and `git status --porcelain -- brain/`, and
  refuses to start a `--split test2` life from a dirty brain/ tree, or without git unless `BRAIN_CODE_COMMIT` and
  `BRAIN_CODE_CLEAN=1` are set (AWS git-archive launches must pass both); dev/test unchanged.
- **Registered-mode provenance (code).** Without `--dev`, a checked provenance field unrecorded on either life makes the
  seed UNDEFINED ("provenance unrecorded"); `--dev` keeps it a note only.
- **Seeds (code, one rule in both scorers).** An ARM life killed by the §10 rule is scored on its completed nights; bars
  needing later nights count that seed as not passing. B is restricted to the arm's completed nights for that seed. A
  missing arm life counts as not passing. A seed missing because B's life is missing or killed is UNDEFINED for that bar,
  not an arm fail; if fewer than 5 B seeds are scorable the bar is UNDEFINED. Every UNDEFINED seed is listed with its
  reason (B missing, B ended before night 15, split differs, provenance differs, both-off differs). A test checks that
  both scorers mark the same seeds.
- **G10 counts lived TRUNK turns only** (the §5 wording; the old 10/16/38 vs 4/3/0 of 166 included the 32 prunedA lines).
  Same-device B numbers: K-P 9/12/20 vs lifeBg 4/1/1 of 134 (s319 fails narrowly 0.067 vs 0.060).
- **G8b any-seed.** A seed-arm-night >= 0.6 is excused if B >= 0.6 at that arm and night in ANY seed. The same-seed
  variant is reported beside it; every excused cell is listed.
- **G9 mode built and validated on DEV** (`life_read.py --emit-cross/--score-cross`; `results/2026-11-23-g9-validation-kp-vs-b.md`).
  Read 1 FAILED its decoy check (decoy compared against B; stands as failed). Changed before read 2, written before it was
  read: decoy reference = the seed's both-off unit (the §7 rule), exactly two readers required, choices parsed strictly
  (else UNDEFINED), a positive-discrimination gate (no decoy above both-off, decoys strictly below both-off in both orders
  on >= `_need(n)` = ceil(4n/6) seeds (4 of 6 at TEST), same-position rate <= 1/3, else UNDEFINED), duplicates
  report-only, all six canaries planted with copy-not-preferred >= 5/6. Combination: either reader invalid -> UNDEFINED;
  both valid -> PASS only if both pass, FAIL if either fails. Tie-passing alone would have passed read 1; it is the §7
  letter but a loosening in effect. Read 2: both readers valid, arm preferred 3/3 (DEV, not a claim). Read 2 passed
  validity at the exact minimum for both readers (decoys strict 2/3, canary 5/6); the perspective canary was preferred by
  both readers in both reads. G9 has never FAILED a real arm: readers preferred DEV K-P 3/3 despite its G7/G8/G10 costs,
  so a G9 PASS means 'not worse by blind read' and does not offset those guards. G9 UNDEFINED -> no KEEP (§6). Limits:
  canaries only catch a reader who prefers the defect; decoy prompts are identifiable by form; 4 of 6 canaries sit on
  s321 at DEV.
- **G3** needs no widening. The refusal-regex boundary case "...I don't know." (leading punctuation) is not a refusal
  under the verbatim rule (0 DEV rows affected); "i'm sorry, but i don't have information" is not a refusal either
  (verbatim rule kept; it only mattered for G11).
- **DEV read against the same-device B** (K-P vs `lifeBg_s319-321`, 3 seeds, no UNDEFINED seed; DEV, not a claim):
  - `life_costs.py`: G7 fails 2/3 (wrong 26/30, 26/30, 17/30 vs 14/30, 5/30, 11/30; s321 passes). G8a fails 3/3 (intact
    2/104, 2/92, 13/95; hoff 9/104, 12/92, 0/95; B 0 everywhere). G8b FAILS (s319 hoff n15 0.73, s320 hoff n11 and n12
    1.00, s321 intact n15 1.00; no cell excused; the same-seed variant also fails). G10 fails 3/3. G11 passes 3/3
    (both-off equal; confabulation 42/42 on both sides).
  - `life_compare.py`: LG-P1 passes 3/3 (KL n15 0.351/0.294/0.323 vs 2.093/2.693/2.616); LG-P2 passes 3/3. G1 passes
    (+1.18/+0.27/+1.36); G2b passes; G4 passes on all rows and on non-refusal rows. G2a fails 2/3 (s320 and s321 both
    -1.03). G3 fails (hoff seeds-ever 2 vs 1, intact 1 vs 0). LG-FIX fails.

## Amendment T2 (2026-11-23, before the freeze): arm list fixed
- K-P identity gate read (`results/2026-11-23-kp-identity-gate-read.md`, 2 refuters not refuted): K-P-U UNDEFINED, K-P-0 FAIL
  (bar 1, marginal). No variant enters. **Arms: B, K-P (unchanged, `--keep-bank prior`), D2-PR2 (`--replay-priority loss_a`).**
  K-P's G7/G8/G10/G2a/G3 Expected-fail stands (header). K-P-f is not built in this window (the identity prereg's timebox bars a
  second bank or mask rule); the §0 K-P-f screen is therefore not run.
