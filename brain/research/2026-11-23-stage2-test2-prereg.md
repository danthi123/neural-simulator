# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-21-stage1-registered-test-1.5b.md; brain/experiments/results/2026-11-22-kp-dev-read-1.5b.md
# Expected-fail: K-P FAILS G7 (self-name) and G8 (refusal) as at DEV, likely G10/G11 too, unless the §0 K-P-f screen fixes the bank; language "fixed" bar (n15 KL <= 2x n1; K-P DEV 3.5-5.3x); P0-v2 if forgetting is total; cued recall, transfer, self-report (Stage-1 negatives)
# Stage-2 registered TEST on split v2: B vs K-P (or K-P-f) (vs D2-PR2 if it passes its DEV prereg), 1.5B, six seeds (prereg, 2026-11-23, revised after review)

Written before any v2 TEST life. Rules: design `2026-11-19-multiweek-life-design.md` §5 (bars), §9 (rotation), §13
(command); split v2 `2026-11-22-fresh-test-split-design.md` (age ladder, P0-v2, FM-v2, §6 decision rules).

## 0. Before the freeze: the bank-cost decision (decided now)
- DEV K-P fails G7 and G8 below on 3/3 seeds (§5 validation). A TEST of K-P unchanged is a probable FAIL (about $40 of
  its lives), and a fix after it forces TEST v3 (§9). Possible mechanism: the bank is the prior's own chat text. Rough
  audit of the two CPU proxy banks (s319/s320, 512 texts each, "User:/Bot:" frames): "don't know / not sure" in 16 and
  10 texts, "my name is / I'm an assistant" in 4 and 10. Not shown to cause the cost.
- **Decision:** a cheap DEV screen of **K-P-f** (K-P with bank texts whose non-User turns match the G8 refusal pattern or
  a self-name pattern dropped and resampled) gets its own prereg by 11-25: the CPU one-night gate on s316 plus 3 DEV
  lives (s319-321, AWS g5 fp16, about $20), read on LG-P1/P2, G7, G8 by 12-01.
  K-P-f passes (primary plus G7 and G8 on 3/3): the TEST arm is K-P-f, K-P is not run. Otherwise K-P runs unchanged,
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
    prose, both-off confabulation where B refused (invented 55 vs 22-27).
  - d12 nights 1-4: s320 -1.24 of 6 (21/29 vs 27/29).
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
- Every life JSON records (code change before the freeze) `device`, `cuda_precision`, TF32 flags, GPU name, `host`,
  `code_commit`, sha256 of each §7 file; `life.py` refuses a registered run from a dirty tree.
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
  are close to its own objective. They cannot carry a KEEP alone; G5 and G9-G11 test conversation quality.
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
| G11 | both-off confabulation (below) | rate <= B + 0.15 |
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
  the freeze (canary and decoy checks as §13).
- **G10 experience claims.** Lived trunk turns (`res["lived"]`, brain replies). Regex
  `\bi(?:'ve| have)\s+(?:seen|been to|been there|visited|tasted|watched|heard|met|had it|never had|never been)\b|\bi saw\b|\bi went\b|\bi slept\b`.
  Rate per life. The reader's experience-claim count is reported beside it.
- **G11 both-off confabulation.** Both-off rows of the same kinds as G8b. Rate = share neither a G8 refusal nor `ok`.
  Both-off differs between arms at DEV, and FM-v2 uses both-off as its reference, so a confabulating both-off would
  loosen FM-v2. FM-v2 is also reported against B's both-off.
- **DEV validation (computed 11-23 on `_scratch/stage2/read/life{KP,S1F}_s319-321.json`). K-P fails each bar on 3/3:**
  G7 wrong 0.87/0.87/0.57 vs B 0.40/0.10/0.23; G8a worst arm 0.087/0.130/0.137 vs 0; G8b nights in §1, B max 0;
  G10 10/16/38 vs 4/3/0 of 166; G11 1.00/1.00/0.00 vs 0.24/0.74/0.74 (fails 2/3). Leak on non-refusal rows: K-P 0/148,
  3/150, 14/138 vs B 39/156, 54/152, 42/152 (G4 holds). `life_costs.py` must reproduce these; regexes never edited
  after a v2 row exists.
- **Margins and noise, honestly.**
  - Every G7-G11 margin was chosen after the DEV read, to sit below the DEV effect.
  - No repeat-B noise estimate exists. Before the freeze, every guard is computed B vs B on DEV (`lifeBg` AWS fp16 vs
    `lifeS1F` CPU, s319-321). A margin that a B-vs-B comparison fails on any seed is widened to that seed's |difference|.
    That rule is fixed now, and the widened values are written into the freeze note.
  - G1 widened 0.5 -> 1.0 (B seed spread 1.82-2.55; DEV K-P +1.45 to +2.0). G7's s319 B is already 0.40.
  - G3 cannot see the DEV refusal nights (B collapsed 3/6 in Stage 1); G8 carries it. G2's -1.0 margin is just inside the DEV -1.24, so a repeat of that cost on a seed fails G2a there.
  - G2's nights 5-15 window was first chosen with the DEV values in view, after the K-P read, and the D2-PR2 "precedent"
    is from the same day. Both windows are therefore bars now.

## 6. Decision (per arm, against B)
- **KEEP + flip-candidate:** G0 holds, the primary passes in full, G1-G11 all pass.
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
- reader and extractor model IDs and prompt templates (by hash), the K-P-f decision (§0), the arm list and the
  B-vs-B margin widening (§5).
- **Gates for the freeze:** the v2 DEV row check `lifeAL_s322` is back and passes v2 §6. `life_costs.py` reproduces the
  §5 DEV counts. The G9 mode is validated. The D2-PR2 and K-P-f DEV reads are committed, or treated as absent.
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
12. G3 blind to refusal nights. **Fixed:** stated; G8 carries it.
13. §13 gaps. **Fixed:** `code_commit`, hashes, dirty-tree refusal, reader/extractor IDs and prompts, nosleep UNDEFINED,
    label-shuffle note, s322 gate, `--inquiry` default.
14-16. Claude spend (token estimate), AWS plan (<= 7 instances, two capped days), noise (B-vs-B rule, G1 1.0, G7 0.25).
17. Minor. **Fixed:** "fixed" expected FAIL kept; no Stage-1-guard-fixed wording; G8 one rule in table and text.
