# Stage-2 test2 read: runbook (prepared 2026-11-27, host-only, nothing here ran a model)

Prereg `brain/research/2026-11-23-stage2-test2-prereg.md` (+ Amendments T1-T4), freeze note `2026-11-23-stage2-test2-FREEZE.md`
(freeze commit `d965bc716b2bf600f450dc6d6dd748db66993d29`). Lives: 18 = {B, KP, PR} x seeds {42,43,44,100,101,102}, 15 days, AWS g5
fp16, landing time NOT predicted here (see section 1: lower bound ~11-28 09:20 UTC, set by the 6-instance cap; real time depends on the launcher). Read deadline 12-10, land 12-14. This file is the whole procedure; the files next to it are
`read_env.sh` (life lists), `preflight.py` (completeness and provenance), `g0_check.py` (G0), `dry/` (dry-run outputs).

## 0. Findings from the dry run (read these first)
Dry run = every scorer command on the three 11-27 smoke lives (seed 7, 2 days: `smokeB`, `smokeKP` (the 20261127-0920-KP copy,
2 nights; the 20261127-0920 copy is a 1-night partial), `smokePR`), plus every command again on the 15-day DEV lives
`_scratch/stage2/read/life{KP,Bg}_s319-321.json` (outputs in `dry/dev/`). Python: `/home/dant123/Projects/sim/.venv/bin/python`
(system python has no torch; importing `brain.eval.life` imports torch). No frozen file was touched or needs a fix.

| # | scorer / command | smoke result | DEV 15-day result |
|---|---|---|---|
| 1 | `life_bars` (B, KP, PR one arm per call; also all 3 together) | rc 0; every night-dependent bar UNDEFINED (P0, FM-v2, SAV-*, TRF-*, LG-KL), as expected at 2 days | rc 0 |
| 2 | `life_costs` registered mode | rc 0; all seeds UNDEFINED ("B life missing" x6, 42-102 not present) | - |
| 3 | `life_costs --dev` KP/PR vs B | rc 0; seed 7 UNDEFINED "B life ended after night 2" | rc 0; reproduces T1 exactly (G7 26/30, 26/30, 17/30 vs 14/30, 5/30, 11/30; G8a intact rate 0.019, 0.022, 0.137 (arm) vs 0 in B, hoff 0.087/0.130/0.000; G8b max night 0.727/1.000/1.000 vs 0.000; G10; G11 PASS) |
| 4 | `life_compare` KP and PR (`--d2pr2`), registered and `--dev` | rc 0; UNDEFINED, same reasons | rc 0; LG-P1/P2 PASS, G1 PASS, G2a FAIL, G2b PASS, G3 FAIL, G4 PASS (= T1) |
| 5 | `life_read --emit` (weeks 1,3 and week 1) | rc 0, 28 and 13 units | rc 0, 29 units, 18 pairs, 171 answer prompts |
| 6 | `life_read --emit-cross` (G9) | **CRASH** (below) | rc 0: 30 prompts per reader, all 6 canaries planted |
| 7 | `life_interview_read --emit-prompts` | rc 0, **0 prompts** (interviews are nights 5/10/15; a 2-day life has none) | rc 0, 501 prompts |
| 8 | `life_table` | rc 0 | rc 0 |
| 9 | `life_bars_null` (K=3, inner 20, smoke only) | rc 0 | - |
| 10 | score paths `--score`, `--score-cross`, `life_interview_read` scoring (synthetic reader files, NOT readings) | - | all rc 0; G9 UNDEFINED and AUROC "-" as they must be on synthetic input |

**The one crash (not a frozen-code bug, smoke-specific, but a live risk):**
`life_read --emit-cross ... --arm-lives smokeKP --b-lives smokeB` -> `ValueError: canary experience has no site on any unit (at most 2 per unit)`
(`life_read.py` line 1018). Cause: the "experience" canary is planted on an `open-feeling` battery row; the test2 bank draws 6 of its 12
open questions per seed with `random.Random(seed + 29).sample(LB.OPEN, 6)` and seed 7's draw has no "feeling" question (probe rows
on the smoke lives: open-advice, -chat, -knowledge, -story only). I recomputed the six real seeds' draws offline: **43, 44 and 102 carry a
"feeling" question (44 has two), 42, 100 and 101 do not**, so "experience" can be planted at TEST. Residual risk: in cross mode all six canaries must
plant (at most 2 per unit) or emission raises, and `wrong-episode`/`perspective` need a strict-right row whose telling is inside the unit
(on DEV K-P the old emit could plant only 4 of 6; the cross mode planted all 6 on DEV 319-321). If emission raises on the real lives:
that is an instrument failure BEFORE any reader output exists, so trying another `--rng-seed N` (emission-only parameter, default 0) does not
depend on any outcome, BUT it changes the decoy and duplicate placement, so the order is fixed now, before any reader output: retry strictly **1, 2, 3, ...** (no skipping, no choosing among emitted prompts by looking at them),
stop at the first seed whose emission succeeds, and use that one seed for every arm's G9 emit (and re-emit G5 at that seed only if the G5 emit itself raised). Write every seed tried, and that it was the first success, into the report. Do NOT edit `life_read.py`. If no seed works, G9 is UNDEFINED
(no KEEP, prereg section 6) and say why. Run the emit-cross dry check the moment 3+ complete seeds of an arm and B exist, not at the end.

**Hazards found (no crash, silent wrong reads):**
1. `life_bars.load()` (line 47) and `life_table` (line 188) accept a directory and glob `life_s*.json`. Test2 files are `life<ARM>_s<S>.json.json`, so a directory
   argument silently reads **0 lives** (empty table, rc 0). **Verified 11-27** by running both scorers on a directory holding a copy named `lifeB_s7.json.json`: `life_table` rc 0 with an empty table, `life_bars` printed "0 seeds: []", rc 0 (the same file passed explicitly was read: "1 seeds: [7]").
   Always pass explicit file paths (as `read_env.sh` builds them). `life_costs`, `life_compare`, `life_read`, `life_interview_read` take explicit file lists and do not glob.
2. `life_bars` pools whatever it is given by `seed`: three lives of different arms with the same seed print "seeds [7,7,7]". One arm per call.
3. Importing the scorers prints a `### RWKV-7 "Goose" enabled ###` banner on stdout. Use the `--out` files, never parse stdout.
4. `life_costs --out` takes a FILE (`costs_KP.json`); `life_compare --out` takes a PREFIX (`.json` + `.md`); `life_bars --out` a PREFIX; `life_table --out` a file;
   `life_read --out` a PREFIX. Mixing these overwrites nothing but drops the artifact where you do not look.
5. `code_commit` and `aws_ami_id` are only REPORTED by the scorers, not checked: the dry run shows smokePR on commit 1e91dce and smokeB/KP on 4ad30c0 with no flag. Real
   lives all launch from the freeze worktree, so `preflight.py` (step 1) checks one code_commit == `d965bc716...`, one AMI == the pinned `ami-012ba162b9cd2729c`, **one value across all 18 lives of each of torch, torch_cuda, gpu_name, tf32_matmul, tf32_cudnn (added 11-27; a difference or an unrecorded value is a PROBLEM line, in addition to the scorers' own per-seed UNDEFINED rule `life_costs.PROV_CHECKED`, section 4)**, and the 19 `frozen_sha256`
   entries against `git show d965bc716:<file>`. (On the smoke lives it flags `brain/brain.py` differing, correctly: they ran from the working branch.)
5b. **`tools/aws_brain_batch.sh` (the launch tool) was replaced twice AFTER the freeze** (prereg `2026-11-27-stage2-test2-amendment-T4.md` addendum 1 and 2), by atomic rename in `wt-test2`: freeze `4d94c7ab...` -> `8ba819ab7851509ac1a9718a0209a79adf4d4479200364b6f63b311a7dc52b5a` (added `< /dev/null`, did not fix the hang; KP42 launched with the old tool) -> `2b83d117ae4381bb4832e77aca984b077ed087ba43152a06d8b67fd85d970722` (final; KP43, KP44, KP100 still hung under addendum 1). It is not in FROZEN_FILES, runs only on the host, and changes nothing that runs on the instance, but no life JSON records which version launched it. `preflight.py` now prints the hash present in `wt-test2` and the recorded ones; the report header must list the three hashes and which lives launched under which (KP42 freeze version; KP43-KP101 as per T4 addendum 2 text; later lives final), taken from `launch_plan.log` timestamps. **Tell the provenance skeptic (section 6 item 2) explicitly** and give it the T4 amendment.
6. The DEV G5 read (`blind_KP`) supplied no pair reads, so its decoy bar FAILED by construction. At TEST supply `pairs.jsonl` answers too (step 5).
7. G0 (engagement) is NOT computed by any of the six frozen scorers; `kp_bars.py` / `d2pr2_bars.py` (DEV, hard-coded paths, unfrozen) did it. `g0_check.py` restates their rule and reproduces DEV s321 (bank 512/512 distinct, 63.5 tokens, 1.2% told key, 15/15 records). Hash it and cite it as an unfrozen helper.
   **The helper adds rules the prereg G0 row does not state; they are the helper's own choices** (taken from the DEV `kp_bars.py` 1a): one `subset_n`, 15 distinct `subset_hash`, `mean_keep_kl` > 0 from night 2, and
   the "told key" regex (whole-word, case-insensitive, over the keys of ledger entries with a telling; the prereg does not define "told key"). It now prints TWO verdict lines per seed: `G0 seed (prereg-literal)`
   (15/15 keep_bank records, bank_size 512, bank file with 512 texts, >= 90% distinct, mean >= 40 tokens, < 10% told-key hits) and `G0 seed (helper-stricter)` (adds the helper extras). Report both side by side; the
   prereg-literal line is the one that enters the section 6 decision; a seed that passes literal and fails stricter is stated as such, not silently failed. Both PASS on DEV s321.
8. A scorer file check: in `wt-mixday` (18 of the 19 FROZEN_FILES hash equal; `brain/brain.py` differs in wt-mixday, see section 2). `life_ladder.py` and `life_langguard.py` are imported by the scorers but are NOT in FROZEN_FILES (checked separately, equal). `life_bars_null.py` is post-freeze (report-only).

## 1. Where the lives are, and "landed"
- Launcher worktree (frozen, DO NOT EDIT OR COMMIT): `/home/dant123/Projects/sim/_scratch/wt-test2`. Log `_scratch/stage2/test2/launch_plan.log`; `asap.out`; pinned AMI `_scratch/stage2/test2/ami`.
- Each life: its own instance and batch id. Pull (`poll`, `AWS_BRAIN_PULL_WORK=1`) writes `wt-test2/_scratch/brainv2/aws/<batch>/` containing
  `life<ARM>_s<S>.json.json` (the life JSON; `life.py` refreshes it after every day, so a PARTIAL file exists mid-run), `life<ARM>_s<S>.json.md`, `.log`, `run.log`, and `work/life<ARM>_s<S>/{trunk,snap_trunk,life_state.json}`
  (K-P bank: `work/lifeKP_s<S>/trunk/keep_bank.json`). `<batch>` is `YYYYMMDD-HHMM[...]`, one per life. Not yet present at 11-27 18:45 (`wt-test2/_scratch/brainv2` does not exist; 4 K-P lives started: s42 16:33, s43 17:13, s44 17:53, s100 18:33, one per 40 min while the launch hang lasted; the hang is fixed in the launcher since T4 addendum 2, 1 s per launch verified on a pool).
- **Landing time (derived, not a promise).** K-P about 6.5 h, B about 4.4 h, PR about 4.5 h (prereg section 3): 18 lives about 92 instance-hours, at most 6 instances at once (T3). With the 4 K-P running at 18:45 (about 4.8 h done) about 87 instance-hours remain; at 6 instances fully packed from 18:45 that is about 14.5 h, so **no earlier than ~11-28 09:20 UTC**, later if slots idle, the per-UTC-day cap binds, or a life is resumed. The earlier "11-28 12:00 UTC" figure was unsourced and is dropped; read `launch_plan.log` / `run.log`, not this estimate.
- A life is LANDED only if its `run.log` has `end life<ARM>_s<S> rc 0` AND the JSON has 15 `nights` (a §10-killed life has fewer and is still a valid input, see section 4).
  Check: `grep -H "end life\|all done" /home/dant123/Projects/sim/_scratch/wt-test2/_scratch/brainv2/aws/*/run.log`; progress `tail -5 .../_scratch/stage2/test2/launch_plan.log`.
- Do not read lives while the launcher still pulls: wait for `launch_plan.log` to say "every life launched and pulled" (or note which seeds are missing; missing -> section 4).

## 2. Setup (once)
```bash
bash -c '
cd /home/dant123/Projects/sim/_scratch/wt-mixday
source brain/experiments/results/2026-11-27-test2-read-prep/read_env.sh      # arrays B KP PR (6 explicit paths each), RD, PY
mkdir -p $RD/lives && cp -n "${B[@]}" "${KP[@]}" "${PR[@]}" $RD/lives/ 2>&1 | tail -2   # optional: freeze copies next to the read
sha256sum "${B[@]}" "${KP[@]}" "${PR[@]}" > $RD/lives.sha256
# (a) the 19 FROZEN_FILES (life.py, prereg section 7; the same list each life records as frozen_sha256). Expect "same" x19 EXCEPT brain/brain.py (see below):
for f in brain/eval/life_bank_test2.py brain/eval/life_bank_test2.sha256 brain/eval/life_cues_test2.py brain/eval/life_cues_test2.sha256 brain/eval/life_interview_spec.py brain/eval/life_heldout.py brain/eval/life_read_spec.py brain/eval/life.py brain/eval/life_age_ladder.py brain/eval/life_bars.py brain/eval/life_leak.py brain/eval/life_costs.py brain/eval/life_compare.py brain/eval/life_read.py brain/eval/life_interview_read.py brain/eval/life_table.py brain/brain.py brain/sleep.py brain/cortex.py; do
  a=$(sha256sum $f | cut -c1-64); b=$(git show d965bc716:$f | sha256sum | cut -c1-64); [ $a = $b ] && echo "same $f" || echo "DIFF $f"; done
# (b) NOT frozen files (not in FROZEN_FILES, not in any life'"'"'s frozen_sha256): helpers the scorers import. Checked against the freeze commit as a courtesy; a DIFF here is a note, not a freeze violation:
for f in brain/eval/life_ladder.py brain/eval/life_langguard.py; do
  a=$(sha256sum $f | cut -c1-64); b=$(git show d965bc716:$f | sha256sum | cut -c1-64); [ $a = $b ] && echo "same(nonfrozen) $f" || echo "DIFF(nonfrozen) $f"; done
'
```
Checked 11-27 in `wt-mixday`: (a) 18 of 19 same; **`brain/brain.py` DIFFERS** (wt-mixday is on the working branch; the lives ran the freeze `brain.py`, and `preflight.py` checks the lives' recorded hashes against the freeze commit, not this tree);
(b) both same. A DIFF in (a) on any scorer/spec file (not `brain.py`/`sleep.py`/`cortex.py`, which the host scorers do not run models from) means stop: the read must run on frozen bytes (run from `wt-test2` read-only with `PYTHONPATH=wt-test2` and `--out` pointing into wt-mixday).
To read from the freeze tree instead: `export PYTHONPATH=/home/dant123/Projects/sim/_scratch/wt-test2` (writes go to `$RD` in wt-mixday only; never into wt-test2).
Optional sanity (about 75 s, host-only): `cd wt-mixday && LIFE_COSTS_REQUIRE_DEV=1 $PY -m pytest -q brain/tests/test_life_costs.py brain/tests/test_life_compare.py brain/tests/test_life_read_cross.py brain/tests/test_life_read.py` (freeze gate wants 0 skipped).
Everything below assumes: `bash -c 'cd /home/dant123/Projects/sim/_scratch/wt-mixday && source brain/experiments/results/2026-11-27-test2-read-prep/read_env.sh && ...'`.
Scorers are CPU-only, seconds each; no `gpu_queue`/`memcap` needed (they load no model). Never use `--dev` at TEST: `--dev` relabels verdicts and relaxes the seed rule.

## 3. Steps in order (commands exact; `$RD` = `brain/experiments/results/2026-11-28-test2-read`, adjust date)

### Step 1. Preflight (no verdicts)
```bash
$PY $PREP/preflight.py $AWSROOT 2>&1 | grep -v RWKV | tee $RD/preflight.txt
```
Expect 18 rows, nights 15, `ok`, one code_commit (`d965bc716...`), one AMI, one torch / torch_cuda / gpu_name / TF32 pair across lives, and the launcher-hash lines (final `2b83d117...`; copy them into the report header, section 0 item 5b). Any row not ok, or a missing life, goes into the report header verbatim and
feeds section 4 (the scorers will list the same seeds as UNDEFINED/not passing; preflight does not decide anything). `--smoke` mode exists for the dry run only.

### Step 2. `life_bars.py` (capability bars, per arm; prereg section 4 "Capability bars", P0-v2 gate first)
```bash
for A in B KP PR; do eval "L=(\"\${$A[@]}\")"; $PY -m brain.eval.life_bars "${L[@]}" --out $RD/bars_$A > /dev/null; done
```
Outputs `bars_{B,KP,PR}.json/.md` (P0, FM-v2, SAV-G/dR, CUE-*, TRF-*, LG-KL/COL, KILL-10, adapter norm; 10000 perms, rng 20261121). Rules: P0-v2 VALID on >= 5/6 else savings UNDEFINED; a capability pass is claimed only if B fails
that bar or the arm beats B on it (done by hand from `bars_*.json`, no frozen script computes arm-vs-B here; record both numbers); capability bars never change the verdict. Add the section 5 notes (TRF-K/TRF-C lenient) to every TRF line you quote.

### Step 3. `life_costs.py` (G7, G8a/b, G10, G11 setup check, leak on non-refusal rows, per prereg section 5)
```bash
$PY -m brain.eval.life_costs --arm "${KP[@]}" --b "${B[@]}" --out $RD/costs_KP.json > $RD/costs_KP.md
$PY -m brain.eval.life_costs --arm "${PR[@]}" --b "${B[@]}" --out $RD/costs_PR.json > $RD/costs_PR.md   # the stdout banner line is the only non-table line
```
Registered mode needs the six registered seeds. (`> .md` keeps the markdown table that is printed; delete the first banner line if quoting.) Expected-fail K-P: G7, G8, G10, G2a, G3 (prereg header); G11 can never FAIL.

### Step 4. `life_compare.py` (LG-P1/P2, LG-FIX, G1, G2a/b, G3, G4, d12; D2-PR2: leak primary and G6)
```bash
$PY -m brain.eval.life_compare --arm-name KP --arm "${KP[@]}" --b "${B[@]}" --out $RD/compare_KP > /dev/null
$PY -m brain.eval.life_compare --arm-name PR --d2pr2 --arm "${PR[@]}" --b "${B[@]}" --out $RD/compare_PR > /dev/null
```
Writes `compare_{KP,PR}.json/.md` (sign-flip p is reported, gates nothing). Both scorers share `life_costs.seed_status`, so they must list the same UNDEFINED seeds: diff the "UNDEFINED seeds" line of `compare_KP.md` and `costs_KP.md`.
A guard FAIL that falls inside the T1 device-read band is LABELLED so in the report (rounded values quoted in prereg Amendment T1 and sourced from the "Widened margins" table of `2026-11-23-test2-b-vs-b-margins.md`, which gives them unrounded: G2a arm >= B - 2.276, G1 arm >= B - 1.182, G2b arm >= B - 1.397, G4 per-seed share <= B + 0.051, G6 <= 1.401 x B, G11 <= B + 0.262; the rounded forms 2.28 / 1.18 / 1.40 / 0.051 / 1.40x / 0.26 match those, the sign is a "B minus" shortfall in the file, not a negative number; quote the unrounded ones); the verdict does not change and no margin is widened.

### Step 5. `life_read.py`: G5 (section 7 blind read) then G9 (cross-arm pairwise). Readers are model calls: one fresh `claude-sonnet-5-5` call per prompt line, two independent readers
(freeze note section 2; any other model ID is not a TEST v2 read). Each output line is JSON carrying the prompt's `id`. Emit is deterministic (`--rng-seed 0`, as DEV; if an emit raises, retry order is fixed: 1, 2, 3, ... see section 0); keep `blind_key.json` unread by the readers.
G5, per arm (B emitted and read ONCE and shared; re-read B only if its canary/decoy check fails):
```bash
for A in B KP PR; do eval "L=(\"\${$A[@]}\")"; $PY -m brain.eval.life_read --emit $RD/blind_$A --lives "${L[@]}" --weeks 1,3 --lived-context all --rng-seed 0 > $RD/blind_$A.emit.json; done
# per arm dir: reader_prompts.jsonl -> reads1.jsonl, reads2.jsonl; pairs.jsonl -> pairs1.jsonl, pairs2.jsonl (both A/B orders are separate lines);
#              answer_prompts.jsonl -> answers1.jsonl, answers2.jsonl   (2 readers each; {"id","verdict"} for answers; {"id","choice"} for pairs)
$PY -m brain.eval.life_read --score --key $RD/blind_KP/blind_key.json --reads $RD/blind_KP/reads1.jsonl --reads2 $RD/blind_KP/reads2.jsonl \
   --pairs $RD/blind_KP/pairs1.jsonl --pairs2 $RD/blind_KP/pairs2.jsonl --answers $RD/blind_KP/answers1.jsonl --answers2 $RD/blind_KP/answers2.jsonl --out $RD/blind_KP/score
```
(repeat for B and PR; outputs `score.json/.md`). Weeks 1,3 as the DEV reads. Checked against the prereg: the G5 row (section 5) says "section 7 blind read ... week 3, both orders ... no defect class absent in B"; the "section 7" it names is the design doc `2026-11-19-multiweek-life-design.md` section 7, whose material is the week-1 AND week-3 transcripts of each unit
(and whose separate milestone bar "week-3 blind defect count not above week 1" is NOT a G5 guard). So G5 is the WEEK-3 units only; the week-1 units are in the batch because the design's material includes them and so the readers see the same mix as at DEV; they do not enter G5. This is the literal reading, not a free choice.
Two pieces the prereg does not spell out, fixed here before any reading and recorded as runbook choices: G5 uses the **intact** week-3 units (`by_arm_week` key `intact|w3`, per arm vs B's `intact|w3`; hippocampus-off and both-off units reported, not gating), and "no defect class absent in B" is evaluated on counts **pooled over the six seeds**, both readers, with the per-seed table beside it.
G5 = no defect class (type counts in `by_arm_week["intact|w3"]["by_type"]` or `readers.<r>.per_unit[*].by_type`, both readers) present in an arm and absent in B. Instrument validity per arm
(canary recall >= 5/6 strict and loose, decoys, kappa) goes in the report; a failing canary check reruns that reader only.
G9, per arm vs B (prereg section 5 G9 + T1 rules):
```bash
$PY -m brain.eval.life_read --emit-cross $RD/g9_KP --arm-lives "${KP[@]}" --b-lives "${B[@]}" --cross-week 3 --rng-seed 0 --cross-id-prefix C > $RD/g9_KP.emit.json
$PY -m brain.eval.life_read --emit-cross $RD/g9_PR --arm-lives "${PR[@]}" --b-lives "${B[@]}" --cross-week 3 --rng-seed 0 --cross-id-prefix D > $RD/g9_PR.emit.json
# each g9_*/cross_pairs.jsonl (48 lines at six seeds: 24 groups x 2 orders; DEV 3 seeds gave 30) -> pairs1.jsonl and pairs2.jsonl from two independent readers (different calls; the CLI refuses the same file twice)
$PY -m brain.eval.life_read --score-cross --key $RD/g9_KP/cross_key.json --pairs $RD/g9_KP/pairs1.jsonl --pairs2 $RD/g9_KP/pairs2.jsonl --out $RD/g9_KP/score
```
Check `.emit.json`: `user_turns_match` true for every seed (else that seed UNDEFINED), `canaries_planted` all six, `over_60k` empty, `leaks` empty. Both readers valid and passing = PASS; either invalid = UNDEFINED; no re-read under changed rules.

### Step 6. `life_interview_read.py` (honest self-report; report-only; the extractor is a model call, `claude-sonnet-5-5`, one `{"unit","claims":[...]}` line per prompt, about 1,000 prompts per arm)
```bash
for A in B KP PR; do eval "L=(\"\${$A[@]}\")"; $PY -m brain.eval.life_interview_read --lives "${L[@]}" --emit-prompts $RD/int_$A; done
# $RD/int_<arm>/prompts.jsonl -> claims.jsonl (extractor), then:
$PY -m brain.eval.life_interview_read --lives "${KP[@]}" --key $RD/int_KP/blind_key.json --claims $RD/int_KP/claims.jsonl --out $RD/int_KP/score
```
(`--lives` must be the same lives, in any order, that built the key.) The extractor prompt hash is `b435055a...` (freeze note section 2).

### Step 7. `life_table.py` (report-only per-night D1/D2, collapse share, seconds, store, norm)
```bash
for A in B KP PR; do eval "L=(\"\${$A[@]}\")"; $PY -m brain.eval.life_table "${L[@]}" --out $RD/table_$A.md > /dev/null; done
```

### Step 8. G0 engagement (no frozen scorer; helper)
```bash
for S in $SEEDS; do echo "KP s$S"; $PY $PREP/g0_check.py kp $(pick KP $S) $(dirname $(pick KP $S))/work/lifeKP_s$S/trunk/keep_bank.json; done 2>&1 | grep -v "RWKV\|^$" | tee $RD/g0_KP.txt
for S in $SEEDS; do echo "PR s$S"; $PY $PREP/g0_check.py pr $(pick PR $S); done 2>&1 | grep -v "RWKV\|^$" | tee $RD/g0_PR.txt
```
K-P G0 per seed, prereg-literal = 15/15 keep_bank records (bank_size 512) + bank file (512 texts, >= 90% distinct, mean >= 40 tokens, < 10% told-key hits); a missing bank file = UNDEFINED. `g0_check.py` prints that line AND a helper-stricter line (one subset_n, 15 distinct subset_hash, keep_kl > 0 from night 2: the helper's own choices, not the prereg); quote both, decide on the literal one. D2-PR2 G0 = permanent and top-6 replay share below the strength-only reference on nights 5 and 10, on >= 5/6 seeds.
The bank refusal-pattern and "my name is" shares (prereg section 0, G0 report only) print on the report-only line (also needed for every bank, TEST included).

### Step 9. Report-only: life_bars_null (see section 7 below), K >= 200, background, one process per file set:
```bash
nice $PY -m brain.eval.life_bars_null "${B[@]}" --out $RD/null_B > /dev/null 2>$RD/null_B.err &   # then KP, PR; same master seed 20261127
```

### Step 10. Verdict sheet (prereg order; build by hand from the JSONs, numbers copied, never from memory)
Per arm: G0 | primary (K-P: LG-P1, LG-P2; D2-PR2: LEAK all rows and non-refusal rows) | G1, G2a, G2b, G3, G4 (compare) | G5 (read) | G6 (PR only, compare) | G7, G8a, G8b, G10 (costs) | G9 (read) | G11 setup check (costs; never a guard) | LG-FIX (reported) | capability bars (reported).
Decision (section 6): KEEP only if G0 holds, primary passes in full, G1-G10 all pass, and NO bar is UNDEFINED. NARROWED: G0 and every guard pass, primary passes in part. FAILED: any guard fails or primary fails outright. UNDEFINED: G0 fails or a bank is missing.
K-P is expected FAILED (Expected-fail header); D2-PR2 narrowed at best if the section-10 kill ends >= 2 of 6 PR seeds while B loses fewer. Any refusal rate above B's (pooled, either arm) needs a written Claude ruling (`tools/answer_rate.py`). A default flip needs land-result (skeptic + lesion that changes the reply).

## 4. UNDEFINED-seed rules (Amendment T1 + prereg section 6, as implemented in `life_costs.seed_status`, shared by `life_compare`)
- A seed is UNDEFINED (neither pass nor fail, dropped from every bar including pooled values, listed with its reason) when: B life missing; B life ended before night 15; split differs; any of device / cuda_precision / torch / torch_cuda / gpu_name differ between arm and B,
  or are unrecorded on either life (registered mode: "provenance unrecorded" = UNDEFINED, `--dev` only notes it); the arm's and B's both-off rows (night, question, reply) differ on the nights both completed (G11 setup mismatch).
- Arm life killed by the section-10 rule (collapse on 2 consecutive probe nights) or ending early: scored on its completed nights k = longest run 1..k with complete trunk night record, intact and hippocampus-off probes, and interview probes on nights 10/15; B restricted to the same nights for that seed; a bar needing a later night counts that seed as NOT passing. A killed PR seed counts as a collapse in G3 and, if >= 2 of 6 PR seeds die while B loses fewer, D2-PR2 is NARROWED at best.
- A missing ARM life = not passing (flag None). A missing/killed B life = seed UNDEFINED for that bar, not an arm fail; fewer than 5 scorable seeds (need = 5 of 6) = the BAR is UNDEFINED. An UNDEFINED seed is neither pass nor fail and the read WAITS for a rerun; no KEEP while any bar is UNDEFINED. A pooled clause is undefined whenever any seed is unscored.
- G0 fail or missing K-P bank file = the arm is UNDEFINED. G9: either reader invalid = UNDEFINED = no KEEP, no re-read with changed rules. Transfer clause "L-nosleep about 0" is UNDEFINED by design (no `--nosleep-arm`).
- A resumed life must be `--resume` on g5 fp16 only (freeze note section 7); a life that moved device or was relaunched is checked by the same provenance rule. The read waits for a rerun; it never substitutes another device's life.
- Pooled definitions: KL/NLL/H = mean over seeds (ratio of means); leak = sum(leaks)/sum(n). Sign-flip over the six per-seed differences (64 flips, one-sided) is reported, not a gate.

## 5. life_bars_null caveats (report-only notes; no prereg change)
Results: `2026-11-27-life-bars-null.md` (DEV lives s319-321, K=200, inner 300, master seed 20261127) and `...-null-sim6.md` (SIMULATED: 6 pseudo-seeds bootstrapped from them; not data). Label-shuffle within seed, `life_bars` scoring unmodified.
Attach these lines wherever a transfer bar is quoted:
- **TRF-K is lenient: it false-passes 50.5% of label-shuffled replicates alone** (sim6: 43%). **TRF-C false-passes 15%** (sim6: 10.5%). Neither can carry a transfer claim alone; observed TRF-K 0.222 sat at rank 73 of 200 (frac_ge 0.363, not beyond p95) and TRF-C 0.830 at rank 15 (frac_ge 0.075, not beyond p95) on the DEV lives. Report any TRF-K/TRF-C PASS as non-discriminating alone.
- The transfer SECTION needs TRF-T4 too: composite false-pass 0.005 (sim6 0.005). TRF-T4 itself false-passes 0.025 (sim6 0.030), "inconclusive" within 2 binomial SE of 0.05; observed 0.812 was beyond p95 (rank 5, frac_ge 0.025), not p99. Quote transfer as "T4 beyond null, K and C lenient".
- SAV-G false-pass 0.035 (inconclusive); SAV-dR 0.050 (sim6 0.030) is not a p-value bar; observed SAV values (DEV) were FAIL and inside the null (frac_ge 0.617 and 0.960). P0 false-pass 0.000. CUE-R (0.000) and CUE-D (0.000) are trusted, observed beyond p99 on DEV (CUE-R 0.095 is a FAIL verdict by its own threshold).
- These numbers are three-seed DEV. At six real seeds each rate moves; the null is rerun on the TEST lives per arm (step 9) before the transfer line is written, and its per-bar false-pass rates replace the DEV ones in the report. TRF-B0/N0, CUE-H/HL, LG-*, KILL-10 and FM-v2 do not depend on the shuffled labels and are not in the null. TRF-K and CUE-D show seed count "None" in the table because they are pooled-statistic bars with no per-seed count; their false-pass rates (TRF-K 0.505, CUE-D 0.000) are still measured.
- Percentile rule: "beyond null" iff statistic >= null p95 (frac_ge <= 0.05), "strongly" iff >= p99; the false-pass rule applies per p-value bar (trusted only if <= 0.05, inconclusive within 2 SE). K=200 with inner 300 permutations is cheap on CPU; the registered observed run keeps life_bars' 10000.

## 6. Five-skeptic plan (prereg section 3: sonnet only, about 1M tokens; land-result section 2)
Run AFTER steps 1-8 are written to `$RD` (so skeptics read files, not memory); all five in parallel, fresh context, read-only, model `claude-sonnet-5-5` (a skeptic is not a TEST reader, so the tier rule applies: sonnet; no opus/Fable unless two skeptics split on a verdict).
Each gets ONLY: the prereg + amendments, the freeze note, `$RD/*.md/*.json`, the six scorer sources, the life JSONs, and the refuter-lens block pasted verbatim from `.claude/skills/land-result/SKILL.md` (REFUTER-LENS-BEGIN..END). Default REFUTED when unsure. Output: per claimed number REPRODUCED / NOT, plus what would flip a verdict.
1. RECOMPUTE: independent code from the raw life JSONs (not importing `life_bars/costs/compare/leak`) for LG-P1/P2, G1-G4, G7, G8a/b, G10, leak all and non-refusal rows, per arm; report any number that differs by more than rounding.
2. SETUP AND PROVENANCE (lens c): `preflight.txt`, one commit/AMI/device/torch, frozen hashes, both-off equality, the UNDEFINED/killed/missing list and whether `life_compare` and `life_costs` agree, `python tools/where_ran.py <life paths>` quoted, K-P bank files present, that no seed was substituted or rerun on another device, and that T1 band labels are applied but no margin widened.
3. ANSWER-RATE AND COSTS (lens b): `tools/answer_rate.py` on arm vs B, G8 pooled rates and collapse nights against the prereg's expected-fail list, the regex hit lists for G7/G8/G10 (false positives such as "I'm Dan's assistant", `ok` vs refusal), whether any arm lowers the answer rate on any seed (a flip needs a Claude ruling).
4. TRANSCRIPTS AND READERS: G5 and G9 validity (canary recall, decoys strictly below both-off, same-position rate, kappa), the model ID `claude-sonnet-5-5` on every reader/extractor call, hand reading of at least 3 lived days per arm and 2 batteries (blind), whether the experience-claim count matches G10, and whether a lesion (hippocampus-off, both-off) changes the REPLY, not only a field.
5. ENGAGEMENT AND WORDING (lens a): G0 per arm (keep_bank records and bank files, replay-share engagement nights 5 and 10), the section 6 decision logic applied bar by bar (UNDEFINED => no KEEP; conjunctive; shared-B correlation stated), report-only items kept report-only (capability bars, LG-FIX, interview, TRF-K/C lenient notes), every load-bearing word against `docs/TERMS.md` ("KEEP", "drift reduced vs B" never "language guard fixed"), nothing re-tuned on v2 (section 9 rotation).
Record skeptic verdicts verbatim in the read note; PARTLY-REFUTED wording is fixed before commit.

## 7. Budget and timing
Scorers: seconds each (about 3 s of imports on smoke; the 6-life runs add 10000-perm tests in life_bars, tens of seconds at most). Model calls (prereg estimate): extractor about 1,100 prompts per arm (about 2M input tokens per arm; DEV had 501 prompts for 3 seeds), G5 readers about 1.2M per arm, G9 about 0.6M, skeptics about 1M: about 10M for three arms (B read once). Order to keep the readers busy: emit all (steps 5, 6) first, dispatch G9 and extractor
prompts in parallel with the scorers of steps 2-4.
