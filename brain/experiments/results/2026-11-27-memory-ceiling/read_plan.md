# Memory reply ceiling: read plan (prepared 11-27, before the seeds land ~21:00 UTC)

Prereg `brain/research/2026-11-27-memory-ceiling-prereg.md` (section 5 pass rule, section 7 outcomes); design
`brain/research/2026-11-27-fable-review-dg-port.md` section 3. Code `brain/experiments/memory_ceiling.py` (`score`, `verdict`,
`handread`). Rule fixed before data: no threshold, arm or population changes at read time.

Target: verdict + hand-read + skeptic inside one hour of the last seed landing.

## 0. Is it done? (watch, do not poll the models)
```
for p in "40 319" "41 320" "42 321"; do set -- $p
  SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes pool$1 'cat ~/brain-lane/route_repo/_scratch/route/kr/mc_done_s'$2'.txt; tail -n 3 ~/brain-lane/route_repo/_scratch/route/kr/mc_s'$2'.log'
done
```
Finished = a line `s<seed> rc 0 <ts> <host>` in each `mc_done_s*.txt`. rc != 0 or no such line: do NOT score; the verdict
would be INCONCLUSIVE (missing seed). A resume (re-run the same `run_node.sh <seed>` on that node) skips finished units, but only
under the same commit and stage0 hash.

## 1. Pull (only the small files; never `work/`, which holds snapshot copies)
Seed to node: s319 pool40, s320 pool41, s321 pool42. Pull into ONE directory, because `score` takes a single `--dir` and the
unit files are seed-prefixed (`u_s319_n5_OFF.json`, `dec_s319_n5.json`), so they do not collide.
```
cd /home/dant123/Projects/sim/_scratch/wt-mixday
D=brain/experiments/results/2026-11-27-memory-ceiling; mkdir -p $D/pulled $D/logs
SSHO="ssh -o IdentitiesOnly=yes"
for p in "40 319" "41 320" "42 321"; do set -- $p
  R=pool$1:~/brain-lane/route_repo/_scratch/route/kr
  SSH_AUTH_SOCK= rsync -av -e "$SSHO" --include='u_s*.json' --include='dec_s*.json' --exclude='*' $R/mc_s$2/ $D/pulled/
  SSH_AUTH_SOCK= scp -o IdentitiesOnly=yes $R/mc_s$2.log $R/mc_done_s$2.txt $D/logs/
  SSH_AUTH_SOCK= ssh -o IdentitiesOnly=yes pool$1 'cat ~/brain-lane/route_repo/COMMIT; sha256sum ~/brain-lane/route_repo/'$D'/stage0.json' > $D/logs/node_s$2.txt
done
```
(Plain `scp -r pool40:...mc_s319` also works but drags `work/`; avoid it. Output files are written atomically via .tmp then
rename: `rm -f $D/pulled/*.tmp` if a pull caught one.)

Pre-score integrity (2 minutes; all must hold or the read is INCONCLUSIVE, say so, do not rescue):
1. Counts: `ls $D/pulled/dec_s*.json | wc -l` = 33 (3 seeds x nights 5-15); `ls $D/pulled/u_s*.json | wc -l` = 6 arms x
   (33 - voids). Known void: s321 n5 (so 192 units, 32 nights non-void if only that night voids). Any other void is reported.
2. One commit and one stage0 hash across all three nodes. `score` returns `INCONCLUSIVE: mixed provenance` otherwise (it reads
   `commit` and `stage0_sha` from every unit and dec file), so compare `logs/node_s*.txt` first: the three COMMIT lines and the
   three stage0.json sha256 must agree. If the pools ran different commits, the verdict is INCONCLUSIVE by rule; re-running one
   seed on a fresh out dir under the matching commit is the only repair (new run, not a rescore).
3. Logs: `grep -il "error\|Traceback\|killed" $D/logs/mc_s*.log`; nothing, or explained.
4. Host/device: `python tools/where_ran.py $D/pulled/u_s319_n5_A.json` (device cpu fp32 expected; note any 3090 run).

## 2. Score
```
python -m brain.experiments.memory_ceiling score --dir brain/experiments/results/2026-11-27-memory-ceiling/pulled \
  --lives brain/experiments/results/2026-11-26-voice-split/lives/lifeKR_s3{19,20,21}.json.json \
  --out brain/experiments/results/2026-11-27-memory-ceiling/verdict.json
```
(use `.venv/bin/python` from the repo root: system python has no pytest/torch.) Run from the worktree, CPU only, seconds. It also
writes `pulled/handread_sheet.json` (blind, shuffled labels) and `pulled/handread_key.json` (label -> arm; do NOT open the key
until the hand-read is saved).

The terminal print shows only status, checks and `A_reproduces_banked` [reproduced rows, rows]. Everything else is in
`verdict.json`: read these keys, quote numbers from the file:
- `status`, `checks.headroom_clauses` (H1..H6), `checks.oracle_minus_off`, `_nonH`, `_per_seed`, `cluster` (wins/losses/ties),
  `oracle_minus_sham`, `uv_knows_oracle_minus_off`.
- Engagement (lens a): `arms.<ARM>.engaged` {told_forced, told_engaged, told_no_right_episode} for ORACLE, SHAM, ORACLE-rekey,
  K-REFIT. An arm with forced ~0 or engaged << forced did not test what it claims: a small ORACLE - OFF is then UNDEFINED, not
  NO-HEADROOM, in the write-up (the verdict code does not gate on it; the read does).
- `arms.<ARM>.fire`, `fire_by_kind`, `fire_never` for OFF, A, K-REFIT (reported, not barred).
- `pairs` (`<ARM>-OFF`: pooled, nonH, per_seed, clusters), `A_reproduces_banked` (reported, not a gate), `void_nights`.
- ORACLE-rekey and K-REFIT are reported, never part of the verdict.

## 3. What each verdict triggers (Fable review section 3, prereg section 7; no rescue, no relaxed bar)
Thresholds, absolute counts on surviving rows (not rescaled for voids): HEADROOM needs all of H1 ORACLE-OFF >= 15, H2 non-H >= 6,
H3 positive on 3/3 seeds, H4 cluster wins > losses, H5 ORACLE-SHAM >= 15, H6 uv_knows ORACLE-OFF <= 5. NO-HEADROOM if
ORACLE-OFF < 8 or ORACLE-SHAM < 8 (checked first). Otherwise UNDEFINED. INCONCLUSIVE if a control arm / seed is missing, the
(seed, night) unit sets or told counts differ across OFF/A/ORACLE/SHAM or the non-void dec set, or provenance is mixed.

| verdict | what happens next |
|---|---|
| **NO-HEADROOM** | Close ALL fetch-side work, including the DG port and the source-monitor port (leave `key='dg'` OFF, no SCAFFOLDS row, DG prereg stays DRAFT marked "not run: wrong stage"). Bank in PREREQUISITES.md: the slept cortex already carries what reinstatement adds. Reachable strength stays unbuilt (its screen is GATED on this ceiling, so it does not start). Next lever: what sleep stores and replays (both speakers' turns): un-gate the voice-design "what sleep stores" item, sequenced after test2 per the stop rule. Caveat to carry: seen seeds, 3 seeds, 1.5B K-P only, strict scorer misses paraphrase; ceiling is not a generalisation claim. |
| **HEADROOM** | A right fetch improves the reply, so fetch quality is worth building. Un-gate reachable strength (nightly replay strengthens the replayed episode's same trace + homeostatic scaling to a set-point; BIO/repetition-retrieval-strengthen-same-trace.md, BIO/homeostatic-scaling-relative-strength.md), its OWN prereg first, arms: replay-strengthen+scaling, scaling only, shuffled-strength sham, fetch-boost lesion; bars: right fetches and oracle-validated replies, false fires <= half of A's; six seeds live (42/43/44/100/101/102). No DG. If a separator is still wanted afterwards: write-side only (orthogonalise a new telling against nearest stored neighbours at `hippocampus.store`), scored on the 25 near-duplicate rows. Voice-design stays sequenced after the reachable-strength read. Seen-seed ceiling: it licenses a build, not a capability claim. |
| **UNDEFINED** | Report as such: name the failing clause(s). No rescue, no relaxed bar, no re-cut of rows. Do not start reachable strength (its gate is a clean HEADROOM) and do not close fetch-side work (that needs NO-HEADROOM). Decide the next step by which clause failed: only H6 (user-voice gap > 5) means the headroom is voice habit, which feeds the voice-design item (what sleep stores / both speakers' turns) rather than fetch work; only H3/H4/H2 means the gain is concentrated in the 4 items behind 128 rows (state the per-seed and cluster numbers); a new run needs a new prereg. Hold both gated items. |
| **INCONCLUSIVE** | Report the reason (missing arm/seed, unit mismatch, mixed commit). No verdict claimed. Repair only by re-running the missing/mismatched unit on a fresh out dir under one commit, then re-score; all gated items stay held. |

HANDOFF.md: write the verdict + the gated items' new state (reachable strength: GO to prereg / held / dead; voice design: sequenced
/ un-gated). A milestone in `brain/ARCHITECTURE.md` section 4 does not move on a ceiling measurement.

## 4. Blind hand-read (40 told rows x up to 5 arms; ~30-40 min, do it BEFORE opening `handread_key.json`)
Sheet: `pulled/handread_sheet.json`: 40 entries `{id: "seed:night:row", question, replies: {A..E: text}}`; arms OFF, A, ORACLE, SHAM,
K-REFIT, labels shuffled per row, same 40 rows for every arm (`HAND_N=40`, seed 20261127, deterministic).
1. Spawn ONE fresh sonnet read-only subagent (hand-reader) with only the sheet and the ledger facts per item (`ledger[item]`'s
   `telling` / `keys` from the lives; the told fact, nothing about arms or the hypothesis). Do not give it the key, the verdict,
   or the prereg.
2. Per reply it answers: (a) states the told fact correctly (Y/N); (b) speaks it in the user's voice, i.e. as the user's own
   statement rather than as the brain's own (Y/N); (c) invents or contradicts (Y/N). Output JSON `{id: {label: {fact, voice, wrong}}}`
   saved as `$D/handread_scores.json` BEFORE the key is opened.
3. Then join with `handread_key.json` (label -> arm, key `seed:night:row:label`), tally per arm: fact Y, voice Y, wrong Y, plus
   ORACLE-OFF and ORACLE-SHAM on fact Y over the 40 rows. Spot-check 5 rows yourself (read the replies, not the counts).
4. Reported, not barred. Use it only to say whether the strict scorer agreed with a human-style read: if hand-read ORACLE-OFF
   disagrees in sign or size (>= 5 of 40) with the strict count scaled to 40/217, say so beside the verdict (the strict scorer
   misses paraphrase: an expected-fail declared in the prereg section 1); it never changes the verdict.

## 5. Skeptic (always, before commit; land-result skill step 2)
One read-only subagent, model sonnet, fresh context. Give it ONLY: the diff (`git diff` of the result + read files), the result
files (`verdict.json`, `handread_scores.json` + tally, `logs/node_s*.txt`), and the runner source
(`brain/experiments/memory_ceiling.py`) plus the prereg and the Fable section 3. Not the conversation, not your interpretation
beyond the claim sentence. Prompt:

```
You are a skeptic. Default verdict: REFUTED when unsure. Read-only; run nothing that writes.
CLAIM: "<paste the one-sentence verdict, e.g. memory reply ceiling = <STATUS>: ORACLE-OFF = x/217, ORACLE-SHAM = y, per-seed ..., engagement ...>"
FILES: brain/experiments/results/2026-11-27-memory-ceiling/verdict.json, handread_scores.json, logs/node_s*.txt,
brain/experiments/memory_ceiling.py, brain/research/2026-11-27-memory-ceiling-prereg.md,
brain/research/2026-11-27-fable-review-dg-port.md (section 3).
Refute the claim. Specifically:
1. Recompute every number in the claim from the raw u_s*_n*_<ARM>.json files (not from verdict.json) and check verdict() applies
   the prereg section 5 rule exactly as written (absolute counts, thresholds 15/6/15/5 and 8/8, clause list H1-H6, NO-HEADROOM checked first).
2. Population: same (seed, night) units and told counts in OFF/A/ORACLE/SHAM; void nights match the dec files; one commit/stage0 hash.
3. Did ORACLE actually force the right episode and did the brain use it? Check forced/engaged/told_no_right_episode per arm and
   that SHAM differs from ORACLE on the forced episode. Did the OFF lesion engage (no recall calls)?
4. Is the claim wider than 3 seen seeds, 1.5B K-P, strict scorer? Is a caveat dropped? Does the next step stated follow the prereg section 7 table for this verdict?
5. Does hand-read agree or conflict with the strict counts, and is that disclosed?
Output: REFUTED / PARTLY REFUTED / SURVIVES, then evidence with file:line or JSON path, then the required caveat text.

<REFUTER-LENS>
REQUIRED LENSES (answer each from recorded data, never from inference):
(a) ENGAGEMENT: did each lesion, permutation or flag actually engage, as recorded in the data (a call count, n_words, a changed field, a trace with route / lesioned / error)? An arm with no record is UNDEFINED, never a PASS or a FAIL.
(b) ANSWER-RATE: for any default-on candidate, does flag-on lower the answer rate or raise abstentions ("I don't know about that.") against flag-off on the packet's own probes, on any seed? Run `tools/answer_rate.py` (prefixed or paired mode) or `iw2_score.py B --off-tag`; a FLIP-HURTS-ANSWERS result means the flip needs a Claude ruling (NEEDS line) before it lands.
(c) WHERE-RAN: which host produced each cited result (run `python tools/where_ran.py <result paths>` and quote its host summary), and does validity depend on it (pool/AWS nodes and the workstation differ in what inputs they hold, for example files under research/findings/raw/ that the provisioner does not ship)?
</REFUTER-LENS>
```
Quote the skeptic's verdict in the commit. Partly refuted: commit with the caveat. Refuted: bank as written, no relaxed bar.
Lens notes for this run: (b) ANSWER-RATE does not apply (nothing flips on; measurement only; say so, run no tool); (a) engagement per arm from `arms.*.engaged` and OFF having no recall (fire/told rows with fetched=0); (c) `python tools/where_ran.py` on the pulled unit files; validity: pool CPU fp32 is the registered host (a 3090 fp16 run, if any, must be named).

## 6. Dry run (11-27, synthetic: no timing-night units are local) and problems found
- `.venv/bin/python -m pytest brain/tests/test_memory_ceiling.py -q`: 28 passed.
- CLI `score` on 6 arms x 3 seeds x 1 night of synthetic unit files using the REAL ledgers of `lifeKR_s319/320/321.json.json` (the
  exact `--lives` arguments above, brace-expanded in zsh): ran, wrote `verdict.json`, `handread_sheet.json`, `handread_key.json` into
  the dir; the H6 failure path produced UNDEFINED as designed. Scratch only, nothing under the repo changed.
Problems / notes:
1. `score` takes ONE `--dir`; the three seeds sit in three node dirs, so pull into one dir (section 1). Not a bug, but easy to miss.
2. The terminal print omits engagement, fire rows, pairs and void list; read them from `--out` (section 2) or the engagement
   check is skipped by accident.
3. The verdict does not gate on engagement; a non-engaging ORACLE with small ORACLE-OFF would print NO-HEADROOM. The read must
   check `engaged` first (section 2) and call that UNDEFINED in the write-up if forced/engaged is near zero (flag it to the skeptic).
4. Provenance gate: three pools on different `COMMIT`s give INCONCLUSIVE; check `logs/node_s*.txt` before scoring.
5. `score` writes the hand-read files into the scored dir (`pulled/`); fine, but do not pull again over a scored dir without
   deleting them (the sheet is deterministic, so a re-score reproduces it).
6. System `python` lacks pytest/torch; use `/home/dant123/Projects/sim/.venv/bin/python`.
