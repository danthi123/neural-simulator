# HANDOFF: chat-fact-fixes -- two defects from the Stage-1 exit packet

Written 2026-09-30 by Claude. Branch `research/chat-fact-fixes` (worktree `/home/dant123/Projects/sim/.claude/worktrees/
chat-fact-fixes`), cut from main `f5e3732fe`, pushed to both remotes. NOT merged: merging is Claude's call (NEEDS-CLAUDE
#1). Both fixes are BUG FIXES (no default-off switch; reasons in section 1). The local model does not edit `webapp/`,
`research/runners/brain_chat_tui.py` or the tests; everything below is read-only re-verification.

Set these first in every new shell (the shell may be zsh; run multi-line blocks inside `bash -c` as shown):
`W=/home/dant123/Projects/sim/.claude/worktrees/chat-fact-fixes; B=/home/dant123/Projects/sim/.claude/worktrees/chat-fact-fixes-base; P=/home/dant123/Projects/sim; PY=$P/.venv/bin/python; O=$W/research/findings/raw/_chat_fact_fixes`

## 0. What was wrong, what changed

1. **Requests were stored as facts** (FAILURE_LOG 2026-09-29, Stage-1 exit dev smoke). `ChatBrain._maybe_acquire`
   (`research/runners/brain_chat_tui.py`) taught any input whose function-word-stripped content was three tokens,
   so "tell me about the sky" became the fact [tell, about, sky], echoed as "the tell abouts the sky". Fix:
   `_is_request_speech_act(text)` declines a request shape before BOTH acquisition paths (B3 on, the production
   default, and the legacy B3-off path) and in the side-effect-free `_is_acquisition_candidate` mirror. A request
   shape is: `please` anywhere; a sentence-initial modal (can/could/would/will/should/shall/may/might/must and their
   negations: inversion, so a question or a request); a closed class of listener-directed request verbs first
   (tell, show, give, teach, explain, describe, talk, say, define, name, list, let/let's/lets, help, remind, think,
   imagine, remember, recall, consider, pretend, guess); or any word followed by me/us. Leading discourse particles
   (now, so, ok, okay, well, then, just, and, oh, hey, hi, hello) and a negative-imperative marker (don't, dont,
   never) are skipped first. Only an input the old path would have ACQUIRED and that has a request shape changes.
   HOST SCAFFOLD, declared: the same standing as the interrogative first-word list it extends; the neural
   replacement is a learned speech-act (mood) read on the substrate, not built.
   Known false positives, by construction: a declarative whose subject is a bare proper name spelled like a modal
   or a request verb ("will ate the pie", "may likes cats") is no longer acquired.
2. **A missing corpus degraded the brain silently** (FAILURE_LOG 2026-09-23 x3, 2026-09-25, 2026-09-29). Fix:
   `webapp/corpus_guard.py`. `_build_chat_brain` calls `check_at_build()` (corpus present: nothing; missing: a loud
   stderr warning, or `CorpusMissingError` under `BRAIN_REQUIRE_CORPUS=1`, which the handler turns into a 400 naming
   the file). The `/api/brain-chat` handler's single exit goes through `mark_response()` (present: the SAME
   response object; missing: the reply gains `corpus_present: false` and `corpus_missing`). `CORPUS_FILES =
   ("tinystories.txt",)`: the only file under `data/corpus/` a full 13-turn production session opened, measured by
   an audit hook in both the pre-fix and the fixed arm (`corpus_files_opened` in `base_iw1.json`, `fix_iw1.json`).
   Read the marker as `reply.get("corpus_present", True)`; it is absent when the corpus is present on purpose, so
   no production reply changes by a byte.

Commits (oldest first): `d3b30e17d` identity runner; `dc3e3c118` both fixes + tests; `5f269f4b1` FAILURE_LOG rows;
`fff5d05f7` negative imperatives; `1f0eb1c1c` pre-fix iw1 arm + the sibling-defect row; `f0409f6ec` runner PMEM
script; `30980b16e` guard message wording; then the evidence data commits.

## 1. Evidence (dev seed 7 only; numpy, stub renderer, LLM off, PYTHONHASHSEED=0, LTM `wikidata_100k`)

Failing-first: `tests/test_chat_fact_fixes.py` read **23 failed, 10 passed** at `dc3e3c118`'s parent tree (the 10
passes were the assertion-still-acquired pins), **33 passed** after; the two negative-imperative probes added in
`fff5d05f7` were acquired as NEGATE facts before that commit. Now 35 tests in the file.

Real-handler identity (`research/runners/chat_fact_fixes_identity.py`, one fresh process per arm, the same runner
file against the pre-fix base checkout `f5e3732fe` and the fixed branch; files under `$O/dev_s7/`):

| script | base vs fix | what differs, and why |
|---|---|---|
| `iw1` (the integration-wave1 identity script's 13 turns, verbatim; two sessions) | **IDENTICAL 13/13**, `sha_canon` AND `sha_raw` equal on every turn (`verdict_iw1.json`) | nothing. No turn is a request, and the corpus is present, so neither fix is reachable. |
| `requests` (13 turns, one session) | DIFFERS on 11 turns; the REPLY (answer/abstained/recalled_svo) differs on 5 (`verdict_requests.json`) | Reply changes: the four request turns the old path ACQUIRED (1 "tell me about the sky", 3 "tell me about wolves", 4 "talk about the moon", 5 "can you describe the moon": junk echo such as "the tell abouts the sky", and on turn 3 a fabricated contradiction "I'd stored that tell about sky; I've revised it in place to tell about wolves") now answer with the honest abstain + curiosity question; and turn 9 "tell me more", whose lead-in "On tell, then --" named the junk topic `tell` the acquisitions had put in working memory, loses that lead-in. The six legitimate teach/recall turns (2, 7, 8, 10, 11, 12) keep answer/abstained/recalled_svo; their bytes differ only in diagnostics that read the knowledge-base size or the working-memory slots the junk topics used to occupy (`activity.n_facts_scanned`, `provenance.d/rate_perceived`, `common_ground_drives.*`, `swap_drives.*` slots, `curiosity.novelty`). Turn 0 (before any request) and turn 6 ("please remember the red ball": the comprehension monitor abstains before acquisition in both arms) are byte-identical. |
| `pmem` (the load_bearing_fraction PMEM driving group, verbatim from `onebrain_regression_battery.py` `pmem_form2`..`pmem_cue`) | DIFFERS on 2 turns; the REPLY differs on 1 (`verdict_pmem.json`) | Turn 3 "tell me about the sky" (the group's third intervening turn): junk echo "the tell abouts the sky" -> honest abstain + curiosity. The cue turn 4 "the bird sings" keeps its reply and its whole `prospective` read (`fired: true`, `rel` 0.3628 vs threshold 0.2); only `activity.n_facts_scanned`, two `activity.rf` magnitudes and `curiosity.novelty` differ. Turns 0-2 byte-identical. So the PMEM row's measured field is unchanged on this seed (intact arm only; the lesion arm was not run). |

Degraded brain (fix 2): detached worktrees with no `data/` at the base and at the fix, script `short` (teach, recall,
"tell me about the sky"), `--allow-degraded`:

| arm | what the replies say |
|---|---|
| `nocorpus_base_short.json` (pre-fix) | every turn answers normally; no reply carries any top-level marker. The only trace of the degrade is a nested `comprehension.error: FileNotFoundError ... tinystories.txt` on turn 0 (the log prints "ONEBRAIN XEDGE build FAILED"). |
| `nocorpus_fix_short.json` | every reply carries `corpus_present: false`, `corpus_missing: ["tinystories.txt"]`; stderr prints the `CORPUS MISSING` warning once. Apart from the marker (and the checkout path inside turn 0's `comprehension.error`), turns 0 and 1 are byte-identical to the pre-fix degraded arm; turn 2 is the request fix. |
| `nocorpus_fix_strict_short.json` (`BRAIN_REQUIRE_CORPUS=1`) | every turn is HTTP 400: "failed to load brain 'tiny-demo' ...: CorpusMissingError: CORPUS MISSING: .../data/corpus lacks ['tinystories.txt'] ... BRAIN_REQUIRE_CORPUS=1: refusing to build." No brain is built. |

Classification: fix (1) changes a REPLY (answer/abstained/recalled_svo) only on a request turn the old path acquired,
or on a later turn whose reply quoted the junk it wrote ("On tell, then --"). Later legitimate turns keep their reply;
their bytes differ only in diagnostics that read the smaller knowledge base or the freed working-memory slots. A script
with no request turn is byte-identical (iw1 13/13, `sha_raw` included). Fix (2) is unreachable with the corpus present
(the same 13/13). Both stay on the branch as bug fixes for Claude to merge; neither is behind a switch.

Arm revisions: every base arm ran at `f5e3732fe`; the fix arms ran at `30980b16e` (iw1, both no-corpus arms),
`9fad52768` (requests) and `b8d32483a` (pmem). The code under test is the same at all three (`git diff 30980b16e
b8d32483a -- webapp research/runners/brain_chat_tui.py tests` is empty; the later commits touched only the runner's
`--compare` and data). `git_dirty_tracked: true` in the arm records is the provenance door's append to
`research/findings/raw/_provenance/runs.jsonl` during the run, nothing else.

Existing tests: the 22-file chat/server test set (section 2 lists the fast pair) is RUNNING at packet time; its result is appended in the next commit.

What this evidence does NOT cover (by the TERMS.md rule, byte-identity is claimed only where it was measured): one
dev seed (7), numpy only, stub renderer, and three scripts. The structural argument for other seeds is that the
request predicate is a pure function of the input text and the guard a pure function of the file system, so neither
can change a turn whose input is not a request with the corpus present; that is an argument, not a measurement.

## 2. Re-verify (only if asked, or after main is merged into the branch)

```bash
cd $W && SIM_NO_PROVENANCE=1 $PY -m pytest -q -p no:cacheprovider tests/test_chat_fact_fixes.py tests/test_open_ended_generate_route.py | tail -1   # must print: 42 passed (warnings are normal)
git checkout -- research/findings/raw/_provenance/runs.jsonl 2>/dev/null
```

Real-handler identity (an iw1 arm takes ~16 min and peaks near 12 GB; never start one unless `mem_ok` passes; run the
two arms one after the other, never together). `$B` is a detached worktree at the pre-fix base `f5e3732fe` with
`data -> $P/data`; recreate it if missing: `cd $P && git worktree add --detach $B f5e3732fe && ln -s $P/data $B/data`.
`$W/data` must be the same link (`ls $W/data/corpus/tinystories.txt`; if missing: `ln -s $P/data $W/data`).

```bash
cd $P && bash -c 'bash tools/mem_ok.sh 11 && bash tools/memcap.sh 15 -- env PYTHONHASHSEED=0 SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '"$PY"' -u '"$W"'/research/runners/chat_fact_fixes_identity.py --root '"$W"' --script iw1 --seed 7 --out '"$O"'/recheck/fix_iw1.json'
cd $P && bash -c 'bash tools/mem_ok.sh 11 && bash tools/memcap.sh 15 -- env PYTHONHASHSEED=0 SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '"$PY"' -u '"$W"'/research/runners/chat_fact_fixes_identity.py --root '"$B"' --script iw1 --seed 7 --out '"$O"'/recheck/base_iw1.json'
$PY $W/research/runners/chat_fact_fixes_identity.py --compare $O/recheck/base_iw1.json $O/recheck/fix_iw1.json --out $O/recheck/verdict_iw1.json | head -12
```

If `mem_ok` refuses, wait and retry later; do not lower its margin.

| `--compare` prints | write (in the WORKLOG / runbook) |
|---|---|
| `"verdict": "IDENTICAL"` | "chat-fact-fixes iw1 identity re-check: IDENTICAL 13/13 (seed 7) at <fix git_sha>." |
| `"verdict": "DIFFERS"` | STOP, NEEDS-CLAUDE #2 (paste the `turn N ... paths=` lines). |
| `"verdict": "UNDEFINED"` | read `preconditions`: a `false` corpus/LTM/status/hash-seed field means a degraded or failed arm; re-run that arm ONCE; still UNDEFINED: NEEDS-CLAUDE #2. |

## 3. Stop conditions

- A test in step 2 fails, or an identity re-check prints DIFFERS, or stays UNDEFINED after one re-run.
- Anything that would need an edit to `webapp/`, `research/runners/brain_chat_tui.py` or the tests.

## 4. NEEDS-CLAUDE

1. **Merge ruling.** Both fixes are bug fixes on the branch; merge `research/chat-fact-fixes` to main (it contains
   main at `f5e3732fe`; re-run section 2's tests after merging main in if main moved).
2. Any re-verification failure from section 2.
3. **Sibling acquisition defect (open, not fixed here; FAILURE_LOG 2026-09-30 row 3).** The same scaffold stores
   non-SVO declaratives: "the dog and the cat walked in" -> [dog, cat, walked] ("the dog cats the walked"), "that
   was wonderful, I am so happy" -> [wonderful, be, happy] ("The wonderful bes happy."). These are the iw1 identity
   script's own turns and the affect/world-model probes' sentences, so a fix changes those replies: a ruling is needed
   (default-off packet vs bug fix).
4. **Runs whose replies change at a merged revision.** Any script with a request turn the old path acquired now
   replies differently on that turn and possibly on a later turn that quoted the junk: the load_bearing_fraction PMEM
   driving group's third intervening turn (`onebrain_regression_battery.py` `pmem_d2` = "tell me about the sky";
   measured in section 1), `_flip_validated_fixes_chat_sha.py` ("tell me about the bird"), the soak scripts that say
   "tell me about the dog", and, with `BRAIN_OPEN_ENDED_ACQUIRE_ROUTE=1`, the affect-tone open-output prompts ("Tell
   me about the mountains", "Describe the city at night", ...), which that route used to send to acquisition. A
   cross-revision byte-identity check spanning the merge must expect those turns to differ.
5. **`BRAIN_REQUIRE_CORPUS=1` for gate jobs.** The guard only marks and warns by default (CI and worktrees without
   the corpus must still build). Decide whether pool/AWS gate job lines should set `BRAIN_REQUIRE_CORPUS=1` so a
   degraded node refuses instead of producing marked shards.
6. The B3 non-contradiction gate (`webapp/server.py`, before the acquisition) still treats a request as an in-scope
   assertion for its spiking polarity recall (e.g. it recalls (tell, about, sky) -> unknown -> accept). Harmless
   now that nothing stores such a triple, and left alone because changing it changes the `noncontradiction`
   diagnostic field and the substrate calls on request turns; say if it should be aligned.
7. Pre-existing, unrelated: `tests/test_open_ended_generation_fluent.py::test_render_hypothesis_fluent_flagged_guess_stub`
   and `::test_render_hypothesis_template_fallback_without_mouth` fail on the pre-fix base `f5e3732fe` too.
