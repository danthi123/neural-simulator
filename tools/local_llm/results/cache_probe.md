# Local-LLM prompt-cache probe results

One llama-server per config (qwen38-27b-iq4nl-mtp-128k-q4, same model/ctx/template for every row), the SAME short multi-turn Claude Code task through each, server log parsed for tokens actually reprocessed per request vs the prompt size submitted. See tools/local_llm/cache_probe.py for the exact method and tools/local_llm/results/cache_probe.json for the full per-request data.

| config | np | extra flags | load s | peak VRAM MiB | headroom MiB | task wall s | # requests | prompt sizes (tokens) | overall reused frac | last-req reused frac | last sim_best / f_keep | total pp time s | checkpoints created |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_np1_baseline | 1 | (none) | 16.0 | 22815 | 1761 | 99.2 | 2 | 21053,40494 | 0.0 | 0.0 | 0.386 / 0.734 | 60.0 | 0 |
| B_np2_kvu | 2 | -kvu | 8.0 | 23678 | 898 | 163.6 | 3 | 21054,40030,40640 | 0.0 | 0.0 | 0.385 / 0.39 | 110.1 | 0 |
| C_np2_kvu_denser_checkpoints | 2 | -kvu -ctxcp 32 -cms 1024 | 8.0 | 23460 | 1116 | 111.9 | 2 | 21063,40618 | 0.0 | 0.0 | 0.384 / 0.73 | 66.8 | 0 |
| D_np2_kvu_checkpoints_cram16g | 2 | -kvu -ctxcp 32 -cms 1024 -cram 16384 | 6.0 | 23439 | 1137 | 165.2 | 3 | 21069,40066,40673 | 0.0 | 0.0 | 0.384 / 0.39 | 113.1 | 0 |
| E_np1_cache_reuse | 1 | --cache-reuse 256 | 6.0 | 22778 | 1798 | 115.9 | 2 | 21059,40568 | 0.0 | 0.0 | 0.385 / 0.731 | 65.1 | 0 |
| A_np1_baseline_ROUND1TEMPLATE | 1 | (none) | 4.0 | 22821 | 1755 | 155.9 | 3 | 21163,40996,41708 | 0.0 | 0.0 | 0.375 / 0.38 | 107.9 | 0 |
| A_np1_baseline_ROUND2TEMPLATE | 1 | (none) | 8.0 | 22813 | 1763 | 119.9 | 3 | 21167,41021,42880 | 0.606 | 0.986 | 0.986 / 1.0 | 46.2 | 0 |

## ROUND 1 (server-flag sweep): 0% reuse on all five configs -- but the wrong layer was blamed

Every one of the five configs above (`-np 1` baseline, `-np 2` with `-kvu`, `-kvu` plus denser context
checkpoints, + doubled `-cram`, and `-np 1` with `--cache-reuse` explicitly enabled) produced **0% prompt-cache
reuse across turns** and **created zero context checkpoints**, for the exact same real, growing, single-thread
Claude Code conversation. This is not "no common prefix to reuse": in every config, the request-2 (or -3)
`selected slot by LCP similarity` line reports `f_keep` of 0.39-0.73 -- llama-server's own slot-selection
heuristic correctly DETECTS that a large fraction of the previously-cached prompt is still a valid prefix of the
new, longer one. Despite that, the subsequent `prompt eval time = ... / N tokens` line for the same request
shows N within a few tokens of the FULL new prompt length every single time.

**ROUND 1's conclusion -- "this is an upstream llama-server hybrid-model limitation, nothing to fix" -- was
WRONG, and is corrected by round 2 below on a coordinator challenge that it rested on an untested, inferred
cause rather than a real captured request.** Round 1 never captured or rendered a real request; it inferred the
mechanism from server-side log symptoms alone. It was right that llama-server's checkpoint/`-ctxcp`/`-cms`/
`-cram`/`--cache-reuse` machinery never engaged (still true, see round 2) -- but wrong about why the prefix was
unusable in the first place. The actual cause was in OUR OWN chat template, entirely fixable, and fixing it took
prompt-cache reuse on this exact model from 0% to 60.6% overall (98.6% on the largest turn) with NO server-flag
changes at all.

## ROUND 2 (coordinator directive, real captured requests): the chat template was hoisting mid-conversation system messages

**Method.** Captured the raw Anthropic `/v1/messages` request bodies of a real 3-turn `claude -p` session via a
logging reverse proxy (`tools/local_llm/capture_requests.py` + `tools/local_llm/capture_session.py`), then
rendered pairs of them through the live chat template via llama-server's own `/apply-template` endpoint
(`tools/local_llm/render_and_diff.py`) to find the exact first character where two consecutive requests'
rendered prompts diverge. Full captures and diffs: `tools/local_llm/results/template_divergence/`.

**Finding.** Claude Code sends its per-turn "system reminders" (a live `<total_tokens>N tokens left</total_tokens>`
line, refreshed every turn; a stable session-start reminder) as literal `role: "system"` entries embedded
directly in the `messages` array, not just in the leading `system` field -- confirmed directly from the captured
JSON (`capture_old_template/0003_POST_v1_messages?beta=true.json`, message indices 1 and 4). The
round-1 chat-template fix (`tools/local_llm/templates/qwen38-27b-iq4nl-mtp.jinja`, "LOCAL FIX 2026-09-25") merged
EVERY such message, wherever it appeared, into the ONE leading system block to stop a real
"System message must be at the beginning" failure. That fix worked for correctness but broke caching: each new
turn's fresh reminder text changes the CONTENT of that leading block, so the leading-block/conversation-turns
BOUNDARY shifts by a few bytes every turn, and everything after it -- the entire rest of the conversation, even
though byte-identical -- counts as changed. Measured directly: rendering turn 1 and turn 2 of the SAME real
session through the round-1 template diverges at char 62857 of turn 1's 82848-char prompt (75.9% in), right at
that exact boundary (`diff_old_template_turn1_vs_turn2.json`).

**Fix.** `tools/local_llm/templates/qwen38-27b-iq4nl-mtp.jinja` ("LOCAL FIX 2026-09-25 ROUND 2"): only the
LEADING contiguous run of system/developer messages is merged into the one stable leading block now (as the
pre-round-1 template did for a single message); every LATER system/developer message renders IN PLACE, as its
own `<|im_start|>system ... <|im_end|>` turn at its own position, and the main loop skips only the leading run
by index rather than every system-role message by role. Never raises (the original bug stays fixed). Verified
offline (no GPU) in `tools/local_llm/templates/test_templates_offline.py::check_prefix_stability_qwen`, which
would fail against the round-1 template (confirmed: divergence at char 366/849 in the synthetic fixture) and
passes against the fix.

**Re-measured on the SAME real captured requests, through the FIXED template:** turn 1's entire 82877-char
rendered prompt is now a byte-for-byte PREFIX of turn 2's 144447-char prompt -- divergence at char 82877, i.e.
100.0% of turn 1 (`diff_new_template_turn1_vs_turn2.json`). **Re-measured end-to-end with a real llama-server
+ a real 3-turn Claude Code session** (`A_np1_baseline_ROUND1TEMPLATE` vs `A_np1_baseline_ROUND2TEMPLATE` in
the table above, same `-np 1` config, template swapped): overall reused fraction 0.0 -> **0.606**, last-turn
reused fraction 0.0 -> **0.986**, total prompt-processing time 107.9s -> **46.2s** for a slightly LARGER
conversation. Context checkpoints created: 0 in both -- expected and fine, not a regression: with a byte-exact
prefix the server needs a plain forward CONTINUATION (pick up decoding where the previous turn's cache already
ends), which every architecture supports natively; a checkpoint-based REWIND (round 1's target) is only needed
when the divergence point is somewhere back in the MIDDLE of the cache, which no longer happens here.

**Hypothesis 2 (assistant turns re-rendered differently from what was generated) was checked directly and
ruled out as a contributing factor**, not merely assumed away: the model's actual streamed "thinking" content
(from the captured SSE response) is byte-identical to what Claude Code resends as history in the next request
(verified on `capture_old_template/0001_...json.response` vs `0003_...json` message index 2), and the template
renders historical assistant turns through the exact same formatting code path used for live generation, so
there is no additional divergence source here to fix.

**Hypothesis (a)** (interleaved small-fast-model side calls) remains unreproduced under
`--dangerously-skip-permissions` (needed for unattended runs) -- every capture shows exactly as many
`/v1/messages` requests as conversation turns, no interleaved extra calls. Not needed to explain the bug either
way: the fix above fully explains and resolves the measured symptom without it.

## Chosen config: `A_np1_baseline` + the FIXED chat template (no llama-server flag changes)

The server-flag sweep (round 1: `-np`/`-kvu`/`-ctxcp`/`-cms`/`-cram`/`--cache-reuse`) is still valid as a
NEGATIVE result on its own terms -- none of those flags move the needle, and `B`/`D` (the `-kvu` multi-slot
variants) cost more peak VRAM for it. The actual fix was the chat template, requires no `-np`/`-kvu` change, and
is already the profile's own template file, so `tools/local_llm/llm.sh`'s `profile_cmd()` and
`tools/local_llm/bakeoff.py`'s `start_server()` keep `-np 1` with no extra flags -- comments at each now point
to this corrected history instead of the retracted round-1 conclusion.

**Agentic-task regression check** (`python3 tools/local_llm/bakeoff.py --profiles qwen38-27b-iq4nl-mtp-128k-q4`,
run against the FIXED template as its own GPU-queue job): all three tasks still PASS -- T1 locate 44s, T2 debug
64s, T3 extend 42s -- noticeably faster than this same profile's original (pre-branch) bake-off with the
round-1 template, 90s/213s/139s; long-context passphrase recall at 120K tokens still `True`; peak VRAM 22831
MiB, in line with every other measurement in this file. The fix does not just avoid regressing these tasks, it
makes the multi-turn ones (T2/T3, which are themselves several turns within one Claude Code session) noticeably
faster, for the same reason the cache-probe numbers above improved.
