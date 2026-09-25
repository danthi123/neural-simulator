# Local-LLM prompt-cache probe results

One llama-server per config (qwen38-27b-iq4nl-mtp-128k-q4, same model/ctx/template for every row), the SAME short multi-turn Claude Code task through each, server log parsed for tokens actually reprocessed per request vs the prompt size submitted. See tools/local_llm/cache_probe.py for the exact method and tools/local_llm/results/cache_probe.json for the full per-request data.

| config | np | extra flags | load s | peak VRAM MiB | headroom MiB | task wall s | # requests | prompt sizes (tokens) | overall reused frac | last-req reused frac | last sim_best / f_keep | total pp time s | checkpoints created |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_np1_baseline | 1 | (none) | 16.0 | 22815 | 1761 | 99.2 | 2 | 21053,40494 | 0.0 | 0.0 | 0.386 / 0.734 | 60.0 | 0 |
| B_np2_kvu | 2 | -kvu | 8.0 | 23678 | 898 | 163.6 | 3 | 21054,40030,40640 | 0.0 | 0.0 | 0.385 / 0.39 | 110.1 | 0 |
| C_np2_kvu_denser_checkpoints | 2 | -kvu -ctxcp 32 -cms 1024 | 8.0 | 23460 | 1116 | 111.9 | 2 | 21063,40618 | 0.0 | 0.0 | 0.384 / 0.73 | 66.8 | 0 |
| D_np2_kvu_checkpoints_cram16g | 2 | -kvu -ctxcp 32 -cms 1024 -cram 16384 | 6.0 | 23439 | 1137 | 165.2 | 3 | 21069,40066,40673 | 0.0 | 0.0 | 0.384 / 0.39 | 113.1 | 0 |
| E_np1_cache_reuse | 1 | --cache-reuse 256 | 6.0 | 22778 | 1798 | 115.9 | 2 | 21059,40568 | 0.0 | 0.0 | 0.385 / 0.731 | 65.1 | 0 |

## What the evidence shows

Every one of the five configs above (`-np 1` baseline, `-np 2` with `-kvu`, `-kvu` plus denser context
checkpoints, + doubled `-cram`, and `-np 1` with `--cache-reuse` explicitly enabled) produced **0% prompt-cache
reuse across turns** and **created zero context checkpoints**, for the exact same real, growing, single-thread
Claude Code conversation. This is not "no common prefix to reuse": in every config, the request-2 (or -3)
`selected slot by LCP similarity` line reports `f_keep` of 0.39-0.73 -- llama-server's own slot-selection
heuristic correctly DETECTS that a large fraction of the previously-cached prompt is still a valid prefix of the
new, longer one (exactly as expected, since Claude Code resends the whole growing history each turn). Despite
that, the subsequent `prompt eval time = ... / N tokens` line for the same request shows N within a few tokens
of the FULL new prompt length every single time -- i.e. detection works, but the engine never actually applies
it: no `after context reuse` line and no `created context checkpoint` line EVER appeared in any of the five logs.

**Root cause: this is a hybrid-architecture limitation of llama-server (b10042/50b29f6), not a missing flag.**
Qwen3.8-27B here is 48 of 64 layers Gated DeltaNet (linear-attention, a FIXED-SIZE recurrent state per layer,
not a token-indexed KV cache) plus 16 full-attention layers. A plain transformer's KV cache can be trimmed back
to an arbitrary earlier token position for free (`memory_seq_rm`); a recurrent layer's state cannot -- it can
only be rewound to a previously-saved CONTEXT CHECKPOINT (`-ctxcp`/`-cms`, the exact mechanism this branch set
out to tune). But context checkpoints are apparently only created around an actual context-shift/eviction event,
never merely because a new request could reuse a slot's prior content -- and a normal multi-turn conversation
here never gets anywhere near the 131072-token limit that would trigger one. With no checkpoint to roll back to,
llama-server has no choice but to reprocess from position 0 every time, regardless of `-np`, `-kvu`, `-ctxcp`,
`-cms`, `-cram`, or `--cache-reuse` -- all five knobs this branch was asked to test.

Hypothesis (a) from the original evidence (Claude Code's small-fast-model side calls sharing the one slot and
evicting the main conversation) was NOT reproducible in this harness: `--dangerously-skip-permissions` (required
to run Claude Code unattended) evidently suppresses whatever side-channel calls an interactive session makes --
every config here shows EXACTLY as many llama-server requests as conversation turns, no interleaved extra
requests at all. That said, hypothesis (a) is not NEEDED to explain the original production symptom: the falling
`sim_best` trend (0.62 -> 0.23) reported live is fully explained by an ever-growing denominator (a fixed-size
old prefix over a growing new prompt naturally yields a falling ratio) plus the same zero-reuse bug reproduced
here with a single, non-interleaved conversation. Hypothesis (c) (prompt content changing early each turn) is
also ruled out: Claude Code's conversation format only ever APPENDS, and the detected `f_keep` overlap proves a
large genuine shared prefix exists -- it is the failure to exploit it that is the bug, not its absence.

## Chosen config: keep `A_np1_baseline` (no change to the production default)

None of the four alternatives improved prompt-cache reuse (all four were also on this list: 0.0 overall reused
fraction, 0 checkpoints), so there is no config-level fix available among the flags llama-server exposes on this
build. `B`/`D` (the `-kvu` multi-slot variants) also cost noticeably more peak VRAM (~23.4-23.7 GB vs `A`'s
~22.8 GB, still under the ~23.5 GiB budget but with less headroom) for zero benefit, and in this run took an
extra conversational round each (a stochastic sampling difference, not a per-token slowdown -- per-token
prompt-processing throughput was similar, ~900-1050 tok/s, across all five). `tools/local_llm/llm.sh`'s
`profile_cmd()` and `tools/local_llm/bakeoff.py`'s `start_server()` are left unchanged (`-np 1`, no new flags),
with a comment added at each recording this investigation so it is not re-derived from scratch later.

**Smallest actual fix given this constraint:** there isn't one at the llama-server flag level on this build.
The two levers that could plausibly help are both outside this branch's scope: (1) upgrading `llama-server` past
b10042 in case a newer release creates checkpoints proactively for hybrid models rather than only on a context
shift, or (2) accepting the full-reprocess cost as inherent to this specific hybrid+MTP model choice and
weighing it against a less linear-attention-heavy model if turn latency on long sessions matters more than this
model's throughput/VRAM profile. Both are follow-up investigations, not flag changes.
