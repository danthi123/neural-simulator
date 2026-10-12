---
type: finding
status: provisional
claim_check: measured (exact per-turn rss_kb read from the committed AWS-pair OFF sidecar; code sites verified by a MiMo Pro delegate reading webapp/server.py and the battery runner directly; awaiting Claude review)
date: 2026-10-12
lane: c5 / g6 battery efficiency (root cause of the 20 GB-slice OOMs that killed the cap-18/cap-30 arms; RULING c5-next-1012 task (b))
mechanism: onebrain_regression_battery --worker runs every turn in one process; brain_chat caches a full ChatBrain per (session, brain, renderer) key in webapp's unbounded _BRAIN_CHATS warm cache, evicted only by req.reset, so each new turn session pins ~5.3 GB for the process lifetime
seeds: [42]
artifacts:
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6_on_c5pair_rss.jsonl
  - research/runners/onebrain_regression_battery.py
  - webapp/server.py
verdict: The ~5.3 GB per-turn-kind retention is the per-(session, brain, renderer) warm cache in webapp/server.py (unbounded _BRAIN_CHATS plus the ~20 per-session registries pinned by the same key); same-key turns add ~0, each new session key adds 5,340,964 to 5,455,512 KiB. Fix is an unimplemented follow-up (bounded cache via the full reset teardown); no post-fix run exists. <!--derived-->
---

# Battery OOM root cause: the webapp warm cache retains ~5.3 GB per new session key

**Status: PROVISIONAL (agent-filed per RULING `c5-next-1012` (b); awaits Claude review).**
RULING `c5-next-1012` asked: find what `onebrain_regression_battery --worker` keeps per turn kind
(~5.3 GB each) — it is a separate efficiency fix and is also why big batteries OOM. This is that
finding, plus a fix design (unimplemented).

## The verdict (one MiMo Pro delegate, RULING `models`, code sites verified by direct read)

The per-turn RSS proves the ~5.3–5.5 GB retention is **per NEW `(session, brain, renderer)` cache
key held in `webapp/server.py`'s unbounded `_BRAIN_CHATS` warm cache**: same-key turns show ~0
growth (bc_a→bc_b −784 KiB), each distinct session adds +5,340,964 to +5,455,512 KiB
(emo→bc_a, bc_b→rich_well, rich_well→rich_open), matching a cache built at
webapp/server.py:4663-4675 and evicted only by `req.reset` at webapp/server.py:4625-4661.

## The arithmetic (exact rss_kb from the committed AWS-pair OFF sidecar)

<!--derived--> research/findings/raw/_regression_battery/c5_offcurve_diag/g6_off_c5pair_rss.jsonl

| turn | rss_kb | Δ vs previous turn (KiB) |
|---|---|---|
| emo | 6234688 | — (process base + first key) |
| bc_a | 11690200 | +5455512 |
| bc_b | 11689416 | −784 (same session key: no growth) |
| rich_well | 17075536 | +5386120 |
| rich_open | 22416700 | +5340964 |

Base ~6.2 GB + ~5.3 GB per distinct session key reproduces the whole curve (6.2 → 11.7 → 11.7 →
17.1 → 22.4 GB) <!--derived-->.

## The code chain

1. **The worker accumulates nothing big itself** — `research/runners/onebrain_regression_battery.py`
   line 1026: `for label in turn_labels:` calls `brain_chat(...)` in ONE process; its own store is
   `responses = {}` (line 1025), reply texts only (the out-file is ~31 KB).
2. **The cache key is per session** — `webapp/server.py:4619`:
   `cache_key = (req.session, req.brain, renderer)`.
3. **First call per key builds and pins a full ChatBrain** — `webapp/server.py:4663-4675`:
   `chat = _BRAIN_CHATS.get(cache_key)`; if missing, `_build_chat_brain(...)` then
   `_BRAIN_CHATS[cache_key] = chat`. The docstring (4578-4580) states the warm cache is by design;
   there is no LRU or size bound.
4. **The only eviction is `req.reset`** — `webapp/server.py:4625-4661` pops `_BRAIN_CHATS`,
   `_BRAIN_RICH`, and ~20 per-session registries (`_SESSION_MOOD`, worldview, multiref, silent-WM,
   selfinit, discourse, pmem, episodic, causal, false-belief, the idle-replay organ).

bc_a and bc_b share one session key, which is why they add zero; every new session builds a second
full brain and keeps it for the worker's lifetime.

## Recommended fix (delegate; NOT implemented — follow-up task)

Extract the `req.reset` teardown block (webapp/server.py:4625-4661) into a reusable
`_evict_session(cache_key)` helper, then bound `_BRAIN_CHATS` with a recency-ordered cap
(env `BRAIN_CHAT_CACHE_MAX`, default 2) that evicts the least-recently-used key through the FULL
teardown (including `_BRAIN_RICH` and the ~20 registries) on new-key insertion. Battery semantics
are preserved because one session's turns run consecutively, so the active key is never the LRU
victim; at cap 2 the battery holds at most ~2 keys (~11 GB) regardless of session count.

**Rejected alternatives:** (a) per-turn LRU eviction inside the request handler — destroys the
experiment fixture's persistent state mid-session, which the battery requires until that session's
final turn; (b) battery calls `brain_chat_reset` between sessions — lives in reserved
`research/runners/*` (sanctioned-branch-only), leaves the unbounded cache unfixed for every other
caller, and papers over the retention site instead of removing it.

## Scope (delegate)

- Shows: the retention site (`_BRAIN_CHATS` plus the per-session registries pinned by `cache_key`),
  the mechanism (warm cache by design, eviction only under `req.reset`), and the arithmetic match
  (~5.3–5.5 GB per new key, ~0 within one key, consistent with big batteries OOMing on the 20 GB
  slice — RULING `wspool-kills-1012`).
- Does NOT show: which component inside the built chat/organ state accounts for the ~5.3 GB — only
  that it is per-session and cache-pinned; intra-brain attribution was not measured.
- Does NOT show: that any fix works. The bounded-cache change is unimplemented; no post-fix RSS run
  exists, and nothing here claims production webapp behavior under many concurrent sessions.

## Provenance

The RSS curve is the AWS pair's OFF arm (r7i.4xlarge, rev `18ec1f86d`, seed 42, RULING
`c5-off-baseline`; artifacts staged at commit `a62e36185`). The worker's structure and the cache
sites were read directly from this checkout; a MiMo Pro delegate re-verified both files before the
verdict above. `g6_off.json` + `g6_off_c26_rss.jsonl` from the workstation cap-26 run (5/5 turns,
rc=0) are landed beside the sidecars as corroboration; its 308 MB `g6_off.log` was deliberately not
committed (size).