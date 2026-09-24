---
type: finding
status: live
lane: d6-learn-and-grow
date: 2026-09-24
mechanism: a default-OFF observability hook (webapp/d6_hebbian_chat.py) that surfaces the ALREADY-WIRED local-Hebbian D6 fact write (BRAIN_D6_HEBBIAN_STORE, GO 6/6 on the K1-K7 capability gate at runner level) in the /api/brain-chat JSON response, plus a re-run of the unchanged K1-K7 gate at a dev seed
seeds: [7]
verdict: PRE-REGISTRATION only (filed before criterion W or K ran). No result is claimed here.
runner: research/runners/_d6_chat_wire_probe.py
artifacts:
  - research/findings/raw/_d6_chat_wire/selftest.json
---

# PRE-REGISTRATION -- D6 chat-wire observability hook + a seed-7 K1-K7 smoke (2026-09-24)

Committed on its own, BEFORE any evaluation run it governs. The code it governs is the commit immediately before
this one (`c0a81e9c9`) on branch `research/d6-chat-wire`; every constant below is fixed there.
Terms follow `docs/TERMS.md`.

## Why

D6 learn-through-use (the local spiking Hebbian rule replacing the host copy of a taught fact's composite into
its store synapses, `research/runners/d6_hebbian_store.py`) is already GO 6/6 on the pre-registered K1-K7
capability gate at runner level
(`research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md`). The write and read code
paths already exist, unconditionally, in the composer and chat brain that `/api/brain-chat` builds -- gated
default-OFF behind `BRAIN_D6_HEBBIAN_STORE` / `BRAIN_D6_ENGRAM_READTIME`, never claimed as a production default
(`one_brain_composer._store_composite` line ~872, `brain_chat_tui.ChatBrain._maybe_acquire` line ~1140,
`webapp/server.py`'s `readtime_refresh` / `visible_kb` calls). None of that surfaces in the actual
`/api/brain-chat` JSON response: nobody reading the live chat can see that a turn's write went through the local
rule, its encode diagnostic, or whether the block is currently held as an engram. This pre-registration governs
(1) proving the new webapp module that adds that surface is byte-identical when its flag is off, and (2) a
cheap re-confirmation, at a dev seed, that the unchanged K1-K7 gate still reads GO through the same
`/api/brain-chat` handler after this branch's two additive edits to `webapp/server.py`.

## What was built (default OFF, byte-identical off; commit `c0a81e9c9`)

- `webapp/d6_hebbian_chat.py`: `d6_hebbian_enabled()` (reads `BRAIN_D6_HEBBIAN_STORE`) and `after_store_d6(chat)`,
  which reads and CONSUMES `composer._d6_last_encode` (set upstream, unchanged, by `_store_composite`) and takes
  one `d6_hebbian_store.engram_held` read of the written block. Returns `None` when the flag is off, before
  reading or touching any chat/composer state.
- One-line hook (`from webapp import d6_hebbian_chat as _D6C; ... resp["d6_hebbian"] = _D6C.after_store_d6(chat)`)
  added at both response-assembly sites in `webapp/server.py` (the rich-path `resp` dict and the single-fact-path
  `_resp` dict), immediately after the existing `da_tag_capture` hook, wrapped in try/except so a reporting
  failure cannot crash a turn.
- `research/runners/_d6_chat_wire_probe.py`: `--selftest` (no brain; already run, PASS -- see below) and
  `--offcheck` (needs a tiny-demo brain build, two subprocesses, hash-compares replies + store synapses between
  this branch and the merge-base with `origin/main`).

## Pre-registered criteria

**W (webapp-wire byte-identical-off).** `_d6_chat_wire_probe.py --offcheck` on `BRAIN_D6_HEBBIAN_STORE` unset:
`replies_sha256` and `store_sha256` are identical between the pinned merge-base tree and this branch, and no
`d6_hebbian` key appears in any reply (`byte_identical_off: true`). This is the ONLY criterion this hook's own
correctness is judged on; it does not change any decision field the K1-K7 gate scores.

**S (selftest, no brain; already run before this file's commit -- reported, not re-run as a criterion).**
`_d6_chat_wire_probe.py --selftest` exits 0: the off path never touches composer state; the on path with no
fresh encode reports `wrote_this_turn: False`; the on path with a fresh encode reports it once and then
CONSUMES it (a second call reports `False`); a composer with no engram-read capability fails the `held` sub-read
closed (caught, not raised) rather than crashing the hook. Result:
`research/findings/raw/_d6_chat_wire/selftest.json` -- `{"off_is_noop": true, "on_no_write_reports_false": true,
"encode_reported_then_consumed": true, "no_composer_is_noop": true, "pass": true}`.

**K (K1-K7 dev-seed smoke, seed 7).** `research/runners/d6_learn_through_use_lb.py --variant capability --seeds
7` (UNCHANGED gate code; this branch does not edit it) reads GO on K1-K7 exactly as defined in
`research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md`. Seed 7 is a DEV/SMOKE
seed (outside the six validation seeds 42/43/44/100/101/102) and this run does NOT extend, replace or contest
the existing 6-seed GO verdict; it only confirms this branch's two additive `webapp/server.py` edits did not
regress the already-GO mechanism at a cheap seed before any pool time is spent on the 6 gate seeds. A NO-GO or
UNDEFINED result here is a signal to inspect this branch's diff, not a re-opening of the v3 finding.

**AGGREGATE (this branch's own verdict, distinct from D6's production-default verdict).** GO iff W, S and K all
hold. W and K are NOT claimed until their artifacts exist (declared below).

## Compute note (honest, as measured in this environment)

At the time of this pre-registration, `bash tools/mem_ok.sh 12 4` on the local box REFUSED (raw available ~12
GB, ~6 GB already committed to other running memcap scopes -> -6 GB headroom against the standard 12 GB tiny-
demo brain-build cap), and `pool2` was unreachable from this worktree (`ssh pool2`: could not resolve hostname).
Criteria W and K therefore could NOT be run inside this session; `research/findings/raw/_d6_chat_wire/` holds
only what selftest (S) produced. The exact commands to run W and K, once RAM frees locally or pool2 is reachable,
are in `_d6_chat_wire_probe.py`'s own docstring and are reproduced verbatim in this branch's build report.

## Honest scope

- This pre-registration governs an OBSERVABILITY hook only. It makes no claim about D6's production-default
  status (unchanged: `BRAIN_D6_HEBBIAN_STORE` stays default-OFF; see `research/findings/2026-09-23-d6-learn-
  through-use-v3-capability-gate-GO-6of6.md` "Honest scope" for the mechanism's own residuals (a)-(j), all of
  which stand unchanged by this branch).
- No `sim/` edit. No change to `research/runners/{d6_hebbian_store,one_brain_composer,brain_chat_tui}.py`.

## Honesty

Functional read-outs only. "wrote_this_turn" / "held" report a measured synaptic-write / neural-activity event
(established by the lesion-verified K1-K7 gate this branch does not modify), never felt experience.
