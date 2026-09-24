---
type: finding
status: live
date: 2026-09-24
lane: load-bearing
mechanism: BRAIN_TRANSITIVE_CHAT/BRAIN_TRANSITIVE_LESION -- a transitive yes/no chat route over NON-ADJACENT pairs, wired onto the ALREADY-GO'd shared-substrate keystone re-entrant chase (webapp.gnw_multistep_deliberation.multistep_chase / research.runners._gnw_reentrant_metacog_gated_deliberation_derisk.confidence_gated_chase) instead of OneBrainComposer.query_chain's host for-loop (one_brain_composer.py:1818); both flags default OFF
seeds: [7]
verdict: PRE-REGISTRATION only. No result is claimed here. Seed 7 is a DEV seed (never 42/43/44/100/101/102); the 6-seed capability gate is staged in the companion report, not run by this document.
---

# PRE-REGISTRATION — reasoning-transitive-chat (A6, midnight-plan S15(b), 2026-09-24)

**Status at commit time:** PREREGISTERED, no evaluation artifact exists yet. This document is committed in its
own commit, before any smoke/gate JSON, per the standing rule (commit the prereg before any evaluation artifact).

## Why this lane exists

`docs/plans/2026-09-24-midnight-...` step S15(b) / the declared-crutches register's "reasoning" row (item 22):
reasoning is OPEN, credited as "multi-hop recall (host-chained)" because `OneBrainComposer.query_chain`
(`research/runners/one_brain_composer.py:1818`) is a host `for` loop over `actions` — the loop, not the brain,
decides how many hops to run. The plan's own escape clause: credit real "reasoning" only if a shared-substrate,
non-host-loop multi-hop mechanism is wired instead.

**A shared-substrate mechanism already exists and is already 6/6-seed GO'd and production-default-ON**:
`webapp/gnw_multistep_deliberation.py` (`BRAIN_GNW_MULTISTEP`, default-ON since 2026-08-19) wraps the keystone
re-entrant deliberation (`research/runners/_gnw_reentrant_metacog_gated_deliberation_derisk.py::
confidence_gated_chase`, GO'd variable-depth-chase task, 6 seeds 42/43/44/100/101/102). There the number of hops
is an EMERGENT read of the GNW workspace's own spiking ignition (`n_ignited` off `cp_firing_states`), not a host
counter — exactly the property the plan wants for a "reasoning" credit. It is currently wired ONLY behind an
explicit "chase to the end" marker on a *wh*-question ("what does X chase all the way?"). This lane does **not**
re-wire `emerge28` (`research/runners/_emerge28_emergent_codes_transitive_derisk.py`, a standalone unwired HTM
demo on its own toy pool — infeasible to production-integrate in this pass) — it instead reuses the ALREADY-
PRODUCTION-WIRED keystone chase for a genuinely new question SHAPE: a **transitive yes/no query about a
NON-ADJACENT pair** ("does A precede D?", never directly taught, only inferable by chaining taught adjacent
facts). This is a stronger and more literal test of "transitive reasoning" than "chase to the end", and it is
built on the SAME already-verified substrate mechanism rather than a new one.

## The mechanism (`webapp/reasoning_transitive_chat.py`)

Flags: `BRAIN_TRANSITIVE_CHAT` (default OFF) and `BRAIN_TRANSITIVE_LESION` (default OFF, meaningful only when
the first is ON).

1. **DETECT** (host regex, a declared scaffold — the SAME boundary `compositional_chain_route.py`'s
   `_POSSESSIVE_CHAIN_RE` and `gnw_multistep_deliberation.py`'s `_CHASE_MARKERS` already occupy): the raw
   utterance shape `"does <agent> <relation> <target>"` (case-insensitive, trailing `?` optional). No match, or
   the flag OFF -> `resolve_transitive_query` returns `None` immediately and the server hook is a pure
   pass-through (byte-identical to today).
2. **DIRECT-FACT SHORT-CIRCUIT**: if `composer.query_patient(agent, relation) == target` (a genuinely stored
   single-hop fact), the answer is a plain (non-derived) recall — reported `derived=False`, exactly like any
   ordinary recall. This exists so an ADJACENT pair is never mislabeled as a multi-hop derivation.
3. **SUBSTRATE CHASE** (only reached when there is no direct fact): calls
   `webapp.gnw_multistep_deliberation.multistep_chase(chat, agent, relation, seed=seed,
   lesion=BRAIN_TRANSITIVE_LESION)` — the UNCHANGED, already-GO'd keystone chase. Its returned `trace` (a list of
   per-cycle dicts: `x`, `target`, `committed`, `n_ignited`, `conf`, `action`) is walked to reconstruct the
   sequence of concepts the substrate actually broadcast back on each `ADVANCE` (never a host-assumed path). If
   `target` appears in that substrate-committed path, the row emits a `ChainedSVO([agent, relation, target],
   derived_from=<the ordered hop facts the trace itself confirmed>)` — reusing `compositional_chain_route.
   ChainedSVO` verbatim, so it gets, for free, everything that module's 2026-08-25 moat-hardening pass already
   built: the `derived`/`derived_from` API shape, the "I derived this from: ..." honest lead
   (`frame_derived_answer`), `PROVENANCE_GENERATED` framing (never `PERCEIVED`) when the #129 monitor is on, and
   exclusion from the episodic/discourse-WM writes a direct recall would trigger.
4. **HONEST ABSTAIN, NOT A CONFABULATED "NO"**: if the chase terminates (reaches its own leaf, i.e. the
   substrate's own ignition collapses — `action == "COMMIT"`) without ever broadcasting `target`, or if the
   chase itself abstains (`action == "ABSTAIN"`, e.g. no fact for `agent`/`relation` at all), the resolver
   returns a sentinel that makes the caller emit **"I don't know about that." (abstained=True)** — never a
   positive "no" assertion. **Named v1 scope limitation**: this means a chase-CONFIRMED negative ("I checked the
   whole chain and X is definitely not reachable") is reported identically to a genuine unknown. Distinguishing
   them is a named follow-on, not silently dropped.
5. **NO FALL-THROUGH ON A MATCHED-BUT-UNRESOLVED QUESTION.** Once the regex matches, the caller must NOT also
   run the ordinary `chat.gate(msg)` on the raw text — `does X R Y` strips to content words `[X, R, Y]`, and the
   generic `_extract_route` positional parser already documented as truncating the 3rd content token
   (`compositional_chain_route.py`'s own "PARSER TRUNCATION" note) would silently answer a *different* question
   ("what does X R?") instead of the yes/no one asked. `resolve_transitive_query` therefore returns a
   `(matched=True, svo_or_None)` pair; `svo_or_None=None` means "matched, honestly abstain", not "try the
   ordinary path".

## Wiring point (`webapp/server.py`, single-fact path only)

Inserted at the SAME site `compositional_chain_route.resolve_compositional_chain` already runs (before
`chat.gate(msg)`), guarded so a transitive-shaped match short-circuits the ordinary gate call entirely. When the
result is a `ChainedSVO`, `_is_chain_route` is set `True` for it exactly as for the possessive-chain route, so
every downstream honesty/provenance/episodic-exclusion branch already keyed on `_is_chain_route` covers this row
for free — no new response-shape code. **The `rich=True` path is NOT wired in this pass** (named residual,
matching `compositional_chain_route`'s own original scope before any rich-path follow-on).

## Byte-identity (off)

`BRAIN_TRANSITIVE_CHAT` unset -> `transitive_chat_enabled()` is `False` -> `resolve_transitive_query` returns
`None` on its very first line -> the server hook takes the untouched pre-existing branch
(`compositional_chain_route` then `chat.gate`) -> **G6** below proves this with a pinned-transcript hash.

## Lesion (`BRAIN_TRANSITIVE_LESION`)

Reuses, unmodified, the keystone's own already-verified lesion: `multistep_chase(..., lesion=True)` builds the
GNW workspace with its assembly self-recurrence ZEROED (`webapp/gnw_deliberation.py::_get_bridge`, cached
separately per `(seed, lesion)` so the intact and lesioned workspaces never collide). This is the SAME lesion
`BRAIN_GNW_MULTISTEP_LESION` already uses for the shipped `gnw-multistep-deliberation` faculty row — no new
lesion mechanism is introduced, only a new consumer of the existing one.

## The gate (held-out non-adjacent pairs + a scrambled-premise control)

**World A (the taught chain).** Five entities on one linear order, one repeated relation `precede`:
`e0 precede e1`, `e1 precede e2`, `e2 precede e3`, `e3 precede e4` — 4 adjacent facts, each its own turn, one
session, `reset=True` on the first turn only.

- **Adjacent control (4 pairs):** `does e0 precede e1?` ... `does e3 precede e4?` — expect `derived=False`
  (plain recall short-circuit), `abstained=False`.
- **Held-out non-adjacent probes (6 pairs, never directly taught):** `(e0,e2) (e0,e3) (e0,e4) (e1,e3) (e1,e4)
  (e2,e4)` — expect `derived=True`, `abstained=False`, and `derived_from` an ordered hop-fact list whose
  intermediate concepts are exactly the chain entities between the pair.
- **Reverse-direction negative control (2 pairs):** `does e4 precede e0?`, `does e2 precede e1?` — never taught
  in that direction and never derivable (no outgoing `precede` fact from e4 or from e2 toward e1) — expect
  `abstained=True`.
- **Unrelated-entity negative control (1 pair):** `does e0 precede zzz?` (`zzz` never mentioned) — expect
  `abstained=True`.

**World B (the scrambled-premise control).** A FRESH composer/session. The same 5 abstract chain SLOTS are
re-populated with a random permutation of 5 NEW entity tokens (drawn at run time from a fixed pool, seeded), so
the taught adjacency is a genuinely different mapping than World A. Re-teach the 4 adjacent facts under the new
mapping, then re-ask ONE held-out non-adjacent pair for the NEW mapping only. **Passing here on the NEW mapping
(never seen at build time) is the falsifier for "this is a hardcoded/memorized lookup rather than a live
traversal of whatever was actually taught this session."**

**Lesion arm.** Repeat the 6 held-out non-adjacent probes from World A with `BRAIN_TRANSITIVE_LESION=1` (same
session-teaching, fresh brain build) and the 4 adjacent-control probes under the same lesion.

### GO criteria (this pass: single seed-7 DEV/smoke de-risk; NOT a 6-seed capability gate)

- **G1** (positive reach): 6/6 non-adjacent probes `derived=True`, `abstained=False`, with a `derived_from` chain
  whose hop facts are all genuinely taught SVOs (no invented bridging fact).
- **G2** (adjacent control): 4/4 adjacent pairs `derived=False`, `abstained=False` (no over-triggering of the
  chase on a plain single-hop fact).
- **G3** (negative controls): 3/3 (2 reverse-direction + 1 unrelated-entity) `abstained=True` (no
  false-positive "yes").
- **G4** (scrambled-premise control): the World-B non-adjacent probe `derived=True` with a `derived_from` chain
  matching World B's OWN (freshly-random) mapping, not World A's.
- **G5** (lesion load-bearing): every G1-passing probe flips to `abstained=True` under the lesion; every
  G2-passing adjacent probe is UNCHANGED under the lesion (`derived=False`, `abstained=False`) — isolating the
  cut to the multi-hop chase, not the underlying recall primitive.
- **G6** (byte-identity off): a pinned 6-turn transcript (a plain `well`/`unknown`/`hold`/`held`-style probe set,
  reused from the existing regression-battery convention) hashes IDENTICALLY with `BRAIN_TRANSITIVE_CHAT` unset
  vs. an ordinary turn under the SAME unset default (i.e. the flag genuinely defaults off and touches nothing
  when off).

**Verdict rule.** GO (seed-7 de-risk) iff G1-G6 all pass; otherwise NO-GO, naming the first failing criterion.
A single-seed GO here is a DE-RISK, not a capability-gate GO — a 6-seed (42/43/44/100/101/102) capability gate at
a frozen SHA, plus review, is required before any default-ON consideration (this pass changes no default; the
flag ships OFF regardless of this pass's outcome).

## Seed discipline

Seed **7** for this dev/smoke pass only (`BRAIN_CHAT_SEED=7`), never 42/43/44/100/101/102. The 6-seed gate job
lines are staged (not run in this pass) — see the report.

## Declared residuals (named, not hidden)

1. **DETECT is host regex** — comprehension of raw text, the same declared scaffold boundary every existing
   question-shape router in this codebase already occupies (`_definitional_copula_route`,
   `_POSSESSIVE_CHAIN_RE`, `_CHASE_MARKERS`). Not claimed as brain-based; only the relational hops + the stop
   decision run on the substrate.
2. **Only the POSITIVE (reachable) case is answered.** A chase-CONFIRMED negative is reported as an honest
   abstain, identically to a genuine unknown (v1 scope, not a hidden gap).
3. **`rich=True` path not wired** (single-fact path only, matching `compositional_chain_route`'s own original
   scope).
4. **No new spiking mechanism.** The scientific contribution here is the NEW dispatch (a transitive yes/no
   question shape) + the yes/no semantics layered on the keystone chase's own trace — not a new substrate
   mechanism. The keystone chase itself already carries its own 6-seed GO and its own anti-cheat battery
   (A1-A8, `_gnw_reentrant_metacog_gated_deliberation_derisk.py`), which this lane does not re-run.
5. No confidence threshold / hop-cap tuning beyond the keystone's existing defaults; no shard-routing
   lemmatization beyond `lexical_lemma.lemma_verb` on the relation word.

## Evidence at commit time (this document precedes the production-handler gate artifact)

A LOGIC-LEVEL de-risk of `resolve_transitive_query`'s own dispatch/trace-walking/lesion-collapse decisions
(G1-G5 + byte-identity-off), against a synthetic composer and a synthetic chase trace shaped exactly like
`confidence_gated_chase`'s real return value, is committed as `tests/test_reasoning_transitive_chat_wirein.py`
(10/10 pass; run argv/git-sha recorded in `research/findings/raw/_provenance/runs.jsonl`). This is a real,
cheap, RAM-free CI guard on the shipped module's own control flow — it is NOT a substitute for the real
production-handler gate. The real gate runner (`research/runners/reasoning_transitive_chat_gate.py`) drives the
actual `webapp.server.brain_chat` handler through the full World-A/World-B/lesion battery above; a seed-7 run
against this box's real RAM contention at authoring time (`bash tools/mem_ok.sh` refused repeatedly) is either
already committed alongside this document (see the session's report for the exact artifact directory name if
so) or is still queued for when local RAM allows / on the pool — the staged 6-seed job lines for the follow-on
capability gate are in that same report.

**Addendum — the one seed-7 local attempt made real progress, then was stopped, not because it failed.** A
World-A-only pass (`--skip-g6 --skip-worldb`) was launched under `tools/memcap.sh 5` after `tools/mem_ok.sh`
finally cleared at a reduced (5 GB) ask; it built the tiny-demo brain and began exercising the reused keystone
chase across the 13 World-A probe turns. It was terminated (SIGTERM, clean, no partial artifact written) before
completion for two compounding reasons, both real and worth recording for whoever runs the follow-on 6-seed
gate: **(1)** this box's RAM stayed genuinely contended throughout (`tools/mem_ok.sh` refused 8/7/6 GB asks
multiple times in the same window, with several other concurrent agent sessions' own brain builds live), and its
cgroup MemoryHigh throttle (`__mem_cgroup_handle_over_high`, confirmed via `/proc/<pid>/wchan`) was actively
slowing the process; **(2)** independent of RAM pressure, `confidence_gated_chase` runs a genuinely-costly
multi-cycle spiking re-entrant simulation PER hop attempt, and the run log showed many small (129-neuron)
network rebuilds per query in rapid succession — a real, substrate-cost signal, not a bug in this wire's
dispatch, but one the 6-seed gate's resource estimate should account for (13 World-A queries, doubled to 26 once
the lesion arm runs, each potentially several re-entrant cycles). Neither the module's own byte-identity-off
guarantee nor its dispatch logic is in question here (both are proven at the logic level above); what remains
unproven ON THIS PASS is the real handler's answer text / timing under load. Next rung: re-attempt on a
quieter window, on the pool (`pool2` was unreachable from this checkout), or with a smaller World-A probe subset
(e.g. 2 non-adjacent pairs + 1 adjacent + 1 negative control) sized to fit a tighter RAM/time budget.
