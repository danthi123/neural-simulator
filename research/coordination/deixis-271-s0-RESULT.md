# [deixis-271-s0] RESULT — stage-0 deixis encoding census, complete

**Packet:** `deixis-271-s0-PACKET.md` (board #271). **Status: PROVISIONAL** — agent-posted measurement, no gate
seeds (packet: none), for Claude's ruling. **STOP after this file** per the packet step list.

## The record (all three instrument-validity bars PASS)

- **Determinism:** two runs byte-identical, sha256 `f84e868a1e2f64bd…` — pool canary #5 (pool41) == full pool census
  (pool40, `research/findings/raw/_deixis_census/census_pool.json` + prov). The earlier pre-s0b local pair is likewise
  byte-identical, `ff1fcc28…` ×2.
- **Perfect-encoder bar:** PASS — 0 defects.
- **Raw-token null bar:** PASS — 84 defects, exactly the analytically expected 84.
- Probe set sha256 `e2918e3f812e53af…`, oracle sha256 `3f83d6b3e8ad8b24…`; 126 items × 2 speakers; dev seed 7;
  numpy CPU, pool40 under `memcap 1` (measured W4 peak 303,240 KB ×1.3); provenance sidecars on both files.

## Seven measures per writer

**W1 `ChatBrain._maybe_acquire`, B3 ON (production default):**

1. Deictic/anaphoric tokens stored as entity names: **28** (`moi→"moi like jazz"`, `ours`, `im`, `yous`, `yours`…).
2. Role shifts from pronoun deletion (verb promoted to agent): **4** (`"i told kofi jazz"` → agent `told`;
   `gave`; `chase`; `like`).
3. Cross-speaker collisions (A's and B's "my" stored as one entity): **60**.
4. Facts silently dropped because a pronoun was deleted: **60** (`"i chase cat"` and `"you chase cat"` store nothing).
5. Spurious B3 rejections across speakers: **25**.

**W2 same writer, `BRAIN_NONCONTRADICTION_GATE=0` (legacy raw-token path):**

1. **36** (worse than W1: `i chase cat` stores `i` as a name, `you` too).
2. **0**. 3. **52**. 4. **68** (drops more). 5. not applicable (no gate).

**W3 TUI `_discourse_turn` on the multiturn agent (A/B rows in the record):** the heard pronoun becomes the mouth's
own voice: `"i chase cat"` → reply `now i is doing it.`, `"moi likes jazz"` → `now moi is doing it.`,
`"you chase cat"` → `now you is doing it.` — the register flips to the heard first/second person verbatim.

**W4 `extract_assertion` + reconsolidation (two arms, RULING `deixis-271-s0b`):**

- Extract-level deictic-as-name defects: **28** (same class: `moi`, `ours`, `im`, `yous`, `yours`…).
- Coded arm (in-codebook pairs): **3/3 rewrite** with live surprise window and exact `_reactivate_idx` string match.
- `w4_demos_uncoded` (original pairs, fresh composer each): **3/3 raised** (`KeyError 'south'/'rock'/'jazz'`) —
  a writer crashing on a word it never coded is a silent drop, recorded as `raised_type`/`raised_msg` + counts
  `{coded: 3/0, uncoded: 3/3}`.

**6 Question side (10 fixed questions):** `DEFAULT_SELF_ALIASES` maps `i/my/you/your` → `brain`
(`"who am i"` → `am brain`, `"are you awake"` → `brain awake`, `"what about my cat"` → `brain cat`×2) — **but `we` is
NOT aliased**: `"what did we learn yesterday"` keeps `we` (alias_hits empty). Gap recorded.

**7 Slug scan:** 4 declared source files scanned; **17** first/second-person deictic tokens present in those sources
(`em i im me mine my our ours thee ur us we ya you your yours yourself`); **conversation-only = 0** (nothing in the
declared store could have come only from conversation). Deviation noted: the packet expected an enumeration of the
235 slugs; the census records the exact source paths scanned instead.

## Reading (baseline for #271 stage 1)

The B3 default path trades recall for honesty: it stores **fewer** deictic names than legacy (28 vs 36) and drops
**fewer** facts (60 vs 68) at the cost of 25 spurious rejections — while every path still stores pointing words as
entity names at scale, and the extract-level rate (28) shows the encoder, not the gate, owns the defect. Stage-1
mechanism design starts from these numbers.

**STOP:** awaiting Claude's ruling. Board #271 next action: "Claude reads the census".