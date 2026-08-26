---
type: finding
status: contributing
lane: conversation-integration
date: 2026-08-25
seed-waiver: single-session qualitative frontier map (one warm brain, one seed); no positive GO claimed — characterization only
---

# Integrated conversational-state diagnostic — how the production brain actually converses (2026-08-25)

**Status:** contributing (a single-session QUALITATIVE map of the integrated `/api/brain-chat` turn — one
seed, one warm brain; NOT a seeded GO. It characterizes the live frontier so the owner can steer the next arc).

**Base:** `origin/main` @ `a645cca3a`. Branch `research/integrated-convo-diagnostic`.
**Artifacts:** the raw turn-by-turn JSON is
`research/findings/raw/_integrated_convo_diagnostic/transcript_2026-08-25.json` (19 turns; the same data
is also kept as `.jsonl`) + `run_log_2026-08-25.txt`. Every quote below is verbatim from that transcript
(numbers shown rounded for readability are marked `<!--derived-->`; their exact values are in the artifact).

## Method

One warm brain, reused across all 19 turns, driven through the REAL production handler
(`webapp/server.py` `brain_chat()` via FastAPI `TestClient`, in-process). `SIM_BACKEND=cupy` (the owner's
real path — a 3090 is present, so production defaults to cupy). Opt-in source-provenance honesty enabled
(`BRAIN_SOURCE_PROVENANCE_HONESTY=1`); everything else at production default (onebrain composer, rich
multi-sentence default, all Gate-B organs, GNW bus, continuous engine, D5, affect/swap/DA drives, generate
channel, 15k-core ship-default).

**One faithfulness caveat, called out up front.** This diagnostic ran in an isolated worktree whose data
lake is incomplete: the Qwen priming corpus (`data/corpus/tinystories.txt`) and the developed-brain bundles
(`bridges/developed/*`) are absent (both present in the primary checkout). Two consequences, both handled:
(1) the auto Qwen mouth hung on warm in the worktree, so the run used `BRAIN_CHAT_RENDERER=stub` — the
template mouth. This does NOT affect the diagnostic's object: every faculty coupling wraps the rendered
text, so all couplings are exercised; only prose FLUENCY is a stub (assessed separately below). (2) the
default brain is `tiny-demo` regardless; the knowledge-core wiring gap (below) was additionally confirmed
by a direct store query, so it is not a worktree artifact.

## 1. Inventory — what is production-default (verified against code, not prose)

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` lists 43 rows; ~40 read `on_by_default: YES`. Verifying the ACTUAL
code defaults: nearly all are genuinely default-ON (unset → on). The named targets this diagnostic was asked
to inventory:

- GNW bus (shadow / 2-organ / 3-organ / deliberation / multistep) — default-ON (`BRAIN_GNW_3ORGAN` default `"1"`, etc.).
- 15k knowledge core (`BRAIN_LTM_SHIP_DEFAULT` → `_LTM_SHIP_DEFAULT_ON=True`) — the bundle resolves by default.
- D5 learn-through-use (`BRAIN_D5_CONSOLIDATE` default-on, runs inside the continuous idle tick).
- Continuous-state engine (`_CONTINUOUS_DEFAULT_ON=True`) + wander-drives + ideation — default-ON.
- DA-gated encoding (`_DA_ENCODING_DEFAULT_ON=True`, flipped default-ON 2026-08-25) — default-ON.
- Affect / swap / DA-mode DRIVE-couplings (`_AFFECT_DRIVES_DEFAULT_ON` / `_SWAP_DRIVES_DEFAULT_ON` /
  `_DA_DRIVES_DEFAULT_ON` all True) — default-ON.
- Source-provenance honesty — OPT-IN (`BRAIN_SOURCE_PROVENANCE_HONESTY` default-OFF); enabled for this run.

**Inventory drift noted (documentation, not behavior).** Several rows' human-readable FACULTY prose and
inline `server.py` comments still say "WIRED default-OFF" while the code default was later flipped ON and the
`on_by_default` field moved (e.g. the `da-gated-encoding` handler comment at `server.py:4253` reads
"Default-OFF" but `da_encoding_enabled()` returns True by default as of 2026-08-25; `server.py:1770` calls
the continuous engine "Default-OFF" while `continuous_engine.py:79` is `_CONTINUOUS_DEFAULT_ON=True`). The
machine-checked `on_by_default` fields are largely correct; the prose lags the flips. A `sync-documentation`
pass should reconcile the stale prose.

## 2. Per-faculty load-bearing assessment (from the live transcript)

### Faculties that DRIVE the surface (observably change the reply)

- **Recall + no-confab abstain** — WORKS but BRITTLE. `"what does the dog chase?"` → `"The dog chases cat.
  The cat eats fish."` (verified, chained). Unknown topics abstain honestly: dragon / physicist / chelsea →
  `"I don't know about that."` The moat holds. BUT recall of a JUST-TAUGHT fact is inconsistent (see B-phase).
- **Affect-drives (#84)** — LOAD-BEARING. The neutral turns read valence 0.0, level 0, empty lead. The
  positive message `"Wow, I absolutely love wolves...!"` moved `ema_valence` 0.0 → 0.407, level 0 → 2, and <!--derived-->
  prepended the lead `"Gladly — "`, which then persisted (EMA) across later turns. The felt state genuinely
  colors the surface.
- **Swap-drives (#85)** — LOAD-BEARING. A topic change yields a transition lead naming the newly-held
  coalition: `"On wolf, then — "`, `"On dog, then — "`, `"On deer, then — "`; a same-topic follow-up holds
  silent (`reason: same_topic_hold`, empty lead). Clean and consistent across turns 8/13/14/15/17.
- **Continuous-state engine (wander)** — LOAD-BEARING. After a real idle gap the log printed
  `continuous tick: evolved 1 idle session(s)` and the next turn led with
  `"(I'd been mulling over cat.) I don't know about that."` — a between-turn wandered concept surfaced into
  the live reply.
- **Self-initiation** — WORKS. An empty turn produced an internally-generated remark+question:
  `"Something's been on my mind — cat eat worm. What does cat eat?"` (path `ca3-wander-cupy`, 600 settle
  steps, pooled_member 0.417 vs pooled_random 0.036 — a genuine selection, not chance). Fidelity caveat: the <!--derived-->
  drawn patients ("worm" for cat, "ball" for dog) are NOT the stored patients (fish / cat), so the concept
  SELECTION is genuine but the utterance PATIENT is an unverified draw.
- **Comprehension / other-repair** — WORKS and is honest. `"say something on your mind"` (out-of-vocab
  "say"/"mind") → `"I followed the shape of that, but I don't know the words 'say' or 'mind' yet — what do
  they refer to?"` — a comprehension-gated clarifying question naming the unresolved tokens.

### Faculties that are WIRED + default-ON but DORMANT or BROKEN in the live turn

- **The whole DOPAMINE axis is inert — a runtime bug on every turn.** All 19 turns show
  `da_drives.reason = "error:ValueError: non-scalar numpy.ndarray cannot be used for fill"`, `acted:false`,
  `da_level:0.0`, `mode:rest`. The SNc afferent read (`webapp/da_mode_drives_chat.py` `observe_turn`) raises
  on the cupy substrate and is swallowed by the handler's guard. Because the DA LEVEL never rises: the
  DA-mode engagement suffix never fires; DA-gated encoding stays `g:1.0` (no modulation); the DA-gated
  curiosity threshold gets `da_level:0.0` (no modulation). Three default-ON faculties are silently
  neutralized by one caught error. (Observed on this cupy build; the guard makes it invisible without
  inspecting the `da_drives` trace.)
- **Curiosity follow-up (D3)** — DORMANT. On every abstain, `curious:false` (and the DA gate is inert), so
  NO honest follow-up question is appended. On an unknown topic the brain abstains flatly rather than
  craving/asking.
- **Open-ended generation (#3E)** — DORMANT here. `"what might the wolf chase?"` → `"...I don't know about
  that."` No flagged hypothesis was volunteered even though `(wolf, hunts, deer)` was in the store.
- **Source-provenance honesty** — WIRED but OFF THE DEFAULT PATH. The provenance monitor runs ONLY on the
  single-SVO (`rich=False`) branch (`server.py:5163`), where it correctly labels a recalled fact
  `perceived` (`d≈1.0, agrees_with_encoded:true`). Every DEFAULT (rich) turn carries `provenance: null` —
  so on the owner's default conversation the honesty label is absent.
- **D5 learn-through-use** — UNTESTABLE this session. Its recall-strength surfacing requires the used memory
  to be recallable; the taught wolf fact was not reliably recalled (below), so no consolidation strength
  could surface and `"recall strength … mV"` never appeared.

### The content core is a single-SVO lookup — it does not reason, and recall is fragile

- **In-loop learning is fragile via morphology.** `"the wolf hunts the deer"` taught `(wolf, hunts, deer)`
  (verified). The immediate recall `"what does the wolf hunt?"` → `"On wolf, then — I don't know about
  that."` — an ABSTAIN on the just-taught fact. The stored action `hunts` does not match the queried `hunt`
  (no lemmatization). The SAME question later recalled correctly at C1/D4, so recall is inconsistent
  (a verb-inflection + WM-swap interaction), not reliably reproducible.
- **No multi-hop reasoning / own-conclusion.** After teaching `(wolf, hunts, deer)` and `(deer, eats,
  grass)`, `"what does the wolf's prey eat?"` → `"...I don't know about that."`, and the fully-spelled-out
  chain `"the wolf hunts the deer and the deer eats grass; so what does what the wolf hunts eat?"` →
  `"...The wolf huntses deer."` — it echoes a single stored fact and never derives `grass`. The
  compositional question is not routed to any multi-hop inference.
- **The 15k knowledge core is not reachable on the default brain.** Every knowledge query abstained. The
  15k store WORKS in isolation (`ShardedPhasorStore.load(...).query_patient("chelsea_fc","country") →
  "united_kingom"`; `"penicillium","instance_of" → "taxonomic_group"`), but `_build_chat_brain` attaches
  the LTM ONLY to the `developed-brain` path (`server.py:3497`), NOT to the default `tiny-demo` brain
  (`server.py:3477`). So the owner's out-of-box chat has no access to the shipped knowledge. Even on a
  developed brain, the query interface needs exact underscore Wikidata tokens (`chelsea_fc`, `country`);
  natural phrasing ("what country is chelsea fc from") does not route (tokenization + the copula→`isa` vs
  store `instance_of` lexicalization gap).

## 3. Honest conversational-capability verdict

The integrated brain currently converses as **an honest, affect-coloured, topic-aware FRAMING layer wrapped
around a brittle single-fact lookup that abstains most of the time and never reasons to a new conclusion.**
The framing faculties are real and genuinely drive the surface — mood ("Gladly —"), topic transitions ("On
wolf, then —"), between-turn wandering ("I'd been mulling over cat"), internally-initiated turns, and honest
comprehension repair ("I don't know those words — what do they refer to?"). The no-confab moat holds; the
brain does not fabricate. That honesty + self-referential framing is a genuine, differentiated surface.

But the CONTENT it frames is thin: recall is a single stored SVO (or an abstain), it is fragile to verb
inflection, it cannot chain two explicitly-stated facts, it cannot reach its own 15k knowledge on the
default brain, and the whole dopamine axis (mode / encoding / curiosity-gate) is silently erroring. So the
conversation reads as sincere and self-aware in FORM while remaining a near-stateless single-fact Q&A in
CONTENT. It does not yet "reason to its own conclusions"; it recalls one fact and dresses it.

## 4. The single biggest next wall + ranked mechanisms

**THE WALL: there is no path from a natural compositional/reasoning question to a multi-hop inference over
the brain's own facts.** Every turn collapses to one atomic SVO recall (or abstain), so the north-star
capability — reasoning to its OWN conclusion — is directly falsified (D3/D4). The many working DRIVE
faculties wrap a non-reasoning core, so improving them further yields sincerer framing around the same empty
content. The inference machinery partly EXISTS (`ShardedPhasorStore.chain_of_thought` / `query_chain`) but
the live comprehension front-end never invokes it.

**Ranked candidate next mechanisms (grounded in the transcript):**

1. **Route compositional/relational questions to the EXISTING multi-hop chain engine.** The bottleneck is
   the comprehension front-end, not the inference store — extend the neural question parser to recognize a
   relational / multi-clause question ("what does X's Y eat", "what does what X hunts eat") and dispatch it
   to `chain_of_thought` / `query_chain` over the fact graph, VERIFY-gated per hop (moat preserved). Highest
   leverage: it converts stored facts into DERIVED answers — the actual north-star — reusing machinery that
   already exists.
2. **Make atomic recall reliable first (verb-lemmatization at store + query).** The `hunts`≠`hunt` miss made
   even a just-taught fact abstain. A morphology-normalized key on both the write and the query removes the
   fragility that would otherwise make any reasoning chain non-reproducible.
3. **Give the reasoner knowledge and repair the neuromodulator: attach the 15k LTM to the default brain (or
   ship a developed brain as the default), and fix the cupy `da_mode_drives_chat.observe_turn` `.fill()`
   error** so the dopamine mode / encoding / curiosity-gate faculties actually fire instead of degrading to
   `rest`. Without (3) the reasoner has nothing to reason over and three default-ON faculties stay inert.

## Caveats

Single seed, one session, one warm brain — a qualitative map, not a seeded verdict; the recall
inconsistency in particular wants a multi-seed repro before any mechanism claim. Stub renderer (worktree
Qwen-corpus gap) — fluency was not assessed on the real mouth; the Qwen warm-hang is very likely a worktree
data-lake artifact, not a production bug, and is reported as such. The DA `.fill()` error and the
knowledge-core / provenance wiring gaps are substrate/wiring facts independent of the renderer and were each
cross-checked (direct store query; source-line inspection).
