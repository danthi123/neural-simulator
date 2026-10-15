# One brain, many conversations: the shared-memory seam (DESIGN)

*2026-10-13. Design only; no production code changes in this document's branch. Owner decisions of 2026-10-13
("Approved on all as per your suggestions"): ONE brain for everyone, memories source-tagged ("X told me") through the
existing source-monitoring faculties; memory steps 1-3 pulled forward from Phase C to now (shared read-only imported
store; one shared learned memory; persistence default-ON in production, latest checkpoint at boot, tests never write);
the wikidata_100k store (78,857 facts, 0 learned) stays a declared prior with the learned share reported beside it.
Step 4 ("remember what matters") stays the Phase C research gate. Step 1 is being built in parallel on
`claude-build/shared-ltm` (at s/main 6a3bfe247 when this was written; assumed to land). Authoritative code ref:
s/main 6a3bfe247; every file:line below was read at that ref.*

Non-negotiables this design is held to: brain-based only (host code = world, body, teacher; a host computation is a
scaffold at most), ONE spiking substrate shared across faculties, no-defer, faithfulness first but falsifiable
(six-seed lesion), the honesty boundary (every self-report is a functional read-out; no phenomenal claim).

## 0. Today, verified

- `webapp/server.py:3012` `_BRAIN_CHATS` caches one `ChatBrain` per `(session, brain, renderer)`; built by
  `_build_chat_brain` (`:4083`). Each build attaches its OWN `ShardedPhasorStore` LTM (`:4151`, ~5.1-5.4 GiB per
  session, finding `2026-10-12-battery-worker-per-kind-retention`). Nothing is shared between conversations except
  the frozen read organs (`:3041-3098`), the xedge pool and the warm Qwen renderer (`:3685`).
- Per-session registries keyed identically: `_BRAIN_RICH :3020`, `_SESSION_MOOD :3030`, `_SESSION_WORLDVIEW :3038`,
  `_SESSION_MULTIREF :3103`, `_SESSION_SILENT_WM :3111`, `_SESSION_DISCOURSE :3118`, `_SESSION_PMEM :3126`,
  `_SESSION_SELFINIT :3133`, plus module registries: D5 `_ORGANS` (`d5_episodic_production_organ.py:146`),
  common ground (`common_ground_ledger_production_organ.py:184`), causal (`causal_whatif_production_organ.py:413`),
  emotion channels (`emotion_category_organ.py:585`), false belief (`webapp/false_belief_chat.py:250`), idle replay
  (`idle_replay_organ.py`, `reset_organ`), and the continuous engine's per-key state (`webapp/continuous_engine.py:59-84`).
  The teardown that enumerates them is `_evict_brain_chat_session` (`server.py:4617-4650`).
- `brain_chat` (`:4652`) is a sync FastAPI handler (threadpool): there is NO turn lock today; only
  `_WARM_QWEN_LOCK` (`:3685`). Two sessions can run turns concurrently against the process-shared organs.
- Persistence (#B8, `research/runners/cross_session_persistence.py`): per-session identity dirs (`:276`, `:455`),
  xedge under `xedge/pool` (`:372`); saves on the sleep-depth tick (`continuous_engine.py:1247-1251`,
  `SLEEP_IDLE_SEC=300` at `:533`) and shutdown (`server.py:1982`); reload after build (`server.py:3407-3420`),
  fingerprint-gated (`:198-206`, `:348-352`); checkpoints with `latest`/pin/retention (`:114-183`);
  `_PERSIST_DEFAULT_ON = False` (`:67`). The pytest subset `tests/test_cross_session_persistence.py:91`
  pins flag-OFF as inert.
- In-session learning ON: D5 learn-through-use (`continuous_engine.py:149-160`), sleep replay (`:311-340`),
  `_maybe_acquire` (`brain_chat_tui.py:1348`), reconsolidation (`server.py:6440-6510`, organ default-ON).
- The D5 organ's topic set is fixed at build: `note_topic` returns False for an unknown topic
  (`d5_episodic_production_organ.py:104-107`). This is the one structural constraint that shapes section 4.

## 1. State inventory

Legend: **B** = BRAIN-LEVEL (one copy, shared, learned, checkpointed); **C** = CONVERSATION-LEVEL (one per live
conversation, never checkpointed, dies with the conversation); **B/C** = a brain-level organ with a small
conversation-level state on it. "Biology" = why the split is the biological one, not a convenience.

| State (file:line) | Class | Biology / rationale |
|---|---|---|
| Composer fact buffer `composer.kb` + runtime codes (`brain_chat_tui.py:1348` writes; `ChatBrain.stored_facts/agents_set :992`) | **B** | Learned facts are synaptic content of the one cortex-hippocampus system; a fact told in one room is known in the next. Today per-ChatBrain, so this is THE Step-2 move. |
| LTM `ShardedPhasorStore` (`server.py:4151`, `tiered_fact_store.py:112`) | **B** (read-only prior) | The imported 78,857-fact store is the declared prior (owner 2026-10-13): one copy per process (Step 1). Never written by a turn; `promote_buffer_to_ltm` is the future consolidation hook. |
| D5 episodic store `EpisodicRecallOrgan` per cache_key (`d5_episodic_production_organ.py:146,166`) | **B** | Episodes ("someone told me about the dog") are hippocampal traces of the one brain. The query "did WE discuss X" becomes "did YOU tell me X" = content completion (D5) + a source read (section 3). Hard case, see below. |
| Idle-replay engram record + excitability tags `t_enc` (`idle_replay_organ.py:449`) | **B** | CREB-excitability allocation tags belong to the engrams, i.e. to the store they index. |
| xedge pool, frozen cross-edge (`onebrain_xedge_production.py`, `xedge/pool`) | **B** | Already process-global; one substrate. |
| Frozen read organs: affect, comprehension, surprise, metacog, worldmodel, curiosity, #129 provenance, pragmatic, self-schema (`server.py:3041-3098`) | **B** | Already process-shared; calibrated once then frozen (`:4238-4260`). The #129 monitor's encode-once-per-fact-key trace (`source_provenance_honesty.py:132`) is brain-level LEARNED state and must be checkpointed (today it is lost at restart). |
| Causal what-if organ per key (`causal_whatif_production_organ.py:413-417`) | **B** | Grounded read-only against the composer; one composer -> one organ, rebuilt when the shared store changes. |
| Self-initiated organ (`_SESSION_SELFINIT :3133`, own mouth + selection substrate) | **B** target, **C** today | One brain has one idle-selection substrate; it is per-session today only to isolate its demo lexicon. Declared residual until idle replay (brain-level) replaces its selection (Stage-2 C5 build). |
| Mood EMA `_SESSION_MOOD :3030` | **B/C** hard case | See "Affect" below: tonic baseline **B**, phasic appraisal **C**. |
| Affective forward-model prediction `_SESSION_WORLDVIEW :3038` | **C** on a **B** organ | The prediction is about THIS interlocutor's next turn; the forward model (the organ) is the brain's. |
| Multi-referent WM `_SESSION_MULTIREF :3103` | **C** | Persistent-activity working memory of the current discourse; a WM register holds one conversation's referents (the docstring's leak argument is the biology: WM is capacity-limited and task-bound). |
| Activity-silent WM `_SESSION_SILENT_WM :3111` | **C** | Short-term synaptic facilitation (Mongillo 2008) decays within seconds-minutes; discourse focus of this conversation. |
| Discourse event register state `_SESSION_DISCOURSE :3118` (register object on the agent) | **C** | Clause-boundary state of the current discourse. |
| Prospective-memory latch `_SESSION_PMEM :3126` | **C** (flag) | A held intention toward THIS person ("remind me when..."). Hard case: an intention is brain-level biologically (BA10 holds it regardless of who is in the room), but its cue and its beneficiary are this interlocutor. Keep **C**; owner call whether an intention may fire in another person's conversation (recommend no). |
| Common-ground ledger per key (`common_ground_ledger_production_organ.py:184`) | **C** | Common ground is with THIS interlocutor (Clark). Partner-specific pacts that outlast a conversation are a **B** memory ABOUT a person (a follow-on; not in scope). Note: its reset is missing from `_evict_brain_chat_session` (known gap, `:4625`). |
| Held emotion channels (`emotion_category_organ.py:585`) | **C** on a **B** organ | The category organ is shared; the held channel is the current interaction's. |
| False-belief scenario (`false_belief_chat.py:250`) | **C** | A ToM scenario about this interlocutor. |
| Rich discourse thread `_BRAIN_RICH :3020` | **C** | "Tell me more" continuation is discourse state. |
| Continuous-engine per key: consolidation budget, `_RECALLED_TOPIC`, `_LAST_REQUEST` (`continuous_engine.py:59-66,553`) | **C** | Which memory THIS conversation just used, and when this interlocutor last spoke. |
| Continuous-engine `_SLEEP_REPLAYED`, `_LAST_SLEEP_KB` (`:74-84`) | **B** | They index the shared store's un-replayed tail; sleep is a brain event (section 2). |
| Self-model: self-schema organ (`self_schema_production_organ.py`), self-knowledge facts (`_load_self_knowledge`) | **B** | One self. |
| Warm Qwen renderer (`server.py:3685`) | **B** (mouth) | Conditioned articulation; already shared; no learned state. |
| Interlocutor identity, conversation id | **C** + a **B** index | The conversation id dies with the conversation; the interlocutor id is what the source tag binds to (section 3). |

**Affect (hard case).** One brain has one neuromodulatory tone, so a hard conversation with A should leave a trace
when B arrives; but today's `_SESSION_MOOD` is the phasic appraisal of the current exchange. Recommendation: split
into a tonic baseline (brain-level, slow EMA over all conversations, checkpointed) and the phasic EMA (per Context,
`_MOOD_EMA_DECAY=0.4` as today). The reply reads tonic+phasic; the honesty read-out may say "my baseline is low from
an earlier exchange" without naming who. Owner call (section 7): whether cross-conversation mood bleed is wanted at
all in deployment; it is the biological default.

**The episodic "this conversation" query (hard case).** "Did we discuss X" today means "did THIS session store X"
because the D5 store is per session. With one store it decomposes into three reads: content completion (D5, spiking,
unchanged); WHO (the source monitor, section 3); WHEN/WHICH conversation. The WHEN pool was scoped and not built
(`d5_episodic_production_organ.py` header, "no spiking recency signal yet"), so "in this conversation" is a declared
host scaffold (the conversation tag on the store record) until a context/time population exists; its biological
replacement is named: a temporal-context assembly (Howard-Kahana) formed at conversation start and co-stored with
each episode, completed at recall.

## 2. Architecture

**`Brain` (one per process)** owns: the composer buffer and its runtime codes; the LTM prior (Step 1); the D5
episodic store; the idle-replay engram record; the xedge pool; every frozen read organ; the #129 provenance trace;
the source-monitor organ (section 3); the self-model; the tonic mood baseline; the sleep scheduler; the checkpoint
identity; and ONE `turn_lock` (`threading.RLock`). It is the only object that owns plastic synapses with a lifetime
longer than a conversation.

**`Context` (one per live conversation)** owns the **C** rows of section 1 (mood phasic EMA, worldview prediction,
multiref WM, silent WM, discourse state, pmem latch, common-ground ledger, held emotion channels, false-belief
scenario, rich thread, consolidation budget, last-request time) plus `conversation_id`, `interlocutor_id` and
`source_slot` (section 3). `_BRAIN_CHATS` becomes `_CONTEXTS: dict[conversation_id, Context]`; the ChatBrain's
per-conversation fields (`held_referent`, `_boundary_seen`, ...) move into Context; its brain-wide fields
(agent/composer/caches) move into Brain.

**Binding a turn.** `brain_chat` resolves `Context` from the request (`session` today; `interlocutor_id` added,
section 7), then `with brain.turn_lock: brain.bind(ctx); reply = brain_reply(...)`. Inside the lock every organ
getter that takes `cache_key` today takes `ctx`; brain-level getters take nothing. The mouth (Qwen render) runs
AFTER the lock is released: the brain's decision is done, articulation is not the brain. The between-turn tick
(`continuous_engine`) also takes the lock; sleep becomes a BRAIN event: idle >= 300 s across ALL contexts (today
`_is_sleep_tick` is per key, `:553`), which is the biology (one organism sleeps) and the persistence cadence.

**Concurrency.** Turns are serialized by `turn_lock` (FIFO; a bounded queue with a per-request timeout returns a
`busy` status instead of stacking). Rationale beyond safety: one global workspace ignites one content at a time; the
serial turn is the GNW's own constraint, not an engineering shortcut. Latency: a tiny-demo stub turn is sub-second,
an LTM recall ~1.1-1.3 s (ledger `scale_note`); N simultaneous interlocutors see at worst N x turn latency. Measure
in M2 (section 5) and surface a "thinking" status in the console. This is a deployment trade, not a cognition one.

**One substrate (non-negotiable 2).** All plasticity with a lifetime beyond a conversation happens in Brain-owned
synapses: the composer store blocks, D5 within-assembly weights, the #129 trace, the source-monitor trace, engram
excitability. Context-owned circuits (multiref WM, silent WM, pmem latch, common-ground NMDA ledger) are
persistent-ACTIVITY or short-term-facilitation states, not long-term synapses; they share process-cached calibration
per seed (as pmem does today `:3126-3131`) and their referent codebooks stay per Context. Declared residual, not new:
each is still its own `SimulationBridge` today; the one-substrate path for them is the xedge `MergedPool`, which is
the existing one-brain integration lane, not this seam. Nothing in this design adds a bridge.

## 3. Source tagging ("X told me")

Three source faculties exist; none is reinvented:

1. **Perceived vs generated** (#129, `source_provenance_production_organ.py`, process-shared, default-ON): did the
   brain SEE the fact or INFER it. Encoded at `server.py:7149-7152` and `brain_conversational_agent.py:822-823`.
2. **Self vs heard** (agency corollary-discharge, finding `2026-08-01-agency-authorship-...-6seed-GO`, wired as the
   DR-3 self-schema `author` read, `self_schema_production_organ.py`): did I say it or did you.
3. **Which external source** (Lane C source monitor: `plastic_source_memory.py` source populations, competitive
   heterosynaptic-LTD encoding, finding `2026-08-11-source-monitor-competitive-encoding-...-6seed`, 5/6 with one
   instrument-invalid seed): which person told me. NOT in production today. This is the "X told me" tag.

**Design.** A brain-level `source_monitor_production_organ` wraps (3) by import (the organ convention of
`source_provenance_production_organ.py`): at STORE time the fact's content pattern is driven together with the
interlocutor's source afferent (encoding only); at RECALL the content alone drives, and the read is which source
population completes. The interlocutor -> source-population map is a host scaffold of the same class as the
common-ground organ's word -> slot map: it assigns a population, it never decides the read. Population count is a
capacity (start n_banks=4: `self`, `imported prior`, and two live interlocutors; grow by allocation). A fact from the
LTM prior reads `imported`; a derived conclusion reads via (1).

**Host bookkeeping, declared.** Each store record (composer kb fact-dict; D5 `store_log`; idle-replay record) gains
`{src: interlocutor_id, conv: conversation_id, t_enc}`, in the same place `t_enc` already lives
(`idle_replay_organ.py:449`). It serves (a) the checkpoint sidecar, (b) the privacy policy gate below, (c) the
ground truth the lesion instrument scores the spiking read against. It is a scaffold: a `docs/SCAFFOLD-LEDGER.md`
row, `Active`, replacement = the organ above. With `BRAIN_SOURCE_MONITOR=0` the reply may use the host tag ONLY if
it says so (`provenance_info.reason = "host_tag"`), exactly as the D5 self-consistency fallback is declared.

**In replies.** `recall_disclosure` (`d5_episodic_production_organ.py:263`) and the provenance framing extend to
four honest frames: "you told me" (source = this interlocutor; "earlier in this conversation" only with the WHEN
scaffold), "someone told me" (source = another interlocutor), "I was given this" (imported prior), "I derived this"
(#129 generated). Each frame is a functional read-out of a monitor; none claims experience.

**Privacy and honesty (owner call, section 7).** One brain will know what A said when B asks. Policies:
(A) open with attribution ("Alice told me"); (B) attributed-anonymous ("someone told me", never the name);
(C) consent-scoped (A's facts recalled only to A unless A marks them shareable); (D) by kind (world facts shareable,
personal facts scoped). Recommendation: **B now**, with (D) as the follow-on once a "personal" classifier is a spiking
read rather than a regex. (B) is honest by construction and the lesion-measurable framing survives; (C)/(D) add a host
policy gate that must be declared a scaffold (it filters the reply; it is not the brain abstaining) and reported in
`provenance_info.policy`. Not a brain decision; the owner's call.

## 4. Persistence (Step 3)

**Production vs tests.** The server is `uvicorn webapp.server:app` (`server.py:15`). Add `BRAIN_PRODUCTION=1`, set
ONLY by the production launcher (a `tools/serve_production.sh` + the homelab unit; `docs/plans/2026-09-30-home-to-
server-MIGRATION.md` conventions). Resolution in `persist_learning_enabled()`: an explicit `BRAIN_PERSIST_LEARNING`
wins; else ON iff `BRAIN_PRODUCTION=1`; else OFF. Belt and braces: `PYTEST_CURRENT_TEST` set -> OFF even if armed,
unless `BRAIN_PERSIST_ALLOW_UNDER_PYTEST=1` (the persistence tests themselves, which use `tmp_path`; precedent
`tools/aws_spend_ledger.py:151-153`). pytest, `-m research.runners.*`, the battery worker, the stage gates and pool
jobs never set either flag, so they stay byte-identical OFF; `test_flag_off_is_default_and_inert` stays green.

**Identity: the one brain.** Replace `_d5_identity_dir(cache_key)`/`_facts_identity_dir(cache_key)` with ONE
identity `bridges/persist/brain/<brain>/<seed>/<layout-fp>/checkpoints/<id>/`; xedge joins it (one checkpoint id for
everything, as `save_session_learning` already mints one id across D5+facts, `:550`). Contents: composer facts +
runtime codes (reuse `save_session_facts`), D5 CSR weights + `store_log` + source/conv tags, xedge weights,
idle-replay engram record, the #129 trace, the source-monitor trace, the tonic mood baseline, and a manifest
(`schema_version`, code layout fingerprint, counts: `n_learned`, `n_prior=78857`, `learned_share`). No Context state
is ever saved.

**Cadence, atomicity, rollback.** Save at brain-level sleep (section 2) and shutdown (both exist), plus a safety
interval (every 30 min of wake activity; owner-tunable). Write `checkpoints/<id>.tmp/` then rename (atomic on one
filesystem). Pinning and rollback exist (`BRAIN_PERSIST_CHECKPOINT`, `:143-160`); add `tools/brain_ckpt.sh
list|pin|rollback|diff` as the operator surface. Retention: `BRAIN_PERSIST_KEEP_N=20` (`:99`) plus one daily
checkpoint kept 30 days (recommend). A checkpoint that fails its own reload self-check is quarantined, never `latest`.

**Fingerprint and migration.** Today a reload rebuilds at (seed, topics) and overwrites `.data` only on a CSR
structure match (`:316-352`); the saved `topics` already sit in the sidecar (`:299`). With a shared brain the topic
set GROWS, so a positional overwrite against the FRESH brain's topics would mismatch on every new agent word and
silently drop learning. Rule: rebuild the D5 organ from the CHECKPOINT's topics (not the live vocabulary), overwrite,
then grow. Growth needs incremental topic allocation in `EpisodicDapMemory` (a build item; `note_topic` refuses
unknown topics today). Until it lands, migration is by REPLAY: re-store the `store_log` in order into a fresh organ
built on the union topic set (the biology: sleep re-encodes; replay is how a trace survives a changed cortex), logged
`migrated_by_replay`, never a positional overwrite across layouts. A code change that alters the substrate layout
changes `<layout-fp>` (hash of the structural constants: n_ca3, D, k_max, organ versions) and therefore the identity
dir; boot then migrates the newest old-layout checkpoint by replay from its facts + store_log and records it. Bump
`SCHEMA_VERSION` on any sidecar change; an unknown version degrades to baseline with a logged count, as today.

**Disk.** Facts json: O(100 KB) per 1k facts. D5: `n_ca3=2000` CSR data in native dtype (expected O(1-10 MB); measure
in M0 and write the number here). xedge, traces, tags: small. Expect O(10 MB) per checkpoint, under 1 GB retained.
`/tmp` is RAM on the hosts: the default stays on disk under `bridges/persist` (`:80-89`).

## 5. Migration plan (default-off, then flip; each step with its validation)

Instrument first. **M0** `research/runners/one_brain_probe.py`: teach in conversation A, ask in B, under arms
intact / shared-store frozen (plasticity gate shut) / shared-store zeroed / restart-with-persistence / restart-without.
Reports reply-level fields (`answer`, `abstained`, `recalled_svo`, `provenance_info`), per `docs/TERMS.md`
"load-bearing". Also lands the `turn_lock` and the bounded queue (pure engineering; single-conversation byte-identity
asserted by the battery hash).

- **M1 Step 1 (other agent), `BRAIN_SHARED_LTM`.** One `ShardedPhasorStore` per process. Validation: battery §9.1
  reply hashes equal ON vs OFF; RSS per new session drops by ~5 GiB (the retention finding's measure).
- **M2 Brain/Context refactor, behind `BRAIN_ONE_BRAIN` (default OFF).** OFF = today's per-session ChatBrain through
  the new classes (every `_SESSION_*` dict becomes a Context field; byte-identical by hash on the battery and the
  stage-1 harness). ON = brain-level rows of section 1 move to Brain. Validation at ON, six seeds: M0 intact arm
  recalls in B what A taught; frozen and zeroed arms abstain; a single-conversation server stays hash-identical to
  OFF except where a brain-level row changes a reply by design (the tonic mood term; listed, not hidden).
- **M3 Source tags.** (a) host tag scaffold (`BRAIN_SOURCE_TAG_HOST`, instrument + sidecar); (b) the
  source-monitor organ default OFF, six seeds: "you told me" vs "someone told me" framing follows the true source,
  collapses to chance under the organ's own encoding lesion (`LEARNING-OFF`), stays correct across a restart.
  Flip (b) only on 6/6 (the 5/6 seed-702 control failure must be re-run; the lane's instrument fix is on record).
- **M4 Privacy policy gate** per the owner's section-7 call; declared scaffold; validation = reply-level test of the
  chosen policy, plus the honesty check that the frame still comes from the monitor.
- **M5 Persistence default-ON in production (`BRAIN_PRODUCTION=1`).** Validation: teach in A, restart, recall in B
  (ON) six seeds; the same with OFF loses it (the #B8 gap proof, now cross-conversation); pytest and the battery
  never write (assert no file under `bridges/persist` after a run); `latest` resolves; pin and rollback round-trip;
  layout-fp change triggers replay migration with the taught fact surviving.
- **M6 Reconsolidation in place in the shared store** (section 6).
- **M7 Learned share.** `/api/brain-chat` status and the board report `learned_share = n_learned / (n_learned +
  78857)`; the Step-4 gate plugs in after this.

Flip rule for every step: six seeds (42/43/44/100/101/102), the battery and the stage-1 harness green in both arms,
`verify-go` before the flip commit, the ledger row (`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`) with the default
anchor. Byte-identity is asserted by hash on data, never inferred (`docs/TERMS.md`).

## 6. Interaction with ongoing work

- **Stage gates.** `stage1_exit_gate.py` runs one fresh process per arm and one session per group (`:93`, `:446-452`,
  `reset=(i==0)`); the Stage-3 draft clones that harness. Per-arm process isolation keeps "one fresh brain per arm"
  true under one-brain. Within a process, groups that assume a fresh session is a clean brain would share learned
  facts after M2-ON. Fix: `reset=True` on a NON-production process (no `BRAIN_PRODUCTION`) evicts the Context AND
  resets the Brain's learned store (`Brain.reset_learned()`, refused when `BRAIN_PRODUCTION=1`); the harness's
  provenance sidecar records `one_brain` and `reset_learned`. Alternative, if a reviewer rejects a test-only reset
  path: one worker process per (group, arm); honest cost = more process builds.
- **Validation battery.** Same `reset` semantics; its isolated-session assumptions (`onebrain_regression_battery.py:
  54-133`) hold again. The 5.3 GB/session retention is removed by M1+M2 (one Brain; a Context is small), so
  `LB_BATTERY_RELEASE_SESSIONS` becomes a no-op and E1 equivalence is re-asserted.
- **Hermes pool jobs.** Runners never set `BRAIN_PRODUCTION`/`BRAIN_PERSIST_LEARNING` and do not start the server;
  unaffected. The battery worker gains an `LB_ONE_BRAIN` provenance key so sidecars say which mode ran.
- **Reconsolidation append gap** (prereg `2026-10-12-validation-battery-prereg.md:308,394`: a verbatim re-tell moved
  F2 to a new slot; a corrected F1 was appended beside the stale slot). Verified route order: the surprise read and
  the in-place rewrite run in `brain_reply` (`server.py:6440-6510`) BEFORE `rich.answer(msg)` (`:6763`), whose
  `ChatBrain.answer` (`brain_chat_tui.py:2185`) calls `_maybe_acquire` (`:1013,1087,1348`), which stores the
  assertion with no reactivation check. The untouched stale slot says the rewrite also did not take on this route
  (candidate: the `TieredFactStore` wrapper, `tiered_fact_store.py:273`, in front of `update_on_mismatch`); isolate
  before fixing (`before_you_build.sh`). Design requirement for the shared store (M6): ONE write decision per heard
  assertion: reactivate (cue match) -> prediction error (spiking surprise) -> rewrite in place, or no trace -> store
  new, or no PE -> re-stabilize with no new slot. `_maybe_acquire` becomes that decision's "store new" branch.
- **Phase C step 4.** It plugs into `Brain.sleep()`: today sleep replay reinforces the tail stored since the last
  sleep (`continuous_engine.py:311-340`) and idle replay carries `t_enc`/salience tags. "Remember what matters" =
  the replay selection policy (DA/affect-weighted priority, premise protection) and the shared buffer's capacity rule
  (promotion to the LTM tier via `promote_buffer_to_ltm`, forgetting). The checkpoint after sleep is exactly where its
  effect is written. Nothing here pre-empts it; M7's learned-share is its baseline number.

## 7. Risks and open owner calls

1. **Cross-person disclosure policy** (section 3). Recommend (B) attributed-anonymous now; (D) by kind later.
2. **Mood bleed across conversations.** Biological default is yes (one tone). Recommend tonic baseline brain-level,
   phasic per Context, honest read-out without naming the other party. Owner may prefer phasic-only in deployment.
3. **Serial turns.** N interlocutors -> N x latency worst case. Recommend lock + bounded queue + "thinking" status;
   measure in M2 before any further mitigation; the mouth renders outside the lock.
4. **Shared-buffer capacity.** The composer working set is capped (`k_max=32`, `tiered_fact_store.py` header) and
   `promote_buffer_to_ltm` is not auto-invoked. One shared learned store fills the buffer in days. Interim (owner
   call): auto-promote at sleep (engineering, declared) vs raise `onebrain_k_max`; the real answer is Step 4.
5. **Interlocutor identity.** `session="default"` (`server.py:4529`) would merge everyone into one source. The
   console must send a stable interlocutor id (recommend a signed cookie id, no accounts) and a conversation id.
6. **Test-only learned-store reset** (section 6). Recommend the refused-in-production `reset_learned()`.
7. **The AI teacher writes into the one brain** through `brain_chat` (`ai_teacher_guard.py` boundaries). Recommend
   yes, tagged `src=teacher`, reported in the learned share by source.
8. **Bad learned state persists** (a wrong or hostile fact survives restarts). Mitigation: rollback/pin, quarantine
   on self-check failure, and the source tag makes it attributable. Owner call: a curation surface in the console.
9. **Fingerprint fragility** across code changes: replay migration (section 4); a lost-learning event is logged as a
   counted degradation, never silent.
10. **Honesty drift**: "someone told me" must be a monitor read. Guard: `provenance_info.reason` names the read
    (`source_monitor` vs `host_tag`); a reply frame with no named read is a gate failure (`gates/production_integration`).

**Pure engineering** (no cognition claim, byte-identity expected): M0 lock/queue, M1 shared LTM, the M2 refactor at
OFF, checkpoint identity/cadence/atomic write/rollback tool/launcher flag, the host tag bookkeeping, the policy gate,
the test-only reset, the learned-share counter. **Cognition seams** (six-seed, lesion-verified, honesty-framed): the
brain-level learned store (M2-ON), the spiking "who told me" source monitor (M3b), the temporal-context "which
conversation" read (named, not built), the tonic/phasic affect split, the one-write-decision reconsolidation (M6),
sleep as a brain event with Step 4's replay policy, incremental D5 topic growth (or replay migration until then).

## 9. Owner rulings on the open calls (2026-10-13)

- **Deployment model: one brain per INSTANCE.** A person or a household runs its own sim. That instance's data stays accessible only to it, and a fresh-brain start is supported, so a brain can be tailored to its users through interaction over time, personal details included. "One brain for everyone" (the earlier ruling) means one brain shared by the instance's users.
- **Disclosure: decide by kind of fact (option D).** Until the brain can tell personal facts from general ones, the interim policy is option B: a fact taught by one user is used for another user, framed anonymously as "someone told me". The goal is that the brain itself LEARNS what to share with whom, as a social-cognition faculty. The host policy gate in M4 is a declared scaffold with that faculty named as its replacement.
- **Identity: a typed name for now.** The TUI and the web console both ask for a name at the start of a conversation. This is UNSETTLED and needs a design pass. The direction is OpenAI-API compatibility: identify the user as OpenAI-compatible chat APIs do, with the request-level `user` field and per-message `name`. Do not build cookie-based identity.
- **Claude decided the remaining calls (2, 3, 4, 6, 7, 8) as recommended in §7.** These are: tonic/phasic affect split; serial turns behind the lock plus a bounded queue; auto-promote at sleep as the interim capacity rule; a test-only reset path; teacher writes tagged src=teacher; rollback/pin/quarantine.
