# Tier-3 Option 2B — promote the develop-with-a-body slice onto the 24/7 develop harness — deep-research / design scoping (READ-ONLY)

**Date:** 2026-07-01 (autonomous loop; owner-directed Tier-3 follow-on)
**Type:** Design / scoping. **READ-ONLY — NO code / `sim/` / GPU edit.** This doc isolates and answers the ONE genuine
design question for **Option 2B**: what is the minimal *additive, default-off* seam that lets the shipped 24/7 develop
harness drive a **live-and-remember FORAGING day** (the validated Option-2A body) as each day's WAKE — instead of the
current LISTEN-only corpus WAKE — while keeping the validated SLEEP(consolidate)/GROW/PERSIST/BUNDLE/`should_continue`
scaffold + the crash-proof/pausable/resumable supervisor intact?

**Predecessors (do NOT re-derive):**
`2026-06-30-tier3-option2-develop-with-a-body-scoping.md` (the Option-2 scoping; §4 flagged **2B** as the follow-on
this doc details, and predicted "an additive default-off `per_day_agent_factory` seam in `develop_gpu`, still NO
`sim/` edit") · `2026-06-30-tier3-live-and-remember-first-slice.md` (Option 1, **6/6 GO** — the perceive-ground-store
`live()` loop) · the **Option-2A runner** `research/runners/_tier3_develop_with_a_body_derisk.py` (the validated
self-contained multi-day body-loop this promotes) · `2026-06-24-week1-develop-loop-console-capstone.md` +
`2026-06-23-longitudinal-develop-loop-GPU-GO.md` (the develop harness this reuses).

**Terms defined once.**
- **The harness** = the shipped 24/7 launcher/supervisor stack: `develop_run.py` (the resumable entry) +
  `develop_loop_supervisor.py` (the crash-proof/fsync/heartbeat/PAUSE hardening) + `scripts/develop.ps1` (the
  start/pause/resume/status verbs) + `docs/2026-06-28-develop-run-guide.md`. All three call **`develop_gpu`**
  (`_longitudinal_develop_loop_gpu.py`), the validated day-loop.
- **WAKE** = the day-stage where the brain acquires the day's knowledge. **Today's WAKE = LISTEN-only:** the
  `StreamCortex` hears the TinyStories corpus window-by-window + `build_agent` teaches authored curriculum facts.
  **The 2B WAKE = a FORAGING day:** the persistent `MergedNavConvAgent` forages a corridor, and its facts are the
  `lived_facts` it PERCEIVED + grounded.
- **The seam** = the additive, default-off hook on `develop_gpu` that swaps the WAKE.
- **2A** = the self-contained body-loop runner (owns its own multi-day loop; cannot regress `develop_gpu`).
  **2B** = wiring the 2A body into the SHIPPED `develop_gpu` + harness so it runs 24/7.

---

## 1. TOP-LINE — a wiring seam, or a genuine build?

**A WIRING SEAM — largely-done + cheap.** Both halves are validated GO and the harness is already engineered for
exactly this kind of substitution. Applying the SURPASS practice (pin the exact new bytes and measure the genuine
residual, don't accept "it's a build"):

**What already exists (the load-bearing machinery is all present):**
- The **body-day loop** is the validated 2A runner (`_tier3_develop_with_a_body_derisk.py`): a persistent
  `MergedNavConvAgent` built ONCE (`_build_agent`, `:418` of the Option-1 file), foraging a cumulatively-richer
  `DevWorld` each day, `_run_multiday` (`:112`) accumulating lived facts, retention re-test + moat + persistence
  anti-cheats. It is the exact "WAKE = a foraging day" artifact.
- **`develop_gpu` ALREADY grew TWO of the three seams the harness needs** — and they are already agent-agnostic:
  - **`per_day_save_hook`** (`_longitudinal_develop_loop_gpu.py:289`, fired at `:460`) — the BUNDLE + fsync + heartbeat
    seam. Default `None` = byte-identical.
  - **`should_continue`** (`:290`, polled at `:376`) — the day-boundary PAUSE predicate. Default `None` =
    byte-identical.
  Both are the seams the supervisor rides (§2d). The ONLY seam `develop_gpu` still lacks is the one that swaps the
  **WAKE+CONVERSE stage itself**.
- The **supervisor is genuinely agent-agnostic on the durability path**: `_fsync_lineage`/`_heartbeat`/PAUSE all ride
  `per_day_save_hook` + `should_continue` (`develop_loop_supervisor.py:198`, `:232`), which never touch the agent
  object. The 24/7 crash-proof/pausable/resumable machinery drives a body-day loop **unchanged** once the WAKE seam
  exists.
- The **BUNDLE path already accepts the `MergedNavConvAgent` verbatim** — a decisive de-risk this doc adds over the
  2A predecessor: `save_developed_brain(agent, ...)` calls `_inner_agent(agent)` = `getattr(agent, "agent", agent)`
  (`developed_brain_io.py:53-56`) and reads `.composer.concepts` / `.composer.kb`. `MergedNavConvAgent` exposes
  `.composer` directly (it has no `.agent` wrapper), so `_inner_agent` returns the merged agent and the extract reads
  the CO-RESIDENT composer's grounded codes + lived facts. **The per-day bundle of a developed foraging brain works
  with ZERO adapter** (the 2A doc marked this "✓" but hadn't confirmed the `_inner_agent` fall-through — it is now
  confirmed).

**The genuine residual (the exact new bytes) — ONE seam + ONE lifecycle guard + ONE sibling entry-point:**
1. **(R-1) The WAKE-swap seam on `develop_gpu` — the ONE genuinely new hook.** Today the WAKE+CONVERSE block
   (`:389-423`) hard-builds `cortex.hear_day(...)` + `read_codes()` + a FRESH `build_agent` + `_inject_grounded` +
   `_teach_fact(day facts)`. To let a foraging day be the WAKE, `develop_gpu` needs an additive default-`None`
   **`per_day_wake_fn(day_index, state, lineage)`** that, when set, REPLACES that whole block and returns
   `(agent, day_facts, day_vocab)` for the rest of the loop (SLEEP/METRICS/GROWTH/PERSIST/BUNDLE) to consume. This is
   ~15-25 lines of additive branch, byte-identical when unset (exactly like `per_day_save_hook`/`should_continue`
   were). **It is a research-runner edit (`_longitudinal_develop_loop_gpu.py`), NOT a `sim/` edit.**
2. **(R-2) The per-day agent LIFECYCLE inverts — a guard, not new machinery.** `develop_gpu` builds AND frees a fresh
   agent every day (`build_agent` `:407`, `_free_agent` `:468`). The body is ONE persistent `MergedNavConvAgent` alive
   across ALL days (it holds the drive/body/composer/perception state). **Critical `_free_agent` hazard** (`:502-509`):
   it frees `composer.bridge._cp`'s memory pool; for the co-resident composer that bridge **IS the shared merged
   bridge** — freeing it mid-life corrupts the persistent brain. The seam sidesteps this by construction: when
   `per_day_wake_fn` is set, the loop must NOT call `build_agent`/`_inject_grounded`/`_free_agent` (the wake_fn owns
   the persistent agent). This is a one-line `if per_day_wake_fn is None:` guard around the free — a lifecycle guard,
   no new code.
3. **(R-3) A sibling harness entry-point.** `develop_run.py` (`:120-135`, `:164`) and `develop_loop_supervisor.py`
   (`:154`, `:178`) hard-build a `GPUGradedCurriculum` + a `StreamCortex` and pass `_shared_cortex`. A body-run has
   NO stream cortex (the body IS the day's brain). So the harness needs a **sibling entry** (a `--with-body` flag on
   `develop_run.py`, OR a thin `develop_with_body_run.py`) that skips the `StreamCortex` build, builds the persistent
   `MergedNavConvAgent` once, and passes a `per_day_wake_fn` that runs a foraging day. It reuses the SAME
   lineage/PAUSE/bundle/`should_continue`/fsync scaffold. ~40-60 lines, no new mechanism.

**Quantified residual:** **one additive default-off param on `develop_gpu`** (`per_day_wake_fn`, ~15-25 LOC branch +
a 1-line free-guard, byte-identical to every current caller when unset) **+ one sibling harness entry** (~40-60 LOC
reusing the shipped scaffold) **+ the 2A body-day already exists**. **No `sim/` edit is predicted** (Option 1 + 2A
each needed none; every seam here is additive/default-off or in a research runner). The residual is a *wiring* seam,
not a new mechanism class — it composes two GO halves through machinery the harness already exposes.

**Verdict in one line:** Option 2B is **largely-done + cheap (a wiring seam)** — an additive default-off
`per_day_wake_fn` on `develop_gpu` (a research runner, NOT `sim/`) + a `_free_agent` lifecycle guard + a sibling
harness entry-point; the body-day, the BUNDLE acceptance, the fsync/PAUSE/resume durability, and the DEPTH-2 bundle
discovery are all ALREADY in place.

---

## 2. DIAGNOSIS — the exact seam + the lifecycle/state/bundle/supervisor reconciliations (cite file:line)

### 2a. The `per_day_wake_fn` seam — signature + slot-in point

**Slot-in point:** `develop_gpu`'s per-day body, the WAKE + CONVERSE stages, `_longitudinal_develop_loop_gpu.py:389-423`
(between the `should_continue` poll `:376` and the SLEEP `consolidate` at `:426`).

**Signature (additive, default `None` = byte-identical):**
```python
def develop_gpu(..., per_day_wake_fn=None):
    ...
    for d in range(n_days):
        day_index = start_day + d
        if should_continue is not None and not should_continue(): break   # :376 (unchanged)
        day_curr = curriculum.day_stream(day_index)                       # :382 (unchanged)
        for c in day_curr["new_concepts"]: ...                            # :385 vocab bookkeeping (unchanged)

        if per_day_wake_fn is None:
            # ===== the VALIDATED LISTEN-only WAKE+CONVERSE (byte-identical, :389-423) =====
            n_windows = cortex.hear_day(day_curr["new_concepts"], ...)    # :393
            _, _, grounded = cortex.read_codes(); learn_fid = cortex.learning_fidelity()   # :401-402
            agent = build_agent(...)                                      # :407
            _inject_grounded(agent, grounded)                             # :410
            if state.facts: [ _teach_fact(agent, f) for f in state.facts ]# :411-413
            for fact in day_curr["facts"]:                                # :416
                if plasticity_on: _teach_fact(agent, fact); state.add_fact(fact)
                state.t += 1
            owns_agent = True
        else:
            # ===== the 2B FORAGING WAKE (a live-and-remember day on the persistent body) =====
            # returns the persistent agent + the day's LIVED facts/vocab (already stored on the composer)
            agent, day_lived_facts, day_lived_vocab, wake_info = per_day_wake_fn(day_index, state, lineage)
            for f in day_lived_facts: state.add_fact(f)     # DevelopState mirrors the lived facts (for PERSIST/METRICS)
            for c in day_lived_vocab:
                if c not in state.vocab: state.vocab.append(c)
            grounded = wake_info.get("grounded", {})        # for the metrics/bundle metadata only
            learn_fid = wake_info.get("learn_fid", 0.0)
            n_windows = wake_info.get("n_steps", 0)
            owns_agent = False                              # the persistent agent is NOT freed at day end

        replayed = consolidate(agent, state, consolidation_on, rng)       # :426 SLEEP  (unchanged; see 2c-3)
        dp = _measure(agent, state, day_curr, replayed, ...)             # :431 METRICS (see 2c-1: probe list)
        plan = maybe_grow(promoter, mastery, state, lineage)             # :438 GROWTH  (unchanged; agent-agnostic)
        _save_state(state, lineage, latest_metrics=dp)                   # :453 PERSIST (unchanged; see 2c-4)
        if per_day_save_hook is not None: per_day_save_hook(day_index, state, grounded, agent)   # :460 BUNDLE
        if owns_agent: _free_agent(agent)                                # :468 GUARDED (R-2)
```
That is the entire seam: an additive branch that, when `per_day_wake_fn` is set, replaces WAKE+CONVERSE with a foraging
day and skips the per-day free. Every other stage (SLEEP/METRICS/GROWTH/PERSIST/BUNDLE) runs unchanged on the
persistent agent. When `per_day_wake_fn is None`, the code path is exactly today's (byte-identical to every current
caller: `develop_run`, the supervisor, the smoke).

**What the body-side `per_day_wake_fn` does** (lives in the sibling harness entry, reusing 2A verbatim):
```python
persistent_agent = _build_agent(seed)                 # 2A's MergedNavConvAgent, built ONCE (Option-1 :418)
hunger = SpikingHunger(persistent_agent._merged_bridge)
cache = set()                                          # cross-day grounded-object cache (never re-ground)
def per_day_wake_fn(day_index, state, lineage):
    world = DevWorld(day_index, order=DEV_ORDER)       # 2A's cumulatively-richer world (:82)
    facts_before = set(tuple(f) for f in live_state.lived_facts)
    live(persistent_agent, hunger, live_state, world, steps_per_day,           # the 2A/Option-1 live() call (:235)
         drive_reward="rate_proxy", perceive=True, commit_facts=True, grounded_obj_cache=cache)
    day_facts = [f for f in live_state.lived_facts if tuple(f) not in facts_before]
    day_vocab = list(live_state.encountered)           # cumulative; DevelopState de-dups on append
    return persistent_agent, day_facts, day_vocab, {"n_steps": steps_per_day}
```
This is a straight reuse of the 2A `DevWorld` + Option-1 `live()` + the persistent-agent lifecycle; the ONLY new
line is returning the day's freshly-lived facts.

### 2b. Stage-by-stage: does each `develop_gpu` stage accept the body?

| Stage | file:line | Accepts the persistent body? |
|---|---|---|
| **`should_continue` poll** | `:376` | ✓ agent-agnostic (zero-arg predicate). The PAUSE seam works unchanged. |
| **vocab bookkeeping** | `:385` | ✓ but sourced from `day_lived_vocab` (the objects encountered) not `day_curr["new_concepts"]`. |
| **WAKE** | `:389-402` | ✗-by-design — this is the LISTEN corpus stream being REPLACED by the foraging day (R-1). |
| **CONVERSE (build/inject/teach)** | `:404-423` | ✗ as-written — it builds a FRESH agent + injects corpus codes. The seam substitutes the persistent body + its lived facts. |
| **SLEEP (`consolidate`)** | `:426` → `_longitudinal_develop_loop.py:345` | ✓ with a store-path caveat (2c-3): `consolidate` re-teaches via `_teach_fact` = `agent.hear("a v p")`. `MergedNavConvAgent.hear` exists (`:2344`); a lived `near`-fact should re-`store` via the composer, not re-`hear` via the parser. |
| **METRICS (`_measure`)** | `:431` → `_query_recall`/`_query_yesno`/`_query_chain` | ✓ for recall/yesno (`what_does` `:2359`, `who_does` `:2373`, `is_it_true` `:2398` all present); ✗ for `_query_chain` → `agent.reason_chain` (R-c, still missing on `MergedNavConvAgent`). Fix: the body-day's probe lists are the 2A batteries (recall on lived facts + moat), with `probe_chain=[]` (the corridor produces only `near`-facts, so chain probes are vacuous). |
| **GROWTH (`maybe_grow`)** | `:438` → `_longitudinal_develop_loop.py:371` | ✓ pure-Python `TierPromoter.step(mastery)` + a lineage growth-event; never touches the agent object. |
| **PERSIST (`_save_state`)** | `:453` → `_longitudinal_develop_loop.py:439` | ✓ persists `DevelopState` (facts/vocab/tier/day/metrics) — BUT the body's load-bearing state also includes the 2A `LiveState` (body energy/pos/Q/drive + `grounded_codes`). Merge required (2c-4). |
| **BUNDLE (`per_day_save_hook`)** | `:460` → `save_developed_brain(agent, ...)` | ✓ **confirmed accepts `MergedNavConvAgent` verbatim** (`_inner_agent` fall-through, `developed_brain_io.py:53-56`; extracts `.composer.concepts`/`.kb`). The per-day bundle IS the developed foraging brain the console loads. |
| **`_free_agent`** | `:468` (`:502`) | ✗ MUST be guarded off for the persistent body (R-2 hazard: frees the shared merged bridge). One-line guard. |

### 2c. The precise seams the JOIN must handle (honest)

1. **(R-c, side-stepped) `reason_chain` missing on `MergedNavConvAgent`.** `_query_chain` → `agent.reason_chain`
   `AttributeError`s. The body-day never needs it (corridor → `near`-facts only), so the body-side probe list sets
   `probe_chain=[]`; the 2A runner already does exactly this (its `_verdict` uses no chain probe). Optional ~5-line
   `reason_chain` shim delegating to `self.composer.query_chain` if a later world produces chains — NOT needed for
   this slice.
2. **(R-2, guarded) `_free_agent` on the shared bridge.** Covered above — the seam skips the free when
   `per_day_wake_fn` is set. This is the single most important correctness point: freeing the co-resident bridge
   mid-life corrupts the persistent brain (`_free_agent` frees `composer.bridge._cp`'s pool, `:502-509`, and the
   co-resident composer's bridge IS the merged bridge).
3. **(SLEEP store-path) self-replay must re-STORE, not re-HEAR.** Lived facts are stored via
   `composer.store(prev,"near",cur)` (Option-1 `:315`); `consolidate`'s `_teach_fact` re-teaches via
   `agent.hear("prev near cur")` (parser path). For the body-day the replay should re-`store` (idempotent, faithful
   to the lived grounding) — a bare 3-token `near`-fact could mis-parse through the merged parser. **Cleanest:** the
   body-side loop passes its OWN consolidation (re-`store` a sample of `live_state.lived_facts`), OR the seam accepts
   a `per_day_consolidate_fn`. Since the 2A runner's retention re-test already re-queries lived facts (and the
   composer store is idempotent), the simplest correct choice is: on the body path, let `consolidate` be a no-op
   (`consolidation_on` gated by the wake_fn) and rely on the composer's durable store + the retention re-test — matching
   2A exactly. (Honest: as `develop_gpu` already documents at `:806`, the composer store is idempotent so the CLS
   retention contrast is a rate/symbol proxy; the load-bearing interference contrast is the spiking-store follow-on —
   inherited, not new to 2B.)
4. **(PERSIST payload) merge `LiveState` ⊕ `DevelopState`.** `DevelopState` (`_longitudinal_develop_loop.py:209`) holds
   facts/vocab/tier/day/metrics; the 2A `LiveState` (`_tier3_live_and_remember_derisk.py:183`) holds
   body/Q/drive/`grounded_codes`/`lived_facts`. For resume to restore BOTH the developmental trajectory AND the exact
   life, the body-run must persist a COMBINED payload. **Cleanest (no `develop_gpu` change):** the sibling entry owns
   the combined persistence — it keeps the 2A `LiveState` and writes `{body, memory, develop:{day,tier,metrics}}` to
   the lineage (the 2A `_persistence_check` path, `:486`, already does the `LiveState` half; add the `develop` dict).
   `develop_gpu`'s `_save_state` still writes `DevelopState` as today (harmless duplicate of facts/vocab/tier);
   resume re-instates the grounded codes + re-stores the lived facts (2A's exact re-instate). NOTE: on the body path,
   `develop_gpu`'s own resume re-hear of the cumulative vocab (`:361-365`) is skipped because there is NO
   `StreamCortex` (`own_cortex` is False when a wake_fn drives the day) — the body re-instates via the combined JSON
   instead.

### 2d. Does the shipped harness drive it — supervisor unchanged?

- **`develop_loop_supervisor.py` — its DURABILITY machinery drives a body-day loop UNCHANGED.** `_fsync_lineage`
  (`:95`), `_heartbeat` (`:119`), the PAUSE `should_continue` (`:232`), and the fsync-on-`per_day_hook` (`:198-217`)
  all ride the agent-agnostic `per_day_save_hook` + `should_continue` seams and NEVER touch the agent. The
  `PAUSE_EXIT_CODE` (42) semantics, the atomic `.new`+`os.replace`+fsync, the resume-on-relaunch — all work verbatim
  on a body-run. **What the supervisor DOES hard-wire** (so it needs the sibling entry, not an edit): it builds a
  `GPUGradedCurriculum` (`:154`) + a `StreamCortex` (`:178`) and passes `_shared_cortex=` to `develop_gpu` (`:250`).
  A body-run has no stream cortex, so `run_resumable` needs a `--with-body` branch that: (i) does NOT build the
  `StreamCortex`, (ii) builds the persistent `MergedNavConvAgent` once, (iii) passes a `per_day_wake_fn` instead of
  `_shared_cortex`. The fsync/heartbeat/PAUSE `per_day_hook` + `should_continue` it already defines are passed through
  unchanged. **⇒ the supervisor's crash-proof/pausable/resumable guarantees transfer to the body-run for free once the
  WAKE seam exists; the only change is a sibling build-branch, not the durability core.**
- **`develop_run.py` — same:** it hard-builds `GPUGradedCurriculum`/`CorpusGradedCurriculum` + `StreamCortex`
  (`:120-135`, `:164`) and passes `_shared_cortex`. A `--with-body` flag (or a `develop_with_body_run.py` sibling)
  that skips the cortex, builds the persistent body, and passes a `per_day_wake_fn` reuses the SAME `ROOT`/PAUSE/
  `BUNDLE_ROOT`/`LINEAGE_ROOT`/`per_day_hook`/`should_continue` scaffold (`:30-38`, `:167-181`). The `--status`
  path (`:40-54`) reads `DevelopState` from the lineage and is agent-agnostic (works unchanged on the body-run's
  `develop` dict if the combined payload keeps a `DevelopState`-shaped section).
- **`scripts/develop.ps1` — a one-line verb change:** the `start`/`resume` default (`:52`) calls
  `develop_run --corpus-curriculum ...`. A body-run adds `--with-body` (or a parallel `develop_body.ps1`). The
  `pause`/`resume`/`status` verbs (`:32-45`) are agent-agnostic (they touch the PAUSE sentinel + `--status`), so they
  work unchanged.
- **DEPTH-2 bundle discovery — works unchanged.** `develop_run.py` saves bundles DIRECTLY under `ROOT` as
  `run3day/day_<N>` (`:34-35`, `BUNDLE_ROOT = ROOT`), because the dashboard `_scan_developed_bundles` picker scans
  depth 1 + depth 2 only. A body-run under its own `ROOT` (e.g. `bridges/developed/body_week`) with the same
  `day_<N>` layout is discovered identically — the picker sees the developed FORAGING brains and the owner loads
  each day + chats with the brain at that developmental stage. (The bundle content is the co-resident composer's
  grounded codes + lived facts, per the confirmed `save_developed_brain` acceptance in §1.)

---

## 3. THE BIOLOGY (catalog-first, cited, brief) — develop-from-lived-experience over a week

Read from `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`. A 24/7 loop where **each day's knowledge
is LIVED, not listened**, is the coarse-grain complementary-learning-systems (CLS) day/night rhythm run continuously
over a developmental horizon — the animal explores by day (drive-biased), consolidates by night (replay), and its
knowledge is a consequence of what it lived:

- **D.01** Episodic memory — encode/store/retrieve/consolidation cycle (`:1085`). The develop-loop's core loop; on a
  body-day the "encode" is the LIVED `perceive_and_ground` + `composer.store(prev,"near",cur)` (the 2A/Option-1 day),
  not the corpus co-occurrence. This is the single entry that most directly grounds "the day's knowledge is lived."
- **D.19** Sharp-wave ripples — replay in quiet wakefulness + NREM (`:1309`) / **N.07** Hippocampal SWRs (the NREM
  replay the SLEEP stage stands in for). The mechanism behind the day-over-day retention re-test.
- **N.12** Sleep-Dependent Memory Consolidation — Stickgold/Tononi (`:4690`, ⭐). **Why a multi-day lived loop needs a
  SLEEP stage at all** — the no-forgetting the RETENTION metric measures across days.
- **N.17** Awake replay during behavioral PAUSES — Foster & Wilson 2006 (`:1010`, ⭐; supplemental `:4640`): SWR-like
  replay fires at rest/goal points *inside* waking behavior, not only in programmed NREM. A body-day naturally has
  eat/rest pauses → this is the biology for making the SLEEP stage *event-triggered* by the lived pauses (a follow-on
  — Option 4 — but N.17 makes it the NATURAL consolidation trigger for a foraging day, exactly the seam a 24/7 body
  harness later exploits).
- **D.23** Misplace system — hippocampal novelty detection driving EXPLORATION (`:1059`): the biological engine for
  "the agent chooses what to experience" (novelty-seeking = open-ended lived experience). The first 2B slice's
  open-endedness is the drive-biased trajectory over an agent-uncontrolled layout (sufficient); an explicit curiosity
  drive is a later slice.
- **O-cluster drive** (the motivational core making the day self-chosen): **O.05** hypothalamic homeostasis (`:4803`,
  ⭐), **O.06** AgRP/POMC feeding loop (`:4815`, ⭐) — the 2A/Option-1 co-resident 2-pool spiking hunger; **O.11**
  drive-reduction reward (`:4875`, ⭐) — the intrinsic `r = drive_before − drive_after` that makes the foraging day
  self-directed (the discriminator between a LIVED day and a scripted one); **O.21** average-reward continuing-task RL
  (`:533`) — the principled long-horizon reward formulation for a persistent 24/7 life (a follow-on upgrade to the
  per-day Q; not a blocker).

**Cross-cutting steer:** the harness's WAKE→SLEEP→GROW alternation over a continuous lineage IS the CLS
developmental rhythm; 2B's only change is that WAKE is a *lived* foraging day (O-cluster drive + D.01/D.23 encode)
rather than a *listened* corpus stream — the same scaffold, a biologically-richer knowledge source.

---

## 4. THE SINGLE RECOMMENDED cheap-first DE-RISK

**Add the additive default-off `per_day_wake_fn` seam to `develop_gpu` (a research runner) + a sibling harness entry
`develop_with_body_run.py` (or `develop_run.py --with-body`), then run a SHORT multi-day body-run through the ACTUAL
harness and confirm the four properties that make 2B a deployment (not just a re-run of 2A): per-day bundles
discoverable + pausable + resumable + the 24/7 durability rides through.** NO `sim/` edit predicted.

### The plan (concrete)
1. **The seam** (`_longitudinal_develop_loop_gpu.py`, additive default-`None` `per_day_wake_fn`): the branch in §2a +
   the one-line `_free_agent` guard (R-2). Assert byte-identity: with `per_day_wake_fn=None`, `develop_gpu` is
   unchanged (run the existing GPU smoke `--n-days 4` → identical trend/verdict to the shipped run).
2. **The sibling entry** (`develop_with_body_run.py`, reusing the shipped scaffold): build the persistent
   `MergedNavConvAgent` once (Option-1 `_build_agent`), define the `per_day_wake_fn` (§2a body-side: `DevWorld` +
   `live()` + return the day's lived facts), keep the 2A `LiveState`, and drive `develop_gpu` with the SAME
   lineage/`per_day_save_hook`(fsync+bundle)/`should_continue`(PAUSE) the supervisor uses. Combined-payload persist
   (§2c-4).
3. **Run through the ACTUAL harness** (GPU, `SIM_BACKEND=cupy`): a short body-run (e.g. `--n-days 3`, `steps_per_day`
   the 2A smoke value) under its own `ROOT` (`bridges/developed/body_week`), with `--per-day-bundles`.

### Decisive checks (GO / BOUNDARY / NEGATIVE)
1. **BYTE-IDENTITY of the unset path (HARD):** `per_day_wake_fn=None` → `develop_gpu` verdict/trends byte-identical to
   the shipped GPU smoke. A drift here means the seam is not additive → fix before anything else.
2. **DEVELOPS-over-days from LIVED experience (GO):** the lived-fact count GROWS day-over-day (the 2A gate-1); the
   day-N brain's `composer.kb` ⊃ day-0's, sourced from `live()`'s `lived_facts`, NEVER an authored list.
3. **NO-FORGETTING / retention (GO):** on the last day ALL accumulated lived facts (incl. day-0's) recall ≥ 0.8 (2A
   gate-2).
4. **NO-CONFAB MOAT byte-frozen (HARD):** every day the unstored `(obj,"chase")` cue ABSTAINS (None) AND the
   conversational synapses stay byte-identical across the whole run (2A gate-5). A breach is a HARD STOP.
5. **PER-DAY BUNDLES DISCOVERABLE (the 2B-specific GO):** each day writes `ROOT/day_<N>/` (via `per_day_save_hook` →
   `save_developed_brain`), the manifest carries the day's grounded codes + lived facts, and the DEPTH-2 picker
   `_scan_developed_bundles` lists them → the console loads a day-N developed FORAGING brain and it answers about what
   it lived + abstains. (This is what 2A did NOT test — the harness deployment.)
6. **PAUSABLE + RESUMABLE THROUGH THE HARNESS (the 2B-specific GO):** create the PAUSE sentinel mid-run → the loop
   stops cleanly at the next day boundary (`should_continue`), the last day is fsync'd; delete the sentinel + relaunch
   → the body-run RESUMES the exact developed life+memory (combined-payload re-instate) and continues developing (not
   a blank slate / not day 0). This is the 2A persistence gate-6 EXERCISED THROUGH THE ACTUAL 24/7 machinery.
7. **REWARD-PROVENANCE (GO):** `r` = drive-reduction (spiking/interoceptive drive), asserted NO `r=f(distance)` —
   inherited verbatim from Option 1/2A.

### Ladder
1. **1-seed GPU smoke through the harness (mechanics):** the seam is byte-identical unset (check 1); a 2-3 day
   body-run develops + per-day bundles land + a PAUSE→resume cycle continues the life (checks 2,3,4,5,6). This is the
   go/no-go rung.
2. **A short multi-day body-run (3-5 days):** vocab/facts grow, a tier fires (`maybe_grow`), bundles are console-loadable,
   the frozen-brain arm (`commit_facts=False`) accumulates no new facts (2A gate-3), the permuted-world arm yields a
   different fact set (2A gate-4).
3. **6-seed** (42/43/44/100/101/102) for the develop-over-a-week + all-anti-cheats claims (the standing 6-seed rule).

### Predicted `sim/` edit
**NONE.** Option 1 (the hardest half) needed none; 2A needed none; the 2B seam is an additive default-off param on a
**research runner** (`_longitudinal_develop_loop_gpu.py`, byte-identical to every current caller when unset) + a
sibling harness entry that reuses the shipped scaffold. The one interface gap (`reason_chain`) is side-stepped by the
corridor's `probe_chain=[]`. If a later, richer world produces chains, the `reason_chain` shim is a ~5-line agent
composition in the runner — still NO `sim/` edit.

---

## 5. HONEST SCOPE / expected boundaries

- **JSON re-instate vs raw `cp_connections` persistence.** Resume re-instates the grounded codes + re-stores the
  lived facts from the combined JSON (2A's exact path, `_tier3_live_and_remember_derisk.py:486-516`), NOT the raw
  merged-bridge synaptic tensor. True `cp_connections` persistence of the merged bridge is a follow-on (the shipped
  develop harness itself uses the same JSON stand-in — `develop_gpu` re-HEARS the vocab on resume, `:361`; the body
  path re-instates codes+facts instead). This is an inherited stand-in, not a 2B regression.
- **The 4-object corridor bound.** The 2A `DevWorld` renders `DEV_ORDER = ["apple","cat","dog","river"]` (a subset of
  the gen stack's `OBJECT_WORDS`, N=4), so the developed graph is a short chain (~3 lived `near`-facts over the growth
  days). A RICHER multi-day development (more perceivable objects, a 2D path-dependent world, pair-accumulation) is a
  SEPARATE follow-on — 2B is the HARNESS wiring, not a world upgrade. Deploying 2B does not require the richer world;
  it makes "a brain develops over a week it LIVED" runnable 24/7 at the validated corridor scale, and the richer world
  drops into the SAME seam later.
- **The CLS retention contrast is a rate/symbol proxy.** The composer store is idempotent (re-storing is a no-op), so
  retention is naturally high; the load-bearing interference contrast (new learning genuinely overwriting old) is the
  fully-spiking-store follow-on — inherited verbatim from the shipped harness (`_longitudinal_develop_loop_gpu.py:806`),
  not new to 2B.
- **Consolidation is a scripted SLEEP phase.** N.17 event-triggered lived consolidation (SWR replay fired at the
  eat-pause) is Option 4 — a follow-on after 2B is GO.
- **The learned spatial policy stays the deferred Tier-4 dendrite wall.** Survival uses the validated Option-1/2A
  rate-proxy Q stand-in; survival (not spatial optimality) is the discriminator.
- **Wall-clock per simulated week.** The body-run builds ONE persistent merged bridge (build once) + runs `live()`
  each day, which steps the bridge only for the drive corr sweep (once) + the per-object groundings (`rate_proxy`
  survival is pure host between groundings). 2A's smoke ran a few short days on one seed in the low-minutes range;
  extrapolated, a short simulated WEEK is order **tens of minutes on one 3090** (the harness already reports
  `mean_day_seconds` + `compressed_week_eta_minutes`, `_longitudinal_develop_loop_gpu.py:745-746` — confirm the actual
  per-day wall-clock in the 1-seed smoke). A full simulated year is a hands-off overnight run via the 24/7 supervisor.
  LOCAL (no VRAM wall — one merged bridge fits 24GB comfortably); GPU required (`SIM_BACKEND=cupy`); pausable on demand
  for gaming (the PAUSE sentinel, not a VRAM cap).

---

## 6. VERDICT for the owner

**Yes — Option 2B is the right next slice after Option 3 (if Option 3 is prioritized first), and it is cheap.** It is
**largely-done + a wiring seam**: the body-day is the validated 2A runner (6/6 GO), the 24/7 crash-proof/pausable/
resumable/fsync/heartbeat/DEPTH-2-bundle machinery is all already shipped and agent-agnostic on its durability path,
and the per-day BUNDLE path **confirmed accepts the `MergedNavConvAgent` verbatim** (no adapter). The genuine residual
is **one additive default-off `per_day_wake_fn` param on `develop_gpu`** (a research runner, byte-identical to every
current caller when unset, ~15-25 LOC) **+ a one-line `_free_agent` lifecycle guard** (the co-resident-bridge hazard)
**+ a sibling harness entry-point** that skips the `StreamCortex` build and drives a foraging day through the SAME
scaffold (~40-60 LOC). **No `sim/` edit is predicted** (Option 1 + 2A each needed none; the seam is on a research
runner, not `sim/`).

**Recommended:** build the `per_day_wake_fn` seam + the sibling entry, then run the 1-seed-smoke → multi-day → 6-seed
ladder **through the ACTUAL harness**, with the two 2B-specific decisive checks foregrounded — **per-day bundles
discoverable** and **PAUSE→resume continues the developed life** — on top of the inherited 2A anti-cheats (develops-
over-days · no-forgetting · frozen-brain-flat · **no-confab moat byte-frozen** · permuted-world-differs ·
reward-provenance). The HARD gates are byte-identity of the unset path and the no-confab moat byte-frozen across the
whole run.

**Stays deferred (off the critical path):** the richer world (more objects / 2D / pair-accumulation), true
`cp_connections` synaptic persistence, N.17 event-triggered lived consolidation (Option 4), O.21 average-reward
continuing-task RL, and the learned spatial policy (Tier-4 dendrite wall). This slice converts the validated
self-contained 2A body-loop into the owner's north-star — **a brain that develops over a week it LIVED, running 24/7,
crash-proof, pausable, resumable, with a loadable developed brain for each day** — on the merged one brain, moat
intact, with no `sim/` edit.
