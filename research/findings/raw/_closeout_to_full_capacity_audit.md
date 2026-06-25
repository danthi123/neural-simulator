# Close-out-to-FULL-CAPACITY audit (2026-06-24)

**Owner critique being audited:** we COMPLETE + validate biologization / one-brain-substrate arcs
but STOP at "validated opt-in / GO de-risk / probe" instead of taking them to FULL CAPACITY (flip
the default, wire into production, retire the legacy to a test-oracle, confirm no-pending). Map every
recent such arc's full-capacity gap + rank the closures so they can be worked to ZERO.

**Standard:** "make the fully-spiking the DEFAULT -> retire the legacy numpy production (keep as test
oracle)" + "anything that can reasonably be default-on, on by default, so there's truly no pending
work." **Nuance (`feedback_close_arcs_to_full_capacity`):** "full capacity" KEEPS the rf/rate/numpy
path as the test-ORACLE + the CPU-portable path; "flip the default" usually means "default-on for the
GPU/production path, retained as oracle/CPU", NOT a blind global flip. Flagged where a flip would break
numpy-CPU portability.

**READ-ONLY.** No edits/runs/webapp.

**Relationship to the prior close-out audit (`2026-06-24-closeout-audit-default-on.md`):** that audit
covered the *burndown's spiking-replacement flags* (it concluded "essentially NO pending code work").
This one is BROADER per the owner's current critique — the one-brain ARCS (option A, persistent_loop,
cross-region A+B), the communicable-brain probe, the console (B1/B2/B3), the develop loop. **Headline:
most of the broader surface ALSO already closed since that audit + the capstone-console scoping (the
commits postdate them) — but THREE genuine full-capacity gaps remain, and exactly ONE is a cheap,
green-gate-pending default-flip (Option A gate 3).**

---

## (1) PER-ARC TABLE

| # | Arc | Validated? (gates) | Full-capacity close-out step | Done / Left | Safe-to-close-now? | Nuance / oracle-to-keep |
|---|-----|--------------------|------------------------------|-------------|--------------------|--------------------------|
| **A** | **Co-resident `OneBrainComposer`** on the merged nav+conv bridge (`co_resident_composer_kind`, default `'rf'`) | **gates 1+2 = 15/15** (`test_nav_conv_merged_agent` 8 + `test_nav_conv_step2b_coresident` 7, default rf unregressed + co-residence + moat); **Probe 1 byte-identical** (CPU atol 1e-9, 30/30 + moat 8/8); **gate 4 GO** (limbic write-side load-bearing at GPU scale, g=1.423 exact, lesion pins 1.0, moat HARD). **gate 3 (nav Δ=0 for the onebrain opt-in + a GPU byte-identity) OUTSTANDING.** | Run **gate 3** (nav-not-regressed Δ=0 with `co_resident_composer_kind='onebrain'`, + a GPU byte-identity) → flip the GPU/merged composer default to `onebrain`, keep `rf` as oracle. | **LEFT** (gate 3 not run; default still `'rf'`). | **The cheapest+safest remaining default-flip.** Keep `rf` (`MergedRFComposer`) as the **byte-identity oracle**. The onebrain inner composer already builds with `persistent_loop=True` (`nav_conv_merged_bridge.py:1870`). A flip forces GPU on the merged-agent path (acceptable — the merged bridge is already `SIM_BACKEND=cupy`-only). |
| **persistent_loop** | `OneBrainComposer(persistent_loop=...)`, default OFF, byte-identical | **De-risk GO** (`_persistent_loop_flat_derisk.json`: 3-seed CPU, `worst_membrane_maxabs=0.0`, matrix identical, moat 0-FA; flat **AND** clause paths). **Finding:** the flat+clause recall is **ALREADY a persistent on-bridge spiking loop** — `persistent_loop=True` only FORMALIZES the clean-phasor re-kick handoff, byte-identical. | Flip `persistent_loop` default-on (confirm byte-identity holds) **OR** leave OFF (it is a byte-identical no-op formalization → flipping changes nothing observable). | **LEFT** (default OFF) — but this is the **softest "gap"**: it is byte-identical, so a flip has zero behavioral effect. Already `True` on the onebrain merged path (`:1870`). | Keep OFF at the library/numpy-oracle default for byte-identity + because the recall is *already* substrate-resident (the flag is a formalization, not a capability). **Low value to flip** — the real one-brain handoff lands via Option A's onebrain default, where it's already on. **No dedicated CI parity test** (only the de-risk JSON pins it). |
| **Cross-region A+B** | **Route A (language→action) default-ON DONE** (`nav_conv_merged_bridge.py:1830-1831`, `co_resident_command_route=None`→True; gates 15/15 + spoken-nav gate-4 GO seed 42). **Route B (perception→compose host-`M`) CLOSED in the runner** (`build_compose_bridge` default `gen_spikes`; 6-seed GO, compose 1.000, moat 0-FA, lesion collapses). | Route A: DONE. Route B: expose `gen_spikes` on the **agent** default (the runner default is already spikes-only). | Route A **DONE**; Route B **LEFT at the agent-constructor surface** (default-flip is at `build_compose_bridge`, not the agent). | Route A: **CLOSED.** Route B: **NEEDS-a-build, not a pure flip** (exposing `gen_spikes` on the agent needs the gen stack + composer in the agent default). | **The `co_resident_perception` + `onebrain` combo raises a guard** (`nav_conv_merged_bridge.py:1804-1806`). **This IS the genuine residual** — Route B's host-`M` writes `composer.concepts`, which the onebrain synaptic-store path doesn't yet consume. So "perception→compose on the onebrain consolidated brain" is the open seam (flagged follow-on in both the A and A+B findings). Keep host-`M` as the `--grounding host_m` A/B comparison only. |
| **Communicable-brain Probe 1** | **6-seed GO** (`2026-06-24-communicable-brain-probe1-GO.md`: NOVEL 1.00, GROUNDED 16.7×, FLAGGED 0-leaks, CALIBRATED 1.00; LESION/PROVENANCE 46/46). | Wire the gate→PROPOSE→VERIFY→emit **"what do you think"** turn into the production agent / interact console (behind a flag). | **LEFT** — it is a standalone PROBE (`_communicable_brain_probe1_whatdoyouthink.py`), NOT wired into `ChatBrain`/`RichAnswerComposer`/the `/api/brain-chat` console. | **NEEDS-a-build + an owner call** (NOT a green-gate flip). | The fluency faculty is a **CPU content-locked stub**; the real GPU spiking-Qwen wire-in is the drop-in follow-on (identical VERIFY contract, already proven to catch a 0.5B's drift). The moat RELAXES to speak-while-flagging (the owner-sanctioned graded-confidence path), NEVER removed — a who/what query on an unstored fact still abstains. Also: CYCLE 537 honesty note — the b2 PROPOSE is **host-sampled** (numpy `rng.choice` over the learned graph); the brain supplies grounding+likelihood, the spiking-SWR-replay sampler is the eventual fix. So wiring it in is a *capability addition*, and the values-call (how talkative) is itself an open arc (CYCLE 538-539 learned-talkativeness, in flight). |
| **B3 activity-viz** | **Live-verified** (CYCLE 524; staleness bug found+fixed). Default-off toggle. | Confirm default-off is the right default (vs default-on). | **DONE** (wired end-to-end: composer `trace` flag default-off → endpoint flips it read-only → frontend `brainchat-show-activity` checkbox, **no `checked` = default-off**). | **CLOSED.** **Default-off is CORRECT** (it is a power-user inspection strip; on-by-default would clutter every turn). | The trace is **read-only / answer-identical / moat-unchanged** (the endpoint flips `_composer.trace=True` post-construction; it only gates `last_trace` recording). Null for the rate composer / pre-scan abstention. Nothing to flip. |
| **Develop loop / capstone** | **Week-1 GPU-GO 1-seed** (`2026-06-24-week1-develop-loop-console-capstone.md`: vocab 6→24, facts 2→11, recall 1.0, retention 1.0, moat 0-FA 7/7, 8 per-day bundles). | A1 save-bundle (DONE), B2 console picker (DONE), surface bundles in `/api/brains` (DONE). | **DONE** for the demo + console picker; **A2 (6-seed scale-horizon) LEFT**; **A3 (persist `cp_connections` so resume LOADS not RE-HEARS) LEFT**. | **CLOSED** for the capstone demonstration + console picker. A2/A3 are **scale/persistence-depth follow-ons** (not default-flips). | The console `/api/brains` lists the 8 `week1/day_*` bundles (`webapp/server.py:3016+`). GROWTH/CONSOLIDATE are **owner-approved STAND-INS** (decision-only growth, self-replay not full-SWR). Resume RE-HEARS to re-derive codes (A3 = the true-persistence build). 1-seed GPU-GO → the 6-seed gate applies before a "works at scale" claim (A2). |
| **B1 rich answers** (console) | Wired into `/api/brain-chat` (`webapp/server.py:3008`, `RichAnswerComposer`); per-sentence VERIFY preserved. | Wire rich + "tell me more" into the Interact tab. | **DONE** (frontend `brainchat-rich` toggle + `tell me more` button, `app.js:2288/2407`). | **CLOSED.** | The console deliberately uses `neural_planner=False` (host planner) for **latency** — the neural dlPFC planner builds a per-turn GPU bridge (~75s timeout). The neural planner stays the `brain_chat_tui --rich` default (see 3G below). The moat is identical either way (the planner only steers WHICH facts). |
| **3G neural discourse-planner** | GO 3/3 (`_burndown_3G_neural_discourse_planner.json`). The prior audit's rank-1 close-out. | Wire `--neural-planner` onto the `--rich` runtime path. | **DONE in the TUI** (`brain_chat_tui.py:674` `--no-neural-planner`; default ON when on GPU). **Intentionally NOT in the webapp** (latency). | **CLOSED** (TUI). Webapp host-planner is the deliberate latency trade. | Quality-parity, not output-identity (neural latency-rank can pick a different-but-equally-valid associate) → kept a flag, not a silent flip. The library `RichAnswerComposer` default stays `neural_planner=False`. |
| **C-2 `integrated_loop`** (spiking K-way cue-match sequencer, shortcut #3) | GO at production scale; **already default-ON at the flagship 320 demo** (`consolidated_320_conversation_demo.py:242` `set_defaults(integrated_loop=True)`, `--composer onebrain` default `:221`). | Flip default-on where it works (320 demo) — DONE; keep OFF at library/small-vocab. | **DONE** at the flagship demo; **correctly OFF** at the library default. | **CLOSED** (the prod demo is the full-capacity surface). | **CHARACTERIZED divnorm code-margin BOUNDARY at small vocab** (V=15/K=4 → over-abstention, the SAFE direction, moat 0-FA). OFF at the library/numpy-CPU/test-oracle default by design. |
| **1A onebrain conversational default** (`enable_spiking_cleanup` + `enable_learned_assoc` + `--composer onebrain`) | Default-ON at the production path (`consolidated_320_conversation_demo.py:221` `--composer onebrain`; agent None-sentinel → spiking-cleanup + learned-assoc on for onebrain). | Flip prod demo to onebrain (DONE); keep library `composer_kind='rf'`. | **DONE** (flagship demo defaults onebrain; rf retained). | **CLOSED.** | Library `BrainConversationalAgent`/`MultiTurnAgent` keep `composer_kind='rf'` for **numpy-CPU portability + test-oracle** (flipping forces GPU on every default agent). Deliberate per CLAUDE.md. |
| **1B nav limbic core / spiking decision** | Episode-level defaults ALL spiking (`g11_bg_runner` `run_moving_goal_episode`: `spiking_wta`/`spiking_snc`/`neural_critic`/`spiking_reward_us` all True). | Flip episode/agent default-on (DONE); keep CLI default = host oracle for benchmark reproduction. | **DONE** (deployed path is fully-spiking; CLI defaults host = oracle). | **CLOSED.** | CLI `--readout-source=motor` etc. stay host-argmax ORACLE for documented standalone benchmark reproduction + measuring the ~16% finite-size decision cost (B-4). |
| **I-7 DA→composer encoding gain** (`enable_da_encoding_gain`) | GO mechanism; load-bearing on the onebrain/magnitude-store path (gate 4, g=1.423 exact). | Listed for completeness. | **LEFT default-OFF on the merged agent** (`:1619`). | **STAYS opt-in** (recommend). | **GREEN_INERT on a clean small store** + inert on the default `MergedRFComposer` (numpy-kb stores PHASES = magnitude-invariant). Becomes live only when Option A flips to onebrain (a magnitude store). So it flips **with** Option A, not before — at that point consider default-on alongside the onebrain flip. |

---

## (2) RANKED CLOSE-OUT LIST (cheapest + safest first)

**The honest top-line: the broad full-capacity surface is mostly ALREADY closed.** B1/B2/B3, 3G (TUI),
1A/1B, C-2, Route A, the capstone+console picker are all DONE. The ranked list below is the *genuinely
remaining* close-out work, cheapest/safest first.

1. **[GREEN-GATE-PENDING FLIP] Option A gate 3 → flip the merged composer default to `onebrain`.**
   The single clean "green-gate-then-flip" item. Gates 1+2 (15/15) + gate 4 (limbic, GPU) are GREEN;
   only gate 3 (nav Δ=0 + a GPU byte-identity for the onebrain opt-in) is unrun. Run it → flip
   `co_resident_composer_kind` default `'rf'`→`'onebrain'`, keep rf as the byte-identity oracle.

2. **[BYTE-IDENTICAL NO-OP — optional] persistent_loop default-on.**
   Byte-identical (`worst_membrane_maxabs=0.0`); already `True` on the onebrain merged path. Flipping
   the *library* default has **zero behavioral effect** (the recall is already substrate-resident), so
   this is the lowest-VALUE close-out. If flipped, add a `persistent_loop=True` parity assertion to
   `test_one_brain_composer_agent.py` (currently only the de-risk JSON pins it). Recommend: leave OFF
   (it lands where it matters via #1) — OR flip purely for "no-pending-flag" tidiness.

3. **[I-7 rides #1] DA→composer encoding gain default-on — ONLY after #1.**
   Inert on the rf numpy-kb (phase store); load-bearing only on the onebrain magnitude store. Once #1
   flips the merged default to onebrain, consider flipping `enable_da_encoding_gain` default-on (it's
   GO + lesion-confirmed + moat-HARD). Couples to #1; not independently flippable.

4. **[NEEDS-A-BUILD] Route B `gen_spikes` on the AGENT default + the `co_resident_perception`+onebrain seam.**
   The runner already defaults spikes-only; exposing it on the agent needs the gen stack + composer in
   the agent default. The genuine residual: `co_resident_perception` + `onebrain` raises a guard
   (host-`M` writes `composer.concepts`, the onebrain synaptic store doesn't consume it). Closing this
   = "perception→compose on the consolidated one brain" — a build, not a flip.

5. **[NEEDS-A-BUILD + OWNER-CALL] Wire the communicable-brain "what do you think" turn into the console.**
   Behind a flag. Depends on: the GPU spiking-Qwen faculty wire-in (drop-in, identical VERIFY) + the
   values-call (the learned-talkativeness arc, CYCLE 538-539 in flight) + the spiking-SWR sampler (to
   convert the host-sampled PROPOSE). A capability addition on the owner's TOP frontier, not a close-out.

6. **[SCALE/PERSISTENCE FOLLOW-ONS] Develop-loop A2 (6-seed horizon) + A3 (persist `cp_connections`).**
   A2 = the standing 6-seed gate before a "works at scale" claim. A3 = resume LOADS not RE-HEARS (the
   long-horizon wall-clock/fidelity fix). Neither is a default-flip.

---

## (3) TOP CLOSURES — EXACT CHANGE + GATE

### Closure 1 (rank 1) — Option A: flip the merged composer default to `onebrain`

- **Gate to run FIRST (the only outstanding gate):** nav-not-regressed **Δ=0** with the onebrain
  composer + a GPU byte-identity. Construct `MergedNavConvAgent(co_resident_composer=True,
  co_resident_composer_kind='onebrain')`, run the nav episode, assert the score is byte-identical to
  the rf-composer run (the composer is array-disjoint from the nav read-out, so Δ=0 is expected — the
  same null-gate logic as step-2b's `nav-not-regressed = 2.0 byte-identical`). This is the gate
  CYCLE 535/536 named as remaining ("the default-flip awaits gates 3+4"; gate 4 is done).
- **Exact change once GREEN:** `research/runners/nav_conv_merged_bridge.py:1612` —
  `MergedNavConvAgent(..., co_resident_composer_kind="rf", ...)` → default `"onebrain"`. (Keep `"rf"`
  accepted as the oracle; the validator at `:1672` already allows both.)
- **Confirm:** re-run `tests/test_nav_conv_merged_agent.py` (8) + `tests/test_nav_conv_step2b_coresident.py`
  (7) — these currently exercise the **rf** path; after the flip they exercise onebrain, so they
  become the onebrain CI guard (and a `co_resident_composer_kind='rf'` parametrization should be added
  to keep the oracle under test). Moat `is None` assertions must stay green.
- **Nuance:** flips only the **merged (GPU) agent** default. The standalone library
  `BrainConversationalAgent`/`MultiTurnAgent` keep `composer_kind='rf'` (numpy-CPU + oracle).

### Closure 2 (rank 2) — persistent_loop default-on (optional, byte-identical)

- **Exact change:** `research/runners/one_brain_composer.py:118` — `persistent_loop=False` → `True`.
- **Gate:** re-run the byte-identity de-risk (`python -m research.runners._persistent_loop_flat_derisk`,
  expect `worst_membrane_maxabs=0.0`, matrix identical, moat 0-FA) + `tests/test_one_brain_composer_agent.py`
  (answer-identical). Recommend ADDING a `persistent_loop=True` parity case to that test (it has none).
- **Nuance:** **byte-identical no-op** (the recall is already substrate-resident; the flag formalizes
  the clean re-kick). Lowest value; already on where it matters (`nav_conv_merged_bridge.py:1870`).
  Defensible to leave OFF as the library/numpy-oracle default.

### Closure 3 (rank 3) — DA→composer encoding gain default-on (ONLY after Closure 1)

- **Exact change:** `research/runners/nav_conv_merged_bridge.py:1619` — `enable_da_encoding_gain=False`
  → `True`. **Do NOT do this before Closure 1** (it is a no-op on the rf phase-store default).
- **Gate:** `_consolidation_probe2_limbic.json` re-run (GPU): g == `g_high/g_tonic` exact, lesion pins
  1.0, moat HARD no-breach; merged-agent CI 15/15.
- **Nuance:** GREEN_INERT at rest (engages only under read stress); a purity/robustness knob. Couple
  the flip to the onebrain flip so it is load-bearing.

---

## (4) HONEST VERDICT

**How much "pending close-out" is actually left: very little, and it is well-characterized.** The broad
full-capacity surface the owner's critique targets is **mostly already closed** since the prior audit +
the capstone-console scoping — B1 rich, B2 brain-picker, B3 activity-viz, 3G `--neural-planner` (TUI),
the 1A onebrain conversational default + 1B nav-spiking default, C-2 `integrated_loop` (flagship demo),
Route A default-ON, and the capstone+console picker are all DONE. The earlier audit's verdict
("essentially NO pending code work") **holds and extends** to these one-brain/console arcs.

**The genuinely-remaining items split cleanly:**

- **Exactly ONE cheap, green-gate-pending DEFAULT-FLIP: Option A gate 3 → flip to `onebrain`.** Gates
  1+2 + gate 4 are GREEN; run the single outstanding nav-Δ=0/byte-identity gate, then flip the merged
  composer default. This is the one item that matches the owner's "stop at validated-opt-in" critique
  *and* is closeable now. (Rank 1.)

- **ONE byte-identical no-op flip (persistent_loop, rank 2)** — optional tidiness; zero behavioral
  effect; already on where it matters.

- **ONE coupled flip (I-7 DA-gain, rank 3)** — rides Closure 1.

- **The rest are GENUINELY-DEFERRED (need a build and/or an owner call), NOT just-not-done-yet:**
  - **Route B perception→compose on the onebrain consolidated brain** — a real build (the
    `co_resident_perception`+onebrain guard / host-`M`→synaptic-store seam), explicitly flagged as a
    follow-on in the A+B finding.
  - **Communicable-brain turn → console** — a capability ADDITION on the owner's top frontier; gated on
    the spiking-Qwen wire-in + the learned-talkativeness values arc (in flight, CYCLE 538-539) + the
    spiking-SWR sampler (the CYCLE-537 honesty correction: the PROPOSE is currently host-sampled).
  - **Develop-loop A2 (6-seed) + A3 (`cp_connections` persistence)** — scale/persistence-depth, not flips.

- **Correctly-deferred BY DESIGN (not close-out):** the library `composer_kind='rf'` oracle + numpy-CPU
  path, the webapp host discourse-planner (latency), the C-2 small-vocab OFF, the CLI host oracles for
  benchmark reproduction, and the deep dendritic/FHRR frontiers (C-1/H-3, B-1, B-3).

**Net:** after running Option A's gate 3 and flipping the merged composer default to `onebrain` (+ the
two coupled/no-op flips), there is **no remaining "validated-and-should-be-default-on-but-isn't"** item.
What's left is honest capability work (Route-B-on-onebrain, the communicable-brain turn) and scale
gates (develop-loop A2/A3) — genuinely-deferred, owner-steered, not pending default-flips.
