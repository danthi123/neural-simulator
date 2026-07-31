---
type: plan
status: live
date: 2026-06-22
---

# Production-wiring execution plan — the shortcut-closure arc's final step

> **Status: ready-to-execute spec, assembled under the GO assumption (2026-06-22).** Every in-flight
> confirmation is assumed to land GO: #6 log-polar 6-seed (seed 42 GO + 4/4 generalizing at write time),
> #5b determinism 3/3 (seed 42 GO + 43/44 confirming), #3-fold `integrated_loop` at 320-scale (already GO
> 4/4 gates). The edits below **cannot run until the in-flight runs finish and free their files** — editing
> `g11_bg_runner.py` mid-#6-6-seed would change the later seeds' config and corrupt the confirmation. This
> doc is the complete spec so the wiring pass runs fast on confirmation with **no re-derivation**.
>
> Supersedes the structure of `docs/plans/2026-06-21-default-on-flip-pass-plan.md` (the earlier flip-pass
> plan; its chunk shape is carried here). Sourced from the definitive inventory
> (`research/findings/2026-06-21-shortcut-inventory-definitive.md`, `ddc3b8db`) — ⛔ SUPERSEDED by `research/findings/2026-06-23-cheats-shortcuts-integration-inventory.md` (the 4-dimension inventory; it says so in its own header), which is ⛔ itself superseded (its 14-item burndown COMPLETED `2f260f15`; the live ledger is `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §7) — plus the overnight closure
> findings (cited per-row below).

---

## 0. Scope, principle, and what this is NOT

**Goal.** Flip the validated-but-opt-in spiking defaults so the production "one brain" runs **fully spiking
end-to-end by default**, on both halves (conversation + navigation), compatible with future neuromorphic
hardware. The two-criteria bar from the inventory — runtime-spiking YES **and** on-substrate/dev-random
structure **and default-on** — is met for each flipped op only once it is the default path.

**Why a flip pass, not invention.** Every item here is a *validated* spiking/on-substrate mechanism that
ships **default-OFF**. The gap to "fully spiking by default" is overwhelmingly configuration (flip a flag,
wire a builder kwarg, plumb a demo), not new mechanisms. This is the cheapest, lowest-risk path to the
owner's bar, gated entirely on the no-confab moat plus non-regression, every step revertible.

**This is NOT:**
- Not a moat trade. The no-confab moat (0 false-accepts) is **never** weakened by any flip here. (Owner
  memory `feedback_moat_not_hard_lossy_memory_ok` softens the moat from a HARD STOP to a "keep-where-free"
  — but this pass does not touch it: every flip is array-disjoint from or answer-identical-through the moat.)
- Not the deep frontier. The exact-inverse FHRR bind *algebra* → learned cortex (Tier-3, owner-sequenced
  last) is out of scope. This pass closes the *structure* residual (host `np.conj` wiring) and the
  *default-on* gap, not the binding-form idealization.
- Not new capability. Consolidation/default-flips of existing capabilities; no behavior the validated
  paths did not already demonstrate.

---

## 1. The full flip / wiring table

Every item: flag, file, current default → target default, the one-line rationale, the owning in-flight
front (so the ordering is explicit — one writer per file). "Owner front" = the in-flight run whose file
this edit touches; the edit waits until that front lands and frees the file.

### 1.1 Composer chunk (`research/runners/one_brain_composer.py`, `research/runners/rf_phasor_composer.py`) — behind the #3-fold finalization

| # | flag | file:loc (verified) | current → target | rationale | owner front |
|---|---|---|---|---|---|
| A4/A5 | `local_reciprocal_unbind` | `one_brain_composer.py:112` (`__init__` default `False`); propagated to inner `RFPhasorComposer` at `:200` | `False` → **`True`** | The unbind (A4) + cleanup-codebook (A5) conjugate STRUCTURE: derive the unbind/cleanup synapse from the bind synapse by a local quadrature-flip wiring rule (byte-identical, `_local_conj == np.conj` bit-for-bit for a unit phasor). Removes the host `np.conj` from the default path → host-free bind+cleanup structure (the neuromorphic-port close). | #3 fold (owns `one_brain_composer.py`) |
| A6 | `enable_spiking_cleanup` | `one_brain_composer.py:111` (`__init__` default `False`); agent passes through at `brain_conversational_agent.py:153/198/215` | `False` → **`True`** (library default) | The cleanup SELECTION (winner-pick): spiking Izhikevich/NEF WTA instead of host argmax (== argmax multi-seed @ D=2048). Already ON in the flagship 320 demo (`consolidated_320_conversation_demo.py` gates it on `composer_kind=="onebrain"`); flip the **library constructor** default. | #3 fold |
| A8/A9/A12/A15 | `integrated_loop` | `one_brain_composer.py:112` (`__init__` default `False`); `:342/614/700/803` route sites | `False` → **`True`** | The cue-match SCAN + first-match routing + moat (the largest live conversational host residual): the spiking K-way sequencer replaces the host Python `==` first-match `_scan` on the `(agent,action)` hot-path (`query_patient`/`ask_yes_no`/`_find_cued_block` → reconsolidation/reason inherit). 320-scale GO 4/4 gates, moat 0-FA. | **#3 fold itself** — this IS the fold's default-flip; do AFTER the 320-scale confirmation lands |

**Composer notes (do NOT re-derive at execution):**
- The four composer flags are *independent* switches but share one file — sequence them in one chunk, one
  writer, TDD, commit each green step.
- `local_reciprocal_unbind` also lives on `rf_phasor_composer.py` (the rf reference / test oracle). The
  *production* default is `OneBrainComposer`; the rf composer's own flag stays as the test-oracle's switch
  (the rf path is the numpy-CPU + oracle). Flip the **`OneBrainComposer`** default; leave the rf
  constructor default as-is so the oracle comparison stays available (the rf composer is what the CI
  diffs against). *Decision recorded:* flipping the production default does not require flipping the rf
  oracle's default — they are compared, so one must stay the legacy reference.
- `integrated_loop`'s build-1 scope covers only the `(agent,action)` hot-path; `query_agent`
  (action,patient), `render_fact`/`describe` (agent-only), the general `_scan` stay host (documented
  bounded follow-ons, still abstaining via the oracle — an honest partial conversion, not a moat hole).
  This is acceptable for the flip: the production who/what + yes-no + reason hot paths go fully spiking.

### 1.2 Agent + demos chunk (`research/runners/brain_conversational_agent.py`, the two production demos, `multi_turn_agent.py`) — behind the #3-fold finalization

| # | flag | file:loc (verified) | current → target | rationale | owner front |
|---|---|---|---|---|---|
| A13 | `enable_learned_assoc` | `brain_conversational_agent.py:153` (`__init__` default `False`); ctor wiring `:229–233`; `_assoc_graph` reads it `:459–460` | `False` → **`True`** (agent default) | The dialogue-planning association GRAPH content: the substrate-learned sparse Hebbian CA3 recurrent (`LearnedAssocGraph`, 24/24 edges / 9/9 top associate, multi-seed) replaces the host Python co-occurrence dict. The spread/selection is *already* spiking; this closes the criterion-2 (structure) residual. | #3 fold (the agent + demos are edited in the same post-fold window) |
| A13-plumb1 | `enable_learned_assoc` (+ `local_reciprocal_unbind`, `integrated_loop`) | `consolidated_320_conversation_demo.py:105/126–129` (`run_seed` builds the agent) | not plumbed → **plumbed + on** for the onebrain path | The flagship demo constructs `BrainConversationalAgent` without the A13/A4-A5 flags → currently host dict + host conj on the flagship path. Plumb so the flagship uses the learned graph + local-conj structure + the spiking sequencer. `integrated_loop` is *already* a `--integrated-loop` CLI arg here (`:225/232`, default off) — flip the default for the onebrain path. | #3 fold |
| A13-plumb2 | `enable_learned_assoc` / `enable_spiking_cleanup` / `integrated_loop` / `local_reciprocal_unbind` | `multi_turn_agent.py:47–56` (`__init__`); inner-agent build `:56` | not exposed → **exposed + plumbed** to the inner agent | `MultiTurnAgent` passes only `composer_kind`/`enable_neural_render` to the inner agent; expose + forward the four flags so multi-turn dialogue inherits the spiking defaults. | #3 fold |

**Agent notes (do NOT re-derive):**
- `enable_neural_render=True`, `enable_attributed=True`, `enable_multiframe=True` are **already** the agent
  defaults (`brain_conversational_agent.py:153–154`) — the CYCLE-269..271 conversational flips are
  already default-on. **VERIFY** they are still on (a no-op check, part of the agent chunk gate); do NOT
  re-flip them.
- A13 honest cost: `store_fact` builds a separate ~1800-neuron GPU bridge once and runs ~450 sim steps per
  `hear()` call — real per-turn wall-clock the host dict does not pay, GPU-only. **Keep a `False` escape**
  for the numpy-CPU + test-oracle path (mirrors `enable_spiking_cleanup` / `local_reciprocal_unbind`).
- Per the A13 scoping (`2026-06-21-A13-dialogue-assoc-graph-scoping.md` §2 nuance): the flip moves the
  *content computation* onto the substrate (the real win) but a host dict still mediates the hand-off to
  the spread bridge. The fully-host-free single-bridge fold is a **deferred follow-on** (§5), NOT this
  pass.

### 1.3 Navigation chunk (`research/runners/g11_bg_runner.py`, `research/runners/nav_conv_merged_bridge.py`) — behind the #6 6-seed + the #5b runs

| # | flag | file:loc (verified) | current → target | rationale | owner front |
|---|---|---|---|---|---|
| #6 | `log_polar` (render) + `log_polar_retina` (episode) | `g11_bg_runner.py:184` (`render_egocentric_goal` `log_polar=False`); `:3708` (`run_moving_goal_episode` `log_polar_retina=False`, `log_polar_d0=1.0`) | `False` → **`True`** for the merged-nav path | The biology-faithful log-polar / foveal-magnified egocentric SC retina restores the bump for far goals the linear `ppc=4` map clipped off-image (retina mass 0.0 → 12.5); the spiking SC orienting read-out then tracks the moving goal on the diagonals. Default-on for the merged-nav path makes it the biology-faithful default. | #6 6-seed (owns `g11_bg_runner.py`) |
| #6-thread | `log_polar` threading into the merged bridge | `nav_conv_merged_bridge.py` (the merged nav episode does NOT yet thread `log_polar_retina`) | not threaded → **threaded + default-on** | The merged path calls the nav episode; thread `log_polar_retina=True`/`log_polar_d0` through `build_merged_nav_conv_bridge`/`MergedNavConvAgent`'s `run_moving_goal_episode` call so the merged-nav default gets the log-polar retina. | #6 6-seed |
| #5b-grid | the `grid_cells` → `place` production wiring | `g11_bg_runner.py` (`_n9_place_sensor_act` stub `:82`) + `nav_conv_merged_bridge.py` (the ~40-line helper `make_grid_code`, currently a probe monkeypatch in `_n5_grid_frontend_onbridge_probe.py`) | probe monkeypatch → **a real `grid_cells` region + `grid_cells→place` pathway** | The spatial-phase grid-cell metric (catalog D.07) gives the self-org `place` pool a decorrelated afferent → locally-selective fields → the graded plateau read-out grades the value 4.5–12.3× (R1 SURPASSED 3/3). The host-Gaussian `vs_place_context` scaffold RETIRES. The grid reads ONLY `(x,y)` self-position (structural anti-cheat). | #5b / #6 (owns `g11_bg_runner.py` + the merged builder) |
| #5b-det | `deterministic_transpose_matvec` keep-ON scope | `g11_bg_runner.py:5510` (captures `_saved_detmv`), `:5548` (restores OFF after STEP-1) | restores OFF after STEP-1 → **a `deterministic_read: bool` kwarg (default `False`) that holds it ON through the value-train + δ-read** | The deterministic-scatter SpMV (already shipped at the 5 critic-path matvec sites in `sim/bridge.py`, gated on `cfg.deterministic_transpose_matvec`, numerically allclose) held ON through the value-train + δ-read gives a seed-stable critic rate → the SNc-burst δ holds 3/3 under one config. The ~3–6 line **1b deploy scope** (default-OFF = byte-identical STEP-1-only). **NO `sim/` edit** (the branch ships). | #5b / #6 |
| B2/B3/B4 | `co_resident_nav_critic` (+ the limbic flips) | `nav_conv_merged_bridge.py:456` (`build_merged_nav_conv_bridge` `co_resident_nav_critic=False`); `MergedNavConvAgent` plumb | `False` → **`True`** (production merge default) | The nav limbic core (spiking reward B2 / value-critic B3 / dopamine-RPE B4): the fully-spiking δ=r−V (US→SNc reward burst − striosome_value GABA_B). Flipping makes it spiking-by-default. **GREEN_INERT caveat:** validated spiking but behaviorally inert on the orient-solvable gridworld — flipping is brain-based purity, not a behavior win. **Document, do not hide, the inertness.** | #6 / #5b |
| B4-oppoint | `td_stdp_w_max`, `n_train` (the merged TD cue-shift critic op-point) | the B4 cooled defaults (runner flags on `_merged_td_cueshift_consolidation_derisk.py`) | `td_stdp_w_max=60`, `n_train=30` → **`td_stdp_w_max=40`, `n_train=15`** | The op-point cooling that restores the strict Schultz signature r < −0.7 on 3/3 seeds co-resident (−0.787/−0.855/−0.850). Cool the merged critic (tighten the per-tap weight cap) + read over its convergence window. Runner-side only, **NO `sim/` edit**; the TD error stays 100% neural. | #6 / #5b |

**Navigation notes (do NOT re-derive):**
- The merged path (`nav_conv_merged_bridge.py`) wraps `run_moving_goal_episode`; the CLI
  `--readout-source` default stays `"motor"` (the host-argmax ORACLE) for benchmark-repro, but the
  **library** default is already `readout_source="spiking_wta"` (#4 default-on, CYCLE-219). Do NOT touch
  `--readout-source`; it is a legit benchmark override, not a cognitive shortcut.
- `#5b-grid` is the one item that is currently a *probe monkeypatch*, not a builder flag — its production
  form (a real `grid_cells` region + `grid_cells→place` pathway via the regions framework) is the only
  genuinely-new wiring here (still NO `sim/` edit; reuse-by-import of the regions framework). The
  `make_grid_code(x,y)` reference helper is in `_n5_grid_frontend_selectivity_smoke.py` (`reads ONLY
  (x,y)`). Promote it to a builder helper; thread `nav_critic_place_selforg=True` (already a builder
  kwarg, `:586`) and re-point the `place_sensors` stub at the grid code.
- B2/B3/B4 share the merged builder's mutually-exclusive `co_resident_*` family (asserts at
  `nav_conv_merged_bridge.py:529/539/555`). `co_resident_nav_critic` is the production limbic flip;
  `co_resident_td_cueshift` is the B4 cue-shift slice (a *separate* DA modulator, mutually exclusive). The
  production merge default is `co_resident_nav_critic=True` (the full nav critic). The B4 cue-shift op-point
  defaults apply when `co_resident_td_cueshift` is the active critic (the cue-shift de-risk path), not the
  default production merge — record this so the two are not conflated.

### 1.4 Items NOT flipped (deliberate — recorded so they are not "forgotten")

| op | why not flipped | status |
|---|---|---|
| `--readout-source` CLI default (`motor`) | a benchmark-repro ORACLE override, not a cognitive shortcut; the library default is already `spiking_wta` | CLOSED-op, default-on at library level |
| GPU-only flags (`enable_rf_cudagraph` megakernel, `composer_kind=onebrain` on the *library* constructor) | deliberately GPU-if-available-guarded to preserve numpy-CPU portability + the rf test-oracle | intentional opt-in |
| B5 (#6 SC orienting beyond log-polar), B6 (#5 place sparsify-to-default), the FHRR two-attribute bind | characterized BOUNDARIES — honest negatives that survived a surpass round; the host scaffold stays | research items, not flips |
| The composer's exact-inverse FHRR bind *algebra* → learned cortex | Tier-3 deep frontier, owner-sequenced last | out of scope |
| The rf composer's own `local_reciprocal_unbind` default | the rf composer is the test oracle the CI diffs against — its legacy default must stay as the reference | intentional reference |

---

## 2. File-scoped chunks + execution order

**The constraint:** concurrent edits to the same file are edit conflicts, not parallelism. One writer per
file. Each chunk runs only after its owning in-flight front lands and frees its file.

### Chunk C — Composer (`one_brain_composer.py`, `rf_phasor_composer.py`)
- **Gated on:** the #3-fold finalization landing (it owns `one_brain_composer.py`; the 320-scale GO 4/4 is
  the unblock). The `integrated_loop` default-flip **is** part of the fold's completion.
- **Edits:** flip `local_reciprocal_unbind` default `True` (A4/A5); flip `enable_spiking_cleanup` library
  default `True` (A6); flip `integrated_loop` default `True` (A8/A9/A12/A15).
- **TDD, commit each green step.** Narrow pathspec (`git add` only the composer files).
- **Gate:** `tests/test_one_brain_composer_agent.py` (11) verbatim + the 320 demo + moat + answer-identity
  (§3 per-flip gate).

### Chunk A — Agent + demos (`brain_conversational_agent.py`, `consolidated_320_conversation_demo.py`, `multi_turn_agent.py`)
- **Gated on:** the #3-fold finalization (same window as Chunk C; these files are not owned by the nav
  fronts). Run after / alongside Chunk C (different files → can parallelize against Chunk C if a second
  writer is available; otherwise sequential).
- **Edits:** flip `enable_learned_assoc` agent default `True` (A13); plumb A13 + A4/A5 + A6 + `integrated_loop`
  through the two demos; VERIFY `enable_neural_render`/`enable_attributed`/`enable_multiframe` still
  default-on (no-op check).
- **Keep a `False` numpy-CPU / test-oracle escape** for the learned-assoc path.
- **Gate:** `elaborate` parity vs the host-dict oracle + moat (`is None` abstentions verbatim) +
  lesion/no-learning collapse + the demos plumbed (§3 per-flip gate).

### Chunk N — Navigation (`g11_bg_runner.py`, `nav_conv_merged_bridge.py`)
- **Gated on:** the #6 6-seed run AND the #5b runs landing (they own `g11_bg_runner.py` + the merged
  builder; editing mid-run corrupts the later seeds). This is the LAST chunk to start.
- **Edits (in order):** #6 log-polar default-on (render + episode + merged thread); #5b grid front-end
  production wiring (promote the helper, re-point the stub, `nav_critic_place_selforg=True`); #5b
  determinism `deterministic_read` kwarg (hold the flag ON through value-train + δ-read); B2/B3/B4
  `co_resident_nav_critic=True` production-merge default (document the GREEN_INERT inertness); B4 op-point
  defaults (`td_stdp_w_max=40`, `n_train=15`) on the cue-shift path.
- **NO `sim/` edit** anywhere in Chunk N (every item is runner/builder-side, reuse-by-import — the
  deterministic branch + the graded plateau + the spiking critic all already ship).
- **Gate:** nav not regressed (the merged-nav score) + the conversational moat (array-disjoint from nav)
  unchanged + each flip validated (§3 per-flip gate).

### Order summary
1. As the #3 fold finalizes (`one_brain_composer.py` freed) → **Chunk C** then **Chunk A** (or in
   parallel if a second writer; different files).
2. As the #6 6-seed + the #5b runs land (`g11_bg_runner.py` + the merged builder freed) → **Chunk N**.
3. After all three chunks → the **combined-config moat validation** (§4).
4. Update the definitive inventory: move the flipped items to CLOSED-fully (both criteria, default-on).

---

## 3. The per-flip HARD gate (applies to EVERY flip)

No flip merges unless ALL five hold:

1. **No-confab moat preserved — 0 false-accepts (HARD, non-negotiable).** Do not flip anything that
   breaches it. For the conversational flips: the `is None` abstentions (`what_does`/`render_fact`/
   `reason_chain`/`describe`/`query_*`) byte-unchanged. For the nav flips: the moat is array-disjoint (the
   nav cascade is `cp_connections`/`cp_membrane_potential_v`/`cp_firing_states`; the composer's complex
   `cp_rf_w_*` synapses are a separate allocation) → preserved by construction, re-asserted by the merged
   tests.
2. **Parity vs the host oracle (or a documented, owner-acceptable delta).** The flipped path equals the
   host path on the validated matrix (e.g. `integrated_loop` == host `_scan` answer-identity; A4/A5
   `_local_conj == np.conj` bit-for-bit; A6 == argmax multi-seed; A13 `elaborate` == the host-dict oracle
   on the validated topics) — OR a documented delta (B4 strict r<−0.7 cooled to −0.79/−0.86/−0.85 vs
   standalone −0.80/−0.77/−0.89; the GREEN_INERT nav-inertness for B2/B3/B4).
3. **No regression.** The shipped CI suites + production demos pass **verbatim**:
   - Conversational: `tests/test_one_brain_composer_agent.py` (11), `tests/test_consolidated_320_conversation.py`,
     `tests/test_brain_conversational_agent.py`, `tests/test_onebrain_integrated_loop_fold.py` (12),
     `tests/test_rf_phasor_composer.py`.
   - Nav-merged: `tests/test_nav_conv_merged_agent.py` (8), `tests/test_nav_conv_step2b_coresident.py` (7),
     `tests/test_merged_rf_composer_coresident.py` (5).
4. **CPU-portability preserved.** GPU-only flags stay opt-in (the numpy-CPU path + the rf test-oracle
   intact). Every flip that needs GPU keeps a `False` escape (A6 `enable_spiking_cleanup`, A13
   `enable_learned_assoc`, `integrated_loop`, `co_resident_nav_critic`); the rf composer + numpy-CPU demos
   stay on the host/oracle path.
5. **Revertible.** Every flip is a default change with a default-OFF escape retained → flip back =
   byte-identical to today. (The flags themselves are unchanged; only the default value moves.)

---

## 4. The COMBINED-CONFIG moat validation (the gate-closer)

After all three chunks, run **all flips ON together** and confirm the moat holds + no regression under the
combined config (mirroring the CYCLE 269–271 conversational-capability consolidation + the B4 moat re-run).

### 4.1 The exact command set

```bash
# --- (a) Conversational CI suites, all flips ON (GPU) ---
SIM_BACKEND=cupy pytest tests/test_one_brain_composer_agent.py -v
SIM_BACKEND=cupy pytest tests/test_onebrain_integrated_loop_fold.py -v
SIM_BACKEND=cupy pytest tests/test_brain_conversational_agent.py -v
# rf oracle + 320 (CPU path stays on the oracle/numpy default; confirms portability):
SIM_BACKEND=numpy pytest tests/test_rf_phasor_composer.py -v
SIM_BACKEND=numpy pytest tests/test_consolidated_320_conversation.py -v

# --- (b) Nav-merged gates, all flips ON (GPU) ---
SIM_BACKEND=cupy pytest tests/test_nav_conv_merged_agent.py -v
SIM_BACKEND=cupy pytest tests/test_nav_conv_step2b_coresident.py -v
SIM_BACKEND=cupy pytest tests/test_merged_rf_composer_coresident.py -v

# --- (c) The conversational moat, explicit, combined config (the no-confab assertions) ---
#   onebrain + integrated_loop + local_reciprocal_unbind + enable_spiking_cleanup + enable_learned_assoc:
SIM_BACKEND=cupy python -m research.runners.consolidated_320_conversation_demo \
    --composer onebrain --integrated-loop --seed 42
#   assert in the demo output: recall 1.00, abstain 1.00 with 0 false-accepts,
#   what_does('dog','go')=='north' AND what_does('river','look') is None AND describe('river') is None.

# --- (d) The merged-nav score, all nav flips ON (GPU) ---
#   log-polar + grid front-end + deterministic-read + co_resident_nav_critic, grid-32:
SIM_BACKEND=cupy python -m research.runners.<merged-nav driver> \
    --seed 42 --grid-size 32 --n-steps 1800   # the merged episode with the production defaults
#   assert: nav score not regressed vs the documented merged-nav baseline (within the deploy bar);
#   the conversational moat (array-disjoint) byte-unchanged through the nav burst.
```

### 4.2 The PASS bar

- **Moat: 0 false-accepts** across (a) + (c) — every `is None`/"unknown" abstention holds; recall 1.00,
  abstain 1.00 at V=320. **HARD — a single false-accept fails the gate.**
- **No regression:** (a) + (b) pass verbatim (the shipped test counts: 11 + 12 + the agent suite + 8 + 7 +
  5, all green).
- **Answer-identity / documented-delta:** the combined conversational config is answer-identical to the
  host oracle on the who/what + yes-no + reason + abstention matrix (`integrated_loop`/A4/A5/A6 parity);
  A13 `elaborate` returns a true co-occurring associate (parity vs the host-dict oracle).
- **Nav not regressed:** (d) the merged-nav score within the deploy bar vs the documented baseline, with
  the GREEN_INERT limbic-core inertness documented (not hidden); the conversational moat byte-unchanged
  through the nav burst.
- **CPU portability:** the numpy-CPU oracle paths (rf composer, 320 demo on the rf/oracle default) pass —
  the flips did not force GPU on the portable path.

---

## 5. The arc-close checklist (cross-ref the AUTONOMOUS_STATE arc-close gate)

The arc is **NOT closed** until ALL of the following are GREEN. (This mirrors the explicit ARC-CLOSE GATE
recorded in `research/findings/AUTONOMOUS_STATE.md` CYCLE 378–384; every item must hold — no item left
deferred-forever, no flipped-partial declared closed, no boundary accepted as exit.)

### (1) The 5 hard residuals — genuinely closed at the load-bearing level
- [x] **#9** dendrite graded-value (δ=1.33, 6/6; overturns the old "characterized dendritic boundary").
- [x] **#4** fully-spiking motor read-out (default-on, 100% commit-burst; the +1.46 nav cost is the honest
      brain-based deliverable / characterized residual).
- [ ] **#6** SC orienting via log-polar — GO seed 42 + 4/4 generalizing → **6/6 on confirmation**.
- [x] **#5b R1** afferent-selectivity (the grid front end; V n/f 4.5–12.3×, 3/3).
- [x] **B4** TD cue-shift strict r<−0.7 (restored 3/3 co-resident, cooled op-point).

### (2) The 2 deferred polish items — genuinely handled (no permanent shelf)
- [x] **B4 r<−0.7** → 3/3 (deferred-item-2, DONE).
- [ ] **#5b determinism δ** → 3/3 under one config (deferred-item-1; seed 42 GO, **43/44 confirming →
      3/3 on confirmation**). The fix is the existing `deterministic_transpose_matvec` held ON; NO `sim/`
      edit.

### (3) The confirmations — multi-seed / production-scale
- [x] **#3 fold** `integrated_loop` 320-scale (V=320/K=32, GO 4/4 gates, moat 0-FA).
- [ ] **#6 6-seed** (4/4 at write time → **6/6 on confirmation**).

### (4) The production wiring + the default-on flip pass + the combined-config moat validation (THIS DOC)
- [ ] **Chunk C** (composer A4/A5/A6/`integrated_loop` default-on) — per-flip gate green.
- [ ] **Chunk A** (agent A13 default-on + the two demos plumbed) — per-flip gate green.
- [ ] **Chunk N** (nav: #6 log-polar default-on + #5b grid front-end wiring [`vs_place_context` retires] +
      #5b determinism keep-ON + B2/B3/B4 limbic default-on + B4 op-point defaults) — per-flip gate green,
      NO `sim/` edit.
- [ ] **The combined-config moat validation** (§4) — moat 0-FA + no-regression + nav-not-regressed +
      CPU-portability.
- [ ] **Inventory update** — move the flipped items to CLOSED-fully (both criteria, default-on) in
      `research/findings/2026-06-21-shortcut-inventory-definitive.md`. ⛔ SUPERSEDED by `research/findings/2026-06-23-cheats-shortcuts-integration-inventory.md` (the 4-dimension inventory; it says so in its own header). That 06-23 doc is ⛔ now superseded too — its 14-item burndown COMPLETED (`2f260f15`, 2026-06-24), so this checkbox is closed and the live ledger is `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §7.

**All four sections GREEN ⇒ the shortcut-closure arc is CLOSED.** The remaining residuals at that point
are the *characterized boundaries* (B5 SC orienting beyond log-polar / B6 place-sparsify-to-default — the
host scaffold stays, honest negatives) and the *Tier-3 deep frontier* (the exact-inverse FHRR bind algebra
→ learned cortex; the limbic→composer integration), both owner-sequenced as separate arcs, NOT blockers
for "fully spiking by default."

---

## 6. Why this is the right shape

Most of the spiking work is already done and validated; the gap to "fully spiking end-to-end by default"
is overwhelmingly configuration (flip + plumb + one grid-region wiring), not invention. That is the
cheapest, lowest-risk path to the owner's bar, gated entirely on the no-confab moat plus non-regression —
no new mechanisms, every step revertible, **zero `sim/` edits** across the whole pass (every item is
runner/builder-side reuse-by-import; the deterministic branch, the graded plateau, the spiking critic, the
local-conj rule, the K-way sequencer, the learned-assoc memory all already ship). The only genuinely-new
wiring is promoting the #5b grid-cell helper from a probe monkeypatch to a real `grid_cells` region — still
NO `sim/` edit, reuse of the regions framework.

---

## 7. Provenance — the overnight closure findings (one source per row)

- **#6 log-polar:** `research/findings/2026-06-22-shortcut6-log-polar-render-derisk.md` (the `log_polar`
  kwarg on `render_egocentric_goal` `g11_bg_runner.py:184`; the `log_polar_retina`/`log_polar_d0` episode
  kwargs `:3708`; GO seed 42, 4/4 generalizing, retina mass 0.0→12.5, SCRAM 25.8× collapse).
- **#5b grid front-end:** `research/findings/2026-06-22-shortcut5b-R1-grid-frontend-derisk.md` (R1
  SURPASSED 3/3, V n/f 4.5–12.3×; the `make_grid_code(x,y)` ~40-line helper; the
  `nav_critic_place_selforg` builder kwarg; the host-Gaussian `vs_place_context` retires).
- **#5b determinism:** `research/findings/2026-06-22-shortcut5b-determinism-deltabar-close.md` (the
  `deterministic_transpose_matvec` keep-ON; the 5 critic-path matvec sites in `sim/bridge.py` already gated;
  the `g11_bg_runner.py:5510/5548` STEP-1-only restore; the ~3–6 line 1b deploy `deterministic_read` kwarg).
  Scoping: `research/findings/2026-06-22-shortcut5b-deterministic-scatter-scoping.md` (`a48ad76f`).
- **B4 op-point:** `research/findings/2026-06-22-shortcut-B4-oppoint-r07-3of3.md` (the cooled defaults
  `td_stdp_w_max=40` + `n_train=15`; strict r<−0.7 restored 3/3 co-resident; runner-side, NO `sim/` edit).
- **#3-fold `integrated_loop`:** `research/findings/2026-06-21-shortcut3-fold-integrated-loop-BUILD.md`
  (the flag + `_seq_block`; 320-scale GO 4/4 gates; the `(agent,action)` hot-path sites; the bounded
  follow-ons). Capability: `research/findings/2026-06-21-shortcut3-K32-capability-surpass.md`.
- **A4/A5 conj structure:** `research/findings/2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md`
  (the unbind local-conj rule; `OneBrainComposer` 6 sites; `_local_conj == np.conj` bit-for-bit) +
  `research/findings/2026-06-20-FHRR-B-cleanup-codebook-local-conj.md` (the cleanup codebook 7 sites;
  total 0 `np.conj` on a full store+query build with the flag ON).
- **A6 spiking cleanup:** `research/findings/2026-06-05-composer-cleanup-NEF-GO.md` + burndown #1
  (`69fd355d`) — spiking WTA == argmax multi-seed @ D=2048.
- **A13 learned-assoc:** `research/findings/2026-06-21-A13-dialogue-assoc-graph-scoping.md` (the
  `LearnedAssocGraph`; the agent wiring `brain_conversational_agent.py:229–233`; the demo plumb sites;
  24/24 edges / 9/9 top associate; the §2 single-bridge-fold follow-on).
- **The flip-pass structure:** `docs/plans/2026-06-21-default-on-flip-pass-plan.md`.
- **The definitive inventory:** `research/findings/2026-06-21-shortcut-inventory-definitive.md` (`ddc3b8db`). ⛔ SUPERSEDED by `research/findings/2026-06-23-cheats-shortcuts-integration-inventory.md` (the 4-dimension inventory; it says so in its own header), which is ⛔ itself superseded (its 14-item burndown COMPLETED `2f260f15`; the live ledger is `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §7 + `research/findings/2026-07-24-accidental-deferral-audit.md`).
- **The arc-close gate:** `research/findings/AUTONOMOUS_STATE.md` CYCLE 378–384.
