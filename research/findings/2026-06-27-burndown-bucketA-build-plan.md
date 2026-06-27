# Burndown Bucket-A build-plan — the operation-conversions to fully-spiking-one-brain (2026-06-27)

**Type:** READ-ONLY build-plan scope (no code written, no `sim/`/composer edit). Refines the Bucket-A row of
`2026-06-27-conversation-depth-brain-based-audit-and-burndown.md` into a precise, sequenced, efficiency-judged
conversion plan, resolving the genuine-shortcut-vs-legitimate-oracle judgment per host/numpy path.

**Directive (the bar):** the END must be FULLY SPIKING on the ONE-BRAIN shared substrate (non-negotiable, owner,
memory `feedback_end_state_fully_spiking_one_brain_path_by_efficiency`); the PATH per-capability is an efficiency
call. **Bucket A = OPERATIONS** whose validated spiking form already exists (converging them is wiring + a
byte/parity check, not research). Bucket B (learned grammar / B2 fluid analogy) is the deep STRUCTURE frontier —
out of scope here except to mark the boundary.

**Terms (defined once):**
- *numpy-reference path* — a CPU computation that reproduces, bit-for-bit, what a validated spiking circuit computes
  (e.g. `RFPhasorComposer` with `enable_substrate_store=False` holds bound facts in a numpy array and cleans up via
  `np.argmax`, instead of holding them in complex synapses + reading them via resonate-and-fire firing).
- *spiking substrate path* — the same operation realized as neurons firing through synapses: the bridge's
  resonate-and-fire (RF) complex-synapse store (`enable_substrate_store=True`) and/or the `OneBrainComposer`'s
  persistent co-resident `SimulationBridge`.
- *test-oracle / CPU-portable path* — a deterministic reference kept on purpose so (a) CI can assert the spiking
  path matches a ground truth, and (b) the agent runs without a GPU (`SIM_BACKEND=numpy`). The project's standing
  pattern (memory `feedback_close_arcs_to_full_capacity`) is to KEEP this as the oracle even after the spiking
  default ships.

---

## 1. Per host/numpy path — the shortcut-vs-oracle judgment (VERIFIED in code this session)

The decisive question the owner posed: is each numpy path a *genuine shortcut to burn* (must run on spikes to
honor the end-state) or a *legitimate CPU-portable oracle to keep*? The answer is not uniform — it splits cleanly
on **whether a spiking production path that exercises the same capability already exists and is wired to a real
user surface.**

| Path (where it lives) | What it is today | Spiking form exists? | Verdict |
|---|---|---|---|
| **`RFPhasorComposer` numpy kb** (`enable_substrate_store=False`; the bind/bundle/unbind/cleanup of a fact held in a numpy array, cleaned via `np.argmax`) | numpy-reference | YES — `enable_substrate_store=True` (RF complex synapses) + `enable_spiking_cleanup=True` (Izhikevich WTA); both byte/parity-validated. AND the `OneBrainComposer` runs the whole pipeline on one persistent bridge. | **KEEP as oracle/CPU path** — but it must NOT be the only path a user surface runs. (see §1a) |
| **The first-chat CONSOLE composer** (`first_chat_console.py:401` builds `RFPhasorComposer(...)`, `:435` builds the agent with `enable_neural_render=False`) | numpy-reference, host word-order | YES (the substrate store, the onebrain path, the spiking serial-order renderer all exist) | **SHORTCUT on the user surface** — the flagship chat surface runs zero firing neurons. Converge a GPU/onebrain console path; keep numpy-console as the CPU oracle. (see §1a) |
| **`ArgStructureComposer.FrameCQ`** (`argstructure_composer.py:125`; the Tier-0.1 verb-frame word-ordering) | numpy rate-coded primacy + `max()` argmax | YES — `_phaseB_serial_order_spiking_derisk` (6/6 GO) drives concept pools with a graded primacy current on a real bridge and reads the rate ranking; packaged as `NeuralSerialOrderRenderer`. | **SHORTCUT to convert** — a numpy reimpl of an existing spiking mechanism, NOT an oracle (the spiking renderer is the reference). (see §2, conversion C1) |
| **`RFPhasorComposer` / `OneBrainComposer` `render_fact` host f-string** (`rf_phasor_composer.py:845`, `one_brain_composer.py:1056` — the `f"{agent} {ac} {pt}"` / `" ".join` when `order_fn is None`) | host literal ordering | YES — both already accept `order_fn=`; `BrainConversationalAgent(enable_neural_render=True)` passes `NeuralSerialOrderRenderer.order`. | **SHORTCUT, but the wire is already there** — the console just opted out (`enable_neural_render=False`). (see §2, conversion C2) |
| **`OneBrainComposer.render_fact` final `" ".join`** of already-neurally-ordered words | host string concat | n/a — this is the BODY emitting motor output | **KEEP** — legitimate per BRAIN-BASED-ONLY (host is allowed for the body's emission; only the cognitive *ordering* must be neural, and C1/C2 make it so). |
| **Tier-2.3 / B1 ordinal-map LEARNING** (`first_chat_console.py:704` `_build_ordinal_map`; `_regimeb_corpus_mined_axis_derisk` Betasort update) | host-side learning objective; the COMPARATOR is already a spiking Wang-2002 accumulator | partial — comparison spiking; the embedding objective host-side | **TRACKED follow-on, NOT Bucket A** — the comparator is already on spikes; the self-organizing embedding is a bounded but genuine on-bridge build (borders Bucket B). (see §3) |
| **Tier-2 regime-A KBs / tags** (analogy KB, size ladder, common-ground/tense tags; all `_build_*` in the console) | host-curated structure (GIVEN) | n/a — these are GIVEN structure, not operations | **NOT Bucket A** — these are Bucket-B *structure acquisition* (the regime-A→B frontier; B1 already showed the cheap mineable cases). (see §3) |

### 1a. The decisive verdict — the console numpy-composer (the question the owner asked)

**Resolution: KEEP the numpy composer as the test-oracle + CPU-portable path, AND build a spiking/onebrain
production path that runs the console + the new Tier-0/1/2 capabilities.** The two are not in tension — they are
two endpoints of the same pattern the project already uses for `consolidated_320_conversation_demo` (which
**defaults to `--composer onebrain`** and keeps `--composer rf` as "the test oracle + the numpy-CPU path", per
`one_brain_composer.py:19-26`).

Why this is the right call (with evidence):

1. **The numpy path is a genuine, load-bearing oracle.** `tests/test_one_brain_composer_agent.py`, `test_rf_phasor_composer.py`, and the rubric all assert the spiking answers *equal* a numpy ground truth; remove it and the spiking path loses its correctness anchor. And `SIM_BACKEND=numpy` is the only path on a GPU-less machine (memory `feedback_close_arcs_to_full_capacity` is explicit: keep rf/numpy as oracle + CPU portability). So "burn the numpy path entirely" would violate a standing directive AND delete the verification anchor.

2. **BUT the console is a *user surface*, and today that surface runs zero firing neurons.** `first_chat_console.py:401` builds a plain `RFPhasorComposer` (numpy kb), `:435` builds the agent with `enable_neural_render=False`, and the console has **no `--composer onebrain` option at all** (verified: the only composer branches are plain-RF / `RoutedComposer` / `ArgStructureComposer`). The flagship "first chat" the owner talks to is therefore the numpy reference end-to-end — which is exactly the gap the directive targets: the *end-state* user experience must be fully spiking.

3. **Therefore the burn is not "delete numpy" — it is "add a spiking default to the surface, demote numpy to the documented oracle/CPU fallback."** This mirrors the `consolidated_320` flip precisely (default onebrain on GPU; rf retained as oracle/CPU). The console gets a `--composer {rf,onebrain}` (or `--substrate`) switch; GPU default → spiking; `SIM_BACKEND=numpy` / `--composer rf` → the oracle. After that, the numpy path is no longer "the chat is fake" — it is "the documented oracle the spiking chat is checked against," which is a KEEP, not a shortcut.

**One honest caveat that shapes the sequence (verified):** the `OneBrainComposer` today exposes the *flat-SVO*
who/what surface (`store`/`hear`/`query_patient`/`query_agent`/`ask_yes_no`/`render_fact`/`query_chain`/
`chain_of_thought`) but **NOT** the Tier-0.1 typed-role argument-structure surface (`store_fact`/`query_role`/
`render` with the verb frame) — those live only on `ArgStructureComposer`, a numpy `RFPhasorComposer` subclass.
So "run the console on the spiking onebrain path" fully delivers the *flat who/what + chain-of-thought + yes/no +
generation* capabilities on spikes, but the *typed verb-frame* capabilities (Tier 0.1/0.3 "goes TO THE park",
wh-filler-gap) would either (a) keep running on the numpy `ArgStructureComposer` oracle for now, or (b) wait for
an `ArgStructureComposer`-on-substrate / onebrain-typed-roles follow-on. That split is the reason the sequence
below puts the *cheapest, already-wired* conversions first and flags the typed-role onebrain port as a larger
(but still Bucket-A-shaped) follow-on.

---

## 2. The Bucket-A conversion sequence (cheapest-first)

Each conversion lists: what it converts (numpy path → validated spiking form), whether the spiking form exists,
the de-risk (byte-equivalence / spiking-parity), the anti-cheats, and the rough cost.

### C1 — numpy `FrameCQ` → the spiking competitive-queuing renderer  ★ RECOMMENDED FIRST (see §4)

- **Converts:** `ArgStructureComposer.FrameCQ` (numpy rate-coded primacy gradient + `max()` argmax,
  `argstructure_composer.py:125-157`) → the **validated spiking** `NeuralSerialOrderRenderer`
  (`neural_serial_order_renderer.py`), which drives concept pools with a graded primacy *current* on a real
  `SimulationBridge` and reads the per-pool spike *rate* ranking (`_phaseB_serial_order_spiking_derisk`, 6/6 GO).
- **Spiking form exists?** YES — `_phaseB_serial_order_spiking_derisk` (6/6 GO) + the multi-frame extension
  (`_phaseB_serial_order_multiframe_derisk`, GO) prove a *frame-conditioned* primacy gradient learns distinct
  orders. `NeuralSerialOrderRenderer.order(frame_concepts)` is the packaged read-out.
- **The exact gap:** `FrameCQ.emit_order` orders frame-unit indices by a numpy primacy vector; the spiking renderer
  orders by a spiking rate ranking driven by the same primacy gradient. They compute the same function; only the
  substrate differs.
- **De-risk:** parity — for each verb frame in `FRAME_LEXICON` and each realized-unit subset, assert
  `spiking_emit_order == numpy_emit_order` for the canonical (full-frame) order at ≥6 seeds; on a real bridge
  (GPU), confirm the rate ranking reproduces the frame order (the spiking derisk already shows true≈1.000 vs
  permuted 0.333). Because FrameCQ's gradient is *deterministic* per frame (teacher-taught identity order), the
  spiking renderer should match it exactly on the full frame and consistently on subsets.
- **Anti-cheats (reuse the existing harness `song_g1_core`):** (i) **equal-drive control FAILS** — a flat primacy
  gradient must NOT reproduce the order (proves the neurons do the serialization, not pool bias); (ii)
  **permuted-order control** — emitted order must beat a permuted target by ≥10%; (iii) **cross-frame control**
  (multi-frame) — the SAME fact under a different frame must produce a DIFFERENT order (the order is
  frame-specific); (iv) **moat unchanged** — rendering is gated by a stored composite (an unstored fact → None),
  so a spelling/order change can never fabricate a fact.
- **Cost:** LOW. The spiking renderer is built and packaged; the work is (a) a thin adapter so
  `ArgStructureComposer.render` calls `NeuralSerialOrderRenderer.order` on the *frame-unit indices* instead of
  `FrameCQ.emit_order` (the indices map cleanly — both order a small set of positions), behind a default-off flag
  so the numpy path stays the oracle; (b) the parity test; (c) the GPU spiking-rate check. No `sim/` edit
  (reuse-by-import). **Caveat:** `NeuralSerialOrderRenderer` today is a single fixed-frame SVO renderer
  (`VOCAB`/`PRIMACY_pA` from the SVO derisk) holding ≤16 pools; the multi-frame primacy is validated in
  `_phaseB_serial_order_multiframe_derisk` but not yet packaged into the renderer class — so C1's first step may
  package the multi-frame gradient into the renderer (small, validated mechanism) before the adapter. That keeps
  C1 self-contained and is the reason it is the cleanest first build.

### C2 — the console's `render_fact` host f-string → the spiking serial-order renderer (flip `enable_neural_render`)

- **Converts:** the console's word-ordering. `first_chat_console.py:435` builds the agent with
  `enable_neural_render=False` → `render_fact`/`describe` use the host f-string (`rf_phasor_composer.py:845`). Flip
  to `enable_neural_render=True` so the agent passes `order_fn=lambda n: self._neural_render.order(...)` (the
  spiking `NeuralSerialOrderRenderer`) — exactly what `BrainConversationalAgent` does by default
  (`brain_conversational_agent.py:314-316, 603`).
- **Spiking form exists?** YES — the wire is already in place on both composers (`render_fact(order_fn=)`,
  `query_patient(order_fn=)`). The console is the only surface that opted out.
- **De-risk:** the default rubric (`--rubric`) must stay 10/10 with the moat 0-leak; the demo transcripts must be
  unchanged in *content* (only the ordering mechanism changes — and on SVO the neural order == SVO, so output is
  byte-identical for well-ordered facts).
- **Anti-cheats:** moat 0-FA (unchanged — gated by stored composite); the equal-drive failure is inherited from C1;
  spelling is the separately-validated A→W primitive, untouched.
- **Cost:** VERY LOW (a flag flip + a rubric regression run), but it depends on a GPU at console-run time (the
  renderer builds a bridge). So C2 is naturally paired with the onebrain console path (C3); on the numpy-CPU oracle
  path `enable_neural_render` stays False (the host f-string is the oracle's legitimate body-emission). **Note:**
  C2 makes the *flat-SVO* render neural; the *typed verb-frame* render (`ArgStructureComposer.render`) is C1's
  target — they are complementary, not redundant.

### C3 — the console composer → the spiking onebrain path (add `--composer onebrain`)

- **Converts:** the console's whole who/what pipeline from the numpy-reference `RFPhasorComposer` to the persistent
  spiking `OneBrainComposer` (parser + RF work registers + complex-synapse fact store + spiking cleanup on ONE
  co-resident `SimulationBridge`).
- **Spiking form exists?** YES — `OneBrainComposer` is the shipped production default for
  `consolidated_320_conversation_demo`; it supports `grounded_codes=` (so it runs on the brain's learned codes,
  the same injection the console uses), `enable_spiking_cleanup=True` (default), and the flat who/what + yes/no +
  `render_fact` + `query_chain` + `chain_of_thought` surface.
- **De-risk:** answer-parity — on the 1454/`brainALL` codes, the onebrain console must produce the SAME who/what
  answers + the SAME abstentions as the numpy console for the flat-SVO capability set; the rubric stays 10/10;
  moat 0-FA. (This is the exact bar `consolidated_320` already meets at 320: recall 1.00, abstain 1.00, 0
  false-accepts.)
- **Anti-cheats:** the no-confab moat is preserved by construction (the onebrain scan abstains on an unmatched
  cue); a co-residence/isolation check (the parser slice byte-isolated from the RF slice); the numpy oracle path
  must stay byte-identical (the default for `SIM_BACKEND=numpy` stays rf).
- **Cost:** MEDIUM. The mechanism is shipped; the work is console plumbing (a `--composer {rf,onebrain}` switch
  that builds `OneBrainComposer(grounded_codes=...)` on GPU, default onebrain when `SIM_BACKEND=cupy`), wiring the
  DiscursiveTurn/proposer/agent over it (they consume the composer through the same API — the onebrain composer is
  an `RFPhasorComposer` API-sibling), and the parity/rubric runs. **Honest scope (verified):** this delivers the
  *flat who/what + chain-of-thought + yes/no + generation* on spikes; the *typed verb-frame* Tier-0.1/0.3
  capabilities still run on the numpy `ArgStructureComposer` until C4. Latency: the onebrain path is already
  *faster* than rf for the flat pipeline (A5 levers, ~4.3× at small scale), so the console stays responsive — but
  the 1454/2012-concept scale should be a measured operating point in the de-risk (the latency arc, memory
  `feedback_prioritize_orchestration_overhead`).

### C4 — typed verb-frame (`ArgStructureComposer`) on the spiking substrate / onebrain  (LARGER Bucket-A follow-on)

- **Converts:** the Tier-0.1/0.3 typed-role surface (`store_fact`/`query_role`/the verb-frame `render`) from the
  numpy `ArgStructureComposer` to a spiking path — either (a) `ArgStructureComposer` with
  `enable_substrate_store=True` + `enable_spiking_cleanup=True` (it inherits both flags from `RFPhasorComposer`,
  so the bind/bundle/unbind/cleanup of typed-role facts can already run on the RF substrate — *unverified for the
  typed-role path*), or (b) a typed-role surface added to `OneBrainComposer`.
- **Spiking form exists?** PARTIAL — the *operations* (typed-role bind/unbind via the parent's RF `_bind`/`unbind`)
  are inherited and the parent's substrate store exists; what is unverified is whether the typed-role recall +
  the frame render hold at parity on the substrate, and `OneBrainComposer` has no typed-role API yet.
- **De-risk:** parity of `query_role`/`render` between `enable_substrate_store=True` and the numpy path at ≥6 seeds
  (the typed-role analogue of the existing flat-SVO substrate parity); moat 0-FA; the agrammatism ablation control
  (drop the closed-class scaffold → telegraphic, proving the scaffold does real work) preserved.
- **Anti-cheats:** the existing argstructure controls (`reparse_to_fact` VERIFY; the function-word ablation; moat).
- **Cost:** MEDIUM-HIGH (the largest Bucket-A item) — it is still "wire validated spiking ops," but the typed-role
  path on the substrate is unproven and may surface the bundle-SNR limits the Tier-2.5 D=64→128 lever already
  hit. Do it AFTER C1–C3 land the high-value flat path.

---

## 3. The boundary — what is NOT Bucket A (deferred / flagged)

These are STRUCTURE-acquisition or GIVEN-structure items, not operation-conversions; they do not belong in this
build-plan and are flagged so the boundary is explicit:

- **Bucket B — learned grammar (the deep frontier).** `FRAME_LEXICON` (verb→slots), `WH_ROLE_CANDIDATES`, the
  closed-class word lists — hand-authored host dicts. The learned version = the brain self-organizing grammar from
  the corpus (BPTT-SNN scale + dendritic frontier; `project_generative_sequence_frontier`,
  `feedback_spiking_structure_must_self_organize`). Months-scale; scaffold-then-learn is the efficient path and
  the scaffolds are tracked. **Out of scope.**
- **Bucket B2 — fluid analogy on raw learned codes.** Regime-B analogy over the brain's similarity geometry
  (king−man≈queen−woman) is the documented NEGATIVE (`2026-06-27-tier2.1-analogy-NEGATIVE.md`) — emerges at LM
  scale, gated on the deep-knowledge corpus, not a missing circuit. **Out of scope.**
- **Regime-A Tier-2 KBs/tags (analogy KB, size ladder, common-ground/tense tags).** GIVEN structure, not
  operations. The *mining* of such structure from the corpus is the regime-A→B converter — B1 already showed the
  ordinal case is cheaply mineable (`2026-06-27-regimeB-corpus-mined-axis-GO.md`). That is a *structure-acquisition*
  follow-on (Bucket-B-adjacent), not an operation-conversion. **Out of scope here.**
- **The Tier-2.3 / B1 ordinal-map *embedding* objective (host-side Betasort).** The comparator is ALREADY a real
  Wang-2002 spiking accumulator; only the position-learning objective runs host-side. Wiring it into a
  rate-Hebbian population-code bridge so the map self-organizes in synapses is a bounded but genuine on-bridge
  build that borders Bucket B (it is *learning structure*, not converting a fixed operation). **Flagged, not in
  the Bucket-A sequence.**

---

## 4. Recommendation — build C1 first (numpy `FrameCQ` → spiking CQ renderer)

**Why C1 is the cheapest, highest-value, cleanest-de-risk first conversion:**

- **Cleanest de-risk:** it is a pure parity conversion of one self-contained mechanism (a small primacy gradient +
  argmax → a spiking rate ranking), and the spiking form is *already validated 6/6* with a pre-registered
  anti-cheat harness (`song_g1_core`) and a packaged class (`NeuralSerialOrderRenderer`). The byte/parity test is
  small and deterministic.
- **Highest value relative to cost:** word ordering is a *cognitive* operation (the parallel→serial conversion the
  owner explicitly cares about — "stilted sentence-gen" is a noted gap). Converting it puts the seed-of-syntax
  serialization on spikes for both the typed-frame path (C1) and, paired with C2, the flat path.
- **Unblocks C2 immediately:** once the spiking renderer is the ordering source, flipping the console's
  `enable_neural_render` (C2) is a one-line follow-on with a rubric regression — so C1 is the keystone for the
  cheap C1→C2 pair before the larger C3 console-onebrain plumbing.
- **No `sim/` edit, no composer-internals edit:** reuse-by-import + a default-off flag, consistent with the whole
  Tier-0/1/2 discipline.

**C1 de-risk + anti-cheat spec (so the build can start from here):**

1. **Package the multi-frame primacy into the renderer (if needed).** `NeuralSerialOrderRenderer` currently holds
   the single SVO frame; lift the validated frame-conditioned primacy gradient from
   `_phaseB_serial_order_multiframe_derisk.FrameCQ` into a `frame_id`-indexed primacy so the renderer can order any
   verb frame's realized units. (Validated mechanism; packaging only.)
2. **Adapter (default-off flag).** Add an opt-in path in `ArgStructureComposer.render` (e.g.
   `use_spiking_cq=True`) that, instead of `self.frame_cq.emit_order(fid, realized_idx)`, calls the spiking
   renderer's `order` on the same realized-unit indices. Default stays the numpy `FrameCQ` (the oracle).
3. **Parity test (CI, ≥6 seeds):** for every verb in `FRAME_LEXICON` and every realized-unit subset, assert
   `spiking_order == numpy_order` for the canonical full-frame order, and that subset orders are
   frame-consistent. (Deterministic full-frame match expected; small RUN_STEPS/primacy-gap tuning is the only
   risk, already characterized in the derisk.)
4. **Spiking-rate check (GPU, ≥6 seeds):** on a real bridge, confirm the per-pool rate ranking reproduces the
   frame order (reuse the derisk's `pool_rates`); record the rate gap.
5. **Anti-cheats (MANDATORY, reuse `song_g1_core`):** (a) **equal-drive control FAILS** (flat primacy → order not
   reliably the frame order — proves neurons serialize, not pool bias); (b) **permuted-order** beaten by ≥10%;
   (c) **cross-frame control** (the same fact under a different frame → a different order); (d) **moat 0-FA**
   (render gated by a stored composite; an unstored fact → None — a spelling/order change can never fabricate a
   fact); (e) **agrammatism ablation still works** (dropping the closed-class scaffold → telegraphic output, so
   the scaffold's role is unchanged by the ordering swap).
6. **Falsification bar:** if the spiking order does not match the numpy order at ≥5/6 seeds on full frames, OR the
   equal-drive control does NOT fail, C1 is NOT a clean conversion — localize (primacy gap / RUN_STEPS) before
   wiring; do not ship a non-parity ordering into the user surface.

**Then:** C2 (flip the console's `enable_neural_render` on the GPU path, rubric stays 10/10) → C3 (add
`--composer onebrain` to the console, answer-parity + rubric + moat, default onebrain on GPU / rf-numpy oracle on
CPU) → C4 (typed verb-frame on the substrate, the larger follow-on).

---

## 5. Bottom line

- **Console numpy-composer verdict:** KEEP it as the test-oracle + CPU-portable path (it is load-bearing for CI
  parity and GPU-less runs, per standing directive), AND build a spiking/onebrain *production* path for the
  console + the new capabilities — exactly the `consolidated_320` pattern (default onebrain on GPU; rf-numpy as
  the documented oracle). The shortcut is "the user surface runs zero neurons," not "numpy exists"; the burn is
  "add a spiking default + demote numpy to oracle," not "delete numpy."
- **Bucket-A sequence (cheapest-first):** **C1** numpy `FrameCQ` → spiking CQ renderer → **C2** flip the console's
  `enable_neural_render` (host f-string → spiking serial order) → **C3** add `--composer onebrain` to the console
  (flat who/what + chain-of-thought + yes/no + generation on the persistent spiking bridge) → **C4** typed
  verb-frame (`ArgStructureComposer`) on the substrate/onebrain (the larger follow-on).
- **Deferred (flagged, not Bucket A):** Bucket B learned grammar; B2 fluid analogy; regime-A KBs/tags + their
  corpus-mining; the ordinal-map self-organizing embedding (comparator already spiking).
- **Recommended FIRST build:** **C1** — the cleanest parity de-risk against an already-6/6-validated spiking
  mechanism, the keystone that unblocks C2, no `sim/`/composer-internals edit, with the exact de-risk +
  anti-cheat spec in §4.

All conversions are reuse-by-import; the no-confab moat is preserved throughout; the numpy/rf path is retained as
the test-oracle + CPU-portable path. NO `sim/` edit is contemplated in Bucket A (C4's substrate-store path is an
existing opt-in flag, not a new edit).
