# B3 — Per-turn "brain activity" visualization for the Interact console (SCOPING ONLY)

**Date:** 2026-06-24
**Status:** READ-ONLY scoping. NO edits, NO runs, NO webapp touch. The live webapp keeps serving at :8765.
**Goal:** when the owner sends a chat message to a brain (`POST /api/brain-chat`), ALSO surface what the brain
DID during that turn — its spiking activity (per-region firing, which concepts/roles fired, the resonate
dynamics) — so a chat turn shows the brain working, not just the text answer.

---

## 1. The chat path + the spiking state that actually exists during a turn

### 1.1 The request path (server side)

`POST /api/brain-chat` → `brain_chat()` (`webapp/server.py:3213`). Per turn:

1. `_pin_bridge_backend()` (re-asserts the cupy/numpy backend cache).
2. `cache_key = (session, brain, renderer)`; the warm `ChatBrain` is built once via `_build_chat_brain()`
   (`webapp/server.py:2917`) and session-cached in `_BRAIN_CHATS`.
3. **rich=False (default):** `gate_svo = chat.gate(msg)` → `chat.render(gate_svo)`.
4. **rich=True:** `_get_rich_composer()` (host discourse planner, `neural_planner=False` — deliberately, for
   webapp latency) → `rich.answer(msg)`.
5. Returns JSON: `answer, abstained, recalled_svo, verified, renderer, brain, source, rich`
   (+ rich-only: `n_sentences, supporting_facts, followup`).

**The endpoint returns NO spiking state today.** It returns only the text answer + the recalled SVO + verify/renderer flags.

### 1.2 Where the spiking actually happens

`ChatBrain.gate()` (`research/runners/brain_chat_tui.py:223`):
- `self.router.match_fact(...)` — a HOST string matcher picks a candidate stored fact (NOT spiking).
- `recalled = self.inner.what_does(a, v)` — **this is the spiking step.** `what_does` →
  `composer.query_patient(agent, action)` (`brain_conversational_agent.py:539`).

`ChatBrain.render()` (`brain_chat_tui.py:272`) — the renderer (stub / qwen / raw) produces fluent prose;
the **brain re-parse** for VERIFY (`self.inner.parse(...)`) is on-bridge for the parser-bearing agents.

The composer under the chat brain is normally an `RFPhasorComposer` (or `OneBrainComposer` for
`composer_kind="onebrain"`). The actual spiking work in `query_patient` / `query_agent` / `ask_yes_no`:

- `_scan_first_match(**cue_roles)` (`rf_phasor_composer.py:420`) → for each cue role, `_unbind_all_phases`
  (batched) → `_resonate(...)` → `_cleanup_all(...)` (matched-filter argmax over the codebook).
- `_resonate(n, conns, kick)` (`rf_phasor_composer.py:156`) is the spiking primitive. It:
  - reuses a **cached `SimulationBridge` per neuron-count** (`self._bridge_cache[n]`), built by
    `_build_rf_bridge` (RESONATE_AND_FIRE neurons);
  - `b.rf_set_complex_weights(conns)` — installs the complex (phasor) synapses;
  - `b.rf_kick(kick, period=self.period, lam=0.0)` — sets each neuron's complex state Z=re+i·im;
  - `b.rf_resonate_steps(self.period + 8)` — runs ~208 RF steps (the bulk of a query; ~83% per the profile);
  - returns `np.asarray(b.rf_read_phases())`.

### 1.3 What spiking state is on that bridge AFTER a resonate (cheap to read, already computed)

After `_resonate`, the cached bridge holds (all on `cp_*`, one element per RF neuron, n = e.g. 2·K·D):

| Array | What it is | Cheap read |
|---|---|---|
| `b.cp_rf_spike_step` (`bridge.py:5682,5754`) | int64, per-neuron first-spike step (= `period` if never crossed → |Z| decayed below the floor) | `rf_read_phases()` already converts → phase∈[0,1) |
| `b.cp_rf_fired` (`bridge.py:5680,5755`) | **bool, per-neuron — DID THIS READOUT NEURON SPIKE this resonate.** This is literally `cp_firing_states` for the RF substrate | `to_host(b.cp_rf_fired).mean()` = fraction of readout neurons that fired = a direct "activity" number |
| `b.cp_membrane_potential_v` / `cp_recovery_variable_u` | the final complex state (re=v, im=u) — magnitude `|Z|` per neuron | `(v*v+u*u)` = per-neuron magnitude (recovery strength) |
| `rf_read_phases()` (`bridge.py:5684`) | the recovered phase vector (D values) | already called and returned by `_resonate` |

And at the **composer/cognitive** level (already computed, returned, FREE):
- `_cleanup_all` returns the **decoded word per role** (the argmax winner) and computes `sims` = the full
  match-score vector over the codebook (`rf_phasor_composer.py:417`). The winning role-words + their match
  confidence are the most meaningful "which concepts/roles fired" signal — and they're computed regardless.
- `_scan_first_match` knows **which stored-fact block matched** (the index), i.e. which memory engram answered.
- The gate already produces `recalled_svo` (agent/action/patient) — surfaced today as text.

**KEY FINDING:** the cheapest, most owner-legible "what the brain did" signal is already computed every turn:
(a) the decoded role-words + their cleanup match-confidence (per role: agent/action/patient), (b) which stored
fact-block matched, (c) a scalar RF activity = the fraction of readout neurons that crossed (`cp_rf_fired.mean()`)
and the mean recovery magnitude `|Z|`. None of this needs an extra resonate; it is a READ of state the query
already produced (or one extra `.mean()` over an array that already exists).

**Caveat (honest):** the composer does NOT today RETURN any of this up to `query_patient`'s caller — it returns
only the decoded word (or None). So B3 needs a thin, read-only "trace" capture: either (i) a small instrumentation
hook that records `cp_rf_fired.mean()` / `|Z|` / `sims` during `_scan_first_match`, or (ii) the cheaper route —
have the endpoint re-derive a per-role activity summary from facts the gate already knows (see §3 option a). The
bridge state is transient (the cached `_resonate` bridge is overwritten by the very next op), so to read
`cp_rf_fired` you must capture it DURING the turn, not after.

---

## 2. Existing brain-viz in the webapp — can B3 reuse it?

Scanned `webapp/static/` (`app.js`, `brain3d.js`, `world.js`, `charts.js`, `index.html`).

### 2.1 The Interact tab (what it renders today)
`#tab-interact` (`index.html:80`) + `setupBrainChat()` (`app.js:2276`). Per turn `appendBrainTurn()`
(`app.js:2468`) renders: the role bubble (you / brain / abstain / error), the answer text, an abstain note
(the no-confab moat), and a hidden **meta strip** (`interact-meta`, toggled by "Show recalled fact"):
`recalled: <a v p> · verified ✓ · via <renderer>` (rich: `grounded in: …; all verified ✓`).
**No activity/firing/raster panel anywhere in the Interact tab.** It is pure text + the recalled-fact strip.

### 2.2 The Brain tab (`brain3d.js`) — 3D scene, firing rates
`brain3d.js` is a Three.js scene of regions + pathways. It renders **per-region firing rates** and animated
spike pulses — BUT only from **run JSON** (a recorded navigation/training run: `/api/runs/{name}`,
`liveActivityHistory`, per-step `regions` firing-rate samples, scrubbed by a step slider; `world.js:86`
`liveActivityHistory`, `brain3d`'s live picker polls `/api/inflight`). It is built for a long multi-step
run with recorded per-region rates — it has **no concept of a chat turn** and **no data source from
`/api/brain-chat`.** The chat composer never records a run JSON and its bridges aren't navigation regions.

### 2.3 The World tab (`world.js`) — 2D gridworld + retina
Navigation-only (the gridworld + the V1 retina image). Irrelevant to a conversational turn.

### 2.4 charts.js
A generic small-chart helper (sparklines/bars) used by the run views. **Reusable** as a lightweight
render primitive for a B3 bar/heat strip (no new chart lib needed).

**FINDING:** B3 is **net-new for the Interact tab.** The existing 3D brain-viz (`brain3d.js`) is coupled to
recorded RUN JSON with per-region rates and step scrubbing — it cannot be fed a single chat turn without
substantial re-plumbing (a fake run, region mapping, recorded rates). The cheap path is a small, self-contained
activity strip rendered inline in the chat turn, reusing `el()` + (optionally) `charts.js` bar primitives.

---

## 3. Cheapest B3 options (ranked cheapest-first)

All options are READ-ONLY of spiking/cognitive state already produced by the turn (see §4). None changes the
answer or the moat. Default-cheap unless noted.

### Option A — "what fired" summary in the chat turn (RECOMMENDED) — cheapest
**Idea:** the endpoint returns a small `activity` object describing what the brain DID this turn, derived from
state the gate/query ALREADY computed; the Interact tab renders it as a compact per-role chip strip + an
"engram matched" line under the answer bubble.

**Concretely returned per turn (`activity`):**
- `roles`: `[{role:"agent", word:"dog", confidence:0.97}, {role:"action", word:"chase", …}, {role:"patient", word:"rabbit", …}]`
  — the decoded role-words + their cleanup match score. (Confidence = the normalized top cleanup `sims`
  value; for the single-fact path the three roles are exactly `recalled_svo` with their match scores.)
- `matched_fact_index` / `n_facts_scanned` — which stored engram answered + how many were scanned (the no-confab
  scan made visible; abstain → `matched_fact_index: null`).
- `rf`: `{n_readout_neurons, frac_fired, mean_magnitude}` — the scalar RF activity:
  `frac_fired = cp_rf_fired.mean()`, `mean_magnitude = mean(sqrt(v²+u²))` over the last resonate. (One `.mean()`
  each over arrays the resonate already produced.)

**Server changes (`webapp/server.py` only):**
- The cheapest sub-variant (NO composer hook): in `brain_chat()`, after `gate_svo` is known, build `activity.roles`
  from `recalled_svo` and recompute each role's cleanup score by calling the composer's existing
  `_cleanup_all`/`unbind` once on the matched block — but that re-resonates, so it is NOT default-cheap. **Prefer
  the hook sub-variant below.**
- The default-cheap sub-variant (a thin trace hook): add a `composer.last_trace` dict that the composer fills
  during `_scan_first_match` / `query_patient` with the already-computed `{role: (word, score)}`, the matched
  index, `n` scanned, and the post-resonate `frac_fired` + `mean_magnitude` read from the cached bridge
  (`cp_rf_fired`, `cp_membrane_potential_v/u`). The endpoint reads `chat.inner.composer.last_trace` after the
  gate and attaches it as `activity`. **This is the one place that needs a small brain-side instrumentation
  hook** — it lives in `research/runners/rf_phasor_composer.py` (and/or `one_brain_composer.py`), NOT in `sim/`.

**Brain instrumentation needed:** a read-only `last_trace` populated from values `_scan_first_match` /
`_cleanup_all` already compute (`sims`, the decoded words, the matched index) + two `.mean()` reads of the
cached `_resonate` bridge's `cp_rf_fired` and `|Z|`. Gated behind a default-False `trace=True` ctor flag so the
test-oracle / numpy-CPU paths stay byte-identical. **Does NOT touch `sim/`** (the bridge already exposes
`cp_rf_fired`, `cp_membrane_potential_v`, `cp_recovery_variable_u`, `rf_read_phases` as public-ish attrs).

**Client changes (`webapp/static/app.js` + `style.css` + `index.html`):** in `appendBrainTurn()`, when
`data.activity` is present, render a small strip below the bubble: three role chips
(`agent: dog ·97%`, `action: chase ·…`, `patient: rabbit ·…`) colored by confidence, an
`engram #2 of 5 matched` line (or `∅ scanned 5, none matched` on abstain), and a tiny two-bar gauge
(`RF fired 0.42 · |Z| 0.88`). A toggle ("Show brain activity", next to "Show recalled fact") gates it.

**Cost:** ~free per turn (reads of already-computed state + 2 `.mean()`s). No extra GPU resonate.
**`sim/` touch:** NONE.

### Option B — per-readout-neuron raster / phase strip — moderate
**Idea:** in addition to the scalars, return the per-neuron arrays so the client draws a real raster: the
`cp_rf_fired` boolean (a dot per readout neuron that crossed) and/or the recovered phase vector (`rf_read_phases`,
D values) as a phase wheel/heat row. This is the most literal "the brain spiking" view.

**Server:** same `last_trace` hook as A but it also stashes `to_host(cp_rf_fired)` (bool, n≈2·K·D) and the
phase vector. For D=128, K up to 32 → n up to ~8192 booleans (≈8 KB packed) per turn — fine over HTTP, but it
is a per-turn array payload, so gate it behind the activity toggle (don't send unless the panel is open).
**Client:** a small canvas (reuse the world/brain3d canvas idiom, or a tiny dedicated one) drawing the raster
+ a phase-color row. More render code than A.

**Cost:** still one extra `to_host` of an existing array (the resonate already produced it); a few KB/turn.
**`sim/` touch:** NONE.

### Option C — reuse `brain3d.js` (the 3D scene) — most expensive, NOT recommended
**Idea:** drive the 3D brain scene from a chat turn — light up "regions" by per-region rates for the turn.
**Why it is expensive:** `brain3d.js` consumes recorded RUN JSON with per-step per-region firing rates +
step scrubbing + a region/pathway layout (`brain3d_layout.json`). The chat composer has no navigation regions,
records no run, and a single turn isn't a multi-step trajectory. Wiring it would mean: synthesize a fake
single-frame "run", map the parser/RF/cleanup slices onto named scene regions, record per-slice rates, and
teach `brain3d` a non-run data source. That is a large build for a per-turn glance and duplicates A/B's signal
without adding owner-legible value (a phasor resonate is not a 3D region cascade).
**`sim/` touch:** NONE, but high webapp cost + low fit.

---

## 4. Anti-cheat / moat confirmation

All three options are **strictly READ-ONLY of state the turn already produces**:
- The decoded role-words + cleanup `sims` are computed by `_cleanup_all` whether or not we read them; reading
  them changes nothing.
- `cp_rf_fired`, `cp_membrane_potential_v/u`, `rf_read_phases()` are the resonate's OUTPUT state; a `.mean()` /
  `to_host` is observational.
- The **no-confab moat is untouched:** abstention (`gate()` → None, `query_patient` → None, `_scan_first_match`
  → None) happens on the SAME code path; the trace only RECORDS that an abstain occurred (`matched_fact_index:
  null`), it never supplies a fallback answer. In fact B3 makes the moat MORE visible (it shows "scanned N, none
  matched" instead of a silent "I don't know").
- The instrumentation is gated behind a default-False `trace` flag, so the CI test oracle + numpy-CPU paths are
  byte-identical (no behavior change for any non-webapp caller).
- No extra GPU work per turn in the recommended option (A) → no contention with a concurrent build (the webapp's
  standing GPU-light policy is preserved).

---

## 5. RECOMMENDATION

**Build Option A (the "what fired" summary), default-cheap sub-variant (the `last_trace` hook).** It is the
cheapest, needs no extra resonate, surfaces the most owner-legible signal (the three decoded role-words + their
match confidence + which engram answered + a scalar RF-fired/|Z| gauge), and makes the no-confab moat visibly
fire on abstain. Leave Option B (the per-neuron raster/phase strip) as a fast follow-on behind the same toggle
(it reuses the same hook, just stashes one more array). **Do NOT** pursue Option C (reusing `brain3d.js`) — wrong
shape and a large build for a per-turn glance.

### Exact files / changes Option A needs
1. **`research/runners/rf_phasor_composer.py`** (and mirror in `one_brain_composer.py` for the onebrain default):
   add a default-False `trace` ctor flag + a `self.last_trace` dict; in `_scan_first_match` / `query_patient`
   (and `query_agent`/`ask_yes_no`) record `{role: (decoded_word, top_sim_score)}` from the `sims` already
   computed in `_cleanup_all`, the matched block index, `n` facts scanned, and (after the last `_resonate`) the
   cached bridge's `frac_fired = float(to_host(b.cp_rf_fired).mean())` + `mean_magnitude = float(mean(sqrt(v²+u²)))`.
   Read-only; reuse-by-import; **NO `sim/` edit.**
2. **`webapp/server.py`** — `brain_chat()`: build the warm `ChatBrain`'s composer with `trace=True`; after the
   gate, read `chat.inner.composer.last_trace` and attach it as `activity` in the JSON (both rich + single-fact
   paths; rich aggregates over its supporting facts). Add the field to the docstring's return schema.
3. **`webapp/static/index.html`** — add a "Show brain activity" toggle next to `#brainchat-show-svo`.
4. **`webapp/static/app.js`** — `appendBrainTurn()`: when `data.activity` is present and the toggle is on,
   render the role-chip strip + the `engram #i of N` line + the two-bar RF gauge (reuse `el()`; optionally
   `charts.js` for the gauge).
5. **`webapp/static/style.css`** — classes for the activity strip (chips colored by confidence, the gauge).

**`sim/` flag:** **NONE.** Option A (and B) read only already-public bridge attributes (`cp_rf_fired`,
`cp_membrane_potential_v`, `cp_recovery_variable_u`, `rf_read_phases`); the only new code is a default-off
read-only trace dict in the runner-level composers + the webapp wiring. Confirm before building that
`OneBrainComposer` (the production `--composer onebrain` default at V=320) exposes the same `_scan_first_match`
trace points — it does (it subclasses the same scan/cleanup machinery, `one_brain_composer.py`), so the hook
goes in both composers.
