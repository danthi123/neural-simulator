# One-Bridge Unification — Step 2 (gated synaptic parser→composer route) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the last Python hand-off in the conversational loop — the parser returning roles as a dict that
Python passes to the composer — with a **synaptic route on the merged bridge**: the parser's role-ensemble firing
*opens a transmission gate* that routes a word's concept code into the composer's bind bank for that role. Then
comprehension routes composition *in spikes*.

**Architecture (design doc §4):** On the unified bridge, add a "word-code" input pool and, for each role
(agent/action/patient), a gated pathway `word_code_pool → composer fill bank[role]` held closed by a
`transmission_gate` and opened by the parser's role-ensemble[role] firing (`couple_gate_to_pool`, the shipped +
validated thalamocortical-gating primitive). Per word: drive the parser conjunction (position × voice) + drive the
word's code into `word_code_pool`; the parser selects the role → its gate opens → the code lands in that role's fill
bank. After the 3 words, the composer binds the SVO fact. **The cross-region hand-off is then synaptic, not Python.**

**Tech Stack:** Python, `sim.bridge.SimulationBridge` (`inject_explicit_wiring`, `set_transmission_gate` /
`couple_gate_to_pool` — already shipped + validated, see CLAUDE.md "transmission_gate" + `tests/test_transmission_gate.py`),
CuPy/NumPy backend, pytest. New code in `research/runners/`; no `sim/` (protected) edits unless a task proves it
strictly necessary (flag + stop if so).

**Terms (defined once):** *transmission gate* = a per-pathway multiplier on effective synaptic CURRENT in [0,1],
opened/closed at runtime; *couple_gate_to_pool* = wire a gate so a control pool's firing opens it; *fill bank* = the
composer's `fill_ON/OFF` source neurons that carry a filler's code into the coincidence (bind) circuit; *role
ensemble* = the parser's output neurons for one role (agent/action/patient).

**The standing gate (every task ends green):** the 10 on-brain tests + the unified-bridge tests pass; and the
synaptic-route path reproduces the Python-hand-off path's capability (the same fact stored + recalled). A regression
is a reportable finding, committed honestly to both remotes — never hidden. Plain professional language. GPU/CuPy for
real runs (numpy only for tiny smoke). Never weaken the frozen bars or the no-confab moat.

---

## Task 1: De-risk — does a parser-role-gated route deposit a word's code in the SELECTED role bank only?

**This is the load-bearing falsification. If the gating does not route cleanly, step 2's approach must change — do
this before building anything.**

**Files:** Create probe `research/findings/raw/_step2_gated_route_probe.py`; Test `tests/test_unified_brain_bridge.py`
(append).

**Step 1 — Write the failing test.** On a `UnifiedBrainBridge` (small proj_dim, e.g. 64): add a `word_code_pool`
(proj_dim neurons) and three gated pathways `word_code_pool → composer.fill_ON[role]` (and `fill_OFF[role]`), one per
role, each tagged with a `transmission_gate` named `route_<role>` and coupled to that role's parser ensemble
(`bridge.couple_gate_to_pool("route_agent", <agent ensemble indices>)`, etc.). Drive the parser conjunction for
position 0, active voice (→ the AGENT ensemble should fire) AND drive a known word code into `word_code_pool`; run a
window; assert the AGENT fill bank received the code's drive (its firing tracks the word code) while the ACTION and
PATIENT fill banks did NOT (their gates stayed closed). Then drive position 2 active (→ PATIENT) and assert the code
now lands in the PATIENT bank, not agent/action. (Re-binding the same code to a different role by changing only which
parser conjunction is driven — zero weight change.)

**Step 2 — Run; expect FAIL** (probe/wiring not built).

**Step 3 — Implement** the probe: build the unified bridge, add the word-code pool + the three gated routes
(`inject_explicit_wiring` for the `word_code_pool → fill_ON/OFF[role]` edges, `weight ~ FILL`-scale; tag each pathway
`transmission_gate="route_<role>"`; `couple_gate_to_pool`), and the drive/readout. Find the exact composer
`fill_ON/OFF[role]` indices from `CoreSimComposer.idx` (offset by the composer_offset). The word code is a concept
code (use a synthetic orthonormal codebook for the probe, as the step-1 tests do).

**Step 4 — Run:**
- **PASS** (code routes to the parser-selected role only, both directions) → the mechanism is validated; proceed to
  Task 2.
- **FAIL** → STOP. Write `research/findings/2026-06-04-step2-gated-route-DErisk.md` with what happened (e.g. the gate
  doesn't open enough on the parser's firing rate, or cross-talk between role banks), and surface to the controller +
  owner: the synaptic route needs a different design (e.g. an intermediate relay, or gain tuning) before the build.
  Do NOT proceed to Task 2 on a failed de-risk.

**Step 5 — Commit** (probe + test); push both remotes. Message `de-risk(B step2): parser-role-gated code routing`.

---

## Task 2 (only if Task 1 PASSES): `UnifiedBrainBridge.hear_synaptic` — comprehend→store via the gated route

**Files:** Modify `research/runners/unified_brain_bridge.py`; Test `tests/test_unified_brain_bridge.py`.

**Step 1 — Failing test:** `u.hear_synaptic("dog go north")` stores the SVO fact via the gated synaptic route (no
Python `{role: word}` dict passed to `store`), and `u.query_patient("dog","go") == "north"` / `query_agent` /
abstention all hold — IDENTICAL to the Python-hand-off `u.hear`/`store`. Voice-invariant: the passive frame stores
the same fact.

**Step 2 — Run; expect FAIL.**

**Step 3 — Implement** `hear_synaptic`: wire the word-code pool + the three role-gated routes once at construction
(or lazily); for each of the 3 words, drive its (position, voice) conjunction + its concept code into the word-code
pool, let the gated route deposit the code in the parser-selected role's fill bank, accumulate the three role drives,
then trigger the composer's bind on the accumulated fill state (reuse the composer's bind path, but sourcing fill
from the synaptic route rather than the orchestrated external drive). Keep the existing Python `hear` as the
reference path (do not delete — it is the regression oracle).

**Step 4 — Run; expect PASS** (synaptic route reproduces the Python path). **Step 5 — Commit + push both remotes.**

---

## Task 3 (only if Task 2 PASSES): no-regression gate via the synaptic route

**Files:** probe `research/findings/raw/_step2_synaptic_capability_probe.py`; Test (append, skip-by-default heavy).

**Step 1 — Failing test (heavy, skip-by-default like step 1's):** at production `D=2048`, multi-seed, the capability
matrix driven through `hear_synaptic` (the synaptic route) is within ±1 trial of the Python-hand-off path (the
synaptic route does not regress recall). **Step 2 — Run; expect FAIL.** **Step 3 — Implement** the probe. **Step 4 —
Run:** PASS → step 2 DONE (comprehension routes composition in spikes, no regression); write
`research/findings/2026-06-04-one-bridge-unification-step2-DONE.md`. Any regression → honest finding with numbers +
the controller decides. **Step 5 — Commit + push.**

---

## Final review + handoff

After Task 3: dispatch a code-quality review over the step-2 diff; confirm no `sim/` edits (or the one strictly-needed
edit flagged); confirm the on-brain + unified tests green; confirm both remotes carry every outcome. Then surface
step-2 completion to the owner and STOP for the step-3 (dlPFC merge — the dt=0.5 + NMDA hard case) decision — do not
auto-start step 3.
