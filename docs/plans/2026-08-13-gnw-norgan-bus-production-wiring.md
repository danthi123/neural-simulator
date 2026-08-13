---
type: plan
status: live
date: 2026-08-13
---

# GNW N-organ ignition bus — DESIGN for the `brain_chat` production wiring (what replaces the host organ-combination)

*Forward-looking design/plan (mechanism: gnw-workspace). Asserts no new measurement of its own; the de-risk it rests on is the finding `research/findings/2026-08-13-gnw-norgan-ignition-bus-substrate-combines-N-organ-reads.md`.*

This is the design deliverable for faculty-map **T1-1 Phase-B**: make the spiking GNW workspace the PRODUCTION organ-combination mechanism, replacing the host Python that currently snapshots each co-resident organ and combines their reads in `ChatBrain.gate()`. The de-risk that this design rests on is `research/runners/_gnw_norgan_bus_derisk.py` (the N-organ generalization of the 2-organ coincidence-integrator keystone) — 6-seed gated, with a `--prototype` mode that already routes 3 REAL composer organ reads through the bus and reproduces `gate()`'s decision.

## 1. What the host orchestration does today (the thing being replaced)

`ChatBrain.gate(question)` (`research/runners/brain_chat_tui.py:229`) resolves a free-text question to a stored SVO fact by COMBINING several organ reads in Python control-flow. Reached in production via `webapp/server.py::brain_chat` -> `chat.gate()` / `chat.answer()` (`server.py:3216`+). The combination is host `if/else`:

- **Organ 1 — spiking recall** (`_substrate_recall` -> `inner.what_does(a,v)` -> `composer.query_patient`): the role-aware recalled patient, or `__ABSTAIN__`.
- **Organ 2 — host QuestionRouter** (`router.match_fact(q, stored_facts)`): a keyword-overlap fact match (the fallback for self/identity + odd forms).
- **Organ 3 — VERIFY re-check** (`recalled = inner.what_does(a,v); if recalled == p`): the router's pick is only returned if the spiking recall CONFIRMS it.

The decision `if recalled == p: return [a,v,p] else None` is exactly a **consensus-with-veto over 3 organ reads, computed by host Python**. The no-confab moat (abstain on a well-formed query the substrate cannot answer) is a Python `return None`. This is the host orchestration the BRAIN-BASED-ONLY standard flags as a shortcut: the organs are neural, but their COMBINATION is host code.

## 2. What the bus replaces it with (the substrate does the combination)

Each organ writes its read as a SUBTHRESHOLD drive vote into a persistent shared `workspace` region (K self-recurrent NMDA slots + one shared inhibitory `workspace_fs` pool). The substrate's ignition dynamics do the combining:

- Organs that agree ACCUMULATE drive on one slot; D_SUB is calibrated so a slot ignites IFF it reaches the consensus quorum Q (unanimity Q=N in the gated arm; a majority quorum Q=2 does plurality — both measured).
- The shared inhibitory pool WTA-selects the most-supported slot (single-content access; measured `mutual_exclusion_frac`).
- The committed spiking winner BROADCASTS BACK as the next premise (re-entry), enabling a multi-hop conclusion the one-shot host pipeline cannot reach.
- No slot reaching quorum => nothing ignites => abstain. **The moat becomes a substrate property (sub-quorum drive), not a Python `return None`.**

The mapping is 1:1 onto `gate()`: organ 1 (recall), organ 2 (corroboration), organ 3 (VERIFY) each cast a subthreshold vote for their candidate patient; the substrate ignites the patient IFF the organs corroborate, and abstains otherwise. The `--prototype` mode demonstrates exactly this (parity with the host `gate()` on stored queries; abstains on unstored/inconsistent).

## 3. The wiring plan into `ChatBrain` (additive, default-off first, then flip to default-on)

Introduce a `WorkspaceBus` collaborator on `ChatBrain` that owns ONE persistent `build_workspace_bridge(...)` (built once per brain, kept warm alongside the composer). A new `ChatBrain.gate_via_bus(question)`:

1. PROPOSE — compute the SAME organ reads `gate()` computes now (recall / corroboration / reverse-VERIFY); NO new organ code, reuse the existing composer methods.
2. WRITE — each present organ writes D_SUB to its candidate's slot; a decoy occupies a rival slot (WTA is exercised).
3. IGNITE — one `_ignite_and_read`; the committed winner (or None) IS the gate decision.
4. RE-ENTER (only for multi-hop / "what does what X eats eat?") — broadcast the winner back as the next premise; the single-hop path takes one cycle.
5. Return the committed SVO (or None -> the existing render/abstain path is unchanged downstream).

**Escape flag + byte-identical-when-off.** Add `ChatBrain(..., organ_bus="off")` (default `"off"` at first). When off, `gate()` runs the EXACT current code path — byte-identical, asserted by a determinism test that hashes the returned SVO over the fixture question set with the bus off vs. the pre-change baseline. When `"on"`, `gate()` delegates to `gate_via_bus`. The webapp exposes the flag but keeps the default until the flip criterion below is met.

**Lesion-load-bearing (the production honest-negative check).** A `organ_bus="lesion"` mode builds the workspace with the assembly self-recurrence zeroed (`build_workspace_bridge(lesion=True)`). In this mode the bus must FAIL to answer multi-organ-consensus queries (collapse) while the single-hop reflex (`query_patient` direct, never routed through the workspace) still answers — the same dissociation the de-risk gates. This is the production proof that the substrate, not a residual host read, is doing the combination.

**Moat-safe.** The bus abstains by sub-quorum drive (no ignition). The gate keeps its existing `__ABSTAIN__` short-circuit for a well-formed query the primary organ misses, so the no-confab guarantee is at least as strong; the bus can only ADD abstentions (consensus-veto on conflicting organs), never remove one. A test asserts the bus abstains on every query the host gate abstains on (superset).

## 4. The flip-to-default criterion (when `organ_bus` default becomes `"on"`)

Per `docs/TERMS.md` **closed**: the end-to-end capability gate passes with anti-cheats at 6 seeds AND the shipped default path uses it. Flip when: (a) the 6-seed N-organ de-risk clears every seed; (b) `gate_via_bus` reproduces the host `gate()` decision on the full console fixture set (parity, per-question) across all six seeds; (c) the lesion mode collapses multi-organ consensus while the reflex survives, in the PRODUCTION `ChatBrain`; (d) the byte-identical-when-off determinism test passes; (e) the moat-superset test passes. Until (a)-(e), the term for the bus is **de-risked**, NOT **closed**, and the default stays `"off"`.

## 5. Rungs de-risked vs. the missing rung (honest status)

- DE-RISKED (6-seed positive in the corpus): spiking ignition (rung-1), mutual exclusion / single-content (rung-2), broadcast + report==reasoning (rung-3/3b/3c/4), re-entry over 3 hops (P1.2), the SUBSTRATE combining TWO subthreshold organ reads (coincidence-integrator keystone), and now N>=3 organ reads via consensus-ignition + WTA + re-entry (this de-risk).
- MISSING (this design's closure step, NOT YET built): the additive `WorkspaceBus` delegated to inside `ChatBrain.gate()` with the flip criterion met — i.e. `webapp/server.py`'s host organ-combination would then be performed by the substrate, not host Python. This design is buildable now against the de-risked bus; the flip is gated on the 6-seed de-risk clearing + the production parity/lesion/byte-identical/moat tests above.
- FOLLOW-ON rungs (named, not blocking): a genuinely heterogeneous non-composer organ (a spiking surprise/familiarity monitor, the P0.3 affect/value organ) as one of the N votes; an ACC-style conflict read of `n_ignited`/disagreement to gate an extra deliberation cycle or raise the abstain threshold; a hyperdirect STN->GPi STOP-SIGNAL as the veto effector; and the fully-continuous (no per-hop-reset) form, gated on Rung-2b (async attractor + adaptation eviction).

## 6. Files
- Bus + gated de-risk + prototype: `research/runners/_gnw_norgan_bus_derisk.py` (reuse-by-import of `build_workspace_bridge`/`_ignite_and_read` from P1.2 and `_assign_slots`/`_pick_decoy` from the coincidence-integrator keystone; NO `sim/` edit).
- The keystone it generalizes: `research/runners/_gnw_coincidence_integrator_derisk.py`, finding `research/findings/2026-08-12-gnw-coincidence-integrator-substrate-combines-two-organ-reads.md`.
- The host orchestration to replace: `research/runners/brain_chat_tui.py::ChatBrain.gate` (reached from `webapp/server.py::brain_chat`).
