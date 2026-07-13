# RUNG 6c — novel-referent (OPEN) discourse binding via a content-agnostic HEBBIAN FAST-WEIGHT (6-seed GO): the emergence-aligned close of the RUNG6b open frontier

**Date:** 2026-07-13
**Runner:** `research/runners/_novel_referent_hebbian_fastweight_derisk.py` (reuse-by-import: sparse developmental-random barcodes + the D3 possession task + `discrete_attractor_rnn`; numpy-CPU; NO `sim/` edit). Research gate: `2026-07-13-novel-referent-binding-research-gate-emergent-Hebbian-fastweights.md`.
**Verdict:** GO (6-seed) for the novel-referent BINDING contribution; the absolute tracking number inherits the D3 attractor's autoregressive-rollout ceiling (a separate axis, reported).

## The frontier (RUNG6b → 6c)
RUNG6b resolved unbounded referent tracking for a **KNOWN** referent set (a SUPERVISED read maps entity codes → slots). The open frontier: a **NOVEL** entity (never seen by the read) must get its OWN fresh distinct slot — dynamic VARIABLE BINDING / indirection, not category generalization. **Emergence-bar decision (research gate):** build the content-agnostic HEBBIAN FAST-WEIGHT (Ba-2016 / Bouchacourt-Buschman flexible-WM / Kriete-O'Reilly-2013 indirection), NOT the FHRR exact-inverse scaffold.

## The mechanism
Referents are sparse developmental-random barcodes (the `entity_instance_layer` primitive). A per-narrative Hebbian fast-weight `W` (K slots × code_dim): on FIRST mention of an entity, assign the next free slot + `W[slot] += code/‖code‖` (one-shot Hebbian bind); on re-mention, `slot = argmax(W@code)` if `max > θ` (retrieve). Content-AGNOSTIC → a novel entity binds identically to a known one BY CONSTRUCTION. The narrative is re-expressed in bounded SLOT space; the validated D3 discrete-attractor tracks the holder slot; the final slot dereferences to the entity.

## Result (6-seed 42/43/44/100/101/102; the tracked entities are MINTED AT TEST, disjoint from the training pool = held-out NOVEL; held-out-DEEPER lengths 6-8; chance 1/6=0.167)
| metric | mean (all 6 seeds) |
|---|---|
| **NOVEL-entity holder-track @deeper** | **0.525** (0.496–0.562) |
| **binding-penalty** (slot-ceiling − entity-level) | **0.000 every seed** |
| **binder collisions** on held-out novel | **0.000 every seed** |
| MERGE-lesion (α=0 identical codes) | **0.000 every seed** (collapses) |
| NO-BIND lesion (random slots) | 0.16 (chance) |
| retention floor | 0.29 (beaten by +0.23) |
| **GO** (penalty<0.05 ∧ novel>retention+0.15 ∧ merge/no-bind<0.35 ∧ collisions<0.02) | **GO — 6/6** |

## What the GO means (and the honest caveats)
- **The novel-referent BINDING is solved, emergently.** The Hebbian binder maps held-out NOVEL entities to clean distinct slots — **zero collisions, zero binding-penalty** (entity-level == the attractor's slot ceiling): the binder adds NO error over the tracker. It **generalizes to never-seen entities** (minted at test, disjoint from training) because the bind is content-agnostic. No supervised read, no fixed FHRR algebra — the emergence-bar-aligned mechanism.
- **Both lesions load-bearing:** MERGE (identical codes → cannot individuate → 0.000) and NO-BIND (random slots → chance) — the barcode's distinctness AND the Hebbian bind are both required.
- **Honest caveat 1 — the absolute number (0.525) is the INHERITED D3 ceiling, not a novel-referent limit.** The attractor's per-step transition is 0.881 (teacher-forced), which over 6-8 autoregressive steps compounds to ~0.53 (0.881^7≈0.42..0.88^6≈0.47). This is the discrete-attractor's own autoregressive-rollout accuracy (D3 reported 0.881 teacher-forced) — a SEPARATE axis from novel-referent binding (which adds 0.000 penalty). Improving it = the attractor's rollout robustness (e.g. re-discretization confidence), not the binder.
- **Honest caveat 2 — barcode layer, decoupled from the reslm generation.** Novel/open referents are barcodes (a novel OOV entity is not in the reslm's fixed vocab, so it cannot be reslm-encoded — the barcode is the correct substrate for open referents). This is the entity-tracking layer; wiring it to perceived/streamed novel entities is the integration follow-on.
- **Honest caveat 3 — host-side pieces:** the cleanup argmax + the barcode mint are numpy (same scaffolds flagged in `entity_instance_layer`). The FULLY-SPIKING realization — the Hebbian fast-weight as STP/Mongillo synaptic working memory + a spiking FS-WTA cleanup — is the next rung.

## ⇒ Ladder status + next
The RUNG6b open frontier (bounded → open referents) is CLOSED at the mechanism level by the emergent Hebbian binder (6-seed GO, generalizes to held-out novel). Combined with RUNG6/6b, the reslm generator's discourse arc: short-range emergent tracking (6) → unbounded for known referents (6b) → **open/novel referents (6c, this)** — all emergent, no composer scaffold.
**NEXT CONCRETE ACTION:** the FULLY-SPIKING realization of the fast-weight binder — STP/Mongillo synaptic WM (a `sim/` STP path already exists) for the one-shot bind + a spiking FS-WTA cleanup for the slot read-out — so the binder is on-substrate (the caveat-3 close). Then wire the barcode referent register into the D3 event-agent / `MultiTurnAgent` path (opt-in, byte-identical off). The attractor's autoregressive-rollout ceiling is a parallel, separate improvement axis.

Reuse-by-import; NO `sim/` edit. Runner: `_novel_referent_hebbian_fastweight_derisk.py`.
