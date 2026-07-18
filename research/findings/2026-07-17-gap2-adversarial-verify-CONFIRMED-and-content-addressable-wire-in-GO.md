# Gap #2 — adversarial-verify CONFIRMED + content-addressable wire-in mechanism 6-seed GO (2026-07-17)

Follow-ons (b) and (c) of "finish gap #2 fully" (the fully-spiking competitive-slot binder). Follow-on (a)
— neuralizing the readout reset — is in `2026-07-17-keystone-2-spiking-slot-binder-STEP1-prereq-GO.md`.

## (b) Adversarial verification — CONFIRMED

An independent skeptic subagent was tasked to REFUTE the P=3 fully-spiking 6-seed GO
(`_keystone2_spiking_slot_binder_derisk.py`). It ran the killer controls the original arc had NOT run:

| Confound probed | Control | Result | Expectation for a genuine result | Verdict |
|---|---|---|---|---|
| Baseline | slot-sep P=3 | **1.00 (6/6)** | — | reproduced |
| — | shared superposition | 0.33 | should cap | holds |
| — | permuted-role | 0.00 | should collapse | holds |
| **write load-bearing** | **NO-TEACH** (gate never opens → zero weights) | **0.167 = chance** | must collapse to chance | ✅ |
| **write load-bearing** | **SCRAMBLE-TEACH** (slots taught wrong fillers) | **0.00** | must collapse (reads taught-wrong) | ✅ |
| **small-space luck** | KF=8 / 10 / 12 (chance 0.125/0.10/0.083) | 1.00 / 0.83 / 0.94 | must stay ≫ chance | ✅ 11× at KF=12 |
| **permuted-null valid** | prediction trace | wrong slot reads its OWN taught filler at real rate | genuine addressing, not silence | ✅ |
| **neural==host** | P=4 head-to-head | both `[0.75 0.75 1 0.75 0.75 0.75]` per-seed | should match | ✅ byte-identical |
| **seed seeds substrate** | threshold hashes 42/43/44 | 3 distinct hashes/means | must differ | ✅ (`cfg.seed` set) |

The single most damning control — **scramble-teach** — collapses to **0.00** (not chance), because the network
reliably reads the *taught-wrong* filler: the read is a deterministic function of `taught(cued_slot)`. That is the
exact signature of a genuine learned, role-addressed binder — not fixed structure, drive-leak, small-space luck,
or recency. **Verdict: CONFIRMED.**

Honest scope limits (author-disclosed, NOT confounds): the NMDA hold is not load-bearing (it is a competitive-slot
plastic-weight LTM store, not the step-2a WM coexistence); "shared caps ~2" is a loose label (shared recovers ~1);
scale is small (KF≤12, K=4, P=3) — conversational-vocab scale is a data lever; P=4=0.79 is graceful degradation
(the closed claim is the SVO P=3 case).

## (c) Content-addressable multi-fact recall — the wire-in mechanism — 6-seed GO

`research/runners/_slotbinder_content_addressable_probe.py`. The de-risk validated the single-bind readout (drive
one slot → read its filler is robust) and that slot-SEPARATION beats superposition. The conversational faculty needs
CONTENT-addressable multi-fact recall (cue agent+verb → the right patient; abstain if unknown). The mechanism:

- store each fact's (agent, verb, patient) into its OWN three slots (separate slots = the gap-#2 win, no superposition);
- recall = a NEURAL SCAN (the accepted pipeline pattern, OneBrainComposer GAP-A): for each stored fact, drive its
  agent- and verb-slots and check the read-back fillers against the cue; read the matching fact's patient-slot for
  the answer; if NO fact matches the cue → **ABSTAIN** (the no-confab moat).

This needs NO coexistence/clear machinery (each (fact,role) slot is stored + read INDEPENDENTLY) and no new
coincidence mechanism — it is repeated application of the validated single-bind readout.

**6-seed (42/43/44/100/101/102), 3 facts, vocab 8:**

| metric | mean |
|---|---|
| query_patient (who→what) | **1.00** |
| query_agent (what→who) | **1.00** |
| moat-abstain (absent cue → None) | **1.00** |
| permuted-cue-abstain (real agent + wrong verb → None) | **1.00** |
| scramble-store query_patient (deranged store must NOT recover truth) | 0.33 (chance) |

All 6 seeds: 3/3 both directions, moat 1, perm 1. The scramble-store control (store the same words with the patient
assignment deranged) recovers the true patient only at chance → the recall is genuinely reading the stored binding.

**⇒ gap #2's binder does content-addressable multi-fact conversational recall on spikes, with the no-confab moat,
6-seed GO.** Fillers are concept POOLS (the validated g20/Pulvermüller distributed-word-ensemble representation);
generalization across SIMILAR concepts is the separate, already-closed cross-modal/PPMI arc, not this binder.

## Remaining for full closure
Promote `ProbeStore` → a `SlotBinderComposer` class implementing the composer contract, wire it as a selectable
composer in `BrainConversationalAgent`, add a CI test that the AGENT answers the who/what matrix + abstains through
it. Then gap #2 is fully closed (fully-spiking, 6-seed, adversarially verified, wired into the real system).
