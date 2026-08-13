---
type: finding
status: active
date: 2026-08-12
mechanism: episodic-dialogue-recall
lane: EPISODIC
seeds: [42, 43, 44, 100, 101, 102]
instrument: standalone numpy-CPU verify `research/runners/_verify_d5_episodic_organ.py` driving the production organ `research/runners/d5_episodic_production_organ.py`, which reuses-by-import (NO reimplementation) `research/runners/_episodic_dap_dialogue_memory.py::EpisodicDapMemory` (the 2026-08-10 kt=8 fix) -> the standing 6/6-GO gap#5 dendritic-dAP readout completion (`_gap5_dendritic_dap_readout_completion_derisk.py`, ab9f7dbe). Load-bearing control: the lesioned read restores the UNFORMED baseline recurrent weights before the apical read.
---

# D5-EPISODIC — the autobiographical episodic RECALL GATE is a genuinely-SPIKING, lesion-load-bearing, 6-seed GO; built as a production co-resident organ + numpy-CPU verified, WIREABLE into /api/brain-chat (fact CONTENT + recency remain declared host residuals)

## Verdict: WIREABLE = YES (the recall GATE), with named residuals

The brain's decision *"have we discussed topic T this conversation / is the referent in episodic memory"* is a genuine
spiking hippocampal pattern-completion — NOT a host list scan. Each spoken TOPIC BTSP-forms a CA3 assembly on a
dedicated dendritic-dAP readout bridge; a later referential cue ("earlier you told me about X") COMPLETES that assembly
cue-specifically via the two-compartment dendritic dAP apical read (the fraction of held-out cells whose intrinsically-
bistable `cp_v_apical` latch reaches the UP state after the cue volley is driven and the bridge is stepped). This is a
`cp_v_apical` state read, NOT a host formula. It was adversarially corrected once (the 2026-08-10 "numpy backend-block"
was a wrong-kthresh non-firing operating point, not a forward-Euler limit) and re-verified at kt=8 in FRESH isolated
builds + an independent coordinator re-verification: 6/6 seeds fire cue-specifically on BOTH numpy and cupy with
perm=nocue=lesion(baseline)=0.

**What I built this session (additive, guarded, NO `sim/` edit):**
- `research/runners/d5_episodic_production_organ.py` — the production co-resident organ (same shape as the affect /
  metacog / surprise / curiosity / worldmodel organs). `episodic_enabled()` (default-ON, `BRAIN_EPISODIC=0` escape) /
  `episodic_lesioned()` (`BRAIN_EPISODIC_LESION=1`). `EpisodicRecallOrgan` wraps ONE `EpisodicDapMemory` per
  conversation (episodic memory ACCUMULATES, so it is conversation-scoped, NOT a process singleton — the honest
  structural difference from the stateless read organs): `note_topic(topic)` = the spiking BTSP WRITE,
  `recall(topic, lesion)` = the spiking dAP READ, `discussed(lesion)` = the spiking decode of which topics complete.
  `get_episodic_organ(cache_key, ...)` / `reset_episodic_organ(cache_key)` mirror the server's `_SESSION_MOOD`
  per-conversation registry. `is_referential` / `extract_referent` (host parse-side) + `recall_disclosure` (the honest
  gate-then-content line; a completion failure is an honest "I don't recall", never a confabulation).
- `research/runners/_verify_d5_episodic_organ.py` — the standalone numpy-CPU verify harness (below).

## Verify — numpy-CPU, seed 42, GO config (kt=8), through the production organ

<!--derived-->

The harness drives the organ end-to-end on the numpy substrate (the production test backend) and checks the four
load-bearing properties:

| check | result (seed 42, numpy) | verdict |
|---|---|---|
| (1) flag semantics + idle-builds-nothing (`BRAIN_EPISODIC`/`_LESION`; an organ that never stores builds NO substrate, `mem is None`) | all flag cases correct; idle `mem is None`, idle recall in_memory=False | PASS |
| (2) INTACT FIRES — `note_topic('dog')` (BTSP `w_within` grown 1.5 -> 82.0, emergent sizes [27,22]) then `recall('dog')` completes | apical_cue = 0.909, perm = 0.000, nocue = 0.000, in_memory=True | PASS |
| (3) LESION COLLAPSES (load-bearing) — `recall('dog', lesion=True)` reads UNFORMED baseline weights | intact 0.909 -> lesion 0.000, in_memory=False | PASS |
| (4) UNSTORED ABSTAINS (honesty floor) — `recall('cat')` never stored | cat cue = 0.000, in_memory=False; `discussed()==['dog']` | PASS |

`ALL_OK=true` (wall 869 s; the BTSP store alone was 510 s on numpy@2000 — the declared latency residual). Honest
disclosures emitted: dog -> *"Earlier you brought up dog — my hippocampal readout completes its assembly for it
(dendritic dAP completion 0.91). A dog runs north."*; cat -> *"I don't recall us discussing cat — no assembly completes
for that cue (a genuine spiking completion failure, so I won't make something up)."* The harness makes the explicit
`tools.lab.attributable_to("dog recall: dAP intact vs lesioned baseline", 0.909, 0.000) = 1.0` <!--derived--> call:
100% of the completion is carried by the BTSP-formed assembly, 0% by the organ glue / feedforward cue. The (2)-vs-(3)
gap is the teeth:
the recall is carried by the BTSP-formed CA3 assembly (restoring baseline weights collapses it), not by the organ glue.
6-seed breadth is the standing GO (`research/findings/raw/_episodic_dap_kthresh/clean_verify_kt8.json`, 6/6 both
topics, perm=nocue=lesion=0, including the smallest 13/14-cell emergent assemblies).

## What is SPIKING vs HOST (the honesty boundary, declared)

- **SPIKING (load-bearing):** the RECALL GATE — which topics completed + is-the-referent-in-memory — decoded from the
  dendritic dAP completion. The lesion (baseline weights) collapses it; the moat can only ever be tightened.
- **HOST (declared residuals, each rides an existing burn-down):**
  1. fact CONTENT ("what you told me about X") is still the per-conversation host oracle buffer — the gate is spiking,
     the retrieved sentence is the next conversion (the same class of residual as affect's appraisal-content scaffold).
  2. temporal/recency ORDER is a host store-index; the WHEN attribute pool of the episodic-cue-recall design
     (2026-08-08 research gate) was scoped but NOT built, so there is NO spiking recency signal yet.
  3. the gap#5 converse->sleep->clean-CA3-replay->converse CAPSTONE (`_gap5_onebrain_capstone.py`, a separate 6-seed
     GO) preserves the conversation through an offline replay, but its replay is a place-field TRAJECTORY decoded by a
     Bayesian instrument, not an autobiographical turn re-encode — it is offline CONSOLIDATION, NOT a per-turn read, so
     it is deliberately not on this per-turn path (named E5-adjacent).
  4. LATENCY: a BTSP store is ~seconds on cupy but ~6 min on numpy@2000 (the load-bearing n_ca3=2000 emergent scale);
     production is cupy, and a precompute->.npz cache of (assembly geometry + formed weights) amortises it.
  5. build-to-build emergent-size non-determinism (FMA/summation reorder) + a small unstored-slot specificity margin
     (the coordinator re-verification in `2026-08-10-episodic-dialogue-recall-wired-to-spiking-dAP-readout-numpy-backend-honest-negative.md`
     saw up to 0.154 <!--derived--> on one seed, still below the 0.20 `in_memory` gate) — both safety-netted by a
     host-oracle self-consistency fallback: a spiking mis-fire can suppress a recall but can NEVER invent one.

## The "emergent-assembly CA3 completion is a boundary (gap#5 E5)" note — SURPASSED, not a wall

Per THE LAW: the E5 emergent-assembly-completion boundary was the kthresh OPERATING POINT (kt=30 fired nothing; the
small ~13-cell emergent assembly needs a lower dendritic coincidence threshold, but too low self-ignites). kt=8 threads
the window and fires the smallest emergent assemblies cue-specifically, 6/6, both backends (2026-08-10). Emergent-DG
membership + the ~23-cell scale are LOAD-BEARING (a pre-assigned 0.18*N assembly does NOT complete), so the recall is
genuinely on the emergent engram, not a hand-set mask.

## Wiring spec (for the integrate agent) — see the structured `wiring_spec`

Hooks into `webapp/server.py::brain_chat` at two points: (a) after a normal answered turn, `note_topic(subject)` (the
spiking WRITE, guarded); (b) on a referential turn (`is_referential(msg)`), `recall(referent, lesion=episodic_lesioned())`
gates whether to surface the host-oracle content or an honest "I don't recall". Per-conversation via a `_SESSION_EPISODIC`
registry keyed by `cache_key`, cleared on `reset_conversation` (alongside `_SESSION_MOOD`). Default-ON, `BRAIN_EPISODIC=0`
byte-identical, `BRAIN_EPISODIC_LESION=1` load-bearing. Composes with the other default-on organs without conflict: it
touches only the referential branch (no overlap with recall/CHOOSE/LEARN/GENERATE/MOAT/AFFECT/SURPRISE/COMPREHENSION/
METACOG/WORLDMODEL/CURIOSITY) and never enters the certainty band.

## Provenance
`d5_episodic_production_organ.py` + `_verify_d5_episodic_organ.py` (NEW, this session); reuse-by-import of
`_episodic_dap_dialogue_memory.py` (committed) -> `_gap5_dendritic_dap_readout_completion_derisk.py` (ab9f7dbe) +
`_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py` + `_gap5_emergent_end_to_end_episodic_loop_derisk.py`. NO `sim/`
edit. numpy-CPU verify raw: `research/findings/raw/_d5_episodic_organ/verify_s42_numpy.json`.
