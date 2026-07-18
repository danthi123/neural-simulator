# Gap #2 spiking slot binder — BUILD STEP 1 (slot-separation prerequisite): GO, + the multi-slot-coexistence challenge precisely identified (2026-07-17)

**Per `2026-07-17-keystone-slot-binder-research-gate.md` #1. Composes the EMERGE-41 spiking competitive pooler (`FSWTAProbe`). CPU/numpy probe; the gate's #1 build begun.**

## Step 1 result (3-seed, drive uniform(0,6) = EMERGE-41's working scale)
Distinct ROLE drives → **DISTINCT competitive slots** via the spiking rank-order (Thorpe latency) pooler:
| seed | slot sizes (R=4 roles) | mean pairwise Jaccard |
|---|---|---|
| 42 | [6,6,6,6] | 0.064 |
| 43 | [6,6,6,6] | 0.079 |
| 44 | [6,6,6,6] | 0.064 |
⇒ each role/bind gets its own near-orthogonal 6-column slot (Jaccard ~0.07 ≪ overlap). **This is the load-bearing property**: capacity converts from SNR-limited (~2, the write-rule store) to slot-count-limited (combinatorial). **Prerequisite GO.**

## The precisely-identified next challenge (build step 2): multi-slot COEXISTENCE on ONE bridge
On a REUSED bridge, sequential selections gave `[6,0,6,0]` — the columns' adaptation + FS inhibition from slot-0's firing **suppress the next selection** (state carryover; the EMERGE-61 adaptation-accumulation family). A fresh bridge per selection avoids it (above) but doesn't test coexistence. The real binder needs the P slots to COEXIST on ONE bridge for retrieval. **The gate's designed fix: the D3 persistent-slot ATTRACTOR holds each selected slot** (stable, zero-input, coexisting) — `_d3_persistent_slot_derisk.py`. So build step 2 = pooler-SELECT the slot → D3-attractor HOLD it (per bind) → role-cued RETRIEVE (drive role → complete the matching slot → decode filler), with a per-selection reset (EMERGE-61 wash-out) or the attractor absorbing the carryover.

## Status + next
- Build step 1 (slot separation) GO. Step 2 (coexistence + retrieval via D3 attractor) is the substantial continuation, with the state-carryover mechanism precisely identified.
- GO bar unchanged: a fact's P≥3 bundle recovers on spikes ≥0.80 where the write-rule capped ~2; anti-cheats permuted-role / lesion-the-competition→~2 / homeostasis-OFF; 6-seed.
- THE LAW: the write-rule method is refuted; this competitive-slot method is progressing; the capability stays OPEN until it works end-to-end.
