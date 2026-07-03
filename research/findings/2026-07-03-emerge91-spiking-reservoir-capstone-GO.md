# EMERGE-91 — the conversational-turn capstone with a SPIKING comprehender: comprehension is now the on-bridge spiking reservoir — **GO** (6-seed)

**Date:** 2026-07-03
**Runner:** `research/runners/_emerge90_conversational_turn_capstone_derisk.py --spiking-reservoir`
**Test:** `tests/test_emerge90_conversational_turn_capstone.py::test_seed42_spiking_reservoir_capstone`
**Raw:** `research/findings/raw/_emerge91_spiking_reservoir_capstone.json`

## Why (removing the "comprehension is rate" caveat)

EMERGE-90's adversarial-verify verdict flagged (MAJOR-2) that the capstone's COMPREHENSION used the EMERGE-88 **rate**
echo-state reservoir, while the memory + production halves were spiking. EMERGE-89 had already proven the on-bridge
**spiking** reservoir (`OnBridgeLSM` — a recurrent Izhikevich `BrainRegion` on a `SimulationBridge`) drives the composer
identically (parse/recall 1.000). EMERGE-91 is that mechanical swap in the FULL capstone: comprehension now runs on the
spiking reservoir, so **all three components are spiking bridges**.

## The mechanism (the swap)

An additive default-off `--spiking-reservoir` flag (`_derisk_one(seed, spiking_reservoir=True)`) injects the EMERGE-82
`OnBridgeLSM` (identical `final_state(U)` signature) into EMERGE-88's `ReservoirComprehender` as its reservoir (with the
EMERGE-82 reduced train, the bridge step being heavy). The rest of the turn — store (RF composer), ASK → recall, SPEAK
(the spiking C_TRANS producer), the gate-first moat, and all anti-cheats — is byte-identical. Default off = the EMERGE-90
rate-reservoir capstone, byte-preserved.

## The de-risk — **GO** (6 seeds; reuse EMERGE-82/89 + the EMERGE-90 capstone; NO `sim/` edit)

| gate | value (6-seed) | bar |
|---|---|---|
| **parse** — the SPIKING reservoir comprehends the heard transitive into (agent, action, patient) | **1.000** | ≥ 0.90 |
| **recall** — composer recalls the patient | **1.000** | ≥ 0.90 |
| **render_exact** — the producer SPEAKS the answer with the spiking emission order | **1.000** | ≥ 0.90 |
| **no-confab MOAT** — unstored → false-accept / producer-invocations-on-abstain (gate-first) | **0.000 / 0** | ≤ 0.05 / == 0 |
| **comprehension-lesion** — reservoir closed-class identity collapsed → render collapses | **0.000** | ≤ 0.30 |
| **producer-no-learn** — learned spiking order removed → spoken order collapses | **0.014** (≈ chance) | ≤ 0.60 |

*(seed 42 confirmed: parse/recall/render 1.000, moat 0/0, lesion 0.000, no-learn 0.042; the 6-seed aggregate is in the
raw json.)*

**The result:** the full conversational turn — HEAR → comprehend → store → ASK → SPEAK — now runs with a **spiking
comprehender**: the recurrent Izhikevich reservoir parses the heard sentence on spikes, the RF composer stores + recalls
on spikes, and the Izhikevich producer speaks the answer with the spiking emission order. All three components are
spiking bridges, 6-seed unanimous, moat gate-first.

## Honest scope (what still remains)

- **This removes MAJOR-2** (comprehension is now spiking, not rate). The **word SURFACES are still host-token**
  (`spell=str`, the spiking ORDER is the load-bearing claim) — the A→W neural word-spell (EMERGE-67/68/69, cupy) is the
  next purity rung.
- **MAJOR-1 still stands:** the three spiking bridges are still **SEPARATE substrates with host-dict hand-offs** — NOT
  one shared bridge, NOT cross-synaptic interaction, so still **NOT "one brain"** in the EMERGE-70/71 sense. Folding the
  three onto ONE bridge + making the comprehend→store→speak hand-offs synaptic (the EMERGE-87 disjoint-slice pattern) is
  the genuine one-brain consolidation — the deepest remaining follow-on.
- Reuse-by-import (EMERGE-82/89 `OnBridgeLSM` + the EMERGE-90 capstone); NO `sim/` edit; the default path (rate
  reservoir) is byte-preserved.

## Files
- `research/runners/_emerge90_conversational_turn_capstone_derisk.py` — the additive `--spiking-reservoir` flag +
  `_derisk_one(spiking_reservoir=True)`.
- `tests/test_emerge90_conversational_turn_capstone.py::test_seed42_spiking_reservoir_capstone` — the slow gate.
- `research/findings/raw/_emerge91_spiking_reservoir_capstone.json` — the 6-seed spiking-comprehender turn.
