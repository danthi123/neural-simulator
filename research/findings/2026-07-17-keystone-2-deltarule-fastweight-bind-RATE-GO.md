# KEYSTONE gap #2 — de-risk #2 (rate rung): a SELF-ORGANIZING fast-weight bind handles the brain's own CORRELATED codes, beating both the fixed algebra and the additive wall (GO, 3-seed numpy; honest limits stated)

**2026-07-17, per the keystone gate `2026-07-17-keystone-binder-research-gate.md` (ranked #2). Runner `research/runners/_phaseB_deltarule_bind_bundled_derisk.py`.**

## Result (3-seed, correlated fillers mean|cos|=0.76, chance 0.016)

| arm | recall (top-1, strict exact-concept) |
|---|---|
| **delta-rule fast-weight** (`W += β(v−Wk)kᵀ`, unbind `Wk`) | **1.000** |
| plain-Hebbian fast-weight (`W += vkᵀ`) | 1.000 |
| permuted-role anti-cheat (unbind by WRONG key) | **0.000** (collapses ⇒ read is genuinely role-addressed, not correlated-code artifact) |
| decorrelated-code control | 1.000 |
| K-sweep (roles/fact 3→5→8→12) | **1.000 throughout**, delta==hebbian |

**Vs the prior arms on the SAME wall:** fixed ±1 FHRR bundled 0.873 (#1); additive point-neuron 0.193; learned-linear-inverse 0.056 (`2026-06-16`). The fast-weight bind **beats all of them (1.000)** with a LOCAL write.

## Why it works (the mechanistic insight) — and why this is on the emergence bar

A fast-weight bind stores each (role-key, filler-value) as an OUTER PRODUCT into a per-fact matrix `W = Σ vₖᵀ`; unbind is `W k_t` — **key-addressed**. So the filler correlation, which crosstalks an elementwise `role⊙filler` product, is **irrelevant to the read** (the read is addressed by the near-orthogonal ROLE key, not by the correlated filler). This is a **different, biology-grounded binding STRUCTURE** — synaptic associative memory (Hopfield/Kanerva/Ba-Hinton fast weights; Tsodyks-Markram STP) — that SIDESTEPS the coincidence-product wall rather than fixing it. Per THE LAW that is a legitimate NEW method: it is self-organizing (a LOCAL write rule, NO backprop, NO weight transport), it has NO hand-set conjugate-inverse tie (unbind is just `Wk`), the role keys are developmental random draws, and the filler values are the LEARNED stream-cortex codes.

## Honest limits (stated, not buried — this is a rate-rung result, NOT gap-#2 closure)
1. **The delta rule is NOT load-bearing here.** Random role-keys are near-orthogonal at D=128 ⇒ no key-crosstalk to error-correct ⇒ plain Hebbian ties delta everywhere (incl. K=12). The mechanism simplifies to a **plain Hebbian associative memory**; the delta rule would matter only with CORRELATED role-keys (not the case). Honest: the win is the KEY-ADDRESSED STRUCTURE, not the error-correction.
2. **This is the CORE bind+bundle+recall, not the full VSA algebra.** It does not yet demonstrate the composer's compositional operations (nested binding, `query_chain` multi-hop) nor generalization-via-correlation (recalling about a never-stored but SIMILAR concept — the fast-weight recalls the EXACT stored value). Those are the fuller gap #2 and the next layer.
3. **Rate-rung numpy only.** The fully-spiking one-brain realization (the mission requirement) is the next rung: store the fact in real SYNAPTIC fast-weights on the bridge (the gate's route: RF complex synapses `cp_rf_w_re/im` / STP / eligibility), unbind via spiking transmission, 6-seed. **Closure = that spiking one-brain version wired into the conversational pipeline; this rate rung only greenlights the structure.**

## Verdict
GO (rate rung): a self-organizing, biology-grounded fast-weight bind handles the brain's own correlated codes, beating the fixed algebra and the additive wall — the OP-structure for gap #2. NOT closure. **Next (per THE LAW): the spiking one-brain realization on synaptic fast-weights + the compositional-reasoning check.**

---

## ⚠️ CONNECTION + CORRECTION (drift-#12 a-1, same day) — the EDGE-5 arc already did most of this; the genuine open edge is precise

A read of our own record (before building the spiking rung) found the **EDGE-5 arc (2026-07-15)** is mechanistically the SAME content-addressable fast-weight store (barcode/role → value/filler). It already establishes:
- **`2026-07-15-edge5-rung2-STP-store-onbridge-6seed-GO`**: the **SINGLE-bind store is realized ON SPIKES** on a real `SimulationBridge` via Mongillo STP facilitation — retrieve 0.97, novel-barcode 0.21 (genuine content-addressing), STP-off lesion 0.06. 6/6 GO. **So the spiking store half is DONE.**
- **BUT the MULTI-bind on-bridge store COLLAPSES** (P1 0.92 → P2 0.11, below chance) because **raw STP facilitation is ADDITIVE, not DELTA** — multiple binds interfere/saturate. **NAMED SURPASS (un-built): a delta-like error-correcting on-bridge write** (read the current prediction, write `value − prediction`).
- EDGE-5 rung-1 (numpy) already showed **delta > additive for a SHARED multi-bind store.**

**Correction to my #2 test above:** it used a PER-FACT isolated `W`, which is why delta tied Hebbian (isolated facts, orthogonal keys → no interference). The realistic conversational memory is a **SHARED persistent store** where facts interfere — and there delta IS load-bearing (EDGE-5 + the numpy rung-1 show it). My per-fact result correctly shows the OP structure works, but it trivialized the regime that makes delta matter.

**⇒ the GENUINE gap-#2 spiking closure edge (concrete, un-built):** an **on-bridge delta-like error-correcting write for the MULTI-bind store**, keyed by SVO roles over the brain's correlated filler codes — the EDGE-5 named surpass, applied to the fact binder. Build: extend `_edge5_rung2_stp_store_onbridge_derisk.py` with a read-then-write-the-error plasticity (reward-modulated or subtractive-normalization) so a new bind subtracts the current prediction before writing; GO bar = multi-bind on-bridge retrieve ≥0.80 at P≥3, delta-vs-additive load-bearing, 6-seed. THAT is the closure step, not building the store from scratch.
