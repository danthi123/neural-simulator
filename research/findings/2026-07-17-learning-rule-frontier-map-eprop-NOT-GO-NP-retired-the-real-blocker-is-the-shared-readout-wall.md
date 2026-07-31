---
type: finding
status: live
date: 2026-07-17
mechanism: deep-credit
---

# The learning-rule frontier, mapped from our own record (2026-07-17): feedforward e-prop is NOT-GO, Node Perturbation is already retired, and the real blocker is a SHARED supervised-readout wall — the working path is the UNSUPERVISED stream cortex

**2026-07-17. A read-only scan of our own findings (NP cluster + the 2 confound docs + R3-REFRAME + both 2026-07-15
emergence-engine gates + the SSM cluster), triggered by the seed-fixed deep-credit not-GO. This CORRECTS a stale
premise in the roadmap (that Node Perturbation is "the next bet"). Sourced; disagreements in the record surfaced, not
papered over.**

## Why this exists

Tonight's valid, seed-fixed result: **feedforward e-prop deep credit does NOT reliably beat a fixed random spiking
reservoir** (4/5 blind seeds negative; GO gate falsified). Natural next question: *what learning rule DOES teach a deep
spiking network, then?* The roadmap's standing answer was **Node Perturbation (NP)**. This scan establishes that answer
is **stale** — and, more usefully, that the whole *supervised-deep-credit-on-spikes* family is blocked by one shared
wall, while an *unsupervised* path already works.

## 1. Node Perturbation is NOT the answer — our own record retired it 2026-07-13

- **Off-brain: robust GO.** NP trains hidden layers to near-oracle held-out generalization (6/6 standard + 6/6 fresh
  pre-registered seeds; credit doesn't fade at depth 2–6) where vanilla feedback-alignment memorizes-but-fails
  (`2026-07-13-fresh-deep-credit-class-NODE-PERTURBATION-seed42-GO-...md`). That headline is solid.
- **But every UNIQUE on-spike advantage failed:** the "beats Kolen-Pollack for the language input map" win was a
  **seed artifact — 12-seed REFUTED**, NP on average slightly *worse* than KP (`2026-07-13-NP-vs-KP-...REFUTED-12seed`);
  the "works recurrently" claim was **retracted** by the fresh-seed gate; the on-bridge realization is **blocked**.
- **A 2026-07-13 adversarial-verify explicitly wrote: "retire node perturbation as a deep-credit candidate"**
  (`2026-07-13-onbridge-NP-small-scale-variance-BOUNDARY-...md` correction block) — non-biological, off-mission, and a
  **readout-independent zeroth-order variance wall** (Werfel-Xie-Seung 2005). The earlier "scale-lever will fix the
  variance ∝1/√N" framing was itself a **category error** (the substrate is deterministic — no readout noise to average
  down). ⇒ the roadmap's "NP is the candidate to succeed on spikes" is **superseded by our own later record.**

## 2. The REAL blocker is a wall SHARED by NP, D1/BDSP, and the whole supervised family — the "spiking-readout-training bottleneck"

On the on-bridge net, NP's *hidden* credit fires directionally (moves ~810/832 synapses, flips a misclassification),
**but the sparse spike-count OUTPUT readout, trained by a host delta-rule, hovers at 0.51–0.58 — below the ~0.61
random-hidden floor — so the hidden credit has no room to matter.** This is *"the SAME
spiking-classifier-readout-training bottleneck that underlay D1's on-bridge BDSP negative, and it is NOT NP-specific."*
Population readout / long settle / fixed-pooling / grad-clip all leave NP = frozen. **⇒ what sank feedforward e-prop
tonight, what sank D1/BDSP, and what blocks NP is ONE wall: supervised global-loss deep credit through a spiking
classifier readout does not train on this substrate.**

## 3. The cheapest decisive de-risk — resolve "substrate limit vs rule limit" in one run

The record itself names the skipped experiment: **a rate-net positive control.** `_nodepert_onbridge_derisk.py`, one
single-variable swap — replace the Izhikevich hard spike-reset with a **graded rate transfer**, same net/task/readout,
frozen-hidden control + `shuffle_dl`/`wrong_sign` anti-cheats, ≥6 seeds, **`cfg.seed` set + two-process threshold-hash
verified** (per tonight's seed bug). Read: rate-net converges + hidden credit clears the frozen floor → the block is
the **point-neuron rate-code** (surpass = graded/population coding); rate-net *also* fails → the block is the **rule's
variance** and supervised deep credit is dead on-spike regardless of substrate. **One CPU run, decisive either way** —
and it characterizes the wall for the *whole* supervised family, not just NP.

## 4. The direction the record keeps pointing to: the UNSUPERVISED on-spike stream cortex

The **HTM / competitive-pooler + the committed BDSP `fused_htm_permanence_update` stream cortex** already **learns
rich multi-layer representations from a stream, on-spike, with NO supervised global-loss deep credit** (the EMERGE-30..57
arc). It **sidesteps the exact readout wall** that just sank feedforward e-prop and NP. Multiple findings name it as
the *actual* mission-critical emergence path. Every refuted/scale-gated supervised direction reduces to the same
residual — *"learn better representations from a stream"* — which the unsupervised path is already doing.

## 5. Two 2026-07-15 gates genuinely DISAGREE (unresolved fork — flagged, not hidden)

- `2026-07-15-offdiagonal-recurrent-credit-ARC-SYNTHESIS`: off-diagonal cross-neuron credit is the **emergence enabler**
  (rate-validated, closes +48–64% of the diagonal↔BPTT gap); next test = does the diagonal boundary even reproduce on
  the real on-bridge recurrent substrate?
- `2026-07-15-emergence-engine-research-gate-horizon-frontier`: off-diagonal credit is **near-dead / not the binding
  constraint**; the frontier is a **non-fading content-addressable store** (which *structurally dissolves* the
  recurrent-credit wall) — but that store is **key-quality-bound**, and key quality reduces back to §4.

**The cheap resolver** (also §2-priced): build a small recurrent on-bridge Izhikevich net on a delayed-cue task,
diagonal e-prop only (reuse `_onbridge_eprop_port_derisk` + a recurrent slice, **seeded**). If real population coding
already recovers the graded eligibility the toy lacked → the off-diagonal frontier *dissolves on the real substrate*
(stop; no MDGL build). If it reproduces → MDGL is warranted.

## Ranked recommendation

1. **Cheap de-risk (hours, CPU): the rate-net positive control (§3)** — resolves the substrate-vs-rule ambiguity that
   blocks *every* supervised-deep-credit-on-spikes rule. Reframed away from "prove NP" toward "characterize the shared
   readout wall." Highest leverage per unit cost.
2. **Cheap de-risk (hours): the off-diagonal on-bridge reproduction test (§5)** — resolves the live fork between the two
   2026-07-15 gates; retires or promotes a whole direction. **Must be seeded.**
3. **Bigger build (only if a de-risk earns it): advance the WORKING unsupervised on-spike stream cortex (§4)** — it
   already learns deep representations from a stream on-spike, sidestepping the readout wall entirely, and it is where
   the "better learned representations" residual actually gets addressed.

**Whack-a-mole / dead ends to AVOID:** a bigger on-bridge NP or deeper NP encoder (no edge over KP; retired);
re-running the fixed-reservoir perplexity ladder chasing "beat the n-gram" (all rungs trigram-bound at tractable
scale); the on-bridge STP store (BANKED as hard, key-quality-bound); **and any pre-2026-07-17 same-seed on-bridge
comparison with margin within ~±0.2 from the 8 unseeded runners until re-run seeded** (the ±0.33 confound is ~3× a
deep-credit effect — `task_f077cbfa`).


> **✅ #1 DE-RISK DONE 2026-07-17 (the rate-net positive control):** GRADED coding does NOT unlock supervised deep credit — 6-seed × {spiking,graded}, `depth_helps 0/6` in both readouts, graded slightly worse. The graded readout marginally improves the reservoir's readability but NP's credit adds nothing on top ⇒ **the block is the RULE (NP's credit assignment), not the readout discreteness.** Scope caveat: nothing beats chance (underpowered config) so it's DIRECTIONAL, not a clean readout-vs-rule proof — but the escape hatch does not obviously work. ⇒ **supervised deep credit stays PARKED; commit to the UNSUPERVISED stream cortex (rec #3).** See `2026-07-17-rate-net-control-graded-coding-does-NOT-unlock-supervised-deep-credit-block-is-the-rule.md`.**

## Bottom line

The 2026-07-17 not-GO is **not a surprise and not a dead end** — it hardens a three-ways-confirmed conclusion
(feedforward deep credit ≈ reservoir; recurrent diagonal e-prop's language win refuted; off-diagonal fragile-on-spikes).
**NP is not the escape hatch; the record already retired it.** The escape hatch is (a) one cheap positive control that
tells us whether the shared readout wall is the substrate or the rule, and (b) the unsupervised stream cortex that is
already learning on-spike without any of this. **The roadmap's "NP next" line is the stale bit and should be replaced.**
