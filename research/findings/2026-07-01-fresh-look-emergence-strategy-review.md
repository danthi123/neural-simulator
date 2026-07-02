# Fresh-look strategic review — the dendrite/emergence bet vs the communication goal (2026-07-01)

> **Type:** READ-ONLY deep-research + adversarial synthesis (a multi-agent workflow: 4 parallel literature threads → synthesis → 3-lens adversarial panel → final). NO code, NO experiments, NO `sim/` edit. Triggered by the owner's request for "a fresh look at project state, goals, and fresh deep research in relation to them" after pausing the autonomous EMERGE arc. Every load-bearing claim trust-but-verified by the controller against real citations; toy-scale / bleeding-edge-unverified flagged where that is the truth. This is a **decision doc, not a build commitment.**

## 0. Why this review

The prior (autonomous) session confirmed, at TOY/numpy scale, that deep biological credit assignment works (Burstprop GO 0.796; Sacramento-Senn microcircuit GO 0.961; interneuron self-organizes; two-compartment burst multiplexing GO R²=0.94) and was driving hard toward a spiking-substrate build. The owner paused to re-examine direction. This review stress-tests the *strategy* — not the science, which is sound — against fresh 2024-2026 literature and the actual goal (an emergent artificial lifeform the owner can **communicate** with, on one RTX 3090, biology-only).

One research thread (data/curriculum) died to a transient rate-limit; its territory was independently covered by the compute thread (BabyLM + Scaling-Data-Constrained-LMs), so the "data is the wall" claim is not left unsupported.

## 1. What the fresh evidence CONFIRMED vs CHANGED (verified citations)

| Project assumption | Fresh verdict | Evidence |
|---|---|---|
| "Bio-plausible deep learning doesn't scale to hard problems" | **OVERSTATED as a blanket** — the depth-instability half is solved | µPC trains 100-128-layer predictive-coding nets (Innocenti et al., arXiv:2505.13124, 2025); EqProp reached full ImageNet within ~1.5% of backprop; deep PC ties BP on Tiny-ImageNet |
| …but the part that matters for THIS project | **CONFIRMED** | Every hard-task success rides on **conv weight-sharing + hand-crafted per-depth precision schedules** — the exact non-emergent scaffolding the mandate forbids (Pogodin NeurIPS 2021, arXiv:2106.13031); the **burst family is the least-scaled** (2025 neuromorphic Burstprop still MNIST-only, depth *hurts* 97.2%→95.9%, Que-Naud); test-gap widens with depth; 5-18× compute |
| Dendrite is NECESSARY-but-NOT-SUFFICIENT; honest outcome = "characterized toy partial," not a conversationalist | **CONFIRMED (if anything understated)** | single-neuron NMDA dendrite ≈ 5-8-layer net (Beniaguev, Neuron 2021); no local/emergent spiking model produces open language at any scale |
| "Sequence generation is a SEPARATE non-dendrite lever" | **OUTDATED** — burst/dendritic credit assignment **already extends to recurrent sequence learning** | Capone et al. burst-*target* recurrent trajectory store/recall + hierarchical imitation (ICML 2022 arXiv:2201.11717; PLoS CB 2022; PNAS 2023); Bouhadjar dendritic-AP high-order sequences unsupervised (PLoS CB 2022, PMID 35727857); e-prop (Bellec, Nat Comm 2020); Senn NLA real-time/recurrent (eLife 2024) |
| Compute is the binding wall (owner need #2) | **OVERSTATED — compute is comfortable** | GeNN procedural connectivity: 4M neurons / 24B synapses on ONE GPU; ~1M point neurons near real-time on a 2080 Ti (Knight & Nowotny, Nat Comp Sci 2021); dendritic *principle* capturable at **~1% overhead** via a dendritic-abstraction unit (Dit-CNN, NeurIPS 2024) — the ~10× is only for full multi-compartment biophysics |
| Data/experience-stream richness is the true wall (owner need #3) | **CONFIRMED as the binding constraint** | data is "the true limiting factor," children reach competence on ~100M words (Muennighoff, Scaling Data-Constrained LMs, JMLR 2025, arXiv:2305.16264; BabyLM 2025, arXiv:2504.08165) |
| Fluency/generation is a separate lever from cognition | **CONFIRMED** | SpikingBrain-1.0 (7B/76B) reaches Transformer-comparable LM at ~2% data but via backprop on hundreds of GPUs — orthogonal to local dendritic emergence (Pan et al., arXiv:2509.05276, 2025) |

**Net:** the fresh evidence kills the *blanket* "can't scale" fear but **confirms every operational limit that actually gates this project.** The risk to a months-scale Burstprop substrate build is essentially unchanged; the two genuinely new facts are (a) **sequence learning folds into the dendritic family** (so a static substrate is not the natural first target) and (b) **compute is not the wall — data + getting-more-brain-into-the-loop is.**

*Controller caveat:* the 2026 "depth-utility / reference-validity" methodology papers (arXiv:2606.21126, 2606.06539) could not be individually verified; the methodological guard they motivate (held-out accuracy can mask dead deep layers → add a representation-emergence probe as a *gating* criterion) is sound regardless and echoes the project's own 2026-05-14 retraction history.

## 2. The three adversarial lenses (each right about a different thing)

- **Premature-build critic:** committing months to the static full-biophysics build now is premature — *no branch of the decision tree reaches it as the correct next protected edit* (a cheap NO-GO kills it; a cheap GO still routes through a cheaper abstraction first). The **stream-richness test** — feed the already-emergent shallow PPMI cortex a 10-100× richer stream and measure whether the *working console* gets more communicable — is the highest-information, near-zero-cost experiment and is currently buried.
- **Steelman-build critic:** de-risk from the most-confirmed asset outward, one variable at a time. The confirmed asset is *static Burstprop, rate*; the clean single-variable step is *static Burstprop, spiking* (a small net reproducing the confirmed toy result on the real substrate). A NEGATIVE there is **clean and build-saving**; a NEGATIVE on a recurrent-sequence probe (two variables changed on an unconfirmed rule) is **uninterpretable**. Stream work is orthogonal → parallel, not instead-of.
- **Goal-alignment critic (the sharpest):** the substrate is the *least-leverage quarter* for communication on the evidence, and even a clean substrate GO does not move the north-star. The reachable high-leverage arc is **"the emergent brain drives MORE of the existing working communication loop, on a richer stream, with cheap biological gating"** — routing via basal-ganglia disinhibition, fact-selection via the already-GO DA-gated recall vigor, grounding from a richer stream — with no new substrate. **Key reframe:** the minimized-transformer-for-fluency + brain-for-cognition split is **biologically defensible** (language production recruits specialized, developmentally-canalized circuitry — Broca/arcuate — the emergent brain *drives*, it does not regrow), **not a shortcut to eventually delete.**

## 3. Controller judgment (trust-but-verified)

I agree with the convergent core and weight it toward the goal-alignment lens:

1. **Do NOT start a months-scale substrate commit now.** All three lenses + the final agree. (Nuance the review slightly missed: the scoping doc's "Stage A" single-neuron multiplexing is **already GO** as of EMERGE-4 — R²=0.936, event~basal 0.999, burst-prob~apical 0.977, P0=0.030. So the substrate track's *next* step is Stage B, the small spiking net — not Stage A.)
2. **The highest-leverage, most goal-aligned, most reachable move is NOT the substrate — it is getting more of the emergent brain into the working communication loop, on a richer stream.** This is cheap, keeps a working demo live throughout, burns down real hand-designed shortcuts (the intent dispatcher, host fact-selection), and directly moves communication. The dendrite is necessary-not-sufficient and gated behind data anyway.
3. **The substrate/dendrite science should continue as a cheap, parallel, gated de-risk — not the headline.** If we probe it, the disciplined first step is the static spiking-net confirmation (single-variable, interpretable negative), with a representation-emergence probe as a *gating* criterion; the recurrent-sequence probe (a new-mechanism de-risk, Capone/Bouhadjar) follows only on GO. Full biophysics stays double-gated.
4. **The central, directive-touching question is the owner's:** is the transformer-fluency faculty a *permanent, biologically-honest faculty split* (per the goal-alignment reframe), or a *temporary scaffold to eventually replace* with a fully-emergent generator (per the master directive's fully-spiking end-state)? The evidence favors the former as honest + reachable, but this touches the directive's core and is the owner's call — it determines whether the next arc leads with communication (accept the split) or with substrate emergence (push the fully-spiking end-state).

## 4. Recommended next arc (my recommendation)

**Lead track — "more emergent brain in the working loop, on a richer stream" (goal-aligned, cheap, no `sim/` risk):**
- The north-star test: feed the emergent PPMI stream-cortex a materially richer/longer/structured stream; measure whether the *working console* gets more communicable (more groundable facts, better generalization). GO ⇒ tilt effort here; plateau ⇒ the wall is representational depth, and the dendrite bet is *earned by evidence*.
- Burn down hand-designed loop decisions with already-built biology: BG-disinhibition / `transmission_gate` as the intent dispatcher (Stocco conditional routing); DA-gated recall vigor (already GO) for fact-selection.

**Parallel science track — cheap, gated substrate de-risk (NOT a months commit):**
- Stage B: a small spiking Burstprop net reproducing the confirmed EMERGE-1 depth-2 result on the real substrate, single-variable, with anti-cheats + a representation-emergence gate. GO ⇒ the recurrent-sequence probe (Capone/Bouhadjar) as a new-mechanism de-risk. Full biophysics (scale) stays behind both gates.

**Success bar (locked):** more communicable capability in the working loop + (separately) a characterized substrate-emergence result — explicitly NOT "fully-emergent open conversation," which no one in the field has at any scale.

## 5. Sources
Workflow raw output (4 threads + 3 critics + final) in session scratchpad; key citations verified inline above. Grounding: `2026-07-01-dendritic-cortex-for-emergence-scoping.md`, `2026-07-01-spiking-burst-substrate-scoping.md`, `2026-07-01-emerge1b-burstprop-MECHANISM-CONFIRMED-partial.md`, EMERGE-4 result (`raw/_emerge4_burst_multiplexing.json`).

_Fresh-look deliverable. The central fork (fluency-faculty-split as permanent vs shortcut; lead with communication-loop vs substrate-emergence) is the owner's to steer — it touches the master directive's core._
