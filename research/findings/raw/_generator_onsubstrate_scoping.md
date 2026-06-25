# Purity backlog #8 — the "spiking generator" on-substrate scoping (READ-ONLY research gate)

**Date:** 2026-06-25. **Scope:** read-only deep-research + catalog review per the standing practice. NO edits, NO runs, NO webapp. **Item:** biology-fidelity audit class-b #7 + the "spiking generator" overclaim (audit row **M10**). This is the DEEPEST/most-frontier purity item — it overlaps the host-designed-structure deep frontier. The brief: scope whether the generator's host-weight forward can become genuine on-substrate dynamical spiking, or characterize it precisely as a deep boundary.

**Bottom line up front (the verdict, stated first):** #8 is **TWO residuals wearing one label, and they split cleanly.**
- The **FORWARD** (the matvec + the nonlinearities) is **on-substrate-CLOSEABLE and largely already closed** — but it is NOT closed by "more dynamical spiking." The honest framing is the opposite: the substrate's dynamical RF machinery is deliberately run in a **degenerate linear regime** (λ=0, ω≈0) so it computes `a@W` exactly, and the nonlinearities are graded reads. Making the matvec "more dynamically spiking" (oscillatory phasor coding, real first-passage spikes) is **possible but BUYS NOTHING and COSTS fidelity** — it re-introduces exactly the rate-code/quantization walls the linear-accumulator escape was designed to avoid. The defensible close for the forward is **composition + an honest relabel**, not added dynamics.
- The **WEIGHTS** are the genuine deep residual. They are the ANN's backprop-distilled host weights, injected via `rf_set_complex_weights`. This is **identical in kind to the FHRR bind-structure residual** (memory `feedback_spiking_structure_must_self_organize`) and the C2 develop-loop's host-orchestrated training: **host-DESIGNED structure of a spiking op.** Closing it = developmental/on-substrate self-organization of the generator weights, which is the **deep frontier**, months-scale, and overlaps the already-deferred H-2 / learned-cortex problem.

So: **the "spiking generator on the bridge" language overstates on BOTH axes, but the two axes have opposite verdicts.** The forward is cheaply relabel-and-compose-closeable; the weights are a genuine deep boundary co-extensive with the project's deepest open problem. The precise residual is **the host-distilled weights, not the forward.**

---

## (1) DIAGNOSIS — exactly what the generator's host-weight forward does, and which bytes are host

### The claim under audit
"The spiking generator generates novel text on the bridge" (the C1 + fully-spiking-C1 + generative-loop arc:
`2026-06-23-generative-loop-DEMONSTRATED.md`, `-spiking-{softmax,gelu,layernorm}-GO.md`, the `_genseq_loopstep3_fullblock_rf_derisk.py` integration). Audit row M10 flagged: "the on-bridge forward is HOST-WEIGHT matvecs (a@W computed exactly) on a conductance-free RF accumulator + host nonlinearity reads — NOT dynamical spiking neurons generating text."

### What the forward ACTUALLY does (confirmed by reading the code, not the finding)

The integration runner is `research/runners/_genseq_loopstep3_fullblock_rf_derisk.py`. It runs one real Gen-F transformer block (`sim/tiny_transformer.py _Block.forward`) on real tokenized TinyStories activations. The forward has exactly two kinds of operation:

**(a) Every LEARNED-WEIGHT matvec → the RF complex-synapse "exact linear accumulator."**
The primitive is `rf_linear_layer_signed` (`_genseq_loopstep3_rf_probe.py:116-138`). It:
1. installs the real weight `W` as complex synapses (`rf_set_complex_weights`, `W_im=0`),
2. kicks `z_in = a_in` (REAL, magnitude = activation, phase 0),
3. runs `rf_resonate_steps(nsteps)` with **`RF_LAMBDA = 0.0`** (no magnitude decay) and **`RF_PERIOD = 100000`** so **`ω = 2π/period ≈ 0`** (the rotation per step ≈ identity),
4. reads `Re(Z_out)/nsteps`.

I verified against the substrate (`sim/bridge.py:5719 _rf_advance_one`): the genuine RF step is `Z ← decay·rot(Z) + W·z` with an Im-zero-crossing spike detector. With `decay=exp(0)=1` and `rot≈identity`, this collapses to a **pure running sum of the matvec**: after `nsteps`, `Re(Z) = nsteps·(a@W)` exactly. The runner MEASURES this: `max|Re(Z)/nsteps − a@W| ≈ 7e-8` (rank 1.000). **No clip, no `g·(V−E)` conductance, no refractory ceiling, no spike actually used in the readout.** The Im-zero-crossing spike detector still runs but its output (`rf_spike_step`) is discarded — the readout reads the analog membrane `Re(Z)`.

⇒ **The substrate is reused as an exact linear matmul accumulator.** It is "on the bridge" (real `SimulationBridge`, real `cp_membrane_potential_v`, real `_run_one_simulation_step`/`rf_resonate_steps` machinery) but it is NOT "dynamical spiking neurons computing": the dynamics are deliberately nulled to identity so the accumulator is a bit-exact GEMV.

**(b) Every PARAMETER-FREE nonlinearity (softmax, GELU, LayerNorm) → a host "faithful read."**
In the integration runner these are plain numpy: `_layernorm(...)`, `gelu_exact(...)`, `np.exp`/normalize for softmax (`_genseq_loopstep3_fullblock_rf_derisk.py:315, 332, 347`). The two residual adds and the biases ride on the host read.

**The result:** full-block output fidelity **spearman ~1.000 / cosine ~1.000** vs the exact-float teacher, with rigorous anti-cheats (shuffled-target collapses; load-bearing lesion — scramble the RF weights — collapses the block to the residual floor, proving the RF matvecs carry the ~95%-of-norm sublayer corrections, NOT the host nonlinearities). So the RESULT is real and anti-cheated; the question is purely about substrate-fidelity of the mechanism.

### Isolating the genuine residual — which exact bytes are host

| Byte | What it is | Host? | On-substrate? |
|---|---|---|---|
| **The weights `W` (Q/K/V/O + W1/W2 = 786,432 params)** | backprop-distilled ANN weights, loaded from `generator_f_gate.ckpt.s42.real.pt`, injected via `rf_set_complex_weights` | **YES — host-DESIGNED structure** | NO — host-computed + injected (NOT learned/self-organized on the bridge) |
| **The matvec `a@W`** | the RF complex accumulator (`_rf_advance_one`) with λ=0, ω≈0 → exact running sum | runs ON the bridge | YES (substrate), but in a **degenerate linear regime** (the dynamics are nulled) |
| **softmax / GELU / LayerNorm** | host numpy in the *integration* runner; **separately** validated as graded-read spiking ops (M11) | host in the integration; **closeable** | the spiking versions exist standalone (see below) but are NOT composed into the full block |
| **biases, LN affine, residual adds** | per-feature scale/shift + adds on the read | host (rides on read) | (a)-legitimate per the BRAIN-BASED-ONLY standard (no cross-feature mixing) |
| **the spike output (`rf_spike_step`)** | computed but **discarded** in the readout | n/a | computed, unused |

**The crucial nuance the audit row gets right:** "host-weight matvecs on the RF accumulator + host nonlinearity reads" is accurate, but it conflates two residuals of OPPOSITE severity:
- the **forward** (matvec regime + nonlinearity reads) — cheaply closeable / largely closed (see §2),
- the **weights** — the genuine deep residual (the host-designed structure).

The "0 novel content" measurement (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`) is a SEPARATE gap (the *retrieval composer* can't free-generate); the generator (this item) is the ANSWER to that gap — a learned generative-sequence model. But the generator's learning is backprop-off-substrate, which is exactly the weights residual here.

---

## (2) REFRAME via biology — can the host-weight matvec become genuine on-substrate dynamical spiking? Is the residual the WEIGHTS or the FORWARD?

### Catalog + literature grounding (what the catalog actually has)
From the read-only catalog review (`E:\Documents\Projects\sim-catalog\references\feature-catalog.md`, clusters A–Q):
- **NO dedicated FHRR/phasor-neuron entry**, **NO transformer/attention/neural-Turing entry**, **NO rate-code-wall entry** (the rate-code wall is a project finding, not a catalog entry).
- Relevant biology that DOES exist: **N.15 theta-gamma multiplex** (Lisman-Idiart 1995: each gamma cycle carries one item-assembly — a phase/time-multiplexed buffer); **N.19 gamma binding-by-synchrony** (ING/PING); **G.09 imagination/generative replay** (Schacter-Addis-Buckner constructive recombination); **F.02 granule-PF expansion recoding** (Marr-Albus = random-feature reservoir); **D.05 CA3 recurrent autoassociative attractor** (Marr 1971); **G.07/H.19 pre-SMA/SMA internally-generated serial-order**; **D.24 theta-paced sequence compression**.

**The biology verdict on "make the matvec dynamically spiking":** real cortex does NOT compute a 256×256 magnitude-coded GEMV via dynamical spiking and then read a precise analog magnitude. The closest biological substrates for *generation* are **recurrent attractor dynamics** (CA3, D.05), **constructive recombination** (G.09), and **phase-multiplexed assemblies** (N.15). NONE of these is a feedforward transformer block. So "the transformer block IS biology realized in dynamical spikes" is not a claim biology supports — the transformer block is an **engineering fluency faculty** (the same category as the off-bridge Qwen, M12, owner-sanctioned), and the on-substrate question for IT is narrower: *can its arithmetic run faithfully on the substrate?* — which is the forward question, answered yes.

### Is the residual the WEIGHTS or the FORWARD? — they split, with OPPOSITE verdicts

**FORWARD = on-substrate-able, AND making it "more dynamically spiking" is the WRONG move.**
- The matvec already runs on the bridge; the only "impurity" is that the dynamics are nulled to identity. The owner's host-structure standard is about WEIGHTS/connectivity self-organizing, not about whether a given op uses oscillatory vs linear dynamics. A linear accumulator IS a legitimate spiking-substrate op (it is literally the RF complex-synapse matvec the production composer uses for bind/unbind — `bridge.py:5691`, the SHIPPED FHRR path).
- Could it be made oscillatory/phasor-dynamical (ω≠0, info in phase, real first-passage spikes)? **Yes** — the RF probe even tests the phase channel (`rf_linear_layer_phase`, secondary readout). But the rf-probe + rf-distill findings show the phase/quantized-spike channel REINTRODUCES the rate-code wall: the cumulative-rank through stacked layers drops (the rate/graded path got cumulative **0.288**; the linear-accumulator escape is what reached 1.000). Per the standing rate-code-wall finding (Mikulasch-Priesemann point-neuron limit), magnitude-coded multi-layer arithmetic in spike-rate/phase codes loses precision — exactly the wall the linear-accumulator regime was DESIGNED to escape. So "add dynamical spiking to the matvec" is a step BACKWARD in fidelity, for no purity gain (the weights are still host).
- The **nonlinearities** are the genuinely-interesting forward sub-question. Per **M11** the spiking realizations are **GO standalone on the same real Gen-F block, on a live GPU bridge inside `_run_one_simulation_step`**: LayerNorm 0.962 (via the shipped `enable_input_mean_adapt` subtractive + `enable_input_divisive_norm` divisive circuits, `bridge.py:6238/6190`), GELU 0.991 (21-knot rectified-basis graded read), softmax 0.9998 (calibrated graded `exp` + divisive-norm denominator). **Honest caveat the softmax finding records:** these are themselves **graded/calibrated reads**, not first-passage dynamical spikes either, and a LOW-TEMPERATURE softmax (logit `exp` dynamic range >1e4) WOULD hit the genuine rate-code wall and need a native spiking-`exp`/log-domain primitive — the trained Gen-F simply doesn't reach that regime. So the nonlinearities are "spiking" in the same graded-read sense as the rest of the project's spiking ops (the divisive-norm circuit is real shipped `sim/` machinery), with one characterized boundary (low-temp softmax) that doesn't bite this model.
- **THE forward's actual gap (cheap, real):** M10 ran the matvec-on-RF with HOST nonlinearities; M11 validated the spiking nonlinearities SEPARATELY. **They were never composed into ONE full-block forward.** That composition (every matvec on RF + every nonlinearity through the spiking circuits, one block, end-to-end) is the cheap, missing piece that would let "fully-spiking generator block on the bridge" be said WITHOUT the M10 host-nonlinearity caveat. It is reuse-by-import, no `sim/` edit, GPU.

**WEIGHTS = the genuine deep residual (host-designed structure).**
- The 786,432 weights are backprop-trained off-substrate (PyTorch, `sim/bptt_snn*.py` / the Gen-F training) and injected. This is **categorically the same residual** as:
  - the FHRR bind-structure (`feedback_spiking_structure_must_self_organize`: "spiking at RUNTIME, host-DESIGNED at the STRUCTURAL level"),
  - the consolidated-ANN weights (audit H-2),
  - the C2 develop-loop's host-orchestrated fine-tune (`2026-06-23-generative-loop-DEMONSTRATED.md`: the train→grow→no-forget loop runs the fine-tune in PyTorch with a host replay mixer; the self-replay no-forget mechanism is biologically-motivated CLS, but the weight UPDATE is host backprop, not on-substrate plasticity).
- Biology's answer (per the host-structure memory + the catalog): structure emerges **developmentally** (genome wiring rules + activity-dependent refinement) or via **on-substrate local plasticity** (Hebbian/three-factor/STDP), NOT host backprop. The project HAS on-substrate learning that works (M1 stream cortex: corr(M,C) 0.894, codes self-organize from listening; the limbic critic learns V via three-factor STDP×DA). But those learn *codes/values*, not a 786K-param generative transformer. **Backprop-trained generative-sequence weights on the point-neuron substrate via local learning is an OPEN, deep problem** — and the SOTA (SpikingBrain/SpikeGPT/SpikeLLM, per the generation-gap finding) ALL train by backprop on ~150B tokens then convert, i.e. nobody has the on-substrate-learned version. This is genuinely months-scale and overlaps H-2 / the learned-cortex frontier.

---

## (3) RANK cheap-first options (vs the deep host-structure boundary)

Ordered cheapest→deepest. The first two close the FORWARD overstatement honestly; the rest concern the WEIGHTS deep frontier.

**Option A (CHEAPEST, recommended) — RELABEL + compose the spiking nonlinearities into ONE full-block forward; characterize the weights as the deferred residual.** No new mechanism. (i) Run the M11 spiking softmax/GELU/LayerNorm + the M10 RF matvec in a SINGLE block forward end-to-end (reuse-by-import, the runners exist), so the "fully-spiking-C1" claim holds without the host-nonlinearity caveat. (ii) Relabel the headline in CLAUDE.md / findings: the generator's **forward arithmetic runs on the substrate** (matvec on the RF complex-synapse accumulator in its exact-linear regime + nonlinearities via the shipped graded/divisive-norm circuits); the **weights are host-distilled (the deferred host-structure residual, = H-2 / the FHRR-structure class)**; the **fluency faculty is engineering, not a biology claim** (same category as the off-bridge LLM). Cost: one GPU run + a doc edit. Buys: the headline stops overstating; the residual is named precisely.

**Option B (CHEAP, honesty) — characterize the linear-accumulator regime as a DELIBERATE escape, and the low-temp-softmax boundary precisely.** Document that the RF dynamics are nulled to identity ON PURPOSE (the linear accumulator escapes the rate-code/clip walls that defeated 4 prior rate/graded/popcode/distill attempts at cumulative ~0.288), and that "adding dynamical spiking" to the matvec is a fidelity regression, not a purity gain. Add the one measured boundary (low-temperature softmax → native spiking-`exp` needed) to the boundary ledger. Cost: doc-only. Buys: pre-empts the "but it's not REALLY spiking dynamics" critique with the principled reason.

**Option C (MEDIUM, optional polish, only if a future model needs it) — native spiking-`exp` / log-domain primitive for low-temperature softmax.** Only bites if a future generator enters logit `exp` dynamic range >1e4. An expansive-f-I / log-domain circuit. Not on the critical path; flagged in the softmax finding.

**Option D (DEEP FRONTIER — the genuine residual) — on-substrate / developmental learning of the generator weights.** Replace host-backprop-distilled weights with weights that EMERGE from on-substrate local plasticity or developmental self-organization. This is the real close for #8's deep half. It is **co-extensive with H-2 (host-designed structure) + the learned-cortex frontier + the FHRR-bind-structure problem**, and the SOTA shows nobody has it (all spiking LMs are backprop-then-convert). Months-scale, high-variance, owner-gated. **NOT cheaply closeable.** Sub-paths (all open): (i) surrogate-gradient BPTT *on* the bridge dynamics (still "backprop", but on-substrate forward — partial); (ii) three-factor/Hebbian-only generative learning (the biology-faithful path — unproven for 786K-param generation, likely a deep negative on point neurons per the rate-code-wall family); (iii) developmental self-organization of the generative connectivity (the genome-wiring-rules path — the deepest, least-charted).

**Why D is correctly deferred (not a cheap fix masquerading):** it fires the research gate's "deep frontier" classification — it is the same family as the multiply-confirmed FHRR-bind-structure boundary and the explicitly-deferred H-2. The owner has flagged host-designed structure as the deepest categorical blocker with developmental self-organization as the genuine (far) target.

---

## (4) Anti-cheats + cheap-first de-risk + GO bars + VERDICT

### Anti-cheats (for Option A's compose-and-relabel — the only thing that would be BUILT)
1. **Composed-block fidelity vs exact-float teacher** — spearman/cosine of the all-spiking-forward (RF matvec + spiking softmax/GELU/LayerNorm) block output vs the exact-float Gen-F block, on real token activations. Must clear the M10/M11 bars (≥0.90, ideally ≥0.95).
2. **Load-bearing lesion** — scramble the RF weights → the block MUST collapse to the residual floor (proves the matvecs, not the nonlinearity circuits, carry the computation). (M10 already does this; re-assert in the composed run.)
3. **Shuffled-target / specificity margin** — the composed block output must be position-specific (matched ≫ mismatched), shuffled-target below real. (M10/M11 already do this.)
4. **No-`sim/`-edit assertion** — Option A is reuse-by-import; assert no protected-code change (the divisive-norm/mean-adapt circuits already shipped).
5. **The honesty anti-cheat (for the RELABEL)** — the doc must NOT claim the WEIGHTS are on-substrate; it must name them as the host-distilled deferred residual. The relabel is only honest if it explicitly cedes the weights.

### Cheap-first de-risk (the single decisive run)
Compose the spiking nonlinearities + the RF matvec into one full-block forward (reuse `_genseq_loopstep3_fullblock_rf_derisk.py`'s teacher + the M11 spiking-op runners; swap the three host reads for the validated spiking circuits). Single GPU run, real Gen-F block-0, real TinyStories activations. ETA: minutes (the pieces are validated; this is composition). If it clears ≥0.90 with the lesion collapsing → the "fully-spiking-forward generator block on the bridge" claim is earned (weights still ceded).

### GO bars
- **Forward-closeable (Option A) = GO** iff the composed all-spiking-forward block ≥0.90 spearman vs exact-float, lesion collapses, shuffled below real — AND the relabel explicitly cedes the weights as the deferred host-structure residual.
- **Forward = PARTIAL** if the composed block lands 0.7–0.9 (report which nonlinearity-circuit accumulates the error — most likely the L1-vs-RMS LayerNorm gap, +0.037, or a low-temp softmax tail).
- **Weights (Option D) = genuine DEEP BOUNDARY** (not a pending build): on-substrate/developmental learning of a 786K-param generative transformer is unproven anywhere, fires the deep-frontier gate, months-scale, owner-call.

### VERDICT

**#8 is BOTH closeable AND a deep boundary — because it is two residuals, and the audit's single label "spiking generator" conflated them. The precise split:**

1. **The FORWARD (matvec + nonlinearities) — CLOSEABLE-ON-SUBSTRATE, and largely already closed.** The matvec runs on the bridge (in a deliberate exact-linear RF regime that ESCAPES the rate-code wall — making it "more dynamically spiking" is a fidelity regression, not a purity gain); the nonlinearities are GO standalone via shipped spiking circuits. The ONLY genuine gap is **composing them into one block forward** (Option A, a cheap reuse-by-import run) + an **honest relabel** (Option B). The forward is NOT the deep residual.

2. **The WEIGHTS — a GENUINE DEEP BOUNDARY, the precise residual.** The 786,432 weights are host-backprop-distilled and injected. This is **categorically the same residual as the FHRR bind-structure + H-2 + the C2 host-orchestrated fine-tune** — host-DESIGNED structure of a spiking op. Closing it = on-substrate/developmental learning of a generative transformer, which is **unproven anywhere (SOTA spiking LMs all backprop-then-convert)**, months-scale, overlaps the learned-cortex deep frontier, and is correctly owner-deferred.

**The precise residual (the one sentence):** *the generator's host residual is the backprop-distilled WEIGHTS (a host-designed-structure / H-2-class deep boundary), NOT the forward — the forward's arithmetic is on-substrate (matvec on the RF complex-synapse accumulator + shipped spiking-nonlinearity circuits) and is cheaply finishable by composing the already-GO pieces into one block and relabeling the headline to cede the weights.*

**Recommendation:** do Option A+B (cheap compose + honest relabel — closes the OVERSTATEMENT, the actual purity-backlog complaint) now; log Option D (on-substrate generative-weight learning) as a named deep-frontier boundary alongside H-2 / FHRR-structure / learned-cortex (do NOT scope a build — it fires the deep-frontier gate and is the owner's call). Option C (spiking-`exp` for low-temp softmax) only if a future model needs it.

---

## Files read (provenance)
- `research/findings/raw/_biology_fidelity_audit_2026-06-24.md` (M10/M11, class-b #7)
- `research/runners/_genseq_loopstep3_fullblock_rf_derisk.py` (the integration — host nonlinearities + RF matvec)
- `research/runners/_genseq_loopstep3_rf_probe.py` (`rf_linear_layer_signed`/`_phase`, λ=0/ω≈0 exact-linear regime)
- `research/runners/_genseq_loopstep3_rf_distill_derisk.py` (the clip-readout residual; RF-no-g(V−E) escape)
- `sim/bridge.py:5646-5800` (`rf_kick`/`rf_set_complex_weights`/`_rf_advance_one`/`rf_resonate_steps` — the genuine dynamical RF step that the generator nulls to identity)
- `research/findings/2026-06-23-spiking-{softmax,layernorm}-GO.md` (M11 — graded-read spiking nonlinearities, GO standalone, low-temp-softmax boundary)
- `research/findings/2026-06-23-generative-loop-DEMONSTRATED.md` (C1+C2; the host-orchestrated fine-tune)
- `research/findings/2026-06-22-generation-novelty-categorical-gap-MEASURED.md` (the SEPARATE retrieval-composer 0-novel gap the generator answers)
- `E:\Documents\Projects\sim-catalog\references\feature-catalog.md` (clusters A–Q; N.15/N.19/G.09/F.02/D.05/G.07/H.19; no FHRR/transformer/rate-code-wall entry)
- memory `feedback_spiking_structure_must_self_organize` (the host-designed-structure standard)
