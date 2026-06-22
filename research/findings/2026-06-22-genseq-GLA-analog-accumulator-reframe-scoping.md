# Generative-sequence frontier (Spine A) — does the GLA / ANALOG-ACCUMULATOR reframe SIDESTEP the per-layer clip-compression wall CHEAPLY? — scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings re-verification, OWNER-CHOSEN cheap-first check BEFORE committing
> to the multi-week differentiable-bridge `sim/` edit (the option (a) the distill-NEGATIVE routed to). **NO `sim/` edits,
> NO experiments, NO GPU.** Single deliverable = this doc. Every load-bearing claim re-verified against the repo
> (file:line). The controller should trust-but-verify the **[VERIFY]** items, push, and present before building. Builds
> on — does not re-derive — the four NEGATIVE findings (`2026-06-22-genseq-loopstep3-{multilayer,popcode,distill}-*`,
> `-graded-PARTIAL-*`), the two prior scopings (`-consolidation-scoping.md`, `-past-saturation-scoping.md`), and the
> surrogate-grad scoping (`2026-06-22-surrogate-grad-on-bridge-scoping.md`).

---

## 0. One-paragraph answer (the rest is the evidence)

**The honest verdict: GLA / analog-accumulator does NOT cheaply sidestep the two measured walls — for the *attention*
sub-op it is a real (but no-cheaper) `sim/` edit; for the *MLP* (the MEASURED wall) it is a NO-OP reframe that lands on
the SAME `g·(V−E)` divergence that killed the distillation. The GLA reframe is a category-confusion at the point that
matters.** The two walls are (W1) deterministic per-layer signal compression through stacked `clip(a@W,0,1)` graded
readouts (`0.846→0.620→0.288`, popcode-NEGATIVE, std `0.00e+00`), and (W2) the trained-weight install does not survive
the live `g·(V−E)`+AdEx dynamics (offline 0.815 → installed 0.444, distill-NEGATIVE — the `[VERIFY]` driving-force gap
confirmed load-bearing). **The decisive fact, verified to source:** *every* synaptic projection on the bridge — spike,
**graded**, NMDA-recurrent, coincidence, GABA_B — lands in `cp_conductance_g_e/g_i` and reads out as `I = g·(E−v)`
(`fused_conductance_decay_and_current`, `kernels.py:214`; the graded path's `clip` is on the SOURCE's `a_cont` but the
TARGET current is still `g·(E−v)`, `bridge.py:6169/6175`). There is **NO current-based (`I` directly, no `g·(V−E)`)
accumulation path** the bridge already supports — the only `I=`-assignments are the static `cp_external_input_current`
and the OU/experiment drives (`bridge.py:6177-6570`), none a `Wᵀ@activity` matvec. **So Q3's hoped-for escape (an
existing current-based accumulator that avoids `g·(V−E)`) does not exist for a rate/graded code.** The one genuine
exception is the **RF resonate-and-fire complex-state path** (`_rf_advance_one`, `bridge.py:5710-5746`): a complex matvec
added DIRECTLY to the complex state `Z` — *no clip, no `g·(V−E)`, no refractory ceiling* — which is the project's proven
linear/multiplicative accumulator. **GLA's value is therefore real but DIFFERENT from "cheap MLP fix":** (i) for the MLP
it reduces to graded-transmission, which is exactly W1+W2, so it adds nothing the four NEGATIVEs didn't already measure;
(ii) for *attention* a GLA running-state accumulator IS a justified additive `sim/` step-block (the consolidation
scoping's ranked option #1) — but it is the SAME cost class as, and is downstream of, the differentiable-bridge edit, and
it does NOT verbatim-reuse Gen-F's softmax weights (it re-architects attention → needs the very finetune the edit
provides). **Verdict: GLA does NOT undercut the differentiable-bridge edit on cost for the measured (MLP) wall; it is the
right *attention* operator AFTER that edit exists, not a way to avoid it. The one thing that genuinely sidesteps `g·(V−E)`
is the RF complex accumulator — a much bigger pivot (re-encode the generator as phasors), parked, not cheap.**

---

## 1. Re-state the two walls EXACTLY (so "sidestep" is judged against the real residual)

Both NEGATIVEs are mechanistically pinned; the GLA reframe must beat BOTH or it does not sidestep the wall.

### W1 — deterministic per-layer `clip` compression (the MLP wall, `2026-06-22-...popcode-NEGATIVE...`)
- The graded readout `a_cont = clip((v−rest)/scale, 0, 1)` (`bridge.py:6144-6147`) **un-saturates** the 0.5
  spike-rate ceiling (graded-PARTIAL: L0 0.32→0.865) — W1 is NOT the rate ceiling (that was de-risk #1, resolved).
- But the cumulative analog-Spearman **deterministically compresses** `0.846 → 0.620 → 0.288` across 3 stacked dense
  blocks, and **population coding is a literal no-op** (FLAT in n_per; within-population std measured `0.00e+00` —
  the graded `a_cont` is noiseless, so there is nothing to average). The loss is *signal compression through the
  stacked saturating `clip`*, NOT noise. (`popcode-NEGATIVE` lines 3, 9-13, 19-22.)

### W2 — the `g·(V−E)` install divergence (the distillation killer, `2026-06-22-...distill-NEGATIVE...`)
- The clip-aware layerwise distillation **recovers the teacher OFFLINE** in the idealized `clip(a@W,0,1)` chain
  (cumulative 0.872 pure-clip / 0.815 df-aware vs verbatim 0.288) — W1 IS trainable-around in the ideal forward.
- But the trained weights **do NOT survive the live-bridge install**: installed cumulative 0.307 (pure-clip) / 0.444
  (df-aware), both ≪ the 0.8 GO bar. The driving-force-aware ARM2 lifting 0.307→0.444 **confirms** the residual IS
  the `g·(V−E)` conductance non-linearity + AdEx settling + per-block gain — the idealized `tanh`-squash proxy doesn't
  capture the live dynamics. (`distill-NEGATIVE` lines 14-28; the `[VERIFY] g·(V−E)` gap, now confirmed load-bearing.)
- (Secondary, carried forward: the final-block teacher reps are cross-char-correlated → the shuffled-target control
  reached 0.542, a metric confound; not the load-bearing wall but a real readout-objective gap.)

**⇒ The genuine residual is W2: the live `g·(V−E)`+AdEx dynamics diverge from any idealized differentiable forward.**
W1 is already trainable-around offline; what fails is that the bridge's *actual* current law is not `clip(a@W,0,1)`.
Any "sidestep" must reach the `g·(V−E)` law itself — either avoid it, or train through the real version of it.

---

## 2. Q1 — what IS the GLA / analog-accumulator reframe ON THE BRIDGE? (and does it preserve per-layer rank?)

**The reframe (SOTA, verified):** the only spiking LM at scale (SpikingBrain-7B, arXiv 2509.05276) does NOT run dense
per-token matmuls as saturating spike/graded layers — it keeps the *linear* ops as a **running-state accumulator**
`S_t = diag(g_t)⊙S_{t-1} + k_tᵀv_t`, `o_t = q_t S_t` (Gated Linear Attention; Katharopoulos arXiv 2006.16236), a
slowly-updated buffer read as analog STATE, reserving spikes/threshold only for the genuinely-nonlinear gating. The
prior past-saturation scoping (§2(e)) flagged this as "the deepest correct framing": **read analog state for linear ops,
spikes for gating.**

**What it IS on THIS bridge — and the decisive distinction the reframe blurs:**

| GLA component | Closest bridge mechanism (verified) | Does it avoid the walls? |
|---|---|---|
| The running-state buffer `S_t` (decaying accumulator) | **`cp_conductance_g_nmda_recurrent`** — a slow (tau~100ms) dual-exp conductance buffer, incremented by a restricted matvec, persists across ticks (`bridge.py:6326-6349`, `1406-1408`, `1820-1823`) | **NO** — the increment is **spike-driven** (`_nr_mat.T @ cp_prev_firing_states`, `:6336`) AND the readout is `g·(E_NMDA−v)` (`fused_nmda_update_and_current`, `:6343-6347`). Same W2 `g·(V−E)`; inputs still hit the 0.5 spike ceiling. |
| The linear read `o_t = q_t S_t` (analog state readout) | The graded path's `a_cont` read | **NO** — that read is the `clip(0,1)` of W1, and the downstream current is still `g·(V−E)` (W2). |
| The "reserve spikes for gating" principle | The bridge's spike threshold + transmission-gate | Conceptually aligned, but does not change the current law for the linear half. |
| A TRUE non-`g·(V−E)`, non-clip linear accumulator | **`_rf_advance_one`** complex-state: `re_new = decay·rot(re,im) + (W_re@re − W_im@im)` — a complex matvec **added directly to the state Z** (`bridge.py:5710-5746`) | **YES** — no clip, no refractory ceiling, no `g·(V−E)`; info in PHASE. This is the project's proven linear/multiplicative accumulator (FHRR composer). But Gen-F is **rate-coded, not phasor** — using it = re-architecting the generator's representation (a much larger pivot, not cheap). |

**Does the GLA reframe preserve per-layer rank for the MLP? NO — because on the bridge the MLP's "analog accumulator"
IS the graded path, which IS W1.** The graded transmission already *is* "read the analog membrane for the linear op"
(`bridge.py:6128-6175` is literally that). The past-saturation scoping §2(e) said exactly this: "for the MLP, (a)
[graded] already realizes the reframe." So **applying GLA's reframe to the MLP yields graded transmission — and graded
transmission is precisely what compressed `0.846→0.620→0.288` (W1) and failed the install (W2).** The reframe does not
add a new mechanism for the MLP; it re-labels the one already measured NEGATIVE. **Q1 answer: GLA-on-bridge for the
*linear/MLP* ops = the graded/NMDA-recurrent accumulator, both of which read out through `clip` (W1) and/or `g·(V−E)`
(W2); it does NOT preserve per-layer rank any better. The only bridge path that genuinely escapes both is the RF complex
accumulator, which is a representation re-architecture, not a cheap re-read.**

---

## 3. Q2 — does it address the MLP clip-compression (the MEASURED wall), or only attention?

**It addresses ATTENTION, not the MLP — and the MLP is the measured wall.** The four NEGATIVEs were all on an **MLP-only
slice** (66→2048→2048→2048, NO attention — `multilayer-NEGATIVE` line 8; `popcode`/`distill` on the narrow-512 MLP). The
wall is the *stacked dense MLP*, which has GELU, not softmax. GLA is an **attention** reformulation — it says nothing
about how to carry a dense `Linear→GELU→Linear` through the substrate without the per-layer compression. So on the
literal MEASURED wall, GLA is off-target.

**Can the bridge represent GELU WITHOUT the rank-crushing `clip(0,1)`? Concrete + honest:**
- The W1 compression is the `clip` UPPER saturation at 1.0 (the graded `a_cont` is hard-clamped). GELU's soft saturation
  is one-sided-ish but unbounded above; a hard `clip(0,1)` per layer is a poorer match than the spike-rate was at the
  *bottom* but crushes the top.
- **A wider linear band** is already a knob: `graded_source_scale_mV` (the scale sweep in graded-PARTIAL: 20→0.327,
  40→0.027, 80→−0.196 — wider scale made it WORSE, because it pushed the operating point into a flatter region; scale=20
  was best). So "widen the band" was tried within the existing mechanism and does not fix the cumulative — the `clip`
  ceiling at exactly 1.0 is structural in `bridge.py:6144-6147` and is the compressor.
- **A two-sided ON/OFF representation** (split each feature into ON and OFF channels, like the signed E/I split for
  weights but for activations) is a real candidate to widen the effective dynamic range — BUT it is NOT free: it is a
  representation change (2× the activation neurons), and crucially the OFF channel still reads out through `g·(V−E)`
  (W2). It mitigates W1's *ceiling* but inherits W2's *install divergence*. It is also not a GLA-specific idea — it is
  the standard ANN→SNN two-sided-coding trick, orthogonal to GLA.
- **A different transfer function** (e.g. an unclipped graded read) would require a `sim/` edit to the graded step block
  (remove/replace the `clip`) — at which point you are editing `sim/` anyway, and you still face W2.

**Q2 answer: GLA does not address the MLP clip-compression — it is an attention reformulation, and the wall is MLP.
The bridge *can* widen the linear band (`graded_source_scale_mV`, two-sided ON/OFF coding), but (a) the scale sweep
already showed widening doesn't recover the cumulative, (b) the structural `clip(0,1)` ceiling needs a `sim/` edit to
remove, and (c) all of these still read the target current as `g·(V−E)` (W2). So even the honest MLP mitigations are
NOT cheaper-than-the-edit and do NOT escape W2.**

---

## 4. Q3 — the `g·(V−E)` divergence: does the analog-accumulator reframe AVOID it?

**NO — verified to source, this is the load-bearing finding of this scoping.** The hoped-for escape was "a CURRENT-based
accumulation (`I` directly, not `g·(V−E)`) the bridge may already support." It does not exist for a rate/graded code:

- **The conductance→current law is `I = g_e·(E_e − v) + g_i·(E_i − v)`** — the literal kernel, `kernels.py:214`
  (`fused_conductance_decay_and_current`). Every projection feeds this.
- **The graded path does NOT avoid it.** The graded mask routes the SOURCE's `a_cont` (the `clip`) into
  `cp_conductance_g_e/g_i` (`bridge.py:6169/6175`), which then goes through the SAME `g·(E−v)` law. The `clip` is on the
  source read; the `g·(V−E)` is on the target current. Both walls, in series.
- **The NMDA-recurrent accumulator does NOT avoid it.** Its increment is spike-driven (`_nr_mat.T @
  cp_prev_firing_states`, `bridge.py:6336`) and its readout is `g_nmda·(E_NMDA − v)` with the Mg2+ block
  (`fused_nmda_update_and_current`, `:6343-6347`). So even the bridge's one persistent slow-state buffer reads out
  through the driving force AND is gated by binary spikes (re-inheriting the 0.5 ceiling on its drive).
- **No matvec writes directly to current.** The only `I=`/`I+=` assignments in the step
  (`bridge.py:6177, 6197, 6217, 6244, 6272-6277, 6288, 6312, 6349, 6422, 6479, 6527, 6553, 6570`) are: the
  conductance-derived `synaptic_current_I_syn_pA`, the static `cp_external_input_current`, the divisive/mean-adapt
  *transforms* of those, neuromodulator scalar drives, experiment stimulus, the OU current, and the various
  `g·(E−v)`-derived conductance currents (NMDA, coincidence plateau, GABA_B, TD). **None is a `Wᵀ@activity` matvec
  deposited as current.** There is no current-based linear-accumulation path.
- **The ONE genuine exception is the RF complex-state path** (`_rf_advance_one`, `bridge.py:5710-5746`): the complex
  matvec `(W_re@re − W_im@im)` is added DIRECTLY to the complex state `Z = re + i·im`, with NO clip, NO `g·(V−E)`, NO
  refractory ceiling, and the readout is the Im zero-crossing PHASE. This is a true linear/multiplicative accumulator —
  but it is the **phasor** substrate (the FHRR composer), and Gen-F is rate-coded. Using it for the generator means
  re-encoding the generator's activations as phasors — the past-saturation scoping §2(d) explicitly parked this as "a
  much larger change … re-architects the generator's representation."

**Q3 answer: the analog-accumulator reframe does NOT avoid `g·(V−E)` — on a rate/graded code there is no current-based
accumulation path on the bridge; the graded and NMDA-recurrent accumulators both read out through `g·(V−E)` (and the
NMDA one is spike-gated too). So GLA-as-MLP-fix faces the SAME W2 divergence that killed the distillation. The only path
that avoids `g·(V−E)` is the RF complex accumulator (a phasor re-architecture, not cheap, not GLA).**

---

## 5. Q4 — cost vs the differentiable-bridge `sim/` edit: is GLA a SMALLER change?

**For the MEASURED (MLP) wall: GLA is NOT a smaller change — it is a NO-OP reframe (= graded transmission, already
measured NEGATIVE), so the "change" is zero and it does not help.** There is no cheaper MLP mechanism hiding in GLA; the
reframe's MLP realization is the graded path that W1+W2 already characterize.

**For ATTENTION (Q1 of the consolidation scoping): a GLA running-state step-block is a real additive `sim/` edit — and
it is the SAME cost class as, and DOWNSTREAM of, the differentiable-bridge edit, not cheaper.** Concretely:
- The differentiable-bridge edit (option (a), surrogate-grad scoping §2) is: a guarded, default-off **backward** through
  the consolidated slice's graded op chain (`a_cont` → masked graded matvec → `g·(V−E)` → AdEx), reusing
  `bptt_snn.backward_unroll`'s validated math (`bptt_snn.py:180-256`). It is the FORWARD-already-exists, ADD-A-BACKWARD
  edit. Multi-week, owner-byte-reviewed. **It fixes W2 by training the *actual* `g·(V−E)` dynamics** (the thing the
  idealized trainer couldn't see).
- A GLA attention step-block is: a NEW forward op (the gated outer-product accumulator + read) PLUS — because GLA
  re-architects attention away from Gen-F's softmax — a **finetune** to recover the re-architected attention's weights
  (Gen-F's Q/K/V are not verbatim-reusable for GLA; SpikingBrain reused Qwen Q/K/V because it KEPT them in a GLA it
  *continued-trained*, arXiv 2509.05276). That finetune is exactly the on-bridge gradient capability the differentiable
  edit provides. **So GLA-attention needs the differentiable edit (or an equivalent trainer) underneath it — it is not a
  way around the edit; it sits on top of it.**
- The consolidation scoping ranked GLA as attention-option #1 precisely as "a small additive guarded default-off
  step-block" — true, but it flagged the cost: "Gen-F's weights are NOT verbatim-reusable; a conversion/finetune step is
  needed" (`-consolidation-scoping.md` §2.3, option #1 "Cost / risk" cell). That finetune IS the differentiable-bridge
  work.

**Q4 answer: GLA is NOT a smaller change for the wall in front of us. For the MLP it is a no-op (graded = the measured
NEGATIVE). For attention it is a real additive step-block but it is downstream of (needs) the same on-bridge gradient
capability the differentiable-bridge edit delivers, and it does not verbatim-reuse Gen-F. The differentiable-bridge edit
is the more fundamental, more reusable artifact (it also IS the C2 growth primitive, surrogate-grad scoping §4); GLA is
the eventual *attention operator*, to be built AFTER the gradient path exists.**

---

## 6. The cheapest DE-RISK (if the owner still wants to probe GLA before the edit) + GO/NO-GO

The honest recommendation is to **proceed to the differentiable-bridge edit** (the distill-NEGATIVE's routed option (a)),
because the analysis above shows GLA does not sidestep the measured wall cheaply. BUT if a cheap GLA-direction probe is
wanted FIRST (cheapest-first discipline), there are exactly two NO-`sim/`-edit probes, both decisive in hours, and one is
genuinely worth running because it tests the deeper reframe rather than re-confirming a known NEGATIVE:

**Probe P1 (the worthwhile one) — RF complex-accumulator faithfulness for ONE linear layer.** Does a single dense linear
layer, installed as RF **complex synapses** (`rf_set_complex_weights`, `bridge.py:5707`) and read by **phase**
(`rf_read_phases`, `:5684`), reproduce the off-bridge linear layer's rank WITHOUT the `clip`/`g·(V−E)` compression?
- Cost: hours, CuPy, reuse-by-import (the RF op API exists; NO `sim/` edit). Drive a calibration activation through one
  `W` as a phasor, read phase, compare rank to the float layer.
- **GO** if per-layer phase-rank ≥ ~0.85 (matching the graded L0 0.865 but WITHOUT the cumulative compression when
  stacked 3 deep, since the complex accumulator has no clip/ceiling). ⇒ the RF accumulator is the substrate-native
  escape from BOTH walls, and the generator-as-phasor re-architecture becomes the (larger but principled) plan.
- **NO-GO** if phase coding cannot carry a dense signed linear layer's magnitude (RF info is in PHASE; a linear layer's
  output magnitude may not map cleanly to unit-magnitude phasors) → the RF path does not rescue a magnitude-coded MLP,
  and the differentiable-bridge edit (training the real `g·(V−E)`) is the remaining path. **This is the genuinely new
  information P1 buys** — it tests the ONLY bridge mechanism that avoids `g·(V−E)`, which none of the four NEGATIVEs did.

**Probe P2 (skip — it re-confirms a NEGATIVE) — GLA-as-MLP via the NMDA-recurrent accumulator.** Tagging the dense MLP
pathways `exc_receptor="nmda_slow"` to route them through the slow accumulator. This is NOT worth running: the increment
is spike-gated (re-inherits the 0.5 ceiling) AND the readout is `g·(E−v)` (W2) — it cannot beat what graded already did.
Predicted NEGATIVE by inspection; do not spend GPU.

**GO/NO-GO for the strategic call:** the scoping's recommendation is **NO-GO on GLA as a cheap sidestep** for the MLP
wall (it is a no-op reframe = the measured NEGATIVE), and **GLA-attention is deferred to AFTER the differentiable-bridge
edit** (it needs the gradient capability and re-architects Gen-F). **Run P1 (RF accumulator, hours, no edit) as the one
cheap probe that tests something new** — if P1 GOes, the phasor re-architecture is the principled both-walls escape; if
P1 NO-GOes, the differentiable-bridge edit (option (a)) is correctly the next build, exactly where the distill-NEGATIVE
routed.

---

## 7. Honest verdict — does GLA sidestep the wall cheaply?

**NO. GLA does not cheaply sidestep the per-layer clip-compression wall, and it faces the same `g·(V−E)` divergence at
the point that matters.** Specifically:

1. **For the MEASURED (MLP) wall, GLA is a NO-OP reframe.** On the bridge, "keep linear ops as analog accumulators" =
   graded transmission = the exact mechanism that compressed `0.846→0.620→0.288` (W1) and failed the install at 0.444
   (W2). The four NEGATIVEs already measured this; GLA re-labels it, adds no new MLP mechanism. (`-past-saturation-
   scoping.md` §2(e) itself said "for the MLP, (a) graded already realizes the reframe.")
2. **The `g·(V−E)` divergence is NOT avoided.** Verified to source: every rate/graded projection reads out as `g·(E−v)`
   (`kernels.py:214`); the graded `clip` is on the source, the `g·(V−E)` on the target; the NMDA-recurrent accumulator
   is spike-gated AND `g·(E−v)`-read; **no current-based matvec path exists**. The only escape is the RF complex
   accumulator (`bridge.py:5710-5746`) — a phasor re-architecture, not cheap, not GLA.
3. **For attention, GLA is the right operator but NOT cheaper than the edit and DOWNSTREAM of it.** A GLA step-block is a
   real additive `sim/` edit that re-architects Gen-F's softmax (weights not verbatim) → needs the on-bridge finetune
   the differentiable-bridge edit provides. It sits on top of the gradient capability, it doesn't replace it.
4. **⇒ Back to the differentiable-bridge edit** (the distill-NEGATIVE's routed option (a)) as the principled fix for W2
   — it trains the *actual* `g·(V−E)`+AdEx dynamics (the residual the idealized trainer couldn't see), reuses
   `bptt_snn`'s validated backward, is bounded/additive/guarded/default-off, AND doubles as the C2 growth primitive
   (surrogate-grad scoping §4). The ONE cheap probe worth running first is **P1 (RF complex accumulator, hours, no
   edit)** — the only bridge mechanism that genuinely avoids both walls, untested by the four NEGATIVEs; GO → phasor
   re-architecture is the both-walls escape, NO-GO → the differentiable-bridge edit is correctly next.

**Bottom line for the controller:** GLA/analog-accumulator is NOT the cheap sidestep it looked like — for the MLP it is
the already-NEGATIVE graded path; for `g·(V−E)` it offers no escape on a rate/graded code; for attention it is the right
operator but needs (is downstream of) the same edit. The differentiable-bridge `sim/` edit remains the justified next
build. Run the RF-accumulator probe P1 first (cheap, no edit, tests the only un-tested escape) before committing the
multi-week edit; everything else GLA promised for THIS wall is already measured NEGATIVE.

---

## 8. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / findings read):**
- **The conductance→current law is `g·(E−v)`** — `kernels.py:208-215` (`fused_conductance_decay_and_current`,
  `I_syn = g_e_new*(E_e − v) + g_i_new*(E_i − v)`), read in full.
- **The graded path's `clip` is on the SOURCE, the target current is still `g·(V−E)`** — `bridge.py:6144-6147`
  (`a_cont = clip((v−rest)/scale, 0, 1)` of `cp_membrane_potential_v`), `:6169/6175` (the graded matvec deposits into
  `cp_conductance_g_e/g_i`, which feed the `g·(E−v)` law). Read in full.
- **NO current-based matvec path** — every `I=`/`I+=` in the step (`bridge.py:6177, 6197, 6217, 6244, 6272-6277, 6288,
  6312, 6349, 6422, 6479, 6527, 6553, 6570`) is conductance-derived `g·(E−v)`, the static `cp_external_input_current`,
  a transform of those, or the OU/experiment/neuromod drive — none a `Wᵀ@activity` deposited as current. Grepped +
  read the span.
- **The NMDA-recurrent accumulator is spike-gated AND `g·(E−v)`-read** — `bridge.py:6326-6349`: increment
  `_nr_mat.T @ cp_prev_firing_states` (`:6336`, binary spikes), readout `fused_nmda_update_and_current(... v, E_e ...)`
  (`:6343-6347`, `g·(E−v)` with Mg2+ block); slow buffer tau~100ms (`:1820-1823`). So the bridge's one persistent
  slow-state buffer does NOT avoid W2 and re-inherits the spike ceiling. Read in full.
- **The RF complex-state path is the one true non-`g·(V−E)`, non-clip accumulator** — `bridge.py:5710-5746`
  (`_rf_advance_one`: `re_new = decay*rot(re,im) + (W_re@re − W_im@im)`, complex matvec added DIRECTLY to `Z`, Im
  zero-crossing readout, no clip, no `g·(V−E)`); `rf_set_complex_weights` (`:5707`), `rf_read_phases` (`:5684`). Gen-F
  is rate-coded → using it is a phasor re-architecture. Read in full.
- **The four walls/NEGATIVEs** — `multilayer-NEGATIVE` (MLP slice, rate-saturation pinned 0.5, lines 8-28);
  `graded-PARTIAL` (graded un-saturates L0 0.865, cumulative 0.327, scale sweep 20-best, lines 9-17);
  `popcode-NEGATIVE` (deterministic `clip` compression `[0.846,0.620,0.288]`, std `0.00e+00`, pop-coding no-op, lines
  3, 9-22); `distill-NEGATIVE` (offline 0.815 → installed 0.444, df-aware ARM2 confirms `g·(V−E)` is the residual,
  shuffled-control 0.542 metric confound, lines 14-35). All read in full.
- **The differentiable-bridge edit (option a) scope** — `2026-06-22-surrogate-grad-on-bridge-scoping.md` §2 (a guarded
  default-off backward through `a_cont`→graded matvec→`g·(V−E)`→AdEx, reusing `bptt_snn.backward_unroll`
  `:180-256`); §4 (it IS the C2 growth primitive); §0/§6 (the distill NO-GO routes here). Read in full.
- **GLA/SDSA/SpikingBrain SOTA** — `2026-06-22-genseq-loopstep3-consolidation-scoping.md` §2.2-2.3 (GLA running-state
  `S_t = diag(g_t)⊙S_{t-1}+k_tᵀv_t` = SpikingBrain arXiv 2509.05276, reuses Qwen Q/K/V verbatim in a CONTINUED-TRAINED
  GLA; option #1 "Gen-F weights NOT verbatim-reusable, finetune needed"); `-past-saturation-scoping.md` §2(e) ("for the
  MLP, (a) graded already realizes the reframe") + §2(d) (RF phasor = "a much larger change, re-architects the
  representation"). Re-read this pass.
- **The convert-GO ran in PyTorch, off-bridge** — `2026-06-22-genseq-convert-GO-spiking-generates.md` §Scope
  ("PyTorch/CUDA"); attention reimplemented in PyTorch SDPA, T = rate-quantization over float ranges → the float→spiking
  math is fine, the wall is the ON-BRIDGE `g·(V−E)`/graded representation. Read in full.

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — the one worthwhile probe]** That the RF complex-accumulator (P1) carries a dense SIGNED linear layer's
   *magnitude*-coded output as phase rank ≥ ~0.85 across 3 stacked layers WITHOUT cumulative compression. RF info is in
   PHASE (unit-magnitude); a transformer linear layer's output is magnitude-coded. Whether magnitude→phase mapping
   preserves rank for a dense layer is untested (the FHRR composer used phasors for *symbolic* codes, not dense linear
   activations). P1 measures it directly; this is the only bridge mechanism that escapes `g·(V−E)`, so it is the genuine
   open question.
2. **[VERIFY]** That a two-sided ON/OFF activation coding (to widen the W1 dynamic range) does not itself re-saturate or
   blow up the synapse count unacceptably — and, even if it widens W1, it still reads through `g·(V−E)` (W2), so it is
   not a complete fix. Not recommended as a primary probe (it is orthogonal to GLA and doesn't escape W2).
3. **[VERIFY]** That a GLA-attention step-block, if ever built, genuinely needs the differentiable-bridge gradient for
   its finetune (vs the HYBRID train-then-install) — the HYBRID already failed for the MLP on the `g·(V−E)` install gap,
   so a GLA attention block trained in an idealized forward would likely hit the same install divergence, implying it too
   needs the real-dynamics backward. Consistent with the verdict (GLA-attention is downstream of the edit), but not
   independently measured.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/findings/2026-06-22-genseq-loopstep3-multilayer-NEGATIVE-rate-saturation.md` (MLP-only slice; the rate
  ceiling; read in full).
- `research/findings/2026-06-22-genseq-loopstep3-graded-PARTIAL-diagnosis-confirmed.md` (graded un-saturates, L0 0.865,
  cumulative 0.327, scale sweep; read in full).
- `research/findings/2026-06-22-genseq-loopstep3-popcode-NEGATIVE-deterministic-readout.md` (W1: deterministic `clip`
  compression `[0.846,0.620,0.288]`, std `0.00e+00`, pop-coding no-op; read in full).
- `research/findings/2026-06-22-genseq-loopstep3-distill-NEGATIVE-live-bridge-gap.md` (W2: offline 0.815 → installed
  0.444, `g·(V−E)` confirmed the residual; read in full).
- `research/findings/2026-06-22-surrogate-grad-on-bridge-scoping.md` (the differentiable-bridge edit = option (a), its
  scope §2, the C2 overlap §4; read in full).
- `research/findings/2026-06-22-genseq-loopstep3-consolidation-scoping.md` (the GLA/SDSA attention options §2.2-2.3, the
  "Gen-F weights NOT verbatim-reusable / finetune needed" cost; read in full).
- `research/findings/2026-06-22-genseq-consolidation-past-saturation-scoping.md` (the §2(e) GLA-reframe flag = "for the
  MLP, graded already realizes the reframe"; §2(d) RF-phasor = larger re-architecture; read in full).
- `research/findings/2026-06-22-genseq-convert-GO-spiking-generates.md` (convert ran PyTorch/CUDA off-bridge; the
  float→spiking math is fine; read in full).
- `research/findings/2026-06-22-genseq-step0-C1-consolidation-GO.md` (single-layer 0.918; positive-only + one-hot
  residuals; read in full).
- `sim/kernels.py`: `fused_conductance_decay_and_current` (`:208-215`, `I = g·(E−v)`); `fused_nmda_update_and_current`
  (`:228+`, `g·(E_NMDA−v)` + Mg2+ block).
- `sim/bridge.py`: graded analog step block (`:6128-6175`, `a_cont` `:6144-6147`, deposits to `g_e/g_i` `:6169/6175`);
  conductance→current call (`:6059-6062`); NMDA-recurrent slow accumulator (`:6326-6349`, increment `:6336`, readout
  `:6343-6347`, alloc `:1406-1408`, decay `:1820-1823`); the full `I=`/`I+=` span (`:6177-6570`); RF complex-state
  accumulator (`_rf_advance_one` `:5710-5746`, `rf_set_complex_weights` `:5707`, `rf_read_phases` `:5684`,
  `rf_kick` `:5646`); `cp_external_input_current` (static drive, `:1115-1128, :2942, :3490-3493`); E/I split
  (`:6084-6126`).
- `sim/regions.py`: `RegionPathway.graded` (`:355-372`, horizontal-cell graded release); `exc_receptor` /
  `coincidence_detector` (`:341-353`).

### Current literature / SOTA (verified via the prior scopings' primary-source pass)
- **GLA running-state accumulator / linear attention** — SpikingBrain-7B, CAS/BICLab 2025, arXiv 2509.05276 (GLA + SWA;
  reuses Qwen Q/K/V verbatim in a CONTINUED-TRAINED GLA, not a verbatim install; the only spiking LM at scale).
  Katharopoulos et al. 2020, arXiv 2006.16236 (linear attention = recurrent accumulator).
- **Spike-driven SDSA (binary-AND + column-sum, no rate readout)** — Yao et al., NeurIPS 2023, arXiv 2307.01694 (vision
  only). Spikformer SSA — Zhou et al., ICLR 2023, arXiv 2209.15425.
- **Point-neuron analog/pre-spike limit** — Mikulasch-Priesemann (project standing-practice: dense/whitening is
  analog/pre-spike; the bridge's graded path is the on-substrate analog stage — but it reads out through `g·(V−E)`,
  which is the residual here). Kandel 6e Ch 22 (retinal graded potentials).
- **Surrogate-gradient BPTT (the differentiable-bridge edit's reused math)** — Neftci-Mostafa-Zenke 2019, arXiv
  1901.09948; the `sim/bptt_snn.py` / `bptt_snn_gpu.py` validated backward.
