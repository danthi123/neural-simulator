# gap#5 readout — RESEARCH GATE: the missing biology is the SHARP-WAVE-RIPPLE (SWR) state, and it is point-neuron-realizable (2026-07-24)

**The 5-method boundary was a brain-STATE error.** Every readout this arc drove a THETA-paced basket-disinhibition sweep,
but offline sequence replay is a SHARP-WAVE-RIPPLE (SWR) phenomenon — a DIFFERENT brain state with a different
interneuron regime. This is the "missing biology" behind the wall (owner's framing: a wall = biology we haven't
supplied). The mechanism is cited, and — unlike the dendritic Kamondi dead-end — POINT-NEURON realizable.

## A. The SWR ignition mechanism (cited, load-bearing, point-neuron-realizable)
Buzsáki *Rhythms of the Brain* (2006), SWR section (`~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/…txt`):
- **State distinction** (L14400-14411): SWRs "occur when the animal has no or minimal interaction with the environment
  … emerges in the excitatory recurrent circuits of the CA3 region." Theta = encoding/exploration; SWR = quiet-rest/sleep
  replay. Fig 11.13 (L13057-13061): "Different constellations of interneuron classes contribute specifically and
  differentially to theta and ripple oscillations." ⇒ the arc applied a theta-state readout to an SWR-state phenomenon.
- **Ignition = a transient E>I imbalance in the CA3 recurrent circuit** (L14452-14454): "inhibition cannot keep up with
  the increased excitation, resulting in a three- to fivefold gain in network excitability"; (L7076-7078): "excitation
  transiently exceeds inhibition by as much as three-fold, providing short temporal windows." A self-organized recurrent
  buildup (nonspecific drive + the learned CA3 attractor basin + feedback-I failing to track the E surge) in a ~100 ms
  envelope (L14428), self-terminating when I catches up. Sequences replay at ripple troughs, ~2× compressed, riding
  SPIKE TIMING not rate (L14447-14448).
- **Point-neuron realizable:** all three ingredients are network/somatic, not dendritic — (1) ignition = recurrent-E
  transiently outrunning feedback-I; (2) handoff = the learned forward-asymmetric links, GAIN-AMPLIFIED within the
  envelope; (3) self-termination = feedback-I re-catching + SFA. Ecker et al. 2022 eLife e71850 (the project's prior
  gate already named this "our exact substrate class"): a spiking CA3 of point-ish (AdExpIF) neurons with structured
  recurrent weights AUTONOMOUSLY generates SWRs + replays forward/reverse from a nonspecific drive. NOT the
  point-neuron-limit family (contrast Kamondi, which failed *because* it is intrinsically dendritic).

## B. Why the readouts failed (op-point diagnostic, real but not a closure)
The completion cue IGNITES (6/6): sustained `recall_drive=700` for `recall_steps=150` + `self_regen=0.15` (bistable LATCH).
The theta-sweep readout does NOT ([0,0,0]): brief `det_dur=12`/100-step pulses + `self_regen_read=0.0` (DE-latched, on
purpose so the bump can move) + `_configure_ou(None)` (no noise). The one readout that DID ignite a discrete assembly —
RANK-1 spontaneous reactivation — used `self_regen=0.15` + weak NON-SPECIFIC Poisson noise + no cue (the self-organized
SWR recipe). ⇒ the boundary methods dropped exactly the SWR recipe. BUT "just match the completion op-point" re-latches
the bistable hold → a stationary bump that won't hand off ([1,0,0]/[3,3,3]) — B trades no-ignition for no-handoff (the
arc's exact dichotomy). The SWR TRANSIENT gain reconciles ignite+move+self-terminate: a time-bounded network "hold"
that ignites, and being transient, permits the moving bump + self-terminates. It also lifts weak `adj_fwd≈38` into the
handoff band (~114-190; the gamma-WTA ordering needs strong adjacent links) — doubly motivated.

## C. Ranked next de-risks (queued; does NOT block the pivot faculties)
**Option 1 — CHEAPEST ~20-min config diagnostic (run FIRST):** set the theta-sweep readout cue to the completion's
igniting op-point (sustained `recall_drive≈700`/`recall_steps≈150` onto assembly-0 + `self_regen_read=0.15` +
`recall_k_thresh=110`) on the decoupled store; read `per_asm_active`. Predicted: IGNITES ([0]≥1) ⇒ confirms ignition is
achievable (boundary was op-point/state, not a substrate wall) but likely [1,0,0]/[3,3,3] ⇒ residual =
ignition-compatible-with-handoff-and-self-termination ⇒ build Option 2. If it does NOT ignite even then → deeper finding.
**Option 2 — TOP PICK, the real build: SWR-state E/I-transient envelope readout** (`_gap5_swr_envelope_replay_derisk.py`,
reuse-by-import, NO `sim/` edit): rest in the SWR state (bistable silent down-state + weak NON-SPECIFIC noise as the
self-organized ignition source, RANK-1 recipe — not a targeted detonator, not theta) → impose a TRANSIENT E>I ENVELOPE
(~100-200 ms: transiently drop `ca3_pv_basket` feedback-I and/or add broad weak CA3-exc, reusing `run_swr_replay_phase`'s
~100/50 ms envelope timing + `ca3_swr_burst` gate) so the most-excitable assembly ignites and its gain-amplified forward
links carry A→B→C → self-terminate (re-raise basket + SFA `d_abs`/`a_abs`) → order within the envelope by TIMING
(gamma-WTA + post-fire silence; NEVER STD/fatigue the chain — banked NEGATIVE).
**GO-gate (6-seed):** `per_asm_active~[1,1,1]` (not [3,3,3]/[0,0,0]) AND `forward_frac ≥1.5×chance` & forward>reverse AND
the net RESTS silent between discrete events, ≥5/6 seeds.
**Anti-cheats (each retires a failure mode):** NO-SWR (constant E/I → collapse; the transient is load-bearing);
SHUFFLED-STORE; REVERSE-CUE; PERMUTED-assembly (→chance); NO-NOISE acid (retires the self-sustaining-attractor confound);
NO-ENCODE; ADAPT-LESION (`d_abs`→0 → [3,3,3] co-fire, Ecker control); FROZEN-plasticity byte-hash; NUMPY-REFERENCE GUARD
(no host per-step silence/argmax in the loop). GPU-preferred (n_ca3=2000); numpy CPU smoke at n_ca3=1000 valid pre-check.

**Residual risk (honest):** reconciling ignite+move+self-terminate on the decoupled store is a TUNING band within a
VALIDATED mechanism (Ecker 2022 proves the regime on point neurons), not a substrate wall: envelope over-drives→[3,3,3],
under-drives→[0,0,0]; the envelope depth×duration×noise-σ sweep is the genuine de-risk. Reusable machinery: RANK-1
`_gap5_spontaneous_reactivation_derisk` (noise+bistable ignition); `consolidation_trainer.run_swr_replay_phase` (envelope
timing); `_gap5_gamma_wta_replay_derisk` (ordering); the decoupled store builder. Every piece exists except the
self-organized E/I-transient generator, buildable with no `sim/` edit.

## 2026-08-20 UPDATE — discrete ignition is SOLVED; the residual is a weak-forward-link MECHANISM gap, not a tuning band
<!--derived-->
A 32-config seed-42 sweep of `_gap5_swr_envelope_replay_derisk.py` (on disk, `research/findings/raw/gap5_r4/swr_*`)
REFINES the "Residual risk" above, and corrects it. The `per_asm_active~[1,1,1]` DISCRETE-ignition state — the
sub-condition the theta-era readouts never reached — IS reachable, but NOT via the depth×duration×noise sweep this
finding predicted: the lever is **`--sel-inhib-spare`** (spare the assembly members from the basket suppression), which
turns the over-driven `[3,3,3]` co-fire into clean `[1,1,1]`. So "ignite + discrete" is no longer the wall.
The FORWARD-SEQUENCING sub-condition (`forward_frac>0`, order A→B→C) is the true residual, and it is a MECHANISM gap,
not an envelope tuning band: (a) the runner's `--env-exc-ramp` ramps env-exc from ~0, so it is subthreshold for most of
the window and KILLS ignition (every ramp config → [0,0,0]); (b) more fundamentally, the store's forward-asymmetric
links (`adj_fwd≈38`) are too weak to bias the order — the handoff band is ~114–190 (Section B above) and NO runner knob
amplifies them. So the next step is a BUILD: gain-amplify the forward links into the handoff band DURING the SWR
envelope (or re-encode the store with stronger forward links), NOT another envelope sweep. This is the convergent
unblock for the D5 learn-through-use transfer + the brain-pure sleep-replay store (both 2026-08-20). Blind envelope
tuning is retired here by the research gate (4 rounds, 32 configs).

## 2026-08-20 UPDATE-2 — the forward-link BUILD was TRIED (gain + directional cue): validated ingredients, but the wall is the STORE ARCHITECTURE, not link strength
<!--derived-->
The "next step: BUILD — gain-amplify the forward links" named just above was built + tested (two agents): a TRANSIENT
`--fwd-gain` (amplify `adj_fwd` into the ~114–190 handoff band only within the envelope, byte-hash-verified restored)
+ a `--onset-cue-asm` directional PREFIX cue (depolarize chain-start assembly A at envelope onset). Both are correct,
transient, reusable INGREDIENTS: the gain lifts `adj_fwd` 38→154 and SUPPRESSES reverse; the onset cue makes A ignite
first 9–10/10 cycles. **But genuine directional replay is a NEGATIVE** — the decisive store-lesion anti-cheats fail:
forward SURVIVES the REVERSE-ASYM-LESION (symmetrized store → forward 1.00), SHUFFLED-STORE (1.00), and
PERMUTED-ASSEMBLY (1.00). So the order rides the cue + an intrinsic positional readout bias, NOT the encoded
forward-asymmetric chain. Corrects UPDATE-1's framing: the residual is NOT weak forward-link strength — it is
ARCHITECTURAL. Two localized substrate facts: (1) the decoupled store's strong within-attractors (~200) reverberate
semi-continuously (pop never fully silences) so the net never SEGMENTS into discrete SWR events — bistable *completion*
is antithetical to a moving-bump *hand-off*; (2) the forward weight asymmetry (38 vs 5) is not load-bearing on order.
**NEXT MECHANISM (no-defer, named — this is the boundary's own long-standing pointer, now confirmed load-bearing):**
the **Ecker-2022 AdEx CA3 model build** — weaker/self-terminating within-attractors + genuine synaptic hand-off + a
ripple-generating interneuron circuit for discrete SWR events; the `--fwd-gain` + `--onset-cue-asm` machinery are
validated ingredients to reuse in it. (Also corrected: the `--env-basket-drop 0 --sel-inhib-spare 20` op-point earlier
called `[1,1,1]` is actually `[0,0,0]`; discreteness needs `env-basket-drop>0`. Runner edits compile, default
byte-identical, uncommitted-ingredients.)
