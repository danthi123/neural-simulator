# Full close-out backlog — work through EVERYTHING to zero (owner directive 2026-06-24)

Owner: "work through issues, not defer them. We want to fully close out everything." This merges the
close-out audit (deferred *builds*) + the biology-fidelity audit (class-b host-cognition *residuals*) +
Option A into ONE ranked backlog, worked to zero — one at a time, respecting file conflicts + the research
gate + the no-confab moat.

## IN FLIGHT
- **Option A — onebrain merged-default flip (Closures 1+3).** Bug-fix build `afda4ecb`: fix the
  `CoResidentOneBrainComposer` construction (its idle parser re-injects from an empty wiring plan + wipes the
  command_route gate) → re-run gate 3 (onebrain vs rf nav Δ=0) → flip if Δ=0 + Probe-1 byte-identity holds.
- **Communicable-brain values arc — learned talkativeness.** De-risk `a341a0ba` (final seed).

## BOUNDED BUILDS — work through IN ORDER (edits that touch the same file are serialized)
1. **Route-B-on-onebrain seam** (`nav_conv_merged_bridge.py:1804` guard): the perception→compose host-`M`
   →synaptic-store seam on the onebrain path is unbuilt (the genuine cross-region residual). [edits
   nav_conv_merged_bridge.py → AFTER Option A frees that file]
2. **Communicable-brain turn → console**: wire the gate→VERIFY→emit "what do you think" turn into the
   agent/console behind a flag (after learned-talkativeness lands + the GPU-Qwen fluency drop-in). [owner's
   top frontier]
3. **b2 generative sampler → spiking SWR-replay** (biology-audit class-b #2 + the generative-act fix): make
   the generative ACT itself the brain's (a spiking SWR-gated replay sampler), not a host numpy `rng.choice`.
   [research-gate: a new mechanism class]
4. **cue-match scan + abstention → spiking sequencer DEFAULT** (class-b #1, the largest live conversational
   host residual): the spiking sequencer GO'd at 320 but reverted at small vocab — close the small-vocab gap
   + make it the default. [research-gate or build]
5. **between-op hand-offs → one persistent spiking loop** (class-b #4, the #1 one-brain gap): host `to_host`
   round-trips persist even on the onebrain path. Closure 2 made the FLAT path persistent; extend to ALL ops.
6. **rich-answer assembly → neural dlPFC planner on the `--rich` runtime** (class-b #5): the GO spiking 3G
   planner exists but isn't wired onto the `--rich` path (host discourse heuristics today).
7. **read-out normalization on-bridge circuit** (class-b #6): the host log-domain double-centring → the
   per-concept feedforward-inhibition + per-hub adaptation circuit (CYCLE 93b scoped).
8. **generator host weights → on-substrate** (class-b #7 / the "spiking generator" overclaim): the generator's
   host-weight matvecs on a conductance-free accumulator → genuine dynamical spiking. [overlaps the deep
   host-structure frontier]
9. **develop-loop A2 (6-seed) + A3 (persist `cp_connections`)**: scale + persistence follow-ons.

## DELIBERATE KEEP — NOT a deferral (owner's own directive feedback_close_arcs_to_full_capacity)
- The rf/numpy composer path as the TEST-ORACLE + the CPU-portable path. Kept on purpose. (If the owner wants
  it retired too, that's a one-line change — but the standing directive is "keep the legacy as oracle.")

## ACTIVE RESEARCH — not a bounded close-out (multiply-NEGATIVE; research-gate, honest it's open)
- The FHRR exact-inverse VSA algebra → a LEARNED spiking cortical binder. The deepest frontier; NEGATIVE
  across multiple attacks (Storkey/DG/fixed-expansion decorrelation; learned-bind bundling). Keep attacking
  via deep-research + cheap-first de-risk — NOT shelved — but it is an open research problem, not a flip I can
  guarantee to close.
- The off-bridge spiking Qwen fluency faculty: owner-sanctioned (fluency-only, lesion-confirmed). Bridge
  co-residence DEMONSTRATED (~14 GB, local); wiring it into the live console is build #2's drop-in.

---

## STATUS (2026-06-25 overnight close-out — purity backlog nearly to ZERO)
- **#1 Route-B-on-onebrain — CLOSED on the agent** (6-seed GO + Option-1 agent wire-in; deployed MergedNavConvAgent perceives+composes via spikes-only grounding on the onebrain composer; last cross-region host-M closed; CYCLE 565/568).
- **#3 b2 generative sampler → spiking — CLOSED** (spiking soft-WTA sampler default-on).
- **#5 between-op hand-offs → one persistent spiking loop — CLOSED** (Closure 2: persistent_loop default-on; FLAT+clause register->register, no host round-trip).
- **#6 rich-answer → neural dlPFC planner on --rich — CLOSED (ALREADY-WIRED)**: the GO spiking 3G planner is on --rich, default-on-GPU, committed 15fdc85d+d9287763; the line below was STALE. Re-validated CYCLE 569 (neural-path GO, lesion 2->0, moat 0-FA, 19/19 content-selection tests).
- **Option A (onebrain merged-default) — CLOSED** (gate-3A functional-neutrality; onebrain default, rf=oracle).
- **#7 read-out normalization on-bridge — CHARACTERIZED READ-BOUNDARY** (subtractive arithmetic EXACT, the point-neuron f-I READ loses ~half; same wall divisive hit; numpy scaffold stays; Option C = future; sim/ reverted; CYCLE 566).
- **#8 generator → on-substrate — SPLIT**: FORWARD on-substrate (matvec on bridge + spiking nonlinearities; B-relabel done; #8-A compose-polish RUNNING); WEIGHTS = a named H-2-class DEEP boundary [D] (backprop-distilled, owner-deferred); CYCLE 567.
- **#4 cue-match scan → spiking sequencer DEFAULT — OPEN (the LAST closure)**: largest live conversational host residual; GO@320 reverted@small-vocab; research-gate scoping dispatched.
- **#2 communicable-turn → console — the FRONTIER (priority 1)**: Stage A+B done; the discursive turn + console = the foreground after purity.
- **#9 develop-loop A2/A3 — curriculum-adjacent** (folds into the curriculum-training work, priority 1).

⇒ Purity (cognitive-shortcut closures) nearly ZERO: only #4 (spiking sequencer) the last open closure + #8-A polish running. Then (1) curriculum + the communicable-turn richness.
