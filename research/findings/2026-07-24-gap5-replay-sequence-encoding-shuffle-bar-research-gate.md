# gap#5 replay — RESEARCH GATE: what carries sequence order in real replay, and is the "shuffle-collapse" bar right? (2026-07-24)

**READ-ONLY deep-research gate** (LOCAL-FIRST: Buzsáki *Rhythms of the Brain* + Kandel 6e + feature-catalog cluster-D
read in depth; then the directly-relevant point-neuron paper Ecker et al. 2022 eLife 71850 confirmed externally). Triggered
by the parallel store-side result: STDP writes no forward asymmetry (theta-reset empties its Δt window) and BTSP writes
strong-but-**FAN-OUT** forward links (`skip_fwd` dominates `adj_fwd`, `adj_dominance≈0.4`) because BTSP's ~1 s eligibility
is inherently broad. That surfaced the crux: **is the shuffle-collapse GO bar testing the right thing, and does real replay
order ride clean adjacent links or a broad overlapping structure?**

## Bottom line up front (the ranked conclusion)
The shuffle-collapse bar is **the RIGHT KIND of test and is literally the control Ecker 2022 uses** — but the current
de-risk applies it in a way that guarantees a false verdict, and the *store* it is applied to is the wrong shape.
Real replay order rides a **LEARNED, GRADED, near-diagonal BANDED recurrent weight profile** (overlapping place fields),
read out as a **moving population bump**, NOT a clean discrete 1-link relay and NOT connectivity geometry. The three
things to change: **(1) STORE = a graded near-diagonal band over ≥8 assemblies / a continuum (not a flat BTSP fan-out over
3 discrete assemblies with hand-wired fan-in); (2) METRIC = population Bayesian-decode trajectory / rank-order (Davidson
2009 / Foster-Wilson), not discrete assembly ignition; (3) SHUFFLE = destroy the BAND structure while preserving the
weight distribution (Ecker's column-identity shuffle) — that is exactly "order rides STRUCTURE not statistics" and it is
the correct, load-bearing bar.** Everything gap#5 needs is a documented, point-neuron-validated recipe (Ecker 2022 on
AdExpIF neurons); the arc has been re-deriving its pieces.

---

## Q1 — What carries sequence order in real forward replay: clean adjacent links (a) or broad overlapping structure (b)?

**Answer: neither extreme — a LEARNED, forward-graded, NEAR-DIAGONAL BANDED recurrent weight matrix over OVERLAPPING
place fields, read as a moving population bump. It is (b)-shaped (population/overlap-level, graded, some fan-out) but the
order is carried by the SPECIFIC learned band STRUCTURE, so it is shuffle-SENSITIVE to a structure-destroying shuffle.**

Evidence, read in depth:

1. **Order = "activity spreads along the strongest synaptic weights" (Buzsáki, *Rhythms of the Brain* 2006, pp. 348–349,
   L14578–14600).** Verbatim mechanism: novel experience "altered the synaptic weights within the CA3 recurrent system …
   the newly created synaptic weights determined the spread of activity … a simple rule of neuronal recruitment is that
   **activity spreads along the path of the strongest synaptic weights** … assemblies activated most during the experience
   are held together by the strongest synaptic connectivity and become the **'burst initiators'** … followed by the
   progressively less activated neurons of the learning episode. **The temporal proximity of neuronal discharges during
   sharp waves can be taken as an indication of the strengths of their synaptic connectedness.**" ⇒ order rides LEARNED
   weight STRENGTHS (shuffle-sensitive in principle), but the structure is a **graded** "progressively less activated"
   profile, not a binary adjacent relay.

2. **The structure is OVERLAPPING-field pairwise correlations, not 1-to-1 links (Wilson & McNaughton 1994; Lee & Wilson
   2002; Buzsáki p. 347–349, L14540–14575).** "Pyramidal cells **with overlapping place fields** preserved their pairwise
   temporal correlations during subsequent sleep, whereas place cells which did not overlap spatially or temporally
   rarely showed correlated firing." Replay recapitulates the **overlapping-field** temporal-correlation structure.

3. **Theta-compression WRITES a graded band including higher-order (non-adjacent) links (catalog D.67, Buzsáki Ch 11
   pp. 315–323; Dragoi-Buzsáki 2006; Skaggs 1996; Mehta 1997).** Within one ~125 ms theta cycle, overlapping-field cells
   fire in spatial order compressed ~20×; **"this compression brings NON-adjacent positions into the STDP window (~10–40
   ms), so STDP stores higher-order sequence relationships, not just immediate-neighbour ones."** Phase precession is
   **monotonic** (Buzsáki p. 314, L13195: "the phase of place-cell spikes shifts monotonically as a function of position")
   → the STDP kernel (fast, decaying with Δt) writes an **adjacent-dominant, distance-decaying forward band**: some fan-out
   is expected and biological, but it must be **monotone-decaying (adjacent strongest)**.

4. **Kandel 6e (pp. 101–102, Fig. 5-2, L7985–8000): replay is a DISCRETE TRAJECTORY read by POPULATION DECODING, forward
   OR reverse from the SAME store.** "Spatial decoding of the activity of place cells active within these short (50–500 ms)
   SWRs reveals that hippocampal neurons recapitulate or replay **discrete trajectories** … replayed either in the same
   direction (forward replay) or the opposite (reverse replay)" (Diba & Buzsáki 2007). ⇒ the SAME weight store supports
   BOTH directions; direction is a dynamical/initiation property, not a rigid one-way wiring.

5. **DECISIVE — Ecker et al. 2022, eLife 71850, on OUR substrate class (AdExpIF POINT neurons, no dendrites):** the learned
   recurrent matrix is "a highly organized structure … **relatively few strong (>1 nS) synapses near the diagonal**
   (pairs of cells with **overlapping place fields**)" — a **graded near-diagonal BAND**, not discrete links. A **symmetric**
   STDP rule (τ±=62.5 ms) → **bidirectional** replay; an **asymmetric** rule (τ±=20 ms, A+=400/A−=−400 pA) → **forward-only**
   chains. **"The STRUCTURE rather than the STATISTICS of recurrent excitatory weights is critical."**

**⇒ Q1 verdict:** order rides the LEARNED graded near-diagonal band (option b, population/overlap-level, monotone-decaying
fan-out) — shuffle-SENSITIVE to a structure-destroying shuffle, but NOT a clean discrete adjacent-link relay (option a is a
modeling idealization). **The parallel run's `skip_fwd > adj_fwd` (adj_dominance 0.4) IS a genuine anti-biological defect:**
biology's band is **adjacent-dominant and distance-decaying**; a flat/inverted fan-out (BTSP's 1 s eligibility) is exactly
what produces the co-fire `[3,3,3]` failure, because a flat lower-triangle drives all downstream assemblies at once.

---

## Q2 — Is "shuffle the between-assembly weights → order must collapse" the biologically-correct test?

**Verdict: YES — it is exactly Ecker 2022's own control and is load-bearing — BUT the current de-risk applies it on a store
+ metric that make it structurally uninformative. Fix the store and metric; keep a corrected shuffle. Add reverse-store.**

- **The bar is correct in principle and matches the literature's decisive control.** Ecker 2022: "**Shuffling column
  identities (destroying diagonal clustering while preserving the weight distribution) → none of the shuffled networks
  exhibited sequence replay … no sharp-wave-like events.**" That is precisely "order rides STRUCTURE, not statistics" —
  the shuffle-collapse bar, done right. So the instinct to require shuffle-collapse is CORRECT.

- **Why the current de-risk's shuffle looked insensitive (the two real defects, both fixable):**
  1. **3 discrete assemblies + hand-wired fan-in = a geometry shortcut.** With only 3 blocks, assembly 2 sits downstream of
     the 0,1 fan-in, so the 0→1→2 order is carried by *where the blocks sit in the feedforward graph* — independent of the
     specific learned weights. Shuffling the weights leaves the geometry (hence the order) intact. This is a **test artifact
     of the discrete-3-assembly + hand-wired-fan-in construction**, not evidence that biology is shuffle-insensitive.
  2. **Discrete-ignition metric on a flat fan-out.** Reading "did assembly N ignite" on a flat lower-triangular fan-out
     yields co-fire `[3,3,3]`; there is no graded band for a *moving bump* to ride, so there is no order to collapse.

- **The CORRECT shuffle (adopt Ecker's exactly):** on a **graded near-diagonal band over ≥8 assemblies / a continuum**,
  **permute the row/column identity of the banded matrix** (destroy the near-diagonal clustering; **preserve** the weight
  marginal distribution). Predicted: the **population-decoded trajectory coherence collapses to chance** and (as in Ecker)
  the SWR-like events themselves degrade. Because there is no hand-wired fan-in and no discrete relay, a collapse now
  *proves* the order rode the learned band, not geometry.

- **Add reverse-store as the cleaner direction control (better than reverse-cue).** Store the sequence **reversed** (or use
  the **symmetric** rule) → the decoded trajectory reverses / becomes bidirectional. This is the direct "the WEIGHTS carry
  the order/direction" test and matches Ecker's symmetric-vs-asymmetric contrast.

- **Do NOT over-constrain with "reverse strictly at chance."** Biology replays the SAME store forward AND reverse (Kandel
  Fig. 5-2 / Diba-Buzsáki 2007; Ecker symmetric rule). Requiring reverse≈chance is only valid if you deliberately encode
  with the **asymmetric** rule (Ecker's forward-only regime) — which the gap#5 forward-asymmetric store does, so
  reverse-collapse is an *acceptable* control **for that regime**, but it is a modeling choice, not a law. The load-bearing
  controls are **structure-shuffle-collapse + reverse-store-reverses + interior-seed-invariance**.

- **Do NOT require "a single adjacent link alone must ignite the next across a coincidence threshold."** That discrete-relay
  idealization is not how biology hands off (Q1/Q3): it is a graded population bump. Requiring it is what pushed the arc
  into the fan-in-geometry corner.

---

## Q3 — Minimal point-neuron mechanism for DISCRETE SEQUENTIAL (not co-firing) forward hand-off in the SWR state

**Fully specified and point-neuron-validated by Ecker 2022 (AdExpIF, no dendrites). Three ingredients — the arc already
has two; the missing one is the STORE SHAPE:**

1. **SWR transient = feedback-inhibition-generated (FINO) ripple in a strongly-driven recurrent PV-basket network; E/I
   ~balanced during the event.** Ecker: "ripple oscillations are generated in the recurrently connected PVBC network by the
   FINO mechanism"; removing PVBC→PVBC connections "eliminated ripples entirely while preserving replay" (recurrent
   inhibition necessary+sufficient for the ripple). Buzsáki (p. 346, L7076/L14452): during SWR "excitation transiently
   exceeds inhibition by 3–5×," a ~100 ms self-terminating envelope, order carried by **spike TIMING not rate** (Chrobak-
   Buzsáki). ⇒ the arc's SWR-state E/I-transient envelope readout is the right state. **Already built.**

2. **SEQUENTIAL travel (not co-fire) = a GRADED near-diagonal band + SPIKE-FREQUENCY ADAPTATION.** Ecker's decisive control:
   remove adaptation (plain ExpIF) → "**there was no sequence replay for any scaling … it was a STATIONARY rather than a
   moving bump.**" With adaptation, the just-fired leading edge fatigues, feedback-I keeps the bump sparse (one gamma
   slot / ripple trough at a time), and the graded band's **next-strongest** forward weights pull the bump one step forward
   → A→B→C travels. **This is exactly the arc's co-fire `[3,3,3]` diagnosis:** a flat fan-out has no "next-strongest"
   gradient, so with a strong transient every downstream assembly fires together. Point-neuron Izhikevich HAS intrinsic
   adaptation (`u`); the banked "intrinsic-fatigue" work was calibrating precisely this.

3. **The MISSING ingredient = the GRADED near-diagonal BAND itself** (not a flat BTSP fan-out). On point neurons this is
   produced by a **theta-compressed OVERLAPPING encode + STDP's fast decaying kernel** (D.67; Ecker's asymmetric τ=20 ms
   rule). **STDP "wrote nothing" ONLY because of the HARD reset** (`_silence_soma_apical` empties the Δt window). In a
   theta-faithful encode the assemblies **overlap** (A still decaying as B onsets, no hard silence) → STDP sees
   pre(A)-before-post(B) at small Δt = strong A→B, at larger Δt = weaker A→C = an **adjacent-dominant distance-decaying
   band with reverse≈0**. The hard reset was the bug; removing it both restores STDP *and* fixes the fan-out.

**Explicitly NOT the mechanism (banked, point-neuron-confirmed dead ends):** intrinsic dendritic phase-precession /
Kamondi single-cell pacemaker (6-seed NEGATIVE on BOTH point AND two-compartment substrates, 2026-07-24) — a fragile order
carrier and unnecessary; Ecker gets sequential replay on pure point neurons with band+adaptation+FINO. Ignition-from-scratch
detonators (spontaneous-bistable 1/6, DG-detonator 0/32) — the order comes from the moving bump on the band, not from
igniting discrete assemblies.

---

## Q4 — Ranked cheap-first de-risks + the RIGHT GO bar

**#1 (CHEAPEST, encode-only, run FIRST) — build the GRADED near-diagonal BAND (fix the fan-out).** Replace the hard-reset
discrete encode with a **theta-compressed OVERLAPPING sweep** over **≥8 assemblies** (or a place-field continuum) and let
**STDP** write the band (remove `_silence_soma_apical`; graded onset overlap so A decays through B's spike). If STDP on the
ca3→ca3 recurrent is too weak on our schedule, fall back to a **short-eligibility BTSP** (`btsp_elig_tau_ms` ~100–200 ms,
config-only) or Ecker's asymmetric-STDP parameters (τ=20 ms). **Encode-only GO pre-check (no rest phase, ~minutes):**
`adj_fwd > skip_fwd > skip2_fwd … ` **monotone-decaying forward**, `adj_dominance > 0.6` (vs the current 0.4), reverse ≈
baseline, within-attractor preserved (≥ ~27). Anti-cheat: the **hard-reset control** must reproduce the flat/inverted
fan-out (adj_dominance ~0.4) — proving the overlap is load-bearing. *This is the single highest-leverage change; it targets
the actual defect the parallel run found.*

**#2 (the real readout GO) — SWR-envelope moving-bump replay on the graded band, scored by POPULATION DECODING.** Reuse the
existing SWR-envelope machinery (transient E>I, self-terminating, noise-seeded) on the ≥8-assembly graded band; rely on
**SFA-driven travel + feedback-I/gamma sparsification** for sequential hand-off. **Metric = the field-standard replay score,
NOT discrete assembly ignition:** Bayesian population decode of position-over-time within each SWR event → constant-velocity
path-fit → **weighted correlation(position, time)** (Davidson-Kloosterman-Wilson 2009; the exact method Ecker uses), or the
simpler **Spearman rank-order correlation of first-spike times** (Foster-Wilson 2006). Forward event = positive slope,
reverse = negative.

**THE RIGHT GO BAR (precise — supersedes the 3-assembly single-link shuffle bar):**
- **Store:** graded near-diagonal band, **≥8 assemblies** (ideally a continuum), on **generic uniform baseline
  connectivity** (no hand-wired fan-in).
- **Primary:** significant **forward** replay events (score > 95th percentile of a **cell-identity shuffle** null,
  Ecker/Davidson) with **forward fraction ≥ 1.5× chance AND forward > reverse**, over **≥10–20 detected events** (report
  `n_events`; the prior n=2–4 is non-inferential).
- **Load-bearing anti-cheats (each retires a specific confound):**
  1. **STRUCTURE-SHUFFLE = COLLAPSE** (Ecker's control): permute row/column identity of the banded matrix (destroy the
     near-diagonal band, **preserve** the weight distribution) → replay score → chance AND SWR-like events degrade. ← the
     corrected, biologically-exact shuffle-collapse bar.
  2. **REVERSE-STORE / SYMMETRIZE → REVERSES / BIDIRECTIONAL** decoded trajectory (weights carry direction).
  3. **INTERIOR-SEED INVARIANCE** (`not_just_seed0` = TRUE): seed interior positions (k=2,3,4,…) → forward continues from
     each. Biology: "a cue can initiate recall at any segment of the episode" (Buzsáki p. 350, L14615). Removes the seed-0
     start lock.
  4. **ADAPT-LESION (SFA→0) → stationary bump / co-fire** (Ecker's necessary-adaptation control).
  5. **NO-ENCODE → no coherent trajectory; NO-NOISE acid → silent; FROZEN-plasticity byte-hash; NUMPY-REFERENCE guard**
     (no host per-step silence/argmax in the loop).
- **NOT required (explicitly relaxed vs the prior bar):** reverse strictly at chance (biology does both — only require it
  under a deliberately asymmetric encode); a single adjacent link alone igniting the next (the discrete-relay idealization).

**#3 (fallback if reuse-by-import can't produce the band) — port Ecker 2022's exact recipe on our AdEx/Izhikevich point
neurons:** symmetric-or-asymmetric STDP on overlapping-field inputs → banded recurrent; AdEx adaptation for travel; a
recurrent PV-basket (FINO) for the ripple; Davidson decode for scoring. This is a documented, point-neuron-complete
solution to the whole capability and de-risks the "is it even possible on point neurons" question to ZERO (it is —
published, on our substrate class).

**Residual honest risk:** the genuine de-risk is a TUNING band inside a VALIDATED mechanism (Ecker proves the regime on
point neurons): band-sharpness × adaptation-strength × transient-depth × noise-σ. Under-sharp band or weak adaptation →
co-fire; over-strong transient → detonate/co-fire; too-short inter-event rest (< ~1/a_abs) → fatigue-lock `[0,0,0]`. Not a
substrate wall.

---

## Citations (read in depth, not skimmed)
- **Ecker, Bagi, Vértes et al. 2022, eLife 11:e71850** — SWR + replay emerge from structured CA3 recurrent weights on
  **AdExpIF point neurons**: graded near-diagonal band from overlapping fields; symmetric→bidirectional, asymmetric→
  forward-only; **adaptation required for a traveling (vs stationary) bump**; PVBC-FINO generates the ripple; Bayesian
  decode + constant-velocity fit; **column-shuffle destroys replay** ("structure not statistics"). *Our exact substrate
  class — a complete recipe.*
- **Buzsáki, *Rhythms of the Brain* (2006)** — SWR section pp. 344–350 (L14400–14620): SWR = self-organized CA3 recurrent
  transient E>I 3–5× gain, ~100 ms, spike-timing not rate; **order rides the strongest learned weights**; overlapping-field
  pairwise correlations preserved (Wilson-McNaughton 1994; Lee-Wilson 2002); cue can initiate at any segment. Ch 11
  pp. 314–323: monotonic phase precession, theta compression → STDP sequence coding.
- **Kandel 6e, Ch 5 pp. 101–102, Fig. 5-2** (L7938–8000) — SWR replay of **discrete trajectories**, **forward OR reverse**,
  10–20× compressed, read by **spatial population decoding** (Diba & Buzsáki 2007).
- **Feature catalog** — D.67 (theta sequences → higher-order STDP band), D.19 (SWR forward/reverse replay), D.03/D.13
  (CA3 autoassociator is **sequential**, theta-paced), N.16/N.17 (SWR = intrinsic CA3 event; awake forward + reverse replay).
- **Davidson, Kloosterman & Wilson 2009, Neuron** — the weighted-correlation Bayesian replay score (the field-standard,
  Ecker's method). **Foster & Wilson 2006** — reverse replay + Spearman rank-order metric. **Diba & Buzsáki 2007** —
  forward + reverse replay from the same store.

**Status:** research gate CLOSED with a source-grounded, point-neuron-complete answer. The imaginative-replay CAPABILITY
stays OPEN; the shuffle-collapse bar is RETAINED but CORRECTED (structure-shuffle on a graded band, population-decode
metric, interior-seed invariance, reverse-store); the FAN-OUT store is the real defect and the graded-band encode (#1) is
the cheap first move. NO `sim/` edit anticipated (encode is drive/schedule + existing STDP/BTSP config; readout reuses the
SWR-envelope machinery). Owner reviews + commits.
