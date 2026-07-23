# gap#5 forward-asymmetry mechanism — RESEARCH GATE (supersedes the refuted causal-STDP prescription) — 2026-07-23

**READ-ONLY deep-research gate.** The prior gate (`2026-07-23-gap5-replay-R0-...`) proved the store is near-symmetric
and prescribed **causal spike-timing STDP** as the fix. That prescription was then **empirically REFUTED**: a pure-LTP
`a_plus=0.5` STDP test produced **ZERO weight change** on the `ca3->ca3` recurrent — STDP structurally does not write
this recurrent under our encode schedule. This gate finds the forward-asymmetry mechanism that is **compatible with the
substrate's actual rules (BTSP + rate-Hebbian), NOT STDP**, ranks cheap-first, and recommends the single smallest
encode change. Verified against the code; grounded in the sequence-replay + phase-precession literature (read in depth).

---

## 0. Why STDP wrote nothing (the mechanistic root — so we don't re-propose a timing rule)

The encode schedule is **BTSP-shaped, not STDP-shaped**, and the two are mutually exclusive here:

- The chain drives assemblies A→B→C in **theta-separated windows** with a hard reset between them
  (`_silence_soma_apical(settle=2)` — clears `v/u/firing/conductances/v_apical`, `_gap5_sequence_replay_derisk.py:96-123,264`).
  The reset **explicitly silences A's soma before B fires**. So at the moment B spikes, A is **not spiking** → the
  recurrent A→B synapse never sees a pre-spike coincident with the post-spike → **STDP's Δt window is empty → dw=0.**
- BTSP works precisely *because* it does **not** need coincident spikes: it reads a **presynaptic eligibility trace**
  (`cp_btsp_pre_elig`, a seconds-long low-pass of firing, `btsp_elig_tau_ms=1000`, `bridge.py:8042-8046`) that
  **survives the soma silencing** — A's eligibility persists into B's plateau window even though A's soma is dark. The
  rule is `dw = eta · Etilde_pre · IS_post · (w_max−w)` (`config.py:339-350`; `fused_btsp_update`), i.e. potentiate
  (pre-that-was-recently-active) → (post-that-is-plateauing).

**⇒ The theta reset that makes BTSP directional is exactly what makes STDP silent.** Any proposed mechanism must ride
the eligibility×plateau rule, not spike timing. (This is also why the banked E→E short-term-depression method failed —
STD moves a bump but does not *write* a directional weight; see §1.)

---

## 1. Diagnosis reframe — the genuine residual is SMALL and is a SEPARATION problem, not a missing asymmetry

**The asymmetry mechanism already exists and already works — weakly.** BTSP with a presynaptic eligibility trace + a
forward-swept drive **is** a temporally-asymmetric *rate* learning rule (Blum & Abbott 1996 — see §2, Q3). In a
forward-only sweep A→B→C with the theta reset:

- Drive A → A builds pre-eligibility. Theta-reset (A silenced, **eligibility kept**). Drive B (+ plateau) → BTSP writes
  **A→B** (A eligible × B plateau). B builds eligibility. Reset. Drive C → writes **A→C, B→C**.
- The **reverse** links do not form spontaneously: for B→A we'd need B eligible when A plateaus, but A plateaus
  **first** (before B ever fires) → B has no eligibility → **B→A ≈ 0 by construction.**

So a *pure* forward sweep writes *only* forward links. The measured near-symmetry is **not** a failure of the rule — it
is caused by **two SYMMETRIC/REVERSE contaminants deliberately in the current schedule**, quantified from R1:

| encode | within | adj_fwd | adj_rev | ratio | what it shows |
|---|---|---|---|---|---|
| wr=8, cf=24, **cr=8** (baseline) | 173.7 | 143.3 | 142.0 | 1.01× | fully symmetric |
| **wr=0**, cf=24, cr=8 (chain only) | 5.9 | ~6.3 | ~5.0 | 1.27× | chain-only forward bias is REAL but small |

Decompose the recurrent matrix the encode writes:

> **W = W_within** (symmetric, REQUIRED, ~174) **+ W_chain_fwd** (asymmetric forward, ~6, the real signal)
> **+ W_chain_rev** (from the `chain_rev=8` REVERSE sweeps → part of the ~5 adj_rev)
> **+ W_refresh_between** (SYMMETRIC, **~137 on BOTH adj_fwd and adj_rev**, an unwanted byproduct of the within-refresh).

The within-refresh (`--within-refresh`, `:294-319`) is **required** — removing it collapses the within-attractor
(174→6, below the RANK-1 completion floor ~27). But it is run with a **BISTABLE (latching, completing) plateau** on an
already-linked chain, so driving assembly m **completion-spreads through the existing recurrent links to its neighbor
m+1**, co-activating m+1 with its own plateau → BTSP writes **m↔m+1 in BOTH directions symmetrically** (~137 each). This
symmetric ~137 **swamps** the chain's ~6 forward bias. (The code comment at `:296-299` asserts the refresh leaves
cross-links untouched; the R1 numbers prove it does not — the completion-spread is the leak.)

**The genuine, irreducible residual (precisely):** rebuild the within-attractor (symmetric co-activity) **without**
also writing symmetric between-adjacent links, and drop the explicit reverse sweeps. Once both symmetric contaminants
are removed, the *already-present* BTSP forward bias (W_chain_fwd) is the sole between-write → adj_fwd ≫ adj_rev.
**Most of the "blocker" is solved; the residual is an encode-schedule separation of two writes that currently collide
in one matrix.**

---

## 2. Literature grounding (read in depth, not skimmed)

**Q1 — how real CA3 builds forward-asymmetric recurrent weights, and which rule is compatible with BTSP (not STDP):**

- **Ecker et al. 2022, eLife 71850 — OUR substrate class (bistable CA3 assemblies + SWR replay).** Decisive, directly
  on point: a **symmetric** learning rule → "**both forward and backward replay**"; an **asymmetric** kernel →
  "**sequence replays occurred only in the forward direction**" / "Learning with an asymmetric STDP rule leads to the
  **absence of backward replay**." **Two independent requirements they separate:** (i) **asymmetric weights set the
  DIRECTION** (forward vs reverse/bidirectional); (ii) **cellular ADAPTATION makes the bump TRAVEL** — "without
  adaptation… no sequence replay… a **stationary** rather than a moving bump." Feedback (PV-basket) inhibition makes
  the ripple, **not** the direction. ⇒ our forward-asymmetric-weight target is exactly Ecker's asymmetric-rule
  condition; and **travel is a separate co-requirement (adaptation), see §5.**
- **Blum & Abbott 1996, Cereb. Cortex 6:406 — the rate-based (NOT spike-timing) asymmetric rule.** Formulated in a
  **rate-coding** picture with an asymmetric potentiation window for **pre-before-post of "a few hundred ms to a few
  seconds."** This is the canonical way to build forward-asymmetric recurrent weights **without STDP**, and its time
  window is **exactly our BTSP presynaptic eligibility (`btsp_elig_tau_ms=1000`).** ⇒ **our BTSP + forward-sweep IS a
  Blum-Abbott asymmetric rate rule.** Prediction (their result): forward-shifted/predictive fields — the same
  asymmetry we want.
- **Bittner, Milstein, Magee et al. 2017, Science (aan3846) — BTSP is itself temporally ASYMMETRIC.** Inputs active in
  a seconds-long window **preceding and immediately following** a plateau are potentiated; inputs **outside** the window
  are depressed; the CA1 integrated-signal decay makes the kernel **temporally asymmetric** (pre-before-post arm
  dominant). ⇒ the substrate's own BTSP rule carries an intrinsic forward asymmetry — we are not bolting one on.
- **Milstein, Magee et al. 2021, eLife 73046 — bidirectional BTSP** (potentiation + a depression arm, weight-dependent).
  Already exposed as knobs (`btsp_milstein_k_pot/k_dep`, `btsp_hetero_dep`, `btsp_mean_subtract`, `config.py:351-375`).
- **Mehta, Barnes, McNaughton 1997, PNAS 94:8918 + Mehta, Quirk, Wilson 2000, Neuron — the in-vivo signature.** Place
  fields expand **backward** (opposite to travel), NMDA-dependent — the behavioral fingerprint of forward-asymmetric
  recurrent weight formation by a temporally-asymmetric rule (their own framing). CA3 fields backward-shift then
  **stabilize** — i.e. CA3 *does* end up with the asymmetric recurrent structure.
- **Skaggs, McNaughton et al. 1996 — theta phase precession / theta sequences** compress a behavioral-timescale order
  into a plasticity-window-sized forward sweep. Our theta-separated A→B→C chain is the discretized version.

**Q2 — keeping the WITHIN (symmetric) separate from the BETWEEN (asymmetric) in ONE recurrent matrix:**

- **The SR-symmetry papers** (Fang, Aronov, Abbott, Miller 2023, eLife 80680; Bono/George/Clopath 2024, PLOS Comp Biol
  1013056 / bioRxiv 595705) make the biological tension explicit: a **symmetric** CA3-recurrent rule yields a
  **symmetrized** (bidirectional) predictive map, while the **asymmetric/forward** map is carried by the **feedforward
  CA1** rule. ⇒ a near-symmetric CA3 recurrent is *biologically expected*; forcing forward-dominance requires either
  (a) the asymmetric-write component to be **protected from the symmetric-write component**, or (b) an asymmetric
  **dynamics** at replay. Our fix is (a).
- **The biological separation is pattern-separation + state-gating:** during encoding, strong feedforward/feedback
  inhibition (DG sparsification + CA3 baskets) keeps the pattern **local** so the auto-associative (within) write does
  not recruit neighbors; the hetero-associative (between) write is carried by the theta *sequence*. In the model this
  maps to **freezing the between-synapse plasticity while the within-attractor is (re)built** — a per-synapse plasticity
  gate is exactly a state-gated / heterosynaptic separation.

**Q3 — ML/comp-neuro mechanisms for forward-asymmetric weights WITHOUT STDP, mapped to our knobs:**

- **Rate-based asymmetric Hebbian with a presynaptic eligibility trace** (Blum-Abbott 1996; Rao-Sejnowski; differential
  Hebbian) → **already our BTSP** (`btsp_elig_tau_ms`, forward-swept `_drive_window` order, `chain_fwd` sweeps).
- **Romani & Tsodyks 2015 — symmetric weights + short-term DEPRESSION → traveling bump but with RANDOM direction**
  ("bursts start at seemingly random locations, travel backward OR forward"). ⇒ **STD alone cannot pick forward** —
  confirms our banked STD-negative and confirms **weight asymmetry is required for start-invariant forward replay.**
- **Heterosynaptic subtractive normalization** (Miller-MacKay 1994) → exposed as `btsp_mean_subtract` — a competition
  that removes the common (symmetric) pedestal, keeping the differential (asymmetric) component (ranked #3 below).

---

## 3. Ranked cheap-first mechanisms (all compatible with the BTSP/Hebbian recurrent)

**#1 (RECOMMENDED) — Remove the two symmetric contaminants: forward-only chain + freeze the between-synapses during
the within-refresh.** Biological basis: pattern-separation/state-gating keeps the auto-associative encode local
(Q2); forward-only theta sweeps are the Blum-Abbott/BTSP asymmetric write (Q1/Q3). Realized by **existing machinery,
runner-side, NO `sim/` edit**:
  - `chain_rev=0` (a plain CLI arg, `--chain-rev 0`, already exists `:609`) → no explicit reverse write; adj_rev falls
    to the un-written baseline (`encode_ca3w≈0.5`).
  - Freeze between-synapse plasticity during the refresh block: set `bridge.cp_plasticity_rate_gain[between_flat] = 0`
    for the within-refresh loop, restore `= 1` after. **The BTSP write is gated per-synapse by exactly this array**
    (`bridge.py:156-159`: `new_w = cur_w + (new_w − cur_w) · gain_bt`), and `between_flat` is **already computed from
    `asm_of_local` + the CSR structure** (the classification at `:328-343` runs post-encode, but the same
    `_extract_ca3ca3_vec` + `asm_of_local` are available *before* the refresh, `:160-166`). The runner already writes
    `cp_*` arrays directly (e.g. `structural_sep`/`selective_inhib` zero `conn.data` at flat indices), so this is the
    same idiom. (If `cp_plasticity_rate_gain is None` because no named gate allocated it, force it once via
    `bridge.set_all_plasticity_gain(1.0)`, `bridge.py:3946-3977`.)
  - **Expected effect:** adj_fwd ≈ W_chain_fwd (~6, pure) ; adj_rev ≈ baseline (~0.5) → **ratio ≈ 10-12×** ; within
    rebuilt by the (between-frozen) refresh (**≥ ~170**, unimpeded on within synapses). Hits the GO gate
    (ratio ≥ 2-3× **and** within ≥ 27).

**#2 — Amplify the forward chain write (only if #1's adj_fwd ≈ 6 is dynamically too weak to seed the next assembly).**
Biological basis: a longer plateau on the *post* assembly widens the eligibility×plateau overlap (BTSP kernel width).
Realized runner-side: use the **BISTABLE plateau on the POST assembly during the forward-only sweep**, but keep the
theta reset that de-latches the **PRE** assembly's apical *before* the post fires (so A's plateau is OFF when B fires →
A→B strengthens, **B→A stays 0**). Combine with longer `seq_win_steps`. Risk: if the reset does not fully de-latch a
bistable pre-plateau, a reverse link can leak — hence #2 is the amplifier, gated behind #1's clean separation. NO
`sim/` edit (drive/schedule change).

**#3 — Heterosynaptic subtractive normalization on the between-write (`btsp_mean_subtract`, Miller-MacKay 1994).**
Biological basis: per-post-cell zero-sum competition removes the symmetric pedestal, keeps the differential forward
component. Already a knob (`config.py:372-375`). **Lower confidence:** a *related* multiplicative hetero-dep knob was
R1-NEGATIVE (inverted the asymmetry + crushed the within, `btsp_hetero≥0.1`). Subtractive (zero-sum) differs from
multiplicative, but it also fights the *within* (a symmetric co-active write) → risk of collapsing the attractor. Rank
below the surgical #1.

**#4 — Cellular-adaptation TRAVEL co-requirement (Ecker 2022) — NOT an asymmetry mechanism, but the other half of the
GO.** Even with adj_fwd ≫ adj_rev, Ecker shows the bump only **travels** A→B→C if pyramidal cells **adapt** (else it is
a stationary bump). Our Izhikevich substrate has intrinsic adaptation (`u` recovery); the banked "intrinsic-fatigue"
work was calibrating exactly this. **Flag:** if #1 achieves the ratio but forward replay still fails, the residual is
the **travel dynamics (adaptation), not the weights** — a distinct, already-scoped lever, and the ratio de-risk cleanly
isolates the weight question from it.

**#5 — Graded theta-sequence overlap (softer reset instead of hard silence).** More biologically faithful phase
precession (A still decaying as B onsets → graded pre-before-post) → a cleaner Blum-Abbott asymmetric write. Higher
variance / more tuning; deprioritized below the surgical #1.

**Banked (do not re-propose):** causal spike-timing STDP (writes 0 — §0); E→E short-term depression alone (travels but
random-direction — Tsodyks-Romani; §1); multiplicative heterosynaptic-depression (inverts + crushes within — R1);
`within_refresh`/`chain_fwd` count knobs (saturate, no asymmetry — R1).

---

## 4. The single recommended CHEAP-FIRST de-risk

**Forward-only chain (`chain_rev=0`) + between-synapse-frozen within-refresh** (mechanism #1). The smallest possible
encode change (one existing CLI arg + one per-synapse gate write already computable from `asm_of_local`), **NO `sim/`
edit**, purely additive/default-preserving (a `--freeze-between-refresh` flag, off = byte-identical to today).

**Target:** `adj_fwd/adj_rev ≥ 2-3×` (predicted ~10×) **AND** within-attractor preserved `≥ ~27` (predicted ~170).

**Anti-cheat controls (the GO gate — the runner already has all of these):**
- **ASYM-LESION** (symmetrize the between weights → forward replay collapses to chance): proves the asymmetry, not a
  readout trick, is load-bearing. *This is the direct test that the fix produced real weight asymmetry.*
- **START-INVARIANCE** (noise-ignite from a RANDOM assembly, not A → forward fraction holds): the R0 acid test —
  asymmetric weights give forward-from-any-ignition; symmetric+STD would give random direction (Tsodyks-Romani). This is
  the load-bearing control that killed the prior "rode start=A" artifact.
- **SCRAMBLE-BETWEEN** (permute between-edge weights → order collapses to the shuffled floor), **NO-ENCODE** (no ordered
  events), **NO-NOISE acid** (silent rest), **FROZEN-plasticity + dendrite-reset** (retire the Wang/`_hard_silence`
  confounds). All already wired (`:548-595`). 6-seed on GPU after the CPU-smoke ratio check.

**Fast pre-check (cheapest possible, no rest phase):** `--encode-only` reports `adj_fwd/adj_rev/within` directly
(`:644-654`). Verify the ratio jumps to ≥3× and within stays ≥27 **before** spending the rest-phase/GPU run.

---

## 5. Verdict — **SURPASSABLE, cheaply**

The forward-asymmetry capability is **surpassable without a new mechanism and without a `sim/` edit.** The substrate's
own BTSP rule + a forward-swept theta chain **is** a Blum-Abbott / temporally-asymmetric rate rule (Bittner-Milstein-
Magee 2017 confirm BTSP's kernel is itself asymmetric), and Ecker 2022 — *our substrate class* — confirms an asymmetric
rule gives forward-only replay. The near-symmetric store is **not** a substrate limit; it is **two schedule contaminants
(reverse sweeps + a completion-spreading within-refresh) writing symmetric between-links that swamp the real ~6 forward
bias.** Removing both with existing per-synapse plasticity gating (#1) is predicted to yield adj_fwd/adj_rev ≫ 3× with
the within-attractor intact.

**The one genuinely-hard piece is separable and already scoped:** Ecker's second requirement — **cellular adaptation to
make the bump TRAVEL** (not merely point forward). The weight-asymmetry de-risk (#1) isolates the DIRECTION question; if
the ratio-GO does not yield functional forward replay, the residual is the adaptation/travel dynamics (the
intrinsic-fatigue lever), not the weights. STDP is correctly retired (it writes nothing here); the path forward is the
BTSP-compatible forward-only + frozen-between refresh.

### Citations
Ecker et al. 2022, eLife 71850 · Blum & Abbott 1996, Cereb. Cortex 6:406 · Bittner, Milstein, Magee et al. 2017,
Science (aan3846) · Milstein, Magee et al. 2021, eLife 73046 · Mehta, Barnes, McNaughton 1997, PNAS 94:8918 ;
Mehta, Quirk, Wilson 2000, Neuron · Skaggs, McNaughton et al. 1996 (phase precession) · Romani & Tsodyks 2015,
Hippocampus · Fang, Aronov, Abbott, Miller 2023, eLife 80680 ; SR-symmetry, PLOS Comp Biol 1013056 / bioRxiv 595705 ·
Miller & MacKay 1994 (subtractive normalization) · Stachenfeld, Botvinick, Gershman 2017, Nat. Neurosci.
