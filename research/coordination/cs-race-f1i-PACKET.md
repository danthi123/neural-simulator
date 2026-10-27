# PACKET cs-race-f1i (Claude, 2026-10-27): dendrite-targeting (SOM-like) inhibition ends the older word's held apical plateau

host-decision: scaffold
decides: which engram each heard cue or mention word connects to (the probe wires cue->E_k and mention->E_k at write time, the cs-race declared residual), which words an encode turn presents to the mention layer (the fact's three words in heard order), that a heard word's sensory volley reaches both the basal (somatic) and the apical arbor of its word-form slot, and (new in f1i) the hand-wired SOM-like pool: every mention cell -> every SOM cell, every SOM cell -> every mention cell's apical compartment (the target compartment is fixed by wiring, as Martinotti axon targeting is fixed by cell type)
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement); the branch pre-decided in research/coordination/cs-race-f1h-PACKET.md step 0 (C6 passes, C3 fails -> dendrite-targeting inhibition)
replaced-by: write-time Hebbian binding of heard word forms to the engram being written, on the composer's spiking store write path, with heard word forms arriving through the hearing path and its own laminar targeting
cap: 2 builds

`replaced-by:` is f1h's text verbatim, so CLASS HD counts this packet in f1h's scaffold group (one cap, not a fresh count).
The SOM wiring has its own neural replacement, named here because the field holds one text: inhibitory plasticity
(the engine's `_apply_inhibitory_stdp`) self-organizing the slot <-> SOM connectivity from an unstructured start.

Inputs: `cs-race-f1h-PACKET.md` with Amendment A1, Hermes's P0 answer `cs-race-f1h-0-PROGRESS.md`, the f1h flags build
(`research/runners/cs_race_f1.py`, `tests/test_cs_race_f1h_flags.py`, merged 3b1ccaf34) and `[cs-race-f1h-0] RESULT` when
posted. One default-OFF `sim/` edit is REQUIRED (section E). It is specified here and NOT built by this packet. Probe-local
runner flags, CPU, numpy, pool, memcap. Gate seeds are not run. Hermes runs STEP 0 only after the E build is merged,
then posts the RESULT and STOPs.

## Why C3 fails by construction in f1h
- The BDSP apical update (`sim/bridge.py:10656-10684`) integrates `cp_v_apical` from three terms only: the runner-written
  `cp_bdsp_apical_drive`, the v-gated self-regen sustain, and the KIR down-state. No synaptic conductance of any kind
  reaches `v_apical`.
- Every inhibitory spike lands in the somatic `g_i` (`bridge.py:9193-9262`). The apical -> soma coupling is one-way and
  positive-only (`bridge.py:9665-9671`), so somatic inhibition cannot pull an apical plateau down. The f1g pool (W_MI/W_IM)
  therefore cannot end an older word's plateau. The f1h flags build saw this on seed 41 (older plateau never ended, 3/3
  pairs; informational, `sim-rev/f1h_PROGRESS.md`, not a result).
- The record already met this compartment gap twice. Somatic (Kim-Kim) selective inhibition could not gate an apical
  plateau attractor, which pointed at dendrite-targeting inhibition
  (`research/findings/2026-07-21-gap5-2assembly-selective-inhibition-family-NEGATIVE-points-to-dendritic.md`). GABA_B
  as a plateau terminator was UNDEFINED because `I_gabab` reads the somatic voltage
  (`research/findings/2026-07-30-gap5-plateau-termination-is-INTRINSIC-and-the-machinery-already-exists.md`).

## Why this mechanism (biology, read from the record)
- **Martinotti / SOM cells target distal dendrites, and that is where inhibition stops a dendritic depolarization.**
  Kandel 6e Ch. 13 (`sim-catalog/.../kandel-pns-6e/full-book.txt:18727-18775`, Fig. 13-16): an axodendritic IPSP is the
  more effective at "preventing local dendritic depolarization", an axosomatic IPSP at stopping action potentials;
  SOM cells "activate slowly" and their IPSPs grow with repetitive activation.
- **Dendrite-targeting interneurons are recruited by high-rate firing.** Buzsaki, Rhythms of the Brain Ch. 3
  (`Buzsaki-RhythmsOfTheBrain-2006.txt:3226-3243`): their input synapses facilitate, so they fire only after several
  pulses; at a low pyramidal rate inhibition stays perisomatic, at a high rate it shifts to the dendrites.
- **An apical IPSP blocks the dendritic spike.** O'Keefe & Nadel 1978 (`OKeefe-Nadel-1978-HippocampusCognitiveMap.txt:
  6979-6990`, Fujita & Nakamura 1961).
- **External, PubMed IDs verified, abstract-level only; read before binding:** Silberberg & Markram 2007 Neuron (PMID
  17329212, Martinotti-mediated disynaptic inhibition between pyramidal cells); Kapfer et al. 2007 Nat Neurosci
  (17515899, supralinear SOM recruitment by a few high-rate pyramidal cells); Murayama et al. 2009 Nature (19151696,
  deep interneurons control dendritic Ca2+ encoding); Perez-Garci et al. 2006 Neuron (16701210, GABA_B1b inhibition of
  dendritic Ca2+ spikes); Lovett-Barron et al. 2012 Nat Neurosci (22246433, tunable dendritic inhibition).
- **Facilitation values in the record:** pyramidal -> interneuron U_SE 0.01-0.05 with tau_facil 750 ms (Tsodyks &
  Markram 1998, `research/coordination/cs-race-f1f-RESEARCH.md:146-152`).
- **No bound biology entry exists** for SOM/Martinotti in `research/biology/`. The E build binds one
  (`research/biology/dendrite-targeting-som-inhibition.md`) from the Kandel and Buzsaki anchors above.

**Why the newer word wins, with no host comparison.** One shared SOM pool, wired symmetrically. A newly heard word fires
its slot at a high rate, which recruits the pool; the held trace fires at 2-20 Hz (C1) and should not. Every mention apical
then receives the IPSP. The newer word's apical is under its own sensory apical drive during the volley (f1h's declared
laminar targeting), so it latches through. The older word has only its self-regen, so it falls below `v_hold` into the KIR
down-state. Which word survives is set by the world's input timing and the spiking dynamics. No host code compares ages,
identities or rates.

## Ranked alternatives (one chosen; the rest are banked branches)
1. **CHOSEN: shared SOM pool, facilitating input, GABA_A conductance on the apical through real synapses (section E).**
2. Lateral per-slot SOM sparing its own slot (slot i -> SOM_i -> apicals of slots j != i). More hand-wired: selective by
   construction. Pre-decided branch B1.
3. SOM cells with the Izhikevich LTS intrinsic type. Needs per-neuron a/b/d, and the 10-20 RS-centred heterogeneity
   defect applies (its fix flag would have to be on). Branch B2.
4. A GABA_B component on the apical (Perez-Garci 2006): the E edit extended to the GABA_B kernel against `v_apical`
   with E_K. Branch B3.
- **Rejected as host arithmetic (CLAUDE.md non-negotiable 1):**
  - The `enable_bdsp_microcircuit` / `enable_selfpredicting_interneuron` path (`bridge.py:10630-10656`,
    `sim/config.py:748-765`). A runner-installed RATE vector `cp_spi_int_rate`
    (`_gap4_selfpredict_interneuron_inengine_derisk.py:182`) is multiplied through a logical matrix, scattered, and
    SUBTRACTED from the apical drive. There is no synapse in `cp_connections`, no spike, no conductance and no driving
    force: a rate-model projection beside the spiking substrate. As a linear subtraction it also cannot shunt.
  - Runner writes into `cp_v_apical`: the 07-21 apical arm picked a winner by host argmax and shunted the losers'
    `cp_v_apical` toward -75 mV (`research/runners/_riii_ca3_synchronous_assembly_derisk.py:405-445`); the 08-20 d5
    adaptation current is also a runner-side write. Both are banked METHODS, not routes.
- **Rejected on function:** intrinsic termination alone (an apical SK/AHP current) times out every plateau; it does not
  let a NEWER word end an OLDER one. Somatic inhibition (f1h) reaches the wrong compartment by construction.

## E. The required `sim/` edit (default OFF; specified here, built separately by Claude, merged before step 0)
- **E1 config** (`sim/config.py`, beside the BDSP apical fields ~783-791):
  - `enable_apical_gaba_routing: bool = False`
  - `apical_gaba_propagation_strength: float = 1.0`, the per-spike apical conductance increment scale in apical-leak
    units (the apical ODE is leak-normalized: `_dvb` terms are conductance/g_L times driving force).
  - The edit reuses `syn_reversal_potential_i` (-75 mV, config.py:393) and `syn_tau_g_i` (10 ms, config.py:395, the
    cached `decay_i`). It adds no new reversal or time constant.
- **E2 routing key.** A plan entry tagged `receptor="gaba_a_apical"`, the existing receptor key that
  `inject_explicit_wiring` already reads (`bridge.py:4563`).
  - After the coincidence mask (~`bridge.py:4840`), mirroring the nmda_slow mask (4800-4820): build
    `cp_apical_gaba_synapse_mask` (capacity-sized, FALSE padding, grown via `_ensure_gate_capacity`) and allocate
    `cp_conductance_g_apical_gaba = zeros(n)`. Only when the flag is on and at least one pathway carries the tag.
  - FAIL LOUDLY: a tagged pathway with the flag off, or with `enable_bdsp` off, or with
    `enable_two_compartment_dap` on, raises. (Today any non-`gaba_b` value silently acts as somatic GABA_A.)
  - `_step_megakernel_can_dispatch` (~`bridge.py:8329-8350`) returns False when the mask exists.
- **E3 split** (section 2, right after the graded split ends ~`bridge.py:9178`, before
  `conductance_decay_and_current` at 9193):
  - capture `_apical_gaba_src = eff.data * mask`;
  - zero those entries in the matrix the g_e/g_i matvec uses (the nmda_slow AMPA-suppression idiom, 9141-9153).
  - A routed synapse then gives NO somatic `g_i`: the axodendritic IPSP's attenuation at the soma is taken as total.
    The transmission gate and STP factor are already in `eff.data`.
- **E4 conductance** (after the g_e/g_i increments, ~`bridge.py:9320`):
  - `g_ap = g_ap * decay_i`;
  - if any neuron fired last step: `g_ap += (csr(_apical_gaba_src).T @ prev_fired) * apical_gaba_propagation_strength`.
  - `deterministic_transpose_matvec` is honoured exactly as in the GABA_B block (9765-9770).
- **E5 apical term** (BDSP block, after the KIR term at 10683, before the update at 10684). A bounded per-term
  exponential-Euler step:
  `_dvb = _dvb + (_atau/_adt) * (E_i - v_apical) * (1 - exp(-g_ap * _adt/_atau))`.
  - Under the outer `(dt/tau)` factor this moves `v_apical` by `(E_i - v0)(1 - exp(-g_ap*dt/tau))`, which reaches `E_i`
    at most and never passes it, at any `g_ap`. This is the 2026-09-29 overshoot family (config.py:975-997), with the
    derivation of `kernels.py:1119-1160` in leak-normalized units (C/dt becomes tau/dt).
  - To first order it equals the explicit `g_ap*(E_i - v0)`. At `g_ap == 0` the term is exactly 0.0.
  - Per-term bound, as the engine documents: an aggregate overshoot is not excluded. At the gap4 values the explicit
    coefficient of the other terms, `(dt/tau)(1 + sr + g_kir) <= (1/15)(1 + 2 + 1) = 0.27`, stays below 1.
- **E6 tests** (`tests/test_apical_gaba_routing.py`):
  - flag off is byte-identical, asserted by a state hash after N steps (the `test_bounded_conductance_integration::
    test_byte_identical_when_off` pattern); flag on with no tagged pathway is also byte-identical;
  - a routed spike raises `g_ap` only on its targets and leaves somatic `g_i` untouched;
  - a closed transmission gate gives `g_ap == 0`;
  - one step never crosses `E_i` for `g_ap` in {0.1, 10, 1e4};
  - each FAIL-LOUDLY case of E2 raises.
- **E7 bookkeeping.** Register `cp_conductance_g_apical_gaba` beside `cp_conductance_g_gabab` in the flag-gated array
  map (`bridge.py:400`) and the state save/restore lists that carry `cp_conductance_g_gabab`.

## Method (runner flags on `research/runners/cs_race_f1.py`; every default reproduces the f1h runner exactly)
- **Base cell.** The f1h circuit at the cell `[cs-race-f1h-0] RESULT` selects by f1h's own rule with C3 waived: smallest
  `--soma-g`, then smallest `W_IM`, among cells passing C0-C2, C4 and C6 on 8/8. The somatic pool (W_MI 20, that W_IM)
  stays. f1i adds the SOM path on top, so LESION is the f1h circuit plus SOM cells whose output is gated shut.
- **SOM pool.** `NSOM` = 8 cells (f1g's NIM), appended after the f1g pool only when `--som` is given, and declared in
  `output_inhibitory_indices`. The intrinsic type is the probe's default cell, as the f1g pool (LTS is branch B2).
- **Slot -> SOM (facilitating).** `plan["msom"] = SC.dense(all_ment, som, W_MS, conn_type="E_TO_I")`, excluded from the
  runner's stp_disabled loop by its key.
  - Facilitation through per-conn-type STP (`bridge.py:9013-9034`): `enable_per_type_stp = True` with
    `stp_U_per_type = [U_row, 0.05, 0.25, 0.25]`, `stp_tau_f_per_type = [tau_f_row, 750, 50, 50]` and
    `stp_tau_d_per_type = [200, 200, 100, 100]`.
  - Index 0 copies the row's global values, so engram recurrence keeps its STP. Every other E_TO_I, I_TO_E and I_TO_I
    pathway in this probe is stp_disabled (factor 1.0), so indices 2-3 are inert; the read-back records it.
- **SOM -> apical.** `plan["sap"] = SC.dense(som, all_ment, W_SA, conn_type="I_TO_E", receptor="gaba_a_apical")`, with
  its own transmission gate `cs_som` and stp_disabled. `enable_apical_gaba_routing = True` when `--som` is given.
- **Flags:** `--som W_MS`, `--som-ap W_SA`, `--som-lesion` (gate `cs_som` closed for the whole run), `--som-nofac`
  (msom stp_disabled at weight `W_MS * 0.05`: a static synapse with the facilitating synapse's resting efficacy, so the
  one thing NOFAC removes is the use-dependence).
- **Lesion persistence.** `_build` and every `race()` / `cal()` gate reset re-open the named gates; with `--som-lesion`
  the runner closes `cs_som` AFTER each reset and records the read-back at measurement time.
- **Trial reset of the new state (self-skeptic S1, S3).** The quiescent snapshot does not carry STP state or the new
  conductance (`_STATE_ARRAYS`, `_gnw_rung1_ignition_curve_derisk.py:219-227`), and `cp_stp_u` starts at the GLOBAL
  `stp_U` (`bridge.py:1289`; 0.30 at row 3), not at rest. The engine's facilitation is the u -> 0 at rest form
  (`kernels.py:1455-1467`; jump `u + U(1 - u)` at a spike, `bridge.py:9049-9060`). So `_reset_apical` also writes, on
  the msom synapses only, u = 0 and x = 1, and zeros `cp_conductance_g_apical_gaba`, before the snapshot and at every
  trial start. The values are read back per trial. Facilitation must not carry from one item into the next.
- **Per-type STP read-back (S2).** The engine builds the per-synapse conn type from neuron TRAITS
  (`bridge.py:1353-1383`) and silently falls back to all E->E when `inhibitory_trait_indices` is empty. The build
  records the realized type per pathway (msom = 1, ee = 0) and the realized U / tau_f / tau_d per pathway. Any mismatch
  is STOP-ID.
- **STP record split (S4).** The runner's STP record (`cs_race_f1.py:749-760`) averages every STP-active synapse as
  "engram recurrence". With msom STP-active it is split by pathway (ee, msom), so the engram record stays the engram's.
- **Frozen values and where they come from** (nothing here was chosen from a run; no SOM run of any kind exists):
  - U = 0.05, tau_f = 750 ms: the record's facilitating range (f1f-RESEARCH.md:146-152). 0.05 is the upper end and the
    conservative one for item C8: a smaller U makes the held trace's spikes weaker still.
  - tau_d = 200 ms for E_TO_I: the engine default (config.py:1298), untouched.
  - E_i = -75 mV and tau = 10 ms: engine GABA_A defaults. NSOM = 8 and W_MI = 20: f1g's frozen values.
- **Disclosure: no run informs any f1i value.** The only seed-41 information on this lane is f1h's informational soma-g
  scan (soma-g 0.004-400 at W_IM 40; f1h Amendment A1). It reaches f1i only through f1h's own selection rule for the
  base cell. Seed 41 stays a dev seed.
- **Disclosure: facilitation alone is not the high-pass at these values.** Hand arithmetic in the engine's form, with
  the frozen U, tau_f and tau_d, gives a per-spike efficacy of about 0.14 at 100 Hz (peak, spikes 5-6) and about 0.2 at
  a steady 10 Hz (approximate; the build's read-back measures the realized values): tau_d = 200 ms depletes the high-rate
  train. Rate x efficacy is about 14/s against 2/s, a factor of 7. A static
  synapse gives 10. Any selectivity must therefore come from rate and the SOM cell's temporal summation. Whether
  facilitation is NECESSARY is measured by the step-1 NOFAC arm, never assumed.

## Step 0 (Hermes, after the E build is merged): seeds 41-48 only, no race, W_GIS 30, row 3
- **P0' engine support (read-only, first).** Show with file:line that the merged E build is present, its tests pass,
  and the runner's `--som` path allocates `cp_conductance_g_apical_gaba` on the mention cells. Missing or failing:
  STOP-ENGINE.
- **Identity (STOP-ID).** (i) No f1i flag: the calibrate JSON minus `wall_s` hashes equal to the f1h runner at the base
  cell on seeds 41-42. (ii) `--som-lesion`: `g_ap == 0` at every recorded ms and the `cs_som` gate reads back 0.0 at
  measurement time (the TERMS.md lesion condition).
- **Grid (frozen):** `W_SA` in {0.125, 0.25, 0.5} x `W_MS` in {100, 200, 400}: 9 cells x 8 seeds = 72 pool lines.
  - `W_SA`: one synchronous volley of all 8 SOM cells delivers `8 * W_SA` = {1, 2, 4} apical-leak units, i.e. {0.5, 1, 2}
    x the gap4 GO self-regen conductance (`coincidence_plateau_self_regen` = 2.0), the conductance a terminating IPSP
    must oppose.
  - `W_MS`: f1g's slot -> pool weight W_MI = 20 divided by the facilitating synapse's efficacy band, 0.2 (facilitated)
    to 0.05 (first spike), giving 100-400 on a factor-2 grid.
- **Items, on 8/8 seeds:**
  - C0-C6 as in f1h. C3 (overwrite, both orders, the three pairs) is now REQUIRED.
  - C7 (new, the eraser), per C3 pair, with X at 0-150 ms, gap, Y_on = 300 ms, Y at 300-450, window [600, 950):
    - (0) precondition (S5): the older slot's min `v_apical` > `v_hold` over [150, 300), i.e. X latched before Y
      arrived. Otherwise the pair is UNDEFINED: a plateau that never latched cannot be "ended".
    - (a) termination latency: the first ms after Y_on from which the older slot's max `v_apical` stays below `v_hold`
      through the window. Bar: <= 300 ms (the window opens at Y_on + 300). The latency is reported per pair.
    - (b) the newer plateau survives its own inhibition: the newer slot's min `v_apical` > `v_hold` over [Y_on + 150,
      Y_on + 650), from its pulse end (when SOM tails off) through the window.
    - (c) engagement: SOM pool spikes in [Y_on, Y_on + 300) >= 1 per SOM cell, and the older slot's `g_ap` peak is > 0
      and occurs before its termination ms.
    - (d) in-cell lesion: the same pair with `--som-lesion` leaves the older plateau above `v_hold` in the window (3/3),
      AND the rate form of C3 (newer 2-20 Hz, older < 1 Hz) fails on that pair (S6). A rate-C3 pass under the lesion
      means the somatic pool (W_IM) silenced the older soma while its plateau held: that pair's C3 belongs to the
      somatic pool, not the SOM path.
  - C8 (new, selectivity): in the C1 protocol (one word, then 650 ms), the SOM pool rate over [pulse end + 50, pulse end
    + 650) is < 1 Hz. The held trace must not recruit its own eraser. The first 50 ms (5 GABA_A decay constants of
    10 ms) are the heard word's own tail, excluded so the item reads the held trace and not the volley.
  - Raw per-ms SOM counts, per-slot `g_ap` and `v_apical` min/max and the gate read-back are kept, so the scorer re-derives
    every item.
- **Selection:** the smallest `W_SA`, then the smallest `W_MS`, among cells passing C0-C4 and C6-C8 on 8/8. An
  UNDEFINED pair or item is not a pass: 8/8 means eight defined passes.
- **STOP rules:** STOP-ENGINE (P0'), STOP-ID, STOP-CAL (no cell passes), STOP-SMOKE (median C5 below 2 pA in either
  clause), and STOP-ATTRIB. A cell whose C3 passes while C7(c) or C7(d) fails is UNATTRIBUTED, never a pass; if every
  C3-passing cell is unattributed, that is STOP-ATTRIB.
- **Branches already decided** (each posts and STOPs; the next ruling takes the named branch, never a retune of this
  grid):
  - Gate: if `[cs-race-f1h-0] RESULT` shows no cell passing C6, or a STOP-ENGINE / STOP-ID, f1i does not run. The
    carrier does not latch, and the ruling returns to the carrier, not the eraser.
  - B1: C7(a) passes but C7(b) fails on every cell (the newer word dies with the older): alternative 2.
  - B2: C8 fails on every cell (the held trace recruits SOM and erodes C6): alternative 3.
  - B3: C7(c) passes but C7(a) fails on every cell (SOM fires, `g_ap` > 0, the plateau holds): post the max `g_ap`
    reached; alternative 4.
  - B4: C3 and C7 pass but STOP-SMOKE: the trace holds and overwrites but does not reach the engram. That is an f1h
    coupling (soma-g) verdict.
- **Post** `[cs-race-f1i-0] RESULT` (every item per cell, P0', the identity hashes, the chosen cell, C7 latencies, C5
  against the +4/+6 pA flip scale, the STOP verdicts, branch @ sha). Then STOP.

## Step 1 (preregistered; runs only after a Claude RULING on step 0)
- **Arms:** TRACE (the chosen cell), LESION (`--som-lesion`: the SOM -> apical gate closed for the whole run, SOM cells
  still driven and firing, everything else identical), and NO-ENC. Plus NOFAC (`--som-nofac`), a mechanism-necessity arm
  (CLAUDE.md non-negotiable 4), not an ownership bar.
- **Bars:** the f1g step-1 ownership bars, unchanged. Clauses (a) and (b) each pass at 8 or more of 10 on at least 7/8 DEV
  1-8 AND at least 14/16 CONFIRM 25-40, with the positive control, clause (c) and clause (g) intact on those seeds. LESION
  passes (a) and (b) on at most 2/8 DEV.
- **Necessity read:** facilitation is reported NECESSARY iff NOFAC fails the TRACE bars on the seeds where TRACE passes;
  otherwise it is reported as not necessary for this capability and kept (faithfulness-first).
- **Engagement:** per draw, the mention spike counts, the plateau-latched flag, the SOM spikes in the newer word's window
  and the older slot's `g_ap` peak. A draw with no latched plateau or no SOM engagement is UNDEFINED.

## Self-skeptic record (Claude, 2026-10-27, before posting)
- Host arithmetic? No step chooses content on the host. The SOM wiring and its apical target are declared structure
  (`decides:`). The trial-start writes (u = 0, x = 1, `g_ap` = 0) are trial-boundary resets of the same class as
  `_reset_quiescent`, not per-step decisions. Who survives is decided by spikes and the world's input timing.
- One-variable lesion? LESION closes one transmission gate on the only output of the SOM cells; they keep firing.
  It is re-closed after every gate reset and read back (S7). NOFAC changes the use-dependence only, at matched resting
  efficacy.
- Can C3 pass without the SOM path? Four guards: C7(0) the older plateau latched first (S5); C7(c) SOM spikes and a
  `g_ap` rise precede the termination; C7(d) the lesion leaves the plateau up AND fails rate-C3 (S6); STOP-ATTRIB.
- Values from a smoke? None (see the disclosure). The grid comes from W_MI, the gap4 self-regen, NIM and the
  Tsodyks-Markram range; the C7 bar comes from the C3 schedule.
- Silent engine fallbacks found while reading, now guarded: STP state and `g_ap` outside the snapshot (S1); the
  trait-built conn type that falls back to E->E (S2); `cp_stp_u` starting at the global U (S3); the pooled STP record
  (S4); the volley tail inside the C8 window (S8).
