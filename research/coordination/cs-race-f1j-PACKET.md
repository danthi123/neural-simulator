# PACKET cs-race-f1j (Claude, 2026-10-28): an input-driven, decaying, Mg-blocked NMDA conductance holds a heard word's apical plateau through its own SOM IPSP

host-decision: scaffold
decides: which engram each heard cue or mention word connects to (the probe wires cue->E_k and mention->E_k at write time, the cs-race declared residual), which words an encode turn presents to the mention layer (the fact's three words in heard order), that a heard word's sensory volley reaches both the basal (somatic) and the apical arbor of its word-form slot, and (new in f1i) the hand-wired SOM-like pool: every mention cell -> every SOM cell, every SOM cell -> every mention cell's apical compartment (the target compartment is fixed by wiring, as Martinotti axon targeting is fixed by cell type)
allowed-by: CLAUDE.md non-negotiable 1 (a declared teaching scaffold for its neural replacement); RULING f1i-0-1028 (next method f1j = A-NMDA)
replaced-by: write-time Hebbian binding of heard word forms to the engram being written, on the composer's spiking store write path, with heard word forms arriving through the hearing path and its own laminar targeting
cap: 2 builds
cap-ruling: f1i-0-1028

`decides:` and `replaced-by:` are f1i's text verbatim. f1j adds no host decision and no new selectivity: the NMDA
conductance is engine state that rises from each slot's own sensory input (the world's write, unchanged since f1h) and
decays on its own time constant. No wiring is added, and no host code compares slots, ages or rates. So `decides:` is
unchanged and CLASS HD counts f1j in the f1h / f1i scaffold group. RULING f1i-0-1028 counts that group's cap (2 builds) as
spent by f1h and f1i, and grants `cap-raise: cs-race-f1j-PACKET.md +1 builds`, only for a method that adds no new
host-decided selectivity. This packet is that method and is cited above as `cap-ruling: f1i-0-1028`.

Inputs: RULING f1i-0-1028 (runbook section 6, origin/main b889153dc); the f1i step-0 score
`research/coordination/cs-race-f1i-0-SCORE.md` (rev 2, scorer `research/runners/cs_race_f1i_score.py` at f9fe0591b, merged
f1a7514f6) and its 155 artifacts in `research/findings/raw/_cs_race/f1i/step0/` (pin ee349f8b1); the f1i packet, whose
structure, base cell, SOM path and guards this packet keeps. Claude's research memo for the gate that fired (record, companion
process, external sources) and its zero-cost open-loop replay over the landed f1i records are summarized below. One
default-OFF `sim/` edit is REQUIRED (section E). It is specified here and NOT built by this packet. Probe-local runner flags,
CPU, numpy, pool, memcap. Gate seeds are not run. Hermes runs STEP 0 only after the E build and the runner flags are merged,
then posts the RESULT and STOPs.

## Why f1i stopped (read from the f1i-0 SCORE)
- **No cell selected.** STOP-CAL, STOP-ATTRIB and STOP-SMOKE (global reading) fired; step 1 did not run.
- **What held (banked METHOD verdict).** Wherever the older plateau latched (82 defined pairs over the grid), the shared SOM
  pool on apical GABA_A ended it within 8-137 ms of Y_on (0 defined C7(a) fails). At the corner (W_SA 0.125, W_MS 215) the
  older plateau ended on 23/23 defined pairs with the path and 0/23 lesioned. That is the first spiking route on this lane that
  ends an older word's held plateau (f1h's somatic pool: 0/216).
- **Defect 2, the one f1j targets: the pulsed word's own SOM IPSP ends its own plateau after its drive stops.** C6 FAIL 51/72.
  45 plateaus fell at 153-199 ms, within 50 ms of the 150 ms box-car's end (the EARLY self-terminations). 6 fell at 308-620 ms,
  each on a seed with 4-8 SOM spikes after 200 ms (the LATE fails, carried as K1 below). The corner's one UNDEFINED pair is the
  same event: at seed 50, frog as X fell at 178 ms, before Y arrived. Under the lesion every one of those plateaus held (lesion
  C6 8/8 at every cell). Per f1i cell (seeds 41, 45-51):

  | f1i cell | early fails | late fails | C6 pass |
  |---|---|---|---|
  | sa0p125_ms215 | 0 | 0 | 8 |
  | sa0p125_ms304 | 3 | 1 | 4 |
  | sa0p125_ms429 | 4 | 3 | 1 |
  | sa0p25_ms215 | 4 | 0 | 4 |
  | sa0p25_ms304 | 5 | 0 | 3 |
  | sa0p25_ms429 | 6 | 2 | 0 |
  | sa0p5_ms215 | 8 | 0 | 0 |
  | sa0p5_ms304 | 7 | 0 | 1 |
  | sa0p5_ms429 | 8 | 0 | 0 |
  | total | 45 | 6 | 21 |

  "Early" = the c1 pulsed slot's `v_apical` first drops below `v_hold` at a ms in [150, 200); "late" = at 200 ms or after.
  The 6 late records are s46 at sa0p125_ms304, s46/s49/s51 at sa0p125_ms429, and s49/s51 at sa0p25_ms429.
- **Defect 1, the C7(c) text.** f1i compared the `g_ap` PEAK with the termination ms. Every routed SOM spike adds a constant
  `W_SA` (the `sap` path is `stp_disabled`), so the peak is the last spike of a volley that is still summing; 12 of 23 corner
  pairs failed on that text while the first `g_ap` rise preceded the termination on 23/23. The ruling did not amend f1i post
  data. f1j states the clause before any run (item C7(c) below).
- **The research gate fired** (a second lever against one defect). Corpus check: `bash tools/before_you_build.sh "the pulsed
  word's own SOM IPSP ends its apical plateau after its drive stops"` (corpus current at 34a4e5f75; logged). No prior finding
  tests SOM-IPSP termination of a plateau. Nearest record:
  `research/findings/2026-08-20-d5-dendritic-latch-self-terminates-into-discrete-apical-plateau-BTSP-window-6seed-GO.md`, which
  ended a KIR-latched plateau with a RUNNER-side SK/AHP adaptation current on `cp_v_apical` (a host write; banked method,
  not a route). Also relevant: `2026-07-25` CRITICAL apical R 333x miscalibration (the plateau amplitude scale is a known
  trap) and `2026-07-02` EMERGE-10 (the plateau kernel must sense a real apical voltage compartment).

## Why this mechanism: the companion process the engine replaced with a constant
- **The record (open-loop replay of the f1i records; section R).** After the drive's edge only the regen term holds the
  plateau (net margin 4.46 mV-units). A steady `g_ap` of 0.098 g_L removes the up state (static critical conductance 0.0975
  for the BDSP regen). The failing records see 0.32-2.27 g_L (the f1i score gives 0.32-2.64 at the max in [150, 250) over all
  51 C6 fails). Deleting the SOM spikes in (125, 200] rescues 45/45.
- **No magnitude-only lever can work.** The SOM pool is dense and uniform, so every mention apical receives the same `g_ap` at
  the same ms. The newer word's own `g_ap` (0.32-2.27 g_L) overlaps the eraser's on the older word (0.30-2.18). What separates
  them is the time since each slot's own input ended: 3-49 ms for the self-recruited IPSP, 158-287 ms for the eraser. A fix
  must be tied in time to each slot's own input, or built into wiring (alternative 3, which adds host selectivity).
- **The two constants.** (a) The runner box-car `_set_apical` (`research/runners/cs_race_f1.py:415-428`) writes 120 for the
  word's 150 ms and 0 from the next ms. That edge is the world: a heard word stops. It stays. (b) The BDSP regen
  (`sim/bridge.py:11185-11193`) has no conductance memory. Commit fee2b0f1b calls it "== fused_coincidence_plateau's form", but
  it drops that kernel's 80 ms reservoir and its Mg block. In a real tuft the synapses that carried the word leave an NMDA
  conductance that decays over tens of ms after the input stops. The engine runs a constant (no memory) where biology runs a
  process. That missing companion process is the method.
- **Why it spares the newer word and not the older one, with no host comparison.** Each slot's NMDA conductance rises only
  from its own input. During the newer word's own IPSP (3-49 ms after its input ended) its conductance is still large and holds
  the plateau. When the eraser reaches the older word (158-287 ms after its input ended) the older word's conductance has
  decayed, and the plateau falls as in f1i. Which word survives is set by the world's input timing and the dynamics.
- **Biology.**
  - The plateau's ionic basis is voltage-gated Ca2+ channels plus NMDA receptors (Kandel 6e Fig. 13-17C, bound in
    `research/biology/dendritic-plateau-coincidence-burst.md`). Antic et al. 2010 (cited by the memo; access level not recorded,
    read before binding): the dendritic plateau is an NMDA spike.
  - Doron et al. 2017 (abstract): inhibition early in an NMDA spike dampens it; inhibition late in it ends it. That is the f1j
    asymmetry: the self-recruited IPSP lands early in the newer word's NMDA event, the eraser lands late in the older one's.
  - tauN (decay). Adult rat L5-L5 paired recordings, Wang et al. 2008 (full text): NMDA-EPSC tau 40.0 +/- 2.1 ms (V1) and
    79.2 +/- 6.8 ms (mPFC), at 35-36 C; ifenprodil (NR2B block) cut them to 32.3 and 38.9 ms. L5 tuft model conventions: 75 ms
    (Rhodes 2006 tau2 75.2; Doron 2017), 70 ms (Farinella 2014, which also tested 35 ms and got shorter spikes), 50 ms
    (Poleg-Polsky 2015). Recombinant receptors near 23 C (Erreger 2005, full text): GluN2A weighted about 42 ms, GluN2B about
    249 ms. Native L5 nucleated patches, P14-28 rats near 23 C (Kampa 2004, full text): weighted about 103 ms. Wang et al. 2004
    PNAS (full text) uses 100 ms in a model. Lester 1990 (abstract): several hundred ms in cultured hippocampal neurons, an upper
    bound. **Rule used here: 40-80 ms is the adult-cortex range; a tauN of 100 ms or more is a juvenile, room-temperature or
    GluN2B-rich choice and is labelled as such.**
  - Mg block. Jahr & Stevens 1990 form, B(V) = 1 / (1 + ([Mg]o / 3.57 mM) exp(-0.062 V)), V in mV, [Mg]o = 1 mM. J&S itself
    was read at abstract level only. The constants were checked through full texts that cite it: Rhodes 2006 prints
    exp(-0.080 V), and Doron 2017 says it shifted gamma from 0.062 to 0.08. **f1j uses the J&S slope 0.062/mV; the 0.08
    steepening is a modellers' choice and is not used.** Caveat (Kampa 2004, full text): unblock is not instantaneous (a
    dominant ~100 us component plus ~4 ms and ~300 ms components), so the instantaneous block is an approximation.
  - NMDA:AMPA. The adult per-synapse peak conductance ratio is about 1:1 to 2:1 (Wang 2008, derived about 1.7; models: Doron
    2017 1:1, Farinella 2014 2:1). Rhodes 2006 argues NMDA spikes in rats up to P35 need about 5:1 or more (silent or
    extrasynaptic receptors). Myme 2003 (abstract only; L2/3, 0 Mg mEPSCs): 27-28%.
- **E binds** `research/biology/apical-nmda-conductance.md` from the full-text anchors above, with the abstract-only ones
  marked as such.

## R. Replay pre-check (open-loop, zero cost): the method stands, its full-rescue point moves up
- **Setup.** CPU numpy over the 155 landed f1i step-0 artifacts; no `sim/` edit, no commit. It is Claude's scratch on the
  workstation, not in the repository: its numbers are predictions for this packet, never results. Checks: the vectorized model
  equals the reference replay exactly; it matches the recorded `v_apical` within 0.0035 mV on 144 records; 0 C6 status
  mismatches over 72 records and 0 C7 mismatches over 216 pairs against the f1i scorer; `g_ap` rebuilt from the recorded SOM
  increments reproduces the recorded trace to 2e-6. The IPSC-12 rows reuse those increments with a 12 ms decay on the apical
  only. `g_ap` is held at the recorded values: the SOM pool does not respond to the changed plateaus (open loop).
- **Model replayed** = the E spec below: saturating gating from the slot's own input (per-step rise 0.5 at full input),
  normalized so the conductance at the input's end equals gN0 (g_L units); decay tauN; Mg block 1 mM, slope 0.062; reversal
  0 mV; bounded exponential-Euler step. The BDSP regen is unchanged.
- **At the step-0 grid** (W_SA x W_MS over the whole f1i grid; rescue = early fails that now pass C6; kept = f1i C6 passes still
  passing; corner = sa0p125_ms215 C7(a) over 24 pairs, all defined with NMDA; primary = W_SA 0.125 at W_MS 215 / 304):

  | IPSC | tauN | gN0 | rescue /45 | late /6 | kept /21 | C6 /72 | corner (a) /24 | grid (a) /defined | grid (b) /216 | primary C6 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 10 | NO-NMDA | - | 0 | 0 | 21 | 21 | 23 (23 defined) | 82/82 | 62 | 8/8, 4/8 |
  | 10 | 40 | 4 | 36 | 1 | 21 | 58 | 24 | 189/189 | 170 | 8/8, 8/8 |
  | 10 | 40 | 8 | 42 | 2 | 21 | 65 | 24 | 212/212 | 197 | 8/8, 8/8 |
  | 10 | 40 | 16 | 45 | 2 | 21 | 68 | 24 | 216/216 | 208 | 8/8, 8/8 |
  | 10 | 50 | 4 | 38 | 2 | 21 | 61 | 24 | 192/192 | 172 | 8/8, 8/8 |
  | 10 | 50 | 8 | 43 | 2 | 21 | 66 | 24 | 213/213 | 203 | 8/8, 8/8 |
  | 10 | 50 | 16 | 45 | 3 | 21 | 69 | 24 | 216/216 | 212 | 8/8, 8/8 |
  | 10 | 75 | 4 | 39 | 3 | 21 | 63 | 24 | 195/195 | 176 | 8/8, 8/8 |
  | 10 | 75 | 8 | 44 | 5 | 21 | 70 | 24 | 214/214 | 212 | 8/8, 8/8 |
  | 10 | 75 | 16 | 45 | 5 | 21 | 71 | 20 | 212/216 | 215 | 8/8, 8/8 |
  | 12 | NO-NMDA | - | 0 | 0 | 13 | 13 | 18 (18 defined) | 51/51 | 44 | 7/8, 3/8 |
  | 12 | 40 | 4 | 27 | 0 | 21 | 48 | 24 | 169/169 | 142 | 8/8, 7/8 |
  | 12 | 40 | 8 | 41 | 1 | 21 | 63 | 24 | 206/206 | 186 | 8/8, 8/8 |
  | 12 | 40 | 16 | 45 | 2 | 21 | 68 | 24 | 216/216 | 206 | 8/8, 8/8 |
  | 12 | 50 | 4 | 27 | 1 | 21 | 49 | 24 | 173/173 | 144 | 8/8, 8/8 |
  | 12 | 50 | 8 | 42 | 2 | 21 | 65 | 24 | 210/210 | 191 | 8/8, 8/8 |
  | 12 | 50 | 16 | 45 | 2 | 21 | 68 | 24 | 216/216 | 208 | 8/8, 8/8 |
  | 12 | 75 | 4 | 29 | 2 | 21 | 52 | 24 | 177/177 | 156 | 8/8, 8/8 |
  | 12 | 75 | 8 | 42 | 3 | 21 | 66 | 24 | 211/211 | 202 | 8/8, 8/8 |
  | 12 | 75 | 16 | 45 | 4 | 21 | 70 | 23 | 215/216 | 215 | 8/8, 8/8 |

  At the IPSC-12 rows, "rescue" and "late" still count the f1i (10 ms) early and late records, and "kept" counts the 21 f1i
  passes. No NMDA setting breaks a baseline pass anywhere in the map. At W_SA 0.125, W_MS 215 and 304, C6 is 8/8 at every
  grid point at the 10 ms IPSC (one 12 ms exception: tauN 40 / gN0 4 at W_MS 304, 7/8); there the eraser alone decides.
- **The replay window** (finer map, tauN 40-125 x gN0 2-16). Requiring 45/45 rescue AND corner 24/24 AND every defined grid
  pair passing C7(a), at BOTH IPSC decays, gives tauN 40-60 ms with gN0 12-16 g_L (centre tauN 50): C6 68-71/72, grid (b)
  205-214/216, and every C6 fail left is a W_MS 429 late fail (K1). At gN0 8 the rescue stops at 42-44/45 (41-42 at the
  12 ms IPSC), and lower below 8.
- **The eraser boundary.** The eraser stays intact while the older word's residual conductance at the newer word's offset,
  gN0 exp(-301/tauN), is at most 0.14 g_L. It erodes from about 0.18-0.21 g_L at the 10 ms IPSC and about 0.29 at 12 ms. At
  tauN 50 / gN0 16 the corner older word still ends at a median 374 ms (baseline 372).
- **Sensitivity.** Linear-integrating kinetics instead of saturating give identical metrics at every grid point (`v_apical`
  differs only inside the volley, up to 31 mV, and is identical from 200 ms on). Explicit instead of bounded integration
  changes counts by at most 2.
- **Against the memo's predictions.** The memo predicted 0/45 early self-terminations left at gN0 8, tauN 50-75; the replay
  leaves 1-2 there (43-44/45), and full rescue needs gN0 10 or more. Its C6 67-68/72 is 66-70 at gN0 8 and 69-71 at gN0 16. Its
  corner 23/23 -> 24/24 (the seed-50 pair defined and passing) and grid C7(a) on every defined pair hold everywhere except
  tauN 75 / gN0 16 (corner 20/24, grid 212/216, its stated risk). The replay does not refute A-NMDA; it moves the full-rescue
  point up and confirms the eraser boundary. **METHOD = A-NMDA, the memo's recommended choice.**
- **Alternative 2 (A-reservoir) is refuted by the same replay.** Replacing the BDSP regen by the `fused_coincidence_plateau`
  sustain (80 ms reservoir, Mg-gated sustain, Mg-blocked current to 0 mV; scale A = R x self_regen, A_m = 0.126 puts a fixed
  point at the BDSP plateau, -24.15 mV): below 0.92 A_m there is no plateau. At 0.96 A_m the plateau sits at -16.6 mV and the
  corner eraser is 3/24; at 1.0 A_m -24.15 mV becomes the unstable middle point, the plateau moves to -14.9 mV and the corner is
  0/24; the engine's two-compartment value (0.30 = 2.38 A_m) puts the plateau at -4.8 mV with a static critical `g_ap` of 1.31
  (BDSP: 0.0975) and gives C6 72/72 with grid C7(a) 0/216. A plateau near 0 mV is beyond the eraser's reach. It is also only
  reachable through `enable_two_compartment_dap`, which the f1i E2 guard (`sim/bridge.py:4864-4867`) forbids with
  `gaba_a_apical`.
- **What the replay cannot see (the closed loop).** The f1i LESION records certify that a single held BDSP plateau recruits 0
  SOM spikes in c1 [200, 800) at W_MS 215 and 304 on all 8 seeds (67 at W_MS 429, at most 18 in one record). NMDA holds the
  post-edge apical higher (median `v_apical` at 199 ms about -10 mV at tauN 50 / gN0 16, against the -24 mV regen plateau), so
  the coupled soma fires more: excess coupling over the LESION plateau up to 31 pA (gN0 16) and 24 pA (gN0 8); the estimated
  mention rate rises x1.43-1.45 in [150, 250) and x1.08 in [200, 800), as if W_MS 215 were about 308-312 in [150, 250) and about
  232 over [200, 800) (W_MS 304: about 434-441 and 325-328). Scaling the replayed SOM tail [100, 300) by x1.5, tauN 50 / gN0 16
  still rescues 45/45 (x2: 42/45); tauN 50 / gN0 8 drops to 39/45. **So the gN0 8 cells sit near F1's floor once the loop
  closes. This is why step 0 runs closed loop at every grid cell and no cell is assumed.**
- **Disclosure, the drive is a current source.** f1h's apical drive is a leak-normalized current, not a conductance, so during
  the volley the recorded `v_apical` reaches up to +17 mV, above any synaptic reversal (median +13 mV at 100 ms). The NMDA
  term (reversal 0 mV) pulls it toward 0 mV inside the volley. The sensitivity row above shows that the in-volley difference
  does not move any metric; the post-edge tail does. The drive is the world's write and is not changed by f1j; a
  conductance-based sensory drive is noted for the board as its own faithfulness item.

## Ranked alternatives (one chosen; the rest banked) and exclusions
1. **CHOSEN: A-NMDA.** An input-driven, decaying, Mg-blocked NMDA conductance on the BDSP apical (section E).
2. A-reservoir (the engine's `fused_coincidence_plateau` sustain in place of the BDSP regen). Refuted by the replay above
   (no plateau below 0.92 A_m; at or above 0.96 A_m the plateau sits too high for the eraser). Banked.
3. Per-slot SOM sparing its own slot (slot i -> SOM_i -> apicals of slots j != i; Silberberg & Markram 2007, abstract only).
   Fino & Yuste 2011 report local SOM -> PC wiring as dense and unspecific. It adds host-decided selectivity by wiring and
   works by construction, so step 0 would barely test the defect. RULING f1i-0-1028: a per-slot wiring packet first names its
   learned `replaced-by`. Banked.
- **Excluded** (each measured or ruled):
  - a scalar `coincidence_plateau_self_regen` change: grid C7(a) 92/138 -> 2/138 in the replay (it holds the older word too);
  - `apical_kir_g` 0: 11/45;
  - VIP silencing of the shared pool: corner 0/23;
  - a 5 ms IPSC: unfaithful for a dendrite-targeting IPSC;
  - GABA_B: a slower inhibitory component lengthens the self-recruited IPSP that causes the defect, and the engine's
    `I_gabab` reads the somatic voltage (f1i alternative 4);
  - retuning the f1i grid below the corner: the retune f1i's branch rule forbids;
  - a runner-written decaying tail on the drive: host-shaped (the host would write the conductance's time course).

## E. The required `sim/` edit (default OFF; specified here, built separately by Claude, merged before step 0)
- **E1 config** (`sim/config.py`, beside the apical GABA fields, ~836):
  - `enable_apical_nmda: bool = False`
  - `apical_nmda_g0: float = 0.0`, apical-leak units: the conductance the gating reaches under sustained full input, i.e. at
    the end of a heard word. With the flag on, 0.0 is the APV arm (receptor blocked, input intact).
  - `apical_nmda_tau_ms: float = 50.0`, the decay (adult L5 range 40-80 ms).
  - `apical_nmda_rise_per_ms: float = 0.6931471805599453` (ln 2): the bounded rise rate at full input. At dt 1 ms the
    per-step rise fraction is exactly 0.5, the replay's value. The replay shows the rise does not move any metric.
  - `apical_nmda_drive_ref: float = 0.0`: the apical drive (leak-normalized mV) that counts as full input; it must be > 0
    when the flag is on (raises otherwise). The f1j runner passes its `--apical-pulse` value (120), so the input factor is 1
    while a word is heard and 0 otherwise, exactly the replay's input.
  - `apical_nmda_mg_mM: float = 1.0`, `apical_nmda_mg_k_mM: float = 3.57`, `apical_nmda_mg_slope_per_mV: float = 0.062`
    (Jahr & Stevens; not 0.08).
  - The reversal reuses `syn_reversal_potential_e` (0 mV, the engine's AMPA/NMDA reversal). No new reversal.
- **E2 state and guards.**
  - `cp_apical_nmda_s = zeros(n, float32)` (the gating), allocated only when the flag is on, grown with capacity like the
    apical GABA conductance; None otherwise.
  - FAIL LOUDLY (ValueError at config or wiring time): the flag on with `enable_bdsp` off; with `enable_two_compartment_dap` on
    (a second writer of `cp_v_apical`, as the f1i E2 guard); with `enable_bdsp_microcircuit` on (the drive it reads would be the
    error-subtracted drive, undefined here); `apical_nmda_g0 < 0`, `apical_nmda_tau_ms <= 0`, `apical_nmda_drive_ref <= 0`,
    `apical_nmda_mg_mM < 0`.
  - Step-time (as the f1i routing check): the flag on with `cp_bdsp_apical_drive` None raises (the gating input and the term
    would be dropped silently); the flag or `enable_bdsp` turned off after allocation raises.
  - `_step_megakernel_can_dispatch` (`sim/bridge.py:8608`) returns False when the flag is on.
- **E3 gating** (BDSP apical block, before the update line, from this step's drive, per cell i):
  - `u_i = max(drive_i, 0) / apical_nmda_drive_ref` (no clip; the bounded rise below is safe at any u);
  - `s_i <- s_i * exp(-dt/tauN)`, then `s_i <- s_i + (1 - s_i) * (1 - exp(-rise * u_i * dt))`;
  - `g_N,i = apical_nmda_g0 * s_i / s_full`, with `s_full = r / (1 - dN + dN r)`, r = 1 - exp(-rise dt), dN = exp(-dt/tauN):
    the discrete steady state at u = 1, a constant fixed by the config, so g_N at the end of a sustained input is g0.
  - The gating reads only the runner's existing write, `cp_bdsp_apical_drive` (the world's input, f1h's laminar targeting). A
    cell with no drive keeps s at exactly 0.0, so g_N is exactly 0.0: inert at rest.
- **E4 term** (computed from v0, before the update, like every other term; applied after the update line, beside the apical
  GABA step):
  - `B = 1 / (1 + (mg / mg_k) * exp(-slope * v0))`;
  - `nmda_step = (E_e - v0) * (1 - exp(clip(-(g_N * B) * dt / tau_ap, -80, 0)))`, then `v_apical += nmda_step`.
  - It reaches E_e at most and never passes it, at any g_N (the 2026-09-29 overshoot family, as the f1i E5 step). At g_N == 0
    the step is exactly 0.0. Per-term bound only, as the engine documents and as the replay integrated it: an aggregate
    overshoot with the explicit terms, the KIR term or the apical GABA step is not excluded. (One joint bounded step toward
    the conductance-weighted reversal would bound the pair exactly; it is not used, because it is not what the replay ran.
    The replay's integration sensitivity, explicit against bounded, is at most 2 counts.)
- **E5 tests** (`tests/test_apical_nmda.py`):
  - flag off: byte-identical, asserted by a state hash after N steps (the `test_byte_identical_when_off` pattern);
  - flag on, g0 > 0, zero drive everywhere: byte-identical to flag off and s == 0.0 exactly (inert at rest);
  - flag on, g0 = 0 (APV) with a drive: `v_apical` and every other state array byte-identical to flag off (s may rise);
  - g_N > 0 only on cells that received drive; every other cell exactly 0.0;
  - after the drive ends, g_N decays by exactly exp(-dt/tauN) per step (float32, relative 1e-6);
  - under sustained full drive g_N converges to g0 (relative 1e-5);
  - B(-80), B(-24), B(0) against the formula with slope 0.062;
  - one step never crosses E_e for g_N B in {0.1, 10, 1e4} (v == E_e exactly at 1e4 from below);
  - each FAIL-LOUDLY case of E2 raises;
  - checkpoint save / restore round-trips `cp_apical_nmda_s`;
  - with the f1i apical GABA routing on: the GABA and NMDA steps are both computed from v0 and each equals its formula (the
    order is pinned; no aggregate bound is claimed);
  - a replay match: a single slot driven 120 for 150 ms then 0, no SOM, at (g0, tauN) = (8, 50), against the replay's
    reference model (ported into the test, about 20 lines of numpy): max |dv| <= 0.01 mV over 800 ms.
- **E6 bookkeeping.** Register `cp_apical_nmda_s` beside `cp_conductance_g_apical_gaba` in the flag-gated array map
  (`sim/bridge.py:408`) and in the state save / restore lists that carry it.

## Method (runner flags on `research/runners/cs_race_f1.py`; every default reproduces the f1i runner exactly)
- **Base cell, unchanged from f1i.** The f1h circuit at soma-g 40, W_IM 40 (RULING f1h-0-1027) with the f1h step-0 flags, plus
  the f1i SOM path (NSOM 8, facilitating slot -> SOM on the MWT98 triple U 0.049 / D 399 ms / F 1797 ms, SOM -> apical GABA_A
  through `receptor="gaba_a_apical"`), at W_SA 0.125.
- **New flags** (all default off; with none of them the record is the f1i runner's, pinned by identity (i)):
  - `--nmda-g0 G` (needs `--som`): `enable_apical_nmda = True`, `apical_nmda_g0 = G`, `apical_nmda_drive_ref =` the
    `--apical-pulse` value. `--nmda-g0 0` is the APV arm.
  - `--nmda-tau T` (needs `--nmda-g0`): `apical_nmda_tau_ms = T`.
  - `--tau-gi T`: `syn_tau_g_i = T`, the GABA_A decay. `--tau-gi 10` must hash-equal no flag. It changes every GABA_A decay,
    the f1g pool's somatic IPSC as well as the apical one; the replay's IPSC-12 rows changed the apical only. The closed-loop arm
    is the test; the difference is disclosed, not corrected.
- **The runner writes no NMDA quantity.** Its only apical write stays `_set_apical`, unchanged: the heard word's drive while
  it is heard. Its onset and its end are the world's timing. No tail, no offset shaping, no `g_N` write. The one exception is
  the trial-boundary reset below, of the same class as `_reset_quiescent`.
- **Trial reset (S9).** The quiescent snapshot does not carry `cp_v_apical`, STP state, `g_ap` or the new gating.
  `_reset_apical` (`cs_race_f1.py:430`) also writes `cp_apical_nmda_s = 0` before the snapshot and at every trial start, and the
  value is read back into the trial record.
- **Records.** Per trial, beside the f1i raw (per-ms `v_apical` min / max and `g_ap` for the pulsed words, per-ms SOM counts,
  per-spike `som_cell_events`, gate read-backs): per-ms `g_N` max for each pulsed word; the per-ms max of `g_N` over all mention
  cells that received no drive in the trial (must be 0.0); in c3 the older slot's `g_N` at Y_on + 150 (the newer word's offset;
  the eraser-boundary quantity); the realized `syn_tau_g_i` and the cached `decay_i`; the NMDA config as realized.
- **Runner tests** (`tests/test_cs_race_f1j_flags.py`): flags off hash-equal to the f1i runner at seed 41 (cal(c3), cal(c1),
  race()); `--tau-gi 10` hash-equal to no flag; `--nmda-g0 0` leaves every f1i field of the record equal to flags off; the reset
  reads back 0.0; the runner source assigns `cp_apical_nmda_s` only in the reset; each flag-dependency error raises.

## Step 0 (Hermes, after the E build and the runner flags are merged): seeds 41 and 45-51 only
Seeds 42-44 and 100-102 are project gate seeds and are not run (f1i R1). Seeds 41 and 45-51 are calibration only.
- **P0' engine support (read-only, first).** Show with file:line that the merged E build is present and its tests pass, and
  that `--nmda-g0` allocates `cp_apical_nmda_s` on the mention cells. Missing or failing: STOP-ENGINE.
- **Grid (frozen before any run): tauN {40, 50, 75} ms x gN0 {4, 8, 16} g_L, at W_SA 0.125, W_MS {215, 304}.**
  - **Literature range.** tauN: 40-80 ms adult cortex. 40 = Wang 2008 adult V1 L5-L5; 50 = Poleg-Polsky 2015 and the replay
    window's centre; 75 = Rhodes 2006 / Doron 2017 tuft value, near Wang 2008 mPFC (79.2). gN0: the drive is a current
    source, so it has no AMPA conductance to scale from. Read as an AMPA conductance with reversal 0 mV it is 1.55 g_L at rest
    (120 / 77.3) and 5.0 g_L at the -24 mV plateau; the adult per-synapse NMDA:AMPA of 1:1 to 2:1 then maps to gN0 of about
    1.6-10 g_L. gN0 4 and 8 overlap that range under one reading or the other; gN0 16 is above it under both (3.2:1 at the
    plateau reading, 10:1 at rest) and inside only Rhodes 2006's argued >= 5:1 (juvenile, silent or extrasynaptic NMDA). gN0
    16 is labelled as such wherever it is reported.
  - **Replay window.** tauN 40-60 with gN0 12-16 (45/45 rescue, corner 24/24, every defined grid pair, both IPSC decays). The
    grid brackets it: gN0 16 at tauN 40 and 50 sits inside it; gN0 4 and 8 below it (rescue 36-44 in the replay); tauN 75 / gN0
    16 above it (the eraser-erosion risk, corner 20/24).
  - **Departure from the memo's grid** (tauN {50, 75, 100} x gN0 {4, 8, 16}), stated before any run: tauN 100 is replaced by
    40. A tauN of 100 ms is a juvenile, room-temperature or GluN2B-rich value by the literature rule, and the replay puts the
    corner eraser at 16/24 (gN0 8) and 0/24 (gN0 16) there. 40 ms is the adult V1 measurement and makes the memo's falsifier
    F2 ("works only at tauN < 50 ms") testable; the memo's grid had no tauN below 50. gN0 is the memo's set.
  - The SOM cell: W_SA 0.125 (the f1i corner's; sa0p125_ms215 was the only f1i cell passing C3 on 8/8). W_MS {215, 304}: the primary range of
    the carried K1 defect (W_MS <= 304). W_MS 429 runs only in the GRID arm, never selectable.
- **Arms** (names are the output stems; counts in the lines spec at the end):
  - **TRACE** (selectable): the 9 NMDA cells x W_MS {215, 304} x 8 seeds.
  - **LESION**: the same with `--som-lesion` (the `cs_som` gate closed the whole run, SOM cells still driven and firing). It
    scores C7(d) and identity (ii) for the TRACE line of the same cell and seed.
  - **APV (NO-NMDA)**: `--nmda-g0 0 --nmda-tau 50` at all 9 f1i cells (W_SA {0.125, 0.25, 0.5} x W_MS {215, 304, 429}) x 8
    seeds. It is identity (iii), and the same-pin baseline for every rescue count. Its lesion condition is read back: `g_N`
    == 0.0 at every recorded ms on every cell while the gating still rises on driven cells (S11).
  - **GRID** (never selectable): the 7 other f1i cells (W_SA 0.25 and 0.5 at every W_MS; W_SA 0.125 at W_MS 429) x the 9 NMDA
    cells x 8 seeds. With TRACE it re-runs the whole f1i grid closed loop at every NMDA cell, so the 45 early records, the 6
    late records and the 21 f1i passes are all measured. It reads F1 and the prediction table.
  - **IPSC-12**: TRACE at `--tau-gi 12` (9 NMDA cells x W_MS {215, 304} x 8 seeds), plus APV at `--tau-gi 12` at W_SA 0.125 x
    W_MS {215, 304} x 8 seeds as its baseline. It reads F3.
  - **K1-NOFAC** (informational, the carried item's falsifier): `--som-nofac --nmda-g0 0 --nmda-tau 50` at W_SA {0.125,
    0.25} x W_MS 429 x 8 seeds, the two cells that hold 5 of the 6 late fails.
  - **Identity (i)**: no f1j flag at all, corner cell (W_SA 0.125, W_MS 215), seeds 41 and 45.
- **Identity (STOP-ID).**
  - (i) The no-flag records hash equal, minus `wall_s`, to the landed f1i `cal_sa0p125_ms215_s41.json` and `_s45.json`.
  - (ii) Every LESION record reads `g_ap` == 0 at every recorded ms and the `cs_som` gate 0.0 at measurement time.
  - (iii) Every APV record equals the landed f1i record of the same cell and seed in every f1i field (each item and every raw
    series; minus `wall_s` and the echo of its own flags), and its f1j fields read `g_N` 0.0 at every ms.
  - (iv) The read-backs among the guards below (S1, S2, S3, S7, S9-S13) hold on every `--som` record. Any failure is STOP-ID.
- **Items, on 8/8 seeds, at each TRACE cell.**
  - C0-C6 as in f1h and f1i. C3 (overwrite, both orders, the three pairs) is REQUIRED. C6: the pulsed slot's `v_apical` stays
    above `v_hold` across the 650 ms window, and every unpulsed slot's stays below it.
  - C7 (the eraser), per C3 pair, X at 0-150 ms, gap, Y_on = 300 ms, Y at 300-450 ms, window [600, 950):
    - (0) precondition (S5): the older slot's min `v_apical` > `v_hold` over [150, 300). Otherwise the pair is UNDEFINED.
    - (a) termination latency: the first ms after Y_on from which the older slot's max `v_apical` stays below `v_hold` through
      the window. Bar <= 300 ms. Reported per pair.
    - (b) the newer plateau survives its own inhibition: the newer slot's min `v_apical` > `v_hold` over [Y_on + 150, Y_on +
      650).
    - **(c) engagement, REWRITTEN before any run (RULING f1i-0-1028):** SOM pool spikes in [Y_on, Y_on + 300) >= 1 per SOM
      cell, AND the older slot's FIRST `g_ap` increase after Y_on (the first ms t >= Y_on with `g_ap`(t) > `g_ap`(t - 1), a
      routed SOM spike arriving; the f1i scorer's `g_ap_onset_ms`) precedes its termination ms. The `g_ap` peak over [300, 950)
      and its ms are REPORTED, never barred. A pair whose older plateau never ends has no termination ms; (c) then reads the
      spikes and `g_ap` > 0 (the f1i scorer's reading).
    - (d) in-cell lesion: the same pair in the LESION record leaves the older plateau above `v_hold` in the window, AND rate-C3
      (newer 2-20 Hz, older < 1 Hz) fails on that pair (S6).
  - C8 (selectivity): in the C1 protocol (one word, then 650 ms) the SOM pool rate over [pulse end + 50, pulse end + 650) is
    < 1 Hz per cell (S8).
  - An UNDEFINED pair or item is not a pass: 8/8 means eight defined passes.
- **Per-NMDA-cell reads** (from TRACE + GRID at the 10 ms IPSC, against APV at the same pin, cell and seed):
  - **rescue** /45: of the 45 f1i early records (the scorer re-derives the list from the landed f1i artifacts and must find
    45, else STOP-ID), those that FAIL C6 in APV and PASS C6 in TRACE / GRID. A record that also passes in APV is not a rescue.
  - **late** /6 (the same, for the 6 late records), **kept** /21 (f1i C6 passes still passing), C6 /72, grid C7(a) over
    defined pairs and grid C7(b) /216.
  - **eraser** /23: the 23 corner pairs defined in f1i, each a defined C7(a) pass at W_SA 0.125, W_MS 215. The seed-50 frog>vole pair is
    reported apart (predicted: defined and passing).
  - A **working** NMDA cell: rescue >= 40/45 AND eraser 23/23 (F1's two quantities).
- **Selection.** Among working NMDA cells with kept 21/21, the TRACE cell passing C0-C4 and C6-C8 on 8/8: W_MS 215 before 304,
  then the smallest gN0, then the shortest tauN (the least added conductance and memory, as f1i chose the smallest W_SA, then
  W_MS). Predicted in open loop, not assumed: tauN 40 / gN0 8 at W_MS 215, whose closed-loop rescue may fall below 40 (the
  exposure note in R).
- **STOP rules.**
  - STOP-ENGINE (P0'); STOP-ID (identity (i)-(iv)).
  - STOP-CAL: no working cell passes the selection items.
  - STOP-SMOKE: the median C5 is below 2 pA in either clause, read at the selected cell (8 seeds) OR over the 144 TRACE records.
    Either reading fires it. f1i's single global reading was 0.0 pA because the failing cells' C5 is 0.0; both are now
    reported and either is enough.
  - STOP-ATTRIB: a cell whose C3 passes while C7(c) or C7(d) fails is UNATTRIBUTED, never a pass. If every C3-passing cell is
    unattributed, STOP-ATTRIB.
  - STOP-FALS: F1, F2 or F3 fired.
- **Falsifiers** (the memo's words, each with its reading fixed here before any run; each fires on the step-0 data alone):
  - **F1: "closed-loop rescue < 40/45 at every (gN0,tauN) where the eraser keeps >= 23/23".** Fires when every NMDA cell
    whose eraser is 23/23 has rescue < 40/45, including the case where no NMDA cell keeps the eraser at 23/23 (no cell can
    rescue without breaking the eraser).
  - **F2: "works only at tauN < 50 ms (then the protocol's 150 ms word/gap is the next suspect, recorded as a finding not
    retuned)".** Fires when there is at least one working cell and every working cell has tauN 40.
  - **F3: "collapses at IPSC decay 12 ms".** Fires when there is at least one working cell and none of them, run at
    `--tau-gi 12` at W_MS 215, passes C6 on 8/8 and C7(a) on every defined corner pair.
- **Decision tree** (each branch posts and STOPs; the next ruling takes the named branch, never a retune of this grid):
  1. STOP-ENGINE or STOP-ID: instrument. Nothing else is read.
  2. F1 fires: the A-NMDA METHOD verdict is banked as not closing the self-termination defect once the loop closes. Next: the
     research gate on the closed-loop residual (what amplifies the self-recruited SOM tail), never a gN0 / tauN retune.
     Alternative 3 needs its learned `replaced-by` named before any packet (RULING f1i-0-1028).
  3. F2 fires (only the 40 ms edge of the adult range works): the finding is recorded; the protocol's 150 ms word and 150 ms
     gap are the next suspect, recorded as a finding and not retuned.
  4. F3 fires: the rescue is fragile to a 20% slower IPSC; recorded as a finding; the ruling chooses.
  5. STOP-CAL with no falsifier: post which item fails at every working cell. Pre-decided readings: C8 or a late C6 fail at
     every working cell, each on a record with SOM spikes after 200 ms, is K1's (K1 goes first); C7(b) failing at every working
     cell (the newer word dies with the older) is f1i's B1 text, and alternative 2's refutation stands, so the ruling chooses.
  6. STOP-ATTRIB: post and STOP.
  7. A cell is selected but STOP-SMOKE fires (f1i's B4): the trace holds and overwrites but does not reach the engram, an f1h
     coupling (soma-g) verdict.
  8. A cell is selected, no STOP and no falsifier: post. Step 1 runs only after a Claude RULING.
- **Post** `[cs-race-f1j-0] RESULT`: every item per cell, P0', the identity hashes, rescue / late / kept / eraser per NMDA
  cell, the chosen cell, C7 latencies with first-rise and peak ms, the older slot's `g_N` at Y_on + 150, C5 in both readings
  against the +4/+6 pA flip scale, F1-F3 and the K1 falsifier read, the closed-loop numbers beside the replay's prediction table,
  the STOP verdicts, branch @ sha. Then STOP.

## K1, carried: the slot -> SOM STP rate gate (its own item, its own falsifier; NOT fixed by f1j)
- **Defect.** On the MWT98 triple in the engine's form the slot -> SOM per-spike efficacy is 0.093 at a steady 10 Hz and
  0.135 at a steady 2 Hz, against 0.005 at the steady state of a 100 Hz volley (first spike 0.047; peak 0.111 at spike 4,
  f1i packet). The held trace's rates get a HIGHER per-spike efficacy than the volley: the rate gate is inverted at steady
  state.
- **Biology.** Berger et al. 2009 (full text): a 20 Hz train of 8 spikes recruits no SOM cell. Kapfer et al. 2007 (full text):
  SOM recruitment at spike 4.9 of 100-125 Hz trains; 5th / 1st EPSP ratio 3.8 (the engine: about 2.4 at its spike-4 peak).
- **Evidence.** The 6 late C6 fails (308-620 ms), each on a seed with 4-8 SOM spikes after 200 ms. The f1i LESION held plateau
  recruits 0 SOM spikes in c1 [200, 800) at W_MS 215 and 304 and 67 at W_MS 429. NMDA's extra post-edge coupling moves the
  held trace toward this region (W_MS 304 behaves like about 315-332 over [200, 800) in the replay's estimate), so C8 at W_MS
  304 is the watch item.
- **Hypothesis H-K1:** the late fails are the held trace recruiting its own eraser through the slot -> SOM synapse's low-rate
  efficacy. **Falsifier, read from the step-0 records:** H-K1 is falsified if (a) any late C6 fail (the c1 plateau falls at
  200 ms or after) in the APV or GRID arms has 0 SOM spikes in [200, fall ms), or (b) the K1-NOFAC arm (static slot -> SOM at the
  facilitating synapse's resting efficacy, W_MS x 0.049) has as many or more late C6 fails than APV on the same 16 (cell, seed)
  records. Prediction, not a bar: fewer under NOFAC (the held trace's per-spike efficacy roughly halves). NOFAC also changes the
  volley's recruitment, so early fails are reported beside it.
- **Scope.** f1j keeps W_MS <= 304 primary and never selects W_MS 429. The fix (the slot -> SOM synapse brought to the
  biology targets: no recruitment from 20 Hz x 8, recruitment near spike 5 at 100-125 Hz, 5th / 1st near 3.8) is its own
  packet and board item, not built here.

## Step 1 (preregistered; runs only after a Claude RULING on step 0)
- **Arms:** TRACE (the chosen cell), LESION (`--som-lesion`), NO-ENC, NOFAC (`--som-nofac`), and APV (`--nmda-g0 0` at the
  chosen cell's tauN, everything else identical). NOFAC and APV are mechanism-necessity arms (CLAUDE.md non-negotiable 4),
  not ownership bars.
- **Bars:** the f1g step-1 ownership bars, unchanged. Clauses (a) and (b) each pass at 8 or more of 10 on at least 7/8 DEV
  1-8 AND at least 14/16 CONFIRM 25-40, with the positive control, clause (c) and clause (g) intact on those seeds. LESION
  passes (a) and (b) on at most 2/8 DEV.
- **Necessity reads:** NMDA is reported NECESSARY iff APV fails the TRACE bars on the seeds where TRACE passes; facilitation
  likewise with NOFAC. Otherwise each is reported as not necessary for this capability and kept (faithfulness-first).
- **Engagement:** per draw, the mention spike counts, the plateau-latched flag, the SOM spikes in the newer word's window, the
  older slot's first `g_ap` rise and peak, and both slots' `g_N` at the newer word's offset. A draw with no latched plateau or
  no SOM engagement is UNDEFINED.

## Guards (f1i S1-S8 kept; S9-S13 new)
- S1: STP state and `g_ap` are outside the snapshot; `_reset_som` writes msom u = 0, x = 1 and `g_ap` = 0 at every trial start.
- S2: the trait-built per-synapse conn type is read back per pathway (msom = 1, ee = 0) with its realized U / F / D; a mismatch is
  STOP-ID.
- S3: `cp_stp_u` starts at the global U, so the msom reset to u = 0 is required.
- S4: the STP record is split by pathway (ee, msom).
- S5: C7(0), the older plateau latched before Y.
- S6: rate-C3 must fail under the lesion.
- S7: `--som-lesion` re-closes `cs_som` after every gate reset; read back at measurement time.
- S8: the volley's 50 ms tail is excluded from the C8 window.
- S9: `cp_apical_nmda_s` is outside the snapshot; it is zeroed before the snapshot and at every trial start and read back.
- S10: inert at rest: `g_N` on every mention cell that received no drive in a trial is exactly 0.0 at every recorded ms.
- S11: the APV lesion holds at measurement: `g_N` == 0.0 at every recorded ms on every cell, while the gating still rises on
  driven cells (the receptor is blocked, not the input).
- S12: the runner writes no NMDA quantity except the trial-boundary reset (runner test plus read-back).
- S13: the realized `syn_tau_g_i` and `decay_i` are read back; the IPSC-12 arm must read 12, every other arm 10.

## Self-skeptic record (Claude, 2026-10-28, before posting)
- **Host arithmetic?** None added. The conductance's rise, decay and Mg block are engine state; its input is the world's
  existing write. The runner writes no conductance, tail or offset. No host code compares slots, ages, identities or rates;
  which plateau survives is set by input timing and the dynamics. The SOM wiring is f1i's declared structure.
- **Selectivity added?** No. Every mention cell has the same mechanism with the same constants. A slot gets a conductance only
  from its own input, as every synapse does.
- **One-variable lesions?** APV changes `apical_nmda_g0` only (the gating still runs); LESION closes one transmission gate on
  the SOM cells' only output. Each is read back at measurement time (TERMS.md "lesion").
- **Can C6 pass without NMDA?** A rescue counts only on a record that fails in APV at the same pin, cell and seed. At W_SA
  0.125, W_MS 215 C6 already passed 8/8 in f1i, so the primary cells test the defect on only 4 records and one pair; that is
  why the GRID arm re-runs the whole f1i grid and why selection requires a working cell.
- **Is the rescue the in-volley clamp rather than the post-edge tail?** The replay's linear and saturating kinetics differ
  inside the volley by up to 31 mV and give identical metrics; only the post-edge conductance matters.
- **Values from a run?** No NMDA run of any kind exists. tauN comes from the adult-cortex literature, gN0 from the memo's set,
  the SOM cell from f1i. The replay on the f1i records (seeds 41, 45-51, the same calibration seeds) placed the grid and gives
  the predictions; this is disclosed, and no step-1 seed has been seen.
- **Departures stated before any run:** tauN 100 -> 40 (the literature rule, F2's testability); C7(c) rewritten; STOP-SMOKE
  read both ways; the IPSC arm changes every GABA_A decay, not only the apical one; ~1042 lines against the memo's ~155 (the
  GRID arm that F1 needs).
- **Silent fallbacks guarded:** the gating outside the snapshot (S9); an unpulsed slot gaining a conductance (S10); an APV arm
  that leaks (S11); a runner-shaped tail (S12); an IPSC flag that did not take (S13); a missing drive array (E2 step-time raise).

## Step-0 pool lines: generator spec (the lines are NOT written by this packet)
- **Generator:** `research/runners/cs_race_f1j_lines.py` (module run, deterministic, no clock), writing
  `research/coordination/cs-race-f1j-0-pool-lines.txt`, one line per run, `cmd<TAB>reason`, the f1i format. A test
  (`tests/test_cs_race_f1j_lines.py`) pins the counts and invariants below and parses every line with the f1i scorer's
  `parse_lines`.
- **cmd template:** `cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 1 1 && mkdir -p research/findings/raw/_cs_race/f1j/step0
  && bash tools/memcap.sh 2 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.cs_race_f1 --calibrate --seeds <seed> --grid-row 3 --w-gis 30
  --w-mi 20 --w-im 40 --nim 8 --encode-words --ment-pa 150 --trace intrinsic --soma-g 40 --apical-pulse 120 --som <W_MS> --som-ap
  <W_SA> <arm flags> --out research/findings/raw/_cs_race/f1j/step0/<stem>.json` (one command line; wrapped here). `<sha>` is a
  revision that contains the merged E build, the runner flags and this packet.
- **Tags:** W_SA 0.125 / 0.25 / 0.5 -> `sa0p125` / `sa0p25` / `sa0p5`; W_MS -> `ms215`; tauN -> `t40`; gN0 -> `g4`.
- **Arms, flags, stems and counts** (seeds 41, 45, 46, 47, 48, 49, 50, 51 in that order inside each cell):

  | wave | arm | arm flags | stem | cells | lines |
  |---|---|---|---|---|---|
  | 1 | identity (i) | none (the `--som 215 --som-ap 0.125` base only) | `identity_f1i_sa0p125_ms215_s<seed>` | seeds 41, 45 | 2 |
  | 1 | APV | `--nmda-g0 0 --nmda-tau 50` | `apv_sa<>_ms<>_s<seed>` | 9 f1i cells | 72 |
  | 1 | K1-NOFAC | `--som-nofac --nmda-g0 0 --nmda-tau 50` | `k1nofac_sa<>_ms429_s<seed>` | W_SA 0.125, 0.25 at W_MS 429 | 16 |
  | 2 | TRACE | `--nmda-g0 <G> --nmda-tau <T>` | `cal_t<T>_g<G>_sa0p125_ms<>_s<seed>` | 9 NMDA x W_MS 215, 304 | 144 |
  | 2 | LESION | TRACE + `--som-lesion` | `lesion_t<T>_g<G>_sa0p125_ms<>_s<seed>` | as TRACE | 144 |
  | 2 | GRID | `--nmda-g0 <G> --nmda-tau <T>` | `grid_t<T>_g<G>_sa<>_ms<>_s<seed>` | 9 NMDA x 7 other f1i cells | 504 |
  | 2 | IPSC-12 | `--tau-gi 12 --nmda-g0 <G> --nmda-tau <T>` | `ipsc12_t<T>_g<G>_sa0p125_ms<>_s<seed>` | 9 NMDA x W_MS 215, 304 | 144 |
  | 2 | IPSC-12 APV | `--tau-gi 12 --nmda-g0 0 --nmda-tau 50` | `ipsc12_apv_sa0p125_ms<>_s<seed>` | W_MS 215, 304 | 16 |
  |  | total |  |  |  | 1042 |

- **Order and waves.** Wave 1 (90 lines) first. When it lands, Hermes runs the f1j scorer's identity mode (Claude builds the
  scorer, `research/runners/cs_race_f1j_score.py`, extending the f1i scorer, before wave 1 lands): identity (i) and (iii),
  S9-S13 on the APV records. A pass queues wave 2 (952 lines) with no ruling needed; a fail is STOP-ID (post and STOP).
- **Reason column:** `cs-race f1j step-0 <ARM> ...` naming the arm, the cell (tauN, gN0, W_SA, W_MS), the seed and what it
  scores; then `packet research/coordination/cs-race-f1j-PACKET.md (Step 0; seeds 41,45-51, no gate seeds)`, `RULING
  f1i-0-1028 (f1j = A-NMDA; cap-raise +1)`, `base cell RULING f1h-0-1027 + the f1i SOM path`, `no NMDA run informs any f1j value
  (tauN from the adult-cortex literature, gN0 the memo's set; the open-loop replay of the f1i records is disclosed in the
  packet)`; ends `mem_gb=1 max_age_h=240`.
- **Invariants the test pins:** 1042 unique `--out` paths, all under `research/findings/raw/_cs_race/f1j/step0/`; no seed in
  {42, 43, 44, 100, 101, 102}; every TRACE line has a LESION twin differing only by `--som-lesion` and the stem; every APV line
  differs from the matching f1i step-0 line only by the NMDA flags, the output directory and the stem; every IPSC-12 line
  differs from its TRACE twin only by `--tau-gi 12` and the stem; every `--nmda-g0` line has `--nmda-tau`; W_MS 429 appears only in
  APV, GRID and K1-NOFAC; waves 1 and 2 are written in that order.
- **Sizing:** f1i measured peak RSS 333 MB and about 5 s for one line, so `mem_gb=1` and memcap 2 as f1i. 1042 lines are about
  87 CPU-minutes in total. Before posting the lines, Claude runs one seed-41 mechanics smoke per arm on the build branch,
  disclosed in this section and setting no value.
