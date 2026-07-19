# gap#5 (i) SWR generative-replay READOUT — precisely localized to a ca3→ca1 CONDUCTANCE CAP (NOT the completion, pathway, threshold, or latched-set). The completion WORKS (latched 116-320 cells) and the Schaffer pathway is abundant (61161 synapses), CA3 fires in phase-2, CA1 receives g_e — but ca1_g_e does NOT scale with schaffer_boost (tracks CA3 firing rate instead), so boosting weights is silently clipped and CA1 never depolarizes past rest. A precisely-characterized hard integration; the next lever is the effective-strength cap / CA1 excitability, NOT a bigger boost.

**2026-07-18.** After the user flagged the gap#4-vs-gap#5 order drift, I re-prioritized gap#5 and ran its "imaginative
replay" readout (`read_ca1`, the two-phase SWR ripple) — which was BUILT but never run. It does not fire CA1
(`ca1_fire=0` at every schaffer_boost 2→400). Four diagnostics localized why, precisely.

## The read (seed 42, n_ca3=2000, the validated completion GO config)
| schaffer_boost | latched (completed CA3) | CA3 phase-2 fire-rate | ca1_g_e | ca1_v | ca1_fire |
|---|---|---|---|---|---|
| 8   | 116-320 | 0.09-0.26 | 0.33-0.59 | -66 to -67 | 0.000 |
| 50  | 112-318 | 0.09-0.25 | 0.23-0.43 | -65.7 to -67.6 | 0.000 |
| 150 | 114-318 | 0.09-0.25 | 0.23-0.43 | -65.7 to -67.6 | 0.000 |

Schaffer ca3→ca1 synapses found = **61161**, mean_w(pre-boost) = 3.558, n_ca1 = 120.

## The precise localization (4 diagnostics — what it is NOT, and what it IS)
- **NOT the completion:** 116-320 CA3 cells latch/fire (far_max ~0.45) — the completion works (it's the CLOSED 5/6-GO).
- **NOT the latched-set threshold:** latched is far from empty (100-320 cells at thr 0.08; ~same at 0.02/0.04).
- **NOT a missing pathway:** 61161 ca3→ca1 (Schaffer) synapses exist, mean_w 3.56, excitatory (inhibitory ca3 excluded).
- **NOT CA3 failing to fire in phase-2:** the latched cells fire at rate 0.09-0.26 under the ripple.
- **NOT `_hard_silence` leaving CA1 clamped:** it is a clean reset (external current → bias 0, no residual).
- **IT IS a ca3→ca1 effective-CONDUCTANCE CAP:** `ca1_g_e` (~0.25-0.6) does NOT scale with schaffer_boost — it tracks
  the CA3 FIRING RATE instead (boost=8/fire=0.26→g_e=0.59; boost=150/fire=0.25→g_e=0.43). Multiplying the ca3→ca1
  weights (cc.data ×boost, up to 3.56×400≈1400) does NOT raise the effective conductance → the boost is SILENTLY
  CLIPPED (a known weight-clip / effective_synaptic_strength cap in this codebase, CLAUDE.md gotcha). With g_e ~0.5nS
  the drive is ~33 pA (g_e × ~66mV driving force) — ~20× too weak to fire an Izhikevich CA1 cell from -66mV, and it
  CANNOT be increased by boosting because the boost doesn't reach g_e. So `ca1_v` stays at rest (-66) and CA1 fires 0.

## BOTH diagnosis-aligned levers EXHAUSTED (2026-07-18) — it is a HARD conductance cap, not weight OR rate
- **Weight lever (schaffer_boost 8→400):** ca1_g_e ~0.25-0.6, unchanged → the boost is clipped.
- **Rate lever (ripple_pA 800→8000):** CA3 phase-2 fire-rate ROSE to 0.32-0.41 (from ~0.1), yet **ca1_g_e stayed
  0.11-0.46** (unchanged/lower), ca1_v -66 to -68 (rest), ca1_fire 0. So g_e does NOT scale with the CA3 firing rate
  EITHER. ⇒ ca1_g_e is HARD-CAPPED at ~0.5 regardless of weight or rate — a bridge-level ca1 synaptic
  conductance/effective-strength cap. Neither runner-side lever can overcome it; the fix is a bridge-level `sim/`
  investigation (the effective_synaptic_strength / g_e cap on the ca1 pathway), a focused future pass.

## Refinement (2026-07-18): the boost IS in the live weight matrix, so it's a deeper g_e-PATH puzzle
`base_synaptic_weights = self.cp_connections.data` (bridge.py:5970) directly — the schaffer_boost edits `cc.data`, so
the boosted weights ARE in the live matrix the g_e matmul uses (not a cached CSR, not a separate array). Yet ca1_g_e
doesn't scale with them. So it is NOT a simple weight-cache staleness; it is a deeper g_e-path issue (e.g. the ca1
region's synaptic-current path, a per-region conductance scaling, or the ca1 cell type's g_e→current conversion). A
genuine `sim/`-level investigation — the honest limit of runner-side diagnosis. Deferred to a focused future pass.

## 🎯 ROOT CAUSE FOUND (2026-07-19): STP DEPRESSION on the Schaffer (ca3→ca1) crushes g_e — the cap is not a mystery
With `enable_short_term_plasticity=False` (diagnostic): PEAK ca1_g_e jumps ~1 → **~3000** (the boost NOW reaches g_e) and
**ca1 FIRES** (0.34). ⇒ STP depression was the cap: `effective_synaptic_strength = cc.data × stp_u × stp_x`, and under
the ripple's sustained ca3 firing the resource `stp_x` depletes so the effective Schaffer strength is capped by the
resource, NOT the weight — boosting the weight is nullified. Two caveats (why the fix is TARGETED, not global STP-off):
(a) global STP-off makes the COMPLETION run away (latched 2000/2000, ca1_v → −30000 numerical blow-up — STP was also
bounding the ca3→ca3 recurrent); (b) g_e=3000 at boost=800 is wildly over-driven. **⇒ THE FIX = disable STP on the
Schaffer (ca3→ca1) pathway ONLY (keep it on ca3→ca3 for the completion) + a MODERATE boost → the completed assembly's
volley reaches ca1 cleanly.** A per-pathway STP config, not a global toggle. Turns the SWR readout from a "hard
integration" into a targeted fix. (Next: implement per-pathway/phase-2 Schaffer STP-off + moderate boost, verify ca1
fires WITH SPECIFICITY — ca1_match >> ca1_cross.)

## 🎯 SPECIFICITY BARRIER FOUND (2026-07-19): fixed-random DENSE Schaffer can't discriminate assemblies — needs LEARNED weights
With the STP fix (phase-2 STP-off → ca1 FIRES), the readout now has the OPPOSITE problem: **no specificity** (ca1_match =
ca1_cross = 1.000 at every boost 0.02→20 + every ca1_ff_inhib 20→150). Every completed assembly drives EVERY ca1 cell
to saturation (fire_sum ~40-50 of 120, ca1_v explodes negative from Izhikevich u-accumulation under the sustained
ripple). ROOT CAUSE (structural): the Schaffer ca3→ca1 projection is FIXED-RANDOM + DENSE (61161 synapses, ~510 inputs
per ca1). A large completed assembly (~300-400 cells) delivers NEAR-IDENTICAL drive (~76±9 inputs) to EVERY ca1 cell →
no cell is preferentially driven by ANY specific assembly → E%-max inhibition can't discriminate a near-tied drive, and
reducing the boost just moves between all-fire and all-silent (no specific-subset window). ⇒ **the SWR readout
specificity fundamentally needs LEARNED Schaffer weights** — the ca3→ca1 association POTENTIATED during encoding (the
CA3-assembly → CA1-target-pattern binding), so recall of an assembly drives ITS specific ca1 pattern. That is the
biologically-correct consolidation mechanism (Schaffer collateral LTP), NOT a fixed-random projection. **NEXT MECHANISM
(clear, biology-grounded): (a) LEARN the Schaffer ca3→ca1 during encoding** (Hebbian/BTSP potentiation when the assembly
co-fires with a target ca1 pattern) → recall gives the specific pattern; OR (b) a BRIEF single-volley sharp-wave read
(not the 60-step sustained ripple) so ca1 fires ONCE (the who-fires-most pattern) + sparse top-k. (a) is the real fix.
⇒ gap#5 (i) went from "hard integration, ca1_fire=0, mystery cap" → precisely: STP crushed g_e (fixed) + fixed-random
Schaffer blocks specificity (needs learned associations). A clear mechanism build, no longer a mystery.

## 🎯 SPECIFICITY BOTTLENECK = COMPLETION DISTINCTNESS (2026-07-19, learned-Schaffer + latched breakdown)
Built the LEARNED-SCHAFFER fix (`swr_learn_schaffer`: each assembly potentiates ca3(assembly)→ca1(distinct sparse
target), non-associated Schaffer held low). With phase-2 STP-off it makes ca1 fire SPARSE + SANE (fire ~0.08-0.10 = ~10
cells, ca1_v back near rest −70, no explosion) — a big improvement — BUT still **match ≈ cross (~0.98)**. The latched
breakdown (per-assembly counts of the completed CA3 cells) shows WHY, decisively: the completion SPREADS across
assemblies AND the pre-assigned assemblies OVERLAP. E.g. cue-1 latches 233 of assembly-1 but ALSO **163 of assembly-0**;
part-1 latches BOTH nearly equally (128 vs 116); the non-assembly count is NEGATIVE (double-counting = the two
assemblies share ~28 cells). ⇒ both cues activate BOTH assemblies → both ca1 targets fire → the ca1 pattern is a common
mix → no discrimination. **So the SWR readout specificity is BOTTLENECKED by the COMPLETION's distinctness** — the same
weak/spreading completion (cue ~0.18-0.22 magnitude residual) + overlapping assemblies. The readout FIRING is fixed
(STP root-cause + learned Schaffer); the SPECIFICITY needs a cleaner, more-distinct, non-overlapping completion (a
DEEPER arc = the completion-magnitude/distinctness residual + disjoint assemblies). ⇒ gap#5 (i) SWR is now precisely a
2-stage chain: (1) firing — SOLVED; (2) specificity — downstream of completion distinctness (the deeper open residual).

## DISJOINT assemblies → specificity PARTIALLY CLOSED (2026-07-19): cross 0.98 → 0.80; residual = completion dominant-attractor
`swr_disjoint` (draw all assemblies from ONE without-replacement pool → no shared cells) removes the overlap that seeded
cross-talk. Result: assembly-0 completes CLEANLY (breakdown [239,0] full, [161,0] partial — zero spread) and specificity
IMPROVED: **ca1_match 0.993 vs ca1_cross 0.803** (was 0.98/0.98). ⇒ removing the overlap fixed HALF the cross-talk. The
RESIDUAL is ASYMMETRIC: assembly-1's completion still co-activates assembly-0 (breakdown [206,232]) — a DOMINANT-ATTRACTOR
effect (one assembly is a stronger attractor pulled in from the other's cue). That is the completion recurrent-BALANCE /
distinctness residual (not overlap). **Next lever (for a future pass): balance the completion attractors (stronger
between-assembly selective inhibition, or balanced-strength encoding) so no assembly dominates → cross → low.** ⇒ SWR (i)
FINAL state this session: firing SOLVED, specificity PARTIALLY closed (cross 0.80), residual precisely named
(completion dominant-attractor). A major advance from "mystery cap / ca1_fire=0 / hard integration".

## Residual root cause = RECURRENT cross-assembly attractor dynamics (sparing REFUTED, 2026-07-19)
Tested the "static spare-all causes spreading" hypothesis: `selective_inhib=OFF` made specificity WORSE (cross
0.775→0.987) and shifted WHICH assembly spreads. ⇒ the sparing is NOT the cause — the cross-assembly spreading is the
RANDOM RECURRENT ca3→ca3 connectivity creating cross-assembly attractor paths, with one assembly an ASYMMETRIC DOMINANT
attractor (e.g. cue-1 completes assembly-1 [233] but the recurrent pulls in assembly-0 [206]). This is the deep
completion-distinctness/BALANCE residual (same class as the completion-magnitude residual). **The real fix = DYNAMIC
between-assembly WTA (lateral inhibition strong enough that the active assembly suppresses the others despite the
recurrent cross-drive) — a genuine mechanism build (pattern-separation competition), NOT the static spare-all the
completion has.** ⇒ SWR (i) FINAL (this session): a precise 2-stage chain — (1) firing SOLVED (STP root-cause + learned
Schaffer + phase-2 STP-off), (2) specificity PARTIALLY closed (disjoint assemblies, cross 0.775) with the residual
root-caused to recurrent cross-assembly dominance → the named fix is dynamic between-assembly WTA (future build). A
major advance from the opening "mystery cap / ca1_fire=0 / hard integration".

## ⚠️ HONEST CORRECTION (2026-07-19): the fb_inhib=40 "specificity" (cross 0.31) was NON-ROBUST — the completion is a NEAR-TIE
A single fb_inhib=40 run showed clean specificity (match 0.99 vs cross 0.31) and I nearly claimed the SWR readout closed.
**Multi-seed (42/43/44) = 0/3 GO** (cross 0.72-0.86); a 3× same-seed variance check = reproducibly cross **0.77**. The
0.31 run had ONE difference: `SWR_DEBUG=1` (read-only instrumentation). Its `to_host` synchronizations shifted the FP
summation order of the NON-DETERMINISTIC transpose SpMV (bridge.py:6193, `deterministic_transpose_matvec` default off),
which FLIPPED the completion's dominant-attractor near-tie from spreading (0.77) to specific (0.31). ⇒ the completion
between-assembly separation is a NEAR-TIE so fragile that debug-vs-no-debug flips it — the SWR specificity is NOT robustly
closed; the honest typical result is cross ~0.77 (not specific). The multi-seed + variance + config-diff discipline caught
a debug-on lucky run I briefly believed. **⇒ SWR (i) HONEST final: firing SOLVED (robust); specificity NOT closed —
bottlenecked by a NEAR-TIE completion (assemblies not distinctly separated), the completion-distinctness residual. The
fix requires a genuinely more-DISTINCT completion (better-separated attractors) — deep completion-quality work, the same
residual class as the completion magnitude.** Not a quick tune (fb_inhib/disjoint help but don't robustly separate).

## STRENGTH SWEEP + ANTI-CHEAT (2026-07-19, parallelized 6-seed) — mechanism VALIDATED, specificity needs PATTERN SEPARATION
Parallelized sweep of the within-assembly separation strength (found via the parallel batch, not the serial grind):
- **strong_within (hebb_lr 4 / lam_dep 1):** 2/3 GO (cross ~0.27 on the separable seeds, 0.845 on seed 44) — best.
- **h8/l2:** 2/6 GO (cross 0.589); **h12/l2:** 1/6 GO (cross 0.708) — MORE strength is WORSE (over-strong recurrent).
- **NO-LEARN anti-cheat (uniform boost, no learned Schaffer):** 0/6, cross 0.999 — the learned ca3→ca1 association is
  GENUINELY LOAD-BEARING for the specificity (not a fixed-random-projection artifact). ⇒ the readout MECHANISM is
  validated; the learned Schaffer is the specificity source.
- **⇒ the specificity is SEED-DEPENDENT (~2/6), NOT robustly closeable by any within-assembly strength lever.** The
  completion near-tie is a FUNDAMENTAL property of RANDOM assembly codes (random recurrent → attractors sometimes
  distinct, sometimes near-tie, by seed). **Robust specificity fundamentally needs PATTERN-SEPARATED assembly codes —
  which is exactly the emergent-DG's job (sparse decorrelated codes).** ⇒ the two gap#5 extensions UNIFY: robust SWR
  specificity REQUIRES the emergent-DG's pattern separation; random pre-assigned codes give seed-dependent near-ties.
  ⇒ SWR (i) FINAL: firing SOLVED + readout mechanism VALIDATED (learned Schaffer, anti-cheat clean); specificity
  bottlenecked by (and requiring) the emergent-DG pattern separation = the shared gap#5 unlock.

## NEAR-TIE RELOCATED (2026-07-19, a-1 RAG catch — avoided a false "CONFUSED") — CA3 completion IS specific; the near-tie is DOWNSTREAM in the CA1 readout
A cross-assembly specificity test (does A's partial cue complete to A not B?) came back "CONFUSED" (within≈cross≈0.02),
but the POSITIVE CONTROL was DEAD (within-completion ~0.015 ≪ the 0.30 completion bar) → the verdict was VOID, not a
real confusion (silent-failure rule #1: never lift a metric from a run whose positive control fails; the test reused the
coincidence-completion-via-language_input path with default params + a weak feedforward drive → no completion fired).
The a-1 RAG check then found the answer already banked: **`2026-07-09-riii-emergent-ca3-completion-kopsick-formation` —
the EMERGENT CA3 completion is 6-seed GO with within/cross ratio 12.6×** (self-organized assembly, DIRECT synchronous
gamma drive, `hebb_max=120 k_thresh=20`; non-assembly / LINEAR / NO-TRAIN / PERM-CUE = 0.000 every seed). ⇒ the CA3
RECURRENT completion is SPECIFIC when the assembly is SELF-ORGANIZED (a strong learned within-recurrent attractor).
**So the SWR specificity near-tie (cross 0.72-0.86 in the strength sweep) is NOT in the CA3 completion — it is DOWNSTREAM
in the Schaffer→CA1 readout, and/or a consequence of the SWR using RANDOM/disjoint assemblies (weak within-recurrent) vs
the emergent-completion's SELF-ORGANIZED assemblies.** The decisive next test: feed SELF-ORGANIZED (or mossy-SELECTED,
`2026-07-19-...-SELECTION-de-risked-GO-6seed`) assemblies into the SWR readout via the runner's `assemblies_ext` hook,
and re-measure CA1 specificity — does the near-tie survive well-separated, strongly-attractored assemblies (⇒ the readout
is the bottleneck) or vanish (⇒ it was the random assemblies)? This connects the emergent-DG SELECTION + the emergent
CA3 completion (both GO) to the SWR readout.

## LATCHED-BREAKDOWN LOCALIZATION (2026-07-19, validated BASE config + SWR_DEBUG) — the near-tie is in BOTH layers (confirms the research "both compound")
Ran the SWR readout (`n_ca3=2000, assembly_frac=0.12, bistable, selective_inhib, hebb_lr=4, learned-Schaffer hi=80/lo=0,
swr_disjoint`) with SWR_DEBUG to read WHICH assembly's cells latch when a specific assembly is cued (n_mem=3, 3 seeds):
- **Seeds 42/43 — CA3 completion is CROSS-CONFUSED.** Cueing assembly A latches A/B/C NEAR-UNIFORMLY (`[144,114,131]`,
  `[56,238,122]` — the cued assembly leads only slightly; total latched ~400 of 720 = ~55% of ALL assembly cells regardless
  of which is cued). ⇒ the 12%-LARGE, ASYNC (`no_sync=True`), no-k_thresh assemblies do NOT pattern-separate → the
  completion SPREADS across assemblies. The CA1 near-tie (match 0.98 / cross 0.97) is INHERITED from this CA3
  cross-completion.
- **Seed 44 — CA3 completion IS SPECIFIC yet CA1 STILL near-ties.** Cueing A latches A dominantly (`[239,72,63]`), but
  ca1_match 0.98 / ca1_cross 0.97 — **the CA1 readout does NOT preserve a specific CA3 pattern.** PEAK-ca1_g_e ~180-250
  drives EVERY CA1 cell (the Valero-2017 "all-fire" collapse: dense-uniform drive → no cell-specific selectivity), and the
  read fires all of them → a broad, non-discriminating CA1 pattern.
- **⇒ BOTH layers are bottlenecks (the research's "both compound" — CONFIRMED empirically):** (b) CA3 cross-completion
  (fix: sparser ~3% + synchronous + k_thresh-specificity / self-organized assemblies) AND (a) CA1 readout degeneracy (fix:
  E%-max top-k read to convert the all-fire g_e 180+ into a winner-set + sparse structured Schaffer). The single
  highest-leverage CHEAP fix is the **E%-max CA1 top-k read** — it fixes seed-44's all-fire collapse directly (fire only
  A's strongest-driven target cells) and is a pure read-side change. NEXT BUILD ORDER: E%-max read → sparser/self-organized
  assemblies → sparse structured Schaffer (the stack, cheapest-first).

## E%-MAX READ ALONE = NOT-GO (2026-07-19) — confirms the research: the completion is the GATING fix
Built the E%-max CA1 top-k read (`swr_ca1_topk`, additive/default-None/byte-identical — the `if _topk_ge is not None`
guard skips all new code when off; the topk=None block reproduced the baseline exactly): fire only the top-k CA1 cells by
peak Schaffer g_e (the winner-set). Result (BASE config, n_mem=3, 3 seeds):
- baseline (topk=None): match 0.981 / cross 0.981 (the near-tie). topk=0.05: match 0.804 / cross 0.848 (ratio 0.95, WORSE);
  topk=0.10: match 0.896 / cross 0.873 (ratio 1.03, barely). **E%-max alone does NOT break the near-tie** (cross stays ~0.85).
- **WHY (confirms Valero 2017 + the scout's "E%-max necessary-not-sufficient"):** the CA3 completion is CROSS-confused (A's
  cue latches A+B+C, per the latched-breakdown), so the Schaffer drives A's AND B's AND C's CA1 targets → the CA1 g_e is
  NON-SPECIFIC → the top-k winner-set includes cells from all targets → cross high. No read-side sharpening can discriminate
  a non-specific DRIVE. **⇒ the GATING fix is the COMPLETION (#3): sparser / synchronous / pattern-separated / self-organized
  assemblies so A's cue completes ONLY A → the Schaffer g_e becomes SPECIFIC → THEN E%-max discriminates.** The emergent-DG
  selection (6-seed GO) is the shared unlock. NEXT: test the cheapest completion fix — sparser assemblies (`assembly_frac`
  0.12→0.03) + synchronous drive (drop `no_sync`) + k_thresh specificity (the emergent-completion 12.6× recipe), then re-add
  E%-max on top. The E%-max option is kept (it's component #1, needed once the completion is clean). NO sim/ edit (runner-only).

## Status (per THE LAW — a precisely-characterized boundary that names the next lever)
- **The SWR readout is BLOCKED by the ca3→ca1 effective-conductance cap** — a real, precisely-localized hard
  integration (the documented "hard fresh-pass integration" snag, now root-caused). It is NOT closeable by the
  schaffer_boost lever the code provides (that's the wrong knob — it's clipped).
- **Next levers (NOT a bigger boost):** (a) find + raise the effective_synaptic_strength / conductance cap on the
  ca3→ca1 pathway (a bridge-level clip); (b) raise CA1 excitability (lower threshold / a controlled depolarizing bias)
  WHILE preserving specificity via a competitive CA1 mechanism (the completed assembly's volley → its OWN ca1 pattern,
  not a uniform bias that fires all CA1); (c) more/faster CA3 firing in phase-2 (higher ripple_pA / longer gamma) to
  raise g_e via the rate (the only thing g_e currently tracks). A focused future pass.
- **BOTH gap#5 extensions are hard, precisely-characterized integrations, NOT quick wins:** (i) this SWR readout
  (ca3→ca1 conductance cap), and (ii) emergent-DG (needs the layer-2 amplification wired in). The completion MECHANISM
  is CLOSED (5/6 GO); extending it in either direction is non-trivial. This is the honest gap#5 map.
- Infra: SWR_DEBUG-gated instrumentation in `_measure_ca1` + the schaffer-boost block (default-off → byte-identical).
