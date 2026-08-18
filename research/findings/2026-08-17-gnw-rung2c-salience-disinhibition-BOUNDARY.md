---
status: live
type: finding
lane: laneC
date: 2026-08-17
---

# GNW Rung-2c — salience-gated DIS-INHIBITION PULSE eviction on a competitive workspace: BOUNDARY (6/6 no clean salience-eviction; a salience-gated release of the shared inhibition lets a challenger IGNITE but cannot REMOVE the incumbent — the dense recurrent attractor is inhibition-resistant, and the attractor depth that makes it evictable makes it fail to hold, so the co-ignition/self-extinction dichotomy TRANSFERS to inhibition-based eviction — the named next lever is to make the incumbent's RECURRENCE dynamically weakenable, e.g. short-term depression on the recurrent synapses / a distributed overwrite workspace, so the pulse has a metastable state to tip)

**Date:** 2026-08-17
**Runner:** `research/runners/_gnw_rung2c_salience_disinhibition_derisk.py` (reuse-by-import of the Rung-1 assembly-loop/snapshot helpers + the Rung-2/2b async competitive harness; the eviction effector is built by EXPLICIT WIRING only — per-slot inhibitory pools + per-slot disinhibitory VIP pools — **NO `sim/` edit**).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** a decisive, well-anti-cheated **method-negative** — a TRANSIENT SALIENCE-GATED DIS-INHIBITION PULSE (the mechanism the Rung-2b boundary NAMED) is NOT, by itself, the workspace-eviction effector on this substrate. This is a verdict on the METHOD, not the CAPABILITY: salience-eviction is NOT abandoned; the next lever (dynamically weakenable recurrence) is named below.
**Builds on / cites:** extends the Rung-2b BOUNDARY `research/findings/2026-08-14-gnw-rung2b-sfa-workspace-eviction-BOUNDARY.md` (intrinsic SFA self-extinguishes before it yields; the continuous no-reset protocol proven with 0 `_restore_state` calls) and its parent Rung-2 `research/findings/2026-07-07-GNW-rung2-competitive-access-mutual-exclusion-GO-salience-eviction-PENDING.md`; **(Dehaene & Changeux, 2011)**, *Neuron* 70(2):200-227 (metastability: an ignited workspace state must be able to be "destabilized" and "spontaneously replaced by another"); and the thalamic-reticular / pulvinar attention-gating framing (destabilization driven from OUTSIDE the incumbent's own recovery current). `NO-EXTERNAL-NEEDED:` the mechanism, its two architectural variants, and the named next lever are all fixed by the corpus (the Rung-2b/2 findings + the banked eviction negatives) and the cited source; this is a method-negative that launches the next build.

## What was tested (the effector Rung-2b named)
The eviction effector is a phasic ATTENTION-SHIFT RELEASE of the inhibition, driven from OUTSIDE the assemblies by the challenger's SALIENCE (`vip_current = salience_gain * drive_chal`, a pulse of `pulse_duration` steps into a VIP-like disinhibitory interneuron pool), NOT the incumbent's own recovery current. Two architectures were built and swept, both continuous (no mid-competition restore):

<!--derived-->
1. **GLOBAL shared-inhibition release** (one shared `workspace_fs` pool; the VIP pulse suppresses it) — the literal "release the shared FS->workspace inhibition".
2. **BIASED cross-inhibition release** (the Wang-2002 two-attractor motif: per-slot pools `fs_A -| A`, `fs_B -| B`, each driven by the OTHER assembly so `A -> fs_B` and `B -> fs_A`; per-slot VIP pools `dis_A -| fs_A`, `dis_B -| fs_B`; the challenger's salience releases the challenger's slot, so the released challenger's firing drives the INCUMBENT's pool to suppress it). This is the mechanistically-correct "attention releases the attended item, which then wins competition" version.

## Result — no frozen point gives ignite-hold-AND-cleanly-evict (the dichotomy TRANSFERS)

<!--derived-->
Both architectures reproduce the Rung-2b dichotomy. The salience-gated release DOES let the challenger IGNITE (measured: driving the challenger's dis pool drops its slot's inhibition, and the challenger reaches the 1/3 plateau), and in the biased architecture the released challenger DOES drive the incumbent's inhibitory pool (`fs_A` firing rose from ~0.04 to ~0.17) — **yet the incumbent HOLDS at its 1/3 period-3 plateau throughout**. The dense weight-30 recurrent attractor is inhibition-resistant: any inhibition still weak enough to permit ignition cannot dislodge an established assembly, so the challenger CO-IGNITES (n_ignited=2, mutual exclusion breaks) instead of DISPLACING it. Sweeping the attractor DEPTH does not open a window: `attractor_weight <= ~22` SELF-EXTINGUISHES (the incumbent never holds; late rate 0.00-0.11 < the 1/6 ignite threshold), `attractor_weight >= 26` holds-but-co-ignites — the metastable middle is EMPTY. Stronger shared/cross inhibition only broadens co-ignition; it never evicts.

Frozen operating point (salience_gain=1.0, dis_to_fs=30, pulse_duration=25, fs_to_ws=16, a2fs=4, ou=40, incumbent_settle=120, drive_inc=5000, chal in [0,8000]x9), biased architecture, all 6 seeds:

| seed | winners[chal 0->8000] | mutual excl | holds weak | takes strong | causal_swap | n_ignited post | verdict |
|---|---|---|---|---|---|---|---|
| 42  | AAAAAA222 | ✗ (co-ignite) | ✓ | ✗ | 0.00 | 2 | NO-GO |
| 43  | AAAAAA222 | ✗ (co-ignite) | ✓ | ✗ | 0.00 | 2 | NO-GO |
| 44  | AAAAA2222 | ✗ (co-ignite) | ✓ | ✗ | 0.00 | 2 | NO-GO |
| 100 | AAAAAA222 | ✗ (co-ignite) | ✓ | ✗ | 0.00 | 2 | NO-GO |
| 101 | AAAAAAA22 | ✗ (co-ignite) | ✓ | ✗ | 0.00 | 2 | NO-GO |
| 102 | AAAAAAA22 | ✗ (co-ignite) | ✓ | ✗ | 0.00 | 2 | NO-GO |

The incumbent holds every sub-crossover weak challenger on all 6 seeds; the strong challenger CO-IGNITES on all 6 (the crossover-to-co-ignition point drifts across seeds, chal 5000-7000). Never a single clean "B" (evict). `any_op_go=False`.

## Anti-cheats (each held — this is a real negative, not an artifact)
- **The pulse is SALIENCE-DRIVEN, not a host reset:** `vip_current = salience_gain * drive_chal`, so a zero-salience challenger gets `vip_current=0` (no pulse) and the pulse amplitude scales monotonically with challenger salience (`pulse_zero_at_zero_salience` and `pulse_scales_with_salience` both True, all 6). It is a current/conductance change into a spiking interneuron pool, timed as a pulse — asserted in code, not a state reset.
- **PULSE-OFF reproduces the negative:** `salience_gain=0` (vip_current=0 everywhere) FAILS the clean takeover on all 6 (`reproduces_negative=True`) — the salience-gated pulse is genuinely load-bearing on the (still-failing) dynamics; its absence does not make eviction better OR clean.
<!--derived-->
- **WTA lesion (`fs_to_ws=0`) co-ignites, and the mutual exclusion is 100% inhibition-attributable** on all 6: at the DISCRIMINATING drive (the strongest challenger the incumbent still HOLDS, chal 4000-6000 across seeds) the intact circuit keeps the challenger suppressed (single content, A-B rate gap ~0.333) while the lesion lets it ignite (co-ignition, gap ~0.000) -> `attributable_to(intact_gap, lesion_gap) = 1.00` on all 6. So the mutual exclusion in the hold region is caused by the (cross-)inhibition, not by the pulse silencing one assembly. (A weak-challenger comparison would be uninformative — a weak challenger stays off even without inhibition; the strongest-held drive is where the two arms actually diverge.)
- **Scaffold removed / continuous:** the headline takeover attempt ran with ZERO `_restore_state` calls (`no_restore_calls=True` all 6) — the eviction failure is genuine competitive dynamics, not a missing per-hop wash-out.
- **Determinism:** build-twice at one seed -> identical hash of the seed-derived Izhikevich params (heterogeneity seeded from `cfg.seed`, NOT `actual_seed_used`) on all 6.
- **Verdict is a clean NO-GO negative, not UNDEFINED:** all VALIDITY preconditions hold (incumbent ignites & holds a weak challenger, pulse salience-gated, WTA load-bearing, PULSE-OFF reproduces the negative, continuous no-restore, determinism); the negative is the ABSENCE of a clean single-winner eviction (co-ignition n_ignited=2), which is the MEASURED outcome, not a failed precondition.

## Why the dichotomy transfers to inhibition-based eviction (the diagnosis)
The ignited assembly is a rigid ALL-OR-NONE period-3 attractor (exactly 1/3 or 0; het+OU did not desync it, per Rung-2b). Somatic/feedforward inhibition — whether global shared release, biased cross-inhibition, or the challenger driving the incumbent's pool — cannot grade it down: below the inhibition that would evict, both co-exist (co-ignition); above it, neither ignites; and the attractor depth that would make the incumbent evictable makes it fail to HOLD (self-extinction). This is the SAME horn Rung-2b hit with intrinsic SFA ("the fatigue that would evict equals the fatigue that kills"), now shown to be a property of the SUBSTRATE (an inhibition-resistant dense recurrent attractor), not of the specific fatigue effector. The **companion-process** question (CLAUDE.md's deepest lesson) resolves it: we replaced the incumbent's DYNAMIC recurrent efficacy with a STATIC frozen weight, and froze it DEEP (far above the Rung-1 ignition knee) — so the metastability Dehaene-Changeux require (a marginally-stable state a small perturbation can flip) does not exist for the pulse to tip.

## The named next lever (this launches the next build; it is NOT a wall)
Make the incumbent's RECURRENCE dynamically weakenable so a metastable state exists for the salience-gated pulse to tip — the missing companion process, not a stronger inhibition:
1. **Short-term DEPRESSION on the recurrent (E->E) synapses** (Mongillo, Barak & Tsodyks 2008, *Science* 319:1543, synaptic theory of WM): a long-held incumbent's LOOP depresses with use, so its attractor is shallow by the time the challenger arrives, and the released challenger's cross-inhibition then tips it — depression OF THE LOOP (evictable-yet-holding) rather than adaptation of the SOMA (Rung-2b's SFA, which kills the neuron). CAUTION: STP is banked as annihilating for eviction (2026-08-01), but on a SINGLE self-exciting pool with no competitor and at a non-Mongillo operating point — it must be re-examined in the competitive setting at the facilitation-dominated Mongillo point, not naively retried.
2. **A distributed OVERWRITE workspace**: replace the two hand-wired localist assemblies with overlapping PATTERNS in one recurrent net + divisive normalization / a shared global-gain resource, so igniting a new salient pattern INHERENTLY de-ignites the incumbent by shared-resource competition (Hopfield-style pattern rivalry) rather than two separable attractors that can co-ignite.

## Do-NOT-retread (banked)
GABA_B eviction KILLED; STP annihilates on a single self-exciting pool; active-clear FS quench works but is a HOST shortcut; intrinsic Izhikevich SFA self-extinguishes before it yields (Rung-2b); **and now: a salience-gated dis-inhibition PULSE — in BOTH the global-shared-release and the biased cross-inhibition forms, across the full attractor-depth range — cannot cleanly evict the dense recurrent workspace attractor; it lets the challenger co-ignite but cannot remove the inhibition-resistant incumbent.** The residual is small and specific: the effector needs a metastable incumbent (dynamic recurrence), which somatic/feedforward inhibition alone cannot create.

## Remaining scaffolds (named, not claimed closed — would still stand on a future PASS)
1. **Salience is host-supplied external drive** (into the challenger's dis pool) — an emergent salience (a value/surprise organ writing the drive + selecting the slot) is a later rung.
2. **The assemblies + the per-slot inhibitory/disinhibitory pools are hand-wired** (dense fixed-weight populations), not self-organized.
3. **The dis pool's TARGET slot is host-routed** to the challenger (the salience signal knows the salient content's slot) — faithful to feature-based attention, but a scaffold until the routing is itself learned/emergent.

## Files
Runner: `research/runners/_gnw_rung2c_salience_disinhibition_derisk.py`. 6-seed frozen-point artifacts (biased architecture): `research/findings/raw/_gnw_rung2c_seed42.json`, `research/findings/raw/_gnw_rung2c_seed43.json`, `research/findings/raw/_gnw_rung2c_seed44.json`, `research/findings/raw/_gnw_rung2c_seed100.json`, `research/findings/raw/_gnw_rung2c_seed101.json`, `research/findings/raw/_gnw_rung2c_seed102.json` (each with a `.prov.json` provenance sidecar).
