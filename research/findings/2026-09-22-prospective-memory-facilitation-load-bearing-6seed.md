---
type: finding
status: live
lane: load-bearing
date: 2026-09-22
---

# Prospective-memory borderline FIXED to load-bearing 6/6 via short-term NMDA-mediated facilitation (default-off flag) (2026-09-22)

Follow-on to `2026-09-22-borderline-separability-stabilizer-is-buildable.md`, which located the prospective-memory
fragile step at the READ level: at the N=3 production protocol (formation -> 3 intervening turns -> cue) the intact
held x cue coincidence read `rel` clears FIRE_THR=0.20 on five seeds (0.266-0.341) but sits JUST below it at seed 44 <!--derived-->
(rel=0.1839, margin -0.016), so prospective-memory was load-bearing only 5/6 (off@s44). The named fix was a <!--derived-->
short-term facilitation of the maintained intention assembly across the intervening turns. This finding BUILDS it,
gated behind a new default-off flag `BRAIN_PMEM_FACILITATION`, and verifies it 6/6 on the load-bearing gate.

## The mechanism (research/runners/_pmem_facilitation_derisk.py; additive, NO sim/ edit; reuse-by-import)
A per-action Tsodyks-Markram SHORT-TERM FACILITATION variable F[a] on the maintained `act_X -> rel_X` projection,
whose EXPRESSED current is NMDA-mediated -- gated by the postsynaptic Mg2+ block:
- `F[a] <- F[a] + U*(F_max-F[a])*h_a - F[a]/tau_F`  (h_a = the act_X assembly firing fraction off cp_firing_states)
- `I_fac[rel_a,i] += fac_g * F[a] * h_a * B(V_i)`, with `B(V)=1/(1+[Mg]/3.57*exp(-0.062 V))` (Jahr & Stevens 1990, <!--derived-->
  the SAME Mg-block form the engine's NMDA kernel uses).

F builds ONLY while the held assembly fires (residual-Ca2+ release-probability increase; Tsodyks & Markram 1997;
Zucker & Regehr 2002; Wang et al. 2006 -- recurrent PFC pyramidal synapses are strongly facilitating, tau_F ~ 1-2 s)
and potentiates the held drive turn-over-turn. The coincidence-specificity is the NMDA receptor's OWN
voltage-dependence: at the intervening (held-ALONE) turns the rel neuron sits near rest -> Mg-BLOCKED -> almost no
facilitated current; at the cue the cue drive DEPOLARIZES the rel neuron -> the Mg block is relieved -> the
facilitated held x cue coincidence current passes and tips the NMDA-recurrent accumulator over FIRE_THR. The
operating point is calibrated WITH facilitation active (the homeostat pins the facilitated held-alone sub-threshold
and the plateau theta sits above the facilitated single input) -- "the instrument is part of the emulation".

## Why the literal (ungated) facilitation FAILED, and this one holds (the load-bearing lesson)
<!--derived-->
A pure presynaptic facilitation current `fac_g*F*h` (no Mg gate) is added whenever the held assembly fires -- so it
lifts the held-ALONE reads too. Calibrated facilitation-OFF it fires s44 (rel 0.222) but REGRESSES the frozen-gate
silence clauses (no_fire_before / no_fire_wrongcue rise to 0.06-0.08 > SILENT_MAX=0.06) -> VOID. Calibrated
facilitation-ON the homeostat ABSORBS the lift (s44 back to ~0.204, fragile). The resolution is the NMDA Mg2+ block:
it is the biological reason the held-alone stays silent while the coincidence fires, and it makes the facilitated
current COINCIDENCE-PREFERENTIAL by construction, so silence holds AND the coincidence keeps its lift. This is the
finding's named "SFA/NMDA-mediated facilitation" -- with the coincidence gate supplied by real receptor biology, not
a label or a hand-tuned threshold.

## Result -- 6-seed, load-bearing gate (LB_PMEM_DRIVE_PROBE=1 + BRAIN_PMEM_FACILITATION=1, numpy CPU, memcap 16)
<!--derived-->
`research/runners/load_bearing_fraction.py --only prospective-memory --repeats 2 --seed <s>`, one process per seed.
| seed | intact `prospective.fired` | lesion `prospective.fired` | load_bearing | null_control_clean |
|---|---|---|---|---|
| 44  | True (rel 0.2111) | False (rel 0.0) | **True** (was False) | True |
| 42  | True | False | True | True |
| 43  | True | False | True | True |
| 100 | True | False | True | True |
| 101 | True | False | True | True |
| 102 | True | False | True | True |

Load-bearing 6/6 (was 5/6, off@s44). Artifacts: research/findings/raw/_load_bearing/pmem_facil_drive_s{42,43,44,100,101,102}.json
(each: `load_bearing_fraction=1.0`, `treatment_diffs=1`, `control_diffs=0`, `lesion_reproduced=true`, `flag_resolves=true`).
The operating-point instrument (`_lbf_borderline_operating_point._prospective`, the runner that produced op_s44.json)
agrees byte-for-byte with the gate (s44 rel=0.2111 on both) and reads 6/6 with the flag; per-seed intact rel with the
flag: 42:0.344 43:0.336 44:0.211 100:0.284 101:0.276 102:0.313 (all >= 0.20; all lesions <= 0.028).

## Default-OFF + byte-identical when off (proven in the data)
<!--derived-->
`BRAIN_PMEM_FACILITATION` defaults OFF. With it off the production organ builds the SAME class as before
(HebbianBindingProspectiveMemory / SFANmdaProspectiveMemory) and the `_step` facilitation hooks are getattr-guarded
no-ops. The operating-point read with the flag OFF, AFTER the code change, reproduces the pre-change baseline EXACTLY
on all 6 seeds (s44=0.183889 identical; n_load_bearing=5/5 both) -- an exact compare, not an inference. Selftest
(`_pmem_facilitation_derisk.selftest`) confirms a plain build carries no `_facilitation_on` and fac_on=False leaves
it off. So the shipped brain + the adequate-battery default are byte-identical to before; the flag is purely additive.
This fix is therefore WIRED (default-off), NOT integrated/on-by-default -- flipping the default is a separate owner
call; the deliverable is the buildable, verified, principled mechanism.

## Anti-cheat / honesty
- The load-bearing SEPARATION is preserved BY CONSTRUCTION: the lesion arm (BRAIN_PMEM_LESION zeroes the latch -> the
  held assembly COLLAPSES) has h_a ~ 0, so I_fac ~ 0 -> the lesioned cue does NOT fire (rel 0.0 at s44). The fire is
  caused by the held x cue coincidence, not a substrate-wide gain.
- The frozen N=5 gate silence clauses STAY 6/6 with the flag on (max_silent 0.039-0.050 < SILENT_MAX=0.06); in the <!--derived-->
  production pipeline the intervening reads at s44 are 0.0/0.03/0.022 (silent). De-risk verdict: GO <!--derived-->
  (research/findings/raw/_pmem_facilitation.json, carries a tools.verdict.Verdict preconditions block).
- Identical F-dynamics params for ALL seeds (fac_g=6000, fac_U=0.18, tau_F=2000); the s44 lift is STABLE across a ~2x
  fac_g range (4000->0.204, 6000->0.211, 8000->0.210) -- a mechanism, not a per-seed tune. FUNCTIONAL correlate only; <!--derived-->
  no claim of phenomenal experience.

## Files
- `research/runners/_pmem_facilitation_derisk.py` (new): the mechanism (`_FacilitationMixin`,
  `FacilitatedProspectiveMemory`, `FacilitatedHebbianProspectiveMemory`), de-risk runner, selftest.
- `research/runners/_pmem_sfa_nmda_amplifier_derisk.py`: two getattr-guarded hook calls in `_step` (byte-identical off).
- `research/runners/prospective_memory_production_organ.py`: `BRAIN_PMEM_FACILITATION` (default-OFF) swaps in the
  facilitation substrate; off builds the same class as before.
