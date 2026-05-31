# Biological-composition arc (owner chose Option 2): check-existing-first finds a CONVERGENT compositional boundary (sets-not-sequences, N mechanisms bounded), and identifies ONE genuinely-untried specific angle at the root cause -- engram-tag the CONJUNCTION of concept-pool + strong-dynamics dlpfc_wm role-selective activity (the c-loop had dlpfc but bound a structureless sum-engram). Cheap-first gate before any build.

**Date:** 2026-05-31
**Status:** Arc-opening synthesis. Owner chose Option 2 (pursue compositional capabilities, biologically sound, new ideas OK, catalog as reference). Per the standing check-existing-first discipline (re-earned twice this session), surveyed prior compositional work BEFORE proposing. Found my suggested #2 (generative replay) is already bounded; mapped the convergent boundary; identified one genuinely-untried specific angle at the shared root cause.

## Check-existing-first result (the essential survey)

Prior compositional mechanisms, all explored + characterized:
- VSA near-orthogonal symbol-grounding (this session): BOUNDED -- 3 coding methods + N-independence; near-ortho unreachable from substrate activity.
- Sequence/positional storage (DIRECTION-A ec_context spatial + DIRECTION-E theta-gamma temporal, 2026-05-24): BOUNDED -- both 0.25-0.33 strict top-1 (4-5x chance, below 0.80 bar). The v16 weak-dynamics substrate stores SETS not SEQUENCES.
- Generative replay for sequence completion (c-loop, 2026-05-24): NEGATIVE -- v1 chance (0.057); the identified SWR-reactivation fix WAS written (v2_stim_global: explicit stimulate_tag during SWR + global-mean decode) but only SMOKE-run -> v2 smoke = 0.02 (BELOW chance). So the fix does NOT rescue it; the engram is a structureless SUM-across-slots; substrate-bounded confirmed.
- theta-gamma mode-unification + SPEAR temporal multiplexing: ceiling'd (2026-05-20, prior).
- WM-emergence / engram-vs-STDP binding (phase-factored integrated loop, 2026-05-30): VOID (two horns -- a selectivity carrier that is both stable AND emergent does not exist; STDP emergent-but-unstable, engram stable-but-not-emergent).

CONVERGENT BOUNDARY: the substrate is a robust content-addressable ASSOCIATIVE MEMORY (simultaneous multitag set-binding 91.7%) but resists SEQUENCE / ROLE / compositional-generation. Shared ROOT CAUSE (DIRECTION-E diagnosis): the deliberate WEAK concept-pool dynamics (needed for multi-concept trainability; "canon amplifies bias collapse") make all pool neurons fire ~equally during engram capture regardless of which role/slot drove them -> the engram captures ~the same neurons across roles -> NO ROLE-DISTINCTNESS. Stronger dynamics would fix role-distinctness but break trainability (the tradeoff).

## The genuinely-untried specific angle (at the root cause)

The root is: the WEAK pools can't carry role-distinctness, and prior fixes drove the role signal THROUGH the weak pools (ec_context spatial drive -> weak pools; theta-gamma temporal window -> weak pools) -- which collapse it. The c-loop's substrate HAS a STRONG-dynamics region (dlpfc_wm, NMDA-bistable, validated for persistent activity at pillar n=98) -- but the c-loop used it only as a "frame buffer" and engram-tagged a structureless SUM over concept-pool activity (no dlpfc role component in the tag).

GENUINELY-UNTRIED IDEA: bind concept-to-role by engram-tagging the CONJUNCTION of (concept-pool activity + dlpfc_wm ROLE-SELECTIVE activity). Drive concept(word) into the concept pools AND a role-specific sub-population of dlpfc_wm simultaneously; commit the engram over the UNION (concept pools + dlpfc). Because dlpfc_wm has strong NMDA dynamics, role-1 vs role-2 drive DISTINCT dlpfc neurons (and sustains them), so the conjunctive engram is ROLE-DISTINCT even when the concept-pool component is role-equal. The dlpfc component carries the role-distinctness the weak pools cannot. This DECOUPLES the tradeoff: concept pools stay weak/trainable; dlpfc provides stable role-selectivity.

Biology grounding: PFC mixed-selectivity / ordinal coding (Rigotti 2013; PFC neurons encode item x position conjunctions via persistent selective activity). Distinct from the bounded approaches: NOT feedforward spatial (ec_context), NOT phase (theta-gamma), NOT a sum-engram (c-loop). It is conjunctive binding to a STRONG-dynamics role carrier.

Honest caveat: this is RELATED to the phase-factored two-horns VOID (the dlpfc engram is the "stable-but-not-emergent" horn). But composition does NOT require EMERGENCE -- it requires functional role-distinct binding + role-selective retrieval. The two-horns VOID was about WM-emergence (a different thesis); the compositional FUNCTION (bind + retrieve role-selectively) may be viable on the stable engram horn alone. That is exactly what the cheap-first test decides.

## Cheap-first gate (before any build)

PROBE (on the cached n=98 hippo+dlpfc substrate, mode_unification_with_hippo_dlpfc_cache/, GPU):
- Bind: for a few (concept, role) pairs, drive concept-pool(word) + dlpfc_wm(role-subpop) together; commit engram over concept-pools UNION dlpfc.
- ROLE-DISTINCTNESS test: is engram(concept-A, role-1) DISTINCT from engram(concept-A, role-2)? Measure cosine between the two engrams' captured patterns; and role-selective retrieval (stim engram -> decode (word, role)).
- CONTROL: the c-loop's concept-pool-only sum-engram (NO dlpfc component) -- must be role-INDISTINCT (cosine high; the bounded behavior).
- A genuine PASS = dlpfc-conjunctive engrams are role-DISTINCT (low cross-role cosine) AND role-selective retrieval works, where the concept-pool-only control is role-indistinct. Three-state RESOLVES/PARTIAL/BOUNDARY; frozen bar; reproduce-the-failure control (concept-pool-only) built in.
- If RESOLVES -> dlpfc-conjunctive role-binding is the compositional substrate; build the role-binding composition mechanism. If BOUNDARY -> the dlpfc role-distinctness also collapses (or doesn't survive engram capture) -> the convergent compositional boundary deepens to include conjunctive-strong-region binding; an honest biology-translatable result + pivot to the next genuinely-new mechanism (e.g. stable-attractor role basis, or dendritic compartmental binding).

## Discipline

Check-existing-first done thoroughly (generative-replay + sequence-storage + WM-binding all surveyed; 2 of my own suggestions found already-bounded -- the discipline working). Cheap-first before any build. Reuse-by-import; the cached substrate is reused byte-unchanged; no protected/frozen/moat module touched. Frozen bar + reproduce-the-failure control + three-state. Honest about the relation to the two-horns VOID (the cheap-first decides whether the compositional FUNCTION is viable on the stable horn). Catalog consulted (PFC ordinal coding; the SCIENCE_ROADMAP confirms dlpfc persistent-activity validation n=98).
