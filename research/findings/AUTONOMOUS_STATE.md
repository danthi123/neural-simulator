# AUTONOMOUS CONTINUATION STATE

> Durable cross-session pointer. Any re-trigger (scheduled watchdog, new
> session, post-compaction) reads THIS first and resumes the exact next
> action without re-deriving context. Update every cycle; commit+push
> both remotes. The conversation is NOT the memory — this file + git are.

**Updated:** 2026-05-31
**Mode:** continuous autonomous (24/7; no self-imposed stopping; only an
explicit user stop/pause or a true safety boundary halts work)

## >>> CURRENT POINTER (read THIS first; 2026-05-31) <<<

TWO live threads:

(1) DONE = DEGRADES-WITH-FANIN (2026-05-31; finding 2026-05-31-P4-multihop-hub-reuse-DECISIVE-
DEGRADES-WITH-FANIN-...md). Multi-seed full-2hop 0.833 at fan-in 2 (>>chance 0.094) -> 0.000 at
fan-in 8. Clean 8/8 was the fan-in-1 easiest case. Bottleneck LOCATED (controller-scrutinized,
verdict survives): hop-1 flat/fine (0.83 all fan-in); entire loss at hop-2 -- querying a crowded
hub returns its many INCOMING nouns and buries the one OUTGOING edge (multitag is undirected/
aggregate-ranked). Anti-cheat held (13-14/14). Fundamental representational limit, not a tuning bug.

(2) NEXT ARC GROUNDED + BANKED (owner-aligned, ready regardless of (1)'s verdict): finding
2026-05-31-theta-multiplexing-conversational-holding-NEXT-ARC-grounding-temporal-not-
spatial-separation.md. The owner's preferred mechanism (theta-phase multiplexing, 2026-05-19
reframe) may SIDESTEP this session's separation-vs-reliability BOUNDARY because it separates
held items in TIME (theta phase slots) not in SPATIAL pattern -- routing around the exact
k-WTA knob that produced the boundary. Existing sim to adopt-from (check-existing-sims-first
directive): Ursino-Cesaretti-Pirazzini 2022 spiking Lisman-Idiart theta-gamma multi-item WM
(PMC10050512). Honest caveat recorded: 2025 Nat Neuro contests strict phase==order -> scope
to HOLDING/non-interference, not order-coding. Substrate already has the ingredients (concept
pools = gamma assemblies; parked integrated-loop Task-2 theta timing controller, reuse-by-import).

thread-(2) CHEAP GATE: DONE = RESOLVES (PASS, scrutinized) 2026-05-31. finding 2026-05-31-theta-
multiplexing-CHEAP-GATE-PASS-...-recovers-Miller-7.md. Pre-reg bar met (N=4 phaseRead 1.000>=0.90,
ctrlRead 0.217<0.50). Survived 3 scrutiny checks: decode margin +0.22 at N<=7 (confident); BOUNDARY-
ESCAPE demonstrated (overlapping cos-0.60 codes -- the regime that FAILED spatial DG separation -- held
0.989 at N=7 via phase, control collapses 0.118); capacity-realism (no-jitter cap 16 is a permissive
artifact; phase jitter 2 bins -> cap exactly 7 = Miller, recovered from theta/gamma ratio). NAMED OPEN
RISK for spiking build: cheap model assumes reader already knows each item's phase slot -> spiking build
MUST test phase-addressing LEARNABILITY + stability across encode/recall. HARD GATE PASSED.

>>> THREAD (2) CORRECTION (2026-05-31): the theta-multiplexing "next arc" is RETRACTED/DOWNGRADED, NOT
a viable new direction. On checking prior in-project work: theta-gamma multiplexing ALGEBRA was already
validated with decisive controls (2026-05-24 Direction E; 2026-05-23 FHRR N16) and its SPIKING-SUBSTRATE
composition already hit a DECISIVE 5-architecture convergent ceiling (2026-05-20-THETA-GAMMA-decisive-
honest-negative). The algebra was never the bottleneck; the substrate composition is the wall. My cheap
gate RE-DERIVED the known algebra (banners on both theta-multiplex docs). The night synthesis (2026-05-31-
NIGHT-ARC-...) had ALREADY pivoted correctly to P4. Residual value kept: the cross-arc insight (temporal
sidesteps the spatial DG boundary) + Miller-7-under-jitter. Do NOT build a theta-multiplexing spiking
arc -- it is a re-tread of ceiling'd work. <<<

DIRECTIONAL FIX DONE = RESCUED-but-BIMODAL (2026-05-31; finding 2026-05-31-P4-multihop-directional-fix-
RESCUES-per-bar-but-BIMODAL-...md). Multi-seed OUT full-2hop at fan-in 8 = 0.583 >= 0.50 bar (vs undirected
ANY 0.000) => RESCUED per the unmoved frozen bar. BUT bimodal: seed 42 = 8/8, seed 44 = 6/8, seed 43 = 0/8.
The directional filter (hop-2 hub query -> hub-first tags only) isolates big_red correctly on ALL seeds;
seed 43's 0/8 is weak UNDERLYING big_red binding on that bridge, not a filter bug. So directional removes the
hub-crowding bottleneck (strict win 0.583 vs 0.000) and EXPOSES residual per-seed binding-quality variance
as multi-hop's next limit. Directional multi-hop = REAL but NOT-UNIFORMLY-ROBUST.

INTEGRATION DONE (2026-05-31, commit 24ea2d4): directional filter SHIPPED into research/runners/
g20_multibridge.py -- _tag_matches_direction pure helper (8 unit tests, no GPU) + query_concept gains
direction='any'(default,backward-compat)/'out'/'in' + return_ranked + new "trace X" 2-hop command. GPU
smoke (160-concept seed 42): existing what-is unaffected; "trace apple" -> "apple relates to big, which
relates to red, new, angry". MULTI-HOP ARC COMPLETE end-to-end (characterized clean->DEGRADES->RESCUE-
bimodal + the validated fix shipped). g20_multibridge no longer byte-unmodified (this is the deliberate
shipped capability); no protected/frozen/moat module touched; backward-compat by construction.

GRID/CONJUNCTIVE BIOLOGICAL ARC = CONCLUDED 2026-05-31 (check-existing-first + cheap-first + scrutiny).
Survey: TEM/tensor-product conjunctive BINDING already covered (2026-05-06 Pick 4) AND the binding algebra
is already validated (FHRR) -> binding was never the blocker. The sharp grid idea (modular REDUNDANT coding)
cheap probe = CANNOT-CONCLUDE (instrument-invalid: M=1 control passes at all densities + id metric saturated
because the RAW activity is ALREADY 16/16 ID-separable, within 0.896 > between 0.768). CLARIFICATION (refines
my own DG "fundamental" overclaim): the substrate activity is already ID-separable (why retrieval works); the
unmet bar is NEAR-ORTHOGONALITY (between->~0) for clean VSA binding -- spiking DG reaches 0.66, clean k-WTA
0.45, neither near-0; and the spiking within-collapse is an implementation artifact (deterministic top-k is
stable). Findings: 2026-05-31-modular-coding-probe-INSTRUMENT-INVALID-...md + survey + DG-boundary banner.
NET: cheap biological VSA-near-orthogonal symbol-grounding is unmet; CONVERGES on night-synthesis P3(c) =
accept the oracle near-orthogonal code as an engineering component + advance the validated P4 retrieval stack.

TRACE BIMODALITY DIAGNOSED 2026-05-31 (finding 2026-05-31-P4-multihop-trace-bimodality-DIAGNOSED-...md):
it is per-pair x per-seed RECALL-STRENGTH (an engram-binding lottery), NOT a filter flaw, NOT seed-global.
Stim the tag, read target rank/32: big->red rank 2(s42)/8(s43,buried)/1(s44) -- mirrors multi-hop 8/0/6.
Other pairs weak on OTHER seeds (hot->dry rank4 s44; cold->wet rank8 s44; s43's hot->dry is rank1 strongest).
Where target falls below trace's top-3, multi-hop misses. Actionable in principle (strengthen weak bindings).
REINFORCEMENT TEST IN FLIGHT (bf2wbr7n7, _perseed_binding_reinforce.py): does re-encoding a weak pair N times
raise its target rank to <=3 (no-code-change fix)? Cases: s43 big->red, s44 cold->wet, s42 big->red (control).

EXACT NEXT CONCRETE ACTION: when the reinforcement test (bf2wbr7n7) lands -> if rank falls to <=3 with more
encodes -> reinforcement is the simple actionable fix for trace bimodality (could wire a "remember a is b"
that reinforces weak bindings, or note it as a usage pattern); record+push. If FLAT (rank stays >3) -> the
per-seed sparse-pattern structure caps the binding -> the deeper balanced-teacher encode (modify
encode_pair_engram_sparse to drive both concepts strongly during the single-pass encode -- a real sparse-
encode change needing re-validation, NOT rushed) OR accept the per-pair lottery as the honest residual.
Either way the multi-hop characterization is COMPLETE (clean->DEGRADES->RESCUE->bimodality DIAGNOSED).
THEN: P4 is well-advanced (160/320 concepts, multitag 90%, directional trace, hierarchy, yes/no, tokenize);
the biological-composition line is boundary-banked. Genuinely-open next directions all need a real design
effort (new biological subsystem mechanism) -- do that via brainstorming->design->cheap-first, check-existing-
first FIRST (theta-gamma + grid arcs were both already-explored). Do NOT unilaterally launch the ~100hr V=640.
moat 7/7;
0.80 bar frozen; cheap-first before spiking; honest negatives/clarifications are the deliverable.
moat 7/7; 0.80 bar frozen; cheap-first before spiking; honest negatives are the deliverable; GPU/CuPy real.


## DG-BIOLOGIZATION CONCLUDED 2026-05-31 = FUNDAMENTAL BOUNDARY; ACTIVE ARC = P4 conversational capability

>>> DG-biologization line CLOSED at a clean fundamental separation-vs-reliability BOUNDARY (finding
2026-05-31-DG-biologization-FUNDAMENTAL-BOUNDARY-...md). The DG separation MECHANISM is confirmed (0.82->0.18)
but no DG SIZE threads separation AND within-concept reliability: 800-sparse separated(0.27)/unstable(0.24);
4000-sparse stable(0.6-0.8)/unseparated(0.66-0.76) -- same competitive-k-WTA tradeoff curve, sweet-spot never
reached; CA3 collapses separation further. The oracle lookup's orthogonality is IRREDUCIBLE on this substrate.
Coherent night deliverable: integrated-loop VOID -> ceiling audit (representational) -> denoiser NEGATIVE ->
3-arc DG convergence -> DG gate PASS -> DG-composition NULL -> this fundamental boundary. Honest biology-
translatable scientific deliverable BANKED. (Controller mis-tuning in one 4000 re-run was caught + corrected;
boundary is the clean tradeoff curve, not an artifact.) <<<

ACTIVE = P4: advance the VALIDATED conversational capability (instant-runnable: g20_sparse_bridges 160-concept
+ g20_sparse_bridges_320 320-concept, multitag retrieval 90% / engram 87.5% / cross-bridge encode / hierarchy /
tokenization). FIRST STEP: confirm the working stack runs (g20_multibridge --sparse / g20_160word_demo), then
advance the highest-value extension toward conversational capability: candidates (a) multi-hop reasoning over
stored associations [known open gap, corrected-NEGATIVE], (b) scale toward 640 concepts [D8 infra scaffolded],
(c) cleaner interactive chat. reuse-by-import; moat 7/7; 0.80 bar frozen; honest negatives are the deliverable.

P4 STEP 1 DONE (clean-condition PASS) + DECISIVE SCRUTINY IN FLIGHT. Multi-hop 2-hop transitive on the
160-concept multitag stack = 8/8 PASS under CLEAN conditions (all-distinct words, anti-cheat 8/8, vs 0.25 prior /
0.094 chance) -- but the EASIEST case (no hub competition at hop-2); mechanism is chained ~100% single-hops via a
shared tag-name middle term, NOT learned inference. DECISIVE hub-reuse+multi-seed scrutiny RUNNING (subagent
a3d6187f2cb233796, _multihop_hubreuse_test.py: hub fan-in 2/4/8 x seeds 42/43/44). ROBUST(>=0.50 at fan-in 8)
-> real multi-hop reasoning capability -> build multi-hop chat demo. DEGRADES-WITH-FANIN -> bounded by hub
crowding (characterize curve). NEGATIVE -> clean 8/8 didn't generalize. [superseded marker for prior in-flight:], _multihop_reasoning_test.py): MULTI-HOP reasoning on the
validated 160-concept g20_multibridge --sparse stack. Encode 2-hop chains (A->B, B->C; A->C NOT directly
encoded), test whether CHAINING the 90% single-hop multitag (query A->B, query B->C) gives reliable transitive
inference. PRE-REG: WORKS if 2-hop transitive >=0.50 (> prior corrected-NEGATIVE 0.25); PARTIAL if >0.25 but
<0.50; NEGATIVE if <=0.25/chance. Controller scrutinizes (A->C genuinely not direct? hop-1 finds B? 2-hop
degradation = hop1*hop2 or worse from drift/loops?). FOLLOW-UP: WORKS -> real multi-hop reasoning capability;
build a multi-hop chat demo + characterize multi-seed. PARTIAL/NEGATIVE -> characterize the chaining limit
(noise compounding / loops) honestly, then next P4 extension (scale-to-640 [D8 infra] or interactive chat).

## (CONCLUDED) DECISION POINT 2026-05-31: three arcs converge on DG PATTERN-SEPARATION

Denoiser arc CONCLUDED = NEGATIVE (finding 2026-05-31-denoiser-arc-NEGATIVE-...-three-arcs-converge-on-
DG-pattern-separation.md). Biologizing shortcut-2 (oracle lookup) via activity grounding FAILS the
{2,3,5} bar: temporal integration denoises VARIANCE (CV ~1.63/sqrt(k) confirmed) but the activity-
grounded symbol is SEPARABILITY-limited (not variance) -- L=3 0.69 / L=5 0.57 plateau below bar at k=32.
The attractor cleanup is CATASTROPHICALLY WORSE (near-chance 0.23-0.26; it needs separable patterns,
collapses on the overlapping activity symbols). Sanity-checked NOT a usage bug (attractor recovers clean
vocab 100% at noise<=0.20). The oracle lookup's irreducible value = the ORTHOGONALITY the substrate
activity lacks. Honest NEGATIVE = the deliverable.

>>> CA3 DIAGNOSTIC DONE = INCONCLUSIVE/CONFOUNDED (CA3 saturated 0.946 active -> within/between 0.90 are artifacts; DG dense 0.37 -> not the separated regime; did NOT cleanly test CA3 on separated codes). Clean DG-side test P5 (larger DG: separation AND stability?) RUNNING (PID 10091, _dg_size_lever_probe, n_dg 800 vs 4000). If larger DG threads sparse-fraction-but-many-active -> stable+separated = resolution; else fundamental boundary (P3). Earlier note kept: CONVERGENCE (the strategic finding): THREE independent arcs now prescribe the SAME missing substrate
mechanism -- DG-style PATTERN SEPARATION:
  - integrated-loop (2026-05-30): wm binding needs stable+lesionable selectivity -> DG pattern-sep.
  - D-arc capacity (2026-05): dedicated-pool geometry erodes -> DG pattern-sep.
  - denoiser (2026-05-31): activity-grounded symbol not separable -> DG pattern-sep (orthogonalize
    before composition).
The project HAS a validated DG (trisynaptic loop, P1 D.12: DG cosine 0.218 from input 0.800, 58pp
orthogonalization). The convergent next arc: insert DG pattern-separation between substrate raw activity
and the composition-symbol derivation, then re-test whether DG-separated activity grounds a composable
symbol. DEEPER arc -> DECISION POINT for owner (this is the third arc to land on DG; it is the strongest-
evidenced direction the project has). DG GATE DONE = PASS (finding 2026-05-31-DG-separation-gate-PASS-...md). The hippocampal DG
ORTHOGONALIZES the overlapping concept activity: pool between-concept 0.806 -> DG 0.296 (sparsity 0.044)
/ 0.169 (sparsity 0.018), bracketing P1's validated 0.218. Multi-seed 42/43/44; genuine trained-substrate
activity (denoise64 caches, baseline 0.82 reproduced); isolation verified (ec at noise floor, no
lang_input); positive control reproduces P1 (0.800->0.218); dg_max 0.59 (no degenerate pairs). Controller
scrutiny: caught an ABANDONED first attempt (untrained pools 0.24 + degenerate silent DG) -- the subagent
independently fixed the SAME two flaws; final verified against JSONs.
LOAD-BEARING CAVEAT (carried to build): separation is SPARSITY-DEPENDENT (k-WTA) -- holds at sparse <=~0.05
(biological; P1 0.007-0.014), degrades if DG driven dense (0.16->DG 0.54, 0.81->0.81). The gate reached
the sparse band by tuning drive/FFi; the BUILD must drive DG into the sparse regime via WIRING, not
hand-tuning -- the build's first risk.

>>> DG ARC GATED-IN. BUILD (next): route concept activity -> DG (sparse k-WTA) -> derive the composition
symbol from the DG-SEPARATED activity -> re-test composition clears the 0.80 bar at {2,3,5} (the bar the
raw-activity symbols FAILED). If YES: oracle lookup biologized via DG pattern-separation = all 3 shortcuts
removable (artificial-life milestone). If NO: DG separation necessary-but-not-sufficient (narrower honest
boundary). Build must (a) wire DG into sparse regime, (b) preserve sparse DG code as the symbol, (c) keep
FHRR composition + moat byte-unchanged. reuse-by-import; no autograd. Cheap-first first: derive symbols
from the gate's DG-separated activity + run mean-of-k + argmax composition (reuse _denoiser_cheap_probe
machinery) before any heavier build. <<<

DG-COMPOSITION DECISIVE TEST DONE = NULL (finding 2026-05-31-DG-composition-NULL-...-needs-CA3-
completion.md). DG-symbol composition WORSE than pool baseline at every load (L2 0.41/L3 0.37/L5 0.33 vs
pool 0.83/0.69/0.58), barely above chance. Mechanism: separation is EXCELLENT (between-concept DG-symbol
cosine 0.18/0.10) but within-concept RELIABILITY collapses -- sparse DG silent on one obs-half for ~1/3-1/2
of words; storage vs query DG of the SAME concept near-disjoint (k-WTA picks different winners) -> unbind
recovers noise. Classic SEPARATION-vs-RELIABILITY tension (gate dose-response: sparse separates/unstable,
dense stable/no-separation; no single DG operating point gives both). 'no-silent' column is a vocab-collapse
artifact (disregarded). DG pattern-separation = NECESSARY-BUT-NOT-SUFFICIENT.

>>> RESOLUTION (biology prescribes it): CA3 PATTERN COMPLETION. The trisynaptic loop is DG->CA3 precisely
because DG separation alone is unstable. CA3 is a recurrent attractor that COMPLETES a sparse/partial DG
pattern to a STABLE stored ensemble -- the within-concept reliability the DG code lacks. P1 validated CA3
completion (D.13, cosine 0.748). Convergent prescription REFINES: not DG-alone but the FULL trisynaptic loop
(DG separates [confirmed 0.82->0.18], CA3 completes/stabilizes). NEXT TEST: drive concept->DG->CA3, TRAIN CA3
ensembles per concept (D.13 direct-CA3: co-fire full pattern + ca3_swr_burst gate to store; recall by
partial/noisy DG drive), derive symbol from the CA3 (completed, stable) code, re-test composition {2,3,5}.
HONEST RISK: D.13 was seed-variable (direct-CA3 passed 0.748; EC-driven FAILED) -> CA3 reliability on the
DG-separated concept activity is uncertain. FIXES -> trisynaptic loop biologizes the oracle lookup
(artificial-life milestone). Cannot-both-separate-and-complete -> deeper honest boundary. reuse-by-import
(builder/validate_trisynaptic_loop D.13 methodology byte-unchanged); no autograd; moat/FHRR byte-unchanged. <<< Standing reframes hold (0.80
bar frozen; moat 7/7; reuse-by-import; no new autograd; honest negatives are the deliverable). <<<

---

## CONCLUDED ARC 2026-05-30 (night): biologize shortcut-2 (activity-grounded symbol DENOISER)

Owner delegated ("whatever you think most productive, keeping goals in mind"). Per the top-level
goal (artificial life / biology-translatable; capabilities instrumental; honest negatives under
strict biology ARE the deliverable) -> chose the BIOLOGY-FAITHFUL path over a scaffold ceiling-break.

CONTEXT (from "check existing sims first" survey + the May-22 findings): phase-coded FHRR
composition is BUILT + VALIDATED -- spiking_phasor_fhrr.py (Orchard, PASS {2,3,5}), resonate_fire_
fhrr.py (Frady-Sommer RF + separated-TPAM cleanup, PASS {2,3,5}), identity-level integration
0.96-0.99 multi-seed. It rests on 3 engineered shortcuts: (1) function-first bind/unbind = BIOLOGIZED
(RF); (3) argmax-over-vocab cleanup = BIOLOGIZED (attractor TPAM); (2) ORACLE LOOKUP (fixed clean
symbol per concept) = STILL ENGINEERED. The May-22 activity-level integration tried to remove
shortcut 2 (derive symbol from real activity) -> NEGATIVE: substrate per-neuron activity CV~1.63
(160% noise); even composition-only collapses to 0.36 (<<0.80). Re-specified: a faithful activity-
grounded symbol needs an ATTRACTOR / TEMPORAL-INTEGRATION DENOISER (CV 1.63 -> ~0.20, the regime
where it composes >0.80). Shortcuts 2+3 COUPLED: a biological attractor grounds AND denoises.

THE ARC: build the denoiser between substrate activity and the FHRR composition layer; reuse the
validated FHRR composition + attractor (TPAM) machinery byte-unchanged; frozen 0.80 bar at loads
{2,3,5}, multi-seed, leakage-guarded.
CHEAP-FIRST GATE (next concrete step, CPU): reuse research/findings/raw/activity_level_integration.py
(captures 3200-dim per-neuron concept-pool activity; measured CV 1.63; composes via byte-unchanged
spiking_phasor_fhrr) -- INSERT a denoiser (temporal integration over k observations: CV~1.63/sqrt(k),
k~66 -> 0.20; AND/OR an attractor settle like the validated TPAM) BEFORE symbol derivation; measure
(a) post-denoiser CV, (b) composition accuracy vs 0.80. If any denoiser gets composition >0.80 (or CV
near 0.20) -> the denoiser arc is VIABLE -> design + build properly. If NONE -> honest NEGATIVE (the
substrate is irreducibly noisy for single-pass activity grounding; the oracle lookup is irreducible
on this substrate) = a biology-translatable deliverable. Cheap-first BEFORE designing big (the
falsify-cheaply discipline). resonate_fire_fhrr.ResonateFireTPAM is the reusable attractor denoiser.
Standing: reuse-by-import; no new autograd; no protected/frozen/moat edit; moat 7/7; 0.80 bar frozen.

CHEAP-FIRST GATE DONE = VIABLE (finding doc 2026-05-30-denoiser-cheap-first-VIABLE-temporal-
integration-denoises-activity-grounded-symbol-CV-falls-as-1-over-sqrt-k.md). k-curve (3-seed,
comp-only): k=1 0.34/0.36/0.41 (reproduces NEGATIVE baseline); k=8 L=2 0.849 PASS; k=16 L=2 0.936
PASS, L=3 0.802 PASS, L=5 0.659 (rising, extrapolates ~0.80 at k~32-48). CV falls ALMOST EXACTLY as
1.63/sqrt(k) (1.518/1.079/0.787/0.552/0.395 vs 1.63/1.15/0.82/0.58/0.41) => the substrate noise is
INDEPENDENT across observations (averageable), NOT correlated. So TEMPORAL INTEGRATION (sustained
encoding) genuinely denoises the activity-grounded symbol; the oracle-lookup shortcut IS biologizable;
required k grows with load. HONEST CAVEAT: 16 cached obs -> bootstrap-overlap may make exact k modestly
optimistic (CV law is overlap-independent so viability is robust; exact k needs more obs).

>>> CORRECTED 2026-05-31: the cheap-first 16-obs "VIABLE" was OPTIMISTIC. The rigorous 64-obs DISTINCT
confirmation (NO substrate confound -- RECOG_CACHE=phase1_800ev constant, both captures used it) shows
temporal integration ALONE is INSUFFICIENT for L>=3. <<<

64-OBS RESULT (distinct, k up to 32; finding doc CORRECTED + banner): CV still falls EXACTLY as
1.63/sqrt(k) (variance-reduction mechanism real) BUT composition PLATEAUS below 0.80 for higher loads:
  k=32 (CV 0.294): L=2 0.834 PASS (only at large k); L=3 0.694; L=5 0.575 (both BELOW bar, plateauing).
The 16-obs cheap-first inflated via vocab/storage observation overlap (16 obs -> cleanup-target vocab
shares obs with storage symbols). HONEST: temporal integration is a real VARIANCE denoiser but the
activity-derived symbol has a residual QUALITY/SEPARABILITY limit (not variance) that averaging cannot
fix -> at higher load, inter-concept crosstalk dominates. BOUNDARY for temporal-integration-ALONE
(L=2 only). NO confound (verified RECOG_CACHE constant).

KEY: the probe used a SIMPLE argmax cleanup, NOT the biological attractor. The May-22 'shortcuts 2+3
coupled' insight = an attractor GROUNDS + DENOISES + its fixed points are clean/separable. So the
residual is exactly what the attractor cleanup should fix.

>>> NEXT = CAPSTONE (well-motivated): temporal-integration denoiser + ResonateFireTPAM ATTRACTOR cleanup
(cleanup_separated), end-to-end on the 64-obs activity-grounded symbols, validate 0.80 bar {2,3,5}.
Does the attractor's recurrent settling lift L=3/L=5 above 0.80 where simple argmax couldn't? If YES ->
activity-grounded symbol biologizable WITH the coupled attractor (all 3 shortcuts removed). If NO ->
activity grounding is fundamentally separability-limited on this substrate (honest biology-translatable
boundary). Build: reuse 64-obs cache denoise64_seed{N}.npz + mean-of-k + resonate_fire_fhrr.
ResonateFireFHRR composition + ResonateFireTPAM.cleanup_separated (read its self-test for the validated
theta_low/high/n_anneal/abstain_threshold). RF + TPAM are time-stepped (slow) -> modest trials, can use
cleanup_separated fast= path if needed. reuse-by-import; spiking_phasor_fhrr / resonate_fire_fhrr / moat
byte-unchanged; no autograd. <<<


P4 PIVOT IS READY (instant, no training): the validated G.20 multitag conversational stack's bridges EXIST on disk -- g20_sparse_bridges/bridge{A-E}_*_sparse.simstate.h5 (160 concepts) + g20_sparse_bridges_320/*_sparse64.simstate.h5 (320 concepts). Runnable now via g20_160word_demo / g20_multibridge --sparse (cross-bridge encode + multitag retrieve + hierarchy + tokenization). DECISION LOGIC: the corrected 4000-low-drive DG test (PID 10146, watcher blw3vj8hv) is the DG-line DECIDER. RESOLVE (4000 reaches sparsity ~0.05 with WITHIN>>0.235 & BETWEEN<=0.5) -> continue trisynaptic line (CA3 next, carefully tuned). CANNOT-REACH-SPARSE or UNSTABLE -> the DG-biologization separated+stable compositional symbol is BLOCKED by a tuning-sensitive separation-reliability tension across systematic attempts (mechanism confirmed, assembly unreached) = honest biology-translatable BOUNDARY, BANK it, PIVOT to P4 (advance the working stack: candidate extensions = multi-hop reasoning [known gap], scale-to-640 [D8 infra scaffolded], or interactive chat). worth-GPU-time frame favors P4 if DG boundaries.
Re-run 64-obs (kill-safe): python -u -m research.findings.raw._denoiser_cheap_probe --capture-obs 64
--distinct --k-list 4 8 16 24 32 (GPU/CuPy for capture).

---

## CONCLUDED ARC 2026-05-30 (PM): conversational-ceiling AUDIT (owner chose "audit the ceiling")

Integrated-loop wm-emergence arc CONCLUDED (two-horns VOID, below). Owner picked A=pivot,
then for the next arc chose "audit the ceiling" over building phase-coded VSA. Survey found
the conversational line already ran 8+ decisive arcs (theta-gamma cue-supp / gentle-replay
6th-arc local-optimum 0.458 / SPEAR phase-multiplex 0.00 / Pirazzini / generative-replay /
staged-recurrence) all NEGATIVE/VOID, framed as a REPRESENTATION ceiling prescribing
phase-coded VSA (Orchard spiking-phasor FHRR; resonate_fire_fhrr.py exists). Audit verifies
that premise before the big build.

PHASE 1 DONE (no GPU, code-read; finding doc 2026-05-30-ceiling-audit-phase1-headline-numbers-
conflate-pipelines-composition-IS-decodable-at-0.46.md):
  - SPEAR full_acc=0.00 gated on RAW FIRING RATE @650 moat (spear...runner.py:515-528). The
    SPEAR units-bug hypothesis (cosine ranked @650 -> trivial abstain) is FALSIFIED (readout
    is genuinely firing-rate scale).
  - BUT headline numbers CONFLATE pipelines: 6th/8th-arc full_acc (0.458/0.315) is lang_output
    COSINE gated @ COMPOSITIONAL_UNIFIED_THRESHOLD=0.1977 (cosine scale); SPEAR is firing-rate
    sum @650. Non-comparable -> "8 arcs converge on ~0.46/0.00 ceiling" is a loose framing.
  - Composition IS decodably represented at ~0.46 TRUSTWORTHY gated emission (6th-arc cosine,
    calibrated gate, margins 0.064-0.118) -> the "composition not a structured decodable
    object / phase-coded VSA needed to make it representable AT ALL" framing is OVERSTATED.
  - Honest: the LITERAL pre-registered artifact (a)="raw vs gated" was NOT met (both gated);
    reported the related conflation + ceiling-reframe instead. Does NOT dissolve the ceiling
    (~0.46 is a real cap below 0.80).

PHASE 2 IN FLIGHT (owner said "ok" -> proceed): decisive latent-composition decode probe.
Subagent a3a208e2fea58fb08 (background) builds a throwaway probe (research/findings/raw/
_ceiling_audit_phase2_decode.py) reusing the EXACT 6th-arc machinery: generative_replay_pfc_
frame_runner.py FULL arm = unified_per_regime_monitor_runner._build_bridge_with_phase1_recipe
+ _encode_facts + _unified_compositional_pairs + _compositional_query_ranked + consolidation_
trainer.run_concept_replay_phase + PFC-frame priming (the ~0.46 regime). Captures the composed
lang_output state + cosine-readout decision + true answer per query (>=200 instances, several
seeds); trains a HELD-OUT linear (sklearn LogisticRegression) + NN decoder with EPISODE-LEVEL
group k-fold (train/test never share an episode -> no leakage); compares decoder held-out acc
(B) vs cosine readout acc (A) on identical test sets.
PRE-REGISTERED (frozen): READOUT-LIMIT if B >= 2x A (and >> chance); REPRESENTATIONAL-CEILING
if no decoder beats A by >= +0.10; INCONCLUSIVE else. Decoder is ANALYSIS-ONLY (sklearn/numpy
CPU linear probe; NOT a sim learning rule, NOT autograd). No protected/frozen/moat/sim edit
(throwaway script only). CONTROLLER forms the official verdict + scrutinizes (a READOUT-LIMIT
result is the surprising/strong claim -> scrutinize it harder than a FAIL: episode-level split
real? regime check ~0.46? chance baseline? class balance?).
PHASE 2 ATTEMPT 1 INVALID (subagent a3a208e2 ended early w/o a valid result; controller
diagnosed): the probe script is STRUCTURALLY SOUND (episode-level GroupKFold no-leakage, diff
pairs/episode, primary lang_output + secondary pool states, pre-reg verdict) BUT FAILS THE
REGIME CHECK -- on the captured data the cosine readout top1 = 0.0 (0/8), only 2/8 predictions
are even adjectives (predicts "dog"/"go" for "small" answers). primary_state dim = 2048 => it
ran on the FULL validated substrate (loaded unified_per_regime/phase1/seed42.simstate.h5, 27MB,
EXISTS), so this is a PIPELINE-REPRODUCTION bug, NOT a substrate-scale issue: the probe does not
reproduce generative_replay_pfc_frame_runner's FULL arm (the ~0.46 regime). Per pre-registration
(STOP if regime wildly off) the decode comparison is INVALID -> NOT trusted.
FIX SUBAGENT (a2649185) STEP A + B DONE; STEP C decisive run IN FLIGHT:
  STEP A DONE: the REAL generative_replay_pfc_frame_runner REPRODUCES -- full_acc 0.40 (seed42 N2),
    0.4583 (3-seed N3). The 0.46 regime is REAL, NOT a regression.
  STEP B DONE (probe fixed to RUNNER-BLEND regime 0.4545, in band) + KEY REFINEMENT FINDING: the
    6th-arc "full_acc 0.46" is a BLEND of DIRECT-retrieval queries (easy, high acc) + COMPOSITIONAL
    queries (hard). The COMPOSITIONAL-cosine readout ALONE is ~0.0-0.30, NOT 0.46. This SHARPENS
    Phase 1: composition-only is decodable at ~0-0.30 (lower than the blended 0.46 implied); the
    Phase-1 doc's "composition decodable at ~0.46" should be read as the BLENDED number, not the
    compositional-only number. (Pending exact A from STEP C.)
  STEP C DECISIVE DONE = REPRESENTATIONAL-CEILING-CONFIRMED (conclusion doc
    2026-05-30-ceiling-audit-CONCLUSION-representational-confirmed-0.46-was-a-blend-VSA-warranted.md).
    3 seeds x 24 episodes, 120 instances, chance 0.25: compositional cosine A=0.24 (answer-subspace)
    / 0.04 (full-vocab); held-out linear B=0.21, NN 0.204, B_best 0.21; secondary pool-firing decoder
    0.218/0.208. ALL ~chance. Verdict REPRESENTATIONAL (B_best 0.21 < A 0.24 + 0.10). Scrutiny passed:
    episode-level no-leakage (decoders at chance not above); secondary state 16-dim WELL-sampled (not
    underdetermined) yet still chance -> composition genuinely not decodable from lang_output OR pool
    firing. Only throwaway probe changed; protected/runner/sim/compose untouched (verified).
  HONEST SELF-CORRECTION: Phase-1's "composition decodable at ~0.46" over-read the BLEND; compositional-
    only is ~chance & not decodable by any held-out decoder. Phase-1 doc banner-corrected.

>>> AUDIT CONCLUDED. The conversational-composition ceiling is REPRESENTATIONAL, not readout-limited.
The phase-coded vector-symbolic (Orchard spiking-phasor FHRR) arc is WARRANTED by a VERIFIED premise.
The audit gate PASSES. <<<

NEXT (owner decision point, surfaced): proceed to the phase-coded VSA arc DESIGN (brainstorm ->
design doc -> cheap-first probe gate -> spiking build under frozen-verdict discipline). The rhythm
must CARRY composition as spike phase so the composed state is a STRUCTURED DECODABLE object (the
exact thing this audit proved is missing). resonate_fire_fhrr.py exists as the spiking-phasor
primitive (reuse-by-import seed); check Orchard 2023/24 + Frady-Sommer resonator networks. Big new
arc -> recommend a proper design pass, NOT a reflexive build; cheap-first probe MUST gate the spiking
build. DEFAULT (no steer): begin the VSA arc brainstorm/design (check existing sims first). Standing
reframes hold (biology-grounded conflict-resolution; 0.80 bar frozen; moat 7/7; reuse-by-import; no
new autograd; honest negatives are the deliverable).

---

## CONCLUDED ARC 2026-05-30: phase-factored integrated closed-loop (goal pivot after closing the D-arc)

D-arc CLOSED at pillar n=110 (cross-bridge FHRR-scaffold capacity track done;
synthesis bbbf98f). Pivoted to the goal-aligned work: the phase-factored
integrated loop — composition as emergent from online theta-ordered episodic
encode + offline shuffled-replay consolidation (resolves the encode-order
conflict the parked Q5 loop stalled on, 2026-05-19). Design 45fe0a7; plan
c1e79b7 (subagent-driven, Task 6 controller-only).

Progress:
- Task 0 grounding pin: DONE (6bee885; 4 pass / 5 skip).
- Task 1 cheap-first falsification probe: DONE (19ef6f1 + strengthen 23ae76e).
  Controller caught the first version was CIRCULAR; strengthened the
  residual-coupling to a GENUINE measurement. Verdict RESOLVES (gate met;
  4e1d10f) but coupling_demonstrated FALSE at toy scale.
  SUBSTRATE CAVEAT (carry forward): toy uses near-orthogonal reps (tiny
  common-mode); the D-arc MEASURED real reps have LARGE common-mode
  (0.18-0.68), so consolidation moves real reps substantially and the
  residual-coupling could be real. The spiking build MUST test index
  survival on real reps + keep the consolidation-updates-index path as
  insurance.

UPDATE 2026-05-30 (Tasks 0-5 DONE; decisive run BLOCKED on instrument soundness):
- Tasks 0-5 complete: grounding pin; cheap-first probe RESOLVES (caught circular,
  strengthened; commit 4e1d10f); two-phase controller built (29 tests; reviewer
  CLEAR Task 4); verdict-reuse pin; no-harm (protected byte-empty, moat 7/7).
- Task 6 decisive orchestrator built (KILL-SAFE per-load; commit 3b04148).
- FULL-SCALE GROUNDING PROBE (32 min, N=2 seed42) = INSTRUMENT UNSOUND (commit
  e8cf685 + diagnosis 45e02d9). v1 wm=0.5 < 0.90 bar (decisive run would VOID).
  Problem 1 (BLOCKER): wm readout is a role->filler BINDING query, near-chance,
  no_bg_gate doesn't collapse it. Problem 2: ep collapses for the genuine task
  (full ep=0.0 vs v1 ep=1.0) = substrate caveat materialized.
- The grounding probe did its job: 32-min catch vs ~8-hr VOID. Decisive run NOT
  launched.
DIAGNOSIS COMPLETE (raw-counts sink, commit 42c851a): the wm failure is
NON-SELECTIVE RETRIEVAL, not gate calibration. Driving a role lights up ALL 8
filler pools ~equally; correct filler top only 1/8 -> chance. PARTIAL SUCCESS:
v1 ep=1.0 -- the two-phase ORDER readout WORKS.

UPDATE 2026-05-30 (decisive ITERATION 1: phase-restructure fix tried; EP DECOUPLING
VALIDATED, WM blocked by a SUBSTRATE-LEVEL cause; findings a54f2a9, fix 06b13c1):
The fix the prior NEXT-ACTION prescribed (move selectivity training to its correct
phase) was implemented faithfully: Phase 1 now FREEZES the selectivity gates so the
in-order pass writes only the ORDER INDEX (engram + theta-gamma slot order); Phase 2
runs the validated v16 SHUFFLED teacher co-fire + STDP to build selectivity, plus the
SWR consolidation. Full-scale v1 re-probe (seed 42, N=2): V1_EP=1.000, V1_WM=0.000.
  - VALIDATED: ep stayed 1.0 through the restructure -> the two-phase ENCODE-ORDER
    DECOUPLING is real (order written online, untouched by offline selectivity). Half
    the two-phase thesis confirmed.
  - NEGATIVE (characterized): wm still non-selective. Checkpoint probes show the
    topographic PRIOR alone gives clean 2/2 role->filler selectivity, but repeated
    selectivity-STDP over epochs ERODES the prior's margin (unbound fillers creep up,
    overtake by ~ep6) in EVERY variant (phase placement, role-only co-fire, off-target
    suppression, SWR on/off). Root cause = STDP-selectivity INSTABILITY on this
    substrate. The deep tension: the STABLE selectivity source (prior) is
    lesion-INVARIANT (can't satisfy lesion-collapse); the lesion-ABLATABLE source
    (STDP) is UNSTABLE. No mechanism here is both. Same substrate theme as the D-arc
    geometry erosion. This is the binding-retrieval problem localized to representation
    stability -- NOT a phase-placement bug (which is now fixed).

EXACT NEXT ACTION (DECISION POINT -- surfaced to user 2026-05-30): the wm instrument
cannot be made sound by controller-side wiring; it needs a selectivity CARRIER that is
both stable and lesion-ablatable. This is a deeper redesign, not a tweak. Two
biology-grounded candidates, both genuine next steps:
  (A) [RECOMMENDED] Hippocampal DG pattern-separation as the selectivity carrier (DG
      orthogonalizes reps AND is a lesionable subsystem -> satisfies lesion-collapse) --
      replaces STDP-on-cortex as the selectivity source. Where the D-arc independently
      pointed. Largely BUILT + P1-validated (trisynaptic loop: D.12 separation 3/3,
      D.13 completion). Function-matched to the failure (DG's job IS clean separation).
      Still a multi-day redesign: wire DG as the wm selectivity carrier, re-validate
      the instrument (v1 wm>=0.90), re-review, then decisive run.
  (B) [DE-RISKED OUT as cheap reuse] homeostatic stabilization. Cheapest-first probe
      (this turn) found the bridge's existing homeostasis (enable_synaptic_scaling,
      Turrigiano 2008, bridge.py:5797-5827) is postsynaptic-RATE-HOMOGENIZING (pulls
      every filler pool toward one target rate) -- that works AGAINST selectivity
      (which needs bound-filler high / unbound low). A genuine stabilizer here is
      divisive normalization across the filler population, which is a NEW learning rule
      (conflicts with reuse-by-import / no-new-rule discipline). So B is NOT a clean
      cheap reuse; A is the better-grounded path.
Both reuse-by-import only (no new autograd; no protected/frozen/moat edits). Decisive
multi-seed run STAYS UNLAUNCHED (v1 wm < 0.90, correctly). The frozen verdict + the
two-phase controller + the grounding discipline all held; the ep result is a real
partial win banked. FORK SURFACED TO OWNER 2026-05-30: (A) commit to the multi-day DG
selectivity-carrier redesign [recommended, goal-aligned: lifts the substrate ceiling
that blocks BOTH this arc and the D-arc]; OR park the integrated-loop instrument here
as an honest characterized NEGATIVE + banked ep win, and pick the next goal-aligned
arc.
OWNER STEER 2026-05-30: "Proceed as suggested" -> path A (DG selectivity carrier).

DE-RISK DONE 2026-05-30 = GO (commit b809ac7+ this update). Read confirmed path A is
wireable by reuse-by-import: (1) the built bridge ALREADY has DG/trisynaptic regions
(_build_bridge -> build_biological_brain_regions(enable_hippocampus_consolidation=True),
integrated_loop_gate.py:762 / phase_factored imports it). (2) The engram API
(start/commit/stimulate/clear/delete_engram_tag) is a dict keyed by name
(bridge.py:2483) -> MULTIPLE concurrent per-binding recordings supported, region-filtered
commits, tag stimulation -- all existing blessed-reuse methods. (3) The ep readout ALREADY
proves the path works: _episodic_order_readout (phase_factored_loop_gate.py:401-429)
stimulate_tag -> DG-separated CA3 completion -> role-pool peaks, ep=1.0. The wm readout
(606-649) currently does NOT use it -- it drives a bare role code + relies on the eroding
cortical dlpfc_verb->filler STDP selectivity. The fix: route wm role->filler retrieval
through the SAME DG/engram path.

EXACT NEXT ACTION = BUILD path A (controller file phase_factored_loop_gate.py ONLY; no
new rule/autograd; no protected/frozen/moat edit; engram API reuse byte-unchanged):
  1. At encode (Phase 1 loop ~269-356), commit a PER-BINDING engram tag per (role,filler)
     -- start_engram_recording("pf_ep%d_bind%d") around each binding's BG-gated co-fire,
     commit_engram_tag(region_filter capturing the role pool + filler pool + ca3). Keep
     the whole-episode tag (ep needs it).
  2. Rework the wm readout (606-649) to retrieve via the engram path: at a role query,
     stim the queried role's per-binding tag(s) (multitag stim-recall variant FIRST --
     87.5%/90% validated, the higher-reliability path; CA3 partial-cue completion as
     fallback) and rank filler pools by reactivation. Keep the DEFAULT_THRESHOLD gate +
     the _wm_raw passive sink.
  3. PRESERVE the 7-lesion partition: no_hippo_store/no_binding (SHARED) remove tag/
     assembly -> wm collapses WITH ep; no_bg_gate (HELPER_WM) degrades the per-binding
     gated co-fire -> wm collapses, ep survives. Map each lesion in comments; re-run the
     tiny-synth lesion probe to confirm.
  4. GATE: v1 full-scale probe (~3.5 min): require v1 wm>=0.90 AND v1 ep>=0.90. Iterate
     (multitag -> CA3-completion). If v1 sound -> re-review (readout changed post-Task-4)
     -> decisive run (controller-only, smell-test). If neither engram variant makes v1
     wm sound -> honest NEGATIVE (DG/engram per-binding retrieval also can't make the wm
     instrument sound at this scale) -> park.
Decisive multi-seed run stays UNLAUNCHED until v1 wm>=0.90.

SUBAGENT DONE + COMMITTED 2026-05-30 (cb6834b, pushed both remotes; HONEST partial-
verification status). The engram-based wm retrieval is built (per-binding tags at encode
+ stimulate_tag/CA3-completion retrieval, multitag). CONTROLLER VERIFICATION STATUS:
  R2 (engram path real, not relabeled STDP): CLEAR (from diff).
  R4 (rng faithful: deterministic sorted tag order, _make_pairs sole shared-rng consumer;
     integrated_loop_core.py byte-empty): CLEAR.
  Tests: 71/71 phase-factored+engram+moat PASS (controller re-ran, CPU). New test is a
     real behavioral spy on engram-API calls, not an impl echo.
  Tiny-synth partition: ep-side CORRECT (no_hippo_store ep=0, rest ep=1); wm-side FLOORED
     at 0 for ALL modes (scale artifact -> tiny-synth CANNOT validate the wm partition;
     R3 REQUIRES full scale).
  R1 (v1 raw-count selectivity smell-test) + R3 (wm-side 7-lesion partition at FULL
     scale): IN FLIGHT. Subagent's full-ladder probe (PID 6256, research.findings.raw.
     _pf_full_ladder_probe) is STILL RUNNING on GPU; watcher bbw6or6pg waits for it +
     dumps the table. After it lands: read the table for R3, then run _pf_v1_probe.py
     (~3.5 min, has the _WM_RAW_SINK) for R1 raw counts. Subagent CLAIMS v1 wm=1.0 ep=1.0,
     bound filler out-firing distractors ~13-50x -- MUST be controller-confirmed (scrutinize
     a PASS harder than a FAIL).
R3 DONE 2026-05-30 = VOID (full-scale partition table, N=2 seed42; finding doc
2026-05-30-phase-factored-decisive-iteration2-engram-wm-SOUND-but-VOID-two-horns-characterized.md):
  v1=(1.0,1.0) SOUND; full=(0.5,1.0); no_binding=(0.5,1.0); no_shared_clock=(0.5,1.0);
  no_hippo_store=(0.0,0.0); no_bg_gate=(0.5,1.0); no_sequencing=(0.5,1.0);
  no_cls_replay=(0.5,1.0); no_neuromod_timing=(0.5,0.0).
  The frozen verdict VOIDs at the discrimination check: no_binding (SHARED) must drop BOTH
  <=0.40 but wm=0.5 -> "not emergent-from-integration / wiring artifact" (and no_bg_gate /
  no_sequencing / no_cls_replay independently fail their checks). wm is FLAT 0.5 for full +
  6/7 lesions, 0.0 only under no_hippo_store -> the per-binding engram tag is a LOCALIZED
  hippocampal-store LOOKUP, lesion-invariant except removing the store. Drilled query passes,
  novel-recombination query fails (=0.5); v1 scores only the drilled query (=1.0).

>>> DO NOT LAUNCH THE DECISIVE RUN. <<< It would VOID identically at every seed (the
discrimination failure is STRUCTURAL -- the engram store is lesion-invariant by construction,
not stochastic). Running ~8 hr to reconfirm a structural VOID violates the grounding discipline.

TWO HORNS NOW CHARACTERIZED (both VOID-certified by the pre-registered verdict, OPPOSITE reasons):
  iter 1 STDP selectivity: EMERGENT but UNSTABLE -> VOID (unsound, v1 wm<0.90).
  iter 2 DG/engram store:   STABLE but NOT EMERGENT -> VOID (non-discriminating).
No mechanism here is both stable-enough-for-soundness AND emergent-enough-for-the-partition.
The integrated-loop wm-emergence thesis (role-filler binding retrieval) is NOT supported on
this substrate. BANKED + unaffected: ep-decoupling validated (ep=1.0 both iters); engram v1
soundness is genuinely SELECTIVE (R1 probe buzja4s2j confirming true filler >> distractors).

R1 DONE 2026-05-30: v1 retrieval DECISIVELY SELECTIVE at full scale (true filler 15x-400x over
distractors; cleanest queries true ~6000-6900 vs distractors <=15). The engram store is a
reliable, sharply selective role-filler memory -- the v1 soundness PASS is real (scrutinized
harder than a FAIL, holds up massively). Finding doc finalized + committed (7f662e0 + R1
sharpen). ARC iteration-2 fully recorded.

>>> INTEGRATED-LOOP wm-EMERGENCE ARC CONCLUDED (honest VOID, two horns characterized). <<<
Next per DEFAULT A: pivot to the next GOAL-ALIGNED arc (conversational capability / artificial
life), banking (i) ep-decoupling validated, (ii) engram = reliable selective role-filler memory,
(iii) the two-horns substrate finding. Do NOT auto-start path B (loop-gated readout = deeper
redesign, reopens instability) without explicit owner steer. When picking the next arc, honor
the standing reframes: check existing biology-grounded sims FIRST; build conversation on the
biological conflict-resolution mechanisms (SPEAR theta-multiplexing / theta-gamma / generative
replay), NOT static retrieval/RAG; bug-discovery-first on chance results; 0.80 multi-seed bar
frozen; moat 7/7 never weakened.

DECISION POINT (surfaced to owner; recommend A): (A) PARK the integrated-loop wm-emergence
thesis as a characterized VOID + bank the ep win, pivot to the next goal-aligned arc
(conversational / artificial-life). (B) attempt a THIRD mechanism -- loop-GATED engram readout
(retrieval depends on BG gate + shared clock + binding so lesions collapse wm) -- deeper
redesign, re-opens the instability risk (walks back toward horn 1), real risk of re-VOID.
DEFAULT (no owner steer): A -- record the two-horns VOID as the arc's honest conclusion + pick
the next goal-aligned arc; do NOT auto-start B (a deeper redesign warrants an explicit steer).

PRE-STAGED RE-REVIEW (R1/R3 detail; the wm readout changed post-Task-4):
  R1 [SELECTIVITY SMELL-TEST -- the load-bearing one]: scrutinize the _wm_raw raw filler
     counts on the scored v1 queries. A real PASS = the TRUE filler fires HIGH and the
     other 7 fire LOW (selective retrieval). A FALSE pass = all 8 fillers fire ~equally
     and the gated top is correct only by luck / because v1's drilled query trivially
     matches. Scrutinize a PASS HARDER than a FAIL. If v1 wm>=0.90 but raw counts show
     non-selective firing, it is NOT sound -> treat as NO-GO.
  R2: the engram path is REAL (per-binding tags + stimulate_tag), not a relabeled
     re-introduction of the eroding cortical dlpfc_verb->filler STDP.
  R3: 7-lesion partition holds at FULL scale (not just tiny-synth): SHARED
     (no_binding/no_shared_clock/no_hippo_store) collapse BOTH; HELPER_WM (no_bg_gate)
     collapses wm not ep; HELPER_EP (no_sequencing/no_cls_replay) collapses ep not wm --
     and each collapses for the RIGHT mechanistic reason via the new engram path.
  R4: RNG faithfulness (_make_pairs SOLE shared-rng consumer; any new tag-stim-order rng
     is a dedicated cross-mode-identical local rng); integrated_loop_core.py byte-empty;
     4 validated subsystems byte-unchanged; no new rule/autograd; moat 7/7; ep still 1.0.
Only if the re-review is CLEAR on all four -> re-confirm decisive cache empty
(research/findings/raw/phase_factored_decisive_cache/ -- verified empty 2026-05-30) ->
run the controller-only decisive multi-seed (phase_factored_decisive.py, seeds 42/43/44,
ladder N=2/4/8) -> mandatory smell-test on the recorded JSON -> honest propagation both
remotes. If NO-GO at v1 -> honest NEGATIVE finding (DG/engram per-binding retrieval also
cannot make the wm instrument sound at this scale) -> park the integrated-loop instrument
+ bank the ep-decoupling win + surface the next goal-aligned arc.
(D8 smoke killed 2026-05-30; marginal post-closure.)

OLD NEXT ACTION (superseded): Task 2 — build the two-phase controller +
order-preserving index readout in the spiking bridge
(research/runners/phase_factored_loop_gate.py), reusing 4 validated
subsystems byte-unchanged (engram-tag API, consolidation_trainer /
Phase-1.3 SWR replay, concept_pool_demo v16 binding, abstention_gate) +
the parked theta-gamma controller (integrated_loop_gate.py). Expose
run_rung(N, seed) emitting the rung shape integrated_loop_core.
integrated_loop_verdict consumes. Tiny-synth CPU-testable; heavy GPU run
is Task 6 (controller-only). Then Task 3 verdict-reuse, Task 4 adversarial
review BEFORE the decisive run, Task 5 no-harm, Task 6 controller-only
decisive multi-seed run.

GPU note: D8 speedup smoke (methodology validation) still running in
background; Task 2 is CPU-buildable so it does not wait.


## [HISTORY ARCHIVED 2026-05-31] older arcs moved to AUTONOMOUS_STATE_ARCHIVE.md

To keep THIS file under the 256KB Read limit (the local watchdog Reads it every
cycle; a >256KB Read errors), the 2026-05-21..05-27 history was moved verbatim to
`research/findings/AUTONOMOUS_STATE_ARCHIVE.md` (zero loss). Archived content:
D-arc Direction Q/P/3/4/R; capability pillars n=105..n=110; D6 V=160 + D7 V=320
sparse-distributed validation; the integrated-loop necessity-instrument 5-route
terminal line; the 2026-05-21 cumulative-deliverable + multiple "preserved state"
blocks; the 2026-05-27 WDDM perf finding (multi-process parallelism is a no-op on
Windows). Full per-arc detail also lives in the dated `research/findings/*.md` docs
+ `INDEX.md`. The LIVE pointer is at the TOP of this file; live-reference (frozen
bars + watchdog guarantee + crash-recovery lesson) is immediately below.

## Pre-registered acceptance / frozen bars (NEVER tuned)

`integrated_loop_core.py` `_IL_*`: V1_MIN 0.90, SCI_MIN 0.80,
LESION_MAX 0.40, SCALE_TOL 0.10, ladder (2,4,8), MIN_SEEDS 3. No-confab
moat `research/runners/abstention_gate.py` + test 7/7 byte-identical.
GPU (CuPy) for every real/decisive run; numpy only for `--tiny-synth`.

## Continuation guarantee (TWO watchdogs — installed)

1. **LOCAL, GPU-capable** — Windows Scheduled Task `SimAutonomousWatchdog`
   runs `scripts/autonomous_watchdog.ps1` every 20 min. Conservative
   stall-gate: fires ONLY if no git commit for >40 min AND no active
   claude/python-sim process AND no fresh `.watchdog.lock`. On stall it
   re-invokes local `claude.exe -p` (bypassPermissions, `--add-dir` repo)
   with a prompt to read THIS file and continue the exact next action
   INCLUDING PENDING-LOCAL-GPU steps. Audit log:
   `research/findings/raw/autonomous_watchdog.log`. This is the primary
   guarantee for GPU-bound work. (Re-verify: `schtasks /query /tn
   SimAutonomousWatchdog`; re-register via `scripts/autonomous_watchdog.ps1`
   contract if missing.)
2. **REMOTE claude.ai routine -- DISABLED 2026-05-20 (owner
   correction: budget consumed; do NOT re-enable or replace).**
   The prior `sim-autonomous-continuation-watchdog` routine
   (`trig_01W7vwnpv4JYWUMjzwHaEKK6`) was disabled by `RemoteTrigger`
   update `enabled: false` after consuming the routine budget.
   Continuity going forward is the LOCAL Windows Scheduled Task
   ONLY. Do NOT create/enable/replace this routine. See
   `memory/feedback_no_claude_routines_for_continuity.md`.

If the local watchdog is missing, RE-CREATE it before other work.
The local watchdog is a fallback for genuine session death --
it is NOT a justification to stop early in a working session.
The in-session discipline NEVER stops on a promise: the next
concrete tool call is always in the same turn after every commit;
ending a turn with "AUTONOMOUS_STATE points the next session at X"
is itself the promise-stall pattern the discipline forbids
(owner-corrected 2026-05-20).

## CRASH-RECOVERY (2026-05-28 ~17:32 EDT): D7 production died with Claude crash; relaunched with proper detachment

Claude desktop crashed during autoupdate (took D7 production PID 30216 with it -- I had wrongly claimed earlier it was "detached," but `-NoNewWindow` kept it attached to the harness's console process group, so the console tree got reaped). KILL-SAFE caches saved 12/15 cells (A_nouns, B_verbs, C_adj, D_spatial all 3 seeds; E_functional/seed42 was at 10%, lost the ~21 min partial).

Relaunched at 17:32:57 as PID 26928 with `Start-Process -WindowStyle Hidden` (creates a separate process group with its own console, no shared console with the harness -> survives client crashes). Cache-skip verified working (each of 12 cached cells "completes" in ~1.1 min by just loading the bridge + activity npz). Will then train E_functional seeds 42/43/44 (~225 min each), then run cross-bridge probe inline. ETA ~05-30 05:00 EDT.

LESSON for future launches: Use `-WindowStyle Hidden` (or omit `-NoNewWindow`) so the python process gets its own console + survives client death. `-NoNewWindow` is convenient for live log piping but makes the process die with the harness. The KILL-SAFE per-cell caches did their job here -- they're the load-bearing recovery mechanism, not the detachment.

