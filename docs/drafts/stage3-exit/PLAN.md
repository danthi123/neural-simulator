# Stage-3 (ADOLESCENT) exit gate — SYNTHESIZED DRAFT DESIGN (Claude build, 2026-10-13)

**Status: DRAFT. Nothing here is registered.** The preregistration text in §12 becomes binding only when Claude commits
it alone as `research/findings/<date>-stage3-exit-gate-PREREGISTRATION.md` (due 11-14). Gate seeds 42/43/44/100/101/102
are refused by the runner until then (`PREREG_REGISTERED = False`). Code facts checked against s/main `1187f7bee` and
s/research/stage2-exit-gate `4b19aafb8`; the derived-marking fix is `claude-build/derived-terminal-honesty @20614ef7d`
(NOT on s/main tonight; REQUIRED in the pin, ruling 3).

Inputs: three independent designs (validity: content-specific steers, fork probe, engagement-vs-mechanism-absent;
ownership: two clauses per criterion, F/ORDER/RECORD/RI/G, R1-R4, the host-share panel; feasibility: calendar, job
budget, Stage-2 reuse map) and Claude's eight rulings. Runner: `research/runners/_stage3_exit_draft.py` (Stage-2-shaped:
`--shard / --emit-pool-jobs / --aggregate / --selftest`); bar-sanity: `docs/drafts/stage3-exit/bar_sanity_s3_common.py`
+ one spec per criterion; tests: `tests/test_stage3_exit_draft.py`.

## 1. Verdict structure (ownership design, adopted by ruling 1)

Every criterion carries two clauses, scored only by `--aggregate`:

- **CAPABILITY** (Stage-2 machinery): correct on 6/6; the silencing LESION has a reply-level effect on its scored turns,
  reproduced against its own `.rep0` (`tools.lab.attributable_to` against the intact_a/intact_b null); the PERMUTED
  exposure is tracked; plus ONE CONTENT-SPECIFIC intervention (validity design, ruling 2) predicting one specific
  DIFFERENT answer, engaged (calls > 0) on 6/6.
- **OWNERSHIP** (who decided): **F** hold the neural state, set the host decision to its alternative (F=1 host-causal; F=0
  with the reply returned to intact = neural); **ORDER/RECORD** permute or delete information only the host holds;
  **RI** re-decide the recorded spiking read with the host threshold ×/÷1.5 offline (≥85 % of items/cycles unchanged);
  **G** silence or drive one population, host code untouched (the reply follows = neural-causal).
- **R1** a decisive verdict computed by host code with no spiking read to vary is HOST, never UNDEFINED. **R2**
  UNDEFINED is reserved for instrument failure (arm missing; patch `applied=False`; dirty null; errored turn / organ
  error; Stage-2 A1). **R3** host actuation in read-out mode (helper currents, query clocks, admission by verb string) is
  OWNED(read-out). **R4** learned-content is reported separately from decision ownership.
- Vocabulary: CAPABILITY ∈ {PASS, FAIL(clause), INCOMPLETE}; OWNERSHIP ∈ {OWNED, OWNED(read-out), HOST(seam), N/A
  (capability absent, no live G), UNDEFINED}; CRITERION = PASS iff CAPABILITY PASS and OWNERSHIP ∈ {OWNED,
  OWNED(read-out)}; "CAPABILITY PASS, HOST-OWNED" is its own reported outcome. STAGE PASS iff C1-C4 PASS; any UNDEFINED
  → STAGE UNDEFINED (INCOMPLETE) with every number reported (`tools.verdict.Verdict` preconditions).
- Per-seed ownership precedence: UNDEFINED (an instrument not applied) > HOST (any F=1, any ORDER change, RI fail) >
  N/A (capability absent and no live G) > OWNED. Group ownership = OWNED(read-out) only on 6/6.
- No seed threading (ruling 7, as Stage 2). S and D of the host-share schema are NOT measured here (budget); F events
  use `research/runners/host_share_schema.py` with `f_outcome="returned_to_neural"` assigned when the F-arm reply equals
  intact_a's (the flip never reached the reply); the finer masked/neural split is the host-share instrument's job.

## 2. Harness (Stage-2 §3, unchanged except the declared chain departure)

Real `webapp.server.brain_chat` in a worker subprocess; tiny-demo + shipped LTM (`source` = "tiny-demo +LTM"); stub
renderer; no LLM; numpy, one BLAS thread; `BRAIN_CHAT_SEED=<seed>`; `PYTHONHASHSEED=0`; `rich` omitted; fresh process
+ fresh ChatBrain per arm; `corpus_present`, `builtin_facts`, peak RSS, LTM bundle recorded; Stage-2 A1 failure rules.
In-process interventions follow Stage-2's `qp_lesion` pattern: a wrapped call recording `calls`, per-turn `events`
(stamped with the turn label) and the substrate quantity changed; `applied = calls > 0`, else that seed is UNDEFINED.
One harness-wide observation patch in every `chain` arm (`chase_rates`: records the chase's per-slot rates and THR per
cycle for RI; it changes no result).

**Declared departure (ruling 3):** `BRAIN_RECONSOLIDATION=0` in EVERY `chain` arm. Production gap, named: teaching a
one-to-many relation ("the puma chases the shrew" then "the puma chases the gecko") opens the D2 surprise window and
reconsolidation rewrites the first fact in place (server.py ~6340), so the production brain cannot hold a branching
relation taught in chat. Next-method item: a surprise window that distinguishes CONTRADICTION (same agent, same relation,
exclusive patient) from ADDITION (a second patient) before the rewrite fires; until then the chain group runs append-only.

## 3. C1 — 3-hop novel conclusion via the workspace (group `chain`)

Teach, out of order (validity design: the 3-hop terminal is neither first- nor last-taught): `the mantis chases the
flea` · `the puma chases the shrew` · `the shrew chases the slug` · `the puma chases the gecko` · `the gecko chases the
mantis`. Graph: puma→shrew→slug (branch 1, 2 hops); puma→gecko→mantis→flea (branch 2, 3 hops). Every word absent from
the tiny-demo vocabulary and both LTM bundles (checked 2026-10-13).

| label | message | intact | permuted (shrew chases the mantis; slug untaught) |
|---|---|---|---|
| p3 | `does puma chase flea` | yes, DERIVED, 3 hops (branch 2) | yes, derived (both branches reach the flea) |
| p2 | `does puma chase slug` | yes, derived, 2 hops | abstain |
| m_adj | `does puma chase gecko` | yes, NOT derived (told; see §3.3) | same |
| m_cross | `does shrew chase flea` | abstain | yes, derived, 2 hops |
| m_rev | `does flea chase puma` | abstain | abstain |
| r_fork | `what does the puma chase all the way` | FORK: names BOTH slug and flea, or abstains; never exactly one | flea (the branches converge: one honest terminal) |
| r_all | `what does the gecko chase all the way` | flea, derived (2 workspace hops) | flea |
| c_direct | `what does the gecko chase` | mantis (must SURVIVE the lesion) | mantis |
| c_tail | `what does the flea chase all the way` | abstain | abstain |
| c_dog | `what does the dog chase` | cat | cat |

Arms (10/seed): intact_a, intact_b · **ws_lesion** + .rep0 {`BRAIN_GNW_MULTISTEP_LESION=1`, `BRAIN_TRANSITIVE_LESION=1`}
(feasibility: one mechanism read by both routes) · permuted · **steer_leaf** (steer; validity): at the cycle whose cue is
the leaf `flea` the slot carrying `slug` is driven at IGNITE_PA → r_all must name slug (a 4-hop derived path) ·
**silence_mantis** (G; ownership): the proposal `mantis` gets no drive → r_all (and p3 if intact answered it) collapse,
p2 unchanged (path-specific) · **order_swap** (ORDER): the two puma teaches swapped → primary fields on p3/p2/m_adj/r_fork
must be identical · **kselect_order** (F): `OneBrainComposer.query_patient` returns the LAST kb match when ≥2 match
(host_share k-select "order"); F on p3 (alternative: yes-flea) and p2 (alternative: abstain) · reach_cand (INFO,
`BRAIN_TRANSITIVE_SPIKING_REACH=1`, the candidate route).

### 3.1 Capability bar (6/6)
`all`: p3, p2, r_all answered with the expected terminal AND `derived_framed`; m_adj answered, not derived (§3.3);
m_cross, m_rev, c_tail abstain; c_direct mantis; c_dog cat; `fork_honest`; `via_workspace` (p3/p2: `transitive_trace.route`
≠ "direct" and ≥2 ADVANCE cycles in its chase trace; r_all: `_last_gnw_multistep.resolved_hops ≥ 2`). Lesion: effect on
p3/p2/r_all (reply level, reproduced) AND dissociation (c_direct unchanged, still mantis; feasibility design) AND engaged
(`transitive_trace.lesioned` True on p3; the chase trace `lesion` True on r_all). Steer: r_all names slug on 6/6 with
calls ≥ 1 at cue flea. Permuted tracks (its own table).

### 3.2 derived_framed (an HONESTY clause, ruling 3 + the verifier's findings)
Scored from the RECORD, never sentence by sentence: `derived` True; `recalled_svo` None; `derived_from[0][0]` == the
asked agent; `derived_from[-1][2]` == the named terminal; for the chase probes `len(derived_from)` == `resolved_hops`.
Plus a reply-level check that a PRODUCTION frame string appears before the first mention of the claim; the strings are
IMPORTED from production code, never retyped: `compositional_chain_route.frame_derived_answer`'s "I derived this from:"
lead, its generic "I reasoned this myself rather than being told it directly." fallback, and
`source_provenance_honesty.provenance_framed_text`'s swap wording ("I believe …, but I reasoned that myself rather than
being told it directly.", which drops the hop list and is on at defaults when the monitor reads GENERATED). Scored
against `derived_from` / `derived_svo`, never against `supporting_facts` membership. Every C1 lesion discrimination reads
`derived`/`derived_from` beside `recalled_svo` (an intact derived answer and a lesion abstain both have `recalled_svo`
None). derived_framed is honesty evidence ONLY: the derived decision and the lead are host code (a count of ADVANCE
steps plus a template), so it is listed in the host-share panel as host-decision debt, never as ownership evidence.

### 3.3 m_adj depends on the pin
A told-directly conclusion that is also reachable by the chain is over-marked derived at HEAD (the direct-fact
short-circuit is `query_patient(agent, rel) == target`, a first-match: with two puma facts it returns shrew, so
"does puma chase gecko" falls to the chase and abstains or, with a fix that scans all stored facts, answers plain). The
builder may land a stored-fact lookup fix tonight. The scorer does not assume either way: `M_ADJ_TOLD_FIX_IN_PIN` (runner
constant, set at GATE_SHA) selects the expected outcome — True: m_adj answered gecko with `derived` False; False: m_adj
is reported (sub-clause `m_adj_told_plain`) and not counted. Ruling owed at registration.

### 3.4 Ownership bar (6/6)
order_swap: identical primary fields on p3/p2/m_adj/r_fork (any change = the kb order decides = HOST). kselect_order: F=0
on every live event (an event is live only when intact's outcome differs from the host alternative; if intact already
answered p3 by branch 2, p3's F is undefined and p2 carries it). silence_mantis: G=1 (r_all changes, p2 unchanged; live iff
mantis was proposed ≥1). RI: the chase's recorded per-slot rates re-decided at THR ×/÷1.5 keep their ignition class
(0 → COMMIT, 1 → ADVANCE, ≥2 → conflict) on ≥85 % of scored cycles.

### 3.5 Host-shortcut analysis (from code)
PROPOSE is `composer.query_patient` = `_fact_shard_first_match` / `_seq_block` host first-match; the proposal goes to slot 0
at IGNITE_PA with random distractors at DISTRACTOR_FRAC, so the WTA ignites the host's pick; the "emergent" stop
(n_ignited == 0) is exactly the host returning None; the yes/no verdict is `target in path[1:]`. On a branching graph
the chase follows the first-taught branch only. H1 last-taught heuristic → out-of-order teach + permuted. H2 host kb
closure + "derived" template → passes p2/p3/derived_framed; caught by r_fork (one terminal), the lesion (no effect),
the steer (still flea), kselect_order (F=1). H3 H2 + all-leaves fork → passes r_fork; caught by lesion + steer. H4 H3
relayed through the workspace with a final "did it ignite" check → passes the lesion (abstains); caught by the steer.
H5 production at main → caught by r_fork, order_swap (ORDER change), kselect_order (F=1). Residual no bar catches: a host
proposer handing the workspace BOTH branches at equal drive passes r_fork through a genuinely spiking conflict read;
candidate generation stays host (panel seam).

### 3.6 Prediction at main: C1 FAIL. Capability FAIL (p3 abstains: the chase follows the shrew branch; fork_honest 0/6:
r_fork names slug alone; derived_framed 0/6 unless 20614ef7d is in the pin; m_adj abstains unless the told fix lands).
Ownership HOST (order_swap changes p3/p2; kselect_order F=1 on p2; R1). Lesion, steer, permuted predicted 6/6.
**Next method:** propose every stored continuation into the workspace at evidence-graded drive so the fork conflict read
is spiking (the ownership design's "spiking proposal and verdict"); route the reach's goal/detect through workspace
ignition; STDP-learned hop projections.

## 4. C2 — self-report tracks ground-truth confidence (group `meta`)

Ruling 4: the brain's NATURAL uncertainty (category-inheritance conflicts, ownership design) is the primary item set;
the opt-out (abstain) is a counted report level (Kepecs-style); the host margin caveat is measured by RI.

Teach: builders tern, kite (`eats the grub` · `flies in the sky` · `nests in the tree`) and carp, bream (`eats the weed`
· `swims in the lake` · `hides in the reef`); members, two frames each — CONSISTENT lark, finch (sky + tree, truth grub),
dace, roach (lake + reef, truth weed); CONFLICTING bat (sky + lake), duck (sky + reef), seal (lake + tree), auk (tree +
reef), truths moth/snail/squid/krill never taught (any answer is wrong; the abstain is the correct opt-out). Probes:
8 × `what does the X eat` (roles interleaved), 4 taught recalls (r_*_1), c_dog, 4 taught recalls again (r_*_2, the INFO
noise arm's second level). 37 turns. PERMUTED: the builders' foods swapped AND the consistent/conflicting ROLES swapped
between the two member sets (a name-keyed confidence cannot track; the answers must follow the food).

Reply level L: 2 = answered, no hedge; 1 = answered with `metacog_production_organ.hedge_prefix()`; 0 = abstain. The
guess flag ("that's a guess") is NOT counted (set by type on every InheritedSVO: a host label). Correct = the reply
names the WORLD truth (an abstain never does).

**Capability bar (6/6):** type-2 AUROC (L predicting correctness) over the 8-item GUESS STRATUM ≥ 0.75 (= half the
conflicting items opting out while every other guess item is answered correctly; stratified so a taught-vs-guess
host proxy cannot carry it); ≥ 2 correct guess items, else FAIL "capability absent (no generalization to score)",
never UNDEFINED (a capability-absent shape must not hide); taught recalls correct; c_dog. Lesion `inherit_lesion`
(`BRAIN_CATEGORY_INHERIT_LESION=1`): ≥1 consistent item changes, reproduced, engaged (`category_inherit.read.lesion`
== "all"), dissociation: the taught recalls unchanged. Steer `steer_frame` (G): the `eat grub` frame assembly is driven
during the bat's read (a near-tie) → the bat answers grub while the lark is unchanged; engaged (calls ≥ 1). Permuted
tracks (its own AUROC ≥ 0.75 with the permuted truth table).

**Ownership bar (6/6):** RI on the opt-out: each guess item re-decided from the recorded `category_inherit.read.rates`
at TIE_MARGIN ∈ {1.1, 1.25, 1.6} × FLOOR_RATE ∈ {0.013, 0.02, 0.03}; L unchanged on ≥ 7/8 items (the dynamics resolved the
near-ties, not the host cut). G=1 (steer_frame). Admission by verb string, the floor and the margin are host cuts
(RULING c1-moat-ruled-1013 logs them as host-decision debt) → OWNED(read-out) at best (R3); RI fail → HOST.

INFO arms: metacog_lesion (`BRAIN_METACOG_LESION=1`; the verbal hedge, predicted inert on guess turns: there is no
recall trace on a guess and `_metacog_qualify` reads only the role-decode scalar); noise_taught (the validity design's
σ-noise instrument demoted to the taught recalls: r_*_1 at σ1, r_*_2 at σ2, `_emergent_graceful_degradation_derisk._noise`
on `composer.store_conns`, restored after the turn; σ levels dev-calibrated; it adds the one read the natural set cannot
give — whether the verbal hedge ever varies under degraded evidence).

Host-shortcut analysis: H1 hedge-by-taught (abstain on every guess) → 0 correct → FAIL. H2 item-keyed → fails the
role-swapped permuted. H3 host conflict counter over the kb (abstain iff a member's frames span two builder categories)
→ AUROC 1.0; caught by inherit_lesion (no effect) and steer_frame (bat stays abstain). H4 never-abstain → AUROC 0.5.
Residual: the evidence for the verbal hedge is host arithmetic (`mean_role_confidence` → `evidence_from_role_conf`);
only the judge is spiking — reported in the panel.

Prediction: capability UNCERTAIN (near-ties often abstain: AUROC may clear 0.75; the hedge never varies); ownership
HOST predicted (RI fails on near-ties flipping between margins 1.1 and 1.6). **Next method:** a spiking opt-out
population fed by the two frame assemblies (mutual inhibition wins the near-tie) driving the metacog workspace
synaptically, no host scalar in between.

## 5. C3 — false belief passes (group `tom`)

Closed grammar of `webapp/false_belief_chat.py` (every sentence pinned by a test against `classify`). Ruling 5.

| scenario | narration (literal; fillers are ordinary turns) | query → intact | permuted (witnessing swapped) → |
|---|---|---|---|
| S1 classic + delay | `Sally puts the marble in the basket.` `Sally leaves the room.` `Anne moves the marble to the box.` `what does the dog chase` `how do you feel` | q1 `Where will Sally look for the marble?` → basket | leave → filler: box |
| S2 COUNTERBALANCED | `Tom puts the key in the drawer.` `Tom moves the key to the jar.` `Tom leaves the room.` `Mia moves the key to the cup.` | q2 `Where will Tom look for the key?` → jar (the SECOND-named location; the slot-0 default says drawer) | leave → filler: cup |
| S3 true belief | `Lea puts the coin in the bag.` `Lea leaves the room.` `Lea returns.` `Ben moves the coin to the tin.` | q3 → tin | return → filler: bag |
| S4 two unwitnessed moves, returns after | `Kate puts the ring in the chest.` `Kate leaves the room.` `Ben moves the ring to the pot.` `Ben moves the ring to the shelf.` `Kate returns.` | q4 → chest | leave → filler: shelf |
| control | | m1 `Where will Ben look for the ring?` → no location frame (agent_mismatch); c_dog | same |

Arms (10/seed): intact_a, intact_b · **store_lesion** + .rep0 (`BRAIN_FALSE_BELIEF_STORE_LESION=1`; effect on q1-q4,
engaged: `false_belief_tom.store_lesioned`) · permuted · **steer_store** (G; validity): on q1, before the read, the `box`
belief assembly is driven with the organ's own write current (`_write_event(slot 1, witnessed=True)`, W_WRITE ×
WRITE_DRIVE_STEPS) → q1 names box · **witness_flip** (F; ownership): `FalseBeliefChatOrgan.observe_event`'s `witnessed`
boolean inverted on every MOVE (the first call per organ is the PLACE) → alternatives q1 box, q2 cup, q3 bag, q4 shelf ·
CANDIDATE (INFO, `BRAIN_FALSE_BELIEF_SPIKING_WITNESS=1`, `BRAIN_FALSE_BELIEF_WITNESS_OP=4`): cand_intact,
cand_witness_lesion (+`BRAIN_FALSE_BELIEF_WITNESS_LESION=1`), cand_presence_clamp (G: presence silenced across S3's move
→ q3 bag; driven across S1's move → q1 box; patch on `SpikingWitnessFalseBeliefOrgan.observe_event`, needs dev-7
verification). `BRAIN_FALSE_BELIEF_LESION` (a host witnessing override) is excluded (validity rule 4; battery Amendment B).

Capability bar (6/6): q1 basket, q2 jar, q3 tin, q4 chest; m1 no location; c_dog; store_lesion changes ≥1 of q1-q4,
reproduced, engaged; steer_store q1 → box with calls == 1; permuted tracks. Ownership bar (6/6): witness_flip F=0 on
every live MOVE event; G=1 (steer_store); RI: the belief argmax and a rate-ratio ≥ 2 read-out agree on q1-q4 (an all-zero
read returns slot 0 — `belief_store_read_ok` already refuses it).

Host-shortcut analysis: H1 first-named location → S2 (jar) and S3. H2 present-at-query → reality → S4 (chest). H3
production at main: `_Scenario.present` (regexes) gates the write; the store read is never maintained between turns
(no simulated time passes; fb-store prereg §2) → passes S1-S4 and the steer; caught by store_lesion (`_fb_store/verdict.json`
STORE-NOT-LOAD-BEARING 0/6) and witness_flip (F=1 by construction). H4 host decision with the store as a silent veto →
passes store_lesion by abstaining; caught by the steer. H5 host computes the belief and writes it → passes the steer;
caught by witness_flip. Known hole: maintenance is not tested (named next method).

Prediction: capability PASS at defaults (S2 is the first counterbalanced read; its risk is named) with store_lesion
predicted NOT load-bearing (0/6) → capability FAIL "not load-bearing"; ownership HOST (F=1 ×4). In candidate mode:
OWNED(read-out). **Next method:** the belief store's free dynamics ticked on every chat turn (time passes between
narration and query); flip the spiking witness to default (T21/P4).

## 6. C4 — deep held-out inference after teacher-corrected episodes ≥ replay-only (group `credit`)

"Deep" = the correction must reach an item NO correction mentioned (ruling 6), which no stored-fact lookup supplies.
Synthesis: the ownership design's turn shape with the validity design's hidden-feature structure (a shared feature carries
the correction; a disjoint feature controls specificity). Exposure (14): robin, wren `flies in the sky` + `eats the worm`;
lark, swift, canary `flies in the sky` + `hunts the wasp`; jay, rook `flies in the sky` + `hunts the cricket`. Episodes:
e1 `what does the lark eat` (the brain's own guess: worm) · k1 `the lark eats the seed` · e2 `what does the swift eat` ·
k2 `the swift eats the seed`. Probes: **H** `what does the canary eat` → seed (held out: a wasp-hunter no correction
mentioned) · J `what does the jay eat` → NOT seed (a cricket-hunter) · c_lark → seed (retained) · c_robin → worm · c_dog.
23 turns.

Arms (10/seed): corrected_a, corrected_b · **replay_only** (counted comparator: e1/e2 → `say something`, the production
idle path; the same facts at the same positions, no error before them) · **no_teacher** + .rep0 (k1/k2 → `say something`;
counted as the "teacher load-bearing" lesion) · permuted (k1/k2 say `nut` → H nut, c_lark nut) · **learn_lesion** + .rep0
(`BRAIN_CATEGORY_INHERIT_LESION=lateral_learn`: the frame→frame gate never opens) · **kb_pop** (RECORD): after k1/k2 the
corrected fact's host kb record is popped (`composer.kb`; the block's synapses and the organ untouched) → H unchanged
means the content is substrate-held (precedent: D6 NOREC_H) · **corr_freeze** (G): `CategoryInheritOrgan.observe` skips
(lark, seed) and (swift, seed) only → H must revert to no_teacher's reply.

Ruling 6 choice: kb_pop + corr_freeze over the PLASTIC_SETS-revert. They answer the two questions directly (is the
content in the host record? did the correction's plasticity carry it?) with verifiable engagement and no snapshot RSS;
the revert needs an enumeration of every plastic set at the pin that no test can verify without a brain and an extra
≤1 GB snapshot. The revert is kept as an optional amendment if the dev-7 RSS budget allows.

Capability bar (6/6): H seed (as a flagged guess is fine); corrected ≥ replay_only on H (the roadmap's literal "≥";
`strict` = replay_only's H ≠ seed reported); no_teacher changes H, reproduced; learn_lesion changes H, reproduced,
engaged (`category_inherit.read.lesion` == "lateral_learn"); permuted H → nut; J not seed; c_lark seed; c_robin worm;
c_dog. Ownership bar (6/6): kb_pop F=0 (H unchanged); corr_freeze G=1 (H reverts); RI on H's read (the C2 grid).

Host-shortcut analysis: H1 most-recent correction for every animal → J. H2 exemplar/nearest-neighbour over stored items
→ passes intact and replay_only equally; caught by corr_freeze (the generalization lives in item traces, survives) and
no_teacher (no change). H3 host rule inferring and writing `canary eats seed` as a new block → caught by kb_pop (H
follows the record: F=1) and corr_freeze. H4 persistent activity → corr_freeze cannot show it (correct: deep credit is a
weight change). Residual: the organ is fed only facts the host SVO extractor acquires (a sense boundary); admission by
verb, floor and margin are host cuts; no organ sees contingency (whether a correction followed an error).

Prediction: FAIL. corrected == replay_only exactly (inference turns open no plasticity gate; "≥" holds as equality);
H stays worm or abstains (the organ is LTP-only with zero decay: `fly sky → eat worm` reinforced by every sky-flyer and
never weakened by a correction) → no_teacher changes nothing → "not exposure-driven"; kb_pop F=0 (content substrate-
held); ownership N/A. Wall question answered: the missing companion process is error-driven depression (a teacher
mismatch → DA dip → LTD on the contradicted association: the P3.1 teacher-credit bridge). **Next method:** an error-gated
plastic pathway between features and properties (P3.1/P3.2 into chat), default OFF, dev seeds first.

## 7. Nulls (bar-sanity generators; each must score 0.000; `tools/bar_sanity.py`, n = 1000, seed 0)
chain: null_last_taught, null_kb_closure_first_match (H2), null_kb_closure_all_leaves (H3), null_closure_relay_ignition
(H4), null_unmarked_terminal, null_single_branch_fork, null_lesion_inert, null_lesion_unreproduced, null_steer_inert,
null_permuted_inert, null_unclean, null_all_abstain, random | perfect: spiking per-hop cue, conflict abstain on the fork,
derived frame, steer follows. meta: null_hedge_by_taught, null_item_keyed, null_never_abstains, null_random_optout,
null_conflict_counter (host over kb; lesion inert, steer inert), null_all_abstain, null_lesion_inert, null_permuted_inert
| perfect: opt-out = f(the organ's near-tie) through the organ. tom: null_first_location, null_present_means_reality,
null_host_rule_unread_store (store lesion inert), null_host_rule_store_veto (steer inert), null_host_written_belief,
null_lesion_inert, null_permuted_inert | perfect: maintained store + witnessing. credit: null_recent_correction,
null_exemplar_nn (teacher not load-bearing), null_host_inferred_block, null_activity_only, null_no_teacher_inert,
null_learn_lesion_inert, null_permuted_inert | perfect: credit lands in non-item plastic weights. The decision is the
REAL aggregate's criterion verdict (capability AND ownership), fed from memory.

## 8. Ownership panel (reported; Phase D's number, target host share ≤ 0.2)
`host_decision_share` over the counted F events: C1 kselect_order on p3/p2 (live only); C3 witness_flip ×4 (q1-q4); C4
kb_pop on H. Beside it: the RI-fail share, the G-pass share, the ORDER-change share (C1), learned-content (C4 kb_pop).
Host-decision DEBT (declared, not measured by an F here): the C1 derived flag + lead (host template over an ADVANCE
count), the DETECT regexes, `target in path`, the C2 evidence scalar, admission/floor/margin cuts, the C3 location→slot
map and argmax, the C4 SVO extractor. Necessity table (non-negotiable 4): one row per lesion arm with its reply effect.

## 9. Jobs, RAM, calendar
| group | arms/seed (counted + info) | turns | est. min/arm (pool) | jobs | job-h |
|---|---|---|---|---|---|
| chain | 9 + 1 | 15 | 22 | 60 | 22 |
| meta | 6 + 2 | 37 | 40 | 48 | 32 |
| tom | 7 + 3 | 24 | 25 | 60 | 25 |
| credit | 10 + 0 | 23 | 25 | 60 | 25 |
| total | 38 | | | **228** (≤ 260, ruling 7) | ~104 |

mem_gb = 9 under memcap 10 (Stage-2 A2: 6.3-8.1 GB), refined by the dev-7 smoke to ceil(1.3 × peak); every line
max_age_h=240. Pool 3 nodes (1 slot each) ≈ 35 h; + AWS r7i.4xlarge (~10 slots at 12 GB) ≈ 8 h, ~$9-12 (under the
$50/day cap, `aws_idle_stop.sh` armed). Calendar: runner + bar-sanity banked 10-13 (this branch); Claude review + prereg
alone by 11-14 (PREREG_SHA); dev-7 smoke of every arm 11-15..11-24 (pool, after the Stage-2 gate lines dispatch);
instrument-only amendments + GATE_SHA by 11-28 (after Stage-2 scoring 11-21); queue 12-01..12-05; re-queue missing
shards once 12-05; `--aggregate` ≤ 12-08; verify-go before 12-19.

## 10. Needs dev-7 verification (patches implemented defensively: applied=False → UNDEFINED)
steer_leaf / silence_mantis (the cue is read off a `query_patient` wrapper; the drive edit inside `_deliberate_hop_conf`);
chase_rates recorder; kselect_order (the kb multi-match on puma); steer_store (`_write_event` on slot 1 before the q1
read); witness_flip (PLACE = first call per organ instance); presence_clamp (candidate mode internals); steer_frame
(a step wrapper on the organ's bridge during the bat's read); noise_taught (σ levels, restore); kb_pop (the fact-shard
rebuild after a `kb` pop); corr_freeze. Also: the surprise notice on the second puma teach with reconsolidation off; the
C2 taught recalls asked twice (discourse leads); "say something" on the credit idle turns; peak RSS of the 37-turn meta
session; m_adj's outcome (§3.3).

## 11. Open rulings for Claude (at registration)
1. derived_framed required in the pin (20614ef7d): confirmed by ruling 3; m_adj's told-directly fix (§3.3): in the pin or
   reported-only (`M_ADJ_TOLD_FIX_IN_PIN`).
2. C2 AUROC bar 0.75 vs 0.80 (0.80 would need 3 of 4 conflicting opt-outs); the "≥2 correct → else FAIL" rule.
3. Whether T21 (spiking witness) lands before 11-28 or C3 is queued knowingly reading HOST on ownership.
4. noise_taught σ levels from dev seeds 7/8/9 only; or drop the INFO arm (−6 jobs).
5. The optional PLASTIC_SETS-revert amendment for C4 if the dev-7 RSS allows.
6. RI thresholds (≥85 % / ≥7 of 8) and the ×/÷1.5 factor.

## 12. DRAFT PREREGISTRATION TEXT (NOT REGISTERED; binding only when committed alone as a findings prereg)

Title: Stage-3 (ADOLESCENT) exit gate on the integrated production brain. Runner `research/runners/stage3_exit_gate.py`
(this draft renamed at registration); harness §2; groups, turns, arms, bars, nulls, verdict rules and predictions exactly
as §§1-8 of this document at the registered SHA; seeds 42/43/44/100/101/102; dev seeds 1-12 govern nothing. Bars fixed
before data: capability (§3.1-3.2, §4, §5, §6), ownership (§3.4, §4, §5, §6), the two-clause criterion rule and the
stage rule (§1). UNDEFINED rules R1-R4 (§1) and Stage-2 A1. Predictions (§3.6, §4, §5, §6): C1 FAIL (capability + HOST),
C2 uncertain capability / HOST ownership, C3 capability FAIL "not load-bearing" + HOST, C4 FAIL (capability absent; N/A),
STAGE FAIL; panel host share predicted > 0.5. Every FAIL names its next method (§§3-6). Amendments: dated, filed before
the aggregate is read; the only planned ones are instrument-only (σ levels, `M_ADJ_TOLD_FIX_IN_PIN`, mem_gb from the
dev-7 peak). bar-sanity: `docs/drafts/stage3-exit/bar_sanity_record_s3_c{1..4}.json` (to be regenerated at the
registered scorer SHA). GATE_SHA must contain 20614ef7d (ruling 3) and this file's scorer.
