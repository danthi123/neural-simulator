---
type: finding
status: contributing
claim_check: synthesis
date: 2026-08-11
mechanism: ROLE-GATE x VARIABLE-SUBJECT-POSITION (LEVER 4) — remove the ONSET confound LEVER 3 exposed by making the subject's ordinal RANDOM and cueing it with an ARBITRARY LEARNABLE CASE TAG (the Bates-MacWhinney competition-model case marker; Japanese-style free word order where the NOMINATIVE marker, not position, identifies the subject). The subject noun carries the NOM tag (tag 0) at a random ordinal in [1,L] (position 0 RESERVED as a distractor slot so the onset prior lands on a distractor); the L distractors carry random NON-NOM tags. Each token's gate input is a COMPOSITE barcode = OR(noun_barcode, tag_barcode) in the SAME 64-dim code space (noun = the CONTENT, tag = the ROLE cue, both linearly readable; W1 unchanged). Which tag is NOM is arbitrary and defined ONLY by the verb-agreement reward, so an onset/untrained/permuted-reward prior loads a DISTRACTOR and only a LEARNED content-conditioned credit signal can fire the subject. Reuse-by-import of the LEVER-3 2-layer CompetitiveHiddenChainedGate (chained-FA+sigma' / canonical-KP transport-free credit + competitive stabilizer) + the REAL spiking D3 SpikingSlot; ONLY the stream + the subject cue + the eval's subject identification change.
lane: emergence engine / working memory x gap#4 / role-gate transport-free reliability
verdict: 6-SEED (42 43 44 100 101 102), real spiking D3 slot, L=5 and L=6 (GO distance L6 = dist 7, chance 0.250, held-out NOVEL fillers). PRIMARY VALIDITY CHECK PASSES (task_valid=True) — the onset confound is REMOVED: UNTRAINED stabilized gate 0.289, permuted-reward FA 0.281 / KP 0.289, ONSET gate 0.289 — ALL fail <= chance+0.15 (0.400), where on the LEVER-3 subject-first task the untrained + permuted gates both scored 1.000. HONEST NEGATIVE (role_go=False), OUTCOME #2 (credit can't, the ceiling can) — the residual is finally CLEAN: with onset unable to cheat, the ALIGNED (weight-transport, exact-feedback) credit CEILING LEARNS the arbitrary case-cue->role mapping RELIABLY (aligned+stabilizer L6 0.950 [min 0.822], gap +0.92 [min +0.86], fires NOM 1.00/obl 0.08 on all 6 seeds; L5 0.991 [min 0.956]), and the case-marker ORACLE ceiling reaches 0.969 [min 0.911] — but the TRANSPORT-FREE arms do NOT clear the reliability bar: chained-FA+stabilizer is BIMODAL (L6 0.511 [min 0.233, max 1.000, std 0.31]: seeds 42/43 solve 1.000/0.889, seeds 44/100/101/102 collapse to the onset floor), canonical-KP+stabilizer COLLAPSES on all 6 (0.289 [min 0.233], fire NOM 0.00). The noun-only IDENTITY control fails (0.272, gap -0.13), the NOM oracle beats the permuted-cue control (0.969 vs 0.274), the n-gram HELD-OUT floor is at chance (0.357 with the 1/(L+1) single-load partial credit), hold is load-bearing (oracle-lesion 0.294). So "does transport-free credit induce role" — untestable for the whole arc under the onset confound — is now DECIDED: transport-free credit (fixed-random / canonical-KP feedback) is genuinely INSUFFICIENT on the real task; the residual is the FEEDBACK-ALIGNMENT RELIABILITY of the transport-free hop under the harder cue-detection objective, isolated clean (same net + stabilizer + credit rule reach role at 6/6 with exact feedback). NO sim/ edit; SIM_BACKEND=numpy.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_var_bind_rolegate_varpos_derisk.py
artifacts:
  - research/findings/raw/_rolegate_varpos/varpos_6seed.json
---

# Role-gate LEVER 4 — a variable-subject-position case-marked stream makes the transport-free CREDIT question testable; the answer is a CLEAN honest negative (exact-feedback credit reaches role reliably, transport-free does not)

## TL;DR

LEVER 3 (`_var_bind_rolegate_competitive_stabilizer_derisk`) closed the fire-everything basin but two killer anti-cheats exposed the result as an ONSET artifact: on the SUBJECT-FIRST stream, an UNTRAINED stabilized gate and a PERMUTED-reward gate both scored 1.000, because the stabilizer's fire-once budget is an onset gate and the subject was always token 0. "Does transport-free credit induce role" was never testable.

LEVER 4 removes that confound. The subject is now the noun carrying an ARBITRARY NOMINATIVE case tag at a RANDOM ordinal (the Bates-MacWhinney competition model of cue-based role assignment; a Japanese-style free-word-order language where case, not position, marks the subject). Which tag is nominative is defined only by the verb-agreement reward, so an onset / untrained / permuted prior loads a distractor and only a learned, content-conditioned credit signal can fire the subject.

Result, 6 seeds, real spiking slot: the task is now VALID (untrained + permuted-reward + onset all fail <= chance+0.15) — and the answer is a CLEAN honest negative. The ALIGNED (exact-feedback) credit ceiling learns the arbitrary case-cue->role mapping RELIABLY (6/6 seeds), the case-marker oracle reaches ~0.97, but TRANSPORT-FREE credit does not: chained-FA is bimodal (2/6 seeds solve), canonical-KP collapses on all 6. The precisely-isolated residual is the feedback-alignment reliability of the transport-free hop — the same net + stabilizer + credit rule reach role at 6/6 the moment the feedback is exact.

## What changed vs LEVER 3 (only the stream + the cue + the eval's subject identification)

The gate, the chained-FA+sigma' / canonical-KP transport-free credit, the competitive forward stabilizer, and the REAL spiking D3 SpikingSlot deployment are the LEVER-3 machinery, reused by import unchanged. The three changes:

1. **Variable subject position.** The subject sits at a random ordinal in [1, L]; position 0 is reserved as a distractor slot. This is a deliberate, documented choice: the stabilizer's fire-once budget creates an ONSET prior (fire the first token), so reserving position 0 makes that prior land on a distractor — the direct demonstration that onset != the answer. No fixed position solves the task (the subject is uniform over L positions), and the tag-identity crux (below) rejects any residual positional shortcut because it demands TAG selectivity.
2. **A learnable case-tag cue.** C=4 case tags; tag 0 = NOM (the subject cue), tags 1..3 = distractor markers. Each token's gate input is the binary OR of its noun barcode and its tag barcode in the same 64-dim space. The noun carries the content (its agreement feature -> the verb), the tag carries the role cue. Which tag is NOM is arbitrary — the gate can only exploit it by LEARNING, from the distal verb reward, that the NOM barcode predicts a correct verb (cue validity = availability x reliability, the competition model's error-driven cue weight).
3. **Subject identified by tag, not position.** The eval writes the loaded token's own feature to the slot and reads the verb; the training target is the feature of the NOM-tagged token wherever it sits. `train_varpos` is a faithful copy of the parent `train_chained` credit loop with exactly these two data changes (composite code, subject-by-tag); the credit rule, stabilizer forward, e-prop eligibility, homeostasis and KP co-adaptation are byte-for-byte the parent's.

## Why this is LEARNABLE-only (removes the onset confound) — the biology grounding

Read (RAG-surfaced, then read in full): `research/findings/2026-06-19-phase2-case-cue-crosslanguage-scoping.md` — the Bates-MacWhinney Competition Model (Bates & MacWhinney 1982/1989; MacWhinney-Bates-Kliegl 1984 English/German/Italian; the Japanese case-vs-order studies, Sasaki & MacWhinney / Kilborn & Ito). Its load-bearing points, applied here:

- **The case marker is an arbitrary, language-specific, LEARNED cue.** Which particle is nominative (ga in Japanese) is not derivable from anything — the comprehender learns it. So a gate with random weights (untrained) cannot know which of C tag barcodes is NOM, and a gate trained on a shuffled reward (permuted) cannot learn it. Both must fall back on the structural onset prior, which now loads a distractor -> chance. This is exactly that scoping doc's NO-LEARNING and PERMUTED-CASE controls, realized as the untrained-gate and permuted-reward arms.
- **In a free-word-order case language, position is uninformative** (word order is available but unreliable; the error-driven learner drives its validity down). Here position carries zero information about the subject, so an onset/positional prior fails by construction — the intended demonstration.
- **The oracle is the case-cue-detector** (fires on the NOM barcode) — a legitimate "knows the rule" ceiling, exactly the installed-validity path that scoping doc uses as its robust arm.

Decisive artifact: `research/findings/raw/_rolegate_varpos/varpos_6seed.json` (the 6-seed merge; every per-seed point is inside it under `points[].per_seed`, reproducible by the fan command in the runner docstring). All 9 verdict preconditions hold on the merge (generalisation defined, oracle ceiling exists, n-gram floor at chance, onset/untrained/permuted-reward all fail, oracle beats permuted-cue, hold zero-input, credit differs from permuted-reward).

Note on identical failing arms (a lever-efficacy warning a reviewer will see): the failure mode of this task is a SINGLE onset attractor — a gate that does not learn the cue fires token 0 (the fire-once budget) and loads the reserved distractor. So every failing arm (onset, untrained, canonical-KP on all seeds, chained-FA on the 4 collapsing seeds, permuted-reward) converges to byte-identical spiking-slot outputs. That numeric identity is the RESULT (all no-credit paths collapse to the same onset floor), not a duplicated computation; the arms are genuinely distinct trainings.

## The validity check — the task is no longer onset-trivial (6-seed means, GO distance L=6, chance 0.250, chance+0.15 = 0.400)

| control (must FAIL <= chance+0.15) | L=6 mean [min, max] | fires NOM/obl | verdict |
|---|---|---|---|
| UNTRAINED stabilized gate | 0.289 [0.233, 0.322] | 0.00 / 0.17 | FAILS (was 1.000 on subject-first) |
| PERMUTED-reward chained-FA | 0.281 [0.000, 0.411] | 0.06 / 0.14 | FAILS |
| PERMUTED-reward canonical-KP | 0.289 [0.233, 0.322] | 0.00 / 0.17 | FAILS |
| ONSET gate (fires t==0) | 0.289 [0.233, 0.322] | 0.00 / 0.17 | FAILS (~chance; onset != answer) |
| identity control (noun-only, tag stripped) | 0.272 [0.178, 0.356] | 0.04 / 0.18 | FAILS (tag is load-bearing) |

The untrained gate is an ONSET gate (fires t==0, loads a distractor) so it sits at the single-load floor (chance plus the 1/(L+1) partial credit that any single load earns when a distractor feature coincidentally matches the subject's). All controls fail decisively. Task is VALID. (On the LEVER-3 subject-first task the untrained gate and permuted-reward BOTH scored 1.000 — the exact confound this lever removes.)

Task validity is corroborated: the NOM oracle beats the PERMUTED-CUE control (move the NOM tag off the subject) 0.969 vs 0.274 (the task is tag-driven); the n-gram HELD-OUT floor is 0.357 (at the single-load floor, no surface n-gram predicts the verb); hold is load-bearing (oracle-lesion, recur=0, 0.294).

## Per-arm 6-seed table (real spiking slot, held-out novel fillers)

| arm | L=5 mean [min] | L=6 mean [min] | fires NOM/obl (L6) | tag-gap (L6) [min] | reliable? |
|---|---|---|---|---|---|
| case-marker ORACLE (ceiling) | 0.983 [0.944] | 0.969 [0.911] | (detector) | — | yes (target exists) |
| aligned + stabilizer (exact-feedback credit CEILING) | 0.991 [0.956] | 0.950 [0.822] | 1.00 / 0.08 | +0.92 [+0.86] | YES, 6/6 |
| chained-FA + stabilizer (TRANSPORT-FREE) | 0.459 [0.200] | 0.511 [0.233] | 0.32 / 0.11 | +0.21 [-0.17] | NO — bimodal 2/6 |
| canonical-KP + stabilizer (TRANSPORT-FREE) | 0.252 [0.200] | 0.289 [0.233] | 0.00 / 0.17 | -0.17 [-0.17] | NO — 0/6 (collapses to onset) |

Per-seed transport-free FA accuracy (L=6): seed 42 = 1.000, 43 = 0.889, 44 = 0.333, 100 = 0.289, 101 = 0.233, 102 = 0.322. The aligned ceiling on the SAME seeds: 0.822, 0.956, 0.989, 0.967, 0.978, 0.989 — reliable everywhere. FA differs from aligned ONLY in the feedback matrices (fixed-random vs weight-transport), so the bimodality is attributable to feedback alignment, not depth, sigma', the stabilizer, the task, or the operating point (all shared with the reliable aligned arm).

## Verdict — OUTCOME #2 (credit can't, but the ceiling can): the residual is finally CLEAN

`role_go=False`, `task_valid=True`. This is the first-class honest negative the whole arc chased, now UNCONFOUNDED. On a genuinely-hard variable-position task where onset cannot cheat:

- **The credit assignment CAN express role with exact feedback** — the aligned (weight-transport) ceiling reaches role reliably (6/6, min 0.822-0.956), the oracle confirms a target exists, the identity control confirms the tag cue is load-bearing. The architecture + task + stabilizer + credit rule are sound.
- **Transport-free credit is genuinely INSUFFICIENT here** — chained-FA aligns and solves on 2/6 seeds and collapses to the onset floor on 4/6; canonical-KP collapses on all 6. This is a real, seed-dependent FEEDBACK-ALIGNMENT reliability failure of the transport-free hop, now isolated with no onset escape hatch to mask or fake it.

This resolves the methodological question LEVER 3 raised. The prior arc's transport-free "1.000" was onset structure; the honest transport-free number on the real task is bimodal-FA / collapsed-KP.

## The precisely-isolated residual + the next candidate

The residual is NOT depth, NOT sigma', NOT the stabilizer, NOT the task, NOT the operating point — the aligned arm holds all of those fixed and reaches role at 6/6. The residual is **the feedback-alignment reliability of the transport-free hop under this harder cue-detection objective**: on a task where the gate must detect an arbitrary NOM barcode among superposed distractors and assign its distal verb credit through the fixed/co-adapting feedback, the random-feedback alignment succeeds on some seeds and collapses on others (the same seed-dependent basin the banked LEVER-2/3 findings described, now on the genuinely-hard task). Named next levers, trained WITH this credit rule (a LEVER-5 arc, out of this lever's scope):

1. **Stronger transport-free alignment** — canonical Kolen-Pollack with a longer co-adaptation schedule / larger kp_lr / an explicit alignment warm-up, or the Akrout weight-mirror phase, so the feedback tracks W2^T reliably before the role objective bites.
2. **A cleaner hidden code for the NOM template** — a wider or sparser hidden layer (or a dedicated cue subspace) so the NOM barcode is linearly separable enough that even noisy feedback lands the credit consistently.
3. **The emergence engine's own ordinal/cue code** supplying the role-relevant conjunction, so the transport-free credit shapes a pre-structured representation rather than discovering the tag from scratch.

## Scope, honesty, reuse

The 2-layer net + chained credit + the competitive stabilizer are HOST math; their on-substrate spiking DA-gated / lateral-inhibitory realisation is the standing next rung (unchanged from LEVER 3). The case-tag lexicon and the composite barcode are legitimate environment/front-end artifacts (the language's data), exactly the boundary the case-cue scoping doc flags. Reuse-by-import of `CompetitiveHiddenChainedGate`, `SpikingSlot`, `role_layout`, `MarkerRoleGate`, `_mint_codes`, the n-gram floor. **NO sim/ edit** (runner-side only). SIM_BACKEND=numpy (sub-1k-neuron LIF loops are launch-bound: CPU faster). Every knob is in the artifact config. 1-seed is a smoke indicator; the 6-seed sweep above is decisive.
