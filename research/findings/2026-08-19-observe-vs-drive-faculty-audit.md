---
type: finding
status: live
date: 2026-08-19
mechanism: observe-vs-drive-faculty-audit
artifacts:
  - research/findings/raw/_observe_vs_drive/audit.json
  - research/findings/raw/_observe_vs_drive/audit.log
  - research/findings/raw/_observe_vs_drive/audit_fix.json
  - research/findings/raw/_observe_vs_drive/audit_affect_coloring.json
runner: research/findings/raw/_observe_vs_drive/audit.py
also_ran:
  - research/findings/raw/_observe_vs_drive/audit_fix.py
  - research/findings/raw/_observe_vs_drive/audit_affect_coloring.py
  - research/findings/raw/_observe_vs_drive/audit_finalize.py
---

# Observe-vs-drive audit of the whole live /api/brain-chat turn — which default-on faculties actually CHANGE what the brain says

**The anti-hollow-integration check ([[feedback_faculties_must_drive_not_observe]]) applied across the WHOLE live
turn, not one faculty.** Of the 31 faculties `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` lists `on_by_default: YES`,
which CHANGE the reply TEXT (DRIVERS), which only produce/relocate substrate another faculty consumes (FEEDERS), and
which compute a neural verdict that goes nowhere observable (DEAD OBSERVERS — the exact hollow-integration drift)?

**Method (no ledger-trust; independent).** Each faculty's TRIGGER probe was run through the REAL
`webapp.server.brain_chat` handler in-process (tiny-demo, seed 42, numpy-CPU, `rich=False` except discourse-planner),
INTACT then LESIONED (its own env lesion flag / master-disable), on clean reset sessions. Every OTHER heavy organ was
disabled as a consistent baseline across both arms so each turn is tractable. Classification is the observed `answer`
diff: **DRIVER** = the reply TEXT changes intact-vs-lesioned; **FEEDER** = answer-preserving substrate/plumbing (a
MECHANISM claim, verified byte-identical under the escape); **DEAD OBSERVER** = answer byte-identical AND a neural
verdict stashed as metadata with no consumer; **NOT-CLEANLY-TESTABLE** = no reliable trigger/lesion on THIS config.

## Verdict <!--derived from research/findings/raw/_observe_vs_drive/audit.json tally_final-->

**23 DRIVERS · 2 FEEDERS · 6 NOT-CLEANLY-TESTABLE (on this config) · 0 DEAD OBSERVERS.** No default-on faculty was
found to be a dead observer. The owner's 10-faculty spot-check (affect / swap / DA / metacog / surprise / world-model
/ pmem / reconsolidation / pragmatic / curiosity = drivers) HELD — all ten were independently confirmed as DRIVERS
through the real handler. The one KNOWN observe-only faculty, the #77 GNW thought-swap (`gnw-thought-swap`), is
`on_by_default: NO` (default-OFF) — outside this set — and was explicitly superseded by its DRIVER counterpart #85
`swap-drives-response`, which this audit confirms IS a driver.

## The map (every per-faculty intact/lesion answer is in `audit.json`)

| Faculty (ledger key) | Lesion mechanism | Class | One-line evidence (intact → lesioned) |
|---|---|---|---|
| surprise-monitor | `BRAIN_SURPRISE_LESION` | DRIVER | `'The dog chases cat.'` → prepends `'That surprises me — my mismatch monitor fired…'` |
| metacog-monitor | `BRAIN_METACOG_LESION` | DRIVER | clean recall → prepends `'My decision-margin reads this as low-confidence…'` |
| worldmodel-forward | `BRAIN_WORLDMODEL_LESION` | DRIVER | `'…expects this to keep going positive'` → `'…negative'` (prediction flips) |
| curiosity-followup | `BRAIN_CURIOSITY_LESION` | DRIVER | `"…My curiosity is piqued…"` → bare `"I don't know about that."` |
| pragmatic-implicature | `BRAIN_PRAGMATIC_LESION` | DRIVER | `'Pragmatically I read "some" as "some but not all"…'` → bare abstain |
| comprehension-monitor | `BRAIN_COMPREHENSION_LESION` | DRIVER | `"I don't know about that."` → `"My role-binding didn't resolve…"` |
| other-repair | `BRAIN_COMPREHENSION_LESION` | DRIVER | targeted `"I caught the verb 'carry'…"` → bare `"My role-binding didn't resolve…"` |
| noncontradiction-gate | `BRAIN_NONCONTRADICTION_LESION` | DRIVER | `"That contradicts what I hold… I won't accept"` → accepts `'The wolf huntses deer.'` |
| affect-drives-response | `BRAIN_AFFECT_DRIVES_LESION` | DRIVER | `'Frankly! The dog chases cat.'` → `'The dog chases cat.'` (lead gone) |
| swap-drives-response | `BRAIN_SWAP_DRIVES_LESION` | DRIVER | `'On dog, then — The dog chases cat.'` → base (transition lead gone) |
| da-mode-drives-response | `BRAIN_DA_DRIVES_LESION` | DRIVER | `"…— there's plenty more to dig into here!"` → base (suffix gone) |
| gnw-bus-shadow | `BRAIN_GNW_BUS_LESION` | DRIVER | `'The dog chases cat.'` → `"I don't know about that."` (ignition collapses) |
| self-initiated-utterance | `BRAIN_SELF_INITIATE_LESION` | DRIVER | `"Something's been on my mind — cat eat worm…"` → neutral idle fallback |
| reconsolidation | `BRAIN_RECONSOLIDATION_LESION` | DRIVER | recall `'The dog gos south.'` → stale `'The dog gos north.'` |
| wm-binding-advanced | `BRAIN_MULTIREF_LESION` | DRIVER | `"I'm holding 2 referents…: dog and cat."` → `"…not holding any referent…"` |
| prospective-memory | `BRAIN_PMEM_LESION` | DRIVER | `'(Reminder — you asked me to call mom…'` → no reminder (`'The I gots home.'`) |
| gnw-multistep-deliberation | `BRAIN_GNW_MULTISTEP_LESION` | DRIVER | chain terminal `'The zorp chases munt.'` → `"I don't know about that."` |
| discourse-planner | `rich=False` escape (`BRAIN_RICH`) | DRIVER | multi-sentence rich reply → single-SVO |
| semantic-recall | internal (`composer.query_*`) | DRIVER | the recalled fact IS the answer (`'The dog chases cat.'`) |
| content-selection | internal (`_substrate_recall`) | DRIVER | recall vs honest abstain (`"I don't know about that."`) |
| moat-verify | `BRAIN_CLAIM_MOAT` / core | DRIVER | ungrounded query → abstain, not a confabulated answer |
| in-loop-learning | internal (recall lesion) | DRIVER | before abstain → after `'The wolf hunts deer.'` (taught mid-chat) |
| anaphora-wm | none (host pronoun + `SpikingLoopContextBuffer`) | DRIVER | `'it'` → referent → `'The cat eats fish.'` |
| one-brain-substrate | `BRAIN_COMPOSER_KIND=rf` escape | FEEDER | byte-identical under the rf oracle (a substrate MECHANISM claim) |
| onebrain-merge-organs | `BRAIN_ONEBRAIN_MERGE=0` escape | FEEDER | byte-identical merged-vs-separate (relocates substrate; feeds surprise/wm) |
| discourse-register | `BRAIN_DISCOURSE_REGISTER_LESION` | NOT-CLEANLY-TESTABLE | register never populated on my clauses (`'no earlier event yet'` both arms) |
| episodic-memory | `BRAIN_EPISODIC_LESION` | NOT-CLEANLY-TESTABLE | numpy DEFERS the BTSP store (cupy-gated) → nothing stored, both arms `"I don't recall…"` |
| gnw-deliberation | `BRAIN_GNW_DELIBERATE_LESION` | NOT-CLEANLY-TESTABLE | could not stage a genuine 2-candidate conflict via teach on the tiny-demo |
| causal-whatif | `BRAIN_CAUSAL_LESION` | NOT-CLEANLY-TESTABLE | the default store holds no causal forward-chain facts → abstains both arms |
| open-ended-generation | `BRAIN_GENERATE_CHANNEL=0` | NOT-CLEANLY-TESTABLE | generation abstained on the sparse 5-fact store (no plausible draw) |
| affect-coloring | `BRAIN_AFFECT_LESION` | NOT-CLEANLY-TESTABLE | rich `n_sentences`=2 both arms; the manner rides the cupy Qwen mouth (numpy = template stub) |

## Dead observers: none in the default-on set — but read the caveat honestly

**No faculty in the 31 computed a neural verdict that went nowhere observable.** Every faculty that fired changed the
reply text (the notice-prepend / lead-prepend / suffix-append / short-circuit / abstain organs all move the `answer`
string; the two substrate rows are answer-preserving by design). The one true observe-only faculty is `gnw-thought-swap`
(#77), which attaches only a `gnw_swap` metadata block and never edits the answer — and it is `on_by_default: NO`. Its
default-on DRIVER replacement `swap-drives-response` (#85) prepends the transition lead and is confirmed here.

**Caveat (do not over-read "0 dead observers").** 6 of the 31 are NOT-CLEANLY-TESTABLE on this numpy / default
tiny-demo config: their DRIVER behaviour is documented in the ledger (each row cites its own real-handler lesion
verify), but this independent audit could not reproduce the trigger here — `episodic-memory` (the BTSP write is
cupy-gated so numpy stores nothing), `causal-whatif` (the tiny-demo has no causal-chain facts to forward-simulate),
`discourse-register` (my two clauses never folded into events), `gnw-deliberation` (teaching a second patient did not
stage a ≥2-candidate conflict), `open-ended-generation` (the 5-fact store yielded no plausible draw), and
`affect-coloring`. These are NOT dead observers — they are config/trigger limits — but this audit did not
independently confirm they drive; that credit still rests on the ledger's own artifacts.

**One honest re-classification for the owner.** The ledger lists TWO affect rows. #84 `affect-drives-response` (the
interoceptive graded-ladder LEAD) is a confirmed DRIVER here. But #13 `affect-coloring` (the Gate-B
forthcomingness/manner coloring) could NOT be confirmed as a text-driver on numpy: with a strong induced mood the rich
reply held at `n_sentences`=2 in both intact and lesioned arms, because the forthcomingness did not cross a threshold
and the manner surface is the external Qwen mouth (a template stub without a GPU). So the "affect drives the chat"
claim holds for the #84 lead but the #13 manner/forthcomingness coloring rides a cupy/threshold-sensitive surface my
numpy audit could not exercise — a real caveat on that specific row, not a refutation.

## Next rungs

**No board next-rungs are filed** — the task scopes board items to genuine dead observers, and there are none. The
follow-on is a verification loop, not a coupling fix: re-run the 6 NOT-CLEANLY-TESTABLE faculties on cupy (episodic
with the BTSP store forced; causal/deliberation with their de-risk fixture facts taught; affect-coloring with the Qwen
mouth live) to convert their ledger-cited DRIVER credit into independent confirmation, and specifically settle whether
`affect-coloring`'s numpy no-op is only a stub/threshold artifact or a thinner effect than the row claims.
