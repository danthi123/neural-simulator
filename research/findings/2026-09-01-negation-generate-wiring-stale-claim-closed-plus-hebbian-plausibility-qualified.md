---
type: finding
status: qualified
date: 2026-09-01
integration_faculty: open-ended-generation
mechanism: (A) VERIFICATION that the #3E generate-channel non-contradiction gate is NOT inert on the
  production onebrain composer for its designed (non-colliding) regime -- closing a STALE ledger claim
  dated 2026-08-13, before the 2026-08-25 store-side negation wiring landed; (B) a NEW, precisely-scoped
  residual this session's testing surfaced (`OneBrainComposer.ask_yes_no`'s one-block-per-(agent,action)
  design cannot recall a SECOND patient for an already-claimed (agent,action) pair, polarity-independent);
  (C) HebbianAssociativePlausibilityOrgan -- an honest QUALIFIED (not adopted) lever converting the #3E
  plausibility organ's synapse weights from a host `P*gain` injection to Hebbian-GROWN weights (replaying the
  brain's own stored facts), the "next rung" named by the 2026-09-01 ensemble finding
verdict: NO NEW PRODUCTION FLIP. (A) is a CLOSED verification (the ledger's stale claim is corrected, no code
  changed). (C) is QUALIFIED -- built, lesion-load-bearing, agreement 0.71-0.75 on a synthetic 4-fact graph
  (2 seeds), below the 0.95 host-parity bar the 2026-09-01 ensemble organ reached -- NOT wired, NOT
  default-on, an honest partial result banked for the next session.
lane: integration-spine / open-ended-generation -- VERIFY-FIRST + honest MAP of the remaining residuals
artifacts:
  - research/findings/raw/_hebbian_plausibility_derisk_2seed.json
verification: >
  (A)/(B): research.runners.brain_chat_tui.ChatBrain built with composer_kind="onebrain" (the production
  default), seed 42, SIM_BACKEND=numpy. Three probes through the REAL production call path: (1)
  `composer.hear("cat chase fish", polarity="NEGATE")` direct on a collision-free (agent,action) pair ->
  `ask_yes_no("cat","chase","fish")` reads "no" (correct) -- the substrate mechanism works. (2)/(3)
  `chat._maybe_acquire("the dog does not eat grass")` (the PRODUCTION text-acquisition path, wired since
  aaf1ad5bc 2026-08-25) on (agent,action) pairs that COLLIDE with an already-stored fact ("dog chase cat",
  "dog eat bone") -> `ask_yes_no` reads "unknown", not "no" -- traced to `OneBrainComposer._seq_block`'s
  documented first-match-per-(agent,action) selection (`one_brain_composer.py:1387-1398`), a PRE-EXISTING,
  polarity-independent architectural property ("a degenerate same-(agent,action) different-patient pair is
  outside the production regime" -- the method's own docstring), not a negation-storage defect.
  (C): research/runners/_hebbian_plausibility_derisk.py, seeds 42/43, a synthetic 4-fact/8-concept graph
  (no ChatBrain needed -- the organ only reads P/row/facts/vocab).
---

# The stale "onebrain negation inert" ledger claim is CLOSED for its regime; a narrower, real residual replaces it; the plausibility gate's Hebbian self-organization rung is QUALIFIED, not yet at parity

## What this session actually found (read this before the two sub-results)

The assignment was to retire the NEXT host-computed shortcut in the live GENERATE path, following the
2026-09-01 plausibility-ensemble pattern. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s open-ended-generation
row (line 91-92, last touched 2026-08-13) named four remaining residuals; two turned out to already be
handled by prior work once checked, one ("onebrain negation unstored") was PARTIALLY stale, and pursuing the
purity rung on a THIRD (the plausibility organ's still-host-set synapse weights) produced a real, working, but
sub-parity mechanism. None of this session's candidates reached a clean default-ON GO -- the honest map below
is the deliverable, per the task's own instruction ("if none is cleanly ready... commit the map").

## (A) The non-contradiction generate-channel gate is NOT inert on onebrain -- for its designed regime

The ledger line (blamed to commit 7b16e0ba2c, 2026-08-13) says: "onebrain negation unstored so the
non-contradiction gate is inert there." This was TRUE on 2026-08-13. It stopped being true on 2026-08-25
(commit `aaf1ad5bc4`, "reasoning-frontier(1/3): verb lemmatization on the in-loop store/query path"), which
also wired `ChatBrain._maybe_acquire` to route every heard assertion through the B3 organ's
`extract_polar_assertion` (detects a negation cue, strips it, returns `(agent, action, patient, "NEGATE")`)
and store it via `self.inner.hear(sentence, polarity=pol)` -- the SAME `composer.hear` API the 2026-08-12 B3
de-risk (`_burndown_B3_onebrain_negation_moat_derisk.py`, 6-seed GO) already validated stores + recalls
polarity correctly on `OneBrainComposer`. The ledger line was simply never moved when that landed -- the
documentation-drift failure mode this project's own `sync-documentation` skill exists to catch.

**Verified this session** (probe 1 above): teaching a collision-free negation ("cat chase fish", never
before stored) through the composer's own `hear(..., polarity="NEGATE")` correctly makes
`ask_yes_no("cat","chase","fish")` return `"no"`. Since `brain_chat_tui.py::_generate_hypothesis`'s
`_contradicts(a, ac, p)` is `self.composer.ask_yes_no(a, ac, p) == "no"` (`_genfrontier_b2_generative_replay_
derisk.py:201`) and `_build_generation_proposer` installs the proposer on `self.inner.composer` -- the SAME
composer instance the chat answers through (`brain_chat_tui.py:822`) -- a negation taught through the normal
production conversation DOES reach the generate-channel's non-contradiction gate on the production onebrain
composer. **The ledger's "inert" claim, as a blanket statement, is false as of 2026-08-25.**

## (B) The REAL residual this exposed: `ask_yes_no` recalls only ONE patient per (agent, action)

Probes 2 and 3 (through the production `_maybe_acquire` text path, not the raw composer API) both read
`"unknown"` instead of `"no"` after teaching a negation -- but both probes happened to pick an (agent,
action) pair that ALREADY had a stored fact ("dog","chase") from "dog chase cat", and ("dog","eat") from a
just-taught "dog eat bone". `OneBrainComposer.ask_yes_no` calls `_seq_block(agent, action)`
(`one_brain_composer.py:1387`), whose host fallback (and, per its own comment, the spiking sequencer path
too) is a **first-match lookup keyed on (agent, action) alone** -- "each (agent, action) selects ONE block."
When a SECOND fact (here, the negation) targets an (agent, action) pair a FIRST fact already claimed, `_seq_
block` still returns the FIRST block's index; `ask_yes_no` then compares that block's ORIGINAL patient
("bone") against the query's patient ("grass"), finds a mismatch, and returns `"unknown"` -- never reaching
the polarity check at all. The method's own docstring names this explicitly: *"a degenerate same-(agent,
action) different-patient pair is outside the production regime."* This is **polarity-independent** (an
AFFIRM-AFFIRM collision, e.g. re-teaching "dog eat meat" after "dog eat bone", hits the identical branch) --
it is a pre-existing scope boundary of the one-block-per-cue store design, not a defect this session's arc
introduced or the negation-wiring work left open. **This is the residual to name going forward**, replacing
the stale "negation unstored" claim: *ask_yes_no cannot distinguish a re-asserted/negated patient for an
already-claimed (agent, action) key.* Lifting it needs a multi-value store per (agent, action) with
polarity-aware selection at read time -- a real, but separately-scoped, architecture change (not attempted
this session; named here so it is not re-derived).

## (C) HebbianAssociativePlausibilityOrgan -- built, lesion-load-bearing, QUALIFIED (not adopted)

The 2026-09-01 ensemble finding named its own honest residual: *"the synaptic weights are still SET from the
co-occurrence counts... online Hebbian self-organization of those weights remains the next rung."*
`research/runners/hebbian_plausibility_organ.py` (`HebbianAssociativePlausibilityOrgan`, subclasses
`SpikingAssociativePlausibilityOrgan`, reuses `related`/`_self_threshold`/`install`/`uninstall`/
`agreement_with_host` UNCHANGED) replaces the host `set_pathway_weights(..., weights=P*gain)` injection with
**REPLAY**: the `cortex_ctx -> dlpfc_wm` synapses start at 0 and are grown by the bridge's own validated
Hebbian co-fire rule (`cfg.enable_hebbian_learning=True`, the same mechanism `_D_sparse_heteroassoc.py` /
`LearnedAssocGraph` already use, 24/24 edges + 9/9 top-associate match vs the Python oracle) while co-driving
each stored fact's three role-concepts' cortex AND dlpfc assemblies together. A recurring co-occurrence
(shared by >1 fact) is replayed again -> growth ACCUMULATES, the substrate analogue of a higher host count.
No backward `dlpfc_wm -> cortex_ctx` pathway is ever declared, so the read stays strictly monosynaptic (the
same property the 2026-09-01 ensemble organ's `density=0.0` achieves, here by construction rather than a
zeroed weight).

**The tuning arc (an honest account, not just the final number).** The first build used full inter-region
connectivity (`inter_density=1.0`, mirroring the ensemble organ's dense per-pair injection) and reached
agreement 0.55 -- barely above the ~0.43 base positive rate. Diagnosis (a standalone debug harness driving
raw `sim.regions`/`sim.bridge` primitives, not the organ): with EVERY (a,b) assembly pair guaranteed a synapse
bundle regardless of true co-occurrence, sheer CONVERGENT SUMMATION from many weakly-grown synapses onto the
same postsynaptic neurons swamped pairwise specificity -- a high-degree concept fired almost everything
downstream near-uniformly. This is the SAME class of failure the ensemble arc's internal-recurrence
`density=0.0` fix targeted, one level up: there it was recurrence WITHIN a region contaminating the read;
here it was FULL density BETWEEN regions contaminating which pairs could grow a synapse at all. Also fixed
along the way: `cfg.enable_stdp` was left at its default `True` (STDP's eligibility trace needs
`runtime_state.current_time_ms` advanced every step -- the project's own documented "STDP IS INERT" trap --
which the replay loop never did); disabled outright since only plain Hebbian co-activity growth is used here.
Dropping `inter_density` to a SPARSE regime (0.08, `n_ensemble=4`, `pattern_size=8`, `replay_cycles=100`,
`hebbian_learning_rate=0.4`) restored specificity (`research/findings/raw/_hebbian_plausibility_derisk_
2seed.json`, rows[].agreement/f1/precision/lesion_shuffle_agreement): **agreement 0.7142857142857143 (seed
42) / 0.75 (seed 43)**, f1 0.6521739130434783 / 0.6818181818181818, precision 0.6818181818181818 /
0.75 -- well above the shuffle-lesion control (0.4642857142857143 / 0.6607142857142857) and the ~0.43 base
rate (24 of 56 ordered concept pairs are host-related), but below the 0.95 parity bar the 2026-09-01
ensemble organ (a DIFFERENT mechanism -- host-injected weights) reached at 1.00.

**Lesion-load-bearing (both seeds):** `lesion="ablate"` (never replay) -> 0 related pairs (nothing ever grew
above the read threshold). `lesion="shuffle"` (replay a role-shuffled fact list, same replay VOLUME, destroyed
co-occurrence structure) -> agreement drops below the intact organ's, both seeds. **Provenance:** the
synapse VALUE never touches a host formula -- `_build_bridge` never calls `set_pathway_weights` at all; every
weight is the residue of `_run_one_simulation_step`'s own Hebbian update after real co-firing.

**Honest verdict: QUALIFIED, NOT adopted.** Below the parity bar on the (tiny, 2-seed) evidence gathered this
session -- not wired into `_generate_hypothesis`, no master-switch flag added, no production change. This is
the SAME status class as the 2026-09-01 finding's own "GRADED sub-lever: built, measured, NOT adopted" --
a real, working, lesion-clean mechanism whose operating point needs more tuning (or possibly a further
lever, e.g. per-fact replay-cycle count scaled to graph size, or a WEIGHT-space ensemble average rather than
firing-fraction) before it can candidate for a parity-level default-ON claim. Banked here rather than forced.

## Reproduce

```bash
# (A)/(B) -- rebuild the same probes (heavy: ~5-9 min wall clock, full onebrain ChatBrain build)
SIM_BACKEND=numpy NEURAL_SIM_DISABLE_LLM=1 PYTHONPATH=. .venv/bin/python -c "
from research.runners.brain_chat_tui import ChatBrain, _build_tiny_demo
agent, aliases, n = _build_tiny_demo(seed=42, use_multiturn=False, enable_neural_render=False, composer_kind='onebrain')
chat = ChatBrain(agent, self_aliases=aliases)
print(chat.inner.composer.hear('cat chase fish', polarity='NEGATE'))
print(chat.inner.composer.ask_yes_no('cat','chase','fish'))  # -> 'no'
"

# (C) -- the Hebbian plausibility organ, cheap (synthetic graph, no ChatBrain), multi-seed. The 2-seed
# artifact this finding cites is research/findings/raw/_hebbian_plausibility_derisk_2seed.json; a wider
# seed set (not yet run) would land at a NEW path of the runner's choosing, e.g. --out <your-path>.json
SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -u -m research.runners._hebbian_plausibility_derisk \
    --seeds 42,43,44,100,101,102
```

## Next actions (named, not re-derived)

1. Update `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s open-ended-generation row: drop the stale "onebrain
   negation unstored" claim, drop the stale "plausibility LIKELIHOOD is host-computed" claim (RETIRED by the
   2026-09-01 ensemble finding, default-ON), add the precise `_seq_block` one-patient-per-(agent,action)
   boundary (B above), note the Hebbian organ as a QUALIFIED next-rung lever (not yet adopted).
2. The `_seq_block` one-block-per-(agent,action) boundary (B) is the cleanest NEXT candidate for a future
   arc if the generate/assertion channels need to handle re-assertion/retraction correctly -- it currently
   silently `"unknown"`s rather than raising, so it fails SAFE (no confab) but not usefully (a real
   correction is invisible to the moat).
3. The Hebbian organ (C) needs a 6-seed run (queued to the pool, command above) before any parity claim,
   and likely one more tuning lever (candidates named in the finding body) to close the ~0.25 agreement gap
   to the ensemble organ's 1.0.
