---
type: finding
status: wired
date: 2026-08-13
mechanism: causal why/what-if forward-model organ wired into the default /api/brain-chat turn (T1-4)
verdict: WIRED (default-ON, real-handler verified numpy-CPU; co-resident forward-model bridge, grounding-by-derivation)
lane: T1-4 · Learned causal forward model — production wiring
artifacts:
  - research/findings/raw/_causal_whatif/production_verify.json
verification: >
  Verified through the REAL webapp.server.brain_chat handler on the production tiny-demo ChatBrain (rf recall,
  SIM_BACKEND=numpy): what-if rolls A=(dog,go,east)->B=(dog,reach,river)->D=(dog,drink,water) and surfaces the
  moat-CONFIRMED consequence "it will drink water" (A->D never taught — a substrate rollout); why reads the
  DO-surviving cause C=(sun,rise,sky) for Y=(dog,wake,morning) and NOT the spurious correlate X=(bird,sing);
  an unmapped causal query AND a grounding-unconfirmed query (chain taught but not D) both ABSTAIN to the honest
  _honest_causal_answer disclaimer with 0 confabulation; BRAIN_CAUSAL_LESION=1 (forward edges zeroed) collapses
  BOTH why + what-if to the abstain; BRAIN_CAUSAL=0 is byte-identical on a recall/abstain panel (flag on==off) and
  a causal query flag-off carries no `causal` key; the canonical brain_chat_tui --smoke is byte-identical with
  webapp/server.py stashed vs present. tools.verdict.Verdict -> GO. Every surfaced fact is a composer.query_patient
  moat read; every prediction/cause a cp_firing_states spiking read.
---

# The causal WHY / WHAT-IF organ, WIRED into the default /api/brain-chat turn (T1-4, 2026-08-13)

## Result
The grounded causal forward model (`2026-08-13-grounded-causal-forward-model-real-fact-why-whatif-6seed-GO`, 6/6
GO — a real-fact "why did X" / "what happens if X" over the brain's learned facts by forward-simulation + a Pearl
DO-probe) is now WIRED, default-ON, into the production conversational turn (`webapp/server.py::brain_chat` ->
the `ChatBrain`). It answers the reasoning rung a host triple-JOIN cannot serve:
- **What-if** (verbatim, real handler): *"If the dog goes east, it will drink water — a consequence I rolled
  forward through dog reaching the river, and my no-confab moat confirms (dog,drink)->water is a fact I stored."*
  D fires via B though A->D was NEVER taught (a substrate rollout, not a recalled edge).
- **Why** (verbatim, real handler): *"The dog wakes (morning) because the sun rises — that cause survives a
  DO-probe (forcing the sun to rise makes the dog wake; forcing the bird to sing does NOT), so it is a cause not
  a mere correlation, and (sun,rise)->sky is a fact I stored."* The cause is C, never the correlate X.

New: `research/runners/causal_whatif_production_organ.py` (reuse-by-import of the grounded de-risk's mechanism +
its toy primitives — NO reimplementation, NO `sim/` edit). Handler block in `webapp/server.py::brain_chat`
(after the affect/episodic/worldmodel/multiref/discourse read-outs, before comprehension/surprise/rich). Verify:
`research/runners/_causal_whatif_production_organ_verify.py` -> `research/findings/raw/_causal_whatif/production_verify.json`.

## The wiring (the finding's 4-step spec, realized)
1. **Built once per brain (lazy).** On the first why/what-if turn, the ~180-neuron directed forward-model bridge
   is built and its event set + causal curriculum are GATED by the LIVE composer's moat recall
   (`enumerate_events` over `chat.inner.composer.query_patient`) — READ-ONLY grounding: the organ never writes a
   fact, it grounds against WHATEVER the brain already learned. Edges trained by temporal-order STDP + phasic-DA.
2. **What-if** — HOLD the chain source, roll the substrate forward, read the spiking successor, map it to a fact,
   and CONFIRM via `query_patient`. Emit ONLY when confirmed.
3. **Why** — read the directed edge INTO the target as the argmax DO-probe predecessor, confirm the cause via
   `query_patient`, and confirm it SURVIVES the DO-probe. Emit ONLY when confirmed.
4. **Moat-safe by construction** — an unconfirmed / unmapped causal query ABSTAINS to the honest
   `_honest_causal_answer` disclaimer (the INTEGRATION #5 fallback); 0 confabulation across every abstain path.

## Verification (real handler, numpy-CPU) — all GO
- **WHAT-IF** moat-confirmed consequence (dog->drink water); **WHY** DO-surviving moat-confirmed cause (sun rose,
  not bird sang). **ABSTAIN(0 confab)**: an unmapped causal query ("why did the dog chase?" — a known fact, not
  the validated target) and a grounding-unconfirmed query (chain taught but NOT D) both decline honestly, neither
  asserting an unconfirmed causal fact. **LESION-load-bearing**: `BRAIN_CAUSAL_LESION=1` (forward edges zeroed)
  collapses BOTH why + what-if to the abstain — the answers are caused by the learned SPIKING edges.
  **BYTE-IDENTICAL-when-off**: a recall/abstain panel is identical flag on==off + a causal query flag-off carries
  no `causal` key + the canonical `brain_chat_tui --smoke` is byte-identical with `webapp/server.py` stashed vs
  present. Default-ON; `BRAIN_CAUSAL=0` -> fully skipped (byte-identical oracle).

## Honest residuals (declared — the named next rungs, per THE LAW + docs/TERMS.md)
- **Grounding-by-DERIVATION, not shared-substrate-merge.** The events are DERIVED from + gated by the LIVE
  composer's moat recall (and the answers re-confirmed by it), but the composer's unbind SPIKES do not yet
  directly DRIVE the forward-model event blocks in ONE merged bridge. The organ runs on its OWN co-resident
  forward-model bridge alongside the recall composer (rides the one-brain merge, burn-down #1), as the affect /
  surprise / world-model organs do. This is why the row is `wired`/`on_by_default` but NOT `scaffold_retired`.
- **The DA sign + causal episode ORDER are teacher-delivered** (the environment boundary); a spiking mismatch
  unit driving the DA is the next rung.
- **First-order + fixed causal STRUCTURE.** state->next (Markov-1; high-order needs HTM-TM); the canonical
  CHAIN/CONFOUND structure is teacher-rendered. The wired SCOPE is the validated chain-source what-if (A->D) +
  confound why (Y<-C); a why/what-if outside that grounded structure abstains honestly.

## Provenance
`research/runners/causal_whatif_production_organ.py` (reuse-by-import of `_causal_forward_model_grounded_derisk`
+ `_causal_forward_model_derisk`; NO `sim/` edit). Handler block in `webapp/server.py::brain_chat` (`BRAIN_CAUSAL`
default-ON, `BRAIN_CAUSAL_LESION` load-bearing). Verify `research/runners/_causal_whatif_production_organ_verify.py`
-> `research/findings/raw/_causal_whatif/production_verify.json` (GO).
