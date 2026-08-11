---
type: finding
status: go
date: 2026-08-10
mechanism: teacher-loop-plasticity-acquisition + learned-familiarity-source-monitor-moat, wired into the live conversational loop
lane: E-language / INTEGRATION
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/plasticity_facts_live_chat_6seed.json
runner: research/runners/_teacher_loop_facts_into_live_chat_derisk.py
instrument: the live multi-turn chat (`_conversation_turing_test_derisk.HUMAN_TURNS` + teacher probes) over `_stageA_full_integration_derisk.build_one_brain`, with the grounded-content path routed through `AcquiredReadComposer` (from `_teacher_loop_contrastive_familiarity_moat_derisk`) so `comp.query_patient` consults an e-prop-learned readout gated by a learned familiarity/source-monitor gate; SIM_BACKEND=numpy; cfg.seed-controlled.
---

# INTEGRATION #7 — the live chat answers about facts the brain LEARNED BY ITS OWN SYNAPTIC PLASTICITY (learned moat), 6/6 GO — the emergence-bar burn-down of #6's host-`comp.store` scaffold

INTEGRATION #6 gave the live chat breadth by HOST-INJECTING mined corpus facts via `comp.store` (a VSA write — a declared
"composer-as-idealization" scaffold; the brain did not learn them). The EMERGENCE BAR (CLAUDE.md, the standing priority)
requires facts the brain LEARNS FROM EXPERIENCE by its own synaptic plasticity. #7 replaces the injection, at demo scale:
the brain is TAUGHT 3 facts by corrective interaction → the fact becomes an **e-prop weight change on a spiking
Izhikevich readout** → the live chat answers about them, with a **LEARNED familiarity/source-monitor gate** as the
no-confab moat for those facts. The two component mechanisms were each a standing 6-seed GO in isolation
(`2026-08-08-teacher-loop-corrective-acquisition-*` acquisition; `2026-08-08-teacher-loop-contrastive-familiarity-moat-*`
moat-closure); this is the NOVEL composition of both into the sustained multi-turn chat (the relationship #6 had to its
pieces — no prior `INTEGRATION #7`, no prior runner built both `build_one_brain` and `AcquiredReadComposer`).

## The wire-in (additive, runner-side, NO `sim/` edit)

<!--derived-->

Every grounded-content decision in the live chat routes through ONE method, `comp.query_patient(agent, action)` (used by
`_classify` routing, `_gm_retrieve_neighbourhood` retrieval, and the post-hoc no-confab moat `_gm_posthoc_verify`). The
plasticity-learned fact lives in the e-prop readout weights of an `OnBridgeEpropNet`, NOT in `comp.kb`, so a raw
`query_patient` abstains. The adapter `AcquiredReadComposer` wraps `comp` and overrides `query_patient`: (1) the
structural VSA kb moat first (byte-identical for the host-stored curated facts + when disabled); (2) on a kb abstain,
the learned e-prop read GATED by the learned familiarity gate (`if not fam.familiar(env,a,v): return None; else return
PATIENT_WORDS[predict_settled(net,env,a,v)]`). A ~2-line `__getattr__` passthrough makes the shim a drop-in `comp`, so
ONE wrap propagates through classification + retrieval + the post-hoc moat automatically. The K=3 facts
(`{dax:grass, dog:bone, cat:fish}`) are taught JOINTLY, one pass, to one `OnBridgeEpropNet` + `ConjunctiveFamiliarityGate`.

## Result — 6/6 GO (artifact `research/findings/raw/lanes/stageA/plasticity_facts_live_chat_6seed.json`, verdict GO, n_go 6/6; coordinator-verified from the raw log)

<!--derived-->

**On the discriminating metrics (not the saturated summaries).** The verdict rests on the treat-vs-frozen CONTRAST and
the varying controls, NOT on any single ceiling/floor field: `recall_frozen`=0 and `recall_attributable_to_weight_change`=1.0
are exactly the CLEAN-CONTROL result (freezing the readout kills recall on every seed — the control WORKING, not a
saturated instrument), and their discriminating power is the pair `recall_treat`=3 **vs** `recall_frozen`=0. The metrics
that genuinely VARY across seeds and carry the abstain/specificity verdict are the familiarity-gate lesion margin
(0.78→0.85, collapsing to 0.00), gate-OFF false-accepts (4–5), held-out recall (0.90–1.00), and mispaired-teacher
(0.00–0.01). Per seed (42/43/44/100/101/102), in the LIVE chat after plasticity-teaching:

| metric | all 6 seeds |
|---|---|
| taught-recall base→treat | **0 → 3/3** |
| taught-recall with FROZEN readout | **0** (content rode the weight change) |
| grounded-reply delta | **+1** |
| moat false-accepts (untaught cues + OOD turns) | **0** |
| held-out recall (dax) | 0.90–1.00 |
| familiarity-gate lesion (novelty margin) | 0.78–0.85 → **0.00** |
| gate-OFF false-accepts | 4–5 (gate is load-bearing) |
| mispaired-teacher vs main | 0.00–0.01 **<** 0.94–0.99 |
| byte-identity (shim OFF vs #6 default) | **fully identical** (threshold hash, concept codes, 25531 neurons, decision transcript) |

The brain answers about facts it learned BY ITS OWN PLASTICITY (taught-recall rises via the weight change), the learned
familiarity gate holds the no-confab moat at chat scale (0 false-accepts on untaught cues AND the out-of-domain
HUMAN_TURNS), and the shim is byte-identical when off.

## Anti-cheats — each is load-bearing and passes on all 6 seeds

<!--derived-->

- **FROZEN-READOUT** (the key control): teaching with `eprop_lr=0` (readout frozen, `frozen-moved=0.0000`) leaves the
  taught cues **unanswerable** (recall 0) → the chat's new content genuinely rode the e-prop **weight change**, not a
  host path or cache.
- **kb-unchanged**: the learned facts are absent from `comp.kb` (`kb_len` unchanged); the answer comes from the net's
  forward record, not `comp.store`.
- **LESION-gate**: silencing the learned familiarity projector (`fam.lesion()`) collapses the novelty margin
  (0.78–0.85→0.00) and untaught cues start answering — the abstain **rides the learned projector**. Gate-OFF still
  false-accepts (4–5) → the gate is load-bearing, not redundant with the kb moat.
- **mispaired-teacher**: the answer is the teacher's SPECIFIC pairing (0.00–0.01 for a mispaired target vs 0.94–0.99 for
  the taught one).
- **byte-identity**: shim `enabled=False` reproduces the #6 default build's decision transcript exactly.

## Honest scope — what burns down vs the declared residual scaffolds (per THE LAW)

<!--derived-->

**Genuinely brain-based now (the emergence-bar win #6 lacked):** the FACT ACQUISITION is a synaptic weight change in a
spiking Izhikevich substrate (e-prop moving `cp_connections`); the MOAT DISCRIMINATION (which cue → which patient, and
the novelty abstain) is the e-prop net's own spiking readout + the learned familiarity gate.

**Declared burn-downs remaining (named, not deferred):**
1. **Two bridges, not one.** `OnBridgeEpropNet` builds its OWN `SimulationBridge`, co-resident with the conversational
   `build_one_brain` bridge (both Izhikevich, one process, numpy) but NOT merged. Brain-based (neurons/synapses) but not
   yet ONE-brain — the merge (an e-prop-plastic readout slice INSIDE the conversational bridge) is the next step.
2. **The familiarity gate is host-idealized.** `RealAntiHebbianFamiliarity` is a numpy anti-Hebbian (Bogacz-Brown /
   perirhinal) projector — correct in MECHANISM (learned-by-imprint, lesionable), host in IMPLEMENTATION. The spiking
   realization `familiarity_gate_v320` exists to swap in.
3. **Conjunctive cue codebook + patient argmax read-out** — the composer-idealization + neural-motor-read-out targets,
   same status as #6.

**Legitimately host (environment/body):** the teacher/curriculum (the AI-teacher social environment) and the generator
mouth (Broca scaffold, OFF in this mouth-free tier).

**Scale reality (the honest ceiling).** This is a SMALL-K (3), JOINTLY-taught demo standing BESIDE #6's host-stored
breadth, NOT replacing it wholesale. Multi-fact continual/sequential acquisition (learn fact-after-fact across turns
without forgetting) is an OPEN arc, not GO (`frac_recalled ~ 1/N`; sleep-replay consolidation PARTIAL; sparse-gated
readout NEGATIVE; neurogenesis PARTIAL). The scale-up to #6's K=40 breadth is explicitly gated on that continual-learning
arc reaching GO — the named next mechanism.
