---
type: finding
status: live
date: 2026-08-27
mechanism: onebrain-xedge-per-turn-live-plasticity
lane: one-brain/integration/production
seed-waiver: 3 seeds (42/43/44) at numpy-CPU scale — the per-turn credit ATOM is the PART-2 6-seed-GO in-brain self-supervised step (`grow_live_selfsupervised`); this verifies the per-turn WIRING, not a new learning rule. Concurrent brain-load agent bounded the soak.
artifacts:
  - research/findings/raw/_onebrain_xedge_per_turn_3seed.json
runner: research/runners/onebrain_xedge_production.py
builds_on: research/findings/2026-08-27-onebrain-xedge-production-live-learning-GO.md
---

# One-brain PRODUCTION integration PART 3 — the d6-WM->comprehension cross-edge now GROWS PER-TURN DURING REAL
CHAT (0.05 -> ~12.6, one credited step per turn, bounded), and what earlier turns taught SIGNS a later turn's
comprehension decision — the brain LEARNS THROUGH THE CONVERSATION ITSELF. GO (3-seed)

**One-line:** PART 2 grew the cross-edge over a BUILD-TIME curriculum then froze it. PART 3 removes the last
limit: the edge is wired to grow ONE in-brain self-supervised credited step PER REAL CHAT TURN in the live
`brain_reply` path. On a turn where a WM referent is HELD (the d6 multiref organ set the focus) AND comprehension
RESOLVES confidently, the DA-coincidence machinery credits `w{held}->sel_{resolved}` — the gate is OPENED for
exactly that one step then RE-FROZEN (every read stays a frozen forward pass). The credit VALUE + teach DIRECTION
are the substrate's OWN confident spikes; no host label. Behind `BRAIN_ONEBRAIN_XEDGE` +
`BRAIN_ONEBRAIN_XEDGE_LEARN` (both default-OFF), additive, byte-identical-off. NO flip-to-default (owner-gated).

Artifact: `research/findings/raw/_onebrain_xedge_per_turn_3seed.json` (n_go 3/3).

## The per-turn credit wiring

`webapp/server.py brain_reply`, right after the D4 comprehension `judge`, calls
`onebrain_xedge_production.credit_live_turn_from_comprehension(corg, svo)` when comprehension resolved (the
confident path). It derives the held referent's DISCOURSE role IN-BRAIN — the sign of the first noun's per-noun
agent-evidence `a0` off `cp_firing_states` (positional focus, declared residual) — then applies ONE credited step
via `pool.credit_live_turn`. That step reuses PART 2's `_credit_turn_step` atom: read the brain's OWN `amb_read`
resolution, and IFF confident drive `teach_{resolved}` for one `_episode` — OPENING the plasticity gate for just
that episode and RE-FREEZING after, so the sel-WTA margin a live read consumes is never read while the gate is
open. Guards: no-op (returns None, no key added) unless both flags are on, the learn build is per-turn, a focus is
held, and the content commits — so with the flags off the turn is byte-identical.

## The per-turn weight trajectory (seed 42, from the cited artifact)

Teaching the held focus the AGENT role over 24 real turns, `w{w0}->sel_agent` RISES from **0.05** one credited
step at a time — **0.05, 0.69, 1.43, 2.33, 3.18, ... 11.65, 11.92, 12.17, 12.40, 12.62** <!--derived--> — a
gradual per-turn curve decelerating toward the soft bound, NOT a build-curriculum jump. 24/24 turns credited,
final ~12.62, bounded (max **< 20** = `stdp_w_max`, F3). Teaching PATIENT instead grows `w{w0}->sel_patient`
0.05 -> ~12.6 the same way <!--derived-->.

## Load-bearing across the session (a later turn reflects what earlier turns taught)

Read AFTER the taught turns, on content-ambiguous transitives through the REAL production `repair_target`: the
content-cancelled WM-resolved balanced margin (the quantity `_wm_resolved_role` thresholds, edge-attributable) is
**+0.0108** agent-taught vs **-0.0122** patient-taught <!--derived--> (opposite sign, both past eps=0.0093), and
the edge SIGNS the real repair-role decision on **5/5** ambiguous items in each session <!--derived-->. So varying
the taught role across the session flips the LATER comprehension resolution agent<->patient — the later turn
reflects what earlier turns taught.

## Lesion (the per-turn plasticity is the load-bearing element)

Running the IDENTICAL credit path with the gate LEFT FROZEN (the credited episode drives, but no weight can
accumulate): `w{w0}->sel_agent` stays at **0.05**, the balanced margin collapses to **-0.0025** (below eps), and
the edge signs **0/5** repair roles <!--derived-->. `attributable_to` on the taught vs frozen-gate margins is
recorded per seed. So the later-turn shift is OWNED by the per-turn plasticity, not by the credit path running.

## Biology (why open-the-gate-for-one-step-then-freeze)

The per-turn design mirrors the narrow dopamine timing window that gates reinforcement plasticity: DA promotes
spine enlargement only within ~0.3-2 s after glutamatergic input, the temporal-contingency detector that converts
an eligibility tag into a lasting change. External source: Yagishita
et al. 2014, Science 345:1616, doi:10.1126/science.1255514 <!--derived--> (citation, not a measurement). Opening the plasticity gate for exactly the
DA-coincident credited step (sel-firing AND teach-firing -> snc -> DA) then re-freezing is the emulation's read of
that window; a live comprehension READ falls OUTSIDE any such window (no teach drive, no DA), so it never writes
— the frozen-forward-pass invariant.

## Guarded / bounded / moat / byte-identical-off

Bounded by `stdp_w_max`=20 (F3, no runaway). Moat-safe (F4): the cross-edge only touches the repair role on a
content-ambiguous transitive the brain already abstains/repairs on — it never manufactures a fact or flips a
comprehended well-formed item. Byte-identical-off verified: with the flags unset the live hook returns None
(touches no organ), and with a focus NOT held it returns None — no `xedge_live_learn` key, unchanged response.

## Declared residuals (honest — carried from PART 2, not deferred)

1. WHICH discourse each turn presents + WHICH referent is held (positional focus = `CAND_POOLS[0]`) is
   host/teacher-scaffold (environment territory) — the credit VALUE + DIRECTION became in-brain, the curriculum
   has not. 2. Semantic referent->pool binding is host-directed (positional proxy). 3. Verified at numpy-CPU
   3-seed scale (concurrent brain-load agent bounded the soak); the credit atom is PART-2 6-seed-GO.

## Verdict

**GO (3-seed):** the one-brain cross-edge now GROWS PER-TURN during real chat — from W0=0.05, one in-brain
self-supervised credited step per turn (gate opened for that step then re-frozen), bounded (F3) — and what
earlier turns taught SIGNS a later turn's real comprehension decision, lesion-attributable to the per-turn
plasticity. This is the emergent north-star: the brain learns THROUGH the conversation. Default-OFF, additive,
byte-identical-off; NO autonomous flip-to-default. Functional read-outs only; no phenomenal-experience claim.
