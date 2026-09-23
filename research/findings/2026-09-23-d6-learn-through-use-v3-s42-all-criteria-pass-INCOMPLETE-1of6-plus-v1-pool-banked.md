---
type: finding
status: live
date: 2026-09-23
lane: D6-learn-and-grow
mechanism: D6 gate v3 (capability gate), seed 42 of 6, local numpy; plus the banked gate-v1 base-variant pool arms and the flag-off chat-mode byte-identity check
seeds: [42]
verdict: INCOMPLETE 1/6. Seed 42 passes all seven gate-v3 criteria; the other five seeds are staged on the pool. Gate v1 (base variant) pool arms banked: s43 NO-GO on C3, the same host-list familiarity leak as s42. Flag-off chat path byte-identical to the pre-D6 code.
runner: research/runners/d6_learn_through_use_lb.py
artifacts:
  - research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_s42.json
  - research/findings/raw/_d6_learn_through_use_v3/s42_USE_H.json
  - research/findings/raw/_d6_learn_through_use_v3/s42_FREEZE_H.json
  - research/findings/raw/_d6_learn_through_use_v3/s42_ABL_H.json
  - research/findings/raw/_d6_learn_through_use_v3/s42_NOREC_H.json
  - research/findings/raw/_d6_learn_through_use_v3/s42_SHUF_H.json
  - research/findings/raw/_d6_learn_through_use_pool_v1/d6_ltu_base_pool_verdict.json
  - research/findings/raw/_d6_learn_through_use_parity/offpath_parity_chat_vs_main_s42.json
---

# D6 learn-through-use, gate v3: seed 42 passes all seven criteria (1 of 6 seeds, INCOMPLETE)

seed-waiver: this reports ONE seed of the six the v3 gate requires, run first by the gate's registered run order
(`research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md`). It is not a D6 verdict.
The other five seeds are staged on the pool.

## Gate v3, seed 42 (local numpy, all 7 arms, `research/findings/raw/_d6_learn_through_use_v3/`)

The question is whether the reply to a later recall question depends on the synaptic write. Scored by
`score_seed_v3` (`research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_s42.json`).

| criterion | result | what the arms did |
|---|---|---|
| K1 learns | pass | USE_H probe "what does the wolf hunt" recalls `[wolf, hunt, deer]` |
| K2 use-specific | pass | SHUF_H abstains on that probe and recalls `berry` on xprobe; USE_H abstains on xprobe |
| K3 the reply depends on the write | pass | FREEZE_H (same input, same host code, eta=0) abstains, and 'deer' is absent from its reply |
| K4 read off the synapse at probe time | pass | ABL_H (taught block zeroed after the teach turn, mean \|w\| 1.80 -> 0.0) abstains |
| K5 host record inert | pass | NOREC_H (FREEZE_H with the frozen fact's kb record removed) equals FREEZE_H on all 5 turns |
| K6 lesion is write-only | pass | see below |
| K7 null | pass | USE_H == USE_H_REP on every turn; ABL_H.teach == USE_H.teach |

K6 has five parts, all of which held:
- lever read at the probe turn: taught-block mean |w| 1.800197 (USE_H), 0.0 (FREEZE_H), 0.0 (ABL_H);
- the write counter was installed in all three lesion arms, and each recorded zero store writes after the teach turn;
- zero homeostatic-scaling calls in any arm;
- the encode episode was the same: `steps` 208, `phase_lock_steps` 184, `block` 5 in both USE_H and FREEZE_H;
- the teach parse was the same, and d1/d2 were identical in FREEZE_H and ABL_H to USE_H.

**Reported, not scored.**
- The teach ack differs, as the registration predicted: "The wolf hunts deer." (FREEZE_H) vs "the wolf hunts the
  deer" (USE_H). This is the same-turn readback of the write.
- Secondary C7 holds: USE_D (production) equals USE_H on teach/d2/probe/xprobe.
- `attributable_to_write` = 1.0.

**K5 was the open prediction, and it held.** FREEZE_H's probe reply is "I don't know about that. My curiosity is
piqued — I haven't learned about wolf yet: what can you tell me about wolf?". Removing the host record of the frozen
fact changed no turn. The host-list familiarity leak that failed v1 (thread swap, grounded topic, novelty 0.0) is gone
under the read-time view on this protocol. The unrouted readers (`ENGRAM_READTIME_NOT_ROUTED`) did not reach any reply
here. That is measured on this protocol only. Off-protocol, for example the describe path, it is not measured.

**What gate v2 would have said.** FREEZE_H vs SHUF_H still differs, by SHUF_H's " — worth going further here." suffix.
That is the word-exposure effect v2 was registered to fail on. It is not scored in v3.

## Gate v1 (base variant) pool arms, banked

The two base-variant runs still running on pool42 at the start of this round are banked here as their measured
result. They were not re-queued. Scored with `--variant base` over every arm file pulled from pool41 and pool42
(`research/findings/raw/_d6_learn_through_use_pool_v1/d6_ltu_base_pool_verdict.json`):
- **s43: NO-GO on C3 only.** C1, C2 and C4-C7 hold.
- s42, s44, s101, s102: UNDEFINED, because arms are missing. The first pool41 pass left them unbuilt, and the resume
  lines are no longer in the pool queue. They are not re-queued: v1 is superseded, and s42's base result already
  exists locally (`research/findings/raw/_d6_learn_through_use/d6_ltu_s42_smoke.json`, NO-GO on C3).
- **s100: UNDEFINED at harvest.** Its USE_D arm was still running.

At both s43 and s100, the FREEZE_H probe is "Setting the held thread aside — On wolf, then — I don't know about that."
with curiosity novelty 0.0. SHUF_H has novelty 0.9701. That is the s42 host-list familiarity leak, replicated on two
more seeds. The v1 C3 failure is a property of the base variant, not a seed accident. The read-time view (v3 flags)
is what removes it at s42.

## Flag-off chat path: byte-identical to the pre-D6 code

`research/findings/raw/_d6_learn_through_use_parity/offpath_parity_chat_vs_main_s42.json` holds one full tiny-demo
brain through `/api/brain-chat`, with every BRAIN_D6_* flag unset. It compared revision `f97b339b2` with the pre-D6
tree `c9b45a30e`, in separate processes on pool41. Result: byte_identical=true, diff_keys=[]. All five turns' full
response-body sha256 are equal, and the final store_conns sha256 is `f94a1c65…` in both.

The D6 production hunks are the same at HEAD. The +/- lines of `git diff c9b45a30e f97b339b2` and
`git diff 5e9a7955b HEAD` over webapp/, brain_chat_tui.py and one_brain_composer.py are identical, 72 lines.
Store mode was already identical, and the pinned-SHA test re-checks it on every run.

## Honest scope

- One seed. D6 is not met until all 6 seeds are defined and pass.
- One learning pathway: in-conversation declarative fact acquisition, on the tiny-demo brain.
- The host shortcuts (a)-(j) in `research/runners/d6_hebbian_store.py` are declared, not replaced. They include:
  - the host-wired instructive pathway;
  - the host phase-lock loop;
  - the W_MAX clamp, so only the phase is learned;
  - the held-threshold;
  - the four-reader read-time view;
  - the block-to-words map.
- Everything is default-OFF. No production default was flipped.
