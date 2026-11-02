---
type: preregistration
status: live
lane: A; plan step 3 of docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md
date: 2026-11-01
mechanism: bg-live-seam-retirement, amendment 2 (stn200_gpe100), code re-pin only
amends: research/findings/2026-11-01-bg-live-seam-gate-AMENDMENT-stn200-gpe100.md (pin only; nothing else)
seeds: [42, 43, 44, 100, 101, 102]
verdict: RE-PIN ADDENDUM. No bar, arm, turn, schedule, env flag or verdict rule changes. No gate seed has run.
runner: research/runners/host_share_bg_seam.py
---

# Addendum: the BG gate amendment's code pin moves to a merge of main

host-decision: none. This addendum changes only which commit the pool runs; every decision stays where the signed
amendment put it (heard/hold -> D1 race, GPi/thalamus gate, spiking commit stage).

bar-sanity: N/A no bar is registered or changed here.

## 1. Why the pin moved

The server provisioner refused the signed pin `218a3378c873ea38415a6aa7f62effc21b410ff1` (receipt
on S, copied to `research/coordination/bg-gate-1101-repin/provision_receipt_218a3378c.json`, state `refused`):
"stale: does not contain published main 1bfe466358b9 ... and main changed more than docs since". The freshness rule
needs a pin that contains published main. So S main `cc436f278` was merged INTO `research/bg-gate-amend-1101`.

- Old pin: `218a3378c873ea38415a6aa7f62effc21b410ff1` (branch base on main: `0814c24b0`, 10-29).
- New pin: `e017eb00e38e69d230a7d930e181049d7822160b` (the merge commit; parents `e5ac1c5fc` and S main `cc436f278`).
- Pool lines: `research/coordination/bg-gate-1101-pool-lines.tsv`, regenerated with the new pin and out roots
  `e017eb0_preset` / `e017eb0_tonic0` / `e017eb0_hybrid` (same layout). Diffed against the old TSV with the pin
  masked: identical. All 24 lines pass `tools/canary_preconditions.py check-line`.

Conflicts: `research/coordination/lane_builds.jsonl` (append log, both sides kept) and the amendment finding itself
(main's signed copy kept). No BG file changed on main since the branch base, so the branch's BG code is the BG code at
the new pin unchanged.

## 2. What main brought in (329 commits since `0814c24b0`)

Code (non-docs) that could reach a BG harness turn:

| Change | Default | Reaches the BG harness? |
|---|---|---|
| C3 topic stop + topic-locked phasic DA (`41d76bdbe`, `webapp/curiosity_topic_stop_chat.py`) | **ON** (new) | Only the curiosity follow-up on an ABSTAIN with an extractable topic (see section 3) |
| #203 plastic-mask enforcement beyond Hebbian (`sim/bridge.py`, `sim/config.py`, `BRAIN_ENFORCE_PLASTIC_MASK`) | OFF | No (flag unset in every pool line) |
| Megakernel guard completeness (`a543af247`) | n/a | No: the cupy fused step only; the pool lines run `SIM_BACKEND=numpy` |
| Divisive norm external-only (`input_divisive_norm_external_only`) | OFF | No |
| C2 composer patient-inhibition / role WTA / row shuffle (`one_brain_composer.py`) | OFF | No |
| Tools (canary_preconditions, pool scripts, pi_driver, memcap hook) | n/a | No runtime effect on the brain |

`BRAIN_EMOTION_CATEGORY_BOUNDED` ON (`0b79336ec`, 10-28) was already in the branch base, so the old pin had it too.

## 3. Default-change analysis

The harness runs the live `webapp.server.brain_chat`. Its base env (`research.runners.host_share.BASE_ENV`) does not
pin `BRAIN_CURIOSITY_TOPIC_STOP` or `BRAIN_CURIOSITY_PHASIC_DA`, so both read ON at the new pin. Code path:

- The BG seam decision (`decide_action_neural` / `decide_action`, `webapp/server.py` ~5743) runs BEFORE the curiosity
  follow-up (`_curiosity_followup`, ~7496 / ~7755), so C3 cannot change a turn's BG decision fields.
- C3 acts only inside `_curiosity_followup`, only on an abstain with an extractable topic, and only decides whether
  the honest follow-up question is appended. Its organ is per-conversation (`Context`), seeded, and reads the DA level
  without writing it. It touches no BG organ state, so it cannot reach a later turn's BG decision either.
- It applies identically in every arm (prod and seam), so an arm-vs-arm reply diff is unaffected unless the seam
  itself changes the abstain. The C3 flip gate already showed answers identical ON vs OFF on six gate seeds.

Expected effect: none on BG decision fields; at most a follow-up-question suffix on an abstaining `hsb_q`, in every arm alike. Measured (section 4): `hsb_q` answers on dev seed 79, so C3 never engaged and the 4-turn run is byte-identical to the old pin.

## 4. Smoke (dev seed 79, `seam_b`, `--commit`, pinned gate env, numpy)

| Run | Turns | Result vs the old pin |
|---|---|---|
| 1-event (`hsb_empty1`) at the new pin vs the old pin, same env | 1 | responses, events and env **byte-identical** (JSON); winner 1 (STAY_SILENT), commit winner 1, read_invariant True, co_commit False, latch onset 157 ms |
| New pin vs the earlier builder's smoke (`1b53d6234`) | 1 | every common field identical; only the STOP-4 rest-bound fields are new (added at `218a3378c`, as expected) |
| 4-turn `seam_b` at the new pin vs the old pin, same env | 4 | responses, events and env **byte-identical** (JSON). Winners (neural / commit): hsb_empty1 1/1 STAY_SILENT, hsb_q 0/0 (answered, not an abstain, so C3 never engaged), hsb_empty2 1/1, hsb_hi 1/1 |

Artifacts: `research/findings/raw/_bg_live_seam/repin_e017eb0/smoke1_s79_seam_b_new.json`, `research/findings/raw/_bg_live_seam/repin_e017eb0/smoke1_s79_seam_b_old.json`, `research/findings/raw/_bg_live_seam/repin_e017eb0/smoke4_s79_seam_b_new.json`, `research/findings/raw/_bg_live_seam/repin_e017eb0/smoke4_s79_seam_b_old.json` (the earlier builder's 1b53d6234 smoke stayed in its scratch).

## 5. Tests at the new pin

`tests/test_bg_gate_amend_1101_instrument.py tests/test_host_share_bg_seam_score.py tests/test_bg_commit.py
tests/test_bg_neural_seam.py tests/test_bg_tonic_loop_identity.py tests/test_izh_membrane_substeps.py`: 79 passed.

## 6. What did not change

No bar, arm, turn, trial schedule, env flag, seed list, report-only field or verdict rule. The signature of the
amendment (section 9) covers the BG code, which is the same at the new pin; only the revision the pool runs moved.

## 7. Provisioning

`provision.req/e017eb00e38e69d230a7d930e181049d7822160b` touched 2026-11-02 04:29Z; receipt (`research/coordination/bg-gate-1101-repin/provision_receipt_e017eb00e.json`) state `ok` at 04:35:30Z,
"provisioned on pool41, pool42, pool40, wspool" (behind published main `03e66832f` by one docs-only commit, accepted).
No pool line was queued by this re-pin; queueing still follows the amendment's canary rule (section 5.2).

## Correction and countersignature (Claude, 2026-11-02)

**Correction to sections 3 and 4.** The statement that C3 acts at most on an abstaining `hsb_q`, and the use of the
s79 seam_b smoke as evidence that C3 never engaged, are wrong. An independent reviewer found that `hsb_q` never abstains
outside a BG hold in the historical `_bg_live_seam` shards, while `hsb_hi` (the last turn) abstains with topic `hi` in
speaking arms. A prod_a seed-79 check at `e017eb00e` with C3 at its default (ON) against
`BRAIN_CURIOSITY_TOPIC_STOP=0 BRAIN_CURIOSITY_PHASIC_DA=0` shows the C3 organ engaging on `hsb_hi` and appending a
follow-up question to `answer` only; every other field (abstained, recalled_svo, supporting_facts, n_sentences, verified,
hedge, hold, bg_select and all events) matched. Because the phasic read uses the chat's DA level, the appended text can
differ between arms. No BG decision field and no scored gate metric moves; `answer` enters only the sham-null fields
(prod_a vs prod_b, both C3-ON and deterministic). The raw outputs of that check are on the workstation under
`_scratch/bgrev/` (not committed).

**Countersignature.** Claude countersigns this re-pin: the gate runs at `e017eb00e38e69d230a7d930e181049d7822160b`,
the merge of S main into `research/bg-gate-amend-1101`, whose BG code is byte-identical to the signed pin `218a3378c`
(reviewer diff over the runner, organ, scorer, battery, GNW bus, curiosity/DA modules and the six BG test files), with
the 24 pool lines of `research/coordination/bg-gate-1101-pool-lines.tsv` re-pinned to it and every bar, arm, turn, env
flag and verdict rule unchanged. The canary rule of the amendment (section 5.2) still applies before release.
