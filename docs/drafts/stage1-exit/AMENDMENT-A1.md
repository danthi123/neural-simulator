# Stage-1 exit gate, Amendment A1: two void controls and their re-measurement (Claude, 2026-10-04)

Amends `research/findings/2026-09-29-stage1-exit-gate-PREREGISTRATION.md` after attempt 1
(`research/findings/2026-10-04-stage1-exit-gate-RESULT.md`). Bars, seeds, probes and the C1/C2 verdicts are
unchanged. Only two instruments change, both because attempt 1 showed the registered manipulation never engaged.

## A1.1 C4: the permuted learned map deranged 0 of 24 words

Pool provisioning excluded `research/findings/raw/`, where the DR-2 learned map lives, so on the nodes the organ read
the raw Warriner norm. Fixed by `tools/pool/runtime_raw_inputs.txt` (shipped by every provisioner, and by
`tools/pool/ship_runtime_inputs.sh` for already-provisioned revisions). The scorer now makes a seed UNDEFINED unless the
permuted map deranged all 24 words. Re-measured at the gate pin c238ff325 (RULINGs s1-c4-void, s1-c4-topup).

## A1.2 C3: the recall lesion cut a path the production default never calls

`lesion_recall` set `BRAIN_SUBSTRATE_RECALL_LESION`, which makes `ChatBrain._substrate_recall` return None. Since the
2026-08-13 scaffold retirement, `brain_reply` installs `gnw_bus_shadow.install_bus_gate`, so a routable recall goes
through `gate_via_bus` -> `composer.query_patient` and never calls `_substrate_recall`. The lesion was inert, and
"reply-level effect on 0 of 6 seeds" was scored as "not load-bearing". The scorer had assumed every env-flag lesion
engaged (`rec["applied"] = True`).

- **New arm.** `lesion_recall` is now the patch `lesion_lived_recall`. It wraps
  `OneBrainComposer.query_patient` and returns a miss only for the lived cue (`LIVED_CUES = {("wolf", "bite")}`). It
  counts the cut calls, so engagement is recorded (`applied` = calls > 0), and built-in facts are untouched.
- **New scorer rule.** On a seed whose capability is present, an env-flag lesion with no reply change and no
  engagement record makes the seed UNDEFINED. An inert lesion and a non-load-bearing path cannot be told apart, so
  neither is a FAIL.
- **Re-measurement.** The runner change needs a new revision, and the server provisions only revisions that contain
  published main. So C3 is re-measured **entirely** at the new pin P2: all 7 `lived` arms x 6 gate seeds, 42 lines,
  out-dir `research/findings/raw/_stage1_exit/gate_c3rerun/`. Intact, lesion and permuted then share one revision.
  C3 is judged at P2 alone and reported as "attempt 1, C3 re-measured at P2". It is never pooled with c238ff325 arms.
- **Dev check before the gate lines** (dev seed 7, labels `t_wolf_bite,p_wolf_bite,c_dog_chase`): see the section
  below.

## Stage verdict after A1

The re-score with both guards (`aggregate-rescore-c4guard-2026-10-04.json` for C4; C3 by the A1.2 rule) reads C1
FAIL, C2 PASS, C3 and C4 INCOMPLETE. The stage is UNDEFINED (INCOMPLETE), and **not passed** either way: C1 fails on
defined data. C1's next method is `docs/drafts/stage1-c1-inheritance/PLAN.md`.
