---
type: result
status: live
lane: stage2-C3
verdict: PASS
date: 2026-11-01
packet: c3-flip
prereg: research/findings/2026-10-29-c3-flip-PREREGISTRATION.md
scorer: research/runners/c3_flip_compare.py
---
# C3 flip (topic stop L + topic-locked phasic DA P) at the six gate seeds: PASS, flipped default ON (RESULT, 2026-11-01)

host-decision: scaffold
decides: unchanged. The follow-up ask is still decided by host arithmetic at its last step (C3 gate prereg section 9 (f)).
The flip makes the per-topic organ the default source of that decision; it does not retire the threshold.

Scored by Claude with the prereg's command:

    python -m research.runners.c3_flip_compare --on research/findings/raw/_c3_flip/gate_on \
        --off research/findings/raw/_c3_flip/gate_off --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_c3_flip/gate_verdict.json

## Verdict

<!--derived-->
DEC **PASS** (bars of RULING c3-flip-1029b, prereg section 2). 48 cells (4 groups x 6 gate seeds x ON/unset), every sidecar
at pin 15db27a73, ON sidecars carry both flags = 1, OFF neither.
- B1: abstained and recalled_svo identical ON vs OFF on every turn, 24/24 cells.
- B2: answer identical on every turn not abstained in both arms, 24/24 cells.
- B3: no FLIP-HURTS-ANSWERS turn; answered on/off equal in every cell (conv 7/7, infer 9/9, emo 5/5, inner 9/9; 180/180).
- B4 (reported, never a pass): other-leaf diffs conv 42, infer 13, emo 99, inner 12. They are the timing and process leaves,
  the curiosity trace leaves (curiosity_topic_stop added ON; curiosity_da absent ON because P replaces the host DA gain),
  and `curious` + `answer` on 4 turns that abstained in both arms (below).
- ENGAGEMENT (RULING c3-flip-smoke-1031): curiosity_topic_stop traces on every seed of every group, ON per seed conv 4,
  infer 1, emo 8, inner 1; OFF 0.

Post-flip smoke (dev seed 7, ask group, intact_a, flags UNSET at flip commit 7f024a1c1;
research/findings/raw/_c3_flip/postflip/s7/ask__unset_intact_a.json): asked vector `---1010111--1-0-0-0` equals the dev C
intact_a shard (research/findings/raw/_c3_topic_stop/dev/s7/ask__C_intact_a.json); answer, abstained and recalled_svo
identical on all 19 turns; 11 topic-stop traces. Unset reads ON.

## What changes for a user (verify-go panel, one independent skeptic, 2026-11-01)

<!--derived-->
- Only abstained turns change, and only in whether the follow-up question is appended: 4 turns in this battery. Three
  are second presentations of a topic (emo s100 `s_sadness_2` "alone", s102 `s_joy_2` "win", s102 `s_fear_2` "wolf"):
  ON stops asking, which is the faculty's purpose. One (infer s44 `c_prey_chase`, "heron's") asks ON and not OFF (P).
- The DA-mode engagement suffix (" — worth going further here.", webapp/da_mode_drives_chat.py) is appended to every
  reply in focus mode. OFF it already follows the follow-up question ("... about alone? — worth going further here.");
  ON, when topic stop drops the question, it follows a bare "I don't know about that." This punctuation is a property
  of the DA-mode suffix, not of this flip; it is filed as its own defect.
- Not measured: the per-conversation organ's first-abstain build (about 0.6 s, 450 neurons) and memory across many live
  sessions. The organ lives on each conversation's Context and an organ failure is recorded with the production
  decision standing (webapp/server.py).
- Where ran: gate cells pool 28, workstation 20 (tools/where_ran.py), all numpy on CPU at one revision; the identity
  bars passed on every pair, so host mixing could only have added differences, not hidden them.

## Flip

Commit 7f024a1c1: `topic_stop_enabled()` and `phasic_da_enabled()` read unset as ON; an explicit 0/false/off/no
escapes. Lesion and battery flags stay default OFF. Tests pin both reads and the explicit-off handler turn
(tests/test_curiosity_topic_stop.py, 87 pass + the opt-in handler test).
