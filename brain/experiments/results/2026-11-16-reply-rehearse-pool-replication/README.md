# Reply-only rehearsal night, pool replication (report-only, seeds 1121-1125)

Pool40/41/42, 1.5B, `brain/eval/open_conv.py` default (ocd) vs `--rehearse-reply` (ocy). Report-only: the decision is
the AWS batch on seeds 1115-1120 (prereg `brain/research/2026-11-16-rehearsal-night-1.5b.md`, addenda 4-5).
Scored with `brain/experiments/rehearsal_night_score.py <dir> 1121 1122 1123 1124 1125`:

| arm | collapsed | recall | false | intrusions | echoes |
|---|---|---|---|---|---|
| default | 1/5 | 16/30 | 4 | 2 | 14 |
| reply-only | 1/5 | 7/30 | 1 | 2 | 16 |

Reply-only recalls 0 on seeds 1124 and 1125 in the intact arm, but its hippocampus-off arm answers 5/6 and 3/6 there.
On 1124 the intact day 2 falls into "I'm not sure, I don't know what your X is." on every question after the first
abstention, including "How many legs does a spider have?". Mechanism, untested: the rehearsal guard includes the
day-1 user questions ("What is my favourite fruit?") with the pretrained cortex's own reply to them, which is an
abstention, so the KL holds abstention on exactly the fact questions.

Echo diagnostic (`echo_diag/`, `brain/eval/open_conv_day2.py` on the 1121 slept default home): the verbatim echo of
the user's turn (14/16) is identical with the night's change off and with both the night and the hippocampus off.
On that seed the echo is the pretrained cortex's in-context behaviour, not the night (this does not show the night plays no part on other seeds). It starts at turn 2, right
after a greeting reply that ends in a question to the user (as on AWS 1118 reply-only, 15/16 pronoun-normalised echoes). Probe output: `echo_diag/echo_onset_probe.json` (1 of 12 cortex-alone conversations echoes).
