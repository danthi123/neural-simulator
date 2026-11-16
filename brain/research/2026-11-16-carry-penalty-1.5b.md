# Pre-registration: cross-turn repetition penalty (reference arm), 1.5B, report-only probe

Written 2026-11-16 before any run with the switch. Record: `brain/experiments/results/2026-11-16-onset-baseline.md`
(the pretrained cortex alone collapses in 14/72 day-2 conversations). Lever count on in-context collapse: habituation,
reply monitor, own-reply gain (flat, copies-only); the research round ran (wf_83f0ad09-15d). This is not a new
mechanism: it runs the mouth as the model family's chat tooling does (occurrence counts carried across the
conversation, decay 0.996 per generated token, same presence/frequency 0.4), where ours resets them every turn.
Host decision: none beyond the existing decoding penalty; scaffold row = the existing mouth sampling.

Probe: `brain/experiments/echo_onset_probe.py --carry-penalty`, cortex alone, the collapse-prone day-2 scripts 1113,
1115, 1116, 1117, 1119, 1124 with the SAME 6 sampling seeds as the baseline (paired). Baseline collapses there:
12/36 (echo > 3 or questions back > 3). Lead: <= 6/36 collapse AND a reading of samples 0-1 of 1115 and 1124 finds
answers, not a paraphrase loop (habituation's failure). A lead goes to open_conv on fresh AWS seeds (prereg addendum
first, bars as in `2026-11-16-reply-monitor-1.5b.md` plus a benefit bar: fewer collapsed seeds than the default).
Expected: partial at best; the penalty does not cover user tokens the brain has not said, so the first echo is
untouched.
