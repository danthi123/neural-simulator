# Capability: Episodic recall
# Pre-registration: S_all + speaker contrast at a lower strength (mu 1.5), live 1.5B unslept day

Written 2026-11-14 before the run. Follows `results/2026-11-14-sall-contrast-1.5b.md`: at mu 3, told +1.00 on the
registered seeds but +0.50 on seeds 200-205; para +1.2 on both; the near-miss lure ("Where does my aunt live?" ->
Oslo, from the grandmother fact) fires on 12/12 seeds against 6/12 without contrast. Question: does a weaker contrast
keep the para gain with the lure back near the no-contrast rate?

Run: `unslept_day_probe --arms c,s,r --speaker-contrast 1.5`, seeds 42/43/44/100/101/102, pool CPU fp32
(`jobs/2026-11-14-pool-sall-contrast-dose.sh`), re-scored with `brain.experiments.rescore`. Compared on the same seeds
with no contrast (`2026-11-14-key-default-1.5b/A/`) and mu 3 (`2026-11-14-sall-contrast-1.5b/`). Arm r only.

- D0: contrast recorded (1.5) in every file, 6 files, rc 0.
- D1: para mu 1.5 - no contrast >= +0.67 (two thirds of the mu 3 gain, +1.33 re-scored).
- D2: told mu 1.5 >= no contrast (5.83).
- D3: aunt lure at mu 1.5 on <= 4 of 6 seeds (no contrast: 4/6, mu 3: 6/6).

All hold -> LEAD: mu 1.5 replaces 3 in the candidate package for the mixed-day test. Otherwise mu 3 stays the
candidate and the lure goes to the confidence gate. Expected: unknown; contrast effects grew with mu in the oracle.
