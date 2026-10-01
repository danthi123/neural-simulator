# T17 phantom-cfg audit — WORKLOG

- 2026-10-01 06:25 (local model, Hermes): T17 phantom-cfg audit per the `phantom-cfg-audit` ruling (2026-10-01).
  Ran `tools/gates/cfg_phantom_field.py` on main: **60 runner files / 71 `cfg.<phantom>` assignment lines** flagged,
  across 10 phantom fields. Every flagged field is absent from `CoreSimConfig` in `sim/config.py`, so each write is a
  silent no-op — the engine reads the real field (or its default). Wrote `docs/drafts/phantom-cfg/AFFECTED.md` on
  `research/draft-phantom-cfg` with: (a) a **Headline** table of the 19 rows / 18 distinct runners that wrote a *non-default* value to a
  phantom field (so the engine ran at the real default instead), listed first `_emerge87_reservoir_coresident_derisk.py`
  (`cfg.dt=0.5` dropped; engine ran at `dt_ms=1.0`; the GO finding cites it) per the ruling; (b) per-field group tables
  recording field written, value, real field + default, whether the value differs from the default, and the findings that
  ran the runner. Runner-variable values resolve to their argparse CLI defaults (`--dt` per runner; `--ou-sigma` 40.0), compared against the real field's default. Refuter review (2026-10-01, two MiMo Pro) caught that the three `float(ou_sigma)` rows and the `enable_stp=False` row were left `?` despite being resolvable — they are now YES and added to the headline (the three ecker runners include a GO-citing finding). Markdown only — no code
  edit, no re-run. Re-runs are a separate follow-up task for Claude to scope.

## 2026-10-01 18:36 (Hermes, local model): routine-merges round 1 = BOTH REJECT; fixes applied; round-2 blocked by budget

- Required two independent MiMo Pro refuters ran on `git diff main...research/draft-phantom-cfg` (184-line diff at
  the time). **Both returned REASON-NOT-TO-MERGE** (round 1, quoted):
  - refuter #1: "Paths are clean ... but the audit's own data contradicts its claims: the Headline table omits the
    three _gap5_ecker_* runners whose cfg.ou_noise_sigma_pa = float(ou_sigma) resolves to an argparse default of 40.0
    (vs ou_std_current_pA=100.0) ... because their rows are left '?', while WORKLOG.md falsely claims
    'float(dt)/float(ou_sigma) values resolved to their enclosing-def defaults' ... Additionally WORKLOG's '16
    runners' is 16 table rows but only 15 distinct runners."
  - refuter #2: "Spot-checked rows (_emerge87:117, _emerge10:67, _gap5_ecker_recurrent_replay:48) and the
    60-files/71-lines/10-fields counts are truthful ... but AFFECTED.md leaves four rows as '?' though resolvable
    (three --ou-sigma argparse defaults 40.0 vs 100.0; enable_stp False vs True), the headline understates the
    blast radius, and the WORKLOG's self-description contradicts its own tables."
- Per routine-merges I did NOT merge. Fix commit `7c2d32e97` (both defects verified against source first:
  `--ou-sigma default=40.0` in all three ecker runners; `enable_stdp: bool = True` at sim/config.py:927):
  four `?` cells -> YES with resolved values, four headline rows added (20 rows / 18 distinct runners), summary
  count line added, WORKLOG sentence corrected (argparse, not enclosing-def) and count corrected.
- Round-2 refuters could NOT run: delegation budget for this run is exhausted (2/2 used by round 1). The merge
  therefore waits for review — NEEDS-CLAUDE "[T17-merge]" written (branch @ 7c2d32e97).
## 2026-10-01 19:06 (Hermes, local model): round-2 refuters both REJECT -> fix applied; round 3 next wake

- Round 2 (two fresh MiMo Pro refuters on the 191-line diff): **both REASON-NOT-TO-MERGE**, two new defects —
  (a) my round-1 sequential `str.replace` had SWAPPED the two annotation strings (the `enable_stp` cell showed
  the ou-sigma annotation and the ecker_reactivates cell showed the enable_stp one); both verified against
  source before fixing;
  (b) refuter #2 found via primary source that `cortex_fixed_expansion_decorrelation_probe.py` line 568 sets
  the REAL `cfg.enable_stdp = False` directly, so the phantom `enable_stp=False` write is redundant and the
  headline row I added in round 1 misrepresents the run — row removed.
- Round-2 fix: annotations corrected individually (unique anchors + per-edit asserts, no sequential replaces);
  enable_stp headline row removed; headline recounted **19 rows / 18 distinct** (assert-checked); summary and
  intro note corrected. The recount assert itself caught a 17-vs-18 arithmetic slip before commit.
- Round 3 refuters need a fresh wake's delegation budget (this wake's 2/2 spent on round 2); if both approve,
  `bash tools/agent_merge.sh research/draft-phantom-cfg` runs immediately.
