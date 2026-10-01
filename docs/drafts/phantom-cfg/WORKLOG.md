# T17 phantom-cfg audit — WORKLOG

- 2026-10-01 06:25 (local model, Hermes): T17 phantom-cfg audit per the `phantom-cfg-audit` ruling (2026-10-01).
  Ran `tools/gates/cfg_phantom_field.py` on main: **60 runner files / 71 `cfg.<phantom>` assignment lines** flagged,
  across 10 phantom fields. Every flagged field is absent from `CoreSimConfig` in `sim/config.py`, so each write is a
  silent no-op — the engine reads the real field (or its default). Wrote `docs/drafts/phantom-cfg/AFFECTED.md` on
  `research/draft-phantom-cfg` with: (a) a **Headline** table of the 20 rows / 18 distinct runners that wrote a *non-default* value to a
  phantom field (so the engine ran at the real default instead), listed first `_emerge87_reservoir_coresident_derisk.py`
  (`cfg.dt=0.5` dropped; engine ran at `dt_ms=1.0`; the GO finding cites it) per the ruling; (b) per-field group tables
  recording field written, value, real field + default, whether the value differs from the default, and the findings that
  ran the runner. Runner-variable values resolve to their argparse CLI defaults (`--dt` per runner; `--ou-sigma` 40.0), compared against the real field's default. Refuter review (2026-10-01, two MiMo Pro) caught that the three `float(ou_sigma)` rows and the `enable_stp=False` row were left `?` despite being resolvable — they are now YES and added to the headline (the three ecker runners include a GO-citing finding). Markdown only — no code
  edit, no re-run. Re-runs are a separate follow-up task for Claude to scope.
