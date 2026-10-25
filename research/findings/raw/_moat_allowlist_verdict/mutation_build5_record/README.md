# Build-5 mutation round: record

Mutation testing of the strict allowlist matcher at `ef51d6243` (workflow `wf_7282d984-21c`, 2026-10-20/21, run in a
scratch worktree outside this repo). Copied here as the record behind `../mutation_build5_summary.json`:

- `results_b5.jsonl`: one row per mutant (its edit, whether the witness corpus or the pytest suite killed it, and for a
  survivor the equivalence verdict or the witness that pins it).
- `build_results5.py`: the script that merged the corpus, pytest and witness-search outputs into `results_b5.jsonl`
  and the summary. Its paths point at the scratch directory; it is a record, not runnable from here.
- `equiv_notes5.py`: the written reason for each mutant judged equivalent.

The 70 non-equivalent survivors are pinned by `tests/test_moat_allowlist_build5_witnesses.py` (`56b85d140`).
