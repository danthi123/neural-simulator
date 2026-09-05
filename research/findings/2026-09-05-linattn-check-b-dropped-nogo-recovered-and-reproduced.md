---
type: finding
status: no-go
date: 2026-09-05
mechanism: recovery + independent fresh reproduction of a silently-dropped gate result — no code change. The
  2026-09-04 linattn flip's bare-default safety check (`check_b_bare_default_linattn_and_affect_go.py`,
  committed at `ac58b81e6` / merged `4ea2ff74`) was RUN, returned a NO-GO, and that NO-GO's own JSON output
  was never committed — only the passing `check_a` artifacts landed. This finding puts the dropped result on
  the record (CLAUDE.md silent-failure discipline: "a NO-GO gate result was produced and silently dropped"),
  and independently re-runs the exact committed script in this worktree, byte-for-byte reproducing it.
seed-waiver: single seed (42) — this is check_b's own scope, unchanged here; `webapp/server.py` hardcodes
  `seed=42` at every organ construction site, so the deployed pipeline this check exercises has no seed axis
  to sweep (see the companion fresh-subprocess re-verification finding for the full argument).
lane: language (own-voice mouth / production flip gate) — record-integrity correction
seeds: [42]
verdict: NO-GO stands, confirmed twice. `BARE_DEFAULT_FLIP_CONFIRM_GO: false` — the bare (unset
  `BRAIN_WKV_MOUTH_*`) production default fails its own determinism control (`Q1_affect_loadbearing_PASS:
  false`, `raw_reproduces_lesion0_vs_lesion0_repeat: false`). Recovered from an abandoned worktree
  (provenance-stamped `git_sha: e9d6325d5`, 2026-09-04T02:24:24) AND independently reproduced fresh in THIS
  worktree today (`git_sha: 34bd67000`, 2026-09-05T00:16:43) — every one of the three generated raw strings
  (`l0`/`l1`/`l0_repeat`) is byte-identical between the two runs, verified by exact string compare, not
  read-by-eye. This ALSO falsifies check_b's own in-file comment, which blames an earlier revision's extra
  "smoke turn" for the determinism failure and claims removing it fixes the issue: the CURRENTLY COMMITTED
  script has no smoke turn and reproduces the identical failure, so the smoke turn was never the cause.
artifacts:
  - research/findings/raw/_linattn_flip_verify/check_b_bare_default_linattn_and_affect_go.py
  - research/findings/raw/_linattn_flip_verify/check_b_bare_default_DROPPED_2026-09-04.json
  - research/findings/raw/_linattn_flip_verify/check_b_bare_default_DROPPED_2026-09-04.json.prov.json
  - research/findings/raw/_linattn_flip_verify/check_b_bare_default.json
  - research/findings/raw/_linattn_flip_verify/check_b_bare_default.json.prov.json
---

# The linattn flip's bare-default check_b NO-GO was dropped, not fixed — recovered and reproduced

`check_b_bare_default_linattn_and_affect_go.py` is the ONE script in the 2026-09-04 linattn flip's own
verification suite that actually tests the bare, unset-`BRAIN_WKV_MOUTH_*` production default (`check_a`
tests the flip OFF; `check_c` tests an explicit `ssm` override — neither exercises the bare-default path
the flip's headline claim rests on). It was run before the merge, returned
`BARE_DEFAULT_FLIP_CONFIRM_GO: false`, and that result was never committed: only `check_a`'s passing
before/after JSON pair landed at `ac58b81e6`. The failing run's own JSON survived only as an untracked file
in an abandoned agent worktree (`.claude/worktrees/agent-a08638b256bb4795d/`), independently
provenance-stamped (`git_sha: e9d6325d5`, 2026-09-04T02:24:24, `sim_backend: numpy`) — recovered here and
committed alongside this finding as
`research/findings/raw/_linattn_flip_verify/check_b_bare_default_DROPPED_2026-09-04.json`.

## What check_b actually measured, and why it says NO-GO

The bare defaults DO resolve to linattn/bpe/broad and the ckpt DOES load
(`bare_default_resolved_linattn_bpe_broad: true`) — the flip's config wiring is not in question. The failure
is `Q1_affect_loadbearing_PASS: false`, and the row data shows exactly why: `raw_differs_lesion0_vs_lesion1:
true` (the affect coupling itself IS load-bearing) but `raw_reproduces_lesion0_vs_lesion0_repeat: false` —
the SAME turn (known topic, lesion OFF), asked twice in the same session with a `BRAIN_AFFECT_LESION=1` turn
in between, produces two DIFFERENT continuations. The determinism control fails, so the affect-load-bearing
result it is meant to validate is unsound by the check's own logic (`PASS = raw_diff_lesion AND
raw_repro_lesion0`). `Q2_moat_with_affect_PASS: true` and `Q1_lesion0_fluent_not_salad: true` both hold.

## The dropped run is reproducible today, byte-for-byte — and that refutes the script's own comment

Re-running the CURRENTLY COMMITTED `check_b_bare_default_linattn_and_affect_go.py` fresh in this worktree
(`git_sha: 34bd67000`, today) reproduced `BARE_DEFAULT_FLIP_CONFIRM_GO: false` again, and all three raw
continuations (`l0`, `l1`, `l0_repeat`) plus both Q2 moat replies are byte-identical (exact string compare,
not eyeballed) to the recovered 2026-09-04 run — despite the fact that the CURRENT script's own in-file
comment claims an earlier revision's extra "smoke turn" (a fourth priming call before the Q1 sequence,
present in the recovered run's `rows.smoke_turn` field but absent from the currently committed script) was
the cause of the determinism failure, and that removing it "self-corrected" the issue. It did not: the
smoke-turn-free script produces the IDENTICAL failure on the IDENTICAL text. The real cause is the same
class of confound `phase6_linattn_clean_isolation.py`'s own docstring already diagnoses for the ORIGINAL
`phase4` script — session-level state (mood EMA / habituation) evolving across FOUR sequential turns of ONE
shared session — which check_b inherits unchanged (it never adopted phase6's fresh-session-per-arm fix, let
alone fresh-subprocess isolation). See the companion finding
(`2026-09-05-linattn-bare-default-fresh-subprocess-verdict.md`) for the corrected re-verification.

## Why this matters

This is the exact failure class CLAUDE.md names: "a NO-GO gate result was produced and silently dropped."
The flip merged with only the passing `check_a` on record, while the one check that actually tested the
bare production default sat, un-committed, in a worktree nobody merged. No code changes accompany this
finding — it exists to put both the original dropped result and its independent reproduction on the record.
