# Lane E (Language) actual state, 2026-07-30: two runners PASS at 6 seeds, two are BLOCKED BY ENVIRONMENT not science

Surfaced by serving the unserved roadmap lanes rather than continuing the gap#5 arc. Lane E had shown
`running=0` all session; the reason turns out to be split.

## PASSING at 6 seeds (42/43/44/100/101/102)

| runner | result |
|---|---|
| `_emerge62c_morphological_invariance_cue_derisk` | render-ok **1.00**, frame-words-covered **True**, **moat calls on abstain 0** |
| `_emerge72_construction_registry_derisk` | producer invoked **5 times over 6 probes** — the abstain never invoked the producer, i.e. **the no-confabulation moat held** |

Identical results at 3 and 6 seeds, so these are stable, not seed-lucky.

## BLOCKED — environment/data, NOT a scientific wall

| runner | blocker |
|---|---|
| `_grounded_lang_integration_derisk` | **`ModuleNotFoundError: No module named 'transformers'`** — the package is not in `.venv`. The runner correctly reports `VERDICT: ERROR` rather than a fake score. |
| `_emerge60_console_spiking_broca_derisk` | **`FileNotFoundError: research/findings/raw/fluidconv/gen_tinystories.bpe.json`** — a generated artifact that is absent (untracked output, and per the migration memory untracked files have NO backup). |

**⇒ NEITHER IS A NEGATIVE.** Both are missing prerequisites. Recorded so the lane is not mistaken for
scientifically stalled: **two of four runners are one `pip install` / one regenerated artifact away from running.**

## Process note (the reason this was found at all)

Lane E read `unserved` for the entire session while I worked one arc serially. Twice I asserted there was no
useful work to run; both times inspection disproved it in under a minute. **When I claim nothing is runnable, I
have not checked — I have inferred it from the arc I happen to be focused on.** The mechanical lane check
(`tools/workflow_check.sh`) now prints each idle lane's exact command, which is how these four got run.

**Also a caveat on my own flag usage:** `_grounded_lang_integration_derisk` takes `--seed` (singular), and my
first launch passed `--seeds`, which argparse rejected. The runner was fine; the invocation was mine.
