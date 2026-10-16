---
type: finding
status: live
lane: integration
date: 2026-10-16
artifacts:
  - research/findings/raw/_stage1_exit/gate_a2/aggregate.json
  - research/findings/raw/_stage1_exit/gate_a2/aggregate.json.prov.json
  - research/queue/s1_attempt2_c02180f2c.tsv
verdict: PASS (C1-C4 primary each 6/6; diagnostics cat_told + val_novel FAIL 0/6, not counted)
---

# Stage-1 exit gate — attempt 2 RESULT (PROVISIONAL)

**PROVISIONAL** — filed per RULING `s1-score-now-1016` / `s1-attempt2-score`. Claude must verify-go and
adversarially verify (HANDOFF NEEDS-CLAUDE #7) before any finding or board change. Bars untouched.

<!--derived--> **Stage-1 exit verdict: PASS** (primary criteria: C1 PASS, C2 PASS, C3 PASS, C4 PASS)

Source of truth: `research/findings/raw/_stage1_exit/gate_a2/aggregate.json` (verdict `PASS`, GO `true`,
missing_shards `[]`, undefined_reasons `[]`), reproduced with
`.venv/bin/python -m research.runners.stage1_exit_gate --report research/findings/raw/_stage1_exit/gate_a2/aggregate.json`.

## Per-criterion verdicts (HANDOFF §8 sentences, numbers from the aggregate table)

- C1 (`cat_disc`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.
- C2 (`word`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.
- C3 (`lived`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.
- C4 (`val_heldout`): PASS. Correct on 6 of 6 seeds; every lesion arm changed the reply on 6 of 6 and reproduced;
  the permuted exposure was tracked on 6 of 6.

Per-seed valence r (reply-visible differential vs Warriner norm, 24 held-out words):
- seed 42: r = 0.817 (n = 24, sign agreement 0.58); lesion r = None; permuted-map r = -0.144; defined
- seed 43: r = 0.805 (n = 24, sign agreement 0.46); lesion r = None; permuted-map r = 0.046; defined
- seed 44: r = 0.824 (n = 24, sign agreement 0.67); lesion r = None; permuted-map r = 0.081; defined
- seed 100: r = 0.809 (n = 24, sign agreement 0.62); lesion r = None; permuted-map r = -0.073; defined
- seed 101: r = 0.836 (n = 24, sign agreement 0.54); lesion r = None; permuted-map r = -0.001; defined
- seed 102: r = 0.820 (n = 24, sign agreement 0.54); lesion r = None; permuted-map r = -0.104; defined

`cat_told` (diagnostic, not counted): FAIL, correct 0/6.
`val_novel` (diagnostic, not counted): FAIL, correct 0/6.

Both diagnostics read `capability absent on 6 of 6 seeds`; the prereg reports them beside C1/C4 and freezes them
as "not a pass condition" (HANDOFF §8: diagnostic, not counted).

## preconditions

All `require` entries in `aggregate.json.preconditions` are `ok: true`:
no primary shard missing; null clean (intact_a == intact_b) on every primary (group, seed); every response from
the expected brain (`tiny-demo +LTM`); `PYTHONHASHSEED=0` in every primary arm; corpus
(`data/corpus/tinystories.txt`) present in every primary arm; every in-process lesion applied (patch called at
least once); and the remaining require-list carried by the runner (reasons verbatim in the artifact).

## Reproducibility / where the numbers came from

216/216 A2 shards at SHA2 `c02180f2c7e1464cf0f2192a46cf2f399bd1c215`, seeds 42,43,44,100,101,102 (36 arms each),
committed with prov sidecars as `1d462264b`. Host summary (`tools/where_ran.py`): aws `ip-172-31-34-243` 90 +
`ip-172-31-47-61` 90 (the two r7i.4xlarge of RULING `s1-awsrun-answer-1016`, spend ~$9.26), pool nodes 19
(seed-42 lane), workstation 17 (seed-42 backfill), server 1 = the local seconds-long aggregate itself. The
corpus and brain inputs the provisioner ships are asserted per-arm by the preconditions above, so validity does
not depend on host class here. Delegate verdict (MiMo Pro, 2/2 budget): **PASS-FILE**.

## artifacts

- `research/findings/raw/_stage1_exit/gate_a2/aggregate.json` (+ `.prov.json`)
- `research/findings/raw/_stage1_exit/gate_a2/s{42,43,44,100,101,102}/*.json` (+ sidecars) — 216 shards
- emit TSV: `research/queue/s1_attempt2_c02180f2c.tsv`
