---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: D6-learn-and-grow (content provenance; the owner's "fancy RAG" concern)
mechanism: the pre-registered content-provenance probe (research/runners/content_provenance_probe.py), scored with
  its own --score-only aggregate command, both mouth variants (prod -- the shipped spiking recall mouth speaks
  first, Qwen only on a VERIFY failure; qwenforced -- BRAIN_SPIKING_MOUTH_RECALL=0, Qwen renders every recalled
  fact); the arm under test is the default-OFF local Hebbian write (BRAIN_D6_HEBBIAN_STORE / BRAIN_D6_ENGRAM_VOCAB
  / BRAIN_D6_ENGRAM_READTIME)
seeds: [42, 43, 44, 100, 101, 102]
prereg: research/findings/2026-09-24-content-provenance-learned-facts-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_content_provenance/cp_verdict.json
  - research/findings/raw/_content_provenance/s*.json
  - research/findings/raw/_content_provenance_qwenforced/cp_verdict.json
  - research/findings/raw/_content_provenance_qwenforced/s*.json
verdict: prod reads GO 6/6 and qwenforced reads GO 6/6 on the registered validation seed set
  (42/43/44/100/101/102); seed 7 is the un-counted dev smoke (prod GO, qwenforced NO-GO on CP1, reported only).
  The learned-content fraction of the shipped DEFAULT path reads 0.00 on every seed, both variants; the opt-in
  USE arm reads 0.60-0.75. Not a flip candidate yet -- no combined-battery regression or production-default run
  has been made with the D6 write flags ON.
---

# Content provenance, scored: both mouth variants read GO 6/6; the shipped default still learns 0%

## What this scores

The 2026-09-24 pre-registration registered two mouth variants of one probe through the real production chat path
(`webapp.server.brain_chat`, `renderer='qwen'`, the rich path, `wikidata` LTM off): `prod` (the shipped
configuration -- the spiking recall mouth renders every bounded transitive-SVO fact and Qwen speaks only when
that surface fails VERIFY) and `qwenforced` (amendment A1's primary variant, `BRAIN_SPIKING_MOUTH_RECALL=0`, so
Qwen renders every recalled fact -- the adversarial form of the owner's "fancy RAG" concern). Each variant needs
all five arms (USE / FREEZE / ABLATE / HEARD / DEFAULT) on all seven seeds (the seed-7 dev smoke plus the six
registered validation seeds).

**Completeness, checked before scoring.** All 70 arm files (35 per variant, plus an optional `s7_DEFAULT_LTM` in
`prod`) and their `.prov.json` sidecars were present in the primary checkout for every registered seed/arm. Every
sidecar's `git_sha` sits inside the governed lineage (`83d406d729` the original prereg commit, `53d884616c` the
A1 amendment that added the variant plumbing) with `git_dirty: false`. No entry for this family was live in
`research/queue/pool.running`, `pool.queue`, or `gpu.queue` (the one `pool.running` line covering the qwenforced
42-102 batch had already produced every file it lists, and no `content_provenance`/`cprov` process was running).
No finding on `main` scores this family (`git log --since=2026-09-23 --grep=content-provenance -i` finds only
the pre-registration, the runner build, an amendment, and a data-recovery/fix pair; no scored finding). This
scores rather than reporting INCOMPLETE.

**Scored with the pre-registration's own verbatim command**, once per variant, against the copied arm files
(re-grading JSON only -- no brain was built for this pass):

```
.venv/bin/python -m research.runners.content_provenance_probe --score-only --variant prod \
    --seeds 7 42 43 44 100 101 102 --arm-dir research/findings/raw/_content_provenance \
    --json research/findings/raw/_content_provenance/cp_verdict.json
.venv/bin/python -m research.runners.content_provenance_probe --score-only --variant qwenforced \
    --seeds 7 42 43 44 100 101 102 --arm-dir research/findings/raw/_content_provenance_qwenforced \
    --json research/findings/raw/_content_provenance_qwenforced/cp_verdict.json
```

## Result

<!--derived-->
| variant | seed | verdict | CP1 taught | CP2 freeze | CP3 zeroed | CP4 heard-only | CP5 leaks USE/DEFAULT | fraction USE | fraction DEFAULT |
|---|---|---|---|---|---|---|---|---|---|
| prod | 42 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| prod | 43 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| prod | 44 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| prod | 100 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| prod | 101 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| prod | 102 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| prod | 7 (dev, uncounted) | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| qwenforced | 42 | GO | pass | pass | pass | pass | 0/0 | 0.75 | 0.00 |
| qwenforced | 43 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| qwenforced | 44 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| qwenforced | 100 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| qwenforced | 101 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| qwenforced | 102 | GO | pass | pass | pass | pass | 0/0 | 0.60 | 0.00 |
| qwenforced | 7 (dev, uncounted) | NO-GO (CP1) | fail | pass | pass | pass | 0/0 | 0.33 | 0.00 |

**Aggregate**, straight from each variant's `cp_verdict.json`: `prod` -> `"n_defined": 6, "n_go": 6, "verdict":
"GO 6/6", "GO": true`. `qwenforced` -> `"n_defined": 6, "n_go": 6, "verdict": "GO 6/6", "GO": true`. `failed_arms`
is empty in both (no VOID). The pre-registration's own rule ("Seed 7 is a dev smoke: it is reported and never
counted") keeps the qwenforced seed-7 NO-GO out of the aggregate.

## Reading it

- **CP1-CP5 hold on all six validation seeds, in both variants**, including `qwenforced` -- the adversarial form
  of the owner's question. With Qwen forced to render every recalled fact, the brain still states the taught
  object with no `prior(item)` word (CP1), FREEZE/ABLATE/HEARD still abstain (CP2-CP4), and neither USE nor
  DEFAULT leaks a Qwen-prior or hand-key word into any of the 20 untaught replies (CP5: 0/20 in both arms, every
  seed). Forcing Qwen to speak does not inject the "fancy RAG" content the owner was concerned about, on this
  protocol.
- **The only failure anywhere in the corpus is the un-counted dev smoke.** `qwenforced` seed 7 fails CP1 -- Qwen
  drops the taught counterfactual fact rather than stating it or a competing one -- exactly the "CP1 is the open
  prediction" risk the pre-registration named. It does not recur on any of the six scored seeds; it is reported
  here per the registered rule, not folded into the verdict, and matches the correction already on record in
  commit `5e52c75ef` (which retracted an earlier commit-message claim that this same seed-7 smoke was GO).
- **The learned-content fraction the owner asked to be tracked reads 0.00 on the shipped DEFAULT path, every
  seed, both variants** -- the production default still writes no plasticity trace today, consistent with the
  2026-09-24 audit that motivated this probe (a host bulk imprint, no plasticity write). The USE arm
  (`BRAIN_D6_HEBBIAN_STORE=1` plus the engram flags, itself default-OFF) reads 0.60 on eleven of twelve validation
  seed/variant cells and 0.75 on qwenforced seed 42 -- an opt-in instrument reading, not a change to anything
  shipped.
- Per the pre-registration's own honest-scope section, the USE arm's write is "a local rule carried by a
  host-wired instructive pathway" that matches a host copy at very high correlation, not self-organized
  acquisition (`docs/TERMS.md`); this finding calls it "a local Hebbian write," not "self-organized."

## Flip-candidate status: NOT YET

The owner's flip bar is 6-seed GO **and** a SOUND review **and** no regression in the combined battery with the
change ON **and** a production-default validation run. Against the D6 local-write flags the USE arm exercises
(`BRAIN_D6_HEBBIAN_STORE`, `BRAIN_D6_ENGRAM_VOCAB`, `BRAIN_D6_ENGRAM_READTIME`):

- Met: 6-seed GO on the registered protocol, in both mouth variants.
- Partial: the *probe instrument* was independently re-reviewed to SOUND after two fix rounds (`dd7bb0ec8`); that
  review assessed the runner's correctness, not a decision to default the D6 write flags on.
- Missing: no combined-battery run exists with the D6 write flags ON alongside the rest of the production organ
  set -- this probe only ever exercises its own five-arm teach/probe/untaught protocol in isolation.
- Missing: no production-default validation run -- the flags stay OFF in the configuration used here; USE is an
  explicit opt-in override, never the DEFAULT arm's configuration.

Before this can be called a flip candidate: a combined-battery regression pass with the D6 write flags ON, and a
production-default run under that configuration.

## Honesty

Functional read-outs only. "Learned" / "states" describe measured reply content and a lesion-verified synapse
state (the ABLATE arm's zeroing is confirmed to still hold at probe time by `lever_ABLATE_zero`), per the
pre-registration's own honesty section. No claim of felt experience is made.
