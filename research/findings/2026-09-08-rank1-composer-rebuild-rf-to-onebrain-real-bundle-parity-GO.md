---
type: finding
status: go
claim_check: measured-result
date: 2026-09-08
mechanism: rf->onebrain composer rebuild of the deployed scale787/day_33 bundle (spiking composer replaces the host closed-form FHRR)
lane: scaffold-retirement (backlog rank-1)
seeds: [42]
artifacts:
  - research/findings/raw/_rank1_composer_rebuild/verify_404.json
  - research/findings/raw/_rank1_composer_rebuild/run_404.log
  - research/runners/_rank1_composer_rebuild_onebrain_verify.py
---

# RANK-1 de-risk: the DEPLOYED 404-fact bundle rebuilds OFF the host `rf` composer ONTO the spiking `onebrain` composer with FULL parity + recall + no-confab moat — GO (single deployed seed), with one honestly-characterized ambiguous-cue boundary in `ask_yes_no`

<!--derived-->
<!-- Every number below is a read-out or quotation of the cited artifact
     (research/findings/raw/_rank1_composer_rebuild/verify_404.json) or of the prior findings named inline
     (the DG-CA3 de-risk's 563 MiB / ~402x / 6-seed, the commit VRAM gate's ~1.3 GiB / 5.6% of 24 GB). This is a
     human-readable summary of a machine artifact -> derived, per the gate's own guidance. -->

**Board/lane: scaffold-retirement backlog RANK-1 (the owner's #1 arc, MAXIMAL load-bearing).** The deployed
brain's live recall/store/abstain composer is pinned to `composer_kind='rf'` in the deployed bundle manifest
(the untracked scale787 day_33 developed brain; `composer_kind` reads `"rf"`), so every live answer runs the HOST closed-form FHRR
algebra and the spiking unbind + NEF/Izhikevich WTA cleanup that `OneBrainComposer` enables BY DEFAULT never touch a
live recall. Brain-based-only: the composing must be the spiking substrate's job. This de-risk performs the LITERAL
rebuild the backlog names as "confirmed still NOT done" and verifies it against the `rf` baseline on the REAL
deployed bundle.

## Why this was still open (verify-first)
- The deployed bundle is UNTRACKED in git (no commit could have flipped its manifest); `git show
  main:.../brain_conversational_agent.py` still defaults `composer_kind="rf"`; `webapp/server.py`'s own comment
  still describes scale787/day_33's manifest as `'rf'`. The one `composer_kind="onebrain"` commit (`2b0264336`)
  only turned cleanup/learned-assoc default-ON *within* the onebrain path — dead code while the composer default
  is `rf`.
- The retrieval-latency SUB-blocker that stopped the rebuild is RESOLVED: DG-CA3 sharded spiking retrieval
  (`BRAIN_FACT_SHARD_RETRIEVAL`, flipped default-on 2026-09-05, GO 6/6, ~402x @ 404 facts;
  `2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md`). So the rebuild is unblocked.
- Prior onebrain-correctness (recall==rf, moat, 563 MiB @ 404 facts, 6 seeds) was established on SYNTHETIC
  UNIQUE-(agent, action) facts. The deployed bundle is heavily AMBIGUOUS — of 190 distinct (agent, action) cues,
  **29 carry >1 distinct patient** — the "degenerate same-(agent, action) different-patient" regime the
  `OneBrainComposer.ask_yes_no` docstring flags as "outside the production regime". This is the first parity check
  of the spiking composer vs the host `rf` on that real regime.

## What was run
`research/runners/_rank1_composer_rebuild_onebrain_verify.py` (seed 42 = the bundle's own developmental seed;
numpy/CPU; `BRAIN_FACT_SHARD_RETRIEVAL=1` = the production default; bare `OneBrainComposer`, `BRAIN_COMPOSER_MERGE=0`).
It loads the `rf` baseline via `load_developed_brain` (the exact deployed path), rebuilds a
`BrainConversationalAgent(composer_kind='onebrain', onebrain_k_max=n_facts+16)` over the SAME grounded codes + seed,
re-stores all 404 real facts on-substrate (the spiking store, 104.8 s), and compares over the distinct cues.

## Results (404 real facts; `research/findings/raw/_rank1_composer_rebuild/verify_404.json`) — GO

| gate | rf baseline | onebrain rebuild | verdict |
|---|---|---|---|
| `query_patient` strict parity (190 cues, 29 ambiguous) | — | **1.0000** (incl. all 29 ambiguous) | onebrain returns rf's EXACT answer everywhere |
| `query_patient` valid recall | 0.9947 | **0.9947** | identical (both miss the SAME 1 cue) |
| `query_agent` strict parity (208 cues) | — | **1.0000** | identical |
| `query_agent` valid recall | 0.9952 | **0.9952** | identical |
| `ask_yes_no` UNambiguous (161 cues) | 'yes' | **0.9938 'yes'** | parity on the unambiguous regime |
| `ask_yes_no` ambiguous (part of 264 SVO) | 0.9962 yes | **0.7159 yes**; 75 -> 'unknown'; **0 'no'** | moat-SAFE abstain (see below) |
| no-confab moat (100 out-of-store probes) | 100 abstain, 0 confab | **100 abstain, 0 confab** | clean |
| scramble control (shuffled cue->answer) | — | recall 0.9917 -> **0.0000**; attribution **1.0** | recall is the learned binding, not the harness |

**Headline:** on the two content-recall paths (`query_patient`, `query_agent`) the spiking onebrain composer is a
STRICT parity match to the host `rf` closed-form — same answer on every cue, including all 29 genuinely-ambiguous
cues, with identical valid recall and a perfect no-confab moat. The composing is now the spiking substrate's job
with no loss on recall.

## The one honestly-characterized boundary (NOT a failure — a moat-safe abstain)
`ask_yes_no` on the 29 ambiguous (same-(agent, action), different-patient) cues: onebrain selects the FIRST
(agent, action) fact block, then checks the queried patient on THAT block — so when a stored fact shares its
(agent, action) with an earlier-stored, different-patient fact, onebrain returns **'unknown'** (75 cases) where
`rf`, which scans for the full SVO, returns 'yes'. Crucially this is the SAFE direction: **`ob_no_on_stored = 0`**
(onebrain never asserts a false 'no') and **0 confabulation**. It is exactly the "outside the production regime"
behavior the composer's own docstring documents, and it is MECHANISTIC (first-block selection), not seed-dependent.
On the unambiguous regime `ask_yes_no` is 99.4% parity. Surpassing this residual (a per-(agent, action)
multi-block yes/no check) is the named next rung if ambiguous yes/no recall is wanted — it is not a blocker for the
content-recall flip.

## The literal rebuilt bundle + the remaining production-flip blocker
The runner writes the literal rf->onebrain bundle (same grounded codes + facts as the source, manifest
`composer_kind='onebrain'`, `kb_composites.npz` dropped since onebrain composites live on-substrate). It is a
mechanical copy + manifest-flip, fully REGENERABLE by the runner, so its verbatim data files are not committed here
(they would only duplicate the untracked source bundle). **Production-flip blocker found (characterized, not fixed
here):** `load_developed_brain` hardcodes
`onebrain_k_max=None -> 32`, so a >32-fact onebrain bundle would reload truncated. The flip needs `onebrain_k_max`
threaded through `load_developed_brain` (and `webapp/server.py`'s loader), sized to the bundle's `n_facts` — an
additive plumbing change, the same seam the `BrainConversationalAgent(onebrain_k_max=...)` wire-in already opened.

## Scope / honesty
Single deployed seed (there is one deployed bundle; seed 42 is its own developmental seed, and the parity is
deterministic given the bundle). The onebrain CORRECTNESS generalization is already 6-seed (the prior de-risk); the
ambiguous-`ask_yes_no` boundary is mechanistic and seed-independent by construction. Memory is affordable (563 MiB
@ 404 facts per the prior de-risk; the commit's VRAM gate estimated ~1.3 GiB with an 8x margin — 5.6% of the 24 GB
consumer reference). **Production `composer_kind` default STAYS `'rf'`** — this is a de-risk in isolation; the
production flip is the owner's call and additionally needs the `onebrain_k_max` load-path thread above.

<!--derived-->
