---
type: finding
status: no-go
date: 2026-09-05
mechanism: metacog spiking recall-margin — PRODUCTION-FLIP verification (ship-the-validated-wins campaign, Track 1) of rank-9's already-de-risked mechanism
lane: introspection-self-model
backlog: research/coordination/scaffold_retirement_backlog.md rank 9 ("metacog confidence host formula")
flag: BRAIN_METACOG_SPIKING_MARGIN (stays default-OFF)
backend: numpy
runner: research/runners/_metacog_spiking_margin_prodflip_verify.py
seeds: 42, 43, 44, 100, 101, 102
artifacts:
  - research/findings/raw/_metacog_spiking_margin_prodflip_verify/full_run.json
supersedes-question-left-open-by: research/findings/2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md
---

# Metacog spiking recall-margin: safe to flip default-ON? NO — the ambiguous-band residual has a real, one-directional overconfidence cost at the conversational surface (3/6 seeds)

**Verdict: NO-GO on the production-default flip.** The underlying mechanism's own PARTIAL de-risk
(`2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md`) remains unchanged and valid as a de-risked,
validated **opt-in** (`docs/TERMS.md`) — this finding is scoped narrowly to the separate question this campaign
tasked: is flipping `BRAIN_METACOG_SPIKING_MARGIN` to default-ON **safe**, verified through the real production
conversational handler rather than the composer-in-isolation the original de-risk deliberately used? The
mechanism is genuinely **load-bearing** (6/6 seeds, lesion-tested at the live response surface, not hollow) and
**content-preserving** (the decoded answer is byte-identical in every condition, every seed, confirmed by both
the turn-matched counterfactual method and a literal separate-build cross-check). But on 3 of 6 seeds
(42, 44, 100), a real, non-hypothetical degraded-recall turn reads **CONFIDENT** under the flag-ON evidence
while the shipped host evidence **correctly and honestly hedges the identical turn** — 4 such instances across
42 natural noise-sweep opportunities, and **zero** instances in the reverse (safer, extra-cautious) direction.
Per this campaign's own decision rule ("hollow or regression on any seed = NO-GO"), this is a **regression on
the honesty-hedge's own content** (which turn from a hedge to an unhedged assertion, and vice versa is not
decoration around the answer — it is the answer's epistemic framing) reproduced independently three times now
(the original de-risk's own seed-42 characterization, an isolated diagnostic, and this integrated 6-seed run),
and it is **directionally one-sided toward overconfidence** — the specific failure mode the honesty-boundary
mission treats as the worse of the two. `BRAIN_METACOG_SPIKING_MARGIN` stays default-OFF.

## What this verifies, and how it differs from the original de-risk

The original de-risk built `OneBrainComposer` **directly**, bypassing `_build_tiny_demo` / `ChatBrain` /
`webapp.server.brain_chat` entirely — a documented, deliberate ~40x-faster scope trade-off, since the mechanism
under test lives entirely inside the composer + the metacog organ. That is the right choice for characterizing
the mechanism itself, but it cannot answer whether **flipping the actual production default** is safe: it never
exercises the real conversational handler, the other ~24 default-on faculties (affect/worldmodel/surprise/
comprehension/curiosity/multiref/...), or the module-level session caches `webapp/server.py` uses in production.

This verification (`research/runners/_metacog_spiking_margin_prodflip_verify.py`) builds the REAL production
stack instead: `research.runners.brain_chat_tui._build_tiny_demo(seed, composer_kind="onebrain")` → `ChatBrain`
→ `webapp.server.brain_chat` (the exact code `/api/brain-chat` runs), with every OTHER `BRAIN_*` faculty flag
left **unset** — i.e. at its real production default (all default-ON) — the opposite of the original de-risk's
own isolation env, which explicitly zeroed ~24 of them. For each of the 6 mandated seeds it builds ONE such
chat-brain with `BRAIN_METACOG_SPIKING_MARGIN=1` (every role chip then carries BOTH the always-computed host
margin fields AND the additive `margin_spiking`), asks the real handler a clean query plus a 7-level synaptic-
noise sweep (mirroring the original de-risk's own sigma range, spanning clean → the characterized ambiguous
middle band → clearly-degraded) plus one turn with an INTEGRATED lesion (forcing
`RFPhasorComposer._spiking_margin`'s lesion path **during** a live `brain_chat` call, not a standalone unit
call), and for every turn reads BOTH the REAL flag-ON response (`resp["metacog"]`, the hedge actually prepended
to `resp["answer"]`) and a counterfactual flag-OFF verdict recomputed from the SAME `resp["activity"]` trace via
the pre-2026-09-05 host-only preference chain, fed through the SAME production `MetacogProductionOrgan`
singleton — a turn-matched, no-separate-build-confound comparison. A literal, fully-separate fresh-build pair
at seed 42 (flag unset vs. `=1`, two independent processes) closes the loop on realism for that method.

## A methodology bug found and fixed mid-verification (worth recording, not just banking)

The FIRST full run of this script showed 5 of 6 seeds reading `metacog=None` on **every** turn, including the
clean query — despite the SAME query answering correctly with `confident=True` in an isolated single-seed
diagnostic. Root cause: `_ask()`'s session ids reset to `'pf00001'..'pf00009'` at the top of **every** seed's
sweep, so seed 43's turns reused the IDENTICAL session strings seed 42 had already used. `webapp/server.py`'s
module-level session caches (`_BRAIN_RICH: dict[(session,brain,renderer), RichAnswerComposer]`,
`_SESSION_DISCOURSE`, ...) are keyed on exactly that tuple and never check whether the cached object's own
`chat` matches the `chat` just passed in — so every seed after the first silently ran its turns against the
FIRST seed's stale `RichAnswerComposer` (built around the first seed's `OneBrainComposer` instance) while the
query's dict lookups used the (correct) new seed's composer, a state mismatch that manifested as an empty
confidence read. **Fixed**: `sid_holder` now carries a seed-unique prefix (`f"pf{seed}_"`, passed in rather than
reset per seed) so no two turns anywhere in the whole run — across any seed — ever share a session key,
matching how a real multi-user production deployment never reuses one either. The buggy run was discarded and
the whole sweep re-run; the results below are entirely from the corrected run. (This is the class of bug the
original de-risk's own scope note flagged as a reason to avoid the full stack for the *mechanism*
characterization — it is exactly the class of bug an INTEGRATED verification exists to catch for the *flip*
question.)

## 6-seed results (all from `research/findings/raw/_metacog_spiking_margin_prodflip_verify/full_run.json`)

| seed | build (s) | evaluable turns | lesion collapses confident→hedge | recalled_svo unchanged under lesion | natural-sweep disagreements | false-confidence direction | extra-caution direction |
|---:|---:|---:|:---:|:---:|---:|---:|---:|
| 42  | 161.1 | 9/9 | yes | yes | 1 (sigma1.5) | 1 | 0 |
| 43  | 152.6 | 9/9 | yes | yes | 0 | 0 | 0 |
| 44  | 120.6 | 9/9 | yes | yes | 1 (sigma0.9) | 1 | 0 |
| 100 | 124.3 | 9/9 | yes | yes | 2 (sigma0.9, sigma2.0) | 2 | 0 |
| 101 | 114.1 | 9/9 | yes | yes | 0 | 0 | 0 |
| 102 | 171.1 | 9/9 | yes | yes | 0 | 0 | 0 |

Every seed's `on_vs_off_disagreement_turns` list additionally contains exactly `clean_LESIONED` — the
deliberate, expected disagreement from the integrated lesion test itself (real-ON reads hedge because it was
forced to; counterfactual-OFF reads confident because the lesion patch never touches the host `margin`/
`margin_norm`/`margin_snr` fields it reads) — excluded from the "natural-sweep disagreements" column above,
which counts only the clean + 7-sigma organic degradation sweep. Confusing that deliberate manipulation with an
organic residual would be exactly the "UNDEFINED, not a score" trap this project's `tools/lab.py` discipline
warns against, applied to the wrong axis.

<!--derived-->

**Load-bearing (criterion 2), unambiguous, 6/6:** every seed's `clean_real_ON_confident: true` →
`lesioned_real_ON_confident: false`, `collapsed_to_hedge: true`. The clean balance reads 0.08564659007052086 on
every single seed (the downstream WTA organ's evidence input saturates at 1.0 for any clearly-confident upstream
margin, host or spiking alike, so a "clearly confident" clean turn is indistinguishable downstream regardless of
which channel fed it — the two channels only diverge where one crosses the confidence threshold and the other
does not, exactly the ambiguous-band turns this finding is about) and collapses to 0.01807477725971096 under the
forced lesion on every seed — this is the SAME lesion mechanism the original de-risk validated,
reproduced here at the live conversational-response surface (the actual `resp["metacog"]["confident"]` boolean
and the actual hedge prefix in `resp["answer"]`), not merely an internal composer number.

**No regression on crashes/other-faculties:** zero `errored` entries across all 54 turns × the 6 tracked other-
faculty keys (`affect`/`worldmodel`/`surprise`/`comprehension`/`curiosity`/`multiref`) = 324 checks, 0 failures,
every seed, every noise level, including the lesioned turn.

**No regression on content:** `recalled_svo` is `["brain","use","spikes"]` in **every** turn of every seed —
clean, all 7 noise levels, and under lesion — byte-identical. The seed-42 literal fresh-build cross-check
(`_seed42_fresh_build_crosscheck` in the artifact) confirms this holds across two fully independent processes
too: `content_preserved_across_literal_builds: true` (answer text minus any hedge prefix, and `recalled_svo`,
identical between the flag-unset and flag=1 builds). This is the mechanism's own documented design
(`margin_spiking` is trace-only, never on the answer-selection path) verified empirically at the integrated
level rather than trusted from the docstring.

**The residual, quantified (criterion 3, why this is NO-GO despite the above):** across the 6×7=42 natural
noise-sweep opportunities, 4 read ON-confident/OFF-hedge (seeds 42 sigma1.5; 44 sigma0.9; 100 sigma0.9 and
sigma2.0) and **0** read the reverse (ON-hedge/OFF-confident). Seed 100's own sigma sweep is tellingly
non-monotonic under the ON evidence (confident, confident, **confident**, hedge, **confident**, hedge, hedge,
hedge as sigma rises 0→4.0) — consistent with the original de-risk's own honest characterization of this
residual as two substrates (a continuous cosine-similarity ratio vs. a discrete Izhikevich spike-count ratio)
"measuring correlated but not identical aspects of the same underlying decisiveness," not a short-window
quantization artifact a longer window would resolve. What this integrated run adds is the piece the
composer-level characterization could not supply on its own: on half the mandated seeds, a REAL user-visible
turn — through the actual `/api/brain-chat` handler, with all other faculties live — would read as a
confidently-delivered, unhedged answer under the candidate default, while the CURRENT shipped default honestly
qualifies the identical turn. Zero of the 4 instances go the other way. A residual that is symmetric in
direction is a precision limit you might tolerate for other reasons; a residual that is 4-for-4 in the
overconfident direction, on a component whose entire job is to make the brain **less** likely to overclaim, is
a cost this flip specifically buys — not a coin flip it introduces.

## Why this reads as "regression," matching the campaign's own decision rule

The task framing for this verification requires "answers preserved vs. the shipped default-OFF baseline" and
treats "regression on any seed" as decisive for NO-GO. The substantive fact content (`recalled_svo`) is
preserved on every turn, so a narrow reading of "answer" would report a clean pass. But the honesty hedge is not
packaging around the answer in this project's own framing (`webapp/server.py`'s own comment: "an honest
functional hedge, never a phenomenal claim, never a content change" — additive TEXT the user reads) — flipping
whether it fires on an identical turn changes what the brain tells the user about its own confidence, which
CLAUDE.md's honesty-boundary deliverable treats as exactly the content that must not silently degrade. Reading
"answers preserved" to include this framing is the interpretation consistent with why the mechanism (and its
hedge) exists at all, and under that reading 3 of 6 seeds show a genuine regression.

## Scope notes (documented, not silent)

**LTM not exercised in the 6-seed sweep.** The sweep does not attach the shipped 100k-fact LTM
(`BRAIN_LTM_SHIP_DEFAULT=off`) — a single LTM-attached build was independently timed at >7 minutes (vs. ~150-170s
without), an order of magnitude beyond what a 6-seed × 9-turn sweep can afford, and orthogonal to this flag by
construction: `margin_spiking` populates identically on whichever composer a query resolves through
(`OneBrainComposer._block_role_scores` for the buffer tier this sweep exercises,
`RFPhasorComposer._cleanup_all_score_stats` for the `ShardedPhasorStore`-backed LTM tier — both gated by the
SAME `spiking_recall_margin` flag, confirmed by inspection), and every query here resolves via the buffer tier
directly (never falling through to LTM). Given the NO-GO verdict already stands on the buffer-tier evidence
above, the LTM-attached `--ltm-smoke` path was not run to completion this session.

**Single query, single fact family.** All 60 turns ask the same clean base query ("what does the brain use")
under 7 synthetic-noise levels + a lesion — matching the original de-risk's own methodology (its calibration
script used the identical query against the identical real handler) and keeping the run tractable, but it does
not sample the tiny-demo's other 4 stored facts. Given the residual's mechanism (a substrate-level property of
the Izhikevich winner-vs-runner-up competition under degradation, not a property of this specific SVO) and its
reproduction across 3 independently-varying seeds, broadening the query set is expected to characterize the
residual's exact rate more precisely, not to overturn the directional finding.

**A pre-existing, environment-level gap, not this flag's doing.** Every build in this worktree logs `[webapp]
ONEBRAIN XEDGE build FAILED -> degrading to standalone organs (FileNotFoundError: ... data/corpus/
tinystories.txt)` — `data/corpus/` is gitignored (a large data file provisioned per-machine, like the LTM
`sim-data` bundles), and this worktree checkout does not have it. The code's own designed degradation ("gracefully
degrading to standalone organs") fires identically in every condition and every seed here, so it cannot bias the
ON-vs-OFF comparison, but it does mean the merged XEDGE consolidation path for comprehension/dopamine-credit was
not exercised by this run.

## What changed in the repository

`research/runners/rf_phasor_composer.py` and `research/runners/metacog_production_organ.py`: the flag was
flipped default-ON in this worktree to test the actual flipped-default code path (not merely simulated via env,
per the task's explicit instruction), verified as above, then **reverted to its original default-OFF state**
given the NO-GO verdict — each site's comment now records the flip attempt, the verdict, and points here, so a
future agent does not re-attempt an unexamined identical flip. `research/runners/_metacog_spiking_margin_prodflip_verify.py`
is new (the integrated verifier itself, reusable for a future attempt at this residual — see below). No `sim/`
edit. No production call site sets `spiking_recall_margin=True` or the env var by default; the mechanism remains
reachable only via the documented env escape, exactly as before this verification.

## Next rung (not attempted here)

The residual is a genuine precision limit in the spiking channel's own decisiveness read near a shared decision
criterion (the original de-risk's own "two-clocks-near-a-boundary" framing, cf. Green & Swets 1966), not an
implementation bug — recalibrating `SPIKING_MARGIN_LO`/`HI` cannot fix a residual the original de-risk already
showed a longer integration window does not resolve. A directionally-aware recalibration — biasing the
confident/hedge threshold crossing so that a genuine disagreement in the ambiguous band defaults toward the
SAFER (extra-hedge) outcome rather than the current apparently-unbiased-but-empirically-one-sided crossing —
is the concrete next lever this finding's evidence points to, and this runner (with the noise-sweep + lesion +
counterfactual machinery already built) is reusable to re-verify it without rebuilding the integration harness.
