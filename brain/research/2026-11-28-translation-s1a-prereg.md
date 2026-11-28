# Pre-registration: translation layer S1a, dictionary free (can any residual write carry a told fact?)

STATUS: REGISTERED 2026-11-28 (pre-data review PROCEED-WITH-CHANGES applied before any model run; engine spec
`tools/engine/specs/translation_s1a.yaml` `status: REGISTERED`). Fixes: (B1) the scored DELTA family gets its own gated
anchor link, DELTA-NULL (the DELTA driver with nothing forced) must reproduce OFF (5.2, U1a), and REPRO gates on all 20 rows
per life instead of 8/7/7 told rows; the reviewer's literal fix (gate DELTA on the banked ORACLE told rows) was not taken,
because the banked battery is sequential with carried state (banked OFF told rows match the fresh-state H rows on 12/22), so
it is reported only; (B2) the engine smoke runs every heavy path once (`--gains 1`, new CLI flag) and a pick fixes the
six-seed memcap at memory_gb 13; non-blocking: REPRO frac None no longer raises, BARRIER needs no positive finding (B4),
`acts` defined (5.7), H1 and H3 need the clustered-bootstrap 5th percentile > 0, H3's effective b >= 8 stated, the
balanced-discordance figure corrected (24, not 26), "fixed family" wording, the off-frame file copied in the 52b846adb rerun,
bundle glob `u_s*.json`, DELTA - SHAM-DELTA on the sham population, residual reads moved to CPU.

Design: `brain/research/2026-11-28-translation-layer-DESIGN.md` section 3, stage S1a (rev 2, recheck PROCEED-WITH-CHANGES).
Capability: Memory selection (pattern completion and separation). Code: `brain/experiments/translation_s1a.py` (imports
`memory_ceiling`, `routing_replay`, `voice_ceiling_oracle`, `voice_tag`; edits none of them). Tests:
`brain/tests/test_translation_s1a.py`. Forward passes only; nothing is trained; no brain default changes.

## 1. Record brief and question

- 11-27 memory ceiling (`results/2026-11-27-memory-ceiling-read.md`): forcing the right episode helps +17/217 told rows and the
  forced memory's user-voice answers copy the stored telling verbatim (7/7). UNDEFINED by its rule.
- 11-27 voice-direction ceiling (`results/2026-11-27-voice-direction-ceiling-read.md`): one residual direction at layer 12 moved
  voice and cost knowing (43 -> 36). Class closed. Its harness is the write path reused here (`PlasticCortex` tag hook,
  `Provenance("all")`, every position).
- Local rule v3 matched the telling's activity without answering (design section 4 item 1).
- **Question.** Can any residual write at layer 12 (dense, top-64 raw, random-projection top-64; no learned dictionary) carry
  a told fact into the reply, over hippocampus-off and over a sham of equal norm? It decides whether S0 (dictionary training)
  earns GPU hours. A measurement of a write path, not a memory method.
- Expected-fail (declared): BARRIER is likely (design odds of dense headroom 40%). A residual write does not reach the matrix
  memory a delta writes (design section 4 item 4). A BARRIER is a verdict on residual writing of a pooled telling vector at
  this scale, not on the capability.

## 2. Population, seeds and scale (deviations from the design, stated before data)

- **Lives.** Only the KR lives 319/320/321 (1.5B, DEV split) have banked night snapshots, banked fresh-state OFF/C rows
  (`vc_full_s*.json`) and banked forced-episode rows (memory-ceiling `u_s*_n14_ORACLE.json`). No life of the six project seeds
  42/43/44/100/101/102 has night snapshots (the test2 lives are frozen TEST lives and are not used). **This screen is DEV**: every
  verdict carries `-DEV`, it is a lead, never a generalisation claim. A six-seed claim needs six-seed DEV lives with night
  snapshots (own prereg); the harness takes `--seeds` and labels a six-seed run `SIX`.
- **Scale.** The design says 0.4B. The KR lives are 1.5B (their adapters exist only at 1.5B), so S1a runs at 1.5B, layer
  `n_layers // 2` = 12 (the design's layer). The design's kill of residual writing needs both sizes; a 1.5B BARRIER does not
  end residual writing by itself (section 8).
- **Population.** Every told question of the life from the banked batteries of nights 1..14 (`voice_tag.told_questions`,
  deduplicated): 25 + 22 + 20 = **67 item-lives** (one question per item per life). The design's "about 400 item-lives over six
  lives" misread 67 as per life; six lives would give about 135. An item-life with no stored right episode
  (`life.episode_is_source`) is outside the population of every arm and is counted (`no_right_episode`).
- **Brain.** The banked night-14 snapshot with its A0snap adapters, fresh clock and fresh state per question, opened exactly as
  the voice ceiling opened it (`voice_ceiling_oracle.session`): abstain gate off (checked), consolidated cortex active (checked).

## 3. Arms

All arms: fresh state per question, the question asked through `Brain.hear` (`life.run_battery`), paired reply draws
(`life._reseed(seed, 500 + night)` before every reply).

| arm | what |
|---|---|
| OFF | hippocampus off, nothing written. Must reproduce the banked voice-ceiling H rows (gate U2) |
| C | the item's telling heard first, then the question (voice-ceiling C, ceiling). Gate U4 |
| DELTA | hippocampus on, the right episode forced (several: highest raw similarity to the cue), reinstated by the brain's own `Hippocampus.reinstate` at the shipped gain (today's memory) |
| DELTA-NOTEXT | DELTA with every stored episode's text withheld (`""`) for the whole battery: matrix delta only |
| SHAM-DELTA | another told episode forced (`memory_ceiling.pick_sham`: same strength tercile, nearest age, never the right episode), report only |
| DELTA-NULL | the DELTA driver (hippocampus on, fresh state, the recall override) with nothing forced: recall returns no episode. Anchor link only (gate U1a): must give OFF's replies and onsets |
| DENSE@g | v = R(item) - mean of the other told items' R, R = residual after block 12 averaged over the tokens of `User: <telling>` read from the birth base (live cortex, adapters on); added at layer 12 on every position of the reply read (prompt and generated tokens) |
| SPARSE-RAW@g | the top-64 coordinates of DENSE's v by magnitude, nothing else (no dictionary) |
| SPARSE-PROJ@g | random overcomplete projection P (16384 x C, N(0, 1/C), seed 20261128); code = top-64 of P v by magnitude; written vector P^T code |
| SHAM-X@g | arm X built from another told item of the same life (`voice_ceiling_oracle.wrong_map`: next item in sorted order), same gain, same norm |
| ORACLE-RES | report only: C minus H at the label position for the item's own question (an in-context oracle), at its natural norm |

Norm: every vector arm writes `g * nu * v / |v|`, nu = the life's median |DENSE v| over its told items (one scalar per life,
recorded), so arms differ in direction only. Gains g in {0.5, 1, 2, 4, 8}. "Raw residual difference" (design) is read as the
telling's token-mean residual centred on the life's other tellings (removes the shared `User:` frame; a memory must be computable
from the telling alone, so the question-dependent C - H vector is ORACLE-RES, report only).

On DELTA versus DELTA-NOTEXT: the shipped reply path never feeds episode text back (`brain/hippocampus.py`: text is "for
inspection only"), so DELTA-NOTEXT is expected byte-identical to DELTA. It is run as the design's fair baseline for a text-free
code and as a check: the count of population rows whose replies differ (`text_path_rows`) is reported; a non-zero count is a
finding (some path reads stored text), not a gate.

## 4. Gain (fixed on dev, never tuned on told answers) and perplexity

- Dev bank: `DEV_FACTS36` (original cue), minus any fact whose key is a key of this life's ledger (barnaby on 319 and 320).
  Per dev fact: its statement's vector built exactly as for the told items (centred on the other dev facts, the same P, the same
  nu), written at gain g, the cue asked, strict `correct_answer` against the statement.
- Neutral-text perplexity ratio per (arm, g): exp(token-mean NLL with the write - without), the first 4 off-frame passages of
  `brain/eval/mix_night_offframe.json` (sha256 checked), mean over the first 4 vectors of the set, the live cortex's full-output
  forward. Two tables: dev vectors (for the pick) and told vectors (for admissibility in sections 5.3-5.4).
- **Pick, per life and write arm:** among gains with dev perplexity ratio <= 1.05, the gain with the most dev strict-correct;
  ties to the smaller gain; if none is admissible, the smallest gain (flagged). This is the gain of the primary and Holm
  contrasts. DELTA uses the brain's shipped gain (no fit).

## 5. Measures and the decision rule (fixed before data; code `translation_s1a.verdict`)

Measures per arm on the population: strict `correct_answer` (the battery's `ok`; primary), `knows` (key named), refusals
(`cascade_screen.refusal`), verbatim echo (the whole telling inside the reply, case and punctuation ignored), first-person echo
(`strict_scorer.is_echo`), user voice given knows. Echo is reported, never gated.

Unit of analysis: the item-life (seed, item). Paired tests on item-lives pooled over the lives:
- **exact McNemar**: b = arm X right and Y wrong, c the reverse; two-sided exact binomial p on (b, c) (`memory_ceiling.sign_p`).
- **exact one-sided bounds on b - c**: with n = b + c, U = n (2 p_U - 1) and L = n (2 p_L - 1), p_U / p_L the one-sided 95%
  Clopper-Pearson bounds of b / n (exact, bisection; tested against scipy).
- **item-clustered bootstrap** of b - c (2000 draws, seed 20261128, item names resampled, every life's cell of a drawn item comes
  with it): 5th and 95th percentiles. Items repeat across lives (pixel, admin, marguerite, volvo), hence clustering.

### 5.1 UNDEFINED (any one; checked in this order)
- U0 seeds not exactly the expected set; a required unit missing; units disagreeing on commit, model, layer, night or tag;
  mixed spec within a life; empty population.
- U2 OFF does not reproduce the banked voice-ceiling H rows on any life: reference covers < 0.9 of the questions, or identical
  replies < 0.9, or identical onsets (|diff| < 1e-4) < 0.9 (`voice_ceiling_oracle.repro_ok`).
- U3 engagement fails on any life (section 5.6).
- U1a the DELTA driver link fails on any life: DELTA-NULL's population rows identical to OFF's (reply, and onset within
  1e-4) on < 0.95 of the life's population, or not every population row matched (section 5.2). No fallback: a miss is a
  defect of this harness, not drift.
- U1 the DELTA anchor fails and the fallback fails (section 5.2).
- U4 C below 0.6 on both measures (section 5.2).

### 5.2 Fallback routes (recheck item 1)
- **DELTA anchor, two links.** No banked run asked a forced episode from fresh state, so no banked row is the scored DELTA
  arm's condition. The anchor is a chain:
  - *Link 1, the forced-episode path reproduces (REPRO).* REPRO re-runs the memory-ceiling ORACLE unit of night 14
    (`memory_ceiling.run_arm`, the banked decision file) and compares it with the banked unit row by row. The banked unit has
    20 battery rows per life (8/7/7 told, the rest forced from the same decision file). The gate is per life on all 20 rows:
    anchored if identical on >= 0.9 (18 of 20; two rows of slack). Told-row agreement (n 8/7/7) is reported, not gated (at
    0.9 it would need 8/8 or 7/7, no slack).
  - *Link 2, the DELTA driver adds nothing but the episode (DELTA-NULL, gate U1a).* `delta_battery` with nothing forced must
    reproduce this commit's OFF rows (>= 0.95 of each life's population, replies and onsets). OFF itself is anchored to the
    banked voice-ceiling H rows (U2). With both links, DELTA = an anchored fresh-state path + the anchored reinstatement.
  - *Reported, not gated:* the DELTA unit's replies against the banked ORACLE told rows asked the same question
    (`vs_banked_oracle`, about 22 rows pooled). Not a gate: the banked battery is asked in sequence with the state carried
    between questions and on its own clock; the banked hippocampus-off told rows of the same battery match the fresh-state
    voice-ceiling H rows on only 12/22 (7/8, 1/7, 4/7), so equality is not expected even with no drift.
  If link 1 misses, REPRO2 re-runs the same unit in a fresh work dir and compares every row with REPRO:
  - REPRO2 identical on >= 0.95 of rows (this commit is self-deterministic; the miss is code or numeric drift since the banked
    commit 52b846adb): the screen proceeds, every arm being from this one commit, and the verdict carries `-UNANCHORED`.
    An unanchored verdict is a lead only (`acts` false): it neither runs S0 nor applies BARRIER. Next step, fixed: rerun S1a in a
    checkout of 52b846adb with `translation_s1a.py` and `brain/eval/mix_night_offframe.json` copied in (the off-frame file does
    not exist at that commit; every module it imports does); if that anchors, its verdict replaces this one.
  - REPRO2 below 0.95: UNDEFINED (U1). Nondeterminism is fixed before any S1a.
- **C under 0.6.** C's strict correct rate over the population is the gate (banked: 44/67 = 0.66; s321 alone 10/20).
  If strict < 0.6 but `knows` >= 0.6 (banked 55/67), every contrast below is scored on `knows` instead and the verdict carries
  `-FALLBACK-KNOWS` (lead only, `acts` false: the strict scorer cannot read the facts even in context, so a strict null would be
  uninformative). If both are under 0.6: UNDEFINED (U4); the next step is a population whose facts the cortex can answer in
  context (own prereg), not a relaxed bar.

### 5.3 Primary contrast and multiplicity (recheck item 4)
- **Primary (one):** DENSE at each life's picked gain minus OFF, exact McNemar, alpha 0.05 two-sided, direction b > c.
- **Secondary family, fixed membership, Holm step-down at alpha 0.05** (standard Holm: ordered by p; all at each life's
  picked gain):
  S1 DENSE - SHAM-DENSE; S2 SPARSE-RAW - OFF; S3 SPARSE-PROJ - OFF; S4 SPARSE-RAW - SHAM-SPARSE-RAW;
  S5 SPARSE-PROJ - SHAM-SPARSE-PROJ; S6 DENSE - DELTA-NOTEXT. A secondary "holds" if b > c and its Holm-adjusted p < 0.05.
- Gates that are ANDed into a decision (H2, H4, H5; B1-B3) are an intersection-union test: each is run at its own one-sided
  0.05 level with no adjustment, which can only make the decision rarer. Everything else (DELTA - OFF, DELTA - SHAM-DELTA on
  the item-lives with a sham candidate, C - OFF, DELTA against the banked ORACLE told rows, ORACLE-RES, the per-gain tables, echo, knows, user voice, text_path_rows) is report only.

### 5.4 HEADROOM (S0 earns its GPU hours). All of (recheck item 3: test and threshold per clause)
| clause | test | threshold |
|---|---|---|
| H1 primary | exact McNemar, DENSE@picked - OFF, and the item-clustered bootstrap | b > c and p < 0.05 (with c = 0 this needs b >= 6: p 0.031; b = 5 gives 0.0625), and the bootstrap 5th percentile of b - c > 0 (a gain carried by 2 items right in every life fails) |
| H2 fluency | neutral perplexity ratio, told vectors, DENSE at the picked gain | <= 1.05 on every life |
| H3 not a sham effect | S1, exact McNemar, Holm-adjusted, and the item-clustered bootstrap | b > c and p_Holm < 0.05 (if S1 has the family's smallest p, p_Holm = 6p: with c = 0 this needs b >= 8, p 0.0078, adjusted 0.047; b = 7 gives 0.094), and the bootstrap 5th percentile > 0 |
| H4 not worse than the shipped memory | exact lower bound L of DENSE - DELTA-NOTEXT (correct) | L > -margin (margin below) |
| H5 refusals | exact upper bound U of DENSE - DELTA-NOTEXT on refusals | U < 0.05 x item-lives (3.35 at 67) |

### 5.5 BARRIER (the residual-write path is the barrier; S0 and dictionary training not run). Recheck item 2
The design's "about 0 correct over OFF (under 3 items per life, no better than SHAM)" becomes an equivalence (non-superiority)
test with a stated margin:
- **Margin:** delta = 3 item-lives per life, pooled: 9 at three lives (13.4% of 67), 18 at six.
- **Admissible gains:** for each write arm, every gain whose told-vector perplexity ratio is <= 1.05 on every life.

| clause | test | threshold |
|---|---|---|
| B1 no gain over OFF | for every write arm and admissible gain: exact upper bound U of X@g - OFF, and the clustered-bootstrap 95th percentile | both < delta |
| B2 no gain over its sham | the same for X@g - SHAM-X@g | both < delta |
| B3 judgeable | every write arm (DENSE, SPARSE-RAW, SPARSE-PROJ) has at least one admissible gain | true |
| B4 no positive finding | H1 and the secondaries S1-S5 | H1 false and none of S1-S5 holds (else NEITHER: a significant gain cannot sit beside BARRIER) |

What the margin admits (exact bound alone, balanced discordance b = c): U < 9 holds up to 24 discordant item-lives
(b = c = 12, U 8.67; b = c = 13 gives 9.01);
a clean null (b = c = 0) gives U = 0. b - c = 9 one-way already fails. The bootstrap bound is the stricter one when the gain
sits on few items (8 one-way cells spread over 8 items can resample above 9): BARRIER needs a null that is both small and not
carried by a few items.

### 5.6 Engagement (recorded per row, gated per arm; U3)
- OFF: no row recalled anything and no row wrote a vector.
- C: every population row heard its telling first.
- DELTA-NULL: on every population row exactly one recall call, no forced episode, no reinstatement.
- DELTA, DELTA-NOTEXT, SHAM-DELTA: on every population row (SHAM: rows with a sham candidate) exactly one recall call whose cue
  is that question, a forced episode, exactly one reinstatement through `Hippocampus.reinstate` with a non-zero delta norm, and the
  recalled text equal to the forced episode's (NOTEXT: `""`); NOTEXT's battery records the text as withheld.
- Every vector arm at every gain: the footprint tagged every token it saw (`Provenance("all")`, tagged = tokens > 0); every row's
  written norm equals g x nu (relative 1e-3) and the label position was tagged; not a no-op (onset equal to OFF's within 1e-6
  on more than half of the life's rows).

### 5.7 Outcome words and precedence
UNDEFINED > HEADROOM > BARRIER > NEITHER. NEITHER names every failed clause. Modifiers in order: `-DEV` (KR lives),
`-UNANCHORED`, `-FALLBACK-KNOWS`. Every `-DEV` verdict is a lead, never a generalisation claim (section 2). `acts` says
whether the outcome's section 8 step is taken: true for HEADROOM-DEV, BARRIER-DEV and NEITHER-DEV (their DEV-level steps:
the S0 prereg, S0 deferred, no S0); false under `-UNANCHORED` or `-FALLBACK-KNOWS` (lead only, the section 5.2 next steps
instead) and for UNDEFINED. No rescue, no relaxed bar, no change of arm,
gain grid, margin or population after data.

## 6. What the banked numbers predict (report, not bars)
OFF strict 32/67 (18/9/5), C 44/67, C knows 55/67 (voice ceiling, same code path). The primary needs about 6 more item-lives
right than OFF with none lost; the C ceiling leaves 12 item-lives of headroom over OFF.

## 7. Compute
Per life: 3 dev-fit arms x 5 gains x 35-36 facts, 2 perplexity tables, 5 fixed batteries, 30 vector batteries, ORACLE-RES and
REPRO, about 36 batteries of 20-25 replies at 1.5B CPU fp32 (voice ceiling: 90-300 s per battery) plus the dev fit: about 3-4 h
per life [E], one life per pool node in parallel (`placement: @kr`), forward only, memcap 13 GB = memory_gb (voice ceiling
ran the same session under 14 GB), fixed by the spec's pick, not by the smoke's measured cap. The engine smoke runs every
heavy path once on seed 319: `--units META,OFF,C,PPL,DEV-DENSE@1,DELTA,SHAM-DELTA,DENSE@1,SHAM-DENSE@1,ORACLE-RES,REPRO
--gains 1 --limit 3 --tag smoke` (perplexity, a dev battery of 35 facts, the DELTA family with DELTA-NULL, a vector arm and
its sham, ORACLE-RES, REPRO on its full 20-row battery). Smoke units carry the tag and are never scored. Units are atomic per file
with resume under one spec (commit, questions, layer, gains, k, m, tag); a different spec in the same dir refuses. Host, commit
and device are in every unit. $0.

## 8. What each outcome does
- **HEADROOM-DEV:** S0 (matched-ablation dictionary gate) gets its prereg and the 3090 slot after the twin; the SPARSE-PROJ and
  SPARSE-RAW secondaries say whether sparsity alone already matches dense (design S1b rule).
- **BARRIER-DEV:** S0 and dictionary training are not run now. The next method is design section 4 item 4 (present one frame's
  vector and let the cortex's own delta rule write it), own prereg. Residual writing is ended only by a BARRIER at both sizes
  (a 0.4B S1a needs 0.4B lives with snapshots) and at six seeds.
- **NEITHER-DEV:** reported with its failed clauses; no S0; no rescue.
- **UNANCHORED / FALLBACK-KNOWS / UNDEFINED:** section 5.2.

## 9. Caveats
Three seen DEV lives, items shared across lives, 67 item-lives: weak power (the margin of 3 per life is wide). 1.5B only. The
strict scorer misses paraphrase. One layer. The pooled telling vector loses binding (design section 4 item 2), so BARRIER does
not test ordered frames. DENSE is centred on the life's other tellings (a population statistic of the store; declared). The
random projection's decoder is its transpose, not a pseudo-inverse. Perplexity uses 4 passages and 4 vectors.

## 10. Recheck items resolved
| item | where |
|---|---|
| (1) fallback when DELTA fails to reproduce, and for C under 0.6 | 5.2 (REPRO2 self-determinism -> UNANCHORED lead and the 52b846adb rerun; else UNDEFINED; C knows fallback) |
| (2) equivalence margin with a stated test for the null | 5.5 (delta = 3 per life; exact Clopper-Pearson bound and clustered bootstrap, both) |
| (3) test and threshold for every ANDed requirement | 5.4 and 5.5 tables |
| (4) one primary contrast, Holm for secondaries | 5.3 |

### Pre-data review 2026-11-28 (PROCEED-WITH-CHANGES, two blockers) and what was done
| item | resolution |
|---|---|
| B1 the DELTA anchor never tests the scored DELTA driver | DELTA-NULL link to OFF, gated (U1a); REPRO gated on all 20 rows per life (n stated); DELTA vs banked ORACLE told rows reported only (different condition, 12/22 baseline) (5.1, 5.2) |
| B2 memcap from a smoke that skips the heavy path | smoke widened to every heavy path (`--gains 1`); pick fixes memcap 13 per seed (section 7) |
| N1 frac None TypeError | `(frac or 0)`, falls to REPRO2 / UNDEFINED (tested) |
| N2 BARRIER beside a significant gain | B4 (5.5; tested on H1 and on S2/S4) |
| N3 `acts` under -DEV | defined (5.7) |
| N4 H1 and H3 not clustered | bootstrap 5th percentile > 0 co-required (5.4; tested) |
| N5 H3 threshold under Holm | stated (5.4) |
| N6 "about 26" | 24 (5.5) |
| N7 "fixed order" | "fixed membership", standard Holm (5.3) |
| N8 off-frame file at 52b846adb | copied in (5.2) |
| N9 bundle glob | `u_s*.json` |
| N10 DELTA - SHAM-DELTA population | the sham population (5.3) |
| N11 GPU tensors in the residual read | `.cpu()` |
