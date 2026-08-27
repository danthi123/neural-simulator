---
type: finding
status: live
date: 2026-08-27
mechanism: comprehension-animacy-cue-lexicon
---

# Comprehension cue-lexicon conversion — open-vocab ANIMACY is corpus-LEARNED, spiking-realized via gap#3-A1's F_anim/F_inanim pools, and wired behind a default-OFF flag

**2026-08-27.** Builds on the 6-seed GO de-risk `research/runners/_comprehension_learned_animacy_cue_derisk.py`
(`research/findings/raw/_comprehension_learned_animacy_cue_6seed.json`, committed on
`research/comprehension-learned-animacy-cue`, fb725a180) that proved open-vocab ANIMACY is learnable from
real TinyStories co-occurrence via label-propagation, on held-out words never given a label. This session (a)
SPIKING-realizes that classification decision by reusing gap#3-A1's already-validated F_anim/F_inanim
coincidence pools, and (b) wires the extended coverage into `comprehension_production_organ.py`
(`comprehension_production_organ.py:41` "VOCAB CEILING" residual) behind a new, default-OFF flag.

## What was converted, and what was not

Every comprehension organ declared the same hand scaffold: `n0 in ANIMACY and n1 in ANIMACY` — a ~19-noun
table. A held-out real word off that table was OUT OF SCOPE. Only the ANIMACY half of that scaffold is
converted here; VERB_SELECTS stays the pre-existing hand-coded closed set (8 verbs) — no GO artifact
validates an open-vocab verb-selects cue, so claiming that conversion would be an overclaim. This rides as a
declared residual, exactly like the D6/D3/T1-6/D2 organs' existing vocab-ceiling notes.

## The mechanism (reused, not invented)

`research/runners/_comprehension_learned_animacy_spiking.py` (`LearnedAnimacyLexicon`):

1. **Offline scaffold** (unchanged mechanism, reused by import from `_comprehension_learned_animacy_cue_derisk.py`):
   a PPMI co-occurrence graph over the ~1500 most frequent content words of the real TinyStories corpus,
   seeded with the small GT_ANIMATE/GT_INANIM word lists, label-spread (Zhou) into a continuous per-word
   score. This is the SAME offline-scaffold split gap#3-A1 declares for its own EM-learned concept signs
   (`_gap3_learned_feature_compat_derisk.py`) — the sign, not the decision, is the pre-computed part.
2. **Spiking realization** (reused by import from `_gap3_spiking_feature_compat_derisk.py`'s `_build`, the
   validated F_anim/F_inanim 2-pool bridge, 6-seed GO, `tests/test_gap3_spiking_feature_compat.py` 7/7): the
   learned score's SIGN drives one pool with a fixed current (gap#3-A1's own `drive` constant — a
   coincidence-style push, not a graded one); the pools compete for 25 ticks; the WINNER, read off
   `cp_firing_states`, is the classification. A word off the learned graph drives NEITHER pool -> the two
   tie at exactly 0 -> ABSTAIN (the no-confab moat).
3. **Lesion** (`set_lesion(True)`): zeroes the drive into both pools regardless of the learned sign, so
   every word ties at 0 and every `classify()` call abstains — verified to STILL HOLD at the moment of
   measurement (re-checked immediately after the call, not merely asserted): `lesion_pre` = 3 x "animate" +
   3 x "inanimate" on held-out words, `lesion_post` = 6 x `None`
   (`research/findings/raw/_comprehension_learned_animacy_spiking_verify.json`).

## Spiking-realization GO (re-running the numpy GO gate THROUGH the spiking read)

`research/runners/_comprehension_learned_animacy_spiking.py --seeds 42,43,44,100,101,102 --k-seed 8`, output
`research/findings/raw/_comprehension_learned_animacy_spiking_verify.json`:

| metric | mean(6 seeds) |
|---|---|
| spiking `classify()` accuracy on held-out (never seed-labelled) words | **0.8030303030303031** |
| shuffled-graph control (same anti-cheat, spiking read) | **0.5037878787878788** |
| gap (spiking − shuffled) | **+0.2992424242424243** <!--derived--> |
| abstain rate on held-out words | **0.0** |

GO-gate (learned>=0.75 AND shuffled<=0.60 AND gap>=0.15): **GO**. Per-seed spiking accuracy
(0.8636363636363636, 0.8181818181818182, 0.7954545454545454, 0.8636363636363636, 0.7045454545454546,
0.7727272727272727) is numerically IDENTICAL, seed for seed, to a fresh numpy-only re-run of the original
de-risk on the same corpus snapshot — the spiking WTA readout loses no signal relative to the offline
`score > 0` read it replaces. Independently re-running the original `_comprehension_learned_animacy_cue_derisk.py`
on the current on-disk corpus snapshot (`data/corpus/tinystories.txt`, a gitignored regenerable cache that
can differ in size run-to-run) into
`research/findings/raw/_comprehension_learned_animacy_cue_6seed_resnapshot.json` also reproduces the
committed 6-seed GO's verdict on this snapshot: mean learned=0.8030303030303031,
mean shuffled=0.49999999999999994 (5/6 seeds individually >=0.75, the seed-101 exception at 0.7045454545454546;
the originally-committed artifact's snapshot had 6/6 >=0.75). This corpus-content sensitivity is a genuine,
minor residual — the MEAN-based GO gate holds either way, but a per-seed floor is snapshot-dependent.

## Wire-in (`comprehension_production_organ.py`), behind `BRAIN_LEARNED_ANIMACY_CUE` (default OFF)

A single choke point, `_animacy_of(n)`, replaces every `n in ANIMACY` membership test (`_lemma_noun`,
`competent`, the OOV branches of `judge`/`repair_target`): it checks the hand ANIMACY table first (always,
byte-identical), and only when that misses AND the flag is on, falls through to
`LearnedAnimacyLexicon.classify`. For the actual spiking margin read (`_read`/`_read_per_noun`), a new
`_evs_for_organ` helper reuses `cue_evidence`'s own `permute_map` parameter (built for its permuted-cue
anti-cheat, repurposed here as designed) to remap a learned-covered noun to a canonical proxy of the SAME
category ("dog" for animate, "ball" for inanimate) — so the untouched, already-validated
`SpikingRoleCompetition` (D4's mean_auc_semantic=1.0, mean_auc_lesion=0.5 circuit, per the pre-existing
`research/findings/raw/_spiking_comprehension_monitor.json`) reads the correct signed vote for it, without
editing `_phaseB_multicue_competition_spiking_derisk.py`.

## Flag-OFF byte-identical (verified in the data, not inferred from reading the code)

Two independent checks, both exact compares:

1. **Organ-level, isolated from conversational state.** `organ.judge(...)`/`organ.repair_target(...)` on the
   4 sentences from `_gateB_repair_production_verify.py`, captured to JSON (full float precision, `sort_keys`)
   on the pre-edit code (`git stash`) and on the post-edit code with the flag unset. `diff` reports **zero
   lines of difference** — an exact match, not an eyeballed one.
2. **Full production-turn regression** (`research/runners/_gateB_repair_production_verify.py`, through
   `webapp.server.brain_chat`): all 6 existing checks (role-agent-targeted, role-animate-generic,
   OOV-token-named, no-false-repair-on-comprehensible, lesion-collapses, flag-off-bare-abstain) still
   `ALL_OK=true` post-edit. One UNRELATED field differed between the pre-edit and post-edit FULL-PIPELINE
   runs: the DA-mode engagement suffix on the "comprehensible" example ("The wolf biteses apple." vs "...—
   worth going further here."). Isolated with check (1): the comprehension organ's own outputs for that exact
   sentence are byte-identical between the two code versions, so the suffix difference is NOT caused by this
   change — it is the project's own documented class of chaotic run-to-run numerical jitter across a
   multi-turn conversation (`research/FAILURE_LOG.md`, 2026-08-25 gap5-store entry: "chaotic spiking...
   amplify... per-step jitter into a different store"), reproduced here in an unrelated downstream module
   three turns after the sentence in question. Check (1) is the load-bearing byte-identical proof; check (2)
   is corroborating (5/6 fields identical, the 6th explained and isolated).

## Load-bearing verification (vary the cue, then lesion it)

`research/runners/_comprehension_learned_animacy_wire_verify.py`, output
`research/findings/raw/_comprehension_learned_animacy_wire_verify.json`. `"the monkey carries the cup"` —
"monkey" is a real corpus word NOT in the hand ANIMACY table (verified: `"monkey" not in ANIMACY`);
"carry"/"cup" are hand-table-covered.

| condition | `competent()` | `judge()` |
|---|---|---|
| flag OFF (default) | `False` | `None` (out of scope, unchanged) |
| flag ON | `True` | `{margin: 0.3375, threshold: 0.24861111111111112, comprehended: True, ...}` |
| flag ON + `BRAIN_LEARNED_ANIMACY_LESION=1` | `False` | `None` |
| flag OFF again | `False` | `None` |

The flag-ON row shows the extended coverage is LOAD-BEARING (the organ now judges a sentence it previously
passed through unchanged). The lesioned row is an exact dict match to both flag-OFF rows
(`lesioned_reverts_to_flag_off_exact_match: true` in the artifact) — the diff this flag introduces VANISHES
under the lesion, not merely shrinks.

## Moat check (0-confab on genuinely unknown words)

Same artifact, `moat_check_oov`. `"the wug blickets the glorp"` (verb and both nouns off the learned graph,
flag ON): `classify("wug")` = `classify("glorp")` = `None` (abstain — neither word is in the learned
lexicon's corpus vocabulary, so no current drives either pool and the pools tie at exactly 0). `judge()`
returns `comprehended: False` (margin=0.026388888888888892, well below the 0.24861111111111112 threshold)
and `repair_target()` correctly names both as OOV tokens (`loadbearing: "host_lexical"`). The learned cue
never invents a category for a word it has no evidence for.

## Status against the load-bearing terms (`docs/TERMS.md`)

This is **wired** in the sense of reachable from `/api/brain-chat` on some request (the code lives inside
`comprehension_production_organ.py`, already reachable per its own docstring) — but the flag defaults OFF,
so it is **NOT on-by-default**, and the hand ANIMACY table is not removed (an EXTENSION, not a replacement),
so it is **NOT scaffold-retired**. Per `docs/TERMS.md`'s "integrated / production-default" row, the correct
status is the partial one: wired (default-off), not integrated. Flipping the default is a follow-on decision
(it changes the comprehension organ's behavior on live traffic, and the calibrated threshold + proxy-word
trick should get a wider battery + a real spiking verb-selects conversion first).

## Files

* `research/runners/_comprehension_learned_animacy_spiking.py` — new: `LearnedAnimacyLexicon`
  (offline label-propagation scaffold + spiking F_anim/F_inanim classification decision) + a
  verify/GO-gate `main()`.
* `research/runners/comprehension_production_organ.py` — `_animacy_of`, `_evs_for_organ`,
  `learned_animacy_cue_enabled`, `learned_animacy_cue_lesioned`; `_lemma_noun`/`competent`/`judge`/
  `repair_target`'s OOV branches converted to the single `_animacy_of` choke point.
* `tests/test_comprehension_learned_animacy_cue.py` — new CI guard, 8/8 passing (held-out coverage,
  off-graph abstain, lesion collapse, flag-off byte-identity, flag-on load-bearing, lesion-reverts, moat).
* `research/runners/_comprehension_learned_animacy_wire_verify.py` — new: produces the byte-identity /
  load-bearing / moat artifact below.
* `research/findings/raw/_comprehension_learned_animacy_spiking_verify.json` — the spiking-realization
  6-seed GO artifact (with `.prov.json` sidecar, auto-attached by `research/runners/__init__.py`).
* `research/findings/raw/_comprehension_learned_animacy_cue_6seed_resnapshot.json` — the original
  label-propagation de-risk re-run on the current corpus snapshot.
* `research/findings/raw/_comprehension_learned_animacy_wire_verify.json` — the organ-level load-bearing +
  moat-check artifact.

## Residuals (declared, ride existing burn-down items)

* VERB_SELECTS stays hand-coded (no validated open-vocab mechanism for it yet).
* The corpus (`data/corpus/tinystories.txt`) is a gitignored, regenerable cache; a re-fetch can shift
  per-seed accuracy near the 0.75 floor even though the mean-based GO gate is robust to it (measured: one
  seed at 0.7045454545454546 on the current snapshot vs the committed artifact's all-six >=0.75). Worth
  pinning a corpus checksum for this specific de-risk if it becomes load-bearing on a production default.
* The proxy-word remap ("dog"/"ball") is a clean reuse of `cue_evidence`'s existing `permute_map` parameter,
  but it means the SpikingRoleCompetition never sees the ACTUAL open-vocab word's own spiking representation
  during the role-competition read — only its category, via a stand-in. A fuller closure would drive a
  dedicated cue population per open-vocab word rather than borrowing a hand-table proxy's population.
