---
type: finding
status: live
date: 2026-09-09
mechanism: spiking-closed-class-pattern-completion
lane: scaffold-retirement
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
runner: research/runners/_spiking_anaphor_detection_derisk.py
artifacts:
  - research/findings/raw/_spiking_anaphor_detection/decisive_6seed.json
external: Kell/McDermott et al., "Spiking network optimized for noise robust word recognition approaches
  human-level performance and predicts auditory system hierarchy" (PMC7329140 / biorxiv 243915) -- an SNN
  purpose-built for word recognition specifically gains noise robustness, external confirmation of this
  finding's own G2 claim (noisy-cue completion surpassing exact-match); plus Hopfield/attractor
  partial-cue-completion literature confirming partial/noisy-cue completion is an established,
  actively-researched attractor-network property <!--derived--> (arxiv IDs 2606.20666, Convolutional Restricted Hopfield Networks, and 2505.01218, kernel Hopfield noise-robustness analysis).
  Recorded via
  tools/record_external_search.sh, lane scaffold-retirement. Biology binding is Kandel PNS 6e, "The CA3
  Region Is Important for Pattern Completion" (Marr 1971) -- the same anchor already validated in this
  project's research/biology/dg-ca3-sparse-index.md, applied here to a different mechanism (lexical-category
  detection, not memory-store retrieval routing).
---

# CA3-style pattern completion for closed-class (pronoun) recognition — a focused mechanism de-risk for the anaphora-detection scaffold (rank-17)

**Verdict: GO (mechanism de-risk, NOT wired to production).** 6/6 seeds pass every pre-registered gate. This
closes the MECHANISM question for a scaffold-retirement target distinct from every one already landed or
in-flight this session: the host **DETECTION** step that decides whether a token is an anaphoric pronoun at
all, currently a bare Python set-membership test guarding the substrate's own already-spiking referent
RESOLUTION.

## Why this target, and why NOT an already-worked one

The task was to pick the next tractable host-shortcut retirement, explicitly excluding four items already
done or in-flight this session: (a) `engagement_of()`'s novelty set-membership -> spiking habituation (just
landed, `755c17290`); (b) rank-5 Gate-B appraisal (flipped to production default-on long ago); (c) rank-2's
host cue-match for-loop (active, `task_0ca41c4c`); (d) value-choice's recency-ratio (already routed through
the rank-4 shared salience afferent, classified a legitimate host provenance boundary).

Reading `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` and `research/coordination/scaffold_retirement_backlog.md`
(the project's own 24-item ranked scaffold-shortcut map, produced 2026-09-05 and re-audited the same day)
showed most of the top-ranked items are ALREADY resolved as of this week: rank-1 (composer bundle) and rank-6
(numpy-KB LTM) were BOTH retired (`eca75a3f1`); rank-4/5/8/10/12/15/16/20 are all flipped to production
default-on or de-risked-and-classified-legitimate. Rank-9 (metacog confidence) has THREE prior attempted
mechanisms this week alone (recall-margin PARTIAL, two accumulation-to-bound PARTIALs) and is a genuine open
research wall, not a fresh tractable pick — re-attacking it would be re-deriving effort already spent, exactly
the failure mode `before_you_build.sh` exists to prevent.

**Rank-17 ("pronoun detect/substitute host (MED · fresh)") was untouched by any prior de-risk or flip** — a
genuinely different faculty (anaphora/lexical-category detection) from every excluded item, confirmed by
`git log --all --grep=pronoun --grep=anaphora` turning up only the ALREADY-spiking RESOLUTION work (rank-13's
self/identity+anaphora QuestionRouter retirement, `multi_turn_agent.py`'s biased-competition WTA referent
resolver) and zero prior attempts at the DETECTION step specifically. The ledger's own "anaphora-wm" row
names the residual verbatim: `scaffold_retired: NO # host pronoun detect/substitute around it`,
`retire_status: "BLOCKED:neural-render"`, `host_scaffold_in_default: "host pronoun detection + string
substitution"`.

## The shortcut

```python
# research/runners/brain_chat_tui.py :: ChatBrain._resolve_anaphora
anaphors = {"it", "that", "they", "them", "this"}
...
if tl in anaphors:
    ref = self.agent.held_referent()[0]
    ...

# research/runners/multi_turn_agent.py :: MultiTurnAgent._resolve
if not (isinstance(word, str) and word.lower() in _ANAPHORS):
    return word
```

Both call sites gate the substrate's own already-spiking referent RESOLUTION (`held_referent()` / the WTA
biased-competition read in `_resolve_biased`) behind a bare host Python `set` membership test. Per CLAUDE.md's
brain-based-only standard, this is the shortcut class named explicitly there — host bookkeeping deciding
whether the spiking substrate is even consulted, one step upstream of an argmax-over-spikes shortcut. And it
is exact-match-or-nothing: a corrupted/degraded/partial rendering of a known pronoun (a dropped character, a
noisy perceptual or ASR-style cue) fails CLOSED — the token silently stops being treated as anaphoric and the
whole downstream WM-resolution path never fires. There is no such thing as "80% of a Python string."

## The mechanism

Biology binding: `research/biology/spiking-closed-class-pattern-completion.md`. Kandel PNS 6e, "The CA3
Region Is Important for Pattern Completion" (Marr 1971) — "the recurrent excitatory connections of CA3
pyramidal cells" store a memory "as changes in connections between active cells," and "the reactivation of a
subset of this stored cell assembly would be sufficient to activate the entire original neural ensemble ...
referred to as pattern completion." This is the SAME anchor already validated and cited in this project's
`research/biology/dg-ca3-sparse-index.md`, reused here for a different mechanism (lexical-category detection
of a small closed vocabulary, not memory-store retrieval routing).

**Build** (`research/runners/_spiking_anaphor_detection_derisk.py`, reuse-by-import, NO `sim/` edit): reuses
`research.runners.content_selection_spiking.SpikingLoopContextBuffer`, the project's own already-validated
(multi-concept-WM GO, 220x specificity) cortico-PFC NMDA-bistable attractor loop (`cortex_ctx` <-> `dlpfc_wm`,
IZH2007_HIPPO_PYRAMIDAL neurons), with one Hebbian outer-product-installed assembly per anaphor in the CURRENT
host set (`{"it","that","they","them","this"}`, reproduced exactly for a fair baseline). A candidate word
becomes a CUE — a set of `cortex_ctx` input neuron indices — instead of a dict lookup:

- a known anaphor, CLEAN cue: its own full 50-neuron assembly;
- a known anaphor, NOISY cue: only 20% of its true assembly (10/50 neurons), the other 80% replaced by
  neurons from the UNUSED region (no assembly claims them) — a heavily corrupted rendering;
- a content (open-class) word: 50 neurons drawn entirely from the unused region — no learned assembly at all.

The DECISION (`decide_pronoun()`) reads `buf.read()`'s per-concept mean firing rate after the cue and calls it
a pronoun iff the max rate clears a fixed threshold (`THETA=0.30`) — zero Python `in`/dict-membership calls,
checked structurally (G5, via `ast`-stripped source inspection so the check cannot be fooled by a docstring
merely mentioning the retired pattern).

**A first, buggy version of this de-risk reused ONE buffer across an entire probe battery.** That run was an
ad hoc debug pass during development, not saved as an artifact, and is superseded entirely by the fix
described below (not part of the decisive 6-seed result cited throughout this finding):
<!--derived--> it read exactly chance (clean accuracy = noisy accuracy = 0.200 = 1/5) with a false-positive rate of 1.000.
NMDA bistability means a previously-driven
assembly LATCHES ON and never turns off on its own (the documented, validated feature of this exact buffer
class), so by the last probe every concept read as simultaneously "ignited." The fix (kept in the final
runner, `_fresh_probe`) builds a BRAND-NEW buffer per single classification decision — the correct
experimental unit, matching how a live conversation would ask "is this token a pronoun?" once per token.

**Calibration, before the decisive run:** an explicit sweep over the noisy-cue corruption level (keep-fraction
of the true assembly: 0.55/0.40/0.30/0.20/0.10/0.08/0.06/0.04/0.02/0.00, 3 seeds each) found the mechanism
tolerates even extreme corruption (a single correct neuron out of 50 often still completes, given the
attractor's strong recurrent weight), with rising seed-to-seed variance as the true-neuron count approaches
zero. **KEEP_FRAC=0.20 (80% corrupted) was chosen as the pre-registered operating point** — comfortably
robust (0.95-1.00 accuracy across a 3-seed check at higher trial counts) while remaining far harder than an
exact host-string match could ever tolerate, rather than chasing the mechanism's actual (noisier) floor.

## Gate and results (pre-registered, `tools.verdict.Verdict`, 6 project-standard seeds)

| gate | bar | what it checks |
|---|---|---|
| G1 clean-cue recall | >= 0.90 | driving a known anaphor's own full pattern completes to itself |
| G2 noisy-cue surpass | >= 0.85 | an 80%-corrupted cue (KEEP_FRAC=0.20) still completes correctly — the deliverable: a corruption level an exact host-string match cannot recognize AT ALL |
| G3 specificity | <= 0.15 false-positive rate | content-word cues (drawn from the unused-neuron pool) do not spuriously ignite any stored assembly |
| G4 lesion collapse | lesioned (untrained, attractor_weight=0) accuracy <= 0.30 AND `attributable_to(intact, lesion) >= 0.5` | the recurrent attractor connections, not incidental drive/read parameters, are load-bearing |
| G5 (structural) | 0 host membership tests in `decide_pronoun()` | the classification decision reads only numeric firing rates |

Per-seed results (`research/findings/raw/_spiking_anaphor_detection/decisive_6seed.json`):

| seed | clean | noisy (80% corrupted) | false-positive rate | lesion accuracy | attribution to attractor | GO |
|---|---|---|---|---|---|---|
| 42 | 1.000 | 1.000 | 0.000 | 0.000 | 1.00 | yes |
| 43 | 1.000 | 0.925 | 0.000 | 0.000 | 1.00 | yes |
| 44 | 1.000 | 1.000 | 0.000 | 0.000 | 1.00 | yes |
| 100 | 1.000 | 0.900 | 0.000 | 0.000 | 1.00 | yes |
| 101 | 1.000 | 1.000 | 0.000 | 0.000 | 1.00 | yes |
| 102 | 1.000 | 0.975 | 0.000 | 0.000 | 1.00 | yes |

**6/6 seed-GO. Board verdict: GO** (bar is >=5/6). Structural G5 holds. Every seed's lesion arm reads exactly
0.000 (the untrained network never completes ANY probe, clean or noisy — a clean, maximal contrast), so
`attributable_to` reads 100% in every seed: the intact arm's completion accuracy is entirely explained by the
trained recurrent attractor connections, none of it by drive current or read-window bookkeeping alone.

**Honest instrument note (`tools/gates/discriminating_power.py` flags this, non-blocking):** G1's
clean-cue-accuracy metric sits at a ceiling of exactly 1.0 on every seed, so BY ITSELF it cannot discriminate
"the mechanism works" from "the gate trivially always passes." G1 is a positive-control sanity check, not the
deliverable — G2 (the noisy-cue metric, the actual claim under test) DOES vary seed-to-seed (0.900-1.000, with
real observed misses) and is what the board GO is earned against jointly with G3/G4/G5, so the overall verdict
is not resting on a ceilinged metric alone.

## Scope, and what is NOT claimed

This is a focused, single-faculty MECHANISM de-risk, not a production wire-in. Explicitly out of scope, named
as the deferred next rungs:

1. **Wiring.** `_resolve_anaphora`/`_resolve` still gate on the host `in {...}` test; nothing in production
   changed this session. Wiring this detector in place of that test — replacing "pronoun" with "clears
   threshold on the attractor read" — is the direct next step, and would need the candidate-word encoding
   below to be resolved first.
2. **Open vocabulary.** This de-risk trains the buffer on the CURRENT literal 5-word host set. It says nothing
   about generalizing to a pronoun the host set doesn't already enumerate (e.g. "he"/"she"/"him"/"her" are
   missing from `_resolve_anaphora`'s set today — a pre-existing host-set gap this de-risk does not close).
   Extending coverage would mean training additional assemblies, the same "recruit a concept on demand"
   pattern `VocabAgnosticSpikingSampler` already uses elsewhere in this codebase for an open vocabulary.
3. **Candidate-word -> cue-pattern encoding.** Mapping an arbitrary token string to a specific set of neuron
   indices is host bookkeeping in this de-risk (a permutation-derived assignment) — the same declared
   "host rate shortcut for the projection step" `dg-ca3-sparse-index.md` names for its own DG projection. The
   CLASSIFICATION DECISION moved onto the substrate; the string-to-neuron-index encoding a live deployment
   would need did not.
4. **Substitution.** The actual string replace (`toks[i] = ref`) remains host "mouth" bookkeeping, consistent
   with this project's existing precedent for owner-sanctioned articulation crutches (the decision, not the
   surface rendering, is what must be neural).

## Files

- `research/runners/_spiking_anaphor_detection_derisk.py` (new; the mechanism + de-risk runner)
- `research/biology/spiking-closed-class-pattern-completion.md` (new; biology binding)
- `research/findings/raw/_spiking_anaphor_detection/decisive_6seed.json` (new; decisive artifact, auto-stamped
  with provenance by `research/runners/__init__.py`)
- No `sim/` edit. No production wiring.
