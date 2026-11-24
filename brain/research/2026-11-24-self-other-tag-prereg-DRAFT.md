# Capability: Self and identity (self/other tag on memories) + Episodic recall (whose words a memory holds)
# Record: brain/research/2026-11-23-biggest-defect-read.md; brain/research/2026-11-23-recall-voice-dev-prereg.md; brain/experiments/results/2026-11-24-recall-voice-dev-read-1.5b.md (M1 KILLED); 2026-11-16-rehearsal-night-1.5b.md and 2026-11-16-reply-rehearse-1.5b.md (nearest night-hold neighbours, both no flip); 2026-11-17-teacher-night-1.5b.md (NARROWED)
# Expected-fail: yes, named. Routing (near-duplicate separation) is missing and untouched, so intact told (P1) is expected to miss even if echo falls; the tag here is host-assigned by channel, not the cortex's own speaker state
# Source-held replay on K-P: the night learns a telling as heard and holds it as own speech (DRAFT prereg, 2026-11-24)

> **BLOCKED 2026-11-24 by two Fable reviews (do not build).** Hippocampus-off echo in K-P is a whole-night perspective flip
> (pooled trunk echo /24 by night: n5 0, n6 7, n7-11 0, n12 7, n13 2, n14 13, n15 0; s320 n6 7/8 first person, right on n5/n7), not
> item-level learning of told sentences, so a per-telling hold does not target it. Also: Stage 0 kill fires on night-matched rows
> (both-off 2 vs hoff 0 of 72), Stage 1 reads at floor (no flip before n6), no sham at Stage 2, W GPU on hold, the recall-voice
> prereg §4/§5 stop rule (routing next, not a 6th perspective lever) not amended, record check not run; nearest neighbour
> results/2026-11-14-near-miss-guard-1.5b.md (night frame hold, NOT A LEAD) uncited. Next: diagnose the flip nights from banked lives.


**STATUS: DRAFT. Not frozen. Claude reviews (before-build record check, second-fix review) before any build or run.**
Written by a judge over three independent designs; nothing in section 1 has been run. The counts in section 0 marked
"11-24 count" were made on the banked K-P lives with no model, while writing this draft.

## 0. Record brief and hypothesis
- Record check owed: `tools/before_you_build.sh "own-label hold of heard tellings at night"` was NOT run for this
  draft. Record read directly (files in the header, `brain/hippocampus.py`, `brain/sleep.py`, `brain/brain.py`).
- Code facts verified 11-24 in this worktree:
  - `Episode.source` (`hippocampus.py:28`, default "other") is read by nothing; only `brain.py:709` and `:751` set
    "self". Seeded identity episodes (`_grow_self`, permanent, "User: q\n\nAssistant: a") keep the default "other",
    so the field alone does NOT separate heard tellings from the brain's own answers today.
  - The night trains next-token on every position of `f"{e.text}\n\n"` (`brain.py:841`, `sleep.py:66-68`): the
    user's first-person sentence, with nothing limiting what is learned to the heard side.
  - `sleep.py` already accepts `(prefix, reply)` guard pairs: KL to the pretrained cortex on the reply positions
    only (`:49-53`, `:79-82`), weight `kl_weight` when `guard_weight` is None. K-P runs with an empty guard, so the
    hold adds a third forward/backward per night step.
  - Tellings are stored at `brain.py:605-609` as "User: <text>", source never set.
- Prior levers on this defect (none passed): lived, lived_key, speaker_contrast (unslept, 11-14); M1 lived_key slept
  (KILLED: language 1.264x on s321, hippocampus-off echo 86 > 84; echo 71 -> 42, told 135/223); M2 speaker_contrast
  slept (running; read due 11-26). All five act on what is stored or reinstated. None acts on what the night learns.
- Night-hold neighbours: `sleep_rehearse=True` (holds the pretrained reply to each telling, all positions) cut recall
  25 -> 10 of 36; `sleep_rehearse="reply"` NO-GO (R4 4/6; pool replication abstained on every question on 2 seeds);
  teacher night NARROWED (told 7.2 -> 5.6, abstentions up). So a night hold can cost told and raise refusals. Those
  held the prefix "User: <telling>\n\nAssistant:"; none held the telling itself under the own label.
- 11-24 count, banked `lifeKP_s319-321`, told-item probe rows, nights 5-15, `is_echo` (all lines and tags, so the
  arms are not row-matched; indicative only):
  | arm | echo | rate |
  |---|---|---|
  | both-off (pretrained, no hippocampus) | 7/120 | 0.058 |
  | consolidated-off (hippocampus on, pretrained synapses) | 6/72 | 0.083 |
  | hippocampus-off (slept synapses, no fetch) | 77/552 | 0.139 |
  | intact-frozen | 70/264 | 0.265 |
  | intact | 71/264 | 0.269 |
  Reading: a fetch over pretrained synapses echoes little; the echo needs the slept synapses, alone (0.139) and more
  so under a fetch (0.269). The night is a necessary part of the defect in both arms.
- 11-24 count, lived tell rows (trunk): K-P "X too"/first-person echo replies are 3/12 on day 1 (all s319, before
  any sleep, nothing recalled: "I drive a red Volvo too.") and 4/162 on days 2-15. In K-P the lived-day "X too" is a
  pretrained chat habit on day 1 and rare afterwards; the ~40 rows in the M1 read came from M1 storing that reply.
  So no night method is asked to fix "X too" here, and it is not a bar (report only).
- Known split (recall-voice prereg): K-P fetch-hurt 41 = 8 right episode, 28 wrong, 5 no fetch. A method that only
  marks the right episode at reinstatement can reach at most the 8.
- Hypothesis. The slept adapters raise the told sentence as the brain's own continuation because replay never says
  whose words it was. If each heard telling is also held, under the brain's own label, to the pretrained cortex,
  the sentence stays learned as heard and own-voice production of it stays at the pretrained level. Then
  hippocampus-off echo falls toward both-off, and intact echo falls with it if the fetch echo runs through the same
  slept synapses (the consolidated-off row says it may). If hippocampus-off echo falls and intact does not, the day
  side (reinstated user-voice state, wrong episode) is the remaining cause.

## 1. Arms, seeds, device, command
One switch, `Brain(source_hold=None)` (default OFF = the night is byte-identical), two values:
- `"base"`: per night episode with `source == "other"`, not permanent, text "User: <words>" with no own turn in it,
  add the guard pair `("Assistant:", " <words>\n\n")`, read from the birth state. Uses the existing guard path.
- `"reinstated"` (graft from design 1): the same pair, but student and reference both read it over
  `add_delta(base, e.delta, GAIN)`, the state the day-time reply is spoken from. Needs a per-item start state in
  `sleep.sleep` (section 6). Targets the fetch x slept-synapse interaction directly.
No episode key, delta or stored text changes (routing byte-identical to K-P). No own turn is stored (M1's path).

**Stage 0 (host only, $0, mostly done above).** Still owed: the trunk-only, row-matched version of the echo table;
intact echo rows split by `right_episode` / wrong / no fetch (the read's "Not shown"); echo by night (first night it
appears). KILL the method before any GPU if row-matched hippocampus-off echo is not above both-off echo.

**Stage 1 (local 24 GB GPU via `tools/gpu_queue.sh` + `tools/memcap.sh`, <= 3 GPU-hours, no AWS).** s319, K-P
settings, days 1-5 (five nights), four arms on the same lived text:
| arm | night |
|---|---|
| off | K-P |
| H-base | `source_hold="base"` |
| H-rein | `source_hold="reinstated"` |
| sham | the same number of "Assistant:" pairs built from other seeds' tellings (never told to this brain) |
Read after night 5, hippocampus-off and intact, on every told item: mean log-probability of the told sentence after
`User: <its ledger question>\n\nAssistant:` (measurement only), plus told right and echo by section 2. Local fp16 is
not the AWS device; Stage 1 only ranks arms against its own `off` arm.

**Stage 2 (only if Stage 1 passes; AWS g5.xlarge fp16, AMI ami-012ba162b9cd2729c, one life per instance).** The one
Stage-1 arm with the larger echo log-probability drop (ties -> H-base, the smaller change), seeds 319/320/321, 15
days, against banked `lifeKP_s319-321`. Command = the `lifeKP` line with module `brain.experiments.life_kw` and
`--brain-kw-extra {"source_hold":"<value>"}` (compact JSON), launched by a copy of
`brain/research/recall_voice_launch/launch.sh` (arm name SH). Result JSON must carry the same `brain_kw_extra`,
else UNDEFINED. Probe-copy lesion (no extra life): at nights 10 and 15 the intact told rows are also read with the
slept adapters from the K-P life of the same seed swapped in (if `life.py` cannot do this, drop the lesion at DEV
and require it at six seeds). Six seeds 42/43/44/100/101/102 on a fresh split only after a KEEP.

## 2. Count method
`brain/experiments/results/2026-11-24-recall-voice-dev-read-1.5b/count.py`, unchanged, for told right (intact and
hippocampus-off, 223 rows on K-P), intact echo (264), hippocampus-off echo (552), hurt/helped/best. It reproduces
the K-P table to the row. Added in a sibling `count_source.py` (host only, same `is_echo`):
- echo per arm for all five probe arms, trunk rows with no tag only, plus the all-rows version above;
- intact echo rows split by `right_episode` / wrong episode / no fetch, from the probe rows' fetch record;
- refusals on intact told trunk rows (`life_costs.is_refusal`), the answer-rate lens;
- lived tell rows with `is_echo(reply, key, told)` by day (report only);
- guard share: night steps, hold items per night, and the hold loss at steps 0 and last (from `sleep_log`).
Guards via `brain/eval/life_costs.py --dev --arm <SH lives> --b <lifeKP lives>`, `life_leak`, `life_bars`,
`nights[].lang_guard`, as in the recall-voice prereg section 3.

## 3. Bars (KILL criteria first; 3 seeds = DEV, never a claim)
**KILL at Stage 1** (no AWS spend) if any of:
- neither hold arm lowers the hippocampus-off echo log-probability against BOTH off and sham;
- told right (hippocampus-off or intact) falls by more than 1 item against off in the promoted arm;
- refusals on told rows rise by more than 2 against off;
- flag-off night is not byte-identical to the K-P night (one local night, adapter tensors compared).

**KILL at Stage 2** if any of:
- language: `lang_guard` KL at night 15 > 1.25x K-P's on any seed;
- collapse (`life_bars.language`, top reply share >= 0.4): seeds-ever per probe arm > K-P's;
- pooled intact told < 123/223 or pooled hippocampus-off told < 115/223;
- P3 fails (below).
A seed whose both-off rows differ from lifeKP's (G11 setup check) is UNDEFINED, not a result.

**Pass bars:**
- **P3 (primary, the method's own claim):** pooled hippocampus-off echo <= 51/552 (K-P 77 minus one third) and
  <= K-P on >= 2/3 seeds. Floor for scale: both-off 0.058 = about 32/552.
- **P2:** pooled intact echo <= 47/264 and <= K-P on >= 2/3 seeds (the recall-voice bar, unchanged).
- **P1:** pooled intact told >= 138/223 and >= K-P on >= 2/3 seeds (unchanged; expected to miss, see header).
- **Guards, unchanged from the recall-voice prereg:** leak <= K-P + 0.03; G7 <= 69/90 + 0.05; G8a worst arm <= K-P
  + 0.03 per seed; stuck lived-day runs <= 1; refusals on intact told trunk rows reported per seed, and a rise of
  more than 5 on >= 2 seeds blocks KEEP.
- **Lesion (if run):** swapping in K-P adapters raises intact echo back on >= 2/3 seeds; else the effect is not the
  hold's and the read is NARROWED.
Report only: lived "X too" rows by day; echo split by right/wrong/no fetch; paraphrase echoes (one whole-transcript
read of s319 next to K-P s319, before the verdict); hold loss; night time per life.

## 4. Decisions
- **KEEP:** P1, P2, P3 and every guard hold. Consequence: six seeds on a fresh split with a flag-off arm and the
  lesion; a candidate for TEST v3 only, never v2 (frozen). No default flips on DEV.
- **NARROWED (a):** P3 and P2 hold, P1 misses: voice fixed, routing is the limit (the named expected-fail). Next
  method is routing (near-duplicate separation), with `source_hold` carried as a candidate.
- **NARROWED (b):** P3 holds, P2 misses: the night side is fixed and the fetch still speaks in the user's voice.
  Next method is the reinstatement-side tag (design 0's efference-copy axis, whose step 0 screens are local), not
  another night hold.
- **KILLED:** any KILL line. If told fell (the hold fought replay, as `sleep_rehearse` did), the next method is
  source-conditional replay (replay heard episodes from a heard-context state), not a stronger hold.
- A negative is a verdict on this hold, not on self/other tagging.

## 5. Prerequisites, host decision, lever count, timebox, cost
- **Prerequisite supplied:** a self/other tag on memories that consolidation reads (reality monitoring: the
  perceived trace is learned, own production of it is held). **Still missing:** a tag the cortex assigns itself
  (corollary discharge), source inferred at retrieval, own-turn episodes, more than two speakers (CA2), identity
  learned in conversation, near-duplicate separation. Scale is not the named limit.
- **Host decision (one new SCAFFOLDS row):** the host assigns "other" by channel (what was heard) and re-reads the
  stored telling under the label "Assistant:" for the hold. Cap: the label only (the label `contrast_speaker`
  already uses), no content words added, no generated text, one item per episode, weight = `kl_weight`.
  Brain-native substitute: the cortex's own speaker state marks its turns. No reply text, no rewrite of what is
  said, no text retrieved into a reply. Never a claim about experience; replies are functional read-outs.
- **Lever count on D2:** lived, lived_key, speaker_contrast, M1, M2 (pending), this = 6th, and the first of the
  self/other-tag class the recall-voice prereg section 5 allows after M2. Second-fix review required before build.
  If KILLED: no further label or night-hold lever; go to routing.
- **Timebox:** Stage 0 finish 11-25; build + Stage 1 by 11-26 (after the M2 read); Stage 2 launch 11-27, read by
  11-29; hard stop 11-30 00:00 UTC (unfinished = UNDEFINED, instances terminated). Six seeds only if they finish by
  12-03; otherwise after 12-06. Nothing of this on AWS 12-04..12-06 (TEST v2 days).
- **Cost:** K-P lives took about 7.5 g5-hours (~$8). The hold adds one pass per night step, so about 1.5x nights:
  ~10.5 h, ~$11 per life, ~$33 for DEV in one UTC day (dead-man raised to ~750 min; worst case to be recomputed by
  the launcher against the $50/day cap). Six seeds ~$66 over two UTC days. Stage 0/1: $0 AWS. Fits one 24 GB GPU
  (one more short sequence per step; `"reinstated"` moves one stored delta to the device per step).

## 6. Build list (nothing built yet)
- `brain/brain.py` `Brain.__init__`: `source_hold: str | None = None`, validated in (None, "base", "reinstated");
  refuse resume on a changed value (as `keep_bank` does).
- `brain/brain.py` `Brain.hear` (:605-609): set `ep.source = "other"` explicitly on the stored telling (no behaviour
  change; removes reliance on the dataclass default).
- `brain/brain.py` `Brain.sleep` (after the `sleep_rehearse` block): `_source_hold_items(night_eps)` returns the
  pairs for episodes with `source == "other" and not permanent`, text starting "User: " and holding no
  "\n\nAssistant:"; appended to `guard`. For "reinstated", items carry the episode so sleep can build the state.
- `brain/sleep.py` `sleep`: for "reinstated" only, a guard item may carry a start state (or a callable returning
  it); reference and student for that item read from it. Every existing item form is unchanged. No new constant.
- `brain/experiments/life_kw.py`: no change expected (passes kwargs); confirm `source_hold` round-trips.
- `brain/SCAFFOLDS.md`: the row in section 5. `brain/PREREQUISITES.md`: no edit until a result.
- `brain/experiments/results/2026-11-24-recall-voice-dev-read-1.5b/count_source.py` (or a new results dir): the
  section 2 additions; must reproduce the section 0 table on lifeKP before any arm is read.
- A Stage-1 runner (short life, four arms, log-probability read) and an SH copy of the launch script.
- Tests (`brain/tests/test_source_hold.py`): flag off = `sleep()` call arguments and losses byte-identical; hold
  items exclude permanent, self-source and lived-format episodes; one item per telling; "reinstated" reference is
  computed with adapters off over the reinstated state; replay RNG sequence unchanged with the hold on (the guard
  has its own RNG, `sleep.py:57`); `life_kw` refuses TEST splits as before.

## Not verified in this draft
- Whether a hold in the context "Assistant: <sentence>" transfers to the probe context "User: <q>\n\nAssistant:".
  Stage 1 measures exactly this.
- That probe rows carry the fetch record needed for the right/wrong split on banked lives.
- The consolidated-off and both-off rows are fewer and not row-matched to intact; the interaction reading is weak
  until the trunk-only table exists.
- Night-time multiplier (1.5x is an estimate) and the M2 result, which may change the lever count wording.
- Biology pointers in the brief (source monitoring, CA2, corollary discharge) were not re-checked.
