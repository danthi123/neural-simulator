# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-21-stage1-registered-test-1.5b.md (language guard FAIL); brain/research/2026-11-19-multiweek-life-design.md (section 13); research round 11-21 (language-drift diagnosis, `_scratch/langdrift/`)
# Expected-fail: yes, named. Sampled prior texts that look like episodes ("User: my X is ...") fight fact writing, as in `sleep_rehearse` (recall 25 -> 10). Low-diversity samples may leave the format drift uncovered.

# Stage-2 method K-P: a large, resampled keep set drawn from the cortex's own prior (prereg, 2026-11-21, written before any K-P life)

## 0. Record brief

**Record.** The Stage-1 TEST read failed the language guard. Held-out NLL went from 1.93 to 3.8-4.25 and KL reached 13-17x its night-1 value. The M-scale norm cut did not fix it: it erased old facts and new facts together, and the transcripts still failed. Held-out KL was never read under any M-scale arm.

**Measured cause** (offline reads on saved homes, `_scratch/langdrift/drift_read.py`, s319 night 4 and s316 night 15). The keep term holds only its own neighbourhood. KL to the pretrained cortex, by text set:

| Text set | n4 KL | n15 KL | n15 ratio to keep texts |
|---|---|---|---|
| The 3 ORDINARY keep texts | 0.013 | 0.032 | 1x |
| Near-paraphrases of the keep texts | 0.065 | 0.450 | 14x |
| Held-out chat | 0.547 | 2.399 | 75x |
| Plain prose | 1.418 | 4.674 | 146x |

- The drift points toward the replayed genre: unseen same-day episodes sit at KL 2.97, against 0.55 for held-out chat.
- Assistant positions drift more than User positions by night 15 (KL 2.91 against 1.81).
- Each night partly rewrites the function. On night 5, KL(n4 || n5) on held-out text is 0.471, while KL to pretrained rose only +0.28.
- Replay over-memorises: replayed episodes sit at NLL 0.47-0.60, against about 3.0 pretrained.

**Not established (do not lean on these).** That the Adam step size is the cause. That the norm is the cause. That 70-80% of the probability rise falls on episode content: that figure is confounded by function words. That held-out KL alone drives the 6/6 reader preference for the untrained cortex: the record also shows identity loss, confabulation and marker leaks. This is one life per condition on two different seeds, so treat the table as a lead.

**Biology.** Complementary learning systems with interleaving of the cortex's own old structure (McClelland, McNaughton & O'Reilly 1995). Pseudo-rehearsal as a model of sleep (Robins 1995). Spontaneous cortical replay of its own generative model. Citations are from memory and are not yet verified.

**Prerequisites.** Nightly sleep, frozen pretrained cortex with adapters off, and `cortex.speak(lg, st, max_tokens, temperature)` are MET (`near_miss.guard_texts` uses the same path). No new prerequisite.

**Host decision.** A declared scaffold with a cap. It REPLACES the "three fixed chat snippets" row of `brain/SCAFFOLDS.md` (line 13) only when the switch is on. The host fixes four numbers: bank size 512, text length 64 tokens, temperature 1, frame "User:". There is no external corpus, no RAG and no English template. The texts are the cortex's own samples.

## 1. Method

Behind `Brain(keep_bank=None)`. The default None is today's behaviour, byte-identical. The only accepted non-None value is `"prior"`.

Code-level changes:

1. `brain/brain.py`, new constants next to `ORDINARY`: `KEEP_BANK_N = 512`, `KEEP_BANK_TOKENS = 64`, `KEEP_BANK_TEMP = 1.0`, `KEEP_BANK_FRAME = "User:"`, `KEEP_BANK_NIGHT = 64`.
2. `brain/brain.py`, `Brain.__init__`: new argument `keep_bank: str | None = None`. Validate it is None or `"prior"`. Store in `meta.json` the way `replay_priority` is stored, and refuse a resume with a different value.
3. `brain/brain.py`, new `Brain._keep_bank()`. Called only when the switch is on, and only inside `sleep`. On first call it samples KEEP_BANK_N texts from the frozen cortex with adapters off, from `base`, with `cortex.speak(..., max_tokens=KEEP_BANK_TOKENS, temperature=KEEP_BANK_TEMP)`. It runs under `torch.random.fork_rng` and `torch.manual_seed(sleep_seed + 104729)`, so the global RNG and the replay sequence are untouched. It writes the texts to `<home>/keep_bank.json` once, then reads that file on every later call. The bank is fixed per seed for the whole life.
4. `brain/brain.py`, `Brain.sleep`: replace `ORDINARY` in the `replay_into_cortex(...)` call with `generic`, where `generic = ORDINARY` when the switch is off and `generic = random.Random(sleep_seed + meta["sleeps"] + 104729).sample(bank, KEEP_BANK_NIGHT)` when on. A fresh subset each night.
5. `brain/sleep.py`, `sleep()`: no code change needed. The `generic` argument is already a list of texts, the reference log-softmax is already computed once per night with adapters off, and `rng.randrange(len(gen_ids))` already draws one keep text per step. The keep term stays KL(pretrained || adapted) with `kl_weight` unchanged (3.0 at 1.5B). Cost: 64 extra no-grad forwards at the start of the night, plus one longer-text keep forward per step.
6. `brain/eval/life.py`: `--keep-bank prior` maps to `kw["keep_bank"]`, like `--replay-priority`. Add it to the resume check at lines 892-896.
7. `brain/SCAFFOLDS.md`: add one row for the bank (cap: the four numbers above), and a note on the ORDINARY row. Written before any K-P life.

Off-switch proof: a unit test that a 3-night home with `keep_bank=None` is byte-identical to the current head (adapter tensors and `meta.json` equal), and that no `keep_bank.json` is written.

Engagement log: each night records the bank size, the subset hash, and the keep-term mean over the night.

## 2. Screen (DEV only, no registered claim)

- **Seeds.** DEV 319, 320 and 321, paired with the banked all-flags B lives of the same seeds (`lifeS1F_s<S>`, pool runs of 11-21, the same B the D2-PR screen uses; K-P and D2-PR are separate arms against one B). Amended 11-21 before any K-P life: the draft named 322-324, which would need three new B lives.
- **Step 0, one-night gate (before any life).** From a saved B home (`_scratch/stage2/home_s316_n15`, DEV s316 night 15, defaults), run ONE more sleep night twice from the same snapshot and seed: B (ORDINARY keep) and K-P (bank keep), then read `_scratch/langdrift/drift_read.py` on both. Gate: K-P's held-out KL increment over the snapshot (KL(snapshot || after)) is at most half of B's, and the replayed episodes' NLL after the night is within 0.2 of B's (the night still writes). Fail = do not launch the screen; report why.
- **Command.** The design section 13 command on `--split dev`, plus the switch:
  `python -m brain.eval.life --model 1.5b --seed S --days 15 --weekend --split dev --full-probe-nights 1,5,10,15 --frozen-probe --span --transfer --pruned-a --cue-ladder --interview --lang-guard --keep-bank prior --homes ~/brain-lane/workLIFEkp_sS --out ~/brain-lane/outLIFE/lifeKP_sS`
- **Allowed cut.** 10 nights: `--days 10 --full-probe-nights 1,5,10 --interview-nights 5,10 --pruned-a-day 6`. Then the primary read is night 10 only, and the bar below uses night 10 alone.
- **Compute.** Pool CPU or AWS g5, as capacity allows (GPU is off the bus until 11-23). The host is recorded. Expected extra cost over B is 10-20% per night, plus about 20-40 min once for the bank (an estimate, not measured).

## 3. Read-outs and bars (pooled over 3 seeds; the direction is also reported per seed)

1. **Engagement (must hold, or the arm is UNDEFINED).**
   - `keep_bank.json` exists with 512 texts, a different subset is drawn each night, and the keep term is active (mean keep KL on the night's bank subset is not zero from night 2).
   - The bank is not degenerate: at least 90% of the 512 texts are distinct and the mean sampled length is at least 40 tokens.
   - Fewer than 10% of bank texts contain a told fact from that seed's ledger (a check that the bank does not leak the episodes).
2. **Primary, held-out language guard** (`--lang-guard`, `life_langguard.measure()` on the frozen HELDOUT set, every trunk night). On nights 10 and 15:
   - KL at night 15 at most half of B, pooled over the 3 seeds, and below B on 3/3 seeds.
   - Held-out NLL at night 15 at most the midpoint between night 1 and B's night 15 (K-P NLL rise at most half of B's rise), on 3/3 seeds.
   - Night 10 must show the same direction on 3/3 seeds. It is reported with numbers, not a separate bar.
   - Reference values, Stage 1 TEST B: NLL 1.93 -> 3.8-4.25, KL 13-17x night 1.
3. **Guards (any fail = KILL).**
   - H with the hippocampus off, nights 5-15, is not lower than B by more than 0.5/6 pooled per seed.
   - Direct hippocampus-off recall of day-1-2 facts is not lower than B by more than 1/6.
   - Collapse is not worse than B.
   - The stock self-reply leak (`brain/eval/life_leak.py`) is not higher than B.
   - The whole-transcript read (3 sonnet readers plus refuters, or the section 7 reader) finds no new defect class: identity loss, name overwrite, a marker leak, a stuck tail, or sampled-bank phrases appearing in replies.
4. **Secondary, report only.**
   - KL on near-paraphrases of the keep texts, on plain prose, and on Assistant vs User positions (`drift_read.py`).
   - Night-to-night rewrite, KL(n-1 || n), against the net KL rise.
   - Recall of once-told E and F facts, adapter norm, interview AUROC, invented-memory rate.
   - Section 7 blind reader, B vs K-P, both orders (report only at DEV).

## 4. Decision

- **KEEP.** 1 + 2 + 3 hold on DEV: go to a registered six-seed test on a FRESH TEST split (section 9: any method change after a TEST read writes a fresh split). The default flips only on that result (land-result).
- **NARROWED.** 1 holds, 3 holds, 2 is partly met. Two cases:
  - KL falls but NLL does not (or the reverse): the bank covers the distribution but not the format drift. Next lever is F, a per-token replay-loss floor, or T, a nightly function-space trust region on the same bank.
  - 2 is met on 2/3 seeds only: report as a lead and name the seed that failed.
- **KILLED.** 3 fails (recall or H drops, or a new defect class appears), or 1 fails twice after one fix of the bank (the sampled texts look like episodes). Then the next method is the CLS two-rate merge-and-reset.

Timebox: 2 days from launch. The grid is B vs K-P on three seeds. No second bank size, temperature or keep weight is tried before this read.

## 5. Amendment A1 (2026-11-21, after gate run 1, before any bank fix or K-P life)

**Gate run 1 (recorded as run; files `_scratch/langdrift/gate/`, W CPU, seed 316, from `home_s316_n15`).**
- Verdict on the §2 bars: **FAIL.** The held-out increment KL(snapshot || after) is 0.76x B's (bar: at most 0.5x).
  Replayed-episode NLL is 0.506 for K-P against 0.770 for B. That misses "within 0.2", but on the writing side:
  K-P writes more, not less.
- Bank engagement (§3.1): **FAIL.** 450 of 512 texts are distinct (87.9%; bar 90%). Mean length is 27.6 tokens
  (bar 40). 0 of 512 contain a told key of s316 (pass).
- Cause of the bank failure (read from code, `brain/cortex.py` `speak`): `speak` stops at the first blank line or
  "\nUser:". Each text is therefore one short User opener with no Assistant turn. It also runs at speak's defaults,
  top_p 0.3 with presence and frequency penalties 0.4, which narrows it to a few stock openers ("Hi, I'm having
  trouble with my computer"). This bank is not the §1 intent of a temperature-1 sample of the prior over 64 tokens.
- Instrument note (recorded, not used to overturn run 1). The increment counts any change, including a move back
  toward the pretrained cortex. On held-out chat after the night, KL to the pretrained cortex fell from 2.40 to 1.59
  under K-P and rose to 2.80 under B. Held-out NLL fell from 4.18 to 3.43 under K-P and rose to 4.59 under B. From a
  drifted home, the increment bar cannot tell damage from repair.

**Fix of the bank (the one fix §4 allows).**
- The bank is sampled by a plain continuation loop, not by `speak`.
- Settings: `cortex.forward` plus `pipe.sample_logits` at temperature 1.0 and top_p 1.0, with no presence or
  frequency penalty. The loop does not stop at a turn boundary. It stops only at 64 tokens or at end of text.
- Each text therefore runs the full 64 tokens of the prior after "User:", crossing into Assistant and later turns.
- The four host numbers are unchanged (512 / 64 / 1.0 / "User:"). The nightly subset of 64 is unchanged.
- Sampled text is stored as decoded and stripped. Nothing is split at turn boundaries.

**Gate run 2 (rules fixed now, before it runs).**
- Same source home, seed and commands as run 1, with the fixed bank. B is not re-run: run 1's B is byte-for-byte
  the same arm.
- **Launch requires all of the following:**
  1. The fixed bank passes every §3.1 engagement bar (90% distinct, mean at least 40 tokens, under 10% told keys).
  2. The night still writes: K-P replayed-episode NLL is at most B's + 0.2. This is the "the night still writes"
     intent of the §2 parenthetical, stated before the run. Under run 1's literal two-sided reading, a stronger
     write would fail.
  3. Either the original increment bar holds (at most 0.5x B), or held-out KL to the pretrained cortex after the
     night is below both the snapshot value (2.40) and B's after-night value (2.80). The second branch is a
     pre-declared override, because of the instrument note above.
- Every launch through the override branch is labelled "gate-overridden" in the screen's record.
- The screen's own §3 bars are the decision and are unchanged.
- If condition 1 fails, that is the second bank failure (§4 KILLED; next method CLS merge-and-reset).

## 6. Amendment A2 (2026-11-22, after gate run 2, before any K-P life)

**Sampler deviation from A1's wording, found in review before gate run 2.**
- A1 names rwkv `pipe.sample_logits` at top_p 1.0. On CPU that call is greedy. The float32 cumulative sum over 65536
  tokens ends below 1.0, so its cutoff falls on the top token: 200 of 200 draws were the argmax on the 0.1B model.
- The bank therefore samples through `Brain._sample_full`: a float64 full softmax at temperature 1, drawn with the
  seeded numpy RNG. This follows A1's stated meaning (a full temperature-1 sample) and recorded no result before
  the change. A regression test pins it.

**Gate run 2 (W CPU, seed 316, from `home_s316_n15`; files `_scratch/langdrift/gate/`).**

| Measure | Snapshot | B | K-P run 1 | K-P run 2 |
|---|---|---|---|---|
| Held-out KL to pretrained | 2.399 | 2.799 | 1.593 | 0.862 |
| Held-out NLL (pretrained 1.931) | 4.179 | 4.589 | 3.435 | 2.812 |
| Prose KL | 4.674 | 4.474 | 3.224 | 1.657 |
| Keep ORDINARY KL | 0.032 | 0.147 | 0.260 | 0.237 |
| Replayed-episode NLL | 0.599 | 0.770 | 0.506 | 0.451 |
| Held-out increment, ratio to B | | 1.00x | 0.76x | 0.99x |

- Bank: 511 of 512 texts distinct, mean 63.4 tokens, 4 of 512 (0.8%) contain a told key, 29.5% contain
  "Assistant:".
- §5 rule applied:
  - Condition 1 (bank bars): PASS.
  - Condition 2 (replayed NLL at most B's + 0.2): PASS, 0.451 against 0.970.
  - Condition 3: the increment bar FAILS (0.99x). The override branch holds (held-out KL 0.862, below 2.40 and
    below 2.80).
- **Verdict: LAUNCH (gate-overridden).** The screen's §3 bars are unchanged and decide.
- Caveats:
  - One night on one seed.
  - The night repairs held-out drift. That is not the same as preventing drift over a life, which the screen tests.
  - The 3 ORDINARY texts lose ground (KL 0.032 -> 0.237).
  - On a loaded CPU the night took 8.95 h, with the bank taking 2.4 h of that.
