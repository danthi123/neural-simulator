# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-19-life-stage0-1.5b.md

**Dev-stage pre-registration: M-use (content-confirmed use) and M-scale (night budget homeostat)**
Date 2026-11-19. Status: DEV ONLY. No registered claim comes from this document. Stage-0 data are interim. s311 and s312 are done; s310, s313 and s314 are still running on pool40, pool42 and pool41.

## 0. Scope and section 9 amendment (dated 2026-11-19)

This line amends design `brain/research/2026-11-19-multiweek-life-design.md` section 9.3. Stage 2 stays capped at two methods, and they are named here:
- **M-use** is the D1-family method. It stops the reward for a fetch that strength won, not content. It is a neighbour of D1 disengagement, not D1 itself.
- **M-scale** is the D2-family "norm budget" variant. It is a budget on the night's net cortical drive.
- The D1 disengagement and D2 triage methods as written in section 9 are deferred, not run.
- Section 9.8 holds: both switches default OFF, so the shipped hippocampus and sleep are unchanged.
- Each method still needs its `tools/before_you_build.sh` record check before code. The local corpus reported STALE, so refresh it first.

Correction for the Stage-0 note, to land with the first commit: "Bramble fetched 86x" counts probe-copy fetches. Live uses were 10 (strength 1 + 0.5 × 10 = 6.0). The night-1 hubs (Pixel 11/14 on s311, 10/13 on s312) and the soup hub on nights 2-3 formed at 0-1 live uses. Part of the hub is geometric, or forms inside the battery, and is not caused by live use.

Killed levers are not reused: the 11-17 in-context collapse levers and the 11-18 self-teacher levers. S_all and the confidence gate are excluded, because both were tested.

## 1. Final rules

### M-use: content-confirmed use ("strength may not create the win it then rewards")

Fetch behaviour is unchanged. The bar (0.12 for a question, 0.45 for a statement) and the MARGIN of 0.02 still apply to the weighted score `sims * tanh(eff)`, and the winner is reinstated as now.

The use update (`uses += 1`, `strength += 0.5`, `last_used = now`) happens only if both conditions hold:
- **(a)** `argmax(sims) == argmax(score)`. This is brain-internal.
- **(b)** `sims[best] - max(sims[others not excluded]) >= MARGIN_USE`. This is a declared host constant.

If either fails, the result is a weak retrieval: the episode is reinstated, but its strength is not raised and its decay clock is not reset.

- `MARGIN_USE = 0.02` is pre-registered and reuses the existing MARGIN, applied to the raw sims. There is no selection grid. The values 0.0 and 0.05 are labelled ablations only, run on s318 and only if the primary passes.
- The rule applies to every episode, permanent ones included. Permanent episodes keep their strength floor of 3.0, and tanh(3) = 0.995, so the gate has no functional effect on them.
- The inquiry lesson stays outside the episode list, as now.
- Declare it in SCAFFOLDS: "reconsolidation gate: MARGIN_USE host constant; rule (a) brain-internal".

### M-scale: budget homeostat, applied before the night

- At the start of each night, before the replay loop: `n = adapter_norm()`. If a budget `b` exists and `n > b`, scale B by `g = b/n` (`B_k, B_v *= g` in every layer; A is untouched). `adapter_norm` is exactly linear in B.
- After the brain's first slept night with the switch on, set `budget = K * adapter_norm()` and persist it in meta.json.
- If a home with adapters but no budget is opened with the switch on, the budget is the current norm and there is no downscale that night. Log that this happened.
- `K = 1.0` is pre-registered. K is a hand-set host constant: the night-1 norm is mostly set by lr × steps × rank, so it is not a self-calibration. Declare it in SCAFFOLDS. K = 1.5 is the only allowed fallback (kill rule M-scale-2, s318). 0.75 is not run.
- Scale B only, and log ||A|| and ||B|| every night. Scaling both by sqrt(g) is a separate variant. It is allowed only under kill rule M-scale-3, and the same timebox applies.

## 2. Switches (all default OFF; byte-identical when off)

| Switch | Where | Default | On |
|---|---|---|---|
| `Hippocampus.USE_GATE`, `MARGIN_USE` | hippocampus.py class attrs | `None`, 0.02 | `"raw_win"` |
| `Hippocampus.FREEZE_USE` (instrument) | hippocampus.py class attr | `False` | probe copies only (harness read-out) |
| `Brain(use_gate=None)` | brain.py; persisted in meta like `key`, only when non-default | None | `"raw_win"` |
| `Brain(sleep_downscale=None)` | brain.py; meta keys `adapter_budget`, `norm_pre_night`, `downscale_g` written only when on | None | K = 1.0 |
| `--use-gate`, `--sleep-downscale K`, `--frozen-probe` | brain/eval/life.py CLI | off | as above |

## 3. Code changes

**brain/hippocampus.py**
- `recall()`: if `USE_GATE is None and not FREEZE_USE`, run the current code verbatim.
- Otherwise, compute `sims = similarities(...)` once and `score = sims * w`. Apply `exclude` to both (set to -1e9) and call `_compete(score, question, now, sims=sims)`.
- `_compete`: the fetch decision is unchanged. Compute `confirmed` from (a) and (b). Run the use update only if `confirmed` (gate on) and `not FREEZE_USE`.
- Stash `self.last_fetch = dict(idx, sim_raw, margin_raw, margin_w, raw_winner, weighted_only_winner, confirmed)` on every fetch, and also on a non-fetch, with `idx=None`.
- `KeyedHippocampus` overrides only `similarities()`, so it needs no change.

**brain/plastic_cortex.py**
- Add `downscale(g)`, which multiplies B_k and B_v under `no_grad`.
- Add `factor_norms()`, which returns `(||A||, ||B||)` and is used for logging only.

**brain/brain.py**
- Add both ctor kwargs, set on the hippocampus in `_make_hippocampus` so a reloaded home keeps them.
- `hear()`: copy `last_fetch` into `info["fetch"]` only when the gate or the freeze is on.
- `sleep()`: add the M-scale block after `plastic = self._plastic()` and before replay. It consumes no RNG and leaves the existing `fork_rng` unchanged. The budget is set after replay and before the meta save.

**brain/eval/life.py**
- Add the three flags.
- `--frozen-probe` adds an `intact-frozen` arm to every morning battery: a copy with `FREEZE_USE=True`, so the battery does not boost itself.
- Add per-fetch dicts to the lived and probe rows.
- New per-night `life_table` columns:

| Group | Columns |
|---|---|
| Fetching | fetched count, right-fetch count, wrong-fetch count (counts, not only ratios) |
| M-use gate | boosts granted / denied |
| Strength | max non-permanent strength |
| Hub | top-1 share of live trunk fetches; top-1 share of frozen-probe fetches; distinct episodes fetched |
| Adapters | `norm_pre`, `g`, `norm_post`, ‖A‖, ‖B‖ |
| H by age | H split by the day the fact was told (day 1-2 vs day 3+) |

- Existing columns are kept.

**brain/SCAFFOLDS.md**: add two rows, MARGIN_USE (reconsolidation gate) and K (norm budget multiplier).

## 4. Tests (brain/tests, CPU, no model unless noted; the existing 155 must still pass)

The tests assert mechanics only, never transcript content or H counts.

**M-use**
1. **OFF byte-identity:** monkeypatched `similarities()` and a fixed clock, 200 random recalls, gate None versus the current code. The returned episodes and the full episode list (strength, uses, last_used) must be equal.
2. **Hub case:**

   | Episode | Strength | Raw sim | Age | Weighted score |
   |---|---|---|---|---|
   | H | 6 | 0.20 | fresh | 0.20 × 1.0 = 0.20 |
   | R | 1 | 0.30 | 2 days | 0.30 × 0.52 = 0.156 |

   With the gate on, H is returned, and its strength, uses and last_used are unchanged.
3. **Clear raw winner** (0.40 vs 0.30): the winner is boosted.
4. **Boundary:** exactly at MARGIN_USE, and `exclude` is respected in the raw runner-up.
5. **FREEZE_USE:** fetch results match the unfrozen path for the first call, and no episode changes.
6. **Toy loop:** 20 episodes and a synthetic cue stream. With the gate on, max strength ≤ 1 + 0.5 × confirmed fetches. With it off, the hub exceeds that bound.
7. `KeyedHippocampus("S_all")` smoke test.

**M-scale**
8. **OFF byte-identity:** a fake or tiny cortex, a 3-episode store, two nights, `sleep_downscale=None` versus the kwarg absent. Adapter state_dict bytes, meta.json and losses must be identical.
9. `downscale(0.5)`: the norm halves to 1e-5 and A is unchanged.
10. The budget equals the post-night-1 norm. Night 2 with norm ≤ budget leaves the adapters untouched before replay.
11. Patch the norm to 3 × budget and run sleep with steps = 1: `norm_pre_night` is recorded and the post-cut norm equals the budget.
12. The budget survives a home reopen. A home switched on without a budget takes the no-downscale path.

**Smoke**
13. A 0.1B, 2-day life with every flag absent versus every flag explicitly off: the JSON must be identical apart from timings.

## 5. Screens before any new life (offline, on the banked s311 and s312 homes)

Run these on whichever pool node frees first, under `tools/memcap.sh` after `tools/mem_ok.sh`. They use the snapshot or trunk homes in `~/brain-lane/outLIFE/`. If a night-4 snapshot is missing, use night 7 only. They are measurements, so no constant is chosen here.

### P0-use (about 1 h per seed)

Run the night-7 battery on a copy with `FREEZE_USE=True` and `last_fetch` logging. Report:
- the share of all fetches and of wrong fetches where the weighted winner is not the raw winner;
- the share with a raw margin below 0.02;
- the share of right fetches that the gate would deny;
- the frozen-probe top-1 share against the banked self-boosting probe.

| P0-use outcome | Decision |
|---|---|
| Under 20% of wrong fetches would be denied | KILL M-use as inert |
| Over 40% of right fetches would be denied | KILL M-use as over-gating; do not retune MARGIN_USE |

### P0-scale (about 30 min per seed)

Use a copy of the night-7 home and set `B *= 105/norm` (g about 0.28 on s311). Run the battery (intact and off) plus the open turns, and compare them with the banked unscaled rows of the same home. Repeat on night 4 if that snapshot exists.

- **Pass:** a whole-transcript read shows the first-person looping and identity drift reduced. Examples are "I think autumn is my favourite season." repeating, and "My name is Meridian." appearing where it is off-cue. Collapse must not be worse.
- **Recorded, not a pass condition:** the H-off cost, because the copy gets no re-replay.
- **Kill:** if the speech is unchanged, kill M-scale. Norm is then not the cause, so do not run lives.

## 6. Dev lives (7 nights, 1.5B, DEV split, full-probe nights 1, 4 and 7, `--frozen-probe` on every arm)

These start when s310, s313 and s314 free the pools, and only for the methods that passed section 5.
- **Arms:** B (baseline at current defaults, with `--nosleep-arm`), U (M-use), S (M-scale K = 1.0), US (both).
- **Seeds:** 315-317 are paired across the four arms. 318 is held for one ablation or fallback, and 319 is a spare.
- Seeds 310-314 are not used for these lives, because their baselines were run without `--frozen-probe`.
- **Load:** 12 lives at about 3.7 h each on 3 nodes is 4 rounds, about 15 h. If one method is killed in section 5, the load falls to 6 lives (B and the surviving arm on 315-317), 2 rounds, about 7.5 h.

| Round | pool40 | pool41 | pool42 |
|---|---|---|---|
| R1 | s315 B | s315 U | s315 S |
| R2 | s315 US | s316 B | s316 U |
| R3 | s316 S | s316 US | s317 B |
| R4 | s317 U | s317 S | s317 US |
| R5 (conditional) | s318 ablation or fallback (section 8) | - | - |

## 7. Pre-registered dev criteria (read after R4; pooled read = nights 4-7, paired by seed, 3 seeds × 4 nights)

A criterion counts only if the pooled mean passes **and** its direction holds on at least 2 of 3 seeds. The H-int baseline swings by 3/6 between nights within one seed, so a single night is never read.

**Whole-transcript read** (Claude reads it in full; the scripted counts alone never promote):
- Read days 4 and 7 lived turns and the night-7 battery of every arm, side by side with B.
- Record, per arm and per night, the repeated first-person lines, off-cue intrusions of wrong-fetch content (Bramble, "I drive a red Volvo"), and on-topic answers to the H questions.
- Any new defect is a fail regardless of the table. Examples are flat or empty replies, new looping, or identity loss.

**M-use passes if all of these hold:**

| Read-out (nights 4-7 unless stated) | Threshold vs B |
|---|---|
| Max non-permanent strength, night 7 | ≤ 3.0 (B s311 was 6.0) |
| Top-1 share of live trunk fetches | lower |
| Top-1 share of frozen-probe fetches | lower |
| Wrong-fetch count per night | at least 1.0 lower (mean) |
| Right-fetch count per night | not lower by more than 0.5 (abstention guard) |
| Fetched count | not lower by more than 1.0 |
| H int | not lower by more than 0.5/6 |
| H on facts told day 1-2 | not lower by more than 0.5/6 (high-use directive) |
| Transcript read | at least as good as B, fewer wrong-fetch intrusions |

**M-scale passes if all of these hold:**

| Read-out (nights 4-7 unless stated) | Threshold |
|---|---|
| Norm, every seed and night | ≤ 2.2 × night-1 norm |
| ‖A‖ growth over nights 2-7 | ≤ 1.5x |
| Col int and col off | not worse than B |
| H off (suppression guard) | not lower than B by more than 0.5/6 |
| H int | not lower than B by more than 0.5/6 |
| H off on day-1-2 facts | not lower than B by more than 1/6 |
| Transcript read | clearly fewer repeated first-person lines and less identity drift on day 7 and night 7 than B |

Fluency bought with lost recall is a trade, not a pass. The no-sleep arm is fluent and scores H 0/6.

**US** is read only if both U and S pass. It must not be worse than the better single arm on its home metrics. Its norm drop is not credited to M-scale beyond S's own drop.

**Not credited:**
- M-scale is not credited with wrong-fetch changes, and M-use is not credited with norm or fluency changes.
- Nights 1-4 hubs with 0-1 uses (Pixel, soup, late-bus) are expected to persist under M-use. That is a stated null, not a failure.

## 8. Kill rules (timebox: 2 days of compute and work per method, from the first dev life; hard stop 12-08)

| Rule | Trigger | Action |
|---|---|---|
| M-use-1 | P0-use inert or over-gating (section 5) | Kill before any life |
| M-use-2 | H int or right-fetch count fails its guard on 2 of 3 seeds | Kill. Do not retune MARGIN_USE. The 0.0 and 0.05 ablations on s318 are diagnostic only, never a rescue |
| M-use-3 | Max strength not bounded with the gate on | Bug; fix and rerun once. A second failure means kill |
| M-scale-1 | P0-scale shows no speech change | Kill |
| M-scale-2 | H-off guard fails pooled with the speech read passing | One fallback run, K = 1.5 on s318 versus a B run on s318. If it still fails, kill |
| M-scale-3 | ‖A‖ compounds (> 1.5x) and the norm escapes the bound | One variant run, sqrt(g) on both A and B, on s318. If it still fails, kill |
| Both | Any transcript-read fail (new defect) | Kill that arm |
| Both | Flag-off smoke not byte-identical | Stop everything until fixed |

A kill is a verdict on this method, not on the capability. Follow-up methods are named but not pre-registered:
- adaptation or refractoriness for the geometric hub;
- a better recall key;
- the D1 disengagement as written in section 9.

## 9. Promotion to the registered life

A method that passes section 7 on dev, with constants frozen (MARGIN_USE = 0.02, K = 1.0 or the 1.5 fallback), goes to the registered six-seed life:
- **Seeds and design:** 42/43/44/100/101/102, a 7-night life, method arm versus a baseline arm on the same seeds, and a flag-off arm on one seed that must equal its baseline byte for byte apart from timings.
- **Load:** about 13 lives, about 17 h on 3 nodes.
- **Same read and thresholds as section 7,** pooled over six seeds with direction on at least 4 of 6, plus the whole-transcript read.
- **Then land-result:** fresh-context skeptic, a lesion that changes the reply, HANDOFF and board update.

No dev number is reported as a result. After any TEST read, a method change rotates the split.