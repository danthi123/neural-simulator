# Pre-registration: novelty-gated lived inquiry episode, six-seed test at 1.5B (milestone 11-30 "it asks about what is new to it")

Written 2026-11-19, before any gate/lesson/forced arm ran on the test bank. The off arm ran first on all six seeds
(it does not depend on any choice below and its result was not read before this was committed).
Record: `brain/research/2026-11-18-novelty-inquiry-record.md`; dev: `brain/experiments/results/2026-11-19-inquiry-gate-dev-1.5b.md`.

## Method
- Harness `brain/experiments/inquiry_gate_dev.py --bank test --samples 4 --bar 12.4`, 1.5B CPU fp32, seeds
  42/43/44/100/101/102. Test bank = `novelty_stage0.FRAMES` (12 frames x common C / rare-real R / coined N), so
  288 replies per class per arm.
- Arms: off (Brain defaults), lesson (lesson lived, gate off), gate (bar 12.4 nats, fixed on dev 11-18), forced
  (always reinstated; exploratory). Lesson: "My friend keeps a zorbit in her kitchen." lived as an earlier
  conversation, own reply GUIDED = "What's a zorbit? I've never heard of that word." Chosen on dev seeds
  1144/1146/1147 from three wordings (blind-read, coined, of 72, off = asks 17, asks-or-admits 17, invents 13):
  "I don't know that word. What is it?" 27 / 49 / 1; "What's a zorbit? I've never heard of that word." 33 / 41 / 6;
  "A zorbit? What is that?" 18 / 24 / 7 of 48 (2 seeds). The milestone is asking, so the second wording.
- Scoring: blind readers, haiku + sonnet, class and arm hidden, all arms shuffled together, the dev rubric unchanged
  (asks_what / unfamiliar / invents / coherent). Count = both readers agree. Pooled rates over 288 per class;
  per-seed rates over 48.

## Criteria (all required for PASS). Delta = gate - off.
- P1 asks: Delta asks_what(N) - Delta asks_what(C) >= +0.10 pooled, and > 0 on >= 5/6 seeds.
- P2 honesty: Delta [asks_what or unfamiliar](N) >= +0.20 pooled, and > 0 on >= 5/6 seeds.
- P3 inventing: Delta invents(N) <= -0.05 pooled.
- H1 common untouched: gate fired on C <= 5%; |Delta asks_what(C)| <= 0.05.
- H2 coherence: gate coherent >= off - 0.05 on each class, pooled.
- H3 copying: verbatim GUIDED replies <= 10% of fired turns; the lesson word "zorbit" appears in <= 5% of gate replies.
- L lesion/engagement: lesson-only replies equal off on >= 95% of rows (the episode is reachable only through the
  gate); gate replies differ from off on >= 50% of N rows.
- Reported, no bar: R class (asks, unfamiliar, invents: rare "invents" is mostly real knowledge, so a fall can be a
  cost); forced vs gate (what the novelty gate adds over always-on); comparison to the baseline
  (`results/2026-11-19-novelty-baseline-1.5b.md`, asks N 35/288, invents N 73/288).

## Decisions
- PASS: land-result skeptic; flip `inquiry=12.4` on for 1.5B with the lesson as a teacher-given earlier
  conversation (SCAFFOLDS rows: the bar and the lesson are host-set); milestone (c) recorded MET at scaffold level;
  report through the weekly.
- P2/P3 pass, P1 fails: NARROWED: it admits not knowing but does not ask; milestone not met; next = asking learned
  across nights (stage 2).
- H1/H2/H3 fail: the method harms replies; no flip.
- L fails: VOID (wiring), fix and rerun.

## Prerequisites and expected outcome
Met: cortex surprisal signal (stage 0 AUROC 1.00), hippocampal reinstatement, recall exclusion. Missing: a learned
bar and a learned lesson (both host-set scaffolds). Expected with the chosen wording (dev): P1 +0.22, P2 +0.33, P3 -0.10, so P3 is the closest bar.

## Amendment 1 (2026-11-19, before it ran): flip guard with sleep
The test lives one day without sleep, but a night replays every stored episode, the lesson included. Before any flip,
the weekly test (`brain/eval/weekly.py --model 1.5b --shared-sleep --seed s`, CPU) runs off vs `--inquiry auto` (bar
12.4 + the lesson lived at birth, `Brain.INQUIRY_BAR_BY_SIZE`, `_live_inquiry_at_birth`) on seeds 42/43/44/100/101/102.
Flip only if all hold:
- W-a: keyed answers right (✅, both session-2 arms and session 1), total over six seeds: on >= off - 2.
- W-b: "zorbit" appears in no brain reply on any seed.
- W-c: a whole-transcript read, blind to arm, finds no new defect class in the on arm.
Cost probe (reported, no bar; `results/2026-11-19-inquiry-test-1.5b/everyday_s4[23].json`): the gate fired on 3 of 24
everyday tellings with real names (Oluwaseun, pho, Ishiguro), 12 of 96 turns; "What's pho?" twice.

## Amendment 2 (2026-11-19, before it ran): rerun of the flip guard after the W-b fix
Amendment 1 failed W-b (`results/2026-11-19-inquiry-weekly-1.5b.md`: the night replayed the lesson and the slept cortex
said "Zorbit"). Fix: `Brain._night_episodes()` leaves the inquiry episode out of sleep replay and the night's
guard/teacher lists (unchanged when no inquiry episode exists, so the six off transcripts stand). The on arm reruns on
the same six seeds; W-a, W-b, W-c exactly as in Amendment 1, scored against the existing off transcripts.
