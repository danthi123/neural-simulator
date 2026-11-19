# Curiosity: six-seed test of the novelty-gated lived inquiry episode at 1.5B (2026-11-19)

Prereg `brain/research/2026-11-19-curiosity-test-prereg.md` (committed 132ac72ed before any gate arm ran on the test
bank; wording and bar fixed on the disjoint dev bank). Seeds 42/43/44/100/101/102, test bank (12 frames x C/R/N),
4 samples, CPU fp32. Arms off / lesson (lived, gate off) / gate (bar 12.4) / forced (always reinstated). Blind readers
(haiku + sonnet, arm and class hidden, all arms shuffled; agreement 0.94-0.99), counts = both agree. Scorer
`brain/experiments/inquiry_test_score.py` -> `2026-11-19-inquiry-test-1.5b/score.json`. Hosts (in-file `host`): off arms
pool nodes; main 42/43 workstation, 44/100/101/102 pool; `tools/where_ran.py` cannot read these files (no sidecar).

**Verdict: all pre-registered criteria hold (PASS), NARROWED by three sonnet skeptics (statistics/instrument, control
integrity, claim scope; workflow wf_e625a8a0-14e, every number recomputed from raw labels).**

| criterion | result | per seed |
|---|---|---|
| P1 asks what it is, DiD (N - C), gate - off >= +0.10 | +0.253 | .229 .229 .271 .333 .292 .167 |
| P2 asks or admits not knowing (N) >= +0.20 | +0.378 | .375 .333 .333 .479 .458 .292 |
| P3 invents a meaning (N) <= -0.05 | -0.118 | -.167 -.125 -.062 -.062 -.146 -.146 |
| H1 common words: gate fired, change in asks | 0 / 288, 0.000 | |
| H2 coherence gate - off (C / R / N) | -0.003 / +0.017 / +0.115 | |
| H3 verbatim lesson reply, "zorbit" in replies | 0 / 384 fired, 0 / 864 | |
| L lesson-only = off; gate changes N replies | 864 / 864; 270 / 288 | |

Absolute rates on coined words, off -> gate: asks 0.080 -> 0.333; asks or admits 0.135 -> 0.514; invents 0.267 -> 0.149.
Rare-real words: asks 0.094 -> 0.170, invents 0.424 -> 0.372. Forced: N the same as gate (asks 0.326), C changed on
254/288 rows (asks 0.007): the novelty gate's work is to keep the episode away from ordinary turns.

**What it is (caveats the skeptics required):**
- Scaffold level and narrow: the bar (12.4 nats), the lesson and its wording are host-set (dev-tuned); nothing is
  learned; single-turn statements from one 12-frame bank; no sleep, no follow-up turn where the answer is used.
- Six seeds are sampling replicates over 12 fixed coined items. The gain is uneven: about 5 of 12 items show none,
  mostly names and places (Drelvik, Quaxlin, Zentrova, vellophone, Ploskovar), although the gate fired on them. Without
  the 3 strongest frames P1 is about +0.12: robust in sign, not in margin. Against the earlier baseline's off rate
  (35/288, different sampling seeds) P1 would be about +0.21.
- A third of coined-word replies ask; some "asks" are primed "What's a X?" forms, some replies are tautologies or still
  invent. Coherence and inventing gains are partly mechanical (a question replaces a statement): not quality claims.
- Costs: the gate fires on 4 of 12 rare real words (cooper, kedgeree, netsuke, geocaching) and on 3 of 24 everyday
  tellings (Oluwaseun, pho, Ishiguro; "What's pho?" 3 times in 96 turns); false ignorance on some known things; reply
  diversity on coined words falls (81 vs 152 distinct replies).
- Answer-rate lens: `tools/answer_rate.py` does not apply to this layout (UNMEASURED); abstention-like replies rise on N
  by design (15 -> 54 / 288).

**Decision:** milestone 11-30 (c) "it asks about what is new to it": MET at scaffold level, narrow (above). The
default flip of `inquiry="auto"` is held for prereg Amendment 1 (weekly with sleep: keyed answers, no "zorbit", blind
transcript read) and a ruling on the false-ignorance cost; the production path (`_live_inquiry_at_birth`) is covered
by a unit test and the weekly guard, not by these six seeds.
