# S_all + slept bar, six-seed gate on the registered seeds (1.5B open conversation): selection REPLICATES; S3 intrusions fail (collapse-driven); no flip

Prereg: `brain/research/2026-11-18-sall-open-conv-1.5b.md` Amendment 2. Board #282.
- Raw: `results/2026-11-18-sall-slept-gate-1.5b/` (transcripts, rows, `scores.txt`, `collapse.txt`).
- Pool CPU fp32, seeds 42/43/44/100/101/102: pool40 42/43, pool41 44/100, pool42 101/102. Both arms of a seed ran on
  one node (rows `host`).
- Arms: d = S0; s = `--key S_all --key-bar slept`.

**Intact arm, day 2, six seeds summed (rows.json):**

| | d | s | bar | |
|---|---|---|---|---|
| S1 reworded right memory fetched | 1/18 | **14/18** | s >= d+4; s >= d on >= 5/6 | **PASS** (6/6) |
| direct right memory fetched | 14/18 | 17/18 | | |
| S2 direct + reworded ok | 16 | 20 | s >= d-1 | holds |
| S2 abstentions | 0 | 0 | | holds |
| S3 false + invented (never-told) | 32 | 33 | s <= d+2 | holds |
| S3 needless recall | 15 | 12 | s <= d+3 | holds |
| S3 intrusions on open turns | 17 | 21 | s <= d+2 | **FAIL** (+4) |
| hippocampus-off ok (lesion) | 18 | 20 | reported | |

**Intrusions follow collapse, not the key** (`collapse.txt`).
- Collapsed means the most repeated day-2 line is at least 40% of replies.
- Collapsed seeds:
  - gate: d 3/6 (42, 44, 102) vs s 2/6 (101, 102);
  - 1132-1137 rerun: 1 vs 1;
  - 1126-1131 size-bar run: 2 vs 2.
- Every collapsed day 2 repeats a told fact on open turns. Seed 101 s alone has 6 intrusions ("I play the bassoon in a
  local band." to nearly every turn), and d had 0 there.
- The two arms differ from day 1, because the key acts on day-1 fetches. So which seed collapses is effectively drawn
  per arm.
- The bar is still recorded as failed: it was preregistered and is not waived.

**Verdict.**
- Meaning-based selection replicates on the six registered seeds. Reworded questions fetch the right telling 14/18
  with S_all + slept bar against 1/18 with S0, s >= d on 6/6; the pooled 12 seeds give 29/36 vs 5/36.
- No flip. S4 failed on 1132-1137, and S3 intrusions fail here.
- The milestone item "recall from reworded questions (meaning-based selection)" is met as a selection read-out on 12
  seeds. It is not met as better conversation: replies move +4/36 and the lesion arm answers as well.
- Skeptic coverage: the three skeptics on the 1132-1137 claim (PARTLY REFUTED, selection-only) apply unchanged. This
  run adds the registered seeds and the intrusion finding.

**Weekly (report).** First attempt crashed on every run (`weekly.py` read `Brain.__init__.__kwdefaults__`, None):
fixed with `inspect.signature` and a test. Rerunning on pool; the result is appended here when it lands.

**Next.** The reply-level defects bound everything at 1.5B:
- collapse into a repeated told line, on 2-3 of 6 seeds per arm;
- an invented answer for every never-told question (18/18).
Both were KILLed for in-context levers at 1.5B. Selection is no longer the bottleneck.
