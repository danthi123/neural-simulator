# Pre-registration: is false recall after sleep set by adapter rank or by what the night trains? (1.5B, six seeds, AWS)

Written 2026-11-15 before any run at rank != 8. Capability: Sleep consolidation + Metacognition (false recall).
A diagnosis, not a method: no new mechanism, one variable (the adapter rank of the night).

**Record brief.**
- After one night at rank 8, never-told "my X" questions get a told fact: arm s false 6.83/12 (banked `nmoff_s*`,
  batch 20261114-2031). M4 (frame guard) halves it but costs told and paraphrase; M4b (told-topic cues dropped) is
  NOT A LEAD (`results/2026-11-15-near-miss-told-filter-1.5b.md`; s100-s102 being rerun with matched halves).
- Two preregs name capacity as the cause of the guard's told cost and neither tested it: M3's falsifier ("then the gap
  is capacity (rank 8 shared by 12 facts), and the next test is adapter rank") and M4/M4b ("the guard and the told
  bindings compete in one rank-8 adapter").
- External research at this wall (lever 9; `tools/deep_research.sh` + two literature sweeps, 2026-11-15):
  - Rank-1 LoRA holds about 500 facts (arXiv 2502.14502, full text), so rank 8 is not short for 6 slept facts.
  - LoRA rank trades learning against forgetting (Biderman et al. 2024, full text).
  - New facts learned by fine-tuning raise hallucination, and the wrong answers on unfamiliar queries copy the
    answers the training examples carried (Gekhman et al. 2024; Kang et al. 2025, full text). That predicts a
    content cause at any rank.
  - In people, gist intrusion after sleep is conditional, not universal (Newbury & Monaghan 2018 meta-analysis,
    abstract-only).
- Lever count on this defect: 9 methods; this is a diagnosis that decides the 10th.

**Prerequisites.** All met: the 1.5B cortex, the night, the probe. No host decision. `--rank` is a probe flag
(`mixed_day_probe.py`, default 8 = every registered run); a CPU smoke at 0.1B showed it saves rank-2 adapters.

**Method.** `mixed_day_probe.py --arms c,s,r --self-q`, 1.5B, `--steps 480 --lr 5e-4 --kl-weight 3`, the same slept
halves as the banked off arm (s42 even, s43 odd, s44 even, s100 odd, s101 even, s102 odd), one AWS batch
(`jobs/2026-11-15-aws-rank-diagnosis.txt`).
- `r2off_s*`: rank 2, no guard.
- `r32off_s*`: rank 32, no guard.
- `r32on_s*`: rank 32, the M4 guard (`--near-miss 4 --near-miss-frame F`).
- Reference: banked rank-8 `nmoff_s*` and `nmon_s*` (`results/2026-11-14-near-miss-guard-1.5b/`).
- All files re-scored before reading.

**Criteria (arm s, six-seed means, of 12; told/para/false/invented/abstention as `near_miss_table.py`).**
- R0: each run records its rank in the JSON, and the rank-2 and rank-32 adapter norms differ from rank 8.
- R1, low rank cures the frame (capacity verdict): r2off false <= nmoff false - 2 and lower on >= 5/6 seeds, with
  told >= nmoff told - 0.5 and para >= nmoff para - 1.
- R2, high rank frees the guard (capacity verdict for the guard's cost): r32on told >= 0.95 x r32off told and para
  >= r32off para - 0.5, with r32on false <= 0.6 x r32off false.
- R3, content verdict: neither R1 nor R2, and |r2off false - nmoff false| < 1 and |r32off false - nmoff false| < 1.

**Decision.**
- R1 holds: capacity sets the frame. Next: re-test M4 at rank 2 (cheaper adapters as well).
- R2 holds: the guard is right but crowded at rank 8. Next: M4 at rank 32 through the LEAD/GO criteria of the M4
  prereg, on fresh seeds' check.
- R3 holds: the cause is what the night trains. Next: the hippocampus-as-teacher night (episodes in context as the
  teacher of the never-told answers; Kang et al. 2025, Gekhman et al. 2024), with a no-training gate first.
- Otherwise: MIXED, reported per seed; no next method is chosen from it.

**Expected.** R3 (content): rank 8 is far above what 6 facts need, and the weaker night already left false recall high.
