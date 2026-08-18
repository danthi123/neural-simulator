---
type: finding
status: smoke-only
date: 2026-08-17
mechanism: wave1-banking
---
## longitudinal-6seed — SMOKE-ONLY (the compressed-week was never run)

**Result.** The 6 seed logs (raw/emerge/longitudinal_develop_s{42,43,44,100,101,102}.log) all ran the **4-day GPU SMOKE** (`--n-days 4`; confirmed by each log header + every .prov.json argv), NOT the 7-day "compressed-week" the smoke declared itself READY for. All 6 hit the runner's smoke gate (6/6 GO), but each overwrote the single output JSON, so only seed 102 survives as `_longitudinal_develop_loop_gpu_smoke.json`; the other five exist only as .log.

**What is genuinely load-bearing (varies by seed, passes 6/6):**
- Stream-cortex Hebbian code-learning: corr(M,C) mean **0.89/0.92/0.89/0.90/0.90/0.89** (gate >=0.3).
- FROZEN-BRAIN anti-cheat (plasticity OFF): frozen arm commits **0 facts**, learn_fidelity **0.0** while plastic arm learns — real control that learning is plasticity-dependent.
- Moat **0** false-accepts; retention_acc **1.00**; persistence reload+resume OK.

**Honest residual (do NOT overclaim):**
- "Development" trends (vocab 6->24, facts 2->11) are **identical across all 6 seeds** = curriculum-prescribed presentation counts, not emergent learning. `day0_vs_dayN_differs` only checks counts rose.
- CONSOLIDATE / GROWTH (arch rebuild+weight-transfer) / RESUME (re-hears vocab, not persisting cp_connections) are **owner-approved STAND-INS** (runner's own `components_standin`).
- This is a "the loop runs end-to-end at GPU scale + stream cortex learns" smoke; it is NOT evidence of multi-day emergent development.

**Verdict: SMOKE-ONLY.** Machinery de-risked 6/6 with a real learning signal + a real frozen control; the compressed-week 6-seed longitudinal-development run does not exist as an artifact on this branch (no longitudinal JSON with n_days>=7 anywhere).

Banked artifacts (this branch): main already tracks the seed-42 GO survivor at `research/findings/raw/_longitudinal_develop_loop_gpu_smoke.json`, so the seed-102 survivor is banked alongside it under the seed-suffixed name `research/findings/raw/_longitudinal_develop_loop_gpu_smoke_s102.json` (+.prov.json) to avoid overwriting the seed-42 record. All six per-seed `.log` files (+.prov.json) under `research/findings/raw/emerge/longitudinal_develop_s{42,43,44,100,101,102}.log` are the full per-seed evidence.

Note: the seed-102 raw JSON is a historical pre-gate artifact (recall_acc ceiling, no `preconditions` block) retained at its origin path (cited above); the committed per-seed `.log` files and this finding carry the verdict evidence.
