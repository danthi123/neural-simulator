---
type: finding
status: live
date: 2026-08-26
mechanism: bulk-structured-kb-fhrr-loading
lane: knowledge
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: none -- a genuine 6-seed sweep ran (this is not an engineering single-build de-risk).
instrument: research/runners/_bulk_kb_load_derisk.py -- vectorized closed-form FHRR bulk encode of N=100k
  agent-action-patient triples at production D=512, cross-checked against the faithful per-op spiking
  RFPhasorComposer.store()/query_patient() on a subsample, plus two anti-cheats (shuffled-triple collapse,
  out-of-store abstain); tools.verdict.Verdict.
runner: research/runners/_bulk_kb_load_derisk.py
external: NO-EXTERNAL-NEEDED -- same closed-form FHRR algebra as the already-landed `tiered_fact_store.encode_fast`
  (bind of unit phasors = phase addition, bundle = the phasor sum's phase); this de-risk independently re-derives
  and re-validates that mechanism, it does not propose a new one.
artifacts:
  - research/findings/raw/_bulk_kb_load_6seed.json
  - research/findings/raw/_bulk_kb_load_smoke.json
---

# Bulk structured-KB ingestion is 6-seed GO, and the capability it de-risks is already shipped under a different lineage

Artifact: research/findings/raw/_bulk_kb_load_6seed.json (verdict field: **GO**, 6/6 seeds).

## The result, honestly, first

The runner's own `tools.verdict.Verdict` block says GO. Per-seed (42/43/44/100/101/102), N=100,000 synthetic
agent-action-patient triples, production FHRR dimension D=512:

| check (worst seed across the 6) | measured | gate |
|---|---|---|
| patient top-1 recall | 1.0000 | >= 0.95 |
| agent top-1 recall | 1.0000 | >= 0.95 |
| moat: new confabulation on out-of-store cues | 0 | == 0 |
| cross-check vs faithful spiking `query_patient` | 1.0000 | == 1.0 |
| bulk load throughput | 11,136.7 f/s | >= 1000 f/s |
| anti-cheat: shuffled-triple recall vs true patient | 0.0010 (chance 0.0005) | < 0.05 |

All six seeds individually GO (`n_go: 6`, `n_seeds: 6`). The runner cross-checks its vectorized closed-form bulk
encode against the composer's actual per-op spiking resonate (`comp.store()` + `comp.query_patient()`) on 40
facts/seed (240 total probes across the sweep), and runs two anti-cheats: shuffling the patient column (recall on
the true mapping collapses to nowhere above chance) and probing out-of-store (agent,action) cues for confabulation
(0/2000 per seed, every seed). This is a real, verified GO -- not a metric lifted from a run whose own verdict was
negative (`docs/TERMS.md` "GO").

## Redundancy assessment -- this is the substantive question for this finding

**The mechanism this runner de-risks is the SAME closed-form FHRR bulk-encode algebra already landed on `main` and
already production-integrated**, via a different, earlier lineage:

1. **`tiered_fact_store.encode_fast`** (2026-08-21,
   `research/findings/2026-08-21-closed-form-bulk-bind-removes-the-fact-store-build-wall-LLM-scale-knowledge.md`,
   GO): the identical algebra --
   `composite = angle(sum_r exp(2*pi*i*(role_phase_r + filler_phase_r))) / (2*pi) mod 1` -- verified
   recall-identical to the spiking `store()` (150/150 `query_patient` + 150/150 `ask_yes_no` matches), moat
   preserved (20/20), 356-670x measured speedup.
2. **`ShardedPhasorStore`** (2026-08-20,
   `research/findings/2026-08-20-sharded-fact-store-removes-the-O-K-query-wall-knowledge-scales-to-LLM-scale.md`,
   GO): the RETRIEVAL half (sublinear routed query at any K), byte-identical answers + moat preserved vs unsharded.
3. **A hardened, rate-limited, paginating Wikidata fetcher** (`12205ffa8`, `1212d5dfe`) for real-world triple
   acquisition, and **a persisted, default-on 15k-fact curated core** (`wikidata_core_15k`, board #133) that
   `webapp/server.py` loads as the production cortical LTM tier (`_default_ltm_bundle_dir`,
   `_LTM_SHIP_DEFAULT_ON = True`, flipped default-on 2026-08-26) via exactly `ShardedPhasorStore.load` /
   `TieredFactStore` -- i.e. `encode_fast`'s closed-form bind is the code path that BUILT it.

Checked directly in this worktree: `webapp/server.py` imports both `ShardedPhasorStore` and `TieredFactStore` and
wires them as the live conversational agent's long-term-memory tier, on by default. Per `docs/TERMS.md`, that
capability is `wired` AND `on-by-default` -- already at a higher integration level than a runner-only 6-seed GO.

**What this runner adds beyond the landed lineage, honestly:** (a) validation at production D=512 and N=100k in
one sweep (the landed build-wall finding measured answer-identity on 150 probes and a 20k-fact fast-only build,
both at D=128); (b) 6-seed reproducibility rather than 1; (c) a **shuffled-triple anti-cheat that the landed
findings do not run** -- it demonstrates the closed-form encode recovers the *correct* stored (agent,action)->patient
mapping, not merely *a* mapping. None of these change a wall/gap status, unlock a new capability, or move a
production default -- they are additional confidence on a mechanism that is already shipped and already
`on-by-default`. Landing this runner's `bulk_encode`/`decode_role` as a second, independent implementation of the
identical algebra `tiered_fact_store.encode_fast` already provides would create exactly the kind of
two-implementations-one-mechanism drift `docs/TERMS.md`'s "one term, one meaning" discipline exists to prevent.

**Verdict on redundancy: substantially subsumed, not a new capability.** The GO is genuine and is banked here (the
biology binding + the JSON artifact), but the runner does not warrant becoming a second production code path. The
biology doc (`research/biology/semantic-store-cortical-capacity.md`) is updated to point `current_finding` at the
completed 6-seed artifact and to record this redundancy note in `current_status`, so a future reader lands on the
landed/wired lineage rather than re-deriving `encode_fast` from scratch.

## Recommendation

- Vikunja board #150's "bulk-kb" citation should be closed/annotated as **subsumed by** board #133 (curated 15k
  core default-on) + the 2026-08-20/2026-08-21 sharding/closed-form-bind findings, not treated as an open item.
- `research/runners/_bulk_kb_load_derisk.py` is kept in the tree as a standalone verification instrument (its
  shuffle anti-cheat is genuinely useful and absent from the landed lineage's own tests) but is NOT wired into any
  production path -- `tiered_fact_store.py` / `webapp/server.py` remain the single production implementation of
  this mechanism.
