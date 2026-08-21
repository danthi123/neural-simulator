---
type: finding
status: live
date: 2026-08-21
mechanism: fast-bulk-bind-teacher-load
lane: integration
seeds: [11]
seed-waiver: An ENGINEERING build-throughput + answer-identity de-risk (a deterministic recall/latency comparison
  of two ways to compute the SAME composite), not a stochastic effect size — the load-bearing evidence is the
  byte-level answer-match + the measured per-fact speedup, single deterministic build per condition.
instrument: research/runners/_knowledge_fast_bulk_bind_verdict.py — builds the sharded LTM fast (closed-form) vs
  resonate (neural) at matched facts, compares query_patient + ask_yes_no answers + the moat, measures the
  per-fact speedup, and demonstrates a fast-only build at a size the resonate path cannot reach; tools.verdict.Verdict.
runner: research/runners/_knowledge_fast_bulk_bind_verdict.py
external: NO-EXTERNAL-NEEDED — the RF bind of unit phasors IS exact phase addition and the bundle IS the sum's
  phase; the resonate merely CONVERGES to that closed form. The optimisation composes the existing validated store.
artifacts:
  - research/findings/raw/_knowledge_fast_bulk_bind_verdict.json
---
# The closed-form bulk bind removes the fact-store's build wall — LLM-scale knowledge is now reachable (~356-670x, recall-identical, moat preserved)

Artifact: research/findings/raw/_knowledge_fast_bulk_bind_verdict.json (GO).

**One line.** The knowledge store's real barrier to LLM-scale was not the query (sharding solved that) but the
BUILD: the neural `store` runs 3-4 RF resonates per fact (~50-63 ms/fact), so a million-fact teacher-load is ~17 h
and 20M is ~150 h. The RF bind of unit phasors is **exact phase addition** and the bundle is **the sum's phase**, so
the bound composite has a CLOSED FORM the resonate only CONVERGES to. Computing it directly is **~356-670x faster,
recall-IDENTICAL to the neural bind (byte-level same answers), moat preserved** — turning ~150 h for 20M facts into
~25 min. Millions-of-facts knowledge is now genuinely reachable.

## The mechanism (`tiered_fact_store.encode_fast` / `build_ltm_from_facts(fast=True)`, NO `sim/` edit)
`encode_fast(comp, fact)` = `angle(sum_r exp(2*pi*i*(role_r + filler_r)))/(2*pi) mod 1`, using `comp._filler_phases`
for every role — so polarity (AFFIRM/NEGATE), attributes, and nested clauses bind identically to `_encode`. It skips
the per-role bridge resonate (the ~208-step RF dynamics) and computes the fixed point the resonate settles to.
`build_ltm_from_facts(fast=True)` uses it for str-patient facts (routing by agent, == `store`'s routing) and falls
back to the neural `store` for clause/attributed patients. `fast=False` (default) keeps the byte-identical neural path.

## The verdict (numpy CPU, D=128) — GO
<!--derived-->
- **Answer-identity:** at matched facts, the fast store's `query_patient` answers == the resonate store's on
  **150/150** probes, and `ask_yes_no` == on **150/150** — the fast bulk-load holds the same representation.
- **Moat preserved:** unknown cue -> abstain, **20/20**.
- **Speedup:** the per-fact store cost drops from tens of ms to tens of us — **356x measured here** (671x on a
  lighter-loaded box); the exact ratio tracks machine load, but the closed form is O(D) arithmetic vs a 208-step
  dynamics loop.
- **Scale:** a fast-only build of **20,000 facts in ~6.7 s, recall 100/100** — a size the resonate path would take
  ~20 min to reach. Derived projections (labelled): **20M facts ~25 min fast vs ~150 h resonate.**

## Honest scope
This is a declared **bulk TEACHER-LOAD optimisation** — the teacher precomputes the composite the neural bind would
produce (recall-identical, measured), so the brain holds the identical representation; the **QUERY / recall (the
cognition) stays FULLY neural** (resonate unbind + cleanup), unchanged. It is the same class of move as
`developed_brain_io`'s persisted-composite reload, applied to the initial write. Clause/attributed patients still go
through the neural `store` (the fast path is str-patient only). Combined with the sharded routing (sub-second query
at any K) and the persisted-store reload (222x), the three together make a millions-of-facts knowledge base
buildable once + reloadable in seconds — the enabler for teaching the brain the knowledge an LLM has.
