---
type: finding
lane: audit
status: live
date: 2026-07-31
claim_check: synthesis
---

# Tier-2 audit — WHY 192 findings cite no artifact: a 25-doc sample says ~40% already HAVE the artifact and only 1 in 25 is truly unrecoverable

**This is a SAMPLE, not a census.** 25 of the 192 flagged findings (13%) were read and classified by hand.
Every proportion below carries a wide interval at n=25 and is stated with one. The per-document
classifications and their evidence are in
[`research/findings/raw/audit_tier2/no_artifact_triage_2026-07-31.json`](research/findings/raw/audit_tier2/no_artifact_triage_2026-07-31.json).

## The denominator is NOT 1843 — correcting the framing of the task

`tools/gates/doc_type.py` only inspects documents that **declare frontmatter**; legacy files without it are
skipped by design, so the gate never scans the whole corpus. Measured now:

| population | count |
|---|---|
| `research/findings/*.md` total | 1843 |
| …carrying frontmatter (from the Tier-1 pass, commit `78b8cc74`) | 281 |
| …of those, flagged "cites no artifact" | **192** |

So 192 is **68% of the 281 classified findings**, not "a large share of 1843". The remaining 1562 findings are
unclassified and their artifact status is **unmeasured** — this audit says nothing about them. The gate's own
comment already anticipated this ("192 doc-type hits … the moment the Tier-1 classification gave them
frontmatter"); the number tracks how far classification has reached, not how bad the corpus is.

## Sampling method (deterministic, reproducible)

The gate emits its hits in sorted path order, which for this corpus is chronological. I drew a **systematic
1-in-7.68 sample**: sort the 192, take indices `floor(i*192/25 + 192/50)` for `i` in `0..24`. This guarantees
date spread without hand-picking, and re-running the snippet reproduces the identical 25. Month spread of the
full 192: 2026-04 → 10, 2026-05 → 50, 2026-06 → 58, 2026-07 → 74.

```bash
.venv/bin/python -c "import sys;sys.path.insert(0,'.');import tools.gates.doc_type as g;[print(p) for p in g.check(None)]"
```

## The distribution

| class | n/25 | share | 95% Wilson CI | what it means |
|---|---|---|---|---|
| **(d) artifact EXISTS** | 10 | 40% | 23–59% | cheap fix — add or reformat a citation |
| **(c) artifact never saved** | 11 | 44% | 27–63% | 10 re-runnable, **1** truly unrecoverable |
| **(a) misfiled plan/design** | 2 | 8% | 2–25% | move to `docs/plans/` |
| **(b) synthesis/audit** | 2 | 8% | 2–25% | mark `claim_check: synthesis` |

The headline: **the flag is mostly not "a measurement was lost."** In this sample the largest recoverable block
is documents whose artifact is sitting on disk right now, and the genuinely unrecoverable case is a single
document out of 25.

## (d) — the artifact exists: 10/25, and it splits two ways

**(d1) cited, but in a form the gate cannot parse — 3/25.** These documents cite the artifact *precisely*; the
gate's `ARTIFACT_RE` simply cannot read the citation. Two distinct causes, both verified against disk:

- **Brace expansion.** `2026-07-20-wkv-cortex-…` cites
  `research/findings/raw/_local_{kp,fa,bptt,kp_randres}_grounded.json`; **all four expand to existing files.**
  The regex character class `[\w.\-*?\[\]]+` admits `*`, `?` and `[]` but not `{`, `}`, `,`.
- **Extension.** `2026-06-11-V640-single-pool-memory-wall` cites `research/findings/raw/_lge_v640_seed42.log`,
  which exists (6831 bytes). The gate accepts only `.json`/`.jsonl`, so a `.log` artifact reads as none.

Corpus-wide (all 192, not just the sample) I brace-expanded and glob-resolved every placeholder-bearing
`.json` citation: **12 resolve to a real file on disk and 3 do not.** So this exact false-positive class is
real but a **minority — 12/192 (6.3%)**, and I did not extend that scan to `.log`/`.csv` citations, so 12 is a
floor, not the full d1 count.

**(d2) genuinely uncited, artifact on disk — 7/25.** These I verified by matching the artifact's numbers to the
document's claims, not by filename similarity:

- `2026-07-15-onsubstrate-bind-onbridge-bdsp-readout-RUNG3` — the runner's own `--out` default,
  `_onsubstrate_bind_onbridge_bdsp_readout.json`, holds `onbridge_train 0.4`, `onbridge_held 0.3571`,
  `lesion_held 0.4286`, `mlp_held 0.3571`: the document's four headline numbers exactly. Cited nowhere.
- `2026-06-16-onsubstrate-learned-binder-…-NEGATIVE` — `_phaseB_learned_bind_bundled_facts.json` gives
  `0.8055 / 0.2854 / 0.2055 / 0.0625` against the document's `0.806 / 0.285 / 0.206 / 0.062`. Cited nowhere.
- `2026-07-25-consolidation-boundary-REATTRIBUTED` — `research/findings/raw/consol_opsweep_gpu/` contains
  **560 files** including `ca1_sparsify_sweep_seed42.json`. The document cites **zero** of them.
- `2026-07-24-P0.3-affect-state-region-6seed-GO` — `_affect_state_region_6seed.json` carries the verdict string
  `QUALIFIED-GO / BOUNDARY (6-seed)` matching the document's corrected verdict, and even has a `.prov.json`
  provenance sidecar. The document names **no file at all**, so nothing pointed at it.

**Implication for (d): cheap, and worth doing first.** d2 is a one-line edit per document with the artifact
already in hand. d1 needs no document edit at all — it needs the gate widened (admit `{}`/`,` in the class,
and accept `.log`/`.csv`/`.npz` as artifacts, or expand-and-stat rather than pattern-match).

## (c) — the measurement's artifact was never saved: 11/25, but only 1 is unrecoverable

This is the largest class, and the important distinction inside it is whether the **probe survived**.

**(c) re-runnable — 10/25.** The document reports its own numbers, no artifact was written, but the runner is
still in the tree, so the artifact is *regenerable*. The cleanest instance is
`2026-07-22-gap4-FAITHFUL-on-bridge-BDSP`: its runner declares
an `--out` default of `bdsp_faithful.json` under `research/findings/raw/gap4/`, and **that file is not
on disk** — the run
happened, the write did not (or was cleaned up). Same shape for `2026-06-04-capacity-curve-…`
(`_capacity_curve_probe.json` absent, though its companion `gpu_resonator_capacity.json` does exist and is
uncited, making that document a d2/c hybrid I scored as d2).

A related sub-pattern: several documents cite a **script** where an artifact belongs.
`2026-07-18-gap5-CA3-completion-CLOSED` cites `_gap5_ca3_bistable_6seed.py`, and only `.py` sweep files exist
in `raw/` — the 6/6 result itself was never serialized. `2026-06-05-phase1-tpam-cleanup-derisk-GO` does the
same. A cited script is reproducibility, but it is not the measurement.

**(c) unrecoverable — 1/25.** `2026-05-09-Phase-1.5-n_motor_2000-interference-REFUTED` reports a single-seed
score table, names only a config (`phase_1_5_interference_only_n_motor_2000`), and **no runner from it survives
in the tree**. Its numbers cannot be regenerated or checked. This is the honest floor of the problem, and at
n=25 the 95% interval on it is 0.7–20% — i.e. somewhere between ~1 and ~38 documents corpus-wide.

**Implication for (c): mark honestly, do not fabricate.** The re-runnable ten should not be back-filled by
re-running and attaching a *new* artifact as if it were the original — that would silently replace a claimed
measurement with a different one. Mark them as measurement-without-artifact and re-run only when the claim is
load-bearing. The unrecoverable one should be marked unverifiable.

## (a) misfiled plan/design — 2/25, and both say so themselves

- `2026-05-19-FOURTH-convergent-structural-finding-…` states it was "reached at design time before any build"
  and explicitly declines to spend GPU because the outcome is structurally certain. There is no measurement.
- `2026-07-18-gap5-specificity-research-gate-…` is a research gate: a Kim-Kim 2025 literature read plus a
  proposed mechanism. Forward-looking by construction.

**Implication: move to `docs/plans/`.** Note this is the *mirror* of the defect the same gate catches from the
other side (92 plans currently assert a result). The plan/finding boundary is blurred in both directions, and
the two counts should be read together.

## (b) synthesis/audit over other findings — 2/25, also self-declaring

`2026-05-16-G20-failure-mechanism-FINAL-SYNTHESIS` ("Consolidates ~10 findings docs into one navigable
conclusion") and `2026-06-21-shortcut-inventory-definitive` ("READ-ONLY audit — this doc is the only write").
Their numbers are correctly sourced from other findings; they have no artifact because they ran nothing.

**Implication: `claim_check: synthesis`,** exactly as this document declares. Both were trivially identifiable
from a self-description in the first 20 lines, which suggests the frontmatter could be part-automated.

## What I could NOT establish

- **Nothing here generalizes to the 1562 unclassified findings.** They have no frontmatter, the gate never
  looked at them, and I did not sample them.
- **The intervals are wide.** At n=25 the (c) and (d) intervals overlap substantially (27–63% vs 23–59%), so
  **this sample cannot rank (c) above (d)**; it establishes only that both are large and that (a)+(b) together
  are small (4/25). Separating (c) from (d) needs a larger sample or the mechanical scan below.
- **d2 is a lower bound and c is an upper bound.** I only credited an artifact when its numbers matched the
  document. Where a plausible file existed but I could not verify the numbers, I scored (c). Two token-matched
  candidates turned out to be **different probes** on inspection — `_keystone2_spiking_slot_binder.json` holds
  step-2 keys, not the step-1 slot sizes its document reports, and `_gap1_synaptic_fluent_gen.json` is a wkv
  probe, not the RF-transduction one. Filename similarity is not evidence; a fully mechanical d2 sweep would
  need number-matching, not globbing.
- **I did not measure how many of the 192 are (d1) via non-JSON extensions** — only the 12/192 brace-expanding
  JSON case was scanned corpus-wide.
- **No claim about correctness.** This audit classifies *why the artifact citation is missing*. It does not
  check whether any document's numbers are right.

## The one concrete recommendation

The cheapest real win is **not** a documentation pass — it is fixing the instrument first, then measuring
again. Widening `ARTIFACT_RE` to expand `{a,b}`, to accept `.log`/`.csv`/`.npz`, and to *stat the path* rather
than pattern-match it would clear the d1 class outright and shrink the 192 before anyone edits a document.
Only then is the residual worth hand-triaging, because only then does the flag mean what it says.
