---
type: biology
id: invariance-from-temporal-continuity
mechanism: Inferior-temporal neurons respond to an object INVARIANTLY across size and position; temporal continuity is the proposed teacher for that invariance
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-02-emerge50-trace-rule-GO.md
current_status: "The trace rule works ON-SUBSTRATE for one invariance -- CATEGORY membership, not position. Held-out sub-category super-acc 0.958 (6 seeds, per-seed 0.917-1.000) vs shuffled-temporal-order 0.556 and permuted-co-occurrence 0.611; held-out within-super overlap 0.503 vs cross-super 0.011; dAP-lesion 0.000. POSITION, SIZE and VIEWPOINT invariance -- the property the ledger row is about -- were NOT tested."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "in which objects are recognized as the same"
    note: "position constancy defined -- recognized as the same regardless of location in the visual field"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "neurons does not vary when an object changes posi"
    note: "the single-neuron measurement: IT responses are unchanged by position shifts within their large receptive fields (Kandel Fig 24-7B)"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the invariance of their shape selectivity even to very"
    note: "size invariance -- shape selectivity survives very big changes in stimulus size; and IT lesions break size constancy"
constants:
  emerge50_trace_decay: 0.8
  emerge50_bout_len: 12
implemented_by:
  - research/runners/_emerge50_trace_rule_derisk.py
findings:
  - research/findings/2026-07-02-emerge50-trace-rule-GO.md
---

# Invariance is the target property; temporal continuity is the candidate teacher

**The claim the code must respect.** Kandel: position constancy is the case "in which objects are recognized as
the same regardless of their location in the visual field," and the pattern of response of many inferior temporal
"neurons does not vary when an object changes position within their large receptive fields." Size behaves the same
way — one of the most striking properties of individual IT neurons is "the invariance of their shape selectivity
even to very big changes in stimulus size," and IT lesions break size constancy in monkeys. Viewpoint invariance
is *graded along the hierarchy*: posterior face patches are tuned to viewing angle, anterior ones are robust to
it.

**Why this row is tagged with it.** The A4b wall is "rich object recognition / IT invariance-at-scale," and the
named surpass is a DiCarlo position-invariance test plus a Földiák trace rule plus competitive pooling. The
biology above is what "invariance" has to mean when that test is written: an *unchanged single-unit response*
across a transformation, not merely a classifier that still gets the label right.

## What is established — and the gap between it and this row

`_emerge50_trace_rule_derisk.py` is a **6-seed on-substrate GO** for temporal-continuity learning, verified
against its banked aggregate `research/findings/raw/_emerge50_trace_rule_6seed.json`: presenting members of the
same superordinate category in contiguous bouts, with a slow eligibility trace on the L2 pre-synaptic activity,
binds them to shared L2 columns. Held-out sub-category **super-acc 0.958** (per-seed 1.00/0.92/0.92/1.00/0.92/1.00,
chance 0.50), held-out **within-super overlap 0.503 vs cross-super 0.011**. The load-bearing control is the one
that isolates the mechanism: presenting the *same* members in **shuffled temporal order** collapses it to 0.556,
and permuted co-occurrence to 0.611; the dAP-lesion reads 0.000.

**The honest gap.** That experiment learns invariance to *which exemplar of a category* is shown. It does **not**
test invariance to position, size, or viewpoint — the transformations the Kandel sources above are about, and the
ones the ledger row names. Treating the trace-rule GO as evidence for IT invariance-at-scale would be exactly the
substitution this row's "validate-or-retire V2/IT" item exists to prevent.

⚠️ **Provenance honesty.** Földiák 1991, the rule the runner implements, is **not in the local corpus** and was
not read for this entry. It is cited by the finding and named in the roadmap; the anchors above are Kandel's, and
they establish the *target property*, not the learning rule. No external anchor is recorded here because none was
verified — an unverified anchor is the folklore this checker exists to reject.

## What this entry cannot catch

`trace_decay` and `bout_len` are recorded under `constants` as the values the GO was measured at, deliberately
**not** under `constraints_config`. The biology constrains them only as an inequality — the trace must outlast a
single presentation, i.e. `bout_len > 1` and the decay slow relative to a bout — and the checker compares by
equality. Pinning 0.8/12 would fire on a legitimate re-tuning, and a gate that cries wolf gets switched off. The
condition that genuinely must hold (grouped, non-shuffled presentation) is an experimental arm, not a config
number, so `biology_check --config` cannot see it at all.
