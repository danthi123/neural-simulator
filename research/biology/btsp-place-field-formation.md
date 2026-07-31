---
type: biology
id: btsp-place-field-formation
mechanism: Behavioral timescale synaptic plasticity (BTSP) — dendritic plateau creates a CA1 place field
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-31-gap5-place-specificity-is-ONE-INDUCTION-PASS-laps-erase-it.md
current_status: "PLACE-SPECIFIC at laps=1 dwell=30 w_max>W0 — 4.49x vs position-shuffled null, p=0.0050, 6 seeds"
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "termed place cells, fire only when"
    note: "place cells exist and are spatially selective — the background claim"
  - path: "PMC7289271 (Bittner, Milstein, Grienberger, Magee 2017, Science 357:1033)"
    anchor: "peak and have a center of mass"
    note: "EXTERNAL, not in the local corpus — cannot be resolved offline; anchor recorded for when it is added"
constants:
  ramp_width_cm: 125.4
  ramp_width_sem_cm: 7.1
  ramp_onset_before_induction_s: 3.8
  fold_potentiation_at_dt0: 3.12
  tau_long_s: 1.31
  tau_short_s: 0.69
constraints_config:
  - key: laps
    value: 1
    why: "BTSP is ONE-SHOT: a SINGLE plateau creates a field. Repeated traversals re-potentiate every position and ERASE the field. Measured 2026-07-31: place-specificity decays 4.40x -> 2.57x -> 1.11x as laps go 1 -> 2 -> 5. A protocol that repeats the induction is not testing induction."
operating_point:
  - key: w_max
    requires: "> W0 (the initial weight)"
    why: "NOT a tuning preference — a precondition for the mechanism to be measurable at all. At w_max=150
      against W0=250 the clamp drags every weight DOWN, so the run measures clamp depth rather than
      potentiation. Asserted by tools.lab.bound_check at the point the bound is CHOSEN."
    measured: "research/findings/2026-07-31-gap5-tuned-point-is-INSIDE-the-bound-trap-97pct-of-dW-is-the-clamp.md"
companion_processes:
  # WHAT THE REAL SYSTEM RUNS ALONGSIDE THIS, THAT WE REPLACED WITH A CONSTANT. This block exists because
  # the proxy usually DOMINATES the measurement: the first entry below owned 97% of a result we read as BTSP.
  - process: "weight-growth limitation (heterosynaptic depression / synaptic scaling — a competitive process
      that renormalises non-potentiated synapses when one is potentiated)"
    status: proxied
    proxied_by: "w_max, a hard per-synapse clamp"
    proxy_share_measured: "0.97 of the observed |dW| at the tuned operating point — identical in the lr=0
      control, i.e. the LEVER moved 3% and the PROXY moved the rest"
    why_it_matters: "a clamp is a scalar where biology runs a process. It does not merely fail to learn; it
      destroys weights uniformly, which reads as a substrate limitation and invites deeper tuning into the trap
      (w_max walked 110 -> 150 -> 220, with 150 selected as 'optimal', because clamp depth was what the metric
      rewarded)."
  - process: "the depression arm of the BTSP kernel itself — the plasticity kernel is BIDIRECTIONAL in Δt,
      not potentiation-only"
    status: implemented_but_hidden_by_the_metric
    proxied_by: "not proxied; MEASURED, then rectified away by the read-out"
    proxy_share_measured: "circ_resultant clips negatives internally and returned 0.000000 for an lr=0 control
      whose mean |dW| was 21.94 — every increment was negative and the metric reported a clean zero"
    why_it_matters: "reported alongside by tools.lab.sign_budget. Any metric built on a rectified quantity
      silently scores the residual of a destructive process."
implemented_by:
  - research/runners/_gap5_btsp_place_field_derisk.py
  - research/runners/_gap5_fieldquality_gpu6.py
findings:
  - research/findings/2026-07-31-gap5-place-specificity-is-ONE-INDUCTION-PASS-laps-erase-it.md
  - research/findings/2026-07-31-gap5-tuned-point-is-INSIDE-the-bound-trap-97pct-of-dW-is-the-clamp.md
---

# BTSP — dendritic plateau creates a place field, in ONE shot

**The claim the code must respect:** a *single* dendritic plateau potential creates a CA1 place field, with an
asymmetric seconds-long eligibility kernel whose centre of mass sits *before* the induction location.

**What this cost before it was written down.** The gap#5 arc ran for weeks with `laps=5` — five traversals of the
track — while sweeping density, `w_max`, `lr` and eligibility τ in turn. None of those was the operative
variable. `laps` was, and it was never swept because nothing connected the runner to the paper. A 21-agent
research round was spent in 2026-07-31 re-establishing a fact that was knowable on day one.

**The measured consequence** (`AGG_laps_dwell.json`, 3 seeds/cell, permutation-gated, outside the clamp trap):
place-specificity ratio **4.40× at laps=1**, **2.57× at laps=2**, **1.11× (n.s.) at laps=5**. A single pass alone
reaches the σ=5 oracle's 4.53× to within 3%.

**⚠️ Provenance honesty.** The Bittner constants above come from the 2026-07-31 research round, which read the
paper's PMC main text. That paper is **not in the local corpus**, so `biology_check.py` cannot resolve its anchor
offline; the entry records that explicitly rather than implying verification it did not do. The supplement, which
holds the numeric peak/COM values, is not open access and was **not** read.

**⚠️ A citation from that same round did NOT survive checking.** It attributed to Kandel, as a verbatim line, the
phrase *"usually have more than one firing field but the fields have no apparent spatial relationship"*. That
string is **not present** in `full-book.txt`. It was caught by this checker before being committed. The Kandel
anchor used here is one that was verified to resolve.
