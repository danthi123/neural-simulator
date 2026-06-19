# Roadmap #4 (TRUE-ONE-BRAIN) — fully-spiking nav motor read-out on the merged "one brain": SCOPE + cheap-first de-risk

**Date:** 2026-06-18
**Type:** SCOPE + CHEAP-FIRST DE-RISK (numpy/CPU composition + moat smoke DONE; few-seed GPU nav comparison appended below).
**Roadmap item:** `2026-06-18-full-spikeification-shared-substrate-roadmap.md` §3 #4 — make the merged nav
action-selection a FULLY-SPIKING decision (the Wang-2002 accumulator `sel_X` + Lo-Wang commit-burst `commit_X`
threshold-crossing IS the decision), retiring the host Python argmax (`g11_bg_runner.py:6901-6909`).
An honest BOUNDARY (a clean spiking decision that navigates WORSE than the host argmax) is a valid deliverable.

**Bottom line (cheap-first):** the fully-spiking read-out **COMPOSES cleanly on the merged "one brain" with the
no-confab moat intact**, and the entire commit-burst-as-primary + Cisek-urgency machinery is **already built and
already threaded through `run_moving_goal_episode`** — so making the merged nav decision fully-spiking is a
RUNNER-ONLY change (two flags), **NO `sim/` edit**. The verdict on whether it navigates as well as the host
argmax is the few-seed GPU comparison (below); the prior standalone evidence says it is the documented
substrate **BOUNDARY** (fully-spiking WTA historically 14.7 vs thal-argmax 2.3), with the Cisek urgency bound
the lever that closes the silent-commit gap.

---

## 1. COMPOSITION (numpy/CPU) — does `enable_spiking_wta_readout` compose on the merged bridge? **YES.**

`research/runners/_merged_spiking_readout_derisk.py --compose-smoke` (numpy/CPU, seed 42):

- `build_merged_nav_conv_bridge(enable_spiking_wta_readout=True)` builds: **55 regions, 3124 neurons, 781932 synapses.**
- The spiking-WTA selection + commit layers are all built: `sel_{N,E,S,W}` (Wang-2002 NMDA accumulators) +
  `sel_FS_{N,E,S,W}` (Rutishauser selective inhibition) + `commit_{N,E,S,W}` (Lo-Wang burst pools) + `commit_OPN`
  (shared omnipause).
- **ZERO index collision:** the WTA indices (|wta|=160) are fully disjoint from the parser+dlPFC slices
  (|conv|=2166), overlap=0. The `sel_X`/`commit_X` region NAMES are nav-prefixed and unique — no collision
  with `parse_conj`/`parse_role`/`cortex_ctx`/`dlpfc_wm`/`rf`/`cortex_it`/`gen_*`/`limbic_*` by construction.
- The merged-config invariant holds: `cfg.enable_homeostasis is False` (the synaptic-scaling clip foot-gun
  that would crush the frozen conv weights stays off).
- **The parser still parses voice-invariantly WITH the WTA layer present:** active `dog go north` →
  `{agent:dog, action:go, patient:north}`; passive `north go dog` → agent=dog. The WTA layer does not perturb
  comprehension.

**⇒ The merge builder's existing `enable_spiking_wta_readout` pass-through is correct and composes. The
spiking-WTA layer is purely additive (appended inside the nav region list, array-disjoint from the conv slices).**

## 2. NO-CONFAB MOAT (numpy/CPU) — intact? **YES.**

`_merged_spiking_readout_derisk.py --moat-smoke` (numpy/CPU, seed 42), via `MergedNavConvAgent`:

- `hear('dog go north')` → `{agent:dog, action:go, patient:north}` (the on-merged-bridge parser comprehends).
- `what_does('dog','go')` → **`'north'`** (recall).
- `what_does('river','look')` → **`None`** (abstain — the moat holds; no confabulation on the unstored cue).

**Why the default agent is the right moat check:** the spiking-WTA read-out is a NAV-side substrate
(`sel_X`/`commit_X`, array-disjoint from the parser + RF composer that carry the moat). The conversational
recall/abstain decision is **identical** whether the nav read-out is host-argmax or spiking-WTA — the WTA
layer cannot reach the parser/composer. §1 proves the WTA layer builds co-resident; §2 proves the moat holds
on that same merged stack. Together: **#4 composes without weakening the moat.**

---

## 3. THE EXACT FLAG/PATH — commit-burst-as-primary + urgency bound (the answer to KEY QUESTION 2)

**The commit-burst is ALREADY the primary decision in the runner — `readout_source="spiking_wta"` is NOT a
naive argmax over sel rates.** Read `g11_bg_runner.py:6889-6909`:

```python
if _use_commit:                       # _use_commit = (spiking_wta AND enable_commit_burst AND commit pools built)
    _primary = commit_counts          # the all-or-none commit_X burst = the PRIMARY decision
    _fallback = sel_counts            # sub-threshold trial -> the sel_X accumulator's lean (Shadlen affordance)
...
if max(_primary.values()) > 0:
    action_idx = max(range(N_ACTIONS), key=lambda i: _primary[...])   # which commit pool BURSTED
    _decision_path = "primary"        # commit burst fired = the spiking decision
elif _fallback is not None and max(_fallback.values()) > 0:
    action_idx = max(... _fallback ...)                              # silent commit -> sel-lean
    _decision_path = "fallback"
else:
    action_idx = random...            # both silent (genuinely undriven trial)
    _decision_path = "random"
```

So the host `max(...)` over `commit_counts` is **OBSERVING which commit pool fired the all-or-none burst** (under
a decisive commit the loser counts are ~0). The DECISION is the **`commit_X` threshold CROSSING** (the spiking
termination event), exactly as #4 requires; the host argmax is a tie-break of last resort. The runner even
tracks `_decision_path_counts` (primary/fallback/random) so a GO requires the commit firing reliably, NOT
quietly leaning on the argmax fallback.

**The Cisek urgency / collapsing bound** is also already in the runner (`g11_bg_runner.py:3769`,
`urgency_max_pA`, RECOMMENDED 180): a ramping action-INDEPENDENT current injected into all `sel_X` over the
readout window so even a weak late release crosses the bound → the commit bursts → the silent-commit→argmax
fallback is eliminated (standalone seed 42: random fallback 25%→1.4%, thal-winner alignment 80%→94.8%, commit
separation 15×→49×).

**The exact path to make the merged nav decision fully-spiking:**
- BUILD path (`MergedNavConvAgent` / `build_merged_nav_conv_bridge`): pass `enable_spiking_wta_readout=True`
  (already a pass-through kwarg — composes, §1).
- EPISODE path (the nav gate, `run_moving_goal_episode`): pass `readout_source="spiking_wta"`,
  `enable_commit_burst=True` (default), `urgency_max_pA=180.0`. `run_moving_goal_episode` already accepts all
  three (`g11_bg_runner.py:3690/3715/3769`) and builds its own nav regions internally via
  `build_bg_brain_regions(enable_spiking_wta_readout=(readout_source=="spiking_wta"), ...)`; the conv slice is
  the appended `extra_regions`/`extra_pathways`, array-disjoint from `sel_X`/`commit_X`. The urgency-injection
  slice is built post-init from `region_indices_cp` (`g11_bg_runner.py:4537-4545`).

**⇒ NO `sim/` edit. NO merge-builder edit (the pass-through exists).** The only change this cycle is RUNNER-ONLY:
expose `--readout-source` + `--urgency-max-pa` on `_nav_gate_merged_run.py` (the `gate6_{standalone,merged}_seed{N}.json`
producer) so the merged nav gate can be run with the fully-spiking decision. Done this cycle (additive flags,
default `motor` = the historical host-argmax gate, byte-preserved).

---

## 4. CHEAP NAV COMPARISON (GPU, few seeds) — spiking-WTA+urgency vs host-argmax vs thal

(Appended after the GPU run. The comparison config is a CHEAPER grid-8 / short multi-goal episode — not the
grid-32/1800-step flagship — to get the read-out deltas fast within budget.)

<!-- GPU_COMPARISON_RESULTS -->

---

## 5. VERDICT + recommended next step

- **COMPOSITION: GO** — `enable_spiking_wta_readout` composes cleanly on the merged bridge, moat intact, NO sim/ edit.
- **FLAG/PATH: settled** — commit-burst-as-primary + Cisek urgency are already built + threaded through
  `run_moving_goal_episode`; the merged nav gate just needs `readout_source="spiking_wta"`,`urgency_max_pA=180`.
- **NAV-SCORE verdict:** see §4. Prior standalone evidence frames this as the documented substrate **BOUNDARY**
  (fully-spiking WTA underperformed host/thal argmax on the nav score); the honest deliverable is the clean
  spiking decision + its nav-score gap, with the urgency bound the lever that narrows it.

**Recommended next step (for the controller):** if the few-seed GPU comparison shows the spiking-WTA+urgency
gap to thal/motor is within ~the noise floor (or a small documented cost), run the full 6-seed
`gate6_{standalone,merged}` campaign at `readout_source="spiking_wta" --urgency-max-pa 180` and report the
merged spiking-decision nav score as the #4 deliverable (GO if within noise; honest BOUNDARY otherwise). If the
gap is large, report the BOUNDARY at few-seed (the substrate finding) and STOP — do not fake a pass.

## Files + reuse
- `research/runners/_merged_spiking_readout_derisk.py` (new) — the cheap-first compose + moat smoke.
- `research/runners/_nav_gate_merged_run.py` (extended) — `--readout-source` + `--urgency-max-pa` (additive,
  default `motor` = the historical gate).
- Reused verbatim: `g11_bg_runner.py` `run_moving_goal_episode` (readout_source/urgency/commit already wired),
  `build_bg_brain_regions(enable_spiking_wta_readout=...)`; `nav_conv_merged_bridge.py`
  `build_merged_nav_conv_bridge`/`MergedNavConvAgent`/`conv_extra_regions_pathways`/`finalize_conv_for_nav_gate`;
  `nav_gate2a_aggregate.py` (the merged-vs-standalone scorer).
