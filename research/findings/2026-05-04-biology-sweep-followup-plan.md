# Biology sweep follow-up plan — both outcomes pre-staged

**Generated:** 2026-05-03 ~22:15 EDT (autonomous overnight, pre-decision)
**Will be resolved:** ~02:00-03:00 EDT when biology sweep completes

The biology sweep tests 4 conditions (baseline / +FS / +Topo / +Topo+FS)
× 6 seeds in parallel-3 with anti-cheat control. Outcomes split into
2 directions; this doc pre-stages what happens for each.

---

## Outcome A: at least one condition gives aligned ≥ 4/6

**Most likely paths:**
- `+Topo+FS` aligned 4-6/6: combined biology fix is required.
- `+Topo only` aligned 4-6/6: topographic prior is THE thing.
- `+FS only` aligned 4-6/6: lateral inhibition is THE thing.
- All conditions aligned: minimal arch can learn but cascade-free
  is the prerequisite.

### Immediate next experiments (Tier 1, run within ~5 hours)

**A1: identify minimum sufficient biology**
If `+Topo+FS` aligned 6/6 but `+Topo only` and `+FS only` both 0/6,
then BOTH are required. If `+Topo` alone gives 4/6, topography is
sufficient — FS is a bonus.

Use existing `experiment_runner.py` with new `experiments/minimum_biology.yaml`:
```yaml
name: minimum-biology
runner: research.runners.text_minimal_isolation
output_dir: research/findings/raw/g11_bg
parallelism: 3
seeds: [42, 43, 44, 100, 101, 102]
base_args:
  n-events-per-direction: 1000
  stim-steps-per-step: 100
  reset-steps: 50
  dt-ms: 1.0
conditions:
  # Half-strength topography (1.3/0.8 instead of 1.5/0.7)
  - name: topo_weak
    args:
      topographic-bias-factor: 1.3
      off-target-bias-factor: 0.8
  # Reduced FS pool (1 neuron instead of 3)
  - name: fs_minimal
    args:
      enable-motor-fs: true
      n-motor-fs-per-action: 1
  # Stronger topography (2.0/0.5)
  - name: topo_strong
    args:
      topographic-bias-factor: 2.0
      off-target-bias-factor: 0.5
out_stats_template: "text_eval_minbio_{name}_seed{seed}.json"
```
Identifies the minimum effective biology dose. Important for
generalizing later: too-strong fixes are not biology-grounded.

**A2: re-introduce cascade with reduced strength**
If minimal arch + biology-fix aligns ≥ 4/6, the next question is
"how much cascade can we add back before alignment breaks?"

Add `--cluster-a-strength` / `--cluster-e-strength` parameters to
the existing v2 architecture (in g11_bg_runner.py) that scale
`weight_mean` of cluster_a/cluster_e pathways. Sweep:
```yaml
conditions:
  - name: cascade_0
    args: {cluster-a-strength: 0.0, cluster-e-strength: 0.0}  # full minimal
  - name: cascade_30
    args: {cluster-a-strength: 0.3, cluster-e-strength: 0.3}
  - name: cascade_60
    args: {cluster-a-strength: 0.6, cluster-e-strength: 0.6}
  - name: cascade_100
    args: {cluster-a-strength: 1.0, cluster-e-strength: 1.0}  # full v2
```
Plot aligned ratio vs cascade strength. The threshold tells us how
much cascade interference biology-fixes can survive.

**A3: scale up to v2 architecture with biology fixes**
After A2 confirms a cascade strength that maintains alignment, add
biology fixes to v2 and re-run the original 100-episode curriculum
(text_eval_R3R6_100ep_HebOff_v2 baseline). If THIS aligns ≥ 4/6,
we have a real word-action learning result for the v2 architecture
— major scientific advance.

### Documentation updates (immediate)

- Update CLAUDE.md: "Biology fixes broke the 0/N alignment streak."
  Specify which fix(es) and what aligned ratio achieved.
- Update README.md: word-action row from "not real learning per
  permuted-label" to specific positive result.
- Update CURRENT-STATE.md: validated capability with specific config.
- Wiki-sync: capture as significant milestone.
- Findings doc: full analysis with confusion matrices, per-seed
  breakdowns, mechanism interpretation.

---

## Outcome B: all conditions stay at aligned 0-1/6

This is the harder case. Indicates the issue is more fundamental
than cascade interference + biology priors.

### Hypotheses to test (Tier 2, more substantive work)

**B1: eval methodology bug**
Build a TRIVIAL synthetic test: hardcoded weights that trivially
encode word→action via a one-shot synaptic pattern. If the eval
gives aligned 6/6 on the hardcoded model, the eval is fine. If it
gives 0/6 even on the hardcoded model, the eval is broken.

```python
# research/runners/eval_sanity_check.py
# Build minimal arch, set language→motor weights to ONLY connect
# correct (word, motor) pairs. Run eval. Should align 6/6.
```

If this fails → fix the eval. If passes → eval is correct, deeper
issue elsewhere.

**B2: sparse-code overlap is fundamentally too high**
Test orthogonal codes (`--token-sparsity 0.05`) which gives ≤ 1
overlap between word codes. Check if alignment improves.
Already wired (token_sparsity flag is in text_minimal_isolation).

**B3: STDP rule itself can't differentiate sparse codes**
Replace STDP+R-STDP with explicit supervised gradient learning on
the language→motor weights only. Gradient learning is non-biology
but lets us see what's possible. If gradient learning aligns,
biology learning rules are the bottleneck.

```python
# Add --use-supervised-readout flag to text_minimal_isolation
# Computes a one-hot target for motor pool given word, applies
# small-step gradient on language_input → motor_X weights at end
# of each event.
```

**B4: training dose is fundamentally too low**
Test 10x events (10000 events/direction = 40000 total). Long run
(~9 hours per seed). 1 seed × 4 hours of GPU = should be tractable
overnight after current biology sweep finishes.

### Documentation updates (immediate)

- Update CLAUDE.md: "Biology-grounded fixes do NOT break the 0/N
  alignment streak. Architecture has fundamental issue beyond
  cascade interference."
- Findings doc: explicit "what we ruled out" + "what remains
  hypothesized" + "next decisive test".

---

## Common to both outcomes

### Update infrastructure
- Result aggregator built-in config "biology" handles output ✓
- Experiment runner can launch follow-ups via YAML ✓
- Universal progress format displays in webapp ✓

### Decision flowchart

```
biology_sweep done
       │
       ▼
aligned ≥ 4/6 in any condition?
   │                         │
  yes                        no
   │                         │
   ▼                         ▼
A1 (min biology) +    B1 (eval sanity check)
A2 (cascade reint) +
A3 (v2 + biology)         If sanity OK:
   │                         B2 (sparse 0.05)
   │                         B3 (gradient readout)
   ▼                         B4 (10x training)
Document new                 │
operational best             ▼
                          Document deeper
                          issue, propose
                          architectural rebuild
```

## Schedule

| Time | Stage | Auto/Manual |
|---|---|---|
| ~22:30 EDT | minimal-iso batch 1 done | auto |
| ~00:00 EDT | minimal-iso batch 2 done | auto |
| ~00:00 EDT | biology waiter triggers anti-cheat | auto |
| ~00:10 EDT | biology sweep 4×6 starts | auto |
| ~03:30 EDT | biology sweep done | auto |
| ~03:35 EDT | result_aggregator runs (or manual) | manual |
| AM | based on outcome, launch A* or B* | manual |

If user is asleep at biology sweep completion, autonomous-runs skill
should: parse results, decide A vs B, launch the FIRST follow-up
experiment in the chosen branch, write findings doc, schedule next
wakeup. No need to wait for user.

## Pre-built tooling status

- [x] `research/result_aggregator.py --config biology` runs immediately
- [x] `research/experiment_runner.py` can launch any follow-up YAML
- [x] `sim/progress.py` for any new runner emits structured events
- [ ] `experiments/minimum_biology.yaml` (build before A1 fires)
- [ ] `experiments/cascade_reintro.yaml` + cluster strength flags in
      g11_bg_runner.py (build after A2 chosen)
- [ ] `research/runners/eval_sanity_check.py` (build if B1 fires)

I'll pre-build the YAML configs for A1 and B-branch tooling now while
GPU is busy, since they require no GPU.
