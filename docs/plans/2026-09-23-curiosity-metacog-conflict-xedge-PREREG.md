# Pre-registration: curiosity driven by metacognition's own spiking read-out (one cross-organ edge)

**Written 2026-09-23, before the smoke run of `research/runners/_curiosity_metacog_conflict_xedge_derisk.py` and
before any 6-seed run.** The gate constants in the runner (`G1_RHO_MAX` ... `G8_RHO_MIN`) are copied from §3 of this
document. Any later change to a gate or threshold goes in the AMENDMENT LOG (§6), with a timestamp and a list of every
result artifact already seen at that time.

## 1. Claim under test

A recall that metacognition reads as low-confidence makes curiosity's ASK (crave) pool fire. A recall it reads as
confident leaves ASK quiet. This must be computed by neurons and synapses. The path:

1. Metacog's production workspace WTA (unmodified E1 organ, production per-region parameter heterogeneity).
2. Metacog's own second-order margin comparator on `meta_schema`, using the organ's legacy `margin` topology:
   assembly k excites meta_k, and assembly k drives an inhibitory relay that inhibits meta_(not k). The comparator
   gains are heterogeneous, drawn per postsynaptic neuron.
3. ONE declared cross-organ edge, `meta_schema -> ask` (excitatory, fixed weight), on one `merge_organs` pool.

Host code only drives metacog's input evidence, the same way production does. The curiosity neuromodulator is not
installed, and `current_novelty_signal` is never set. So every ASK spike is caused by metacog spikes through this edge.

## 2. Operating point (declared; calibration seed = 42)

The operating point was chosen from scratchpad probes on **seed 42 only**, 2026-09-23, before this document:
`CMP_EXC=(1.2,2.4)`, `CMP_REL=2.5`, `CMP_INH=(6.0,14.0)`, relay 2x30, `XEDGE_W=4.0`.

- Probe (seed 42): ASK read 5.7 Hz at evidence 0 and 0.2 Hz at evidence 1, monotone across 11 levels.
- The runner reseeds the comparator weight draw (crc32-keyed), so its seed-42 numbers will differ somewhat from the
  probe.
- Seeds 43/44/100/101/102 are HELD OUT, and the verdict reports them separately.
- The probes also showed that uniform (non-heterogeneous) comparator weights, or metacog without heterogeneity, give
  a quantized, synchronous response. Those variants are not used.

## 3. Gates (per seed; GO requires every REQUIRED gate on 6/6 seeds 42/43/44/100/101/102)

| id | required | measures | pass condition |
|---|---|---|---|
| G1 | yes | monotone coupling | Spearman rho(evidence, level-mean ASK Hz) over 11 levels <= -0.8, AND intact ASK range >= 1.0 Hz (else UNDEFINED = fail) |
| G3 | yes | lesion of ONLY the metacog->curiosity edge | zeroing exactly the `x_metacog_meta_to_curiosity_ask` synapses removes >= 80% of the ASK dynamic range (`tools.lab.attributable_to`) AND the lesioned ASK peak is <= 20% of the intact peak |
| G3b | yes | metacog's evidence still varies under that lesion | metacog balance per level is EXACTLY equal (==) intact vs lesioned, and still varies. The metacog raster sha256 and the comparator raster sha256 are equal intact vs lesioned |
| G4 | yes | permutation NULL DISTRIBUTION | Spearman rho over the 88 per-rep observations (11 levels x 8 jittered reads) vs 10,000 random permutations of the evidence labels. One-sided p = P(null <= observed) <= 0.01. Percentile reported |
| G5 | yes | metacog unchanged, EXACT | metacog balance, threshold, confident flags (==) and workspace/workspace_fs spike-raster sha256 are identical between the coupled pool and (a) the same pool without the edge and (b) a bare [metacog, curiosity] pool with no comparator and no edge |
| G6 | yes | determinism | the intact-arm digest (sha256 over per-rep ASK rates, balances, raster hashes) is identical in a FRESH subprocess |
| G7 | yes | class-symmetry anti-cheat | evidence driven into the OTHER assembly: rho <= -0.8. The circuit does not know which class is "correct" |
| G8 | yes | mechanism specificity | lesion metacog's comparator relay (relay->meta inhibition) with the edge intact: rho > -0.5, or UNDEFINED. The coupling must need the margin computation, not just metacog activity |
| S1 | **no (secondary)** | does the synaptic drive ALONE reach the production curious threshold? | reported only. Expected NO at this operating point: probe peak ~6 Hz vs the organ's ~21 Hz threshold |

Integrity smokes. These are reported and are NOT counted as evidence, because they pass by construction if the
code is right:
- byte-off: base connectivity is identical without the edge.
- restore-exact: re-reading after the lesions restores the intact digest.
- no host novelty / no neuromodulator installed.

## 4. What a GO would and would not mean

- **GO means:** on this 2-organ merged pool, curiosity's ASK firing is a monotone, class-symmetric function of
  metacognition's own spiking margin computation. The effect is carried by one lesionable edge, and metacog is
  unperturbed.
- **GO does not mean:**
  - that production curiosity now fires on low-confidence recalls (S1, and nothing is wired into the chat path);
  - that the edge self-organized (the weights are hand-set);
  - that it runs on the 11-organ production pool (that pool's gain-0 freeze forbids cross-edges into `ask`, which
    needs a seam change).

  Those are the next rungs.

## 5. Compute

- Smoke: 1 seed (42), local, numpy CPU.
- 6-seed: on the mini-PC pool via `tools/pool_queue.sh`, pinned to this branch's revision.
- Output: `research/findings/raw/_curiosity_metacog_conflict_xedge_6seed.json`.

## 6. AMENDMENT LOG

(none yet)
