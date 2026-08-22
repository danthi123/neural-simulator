---
type: finding
status: negative
date: 2026-08-21
mechanism: d5-pattern-separation-setpoint-knob1
lane: integration
integration_faculty: d5-live-consolidation
seeds: [42, 43, 44, 100, 101, 102]
instrument: research/runners/_d5_graded_flip_soak.py (4-turn OFF-vs-ON no-regression + simulated
  mid-consolidation crash-rollback) run WITH the DG pattern-separation set-point ACTIVE (--sep-bias 1000),
  through the REAL EpisodicRecallOrgan.recall + recall_disclosure + continuous_engine.consolidate_used_memory
  at the production encode (train_events=40). Disjointness cross-checked by
  research/runners/_d5_pattern_separation_setpoint_derisk.py (Layer A membership overlap, numpy).
runner: research/runners/_d5_graded_flip_soak.py
external: Guzman, Schlögl, Frotscher & Jonas (2016), "Synaptic mechanisms of pattern completion in the hippocampal
  CA3 network", Science (via PubMed; DOI in body) — CA3 pattern completion runs
  on SPARSE recurrent connectivity + disynaptic motifs, NOT a dense recurrent matrix. This localizes the residual:
  our episodic readout uses a DENSE recurrent matrix, which is the cross-assembly bleed path the quantized
  up-fraction registers; sparse recurrent readout connectivity + recall-time feedback inhibition is the
  biologically-grounded next lever beyond formation-time membership separation. Local record: our own
  2026-08-08-episodic-CA3-completion-CLOSED-within-assembly-attractor-potentiation-loadbearing-WTA-still-negative.
artifacts:
  - research/findings/raw/_d5_ltu_flip_soak/soak_summary_6seed.json
---
# D5 learn-through-use default-ON flip is NO-GO — the DG pattern-separation set-point (knob-1) makes assemblies DISJOINT and closes the GRADED-read crosstalk, but a QUANTIZED binary-readout residual still fails the no-regression soak (0/6)

## Verdict

Artifact: `research/findings/raw/_d5_ltu_flip_soak/soak_summary_6seed.json` (6-seed soak, sep_bias=1000, te=40).
Reproduce: `SIM_BACKEND=cupy python -m research.runners._d5_graded_flip_soak --seeds 42 43 44 100 101 102 --sep-bias 1000`.

**0/6 GO.** With the DG pattern-separation set-point ACTIVE (sep_bias=1000, membership confirmed disjoint), the
D5 learn-through-use default-ON flip STILL fails the production no-regression soak on all six seeds
(42/43/44/100/101/102). The flip is NOT applied. `BRAIN_D5_CONSOLIDATE` stays default-OFF.

The faculty itself is intact where the assembly survives: `on_dog_rose=True` on 5/6 (the used memory strengthens),
crash-rollback intact 6/6. The blocker is `no_regression=False` on 6/6: consolidating the USED memory (dog) still
produces a reply-visible change on an UNTOUCHED neighbor topic (bird).

## What the separator DID close, and what it did NOT

The knob-1 separator works AS SPECIFIED at the membership level: numpy Layer A, s42, assembly membership overlap
`max_shared 4 -> 0` (OFF sizes [27,22,28] -> ON [27,20,24]) — disjoint, non-empty, non-dense.

<!--derived-->
It also closed the GRADED-read crosstalk it was built for. The neighbor's surfaced graded strength `depth_hold`
(mean-held max(cp_v_apical − v_hold, 0)) is now byte-stable: bird `depth_hold` delta = **0.0 exactly** on 5/6 seeds
(−0.016 mV on s44 — derived from the OFF/ON `t4_bird` values in the artifact), versus the pre-separator 0.13 mV
shift (s42 bird 30.77 -> 30.64) quoted from finding
2026-08-21-d5-graded-apical-read-conversation-visible-in-production-flip-blocked-on-emergent-assembly-crosstalk.
So the disjoint-membership set-point removes the WITHIN-assembly shared-cell path exactly as its
structural proof claimed.

The residual is a DIFFERENT read. The no-regression check fails on the QUANTIZED binary `apical_cue` (the UP-fraction
= the moat completion gate value), which is surfaced in the recall reply as "dendritic dAP completion X.XX". On 5/6
seeds bird's reply string differs OFF-vs-ON purely because that displayed completion number shifts, while bird's
`depth_hold` and its `in_memory` verdict are both preserved. The binary UP-fraction is a threshold-crossing count, so
it is knife-edge sensitive to the tiny cross-assembly recurrent bleed that disjoint MEMBERSHIP does not sever; the
continuous graded read averages over it and stays stable.

The read is deterministic (not read noise): in every seed the OFF path reads dog twice on the SAME weights and gets
byte-identical records (`off_store_flat=True`, dog t2==t4). So bird changing across the OFF/ON weight difference is a
genuine weight-mediated shift, not run-to-run GPU nondeterminism.

## Mechanistic reason

Disjoint assembly MEMBERSHIP severs the within-assembly shared-cell path (A_cell -> A_cell weights that also sit in
B's read), which is why `depth_hold` is now stable. It does NOT sever the CROSS-assembly recurrent-readout path: the
readout recurrent matrix is dense, so a neighbor's cue partially bleeds into / is fed back from the consolidated
assembly's now-stronger recurrence, and the QUANTIZED up-fraction registers that sub-threshold bleed as a
completion-number flicker. Formation-time separation (knob-1) is necessary but not sufficient; the residual lives at
RECALL/READ time.

## Secondary cost of the winner-fatigue separator

The winner-fatigue bias SHRINKS assemblies (a recruited cell is depressed for later patterns), giving small dog
assemblies (12–33 cells; dog=13 on s42, dog=12 on s44). On s102 (dog=24) the faculty itself weakened:
`on_dog_rose=False`, `dog_inmem_same=False` — a too-small/unstable dog assembly did not reliably strengthen or
complete. So sep_bias=1000 also trades against faculty reliability on some seeds.

## Per-seed (soak, sep_bias=1000, te=40)

The `dog rise` and `bird depth_hold Δ` columns are DERIVED from the per-seed raw `t2_dog`/`t4_dog` and
OFF/ON `t4_bird` `depth_hold` values in the cited artifact (soak_summary_6seed.json).

<!--derived-->
| seed | GO | dog rise (mV) | bird depth_hold Δ | bird reply differs | on_dog_rose | notes |
|------|----|--------------:|------------------:|:------------------:|:-----------:|-------|
| 42   | NO | +0.296 | 0.000  | yes | True  | dog assembly=13 |
| 43   | NO | +0.497 | 0.000  | yes | True  | |
| 44   | NO | +0.296 | −0.016 | yes | True  | dog assembly=12 |
| 100  | NO | +0.812 | 0.000  | yes | True  | |
| 101  | NO | +0.590 | 0.000  | yes | True  | |
| 102  | NO |  0.000 | 0.000  | no  | False | dog did not rise; in_mem unstable |

Every seed: `no_regression=False`, `crash_ok=True`. bird `in_memory` gate preserved on all seeds (the moat does not
flip; only the displayed completion number moves).

## Next mechanism (named, evidence-ordered)

The residual is now precisely localized to the RECALL-time QUANTIZED binary read, with the continuous graded read
already robust. In likely order of leverage:

1. **Surface the crosstalk-robust read in the disclosure.** The graded `depth_hold` is byte-stable under the
   separator; the raw binary `apical_cue` (a threshold count) is not. Making `recall_disclosure` report the stable
   graded magnitude (and gate on the stable `in_memory` verdict) instead of the knife-edge completion number would
   make bird's reply byte-identical. This is the ledger's "crosstalk-robust surfaced read" alternative and is the
   cheapest lever; it must be checked to still be conversation-visible for the USED memory.
2. **Recall-time feedback-inhibition winner-take-all.** A stronger CA3 PV-basket (ca3_pv_basket) inhibitory
   companion DURING the read, so a cue completes only its own assembly and does not partially ignite the neighbor —
   pinning even the binary up-fraction against sub-threshold cross-assembly bleed. This is the CLAUDE.md
   "companion-process" reframe: separation during FORMATION (knob-1) plus inhibition-maintained sparsity during
   RECALL.
2b. **Sparsify the recurrent readout connectivity.** Guzman et al. (2016, Science, doi:10.1126/science.aaf1836) <!--derived-->
   show real CA3 pattern completion runs on SPARSE recurrent connectivity + disynaptic motifs, not the DENSE
   recurrent matrix our readout uses. A sparse recurrent readout would shrink the cross-assembly bleed path
   directly (fewer dog->bird / bird->dog synapses to carry the strengthened recurrence into a neighbor's read).
3. **Heterosynaptic isolation of cross-assembly weights** during consolidation (heterosynaptic LTD on
   non-co-active cross-assembly synapses) so a strengthened memory becomes actively more isolated, not merely
   non-overlapping in membership.
4. **Address the assembly-shrink cost** in parallel: lower sep_bias with a k-of-N size floor, so disjointness does
   not starve the faculty (the s102 failure).

## Wiring landed on this branch (default-OFF, byte-identical when off)

Additive, no sim/ edit. The separator is armed exactly when the D5 learn-through-use flag is (so a future working
flip gets disjoint assemblies + strengthen together); with `BRAIN_D5_CONSOLIDATE` off (the default) sep_bias=0 and
assembly formation is the UNMODIFIED `emergent_assemblies` — byte-identical to HEAD.

- `research/runners/_d5_pattern_separation_setpoint_derisk.py` — the separator mechanism, ported from
  research/memory-separator-readout (`_emergent_assemblies_setpoint`, winner-fatigue intrinsic-excitability bias;
  host-applied bias, on-substrate theta SELECTION — the on-substrate spiking intrinsic-plasticity form is the
  tracked residual).
- `research/runners/_episodic_dap_dialogue_memory.py` — `EpisodicDapMemory(sep_bias=0.0)`: sep_bias>0 forms
  assemblies through the set-point (lazy import), else unmodified.
- `research/runners/d5_episodic_production_organ.py` — `EpisodicRecallOrgan(sep_bias=)` threaded;
  `get_episodic_organ(sep_bias=None)` reads `_default_sep_bias()` = D5_SEP_BIAS (1000) iff
  `d5_consolidate_enabled()`, else 0.
- `research/runners/_d5_graded_flip_soak.py` — `--sep-bias` (default D5_SEP_BIAS) so the soak runs with the
  separator active.
