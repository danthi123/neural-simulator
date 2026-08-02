---
type: finding
status: contributing
date: 2026-08-02
mechanism: deep-credit-on-spikes
artifacts:
  - research/runners/_gap4_fa_kp_alignment_probe.py
  - research/runners/_gap4_fa_kp_alignment_hier3.py
---

# gap#4 crux — the KP-depth-rescue is UNTESTABLE on spikes at depth-3+ because the wall is UPSTREAM of the feedback question: the transport-free LOCAL rule does not get a DEEP (N>=3) spiking net into the learning regime at all (both FA and KP collapse to majority-class), so credit-alignment — a LEARNED quantity — is undefined there; the one VALID depth-2 point is KP > fixed-FA (directional, consistent with the rate result)

<!--derived-->
**One-line verdict.** After two failed attempts to BUILD an obligatory-depth-3 accuracy task on spikes (nested-XOR =
unoptimizable parity; hier3 = memorization shortcut + depth-3 underfit), this measures the depth-rescue MECHANISM
directly and task-independently, via CREDIT ALIGNMENT: per hidden layer, the cosine between the DELIVERED FA/KP credit
(`_chained_fa_grads`) and the TRUE surrogate-BPTT credit (`backward_unroll_xp`) on the SAME forward state. The result
relocates the wall: **at N=2 (both arms genuinely train) KP's deepest-hidden alignment (+0.435) exceeds fixed-FA's
(+0.255) — directionally consistent with the rate KP-rescue — but at N>=3 NEITHER arm trains** (both collapse to
byte-identical majority-class output), and **alignment is a LEARNED quantity** (Lillicrap et al. 2016, *Random synaptic
feedback weights support error backpropagation for deep learning*, Nat. Commun. 7:13276: the forward weights ADAPT so
that W^T aligns with the fixed feedback — that adaptation is WHY FA descends), so a net that never left chance has only
random-init alignment noise. **⇒ the KP-vs-fixed-FA feedback question cannot be adjudicated at N>=3 on spikes, because
the binding wall sits UPSTREAM: the transport-free local rule does not get a deep spiking net into the learning regime
in the first place.** No `sim/` edit (standalone probe, reuse-by-import).

## The measurement (task-independent — this is the point)

<!--derived-->
For each depth N in {2,3,4} and seed, on the depth-2 XOR->threshold task (matched config hidden 32, T 24, 200 epochs,
lr 0.05): train the FIXED-FA arm and the KP arm via the runner's own `_train_snn_arm`; then on a fresh held batch, from
ONE forward pass, compute (a) the TRUE per-hidden-layer weight gradient via surrogate-BPTT (`backward_unroll_xp`) = the
reference correct-credit direction, and (b) the DELIVERED per-hidden-layer weight gradient via the transport-free
chained-FA/KP credit (`_chained_fa_grads`, `kp_cfg=None` so Y is used as-learned). Per layer, alignment = cosine(true,
delivered). Hidden li=0 is the DEEPEST (furthest from output); li=N-1 is the TOP (reads output error directly). Probe:
`research/runners/_gap4_fa_kp_alignment_probe.py` (reproduce: `SIM_BACKEND=numpy .venv/bin/python -m
research.runners._gap4_fa_kp_alignment_probe --n-list 2 3 4 --seeds 42 43 --epochs 200`).

## Alignment table — XOR (mean over seeds 42, 43)

<!--derived-->
| N | arm | deepest hidden (li=0) | top hidden (li=N-1) | held acc | trained? |
|---|---|---|---|---|---|
| 2 | fixed-FA | +0.255 | +0.494 | 0.75–0.88 | YES |
| 2 | KP | +0.435 | +0.498 | 0.84–0.87 | YES |
| 3 | fixed-FA | +0.198 | +0.064 | 0.45–0.54 | NO (chance) |
| 3 | KP | −0.431 | −0.150 | 0.45–0.54 | NO (chance) |
| 4 | fixed-FA | +0.479 | −0.245 | 0.45–0.54 | NO (chance) |
| 4 | KP | +0.313 | −0.399 | 0.45–0.54 | NO (chance) |

## The decisive read — CONFOUNDED BY TRAINING-FAILURE-AT-DEPTH, which is itself the finding

<!--derived-->
At N>=3 NEITHER arm trains: accuracy collapses to chance, and the fingerprint is that FA and KP give byte-identical
held/train accuracy at BOTH N=3 and N=4 (the net collapsed to majority-class prediction, accuracy set purely by class
balance, independent of N or arm). The N>=3 alignment numbers are scattered random-init noise (e.g. FA N=4 deepest
+0.980 at one seed vs −0.022 at the other), with the output layer pinned at +1.000 — the degenerate-dynamics signature.
Because alignment is a LEARNED quantity (Lillicrap 2016), a net that never organized its forward has no meaningful
credit-path alignment to read. So the intended "does FA degrade with depth while KP holds" trend CANNOT be measured
here — not a rescue, not a clean attenuation, not a measurement bug, but **the depth-rescue question is premature**.

<!--derived-->
**The one VALID data point + the sanity check.** N=2 XOR (both arms genuinely trained): KP deepest-hidden alignment
+0.435 > fixed-FA +0.255 — directionally consistent with KP-realignment helping the deepest layer, but a SINGLE valid
depth point cannot establish a gap-widens-with-depth trend. Interface is SOUND, not a measurement bug: N=2 top-hidden
alignment is ~+0.49 for both arms and output-layer alignment is high where trained. It sits near +0.5 (not +1.0) for a
principled reason: the delivered credit uses the e-prop eligibility-trace pre-factor + spatial surrogate while true
BPTT uses instantaneous pre-activity + through-time recurrence — genuinely different temporal treatments cap the
weight-gradient cosine below 1 even for perfectly-aligned feedback.

## hier3 secondary probe — not XOR-specific

<!--derived-->
On the obligatory-depth-3 `hier3` task the transport-free local rule (FA AND KP) does not train at N=2 OR N=3 even at
200 epochs (stuck at chance 0.167, k=9), so alignment is invalid there too. In NEITHER task can a valid >=2-point depth
trend be obtained on a trained net. Probe: `research/runners/_gap4_fa_kp_alignment_hier3.py`.

## The SECOND, INDEPENDENT wall — the task itself cannot be built (hier3 tuning, 17 configs, 0 gate)

<!--derived-->
The alignment wall is "the spiking rule won't train deep". There is a SECOND, independent wall behind it: a task whose
depth-3 GENERALIZATION is obligatory (a backprop oracle: depth-2 fails held-out, depth-3 succeeds) could not be built
either. A research-informed 17-config tuning sweep of `hier3` (killing the memorization shortcut: member_id_dim->0, more
classes S=4, narrower hidden, larger held fraction, more members) gated on 0/17 — full table:
`research/findings/raw/gap4/realspikes/verify/hier3_tuning_sweep_17configs.txt`. The default seed-42 stage0
(`research/findings/raw/gap4/realspikes/hier3_stage0_seed42.json`) already showed the trap: l2_train 1.000 (depth-2
MEMORIZES) with l2_te/l3_te both at chance (0.167) — depth-2 memorizes, nothing generalizes. The 17-config sweep then
confirmed the pattern is the wall itself: every lever that makes the depth-3 oracle GENERALIZE (l3_te high) ALSO makes
the depth-2 oracle generalize (l2_te high -> no depth requirement, e.g. S4_mem6_obs10_h32 l2_te 0.89), while every lever
that forces depth-2 to fail (l2_te ~ chance) ALSO starves depth-3 (l3_te ~ chance) — there is no config with l2_te ~
chance AND l3_te >= 0.80. This is the project's
own 2026-07-08 map made empirical ("supervised deep-credit depth-benefit is NARROW — a nonlinear conjunction/binding
resisting the scalar, linear, AND memorization shortcuts"; even the depth-2 instrument is fragile). ⇒ the depth-rescue
test is blocked by BOTH a task-construction wall AND a deep-spiking-training wall — two independent reasons, converging.

## Honest scope + the named next mechanism

<!--derived-->
**Scope guard — the crux CORE is UNTOUCHED:** transport-free deep credit works on a trainable spiking substrate at the
REQUIRED depth-2 (6/6, beats reservoirs on XOR +0.150, matches/exceeds BPTT); the N=2 alignment point (KP +0.435 > FA
+0.255) is consistent with it. What this resolves is the depth-SCALING frontier: the KP-depth-rescue that carried the
rate result is UNTESTABLE on spikes at N>=3 as things stand. **Caveat — the N>=3 non-training is itself confounded:**
on XOR, depth-3+ is REDUNDANT (a depth-2 task), so "doesn't train deep" conflates redundant-depth optimization
difficulty with a fundamental depth wall; on hier3 the local rule can't train the task at all. Neither cleanly isolates
"N>=3 spiking training is a fundamental wall" — but the COMMON thread holds: with this transport-free e-prop-style local
rule, deep (N>=3) spiking nets do not enter the learning regime in the tested regimes.

<!--derived-->
**The wall sits UPSTREAM of the FA-vs-KP feedback question**, and that relocation is the deliverable: a transport-free
local rule must first get the deep spiking net INTO the learning regime for credit-alignment to exist at all. The
e-prop temporal-credit approximation is directly implicated — its eligibility-trace pre-factor already caps alignment
near +0.5 at the trainable N=2, and deeper nets collapse to majority-class output rather than firing informatively. The
productive next levers are about ENTERING the deep-spiking learning regime at obligatory depth (stronger temporal /
e-prop credit, per-layer credit normalization, or a surrogate/initialization that keeps deep hidden layers
informative), AFTER which KP-vs-fixed-FA becomes a measurable second-order question — not before. This supersedes the
"build a depth-3 accuracy task" line (both attempts hit named traps; see
`2026-08-02-gap4-crux-transport-free-rule-...-6seed.md` Updates 2-3): the binding constraint is the deep-spiking
LEARNING REGIME, not the task construction.
