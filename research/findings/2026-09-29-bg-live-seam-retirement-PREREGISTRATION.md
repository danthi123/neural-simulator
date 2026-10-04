---
type: preregistration
status: active
date: 2026-09-29
mechanism: bg-live-seam-retirement
runner: research/runners/host_share_bg_seam.py
lane: A; plan step 3 of docs/plans/2026-09-25-prove-who-owns-the-computation-PLAN.md
---

# Preregistration: retire the host admission, scoring and commit read of the live basal-ganglia seam

**Filed before any gate seed (42/43/44/100/101/102) has run.** Operating point chosen on dev seeds 1-12
(`research/findings/raw/_bg_neural_seam/dev_organ/`). Handoff with the exact commands:
`docs/drafts/selection-seam/HANDOFF.md`. This replaces the withdrawn toy prereg
(`docs/drafts/selection-seam/toy-need-reversal-derisk-DESIGN-NOT-A-PREREG.md`).

## Question

On the live `/api/brain-chat` turn, the basal-ganglia SPEAK / STAY-SILENT seam has three host decisions: the admission
pre-filter (`decide_action`: the race runs only if the host's own scores already favor STAY-SILENT), the host score
(`salience(msg)`, a content-token count, injected as `salience x 600 pA`), and the commit read (first motor pool to 12
spikes, then `break`). The ignition bus's commit is a host argmax over workspace rates plus a threshold
(`norgan_hop`). With the default-off flags `BRAIN_BG_NEURAL_SEAM=1` and `BRAIN_GNW_UNIQUE_IGNITION_READ=1`: is the reply
still correct, is it caused by the neural route, and is the remaining host read a transducer?

## Fixed build (frozen at the commit that adds this file)

- `research/runners/bg_action_selection_production_organ.py`, region `neural-seam`: words rendered to `heard` groups
  (12 neurons per word, 500 pA); `heard -> str_d1_SPEAK` and `hold -> str_d1_STAY_SILENT` (density 0.5, weight 20,
  slow NMDA, gate `bg_value_afferent`); `heard -> hold_inh -| hold`; `hold` tonic 350 pA; per-channel latch (30 RS,
  internal density 0.5 weight 64, NMDA), motor -> latch weight 40, latch FS cross weight 10; drive 300 ms, hold 200 ms,
  read the last 100 ms at 0.02 spikes/neuron/ms. Learned-content fraction 0 (routes wired).
- `research/runners/_gnw_norgan_bus_derisk.py`, region `neural-seam read`: commit iff exactly one slot is ignited.
- `webapp/server.py`: with `BRAIN_BG_NEURAL_SEAM=1` the BG block calls `decide_action_neural` on every turn.
- Scorer: `research/runners/host_share_bg_seam.py` (`score`, `verdict`, `BARS`), which imports lane C's instrument
  `research/runners/host_share.py` for `reply_diff`, `sham_diff`, `classify_seam`, `DECISION_FIELDS`, `BASE_ENV`.
- numpy, stub renderer, no LLM, `PYTHONHASHSEED=0`, `BRAIN_CHAT_SEED=<seed>`, one fresh ChatBrain per arm.

## Turns and arms (per seed)

Turns, one session: `hsb_empty1` "...", `hsb_q` "what does the dog chase", `hsb_empty2` "?", `hsb_hi` "hi".
Arms (definitions in the runner docstring): `prod_a`, `prod_b` (sham), `prod_intervene`, `prod_substitute`, `seam_b`,
`seam_readout`, `seam_lesion_value`.

## Gates (pooled over six seeds x target turns; code: `verdict` in the runner)

Data-bearing gates (each can fail on data):

| gate | measure | bar |
|---|---|---|
| G1 production host owns the reply | F: the reply follows the flipped host decision (prod_intervene vs prod_a), lane C's definition | F >= 0.8 |
| G2a neural seam holds empty turns | fraction of `hsb_empty1/2` events held in `seam_b` | >= 0.75 |
| G2b neural seam never silences a question | fraction of `hsb_q` events held in `seam_b` | = 0 |
| G3 the neural route causes the reply | fraction of target events whose reply changes with `BRAIN_BG_NEURAL_SEAM_LESION=value` | >= 0.3 |
| G4 the body's latch read is a transducer | fraction of `seam_b` neural trials whose committed action changes at latch threshold x0.5 or x2 | <= 0.10 |
| G5 the bus argmax does no work | fraction of `norgan_hop` calls (prod_b + seam_b) where the unique-ignition read equals the argmax read | >= 0.95 |

Preconditions (a failure makes the verdict UNDEFINED, never NO-GO): the sham null (`prod_a` == `prod_b` on the decision
fields and the answer) holds on every target event; every neural-seam trial keeps the membrane in [-150, +100] mV
(checked every step, including reset and washout); every gated measure has at least 6 events.

**Construction guards (not evidence; a failure is a bug to fix before scoring):** `tests/test_bg_neural_seam.py`
(flag-off byte identity of `decide_action` and `norgan_hop` against hashes captured on main 3c2907b85; the server calls
`decide_action_neural` only under the flag; the neural path never reads `salience`; the race runs on content turns).
With the flag on, the host admission and score no longer exist, so their host share after retirement is 0 by
construction and is not reported as a measurement.

**Reported, not gated:** lane C's class for the production seam (`classify_seam(F, S, D)`; expected TRANSDUCED:
host-causal and replaceable); S and D; the fraction of `seam_b` trials in which the production first-to-12 read would
have committed a different action (how much of the commitment the old host read owned); `seam_readout` F; the
one-word turn's hold rate; per-turn latency.

**Verdict:** GO iff all preconditions hold and G1-G5 pass. NO-GO iff preconditions hold and any gate fails.
UNDEFINED otherwise. A GO means: behind the flags, the live seam's admission, score and commit read are neural; the
remaining host read is threshold-invariant; the bus argmax does no work on these turns. It does not mean the value
mapping was learned, that the flags are on by default, or that the one-brain substrate hosts the organ (it is still
its own bridge).

## DEV expectations (organ level, dev seeds 1-12, 20 trials per class; stated before any gate seed)

`research/findings/raw/_bg_neural_seam/dev_organ/`: empty messages commit 0.75-1.00 with P(STAY-SILENT | commit)
0.79-1.00 (mean 0.90); content messages P(STAY-SILENT | commit) 0.00-0.10 (mean 0.02), so G2b (0 of 6 question turns
held) has roughly a 10% chance to fail on noise alone; value lesion P(STAY-SILENT | commit) 0.44-0.54 in every class;
latch lesion and direct-path lesion: no commit; arousal lesion: empty 0.00-0.05, content 0.10-0.75 commit (SPEAK);
latch threshold x0.5 / x2 changes the committed action on 0.4-4.6% of trials; the production first-to-12 read would
change it on 11-41%. Membrane range over every step: [-113, +72] mV.

## Run and score (verbatim; the pool runs one seed per job)

```bash
cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 8 && bash tools/memcap.sh 12 -- env PYTHONHASHSEED=0 \
  .venv/bin/python -u -m research.runners.host_share_bg_seam --run-all research/findings/raw/_bg_live_seam/gate \
  --seed <s> --allow-gate-seeds
.venv/bin/python -m research.runners.host_share_bg_seam --score research/findings/raw/_bg_live_seam/gate \
  --seeds 42 43 44 100 101 102
```

(`--out` is optional and defaults to stdout-only; the persisted-artifact invocation, with an `--out` flag naming
the score file under the gate directory, is HANDOFF Step 6 in `docs/drafts/selection-seam/HANDOFF.md` -- not
duplicated here so this prereg does not cite, before any run, an artifact path that cannot exist yet.)
