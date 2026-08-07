---
type: finding
status: contributing
date: 2026-08-07
mechanism: grounded-message-to-word
runner: research/runners/_grounded_message_to_word_onbridge_derisk.py
artifacts:
  - research/findings/raw/grounded_message_to_word/onbridge_3seed_confirm.json
  - research/findings/raw/grounded_message_to_word/onbridge_smoke.json
---

# Grounded message-to-word ON THE SHARED SPIKING BRIDGE: the naming map runs on real Izhikevich neurons (CPU de-risk, 3-seed GO)

<!--derived-->
**One-line verdict.** The referent-naming map that the 2026-08-07 rate de-risk realized as a numpy matrix
(`argmax(W @ x)`) now runs as a PLASTIC synaptic pathway between exc pools on a real `SimulationBridge`:
percept-assembly neurons drive word pools through weights learned by the bridge's own on-bridge rate-Hebbian
rule, and the referent word is decoded from WORD-POOL SPIKE COUNTS — not a host matmul. This burns down the
"rate-proxy" residual the rate finding named. Three-seed CPU (`SIM_BACKEND=numpy`, real spiking Izhikevich):
brain-native naming accuracy 0.929 (per-seed 0.887–0.963) under stressing percept noise versus an untrained
random-map control 0.221, so 76% of the accuracy is attributable to the learned weights. Every anti-cheat
holds on spikes: render-faithful 1.0, the spiking request/silence gate routes sated → silence → zero word
output on every seed, lesion collapses to chance with no confident decode, a permuted map follows the
permutation (0.921) and rejects the original (0.025), and a novel percept abstains on every seed. This matches
the rate map (spiking 0.929 vs rate 0.952) on real neurons. Artifact:
`research/findings/raw/grounded_message_to_word/onbridge_3seed_confirm.json`.

seed-waiver: this is a 3-seed CPU de-risk confirmation of a newly-built mechanism; the 6-seed generalization
run is explicitly delegated to the parent session by the driving task (do-not-run-the-full-6-seed here), and
the naming PATH it realizes is already 6-seed GO in the rate finding it burns down. Command for the parent:
`--seeds 42 43 44 100 101 102`.

## Re-anchor: what was rate, what is now spiking

The 2026-08-07 rate de-risk
(`2026-08-07-grounded-message-to-word-brain-native-naming-replaces-host-decoder.md`, GO 6/6) established the
brain-native naming PATH — a gated local-Hebbian map, no weight transport, true label never on the inference
read path — but two pieces were host/rate proxies, and the finding named both as residuals: *"the rate-proxy
gate (spiking form already GO)"* and *"Run the naming map on the shared spiking bridge with the spiking
request/silence gate."* The request/silence **gate** was already GO in spiking form
(`2026-08-03-grounded-speech-action-loop-6seed-GO`), so the genuinely un-built step was the **naming map** on
the spiking substrate. That is this rung.

## Mechanism (what became neural)

- **Substrate.** One `SimulationBridge`: a `percept` exc region (240 neurons, the sparse 24-of-240 assemblies),
  a `word` exc region (4 word pools × 48 neurons), and the request/silence/gate_fs pools of the spiking gate.
- **Naming map (brain).** A plastic `percept → word` pathway, zero-ish init, learned by the bridge's on-bridge
  **rate-Hebbian** rule — the matched rule for a SYMMETRIC referent↔word coincidence; STDP is measured-negative
  on symmetric co-firing (`2026-06-15` on-bridge co-occurrence finding, `Δt≈0 → 0` weight change). Plasticity
  is GATED (a `plasticity_gate` open only during the teacher naming event); teaching co-drives the assembly and
  the teacher-selected word pool, and the learned diagonal weights saturate near `hebbian_max_weight` while
  off-diagonal stays ~0.
- **Decode (brain, on spikes).** At inference the plasticity gate is closed; driving ONLY the percept assembly
  propagates through the learned synapses and the referent is the argmax of WORD-POOL SPIKE COUNTS. Off-target
  pools receive ~0 learned input so they never cross threshold; a wider word population gives the winning pool
  the SNR to survive percept noise while the wrong pools stay at zero (the CYCLE-91 population-code lift).
- **Gate (brain, on spikes).** A request-vs-silence spike-count race with a shared FS inhibitor on the same
  bridge routes whether the naming circuit is engaged.
- **Articulation (body, host).** Each word pool has a fixed binding to one WKV vocab token; the numpy WKV
  forward (grounded ft checkpoint, RF-spiking-forward parity GO) renders the spike-decoded word.

## Three-seed result

<!--derived-->
Seeds 42/43/44, one CPU process (`SIM_BACKEND=numpy`, real spiking Izhikevich):

<!--derived-->
| check | result |
|---|---:|
| spiking naming accuracy (noisy percept) | 0.929 mean, per-seed 0.938 / 0.887 / 0.963 (chance 0.25) |
| untrained random-map control accuracy | 0.221 mean |
| naming accuracy attributable to learned weights | 76% |
| render-faithful (spoken word == spike-decoded referent) | 1.0 every seed |
| spiking gate: hungry → request / sated → silence → zero word output | True every seed |
| lesion of the plastic pathway → accuracy / confident decodes | 0.25 / 0 confident (fails safe to silence) |
| permutation followed / original accepted after permute | 0.921 / 0.025 |
| novel untaught percept abstains | True every seed |

Smoke (seed 42, 15 trials) reproduced the same GO: naming 0.967 vs chance 0.200, all controls moving.

Exact command (this 3-seed confirmation):

```bash
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_message_to_word_onbridge_derisk \
  --seeds 42 43 44 --n-trials 20 \
  --out research/findings/raw/grounded_message_to_word/onbridge_3seed_confirm.json
```

## What the controls establish (on spikes)

<!--derived-->
- **Learned, not a substrate bias.** An untrained random `percept→word` map decodes at chance (0.221); 76% of
  the trained accuracy is attributable to the learned weights.
- **Learned, not wired.** Teaching a permuted referent→word map decodes the permutation (0.921) and rejects the
  original word (0.025) on every seed.
- **Fails safe.** Zeroing the plastic pathway collapses accuracy to chance and drops every clean decode below
  the confidence floor — the brain declines to name rather than emitting a confident wrong word (a taught pool
  fires ≥20 spikes, a lesioned/novel percept only a ≤2-spike flicker).
- **Specific.** An untaught percept assembly stays below the confidence floor on every seed.
- **Gate-first, mirroring the moat.** A sated trial routes to silence on the spiking race and the word output is
  reached zero times; a hungry trial engages it exactly once.

## Honest scope and boundary

<!--derived-->
The spiking naming accuracy (0.929) sits just below the rate map (0.952) — a small spiking-substrate cost, not
a gap in the mechanism: it is driven by the weak, adaptation/STP-capped word-pool firing (a taught pool fires
~20–40 spikes over the decode window), recovered to a robust operating point by the population code (word pool
of 48). **Disabled for this de-risk (declared, not hidden):** structural plasticity, threshold homeostasis and
synaptic scaling are frozen around the naming map so the learned pathway is stable across the long inference
battery (mirrors the Aug-03 loop's inference isolation). Still scaffold, named as such: the carrier frame
`the <agent> <verb> ___` is host phrasing; the percept assemblies are deterministic rather than emerged from
vision; the gate ROUTES here (its full grounded spiking form is GO in the Aug-03 loop); the WKV cortex is
conventionally trained (used off-bridge as the fixed articulatory scaffold). A tried-and-rejected lever: a
word-pool WTA inhibitor was inert at these low firing rates (the pools fire too sparsely to drive the shared
inhibitor) and was removed rather than claimed.

## Next mechanism

1. Six-seed generalization (parent): `--seeds 42 43 44 100 101 102`.
2. Drive the percept assemblies from the neural retina/visual features rather than fixed codes.
3. Run the on-bridge WKV articulation (currently the numpy forward) on the shared bridge, closing the last
   off-bridge seam.
4. Learn the intent (request/comment/ask) from contingent outcomes so the whole message is brain-selected.

## ✅ PARENT-VERIFIED (6-seed) — GO
<!--derived-->
The parent ran the full 6-seed (42/43/44/100/101/102, real Izhikevich, numpy; artifact
`research/findings/raw/grounded_message_to_word/message_to_word_onbridge_6seed.json`, verdict GO, 964.6s).
Spiking naming accuracy 0.85 (mean) vs chance 0.25 / untrained random-map 0.243 (learned, separation 0.61);
lesion collapses to chance with 0 confident decodes (fails safe); the spike-decoded word renders (1.0); the
on-bridge spiking gate routes hungry->request and sated->silence every seed; permutation-followed 0.85 /
original-rejected 0.06; novel percept abstains. All preconditions pass. The 0.85 mean is modestly below the
3-seed 0.929 but clears the >=0.8 bar with every anti-cheat intact. VERDICT: GO — the brain-native referent
naming map holds on REAL SPIKES, decoded from word-pool spike counts (no host matmul), no weight transport, the
true label never on the inference read path. The rate-proxy naming residual is burned down; the word the brain
produces is now selected by a plastic on-bridge spiking pathway.
