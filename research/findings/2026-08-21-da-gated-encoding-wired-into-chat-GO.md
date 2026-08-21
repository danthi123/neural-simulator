---
type: finding
status: live
date: 2026-08-21
mechanism: da-gated-encoding-wired-into-chat-store
lane: integration
integration_faculty: da-gated-encoding
seeds: [42]
seed-waiver: A WIRE-IN verify of a deterministic config coupling (does the live self-produced DA reach the composer's
  store hook and produce a differential write gain; is OFF a no-op; does the lesion sever it) — a plumbing/attribution
  proof, not a stochastic effect size. The gain map + its magnitude mechanism are the 3-seed / 6-seed GO de-risks it
  reuses (_burndown_I7_dopamine_encoding_deploy_derisk, _phaseB_dopamine_encoding_gain_derisk); the per-fact RF codes at
  seed 42 are representative. The DA levels are the real spiking SNc read, not a host constant.
verdict: GO
runner: research/runners/_da_encoding_wired_verify.py
instrument: through the REAL production webapp/server.py::brain_chat handler (numpy-CPU, rf recall), teach an SVO under
  three regimes and read the write gain off the response's da_encoding trace + the live composer's encoding_gain_fn.
  (A) BRAIN_DA_ENCODING unset -> no da_encoding key + encoding_gain_fn is None (byte-identical store). (B) enabled, the
  same fact taught under a HIGH-DA turn (BRAIN_DA_DRIVES_INDUCE=1300) vs a LOW-DA turn (=100): g_high > g_low, and that
  same g writes a measurably stronger trace on a magnitude-carrying store (stored |w| ratio == g_high/g_low). (C)
  BRAIN_DA_ENCODING_LESION=1 pins g=1.0 regardless of DA -> the differential vanishes.
artifact: research/findings/raw/_da_encoding_wired/verify.json
external: NO-EXTERNAL-NEEDED — reuses the in-repo validated I-7-b gain map + the #76/#79 spiking DA-mode read.
---

# DA-gated encoding wired into the live chat store — the self-produced dopamine scales fact-encoding strength (WIRE-IN GO, default-OFF)

**Verdict: GO (wire-in, default-OFF — NOT flipped on).** The brain's OWN self-produced tonic dopamine now scales the
WRITE MAGNITUDE of a taught fact at store time (Lisman-Grace hippocampal-VTA loop; Kandel D.16 — dopamine gates entry
into LONG-TERM memory: a fact heard while the SNc bursts is encoded stronger and stays more stable). This is the WAVE-0
write-side Gap-4 coupling — the counterpart to the #76/#79 DA-mode READ-side (which drives the reply's forthcomingness):
the same self-produced tonic DA that colors HOW the brain answers now also sets HOW STRONGLY it records what it is told.

## What was built

- `webapp/da_encoding_drives_chat.py` — the production glue. `da_encoding_enabled()` (flag `BRAIN_DA_ENCODING`,
  default-OFF), `da_encoding_lesioned()` (`BRAIN_DA_ENCODING_LESION`), and `install_encoding_gain(chat)` which sets
  `chat.inner.composer.encoding_gain_fn` to a closure reading the live self-produced DA. The gain map is reused verbatim
  from the VALIDATED board I-7-b de-risk: `g = clip(0.5, 3.0, 1 + 2.0*(DA - 0.5))` (tonic 0.5 -> g = 1.0; k_DA = 2.0).
- `webapp/server.py::brain_chat` — installs the gain right after the DA-mode read (`chat._last_da_drives["da_level"]`
  fresh) and before the gate/acquire that stores, so a fact taught while engaged is encoded stronger. Additive; when
  `BRAIN_DA_ENCODING` is unset the block is skipped, `encoding_gain_fn` stays None, and no `da_encoding` key is attached
  -> the store + response are byte-identical to HEAD.

## The verify (through the real handler; numpy-CPU, rf recall)

`research/runners/_da_encoding_wired_verify.py` — a `tools.verdict.Verdict`, teaching `dog eat grass` under three
regimes on FRESH sessions through `webapp.server.brain_chat`:

<!--derived-->
_(values below are rounded from the cited `research/findings/raw/_da_encoding_wired/verify.json`; exact values there.)_

| proof | result |
| --- | --- |
| (A) OFF byte-identical | no `da_encoding` key; live `encoding_gain_fn is None`; recall `grass` (correct) |
| (B) ON, load-bearing | g_high = 2.477 (self-produced DA 1.239, INDUCE 1300 pA) > g_low = 0.500 (DA 0.046, INDUCE 100 pA) |
| (B) mechanism (magnitude store) | stored \|w\| ratio 4.955 == g_high/g_low 4.955 (the gain scales the real stored trace) |
| (C) lesion severs | g_high = g_low = 1.000 (the differential vanishes; attribution to the DA read) |

GO = OFF byte-identical AND on-load-bearing (g_high > g_low) AND lesion-severs. The DA levels are the real spiking SNc
read (self-produced off the neuromodulator bus), not a host constant; the low-DA turn's gain hits the g_min floor (0.5).

## Honest scope (NOT flipped on)

This is a WIRED, default-OFF coupling — the flip to default-on is a separate step (a soak / no-regression gate). The
WRITE gain the live DA drives bites the STORED trace only on a MAGNITUDE-carrying composer — the production-default
`OneBrainComposer` (`store_conns`) and the RF substrate store (proven in the mechanism sub-check + the I-7-b GO). On the
`BRAIN_COMPOSER_KIND=rf` numpy FAST-path recall used here for the handler's speed the stored recall is
magnitude-INVARIANT (phases only), so the coupling is a write-side reserve on THAT store — the handler proof (A/B/C) is
the WIRING (the live DA reaches the store hook and produces a differential write gain), and the mechanism sub-check
confirms the gain scales a real magnitude store. The message->engagement->SNc-afferent scalar is the same host
sensory/comprehension boundary the #79 DA-mode read names as its residual; the DA LEVEL is the brain's own spiking SNc
read (lesion-proven: pinning the gain severs the differential even though the level still varies).

## Next rung

Flip default-on after a no-regression soak on the magnitude-carrying production-default composer (onebrain), where the
stronger trace is behaviorally visible (recall survival under read stress — the I-7-b behavioral knee).
