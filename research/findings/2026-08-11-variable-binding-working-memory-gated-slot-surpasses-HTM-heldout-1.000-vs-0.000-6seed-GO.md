---
type: finding
status: go
date: 2026-08-11
mechanism: VARIABLE-BINDING WORKING MEMORY — a BG-gated slow-NMDA bistable HOLD slot whose content is a content-agnostic Hebbian fast-weight role->filler bind; latches a variable and carries it invariantly across NOVEL intervening tokens
lane: emergence engine / working memory (the mechanism the emergence + Gate-B + continual lanes converged on)
verdict: 6-SEED GO — held-out agreement 1.000 (novel fillers, up to L=4) vs the HTM emergence-engine's 0.004; the MEMORY composition is spiking + load-bearing. Residual (precisely named): the gate's TIMING is a host marker (the LEARNED, spiking, ROLE-based gate is the open rung), the REINFORCE math + fast-weight bind spiking-realization are host.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_var_bind_gated_slot_derisk.py
artifacts:
  - research/findings/raw/_var_bind_gated_slot/gated_slot_6seed.json
instrument: composition of THREE banked GOs (each re-verified by reading the finding+runner this session) — HOLD = the D3 slow-NMDA bistable persistent slot (`build_persistent_slot`; holds 1.000/6 with input identically zero); BIND = the RUNG6c content-agnostic Hebbian fast-weight binder (`HebbianBinder`; 0 collisions on held-out entities minted at test); WRITE-GATE = a BG Go/NoGo cascade opening a clear-then-load write. Task = the SAME agreement stream the HTM-TM failed (`_emerge_stream_language_derisk` generator): [subject]+[L varying fillers]+[verb], verb agrees L+1 back, scored on HELD-OUT novel filler paths. SIM_BACKEND=numpy; NO sim/ edit.
---

# Variable-binding working memory (gated bistable slot + fast-weight bind) SURPASSES the emergence-engine failure — held-out 1.000 vs the HTM's 0.004 (6-seed GO); the learned ROLE-based spiking gate is the precisely-named residual

Three lanes converged (emergence-stream, Gate B, continual) on the SAME missing mechanism: a latent-variable / variable-
binding working memory — a gated persistent-activity slot that latches a variable and holds it across a span so
downstream computation generalises to novel intervening tokens. A 5-modality deep-research sweep (local RAG + biology
catalog/Kandel + external biology + external ML + our substrate) designed it as a COMPOSITION of three of our own banked
GOs. This de-risk builds + 6-seed-tests it, on the exact task the HTM emergence engine scored 0.000 on.

## Result — 6-seed GO (`research/findings/raw/_var_bind_gated_slot/gated_slot_6seed.json`)

<!--derived-->
At n_fill=6, L=4 (distance 5, 1296 paths, held-out NOVEL fillers): the gated-slot WM scores **held-out branch(verb)
accuracy 1.000**, versus the **HTM emergence engine 0.004** (its memorise-not-generalise baseline, reproduced in-runner
on the identical stream), the best fixed-order n-gram floor 0.273, and chance 0.250. `attributable_to` the mechanism
over HTM: **+0.996**; over the n-gram floor: +0.727. GO at 6 seeds.

**All anti-cheat teeth bite** (this is the surpass, not a scaffold artifact):
- **LESION-the-hold** (kill the recurrence) → 0.074 (hold-alive 0.0003) — the spiking slow-NMDA slot is LOAD-BEARING; it
  is genuinely holding the binding, not a host store.
- **ALWAYS-OPEN / shuffled gate** → 0.000 — the GATE protects the latch from intervening fillers (the piece the HTM
  lacked); the attractor's own overwrite-resistance is not what carries it.
- **SLOT-SCRAMBLE / permuted-binding** → 0.100 — the BIND is load-bearing (content, not capacity).
- **REFERENT-SHUFFLE** → 0.000 — no topic→answer leakage.
- **HOLD-NOT-RE-READ** → external input ASSERTED zero across the span (hold-alive 0.0949) — the slot SUSTAINS, it does
  not re-read a host store per step (as D3 does).

## Scope / honesty — what is genuinely solved+spiking vs the named residual ladder (per brain-based-only)

<!--derived-->
NO-EXTERNAL-NEEDED: grounded in our OWN verified GOs (D3 hold, RUNG6c bind, BG gate), each re-read at source this
session; the external biology (PBWM/Wang/Mongillo) is corroborating context, not load-bearing, and was NOT relied on.

- **SOLVED + spiking + load-bearing:** the MEMORY composition — the slow-NMDA bistable HOLD (spiking, lesion-load-bearing),
  the clear-then-load write the gate triggers (input alone cannot overwrite the slot, per D3), and the content-agnostic
  bind that generalises to novel fillers. This is the direct de-shortcutting of the earlier RUNG2 (which hit ~0.971
  functionally but with a host doc-marker gate + fixed bijection) in the MEMORY dimension.
- **THE RESIDUAL, precisely named (the honest a-1 boundary):**
  1. **The gate's TIMING is a host MARKER** (it opens on the subject token). A REINFORCE write-gate trained ONLY on the
     verb-prediction reward (three-factor, no token-type label) DID learn to fire LOAD on the subject (held-out
     precision/recall 1.00) — but on a stream where the subject is a BARCODE-SEPARABLE CLASS, so it learned a token-CLASS
     boundary, **not syntactic ROLE**. In real language the same token is subject-or-not by position/syntax; that
     ROLE-based gate is UNTESTED and is the genuine next problem.
  2. The REINFORCE update math is HOST → the on-substrate spiking three-factor DA-gated gate is the next rung (gap#4
     distal-credit territory).
  3. The fast-weight BIND is host numpy (its spiking-STP realisation is a banked next rung); the verb read-out is a host
     deref of the held pool.
- **Named next build (dependency-ordered):** (a) the ROLE-based gate (subject-by-syntax, not barcode-class); then (b) the
  on-substrate spiking DA-gated gate; then (c) the spiking bind; then (d) wire the composition into the emergence stream.
  Reuse-by-import; NO `sim/` edit in this de-risk.
