---
type: finding
status: scoping
date: 2026-08-28
mechanism: mouth-crutch-burndown-scope
lane: e-mouth-fluency / A1
verdict: SCOPE-ONLY (bounded, per instruction). Maps where Qwen actually sits in the LIVE generation pipeline, verifies the existing "mouth de-Qwen audit" claim, and names the smallest concrete next de-Qwen rung the 2026-08-28 fullscale-confirmation readout unlocks. One cheap CPU smoke re-confirms the surrounding WKV spiking-generation pipeline is live today.
artifacts:
  - research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md
  - research/findings/2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md
  - GAP_CLOSURE_MISSION.md (line 60, "Mouth de-Qwen AUDIT done")
  - webapp/server.py
  - research/runners/_grounded_lang_integration_derisk.py (SpikingQwenFaculty)
  - research/runners/_wkv_fewspike_read_derisk.py
  - research/findings/raw/_mouth_crutch_burndown_scope/fewspike_smoke.json
---

# Mouth crutch-burndown — scope, not build: where Qwen sits, what the 0.87 readout does and does not yet touch, and the smallest concrete next de-Qwen rung

## 0. What this doc is

A bounded scoping pass, requested to answer three questions before any build: (1) has the "mouth de-Qwen audit"
already claimed in the board actually been done, and what did it find; (2) exactly where does Qwen sit in the LIVE
`/api/brain-chat` generation pipeline today; (3) given the 2026-08-28 full-scale confirmation that a locally-learned
spiking readout recovers <!--derived--> `sub_recov_ratio_mean = 0.8686` of a previously-copied read-out head (per the
already-cited fullscale-confirmation finding), what is the smallest
concrete next rung toward reducing Qwen's role in actual word-generation. No `sim/` edit; no production wiring
built. One small CPU/numpy smoke re-run for supporting evidence (below).

## 1. The audit already exists — verified, not re-derived

`GAP_CLOSURE_MISSION.md:60` already records: **"Mouth de-Qwen AUDIT done: the bounded-SVO touchpoints already
bypass Qwen (spiking-mouth-recall); read-SNR ensemble de-risk BUILT + staged (branch
research/mouth-read-snr-ensemble-dendritic-derisk)."** This is accurate and current — verified against the actual
wiring below, not just re-quoted. Two prior threads sit behind it, both real and both landed:

- **The spiking-Broca RECALL mouth is production default-ON as of `75a3a96ee` (2026-08-26 wave-3 flip).**
  `research/runners/spiking_mouth_recall_prod.py::_RECALL_MOUTH_DEFAULT_ON = True`. For the bounded
  transitive-SVO frame inventory, a grounded recalled fact is now rendered "the S V-3sg the O" **on firing
  neurons** (per-pool spiking-rate word-order, EMERGE-59/61 lineage), re-parse VERIFIED against the recalled
  triple — Qwen is bypassed for that slice. See
  [`2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md`](2026-08-26-spiking-broca-mouth-recall-surface-production-wirein-GO.md).
  Scope-guard (honest, unchanged): only single-word-role, non-copula, non-irregular transitive facts route on
  spikes (15/17 of the bounded smoke's stored facts); irregular verbs, copula/attribute facts, and open/multi-word
  prose fall straight through to the current mouth (Qwen or the template stub), byte-identical, never a leak.
- **The mouth read-SNR arc (gap#4 read regime) is a SEPARATE, still-open residual on the WKV read-out's own
  learning fidelity**, staged on branch `research/mouth-read-snr-ensemble-dendritic-derisk` /
  `research/mouth-read-snr-dendritic`: the ensemble (`--sub-pop`) lever is an honest NO-GO (the graded read is
  P-invariant by construction — pool replicas share one noisy hidden population, so population-averaging removes
  zero common-mode noise); a dendritic (Urbanczik-Senn two-compartment) lever is built, byte-identical when off,
  smoke-clean, and staged for a 6-seed decisive run
  ([`2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md`](2026-08-27-mouth-read-snr-ensemble-verdict-and-dendritic-lever.md)).
  **This residual predates, and is now substantially smaller than expected after, the 2026-08-27/28 stale-cache
  training fix** — that finding's own plateau numbers (`sub_learned_recov_mean` ~0.34-0.37) are from BEFORE the
  fix; the fixed training reaches 0.85 mean / 0.84 min at full scale. The dendritic lever may still be the right
  next move to close the residual `go_count 3/6 -> 6/6` gap, but it is a smaller, already-scoped, already-staged
  problem — not blocking the rung below.

## 2. Terminology correction (load-bearing): "Qwen" names TWO different things in this arc

Reading the 2026-08-14 -> 2026-08-28 mouth-readout finding lineage, the phrase "the read-out head weights were
Qwen's — LOADED, not LEARNED" is **not literal**. Tracing the actual checkpoint
(`bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz`, loaded by `research/runners/_wkv_fewspike_read_derisk.py::WKVReadout`
and by `_wkv_mouth_readout_eprop_batched_substrate_derisk.py`) shows a **small, from-scratch, home-grown recurrent
SSM/RWKV-style language model** (vocab V=1000, hidden D=128, its own embeddings/`Wv`/`Wr`/`Wo_sp`/decay-store/
`head.weight`/`head.bias`, trained on TinyStories) — architecturally unrelated to Qwen2.5-0.5B (vocab ~151936,
hidden 896). No distillation-from-Qwen or teacher-Qwen reference exists anywhere in this line's training code.
So there are genuinely two separate "mouth" tracks, both filed under the roadmap's "A1 — retire the Qwen mouth"
banner (`docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md:351`):

1. **`SpikingQwenFaculty`** (`research/runners/_grounded_lang_integration_derisk.py:168`) — the LITERAL Qwen2.5-0.5B-Instruct,
   loaded via `AutoModelForCausalLM.from_pretrained`, with its internal SiLU/softmax/div nonlinearities converted
   to spiking pool computations at install time. **Every matrix (embeddings, attention Q/K/V/O, MLP, LM head)
   remains Qwen's own pretrained parameter.** This is what is actually wired live in `webapp/server.py`
   (`_get_warm_qwen_renderer()`, the `qwen` renderer) — see §3.
2. **The WKV/SSM home-grown cortex** — the `wkv_ssmU*` checkpoints and the whole `research/runners/_wkv_*` /
   `_emerge_wkv_onbridge_derisk.py` lineage. This is the architecture the roadmap actually intends as the Qwen
   REPLACEMENT for open generation (progressively biologized: recurrent store -> diagonal e-prop local-rule GO,
   2026-08-12; input encode -> RF-phase spiking delivery GO, 2026-07-20; read-out head -> now locally e-prop-learned
   instead of copied, <!--derived--> 0.8686 ratio, 2026-08-28). **This is where the "0.87 readout" result actually lives — and it
   has never been wired into `webapp/server.py`.** The "Qwen's" language in the 2026-08-14 finding is loose
   shorthand for "the externally-scaffolded / host-backprop-copied head," carried over from the generic
   "scaffold-to-retire" framing, not a literal claim about Qwen2.5's weights.

This distinction matters for the crutch-burndown: the 0.87 result does not yet reduce ANY call into the actual
Qwen2.5-0.5B model. It makes a *different*, purpose-built, from-scratch spiking-native generator's own final layer
more legitimate (locally-learned rather than host-copied) — which is the precondition for wiring THAT generator in
as a genuine Qwen-reduction lever, not itself the reduction.

## 3. Exact map — where Qwen sits in the LIVE `/api/brain-chat` pipeline today

Verified directly in `webapp/server.py` (renderer selection at line ~3634-3734, `_default_brain_renderer` at
~3420, open-ended block at ~4504-4535):

- **Renderer options are exactly `raw` / `qwen` / `stub`** (`rname ==` branches at server.py:3728/3730; no `wkv`
  option exists). `_default_brain_renderer()` picks `qwen` only when `SIM_BACKEND=cupy` AND a CUDA GPU is present
  (else the GPU-free template `stub`); `BRAIN_CHAT_RENDERER` can force any of the three.
- **Touchpoint A — grounded recall / rich-answer surface rendering.** `ChatBrain.render` (ChatBrain in
  `brain_chat_tui.py`) and `RichAnswerComposer._render_one_verified` both now try the spiking-Broca mouth FIRST
  (default-ON, §1); Qwen (`_get_warm_qwen_renderer()`) or the template stub is the byte-identical FALLBACK for
  anything the bounded transitive-SVO frame can't cover — irregular verbs, copula/attribute facts, multi-word
  roles, and all open/general prose. This is the majority of what "Qwen" still does in a normal grounded-fact
  turn: render the residual the spiking mouth's scope-guard declines.
  Note (worth flagging, non-blocking): `webapp/server.py:4522`'s comment says the open-ended block reuses "the ONE
  warm Qwen faculty the server already loaded for the qwen renderer" — accurate for the SpikingQwenFaculty
  singleton pattern; unrelated to the WKV track.
- **Touchpoint B — open-ended free generation** (`BRAIN_OPEN_ENDED`, **default OFF**, server.py:4504-4535): when a
  turn's topic isn't gate-verifiable, this branch builds a `StateContext` from the live affect read + grounded
  familiarity, generates FREELY through the SAME warm `SpikingQwenFaculty` (`_get_warm_qwen_renderer()._fac`), then
  runs a VERIFY post-filter (the honesty moat: reframed per the 2026-08-19 owner directive as "Qwen = a FORM
  scaffold, honesty = STATE-fidelity"). This is the ONLY place Qwen supplies genuinely free (non-templated,
  non-SVO-gated) prose, and it is currently the SOLE generator available to that channel — no fallback, no
  alternative.
- **The WKV mouth touches NEITHER touchpoint today.** Its production-shaped prototypes (`OnBridgeWKVFaculty`,
  `FluidChat(renderer="wkv")`, referenced in `GAP_CLOSURE_MISSION.md` around 2026-07-20) were standalone research
  REPLs / de-risk harnesses, never carried into `webapp/server.py`'s renderer selection.

## 4. The concrete next de-Qwen rung the 0.87 readout unlocks

**Wire the WKV mouth into `webapp/server.py` as a bounded, additive, default-OFF alternate generator for
Touchpoint B (`BRAIN_OPEN_ENDED`) — not Touchpoint A.** Concretely: a new renderer path (e.g.
`BRAIN_OPEN_ENDED_GENERATOR=wkv`, mirroring the existing `BRAIN_CHAT_RENDERER` override pattern) that, when the
open-ended block fires, calls the WKV mouth's few-spike generation
(`research/runners/_wkv_fewspike_read_derisk.py::FewSpikeWordRead` + `WKVReadout`, swapping in the newly
e-prop-learned `W_hat` from the stale-cache-fix artifact in place of the checkpoint's copied `head.weight`) instead
of `SpikingQwenFaculty`, through the SAME VERIFY post-filter the Qwen path already runs.

**Why this is the right first step, and why now specifically:**
- It targets the ONE touchpoint that is (a) already default-OFF in production — zero live-traffic risk to land
  and A/B — and (b) already architected around a generate-then-verify honesty gate, which is exactly the
  gate-first pattern the WKV mouth's own `grounded_reply`/CONSTRAIN-VERIFY loop already uses, so the wiring is a
  swap of the generator, not a new safety mechanism.
- It is the ONLY touchpoint where Qwen currently has no fallback or competitor — Touchpoint A already has one
  (the spiking-Broca mouth, default-ON); putting the WKV mouth at Touchpoint B is the marginal place a new
  generator actually displaces Qwen calls rather than adding a third redundant path.
- The 2026-08-28 result is what makes this a legitimate integration step rather than premature wiring: before the
  stale-cache fix, the WKV mouth's OWN final decision layer was itself a host-copied weight matrix — wiring it in
  would have swapped one shortcut (Qwen's full transformer) for a smaller but still-real one (a BPTT-transported,
  not locally-learned, read-out). Now the layer that decides which word to emit is trained by a local three-factor
  rule reading its own error off the real spiking substrate (`host_matmul_on_forward == 0`, no weight transport
  asserted and anti-cheat-checked) — closing exactly the residual the brain-based-only standard flags.
- The existing infrastructure survives contact: §5 below re-confirms the surrounding few-spike generation pipeline
  runs cleanly today, unmodified, on CPU.

**Honest residual, stated precisely (do not overclaim it away):**
1. **Vocabulary mismatch.** The WKV mouth's checkpoint vocab is V=1000 TinyStories-domain words; the production
   brain's actual chat vocabulary/topics are broader. A first wire-in would be scoped to turns whose topic falls
   inside (or near) that vocabulary — likely most naturally the self-initiated / narrative-style utterances rather
   than arbitrary open Q&A, which is a real scope narrowing versus what Qwen currently covers unconditionally.
2. **The read-out layer is the ONLY component newly verified as locally-learned.** Whether the SAME checkpoint's
   recurrent-store weights (`Wv`/`Wr`/`Wo_sp`/decay/embeddings) were trained by the diagonal-e-prop local rule
   (2026-08-12 GO) or by the earlier host-BPTT pass — the finding lineage documents BOTH methods existing for this
   architecture family, at different checkpoint generations (`wkv_ssmU_...` vs `wkv_ssmU6_...`) — is **not verified
   in this pass**. This is a flagged open provenance check, not a claim either way; resolving it (read the
   checkpoint-producing runner's own training loop, or re-derive under `--feature substrate` end-to-end) should be
   the FIRST step of any actual wiring build, before writing the webapp integration code.
3. **The `go_count 3/6` marginal residual** (§1) means ~half the seeds of the learned readout sit just under the
   strict per-seed bar; a production wire-in should either wait for that to close (candidate levers already named
   in the 2026-08-28 finding: more epochs, per-seed lr, or the staged dendritic lever) or accept the
   ratio-0.8686<!--derived--> mean as the honest current ceiling and gate the wire-in's own soak on it explicitly.
4. This rung reduces Qwen calls at ONE currently-off touchpoint. It does not touch Touchpoint A's residual (the
   spiking-Broca mouth's own irregular-verb / copula / open-prose fallback), which remains Qwen's largest live
   share of actual production render calls today and is a separate, already-tracked burn-down (grow the frame
   inventory / verb coverage of the spiking-Broca mouth itself).

## 5. Supporting de-risk (bounded, CPU/numpy, detached, ~3s)

To avoid recommending a wire-in onto infrastructure that might be bit-rotted, re-ran the WKV mouth's few-spike
generation smoke fresh, unmodified, on CPU:

```
SIM_BACKEND=numpy python -m research.runners._wkv_fewspike_read_derisk --smoke --seed 42 \
  --json research/findings/raw/_mouth_crutch_burndown_scope/fewspike_smoke.json
```

Result (artifact: `research/findings/raw/_mouth_crutch_burndown_scope/fewspike_smoke.json`, provenance sidecar
auto-recorded — a throwaway sanity check, not a claimed finding): the checkpoint loads, drives a real Izhikevich
`SimulationBridge` (64-1024 neurons across the swept operating points), and free-generates coherent TinyStories-domain
continuations via the genuine few-spike spiking WTA word-decode (NOT host argmax) — e.g. seed 42, read-window 40,
P=8: *"once upon a time there was a little boy named tim saw a big red ball that he loved to play with his toys and
his friends were very happy..."* <!--derived--> (self-NLL 0.819, close to the host-sample ceiling; quoted from the
run log, not a JSON field). GO clears at P=8 and P=16 operating points (`read_fidelity_vs_sampler` 1.26-1.46, `argmax_agree`
0.77-0.95, both in the cited JSON's `results[]`); anti-cheats hold (`mass_scramble` ~0.02, `noise_ablation_deterministic
== true`, `host_rng_draws_on_read_path == 0`). This uses the
checkpoint's ORIGINAL copied `head.weight` (not the newly e-prop-learned `W_hat` — swapping that in is part of the
actual wiring build, out of scope here), so it does not test the 0.87 result directly; it confirms the surrounding
generation pipeline the learned readout would need to plug into is live and unbroken today, which is the
precondition the recommendation in §4 depends on.

## 6. Bottom line

The mouth de-Qwen audit referenced on the board is real and accurate. Qwen's live production footprint is smaller
than the raw "Qwen is the mouth" framing suggests (the bounded-SVO recall surface already bypasses it by default),
concentrated now in (a) the irregular/open-prose fallback of the recall surface and (b) the entire, currently-off,
open-ended free-generation channel. The 2026-08-28 readout result does not itself touch either Qwen touchpoint —
it lives on an architecturally separate, not-yet-wired WKV cortex — but it removes the one property (a host-copied,
not locally-learned, final layer) that made wiring that cortex into production premature. The smallest concrete
next rung is a bounded, additive, default-OFF swap of the open-ended generation channel's generator from
`SpikingQwenFaculty` to the WKV mouth (learned readout installed), behind the SAME verify/honesty gate already
built for that channel — scoped to its vocabulary/topic limits and pending the two flagged provenance/residual
checks in §4. This is a genuine build (webapp wiring + a vocab/topic gate + a soak), correctly out of scope for
this pass.
