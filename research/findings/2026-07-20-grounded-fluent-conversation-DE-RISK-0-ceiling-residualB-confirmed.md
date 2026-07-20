# Grounded fluent conversation — DE-RISK 0 (CEILING): the raw WKV renderer needs a format fine-tune (residual-B), bounded

**Date:** 2026-07-20 · **Status:** DE-RISK 0 GO (ceiling bounded; the critical path is residual-B) · **Frontier:**
the north-star "a brain you COMMUNICATE with" — swap the fluid console's ~21M ANN renderer for the new spiking WKV
(gap#1's open generator). NO `sim/` edit, pure numpy, no training.

## The frontier (research-gate scoped 2026-07-20)

Grounded fluent conversation is NOT "build a console." The fluid console (`_fluidconv_chat_repl.py`) already does
end-to-end grounded fluent conversation TODAY — gate-first no-confab moat, grounded-fact retrieval, multi-intent
dispatch, the grounding→generator handoff (`_answer:319-333` = GATE → prompt-condition → VERIFY → template fallback),
growth, persistence — ALL GO. Its ONE shortcut is the **fluency renderer = the ~21M TinyStories ANN** (`FTFaculty`,
spiking-forward-deferred). The genuine residual is a single swap: replace `self.faculty = FTFaculty()` with the
spiking WKV (`WKVFaculty`), behind the unchanged moat + VERIFY.

**Trust-but-verify (done, all three load-bearing claims confirmed myself):** (1) `_answer:319-333` IS the gate-first
handoff (`p is None → "I don't know"` BEFORE `faculty.answer` is reached; VERIFY + template fallback) — read in code.
(2) `FTFaculty.answer(facts_ctx, question)` is the interface to match — read in code. (3) **vocab compatibility: 0
missing** — all 50 needed tokens (37 curriculum concept words + `the/a/can/does/not/is` + the 6 verb 3sg inflections)
are in the WKV V=4000 vocab (checked directly against `wkv_ssmU_v4000_d256_big_seed42.npz`). So the spiking WKV can
spell every grounded fact with its OWN trained head — no A→W bridge, no numpy/cupy co-execution seam.

## The residual has a known shape: residual-A (wiring, trivial) + residual-B (format fine-tune, the real work)

The WKV is a TinyStories **continuation** LM, not a QA model, and — verified here — it is **word-level with NO
punctuation/format tokens** (`.`, `?`, `:`, `facts` are all OOV). So the FTFaculty `facts:…question:…answer:` format
is un-representable on the raw model, and (per the P1/P2 record) a raw continuation LM RAMBLES when prompt-conditioned.
Residual-B = a DATA/format fine-tune (the EMERGE-57 lever) so the WKV answers-not-rambles.

## DE-RISK 0 — the ceiling, quantified (bounds residual-B before spending training compute)

`_gap_grounded_wkv_ceiling_probe.py` runs the raw WKV on the 22-fact grounded curriculum through the console's OWN
VERIFY (`_extract_all_svos`/`_fact_key`), two natural-prompt strategies (the only representable ones):
- **CONT** (prompt-condition on the fact, generate, VERIFY — mirrors `_answer`): **verified-fluent 1/6 = 0.17**;
  would-confab **0/6 = 0.00**; fallback-to-template **5/6 = 0.83**.
- **COMPLETE** (prime "the A v3 P the A v3", is the next word P?): **top-1 0/6, top-5 0/6** — the raw WKV does NOT
  carry the just-stated fact into the completion (it emits function words `and/the/for/…`); the ~4-token decay
  (`decay=0.768`) + the TinyStories prior swamp the fact. **No cheap no-training path.**

Example: primed with "the dog eats meat", the raw WKV continues *"and the apple fell on the ground and the ground
became even bigger…"* — fluent TinyStories, but not a grounded answer → the console falls back to the grounded
template ("The dog eats meat.").

## Read-out

- **VERDICT: residual-B (format fine-tune) NEEDED** — confirmed quantitatively (0.17 fluent-verified), exactly as the
  scoping predicted. The ceiling is bounded: the swap works mechanically (the console still answers, via template
  fallback, 0.83) and the **moat is intrinsically safe even raw (0 confab — the raw WKV never fabricates a curriculum
  fact)**, but the CAPABILITY (fluency) requires the fine-tune.
- **Reusable artifact:** `research/runners/_wkv_faculty.py` — `WKVFaculty` (drop-in for `FTFaculty`, bit-matches the
  on-bridge rate-SSM analog forward), used by De-risk 0 and by the wiring/fine-tune/on-bridge rungs to come.
- **Next (the ladder):** De-risk 1 (wiring + moat, cheap, no training — swap `WKVFaculty` into `FluidChat`, GO gate =
  suite passes + 0 WKV-invocation on abstains + default-path parity); De-risk 2 (the FLUENCY GO — the format
  fine-tune, EMERGE-57 lever, RA-faithful + anti-forgetting); De-risk 3 (fully-spiking on-bridge, RF-phase input,
  gap#1 parity); De-risk 4 (open/rich prose, the honest field wall — render-per-fact + VERIFY).

Runners: `_wkv_faculty.py`, `_gap_grounded_wkv_ceiling_probe.py`. Out: `research/findings/raw/_gap_grounded_wkv_ceiling.json`.
