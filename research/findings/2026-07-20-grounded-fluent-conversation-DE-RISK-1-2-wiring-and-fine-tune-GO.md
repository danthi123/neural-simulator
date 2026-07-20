# Grounded fluent conversation — DE-RISK 1 (wiring+moat) + DE-RISK 2 (format fine-tune) BOTH GO

**Date:** 2026-07-20 · **Status:** DE-RISK 1 GO + DE-RISK 2 GO — the spiking WKV renders grounded fluent answers in
the fluid console, gate-first moat intact. Off-bridge numpy (De-risk 3 = the on-bridge spiking parity swap). NO
`sim/` edit; additive default-preserving wiring (`renderer="ft"` default = byte-identical to the shipped console).

## The frontier + the residual (recap, De-risk 0)

The fluid console (`_fluidconv_chat_repl.py`) already does grounded fluent conversation — gate-first no-confab moat +
grounded-fact retrieval + the grounding→generator handoff + VERIFY + template fallback are ALL GO. Its one shortcut
is the fluency renderer = the ~21M TinyStories **ANN** (`FTFaculty`). The residual = swap it for gap#1's **spiking
WKV**. De-risk 0 bounded the residual: the raw WKV renders 0.17 verified-fluent (rambles → template fallback) and
needs a **format fine-tune (residual-B)** to answer-not-ramble.

## DE-RISK 2 — the FORMAT FINE-TUNE (residual-B): GO

`_gap_grounded_wkv_finetune.py` — the EMERGE-57 lever ported to the WKV. A torch module that **bit-matches** the
`WKVFaculty` numpy forward (verify-first: torch-vs-numpy logits **corr=1.000000, maxdiff=0.0000** asserted BEFORE any
training) continue-trains the fluent `wkv_ssmU_v4000_d256_big` checkpoint on GROUNDED COPY frames (`the A v3 P <ans>
the A v3 P <eos>`, drawn from 25200 broad in-vocab SVO combos with the 22 curriculum facts HELD OUT) INTERLEAVED with
raw TinyStories (anti-forgetting). Two format markers `<ans>`/`<eos>` appended to the V=4000 word vocab (transparent
to the on-bridge path: 2 more emb/head rows). 3000 steps, ~15 s on GPU.

**GO gate (mirror P2/EMERGE-57), on facts HELD OUT of training (generalization, not memorization):**
- **focused-grounded 0.83** (5/6 clean fluent — "the dog eats meat", "the bird eats seed", "the fox chases rabbit",
  "the bee makes honey", "the fish sees water"; up from **0.17** raw). The 1 miss (cat→fish, the fish patient/subject
  collision) falls back to the grounded template (safe).
- **RA-faithful 1.00** (12/12, **0 bias**): prompt a DIFFERENT in-vocab patient (ball/cake/hat/bread/toy) → the WKV
  follows the PROMPT fact, never the memorized one → it learned the COPY skill (grounded on the retrieved fact),
  generalizing to any fact incl. taught/Wikidata.
- **anti-forgetting OK**: TinyStories held-out ppl 28.121 → 28.416 (+1%). The decay adapted 0.768→0.814
  (memory 4.3→5.4 tokens — the model learned longer memory to hold the subject for the copy).

**A silent-failure bug caught + fixed (verify-first discipline):** `WKVFaculty.unk = V-1` — a rotted assumption. The
fine-tune appends `<ans>`/`<eos>` AFTER `<unk>`, so `V-1` pointed at `<eos>`, and `no_unk=True` was **suppressing
`<eos>` itself** → degenerate repetition ("meat meat meat"). Diagnosed by a 3-way logit comparison (a probe said eos
top-1; free-running said eos absent — same frame). Fixed by indexing `<unk>` **by name**. After the fix every answer
is clean (eos fires). (Also added standard no-repeat-ngram loop-stop decoding as belt-and-suspenders.)

## DE-RISK 2 — 6-SEED FIRMING (n=22 held-out facts): GO, dev/blind reported separately

`_gap_grounded_wkv_multiseed_firm.py` — fine-tune the SAME base checkpoint with 6 different TRAINING seeds (varies
frame sampling + marker init + torch RNG), eval focused-grounded + RA-faithful on ALL 22 curriculum SVO facts (held
out of training):
- **DEV (42/43/44):** focused-grounded **0.833** (min 0.818), RA-faithful **1.00** (min 1.00), confab 0.167.
- **BLIND (100/101/102):** focused-grounded **0.849** (min 0.818), RA-faithful **0.992** (min 0.977), confab 0.151.
- anti-forgetting stable every seed (TinyStories ppl 28.12 → 28.1–28.4). Blind ≈ dev (no overfit to the dev seeds).

⇒ the single-seed caveat is resolved: the format-fine-tuned WKV renders grounded fluent answers at a **6-seed GO**
(focused-grounded ~0.84, RA-faithful ~1.00, confab ~0.16 → safe template fallback), on facts held out of training.

## DE-RISK 1 — WIRING + MOAT: GO

`FluidChat(renderer="wkv")` (additive; default `"ft"` = byte-identical) drops `WKVFaculty` in place of `FTFaculty`
(same `answer(facts_ctx, question)` interface + `.npar`/`.device`). The console runs end-to-end on the SPIKING WKV:
- **grounded Q&A fluent**: "what does the dog eat?" → "The dog eats meat"; "what does the fox chase?" → "The fox
  chases rabbit"; "what does the bee make?" → "The bee makes honey".
- **growth**: teach "the wolf eats rabbit" live → "what does the wolf eat?" → "The wolf eats rabbit" (the WKV renders
  a fact it NEVER saw in fine-tune, riding the brain's retrieval = live RA-faithful generalization).
- **GATE-FIRST MOAT VERIFIED (not asserted)**: 3 grounded / 3 untaught (lion/dragon/zzz). On every abstain the WKV
  faculty is invoked **0 times** (`n_invocations==0`; the `_answer` `p is None` short-circuit fires BEFORE the
  faculty is reached) → "I don't know." The moat holds by construction.

**Portability bonus:** the ANN `FTFaculty` checkpoint (`gen_tinystories_ra_ft.ckpt.pt`, 85 MB, gitignored) is ABSENT
on this migrated machine → the default console is not runnable here; the WKV renderer (9.8 MB npz, committed) is the
ONLY runnable grounded-fluent renderer, and it is the mission-aligned SPIKING cortex — the swap retires the ANN
scaffold AND makes the console portable.

## Read-out + next

- **⇒ the spiking WKV cortex is a drop-in grounded-fluent renderer for the fluid console: focused-grounded 0.83 +
  RA-faithful 1.00 + gate-first moat intact + anti-forget OK, on facts held out of training.** The ANN scaffold is
  retired for the render path (off-bridge; the numpy WKV forward is the CPU-portable reference).
- **Next: De-risk 3 (fully-spiking on-bridge)** — run the fine-tuned WKV on-bridge with the RF-phase / fully-synaptic
  spiking input (gap#1 parity, validated to ±0.015 nat), so the fluency is produced ON the spiking substrate = the
  genuine north-star GO ("type a question → the brain answers with fluent grounded prose, fluency on spikes"). Then
  De-risk 4 (open/rich prose — the honest field wall: render-per-fact + VERIFY).

Runners: `_wkv_faculty.py` (WKVFaculty, +npar/+device/+invocation-counter/+unk-by-name/+no-repeat), `_gap_grounded_wkv_finetune.py`,
`_gap_grounded_wkv_ceiling_probe.py` (+RA-faithful). Console: `_fluidconv_chat_repl.py` (`--renderer wkv`, default ft
byte-identical). Ckpt: `bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz`.
