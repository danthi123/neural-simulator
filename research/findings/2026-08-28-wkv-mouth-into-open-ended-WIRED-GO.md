---
type: finding
status: wired
date: 2026-08-28
mechanism: from-scratch WKV/SSM spiking mouth wired into the BRAIN_OPEN_ENDED free-generation channel as an
  alternate, in-vocab-scoped generator, behind a NEW default-OFF flag (BRAIN_OPEN_ENDED_WKV_MOUTH)
verdict: WIRED-GO (default-OFF; reachable from /api/brain-chat on BRAIN_OPEN_ENDED=1 + BRAIN_OPEN_ENDED_WKV_MOUTH=1
  + an in-vocab prompt). Flag-off content is byte-identical to the pre-edit path (exact dict compare, WKV module
  never imported). Flag-on + in-vocab: the real few-spike Izhikevich spiking-WTA decode generates coherent
  TinyStories-domain prose (self-NLL 1.02 nats vs chance 6.91 nats) through the SAME post_filter honesty gate.
  Out-of-vocab / a forced WKV-path exception falls back to Qwen cleanly. Two honest residuals carried forward,
  not resolved here (see §4): the specific e-prop-learned 0.87-ratio read-out head was never persisted to disk
  by the run that produced it, so this wires in the checkpoint's own native head instead; the checkpoint's
  recurrent-store training provenance (local rule vs host-BPTT) remains unverified, though a provenance sanity
  check here confirms it is a legitimately-trained model, not a placeholder.
lane: e-mouth-fluency / A1 (crutch-burndown, the north-star production rung named by
  2026-08-28-mouth-crutch-burndown-scope.md §4)
artifacts:
  - research/findings/raw/_wkv_mouth_open_ended_wiring_verify.json
  - research/findings/raw/_mouth_stale_coo_training_fix/eprop_STALEFIX_6seed_frommain.json
  - research/findings/2026-08-28-mouth-crutch-burndown-scope.md
  - research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md
  - webapp/wkv_mouth_generator.py
  - webapp/open_ended_chat.py
  - webapp/server.py
  - bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz
runner: research/runners/_wkv_mouth_open_ended_wiring_verify.py
---

# WKV mouth wired into `BRAIN_OPEN_ENDED` — a real, additive, default-OFF crutch-burndown rung; two honest residuals carried forward

## 0. What this is

The 2026-08-28 scope pass ([`2026-08-28-mouth-crutch-burndown-scope.md`](2026-08-28-mouth-crutch-burndown-scope.md))
mapped the live `/api/brain-chat` pipeline and found exactly ONE touchpoint where the literal Qwen2.5-0.5B model
(`SpikingQwenFaculty`) is the sole generator with no fallback: `BRAIN_OPEN_ENDED` (server.py:4505-4535,
default-OFF), the free-generation channel. This rung wires a genuinely different, from-scratch, home-grown
recurrent SSM/RWKV-style spiking cortex — `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz` (V=1000
TinyStories vocabulary, D=128, its own embeddings/Wv/Wr/Wo_sp/head, architecturally unrelated to Qwen) — into that
channel as an alternate generator for in-vocab prompts, behind a SECOND, independent, default-OFF flag
(`BRAIN_OPEN_ENDED_WKV_MOUTH`). This is a genuine, additive, zero-live-risk step (`BRAIN_OPEN_ENDED` itself is
already off in production): a real capability with a real fallback, not a de-risk harness.

## 1. What changed (three files)

- **`webapp/wkv_mouth_generator.py`** (new). `in_vocab_scope(text, seed)` — a scope gate over the checkpoint's
  V=1000 vocabulary (see §3 for why a naive word-overlap version was insufficient). `generate(prompt, seed,
  max_new_tokens, ...)` — free-generates via the GENUINE few-spike Izhikevich soft-WTA population read
  (`WKVReadout` + `FewSpikeWordRead`, reused **verbatim by import** from the GO-verified
  `research/runners/_wkv_fewspike_read_derisk.py`; only the free-generation driving loop, which lived as a nested
  closure in that module's `run_seed`, is lifted out into a standalone function — the mechanism itself is
  unchanged). Wrapped in `_RngIsolation`, modeled directly on `webapp/affect_drives_chat.py`'s `_isolated`
  pattern (the #77 fix): `SimulationBridge` construction reseeds the process-global numpy/cupy/random RNGs
  (`sim/bridge.py:1625-1627`), so every entry point here snapshots the host RNG, runs on a private per-seed
  timeline, and restores the host state afterward — measured, not assumed (§2, check (d)).
- **`webapp/open_ended_chat.py`**. `wkv_mouth_enabled()` reads `BRAIN_OPEN_ENDED_WKV_MOUTH` (default-OFF, same
  truthy-string convention as the existing `open_ended_enabled()`/`gen_time_honesty_enabled()`). `answer_turn()`
  gained one new block, inserted before the existing generator selection: `if wkv_mouth_enabled(): try: ... if
  _WKV.in_vocab_scope(msg, seed=seed): raw, secs = _WKV.generate(...); wkv_used = True ... except Exception:
  wkv_used = False`, followed by `if wkv_used: pass elif <existing gen-time-honesty condition>: ... else: raw, secs
  = gen.generate(...)` — the pre-existing Qwen path is completely unchanged, just no longer unconditionally first.
  Two purely-additive trace keys, `generator` (`"qwen"`|`"wkv_mouth"`) and `wkv_mouth_used` (bool), were added to
  the returned dict.
- **`webapp/server.py`**. Only the `_oe_resp["open_ended"]` trace dict (inside the SAME pre-existing
  `BRAIN_OPEN_ENDED`-guarded block) gained `"generator": _oe.get("generator", "qwen")` and `"wkv_mouth_used":
  _oe.get("wkv_mouth_used", False)`, via `.get()` with safe defaults. Nothing else in `server.py` changed.

## 2. Verification — `research/runners/_wkv_mouth_open_ended_wiring_verify.py`, `tools.verdict.Verdict` → **GO**

CPU/numpy only (~512 neurons), no GPU, no Qwen render (stubbed to isolate the wiring from Qwen's own weights/
latency). Command: `SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_mouth_open_ended_wiring_verify`.
Artifact: `research/findings/raw/_wkv_mouth_open_ended_wiring_verify.json` (provenance sidecar auto-recorded).

**(a) Flag-OFF byte-identical content** — asserted in the data, not inferred from reading the code:
`answer_turn`'s full result (`answer`/`raw`/`filtered`/`topic`/`known`/`facts`/`gen_seconds`/
`gen_time_honesty_used`/`gen_time_trace`/`state`/`n_sentences`) was compared, key-by-key, between the PATCHED
module and a `git show HEAD:webapp/open_ended_chat.py` snapshot of the pre-edit original, with the Qwen call
stubbed identically in both and `BRAIN_OPEN_ENDED_WKV_MOUTH` unset: **`off_content_diffs: []`,
`off_byte_identical: true`**. A poison-pill `webapp.wkv_mouth_generator` module (raises `AssertionError` if any of
its functions are called) was substituted for the real one on this path and never fired — the WKV module is not
merely inert, it is **never imported at all** when the flag is off (`wkv_mouth_enabled()` short-circuits the `from
webapp import wkv_mouth_generator` line before it executes). Only the two new trace keys differ, at their
documented flag-off defaults (`generator=="qwen"`, `wkv_mouth_used==False`) — excluded from the diff by name, not
by silently discarding a real difference.

**(b) Flag-ON, in-vocab: coherent generation, honesty gate intact** — `on_used_wkv: true` (the Qwen stub, primed
to fire if the Qwen path were taken, never fired: `on_qwen_stub_fired: false`); `on_post_filter_ran: true` (the
existing honesty gate ran on the WKV output exactly as it would on a Qwen output). Coherence measured
objectively, not eyeballed: the raw continuation's own self-NLL under the checkpoint's teacher-forced next-word
distribution (gating on the PREVIOUS token to predict the next — the same convention
`_wkv_fewspike_read_derisk.run_seed`'s own `_free_gen` uses) is **`self_nll_wkv_continuation = 1.0245` nats**
against **`chance_nll_uniform_over_V = 6.9078`** nats (`log(1000)`) — a **5.88-nat separation**, i.e. the
generated text is drastically more on-distribution than chance, consistent with (and close to) the
already-recorded 0.819-nat self-NLL in the scope doc's own re-confirmed smoke <!--derived--> (§5 of that doc; quoted from that doc's run log, not a JSON field there either). Sample output
(seed 42, prompt `"once upon a time there was a little boy named tim who had a dog"`): *"once upon a time there
was a little boy named tim who had a dog named max was playing in the yard with his toy car and a red ball that
he loved to play with his toys and listen to his mom and dad and they became good friends and played together
every day"* — fluent, non-repetitive TinyStories-register prose, via genuine spiking WTA word-decode (not a host
argmax/softmax-sample; the underlying mechanism's own anti-cheats — scramble/equal-drive/noise-ablation/
provenance — are `_wkv_fewspike_read_derisk`'s own GO result, reused unchanged, not re-derived here).

**(c) Fallback is genuine, not decorative.** An out-of-vocab prompt with the flag ON falls back to Qwen —
`oov_falls_back_to_qwen: true`, output byte-identical to the flag-off case (the Qwen stub fires). A forced
exception inside the WKV generator (a LESION of the new path) also falls back cleanly:
`lesion_falls_back_to_qwen: true` — the turn never crashes.

**(d) RNG discipline, measured not assumed.** Host process-global numpy RNG state
(`np.random.get_state()`) is **byte-identical** immediately before and after a WKV-path call:
`rng_untouched_across_wkv_call: true`.

**(e) Provenance.** The wiring's own `FewSpikeWordRead` instance makes **zero** host categorical draws on the
read path: `host_rng_draws_on_read_path: 0` — the winner is read from `cp_firing_states`, not sampled by the
host.

**Independent adversarial check.** A second agent, given only the file diffs and no access to this doc, re-ran
the verify runner fresh and independently re-derived every number above (byte-identical content, self-NLL 1.0245
vs chance 6.908, RNG untouched, zero host draws, both fallback paths), wrote its own 3 fresh prompts through
`generate()` and judged the output fluent, and independently re-implemented the RNG-restoration check. It flagged
one real weakness in the FIRST version of `in_vocab_scope` (§3) — fixed before this doc was written, not after.

## 3. A real bug found and fixed mid-verification: the vocab-scope gate was gameable

The independent adversarial pass found that a pure word-overlap `in_vocab_scope` (≥60% of a prompt's words present
in the checkpoint's vocabulary) is **gameable**: the checkpoint's own vocabulary is frequency-sorted, so its
top-40 entries are common English function words (`the/and/a/to/was/they/he/it/she/her/with/in/his/you/but/not/
on/i/of/there/so/for/that`, verified directly:
`np.load("bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed42.npz")["words"][:40]`). A prompt like `"the a to and"` (zero
content) or `"what is your opinion on the stock market today"` (topically empty relative to TinyStories) both
scored above the 60% threshold on function-word overlap alone. **Fixed before landing**: `in_vocab_scope` now
additionally requires `min_content_hits` matches that are NOT in a small `_FUNCTION_WORDS` set (the checkpoint's
own dominant function words, plus a few possessive/deictic words the frequency-sort alone didn't separate —
`your`/`my`/`our`/`its`/`today`/`now`/`yesterday`/`tomorrow`, found empirically to slip past the first fix). Both
example prompts above now correctly return `False`; genuine TinyStories-domain prompts (`"once upon a time there
was a little boy named tim who had a dog"`, `"tell me a story about a happy dog and his best friend"`) still
return `True`; the geopolitics/cryptography out-of-vocab example still correctly returns `False`. The full wiring
verify (§2) was re-run after this fix and remains GO with unchanged numbers — this was purely a scope-gate
precision fix, not a mechanism change.

**Honest residual on this gate, stated precisely, not hidden**: it is a heuristic (word-overlap + a hand-audited
function-word exclusion), not a semantic topic classifier. It will still admit some prompts whose vocabulary
happens to overlap the checkpoint's 1000 words but whose intended meaning is genuinely out-of-domain (e.g. a
prompt built entirely from TinyStories-common nouns arranged into an adult-register question), and it will still
reject some genuinely in-domain prompts that phrase things with one uncommon synonym. This is an acceptable,
disclosed v1 scope boundary for a default-OFF rung — not a claim of precise topic detection.

## 4. Two honest residuals from the crutch-burndown scope doc, NOT resolved by this rung

**(1) The specific 0.87-ratio e-prop-LEARNED read-out head was never persisted to disk.** Re-read
`research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py` (the runner behind the 2026-08-28
full-scale confirmation, `sub_recov_ratio_mean = 0.8686`) end to end: `W` is initialized `0.01 *
rng.standard_normal((V, D))` (near-zero random, `research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py:536`),
trained in-memory by the local three-factor rule into `W_hat`, evaluated against `hw.copy()` (the checkpoint's own
`head.weight`, called "copied" in that runner's own vocabulary), and **only summary metrics** (`w_hat_norm`,
recovery ratios, etc.) are written to its JSON artifact — grepping the runner for `np.save`/`savez` on the learned
matrix finds nothing. The learned `W_hat` exists only transiently inside that one run's process memory and is
discarded when the process exits. **Consequence**: this wiring uses the checkpoint's OWN existing `head.weight`
(the run's "copied" reference, `sub_copied_recov_mean = 0.9785` in that same confirmation) — itself genuinely
WKV-native, architecturally unrelated to Qwen, and empirically coherent (§2b, §5 below) — rather than the specific
0.87-ratio matrix. Wiring in the actual e-prop-learned weights is a concrete, well-scoped next step: extend that
runner to persist `W_hat` (one `np.savez` call) and re-run a training pass (GPU-bound per that runner's own
docstring, ~874s/seed at full B=48 scale per the confirmation finding) — named here, not done in this rung, to
keep this build small/CPU-only per the task's own memory-budget constraint (a GPU run was in flight elsewhere).

**(2) The checkpoint's recurrent-store training provenance remains unverified** — carried forward unchanged from
the scope doc's own §4 residual 2: whether `Wv`/`Wr`/`Wo_sp`/embeddings were produced by the diagonal e-prop
local rule (2026-08-12 GO) or an earlier host-BPTT pass is not resolved by this rung either. What this rung DOES
add, per the task's explicit request for a provenance sanity check ("is `bridges/wkv_ckpt/*.npz` a legitimately-
trained checkpoint, or a placeholder?"):

- **Vocabulary is frequency-rank-ordered and real**: `words[:40]` are exactly the common-word profile a
  TinyStories-frequency sort produces (`the, and, a, to, was, they, he, it, she, her, with, said, day, in, his,
  you, big, but, one, not, had, endoftext, mom, happy, on, saw, i, play, very, lily, of, there, so, little, time,
  tom, named, for, that`) — not random tokens, not a placeholder's fixed/degenerate list. `<unk>` sits at the
  vocabulary's tail as the low-frequency sentinel, as expected.
- **Weight statistics are non-degenerate**: `head.weight` shape `(1000, 128)`, mean `6.2e-4`, std `0.1049`, range
  `[-0.44, 0.51]`; `emb.weight` mean `6.2e-5`, std `0.11678` — bounded, regularized-looking distributions, no NaN,
  no all-zero/all-identical rows (which a never-trained or corrupted placeholder would show; all four measured
  and saved by `_wkv_mouth_open_ended_wiring_verify.py`'s own `checkpoint_provenance_sanity` block, not eyeballed).
- **Empirically coherent generation** (§2b, and independently re-confirmed by the scope doc's own §5 smoke,
  self-NLL 0.819 <!--derived--> quoted from that doc's run log) is itself strong evidence against "placeholder": a
  randomly-initialized or corrupted head could
  not produce fluent, grammatical, topically-consistent TinyStories prose with self-NLL close to the model's own
  ideal-sampler ceiling.

This is a legitimately-trained model — the sanity check clears — but WHICH training method (the thing the
crutch-burndown ultimately cares about, for the brain-based-only standard) is still not determined here, and is
correctly flagged rather than assumed either way.

## 5. What this rung is, and is not

**Is**: a real, reachable (per `docs/TERMS.md`'s `wired` definition — a call path exists from `/api/brain-chat` →
`ChatBrain` → this code, on a request with both flags set + an in-vocab prompt), additive, default-OFF alternate
generator at the ONE live touchpoint where Qwen previously had zero competition. Verified byte-identical when
off, genuinely functioning (not a stub) when on, honestly scoped to a bounded vocabulary, with a real fallback and
a real lesion-safety property.

**Is not**: `on-by-default`, `scaffold-retired`, or `integrated` in this codebase's precise sense (per
`docs/TERMS.md`) — `BRAIN_OPEN_ENDED` itself stays default-OFF in production (unchanged by this rung), Qwen
remains the default and the only generator for out-of-vocab/broad-topic open-ended prompts, and Touchpoint A (the
spiking-Broca recall surface's irregular-verb/copula/open-prose fallback) — the LARGER live share of actual
Qwen render calls today per the scope doc — is untouched. Not a claim of "fully spiking": the checkpoint's own
training provenance (§4.2) is unresolved, and the specific locally-learned 0.87-ratio readout is not what is
wired in (§4.1) — the checkpoint's pre-existing native head is.

**Next steps** (not this rung): (i) persist `W_hat` in the eprop runner and re-run to actually swap in the
locally-learned head; (ii) resolve the recurrent-store training-method provenance; (iii) a flip-soak of
`BRAIN_OPEN_ENDED_WKV_MOUTH` gated on both residuals closing, plus real end-to-end testing with the live warm
Qwen faculty present (this verify stubbed Qwen to isolate the wiring; it does not test Qwen-vs-WKV output quality
head to head; it does not test the GPU/cupy backend path, deliberately, to stay memory-light while a GPU run was
in flight elsewhere).
