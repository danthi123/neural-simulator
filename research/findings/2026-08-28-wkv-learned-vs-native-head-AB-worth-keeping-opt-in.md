---
type: finding
status: qualified
date: 2026-08-28
mechanism: A/B of the e-prop LOCALLY-LEARNED WKV-mouth read-out head vs the checkpoint's NATIVE (host-trained
  "copied") head, through the PRODUCTION `webapp.wkv_mouth_generator.generate()` entry point
verdict: WORTH-KEEPING-AS-OPT-IN (default-OFF, no default-flip recommended). The learned head generates
  DECISIVELY coherently -- self-NLL 1.51 nats mean vs chance 6.91 nats (5.39-nat separation) on all 8 in-vocab
  prompts, in-vocabulary grammatical TinyStories-domain words throughout -- and on the primary self-NLL metric
  is statistically close to native (+0.045 nats aggregate <!--derived-->, actually LOWER/better than native on
  5 of 8 individual prompts, with the aggregate gap driven by 3 fat-tailed outlier prompts). An adversarial
  verify-go pass (2 independent skeptics) additionally found and corrected two real problems with the FIRST
  draft of this finding (a wrong win/loss direction, and an UNDERSTATED repetition residual): eyeball
  inspection of all 8 continuations shows a genuine, disclosed quality gap self-NLL alone does not fully
  surface -- word/short-phrase repetition loops appear in 5 of 8 learned samples (2 severe), vs 0 severe / 1
  mild genre-typical template repeat in native. This is a real residual, not a blocker to the opt-in verdict
  (every learned sample stays 4.7+ nats below chance and none is meaningless word-salad).
lane: e-mouth-fluency / A1 (crutch-burndown -- the concrete next step named by
  2026-08-28-persist-eprop-head-scope.md SS4)
artifacts:
  - research/findings/raw/_wkv_learned_vs_native_head_ab.json
  - research/findings/raw/_wkv_learned_vs_native_head_ab.json.prov.json
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz
  - research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz.prov.json
  - research/findings/2026-08-28-persist-eprop-head-scope.md
  - research/findings/2026-08-28-wkv-mouth-into-open-ended-WIRED-GO.md
  - research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md
  - webapp/wkv_mouth_generator.py
runner: research/runners/_wkv_learned_vs_native_head_ab.py
---

# WKV mouth: learned-head vs native-head A/B -- coherent, close on self-NLL, an honest repetition residual

## 0. What this closes

`2026-08-28-persist-eprop-head-scope.md` persisted an opt-in load path (`BRAIN_WKV_MOUTH_LEARNED_HEAD` /
`BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH`, both default-OFF, fail-safe on any error) for the e-prop LOCALLY-LEARNED
WKV-mouth read-out head `W_hat` -- trained by a local three-factor rule against the batched-substrate spiking
forward, **no weight transport, no host gradient**
(`2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md`: 6-seed `sub_recov_ratio_mean=0.8686` <!--derived-->,
min 0.8399 <!--derived--> -- quoted from that finding's own headline, not re-measured here) -- and named its
own concrete next step: *"a qualitative A/B against the native head on in-vocab prompts (self-NLL,
coherence)"*. This is that A/B.

**Framing, stated up front so the numbers below are read correctly**: the learned head recovers less than 100%
of the native head's own recovery of the checkpoint's target, so it is *expected* to generate somewhat worse
than native, not better. A GO here means *"generates coherently, is a legitimate opt-in"* -- never *"beats
native"*. This finding does not recommend, and this repo's convention (`docs/TERMS.md`) forbids inferring, a
default-on flip from a single-seed, single-artifact A/B.

## 1. An honest correction to the persisted artifact's own scope, found while building this A/B

<!--derived-->

The eprop runner's `--save-w-hat <path>` supports `{seed}`-templating (verified by reading
`research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py:714-722`:
`p = Path(save_path.format(seed=seed) if "{seed}" in save_path else save_path)`) -- but the actual invocation
that produced `wkv_eprop_learned_head_6seed.npz` passed a **literal** path with no `{seed}` in it
(`eprop_persist_6seed.json`'s own recorded `argv`: `--save-w-hat
research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz`). Run across
`--seeds 42,43,44,100,101,102` in that order, each seed's `np.savez` **overwrote the same file** -- so the npz
(despite its `_6seed` name) holds only the **last-processed seed's head: seed=102**, `sub_recov_ratio=0.9132`
(read directly from the npz's own `sub_recov_ratio` field). Per `eprop_persist_6seed.json`'s six per-seed
results, 0.9132 is the **best** of the six ratios (42:0.880, 43:0.840, 44:0.885, 100:0.845, 101:0.850,
**102:0.913**), not the 0.8686 six-seed **mean** the persist-scope finding headlines. This A/B therefore
necessarily runs at seed=102 (the checkpoint the head was trained against,
`bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed102.npz`) and reports the actual 0.9132-ratio artifact, not the
mean -- a real, disclosed discrepancy, not silently inherited. **Follow-up, not done here** (a good
independent next step): re-run `--save-w-hat ...seed{seed}.npz` with proper templating to persist all six
heads, so a future A/B can report the genuinely-typical (not best-case) seed.

Note on `eprop_persist_6seed.json` itself: it is NOT committed alongside this finding (unlike the npz) --
its own provenance sidecar reads `corpus_check_fresh: false` (the corpus check active when that 6-seed
training run launched was 24h-stale and unrelated in topic), so `gates/corpus-check-required` correctly
declines to let it land as a newly-added artifact in THIS commit. The per-seed ratios quoted above were read
directly from that file (directory `_persist_eprop_head_scope`, same directory as the npz) before this doc
was written; a future commit that re-checks the corpus for that specific training run (or re-runs it after
one) can land it properly.

## 2. Method

`research/runners/_wkv_learned_vs_native_head_ab.py`, CPU/numpy, run detached
(`SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_learned_vs_native_head_ab`, ~1.5s elapsed,
`research/findings/raw/_wkv_learned_vs_native_head_ab.json`, provenance sidecar auto-recorded). 8 in-vocab
TinyStories-domain prompts (verified individually via `webapp.wkv_mouth_generator.in_vocab_scope`), each
generated **twice** through the SAME production `generate()` call (seed=102, `max_new_tokens=50,
read_window=40, pop=8, topk=64, gen_temp=0.8`) -- once with `BRAIN_WKV_MOUTH_LEARNED_HEAD=0` (native), once
with `=1` (learned, pointed at the seed-102 npz) -- the flag is the ONLY thing that differs between the two
calls for a given prompt. Self-NLL of the generated continuation is scored by teacher-forced replay under the
ACTIVE head's own next-word distribution (gate on the PREVIOUS token to predict the NEXT -- the same
convention `_free_gen` / `_wkv_mouth_open_ended_wiring_verify` both use), against chance `log(1000)=6.9078`
nats. Coherence beyond self-NLL: distinct-1/2/3 n-gram ratios and the longest consecutive-repeated-word run,
per continuation.

## 3. Adversarial verification (verify-go, 2 independent skeptics) -- found and fixed two real problems

Before this doc was written, two independent skeptics attacked the runner and its raw JSON from five angles
(instrument-trust on self-NLL, lever-firing/silent-fallback, RNG confound, sample-size/power, eyeball
coherence). **Three angles PASSED clean**: (a) the teacher-forced self-NLL replay is gated correctly and reads
the SAME cached `(seed, learned)` readout object that just generated -- no stale-head scoring; (b)
`heads_differ=True` and `all_learned_applied=True` hold on every one of the 8 learned-arm calls (native
`head_hash` constant at `76f3cb4644694359` across all 8, learned constant at `c6bb00a04ff528bc`, genuinely
different -- the swap fired every time, never a silent fallback); (c) the RNG is, if anything, a **cleaner**
control than intended: `FewSpikeWordRead._build_bank()` rebuilds and `SimulationBridge` deterministically
reseeds `np`/`cp`/`random` to `seed=102` on **every** `generate()` call regardless of arm, so both arms consume
the identical Izhikevich noise realization -- the head swap is the sole independent variable, and host RNG
state is verified byte-identical before/after the whole run (`rng_untouched_across_ab: true`).

<!--derived-->

**Two angles found real problems in the FIRST draft of this finding, both corrected here, not silently
dropped:**

- **Direction error.** The draft claimed "learned beats native on 3 of 8 prompts." Independently recomputed
  per-prompt deltas (`learned_self_nll - native_self_nll`) show the **opposite direction and count**: learned
  is actually LOWER (better) on **5 of 8** prompts (indices 0,2,3,4,6: -0.116, -0.163, -0.189, -0.009, -0.212
  nats), native wins on only **3 of 8** (indices 1,5,7: +0.525, +0.265, +0.256 nats) -- but those 3 losses are
  large enough (a fat right tail) that they alone account for the entire +0.045-nat aggregate gap (5 wins sum
  to -0.689, 3 losses sum to +1.046, net +0.045/8 mean). "Roughly on par, aggregate driven by outliers" is
  defensible; "learned wins 3/8" was backwards.
- **Understated repetition.** The draft characterized repetition as "isolated to a couple of the 8 samples."
  Reading every one of the 8 learned continuations verbatim (not just the automated distinct-2/max-repeat-run
  scores) finds a repeated word or short phrase in **5 of 8** (indices 0, 1, 4, 6, 7), with **2 severe**:
  index 0 -- `"...it was too high high high high in the air high house..."` (the literal word "high" 4x
  consecutive, `max_repeat_run=4`); index 1 -- `"...and again ever again and again ever again and again ever
  again and again again and again i was safe again and again..."`, a genuine degenerate collapse, and (not a
  coincidence) the single worst self-NLL outlier of the whole run (2.176 nats, native's self-NLL on that same
  prompt was 1.651). The milder cases: index 4 (`"food for food"`), index 6 (`"...or again or again or again
  or again or again or again"`), index 7 (`"food for food"`, `"books and books"`, `"together together...
  together"` 3x). Native shows 0 severe repetition and one mild TinyStories-genre-typical templated repeat
  (index 4: `"...they all lived happily ever work together...they all lived happily ever after...and they
  all"` -- itself an in-distribution formula of the training corpus, not a degenerate loop).

**A genuine instrument-limitation surfaced by this check, worth stating plainly**: the automated
`max_repeat_run` metric (longest run of *immediately consecutive* identical tokens) is blind to the
non-adjacent "A B A B A B" loop shape -- index 6's visible `"or again or again or again..."` scores
`max_repeat_run=1` (each "again" is separated by "or"). More strikingly, index 6's self-NLL is actually
**better** than native's on that same prompt (1.291 vs 1.504 nats) *despite* the visible loop -- direct
evidence that a model can be highly self-confident of ITS OWN repetition (a well-known autoregressive
degenerate-attractor failure mode: once "X or" has been generated, predicting "again" next is genuinely
high-probability under the model's own distribution, so self-NLL alone systematically under-penalizes this
class of artifact). Self-NLL is a real coherence signal (no learned sample gets anywhere near chance) but is
not sufficient alone to certify fluency; the eyeball + n-gram checks are load-bearing, not decorative.

## 4. Results (corrected, post-verification)

<!--derived-->

| | native | learned | chance |
|---|---|---|---|
| self-NLL mean (nats) | **1.469** | **1.514** | 6.908 |
| self-NLL range | 1.280 - 1.651 | 1.232 - 2.176 | -- |
| separation from chance | 5.439 | 5.394 | -- |
| distinct-2 mean | 0.839 | 0.898 | -- |
| max-repeat-run mean | 1.00 | 1.625 | -- |
| severe repetition-loop samples (eyeball) | 0/8 | 2/8 | -- |
| any repetition/stutter (eyeball) | 1/8 (mild, genre-typical) | 5/8 | -- |
| prompts where this arm has the lower (better) self-NLL | 3/8 | 5/8 | -- |

Both arms are decisively, overwhelmingly more on-distribution than chance (>5.3-nat separation on every
sample) -- every generated continuation, in both arms, uses grammatical, in-vocabulary TinyStories-domain
words in sensible local word order; none is word-salad. On the primary self-NLL metric the two arms are close
(learned +0.045 nats worse on aggregate, actually better on a majority of individual prompts, gap driven by 3
outlier prompts) -- closer than the 0.9132 substrate-recovery ratio alone might suggest. The real, disclosed
gap is in repetition/looping: learned shows this artifact in 5/8 samples (2 severe) where native shows it in
essentially 0-1/8, a genuine quality residual self-NLL under-detects for the reason given in SS3.

Anti-cheat / sanity, all read directly from the artifact: `heads_differ: true`, `all_learned_applied: true`
(no silent fallback on any of the 8 learned calls -- every `learned_head_status.reason` is `null`),
`rng_untouched_across_ab: true`. The runner's own `tools.verdict.Verdict` (four preconditions: lever moved,
fail-safe applied on every call, host RNG untouched, self-NLL-vs-chance separation > 2.0 nats) reads **GO**.

## 5. Verdict

**WORTH-KEEPING-AS-OPT-IN.** The e-prop locally-learned WKV-mouth read-out head -- trained with no weight
transport and no host gradient, purely by a local three-factor rule against the genuine batched-substrate
spiking forward -- generates decisively coherent, in-vocabulary, grammatical TinyStories-domain prose through
the exact production entry point (`webapp.wkv_mouth_generator.generate()`), and on the primary self-NLL metric
sits close to (and on a majority of individual prompts, better than) the checkpoint's native, host-trained
head. This is a genuine crutch-burndown milestone: a mouth read-out the BRAIN's own local learning rule
produced, not a host-BPTT-copied one, now demonstrably usable end-to-end. The opt-in
(`BRAIN_WKV_MOUTH_LEARNED_HEAD`, default-OFF, fail-safe) stays exactly as scoped by the persist finding --
this A/B does not change its default-OFF status and does not recommend flipping it, per the task's own scope
and this repo's standing convention that a single-seed A/B does not license a default change.

**Honest residuals, named as next levers, not swept under the verdict:**
1. **Repetition/looping** appears in a genuine, non-trivial share of learned samples (5/8, 2 severe) that
   native does not show at this sample size -- a real quality gap. Candidate next levers: a repetition
   penalty / no-repeat n-gram constraint at generation time (cheap, host-side, does not touch the learned
   weights); more e-prop training epochs specifically targeting the seeds that under-performed (43, 100, 101
   sat below the strict per-seed bar in the original 6-seed confirmation); or a decorrelation-read primitive
   (the same read-fidelity theme already named as a residual lever in
   `2026-08-28-mouth-stale-coo-training-fix-fullscale-confirmation-GO.md`).
2. **The persisted artifact is single-seed, best-of-six (SS1)**, not the 6-seed-mean-representative head. A
   genuinely representative A/B needs the eprop runner re-run with proper `{seed}`-templated `--save-w-hat`
   so all six heads persist, and this A/B (or its equivalent) repeated per-seed.
3. **This is n=8 prompts, one seed, one hyperparameter configuration** -- sufficient to certify "generates
   coherently, is a legitimate opt-in" (the self-NLL separation from chance is enormous and robust across
   every single sample), but not sufficient to certify a precise "X% as good as native" quality number; that
   would need multi-seed replication, which is explicitly out of scope for this bounded A/B.
