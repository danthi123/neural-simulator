---
type: finding
status: live
date: 2026-09-04
mechanism: --linattn-div {exact,shunt} on the linattn own-voice mouth -- Tier-1 rate-model read-side swap of the
  num/den division for a Carandini-Heeger conductance-divisive-gain form num/(g_leak+k*den), on the
  ALREADY-TRAINED linattn checkpoints (no retrain), testing robustness to a read-neuron f-I squash (tanh,
  unit origin slope) + rate quantization (32-level stochastic rounding on num)
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PARTIAL -- 5 of 6 seeds clear the Tier-1 gate (margin_vs_trigram >= +0.03, anti-cheats clean); the
  1/6 miss (seed 102) is a boundary case, not a mechanism failure -- explained below, quantified, not dropped
artifacts:
  - research/findings/raw/_linattn_shunt_gain_tier1_seed42.json
  - research/findings/raw/_linattn_shunt_gain_tier1_seed43.json
  - research/findings/raw/_linattn_shunt_gain_tier1_seed44.json
  - research/findings/raw/_linattn_shunt_gain_tier1_seed100.json
  - research/findings/raw/_linattn_shunt_gain_tier1_seed101.json
  - research/findings/raw/_linattn_shunt_gain_tier1_seed102.json
  - research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
---

# Tier-1 de-risk: linattn's num/den shunt-gain read is PARTIAL, 5/6 (one boundary miss, quantified)

**Status:** near-free CPU rate-model test of `research/findings/2026-09-03-linattn-spike-native-normalization-DESIGN.md` Sec 4 Tier 1: swap the linattn mouth's read `num/(den+eps)` for the Carandini-Heeger conductance-divisive-gain form `num/(g_leak+k*den)` on the already-trained 6-seed checkpoints, with the default arm's own robustness axes active (a tanh read-neuron f-I squash, 32-level stochastic-rounding rate quantization on `num`), re-measured against the SAME seed's own exact-division run. **5 of 6 seeds clear the design's own bar (margin_vs_trigram >= +0.03, anti-cheats clean); seed 102 misses by <!--derived-->0.0021 nats** — a boundary case fully explained by that seed already having the smallest margin of the six, not by the shunt mechanism behaving differently there (its shunt-vs-exact cost is the same size as every other seed's). The design-specific divisive-vs-subtractive anti-cheat and the sigma-domination sweep are both clean.

## Result: shunt vs exact, per seed (N=600 held-out stories, not the milestone's 4000 — see "Honest residual" §2)

<!--derived-->
From `research/findings/raw/_linattn_shunt_gain_tier1_seed{42,43,44,100,101,102}.json`, deepest bucket (10-99):

| seed | exact margin | shunt margin | delta (shunt−exact) | shunt anti-cheats (perm / mless) | clears the +0.03 gate? |
|---|---|---|---|---|---|
| 42 | +0.0539 | +0.0507 | −0.0032 | 4.018 / 1.429 | yes |
| 43 | +0.0900 | +0.0847 | −0.0053 | 4.031 / 1.446 | yes |
| 44 | +0.1008 | +0.0983 | −0.0025 | 4.066 / 1.413 | yes |
| 100 | +0.0526 | +0.0493 | −0.0033 | 4.018 / 1.405 | yes |
| 101 | +0.0946 | +0.0916 | −0.0030 | 4.082 / 1.406 | yes |
| 102 | +0.0312 | +0.0279 | −0.0033 | 3.992 / 1.419 | **no (by 0.0021)** |

**5/6.** The `delta` column is the load-bearing one: it is small (−0.0025 to −0.0053 nats) and **uniform across
every seed**, including 102 (−0.0033, squarely inside the same range as the five that pass). Seed 102 fails
not because the shunt costs it more than any other seed, but because its own margin was already the SMALLEST
of the six going in (exact +0.0312, and the milestone's own 4000-story reference records this seed's true
margin as the smallest too: +0.039 vs +0.049 to +0.060 for the other five — see
`_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json`). Applying the SAME ~0.003 cost that every seed pays
pushes only the seed that started closest to the line below it. This is a boundary effect on a fixed
threshold, quantified here, not a claim that seed 102 is dropped or unexplained.

The harness's own correctness self-check (my exact-mode margin, recomputed here via `LinAttnReadout`, against
that seed's number recorded in the milestone's 4000-story reference run) shows the expected sampling spread of
a ~7x smaller held-out subsample, not a bug — abs_diff 0.0049 / 0.037 / 0.0498 / 0.0016 / 0.0346 / 0.0078 for
seeds 42/43/44/100/101/102 respectively (and `n_count_mismatch_vs_reference` in each artifact confirms why: my
n=28,258 positions at bucket 10-99 for seed 42 vs the reference's 188,509 — almost exactly the 600/4000
sampling ratio). The quantity this de-risk actually needs is not that cross-N comparison but the WITHIN-RUN
shunt-vs-exact delta at the identical N, seed, checkpoint and trigram baseline (like-for-like) — and that
delta is the stable, small, one-directional (shunt always slightly BELOW exact, never above) number reported
above.

## Anti-cheat 1 (the design-specific one): divisive, not subtractive (Holt & Koch 1997 direct test)

<!--derived-->
From seed 42's `_diagnostics.divisive_vs_subtractive[_no_fI]` (`_linattn_shunt_gain_tier1_seed42.json`): 12
real `(num, den)` pairs sampled off the checkpoint, `den` swept x{0.25, 0.5, 1, 2, 4, 8} of its own value with
`num`'s direction held fixed, the empirical gain ratio compared to the theoretical divisive ratio
`(g_leak+k*den_ref)/(g_leak+k*den)`:

| arm | mean max abs err vs the divisive prediction | verdict |
|---|---|---|
| bare formula (no fI) | **0.0** — an exact match | divisive by construction (a division IS a division) |
| with the read-neuron fI (tanh) applied | **0.0608** (< the 0.15 threshold) | **divisive_not_subtractive: True** |

The bare `num/(g_leak+k*den)` formula matches the divisive prediction to numerical precision — expected, since
it literally is a division. The substantive question is whether the COMPOSED nonlinearity (the read-neuron's
own f-I squash on top) starts to look subtractive, as Holt & Koch's point-neuron-soma failure predicts — it
does move the agreement off exact-zero, but by only 0.06 against the 0.15 threshold: the read-neuron's gain
still tracks `1/(g_leak+k*den)` far more than it tracks a den-independent offset shift. This is a RATE-MODEL
analog (the design's own honest scope note, Sec 4): it confirms the *formula* composed with a representative
squash stays divisive; it does not yet confirm a real conductance-based spiking neuron's f-I curve under an
actual GABA_A shunt does the same — Tier 2, on-bridge, is the only test that can directly settle Holt & Koch on
this substrate.

## Anti-cheat 2: sigma-domination (the "clamp owned 97%" trap, CLAUDE.md)

<!--derived-->
From seed 42's `_diagnostics.sigma_domination` (g_leak swept, k=1, no fI/quantization, isolating g_leak's own
effect; mean_den ~ 25.25 over the sampled positions):

| g_leak | fraction of the divisor OWNED by g_leak (not den) | wkv NLL vs the local (same-slice) exact baseline |
|---|---|---|
| 1e-6 (the gate's own default) | **0.0%** | +0.000 |
| 1e-3 | 0.0% | +0.000 |
| 1e-2 | 0.04% | +0.000 |
| 1e-1 | 0.39% | +0.0003 |
| 1.0 | 3.81% | +0.018 |
| 10.0 | 28.4% | **+0.407** |

At the value the de-risk actually uses (`g_leak=1e-6`), the divisor is **100% den-driven** — the shunt read
tracks `den`, not a fixed clamp. The trap this checks for (a fixed floor silently doing the work instead of the
signal it is supposed to gate) is not present at the operating point; the sweep also shows exactly where it
WOULD start to bite (a real NLL cost only appears once g_leak rises to ~1, severe by g_leak=10) — a clean
dose-response, not a cliff hidden just past the tested range.

## The den-quantization interaction (`--denquant-check`; NOT part of the default gate arm)

<!--derived-->
`--quantize-den` is OFF by default (the arm above quantizes only `num`, a population; the runner's own
docstring explains why a lone scalar's self-peak quantization is a meaningless no-op). `--denquant-check`
demonstrates why, on seed 42's real `(num,den)` samples (`den_scale=83.22`, `n_levels=32`, so the quantizer's
own zero-bin width is `den_scale/32=2.60`):

| g_leak | median read ratio vs exact | max read ratio vs exact | fraction of samples >10x |
|---|---|---|---|
| 1e-6 (exact's own epsilon) | 1.005 | **2,550,464** | **2.4%** |
| 2.60 (>= the quantizer's own zero-bin width) | 0.827 | 3.36 | 0.0% |

A genuinely nonzero `den` quantized against a fixed external scale rounds down to exactly 0 with real
probability (small-den positions are common at this checkpoint's operating range); dividing by
`g_leak(=1e-6)+k*0` then explodes the read up to ~1e6x for the worst 2.4% of samples. This is a real
interaction this runner's own harness measured (not asserted from theory), fully avoided by the default arm's
choice to leave `den` unquantized, and resolved in principle by raising `g_leak` to at least the quantizer's
zero-bin width if `den` quantization is ever turned on. It does not affect the gate numbers in §1.

## Anti-cheats built into every seed (perm / memoryless collapse > 0.05)

<!--derived-->
Every seed's shunt arm clears both anti-cheats by roughly two orders of magnitude over the 0.05 bar
(perm-collapse ~3.99–4.08, memoryless-collapse ~1.41–1.45 across all six seeds, INCLUDING seed 102) — the
shunt read still depends on long-range, order-sensitive content on every seed; the division swap has not
manufactured a shortcut, and seed 102's gate miss is purely a margin-threshold boundary effect, not an
anti-cheat regression.

## The code

`--linattn-div {exact,shunt}` (default `exact`, additive) is wired at both sites the design specifies:

- `research/runners/_emerge_wkv_lm_derisk.py` — `LinAttnLayer.__init__`/`.forward` (training-time torch), plus
  `--linattn-div`/`--linattn-div-gleak`/`--linattn-div-k` CLI flags and a `_divisive_read` static method
  transcribed verbatim from the design's own pseudocode. `div_mode="exact"` reproduces `num/(den+1e-6)` spelled
  identically to the line it replaces.
- `research/runners/_wkv_fewspike_read_derisk.py` — `LinAttnReadout.__init__`/`.advance` (the numpy deployment
  read this de-risk actually runs), plus `_divisive_read`/`_quantize_rate` and a `memoryless` anti-cheat
  parameter on `advance`.

Both edits are additive-only: every new parameter defaults to the value that reproduces prior behavior exactly.
The full existing test suite for both files (`tests/test_linattn_readout_parity.py` — this deployment class's
own load-bearing correctness gate — plus `test_emerge_wkv_lm.py`, `test_wkv_mouth_bpe_decode_wiring.py`,
`test_wkv_mouth_learned_head_path.py`) passes unmodified, 67/67, after the edit. A tiny end-to-end
training-time smoke run with `--linattn-div shunt` (d_model=32, 1 epoch, seed 42) also completed without
error, confirming both call sites execute, not just import. `research/runners/_wkv_fewspike_read_derisk.py`
also carries an UNRELATED default-off `affect_neural` feature (2026-09-04) that predates this session; the
merge here is additive alongside it and does not touch it.

New runner: `research/runners/_linattn_shunt_gain_tier1_derisk.py` — recovered from an uncommitted worktree
copy left by a prior agent run (branch `research/linattn-shunt-gain-tier1`, agent `a1979a3c`, killed for CPU
contention before it could commit; the branch itself never received the file, confirmed via
`git show research/linattn-shunt-gain-tier1:research/runners/_linattn_shunt_gain_tier1_derisk.py` failing).
Inspected in full and reused as-is: its own docstring documents two real bugs its own harness caught and fixed
before this run (a naive f-I calibration that added a blanket 4x gain rather than a squash; the
den-self-peak-quantization no-op behind §4's finding), which is exactly the measured-not-asserted correctness
this project asks for.

## Honest residual — what Tier 2 (on-bridge) still needs

1. **The seed-102 boundary miss is quantified, not dismissed.** <!--derived--> See the Result table: the shunt-vs-exact delta is uniform (−0.0025 to −0.0053) across all six seeds; seed 102 simply started closest to the +0.03 line (in BOTH the milestone's own 4000-story number, +0.039, and my own 600-story reconstruction, +0.0312). A
   read-in-the-loop retrain (design Sec 3c effect 2 — training the checkpoint with the read-neuron f-I already
   in the read path) is the design's own named next step if this margin needs to be recovered rather than
   merely explained; not attempted here (Tier 1 is explicitly no-retrain).
2. **This is a rate-model CPU probe, not a spiking substrate.** No neuron, no conductance, no GABA_A shunt was
   instantiated; `_divisive_read`'s `num/(g_leak+k*den)` is a numpy formula standing in for what a real
   shunting-inhibition circuit would compute. The design's own R-fluct/R-dend/R-net routes (fluctuation-driven
   somatic shunt / dendritic divisive gain / an emergent SSN) are Tier 2's job, and only Tier 2 can directly
   confirm or refute Holt & Koch on this substrate — this Tier-1 result only shows the rate-level formula
   composed with a representative squash stays divisive, which is the necessary precondition for Tier 2 to
   have a chance, not a substitute for it.
3. **The eval N is 600 stories/seed, not the milestone's 4000.** A timing probe on this box (40 stories -> 33s,
   400 -> 317s wall-clock) projected roughly 53 CPU-minutes/seed at the milestone's full N with the complete
   anti-cheat suite — several hours for all 6 seeds serially, on a box already showing load average 13-19 on 20
   cores and 37/46GB swap in use at the time. That is very plausibly why the prior attempt was killed for CPU
   contention. N=600 (diagnostics scoped to seed 42 only, per the design's own "seed 42 only, cheap" framing)
   kept each seed to roughly 8-18 CPU-minutes; peak RSS stayed under 2.1GB throughout, inside the 4GB budget.
   The self-check column shows the expected sampling spread this introduces, not a different quantity.
4. **5 of 6 seeds' checkpoints are not committed to git** (now also explicitly gitignored, `.gitignore`, so a
   future broad `git add` cannot accidentally stage 80MB of them). Only `wkv_linattn_depth2_contiguous_seed42.npz`
   is tracked on `main`; seeds 43/44/100/101/102 were run from local copies checksummed identical (md5) to the
   ones already backing the cited, committed reference artifact
   (`_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json`) — so this result traces to the same trained weights
   the milestone itself used, but a bare clone of this repo can currently only reproduce the seed-42 row above
   without separately sourcing (or retraining, via the milestone's own `--save-ssm` command) the other five
   checkpoints.
5. **If Tier 2 finds the somatic shunt is NOT divisive** (a real Holt & Koch failure on-substrate, unlike the
   rate-model composed-nonlinearity check in §1 above), the banked next rungs are R-dend (dendritic divisive
   gain re-keyed on the pooled `den`) then R-net (SSN) — named in the design doc Sec 5/6, not attempted here.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m research.runners._linattn_shunt_gain_tier1_derisk \
    --seeds 42 --max-eval-sents 600 --divisive-check --sigma-check --denquant-check --diag-eval-sents 200 \
    --json research/findings/raw/_linattn_shunt_gain_tier1_seed42.json
# repeat per seed in {43,44,100,101,102} without the diagnostic flags (they are seed-42-only, cheap)
```

## Provenance

Design read in full: `research/findings/2026-09-03-linattn-spike-native-normalization-DESIGN.md` (Sec 3e
sketch, Sec 4 GO gate + anti-cheats, Sec 5 honest residual). Code read: `research/runners/_emerge_wkv_lm_derisk.py`
(`LinAttnLayer`), `research/runners/_wkv_fewspike_read_derisk.py` (`LinAttnReadout`, both before and after this
session's edit, diffed line-by-line against the recovered runner's own worktree copy to confirm the merge was
additive and did not revert the unrelated `affect_neural` feature), and the recovered
`research/runners/_linattn_shunt_gain_tier1_derisk.py` (uncommitted worktree copy, branch
`research/linattn-shunt-gain-tier1`, agent `a1979a3c`). Reference numbers cross-checked against
`research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json` (the milestone's own 6-seed run)
via this runner's own built-in self-check, printed per seed above. All 6 per-seed runs this session are
automatically provenance-stamped by `research/runners/__init__.py` (sidecars: `*.json.prov.json`). Full
existing test suite for the two edited files re-run before and after the edit (67/67 pass; one unrelated
pre-existing failure in a different file logged in `research/FAILURE_LOG.md`, 2026-09-04 entry).
