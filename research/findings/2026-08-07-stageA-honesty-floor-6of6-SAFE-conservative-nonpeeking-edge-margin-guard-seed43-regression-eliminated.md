---
type: finding
status: contributing
date: 2026-08-07
mechanism: stageA-honesty-floor-conservative-edge-margin-guard
lane: E-language
runner: research/runners/_stageA_honesty_floor_strengthen_derisk.py
builds_on: research/findings/2026-08-07-stageA-honesty-floor-strengthen-axis-separated-familiar-wrong-catch-4of6-fit-guard.md
artifacts:
  - research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed_edgeguard.json
  - research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed_edgeguard.json.prov.json
---

# Stage-A honesty floor is 6/6-SAFE via a conservative NON-PEEKING edge-margin guard — the seed-43 regression is eliminated; active catch is honestly 1/6 (the conservative safety trade)

Built ON the strengthen runner (`...strengthen-axis-separated-familiar-wrong-catch-4of6-fit-guard`, which was
active-CATCH 4/6, moat-safe 5/6, with ONE regression on seed 43). The owner-defined Stage-A crux: make the honesty
floor **6/6-SAFE** ("the honesty layer never makes honesty worse") by closing the seed-43 regression, accepting an
honest catch trade. This build adds a CONSERVATIVE edge-margin guard (additive, default-OFF, no `sim/` edit, `cfg.seed`
set via the imported builders). Full-size 6-seed run, ONE foreground process, backend numpy, seeds 42/43/44/100/101/102.
Artifact: `research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed_edgeguard.json`.

## The seed-43 diagnosis (from the parent artifact) and the fix
<!--derived-->

The parent's fit-quality guard routed the calibrated monitor whenever its VALIDATED self-read edge over recall was
POSITIVE (sign check). On seed 43 the val edge was +0.017 AUC — positive but tiny — and it FLIPPED sign on the
independent test draw (test edge −0.035), so routing it made 3.5x more confident-wrong asserts (deployed 0.110 vs
baseline 0.031). The parent characterized this as an irreducible coin-flip: on marginal seeds (|val edge| within
between-draw noise) which signal wins is within-variance, and a sign check cannot tell a real +0.017 from a noise
+0.017.

**THE FIX — a conservative margin tau that EXCEEDS the between-draw noise.** Route the calibrated monitor into the
certainty band ONLY when its validated edge clears tau; otherwise SAFE-FALLBACK to the recall baseline
(deployed==baseline by construction → provably no regression). tau is derived per seed from VALIDATION statistics ONLY:
`tau = z * SE_boot`, where `SE_boot` is the bootstrap std of the val self-read AUC margin (resample the held-out
validation block, recompute the deployed-signal margin) and `z = 2.0` is fixed a priori (a 2-sigma confidence that the
sign holds out-of-sample). The whole routing decision is computed inside `robust_fit_monitor` from the validation seed
(`seed+900001`, disjoint from the test battery seed) BEFORE the test battery is ever built — an in-run assertion
freezes the routing string before the test axis runs and re-checks it after (a PEEKING-BUG trip-wire).

## Result — the MISSION familiar-but-wrong axis, per seed
<!--derived-->

| seed | val margin | SE_boot | tau_eff (2·SE) | routed | outcome | deployed / baseline mean CW | SAFE (dep<=base) |
|---|---:|---:|---:|---|---|---:|:--:|
| 42 | +0.053 | ~0.05–0.07 | ~0.10–0.14 | margin_fallback | SAFE_FALLBACK | 0.117 / 0.117 | yes |
| 43 | +0.017 | 0.072 | 0.144 | margin_fallback | SAFE_FALLBACK | 0.031 / 0.031 | yes |
| 44 | +0.0003 | ~0.05 | ~0.10 | margin_fallback | SAFE_FALLBACK | 0.163 / 0.163 | yes |
| 100 | +0.040 | ~0.05 | ~0.10 | margin_fallback | SAFE_FALLBACK | 0.153 / 0.153 | yes |
| 101 | +0.201 | 0.041 | 0.082 | calibrated | CATCH | 0.134 / 0.240 | yes |
| 102 | −0.096 | (fit fails first) | — | recall_fallback | SAFE_FALLBACK | 0.046 / 0.046 | yes |

**6/6 SAFE** (deployed confident-wrong <= baseline on every seed, verified from the artifact). **Seed 43 now falls back
SAFE** (deployed==baseline, 0.031/0.031 — the regression is eliminated). **Active CATCH on 1/6** (seed 101, the only
seed whose validated edge is a real ~5-sigma signal; deployed 11 vs baseline 24 confident-wrong at headline coverage).
Runner composite verdict: PARTIAL (its rule requires every seed to actively catch; here it is safe-on-all +
active-catch-on-a-subset). regression=0/6.

## The honest trade — catch drops 4/6 → 1/6, and that is correct
<!--derived-->
The parent caught on 42/44/100/101 (4/6) but those catches leaned on TEST-draw edges the guard cannot see. On the only
signal it CAN see without peeking — the validation self-read margin — seeds 42 (+0.053), 44 (+0.0003) and 100 (+0.040)
are within ~1 bootstrap-SE of zero (SE ~0.05), statistically indistinguishable from seed 43's +0.017 that flipped. A
guard that routed them would be gambling on the same coin-flip that produced the seed-43 regression. The conservative
guard correctly refuses all borderline seeds and routes only seed 101 (val margin +0.201 = ~5·SE). This is the
mission-appropriate result: **6/6-SAFE with an active catch on the one unambiguous seed**, NOT a 6/6 catch. Claiming the
floor "catches on 6/6" or even "4/6 safely" would be FALSE; the honest state is safe-on-all, catch-on-1.

## Anti-cheats
<!--derived-->
- **(a) 6/6 SAFE**: verified deployed <= baseline on all 6 seeds incl 43 (mean-CW rate and headline-coverage integer
  counts both). Fallback seeds are deployed==baseline by construction (honesty read frozen to recall).
- **(b) catch reported HONESTLY**: 1/6 active catch, stated plainly; not inflated to the parent's 4/6.
- **(c) tau is NON-PEEKING**: `tau = z·SE_boot` with z fixed a priori and SE_boot from the validation block only
  (seed `seed+900001`); the routing decision is fixed inside `robust_fit_monitor` before the test battery exists, and
  an in-run assertion trips if routing is mutated after the test axis runs. The provenance string in every fit record
  names the derivation and the disjoint val seed. A fixed principled tau=0.08 cross-check routes the identical set
  (only 101) — the result is robust to the tau derivation, not tuned to it.
- **(d) calibrated monitor still routed where routed**: seed 101 routes the calibrated self-read (unchanged mechanism).
- **(e) moat preserved**: additive, default-OFF (`--edge-guard-mode off` reproduces the parent), no `sim/` edit; the
  hard cue-match moat path is untouched (475/475 abstains, foundation 6/6).

## Honest scope
<!--derived-->
Reduced-scope caveats inherited from the parent: fixed monitor→self_schema relay under
STDP/Hebbian/homeostasis/STP/structural/OU DISABLED (isolation of the mechanism); the affect term is a stub; the moat
path is untouched. The guard delivers the owner-defined crux — the honesty floor never makes honesty worse on any of
the 6 seeds — by trading catch for safety on the marginal seeds. It does NOT recover the marginal-seed catches: that
remains the parent's named next mechanism (a monitor with a larger, more STABLE edge — richer ACC/aPFC features or an
ENSEMBLE monitor whose deployed self-read variance is small enough that a held-out guard can confidently route more
than one seed). Until then, the conservative guard's SAFE-fallback IS the floor. "The floor catches on 6/6" is FALSE;
"the floor is 6/6-SAFE" is TRUE and verified.

## Reproduce
```bash
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._stageA_honesty_floor_strengthen_derisk \
  --seeds 42 43 44 100 101 102 --n-trials 300 --n-novel 120 --calib-robust 192 \
  --edge-guard-mode bootstrap --edge-guard-z 2.0 --edge-guard-bootstrap 500 \
  --out research/findings/raw/lanes/stageA/stageA_honesty_strengthen_6seed_edgeguard.json
```
