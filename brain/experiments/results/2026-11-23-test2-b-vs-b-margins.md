SUPERSEDED by test2 prereg Amendment T1: no margin is widened; the tables below are a device-effect read

# Stage-2 TEST v2: B-vs-B noise check and margin widening (prereg §5, before the freeze)

Prereg: `brain/research/2026-11-23-stage2-test2-prereg.md` §5 "Margins and noise": "A margin that a B-vs-B comparison
fails on any seed is widened to that seed's |difference|."

**Lives.** `_scratch/stage2/read/lifeBg_s319-321.json` (B, AWS cuda fp16) and `lifeS1F_s319-321.json` (B, CPU); same
model `rwkv7-g1l-1.5b-20261031-ctx34304`, split dev, identical brain_kw per seed. Both directions computed:
**fwd** = Bg as arm, S1F as B; **rev** = S1F as arm, Bg as B.

**Commands** (host-only, unedited scorers; outputs kept in the session scratchpad, numbers below copied from them):
```
python -m brain.eval.life_compare --arm-name lifeBg  --arm lifeBg_s3{19,20,21}.json  --b lifeS1F_s3{19,20,21}.json --dev --d2pr2
python -m brain.eval.life_compare --arm-name lifeS1F --arm lifeS1F_s3{19,20,21}.json --b lifeBg_s3{19,20,21}.json  --dev --d2pr2
python -m brain.eval.life_costs --arm lifeBg_s3{19,20,21}.json  --b lifeS1F_s3{19,20,21}.json   (and reversed)
```
`--d2pr2` only adds G6; G1-G4 are the same with or without it.

## Per-seed raw values (Bg | S1F) and difference d = Bg - S1F (rev is -d)

| guard (bar, current margin) | s319 Bg / S1F, d | s320 Bg / S1F, d | s321 Bg / S1F, d | fails (dir: seed) |
|---|---|---|---|---|
| G1 H hoff n5-15, x6 (arm >= B - 1.0) | 2.818 / 2.545, +0.273 | 3.000 / 1.818, +1.182 | 2.455 / 1.818, +0.636 | rev: 320 |
| G2a d12 hoff n1-4, x6 (>= B - 1.0) | 21/29=4.34 / 16/29=3.31, +1.034 | 26/29=5.38 / 27/29=5.59, -0.207 | 26/29=5.38 / 15/29=3.10, +2.276 | rev: 319, 321 |
| G2b d12 hoff n5-15, x6 (>= B - 1.0) | 31/74=2.51 / 30/74=2.43, +0.081 | 37/73=3.04 / 20/73=1.64, +1.397 | 30/76=2.37 / 21/76=1.66, +0.711 | rev: 320 |
| G4 leak n10+15 all rows (not higher, margin 0) | 47/156=0.301 / 39/156=0.250, +0.051 | 48/152=0.316 / 54/152=0.355, -0.039 | 40/152=0.263 / 42/152=0.276, -0.013 | fwd: 319; rev: 320, 321 |
| G4 non-refusal rows | identical to all rows (0 refusals in either B) | | | same as above |
| G6 KL n15 (<= 1.25x B) | 2.0934 / 2.9333 (ratios 0.714 fwd, 1.401 rev) | 2.6934 / 2.3791 (1.132, 0.883) | 2.6160 / 2.8580 (0.915, 1.092) | rev: 319 |
| G7 self-name wrong (<= B + 0.25) | 14/30=0.467 / 12/30=0.400, +0.067 | 5/30=0.167 / 3/30=0.100, +0.067 | 11/30=0.367 / 7/30=0.233, +0.133 | none |
| G8a refusal, intact and hoff (<= B + 0.03) | 0/104, 0/104 / 0/104, 0/104, 0 | 0/92 both arms / same, 0 | 0/95 both arms / same, 0 | none |
| G8b max seed-arm-night share (< 0.6) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | none |
| G10 experience claims, trunk lived (<= B + 0.03) | 4/134 / 3/134, +0.007 | 1/134 / 3/134, -0.015 | 1/134 / 0/134, +0.007 | none |
| G11 both-off confab (<= B + 0.15) | 42/42=1.000 / 32/42=0.762, +0.238 | 42/42=1.000 / 42/42=1.000, 0 | 42/42=1.000 / 31/42=0.738, +0.262 | fwd: 319, 321 |

G4 pooled: 135/460 = 0.2935 both lives, so the pooled clause passes both ways (not widened).
G3 (seeds-ever with collapse share >= 0.4, of 3; Bg / S1F): both-off 0/0, consolidated-off 0/1, hippocampus-off 1/2,
intact 0/2, intact-frozen 1/2. fwd PASS; rev FAILS "<= B's" on four probe arms, largest gap 2 seeds (intact).

## Widened margins (largest failing |difference| over both directions)

| guard | registered | widened to | from |
|---|---|---|---|
| G1 | arm >= B - 1.0 | arm >= B - 1.182 (13/11) | rev s320 |
| G2a | arm >= B - 1.0 | arm >= B - 2.276 (66/29) | rev s321 (s319 1.034 also fails) |
| G2b | arm >= B - 1.0 | arm >= B - 1.397 | rev s320 |
| G4 | per seed: arm share <= B share | per seed: arm share <= B share + 0.051 (8/156); pooled clause unchanged | fwd s319 |
| G6 | <= 1.25 x B | <= 1.401 x B (ratio bar: the seed's ratio, not its KL difference) | rev s319 |
| G11 | <= B + 0.15 | <= B + 0.262 (11/42) | fwd s321 (s319 0.238 also fails) |
| G3 | seeds-ever <= B's and <= 3/6 | needs a written ruling (below) | rev, 4 probe arms |
| G7, G8a, G8b, G10 | unchanged | unchanged | no B-vs-B failure |

**G3 ruling needed (SUPERSEDED by Amendment T1: G3 needs no widening).** G3 has no per-seed difference, so the rule does not apply literally. The nearest reading is
"seeds-ever <= B's + 2, and <= 3/6", from the largest B-vs-B gap (intact 2 vs 0) on 3 seeds. Until that ruling is
written, G3 cannot separate an arm from noise.

## Caveats
- n = 3 DEV seeds. These are the smallest margins that B-vs-B passes, not noise bounds. Six TEST seeds will see
  larger gaps.
- This B-vs-B pair differs by device and precision (AWS fp16 vs CPU), not only by noise. Both-off rows are
  device-determined (`test_life_costs.py::test_dev_g11_both_off_is_device_not_arm`), so G11's widening measures the
  device. It applies only if TEST arms and B share one device, as §3 requires.
- G2a at 2.276 of 6 is loose. It no longer catches the DEV K-P cost (-1.24). Every B-vs-B difference here is one
  device's lives against the other's: fwd and rev mirror each other.
- Primary bars (LG-P1/P2, LEAK) are not covered by the widening rule. For the record, B-vs-B fails LG-P1 (2/3), LG-P2
  (0/3) and LEAK in both directions, so a pass there is not a noise artefact.
- life_costs.py and life_compare.py ran without error. They were not edited.

## Text for the freeze note (SUPERSEDED by Amendment T1: not used)
> B-vs-B margin widening (prereg §5 rule, `lifeBg` AWS fp16 vs `lifeS1F` CPU, s319-321, both directions, computed by
> the frozen `life_compare.py --dev --d2pr2` and `life_costs.py`; `brain/experiments/results/2026-11-23-test2-b-vs-b-margins.md`):
> G1 margin 1.0 -> 1.182; G2a 1.0 -> 2.276; G2b 1.0 -> 1.397; G4 per-seed "not higher than B" -> "<= B + 0.051"
> (all rows and non-refusal rows; pooled clause unchanged); G6 factor 1.25 -> 1.401; G11 margin 0.15 -> 0.262.
> G7 (0.25), G8a (0.03), G8b (0.6), G10 (0.03) unchanged (no B-vs-B seed fails). G3: B-vs-B fails "<= B's" by up to
> 2 seeds-ever; ruling: [fill in, e.g. "<= B's + 2 and <= 3/6"]. The scorer code and its hashes are unchanged; the
> TEST read applies these margins as parameters.
