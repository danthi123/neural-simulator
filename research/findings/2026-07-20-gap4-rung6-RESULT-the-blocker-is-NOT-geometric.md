# gap#4 RUNG 6 — RESULT: the geometric hypothesis is FALSIFIED. The blocker is NOT the spacing/shift collision.

Pre-registered at `b58c504d` before the run, seeds 1000-1005, both arms on a 40-bin track so **spacing was the
single variable**.

## The band arm is UNMEASURABLE — and I am not reading anything from it

`MAIN_ruleON` returns `dw = 0` or `map_ok = 0` on essentially every seed in both arms. The band parameters were
derived from the **20-bin** track's eligibility distribution and do not transfer to a 40-bin lap (8000 steps rather
than 4000). **P2 is therefore UNMEASURABLE, not falsified-by-separation**, and its `c_adj` values (1.000, 0.285,
0.143, 0.098) are degenerate artifacts of an unlearned read-out. Recording that explicitly so they are never mined
later as evidence in either direction.

## But the CONTROL arms test the hypothesis directly — and refute it

`P4_ruleOFF` is plain BTSP with no band. It is **valid in both arms** (`map_ok = 1`, real `dw`), and identical
across all six seeds:

| arm | spacing vs shift(4-6) | **adjacent contrast** | far contrast | dw |
|---|---|---|---|---|
| A | 4 — **collision** | **1.449** | 3.225 | -1262 |
| B | 8 — **no collision** | **1.227** | n/a | -632.8 |

**Wider spacing gives LOWER adjacent contrast, not higher.** The geometric hypothesis predicted the opposite: if
the deficit were caused by "the adjacent field" and "where this field forms" occupying the same lag, then separating
them in lag (spacing 8 > shift 4-6) should have IMPROVED adjacent contrast. It made it slightly worse.

⇒ **P2's hypothesis is falsified by the control arms.** The blocker is not the spacing/shift collision.

**P1 also fails:** arm A was predicted to reproduce the deficit at <= 1.35x and reads **1.449**, so the 40-bin track
does not quantitatively reproduce the 20-bin phenomenon (1.213x) either. Both failures point the same way — the
adjacent-contrast deficit is not a function of the geometry I hypothesized it was.

## Confounds I am disclosing rather than burying

- The two arms have different cell counts (10 vs 5), an unavoidable consequence of fixing track length while varying
  spacing. So "adjacent" means 4 bins in arm A and 8 bins in arm B — the comparison is *each field against its own
  nearest neighbour*, which is the right question, but the populations differ in size.
- `c_far` is `nan` in arm B (with 5 cells and target_cell=1, the far set is degenerate), so the far-field leg is
  unavailable there and only the adjacent comparison is usable.
- On the 40-bin track plain BTSP's `dw` is NEGATIVE in both arms (-1262, -632.8) where the 20-bin track gave +445.5.
  A doubled lap accumulates far more depression. **This is a real, unexplained change in the baseline** and it means
  the 40-bin results are not directly comparable to the 20-bin ones in absolute terms — only the A-vs-B contrast,
  which is what the design isolates, is safe to read.

## The cap fires

The pre-registration stated: *"One geometry. If P2 fails I do not try a third spacing — the verdict becomes that the
blocker is not geometric, and the remaining route is the non-local instructive signal."*

**Invoked.** No third spacing.

## Where this leaves gap#4 — the honest closing position

**SEVEN routes now closed, seven distinct causes.** All six local-rule families, plus the geometric route:

| route | cause |
|---|---|
| split-threshold band | placement destroys field formation (2 attempts, cap) |
| zero-DC DoG | trace-amplitude mismatch |
| two-sigmoid on `ET*IS` | no separating axis (1.001x) |
| Miller-MacKay (both forms) | hard floor absorbs negative mass; `w_min<0` breaks Dale |
| expansive read-out | metric inflation |
| rank-based STC capture | rank monotone in a non-separating magnitude |
| **geometric separation** | **wider spacing does not improve adjacent contrast — refuted** |

**The measured core stands unchanged and is the arc's most robust object:** adjacent contrast is deficient relative
to far contrast, reproduced on fresh seeds across four independent runs and two track lengths.

**The single remaining named route** is the one the second gate ranked fourth and which no experiment here has
touched: **a non-local instructive signal — feedback inhibition gating plateau probability (Milstein's own
answer)**. It requires a task rewrite rather than a rule change, which is precisely why it was deferred, and it is
now the only candidate not eliminated. That is a well-posed next arc, not a wall.

---

## REGIME CHECK — I tested the threat to my own falsification, and it half-holds

I recorded above that plain BTSP's `dw` is "NEGATIVE in both arms (-1262, -632.8) where the 20-bin track gave
+445.5". **That statement conflates two quantities and is corrected here:** those are *stage-2* `dw` from `one_run`
(the L2 read learning), not the stage-1 lap. Measuring the stage-1 lap directly:

| | 20-bin | 40-bin |
|---|---|---|
| total stage-1 dw | **+205,969** | **+418,724** — positive in BOTH |
| synapses increased | 5039 | 12,856 |
| synapses decreased | 3281 | 19,784 |
| **majority sign** | **UP** | **DOWN — flips** |
| at `w_min` floor | 2880 / 8320 (34.6%) | 13,680 / 32,640 (**41.9%**) |

**So the longer lap IS measurably more depression-dominated** — the majority of synapses now decrease, and floor
occupancy rises 7 points — even though the total remains positive because the increases are larger.

### What this does and does not do to the rung-6 falsification

**It does NOT invalidate it.** The comparison was A-vs-B *within* the 40-bin track, one variable, same regime for
both arms: spacing 8 gives adjacent contrast 1.227 against spacing 4's 1.449. The geometric hypothesis predicted
the reverse. That is a valid falsification **of the hypothesis as tested**.

**It DOES bound it.** The 40-bin regime differs measurably from the 20-bin one in which the deficit was originally
characterized. So the precise claim is: **the geometric hypothesis is falsified in the only regime where it is
testable at all** — testing it *requires* spacing > shift, which *requires* the longer track, which is *itself*
the regime shift. That circularity is unavoidable given the substrate, and I am stating it rather than letting the
falsification read as broader than it is.

**Corrected verdict:** the blocker is not the spacing/shift collision, established within the regime where the
question can be posed. A residual possibility survives and is named: that widening the spacing helps at 20 bins but
the confound of lengthening the track masks it. Testing that would need a way to widen spacing without lengthening
the lap — which this task geometry does not admit. Recorded as an honest limitation, not resolved.
