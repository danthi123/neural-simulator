# Merged nav-critic value-train — root-cause + build-plan (CYCLE 210 controller prep)

**Status:** CONTROLLER ROOT-CAUSE + BUILD-PLAN, written while the value-train de-risk
subagent (`a8e9a3a28427c3507`) is in flight. This is the controller's code-reading
diagnostic; it is COMPLEMENTARY to the subagent's empirical operating-point measurement,
not a substitute. Load-bearing claims to be confirmed against the subagent's GPU result
(trust-but-verify) before any build.

## The problem being solved

The full nav reward/critic ("limbic core": spiking `reward_us` US-afferent → `snc`
DOPAMINE neuron, value `V` subtracted at the SNc by the GABAergic `striosome_value`
MSN-D1 critic) is LIFTED onto the merged "one brain" (`co_resident_nav_critic`, CYCLE
209): it builds clean (45 regions), the SNc f-I is restored, and the no-confab moat is
unperturbed (MOAT GO). The remaining piece is the **value-train**: the critic LEARNING
`V` so the dopamine signal is the graded reward-prediction-error `δ = r − V` (the
separable LEARNING increment of the Schultz RPE).

The blocker observed at CYCLE 209: the SNc **saturates** at the homeostasis-boosted rate,
so the GABA_B value subtraction can't grade δ (gap ~1.05 — correct DIRECTION pred<unpred,
but weak).

## Root cause (high confidence, from the nav runner's validated recipe)

`nav_conv_merged_bridge.py:524-526` (my CYCLE-209 edit) force per-region homeostasis onto
**three** regions — `snc`, `reward_us`, AND `striosome_value`:

```python
for _r in nav_regions:
    if _r.name in ("snc", "reward_us", "striosome_value"):
        _r.enable_homeostasis = True
```

This was correct for the **burst gate** (it restored the SNc reward-burst: 446 Hz / 5.47×
vs the un-boosted 111 Hz / 3.53×) — that gate is already passed + committed. But it is
**wrong for the value-train**: forcing homeostasis onto the SNc drives it to ~446 Hz =
near-saturation, leaving no headroom for the critic's GABA_B inhibition to grade it down
by `V`. The two gates want different SNc operating points.

The validated nav recipe (`g11_bg_runner.py:build_bg_brain_regions`) never homeostasis-
boosts the SNc or `reward_us`:
- `reward_us` (`:1158-1163`): NO homeostasis (RS-pyramidal relay, driven by perceived reward).
- `snc` (`:1133-1142`): NO homeostasis (stays near vpeak; fires from `reward_us` excitation,
  disinhibited by the graded critic).
- `striosome_value` (`:1230`, `:1252`, `:1279`): homeostasis ONLY if `enable_critic_homeostasis`.
- `place` self-org pool (`:1188-1190`, `:1216`): NO homeostasis (anti-cheat: fires from the
  LEARNED current, not a threshold collapse).

The un-boosted SNc's "broken-looking" 111 Hz is ~25% of the saturated 446 Hz — well below
saturation, so it is the **GOOD non-saturated operating point for a graded value-train**
(it still bursts: 3.53× tonic ratio crosses the DA threshold for LTP gating). The "broken"
label was relative to the burst-gate target, not an absolute failure.

## The other controller find — DO NOT use the legacy `_run_critic_warmup`

`g11_bg_runner.py:5780-5784` documents that the **legacy** `_run_critic_warmup`
(vs_place_context afferent) hit an *unresolved nav-bridge blocker even standalone*: the
MSN-D1 critic plateaus at ~−79.6 mV (vs −71 mV in the byte-identical CPU de-risk on the
same `g_exc`) and would not fire even at 10× drive. The merged value-train must therefore
ride the **N9 self-org path**, not the legacy warmup:
- `_run_place_selforg()` (`:5843`) — STEP-1: self-organize the spiking place fields, then freeze.
- `_run_place_value_training()` (`:5857`) — STEP-2: pair-then-reward DA-gated STDP grows
  `place → striosome_value` `V` (`vs_place_to_value_weight=0.2` → ~0.58), then freeze.
- `_run_stage_b_smoke()` (`:5861`) — the load-bearing critic gate = the RPE battery itself:
  LEARNS-V / CRITIC FIRE+GRADE / GABA_B gap + lesion.

One more nav detail the merged path must replicate: the DA production-rule threshold is
**calibrated** to the SNc's measured tonic firing fraction (~0.02) during value-training so
a reward burst → DA>baseline → LTP, then restored to the nav's 0.30 (`:5739-5779`,
`:5833-5837`). CYCLE-209 set the merged dopamine modulator threshold to 0.0 (neutral-at-
rest); the value-train needs the same measure-then-calibrate step.

## Build plan — GO branch (if the subagent confirms a graded operating point co-resident)

In `build_merged_nav_conv_bridge`'s `co_resident_nav_critic` block:
1. REMOVE the `snc`/`reward_us` entries from the lines 524-526 homeostasis force (keep the
   burst-gate path available behind a flag if needed, but the value-train build must not
   saturate the SNc). Leave `striosome_value` homeostasis to `enable_critic_homeostasis`.
2. Pass `enable_critic_fs_inhibition=True` + `critic_fs_weight=16` to `build_bg_brain_regions`
   (the spiking critic rate-clamp — "root fix vs GIRK cap"; "critic 126→8 Hz physiological,
   gap 3.75× graded", `g11_bg_runner.py:394-396`).
3. Pass `neural_place_selforg=True` so the value-train rides the validated self-org path
   (NOT the legacy −79.6 mV warmup).
4. Drive the merged value-train through STEP-1 self-org → STEP-2 value-train → the
   `stage_b_smoke` RPE battery, with the DA-threshold calibrate-then-restore.

## Build plan — BOUNDARY branch (if no graded point exists co-resident)

If even the un-boosted SNc cannot grade δ co-resident (the merged-config transmission
factor caps it), document the honest boundary (the co-resident value-train is operating-
point-bounded by the merged-config transmission factor) and PIVOT to a shared-system
consolidation that does NOT hinge on fine GABA_B dynamics — #4 fully-spiking motor read-out
(`enable_spiking_wta_readout`) or #5 neural place code — both already de-risked by the
CYCLE-208 per-region homeostasis enabler. This matches the CYCLE-209 EXACT-NEXT option (b).

## Anti-cheat / discipline notes

- The fix is NO new `sim/` edit — it reuses the committed per-region homeostasis mask
  (CYCLE 208) + the validated nav `enable_critic_homeostasis` / `enable_critic_fs_inhibition`
  builder flags.
- `stage_b_smoke` already carries the lesion anti-cheat (sever the critic → δ collapses).
- The no-confab moat must stay GO across the value-train (the dopamine `scope=all` broadcast
  must not perturb the frozen conversational slice — already validated MOAT GO at CYCLE 209).
- 6-seed for the variable δ-gap effect; 3 clean for the mechanistic lesion.
