# Verdict: merge-review-pi-203-frozen-synapse-freeze

Status: I've read the packet and the full `sim/` diff against main, and run the tests. 15 of 15 pass, including the pre-fix byte-identity test. I'm now writing the verdict.

**Review of `research/pi-203-frozen-synapse-freeze` (3 commits off main `710883be5`)**

**What I checked**
- **Default-OFF gating.** `sim/config.py` adds `enforce_plastic_mask_beyond_hebbian: bool = False`. Every new branch in `sim/bridge.py` runs only when `_plastic_mask_enforced(...)` is true. That is true if the env var `BRAIN_ENFORCE_PLASTIC_MASK` is truthy or the config flag is set.
- **Helper return values when OFF.** `_live_plastic_mask` and `_plastic_mask_for_remap` return `None`, so the old code paths run unchanged. The remap site used to read `cp_plastic_mask`. I grepped `sim/` and nothing assigns that name, so the old remap branch was dead code and returning `None` matches it.
- **The six write sites.** I read the code for synaptic scaling and its clip, the reward update and its clip, structural elimination, the mask remap, compaction, and formation labelling. Each is masked correctly.
- **The length hazards from the second-opinion passes.** Compaction pairs only the overlapping prefix, and the remap clamps to the shorter array. Formation grows the mask with `fill=False` and then sets the new entries to `True`. Those fixes are sound, and the invariant that justifies the prefix pairing is documented in the code.
- **Tests.** Run with the main checkout's venv (this worktree has no `.venv`): `tests/test_plastic_mask_beyond_hebbian.py` and `tests/test_enforce_plastic_mask.py` gave 15 passed.
- **Evidence files.**
  - `byte_identity_record.json` shows 36 of 36 comparisons identical, with no mismatches, on six seeds and six rules. The reference is the git-archive of `710883be5`.
  - `comprehension_no_regression.json` shows 6 of 6 seeds identical and 0 reply changes. The branch reports this as a negative, not as a pass.
- **No default flip.** The finding and `GAP_CLOSURE_MISSION.md` say the default flip stays owner-gated, which matches the code.
- **`FAILURE_LOG.md`.** The new entry names a gate, `tests/test_plastic_mask_beyond_hebbian.py`.

**Caveats and non-blocking notes**
1. **Shared env var.** `BRAIN_ENFORCE_PLASTIC_MASK` now turns on both the old Hebbian enforcement and the new six-site enforcement. Anyone who sets it for the Hebbian fix alone gets the wider behaviour. Nothing in `sim/` or the runners sets it outside tests and the pi203 runners, so there is no production impact. The doc comments say so.
2. **RNG trajectory.** With the flag ON, structural elimination can skip a random draw, so ON and OFF runs have different RNG trajectories. The code comment records this as by design. It does not affect OFF.
3. **Evidence is CPU/numpy only.** The byte-identity battery ran on the numpy backend, and the comprehension arm ran on CPU. The new code uses only backend-agnostic `cp.where`, `cp.clip` and indexing, so I see no cupy risk. A cupy confirmation is still worth doing before anyone flips the default.
4. **Reply-level result.** The flag changes the comprehension reply zero times. So this is a verified correctness fix for the write sites, not a load-bearing or reply-changing faculty. The branch does not claim more than that.
5. **Clutter.** `fix.diff`, `fix_final.diff`, `fix_full.diff` and `branch_vs_main.diff` are duplicate raw artifacts. It's harmless but untidy. Unrelated churn also rides along in `tool_health.json` and `lane_builds.jsonl`.

**Conditions for the driver**
- Merge as-is, with the flag staying OFF by default.
- Don't claim a capability or reply-level gain from this merge.
- Keep the owner-gated flip and the reply-reach follow-up (which write paths reach that organ's read) on the board.
- Push with `bash tools/push_both.sh`, and don't use `--no-verify`.

VERDICT: APPROVE
