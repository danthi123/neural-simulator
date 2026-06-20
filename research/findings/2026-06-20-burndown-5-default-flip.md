# Burndown #5 PRODUCTION-FLIP — the shipped 320 conversation defaults to the ON-BRIDGE neural read-out normalization (2026-06-20)

## Verdict: GREEN (default flipped to the neural read-out; == the curated/host who/what baseline; moat 0 false-accepts, multi-seed). No `sim/` edit; no re-gen needed.

Burndown #5 (`fb90b8a5`, `2026-06-20-burndown-5-ppmi-norm-onbridge.md`) proved the on-bridge NEURAL read-out
normalization (per-hub spike-frequency ADAPTATION + per-concept FEEDFORWARD INHIBITION, POST-f-I — the CYCLE-93b
prescription) reproduces the host `double_center` EXACTLY through the real who/what + moat pipeline, but left the
production path on the host codes. This closure **flips the production default to the neural-norm codes** and
regression-guards it.

## Cost determination (cheap-first) — ALREADY PAID, no re-gen run

`--readout-norm neural` (in the cortex code-gen runner `_phaseB_onbridge_stream_conversation_derisk.py:116`) is a
FULL GPU stream re-derivation of the cortex codes (it re-streams the corpus when run without `--codes-npy`,
`neural_norm(L, ...)` at line 118, ~9 min/seed). It is NOT a quick re-apply.

**But the cost was already paid:** the neural-norm 320 codes for ALL THREE seeds already exist on disk from the
2026-06-17 production GO:
```
research/findings/raw/_phaseB_stream_codes_320_neural_seed42.npy   (2026-06-17)
research/findings/raw/_phaseB_stream_codes_320_neural_seed43.npy   (2026-06-17)
research/findings/raw/_phaseB_stream_codes_320_neural_seed44.npy   (2026-06-17)
```
These were the basis of `2026-06-17-consolidated-320-production-conversation-GO.md`. The cached codes are local
research artifacts (not committed — same as the host codes; `~750 KB` each, `.npy` untracked by design). So **no
GPU stream re-train was needed** — the flip is a default + guard change on the existing neural codes.

For reproducibility, the exact re-gen command (if the controller ever wants to regenerate them — NOT required for
this closure) is:
```bash
SIM_BACKEND=cupy python -m research.runners._phaseB_onbridge_stream_conversation_derisk \
    --taxonomy 40x8 --seeds 42 --readout-norm neural \
    --codes-npy research/findings/raw/_phaseB_stream_codes_320_neural_seedSEED.npy \
    --out research/findings/raw/_phaseB_onbridge_stream_conversation_neural_s42.json
# (repeat --seeds 43 / 44; "SEED" in --codes-npy is the literal placeholder the runner substitutes.)
# WITHOUT --codes-npy it forces a fresh stream; WITH it + an existing file it reloads (instant).
```

## The flip

The demo `research/runners/consolidated_320_conversation_demo.py` `--readout` argument **already defaulted to
`neural`** (line 201) — but the help text claimed neural was "seed 42 only so far" (stale: 43/44 exist + were the
2026-06-17 GO basis) and there was NO CI guard on the neural path (only the host path was guarded). This closure:

1. **Corrects the demo `--readout` help** to state the neural read-out IS the production default (burndown #5,
   per-hub adaptation + per-concept feedforward inhibition, seeds 42/43/44 == host with moat 0-FA); `host` is the
   escape / test-oracle path.
2. **Adds the neural-path CI guard** (`tests/test_consolidated_320_conversation.py::
   test_production_agent_on_neural_readout_codes_seed42`) alongside the retained host guard — the new production
   default is now regression-guarded with the SAME load-bearing assertions (0 false-accepts, recall 1.00, the full
   who/what + yes/no + describe + elaborate turn).

The host-norm path stays fully available (`--readout host`, the escape) and the demo loads whichever cached codes
the chosen read-out names.

## Verification — neural default == the curated/host baseline, moat 0-FA, multi-seed

Through the EXACT production who/what + moat pipeline (`run_seed`, the `rf` numpy-CPU test-oracle composer), all 3
seeds, host vs neural side by side:

| read-out | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| **host** (escape) | GO: recall 1.00, abstain 1.00, **FA 0** | GO: 1.00 / 1.00 / 0 | GO: 1.00 / 1.00 / 0 |
| **neural** (PRODUCTION DEFAULT) | GO: recall 1.00, abstain 1.00, **FA 0** | GO: 1.00 / 1.00 / 0 | GO: 1.00 / 1.00 / 0 |

Every neural seed is GO == the host baseline: recall 1.00, abstain 1.00, **0 false-accepts**, yes/no + describe +
elaborate all correct. The grounded-code phase-cosine structure is essentially equal (neural mean +0.130/+0.131/
+0.135 vs host +0.115/+0.100/+0.114 over the demo's used words) — i.e. the documented #5 margin cost (gap +0.401
vs host +0.416) does **NOT** push any false-accept at the conversational level. The moat held on the production
default with no loosening of the gate (the gate is the a-priori learned/relational moat; 0-FA is an assertion).

CI (both guards GREEN, CPU/numpy):
- `tests/test_consolidated_320_conversation.py` — 2 tests (host-escape + neural-default), both PASS.
- `tests/test_ppmi_readout_norm_conversation.py` — 11 tests (the #5 mechanism guard), all PASS (unchanged).

## Honest scope

- The flip is a **default + guard** change on the EXISTING neural codes; no GPU re-train was run (the codes were
  already cached from 2026-06-17). The cached `.npy` codes (host AND neural) are local research artifacts, not
  committed — the production "default" is the demo's `--readout` flag + the guarded code path, not a checked-in
  blob.
- The neural codes carry the documented small margin cost (+0.401 vs host +0.416 from #5); confirmed here NOT to
  cost any answer or any false-accept at 320-scale across 3 seeds. Had the lower margin leaked a false-accept that
  would have been an honest finding (report, do NOT loosen the gate) — it did not.
- The on-bridge **circuit build** (the literal per-concept FS-feedforward-inhibition + per-hub-adaptation circuit
  at read-out) remains the faithful-realization follow-on noted in #5; `neural_norm` is its validated
  specification. This closure is the production-default flip, not that circuit.

## Files (PATHSPEC, `main`)

- `research/runners/consolidated_320_conversation_demo.py` — `--readout` help corrected (neural = production
  default; host = escape). Default was already `neural`.
- `tests/test_consolidated_320_conversation.py` — added the neural-path CI guard alongside the retained host
  guard (refactored the shared run/assert helpers; both skip gracefully if their codes cache is absent).
- This finding.

Stayed on `main`; PATHSPEC commit (demo help + test + this doc only); touched ONLY the demo/readout-norm/codes
path (no `sim/` edit, no sequencer/composer/nav edit); the no-confab moat held at 0 false-accepts on the neural
production default.
