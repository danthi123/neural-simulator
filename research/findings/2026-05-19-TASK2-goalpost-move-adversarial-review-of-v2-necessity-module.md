# TASK-2 — Independent goalpost-move adversarial review of the v2 necessity module

**Reviewer role:** dedicated, independent, zero-prior-context goalpost-move
adversary. No files modified. One findings note, committed (no push).
**Subject:** `research/runners/integrated_loop_core_v2.py` (+ matrix
`tests/test_integrated_loop_core_v2.py`), commit `36a7975`, claimed to
SUPERSEDE the necessity HYPOTHESIS of the original frozen module
`research/runners/integrated_loop_core.py` (`2048750`) via EXACTLY ONE
biologically-cited partition correction: `no_cls_replay` moves from the
episodic-helper set to the working-memory/concept-helper set.

**The decisive question:** is moving `no_cls_replay` from `_HELPER_EP` to
`_HELPER_WM` a LEGITIMATE catalog-derived correction of a genuinely
falsified pre-registered prediction, or a RATIONALIZED repartition
engineered to let a distinct-readout-pathways candidate clear the duty
that VOIDed every prior attempt?

---

## The 6 mandated checks (file:line evidence + independent judgment)

### 1. Bars verbatim — **SOUND**

`integrated_loop_core_v2.py:125-130`:
`_ILV2_LADDER=(2,4,8)`, `_ILV2_V1_MIN=0.90`, `_ILV2_SCI_MIN=0.80`,
`_ILV2_LESION_MAX=0.40`, `_ILV2_SCALE_TOL=0.10`, `_ILV2_MIN_SEEDS=3`.
Original `integrated_loop_core.py:52-57`: `_IL_LADDER=(2,4,8)`,
`_IL_V1_MIN=0.90`, `_IL_SCI_MIN=0.80`, `_IL_LESION_MAX=0.40`,
`_IL_SCALE_TOL=0.10`, `_IL_MIN_SEEDS=3`. Value-identical, six for six.
The a-priori justification block (`v2:21-51`) is copied verbatim from
`orig:19-47` with only the symbol prefix changed; no rationale was
softened or re-derived from a result. Drift pin
`tests/test_integrated_loop_core_v2.py:183-191` (test_17) asserts each
`_ILV2_*` byte-equal to both its `_IL_*` counterpart AND the literal
pre-registered constant; it passes. No softening anywhere.

### 2. Exactly one change — **SOUND**

`v2:140-143`: `_ILV2_SHARED=("no_binding","no_shared_clock",
"no_hippo_store")`; `_ILV2_HELPER_WM=("no_bg_gate","no_cls_replay")`;
`_ILV2_HELPER_EP=("no_sequencing",)`; `_ILV2_HELPER_BOTH=
("no_neuromod_timing",)`.
Original `orig:62-65`: `_SHARED=("no_binding","no_shared_clock",
"no_hippo_store")`; `_HELPER_WM=("no_bg_gate",)`;
`_HELPER_EP=("no_sequencing","no_cls_replay")`;
`_HELPER_BOTH=("no_neuromod_timing",)`.
Diff exactly: `_ILV2_SHARED`==`_SHARED`; `_ILV2_HELPER_BOTH`==
`_HELPER_BOTH`; `_ILV2_HELPER_WM`==`_HELPER_WM` ∪ {no_cls_replay};
`_ILV2_HELPER_EP`==`_HELPER_EP` − {no_cls_replay}. Total lesion-name
set identical (the authoritative 7: 3 shared + 2 wm + 1 ep + 1 both;
`v1`/`full` are not lesions). test_18 (`v2 test:194-227`) pins the
symmetric difference on each affected helper set to exactly
{"no_cls_replay"} and the full union equal to the original union; it
passes. No other membership moved; no lesion added or removed.

### 3. Verdict LOGIC not weakened — **SOUND**

Side-by-side, `integrated_loop_verdict_v2` (`v2:173-315`) vs
`integrated_loop_verdict` (`orig:94-235`): the two function bodies are
token-identical except every `_IL_*` symbol is renamed `_ILV2_*` and
the docstring/error strings are unchanged in meaning. Same
instrument-validity-FIRST ordering (`v2:218` / `orig:139`); same
top-level guards (non-empty list, orderable-by-N, integer-coercible N,
ladder==pre-registered) returning `void(...)` not raising
(`v2:203-216` ≡ `orig:124-137`); same per-rung seed/v1/full/lesions
soundness then the SHARED-or-BOTH "collapse BOTH" branch
(`v2:244-253` ≡ `orig:165-173`), the HELPER_WM "collapse wm" branch
(`v2:254-261` ≡ `orig:174-181`), the HELPER_EP "collapse ep" branch
(`v2:262-269` ≡ `orig:182-189`); identical `full_min` accumulation and
the identical PASS / WORKS-SMALL(FAIL) / FAIL precedence with the same
`all_science_ok` / `monotone` / `top_ok` predicates
(`v2:278-315` ≡ `orig:198-235`). `void()` returns
`instrument_valid=False, classification="VOID"` identically
(`v2:198-201` ≡ `orig:119-122`). malformed/non-numeric/NaN/too-few-
seeds → VOID-not-raise preserved (`_num`/`_pair` `v2:149-170` ≡
`orig:70-91`; bool excluded as a number). The ONLY behavioral
difference is which partition tuple a lesion name is tested against —
exactly the single intended change, no hidden logic relaxation.

### 4. Original untouched — **SOUND**

`git diff 2048750..HEAD -- research/runners/integrated_loop_core.py`
returns empty (byte-unchanged since `2048750`).
`git show --stat 36a7975` = exactly 2 files (`integrated_loop_core_v2.py`
+315, `test_integrated_loop_core_v2.py` +227), 542 insertions, 0
deletions — the original module and its test are not in the commit.
`tests/test_integrated_loop_core.py` still has 16 `def test_` and runs
16/16 green; `tests/test_integrated_loop_core_v2.py` 18/18 green
(34 passed total). The original's prior VOID record is preserved
unedited; the finding chain
(`2026-05-19-THIRD-convergent-...`, `...phase-factored-VOID-by-
construction...`, `...PROGRAM-LEVEL-encode-order-conflict...`,
`...CONTROLLER-precommitted-honesty-ceiling...`) explicitly keeps the
original "cannot conclude" as the permanent honest record that the
original pre-registered prediction was falsified — not deleted, not
superseded-in-place.

### 5. Biology independence (the heart) — **SOUND**

Quoted docstring justification, `v2:55-76`:
> "Episodic-sequence ORDER is a property of the ONLINE hippocampal
> trisynaptic pattern-completion path: the entorhinal -> dentate ->
> CA3 -> CA1 trisynaptic loop performs pattern separation and pattern
> completion that recover the serial order written online
> (reference-catalog D.03 ... D.12 pattern separation, Kandel 6e Ch 54
> pp 1357-1360; D.13 pattern completion, Kandel 6e Ch 54 pp 1342,
> 1360-1361, Marr 1971; the project's validated
> validate_trisynaptic_loop.py). The order-INVARIANT neocortical
> concept/schema representation that the working-memory readout reads
> is NOT built online: it is built by the OFFLINE
> complementary-learning-systems consolidation system, which replays
> interleaved (shuffled, order-destroying) experience into neocortex
> (McClelland 1995; Buzsaki 2013; the project's validated Phase-1.3
> consolidation, 3/3 strict anti-cheat multi-seed). Therefore the
> consolidation/replay lesion (no_cls_replay) is necessary for the
> WORKING-MEMORY/concept readout ... and is NOT necessary for the
> EPISODIC-ORDER readout ..."

This is a conclusion a neuroscientist reaches from the catalog ALONE.
The two premises are independently grounded in the project's OWN
validated assets, with no reference to any candidate: (a) the
hippocampal trisynaptic loop performs pattern separation (D.12) and
pattern completion (D.13, Marr 1971) — `CLAUDE.md:726-762`,
`validate_trisynaptic_loop.py`, D.12 PASS / D.13 PASS multi-seed; (b)
complementary-learning-systems consolidation transfers an
order-invariant trace into neocortex via interleaved/shuffled replay —
`CLAUDE.md:2386-2401` Phase-1.3 (McClelland 1995 / Buzsaki 2013),
hippo-OFF retention 94%, 3/3 strict anti-cheat multi-seed. From (a)+(b)
alone it follows directly that removing the offline replay system
collapses the order-invariant neocortical concept/WM read and NOT the
online-trisynaptic-served episodic-order read — the standard CLS
division of labor, derivable with zero knowledge of any architecture's
pass condition.

Decisively, the same correction is FORCED independently from THREE
convergent faithful GPU-verified negatives (each reached BEFORE the v2
module existed): (1) single online pass — concept binding needs
shuffled order, episodic store needs fixed order: contradictory
(`...PROGRAM-LEVEL-encode-order-conflict...`); (2) phase-factored —
shuffled offline replay needed for concept selectivity destroys
episodic order while skipping consolidation preserves it, inverting the
original's no_cls_replay→episodic duty
(`...phase-factored-VOID-by-construction...:41-69`); (3)
distinct-readout-pathways design — episodic order from the
order-preserving trisynaptic path, concept/WM from the order-invariant
neocortical schema: the same refutation from the opposite direction
(`...THIRD-convergent-...:17-54`). Three independent directions, one
conclusion: strong evidence of legitimacy over convenience. The
docstring's argument does NOT smuggle candidate-rescue reasoning: it
cites only catalog entries and validated subsystems; the
candidate/pre-committed-bound material is quarantined in docstring
section (iv) (`v2:98-118`) as an explicit AFTER-THE-FACT bound, never
offered as SUPPORT for the membership choice (sections (i)-(iii),
`v2:55-97`, justify the move purely from biology + the falsification
record, and (iii) states verbatim it "is NOT derived from what makes
any candidate architecture pass"). I cite none of "it makes the
candidate pass" / "it is needed for the architecture" as support; I do
not need to — the biology is independently sufficient and thrice-forced.

### 6. No code beyond the 2 new files; no autograd; stdlib+typing only — **SOUND**

`v2:121-123`: only `from __future__ import annotations`, `import math`,
`from typing import Dict`. No `torch`, no autograd, no numpy/cupy import
(docstring `v2:18-19` asserts this; verified by import scan). It does
NOT import `integrated_loop_core` (the 3 textual hits for that name in
v2 are docstring/comment prose asserting non-import; there is no
`from research...`/`import` statement). `git show --stat 36a7975`
confirms the commit touches exactly the two new files, +542 / -0.

---

## Decisive biology-independence judgment (3-4 sentences)

The correction is independently reachable from the project's own
catalog: the hippocampal trisynaptic loop owns serial-order recovery
via pattern separation/completion (D.12/D.13, Marr 1971, validated), and
CLS consolidation builds the order-invariant neocortical schema via
shuffled offline replay (McClelland 1995 / Buzsaki 2013, Phase-1.3
validated 3/3 strict anti-cheat) — from these two validated premises a
neuroscientist concludes, with no knowledge of any candidate, that
removing the replay/consolidation system collapses concept/WM and not
episodic-order recall. The same single correction is moreover FORCED
independently by three convergent faithful GPU-verified architectures
(single-pass, phase-factored, distinct-pathways), each reached before
the v2 module existed; convergence from three directions is strong
evidence of legitimacy over convenience. The candidate/pre-committed-
bound text is strictly quarantined in docstring section (iv) and is
never used as support for the membership; sections (i)-(iii) justify
the move from biology and the falsification record alone. This is a
catalog-derived correction of a genuinely falsified pre-registered
prediction, not a candidate-rescue repartition.

---

## ADVERSARIAL VERDICT: CLEAR

All six necessary conditions hold (bars verbatim; exactly one
membership change; verdict logic token-identical save the renamed
symbols; original byte-unchanged with its VOID preserved; biology
independently catalog-implied AND thrice-forced; no code beyond the two
new stdlib-only files), and there is no outright rationalization: the
single line that most drove this verdict is that the identical
`no_cls_replay`→WM correction is independently forced by three prior
convergent faithful GPU-verified negatives that all predate the v2
module, which a pure goalpost-move could not produce.

**Explicit restatement of the irreducible limit (binding):** a CLEAR
verdict does NOT lift the program's pre-committed honesty ceiling
(`2026-05-19-CONTROLLER-precommitted-honesty-ceiling-...`). Legitimacy
and convenience COINCIDE here by construction — the biologically-correct
membership is exactly the membership that enables a candidate PASS — and
that coincidence is irreducible and cannot be certified away by any
review. This review only rules OUT outright rationalization and confirms
the necessary conditions. A later candidate PASS against this v2 module
remains "consistent-with the corrected (biologically-revised) necessity
structure" ONLY — never a scale-confident validated deliverable, never
spun. The single load-bearing scale-confident scientific result of this
line remains the thrice-convergent FALSIFICATION of the original
pre-registered necessity prediction; the original
`integrated_loop_core.py` VOID stands permanently as the honest record.
Were any necessary condition to have failed, the verdict would be
GOALPOST-MOVE, the build blocked, and that original VOID would stand as
the deeper terminal finding.
