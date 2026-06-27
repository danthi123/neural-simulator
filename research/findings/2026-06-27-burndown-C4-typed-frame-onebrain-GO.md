# Burndown conversion C4 — typed verb-frame argument-structure on the spiking onebrain substrate — GO (2026-06-27)

**Verdict: GO at D=128 (the console brain D) — CLEAN CONVERSION, wired + tested.** The TYPED verb-frame
argument-structure surface (the typed roles GOAL/THEME/RECIPIENT/LOCATION/SOURCE/INSTRUMENT/TIME — `store_fact` /
`query_role` / the verb-frame `render`) now runs on the persistent spiking `OneBrainComposer` substrate
ANSWER-IDENTICAL to the numpy `ArgStructureComposer` oracle on every validated typed case, moat 0-FA. This is the
**LAST Bucket-A operation-conversion** — C1/C2/C3 put the word-ordering + the flat-SVO recall/answer on spikes; C4
adds the typed verb-frame path, so the console's `--argstructure` path can run `--composer onebrain`. Spec:
`2026-06-27-burndown-bucketA-build-plan.md` (C4 + §1a's typed-role honest caveat).

**The conversion is reuse-by-import + additive — NO `sim/` edit.** The `OneBrainComposer` was ALREADY fully
parameterized over `self.bind_roles` / `self.main_roles` / `self.n_roles` (the bind / store / unbind / cleanup all
iterate them uniformly), and its binding is role-AGNOSTIC. So the typed-role API is: (1) give each typed role a phasor
code on the inner composer's `self.comp.roles` from a DISJOINT rng stream (`seed+2000`, == `ArgStructureComposer` +
`OrderedPositionWM`, so the parent's concept/role codes stay byte-identical); (2) insert the typed roles into
`bind_roles` (and so `main_roles`) BEFORE polarity (preserving the polarity-LAST layout invariant); (3) add
`store_fact` / `query_role` / `render` mirroring `ArgStructureComposer`, routing through the existing on-bridge
`_store_composite` (RF complex-synapse store) + `_read_blocks` (the resonate scan/unbind/cleanup) + the C1 spiking
competitive-queuing renderer for the frame word-order. The per-fact BUNDLE never exceeds the few roles a single verb
frame realizes (go → agent+action+GOAL = 3; give → agent+action+THEME+RECIPIENT = 4), since `_store_composite` binds
only the roles a fact ACTUALLY has — the same density the flat+attribute path already validates.

---

## The de-risk (the HARD GATE) — `_burndown_C4_typed_frame_onebrain_derisk.py`, GPU (cupy), 6 seeds (42–47)

The substrate typed composer (`OneBrainComposer(typed_roles=TYPED_ROLES)`) vs the numpy `ArgStructureComposer` oracle,
identical seed/D/vocab, the same validated typed facts (== `tests/test_argstructure_composer.py`):
`{boy go GOAL:park}`, `{girl give THEME:ball RECIPIENT:dog}`, `{dog put THEME:bone LOCATION:table}`,
`{cat chase patient:river}`.

| Check | D=64 | **D=128 (console D)** |
|---|---|---|
| typed RECALL substrate==oracle==truth (GOAL/THEME/RECIPIENT/LOCATION/patient + the agent inverse) | 4/6 seeds | **6/6 seeds, 7/7 each** |
| RENDER substrate==oracle==target (incl the default spiking-CQ order) — "the boy goes to the park", "the girl gives the ball to the dog", "the dog puts the bone on the table", "the cat chases the river" | 4/6 | **6/6, every frame MATCH** |
| **MOAT false-accepts** (unstored cue → None) | **0 (all 6 seeds)** | **0 (all 6 seeds)** |
| moat abstain-all + oracle-parity | True | **True** |
| VERIFY re-parse (rendered prose → stored typed fact) | — | **all OK** |
| agrammatism ablation ("boy go park", no func-words/no tense) | True | **True** |
| **overall** | **GO 4/6** | **GO 6/6** |

Raw: `research/findings/raw/_burndown_C4_typed_frame_onebrain.json` (the D=128 GO 6/6 run; 984.7s).

### The bundle-SNR / D boundary — exactly the prompt-flagged case, resolved by the standard VSA lever

The prompt flagged: *"WATCH for a bundle-SNR / D boundary — typed frames are DENSE composites... if the substrate
can't hold typed frames cleanly at the console D, RAISE D (the standard VSA lever)."* That is precisely what was
observed:

- **At D=64 (the test default), GO 4/6** — seeds 44/45/46/47 perfect (7/7 + all renders); seeds 42/43 mis-decode the
  DENSEST 4-role frame (`put` = agent+action+THEME+LOCATION+polarity): seed 42 reads THEME as "chase" (a verb!),
  seed 43 reads the agent as "run". This is a per-seed substrate READ-fidelity issue on the thinnest-margin codes at
  low D — NOT a mechanism break (the moat is 0-FA at every seed; the abstentions are exact).
- **At D=128, GO 6/6** — the higher dimensionality restores the per-role unbind margin; every seed reads all 7 typed
  cases and renders all 4 frames exactly.

So C4 is GO **at D≥128 (the console's 7K brain is D=128)**; the honest scope is that the typed-frame path needs
D≥128 (a low-D brain mis-decodes the densest 4-role frames). The console build warns when `--argstructure --composer
onebrain` is used at D<128.

**Falsification bar (substrate diverges beyond a D-liftable margin, OR the moat breaks): NOT hit.** The divergence at
D=64 was fully closed by the standard VSA dimensionality lever, and the moat was 0-FA at BOTH D=64 and D=128 — the
safety property never broke. So this is a clean conversion gated on D≥128, not a NEGATIVE.

### GPU transcript (the boy/park GOAL case + the typed recalls + renders via the onebrain)

```
[BURNDOWN C4] ... backend=cupy(GPU substrate), D=128, seeds=(42, 43, 44, 45, 46, 47)
  [seed 42] RECALL parity 7/7, correct 7/7: GOAL(=park), agent(=boy), THEME(=ball), RECIPIENT(=dog),
                                            THEME(=bone), LOCATION(=table), patient(=river)
  [seed 42] RENDER go:   sub="the boy goes to the park"        (target MATCH; cq MATCH)
  [seed 42] RENDER give: sub="the girl gives the ball to the dog" (target MATCH; cq MATCH)
  [seed 42] RENDER put:  sub="the dog puts the bone on the table" (target MATCH; cq MATCH)
  [seed 42] RENDER chase:sub="the cat chases the river"        (target MATCH; cq MATCH)
  [seed 42] MOAT: recall_ok=True, abstain 3/3, false_accepts=0 (parity=True)
  [seed 42] AGRAMMATISM (ablate scaffold): "boy go park" -> OK
  ... (seeds 43–47 all 7/7 + every render MATCH + moat 0-FA) ...
  SUMMARY (6 seeds, D=128): GO 6/6
    typed RECALL substrate==oracle==truth: True
    RENDER substrate==oracle==target (incl spiking-CQ): True
    MOAT false-accepts total: 0 (must be 0); abstain all=True; parity=True
    VERIFY re-parse all: True
    agrammatism (ablate->telegraphic): True
```

"where does the boy go?" → `query_role("GOAL", agent="boy", action="go")` → **"park"** on the spiking substrate,
rendered "the boy goes to the park" — the GOAL bound + stored in RF complex synapses, recalled via the resonate
scan/unbind, the preposition "to" + determiner "the" from the verb frame, the word-order from the C1 spiking
competitive-queuing read-out.

---

## What shipped — reuse-by-import, additive, NO `sim/` edit

- **`research/runners/one_brain_composer.py`** (additive):
  - `OneBrainComposer.__init__(typed_roles=None, framecq_seed=None, use_spiking_cq=None, ...)` — `typed_roles` (default
    None = **byte-identical** flat path) extends `bind_roles`/`main_roles` and registers the typed-role phasors on
    `self.comp.roles` from the `seed+2000` disjoint stream; `framecq_seed`/`use_spiking_cq` configure the render's
    serial-order engine (the C1 spiking CQ on GPU / the numpy FrameCQ oracle on CPU — the consolidated_320 pattern).
  - `store_fact(fact)` / `query_role(role, **cue_roles)` / `render(fact, comp=None, ablate_closed_class=, use_framecq=)`
    + `_composite_for_typed` + `_ordering_engine` — the typed verb-frame API, routing through the existing on-bridge
    `_store_composite` + `_read_blocks` (so bind/store/unbind/cleanup run on FIRING NEURONS) + the frame lexicon +
    the C1 spiking CQ renderer. `query_role` abstains (the moat) on an unmatched cue OR a role a matching fact did not
    bind (never confabulating a role the fact lacks).
- **`research/runners/first_chat_console.py`** (additive): `--argstructure --composer onebrain` (BURNDOWN C4) now
  builds `OneBrainComposer(typed_roles=TYPED_ROLES, enable_spiking_cleanup=False)` — `enable_spiking_cleanup=False`
  MATCHES the numpy oracle's host-argmax cleanup exactly (the substrate store == the numpy kb bit-for-bit; the same
  reasoning C3 used for the flat path's exact parity at the crowded V=1454/D=128 console scale). `--composer rf`
  (default) keeps the numpy `ArgStructureComposer` oracle. A D<128 warning is printed. `store_fact`/`query_role`/
  `render` are API-identical across the rf/onebrain typed paths, so the DiscursiveTurn/agent/wh-routing are unchanged.
- **`research/runners/_burndown_C4_typed_frame_onebrain_derisk.py`** (new de-risk): the multi-seed substrate==oracle
  HARD GATE (configurable `C4_D`, `C4_SEEDS`).
- **`tests/test_argstructure_onebrain.py`** (new CI guard, GPU-gated, skips off-GPU): pins the typed-recall parity, the
  render parity (boy/park + give/put/chase + the default spiking-CQ), the no-confab moat 0-FA, VERIFY re-parse, the
  agrammatism anti-cheat, and the default-path byte-identity (no typed_roles → bind_roles + concept codes unchanged).

### Verification

- **`SIM_BACKEND=cupy`** (the substrate): the C4 GPU CI guard `tests/test_argstructure_onebrain.py` → **7 passed**
  (the full typed matrix at D=128). The de-risk → **GO 6/6** at D=128.
- **No regression** on the flat onebrain path: `test_one_brain_composer_agent.py` (default-path-unaffected,
  encoding-gain-default-off byte-identical, batched==per-block, the who/what matrix + moat) → **4 passed** with the
  additive `__init__` change.
- **`SIM_BACKEND=numpy`** (the oracle/CPU path stays byte-identical): `test_argstructure_composer.py` → **8 passed**
  (the numpy `ArgStructureComposer` oracle unchanged); the new C4 guard skips off-GPU (as designed). A numpy smoke of
  the typed `OneBrainComposer` confirmed the typed recall + render + moat + the default-path byte-identity on CPU too.

**Scope honored:** NO `sim/` edit anywhere; reuse-by-import; the default flat `OneBrainComposer` + the numpy
`ArgStructureComposer` oracle are byte-identical (the typed roles draw from a disjoint stream and default off). The
binding stays the fixed exact-inverse FHRR algebra (the genuine learned-cortex bind = the separate step-3 frontier).

**Honest residual / follow-on:** (1) the typed-frame path needs **D≥128** (a low-D brain mis-decodes the densest
4-role frames — the bundle-SNR boundary, the standard-VSA-lever fix); (2) per-instance the substrate composer builds
its own bridge (the same per-instance cost C1 noted; the console builds ONE); (3) `--argstructure --composer onebrain`
is single-bridge (a RoutedComposer-of-typed-onebrain is a deeper-knowledge-scaling follow-on, not on the C4 path).
With C4, ALL FOUR Bucket-A operation-conversions (C1 word-order, C2 console render, C3 flat who/what, C4 typed
verb-frame) run on the spiking substrate.
