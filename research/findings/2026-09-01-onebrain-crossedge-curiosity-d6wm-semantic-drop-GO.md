---
type: finding
status: live
date: 2026-09-01
mechanism: onebrain-xedge-curiosity-d6-semantic-drop
lane: one-brain/integration/production
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_6seed.json
runner: research/runners/onebrain_xedge_curiosity_d6_production.py
builds_on:
  - research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-production-wire-GO.md
  - research/findings/2026-09-01-onebrain-crossedge-curiosity-to-d6wm-GO.md
---

# SEMANTIC-DROP rung: the curiosity->d6.w0 cross-edge now GENUINELY drops the register-0-bound referent from D6's
# own held set — a real hyperpolarizing pull on the session's physical w0 register, not an appended qualifier —
# 6-seed GO (6/6), lesion-attributable, byte-identical-off, auto-flip default-ON

**One-line:** the production wire-in's own declared residual ("the qualifier never causes the READOUT to
actually DROP the competed referent from the holding-N-referents list") is closed for register 0 (w0): when a
session's recent curiosity crave clears the SAME validated suppression floor that already gates the appended
qualifier, the frozen cross-edge's own measured weight (`pool.cross_weight`, zeroed by lesion) is now translated
into a genuine hyperpolarizing current on THIS session's own physical `w0` register
(`MultiSlotHold.apply_register_drive`, added to the ALREADY-VERIFIED multi-slot WM de-risk), applied inside
`MultiReferentWMOrgan.load()` BEFORE its own read — so the referent bound there is dropped from `recovered` by
the D6 substrate's own post-drive spiking state, not by a host if-statement on a diagnostic number. 6-seed GO
(6/6) on a self-test exercising the REAL production functions end to end, plus a new pytest test through the
REAL `/api/brain-chat` handler. Gated behind `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP`, byte-identical
when off, moat-safe, and now flipped default-ON per the 2026-09-01 auto-flip policy.

## 1. Required background reading (before building)

`bash tools/before_you_build.sh "curiosity gates working-memory slot maintenance semantic drop"` surfaced the two
parent findings (the runner-level GO and the production-wire GO) as the top two non-self hits, plus the
multi-slot/variable-binding WM GOs this rung reuses. `rag_search.py --corpus kandel` returned only generic
memory-systems passages (H.M./medial-temporal-lobe boilerplate) — no specific citable mechanism for
"novelty/DA gates *which* WM item is dropped" beyond what the parent finding's own §1 already cites (Lisman &
Grace 2005; Bunzeck & Duzel 2006; Braver/Cohen-O'Reilly & Frank 2006 adaptive gating; Berti & Schroger 2003;
SanMiguel/Corral/Escera 2008 on attention-capture disrupting WM maintenance) — this rung extends that SAME
attentional-capture/resource-competition account (a salient/novel signal disrupts *maintenance* of what is
currently held) rather than introducing a new one: the novel piece here is architectural (bind the ALREADY-
VALIDATED suppression signal to a REAL erase of the specific held item), not a new biological claim.

## 2. What was built (reuse-by-import; no `sim/` edit)

  1. `research/runners/_multi_slot_binding_derisk.py` — `MultiSlotHold.apply_register_drive(reg, pa, steps)`
     (additive): injects an external current directly onto register `reg`'s own band of `n_slot` pools (never
     touching the shared FS pool, so a co-held register is unaffected). Purely additive; `write`/`hold`/`read`
     are unchanged.
  2. `research/runners/d6_multiref_wm_production_organ.py` — `MultiReferentWMOrgan.load()`/`.judge()` gain an
     optional `xedge_drop_current=(pa, steps)` parameter (default `None` -> byte-identical): when given, the
     drive is applied to PHYSICAL register 0 (`w0`) after the normal write+hold span and before the final read,
     so whichever referent this session has semantically bound there is read back off the ACTUAL post-drive
     spiking state.
  3. `research/runners/onebrain_xedge_curiosity_d6_production.py` — `xedge_curiosity_d6_semantic_drop_enabled()`
     (env `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP`, default OFF pending this finding) and
     `semantic_drop_current(pool, d6org)`: `scale = clip(pool.cross_weight, 0, 1)`; `erase_pa =
     -abs(d6org.buf.clear_gain) * scale` for `d6org.buf.clear_steps` steps — reusing `MultiSlotHold`'s OWN
     clear-strength constants (the same magnitude `write()`'s overwrite-clear protocol already trusts), not a
     new magic number. Lesioning the cross-edge zeroes `cross_weight` -> `scale=0` -> this function returns
     `None` -> no drive is ever injected (the lesion is structural, not a second if-branch).
  4. `webapp/server.py` — the D6 hold-query branch now resolves the crave/suppression gate (the SAME
     `crossedge_w0_shift` + registered-floor check that already gates the appended qualifier) BEFORE calling
     `d6org.judge()`, so the resulting `xedge_drop_current` can be threaded INTO the SAME `load()`/read call that
     produces `recovered` — the drop and the qualifier now share one substrate read, not two independent ones.
  5. `tests/test_webapp_server.py` —
     `test_brain_chat_xedge_curiosity_d6_semantic_drop_genuinely_drops_referent` (new): through the REAL
     `/api/brain-chat` handler, crave+intact drops 'dog' (register 0) from the readout ("I'm holding one referent
     in working memory: cat." + qualifier); crave+lesioned recovers both; the flag OFF is byte-identical to the
     pre-existing qualifier-only behaviour.

## 3. The empirical de-risk that shaped the design (an honest methodological note)

The FIRST design tried was a forward (excitatory) drive at the cross-edge's own `ASK_DRIVE_PA` magnitude, mirroring
the toy pool's own read protocol literally. A standalone probe (seeds 42/43/44, whole register-0 band, several
magnitudes) showed this is **non-monotonic and seed-inconsistent**: the register-0 readback's recovered local
slot flipped to a WRONG code at some magnitudes (a genuine "drop", but unpredictably) and stayed correct at
others, with no clean magnitude->effect relationship. A second attempt drove the SHARED FS pool (a resource-
competition analog: both registers compete for one inhibitory resource) — this DID produce a clean collapse at
high gain, but WHICH register collapsed first was seed-dependent (register 0 in some seeds, register 1 in
others), failing the requirement that the cross-edge specifically targets `w0`. A third attempt (extra idle
hold time, a "distraction consumes maintenance cycles" account) found the bistable NMDA hold essentially
undecaying over 320 extra hold steps — this substrate's persistent-activity slot is *designed* to be robust to
idle time, so "attention elsewhere for a while" does not erode it on its own.

The mechanism that WORKED, reliably, on all 6 target seeds: a **negative (hyperpolarizing) current directly on
register 0's own band**, at magnitude/duration matching `MultiSlotHold`'s own pre-existing `clear_gain`=1500pA /
`clear_steps`=200 (the SAME strength `write()`'s overwrite-clear protocol already trusts to erase a bump) —
this collapses register 0's subsequent `read()` to `(-1, 0.0)` on every one of seeds 42/43/44/100/101/102 while
leaving an undriven co-held register fully intact. This is a genuinely MEASURED substrate property (this
bistable hold resists graded/partial suppression at every magnitude probed short of clear-strength; only an
all-or-nothing erase proved reliable), not an assumption — consistent with CLAUDE.md's own standing lesson that
an operating point is implicit and a mechanism you cannot measure correctly gets tuned in the wrong direction
confidently. The residual this leaves (declared in §6) is that the drop is binary, not continuously graded.

## 4. 6-seed GO (42/43/44/100/101/102), numpy CPU

`SIM_BACKEND=numpy python -m research.runners.onebrain_xedge_curiosity_d6_production --grow --semantic-drop
--seeds 42,43,44,100,101,102 --out
research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_6seed.json` — 6/6 GO. Each seed's self-test
(`_selftest_semantic_drop`) exercises the REAL production functions end to end (`MultiReferentWMOrgan.judge()` +
`semantic_drop_current()`, not a bespoke probe): two referents ('dog'->register0/w0, 'cat'->register1) are
loaded, then a hold-query is judged four ways.

| seed | cross_weight (intact) | no-crave recovered | crave+intact recovered | crave+LESIONED recovered | flag-OFF recovered | GO |
|---|---|---|---|---|---|---|
| 42 | 2.0202 | dog, cat | **cat** | dog, cat | dog, cat | GO |
| 43 | 1.8899 | dog, cat | **cat** | dog, cat | dog, cat | GO |
| 44 | 1.9818 | dog, cat | **cat** | dog, cat | dog, cat | GO |
| 100 | 1.9708 | dog, cat | **cat** | dog, cat | dog, cat | GO |
| 101 | 2.1176 | dog, cat | **cat** | dog, cat | dog, cat | GO |
| 102 | 1.7391 | dog, cat | **cat** | dog, cat | dog, cat | GO |

On every seed: (a) no crave -> both referents recovered (the honest baseline); (b) crave + semantic-drop ON +
cross-edge INTACT -> 'dog' (register 0, loaded first under the role-by-position marker) is GENUINELY absent from
`recovered` — the readout drops from "I'm holding 2 referents in working memory at once: dog and cat." to "I'm
holding one referent in working memory: cat." (plus the pre-existing qualifier); (c) the SAME crave, cross-edge
LESIONED (`pool.cross_weight` collapses to ~0.0000) -> both referents recovered again — the anti-hollow proof;
(d) crave present but the SEMANTIC-DROP FLAG left off -> byte-identical to (a) (confirms the FLAG, not the crave
alone, gates the new behaviour). All four conditions held on all 6 seeds (`n_go=6/6`).

Confirmed end-to-end through the REAL `/api/brain-chat` handler
(`tests/test_webapp_server.py::test_brain_chat_xedge_curiosity_d6_semantic_drop_genuinely_drops_referent`, plus
the pre-existing five `xedge_curiosity_d6` tests re-run green — no regression from restructuring the hold-query
branch to resolve the crave gate before calling `d6org.judge()`).

## 5. Moat-safety, byte-identical-off, no fabrication (checked, not assumed)

  * **Moat-safe.** The drop never invents a referent, never flips an abstain, never claims a fact — it only
    changes WHICH of the session's own already-loaded referents the spiking buffer's OWN read recovers, exactly
    the same class of read the un-suppressed path already used. The count "I'm holding N referents" now honestly
    reflects the substrate's own degraded read when N drops to 1.
  * **Byte-identical-off.** `BRAIN_ONEBRAIN_XEDGE_CURIOSITY_D6_SEMANTIC_DROP` unset ->
    `xedge_curiosity_d6_semantic_drop_enabled()` is False -> `_cd6_drop_current` stays `None` at BOTH new call
    sites -> `load()`'s new parameter default (`None`) skips the new code path entirely -> no current is ever
    injected. Checked via the new pytest test's flag-off arm (byte-identical readout to the pre-existing
    qualifier-only rung) and via the unchanged pass of the five pre-existing `xedge_curiosity_d6` tests.
  * **Lesion-attributable, and the lesion structurally holds.** `semantic_drop_current` reads `pool.cross_weight`
    directly; `pool.lesion_cross()` zeroes it in place on a FROZEN pool (`enable_hebbian_learning=False`, so it
    cannot regrow mid-measurement) — the lesion is verified to still hold at the moment of every read in this
    finding's own 6-seed table (`cross_weight_lesioned` prints ~0.0000 on all 6 seeds).
  * **Session-isolated** (inherited, unchanged): the drive magnitude is a pure function of `pool.cross_weight`
    (a process-shared, stateless-given-input read) and `d6org.buf` (the CALLING session's own per-session
    `MultiSlotHold`) — never a shared, cross-session mutation.

## 6. Honest residuals (declared, not hidden)

  * **Binary, not graded.** The empirically-validated mechanism (a clear-strength hyperpolarizing pull) is
    all-or-nothing: register 0's bump either fully survives or is fully erased, scaled only by whether the
    cross-edge is intact/lesioned (via `pool.cross_weight`'s clamp to {0,1}), not by a continuously graded
    "partial suppression." §3 documents that graded suppression was probed and found unreliable on this
    substrate at every non-clear-strength magnitude tried — a measured substrate property, not an assumption.
  * **Only register 0 (w0) is ever targeted.** The other four registers (w1-w4) have no declared cross-edge and
    are structurally unaffected by this rung, matching the base cross-edge's own one-sided-by-design scope.
  * **The erase is applied WITHIN one `load()`/`judge()` call, not a persistent deletion.** `_slot_of_ref`/
    `_ref_of_slot` host bookkeeping is untouched, so a LATER unrelated turn re-writes and re-reads the referent
    successfully (matching the pre-existing qualifier's own "fires once per crave episode, consumed" semantics)
    — this is a per-reply functional drop ("suppressed right now"), not a claim the brain has permanently
    forgotten the referent.
  * **Training/region-pair choice residuals are inherited unchanged** from the two parent findings (host-
    supervised tonic co-drive; hand-directed region-pair selection).

## 7. Decision: GO, load-bearing, and AUTO-FLIP

Per the 2026-09-01 auto-flip policy (validated-GO + genuinely load-bearing on the live `/api/brain-chat` +
moat-safe + byte-identical-off + no-regression -> flip; the only guard is the hollow-flip trap): this is NOT a
hollow flip. Unlike the ALREADY-flipped qualifier-only rung's own starting point, this rung changes the actual
SET of referents reported held, not only appended text — a strictly stronger load-bearing bar. `_CD6_SEMANTIC_
DROP_DEFAULT_ON` is flipped to `True` in `onebrain_xedge_curiosity_d6_production.py`.

Per `docs/TERMS.md`: this faculty is `wired` (reachable from `/api/brain-chat` on the hold-query path) and
`on-by-default`. It does not (yet) qualify as `scaffold-retired`/`integrated` in the stricter sense — there is no
prior host shortcut being replaced here (this is a NEW cross-organ effect on an existing spiking buffer, not a
displacement of host computation), so that stronger term is not claimed.

## 8. Files

`research/runners/_multi_slot_binding_derisk.py` (+`apply_register_drive`) ·
`research/runners/d6_multiref_wm_production_organ.py` (+`xedge_drop_current` param on `load`/`judge`) ·
`research/runners/onebrain_xedge_curiosity_d6_production.py` (+flag, +`semantic_drop_current`,
+`_selftest_semantic_drop`, +`--semantic-drop` CLI, default flipped ON) · `webapp/server.py` (hold-query branch
restructured, additive) · `tests/test_webapp_server.py` (+1 new test) ·
`research/findings/raw/_onebrain_xedge_curiosity_d6_semantic_drop_6seed.json`. Reused, unmodified:
`research/runners/_multi_slot_binding_derisk.MultiSlotHold` (its pre-existing `read`/`write`/`hold`),
`research/runners/_onebrain_crossedge_curiosity_to_d6wm.py` (`AskToW0Pool`, unchanged). No `sim/` file touched.

Functional read-outs only; no phenomenal-experience claim.
