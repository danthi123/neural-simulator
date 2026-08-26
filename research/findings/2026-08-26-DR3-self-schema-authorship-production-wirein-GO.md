---
type: finding
status: live
date: 2026-08-26
mechanism: self-schema
---

# DR-3 self-schema AUTHORSHIP (self-vs-heard) — production wire-in behind BRAIN_SELF_SCHEMA (default OFF), GO

The DR-3 self-schema `author` sub-block (6/6-seed GO,
[`2026-07-23-DR3-self-schema-region-6seed-GO.md`](2026-07-23-DR3-self-schema-region-6seed-GO.md); authorship
acc 1.000, self-lesion collapses author to chance 6/6) <!--derived--> (quoted from the cited 2026-07-23 de-risk finding)
is now WIRED into the live `brain_chat` rich path to BACK
the host 'a guess, not something I was taught' flag with a genuinely-SPIKING authorship read. Additive, guarded
behind a NEW env flag `BRAIN_SELF_SCHEMA` (**default OFF** — the parent flips default-on after the pool soak).
NO `sim/` edit; reuse-by-import of the de-risk's `build_self_schema_bridge` + `_run_trial`.

## What was built
- **Production organ** `research/runners/self_schema_production_organ.py` — a process-shared
  `SelfSchemaAuthorshipOrgan` (reuse-by-import). Build-time it calibrates a self-vs-heard threshold (midpoint of
  one volunteered read and one recalled read); each read ignites a fixed workspace content, holds the authorship
  drive (high for a volunteered proposition, zero for a recalled fact), free-runs, and reads the `author`
  sub-block's late-window firing RATE. `read_author(authored, lesion)` decodes 'self' (rate >= threshold) vs
  'heard'. `authorship_marker()` is the honest own-guess prefix. `self_schema_enabled()` /
  `self_schema_lesioned()` mirror the metacog/curiosity convention.
- **Guarded wiring** in `webapp/server.py` `brain_chat` rich path, inside the existing `if is_hyp:` block
  (a generated HYPOTHESIS = a volunteered, self-authored proposition), marked `BEGIN/END faculty: DR-3
  self-schema AUTHORSHIP` so the parent can merge it beside other faculties' blocks. When `BRAIN_SELF_SCHEMA`
  is on, the author pool reads 'self' and an honest own-guess MARKER is prepended to the reply; the read is
  attached as `resp["authorship"]`. Plus a `_get_self_schema_organ()` helper beside the other organ getters.
- **Soak** `research/runners/_self_schema_authorship_flip_soak.py` (models `_d5_graded_flip_soak.py`) — the
  6-seed no-regression gate the parent runs before the default-on flip.

## The host boundary (declared, unchanged from the de-risk)
WHICH authorship context a turn carries — a volunteered proposition (self) vs a recalled fact (heard) — is
supplied by the CALLER (`is_hyp`, the host's open-ended-generation branch), exactly as the de-risk's own
authorship current is externally set per trial. The genuine SPIKING part, and the thing the marker RIDES, is the
author pool's readback: it fires 'self' only when driven AND intact; the self-lesion collapses it to 'heard'.

## Verification
**FLAG-OFF byte-identical (PROVEN).** The wiring lives entirely inside `if is_hyp:` and is guarded by
`self_schema_enabled()` (default False), so an ordinary turn never touches it and a flag-off turn is byte-identical
by construction. Empirically confirmed by the 6-seed soak: on an ORDINARY (non-hypothesis) turn, flag-OFF ==
flag-ON byte-identical (no `authorship` key, no text change), 6/6 seeds, through a faithful reproduction of the
server block driving the REAL organ.

**LOAD-BEARING (PROVEN) — the de-risk's own self-lesion oracle, in production.** On a HYPOTHESIS (triggered) turn:
- flag OFF -> byte-identical host default (no marker, no key).
- flag ON + INTACT -> the author pool reads 'self' (author_rate ~0.092 >= threshold ~0.046) <!--derived--> -> the marker is
  prepended -> the answer demonstrably changes (values rounded from the cited soak JSON per-seed author_rate/threshold).

- flag ON + LESIONED (`BRAIN_SELF_SCHEMA_LESION=1`, author access severed `schema_access=False`) -> the pool
  goes silent (author_rate 0.000), the read collapses to 'heard', the marker VANISHES, and the answer reverts
  BYTE-IDENTICALLY to the flag-OFF/host-default text — while the recalled/content body is unchanged throughout.

Organ-level (2026-08-26, seed 42): intact self author_rate 0.0919 -> 'self' (marker); intact heard 0.0000 -> <!--derived-->
'heard' (no marker); lesioned-self 0.0000 -> 'heard' (marker VANISHES). 6-seed soak
(`_self_schema_authorship_flip_soak --seeds 42 43 44 100 101 102`): **6/6 GO** — ordinary byte-identical True,
load-bearing True, lesion-vanishes True, on every seed. Raw:
`research/findings/raw/_self_schema_authorship_prodflip/soak_summary_6seed.json`.

## The authorship-axis self-lesion (why `schema_access=False` is the whole lesion here)
The de-risk anti-cheat (1) severs the schema's access via `schema_access=False` in `_run_trial` (author drive ->
0) while the workspace still ignites, so the BRAIN state is unchanged and only the schema's READ is cut. For the
AUTHORSHIP axis specifically that is the whole lesion — the member->attend `lesion_schema` weight only touches
the ATTENTION axis, which this organ does not read. The organ therefore builds `lesion_schema=False` and applies
the lesion at read time (`schema_access=False`), so the SAME organ serves a normal request and a
lesion-verification probe.

## Honest residual
The full end-to-end trigger through `/api/brain-chat` on a live HYPOTHESIS turn was NOT run: the ChatBrain +
rich-composer warm wedges in this isolated worktree (no data lake; the known warm wedge the wave brief warns
about), even with the stub renderer and the heavy faculties disabled. The coupling is instead proven at the
organ level and through the faithful production-wiring soak (the server block reproduced in exact
correspondence and driven against the real organ). The parent should confirm one live hypothesis turn on a warm
cupy deployment when convenient — it exercises the SAME `read_author` + marker-prepend the soak already gates.

## Scope / honesty
FUNCTIONAL self-model / agency correlate (a learned-substrate authorship read), NEVER a claim of subjective
experience. Moat-safe + additive: the organ never produces an answer, flips an abstain, or changes a recalled
fact — it only prepends an honest own-guess marker onto an already-host-flagged guess turn. NO `sim/` edit.
The default-on flip is left to the PARENT after the 6-seed pool soak.
