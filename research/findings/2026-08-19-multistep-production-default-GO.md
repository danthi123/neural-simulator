---
status: go
type: finding
lane: T1-1
date: 2026-08-19
integration_faculty: gnw-multistep-deliberation
---
# MULTI-STEP re-entrant deliberation is now ON BY DEFAULT in the live brain — the production-default flip (board #68): a chase-form question through the real `/api/brain-chat` handler now works the WHOLE chain by default, with `BRAIN_GNW_MULTISTEP=0` the reversible byte-identical DISABLE override — GO

**Date:** 2026-08-19 · **Flip:** [`webapp/server.py`](../../webapp/server.py) `_GNW_MULTISTEP_DEFAULT_ON = True` (the production-integration anchor, mirroring `_GNW_DELIBERATE_DEFAULT_ON`) + [`webapp/gnw_multistep_deliberation.py`](../../webapp/gnw_multistep_deliberation.py) `multistep_enabled()` default-ON · **Artifact:** `research/findings/raw/_multistep_production_default/verify.json` · **Ledger:** row `gnw-multistep-deliberation` in [`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`](../../docs/PRODUCTION_INTEGRATION_LEDGER.yaml) flipped `on_by_default: NO -> YES` · **Scope:** webapp production-default flip, additive + reversible, `NO sim/ edit` (`git diff sim/` empty). FUNCTIONAL correlate only; NO phenomenal claim.

## Verdict: GO — the 6/6-seed-GO multi-step faculty is now the production default turn; all four anti-cheats hold through the REAL ChatBrain + handler

The multi-step re-entrant deliberation loop landed 2026-08-19 as a DEFAULT-OFF de-risk, 6/6-seed adversarially-verified GO ([`2026-08-19-gnw-multistep-reentrant-deliberation-GO.md`](2026-08-19-gnw-multistep-reentrant-deliberation-GO.md)), wired into the live gate but gated behind `BRAIN_GNW_MULTISTEP` (unset = OFF). This flip makes it the production default: on an explicit chase-form question the workspace re-enters and the substrate's OWN spiking `n_ignited` read decides how many cycles to run, LIVE, with NO env var set. The env var is retained as the reversible DISABLE override (`BRAIN_GNW_MULTISTEP=0` -> the installed wrapper is a pure pass-through -> byte-identical to the pre-flip default). The underlying faculty's substrate control is the committed 6/6-seed GO; this finding is the production-default flip and its four live-handler anti-cheats.

## The change (webapp only, additive, reversible)

- `webapp/server.py`: added `_GNW_MULTISTEP_DEFAULT_ON = True` next to `_GNW_DELIBERATE_DEFAULT_ON` (the production-integration anchor the CLASS-PI gate checks); the `brain_chat` install block now installs the multi-step gate whenever the anchor is True (was: installed only when `multistep_enabled()`). The gate wrapper is therefore ALWAYS installed by default; when disabled it pass-throughs.
- `webapp/gnw_multistep_deliberation.py`: `multistep_enabled()` flipped to DEFAULT-ON — `os.environ.get("BRAIN_GNW_MULTISTEP", "1") not in {0,false,off,no}` (was default-OFF `... in {1,true,on,yes}`), mirroring `deliberate_enabled()`. The wrapper's existing `if not multistep_enabled(): return inner_gate(question)` line makes `BRAIN_GNW_MULTISTEP=0` a pure pass-through. The handler never reads `_last_gnw_multistep` into the response, so installed-but-disabled is byte-identical to not-installed.

No `sim/` edit; no change to the chase mechanism, the detector, or the moat. Only the install default + the flag polarity moved.

## The four anti-cheats — verified through the REAL `/api/brain-chat` handler (numpy, `BRAIN_COMPOSER_KIND=rf`)

Each arm ran in a fresh process against the real `webapp.server.brain_chat` handler, teaching a transitive chain `zorp -chase-> blib -chase-> krad -chase-> munt` via the production `hear` path, then asking `"what does zorp chase all the way?"`. A non-chase reactive panel (recall / abstain / self / acquisition / anaphora — no chase marker) is hashed per arm as the ordinary-turn unit. Pre-flip baseline was captured on the base `fa300e1a0` code before editing.

| arm (real handler) | chase answer | chase `recalled_svo` | non-chase panel md5 |
|---|---|---|---|
| PRE-FLIP default (base, no env) | The zorp chases blib. | `[zorp, chase, blib]` (first hop) | `a1313788…93ed` |
| POST default-ON (no env) | **The zorp chases munt.** | `[zorp, chase, munt]` (terminal) | `a1313788…93ed` |
| POST disabled (`BRAIN_GNW_MULTISTEP=0`) | The zorp chases blib. | `[zorp, chase, blib]` (first hop) | `a1313788…93ed` |
| POST lesion (`BRAIN_GNW_MULTISTEP_LESION=1`) | I don't know about that. | `null` (abstains) | `a1313788…93ed` |

1. **Default-ON works live (no env var).** The chase question returns the chain TERMINAL "The zorp chases munt." (`recalled_svo [zorp, chase, munt]`) — the terminal of the taught chain, not the first hop "blib". The multi-step path is active in production with nothing set.
2. **Explicit-disable is byte-identical.** With `BRAIN_GNW_MULTISTEP=0`, the full chase response md5 (`0ad0f233…e1be`) AND the non-chase panel md5 (`a1313788…93ed`) EQUAL the pre-flip default byte-for-byte; the chase answer is the pre-flip first-hop "blib". The off-path is untouched — the flip is reversible and provably byte-identical when disabled.
3. **Live lesion probe is load-bearing.** With the multi-step path active by default, `BRAIN_GNW_MULTISTEP_LESION=1` (the recurrence-zeroed workspace) collapses the chase to abstain ("I don't know") — the terminal is no longer reached — while the ordinary/one-hop turns (non-chase panel md5) are UNCHANGED; and the alternative lesion form (disable) degrades the multi-step answer to first-hop while ordinary turns stay identical. The now-default faculty is provably doing work in production: removing the workspace recurrence it depends on removes the multi-step answer, and one-hop recall is untouched.
4. **No regression on ordinary turns.** Under default-ON the non-chase reactive panel md5 (`a1313788…93ed`) is IDENTICAL to the pre-flip baseline. The chase-form detector does not hijack normal turns — recall, abstain, self-identity, acquisition, and anaphora all answer exactly as before the flip.

All four pass (`research/findings/raw/_multistep_production_default/verify.json`, `all_four_pass: true`).

## Honest residuals (unchanged from the de-risk; the flip does not close them)

- **scaffold_retired stays NO.** The declared boundaries of the de-risk are unchanged by making it default: the chase-form DETECT + the (agent, action) EXTRACT are host comprehension of the teacher/world utterance (the same boundary the SVO question parser occupies); PROPOSE is `composer.query_patient` (the declared modular-processor boundary). The substrate's independent work — the CYCLE COUNT / when-to-halt as a spiking `n_ignited` read — is what runs by default now.
- **Per-hop-reset form only** (snapshot-restore wash-out; the continuous no-reset train-of-thought is gated on the unbuilt async attractor). **Co-resident** on the shared P1.2 workspace bridge (the same one the single-hop deliberation uses via `_get_bridge`), not merged with the recall composer's bridge — so it adds no NEW spiking substrate to the default-on-spiking count.
- FUNCTIONAL correlate only; NO phenomenal claim. This is re-entrant multi-step deliberation with the measured improvement, NOT "reasoning to a true conclusion".

Cites: the de-risk GO [`2026-08-19-gnw-multistep-reentrant-deliberation-GO.md`](2026-08-19-gnw-multistep-reentrant-deliberation-GO.md); the single-hop sibling flip pattern [`2026-08-18-gnw-deliberation-wired-brain-chat-GO.md`](2026-08-18-gnw-deliberation-wired-brain-chat-GO.md).
