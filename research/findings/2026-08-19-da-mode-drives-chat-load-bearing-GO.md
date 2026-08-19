---
status: live
type: finding
lane: laneC
date: 2026-08-19
integration_faculty: da-mode-drives-response
---

# DA-MODE DRIVES THE LIVE RESPONSE — the brain's OWN spiking dopamine mode (rest/focus/arousal) made LOAD-BEARING on `/api/brain-chat` (board #79). The #76 spiking DA nucleus (the `snc` population self-produces the tonic DA LEVEL from a reward/context afferent, off the neuromodulator bus — NEVER a host `set_concentration`) reads the message's engagement each turn; the self-produced level is binned to a MODE, and the mode modulates HOW forthcoming the reply is (a graded ENGAGEMENT SUFFIX — a distinct axis + a suffix, not a third prefix, vs #84 valence-lead / #85 topic-lead). Verified through the REAL handler: (A) the mode tracks the conversation (a dull/greeting opening reads REST/no-suffix; an engaging/novel exchange reaches FOCUS/AROUSAL/suffix); (B) message FIXED, inducing focus vs rest (via the SNc afferent) changes the reply suffix with the base sentence + content byte-identical, and the difference VANISHES under the neural SNc-nucleus lesion (silence the nucleus → DA collapses → REST → suffix gone → == base); (C) content mode-invariant, byte-identical with the flag OFF. DEFAULT-ON. GO.

**Date:** 2026-08-19 · **Board:** #79 (INTEGRATION-TO-PRODUCTION). **Backend:** CPU (numpy). **Verdict:** **GO** through the real `/api/brain-chat` handler (in-process). **No `sim/` edit** (`git diff sim/` empty).

**Files:** `webapp/da_mode_drives_chat.py` (NEW — reuses the #76 `_neuromod_spiking_da_mode_derisk` machinery: `PM.build` the fixed-anatomy BG substrate + `make_manager` the `dopamine_mode` bus + `measure_self_driven` the live SNc→DA loop; maps the self-produced level → a mode → an engagement suffix), `webapp/server.py` (the `_DA_DRIVES_DEFAULT_ON=True` anchor + `_da_drives_on()` + the per-turn read block + the `da_drives_suffix` append / `da_drives` attach on both main return paths), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `da-mode-drives-response`). **Artifact:** `research/findings/raw/_da_drives_chat/verify.json`.

**Reproduce:** `SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u research/findings/raw/_da_drives_chat/verify_da_drives.py` — drives `webapp.server.brain_chat(BrainChatRequest(...))` directly (the REAL handler). The heavy default organs are disabled for a tractable in-process verify (a consistent baseline across ALL arms; da-drives reads its own #76 substrate + appends its suffix regardless of the others, so the isolation cannot change any da-drives verdict). renderer=stub (GPU-free deterministic surface).

## What this closes
The #76 spiking DA-mode GO (`2026-08-19-neuromod-spiking-da-mode-GO`) established the brain's OWN dopamine nucleus self-producing the mode (6/6 seeds; silence the SNc → the level collapses byte-for-byte), but shipped as a DEFAULT-OFF de-risk RUNNER — its closing note names exactly this rung: *"wiring `dopamine_mode` + the SNc read into the live default brain so its own DA nucleus sets the mode moment-to-moment during a conversation (rest/focus/appetitive/aversive)."* The ledger confirmed the gap: *"no standalone reward/value neuromodulator drives the live chat turn."* This finding makes the self-produced DA mode CHANGE the surface: the brain's dopaminergic state is now load-bearing on how forthcoming the reply is, and the change VANISHES when the neural DA nucleus is lesioned. Mirrors the board-#84 affect-DRIVES + board-#85 swap-DRIVES paths.

## The coupling (neural level → surface; host boundary vs neural decision)
- **READ (the #76 neural mechanism, reused-by-import):** each turn the message's ENGAGEMENT (novelty of its content vs the session + richness — the SAME host-comprehension/sensory boundary the SVO parser / #84 appraisal / #85 topic occupy) is EMA-folded into a persistent per-session engagement scalar (a content-free turn HOLDS it → cross-turn persistence). This scalar is the ENVIRONMENTAL reward/context afferent the #76 finding names as its residual. It is mapped to the SNc afferent current and driven into the spiking SNc nucleus; the SNc FIRES; the neuromodulator bus (`from_region_firing_signed` on `["snc"]`) reads its rate and SELF-PRODUCES the tonic DA concentration — NEVER a host `set_concentration`. The FELT DA LEVEL is that bus concentration; the MODE is the level binned rest/neutral/focus/arousal (tonic DA 0.5; <0.40 rest, 0.62–1.00 focus, ≥1.00 arousal). The substrate is built ONCE per session and its full dynamic state snapshotted, so each read is a deterministic function of THIS turn's afferent (the cross-turn persistence lives in the host EMA, like #84's body-state).
- **DRIVE (the load-bearing surface change):** the neural DA mode → a graded ENGAGEMENT SUFFIX APPENDED to the answer surface — REST/NEUTRAL → `""` (byte-identical; the withhold/floor), FOCUS → `" — worth going further here."`, AROUSAL → `" — there's plenty more to dig into here!"`. Chosen as a SUFFIX (distinct from the #84/#85 prefixes) on the engagement/arousal axis (distinct from valence/topic) so the graded length/engagement change is orthogonal and the lesion produces a CLEAN byte-identical vanish. The suffix is an honest EXPRESSION of the brain's approach/engagement state (the dopaminergic Go/NoGo action-readiness switch — Albin-DeLong-Penney; Gerfen & Surmeier 2011), NOT content: the FACT before it is the SAME gate-matched, moat-verified answer.

## Result — the DELIVERABLE, through the REAL handler (artifact `research/findings/raw/_da_drives_chat/verify.json`, verdict GO)

### (A) The DA mode tracks the conversation — a dull opening reads REST, an engaging/novel exchange reaches FOCUS/AROUSAL
One `/api/brain-chat` session, `BRAIN_DA_DRIVES` default-on. Each turn's `da_drives` read (the self-produced DA level → mode → suffix):

| # | user turn | kind | self-produced DA | mode | engagement suffix |
|---|---|---|---|---|---|
| 0 | hi | rest | 0.04616 | rest | `''` |
| 1 | ok | rest | 0.04616 | rest | `''` |
| 2 | what does the dog chase | engaged | 0.701 | **focus** | ` — worth going further here.` |
| 3 | what colour is the sky ocean mountain forest river | engaged | 1.0253 | **arousal** | ` — there's plenty more to dig into here!` |
| 4 | photosynthesis chloroplast thylakoid electron transport | engaged | 1.0408 | **arousal** | ` — there's plenty more to dig into here!` |

**The mode MOVES sensibly: rest-DA ≤ 0.04616, engaged-DA ≥ 0.701 (monotone, engaged > rest), the very-novel+rich exchange reaches AROUSAL, and the engagement suffix is present IFF the mode is focus/arousal (suffix-iff-engaged = True, every row).** A content-free turn (`hi`/`ok`) holds the engagement at the floor → REST → no suffix (the withhold/byte-identical case).

### (B) The DA mode DRIVES the response — message FIXED, induced focus vs rest changes the reply; the neural lesion collapses it
Same fixed probe `"what does the dog chase?"`, the DA mode INDUCED by the SNc reward/context afferent (message held fixed; CLEAN separate sessions — the #84 session-leak lesson):

- **focus-induce** (SNc afferent 1300 pA → self-produced DA **1.2387** → AROUSAL): suffix present, answer `"The dog chases cat. — there's plenty more to dig into here!"`
- **rest-induce** (SNc afferent 100 pA → self-produced DA **0.04616** → REST): no suffix, answer `"The dog chases cat."`

`intact_diff = True` (the answers + suffixes differ); `base_identical = True` (the fact under the suffix == the rest-arm answer, `"The dog chases cat."`); `content_identical = True` (the abstain/recall/verify md5 is identical across the two arms). So the SAME message yields a DIFFERENT reply solely because the neural self-produced DA level differs — the dopaminergic mode is load-bearing on the surface.

**LESION (`BRAIN_DA_DRIVES_LESION=1`, the #76 anti-cheat-2 — silence the SNc nucleus):** the SAME focus-induce afferent (1300 pA) can NO LONGER raise the level: self-produced DA collapses to **0.04616** (REST), suffix `''`, answer `"The dog chases cat."` == the base. So the engagement suffix VANISHES and the answer reverts byte-identically to the no-suffix base — the surface change RIDES the SPIKING SNc→DA read, not a host `if engagement>x`: kill the neural DA nucleus and the mode-difference disappears even though the world input (an engaging afferent) is unchanged.

### (C) No-regression — content mode-invariant + byte-identical-off
- **C1 (rest byte-identity):** a rest panel (`hi`·`ok`·`no`, content-free → mode rest → no suffix) run OFF then default-ON under identical per-turn RNG. OFF response never carries a `da_drives` key; ON always does; **ON-minus-`da_drives` == OFF (md5 `af57eeaf`) on every turn**, with an empty suffix. Enabling the DA mode changes NOTHING on a rest turn but adds the additive read → byte-identical to pre-wiring.
- **C2 (content-invariance under an ACTIVE mode):** the same fixed probe under {off, focus-induce, rest-induce} — the CONTENT fields (`abstained`/`recalled_svo`/`verified`) md5 is identical (`350812dd`) across all three; only the engagement suffix differs. The mode decorates the surface, never a fact.

## Why the DA read does not perturb the rest of the pipeline (the RNG-isolation fix, inherited from #77/#84)
`PM.build` reseeds `cfg.seed` (the substrate build draws from the process-global RNG). `DaModeDrivesWorkspace._isolated` runs the substrate build + the SNc→DA read on the workspace's OWN private RNG timeline and restores the host RNG afterward, so the DA read is RNG-neutral to the host pipeline — the OTHER response fields stay byte-identical (the #77 footgun, handled the standard save/restore way). The C1 byte-identity confirms it end-to-end; a standalone check confirms the host RNG is byte-untouched after the reads.

## Anti-cheats
- **Additive + reversible:** `_DA_DRIVES_DEFAULT_ON=True` is the production anchor; `BRAIN_DA_DRIVES=0` → the block is fully skipped (no substrate built, no `da_drives` key, no suffix) → byte-identical oracle (C1: OFF carries no key, a rest ON-minus-key == OFF md5 `af57eeaf`).
- **Load-bearing, not cosmetic:** the engagement suffix rides the neural self-produced DA LEVEL — the SAME message reads focus-or-rest by the induced afferent (B intact), and the neural SNc-nucleus lesion collapses the level → the suffix vanishes → == base (B lesion). Content byte-identical throughout (B/C).
- **The level is neural (the #76 anti-cheats, reused):** the DA concentration is produced by SNc spikes off the bus (no `set_concentration` in the loop); silencing the SNc nucleus collapses the self-produced level regardless of the afferent (the #76 anti-cheat-2, reused as the drive lesion). The host supplies only the engagement scalar (world input) + the mode→suffix STRING (articulation scaffold).
- **Never crashes a turn:** `observe_turn` is guarded (any error → an inert no-suffix info dict); a wiring failure degrades to the unchanged turn.

## Honest limits / remaining scaffolds (named, not claimed closed)
1. **The message→ENGAGEMENT scalar (novelty + richness → the SNc reward/context afferent) is host** (a language/sensory-comprehension boundary, the SVO-parser boundary). The DA LEVEL (set by SNc spikes off the bus) + its SNc-nucleus dependence ARE the #76 neural mechanism (lesion-proven). The #76 finding itself flags the reward/context scalar's ORIGIN as its residual; computing it from the brain's own sensory stream is a SEPARATE faculty (the named next rung).
2. **The mode→SUFFIX-STRING map is a HOST conditioned-articulation scaffold** (the discourse "mouth") DRIVEN by the neural level (owner-sanctioned articulation-crutch — the lesion collapses the suffix); a brain-native spiking engagement mouth is the named next rung.
3. **Operating-point + reward-origin residuals inherited from #76** (bounded): the reconfiguring regime (str tone 40) + the SNc→DA gain are tuned so the self-produced level spans the mode band; the afferent-to-a-DA-nucleus form is faithful (VTA/SNc receive reward-carrying afferents) but the scalar's ORIGIN is host.
4. **FUNCTIONAL correlate, NOT a phenomenal claim** — the "mode" is a dopaminergic state read (rest/focus/arousal), never asserted as felt arousal.
5. **CO-RESIDENT** on its own #76 BG substrate, run alongside the recall composer — rides the one-brain merge (burn-down #1).

## Files
`webapp/da_mode_drives_chat.py`, `webapp/server.py` (anchor + `_da_drives_on()` + the per-turn read block + the `da_drives_suffix` append / `da_drives` attach on both return paths), `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (row `da-mode-drives-response`). Artifact: `research/findings/raw/_da_drives_chat/verify.json` (+ `verify_da_drives.py`). Reuse-by-import from `research/runners/_neuromod_spiking_da_mode_derisk` (#76 6/6 GO) → `_neuromod_reconfiguration_derisk` (#64) + `_perturb_and_measure_derisk` (#63); NO `sim/` edit. de-risk findings `2026-08-19-neuromod-spiking-da-mode-GO` / `2026-08-19-neuromod-reconfiguration-GO`.
