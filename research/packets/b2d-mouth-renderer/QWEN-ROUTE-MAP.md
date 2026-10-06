# Where the Qwen mouth speaks at production and console defaults (route map, 2026-10-06)

Written by Claude for B2d Amendment 2 (`research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md`).
Method: a code read of `webapp/server.py` (`brain_chat` -> `brain_reply`), `research/runners/brain_chat_tui.py`
(`ChatBrain`, `QwenRenderer`), `research/runners/rich_answer_composer.py`, `research/runners/affect_production_organ.py`
and `webapp/static/app.js` at main `864e8af33`, pinned by unit-level tests (`tests/test_b2d_amend2.py`, CPU, no brain
and no model), and read against two runs already in the record: the B2d mouth arm's 72 arm files
(`research/findings/2026-10-06-B2d-mouth-arm-affect-rows-mouth-silent.md`) and the per-touchpoint measurement
(`research/findings/raw/_per_touchpoint_qwen_share_shipped_default_AFTER_recall_gate_fix.json`). No brain was run for
this map. Line numbers are at `864e8af33` and drift; the function names do not.

## 1. The defaults that decide the route

| setting | console default | API / OpenAI shim default | source |
|---|---|---|---|
| renderer | picker "auto" sends `renderer: null` | omitted | `app.js` `rendererParam`; `server.py` `_default_brain_renderer` |
| renderer resolved | Qwen only when `SIM_BACKEND=cupy` and CUDA is up, else the template stub; `BRAIN_CHAT_RENDERER` forces one | same | `_default_brain_renderer` |
| `rich` | "Rich answers" unchecked: sends `rich: false` | omitted: `_brain_rich_default()` = True | `app.js` `richOn`; `server.py` `_brain_rich_default`; the shim's `BrainChatRequest` sets no `rich` |
| spiking recall mouth (`BRAIN_SPIKING_MOUTH_RECALL`) | on | on | `spiking_mouth_recall_prod.recall_mouth_enabled` |
| spiking Broca for hypotheses (`BRAIN_SPIKING_MOUTH`) | on | on | `ChatBrain._spiking_mouth_enabled` |
| open-ended generator (`BRAIN_OPEN_ENDED`) | off | off | `webapp/open_ended_chat.open_ended_enabled` |
| affect coloring (`BRAIN_AFFECT`) | on | on | `affect_production_organ.affect_enabled` |
| affect-drives marker on the reply (`BRAIN_AFFECT_MARKER_SURFACE`) | off (recorded, not spoken) | off | `webapp/affect_drives_chat.affect_marker_surface_enabled` |
| batched rich render (`BRAIN_RICH_BATCH_RENDER`) | off | off | `rich_answer_composer._batch_render_enabled` |
| long-term memory | `wikidata_100k` (else `wikidata_core_15k`) when `~/Projects/sim-data` exists | same | `server.py` `_default_ltm_bundle_dir` |

## 2. The only doors into Qwen

Qwen generates only inside `SpikingQwenFaculty._generate` / `_generate_batch`
(`research/runners/_grounded_lang_integration_derisk.py`). At the defaults these are reached by:

1. **`ChatBrain.render(svo)`** (single-fact path, and episodic Hook A's disclosure of an in-memory fact). It first
   calls `spiking_recall_surface(a, v, p)`, which returns a spiking-mouth sentence unless the fact cannot be framed
   (`spiking_mouth_recall_prod.frame_supported`: any role not purely alphabetic, a copula verb, or subject equal to
   object) or the spiking sentence fails its re-parse check. Only then does it call `chat.renderer.render_svo` (Qwen),
   and on a VERIFY miss `render_svo_regen` once.
2. **`RichAnswerComposer._render_one_verified`** (rich path), once per gathered fact, with the same spiking-first rule.
3. **`ChatBrain.render_hypothesis_verified`**: the spiking Broca first; the renderer only for a hypothesis the Broca
   cannot frame, or under `BRAIN_SPIKING_MOUTH=0`.
4. **The open-ended block** (off at defaults): calls the warm faculty's `model.generate` directly. The harness's call
   counter and the new mouth trace do not see this path; it is off in every Amendment 2 arm.

Every long-term-memory fact has underscored slugs (`angora_turkey`, `l_quipe_de_france`), so the spiking recall mouth
cannot frame it, and Qwen renders it. The per-touchpoint run shows this on the rich path: every known-topic row had
`spiking_hit_count` 0 and one renderer call per sentence. The tiny-demo's own facts (`dog chase cat`) are framed and
spoken by the spiking mouth, which is why A1.2's turns and the dev-seed-7 recall probe never reached Qwen. One
exception was seen: seed 100's `well` echo called Qwen once (`Wolf bites apple.`), consistent with a spiking re-parse
miss on that seed.

## 3. Every reply route in `brain_reply`, and whether Qwen can author it

| route (order in the handler) | trigger | who writes the reply | Qwen? |
|---|---|---|---|
| self-initiated utterance | empty or "say something" turn | the self-initiation organ's own RF mouth | no |
| vision identity | a turn carrying a percept | host read-out of the spiking HMAX category | no |
| BG stay-silent | the BG race commits to silence on a content-empty turn | `HOLD_TEXT` template | no |
| inner-state read-out | "how do you feel" | `feel_readout` template | no |
| open-ended generation | `BRAIN_OPEN_ENDED` (off) | the open-ended generator | yes, uncounted (off by default) |
| prospective-memory formation | "remind me when ..." | acknowledgement template | no |
| episodic recall (Hook A) | referential "what did I say ..." | disclosure template, plus `ChatBrain.render` of an in-memory fact | only by door 1 |
| worldview read-out, multi-referent hold/query, discourse register, false belief, causal why/what-if, comprehension "didn't follow", non-contradiction rejection | their own triggers | organ read-out or template | no |
| **rich path** | `rich` true (API and shim default) | the composer: per fact, the spiking mouth if it can frame it, else Qwen | **yes**, for every unframeable fact |
| **single-fact path** | everything else (console default) | gate, then `ChatBrain.render`; on no match the abstention template `I don't know about that.` plus a curiosity question | **yes**, for an unframeable recalled fact (or a spiking re-parse miss); never on an abstention |

After the rich and single-fact paths compose an answer, host text may be added around it: the curiosity question, the
DA-mode suffix (`— worth going further here.`), the common-ground, swap, wander, affective-ToM and global-stop leads,
the spiking recall mouth's tone mark (`!` or `.`). None of these is written by Qwen.

## 4. Which affect inputs reach the Qwen conditioning

- **Gate-B tone level.** The turn's appraisal (the learned DR-2 valence map, `research/findings/raw/
  _affect_learned_valence_map.json`; the raw Warriner norm when the file is missing) updates the session mood; the
  spiking graded-affect ladder reads it (`AffectProductionOrgan.read_differential`; by default the appraisal enters the
  ladder through the interoceptive relay); a host staircase quantizes the differential to `tone_level` (-3..3).
- **Into the Qwen prompt.** From the first affect turn of a session the renderer is wrapped in
  `MoodConditionedRenderer`. When `tone_level` is not 0 its `render_svo` appends a manner clause to Qwen's constrain
  prompt (`manner_template_for`: "warm, friendly" for >= 1, "blunt, matter-of-fact" for <= -1). `render_svo_regen`
  never carries the manner. The wrapper has no `render_svo_batch`, so the batched path stays off whenever affect is on.
- **Into how much Qwen says (rich path only).** `content_plan(tone_level)` sets the composer's sentence and
  elaboration caps (level >= 1: 4 and 3; neutral: 4 and 2; -1: 2 and 1; <= -2: 1 and 0); confidence-forthcomingness
  may then add one fact. More gathered unframeable facts means more Qwen sentences.
- **The lesions.** `BRAIN_AFFECT_LESION` clamps the ladder's `affect_out`, and `BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION`
  cuts the relay-to-ladder synapses: either way the differential falls toward 0, the level to 0, and both the manner
  clause and the positive content plan go. `BRAIN_AFFECT_DRIVES_LESION` acts on the #84 felt-state ladder, whose outputs
  at defaults are the recorded marker (`affect_drives.lead`, not spoken), its congruence gate (which acts on that marker
  only) and the open-ended gated turn (off). No code path carries it into the Qwen prompt or the rich plan.
- **Not Qwen.** The spiking recall mouth's tone mark (`_apply_mouth_mood_tone`: a spiking WTA picks `!` or `.`, host
  code appends it) is the affect path on spiking-mouth turns.

## 5. What this means for the B2d mouth arm

A turn on which Qwen speaks while the mood is engaged must (1) end on the rich or the single-fact path, (2) recall a
fact the spiking recall mouth cannot frame, at defaults a long-term-memory fact, and (3) be in a session whose
`tone_level` is not 0 when the fact is rendered. A1.2's `emo` fails (1) (the abstention template); `well` fails (2)
on 5 of 6 seeds and (3) on all six. Amendment 2's candidates follow this shape: the `emo` message, then a
long-term-memory question in the same session, single-fact (console) and rich (API) forms.

## 6. What the instrument can see

- `_arm_renderer.qwen_generate_calls` (harness, A1.1): how many Qwen calls a turn made. Blind to door 4.
- `mouth_trace` (new, `BRAIN_MOUTH_TRACE=1`; `webapp/server.py` `_mouth_trace`): each call the wrapper received, its
  SVO, the manner clause appended, generative or template, the surface, and whether it is in the answer. Attached on the
  rich and single-fact paths only; an earlier return carries no block. `wrapped: false` means "not traced", never "no
  calls" (affect off, or no affect turn yet in the session).
- Prompt capture (the Amendment 2 probe): the exact prompts Qwen received, split per turn by the counter.

Residual for the map itself: it is a code read plus unit pins. The first end-to-end check that a candidate turn
reaches Qwen with the manner clause is Amendment 2's dev-seed route arm, which the HANDOFF reads before any gate line.
