---
type: finding
status: verified
date: 2026-09-03
mechanism: linattn spiking mouth (LinAttnReadout + the genuine few-spike Izhikevich FewSpikeWordRead WTA read,
  `webapp.wkv_mouth_generator`) — free-generation coherence check against its own already-measured
  trigram-crossing NLL, plus a like-for-like control against the same-arc ssm/dual-nonneg NO-GO family.
lane: language (own-voice mouth / retire the Qwen scaffold)
---

# linattn spiking mouth: does the trigram-crossing NLL translate into coherent generation? — viable, with two real caveats

**Question.** `research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.log` measured the linattn
recurrence (`--recurrence linattn --n-layers 2 --uniform-decay`, BPE V=8001, d_model=192, SimpleWiki corpus)
crossing the fair-trigram bound on next-token NLL at depth 10-99: seed 42 `margin_vs_trigram +0.049`, seed 43
`+0.053` (both logged as `-> GO` by that runner's own per-seed gate; transcribed into
`research/findings/raw/_linattn_mouth_gen_check/prior_nll_crossing_quoted.json` since the runner's own `.json`
artifact does not exist yet — the 6-seed run was still training seeds 44/100/101/102 at verification time).
Beating a trigram on TEACHER-FORCED
next-token perplexity is not the same claim as producing coherent free-running generation — a model can win a
narrow one-step statistic while its own multi-step rollout degenerates. This finding drives the actual mouth
(the numpy `LinAttnReadout` state machine + the genuine few-spike Izhikevich `FewSpikeWordRead` population-coded
WTA read, exactly as `webapp/wkv_mouth_generator.py` wires them for production) and reads the free-generated text
by eye against 10 prompts, on the two checkpoints that exist (seed 42, seed 43 — the 6-seed BPE-linattn training
run was still in progress on seeds 44/100/101/102 at verification time, and this check never touched it).

**Constraint discipline (why CPU-only).** The 6-seed linattn training (`bridges/wkv_ckpt/wkv_linattn_depth2_
contiguous_seed{seed}.npz`, PID 1178865 at verification time, `nvidia-smi` showing 100% util / 19.5GB) owned the
GPU throughout. `FewSpikeWordRead._build_bank` builds a real `SimulationBridge` (512 Izhikevich neurons: `topk=64`
candidate-word pools x `pop=8`), and `sim.backend.get_backend()` defaults to CuPy whenever a GPU is present and
`SIM_BACKEND` is unset — so every invocation below ran with `SIM_BACKEND=numpy` set BEFORE process start (that
env var is read-and-cached on first call, so it must precede any import) and `CUDA_VISIBLE_DEVICES=""` for a
second, belt-and-braces guard. Every generation log line below confirms `SIM_BRIDGE: NumPy backend (CPU (NumPy
backend)...)`. This script only `np.load()`s the already-saved checkpoint files and runs the tiny 512-neuron bank
on CPU; it never touched cupy or the running training process.

## How the mouth was driven

The minimal path is the ALREADY-BUILT production entry point, `webapp.wkv_mouth_generator.generate()`, with its
existing (default-off) linattn/BPE routing flags — no new generation code was written. From a fresh process
(`SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES="" python ...`):

```python
import os
os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "linattn"
from webapp import wkv_mouth_generator as wmg
wmg._CKPT_TEMPLATE = ".../bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"  # module-global,
wmg._CKPT_CACHE.clear()                                                                    # read-at-import-time
text, secs = wmg.generate(prompt, seed=42, max_new_tokens=60, topk=64, read_window=40, pop=8, gen_temp=0.8)
```

`_CKPT_TEMPLATE` is computed once from `BRAIN_WKV_MOUTH_CKPT` at module import (not re-read per call), so
switching checkpoints/recurrence families across calls in one process means overwriting that module global
directly (`recurrence_mode()`/`tokenizer_mode()` DO re-read their env vars per call, so those two toggle freely).
`generate()` internally builds `LinAttnReadout` (`_get_readout`), a `FewSpikeWordRead(topk, pop, seed,
read_window=read_window)` reader, and drives `_free_gen_linattn` — the genuine few-spike spiking WTA read
(`reader.read(p)`, population firing off `cp_firing_states` over `read_window` Izhikevich steps), not a host
argmax/softmax-sample. `sentence_facts=None`/`facts=None` (defaults) keep the fact-to-sentence and fact-boost
levers off, so this is pure free-running generation. Full driver script:
`research/findings/raw/_linattn_mouth_gen_check/linattn_gen_check.py`; full run log (confirms NumPy backend on
every call): `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_check_full_stdout.txt`; structured
output: `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_results.json`.

**Control.** For a like-for-like ablation (same corpus, same BPE vocab, same d_model=192, same
`depth2_contiguous` training recipe — the ONLY thing that differs is the recurrence family), the same script also
drives `WKVReadout` against `wkv_ssm_bpe8k_d192_simplewiki_depth2_contiguous_seed{42,43}.npz` — the ssm/dual-nonneg
family that the SAME 6-seed BPE arc already measured NO-GO on this exact corpus/scale
(`research/findings/raw/_emerge_wkv_lm_ssm_depth2_contiguous_6seed.log`: seed 42 `margin_vs_trigram -0.125`, seed
43 `-0.125`, both `-> no-go`). This answers "does the measured NLL gap (+0.049/+0.053 vs -0.125) show up as a
VISIBLE generation-quality difference, or is it too small to matter in practice."

10 prompts, both families, both seeds, `max_new_tokens=60`, default decode knobs (`topk=64, gen_temp=0.8,
repetition_penalty=1.0, no_repeat_ngram_size=0` — i.e. exactly `generate()`'s own shipped defaults, nothing tuned
for this check).

## Actual samples — DEFAULT decode settings (seed 42, linattn)

```
'The sun is'
  -> he sun is a member of the national assembly of the national assembly of the senate from to he was
     the president of the senate from to he was elected to the senate from to he was elected to the
     election to the candidate for the election to the election to the election to the election candidate

'My favorite food is'
  -> y favorite food is a american romantic drama movie directed by robert byzantine and starring mike
     pelin and stars robert byzantine and was distributed by the movie director of the movie was produced
     by robert batia and was nominated for the movie the movie was nominated

'Water is made of'
  -> ater is made of two small islands and the other islands are the largest island in the world the
     island of the island of the island of the island of the island of the island of the island of the
     island of the island of the island of the small provinces of the island of the island of the island of an
```
(seed 43, linattn)
```
'A dog is an animal that'
  ->  dog is an animal that is a kind of spiny it does not have a spiny called spiny spiny the spiny
      spiny spiny spiny spiny spiny spiny spinker mathematicy has been used in the photo

'Once upon a time there was'
  -> nce upon a time there was a small town in the town of nessy in the town of neng neng nensen was a
     municipality in the district of neng in the canton of ticino in inhabitants szschen wolf he want to go to the
```
Full 20 samples (10 prompts x 2 seeds) in `linattn_gen_check_full_stdout.txt` lines 1-267.

**Reading these**: real, well-formed multi-word English phrases appear throughout — "a member of the national
assembly", "an american romantic drama movie directed by X and starring Y", "the largest island in the world",
"a small town in the town of X" — genuine subject-verb-object clause structure and correct noun-phrase
composition, in a register that is instantly recognisable as Simple-Wikipedia biography/geography/filmography
boilerplate (the training corpus). This is qualitatively different from what a trigram can do: a trigram
conditions on only the last two tokens and cannot hold a noun phrase's agreement or a clause's structure across
5-10 tokens the way "directed by robert byzantine and starring mike pelin and stars robert byzantine" does.
**But under these DEFAULT decode settings every single sample degenerates into a hard repetition loop** within
15-25 tokens ("of the island" x9, "spiny" x8, "the election" x5, "the movie...the movie...the movie") — the
classic greedy/low-diversity-decoding failure mode, well documented in the LM literature (Holtzman et al. 2019,
"The Curious Case of Neural Text Degeneration").

## Actual samples — repetition-guard ON (`repetition_penalty=1.3, no_repeat_ngram_size=3`, still linattn)

The module already ships a default-OFF decode-time repetition guard (`_apply_repetition_controls` in
`webapp/wkv_mouth_generator.py`) built for exactly this failure mode. Turning it on (no other change) removes the
loops almost entirely:

```
'The sun is' (seed 42)
  -> he sun is a member of the national assembly from to and was elected into the new york city council
     in he became president for years before becoming mayor of chicago illinois in june he moved to
     california state senate on april he studied at romont university after graduating with his first term in s

'A dog is an animal that' (seed 42)
  ->  dog is an animal that lives in the sheep and other animals they are a small fish called tope or
      bacteria it can be found on the island of kitts and saturn by the sea of mesoid alfoid fumes which
      have been known as dinner

'My favorite food is' (seed 42)
  -> y favorite food is a american romantic drama movie directed by robert byzantine and starring mike
     deutsche finger in the movie was based on the novel of the same name it stars jaws heim kadham
     sherik sabha thorn ma

'In the beginning of the story' (seed 43)
  -> n the beginning of the story is a american romantic comedy movie directed by napoleon and starring
     he was born in los angeles raids his first novel was married to john mcgregor she had two children
     with her sister willic sley died at this time after marshy will
```
Full 20 samples: `research/findings/raw/_linattn_mouth_gen_check/linattn_repguard_results.json` (script:
`linattn_gen_check_reppen.py`).

**This is a materially more coherent reading.** Correct passive voice ("was elected into", "was based on the
novel of the same name"), correct multi-clause chaining with plausible real-world entity progression ("elected
into the new york city council ... became president for years before becoming mayor of chicago illinois ...
moved to california state senate ... studied at [university]"), correctly formed complex NPs ("an american
romantic drama/comedy movie directed by X and starring Y"). This sustains grammatical well-formedness over
20-40 tokens, well beyond a trigram's structural reach. It is honestly still: (a) genre-locked to Simple-Wikipedia
biography/filmography prose regardless of the prompt's actual topic (the "food" and "dog" prompts drift into
movie/biology-textbook register almost immediately — a known, separately-documented topical-faithfulness
residual, not something this check discovers new), and (b) prone to inventing non-word fragments deeper into a
generation ("mesoid alfoid fumes", "kadham sherik sabha thorn", "polifeldorf willid rosson whinseted") — a
genuine degeneration mode distinct from repetition, most visible past ~30 tokens.

## Control samples — same-arc SSM/dual-nonneg mouth (NO-GO family), same prompts, same seeds

DEFAULT decode settings, seed 42 — four of ten prompts return a **byte-identical** continuation regardless of
the prompt's own words:

```
'The sun is' / 'The city of London is' / 'The most important thing about science is' / 'My favorite food is'
  -> [prompt] a list of the same time in the same time in the united states and the other parts of the
     same time in the same time it was the first time it was a major league of the same time in the world
     and worked on the other side of the same time in the movie
```
i.e. the SSM/dual-nonneg recurrence at this scale collapses to a small number of PROMPT-INDEPENDENT attractor
continuations — not merely repetitive, but insensitive to what was actually typed. With the identical repetition
guard turned on (same `1.3`/`3` settings), this does not go away: seed 42's "The sun is", "The city of London
is", and "The most important thing about science is" still converge to the exact same 60-token continuation
verbatim (`ssm_repguard_results.json`), and the text that IS prompt-sensitive reads far more like word-salad than
the linattn samples above: *"he was a big more than one of these types are in the movie and other parts of the
world to the first people who has been used for the game boy advance from the ymunlargest city on may he was
made into a new yorz"* — broken subject-verb agreement, a non-word ("ymunlargest", "new yorz"), no sustained
clause structure. Full comparison artifacts:
`research/findings/raw/_linattn_mouth_gen_check/linattn_gen_results.json` (both families, default decode) and
`research/findings/raw/_linattn_mouth_gen_check/ssm_repguard_results.json` (control, repetition-guard on).
Scripts: `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_check.py`,
`research/findings/raw/_linattn_mouth_gen_check/ssm_gen_check_reppen.py`.

**This directly answers the "is linattn visibly better" question, honestly: yes, on two distinct axes** — (1)
prompt-sensitivity (linattn's openings visibly track the prompt's own words and topic register; the SSM control
frequently ignores the prompt outright and reproduces a fixed attractor), and (2) local grammaticality (linattn
sustains correct multi-word syntactic structure; the SSM control's non-repeated portions are closer to word-salad
with broken agreement and invented non-words). The measured NLL gap (+0.05 vs -0.125) is small in absolute terms
but corresponds to a qualitative, easily-eyeballed difference in generation, not a distinction that only shows up
in aggregate statistics.

## An honest new failure mode found in passing: the BPE tokenizer silently eats capital letters

Every sample above is visibly missing the FIRST LETTER of the prompt ("The sun is" -> "he sun is", "A dog is..."
-> " dog is...", "Water is made of" -> "ater is made of", "Yesterday..." -> "esterday...", "My favorite..." ->
"y favorite...", and mid-sentence "London" -> "ondon"). This reproduces identically for BOTH recurrence families
(it is a shared prompt-encode/decode defect, not a linattn-specific bug) and is now fully diagnosed:
`bridges/wkv_ckpt/wkv_bpe8k.json`'s vocabulary was built from a corpus that had no meaningful uppercase content
(`sorted(set(c for c in "".join(bt.vocab) if c.isupper()))` -> only `['K','N','U']`), so `BPETokenizer.encode()`
maps any other uppercase letter to `<UNK>` (id 0) — and `BPETokenizer.decode()` (`sim/bpe_tokenizer.py:92-108`)
explicitly does `if sym == "<UNK>": continue`, silently DROPPING the character rather than rendering a
placeholder. Directly reproduced (`research/findings/raw/_linattn_mouth_gen_check/bpe_bug_check.py`):

```
'The'    -> ids=[0, 3847]  syms=['<UNK>', 'he</w>']        decode='he'
'A'      -> ids=[0, 16]    syms=['<UNK>', '</w>']          decode=''
'I'      -> ids=[0, 16]    syms=['<UNK>', '</w>']          decode=''
'Water'  -> ids=[0, 807]   syms=['<UNK>', 'ater</w>']      decode='ater'
'London' -> ids=[0, 5517, 2593] syms=['<UNK>','on','don</w>'] decode='ondon'
```
Standalone capitalised words ("A", "I") decode to an EMPTY string, not just a truncated one — visible above as
the double space in "esterday  went to the" (the dropped "I"). This affects every caller of `BPETokenizer.decode`
on capitalised input, not just this check; it was not previously noticed because no prior wiring of the BPE mode
had fed it naturally-capitalised prompt text end-to-end with visual inspection of the output. Added to
`research/FAILURE_LOG.md` per the repo's coverage discipline (not fixed here — this task was scoped to
verification, not repair).

## Verdict

**The linattn mouth IS generating genuine structured language, not word-salad — the measured trigram-crossing
NLL is not an instrument artifact that dissolves on inspection.** Its free-running output sustains real
multi-word syntactic structure (correct NP composition, subject-verb-object ordering, passive voice, multi-clause
chaining) well past what trigram-level statistics can support, and — checked directly against the SAME-scale
SSM/dual-nonneg family that failed this arc's own NLL gate — is visibly more prompt-sensitive and more
grammatical, not just marginally ahead on a held-out number.

Two honest caveats keep this from being a clean production-ready result:

1. **Under `generate()`'s own shipped DEFAULT decode settings** (`repetition_penalty=1.0,
   no_repeat_ngram_size=0`), **every one of the 20 base samples degenerates into a hard repetition loop**
   within 15-25 tokens. The module's own already-built repetition guard (default-OFF) fixes this convincingly,
   but is not the default — flipping it on (or an equivalent) is a real, cheap, concrete next step before this
   mouth is presentable in production, not a hypothetical one.
2. Topical faithfulness to the actual prompt is weak — generations drift within a few tokens into whichever
   Simple-Wikipedia genre (biography/filmography/geography) dominates the training distribution, regardless of
   what was asked. This is a pre-existing, separately-documented scope residual (the module's own fact-grounding
   and fact-to-sentence levers exist precisely because free generation alone does not stay on-topic), not a new
   finding, but it bears directly on "is this usable for genuine conversation" and should not be read past.

Scope: this is a 2-seed (42, 43) qualitative read of generation quality, matching the 2 checkpoints that existed
at verification time (the 6-seed linattn training's remaining seeds 44/100/101/102 were still on the GPU and
were not touched). It is not a quantitative 6-seed generalisation claim, and none is made here — the NLL
crossing itself already has its own 6-seed-in-progress measurement in `_emerge_wkv_lm_linattn_depth2_contiguous_
6seed.log`/`.json`, orthogonal to this check.

## Artifacts (inline paths)

- `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_check.py` — main driver (both families, default
  decode).
- `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_check_reppen.py` — linattn, repetition-guard on.
- `research/findings/raw/_linattn_mouth_gen_check/ssm_gen_check_reppen.py` — SSM control, repetition-guard on.
- `research/findings/raw/_linattn_mouth_gen_check/bpe_bug_check.py` — standalone repro of the tokenizer bug.
- `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_check_full_stdout.txt` — full stdout, both families,
  default decode, 20 samples each (confirms `NumPy backend` on every call).
- `research/findings/raw/_linattn_mouth_gen_check/linattn_gen_results.json`,
  `linattn_repguard_results.json`, `ssm_repguard_results.json` — structured per-prompt outputs.
- Checkpoints read (unmodified, already on disk before this check ran): `bridges/wkv_ckpt/
  wkv_linattn_depth2_contiguous_seed{42,43}.npz`, `bridges/wkv_ckpt/
  wkv_ssm_bpe8k_d192_simplewiki_depth2_contiguous_seed{42,43}.npz`, `bridges/wkv_ckpt/wkv_bpe8k.json`.
- Prior NLL measurement cited: `research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.log`
  (linattn), `research/findings/raw/_emerge_wkv_lm_ssm_depth2_contiguous_6seed.log` (SSM control).
- Code exercised (read-only, unmodified): `webapp/wkv_mouth_generator.py`
  (`_free_gen_linattn`/`_free_gen`/`generate`), `research/runners/_wkv_fewspike_read_derisk.py`
  (`LinAttnReadout`/`WKVReadout`/`FewSpikeWordRead`), `sim/bpe_tokenizer.py` (`BPETokenizer.encode`/`.decode`,
  where the caps-drop bug lives).
- Provenance: git SHA `b53ac47719391b6657a68c491247ff5d40865f52` at time of running; both scripts run as
  `SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES="" .venv/bin/python <script>.py` from a worktree checkout, so no
  `research/runners/__init__` auto-provenance sidecar applies (these are ad hoc verification scripts, not
  `-m research.runners.X` invocations) — recorded here manually instead.
