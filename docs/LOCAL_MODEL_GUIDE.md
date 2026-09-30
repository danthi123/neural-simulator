# Running a local model — the owner's guide (no Claude session needed)

This is everything you need to add a new local model, switch which one is the default, and check it's working,
without asking Claude to do it. The serving stack is: `llm.sh` (`tools/local_llm/llm.sh`, aliased to `llm` if
you followed its own header comment) starts **llama-swap**, a small proxy that runs the real model server
(`llama-server`, from llama.cpp) for you and can hold several models' configs at once, switching between them
on demand. Everything below is editing one file: `tools/local_llm/llama-swap.yaml`.

## 1. Download a new GGUF model file

Pick a quantized GGUF from Hugging Face and download it into `~/models/<something>/`, matching how the existing
ones are laid out (see `tools/local_llm/profiles.json` for examples). Example:

```bash
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf \
    --local-dir ~/models/qwen3.8-27b-gguf/q4km
```

(`hf` is the current Hugging Face CLI — installed already at `~/.local/bin/hf`. If a repo needs a login token,
`hf auth login` once, first.) Pick a quantization that fits the 3090 with room to spare: the models already in
use are IQ4_NL/IQ4_XS (~14-16 GB for a ~24-27B model) — check the file size on the Hugging Face page before
downloading a bigger one.

## 2. Add it as a model in `tools/local_llm/llama-swap.yaml`

Open the file. Find any existing entry under `models:` — say `devstral-small2-24b-iq4xs` — and copy its whole
block (from its name down through the `cmd: >` lines) to a new spot under `models:`. You only need to change
**3 things** in your copy:

1. The key name on the first line (the id, e.g. `devstral-small2-24b-iq4xs` → `my-new-model`).
2. The `-m /path/to/...gguf` path, right after `llama-server`, to point at the file you just downloaded.
3. The `-c <number>` (context size, in tokens) if your new model has a different context window than the one
   you copied from — check the model's own README on Hugging Face for its max context.

Leave every other flag alone (see the flag glossary below if you're curious what they do), **except** see the
"do not touch" note further down about `--chat-template-file`.

Then add a matching entry under `profiles:` (further down the same file) — copy an existing one, e.g.:

```yaml
  my-new-model:
    pins: { local: my-new-model }
```

using the SAME name you just picked. This is what lets `llm on my-new-model` work — without it, llama-swap
would have the model available but `llm` wouldn't know how to select it.

**Save the file. That's it — no restart needed.** llama-swap watches this file (`-watch-config`, set by
`llm.sh` when it starts) and picks up your edit within a couple of seconds.

## 3. Switch the default

The "default" is just what `llm on` (with no name given) loads. Two files agree on it:

- `tools/local_llm/default_profile` — a one-line text file. Change its contents to your new model's name.
- `tools/local_llm/llama-swap.yaml`'s `hooks: on_startup: profile:` line near the bottom — change it to match.

Then run `llm off` followed by `llm on` (or just `llm on my-new-model` directly) to switch right away, or leave
it and it'll take effect the next time the model gets loaded.

## 4. Check it works

```bash
llm status
```

This shows whether llama-swap itself is up, and which profile is actually resident in VRAM right now (it can
legitimately say "none" if the model auto-unloaded from being idle — see the TTL note below — the next request
reloads it). Then a one-line functional check, straight against the server:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
    -d '{"model":"local","messages":[{"role":"user","content":"say hi in five words"}],"max_tokens":20}'
```

A real reply back means it works end to end. If it hangs or errors, run `llm status` again (a first request to
a cold model can take a couple of minutes to load — that's normal) and check the log with:
`journalctl --user -u local-llm -n 60 --no-pager`.

## What the flags mean (one line each)

These are the pieces of each model's `cmd:` line in `llama-swap.yaml`:

- `-m <path>` — the GGUF model file to load.
- `--host 127.0.0.1 --port ${PORT}` — where llama-swap's own copy of llama-server listens internally;
  `${PORT}` is filled in automatically by llama-swap, a different number per model. Leave this exactly as-is.
- `--alias local` — the name this llama-server instance reports itself as; cosmetic only, llama-swap's own
  `profiles:` pins are what actually route requests.
- `-ngl 99` — offload all model layers to the GPU (as opposed to splitting some onto the CPU, which is slower).
- `-np 1` — one request "slot" at a time. Investigated and kept deliberately — see the comment above
  `profile_cmd()` in `tools/local_llm/llm.sh` if you want the full story; the short version is that a higher
  number here didn't help and this repo's actual prompt-caching problem turned out to be somewhere else.
- `-fa on` — flash attention (a faster, lower-memory way to compute attention); on for every profile here.
- `-c <n>` — the context window, in tokens (how much conversation history + prompt the model can see at once).
- `-ctk <quant> -ctv <quant>` — the KV cache's own quantization (its memory format), separately for keys (k)
  and values (v) — `q8_0` uses more VRAM at slightly higher precision than `q4_0`.
- `--jinja` — use the model's chat template (see below) instead of a generic built-in one.
- `--chat-template-file <path>` — see "do not touch" below.
- `--spec-type draft-mtp --spec-draft-n-max 3` — speculative decoding: a small "draft" model guesses a few
  tokens ahead and the main model checks them in one pass, which is faster when the guesses are right. Only
  present on profiles whose GGUF actually has this built in (MTP = multi-token prediction).
- `--temp / --top-p / --top-k` — sampling parameters (how random vs. deterministic the output is). Copied as-is
  from whichever profile you started from unless you have a specific reason to change them.

## Do NOT change the `--chat-template-file` line

Every `qwen38-27b-iq4nl-mtp*` profile points `--chat-template-file` at a patched copy of that model's chat
template under `tools/local_llm/templates/`. The model's own built-in (embedded) template rejects a
mid-conversation "system reminder" message that Claude Code sends every turn — without this patched file, every
`llm claude` session breaks partway through. An earlier, cruder fix for that same error also broke the model's
prompt-cache reuse without anyone noticing for a while (0% reuse instead of ~60%, meaning every turn reprocessed
the entire conversation from scratch) — see `tools/local_llm/results/cache_probe.md` for the full story. If you
add a brand-new model and llama-server logs a "System message must be at the beginning" error for it, that
means IT needs the same kind of patched template — copy the pattern in `tools/local_llm/templates/`, do not
just delete the `--chat-template-file` line to make the error go away.

## Idle auto-unload (TTL)

`llama-swap.yaml`'s `globalTTL: 3600` means a model that's been idle (no requests) for an hour unloads itself
automatically, freeing the 3090 for other work. The next request for it (or `llm on <profile>`) reloads it
transparently — this is expected, not a failure. Set `globalTTL: 0` to turn this off entirely, or add a `ttl:`
line under one specific model's own entry to override it just for that one.

## What NOT to do

- Don't edit the `-listen` address or remove `-watch-config` from `tools/local_llm/llm.sh`'s `llama_swap_cmd()`
  unless you mean to change how llama-swap is reached — the LAN address is there on purpose, for whatever else
  on the LAN needs it (2026-09-30: the sim-agent harness was retired; the owner runs Hermes Agent on
  homelab-server (`~/apps/hermes`) as a general harness; the sim driver is `tools/local_llm/llm.sh claude` on
  the workstation).
- Don't add an `apiKeys:` block to `llama-swap.yaml` without checking what else on the LAN currently talks to it
  unauthenticated — today nothing needs a key, so adding one would break existing callers until they're updated
  too.
