# Saved bridge checkpoints (HDF5 + sidecar JSON)

Files in this directory:
- `<name>.simstate.h5` — full bridge state (weights, regions, indices)
- `<name>.simstate.h5.meta.json` — sidecar metadata (mode, seed, training events, save date)

Listed via webapp `/api/bridges` endpoint.

Save by running:
```bash
python -m research.runners.chat_repl --mode synonym --seed 42 \
    --save-bridge bridges/synonym_8word_seed42.simstate.h5
```

Reload by running:
```bash
python -m research.runners.chat_repl --mode synonym --seed 42 \
    --load-bridge bridges/synonym_8word_seed42.simstate.h5
```
