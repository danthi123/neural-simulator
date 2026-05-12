# iter OO contingency — smaller pools, larger lang_input

**Trigger:** If iter NN (orthogonal codes) also fails.

## Hypothesis

Per-seed structural pool_1 bias at biological scale scales linearly
with internal recurrent connections (N × density × weight). iter LL
500-neuron pools with 0.05 density = 25 recurrent inputs/neuron.

iter AA's success at 100-neuron pools (5 recurrent inputs/neuron) had
low enough recurrent activity that lang_input drive dominated the
pool firing pattern (i.e. discrimination was input-driven, not
recurrent-driven). At biological scale, recurrent dominates.

iter OO test: keep biological lang_input (n_lang_input=2048) and
training events (400) but REVERT pool size to iter AA (100 neurons,
12 FS, 200 lang_out_pool). This is "iter AA architecture but more
lang_input."

If iter OO passes 6/6: the issue was specifically pool size, not
overall scale. Lang_input scale is fine; pool scale destroys
discrimination.

If iter OO matches iter AA at 4/6: lang_input scale doesn't change
anything — iter AA's ceiling is preserved regardless of lang_input
size.

## Configuration

```bash
python -m research.runners.validate_ventral_semantic --seed 42 \
    --n-train-events 400 --n-replay-cycles 40 \
    --n-lang-input 2048 \
    --enable-multi-pool-wernicke --n-wernicke-pools 2 \
    --n-per-wernicke-pool 100 --n-per-wernicke-pool-fs 12 \
    --interleaved-training \
    --enable-per-concept-lang-out-pools --n-per-lang-out-pool 200 \
    --apply-wernicke-topographic \
    --wernicke-topographic-factor 1.5 \
    --wernicke-off-target-factor 0.7 \
    --n-recognition-trials 1 \
    --out research/findings/raw/g11_bg/iter_OO/iter_OO_seed42.json
```

Only differences from iter LL: pool sizes back to iter AA (100/12/200);
multi-trial averaging OFF (iter AA used 1 trial); standard
topographic factor (1.5/0.7).

## Time

~9-10 min (similar to iter AA at toy scale, slightly slower due to
larger lang_input).
