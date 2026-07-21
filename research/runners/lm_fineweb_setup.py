"""FineWeb-Edu corpus setup for the LM-training workflow: stream FineWeb-Edu -> train+freeze a fast Rust BPE
(tokenizers) -> tokenize to uint16 tokens_{train,val}.npy in the run dir (the format lm_train_run.start reads cached).
--max-tokens caps the slice (start small to validate, then extend). No RAM blowup: preallocated uint16 array."""
import argparse, json, time
from pathlib import Path
import numpy as np


def stream(subset):
    from datasets import load_dataset
    return load_dataset("HuggingFaceFW/fineweb-edu", subset, split="train", streaming=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--max-tokens", type=int, default=500_000_000)
    ap.add_argument("--bpe-train-docs", type=int, default=30000)
    ap.add_argument("--val-frac", type=float, default=0.004)
    args = ap.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tk_path = root / "tokenizer.json"

    # 1. train + freeze BPE on the first N docs (if not already)
    if tk_path.exists():
        tk = Tokenizer.from_file(str(tk_path)); print(f"[bpe] cached vocab={tk.get_vocab_size()}", flush=True)
    else:
        print(f"[bpe] training vocab={args.vocab_size} on first {args.bpe_train_docs} docs...", flush=True)
        sample = []
        for i, ex in enumerate(stream(args.subset)):
            sample.append(ex["text"])
            if i + 1 >= args.bpe_train_docs: break
        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tk.decoder = decoders.ByteLevel()  # matching decoder -> decode() cleans the 'Ġ' space markers back to text
        tk.train_from_iterator(sample, trainers.BpeTrainer(vocab_size=args.vocab_size,
                                                           special_tokens=["<unk>", "<eos>"]))
        tk.save(str(tk_path)); print(f"[bpe] trained vocab={tk.get_vocab_size()}", flush=True)
    eos = tk.token_to_id("<eos>") or 0
    dt = np.uint16 if tk.get_vocab_size() < 65536 else np.int32

    # 2. stream + tokenize up to max_tokens into a preallocated array (memory-bounded)
    print(f"[tokenize] up to {args.max_tokens:,} tokens...", flush=True)
    arr = np.empty(args.max_tokens + 100000, dtype=dt); pos = 0; t0 = time.time(); nextlog = 50_000_000
    for ex in stream(args.subset):
        ids = tk.encode(ex["text"]).ids
        k = len(ids)
        if pos + k + 1 > args.max_tokens: break
        arr[pos:pos + k] = ids; arr[pos + k] = eos; pos += k + 1
        if pos >= nextlog:
            print(f"  {pos:,} tokens  ({pos/(time.time()-t0)/1e6:.2f}M tok/s)", flush=True); nextlog += 50_000_000
    arr = arr[:pos]
    cut = int(len(arr) * (1 - args.val_frac))
    np.save(root / "tokens_train.npy", arr[:cut]); np.save(root / "tokens_val.npy", arr[cut:])
    json.dump({"corpus": f"fineweb-edu/{args.subset}", "vocab_size": tk.get_vocab_size(), "tokenizer": "hf",
               "n_tokens": int(len(arr)), "n_train": int(cut), "n_val": int(len(arr) - cut)},
              open(root / "corpus_meta.json", "w"), indent=1)
    print(f"[done] {len(arr):,} tokens -> train {cut:,} val {len(arr)-cut:,}  vocab={tk.get_vocab_size()}  "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
