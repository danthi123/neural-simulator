"""FineWeb-Edu -> plain .txt corpus (2026-09-06, mouth token-scaling fork, STEP 2 build-ahead). Companion to
`lm_fineweb_setup.py` (which tokenizes to a cached uint16 npy for the SEPARATE `lm_train_run` scalable-training
path) -- this script instead streams FineWeb-Edu straight to a PLAIN-TEXT corpus file in the SAME flat format as
`data/corpus/wikitext103.txt`/`simplewiki.txt` (one document per line, whitespace-normalized), so it is a DROP-IN
`--corpus`/`--eval-corpus` argument for `research/runners/_emerge_wkv_lm_derisk.py` -- NO runner change needed.

WHY A SEPARATE SCRIPT (not the npy path): the mouth token-scaling fork's hard requirement is COMPARABILITY to the
2026-09-05 wt103(+simplewiki)-trained / wt103-held-out-eval baseline (see
`research/findings/2026-09-05-mouth-token-scaling-step1-simplewiki-domain-mix-NO-GO.md`), which is measured by
`_emerge_wkv_lm_derisk.py`'s own `load_stories`/`fit_interp_trigram`/`eval_perdepth` + anti-cheats on the
wikitext103 held-out deep-context buckets. That instrument reads `--corpus`/`--eval-corpus` as RAW TEXT FILES
(`load_stories` regex-tokenizes to WORDS, `load_sentences` splits on sentence punctuation) -- an npy file of
pre-tokenized ids (a DIFFERENT tokenizer's vocab, `lm_fineweb_setup.py`'s own from-scratch 16k BPE) cannot feed
either loader without rewriting the eval instrument itself, which is the one thing the fork's spec rules out
("NO change to the eval instrument"). A plain FineWeb .txt is therefore the only byte-comparable path.

MEMORY SAFETY: streams document-by-document (`datasets.load_dataset(..., streaming=True)`), writes each doc as it
arrives, and never buffers the corpus -- the SAME bounded-memory shape as `lm_fineweb_setup.py`. Downstream,
`_emerge_wkv_lm_derisk.py --contiguous` (`load_stories`) ALSO never reads the whole file -- it bounds its read to
`n_sentences * max_len * 8` bytes from the START of the file -- so an arbitrarily large output file here is safe to
consume later regardless of how big `--target-words` is set; the ONLY memory-relevant knob on the consumer side is
`--n-sentences`/`--max-train-sents` (how many passages get materialized as Python objects), not this file's size.

--target-words counts WORD-regex matches (`load_stories`'s own `[a-z']+` tokenization unit), NOT final BPE
sub-word tokens -- the BPE tokenizer (`--tokenizer bpe`, `wkv_bpe8k.json`) used downstream typically yields
~1.15-1.3x as many sub-word tokens as whitespace words for English prose (measured range across the wt103/
simplewiki BPE runs in this lane), so `--target-words 500_000_000` corresponds to roughly ~575M-650M *training*
(BPE) tokens once tokenized by the LM runner -- report the exact ratio for THIS corpus once tokenized (the
--tok-cache pass logs the true token/word count).

Run (smoke, seconds): python -m research.runners.lm_fineweb_to_txt --out /tmp/fw_smoke.txt --target-words 200000
Run (production slice): python -m research.runners.lm_fineweb_to_txt --out data/corpus/fineweb_edu.txt \
    --target-words 700000000 --subset sample-10BT
"""
import argparse, re, time
from pathlib import Path

_WS = re.compile(r"\s+")


def stream(subset):
    from datasets import load_dataset
    return load_dataset("HuggingFaceFW/fineweb-edu", subset, split="train", streaming=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .txt path (one FineWeb-Edu document per line)")
    ap.add_argument("--subset", default="sample-10BT",
                    help="FineWeb-Edu HF config name (sample-10BT/sample-100BT/sample-350BT/default); "
                         "sample-10BT matches the already-running sibling npy prep (data/corpus/fineweb_edu_run/) "
                         "for consistency of provenance, and is far larger than any --target-words used here.")
    ap.add_argument("--target-words", type=int, default=500_000_000,
                    help="stop once this many whitespace-ish words have been written (load_stories's own token "
                         "unit; see module docstring for the word->BPE-token inflation caveat)")
    ap.add_argument("--min-doc-words", type=int, default=20,
                    help="skip documents shorter than this (matches load_stories's own min_len-style filtering "
                         "intent -- avoid polluting the passage pool with fragments)")
    ap.add_argument("--log-every-words", type=int, default=25_000_000)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); words_written = 0; docs_written = 0; docs_skipped_short = 0; next_log = args.log_every_words
    with open(out, "w", encoding="utf-8") as f:
        for ex in stream(args.subset):
            text = ex.get("text", "")
            if not text:
                continue
            flat = _WS.sub(" ", text).strip()          # ONE line per doc (matches wikitext103.txt's flat-line shape)
            n_words = flat.count(" ") + 1 if flat else 0
            if n_words < args.min_doc_words:
                docs_skipped_short += 1
                continue
            f.write(flat); f.write("\n")
            words_written += n_words; docs_written += 1
            if words_written >= next_log:
                rate = words_written / (time.time() - t0) / 1e6
                print(f"  {words_written:,} words  {docs_written:,} docs  ({rate:.2f}M words/s)  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
                next_log += args.log_every_words
            if words_written >= args.target_words:
                break
    print(f"[done] {out}  words={words_written:,}  docs={docs_written:,}  skipped_short={docs_skipped_short:,}  "
          f"bytes={out.stat().st_size:,}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    # WORKAROUND (observed 2026-09-06 smoke): the HF `datasets`/pyarrow streaming iterator can leave a native
    # thread alive that segfaults during normal CPython interpreter teardown (PyGILState_Release on a thread
    # state that is no longer current) -- happens AFTER the corpus file is fully written+closed (the `with open`
    # block above already exited and `[done]` was printed), so it is a clean-exit cosmetic issue, not a data
    # integrity one. os._exit(0) skips the finalizer sequence that crashes, giving callers (gpu_queue.sh, a
    # background `&`, etc.) a clean rc=0 instead of a core-dumped non-zero exit.
    import os
    os._exit(0)
