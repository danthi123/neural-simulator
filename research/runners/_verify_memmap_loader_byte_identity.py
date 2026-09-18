"""Byte-identity verifier for load_stories_memmap / build_or_load_passages vs the legacy load_stories.

This is the LOAD-BEARING correctness check for the token-supply scaling arc: the memory-efficient memmap loader
must produce EXACTLY the passages the legacy in-RAM loader did, or every prior scaling result becomes
incomparable and the extended-d384 curve is poisoned. Run it before relying on the memmap loader.

CPU-only, sub-second, tiny self-contained corpora (no big-file / GPU dependency):
  CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m research.runners._verify_memmap_loader_byte_identity

Exit 0 = byte-identical on every case; non-zero = a real divergence (do NOT run the decisive test).

History: written 2026-09-18 to verify the agent-built memmap loader (commit 229a1169). It found two real
edge-case bugs — an empty-corpus np.memmap-of-0-bytes crash and a token_cap<=0 off-by-one — both fixed in the
same landing. Boundary handling is exercised at chunk_chars=1 (a token boundary on every character).
"""
import os, re, sys, tempfile

import research.runners._emerge_wkv_lm_derisk as M

_LC = re.compile(r"[a-z']+")


def main():
    fails = []

    # CASE 1: streaming tokenizer == re.findall over the whole lowered read, for many tiny chunk sizes + token_cap.
    tricky = ("The QUICK brown Fox's tail; don't stop!!  ΟΔΟΣ Σoφia sigma-final "
              "ΛΟΓΟΣ.\r\n café naive Istanbul İstanbul KELVIN K end. "
              + "supercalifragilisticexpialidocious " * 40 + "\n"
              "a'b''c d\t\te   f'''  wonderfully-hyphenated words here\n") * 30
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(tricky); tricky_path = tf.name
    char_cap = len(tricky) * 4
    gold = _LC.findall(open(tricky_path, encoding="utf-8", errors="ignore").read(char_cap).lower())
    for cc in (1, 2, 3, 5, 7, 13, 16, 64, 4096, 1 << 22):
        got = list(M._stream_lc_word_tokens(tricky_path, char_cap, token_cap=None, chunk_chars=cc))
        if got != gold:
            k = next((i for i in range(min(len(got), len(gold))) if got[i] != gold[i]), min(len(got), len(gold)))
            fails.append(f"CASE1 chunk_chars={cc}: len got={len(got)} gold={len(gold)} first-diff@{k} "
                         f"got={got[max(0, k - 1):k + 2]!r} gold={gold[max(0, k - 1):k + 2]!r}")
        for tc in (0, 1, 5, len(gold) - 1, len(gold), len(gold) + 10):
            if tc < 0:
                continue
            capped = list(M._stream_lc_word_tokens(tricky_path, char_cap, token_cap=tc, chunk_chars=cc))
            if capped != gold[:tc]:
                fails.append(f"CASE1-cap chunk={cc} token_cap={tc}: got {len(capped)} want {len(gold[:tc])}")

    # CASE 2: build_or_load_passages passages == load_stories passages, incl. tail-partial, cap, and empty cases.
    base = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar".split()

    def make_corpus(n_tokens):
        return " ".join(base[i % len(base)] for i in range(n_tokens))

    cache_root = tempfile.mkdtemp(prefix="passcache_")
    for n_tokens, max_stories, max_len in [
        (1000, 10, 48), (1000, 5, 48), (100, 3, 48),
        (48 * 20, 20, 48), (48 * 20 + 7, 20, 48),
        (48 * 20 + 30, 25, 48), (500, 8, 24), (77, 2, 24),
        (48 * 6 - 3, 100, 48),
        (7, 100, 48),                                    # single partial <8 -> dropped -> empty (was the mmap crash)
    ]:
        corpus = make_corpus(n_tokens)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(corpus); cpath = tf.name
        legacy = M.load_stories(cpath, max_stories, max_len=max_len)
        cdir = os.path.join(cache_root, f"n{n_tokens}_s{max_stories}_l{max_len}")
        mm_list = list(M.build_or_load_passages(cpath, max_stories, max_len=max_len, cache_dir=cdir, verbose=False))
        if mm_list != legacy:
            fails.append(f"CASE2 n_tok={n_tokens} s={max_stories} l={max_len}: memmap n={len(mm_list)} "
                         f"legacy n={len(legacy)}")
            for i in range(min(len(mm_list), len(legacy))):
                if mm_list[i] != legacy[i]:
                    fails.append(f"   first-diff passage@{i}: memmap={mm_list[i][:6]} legacy={legacy[i][:6]}")
                    break
        mm2 = list(M.build_or_load_passages(cpath, max_stories, max_len=max_len, cache_dir=cdir, verbose=False))
        if mm2 != legacy:                                 # cache-HIT path must also match
            fails.append(f"CASE2-cachehit n_tok={n_tokens} s={max_stories} l={max_len}: hit != legacy")

    print("=" * 70)
    if fails:
        print(f"BYTE-IDENTITY FAILED -- {len(fails)} divergence(s):")
        for f in fails[:40]:
            print("  " + f)
        return 1
    print("BYTE-IDENTICAL: streaming tokenizer matches re.findall across 10 chunk sizes (incl. chunk_chars=1) "
          "+ token_cap; memmap passages == load_stories across 10 size/cap/tail/empty cases + cache-hit path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
