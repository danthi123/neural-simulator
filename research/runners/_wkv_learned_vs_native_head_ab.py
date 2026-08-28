"""A/B: e-prop LOCALLY-LEARNED WKV-mouth read-out head vs the checkpoint's NATIVE (host-trained/copied) head,
through the PRODUCTION `webapp.wkv_mouth_generator.generate()` entry point. (2026-08-28)

CONTEXT. `research/findings/2026-08-28-persist-eprop-head-scope.md` persisted an opt-in load path
(`BRAIN_WKV_MOUTH_LEARNED_HEAD` / `_LEARNED_HEAD_PATH`, both default-OFF) for the e-prop LOCALLY-LEARNED
read-out head `W_hat` -- trained by a local three-factor rule against the batched-substrate spiking forward,
NO weight transport, NO host gradient (`research/findings/2026-08-28-mouth-stale-coo-training-fix-fullscale-
confirmation-GO.md`: 6-seed `sub_recov_ratio_mean=0.8686`, min 0.8399). That finding named the concrete next
step (its own SS4): "a qualitative A/B against the native head on in-vocab prompts (self-NLL, coherence)".
THIS runner does that A/B.

HONEST SCOPE OF THE PERSISTED ARTIFACT (read before interpreting results). The eprop runner's `--save-w-hat
<path>` takes a LITERAL path with no `{seed}` templating; run across `--seeds 42,43,44,100,101,102` in that
order, each seed's `np.savez` OVERWRITES the same file -- so `wkv_eprop_learned_head_6seed.npz` (despite its
name) holds ONLY the LAST-processed seed's head: seed=102, `sub_recov_ratio=0.9132` (the BEST of the six
per-seed ratios in `eprop_persist_6seed.json`, not the 0.8686 SIX-SEED MEAN headlined by the finding this A/B
was scoped from -- a discrepancy this runner surfaces, not silently inherits). This A/B therefore necessarily
runs at seed=102 (the checkpoint `bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed102.npz` the head was trained
against) and reports the actual `sub_recov_ratio=0.9132` read from the npz itself, not the six-seed mean.

WHAT THIS MEASURES, per prompt (8 in-vocab TinyStories-domain prompts, verified via `in_vocab_scope`), each
condition generated via the SAME production `webapp.wkv_mouth_generator.generate()` call (same seed=102, same
`max_new_tokens`, same sampling params) -- only `BRAIN_WKV_MOUTH_LEARNED_HEAD` differs between the two arms:
  (1) self-NLL (nats) of the generated continuation under the ACTIVE head's own teacher-forced next-word
      distribution -- gate on the PREVIOUS token to predict the NEXT (`_free_gen`'s own convention, replicated
      here for the replay, NOT re-derived differently) -- vs chance `log(V)=6.9078` nats.
  (2) coherence beyond self-NLL: distinct-1/2/3 (type-token ratio over generated words/bigrams/trigrams -- a
      LOW ratio flags degenerate repetition self-NLL alone can miss, per the TINYSMOKE toy-head's own recorded
      failure mode "the big box box the big box and a big box").
  (3) LEVER sanity: `ro.head_w` must actually DIFFER between the native and learned arms (else the "learned"
      arm silently ran on the native head -- the #1 instrument failure `tools.lab.lever` exists to catch), and
      the loader's own provenance dict must read `applied=True` (not a silent fail-safe fallback) on every
      learned-arm call.
  (4) RNG discipline: host process-global numpy RNG state byte-identical before/after the WHOLE A/B (the #77
      footgun `_RngIsolation` exists to prevent; measured here, not assumed).

NOT re-derived here (already GO'd elsewhere, cited not repeated): the few-spike Izhikevich read's own
anti-cheats (scramble/equal-drive/noise-ablation/provenance) -- `_wkv_fewspike_read_derisk`'s own GO; the
wiring's flag-off byte-identical / fallback / lesion-safety properties -- `2026-08-28-wkv-mouth-into-open-
ended-WIRED-GO.md`'s own GO.

HONEST FRAMING (do not overclaim): the learned head recovers ~91% (this specific seed) / ~87% (6-seed mean) of
the NATIVE head's OWN recovery of the checkpoint's target -- so it is EXPECTED to generate slightly WORSE than
native, not better. GO here means "generates coherently as a legitimate opt-in", never "beats native" -- this
runner does not recommend, and this repo's own convention forbids, a default-on flip from this result alone.

CPU/numpy only (~512 neurons per read, read_window=40, K=64 pools x P=8 pop). Detached-run friendly: prints
progress per prompt so a `nohup ... &` caller can tail it.

Run:  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_learned_vs_native_head_ab
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tools.verdict import Verdict  # noqa: E402
from tools.lab import lever, void_if  # noqa: E402

LEARNED_HEAD_NPZ = _REPO / "research/findings/raw/_persist_eprop_head_scope/wkv_eprop_learned_head_6seed.npz"
SEED = 102          # the ONLY seed whose e-prop-learned head this npz actually holds (see module docstring)
MAX_NEW_TOKENS = 50
READ_WINDOW = 40
POP = 8
TOPK = 64
GEN_TEMP = 0.8

OUT = _REPO / "research" / "findings" / "raw" / "_wkv_learned_vs_native_head_ab.json"

PROMPTS = [
    "once upon a time there was a little boy named tim who had a dog",
    "tell me a story about a happy dog and his best friend",
    "lily and her mom went to the park to play with a ball",
    "the little girl saw a big red ball in the yard",
    "tom was very happy because he had a new toy car",
    "one day a cat and a dog became good friends and played together",
    "the little boy was so happy to see his mom and dad",
    "sam and his sister went outside to play in the sun",
]


def _sha1(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _ngram_distinct(words, n):
    if len(words) < n:
        return None
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def _max_repeat_run(words):
    if not words:
        return 0
    best = cur = 1
    for i in range(1, len(words)):
        cur = cur + 1 if words[i] == words[i - 1] else 1
        best = max(best, cur)
    return best


def _self_nll(ro, word_to_id, text: str) -> tuple[float | None, int]:
    """Teacher-forced self-NLL of `text` under `ro`'s OWN next-word distribution -- gate on the PREVIOUS
    (already-advanced) token to predict the NEXT, the SAME convention `_free_gen` /
    `_wkv_mouth_open_ended_wiring_verify` both use (gating on the token BEING predicted instead reads a
    different, meaningless quantity)."""
    words = [w for w in text.split() if w in word_to_id]
    if len(words) < 3:
        return None, len(words)
    ap = np.zeros(ro.D)
    an = np.zeros(ro.D)
    prev_id = word_to_id[words[0]]
    ap, an = ro.advance(ap, an, prev_id)
    nlls = []
    for w in words[1:]:
        lg = ro.logits(ap, an, prev_id)
        p = np.exp(lg - lg.max())
        p = p / p.sum()
        tid = word_to_id[w]
        nlls.append(-float(np.log(max(p[tid], 1e-12))))
        ap, an = ro.advance(ap, an, tid)
        prev_id = tid
    return sum(nlls) / len(nlls), len(words)


def _run_arm(W, prompt: str, learned: bool) -> dict:
    os.environ["BRAIN_WKV_MOUTH_LEARNED_HEAD"] = "1" if learned else "0"
    ro, vocab, word_to_id = W._get_readout(SEED)
    head_hash = _sha1(ro.head_w)
    status = W.learned_head_status(SEED) if learned else None
    text, secs = W.generate(prompt, seed=SEED, max_new_tokens=MAX_NEW_TOKENS, topk=TOPK,
                             read_window=READ_WINDOW, pop=POP, gen_temp=GEN_TEMP)
    cont = text[len(prompt):].strip() if text.startswith(prompt) else text
    cont_words = cont.split()
    self_nll, n_scored = _self_nll(ro, word_to_id, text)
    return {
        "prompt": prompt, "learned": learned, "text": text, "continuation": cont, "gen_seconds": secs,
        "head_hash": head_hash, "learned_head_status": status,
        "self_nll": self_nll, "n_words_scored": n_scored,
        "n_continuation_words": len(cont_words),
        "distinct_1": _ngram_distinct(cont_words, 1), "distinct_2": _ngram_distinct(cont_words, 2),
        "distinct_3": _ngram_distinct(cont_words, 3), "max_repeat_run": _max_repeat_run(cont_words),
    }


def main():
    os.environ["BRAIN_WKV_MOUTH_LEARNED_HEAD_PATH"] = str(LEARNED_HEAD_NPZ)   # read at IMPORT time -> set FIRST
    from webapp import wkv_mouth_generator as W  # noqa: E402  (deliberately late; see line above)

    npz = np.load(LEARNED_HEAD_NPZ, allow_pickle=True)
    npz_meta = {k: (npz[k].item() if npz[k].shape == () else npz[k].shape) for k in npz.files}
    print(f"[ab] learned-head artifact: {LEARNED_HEAD_NPZ.name}  meta={npz_meta}")
    void_if(int(npz_meta.get("seed", -1)) != SEED,
            f"npz seed={npz_meta.get('seed')} != runner SEED={SEED} -- checkpoint/head mismatch")

    host_rng_before = np.random.get_state()[1].copy()

    native_runs, learned_runs = [], []
    t0 = time.time()
    for i, p in enumerate(PROMPTS):
        r_native = _run_arm(W, p, learned=False)
        r_learned = _run_arm(W, p, learned=True)
        native_runs.append(r_native)
        learned_runs.append(r_learned)
        print(f"[ab] {i+1}/{len(PROMPTS)}  native_nll={r_native['self_nll']}  "
              f"learned_nll={r_learned['self_nll']}  learned_applied={(r_learned['learned_head_status'] or {}).get('applied')}"
              f"  elapsed={time.time()-t0:.1f}s")

    host_rng_after = np.random.get_state()[1].copy()
    rng_untouched = bool((host_rng_before == host_rng_after).all())

    native_hashes = {r["head_hash"] for r in native_runs}
    learned_hashes = {r["head_hash"] for r in learned_runs}
    lever("head_w hash native vs learned", next(iter(native_hashes)), next(iter(learned_hashes)))
    heads_differ = (native_hashes != learned_hashes) and len(native_hashes) == 1 and len(learned_hashes) == 1
    all_applied = all((r["learned_head_status"] or {}).get("applied") is True for r in learned_runs)

    native_nlls = [r["self_nll"] for r in native_runs if r["self_nll"] is not None]
    learned_nlls = [r["self_nll"] for r in learned_runs if r["self_nll"] is not None]
    chance_nll = float(np.log(1000))

    def _mean(xs):
        return (sum(xs) / len(xs)) if xs else None

    native_nll_mean = _mean(native_nlls)
    learned_nll_mean = _mean(learned_nlls)
    native_d2_mean = _mean([r["distinct_2"] for r in native_runs if r["distinct_2"] is not None])
    learned_d2_mean = _mean([r["distinct_2"] for r in learned_runs if r["distinct_2"] is not None])
    native_maxrun_mean = _mean([r["max_repeat_run"] for r in native_runs])
    learned_maxrun_mean = _mean([r["max_repeat_run"] for r in learned_runs])

    art = {
        "probe": "wkv_learned_vs_native_head_ab", "backend": "numpy", "seed": SEED,
        "npz_path": str(LEARNED_HEAD_NPZ.relative_to(_REPO)), "npz_meta": npz_meta,
        "n_prompts": len(PROMPTS), "max_new_tokens": MAX_NEW_TOKENS, "read_window": READ_WINDOW,
        "native_runs": native_runs, "learned_runs": learned_runs,
        "native_self_nll_mean": native_nll_mean, "learned_self_nll_mean": learned_nll_mean,
        "chance_nll": chance_nll,
        "native_distinct2_mean": native_d2_mean, "learned_distinct2_mean": learned_d2_mean,
        "native_max_repeat_run_mean": native_maxrun_mean, "learned_max_repeat_run_mean": learned_maxrun_mean,
        "heads_differ": heads_differ, "all_learned_applied": all_applied,
        "rng_untouched_across_ab": rng_untouched,
        "n_native_scored": len(native_nlls), "n_learned_scored": len(learned_nlls),
        "elapsed_s": round(time.time() - t0, 1),
    }

    v = Verdict("the e-prop LOCALLY-LEARNED WKV-mouth read-out head generates COHERENTLY through the "
                "production entry point, as a legitimate default-OFF opt-in (not a default-on claim)")
    v.require("(lever) native and learned head_w actually differ (the swap executed on every call)",
              heads_differ, expect=True)
    v.require("(fail-safe) the learned head loader reports applied=True on every learned-arm call "
              "(no silent fallback to native)", all_applied, expect=True)
    v.require("(RNG) host process-global numpy RNG state is byte-identical before/after the whole A/B",
              rng_untouched, expect=True)
    if learned_nll_mean is not None:
        v.control("learned-head self-NLL vs chance (uniform over V=1000)",
                  treatment=chance_nll - learned_nll_mean, control=0.0, min_separation=2.0,
                  note=f"learned self_nll={learned_nll_mean:.3f} nats vs chance={chance_nll:.3f} nats")
    else:
        v.require("learned-head self-NLL was measurable", False, expect=True,
                  note="fewer than 3 in-vocab words scored across all prompts")

    go = (heads_differ and all_applied and rng_untouched
         and learned_nll_mean is not None and (chance_nll - learned_nll_mean) > 2.0)
    decided = v.decide(go=go)
    art["verdict"] = decided
    art["GO"] = bool(go)
    art["preconditions"] = decided.get("preconditions", [])   # gates/verdict_preconditions reads this TOP-LEVEL

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=1))
    print(json.dumps({
        "native_self_nll_mean": native_nll_mean, "learned_self_nll_mean": learned_nll_mean,
        "chance_nll": chance_nll, "native_distinct2_mean": native_d2_mean,
        "learned_distinct2_mean": learned_d2_mean, "heads_differ": heads_differ,
        "all_learned_applied": all_applied, "rng_untouched_across_ab": rng_untouched, "GO": go,
    }, indent=1))
    print(f"wrote {OUT} -> {decided['status']}")
    return decided["status"]


if __name__ == "__main__":
    main()
