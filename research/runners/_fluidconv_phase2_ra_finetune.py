"""Phase-2 build: the RETRIEVAL-AUGMENTED render/QA FINE-TUNE of the ~21M generator (the roadmap's "brain-train it"
lever -- the direct enabler of focused conversational Q&A, i.e. "talk to it like an LLM").

Phase-1 v3 GO showed prompt-conditioned free-gen + post-hoc VERIFY = fluid grounded RENDERING, but the base TinyStories
model CONTINUES STORIES, it does not ANSWER questions (it rambles on-topic). This fine-tune teaches the 21M the
retrieval-augmented QA FORMAT: given the brain's retrieved fact(s) in the prompt + a question, produce a FOCUSED
fluent grounded answer -- and ABSTAIN ("i do not know") when the relevant fact is not provided (the moat, learned).

KEY DESIGN (so it GENERALIZES the FORMAT, not memorizes facts):
  - BROAD synthetic vocab (dozens of subjects/verbs/objects/adjectives, simple TinyStories-register English) -> the
    fact in any example is RANDOM; the only learnable regularity is "use the provided facts to answer" + "abstain if
    absent". The model cannot memorize a fixed fact table; it must learn the retrieval-augmented behaviour.
  - The fact is ALWAYS in the "facts :" context (retrieval-augmented); the answer USES it. Abstain examples give facts
    about OTHER subjects and ask about an absent one -> "i do not know" (the learned moat).
  - ANSWER PHRASING VARIETY (several templates per query type) so the model learns fluent answering, not rote copy.
  - INTERLEAVED with raw TinyStories text (McClelland-1995 complementary-learning-systems / self-replay
    anti-forgetting -- the SAME principle the C2 grow-without-forget result validated) so the fine-tune KEEPS the
    base fluency instead of catastrophically forgetting it.
  - SAME BPE (V=2049) as the base 21M -- no re-fit; the embedding matrix is unchanged; a fresh low-LR AdamW over the
    fine-tune steps, initialized from the 21M WEIGHTS (a continue-train, not from scratch).

This is a MINIMIZED + brain-trained + brain-gated generator (NOT the Qwen fallback). NO sim/ edit; a local GPU run.

Run: python -m research.runners._fluidconv_phase2_ra_finetune --steps 2500 --n-qa 16000
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BASE_CKPT = "research/findings/raw/fluidconv/gen_tinystories_20M.ckpt.pt"
BPE = "research/findings/raw/fluidconv/gen_tinystories.bpe.json"
FT_CKPT = "research/findings/raw/fluidconv/gen_tinystories_ra_ft.ckpt.pt"
CORPUS = "data/corpus/ra_finetune_corpus.txt"
TINYSTORIES = "data/corpus/tinystories_train.txt"
SEP = " * "  # in-vocab (id 57) inter-example separator = the learned end-of-answer stop signal
ARCH = dict(vocab_size=2049, d_model=512, n_layer=6, n_head=8, block_size=512)

# --- broad, simple, TinyStories-register vocab (RANDOM facts -> the model learns the FORMAT, cannot memorize) ---
SUBJECTS = ("dog cat bird cow fox bee hen fish bear mole owl frog duck pig goat wolf deer mouse rat lamb "
            "tom lily sam ben max sara mia leo ana kai zoe jack rose finn nora sky girl boy man cub").split()
# (base, 3rd-person-sg, past) transitive verbs
VERBS = [("eat", "eats", "ate"), ("chase", "chases", "chased"), ("like", "likes", "liked"),
         ("give", "gives", "gave"), ("make", "makes", "made"), ("find", "finds", "found"),
         ("want", "wants", "wanted"), ("see", "sees", "saw"), ("hold", "holds", "held"),
         ("ride", "rides", "rode"), ("hug", "hugs", "hugged"), ("kick", "kicks", "kicked"),
         ("throw", "throws", "threw"), ("catch", "catches", "caught"), ("carry", "carries", "carried"),
         ("pick", "picks", "picked"), ("wash", "washes", "washed"), ("paint", "paints", "painted")]
OBJECTS = ("meat fish seed grass bone milk worm hay honey egg web bread cake ball key ring hat box cup toy "
           "book bell drum flag leaf rock nut plum apple sock shoe cap coat kite pot dish spoon "
           # the curriculum's patient words (so a de-risk fact like fox/eat/rabbit or sun/give/light renders):
           "rabbit cat mouse light water shade tree ground cave").split()
ADJS = "big small red blue green yellow soft warm cold fast slow clever kind brave happy shy".split()


def _rng(seed):
    import random
    return random.Random(seed)


def _make_example(r):
    """Emit one 'facts : ... question : ... answer : ...' example (with phrasing variety), broad-vocab + random."""
    S = r.choice(SUBJECTS); vb, v3, _vp = r.choice(VERBS); O = r.choice(OBJECTS)
    fact = f"the {S} {v3} {O} ."
    qtype = r.choices(["what", "who", "yesno_y", "yesno_n", "describe", "abstain"],
                      weights=[3, 2, 2, 2, 2, 3])[0]
    # optionally add 1-2 distractor facts about OTHER subjects (teach retrieval from multiple facts)
    distractors = []
    for _ in range(r.choice([0, 0, 1, 2])):
        S2 = r.choice([s for s in SUBJECTS if s != S]); _b, v32, _p = r.choice(VERBS); O2 = r.choice(OBJECTS)
        distractors.append(f"the {S2} {v32} {O2} .")

    if qtype == "what":
        ctx = " ".join(([fact] + distractors) if not distractors else _shuffle(r, [fact] + distractors))
        q = f"what does the {S} {vb} ?"
        a = r.choice([f"the {S} {v3} {O} .", f"it {v3} {O} .", f"the {S} likes to {vb} {O} .",
                      f"the {S} {v3} {O} , yes ."])
    elif qtype == "who":
        ctx = " ".join(_shuffle(r, [fact] + distractors))
        q = f"who {v3} {O} ?"
        a = r.choice([f"the {S} does .", f"the {S} {v3} {O} .", f"it is the {S} ."])
    elif qtype == "yesno_y":
        ctx = " ".join(_shuffle(r, [fact] + distractors))
        q = f"does the {S} {vb} {O} ?"
        a = r.choice([f"yes , the {S} {v3} {O} .", f"yes . the {S} {v3} {O} .", "yes ."])
    elif qtype == "yesno_n":
        O2 = r.choice([o for o in OBJECTS if o != O])
        ctx = " ".join(_shuffle(r, [fact] + distractors))
        q = f"does the {S} {vb} {O2} ?"
        a = r.choice(["no .", f"no , the {S} does not {vb} {O2} .", f"no , the {S} {v3} {O} ."])
    elif qtype == "describe":
        ctx = " ".join(_shuffle(r, [fact] + distractors))
        q = r.choice([f"tell me about the {S} .", f"what about the {S} ?", f"say something about the {S} ."])
        a = r.choice([f"the {S} {v3} {O} .", f"the {S} likes to {vb} {O} ."])
    else:  # abstain -- the relevant subject's fact is NOT in ctx (the learned moat)
        # ctx = ONLY distractors (facts about OTHER subjects); if none, invent one
        if not distractors:
            S2 = r.choice([s for s in SUBJECTS if s != S]); _b, v32, _p = r.choice(VERBS); O2 = r.choice(OBJECTS)
            distractors = [f"the {S2} {v32} {O2} ."]
        ctx = " ".join(distractors)
        q = r.choice([f"what does the {S} {vb} ?", f"who {v3} {O} ?", f"tell me about the {S} ."])
        a = r.choice(["i do not know .", "i am not sure .", f"i do not know about the {S} .", "i can not say ."])
    return f"facts : {ctx} question : {q} answer : {a}"


def _shuffle(r, xs):
    ys = list(xs); r.shuffle(ys); return ys


def build_corpus(out_path, n_qa, tinystories_path, mix_chars_ratio, seed):
    """Write the mixed fine-tune corpus: QA examples (separated by SEP) INTERLEAVED with raw TinyStories chunks
    (anti-forgetting). mix_chars_ratio ~= fraction of the corpus that is raw TinyStories (by chars)."""
    r = _rng(seed)
    qa = [_make_example(r) for _ in range(n_qa)]
    qa_text = (SEP).join(qa)
    qa_chars = len(qa_text)
    # pull an equal-ish chunk of raw TinyStories (fluency retention)
    ts_target = int(qa_chars * mix_chars_ratio / max(1e-9, (1.0 - mix_chars_ratio)))
    ts_text = ""
    if os.path.exists(tinystories_path):
        with open(tinystories_path, "r", encoding="utf-8", errors="replace") as fh:
            ts_text = fh.read(ts_target)
    # interleave: split TinyStories into chunks and weave between QA blocks (every ~20 QA examples, a story chunk)
    r2 = _rng(seed + 1)
    ts_chunks = [ts_text[i:i + 1500] for i in range(0, len(ts_text), 1500)] if ts_text else []
    r2.shuffle(ts_chunks)
    blocks, ci = [], 0
    for i in range(0, len(qa), 20):
        blocks.append((SEP).join(qa[i:i + 20]))
        if ci < len(ts_chunks):
            blocks.append(" " + ts_chunks[ci].replace("\n", " ") + " ")
            ci += 1
    blocks.extend(" " + c.replace("\n", " ") + " " for c in ts_chunks[ci:])
    corpus = (SEP).join(blocks)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(corpus, encoding="utf-8")
    return {"n_qa": n_qa, "qa_chars": qa_chars, "ts_chars": len(ts_text), "total_chars": len(corpus),
            "ts_fraction": round(len(ts_text) / max(1, len(corpus)), 3)}


def _encode_corpus(tok, text):
    """Word-memoized BPE encode (the BPE is word-level; encode each UNIQUE whitespace token once). Fast on the
    repetitive synthetic corpus."""
    import numpy as np
    cache = {}
    ids = []
    for w in text.split(" "):
        e = cache.get(w)
        if e is None:
            e = tok.encode(w + " ")   # keep the trailing-space word-boundary the BPE expects
            cache[w] = e
        ids.extend(e)
    return np.array(ids, dtype=np.int64)


def finetune(steps, lr, batch_size, seed, n_qa, mix_ratio, warmup, print_every):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sim.tiny_transformer import TinyGPT
    from sim.bpe_tokenizer import BPETokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ra-ft] device={dev}; building corpus (n_qa={n_qa}, ts_mix={mix_ratio})...", flush=True)
    stats = build_corpus(CORPUS, n_qa, TINYSTORIES, mix_ratio, seed)
    print(f"[ra-ft] corpus: {stats}", flush=True)

    tok = BPETokenizer.load(BPE)
    t_enc = time.time()
    data = _encode_corpus(tok, Path(CORPUS).read_text(encoding="utf-8"))
    print(f"[ra-ft] encoded {len(data)} tokens in {time.time()-t_enc:.1f}s", flush=True)

    model = TinyGPT(**ARCH, dropout=0.1).to(dev)
    st = torch.load(BASE_CKPT, map_location=dev, weights_only=True)   # own trusted ckpt
    model.load_state_dict(st["model"])
    model.train(True)
    npar = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[ra-ft] loaded base 21M (~{npar:.1f}M) from {BASE_CKPT}; fine-tuning {steps} steps @ lr {lr}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_at(step):
        if warmup and step < warmup:
            return lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, steps - warmup)
        return 0.5 * lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    blk = ARCH["block_size"]
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    init_loss = None
    for step in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        ix = torch.randint(0, len(data) - blk - 1, (batch_size,), generator=g)
        xb = torch.stack([torch.from_numpy(data[i:i + blk]) for i in ix]).to(dev)
        yb = torch.stack([torch.from_numpy(data[i + 1:i + 1 + blk]) for i in ix]).to(dev)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if init_loss is None:
            init_loss = float(loss.item())
        if step % print_every == 0 or step == steps - 1:
            print(f"[ra-ft] step {step}/{steps} loss={loss.item():.4f} lr={lr_at(step):.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            tmp = FT_CKPT + ".tmp"
            torch.save({"model": model.state_dict(), "arch": ARCH, "step": step,
                        "init_loss": init_loss, "final_loss": float(loss.item())}, tmp)
            os.replace(tmp, FT_CKPT)
    print(f"[ra-ft] done ({time.time()-t0:.0f}s) init={init_loss:.4f} final={float(loss.item()):.4f} -> {FT_CKPT}",
          flush=True)
    return {"steps": steps, "init_loss": init_loss, "final_loss": float(loss.item()),
            "corpus_stats": stats, "npar_M": round(npar, 1), "ckpt": FT_CKPT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-qa", type=int, default=16000)
    ap.add_argument("--mix-ratio", type=float, default=0.5, help="fraction of corpus that is raw TinyStories (anti-forget)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--print-every", type=int, default=250)
    ap.add_argument("--out", default="research/findings/raw/_fluidconv_phase2_ra_finetune.json")
    a = ap.parse_args()
    if not os.path.exists(BASE_CKPT):
        print(f"NOT-RUNNABLE: base 21M ckpt absent ({BASE_CKPT})"); return 2
    t0 = time.time()
    res = finetune(a.steps, a.lr, a.batch_size, a.seed, a.n_qa, a.mix_ratio, a.warmup, a.print_every)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"[ra-ft] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
