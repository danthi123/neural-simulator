"""P1b STEP A: Qwen2.5-0.5B-Instruct ANN (non-spiking) baseline.

Cheap de-risk before the spiking convert (STEP B). Confirms the model
downloads + loads on GPU + is fluent, and establishes the ANN baseline:
  (a) held-out perplexity on the TinyStories held-out tail
  (b) ~5 generation samples (greedy + sampled) saved verbatim
  (c) the model config (LLaMA-stack ops/shapes to convert in STEP B)

Writes research/findings/raw/_grounded_lang_p1b_ann_baseline.json.
Foreground/blocking by design. GPU (RTX 3090).
"""
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "research" / "findings" / "raw" / "_grounded_lang_p1b_ann_baseline.json"
CORPUS = REPO / "data" / "corpus" / "tinystories.txt"


def log(msg):
    print(f"[p1b-ann] {msg}", flush=True)


def main():
    result = {
        "model_id": MODEL_ID,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    t_start = time.time()

    # ---- 1. DOWNLOAD + LOAD ----
    log(f"loading tokenizer for {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    log("loading model (fp16) onto GPU ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).cuda()
    model.eval()
    load_s = time.time() - t0
    log(f"model loaded in {load_s:.1f}s; device={next(model.parameters()).device}")

    n_params = sum(p.numel() for p in model.parameters())
    result["n_params"] = int(n_params)
    result["load_seconds"] = round(load_s, 2)
    result["param_device"] = str(next(model.parameters()).device)
    result["param_dtype"] = str(next(model.parameters()).dtype)
    log(f"n_params = {n_params/1e6:.1f}M")

    # ---- 1c. CONFIG (the ops/shapes to convert in STEP B) ----
    cfg = model.config
    cfg_dict = cfg.to_dict()
    # Pull the LLaMA-stack-relevant fields explicitly for the controller.
    convert_spec = {
        "architectures": cfg_dict.get("architectures"),
        "model_type": cfg_dict.get("model_type"),
        "num_hidden_layers": cfg_dict.get("num_hidden_layers"),
        "hidden_size": cfg_dict.get("hidden_size"),
        "intermediate_size": cfg_dict.get("intermediate_size"),
        "num_attention_heads": cfg_dict.get("num_attention_heads"),
        "num_key_value_heads": cfg_dict.get("num_key_value_heads"),
        "head_dim": cfg_dict.get("head_dim",
                                 (cfg_dict.get("hidden_size") or 0) // max(cfg_dict.get("num_attention_heads") or 1, 1)),
        "vocab_size": cfg_dict.get("vocab_size"),
        "max_position_embeddings": cfg_dict.get("max_position_embeddings"),
        "rms_norm_eps": cfg_dict.get("rms_norm_eps"),
        "rope_theta": cfg_dict.get("rope_theta"),
        "hidden_act": cfg_dict.get("hidden_act"),
        "tie_word_embeddings": cfg_dict.get("tie_word_embeddings"),
        "attention_bias": cfg_dict.get("attention_bias"),
        "torch_dtype": str(cfg_dict.get("torch_dtype")),
        "sliding_window": cfg_dict.get("sliding_window"),
        "use_sliding_window": cfg_dict.get("use_sliding_window"),
    }
    # rope_theta moved under cfg.rope_parameters in newer transformers; recover it.
    if convert_spec["rope_theta"] is None:
        rp = cfg_dict.get("rope_parameters") or {}
        convert_spec["rope_theta"] = rp.get("rope_theta", getattr(cfg, "rope_theta", None))
    if convert_spec["torch_dtype"] in (None, "None"):
        convert_spec["torch_dtype"] = str(cfg_dict.get("dtype"))
    if convert_spec["attention_bias"] is None:
        # Qwen2 uses bias on QKV projections but not o_proj (config field absent => True for q/k/v).
        convert_spec["attention_bias"] = "qkv_only (Qwen2: bias on q/k/v_proj, none on o_proj)"
    result["config_full"] = cfg_dict
    result["convert_spec"] = convert_spec
    log(f"config: {convert_spec['num_hidden_layers']}L d={convert_spec['hidden_size']} "
        f"heads={convert_spec['num_attention_heads']}/{convert_spec['num_key_value_heads']}kv "
        f"vocab={convert_spec['vocab_size']} act={convert_spec['hidden_act']} "
        f"norm_eps={convert_spec['rms_norm_eps']} rope_theta={convert_spec['rope_theta']}")
    print("[p1b-ann] FULL model.config:", flush=True)
    print(cfg, flush=True)

    # ---- 2a. HELD-OUT PERPLEXITY ----
    # Use the TinyStories held-out tail (last ~120k chars), measured as
    # token-level cross-entropy under a sliding window. Reproducible.
    log("computing held-out perplexity on TinyStories tail ...")
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()
    held_out = text[-120_000:]
    # Start at a clean story boundary so we don't begin mid-word.
    delim = "<|endoftext|>"
    idx = held_out.find(delim)
    if idx != -1:
        held_out = held_out[idx + len(delim):].lstrip()
    result["ppl_corpus"] = "tinystories_tail"
    result["ppl_corpus_chars"] = len(held_out)

    enc = tok(held_out, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    n_tokens_total = input_ids.shape[1]
    result["ppl_n_tokens"] = int(n_tokens_total)
    log(f"held-out: {len(held_out)} chars -> {n_tokens_total} tokens")

    max_len = min(getattr(cfg, "max_position_embeddings", 2048) or 2048, 2048)
    stride = max_len  # non-overlapping windows (simple, reproducible)
    nll_sum = 0.0
    n_tok_scored = 0
    t0 = time.time()
    with torch.no_grad():
        for begin in range(0, n_tokens_total, stride):
            end = min(begin + max_len, n_tokens_total)
            ids = input_ids[:, begin:end]
            if ids.shape[1] < 2:
                break
            out = model(ids, labels=ids)
            # out.loss is mean NLL over (ids.shape[1]-1) shifted tokens
            n_scored = ids.shape[1] - 1
            nll_sum += float(out.loss) * n_scored
            n_tok_scored += n_scored
    ppl_s = time.time() - t0
    mean_nll = nll_sum / max(n_tok_scored, 1)
    ppl = math.exp(mean_nll)
    result["held_out_mean_nll"] = round(mean_nll, 4)
    result["held_out_perplexity"] = round(ppl, 4)
    result["ppl_n_tokens_scored"] = int(n_tok_scored)
    result["ppl_seconds"] = round(ppl_s, 2)
    log(f"held-out mean NLL = {mean_nll:.4f}  -> PERPLEXITY = {ppl:.4f}  "
        f"({n_tok_scored} tokens scored in {ppl_s:.1f}s)")

    # ---- 2b. GENERATION SAMPLES (verbatim) ----
    log("generating samples ...")
    samples = []

    def chat_generate(user_msg, do_sample, temperature=0.8, max_new_tokens=120, seed=42):
        msgs = [{"role": "user", "content": user_msg}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        torch.manual_seed(seed)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**ids, **gen_kwargs)
        new_tokens = out[0, ids.input_ids.shape[1]:]
        gen_text = tok.decode(new_tokens, skip_special_tokens=True)
        return prompt, gen_text

    # Prompts mixing story-completion + a simple Q, greedy + sampled.
    spec = [
        ("Once upon a time", False, 0.0),
        ("Once upon a time", True, 0.8),
        ("The weather today", False, 0.0),
        ("What do dogs eat?", False, 0.0),
        ("What do dogs eat?", True, 0.8),
    ]
    for i, (msg, do_sample, temp) in enumerate(spec):
        t0 = time.time()
        prompt, gen = chat_generate(msg, do_sample=do_sample, temperature=temp, seed=42 + i)
        dt = time.time() - t0
        mode = f"sampled(temp={temp},top_p=0.9)" if do_sample else "greedy"
        samples.append({
            "idx": i,
            "user_prompt": msg,
            "mode": mode,
            "generated_text": gen,
            "gen_seconds": round(dt, 2),
        })
        log(f"sample {i} [{mode}] '{msg}' ({dt:.1f}s):")
        print("    " + gen.replace("\n", "\n    "), flush=True)

    result["generation_samples"] = samples
    result["total_seconds"] = round(time.time() - t_start, 2)

    # Reset terminal autodetect of fluency is the controller's job; we store verbatim.
    result["verdict"] = "BASELINE-DATA-WRITTEN"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log(f"wrote {OUT}")
    log(f"DONE in {result['total_seconds']:.1f}s  ppl={ppl:.3f}")


if __name__ == "__main__":
    main()
