"""TRAINING-TIME ONLY local teacher (Qwen2.5-0.5B-Instruct) for
sequence-level / data distillation (Kim & Rush 2016). Generates clean
English text the student trains on. NEVER imported by the
self-contained runtime path (test-enforced). Offline after the
one-time cached fetch."""
from __future__ import annotations
import os
LOCAL_FILES_ONLY = True
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def assert_training_time_only() -> None:
    """Loud marker; the runtime path must never call this."""
    return None

def generate_corpus(n_passages: int = 200, max_new_tokens: int = 160,
                     seed: int = 42) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert_training_time_only()
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(_MODEL, local_files_only=LOCAL_FILES_ONLY)
    # decoder-only batched generation needs left padding
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, local_files_only=LOCAL_FILES_ONLY,
        dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    prompts = [
        "Write a short, simple paragraph of plain English prose.",
        "Tell a brief everyday story in simple sentences.",
        "Describe a common object in a few clear sentences.",
        "Explain a simple idea in plain language.",
    ]
    batch_size = 16  # GPU-batched: ~order-of-magnitude faster than 1-by-1
    out = []
    for start in range(0, n_passages, batch_size):
        chunk = [prompts[i % len(prompts)]
                 for i in range(start, min(start + batch_size, n_passages))]
        texts = [
            tok.apply_chat_template([{"role": "user", "content": c}],
                                     tokenize=False,
                                     add_generation_prompt=True)
            for c in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        n_in = enc["input_ids"].shape[1]
        gen = model.generate(**enc, do_sample=True, temperature=0.8,
                              top_p=0.95, max_new_tokens=max_new_tokens,
                              pad_token_id=tok.eos_token_id)
        for row in gen:
            out.append(tok.decode(row[n_in:],
                                   skip_special_tokens=True).strip())
        print(f"  teacher gen {len(out)}/{n_passages}", flush=True)
    return "\n\n".join(out)
