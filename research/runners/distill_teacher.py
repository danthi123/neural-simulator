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
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, local_files_only=LOCAL_FILES_ONLY,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    prompts = [
        "Write a short, simple paragraph of plain English prose.",
        "Tell a brief everyday story in simple sentences.",
        "Describe a common object in a few clear sentences.",
        "Explain a simple idea in plain language.",
    ]
    out = []
    for i in range(n_passages):
        p = prompts[i % len(prompts)]
        msgs = [{"role": "user", "content": p}]
        # tokenize=False then tokenize: robust across transformers
        # versions (apply_chat_template return type varies).
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        n_in = enc["input_ids"].shape[1]
        gen = model.generate(**enc, do_sample=True, temperature=0.8,
                              top_p=0.95, max_new_tokens=max_new_tokens,
                              pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0][n_in:], skip_special_tokens=True)
        out.append(txt.strip())
    return "\n\n".join(out)
