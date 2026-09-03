import os, sys, json
assert os.environ.get("SIM_BACKEND") == "numpy"
REPO_ROOT = "/home/dant123/Projects/sim"
sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-aed59f0b381164491")
from webapp import wkv_mouth_generator as wmg
os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"
os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = "ssm"
wmg._CKPT_TEMPLATE = REPO_ROOT + "/bridges/wkv_ckpt/wkv_ssm_bpe8k_d192_simplewiki_depth2_contiguous_seed{seed}.npz"
wmg._CKPT_CACHE.clear()
PROMPTS = ["The sun is", "A dog is an animal that", "Water is made of", "Yesterday I went to the",
           "The city of London is", "In the beginning of the story", "The most important thing about science is",
           "My favorite food is", "The weather today", "Once upon a time there was"]
GEN_KW = dict(max_new_tokens=60, topk=64, read_window=40, pop=8, gen_temp=0.8,
              repetition_penalty=1.3, no_repeat_ngram_size=3)
out = {}
for seed in (42, 43):
    rows = []
    for p in PROMPTS:
        text, secs = wmg.generate(p, seed=seed, **GEN_KW)
        rows.append({"prompt": p, "text": text})
        print(f"[SSM-BPE+repguard seed{seed}] {p!r}\n    -> {text}\n", flush=True)
    out[seed] = rows
with open("/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/ssm_repguard_results.json", "w") as f:
    json.dump(out, f, indent=2)
