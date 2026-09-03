import sys
sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-aed59f0b381164491")
from sim.bpe_tokenizer import BPETokenizer
bt = BPETokenizer.load("/home/dant123/Projects/sim/bridges/wkv_ckpt/wkv_bpe8k.json")
upper = sorted(set(c for c in "".join(bt.vocab) if c.isupper()))
print("uppercase chars anywhere in vocab symbols:", upper)
print("vocab[0]:", repr(bt.vocab[0]))
for w in ["The", "the", "A", "a", "I", "Water", "water", "London", "london", "My", "my"]:
    ids = bt.encode(w)
    syms = [bt.vocab[i] if 0 <= i < len(bt.vocab) else "<OOB>" for i in ids]
    print(f"{w!r:12} -> ids={ids} syms={syms} decode={bt.decode(ids)!r}")
