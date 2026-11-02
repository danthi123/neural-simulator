import os, sys, time
os.environ["RWKV_V7_ON"] = "1"; os.environ["RWKV_JIT_ON"] = "1"; os.environ["RWKV_CUDA_ON"] = "0"
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
path = sys.argv[1]
t = time.time(); m = RWKV(model=path[:-4], strategy="cpu fp32"); print("load s", round(time.time()-t, 1))
pipe = PIPELINE(m, "rwkv_vocab_v20230424")
args = PIPELINE_ARGS(temperature=1.0, top_p=0.3, alpha_presence=0.5, alpha_frequency=0.5, token_stop=[0])
convo = ["Hi! What's your name?", "I'm Dan. I work as a system administrator. What do you like to do?",
         "If I told you my cat is called Pixel, what would you say about her?", "What's my name, and what's my cat called?"]
ctx = ""
for u in convo:
    ctx += f"User: {u}\n\nAssistant:"
    t = time.time(); out = pipe.generate(ctx, token_count=80, args=args)
    out = out.split("\n\nUser:")[0].split("\n\n")[0].strip()
    print(f"\nUSER: {u}\nBRAIN ({round(time.time()-t,1)}s): {out}")
    ctx += " " + out + "\n\n"
