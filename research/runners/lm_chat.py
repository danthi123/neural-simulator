"""lm_chat — talk to a trained WKV language cortex checkpoint (the owner's "chat with the brain" interface).

Loads a run's frozen config + tokenizer + model + latest (or best) checkpoint weights, then either answers a one-shot
--prompt or runs an interactive prompt→generate loop. Reuses the lm_train_lib APIs (no retrain, read-only on the ckpt).

  # interactive:
  python -m research.runners.lm_chat --root bridges/lmtrain/run2 --device cuda
  # one-shot:
  python -m research.runners.lm_chat --root bridges/lmtrain/run3 --prompt "The best way to learn is" --gen-tokens 80
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
from research.runners.lm_train_lib import TrainConfig, _load_tokenizer, build_model, generate  # noqa: E402


def _load(root: str, device: str, which: str):
    rd = Path(root)
    cfg_p = rd / "config.json"
    if not cfg_p.exists():
        raise SystemExit(f"[lm_chat] no config.json at {rd} — is this a trained run dir? (run `lm_train_run start` first)")
    cfg = TrainConfig(**json.loads(cfg_p.read_text()))
    tok = _load_tokenizer(rd, cfg)
    V = getattr(tok, "vocab_size", cfg.vocab_size)
    model = build_model(cfg, V, device)
    ckp = rd / "ckpt" / (f"{which}.pt")
    if not ckp.exists():
        ckp = rd / "ckpt" / "latest.pt"
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[lm_chat] {root}  ckpt={ckp.name} step={ck.get('step','?')} tokens={ck.get('tokens_seen','?'):,} "
          f"params={n:.1f}M V={V} device={device}", flush=True)
    return cfg, tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a trained run dir, e.g. bridges/lmtrain/run3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--which", default="best", choices=["best", "latest"], help="checkpoint to load (falls back to latest)")
    ap.add_argument("--gen-tokens", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default=None, help="one-shot prompt; omit for an interactive loop")
    a = ap.parse_args()
    cfg, tok, model = _load(a.root, a.device, a.which)
    cfg.gen_tokens = a.gen_tokens

    def reply(p, seed):
        with torch.no_grad():
            return generate(model, tok, p, a.gen_tokens, a.device, cfg, temp=a.temp, seed=seed)

    if a.prompt is not None:
        print(reply(a.prompt, a.seed))
        return
    print("[lm_chat] interactive — type a prompt (blank line to skip, 'quit'/Ctrl-D to exit). "
          f"temp={a.temp} gen_tokens={a.gen_tokens}", flush=True)
    i = 0
    while True:
        try:
            p = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if p in ("quit", "exit"):
            break
        if not p:
            continue
        print("brain>", reply(p, a.seed + i), flush=True)
        i += 1


if __name__ == "__main__":
    main()
