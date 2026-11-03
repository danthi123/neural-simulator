"""Talk to the brain.  python -m brain.chat [--home DIR] [--name Nova] [--friend Dan] [--device cpu|cuda]
[--model 0.1b|0.4b|1.5b|PATH] [--night restart|separate|queue] [--show-memory]

Commands: /quit (sleeps if anything new was lived, then saves), /memory (list stored episodes with strength, and the
memory accounting), /forget-off (lesion: hippocampus off), /consolidated-off (lesion: speak without what sleep
consolidated), /sleep.

--night restart (default): restart = night inside this process (start and /quit sleep when something is pending).
--night separate: chat never sleeps; the night is its own process (python -m brain.night, adopt-1.5b plan step 7) and
  this chat speaks with the synapses of the last night meanwhile (the hippocampus still holds today's facts).
--night queue: as separate, and on /quit a night is queued on the GPU (tools/brain_gpu.sh night) if anything is pending.
The home is locked while chat runs, so a night cannot sleep the same brain at the same time.
"""
from __future__ import annotations

import argparse
import os
import subprocess

from brain.brain import Brain

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def queue_night(home: str, model: str | None):
    cmd = ["bash", os.path.join(REPO, "tools", "brain_gpu.sh"), "night", "--home", os.path.abspath(home)]
    if model:
        cmd += ["--model", model]
    return subprocess.run(cmd, check=False).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="brain_data/default")
    ap.add_argument("--name", default="Nova")
    ap.add_argument("--friend", default="Dan")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=None, help="cortex checkpoint path or size tag (default: the 0.1B cortex)")
    ap.add_argument("--night", choices=("restart", "separate", "queue"), default="restart")
    ap.add_argument("--show-memory", action="store_true", help="print which memory was reinstated each turn")
    ap.add_argument("--no-sleep", action="store_true", help="do not sleep on restart / shutdown")
    a = ap.parse_args()
    inline = a.night == "restart" and not a.no_sleep
    b = Brain(a.home, name=a.name, friend=a.friend, device=a.device, cortex_model=a.model, sleep_on_restart=inline,
              sleep_log=print, lock=True)
    print(f"[{b.meta['name']} is awake ({b.cortex_model}); {len(b.hippocampus.episodes)} memories;"
          f" {b.meta.get('sleeps', 0)} nights slept; {len(b.pending())} lived since; /quit to save and exit]")
    try:
        while True:
            text = input("you> ").strip()
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/memory":
                for e in b.hippocampus.episodes:
                    print(f"  {b.hippocampus.effective_strength(e):5.2f}  uses={e.uses}  {e.text}")
                print(f"  {b.memory_report()}")
                continue
            if text == "/forget-off":
                b.hippocampus.enabled = not b.hippocampus.enabled
                print(f"  hippocampus {'ON' if b.hippocampus.enabled else 'OFF (lesion)'}")
                continue
            if text == "/consolidated-off":
                b.cortex.consolidated_on = not b.cortex.consolidated_on
                print(f"  consolidated synapses {'ON' if b.cortex.consolidated_on else 'OFF (lesion)'}")
                continue
            if text == "/sleep":
                b.sleep()
                b.save()
                print(f"  [slept; night {b.meta['sleeps']}]")
                continue
            reply, info = b.hear(text)
            print(f"{b.meta['name']}> {reply}")
            if a.show_memory and info["recalled"]:
                print(f"  [recalled: {info['recalled']}  sim={info['similarity']}]")
            b.save()
    except (EOFError, KeyboardInterrupt):
        pass
    pending = len(b.pending())
    b.close()
    if inline:
        print(f"[slept and saved; night {b.meta.get('sleeps', 0)}]")
    else:
        print(f"[saved; {pending} lived since the last night]")
        if a.night == "queue" and pending:
            queue_night(a.home, a.model)


if __name__ == "__main__":
    main()
