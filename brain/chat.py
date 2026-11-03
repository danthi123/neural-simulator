"""Talk to the brain.  python -m brain.chat [--home DIR] [--name Nova] [--friend Dan] [--device cpu|cuda] [--show-memory]

Commands: /quit (sleeps if anything new was lived, then saves), /memory (list stored episodes with strength),
/forget-off (lesion: hippocampus off), /consolidated-off (lesion: speak without what sleep consolidated), /sleep.
"""
from __future__ import annotations

import argparse

from brain.brain import Brain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="brain_data/default")
    ap.add_argument("--name", default="Nova")
    ap.add_argument("--friend", default="Dan")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show-memory", action="store_true", help="print which memory was reinstated each turn")
    ap.add_argument("--no-sleep", action="store_true", help="do not sleep on restart / shutdown")
    a = ap.parse_args()
    b = Brain(a.home, name=a.name, friend=a.friend, device=a.device, sleep_on_restart=not a.no_sleep,
              sleep_log=print)
    print(f"[{b.meta['name']} is awake; {len(b.hippocampus.episodes)} memories; {b.meta.get('sleeps', 0)} nights slept;"
          f" /quit to sleep, save and exit]")
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
    b.close()
    print(f"[slept and saved; night {b.meta.get('sleeps', 0)}]")


if __name__ == "__main__":
    main()
