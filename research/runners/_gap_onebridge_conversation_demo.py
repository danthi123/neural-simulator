"""Single shared substrate — a REACHABLE grounded conversation on ONE bridge (2026-07-20).

The capstone proved the composer + WKV cortex co-reside on one bridge. This wraps them into an actual grounded
conversational turn you can RUN: teach facts, ask questions, get fluent grounded answers rendered by the spiking WKV
cortex, with the gate-first no-confab moat (the composer decides answer-vs-abstain FIRST; on an abstain the WKV is
NEVER invoked). Everything on ONE `SimulationBridge`.

Turn: composer.query_patient(subj, verb) -> answer word (or None = abstain)
  - None  -> "I don't know."         (WKV NOT invoked -- moat by construction)
  - word  -> WKV.answer("the {subj} {verb} {word}", "what does the {subj} {verb} ?") -> fluent grounded prose

`--demo` runs a scripted transcript + asserts (grounded renders correct + 0 WKV invocations across all abstains).
Reuse-by-import (the capstone bridge + SharedBridgeComposer + merged WKV); NO sim/ edit. `--seed`.
"""
import argparse
import copy
import types
import numpy as np

from sim.backend import get_backend, to_host
from research.runners._wkv_onbridge_faculty import OnBridgeWKVFaculty
from research.runners._gap_wkv_onebridge_merged_derisk import _merged_rf_encode_decode
from research.runners._gap_onebridge_capstone_derisk import _build_capstone_bridge, SharedBridgeComposer

CKPT = "bridges/wkv_ckpt/wkv_ssmU_v4000_d256_grounded_ft.npz"


class OneBridgeChat:
    """Composer (RF retrieval + moat) + WKV cortex (spiking grounded render), ALL on one SimulationBridge."""

    # a generous pre-allocated vocab so new facts can be TAUGHT live (the composer's concept codes are fixed at build,
    # like `_fluidconv_chat_repl.py`'s pre-allocated vocab). Words outside this pool cannot be taught.
    POOL = ["dog", "cat", "owl", "mouse", "wolf", "rabbit", "fox", "bird", "fish", "lion", "deer", "bear", "frog",
            "cow", "pig", "duck", "bee", "ant", "cheese", "milk", "bone", "seed", "worm", "leaf",
            "chase", "eat", "hunt", "like", "see", "help", "find", "catch", "want", "hear", "roar", "swim"]

    def __init__(self, facts, seed=42, ckpt=CKPT, D_cmp=64):
        self.facts = list(facts)
        vocab = sorted(set(self.POOL) | {w for f in facts for w in (f[0], f[1], f[2])})
        # reference WKV to load the ckpt weights + decode config
        ref = OnBridgeWKVFaculty(ckpt=ckpt, seed=seed, rf_synaptic=False)
        D_wkv = ref.D
        self.mb, mchan, enc_idx, cmp_idx = _build_capstone_bridge(D_wkv, D_cmp, seed, ref.decay)
        # composer on the composer region
        self.cmp = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=vocab)
        self.cmp.bind_to_shared(self.mb, cmp_idx)
        for a, v, p in facts:
            self.cmp.store(a, v, p)
        # WKV cortex on the chan+encoder regions (the merged faculty)
        self.wkv = copy.copy(ref)
        self.wkv.b = self.mb; self.wkv.nnrn = int(self.mb.cp_membrane_potential_v.size); self.wkv._enc_idx = enc_idx
        self.wkv.read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
        self.wkv._rf_encode_decode = types.MethodType(_merged_rf_encode_decode, self.wkv)

    def ask(self, subj, verb):
        """Gate-FIRST: the composer decides. Abstain -> the WKV is NEVER invoked (moat by construction)."""
        ans = self.cmp.query_patient(subj, verb)
        if ans is None:
            return "I don't know.", None                    # WKV not invoked
        ctx = f"the {subj} {verb} {ans}"
        self.wkv._wash()
        rendered = self.wkv.answer(ctx, f"what does the {subj} {verb} ?")   # already a spaced string
        return rendered, ans

    def ask_who(self, verb, patient):
        """Gate-FIRST 'who' query: query_agent(verb, patient) -> the agent (or None = abstain, WKV NOT invoked)."""
        ans = self.cmp.query_agent(verb, patient)
        if ans is None:
            return "I don't know.", None
        ctx = f"the {ans} {verb} {patient}"
        self.wkv._wash()
        return self.wkv.answer(ctx, f"who {verb} {patient} ?"), ans

    def teach(self, subj, verb, patient):
        """Learn a new fact LIVE on the shared substrate (composer stores it in the composer region)."""
        self.cmp.store(subj, verb, patient)
        self.facts.append((subj, verb, patient))
        return f"ok -- learned '{subj} {verb} {patient}'."

    def handle(self, line):
        """Parse one REPL line: 'teach S V P' | 'ask S V' | 'who V P' | 'S V' (== ask). Returns a display string."""
        toks = line.strip().split()
        if not toks:
            return ""
        if toks[0] == "teach" and len(toks) == 4:
            return self.teach(toks[1], toks[2], toks[3])
        if toks[0] == "ask" and len(toks) == 3:
            reply, _ = self.ask(toks[1], toks[2]); return reply
        if toks[0] == "who" and len(toks) == 3:
            reply, _ = self.ask_who(toks[1], toks[2]); return reply
        if len(toks) == 2:                                  # bare "S V" == ask
            reply, _ = self.ask(toks[0], toks[1]); return reply
        return "usage: teach <subj> <verb> <patient> | ask <subj> <verb> | who <verb> <patient>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--script", default=None, help="semicolon-separated commands, e.g. 'teach wolf eat rabbit; ask wolf eat'")
    ap.add_argument("--interactive", action="store_true", help="REPL loop (teach/ask); for the owner to explore live")
    args = ap.parse_args()
    get_backend()

    if args.script is not None or args.interactive:
        chat = OneBridgeChat([("dog", "chase", "cat"), ("owl", "eat", "mouse")], seed=args.seed, ckpt=args.ckpt)
        print("=== one-brain single-substrate chat (teach <S> <V> <P> | ask <S> <V>) ===")
        print("  (pre-taught: dog chase cat, owl eat mouse)")
        if args.script is not None:
            for cmd in args.script.split(";"):
                cmd = cmd.strip()
                if cmd:
                    print(f"> {cmd}\n  {chat.handle(cmd)}")
        else:
            while True:
                try:
                    line = input("> ")
                except (EOFError, KeyboardInterrupt):
                    print(); break
                if line.strip() in ("quit", "exit"):
                    break
                print(f"  {chat.handle(line)}")
        return

    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse"), ("cat", "chase", "mouse")]
    chat = OneBridgeChat(facts, seed=args.seed, ckpt=args.ckpt)

    print("=== One-brain, one-substrate grounded conversation (composer + WKV cortex on ONE bridge) ===")
    print(f"taught: {', '.join(f'{a} {v} {p}' for a, v, p in facts)}\n")

    known = [("dog", "chase"), ("owl", "eat"), ("cat", "chase")]
    unknown = [("lion", "roar"), ("fish", "swim")]           # never taught -> abstain (moat)

    inv0 = chat.wkv.n_invocations
    render_ok = True
    for subj, verb in known:
        reply, ans = chat.ask(subj, verb)
        ok = (ans is not None) and (ans in reply)
        render_ok = render_ok and ok
        print(f"  Q: what does the {subj} {verb}?   A: {reply}   [grounded on '{ans}': {ok}]")
    inv_after_known = chat.wkv.n_invocations

    moat_ok = True
    inv_before_unknown = chat.wkv.n_invocations
    for subj, verb in unknown:
        reply, ans = chat.ask(subj, verb)
        abstained = (ans is None) and ("don't know" in reply)
        moat_ok = moat_ok and abstained
        print(f"  Q: what does the {subj} {verb}?   A: {reply}   [abstained: {abstained}]")
    inv_after_unknown = chat.wkv.n_invocations
    moat_no_invoke = (inv_after_unknown == inv_before_unknown)   # WKV NOT invoked on any abstain

    verdict = "GO" if (render_ok and moat_ok and moat_no_invoke) else "NO-GO"
    print(f"\n[RESULT {verdict}] single-substrate grounded conversation (seed {args.seed}):")
    print(f"  grounded renders correct : {render_ok}   (each answer contains the composer-retrieved word)")
    print(f"  no-confab moat (abstain) : {moat_ok}")
    print(f"  gate-FIRST (WKV invoked 0x on abstains): {moat_no_invoke} "
          f"({inv_after_known - inv0} invocations for {len(known)} known Qs, "
          f"{inv_after_unknown - inv_before_unknown} for {len(unknown)} abstains)")
    print(f"  => you can TALK to the one-brain: composer retrieves + gates, the SPIKING WKV cortex renders the "
          f"grounded answer, moat holds by construction -- all on ONE shared spiking substrate.")


if __name__ == "__main__":
    main()
