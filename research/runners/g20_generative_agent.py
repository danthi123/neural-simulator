"""Stage-1 grounded generative conversational agent over the
validated 320 G.20 ensemble. Intent-parse -> abstention-gated
retrieval (reuse _query_top: its rate IS the gate's confidence) ->
productive concept-grammar -> dialogue-state coref. Appends a
concept-sequence JSONL (Stage-2 fuel). No retrain, no architecture
change, no external LLM."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from research.runners.g20_multibridge import SharedPoolMember, read_vocab_file
from research.runners.g20_xbridge_benchmark import _query_top
from research.runners.concept_grammar import render
from research.runners.abstention_gate import gate, abstain, DEFAULT_THRESHOLD
from research.runners.dialogue_state import DialogueState
from research.runners.conversation_log import make_record

def parse(line, state):
    t = line.strip().lower().rstrip("?.")
    if not t or t in ("quit","exit"): return ("quit", None, None)
    def resolve(tok): return state.resolve(tok) or tok
    if t.startswith("remember ") and " is " in t:
        a,b = t[len("remember "):].split(" is ",1); return ("remember",resolve(a.strip()),resolve(b.strip()))
    if t.startswith("what is "): return ("ask", resolve(t[len("what is "):].strip()), None)
    if t.startswith("what about "):
        return ("ask", state.last_subject() or resolve(t[len("what about "):].strip()), None)
    if t.startswith("is ") and " " in t[3:]:
        parts=t[3:].split(); return ("yesno", resolve(parts[0]), resolve(parts[-1]))
    return ("ask", resolve(t.split()[-1]) if t.split() else t, None)

def respond(intent_tuple, members, state, threshold):
    kind, a, b = intent_tuple
    if kind == "remember":
        ma = next((m for m in members if a in m.vocab_set), None)
        mb = next((m for m in members if b in m.vocab_set), None)
        if ma and mb:
            tag=f"{a}_{b}"; ma.encode_partial(a,tag); mb.encode_partial(b,tag)
            for m in (ma,mb):
                if tag not in m.encoded_tags: m.encoded_tags.append(tag)
            state.push(a,"SUBJ")
            return render("attr",{"SUBJ":a,"ATTR":b}), [(b, 9999.0)], False
        return render("unknown",{"SUBJ":a if not ma else b}), [], True
    ranked = _query_top(members, a)
    top = gate(ranked, threshold)
    state.push(a, "SUBJ")
    if top is None:
        return render("unknown",{"SUBJ":a}), [], True
    if kind == "yesno":
        ok = any(c==b and not abstain(r) for c,r,_ in ranked)
        return (render("yesno_yes" if ok else "yesno_no",{"SUBJ":a,"ATTR":b}),
                [(top[0],top[1])], False)
    return render("assoc",{"SUBJ":a,"OBJ":top[0]}), [(top[0],top[1])], False

def _recover(members, steps):
    """Inter-turn neural recovery. Repeated stimulate_tag of the same
    engram across turns induces adaptation / STP depression in the
    shared pool; _query_top only settles ~20 internal steps, too few
    to recover cross-turn (controlled apple x4 probe: 677 -> 601 ->
    sub-noise). Free-run every loaded bridge with ZERO external drive
    for `steps` so adaptation/STP recover before the next turn. This
    is the documented bridge behaviour (CLAUDE.md: state "self-recovers
    in ~10ms free running"); dt=0.5ms so 200 steps ~= 100ms >> 10ms.
    Reuses the project's settle idiom (cp_external_input_current[:]=0
    + _run_one_simulation_step loop) verbatim; backend-agnostic."""
    if steps <= 0:
        return
    for m in members:
        m.bridge.cp_external_input_current[:] = 0.0
        for _ in range(steps):
            m.bridge._run_one_simulation_step()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bridges",nargs="+",required=True)
    p.add_argument("--vocab-files",nargs="+",required=True)
    p.add_argument("--names",nargs="+",required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--n-shared-pool",type=int,default=2000)
    p.add_argument("--sparsity",type=float,default=0.007)
    p.add_argument("--pattern-size",type=int,default=100)
    p.add_argument("--sparse",action="store_true")
    p.add_argument("--threshold",type=float,default=DEFAULT_THRESHOLD)
    p.add_argument("--scripted",type=str,default=None)
    p.add_argument("--recover-steps",type=int,default=200,
        help="inter-turn zero-drive free-run steps so adaptation/STP "
             "recover before the next turn (dt=0.5ms; 200~=100ms). "
             "0 disables (reproduces the pre-fix decay).")
    p.add_argument("--log",type=str,default=None)
    a=p.parse_args()
    members=[]
    for bp,vp,nm in zip(a.bridges,a.vocab_files,a.names):
        m=SharedPoolMember(bridge_path=bp,vocab=read_vocab_file(vp),name=nm,
            n_shared_pool=a.n_shared_pool,sparsity=a.sparsity,
            sparse=a.sparse,pattern_size=a.pattern_size)
        m.load(a.seed); members.append(m)
    state=DialogueState(); logf=open(a.log,"w") if a.log else None
    inputs = ([s.strip() for s in a.scripted.split(",")] if a.scripted
              else iter(lambda: input("> "), None))
    turn=0
    for line in inputs:
        turn+=1; it=parse(line,state)
        if it[0]=="quit": break
        resp,retr,abst=respond(it,members,state,a.threshold)
        print(f"> {line}\n  {resp}",flush=True)
        if logf: logf.write(json.dumps(make_record(turn,line,it[0],retr,abst,resp))+"\n")
        _recover(members, a.recover_steps)
    if logf: logf.close()
    print("Done.",flush=True)

if __name__=="__main__": sys.exit(main())
