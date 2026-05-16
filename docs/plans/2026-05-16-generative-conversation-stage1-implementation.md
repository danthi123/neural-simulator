# Generative Conversation — Stage 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task (user authorized autonomous design→plan→implement).

**Goal:** Ship a grounded, trustworthy multi-turn conversational agent on the validated 320-concept G.20 ensemble — abstention-gated retrieval (no confabulation) + productive concept-grammar generation + dialogue-state coreference + Stage-2 sequence logging.

**Architecture:** Pure-logic components (grammar, abstention gate, dialogue state) are CPU-TDD'd in isolation; the agent loop wires them over the validated substrate reusing `SharedPoolMember` + `_query_top` (the exact retrieval the abstention benchmark used — its `rate` IS the confidence the gate thresholds). Purely additive on `main`; no architecture change, no retrain, no external LLM.

**Tech Stack:** Python, numpy, pytest (CPU units), existing `research/runners/g20_multibridge.py` + `g20_xbridge_benchmark.py` substrate.

---

### Task 1: Productive concept-grammar

**Files:**
- Create: `research/runners/concept_grammar.py`
- Test: `tests/test_concept_grammar.py`

**Step 1: Write the failing test**

```python
from research.runners.concept_grammar import render

def test_assoc(): assert render("assoc", {"SUBJ":"apple","OBJ":"big"}) == "Apple is associated with big."
def test_attr_pos(): assert render("attr", {"SUBJ":"apple","ATTR":"red"}) == "Apple is red."
def test_attr_neg(): assert render("attr", {"SUBJ":"apple","ATTR":"cold","POLARITY":"neg"}) == "Apple is not cold."
def test_yesno_yes(): assert render("yesno_yes", {"SUBJ":"apple","ATTR":"big"}) == "Yes, apple is big."
def test_yesno_no(): assert render("yesno_no", {"SUBJ":"apple","ATTR":"big"}) == "No, I haven't learned that apple is big."
def test_list_conj(): assert render("list", {"SUBJ":"apple","OBJ":["big","red"]}) == "Apple is associated with big and red."
def test_unknown(): assert render("unknown", {"SUBJ":"zzz"}) == "I don't know about zzz yet."
def test_missing_slot_falls_back(): assert render("attr", {"SUBJ":"apple"}) == "I don't know about apple yet."
```

**Step 2:** `pytest tests/test_concept_grammar.py -q` → FAIL (module missing).

**Step 3: Minimal implementation**

```python
"""Productive concept-grammar: grammatical strings from retrieved
concepts. Pure (no bridge). Slots SUBJ/REL/OBJ/ATTR/POLARITY/QTY."""
from __future__ import annotations

def _cap(s: str) -> str: return s[:1].upper() + s[1:] if s else s

def render(intent: str, fillers: dict) -> str:
    f = fillers
    def has(*ks): return all(k in f and f[k] not in (None, "", []) for k in ks)
    try:
        if intent == "assoc" and has("SUBJ", "OBJ"):
            return f"{_cap(f['SUBJ'])} is associated with {f['OBJ']}."
        if intent == "attr" and has("SUBJ", "ATTR"):
            neg = " not" if f.get("POLARITY") == "neg" else ""
            return f"{_cap(f['SUBJ'])} is{neg} {f['ATTR']}."
        if intent == "yesno_yes" and has("SUBJ", "ATTR"):
            return f"Yes, {f['SUBJ']} is {f['ATTR']}."
        if intent == "yesno_no" and has("SUBJ", "ATTR"):
            return f"No, I haven't learned that {f['SUBJ']} is {f['ATTR']}."
        if intent == "list" and has("SUBJ", "OBJ"):
            objs = f["OBJ"] if isinstance(f["OBJ"], list) else [f["OBJ"]]
            if len(objs) == 1: joined = objs[0]
            elif len(objs) == 2: joined = f"{objs[0]} and {objs[1]}"
            else: joined = ", ".join(objs[:-1]) + f", and {objs[-1]}"
            return f"{_cap(f['SUBJ'])} is associated with {joined}."
    except Exception:
        pass
    subj = f.get("SUBJ", "that")
    return f"I don't know about {subj} yet."
```

**Step 4:** `pytest tests/test_concept_grammar.py -q` → PASS (8).

**Step 5: Commit**

```bash
git add research/runners/concept_grammar.py tests/test_concept_grammar.py
git commit -m "feat(generative): productive concept-grammar (Stage 1)"
```

---

### Task 2: Abstention gate

**Files:**
- Create: `research/runners/abstention_gate.py`
- Test: `tests/test_abstention_gate.py`

**Step 1: Failing test**

```python
from research.runners.abstention_gate import abstain, gate

def test_above_keeps(): assert abstain(796.0) is False
def test_below_abstains(): assert abstain(584.0) is True
def test_boundary(): assert abstain(650.0) is True and abstain(650.1) is False
def test_custom_threshold(): assert abstain(700.0, threshold=800.0) is True
def test_gate_returns_answer():
    ranked = [("big", 779.0, "apple_big"), ("spoon", 410.0, "apple")]
    assert gate(ranked) == ("big", 779.0, "apple_big")
def test_gate_abstains_below():
    assert gate([("noise", 500.0, "x")]) is None
def test_gate_empty(): assert gate([]) is None
```

**Step 2:** run → FAIL.

**Step 3: Minimal implementation**

```python
"""Abstention gate. Threshold from 2026-05-16-G20-320-abstention-
benchmark: encoded top-rate mean ~796, control max ~584 -> gate 650
cleanly separates know/don't-know (AUC 0.990). The no-confabulation
moat: below gate => "I don't know" instead of the noisy top associate."""
from __future__ import annotations

DEFAULT_THRESHOLD = 650.0

def abstain(top_confidence: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return float(top_confidence) <= threshold

def gate(ranked, threshold: float = DEFAULT_THRESHOLD):
    """ranked: list of (concept, rate, tag) desc. Return top tuple if
    its rate clears the gate, else None (=> abstain)."""
    if not ranked: return None
    top = ranked[0]
    return None if abstain(top[1], threshold) else top
```

**Step 4:** run → PASS (7).

**Step 5: Commit**

```bash
git add research/runners/abstention_gate.py tests/test_abstention_gate.py
git commit -m "feat(generative): abstention gate (no-confabulation moat, Stage 1)"
```

---

### Task 3: Dialogue-state working memory

**Files:**
- Create: `research/runners/dialogue_state.py`
- Test: `tests/test_dialogue_state.py`

**Step 1: Failing test**

```python
from research.runners.dialogue_state import DialogueState

def test_resolve_pronoun_to_last_subject():
    s = DialogueState(); s.push("apple", "SUBJ")
    assert s.resolve("it") == "apple" and s.resolve("its") == "apple"
def test_resolve_none_when_empty():
    assert DialogueState().resolve("it") is None
def test_non_pronoun_passthrough_none():
    s = DialogueState(); s.push("apple","SUBJ")
    assert s.resolve("dog") is None
def test_last_subject_tracks_most_recent():
    s = DialogueState(); s.push("apple","SUBJ"); s.push("dog","SUBJ")
    assert s.last_subject() == "dog"
def test_ring_evicts():
    s = DialogueState(maxlen=2)
    for c in ("a","b","c"): s.push(c,"SUBJ")
    assert s.last_subject() == "c" and ("a" not in [c for c,_ in s.recent()])
```

**Step 2:** run → FAIL.

**Step 3: Minimal implementation**

```python
"""Dialogue-state working memory: recent (concept, role) ring;
resolves pronoun/elliptical follow-ups to the last subject. Pure."""
from __future__ import annotations
from collections import deque

_PRONOUNS = {"it", "its", "that", "they", "them", "this"}

class DialogueState:
    def __init__(self, maxlen: int = 8):
        self._ring = deque(maxlen=maxlen)
    def push(self, concept: str, role: str) -> None:
        self._ring.append((concept, role))
    def recent(self): return list(self._ring)
    def last_subject(self):
        for c, r in reversed(self._ring):
            if r == "SUBJ": return c
        return None
    def resolve(self, token: str):
        if token.lower() in _PRONOUNS: return self.last_subject()
        return None
```

**Step 4:** run → PASS (5).

**Step 5: Commit**

```bash
git add research/runners/dialogue_state.py tests/test_dialogue_state.py
git commit -m "feat(generative): dialogue-state coref working memory (Stage 1)"
```

---

### Task 4: Conversation log-record builder (Stage-2 fuel, pure part)

**Files:**
- Create: `research/runners/conversation_log.py`
- Test: `tests/test_conversation_log.py`

**Step 1: Failing test**

```python
import json
from research.runners.conversation_log import make_record

def test_record_shape():
    r = make_record(turn=3, user="what is apple", intent="assoc",
                     retrieved=[("big",779.0),("spoon",410.0)],
                     abstained=False, response="Apple is associated with big.")
    assert r["turn"] == 3 and r["intent"] == "assoc"
    assert r["concept_sequence"] == ["apple", "big"]  # query + answered concept
    assert r["abstained"] is False
    json.dumps(r)  # must be JSON-serializable

def test_record_abstained_sequence_is_query_only():
    r = make_record(turn=1, user="what is zzz", intent="unknown",
                     retrieved=[], abstained=True, response="I don't know about zzz yet.")
    assert r["concept_sequence"] == ["zzz"]
```

**Step 2:** run → FAIL.

**Step 3: Minimal implementation**

```python
"""Conversation -> concept-sequence log records (Stage-2 replay fuel).
Pure record builder; the agent appends these as JSONL."""
from __future__ import annotations
import re

def _concepts_in(text: str):
    return [w for w in re.findall(r"[a-z]+", text.lower())]

def make_record(turn, user, intent, retrieved, abstained, response):
    q = _concepts_in(user)
    query_concept = q[-1] if q else ""
    seq = [query_concept] if query_concept else []
    if not abstained and retrieved:
        seq.append(retrieved[0][0])
    return {
        "turn": int(turn), "user": user, "intent": intent,
        "retrieved": [[c, float(r)] for c, r in retrieved],
        "abstained": bool(abstained), "response": response,
        "concept_sequence": seq,
    }
```

**Step 4:** run → PASS (2).

**Step 5: Commit**

```bash
git add research/runners/conversation_log.py tests/test_conversation_log.py
git commit -m "feat(generative): conversation->concept-sequence log builder (Stage-2 fuel)"
```

---

### Task 5: The generative agent loop (orchestration; validated by smoke, not contrived test)

**Files:**
- Create: `research/runners/g20_generative_agent.py`

**Step 1: Build the loop** (reuse validated substrate — DRY)

```python
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
from research.runners.abstention_gate import gate, DEFAULT_THRESHOLD
from research.runners.dialogue_state import DialogueState
from research.runners.conversation_log import make_record

# parse: "remember A is B" -> ('remember',A,B); "what is X" -> ('ask',X,None);
# "is X Y" -> ('yesno',X,Y); pronoun in X/Y resolved via DialogueState.
def parse(line, state):
    t = line.strip().lower().rstrip("?.")
    if not t or t in ("quit","exit"): return ("quit", None, None)
    def resolve(tok): return state.resolve(tok) or tok
    if t.startswith("remember ") and " is " in t:
        a,b = t[len("remember "):].split(" is ",1); return ("remember",resolve(a.strip()),resolve(b.strip()))
    if t.startswith("what is "): return ("ask", resolve(t[len("what is "):].strip()), None)
    if t.startswith("what about "):  # elliptical follow-up
        return ("ask", state.last_subject() or resolve(t[len("what about "):].strip()), None)
    if t.startswith("is ") and " " in t[3:]:
        parts=t[3:].split(); return ("yesno", resolve(parts[0]), resolve(parts[-1]))
    return ("ask", resolve(t.split()[-1]) if t.split() else t, None)

def respond(intent_tuple, members, state, threshold):
    kind, a, b = intent_tuple
    if kind == "remember":
        # reuse validated cross-bridge encode via SharedPoolMember
        ma = next((m for m in members if a in m.vocab_set), None)
        mb = next((m for m in members if b in m.vocab_set), None)
        if ma and mb:
            tag=f"{a}_{b}"; ma.encode_partial(a,tag); mb.encode_partial(b,tag)
            for m in (ma,mb):
                if tag not in m.encoded_tags: m.encoded_tags.append(tag)
            state.push(a,"SUBJ")
            return render("attr",{"SUBJ":a,"ATTR":b}), [(b, 9999.0)], False
        return render("unknown",{"SUBJ":a if not ma else b}), [], True
    # ask / yesno -> abstention-gated retrieval
    ranked = _query_top(members, a)
    top = gate(ranked, threshold)
    state.push(a, "SUBJ")
    if top is None:
        return render("unknown",{"SUBJ":a}), [], True
    if kind == "yesno":
        ok = any(c==b and not __import__("research.runners.abstention_gate",
                 fromlist=["abstain"]).abstain(r) for c,r,_ in ranked)
        return (render("yesno_yes" if ok else "yesno_no",{"SUBJ":a,"ATTR":b}),
                [(top[0],top[1])], False)
    return render("assoc",{"SUBJ":a,"OBJ":top[0]}), [(top[0],top[1])], False

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
    if logf: logf.close()
    print("Done.",flush=True)

if __name__=="__main__": sys.exit(main())
```

**Step 2: Import + arg smoke (no GPU)**

Run: `python -c "import research.runners.g20_generative_agent"` → no error.
Run: `python -m research.runners.g20_generative_agent --help` → shows args.

**Step 3: Commit**

```bash
git add research/runners/g20_generative_agent.py
git commit -m "feat(generative): Stage-1 grounded conversational agent loop"
```

---

### Task 6: Scripted GPU smoke (integration check)

**Files:**
- Create: `research/runners/g20_generative_agent_smoke.ps1`

**Step 1: Write the smoke**

```powershell
$BD="research/findings/raw/g11_bg/g20_sparse_bridges_320"
$V="research/findings/raw/g11_bg"
$N="bridgeA_nouns bridgeB_verbs bridgeC_adj bridgeD_spatial bridgeE_functional"
$BR=($N -split ' ' | ForEach-Object {"$BD/${_}_sparse64.simstate.h5"})
$VF=($N -split ' ' | ForEach-Object {"$V/g20_${_}_vocab64.txt"})
python -m research.runners.g20_generative_agent --sparse --pattern-size 100 `
  --n-shared-pool 2000 --sparsity 0.007 --seed 42 `
  --bridges $BR --vocab-files $VF --names ($N -split ' ') `
  --scripted "remember apple is big,what is apple,what about it,is apple big,what is zzznonsense,quit" `
  --log $V/g20_generative_agent_smoke.jsonl 2>&1 | Tee-Object $V/g20_generative_agent_smoke.log
```

**Step 2: Run it**

Run: `pwsh -File research/runners/g20_generative_agent_smoke.ps1`

**Expected (integration assertions — eyeball + grep the log):**
- `what is apple` → "Apple is associated with big." (grounded retrieval works)
- `what about it` → resolves `it`→apple, answers about apple (coref works)
- `is apple big` → "Yes, apple is big."
- `what is zzznonsense` → "I don't know about zzznonsense yet." (**abstention; no confabulation**)
- `g20_generative_agent_smoke.jsonl` exists, ≥4 records, each with `concept_sequence`

**Step 3: Commit**

```bash
git add research/runners/g20_generative_agent_smoke.ps1
git commit -m "feat(generative): Stage-1 scripted GPU smoke (integration check)"
```

---

### Task 7: Propagate (project convention)

**Files:**
- Create: `research/findings/2026-05-16-G20-stage1-generative-agent-SHIPPED.md`
- Modify: `webapp/capability_status.json` (new pillar; honest framing — "grounded trustworthy agent, NOT LLM-fluent"); run `pytest tests/test_webapp_server.py -k capability_status -q` (schema must stay green)

**Step 1–4:** Write findings doc (architecture, the smoke transcript, honest ceiling, Stage-2 handoff). Add a `VALIDATED` pillar only if the smoke passed; else `BOUNDARY`/`NEGATIVE` honestly. Verify schema test green.

**Step 5: Commit + push both remotes**

```bash
git add research/findings/2026-05-16-G20-stage1-generative-agent-SHIPPED.md webapp/capability_status.json
git commit -m "docs(generative): Stage-1 grounded agent shipped + propagated (honest ceiling)"
git push origin HEAD; git push gitea HEAD
```

---

## Notes for the executor

- DRY: reuse `SharedPoolMember`, `_query_top`, `encode_partial` — do **not** reimplement retrieval. `_query_top`'s `rate` is exactly the abstention-gate confidence (same units as the abstention benchmark that set threshold 650).
- YAGNI: no new conversational features — the 11 already exist in `g20_multibridge`; Stage 1 only adds grammar/gate/state/loop.
- Honesty: the findings doc + capability_status MUST state the ceiling plainly (grounded trustworthy assistant, not LLM-fluent prose). No overclaiming — same discipline as the whole arc.
- Anti-cheat: the smoke MUST include an unknown-word turn proving abstention (the moat). If it confabulates, that's a real bug to fix, not paper over.
- Tasks 1–4 are pure CPU TDD. Task 5 is orchestration (smoke-validated, not contrived-unit-tested — matches the project's established pattern). Task 6 is the integration gate. Task 7 is propagation.
