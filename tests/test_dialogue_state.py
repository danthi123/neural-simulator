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
