import json
from research.runners.conversation_log import make_record

def test_record_shape():
    r = make_record(turn=3, user="what is apple", intent="assoc",
                     retrieved=[("big",779.0),("spoon",410.0)],
                     abstained=False, response="Apple is associated with big.")
    assert r["turn"] == 3 and r["intent"] == "assoc"
    assert r["concept_sequence"] == ["apple", "big"]
    assert r["abstained"] is False
    json.dumps(r)

def test_record_abstained_sequence_is_query_only():
    r = make_record(turn=1, user="what is zzz", intent="unknown",
                     retrieved=[], abstained=True, response="I don't know about zzz yet.")
    assert r["concept_sequence"] == ["zzz"]
