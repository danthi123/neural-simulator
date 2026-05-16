from research.runners.concept_grammar import render

def test_assoc(): assert render("assoc", {"SUBJ":"apple","OBJ":"big"}) == "Apple is associated with big."
def test_attr_pos(): assert render("attr", {"SUBJ":"apple","ATTR":"red"}) == "Apple is red."
def test_attr_neg(): assert render("attr", {"SUBJ":"apple","ATTR":"cold","POLARITY":"neg"}) == "Apple is not cold."
def test_yesno_yes(): assert render("yesno_yes", {"SUBJ":"apple","ATTR":"big"}) == "Yes, apple is big."
def test_yesno_no(): assert render("yesno_no", {"SUBJ":"apple","ATTR":"big"}) == "No, I haven't learned that apple is big."
def test_list_conj(): assert render("list", {"SUBJ":"apple","OBJ":["big","red"]}) == "Apple is associated with big and red."
def test_unknown(): assert render("unknown", {"SUBJ":"zzz"}) == "I don't know about zzz yet."
def test_missing_slot_falls_back(): assert render("attr", {"SUBJ":"apple"}) == "I don't know about apple yet."
