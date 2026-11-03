from brain.experiments.strict_scorer import score


def test_correct_second_person():
    assert score("Your cat is called Pixel.", "pixel", "What is my cat called?")["hit"]
    assert score("Pixel.", "pixel")["hit"]
    assert score("You are allergic to peanuts.", "peanut")["hit"]


def test_first_person_echo_rejected():
    r = score("My cat is called Pixel.", "pixel", "What is my cat called?")
    assert not r["hit"] and r["echo"] and r["lenient"]
    assert score("I was born in Lyon.", "lyon")["echo"]
    assert score("I think your cat is Pixel", "pixel")["echo"]   # conservative: any first-person word


def test_missing_or_embedded_key():
    assert not score("I don't know.", "pixel")["lenient"]
    assert not score("Your dog is Biscuit.", "pixel")["hit"]
    assert not score("A remarch of events", "march")["hit"]      # inside another word
    assert score("A remarch of events", "march")["lenient"]      # the old substring scorer would have counted it


def test_key_in_question_does_not_count():
    assert not score("What colour is teal?", "teal", "Is my favourite colour teal?")["hit"]
