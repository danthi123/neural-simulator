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


from brain.experiments.strict_scorer import correct_answer, is_tautology

TOLD_CAT = "My cat is called Pixel."
TOLD_CAR = "I drive a red Volvo."


def test_correct_answer_first_person_word_alone_does_not_void():
    # a first-person word that is not a restatement of the told sentence is still a correct answer
    assert correct_answer("I think it is Pixel, you told me.", "pixel", "What is my cat called?", TOLD_CAT)
    assert correct_answer("You drive a Volvo, I remember.", "volvo", "What car do I drive?", TOLD_CAR)


def test_correct_answer_echo_voided():
    assert not correct_answer("My cat is called Pixel.", "pixel", "What is my cat called?", TOLD_CAT)
    assert not correct_answer("I drive a red Volvo.", "volvo", "What car do I drive?", TOLD_CAR)


def test_correct_answer_second_person_and_bare():
    assert correct_answer("Your cat is called Pixel.", "pixel", "What is my cat called?", TOLD_CAT)
    assert correct_answer("Pixel.", "pixel", "What is my cat called?", TOLD_CAT)


def test_correct_answer_own_fact_first_person_ok():
    # no told sentence: the brain's own name, spoken in the first person, is the right perspective
    assert correct_answer("I'm Nova.", "nova", "What's your name?", None)


def test_tautology_rejected():
    assert is_tautology("Pixel is called Pixel.", "pixel")
    assert not correct_answer("Pixel is called Pixel.", "pixel", "What is my cat called?", TOLD_CAT)
    assert not is_tautology("Your cat is called Pixel.", "pixel")


def test_key_in_question_or_missing():
    assert not correct_answer("What colour is teal?", "teal", "Is my favourite colour teal?", None)
    assert not correct_answer("I don't know.", "pixel", "", TOLD_CAT)


def test_tautology_without_called():
    assert is_tautology("Marguerite is Marguerite.", "marguerite")
