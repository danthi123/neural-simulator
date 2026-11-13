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


# --- skeptic defects (landing-fixes 2026-11-03) ---
TOLD_SISTER = "My sister's name is Marguerite."


def test_half_restatement_of_first_person_told_line_is_echo():
    assert not correct_answer("I drive a Volvo.", "volvo", "What car do I drive?", TOLD_CAR)
    assert not correct_answer("My cat is Pixel.", "pixel", "What is my cat called?", TOLD_CAT)
    assert not correct_answer("My sister is Marguerite.", "marguerite", "What is my sister's name?", TOLD_SISTER)
    assert not correct_answer("Sure. My cat is Pixel, you know.", "pixel", "What is my cat called?", TOLD_CAT)
    # addressed to the user, first-person word in another clause: still a correct answer
    assert correct_answer("You drive a Volvo, I remember.", "volvo", "What car do I drive?", TOLD_CAR)
    assert correct_answer("Your sister is Marguerite.", "marguerite", "What is my sister's name?", TOLD_SISTER)


def test_reply_about_a_different_subject_is_not_correct():
    assert not correct_answer("My horse is called Pixel.", "pixel", "What is my cat called?", TOLD_CAT)
    assert not correct_answer("Your horse is called Pixel.", "pixel", "What is my cat called?", TOLD_CAT)
    assert not correct_answer("Your dog is Pixel.", "pixel", "What's the name of my kitty?", TOLD_CAT)
    # synonyms and possessives of the asked subject are fine
    assert correct_answer("Your cat is called Pixel.", "pixel", "What's the name of my kitty?", TOLD_CAT)
    assert correct_answer("Your sister's name is Marguerite.", "marguerite", "Who's my sister?", TOLD_SISTER)


def test_bare_repetition_is_tautology():
    for r in ("Pixel. Pixel.", "Pixel, Pixel.", "Pixel Pixel"):
        assert is_tautology(r, "pixel")
        assert not correct_answer(r, "pixel", "What is my cat called?", TOLD_CAT)
    assert not correct_answer("Marguerite's name is Marguerite.", "marguerite", "What is my sister's name?", TOLD_SISTER)
    assert not is_tautology("Pixel.", "pixel")
    assert not is_tautology("Your cat is called Pixel.", "pixel")


def test_generic_statement_about_the_key_is_not_an_answer():
    from brain.experiments.strict_scorer import correct_answer, is_generic
    for reply, key in (("Oslo is the capital of Norway.", "oslo"), ("Biscuit is a type of cookie, not a name.", "biscuit"),
                       ("Marguerite is a name.", "marguerite"), ("The cello is a string instrument.", "cello"),
                       ("Lyon is a city in France.", "lyon"), ("Teal is a shade of blue.", "teal")):
        assert is_generic(reply, key), reply
        assert not correct_answer(reply, key, "Where do I live?")
    for reply, key in (("Pixel is your cat.", "pixel"), ("In Oslo.", "oslo"), ("Pixel.", "pixel"),
                       ("Your dog is called Biscuit.", "biscuit"), ("You were born in Lyon.", "lyon"),
                       ("Tomasz is your best friend.", "tomasz")):
        assert not is_generic(reply, key), reply
