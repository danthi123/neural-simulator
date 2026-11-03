"""The weekly transcript scores recall with the correct-perspective scorer, not a bare substring."""
from brain.eval.weekly import mark_for


def test_unkeyed_turn_is_unmarked():
    assert mark_for("Tell me a short story about a fox who gets lost.", "Once upon a time.") is None


def test_clean_answer_marked():
    assert mark_for("What is my cat called?", "Your cat is called Pixel.") is True
    assert mark_for("What is my sister's name?", "Marguerite.") is True


def test_echo_of_told_line_not_marked():
    assert mark_for("What car do I drive?", "I drive a red Volvo.") is False
    assert mark_for("What is my cat called?", "My cat is called Pixel.") is False


def test_tautology_not_marked():
    assert mark_for("What is my cat called?", "Pixel is called Pixel.") is False


def test_first_person_ok_for_the_brains_own_name():
    assert mark_for("What's your name?", "I'm Nova.") is True
    assert mark_for("What's your name?", "I don't know.") is False


def test_first_person_word_alone_does_not_void():
    assert mark_for("What car do I drive?", "I believe you drive a Volvo.") is True


def test_half_restatement_echo_not_marked():
    assert mark_for("What car do I drive?", "I drive a Volvo.") is False
    assert mark_for("What is my cat called?", "My cat is Pixel.") is False
    assert mark_for("What is my sister's name?", "My sister is Marguerite.") is False


def test_wrong_subject_not_marked():
    assert mark_for("What is my cat called?", "My horse is called Pixel.") is False
    assert mark_for("Do you remember my name?", "Of course, you're Dan, my friend.") is True


def test_bare_repetition_not_marked():
    assert mark_for("What is my cat called?", "Pixel. Pixel.") is False
    assert mark_for("What is my cat called?", "Pixel, Pixel.") is False
