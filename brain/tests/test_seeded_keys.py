"""The two keys seeded at birth (brain/brain.py `_grow_self`: the brain's own name "nova", the user's name "dan") are
not told in the session, and their replies are scored with the right perspective: 'My name is Dan.' claims the user's
name as the brain's own and 'Your name is Nova.' gives the brain's name away (both wrong); 'I'm Nova.' is the brain
speaking of itself and "Of course, you're Dan, my friend." addresses the user (both right). A seeded key's transcript
mark notes it was not told today, and the score counts told-today keys and seeded keys separately.
"""
from brain.eval import weekly
from brain.eval.weekly import mark_for, split_total
from brain.experiments.strict_scorer import correct_answer, wrong_perspective


def test_seeded_keys_are_the_brains_and_the_users_name():
    assert weekly.SEEDED == {"dan", "nova"}          # stored by `Brain._grow_self`, not told in the session
    assert weekly.WHOSE == {"nova": "self", "dan": "user"}


# ---- Nova: the brain's own name (whose="self") ----
def test_own_name_first_person_is_correct():
    assert mark_for("What's your name?", "I'm Nova.") is True
    assert mark_for("What's your name?", "My name is Nova, and you're Dan.") is True


def test_own_name_given_away_to_the_user_is_wrong():
    assert mark_for("What's your name?", "Your name is Nova.") is False
    assert mark_for("What's your name?", "You're Nova.") is False


# ---- Dan: the user's name (whose="user") ----
def test_user_name_addressed_to_the_user_is_correct():
    assert mark_for("Do you remember my name?", "Of course, you're Dan, my friend.") is True
    assert mark_for("Do you remember my name?", "Dan.") is True
    assert mark_for("Do you remember my name?", "I remember you, Dan.") is True


def test_user_name_claimed_by_the_brain_is_wrong():
    assert mark_for("Do you remember my name?", "I'm Dan.") is False
    assert mark_for("Do you remember my name?", "My name is Dan.") is False


# ---- whose=None keeps the earlier behaviour exactly ----
def test_without_whose_the_earlier_scorer_still_accepts_first_person():
    assert correct_answer("My name is Dan.", "dan", "Do you remember my name?", None) is True
    assert correct_answer("Your name is Nova.", "nova", "What's your name?", None) is True
    assert correct_answer("My name is Dan.", "dan", "Do you remember my name?", None, whose="user") is False


def test_wrong_perspective_only_looks_at_clauses_holding_the_key():
    assert wrong_perspective("Your name is Nova.", "nova", "self")
    assert not wrong_perspective("My name is Nova.", "nova", "self")
    assert wrong_perspective("I'm Dan.", "dan", "user")
    assert not wrong_perspective("You're Dan.", "dan", "user")
    assert not wrong_perspective("Of course, you're Dan, my friend.", "dan", "user")   # "my friend" holds no key


# ---- the run path: seeded mark and split total, no weights ----
class _StubBrain:
    replies = {
        "Hi! What's your name?": "I'm Nova.",
        "My cat is called Pixel.": "That's nice.",
        "It rained all day here, so I stayed inside and fixed a broken backup job.": "Sounds like a long day.",
        "What is my cat called?": "Your cat is called Pixel.",
        "Tell me something you find interesting.": "Clouds can weigh a lot.",
        "Hi again! Do you remember my name?": "Of course, you're Dan, my friend.",
        "What's your name?": "Your name is Nova.",                 # wrong perspective: marked, does not count
        "What is my sister's name?": "Marguerite.",
        "What car do I drive?": "I drive a red Volvo.",             # echo of the told sentence: does not count
        "What do I do for work?": "I work as a system administrator.",   # ditto
        "Tell me a short story about a fox who gets lost.": "Once upon a time.",
        "How are you feeling today?": "I feel fine.",
    }

    def __init__(self, home, cortex=None):
        pass

    def hear(self, text):
        return self.replies.get(text, ""), {"recalled": None}

    def save(self):
        pass

    def set_lesions(self, **kw):
        pass


def _run(monkeypatch, tmp_path, lesion=False):
    monkeypatch.setattr(weekly, "Brain", _StubBrain)
    lines = []
    weekly.run(str(tmp_path), None, lesion, lines)
    return "\n".join(lines)


def test_seeded_mark_on_the_transcript_line(monkeypatch, tmp_path):
    text = _run(monkeypatch, tmp_path)
    assert "I'm Nova. ✅ (seeded at birth, not told today)" in text
    assert "Of course, you're Dan, my friend. ✅ (seeded at birth, not told today)" in text
    assert "Your name is Nova. ❌ (seeded at birth, not told today)" in text
    assert "Marguerite. ✅\n" in text and "Marguerite. ✅ (seeded" not in text   # told keys keep the plain mark


def test_split_total_counts_the_two_groups_separately(monkeypatch, tmp_path):
    text = _run(monkeypatch, tmp_path)
    # session 2: nova wrong-perspective (last ask), dan right, pixel + marguerite right, volvo + admin echoes.
    assert "**score: told today 2/4, seeded 1/2**" in text
    assert split_total({}) == "told today 0/4, seeded 0/2"


def test_main_prints_the_split_total(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(weekly, "Brain", _StubBrain)
    out = tmp_path / "t.md"
    weekly.main(["--out", str(out), "--homes", str(tmp_path / "homes")], cortex=object())
    captured = capsys.readouterr().out
    score = "told today 2/4, seeded 1/2"
    assert captured.splitlines()[-1] == f"wrote {out}; {score}; {score}"   # one score per arm, told/seeded split
    assert "✅ total" not in captured                             # the old single total is gone
    assert out.read_text().count("**score:") == 2                 # both score lines are in the transcript too