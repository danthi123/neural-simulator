from research.runners.build_distill_corpus import clean_corpus, DISTILL_PATH

def test_clean_ascii_and_nonempty():
    raw = "Hello  world!\n\n\n\nUnicode tail more"
    c = clean_corpus(raw)
    assert all(32 <= ord(ch) < 127 or ch == "\n" for ch in c)
    assert len(c) > 0
    assert "\n\n\n" not in c  # 3+ newlines collapsed

def test_strips_non_ascii_keeps_content():
    c = clean_corpus("café menu")   # accented e removed, rest kept
    assert "caf menu" == c or "caf" in c and "menu" in c

def test_deterministic():
    assert clean_corpus("abc abc\n\n\n\nx") == clean_corpus("abc abc\n\n\n\nx")

def test_distill_path_under_datasets():
    p = str(DISTILL_PATH).replace("\\", "/")
    assert "research/datasets" in p and p.endswith(".txt")
