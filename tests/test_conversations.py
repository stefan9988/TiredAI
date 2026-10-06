from tiredai.conversations import TITLE_CHARS, title_from


def test_title_is_the_first_message_on_one_line():
    assert title_from("  I need\n\nwinter   tires\t") == "I need winter tires"


def test_long_titles_are_cut():
    assert title_from("x" * TITLE_CHARS) == "x" * TITLE_CHARS
    assert title_from("x" * (TITLE_CHARS + 1)) == "x" * (TITLE_CHARS - 1) + "…"
    # No space is left before the ellipsis.
    assert title_from("x" * (TITLE_CHARS - 2) + " tires") == "x" * (TITLE_CHARS - 2) + "…"
