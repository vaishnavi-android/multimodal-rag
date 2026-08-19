import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.cleaner import clean_text


def test_collapses_excess_blank_lines():
    text = "Line one.\n\n\n\n\nLine two."
    cleaned = clean_text(text)
    assert "\n\n\n" not in cleaned


def test_collapses_repeated_spaces():
    text = "Too    many     spaces."
    cleaned = clean_text(text)
    assert "  " not in cleaned


def test_preserves_numbers_and_units():
    text = "The minimum age is 18 years, per Section 4.2."
    cleaned = clean_text(text)
    assert "18 years" in cleaned
    assert "Section 4.2" in cleaned


def test_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
