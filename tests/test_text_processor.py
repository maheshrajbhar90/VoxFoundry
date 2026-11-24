"""Unit tests for :mod:`clonemyvoice.text_processor`."""

import pathlib
import tempfile
import textwrap

import pytest

import clonemyvoice.text_processor as tp
from clonemyvoice.text_processor import (
    build_segments,
    make_chunks,
    parse_script,
    parse_script_text,
    pause_after,
)

pytestmark = pytest.mark.unit


def _create_script(content: str) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt")
    tmp.write(content)
    tmp.close()
    return pathlib.Path(tmp.name)


# --- original tests (corrected: markers must be on their own line, as in the legacy format) ---
def test_parse_script_basic():
    script = textwrap.dedent(
        """
        [EN]
        Hello world, this is the English line of the script.
        [HI]
        नमस्ते दुनिया, यह स्क्रिप्ट की हिंदी पंक्ति है।
        """
    )
    items = parse_script(_create_script(script), default_language="en")
    assert len(items) == 2
    assert items[0]["lang"] == "en"
    assert items[1]["lang"] == "hi"


def test_parse_script_paragraph_break():
    # A blank line marks a paragraph break on the line BEFORE it (legacy semantics).
    script = textwrap.dedent(
        """
        [EN]
        First line is long enough not to be merged.

        Second line is also long enough not to be merged.
        """
    )
    items = parse_script(_create_script(script))
    assert items[0]["para_break"] is True
    assert items[1]["para_break"] is False


def test_make_chunks():
    text = "This is a short sentence. This is a very long sentence that should be split into smaller chunks because it exceeds the limit"
    chunks = make_chunks(text, limit=30)
    assert all(len(c) <= 30 for c in chunks)
    assert chunks[0] == "This is a short sentence."


def test_pause_after():
    item = {"text": "Hello!", "para_break": False}
    assert pause_after(item) == 0.3
    item = {"text": "Bye...", "para_break": True}
    assert pause_after(item) >= 0.6


# --- legacy compatibility: values must equal clone_voice.py ---
def test_legacy_values_pinned():
    assert tp.PAUSE_INSIDE_LINE == 0.25
    assert tp.PAUSE_PERIOD == 0.25
    assert tp.PAUSE_QUESTION == 0.45
    assert tp.PAUSE_EXCLAIM == 0.30
    assert tp.PAUSE_ELLIPSIS == 0.60
    assert tp.PAUSE_OTHER == 0.20
    assert tp.PAUSE_PARAGRAPH == 0.70
    assert tp.MIN_LINE_CHARS == 30
    assert tp.MAX_CHARS == {"hi": 130, "en": 200}


@pytest.mark.parametrize(
    "text,para,expected",
    [
        ("Done.", False, 0.25),
        ("Really?", False, 0.45),
        ("Wow!", False, 0.30),
        ("Wait...", False, 0.60),
        ("Wait…", False, 0.60),
        ("और फिर।", False, 0.25),
        ("no punctuation", False, 0.20),
        ("Done.", True, 0.70),
        ("Wait...", True, 0.70),
    ],
)
def test_pause_after_values(text, para, expected):
    assert pause_after({"text": text, "para_break": para}) == pytest.approx(expected)


def test_short_lines_merge_into_next_same_language():
    items = parse_script_text("[EN]\nThen November happened.\nIn just two weeks, investors bought back in.\n")
    assert len(items) == 1
    assert items[0]["text"] == "Then November happened. In just two weeks, investors bought back in."


def test_short_line_before_blank_line_is_kept_alone():
    items = parse_script_text("[EN]\nThen November happened.\n\nIn just two weeks, investors bought back in.\n")
    assert [i["text"] for i in items] == ["Then November happened.", "In just two weeks, investors bought back in."]
    assert items[0]["para_break"] is True


def test_short_line_not_merged_across_languages():
    items = parse_script_text("[EN]\nShort English.\n[HI]\nयह एक लंबी हिंदी पंक्ति है जो मर्ज नहीं होगी।\n")
    assert [i["lang"] for i in items] == ["en", "hi"]


def test_unknown_marker_rejected_with_line_number():
    with pytest.raises(ValueError, match=r"\[HOOK\] at line 1"):
        parse_script_text("[HOOK]\nSome text here.\n")


@pytest.mark.parametrize("content", ["", "   \n\n", "[EN]\n[HI]\n"])
def test_empty_scripts_rejected(content):
    with pytest.raises(ValueError):
        parse_script_text(content)


def test_punctuation_numbers_and_symbols_preserved():
    line = "Nifty is at 25,228 - up 12% since October; 'neutral', 48% confidence!"
    items = parse_script_text(line)
    assert items[0]["text"] == line


def test_hindi_danda_splits_sentences():
    text = "पहला वाक्य है। दूसरा वाक्य है। तीसरा वाक्य है।"
    assert make_chunks(text, limit=20) == ["पहला वाक्य है।", "दूसरा वाक्य है।", "तीसरा वाक्य है।"]


def test_make_chunks_never_exceeds_limit_and_keeps_all_words():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words) + "."
    chunks = make_chunks(text, limit=200)
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_build_segments_order_and_pauses():
    long_en = " ".join(["This sentence is about forty characters."] * 8)  # > 200 chars
    items = [
        {"lang": "en", "text": long_en, "para_break": True},
        {"lang": "hi", "text": "क्या यह काम करता है?", "para_break": False},
    ]
    segs = build_segments(items)
    assert [s.index for s in segs] == list(range(len(segs)))
    en = [s for s in segs if s.line == 1]
    assert len(en) > 1 and all(len(s.text) <= 200 for s in en)
    assert [s.chunk for s in en] == list(range(len(en)))
    assert all(s.pause_after == 0.25 for s in en[:-1])
    assert en[-1].pause_after == 0.70  # paragraph pause after the line's last chunk
    assert segs[-1].lang == "hi" and segs[-1].pause_after == 0.45
    assert " ".join(s.text for s in en) == long_en


def test_build_segments_custom_limit():
    items = [{"lang": "en", "text": "One two three four five six seven eight nine ten.", "para_break": False}]
    segs = build_segments(items, {"en": 20})
    assert all(len(s.text) <= 20 for s in segs) and len(segs) > 1
