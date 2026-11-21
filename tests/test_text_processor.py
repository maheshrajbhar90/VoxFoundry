"""Unit tests for :mod:`voxfoundry.text_processor`."""

import pathlib
import tempfile
import textwrap

import pytest

from voxfoundry.text_processor import (
    parse_script,
    make_chunks,
    pause_after,
)


def _create_script(content: str) -> pathlib.Path:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return pathlib.Path(tmp.name)


def test_parse_script_basic():
    script = textwrap.dedent(
        """
        [EN] Hello world.
        [HI] नमस्ते दुनिया.
        """
    )
    path = _create_script(script)
    items = parse_script(path, default_language="en")
    assert len(items) == 2
    assert items[0]["lang"] == "en"
    assert items[1]["lang"] == "hi"


def test_parse_script_paragraph_break():
    script = textwrap.dedent(
        """
        [EN] First line.
        
        Second line.
        """
    )
    path = _create_script(script)
    items = parse_script(path)
    assert items[0]["para_break"] is False
    assert items[1]["para_break"] is True


def test_make_chunks():
    text = "This is a short sentence. This is a very long sentence that should be split into smaller chunks because it exceeds the limit"
    chunks = make_chunks(text, limit=30)
    # All chunks should be <= 30 characters
    assert all(len(c) <= 30 for c in chunks)
    # The first chunk should contain "This is a short sentence."
    assert chunks[0] == "This is a short sentence."


def test_pause_after():
    item = {"text": "Hello!", "para_break": False}
    assert pause_after(item) == 0.3
    item = {"text": "Bye...", "para_break": True}
    assert pause_after(item) >= 0.6