"""Text parsing, segmentation and post‑processing utilities.

The legacy scripts implemented these functions directly.  They are now
collected in a dedicated module so that back‑end adapters and other tools can
reuse them.
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List

import numpy as np

# ---------------------------------------------------------------------
# Configuration – values are identical to clone_voice.py (legacy XTTS script).
# tests/test_text_processor.py pins them so they cannot drift again.
# ---------------------------------------------------------------------
PAUSE_INSIDE_LINE = 0.25   # between chunks of one long line
PAUSE_PERIOD = 0.25
PAUSE_QUESTION = 0.45
PAUSE_EXCLAIM = 0.30
PAUSE_ELLIPSIS = 0.60
PAUSE_OTHER = 0.20
PAUSE_PARAGRAPH = 0.70     # blank line in script = dramatic pause
MIN_LINE_CHARS = 30        # shorter lines get merged into the next one
MAX_CHARS = {"hi": 130, "en": 200}   # XTTS warns above ~150 (hi) / ~250 (en)
DEFAULT_MAX_CHARS = 200
SUPPORTED_LANGUAGES = {"HI": "hi", "EN": "en"}


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def parse_script(path: str | Path, default_language: str = "en") -> List[dict]:
    """Parse a script file into a list of speech items.

    The format is identical to the legacy scripts: a line consisting only of a
    language marker such as ``[EN]`` or ``[HI]`` sets the language for
    subsequent lines until another marker appears.  A blank line marks a
    paragraph break on the line *before* it.
    """

    return parse_script_text(Path(path).read_text(encoding="utf-8-sig"), default_language)


def parse_script_text(content: str, default_language: str = "en") -> List[dict]:
    """Parse script *content* (same format as :func:`parse_script`)."""

    if not content.strip():
        raise ValueError("Script contains no text.")

    items: List[dict] = []
    lang = default_language
    for n, raw in enumerate(content.splitlines(), 1):
        s = raw.strip()
        marker = re.fullmatch(r"\[([A-Za-z]+)\]", s)
        if marker:
            tag = marker.group(1).upper()
            if tag not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported marker [{tag}] at line {n}. Use [HI] or [EN].")
            lang = SUPPORTED_LANGUAGES[tag]
            continue
        if not s:
            if items:
                items[-1]["para_break"] = True
            continue
        items.append({"lang": lang, "text": s, "para_break": False})

    # Merge very short lines into the next one (XTTS is unstable on 1‑2 word clips),
    # except when a blank line follows (that's an intentional standalone beat).
    merged: List[dict] = []
    i = 0
    while i < len(items):
        cur = items[i]
        if (
            len(cur["text"]) < MIN_LINE_CHARS
            and not cur["para_break"]
            and i + 1 < len(items)
            and items[i + 1]["lang"] == cur["lang"]
        ):
            nxt = items[i + 1]
            merged.append({
                "lang": cur["lang"],
                "text": cur["text"] + " " + nxt["text"],
                "para_break": nxt["para_break"],
            })
            i += 2
        else:
            merged.append(cur)
            i += 1
    if not merged:
        raise ValueError("No valid speech lines found.")
    return merged


def make_chunks(text: str, limit: int) -> List[str]:
    """Split ``text`` into chunks no longer than ``limit``.

    Sentences are split on punctuation.  If a sentence exceeds the limit it is
    broken into words.  The algorithm is intentionally deterministic and
    matches the legacy behaviour.
    """

    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s.strip()]
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if len(s) > limit:
            # long sentence – break on words
            if cur:
                chunks.append(cur)
                cur = ""
            part = ""
            for w in s.split():
                cand = (part + " " + w).strip()
                if len(cand) <= limit:
                    part = cand
                else:
                    if part:
                        chunks.append(part)
                    part = w
            if part:
                cur = part
            continue
        cand = (cur + " " + s).strip()
        if len(cand) <= limit:
            cur = cand
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def pause_after(item: dict) -> float:
    """Return a pause duration after the spoken item.

    The returned value is in seconds.
    """

    t = item["text"].rstrip()
    if t.endswith(("...", "…")):
        p = PAUSE_ELLIPSIS
    elif t.endswith("?"):
        p = PAUSE_QUESTION
    elif t.endswith("!"):
        p = PAUSE_EXCLAIM
    elif t.endswith((".", "।")):
        p = PAUSE_PERIOD
    else:
        p = PAUSE_OTHER
    if item["para_break"]:
        p = max(p, PAUSE_PARAGRAPH)
    return p


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(sr * seconds), dtype=np.float32)


@dataclass(frozen=True)
class Segment:
    """One model call: a chunk of a script line, in playback order."""

    index: int          # global order
    line: int           # 1-based index into the parsed items
    chunk: int          # 0-based chunk index within the line
    text: str
    lang: str
    pause_after: float  # seconds of silence to insert after this segment


def build_segments(items: List[dict], max_chars: dict | None = None) -> List[Segment]:
    """Split parsed items into ordered model-sized segments, exactly as clone_voice.py does.

    Every chunk except a line's last is followed by ``PAUSE_INSIDE_LINE``; the last
    chunk is followed by :func:`pause_after` for the line.
    """
    limits = MAX_CHARS if max_chars is None else max_chars
    segments: List[Segment] = []
    for n, item in enumerate(items, 1):
        limit = limits.get(item["lang"], DEFAULT_MAX_CHARS)
        chunks = make_chunks(item["text"], limit)
        if not chunks:
            raise ValueError(f"Line {n} produced no text to synthesize")
        for ci, chunk in enumerate(chunks):
            last = ci == len(chunks) - 1
            segments.append(
                Segment(
                    index=len(segments),
                    line=n,
                    chunk=ci,
                    text=chunk,
                    lang=item["lang"],
                    pause_after=pause_after(item) if last else PAUSE_INSIDE_LINE,
                )
            )
    return segments
