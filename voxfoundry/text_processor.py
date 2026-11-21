"""Text parsing, segmentation and post‑processing utilities.

The legacy scripts implemented these functions directly.  They are now
collected in a dedicated module so that back‑end adapters and other tools can
reuse them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Configuration – kept simple for now.  The values mirror the legacy
# implementation.
# ---------------------------------------------------------------------
PAUSE_ELLIPSIS = 0.5
PAUSE_QUESTION = 0.3
PAUSE_EXCLAIM = 0.3
PAUSE_PERIOD = 0.3
PAUSE_OTHER = 0.2
PAUSE_PARAGRAPH = 0.6
MIN_LINE_CHARS = 2
SUPPORTED_LANGUAGES = {"HI": "hi", "EN": "en"}


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def parse_script(path: str | Path, default_language: str = "en") -> List[dict]:
    """Parse a script file into a list of speech items.

    The format is identical to the legacy scripts: a line may start with a
    language marker such as ``[EN]`` or ``[HI]`` which sets the language for
    subsequent lines until another marker appears.  Empty lines are treated as
    paragraph breaks.
    """

    content = Path(path).read_text(encoding="utf-8-sig")
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
