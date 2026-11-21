"""VoxFoundry package

This package provides a modular framework for voice‑cloning, audio‑mastering and
text‑to‑speech back‑end adapters.  The legacy scripts in the repository are
wrappers around this API.
"""

"""Public entry‑point for the :mod:`voxfoundry` package.

The original ``__init__`` attempted to expose a non‑existent ``TextProcessor``
class which caused an :class:`ImportError` as soon as the package was
imported.  The module now re‑exports the useful symbols from the
sub‑modules so that consumers can simply ``import voxfoundry``.
"""

from .text_processor import (
    parse_script,
    make_chunks,
    pause_after,
    silence,
)
from .backend_adapter import BackendAdapter, XTTSAdapter, MockXTTSAdapter
from .cli import main

__all__ = [
    "parse_script",
    "make_chunks",
    "pause_after",
    "silence",
    "BackendAdapter",
    "XTTSAdapter",
    "MockXTTSAdapter",
    "main",
]