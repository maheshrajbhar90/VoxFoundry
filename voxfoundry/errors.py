"""Exception hierarchy for VoxFoundry.

Every error raised deliberately by VoxFoundry derives from :class:`VoxFoundryError`,
so callers can catch the whole family. Each class also inherits the closest
built-in exception, so existing code that catches ``FileNotFoundError``,
``ValueError`` or ``RuntimeError`` keeps working.
"""

from __future__ import annotations

__all__ = [
    "VoxFoundryError",
    "ModelDependencyError",
    "ModelLoadError",
    "DeviceUnavailableError",
    "ReferenceAudioError",
    "ReferenceAudioNotFoundError",
    "UnsupportedLanguageError",
    "TextValidationError",
    "AdapterStateError",
    "InferenceError",
    "AudioValidationError",
]


class VoxFoundryError(Exception):
    """Base class for all VoxFoundry errors."""


class ModelDependencyError(VoxFoundryError, ImportError):
    """A package required by a backend (e.g. ``coqui-tts``, ``torch``) is missing or broken."""


class ModelLoadError(VoxFoundryError, RuntimeError):
    """The model checkpoint could not be loaded or moved to the requested device."""


class DeviceUnavailableError(ModelLoadError):
    """The explicitly requested device (``mps``, ``cuda``) is not available. No fallback is attempted."""


class ReferenceAudioError(VoxFoundryError, ValueError):
    """The reference recording is unreadable, corrupt, silent, too short or otherwise unusable."""


class ReferenceAudioNotFoundError(ReferenceAudioError, FileNotFoundError):
    """The reference recording does not exist."""


class UnsupportedLanguageError(VoxFoundryError, ValueError):
    """The language code is not supported by the loaded model."""


class TextValidationError(VoxFoundryError, ValueError):
    """The text to synthesise is empty or exceeds the backend's per-call limit."""


class AdapterStateError(VoxFoundryError, RuntimeError):
    """An operation was called in the wrong lifecycle state (e.g. ``generate`` before ``load``)."""


class InferenceError(VoxFoundryError, RuntimeError):
    """The model raised an error, or produced no usable audio, during synthesis."""


class AudioValidationError(VoxFoundryError, ValueError):
    """Generated audio failed validation (empty, non-finite, silent, wrong shape or sample rate)."""
