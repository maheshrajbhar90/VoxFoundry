"""CloneMyVoice: a modular framework for voice-cloning back-ends.

Importing the package is lightweight: ``torch`` and ``coqui-tts`` are imported only
when an :class:`XTTSAdapter` loads its model.
"""

from .audio import AudioResult, AudioStats, analyze_audio, validate_audio, validate_reference
from .backend_adapter import AdapterState, BackendAdapter, MockXTTSAdapter, XTTSAdapter, XTTSSettings
from .errors import (
    AdapterStateError,
    AudioValidationError,
    DeviceUnavailableError,
    InferenceError,
    ModelDependencyError,
    ModelLoadError,
    ReferenceAudioError,
    ReferenceAudioNotFoundError,
    TextValidationError,
    UnsupportedLanguageError,
    CloneMyVoiceError,
)
from .text_processor import Segment, build_segments, make_chunks, parse_script, parse_script_text, pause_after, silence

__all__ = [
    "parse_script",
    "parse_script_text",
    "make_chunks",
    "pause_after",
    "silence",
    "build_segments",
    "Segment",
    "AudioResult",
    "AudioStats",
    "analyze_audio",
    "validate_audio",
    "validate_reference",
    "AdapterState",
    "BackendAdapter",
    "XTTSAdapter",
    "XTTSSettings",
    "MockXTTSAdapter",
    "main",
    "CloneMyVoiceError",
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


def __getattr__(name: str):
    # Lazy so that ``python -m clonemyvoice.cli`` does not import the CLI module twice.
    if name == "main":
        from .cli import main

        return main
    raise AttributeError(f"module 'clonemyvoice' has no attribute {name!r}")
