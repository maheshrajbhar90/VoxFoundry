"""REAL-MODEL tests: load the actual XTTS-v2 checkpoint and run inference.

Run explicitly:  python -m pytest -m real_model -v
Environment:     CLONEMYVOICE_TEST_DEVICE (default: mps), CLONEMYVOICE_TEST_REFERENCE
                 (default: references/my_voice.wav).
These tests skip when the checkpoint is not already in the local Hugging Face cache,
and the adapter forces offline mode, so they never download anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from clonemyvoice.audio import analyze_audio
from clonemyvoice.backend_adapter import XTTSAdapter
from clonemyvoice.errors import UnsupportedLanguageError

pytestmark = pytest.mark.real_model

REPO = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("CLONEMYVOICE_TEST_DEVICE", "mps")
REFERENCE = Path(os.environ.get("CLONEMYVOICE_TEST_REFERENCE", REPO / "references" / "my_voice.wav"))


def _checkpoint_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache("tts-hub/XTTS-v2", "model.pth")
    return isinstance(hit, str) and Path(hit).is_file()


if not _checkpoint_cached():
    pytest.skip("XTTS-v2 checkpoint not in local cache (no downloads in tests)", allow_module_level=True)
if not REFERENCE.is_file():
    pytest.skip(f"Reference recording not found: {REFERENCE}", allow_module_level=True)


@pytest.fixture(scope="module")
def adapter():
    a = XTTSAdapter(device=DEVICE)
    a.load(REFERENCE)
    yield a
    a.unload()


def test_model_loads_on_requested_device(adapter):
    assert adapter.is_ready and adapter.model_load_count == 1
    assert adapter.device == DEVICE and adapter.sample_rate == 24000


def test_english_generates_speech(adapter):
    r = adapter.synthesize_result("This is a real voice generation test.", "en")
    s = analyze_audio(r.audio, r.sample_rate)
    assert s.non_finite == 0 and 1.0 < s.duration_seconds < 8.0
    assert s.speech_likely, s
    assert not r.is_mock


def test_model_reused_across_calls(adapter):
    before = adapter.inference_call_count
    for text in ("First reuse check.", "Second reuse check."):
        adapter.synthesize_result(text, "en")
    assert adapter.model_load_count == 1
    assert adapter.inference_call_count >= before + 2


def test_unsupported_language_rejected_before_inference(adapter):
    before = adapter.inference_call_count
    with pytest.raises(UnsupportedLanguageError):
        adapter.synthesize_result("Hello", "xx")
    assert adapter.inference_call_count == before


def test_unload_and_reload():
    a = XTTSAdapter(device=DEVICE)
    a.load(REFERENCE)
    a.unload()
    assert a.tts is None
    a.load(REFERENCE)
    r = a.synthesize_result("Reload works.", "en")
    assert a.model_load_count == 2 and r.audio.size > 0
    a.unload()
