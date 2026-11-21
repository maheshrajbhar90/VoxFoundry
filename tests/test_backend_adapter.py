"""Unit tests for the :class:`voxfoundry.backend_adapter.XTTSAdapter`.

The tests use monkeypatching to avoid downloading the real XTTS model and
to isolate the adapter logic.  The heavy dependency ``TTS`` is replaced by a
minimal stub that mimics the public ``tts`` method.
"""

import os
import pathlib
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
import voxfoundry.backend_adapter as ba


class DummyTTS:
    def __init__(self, device: str):
        self.device = device
        self.synthesizer = type("S", (), {"output_sample_rate": 24000})
        self.calls = 0

    def tts(self, **kwargs):  # pragma: no cover - stub
        self.calls += 1
        # Return a simple sine wave depending on the text length.
        length = len(kwargs.get("text", ""))
        fr = self.synthesizer.output_sample_rate
        t = np.linspace(0, 1, fr, False)
        return 0.1 * np.sin(2 * np.pi * 220 * t[: length])


def tmp_cwd(tmp_path: pathlib.Path):
    """Temporarily change working directory to *tmp_path*.

    Yields the previous working directory so callers can restore it.
    """
    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(prev)


def test_cache_dir_creation(monkeypatch, tmp_path):
    """The adapter must create the cache directory automatically."""
    # Ensure we start in a fresh directory
    with tmp_cwd(tmp_path):
        adapter = ba.XTTSAdapter()
        assert (tmp_path / ".voxfoundry_cache").exists()


def test_load_and_generate(monkeypatch, tmp_path):
    """Loading the adapter and generating audio should use the dummy model."""
    adapter = ba.XTTSAdapter()

    # Patch the model creation to use DummyTTS.
    def fake_load(self, model_path):  # noqa: D401
        self.tts = DummyTTS(device="cpu")
        self.sample_rate = 24000
        # Pretend reference audio exists
        self.reference_audio = Path("voice.wav")
    monkeypatch.setattr(ba.XTTSAdapter, "load", fake_load, raising=True)

    adapter.load("voice.wav")
    # Monkeypatch _synth_once to capture calls
    counter = {"calls": 0}
    original = adapter._synth_once
    def counter_func(text, lang):  # pragma: no cover - trivial
        counter["calls"] += 1
        return original(text, lang)
    monkeypatch.setattr(adapter, "_synth_once", counter_func)

    audio1 = adapter.generate("hello world")
    assert isinstance(audio1, np.ndarray)
    assert audio1.size > 0
    # Generate the same text again – should hit cache
    audio2 = adapter.generate("hello world")
    assert counter["calls"] == 1
    assert np.allclose(audio1, audio2)


def test_unload_and_reuse(monkeypatch, tmp_path):
    """Model should stay loaded across multiple calls unless explicitly unloaded."""
    adapter = ba.XTTSAdapter()
    def fake_load(self, model_path):  # noqa: D401
        self.tts = DummyTTS(device="cpu")
        self.sample_rate = 24000
        self.reference_audio = Path("voice.wav")
    monkeypatch.setattr(ba.XTTSAdapter, "load", fake_load, raising=True)
    adapter.load("voice.wav")
    call_count = adapter.tts.calls
    adapter.generate("text one")
    assert adapter.tts.calls == call_count + 1
    # Unload and ensure it is cleared
    adapter.unload()
    assert adapter.tts is None
    # Re‑loading should reset call count
    adapter.load("voice.wav")
    adapter.generate("text two")
    assert adapter.tts.calls == 1


def test_invalid_reference_path(monkeypatch):
    adapter = ba.XTTSAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.load("/does/not/exist.wav")


def test_invalid_generate_without_load(monkeypatch):
    adapter = ba.XTTSAdapter()
    with pytest.raises(RuntimeError):
        adapter.generate("test")
