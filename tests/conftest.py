"""Shared fixtures.

``FakeTTS`` imitates the parts of ``TTS.api.TTS`` that :class:`XTTSAdapter` uses, so
unit and integration tests exercise the real adapter code without a checkpoint.
Nothing here produces or claims to produce real speech.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

SR = 24000
XTTS_LANGS = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]


def speech_like(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Broadband, amplitude-modulated harmonic signal with pauses (passes the speech heuristic)."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 15))
    syllables = (np.sin(2 * np.pi * 4 * t) > -0.2).astype(float)
    noise = rng.normal(0, 0.05, n)
    wav = 0.15 * voiced * syllables + noise * syllables
    return (0.5 * wav / np.max(np.abs(wav))).astype(np.float32)


class FakeTTS:
    """Stand-in for ``TTS.api.TTS`` (fake model; output is synthetic)."""

    instances: list["FakeTTS"] = []

    def __init__(self, model_name: str = "fake", progress_bar: bool = False, *, output=None, raises=None):
        self.model_name = model_name
        self.device = "cpu"
        self.calls: list[dict[str, Any]] = []
        self.hf_offline_at_init = os.environ.get("HF_HUB_OFFLINE")
        self._output = output
        self._raises = raises
        cfg = type("Cfg", (), {"languages": list(XTTS_LANGS)})()
        tok = type("Tok", (), {"char_limits": {"en": 250, "hi": 150}})()
        model = type("Model", (), {"config": cfg, "tokenizer": tok})()
        self.synthesizer = type("Synth", (), {"output_sample_rate": SR, "tts_model": model})()
        FakeTTS.instances.append(self)

    def to(self, device: str) -> "FakeTTS":
        self.device = device
        return self

    def tts(self, **kwargs: Any) -> list[float]:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if self._output is not None:
            out = self._output(len(self.calls), kwargs) if callable(self._output) else self._output
            return list(np.asarray(out, dtype=np.float32))
        seconds = max(0.5, len(kwargs["text"]) / 13.0)
        return list(speech_like(seconds, seed=len(self.calls)))


@pytest.fixture
def fake_factory():
    """Return ``make(**fake_kwargs) -> factory`` for ``XTTSAdapter(tts_factory=...)``."""
    FakeTTS.instances.clear()

    def make(**fake_kwargs):
        def factory(model_name: str, progress_bar: bool) -> FakeTTS:
            return FakeTTS(model_name, progress_bar, **fake_kwargs)

        return factory

    return make


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    """A 6-second synthetic reference recording (meets the 5 s legacy minimum)."""
    p = tmp_path / "reference.wav"
    sf.write(str(p), speech_like(6.0, 22050), 22050, subtype="PCM_16")
    return p


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run every test in its own temp directory so nothing is written into the repo."""
    monkeypatch.chdir(tmp_path)
