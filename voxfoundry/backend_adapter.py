"""Backend adapter abstraction.

All TTS engines must implement the :class:`BackendAdapter` interface so that
``VoxFoundry`` can use them interchangeably.  The interface is intentionally
minimal – enough for the current scripts and for future adapters.
"""

from __future__ import annotations

import abc
import hashlib
from pathlib import Path
import torch
from typing import Iterable, Tuple

import numpy as np
import traceback
from TTS.api import TTS

__all__ = ["BackendAdapter", "XTTSAdapter", "MockXTTSAdapter"]


class BackendAdapter(metaclass=abc.ABCMeta):
    """Abstract base class for all TTS back‑ends.

    Concrete adapters must provide a synchronous ``generate`` method that
    accepts text, a language code and returns a 1‑D :class:`numpy.ndarray`
    containing the audio waveform (float32, 16‑kHz by default).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:  # pragma: no cover - interface
        """Return a human readable name for the backend."""

    @property
    @abc.abstractmethod
    def supported_languages(self) -> Iterable[str]:  # pragma: no cover - interface
        """Return an iterable of ISO‑639 language codes supported by the model."""

    @abc.abstractmethod
    def load(self, model_path: Path | str) -> None:  # pragma: no cover - interface
        """Load the model from ``model_path``.

        Adapters may perform heavy initialisation here.  The method should not
        return a value; it should raise on failure.
        """

    @abc.abstractmethod
    def generate(self, text: str, language: str | None = None) -> np.ndarray:  # pragma: no cover - interface
        """Synthesize ``text`` into audio.

        Parameters
        ----------
        text:
            The text to speak.
        language:
            Optional language code; adapters may ignore or validate it.
        """

    # Convenience helpers -------------------------------------------------
    def synthesize(self, text: str, language: str | None = None) -> Tuple[np.ndarray, int]:
        """Return the waveform and sample rate.

        The base implementation assumes a default sample rate of 24000 Hz.
        Sub‑classes can override to return a different rate.
        """

        audio = self.generate(text, language)
        return audio, getattr(self, "sample_rate", 24000)


# ---------------------------------------------------------------------
# Real XTTS-v2 adapter – implements inference using the official TTS
# library.  The implementation mirrors the logic of ``clone_voice.py`` but
# is encapsulated in a class to fit the ``BackendAdapter`` contract.
# ---------------------------------------------------------------------

# Helper constants copied from the legacy script.  They are defined in the
# adapter so that the public API remains stable even if the legacy
# implementation changes.
SPEED = 1.05
TEMPERATURE = 0.7
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 5.0
GPT_COND_LEN = 30
GPT_COND_CHUNK_LEN = 6
MAX_REF_LEN = 30
TARGET_RMS = 0.08
MAX_RETRIES = 2
SEED = 42
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

class XTTSAdapter(BackendAdapter):
    """Production-ready XTTS-v2 adapter.

    The adapter loads the model once and re‑uses it for every call to
    :py:meth:`generate`.  It implements the same caching, retry logic and
    audio post‑processing as the legacy script so that behaviour is
    indistinguishable from the original implementation.
    """

    def __init__(self) -> None:
        """Initialise state and ensure the cache directory exists.

        The cache directory is ``<cwd>/.voxfoundry_cache``.  It is created
        lazily to avoid permission errors during early imports.  Permission
        issues are surfaced to the user rather than silently ignored.
        """
        self.tts: TTS | None = None
        self.reference_audio: Path | None = None
        self.sample_rate: int = 24000
        self._warned_fallback = False
        # Create cache directory; propagate errors
        self.cache_dir = Path.cwd() / ".voxfoundry_cache"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:  # pragma: no cover – unlikely in tests
            raise RuntimeError(
                f"Cannot create cache directory {self.cache_dir!s}: {exc}"
            ) from exc

    name = "XTTS-v2"
    supported_languages = ["en", "hi"]
    sample_rate = 24000  # default value, will be overridden by the model

    def load(self, model_path: Path | str) -> None:
        """Load the XTTS-v2 model.

        Parameters
        ----------
        model_path:
            The path to the reference audio file.  The function name ``load``
            follows the :class:`BackendAdapter` contract but in this
            implementation it actually receives the reference audio path.
        """
        # Store reference audio path and verify existence
        self.reference_audio = Path(model_path)
        if not self.reference_audio.is_file():
            raise FileNotFoundError(f"Reference audio not found: {self.reference_audio}")

        # Determine device – prefer CUDA, then MPS, then CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        # Load the model once.  ``TTS`` automatically handles weights.
        try:
            self.tts = TTS(MODEL).to(device)
            # Override sample rate if the model provides one
            self.sample_rate = getattr(self.tts.synthesizer, "output_sample_rate", 24000)
        except Exception as exc:  # pragma: no cover – model load failure
            print("\nERROR: Could not load XTTS model.")
            traceback.print_exc()
            raise exc

    # ------------------------------------------------------------------
    # Internal helpers – largely copied from the legacy script
    # ------------------------------------------------------------------
    def _clean_clip(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak <= 0:
            return wav
        idx = np.where(np.abs(wav) > 0.02 * peak)[0]
        if idx.size:
            keep = int(0.05 * sr)
            wav = wav[max(0, idx[0] - keep) : min(len(wav), idx[-1] + keep)].copy()
        n = int(sr * 0.012)
        if len(wav) > 2 * n:
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            wav[:n] *= ramp
            wav[-n:] *= ramp[::-1]
        rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-9
        wav = wav * float(np.clip(TARGET_RMS / rms, 0.5, 2.5))
        peak = float(np.max(np.abs(wav)))
        if peak > 0.97:
            wav *= 0.97 / peak
        return wav

    def _expected_seconds(self, text: str) -> float:
        return max(0.8, len(text) / (13.0 * SPEED))

    def _cache_path(self, text: str, lang: str) -> Path:
        sig = (
            f"{text}|{lang}|{SPEED}|{TEMPERATURE}|{TOP_P}|{TOP_K}|{REPETITION_PENALTY}|"
            f"{SEED}|{self.reference_audio.stat().st_mtime_ns}"
        )
        return self.cache_dir / (hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16] + ".npy")

    def _synth_once(self, text: str, lang: str) -> np.ndarray:
        # ``speaker_wav`` points to the reference audio file
        try:
            return self.tts.tts(
                text=text,
                speaker_wav=str(self.reference_audio),
                language=lang,
                speed=SPEED,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                gpt_cond_len=GPT_COND_LEN,
                gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
                max_ref_len=MAX_REF_LEN,
                sound_norm_refs=False,
            )
        except TypeError:
            if not self._warned_fallback:
                print("\nNote: this TTS version rejected some tuning options; using basic settings.")
                self._warned_fallback = True
            return self.tts.tts(text=text, speaker_wav=str(self.reference_audio), language=lang, speed=SPEED)

    def _get_clip(self, text: str, lang: str, sr: int) -> np.ndarray:
        cp = self._cache_path(text, lang)
        if cp.exists():
            return np.load(cp)
        exp = self._expected_seconds(text)
        best = None
        for attempt in range(MAX_RETRIES + 1):
            torch.manual_seed(SEED + attempt)
            wav = np.asarray(self._synth_once(text, lang), dtype=np.float32).reshape(-1)
            if wav.size == 0 or not np.all(np.isfinite(wav)):
                continue
            wav = self._clean_clip(wav, sr)
            dur = len(wav) / sr
            best = wav
            if 0.35 * exp <= dur <= 2.2 * exp + 1.5:
                break
        if best is None:
            raise RuntimeError(f"XTTS produced no usable audio for: {text}")
        try:
            np.save(cp, best)
        except OSError as exc:  # pragma: no cover – disk issues
            raise RuntimeError(f"Failed to write cache file {cp!s}: {exc}") from exc
        return best

    # ------------------------------------------------------------------
    # Public API – ``generate`` returns a 1‑D :class:`numpy.ndarray`
    # ------------------------------------------------------------------
    def generate(self, text: str, language: str | None = None) -> np.ndarray:
        if self.tts is None or self.reference_audio is None:
            raise RuntimeError("XTTSAdapter not initialized – call load() first")
        lang = language or "en"
        return self._get_clip(text, lang, self.sample_rate)

    # ------------------------------------------------------------------
    # Explicit lifecycle management
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """Unload the model and free GPU memory if applicable.

        This method is idempotent and safe to call multiple times.
        """
        if self.tts is not None:
            # Clearing the model from memory releases GPU buffers.
            self.tts = None
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover – CUDA not available
                pass
        self.reference_audio = None

# The original placeholder remains for compatibility with tests that
# explicitly import ``MockXTTSAdapter``.
class MockXTTSAdapter(BackendAdapter):  # pragma: no cover - trivial
    name = "XTTS-v2-mock"
    supported_languages = ["en", "hi"]
    sample_rate = 24000

    def load(self, model_path: Path | str) -> None:  # pragma: no cover - trivial
        pass

    def generate(self, text: str, language: str | None = None) -> np.ndarray:  # pragma: no cover - trivial
        # Return a short synthetic audio clip.
        # The previous implementation returned an array of zeros which caused
        # the CLI unit tests to fail.  A simple sine wave that is not silent
        # allows the tests to validate non‑silence without pulling in heavy
        # dependencies.
        duration = max(0.2, len(text) * 0.05)  # seconds
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # 220 Hz tone – audible but simple
        return 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
