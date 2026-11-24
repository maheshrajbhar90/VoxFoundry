"""Backend adapter abstraction and the XTTS-v2 backend.

All TTS engines implement :class:`BackendAdapter` so that VoxFoundry can use them
interchangeably. Adapters own *model I/O only*: they return the raw model waveform.
Text segmentation lives in :mod:`voxfoundry.text_processor`; optional post-processing
lives in :mod:`voxfoundry.audio`.

Lifecycle of :class:`XTTSAdapter`::

    INITIALIZED --load()--> READY --synthesize()--> GENERATING --> READY
         ^                    |                                      |
         +----- UNLOADED <----+------------- unload() ---------------+

``LOADED`` is the transient state between the checkpoint being in memory and the
reference recording being accepted. Errors never leave the adapter in
``GENERATING``; a failed ``load()`` leaves it in its previous stable state.
"""

from __future__ import annotations

import abc
import contextlib
import enum
import gc
import hashlib
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Tuple

import numpy as np

from .audio import AudioResult, ReferenceInfo, validate_audio, validate_reference
from .errors import (
    AdapterStateError,
    DeviceUnavailableError,
    InferenceError,
    ModelDependencyError,
    ModelLoadError,
    TextValidationError,
    UnsupportedLanguageError,
)

__all__ = [
    "AdapterState",
    "BackendAdapter",
    "XTTSSettings",
    "XTTSAdapter",
    "MockXTTSAdapter",
    "MODEL",
    "XTTS_V2_LANGUAGES",
    "SUPPORTED_DEVICES",
]

logger = logging.getLogger(__name__)

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Languages listed in the XTTS-v2.0.3 config.json / model card. After load() the
# adapter uses the loaded model's own config instead of this list.
XTTS_V2_LANGUAGES: tuple[str, ...] = (
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
)
SUPPORTED_DEVICES = ("mps", "cpu", "cuda")
DEFAULT_DEVICE = "mps"

# Legacy constants (clone_voice.py). Kept as module attributes for backwards compatibility;
# XTTSSettings is the source of truth.
SPEED = 1.05
TEMPERATURE = 0.7
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 5.0
GPT_COND_LEN = 30
GPT_COND_CHUNK_LEN = 6
MAX_REF_LEN = 30
MAX_RETRIES = 2
SEED = 42


class AdapterState(str, enum.Enum):
    INITIALIZED = "initialized"
    LOADED = "loaded"
    READY = "ready"
    GENERATING = "generating"
    UNLOADED = "unloaded"


class BackendAdapter(metaclass=abc.ABCMeta):
    """Abstract base class for all TTS back-ends.

    Contract:

    * ``load(reference_audio)`` prepares the backend for a reference voice.
    * ``generate(text, language)`` returns a 1-D float32 waveform (raw model output).
    * ``synthesize(text, language)`` returns ``(waveform, sample_rate)``.
    * ``synthesize_result(text, language)`` returns an :class:`AudioResult` with metadata.
    * ``unload()`` releases resources; it is idempotent.
    """

    is_mock: bool = False
    sample_rate: int = 24000
    device: str = "cpu"

    @property
    @abc.abstractmethod
    def name(self) -> str:  # pragma: no cover - interface
        """Human-readable backend name."""

    @property
    @abc.abstractmethod
    def supported_languages(self) -> Iterable[str]:  # pragma: no cover - interface
        """Language codes accepted by :meth:`generate`."""

    @abc.abstractmethod
    def load(self, reference_audio: Path | str) -> None:  # pragma: no cover - interface
        """Prepare the backend for ``reference_audio``. Raises on failure."""

    @abc.abstractmethod
    def generate(self, text: str, language: str | None = None) -> np.ndarray:  # pragma: no cover - interface
        """Synthesize ``text`` and return the raw waveform."""

    def synthesize_result(self, text: str, language: str | None = None) -> AudioResult:
        t0 = time.perf_counter()
        audio = self.generate(text, language)
        return AudioResult(
            audio=audio,
            sample_rate=int(self.sample_rate),
            backend=str(self.name),
            device=str(self.device),
            language=language or "en",
            text=text,
            generation_seconds=time.perf_counter() - t0,
            is_mock=self.is_mock,
        )

    def synthesize(self, text: str, language: str | None = None) -> Tuple[np.ndarray, int]:
        """Return ``(waveform, sample_rate)``. Kept for backwards compatibility."""
        result = self.synthesize_result(text, language)
        return result.audio, result.sample_rate

    def unload(self) -> None:
        """Release resources. Default: nothing to release."""


# ---------------------------------------------------------------------------
# XTTS-v2
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XTTSSettings:
    """Inference and speaker-conditioning parameters. Defaults equal ``clone_voice.py``."""

    speed: float = SPEED
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    repetition_penalty: float = REPETITION_PENALTY
    gpt_cond_len: int = GPT_COND_LEN
    gpt_cond_chunk_len: int = GPT_COND_CHUNK_LEN
    max_ref_len: int = MAX_REF_LEN
    sound_norm_refs: bool = False
    seed: int = SEED

    def tts_kwargs(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("seed")
        return d


def _default_tts_factory(model_name: str, progress_bar: bool) -> Any:
    try:
        from TTS.api import TTS
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ModelDependencyError(
            f"coqui-tts is not importable ({exc}). Activate .venv-xtts and see docs/XTTS_MODEL_COMPATIBILITY.md."
        ) from exc
    return TTS(model_name, progress_bar=progress_bar)


@contextlib.contextmanager
def _hf_offline(enabled: bool) -> Iterator[None]:
    """Force Hugging Face hub offline mode for the duration of the block.

    coqui-tts 0.28.0 calls ``snapshot_download`` on every load, which can contact the
    hub and fetch a newer revision. Offline mode makes it use the local cache only.
    """
    if not enabled:
        yield
        return
    prev_env = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    constants = None
    prev_const = None
    try:
        import huggingface_hub.constants as constants  # type: ignore[no-redef]

        prev_const = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = True
    except ImportError:  # pragma: no cover - hub not installed
        constants = None
    try:
        yield
    finally:
        if constants is not None:
            constants.HF_HUB_OFFLINE = prev_const
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env


def _check_device(device: str) -> None:
    if device not in SUPPORTED_DEVICES:
        raise ModelLoadError(f"Unknown device {device!r}; choose one of {', '.join(SUPPORTED_DEVICES)}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ModelDependencyError(f"PyTorch is not importable: {exc}") from exc
    if device == "mps" and not (torch.backends.mps.is_built() and torch.backends.mps.is_available()):
        raise DeviceUnavailableError("Device 'mps' was requested but MPS is not available. Use --device cpu explicitly.")
    if device == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("Device 'cuda' was requested but CUDA is not available.")


def _release_device_memory(device: str) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _seed(seed: int) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.manual_seed(seed)


class XTTSAdapter(BackendAdapter):
    """XTTS-v2 backend using ``coqui-tts``.

    Parameters
    ----------
    device:
        ``"mps"`` (default), ``"cpu"`` or ``"cuda"``. Never changed automatically; an
        unavailable device raises :class:`DeviceUnavailableError`.
    settings:
        Inference parameters (defaults reproduce ``clone_voice.py``).
    cache_dir:
        Optional directory for an on-disk cache of *raw output audio*. ``None`` (default)
        disables it. This is unrelated to model reuse, which is always on.
    allow_download:
        ``False`` (default) forces Hugging Face offline mode while loading, so a missing
        checkpoint raises instead of being downloaded.
    max_retries:
        Extra attempts (new seed each) when output is non-finite/empty or its duration is
        implausible for the text, as in ``clone_voice.py``.
    tts_factory:
        ``callable(model_name, progress_bar) -> TTS``. Injected by tests; defaults to ``TTS.api.TTS``.
    """

    name = "XTTS-v2"

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        *,
        model_name: str = MODEL,
        settings: XTTSSettings | None = None,
        cache_dir: Path | str | None = None,
        allow_download: bool = False,
        max_retries: int = MAX_RETRIES,
        progress_bar: bool = False,
        tts_factory: Callable[[str, bool], Any] | None = None,
    ) -> None:
        if device not in SUPPORTED_DEVICES:
            raise ValueError(f"Unknown device {device!r}; choose one of {', '.join(SUPPORTED_DEVICES)}")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.device = device
        self.model_name = model_name
        self.settings = settings or XTTSSettings()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.allow_download = allow_download
        self.max_retries = max_retries
        self.progress_bar = progress_bar
        self._tts_factory = tts_factory or _default_tts_factory

        self.tts: Any | None = None
        self.reference_audio: Path | None = None
        self.reference_info: ReferenceInfo | None = None
        self._reference_digest: str | None = None
        self.sample_rate = 24000
        self._languages: tuple[str, ...] = XTTS_V2_LANGUAGES
        self._char_limits: dict[str, int] = {}
        self.state = AdapterState.INITIALIZED

        # Counters used to verify model reuse.
        self.model_load_count = 0
        self.inference_call_count = 0
        self.last_load_seconds: float | None = None

    # ------------------------------------------------------------------ properties
    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self._languages

    @property
    def is_model_loaded(self) -> bool:
        return self.tts is not None

    @property
    def is_ready(self) -> bool:
        return self.state is AdapterState.READY

    def char_limit(self, language: str) -> int | None:
        return self._char_limits.get(language)

    # ------------------------------------------------------------------ lifecycle
    def load(self, reference_audio: Path | str) -> None:
        """Validate ``reference_audio``, load the checkpoint if needed, and become READY.

        The checkpoint is loaded at most once per lifecycle; calling ``load`` again with a
        different reference only swaps the reference. Speaker conditioning latents are
        computed by Coqui from the reference on each synthesis call.
        """
        if self.state is AdapterState.GENERATING:
            raise AdapterStateError("Cannot load while generating")

        info = validate_reference(reference_audio)  # raises ReferenceAudioError; state unchanged
        for w in info.warnings:
            logger.warning("Reference: %s", w)

        if self.tts is None:
            self._load_model()

        self.reference_audio = info.path
        self.reference_info = info
        self._reference_digest = hashlib.sha1(info.path.read_bytes()).hexdigest()
        self.state = AdapterState.READY
        logger.info(
            "XTTS ready: device=%s reference=%s (%.1fs, %d Hz)",
            self.device, info.path, info.duration_seconds, info.sample_rate,
        )

    def _load_model(self) -> None:
        _check_device(self.device)
        t0 = time.perf_counter()
        tts = None
        try:
            with _hf_offline(not self.allow_download):
                tts = self._tts_factory(self.model_name, self.progress_bar)
            tts = tts.to(self.device)
        except ModelDependencyError:
            raise
        except ImportError as exc:
            raise ModelDependencyError(f"Missing dependency while loading XTTS: {exc}") from exc
        except Exception as exc:
            del tts
            gc.collect()
            hint = "" if self.allow_download else " (downloads are disabled; is the checkpoint in the local cache?)"
            raise ModelLoadError(f"Could not load {self.model_name} on {self.device}{hint}: {exc}") from exc

        self.tts = tts
        self.state = AdapterState.LOADED
        self.model_load_count += 1
        self.last_load_seconds = time.perf_counter() - t0

        synth = getattr(tts, "synthesizer", None)
        self.sample_rate = int(getattr(synth, "output_sample_rate", 24000) or 24000)
        model = getattr(synth, "tts_model", None)
        langs = getattr(getattr(model, "config", None), "languages", None)
        if langs:
            self._languages = tuple(langs)
        limits = getattr(getattr(model, "tokenizer", None), "char_limits", None)
        if isinstance(limits, dict):
            self._char_limits = dict(limits)
        logger.info(
            "Loaded %s on %s in %.1fs (sample_rate=%d, %d languages)",
            self.model_name, self.device, self.last_load_seconds, self.sample_rate, len(self._languages),
        )

    def unload(self) -> None:
        """Release the model and reference. Idempotent; safe in any state except GENERATING."""
        if self.state is AdapterState.GENERATING:
            raise AdapterStateError("Cannot unload while generating")
        had_model = self.tts is not None
        self.tts = None
        self.reference_audio = None
        self.reference_info = None
        self._reference_digest = None
        if had_model:
            gc.collect()
            _release_device_memory(self.device)
        self.state = AdapterState.UNLOADED

    # ------------------------------------------------------------------ inference
    def _validate_request(self, text: Any, language: str | None) -> tuple[str, str, list[str]]:
        if self.state is not AdapterState.READY or self.tts is None or self.reference_audio is None:
            raise AdapterStateError(f"XTTSAdapter is not ready (state={self.state.value}); call load() first")
        if not isinstance(text, str) or not text.strip():
            raise TextValidationError("Text to synthesize is empty")
        lang = language or "en"
        if lang not in self._languages:
            raise UnsupportedLanguageError(
                f"Language {lang!r} is not supported by {self.name}. Supported: {', '.join(self._languages)}"
            )
        warnings: list[str] = []
        limit = self._char_limits.get(lang)
        if limit is not None and len(text) > limit:
            warnings.append(
                f"Text is {len(text)} chars, above XTTS's {limit}-char limit for {lang!r}; "
                "Coqui splits it into sentences, but quality may drop. Chunk with text_processor.make_chunks."
            )
        return text, lang, warnings

    def _expected_seconds(self, text: str) -> float:
        return max(0.8, len(text) / (13.0 * self.settings.speed))

    def _cache_path(self, text: str, lang: str) -> Path | None:
        if self.cache_dir is None:
            return None
        sig = "|".join(
            [
                "raw-v1", self.model_name, self.device, text, lang,
                repr(sorted(asdict(self.settings).items())), str(self._reference_digest),
            ]
        )
        return self.cache_dir / (hashlib.sha1(sig.encode("utf-8")).hexdigest()[:20] + ".npy")

    def _call_model(self, text: str, lang: str) -> np.ndarray:
        assert self.tts is not None and self.reference_audio is not None
        self.inference_call_count += 1
        try:
            out = self.tts.tts(
                text=text,
                speaker_wav=str(self.reference_audio),
                language=lang,
                **self.settings.tts_kwargs(),
            )
        except Exception as exc:
            raise InferenceError(f"XTTS inference failed on {self.device} for {text[:60]!r}: {exc}") from exc
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def synthesize_result(self, text: str, language: str | None = None) -> AudioResult:
        text, lang, warnings = self._validate_request(text, language)

        cp = self._cache_path(text, lang)
        if cp is not None and cp.exists():
            audio = np.load(cp)
            validate_audio(audio, self.sample_rate)
            return AudioResult(
                audio=audio, sample_rate=self.sample_rate, backend=self.name, device=self.device,
                language=lang, text=text, generation_seconds=0.0, from_cache=True, warnings=warnings,
            )

        expected = self._expected_seconds(text)
        best: np.ndarray | None = None
        attempts = 0
        duration_ok = False
        self.state = AdapterState.GENERATING
        t0 = time.perf_counter()
        try:
            for attempt in range(self.max_retries + 1):
                attempts = attempt + 1
                _seed(self.settings.seed + attempt)
                wav = self._call_model(text, lang)
                if wav.size == 0 or not np.all(np.isfinite(wav)):
                    warnings.append(f"attempt {attempts}: empty or non-finite output, retrying")
                    logger.warning(warnings[-1])
                    continue
                best = wav
                dur = wav.size / self.sample_rate
                if 0.35 * expected <= dur <= 2.2 * expected + 1.5:
                    duration_ok = True
                    break
                warnings.append(
                    f"attempt {attempts}: duration {dur:.1f}s outside expected range for ~{expected:.1f}s"
                )
                logger.warning(warnings[-1])
        finally:
            self.state = AdapterState.READY
        elapsed = time.perf_counter() - t0

        if best is None:
            raise InferenceError(f"XTTS produced no usable audio after {attempts} attempts for: {text[:60]!r}")
        if not duration_ok:
            warnings.append(f"returned attempt with implausible duration after {attempts} attempts")
            logger.warning(warnings[-1])

        validate_audio(best, self.sample_rate)  # AudioValidationError on silence etc.

        if cp is not None:
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.save(cp, best)

        return AudioResult(
            audio=best,
            sample_rate=self.sample_rate,
            backend=self.name,
            device=self.device,
            language=lang,
            text=text,
            generation_seconds=elapsed,
            attempts=attempts,
            warnings=warnings,
            metadata={
                "seed": self.settings.seed + attempts - 1,
                "expected_seconds": expected,
                "duration_ok": duration_ok,
                "settings": asdict(self.settings),
                "model": self.model_name,
            },
        )

    def generate(self, text: str, language: str | None = None) -> np.ndarray:
        """Return the raw waveform (no trimming, fades or level-matching)."""
        return self.synthesize_result(text, language).audio


# ---------------------------------------------------------------------------
# Mock (tests and dry runs only; never used as a fallback)
# ---------------------------------------------------------------------------
class MockXTTSAdapter(BackendAdapter):
    """Synthetic 220 Hz tone generator. **Not speech.** Selected only explicitly."""

    name = "XTTS-v2-mock"
    supported_languages = list(XTTS_V2_LANGUAGES)
    sample_rate = 24000
    is_mock = True

    def __init__(self, device: str = "cpu", **_: Any) -> None:
        self.device = device
        self.state = AdapterState.INITIALIZED

    def load(self, reference_audio: Path | str) -> None:
        self.state = AdapterState.READY

    def generate(self, text: str, language: str | None = None) -> np.ndarray:
        duration = max(0.2, len(text) * 0.05)
        n_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def unload(self) -> None:
        self.state = AdapterState.UNLOADED
