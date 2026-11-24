"""Audio contract, analysis and validation shared by every backend.

Nothing in this module modifies audio unless the function name says so
(:func:`legacy_clean_clip`). Reference recordings are only read, never written.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import AudioValidationError, ReferenceAudioError, ReferenceAudioNotFoundError

__all__ = [
    "AudioStats",
    "AudioResult",
    "ReferenceInfo",
    "analyze_audio",
    "validate_audio",
    "validate_reference",
    "legacy_clean_clip",
    "MIN_REFERENCE_SECONDS",
    "LONG_REFERENCE_SECONDS",
    "MIN_REFERENCE_PEAK",
    "SILENCE_PEAK",
]

# Legacy thresholds from clone_voice.py.
MIN_REFERENCE_SECONDS = 5.0
LONG_REFERENCE_SECONDS = 35.0
MIN_REFERENCE_PEAK = 0.02
LEGACY_TARGET_RMS = 0.08

# Output is treated as silent below this absolute peak (≈ -80 dBFS).
SILENCE_PEAK = 1e-4

# Speech-presence heuristic parameters.
_FRAME_SECONDS = 0.02
_ACTIVE_FLOOR_DB = -45.0  # absolute frame-RMS floor
_ACTIVE_RELATIVE_DB = -35.0  # relative to the loudest frame
_TONAL_BAND_HZ = 15.0


@dataclass(frozen=True)
class AudioStats:
    """Objective measurements of a mono waveform.

    ``speech_likely`` is a heuristic (active-frame ratio + spectral concentration).
    It separates speech from silence and steady test tones; it does **not** prove
    that intelligible speech was produced. Manual listening is still required.
    """

    samples: int
    sample_rate: int
    duration_seconds: float
    peak: float
    rms: float
    rms_dbfs: float
    non_finite: int
    active_ratio: float
    tonal_concentration: float
    speech_likely: bool

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class AudioResult:
    """Output of one synthesis call.

    ``audio`` is the raw model output: mono float32, no normalisation, trimming or mastering.
    """

    audio: np.ndarray
    sample_rate: int
    backend: str
    device: str
    language: str
    text: str
    generation_seconds: float
    is_mock: bool = False
    attempts: int = 1
    from_cache: bool = False
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return 1

    @property
    def duration_seconds(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    @property
    def real_time_factor(self) -> float:
        """Generation time / audio duration (< 1 is faster than real time)."""
        d = self.duration_seconds
        return self.generation_seconds / d if d > 0 else float("inf")


@dataclass(frozen=True)
class ReferenceInfo:
    path: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    subtype: str
    peak: float
    rms: float
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _to_mono_float(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 2:
        a = a.mean(axis=1)
    return a.reshape(-1)


def analyze_audio(audio: np.ndarray, sample_rate: int) -> AudioStats:
    """Measure ``audio`` without modifying it. Accepts 1-D or (samples, channels) arrays."""
    a = _to_mono_float(audio)
    n = int(a.size)
    finite_mask = np.isfinite(a)
    non_finite = int(n - finite_mask.sum())
    f = a[finite_mask] if non_finite else a

    if f.size == 0 or sample_rate <= 0:
        return AudioStats(n, sample_rate, 0.0, 0.0, 0.0, float("-inf"), non_finite, 0.0, 0.0, False)

    peak = float(np.max(np.abs(f)))
    rms = float(np.sqrt(np.mean(f**2)))
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else float("-inf")
    duration = n / sample_rate

    # Active-frame ratio.
    frame = max(1, int(sample_rate * _FRAME_SECONDS))
    n_frames = f.size // frame
    if n_frames:
        frames = f[: n_frames * frame].reshape(n_frames, frame)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-12
        frame_db = 20 * np.log10(frame_rms)
        threshold = max(_ACTIVE_FLOOR_DB, float(frame_db.max()) + _ACTIVE_RELATIVE_DB)
        active_ratio = float(np.mean(frame_db > threshold))
    else:
        active_ratio = 0.0

    # Fraction of spectral energy in a narrow band around the strongest frequency.
    # A steady sine is ~1.0; speech is spread over many harmonics/formants (typically < 0.2).
    spec = np.abs(np.fft.rfft(f)) ** 2
    total = float(spec.sum())
    if total > 0:
        k = int(np.argmax(spec))
        half_band = max(1, int(_TONAL_BAND_HZ * f.size / sample_rate))
        tonal = float(spec[max(0, k - half_band) : k + half_band + 1].sum() / total)
    else:
        tonal = 0.0

    speech_likely = (
        peak >= SILENCE_PEAK and duration >= 0.3 and active_ratio >= 0.2 and tonal < 0.5 and non_finite == 0
    )
    return AudioStats(
        samples=n,
        sample_rate=sample_rate,
        duration_seconds=duration,
        peak=peak,
        rms=rms,
        rms_dbfs=float(rms_dbfs),
        non_finite=non_finite,
        active_ratio=active_ratio,
        tonal_concentration=tonal,
        speech_likely=bool(speech_likely),
    )


def validate_audio(audio: Any, sample_rate: Any, *, require_speech: bool = False) -> AudioStats:
    """Raise :class:`AudioValidationError` if ``audio`` is not a usable mono waveform.

    Checks type, shape (1-D), sample rate, emptiness, NaN/Inf and silence. With
    ``require_speech=True`` it also rejects output the speech heuristic flags
    (e.g. a steady tone). Returns the measurements on success.
    """
    if not isinstance(audio, np.ndarray):
        raise AudioValidationError(f"Audio must be a numpy.ndarray, got {type(audio).__name__}")
    if audio.ndim != 1:
        raise AudioValidationError(f"Audio must be 1-D (mono), got shape {audio.shape}")
    if not isinstance(sample_rate, (int, np.integer)) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise AudioValidationError(f"Invalid sample rate: {sample_rate!r}")
    if audio.size == 0:
        raise AudioValidationError("Audio is empty")
    if not np.issubdtype(audio.dtype, np.floating):
        raise AudioValidationError(f"Audio must be floating point, got dtype {audio.dtype}")

    stats = analyze_audio(audio, int(sample_rate))
    if stats.non_finite:
        raise AudioValidationError(f"Audio contains {stats.non_finite} NaN/infinite samples")
    if stats.peak < SILENCE_PEAK:
        raise AudioValidationError(f"Audio is silent (peak {stats.peak:.2e})")
    if require_speech and not stats.speech_likely:
        raise AudioValidationError(
            "Audio does not look like speech "
            f"(active_ratio={stats.active_ratio:.2f}, tonal_concentration={stats.tonal_concentration:.2f})"
        )
    return stats


# ---------------------------------------------------------------------------
# Reference recordings
# ---------------------------------------------------------------------------
def validate_reference(
    path: str | os.PathLike[str],
    *,
    min_seconds: float = MIN_REFERENCE_SECONDS,
    min_peak: float = MIN_REFERENCE_PEAK,
) -> ReferenceInfo:
    """Validate a reference recording without modifying it.

    Mirrors ``clone_voice.validate_files`` (exists, non-empty, decodable, has
    frames, ≥ ``min_seconds``, peak ≥ ``min_peak``) and adds read-permission and
    finite-sample checks.
    """
    import soundfile as sf

    p = Path(path)
    if not p.exists():
        raise ReferenceAudioNotFoundError(f"Reference audio not found: {p}")
    if not p.is_file():
        raise ReferenceAudioError(f"Reference audio is not a file: {p}")
    if not os.access(p, os.R_OK):
        raise ReferenceAudioError(f"Reference audio is not readable (permissions): {p}")
    if p.stat().st_size == 0:
        raise ReferenceAudioError(f"Reference audio file is empty: {p}")

    try:
        info = sf.info(str(p))
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
    except Exception as exc:  # soundfile raises RuntimeError/LibsndfileError for bad files
        raise ReferenceAudioError(
            f"Cannot decode reference audio {p} ({exc}). "
            "Convert it to WAV, e.g.: ffmpeg -i input.m4a -ar 22050 -ac 1 reference.wav"
        ) from exc

    if info.frames <= 0 or data.size == 0:
        raise ReferenceAudioError(f"Reference audio contains no frames: {p}")
    if info.duration < min_seconds:
        raise ReferenceAudioError(f"Reference audio too short: {info.duration:.2f}s (minimum {min_seconds:.0f}s)")

    mono = _to_mono_float(data)
    if not np.all(np.isfinite(mono)):
        raise ReferenceAudioError(f"Reference audio contains NaN/infinite samples: {p}")
    peak = float(np.max(np.abs(mono)))
    if peak < min_peak:
        raise ReferenceAudioError(f"Reference audio is almost silent (peak {peak:.4f} < {min_peak})")
    rms = float(np.sqrt(np.mean(mono**2)))

    warnings: list[str] = []
    if info.duration > LONG_REFERENCE_SECONDS:
        warnings.append(
            f"Reference is {info.duration:.1f}s; XTTS uses the first max_ref_len seconds. "
            "A clean 15-30s take usually works best."
        )
    return ReferenceInfo(
        path=p,
        duration_seconds=float(info.duration),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=str(info.subtype),
        peak=peak,
        rms=rms,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Legacy post-processing (opt-in only)
# ---------------------------------------------------------------------------
def legacy_clean_clip(wav: np.ndarray, sr: int, target_rms: float = LEGACY_TARGET_RMS) -> np.ndarray:
    """Exact port of ``clone_voice.clean_clip``: trim, 12 ms fades, RMS level-match, peak limit.

    This **modifies** audio. CloneMyVoice never applies it implicitly; the CLI applies
    it only with ``--legacy-clip-cleanup``.
    """
    wav = np.asarray(wav, dtype=np.float32).reshape(-1).copy()
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
    rms = float(np.sqrt(np.mean(wav**2))) + 1e-9
    wav = wav * float(np.clip(target_rms / rms, 0.5, 2.5))
    peak = float(np.max(np.abs(wav)))
    if peak > 0.97:
        wav *= 0.97 / peak
    return wav.astype(np.float32)
