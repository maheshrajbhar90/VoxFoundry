"""Unit tests for :mod:`voxfoundry.audio` (validation, analysis, reference checks)."""

import os
import sys

import numpy as np
import pytest
import soundfile as sf

from conftest import speech_like
from voxfoundry.audio import (
    AudioResult,
    analyze_audio,
    legacy_clean_clip,
    validate_audio,
    validate_reference,
)
from voxfoundry.errors import AudioValidationError, ReferenceAudioError, ReferenceAudioNotFoundError

pytestmark = pytest.mark.unit
SR = 24000


def tone(seconds=1.0, hz=220, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (0.1 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


# --- analysis / speech heuristic ---
def test_speech_like_signal_detected():
    s = analyze_audio(speech_like(2.0), SR)
    assert s.speech_likely and s.non_finite == 0 and s.duration_seconds == pytest.approx(2.0)


def test_pure_tone_not_speech():
    s = analyze_audio(tone(), SR)
    assert not s.speech_likely and s.tonal_concentration > 0.9


def test_silence_not_speech():
    s = analyze_audio(np.zeros(SR, dtype=np.float32), SR)
    assert not s.speech_likely and s.peak == 0.0


def test_analyze_does_not_modify_input():
    a = speech_like(1.0)
    before = a.copy()
    analyze_audio(a, SR)
    assert np.array_equal(a, before)


# --- validate_audio ---
@pytest.mark.parametrize(
    "audio,sr,msg",
    [
        ([0.1, 0.2], SR, "numpy"),
        (np.zeros((10, 2), dtype=np.float32), SR, "1-D"),
        (np.zeros(0, dtype=np.float32), SR, "empty"),
        (np.ones(10, dtype=np.int16), SR, "floating"),
        (np.ones(10, dtype=np.float32), 0, "sample rate"),
        (np.ones(10, dtype=np.float32), True, "sample rate"),
        (np.array([0.1, np.nan, 0.2], dtype=np.float32), SR, "NaN"),
        (np.array([0.1, np.inf], dtype=np.float32), SR, "NaN"),
        (np.zeros(SR, dtype=np.float32), SR, "silent"),
    ],
)
def test_validate_audio_rejects(audio, sr, msg):
    with pytest.raises(AudioValidationError, match=msg):
        validate_audio(audio, sr)


def test_validate_audio_require_speech_rejects_tone():
    validate_audio(tone(), SR)  # a tone is valid audio...
    with pytest.raises(AudioValidationError, match="speech"):
        validate_audio(tone(), SR, require_speech=True)  # ...but not speech


def test_audio_result_metrics():
    r = AudioResult(audio=np.zeros(48000, np.float32), sample_rate=SR, backend="x", device="cpu",
                    language="en", text="t", generation_seconds=1.0)
    assert r.duration_seconds == 2.0 and r.real_time_factor == 0.5 and r.channels == 1


# --- validate_reference ---
def write(path, data, sr=22050, **kw):
    sf.write(str(path), data, sr, **kw)
    return path


def test_reference_ok(reference_wav):
    info = validate_reference(reference_wav)
    assert info.duration_seconds == pytest.approx(6.0) and info.sample_rate == 22050 and info.channels == 1
    assert info.warnings == ()


def test_reference_file_unchanged(reference_wav):
    before = reference_wav.read_bytes()
    validate_reference(reference_wav)
    assert reference_wav.read_bytes() == before


def test_reference_missing_is_filenotfound(tmp_path):
    with pytest.raises(ReferenceAudioNotFoundError):
        validate_reference(tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError):  # backwards compatible
        validate_reference(tmp_path / "missing.wav")


def test_reference_directory_rejected(tmp_path):
    with pytest.raises(ReferenceAudioError, match="not a file"):
        validate_reference(tmp_path)


def test_reference_empty_file(tmp_path):
    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(ReferenceAudioError, match="empty"):
        validate_reference(p)


def test_reference_corrupt_file(tmp_path):
    p = tmp_path / "corrupt.wav"
    p.write_bytes(b"RIFF\x00\x00not really a wav file" * 10)
    with pytest.raises(ReferenceAudioError, match="Cannot decode"):
        validate_reference(p)


def test_reference_too_short(tmp_path):
    p = write(tmp_path / "short.wav", speech_like(3.0, 22050))
    with pytest.raises(ReferenceAudioError, match="too short"):
        validate_reference(p)


def test_reference_almost_silent(tmp_path):
    p = write(tmp_path / "quiet.wav", (speech_like(6.0, 22050) * 0.01).astype(np.float32))
    with pytest.raises(ReferenceAudioError, match="silent"):
        validate_reference(p)


def test_reference_non_finite(tmp_path):
    data = speech_like(6.0, 22050)
    data[100] = np.nan
    p = write(tmp_path / "nan.wav", data, subtype="FLOAT")
    with pytest.raises(ReferenceAudioError, match="NaN"):
        validate_reference(p)


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX non-root permissions")
def test_reference_unreadable(reference_wav):
    os.chmod(reference_wav, 0)
    try:
        with pytest.raises(ReferenceAudioError, match="not readable"):
            validate_reference(reference_wav)
    finally:
        os.chmod(reference_wav, 0o644)


def test_reference_long_warns(tmp_path):
    p = write(tmp_path / "long.wav", speech_like(40.0, 22050))
    info = validate_reference(p)
    assert info.warnings and "40.0s" in info.warnings[0]


# --- legacy cleanup (opt-in) ---
def test_legacy_clean_clip_matches_legacy_behaviour():
    sr = SR
    wav = np.concatenate([np.zeros(sr), speech_like(1.0) * 0.3, np.zeros(sr)]).astype(np.float32)
    original = wav.copy()
    out = legacy_clean_clip(wav, sr)
    assert np.array_equal(wav, original)  # input array is not modified in place
    assert len(out) < len(wav)  # leading/trailing silence trimmed
    assert out[0] == 0.0  # fade-in starts at zero
    rms = float(np.sqrt(np.mean(out**2)))
    assert np.max(np.abs(out)) <= 0.97 + 1e-6  # peak limit
    # Level-matched to TARGET_RMS 0.08 unless the 0.97 peak limiter had to pull it down.
    assert rms == pytest.approx(0.08, rel=0.05) or np.max(np.abs(out)) == pytest.approx(0.97, rel=1e-3)
