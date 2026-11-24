"""Unit tests for :class:`clonemyvoice.backend_adapter.XTTSAdapter`.

A ``FakeTTS`` (tests/conftest.py) replaces ``TTS.api.TTS`` via the adapter's
``tts_factory`` parameter, so the real adapter code runs without a checkpoint.
These are NOT real-model tests; see tests/test_real_xtts.py for those.
"""

import os

import numpy as np
import pytest
import torch

import clonemyvoice.backend_adapter as ba
from conftest import FakeTTS, speech_like
from clonemyvoice.errors import (
    AdapterStateError,
    AudioValidationError,
    DeviceUnavailableError,
    InferenceError,
    ModelLoadError,
    ReferenceAudioError,
    TextValidationError,
    UnsupportedLanguageError,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def adapter(fake_factory):
    return ba.XTTSAdapter(device="cpu", tts_factory=fake_factory())


# --- original tests, updated to the corrected contract ---
def test_cache_is_opt_in_and_lazy(fake_factory, reference_wav, tmp_path):
    """Construction writes nothing; the output cache exists only when requested and used."""
    ba.XTTSAdapter(device="cpu", tts_factory=fake_factory())
    assert not (tmp_path / ".clonemyvoice_cache").exists()

    cache = tmp_path / "cache"
    a = ba.XTTSAdapter(device="cpu", cache_dir=cache, tts_factory=fake_factory())
    assert not cache.exists()
    a.load(reference_wav)
    a.generate("hello world, this is a cache test")
    assert len(list(cache.glob("*.npy"))) == 1


def test_load_and_generate_with_output_cache(fake_factory, reference_wav, tmp_path):
    a = ba.XTTSAdapter(device="cpu", cache_dir=tmp_path / "cache", tts_factory=fake_factory())
    a.load(reference_wav)
    audio1 = a.generate("hello world")
    r2 = a.synthesize_result("hello world")
    assert isinstance(audio1, np.ndarray) and audio1.size > 0
    assert r2.from_cache and a.inference_call_count == 1
    assert np.allclose(audio1, r2.audio)


def test_unload_and_reuse(adapter, reference_wav):
    adapter.load(reference_wav)
    adapter.generate("text one is here")
    adapter.generate("text two is here")
    assert adapter.model_load_count == 1  # model reused across calls
    assert len(FakeTTS.instances[0].calls) == 2
    adapter.unload()
    assert adapter.tts is None and adapter.state is ba.AdapterState.UNLOADED
    adapter.load(reference_wav)  # reload works
    adapter.generate("text three is here")
    assert adapter.model_load_count == 2
    assert len(FakeTTS.instances[-1].calls) == 1


def test_invalid_reference_path(adapter):
    with pytest.raises(FileNotFoundError):
        adapter.load("/does/not/exist.wav")
    assert adapter.tts is None  # reference is validated before the model is loaded


def test_invalid_generate_without_load(adapter):
    with pytest.raises(RuntimeError):
        adapter.generate("test")
    with pytest.raises(AdapterStateError):
        adapter.generate("test")


# --- A2: explicit device, no silent switching ---
def test_default_device_is_mps():
    assert ba.XTTSAdapter().device == "mps"


def test_unknown_device_rejected():
    with pytest.raises(ValueError):
        ba.XTTSAdapter(device="gpu")


def test_mps_unavailable_raises_instead_of_falling_back(fake_factory, reference_wav, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    a = ba.XTTSAdapter(device="mps", tts_factory=fake_factory())
    with pytest.raises(DeviceUnavailableError):
        a.load(reference_wav)
    assert a.tts is None and a.device == "mps" and not FakeTTS.instances


def test_model_moved_to_requested_device(fake_factory, reference_wav):
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory())
    a.load(reference_wav)
    assert FakeTTS.instances[0].device == "cpu"


# --- A3: no TypeError fallback ---
def test_type_error_is_not_swallowed(fake_factory, reference_wav):
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory(raises=TypeError("bad kwarg")))
    a.load(reference_wav)
    with pytest.raises(InferenceError, match="bad kwarg"):
        a.generate("some text to speak")
    assert len(FakeTTS.instances[0].calls) == 1  # no second, reduced-parameter call
    assert a.state is ba.AdapterState.READY


def test_legacy_parameters_passed_to_model(adapter, reference_wav):
    adapter.load(reference_wav)
    adapter.generate("check the parameters")
    call = FakeTTS.instances[0].calls[0]
    assert call["speaker_wav"] == str(reference_wav) and call["language"] == "en"
    assert {k: call[k] for k in ("speed", "temperature", "top_p", "top_k", "repetition_penalty",
                                 "gpt_cond_len", "gpt_cond_chunk_len", "max_ref_len", "sound_norm_refs")} == {
        "speed": 1.05, "temperature": 0.7, "top_p": 0.85, "top_k": 50, "repetition_penalty": 5.0,
        "gpt_cond_len": 30, "gpt_cond_chunk_len": 6, "max_ref_len": 30, "sound_norm_refs": False,
    }


# --- A4: raw output ---
def test_output_is_raw_model_output(fake_factory, reference_wav):
    raw = np.concatenate([np.zeros(12000), speech_like(1.5) * 0.9, np.zeros(12000)]).astype(np.float32)
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory(output=raw))
    a.load(reference_wav)
    out = a.generate("eighteen characters")
    assert np.array_equal(out, raw)  # no trim, fade or gain


# --- A5/A6: retries are reported, not silent ---
def test_retry_on_implausible_duration_uses_new_seed_and_warns(fake_factory, reference_wav, monkeypatch):
    seeds = []
    monkeypatch.setattr(ba, "_seed", lambda s: seeds.append(s))
    too_long = speech_like(30.0)
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory(output=too_long))
    a.load(reference_wav)
    r = a.synthesize_result("short text")
    assert seeds == [42, 43, 44] and r.attempts == 3
    assert r.metadata["duration_ok"] is False
    assert any("implausible duration" in w for w in r.warnings)


def test_non_finite_output_retried_then_fails(fake_factory, reference_wav):
    bad = np.full(24000, np.nan, dtype=np.float32)
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory(output=bad))
    a.load(reference_wav)
    with pytest.raises(InferenceError, match="no usable audio after 3 attempts"):
        a.generate("some text to speak")
    assert a.state is ba.AdapterState.READY


def test_silent_output_rejected(fake_factory, reference_wav):
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory(output=np.zeros(24000, np.float32)))
    a.load(reference_wav)
    with pytest.raises(AudioValidationError, match="silent"):
        a.generate("some text to speak")


def test_cache_key_changes_with_device_and_settings(fake_factory, reference_wav, tmp_path):
    keys = set()
    for device, settings in [("cpu", ba.XTTSSettings()), ("cpu", ba.XTTSSettings(temperature=0.5))]:
        a = ba.XTTSAdapter(device=device, settings=settings, cache_dir=tmp_path, tts_factory=fake_factory())
        a.load(reference_wav)
        keys.add(a._cache_path("same text", "en"))
    a.device = "mps"
    keys.add(a._cache_path("same text", "en"))
    assert len(keys) == 3


# --- A7/A14: lifecycle and cleanup ---
def test_unload_is_idempotent_and_releases_mps_cache(fake_factory, reference_wav, monkeypatch):
    released = []
    monkeypatch.setattr(ba, "_release_device_memory", lambda d: released.append(d))
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory())
    a.unload()  # before load: no error, nothing to release
    a.load(reference_wav)
    a.unload()
    a.unload()
    assert released == ["cpu"] and a.state is ba.AdapterState.UNLOADED


def test_release_device_memory_calls_mps_not_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    ba._release_device_memory("mps")
    assert calls == ["mps"]


def test_second_load_reuses_model_and_swaps_reference(adapter, reference_wav, tmp_path):
    import shutil

    other = tmp_path / "other.wav"
    shutil.copy(reference_wav, other)
    adapter.load(reference_wav)
    adapter.load(other)
    assert adapter.model_load_count == 1 and adapter.reference_audio == other


def test_failed_reference_on_reload_keeps_previous_state(adapter, reference_wav, tmp_path):
    adapter.load(reference_wav)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"junk")
    with pytest.raises(ReferenceAudioError):
        adapter.load(bad)
    assert adapter.is_ready and adapter.reference_audio == reference_wav


def test_model_load_failure_leaves_clean_state(fake_factory, reference_wav):
    def broken(model_name, progress_bar):
        raise RuntimeError("checkpoint missing")

    a = ba.XTTSAdapter(device="cpu", tts_factory=broken)
    with pytest.raises(ModelLoadError, match="downloads are disabled"):
        a.load(reference_wav)
    assert a.tts is None and a.state is ba.AdapterState.INITIALIZED and a.model_load_count == 0


# --- A10: language / text validation ---
def test_unsupported_language_rejected(adapter, reference_wav):
    adapter.load(reference_wav)
    with pytest.raises(UnsupportedLanguageError, match="'xx'"):
        adapter.generate("hello there", "xx")
    assert not FakeTTS.instances[0].calls


def test_languages_come_from_loaded_model(adapter, reference_wav):
    adapter.load(reference_wav)
    assert "hi" in adapter.supported_languages and "en" in adapter.supported_languages


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_rejected(adapter, reference_wav, text):
    adapter.load(reference_wav)
    with pytest.raises(TextValidationError):
        adapter.generate(text)


def test_text_over_char_limit_warns(adapter, reference_wav):
    adapter.load(reference_wav)
    r = adapter.synthesize_result("word " * 60)  # 300 chars > 250 for en
    assert any("char limit" in w for w in r.warnings)


# --- downloads ---
def test_load_forces_hf_offline_by_default(fake_factory, reference_wav, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    a = ba.XTTSAdapter(device="cpu", tts_factory=fake_factory())
    a.load(reference_wav)
    assert FakeTTS.instances[0].hf_offline_at_init == "1"
    assert "HF_HUB_OFFLINE" not in os.environ  # restored


def test_allow_download_leaves_hf_online(fake_factory, reference_wav, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    a = ba.XTTSAdapter(device="cpu", allow_download=True, tts_factory=fake_factory())
    a.load(reference_wav)
    assert FakeTTS.instances[0].hf_offline_at_init is None


# --- A16: result contract ---
def test_synthesize_result_contract(adapter, reference_wav):
    adapter.load(reference_wav)
    r = adapter.synthesize_result("contract check text", "en")
    assert r.backend == "XTTS-v2" and r.device == "cpu" and r.sample_rate == 24000 and not r.is_mock
    assert r.audio.dtype == np.float32 and r.audio.ndim == 1 and r.generation_seconds >= 0
    audio, sr = adapter.synthesize("contract check text", "en")  # legacy tuple form still works
    assert sr == 24000 and audio.ndim == 1


def test_mock_is_labelled():
    m = ba.MockXTTSAdapter()
    assert m.is_mock and not ba.XTTSAdapter.is_mock
    m.load("anything")
    assert m.synthesize_result("hi").is_mock
