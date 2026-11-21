"""CLI integration tests.

MOCKED: these tests never run real inference. They use either the
:class:`MockXTTSAdapter` (synthetic tone) or the real :class:`XTTSAdapter`
driven by a ``FakeTTS`` model from tests/conftest.py.
"""

import functools
import logging

import numpy as np
import pytest
import soundfile as sf

import voxfoundry.backend_adapter as ba
import voxfoundry.cli as cli
from conftest import FakeTTS
from voxfoundry.backend_adapter import MockXTTSAdapter

pytestmark = pytest.mark.integration


def run_generate(tmp_path, script_content, ref_path, output_path, *extra):
    script_file = tmp_path / "script.txt"
    script_file.write_text(script_content, encoding="utf-8")
    argv = [
        "generate", "--engine", "xtts", "--device", "cpu",
        "--script", str(script_file), "--reference-audio", str(ref_path), "--output", str(output_path), *extra,
    ]
    return cli.main(argv)


@pytest.fixture
def fake_xtts(monkeypatch, fake_factory):
    """Make the CLI build the REAL XTTSAdapter around a FAKE model."""
    monkeypatch.setattr(cli, "XTTSAdapter", functools.partial(ba.XTTSAdapter, tts_factory=fake_factory()))


# --- original tests (reference is now a 6 s recording to satisfy the legacy minimum) ---
def test_cli_success(tmp_path, monkeypatch, reference_wav):
    monkeypatch.setattr(cli, "XTTSAdapter", MockXTTSAdapter, raising=False)
    output_file = tmp_path / "out.wav"
    assert run_generate(tmp_path, "Hello", reference_wav, output_file) == 0
    data, sr = sf.read(str(output_file))
    assert sr == 24000 and data.size > 0 and np.any(data != 0)


def test_cli_output_dir_creation(tmp_path, monkeypatch, reference_wav):
    output_file = tmp_path / "sub" / "dir" / "out.wav"
    monkeypatch.setattr(cli, "XTTSAdapter", MockXTTSAdapter, raising=False)
    assert run_generate(tmp_path, "Hi", reference_wav, output_file) == 0
    assert output_file.exists()


def test_cli_invalid_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "XTTSAdapter", MockXTTSAdapter, raising=False)
    assert run_generate(tmp_path, "Hello", tmp_path / "nonexistent.wav", tmp_path / "out.wav") == cli.EXIT_INPUT


def test_cli_silent_output(tmp_path, monkeypatch, reference_wav):
    class SilentAdapter(MockXTTSAdapter):
        def generate(self, text, language=None):  # the primitive every result path uses
            return np.zeros(24000, dtype=np.float32)

    monkeypatch.setattr(cli, "XTTSAdapter", SilentAdapter, raising=False)
    output_file = tmp_path / "out.wav"
    assert run_generate(tmp_path, "Hello", reference_wav, output_file) == cli.EXIT_GENERATION
    assert not output_file.exists()


# --- new behaviour ---
def test_refuses_to_overwrite_existing_output(tmp_path, fake_xtts, reference_wav):
    out = tmp_path / "out.wav"
    out.write_bytes(b"precious")
    assert run_generate(tmp_path, "Hello there, this is a test.", reference_wav, out) == cli.EXIT_INPUT
    assert out.read_bytes() == b"precious"
    assert not FakeTTS.instances  # failed before loading any model


def test_overwrite_flag_replaces_output(tmp_path, fake_xtts, reference_wav):
    out = tmp_path / "out.wav"
    out.write_bytes(b"old")
    assert run_generate(tmp_path, "Hello there, this is a test.", reference_wav, out, "--overwrite") == 0
    assert sf.info(str(out)).samplerate == 24000
    assert not list(tmp_path.glob(".*.partial"))


def test_text_argument(tmp_path, fake_xtts, reference_wav):
    out = tmp_path / "t.wav"
    rc = cli.main(["generate", "--device", "cpu", "--text", "This is a real speech generation test.",
                   "--reference-audio", str(reference_wav), "--output", str(out)])
    assert rc == 0 and out.exists()
    assert FakeTTS.instances[0].calls[0]["text"] == "This is a real speech generation test."


def test_backend_alias_is_honoured(tmp_path, monkeypatch, reference_wav, fake_factory, caplog):
    monkeypatch.setattr(cli, "XTTSAdapter", functools.partial(ba.XTTSAdapter, tts_factory=fake_factory()))
    out = tmp_path / "o.wav"
    with caplog.at_level(logging.INFO, logger="voxfoundry"):
        rc = run_generate(tmp_path, "Hello", reference_wav, out, "--backend", "mock")
    assert rc == 0 and not FakeTTS.instances
    assert "MOCK" in caplog.text


def test_real_backend_is_labelled_real(tmp_path, fake_xtts, reference_wav, caplog):
    with caplog.at_level(logging.INFO, logger="voxfoundry"):
        run_generate(tmp_path, "Hello there, this is a test.", reference_wav, tmp_path / "o.wav")
    assert "REAL model inference" in caplog.text and "device=cpu" in caplog.text
    assert "RTF" in caplog.text


def test_long_line_is_chunked_in_order(tmp_path, fake_xtts, reference_wav):
    line = " ".join(f"Sentence number {i} is right here." for i in range(12))  # ~400 chars
    assert run_generate(tmp_path, line, reference_wav, tmp_path / "o.wav") == 0
    texts = [c["text"] for c in FakeTTS.instances[0].calls]
    assert len(texts) > 1 and all(len(t) <= 200 for t in texts)
    assert " ".join(texts) == line


def test_pauses_inserted_between_segments(tmp_path, monkeypatch, reference_wav):
    class Fixed(MockXTTSAdapter):
        def generate(self, text, language=None):
            return np.full(2400, 0.2, dtype=np.float32)  # 0.1 s

    monkeypatch.setattr(cli, "XTTSAdapter", Fixed)
    out = tmp_path / "o.wav"
    script = "[EN]\nThis first line ends with a question?\nThis second line is a statement.\n"
    assert run_generate(tmp_path, script, reference_wav, out) == 0
    data, sr = sf.read(str(out))
    # 0.1 + 0.45 (question) + 0.1 + 0.25 (period)
    assert len(data) / sr == pytest.approx(0.9, abs=1e-3)


def test_unsupported_language_exit_code(tmp_path, fake_xtts, reference_wav):
    rc = run_generate(tmp_path, "Hello there, this is a test.", reference_wav, tmp_path / "o.wav", "--language", "xx")
    assert rc == cli.EXIT_GENERATION and not (tmp_path / "o.wav").exists()


def test_model_load_failure_exit_code(tmp_path, monkeypatch, reference_wav):
    def broken(model_name, progress_bar):
        raise RuntimeError("no checkpoint")

    monkeypatch.setattr(cli, "XTTSAdapter", functools.partial(ba.XTTSAdapter, tts_factory=broken))
    assert run_generate(tmp_path, "Hello there.", reference_wav, tmp_path / "o.wav") == cli.EXIT_ENVIRONMENT


def test_legacy_clip_cleanup_is_opt_in(tmp_path, fake_xtts, reference_wav):
    text = "Hello there, this is a cleanup comparison test."
    run_generate(tmp_path, text, reference_wav, tmp_path / "raw.wav")
    run_generate(tmp_path, text, reference_wav, tmp_path / "clean.wav", "--legacy-clip-cleanup")
    raw, _ = sf.read(str(tmp_path / "raw.wav"))
    clean, _ = sf.read(str(tmp_path / "clean.wav"))
    assert not np.array_equal(raw, clean)


def test_unknown_marker_is_input_error(tmp_path, fake_xtts, reference_wav):
    assert run_generate(tmp_path, "[HOOK]\nText.\n", reference_wav, tmp_path / "o.wav") == cli.EXIT_INPUT


def test_master_not_implemented(tmp_path):
    assert cli.main(["master", "--input", "a.wav", "--output", "b.wav"]) == cli.EXIT_NOT_IMPLEMENTED
