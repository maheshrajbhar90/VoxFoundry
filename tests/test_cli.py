"""CLI integration tests.

These tests exercise the command‑line entry point without invoking a full
process.  They validate argument parsing, error handling, and basic
functionality of the ``generate`` command.  The heavy TTS workload is
patched out by using :class:`voxfoundry.backend_adapter.MockXTTSAdapter`.
"""

import os
import pathlib
from pathlib import Path
import tempfile
import soundfile as sf
import numpy as np
import numpy as np
import voxfoundry.cli as cli
from voxfoundry.backend_adapter import MockXTTSAdapter


def run_generate(tmp_path, script_content, ref_path, output_path, **kwargs):
    """Helper to invoke the CLI with the supplied arguments.

    ``kwargs`` are additional command‑line arguments passed as ``--key value``.
    """
    # Create script file
    script_file = tmp_path / "script.txt"
    script_file.write_text(script_content)
    # Write dummy reference audio (sine wave)
    fr = 24000
    t = np.linspace(0, 1, fr, False)
    wav = 0.1 * np.sin(2 * np.pi * 220 * t)
    import soundfile as sf
    sf.write(str(ref_path), wav, fr)
    argv = [
        "generate",
        "--engine", "xtts",
        "--script", str(script_file),
        "--reference-audio", str(ref_path),
        "--output", str(output_path),
    ]
    for k, v in kwargs.items():
        argv += [f"--{k}", str(v)]
    return cli.main(argv)


def test_cli_success(tmp_path, monkeypatch):
    # Replace the real adapter with the mock within the CLI module
    monkeypatch.setattr(cli, "XTTSAdapter", MockXTTSAdapter, raising=False)
    output_file = tmp_path / "out.wav"
    exit_code = run_generate(tmp_path, "Hello", tmp_path / "voice.wav", output_file)
    assert exit_code == 0
    # Verify that output file exists and contains non‑empty audio
    import soundfile as sf
    data, sr = sf.read(str(output_file))
    assert sr == 24000
    assert data.size > 0
    assert np.any(data != 0)


def test_cli_output_dir_creation(tmp_path, monkeypatch):
    # Use a nested output directory that does not exist yet
    nested_dir = tmp_path / "sub" / "dir"
    output_file = nested_dir / "out.wav"
    monkeypatch.setattr(cli, "XTTSAdapter", MockXTTSAdapter, raising=False)
    exit_code = run_generate(tmp_path, "Hi", tmp_path / "voice.wav", output_file)
    assert exit_code == 0
    assert output_file.exists()


def test_cli_invalid_reference(tmp_path, monkeypatch):
    # Do NOT create a reference audio file – the CLI should detect the missing file.
    monkeypatch.setattr(cli, "XTTSAdapter", MockXTTSAdapter, raising=False)
    output_file = tmp_path / "out.wav"
    argv = [
        "generate",
        "--engine", "xtts",
        "--script", str(tmp_path / "script.txt"),
        "--reference-audio", str(tmp_path / "nonexistent.wav"),
        "--output", str(output_file),
    ]
    # Write a dummy script file so that parse_script does not fail.
    (tmp_path / "script.txt").write_text("Hello")
    exit_code = cli.main(argv)
    assert exit_code != 0


def test_cli_silent_output(tmp_path, monkeypatch):
    # Patch the mock adapter to return silence
    class SilentAdapter(MockXTTSAdapter):
        def synthesize(self, text: str, language: str | None = None):  # type: ignore[override]
            return np.zeros(24000), 24000
    monkeypatch.setattr(cli, "XTTSAdapter", SilentAdapter, raising=False)
    output_file = tmp_path / "out.wav"
    exit_code = run_generate(tmp_path, "Hello", tmp_path / "voice.wav", output_file)
    assert exit_code != 0
