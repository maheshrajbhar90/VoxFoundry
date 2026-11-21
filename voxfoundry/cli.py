"""Command‑line interface for VoxFoundry.

The real CLI is complex; this stub demonstrates how the package can be
invoked.  It currently supports two sub‑commands:

``generate`` – generate speech for a text file using a selected backend.
``master``   – placeholder for the audio‑mastering pipeline.

The implementation is intentionally lightweight – it delegates the heavy
lifting to the :class:`voxfoundry.text_processor.TextProcessor` and the
``BackendAdapter``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .backend_adapter import XTTSAdapter, MockXTTSAdapter
from .text_processor import parse_script, pause_after, silence
import numpy as np


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``voxfoundry`` console script."""

    parser = argparse.ArgumentParser(prog="voxfoundry", description="Voice‑cloning CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate audio from a script")
    gen.add_argument("--script", required=True, help="Path to the script file")
    gen.add_argument("--output", required=True, help="Output audio file")
    # Backwards compatibility: --backend still accepted as alias for --engine
    gen.add_argument("--engine", default="xtts", help="Backend to use (default: xtts)")
    gen.add_argument("--backend", help="Alias for --engine (deprecated)")
    gen.add_argument("--reference-audio", required=True, help="Path to reference voice wav file")
    gen.add_argument("--language", default="en", help="Default language code")

    master = sub.add_parser("master", help="Master an existing audio file")
    master.add_argument("--input", required=True, help="Input audio file")
    master.add_argument("--output", required=True, help="Output audio file")

    args = parser.parse_args(argv)
    if args.cmd == "generate":
        return _cmd_generate(args)
    if args.cmd == "master":
        return _cmd_master(args)
    return 0


def _cmd_generate(args) -> int:  # pragma: no cover - thin wrapper
    """Generate speech using the selected backend.

    The function is intentionally defensive: it validates inputs, ensures
    directories exist, and performs minimal audio sanity checks before
    writing the output.  Any exception results in a non‑zero exit code.
    """
    try:
        # Resolve backend name – support legacy ``--backend`` alias.
        engine = (args.engine or getattr(args, "backend", ""))
        if engine.lower() == "xtts":
            adapter = XTTSAdapter()
        else:
            raise NotImplementedError(f"Backend {engine!r} not implemented")

        # Validate reference audio
        ref_path = Path(args.reference_audio)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")
        try:
            import soundfile as sf
            sf_info = sf.info(str(ref_path))
            if sf_info.samplerate <= 0:
                raise ValueError("Reference audio has invalid sample rate")
        except Exception as exc:  # pragma: no cover – soundfile optional
            raise ValueError(f"Unable to read reference audio {ref_path}: {exc}") from exc

        adapter.load(ref_path)

        items = parse_script(args.script, default_language=args.language)
        sr = adapter.sample_rate
        audio_chunks = []
        for item in items:
            wav, _ = adapter.synthesize(item["text"], item["lang"])  # returns (array, sr)
            if not isinstance(wav, np.ndarray):
                raise TypeError("Adapter returned non‑NumPy array")
            if wav.size == 0:
                raise ValueError("Generated audio is empty")
            if not np.all(np.isfinite(wav)):
                raise ValueError("Generated audio contains non‑finite values")
            # Simple silence check: all samples < 1e-5
            if np.max(np.abs(wav)) < 1e-5:
                raise ValueError("Generated audio is silent")
            if wav.dtype != np.float32:
                wav = wav.astype(np.float32)
            audio_chunks.append(wav)
            audio_chunks.append(silence(pause_after(item), sr))

        final_audio = np.concatenate(audio_chunks)
        if final_audio.size == 0:
            raise RuntimeError("Final audio buffer is empty")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), final_audio, sr)
        return 0
    except Exception as exc:  # pragma: no cover – error reporting
        print(f"Error: {exc}")
        return 1


def _cmd_master(args) -> int:  # pragma: no cover - placeholder
    # In the real project this would perform audio mastering.
    print("Mastering not implemented in this stub.")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())