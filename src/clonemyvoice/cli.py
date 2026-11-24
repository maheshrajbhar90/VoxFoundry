"""Command-line interface for CloneMyVoice.

Sub-commands:

``generate``  synthesize a script file or a text string with a selected backend.
``master``    not part of Phase 2.5; exits with code 6.

Exit codes:

====  ==========================================================
0     success
1     unexpected internal error
2     invalid command-line usage (argparse)
3     invalid input: reference audio, script/text, output path
4     environment: missing dependency, model load, device unavailable
5     generation: unsupported language, inference or audio validation failure
6     command not implemented
====  ==========================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from .audio import analyze_audio, legacy_clean_clip, validate_audio, validate_reference
from .backend_adapter import DEFAULT_DEVICE, SUPPORTED_DEVICES, MockXTTSAdapter, XTTSAdapter
from .errors import (
    AdapterStateError,
    AudioValidationError,
    InferenceError,
    ModelDependencyError,
    ModelLoadError,
    ReferenceAudioError,
    TextValidationError,
    UnsupportedLanguageError,
)
from .text_processor import MAX_CHARS, build_segments, parse_script, parse_script_text, silence

logger = logging.getLogger("clonemyvoice")

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_INPUT = 3
EXIT_ENVIRONMENT = 4
EXIT_GENERATION = 5
EXIT_NOT_IMPLEMENTED = 6

ENGINES = ("xtts", "mock")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clonemyvoice", description="CloneMyVoice voice-generation CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Generate speech from a script file or text")
    src = gen.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", help="Script file ([EN]/[HI] marker lines, blank line = paragraph)")
    src.add_argument("--text", help="Text to speak (same format as a script)")
    gen.add_argument("--output", required=True, help="Output WAV file (PCM 16-bit)")
    gen.add_argument("--reference-audio", required=True, help="Authorized reference voice recording (WAV/FLAC)")
    gen.add_argument("--language", default="en", help="Default language code (default: en)")
    gen.add_argument("--engine", choices=ENGINES, default="xtts", help="Backend (default: xtts; 'mock' = test tone)")
    gen.add_argument("--backend", choices=ENGINES, help="Deprecated alias for --engine; overrides it when given")
    gen.add_argument("--device", choices=SUPPORTED_DEVICES, default=DEFAULT_DEVICE,
                     help=f"Inference device (default: {DEFAULT_DEVICE}); never switched automatically")
    gen.add_argument("--max-chars", type=int, default=None,
                     help=f"Chunk limit for every language (default per language: {MAX_CHARS})")
    gen.add_argument("--overwrite", action="store_true", help="Replace --output if it already exists")
    gen.add_argument("--legacy-clip-cleanup", action="store_true",
                     help="Apply clone_voice.py per-clip trim/fade/level-match (modifies audio; off by default)")
    gen.add_argument("--cache-dir", help="Opt-in cache for raw segment audio (off by default)")
    gen.add_argument("--allow-download", action="store_true",
                     help="Allow the model checkpoint to be downloaded if not cached (off by default)")
    gen.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    master = sub.add_parser("master", help="(not implemented in Phase 2.5)")
    master.add_argument("--input", required=True)
    master.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``clonemyvoice`` console script."""
    args = _build_parser().parse_args(argv)
    if args.cmd == "generate":
        return _cmd_generate(args)
    if args.cmd == "master":
        return _cmd_master(args)
    return EXIT_INTERNAL  # pragma: no cover - argparse enforces a sub-command


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def _fail(code: int, message: str) -> int:
    logger.error(message)
    return code


def _cmd_generate(args: argparse.Namespace) -> int:
    _configure_logging(args.verbose)
    engine = args.backend or args.engine
    if args.backend:
        logger.warning("--backend is deprecated; use --engine")

    # ---- inputs (no model work yet) --------------------------------------------------
    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        return _fail(EXIT_INPUT, f"Output exists: {out_path} (pass --overwrite to replace it)")
    if out_path.exists() and out_path.is_dir():
        return _fail(EXIT_INPUT, f"Output path is a directory: {out_path}")
    if args.max_chars is not None and args.max_chars < 20:
        return _fail(EXIT_INPUT, "--max-chars must be at least 20")

    try:
        ref_info = validate_reference(args.reference_audio)
        if args.script:
            items = parse_script(args.script, default_language=args.language)
        else:
            items = parse_script_text(args.text, default_language=args.language)
        limits = None if args.max_chars is None else {k: args.max_chars for k in ("en", "hi", args.language)}
        segments = build_segments(items, limits)
    except ReferenceAudioError as exc:
        return _fail(EXIT_INPUT, f"Reference audio: {exc}")
    except FileNotFoundError as exc:
        return _fail(EXIT_INPUT, f"Script not found: {exc}")
    except (ValueError, UnicodeDecodeError) as exc:
        return _fail(EXIT_INPUT, f"Script: {exc}")

    # ---- backend ---------------------------------------------------------------------
    if engine == "mock":
        adapter = MockXTTSAdapter(device=args.device)
    else:
        adapter = XTTSAdapter(
            device=args.device,
            cache_dir=args.cache_dir,
            allow_download=args.allow_download,
        )
    kind = "MOCK (synthetic test tone, not speech)" if adapter.is_mock else "REAL model inference"
    logger.info("Backend: %s | %s | device=%s", adapter.name, kind, adapter.device)
    logger.info(
        "Reference: %s (%.1fs, %d Hz, %d ch)",
        ref_info.path, ref_info.duration_seconds, ref_info.sample_rate, ref_info.channels,
    )
    logger.info("Script: %d lines -> %d segments", len(items), len(segments))

    try:
        t_load = time.perf_counter()
        adapter.load(ref_info.path)
        load_seconds = time.perf_counter() - t_load
        sr = int(adapter.sample_rate)

        parts: list[np.ndarray] = []
        gen_seconds = 0.0
        speech_seconds = 0.0
        warnings = 0
        for seg in segments:
            result = adapter.synthesize_result(seg.text, seg.lang)
            if result.sample_rate != sr:
                raise AudioValidationError(f"Segment {seg.index}: sample rate {result.sample_rate} != {sr}")
            stats = validate_audio(np.asarray(result.audio), result.sample_rate)
            if not adapter.is_mock and not stats.speech_likely:
                logger.warning(
                    "Segment %d may not contain speech (active=%.2f, tonal=%.2f) - listen to it",
                    seg.index, stats.active_ratio, stats.tonal_concentration,
                )
            wav = np.asarray(result.audio, dtype=np.float32)
            if args.legacy_clip_cleanup:
                wav = legacy_clean_clip(wav, sr)
            parts.append(wav)
            parts.append(silence(seg.pause_after, sr))
            gen_seconds += result.generation_seconds
            speech_seconds += result.duration_seconds
            warnings += len(result.warnings)
            for w in result.warnings:
                logger.warning("Segment %d: %s", seg.index, w)
            logger.info(
                "[%d/%d] line %d.%d [%s] %.2fs audio in %.2fs (attempts=%d%s) %s",
                seg.index + 1, len(segments), seg.line, seg.chunk, seg.lang, result.duration_seconds,
                result.generation_seconds, result.attempts, ", cached" if result.from_cache else "", seg.text[:50],
            )

        final = np.concatenate(parts)
        final_stats = validate_audio(final, sr)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        tmp = out_path.with_name(f".{out_path.name}.partial")
        sf.write(str(tmp), final, sr, subtype="PCM_16", format="WAV")
        os.replace(tmp, out_path)
    except (ModelDependencyError, ModelLoadError) as exc:
        return _fail(EXIT_ENVIRONMENT, str(exc))
    except ReferenceAudioError as exc:
        return _fail(EXIT_INPUT, f"Reference audio: {exc}")
    except (UnsupportedLanguageError, TextValidationError, InferenceError, AudioValidationError,
            AdapterStateError) as exc:
        return _fail(EXIT_GENERATION, str(exc))
    except Exception:
        logger.exception("Unexpected error")
        return EXIT_INTERNAL
    finally:
        adapter.unload()

    rtf = gen_seconds / speech_seconds if speech_seconds else float("inf")
    logger.info("Output: %s", out_path)
    logger.info(
        "Duration %.2fs | %d Hz | peak %.3f | RMS %.1f dBFS | model load %.1fs | generation %.1fs | RTF %.2f | warnings %d",
        final_stats.duration_seconds, sr, final_stats.peak, final_stats.rms_dbfs, load_seconds, gen_seconds, rtf,
        warnings,
    )
    if adapter.is_mock:
        logger.warning("MOCK backend used: the output is a test tone, not real speech.")
    return EXIT_OK


def _cmd_master(args: argparse.Namespace) -> int:
    _configure_logging(False)
    return _fail(EXIT_NOT_IMPLEMENTED, "Mastering is not part of Phase 2.5 and is not implemented.")


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
