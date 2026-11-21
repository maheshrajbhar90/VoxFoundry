#!/usr/bin/env python3
"""
Voiceover cleanup and mastering for YouTube.

Pipeline:
  1) Conservative silence/gap trimming (keeps a natural pad around speech)
  2) Optional stationary-noise reduction (requires noisereduce)
  3) High-pass + gentle presence EQ
  4) Gentle compressor + subtle high-frequency de-esser
  5) Integrated loudness normalization and true-peak limiting
  6) Export 48 kHz / 24-bit WAV

Install:
    brew install ffmpeg
    python3 -m pip install -r requirements_voiceover.txt

Example:
    python3 voiceover_master.py input.m4a -o mastered.wav
    python3 voiceover_master.py input.m4a -o mastered.wav --trim-silence
    python3 voiceover_master.py input.m4a -o mastered.wav --noise-profile-seconds 0.5

Notes:
- Noise reduction works best when the beginning of the recording contains a short
  section of room tone/noise without speech. If there is no clean noise-only
  section, use --no-noise-reduction to avoid artifacts.
- This script does not generate, replace, pitch-shift, or time-stretch speech.
- Silence trimming is OFF by default to protect quiet words and breaths.
  Enable it only with --trim-silence, then review the complete output.
"""

from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
import tempfile
import re
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import resample_poly

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

try:
    import noisereduce as nr
except ImportError:
    nr = None


def load_audio(path: Path):
    """Load audio. Use FFmpeg for M4A/AAC and other formats not supported by libsndfile."""
    try:
        audio, sr = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as soundfile_error:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"Could not read {path.name} directly. M4A requires FFmpeg. "
                "Install it with: brew install ffmpeg"
            ) from soundfile_error

        # Decode to a temporary PCM WAV, which SoundFile can read reliably.
        with tempfile.TemporaryDirectory(prefix="voiceover_decode_") as temp_dir:
            decoded_wav = Path(temp_dir) / "decoded.wav"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(decoded_wav)
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0 or not decoded_wav.exists():
                detail = result.stderr.strip() or "Unknown FFmpeg decoding error."
                raise RuntimeError(f"FFmpeg could not decode the input audio: {detail}")
            audio, sr = sf.read(str(decoded_wav), always_2d=True, dtype="float32")

    if audio.size == 0:
        raise ValueError("The input audio file is empty.")
    return audio, int(sr)


def resample_audio(audio: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    if old_sr == new_sr:
        return audio
    from math import gcd
    g = gcd(old_sr, new_sr)
    return resample_poly(audio, new_sr // g, old_sr // g, axis=0).astype(np.float32)


def trim_long_silences(audio: np.ndarray, sr: int, threshold_db: float = -48.0,
                       min_silence: float = 0.65, keep_before: float = 0.12,
                       keep_after: float = 0.18, frame_ms: float = 20.0):
    """Remove only long low-energy regions, retaining padding at speech edges."""
    mono = np.mean(audio, axis=1)
    frame = max(1, int(sr * frame_ms / 1000))
    hop = frame
    n_frames = int(np.ceil(len(mono) / hop))
    rms = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        chunk = mono[i * hop:min((i + 1) * hop, len(mono))]
        if len(chunk):
            rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-12)

    db = 20 * np.log10(np.maximum(rms, 1e-8))
    active = db > threshold_db

    # Fill very short inactive holes so natural micro-pauses remain untouched.
    max_hole = int(0.12 * sr / hop)
    i = 0
    while i < len(active):
        if active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and not active[j]:
            j += 1
        if i > 0 and j < len(active) and (j - i) <= max_hole:
            active[i:j] = True
        i = j

    min_frames = max(1, int(min_silence * sr / hop))
    keep_pre = int(keep_before * sr)
    keep_post = int(keep_after * sr)

    pieces = []
    cursor = 0
    i = 0
    removed = 0
    while i < len(active):
        if active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and not active[j]:
            j += 1
        start = i * hop
        end = min(j * hop, len(audio))
        if (j - i) >= min_frames:
            left = max(cursor, start - keep_pre)
            right = min(len(audio), end + keep_post)
            if left > cursor:
                pieces.append(audio[cursor:left])
                removed += left - cursor
            cursor = max(cursor, right)
        i = j

    if cursor < len(audio):
        pieces.append(audio[cursor:])
    if not pieces:
        return audio, 0.0
    result = np.concatenate(pieces, axis=0)
    return result, removed / sr


def apply_filter(audio: np.ndarray, sr: int, kind: str, cutoff, order=2):
    sos = signal.butter(order, cutoff, btype=kind, fs=sr, output="sos")
    # Zero-phase filtering avoids phase shift; fallback for very short clips.
    try:
        return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, audio, axis=0).astype(np.float32)


def peaking_eq(audio: np.ndarray, sr: int, freq: float, gain_db: float,
               q: float = 0.8):
    """Second-order peaking EQ using RBJ biquad coefficients."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    a_coeff = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    b = np.asarray(b) / a_coeff[0]
    a_coeff = np.asarray(a_coeff) / a_coeff[0]
    return signal.lfilter(b, a_coeff, audio, axis=0).astype(np.float32)


def gentle_compressor(audio: np.ndarray, sr: int, threshold_db=-20.0,
                      ratio=2.0, attack_ms=12.0, release_ms=140.0,
                      makeup_db=1.0):
    """Simple linked-channel RMS compressor with smooth gain envelope."""
    mono = np.mean(audio, axis=1)
    window = max(1, int(sr * 0.010))
    kernel = np.ones(window, dtype=np.float32) / window
    power = np.convolve(mono * mono, kernel, mode="same")
    level_db = 10 * np.log10(np.maximum(power, 1e-12))

    over = level_db - threshold_db
    compressed_db = np.where(over > 0, threshold_db + over / ratio, level_db)
    gain_db = compressed_db - level_db

    attack = np.exp(-1.0 / max(1.0, sr * attack_ms / 1000.0))
    release = np.exp(-1.0 / max(1.0, sr * release_ms / 1000.0))
    smooth = np.empty_like(gain_db)
    prev = 0.0
    for i, target in enumerate(gain_db):
        coeff = attack if target < prev else release
        prev = coeff * prev + (1.0 - coeff) * target
        smooth[i] = prev

    gain = (10 ** ((smooth + makeup_db) / 20)).astype(np.float32)
    return (audio * gain[:, None]).astype(np.float32)


def subtle_deesser(audio: np.ndarray, sr: int, threshold_db=-30.0,
                   reduction_db=3.0):
    """Reduce only strong high-band bursts; intentionally conservative."""
    if sr < 12000:
        return audio
    high = apply_filter(audio, sr, "bandpass", [5000, min(10000, sr * 0.45)], 2)
    env = np.sqrt(np.mean(high * high, axis=1) + 1e-12)
    env_db = 20 * np.log10(np.maximum(env, 1e-8))
    amount = np.clip((env_db - threshold_db) / 10.0, 0.0, 1.0)
    attenuation = 10 ** (-(amount * reduction_db) / 20)
    # Apply attenuation to the sibilant band only, not the full voice.
    return (audio - high + high * attenuation[:, None]).astype(np.float32)


def reduce_noise(audio: np.ndarray, sr: int, profile_seconds: float):
    if nr is None:
        raise RuntimeError("noisereduce is not installed. Install requirements or use --no-noise-reduction.")
    n = int(profile_seconds * sr)
    if n < int(0.1 * sr):
        raise ValueError("Noise profile must be at least 0.1 seconds.")
    noise_clip = audio[:min(n, len(audio))]
    if len(noise_clip) < int(0.1 * sr):
        raise ValueError("Recording is too short for the selected noise profile.")
    channels = []
    for ch in range(audio.shape[1]):
        channels.append(nr.reduce_noise(
            y=audio[:, ch], y_noise=noise_clip[:, ch], sr=sr,
            prop_decrease=0.55, stationary=True, n_fft=1024, hop_length=256
        ))
    return np.stack(channels, axis=1).astype(np.float32)


def measure_loudness(audio: np.ndarray, sr: int):
    if pyln is None:
        raise RuntimeError("pyloudnorm is not installed. Install requirements_voiceover.txt.")
    meter = pyln.Meter(sr)
    # pyloudnorm accepts mono or multichannel arrays.
    return float(meter.integrated_loudness(audio))


def true_peak_db(audio: np.ndarray):
    # 4x oversampling for a more reliable inter-sample peak estimate.
    oversampled = resample_poly(audio, 4, 1, axis=0)
    peak = float(np.max(np.abs(oversampled))) if oversampled.size else 0.0
    return 20 * np.log10(max(peak, 1e-12))


def loudness_master(audio: np.ndarray, sr: int, target_lufs: float,
                    ceiling_dbtp: float):
    """Two-pass EBU R128 loudness normalization with true-peak limiting."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required for loudness mastering. Install with: brew install ffmpeg")

    with tempfile.TemporaryDirectory(prefix="voiceover_master_") as temp_dir:
        temp_dir = Path(temp_dir)
        source_wav = temp_dir / "pre_master.wav"
        mastered_wav = temp_dir / "mastered.wav"
        sf.write(str(source_wav), audio, sr, subtype="FLOAT")

        target_lra = 7.0
        first_filter = (
            f"loudnorm=I={target_lufs}:TP={ceiling_dbtp}:"
            f"LRA={target_lra}:print_format=json"
        )
        first_cmd = [
            ffmpeg, "-hide_banner", "-nostats", "-i", str(source_wav),
            "-af", first_filter, "-f", "null", "-"
        ]
        first = subprocess.run(first_cmd, capture_output=True, text=True)
        if first.returncode != 0:
            raise RuntimeError("FFmpeg loudness analysis failed: " + first.stderr[-1500:])

        matches = re.findall(r'\{\s*"input_i".*?\}', first.stderr, flags=re.S)
        if not matches:
            raise RuntimeError("Could not parse FFmpeg loudness measurements: " + first.stderr[-1500:])
        measured = json.loads(matches[-1])

        second_filter = (
            f"loudnorm=I={target_lufs}:TP={ceiling_dbtp}:LRA={target_lra}:"
            f"measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:linear=true:print_format=summary"
        )
        second_cmd = [
            ffmpeg, "-hide_banner", "-y", "-i", str(source_wav),
            "-af", second_filter, "-ar", str(sr),
            "-c:a", "pcm_f32le", str(mastered_wav)
        ]
        second = subprocess.run(second_cmd, capture_output=True, text=True)
        if second.returncode != 0 or not mastered_wav.exists():
            raise RuntimeError("FFmpeg loudness mastering failed: " + second.stderr[-1500:])

        mastered, output_sr = sf.read(str(mastered_wav), always_2d=True, dtype="float32")
        if output_sr != sr:
            raise RuntimeError(f"Unexpected mastered sample rate: {output_sr}")
        return mastered.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Clean and master a spoken-word recording.")
    parser.add_argument("input", type=Path, help="Input audio file (M4A, WAV, MP3, etc.; FFmpeg is used to decode M4A/AAC).")
    parser.add_argument("-o", "--output", type=Path, default=Path("voiceover_mastered.wav"))
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--true-peak", type=float, default=-1.0)
    parser.add_argument("--trim-silence", action="store_true",
                        help="Enable conservative silence trimming. Off by default to protect quiet speech.")
    parser.add_argument("--silence-threshold", type=float, default=-60.0,
                        help="RMS threshold in dBFS; default -60 is deliberately low to protect quiet syllables.")
    parser.add_argument("--min-silence", type=float, default=1.20,
                        help="Only shorten gaps at least this many seconds long; default 1.20.")
    parser.add_argument("--noise-profile-seconds", type=float, default=0.0,
                        help="Seconds at the very start containing noise only (no speech). 0 disables noise reduction.")
    parser.add_argument("--no-noise-reduction", action="store_true")
    
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    audio, sr = load_audio(args.input)
    original_duration = len(audio) / sr

    # Process at the source sample rate, then export at 48 kHz.
    # Do not trim by default: low-volume consonants, breaths, and sentence
    # endings can resemble silence. Enable only when you have checked the source.
    removed = 0.0
    if args.trim_silence:
        audio, removed = trim_long_silences(
            audio, sr, threshold_db=args.silence_threshold,
            min_silence=args.min_silence,
            keep_before=0.30, keep_after=0.35, frame_ms=10.0
        )

    if args.noise_profile_seconds > 0 and not args.no_noise_reduction:
        audio = reduce_noise(audio, sr, args.noise_profile_seconds)

    # Conservative speech EQ: rumble cut, slight warmth/mud reduction,
    # and a modest presence lift.
    audio = apply_filter(audio, sr, "highpass", 75, 2)
    audio = peaking_eq(audio, sr, 250, -1.2, q=0.9)
    audio = peaking_eq(audio, sr, 3200, 1.4, q=0.8)

    audio = gentle_compressor(audio, sr)
    audio = subtle_deesser(audio, sr)

    audio = resample_audio(audio, sr, 48000)
    sr_out = 48000
    audio = loudness_master(audio, sr_out, args.target_lufs, args.true_peak)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, sr_out, subtype="PCM_24")

    # Measure the file that was actually exported, not the pre-quantized
    # in-memory signal. This verifies the delivered WAV's real format/levels.
    exported_audio, exported_sr = sf.read(
        str(args.output), always_2d=True, dtype="float32"
    )
    export_info = sf.info(str(args.output))
    if exported_sr != 48000 or export_info.subtype != "PCM_24":
        raise RuntimeError(
            "Export verification failed: expected 48 kHz / 24-bit PCM, "
            f"got {export_info.samplerate} Hz / {export_info.subtype}."
        )

    final_lufs = measure_loudness(exported_audio, exported_sr)
    final_peak = true_peak_db(exported_audio)
    final_duration = len(exported_audio) / exported_sr

    print("\nProcessing complete")
    print(f"Export format  : {export_info.samplerate} Hz / {export_info.subtype}")
    print(f"Input duration : {original_duration:.2f} s")
    print(f"Output duration: {final_duration:.2f} s")
    print(f"Long silence shortened (approx.): {removed:.2f} s")
    print(f"Integrated LUFS: {final_lufs:.2f}")
    print(f"True peak      : {final_peak:.2f} dBTP")
    print(f"Output         : {args.output.resolve()}")
    print("\nListen through the complete export before using it. Check quiet words, breaths,")
    print("sentence endings, and the beginning/end for accidental cuts.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
