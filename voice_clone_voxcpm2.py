"""
VoxCPM2 voice-cloning script
Uses the same project files as the original Chatterbox script:
  references/ref3_10s.wav
  scripts/script.txt

Run from the activated .venv-multiengine environment:
  python generate_voxcpm2.py
"""

import hashlib
import random
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from voxcpm import VoxCPM


# ============================================================
# 1. PATHS — intentionally unchanged from the Chatterbox file
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

REFERENCE = BASE_DIR / "references" / "ref3_10s.wav"
SCRIPT = BASE_DIR / "scripts" / "script.txt"

OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "cache_voxcpm2"

RAW_OUT = OUTPUT_DIR / "raw_voxcpm2.wav"
FINAL_OUT = OUTPUT_DIR / "final_voxcpm2.mp3"


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

MODEL_ID = "openbmb/VoxCPM2"

# "auto" selects CUDA, then Apple Silicon MPS, then CPU.
DEVICE = "cpu"
# STYLE = (
#     "Indian English male documentary narrator. "
#     "Confident, intelligent and conversational delivery. "
#     "Medium-slow pace with deliberate pauses between important ideas. "
#     "Use natural pitch variation and expressive emphasis on key facts, "
#     "statistics and surprising statements. "
#     "Build curiosity while explaining complex topics clearly. "
#     "Sound authoritative but approachable, like a premium business "
#     "documentary narrator. Avoid a robotic, overly dramatic or "
#     "news-anchor delivery. Maintain natural breathing and rhythm."
# )


# STYLES = {
#     "HOOK": (
#         "Intriguing documentary narration, deliberate pacing, "
#         "strong emphasis on surprising facts, controlled intensity."
#     ),

#     "EXPLANATION": (
#         "Conversational business documentary narration, "
#         "medium pace, clear articulation, natural rhythm."
#     ),

#     "DATA": (
#         "Analytical and authoritative delivery, measured pace, "
#         "precise pronunciation of financial figures."
#     ),

#     "WARNING": (
#         "Serious and restrained delivery, slightly slower pace, "
#         "subtle tension, meaningful pauses."
#     ),

#     "CONCLUSION": (
#         "Confident and decisive delivery, controlled energy, "
#         "strong emphasis on the final takeaway."
#     )
# }

STYLES = {
    "HOOK": "curious, punchy, slightly faster, rising intonation, storyteller energy",
    "EXPLANATION": "friendly, conversational, clear, natural rhythm",
    "DATA": "calm, precise, emphasis on numbers",
    "WARNING": "lower, slower, serious, measured",
    "CONCLUSION": "confident, warm, decisive, slow final line",
}

# Recommended conservative settings for Apple Silicon.
LOAD_DENOISER = False
OPTIMIZE = False

# VoxCPM2 defaults are CFG 2.0 and 10 diffusion steps.
CFG_VALUE = 1.8
INFERENCE_TIMESTEPS = 10

# Keep chunks reasonably short for stable long-form narration.
MAX_CHARS = 350
MIN_LINE_CHARS = 30
MAX_RETRIES = 2

# False preserves numbers, tickers and financial notation exactly.
# Set True if you want VoxCPM2's text normalizer to expand numbers.
NORMALIZE_TEXT = False

# Generate only the first N paragraphs for a quick test.
# Use None to generate the complete script.
PREVIEW_LINES = None


# ============================================================
# 3. PAUSES
# ============================================================

PAUSE_INSIDE_LINE = 0.30
PAUSE_EXCLAIM = 0.40
PAUSE_OTHER = 0.30
# PAUSE_PERIOD = 0.40
# PAUSE_QUESTION = 0.55
# PAUSE_ELLIPSIS = 0.70
# PAUSE_PARAGRAPH = 0.90


PAUSE_PERIOD = 0.35
PAUSE_QUESTION = 0.65     # let the question land
PAUSE_ELLIPSIS = 0.85
PAUSE_PARAGRAPH = 1.0


# ============================================================
# 4. AUDIO POLISH
# ============================================================

TARGET_RMS = 0.08
MIN_REFERENCE_SECONDS = 5
CHARS_PER_SEC = 14.0

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. VALIDATION
# ============================================================

def validate_files():
    if not REFERENCE.is_file():
        raise FileNotFoundError(f"Voice reference not found: {REFERENCE}")

    if not SCRIPT.is_file():
        raise FileNotFoundError(f"Script file not found: {SCRIPT}")

    if REFERENCE.stat().st_size == 0:
        raise ValueError("Voice reference file is empty.")

    if SCRIPT.stat().st_size == 0:
        raise ValueError("Script file is empty.")

    try:
        info = sf.info(str(REFERENCE))
        data, _ = sf.read(str(REFERENCE), dtype="float32")
    except Exception as exc:
        raise ValueError(
            f"Cannot read reference audio: {exc}\n"
            "Convert it with:\n"
            "ffmpeg -i input.m4a -ar 24000 -ac 1 "
            "references/ref3_10s.wav"
        ) from exc

    if info.frames <= 0 or data.size == 0:
        raise ValueError("Reference audio contains no frames.")

    if info.duration < MIN_REFERENCE_SECONDS:
        raise ValueError(
            f"Reference audio is too short: {info.duration:.2f}s. "
            f"Use at least {MIN_REFERENCE_SECONDS}s."
        )

    peak = float(np.max(np.abs(data)))
    if peak < 0.02:
        raise ValueError("Reference audio is almost silent.")

    print("\nREFERENCE AUDIO")
    print("-" * 45)
    print("File       :", REFERENCE.name)
    print("Duration   :", round(info.duration, 2), "s")
    print("Sample rate:", info.samplerate)
    print("Channels   :", info.channels)
    if info.duration > 35:
        print("Tip        : A clean 5–30 second reference is a practical range.")
    print("-" * 45)


# ============================================================
# 6. SCRIPT PARSER
#    Supports the existing [HI] and [EN] language markers,
#    plus style markers: [HOOK] [EXPLANATION] [DATA]
#    [WARNING] [CONCLUSION]. A style stays active until the
#    next style marker.
#    VoxCPM2 supports multilingual text, so no model switch is
#    needed when the language marker changes.
# ============================================================

def parse_script(path):
    content = Path(path).read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError("Script contains no text.")

    items = []
    language = "en"
    style = "EXPLANATION"

    for line_number, raw in enumerate(content.splitlines(), 1):
        text = raw.strip()

        marker = re.fullmatch(r"\[([A-Za-z]+)\]", text)
        if marker:
            tag = marker.group(1).upper()
            if tag in {"HI", "EN"}:
                language = tag.lower()
            elif tag in STYLES:
                style = tag
            else:
                raise ValueError(
                    f"Unsupported marker [{tag}] at line "
                    f"{line_number}. Use [HI], [EN] or one of "
                    f"{list(STYLES)}."
                )
            continue

        if not text:
            if items:
                items[-1]["para_break"] = True
            continue

        items.append({
            "lang": language,
            "style": style,
            "text": text,
            "para_break": False,
        })

    # Merge very short lines into the next line unless a blank line
    # indicates an intentional dramatic pause.
    merged = []
    i = 0

    while i < len(items):
        current = items[i]

        if (
            len(current["text"]) < MIN_LINE_CHARS
            and not current["para_break"]
            and i + 1 < len(items)
            and items[i + 1]["lang"] == current["lang"]
            and items[i + 1]["style"] == current["style"]
        ):
            following = items[i + 1]
            merged.append({
                "lang": current["lang"],
                "style": current["style"],
                "text": current["text"] + " " + following["text"],
                "para_break": following["para_break"],
            })
            i += 2
        else:
            merged.append(current)
            i += 1

    # Join lines within each paragraph to improve speech flow.
    paragraphs = []
    buffer = None

    for item in merged:
        if buffer is not None and (
            buffer["lang"] != item["lang"]
            or buffer["style"] != item["style"]
        ):
            buffer["para_break"] = True
            paragraphs.append(buffer)
            buffer = None

        if buffer is None:
            buffer = {
                "lang": item["lang"],
                "style": item["style"],
                "text": item["text"],
                "para_break": item["para_break"],
            }
        else:
            buffer["text"] += " " + item["text"]
            buffer["para_break"] = item["para_break"]

        if item["para_break"]:
            paragraphs.append(buffer)
            buffer = None

    if buffer:
        paragraphs.append(buffer)

    if not paragraphs:
        raise ValueError("No valid speech text found.")

    print(
        f"\nSCRIPT: {len(paragraphs)} paragraphs, "
        f"{sum(len(item['text']) for item in paragraphs)} characters"
    )
    return paragraphs


# ============================================================
# 7. TEXT CHUNKING
# ============================================================

def make_chunks(text, limit=MAX_CHARS):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?।])\s+", text.strip())
        if sentence.strip()
    ]

    chunks = []
    current = ""

    for sentence in sentences:
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""

            part = ""
            for word in sentence.split():
                candidate = (part + " " + word).strip()
                if len(candidate) <= limit:
                    part = candidate
                else:
                    if part:
                        chunks.append(part)
                    part = word

            if part:
                current = part
            continue

        candidate = (current + " " + sentence).strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def pause_after(item):
    text = item["text"].rstrip()

    if text.endswith(("...", "…")):
        pause = PAUSE_ELLIPSIS
    elif text.endswith("?"):
        pause = PAUSE_QUESTION
    elif text.endswith("!"):
        pause = PAUSE_EXCLAIM
    elif text.endswith((".", "।")):
        pause = PAUSE_PERIOD
    else:
        pause = PAUSE_OTHER

    if item["para_break"]:
        pause = max(pause, PAUSE_PARAGRAPH)

    return pause


def silence(seconds, sample_rate):
    n = int(sample_rate * seconds * random.uniform(0.85, 1.25))
    return (np.random.randn(n) * 0.0006).astype(np.float32)   # room tone, not zeros
    # return np.zeros(int(sample_rate * seconds), dtype=np.float32)


# ============================================================
# 8. CLIP CLEANUP
# ============================================================

def clean_clip(wav):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    if wav.size == 0:
        return wav

    peak = float(np.max(np.abs(wav)))
    if peak <= 0:
        return wav

    # Trim leading/trailing silence while retaining a small breath.
    indices = np.where(np.abs(wav) > 0.02 * peak)[0]
    if indices.size:
        keep = int(0.05 * SAMPLE_RATE)
        wav = wav[
            max(0, indices[0] - keep):
            min(len(wav), indices[-1] + keep)
        ].copy()

    # Short fades prevent clicks at clip joins.
    fade_samples = int(SAMPLE_RATE * 0.012)
    if len(wav) > 2 * fade_samples:
        ramp = np.linspace(0, 1, fade_samples, dtype=np.float32)
        wav[:fade_samples] *= ramp
        wav[-fade_samples:] *= ramp[::-1]

    # Conservative per-clip level matching.
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-9
    # wav *= float(np.clip(TARGET_RMS / rms, 0.5, 2.5))
    wav *= float(np.clip(TARGET_RMS / rms, 0.8, 1.25))

    peak = float(np.max(np.abs(wav)))
    if peak > 0.97:
        wav *= 0.97 / peak

    return wav


# ============================================================
# 9. MODEL + SYNTHESIS
# ============================================================

def load_model():
    print("\nLoading VoxCPM2...")
    print("Device setting:", DEVICE)
    print("First run may download several GB of model weights.")

    model = VoxCPM.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        load_denoiser=LOAD_DENOISER,
        optimize=OPTIMIZE,
    )

    print("VoxCPM2 loaded successfully.")
    return model

def generate_section(model, text, section_type="EXPLANATION"):
    style = STYLES.get(section_type, STYLES["EXPLANATION"])

    return model.generate(
        text=f"({style}) {text}",
        reference_wav_path=str(REFERENCE),
        cfg_value=CFG_VALUE,
        inference_timesteps=INFERENCE_TIMESTEPS,
        normalize=NORMALIZE_TEXT,
        denoise=False,
        retry_badcase=True,
        retry_badcase_max_times=2,
    )

def synth_once(model, text, style):
    # VoxCPM2's reference_wav_path mode clones the reference timbre.
    # It does not require a transcript of the reference recording.
    # wav = model.generate(
    #     text=text,
    #     reference_wav_path=str(REFERENCE),
    #     cfg_value=CFG_VALUE,
    #     inference_timesteps=INFERENCE_TIMESTEPS,
    #     normalize=NORMALIZE_TEXT,
    #     denoise=False,
    #     retry_badcase=True,
    #     retry_badcase_max_times=2,
    # )
#     wav = model.generate(
#     text=f"({STYLE}) {text}",
#     reference_wav_path=str(REFERENCE),
#     cfg_value=1.8,
#     inference_timesteps=10,
#     normalize=False,
#     denoise=False,
#     retry_badcase=True,
# )
    return generate_section(
        model=model,
        text=text,
        section_type=style
    )


def expected_seconds(text):
    return max(0.8, len(text) / CHARS_PER_SEC)


def cache_path(text, style):
    # The style description is part of the key, so editing a style
    # string in STYLES automatically invalidates old cached clips.
    style_text = STYLES.get(style, STYLES["EXPLANATION"])
    signature = (
        f"{MODEL_ID}|{style}|{style_text}|{text}|{CFG_VALUE}|"
        f"{INFERENCE_TIMESTEPS}|{NORMALIZE_TEXT}|"
        f"{REFERENCE.stat().st_mtime_ns}|{REFERENCE.stat().st_size}"
    )
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.npy"


def get_clip(model, text, style):
    cache_file = cache_path(text, style)

    if cache_file.exists():
        return np.load(cache_file), True

    expected = expected_seconds(text)
    best = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            wav = synth_once(model, text, style)
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)

            if wav.size == 0 or not np.all(np.isfinite(wav)):
                print(f"   Attempt {attempt + 1}: invalid audio; retrying.")
                continue

            wav = clean_clip(wav)
            duration = len(wav) / SAMPLE_RATE
            best = wav

            if 0.35 * expected <= duration <= 2.5 * expected + 2.0:
                break

            print(
                f"   Attempt {attempt + 1}: unusual duration "
                f"({duration:.1f}s; expected about {expected:.1f}s)."
            )

        except Exception:
            if attempt >= MAX_RETRIES:
                raise
            print(f"   Attempt {attempt + 1} failed; retrying.")
            traceback.print_exc()

    if best is None:
        raise RuntimeError(f"VoxCPM2 produced no usable audio for: {text}")

    np.save(cache_file, best)
    return best, False


def generate_speech(model, items):
    if PREVIEW_LINES:
        items = items[:PREVIEW_LINES]
        print(f"\nPREVIEW MODE: first {len(items)} paragraphs only")

    segments = []
    total = len(items)

    for number, item in enumerate(items, 1):
        chunks = make_chunks(item["text"])
        print(
            f"\n[{number}/{total}] [{item['lang'].upper()}] "
            f"[{item['style']}] {item['text'][:80]}"
        )

        for chunk_number, chunk in enumerate(chunks):
            try:
                wav, cached = get_clip(model, chunk, item["style"])
            except Exception as exc:
                print("\nSPEECH GENERATION FAILED")
                print("Paragraph:", number)
                print("Text:", chunk)
                raise RuntimeError(
                    f"VoxCPM2 failed on paragraph {number}: {exc}"
                ) from exc

            segments.append(wav)
            print(
                f"   {len(wav) / SAMPLE_RATE:.1f}s"
                f"{' (cached)' if cached else ''}"
            )

            is_last_chunk = chunk_number == len(chunks) - 1
            pause = (
                pause_after(item)
                if is_last_chunk
                else PAUSE_INSIDE_LINE
            )
            segments.append(silence(pause, SAMPLE_RATE))

    if not segments:
        raise RuntimeError("No audio was generated.")

    return np.concatenate(segments)


# ============================================================
# 10. SAVE + FINAL AUDIO POLISH
# ============================================================

def save_audio(audio):
    sf.write(str(RAW_OUT), audio, SAMPLE_RATE, subtype="PCM_16")

    print("\n" + "=" * 60)
    print("RAW AUDIO:", RAW_OUT)
    print("Duration :", round(len(audio) / SAMPLE_RATE, 1), "seconds")
    print("Sample rate:", SAMPLE_RATE)
    print("=" * 60)


def polish():
    if not shutil.which("ffmpeg"):
        print(
            "ffmpeg not found; skipping MP3 polish. "
            "Install with: brew install ffmpeg"
        )
        return

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(RAW_OUT),
            "-filter:a",
            "highpass=f=70,"
            "equalizer=f=180:t=q:w=1:g=1.5,"      # chest warmth
            "equalizer=f=3500:t=q:w=1:g=1.5,"     # presence
            "acompressor=threshold=-20dB:ratio=2:attack=10:release=120,"
            "loudnorm=I=-16:TP=-1.5:LRA=9",
            "-b:a", "192k",
            str(FINAL_OUT),
        ],
        check=True,
    )
    print("FINAL AUDIO:", FINAL_OUT)


# ============================================================
# 11. MAIN
# ============================================================

def main():
    global SAMPLE_RATE

    validate_files()

    items = parse_script(SCRIPT)
    model = load_model()

    # VoxCPM2 documentation uses model.tts_model.sample_rate for saving.
    SAMPLE_RATE = int(model.tts_model.sample_rate)
    print("Output sample rate:", SAMPLE_RATE)

    audio = generate_speech(model, items)
    save_audio(audio)
    polish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled. Completed clips are cached; rerun to resume.")
        sys.exit(130)
    except Exception as exc:
        print("\n" + "=" * 60)
        print("VOICE GENERATION FAILED")
        print("=" * 60)
        print("Error:", exc)
        traceback.print_exc()
        sys.exit(1)
