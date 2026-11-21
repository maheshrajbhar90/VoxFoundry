import os
import re
import sys
import shutil
import hashlib
import subprocess
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from TTS.api import TTS





# ============================================================
# 1. PATHS
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
MODEL      = "tts_models/multilingual/multi-dataset/xtts_v2"

REFERENCE  = BASE_DIR / "references" / "my_voice.wav"
SCRIPT     = BASE_DIR / "scripts" / "script.txt"

OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = OUTPUT_DIR / "cache"
RAW_OUT    = OUTPUT_DIR / "raw.wav"
FINAL_OUT  = OUTPUT_DIR / "xtts_final.mp3"

DEFAULT_LANGUAGE = "en"            # "hi" or "en" (used when no [HI]/[EN] marker)
SUPPORTED_LANGUAGES = {"HI": "hi", "EN": "en"}

# Set to a number (e.g. 10) to generate only the first N lines as a quick test.
PREVIEW_LINES = None

# ============================================================
# 2. VOICE SETTINGS  (the knobs that decide "natural")
# ============================================================
SPEED = 1.05                 # 1.0 normal | 1.05-1.15 natural-energetic | >1.25 robotic
TEMPERATURE = 0.7           # lower = steadier/flatter, higher = more expressive
TOP_P = 0.85
TOP_K = 50
REPETITION_PENALTY = 5.0     # too low (<3) causes stutters / repeated sounds
SEED = 42                    # same seed = repeatable result

GPT_COND_LEN = 30
GPT_COND_CHUNK_LEN = 6
MAX_REF_LEN = 30

MAX_CHARS = {"hi": 130, "en": 200}   # XTTS warns above ~150 (hi) / ~250 (en)
MIN_LINE_CHARS = 30                  # shorter lines get merged into the next one
MAX_RETRIES = 2                      # regenerate a chunk if it sounds broken

# ============================================================
# 3. PAUSES (seconds) - varied by punctuation like real speech
# ============================================================

PAUSE_INSIDE_LINE = 0.25      # between chunks of one long line
PAUSE_PERIOD      = 0.25
PAUSE_QUESTION    = 0.45
PAUSE_EXCLAIM     = 0.30
PAUSE_ELLIPSIS    = 0.60
PAUSE_OTHER       = 0.20
PAUSE_PARAGRAPH   = 0.70    # blank line in script = dramatic pause     

   

# ============================================================
# 4. OUTPUT POLISH
# ============================================================
TARGET_RMS = 0.08            # per-clip loudness matching
MIN_REFERENCE_SECONDS = 5

os.environ["COQUI_TOS_AGREED"] = "1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print(" XTTS-V2 NATURAL VOICE GENERATOR")
print("=" * 60)


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
            "Convert it with: ffmpeg -i my_voice.m4a -ss 0 -t 30 -ar 22050 -ac 1 my_voice_1.wav"
        ) from exc

    if info.frames <= 0 or data.size == 0:
        raise ValueError("Reference audio contains no frames.")
    if info.duration < MIN_REFERENCE_SECONDS:
        raise ValueError(f"Reference audio too short: {info.duration:.2f}s")

    peak = float(np.max(np.abs(data)))
    if peak < 0.02:
        raise ValueError("Reference audio is almost silent. Re-export it with a valid time range.")

    print("\nREFERENCE AUDIO")
    print("-" * 40)
    print("File       :", REFERENCE.name)
    print("Duration   :", round(info.duration, 2), "s")
    print("Sample rate:", info.samplerate)
    print("Channels   :", info.channels)
    if info.duration > 35:
        print("Tip        : clip is long; a clean 15-30s energetic take works best.")
    print("-" * 40)


# ============================================================
# 6. SCRIPT PARSER  (one line = one spoken line)
# ============================================================
def parse_script(path, default_language):
    content = Path(path).read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError("Script contains no text.")

    items, lang = [], default_language

    for n, raw in enumerate(content.splitlines(), 1):
        s = raw.strip()

        marker = re.fullmatch(r"\[([A-Za-z]+)\]", s)
        if marker:
            tag = marker.group(1).upper()
            if tag not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported marker [{tag}] at line {n}. Use [HI] or [EN].")
            lang = SUPPORTED_LANGUAGES[tag]
            continue

        if not s:
            if items:
                items[-1]["para_break"] = True
            continue

        items.append({"lang": lang, "text": s, "para_break": False})

    # Merge very short lines into the next one (XTTS is unstable on 1-2 word clips),
    # except when a blank line follows (that's an intentional standalone beat).
    merged, i = [], 0
    while i < len(items):
        cur = items[i]
        if (len(cur["text"]) < MIN_LINE_CHARS and not cur["para_break"]
                and i + 1 < len(items) and items[i + 1]["lang"] == cur["lang"]):
            nxt = items[i + 1]
            merged.append({
                "lang": cur["lang"],
                "text": cur["text"] + " " + nxt["text"],
                "para_break": nxt["para_break"],
            })
            i += 2
        else:
            merged.append(cur)
            i += 1

    if not merged:
        raise ValueError("No valid speech lines found.")

    print(f"\nSCRIPT: {len(merged)} lines, {sum(len(m['text']) for m in merged)} characters")
    return merged


# ============================================================
# 7. CHUNKING (only for lines longer than the limit)
# ============================================================
def make_chunks(text, limit):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s.strip()]
    chunks, cur = [], ""

    for s in sentences:
        if len(s) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            part = ""
            for w in s.split():
                cand = (part + " " + w).strip()
                if len(cand) <= limit:
                    part = cand
                else:
                    if part:
                        chunks.append(part)
                    part = w
            if part:
                cur = part
            continue

        cand = (cur + " " + s).strip()
        if len(cand) <= limit:
            cur = cand
        else:
            chunks.append(cur)
            cur = s

    if cur:
        chunks.append(cur)
    return chunks


def pause_after(item):
    t = item["text"].rstrip()
    if t.endswith(("...", "…")):
        p = PAUSE_ELLIPSIS
    elif t.endswith("?"):
        p = PAUSE_QUESTION
    elif t.endswith("!"):
        p = PAUSE_EXCLAIM
    elif t.endswith((".", "।")):
        p = PAUSE_PERIOD
    else:
        p = PAUSE_OTHER
    if item["para_break"]:
        p = max(p, PAUSE_PARAGRAPH)
    return p


def silence(seconds, sr):
    return np.zeros(int(sr * seconds), dtype=np.float32)


# ============================================================
# 8. CLIP CLEANUP (trim, fade, level-match)
# ============================================================
def clean_clip(wav, sr):
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak <= 0:
        return wav

    # trim leading/trailing silence, keep a little breathing room
    idx = np.where(np.abs(wav) > 0.02 * peak)[0]
    if idx.size:
        keep = int(0.05 * sr)
        wav = wav[max(0, idx[0] - keep): min(len(wav), idx[-1] + keep)].copy()

    # 12 ms fades remove clicks at joins
    n = int(sr * 0.012)
    if len(wav) > 2 * n:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        wav[:n] *= ramp
        wav[-n:] *= ramp[::-1]

    # match loudness so no line is suddenly louder/quieter
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-9
    wav = wav * float(np.clip(TARGET_RMS / rms, 0.5, 2.5))
    peak = float(np.max(np.abs(wav)))
    if peak > 0.97:
        wav *= 0.97 / peak
    return wav


# ============================================================
# 9. MODEL + SYNTHESIS
# ============================================================
def load_model():
    print("\nLoading XTTS-v2 model...")
    try:
        model = TTS(MODEL).to("cpu")
        print("XTTS-v2 loaded successfully.")
        return model
    except Exception:
        print("\nERROR: Could not load XTTS model.")
        traceback.print_exc()
        raise


_warned_fallback = False

def synth_once(tts, text, lang):
    global _warned_fallback
    try:
        return tts.tts(
            text=text,
            speaker_wav=str(REFERENCE),
            language=lang,
            speed=SPEED,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            gpt_cond_len=GPT_COND_LEN,
            gpt_cond_chunk_len=GPT_COND_CHUNK_LEN,
            max_ref_len=MAX_REF_LEN,
            sound_norm_refs=False,
        )
    except TypeError:
        if not _warned_fallback:
            print("\nNote: this TTS version rejected some tuning options; using basic settings.")
            _warned_fallback = True
        return tts.tts(text=text, speaker_wav=str(REFERENCE), language=lang, speed=SPEED)


def expected_seconds(text):
    return max(0.8, len(text) / (13.0 * SPEED))


def cache_path(text, lang):
    sig = (f"{text}|{lang}|{SPEED}|{TEMPERATURE}|{TOP_P}|{TOP_K}|{REPETITION_PENALTY}|"
           f"{SEED}|{REFERENCE.stat().st_mtime_ns}")
    return CACHE_DIR / (hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16] + ".npy")


def get_clip(tts, text, lang, sr):
    cp = cache_path(text, lang)
    if cp.exists():
        return np.load(cp), True

    exp = expected_seconds(text)
    best = None

    for attempt in range(MAX_RETRIES + 1):
        torch.manual_seed(SEED + attempt)
        wav = np.asarray(synth_once(tts, text, lang), dtype=np.float32).reshape(-1)

        if wav.size == 0 or not np.all(np.isfinite(wav)):
            print(f"   attempt {attempt + 1}: invalid audio, retrying")
            continue

        wav = clean_clip(wav, sr)
        dur = len(wav) / sr
        best = wav

        if 0.35 * exp <= dur <= 2.2 * exp + 1.5:
            break
        print(f"   attempt {attempt + 1}: odd duration ({dur:.1f}s vs ~{exp:.1f}s expected), retrying")

    if best is None:
        raise RuntimeError(f"XTTS produced no usable audio for: {text}")

    np.save(cp, best)
    return best, False


def generate_speech(tts, items, sr):
    if PREVIEW_LINES:
        items = items[:PREVIEW_LINES]
        print(f"\nPREVIEW MODE: first {len(items)} lines only")

    segments = []
    total = len(items)

    for n, item in enumerate(items, 1):
        lang = item["lang"]
        chunks = make_chunks(item["text"], MAX_CHARS[lang])
        print(f"\n[{n}/{total}] [{lang.upper()}] {item['text'][:70]}")

        for ci, chunk in enumerate(chunks):
            try:
                wav, cached = get_clip(tts, chunk, lang, sr)
            except Exception as exc:
                print("\nSPEECH GENERATION FAILED")
                print("Line :", n)
                print("Text :", chunk)
                raise RuntimeError(f"XTTS failed on line {n}") from exc

            segments.append(wav)
            print(f"   {len(wav) / sr:.1f}s{' (cached)' if cached else ''}")

            last = ci == len(chunks) - 1
            segments.append(silence(pause_after(item) if last else PAUSE_INSIDE_LINE, sr))

    if not segments:
        raise RuntimeError("No audio generated.")
    return np.concatenate(segments)


# ============================================================
# 10. SAVE + POLISH
# ============================================================
def save_audio(audio, sr):
    sf.write(str(RAW_OUT), audio, sr, subtype="PCM_16")
    print("\n" + "=" * 60)
    print("RAW AUDIO:", RAW_OUT)
    print("Duration :", round(len(audio) / sr, 1), "seconds")
    print("=" * 60)


def polish():
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; skipping final polish (install with: brew install ffmpeg)")
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(RAW_OUT), "-filter:a",
         "highpass=f=80,acompressor=threshold=-18dB:ratio=2.5:attack=5:release=80,"
         "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-b:a", "192k", str(FINAL_OUT)],
        check=True,
    )
    print("FINAL AUDIO:", FINAL_OUT)


# ============================================================
# 11. MAIN
# ============================================================
def main():
    validate_files()
    items = parse_script(SCRIPT, DEFAULT_LANGUAGE)
    tts = load_model()
    sr = getattr(tts.synthesizer, "output_sample_rate", 24000)
    print("Output sample rate:", sr)

    audio = generate_speech(tts, items, sr)
    save_audio(audio, sr)
    polish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled. Finished lines are cached, so rerunning resumes quickly.")
        sys.exit(130)
    except Exception as exc:
        print("\n" + "=" * 60)
        print("VOICE GENERATION FAILED")
        print("=" * 60)
        print("Error:", exc)
        sys.exit(1)




# import torchaudio as ta
# from chatterbox.tts import ChatterboxTTS

# model = ChatterboxTTS.from_pretrained(device="cpu")   # "mps" on Apple Silicon may work
# wav = model.generate(
#     "Five thousand seven hundred thirty-seven crore rupees. In one day.",
#     audio_prompt_path="references/my_voice.wav",
#     exaggeration=0.6,    # 0.5 neutral, higher = more dramatic
#     cfg_weight=0.4,      # lower = slower, more deliberate pacing
# )
# ta.save("test.wav", wav, model.sr)