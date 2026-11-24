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
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# ============================================================
# 1. PATHS
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent

REFERENCE  = BASE_DIR / "references" / "ref3_10s.wav"
SCRIPT     = BASE_DIR / "scripts" / "script.txt"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR  = OUTPUT_DIR / "cache_chatterbox"
RAW_OUT    = OUTPUT_DIR / "raw.wav"
FINAL_OUT  = OUTPUT_DIR / "final.mp3"

DEVICE = "cpu"                     # try "mps" later for speed on Apple Silicon

DEFAULT_LANGUAGE = "en"            # "en" (standard model) or "hi" (needs USE_MULTILINGUAL)
SUPPORTED_LANGUAGES = {"HI": "hi", "EN": "en"}

# Standard Chatterbox is English only. Set True to load the multilingual model
# (needed for [HI] lines). Check the Chatterbox README if the import fails.
USE_MULTILINGUAL = False

# Set to a number (e.g. 10) to generate only the first N lines as a quick test.
PREVIEW_LINES = None

# ============================================================
# 2. VOICE SETTINGS
# ============================================================
EXAGGERATION = 0.40          # 0.5 neutral | 0.6-0.75 more dramatic | >0.8 can get unstable
CFG_WEIGHT   = 0.40           # lower = slower, more deliberate | higher = faster
TEMPERATURE  = 0.70
MAX_CHARS    = {"hi": 130, "en": 200}
MIN_LINE_CHARS = 60
SEED         = 42            # same seed = repeatable result

CHARS_PER_SEC = 14.0         # only used to sanity-check clip length
MAX_CHARS = {"hi": 130, "en": 250}
MIN_LINE_CHARS = 30          # shorter lines get merged into the next one
MAX_RETRIES = 2              # regenerate a chunk if it sounds broken

# ============================================================
# 3. PAUSES (seconds) - varied by punctuation like real speech
# ============================================================
PAUSE_INSIDE_LINE = 0.30     # between chunks of one long line
PAUSE_PERIOD      = 0.40
PAUSE_QUESTION    = 0.55
PAUSE_EXCLAIM     = 0.40
PAUSE_ELLIPSIS    = 0.70
PAUSE_OTHER       = 0.30
PAUSE_PARAGRAPH   = 0.90     # blank line in script = dramatic pause

# ============================================================
# 4. OUTPUT POLISH
# ============================================================
TARGET_RMS = 0.08            # per-clip loudness matching
MIN_REFERENCE_SECONDS = 5

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print(" CHATTERBOX NATURAL VOICE GENERATOR")
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
            "Convert it with: ffmpeg -i my_voice.m4a -ss 0 -t 30 -ar 22050 -ac 1 references/my_voice.wav"
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
        print("Tip        : clip is long; a clean 10-30s natural take works best.")
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

    if not USE_MULTILINGUAL and any(it["lang"] != "en" for it in items):
        raise ValueError(
            "Script has Hindi ([HI]) lines but USE_MULTILINGUAL is False. "
            "Standard Chatterbox is English only."
        )

    # Merge very short lines into the next one (very short clips sound clipped),
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

    # PARAGRAPH MODE: join lines within a paragraph so intonation flows naturally.
    # A block closes at a blank line (para_break) or when the language changes.
    paragraphs, buf = [], None
    for it in merged:
        if buf is not None and buf["lang"] != it["lang"]:
            buf["para_break"] = True
            paragraphs.append(buf)
            buf = None

        if buf is None:
            buf = {"lang": it["lang"], "text": it["text"], "para_break": it["para_break"]}
        else:
            buf["text"] += " " + it["text"]
            buf["para_break"] = it["para_break"]

        if it["para_break"]:
            paragraphs.append(buf)
            buf = None

    if buf:
        paragraphs.append(buf)
    merged = paragraphs

    if not merged:
        raise ValueError("No valid speech lines found.")

    print(f"\nSCRIPT: {len(merged)} paragraphs, {sum(len(m['text']) for m in merged)} characters")

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
            if cur:
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
    print("\nLoading Chatterbox model (first run downloads the weights)...")
    try:
        # Makes checkpoints load correctly on CPU / Apple Silicon.
        original_load = torch.load
        map_location = torch.device(DEVICE)

        def patched_load(*args, **kwargs):
            kwargs.setdefault("map_location", map_location)
            return original_load(*args, **kwargs)

        torch.load = patched_load

        if USE_MULTILINGUAL:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=DEVICE)

        print("Chatterbox loaded successfully.")
        return model
    except Exception:
        print("\nERROR: Could not load Chatterbox model.")
        traceback.print_exc()
        raise


def synth_once(model, text, lang):
    kwargs = dict(
        audio_prompt_path=str(REFERENCE),
        exaggeration=EXAGGERATION,
        cfg_weight=CFG_WEIGHT,
        temperature=TEMPERATURE,
    )
    if USE_MULTILINGUAL:
        kwargs["language_id"] = lang
    wav = model.generate(text, **kwargs)
    return wav.squeeze().cpu().numpy()


def expected_seconds(text):
    return max(0.8, len(text) / CHARS_PER_SEC)


def cache_path(text, lang):
    sig = (f"{text}|{lang}|{EXAGGERATION}|{CFG_WEIGHT}|{TEMPERATURE}|"
           f"{SEED}|{USE_MULTILINGUAL}|{REFERENCE.stat().st_mtime_ns}")
    return CACHE_DIR / (hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16] + ".npy")


def get_clip(model, text, lang, sr):
    cp = cache_path(text, lang)
    if cp.exists():
        return np.load(cp), True

    exp = expected_seconds(text)
    best = None

    for attempt in range(MAX_RETRIES + 1):
        torch.manual_seed(SEED + attempt)
        wav = np.asarray(synth_once(model, text, lang), dtype=np.float32).reshape(-1)

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
        raise RuntimeError(f"Chatterbox produced no usable audio for: {text}")

    np.save(cp, best)
    return best, False


def generate_speech(model, items, sr):
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
                wav, cached = get_clip(model, chunk, lang, sr)
            except Exception as exc:
                traceback.print_exc() 
                print("\nSPEECH GENERATION FAILED")
                print("Line :", n)
                print("Text :", chunk)
                raise RuntimeError(f"Chatterbox failed on line {n}") from exc

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
         "highpass=f=80,acompressor=threshold=-18dB:ratio=1.8:attack=5:release=80,"
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
    model = load_model()
    sr = model.sr
    print("Output sample rate:", sr)

    audio = generate_speech(model, items, sr)
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