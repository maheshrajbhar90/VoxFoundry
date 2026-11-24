# VoxFoundry

VoxFoundry clones **your own voice** from a short recording and uses it to read long voiceover
scripts in English and Hindi. Everything runs **locally** on your machine with open-source
text-to-speech models. No audio or text is sent to a cloud service.

It was built to produce narration for short-form videos without recording every take by hand.
You record yourself once, write a script, and VoxFoundry generates the full voiceover.

> **Use only voices you are authorised to clone.** That means your own voice, or someone who has
> given explicit consent.

---

## Voice-cloning engines

The project includes three local engines, all run from the same Python environment:

| Engine | How you run it | Best for | Output |
|---|---|---|---|
| **XTTS-v2** (main) | `voxfoundry generate ...` | Reliable English + Hindi voiceovers, tested CLI | `.wav` (path you choose) |
| **VoxCPM2** | `python voice_clone_voxcpm2.py` | Expressive narration with a different style per section (`[HOOK]`, `[DATA]` ...) | `output/final_voxcpm2.mp3` |
| **Chatterbox** | `python voice_clone_chatterbox.py` | English narration with adjustable emotion and intensity | `output/final.mp3` |

`voxfoundry` is the production package: it has command-line options, input validation, clear
error codes and a test suite. The two engine scripts are standalone, and you configure them by
editing the settings at the top of each file.

---

## How cloning works

```text
 reference.wav ─┐
 (your voice)   │      ┌────────────────┐     ┌──────────────────────┐     ┌──────────────┐
                ├────▶ │ Text processor │ ──▶ │ Local TTS model      │ ──▶ │  Assembler   │ ──▶ voiceover
 script.txt ────┘      │ parse + chunk  │     │ (MPS / CPU / CUDA)   │     │ pauses + WAV │
                       └────────────────┘     └──────────────────────┘     └──────────────┘
```

All three engines follow the same pipeline. For XTTS-v2 (`voxfoundry`):

1. **Reference check.** The recording is validated before the model loads: it must exist,
   decode, contain audible signal, and be at least 5 seconds long.
2. **Voice conditioning.** XTTS-v2 reads up to the first 30 seconds of the reference and turns it
   into a *speaker embedding* and *GPT conditioning latents*, a compact numeric description of
   your timbre, pitch and speaking style. Nothing is trained or fine-tuned: cloning is
   **zero-shot**, so a new reference takes effect immediately.
3. **Text processing.** The script is split by language (`[EN]` / `[HI]`). Very short lines are
   merged, and long lines are chunked under the model's limits (200 characters for English, 130
   for Hindi) so it doesn't hallucinate or cut off words.
4. **Synthesis.** Each chunk is generated in your voice at 24 kHz mono. A fixed seed (`42`) makes
   runs reproducible. If a chunk comes out empty, or its length doesn't fit the text, it is
   retried with a new seed.
5. **Assembly.** Chunks are joined with natural pauses: longer after `?` and `...`, and a
   paragraph pause for a blank line. The result is written as 16-bit PCM WAV, and its duration,
   peak level, loudness and generation speed are printed.

The model loads once per run, and Hugging Face runs in **offline mode** unless you pass
`--allow-download`. After the first download, no network is needed.

VoxCPM2 and Chatterbox work the same way with their own models, chunk sizes and pause timings.
They also use `ffmpeg` to polish the result into an MP3.

---

## Project structure

```text
VoxFoundry/
├── src/voxfoundry/               # production package (XTTS-v2)
│   ├── cli.py                    #   command-line interface, orchestration, exit codes
│   ├── text_processor.py         #   script parsing, chunking, pause timing
│   ├── backend_adapter.py        #   model loading and inference (XTTSAdapter, mock)
│   ├── audio.py                  #   audio validation, measurements, reference checks
│   └── errors.py                 #   typed exceptions
├── voice_clone_voxcpm2.py        # VoxCPM2 engine script
├── voice_clone_chatterbox.py     # Chatterbox engine script
├── scripts/script.txt            # input script read by the two engine scripts
├── examples/sample_script.txt    # sample input for voxfoundry
├── tests/                        # unit, integration and opt-in real-model tests
├── docs/                         # architecture, model compatibility, validation reports
├── references/                   # your voice recordings (git-ignored, never uploaded)
├── output/                       # generated audio (git-ignored)
├── archive/                      # retired scripts and old dependency lists
├── .github/workflows/tests.yml   # CI: runs the test suite on every push
├── pyproject.toml                # package metadata, `voxfoundry` command, pytest config
└── requirements.txt              # exact dependency versions
```

Design details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Requirements

- Python **3.11**
- macOS with Apple Silicon is the tested setup. CPU works everywhere, and CUDA is supported by XTTS-v2.
- About 10 GB of free disk space for the environment and model checkpoints
- `ffmpeg`, used to convert recordings and to create the MP3 output of the engine scripts
  (`brew install ffmpeg`)

## Installation

```bash
git clone https://github.com/<your-username>/VoxFoundry.git
cd VoxFoundry

python3.11 -m venv .venv-multiengine
source .venv-multiengine/bin/activate

pip install -r requirements.txt
pip install -e . --no-deps        # installs the `voxfoundry` command
```

Activate the environment (`source .venv-multiengine/bin/activate`) in every new terminal before
running anything below.

The first XTTS-v2 run must download the model (~1.88 GB) once:

```bash
voxfoundry generate --allow-download --text "Hello." \
    --reference-audio references/final_reference.wav --output output/first.wav
```

---

## Usage

### 1. Record a reference

Record 15–30 seconds of clear, natural speech in a quiet room, with no music and no other
speakers. Speak the way you want the voiceover to sound. Convert it to mono WAV and save it in
`references/`:

```bash
ffmpeg -i my_recording.m4a -ar 22050 -ac 1 references/final_reference.wav
```

### 2. Write a script

```text
[EN]
Welcome to VoxFoundry. This short script shows the supported format.

Each line is spoken as one sentence group. A blank line adds a paragraph pause.

[HI]
नमस्ते, यह हिंदी में एक छोटा सा उदाहरण है।
```

- A line containing only `[EN]` or `[HI]` switches the language.
- A blank line adds a paragraph pause.
- Lines shorter than 30 characters merge into the next line of the same language.
- Write numbers as words ("twenty-five thousand") for the most natural reading.

See [examples/sample_script.txt](examples/sample_script.txt).

### 3. Generate with XTTS-v2 (`voxfoundry`)

```bash
# From a script file
voxfoundry generate \
    --script examples/sample_script.txt \
    --reference-audio references/final_reference.wav \
    --output output/voiceover.wav

# Quick test from inline text
voxfoundry generate --text "This is a quick test." \
    --reference-audio references/final_reference.wav --output output/test.wav
```

| Option | Purpose |
|---|---|
| `--script FILE` / `--text "..."` | Input: a script file or inline text (one is required). |
| `--reference-audio FILE` | Your voice recording (WAV/FLAC). |
| `--output FILE` | Where to write the WAV. |
| `--device mps\|cpu\|cuda` | Inference device (default `mps`). Never switched silently: if MPS is unavailable, pass `--device cpu`. |
| `--language en\|hi` | Language for lines before the first marker (default `en`). |
| `--max-chars N` | Override the chunk length limit. |
| `--overwrite` | Replace an existing output file. |
| `--cache-dir DIR` | Cache generated chunks, so re-runs after small script edits are fast. |
| `--legacy-clip-cleanup` | Trim, fade and level-match each chunk. |
| `--allow-download` | Allow downloading the model checkpoint. |
| `--engine mock` | Test tone instead of speech, to check the pipeline without the model. |
| `-v` | Verbose logging. |

`python -m voxfoundry generate ...` works the same way.

Exit codes: `0` ok · `1` internal · `2` usage · `3` invalid input · `4` environment/model/device ·
`5` generation failure · `6` not implemented.

### 4. Generate with VoxCPM2 or Chatterbox

Both scripts read `scripts/script.txt` and use `references/ref3_10s.wav` as the voice. To change
anything, edit the settings block at the top of the script.

```bash
python voice_clone_voxcpm2.py       # → output/raw_voxcpm2.wav + output/final_voxcpm2.mp3
python voice_clone_chatterbox.py    # → output/raw.wav + output/final.mp3
```

| Setting | VoxCPM2 | Chatterbox |
|---|---|---|
| Voice file | `REFERENCE` | `REFERENCE` |
| Device | `DEVICE = "cpu"` | `DEVICE = "cpu"` |
| Expressiveness | `STYLES` per section, `CFG_VALUE` | `EXAGGERATION`, `CFG_WEIGHT`, `TEMPERATURE` |
| Quick test | `PREVIEW_LINES = 10` | `PREVIEW_LINES = 10` |
| Languages | `[EN]` and `[HI]` markers (Hindi quality not yet checked) | English (set `USE_MULTILINGUAL = True` for `[HI]`) |

**VoxCPM2 section markers.** Besides `[EN]` / `[HI]`, the VoxCPM2 script accepts `[HOOK]`,
`[EXPLANATION]`, `[DATA]`, `[WARNING]` and `[CONCLUSION]`. Each section is spoken in its own style,
for example a punchy hook or a calm, precise delivery for numbers. These markers work only in
`voice_clone_voxcpm2.py`, not in `voxfoundry`.

Both scripts cache finished chunks in `output/cache_voxcpm2/` and `output/cache_chatterbox/`, so a
re-run only regenerates lines you changed.

---

## Testing

```bash
python -m pytest                     # unit + integration tests (no model needed, a few seconds)
python -m pytest -m unit
python -m pytest -m real_model -v    # real XTTS-v2 inference (~1 min); needs the cached checkpoint
```

Real-model tests are skipped unless selected, and they never download anything. CI
(`.github/workflows/tests.yml`) runs the default suite on every push and pull request. The tests
cover `voxfoundry`; the two engine scripts are not covered.

---

## Privacy

Voice recordings are biometric data. [.gitignore](.gitignore) blocks every common audio and video
format, the contents of `references/` and `output/`, speaker embeddings (`.npy`, `.pkl`), model
files, logs and `.env` files, so they can't be committed by accident. Keep a separate backup of
your reference recordings, because git doesn't store them.

## Licences

| Model | Licence | Commercial use |
|---|---|---|
| XTTS-v2 | [Coqui Public Model License](https://coqui.ai/cpml) | **No**, non-commercial only |
| VoxCPM2 (`voxcpm`) | Apache-2.0 | Yes |
| Chatterbox (`chatterbox-tts`) | MIT | Yes |

Audio generated with XTTS-v2 inherits its non-commercial restriction. For monetised content,
use VoxCPM2 or Chatterbox, and check each model's current licence terms yourself. More detail:
[docs/XTTS_MODEL_COMPATIBILITY.md](docs/XTTS_MODEL_COMPATIBILITY.md).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: voxfoundry` | Activate `.venv-multiengine`, then run `pip install -e . --no-deps`. |
| `ModelLoadError ... downloads are disabled` | Run once with `--allow-download`. |
| `DeviceUnavailableError: ... 'mps'` | Use `--device cpu`. |
| `ModuleNotFoundError: six` | `pip install six==1.17.0` (coqui-tts doesn't declare it). |
| `cannot import name 'TransformGetItemToIndex'` | torch is older than 2.6. Reinstall from `requirements.txt`. |
| `Unsupported marker [HOOK]` in `voxfoundry` | Section markers only work in `voice_clone_voxcpm2.py`. Use `[EN]` / `[HI]`. |
| Exit code 3, "Output exists" | Choose a new `--output` or pass `--overwrite`. |
| `ffmpeg: command not found` | `brew install ffmpeg`. |
| Voice doesn't sound like you | Use a cleaner 15–30 s reference with steady volume and no background noise. |
