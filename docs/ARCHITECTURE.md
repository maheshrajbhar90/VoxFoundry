# CloneMyVoice Architecture

## Layers

```text
            ┌──────────────────────────────────────────────────────────┐
 CLI        │ src/clonemyvoice/cli.py  argument validation, orchestration,  │
            │                      exit codes, logging, file output     │
            └───────┬───────────────────┬──────────────────────┬───────┘
                    │                   │                      │
 Shared     text_processor.py      audio.py               errors.py
            parse → chunk →        AudioResult, analyze,  typed exceptions
            Segment list           validate, reference,
            (model-independent)    legacy_clean_clip (opt-in)
                    │                   │
                    └─────────┬─────────┘
                              ▼
 Backend    backend_adapter.py  BackendAdapter (ABC)
                                 ├── XTTSAdapter      (coqui-tts, real)
                                 └── MockXTTSAdapter  (test tone, explicit only)
```

| Module | Owns | Must not |
|---|---|---|
| `text_processor` | script format, short-line merge, chunk limits, pause durations, segment order | import torch/TTS, know about any model |
| `audio` | output contract, measurements, validation, reference checks, explicit post-processing | change audio implicitly; write reference files |
| `backend_adapter` | model loading, device, inference parameters, retries, raw output | segment text; normalise/trim; pick a device on its own |
| `cli` | user-facing behaviour, chunk → synthesize → pause → concatenate → write | contain model-specific logic |
| `errors` | the exception vocabulary | — |

Adding a backend (e.g. Chatterbox) = one new `BackendAdapter` subclass plus a CLI `--engine`
choice. Text processing, validation, output writing and error handling are reused unchanged.

## XTTSAdapter lifecycle

```text
INITIALIZED ──load(ref)──▶ [validate ref] ─▶ [load checkpoint once] ─▶ LOADED ─▶ READY
                                                                                 │  ▲
                                                               synthesize_result │  │ (always, even on error)
                                                                                 ▼  │
                                                                             GENERATING
READY / LOADED / INITIALIZED ──unload()──▶ UNLOADED ──load(ref)──▶ … (checkpoint reloaded)
```

* Reference validation happens **before** the checkpoint is touched; a bad reference never
  changes the adapter's state.
* A second `load()` with a different reference reuses the loaded model.
* `unload()` is idempotent, drops all references, runs `gc.collect()` and the device-specific cache
  release (`torch.mps.empty_cache()` / `torch.cuda.empty_cache()`).
* Loading forces Hugging Face offline mode unless `allow_download=True`.

## Output contract

`synthesize_result(text, language) -> AudioResult`:

| Field | Meaning |
|---|---|
| `audio` | 1-D float32, **raw** model output |
| `sample_rate`, `channels` | 24 000 Hz, 1 for XTTS-v2 |
| `backend`, `device`, `is_mock` | provenance |
| `generation_seconds`, `real_time_factor` | performance |
| `attempts`, `warnings`, `metadata` | retries, implausible durations, seed, settings |

`synthesize()` still returns `(audio, sample_rate)` and `generate()` still returns the array.
**Migration note:** before Phase 2.5, `generate()` returned *cleaned* audio
(trim + fades + RMS level-match) and cached it automatically. It now returns raw audio and
caching is opt-in (`cache_dir=`). The CLI flag `--legacy-clip-cleanup` reproduces the old
per-clip cleanup.

## Configuration

* XTTS inference/conditioning parameters: `XTTSSettings` (defaults = legacy `archive/legacy_scripts/clone_voice.py`).
* Text pacing and chunk limits: constants at the top of `text_processor.py` (= `clone_voice.py`).
* Device, cache, downloads: `XTTSAdapter(...)` arguments / CLI flags.
No configuration framework is used; there are only these three places.

## Environments

| Environment | Python | Purpose |
|---|---|---|
| `.venv-multiengine` | 3.11 | The only environment: `clonemyvoice` (XTTS-v2), tests, `voice_clone_chatterbox.py`, `voice_clone_voxcpm2.py`. Recreate from `requirements.txt` (+ `pip install -e . --no-deps`). |

## Usage

```bash
source .venv-multiengine/bin/activate

python -m clonemyvoice generate \
    --engine xtts --device mps \
    --reference-audio references/final_reference.wav \
    --language en \
    --text "This is a real speech generation test." \
    --output output/xtts_test.wav

python -m clonemyvoice generate --script examples/sample_script.txt \
    --reference-audio references/final_reference.wav --output output/voiceover.wav --device cpu
```

Script format: a line containing only `[EN]` or `[HI]` switches language; a blank line adds a
paragraph pause after the previous line; lines under 30 characters merge into the next line of
the same language.

Exit codes: 0 ok · 1 internal · 2 usage · 3 input · 4 environment/model/device · 5 generation · 6 not implemented.

## Testing

```bash
python -m pytest                    # unit + integration (fake model / mock), no checkpoint needed
python -m pytest -m unit
python -m pytest -m integration     # MOCKED: never real inference
python -m pytest -m real_model -v   # actual XTTS-v2; CLONEMYVOICE_TEST_DEVICE=mps|cpu
```

Real-model tests skip if the checkpoint is not already cached and never download.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: cannot import name 'TransformGetItemToIndex'` | torch < 2.6 with transformers 5.x. Use the versions in `requirements.txt`. |
| `ModuleNotFoundError: six` | coqui-tts 0.28.0 does not declare `six`; `pip install six==1.17.0`. |
| `DeviceUnavailableError: … 'mps'` | MPS not available; rerun with `--device cpu` (explicit). |
| `ModelLoadError … downloads are disabled` | Checkpoint not cached; rerun once with `--allow-download` (≈1.88 GB, CPML licence). |
| `Unsupported marker [HOOK]` | XTTS scripts accept only `[EN]` / `[HI]` markers. |
| Exit 3 "Output exists" | Choose a new `--output` or pass `--overwrite`. |
