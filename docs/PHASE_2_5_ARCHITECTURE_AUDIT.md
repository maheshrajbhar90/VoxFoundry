# Phase 2.5 — Architecture Audit

**Stage:** 1 (audit only; no source code changed in this stage)
**Date:** 2026-11-21
**Scope:** `voxfoundry/`, `tests/`, `clone_voice.py` (legacy XTTS), with reference reading of
`clone_voice_with_chatterbox.py`, `clone_voice_with_voxcpm2.py`, `voiceover_master.py`, `run.sh`.

Every claim below comes from reading the code or running it. Test outcomes in §3.4 are observed
results from the baseline run.

---

## 0. Blocking environment status

| Item | Status |
|---|---|
| `.venv-xtts` | **Missing.** It existed at 11:47 on 2026-11-21 and was gone at the start of this audit (removed outside the assistant session). |
| Rollback record | `venv-xtts-freeze-before-phase25.txt` (state *before* the torch/transformers fix — that state was broken, see §5). |
| `.venv-multiengine` (renamed from `voiceclone/`) | Present. Python **3.11.18**, coqui-tts 0.28.0, torch 2.6.0, transformers 5.2.0, numpy 1.26.4, plus Chatterbox and VoxCPM2. No `pip`, no `pytest`. |
| Project root | Is itself a Python 3.11 venv (`bin/`, `lib/`, `include/`, `pyvenv.cfg`, 5.1 GB), originally created at `AutoGraphEngine/voiceclone`. |
| Git | Repository has **no commits**; all files untracked. There is no baseline to diff against. |
| Baseline test run | **8 passed, 5 failed** (run after `.venv-xtts` was recreated; commit `9037c57`). See §3.4. |

`.venv-xtts` was recreated (Python 3.12.14) with approval. `six==1.17.0` had to be installed
explicitly: coqui-tts 0.28.0 imports `six` (`TTS/tts/utils/text/punctuation.py`) but does not
declare it as a dependency.

---

## 1. Current architecture

```text
voxfoundry/
  __init__.py         re-exports text_processor, backend_adapter, cli.main
  text_processor.py   parse_script, make_chunks, pause_after, silence (copied from legacy)
  backend_adapter.py  BackendAdapter (ABC), XTTSAdapter, MockXTTSAdapter
  cli.py              argparse: `generate` (XTTS only), `master` (stub)
tests/
  test_text_processor.py   pure unit tests
  test_backend_adapter.py  adapter tests with a DummyTTS
  test_cli.py              CLI tests using MockXTTSAdapter
legacy (standalone, hard-coded paths):
  clone_voice.py                  XTTS-v2 pipeline (reference: references/my_voice.wav)
  clone_voice_with_chatterbox.py  Chatterbox pipeline (references/ref3_10s.wav)
  clone_voice_with_voxcpm2.py     VoxCPM2 pipeline (references/ref3_10s.wav)
  voiceover_master.py             mastering (out of scope for this phase)
```

Data flow (`cli generate`): `parse_script` → per item `adapter.synthesize(text, lang)` →
append `silence(pause_after(item))` → `np.concatenate` → `sf.write`.

---

## 2. Strengths

1. A backend interface exists (`BackendAdapter`) with `load` / `generate` / `synthesize`.
2. Text parsing is a separate, model-independent module with its own tests.
3. The model is loaded once in `load()` and held on the instance; `generate()` reuses it.
4. Inference parameters are faithfully copied from the legacy script (speed, temperature,
   top-p/k, repetition penalty, conditioning lengths, seed).
5. The CLI already checks output for empty, non-finite and fully silent audio and returns a
   non-zero exit code on failure.
6. `MockXTTSAdapter` is a separate class, not an automatic fallback path.

---

## 3. Structural problems (verified defects)

Severity: **C** = critical (blocks correct real inference or violates a Phase 2.5 rule),
**M** = major, **m** = minor.

### 3.1 `backend_adapter.py`

| # | Sev | Defect | Location | Why it matters |
|---|---|---|---|---|
| A1 | C | File did not compile (stray docstring fragment + duplicate `__init__`). **Already fixed** earlier this session (minimal removal; original saved in the session scratchpad). | former L123–136 | Nothing in `voxfoundry` could be imported. |
| A2 | C | Device is auto-selected (CUDA → MPS → CPU) with no override. Legacy uses CPU. | `load()` | Violates rule 8 (no silent device switching); silently changes legacy behaviour. |
| A3 | C | `except TypeError:` silently retries with fewer parameters. | `_synth_once()` | Violates "no silent fallbacks". Would also mask genuine `TypeError` bugs. In coqui-tts 0.28.0 all tuning kwargs are accepted (`Xtts.synthesize(**kwargs)`), so the fallback is never needed. |
| A4 | C | `generate()` returns audio after `_clean_clip` (trim, fades, RMS gain 0.5–2.5×, peak limit). | `_get_clip()` | Raw output is not available; violates Phase E ("no normalize/trim in this phase"). |
| A5 | M | Output-audio cache in `<cwd>/.voxfoundry_cache`, created in `__init__`. Key = text, lang, sampling params, seed, reference **mtime**. It does not include the model checkpoint, device, or `GPT_COND_*`/`MAX_REF_LEN`. | `__init__`, `_cache_path()` | Constructing an adapter writes to the filesystem; stale audio can be served after a model/device/conditioning change; timing tests measure cache hits, not inference. |
| A6 | M | Retry loop returns the last clip even when its duration is outside the expected range, with no signal to the caller. | `_get_clip()` | A "succeeded" result can be a known-bad clip. |
| A7 | M | `unload()` only calls `torch.cuda.empty_cache()`; no `torch.mps.empty_cache()`, no `gc.collect()`. | `unload()` | MPS memory is not released; violates "CUDA-only functions are not MPS cleanup". |
| A8 | M | Errors: `print` + `traceback.print_exc()` then re-raise generic exceptions. No typed errors. | `load()` | Callers cannot distinguish dependency / load / reference / language / inference failures. |
| A9 | M | Reference validation = `is_file()` only. Legacy also checks size, decodability, frames, ≥5 s duration, peak ≥ 0.02. | `load()` | Corrupt / silent / too-short references reach the model. |
| A10 | M | Language not validated by VoxFoundry. Coqui validates with `assert` (skipped under `python -O`; the expression also always passes for `"zh"`). `supported_languages = ["en","hi"]` is declared but never used. | `generate()` | Unsupported languages fail late with `AssertionError`, or not at all under `-O`. |
| A11 | M | Speaker conditioning (`clone_voice` → `get_conditioning_latents`) is recomputed on **every** `tts()` call because no `speaker` id is passed. | `_synth_once()` | Repeated cost per chunk; the "model caching" the plan asks about exists, but conditioning caching does not. |
| A12 | M | `import TTS.api` / `torch` at module import. `voxfoundry/__init__.py` imports `backend_adapter`, so even `import voxfoundry.text_processor` pulls in Coqui + torch. | top of file | Unit tests for text processing need the full ML stack; slow imports; no `ModelDependencyError` possible. |
| A13 | m | `load(model_path)` actually receives the reference audio path (documented in the docstring). | ABC + `load()` | Misleading contract for future backends. |
| A14 | m | Calling `load()` twice reloads the model from disk (no "already loaded" check). | `load()` | Wasted time/memory; unclear lifecycle. |
| A15 | m | Module-level tuning constants (duplicated from legacy) rather than per-instance configuration. | top of file | Cannot vary parameters without editing source. |
| A16 | m | `synthesize()` returns a bare `(ndarray, int)` tuple; no backend name, timing, or metadata. | ABC | Phase E output contract not met. |

### 3.2 `cli.py`

| # | Sev | Defect | Why it matters |
|---|---|---|---|
| C1 | C | **No chunking.** Each script line is sent whole to the model; `make_chunks` is imported nowhere in the CLI. Legacy chunks at `MAX_CHARS = {"hi":130,"en":200}`. | Long lines exceed XTTS limits (Coqui: en 250, hi 150 chars) → truncation/quality loss. Legacy behaviour not reproduced. |
| C2 | M | `engine = args.engine or args.backend` — `--engine` defaults to `"xtts"`, so `--backend` is never honoured. | Deprecated alias silently ignored. |
| C3 | M | Output file is overwritten without check. | Violates rule 9. |
| C4 | M | No `--device`, no `--text`, no verbose/logging, no metrics, no indication of mock vs real backend. | Phase J requirements. |
| C5 | M | `except Exception` → `print("Error: …")`, exit 1. Same exit code for every failure; traceback lost. | Poor diagnosability. |
| C6 | m | `master` prints "not implemented" and returns **0** (success). | Misleading exit status. |
| C7 | m | Module docstring refers to a non-existent `TextProcessor` class. | Stale documentation. |

### 3.3 `text_processor.py` — differs from the legacy XTTS script

| Setting | `clone_voice.py` | `text_processor.py` | Effect |
|---|---|---|---|
| `MIN_LINE_CHARS` (merge short lines) | **30** | **2** | Short lines (< 30 chars) are no longer merged → more 1–4 word clips, which the legacy code comments say XTTS is unstable on. |
| `PAUSE_PERIOD` | 0.25 | 0.30 | Different pacing. |
| `PAUSE_QUESTION` | 0.45 | 0.30 | Different pacing. |
| `PAUSE_EXCLAIM` | 0.30 | 0.30 | Same. |
| `PAUSE_ELLIPSIS` | 0.60 | 0.50 | Different pacing. |
| `PAUSE_OTHER` | 0.20 | 0.20 | Same. |
| `PAUSE_PARAGRAPH` | 0.70 | 0.60 | Different pacing. |
| `PAUSE_INSIDE_LINE` | 0.25 | absent | Needed once chunking is restored. |
| `MAX_CHARS` per language | `{"hi":130,"en":200}` | absent | See C1. |

`parse_script`, `make_chunks` and `pause_after` logic is otherwise line-for-line identical to legacy.
`tests/test_pause_after` asserts `"Hello!"` → 0.3 (true in both). No test pins the other values,
so the divergence is untested. Restoring legacy values is a behaviour change that needs your
decision (§6, Q3).

### 3.4 Tests

| Test | Problem | Expected outcome |
|---|---|---|
| `test_cache_dir_creation` | `tmp_cwd` is a plain generator, used with `with` (no `@contextmanager`). | **Fails (observed):** `TypeError: 'generator' object does not support the context manager protocol`. |
| `test_load_and_generate` | Fake `load` sets `reference_audio = Path("voice.wav")`, which does not exist; `_cache_path` calls `.stat()` on it. | **Fails (observed):** `FileNotFoundError: voice.wav`. |
| `test_unload_and_reuse` | Same `.stat()` problem. | **Fails (observed):** `FileNotFoundError: voice.wav`. |
| `test_invalid_reference_path`, `test_invalid_generate_without_load` | OK. | Pass (observed). |
| All `XTTSAdapter()` tests | Constructor creates `.voxfoundry_cache/` in the **repo** working directory. | Side effect on the repo. |
| `test_cli_*` | Use `MockXTTSAdapter` via monkeypatch (labelled in the docstring). Reference is a 1 s sine wave — would fail the legacy ≥5 s check once validation is added. | Pass (observed); need updating with A9. |
| `test_parse_script_basic` | Puts `[EN] Hello world.` on one line; the parser only recognises a marker alone on its line (`re.fullmatch`), so both lines are parsed as `en` text. | **Fails (observed):** `assert 'en' == 'hi'`. |
| `test_parse_script_paragraph_break` | Expects `para_break` on the line *after* the blank line; the parser (legacy and package) sets it on the line *before*. | **Fails (observed):** `assert True is False`. |
| General | No pytest config, no `unit` / `integration` / `real_model` markers, no real-model tests. Duplicate `import numpy as np` in `test_cli.py`. | — |

### 3.5 Repository / configuration

| # | Sev | Issue |
|---|---|---|
| R1 | M | No commits in git — no safe rollback for Stage 2 edits. |
| R2 | M | `requirements-xtts.txt` and `requirements-xtts-locked.txt` do not exist. `requirements-original.txt` and `requirements_voiceover.txt` are identical, mixed-backend lists. |
| R3 | M | `scripts/script.txt` uses `[HOOK]`, `[EXPLANATION]`, `[DATA]`, `[WARNING]` (VoxCPM2 format). Both the legacy XTTS parser and `text_processor.parse_script` reject these (`Unsupported marker [HOOK]`). The bundled script cannot be voiced by XTTS today. |
| R4 | m | The project root is itself a venv; `bin/`, `lib/`, `include/` are not in `.gitignore`. |
| R5 | m | `.gitignore` lists `*.wav`, `*.mp3`, `*.m4a` twice, and does not list `.voxfoundry_cache/` or `.venv-xtts/`. |
| R6 | m | `run.sh` activates a venv twice, then the root venv, and calls `clone_voice_with_chatterbox.py.py` (double extension). |
| R7 | m | `voxfoundry/__init__.py` has two module docstrings (second is a no-op string expression). |
| R8 | m | Each legacy script re-implements parse/chunk/pause/clean (~4× duplication). Expected for legacy; the package is the intended consolidation point. |

No circular imports were found (`cli` → `backend_adapter`, `text_processor`; neither imports `cli`).

---

## 4. Legacy compatibility report (`clone_voice.py` vs VoxFoundry)

| Aspect | `clone_voice.py` | VoxFoundry (current) | Compatible? |
|---|---|---|---|
| Model | `tts_models/multilingual/multi-dataset/xtts_v2` | same | Yes |
| Device | `.to("cpu")` fixed | auto CUDA→MPS→CPU | **No** (A2) |
| Reference default | `references/my_voice.wav` (42.2 s) | caller-supplied | n/a — must be chosen explicitly |
| Reference validation | exists, size>0, decodable, frames>0, ≥5 s, peak ≥0.02, tip if >35 s | `is_file()` | **No** (A9) |
| TOS handling | sets `COQUI_TOS_AGREED=1` | relies on existing `tos_agreed.txt` | Partially |
| Language handling | `[HI]`/`[EN]` markers, default `en` | same parser; adapter does not validate | Partially (A10) |
| Short-line merge | `< 30` chars | `< 2` chars | **No** (§3.3) |
| Chunking | `make_chunks(text, MAX_CHARS[lang])` | none in CLI | **No** (C1) |
| Inference params | speed 1.05, temp 0.7, top_p 0.85, top_k 50, rep 5.0, gpt_cond_len 30, chunk 6, max_ref 30, sound_norm_refs False | identical | Yes |
| TypeError fallback | yes (silent-ish, prints a note) | yes | Same — but violates Phase 2.5 rules (A3) |
| Seeding | `manual_seed(SEED + attempt)` | identical | Yes |
| Retry | up to 2 retries on invalid / odd duration | identical | Yes (A6 applies to both) |
| Clip cleanup | trim, 12 ms fades, RMS → 0.08 | identical, but inside the adapter | Same output; wrong layer (A4) |
| Inter-chunk pause | 0.25 s | n/a (no chunking) | **No** |
| Line pauses | §3.3 values | different values | **No** |
| Concatenation | `np.concatenate(segments)` | same | Yes |
| Output sample rate | `synthesizer.output_sample_rate` (24 000) | same | Yes |
| Output cache | `output/cache/*.npy`, same key scheme | `.voxfoundry_cache/*.npy`, same key scheme | Same scheme, different dir |
| Output file | `output/raw.wav` (PCM_16), then ffmpeg → `xtts_final.mp3` | `--output` via `sf.write` (default subtype) | Partially; mastering out of scope |

**Conclusion:** the adapter reproduces legacy *inference parameters* but not legacy *pipeline
behaviour* (device, chunking, short-line merging, pauses, reference validation). Output from the
current CLI would differ audibly from `clone_voice.py` on the same script.

---

## 5. Environment and dependency findings (from earlier in this session)

* `.venv-xtts` originally had torch 2.4.1 + transformers 5.23.0. transformers ≥5.x disables
  PyTorch < 2.6, and XTTS failed at import:
  `ImportError: cannot import name 'TransformGetItemToIndex' from 'torch._dynamo._trace_wrapped_higher_order_op'`.
* With approval, torch/torchaudio were moved to 2.6.0 and transformers to 5.2.0 (the pins recorded in
  `requirements-original.txt`); numpy stayed 2.5.5 because librosa 1.0.1 / scipy 1.18.1 require
  numpy ≥2. `pip check` was clean and `from TTS.tts.models.xtts import Xtts` succeeded.
* That environment has since been deleted (§0). **The Phase 2.5 document's stated baseline
  (PyTorch 2.4.1 / torchaudio 2.4.1) is not a working configuration** with transformers 5.x.

---

## 6. Recommended changes (Stage 2 — smallest safe corrections)

Each change will ship with a test. None is a re-architecture.

| # | Change | Files | Fixes |
|---|---|---|---|
| S1 | Explicit `device` argument (`"mps"`/`"cpu"`/`"cuda"`), default **from configuration, not auto-detection**; raise if the requested device is unavailable. | `backend_adapter.py`, `cli.py` | A2, C4 |
| S2 | Remove the `TypeError` fallback; let errors propagate as `InferenceError`. | `backend_adapter.py` | A3 |
| S3 | `generate_raw()` returns the untouched model output; `_clean_clip` becomes an explicit, opt-in post-processing step used by the CLI only when requested (default preserves legacy output). | `backend_adapter.py`, `cli.py` | A4 |
| S4 | Output cache: opt-in, lazily created, key extended with checkpoint id, device and conditioning params; disabled for real-model timing tests. | `backend_adapter.py` | A5 |
| S5 | Typed errors in a new `voxfoundry/errors.py` (`ModelDependencyError`, `ModelLoadError`, `ReferenceAudioError`, `UnsupportedLanguageError`, `InferenceError`, `AudioValidationError`). | new + `backend_adapter.py`, `cli.py` | A8, C5 |
| S6 | Reference validation ported from legacy (no modification of the file). | `backend_adapter.py` | A9 |
| S7 | Language validation against the **loaded model's** `config.languages`. | `backend_adapter.py` | A10 |
| S8 | **Deferred.** Coqui only caches conditioning via an on-disk `voice_dir`; `TTS.tts()` also splits sentences and pads 10 000 samples of silence per sentence, so bypassing it with `Xtts.inference` would change output. Keep the legacy path; measure the conditioning cost in Phase H and decide afterwards. | — | A11 |
| S9 | Lazy import of `TTS`/`torch` inside `load()`; `ModelDependencyError` if missing. | `backend_adapter.py` | A12 |
| S10 | Lifecycle state (`UNINITIALIZED → LOADED/READY → GENERATING → READY → UNLOADED`); `unload()` releases references, runs `gc.collect()` and the correct device cache clear; idempotent. | `backend_adapter.py` | A7, A14 |
| S11 | `AudioResult` dataclass returned by a new `synthesize_result()`; existing `synthesize()` keeps returning `(audio, sr)` for compatibility. | `backend_adapter.py` | A16 |
| S12 | CLI: chunk with `make_chunks` + per-language limits and inside-line pause; honour `--backend`; `--device`, `--text`, `--overwrite`, `-v`; print backend, device, mock/real, metrics. | `cli.py`, `text_processor.py` | C1–C7 |
| S13 | Fix the three broken adapter tests; add pytest markers (`unit`, `integration`, `real_model`), real-model tests that are skipped unless selected **and** the checkpoint is already cached (no downloads). | `tests/`, new `pytest.ini`/`conftest.py` | §3.4 |
| S14 | `requirements-xtts.txt` (direct deps) and `requirements-xtts-locked.txt` (`pip freeze` of a verified env). | new | R2 |

---

## 7. Files

**Will be modified (Stage 2, after your go-ahead):**
`voxfoundry/backend_adapter.py`, `voxfoundry/cli.py`, `voxfoundry/text_processor.py` (only if Q3 = restore legacy values; adds `MAX_CHARS`/`PAUSE_INSIDE_LINE`), `voxfoundry/__init__.py` (lazy import / docstring), `tests/*`.

**Will be created:**
`voxfoundry/errors.py`, `pytest.ini` or `tests/conftest.py`, `requirements-xtts.txt`, `requirements-xtts-locked.txt`, `docs/ARCHITECTURE.md`, `docs/PHASE_2_5_VALIDATION_REPORT.md`.

**Must remain untouched:**
`clone_voice.py`, `clone_voice_with_chatterbox.py`, `clone_voice_with_voxcpm2.py` (one docstring line was already edited during the venv rename), `voiceover_master.py`, everything in `references/`, all existing audio (`*.wav`, `*.mp3`, `*.m4a`), `scripts/script.txt`, `output/cache*`, the model checkpoint, `.venv-multiengine`, and every environment outside this repo.

---

## 8. Decisions needed before Stage 2

1. **Environment:** recreate `.venv-xtts` (Python 3.12, torch 2.6.0, torchaudio 2.6.0, transformers 5.2.0, coqui-tts 0.28.0, numpy 2.x, + pytest), or run Phase 2.5 in `.venv-multiengine` (Python 3.11)?
2. **Git baseline:** may I make an initial commit of the current source (code, tests and docs only; venvs, audio and models are excluded) so every Stage 2 change has a reviewable diff?
3. **Text-processing values:** restore the legacy values (`MIN_LINE_CHARS = 30`, legacy pauses), or keep the current package values and document the difference?
4. **Default device:** MPS (plan preference) or CPU (legacy behaviour)? Either way it will be explicit and logged; no automatic switching.
5. **Reference recording:** `references/my_voice.wav` (the legacy default, 42.2 s) or `references/voice_sample.wav` (22.0 s)?
