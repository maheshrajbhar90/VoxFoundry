# Phase 2.5 — Validation Report

**Date:** 2026-11-21 · **Environment:** `.venv-xtts` (Python 3.12.14, coqui-tts 0.28.0, torch 2.6.0,
torchaudio 2.6.0, transformers 5.2.0, numpy 2.5.5) · **Hardware:** Apple Silicon, 24 GB, **MPS**
**Model:** XTTS-v2.0.3 (cached 2026-11-14; SHA-256 verified; **no download performed**)
**Reference:** `references/my_voice.wav` (legacy default; 42.2 s, 22 050 Hz, mono)
**Outputs:** `output/phase25/` (new files only; git-ignored). Raw metrics: `output/phase25/metrics_mps.json`,
`output/phase25/mem_loop_mps.json`.

All numbers below were measured in this session. Nothing is estimated unless it says so.

---

## 1. Acceptance summary

| Requirement | Result | Evidence |
|---|---|---|
| Architecture | **PASS** | Text, audio contract, errors, backend and CLI are separate modules (`docs/ARCHITECTURE.md`). |
| Environment | **PASS** | `.venv-xtts` recreated, `pip check` clean; locked in `requirements-xtts-locked.txt`. |
| Dependencies | **PASS** | torch 2.6.0 + transformers 5.2.0 + coqui-tts 0.28.0 (+ undeclared `six`). torch 2.4.1 is **incompatible** with transformers 5.x. |
| Model download | **NOT TESTED** (not needed) | Checkpoint already cached; adapter forces offline mode. |
| Model loading | **PASS** | Loads in 15.6–17.7 s on MPS. |
| MPS inference | **PASS** | All real tests ran on MPS with no unsupported-operator errors. |
| CPU inference | **NOT TESTED** | Not required: MPS succeeded. Available with `--device cpu`. |
| Direct Coqui inference | **PASS** | `t1_direct_plain_en_mps.wav`, `t1_direct_legacyparams_en_mps.wav`. |
| VoxFoundry adapter inference | **PASS** | `t2_adapter_en_mps.wav` is **bit-identical** to the direct call with the same parameters and seed (max abs diff 0.0). |
| Audio contract | **PASS** | 1-D float32, 24 kHz mono, finite, metadata in `AudioResult`. |
| Reference handling | **PASS** | Missing/empty/corrupt/short/silent/NaN/unreadable references rejected (unit tests). |
| Model lifecycle | **PASS** | One load reused for 6 calls; unload frees MPS memory (1.868 GB → 0.0); reload works. |
| Caching (model reuse) | **PASS** | `model_load_count == 1` after 15 calls. Output-audio cache is separate and off by default. |
| Memory stability | **FAIL — open issue** | See §5. Identical input: flat. Varying input: RSS +1.5 GB over 15 calls (MPS memory flat). |
| Text processing | **PASS** | Legacy values restored and pinned; chunking matches `clone_voice.py`. |
| CLI | **PASS** | End-to-end real run, exit 0; overwrite refused. |
| Error handling | **PASS** | No mock substitution, no device fallback, no TypeError fallback; typed exit codes. |
| Tests | **PASS** | Unit + integration: 100 passed. Real model: 5 passed. |
| English speech | **PASS** | Speech heuristic positive on every English output. |
| Hinglish (as `en`) | **PASS (experimental)** | Generated, speech heuristic positive. Pronunciation **not yet judged by ear.** |
| Hindi (`hi`) | **PASS — supported** | Accepted and generated. XTTS-v2.0.3 lists Hindi (17 languages). Pronunciation **not yet judged by ear.** |
| Manual listening | **PARTIAL** | User: `t2_adapter_en_mps.wav` is "close but not perfect". Other files not yet reviewed. |
| Legacy compatibility | **PASS with documented differences** | §6. |
| Documentation | **PASS** | `ARCHITECTURE.md`, `XTTS_MODEL_COMPATIBILITY.md`, audit, this report. |

**Not production-ready yet.** Open items: memory growth under varied input (§5), full manual listening,
and a side-by-side listening comparison against `clone_voice.py` output.

---

## 2. Test suites

| Suite | Command | Result |
|---|---|---|
| Baseline (before changes) | `pytest tests/` | 8 passed, 5 failed (test bugs; see audit §3.4) |
| Unit | `pytest -m unit` | included below |
| Integration (**mocked**: FakeTTS / Mock backend, no real inference) | `pytest -m integration` | included below |
| Unit + integration | `pytest` | **100 passed**, 5 deselected |
| Real model (MPS) | `VOXFOUNDRY_TEST_DEVICE=mps pytest -m real_model -v` | **5 passed** in 72 s |

---

## 3. Real inference results (MPS)

Measured by `analyze_audio` on the raw output. *Speech* = heuristic (active-frame ratio ≥ 0.2,
tonal concentration < 0.5); it rules out silence and test tones but is not proof of intelligibility.

| File | Lang | Duration | Peak | RMS dBFS | NaN/Inf | Active | Tonal | Speech | Gen time | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| `t1_direct_plain_en` (Coqui defaults) | en | 2.73 s | 0.814 | −17.6 | 0 | 0.79 | 0.25 | yes | 19.77 s* | 7.24* |
| `t1_direct_legacyparams_en` | en | 2.45 s | 0.819 | −18.7 | 0 | 0.77 | 0.20 | yes | 3.62 s | 1.48 |
| `t2_adapter_en` | en | 2.45 s | 0.819 | −18.7 | 0 | 0.77 | 0.20 | yes | 2.44 s | 0.99 |
| `t3_adapter_repeat1_en` | en | 5.06 s | 0.753 | −18.7 | 0 | 0.79 | 0.10 | yes | 8.05 s | 1.59 |
| `t3_adapter_repeat2_en` | en | 5.33 s | 0.915 | −18.5 | 0 | 0.78 | 0.12 | yes | 6.79 s | 1.27 |
| `t3_adapter_repeat3_en` | en | 6.44 s | 0.908 | −19.6 | 0 | 0.61 | 0.08 | yes | 8.49 s | 1.32 |
| `t4_hinglish_as_en` | en | 7.19 s | 0.902 | −19.1 | 0 | 0.74 | 0.09 | yes | 10.01 s | 1.39 |
| `t4_hindi_hi` | hi | 7.56 s | 0.919 | −19.9 | 0 | 0.77 | 0.08 | yes | 8.47 s | 1.12 |
| `t5_after_reload_en` | en | 1.43 s | 0.755 | −19.0 | 0 | 0.63 | 0.17 | yes | — | — |
| `t6_cli_e2e` (CLI, 5 segments) | en+hi | 34.85 s | 0.945 | −19.0 | 0 | 0.70 | 0.05 | yes | 47.8 s | 1.46 |

\* First inference after model load (warm-up), including MPS kernel compilation.

All files are 24 000 Hz, mono. `t1`–`t5` are written as 32-bit float WAV (exact raw samples);
`t6` is written by the CLI as PCM 16-bit.

**Direct vs adapter:** same length (58 896 samples), identical samples. The adapter adds no
processing and reproduces Coqui exactly for the same parameters and seed.

**Hindi:** the model accepted `language="hi"` without error and produced speech-like audio.
Earlier XTTS-v2 documentation listed 16 languages without Hindi; v2.0.3 lists 17 including Hindi.
Hindi is therefore recorded as **supported by this backend**, pending a listening check of pronunciation.

---

## 4. Performance (MPS)

| Metric | Value |
|---|---|
| `TTS()` construction (checkpoint → CPU) | 17.7 s |
| `.to("mps")` | 0.35 s |
| Adapter `load()` total | 15.6 s (CLI run: 17.6 s) |
| Reload after unload | 15.4 s |
| First inference (warm-up) | 19.8 s for 2.7 s audio (RTF 7.2) |
| Subsequent inference, RTF range | 0.99 – 1.59 (≈ real time) |
| 15-call loop, mean RTF | ≈ 1.15 (generation 6.6–11.2 s for 6.4–9.3 s audio) |
| Speaker conditioning per call | **0.11 s** (recomputed by Coqui every call) |
| MPS allocated with model loaded | 1.868 GB (constant during generation) |
| MPS driver memory | 2.19 GB → 2.39 GB over 15 varied calls |
| MPS memory after unload | 0.0 GB allocated, ~0.09 GB driver |

Conditioning costs ~0.11 s versus 2.4–11 s of generation (~1–4 %). Caching it is **not worth the
added complexity** now; audit item S8 stays deferred.

---

## 5. Memory stability — open issue

| Scenario | Process RSS | MPS allocated |
|---|---|---|
| `get_conditioning_latents` × 20 | flat (1.362 GB) | — |
| `Xtts.inference` (precomputed latents) × 8, same text | flat (~1.1–1.2 GB) | — |
| `TTS.tts` × 8, same text, default params | +0.016 GB | — |
| `TTS.tts` × 8, same text, legacy params | +0.008 GB | — |
| `XTTSAdapter.synthesize_result` × 8, same text | +0.003 GB | — |
| `XTTSAdapter.synthesize_result` × 15, **different text each call** | **2.49 → 4.00 GB (≈ +0.1 GB/call)** | flat, 1.868 GB |

Interpretation: no growth when the input is identical, so the adapter and Coqui do not leak
per call. Growth appears only with varying input lengths while device memory stays flat. The
likely cause is PyTorch/MPS caching kernels per input shape on the CPU side, but this is a
**hypothesis**: whether RSS plateaus after enough distinct lengths was not measured.
**Next step:** a ~100-call varied-text run on MPS and on CPU to see whether it plateaus and
whether it is MPS-specific. Until then, long scripts should be generated in one process with
headroom (24 GB machine; 4 GB observed after 15 calls), or split across processes.

---

## 6. Legacy compatibility (`clone_voice.py`)

| Aspect | Status |
|---|---|
| Model, inference parameters, seeding, retries | identical |
| Short-line merging, pauses, chunk limits, inside-line pause | identical (restored, pinned by tests) |
| Output sample rate / format | identical (24 kHz, PCM 16-bit in the CLI) |
| Reference validation | identical checks + permission and NaN checks |
| **Device** | **differs:** legacy CPU, VoxFoundry MPS by default (`--device cpu` reproduces legacy) |
| **Per-clip cleanup** | **differs:** legacy always applies trim/fade/RMS; VoxFoundry outputs raw audio, `--legacy-clip-cleanup` reproduces legacy |
| ffmpeg polish to MP3 | not in VoxFoundry (mastering is out of scope) |
| Output cache | legacy always on (`output/cache`); VoxFoundry opt-in (`--cache-dir`) |

Not done: an audible A/B of the same script through `clone_voice.py` and the VoxFoundry CLI.

---

## 7. Manual listening

| File | Speaker identity | Pronunciation | Pacing / flow | Pauses / breathing | Artifacts |
|---|---|---|---|---|---|
| `t2_adapter_en_mps.wav` | "close but not perfect" (user) | not reported | not reported | not reported | not reported |
| `t4_hinglish_as_en_mps.wav` | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| `t4_hindi_hi_mps.wav` | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| `t6_cli_e2e_mps.wav` | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |

Speaker-similarity options for a later phase (none applied, per Phase 2.5 rule 11): a clean
15–30 s reference (the current one is 42 s, and XTTS uses only the first 30 s), several
reference clips, and temperature/speed tuning.

---

## 8. Remaining risks

1. Memory growth with varied input (§5).
2. CPML licence: XTTS-v2 output is **non-commercial only** (`docs/XTTS_MODEL_COMPATIBILITY.md` §2).
3. `scripts/script.txt` uses VoxCPM2 section markers (`[HOOK]` etc.) that the XTTS parser rejects.
4. The project root is itself a legacy venv; `run.sh` has pre-existing bugs (audit R4, R6).
5. CPU path untested with the real model in this phase.
