# XTTS-v2 — Model, Licence, Storage and Compatibility

**Date:** 2026-11-21
**Model id:** `tts_models/multilingual/multi-dataset/xtts_v2`
**Sources:** coqui-tts 0.28.0 `TTS/.models.json`, Hugging Face repo `tts-hub/XTTS-v2` (model card,
`LICENSE.txt`, file tree API), the downloaded `config.json`, installed Coqui source.

---

## 1. Model identity

| Field | Value | Source |
|---|---|---|
| Registry description | "XTTS-v2.0.3 by Coqui with 17 languages." | `.models.json` |
| Hosted at | `huggingface.co/tts-hub/XTTS-v2` (base model `coqui/XTTS-v2`) | `.models.json`, model card |
| `tos_required` | `true` | `.models.json` |
| Output sample rate | 24 000 Hz | `config.json` (`audio.output_sample_rate`) |
| `gpt_max_text_tokens` | 402 | `config.json` |
| Paper | arXiv 2406.04904 | model card |

## 2. Licences — weights vs software

| Component | Licence | Commercial use |
|---|---|---|
| **Model weights and their outputs** (`model.pth`, etc.) | **Coqui Public Model License 1.0.0 (CPML)** — `https://tts-hub.github.io/cpml` | **Not permitted.** |
| **Software** (`coqui-tts` 0.28.0, idiap fork) | **MPL-2.0** | Permitted under MPL-2.0 terms. |

The MPL-2.0 software licence does **not** extend to the weights. Using `coqui-tts` commercially
does not make XTTS-v2 output usable commercially.

### CPML terms (summary of `LICENSE.txt`)

* Licence is granted **only for non-commercial purposes**, covering the model **and its output**.
* Non-commercial includes: personal research, experiment and testing; personal study; private
  entertainment; hobby/amateur projects; testing, evaluation and non-commercial R&D by for-profit
  companies; charitable use.
* Excluded: any use where you receive **direct or indirect payment** from the model or its output;
  training other models for commercial use; revenue-generating activity, including
  government-grant-funded projects.
* **Attribution / notice:** anyone who receives a copy of the model, a modification, **or its
  output** from you must also receive the licence terms or their URL.
* No sublicensing or transfer. Patent licence ends if you assert a patent claim against the model.
* 30-day cure period after the first written notice of violation.

**Implication for CloneMyVoice:** any voiceover published on a monetised channel, or produced for a
client, falls outside the CPML. XTTS-v2 is suitable for evaluation and non-commercial work only.
Commercial use would need a different backend or a separate licence from the rights holder.

Acceptance on this machine: `~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2/tos_agreed.txt`
("I have read, understood and agreed to the Terms and Conditions."), dated 2026-11-14.

## 3. Files, size and storage

| File | Size | Downloaded by Coqui |
|---|---|---|
| `model.pth` | 1 867.9 MB | yes |
| `speakers_xtts.pth` | 7.8 MB | yes |
| `vocab.json` | 0.36 MB | yes |
| `config.json`, `mel_stats.pth`, `hash.md5`, `LICENSE.txt`, `README.md` | < 0.01 MB each | yes |
| `dvae.pth` | 210.5 MB | **no** (in the ignore list; only needed for training) |
| `samples/*` | ~2.7 MB | **no** |
| **Total download** | **≈ 1.88 GB** | |

* **Location used by coqui-tts 0.28.0:** the Hugging Face hub cache,
  `~/.cache/huggingface/hub/models--tts-hub--XTTS-v2/snapshots/345b56f6fbe25cca7103f7f34e471b8fe8e4945f/`.
  The legacy folder `~/Library/Application Support/tts/…xtts_v2/` holds only `tos_agreed.txt`.
* **Status:** **already present**, downloaded 2026-11-14. There is exactly one copy on disk.
* **Integrity:** SHA-256 of `model.pth` = `d5822cb1d4bf97c418855e326562a934caa20cef01ebf4c90bf792df0fd5b99e`,
  identical to the LFS object id published by Hugging Face. (`hash.md5` is Coqui's update marker,
  not a checksum of `model.pth`, so it does not match the file's MD5 — this is expected.)
* **Free disk space:** 680 GiB.

**No download is required for Phase 2.5.**

## 4. Memory

* Weights are ~1.87 GB in fp32. Loaded model plus inference buffers are expected to need
  **roughly 2–4 GB** of unified memory. This is an **estimate**, not a measurement; Phase H
  will measure it. The 24 GB machine has ample headroom.

## 5. Languages

`config.json` and the model card both list **17** languages:

`en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi`

* **Hindi (`hi`) is listed** for v2.0.3. Earlier XTTS-v2 releases documented 16 languages without
  Hindi; the Phase 2.5 brief reflects that older documentation. Hindi will be **tested** with
  `language="hi"` and reported from actual results; it is not assumed to work or to fail.
* Per-language character limits in the Coqui tokenizer (warns above these):
  en 250, hi 150, de 253, fr 273, es 239, it 213, pt 203, pl 224, zh 82, ar 166, cs 186, ru 182,
  nl 251, tr 226, ja 71, hu 224, ko 95. Legacy CloneMyVoice chunk limits are stricter: en 200, hi 130.
* Validation in Coqui is an `assert` in `Xtts.synthesize`; it is skipped under `python -O`.
  CloneMyVoice should validate languages itself (audit item A10).
* Romanized Hinglish has no language code; it can only be sent as `en` (experimental).

## 6. Installed API (coqui-tts 0.28.0, verified in source)

| Concern | Finding |
|---|---|
| Construction | `TTS(model_name, progress_bar=…)` loads on CPU; `.to(device)` moves it. |
| High-level inference | `TTS.tts(text, speaker_wav=…, language=…, **kwargs)` → `Synthesizer.tts` → `Xtts.synthesize`. |
| Tuning kwargs | `Xtts.synthesize(**kwargs)` accepts `gpt_cond_len`, `gpt_cond_chunk_len`, `max_ref_len`, `sound_norm_refs` (conditioning) and passes `temperature`, `top_p`, `top_k`, `repetition_penalty`, `length_penalty`, `speed`, `enable_text_splitting` to `Xtts.inference`. All legacy parameters are accepted, so the legacy `TypeError` fallback is never needed. |
| Speaker conditioning | Without a `speaker` id, `synthesize` calls `clone_voice(speaker_wav, …)` → `get_conditioning_latents(...)` **on every call**. `get_conditioning_latents(audio_path, max_ref_length=30, gpt_cond_len=6, gpt_cond_chunk_len=6, librosa_trim_db=None, sound_norm_refs=False, load_sr=22050)` can be called once and the latents reused with `Xtts.inference`. |
| Reference audio | Loaded at 22 050 Hz; should be > 3 s (Coqui docstring); first `max_ref_len` seconds used. |
| Device selection | Not automatic in Coqui; the caller chooses with `.to()`. |

## 7. Dependency compatibility

| Dependency | Requirement | Finding |
|---|---|---|
| Python | coqui-tts 0.28.0: `Requires-Python >=3.10,<3.15` (classifiers 3.10–3.14) | 3.12 OK (was installed). |
| transformers | coqui-tts: `>=4.57` | transformers **5.x requires torch ≥ 2.6** and disables PyTorch otherwise. |
| torch / torchaudio | coqui-tts: `>=2.2` | **torch 2.4.1 + transformers 5.23.0 failed** at XTTS import (`ImportError: cannot import name 'TransformGetItemToIndex'`). **torch 2.6.0 + transformers 5.2.0 imports successfully.** |
| numpy | coqui-tts: `>=1.26` | librosa 1.0.1 and scipy 1.18.1 need numpy ≥ 2 → numpy 2.5.5. |
| MPS | torch 2.6.0 reports MPS built and available | Operator compatibility for XTTS on MPS is **not yet tested**. |

**Verified working pin set (import level only):** Python 3.12, coqui-tts 0.28.0, torch 2.6.0,
torchaudio 2.6.0, transformers 5.2.0, numpy 2.5.5. Real inference has not yet been run.

**Not compatible:** torch 2.4.1 / torchaudio 2.4.1 together with transformers ≥ 5.

## 8. Open concerns

1. `.venv-xtts` was deleted outside this session and must be recreated.
2. XTTS on MPS may hit unsupported operators; it will be tested, and any CPU run will be
   requested and reported separately.
3. transformers 5.x is a new major version for the XTTS GPT generation path; only real inference
   can confirm it works.
4. CPML forbids commercial use of the output (§2).
