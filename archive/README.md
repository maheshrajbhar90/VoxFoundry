# Archive

Superseded files, kept for reference only. They are **not maintained** and are not part of the
`voxfoundry` package.

| Path | What it is |
|---|---|
| `legacy_scripts/clone_voice.py` | Original XTTS-v2 script. VoxFoundry's defaults reproduce it. |
| `legacy_scripts/clone_voice_with_chatterbox.py` | Experiment with the Chatterbox TTS engine. |
| `legacy_scripts/clone_voice_with_voxcpm2.py` | Experiment with the VoxCPM2 engine (uses `[HOOK]`-style section markers). |
| `legacy_scripts/voiceover_master.py` | Noise reduction / loudness mastering of a recorded voiceover. |
| `legacy_scripts/run.sh` | Old ad-hoc run commands (known to be broken). |
| `requirements/` | Earlier dependency lists: the multi-engine venv freeze, the mastering deps, and the XTTS lock. |
| `scripts/` | Earlier voiceover scripts. |

The legacy scripts resolve paths from the repository root, so they run from here with the
Python 3.11 `.venv-multiengine` environment, for example
`python archive/legacy_scripts/clone_voice.py`.
