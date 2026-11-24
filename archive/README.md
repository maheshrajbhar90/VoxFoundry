# Archive

Superseded files, kept for reference only. They are **not maintained** and are not part of the
`clonemyvoice` package.

| Path | What it is |
|---|---|
| `legacy_scripts/clone_voice.py` | Original XTTS-v2 script. CloneMyVoice's defaults reproduce it. |
| `legacy_scripts/voiceover_master.py` | Noise reduction / loudness mastering of a recorded voiceover. |
| `legacy_scripts/run.sh` | Old ad-hoc run commands (known to be broken). |
| `requirements/` | Earlier dependency lists: the multi-engine venv freeze, the mastering deps, and the XTTS lock. |
| `scripts/` | Earlier voiceover scripts. |

The Chatterbox and VoxCPM2 engine scripts are **active** and live in the repository root.

The legacy scripts resolve paths from the repository root, so they run from here with the
Python 3.11 `.venv-multiengine` environment, for example
`python archive/legacy_scripts/clone_voice.py`.
