#!/bin/bash
set -e
# Run the updated pipeline
source .venv-multiengine/bin/activate
source .venv-multiengine/bin/activate
source bin/activate
python clone_voice.py
python clone_voice_with_chatterbox.py.py
python -u clone_voice_with_voxcpm2.py
python3 voiceover_master.py input.m4a -o mastered.wav --noise-profile-seconds 0.5
python voiceover_master.py "input.m4a" -o "final_voiceover.wav"

generate_voxcpm2.py


ffmpeg -i reference/my_voice.m4a -ar 22050 -ac 1 reference/my_voice.wav
# voiceclone/references/my_voice.m4a
