#!/bin/bash
# Voice Debug Panel launcher
export PATH="/home/elf/miniforge3/envs/eeg/bin:$PATH"
export PYTHONPATH="/home/elf/Projects:$PYTHONPATH"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
cd /home/elf/Projects
exec /home/elf/miniforge3/envs/eeg/bin/python3 /home/elf/Projects/voice_assistant/debug_panel.py
