#!/bin/bash
set -u

LOG="$HOME/voice_assistant.log"
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG" 2>/dev/null || echo 0)" -gt 20971520 ]; then
    tail -c 5242880 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
exec >> "$LOG" 2>&1
echo "=== $(date) Starting voice assistant ==="

# BBH routing is maintained by the lightweight bbh-audio.service.
bash "$HOME/Projects/voice_assistant/setup_audio.sh"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate eeg
cd "$HOME/Projects"
exec env PYTHONPATH="$HOME/Projects:${PYTHONPATH:-}" python3 -m voice_assistant.voice_agent
