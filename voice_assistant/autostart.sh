#!/bin/bash
LOG="$HOME/voice_assistant.log"
exec >> "$LOG" 2>&1
echo "=== $(date) Starting voice assistant ==="

# Connect JBL
for i in $(seq 1 20); do
    if bluetoothctl info 64:B9:71:50:08:4C 2>/dev/null | grep -q "Connected: yes"; then
        echo "JBL connected (attempt $i)"
        break
    fi
    bluetoothctl connect 64:B9:71:50:08:4C 2>/dev/null
    sleep 2
done

sleep 3

# Force A2DP profile
for i in $(seq 1 10); do
    JBL=$(pactl list sinks short 2>/dev/null | grep bluez | grep a2dp | awk '{print $2}')
    if [ -n "$JBL" ]; then
        pactl set-default-sink "$JBL"
        echo "JBL A2DP: $JBL"
        break
    fi
    HFP=$(pactl list sinks short 2>/dev/null | grep bluez | grep handsfree | awk '{print $2}')
    if [ -n "$HFP" ]; then
        echo "JBL in HFP mode, switching to A2DP..."
        pactl set-card-profile bluez_card.64_B9_71_50_08_4C a2dp_sink 2>/dev/null
    fi
    sleep 2
done

bash ~/Projects/voice_assistant/setup_audio.sh
source ~/miniforge3/etc/profile.d/conda.sh
conda activate eeg
cd ~/Projects
PYTHONPATH="$HOME/Projects:$PYTHONPATH" python3 -m voice_assistant.voice_agent

# Set NPU to userspace 1GHz (required for RKLLM)
echo elf | sudo -S bash -c "echo userspace > /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null && echo 1000000000 > /sys/class/devfreq/fdab0000.npu/userspace/set_freq 2>/dev/null" 2>/dev/null
