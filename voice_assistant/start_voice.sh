#!/bin/bash
# start_voice.sh — Launch the 精灵精灵 voice assistant
# Usage: bash ~/Projects/voice_assistant/start_voice.sh

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$DIR")"

echo "================================"
echo " 精灵精灵 — Voice Assistant"
echo "================================"
echo ""

# Activate conda
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
conda activate eeg

# Check dependencies
echo "[1/3] Checking dependencies..."
python3 -c "import sherpa_onnx; import sounddevice" 2>/dev/null || {
    echo "Missing packages. Installing..."
    pip install sherpa-onnx sounddevice -q
}
echo "  Dependencies OK"

# Check models
echo "[2/3] Checking models..."
python3 "$DIR/check_models.py" || {
    echo ""
    echo "Models not found!"
    echo ""
    echo "Option 1: Download directly (may take 30-60 min):"
    echo "  bash $DIR/download_models.sh"
    echo ""
    echo "Option 2: Download on another machine and copy to:"
    echo "  $PROJECT_ROOT/models/sherpa/"
    echo ""
    echo "Required:"
    echo "  kws/tokens.txt, kws/*.onnx   (KWS: ~32MB)"
    echo "  asr/tokens.txt, asr/*.onnx   (ASR: ~68MB)"
    echo "  tts/model.onnx, tts/tokens.txt (TTS: ~100MB)"
    exit 1
}

# Launch
echo "[3/3] Starting voice agent..."
echo ""
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH" python3 -m voice_assistant.voice_agent
