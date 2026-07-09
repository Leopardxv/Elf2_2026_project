#!/bin/bash
# download_models.sh — Download sherpa-onnx voice assistant models
# ====================================================================
# NOTE: GitHub must be accessible. If blocked:
#   1. Download on another machine and transfer via SCP/USB
#   2. Place files in ~/Projects/models/sherpa/{kws,asr,tts}/
#
# Model sources (download these on a machine with GitHub access):
#   KWS: github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
#        → sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2 (32MB)
#   ASR: github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
#        → sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 (68MB)
#   TTS: github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
#        → sherpa-onnx-vits-melo-tts-zh-en.tar.bz2 (100MB)
#
# After downloading, extract each to:
#   ~/Projects/models/sherpa/kws/   (tokens.txt, encoder*.onnx, decoder*.onnx, joiner*.onnx)
#   ~/Projects/models/sherpa/asr/   (tokens.txt, encoder*.onnx, decoder*.onnx, joiner*.onnx)
#   ~/Projects/models/sherpa/tts/   (model.onnx, tokens.txt, lexicon.txt, dict/)
# ====================================================================

set -e

MODEL_DIR="$HOME/Projects/models/sherpa"
TMP="/tmp/sherpa_dl"

mkdir -p "$MODEL_DIR"/{kws,asr,tts} "$TMP"

BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"

download_and_extract() {
    local tag="$1" file="$2" dest="$3" label="$4"

    if [ -f "$dest/tokens.txt" ] || [ -f "$dest/model.onnx" ]; then
        echo "  [$label] Already present. Skipping."
        return 0
    fi

    echo "  [$label] Downloading ${file}..."
    local url="${BASE}/${tag}/${file}"

    wget -c --timeout=300 --tries=3 -O "$TMP/${file}" "$url" 2>&1 || {
        echo "  [$label] Download FAILED!"
        echo "  Download manually from: ${url}"
        echo "  And extract to: ${dest}"
        return 1
    }

    echo "  [$label] Extracting..."
    tar xf "$TMP/${file}" -C "$dest" --strip-components=1 2>/dev/null || \
    tar xf "$TMP/${file}" -C "$dest" 2>/dev/null || {
        echo "  [$label] Extract failed — try: tar xf $TMP/${file} -C $dest"
        return 1
    }
    rm -f "$TMP/${file}"
    echo "  [$label] Done ($(du -sh "$dest" 2>/dev/null | cut -f1))"
}

echo "=== Voice Assistant Model Download ==="
echo "Target: $MODEL_DIR"
echo ""

download_and_extract "kws-models" \
    "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2" \
    "$MODEL_DIR/kws" "KWS"

download_and_extract "asr-models" \
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2" \
    "$MODEL_DIR/asr" "ASR"

download_and_extract "tts-models" \
    "vits-melo-tts-zh_en.tar.bz2" \
    "$MODEL_DIR/tts" "TTS"

rm -rf "$TMP"
echo ""
echo "=== Done ==="
echo "Models in: $MODEL_DIR"
echo "Run: bash ~/Projects/voice_assistant/start_voice.sh"
