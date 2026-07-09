#!/usr/bin/env python3
"""
Check if voice assistant models are ready.
"""
import os
import sys

_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models")


def check_models():
    all_ok = True

    # Vosk ASR model
    vosk_path = os.path.join(_MODEL_ROOT, "vosk", "vosk-model-small-cn-0.22")
    vosk_ok = os.path.isdir(vosk_path) and os.path.isdir(os.path.join(vosk_path, "am"))
    status = "[OK]     " if vosk_ok else "[MISSING]"
    print(f"  {status} Vosk ASR: {vosk_path}")
    if not vosk_ok:
        all_ok = False

    # TTS model (sherpa-onnx VITS - this one works on ARM)
    tts_model = os.path.join(_MODEL_ROOT, "sherpa", "tts", "model.onnx")
    tts_ok = os.path.isfile(tts_model)
    status = "[OK]     " if tts_ok else "[MISSING]"
    print(f"  {status} TTS: {tts_model}")
    if not tts_ok:
        all_ok = False

    # LLM model
    llm_path = os.path.join(_MODEL_ROOT, "qwen2.5-0.5b-instruct-q2_k.gguf")
    llm_ok = os.path.isfile(llm_path)
    status = "[OK]     " if llm_ok else "[MISSING]"
    print(f"  {status} LLM: {llm_path}")
    if not llm_ok:
        all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=== Voice Assistant Model Check ===")
    ok = check_models()
    if ok:
        print("\nAll models ready!")
        sys.exit(0)
    else:
        print("\nModels missing!")
        print("  Vosk: download from https://alphacephei.com/vosk/models")
        print(f"  Place in: {_MODEL_ROOT}/vosk/")
        sys.exit(1)
