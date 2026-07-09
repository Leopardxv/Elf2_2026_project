#!/usr/bin/env python3
"""
Quick audio hardware test — record 3s, play it back.

Useful for verifying mic + speaker before running voice agent.
"""
import sys
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 3.0

def test_audio():
    print("=== Audio Hardware Test ===")

    # Test mic
    print(f"\n[MIC] Recording {DURATION}s... (speak now)")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=CHANNELS,
                   dtype='float32', device=0)
    sd.wait()
    peak = np.abs(audio).max()
    print(f"[MIC] Peak: {peak:.4f} (should be >0.01 for voice)")

    if peak < 0.005:
        print("[WARN] Mic signal very weak — check hardware or gain")
        return False

    # Test speaker
    print(f"\n[SPK] Playing back your recording...")
    try:
        audio_out = audio.reshape(-1, 1)
        sd.play(audio_out * 0.8, samplerate=SAMPLE_RATE, device=3)
        sd.wait()
        print("[SPK] Playback complete")
    except Exception as e:
        print(f"[SPK] Error: {e}")
        return False

    print("\n=== Hardware OK ===")
    return True


if __name__ == "__main__":
    ok = test_audio()
    sys.exit(0 if ok else 1)
