#!/bin/bash
# Set up audio for voice assistant - unmute and set volumes
# Speaker
amixer -c 1 sset 'Headphone' 50 unmute >/dev/null 2>&1
amixer -c 1 sset 'Headphone ZC' on >/dev/null 2>&1
amixer -c 1 sset 'Speaker' 63 unmute >/dev/null 2>&1
amixer -c 1 sset 'PCM' 255 >/dev/null 2>&1
# Mic
amixer -c 1 sset 'PGA' 28 >/dev/null 2>&1
amixer -c 1 sset 'PGA Boost' 1 >/dev/null 2>&1
amixer -c 1 sset 'Aux Boost' 3 >/dev/null 2>&1
amixer -c 1 sset 'Right Input Mixer MicP' on >/dev/null 2>&1
amixer -c 1 sset 'Left Input Mixer MicP' on >/dev/null 2>&1
amixer -c 1 sset 'Main Mic' on >/dev/null 2>&1
amixer -c 1 sset 'Left Output Mixer LDAC' on >/dev/null 2>&1
amixer -c 1 sset 'Right Output Mixer LDAC' on >/dev/null 2>&1
amixer -c 1 sset 'Right Output Mixer RDAC' on >/dev/null 2>&1
amixer -c 1 sset 'Left Output Mixer RDAC' on >/dev/null 2>&1
echo "Audio setup done (PGA=28, Boost=1)"
