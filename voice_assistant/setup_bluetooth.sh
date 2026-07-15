#!/bin/bash
# Keep the BBH headset connected and make it the only voice I/O route.
set -u

BBH_MAC="5A:4D:F7:B4:5F:25"
BBH_ID="${BBH_MAC//:/_}"
DEVICE_PATH="/org/bluez/hci0/dev_${BBH_ID}"
CARD="bluez_card.${BBH_ID}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export PULSE_SERVER="${PULSE_SERVER:-unix:${XDG_RUNTIME_DIR}/pulse/native}"
export LC_ALL=C

log() {
    printf '[BBH] %s\n' "$*"
}

is_connected() {
    timeout 3s busctl --system get-property org.bluez "$DEVICE_PATH" \
        org.bluez.Device1 Connected 2>/dev/null | grep -q 'b true'
}

wait_for_pulse() {
    local attempt
    for attempt in {1..10}; do
        pactl info >/dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

route_bbh() {
    local profile current_profile sink source input

    pactl list cards short 2>/dev/null | awk -v card="$CARD" '$2 == card {found=1} END {exit !found}' || return 1
    current_profile=$(pactl list cards 2>/dev/null | awk -v card="$CARD" '
        $0 == "Name: " card {found=1}
        found && /^Active Profile:/ {print $3; exit}
    ')

    # A2DP has no microphone. Prefer headset_head_unit so both ASR/KWS input
    # and TTS output are on BBH, even though this profile has voice bandwidth.
    for profile in headset_head_unit handsfree_head_unit; do
        if [ "$current_profile" = "$profile" ] || \
                pactl set-card-profile "$CARD" "$profile" >/dev/null 2>&1; then
            sleep 1
            sink=$(pactl list sinks short 2>/dev/null | awk -v id="$BBH_ID" '$2 ~ /^bluez_(sink|output)\./ && index($2, id) {print $2; exit}')
            source=$(pactl list sources short 2>/dev/null | awk -v id="$BBH_ID" '$2 ~ /^bluez_(source|input)\./ && index($2, id) {print $2; exit}')
            if [ -n "$sink" ] && [ -n "$source" ]; then
                pactl set-default-sink "$sink"
                pactl set-default-source "$source"
                while read -r input; do
                    [ -n "$input" ] && pactl move-sink-input "$input" "$sink" >/dev/null 2>&1 || true
                done < <(pactl list sink-inputs short 2>/dev/null | awk '{print $1}')
                log "ready: profile=$profile sink=$sink source=$source"
                return 0
            fi
        fi
    done

    log "BBH connected, but its microphone profile is not ready"
    return 1
}

connect_bbh() {
    # A stale discovery request and an unfinished A2DP session were causing
    # BlueZ 'resource busy' retries. Stop discovery before a bounded connect.
    timeout 3s busctl --system call org.bluez /org/bluez/hci0 \
        org.bluez.Adapter1 StopDiscovery >/dev/null 2>&1 || true
    timeout 8s busctl --system call org.bluez "$DEVICE_PATH" \
        org.bluez.Device1 Connect >/dev/null 2>&1 || true
}

wait_for_pulse || {
    log "PulseAudio is unavailable"
    exit 1
}

log "managing BBH $BBH_MAC"
while true; do
    if is_connected; then
        if ! route_bbh; then
            # PulseAudio can restart after BlueZ has already reported a
            # connection, which means it never receives the add-device event.
            # A bounded reconnect recreates that event without restarting AI.
            log "refreshing Bluetooth audio discovery"
            timeout 5s bluetoothctl disconnect "$BBH_MAC" >/dev/null 2>&1 || true
            sleep 2
            connect_bbh
            sleep 6
            continue
        fi
        sleep 8
        continue
    fi

    log "connecting"
    connect_bbh
    # BlueZ Connect() is asynchronous. Let it settle before the next try so
    # parallel A2DP/HFP attempts cannot keep the controller in busy state.
    sleep 12
done
