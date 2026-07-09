#!/bin/bash
# ============================================================
#  OpenBCI 注意力检测一键启动脚本
#  硬件: Cyton + Daisy (16通道)  |  输出: UDP Focus → 127.0.0.1:12345
#  ARM64 适配: 自动修补 brainflow JAR 中的 x86-64 原生库
# ============================================================

# ---- 路径配置 ----
BASE_DIR="/home/elf/Projects/bci"
GUI_LAUNCHER="$BASE_DIR/OpenBCI_GUI_source/build_output/OpenBCI_GUI"
GUI_DATA_DIR="$BASE_DIR/OpenBCI_GUI"
RECEIVER_SCRIPT="$BASE_DIR/focus_receiver.py"
SETTINGS_SRC="$GUI_DATA_DIR/Settings/DaisyUserSettings.json"
GUI_LOG="/tmp/openbci_gui.log"
BRAINFLOW_JAR="$BASE_DIR/OpenBCI_GUI_source/build_output/lib/brainflow.jar"
BRAINFLOW_JAR_BAK="$BASE_DIR/OpenBCI_GUI_source/build_output/lib/brainflow.jar.bak"
PY_BRAINFLOW_LIB="/home/elf/miniforge3/envs/eeg/lib/python3.10/site-packages/brainflow/lib"
SERIAL_DEVICES=("/dev/ttyUSB0" "/dev/ttyUSB1" "/dev/ttyACM0" "/dev/ttyACM1")

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}   OpenBCI 注意力检测系统 - 一键启动${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# ---- ❿ 清理残留进程 ----
cleanup_procs() {
    local killed=0
    for proc in $(ps aux | grep -iE "[O]penBCI_GUI|[f]ocus_receiver" | awk '{print $2}'); do
        kill "$proc" 2>/dev/null && killed=1
    done
    [ $killed -eq 1 ] && sleep 1 && echo -e "${YELLOW}[*] 已清理残留进程${NC}"
}
cleanup_procs

# ---- ❶ 修补 brainflow JAR（ARM64 适配） ----
patch_brainflow_jar() {
    # 快速检查：JAR 内的 libBoardController.so 是否已是 ARM64
    local tmp_check="/tmp/opencode_bci_check_$$"
    mkdir -p "$tmp_check"
    cd "$tmp_check"
    jar xf "$BRAINFLOW_JAR" brainflow/libBoardController.so 2>/dev/null
    local arch=$(file brainflow/libBoardController.so 2>/dev/null | grep -o 'ARM aarch64')
    rm -rf "$tmp_check"

    if [ -n "$arch" ]; then
        echo -e "${GREEN}[√] brainflow JAR 已是 ARM64，无需修补${NC}"
        return 0
    fi

    echo -e "${YELLOW}[*] 检测到 x86-64 原生库，正在修补为 ARM64...${NC}"

    if [ ! -d "$PY_BRAINFLOW_LIB" ]; then
        echo -e "${RED}[错误] 找不到 Python brainflow 原生库: $PY_BRAINFLOW_LIB${NC}"
        return 1
    fi

    # 备份原始 JAR
    if [ ! -f "$BRAINFLOW_JAR_BAK" ]; then
        cp "$BRAINFLOW_JAR" "$BRAINFLOW_JAR_BAK"
    fi

    # 解压 → 替换 ARM64 .so → 重新打包
    local tmp="/tmp/opencode_bci_jar_$$"
    rm -rf "$tmp"; mkdir -p "$tmp"; cd "$tmp"
    jar xf "$BRAINFLOW_JAR"

    local libs=("libBoardController.so" "libBrainBitLib.so" "libDataHandler.so"
                "libGanglionLib.so" "libMLModule.so" "libMuseLib.so")
    local count=0
    for lib in "${libs[@]}"; do
        if [ -f "$PY_BRAINFLOW_LIB/$lib" ] && [ -f "brainflow/$lib" ]; then
            cp "$PY_BRAINFLOW_LIB/$lib" "brainflow/$lib" && ((count++))
        fi
    done

    jar cf /tmp/brainflow_patched_$$.jar .
    cp /tmp/brainflow_patched_$$.jar "$BRAINFLOW_JAR"
    rm -rf "$tmp" /tmp/brainflow_patched_$$.jar

    # 清除 JNA 缓存中的 x86-64 库
    for lib in "${libs[@]}"; do
        rm -f "$HOME/.cache/JNA/temp/$lib" 2>/dev/null
    done

    echo -e "${GREEN}[√] JAR 已修补 ($count 个库 → ARM64)${NC}"
}
patch_brainflow_jar

# ---- ❶B 修补 ComPortBox 字节码（ARM64 串口名称适配） ----
patch_comportbox() {
    local gui_jar="$BASE_DIR/OpenBCI_GUI_source/build_output/lib/OpenBCI_GUI.jar"
    local cls_name="OpenBCI_GUI\$ComPortBox.class"

    # 快速检查：是否已修补（检查常量池中是否有 "USB" 替代 "VCP"）
    local tmp_check="/tmp/opencode_comport_check_$$"
    mkdir -p "$tmp_check" && cd "$tmp_check"
    jar xf "$gui_jar" "$cls_name" 2>/dev/null
    if strings "$cls_name" | grep -q $'^USB$'; then
        rm -rf "$tmp_check"
        return 0
    fi
    rm -rf "$tmp_check"

    echo -e "${YELLOW}[*] 修补串口检测代码 (VCP→USB)...${NC}"

    local tmp="/tmp/opencode_comport_$$"
    mkdir -p "$tmp" && cd "$tmp"
    jar xf "$gui_jar" "$cls_name" 2>/dev/null

    python3 -c "
data = open('$cls_name', 'rb').read()
# 替换常量池 Utf8: 'VCP' → 'USB' (同为 3 字节)
old = bytes([0x01, 0x00, 0x03, 0x56, 0x43, 0x50])
new = bytes([0x01, 0x00, 0x03, 0x55, 0x53, 0x42])
if data.count(old) == 1:
    data = data.replace(old, new)
    open('$cls_name', 'wb').write(data)
    print('OK')
else:
    print('SKIP')
" 2>&1

    jar uf "$gui_jar" "$cls_name"
    rm -rf "$tmp"
    echo -e "${GREEN}[√] 串口检测代码已适配 ARM64${NC}"
}
patch_comportbox

# ---- ❷ 检查串口设备 ----
FOUND=""
for dev in "${SERIAL_DEVICES[@]}"; do
    if [ -e "$dev" ]; then FOUND="$dev"; break; fi
done
if [ -z "$FOUND" ]; then
    echo -e "${RED}[错误] 未检测到 OpenBCI 串口${NC}"
    echo "  预期的设备路径: ${SERIAL_DEVICES[*]}"
    ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "  (无)"
    exit 1
fi
echo -e "${GREEN}[√] 检测到串口: ${FOUND}${NC}"

# ---- ❸ 设置串口权限 ----
if [ ! -w "$FOUND" ]; then
    sudo chmod a+rw "$FOUND" 2>/dev/null && \
        echo -e "${GREEN}[√] 串口权限已设置${NC}" || \
        echo -e "${YELLOW}[!] 无法自动设置权限${NC}"
fi

# ---- ❹ 同步用户设置文件 ----
mkdir -p "$GUI_DATA_DIR/Settings"
cp "$SETTINGS_SRC" "$GUI_DATA_DIR/Settings/DaisyUserSettings.json" 2>/dev/null || true
echo -e "${GREEN}[√] 用户设置已就绪（UDP Focus → 127.0.0.1:12345）${NC}"

# ---- ❺ 启动 OpenBCI GUI ----
echo ""
echo -e "${CYAN}[*] 正在启动 OpenBCI GUI...${NC}"
cd "$BASE_DIR"
"$GUI_LAUNCHER" &>"$GUI_LOG" &
disown
echo -e "${GREEN}[√] GUI 已后台启动 (日志: ${GUI_LOG})${NC}"

# ---- ❻ 启动 Focus 接收器 ----
echo -e "${CYAN}[*] 启动 Focus UDP 接收器...${NC}"
if command -v gnome-terminal &>/dev/null; then
    gnome-terminal --title="Focus Receiver" -- bash -c "
        echo -e '\033[0;36m========================================\033[0m'
        echo -e '\033[0;36m  Focus UDP 接收器 (端口 12345)\033[0m'
        echo -e '\033[0;36m========================================\033[0m'
        echo ''
        python3 '$RECEIVER_SCRIPT'
        echo ''
        read -p '按回车键关闭此窗口...'
    " &
elif command -v xterm &>/dev/null; then
    xterm -T "Focus Receiver" -e bash -c "
        python3 '$RECEIVER_SCRIPT'; read -p '按回车键关闭...'
    " &
else
    echo -e "${YELLOW}[!] 未找到终端模拟器，请手动运行: python3 $RECEIVER_SCRIPT${NC}"
fi

# ---- ❼ 操作说明 ----
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}   请在 OpenBCI GUI 中完成以下操作:${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""
echo -e "  ${GREEN}步骤 1${NC}  左侧面板选择数据源: ${YELLOW}Cyton (with Daisy)${NC}"
echo -e "  ${GREEN}步骤 2${NC}  点击: ${YELLOW}START SESSION${NC}"
echo -e "              → GUI 自动扫描并连接 ${FOUND}"
echo -e "  ${GREEN}步骤 3${NC}  等待初始化完成 (约 5 秒)"
echo -e "              → 看到波形后，按 ${YELLOW}空格键${NC} 启动数据流"
echo -e "              → 按 ${YELLOW}Shift+N${NC}  加载预设网络配置 (UDP/Focus)"
echo ""
echo -e "  ${GREEN}Widget${NC}    右侧下拉添加 ${YELLOW}Focus${NC} 面板 → 查看注意力值"
echo -e "              下拉添加 ${YELLOW}Networking${NC} → 开启 UDP 发送"
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${GREEN}启动完成。GUI 日志: ${GUI_LOG}${NC}"
echo ""
