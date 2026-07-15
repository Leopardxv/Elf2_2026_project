#!/usr/bin/env bash
set -euo pipefail

exec 9>/tmp/elf2-server-web.lock
flock -n 9 || exit 0

cd /home/laipengxu/server
mkdir -p runtime

while true; do
  /usr/bin/python3 serve.py >>runtime/server.log 2>&1
  sleep 3
done
