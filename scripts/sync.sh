#!/usr/bin/env bash
# Отправить код на сервер (без .venv и результатов): bash scripts/sync.sh
set -e
HOST=${QHNN_HOST:-Sixteen@192.168.1.95}
cd "$(dirname "$0")/.."
tar -czf /tmp/qhnn_code.tgz qhnn configs scripts tests pyproject.toml README.md
scp -q -o BatchMode=yes /tmp/qhnn_code.tgz "$HOST":D:/qhnn_code.tgz
ssh -o BatchMode=yes "$HOST" "tar -xzf D:\qhnn_code.tgz -C D:\qhnn && del D:\qhnn_code.tgz && echo synced"
