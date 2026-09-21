#!/usr/bin/env bash
# Обновление сайта через этот ПК: сервер (V100) -> export JSON -> VPS.
# bash scripts/relay_publish.sh [интервал_сек]
SRV=${QHNN_HOST:-Sixteen@192.168.1.95}
VPS=${QHNN_VPS:-root@195.133.75.140}
INT=${1:-90}
TMP=$(mktemp -d)
while true; do
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$SRV" \
       "cd /d D:\qhnn && .venv-cpu\Scripts\python.exe -m qhnn.experiments.export_site --out D:\qhnn\site\data" >/dev/null 2>&1 \
     && scp -q -o BatchMode=yes "$SRV:D:/qhnn/site/data/*.json" "$TMP/" \
     && scp -q -o BatchMode=yes "$TMP"/*.json "$VPS:/var/www/qhnn/data/"; then
    echo "$(date +%H:%M:%S) published $(ls "$TMP" | wc -l) files"
  else
    echo "$(date +%H:%M:%S) publish FAILED"
  fi
  sleep "$INT"
done
