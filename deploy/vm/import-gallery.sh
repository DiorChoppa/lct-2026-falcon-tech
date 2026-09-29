#!/usr/bin/env bash
# Импорт галереи на ВМ частями (deploy/vm/README.md, «Данные для демо»): CSV датасета
# режется на куски по CHUNK строк, каждый уходит в api-gateway на localhost (мимо таймаута
# прокси) и помечается .done. Импорт не идемпотентен, поэтому после обрыва скрипт
# продолжает с первой непомеченной части, а не с начала.
#
#   cd /srv/lct && sudo -u gitlab-runner deploy/vm/import-gallery.sh dataset/train.csv
#
# Часть, где импортировано меньше строк, чем отправлено, не помечается и останавливает
# скрипт: её записи могли частично лечь в БД — разобраться руками до повтора.
set -euo pipefail

CSV="${1:?CSV датасета: image_id,x,y,w,h[,vehicle_id,camera_id]}"
# Состояние — вне /srv/lct: data/ там принадлежит root (тома Docker), а рабочая копия — CI.
STATE="${2:-$HOME/.cache/lct-import/$(basename "$CSV" .csv)}"
CHUNK="${CHUNK:-500}"
API="${API:-http://127.0.0.1:8080}"

mkdir -p "$STATE"
if [ ! -f "$STATE/.split" ]; then
    head -1 "$CSV" > "$STATE/header"
    tail -n +2 "$CSV" | split -l "$CHUNK" -d -a 3 - "$STATE/part_"
    touch "$STATE/.split"
fi

parts=("$STATE"/part_???)
for part in "${parts[@]}"; do
    [ -f "$part.done" ] && continue
    rows=$(( $(wc -l < "$part") ))
    started=$(date +%s)
    resp=$(cat "$STATE/header" "$part" | curl -sS -F "csv=@-;filename=$(basename "$part").csv" \
        -F images_dir=images "$API/api/gallery/import")
    imported=$(grep -o '"imported":[0-9]*' <<<"$resp" | cut -d: -f2)
    echo "$(date +%T) $(basename "$part"): $imported/$rows за $(( $(date +%s) - started )) с"
    if [ "${imported:-0}" -ne "$rows" ]; then
        echo "$resp" >&2
        exit 1
    fi
    touch "$part.done"
done
echo "готово: $(ls "$STATE"/*.done | wc -l)/${#parts[@]} частей"
