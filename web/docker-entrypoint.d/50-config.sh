#!/bin/sh
# Рантайм-конфиг веба: ключ Яндекс Карт из окружения контейнера (YMAPS_KEY) без
# пересборки образа. Ключ JS API и так виден в браузере — защищает его ограничение
# по доменам в кабинете разработчика. Лишние символы вырезаются: в JS попадает
# только [A-Za-z0-9-].
set -e
key=$(printf '%s' "${YMAPS_KEY:-}" | tr -cd 'A-Za-z0-9-')
printf 'window.__APP_CONFIG__ = { ymapsKey: "%s" };\n' "$key" > /usr/share/nginx/html/config.js
if [ -n "$key" ]; then echo "config.js: ключ Яндекс Карт задан"; else echo "config.js: ключ Яндекс Карт не задан — карта будет заглушкой"; fi
