#!/bin/sh
# Включает basic auth, если заданы BASIC_AUTH_USER и BASIC_AUTH_PASSWORD (публичный
# инстанс на ВМ, 02-plan.md этап 4). Локально переменные пустые — nginx без пароля.
set -e
CONF=/etc/nginx/basic-auth.conf
if [ -n "$BASIC_AUTH_USER" ] && [ -n "$BASIC_AUTH_PASSWORD" ]; then
    htpasswd -bc /etc/nginx/.htpasswd "$BASIC_AUTH_USER" "$BASIC_AUTH_PASSWORD" >/dev/null 2>&1
    printf 'auth_basic "reid";\nauth_basic_user_file /etc/nginx/.htpasswd;\n' > "$CONF"
    echo "basic auth: включён для пользователя $BASIC_AUTH_USER"
else
    : > "$CONF"
    echo "basic auth: выключен (BASIC_AUTH_USER/BASIC_AUTH_PASSWORD не заданы)"
fi
