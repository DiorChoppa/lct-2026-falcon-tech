#!/usr/bin/env bash
# Первичная настройка ВМ (deploy/vm/README.md). Ubuntu 22.04/24.04, x86_64, без GPU,
# пользователь с sudo. Идемпотентен. Код и выкатка — deploy/vm/deploy.sh (job `deploy` в CI).
#
#   DOMAIN=lct.znatalk.ai RUNNER_TOKEN_FILE=~/runner-token bash setup.sh
#
# Переменные:
#   DOMAIN             — домен для TLS (Caddy, Let's Encrypt); A-запись уже смотрит на ВМ
#   APP_DIR            — каталог стека (по умолчанию /srv/lct), владелец gitlab-runner
#   RUNNER_TOKEN_FILE  — файл с токеном glrt-… проектного раннера (тег lct-vm); после
#                        регистрации удаляется. Без него раннер не регистрируется.
set -euo pipefail

DOMAIN="${DOMAIN:?задать DOMAIN}"
APP_DIR="${APP_DIR:-/srv/lct}"
RUNNER_TOKEN_FILE="${RUNNER_TOKEN_FILE:-}"

echo "== Docker"
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
fi
sudo docker compose version >/dev/null

echo "== Git LFS (models/model.onnx, 607 МБ, хранится в LFS)"
if ! command -v git-lfs >/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs
fi

echo "== gitlab-runner"
if ! command -v gitlab-runner >/dev/null; then
    curl -fsSL https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh | sudo bash
    sudo apt-get install -y -qq gitlab-runner
fi
sudo usermod -aG docker gitlab-runner
# Ubuntu-шный ~/.bash_logout чистит терминал и валит shell executor ("prepare environment").
sudo rm -f /home/gitlab-runner/.bash_logout
sudo -u gitlab-runner git lfs install >/dev/null
if [ -n "$RUNNER_TOKEN_FILE" ]; then
    if ! sudo gitlab-runner list 2>&1 | grep -q lct-vm; then
        sudo gitlab-runner register --non-interactive --url https://gitlab.com \
            --token "$(cat "$RUNNER_TOKEN_FILE")" --executor shell --name lct-vm
    fi
    rm -f "$RUNNER_TOKEN_FILE"
fi
sudo systemctl enable --now gitlab-runner >/dev/null

echo "== $APP_DIR"
sudo mkdir -p "$APP_DIR"
sudo chown gitlab-runner:gitlab-runner "$APP_DIR"
if ! sudo test -f "$APP_DIR/.env"; then
    # Публичный инстанс: пароль обязателен, auth в стеке нет (01-concept.md §3)
    # не tr </dev/urandom | head: под pipefail SIGPIPE у tr валит скрипт
    PASS="$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | cut -c1-16)"
    sudo -u gitlab-runner tee "$APP_DIR/.env" >/dev/null <<EOF
# Публичная ВМ: overlay с Caddy/TLS и портами только на localhost подключается всегда.
COMPOSE_FILE=docker-compose.yml:deploy/vm/docker-compose.vm.yml
COMPOSE_PROJECT_NAME=reid
DOMAIN=$DOMAIN
PG_SHARED_BUFFERS=512MB
BASIC_AUTH_USER=jury
BASIC_AUTH_PASSWORD=$PASS
EOF
    sudo chmod 600 "$APP_DIR/.env"
    echo "   basic auth: jury / $PASS   (в $APP_DIR/.env)"
fi

echo
echo "готово. Выкатка: push в master (job deploy) или: sudo -u gitlab-runner $APP_DIR/deploy/vm/deploy.sh"
