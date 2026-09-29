#!/usr/bin/env bash
# Выкатка стека на ВМ в APP_DIR (по умолчанию /srv/lct): job `deploy` в .gitlab-ci.yml
# (gitlab-runner на ВМ, shell executor) или руками: sudo -u gitlab-runner deploy/vm/deploy.sh
#
# В CI забирает $CI_COMMIT_SHA и веса из LFS по CI_JOB_TOKEN — долгоживущих токенов
# доступа к репозиторию на ВМ нет. Без CI выкатывает то, что уже лежит в APP_DIR.
# Данные (data/pg, data/crops, data/caddy, dataset/images) и .env git не трогает.
set -euo pipefail

APP_DIR="${APP_DIR:-/srv/lct}"
cd "$APP_DIR"
[ -f .env ] || { echo "нет $APP_DIR/.env — сначала deploy/vm/setup.sh" >&2; exit 1; }

if [ -n "${CI_JOB_TOKEN:-}" ]; then
    echo "== код: $CI_COMMIT_SHA"
    url="https://gitlab-ci-token:${CI_JOB_TOKEN}@${CI_SERVER_HOST}/${CI_PROJECT_PATH}.git"
    [ -d .git ] || git init -q
    GIT_LFS_SKIP_SMUDGE=1 git fetch -q --no-tags "$url" "$CI_COMMIT_SHA"
    GIT_LFS_SKIP_SMUDGE=1 git checkout -q --force "$CI_COMMIT_SHA"
    git -c "lfs.url=$url/info/lfs" lfs pull
fi

echo "== веса"
want="$(grep -o '"sha256": *"[0-9a-f]*"' models/model.json | grep -o '[0-9a-f]\{64\}')"
have="$(sha256sum models/model.onnx | cut -d' ' -f1)"
if [ "$want" != "$have" ]; then
    echo "models/model.onnx: sha256 $have, в model.json $want (LFS не скачан?)" >&2
    exit 1
fi

echo "== сборка и запуск"
docker compose build
docker compose up -d --remove-orphans

echo "== health"
for _ in $(seq 60); do
    if curl -fsS http://127.0.0.1:8080/api/health 2>/dev/null | grep -q '"status":"ok"'; then
        curl -fsS http://127.0.0.1:8080/api/info; echo
        docker image prune -f >/dev/null
        exit 0
    fi
    sleep 2
done
echo "стек не ответил ok за 120 с" >&2
docker compose ps
docker compose logs --tail 30
exit 1
