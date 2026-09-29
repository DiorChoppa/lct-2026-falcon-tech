# Хостинг прототипа на ВМ

Ручная/полуручная настройка (02-plan.md этап 4). Демо-стенд на CPU-ВМ в
Yandex Compute: скорость жюри меряет на своём GPU через образ `reid-submit`,
наш инстанс — только демо сервиса (тай-брейкер).

## ВМ

| Параметр | Значение | Почему |
|---|---|---|
| Платформа | Intel Ice Lake (`standard-v3`), доля vCPU 100% | AVX-512 для ONNX Runtime; прерываемую не брать — инстанс живёт до защиты |
| vCPU / RAM | **8 vCPU / 16 ГБ** | forward ViT-L/16 на CPU почти целиком и масштабируется по ядрам (на M4 Pro 4 → 8 → 14 ядер: 750 → 483 → 380 мс); inference держит ~3 ГБ, сборка Rust — ещё несколько ГБ |
| Диск | 50 ГБ network-ssd | образы ~3 ГБ + кэш сборки Rust ~10 ГБ + кадры демо 1,2 ГБ + БД |
| Образ | Ubuntu 24.04 LTS, x86_64 | — |
| Сеть | **статический** публичный IPv4 | адрес идёт на слайды; динамический меняется при остановке ВМ |

Замер на стенде 23.09 (8 vCPU Ice Lake): поиск `/api/search` p50 0,68 с /
p95 0,70 с на ВМ, 1,5 с снаружи через HTTPS с загрузкой кадра; импорт
галереи 750 кропов — 7 мин 50 с. Если для демо нужно быстрее: остановить
ВМ → 16 vCPU / 32 ГБ → запустить (пара минут, статический IP сохраняется); после защиты — удалить.
Порядок цены по тарифам Ice Lake на 03.2026 (1,15 ₽ за vCPU·ч, 0,31 ₽ за
ГБ·ч): 8/16 ≈ 14 ₽/ч ≈ 340 ₽/сутки, 16/32 ≈ 28 ₽/ч, плюс диск и IP —
сверить в калькуляторе консоли.

Группа безопасности: входящие 22 (свой IP), 80 и 443 (все; 80 нужен
Let's Encrypt и редиректу на https). Остальные порты стек публикует только на
127.0.0.1 (`docker-compose.vm.yml`), наружу торчит один Caddy.

Текущий стенд: `lct.znatalk.ai` → 158.160.224.157, пользователь `lct-admin`.

## Как устроено

| Что | Где |
|---|---|
| Стек | `/srv/lct` (владелец `gitlab-runner`): рабочая копия репозитория, `.env`, данные `data/{pg,crops,caddy}`, кадры `dataset/images/` |
| Compose | `.env` задаёт `COMPOSE_FILE=docker-compose.yml:deploy/vm/docker-compose.vm.yml` и `COMPOSE_PROJECT_NAME=reid` — любой `docker compose` в `/srv/lct` идёт с overlay |
| TLS | Caddy (`deploy/vm/Caddyfile`): сертификат Let's Encrypt для `DOMAIN`, http → https; basic auth проверяет nginx в `web`, `/api/health` открыт |
| Автозапуск | `restart: unless-stopped` у всех сервисов |
| Выкатка | job `deploy` в `.gitlab-ci.yml` на push в `master`: раннер `lct-vm` (shell executor на этой ВМ) запускает `deploy/vm/deploy.sh` — код и веса из LFS по `CI_JOB_TOKEN`, проверка SHA весов, `docker compose build && up -d`, ожидание `/api/health` |

## Первый запуск

1. В GitLab: Settings → CI/CD → Runners → New project runner, тег `lct-vm`,
   без «Run untagged jobs»; токен `glrt-…` положить на ВМ в `~/runner-token`.
2. Настройка ВМ (Docker, Git LFS, gitlab-runner, `/srv/lct/.env` с паролем
   `jury`, печатается один раз):

   ```sh
   scp deploy/vm/setup.sh lct-admin@<ip>:
   ssh lct-admin@<ip> 'DOMAIN=lct.znatalk.ai RUNNER_TOKEN_FILE=~/runner-token bash setup.sh'
   ```

3. Выкатка: push в `master` (или Run pipeline) — job `deploy` поднимет стек.
   Руками, из того, что уже лежит в `/srv/lct`:
   `ssh lct-admin@<ip> 'cd /srv/lct && sudo -u gitlab-runner bash deploy/vm/deploy.sh'`.

Сборка Rust-образов идёт с общим кэшем cargo (`deploy/rust.Dockerfile`):
зависимости компилируются один раз, четыре бинарника — по очереди.

## Данные для демо

CSV лежат в git; кадры (`dataset/images/`) — нет. Галерея демо — две части
(`docs/demo.md`): галерея теста и галерея development-эпизода SEARCH (с
`vehicle_id`, по ней видно верность совпадений). Нужны их кадры и кадры
запросов (~1,8 ГБ из 7):

```sh
# на своей машине, из корня репозитория
E=ml/validation/search
tail -n +2 -q dataset/test_gallery.csv dataset/test_query.csv $E/gallery.csv $E/query.csv \
    | cut -d, -f1 | sed 's/$/.jpg/' | sort -u > /tmp/demo-images.txt
rsync -a --rsync-path="sudo -u gitlab-runner rsync" --files-from=/tmp/demo-images.txt \
    dataset/images/ lct-admin@<ip>:/srv/lct/dataset/images/
# импорт — на самой ВМ, мимо прокси (на CPU он дольше proxy_read_timeout 600s в
# web/nginx.conf): import-gallery.sh режет CSV на части и продолжает после обрыва
for csv in dataset/test_gallery.csv ml/validation/search/gallery.csv; do
    ssh lct-admin@<ip> "cd /srv/lct && sudo -u gitlab-runner deploy/vm/import-gallery.sh $csv"
done
```

Импорт не идемпотентен (повтор добавит дубли), поэтому только через
`import-gallery.sh`: готовые части помечаются в `~gitlab-runner/.cache/lct-import/`.

Остальной train для «большой» базы — без кадров машин эпизода: иначе запросы
из `docs/demo.md` найдут сами себя, а «честный отказ» перестанет быть отказом.
Это 8 221 кадр из 9 556 (минус 1 335 кадров 215 машин эпизода), ~5 ГБ кадров,
импорт ~80 мин на 8 vCPU:

```sh
cd ml && uv run python -c "
import pandas as pd; E='artifacts/episodes/v1/search'
tr = pd.read_csv('../dataset/train.csv', dtype={'image_id': str})
ep = set(pd.read_csv(f'{E}/gallery.csv').vehicle_id) | set(pd.read_csv(f'{E}/query.csv').vehicle_id)
tr[~tr.vehicle_id.isin(ep)].to_csv('/tmp/train-demo.csv', index=False)"
# кадры: cut -d, -f1 /tmp/train-demo.csv → rsync, как выше; CSV — на ВМ, затем
# sudo -u gitlab-runner deploy/vm/import-gallery.sh <путь к train-demo.csv>
```

На стенде 23.09: галерея теста 750 + галерея эпизода 750 + train без машин
эпизода 8 221 = 9 721 запись.

## Обслуживание

```sh
ssh lct-admin@<ip>
cd /srv/lct
sudo -u gitlab-runner docker compose ps          # или logs -f inference / caddy
sudo grep BASIC_AUTH_PASSWORD .env                # пароль jury
sudo journalctl -u gitlab-runner -f              # что делает раннер
```

- `YMAPS_KEY` — ключ Яндекс Карт (JavaScript API) для экрана «Маршрут ТС»; ограничить
  доменом `lct.znatalk.ai` в кабинете разработчика. Строка `YMAPS_KEY=<ключ>` — в
  `/srv/lct/.env` (`sudo -u gitlab-runner nano /srv/lct/.env`); применяется
  перезапуском web (`sudo -u gitlab-runner docker compose up -d web`), пересборка не нужна.

Образы из registry вместо сборки на ВМ — `docker-compose.registry.yml`
(добавить в `COMPOSE_FILE`, веса уже в образе inference).

## Проверка

- `https://<домен>/api/health` — без пароля, `{"status":"ok",…}`.
- `https://<домен>/` — запрос пароля, экран поиска.
- `https://<домен>/api/docs/` — Swagger.
