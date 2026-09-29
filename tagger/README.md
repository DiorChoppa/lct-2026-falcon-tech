# tagger

Асинхронные теги деталей по кропу (см. `docs/01-concept.md` §5.1). Владелец: Иван.

- `server.py` — gRPC-сервер по `proto/reid/v1/tagger.proto`: `Enqueue` кладёт
  задачу в очередь и сразу отвечает.
- `queue_worker.py` — фоновый поток: тегирует, вызывает `gallery.SetTags`.
  Без брокера — при падении теряются только незавершённые задачи;
  `gallery reindex --tags` доставляет пропущенные.
- `backend.py` — интерфейс `TaggerBackend`. Сейчас единственная реализация —
  `NullBackend`, всегда возвращающая пустой список тегов: настоящий
  zero-shot детектор (CLIP/OWLv2 по промптам из `vocab.yaml`) требует весов
  модели, которых нет в этом окружении. Проводка (`Enqueue` → очередь →
  воркер → `gallery.SetTags`) полностью реализована и покрыта тестами;
  подключение реальной модели — это один класс, реализующий `TaggerBackend`,
  без изменений в остальном коде.
- `gallery_client.py` — gRPC-клиент к `gallery.SetTags`. Кроме этого tagger
  никуда не пишет: ни в БД, ни в S3.
- `vocab.yaml` — словарь тегов для будущего zero-shot детектора.
- `*_pb2*.py` — сгенерированные стабы (`just proto-py`), в git не хранятся.

Запуск локально: `uv sync && just proto-py && uv run python server.py`.
Тесты: `uv sync --group dev && just proto-py && uv run pytest -q`.
