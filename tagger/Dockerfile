# Контекст сборки — корень репо (нужен proto/). См. docker-compose.yml.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY tagger/pyproject.toml ./
RUN uv sync --group dev --no-install-project
COPY proto /proto
RUN uv run python -m grpc_tools.protoc -I /proto/reid/v1 --python_out=. --grpc_python_out=. \
    /proto/reid/v1/inference.proto /proto/reid/v1/tagger.proto /proto/reid/v1/gallery.proto
COPY tagger/ .
ENV TAGGER_ADDR=0.0.0.0:50052
ENV GALLERY_ADDR=gallery:50054
# --no-sync: окружение собрано на этапе сборки, в рантайме интернета нет
CMD ["uv", "run", "--no-sync", "python", "server.py"]
