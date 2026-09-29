# Один Dockerfile на все Rust-бинарники: BIN=inference | gallery | search | api-gateway | reid-cli.
# Веса модели запекаются в образ inference (стадия models), поэтому образ
# работает без интернета и без volume — требование ТЗ.
#
# trixie, а не bookworm: inference статически линкует ONNX Runtime (`ort`,
# download-binaries) — пребилд собран под glibc/libstdc++ новее, чем в
# bookworm (glibc 2.36), и линковка падает на `__isoc23_strtol` и
# `_M_replace_cold`. В trixie (glibc 2.41, libstdc++ 14) собирается и
# запускается. Бинарники api-gateway/reid-cli это не задевает — совместимость
# glibc обратная.
FROM rust:1.92-trixie AS build
ARG BIN
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY proto ./proto
# Миграции gallery (crates/gallery/migrations) встраивает sqlx::migrate! на
# этапе компиляции — отдельная копия в образе не нужна.
# Общий кэш target/ и реестра между BIN: зависимости workspace компилируются
# один раз, а не в каждом из четырёх образов; sharing=locked выстраивает
# параллельные сборки compose в очередь (на ВМ — ещё и потолок по памяти).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,sharing=locked \
    cargo build --release --bin ${BIN} && cp target/release/${BIN} /bin-out

# Не у каждого BIN есть crates/<bin>/config (inference, gallery и search
# читают ./config/<run_mode> от CWD — см. infrastructure/setup/config.rs);
# api-gateway/reid-cli его не используют. Отдельная стадия вместо COPY
# напрямую в финальный образ: путь не всегда существует, а COPY с
# несуществующим source валит сборку.
FROM build AS config
ARG BIN
RUN mkdir -p /out/config && cp -r crates/${BIN}/config/. /out/config/ 2>/dev/null || true

# models/model.json читают inference, search и api-gateway (порог) и reid-cli;
# веса (~600 МБ) нужны только inference и reid-cli. bind-mount вместо COPY,
# чтобы веса не попадали в слои остальных образов. Без весов (нет файла или
# LFS-указатель) inference собирается и стартует, Embed падает до их появления.
# inference получает сервисный граф: тот же граф жюри + выход patches для сверки
# по деталям (deploy/service_model.py); reid-cli — граф жюри как есть (паритет).
FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a AS models
ARG BIN
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --quiet --no-compile onnx==1.23.0 numpy==2.5.3
RUN --mount=type=bind,source=models,target=/models \
    --mount=type=bind,source=deploy/service_model.py,target=/service_model.py \
    mkdir -p /out && cp /models/model.json /models/DINOV3_LICENSE.md /out/ && \
    if [ "${BIN}" = inference ] || [ "${BIN}" = reid-cli ]; then \
      python -c 'import hashlib,json,pathlib; p=pathlib.Path("/models"); m=json.loads((p/"model.json").read_text()); actual=hashlib.file_digest((p/m["file"]).open("rb"),"sha256").hexdigest(); assert actual == m["sha256"], "Missing, stale or LFS-pointer model weights"' && \
      if [ "${BIN}" = inference ]; then python /service_model.py /models/model.json /out; \
      else cp /models/model.onnx /out/; fi; \
    fi

FROM debian:trixie-slim
ARG BIN
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /bin-out /app/bin
COPY --from=models /out /app/models
COPY --from=config /out/config /app/config
ENTRYPOINT ["/app/bin"]
