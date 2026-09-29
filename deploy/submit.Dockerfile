# syntax=docker/dockerfile:1
# Jury artifact: images/ + CSVs -> submission.csv, embeddings.npy, candidates.csv in one
# command, offline at run time. Native Rust extractor (ml/inference) + ONNX Runtime CUDA,
# same sources and wheel set as the qualified development leader
# (ml/MODEL_GUIDE.md, ml/validation/metrics.json).
#
# Build (network allowed; models/model.onnx must be present, see models/README.md):
#   docker build -f deploy/submit.Dockerfile -t reid-submit .          (just submit-image)
# Run on the jury host (RTX A5000, CUDA driver 12.2):
#   docker run --rm --gpus all -v /path/to/data:/data -v $PWD/submission:/out reid-submit \
#       --images /data/images --query /data/test_query.csv --gallery /data/test_gallery.csv --out /out
ARG RUST_IMAGE=rust:1.94.1-bookworm@sha256:6ae102bdbf528294bc79ad6e1fae682f6f7c2a6e6621506ba959f9685b308a55
# Base with the pinned CUDA wheels. Default builds from python:3.12.12-slim; to reuse a local image
# that already has the same wheel set: --build-arg PYTHON_IMAGE=<image> --build-arg SKIP_WHEELS=1
ARG PYTHON_IMAGE=python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
FROM ${RUST_IMAGE} AS rust-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake=3.25.1-1 nasm=2.16.01-1 \
 && rm -rf /var/lib/apt/lists/*
ENV CMAKE_BUILD_PARALLEL_LEVEL=2 TURBOJPEG_STATIC=1 CARGO_TARGET_DIR=/build/target
WORKDIR /build
COPY ml/inference/rust/Cargo.toml ml/inference/rust/Cargo.lock ./inference/rust/
COPY ml/inference/rust/src ./inference/rust/src
COPY ml/inference/offline/Cargo.toml ml/inference/offline/Cargo.lock ./inference/offline/
COPY ml/inference/offline/src ./inference/offline/src
COPY ml/inference/offline/tests ./inference/offline/tests
COPY ml/inference/retrieval.rs ./inference/retrieval.rs
RUN cargo build --manifest-path inference/offline/Cargo.toml --release --locked -j 2 \
 && cargo build --manifest-path inference/rust/Cargo.toml --release --locked -j 2 \
 && cargo test --manifest-path inference/offline/Cargo.toml --release --locked -j 2 -- --test-threads=2

FROM ${PYTHON_IMAGE}
ARG SKIP_WHEELS=0
COPY ml/deployment/python-freeze-linux.txt /tmp/requirements.txt
# ~3.5 GB of CUDA wheels: cache mount keeps a retried build from re-downloading.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$SKIP_WHEELS" = 1 ]; then pip freeze | diff - /tmp/requirements.txt; else \
    apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* && \
    pip install --retries 10 --timeout 120 --no-deps -r /tmp/requirements.txt; fi
COPY --from=rust-build /build/target/release/lct-offline /build/target/release/lct-inference /usr/local/bin/
COPY ml/inference/rust/THIRD_PARTY_NOTICES.md ml/inference/rust/PILLOW-LICENSE.txt /opt/lct/licenses/
COPY ml/inference/rust/licenses /opt/lct/licenses
COPY ml/deployment/launch-offline.py ml/deployment/launch-inference.py ml/deployment/submit.py /opt/lct/
COPY ml/scripts/verify_submission.py /opt/lct/verify_submission.py
COPY models/model.json models/model.onnx /opt/lct/model/
COPY models/policy.json /opt/lct/model/policy.json
COPY models/DINOV3_LICENSE.md /opt/lct/licenses/DINOV3_LICENSE.md
# Второй вход — Python-контракт extractor.py (ответы 31, 43: замер «внутри extract()»,
# текст контракта не опубликован): ml/extractor.py на том же графе и ORT CUDA,
# препроцессинг Pillow. Нативный путь выше не меняется. Проверка:
#   docker run --rm --gpus all -v /data:/data --entrypoint python reid-submit \
#       -c "from extractor import extract; print(extract('/data/images/<id>.jpg', (x, y, w, h)).shape)"
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps pillow==12.3.0
COPY ml/extractor.py /opt/lct/python/
COPY ml/reid/__init__.py ml/reid/preprocess.py /opt/lct/python/reid/
COPY ml/reid/perf.py /opt/lct/python/reid/
# extractor.py ищет манифест в ../models/model.json от своего каталога
RUN ln -s model /opt/lct/models
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/lct/python
ENV LCT_REQUIRE_CUDA=1
ENV CUDA_DISABLE_PTX_JIT=1
WORKDIR /work
ENTRYPOINT ["python", "/opt/lct/submit.py"]
