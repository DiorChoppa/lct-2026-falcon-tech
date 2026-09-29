# Команды разработки. `just` без аргументов показывает список.

default:
    @just --list

# Собрать все Rust-крейты
build:
    cargo build --workspace

# Тесты Rust + Python
test: proto-py
    cargo test --workspace
    cd ml && uv run pytest -q
    cd tagger && uv run pytest -q

# Линтеры
lint:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cd ml && uv run ruff check .
    cd tagger && uv run ruff check .

# Сгенерировать Python-стабы из proto для сервиса tagger (клиент к gallery.SetTags)
proto-py:
    cd tagger && uv run python -m grpc_tools.protoc -I ../proto/reid/v1 --python_out=. --grpc_python_out=. \
        ../proto/reid/v1/inference.proto ../proto/reid/v1/tagger.proto ../proto/reid/v1/gallery.proto

# Поднять только БД для локальной разработки (gallery/search вне Docker)
db:
    docker compose up -d db

# Перегенерировать docs/openapi.json из кода api-gateway (тест openapi.rs сверяет)
openapi:
    cargo run -q -p api-gateway -- --print-openapi > docs/openapi.json

# Поднять всё решение
up:
    docker compose up --build

# Скачать текущие веса в models/ (ссылка в общем чате)
fetch-models:
    git lfs pull --include="models/model.onnx"

# ANN-демо: 10^6 случайных векторов в pgvector, HNSW, латентность и recall (bench/results/)
ann-bench n="1000000":
    uv run bench/ann_bench.py --n {{n}}

# Артефакты сдачи (три файла) Python-пайплайном жюри: ml/extractor.py + reid.submit
submit out="submission":
    cd ml && uv run python -m reid.submit --images ../dataset/images --query ../dataset/test_query.csv \
        --gallery ../dataset/test_gallery.csv --out ../{{out}}

# Замер скорости по протоколу жюри (07-organizers-qa.md §1); на GPU — где есть CUDA
perf:
    cd ml && uv run python -m reid.perf --images ../dataset/images --csv ../dataset/test_query.csv

# Python extract() рабочего дерева и нативный путь, по два прогона с побитным сравнением. Нужны
# docker + NVIDIA toolkit, собранный образ (just submit-image), models/model.onnx и dataset/.
# Скорость и детерминизм на GPU в образе жюри (docs/speed.md), итоги в каталоге out
perf-gpu image="reid-submit" out="perf-gpu" image_volume="":
    #!/usr/bin/env bash
    set -euo pipefail
    [ ! -e "{{out}}" ] || { echo "Output already exists: {{out}}" >&2; exit 1; }
    mkdir -p "{{out}}" && out=$(cd "{{out}}" && pwd)
    repo="$PWD"
    images=$(cd dataset/images && pwd -P)
    if command -v cygpath >/dev/null 2>&1; then
        repo=$(cygpath -m "$repo")
        out=$(cygpath -m "$out")
        images=$(cygpath -m "$images")
        export MSYS_NO_PATHCONV=1
    fi
    image_mount=(-v "$images:/repo/dataset/images:ro")
    if [ -n "{{image_volume}}" ]; then image_mount=(--mount "type=volume,source={{image_volume}},target=/repo/dataset/images,volume-subpath=images,readonly"); fi
    run() { docker run --rm --label "lct.audit=${LCT_AUDIT_RUN:-manual}" --gpus all --network none -e "CUDA_DISABLE_PTX_JIT=${CUDA_DISABLE_PTX_JIT:-1}" -v "$repo:/repo:ro" "${image_mount[@]}" -v "$out:/out" "$@"; }
    py() { run -w /opt/lct/python --entrypoint python "{{image}}" "$@"; }
    docker image inspect "{{image}}" > "$out/image.json"
    nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv > "$out/gpu.txt"
    (lscpu 2>/dev/null | grep -m1 'Model name' || true) >> "$out/gpu.txt"
    cat "$out/gpu.txt"
    echo "Python extract() baked into the image: 50 warmups, 300 batch-one runs"
    py -m reid.perf --images /repo/dataset/images --csv /repo/dataset/test_query.csv --json /out/python.json --embeddings /out/python-run1.npy
    py -m reid.perf --images /repo/dataset/images --csv /repo/dataset/test_query.csv --batches "" --json /out/python-repeat.json --embeddings /out/python-run2.npy
    echo "Native full extraction benchmark"
    awk -F, 'NR==1 {print "image_id,path,x,y,w,h"; next} {print $1",/repo/dataset/images/"$1".jpg,"$2","$3","$4","$5}' dataset/test_query.csv > "$out/native-manifest.csv"
    py -c 'import json; m = json.load(open("/opt/lct/model/model.json")); json.dump({"size": m["input_height"], "mean": m["mean"], "std": m["std"], "mode": "stretch", "crop_pct": 1.0}, open("/out/native-config.json", "w"))'
    run --entrypoint python "{{image}}" /opt/lct/launch-inference.py benchmark /out/native-config.json /opt/lct/model/model.onnx /out/native-manifest.csv /out/native.json --metadata-policy /opt/lct/model/policy.json --preprocess-workers 4 --reuse-identical-preprocessing true
    echo "Two complete offline submissions"
    for i in 1 2; do
        run "{{image}}" --images /repo/dataset/images --query /repo/dataset/test_query.csv --gallery /repo/dataset/test_gallery.csv --out /out/submit$i
    done
    py -c 'import json; r = json.load(open("/out/python.json")); print("Python:", r["latency_b1_ms"], "ms", r["best_fps"], "FPS")'
    py -c 'import json; r = json.load(open("/out/native.json")); assert r["status"] == "completed", r["status"]; print("Native:", round(r["latency_median_ms"], 2), "ms", round(r["best_fps"], 2), "FPS")'
    cmp "$out/python-run1.npy" "$out/python-run2.npy"
    for name in embeddings.npy submission.csv candidates.csv; do
        cmp "$out/submit1/$name" "$out/submit2/$name"
    done
    py /opt/lct/verify_submission.py --submission /out/submit1 --query /repo/dataset/test_query.csv --gallery /repo/dataset/test_gallery.csv --manifest /opt/lct/model/model.json --json /out/submission-check.json
    echo "PASS: repeated Python embeddings and all three native outputs are byte-identical"

# Offline jury image (Linux x86_64, CUDA GPU required).
submit-image:
    docker build --platform linux/amd64 -f deploy/submit.Dockerfile -t reid-submit .

# Собрать и запушить образы стека в registry (веса inference внутри): just push-images cr.yandex/<id>/reid
push-images registry tag="latest":
    docker compose build
    for s in inference gallery search tagger api web; do \
        img=$s; [ "$s" = api ] && img=api-gateway; \
        docker tag reid-$s {{registry}}/$img:{{tag}} && docker push {{registry}}/$img:{{tag}}; \
    done

# Паритет Rust-инференса с Python на одном кропе: cargo test -p inference -- --ignored
# (нужны models/model.onnx и ml/parity/). На всём тесте: те же три файла Rust-кодом
# сервиса → submission-rust/, затем сравнение с submission/ (топ-10, топ-1, отказы).
submit-rust out="submission-rust":
    cargo run --release --bin reid-cli -- submit --out {{out}}
    cd ml && uv run python -m reid.compare ../submission ../{{out}}

# Обучение модели заново — ml/TRAINING.md (CUDA-GPU с BF16, torch/timm в ml/.venv; пути абсолютные или от ml/)

# Проверка входов обучения без GPU: вывод FIT-v8, конфиги этапов, сэмплер 2+2
train-check:
    cd ml && uv run python -m train.fit_v8 && uv run pytest -q tests/test_train.py

# Этап 1: VeRi-Wild TRAIN (внешний датасет) из весов DINOv3 → runs/stage1_veriwild/backbone.safetensors
train-veriwild veriwild_root foundation run="runs/stage1_veriwild":
    cd ml && uv run python -m train.stage1_veriwild --config train/configs/stage1_veriwild.json \
        --veriwild-root {{veriwild_root}} --weights {{foundation}} --run-dir {{run}}

# Этап 2: FIT-v8 + сэмплер 2+2 из бэкбона этапа 1 → чекпоинт epoch0008-batch0000-step0002864.pt
train-fit backbone run="runs/stage2_fit_v8":
    cd ml && uv run python -m train.stage2_fit --config train/configs/stage2_fit_v8_2plus2.json \
        --weights {{backbone}} --run-dir {{run}}

# Экспорт чекпоинта этапа 2 в смешанный FP16/FP32 ONNX (CPU; конвертер onnxruntime 1.24.4)
train-export checkpoint out="runs/export":
    cd ml && uv run --with onnxruntime==1.24.4 python -m train.export_onnx --checkpoint {{checkpoint}} --output {{out}}
