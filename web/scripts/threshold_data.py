"""Данные для графиков порога на странице «Решение» → src/data/threshold.json.

Источник — эпизод калибровки (ml/validation/holdout.json, косинус без DBA — как в живом
сервисе; 200 запросов, 40 без пары) и отдельно аудит для проверки выбранного порога.
Правила подсчёта — как у evaluate.py организаторов (docs/07-organizers-qa.md §3):
решение по верхнему кандидату, TP — пара есть и топ-1 верен, FP — ответ дан,
но топ-1 неверен или пары нет, FN — пара есть, ответа нет, TN — пары нет, отказ.
Скрипт сверяет рабочие точки с holdout.json (порог — верхний threshold models/model.json).

Запуск из корня репозитория: python3 web/scripts/threshold_data.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ml/validation/holdout.json"
OUT = ROOT / "web/src/data/threshold.json"

holdout = json.loads(SRC.read_text())["methods"]["cosine"]
threshold = json.loads((ROOT / "models/model.json").read_text())["threshold"]
queries = holdout["calibration"]["queries"]


def counts(t: float, qs: list | None = None) -> dict:
    """g: correct — пара есть и топ-1 той же машины; wrong — топ-1 чужой; absent — пары нет."""
    tp = fp = fn = tn = 0
    for q in queries if qs is None else qs:
        accepted = q["c"] >= t
        g = q["g"]
        if g == "absent":
            fp, tn = (fp + 1, tn) if accepted else (fp, tn + 1)
        elif not accepted:
            fn += 1
        elif g == "correct":
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_unknown = sum(q["g"] == "absent" for q in (queries if qs is None else qs))
    tnr = tn / n_unknown if n_unknown else 0.0
    return {
        "t": round(t, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "f1": round(f1, 4), "tnr": round(tnr, 4),
        "score": round(0.7 * f1 + 0.3 * tnr, 4),
    }


op = counts(threshold)
audit = counts(threshold, holdout["audit"]["queries"])
for got, ref in ((op, holdout["calibration"]["manifest"]), (audit, holdout["audit"]["manifest"])):
    assert (got["tp"], got["fp"], got["fn"], got["tn"]) == (ref["TP"], ref["FP"], ref["FN"], ref["TN"]), got

    assert abs(got["f1"] - ref["F1"]) <= 0.00005
    assert abs(got["tnr"] - ref["TNR"]) <= 0.00005

steps = [round(0.30 + i * 0.005, 3) for i in range(int((0.95 - 0.30) / 0.005) + 1)]
OUT.write_text(json.dumps({
    "source": str(SRC.relative_to(ROOT)),
    "threshold": threshold,
    "operatingPoint": op,
    "audit": audit,
    "curve": [counts(t) for t in steps],
    "queries": queries,
}, ensure_ascii=False, separators=(",", ":")) + "\n")
print(f"{OUT.relative_to(ROOT)}: {len(steps)} точек кривой, {len(queries)} запросов, рабочая точка {op}, аудит {audit}")
