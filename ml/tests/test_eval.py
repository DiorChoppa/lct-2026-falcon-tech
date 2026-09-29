"""Метрики ReID на игрушечном примере с известным ответом.

Галерея: g0 (id1, cam2), g1 (id2, cam1), g2 (id1, cam1), g3 (id3, cam3).
Запросы:  q0 (id1, cam1) — g2 та же машина с той же камеры (junk), по ТЗ §9 исключается;
          q1 (id2, cam2) — верный g1 на втором месте;
          q2 (id9, cam3) — пары в галерее нет (для TNR).
"""

import numpy as np
import pytest

from reid.eval import evaluate

G_IDS = np.array([1, 2, 1, 3])
G_CAMS = np.array([2, 1, 1, 3])
Q_IDS = np.array([1, 2, 9])
Q_CAMS = np.array([1, 2, 3])
SIM = np.array(
    [
        [0.9, 0.8, 0.95, 0.1],
        [0.7, 0.6, 0.2, 0.1],
        [0.3, 0.2, 0.1, 0.05],
    ]
)


def test_ranking_metrics_exclude_same_camera_matches():
    r = evaluate(SIM, Q_IDS, G_IDS, Q_CAMS, G_CAMS)
    assert r["mAP10"] == pytest.approx(0.75)  # q0: junk g2 выброшен, g0 первый; q1: AP=1/2
    assert r["mAP"] == pytest.approx(0.75)
    assert r["rank1"] == pytest.approx(0.5)
    assert r["rank5"] == pytest.approx(1.0)
    assert r["mINP"] == pytest.approx(0.75)  # q0: 1/1, q1: 1/2
    assert r["n_with_match"] == 2
    assert r["n_without_match"] == 1


def test_junk_only_positive_makes_query_open_set():
    # q2 → id3 cam3: единственная пара g3 снята той же камерой — без камер это пара, с камерами нет
    q_ids, q_cams = np.array([1, 2, 3]), np.array([1, 2, 3])
    assert evaluate(SIM, q_ids, G_IDS, q_cams, G_CAMS)["n_with_match"] == 2
    assert evaluate(SIM, q_ids, G_IDS)["n_with_match"] == 3


def test_map10_counts_only_submitted_top_k():
    # 12 галерейных, верный на 11-м месте: в полный mAP входит, в топ-10 не попадает
    sim = np.array([np.linspace(1.0, 0.0, 12)])
    g_ids = np.array([0] * 10 + [7, 0])
    r = evaluate(sim, np.array([7]), g_ids)
    assert r["mAP"] == pytest.approx(1 / 11)
    assert r["mAP10"] == 0.0


def test_refusal_metrics_at_fixed_threshold():
    # q0 принят и топ-1 та же машина (TP), q1 принят, но топ-1 чужой (FP), q2 отклонён без пары (TN)
    r = evaluate(SIM, Q_IDS, G_IDS, Q_CAMS, G_CAMS, threshold=0.65)
    assert r["threshold"] == 0.65
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(1.0)  # FN нет: q1 не отказан, а принят с ошибкой
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["tnr"] == pytest.approx(1.0)


def test_low_threshold_accepts_unmatched_query_and_zeroes_tnr():
    r = evaluate(SIM, Q_IDS, G_IDS, Q_CAMS, G_CAMS, threshold=0.25)
    assert r["tnr"] == 0.0
    assert r["precision"] == pytest.approx(1 / 3)


SIM_OPEN_SET_CONFIDENT = np.array(
    [
        [0.9, 0.8, 0.95, 0.1],
        [0.2, 0.7, 0.1, 0.1],
        [0.8, 0.1, 0.1, 0.1],
    ]
)


def test_pr_auc_ranks_has_match_label_by_top1_score():
    # По убыванию top-1: q0 0.95 (пара есть), q2 0.8 (пары нет), q1 0.7 (пара есть)
    r = evaluate(SIM_OPEN_SET_CONFIDENT, Q_IDS, G_IDS, Q_CAMS, G_CAMS, threshold=0.7)
    assert r["pr_auc"] == pytest.approx((1 + 2 / 3) / 2)


def test_default_threshold_maximises_f1():
    # t=0.95 принимает только q0: F1=2/3; t=0.7 даёт тот же F1 — берём порог выше
    r = evaluate(SIM, Q_IDS, G_IDS, Q_CAMS, G_CAMS)
    assert r["threshold"] == pytest.approx(0.95)
    assert r["f1"] == pytest.approx(2 / 3)
    thresholds = [p["threshold"] for p in r["curve"]]
    assert thresholds == sorted(thresholds, reverse=True)
    assert len(thresholds) == 3


def test_min_tnr_constraint_changes_chosen_threshold():
    # По F1 лучший порог t=0.7 (F1=0.8), но там q2 принят и TNR=0;
    # при TNR ≥ 0.5 остаётся t=0.95 с F1=2/3.
    free = evaluate(SIM_OPEN_SET_CONFIDENT, Q_IDS, G_IDS, Q_CAMS, G_CAMS, objective="f1")
    assert free["threshold"] == pytest.approx(0.7)
    assert free["f1"] == pytest.approx(0.8)
    assert free["tnr"] == 0.0
    strict = evaluate(
        SIM_OPEN_SET_CONFIDENT, Q_IDS, G_IDS, Q_CAMS, G_CAMS, min_tnr=0.5, objective="f1"
    )
    assert strict["threshold"] == pytest.approx(0.95)
    assert strict["f1"] == pytest.approx(2 / 3)


def test_jury_objective_trades_f1_for_tnr():
    # Балл жюри 0.7·F1 + 0.3·TNR: t=0.7 даёт 0.56 (TNR=0), t=0.95 — 0.7·2/3 + 0.3 ≈ 0.767
    r = evaluate(SIM_OPEN_SET_CONFIDENT, Q_IDS, G_IDS, Q_CAMS, G_CAMS)
    assert r["threshold"] == pytest.approx(0.95)
    assert r["jury"] == pytest.approx(0.7 * 2 / 3 + 0.3)


def test_unattainable_min_tnr_raises():
    # Запрос без пары уверенней всех: любой порог его принимает, TNR=0 везде
    sim = np.array(
        [
            [0.9, 0.8, 0.95, 0.1],
            [0.2, 0.7, 0.1, 0.1],
            [0.99, 0.1, 0.1, 0.1],
        ]
    )
    with pytest.raises(ValueError, match="TNR"):
        evaluate(sim, Q_IDS, G_IDS, Q_CAMS, G_CAMS, min_tnr=0.5)
