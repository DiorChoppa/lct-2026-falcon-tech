/// Косинусное сходство. Векторы считаются L2-нормированными (это контракт
/// модели, см. models/model.json), поэтому достаточно скалярного произведения.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Кандидат после ранжирования: индекс в галерее и сходство с запросом.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ranked {
    pub index: usize,
    pub score: f32,
}

/// Ранжирует галерею по убыванию сходства с запросом и возвращает топ-`k`.
/// Используется CLI для submission.csv; api делает то же через pgvector.
pub fn rank(query: &[f32], gallery: &[Vec<f32>], k: usize) -> Vec<Ranked> {
    let mut all: Vec<Ranked> = gallery
        .iter()
        .enumerate()
        .map(|(index, g)| Ranked {
            index,
            score: cosine(query, g),
        })
        .collect();
    all.sort_by(|a, b| b.score.total_cmp(&a.score));
    all.truncate(k);
    all
}

/// Режим предложения кандидатов: оставляет только тех, чей score не ниже
/// порога. Пустой результат — это отказ.
pub fn accept(ranked: &[Ranked], threshold: f32) -> Vec<Ranked> {
    ranked
        .iter()
        .copied()
        .filter(|r| r.score >= threshold)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_orders_by_similarity() {
        let q = vec![1.0, 0.0];
        let g = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.7, 0.7]];
        let r = rank(&q, &g, 2);
        assert_eq!(r[0].index, 1);
        assert_eq!(r[1].index, 2);
    }

    #[test]
    fn accept_returns_empty_below_threshold() {
        let r = vec![Ranked {
            index: 0,
            score: 0.3,
        }];
        assert!(accept(&r, 0.5).is_empty());
    }
}
