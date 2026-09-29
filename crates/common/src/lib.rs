//! Общие типы и чистая логика, которую делят api, inference и cli:
//! bbox, эмбеддинг, косинусное сходство, ранжирование, порог отказа.
//! Здесь нет ни сети, ни БД, ни ONNX — только то, что легко покрыть тестами.

pub mod bbox;
pub mod crop;
pub mod ranking;

pub use bbox::BBox;
pub use crop::crop_jpeg;
pub use ranking::{cosine, rank};
