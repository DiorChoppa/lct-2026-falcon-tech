use serde::{Deserialize, Serialize};

/// Ограничивающий прямоугольник в пикселях исходного кадра: левый верхний
/// угол, ширина, высота. Формат совпадает с CSV датасета.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BBox {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BBoxError {
    #[error("bbox has zero width or height")]
    Empty,
    #[error("bbox exceeds frame {frame_w}x{frame_h}")]
    OutOfFrame { frame_w: u32, frame_h: u32 },
}

impl BBox {
    /// Проверка, что bbox непустой и целиком лежит в кадре.
    pub fn validate(&self, frame_w: u32, frame_h: u32) -> Result<(), BBoxError> {
        if self.w == 0 || self.h == 0 {
            return Err(BBoxError::Empty);
        }
        if self.x + self.w > frame_w || self.y + self.h > frame_h {
            return Err(BBoxError::OutOfFrame { frame_w, frame_h });
        }
        Ok(())
    }

    /// Обрезает bbox по границам кадра (13% кропов в датасете касаются края,
    /// и в разметке встречаются выходы на 1–2 px за кадр).
    pub fn clamp(&self, frame_w: u32, frame_h: u32) -> BBox {
        let x = self.x.min(frame_w.saturating_sub(1));
        let y = self.y.min(frame_h.saturating_sub(1));
        BBox {
            x,
            y,
            w: self.w.min(frame_w - x),
            h: self.h.min(frame_h - y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_inside_frame() {
        assert_eq!(
            BBox {
                x: 0,
                y: 0,
                w: 10,
                h: 10
            }
            .validate(10, 10),
            Ok(())
        );
    }

    #[test]
    fn validate_rejects_out_of_frame() {
        let err = BBox {
            x: 5,
            y: 0,
            w: 10,
            h: 10,
        }
        .validate(10, 10)
        .unwrap_err();
        assert!(matches!(err, BBoxError::OutOfFrame { .. }));
    }

    #[test]
    fn clamp_trims_to_frame() {
        let b = BBox {
            x: 1915,
            y: 1075,
            w: 20,
            h: 20,
        }
        .clamp(1920, 1080);
        assert_eq!(
            b,
            BBox {
                x: 1915,
                y: 1075,
                w: 5,
                h: 5
            }
        );
    }
}
