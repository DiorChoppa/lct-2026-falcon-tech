use std::path::Path;

use serde::Deserialize;

/// The subset of `models/model.json` search applies. `model_version` comes
/// from `inference.Info` instead — that's the loaded model's identity, this
/// file is where Sergey publishes the threshold/alpha decision for it (see
/// docs/02-plan.md "Разделение между ML и сервисом").
#[derive(Debug, Clone, Deserialize)]
pub struct ThresholdManifest {
    /// Candidates with `confidence` below this are rejected.
    pub threshold: f32,
    /// `confidence = alpha * score + (1 - alpha) * local_score`. `1.0`
    /// (default until validated on real data) ignores local_score entirely.
    #[serde(default = "default_alpha")]
    pub alpha: f32,
}

fn default_alpha() -> f32 {
    1.0
}

impl ThresholdManifest {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}
