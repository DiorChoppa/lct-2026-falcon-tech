use crate::domain::ThresholdManifest;

#[test]
fn load_reads_threshold_and_alpha_from_model_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.json");
    std::fs::write(&path, r#"{"threshold": 0.62, "alpha": 0.8}"#).unwrap();

    let manifest = ThresholdManifest::load(&path).unwrap();

    assert_eq!(0.62, manifest.threshold);
    assert_eq!(0.8, manifest.alpha);
}

#[test]
fn load_defaults_alpha_to_one_when_absent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.json");
    std::fs::write(&path, r#"{"threshold": 0.5}"#).unwrap();

    let manifest = ThresholdManifest::load(&path).unwrap();

    assert_eq!(1.0, manifest.alpha);
}
