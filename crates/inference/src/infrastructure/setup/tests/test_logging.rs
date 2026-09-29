use crate::infrastructure::setup::init_logger;
use crate::infrastructure::setup::logging::LoggerConfig;

#[test]
fn test_init_logger_json_format() {
    let json = r#"{ "format": "json" }"#;
    let config: LoggerConfig = serde_json::from_str(json).unwrap();

    let result = init_logger(&config);

    assert!(result.is_ok());
}

#[test]
fn test_init_logger_pretty_format() {
    let json = r#"{ "format": "pretty" }"#;
    let config: LoggerConfig = serde_json::from_str(json).unwrap();

    let result = init_logger(&config);

    assert!(result.is_ok());
}
