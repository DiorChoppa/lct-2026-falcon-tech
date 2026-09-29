use crate::infrastructure::setup::logging::LogFormat;
use crate::infrastructure::setup::ServiceConfig;

#[test]
fn test_service_config_new_with_development_mode() {
    let config = ServiceConfig::new().expect("Failed to create service config");

    assert_eq!(&LogFormat::Pretty, config.logger().format());
    assert_eq!("inference", config.system().application_name());
    assert_eq!("0.0.0.0:8081", config.http_server().address());
    assert_eq!("0.0.0.0:50051", config.grpc_server().address());
}
