use super::GrpcServerConfig;

#[test]
fn test_address_grpc() {
    let config = GrpcServerConfig {
        host: "0.0.0.0".to_string(),
        port: 50051,
    };

    assert_eq!("0.0.0.0:50051", config.address());
}
