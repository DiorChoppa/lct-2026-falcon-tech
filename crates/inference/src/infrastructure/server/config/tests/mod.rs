use super::HttpServerConfig;

#[test]
fn test_address_http() {
    let config = HttpServerConfig {
        host: "localhost".to_string(),
        port: 8081,
    };

    assert_eq!("localhost:8081", config.address());
}
