use serde::Deserialize;

#[derive(Deserialize)]
pub struct GrpcServerConfig {
    host: String,
    port: u16,
}

impl GrpcServerConfig {
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}
