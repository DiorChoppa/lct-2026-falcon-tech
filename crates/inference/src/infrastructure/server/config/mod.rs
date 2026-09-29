#[cfg(test)]
mod tests;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct HttpServerConfig {
    host: String,
    port: u16,
}

impl HttpServerConfig {
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}
