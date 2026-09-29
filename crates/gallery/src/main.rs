use std::sync::Arc;

use tokio::sync::Notify;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stop_signal = Arc::new(Notify::new());
    gallery::run(stop_signal).await
}
