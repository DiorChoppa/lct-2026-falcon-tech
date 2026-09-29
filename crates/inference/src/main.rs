use std::sync::Arc;

use inference::infrastructure::setup::configurators::do_nothing;
use inference::run;
use tokio::sync::Notify;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stop_signal = Arc::new(Notify::new());
    run(stop_signal, do_nothing).await
}
