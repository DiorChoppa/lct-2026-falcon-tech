use std::sync::Arc;

use super::Application;

pub async fn do_nothing(_: Arc<Application>) -> anyhow::Result<()> {
    Ok(())
}
