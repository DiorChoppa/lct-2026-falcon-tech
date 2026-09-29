use std::sync::Arc;

use crate::application::use_cases::UseCases;

use super::config::ServiceConfig;

pub struct Application {
    pub config: ServiceConfig,
    pub use_cases: Arc<UseCases>,
}
