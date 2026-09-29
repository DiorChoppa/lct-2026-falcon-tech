use std::sync::Arc;

use axum::Router;

use crate::application::use_cases::UseCases;

use super::config::ServiceConfig;

pub struct Application {
    pub config: ServiceConfig,
    pub root_router: Router,
    pub use_cases: Arc<UseCases>,
}
