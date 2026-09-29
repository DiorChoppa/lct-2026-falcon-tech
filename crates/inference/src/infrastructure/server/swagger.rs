use utoipa::OpenApi;
use utoipa_rapidoc::RapiDoc;

use crate::infrastructure::server::routes::*;

const OPENAPI_JSON_URL: &str = "/api-docs/openapi.json";
const OPENAPI_DOCS_URL: &str = "/rapidoc";

#[derive(OpenApi)]
#[openapi(
    info(
        description = "inference: auxiliary HTTP (health-check). gRPC — proto/reid/v1/inference.proto."
    ),
    paths(health)
)]
struct ApiDoc;

pub trait SwaggerExamples {
    type Example: serde::Serialize;

    fn example(value: Option<&str>) -> Self::Example;
}

pub fn init_api_doc() -> RapiDoc {
    let api_doc = ApiDoc::openapi();
    RapiDoc::with_openapi(OPENAPI_JSON_URL, api_doc).path(OPENAPI_DOCS_URL)
}
