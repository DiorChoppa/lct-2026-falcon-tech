//! Контракт OpenAPI: спека генерируется из кода и коммитится в docs/openapi.json.
//! Тест падает, если файл отстал от кода: обновить — `just openapi`.

use api_gateway::openapi::ApiDoc;
use utoipa::OpenApi;

#[test]
fn spec_lists_implemented_routes() {
    let spec = ApiDoc::openapi();
    for path in ["/api/health", "/api/info", "/api/embed"] {
        assert!(spec.paths.paths.contains_key(path), "нет пути {path}");
    }
}

#[test]
fn committed_spec_matches_generated() {
    let generated = ApiDoc::openapi().to_pretty_json().unwrap();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/openapi.json");
    let committed = std::fs::read_to_string(path).expect("docs/openapi.json отсутствует");
    assert!(
        committed.trim_end() == generated.trim_end(),
        "docs/openapi.json отстал от кода, выполни `just openapi`"
    );
}
