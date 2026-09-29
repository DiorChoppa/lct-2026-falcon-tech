//! Спека OpenAPI. Каждый новый handler добавляется в `paths(...)`, иначе тест
//! crates/api-gateway/tests/openapi.rs его не увидит.

use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "ReID ТС без ГРЗ",
        description = "Сервис формирования цифрового признака транспортного средства: \
                       галерея, поиск по кропу, режим отказа, привязка ГРЗ, экспорт. \
                       Ошибки — всегда `{ error: { code, message } }`."
    ),
    paths(
        crate::health,
        crate::info,
        crate::embed::embed,
        crate::gallery::create,
        crate::gallery::import,
        crate::gallery::list,
        crate::gallery::get_one,
        crate::gallery::set_plate,
        crate::searches::search,
        crate::searches::compare,
        crate::searches::export_csv
    ),
    tags(
        (name = "service", description = "Состояние сервиса и модели, прямой эмбеддинг для замера"),
        (name = "gallery", description = "Галерея ранее виденных ТС"),
        (name = "search", description = "Поиск по кропу, режим отказа, история и экспорт")
    )
)]
pub struct ApiDoc;
