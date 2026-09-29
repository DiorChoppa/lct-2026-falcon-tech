//! Галерея через шлюз: валидация кадра и bbox, DTO из ответа gallery,
//! импорт CSV из каталога датасета, список, ГРЗ, отдача кропов с тома.

mod common;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use common::{
    crop_file, ctx, ctx_with, get, jpeg, multipart, oversized_png, png, post_gallery, put_json,
    send, send_raw, FakeSearch,
};

#[tokio::test]
async fn post_gallery_creates_item_via_gallery_add() {
    let c = ctx();
    let (status, item) = post_gallery(&c.state, r#"{"x":10,"y":10,"w":50,"h":40}"#).await;
    assert_eq!(status, StatusCode::CREATED, "{item}");
    assert_eq!(item["id"], 1);
    assert_eq!(item["vehicleId"], "403");
    assert_eq!(item["plate"], "A123BC77");
    assert_eq!(item["bbox"]["w"], 50);
    assert_eq!(item["tags"], serde_json::json!([]));
    assert!(item["createdAt"]
        .as_str()
        .unwrap()
        .starts_with("2023-11-14T"));
    // file://<crops_dir>/gallery-crops/<image_id>.jpg → /files/crops/gallery-crops/<image_id>.jpg
    let image_id = item["imageId"].as_str().unwrap();
    assert_eq!(
        item["cropUrl"],
        format!("/files/crops/gallery-crops/{image_id}.jpg")
    );
}

#[tokio::test]
async fn post_gallery_accepts_png_and_empty_optional_fields() {
    let c = ctx();
    let req = multipart(
        "/api/gallery",
        &[
            ("bbox", r#"{"x":0,"y":0,"w":30,"h":30}"#),
            ("vehicle_id", ""),
        ],
        ("image", "frame.png", &png(60, 40)),
    );
    let (status, item, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::CREATED, "{item}");
    assert!(item["vehicleId"].is_null());
    assert!(item["plate"].is_null());
}

#[tokio::test]
async fn post_gallery_rejects_bbox_outside_frame() {
    let c = ctx();
    let (status, body) = post_gallery(&c.state, r#"{"x":180,"y":10,"w":50,"h":40}"#).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "VALIDATION_ERROR");
    assert!(c.state.gallery.list(1, 10).await.unwrap().items.is_empty());
}

#[tokio::test]
async fn post_gallery_rejects_non_image() {
    let c = ctx();
    let req = multipart(
        "/api/gallery",
        &[("bbox", r#"{"x":0,"y":0,"w":1,"h":1}"#)],
        ("image", "frame.jpg", b"not an image at all"),
    );
    let (status, body, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"]["code"], "VALIDATION_ERROR");
}

#[tokio::test]
async fn post_gallery_rejects_frame_over_limit() {
    let c = ctx();
    let req = multipart(
        "/api/gallery",
        &[("bbox", r#"{"x":0,"y":0,"w":1,"h":1}"#)],
        ("image", "frame.png", &oversized_png((3 << 20) + 1)),
    );
    let (status, body, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "VALIDATION_ERROR");
    assert!(body["error"]["message"].as_str().unwrap().contains("3 МиБ"));
    assert!(c.state.gallery.list(1, 10).await.unwrap().items.is_empty());
}

/// Ошибки экстракторов axum — в том же JSON, что и остальные.
#[tokio::test]
async fn extractor_rejections_are_json_422() {
    let c = ctx();
    let (status, body, _) = send(&c.state, get("/api/gallery/abc")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "VALIDATION_ERROR");

    let (status, body, _) = send(&c.state, put_json("/api/gallery/1/plate", "{not json")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "VALIDATION_ERROR");

    let req = Request::post("/api/gallery")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("{}"))
        .unwrap();
    let (status, body, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["error"]["code"], "VALIDATION_ERROR");
}

#[tokio::test]
async fn post_gallery_reports_gallery_outage_as_502() {
    let c = ctx_with(true, FakeSearch::default(), false);
    let (status, body) = post_gallery(&c.state, r#"{"x":10,"y":10,"w":50,"h":40}"#).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"]["code"], "GALLERY_UNAVAILABLE");
}

#[tokio::test]
async fn import_reads_dataset_csv_and_lists_items() {
    let c = ctx();
    let images = c.state.dataset_dir.join("images");
    std::fs::create_dir_all(&images).unwrap();
    std::fs::write(images.join("aaa.jpg"), png(200, 100)).unwrap();
    std::fs::write(images.join("bbb.jpg"), jpeg(300, 100)).unwrap();
    std::fs::write(images.join("broken.jpg"), b"garbage").unwrap();
    std::fs::write(images.join("huge.jpg"), oversized_png((3 << 20) + 1)).unwrap();
    // Кадр вне images_dir: строка с image_id «../escape» не должна его прочитать.
    std::fs::write(c.state.dataset_dir.join("escape.jpg"), png(10, 10)).unwrap();
    // Формат train.csv; bbb выходит за кадр на 10 px (clamp), zzz — нет файла
    let csv = "image_id,x,y,w,h,vehicle_id,camera_id\n\
               aaa,0,0,100,100,403,90\n\
               bbb,10,10,300,50,404,89\n\
               broken,0,0,10,10,405,1\n\
               zzz,0,0,10,10,406,1\n\
               huge,0,0,2,2,407,1\n\
               ../escape,0,0,2,2,408,1\n";
    let req = multipart(
        "/api/gallery/import",
        &[("images_dir", "images")],
        ("csv", "train.csv", csv.as_bytes()),
    );
    let (status, report, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["imported"], 2);
    assert_eq!(report["failed"], 4);
    assert_eq!(report["errors"][0]["imageId"], "broken");
    assert_eq!(report["errors"][1]["imageId"], "zzz");
    assert_eq!(report["errors"][2]["imageId"], "huge");
    assert!(report["errors"][2]["message"]
        .as_str()
        .unwrap()
        .contains("3 МиБ"));
    assert_eq!(report["errors"][3]["imageId"], "../escape");
    assert!(report["errors"][3]["message"]
        .as_str()
        .unwrap()
        .starts_with("image_id:"));

    let (status, page, _) = send(&c.state, get("/api/gallery?page=1&pageSize=1")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(page["data"].as_array().unwrap().len(), 1);
    assert_eq!(page["data"][0]["imageId"], "bbb");
    assert_eq!(page["data"][0]["bbox"]["w"], 290, "bbox обрезан по кадру");
    assert_eq!(page["pagination"]["page"], 1);
    assert_eq!(page["pagination"]["pageSize"], 1);
    assert_eq!(page["pagination"]["totalItems"], 2);
    assert_eq!(page["pagination"]["totalPages"], 2);

    let id = page["data"][0]["id"].as_i64().unwrap();
    let (status, item, _) = send(&c.state, get(&format!("/api/gallery/{id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(item["id"], id);
    assert_eq!(item["vehicleId"], "404");

    let (status, body, _) = send(&c.state, get("/api/gallery/999999")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "NOT_FOUND");
}

#[tokio::test]
async fn import_rejects_escaping_images_dir_and_bad_csv() {
    let c = ctx();
    let req = multipart(
        "/api/gallery/import",
        &[("images_dir", "../secret")],
        ("csv", "x.csv", b"image_id,x,y,w,h\n"),
    );
    let (status, body, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");

    let req = multipart(
        "/api/gallery/import",
        &[],
        ("csv", "x.csv", b"image_id,x\naaa,notanumber\n"),
    );
    let (status, body, _) = send(&c.state, req).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .starts_with("csv:"));
}

#[tokio::test]
async fn crop_files_are_served_from_crops_dir() {
    let c = ctx();
    crop_file(&c.state.crops_dir, "gallery-crops/x.jpg");
    let (status, headers, bytes) =
        send_raw(&c.state, get("/files/crops/gallery-crops/x.jpg")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!bytes.is_empty());
    assert_eq!(headers[header::CONTENT_TYPE], "image/jpeg");
}

#[tokio::test]
async fn plate_can_be_set_and_cleared() {
    let c = ctx();
    let (_, item) = post_gallery(&c.state, r#"{"x":0,"y":0,"w":40,"h":30}"#).await;
    let id = item["id"].as_i64().unwrap();

    let (status, body, _) = send(
        &c.state,
        put_json(
            &format!("/api/gallery/{id}/plate"),
            r#"{"plate":" А123ВС77 "}"#,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["plate"], "А123ВС77");

    let (status, body, _) = send(
        &c.state,
        put_json(&format!("/api/gallery/{id}/plate"), r#"{"plate":null}"#),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["plate"].is_null());

    let (status, _, _) = send(
        &c.state,
        put_json(&format!("/api/gallery/{id}/plate"), r#"{"plate":""}"#),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let long = "X".repeat(33);
    let (status, _, _) = send(
        &c.state,
        put_json(
            &format!("/api/gallery/{id}/plate"),
            &format!(r#"{{"plate":"{long}"}}"#),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn plate_on_unknown_item_is_404() {
    let c = ctx();
    let (status, body, _) = send(
        &c.state,
        put_json("/api/gallery/424242/plate", r#"{"plate":"X"}"#),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "NOT_FOUND");
}
