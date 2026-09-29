//! Фейковые gRPC-клиенты в памяти и сборка приложения для тестов без сети.
//! Каждый тестовый файл использует свою часть — отсюда allow(dead_code).
#![allow(dead_code)]

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use api_gateway::clients::{GalleryClient, InferenceClient, SearchClient};
use api_gateway::{app, AppState};
use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use http_body_util::BodyExt;
use tonic::Status;
use tower::ServiceExt;

pub const DIM: usize = 4;

/// Галерея в памяти; crop_uri — file:// под crops_root, как у gallery на общем томе.
pub struct FakeGallery {
    pub items: Mutex<Vec<proto::GalleryItem>>,
    pub crops_root: PathBuf,
    pub down: bool,
}

impl FakeGallery {
    fn insert(
        &self,
        image_id: String,
        vehicle_id: String,
        bbox: Option<proto::BBox>,
        plate: String,
    ) -> proto::GalleryItem {
        let mut items = self.items.lock().unwrap();
        let id = items.len() as i64 + 1;
        let image_id = if image_id.is_empty() {
            format!("gen-{id}")
        } else {
            image_id
        };
        let item = proto::GalleryItem {
            id,
            crop_uri: format!(
                "file://{}/gallery-crops/{image_id}.jpg",
                self.crops_root.display()
            ),
            image_id,
            vehicle_id,
            bbox,
            plate,
            tags: vec![],
            model_version: "v1".into(),
            created_at_unix: 1_700_000_000,
        };
        items.push(item.clone());
        item
    }

    fn not_found() -> Status {
        Status::not_found("gallery item not found")
    }
}

#[async_trait]
impl GalleryClient for FakeGallery {
    async fn add(&self, req: proto::AddRequest) -> Result<proto::GalleryItem, Status> {
        if self.down {
            return Err(Status::unavailable("gallery down"));
        }
        Ok(self.insert(req.image_id, req.vehicle_id, req.bbox, req.plate))
    }

    async fn import(&self, items: Vec<proto::ImportItem>) -> Result<proto::ImportResponse, Status> {
        if self.down {
            return Err(Status::unavailable("gallery down"));
        }
        for it in &items {
            self.insert(
                it.image_id.clone(),
                it.vehicle_id.clone(),
                it.bbox,
                String::new(),
            );
        }
        Ok(proto::ImportResponse {
            imported: items.len() as i32,
            failed: 0,
            errors: vec![],
        })
    }

    async fn get(&self, id: i64) -> Result<proto::GalleryItem, Status> {
        if self.down {
            return Err(Status::unavailable("gallery down"));
        }
        let items = self.items.lock().unwrap();
        items
            .iter()
            .find(|i| i.id == id)
            .cloned()
            .ok_or_else(Self::not_found)
    }

    async fn list(&self, page: u32, page_size: u32) -> Result<proto::ListResponse, Status> {
        if self.down {
            return Err(Status::unavailable("gallery down"));
        }
        let items = self.items.lock().unwrap();
        let (page, page_size) = (page.max(1) as usize, page_size.clamp(1, 200) as usize);
        let mut all: Vec<_> = items.iter().cloned().collect();
        all.reverse();
        Ok(proto::ListResponse {
            items: all
                .into_iter()
                .skip((page - 1) * page_size)
                .take(page_size)
                .collect(),
            total_items: items.len() as u64,
        })
    }

    async fn set_plate(&self, id: i64, plate: String) -> Result<proto::GalleryItem, Status> {
        if self.down {
            return Err(Status::unavailable("gallery down"));
        }
        let mut items = self.items.lock().unwrap();
        let item = items
            .iter_mut()
            .find(|i| i.id == id)
            .ok_or_else(Self::not_found)?;
        item.plate = plate;
        Ok(item.clone())
    }

    async fn health(&self) -> Result<(), Status> {
        if self.down {
            Err(Status::unavailable("gallery down"))
        } else {
            Ok(())
        }
    }
}

/// Возвращает заранее заданный ответ Search, Compare и CSV для одного search_id.
pub struct FakeSearch {
    pub response: proto::SearchResponse,
    pub export_id: i64,
    pub csv: Vec<u8>,
    pub compare_id: i64,
    pub compare_gallery_id: i64,
    pub compare_response: proto::CompareResponse,
    pub down: bool,
}

impl Default for FakeSearch {
    fn default() -> Self {
        FakeSearch {
            response: proto::SearchResponse {
                search_id: 7,
                query_crop_uri: String::new(),
                candidates: vec![],
                accepted: false,
            },
            export_id: 7,
            csv: b"gallery_id,score,confidence,plate\n1,0.9,0.9,A777AA77\n".to_vec(),
            compare_id: 7,
            compare_gallery_id: 1,
            compare_response: proto::CompareResponse {
                matches: vec![],
                local_score: 0.0,
                note: String::new(),
                candidate_tags: vec![],
            },
            down: false,
        }
    }
}

#[async_trait]
impl SearchClient for FakeSearch {
    async fn search(&self, _req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
        if self.down {
            return Err(Status::unavailable("search down"));
        }
        Ok(self.response.clone())
    }

    async fn compare(
        &self,
        search_id: i64,
        gallery_id: i64,
    ) -> Result<proto::CompareResponse, Status> {
        if self.down {
            return Err(Status::unavailable("search down"));
        }
        if search_id != self.compare_id || gallery_id != self.compare_gallery_id {
            return Err(Status::not_found(format!(
                "search {search_id} or gallery {gallery_id} not found"
            )));
        }
        Ok(self.compare_response.clone())
    }

    async fn export(&self, search_id: i64) -> Result<Vec<u8>, Status> {
        if self.down {
            return Err(Status::unavailable("search down"));
        }
        if search_id != self.export_id {
            return Err(Status::not_found(format!("search {search_id} not found")));
        }
        Ok(self.csv.clone())
    }

    async fn health(&self) -> Result<(), Status> {
        if self.down {
            Err(Status::unavailable("search down"))
        } else {
            Ok(())
        }
    }
}

/// Вектор [1,0,0,0] на каждый bbox, фиксированный Info.
pub struct FakeInference {
    pub down: bool,
}

#[async_trait]
impl InferenceClient for FakeInference {
    async fn embed(&self, req: proto::EmbedRequest) -> Result<proto::EmbedResponse, Status> {
        if self.down {
            return Err(Status::unavailable("inference down"));
        }
        let mut v = vec![0.0; DIM];
        v[0] = 1.0;
        let embeddings = req
            .frames
            .iter()
            .flat_map(|f| {
                f.boxes.iter().enumerate().map(|(i, _)| proto::Embedding {
                    frame_id: f.frame_id.clone(),
                    box_index: i as i32,
                    values: v.clone(),
                    patches: vec![],
                })
            })
            .collect();
        Ok(proto::EmbedResponse {
            embeddings,
            inference_us: 1234,
        })
    }

    async fn info(&self) -> Result<proto::InfoResponse, Status> {
        if self.down {
            return Err(Status::unavailable("inference down"));
        }
        Ok(proto::InfoResponse {
            model_name: "placeholder".into(),
            model_version: "0.0.0".into(),
            dim: DIM as i32,
            input_height: 256,
            input_width: 256,
            execution_provider: "cpu".into(),
            supports_patches: false,
            patch_grid_h: 0,
            patch_grid_w: 0,
            patch_dim: 0,
        })
    }

    async fn health(&self) -> Result<(), Status> {
        if self.down {
            Err(Status::unavailable("inference down"))
        } else {
            Ok(())
        }
    }
}

pub struct Ctx {
    pub state: AppState,
    pub tmp: tempfile::TempDir,
}

/// Все сервисы живы; crops_dir и dataset_dir — во временном каталоге.
pub fn ctx() -> Ctx {
    ctx_with(false, FakeSearch::default(), false)
}

pub fn ctx_with(gallery_down: bool, search: FakeSearch, inference_down: bool) -> Ctx {
    let tmp = tempfile::tempdir().unwrap();
    let crops_dir = tmp.path().join("crops");
    std::fs::create_dir_all(&crops_dir).unwrap();
    let state = AppState {
        gallery: Arc::new(FakeGallery {
            items: Mutex::new(vec![]),
            crops_root: crops_dir.clone(),
            down: gallery_down,
        }),
        search: Arc::new(search),
        inference: Arc::new(FakeInference {
            down: inference_down,
        }),
        crops_dir,
        dataset_dir: tmp.path().join("dataset"),
        threshold: 0.5,
    };
    Ctx { state, tmp }
}

pub fn jpeg(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 30]));
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

pub fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_pixel(w, h, image::Rgb([30, 30, 200]));
    let mut buf = Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

/// multipart/form-data: текстовые поля + один файл.
pub fn multipart(path: &str, fields: &[(&str, &str)], file: (&str, &str, &[u8])) -> Request<Body> {
    let boundary = "----reid-test-boundary";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .into_bytes(),
        );
    }
    let (name, filename, bytes) = file;
    body.extend(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .into_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend(format!("\r\n--{boundary}--\r\n").into_bytes());
    Request::post(path)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap()
}

pub fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

pub fn put_json(path: &str, body: &str) -> Request<Body> {
    Request::put(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Ответ целиком: статус, заголовки, тело.
pub async fn send_raw(state: &AppState, req: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>) {
    let resp = app(state.clone()).oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    (status, headers, bytes)
}

pub async fn send(
    state: &AppState,
    req: Request<Body>,
) -> (StatusCode, serde_json::Value, Vec<u8>) {
    let (status, _, bytes) = send_raw(state, req).await;
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json, bytes)
}

/// PNG-заголовок + мусор до `len` байт: `frame_dimensions` его распознаёт, а
/// лимит размера кадра — нет.
pub fn oversized_png(len: usize) -> Vec<u8> {
    let mut bytes = png(4, 4);
    bytes.resize(len, 0);
    bytes
}

/// Кадр 200×100 + bbox через POST /api/gallery.
pub async fn post_gallery(state: &AppState, bbox: &str) -> (StatusCode, serde_json::Value) {
    let req = multipart(
        "/api/gallery",
        &[("bbox", bbox), ("vehicle_id", "403"), ("plate", "A123BC77")],
        ("image", "frame.jpg", &jpeg(200, 100)),
    );
    let (status, json, _) = send(state, req).await;
    (status, json)
}

pub fn crop_file(crops_dir: &Path, rel: &str) -> PathBuf {
    let path = crops_dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, jpeg(20, 20)).unwrap();
    path
}
