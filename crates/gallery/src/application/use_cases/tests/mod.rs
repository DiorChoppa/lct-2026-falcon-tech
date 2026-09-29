mod fixtures;

use std::sync::Arc;

use common::BBox;
use storage::test_utils::MockObjectRepository;

use crate::application::repositories::GalleryRepository;
use crate::application::structures::ImportItem;
use crate::application::use_cases::UseCases;
use crate::test_utils::{MockEmbeddingClient, MockGalleryRepository, MockTaggerClient};
use fixtures::solid_png;
use inference_client::Embedded;

fn use_cases(
    gallery: Arc<MockGalleryRepository>,
    storage: Arc<MockObjectRepository>,
    embedding: Arc<MockEmbeddingClient>,
) -> UseCases {
    UseCases::new(
        gallery,
        storage,
        embedding,
        Arc::new(MockTaggerClient::default()),
    )
}

#[tokio::test]
async fn add_crops_embeds_and_inserts_a_new_item() {
    let gallery = Arc::new(MockGalleryRepository::default());
    let storage = Arc::new(MockObjectRepository::default());
    let embedding = Arc::new(MockEmbeddingClient::default());
    let use_cases = use_cases(gallery.clone(), storage.clone(), embedding.clone());
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };

    let item = use_cases
        .add(solid_png(), bbox, None, Some("v1".into()), None)
        .await
        .unwrap();

    assert!(!item.image_id.is_empty(), "should generate an image_id");
    assert_eq!(Some("v1".to_string()), item.vehicle_id);
    assert_eq!("v1", item.model_version);
    assert_eq!(1, storage.received_puts().len());
    assert_eq!(Some(item.clone()), gallery.get(item.id).await.unwrap());
}

#[tokio::test]
async fn add_enqueues_tagging_for_the_inserted_item() {
    let gallery = Arc::new(MockGalleryRepository::default());
    let tagger = Arc::new(MockTaggerClient::default());
    let use_cases = UseCases::new(
        gallery,
        Arc::new(MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
        tagger.clone(),
    );
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };

    let item = use_cases
        .add(solid_png(), bbox, None, None, None)
        .await
        .unwrap();

    let calls = tagger.received_enqueues();
    assert_eq!(1, calls.len());
    assert_eq!(item.id, calls[0].0);
    assert!(!calls[0].1.is_empty(), "should enqueue the crop bytes");
}

#[tokio::test]
async fn add_still_succeeds_when_tagger_is_unavailable() {
    struct FailingTaggerClient;
    #[async_trait::async_trait]
    impl crate::application::repositories::TaggerClient for FailingTaggerClient {
        async fn enqueue(&self, _item_id: i64, _crop: Vec<u8>) -> anyhow::Result<()> {
            anyhow::bail!("tagger unreachable")
        }
    }

    let use_cases = UseCases::new(
        Arc::new(MockGalleryRepository::default()),
        Arc::new(MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
        Arc::new(FailingTaggerClient),
    );
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };

    let item = use_cases
        .add(solid_png(), bbox, None, None, None)
        .await
        .unwrap();

    assert!(!item.image_id.is_empty());
}

#[tokio::test]
async fn add_keeps_given_image_id() {
    let gallery = Arc::new(MockGalleryRepository::default());
    let storage = Arc::new(MockObjectRepository::default());
    let embedding = Arc::new(MockEmbeddingClient::default());
    let use_cases = use_cases(gallery, storage, embedding);
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };

    let item = use_cases
        .add(solid_png(), bbox, Some("frame-42".into()), None, None)
        .await
        .unwrap();

    assert_eq!("frame-42", item.image_id);
}

#[tokio::test]
async fn import_reports_per_item_success_and_failure() {
    let gallery = Arc::new(MockGalleryRepository::default());
    let storage = Arc::new(MockObjectRepository::default());
    let embedding = Arc::new(MockEmbeddingClient::with_embedded(vec![
        Embedded {
            embedding: vec![1.0, 0.0],
            patches: None,
        },
        Embedded {
            embedding: vec![0.0, 1.0],
            patches: None,
        },
    ]));
    let use_cases = use_cases(gallery, storage, embedding);
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };
    let items = vec![
        ImportItem {
            image: solid_png(),
            bbox,
            image_id: "ok-1".into(),
            vehicle_id: None,
        },
        ImportItem {
            image: b"not an image".to_vec(),
            bbox,
            image_id: "bad-1".into(),
            vehicle_id: None,
        },
    ];

    let report = use_cases.import(items).await.unwrap();

    assert_eq!(1, report.imported);
    assert_eq!(1, report.failed);
    assert_eq!("bad-1", report.errors[0].0);
}

#[tokio::test]
async fn import_enqueues_tagging_only_for_successfully_inserted_items() {
    let tagger = Arc::new(MockTaggerClient::default());
    let embedding = Arc::new(MockEmbeddingClient::with_embedded(vec![
        Embedded {
            embedding: vec![1.0, 0.0],
            patches: None,
        },
        Embedded {
            embedding: vec![0.0, 1.0],
            patches: None,
        },
    ]));
    let use_cases = UseCases::new(
        Arc::new(MockGalleryRepository::default()),
        Arc::new(MockObjectRepository::default()),
        embedding,
        tagger.clone(),
    );
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };
    let items = vec![
        ImportItem {
            image: solid_png(),
            bbox,
            image_id: "ok-1".into(),
            vehicle_id: None,
        },
        ImportItem {
            image: b"not an image".to_vec(),
            bbox,
            image_id: "bad-1".into(),
            vehicle_id: None,
        },
    ];

    use_cases.import(items).await.unwrap();

    assert_eq!(1, tagger.received_enqueues().len());
}

#[tokio::test]
async fn import_returns_empty_report_for_no_items() {
    let use_cases = use_cases(
        Arc::new(MockGalleryRepository::default()),
        Arc::new(MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
    );

    let report = use_cases.import(vec![]).await.unwrap();

    assert_eq!(0, report.imported);
    assert_eq!(0, report.failed);
}

#[tokio::test]
async fn set_plate_trims_and_validates_length() {
    let gallery = Arc::new(MockGalleryRepository::default());
    let use_cases = use_cases(
        gallery.clone(),
        Arc::new(MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
    );
    let item = use_cases
        .add(
            solid_png(),
            BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            None,
            None,
            None,
        )
        .await
        .unwrap();

    let updated = use_cases
        .set_plate(item.id, Some("  A123BC ".into()))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(Some("A123BC".to_string()), updated.plate);

    let err = use_cases
        .set_plate(item.id, Some("x".repeat(33)))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("32 characters"));

    let err = use_cases
        .set_plate(item.id, Some("   ".into()))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("empty string"));
}

#[tokio::test]
async fn delete_removes_item_and_its_crop() {
    let gallery = Arc::new(MockGalleryRepository::default());
    let storage = Arc::new(MockObjectRepository::default());
    let use_cases = use_cases(
        gallery.clone(),
        storage.clone(),
        Arc::new(MockEmbeddingClient::default()),
    );
    let item = use_cases
        .add(
            solid_png(),
            BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            None,
            None,
            None,
        )
        .await
        .unwrap();

    let deleted = use_cases.delete(item.id).await.unwrap();

    assert!(deleted);
    assert_eq!(None, gallery.get(item.id).await.unwrap());
    assert_eq!(vec![item.crop_uri], storage.received_deletes());
}

#[tokio::test]
async fn delete_returns_false_for_missing_item() {
    let use_cases = use_cases(
        Arc::new(MockGalleryRepository::default()),
        Arc::new(MockObjectRepository::default()),
        Arc::new(MockEmbeddingClient::default()),
    );

    assert!(!use_cases.delete(404).await.unwrap());
}

#[tokio::test]
async fn reindex_updates_items_on_an_old_model_version() {
    use crate::domain::NewItem;

    let gallery = Arc::new(MockGalleryRepository::default());
    let storage = Arc::new(MockObjectRepository::with_bytes(solid_png()));
    let embedding = Arc::new(MockEmbeddingClient::default()); // model_version() == "v1"
    let use_cases = use_cases(gallery.clone(), storage.clone(), embedding);
    let bbox = BBox {
        x: 0,
        y: 0,
        w: 4,
        h: 4,
    };
    let stale = gallery
        .insert(NewItem {
            image_id: "stale".into(),
            vehicle_id: None,
            bbox,
            crop_uri: "mock://bucket/stale.jpg".into(),
            embedding: vec![0.0, 0.0],
            model_version: "v0".into(),
            plate: None,
        })
        .await
        .unwrap();

    let reindexed = use_cases.reindex(None).await.unwrap();

    assert_eq!(1, reindexed);
    assert_eq!(
        "v1",
        gallery.get(stale.id).await.unwrap().unwrap().model_version
    );
    assert_eq!(vec!["mock://bucket/stale.jpg"], storage.received_gets());
}

#[tokio::test]
async fn reindex_ignores_items_not_matching_only_version_filter() {
    use crate::domain::NewItem;

    let gallery = Arc::new(MockGalleryRepository::default());
    let use_cases = use_cases(
        gallery.clone(),
        Arc::new(MockObjectRepository::with_bytes(solid_png())),
        Arc::new(MockEmbeddingClient::default()),
    );
    let stale = gallery
        .insert(NewItem {
            image_id: "stale".into(),
            vehicle_id: None,
            bbox: BBox {
                x: 0,
                y: 0,
                w: 4,
                h: 4,
            },
            crop_uri: "mock://bucket/stale.jpg".into(),
            embedding: vec![0.0, 0.0],
            model_version: "v0".into(),
            plate: None,
        })
        .await
        .unwrap();

    let reindexed = use_cases.reindex(Some("v_other".into())).await.unwrap();

    assert_eq!(0, reindexed);
    assert_eq!(
        "v0",
        gallery.get(stale.id).await.unwrap().unwrap().model_version
    );
}

#[test]
fn reindex_full_image_bbox_fits_int32_wire_format() {
    // proto::BBox is int32: `u32::MAX as i32` arrived as -1 and inference
    // rejected every Reindex (live gallery, 29.09).
    let bbox = crate::application::use_cases::FULL_IMAGE_BBOX;
    assert!(i32::try_from(bbox.w).is_ok());
    assert!(i32::try_from(bbox.h).is_ok());
}
