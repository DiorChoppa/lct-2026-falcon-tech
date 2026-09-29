-- Схема владеет gallery: sqlx migrate при его старте (04-architecture.md §5).
-- Пароли ролей — dev-заглушки; для сдачи переопределяются из .env через
-- ALTER ROLE в deploy, сюда секреты не попадают.
CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'gallery_rw') THEN
        CREATE ROLE gallery_rw LOGIN PASSWORD 'gallery_rw';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'search_ro') THEN
        CREATE ROLE search_ro LOGIN PASSWORD 'search_ro';
    END IF;
END
$$;

-- Галерея: одна запись = один кроп ТС с эмбеддингом текущей model_version.
CREATE TABLE gallery_items (
    id            BIGSERIAL PRIMARY KEY,
    image_id      TEXT NOT NULL,
    vehicle_id    TEXT,
    bbox_x        INT NOT NULL,
    bbox_y        INT NOT NULL,
    bbox_w        INT NOT NULL,
    bbox_h        INT NOT NULL,
    crop_uri      TEXT NOT NULL,        -- object_store URL, см. crates/storage
    embedding     vector(768) NOT NULL, -- L2-нормирован; dim из models/model.json (CLIP ViT-B/16, 21.09)
    model_version TEXT NOT NULL,
    plate         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX gallery_items_embedding_hnsw
    ON gallery_items USING hnsw (embedding vector_cosine_ops);
CREATE INDEX gallery_items_vehicle_id ON gallery_items (vehicle_id);
CREATE INDEX gallery_items_plate ON gallery_items (plate) WHERE plate IS NOT NULL;
CREATE INDEX gallery_items_model_version ON gallery_items (model_version);

-- Теги деталей: пишет gallery по результату tagger.SetTags.
CREATE TABLE tags (
    id          BIGSERIAL PRIMARY KEY,
    item_id     BIGINT NOT NULL REFERENCES gallery_items (id) ON DELETE CASCADE,
    tag         TEXT NOT NULL,
    confidence  REAL NOT NULL,
    box_x       INT,
    box_y       INT,
    box_w       INT,
    box_h       INT
);

CREATE INDEX tags_item_id ON tags (item_id);

-- История поисков: пишет search, читают search/api-gateway (export).
CREATE TABLE searches (
    id             BIGSERIAL PRIMARY KEY,
    query_crop_uri TEXT NOT NULL,
    bbox_x         INT NOT NULL,
    bbox_y         INT NOT NULL,
    bbox_w         INT NOT NULL,
    bbox_h         INT NOT NULL,
    result         JSONB NOT NULL,
    accepted       BOOLEAN NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

GRANT ALL PRIVILEGES ON gallery_items, tags TO gallery_rw;
GRANT USAGE, SELECT ON SEQUENCE gallery_items_id_seq, tags_id_seq TO gallery_rw;
GRANT ALL PRIVILEGES ON searches TO gallery_rw;
GRANT USAGE, SELECT ON SEQUENCE searches_id_seq TO gallery_rw;

GRANT SELECT ON gallery_items, tags TO search_ro;
GRANT INSERT, SELECT ON searches TO search_ro;
GRANT USAGE, SELECT ON SEQUENCE searches_id_seq TO search_ro;
