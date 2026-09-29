-- Model switch: CLIP ViT-B/16 (768-D) -> ViT-L/16 DINOv3 2+2 leader (1024-D, models/model.json).
-- Vectors of different models are not comparable and pgvector cannot cast 768-D to 1024-D,
-- so existing rows are dropped (tags cascade); re-import the gallery after deploying.
DROP INDEX gallery_items_embedding_hnsw;
DELETE FROM gallery_items;
ALTER TABLE gallery_items ALTER COLUMN embedding TYPE vector(1024);
CREATE INDEX gallery_items_embedding_hnsw
    ON gallery_items USING hnsw (embedding vector_cosine_ops);
