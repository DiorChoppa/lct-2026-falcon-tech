mod gallery_read_repository;
mod search_repository;

pub use gallery_read_repository::{CandidateRow, GalleryReadRepository};
pub use inference_client::{EmbeddingClient, Frame};
pub use search_repository::SearchRepository;
pub use storage::ObjectRepository;
