use common::BBox;

pub struct ImportItem {
    pub image: Vec<u8>,
    pub bbox: BBox,
    pub image_id: String,
    pub vehicle_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportReport {
    pub imported: u32,
    pub failed: u32,
    pub errors: Vec<(String, String)>,
}
