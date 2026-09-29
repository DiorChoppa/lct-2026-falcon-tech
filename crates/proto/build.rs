//! Генерация Rust-кода из proto/. protox компилирует .proto без внешнего
//! protoc, чтобы сборка в Docker и CI не зависела от системных пакетов.
//!
//! Корень — proto/reid/v1 (не proto/): import-ы внутри .proto-файлов пишутся
//! без префикса "reid/v1/", чтобы тот же набор файлов с тем же -I протоку
//! (grpc_tools.protoc для tagger, `just proto-py`) резолвился одинаково.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../proto/reid/v1");
    let files = [
        "inference.proto",
        "tagger.proto",
        "gallery.proto",
        "search.proto",
    ];
    for f in files {
        println!("cargo:rerun-if-changed={}", root.join(f).display());
    }
    let fds = protox::compile(files, [root.to_str().unwrap()])?;
    tonic_prost_build::configure().compile_fds(fds)?;
    Ok(())
}
