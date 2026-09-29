//! gRPC-контракты. Единственный источник правды — файлы в proto/.
//! Python-сервис tagger генерирует свои стабы из тех же файлов (`just proto-py`).

pub mod reid {
    pub mod v1 {
        tonic::include_proto!("reid.v1");
    }
}

pub use reid::v1::*;
