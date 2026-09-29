# Pillow-compatible resizing

`src/resize.rs` implements the bicubic filter and coefficient/rounding sequence documented by Pillow 12.3.0's [`src/libImaging/Resample.c`](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c): separable filtering, antialias support enlargement when reducing, signed 22-bit fixed-point coefficients, and clipping after each pass. The Rust implementation is adapted from that algorithm. Pillow's license and copyright notices are reproduced in [PILLOW-LICENSE.txt](PILLOW-LICENSE.txt).

Other dependencies are pinned in `Cargo.lock`; retain their upstream license notices when packaging binaries/native runtime libraries. `image` supplies JPEG decoding through `zune-jpeg`; this is distinct from Pillow's JPEG backend and is not claimed to be pixel-identical.
