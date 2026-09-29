from PIL import Image

from reid.data import load_crop


def test_load_crop_clamps_bbox_to_image_bounds(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80), "red").save(path)
    crop = load_crop(path, x=90, y=70, w=30, h=30)  # bbox вылезает за край кадра
    assert crop.size == (10, 10)
    assert crop.mode == "RGB"
