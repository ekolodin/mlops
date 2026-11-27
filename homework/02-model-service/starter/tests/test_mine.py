from io import BytesIO

import pytest
from model import MAX_PIXELS, CaptionInput
from PIL import Image
from pydantic import ValidationError


def image_bytes(size, fmt):
    out = BytesIO()
    Image.new("RGB", size).save(out, format=fmt)
    return out.getvalue()


def test_pixel_limit_boundary_accepted():
    image = image_bytes((1024, 1024), "PNG")
    assert 1024 * 1024 == MAX_PIXELS
    assert CaptionInput(image=image).image == image


def test_gif_rejected():
    with pytest.raises(ValidationError, match="Invalid image bytes|JPEG and PNG"):
        CaptionInput(image=image_bytes((8, 8), "GIF"))