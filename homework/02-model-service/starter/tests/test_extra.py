"""Два дополнительных теста сверх выданных: граница пикселей и чужой формат."""

from io import BytesIO

import pytest
from model import MAX_PIXELS, CaptionInput
from PIL import Image
from pydantic import ValidationError


def _png(width, height):
    out = BytesIO()
    Image.new("RGB", (width, height)).save(out, format="PNG")
    return out.getvalue()


def test_pixel_limit_boundary_accepted():
    """Ровно 1024×1024 = 1 048 576 пикселей — граница принимается, а не отклоняется."""
    image = _png(1024, 1024)
    value = CaptionInput(image=image)
    assert value.image == image
    assert 1024 * 1024 == MAX_PIXELS


def test_gif_rejected():
    """GIF — валидный формат, но не JPEG/PNG, поэтому отклоняется с понятным сообщением."""
    out = BytesIO()
    Image.new("RGB", (64, 64)).save(out, format="GIF")
    with pytest.raises(ValidationError, match="JPEG and PNG"):
        CaptionInput(image=out.getvalue())