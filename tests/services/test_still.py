"""사진 가져오기(사진 → 무음 영상)와 지금 화면 저장."""

import av
import numpy as np
import pytest

pytest.importorskip("cv2")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ks_studio.services.still import image_to_video, is_image, read_image, save_still  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


def _png(path, w=101, h=60):
    image = QImage(w, h, QImage.Format.Format_RGB888)
    image.fill(0x3366CC)
    assert image.save(str(path))
    return path


def test_image_to_video_makes_even_sized_silent_clip(tmp_path):
    photo = _png(tmp_path / "photo.png")
    assert is_image(photo) and not is_image(tmp_path / "a.mp4")
    out = image_to_video(photo, tmp_path / "photo.mp4", seconds=1.0, fps=30)
    with av.open(str(out)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        assert not c.streams.audio
    assert len(frames) == 30 and frames[0].shape == (60, 100, 3)  # 홀수 폭은 짝수로
    assert abs(int(frames[0][30, 50, 2]) - 0xCC) < 12


def test_image_to_video_rejects_broken_file(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    assert read_image(bad) is None
    with pytest.raises(ValueError):
        image_to_video(bad, tmp_path / "bad.mp4")
    assert not (tmp_path / "bad.mp4").exists()


def test_save_still_roundtrip(tmp_path):
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[:, :30] = (250, 10, 10)
    out = save_still(rgb, tmp_path / "shots" / "a.png")
    back = read_image(out)
    assert back.shape == (40, 60, 3) and back[5, 5, 0] > 240 and back[5, 50, 0] < 15
