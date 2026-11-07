"""얼굴 흐리기 화소: 상자 안 타원만 바뀌고 밖은 그대로, 영상 프레임(YUV)도 같은 자리."""

import av
import numpy as np
import pytest

pytest.importorskip("cv2")

from ks_studio.services.blur import blur_frame, blur_image  # noqa: E402


def checker(h: int = 400, w: int = 400) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    return (((xs // 4 + ys // 4) % 2) * 255).astype(np.uint8)[..., None].repeat(3, axis=2)


def test_mosaic_and_blur_only_inside_box():
    for style in ("mosaic", "blur"):
        image = checker()
        before = image.copy()
        blur_image(image, [(100, 100, 300, 300)], style)
        assert (image[:100] == before[:100]).all() and (image[:, 300:] == before[:, 300:]).all()
        assert np.abs(image[200, 190:210].astype(int) - before[200, 190:210]).sum() > 500  # 가운데는 바뀌었다
        assert (image[102, 102] == before[102, 102]).all()  # 상자 모서리 (타원 밖)는 그대로


def test_blur_frame_yuv_and_box_partly_outside():
    frame = av.VideoFrame.from_ndarray(checker(), format="rgb24").reformat(format="yuv420p")
    before = frame.to_ndarray(format="rgb24")
    after = blur_frame(frame, [(-50, 300, 150, 500)]).to_ndarray(format="rgb24")  # 화면 밖으로 걸친 상자
    assert (after[:250] == before[:250]).all()
    assert (after[320:380, 0:60] != before[320:380, 0:60]).any()
    assert blur_frame(frame, []) is frame
