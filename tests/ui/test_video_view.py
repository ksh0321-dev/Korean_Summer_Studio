"""미리보기 위젯: 줄 끝에 여백이 붙은(연속 메모리가 아닌) 프레임도 그린다."""

import av
import numpy as np
from PySide6.QtWidgets import QApplication

from ks_studio.ui.widgets.video_view import VideoView


def test_shows_padded_frame_from_portrait_preview():
    app = QApplication.instance() or QApplication([])  # noqa: F841
    # 1440x1920 세로 결과물의 미리보기 크기(404x540)는 한 줄이 1212바이트인데 PyAV가 1248바이트로 맞춘다
    frame = av.VideoFrame.from_ndarray(np.full((1080, 1920, 3), 200, np.uint8), format="rgb24").reformat(width=404, height=540)
    rgb = frame.to_ndarray()
    assert not rgb.flags["C_CONTIGUOUS"]  # 실제로 여백이 붙은 경우를 재현
    view = VideoView(mirrored=False)
    view.show_rgb(rgb)  # 예전엔 BufferError
    assert view._image.width() == 404 and view._image.pixelColor(10, 10).red() == 200
