"""자막 그림·굽기: 아래 가운데, 긴 글 줄바꿈, 자막 밖 화소는 비트까지 그대로, 그 시각의 자막만."""

import av
import numpy as np
import pytest

pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ks_studio.core.subtitles import CaptionStyle, Cue, Subtitles  # noqa: E402
from ks_studio.services.captions import CaptionBurner, burn, caption_image  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


def test_caption_sits_bottom_center_and_wraps():
    style = CaptionStyle()
    short, x, y = caption_image("Did you buy it?", style, 1920, 1080)
    assert abs((x + short.shape[1] / 2) - 960) <= 2  # 가운데
    assert abs((y + short.shape[0]) - 1080 * (1 - style.bottom)) <= 1  # 아래 여백
    assert short[..., 3].max() == 255 and short[0, 0, 3] == 0  # 글은 불투명, 가장자리는 투명
    long, _, _ = caption_image("This is a really long subtitle line that will not fit on one line of a vertical video", style, 1080, 1920)
    assert long.shape[1] <= 1080 * 0.9 + 60 and long.shape[0] > 1.8 * short.shape[0]  # 줄바꿈
    assert caption_image("  ", style, 1920, 1080) is None
    korean, _, _ = caption_image("만족했어?", style, 1920, 1080)
    assert korean[..., 3].mean() > 10  # 한글도 그려진다


def test_burn_changes_only_caption_pixels():
    rng = np.random.default_rng(0)
    frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (360, 640, 3), dtype=np.uint8), format="rgb24").reformat(format="yuv420p")
    before = frame.to_ndarray().copy()
    overlay = caption_image("Hello", CaptionStyle(), 640, 360)
    out = burn(frame, overlay).to_ndarray()
    _, x, y = overlay
    assert (out[:y] == before[:y]).all()  # 자막 위쪽 Y 줄은 그대로
    assert not (out[y : 360] == before[y:360]).all()
    rgb = av.VideoFrame.from_ndarray(out, format="yuv420p").to_ndarray(format="rgb24")
    array = overlay[0]
    ys, xs = np.nonzero(array[..., 3] == 255)
    inside = rgb[y + ys[len(ys) // 2], x + xs[len(xs) // 2]]
    assert inside.min() > 200 or inside.max() < 60  # 흰 글자 또는 검은 테두리 자리


def test_burner_uses_cue_at_frame_time():
    subs = Subtitles((Cue(1.0, 2.0, "One"),))
    burner = CaptionBurner(subs, (320, 180), 30)
    frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), 100, np.uint8), format="rgb24").reformat(format="yuv420p")
    frame.pts = 15  # 0.5초: 자막 없음
    assert burner(frame) is frame
    frame.pts = 45  # 1.5초
    assert not (burner(frame).to_ndarray() == frame.to_ndarray()).all()


def test_burn_on_height_with_odd_half():
    """1422 높이(9:16 저해상도 미리보기 800×1422): 높이 절반이 홀수라 판을 이어 붙인 배열로는 깨졌다."""
    frame = av.VideoFrame.from_ndarray(np.full((1422, 800, 3), 90, np.uint8), format="rgb24").reformat(format="yuv420p")
    out = burn(frame, caption_image("Odd height", CaptionStyle(), 800, 1422))
    rgb = out.to_ndarray(format="rgb24")
    assert rgb.shape == (1422, 800, 3) and np.abs(rgb[:600].astype(int) - 90).max() <= 2  # 위쪽은 그대로
    assert rgb[1200:1320].max() > 200  # 아래에 흰 글자


def test_bilingual_caption_draws_smaller_second_line_in_its_color():
    style = CaptionStyle()
    one, _, y1 = caption_image("이쁘지?", style, 1080, 1920)
    two, _, y2 = caption_image("이쁘지?", style, 1080, 1920, "Isn't it pretty?")
    assert two.shape[0] > one.shape[0] and y2 < y1  # 두 줄 — 아래 여백은 그대로라 위로 자란다
    lower = two[one.shape[0] :]
    filled = lower[lower[..., 3] == 255][:, :3].astype(int)
    yellowish = (filled[:, 0] > 200) & (filled[:, 1] > 180) & (filled[:, 2] < 150)
    assert yellowish.sum() > 50  # 둘째 줄은 노란색 (style.second_color)
    assert caption_image("", style, 1080, 1920, "Only English") is not None


def test_box_and_outline_options():
    plain, _, _ = caption_image("Hi", CaptionStyle(outline_width=0.0), 1080, 1920)
    assert not ((plain[..., 3] == 255) & (plain[..., :3].max(axis=2) < 40)).any()  # 테두리 0 → 검은 화소 없음
    boxed, _, _ = caption_image("Hi", CaptionStyle(box=True, box_color="#0000ff", box_opacity=1.0), 1080, 1920)
    corner = boxed[boxed.shape[0] // 2, 2]
    assert corner[3] == 255 and corner[2] > 200 and corner[0] < 40  # 글자 밖 가장자리도 파란 상자
    assert boxed.shape[1] > plain.shape[1]  # 상자 안쪽 여백만큼 넓다
