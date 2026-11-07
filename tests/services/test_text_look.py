"""글자 2단계: 그림자 · 굵게/기울임 · 줄 맞춤 · 페이드 · 기본 제공 모양 · 여러 줄."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ks_studio.core.project import Project, TextItem  # noqa: E402
from ks_studio.core.subtitles import STYLE_TEMPLATES, CaptionStyle  # noqa: E402
from ks_studio.services.captions import caption_image, fade, text_overlay  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


BASE = CaptionStyle(size=0.08, bottom=0.0, outline_width=0.0)


def test_shadow_draws_dark_blur_and_keeps_center():
    plain = text_overlay("그림자", BASE, 0.5, 0.5, 640, 360)
    shadowed = text_overlay("그림자", replace(BASE, shadow=1.0), 0.5, 0.5, 640, 360)
    (a0, x0, y0), (a1, x1, y1) = plain, shadowed
    assert a1.shape[0] > a0.shape[0] and a1.shape[1] > a0.shape[1]  # 번질 자리
    assert abs((x0 + a0.shape[1] / 2) - (x1 + a1.shape[1] / 2)) <= 1  # 가운데는 그대로
    dark = (a1[..., 3] > 40) & (a1[..., :3].max(axis=2) < 40)
    assert dark.sum() > 50  # 검은 그림자 화소
    assert (a0[..., 3] > 40).sum() < (a1[..., 3] > 40).sum()


def test_shadow_keeps_caption_bottom():
    _, _, y0 = caption_image("자막", replace(BASE, bottom=0.1), 640, 360)
    a1, _, y1 = caption_image("자막", replace(BASE, bottom=0.1, shadow=1.0), 640, 360)
    a0 = caption_image("자막", replace(BASE, bottom=0.1), 640, 360)[0]
    margin = (a1.shape[0] - a0.shape[0]) // 2
    assert abs((y0 + a0.shape[0]) - (y1 + a1.shape[0] - margin)) <= 2  # 글 칸 아래끝이 같은 자리


def _ink_columns(array):
    cols = np.nonzero((array[..., 3] > 128).any(axis=0))[0]
    return cols.min(), cols.max()


def test_align_bold_italic_change_drawing():
    text = "아주 긴 첫째 줄입니다\n짧다"
    left = caption_image(text, replace(BASE, align="left"), 1280, 720)[0]
    right = caption_image(text, replace(BASE, align="right"), 1280, 720)[0]
    assert left.shape == right.shape
    h = left.shape[0]
    second_left = _ink_columns(left[h // 2 :])
    second_right = _ink_columns(right[h // 2 :])
    assert second_left[0] < second_right[0] - 50  # 짧은 둘째 줄이 왼쪽 / 오른쪽에 붙는다
    bold = caption_image("굵기", BASE, 640, 360)[0]
    thin = caption_image("굵기", replace(BASE, bold=False), 640, 360)[0]
    italic = caption_image("굵기", replace(BASE, italic=True), 640, 360)[0]
    assert (bold[..., 3] > 128).sum() > (thin[..., 3] > 128).sum()
    assert bold.shape != italic.shape or not np.array_equal(bold, italic)


def test_fade_opacity_and_overlay():
    item = TextItem(1.0, 3.0, "a", fade_in_ms=500, fade_out_ms=1000)
    assert item.opacity_at(1.0) == 0 and abs(item.opacity_at(1.25) - 0.5) < 1e-6 and item.opacity_at(2.0) == 1
    assert abs(item.opacity_at(2.5) - 0.5) < 1e-6
    assert TextItem(0, 1, "b").opacity_at(0.0) == 1  # 페이드 없으면 바로
    drawn = text_overlay("페이드", BASE, 0.5, 0.5, 320, 180)
    half = fade(drawn, 0.5)
    assert fade(drawn, 1.0) is drawn
    assert abs(int(half[0][..., 3].max()) - drawn[0][..., 3].max() // 2) <= 1 and half[1:] == drawn[1:]


def test_new_fields_roundtrip_and_old_files(tmp_path):
    style = replace(BASE, shadow=0.4, bold=False, italic=True, align="right")
    project = Project(texts=(TextItem(0.0, 2.0, "줄1\n줄2", style=style, fade_in_ms=300, fade_out_ms=700),))
    path = tmp_path / "project.json"
    project.save(path)
    loaded = Project.load(path).texts[0]
    assert loaded.style == style and (loaded.fade_in_ms, loaded.fade_out_ms) == (300, 700) and loaded.text == "줄1\n줄2"
    old = CaptionStyle.from_dict({"size": 0.05, "color": "#ffffff"})  # 이전 버전 모양
    assert old.shadow == 0 and old.bold and not old.italic and old.align == "center"


def test_templates_are_valid_styles():
    assert len(STYLE_TEMPLATES) >= 4
    for style in STYLE_TEMPLATES.values():
        assert isinstance(style, CaptionStyle) and style.bottom == 0.0
        assert caption_image("템플릿", style, 640, 360) is not None
