"""색 보정 계산: 값 → 색 변환표(3D LUT), .cube 읽고 쓰기, 저장, AI 보정본이 원본의 색을 따르기."""

import json

import numpy as np
import pytest

from ks_studio.core.grade import (
    BUILTIN_LOOKS,
    HSL_COLORS,
    LUMA,
    ColorGrade,
    ColorState,
    apply,
    apply_image,
    auto_grade,
    build_lut,
    lattice,
    load_color_state,
    load_look,
    read_cube,
    sample_lut,
    save_color_state,
    save_look,
    write_cube,
)

GRAY = np.array([0.5, 0.5, 0.5], np.float32)


def hsl_with(color: str, hue=0.0, sat=0.0, lum=0.0):
    rows = [(0.0, 0.0, 0.0)] * len(HSL_COLORS)
    rows[list(HSL_COLORS).index(color)] = (hue, sat, lum)
    return tuple(rows)


def test_identity_is_exact():
    assert ColorGrade().is_identity and ColorGrade(lut="a.cube", lut_strength=0.0).is_identity
    assert not ColorGrade(exposure=0.1).is_identity and not ColorGrade(lut="a.cube").is_identity
    assert np.abs(build_lut(ColorGrade()) - lattice()).max() < 1e-6
    image = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    assert np.array_equal(apply_image(image, build_lut(ColorGrade())), image)


def test_light_controls_move_brightness_the_right_way():
    def luma(g, color=GRAY):
        return float(apply(color, g) @ LUMA)

    assert luma(ColorGrade(exposure=0.5)) > 0.6 and luma(ColorGrade(exposure=-0.5)) < 0.4
    dark, bright = np.array([0.15] * 3, np.float32), np.array([0.85] * 3, np.float32)
    assert luma(ColorGrade(shadows=1.0), dark) > 0.2 and abs(luma(ColorGrade(shadows=1.0), bright) - 0.85) < 0.02
    assert luma(ColorGrade(highlights=-1.0), bright) < 0.8 and abs(luma(ColorGrade(highlights=-1.0), dark) - 0.15) < 0.02
    assert luma(ColorGrade(contrast=1.0), dark) < 0.15 and luma(ColorGrade(contrast=1.0), bright) > 0.85
    assert luma(ColorGrade(fade=1.0), np.zeros(3, np.float32)) > 0.08  # 검정이 뜬다


def test_color_controls():
    warm = apply(GRAY, ColorGrade(temperature=1.0))
    assert warm[0] > 0.55 and warm[2] < 0.45 and abs(float(warm @ LUMA) - 0.5) < 0.03  # 밝기는 그대로
    magenta = apply(GRAY, ColorGrade(tint=1.0))
    assert magenta[1] < magenta[0] and magenta[1] < magenta[2]
    color = np.array([0.8, 0.4, 0.3], np.float32)
    gray = apply(color, ColorGrade(saturation=-1.0))
    assert np.ptp(gray) < 1e-3
    vivid = apply(color, ColorGrade(saturation=0.5))
    assert np.ptp(vivid) > np.ptp(color)
    pale, strong = np.array([0.6, 0.5, 0.45], np.float32), np.array([0.9, 0.2, 0.1], np.float32)
    boost = lambda c: np.ptp(apply(c, ColorGrade(vibrance=1.0))) / np.ptp(c)  # noqa: E731
    assert boost(pale) > boost(strong)  # 덜 진한 색을 더 많이


def test_hsl_touches_only_that_color():
    red, blue, gray = (np.array(c, np.float32) for c in ([0.8, 0.1, 0.1], [0.1, 0.2, 0.8], [0.5, 0.5, 0.5]))
    g = ColorGrade(hsl=hsl_with("red", hue=1.0, sat=-0.5))
    moved = apply(red, g)
    assert moved[1] > 0.2 and np.ptp(moved) < np.ptp(red)  # 주황 쪽으로, 덜 진하게
    assert np.abs(apply(blue, g) - blue).max() < 1e-3 and np.abs(apply(gray, g) - gray).max() < 1e-3
    darker_blue = apply(blue, ColorGrade(hsl=hsl_with("blue", lum=-1.0)))
    assert darker_blue.max() < blue.max()


def test_lut_matches_direct_math_and_builtin_looks_are_sane():
    image = np.random.default_rng(1).integers(0, 256, (40, 60, 3), dtype=np.uint8)
    for look in BUILTIN_LOOKS.values():
        table = build_lut(look)
        assert table.shape == (33, 33, 33, 3) and 0 <= table.min() and table.max() <= 1
        direct = (apply(image / 255.0, look) * 255 + 0.5).astype(int)
        error = np.abs(apply_image(image, table).astype(int) - direct)
        assert error.max() <= 4 and error.mean() < 0.3  # 33칸 표의 보간 오차 (미리보기와 내보내기는 같은 표를 써서 서로는 같다)


def test_cube_roundtrip_and_user_lut(tmp_path):
    table = build_lut(BUILTIN_LOOKS["필름"])
    path = write_cube(tmp_path / "film.cube", table)
    assert np.abs(read_cube(path) - table).max() < 1e-4
    invert = 1 - lattice(17)
    write_cube(tmp_path / "invert.cube", invert)
    user = read_cube(tmp_path / "invert.cube")
    assert np.abs(sample_lut(user, np.array([0.2, 0.4, 0.9], np.float32)) - [0.8, 0.6, 0.1]).max() < 1e-4
    half = apply(np.array([0.2, 0.4, 0.9], np.float32), ColorGrade(lut="x.cube", lut_strength=0.5), user)
    assert np.abs(half - [0.5, 0.5, 0.5]).max() < 1e-3  # 반만 섞기
    (tmp_path / "bad.cube").write_text("LUT_3D_SIZE 4\n0 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_cube(tmp_path / "bad.cube")


def test_auto_grade_removes_color_cast():
    rng = np.random.default_rng(2)
    gray_scene = rng.uniform(0.2, 0.7, (60, 80, 1)).repeat(3, axis=2)
    blue_cast = np.clip(gray_scene * [0.8, 0.95, 1.2], 0, 1)
    image = (blue_cast * 255).astype(np.uint8)
    g = auto_grade(image)
    assert g.temperature > 0.1  # 파란 기 → 따뜻하게
    fixed = apply(image / 255.0, g).reshape(-1, 3).mean(axis=0)
    before = (image / 255.0).reshape(-1, 3).mean(axis=0)
    assert np.ptp(fixed) < np.ptp(before) * 0.5


def test_state_saving_and_ai_output_follows_original(tmp_path):
    g = ColorGrade(exposure=0.2, hsl=hsl_with("green", sat=0.3), lut="luts/a.cube", lut_strength=0.7)
    state = ColorState({"recordings/20261108_101010.mp4": g, "media/empty.mp4": ColorGrade()}, ("media/x.mp4",))
    path = tmp_path / "color" / "settings.json"
    save_color_state(path, state)
    loaded = load_color_state(path)
    assert loaded.grades == {"recordings/20261108_101010.mp4": g}  # 아무것도 안 바꾼 것은 저장하지 않는다
    assert loaded.imported == ("media/x.mp4",)
    assert loaded.grade_for("recordings/20261108_101010.mp4") == g
    assert loaded.grade_for("ai/20261108_101010_보정.mp4") == g  # AI 보정본은 원본의 색을 따른다
    assert loaded.grade_for("media/other.mp4") is None
    path.write_text("{망가짐", encoding="utf-8")
    assert load_color_state(path) == ColorState()
    assert ColorGrade.from_dict({"exposure": "많이", "hsl": [[1, 2]], "contrast": True, "saturation": 0.3}) == ColorGrade(saturation=0.3)


def test_looks_save_without_lut(tmp_path):
    save_look(tmp_path, "내 모양", ColorGrade(contrast=0.3, lut="luts/a.cube", lut_strength=0.4))
    assert load_look(tmp_path, "내 모양") == ColorGrade(contrast=0.3)
    data = json.loads((tmp_path / "내 모양.json").read_text(encoding="utf-8"))
    assert data["grade"]["lut"] == ""
