"""색 보정을 영상에: .cube 캐시, lut3d 필터, 타임라인 렌더(AI 보정본 포함), 영상 하나 내보내기."""

from dataclasses import replace

import av
import numpy as np
import pytest

from ks_studio.core.grade import ColorGrade, ColorState, apply, save_color_state
from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.grade import LutFilter, color_state, cube_for, source_cube
from ks_studio.services.render import TimelineRenderer
from ks_studio.services.sources import SourceCache
from tests.services.test_render_tracks import SIZE, make_source

WARM = ColorGrade(fade=1.0, saturation=-0.5, temperature=0.5)  # 순수한 파랑도 확 바뀌는 값 (색온도만으론 빨강이 0인 파랑이 그대로다)


@pytest.fixture()
def workspace(tmp_path):
    make_source(tmp_path / "blue.mp4", 2, 30, 6)  # 파랑 화면
    ws = Workspace(tmp_path)
    ws.color_dir.mkdir()
    return ws


def test_cube_is_cached_and_identity_has_none(workspace):
    assert cube_for(workspace, None) is None and cube_for(workspace, ColorGrade()) is None
    first = cube_for(workspace, WARM)
    assert first.exists() and cube_for(workspace, WARM) == first
    assert cube_for(workspace, replace(WARM, tint=0.1)) != first


def test_lut_filter_keeps_format_and_matches_math(workspace):
    cube = cube_for(workspace, WARM)
    rgb = np.zeros((32, 48, 3), np.uint8)
    rgb[:] = (60, 120, 200)
    out = LutFilter(cube).rgb(rgb)
    expected = apply(rgb[0, 0] / 255.0, WARM) * 255
    assert np.abs(out[16, 24].astype(float) - expected).max() <= 3
    yuv = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
    yuv.pts = 7
    graded = LutFilter(cube).frame(yuv)
    assert graded.format.name == "yuv420p" and (graded.width, graded.height) == (48, 32) and graded.pts == 7


def test_timeline_uses_source_grade_and_ai_output_follows(workspace):
    def center(source):
        project = Project(fps=30, clips=(Clip(source, 0, 6, fps=30),))
        frames = list(TimelineRenderer(project, workspace, SourceCache(), (SIZE, SIZE), "rgb24").frames(with_audio=False))
        return frames[2][1].to_ndarray(format="rgb24")[SIZE // 2, SIZE // 2].astype(int)

    plain = center("blue.mp4")
    save_color_state(workspace.color_settings_file, ColorState({"blue.mp4": WARM}))
    warm = center("blue.mp4")
    assert warm[0] > plain[0] + 20 and warm[2] < plain[2] - 20  # 빨강↑ 파랑↓
    workspace.ai_dir.mkdir()
    make_source(workspace.ai_dir / "blue_보정.mp4", 2, 30, 6)
    assert source_cube(workspace, "ai/blue_보정.mp4") == source_cube(workspace, "blue.mp4")
    ai = center("ai/blue_보정.mp4")
    assert np.abs(ai - warm).max() <= 4
    assert color_state(workspace).grade_for("blue.mp4") == WARM


def test_export_video_applies_grade(workspace, tmp_path):
    from PySide6.QtWidgets import QApplication

    from ks_studio.services.deliver import OutputFormat, export_video

    QApplication.instance() or QApplication([])
    fmt = OutputFormat("original", "crop", short=0)
    plain, graded = tmp_path / "plain.mp4", tmp_path / "graded.mp4"
    export_video(workspace.root / "blue.mp4", plain, fmt, None)
    export_video(workspace.root / "blue.mp4", graded, fmt, None, grade=cube_for(workspace, WARM))

    def middle(path):
        with av.open(str(path)) as c:
            frame = next(c.decode(video=0))
        return frame.to_ndarray(format="rgb24")[SIZE // 2, SIZE // 2].astype(int)

    a, b = middle(plain), middle(graded)
    assert b[0] > a[0] + 20 and b[2] < a[2] - 20
