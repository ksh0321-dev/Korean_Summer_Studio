"""내보낼 때 AI 보정: 쓰는 구간만 분석 · 캐시 · 내보낼 크기로 입히기 · 글자는 그 뒤에 · 설정 조건."""

import numpy as np
import pytest

pytest.importorskip("cv2")  # AI 패키지([ai])가 없으면 건너뛴다

import av  # noqa: E402

from ks_studio.core.beauty import AiState, BeautySettings, save_ai_state  # noqa: E402
from ks_studio.core.project import Clip, Project, TextItem  # noqa: E402
from ks_studio.core.workspace import Workspace  # noqa: E402
from ks_studio.services import timeline_beauty as tb  # noqa: E402
from ks_studio.services.beautify import track_path_for  # noqa: E402
from ks_studio.services.deliver import OutputFormat, export_timeline  # noqa: E402
from ks_studio.services.sources import SourceCache  # noqa: E402
from tests.core.test_beauty import fake_face  # noqa: E402
from tests.services.test_beautify import GRAY, make_video  # noqa: E402

FRAMES = 60
SIDE = 320  # make_video의 화면 크기 (fake_face는 1000px 기준 → × 0.32)


def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def setup(tmp_path):
    app()
    ws = Workspace(tmp_path / "proj")
    ws.root.mkdir()
    video = tmp_path / "clip.mp4"
    make_video(video, frames=FRAMES)
    source = ws.to_source(video)
    sources = SourceCache()
    sources.set_cache_dir(ws.cache_dir)
    return ws, video, source, sources


def settle(ws, source, settings: BeautySettings) -> None:
    save_ai_state(ws.ai_settings_file, AiState(settings={source: settings}))


def full_track(ws, video, settings) -> None:
    """AI 보정 탭에서 영상 전체를 분석해 둔 것처럼 (가짜 얼굴 점)."""
    path = track_path_for(ws.cache_dir, video, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack([fake_face() * (SIDE / 1000)] * FRAMES).astype(np.float32))


def export(ws, sources, project, out):
    export_timeline(project, ws, sources, out, OutputFormat(short=0))
    with av.open(str(out)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def test_needed_ranges_merge_and_add_margin():
    project = Project(clips=(Clip("a.mp4", 10, 20, id="1"), Clip("a.mp4", 22, 40, id="2"), Clip("b.mp4", 0, 5, id="3"),
                             Clip("a.mp4", 100, 110, id="4")))
    ranges = tb.needed_ranges(project, {"a.mp4"})
    m = tb.SMOOTH_MARGIN
    assert ranges == {"a.mp4": [(10 - m, 40 + m), (100 - m, 110 + m)]}


def test_span_track_missing_and_round_trip(tmp_path):
    track = tb.SpanTrack(100)
    track.done[10:20] = True
    track.landmarks[10:20] = 1.0
    assert track.missing([(0, 30), (95, 120)]) == [(0, 10), (20, 30), (95, 100)]
    path = tmp_path / "x.faces_largest.spans.npz"
    track.save(path)
    again = tb.SpanTrack.load(path, 100)
    assert again.done[10:20].all() and not again.done[:10].any() and np.allclose(again.landmarks[10:20], 1.0)


def test_work_size_only_as_big_as_output():
    assert tb.work_size((2160, 3840), (1080, 1920), Clip("a", 0, 1)) == (1080, 1920)
    assert tb.work_size((1080, 1920), (2160, 3840), Clip("a", 0, 1)) == (1080, 1920)  # 원본보다 크게는 안 한다
    assert tb.work_size((2160, 3840), (1080, 1920), Clip("a", 0, 1, scale=1.5)) == (1620, 2880)  # 키운 클립은 더 크게


def test_export_applies_beauty_before_texts(setup):
    ws, video, source, sources = setup
    settings = BeautySettings(blush=1.0, approved=True)
    settle(ws, source, settings)
    full_track(ws, video, settings)
    text = TextItem(0.0, 10.0, "TEST", 0.5, 0.1)
    project = Project(clips=(Clip(source, 0, FRAMES, id="A"),), texts=(text,))
    with_beauty = export(ws, sources, project, ws.root / "a.mp4")
    settle(ws, source, BeautySettings(blush=1.0, approved=False))  # 샘플 컷을 OK하지 않았으면 입히지 않는다
    plain = export(ws, sources, project, ws.root / "b.mp4")
    assert len(with_beauty) == len(plain) == FRAMES
    from ks_studio.core.beauty import cheek_ellipses

    cheek = cheek_ellipses(fake_face() * (SIDE / 1000))[0]
    y, x = round(cheek.cy), round(cheek.cx)
    a, b = with_beauty[30][y, x].astype(int), plain[30][y, x].astype(int)
    assert a[0] > b[0] + 10  # 볼에 블러셔
    assert abs(int(plain[30][y, x][0]) - GRAY) < 6
    top = slice(0, SIDE // 5)  # 글자는 보정 뒤에 구워져 둘이 같다
    assert np.abs(with_beauty[30][top].astype(int) - plain[30][top].astype(int)).mean() < 1.0
    assert plain[30][top].max() > 200  # 글자(흰색)가 실제로 있다


def test_analyzes_only_used_frames_and_reuses_cache(setup, monkeypatch):
    ws, video, source, sources = setup
    settings = BeautySettings(blush=1.0, approved=True)
    settle(ws, source, settings)
    calls = []

    def fake_analyze(detector, path, target, on_progress=None, start=0, count=None, everyone=None):
        calls.append((start, count))
        return np.stack([fake_face() * (SIDE / 1000)] * count)

    monkeypatch.setattr("ks_studio.services.face.analyze_video", fake_analyze)
    project = Project(clips=(Clip(source, 20, 30, id="A"),))
    beauty = tb.TimelineBeauty(project, ws, sources)
    assert beauty.active and beauty.analysis_needed() == 10 + 2 * tb.SMOOTH_MARGIN
    beauty.prepare()
    assert calls == [(20 - tb.SMOOTH_MARGIN, 10 + 2 * tb.SMOOTH_MARGIN)]
    spans = tb.spans_file(track_path_for(ws.cache_dir, video, settings))
    assert spans.exists()
    again = tb.TimelineBeauty(project, ws, sources)
    assert again.analysis_needed() == 0
    again.prepare()
    assert len(calls) == 1  # 두 번째 내보내기는 분석하지 않는다
    wider = tb.TimelineBeauty(Project(clips=(Clip(source, 20, 40, id="A"),)), ws, sources)
    wider.prepare()
    assert calls[-1] == (30 + tb.SMOOTH_MARGIN, 10)  # 늘린 만큼만


def test_unused_or_identity_settings_do_nothing(setup):
    ws, video, source, sources = setup
    settle(ws, source, BeautySettings(approved=True))  # 효과 없음
    assert not tb.TimelineBeauty(Project(clips=(Clip(source, 0, 10),)), ws, sources).active
    settle(ws, "other.mp4", BeautySettings(blush=1.0, approved=True))  # 타임라인에 없는 원본
    assert not tb.TimelineBeauty(Project(clips=(Clip(source, 0, 10),)), ws, sources).active


def test_cleanup_keeps_spans_cache(setup):
    from ks_studio.core.project import Project as P
    from ks_studio.services.cleanup import project_files

    ws, video, source, sources = setup
    settings = BeautySettings(blush=1.0, approved=True)
    settle(ws, source, settings)
    P(clips=(Clip(source, 0, 10),)).save(ws.project_file)
    _live, keep = project_files(ws)
    assert tb.spans_file(track_path_for(ws.cache_dir, video, settings)) in keep
