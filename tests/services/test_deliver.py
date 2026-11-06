"""최종 내보내기: 출력 크기, 자르기/여백, 실제 파일(비율·자막·소리), 저해상도 미리보기, 취소."""

import threading
from fractions import Fraction

import av
import numpy as np
import pytest

pytest.importorskip("cv2")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ks_studio.core.subtitles import Cue, Subtitles  # noqa: E402
from ks_studio.services.deliver import OutputFormat, export_video, fit_filters, target_size  # noqa: E402
from ks_studio.services.optimize import Cancelled  # noqa: E402
from ks_studio.services.sources import AUDIO_RATE  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


def test_target_sizes():
    assert target_size((1920, 1080), OutputFormat("9:16", short=1080)) == (1080, 1920)
    assert target_size((1920, 1080), OutputFormat("16:9", short=720)) == (1280, 720)
    assert target_size((1920, 1080), OutputFormat("4:5", short=1080)) == (1080, 1350)
    assert target_size((1440, 1920), OutputFormat("original", short=1080)) == (1080, 1440)
    assert target_size((1280, 720), OutputFormat("original", short=1080)) == (1280, 720)  # 원본 비율은 키우지 않는다
    assert target_size((1920, 1080), OutputFormat("original"), short=800) == (1422, 800)  # 저해상도 미리보기


def test_fit_filters_crop_and_pad():
    assert fit_filters((1920, 1080), (1280, 720), "crop") == (("scale", "1280:720"),)
    assert fit_filters((1920, 1080), (1080, 1920), "crop") == (("scale", "3414:1920"), ("crop", "1080:1920:1167:0"))
    assert fit_filters((1920, 1080), (1080, 1920), "pad") == (("scale", "1080:608"), ("pad", "1080:1920:0:656:black"))


def make_wide(path, seconds=1.0, fps=30):
    """320×180, 왼쪽 절반 빨강·오른쪽 절반 파랑, 소리 있음."""
    image = np.zeros((180, 320, 3), np.uint8)
    image[:, :160] = (200, 30, 30)
    image[:, 160:] = (30, 30, 200)
    with av.open(str(path), "w", format="mp4") as c:
        v = c.add_stream("libx264", rate=fps)
        v.width, v.height, v.pix_fmt = 320, 180, "yuv420p"
        a = c.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        for k in range(round(seconds * fps)):
            f = av.VideoFrame.from_ndarray(image, format="rgb24")
            f.pts, f.time_base = k, Fraction(1, fps)
            c.mux(v.encode(f))
        c.mux(v.encode())
        samples = (0.3 * np.sin(np.arange(round(seconds * AUDIO_RATE)) * 2 * np.pi * 440 / AUDIO_RATE)).astype(np.float32)
        for i in range(0, len(samples), 1024):
            chunk = av.AudioFrame.from_ndarray(samples[None, i : i + 1024], format="flt", layout="mono")
            chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, i, Fraction(1, AUDIO_RATE)
            c.mux(a.encode(chunk))
        c.mux(a.encode())


def test_export_square_crop_with_subtitles(tmp_path):
    src, out = tmp_path / "wide.mp4", tmp_path / "square.mp4"
    make_wide(src)
    subs = Subtitles((Cue(0.5, 1.0, "Hello"),))
    plan = export_video(src, out, OutputFormat("1:1", "crop", short=180), subs)
    assert plan.size == (180, 180)
    with av.open(str(out)) as c:
        frames = [f.to_ndarray(format="rgb24").astype(int) for f in c.decode(video=0)]
        assert c.streams.audio
    assert len(frames) == 30 and frames[0].shape == (180, 180, 3)
    first, late = frames[0], frames[25]
    assert first[90, 20, 0] > 150 and first[90, 160, 2] > 150  # 가운데를 잘라 빨강|파랑 경계가 가운데
    bottom = slice(100, 170)
    assert np.abs(first[bottom] - late[bottom]).max() > 100  # 0.5초 뒤엔 아래에 자막
    assert np.abs(first[:60] - late[:60]).max() < 20  # 위쪽은 그대로


def test_preview_is_small_and_cancel_removes_file(tmp_path):
    src = tmp_path / "wide.mp4"
    make_wide(src)
    plan = export_video(src, tmp_path / "p.mp4", OutputFormat("9:16", "pad"), None, preview=True)
    assert plan.size == (800, 1422)  # 최종(1080×1920)과 같은 모양, 짧은 변 800
    plan = export_video(src, tmp_path / "p.mp4", OutputFormat("9:16", "pad", short=720), None, preview=True)
    assert plan.size == (720, 1280)  # 최종이 800보다 작으면 최종 크기
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        export_video(src, tmp_path / "c.mp4", OutputFormat(), None, cancel=cancel)
    assert not (tmp_path / "c.mp4").exists() and not (tmp_path / "c.mp4.part").exists()


def test_export_timeline_burns_captions_and_keeps_sound(tmp_path):
    from ks_studio.core.project import Clip, Project
    from ks_studio.core.workspace import Workspace
    from ks_studio.services.deliver import export_timeline
    from ks_studio.services.sources import SourceCache

    make_wide(tmp_path / "wide.mp4", seconds=1.0)
    project = Project(fps=30, clips=(Clip("wide.mp4", 0, 30),), subtitles=Subtitles((Cue(0.5, 1.0, "Hello"),)))
    out = tmp_path / "timeline.mp4"
    size = export_timeline(project, Workspace(tmp_path), SourceCache(), out, OutputFormat("1:1", "crop", short=180))
    assert size == (180, 180)
    with av.open(str(out)) as c:
        frames = [f.to_ndarray(format="rgb24").astype(int) for f in c.decode(video=0)]
        assert c.streams.audio
    assert len(frames) == 30 and frames[0].shape == (180, 180, 3)
    bottom = slice(100, 170)
    assert np.abs(frames[0][bottom] - frames[25][bottom]).max() > 100  # 0.5초 뒤엔 자막
    plain = tmp_path / "plain.mp4"
    export_timeline(project, Workspace(tmp_path), SourceCache(), plain, OutputFormat("1:1", "crop", short=180), with_subtitles=False)
    with av.open(str(plain)) as c:
        last = [f.to_ndarray(format="rgb24").astype(int) for f in c.decode(video=0)][25]
    assert np.abs(frames[0][bottom] - last[bottom]).max() < 30  # 자막 끄면 그대로
