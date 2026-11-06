"""편집용 사본: 필요한 원본을 알아보고, 고정 fps·BT.709로 바꾸면서 소리와 화면의 시각을 지킨다.

회전·HLG·P3는 합성 파일로 만들기 어려워 tools/spikes/optimize_spike.py로 실제 휴대폰 원본에서 확인했다.
"""

from fractions import Fraction

import av
import numpy as np
import pytest

from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.optimize import inspect, make_optimized
from ks_studio.services.render import TimelineRenderer
from ks_studio.services.sources import AUDIO_RATE, SourceCache, decode_audio, optimized_path

SIZE = (320, 240)
EVENT_S = 1.0  # 이 순간에 화면이 하얘지고 '딱' 소리가 난다


def make_source(path, times: list[float], pix_fmt: str = "yuv420p", size=SIZE, with_audio: bool = True) -> None:
    """times(초)마다 프레임 하나. EVENT_S 이후 프레임은 흰색, 그 전은 어두운 회색. 소리는 EVENT_S에 짧은 펄스."""
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", options={"bf": "0"})  # 간격이 들쑥날쑥한 시각을 ms 단위로 그대로 쓴다
        video.width, video.height = size
        video.pix_fmt = pix_fmt
        video.codec_context.time_base = Fraction(1, 1000)
        video.time_base = Fraction(1, 1000)
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono") if with_audio else None
        for t in times:
            value = 235 if t >= EVENT_S else 60
            frame = av.VideoFrame.from_ndarray(np.full((size[1], size[0], 3), value, np.uint8), format="rgb24")
            frame = frame.reformat(format=pix_fmt)
            frame.pts, frame.time_base = round(t * 1000), Fraction(1, 1000)
            container.mux(video.encode(frame))
        container.mux(video.encode())
        if audio is not None:
            samples = np.zeros(int((times[-1] + 0.1) * AUDIO_RATE), np.float32)
            at = int(EVENT_S * AUDIO_RATE)
            samples[at : at + 480] = 0.8
            for i in range(0, len(samples), 1024):
                chunk = av.AudioFrame.from_ndarray(samples[None, i : i + 1024], format="flt", layout="mono")
                chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, i, Fraction(1, AUDIO_RATE)
                container.mux(audio.encode(chunk))
            container.mux(audio.encode())


def vfr_times(seconds: float) -> list[float]:
    """휴대폰처럼 간격이 들쑥날쑥: 20ms, 47ms를 번갈아 (평균 30fps 남짓)."""
    times, t, steps = [], 0.0, (0.020, 0.047)
    while t < seconds:
        times.append(round(t, 3))
        t += steps[len(times) % 2]
    return times


def brightness(path, index: int) -> float:
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return float(frame.to_ndarray(format="rgb24").mean())
    raise IndexError(index)


def test_regular_bt709_source_needs_no_copy(tmp_path):
    path = tmp_path / "cfr.mp4"
    make_source(path, [i / 30 for i in range(60)])
    assert inspect(path) is None


def test_10bit_sdr_alone_needs_no_copy(tmp_path):
    """10bit여도 색이 BT.709 SDR이고 고정 fps면 렌더러가 그대로 읽는다 (실측: 20분 4K 10bit 하나로 8분·13GB를 썼다)."""
    path = tmp_path / "deep.mp4"
    make_source(path, [i / 30 for i in range(60)], pix_fmt="yuv420p10le", with_audio=False)
    assert inspect(path) is None


def test_copy_keeps_every_frame_to_the_end(tmp_path):
    """디코더에 남은 마지막 프레임까지 꺼낸다 (예전엔 끝 0.4초를 잃었다)."""
    path, output = tmp_path / "vfr.mp4", tmp_path / "out.mp4"
    make_source(path, vfr_times(2.0))
    make_optimized(path, output, inspect(path))
    with av.open(str(output)) as container:
        frames = sum(1 for _ in container.decode(video=0))
    assert abs(frames - 2.0 * 30) <= 2


def test_vfr_becomes_exact_cfr_and_keeps_sync(tmp_path):
    path, output = tmp_path / "vfr.mp4", tmp_path / "out.mp4"
    make_source(path, vfr_times(2.0))
    plan = inspect(path)
    assert plan is not None and "가변 프레임(VFR)" in plan.reasons and plan.fps == 30
    make_optimized(path, output, plan)
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
        assert set(np.diff(pts)) == {round(1 / 30 / stream.time_base)}  # 간격이 전부 같다
    # 하얀 화면은 1초(30번째 프레임)부터, 소리 펄스도 1초에 (±1프레임 / ±2ms)
    first_white = next(i for i in range(25, 36) if brightness(output, i) > 150)
    assert abs(first_white - 30) <= 1
    audio = decode_audio(output).astype(np.float32) / 32768
    onset = int(np.flatnonzero(np.abs(audio) > 0.3)[0])
    assert abs(onset / AUDIO_RATE - EVENT_S) < 0.002


def test_falls_back_to_cpu_encoder_when_gpu_cannot_open(tmp_path):
    """NVENC는 아주 작은 화면(64x48)을 못 연다 → libx264로 만든다 (NVIDIA GPU가 없는 PC도 같은 길)."""
    path, output = tmp_path / "tiny.mp4", tmp_path / "out.mp4"
    make_source(path, vfr_times(1.0), size=(64, 48), with_audio=False)
    make_optimized(path, output, inspect(path))
    with av.open(str(output)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.video[0].frames > 0


def test_full_range_is_converted_to_tv_range_with_same_look(tmp_path):
    path, output = tmp_path / "full.mp4", tmp_path / "out.mp4"
    make_source(path, [i / 30 for i in range(45)], pix_fmt="yuvj420p", with_audio=False)
    plan = inspect(path)
    assert plan is not None and "풀 레인지" in plan.reasons
    make_optimized(path, output, plan)
    with av.open(str(output)) as container:
        frame = next(container.decode(video=0))
        assert frame.format.name == "yuv420p"
        assert abs(float(frame.to_ndarray(format="rgb24").mean()) - brightness(path, 0)) < 6  # 밝기가 그대로


def test_source_cache_reads_the_copy_once_it_exists(tmp_path):
    path = tmp_path / "vfr.mp4"
    make_source(path, vfr_times(1.0))
    sources = SourceCache()
    sources.set_cache_dir(tmp_path / "cache")
    assert sources.media(path) == path
    copy = optimized_path(tmp_path / "cache", path)
    make_optimized(path, copy, inspect(path))
    assert sources.media(path) == copy
    assert sources.info(path).fps == 30


@pytest.mark.parametrize("canvas", [(64, 36), (36, 64)])
def test_other_aspect_is_fitted_with_black_bars(tmp_path, canvas):
    """세로 원본을 가로 캔버스에 (그리고 그 반대로) 늘리지 않고 가운데에 둔다."""
    source_size = (36, 64) if canvas[0] > canvas[1] else (64, 36)
    path = tmp_path / "src.mp4"
    make_source(path, [i / 30 for i in range(40)], size=source_size, with_audio=False)
    project = Project(clips=(Clip("src.mp4", 0, 40, id="A"),))
    renderer = TimelineRenderer(project, Workspace(tmp_path), SourceCache(), canvas, "rgb24")
    _, frame, _ = next(renderer.frames(35, with_audio=False))
    rgb = frame.to_ndarray().astype(int)
    h, w, _ = rgb.shape
    assert rgb[h // 2, w // 2].mean() > 150  # 가운데는 영상
    corner = rgb[1, 1]
    edge = rgb[h // 2, 1] if canvas[0] > canvas[1] else rgb[1, w // 2]
    assert edge.max() < 10 and corner.max() < 10  # 여백은 검정
