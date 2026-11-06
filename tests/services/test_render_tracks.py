"""렌더: 위 줄이 아래 줄을 덮는지 프레임마다 확인 (원본마다 색과 프레임 번호를 픽셀에 새겨 둔다).

소리는 원본마다 다른 높이(Hz)의 음을 넣어, 섞인 결과를 주파수로 가려낸다.
"""

from fractions import Fraction

import av
import numpy as np
import pytest

from dataclasses import replace

from ks_studio.core.edits import add_track, place_overlay, toggle_track
from ks_studio.core.project import Clip, Placement, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.render import TimelineRenderer
from ks_studio.services.sources import AUDIO_RATE, SourceCache

SIZE = 64
STEP = 4  # 프레임 번호 k → 밝기 k*STEP


def make_source(path, channel: int, fps: int, frames: int, tone: float | None = None) -> None:
    with av.open(str(path), "w") as container:
        video = container.add_stream("libx264", rate=fps, options={"qp": "0"})
        video.width = video.height = SIZE
        video.pix_fmt = "yuv444p"
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono") if tone else None
        for k in range(frames):
            rgb = np.zeros((SIZE, SIZE, 3), np.uint8)
            rgb[..., channel] = 255
            rgb[..., (channel + 1) % 3] = k * STEP
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = k, Fraction(1, fps)
            container.mux(video.encode(frame))
        container.mux(video.encode())
        if audio is not None:
            t = np.arange(int(frames / fps * AUDIO_RATE)) / AUDIO_RATE
            samples = (0.9 * np.sin(2 * np.pi * tone * t)).astype(np.float32)
            for i in range(0, len(samples), 1024):
                chunk = av.AudioFrame.from_ndarray(samples[None, i : i + 1024], format="flt", layout="mono")
                chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, i, Fraction(1, AUDIO_RATE)
                container.mux(audio.encode(chunk))
            container.mux(audio.encode())


def read(frame: av.VideoFrame) -> tuple[int, int] | None:
    """(색 채널, 원본 프레임 번호). 검은 화면이면 None."""
    rgb = frame.to_ndarray(format="rgb24").astype(int)
    pixel = rgb[SIZE // 2, SIZE // 2]
    if pixel.max() < 128:
        return None
    channel = int(np.argmax(pixel))
    return channel, round(pixel[(channel + 1) % 3] / STEP)


@pytest.fixture(scope="module")
def sources_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("render")
    make_source(folder / "red30.mp4", 0, 30, 60, tone=440)  # 2초
    make_source(folder / "green60.mp4", 1, 60, 60, tone=1000)  # 원본 60fps 1초
    make_source(folder / "blue30.mp4", 2, 30, 30)  # 1초, 소리 없음
    return folder


def test_upper_tracks_cover_lower_frame_by_frame(sources_dir):
    # V1 red[0,60)  V2 green[20,50)  V3 blue[35,65) → V1 끝(60) 뒤 5프레임은 V3만
    project = Project(
        fps=30,
        clips=(Clip("red30.mp4", 0, 60, fps=30, id="R"),),
        overlays=(
            Placement(Clip("green60.mp4", 0, 60, fps=60, id="G"), 2, 20),
            Placement(Clip("blue30.mp4", 0, 30, fps=30, id="B"), 3, 35),
        ),
    )
    renderer = TimelineRenderer(project, Workspace(sources_dir), SourceCache(), (SIZE, SIZE), "rgb24")
    got = [(t, read(frame), samples) for t, frame, samples in renderer.frames(0)]

    expected = []
    for t in range(65):
        if 35 <= t < 65:
            expected.append((2, t - 35))
        elif 20 <= t < 35:
            expected.append((1, 2 * (t - 20)))  # 60fps 원본 → 30fps 타임라인은 한 장씩 건너뛴다
        else:
            expected.append((0, t))
    assert [t for t, _, _ in got] == list(range(65))
    assert [shown for _, shown, _ in got] == expected

    loud = [float(np.sqrt(np.mean(samples**2))) for _, _, samples in got]
    assert min(loud[25:55]) > 0.1  # 덮여 있어도 V1 소리는 계속 난다
    assert max(loud[60:]) == 0  # V1이 끝나면 조용 (V3 원본엔 소리가 없다)


def test_start_in_the_middle_of_a_covered_main_clip(sources_dir):
    project = Project(
        fps=30,
        clips=(Clip("red30.mp4", 0, 60, fps=30, id="R"),),
        overlays=(Placement(Clip("blue30.mp4", 0, 30, fps=30, id="B"), 2, 10),),
    )
    renderer = TimelineRenderer(project, Workspace(sources_dir), SourceCache(), (SIZE, SIZE), "rgb24")
    got = [(t, read(frame)) for t, frame, _ in renderer.frames(15, with_audio=False)]
    assert got[0] == (15, (2, 5))
    assert got[25] == (40, (0, 40))  # 위 줄이 끝나면 V1의 그 시각 장면으로 돌아온다


def mixed() -> Project:
    # V1 red(440Hz)[0,60)  V2 green(1000Hz)[0,30)
    return Project(
        fps=30,
        clips=(Clip("red30.mp4", 0, 60, fps=30, id="R"),),
        overlays=(Placement(Clip("green60.mp4", 0, 60, fps=60, id="G"), 2, 0),),
    )


def render(sources_dir, project: Project, start: int = 0):
    renderer = TimelineRenderer(project, Workspace(sources_dir), SourceCache(), (SIZE, SIZE), "rgb24")
    got = list(renderer.frames(start))
    return [read(frame) for _, frame, _ in got], np.concatenate([samples for _, _, samples in got])


def level(audio: np.ndarray, hz: float) -> float:
    """audio에서 hz 성분의 세기 (0.9 진폭 사인 하나면 약 0.9)."""
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)))) * 4 / len(audio)
    return float(spectrum[round(hz * len(audio) / AUDIO_RATE)])


def middle(audio: np.ndarray) -> np.ndarray:
    """V2가 걸린 0~1초 중 앞뒤를 뺀 0.2~0.8초."""
    return audio[int(0.2 * AUDIO_RATE) : int(0.8 * AUDIO_RATE)]


def test_all_tracks_are_mixed(sources_dir):
    shown, audio = render(sources_dir, mixed())
    assert shown[10][0] == 1  # 화면은 V2
    assert level(middle(audio), 440) > 0.5  # 덮인 V1 소리도 난다
    assert level(middle(audio), 1000) > 0.5
    assert np.abs(audio).max() <= 1.0  # 0.9 + 0.9가 넘치는 부분은 잘린다
    assert np.abs(audio).max() == 1.0


def test_muted_track_is_silent_but_still_visible(sources_dir):
    shown, audio = render(sources_dir, toggle_track(mixed(), 2, "mute"))
    assert shown[10][0] == 1
    assert level(middle(audio), 1000) < 0.01
    assert level(middle(audio), 440) > 0.5
    _, audio = render(sources_dir, toggle_track(mixed(), 1, "mute"))
    assert level(middle(audio), 440) < 0.01
    assert level(middle(audio), 1000) > 0.5


def test_hidden_track_is_invisible_but_still_audible(sources_dir):
    shown, audio = render(sources_dir, toggle_track(mixed(), 2, "hide"))
    assert all(s[0] == 0 for s in shown)  # 화면은 V1
    assert level(middle(audio), 1000) > 0.5


def test_overlay_audio_offset(sources_dir):
    """V2만 소리를 켜고 +100ms 늦추면, V2가 시작하는 1초가 아니라 1.1초부터 소리가 난다."""
    base = mixed()
    green = replace(base.overlays[0].clip, audio_offset_ms=100)
    project = toggle_track(replace(base, overlays=(Placement(green, 2, 30),)), 1, "mute")
    _, audio = render(sources_dir, project)
    first = int(np.flatnonzero(np.abs(audio) > 0.05)[0])
    assert abs(first / AUDIO_RATE - 1.1) < 0.002


def test_added_top_track_covers_and_is_mixed(sources_dir):
    """줄을 5개로 늘려 V5에 놓으면 V5가 맨 위에서 덮고, 소리도 섞인다."""
    project = add_track(add_track(mixed(), 4), 5)
    project = place_overlay(project, Clip("green60.mp4", 0, 60, fps=60, id="G5"), 5, 30)
    project = toggle_track(project, 2, "mute")
    shown, audio = render(sources_dir, project)
    assert shown[10][0] == 1 and shown[40] == (1, 20)  # 0~1초 V2, 1~2초 V5 (둘 다 초록)
    late = audio[int(1.2 * AUDIO_RATE) : int(1.8 * AUDIO_RATE)]
    assert level(late, 1000) > 0.5 and level(late, 440) > 0.5


def test_clip_volume_and_fades(sources_dir):
    """-6dB면 진폭이 절반, 페이드 인 0.5초면 시작은 0에서 0.5초에 원래 크기, 페이드 아웃은 끝에서 0."""
    plain = Clip("red30.mp4", 0, 60, fps=30, id="R")
    clip = Clip("red30.mp4", 0, 60, fps=30, id="R", volume_db=-6.0206, fade_in_ms=500, fade_out_ms=500)
    _, reference = render(sources_dir, Project(fps=30, clips=(plain,)))
    _, audio = render(sources_dir, Project(fps=30, clips=(clip,)))

    def peak(samples, t0, t1):
        return float(np.abs(samples[int(t0 * AUDIO_RATE) : int(t1 * AUDIO_RATE)]).max())

    assert peak(audio, 0.7, 1.2) / peak(reference, 0.7, 1.2) == pytest.approx(0.5, abs=0.01)  # -6dB = 절반
    assert peak(audio, 0.0, 0.02) < 0.03  # 페이드 인 시작은 거의 0
    assert peak(audio, 0.24, 0.26) / peak(reference, 0.24, 0.26) == pytest.approx(0.25, abs=0.03)  # 0.25초면 페이드 절반 × 볼륨 절반
    assert peak(audio, 1.98, 2.0) < 0.03  # 페이드 아웃 끝


def pixels(sources_dir, project: Project, frames: list[int], with_audio=False):
    renderer = TimelineRenderer(project, Workspace(sources_dir), SourceCache(), (SIZE, SIZE), "rgb24")
    got = {t: (frame.to_ndarray(format="rgb24").astype(int), samples) for t, frame, samples in renderer.frames(0, with_audio)}
    return [got[t] for t in frames]


def test_picture_in_picture_position_and_opacity(sources_dir):
    """V2 파랑을 절반 크기로 왼쪽 위(0.25, 0.25)에 → 그 자리만 파랑, 나머지는 V1 빨강. 투명도 50%면 섞인다."""
    small = Clip("blue30.mp4", 0, 30, fps=30, id="B", scale=0.5, pos_x=0.25, pos_y=0.25)
    project = Project(fps=30, clips=(Clip("red30.mp4", 0, 60, fps=30, id="R"),), overlays=(Placement(small, 2, 0),))
    (rgb, _), = pixels(sources_dir, project, [10])
    q = SIZE // 4
    assert np.argmax(rgb[q, q]) == 2 and rgb[q, q][2] > 200  # 왼쪽 위 가운데 = 파랑
    assert np.argmax(rgb[3 * q, 3 * q]) == 0  # 오른쪽 아래 = 빨강
    half = Placement(replace(small, opacity=0.5), 2, 0)
    (rgb, _), = pixels(sources_dir, replace(project, overlays=(half,)), [10])
    assert 100 < rgb[q, q][0] < 160 and 100 < rgb[q, q][2] < 160  # 빨강·파랑 반반


def test_dissolve_blends_picture_and_crossfades_sound(sources_dir):
    """빨강(소리 있음) 60프레임 → 파랑(소리 없음) 30프레임, 500ms(15프레임) 디졸브: 45~60에서 서서히 파랑, 소리는 줄어든다."""
    project = Project(
        fps=30,
        clips=(Clip("red30.mp4", 0, 60, fps=30, id="R"), Clip("blue30.mp4", 0, 30, fps=30, id="B", transition_ms=500)),
    )
    assert project.duration == 75
    frames = [40, 46, 52, 58, 62]
    got = pixels(sources_dir, project, frames, with_audio=True)
    blue = [rgb[SIZE // 2, SIZE // 2][2] for rgb, _ in got]
    assert blue[0] < 20 and blue[-1] > 230  # 디졸브 전 빨강, 후 파랑
    assert blue[1] < blue[2] < blue[3]  # 겹친 동안 점점 파랑
    loud = [float(np.sqrt(np.mean(samples**2))) for _, samples in got]
    assert loud[0] > loud[2] > loud[3] > 0 and loud[4] == 0  # 빨강 소리가 디졸브 동안 줄어들고, 끝나면 없음
