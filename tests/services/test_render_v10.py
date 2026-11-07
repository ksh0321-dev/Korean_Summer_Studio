"""렌더 0.3 기능: 속도(영상·소리 높이 유지), 자르기·회전·뒤집기·색 보정·LUT, 소리 줄(배경 음악 덕킹·떼어 낸 소리), 구간 지우기."""

from dataclasses import replace

import numpy as np
import pytest

from ks_studio.core.color import grade, shape
from ks_studio.core.edits import add_sound, delete_range, detach_audio
from ks_studio.core.project import Clip, Project, Sound, TextItem
from ks_studio.core.workspace import Workspace
from ks_studio.services.render import TimelineRenderer, stretch_audio
from ks_studio.services.sources import AUDIO_RATE, SourceCache
from tests.services.test_render_tracks import SIZE, make_source, read


def pitch(samples: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return float(np.argmax(spectrum) * AUDIO_RATE / len(samples))


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("render10")
    make_source(folder / "red.mp4", 0, 30, 60, tone=440)  # 2초, 440Hz (프레임 번호를 밝기로 새겨서 64장까지)
    make_source(folder / "music.mp4", 1, 30, 60, tone=1000)  # '배경 음악' 1000Hz
    return folder


def render(media, project: Project, size=(SIZE, SIZE)):
    return list(TimelineRenderer(project, Workspace(media), SourceCache(), size, "rgb24").frames())


def test_speed_changes_length_frames_and_keeps_pitch(media):
    project = Project(fps=30, clips=(Clip("red.mp4", 0, 60, fps=30, speed=2.0),))
    assert project.length(project.clips[0]) == 30 and project.source_frame(project.clips[0], 10) == 20
    assert project.local_of(project.clips[0], 1.0) == 15  # 원본 1초 = 2배속 타임라인 0.5초
    frames = render(media, project)
    assert len(frames) == 30 and read(frames[10][1]) == (0, 20)  # 2배속: 10번째 칸에 원본 20번째
    audio = np.concatenate([a for _, _, a in frames[3:27]])
    assert abs(pitch(audio) - 440) < 20  # 높이는 그대로 (칩멍크가 아니다)


def test_stretch_audio_keeps_pitch_and_scales_length():
    t = np.arange(AUDIO_RATE) / AUDIO_RATE
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    for speed in (0.25, 0.5, 2.0, 3.0):
        out = stretch_audio(tone, speed)
        assert abs(len(out) / AUDIO_RATE - 1 / speed) < 0.06
        assert abs(pitch(out[1000:9192].astype(np.float32)) - 440) < 20
    assert stretch_audio(tone, 1.0) is tone


def test_shape_and_grade():
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[:, :30] = (200, 50, 50)  # 왼쪽 반 빨강
    assert shape(rgb, (0.25, 0, 0, 0), 0, False).shape == (40, 45, 3)  # 왼쪽 1/4 자르기
    assert shape(rgb, (0.9, 0, 0, 0), 0, False).shape == (40, 33, 3)  # 한쪽은 45%까지만
    turned = shape(rgb, (0, 0, 0, 0), 90, False)
    assert turned.shape == (60, 40, 3) and turned[0, -1, 0] == 200  # 시계 방향: 왼쪽 위 → 오른쪽 위
    assert shape(rgb, (0, 0, 0, 0), 0, True)[0, -1, 0] == 200  # 뒤집기
    gray = np.full((2, 2, 3), 128, np.uint8)
    assert grade(gray) is gray
    assert grade(gray, brightness=0.5)[0, 0, 0] > 150
    warm = grade(gray, temperature=1.0)[0, 0]
    assert warm[0] > 128 > warm[2]  # 따뜻하게: 빨강↑ 파랑↓
    mono = grade(rgb, look="mono")[0, 0]
    assert abs(int(mono[0]) - int(mono[2])) <= 1  # 흑백


def test_renderer_crops_rotates_and_grades(media):
    clip = Clip("red.mp4", 0, 30, fps=30, rotation=90, crop=(0.25, 0, 0.25, 0), saturation=-1.0)
    t, frame, _ = render(media, Project(fps=30, clips=(clip,)), size=(64, 64))[0]
    rgb = frame.to_ndarray(format="rgb24").astype(int)
    middle = rgb[32, 32]
    assert abs(middle[0] - middle[1]) < 12  # 채도 -1 → 회색
    # 좌우를 1/4씩 잘라 32×64(세로) → 90도 돌려 64×32(가로) → 64×64 칸에 위아래 여백을 두고 들어간다
    assert rgb[2, 32].max() < 30 and rgb[32, 2].max() > 60


def test_lut_file(media, tmp_path):
    cube = media / "swap.cube"
    lines = ["LUT_3D_SIZE 2"] + [f"{g} {r} {b}" for b in range(2) for g in range(2) for r in range(2)]  # 빨강↔초록
    cube.write_text("\n".join(lines))
    clip = Clip("red.mp4", 0, 30, fps=30, lut="swap.cube")
    rgb = render(media, Project(fps=30, clips=(clip,)))[0][1].to_ndarray(format="rgb24").astype(int)
    assert rgb[32, 32, 1] > 200 and rgb[32, 32, 0] < 60


def test_music_ducks_under_voice_and_detached_audio_follows_sound_track(media):
    project = Project(fps=30, clips=(Clip("red.mp4", 0, 60, fps=30, volume_db=-60),))  # 말소리 거의 없음
    quiet = add_sound(project, "music.mp4", 0, 2000)
    loud = add_sound(replace(project, clips=(Clip("red.mp4", 0, 60, fps=30),)), "music.mp4", 0, 2000)

    def music_level(p):
        audio = np.concatenate([a for _, _, a in render(media, p)[20:50]])
        spectrum = np.abs(np.fft.rfft(audio))
        return spectrum[int(1000 * len(audio) / AUDIO_RATE)]

    assert music_level(loud) < music_level(quiet) * 0.4  # 말소리가 있으면 음악이 작아진다 (-12dB 쪽)


def test_detach_audio_moves_sound_to_sound_track(media):
    clip = Clip("red.mp4", 30, 60, fps=30, audio_offset_ms=100)
    project = Project(fps=30, clips=(clip,))
    detached = detach_audio(project, clip.id)
    sound = detached.sounds[0]
    assert detached.clips[0].detached and not sound.duck
    assert (sound.start, sound.in_ms, sound.length_ms) == (0, 900, 1000)  # 원본 1초 - 오프셋 0.1초
    before = np.concatenate([a for _, _, a in render(media, project)])
    after = np.concatenate([a for _, _, a in render(media, detached)])
    assert np.abs(before - after).max() < 0.02  # 같은 소리 (이제 소리 줄에서 난다)
    assert detach_audio(detached, clip.id) is detached  # 두 번은 안 뗀다
    fast = Project(fps=30, clips=(replace(clip, speed=2.0),))
    assert detach_audio(fast, clip.id) is fast


def test_delete_range_cuts_texts_and_sounds():
    project = Project(fps=30, clips=(Clip("a.mp4", 0, 300, fps=30),),
                      texts=(TextItem(1.0, 3.0, "제목"), TextItem(6.0, 7.0, "끝")),
                      sounds=(Sound("music.mp3", 0, 0, 10_000),))
    cut = delete_range(project, 60, 120)  # 2~4초를 지운다
    assert [(t.start, t.end) for t in cut.texts] == [(1.0, 2.0), (4.0, 5.0)]
    assert [(s.start, s.in_ms, s.length_ms) for s in cut.sounds] == [(0, 0, 2000), (60, 4000, 6000)]


def test_project_v10_roundtrip(tmp_path):
    project = Project(fps=30, clips=(Clip("a.mp4", 0, 30, speed=1.5, crop=(0.1, 0, 0, 0.2), rotation=270, flip=True, look="film",
                                          lut="looks/x.cube", detached=True),),
                      texts=(TextItem(0.5, 2.0, "안녕", x=0.3, y=0.7),), sounds=(Sound("m.mp3", 15, 200, 5000, volume_db=-6),))
    project.save(tmp_path / "project.json")
    assert Project.load(tmp_path / "project.json") == project
