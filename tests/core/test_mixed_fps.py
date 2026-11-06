from ks_studio.core.edits import set_fps, split
from ks_studio.core.project import Clip, Project, timeline_fps_for


def test_60fps_clip_on_30fps_timeline_skips_every_other_frame():
    p = Project(fps=30, clips=(Clip("a.mp4", 0, 120, fps=60, id="A"),))  # 원본 2초
    assert p.length(p.clips[0]) == 60
    assert [p.source_frame(p.clips[0], k) for k in range(4)] == [0, 2, 4, 6]


def test_30fps_clip_on_60fps_timeline_repeats_each_frame():
    p = Project(fps=60, clips=(Clip("a.mp4", 10, 40, fps=30, id="A"),))  # 원본 1초
    assert p.length(p.clips[0]) == 60
    assert [p.source_frame(p.clips[0], k) for k in range(4)] == [10, 10, 11, 11]
    assert p.source_frame(p.clips[0], 59) == 39  # 마지막도 원본 구간 안


def test_29_97_source_keeps_real_time():
    """29.97fps를 30으로 반올림하면 1분에 약 60ms씩 소리와 어긋난다 → 원본 fps를 소수 그대로 쓴다."""
    fps = 30000 / 1001
    one_minute = round(60 * fps)
    p = Project(fps=30, clips=(Clip("a.mp4", 0, one_minute, fps=fps, id="A"),))
    clip = p.clips[0]
    end_time = p.source_time(clip, p.length(clip))
    assert abs(end_time - one_minute / fps) < 1 / 30


def test_slightly_slower_source_shows_nearest_frame_not_previous():
    """실측 갤럭시 영상 29.996fps: 내림을 쓰면 타임라인 1번에 원본 0번이 나와 계속 한 프레임 늦었다."""
    fps = 3910000 / 130349
    p = Project(fps=30, clips=(Clip("a.mp4", 0, 391, fps=fps, id="A"),))
    clip = p.clips[0]
    for local in range(p.length(clip)):
        shown_time = p.source_frame(clip, local) / fps
        assert abs(shown_time - local / 30) <= 0.5 / fps + 1e-9


def test_split_maps_timeline_position_to_source_frame():
    p = Project(fps=30, clips=(Clip("a.mp4", 0, 120, fps=60, id="A"),))
    p = split(p, 15)  # 타임라인 0.5초 = 원본 30번 프레임
    assert [(c.in_frame, c.out_frame) for c in p.clips] == [(0, 30), (30, 120)]
    assert p.duration == 60


def test_tiny_clip_never_vanishes():
    p = Project(fps=30, clips=(Clip("a.mp4", 0, 1, fps=60, id="A"),))
    assert p.length(p.clips[0]) == 1


def test_changing_timeline_fps_keeps_duration_in_seconds():
    p = Project(fps=30, clips=(Clip("a.mp4", 0, 90, fps=30), Clip("b.mp4", 0, 120, fps=60)))
    assert p.duration == 150  # 5초
    assert set_fps(p, 60).duration == 300  # 여전히 5초


def test_timeline_fps_follows_first_source():
    assert timeline_fps_for(59.94) == 60
    assert timeline_fps_for(30) == 30
    assert timeline_fps_for(25) == 30
