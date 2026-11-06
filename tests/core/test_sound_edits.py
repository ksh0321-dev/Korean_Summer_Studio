"""소리 편집: 클립 볼륨·페이드, 무음 구간 찾기·잘라내기."""

import pytest

from ks_studio.core.edits import delete_ranges, set_clip_props, split
from ks_studio.core.project import Clip, Project
from ks_studio.core.silence import silent_spans, timeline_ranges

LOUD, QUIET = 0.5, 0.001  # 0.001 ≈ -60dB


def test_silent_spans_need_min_length_and_keep_padding():
    # 10ms 단위: 0.5초 소리 · 0.3초 조용 · 0.5초 소리 · 1초 조용 · 0.2초 소리
    peaks = [LOUD] * 50 + [QUIET] * 30 + [LOUD] * 50 + [QUIET] * 100 + [LOUD] * 20
    spans = silent_spans(peaks, threshold_db=-40, min_ms=500, pad_ms=100)
    assert spans == [pytest.approx((1.4, 2.2))]  # 1.3~2.3초 조용 → 앞뒤 0.1초 남김 (0.3초 쉼은 짧아서 그대로)


def test_trailing_silence_is_found():
    spans = silent_spans([LOUD] * 10 + [QUIET] * 80, threshold_db=-40, min_ms=500, pad_ms=0)
    assert spans == [(0.1, 0.9)]


def test_padding_larger_than_silence_leaves_nothing():
    assert silent_spans([LOUD] * 10 + [QUIET] * 50 + [LOUD] * 10, threshold_db=-40, min_ms=500, pad_ms=300) == []


def test_timeline_ranges_follow_clip_position_and_audio_offset():
    # V1: A[0,30) (원본 0~1초) · B[30,90) (원본 2~4초, 오디오 +100ms 늦춤)
    project = Project(clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 60, 120, id="B", audio_offset_ms=100)))
    spans = {"A": [(0.5, 2.0)], "B": [(2.5, 3.0)]}  # 원본 오디오 시각
    ranges = timeline_ranges(project, spans)
    # A: 0.5초~클립 끝(1초) → [15,30). B: 원본 2.5초 소리는 +0.1초 늦게 들리므로 클립 안 0.6~1.1초 → [48,63)
    assert ranges == [(15, 30), (48, 63)]


def test_adjacent_ranges_across_clips_merge():
    project = Project(clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 30, id="B")))
    assert timeline_ranges(project, {"A": [(0.5, 1.0)], "B": [(0.0, 0.5)]}) == [(15, 45)]


def test_delete_ranges_from_back_keeps_positions():
    project = Project(clips=(Clip("a.mp4", 0, 100, id="A"),))
    p = delete_ranges(project, [(10, 20), (50, 60)])
    assert p.duration == 80
    assert [(c.in_frame, c.out_frame) for c in p.clips] == [(0, 10), (20, 50), (60, 100)]


def test_split_keeps_fade_in_on_left_and_fade_out_on_right():
    project = Project(clips=(Clip("a.mp4", 0, 100, id="A", fade_in_ms=500, fade_out_ms=800),))
    left, right = split(project, 50).clips
    assert (left.fade_in_ms, left.fade_out_ms) == (500, 0)
    assert (right.fade_in_ms, right.fade_out_ms) == (0, 800)


def test_set_clip_props_changes_only_given_values():
    project = Project(clips=(Clip("a.mp4", 0, 30, id="A", fade_in_ms=200), Clip("b.mp4", 0, 30, id="B")))
    p = set_clip_props(project, {"A", "B"}, volume_db=-6.0)
    assert [(c.volume_db, c.fade_in_ms) for c in p.clips] == [(-6.0, 200), (-6.0, 0)]
