"""하이라이트 분할: 점수 → 하이라이트 구간 → 자를 자리 → V1 클립 나누기."""

import numpy as np
import pytest

from ks_studio.core.highlight import AUDIO_STEP, BIN, Scores, Signals
from ks_studio.core.highlight_cuts import (
    MIN_PIECE,
    HighlightBook,
    Options,
    cut_points,
    find_highlights,
    quiet_point,
    relative,
    split_project,
    thresholds,
)
from ks_studio.core.project import Clip, Placement, Project


def curve(duration: float, high: list[tuple[float, float]], low: float = 10.0, peak: float = 90.0) -> Scores:
    """보통 low점, high 구간(초)만 peak점인 표시 점수."""
    n = round(duration / BIN)
    values = np.full(n, low, np.float32)
    for a, b in high:
        values[round(a / BIN) : round(b / BIN)] = peak
    return Scores(values, {"level": values / 100}, ("sound",))


def test_user_example_twenty_seconds():
    """20초 중 6~15초가 하이라이트 → 5초·16초 언저리에서 잘라 [0~5] [5~16] [16~20]."""
    scores = curve(20, [(6, 15)])
    found = find_highlights(scores, 0, 20)
    assert [(h.start, h.end) for h in found] == [(6, 15)]
    assert found[0].score == pytest.approx(90)
    assert cut_points(found, 0, 20) == [5, 16]


def test_hysteresis_keeps_a_highlight_whole():
    """하이라이트 중간에 잠깐 기준 밑으로 내려가도 (나올 기준 위면) 한 덩어리."""
    scores = curve(30, [(10, 20)])
    scores.values[round(14 / BIN)] = 60  # 들어갈 기준(≈ 22~)은 아래지만 나올 기준(보통 + 60%)보다는 위
    enter, leave = thresholds(scores.values, 50)
    assert leave < 60 < 90 and enter <= 90
    assert [(h.start, h.end) for h in find_highlights(scores, 0, 30)] == [(10, 20)]


def test_close_highlights_merge_and_short_ones_drop():
    scores = curve(60, [(10, 15), (16, 20), (40, 41)])  # 1초 틈 → 합침, 1초짜리 → 버림
    assert [(h.start, h.end) for h in find_highlights(scores, 0, 60)] == [(10, 20)]
    assert find_highlights(scores, 0, 60, Options(min_length=0.5))[-1].start == 40


def test_flat_video_has_no_highlight():
    rng = np.random.default_rng(0)
    values = (30 + rng.normal(0, 2, 200)).astype(np.float32)  # 고른 영상 (잡음만)
    assert find_highlights(Scores(values, {}, ()), 0, 100, Options(sensitivity=100)) == []


def test_sensitivity_finds_more_when_higher():
    n = 200
    values = np.full(n, 10.0, np.float32)
    for i, height in enumerate((40, 55, 70, 85)):  # 점점 센 봉우리 4개 (각 5초)
        values[20 + i * 40 : 30 + i * 40] = height
    scores = Scores(values, {}, ())
    counts = [len(find_highlights(scores, 0, 100, Options(sensitivity=s))) for s in (0, 50, 100)]
    assert counts[0] <= counts[1] <= counts[2] and counts[0] < counts[2]


def test_thresholds_come_from_whole_video():
    """같은 영상의 다른 클립(조용한 부분)에서는 하이라이트를 지어내지 않는다."""
    scores = curve(60, [(40, 50)])
    assert find_highlights(scores, 0, 30) == []
    assert len(find_highlights(scores, 30, 60)) == 1


def test_highlight_at_clip_edge_cuts_only_the_other_side():
    scores = curve(20, [(0, 8)])
    found = find_highlights(scores, 0, 20)
    assert cut_points(found, 0, 20) == [9]  # 앞쪽은 자를 자리가 없다


def test_tiny_pieces_are_not_made():
    scores = curve(12, [(1.5, 10.5)])
    found = find_highlights(scores, 0, 12)
    assert cut_points(found, 0, 12) == []  # 0.5초·0.5초 → 앞뒤 둘 다 MIN_PIECE보다 짧다
    assert MIN_PIECE == 1.0


def test_short_piece_between_highlights_is_not_made():
    """하이라이트 사이가 3.5초면 여유 1초씩 빼고 1.5초 조각이 남는다 → 두 하이라이트를 한 컷으로."""
    found = find_highlights(curve(40, [(5, 10), (13.5, 20)]), 0, 40)
    assert len(found) == 2
    assert cut_points(found, 0, 40) == [4, 21]
    far = find_highlights(curve(40, [(5, 10), (15, 20)]), 0, 40)
    assert cut_points(far, 0, 40) == [4, 11, 14, 21]


def test_pad_option():
    found = find_highlights(curve(30, [(10, 20)]), 0, 30)
    assert cut_points(found, 0, 30, pad=2.0) == [8, 22]


def test_cut_snaps_to_quiet_moment():
    db = np.full(round(30 / AUDIO_STEP), -20.0, np.float32)  # 계속 말하는 중
    db[round(4.7 / AUDIO_STEP)] = -55  # 4.7초에 숨 고르기
    assert quiet_point(db, 5.0, 1, 29) == pytest.approx(4.7)
    assert quiet_point(db, 8.0, 1, 29) == 8.0  # 범위 안에 조용한 곳이 없으면 그대로 (다 같은 세기 → 가장 가까운 곳)
    found = find_highlights(curve(30, [(6, 15)]), 0, 30)
    assert cut_points(found, 0, 30, db) == [pytest.approx(4.7), 16]


def test_quiet_point_follows_audio_offset():
    """소리 맞춤 +0.3초: 원본 오디오 4.7초의 숨 고르기는 영상 5.0초에 들린다."""
    db = np.full(300, -20.0, np.float32)
    db[47] = -55
    assert quiet_point(db, 5.2, 1, 29, audio_offset=0.3) == pytest.approx(5.0)


def test_quiet_point_stays_inside_bounds():
    db = np.full(300, -20.0, np.float32)
    db[5] = -60  # 0.5초 — 범위(1초 이후) 밖
    assert quiet_point(db, 1.2, 1.0, 29) >= 1.0


def test_relative_scales_ordinary_to_0_and_best_moment_to_100():
    shown = relative(curve(20, [(6, 15)], low=15, peak=50))
    assert shown.values.min() == pytest.approx(0) and shown.values.max() == pytest.approx(100)
    dull = relative(curve(20, [(6, 15)], low=5, peak=20))  # 폭이 좁은(고른) 영상은 덜 키운다: 15 / 25
    assert dull.values.max() == pytest.approx(60)


# ── 타임라인에서 나누기 ──────────────────────────────────────────────────────────


def loud_signals(duration: float, loud: list[tuple[float, float]]) -> Signals:
    rng = np.random.default_rng(3)
    db = -40 + rng.normal(0, 1, round(duration / AUDIO_STEP))
    for a, b in loud:
        db[round(a / AUDIO_STEP) : round(b / AUDIO_STEP)] = -10 + rng.normal(0, 1, round((b - a) / AUDIO_STEP))
    return Signals(duration, db.astype(np.float32))


def test_split_project_cuts_v1_around_highlight_only():
    clip = Clip("a.mp4", 0, 600, id="A")  # 20초 (30fps)
    overlay = Placement(Clip("b.mp4", 0, 90, id="B"), 2, 30)
    project = Project(clips=(clip,), overlays=(overlay,))
    book = HighlightBook({"a.mp4": loud_signals(20, [(6, 15)])})
    result = split_project(project, book)
    assert result.highlights == 1 and result.cuts == 2 and result.clips == 1
    pieces = [(c.in_frame / 30, c.out_frame / 30) for c in result.project.clips]
    assert len(pieces) == 3 and pieces[0][0] == 0 and pieces[-1][1] == 20
    (_, a), (_, b) = pieces[0], pieces[1]
    assert 3.9 <= a <= 5.6 and 15.4 <= b <= 17.1  # 하이라이트 바깥 약 1초 (조용한 곳으로 ±0.5초, 점수 곡선이 0.5초쯤 번진다)
    assert result.project.overlays == (overlay,)  # 위 줄은 그대로
    assert result.project.duration == project.duration
    middle = book.clip(result.project.clips[1])
    assert middle.is_highlight and not book.clip(result.project.clips[0]).is_highlight


def test_split_project_twice_does_not_split_again():
    project = Project(clips=(Clip("a.mp4", 0, 600, id="A"),))
    book = HighlightBook({"a.mp4": loud_signals(20, [(6, 15)])})
    once = split_project(project, book).project
    twice = split_project(once, book)
    assert twice.cuts == 0 and twice.project.clips == once.clips


def test_clip_without_signals_is_left_alone():
    project = Project(clips=(Clip("a.mp4", 0, 600, id="A"), Clip("other.mp4", 0, 300, id="B")))
    book = HighlightBook({"a.mp4": loud_signals(20, [(6, 15)])})
    result = split_project(project, book)
    assert result.clips == 1 and result.project.clips[-1].id == "B"
    assert book.clip(project.clips[1]) is None


def test_trimmed_clip_uses_its_own_range_and_speed():
    """원본 10~20초만 쓰는 2배속 클립: 원본 시각 기준으로 찾고, 타임라인 프레임으로 바꿔 자른다."""
    clip = Clip("a.mp4", 300, 1200, id="A", speed=2.0)  # 원본 10~40초 → 타임라인 15초
    project = Project(clips=(clip,))
    book = HighlightBook({"a.mp4": loud_signals(60, [(20, 28)])})
    result = split_project(project, book)
    pieces = [(c.in_frame / 30, c.out_frame / 30) for c in result.project.clips]
    assert len(pieces) == 3
    assert 17.9 <= pieces[0][1] <= 19.6 and 28.4 <= pieces[1][1] <= 30.1


def test_no_highlight_means_no_change():
    project = Project(clips=(Clip("a.mp4", 0, 600, id="A"),))
    flat = Signals(20, (-30 + np.random.default_rng(1).normal(0, 0.5, 200)).astype(np.float32))
    result = split_project(project, HighlightBook({"a.mp4": flat}))
    assert result.highlights == 0 and result.project is project


def test_book_recomputes_when_options_or_signals_change():
    clip = Clip("a.mp4", 0, 600, id="A")
    book = HighlightBook({"a.mp4": loud_signals(20, [(6, 15)])})
    first = book.clip(clip)
    assert book.clip(clip) is first  # 같은 값이면 다시 계산하지 않는다
    book.set_options(Options(min_length=20))
    assert book.clip(clip).highlights == ()
    book.add("a.mp4", loud_signals(20, [(2, 4)]))
    assert book.clip(clip) is not first


# ── 나눈 뒤에 이어 할 일 ──────────────────────────────────────────────────────────


def split_example():
    from ks_studio.core.highlight_cuts import highlight_clips, low_clips, other_clips  # noqa: F401

    project = Project(clips=(Clip("a.mp4", 0, 600, id="A"), Clip("other.mp4", 0, 90, id="X")))
    book = HighlightBook({"a.mp4": loud_signals(20, [(6, 15)])})
    return split_project(project, book).project, book


def test_highlight_and_other_clips():
    from ks_studio.core.highlight_cuts import highlight_clips, other_clips

    project, book = split_example()
    a_clips = [c for c in project.clips if c.source == "a.mp4"]
    assert len(a_clips) == 3
    keep, drop = highlight_clips(project, book), other_clips(project, book)
    assert keep.ids == {a_clips[1].id} and drop.ids == {a_clips[0].id, a_clips[2].id}
    assert keep.unscored == drop.unscored == 1  # 분석하지 않은 원본(other.mp4)은 고르지 않는다
    lengths = {c.id: project.length(c) for c in project.clips}
    assert drop.frames == lengths[a_clips[0].id] + lengths[a_clips[2].id]


def test_low_clips_and_means():
    from ks_studio.core.highlight_cuts import clip_means, low_clips

    project, book = split_example()
    means = clip_means(project, book)
    assert len(means) == 3 and means[1] > max(means[0], means[2])
    assert low_clips(project, book, 0).ids == frozenset()
    assert len(low_clips(project, book, 101).ids) == 3
    middle = sorted(means)[1]
    assert len(low_clips(project, book, middle).ids) == 1  # 기준보다 '낮은' 것만
