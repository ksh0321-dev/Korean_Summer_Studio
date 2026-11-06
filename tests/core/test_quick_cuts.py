"""빠른 컷 편집: 재생 헤드 앞/뒤 잘라내기(Q/W), 구간 지우기(I/O), 편집점 이동."""

from ks_studio.core.edits import delete_range, edit_points, follow_main, trim_to
from ks_studio.core.project import Clip, Placement, Project


def base() -> Project:
    # V1: A[0,30) B[30,90)   V2: X[40,50) on B · Y[100,110) V1 끝 뒤
    return Project(
        clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 60, id="B")),
        overlays=(Placement(Clip("x.mp4", 0, 10, id="X"), 2, 40), Placement(Clip("y.mp4", 0, 10, id="Y"), 2, 100)),
    )


def shown(p: Project, t: int) -> tuple[str, int] | None:
    seg = next((s for s in p.visible_segments() if s.start <= t < s.end), None)
    return (seg.clip.source, p.source_frame(seg.clip, t - seg.clip_start)) if seg and seg.clip else None


def main(p: Project) -> list[tuple[str, int, int]]:
    return [(c.source, c.in_frame, c.out_frame) for c in p.clips]


def test_q_removes_front_of_main_clip_and_closes_gap():
    p = trim_to(base(), 50, "left", "B")
    assert main(p) == [("a.mp4", 0, 30), ("b.mp4", 20, 60)]
    assert p.main_duration == 70


def test_w_removes_back_of_main_clip():
    p = trim_to(base(), 50, "right", "B")
    assert main(p) == [("a.mp4", 0, 30), ("b.mp4", 0, 20)]


def test_q_on_main_with_linking_drops_overlay_on_removed_part():
    """X는 B의 잘려 나간 앞부분(40) 위에 있었다 → 같이 지워진다. V1 끝 뒤의 Y는 V1 끝을 따라 당겨진다."""
    before = base()
    p = follow_main(before, trim_to(before, 50, "left", "B"))
    assert p.find("X") is None
    assert p.find("Y").start == 80


def test_q_w_on_overlay_keep_the_rest_in_place():
    p = trim_to(base(), 45, "left", "X")
    assert [(o.start, o.clip.in_frame, o.clip.out_frame) for o in p.overlays if o.clip.source == "x.mp4"] == [(45, 5, 10)]
    p = trim_to(base(), 45, "right", "X")
    assert [(o.start, o.clip.in_frame, o.clip.out_frame) for o in p.overlays if o.clip.source == "x.mp4"] == [(40, 0, 5)]
    assert p.main_duration == 90  # V1은 그대로


def test_trim_outside_clip_does_nothing():
    p = base()
    assert trim_to(p, 30, "left", "B") is p  # 클립 시작
    assert trim_to(p, 95, "left", "B") is p


def test_delete_range_removes_time_on_every_track():
    before = base()
    p = delete_range(before, 35, 45)
    assert p.duration == before.duration - 10
    for t in range(p.duration):  # 구간 앞은 그대로, 구간 뒤는 10프레임 당겨진 화면
        expected = shown(before, t) if t < 35 else shown(before, t + 10)
        assert shown(p, t) == expected, t


def test_delete_range_then_linking_does_not_move_twice():
    before = base()
    p = follow_main(before, delete_range(before, 35, 45))
    assert [shown(p, t) for t in range(p.duration)] == [shown(delete_range(before, 35, 45), t) for t in range(p.duration)]


def test_delete_range_beyond_main_end():
    before = base()
    p = delete_range(before, 80, 105)  # V1 끝(90)을 넘어 Y 일부까지
    assert p.main_duration == 80
    assert [(o.start, o.clip.in_frame) for o in p.overlays if o.clip.source == "y.mp4"] == [(80, 5)]


def test_edit_points_cover_all_tracks():
    assert edit_points(base()) == [0, 30, 40, 50, 90, 100, 110]
