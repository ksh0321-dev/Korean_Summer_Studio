"""여러 클립 지우기·붙여넣기·복제."""

from ks_studio.core.edits import delete_clips, duplicate, follow_main, paste
from ks_studio.core.project import Clip, Placement, Project


def base() -> Project:
    # V1: A[0,30) B[30,90)   V2: X[40,50) · Y[60,70)
    return Project(
        clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 60, id="B")),
        overlays=(Placement(Clip("x.mp4", 0, 10, id="X"), 2, 40), Placement(Clip("y.mp4", 0, 10, id="Y"), 2, 60)),
    )


def layout(p: Project) -> list[tuple[int, int, str]]:
    return sorted((x.track, x.start, x.clip.source) for x in p.placed())


def test_delete_several_across_tracks():
    p = delete_clips(base(), {"A", "Y"})
    assert layout(p) == [(1, 0, "b.mp4"), (2, 40, "x.mp4")]


def test_delete_several_with_linking():
    before = base()
    p = follow_main(before, delete_clips(before, {"A"}))
    assert layout(p) == [(1, 0, "b.mp4"), (2, 10, "x.mp4"), (2, 30, "y.mp4")]


def test_paste_main_clip_at_edit_point_and_overlays_keep_spacing():
    p = base()
    items = [x for x in p.placed() if x.clip.id in {"A", "X", "Y"}]  # A[0,30) X[40,50) Y[60,70)
    q = paste(p, items, 90)  # V1 끝에 붙여넣기
    assert [c.source for c in q.clips] == ["a.mp4", "b.mp4", "a.mp4"]
    assert (2, 130, "x.mp4") in layout(q) and (2, 150, "y.mp4") in layout(q)  # 기준(0)에서 40·60 떨어진 자리
    assert len({x.clip.id for x in q.placed()}) == 7  # 전부 새 id


def test_paste_in_middle_of_main_clip_goes_after_it():
    p = base()
    q = paste(p, [x for x in p.placed() if x.clip.id == "A"], 50)  # B 한가운데 → B 뒤
    assert [c.source for c in q.clips] == ["a.mp4", "b.mp4", "a.mp4"]


def test_paste_overlay_onto_occupied_spot_goes_up():
    p = base()
    q = paste(p, [x for x in p.placed() if x.clip.id == "X"], 60)  # Y[60,70) 자리
    assert (3, 60, "x.mp4") in layout(q)


def test_duplicate_places_copy_right_after():
    q = duplicate(base(), {"A", "X"})
    assert [c.source for c in q.clips] == ["a.mp4", "a.mp4", "b.mp4"]
    assert (2, 50, "x.mp4") in layout(q)  # X[40,50) 바로 뒤


def test_duplicate_neighbouring_overlays_side_by_side():
    q = duplicate(base(), {"Y", "X"})
    xs = [(t, s) for t, s, src in layout(q) if src == "x.mp4"]
    assert xs == [(2, 40), (2, 50)]
    ys = [(t, s) for t, s, src in layout(q) if src == "y.mp4"]
    assert ys == [(2, 60), (2, 70)]
