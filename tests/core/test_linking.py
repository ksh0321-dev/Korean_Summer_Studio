"""연동: V1이 바뀌면 위 줄 클립이 붙어 있던 V1 장면을 따라간다 (follow_main)."""

from dataclasses import replace

from ks_studio.core.edits import (
    delete_clip,
    follow_main,
    insert_clip,
    move_clip,
    move_to,
    set_fps,
    set_linked,
    split,
    trim_clip,
)
from ks_studio.core.project import Clip, Placement, Project


def base() -> Project:
    # V1: A[0,30) B[30,90) C[90,120)
    # V2: W[10,20) on A · X[40,50) on B · Y[100,110) on C     V3: Z[130,140) V1 끝 뒤
    return Project(
        clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 60, id="B"), Clip("c.mp4", 0, 30, id="C")),
        overlays=(
            Placement(Clip("w.mp4", 0, 10, id="W"), 2, 10),
            Placement(Clip("x.mp4", 0, 10, id="X"), 2, 40),
            Placement(Clip("y.mp4", 0, 10, id="Y"), 2, 100),
            Placement(Clip("z.mp4", 0, 10, id="Z"), 3, 130),
        ),
    )


def edit(project: Project, change) -> Project:
    """편집 화면이 하는 그대로: 편집 → 연동."""
    return follow_main(project, change(project))


def starts(p: Project) -> dict[str, tuple[int, int]]:
    return {o.clip.id: (o.track, o.start) for o in p.overlays}


def under(p: Project, clip_id: str) -> tuple[str, int]:
    """위 줄 클립 시작 지점 아래 V1의 (원본, 원본 프레임)."""
    start = p.find(clip_id).start
    main = next(m for m in p.placed() if m.track == 1 and m.start <= start < m.end)
    return main.clip.source, p.source_frame(main.clip, start - main.start)


def test_delete_main_clip_pulls_later_overlays_and_deletes_its_own():
    p = edit(base(), lambda p: delete_clip(p, "B"))
    assert starts(p) == {"W": (2, 10), "Y": (2, 40), "Z": (3, 70)}
    assert under(p, "Y") == under(base(), "Y")


def test_unlinked_keeps_overlays_in_place():
    p = edit(set_linked(base(), False), lambda p: delete_clip(p, "B"))
    assert starts(p) == starts(base())


def test_reorder_carries_overlays_with_their_clip():
    p = edit(base(), lambda p: move_clip(p, "C", 0))  # C A B
    assert starts(p) == {"Y": (2, 10), "W": (2, 40), "X": (2, 70), "Z": (3, 130)}
    for clip_id in "WXY":
        assert under(p, clip_id) == under(base(), clip_id)


def test_insert_pushes_later_overlays():
    p = edit(base(), lambda p: insert_clip(p, 1, Clip("d.mp4", 0, 20, id="D")))
    assert starts(p) == {"W": (2, 10), "X": (2, 60), "Y": (2, 120), "Z": (3, 150)}


def test_trim_head_keeps_overlay_on_same_scene():
    p = edit(base(), lambda p: trim_clip(p, "B", 5, 60))  # B 앞 5프레임 잘라냄
    assert starts(p)["X"] == (2, 35) and starts(p)["Y"] == (2, 95)
    assert under(p, "X") == under(base(), "X")


def test_trim_head_past_overlay_sticks_it_to_clip_start():
    p = edit(base(), lambda p: trim_clip(p, "B", 15, 60))
    assert starts(p)["X"] == (2, 30)


def test_split_moves_nothing():
    p = edit(base(), lambda p: split(p, 45, "B"))
    assert starts(p) == starts(base())
    p = edit(base(), lambda p: split(p, 35, "B"))  # X는 뒤 조각(새 id) 위에 있다
    assert starts(p) == starts(base())


def test_lifting_main_clip_up_leaves_its_overlays():
    p = edit(base(), lambda p: move_to(p, "B", 3, 200))
    assert starts(p)["X"] == (2, 40)  # B에 붙어 있었지만 B는 위 줄로 갔다 → 제자리
    assert starts(p)["Y"] == (3, 40)  # C를 따라 40으로 → 제자리에 남은 X와 겹쳐 V3로
    assert starts(p)["B"] == (3, 200)


def test_collision_moves_follower_up_and_adds_track_if_needed():
    # V1 A[0,30) B[30,60) C[60,90) · V2: P[20,50) on A, Q[70,80) on C
    p = Project(
        clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 30, id="B"), Clip("c.mp4", 0, 30, id="C")),
        overlays=(Placement(Clip("p.mp4", 0, 30, id="P"), 2, 20), Placement(Clip("q.mp4", 0, 10, id="Q"), 2, 70)),
    )
    q = edit(p, lambda p: delete_clip(p, "B"))  # Q → 40, P[20,50)과 겹침
    assert starts(q) == {"P": (2, 20), "Q": (3, 40)}
    full = replace(p, overlays=p.overlays + (Placement(Clip("r.mp4", 0, 30, id="R"), 3, 20),))  # R도 A 위
    q = edit(full, lambda p: delete_clip(p, "B"))  # V3도 막혀 있다 → V4 생성
    assert starts(q)["Q"] == (4, 40) and q.video_tracks == 4


def test_changing_fps_keeps_overlay_time():
    p = set_fps(base(), 60)
    assert starts(p)["X"] == (2, 80)
    assert under(p, "X") == under(base(), "X")
