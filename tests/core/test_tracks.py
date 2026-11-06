"""여러 영상 줄: V1(자석식) + 위 줄(자유 배치). 위 줄이 아래 줄을 덮는다."""

import json

from ks_studio.core.edits import add_track, delete_clip, delete_track, move_to, place_overlay, split, toggle_track, trim_clip
from ks_studio.core.project import Clip, Placement, Project, Segment


def base() -> Project:
    # V1: A[0,30) B[30,90)   V2: X[40,60)   V3: Y[50,70)
    return Project(
        clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 60, id="B")),
        overlays=(
            Placement(Clip("x.mp4", 0, 20, id="X"), 2, 40),
            Placement(Clip("y.mp4", 0, 20, id="Y"), 3, 50),
        ),
    )


def shown(p: Project, t: int) -> tuple[str, int] | None:
    """타임라인 t에 화면에 나오는 (원본, 원본 프레임)."""
    seg = next(s for s in p.visible_segments() if s.start <= t < s.end)
    return (seg.clip.source, p.source_frame(seg.clip, t - seg.clip_start)) if seg.clip else None


def ids(segments: list[Segment]) -> list[tuple[int, int, str | None]]:
    return [(s.start, s.end, s.clip.id if s.clip else None) for s in segments]


def test_upper_track_covers_lower():
    p = base()
    assert ids(p.visible_segments()) == [(0, 30, "A"), (30, 40, "B"), (40, 50, "X"), (50, 70, "Y"), (70, 90, "B")]


def test_main_clip_keeps_its_own_position_after_being_covered():
    """덮였다가 다시 보이는 V1 클립은 덮이기 전과 같은 원본 위치를 이어서 보여준다."""
    last = base().visible_segments()[-1]
    assert (last.clip.id, last.clip_start) == ("B", 30)


def test_gap_after_main_is_black_and_counts_in_duration():
    p = Project(clips=(Clip("a.mp4", 0, 30, id="A"),), overlays=(Placement(Clip("x.mp4", 0, 10, id="X"), 2, 50),))
    assert p.duration == 60
    assert ids(p.visible_segments()) == [(0, 30, "A"), (30, 50, None), (50, 60, "X")]


def test_empty_project_has_no_segments():
    assert Project().visible_segments() == []


def test_overlap_on_same_track_is_refused():
    p = base()
    x2 = Clip("x.mp4", 0, 20, id="X2")
    assert place_overlay(p, x2, 2, 50) is p  # X[40,60)과 겹침
    assert place_overlay(p, x2, 2, 60) is not p  # 딱 붙이기는 된다
    assert place_overlay(p, x2, 2, -1) is p
    assert place_overlay(p, x2, 3, 35) is p  # Y[50,70)과 겹침


def test_move_main_clip_up_closes_gap_in_main():
    p = move_to(base(), "A", 2, 0)
    assert [c.id for c in p.clips] == ["B"]
    assert p.find("A").track == 2 and p.find("A").start == 0
    assert p.find("B").start == 0  # 자석식이라 B가 당겨진다


def test_move_overlay_down_into_main():
    p = move_to(base(), "X", 1, 1)
    assert [c.id for c in p.clips] == ["A", "X", "B"]
    assert all(o.clip.id != "X" for o in p.overlays)


def test_move_overlay_within_track_ignores_itself_but_not_others():
    p = base()
    assert move_to(p, "X", 2, 45).find("X").start == 45  # 자기 자리와 겹쳐도 된다
    assert move_to(p, "Y", 2, 50) is p  # V2의 X와 겹침
    assert move_to(p, "Y", 2, 60).find("Y").track == 2


def test_delete_overlay_leaves_gap_main_ripples():
    p = delete_clip(base(), "X")
    assert p.duration == 90 and p.find("B").start == 30
    p = delete_clip(base(), "A")
    assert p.find("B").start == 0
    assert p.find("Y").start == 50  # 위 줄은 그대로 (따라오기는 4단계)


def test_split_overlay_keeps_both_halves_in_place():
    p = split(base(), 45, "X")
    halves = sorted((o for o in p.overlays if o.track == 2), key=lambda o: o.start)
    assert [(o.start, o.clip.in_frame, o.clip.out_frame) for o in halves] == [(40, 0, 5), (45, 5, 20)]
    assert halves[0].clip.id == "X"


def test_split_without_selection_cuts_every_track_under_playhead():
    p = split(base(), 55)
    assert len(p.clips) == 3  # B가 나뉨
    assert len(p.overlays) == 4  # X, Y 모두 나뉨
    assert [shown(p, t) for t in range(90)] == [shown(base(), t) for t in range(90)]  # 화면은 그대로


def test_split_at_clip_edge_does_nothing():
    p = base()
    assert split(p, 40, "X") == p
    assert split(p, 60, "X") == p


def test_trim_overlay_left_keeps_end_right_keeps_start():
    p = trim_clip(base(), "X", 5, 20)
    assert (p.find("X").start, p.find("X").end) == (45, 60)
    p = trim_clip(base(), "X", 0, 10)
    assert (p.find("X").start, p.find("X").end) == (40, 50)


def test_trim_overlay_into_neighbor_is_refused():
    p = place_overlay(base(), Clip("z.mp4", 0, 10, id="Z"), 2, 60)  # X[40,60) 바로 뒤
    assert trim_clip(p, "X", 0, 25) is p


def test_save_load_roundtrip_and_version_1_file(tmp_path):
    path = tmp_path / "project.json"
    base().save(path)
    assert Project.load(path) == base()
    old = {"version": 1, "fps": 30, "media": [], "clips": [{"source": "a.mp4", "in_frame": 0, "out_frame": 30, "id": "A"}]}
    path.write_text(json.dumps(old), encoding="utf-8")
    assert Project.load(path).overlays == ()


def test_hidden_track_falls_through_to_lower():
    p = toggle_track(base(), 3, "hide")
    assert ids(p.visible_segments()) == [(0, 30, "A"), (30, 40, "B"), (40, 60, "X"), (60, 90, "B")]
    assert p.duration == base().duration


def test_hiding_every_track_is_black_but_keeps_length():
    p = base()
    for track in (1, 2, 3):
        p = toggle_track(p, track, "hide")
    assert ids(p.visible_segments()) == [(0, 90, None)]


def test_mute_and_hide_are_independent_toggles():
    p = toggle_track(base(), 2, "mute")
    assert p.muted == (2,) and p.hidden == ()
    assert p.visible_segments() == base().visible_segments()  # 소리를 꺼도 화면은 그대로
    assert toggle_track(p, 2, "mute") == base()


def test_track_flags_are_saved(tmp_path):
    path = tmp_path / "project.json"
    p = toggle_track(toggle_track(base(), 2, "mute"), 1, "hide")
    p.save(path)
    assert Project.load(path) == p


def test_insert_track_shifts_upper_tracks_and_keeps_picture():
    p = toggle_track(toggle_track(base(), 3, "mute"), 2, "hide")
    q = add_track(p, 3)  # V2 위에 새 줄 → 기존 V3가 V4로
    assert q.video_tracks == 4
    assert (q.find("X").track, q.find("Y").track) == (2, 4)
    assert (q.muted, q.hidden) == ((4,), (2,))
    assert [shown(q, t) for t in range(90)] == [shown(p, t) for t in range(90)]


def test_add_track_on_top_and_place_there():
    p = add_track(base(), 4)
    assert p.video_tracks == 4 and p.find("Y").track == 3
    p = place_overlay(p, Clip("z.mp4", 0, 10, id="Z"), 4, 55)
    assert (55, 65, "Z") in ids(p.visible_segments())  # 맨 위 V4가 V3를 덮는다


def test_cannot_place_on_missing_track_or_add_below_main():
    p = base()
    assert place_overlay(p, Clip("z.mp4", 0, 10, id="Z"), 4, 0) is p
    assert add_track(p, 1) is p
    assert add_track(p, 5) is p


def test_delete_track_removes_its_clips_and_lowers_upper_tracks():
    p = toggle_track(base(), 3, "mute")
    q = delete_track(p, 2)
    assert q.video_tracks == 2
    assert q.find("X") is None
    assert q.find("Y").track == 2 and q.muted == (2,)


def test_delete_track_keeps_main_and_minimum():
    p = base()
    assert delete_track(p, 1) is p
    two = delete_track(p, 3)
    assert two.video_tracks == 2
    assert delete_track(two, 2) is two  # 최소 2줄


def test_old_file_gets_enough_tracks(tmp_path):
    path = tmp_path / "project.json"
    p = add_track(add_track(base(), 4), 5)
    p = place_overlay(p, Clip("z.mp4", 0, 10, id="Z"), 5, 0)
    p.save(path)
    assert Project.load(path).video_tracks == 5
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["video_tracks"]  # 버전 3 파일 흉내
    path.write_text(json.dumps(data), encoding="utf-8")
    assert Project.load(path).video_tracks == 5
