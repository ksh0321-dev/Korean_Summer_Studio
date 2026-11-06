from pathlib import Path

from ks_studio.core.edits import (
    add_media,
    delete_clip,
    insert_clip,
    move_clip,
    remove_media,
    set_audio_offset,
    split,
    trim_clip,
)
from ks_studio.core.history import History
from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace


def three_clips() -> Project:
    # 타임라인: A[0,30) B[30,90) C[90,100)
    return Project(
        clips=(
            Clip("recordings/a.mp4", 0, 30, id="A"),
            Clip("recordings/b.mp4", 100, 160, id="B"),
            Clip("recordings/c.mp4", 5, 15, id="C"),
        )
    )


def test_duration_and_locate():
    p = three_clips()
    assert p.duration == 100
    assert p.locate(0) == (0, 0)
    assert p.locate(30) == (1, 0)
    assert p.locate(89) == (1, 59)
    assert p.locate(100) is None


def test_split_in_middle_keeps_total_and_source_continuity():
    p = split(three_clips(), 45)
    assert [p.length(c) for c in p.clips] == [30, 15, 45, 10]
    left, right = p.clips[1], p.clips[2]
    assert (left.in_frame, left.out_frame) == (100, 115)
    assert (right.in_frame, right.out_frame) == (115, 160)
    assert left.id == "B" and right.id != "B"
    assert p.duration == 100


def test_split_on_boundary_or_outside_is_noop():
    p = three_clips()
    assert split(p, 30) is p
    assert split(p, 150) is p


def test_delete_pulls_following_clips():
    p = delete_clip(three_clips(), "B")
    assert [c.id for c in p.clips] == ["A", "C"]
    assert p.locate(30) == (1, 0)


def test_move_and_insert():
    p = move_clip(three_clips(), "C", 0)
    assert [c.id for c in p.clips] == ["C", "A", "B"]
    p = insert_clip(p, 99, Clip("x.mp4", 0, 5, id="X"))
    assert [c.id for c in p.clips] == ["C", "A", "B", "X"]


def test_trim_keeps_at_least_one_frame():
    p = trim_clip(three_clips(), "A", 10, 5)
    assert (p.clips[0].in_frame, p.clips[0].out_frame) == (10, 11)


def test_audio_offset_and_media():
    p = set_audio_offset(three_clips(), "B", -40)
    assert p.clips[1].audio_offset_ms == -40
    p = add_media(add_media(p, "media/x.mp4"), "media/x.mp4")
    assert p.media == ("media/x.mp4",)
    assert remove_media(p, "media/x.mp4").media == ()


def test_original_project_is_not_modified():
    p = three_clips()
    split(p, 45)
    delete_clip(p, "A")
    assert p == three_clips()


def test_history_undo_redo_round_trip():
    h = History(three_clips())
    original = h.present
    assert h.push(split(h.present, 45))
    assert h.push(delete_clip(h.present, "A"))
    assert not h.push(h.present)  # 바뀐 게 없으면 기록하지 않음
    assert h.undo() and h.undo()
    assert h.present == original
    assert not h.can_undo
    assert h.redo()
    assert len(h.present.clips) == 4
    h.push(delete_clip(h.present, "C"))
    assert not h.can_redo  # 새 편집을 하면 다시 하기 기록은 사라진다


def test_save_load_round_trip(tmp_path):
    p = set_audio_offset(add_media(three_clips(), "D:/videos/x.mp4"), "C", 25)
    path = tmp_path / "project.json"
    p.save(path)
    assert Project.load(path) == p
    assert Project.load(tmp_path / "none.json") == Project()


def test_workspace_source_paths(tmp_path):
    ws = Workspace(tmp_path / "proj")
    inside = ws.recordings_dir / "a.mp4"
    outside = tmp_path / "elsewhere" / "b.mp4"
    assert ws.to_source(inside) == "recordings/a.mp4"
    assert ws.resolve("recordings/a.mp4") == ws.root / "recordings/a.mp4"
    assert Path(ws.to_source(outside)) == outside.resolve()
    assert ws.resolve(ws.to_source(outside)) == outside.resolve()


def test_remove_from_workspace_hides_recording_and_drops_its_clips(tmp_path):
    from ks_studio.core.edits import clips_using, remove_from_workspace
    from ks_studio.core.project import Clip, Placement, Project
    from ks_studio.core.workspace import Workspace

    a, b = Clip("recordings/a.mp4", 0, 30), Clip("media/b.mp4", 0, 30)
    p = Project(media=("media/b.mp4",), clips=(a, b), overlays=(Placement(Clip("recordings/a.mp4", 0, 10), 2, 0),))
    assert len(clips_using(p, "recordings/a.mp4")) == 2
    hidden = remove_from_workspace(p, "recordings/a.mp4", recording=True)
    assert hidden.removed == ("recordings/a.mp4",) and hidden.clips == (b,) and hidden.overlays == ()
    dropped = remove_from_workspace(p, "media/b.mp4", recording=False)
    assert dropped.media == () and dropped.clips == (a,)
    ws = Workspace(tmp_path)
    ws.recordings_dir.mkdir()
    (ws.recordings_dir / "a.mp4").write_bytes(b"x")
    hidden.save(ws.project_file)
    assert Project.load(ws.project_file).removed == ("recordings/a.mp4",)
    assert "recordings/a.mp4" not in ws.videos()  # 자막·내보내기 목록에서도 빠진다
