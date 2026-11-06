"""화면 구성: 화면 속 화면(겹쳐 그리기), V1 디졸브(캡컷 방식 겹침), 마커."""

from ks_studio.core.edits import delete_range, edit_points, set_clip_props, set_fps, split, toggle_marker, toggle_transition
from ks_studio.core.project import Clip, Layer, Placement, Project


def pip() -> Project:
    # V1 A[0,60) · V2 X[10,40) 작게 (화면 속 화면)
    return Project(
        clips=(Clip("a.mp4", 0, 60, id="A"),),
        overlays=(Placement(Clip("x.mp4", 0, 30, id="X", scale=0.5, pos_x=0.25), 2, 10),),
    )


def test_small_overlay_is_drawn_above_main_instead_of_covering():
    segments = pip().visible_segments()
    middle = next(s for s in segments if s.start == 10)
    assert middle.clip.id == "A" and [layer.clip.id for layer in middle.above] == ["X"]
    assert [s.above for s in segments if s.start != 10] == [(), ()]


def test_full_overlay_becomes_base_and_hides_what_is_below():
    p = pip()
    full = Placement(Clip("y.mp4", 0, 10, id="Y"), 3, 0)  # 기본 크기 = 화면을 덮음
    p = Project(clips=p.clips, overlays=(*p.overlays, Placement(Clip("z.mp4", 0, 10, id="Z", opacity=0.5), 3, 20)))
    first = Project(clips=p.clips, overlays=(full,)).visible_segments()[0]
    assert first.clip.id == "Y" and first.above == ()
    at_20 = next(s for s in p.visible_segments() if s.start == 20)
    assert at_20.clip.id == "A" and [layer.clip.id for layer in at_20.above] == ["X", "Z"]  # 아래 → 위


def test_hidden_track_drops_its_layers():
    p = Project(clips=pip().clips, overlays=pip().overlays, hidden=(2,))
    assert all(s.above == () for s in p.visible_segments())


def dissolve() -> Project:
    # A 30프레임, B 60프레임, B 앞 500ms(=15프레임) 디졸브
    return Project(clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 60, id="B", transition_ms=500)))


def test_dissolve_overlaps_and_shortens_like_capcut():
    p = dissolve()
    starts = [(x.start, x.end) for x in p.main_placed()]
    assert starts == [(0, 30), (15, 75)]
    assert p.main_duration == 75 and p.main_overlaps() == {"A": (0, 15), "B": (15, 0)}
    segments = p.visible_segments()
    assert [(s.start, s.end, s.clip.id) for s in segments] == [(0, 15, "A"), (15, 30, "A"), (30, 75, "B")]
    assert segments[1].above == (Layer(p.clips[1], 15, 15),)


def test_dissolve_longer_than_clip_is_shortened():
    p = Project(clips=(Clip("a.mp4", 0, 10, id="A"), Clip("b.mp4", 0, 60, id="B", transition_ms=5000)))
    assert p.main_placed()[1].start == 1  # 앞 클립을 다 먹지 않게 (최대 길이-1)


def test_toggle_transition_at_nearest_cut_and_split_keeps_it_on_left():
    p = Project(clips=(Clip("a.mp4", 0, 30, id="A"), Clip("b.mp4", 0, 60, id="B")))
    q, clip_id = toggle_transition(p, 33, 500)
    assert clip_id == "B" and q.clips[1].transition_ms == 500
    assert toggle_transition(q, 20, 500)[0].clips[1].transition_ms == 0  # 다시 누르면 뺀다
    left, right = split(q, 50).clips[1:]
    assert (left.transition_ms, right.transition_ms) == (500, 0)


def test_locate_inside_dissolve_picks_earlier_clip():
    assert dissolve().locate(20) == (0, 20)
    assert dissolve().locate(30) == (1, 15)


def test_set_clip_props_for_screen():
    p = set_clip_props(pip(), {"X"}, scale=0.3, pos_y=0.8, opacity=0.7)
    x = p.find("X").clip
    assert (x.scale, x.pos_x, x.pos_y, x.opacity) == (0.3, 0.25, 0.8, 0.7)


def test_markers_toggle_follow_range_delete_and_fps():
    p = toggle_marker(toggle_marker(pip(), 50), 5)
    assert p.markers == (5, 50)
    assert toggle_marker(p, 5).markers == (50,)
    assert delete_range(p, 20, 30).markers == (5, 40)
    assert delete_range(p, 0, 10).markers == (40,)  # 구간 안 마커는 사라진다
    assert set_fps(p, 60).markers == (10, 100)
    assert 50 in edit_points(p)
