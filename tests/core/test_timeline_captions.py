"""컷편집 자막: V1 편집을 따라가고(연동), 구간 지우기에 같이 잘리고, 직접 고친 자막은 그대로."""

from ks_studio.core.edits import delete_clip, delete_range, follow_main, set_subtitles
from ks_studio.core.project import Clip, Project
from ks_studio.core.subtitles import Cue, Subtitles


def project_with(cues) -> Project:
    a, b = Clip("a.mp4", 0, 60, id="a"), Clip("b.mp4", 0, 60, id="b")  # 30fps 2초씩: a [0,2)초 · b [2,4)초
    return Project(fps=30, clips=(a, b), subtitles=Subtitles(tuple(cues)))


def test_captions_follow_v1_and_vanish_with_their_clip():
    before = project_with([Cue(0.5, 1.0, "on a"), Cue(2.5, 3.0, "on b")])
    swapped = follow_main(before, Project(fps=30, clips=(before.clips[1], before.clips[0]), subtitles=before.subtitles))
    assert [(c.text, c.start) for c in swapped.subtitles.cues] == [("on b", 0.5), ("on a", 2.5)]  # 순서를 바꾸면 같이
    gone = follow_main(before, delete_clip(before, "a"))
    assert [(c.text, c.start) for c in gone.subtitles.cues] == [("on b", 0.5)]  # a를 지우면 a 위 자막도, b 위 자막은 당겨진다


def test_delete_range_cuts_captions_and_edits_stay():
    before = project_with([Cue(0.5, 1.0, "inside"), Cue(1.2, 2.5, "across"), Cue(3.0, 3.5, "after")])
    cut = follow_main(before, delete_range(before, 15, 45))  # 0.5초~1.5초를 지운다
    assert [(c.text, c.start, c.end) for c in cut.subtitles.cues] == [("across", 0.5, 1.5), ("after", 2.0, 2.5)]
    edited = set_subtitles(before, Subtitles((Cue(0.1, 0.5, "new"),)))
    assert follow_main(before, edited).subtitles == edited.subtitles  # 자막만 고친 편집은 그대로
