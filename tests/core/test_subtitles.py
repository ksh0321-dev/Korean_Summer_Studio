"""자막 줄 편집·저장·SRT."""

from ks_studio.core.subtitles import (
    MIN_CUE,
    CaptionStyle,
    Cue,
    SubtitleState,
    Subtitles,
    add_cue,
    delete_cue,
    load_subtitles,
    save_subtitles,
    to_srt,
    update_cue,
)


def test_add_cue_stops_before_next_and_keeps_order():
    subs = Subtitles((Cue(0, 1, "a"), Cue(5, 6, "c")))
    subs, index = add_cue(subs, 4.0, "b")
    assert index == 1 and subs.cues[1] == Cue(4.0, 5.0, "b")  # 다음 줄 앞에서 끝
    subs, index = add_cue(subs, 4.9)
    assert subs.cues[index].end - subs.cues[index].start >= MIN_CUE - 1e-9  # 너무 짧게는 안 만든다


def test_update_reorders_and_keeps_minimum_length():
    subs = Subtitles((Cue(0, 1, "a"), Cue(2, 3, "b")))
    subs, index = update_cue(subs, 1, start=0.5)  # b가 a 사이로 → 순서가 바뀐다
    assert index == 1 and subs.cues[1].text == "b"
    subs, index = update_cue(subs, 1, start=-1.0)
    assert subs.cues[index] == Cue(0.0, 3, "b")  # 0초 밑으로는 안 간다
    subs, index = update_cue(subs, index, end=0.0, text="B")
    assert subs.cues[index] == Cue(0.0, MIN_CUE, "B")
    assert delete_cue(subs, index).cues == (Cue(0, 1, "a"),)


def test_cue_at_time():
    subs = Subtitles((Cue(0, 2, "a"), Cue(1.5, 3, "b")))
    assert subs.at(0.5) == 0 and subs.at(1.7) == 1 and subs.at(3.0) is None


def test_srt_and_roundtrip(tmp_path):
    subs = Subtitles((Cue(0.5, 1.25, "Did you buy it?"), Cue(2, 3, "  "), Cue(3661.0, 3662.5, "Yes")), CaptionStyle(size=0.07), "transcribe")
    assert to_srt(subs) == "1\n00:00:00,500 --> 00:00:01,250\nDid you buy it?\n\n2\n01:01:01,000 --> 01:01:02,500\nYes\n"
    path = tmp_path / "subtitles" / "subtitles.json"
    state = SubtitleState({"recordings/a.mp4": subs}, ("D:/b.mp4",))
    save_subtitles(path, state)
    assert load_subtitles(path) == state
    assert load_subtitles(tmp_path / "none.json") == SubtitleState()
