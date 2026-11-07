"""자막 줄 편집·저장·SRT."""

from ks_studio.core.subtitles import (
    MIN_CUE,
    CaptionStyle,
    Cue,
    Subtitles,
    add_cue,
    delete_cue,
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


def test_srt_and_project_roundtrip(tmp_path):
    """자막은 컷편집 project.json에 (타임라인 시각) 저장된다."""
    from ks_studio.core.project import Project

    subs = Subtitles((Cue(0.5, 1.25, "Did you buy it?"), Cue(2, 3, "  "), Cue(3661.0, 3662.5, "Yes")), CaptionStyle(size=0.07), "transcribe")
    assert to_srt(subs) == "1\n00:00:00,500 --> 00:00:01,250\nDid you buy it?\n\n2\n01:01:01,000 --> 01:01:02,500\nYes\n"
    path = tmp_path / "project.json"
    Project(subtitles=subs).save(path)
    assert Project.load(path).subtitles == subs
    path.write_text('{"version": 8, "fps": 30, "media": [], "clips": []}', encoding="utf-8")
    assert Project.load(path).subtitles == Subtitles()  # 자막이 없던 예전 파일


# ── 두 줄 자막 · 단어 시간 · 자동 줄 나누기 ─────────────────────────


def words_of(text: str, start: float = 0.0, step: float = 0.3, gaps: dict | None = None) -> list:
    """띄어쓰기 단위 단어들, step초씩 (gaps = {단어 번호: 그 앞에 더 쉬는 초})."""
    from ks_studio.core.subtitles import Word

    words, t = [], start
    for i, token in enumerate(text.split(" ")):
        t += (gaps or {}).get(i, 0.0)
        words.append(Word(round(t, 3), round(t + step, 3), " " + token))
        t += step
    return words


def test_split_long_speech_into_short_cues():
    from ks_studio.core.subtitles import split_words, text_width

    words = words_of("오늘은 코스트코에 가서 장을 보고 집에 와서 맛있는 저녁을 만들어 먹을 거예요 그리고 영화도 볼 거예요")
    cues = split_words(words, line_chars=8, max_lines=2)
    assert len(cues) >= 3
    assert all(text_width(c.text) <= 16 for c in cues)  # 한 줄 8글자 × 2줄
    assert " ".join(c.text for c in cues).split() == [w.text.strip() for w in words]  # 빠진 단어 없이 차례대로
    assert all(a.end <= b.start for a, b in zip(cues, cues[1:]))  # 겹치지 않는다


def test_split_at_pause_and_merge_tiny_pieces():
    from ks_studio.core.subtitles import split_words

    # '예뻐'(0.3초)는 혼자면 너무 빨리 지나간다 → 바로 뒤 말과 합친다. 긴 쉼(1초) 뒤는 새 줄
    words = words_of("예뻐 잘 만든 것 같아 이건 송편", gaps={1: 0.2, 5: 1.0})
    cues = split_words(words, line_chars=16, max_lines=2)
    assert [c.text for c in cues] == ["예뻐 잘 만든 것 같아", "이건 송편"]


def test_bilingual_cue_roundtrip_and_text_edit_drops_words(tmp_path):
    from ks_studio.core.project import Project
    from ks_studio.core.subtitles import update_cue

    cue = Cue(0.4, 1.4, "이쁘지?", "Isn't it pretty?", tuple(words_of("이쁘지?", 0.4)))
    subs = Subtitles((cue,), line_chars=12, max_lines=1)
    assert to_srt(subs) == "1\n00:00:00,400 --> 00:00:01,400\n이쁘지?\nIsn't it pretty?\n"
    path = tmp_path / "project.json"
    Project(subtitles=subs).save(path)
    loaded = Project.load(path).subtitles
    assert loaded == subs and loaded.cues[0].words == cue.words and loaded.line_chars == 12
    edited, _ = update_cue(subs, 0, translation="Pretty, right?")
    assert edited.cues[0].words == cue.words  # 번역만 고치면 단어 시간은 그대로
    edited, _ = update_cue(subs, 0, text="예쁘지?")
    assert edited.cues[0].words == () and edited.cues[0].translation == "Isn't it pretty?"


def test_cut_keeps_translation_and_moves_words():
    from ks_studio.core.subtitles import map_times

    cue = Cue(2.0, 4.0, "만든 거", "made it", tuple(words_of("만든 거", 2.0, 1.0)))
    moved = map_times(cue, lambda t: t - 1.0)
    assert moved.translation == "made it" and [w.start for w in moved.words] == [1.0, 2.0]
    assert map_times(cue, lambda t: 1.0) is None  # 다 잘려 나간 줄


def test_long_cue_split_at_biggest_gap():
    from ks_studio.core.subtitles import Word, split_words

    # 문장부호 없이 천천히 말한 9초 — 글자 수는 한도 안이지만 너무 오래 보인다
    words = [Word(0.0, 2.0, " 이거는"), Word(2.1, 4.0, " 곶감"), Word(4.3, 6.0, " 정과"), Word(6.05, 9.0, " 아니야")]
    cues = split_words(words)
    assert [c.text for c in cues] == ["이거는 곶감", "정과 아니야"]
    assert all(c.words[-1].end - c.start <= 5.0 for c in cues)  # 말하는 시간 (뒤에 HOLD만큼 더 보인다)
