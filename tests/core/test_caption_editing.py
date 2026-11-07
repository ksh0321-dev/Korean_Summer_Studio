"""자막 다듬기: 옮기기 · 나누기 · 합치기 · 찾아 바꾸기 · 용어집 · 모양 프리셋 (순수 파이썬)."""

from ks_studio.core.glossary import Term, load_glossary, save_glossary, speech_hint, translation_hint
from ks_studio.core.subtitles import (
    CaptionStyle,
    Cue,
    Subtitles,
    Word,
    load_style,
    merge_cues,
    move_cue,
    replace_text,
    save_style,
    split_cue,
)


def spoken(text: str, start: float, step: float = 0.5) -> Cue:
    words = tuple(Word(start + i * step, start + (i + 1) * step, " " + w) for i, w in enumerate(text.split()))
    return Cue(start, words[-1].end, text, "", words)


def test_move_cue_shifts_words_and_stops_at_zero():
    subs = Subtitles((spoken("가 나", 1.0), spoken("다 라", 3.0)))
    moved, index = move_cue(subs, 1, -2.5)  # 앞 줄보다 앞으로 → 순서가 바뀐다
    assert index == 0 and moved.cues[0].start == 0.5 and moved.cues[0].words[0].start == 0.5
    clamped, _ = move_cue(subs, 0, -5)
    assert clamped.cues[0].start == 0.0 and clamped.cues[0].end == 1.0


def test_split_by_words_and_translation_by_ratio():
    cue = Cue(0.0, 2.0, "오늘은 코스트코 가요", "Today I'm going to Costco",
              (Word(0.0, 0.6, " 오늘은"), Word(0.6, 1.4, " 코스트코"), Word(1.4, 2.0, " 가요")))
    subs, second = split_cue(Subtitles((cue,)), 0, 1.0)  # 헤드가 '코스트코' 한가운데 → 그 단어는 뒤 줄로
    a, b = subs.cues
    assert second == 1 and (a.text, b.text) == ("오늘은", "코스트코 가요")
    assert a.end == b.start == 1.0 and len(a.words) == 1 and len(b.words) == 2
    assert a.translation and b.translation and f"{a.translation} {b.translation}" == cue.translation
    assert split_cue(subs, 0, 0.1) is None  # 양 끝에서 MIN_CUE 안쪽은 못 나눈다


def test_split_without_words_uses_time_ratio():
    subs, _ = split_cue(Subtitles((Cue(0.0, 4.0, "하나 둘 셋 넷"),)), 0, 2.0)
    assert [c.text for c in subs.cues] == ["하나 둘", "셋 넷"] and subs.cues[0].words == ()


def test_merge_joins_text_translation_and_words():
    subs = Subtitles((Cue(0, 1, "이쁘지?", "Pretty?", (Word(0, 1, " 이쁘지?"),)), Cue(1.2, 2, "예뻐.", "Yes.", (Word(1.2, 2, " 예뻐."),))))
    merged, index = merge_cues(subs, 0)
    cue = merged.cues[0]
    assert index == 0 and len(merged.cues) == 1
    assert (cue.start, cue.end, cue.text, cue.translation, len(cue.words)) == (0, 2, "이쁘지? 예뻐.", "Pretty? Yes.", 2)
    assert merge_cues(merged, 0) is None  # 다음 줄이 없다


def test_replace_text_counts_and_keeps_word_times_when_possible():
    subs = Subtitles((Cue(0, 1, "곳감 정과", "song emotion", (Word(0, 0.5, " 곳감"), Word(0.5, 1, " 정과"))),
                      Cue(1, 2, "곳감 단지", "piece of cake")))
    fixed, count = replace_text(subs, "곳감", "곶감")
    assert count == 2 and fixed.cues[0].text == "곶감 정과" and fixed.cues[0].words[0].text == " 곶감"  # 단어 시간 그대로
    spanning, _ = replace_text(subs, "곳감 정과", "곶감정과")
    assert spanning.cues[0].text == "곶감정과" and spanning.cues[0].words == ()  # 두 단어에 걸치면 버린다
    english, count = replace_text(subs, "SONG EMOTION", "candied persimmon", where="translation")
    assert count == 1 and english.cues[0].translation == "candied persimmon" and english.cues[0].text == "곳감 정과"
    assert replace_text(subs, "SONG", "x", case=True)[1] == 0


def test_glossary_roundtrip_and_hints(tmp_path):
    path = tmp_path / "glossary.json"
    assert load_glossary(path) == ()
    save_glossary(path, [Term("곶감", "dried persimmon"), Term(" 송편 ", "songpyeon"), Term("", "ghost")])
    terms = load_glossary(path)
    assert terms == (Term("곶감", "dried persimmon"), Term("송편", "songpyeon"))
    assert speech_hint(terms) == "곶감 송편" and speech_hint(()) is None
    assert translation_hint(terms, "이거는 곶감 정과?") == "dried persimmon"
    assert translation_hint(terms, "떡") is None
    path.write_text("{broken", encoding="utf-8")
    assert load_glossary(path) == ()  # 깨진 파일은 빈 용어집으로


def test_style_preset_roundtrip_and_tolerant_loading(tmp_path):
    style = CaptionStyle(size=0.07, box=True, box_color="#203040", font="Malgun Gothic", outline_width=0.0)
    save_style(tmp_path, "쇼츠", style)
    assert load_style(tmp_path, "쇼츠") == style
    assert CaptionStyle.from_dict({"size": 1, "box": "yes", "unknown": 3, "color": 5}) == CaptionStyle(size=1.0)
