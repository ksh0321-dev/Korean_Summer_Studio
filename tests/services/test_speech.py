"""음성 인식 결과 정리: 빈 글·영상 밖·지어낸 반복을 버리고, 단어 시간으로 짧은 자막 줄을 만든다 (모델 없이)."""

from ks_studio.core.subtitles import Word
from ks_studio.services.speech import cues_from_segments, segment_words


def test_segment_words_drops_empty_outside_and_repeats():
    segments = [
        (0.0, 1.0, " Did you buy it?", [(0.0, 0.4, " Did"), (0.4, 0.6, " you"), (0.6, 0.8, " buy"), (0.8, 1.0, " it?")]),
        (2.0, 3.0, " ", []),
        (4.0, 5.0, " Yes", [(4.0, 5.0, " Yes")]),
        (5.0, 6.0, " Yes", [(5.0, 6.0, " Yes")]),  # 음악 구간에서 지어낸 반복
        (9.9, 12.0, " Thanks", []),  # 단어 시간이 없으면 통째로, 영상 끝에서 자른다
        (16.0, 18.0, " ghost", [(16.0, 18.0, " ghost")]),  # 영상 끝 뒤
    ]
    words = segment_words(segments, 10.0)
    assert [w.text for w in words] == [" Did", " you", " buy", " it?", " Yes", " Thanks"]
    assert words[-1] == Word(9.9, 10.0, " Thanks")


def test_cues_from_segments_split_at_sentence_end():
    segments = [(0.0, 3.0, " 내가 만든 건 아니지만 어때? 이쁘지?",
                 [(0.4, 0.8, " 내가"), (0.8, 1.2, " 만든"), (1.2, 1.5, " 건"), (1.5, 2.6, " 아니지만"), (2.6, 3.0, " 어때?"),
                  (3.3, 4.6, " 이쁘지?")])]
    cues = cues_from_segments(segments, 10.0)
    assert [c.text for c in cues] == ["내가 만든 건 아니지만 어때?", "이쁘지?"]
    assert cues[0].start == 0.4 and cues[0].end == 3.3  # 다음 줄 시작까지만 더 둔다
    assert len(cues[0].words) == 5
