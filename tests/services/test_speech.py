"""음성 인식 결과 정리: 빈 글·영상 밖·지어낸 반복을 버린다 (모델 없이)."""

from ks_studio.core.subtitles import Cue
from ks_studio.services.speech import clean


def test_clean_drops_empty_outside_and_repeats():
    segments = [(0.0, 1.0, " Did you buy it?"), (2.0, 3.0, " "), (4.0, 5.0, " Yes"), (5.0, 6.0, " Yes"), (9.9, 12.0, " Thanks"), (16.0, 18.0, " ghost")]
    assert clean(segments, 10.0) == [Cue(0.0, 1.0, "Did you buy it?"), Cue(4.0, 5.0, "Yes"), Cue(9.9, 10.0, "Thanks")]
