"""하이라이트 점수: 소리·표정·움직임 재료 → 0.5초마다 0~100점."""

import math

import numpy as np

from ks_studio.core.highlight import AUDIO_STEP, BIN, SAMPLE_STEP, Signals, expression, score, stand_out
from tests.core.test_beauty import fake_face


def quiet_with_loud(duration: float, loud: tuple[float, float], base: float = -40.0, peak: float = -10.0) -> np.ndarray:
    """조용히 말하다가 loud 구간(초)만 크게 — 0.1초마다 dB. 작은 흔들림을 섞는다 (실제 말소리처럼)."""
    rng = np.random.default_rng(0)
    db = base + rng.normal(0, 1.5, round(duration / AUDIO_STEP))
    a, b = (round(t / AUDIO_STEP) for t in loud)
    db[a:b] = peak + rng.normal(0, 1.5, b - a)
    return db.astype(np.float32)


def test_stand_out_is_zero_for_ordinary_and_one_for_far_above():
    values = np.r_[np.zeros(90), np.full(10, 5.0)]
    out = stand_out(values, floor=1.0)
    assert out[:90].max() == 0  # 보통(중앙값)은 0
    assert out[95] > 0.9


def test_stand_out_floor_keeps_flat_video_low():
    """거의 고른 영상: 작은 흔들림이 '두드러짐'으로 부풀지 않는다."""
    values = np.random.default_rng(1).normal(0, 0.2, 200)
    assert stand_out(values, floor=6.0).max() < 0.1


def test_stand_out_ignores_nan():
    out = stand_out(np.array([np.nan, 0, 0, 0, 10.0]), floor=1.0)
    assert out[0] == 0 and out[-1] == 1


def test_loud_part_scores_high_and_says_why():
    signals = Signals(duration=30.0, loudness=quiet_with_loud(30.0, (12.0, 18.0)))
    scores = score(signals)
    assert len(scores.values) == 60 and scores.groups == ("sound",)
    assert scores.mean(12.5, 17.5) > 50
    assert scores.mean(0, 8) < 10 and scores.mean(22, 30) < 10
    assert 0 <= scores.values.min() and scores.values.max() <= 100
    assert "목소리 커짐" in scores.reasons(12, 18)
    assert scores.reasons(0, 8) == []


def test_sudden_loud_moment_counts_as_jump():
    db = quiet_with_loud(20.0, (10.0, 10.5))  # 0.5초 감탄
    scores = score(Signals(20.0, db))
    assert scores.parts["jump"][20] > 0.9  # 10.0~10.5초 칸
    assert scores.at(10.2) > scores.at(4.0) + 10


def test_audio_offset_moves_sound_onto_video_time():
    """소리 맞춤이 +2초면 원본 오디오 10초의 소리는 영상 12초에 들린다."""
    db = quiet_with_loud(30.0, (10.0, 14.0))
    plain, shifted = score(Signals(30.0, db)), score(Signals(30.0, db), audio_offset=2.0)
    assert np.argmax(plain.parts["level"]) // 2 in range(10, 14)
    assert np.argmax(shifted.parts["level"]) // 2 in range(12, 16)


def test_silence_floor_ignores_breathing_differences():
    """아주 조용한 부분끼리의 차이(-90 dB vs -70 dB)는 점수가 되지 않는다."""
    db = np.r_[np.full(100, -90.0), np.full(100, -70.0)].astype(np.float32)
    assert score(Signals(20.0, db)).values.max() < 1


def test_motion_only_video():
    motion = np.full(round(20 / SAMPLE_STEP), 0.005, np.float32)
    motion[50:60] = 0.08  # 10~12초 크게 움직임
    motion[0] = np.nan
    scores = score(Signals(20.0, motion=motion))
    assert scores.groups == ("motion",)
    assert scores.mean(10, 12) > 40 and scores.mean(15, 20) < 5
    assert scores.reasons(10, 12) == ["움직임 큼"]


def test_nothing_to_score_gives_zeros():
    scores = score(Signals(5.0))
    assert len(scores.values) == 10 and scores.values.max() == 0 and scores.groups == ()


def smiling(lm: np.ndarray, amount: float) -> np.ndarray:
    """입꼬리를 바깥·위로 (amount = 눈 사이 거리 400px 대비)."""
    lm = lm.copy()
    lm[52] += (-amount * 200, -amount * 200)
    lm[61] += (amount * 200, -amount * 200)
    return lm


def test_expression_reads_smile_mouth_and_brows():
    base = fake_face()
    smile0, mouth0, brow0 = expression(base)
    assert expression(smiling(base, 0.1))[0] > smile0 + 0.1
    opened = base.copy()
    opened[[57, 60, 54]] += (0, 60)  # 아랫입술 안쪽을 내린다
    assert expression(opened)[1] > mouth0 * 2
    raised = base.copy()
    raised[[48, 49, 51, 50, 102, 103, 104, 105]] -= (0, 40)
    assert math.isclose(expression(raised)[2], brow0 + 40 / 400, rel_tol=1e-6)


def test_expression_does_not_depend_on_face_size_or_tilt():
    base = expression(fake_face())
    small = expression(fake_face() * 0.25)
    tilted = expression(fake_face(angle=0.3))
    assert np.allclose(base, small) and np.allclose(base, tilted, atol=1e-6)


def face_rows(n: int, smile_at: slice, presence: float = 1.0) -> np.ndarray:
    smile, mouth, brow = expression(fake_face())
    rng = np.random.default_rng(2)
    rows = np.tile([smile, mouth, brow, presence], (n, 1)).astype(np.float32)
    rows[:, :3] += rng.normal(0, 0.003, (n, 3))
    rows[smile_at, 0] += 0.2
    return rows


def test_smiling_face_scores_and_side_faces_count_less():
    n = round(20 / SAMPLE_STEP)
    front = score(Signals(20.0, face=face_rows(n, slice(50, 75))))  # 10~15초 웃음
    assert front.groups == ("face",)
    assert front.mean(10.5, 14.5) > 30 and front.mean(0, 8) < 5
    assert front.reasons(10, 15)[0] == "웃는 얼굴"
    side = score(Signals(20.0, face=face_rows(n, slice(50, 75), presence=0.3)))
    assert side.mean(10.5, 14.5) < front.mean(10.5, 14.5) * 0.5


def test_no_face_found_drops_face_group():
    n = round(10 / SAMPLE_STEP)
    face = np.full((n, 4), np.nan, np.float32)
    scores = score(Signals(10.0, loudness=quiet_with_loud(10.0, (4.0, 6.0)), face=face))
    assert scores.groups == ("sound",)  # 얼굴이 한 번도 없으면 표정은 빼고 나눈다 (점수가 반토막 나지 않게)
    assert scores.mean(4.5, 5.5) > 50


def test_weights_mix_groups():
    n = round(20 / SAMPLE_STEP)
    motion = np.full(n, 0.005, np.float32)
    motion[0:25] = 0.08  # 0~5초 움직임
    signals = Signals(20.0, loudness=quiet_with_loud(20.0, (12.0, 17.0)), motion=motion)
    sound_heavy = score(signals, weights={"sound": 1.0, "motion": 0.1})
    motion_heavy = score(signals, weights={"sound": 0.1, "motion": 1.0})
    assert sound_heavy.mean(13, 16) > sound_heavy.mean(1, 4)
    assert motion_heavy.mean(1, 4) > motion_heavy.mean(13, 16)
    assert score(signals, weights={"motion": 0}).groups == ("sound",)


def test_bins_cover_the_whole_video():
    scores = score(Signals(10.2, loudness=np.zeros(102, np.float32)))
    assert len(scores.values) == math.ceil(10.2 / BIN)
    assert scores.at(99) == scores.values[-1] and scores.at(-1) == scores.values[0]
    assert scores.mean(5, 5) == scores.values[10]  # 길이 0이면 그 칸 하나
