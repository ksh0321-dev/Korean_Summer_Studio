"""하이라이트 재료 뽑기: 실제 영상 파일 → 소리 세기·움직임 (얼굴 모델 없이) · 캐시."""

from fractions import Fraction

import av
import numpy as np
import pytest

from ks_studio.core.highlight import AUDIO_STEP, SAMPLE_STEP, Signals, score
from ks_studio.services import highlight
from ks_studio.services.sources import AUDIO_RATE

FPS, SECONDS, SIZE = 30, 10, 160


def make_video(path, fps: int = FPS) -> None:
    """10초: 4~6초에 흰 네모가 움직이고, 6~8초에 소리가 크다 (나머지는 작은 소리)."""
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width = video.height = SIZE
        video.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        for k in range(SECONDS * fps):
            t = k / fps
            image = np.full((SIZE, SIZE, 3), 60, np.uint8)
            if 4 <= t < 6:
                x = int((t - 4) * 60) % (SIZE - 40)
                image[40:80, x : x + 40] = 230
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = k, Fraction(1, fps)
            container.mux(video.encode(frame))
        container.mux(video.encode())
        n = SECONDS * AUDIO_RATE
        t = np.arange(n) / AUDIO_RATE
        volume = np.where((t >= 6) & (t < 8), 0.5, 0.01)
        samples = (volume * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        for i in range(0, n, 1024):
            chunk = av.AudioFrame.from_ndarray(samples[None, i : i + 1024], format="flt", layout="mono")
            chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, i, Fraction(1, AUDIO_RATE)
            container.mux(audio.encode(chunk))
        container.mux(audio.encode())


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    make_video(path)
    return path


def test_loudness_is_db_every_tenth_second():
    t = np.arange(AUDIO_RATE) / AUDIO_RATE
    tone = (0.5 * 32767 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    db = highlight.loudness(np.r_[tone, np.zeros(AUDIO_RATE, np.int16)])
    assert len(db) == 20
    assert np.allclose(db[:10], 20 * np.log10(0.5 / np.sqrt(2)), atol=0.1)  # 사인파 RMS
    assert db[10:].max() <= -99  # 무음은 바닥값


def test_analyze_finds_motion_and_loud_parts(clip):
    signals = highlight.analyze(clip)
    assert abs(signals.duration - SECONDS) < 0.1
    assert signals.face is None
    assert abs(len(signals.motion) - SECONDS / SAMPLE_STEP) <= 1
    assert abs(len(signals.loudness) - SECONDS / AUDIO_STEP) <= 2
    moving = np.nanmean(signals.motion[round(4.2 / SAMPLE_STEP) : round(5.8 / SAMPLE_STEP)])
    still = np.nanmax(signals.motion[round(0.4 / SAMPLE_STEP) : round(3.8 / SAMPLE_STEP)])
    assert moving > 0.01 and still < 0.002
    assert np.median(signals.loudness[62:78]) > np.median(signals.loudness[10:50]) + 25
    scores = score(signals)
    assert scores.mean(6.2, 7.8) > scores.mean(0.5, 3.5) + 20
    assert scores.mean(4.2, 5.8) > scores.mean(0.5, 3.5) + 5


def test_variable_frame_gaps_keep_sample_times(tmp_path):
    """프레임이 1초에 2장뿐이어도 0.2초 칸 수는 영상 길이대로 (시각이 밀리지 않게)."""
    path = tmp_path / "slow.mp4"
    make_video(path, fps=2)
    signals = highlight.analyze(path)
    assert abs(len(signals.motion) - SECONDS / SAMPLE_STEP) <= 3


def test_progress_and_cancel(clip):
    seen = []
    highlight.analyze(clip, on_progress=seen.append)
    assert seen and seen[0] == 0 and all(0 <= p <= 1 for p in seen) and seen == sorted(seen)
    with pytest.raises(highlight.Cancelled):
        highlight.analyze(clip, cancel=lambda: True)


def test_cache_round_trip_and_reanalyze_for_faces(clip, tmp_path):
    cache = tmp_path / "cache"
    first = highlight.load_or_analyze(cache, clip)
    file = highlight.cache_file(cache, clip)
    assert file.exists() and file.name.endswith(".highlight1.npz")
    again = highlight.cached(cache, clip, want_face=False)
    assert again is not None and again.duration == pytest.approx(first.duration)
    assert np.array_equal(again.motion, first.motion, equal_nan=True)
    assert highlight.cached(cache, clip, want_face=True) is None  # 표정 없이 분석한 것 → 얼굴 모델이 생기면 다시


def test_cache_keeps_face_rows(tmp_path):
    file = tmp_path / "x.highlight1.npz"
    face = np.array([[1.0, 0.1, 0.2, 0.9], [np.nan] * 4], np.float32)
    highlight.save(file, Signals(1.0, np.zeros(10, np.float32), np.zeros(5, np.float32), face))
    loaded = highlight.load(file)
    assert np.array_equal(loaded.face, face, equal_nan=True)


def test_broken_cache_is_ignored(clip, tmp_path):
    cache = tmp_path / "cache"
    file = highlight.cache_file(cache, clip)
    file.parent.mkdir(parents=True)
    file.write_bytes(b"not npz")
    assert highlight.cached(cache, clip, want_face=False) is None


def test_cleanup_keeps_highlight_cache_of_used_video(clip, tmp_path):
    from ks_studio.services.cleanup import CURRENT_SUFFIXES

    assert any(highlight.cache_file(tmp_path, clip).name.endswith(s) for s in CURRENT_SUFFIXES)
