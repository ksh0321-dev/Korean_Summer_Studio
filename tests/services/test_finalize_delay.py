from fractions import Fraction

import av
import numpy as np
import pytest

from ks_studio.services.media import AUDIO_DELAY_TAG, AUDIO_RATE, finalize_recording
from ks_studio.services.sources import decode_audio


def make_partial(path, delay_ms: int) -> None:
    """엔진이 쓰는 것과 같은 MKV(H.264 + PCM). 원본 1.000초에 클릭."""
    with av.open(str(path), "w", format="matroska") as out:
        out.metadata[AUDIO_DELAY_TAG] = str(delay_ms)
        video = out.add_stream("libx264", rate=30)
        video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        video.time_base = Fraction(1, 30)
        audio = out.add_stream("pcm_s16le", rate=AUDIO_RATE, layout="mono")
        for i in range(60):
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
            frame.pts = i
            out.mux(video.encode(frame))
        out.mux(video.encode())
        sound = np.zeros(2 * AUDIO_RATE, np.int16)
        sound[AUDIO_RATE : AUDIO_RATE + 48] = 30000
        chunk = av.AudioFrame.from_ndarray(sound.reshape(1, -1), format="s16", layout="mono")
        chunk.sample_rate, chunk.pts = AUDIO_RATE, 0
        out.mux(audio.encode(chunk))
        out.mux(audio.encode())


@pytest.mark.parametrize("delay_ms", [0, 105, -40])
def test_finalize_shifts_audio_by_calibration(tmp_path, delay_ms):
    partial, output = tmp_path / "rec.mkv", tmp_path / "rec.mp4"
    make_partial(partial, delay_ms)
    finalize_recording(partial, output)
    audio = decode_audio(output)
    click = np.flatnonzero(np.abs(audio) > 10000)[0] / AUDIO_RATE
    assert click == pytest.approx(1.0 + delay_ms / 1000, abs=0.002)
    assert not partial.exists()
